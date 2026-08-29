#pragma once

// PSTDynamic (paper Section 5, Algorithms 4-6, pages 16-19): the actual matching engine
// built on PS-Tree. One PS-Tree per dimension; each subscription's highest-selectivity
// predicate (its "access predicate") is inserted into that dimension's tree; subscriptions
// sharing a leaf are grouped by a Bloom-filter "dimension signature" over every dimension
// they have a predicate on (confirmed via Fig. 3's own worked example - the signature
// covers the access predicate's own dimension too, not just the "other" ones), pruning most
// candidates at the group level before the final per-subscription Match() (predicate.hpp).
//
// Deliberate design decisions the paper leaves unspecified (see also dim_sig.hpp's own
// comment for the Bloom filter itself):
//   - SelectAccPred uses the paper's own static selectivity ranking (Section 2.3:
//     "{=} > {in} > {in} > {<,<=,>,>=} > {!in} > {!=}" - i.e. kEq > kElemOf > kBetween >
//     {kLt,kLe,kGt,kGe} > kNotElemOf > kNe) as the primary key. Verified against Fig. 3's own
//     6 subscriptions: this ranking alone picks exactly the access predicates the paper
//     states it does for every one of them (see test_pst_dynamic.cpp) - none of Fig. 3's
//     subscriptions have two same-tier predicates, so the tie-break below is never consulted
//     there.
//   - Within the same tier, kElemOf/kNotElemOf break ties by real-world SELECTIVITY, not raw
//     value-list width: `vals.size() / observedValues.size()` for the predicate's own
//     dimension (see DimensionIndex::observedValues), smaller wins. A real, measured bug in
//     the paper's own literal "narrower list wins" heuristic: it conflates "few literals" with
//     "selective," which only holds when every candidate dimension has a similarly-sized real
//     domain. A subscription like `exchange in (1-4 values)` (a 19-value real domain) vs.
//     `symbol in (8-128 values)` (a ~12,000-value real domain) in the SAME subscription always
//     picked `exchange` under the old raw-width rule - indexing on a 19-value domain gives
//     almost no partitioning once subscription counts reach into the thousands, since nearly
//     every subscription becomes a per-event candidate regardless of which exchange value the
//     event carries. Normalizing by each dimension's own observed cardinality picks `symbol`
//     instead once enough data has been seen, since it covers a far smaller FRACTION of its
//     own (much larger) domain - measured on a real downstream benchmark
//     (nats_sidecar's K=4000-32000 exchange/symbol set-membership benchmark) at 70-135x
//     higher search throughput. `kBetween`'s own numeric-range width tie-break is untouched -
//     no comparable bug found there, not touched.
//   - Because of the above, SelectAccPred is NO LONGER a pure, time-invariant function of the
//     subscription alone - `observedValues` grows as more subscriptions are inserted, so the
//     SAME subscription could select a different access predicate depending on insertion
//     order. This is fine for correctness (which predicate ends up "the" access predicate is
//     an internal indexing choice, not an observable one) but breaks the OLD invariant this
//     comment used to state ("DeleteSubscription must reselect the identical access predicate
//     InsertSubscription chose" via a second, later call to the same pure function) - fixed by
//     no longer recomputing at delete time at all: the selected index is computed once, at
//     insert time, and stored alongside the subscription (see `subscriptions_`'s value type)
//     for DeleteSubscription to read back directly.
//   - grow/shrinkThreshold: the paper's own Algorithm 4/6 pseudocode reuses ONE
//     `thresholds[DimSigLen]` array for BOTH the grow check (Algorithm 4, line 8:
//     `subNum >= thresholds[DimSigLen]`) and the shrink check (Algorithm 6, line 8:
//     `subNum <= thresholds[DimSigLen]`, evaluated at whatever DimSigLen currently IS,
//     which just became DimSigLen+1 if growth JUST happened) - reusing the exact same array
//     this way risks immediate thrashing (grow, then the very next delete immediately
//     shrinks back) unless the thresholds happen to be spaced with real hysteresis, which
//     the paper never specifies. This implementation uses two deliberately-separated
//     functions instead (grow at capacity(len), shrink only once BELOW capacity(len)/4),
//     giving genuine hysteresis by construction.
//   - kNe/kNotElemOf, if selected as the (least-bad available) access predicate, have no
//     representable contiguous PS-Tree range at all (this is Section 2.3's own stated
//     reason they rank lowest) - handled by inserting `>= <the dimension's minimum
//     representable value>`, i.e. "matches every leaf", correct (never a false negative)
//     but forgoes any pruning for that subscription - exactly the outcome the paper's own
//     selectivity reasoning anticipates for these operators, just made concrete instead of
//     left unhandled.
//   - kElemOf, if selected, has no single contiguous range either - inserted as one kEq per
//     literal value (a "disjunction of equalities"), all pointing at the same subscription;
//     DeleteSubscription mirrors this exactly (same per-value decomposition, so every
//     insert-side leaf is correctly found and decremented).
//
// list_valued attributes (nats_sidecar's string_list/integer_list) are NOT supported here -
// the paper's own model has no analog (an event attribute is always a single value, not a
// list) - flagged in the repo README as follow-up work, not assumed solved.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pstree/dim_sig.hpp"
#include "pstree/order_key.hpp"
#include "pstree/predicate.hpp"
#include "pstree/ps_tree.hpp"

namespace pstree {

enum class ValueType { kBoolean, kInteger, kFloat, kString };

struct AttrSchema {
    std::string name;
    ValueType type;
    std::size_t stringMaxLen = 64; // only meaningful when type == kString
};

namespace detail {

inline KeyShape shapeFor(const AttrSchema& schema) {
    switch (schema.type) {
        case ValueType::kBoolean: return BoolCodec::shape();
        case ValueType::kInteger: return Int64Codec::shape();
        case ValueType::kFloat: return DoubleCodec::shape();
        case ValueType::kString: return StringCodec(schema.stringMaxLen).shape();
    }
    throw std::logic_error("pstree: unreachable ValueType in shapeFor");
}

inline ElementKey encodeValue(const Value& v, const AttrSchema& schema) {
    switch (schema.type) {
        case ValueType::kBoolean: return BoolCodec::encode(std::get<bool>(v));
        case ValueType::kInteger: return Int64Codec::encode(std::get<std::int64_t>(v));
        case ValueType::kFloat: return DoubleCodec::encode(std::get<double>(v));
        case ValueType::kString: return StringCodec(schema.stringMaxLen).encode(std::get<std::string>(v));
    }
    throw std::logic_error("pstree: unreachable ValueType in encodeValue");
}

// Used for the kNe/kNotElemOf "matches every leaf" fallback (see file-level comment).
inline ElementKey minKeyFor(const AttrSchema& schema) {
    switch (schema.type) {
        case ValueType::kBoolean: return BoolCodec::encode(false);
        case ValueType::kInteger: return Int64Codec::encode(std::numeric_limits<std::int64_t>::min());
        case ValueType::kFloat: return DoubleCodec::encode(-std::numeric_limits<double>::infinity());
        case ValueType::kString: return StringCodec(schema.stringMaxLen).encode("");
    }
    throw std::logic_error("pstree: unreachable ValueType in minKeyFor");
}

inline int opRank(CmpOp op) {
    switch (op) {
        case CmpOp::kEq: return 0;
        case CmpOp::kElemOf: return 1;
        case CmpOp::kBetween: return 2;
        case CmpOp::kLt:
        case CmpOp::kLe:
        case CmpOp::kGt:
        case CmpOp::kGe: return 3;
        case CmpOp::kNotElemOf: return 4;
        case CmpOp::kNe:
        case CmpOp::kIsNotNull: return 5; // same "matches everything" fallback tier as kNe
        case CmpOp::kIsNull: return 6;    // worse than kNe: cannot be indexed AT ALL (see applyToTree)
    }
    return 6;
}

// Smaller = more selective. Only meaningful within the same rank tier (Section 2.3: "When
// the operators have the same values, we consider predicates with wider value sets to have
// lower selectivity"). Operators with no natural width always return 0 (a tie, falling
// through to subscription order).
//
// kElemOf/kNotElemOf's own case is normalized by `domainCardinality` - the number of DISTINCT
// values ever observed for this predicate's dimension, across every subscription inserted so
// far (see DimensionIndex::observedValues, fed by PSTDynamic::insertSubscription; the caller,
// PSTDynamic::selectAccPredIndex, looks this up and passes it in - kept out of this function
// so it stays a plain, dependency-free comparison helper). A real, measured bug in the
// paper's own literal "narrower list wins" rule: raw list width alone conflates "few
// literals" with "selective," which only holds when every candidate dimension has a
// similarly-sized real domain - see pst_dynamic.hpp's own file-level comment for the exact
// scenario (a low-cardinality dimension's narrow list always "winning" over a
// high-cardinality dimension's wider one, even though indexing on the low-cardinality one
// gives almost no partitioning at scale) and the measured fix (70-135x higher search
// throughput on a real downstream benchmark once corrected). `domainCardinality` is ignored
// for every other operator, including kBetween's own separate numeric-range width tie-break.
inline double widthTieBreak(const SubPredicate& p, std::size_t domainCardinality) {
    switch (p.op) {
        case CmpOp::kBetween:
            if (p.vals.size() == 2 && std::holds_alternative<std::int64_t>(p.vals[0]) &&
                std::holds_alternative<std::int64_t>(p.vals[1])) {
                return static_cast<double>(std::get<std::int64_t>(p.vals[1]) - std::get<std::int64_t>(p.vals[0]));
            }
            if (p.vals.size() == 2 && std::holds_alternative<double>(p.vals[0]) &&
                std::holds_alternative<double>(p.vals[1])) {
                return std::get<double>(p.vals[1]) - std::get<double>(p.vals[0]);
            }
            return 0.0;
        case CmpOp::kElemOf:
        case CmpOp::kNotElemOf:
            return static_cast<double>(p.vals.size()) /
                   static_cast<double>(std::max<std::size_t>(domainCardinality, 1));
        default:
            return 0.0;
    }
}

constexpr std::size_t kInitialDimSigLen = 4;

inline std::size_t growThreshold(std::size_t dimSigLen) {
    return std::size_t{4} << (dimSigLen - kInitialDimSigLen);
}
inline std::size_t shrinkThreshold(std::size_t dimSigLen) {
    return growThreshold(dimSigLen) / 4;
}

// Dimensions to hash into a subscription's dimension signature - every predicate's
// dimension EXCEPT kIsNull's. The signature's whole pruning contract (Algorithm 5, line 8:
// skip a group unless its signature's dimensions are all present in the event) assumes
// "this subscription needs dimension D" means "D must be PRESENT in the event to possibly
// match" - true for every real operator, but backwards for kIsNull, which is satisfied
// exactly when its dimension is ABSENT. Including a kIsNull dimension in the signature
// would make the pruning incorrectly reject the exact events kIsNull is meant to match
// (found by test_pst_dynamic.cpp/test_pst_dynamic_stress.cpp, not anticipated when kIsNull
// was first added). Omitting it only makes the signature LESS specific (more candidates
// pass the group-level prune), which is always safe - the final matchSubscription() call
// still re-verifies kIsNull correctly regardless. kIsNotNull is NOT excluded: it genuinely
// does require its dimension present, matching the signature's normal assumption.
inline std::vector<std::string> dimSigDimensions(const Subscription& sub) {
    std::vector<std::string> dims;
    dims.reserve(sub.predicates.size());
    for (auto& p : sub.predicates) {
        if (p.op != CmpOp::kIsNull) dims.push_back(p.attr);
    }
    return dims;
}

} // namespace detail

// selectAccPredIndex (Section 2.3's static heuristic, plus the observed-cardinality
// correction described in the file-level comment) lives as a private member function of
// PSTDynamic, not a free function here - it needs `dimensions_` to look up each candidate
// predicate's observed cardinality, which a free function in this position can't reach
// (DimensionIndex is a private nested type). See PSTDynamic::selectAccPredIndex below.

class PSTDynamic {
public:
    explicit PSTDynamic(std::vector<AttrSchema> schema) {
        for (auto& s : schema) {
            std::string name = s.name;
            dimensions_.try_emplace(name, std::move(s));
        }
    }

    // Section 2.3's static heuristic (operator-tier ranking), plus an observed-cardinality
    // correction within the kElemOf/kNotElemOf tier - see the file-level comment for the full
    // rationale and the measured real-world win. Public (not just an internal implementation
    // detail of InsertSubscription): side-effect-free given a subscription, and useful on its
    // own for a caller wanting to know which predicate would be chosen (e.g. testing - see
    // test_pst_dynamic.cpp's own Fig. 3 reproduction - or diagnostics/tuning).
    //
    // No longer a pure, time-invariant function of the subscription alone: `observedValues`
    // (below) changes as more subscriptions are inserted, so calling this twice for the same
    // subscription at two different points in time could return different answers. Called
    // EXACTLY ONCE per subscription by InsertSubscription itself - the result is stored in
    // StoredSubscription::accIdx and read back directly by DeleteSubscription, never
    // recomputed - so this method being callable repeatedly (e.g. from a test) never risks
    // insert/delete disagreeing about which predicate a given subscription actually used.
    std::size_t selectAccPredIndex(const Subscription& sub) const {
        if (sub.predicates.empty()) {
            throw std::invalid_argument("pstree: subscription " + std::to_string(sub.id) + " has no predicates");
        }
        auto cardinalityFor = [&](const SubPredicate& p) -> std::size_t {
            auto it = dimensions_.find(p.attr);
            return it == dimensions_.end() ? 0 : it->second.observedValues.size();
        };
        std::size_t best = 0;
        for (std::size_t i = 1; i < sub.predicates.size(); ++i) {
            int rankBest = detail::opRank(sub.predicates[best].op);
            int rankI = detail::opRank(sub.predicates[i].op);
            bool better = rankI < rankBest ||
                (rankI == rankBest &&
                 detail::widthTieBreak(sub.predicates[i], cardinalityFor(sub.predicates[i])) <
                 detail::widthTieBreak(sub.predicates[best], cardinalityFor(sub.predicates[best])));
            if (better) best = i;
        }
        // kIsNull ranks worst specifically so it's only ever chosen when nothing else is
        // available (see opRank) - and when that happens, the subscription genuinely can't be
        // indexed at all: "X is null" only matches events where X is absent, but MatchEvent
        // never consults X's own dimension tree for such events in the first place (there's no
        // pair to route through). Caught here, at selection time, with a clear message - not
        // silently left to produce a subscription that can never match anything.
        if (sub.predicates[best].op == CmpOp::kIsNull) {
            throw std::invalid_argument(
                "pstree: subscription " + std::to_string(sub.id) +
                " cannot be indexed - its best available access predicate is 'is null' on '" +
                sub.predicates[best].attr +
                "', which has no representable PS-Tree range (add at least one other, "
                "indexable predicate to this subscription)");
        }
        return best;
    }

    // Algorithm 4, InsertSubscription.
    void insertSubscription(const Subscription& sub) {
        if (subscriptions_.count(sub.id) != 0) {
            throw std::invalid_argument("pstree: subscription id " + std::to_string(sub.id) + " already exists");
        }

        // Feed every predicate's own literal values into its dimension's observed-cardinality
        // tracker BEFORE selecting this subscription's own access predicate (see
        // DimensionIndex::observedValues and selectAccPredIndex's own comment) - including
        // predicates that DON'T end up chosen as the access predicate, so a wide-domain
        // dimension's true cardinality becomes visible even from subscriptions that (correctly)
        // keep preferring some other, still-more-selective predicate. Without this, a dimension
        // nothing has ever been indexed ON would never accumulate the evidence needed to
        // eventually win a tie-break against a narrower-but-lower-cardinality one.
        for (const auto& pred : sub.predicates) {
            auto predDimIt = dimensions_.find(pred.attr);
            if (predDimIt == dimensions_.end()) continue; // unknown dimension - rejected below if selected
            for (const auto& v : pred.vals) {
                predDimIt->second.observedValues.insert(detail::encodeValue(v, predDimIt->second.schema));
            }
        }

        std::size_t accIdx = selectAccPredIndex(sub);
        const SubPredicate& accPred = sub.predicates.at(accIdx);
        auto dimIt = dimensions_.find(accPred.attr);
        if (dimIt == dimensions_.end()) {
            throw std::invalid_argument("pstree: unknown dimension '" + accPred.attr + "'");
        }
        DimensionIndex& dim = dimIt->second;

        // Stored BEFORE the leaf loop below, not after: reorganizeGroups() (triggered by
        // growth mid-loop) needs every currently-grouped subscription's own storage, including
        // this one, already in place. `subPtr` points directly into this map's own element
        // storage - safe to keep past this call and reuse from MatchEvent's hot loop below
        // (see LeafGroupState::groups' own comment): unordered_map guarantees references/
        // pointers to elements are never invalidated by insertion (only by erasing that same
        // element, which DeleteSubscription always does only after removing every `ids` entry
        // pointing at it - see there). `accIdx` is stored alongside the subscription, not
        // recomputed later, so DeleteSubscription reads back exactly what was chosen here even
        // though `observedValues` keeps changing afterward - see selectAccPredIndex's own
        // comment and the file-level comment for why recomputing at delete time would be wrong.
        auto [subIt, subInserted] = subscriptions_.emplace(sub.id, StoredSubscription{sub, accIdx});
        const Subscription* subPtr = &subIt->second.sub;

        auto lowLevel = buildLowLevel(dim, accPred);
        std::vector<LeafNode*> leafNodes = applyLowLevel(dim, lowLevel, /*insert=*/true);

        std::vector<std::string> subDims = detail::dimSigDimensions(sub);

        for (LeafNode* leaf : leafNodes) {
            LeafGroupState& state = groupStateFor(leaf);
            DimSig sig = calculateDimSig(subDims, state.dimSigLen);
            state.groups[sig].emplace_back(sub.id, subPtr);
            state.subNum += 1;
            if (state.subNum >= detail::growThreshold(state.dimSigLen)) {
                state.dimSigLen += 1;
                reorganizeGroups(state);
            }
        }
    }

    // Algorithm 5, MatchEvent. Restructured from the paper's own two-phase loop (collect
    // every leaf first, then process them) into one pass - each event attribute maps to a
    // DIFFERENT dimension's own PS-Tree, so there's no shared state between iterations the
    // separate phases could have been protecting; the result is identical either way.
    std::vector<std::uint64_t> matchEvent(const Event& event) const {
        std::vector<std::string> eventDims;
        eventDims.reserve(event.size());
        for (auto& pair : event) eventDims.push_back(pair.attr);

        std::vector<std::uint64_t> matchingSubs;
        for (auto& pair : event) {
            auto dimIt = dimensions_.find(pair.attr);
            if (dimIt == dimensions_.end()) continue; // event attribute outside the schema - ignore, not an error
            const DimensionIndex& dim = dimIt->second;

            ElementKey key = detail::encodeValue(pair.val, dim.schema);
            // A point can now be covered by several buckets at once (an equality bucket plus
            // any number of ancestor range markers along its path - see ps_tree.hpp's
            // canonical-decomposition redesign), not just one leaf.
            for (LeafNode* leaf : dim.tree.matchPoint(key)) {
                if (leaf->userData == nullptr) continue; // no subscription ever attached to this bucket

                const LeafGroupState& state = *static_cast<LeafGroupState*>(leaf->userData);
                DimSig eventSig = calculateDimSig(eventDims, state.dimSigLen);
                for (auto& [groupSig, ids] : state.groups) {
                    if (!groupSig.isSubsetOf(eventSig)) continue; // group needs a dimension this event doesn't have
                    for (auto& [id, subPtr] : ids) {
                        if (matchSubscription(event, *subPtr)) {
                            matchingSubs.push_back(id);
                        }
                    }
                }
            }
        }
        return matchingSubs;
    }

    // Algorithm 6, DeleteSubscription.
    void deleteSubscription(std::uint64_t subId) {
        auto subIt = subscriptions_.find(subId);
        if (subIt == subscriptions_.end()) {
            throw std::invalid_argument("pstree: deleting unknown subscription id " + std::to_string(subId));
        }
        const Subscription sub = subIt->second.sub; // copy - erased from subscriptions_ before returning
        // Read back the index InsertSubscription chose and stored, rather than recomputing via
        // selectAccPredIndex() - see that function's own comment for why recomputing here could
        // now pick a DIFFERENT predicate than insert time did (observedValues keeps changing as
        // more subscriptions are inserted in between).
        std::size_t accIdx = subIt->second.accIdx;
        const SubPredicate& accPred = sub.predicates.at(accIdx);
        DimensionIndex& dim = dimensions_.at(accPred.attr);

        auto lowLevel = buildLowLevel(dim, accPred);
        std::vector<LeafNode*> leafNodes = applyLowLevel(dim, lowLevel, /*insert=*/false);

        std::vector<std::string> subDims = detail::dimSigDimensions(sub);

        for (LeafNode* leaf : leafNodes) {
            if (leaf->userData == nullptr) {
                throw std::logic_error("pstree: leaf group state missing on delete - insert/delete bookkeeping bug");
            }
            LeafGroupState& state = *static_cast<LeafGroupState*>(leaf->userData);
            DimSig sig = calculateDimSig(subDims, state.dimSigLen);
            auto groupIt = state.groups.find(sig);
            if (groupIt == state.groups.end()) {
                throw std::logic_error("pstree: subscription's group missing on delete - insert/delete bookkeeping bug");
            }
            auto& ids = groupIt->second;
            auto idIt = std::find_if(ids.begin(), ids.end(),
                                      [subId](const auto& entry) { return entry.first == subId; });
            if (idIt == ids.end()) {
                throw std::logic_error("pstree: subscription id missing from its own group on delete");
            }
            ids.erase(idIt);
            if (ids.empty()) state.groups.erase(groupIt);
            state.subNum -= 1;
            if (state.dimSigLen > detail::kInitialDimSigLen &&
                state.subNum <= detail::shrinkThreshold(state.dimSigLen)) {
                state.dimSigLen -= 1;
                reorganizeGroups(state);
            }
        }

        // Space reclamation ("MergeSpaces" - see ps_tree.hpp's own file-level comment and this
        // repo's README "Space merging is deferred" note): now that every leaf's userData has
        // been fully read/updated above (the ONLY reason freeing had to wait - see reclaim()'s
        // own doc comment), it's safe to free any bucket that just dropped to a zero
        // predCounter, and to prune any InnerNode left with nothing in it as a result. Called
        // per low-level predicate, same as insert/delete themselves.
        for (auto& p : lowLevel) {
            dim.tree.reclaim(p);
        }

        subscriptions_.erase(subIt);
    }

private:
    // Attached directly to each PS-Tree leaf via LeafNode::userData (ps_tree.hpp's generic
    // escape hatch - see its file-level comment #4), NOT a side-table keyed by LeafNode*.
    // This matters for real correctness, not just convenience: a leaf a subscription is
    // already grouped under can be SPLIT into two by a LATER, wholly unrelated
    // subscription's own insertion (same as predCounter needing to be copied to the new
    // piece) - a LeafNode*-keyed side-table would silently leave the new piece with no
    // group state at all, exactly the bug a randomized stress test caught here (see
    // project history/README) before this design replaced the original side-table
    // approach. Attaching the state directly to the leaf means PSTree's own
    // copyLeafNode()/cloneUserData_ propagates it automatically on every split, the same
    // way predCounter already does.
    struct LeafGroupState {
        std::size_t dimSigLen = detail::kInitialDimSigLen;
        std::size_t subNum = 0;
        // Each entry pairs a subscription's id (needed only for MatchEvent's own returned
        // result list) with a POINTER directly into subscriptions_'s own element storage -
        // deliberately not just an id, to avoid an unordered_map::at(id) hash lookup for every
        // single candidate MatchEvent's innermost loop visits. Confirmed via `perf annotate` to
        // matter in practice: that lookup was ~63% of MatchEvent's own self-time on a
        // many-thousand-subscription workload before this change (see nats_sidecar's own
        // matching-engine benchmark/README). Safe because unordered_map never invalidates
        // references/pointers to an element except by erasing that exact element - see
        // InsertSubscription's and DeleteSubscription's own comments for why the ordering
        // guarantees a pointer is never read here after its target is erased.
        std::unordered_map<DimSig, std::vector<std::pair<std::uint64_t, const Subscription*>>, DimSig::Hasher> groups;
    };

    static void destroyLeafGroupState(void* p) { delete static_cast<LeafGroupState*>(p); }

    struct DimensionIndex {
        explicit DimensionIndex(AttrSchema s)
            : schema(std::move(s)), tree(detail::shapeFor(schema), destroyLeafGroupState) {}
        AttrSchema schema;
        PSTree tree;
        // Distinct literal values ever referenced by ANY predicate on this dimension, across
        // every subscription ever inserted (see insertSubscription's own comment for why - not
        // just chosen access predicates) - used by selectAccPredIndex to normalize the
        // kElemOf/kNotElemOf width tie-break by real domain cardinality instead of raw list
        // width alone. Grows monotonically, never cleared on delete: "how diverse has this
        // dimension's real data ever looked" is a lifetime signal, not a live-population count,
        // and real attribute domains are finite in practice (same assumption the rest of this
        // codebase already makes about domains being bounded).
        std::unordered_set<ElementKey, ElementKey::Hasher> observedValues;
    };

    // Subscription storage, paired with the access-predicate index InsertSubscription chose for
    // it - see selectAccPredIndex's own comment for why this must be computed once and
    // remembered, not recomputed at delete time.
    struct StoredSubscription {
        Subscription sub;
        std::size_t accIdx;
    };

    // Returns this leaf's LeafGroupState, allocating one on first touch.
    static LeafGroupState& groupStateFor(LeafNode* leaf) {
        if (leaf->userData == nullptr) leaf->userData = new LeafGroupState();
        return *static_cast<LeafGroupState*>(leaf->userData);
    }

    // Builds the low-level pstree::Predicate list for one access predicate (usually one
    // entry; kElemOf produces one kEq per literal value - see file-level comment). Pure and
    // deterministic from `pred`/`dim.schema` alone - exactly what guarantees insert and delete
    // (and, for delete, the later reclaim() pass - see deleteSubscription) all agree on the
    // identical decomposition.
    static std::vector<Predicate> buildLowLevel(DimensionIndex& dim, const SubPredicate& pred) {
        std::vector<Predicate> lowLevel;
        switch (pred.op) {
            case CmpOp::kLt: {
                auto prev = prevElementKey(dim.tree.shape(), detail::encodeValue(pred.vals.at(0), dim.schema));
                if (prev) lowLevel.push_back({Op::kLe, *prev, {}});
                break;
            }
            case CmpOp::kLe:
                lowLevel.push_back({Op::kLe, detail::encodeValue(pred.vals.at(0), dim.schema), {}});
                break;
            case CmpOp::kEq:
                lowLevel.push_back({Op::kEq, detail::encodeValue(pred.vals.at(0), dim.schema), {}});
                break;
            case CmpOp::kGt: {
                auto next = nextElementKey(dim.tree.shape(), detail::encodeValue(pred.vals.at(0), dim.schema));
                if (next) lowLevel.push_back({Op::kGe, *next, {}});
                break;
            }
            case CmpOp::kGe:
                lowLevel.push_back({Op::kGe, detail::encodeValue(pred.vals.at(0), dim.schema), {}});
                break;
            case CmpOp::kBetween:
                lowLevel.push_back({Op::kIn, detail::encodeValue(pred.vals.at(0), dim.schema),
                                     detail::encodeValue(pred.vals.at(1), dim.schema)});
                break;
            case CmpOp::kElemOf: {
                // Deduplicate by encoded key: kElemOf's set semantics mean a repeated
                // literal (['ee','ee']) must behave identically to a single one (['ee']).
                // Skipping this would insert kEq(V) twice for the same V, returning the
                // SAME leaf twice in the union below - and since insertSubscription's own
                // caller loop groups every entry in that union, the subscription would get
                // added to its own group twice, double-counting it in every future match.
                // Caught by test_pst_dynamic_stress.cpp generating exactly this case.
                std::vector<ElementKey> seen;
                for (auto& v : pred.vals) {
                    ElementKey key = detail::encodeValue(v, dim.schema);
                    if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
                    seen.push_back(key);
                    lowLevel.push_back({Op::kEq, key, {}});
                }
                break;
            }
            case CmpOp::kNe:
            case CmpOp::kNotElemOf:
            case CmpOp::kIsNotNull:
                // "X is not null" as an access predicate means exactly "X is present" -
                // which is precisely the condition under which MatchEvent ever consults
                // this dimension's tree in the first place (its own per-pair loop only
                // visits dimensions the event actually has). So "matches every leaf" is
                // not just a safe fallback here, it's the CORRECT indexing: every event
                // that reaches this tree already satisfies kIsNotNull by construction.
                lowLevel.push_back({Op::kGe, detail::minKeyFor(dim.schema), {}});
                break;
            case CmpOp::kIsNull:
                // The opposite direction has no indexable meaning at all: "X is null"
                // only matches events where X is ABSENT, but MatchEvent structurally never
                // consults X's own tree for such events (there's no pair to route through).
                // Rejected by insertSubscription/deleteSubscription before this function is
                // ever reached when kIsNull would be the chosen access predicate (see their
                // own doc comments) - reaching this case means that guard was bypassed.
                throw std::logic_error(
                    "pstree: kIsNull cannot be used as an access predicate - "
                    "this should have been rejected before reaching buildLowLevel");
        }
        return lowLevel;
    }

    // Applies a pre-built low-level predicate list (insert or delete) and unions the affected
    // leaves - split out from buildLowLevel so deleteSubscription can hold onto the same
    // `lowLevel` it applied here and pass it again to reclaim() afterward (see there for why
    // that has to happen as a distinct, later step, not folded into this one).
    static std::vector<LeafNode*> applyLowLevel(DimensionIndex& dim, const std::vector<Predicate>& lowLevel, bool insert) {
        std::vector<LeafNode*> all;
        for (auto& p : lowLevel) {
            auto leaves = insert ? dim.tree.insertPredicate(p) : dim.tree.deletePredicate(p);
            all.insert(all.end(), leaves.begin(), leaves.end());
        }
        return all;
    }

    // Re-hashes every currently-grouped subscription at the leaf's NEW dimSigLen and
    // rebuilds the group map from scratch. Not given as pseudocode by the paper at all
    // (ReorganizeGroups is referenced by name only) - this is the natural, only-sensible
    // implementation of what it must do given the rest of Algorithm 4/6's own description.
    void reorganizeGroups(LeafGroupState& state) {
        std::vector<std::pair<std::uint64_t, const Subscription*>> allSubs;
        for (auto& [sig, ids] : state.groups) {
            allSubs.insert(allSubs.end(), ids.begin(), ids.end());
        }
        state.groups.clear();
        for (auto& [id, subPtr] : allSubs) {
            DimSig sig = calculateDimSig(detail::dimSigDimensions(*subPtr), state.dimSigLen);
            state.groups[sig].emplace_back(id, subPtr);
        }
    }

    std::unordered_map<std::string, DimensionIndex> dimensions_;
    std::unordered_map<std::uint64_t, StoredSubscription> subscriptions_;
};

} // namespace pstree
