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
//     {kLt,kLe,kGt,kGe} > kNotElemOf > kNe), with narrower value width as a tie-break within
//     the same tier, and first-in-subscription-order as the final tie-break - a pure,
//     deterministic function of the subscription alone (required: DeleteSubscription must
//     reselect the identical access predicate InsertSubscription chose). Verified against
//     Fig. 3's own 6 subscriptions: this heuristic picks exactly the access predicates the
//     paper states it does for every one of them (see test_pst_dynamic.cpp).
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
        case CmpOp::kNe: return 5;
    }
    return 5;
}

// Smaller = more selective (narrower). Only meaningful within the same rank tier (Section
// 2.3: "When the operators have the same values, we consider predicates with wider value
// sets to have lower selectivity"). Operators with no natural width always return 0 (a tie,
// falling through to subscription order).
inline double widthTieBreak(const SubPredicate& p) {
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
            return static_cast<double>(p.vals.size());
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

} // namespace detail

// Deterministic access-predicate selection (Section 2.3's static heuristic - see
// file-level comment). Same function used by both InsertSubscription and
// DeleteSubscription, so a subscription's access predicate never changes across its
// lifetime.
inline std::size_t selectAccPredIndex(const Subscription& sub) {
    if (sub.predicates.empty()) {
        throw std::invalid_argument("pstree: subscription " + std::to_string(sub.id) + " has no predicates");
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < sub.predicates.size(); ++i) {
        int rankBest = detail::opRank(sub.predicates[best].op);
        int rankI = detail::opRank(sub.predicates[i].op);
        bool better = rankI < rankBest ||
            (rankI == rankBest && detail::widthTieBreak(sub.predicates[i]) < detail::widthTieBreak(sub.predicates[best]));
        if (better) best = i;
    }
    return best;
}

class PSTDynamic {
public:
    explicit PSTDynamic(std::vector<AttrSchema> schema) {
        for (auto& s : schema) {
            std::string name = s.name;
            dimensions_.try_emplace(name, std::move(s));
        }
    }

    // Algorithm 4, InsertSubscription.
    void insertSubscription(const Subscription& sub) {
        if (subscriptions_.count(sub.id) != 0) {
            throw std::invalid_argument("pstree: subscription id " + std::to_string(sub.id) + " already exists");
        }
        std::size_t accIdx = selectAccPredIndex(sub);
        const SubPredicate& accPred = sub.predicates.at(accIdx);
        auto dimIt = dimensions_.find(accPred.attr);
        if (dimIt == dimensions_.end()) {
            throw std::invalid_argument("pstree: unknown dimension '" + accPred.attr + "'");
        }
        DimensionIndex& dim = dimIt->second;

        // Stored BEFORE the leaf loop below, not after: reorganizeGroups() (triggered by
        // growth mid-loop) looks up every currently-grouped subscription, including this
        // one, via subscriptions_.at() - it must already be there.
        subscriptions_.emplace(sub.id, sub);

        std::vector<LeafNode*> leafNodes = applyToTree(dim, accPred, /*insert=*/true);

        std::vector<std::string> subDims;
        subDims.reserve(sub.predicates.size());
        for (auto& p : sub.predicates) subDims.push_back(p.attr);

        for (LeafNode* leaf : leafNodes) {
            LeafGroupState& state = groupStateFor(leaf);
            DimSig sig = calculateDimSig(subDims, state.dimSigLen);
            state.groups[sig].push_back(sub.id);
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
            LeafNode* leaf = dim.tree.matchPair(key);
            if (leaf->userData == nullptr) continue; // no subscription ever attached to this leaf

            const LeafGroupState& state = *static_cast<LeafGroupState*>(leaf->userData);
            DimSig eventSig = calculateDimSig(eventDims, state.dimSigLen);
            for (auto& [groupSig, ids] : state.groups) {
                if (!groupSig.isSubsetOf(eventSig)) continue; // group needs a dimension this event doesn't have
                for (auto id : ids) {
                    const Subscription& sub = subscriptions_.at(id);
                    if (matchSubscription(event, sub)) {
                        matchingSubs.push_back(id);
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
        const Subscription sub = subIt->second; // copy - erased from subscriptions_ before returning

        std::size_t accIdx = selectAccPredIndex(sub);
        const SubPredicate& accPred = sub.predicates.at(accIdx);
        DimensionIndex& dim = dimensions_.at(accPred.attr);

        std::vector<LeafNode*> leafNodes = applyToTree(dim, accPred, /*insert=*/false);

        std::vector<std::string> subDims;
        subDims.reserve(sub.predicates.size());
        for (auto& p : sub.predicates) subDims.push_back(p.attr);

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
            auto idIt = std::find(ids.begin(), ids.end(), subId);
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
        std::unordered_map<DimSig, std::vector<std::uint64_t>, DimSig::Hasher> groups;
    };

    static void* cloneLeafGroupState(void* p) {
        if (p == nullptr) return nullptr;
        return new LeafGroupState(*static_cast<LeafGroupState*>(p));
    }
    static void destroyLeafGroupState(void* p) { delete static_cast<LeafGroupState*>(p); }

    struct DimensionIndex {
        explicit DimensionIndex(AttrSchema s)
            : schema(std::move(s)), tree(detail::shapeFor(schema), cloneLeafGroupState, destroyLeafGroupState) {}
        AttrSchema schema;
        PSTree tree;
    };

    // Returns this leaf's LeafGroupState, allocating one on first touch.
    static LeafGroupState& groupStateFor(LeafNode* leaf) {
        if (leaf->userData == nullptr) leaf->userData = new LeafGroupState();
        return *static_cast<LeafGroupState*>(leaf->userData);
    }

    // Builds the low-level pstree::Predicate list for one access predicate (usually one
    // entry; kElemOf produces one kEq per literal value - see file-level comment), then
    // applies each via PSTree's own insertPredicate/deletePredicate, unioning the affected
    // leaves. `insert` selects which PSTree operation runs; the predicate list itself is
    // identical either way (deterministic from `pred` and `dim.schema` alone), which is
    // exactly what guarantees delete finds every leaf insert touched.
    static std::vector<LeafNode*> applyToTree(DimensionIndex& dim, const SubPredicate& pred, bool insert) {
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
                lowLevel.push_back({Op::kGe, detail::minKeyFor(dim.schema), {}});
                break;
        }
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
        std::vector<std::uint64_t> allSubs;
        for (auto& [sig, ids] : state.groups) {
            allSubs.insert(allSubs.end(), ids.begin(), ids.end());
        }
        state.groups.clear();
        for (auto id : allSubs) {
            const Subscription& sub = subscriptions_.at(id);
            std::vector<std::string> dims;
            dims.reserve(sub.predicates.size());
            for (auto& p : sub.predicates) dims.push_back(p.attr);
            DimSig sig = calculateDimSig(dims, state.dimSigLen);
            state.groups[sig].push_back(id);
        }
    }

    std::unordered_map<std::string, DimensionIndex> dimensions_;
    std::unordered_map<std::uint64_t, Subscription> subscriptions_;
};

} // namespace pstree
