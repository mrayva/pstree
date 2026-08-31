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
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pstree/dim_sig.hpp"
#include "pstree/order_key.hpp"
#include "pstree/predicate.hpp"
#include "pstree/ps_tree.hpp"

namespace pstree {

enum class ValueType { kBoolean, kInteger, kFloat, kString };

// String interning for PS-Tree indexing: this project's shared expression grammar (confirmed by
// reading be-tree's own parser.y directly - num_comp_value, the value type behind </<=/>/>=, is
// integer/float only, with no string alternative anywhere in the grammar) can NEVER produce an
// ordering predicate (kLt/kLe/kGt/kGe/kBetween) against a string-typed attribute - only
// kEq/kNe/kElemOf/kNotElemOf ever reach a string dimension. PS-Tree's per-dimension element
// encoding only needs to be ORDER-PRESERVING to support ordering predicates correctly; for
// equality/set-membership alone, any stable, collision-free (injective) mapping works. That's
// exactly what this table provides - real strings get interned to small int64_t ids, then flow
// through the existing Int64Codec (16 elements) instead of StringCodec (up to stringMaxLen
// elements, 32 in nats_sidecar's own configured usage) - the same mechanism already measured to
// explain most of pstree's own string-vs-integer matching-cost gap (nats_sidecar's
// diag_int_attrs.cpp/perf_search_loop experiment), now paid once per with_string()-equivalent
// event-population call instead of once per PS-Tree comparison, matching how a-tree's own
// strings.rs::StringTable already works.
//
// internForInsert/lookupForSearch are deliberately separate, not one function with a bool flag
// (this codebase's own convention - see e.g. applyLowLevel's `insert` bool being the one
// exception, kept only because it mirrors PSTree's own insert/delete pairing): a predicate's own
// literal value DEFINES the domain (an unseen literal must get a fresh id), while an event's
// live attribute value must NEVER be allowed to allocate one (a value no subscription ever
// referenced has to compare unequal to every real interned id, not silently collide with
// whatever id happens to be allocated next) - returning kSentinel on a lookup miss guarantees
// that, mirroring a-tree's own StringTable::get() (miss) vs get_or_update() (insert) split.
//
// Both are called from exactly two places, each ONCE per (subscription | event), not per
// comparison - PSTDynamic::insertSubscription interns every string-typed predicate value of a
// subscription up front (not just its chosen access predicate's - see that function's own
// comment for why this matters for predicate.hpp's own full-verification path, not just this
// file's PS-Tree indexing) and stores the ALREADY-INTERNED subscription; PSTDynamic::matchEvent
// interns every string-typed event attribute value up front and uses that interned copy for
// both the PS-Tree lookup below AND predicate.hpp's matchSubscription() full-verification call.
// Because of this, `encodeValue` below (used by both PS-Tree indexing and, indirectly by never
// needing special-casing, predicate.hpp's own comparisons - see that file for why it needs NO
// changes at all) always sees an already-interned std::int64_t for a kString dimension by the
// time it's called - never a raw std::string.
class StringInternTable {
public:
    // The smallest possible int64_t - matches minKeyFor's own existing "smallest representable
    // key" convention for kInteger, and can never collide with a real id (ids are allocated
    // starting at 0, monotonically increasing - see internForInsert).
    static constexpr std::int64_t kSentinel = std::numeric_limits<std::int64_t>::min();

    std::int64_t internForInsert(std::string_view s) {
        if (auto it = ids_.find(s); it != ids_.end()) return it->second;
        std::int64_t id = static_cast<std::int64_t>(ids_.size());
        auto [it, inserted] = ids_.emplace(std::string(s), id);
        (void)inserted;
        return it->second;
    }

    std::int64_t lookupForSearch(std::string_view s) const {
        auto it = ids_.find(s);
        return it != ids_.end() ? it->second : kSentinel;
    }

private:
    // Heterogeneous lookup (std::string key, std::string_view probe) - same
    // transparent-hash/equal_to<> pattern nats_sidecar's own string_view_lookup_map uses, for
    // the same reason: avoid constructing a temporary std::string just to look up an existing
    // key. Table lifetime matches one DimensionIndex's own lifetime (see AttrSchema::stringIntern
    // and PSTDynamic's constructor) - built fresh alongside each from-scratch tree rebuild, never
    // needing cross-rebuild consistency.
    struct TransparentStringHash {
        using is_transparent = void;
        std::size_t operator()(std::string_view sv) const noexcept {
            return std::hash<std::string_view>{}(sv);
        }
    };
    std::unordered_map<std::string, std::int64_t, TransparentStringHash, std::equal_to<>> ids_;
};

struct AttrSchema {
    std::string name;
    ValueType type;
    std::size_t stringMaxLen = 64; // unused since string interning replaced StringCodec - kept
                                    // only so any external caller still passing it compiles;
                                    // remove once nats_sidecar's own kPstreeStringMaxLen is
                                    // retired.
    // Lazily allocated by DimensionIndex's constructor for kString attributes - see
    // StringInternTable's own comment. shared_ptr (not a value member) because AttrSchema gets
    // copied in places that must all observe the SAME table for a given dimension, not each get
    // their own.
    std::shared_ptr<StringInternTable> stringIntern;
};

namespace detail {

inline KeyShape shapeFor(const AttrSchema& schema) {
    switch (schema.type) {
        case ValueType::kBoolean: return BoolCodec::shape();
        case ValueType::kInteger: return Int64Codec::shape();
        case ValueType::kFloat: return DoubleCodec::shape();
        case ValueType::kString: return Int64Codec::shape(); // interned id, see StringInternTable
    }
    throw std::logic_error("pstree: unreachable ValueType in shapeFor");
}

// Encodes an already-processed Value into its ElementKey. For kString, `v` MUST already hold
// the interned std::int64_t id (see this file's own top comment for exactly where that
// interning happens - insertSubscription/matchEvent, never here) - this function itself no
// longer knows or cares whether a dimension is "logically" a string; once interned, kString and
// kInteger are representationally identical, sharing Int64Codec.
inline ElementKey encodeValue(const Value& v, const AttrSchema& schema) {
    switch (schema.type) {
        case ValueType::kBoolean: return BoolCodec::encode(std::get<bool>(v));
        case ValueType::kInteger:
        case ValueType::kString:
            return Int64Codec::encode(std::get<std::int64_t>(v));
        case ValueType::kFloat: return DoubleCodec::encode(std::get<double>(v));
    }
    throw std::logic_error("pstree: unreachable ValueType in encodeValue");
}

// Used for the kNe/kNotElemOf "matches every leaf" fallback (see file-level comment). A
// codec-level constant (the smallest representable key in the dimension's own keyspace), not
// tied to any real value - kString now shares kInteger's own int64_t keyspace (see
// StringInternTable), so it shares this same fallback with no string-specific logic needed.
inline ElementKey minKeyFor(const AttrSchema& schema) {
    switch (schema.type) {
        case ValueType::kBoolean: return BoolCodec::encode(false);
        case ValueType::kInteger:
        case ValueType::kString:
            return Int64Codec::encode(std::numeric_limits<std::int64_t>::min());
        case ValueType::kFloat: return DoubleCodec::encode(-std::numeric_limits<double>::infinity());
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
    // Generous, fixed upper bound on schema size. Two independent things depend on it fitting
    // in a small range: matchEvent() builds its per-call `indexed` lookup array (see that
    // function's own comment) on the stack instead of via a heap allocation, and
    // SubPredicate::attrIndex (predicate.hpp) is a uint8_t, not a size_t, specifically so
    // adding it didn't grow SubPredicate's own size at all (see that field's own comment for
    // why that mattered - a real, measured throughput regression, isolated by testing the
    // heap-vs-stack array question SEPARATELY from the struct-size question: switching
    // matchEvent's `indexed` from std::vector to this fixed std::array did NOT recover the
    // regression on its own, but shrinking attrIndex from size_t to uint8_t did - the array
    // itself was kept anyway, since avoiding a per-event heap allocation is a real
    // improvement even though it wasn't the confirmed cause here). Enforced once, at
    // construction (below) - every real schema this project uses (2-10 attributes) sits far
    // under this bound; raise it (and attrIndex's own type, if it would no longer fit a
    // uint8_t) if a future one ever needs more.
    static constexpr std::size_t kMaxSchemaAttrs = 32;

    explicit PSTDynamic(std::vector<AttrSchema> schema) {
        if (schema.size() > kMaxSchemaAttrs) {
            throw std::invalid_argument(
                "pstree: schema has " + std::to_string(schema.size()) +
                " attributes, more than kMaxSchemaAttrs (" + std::to_string(kMaxSchemaAttrs) +
                ") - raise the bound in pst_dynamic.hpp if a real schema ever needs that many");
        }
        for (std::size_t i = 0; i < schema.size(); ++i) {
            AttrSchema& s = schema[i];
            std::string name = s.name;
            if (s.type == ValueType::kString && !s.stringIntern) {
                s.stringIntern = std::make_shared<StringInternTable>();
            }
            dimensions_.try_emplace(name, std::move(s), i);
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
        // Ordering predicates against a string-typed dimension have no valid meaning under
        // string interning (see StringInternTable's own comment: interned ids are NOT
        // order-preserving w.r.t. the original strings) used to be checked HERE, scoped only to
        // whichever predicate this function chose as the access predicate - too narrow: a
        // subscription's OTHER predicates never reach PS-Tree indexing at all (they're checked
        // via predicate.hpp's full-verification path instead), but string interning now applies
        // to ALL of a subscription's string-typed predicate values (see insertSubscription's own
        // comment for why), so an ordering op on a NON-access string predicate is equally
        // unsupported and needs the identical rejection. Moved to insertSubscription's own
        // interning loop instead, which walks every predicate (not just whichever one this
        // function ends up choosing) and runs even earlier, before any state mutation at all -
        // see that comment for the full rationale. Caught by test_pst_dynamic_stress.cpp's own
        // randomized fuzzing generating exactly this case (a non-access kLt on a string
        // dimension) once this file's random op generator started exercising it.
        return best;
    }

    // Algorithm 4, InsertSubscription.
    void insertSubscription(const Subscription& sub) {
        if (subscriptions_.count(sub.id) != 0) {
            throw std::invalid_argument("pstree: subscription id " + std::to_string(sub.id) + " already exists");
        }

        // Interned copy of the whole subscription, built ONCE here - every string-typed
        // predicate's own literal value gets replaced by its interned int64_t id (NOT just the
        // one that ends up chosen as this subscription's access predicate). This is what lets
        // predicate.hpp's full-verification path (matchSubscription/matchValue/
        // elemOfTypedContains, checked for every predicate OTHER than the access predicate) take
        // the cheap integer-comparison path too, with zero changes needed there - those
        // functions already dispatch generically on Value's actual variant alternative, so
        // feeding them an int64_t instead of a std::string for a "logically string" dimension is
        // enough on its own (found via `perf`: elemOfTypedContains's std::binary_search over
        // std::string values, real string memcmp included, was the single largest cost in the
        // whole profile for a two-string-predicate subscription shape like `exchange in (...)
        // and symbol in (...)` - the FIRST string-interning pass only sped up the ONE predicate
        // chosen as the access predicate, leaving this exact cost completely untouched).
        // The caller's own `sub` is never mutated - `vals` being `mutable` (predicate.hpp) is
        // for lazy comparison caches, not license to silently rewrite values the caller passed
        // in and might still hold a reference to.
        Subscription interned = sub;
        for (auto& pred : interned.predicates) {
            auto predDimIt = dimensions_.find(pred.attr);
            if (predDimIt == dimensions_.end()) continue; // unknown dimension - rejected below if selected
            DimensionIndex& predDim = predDimIt->second;
            // Resolved from the SAME lookup already needed for interning/cardinality-tracking
            // below - no extra hashmap hit. See SubPredicate::attrIndex's own comment.
            pred.attrIndex = static_cast<std::uint8_t>(predDim.index);
            if (predDim.schema.type == ValueType::kString) {
                // Ordering predicates against a string-typed dimension have no valid meaning
                // under interning (interned ids are NOT order-preserving w.r.t. the original
                // strings - see StringInternTable's own comment) - checked for EVERY predicate
                // here, not just whichever one selectAccPredIndex later chooses, since a
                // subscription's non-access predicates get interned too and go through
                // predicate.hpp's full-verification path, which would otherwise silently compare
                // wrong (non-order-preserving) integers instead of the original strings. Checked
                // before this predicate's own values are touched below, and before any state
                // mutation elsewhere in this function - a caller catching this exception (the
                // same std::invalid_argument every other insertSubscription rejection already
                // throws) sees a clean "not inserted," never a half-registered subscription.
                switch (pred.op) {
                    case CmpOp::kLt:
                    case CmpOp::kLe:
                    case CmpOp::kGt:
                    case CmpOp::kGe:
                    case CmpOp::kBetween:
                        throw std::invalid_argument(
                            "pstree: subscription " + std::to_string(sub.id) +
                            " has an ordering predicate (</<=/>/>=/between) on string-typed "
                            "attribute '" + pred.attr + "', which interned string ids do not "
                            "support");
                    default:
                        break;
                }
                for (auto& v : pred.vals) {
                    v = Value(predDim.schema.stringIntern->internForInsert(std::get<std::string>(v)));
                }
            }
            // Feed every predicate's own (now-interned) literal values into its dimension's
            // observed-cardinality tracker BEFORE selecting this subscription's own access
            // predicate (see DimensionIndex::observedValues and selectAccPredIndex's own
            // comment) - including predicates that DON'T end up chosen as the access predicate,
            // so a wide-domain dimension's true cardinality becomes visible even from
            // subscriptions that (correctly) keep preferring some other, still-more-selective
            // predicate. Without this, a dimension nothing has ever been indexed ON would never
            // accumulate the evidence needed to eventually win a tie-break against a
            // narrower-but-lower-cardinality one.
            for (const auto& v : pred.vals) {
                predDim.observedValues.insert(detail::encodeValue(v, predDim.schema));
            }

            // Builds this predicate's comparison cache NOW, single-threaded, before this
            // subscription is ever reachable by a concurrent matchEvent() call (the
            // subscriptions_.emplace below is what first makes it reachable) - see
            // ensurePredicateCachedForInsert's own comment in predicate.hpp for the real
            // thread-safety bug this fixes (a SIGSEGV reproduced with worker_threads>1: two
            // threads racing to lazily build/reset the SAME predicate's cache on their first
            // independent call into matchValue). Applied to EVERY predicate, not just this
            // subscription's eventual access predicate - matchValue's own full-verification
            // path evaluates every OTHER predicate too.
            ensurePredicateCachedForInsert(pred);
        }

        std::size_t accIdx = selectAccPredIndex(interned);
        const std::string& accAttr = interned.predicates.at(accIdx).attr;
        if (dimensions_.find(accAttr) == dimensions_.end()) {
            throw std::invalid_argument("pstree: unknown dimension '" + accAttr + "'");
        }
        std::vector<std::string> subDims = detail::dimSigDimensions(interned);

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
        // `sub.id` (not `interned.id`) as the map key/id, for clarity - both are identical since
        // `interned` is a straight copy of `sub` with only predicate VALUES rewritten.
        auto [subIt, subInserted] =
            subscriptions_.emplace(sub.id, StoredSubscription{std::move(interned), accIdx});
        const Subscription* subPtr = &subIt->second.sub;
        // Re-bound from the STORED (moved-into) copy, not the now-moved-from local `interned` -
        // both hold the identical interned values either way, this just avoids reading through
        // a moved-from object.
        const SubPredicate& accPred = subPtr->predicates.at(accIdx);
        DimensionIndex& dim = dimensions_.at(accPred.attr);

        auto lowLevel = buildLowLevel(dim, accPred);
        std::vector<LeafNode*> leafNodes = applyLowLevel(dim, lowLevel, /*insert=*/true);

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
        // Interned copy of the whole event, built ONCE here - every string-typed attribute's
        // value gets replaced by its interned int64_t id (via lookupForSearch: read-only, a
        // value no subscription ever referenced maps to StringInternTable::kSentinel rather
        // than allocating - see that class's own comment). Used for BOTH the per-dimension
        // PS-Tree lookup below AND matchSubscription()'s own full-verification re-lookup of
        // this same event's values (via findAttr, once per predicate) - interning here, once
        // per event attribute, instead of leaving matchSubscription to repeatedly compare a raw
        // std::string against however many candidate subscriptions' own predicates reference it,
        // is what actually makes the full-verification path fast too, not just the one
        // predicate PS-Tree itself indexes - see insertSubscription's own comment for the other
        // half of this (why the STORED subscription needs its OTHER predicates interned too,
        // not just its access predicate).
        //
        // `indexed` is built in the SAME pass, for the same "pay once per event attribute,
        // amortize across every candidate subscription" reason: indexed[i] is the Value for
        // whichever event attribute has ordinal `i` (DimensionIndex::index), or nullptr if
        // this event doesn't have that attribute at all - see matchSubscriptionIndexed's own
        // comment in predicate.hpp for why this replaces findAttr()'s per-(predicate,
        // candidate) name scan with an O(1) array read. Local to this call, never shared
        // across matchEvent() invocations or threads - see StringInternTable's own comment for
        // why a shared mutable scratch buffer here would reintroduce the exact class of race
        // just fixed (found via `perf`, 2026-08-30: real, comparable in size to std::variant's
        // own dispatch overhead at real subscription counts).
        // dimensions_.size() <= kMaxSchemaAttrs is a permanent, whole-lifetime invariant,
        // enforced once in the constructor - no runtime check needed on this per-event path.
        Event internedEvent = event;
        std::array<const Value*, kMaxSchemaAttrs> indexedStorage{};
        std::span<const Value*> indexed(indexedStorage.data(), dimensions_.size());
        for (auto& pair : internedEvent) {
            auto dimIt = dimensions_.find(pair.attr);
            if (dimIt == dimensions_.end()) continue; // event attribute outside the schema - ignore, not an error
            if (dimIt->second.schema.type == ValueType::kString) {
                pair.val = Value(dimIt->second.schema.stringIntern->lookupForSearch(std::get<std::string>(pair.val)));
            }
            indexed[dimIt->second.index] = &pair.val;
        }

        std::vector<std::string> eventDims;
        eventDims.reserve(internedEvent.size());
        for (auto& pair : internedEvent) eventDims.push_back(pair.attr);

        std::vector<std::uint64_t> matchingSubs;
        for (auto& pair : internedEvent) {
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
                        if (matchSubscriptionIndexed(indexed, *subPtr)) {
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
        DimensionIndex(AttrSchema s, std::size_t idx)
            : schema(std::move(s)), index(idx), tree(detail::shapeFor(schema), destroyLeafGroupState) {}
        AttrSchema schema;
        // This dimension's fixed ordinal among every attribute in the schema PSTDynamic was
        // constructed with (0-based, in the order the caller's own AttrSchema vector listed
        // them) - assigned once, at construction, never reassigned. Lets matchEvent() build a
        // plain array indexed by this value (see that function's own comment) instead of every
        // predicate-vs-candidate check doing a name-based lookup.
        std::size_t index;
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
    // Precondition: an ordering op (kLt/kLe/kGt/kGe/kBetween) against a string-typed dimension
    // is already rejected earlier, at selectAccPredIndex time - see that function's own comment
    // for why the check has to happen there and not here (this runs after
    // insertSubscription has already registered the subscription in subscriptions_).
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
