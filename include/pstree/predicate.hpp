#pragma once

// The high-level subscription language (paper Section 2.1): a predicate is
// <attribute, operator, value(s)>, a subscription is a conjunction of predicates, an event
// is a set of attribute-value pairs. This is deliberately separate from pstree::Op/
// pstree::Predicate (ps_tree.hpp) - those are PS-Tree's own low-level, ElementKey-based
// insertion/deletion vocabulary (only >=, =, <=, in - the four operators PS-Tree itself can
// represent as a single contiguous predicate space). This file is the paper's actual
// surface language, matching Section 2.1's full operator set (<,<=,=,!=,>,>=,in-BETWEEN,
// element-of, not-element-of) - the only operators from a Subscription that ever reach
// PSTree directly are whichever one gets selected as the access predicate (see
// pst_dynamic.hpp's SelectAccPred), always translated through pstree::Op's narrower set;
// every predicate (including the access predicate itself, re-checked) is evaluated here via
// Match() for the final per-subscription confirmation (Algorithm 5, line 11).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pstree {

// Section 2.1's full operator set. kBetween is the paper's own "in" (SQL BETWEEN, [lo,hi]
// inclusive) - named differently here to avoid colliding with the set-membership meaning
// most readers expect from "in"; kElemOf/kNotElemOf are the paper's actual set operators
// (also called ∈/∉ in the paper), each matching against a literal LIST of values.
//
// kIsNull/kIsNotNull are NOT part of the paper's own model at all - the paper has no
// concept of an attribute being explicitly "absent but tested for" (Section 2.2's semantics
// already treat a missing attribute as an automatic non-match for whatever predicate
// references it, full stop). They exist here because a real caller (nats_sidecar) needs to
// test presence/absence itself as a first-class predicate ("discount is null"), which
// `vals`-based value comparison can't express - see matchSubscription()'s own special-casing
// for them below, and pst_dynamic.hpp's SelectAccPred/applyToTree comments for why kIsNull
// specifically can never be used as an access predicate (no tree can index "this dimension
// was absent", since MatchEvent only ever consults a dimension's tree for events that DO
// have it).
enum class CmpOp {
    kLt,
    kLe,
    kEq,
    kNe,
    kGt,
    kGe,
    kBetween,   // paper's "in": vals[0] <= x <= vals[1]
    kElemOf,    // paper's "∈": x equals any of vals
    kNotElemOf, // paper's "∉": x equals none of vals
    kIsNull,    // vals empty: true iff the attribute is ABSENT from the event
    kIsNotNull, // vals empty: true iff the attribute is PRESENT in the event (any value)
};

// A predicate's value type - every attribute in a schema has exactly one of these types,
// consistently, across every subscription and event that references it (the same
// assumption a-tree/be-tree make; not re-validated per-call here, see matchValue()'s own
// doc comment for what happens if a caller violates it).
using Value = std::variant<bool, std::int64_t, double, std::string>;

// Lazily-cached, concretely-typed form of a single Value - used for kBetween's second operand
// (see SubPredicate::betweenUpperCache below).
using ScalarCache = std::variant<std::monostate, bool, std::int64_t, double, std::string>;

// Lazily-cached, concretely-typed form of EITHER an elemOf-family predicate's whole (sorted)
// value list OR a scalar-family predicate's single vals[0] threshold - see SubPredicate's own
// comment for why these two share one variant instead of each getting their own.
using PrimaryCache = std::variant<std::monostate, std::vector<bool>, std::vector<std::int64_t>,
                                   std::vector<double>, std::vector<std::string>,
                                   bool, std::int64_t, double, std::string>;

// Sentinel meaning "this predicate's attribute is not part of any known schema" - the same
// case findAttr()/an absent event attribute already handles as "never matches" (see
// matchSubscriptionIndexed's own comment), just reached via an out-of-range index instead of a
// null pointer.
inline constexpr std::size_t kNoAttrIndex = static_cast<std::size_t>(-1);

struct SubPredicate {
    std::string attr;
    CmpOp op;
    // Resolved ONCE, at insert time (PSTDynamic::insertSubscription, alongside string
    // interning and cache-building - see that function's own comment), to this predicate's
    // attribute's ordinal position in its schema (PSTDynamic::dimensions_' own per-dimension
    // `index` - see DimensionIndex). Lets matchSubscriptionIndexed() below look up this
    // predicate's event value by O(1) array index instead of findAttr()'s linear name scan -
    // found via `perf` (2026-08-30) to be a real, comparable-in-size cost next to std::variant's
    // own dispatch overhead, at the call volume real subscription counts produce (once per
    // predicate, per candidate subscription, per event). kNoAttrIndex (the default) means
    // "not resolved" - either this predicate was never inserted through PSTDynamic at all (a
    // predicate built for direct matchValue()/matchSubscription() testing, which never uses
    // this field), or its attribute isn't in the schema (mirrors findAttr() returning nullptr
    // for the same case - both mean "never matches").
    mutable std::size_t attrIndex = kNoAttrIndex;
    // mutable: matchValue() lazily sorts this in place (once, on the first kElemOf/kNotElemOf
    // match - see matchValue()'s own comment) so it can binary_search a large value list
    // instead of a linear scan. The SET of values a predicate holds never changes this way,
    // only their storage order - safe regardless of when it happens relative to insertion:
    // pst_dynamic.hpp's own buildLowLevel() (kElemOf's index-construction path) already
    // consumes these order-independently (a plain set union via a `for (auto& v : pred.vals)`
    // loop), and kBetween/kEq/etc's own single- or positional-index accesses (vals.at(0),
    // vals.at(1) for kBetween's [lo,hi]) are untouched since only kElemOf/kNotElemOf ever get
    // sorted.
    mutable std::vector<Value> vals; // size 1 for most ops, 2 for kBetween, >=1 for kElemOf/kNotElemOf

    // elemOf-family (kElemOf/kNotElemOf) and scalar-family (kLt/kLe/kEq/kNe/kGt/kGe/kBetween)
    // caches are MUTUALLY EXCLUSIVE by `op` - fixed permanently at construction to one family or
    // the other, never both - so sharing one storage slot between them avoids paying for a cache
    // this predicate will never use. This matters a lot here: SubPredicate sits INLINE
    // (kInlineCapacity = 4, see PredicateList below) inside every Subscription, so its own size
    // directly multiplies into how many cache lines a whole Subscription spans. Found
    // 2026-08-30: with this project's own earlier caching additions left as three SEPARATE
    // fields (an elemOf-only variant, plus two scalar-only ones), SubPredicate reached 208
    // bytes - 4x that inline pushed a Subscription to 864 bytes (13.5 cache lines), landing
    // PredicateList's own heap_/size_ bookkeeping (read on EVERY predicate-list access) 840
    // bytes from the object's start - the direct, measured cause of matchEvent's own dominant
    // cache-miss cost (see that class's own comment). elemOf-family stores its whole sorted
    // value list here (one of the four vector alternatives); scalar-family stores its single
    // vals[0] threshold (one of the four scalar alternatives). std::monostate before the first
    // match, same meaning either way: "not built yet."
    mutable bool primaryCached = false;
    mutable PrimaryCache primaryCache;

    // kBetween's SECOND operand only (vals[1]) - heap-indirected rather than inline, because
    // kBetween is confirmed dead code in this project's own dialect translation
    // (pstree_dialect.cpp never constructs it - "X >= lo and X <= hi" always compiles to two
    // separate kGe/kLe predicates instead, see pst_dynamic.hpp's own comment) - paying a full
    // cache's worth of inline space on literally every predicate for an operator this project's
    // real callers never construct wasn't defensible once it was this visibly implicated in a
    // real regression. Allocated lazily, only if a kBetween predicate is ever actually matched
    // (see ensureScalarCached) - null for every other predicate, at the cost of one pointer
    // (8 bytes) instead of a whole ScalarCache (40 bytes) most predicates never touch.
    mutable std::unique_ptr<ScalarCache> betweenUpperCache;

    // A unique_ptr member makes SubPredicate non-aggregate and non-copyable by default - this
    // constructor keeps every existing 3-arg brace-init call site (`SubPredicate{attr, op,
    // vals}`, all over pstree_dialect.cpp and this repo's own tests) working exactly as before,
    // and the copy operations below deep-copy betweenUpperCache's pointee (rather than the
    // pointer itself) so a copied predicate doesn't alias or double-free the original's cache -
    // PredicateList's own copy constructor/assignment (used when a Subscription itself is
    // copied) relies on SubPredicate being genuinely copyable, not just movable.
    SubPredicate(std::string attr_, CmpOp op_, std::vector<Value> vals_)
        : attr(std::move(attr_)), op(op_), vals(std::move(vals_)) {}

    SubPredicate(const SubPredicate& other)
        : attr(other.attr), op(other.op), vals(other.vals), attrIndex(other.attrIndex),
          primaryCached(other.primaryCached), primaryCache(other.primaryCache),
          betweenUpperCache(other.betweenUpperCache
                                 ? std::make_unique<ScalarCache>(*other.betweenUpperCache)
                                 : nullptr) {}
    SubPredicate& operator=(const SubPredicate& other) {
        if (this == &other) return *this;
        attr = other.attr;
        op = other.op;
        vals = other.vals;
        attrIndex = other.attrIndex;
        primaryCached = other.primaryCached;
        primaryCache = other.primaryCache;
        betweenUpperCache = other.betweenUpperCache
                                 ? std::make_unique<ScalarCache>(*other.betweenUpperCache)
                                 : nullptr;
        return *this;
    }
    SubPredicate(SubPredicate&&) noexcept = default;
    SubPredicate& operator=(SubPredicate&&) noexcept = default;
    ~SubPredicate() = default;
};

// Small-vector-optimized storage for a subscription's predicates: real subscriptions almost
// always have very few (this project's own benchmark generates 1-2 - see
// nats_sidecar/benchmarks/matching_engine_bench.cpp's generator) - inlining up to
// kInlineCapacity of them directly in the owning Subscription avoids a SEPARATE heap
// allocation, and its own independent cache-miss, for the common case. Falls back to a
// heap-allocated buffer (identical growth doubling to std::vector) only when a subscription
// genuinely has more predicates than that.
//
// Found via `perf annotate` on nats_sidecar's matching_engine_bench: with the wide-range
// insert-scaling redesign and three earlier search-side fixes all in place (see both repos'
// READMEs), MatchEvent's own remaining self-time was ~86% a cache-miss-dominated pointer
// chase - dereferencing a candidate Subscription's own storage, THEN separately dereferencing
// its predicates vector's own SEPARATE heap allocation just to read the first predicate's
// attribute name. Inlining predicates removes that second, independent allocation for the
// common case: touching the Subscription's own memory (unavoidable - it has to be read
// regardless) now also gives the predicate data itself, for free, in the same cache line(s).
//
// Deliberately hand-specialized to SubPredicate, not a reusable SmallVector<T,N> template:
// this is the one place in the codebase that needs this, and a narrow, fully-reasoned-about
// type carries less correctness risk than a general-purpose one would (no template
// instantiation surface beyond what's actually exercised and tested - see
// tests/test_predicate_list.cpp for the dedicated coverage this type's manual placement-new/
// destroy bookkeeping needs, on top of the existing indirect coverage via test_predicate.cpp/
// test_pst_dynamic*.cpp).
class PredicateList {
public:
    static constexpr std::size_t kInlineCapacity = 4;

    PredicateList() noexcept = default;

    PredicateList(std::initializer_list<SubPredicate> init) {
        reserveAtLeast(init.size());
        for (const auto& p : init) pushBackUnchecked(p);
    }

    // Non-explicit: needed so Subscription (still a plain aggregate - see below) can
    // aggregate-initialize its `predicates` member directly from an existing
    // std::vector<SubPredicate> (nats_sidecar's own pstree_clause is exactly that type -
    // see matching_engine.cpp's `Subscription{clauseId, clause}`).
    PredicateList(const std::vector<SubPredicate>& v) {
        reserveAtLeast(v.size());
        for (const auto& p : v) pushBackUnchecked(p);
    }
    PredicateList(std::vector<SubPredicate>&& v) {
        reserveAtLeast(v.size());
        for (auto& p : v) pushBackUnchecked(std::move(p));
    }

    PredicateList(const PredicateList& other) {
        reserveAtLeast(other.size_);
        for (std::size_t i = 0; i < other.size_; ++i) pushBackUnchecked(other.data()[i]);
    }
    PredicateList(PredicateList&& other) noexcept {
        if (other.heap_ != nullptr) {
            heap_ = other.heap_;
            heapCapacity_ = other.heapCapacity_;
            size_ = other.size_;
            other.heap_ = nullptr;
            other.heapCapacity_ = 0;
            other.size_ = 0;
        } else {
            for (std::size_t i = 0; i < other.size_; ++i) pushBackUnchecked(std::move(other.inlineData()[i]));
            other.clear();
        }
    }

    PredicateList& operator=(const PredicateList& other) {
        if (this == &other) return *this;
        clear();
        reserveAtLeast(other.size_);
        for (std::size_t i = 0; i < other.size_; ++i) pushBackUnchecked(other.data()[i]);
        return *this;
    }
    PredicateList& operator=(PredicateList&& other) noexcept {
        if (this == &other) return *this;
        clear();
        if (heap_ != nullptr) {
            ::operator delete(heap_);
            heap_ = nullptr;
            heapCapacity_ = 0;
        }
        if (other.heap_ != nullptr) {
            heap_ = other.heap_;
            heapCapacity_ = other.heapCapacity_;
            size_ = other.size_;
            other.heap_ = nullptr;
            other.heapCapacity_ = 0;
            other.size_ = 0;
        } else {
            for (std::size_t i = 0; i < other.size_; ++i) pushBackUnchecked(std::move(other.inlineData()[i]));
            other.clear();
        }
        return *this;
    }

    ~PredicateList() {
        clear();
        if (heap_ != nullptr) ::operator delete(heap_);
    }

    void push_back(const SubPredicate& p) {
        ensureCapacity(size_ + 1);
        pushBackUnchecked(p);
    }
    void push_back(SubPredicate&& p) {
        ensureCapacity(size_ + 1);
        pushBackUnchecked(std::move(p));
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    SubPredicate& operator[](std::size_t i) noexcept { return data()[i]; }
    const SubPredicate& operator[](std::size_t i) const noexcept { return data()[i]; }
    SubPredicate& at(std::size_t i) {
        if (i >= size_) throw std::out_of_range("pstree::PredicateList::at");
        return data()[i];
    }
    const SubPredicate& at(std::size_t i) const {
        if (i >= size_) throw std::out_of_range("pstree::PredicateList::at");
        return data()[i];
    }

    SubPredicate* begin() noexcept { return data(); }
    SubPredicate* end() noexcept { return data() + size_; }
    const SubPredicate* begin() const noexcept { return data(); }
    const SubPredicate* end() const noexcept { return data() + size_; }

private:
    // Raw, uninitialized storage for up to kInlineCapacity elements - placement-new'd and
    // manually destroyed exactly like std::vector's own heap buffer, just embedded directly
    // in the object instead of allocated, for the common (small) case.
    alignas(SubPredicate) unsigned char inline_[kInlineCapacity * sizeof(SubPredicate)];
    SubPredicate* heap_ = nullptr; // non-null once spilled to the heap; inline_ unused thereafter
    std::size_t heapCapacity_ = 0; // element capacity of *heap_; meaningless while heap_ == nullptr
    std::size_t size_ = 0;         // constructed-element count, in whichever storage is live

    SubPredicate* inlineData() noexcept { return reinterpret_cast<SubPredicate*>(inline_); }
    const SubPredicate* inlineData() const noexcept { return reinterpret_cast<const SubPredicate*>(inline_); }
    SubPredicate* data() noexcept { return heap_ != nullptr ? heap_ : inlineData(); }
    const SubPredicate* data() const noexcept { return heap_ != nullptr ? heap_ : inlineData(); }

    void reserveAtLeast(std::size_t n) {
        if (n > kInlineCapacity) growTo(n);
    }

    // Grows (or first establishes) heap storage to hold at least `minCapacity` elements,
    // move-constructing every already-constructed element from wherever it currently lives
    // (inline, or a smaller heap buffer) into the new buffer and destroying the old copies -
    // the same reallocation shape std::vector's own growth uses. Never shrinks; never called
    // with a `minCapacity` the current storage already satisfies.
    void growTo(std::size_t minCapacity) {
        std::size_t newCapacity = heap_ != nullptr ? heapCapacity_ : kInlineCapacity;
        if (newCapacity < 1) newCapacity = 1;
        while (newCapacity < minCapacity) newCapacity *= 2;
        auto* newBuf = static_cast<SubPredicate*>(::operator new(newCapacity * sizeof(SubPredicate)));
        SubPredicate* oldData = data();
        std::size_t constructedInNew = 0;
        try {
            for (; constructedInNew < size_; ++constructedInNew) {
                ::new (static_cast<void*>(newBuf + constructedInNew)) SubPredicate(std::move(oldData[constructedInNew]));
            }
        } catch (...) {
            for (std::size_t j = 0; j < constructedInNew; ++j) newBuf[j].~SubPredicate();
            ::operator delete(newBuf);
            throw;
        }
        for (std::size_t j = 0; j < size_; ++j) oldData[j].~SubPredicate();
        if (heap_ != nullptr) ::operator delete(heap_);
        heap_ = newBuf;
        heapCapacity_ = newCapacity;
    }

    void ensureCapacity(std::size_t needed) {
        std::size_t currentCapacity = heap_ != nullptr ? heapCapacity_ : kInlineCapacity;
        if (needed > currentCapacity) growTo(needed);
    }

    // Appends WITHOUT checking capacity - every caller (push_back, and the constructors
    // above via reserveAtLeast first) must have already ensured room for one more element.
    void pushBackUnchecked(const SubPredicate& p) {
        ::new (static_cast<void*>(data() + size_)) SubPredicate(p);
        ++size_;
    }
    void pushBackUnchecked(SubPredicate&& p) {
        ::new (static_cast<void*>(data() + size_)) SubPredicate(std::move(p));
        ++size_;
    }

    void clear() noexcept {
        SubPredicate* d = data();
        for (std::size_t i = 0; i < size_; ++i) d[i].~SubPredicate();
        size_ = 0;
    }
};

struct Subscription {
    std::uint64_t id;
    PredicateList predicates;
};

struct EventPair {
    std::string attr;
    Value val;
};
using Event = std::vector<EventPair>;

inline const Value* findAttr(const Event& event, std::string_view attr) {
    for (const auto& pair : event) {
        if (pair.attr == attr) return &pair.val;
    }
    return nullptr;
}

// Fills pred.primaryCache from pred.vals (assumed already sorted by Value's own operator<
// - see matchValue's kElemOf/kNotElemOf case) - extracting same-index elements in order
// preserves sortedness for the underlying type too, since std::variant's own ordering for two
// same-alternative values IS that alternative's own ordering. Empty `vals` leaves the cache as
// std::monostate deliberately (see elemOfTypedContains's own comment for why that's correct,
// not an oversight).
inline void buildElemOfTypedCache(const SubPredicate& pred) {
    if (pred.vals.empty()) return;
    switch (pred.vals.front().index()) {
        case 0: {
            std::vector<bool> v;
            v.reserve(pred.vals.size());
            for (const auto& x : pred.vals) v.push_back(std::get<bool>(x));
            pred.primaryCache = std::move(v);
            return;
        }
        case 1: {
            std::vector<std::int64_t> v;
            v.reserve(pred.vals.size());
            for (const auto& x : pred.vals) v.push_back(std::get<std::int64_t>(x));
            pred.primaryCache = std::move(v);
            return;
        }
        case 2: {
            std::vector<double> v;
            v.reserve(pred.vals.size());
            for (const auto& x : pred.vals) v.push_back(std::get<double>(x));
            pred.primaryCache = std::move(v);
            return;
        }
        default: {
            std::vector<std::string> v;
            v.reserve(pred.vals.size());
            for (const auto& x : pred.vals) v.push_back(std::get<std::string>(x));
            pred.primaryCache = std::move(v);
            return;
        }
    }
}

// Binary-searches pred.primaryCache using val's OWN concrete alternative - exactly one
// variant-dispatch (std::get<T>(val) below) per call, then a plain std::binary_search over a
// same-typed std::vector<T> that never touches std::variant again for any of its O(log n)
// comparisons. If val's type doesn't match what's cached (get_if returns nullptr) this
// silently returns false rather than throwing - identical to this codebase's own pre-existing
// behavior for a mismatched `val` on any call AFTER a kElemOf/kNotElemOf predicate's first
// (type-checked) sort, not a new gap introduced here (see matchValue's own checkSameType,
// which - both before and after this cache existed - only ever runs on that first call).
// std::monostate (an empty `vals` list) correctly matches nothing, for either case: the
// nullptr get_if of any of the four vector alternatives.
inline bool elemOfTypedContains(const SubPredicate& pred, const Value& val) {
    switch (val.index()) {
        case 0:
            if (auto* v = std::get_if<std::vector<bool>>(&pred.primaryCache)) {
                return std::binary_search(v->begin(), v->end(), std::get<bool>(val));
            }
            return false;
        case 1:
            if (auto* v = std::get_if<std::vector<std::int64_t>>(&pred.primaryCache)) {
                return std::binary_search(v->begin(), v->end(), std::get<std::int64_t>(val));
            }
            return false;
        case 2:
            if (auto* v = std::get_if<std::vector<double>>(&pred.primaryCache)) {
                return std::binary_search(v->begin(), v->end(), std::get<double>(val));
            }
            return false;
        default:
            if (auto* v = std::get_if<std::vector<std::string>>(&pred.primaryCache)) {
                return std::binary_search(v->begin(), v->end(), std::get<std::string>(val));
            }
            return false;
    }
}

// Extracts v's own concrete alternative into `cache` - works for either ScalarCache (kBetween's
// upper bound) or PrimaryCache (a scalar-family predicate's own vals[0]), since both variants
// carry bool/int64_t/double/string as alternatives; the assignment picks whichever alternative
// the destination variant actually has for that type.
template <typename CacheVariant>
inline void cacheScalarValue(const Value& v, CacheVariant& cache) {
    switch (v.index()) {
        case 0: cache = std::get<bool>(v); return;
        case 1: cache = std::get<std::int64_t>(v); return;
        case 2: cache = std::get<double>(v); return;
        default: cache = std::get<std::string>(v); return;
    }
}

// One-time cache build for the scalar comparison ops (kLt/kLe/kEq/kNe/kGt/kGe/kBetween) -
// mirrors matchValue's own kElemOf/kNotElemOf lazy-sort-then-cache pattern. vals[0] goes into
// the shared primaryCache (see SubPredicate's own comment for why it's shared with the elemOf
// path); kBetween's vals[1] is the only case that also needs betweenUpperCache, allocated here
// on first use.
inline void ensureScalarCached(const SubPredicate& pred) {
    if (pred.primaryCached) return;
    cacheScalarValue(pred.vals[0], pred.primaryCache);
    if (pred.op == CmpOp::kBetween) {
        pred.betweenUpperCache = std::make_unique<ScalarCache>();
        cacheScalarValue(pred.vals[1], *pred.betweenUpperCache);
    }
    pred.primaryCached = true;
}

// Compares val against an already-cached scalar threshold using `cmp` (std::less<>{},
// std::equal_to<>{}, etc.) - exactly one variant-dispatch (val's own index) per call, then a
// plain typed comparison, never invoking std::variant's own generic operator<=>. Measured as
// matchValue's single dominant self-time cost once trade_volume-style range predicates
// (kGe/kLe on a wide integer domain) were exercised at real scale: every scalar comparison
// here previously funneled through libstdc++'s synthesized variant<=>, which double-dispatches
// on BOTH operands even though a predicate's own threshold type never changes across calls.
// Same "move dispatch outside the hot path, don't try to make the dispatch itself cheaper"
// lesson as elemOfTypedContains's own history (two earlier attempts at a cheaper
// per-comparison dispatch, there, measured slower rather than faster). Templated on the cache's
// own variant type so it works for both primaryCache (a scalar-family predicate's vals[0]) and
// *betweenUpperCache (kBetween's vals[1]) without duplicating this switch.
template <typename Cmp, typename CacheVariant>
inline bool scalarCompare(const Value& val, const CacheVariant& cache, Cmp cmp) {
    switch (val.index()) {
        case 0: return cmp(std::get<bool>(val), std::get<bool>(cache));
        case 1: return cmp(std::get<std::int64_t>(val), std::get<std::int64_t>(cache));
        case 2: return cmp(std::get<double>(val), std::get<double>(cache));
        default: return cmp(std::get<std::string>(val), std::get<std::string>(cache));
    }
}

// Builds this predicate's ENTIRE comparison cache from its own literal values alone (no event
// value needed/used) - primaryCache/betweenUpperCache for scalar ops, the sorted-and-typed
// primaryCache for kElemOf/kNotElemOf. Callers: PSTDynamic::insertSubscription, exactly once per
// predicate, single-threaded, before the subscription becomes reachable by any concurrent
// matchEvent() call (see pst_dynamic.hpp's own comment on that call site for the real bug this
// fixes - a SIGSEGV in std::variant::_M_reset, root-caused to two worker threads racing to
// lazily build/reset this SAME cache the first time each independently reached matchValue for
// the same subscription. matching_engine::search() is explicitly documented as safe to call
// concurrently across worker threads - that promise was being broken by this lazy-build-on-
// first-match pattern, not upheld by it). matchValue() below now REQUIRES the cache already be
// built (throws otherwise) rather than lazily building it - a hot-path read is trivially safe to
// share across threads; a hot-path WRITE is exactly what was racing.
//
// The internal-consistency check below (every kElemOf/kNotElemOf literal shares one type)
// replaces the old lazy path's per-call checkSameType-against-the-incoming-event-value loop,
// which could only run against whichever event happened to trigger the first-ever match - this
// version checks the predicate's own literals against EACH OTHER, which is both the real
// invariant being protected and now checkable with no event in hand at all.
inline void ensurePredicateCachedForInsert(const SubPredicate& pred) {
    if (pred.primaryCached) return;
    switch (pred.op) {
        case CmpOp::kLt:
        case CmpOp::kLe:
        case CmpOp::kEq:
        case CmpOp::kNe:
        case CmpOp::kGt:
        case CmpOp::kGe:
        case CmpOp::kBetween:
            ensureScalarCached(pred);
            return;
        case CmpOp::kElemOf:
        case CmpOp::kNotElemOf: {
            if (!pred.vals.empty()) {
                std::size_t expectedIndex = pred.vals.front().index();
                for (const auto& v : pred.vals) {
                    if (v.index() != expectedIndex) {
                        throw std::invalid_argument(
                            "pstree: predicate on attribute '" + pred.attr +
                            "' has inconsistently-typed literal values");
                    }
                }
            }
            std::sort(pred.vals.begin(), pred.vals.end());
            buildElemOfTypedCache(pred);
            pred.primaryCached = true;
            return;
        }
        case CmpOp::kIsNull:
        case CmpOp::kIsNotNull:
            return; // no value-based cache - matchSubscription intercepts both before matchValue
    }
}

[[noreturn, gnu::cold, gnu::noinline]]
inline void throwPredicateCacheNotBuilt(const std::string& attr) {
    throw std::logic_error(
        "pstree: predicate cache for attribute '" + attr + "' was not built at insert time - "
        "ensurePredicateCachedForInsert must be called from PSTDynamic::insertSubscription "
        "before this subscription is reachable by matchEvent");
}

// Out-of-line and marked cold/noinline so the exception-construction code (a string
// concatenation, an allocation, the throw itself) never counts against the inliner's size
// estimate for whatever calls it. Found via `perf annotate` on the real trade_volume/trade_price
// benchmark (2026-08-30): matchValue's own checkSameType lambda - despite being one comparison
// on the hot path - was NOT being inlined at any of its call sites, because the compiler weighs
// a candidate's cold/never-taken paths against its size just like its hot ones. Disassembling
// the resulting out-of-line lambda showed ~57% of its cost was plain call/ret prologue-epilogue
// (4 callee-saved register spills) and ~40% was stack-protector canary setup/check - the actual
// comparison was ~2%. Extracting the throw into its own cold function shrinks checkSameType back
// down to the trivial size the inliner will actually fold into every one of matchValue's call
// sites, eliminating that call/canary overhead from the hot path entirely.
[[noreturn, gnu::cold, gnu::noinline]]
inline void throwMatchValueTypeMismatch(const std::string& attr) {
    throw std::invalid_argument("pstree: matchValue type mismatch for attribute '" + attr + "'");
}

// Evaluates one predicate against one concrete value. Throws if `val` and `pred`'s own
// value(s) aren't the same variant alternative - a schema/caller bug (mixing types for the
// same attribute), not a matching outcome, so it's surfaced loudly rather than silently
// answered via std::variant's own index-based ordering (which would produce a
// deterministic but semantically meaningless true/false for mismatched types).
//
// Never actually called for kIsNull/kIsNotNull - matchSubscription() below intercepts both
// before reaching here, since they're evaluated on ABSENCE, not on a concrete value at all
// (there's no `val` to hand this function when the attribute is absent in the first place).
// The cases exist only so this switch stays exhaustive; reaching them is a caller bug.
inline bool matchValue(const Value& val, const SubPredicate& pred) {
    auto checkSameType = [&](const Value& other) {
        if (val.index() != other.index()) {
            throwMatchValueTypeMismatch(pred.attr);
        }
    };
    // The cache is REQUIRED to already be built - see ensurePredicateCachedForInsert's own
    // comment for why this is a hard requirement (a thread-safety fix), not a convenience
    // check: lazily building it here, from whichever worker thread happens to reach it first,
    // is exactly the unsynchronized write that used to corrupt this predicate's cache under
    // concurrent matchEvent() calls.
    if (pred.op != CmpOp::kIsNull && pred.op != CmpOp::kIsNotNull && !pred.primaryCached) {
        throwPredicateCacheNotBuilt(pred.attr);
    }
    switch (pred.op) {
        case CmpOp::kLt:
            checkSameType(pred.vals.at(0));
            return scalarCompare(val, pred.primaryCache, std::less<>{});
        case CmpOp::kLe:
            checkSameType(pred.vals.at(0));
            return scalarCompare(val, pred.primaryCache, std::less_equal<>{});
        case CmpOp::kEq:
            checkSameType(pred.vals.at(0));
            return scalarCompare(val, pred.primaryCache, std::equal_to<>{});
        case CmpOp::kNe:
            checkSameType(pred.vals.at(0));
            return !scalarCompare(val, pred.primaryCache, std::equal_to<>{});
        case CmpOp::kGt:
            checkSameType(pred.vals.at(0));
            return scalarCompare(val, pred.primaryCache, std::greater<>{});
        case CmpOp::kGe:
            checkSameType(pred.vals.at(0));
            return scalarCompare(val, pred.primaryCache, std::greater_equal<>{});
        case CmpOp::kBetween:
            checkSameType(pred.vals.at(0));
            checkSameType(pred.vals.at(1));
            return scalarCompare(val, pred.primaryCache, std::greater_equal<>{}) &&
                   scalarCompare(val, *pred.betweenUpperCache, std::less_equal<>{});
        // Sorts pred.vals AND builds primaryCache (see both fields' own comments) instead
        // of a linear std::variant::operator== scan - real values lists here run up to ~128
        // elements (e.g. a `symbol in (...)` set-membership subscription). elemOfTypedContains
        // does the actual O(log n) comparisons on plain typed values, not std::variant ones -
        // both the sort and the cache build already happened at insert time (see
        // ensurePredicateCachedForInsert), so this is a pure read.
        case CmpOp::kElemOf:
            checkSameType(pred.vals.at(0));
            return elemOfTypedContains(pred, val);
        case CmpOp::kNotElemOf:
            checkSameType(pred.vals.at(0));
            return !elemOfTypedContains(pred, val);
        case CmpOp::kIsNull:
        case CmpOp::kIsNotNull:
            throw std::logic_error("pstree: kIsNull/kIsNotNull must be intercepted by matchSubscription, never reach matchValue");
    }
    return false;
}

// Section 2.2's matching semantics, transcribed directly: a subscription matches an event
// iff EVERY one of its predicates has a corresponding attribute-value pair in the event
// that satisfies it. A predicate whose attribute is simply absent from the event fails the
// subscription outright (the formal "P in S -> exists <attr,val> in E" implication is false
// when no such pair exists) - this is what correctly excludes, e.g., a subscription with a
// predicate on an attribute the event never mentions, even if the subscription's OTHER
// predicates would otherwise match.
inline bool matchSubscription(const Event& event, const Subscription& sub) {
    for (const auto& pred : sub.predicates) {
        const Value* val = findAttr(event, pred.attr);
        // kIsNull/kIsNotNull test presence itself, not a value comparison - intercepted
        // here, before the "absent attribute always fails" rule below would otherwise
        // incorrectly reject kIsNull for exactly the case it's meant to accept.
        if (pred.op == CmpOp::kIsNull) {
            if (val != nullptr) return false;
            continue;
        }
        if (pred.op == CmpOp::kIsNotNull) {
            if (val == nullptr) return false;
            continue;
        }
        if (val == nullptr) return false;
        if (!matchValue(*val, pred)) return false;
    }
    return true;
}

// Same matching semantics as matchSubscription() above (identical logic, byte for byte) but
// looks up each predicate's event value via O(1) array index (pred.attrIndex, resolved once at
// insert time - see SubPredicate's own comment) instead of findAttr()'s O(event size) name
// scan. `indexed` is a schema-sized array where indexed[i] is the Value for whichever event
// attribute has ordinal `i` (PSTDynamic::DimensionIndex::index), or nullptr if that attribute
// is absent from this particular event - built ONCE per matchEvent() call (see that function's
// own comment) and reused across every candidate subscription checked against that event,
// which is the whole point: findAttr()'s per-(predicate, candidate) cost was real at the call
// volume real subscription counts produce, even though each individual call was cheap.
//
// Deliberately a separate function, not a parameter-swapped overload of matchSubscription:
// this project's whole differential-testing discipline (test_pst_dynamic_stress.cpp's own
// brute-force oracle, in particular) depends on matchSubscription/findAttr/Event staying the
// simple, obviously-correct, schema-agnostic reference implementation, completely unchanged and
// untouched by this optimization - a real bug in matchSubscriptionIndexed shows up as a genuine
// mismatch against that untouched oracle, not something both sides could coincidentally share.
//
// pred.attrIndex may legitimately be kNoAttrIndex (this predicate's attribute isn't in the
// schema this indexed array was built from, or this predicate was never inserted through
// PSTDynamic at all) or >= indexed.size() - both correctly fall through to `val == nullptr`
// below via the bounds check, exactly mirroring findAttr() returning nullptr for an absent
// attribute.
inline bool matchSubscriptionIndexed(const std::vector<const Value*>& indexed, const Subscription& sub) {
    for (const auto& pred : sub.predicates) {
        const Value* val = (pred.attrIndex < indexed.size()) ? indexed[pred.attrIndex] : nullptr;
        if (pred.op == CmpOp::kIsNull) {
            if (val != nullptr) return false;
            continue;
        }
        if (pred.op == CmpOp::kIsNotNull) {
            if (val == nullptr) return false;
            continue;
        }
        if (val == nullptr) return false;
        if (!matchValue(*val, pred)) return false;
    }
    return true;
}

} // namespace pstree
