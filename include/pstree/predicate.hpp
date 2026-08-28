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

#include <cstddef>
#include <cstdint>
#include <initializer_list>
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

struct SubPredicate {
    std::string attr;
    CmpOp op;
    std::vector<Value> vals; // size 1 for most ops, 2 for kBetween, >=1 for kElemOf/kNotElemOf
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
            throw std::invalid_argument("pstree: matchValue type mismatch for attribute '" + pred.attr + "'");
        }
    };
    switch (pred.op) {
        case CmpOp::kLt:
            checkSameType(pred.vals.at(0));
            return val < pred.vals[0];
        case CmpOp::kLe:
            checkSameType(pred.vals.at(0));
            return val <= pred.vals[0];
        case CmpOp::kEq:
            checkSameType(pred.vals.at(0));
            return val == pred.vals[0];
        case CmpOp::kNe:
            checkSameType(pred.vals.at(0));
            return !(val == pred.vals[0]);
        case CmpOp::kGt:
            checkSameType(pred.vals.at(0));
            return val > pred.vals[0];
        case CmpOp::kGe:
            checkSameType(pred.vals.at(0));
            return val >= pred.vals[0];
        case CmpOp::kBetween:
            checkSameType(pred.vals.at(0));
            checkSameType(pred.vals.at(1));
            return val >= pred.vals[0] && val <= pred.vals[1];
        case CmpOp::kElemOf:
            for (const auto& v : pred.vals) {
                checkSameType(v);
                if (val == v) return true;
            }
            return false;
        case CmpOp::kNotElemOf:
            for (const auto& v : pred.vals) {
                checkSameType(v);
                if (val == v) return false;
            }
            return true;
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

} // namespace pstree
