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

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
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

struct Subscription {
    std::uint64_t id;
    std::vector<SubPredicate> predicates;
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
