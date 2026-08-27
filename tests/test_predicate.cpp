// Correctness tests for predicate.hpp's Match() evaluator: every operator in Section 2.1's
// language, across value types, plus the "missing attribute fails the subscription" rule
// from Section 2.2's formal matching semantics (a predicate whose attribute the event
// simply doesn't mention can never be satisfied, regardless of the subscription's other
// predicates).

#include <iostream>
#include <string>

#include "pstree/predicate.hpp"

namespace {

int g_failures = 0;

void require(bool cond, const std::string& message) {
    if (!cond) {
        std::cerr << "FAIL: " << message << "\n";
        g_failures++;
    }
}

pstree::SubPredicate pred(std::string attr, pstree::CmpOp op, std::vector<pstree::Value> vals) {
    return pstree::SubPredicate{std::move(attr), op, std::move(vals)};
}

void test_relational_operators_int() {
    using pstree::CmpOp;
    require(pstree::matchValue(pstree::Value(std::int64_t{5}), pred("x", CmpOp::kLt, {std::int64_t{10}})), "5 < 10");
    require(!pstree::matchValue(pstree::Value(std::int64_t{10}), pred("x", CmpOp::kLt, {std::int64_t{10}})), "10 !< 10");
    require(pstree::matchValue(pstree::Value(std::int64_t{10}), pred("x", CmpOp::kLe, {std::int64_t{10}})), "10 <= 10");
    require(pstree::matchValue(pstree::Value(std::int64_t{10}), pred("x", CmpOp::kEq, {std::int64_t{10}})), "10 == 10");
    require(!pstree::matchValue(pstree::Value(std::int64_t{9}), pred("x", CmpOp::kEq, {std::int64_t{10}})), "9 != 10 (kEq false)");
    require(pstree::matchValue(pstree::Value(std::int64_t{9}), pred("x", CmpOp::kNe, {std::int64_t{10}})), "9 != 10 (kNe true)");
    require(!pstree::matchValue(pstree::Value(std::int64_t{10}), pred("x", CmpOp::kNe, {std::int64_t{10}})), "10 == 10 (kNe false)");
    require(pstree::matchValue(pstree::Value(std::int64_t{11}), pred("x", CmpOp::kGt, {std::int64_t{10}})), "11 > 10");
    require(pstree::matchValue(pstree::Value(std::int64_t{10}), pred("x", CmpOp::kGe, {std::int64_t{10}})), "10 >= 10");
}

void test_between() {
    using pstree::CmpOp;
    auto p = pred("x", CmpOp::kBetween, {std::int64_t{1}, std::int64_t{5}});
    require(pstree::matchValue(pstree::Value(std::int64_t{1}), p), "1 in [1,5]");
    require(pstree::matchValue(pstree::Value(std::int64_t{5}), p), "5 in [1,5]");
    require(pstree::matchValue(pstree::Value(std::int64_t{3}), p), "3 in [1,5]");
    require(!pstree::matchValue(pstree::Value(std::int64_t{0}), p), "0 not in [1,5]");
    require(!pstree::matchValue(pstree::Value(std::int64_t{6}), p), "6 not in [1,5]");
}

void test_elem_of_and_not_elem_of() {
    using pstree::CmpOp;
    auto p = pred("x", CmpOp::kElemOf, {std::string("a"), std::string("b"), std::string("c")});
    require(pstree::matchValue(pstree::Value(std::string("b")), p), "'b' elem of {a,b,c}");
    require(!pstree::matchValue(pstree::Value(std::string("z")), p), "'z' not elem of {a,b,c}");

    auto pNot = pred("x", CmpOp::kNotElemOf, {std::string("a"), std::string("b"), std::string("c")});
    require(!pstree::matchValue(pstree::Value(std::string("b")), pNot), "'b' fails not-elem-of {a,b,c}");
    require(pstree::matchValue(pstree::Value(std::string("z")), pNot), "'z' satisfies not-elem-of {a,b,c}");
}

void test_bool_and_double() {
    using pstree::CmpOp;
    require(pstree::matchValue(pstree::Value(true), pred("flag", CmpOp::kEq, {true})), "true == true");
    require(!pstree::matchValue(pstree::Value(false), pred("flag", CmpOp::kEq, {true})), "false != true");
    require(pstree::matchValue(pstree::Value(2.5), pred("x", CmpOp::kGt, {1.0})), "2.5 > 1.0");
    require(pstree::matchValue(pstree::Value(-0.0), pred("x", CmpOp::kEq, {0.0})), "-0.0 == 0.0");
}

void test_type_mismatch_throws() {
    using pstree::CmpOp;
    bool threw = false;
    try {
        pstree::matchValue(pstree::Value(std::int64_t{5}), pred("x", CmpOp::kEq, {std::string("5")}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "comparing an int64 event value against a string predicate value should throw");
}

void test_missing_attribute_fails_subscription() {
    pstree::Subscription sub;
    sub.id = 1;
    sub.predicates.push_back(pred("attr1", pstree::CmpOp::kLt, {std::int64_t{0}}));
    sub.predicates.push_back(pred("attr2", pstree::CmpOp::kBetween, {std::int64_t{1}, std::int64_t{5}}));

    // Event has attr2 (satisfying) but not attr1 at all.
    pstree::Event event = {{"attr2", pstree::Value(std::int64_t{2})}};
    require(!pstree::matchSubscription(event, sub), "subscription requiring an absent attribute must not match");

    // Same subscription, now with attr1 present and satisfying too - should match.
    pstree::Event fullEvent = {{"attr1", pstree::Value(std::int64_t{-1})}, {"attr2", pstree::Value(std::int64_t{2})}};
    require(pstree::matchSubscription(fullEvent, sub), "subscription should match once every predicate's attribute is present and satisfied");

    // Extra, irrelevant attributes in the event should not affect the outcome.
    pstree::Event extraEvent = fullEvent;
    extraEvent.push_back({"unrelated", pstree::Value(std::string("noise"))});
    require(pstree::matchSubscription(extraEvent, sub), "extra unrelated event attributes should not block a match");
}

} // namespace

int main() {
    test_relational_operators_int();
    test_between();
    test_elem_of_and_not_elem_of();
    test_bool_and_double();
    test_type_mismatch_throws();
    test_missing_attribute_fails_subscription();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All predicate tests passed\n";
    return 0;
}
