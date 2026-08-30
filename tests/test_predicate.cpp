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

// Regression test: matchValue() lazily sorts a kElemOf/kNotElemOf predicate's vals in place
// (once) so it can binary_search instead of a linear scan (see SubPredicate's own field
// comment and matchValue()'s kElemOf/kNotElemOf case comment in predicate.hpp) - the OTHER
// test above happens to use an already-alphabetically-sorted input list ({a,b,c}), which
// would pass even with a broken sort/binary_search pairing. This one uses a deliberately
// scrambled, larger list (including a repeated value and mixed-magnitude integers, not just
// alphabetically-orderable strings) and calls matchValue() many times against the SAME
// SubPredicate object (mirroring the real reuse-across-many-events pattern) to also exercise
// the "already sorted from a prior call" branch, not just the first-call sort path.
void test_elem_of_with_unsorted_input_and_repeated_calls() {
    using pstree::CmpOp;
    auto p = pred("port", CmpOp::kElemOf,
                   {std::int64_t{80}, std::int64_t{22}, std::int64_t{443}, std::int64_t{22},
                    std::int64_t{8080}, std::int64_t{1}});
    require(pstree::matchValue(pstree::Value(std::int64_t{22}), p), "22 elem of scrambled {80,22,443,22,8080,1} (1st call)");
    require(pstree::matchValue(pstree::Value(std::int64_t{1}), p), "1 elem of scrambled list (2nd call, already sorted)");
    require(pstree::matchValue(pstree::Value(std::int64_t{8080}), p), "8080 elem of scrambled list (3rd call)");
    require(!pstree::matchValue(pstree::Value(std::int64_t{9999}), p), "9999 not elem of scrambled list");

    auto pNot = pred("port", CmpOp::kNotElemOf,
                      {std::int64_t{80}, std::int64_t{22}, std::int64_t{443}, std::int64_t{22},
                       std::int64_t{8080}, std::int64_t{1}});
    require(!pstree::matchValue(pstree::Value(std::int64_t{443}), pNot), "443 fails not-elem-of scrambled list");
    require(pstree::matchValue(pstree::Value(std::int64_t{9999}), pNot), "9999 satisfies not-elem-of scrambled list (repeat call)");
}

// Guards the scalarCache0/scalarCache1 lazy-cache mechanism added 2026-08-29 (see matchValue's
// own kLt/kLe/kEq/kNe/kGt/kGe cases) - reuses ONE kGe predicate (the exact shape a
// trade_volume-style "attr >= threshold" range predicate compiles down to) across several
// calls with DIFFERENT values, some on either side of the threshold and one exactly on it,
// so a stale or incorrectly-cached comparison would show up as a wrong result on a later
// call, not just the first (uncached) one.
void test_scalar_cache_with_repeated_calls() {
    using pstree::CmpOp;
    auto p = pred("volume", CmpOp::kGe, {std::int64_t{100}});
    require(!pstree::matchValue(pstree::Value(std::int64_t{50}), p), "50 !>= 100 (first call, builds cache)");
    require(pstree::matchValue(pstree::Value(std::int64_t{100}), p), "100 >= 100 (cached, on threshold)");
    require(pstree::matchValue(pstree::Value(std::int64_t{1000}), p), "1000 >= 100 (cached, well above)");
    require(!pstree::matchValue(pstree::Value(std::int64_t{99}), p), "99 !>= 100 (cached, just below)");

    auto p2 = pred("price", CmpOp::kLe, {2.5});
    require(pstree::matchValue(pstree::Value(1.0), p2), "1.0 <= 2.5 (first call, builds cache)");
    require(pstree::matchValue(pstree::Value(2.5), p2), "2.5 <= 2.5 (cached, on threshold)");
    require(!pstree::matchValue(pstree::Value(3.0), p2), "3.0 !<= 2.5 (cached, above)");
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

// kIsNull/kIsNotNull test presence itself, not a value - distinct from the "absent
// attribute always fails" rule above, which is exactly what these two are designed to
// override for their own attribute.
void test_is_null_and_is_not_null() {
    pstree::Subscription sub;
    sub.id = 1;
    sub.predicates.push_back(pred("discount", pstree::CmpOp::kIsNull, {}));

    require(pstree::matchSubscription({}, sub), "discount absent should satisfy 'discount is null'");
    pstree::Event withDiscount = {{"discount", pstree::Value(std::int64_t{5})}};
    require(!pstree::matchSubscription(withDiscount, sub), "discount present should fail 'discount is null'");

    pstree::Subscription notNullSub;
    notNullSub.id = 2;
    notNullSub.predicates.push_back(pred("discount", pstree::CmpOp::kIsNotNull, {}));
    require(pstree::matchSubscription(withDiscount, notNullSub), "discount present should satisfy 'discount is not null'");
    require(!pstree::matchSubscription({}, notNullSub), "discount absent should fail 'discount is not null'");

    // Combined with a normal predicate: "price>100 and discount is null" - the common real
    // pattern (an is-null check alongside at least one indexable predicate).
    pstree::Subscription combined;
    combined.id = 3;
    combined.predicates.push_back(pred("price", pstree::CmpOp::kGt, {std::int64_t{100}}));
    combined.predicates.push_back(pred("discount", pstree::CmpOp::kIsNull, {}));
    pstree::Event highPriceNoDiscount = {{"price", pstree::Value(std::int64_t{150})}};
    require(pstree::matchSubscription(highPriceNoDiscount, combined), "price>100 and discount absent should match");
    pstree::Event highPriceWithDiscount = {{"price", pstree::Value(std::int64_t{150})}, {"discount", pstree::Value(std::int64_t{5})}};
    require(!pstree::matchSubscription(highPriceWithDiscount, combined), "price>100 but discount present should not match");
}

} // namespace

int main() {
    test_relational_operators_int();
    test_between();
    test_scalar_cache_with_repeated_calls();
    test_elem_of_and_not_elem_of();
    test_elem_of_with_unsorted_input_and_repeated_calls();
    test_bool_and_double();
    test_type_mismatch_throws();
    test_missing_attribute_fails_subscription();
    test_is_null_and_is_not_null();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All predicate tests passed\n";
    return 0;
}
