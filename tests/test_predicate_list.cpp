// Dedicated correctness tests for predicate.hpp's PredicateList - the small-vector-optimized
// container backing Subscription::predicates (see its own file-level comment for why it
// exists: removing a second, independently-heap-allocated cache-miss from MatchEvent's hot
// path). This type hand-rolls placement-new/manual-destroy bookkeeping (not just delegating
// to std::vector), which is exactly the kind of code that needs its OWN direct, exhaustive
// tests - not just indirect coverage via Subscription/PSTDynamic's higher-level tests - to
// catch bugs in growth, copy/move, and self-assignment specifically. Run under ASan+UBSan in
// CI, same as every other suite here, to catch double-frees/use-after-free/leaks directly.

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "pstree/predicate.hpp"

namespace {

int g_failures = 0;

void require(bool cond, const std::string& message) {
    if (!cond) {
        std::cerr << "FAIL: " << message << "\n";
        g_failures++;
    }
}

pstree::SubPredicate pred(std::string attr, std::int64_t v = 0) {
    return pstree::SubPredicate{std::move(attr), pstree::CmpOp::kEq, {v}};
}

void test_default_construction() {
    pstree::PredicateList list;
    require(list.empty(), "default-constructed list should be empty");
    require(list.size() == 0, "default-constructed list should have size 0");
}

// Pushes well past kInlineCapacity (4), checking size/order/contents after EVERY push - not
// just at the end - so a bug introduced exactly at the inline->heap spill point (the single
// most failure-prone transition in this kind of type) is caught precisely.
void test_push_back_growth_past_inline_capacity() {
    pstree::PredicateList list;
    constexpr int kCount = 12; // several times past kInlineCapacity=4
    for (int i = 0; i < kCount; ++i) {
        list.push_back(pred("attr" + std::to_string(i), i));
        require(list.size() == static_cast<std::size_t>(i + 1),
                "size should be " + std::to_string(i + 1) + " after " + std::to_string(i + 1) + " push_backs");
        for (int j = 0; j <= i; ++j) {
            require(list[static_cast<std::size_t>(j)].attr == "attr" + std::to_string(j),
                    "element " + std::to_string(j) + " should still be attr" + std::to_string(j) +
                        " after " + std::to_string(i + 1) + " total push_backs");
        }
    }
    require(list.size() == kCount, "final size should be " + std::to_string(kCount));
}

void test_construct_from_initializer_list() {
    {
        pstree::PredicateList small{pred("a"), pred("b")}; // under inline capacity
        require(small.size() == 2, "small initializer_list construction: size 2");
        require(small[0].attr == "a" && small[1].attr == "b", "small initializer_list: order preserved");
    }
    {
        pstree::PredicateList large{pred("a"), pred("b"), pred("c"), pred("d"), pred("e"), pred("f")}; // past capacity
        require(large.size() == 6, "large initializer_list construction: size 6");
        const char* expected[] = {"a", "b", "c", "d", "e", "f"};
        for (std::size_t i = 0; i < 6; ++i) {
            require(large[i].attr == expected[i], "large initializer_list: element " + std::to_string(i) + " correct");
        }
    }
}

void test_construct_from_vector() {
    std::vector<pstree::SubPredicate> v = {pred("x"), pred("y"), pred("z")};
    pstree::PredicateList fromLvalue(v);
    require(fromLvalue.size() == 3, "construct from vector lvalue: size 3");
    require(fromLvalue[0].attr == "x" && fromLvalue[1].attr == "y" && fromLvalue[2].attr == "z",
            "construct from vector lvalue: contents correct");
    // Original vector must be untouched by a const-lvalue-source construction.
    require(v.size() == 3 && v[0].attr == "x", "source vector unmodified after lvalue construction");

    std::vector<pstree::SubPredicate> v2 = {pred("p"), pred("q")};
    pstree::PredicateList fromRvalue(std::move(v2));
    require(fromRvalue.size() == 2, "construct from vector rvalue: size 2");
    require(fromRvalue[0].attr == "p" && fromRvalue[1].attr == "q", "construct from vector rvalue: contents correct");
}

void test_copy_independence_inline_and_heap() {
    // Inline-sized copy.
    pstree::PredicateList original{pred("a"), pred("b")};
    pstree::PredicateList copy = original;
    copy[0].attr = "mutated";
    require(original[0].attr == "a", "mutating an inline-sized copy must not affect the original");
    require(copy[0].attr == "mutated", "the copy itself should reflect the mutation");

    // Heap-spilled copy.
    pstree::PredicateList originalBig;
    for (int i = 0; i < 8; ++i) originalBig.push_back(pred("attr" + std::to_string(i), i));
    pstree::PredicateList copyBig = originalBig;
    require(copyBig.size() == originalBig.size(), "heap-spilled copy: size matches");
    copyBig[7].attr = "mutated";
    require(originalBig[7].attr == "attr7", "mutating a heap-spilled copy must not affect the original");
    for (std::size_t i = 0; i < 7; ++i) {
        require(copyBig[i].attr == originalBig[i].attr, "heap-spilled copy: untouched elements still match at " + std::to_string(i));
    }
}

void test_copy_assignment() {
    pstree::PredicateList a{pred("a1"), pred("a2"), pred("a3")};
    pstree::PredicateList b{pred("b1")};
    b = a;
    require(b.size() == 3, "copy-assignment: size updated to source's size");
    require(b[0].attr == "a1" && b[1].attr == "a2" && b[2].attr == "a3", "copy-assignment: contents match source");
    b[0].attr = "changed";
    require(a[0].attr == "a1", "copy-assignment: independence after assignment (mutating b doesn't affect a)");

    // Self-assignment must be a safe no-op, not a use-after-free/corruption.
    a = a;
    require(a.size() == 3 && a[0].attr == "a1" && a[1].attr == "a2" && a[2].attr == "a3",
            "self copy-assignment should leave contents unchanged");
}

void test_move_construction_and_assignment() {
    // Inline-sized move.
    pstree::PredicateList smallSrc{pred("x"), pred("y")};
    pstree::PredicateList smallDst = std::move(smallSrc);
    require(smallDst.size() == 2 && smallDst[0].attr == "x" && smallDst[1].attr == "y",
            "inline move-construction: destination has source's contents");
    require(smallSrc.size() == 0, "inline move-construction: source left empty");

    // Heap-spilled move (steal-the-pointer path).
    pstree::PredicateList bigSrc;
    for (int i = 0; i < 9; ++i) bigSrc.push_back(pred("attr" + std::to_string(i), i));
    pstree::PredicateList bigDst = std::move(bigSrc);
    require(bigDst.size() == 9, "heap move-construction: destination has all 9 elements");
    for (int i = 0; i < 9; ++i) {
        require(bigDst[static_cast<std::size_t>(i)].attr == "attr" + std::to_string(i),
                "heap move-construction: element " + std::to_string(i) + " correct");
    }
    require(bigSrc.size() == 0, "heap move-construction: source left empty");

    // Move-assignment onto a list that already owns its own heap buffer - must free its own
    // prior allocation, not leak it, before taking over the source's.
    pstree::PredicateList target;
    for (int i = 0; i < 6; ++i) target.push_back(pred("old" + std::to_string(i), i));
    pstree::PredicateList moveSrc;
    for (int i = 0; i < 7; ++i) moveSrc.push_back(pred("new" + std::to_string(i), i));
    target = std::move(moveSrc);
    require(target.size() == 7, "move-assignment onto an existing heap buffer: size updated");
    require(target[0].attr == "new0" && target[6].attr == "new6", "move-assignment: contents replaced correctly");
    require(moveSrc.size() == 0, "move-assignment: source left empty");

    // Self-move-assignment must be a safe no-op.
    pstree::PredicateList selfMove{pred("s1"), pred("s2")};
    selfMove = std::move(selfMove);
    require(selfMove.size() == 2 && selfMove[0].attr == "s1" && selfMove[1].attr == "s2",
            "self move-assignment should leave contents unchanged");
}

void test_at_bounds_checking() {
    pstree::PredicateList empty;
    bool threwOnEmpty = false;
    try {
        (void)empty.at(0);
    } catch (const std::out_of_range&) {
        threwOnEmpty = true;
    }
    require(threwOnEmpty, "at(0) on an empty list should throw std::out_of_range");

    pstree::PredicateList list{pred("only")};
    require(list.at(0).attr == "only", "at(0) on a valid index should return the right element");
    bool threwPastEnd = false;
    try {
        (void)list.at(1);
    } catch (const std::out_of_range&) {
        threwPastEnd = true;
    }
    require(threwPastEnd, "at(size()) should throw std::out_of_range");
}

void test_range_for_iteration_order() {
    pstree::PredicateList list;
    for (int i = 0; i < 10; ++i) list.push_back(pred("a" + std::to_string(i), i));
    std::size_t idx = 0;
    for (const auto& p : list) {
        require(p.attr == "a" + std::to_string(idx), "range-for element " + std::to_string(idx) + " in order");
        ++idx;
    }
    require(idx == 10, "range-for should visit exactly 10 elements");
}

// Directly exercises the two real call-site shapes found across pstree/nats_sidecar:
// aggregate-init from a braced SubPredicate list, and aggregate-init from an existing
// std::vector<SubPredicate> (nats_sidecar's own pstree_clause type).
void test_subscription_aggregate_init_compatibility() {
    pstree::Subscription fromBraces{7, {pred("a"), pred("b")}};
    require(fromBraces.id == 7 && fromBraces.predicates.size() == 2 && fromBraces.predicates[0].attr == "a",
            "Subscription aggregate-init from a braced predicate list should work");

    std::vector<pstree::SubPredicate> clause = {pred("x"), pred("y"), pred("z")};
    pstree::Subscription fromVector{8, clause};
    require(fromVector.id == 8 && fromVector.predicates.size() == 3 && fromVector.predicates[2].attr == "z",
            "Subscription aggregate-init from an existing std::vector<SubPredicate> should work");

    // The assignment-from-braces pattern used by test_pst_dynamic.cpp's fig3Subscriptions().
    std::vector<pstree::Subscription> subs(2);
    subs[0] = {9, {pred("p"), pred("q")}};
    require(subs[0].id == 9 && subs[0].predicates.size() == 2 && subs[0].predicates[1].attr == "q",
            "Subscription assignment from a braced aggregate should work");
}

// Interleaved push_back/copy/move cycles - not targeting one specific transition, just
// broad coverage under ASan/UBSan for anything the more targeted tests above didn't happen
// to hit.
void test_mixed_operations_stress() {
    std::vector<pstree::PredicateList> lists;
    for (int round = 0; round < 20; ++round) {
        pstree::PredicateList list;
        int count = round % 7; // sweeps across and past kInlineCapacity=4 repeatedly
        for (int i = 0; i < count; ++i) list.push_back(pred("r" + std::to_string(round) + "_" + std::to_string(i), i));
        require(list.size() == static_cast<std::size_t>(count), "stress round " + std::to_string(round) + ": size correct");

        if (round % 3 == 0) {
            pstree::PredicateList copy = list; // copy ctor
            lists.push_back(std::move(copy));  // then move into the vector
        } else if (round % 3 == 1) {
            lists.push_back(list); // copy directly into the vector
        } else {
            lists.push_back(std::move(list)); // move directly
        }
    }
    require(lists.size() == 20, "stress: all 20 rounds landed in the vector");
    for (std::size_t round = 0; round < lists.size(); ++round) {
        std::size_t expectedCount = round % 7;
        require(lists[round].size() == expectedCount,
                "stress: stored list " + std::to_string(round) + " has expected size after copy/move");
        for (std::size_t i = 0; i < expectedCount; ++i) {
            require(lists[round][i].attr == "r" + std::to_string(round) + "_" + std::to_string(i),
                    "stress: stored list " + std::to_string(round) + " element " + std::to_string(i) + " intact");
        }
    }
    // std::vector<PredicateList> itself growing/reallocating exercises PredicateList's own
    // move constructor being called by ITS OWNER's reallocation, not just directly here -
    // real coverage for the noexcept move path std::vector relies on to avoid falling back
    // to copies during its own growth.
}

} // namespace

int main() {
    test_default_construction();
    test_push_back_growth_past_inline_capacity();
    test_construct_from_initializer_list();
    test_construct_from_vector();
    test_copy_independence_inline_and_heap();
    test_copy_assignment();
    test_move_construction_and_assignment();
    test_at_bounds_checking();
    test_range_for_iteration_order();
    test_subscription_aggregate_init_compatibility();
    test_mixed_operations_stress();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All PredicateList tests passed\n";
    return 0;
}
