// PS-Tree correctness tests. Two of these are the paper's own worked examples,
// transcribed directly so this implementation's output is checked against the paper's
// own stated numbers, not just against itself:
//   - Section 4.4's dynamic-adjustment example (S1{age,in,[20,60]}, S2{age,in,[30,80]}) -
//     the ONE example in the paper given with exact, unambiguous round-number boundaries
//     and a full counter table, so it's trustworthy to pin a test to directly.
//   - A hand-verified example inspired by (not a literal transcription of) Fig. 1 - Fig.
//     1's own prose gives S1=[0,4) (exclusive 4) but then states the resulting space
//     (2,4] (inclusive 4) is associated with {S1,S2}, which is inconsistent unless S1's
//     upper bound actually is inclusive - rather than guess which side has the transcription
//     error, this test uses its own clean, hand-verified numbers exercising the identical
//     "two overlapping BETWEEN ranges -> three disjoint spaces" structure the figure
//     illustrates.
//
// ps_tree.hpp's own file-level comment explains the canonical-ancestor-marker redesign these
// tests check: a point can now be covered by SEVERAL buckets at once (an equality bucket plus
// any number of ancestor range markers along its path), not the single leaf the original
// leaf-chain design produced - `sumCoverage()` below sums predCounter across every bucket
// matchPoint() returns, which is the direct replacement for the old
// `matchPair(v)->predCounter` single-leaf read.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "pstree/order_key.hpp"
#include "pstree/ps_tree.hpp"

namespace {

int g_failures = 0;

void require(bool cond, const std::string& message) {
    if (!cond) {
        std::cerr << "FAIL: " << message << "\n";
        g_failures++;
    }
}

pstree::Predicate betweenInt(std::int64_t lo, std::int64_t hi) {
    pstree::Predicate p;
    p.op = pstree::Op::kIn;
    p.vals0 = pstree::Int64Codec::encode(lo);
    p.vals1 = pstree::Int64Codec::encode(hi);
    return p;
}

std::uint64_t sumCoverage(const pstree::PSTree& tree, std::int64_t v) {
    std::uint64_t total = 0;
    for (pstree::LeafNode* bucket : tree.matchPoint(pstree::Int64Codec::encode(v))) {
        total += bucket->predCounter;
    }
    return total;
}

// Section 4.4's worked example, transcribed directly - re-derived as a grid of sumCoverage()
// checks at each predicate space's own values, since the old leaf-chain "walk every leaf via
// next, collect predCounter" helper no longer applies (there is no single chain to walk
// anymore - see ps_tree.hpp's own file-level comment for what replaced it).
void test_section_4_4_dynamic_adjustment() {
    pstree::PSTree tree(pstree::Int64Codec::shape());

    auto s1Leaves = tree.insertPredicate(betweenInt(20, 60));
    require(!s1Leaves.empty(), "S1[20,60] alone should touch at least one bucket");

    auto s2Leaves = tree.insertPredicate(betweenInt(30, 80));
    require(!s2Leaves.empty(), "S2[30,80] should touch at least one bucket");

    // Paper: "the value domain is divided into 5 predicate spaces, [1,20), [20,30),
    // [30,60], (60,80], and (80,100], with counter as 0, 1, 2, 1, and 0, respectively."
    require(sumCoverage(tree, 10) == 0, "[1,20): covered by neither S1 nor S2");
    require(sumCoverage(tree, 25) == 1, "[20,30): covered by S1 only");
    require(sumCoverage(tree, 45) == 2, "[30,60]: covered by both S1 and S2");
    require(sumCoverage(tree, 70) == 1, "(60,80]: covered by S2 only");
    require(sumCoverage(tree, 90) == 0, "(80,100]: covered by neither");

    // "When S2 is removed, the counters of these predicate spaces become 0, 1, 1, 0, 0."
    tree.deletePredicate(betweenInt(30, 80));
    require(sumCoverage(tree, 10) == 0, "after deleting S2: [1,20) still 0");
    require(sumCoverage(tree, 25) == 1, "after deleting S2: [20,30) still S1 only");
    require(sumCoverage(tree, 45) == 1, "after deleting S2: [30,60] now S1 only");
    require(sumCoverage(tree, 70) == 0, "after deleting S2: (60,80] back to 0");
    require(sumCoverage(tree, 90) == 0, "after deleting S2: (80,100] still 0");

    // Removing S1 too should bring every sampled point back to zero coverage - "the PS-Tree
    // index will be recovered to a status as if the predicate was never inserted" (page 14
    // prose), the observable part of that guarantee (buckets themselves are not necessarily
    // freed - see ps_tree.hpp's own file-level comment on why that's deliberate).
    tree.deletePredicate(betweenInt(20, 60));
    for (std::int64_t v : {10, 25, 45, 70, 90}) {
        require(sumCoverage(tree, v) == 0, "every sampled point should be uncovered after removing both S1 and S2");
    }
}

// Hand-verified example in the same spirit as Fig. 1 (see file-level comment for why this
// isn't a literal transcription): S1 = price in [0,3], S2 = price in [2,5]. Overlap [2,3]
// -> three disjoint spaces: [0,1] only S1, [2,3] both, [4,5] only S2.
void test_matchpoint_membership_via_overlapping_ranges() {
    pstree::PSTree tree(pstree::Int64Codec::shape());
    tree.insertPredicate(betweenInt(0, 3)); // S1
    tree.insertPredicate(betweenInt(2, 5)); // S2

    require(sumCoverage(tree, 0) == 1, "price=0 should be covered by exactly S1");
    require(sumCoverage(tree, 1) == 1, "price=1 should be covered by exactly S1");
    require(sumCoverage(tree, 2) == 2, "price=2 should be covered by both S1 and S2");
    require(sumCoverage(tree, 3) == 2, "price=3 should be covered by both S1 and S2");
    require(sumCoverage(tree, 4) == 1, "price=4 should be covered by exactly S2");
    require(sumCoverage(tree, 5) == 1, "price=5 should be covered by exactly S2");
    require(sumCoverage(tree, -100) == 0, "price=-100 (never inserted) should be covered by nothing");
    require(sumCoverage(tree, 100) == 0, "price=100 (never inserted) should be covered by nothing");
}

// Single-predicate insert/match/delete round trip, per operator, per value type - a much
// smaller-scoped sanity check than the two examples above, aimed at catching a bug specific
// to one operator or one codec that a single combined example might not exercise.
void test_single_predicate_round_trips_int64() {
    {
        pstree::PSTree tree(pstree::Int64Codec::shape());
        pstree::Predicate p{pstree::Op::kGe, pstree::Int64Codec::encode(10), {}};
        tree.insertPredicate(p);
        require(sumCoverage(tree, 10) == 1, "kGe: boundary value included");
        require(sumCoverage(tree, 11) == 1, "kGe: value above boundary included");
        require(sumCoverage(tree, 9) == 0, "kGe: value below boundary excluded");
        tree.deletePredicate(p);
        require(sumCoverage(tree, 10) == 0, "kGe: deleted, boundary back to 0");
        require(sumCoverage(tree, 11) == 0, "kGe: deleted, above back to 0");
    }
    {
        pstree::PSTree tree(pstree::Int64Codec::shape());
        pstree::Predicate p{pstree::Op::kLe, pstree::Int64Codec::encode(10), {}};
        tree.insertPredicate(p);
        require(sumCoverage(tree, 10) == 1, "kLe: boundary value included");
        require(sumCoverage(tree, 9) == 1, "kLe: value below boundary included");
        require(sumCoverage(tree, 11) == 0, "kLe: value above boundary excluded");
        tree.deletePredicate(p);
        require(sumCoverage(tree, 10) == 0, "kLe: deleted, boundary back to 0");
    }
    {
        pstree::PSTree tree(pstree::Int64Codec::shape());
        pstree::Predicate p{pstree::Op::kEq, pstree::Int64Codec::encode(10), {}};
        tree.insertPredicate(p);
        require(sumCoverage(tree, 10) == 1, "kEq: exact value included");
        require(sumCoverage(tree, 9) == 0, "kEq: value below excluded");
        require(sumCoverage(tree, 11) == 0, "kEq: value above excluded");
        tree.deletePredicate(p);
        require(sumCoverage(tree, 10) == 0, "kEq: deleted, back to 0");
    }
}

void test_single_predicate_round_trips_double() {
    pstree::PSTree tree(pstree::DoubleCodec::shape());
    pstree::Predicate p{pstree::Op::kIn, pstree::DoubleCodec::encode(-1.5), pstree::DoubleCodec::encode(2.5)};
    tree.insertPredicate(p);
    auto sum = [&](double v) {
        std::uint64_t total = 0;
        for (auto* b : tree.matchPoint(pstree::DoubleCodec::encode(v))) total += b->predCounter;
        return total;
    };
    require(sum(-1.5) == 1, "double kIn: lower boundary included");
    require(sum(2.5) == 1, "double kIn: upper boundary included");
    require(sum(0.0) == 1, "double kIn: midpoint included");
    require(sum(-1.6) == 0, "double kIn: just below lower excluded");
    require(sum(2.6) == 0, "double kIn: just above upper excluded");
    tree.deletePredicate(p);
    require(sum(0.0) == 0, "double kIn: deleted, back to 0");
}

void test_single_predicate_round_trips_string() {
    pstree::StringCodec codec(16);
    pstree::PSTree tree(codec.shape());
    pstree::Predicate p{pstree::Op::kEq, codec.encode("hello"), {}};
    tree.insertPredicate(p);
    auto sum = [&](const char* s) {
        std::uint64_t total = 0;
        for (auto* b : tree.matchPoint(codec.encode(s))) total += b->predCounter;
        return total;
    };
    require(sum("hello") == 1, "string kEq: exact match included");
    require(sum("hellp") == 0, "string kEq: near-miss excluded");
    require(sum("") == 0, "string kEq: empty string excluded");
    tree.deletePredicate(p);
    require(sum("hello") == 0, "string kEq: deleted, back to 0");
}

void test_single_predicate_round_trips_bool() {
    pstree::PSTree tree(pstree::BoolCodec::shape());
    pstree::Predicate p{pstree::Op::kEq, pstree::BoolCodec::encode(true), {}};
    tree.insertPredicate(p);
    auto sum = [&](bool v) {
        std::uint64_t total = 0;
        for (auto* b : tree.matchPoint(pstree::BoolCodec::encode(v))) total += b->predCounter;
        return total;
    };
    require(sum(true) == 1, "bool kEq true: included");
    require(sum(false) == 0, "bool kEq true: false excluded");
    tree.deletePredicate(p);
    require(sum(true) == 0, "bool kEq true: deleted, back to 0");
}

} // namespace

int main() {
    test_section_4_4_dynamic_adjustment();
    test_matchpoint_membership_via_overlapping_ranges();
    test_single_predicate_round_trips_int64();
    test_single_predicate_round_trips_double();
    test_single_predicate_round_trips_string();
    test_single_predicate_round_trips_bool();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All ps_tree tests passed\n";
    return 0;
}
