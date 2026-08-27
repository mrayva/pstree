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

// Walks every leaf via `next` starting from the tree's first leaf (root->leaf_g),
// collecting predCounter values in order - used to check the whole predicate-space
// structure matches the paper's stated numbers, not just individual query results.
std::vector<std::uint64_t> allCounters(const pstree::PSTree& tree) {
    std::vector<std::uint64_t> out;
    for (pstree::LeafNode* leaf = tree.root()->leaf_g; leaf != nullptr; leaf = leaf->next) {
        out.push_back(leaf->predCounter);
    }
    return out;
}

// Section 4.4's worked example, transcribed directly.
void test_section_4_4_dynamic_adjustment() {
    pstree::PSTree tree(pstree::Int64Codec::shape());

    auto s1Leaves = tree.insertPredicate(betweenInt(20, 60));
    require(s1Leaves.size() == 1, "S1[20,60] alone should cover exactly 1 leaf (whole domain undivided at insert time)");

    auto s2Leaves = tree.insertPredicate(betweenInt(30, 80));
    require(s2Leaves.size() == 2, "S2[30,80] should cover exactly 2 leaves ([30,60] and (60,80])");

    // Paper: "the value domain is divided into 5 predicate spaces, [1,20), [20,30),
    // [30,60], (60,80], and (80,100], with counter as 0, 1, 2, 1, and 0, respectively."
    // This tree has no explicit [1,100] domain bound inserted, so its own leftmost/
    // rightmost leaves are unbounded rather than clipped at 1/100 - counters are identical
    // either way since bounding the domain doesn't change any predicate's own coverage.
    std::vector<std::uint64_t> expectedAfterInsert = {0, 1, 2, 1, 0};
    require(allCounters(tree) == expectedAfterInsert, "counters after inserting S1 and S2 should be [0,1,2,1,0]");

    // "When S2 is removed, the counters of these predicate spaces become 0, 1, 1, 0, 0."
    tree.deletePredicate(betweenInt(30, 80));
    std::vector<std::uint64_t> expectedAfterDelete = {0, 1, 1, 0, 0};
    require(allCounters(tree) == expectedAfterDelete,
            "counters after deleting S2 should be [0,1,1,0,0] (merging is deferred - see ps_tree.hpp)");

    // Removing S1 too should bring every leaf back to zero - "the PS-Tree index will be
    // recovered to a status as if the predicate was never inserted" (page 14 prose),
    // which this test reads as "every leaf's counter returns to 0", the part of that
    // guarantee this implementation actually provides (structure isn't collapsed back to
    // a single leaf since merging is deferred, but coverage is correctly all-zero).
    tree.deletePredicate(betweenInt(20, 60));
    for (auto c : allCounters(tree)) {
        require(c == 0, "every leaf's counter should be 0 after removing both S1 and S2");
    }
}

// Hand-verified example in the same spirit as Fig. 1 (see file-level comment for why this
// isn't a literal transcription): S1 = price in [0,3], S2 = price in [2,5]. Overlap [2,3]
// -> three disjoint spaces: [0,1] only S1, [2,3] both, [4,5] only S2. Checked via matchPair
// end-to-end (not just counters), since this is the query path real callers depend on.
void test_matchpair_membership_via_overlapping_ranges() {
    pstree::PSTree tree(pstree::Int64Codec::shape());
    tree.insertPredicate(betweenInt(0, 3)); // S1
    tree.insertPredicate(betweenInt(2, 5)); // S2

    auto counterAt = [&](std::int64_t v) {
        return tree.matchPair(pstree::Int64Codec::encode(v))->predCounter;
    };

    require(counterAt(0) == 1, "price=0 should be covered by exactly S1");
    require(counterAt(1) == 1, "price=1 should be covered by exactly S1");
    require(counterAt(2) == 2, "price=2 should be covered by both S1 and S2");
    require(counterAt(3) == 2, "price=3 should be covered by both S1 and S2");
    require(counterAt(4) == 1, "price=4 should be covered by exactly S2");
    require(counterAt(5) == 1, "price=5 should be covered by exactly S2");
    require(counterAt(-100) == 0, "price=-100 (never inserted) should be covered by nothing");
    require(counterAt(100) == 0, "price=100 (never inserted) should be covered by nothing");

    // Same leaf pointer identity for two values inside the same predicate space.
    auto leafAt2 = tree.matchPair(pstree::Int64Codec::encode(2));
    auto leafAt3 = tree.matchPair(pstree::Int64Codec::encode(3));
    require(leafAt2 == leafAt3, "price=2 and price=3 should resolve to the SAME leaf (same predicate space)");
    auto leafAt0 = tree.matchPair(pstree::Int64Codec::encode(0));
    require(leafAt0 != leafAt2, "price=0 and price=2 should resolve to DIFFERENT leaves");
}

// Single-predicate insert/match/delete round trip, per operator, per value type - a much
// smaller-scoped sanity check than the two examples above, aimed at catching a bug
// specific to one operator or one codec that a single combined example might not exercise.
void test_single_predicate_round_trips_int64() {
    {
        pstree::PSTree tree(pstree::Int64Codec::shape());
        pstree::Predicate p{pstree::Op::kGe, pstree::Int64Codec::encode(10), {}};
        tree.insertPredicate(p);
        require(tree.matchPair(pstree::Int64Codec::encode(10))->predCounter == 1, "kGe: boundary value included");
        require(tree.matchPair(pstree::Int64Codec::encode(11))->predCounter == 1, "kGe: value above boundary included");
        require(tree.matchPair(pstree::Int64Codec::encode(9))->predCounter == 0, "kGe: value below boundary excluded");
        tree.deletePredicate(p);
        require(tree.matchPair(pstree::Int64Codec::encode(10))->predCounter == 0, "kGe: deleted, boundary back to 0");
        require(tree.matchPair(pstree::Int64Codec::encode(11))->predCounter == 0, "kGe: deleted, above back to 0");
    }
    {
        pstree::PSTree tree(pstree::Int64Codec::shape());
        pstree::Predicate p{pstree::Op::kLe, pstree::Int64Codec::encode(10), {}};
        tree.insertPredicate(p);
        require(tree.matchPair(pstree::Int64Codec::encode(10))->predCounter == 1, "kLe: boundary value included");
        require(tree.matchPair(pstree::Int64Codec::encode(9))->predCounter == 1, "kLe: value below boundary included");
        require(tree.matchPair(pstree::Int64Codec::encode(11))->predCounter == 0, "kLe: value above boundary excluded");
        tree.deletePredicate(p);
        require(tree.matchPair(pstree::Int64Codec::encode(10))->predCounter == 0, "kLe: deleted, boundary back to 0");
    }
    {
        pstree::PSTree tree(pstree::Int64Codec::shape());
        pstree::Predicate p{pstree::Op::kEq, pstree::Int64Codec::encode(10), {}};
        tree.insertPredicate(p);
        require(tree.matchPair(pstree::Int64Codec::encode(10))->predCounter == 1, "kEq: exact value included");
        require(tree.matchPair(pstree::Int64Codec::encode(9))->predCounter == 0, "kEq: value below excluded");
        require(tree.matchPair(pstree::Int64Codec::encode(11))->predCounter == 0, "kEq: value above excluded");
        tree.deletePredicate(p);
        require(tree.matchPair(pstree::Int64Codec::encode(10))->predCounter == 0, "kEq: deleted, back to 0");
    }
}

void test_single_predicate_round_trips_double() {
    pstree::PSTree tree(pstree::DoubleCodec::shape());
    pstree::Predicate p{pstree::Op::kIn, pstree::DoubleCodec::encode(-1.5), pstree::DoubleCodec::encode(2.5)};
    tree.insertPredicate(p);
    require(tree.matchPair(pstree::DoubleCodec::encode(-1.5))->predCounter == 1, "double kIn: lower boundary included");
    require(tree.matchPair(pstree::DoubleCodec::encode(2.5))->predCounter == 1, "double kIn: upper boundary included");
    require(tree.matchPair(pstree::DoubleCodec::encode(0.0))->predCounter == 1, "double kIn: midpoint included");
    require(tree.matchPair(pstree::DoubleCodec::encode(-1.6))->predCounter == 0, "double kIn: just below lower excluded");
    require(tree.matchPair(pstree::DoubleCodec::encode(2.6))->predCounter == 0, "double kIn: just above upper excluded");
    tree.deletePredicate(p);
    require(tree.matchPair(pstree::DoubleCodec::encode(0.0))->predCounter == 0, "double kIn: deleted, back to 0");
}

void test_single_predicate_round_trips_string() {
    pstree::StringCodec codec(16);
    pstree::PSTree tree(codec.shape());
    pstree::Predicate p{pstree::Op::kEq, codec.encode("hello"), {}};
    tree.insertPredicate(p);
    require(tree.matchPair(codec.encode("hello"))->predCounter == 1, "string kEq: exact match included");
    require(tree.matchPair(codec.encode("hellp"))->predCounter == 0, "string kEq: near-miss excluded");
    require(tree.matchPair(codec.encode(""))->predCounter == 0, "string kEq: empty string excluded");
    tree.deletePredicate(p);
    require(tree.matchPair(codec.encode("hello"))->predCounter == 0, "string kEq: deleted, back to 0");
}

void test_single_predicate_round_trips_bool() {
    pstree::PSTree tree(pstree::BoolCodec::shape());
    pstree::Predicate p{pstree::Op::kEq, pstree::BoolCodec::encode(true), {}};
    tree.insertPredicate(p);
    require(tree.matchPair(pstree::BoolCodec::encode(true))->predCounter == 1, "bool kEq true: included");
    require(tree.matchPair(pstree::BoolCodec::encode(false))->predCounter == 0, "bool kEq true: false excluded");
    tree.deletePredicate(p);
    require(tree.matchPair(pstree::BoolCodec::encode(true))->predCounter == 0, "bool kEq true: deleted, back to 0");
}

} // namespace

int main() {
    test_section_4_4_dynamic_adjustment();
    test_matchpair_membership_via_overlapping_ranges();
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
