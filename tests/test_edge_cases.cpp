// Edge-case hardening for PS-Tree, beyond the core correctness tests in test_ps_tree.cpp:
// strict >/< (kGt/kLt), domain-boundary overflow/underflow, duplicate and double-delete
// handling, repeated insert/delete churn (leak/UAF exposure under ASan), and multi-way (3+)
// overlapping ranges. `sumCoverage()` sums predCounter across every bucket matchPoint()
// returns for a value - the canonical-ancestor-marker redesign's replacement for the old
// single-leaf `matchPair(v)->predCounter` read (see ps_tree.hpp's own file-level comment).

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

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

pstree::Predicate simple(pstree::Op op, std::int64_t v) {
    return pstree::Predicate{op, pstree::Int64Codec::encode(v), {}};
}

std::uint64_t sumCoverage(const pstree::PSTree& tree, std::int64_t v) {
    std::uint64_t total = 0;
    for (pstree::LeafNode* bucket : tree.matchPoint(pstree::Int64Codec::encode(v))) {
        total += bucket->predCounter;
    }
    return total;
}

void test_strict_gt_lt_int64() {
    pstree::PSTree tree(pstree::Int64Codec::shape());
    auto pGt = simple(pstree::Op::kGt, 10);
    tree.insertPredicate(pGt);
    require(sumCoverage(tree, 11) == 1, "kGt: value above boundary included");
    require(sumCoverage(tree, 10) == 0, "kGt: boundary value itself excluded");
    require(sumCoverage(tree, 9) == 0, "kGt: value below boundary excluded");
    tree.deletePredicate(pGt);
    require(sumCoverage(tree, 11) == 0, "kGt: deleted, back to 0");

    auto pLt = simple(pstree::Op::kLt, 10);
    tree.insertPredicate(pLt);
    require(sumCoverage(tree, 9) == 1, "kLt: value below boundary included");
    require(sumCoverage(tree, 10) == 0, "kLt: boundary value itself excluded");
    require(sumCoverage(tree, 11) == 0, "kLt: value above boundary excluded");
    tree.deletePredicate(pLt);
    require(sumCoverage(tree, 9) == 0, "kLt: deleted, back to 0");
}

// kGt/kLt at adjacent integers should compose correctly with each other - ">9" covers
// [10,+inf), "<11" covers (-inf,10], so value 10 is covered by both, 9 only by "<11", and
// 11 only by ">9" - a targeted check that the kGt->kGe(next) / kLt->kLe(prev) normalization
// composes correctly across two independently-inserted predicates, not just one.
void test_strict_gt_lt_adjacent_composition() {
    pstree::PSTree tree(pstree::Int64Codec::shape());
    tree.insertPredicate(simple(pstree::Op::kGt, 9));  // covers >= 10
    tree.insertPredicate(simple(pstree::Op::kLt, 11)); // covers <= 10
    require(sumCoverage(tree, 10) == 2, "value 10 covered by both >9 and <11");
    require(sumCoverage(tree, 9) == 1, "value 9 covered by <11 only");
    require(sumCoverage(tree, 11) == 1, "value 11 covered by >9 only");
}

// A kGt/kLt at the extreme representable value has no adjacent key to normalize to, and
// per order_key.hpp's nextElementKey/prevElementKey contract, correctly matches nothing -
// insertPredicate/deletePredicate must both handle this without touching the tree.
void test_strict_gt_lt_domain_edge() {
    pstree::PSTree tree(pstree::Int64Codec::shape());
    auto pGtMax = simple(pstree::Op::kGt, std::numeric_limits<std::int64_t>::max());
    auto inserted = tree.insertPredicate(pGtMax);
    require(inserted.empty(), "kGt at INT64_MAX should touch zero buckets (nothing is greater)");
    require(sumCoverage(tree, std::numeric_limits<std::int64_t>::max()) == 0,
            "kGt at INT64_MAX: the max value itself still uncovered");
    // Deleting the same never-actually-inserted predicate must also be a safe no-op, not
    // an underflow - it recomputes the identical "no adjacent key" result deterministically.
    auto deleted = tree.deletePredicate(pGtMax);
    require(deleted.empty(), "deleting a kGt-at-max predicate should also touch zero buckets");

    auto pLtMin = simple(pstree::Op::kLt, std::numeric_limits<std::int64_t>::min());
    require(tree.insertPredicate(pLtMin).empty(), "kLt at INT64_MIN should touch zero buckets (nothing is smaller)");
    require(tree.deletePredicate(pLtMin).empty(), "deleting a kLt-at-min predicate should also touch zero buckets");
}

// Inserting the exact same predicate twice should double-count it (predCounter reaches 2),
// and each delete should undo exactly one of the two insertions.
void test_duplicate_insert_delete() {
    pstree::PSTree tree(pstree::Int64Codec::shape());
    auto p = simple(pstree::Op::kEq, 42);
    tree.insertPredicate(p);
    tree.insertPredicate(p);
    require(sumCoverage(tree, 42) == 2, "duplicate insert should reach counter 2");
    tree.deletePredicate(p);
    require(sumCoverage(tree, 42) == 1, "one delete should bring counter back to 1");
    tree.deletePredicate(p);
    require(sumCoverage(tree, 42) == 0, "second delete should bring counter back to 0");
}

// Deleting a predicate that was never inserted (or deleting one twice) must fail loudly,
// not silently wrap predCounter (an unsigned integer) to a huge value - see
// deletePredicate's own doc comment for why this is checked rather than left as UB-by-
// convention the way a plain C library might.
void test_delete_underflow_throws() {
    pstree::PSTree tree(pstree::Int64Codec::shape());
    bool threw = false;
    try {
        tree.deletePredicate(simple(pstree::Op::kEq, 5));
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "deleting a never-inserted predicate should throw std::logic_error");

    // Double-delete: insert once, delete once (fine), delete again (should throw).
    pstree::PSTree tree2(pstree::Int64Codec::shape());
    auto p = simple(pstree::Op::kGe, 100);
    tree2.insertPredicate(p);
    tree2.deletePredicate(p);
    threw = false;
    try {
        tree2.deletePredicate(p);
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "double-deleting a predicate should throw std::logic_error");
}

// Many insert/delete cycles of overlapping ranges at the same points, run under a normal
// build here and (separately, via CI/build-asan) under ASan+UBSan with leak detection - this
// is the test most likely to expose a use-after-free or leak in the new marker-bucket
// insert/delete paths, since it repeatedly creates the same region of tree structure many
// times over (buckets are deliberately never freed on decrement - see ps_tree.hpp's own
// file-level comment on why - so this also exercises that zero-counter "zombie" buckets stay
// harmless across many cycles, not just one).
void test_repeated_churn() {
    pstree::PSTree tree(pstree::Int64Codec::shape());
    for (int cycle = 0; cycle < 200; ++cycle) {
        auto a = simple(pstree::Op::kIn, 0);
        a.vals1 = pstree::Int64Codec::encode(50);
        auto b = simple(pstree::Op::kIn, 25);
        b.vals1 = pstree::Int64Codec::encode(75);
        tree.insertPredicate(a);
        tree.insertPredicate(b);
        require(sumCoverage(tree, 30) == 2, "churn cycle: overlap region covered twice");
        tree.deletePredicate(b);
        tree.deletePredicate(a);
        require(sumCoverage(tree, 30) == 0, "churn cycle: back to zero after both deletes");
    }
}

// Three overlapping ranges (not just two) stressing multiple simultaneous canonical
// decompositions sharing common ancestor nodes.
void test_three_way_overlap() {
    pstree::PSTree tree(pstree::Int64Codec::shape());
    auto a = simple(pstree::Op::kIn, 0);
    a.vals1 = pstree::Int64Codec::encode(30); // [0,30]
    auto b = simple(pstree::Op::kIn, 10);
    b.vals1 = pstree::Int64Codec::encode(40); // [10,40]
    auto c = simple(pstree::Op::kIn, 20);
    c.vals1 = pstree::Int64Codec::encode(50); // [20,50]
    tree.insertPredicate(a);
    tree.insertPredicate(b);
    tree.insertPredicate(c);

    require(sumCoverage(tree, 5) == 1, "v=5 covered by A only");
    require(sumCoverage(tree, 15) == 2, "v=15 covered by A and B");
    require(sumCoverage(tree, 25) == 3, "v=25 covered by A, B, and C");
    require(sumCoverage(tree, 35) == 2, "v=35 covered by B and C");
    require(sumCoverage(tree, 45) == 1, "v=45 covered by C only");
    require(sumCoverage(tree, -5) == 0, "v=-5 covered by none");
    require(sumCoverage(tree, 55) == 0, "v=55 covered by none");

    tree.deletePredicate(b);
    require(sumCoverage(tree, 15) == 1, "after deleting B, v=15 covered by A only");
    require(sumCoverage(tree, 25) == 2, "after deleting B, v=25 covered by A and C");
    require(sumCoverage(tree, 35) == 1, "after deleting B, v=35 covered by C only");
}

// `in` with lo == hi is a valid single-point range, equivalent to `=` for matching purposes -
// this is the REQUIRED base case in insertBetween/deleteBetween (see ps_tree.hpp's own
// file-level comment: the general LCA decomposition has no defined "first level where lo/hi
// diverge" when they're equal), not just a nice-to-have, so this is a direct regression test
// for that specific base case.
void test_in_with_equal_endpoints() {
    pstree::PSTree tree(pstree::Int64Codec::shape());
    auto p = simple(pstree::Op::kIn, 7);
    p.vals1 = pstree::Int64Codec::encode(7);
    auto leaves = tree.insertPredicate(p);
    require(leaves.size() == 1, "in[7,7] should touch exactly one bucket (routed to the equality path)");
    require(sumCoverage(tree, 7) == 1, "in[7,7]: exact value included");
    require(sumCoverage(tree, 6) == 0, "in[7,7]: value below excluded");
    require(sumCoverage(tree, 8) == 0, "in[7,7]: value above excluded");
    tree.deletePredicate(p);
    require(sumCoverage(tree, 7) == 0, "in[7,7]: deleted, back to 0");
}

// `in` where lo and hi diverge only at the LAST digit level - the other required base case
// (see ps_tree.hpp's file-level comment: lcaLevel == depth-1 collapses to one bounded marker
// at the lca node itself, since there's no deeper level to descend into for either branch).
void test_in_diverging_at_last_level() {
    pstree::PSTree tree(pstree::Int64Codec::shape());
    // 100 and 103 share every digit except the last 4-bit chunk (both small positive
    // integers, same sign/high bits, low nibble 100=0x64 vs 103=0x67 differs only in the
    // last hex digit of the encoded key).
    auto p = simple(pstree::Op::kIn, 100);
    p.vals1 = pstree::Int64Codec::encode(103);
    auto touched = tree.insertPredicate(p);
    require(touched.size() == 1, "in[100,103] diverging only at the last level should touch exactly one bucket");
    require(sumCoverage(tree, 99) == 0, "in[100,103]: value below excluded");
    require(sumCoverage(tree, 100) == 1, "in[100,103]: lower boundary included");
    require(sumCoverage(tree, 101) == 1, "in[100,103]: interior value included");
    require(sumCoverage(tree, 103) == 1, "in[100,103]: upper boundary included");
    require(sumCoverage(tree, 104) == 0, "in[100,103]: value above excluded");
    tree.deletePredicate(p);
    require(sumCoverage(tree, 101) == 0, "in[100,103]: deleted, back to 0");
}

} // namespace

int main() {
    test_strict_gt_lt_int64();
    test_strict_gt_lt_adjacent_composition();
    test_strict_gt_lt_domain_edge();
    test_duplicate_insert_delete();
    test_delete_underflow_throws();
    test_repeated_churn();
    test_three_way_overlap();
    test_in_with_equal_endpoints();
    test_in_diverging_at_last_level();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All edge case tests passed\n";
    return 0;
}
