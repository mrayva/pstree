// Randomized property-based test: a sequence of random insert/delete operations on a
// PSTree<int64_t>, cross-checked at every step against a brute-force reference (a plain
// vector of active predicates, each query answered by literally testing every one). This
// is the kind of test that catches bugs the hand-picked worked examples don't happen to
// exercise - fixed seed for reproducibility, so a failure here is always re-runnable.
//
// Also covers, beyond the original narrow-range mix: (1) a deliberate bias toward wide,
// unbounded ranges (kGe/kLe with thresholds near the domain's edges) - exactly the shape that
// exposed the real scaling problem this file's canonical-ancestor-marker redesign fixes (see
// ps_tree.hpp's own file-level comment); (2) a direct complexity regression test pinning that
// a wide kGe insertion's own touched-bucket count stays a small, K-independent constant
// (O(depth)) regardless of how many other predicates already exist on the dimension - there
// was previously no test asserting the complexity claim itself, only functional correctness
// of a single insert; (3) a small exhaustive brute-force check of every (lo,hi) BETWEEN pair
// in a tiny value range, specifically to nail down the LCA three-way split (including the
// lo==hi and "diverges only at the last level" base cases) beyond what randomization alone
// might happen to hit.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
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

struct ActivePredicate {
    pstree::Op op;
    std::int64_t lo; // sole value for kGe/kGt/kEq/kLt/kLe; low endpoint for kIn
    std::int64_t hi; // only meaningful for kIn

    bool covers(std::int64_t v) const {
        switch (op) {
            case pstree::Op::kGe: return v >= lo;
            case pstree::Op::kGt: return v > lo;
            case pstree::Op::kEq: return v == lo;
            case pstree::Op::kLt: return v < lo;
            case pstree::Op::kLe: return v <= lo;
            case pstree::Op::kIn: return v >= lo && v <= hi;
        }
        return false;
    }
};

pstree::Predicate toPredicate(const ActivePredicate& p) {
    pstree::Predicate pred;
    pred.op = p.op;
    pred.vals0 = pstree::Int64Codec::encode(p.lo);
    if (p.op == pstree::Op::kIn) pred.vals1 = pstree::Int64Codec::encode(p.hi);
    return pred;
}

std::uint64_t bruteForceCoverage(const std::vector<ActivePredicate>& active, std::int64_t v) {
    std::uint64_t count = 0;
    for (const auto& p : active) {
        if (p.covers(v)) count++;
    }
    return count;
}

std::uint64_t sumCoverage(const pstree::PSTree& tree, std::int64_t v) {
    std::uint64_t total = 0;
    for (pstree::LeafNode* bucket : tree.matchPoint(pstree::Int64Codec::encode(v))) {
        total += bucket->predCounter;
    }
    return total;
}

void test_random_insert_delete_matches_brute_force() {
    constexpr int kOperations = 500;
    constexpr int kQueriesPerCheck = 40;
    constexpr std::int64_t kValueRange = 200; // values drawn from [-200, 200]

    std::mt19937 rng(0xC0FFEE); // fixed seed - a failure here must be reproducible
    std::uniform_int_distribution<std::int64_t> valueDist(-kValueRange, kValueRange);
    std::uniform_int_distribution<int> opDist(0, 5);
    // Bias 30% of generated kGe/kLe predicates toward a threshold near a domain edge
    // (INT64_MIN/MAX, not just [-200,200]) - exactly the "wide, unbounded" shape that exposed
    // the scaling problem this redesign fixes, so the random walk actually exercises deep,
    // many-level canonical decompositions, not just shallow ones confined to a narrow range.
    std::uniform_int_distribution<int> wideBiasDist(0, 99);
    std::uniform_int_distribution<std::int64_t> wideValueDist(
        std::numeric_limits<std::int64_t>::min() / 2, std::numeric_limits<std::int64_t>::max() / 2);

    pstree::PSTree tree(pstree::Int64Codec::shape());
    std::vector<ActivePredicate> active;

    for (int iter = 0; iter < kOperations; ++iter) {
        // 65% insert, 35% delete-a-random-active-one (once any exist).
        bool doInsert = active.empty() || (rng() % 100) < 65;
        if (doInsert) {
            ActivePredicate p;
            p.op = static_cast<pstree::Op>(opDist(rng));
            bool useWide = (p.op == pstree::Op::kGe || p.op == pstree::Op::kLe) && wideBiasDist(rng) < 30;
            p.lo = useWide ? wideValueDist(rng) : valueDist(rng);
            if (p.op == pstree::Op::kIn) {
                std::int64_t other = valueDist(rng);
                p.hi = std::max(p.lo, other);
                p.lo = std::min(p.lo, other);
            } else {
                p.hi = p.lo;
            }
            tree.insertPredicate(toPredicate(p));
            active.push_back(p);
        } else {
            std::uniform_int_distribution<std::size_t> pick(0, active.size() - 1);
            std::size_t idx = pick(rng);
            pstree::Predicate deleted = toPredicate(active[idx]);
            tree.deletePredicate(deleted);
            // Exercise reclaim() on every single delete here too, not just the final drain
            // below - this walk's own wide-range bias (30% of kGe/kLe near INT64_MIN/MAX) means
            // predicates constantly overlap and share buckets/ancestor nodes with each other,
            // a much more adversarial mix for reclaim's zero-check-before-free/prune logic than
            // the isolated single-predicate cases test_ps_tree.cpp's own reclaim tests use.
            tree.reclaim(deleted);
            active.erase(active.begin() + idx);
        }

        // Spot-check a batch of random query points against the brute-force oracle.
        for (int q = 0; q < kQueriesPerCheck; ++q) {
            std::int64_t v = valueDist(rng);
            std::uint64_t expected = bruteForceCoverage(active, v);
            std::uint64_t actual = sumCoverage(tree, v);
            if (expected != actual) {
                require(false, "mismatch at iter " + std::to_string(iter) + " value " + std::to_string(v) +
                                    ": expected " + std::to_string(expected) + " got " + std::to_string(actual));
                return; // one detailed failure is more useful than hundreds of cascading ones
            }
        }
    }

    // Drain every remaining active predicate and confirm the tree returns to all-zero
    // everywhere sampled - exercises DeletePredicate's underflow guard never firing on a
    // legitimate sequence, and confirms no coverage was ever double-counted or dropped.
    for (const auto& p : active) {
        pstree::Predicate deleted = toPredicate(p);
        tree.deletePredicate(deleted);
        tree.reclaim(deleted);
    }
    for (int q = -kValueRange; q <= kValueRange; q += 7) {
        require(sumCoverage(tree, q) == 0, "after draining all predicates, value " + std::to_string(q) + " should be uncovered");
    }
    // Stronger than the coverage check above: after reclaiming every predicate this whole
    // randomized walk ever inserted (hundreds of operations, heavily biased toward wide,
    // deep, overlapping ranges), the tree's actual structure - not just observable coverage -
    // should collapse all the way back to an empty root, same as test_ps_tree.cpp's own
    // single-predicate reclaim tests, now proven under real adversarial churn.
    require(tree.root()->p.empty(),
            "after draining and reclaiming every predicate, the tree structure should be fully "
            "pruned back to an empty root, not just zero-coverage");
}

// Direct complexity regression: a wide kGe insertion's own touched-bucket count must stay a
// small, K-independent constant (bounded by the encoding's depth, 16 for int64) regardless of
// how many OTHER, unrelated predicates already exist on the dimension - the actual property
// this redesign exists to establish, not just "still functionally correct".
void test_wide_predicate_insert_is_k_independent() {
    constexpr std::size_t kDepth = 16; // Int64Codec::kChunks
    std::mt19937 rng(0xBEEF);
    std::uniform_int_distribution<std::int64_t> valueDist(
        std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max());

    for (std::size_t priorCount : {std::size_t{10}, std::size_t{1000}, std::size_t{20000}}) {
        pstree::PSTree tree(pstree::Int64Codec::shape());
        for (std::size_t i = 0; i < priorCount; ++i) {
            pstree::Predicate p{pstree::Op::kGe, pstree::Int64Codec::encode(valueDist(rng)), {}};
            tree.insertPredicate(p);
        }
        pstree::Predicate probe{pstree::Op::kGe, pstree::Int64Codec::encode(valueDist(rng)), {}};
        auto touched = tree.insertPredicate(probe);
        require(touched.size() <= kDepth,
                "a wide kGe insertion should touch at most depth (" + std::to_string(kDepth) +
                    ") buckets regardless of " + std::to_string(priorCount) +
                    " prior predicates - got " + std::to_string(touched.size()));
    }
}

// Exhaustive brute-force check of every (lo,hi) BETWEEN pair in a small value range,
// specifically nailing down the LCA three-way split's base cases (lo==hi, and lo/hi
// diverging only at the last digit level) beyond what randomization alone might happen to
// hit reliably.
void test_between_exhaustive_small_domain() {
    constexpr std::int64_t kMin = -8;
    constexpr std::int64_t kMax = 8;

    for (std::int64_t lo = kMin; lo <= kMax; ++lo) {
        for (std::int64_t hi = lo; hi <= kMax; ++hi) {
            pstree::PSTree tree(pstree::Int64Codec::shape());
            pstree::Predicate p{pstree::Op::kIn, pstree::Int64Codec::encode(lo), pstree::Int64Codec::encode(hi)};
            tree.insertPredicate(p);
            for (std::int64_t v = kMin; v <= kMax; ++v) {
                bool expected = v >= lo && v <= hi;
                bool actual = sumCoverage(tree, v) != 0;
                if (expected != actual) {
                    require(false, "BETWEEN[" + std::to_string(lo) + "," + std::to_string(hi) +
                                        "] at v=" + std::to_string(v) + ": expected " +
                                        (expected ? "covered" : "uncovered") + " got " +
                                        (actual ? "covered" : "uncovered"));
                    return;
                }
            }
            tree.deletePredicate(p);
            for (std::int64_t v = kMin; v <= kMax; ++v) {
                require(sumCoverage(tree, v) == 0, "BETWEEN[" + std::to_string(lo) + "," + std::to_string(hi) +
                                                        "] at v=" + std::to_string(v) + ": should be uncovered after delete");
            }
        }
    }
}

} // namespace

int main() {
    test_random_insert_delete_matches_brute_force();
    test_wide_predicate_insert_is_k_independent();
    test_between_exhaustive_small_domain();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All random stress tests passed\n";
    return 0;
}
