// Randomized property-based test: a sequence of random insert/delete operations on a
// PSTree<int64_t>, cross-checked at every step against a brute-force reference (a plain
// vector of active predicates, each query answered by literally testing every one). This
// is the kind of test that catches bugs the hand-picked worked examples don't happen to
// exercise - fixed seed for reproducibility, so a failure here is always re-runnable.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
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

void test_random_insert_delete_matches_brute_force() {
    constexpr int kOperations = 500;
    constexpr int kQueriesPerCheck = 40;
    constexpr std::int64_t kValueRange = 200; // values drawn from [-200, 200]

    std::mt19937 rng(0xC0FFEE); // fixed seed - a failure here must be reproducible
    std::uniform_int_distribution<std::int64_t> valueDist(-kValueRange, kValueRange);
    std::uniform_int_distribution<int> opDist(0, 5);

    pstree::PSTree tree(pstree::Int64Codec::shape());
    std::vector<ActivePredicate> active;

    for (int iter = 0; iter < kOperations; ++iter) {
        // 65% insert, 35% delete-a-random-active-one (once any exist).
        bool doInsert = active.empty() || (rng() % 100) < 65;
        if (doInsert) {
            ActivePredicate p;
            p.op = static_cast<pstree::Op>(opDist(rng));
            p.lo = valueDist(rng);
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
            tree.deletePredicate(toPredicate(active[idx]));
            active.erase(active.begin() + idx);
        }

        // Spot-check a batch of random query points against the brute-force oracle.
        for (int q = 0; q < kQueriesPerCheck; ++q) {
            std::int64_t v = valueDist(rng);
            std::uint64_t expected = bruteForceCoverage(active, v);
            std::uint64_t actual = tree.matchPair(pstree::Int64Codec::encode(v))->predCounter;
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
        tree.deletePredicate(toPredicate(p));
    }
    for (int q = -kValueRange; q <= kValueRange; q += 7) {
        require(tree.matchPair(pstree::Int64Codec::encode(q))->predCounter == 0,
                "after draining all predicates, value " + std::to_string(q) + " should be uncovered");
    }
}

} // namespace

int main() {
    test_random_insert_delete_matches_brute_force();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All random stress tests passed\n";
    return 0;
}
