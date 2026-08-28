// Randomized property-based test for PSTDynamic: a sequence of random insert/delete
// subscription operations (multiple predicates each, mixed types and operators), checked
// after every step against a brute-force oracle - directly evaluate every currently-active
// subscription against a random event via matchSubscription(), bypassing PSTDynamic's own
// indexing entirely. This is the same discipline that caught a real, deep bug in PS-Tree
// itself (see ps_tree.hpp's file-level comment #3 and the project's commit history) - the
// hand-picked Fig. 3 worked example alone would not have been enough to catch an analogous
// class of bug here either. Fixed seed for reproducibility.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "pstree/pst_dynamic.hpp"

namespace {

int g_failures = 0;

void require(bool cond, const std::string& message) {
    if (!cond) {
        std::cerr << "FAIL: " << message << "\n";
        g_failures++;
    }
}

struct Dim {
    std::string name;
    pstree::ValueType type;
};

std::vector<Dim> makeDims() {
    return {
        {"i0", pstree::ValueType::kInteger}, {"i1", pstree::ValueType::kInteger}, {"i2", pstree::ValueType::kInteger},
        {"f0", pstree::ValueType::kFloat}, {"f1", pstree::ValueType::kFloat},
        {"b0", pstree::ValueType::kBoolean},
        {"s0", pstree::ValueType::kString}, {"s1", pstree::ValueType::kString},
    };
}

std::vector<pstree::AttrSchema> makeSchema(const std::vector<Dim>& dims) {
    std::vector<pstree::AttrSchema> schema;
    for (auto& d : dims) schema.push_back({d.name, d.type, 8});
    return schema;
}

pstree::Value randomValue(const Dim& dim, std::mt19937& rng) {
    switch (dim.type) {
        case pstree::ValueType::kInteger: {
            std::uniform_int_distribution<std::int64_t> d(-20, 20);
            return pstree::Value(d(rng));
        }
        case pstree::ValueType::kFloat: {
            std::uniform_real_distribution<double> d(-20.0, 20.0);
            return pstree::Value(d(rng));
        }
        case pstree::ValueType::kBoolean: {
            std::uniform_int_distribution<int> d(0, 1);
            return pstree::Value(d(rng) == 1);
        }
        case pstree::ValueType::kString: {
            std::uniform_int_distribution<int> d(0, 4);
            static const char* words[] = {"aa", "bb", "cc", "dd", "ee"};
            return pstree::Value(std::string(words[d(rng)]));
        }
    }
    return pstree::Value(std::int64_t{0});
}

pstree::CmpOp randomOp(std::mt19937& rng) {
    std::uniform_int_distribution<int> d(0, 10); // includes kIsNull(9)/kIsNotNull(10)
    return static_cast<pstree::CmpOp>(d(rng));
}

pstree::SubPredicate randomPredicate(const Dim& dim, std::mt19937& rng) {
    pstree::CmpOp op = randomOp(rng);
    if (op == pstree::CmpOp::kBetween) {
        pstree::Value a = randomValue(dim, rng);
        pstree::Value b = randomValue(dim, rng);
        if (b < a) std::swap(a, b);
        return {dim.name, op, {a, b}};
    }
    if (op == pstree::CmpOp::kElemOf || op == pstree::CmpOp::kNotElemOf) {
        std::uniform_int_distribution<int> countDist(1, 3);
        std::vector<pstree::Value> vals;
        for (int i = 0; i < countDist(rng); ++i) vals.push_back(randomValue(dim, rng));
        return {dim.name, op, vals};
    }
    if (op == pstree::CmpOp::kIsNull || op == pstree::CmpOp::kIsNotNull) {
        return {dim.name, op, {}};
    }
    return {dim.name, op, {randomValue(dim, rng)}};
}

pstree::Subscription randomSubscription(std::uint64_t id, const std::vector<Dim>& dims, std::mt19937& rng) {
    std::vector<Dim> shuffled = dims;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    std::uniform_int_distribution<std::size_t> countDist(1, 4);
    std::size_t count = std::min(countDist(rng), shuffled.size());
    pstree::Subscription sub;
    sub.id = id;
    for (std::size_t i = 0; i < count; ++i) {
        sub.predicates.push_back(randomPredicate(shuffled[i], rng));
    }
    return sub;
}

pstree::Event randomEvent(const std::vector<Dim>& dims, std::mt19937& rng) {
    pstree::Event event;
    for (auto& dim : dims) {
        // Each dimension present with 70% probability - a mix of full and partial events.
        if (rng() % 100 < 70) {
            event.push_back({dim.name, randomValue(dim, rng)});
        }
    }
    return event;
}

std::vector<std::uint64_t> bruteForceMatch(const pstree::Event& event, const std::vector<pstree::Subscription>& active) {
    std::vector<std::uint64_t> out;
    for (auto& sub : active) {
        if (pstree::matchSubscription(event, sub)) out.push_back(sub.id);
    }
    std::sort(out.begin(), out.end());
    return out;
}

void test_random_insert_delete_matches_brute_force() {
    constexpr int kOperations = 400;
    constexpr int kEventsPerCheck = 15;

    std::mt19937 rng(0xBADC0FFE);
    auto dims = makeDims();
    pstree::PSTDynamic pstd(makeSchema(dims));
    std::vector<pstree::Subscription> active;
    std::uint64_t nextId = 1;

    for (int iter = 0; iter < kOperations; ++iter) {
        bool doInsert = active.empty() || (rng() % 100) < 70;
        if (doInsert) {
            auto sub = randomSubscription(nextId++, dims, rng);
            // A subscription whose every predicate happens to be kIsNull (possible now
            // that randomOp can generate one) is legitimately rejected by
            // insertSubscription - see pst_dynamic.hpp's own comment. Skip it here rather
            // than treat it as a test failure, the same way a real caller validating its
            // own subscriptions would.
            try {
                pstd.insertSubscription(sub);
            } catch (const std::invalid_argument&) {
                continue;
            }
            active.push_back(sub);
        } else {
            std::uniform_int_distribution<std::size_t> pick(0, active.size() - 1);
            std::size_t idx = pick(rng);
            pstd.deleteSubscription(active[idx].id);
            active.erase(active.begin() + idx);
        }

        for (int e = 0; e < kEventsPerCheck; ++e) {
            pstree::Event event = randomEvent(dims, rng);
            auto expected = bruteForceMatch(event, active);
            auto actual = pstd.matchEvent(event);
            std::sort(actual.begin(), actual.end());
            if (expected != actual) {
                std::cerr << "MISMATCH at iter " << iter << ": expected " << expected.size()
                          << " matches, got " << actual.size() << "\n";
                std::cerr << "  expected ids:";
                for (auto id : expected) std::cerr << " " << id;
                std::cerr << "\n  actual ids:  ";
                for (auto id : actual) std::cerr << " " << id;
                std::cerr << "\n";
                require(false, "PSTDynamic result diverged from brute-force oracle");
                return;
            }
        }
    }

    // Drain everything, confirm nothing matches anymore anywhere.
    for (auto& sub : active) pstd.deleteSubscription(sub.id);
    for (int e = 0; e < 30; ++e) {
        pstree::Event event = randomEvent(dims, rng);
        require(pstd.matchEvent(event).empty(), "after deleting every subscription, nothing should match");
    }
}

} // namespace

int main() {
    test_random_insert_delete_matches_brute_force();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All pst_dynamic stress tests passed\n";
    return 0;
}
