// Tests for matchEventCount()/matchEventEach() (pst_dynamic.hpp) - the count-only/callback-based
// siblings of matchEvent() added to let a caller (nats_sidecar's pstree_matching_engine::
// search_count()) get a match count or visit matched ids one at a time without ever
// materializing the std::vector<uint64_t> matchEvent() itself builds. All three share one
// underlying implementation (scanCandidates()+walkCandidates()) by construction - see that
// refactor's own comment in pst_dynamic.hpp - so these tests exist to catch exactly the class of
// bug that refactor itself already introduced once (a dangling-pointer bug from `internedEvent`
// briefly becoming a callee-local instead of staying caller-owned, caught immediately by the
// EXISTING test suite before being committed - not by anything in this file). Differential
// testing against matchEvent() directly is the actual guard against any FUTURE regression of the
// same shape.

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

pstree::SubPredicate P(std::string attr, pstree::CmpOp op, std::vector<pstree::Value> vals) {
    return pstree::SubPredicate{std::move(attr), op, std::move(vals)};
}

std::vector<pstree::AttrSchema> fig3Schema() {
    return {
        {"attr1", pstree::ValueType::kInteger, 0},
        {"attr2", pstree::ValueType::kInteger, 0},
        {"attr3", pstree::ValueType::kInteger, 0},
    };
}

// Same six subscriptions as test_pst_dynamic.cpp's own fig3Subscriptions() - duplicated (not
// shared via a header) to match this project's existing one-file-per-test-executable convention.
std::vector<pstree::Subscription> fig3Subscriptions() {
    using pstree::CmpOp;
    std::vector<pstree::Subscription> subs(6);
    subs[0] = {1, {P("attr1", CmpOp::kLt, {std::int64_t{0}}), P("attr2", CmpOp::kBetween, {std::int64_t{1}, std::int64_t{5}})}};
    subs[1] = {2, {P("attr1", CmpOp::kGt, {std::int64_t{8}}), P("attr2", CmpOp::kBetween, {std::int64_t{1}, std::int64_t{5}})}};
    subs[2] = {3, {P("attr2", CmpOp::kBetween, {std::int64_t{1}, std::int64_t{5}}), P("attr3", CmpOp::kLt, {std::int64_t{0}})}};
    subs[3] = {4, {P("attr2", CmpOp::kBetween, {std::int64_t{1}, std::int64_t{5}}), P("attr3", CmpOp::kGt, {std::int64_t{8}})}};
    subs[4] = {5, {P("attr1", CmpOp::kEq, {std::int64_t{2}}), P("attr2", CmpOp::kLt, {std::int64_t{1}}), P("attr3", CmpOp::kGt, {std::int64_t{8}})}};
    subs[5] = {6, {P("attr1", CmpOp::kLt, {std::int64_t{0}}), P("attr2", CmpOp::kGt, {std::int64_t{5}}), P("attr3", CmpOp::kEq, {std::int64_t{6}})}};
    return subs;
}

void test_matchEventCount_matches_matchEvent_size_on_fig3() {
    pstree::PSTDynamic pstd(fig3Schema());
    for (auto& sub : fig3Subscriptions()) pstd.insertSubscription(sub);

    // A handful of events spanning "matches nothing" through "matches several" - the same
    // section-5.3 event plus a few hand-picked variations, not just the paper's single example.
    std::vector<pstree::Event> events = {
        {{"attr1", pstree::Value(std::int64_t{2})}, {"attr2", pstree::Value(std::int64_t{0})}, {"attr3", pstree::Value(std::int64_t{10})}},
        {{"attr2", pstree::Value(std::int64_t{2})}, {"attr3", pstree::Value(std::int64_t{10})}},
        {{"attr1", pstree::Value(std::int64_t{100})}, {"attr2", pstree::Value(std::int64_t{100})}, {"attr3", pstree::Value(std::int64_t{100})}},
        {},
    };
    for (auto& e : events) {
        auto matches = pstd.matchEvent(e);
        require(pstd.matchEventCount(e) == matches.size(),
                "matchEventCount must equal matchEvent(e).size()");
    }
}

void test_matchEventEach_visits_same_ids_as_matchEvent() {
    pstree::PSTDynamic pstd(fig3Schema());
    for (auto& sub : fig3Subscriptions()) pstd.insertSubscription(sub);

    pstree::Event event = {{"attr2", pstree::Value(std::int64_t{2})}, {"attr3", pstree::Value(std::int64_t{10})}};
    auto viaMatchEvent = pstd.matchEvent(event);
    std::sort(viaMatchEvent.begin(), viaMatchEvent.end());

    std::vector<std::uint64_t> viaEach;
    pstd.matchEventEach(event, [&viaEach](std::uint64_t id) { viaEach.push_back(id); });
    std::sort(viaEach.begin(), viaEach.end());

    require(viaMatchEvent == viaEach, "matchEventEach must visit exactly the same id set as matchEvent");
}

void test_matchEventCount_edge_cases() {
    pstree::PSTDynamic pstd(fig3Schema());
    require(pstd.matchEventCount({}) == 0, "empty tree, empty event: count must be 0");
    require(pstd.matchEventCount({{"attr1", pstree::Value(std::int64_t{0})}}) == 0,
            "empty tree, non-empty event: count must be 0");

    for (auto& sub : fig3Subscriptions()) pstd.insertSubscription(sub);
    require(pstd.matchEventCount({}) == 0, "empty event against a populated tree: count must be 0");

    pstree::Event matchesEverything = {
        {"attr1", pstree::Value(std::int64_t{-100})},
        {"attr2", pstree::Value(std::int64_t{3})},
        {"attr3", pstree::Value(std::int64_t{-100})},
    };
    require(pstd.matchEventCount(matchesEverything) == pstd.matchEvent(matchesEverything).size(),
            "count/matchEvent parity must hold even for an event matching several subscriptions at once");
}

// --- Random differential stress: matchEventCount()/matchEventEach() vs. matchEvent() itself,
// across a randomized sequence of insert/delete operations - the actual guard against a future
// regression of the exact scanCandidates()/walkCandidates() lifetime bug this refactor caught
// once already (see this file's own top comment). Generators mirror test_pst_dynamic_stress.cpp's
// own (duplicated, not shared - this project's existing per-file convention), trimmed to the
// subset of types/ops needed here.

struct Dim {
    std::string name;
    pstree::ValueType type;
};

std::vector<Dim> makeDims() {
    return {
        {"i0", pstree::ValueType::kInteger}, {"i1", pstree::ValueType::kInteger},
        {"f0", pstree::ValueType::kFloat},
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
        case pstree::ValueType::kDecimal:
            return pstree::Value(std::int64_t{0}); // not used by makeDims() above
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
    for (std::size_t i = 0; i < count; ++i) sub.predicates.push_back(randomPredicate(shuffled[i], rng));
    return sub;
}

pstree::Event randomEvent(const std::vector<Dim>& dims, std::mt19937& rng) {
    pstree::Event event;
    for (auto& dim : dims) {
        if (rng() % 100 < 70) event.push_back({dim.name, randomValue(dim, rng)});
    }
    return event;
}

void test_matchEventCount_random_stress() {
    constexpr int kOperations = 300;
    constexpr int kEventsPerCheck = 10;

    std::mt19937 rng(0x5EED1234);
    auto dims = makeDims();
    pstree::PSTDynamic pstd(makeSchema(dims));
    std::vector<std::uint64_t> activeIds;
    std::uint64_t nextId = 1;

    for (int iter = 0; iter < kOperations; ++iter) {
        bool doInsert = activeIds.empty() || (rng() % 100) < 70;
        if (doInsert) {
            auto sub = randomSubscription(nextId++, dims, rng);
            try {
                pstd.insertSubscription(sub);
            } catch (const std::invalid_argument&) {
                continue; // an all-kIsNull subscription is legitimately rejected - see pst_dynamic.hpp
            }
            activeIds.push_back(sub.id);
        } else {
            std::uniform_int_distribution<std::size_t> pick(0, activeIds.size() - 1);
            std::size_t idx = pick(rng);
            pstd.deleteSubscription(activeIds[idx]);
            activeIds.erase(activeIds.begin() + idx);
        }

        for (int e = 0; e < kEventsPerCheck; ++e) {
            pstree::Event event = randomEvent(dims, rng);
            auto viaMatchEvent = pstd.matchEvent(event);
            std::size_t count = pstd.matchEventCount(event);
            if (count != viaMatchEvent.size()) {
                require(false, "matchEventCount()==" + std::to_string(count) +
                                    " diverged from matchEvent().size()==" +
                                    std::to_string(viaMatchEvent.size()) + " at iter " + std::to_string(iter));
                return;
            }

            std::sort(viaMatchEvent.begin(), viaMatchEvent.end());
            std::vector<std::uint64_t> viaEach;
            pstd.matchEventEach(event, [&viaEach](std::uint64_t id) { viaEach.push_back(id); });
            std::sort(viaEach.begin(), viaEach.end());
            if (viaEach != viaMatchEvent) {
                require(false, "matchEventEach's visited id set diverged from matchEvent() at iter " +
                                    std::to_string(iter));
                return;
            }
        }
    }
}

} // namespace

int main() {
    test_matchEventCount_matches_matchEvent_size_on_fig3();
    test_matchEventEach_visits_same_ids_as_matchEvent();
    test_matchEventCount_edge_cases();
    test_matchEventCount_random_stress();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All pst_dynamic count tests passed\n";
    return 0;
}
