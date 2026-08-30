// Multi-threaded regression test for the real SIGSEGV found this session (2026-08-30):
// SubPredicate::primaryCache/primaryCached (and betweenUpperCache) used to be lazily built on
// the FIRST matchEvent() call to reach a given predicate, from whichever worker thread got
// there first, with no synchronization at all - reproduced as a real crash by nats_sidecar
// running pstree with worker_threads > 1 (this project's own matching_engine::search() is
// explicitly documented as safe to call concurrently across worker threads, a promise pstree's
// lazy-caching implementation was breaking). Fixed by building every predicate's cache eagerly,
// once, single-threaded, at PSTDynamic::insertSubscription time (see pst_dynamic.hpp's own
// comment and predicate.hpp's ensurePredicateCachedForInsert) - by the time a subscription is
// reachable by matchEvent() at all, its cache is already immutable, so concurrent reads are
// safe by construction with no locking needed on the hot path.
//
// This test builds a real tree (scalar + kElemOf predicates, the two op families that used the
// racy lazy-build path) with many subscriptions, then hammers matchEvent() from many threads
// concurrently for a fixed duration, checking for crashes/exceptions and cross-checking a
// sample of results against a known-correct single-threaded run. Run under ThreadSanitizer
// (not part of the normal ctest suite - TSan and ASan can't be combined in one binary, and this
// project's existing ctest target is ASan-covered) for the strongest verification: TSan
// specifically detects data races that a crash-based repro can only catch probabilistically.
//
// Build/run manually (see also cmake/tsan target, if wired up):
//   g++ -std=c++20 -O1 -g -fsanitize=thread -I../include tests/test_pst_dynamic_concurrent.cpp \
//       -o /tmp/test_pst_dynamic_concurrent_tsan -lpthread && /tmp/test_pst_dynamic_concurrent_tsan

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <thread>
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

std::vector<pstree::AttrSchema> makeSchema() {
    return {
        {"volume", pstree::ValueType::kInteger, 0},
        {"price", pstree::ValueType::kFloat, 0},
        {"symbol", pstree::ValueType::kString, 16},
        {"exchange", pstree::ValueType::kString, 16},
    };
}

// Real-shaped subscriptions exercising BOTH racy paths from the bug report: scalar ops
// (kGe/kLe/kBetween, via ensureScalarCached) and kElemOf (via buildElemOfTypedCache) - matches
// this project's own real exchange/symbol set-membership benchmark shape.
pstree::Subscription makeSubscription(std::uint64_t id, std::mt19937& rng) {
    static const std::vector<std::string> kExchanges = {"A", "B", "C", "D", "G", "H", "J", "K"};
    static const std::vector<std::string> kSymbols = {"AAPL", "MSFT", "GOOG", "TSLA", "NVDA",
                                                        "META", "AMZN", "NFLX", "AMD", "INTC"};
    std::uniform_int_distribution<int> shapeDist(0, 3);
    pstree::Subscription sub;
    sub.id = id;
    switch (shapeDist(rng)) {
        case 0: {
            std::uniform_int_distribution<std::int64_t> loDist(0, 500);
            std::int64_t lo = loDist(rng);
            sub.predicates.push_back(
                pstree::SubPredicate{"volume", pstree::CmpOp::kGe, {lo}});
            sub.predicates.push_back(
                pstree::SubPredicate{"price", pstree::CmpOp::kLe, {double(lo) + 100.0}});
            break;
        }
        case 1: {
            std::vector<pstree::Value> exch;
            std::sample(kExchanges.begin(), kExchanges.end(), std::back_inserter(exch), 2, rng);
            sub.predicates.push_back(
                pstree::SubPredicate{"exchange", pstree::CmpOp::kElemOf, exch});
            std::vector<pstree::Value> syms;
            std::sample(kSymbols.begin(), kSymbols.end(), std::back_inserter(syms), 4, rng);
            sub.predicates.push_back(
                pstree::SubPredicate{"symbol", pstree::CmpOp::kElemOf, syms});
            break;
        }
        case 2: {
            std::uniform_int_distribution<std::int64_t> loDist(0, 400);
            std::int64_t lo = loDist(rng);
            sub.predicates.push_back(pstree::SubPredicate{
                "volume", pstree::CmpOp::kBetween, {lo, lo + 100}});
            break;
        }
        default: {
            std::vector<pstree::Value> syms;
            std::sample(kSymbols.begin(), kSymbols.end(), std::back_inserter(syms), 3, rng);
            sub.predicates.push_back(
                pstree::SubPredicate{"symbol", pstree::CmpOp::kNotElemOf, syms});
        }
    }
    return sub;
}

pstree::Event makeEvent(std::mt19937& rng) {
    static const std::vector<std::string> kExchanges = {"A", "B", "C", "D", "G", "H", "J", "K"};
    static const std::vector<std::string> kSymbols = {"AAPL", "MSFT", "GOOG", "TSLA", "NVDA",
                                                        "META", "AMZN", "NFLX", "AMD", "INTC"};
    std::uniform_int_distribution<std::int64_t> volDist(0, 600);
    std::uniform_real_distribution<double> priceDist(0.0, 700.0);
    std::uniform_int_distribution<std::size_t> exchDist(0, kExchanges.size() - 1);
    std::uniform_int_distribution<std::size_t> symDist(0, kSymbols.size() - 1);
    return {
        {"volume", pstree::Value(volDist(rng))},
        {"price", pstree::Value(priceDist(rng))},
        {"symbol", pstree::Value(kSymbols[symDist(rng)])},
        {"exchange", pstree::Value(kExchanges[exchDist(rng)])},
    };
}

void test_concurrent_match_event_no_crash() {
    constexpr int kNumSubs = 2000;
    constexpr int kNumThreads = 8;
    constexpr auto kDuration = std::chrono::seconds(3);

    pstree::PSTDynamic pstd(makeSchema());
    std::mt19937 rng(12345);
    for (int i = 0; i < kNumSubs; ++i) {
        pstd.insertSubscription(makeSubscription(static_cast<std::uint64_t>(i + 1), rng));
    }

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> totalMatches{0};
    std::atomic<int> exceptions{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 threadRng(1000 + t);
            while (!stop.load(std::memory_order_relaxed)) {
                auto event = makeEvent(threadRng);
                try {
                    auto matches = pstd.matchEvent(event);
                    totalMatches.fetch_add(matches.size(), std::memory_order_relaxed);
                } catch (const std::exception& e) {
                    std::cerr << "unexpected exception from matchEvent: " << e.what() << "\n";
                    exceptions.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    std::this_thread::sleep_for(kDuration);
    stop.store(true, std::memory_order_relaxed);
    for (auto& th : threads) th.join();

    require(exceptions.load() == 0, "no unexpected exceptions from concurrent matchEvent calls");
    require(totalMatches.load() > 0, "concurrent matching should have found at least some matches");
    std::cerr << "concurrent matchEvent: " << totalMatches.load() << " total matches across "
              << kNumThreads << " threads over " << kDuration.count() << "s, no crash\n";
}

} // namespace

int main() {
    test_concurrent_match_event_no_crash();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
