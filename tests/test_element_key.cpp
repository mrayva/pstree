// Dedicated correctness tests for order_key.hpp's ElementKey - the small-vector-optimized
// container replacing std::vector<uint16_t> for the value-encoding key (see its own
// file-level comment for why: removing a per-row heap allocation MatchEvent's hot path was
// paying on every single event). This type hand-rolls its own memcpy-based storage
// management (not just delegating to std::vector), which is exactly the kind of code that
// needs its OWN direct, exhaustive tests - not just indirect coverage via order_key's own
// codec round-trip tests and PSTree's higher-level tests - to catch bugs in growth,
// copy/move, and self-assignment specifically. Run under ASan+UBSan in CI, same as every
// other suite here, to catch use-after-free/leaks/overflows directly.

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pstree/order_key.hpp"

namespace {

int g_failures = 0;

void require(bool cond, const std::string& message) {
    if (!cond) {
        std::cerr << "FAIL: " << message << "\n";
        g_failures++;
    }
}

void test_default_construction() {
    pstree::ElementKey key;
    require(key.empty(), "default-constructed key should be empty");
    require(key.size() == 0, "default-constructed key should have size 0");
}

void test_sized_construction_is_zero_filled() {
    pstree::ElementKey small(4);   // inline case
    pstree::ElementKey large(64);  // heap case (past kInlineCapacity=16)
    require(small.size() == 4, "sized ctor: inline size correct");
    require(large.size() == 64, "sized ctor: heap size correct");
    for (std::size_t i = 0; i < small.size(); ++i) {
        require(small[i] == 0, "sized ctor: inline element " + std::to_string(i) + " zero-filled");
    }
    for (std::size_t i = 0; i < large.size(); ++i) {
        require(large[i] == 0, "sized ctor: heap element " + std::to_string(i) + " zero-filled");
    }
}

void test_initializer_list_construction() {
    pstree::ElementKey key{1, 2, 3};
    require(key.size() == 3, "initializer_list ctor: size 3");
    require(key[0] == 1 && key[1] == 2 && key[2] == 3, "initializer_list ctor: contents correct");

    pstree::ElementKey empty{};
    require(empty.empty(), "empty initializer_list (or {}) should construct an empty key, not a 1-element one");
}

// Exercises the inline<->heap boundary explicitly, since that transition is where this kind
// of hand-rolled storage bug usually hides - kInlineCapacity is 16.
void test_inline_heap_boundary() {
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{15}, std::size_t{16}, std::size_t{17}, std::size_t{128}}) {
        pstree::ElementKey key(n);
        require(key.size() == n, "boundary test: size " + std::to_string(n) + " correct");
        for (std::size_t i = 0; i < n; ++i) key[i] = static_cast<std::uint16_t>(i % 65536);
        for (std::size_t i = 0; i < n; ++i) {
            require(key[i] == static_cast<std::uint16_t>(i % 65536),
                    "boundary test: size " + std::to_string(n) + " element " + std::to_string(i) + " round-trips");
        }
    }
}

void test_copy_independence_inline_and_heap() {
    pstree::ElementKey originalSmall{1, 2, 3};
    pstree::ElementKey copySmall = originalSmall;
    copySmall[0] = 99;
    require(originalSmall[0] == 1, "inline copy: mutating the copy must not affect the original");
    require(copySmall[0] == 99, "inline copy: the copy itself reflects the mutation");

    pstree::ElementKey originalBig(40);
    for (std::size_t i = 0; i < 40; ++i) originalBig[i] = static_cast<std::uint16_t>(i);
    pstree::ElementKey copyBig = originalBig;
    copyBig[39] = 9999;
    require(originalBig[39] == 39, "heap copy: mutating the copy must not affect the original");
    for (std::size_t i = 0; i < 39; ++i) {
        require(copyBig[i] == originalBig[i], "heap copy: untouched elements still match at " + std::to_string(i));
    }
}

void test_copy_assignment_including_size_transitions() {
    // Assigning a HEAP-sized value into a key that already holds an INLINE value, and
    // vice versa - both directions of the inline<->heap transition via assignment, not just
    // construction.
    pstree::ElementKey a(3);
    pstree::ElementKey big(50);
    for (std::size_t i = 0; i < 50; ++i) big[i] = static_cast<std::uint16_t>(i);
    a = big;
    require(a.size() == 50, "copy-assign inline->heap: size updated");
    require(a[49] == 49, "copy-assign inline->heap: contents correct");
    big[0] = 12345;
    require(a[0] == 0, "copy-assign inline->heap: independence after assignment");

    pstree::ElementKey b(50);
    for (std::size_t i = 0; i < 50; ++i) b[i] = static_cast<std::uint16_t>(i + 1);
    pstree::ElementKey small3{7, 8, 9};
    b = small3;
    require(b.size() == 3, "copy-assign heap->inline: size shrunk correctly");
    require(b[0] == 7 && b[1] == 8 && b[2] == 9, "copy-assign heap->inline: contents correct");

    // Self-assignment must be a safe no-op.
    pstree::ElementKey self{4, 5, 6};
    self = self;
    require(self.size() == 3 && self[0] == 4 && self[1] == 5 && self[2] == 6,
            "self copy-assignment should leave contents unchanged");
}

void test_move_construction_and_assignment() {
    pstree::ElementKey smallSrc{1, 2};
    pstree::ElementKey smallDst = std::move(smallSrc);
    require(smallDst.size() == 2 && smallDst[0] == 1 && smallDst[1] == 2,
            "inline move-construction: destination has source's contents");
    require(smallSrc.size() == 0, "inline move-construction: source left empty");

    pstree::ElementKey bigSrc(30);
    for (std::size_t i = 0; i < 30; ++i) bigSrc[i] = static_cast<std::uint16_t>(i);
    pstree::ElementKey bigDst = std::move(bigSrc);
    require(bigDst.size() == 30, "heap move-construction: destination has all 30 elements");
    require(bigDst[29] == 29, "heap move-construction: last element correct");
    require(bigSrc.size() == 0, "heap move-construction: source left empty");

    // Move-assignment onto a key that already owns its own heap buffer - must free its own
    // prior allocation, not leak it, before taking over the source's.
    pstree::ElementKey target(20);
    pstree::ElementKey moveSrc(25);
    for (std::size_t i = 0; i < 25; ++i) moveSrc[i] = static_cast<std::uint16_t>(100 + i);
    target = std::move(moveSrc);
    require(target.size() == 25, "move-assignment onto an existing heap buffer: size updated");
    require(target[0] == 100 && target[24] == 124, "move-assignment: contents replaced correctly");
    require(moveSrc.size() == 0, "move-assignment: source left empty");

    // Self-move-assignment must be a safe no-op.
    pstree::ElementKey selfMove{9, 8, 7};
    selfMove = std::move(selfMove);
    require(selfMove.size() == 3 && selfMove[0] == 9 && selfMove[1] == 8 && selfMove[2] == 7,
            "self move-assignment should leave contents unchanged");
}

void test_at_bounds_checking() {
    pstree::ElementKey empty;
    bool threwOnEmpty = false;
    try {
        (void)empty.at(0);
    } catch (const std::out_of_range&) {
        threwOnEmpty = true;
    }
    require(threwOnEmpty, "at(0) on an empty key should throw std::out_of_range");

    pstree::ElementKey key{42};
    require(key.at(0) == 42, "at(0) on a valid index should return the right element");
    bool threwPastEnd = false;
    try {
        (void)key.at(1);
    } catch (const std::out_of_range&) {
        threwPastEnd = true;
    }
    require(threwPastEnd, "at(size()) should throw std::out_of_range");
}

void test_equality() {
    pstree::ElementKey a{1, 2, 3};
    pstree::ElementKey b{1, 2, 3};
    pstree::ElementKey c{1, 2, 4};
    pstree::ElementKey shorter{1, 2};
    require(a == b, "equal-content keys should compare equal");
    require(!(a == c), "different-content keys should compare unequal");
    require(a != c, "operator!= should agree with operator==");
    require(!(a == shorter), "different-length keys should compare unequal even on a shared prefix");

    // Equality must also hold across the inline/heap boundary - two keys with the same long
    // content, one built to force a heap allocation, should still compare equal.
    pstree::ElementKey heapA(20);
    pstree::ElementKey heapB(20);
    for (std::size_t i = 0; i < 20; ++i) { heapA[i] = static_cast<std::uint16_t>(i); heapB[i] = static_cast<std::uint16_t>(i); }
    require(heapA == heapB, "equal-content heap-sized keys should compare equal");
    heapB[19] = 9999;
    require(!(heapA == heapB), "heap-sized keys differing in the last element should compare unequal");
}

void test_range_for_iteration_order() {
    pstree::ElementKey key(30);
    for (std::size_t i = 0; i < 30; ++i) key[i] = static_cast<std::uint16_t>(i * 2);
    std::size_t idx = 0;
    for (std::uint16_t v : key) {
        require(v == static_cast<std::uint16_t>(idx * 2), "range-for element " + std::to_string(idx) + " in order");
        ++idx;
    }
    require(idx == 30, "range-for should visit exactly 30 elements");
}

// The one place ElementKey needs to work correctly as an unordered_map key - PSTree's own
// equality_ hashmap (ps_tree.hpp). Two independently-constructed keys with identical content
// must hash-and-compare as the same key.
void test_usable_as_hashmap_key() {
    struct Hash {
        std::size_t operator()(const pstree::ElementKey& k) const noexcept {
            std::size_t h = k.size();
            for (auto e : k) h ^= std::hash<std::uint16_t>{}(e) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
    std::unordered_map<pstree::ElementKey, int, Hash> map;
    pstree::ElementKey k1{5, 6, 7};
    map[k1] = 42;

    pstree::ElementKey k2{5, 6, 7}; // separately constructed, same content
    auto it = map.find(k2);
    require(it != map.end() && it->second == 42, "a separately-constructed equal key should find the same map entry");

    pstree::ElementKey k3{5, 6, 8};
    require(map.find(k3) == map.end(), "a different key should not be found");
}

// Interleaved construction/copy/move cycles across the inline/heap boundary, broad coverage
// under ASan/UBSan for anything the more targeted tests above didn't happen to hit.
void test_mixed_operations_stress() {
    std::vector<pstree::ElementKey> keys;
    for (int round = 0; round < 40; ++round) {
        std::size_t n = static_cast<std::size_t>(round % 24); // sweeps across and past kInlineCapacity=16
        pstree::ElementKey key(n);
        for (std::size_t i = 0; i < n; ++i) key[i] = static_cast<std::uint16_t>(round * 100 + static_cast<int>(i));
        require(key.size() == n, "stress round " + std::to_string(round) + ": size correct");

        if (round % 3 == 0) {
            pstree::ElementKey copy = key;
            keys.push_back(std::move(copy));
        } else if (round % 3 == 1) {
            keys.push_back(key);
        } else {
            keys.push_back(std::move(key));
        }
    }
    require(keys.size() == 40, "stress: all 40 rounds landed in the vector");
    for (std::size_t round = 0; round < keys.size(); ++round) {
        std::size_t expectedN = round % 24;
        require(keys[round].size() == expectedN,
                "stress: stored key " + std::to_string(round) + " has expected size after copy/move");
        for (std::size_t i = 0; i < expectedN; ++i) {
            require(keys[round][i] == static_cast<std::uint16_t>(round * 100 + static_cast<int>(i)),
                    "stress: stored key " + std::to_string(round) + " element " + std::to_string(i) + " intact");
        }
    }
}

} // namespace

int main() {
    test_default_construction();
    test_sized_construction_is_zero_filled();
    test_initializer_list_construction();
    test_inline_heap_boundary();
    test_copy_independence_inline_and_heap();
    test_copy_assignment_including_size_transitions();
    test_move_construction_and_assignment();
    test_at_bounds_checking();
    test_equality();
    test_range_for_iteration_order();
    test_usable_as_hashmap_key();
    test_mixed_operations_stress();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All ElementKey tests passed\n";
    return 0;
}
