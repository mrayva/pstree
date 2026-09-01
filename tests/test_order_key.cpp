// Correctness tests for order_key.hpp's per-type encoders: round-trip (decode(encode(v))
// == v) and order-preservation (a < b in the real domain iff encode(a) < encode(b)
// lexicographically as element sequences), across boundary values and a spread of normal
// ones. These properties are exactly what PS-Tree's leaf `next` chain and boundary logic
// depend on being true - if either broke, PS-Tree's own tests would fail in confusing
// ways far from the actual bug, so they're pinned here directly.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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

bool lexLess(const pstree::ElementKey& a, const pstree::ElementKey& b) {
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) return a[i] < b[i];
    }
    return false;
}

void test_int64_round_trip_and_order() {
    std::vector<std::int64_t> values = {
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::min() + 1,
        -1000000000LL, -5, -1, 0, 1, 5, 1000000000LL,
        std::numeric_limits<std::int64_t>::max() - 1,
        std::numeric_limits<std::int64_t>::max(),
    };
    for (auto v : values) {
        auto key = pstree::Int64Codec::encode(v);
        require(pstree::Int64Codec::decode(key) == v, "int64 round-trip for " + std::to_string(v));
    }
    for (std::size_t i = 0; i + 1 < values.size(); ++i) {
        auto ka = pstree::Int64Codec::encode(values[i]);
        auto kb = pstree::Int64Codec::encode(values[i + 1]);
        require(lexLess(ka, kb), "int64 order " + std::to_string(values[i]) + " < " + std::to_string(values[i + 1]));
    }
}

void test_double_round_trip_and_order() {
    std::vector<double> values = {
        -1e300, -1e9, -5.5, -1.0, -0.0001, 0.0, 0.0001, 1.0, 5.5, 1e9, 1e300,
    };
    for (auto v : values) {
        auto key = pstree::DoubleCodec::encode(v);
        require(pstree::DoubleCodec::decode(key) == v, "double round-trip for " + std::to_string(v));
    }
    for (std::size_t i = 0; i + 1 < values.size(); ++i) {
        auto ka = pstree::DoubleCodec::encode(values[i]);
        auto kb = pstree::DoubleCodec::encode(values[i + 1]);
        require(lexLess(ka, kb), "double order " + std::to_string(values[i]) + " < " + std::to_string(values[i + 1]));
    }
}

void test_bool_codec() {
    require(pstree::BoolCodec::decode(pstree::BoolCodec::encode(false)) == false, "bool round-trip false");
    require(pstree::BoolCodec::decode(pstree::BoolCodec::encode(true)) == true, "bool round-trip true");
    require(lexLess(pstree::BoolCodec::encode(false), pstree::BoolCodec::encode(true)), "bool order false < true");
}

void test_string_codec() {
    pstree::StringCodec codec(8);
    require(codec.decode(codec.encode("hello")) == "hello", "string round-trip 'hello'");
    require(codec.decode(codec.encode("")) == "", "string round-trip empty");
    require(codec.decode(codec.encode("12345678")) == "12345678", "string round-trip exact-length");
    require(codec.decode(codec.encode("123456789extra")) == "12345678", "string truncation past max_length");

    // Lexicographic ordering, including the "shorter prefix sorts first" case that only
    // works correctly because the end-of-string sentinel sorts below every real byte.
    require(lexLess(codec.encode("abc"), codec.encode("abd")), "string order abc < abd");
    require(lexLess(codec.encode("abc"), codec.encode("abcd")), "string order abc < abcd (prefix)");
    require(lexLess(codec.encode("ab"), codec.encode("abc")), "string order ab < abc (prefix)");
    require(!lexLess(codec.encode("abc"), codec.encode("abc")), "string order abc !< abc (equal)");
}

// -0.0 and +0.0 compare equal under IEEE-754 but have different raw bit patterns - without
// DoubleCodec's explicit normalization they'd encode to different keys, silently breaking
// matches between a predicate inserted at one and a query at the other.
void test_double_negative_zero() {
    require(pstree::DoubleCodec::encode(-0.0) == pstree::DoubleCodec::encode(0.0),
            "-0.0 and +0.0 must encode identically");
}

// NaN has no total order - encoding it would silently place it somewhere arbitrary and
// wrong rather than fail, so it's rejected outright.
void test_double_nan_rejected() {
    bool threw = false;
    try {
        pstree::DoubleCodec::encode(std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "encoding NaN should throw std::invalid_argument");
}

// Int256: same round-trip/order properties as int64/double, but exercised specifically at
// multi-limb boundaries (values differing only in the lowest limb vs. only in the highest limb,
// and the carry/borrow cases below) - a single-limb type has no equivalent of these, and
// Int256Codec's own multi-limb chunk256/unchunk256 (order_key.hpp) is new code with no prior
// coverage from int64/double's own tests.
void test_int256_round_trip_and_order() {
    auto mk = [](std::uint64_t l0, std::uint64_t l1, std::uint64_t l2, std::uint64_t l3) {
        pstree::Int256 v;
        v.limb = {l0, l1, l2, l3};
        return v;
    };
    const std::uint64_t kSignBit = std::uint64_t{1} << 63;
    std::vector<pstree::Int256> values = {
        mk(0, 0, 0, kSignBit),                                          // most negative representable
        mk(1, 0, 0, kSignBit),                                          // most negative + 1
        mk(~std::uint64_t{0}, ~std::uint64_t{0}, ~std::uint64_t{0}, ~std::uint64_t{0}), // -1
        mk(0, 0, 0, 0),                                                 // 0
        mk(1, 0, 0, 0),                                                 // 1
        mk(5, 0, 0, 0),                                                 // differs from next only in limb[0]
        mk(6, 0, 0, 0),
        mk(0, 0, 0, 5),                                                 // differs from next only in limb[3] (high limb)
        mk(0, 0, 0, 6),
        mk(~std::uint64_t{0}, ~std::uint64_t{0}, ~std::uint64_t{0}, std::uint64_t{0x7FFFFFFFFFFFFFFF}), // max
    };
    for (const auto& v : values) {
        auto key = pstree::Int256Codec::encode(v);
        require(pstree::Int256Codec::decode(key) == v, "Int256 round-trip");
    }
    // Order property against a manually-sorted subset spanning the interesting boundaries.
    std::vector<pstree::Int256> ordered = {
        mk(0, 0, 0, kSignBit),                                          // min
        mk(1, 0, 0, kSignBit),                                          // min + 1
        mk(~std::uint64_t{0}, ~std::uint64_t{0}, ~std::uint64_t{0}, ~std::uint64_t{0}), // -1
        mk(0, 0, 0, 0),                                                 // 0
        mk(1, 0, 0, 0),                                                 // 1
        mk(5, 0, 0, 0),
        mk(6, 0, 0, 0),
        mk(0, 0, 0, 5),                                                 // >> 6*2^64 range, far larger than limb[0]=6
        mk(0, 0, 0, 6),
        mk(~std::uint64_t{0}, ~std::uint64_t{0}, ~std::uint64_t{0}, std::uint64_t{0x7FFFFFFFFFFFFFFF}), // max
    };
    require(ordered[0] < ordered[1], "Int256 native operator< sanity: min < min+1");
    for (std::size_t i = 0; i + 1 < ordered.size(); ++i) {
        require(ordered[i] < ordered[i + 1], "Int256 native operator< monotonic at index " + std::to_string(i));
        auto ka = pstree::Int256Codec::encode(ordered[i]);
        auto kb = pstree::Int256Codec::encode(ordered[i + 1]);
        require(lexLess(ka, kb), "Int256 encoded order at index " + std::to_string(i));
    }
}

// Carry/borrow propagation across ALL THREE multi-limb boundaries (limb[0]->limb[1],
// limb[1]->limb[2], limb[2]->limb[3]) - a single-limb type only ever has one boundary
// (the sign-changing one, already covered for int64 above); Int256 has three additional ones
// nextElementKey/prevElementKey's own generic mixed-radix stepping must get right.
void test_int256_multilimb_carry_borrow() {
    auto mk = [](std::uint64_t l0, std::uint64_t l1, std::uint64_t l2, std::uint64_t l3) {
        pstree::Int256 v;
        v.limb = {l0, l1, l2, l3};
        return v;
    };
    const std::uint64_t kMaxLimb = ~std::uint64_t{0};
    struct Boundary { pstree::Int256 v, expectedNext, expectedPrevOfNext; const char* label; };
    std::vector<Boundary> boundaries = {
        {mk(kMaxLimb, 0, 0, 0), mk(0, 1, 0, 0), mk(kMaxLimb, 0, 0, 0), "limb0->limb1 carry"},
        {mk(kMaxLimb, kMaxLimb, 0, 0), mk(0, 0, 1, 0), mk(kMaxLimb, kMaxLimb, 0, 0), "limb1->limb2 carry"},
        {mk(kMaxLimb, kMaxLimb, kMaxLimb, 0), mk(0, 0, 0, 1), mk(kMaxLimb, kMaxLimb, kMaxLimb, 0), "limb2->limb3 carry"},
    };
    for (const auto& b : boundaries) {
        auto key = pstree::Int256Codec::encode(b.v);
        auto next = pstree::nextElementKey(pstree::Int256Codec::shape(), key);
        require(next.has_value(), std::string("next() should exist: ") + b.label);
        require(pstree::Int256Codec::decode(*next) == b.expectedNext,
                std::string("next() carries correctly: ") + b.label);
        auto backAgain = pstree::prevElementKey(pstree::Int256Codec::shape(), *next);
        require(backAgain.has_value(), std::string("prev(next()) should exist: ") + b.label);
        require(pstree::Int256Codec::decode(*backAgain) == b.expectedPrevOfNext,
                std::string("prev(next()) borrows back correctly: ") + b.label);
    }
    // True overflow/underflow at the representable extremes, same shape as int64's own test.
    {
        pstree::Int256 maxVal = mk(kMaxLimb, kMaxLimb, kMaxLimb, std::uint64_t{0x7FFFFFFFFFFFFFFF});
        auto kMax = pstree::Int256Codec::encode(maxVal);
        require(!pstree::nextElementKey(pstree::Int256Codec::shape(), kMax).has_value(),
                "next(Int256 max) should not exist");
        pstree::Int256 minVal = mk(0, 0, 0, std::uint64_t{1} << 63);
        auto kMin = pstree::Int256Codec::encode(minVal);
        require(!pstree::prevElementKey(pstree::Int256Codec::shape(), kMin).has_value(),
                "prev(Int256 min) should not exist");
    }
}

// nextElementKey/prevElementKey: ordinary mixed-radix increment/decrement, generic across
// every codec's own KeyShape - exercised directly here (rather than only indirectly via
// PS-Tree's kGt/kLt) so a bug in the stepping logic itself isn't masked by tree behavior.
void test_adjacent_key_stepping() {
    // No carry needed: last digit has room to increment/decrement.
    {
        auto k = pstree::Int64Codec::encode(5);
        auto next = pstree::nextElementKey(pstree::Int64Codec::shape(), k);
        require(next.has_value(), "next(5) should exist");
        require(pstree::Int64Codec::decode(*next) == 6, "next(5) should decode to 6");

        auto prev = pstree::prevElementKey(pstree::Int64Codec::shape(), k);
        require(prev.has_value(), "prev(5) should exist");
        require(pstree::Int64Codec::decode(*prev) == 4, "prev(5) should decode to 4");
    }
    // Carry propagation across a sign-changing boundary (-1 -> 0 -> 1).
    {
        auto k = pstree::Int64Codec::encode(-1);
        auto next = pstree::nextElementKey(pstree::Int64Codec::shape(), k);
        require(next.has_value() && pstree::Int64Codec::decode(*next) == 0, "next(-1) should decode to 0");
        auto prev = pstree::prevElementKey(pstree::Int64Codec::shape(), k);
        require(prev.has_value() && pstree::Int64Codec::decode(*prev) == -2, "prev(-1) should decode to -2");
    }
    // True overflow/underflow at the representable extremes.
    {
        auto kMax = pstree::Int64Codec::encode(std::numeric_limits<std::int64_t>::max());
        require(!pstree::nextElementKey(pstree::Int64Codec::shape(), kMax).has_value(),
                "next(INT64_MAX) should not exist");
        auto kMin = pstree::Int64Codec::encode(std::numeric_limits<std::int64_t>::min());
        require(!pstree::prevElementKey(pstree::Int64Codec::shape(), kMin).has_value(),
                "prev(INT64_MIN) should not exist");
    }
    // Bool: radix 2, so next(false)=true exists, next(true) does not.
    {
        auto kFalse = pstree::BoolCodec::encode(false);
        auto next = pstree::nextElementKey(pstree::BoolCodec::shape(), kFalse);
        require(next.has_value() && pstree::BoolCodec::decode(*next) == true, "next(false) should be true");
        auto kTrue = pstree::BoolCodec::encode(true);
        require(!pstree::nextElementKey(pstree::BoolCodec::shape(), kTrue).has_value(),
                "next(true) should not exist");
    }
}

} // namespace

int main() {
    test_int64_round_trip_and_order();
    test_double_round_trip_and_order();
    test_bool_codec();
    test_string_codec();
    test_double_negative_zero();
    test_double_nan_rejected();
    test_int256_round_trip_and_order();
    test_int256_multilimb_carry_borrow();
    test_adjacent_key_stepping();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All order_key tests passed\n";
    return 0;
}
