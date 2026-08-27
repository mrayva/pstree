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

} // namespace

int main() {
    test_int64_round_trip_and_order();
    test_double_round_trip_and_order();
    test_bool_codec();
    test_string_codec();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All order_key tests passed\n";
    return 0;
}
