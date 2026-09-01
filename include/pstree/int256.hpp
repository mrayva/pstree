#pragma once

// Fixed-width 256-bit signed two's-complement integer - pstree's own minimal, dependency-free
// representation backing native DECIMAL32/64/128/256 support. Comparison only, no arithmetic
// beyond what Int256Codec's encoding needs (order_key.hpp) - rescaling a Postgres/Arrow decimal
// value to this attribute's canonical scale happens at the ingestion boundary (nats_sidecar),
// using Arrow's own richer decimal API (arrow::BasicDecimal256::Rescale); by the time a value
// becomes an Int256 here, it is already a raw, canonically-scaled integer magnitude with no
// scale/precision tag traveling alongside it (see pst_dynamic.hpp's AttrSchema::decimalScale for
// where that lives instead - once per attribute, not once per value).
//
// Deliberately pstree-owned rather than arrow::BasicDecimal256 directly: BasicDecimal256's own
// comparison/rescale methods are ARROW_EXPORT (compiled into libarrow, not header-only), so using
// it here would mean this otherwise dependency-free library takes on a real new external link
// dependency for a handful of operators it can implement itself in well under 100 lines, mirroring
// DoubleCodec's own well-tested sign-bit-flip technique (order_key.hpp) just widened to 256 bits.

#include <array>
#include <cstdint>

namespace pstree {

struct Int256 {
    // limb[0] = least-significant 64 bits ... limb[3] = most-significant 64 bits (carries the
    // two's-complement sign in its own top bit). Value = sum(limb[i] << (64*i)).
    std::array<std::uint64_t, 4> limb{};

    friend bool operator==(const Int256& a, const Int256& b) noexcept { return a.limb == b.limb; }
    friend bool operator!=(const Int256& a, const Int256& b) noexcept { return !(a == b); }

    // Signed multi-limb comparison: the most-significant limb carries the overall two's-
    // complement sign, so it must be compared AS SIGNED first; once limbs agree there, every
    // remaining limb (moving from most to least significant) compares correctly as UNSIGNED -
    // the same "sign bit decides, then raw magnitude bits already sort correctly" property a
    // single 64-bit signed/unsigned comparison already relies on (see Int64Codec's own comment
    // in order_key.hpp), just extended across four limbs instead of one.
    friend bool operator<(const Int256& a, const Int256& b) noexcept {
        if (a.limb[3] != b.limb[3]) {
            return static_cast<std::int64_t>(a.limb[3]) < static_cast<std::int64_t>(b.limb[3]);
        }
        for (int i = 2; i >= 0; --i) {
            if (a.limb[i] != b.limb[i]) return a.limb[i] < b.limb[i];
        }
        return false;
    }
    friend bool operator>(const Int256& a, const Int256& b) noexcept { return b < a; }
    friend bool operator<=(const Int256& a, const Int256& b) noexcept { return !(b < a); }
    friend bool operator>=(const Int256& a, const Int256& b) noexcept { return !(a < b); }
};

} // namespace pstree
