#pragma once

// Dimension signature (paper Section 5.1): "A group has a dimension signature, which is in
// fact a Bloom filter. A subscription is assigned to a group if that group's dimension
// signature is equal to the result when we create a Bloom filter on that subscription's
// dimensions." Per Fig. 3's own worked example (a subscription's signature covers EVERY
// dimension it has a predicate on, including its own access predicate's dimension, not just
// the "other" ones - confirmed there: S1/S2 have predicates on {attr1,attr2}, signature
// "110"; S5/S6 have predicates on all three dimensions, signature "111").
//
// The paper gives no pseudocode for the Bloom filter itself (hash count, bit width, or the
// thresholds[] growth schedule) - Fig. 3's own signatures ("110"/"011"/"111") are exact
// 3-bit presence vectors, which only look like real Bloom filter output because the example
// has exactly 3 dimensions and a 3-bit filter (no real hash collisions possible at that
// size) - not something a real hash function's output can be reproduced byte-for-byte from
// the paper alone. This implementation is therefore verified against the WORKED EXAMPLE at
// the semantic level (which subscriptions land in the same vs. different groups), not by
// trying to reproduce its literal bit strings - see test_pst_dynamic.cpp.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

namespace pstree {

class DimSig {
public:
    explicit DimSig(std::size_t numBits) : numBits_(numBits), words_((numBits + 63) / 64, 0) {}

    std::size_t numBits() const { return numBits_; }

    // Standard Kirsch-Mitzenmacher double hashing: two independent-enough hashes from one
    // std::hash call (split high/low), combined linearly for `kNumHashes` derived hash
    // functions - avoids computing kNumHashes genuinely independent hash functions.
    void add(std::string_view dim) {
        std::uint64_t full = std::hash<std::string_view>{}(dim);
        std::uint32_t h1 = static_cast<std::uint32_t>(full);
        std::uint32_t h2 = static_cast<std::uint32_t>(full >> 32) | 1u; // odd, avoids h2=0 degenerating to one hash
        for (int i = 0; i < kNumHashes; ++i) {
            std::uint64_t combined = static_cast<std::uint64_t>(h1) + static_cast<std::uint64_t>(i) * h2;
            setBit(combined % numBits_);
        }
    }

    // True iff every bit set in `this` is also set in `other` - i.e. `this` is a
    // (Bloom-filter-approximate) subset of `other`. Used both ways in this codebase: a
    // group's own dimension signature must be a subset of an event's (Algorithm 5, line 8),
    // and two DimSigs of the SAME length are compared for exact equality (operator==) to
    // decide whether a subscription belongs to an existing group.
    bool isSubsetOf(const DimSig& other) const {
        std::size_t n = std::min(words_.size(), other.words_.size());
        for (std::size_t i = 0; i < n; ++i) {
            if ((words_[i] & ~other.words_[i]) != 0) return false;
        }
        // Any words beyond `other`'s length are implicitly all-zero on other's side - if
        // `this` has any bit set there, it can't be a subset.
        for (std::size_t i = n; i < words_.size(); ++i) {
            if (words_[i] != 0) return false;
        }
        return true;
    }

    bool operator==(const DimSig& other) const { return numBits_ == other.numBits_ && words_ == other.words_; }

    struct Hasher {
        std::size_t operator()(const DimSig& sig) const {
            std::size_t h = sig.numBits_;
            for (auto w : sig.words_) {
                h ^= std::hash<std::uint64_t>{}(w) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            }
            return h;
        }
    };

private:
    static constexpr int kNumHashes = 2;

    void setBit(std::size_t bit) { words_[bit / 64] |= (std::uint64_t{1} << (bit % 64)); }

    std::size_t numBits_;
    std::vector<std::uint64_t> words_;
};

// Builds a signature covering every dimension in `dims`, at the given bit width - the
// per-subscription (or per-event) signature computation Algorithm 4/5/6 call
// CalculateDimSig.
template <typename DimRange>
DimSig calculateDimSig(const DimRange& dims, std::size_t numBits) {
    DimSig sig(numBits);
    for (const auto& dim : dims) {
        sig.add(dim);
    }
    return sig;
}

} // namespace pstree
