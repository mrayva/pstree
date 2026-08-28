#pragma once

// Order-preserving element encoding for PS-Tree (paper Section 4.5/4.1, Fig. 2).
//
// The paper decomposes a value's bit pattern into "elements" (digits) so PS-Tree can
// walk one element per tree level. Their worked examples use an irregular split (byte:
// 1 sign bit + 3 bits + 4 bits; float32: 1 sign bit + 8-bit exponent split into 2 + 23-bit
// mantissa split into 6 = 9 elements) that is specific to their examples, not a
// requirement of the algorithm itself - InsertPredicate/MatchPair/DeletePredicate only
// need SOME fixed-depth, fixed-per-level-radix decomposition that preserves the value's
// total order (so a leaf's `next` chain walks values in increasing order) and round-trips
// bijectively (so re-inserting/deleting the same value always reaches the same leaf).
//
// This implementation instead uses the standard "sortable bit pattern" bijections (the
// same trick radix sort implementations use for signed integers and IEEE-754 floats):
// XOR the raw bit pattern with a fixed mask so the transformed pattern, compared as an
// UNSIGNED integer, is monotonic in the original value. That collapses int64_t and double
// (both 64-bit) onto one shared "uniform 4-bit chunk" decomposition (16 elements, radix 16
// each level) instead of replicating the paper's irregular per-type split - simpler to
// implement and reason about, and still isolates value-type/operator details from the
// upper PSTDynamic layer exactly as Section 4.5 intends.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pstree {

// One decomposed value: a fixed-length sequence of element indices, one per tree level,
// each in [0, radix_at_that_level). Small-vector-optimized: inline storage for up to
// kInlineCapacity elements, which covers int64_t/double (always exactly 16) and bool
// (always exactly 1) with ZERO heap allocation - only StringCodec's longer keys (up to its
// configured max_length, e.g. 128 in nats_sidecar's usage) spill to the heap, same as
// before. Found via `perf annotate` on nats_sidecar's matching_engine_bench (real NYSE
// fleet-scale run): every single MatchEvent call encodes its event value into a fresh
// ElementKey before any matching can happen at all, and that was a genuine, previously-
// unexamined per-row heap allocation (a plain std::vector<uint16_t>) - unlike a-tree/be-tree,
// which compare native values directly with no encoding step, PS-Tree's whole indexing
// design requires this encode on every single event, so it's paid regardless of how many
// subscriptions exist (see pst_dynamic.hpp's own architecture notes on when that tradeoff
// pays off vs. doesn't).
//
// Deliberately hand-specialized to uint16_t (not a reusable SmallVector<T,N> template), and
// simpler than PredicateList's own equivalent fix (pst_dynamic.hpp... see predicate.hpp):
// uint16_t is trivially copyable/destructible, so this type needs no placement-new/manual-
// destroy bookkeeping at all - plain memcpy suffices for every copy/move.
class ElementKey {
public:
    static constexpr std::size_t kInlineCapacity = 16;

    ElementKey() noexcept = default;

    // Sized construction: `n` zero-valued elements, matching std::vector<uint16_t>(n)'s own
    // value-initialization semantics - every real caller (chunk64, StringCodec::encode)
    // immediately overwrites every index anyway, but zero-initializing removes any chance of
    // a future caller reading an unwritten slot as UB.
    explicit ElementKey(std::size_t n) {
        reallocateDiscardingContents(n);
        std::fill(data(), data() + n, std::uint16_t{0});
    }

    ElementKey(std::initializer_list<std::uint16_t> init) {
        reallocateDiscardingContents(init.size());
        std::copy(init.begin(), init.end(), data());
    }

    ElementKey(const ElementKey& other) {
        reallocateDiscardingContents(other.size_);
        std::memcpy(data(), other.data(), other.size_ * sizeof(std::uint16_t));
    }
    ElementKey(ElementKey&& other) noexcept {
        if (other.heap_ != nullptr) {
            heap_ = other.heap_;
            heapCapacity_ = other.heapCapacity_;
            size_ = other.size_;
            other.heap_ = nullptr;
            other.heapCapacity_ = 0;
            other.size_ = 0;
        } else {
            size_ = other.size_;
            std::memcpy(inline_, other.inline_, other.size_ * sizeof(std::uint16_t));
            other.size_ = 0;
        }
    }

    ElementKey& operator=(const ElementKey& other) {
        if (this == &other) return *this;
        reallocateDiscardingContents(other.size_);
        std::memcpy(data(), other.data(), other.size_ * sizeof(std::uint16_t));
        return *this;
    }
    ElementKey& operator=(ElementKey&& other) noexcept {
        if (this == &other) return *this;
        if (heap_ != nullptr) {
            delete[] heap_;
            heap_ = nullptr;
            heapCapacity_ = 0;
        }
        if (other.heap_ != nullptr) {
            heap_ = other.heap_;
            heapCapacity_ = other.heapCapacity_;
            size_ = other.size_;
            other.heap_ = nullptr;
            other.heapCapacity_ = 0;
            other.size_ = 0;
        } else {
            size_ = other.size_;
            std::memcpy(inline_, other.inline_, other.size_ * sizeof(std::uint16_t));
            other.size_ = 0;
        }
        return *this;
    }

    ~ElementKey() {
        if (heap_ != nullptr) delete[] heap_;
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    std::uint16_t& operator[](std::size_t i) noexcept { return data()[i]; }
    std::uint16_t operator[](std::size_t i) const noexcept { return data()[i]; }
    std::uint16_t& at(std::size_t i) {
        if (i >= size_) throw std::out_of_range("pstree::ElementKey::at");
        return data()[i];
    }
    std::uint16_t at(std::size_t i) const {
        if (i >= size_) throw std::out_of_range("pstree::ElementKey::at");
        return data()[i];
    }

    std::uint16_t* begin() noexcept { return data(); }
    std::uint16_t* end() noexcept { return data() + size_; }
    const std::uint16_t* begin() const noexcept { return data(); }
    const std::uint16_t* end() const noexcept { return data() + size_; }

    bool operator==(const ElementKey& other) const noexcept {
        if (size_ != other.size_) return false;
        return std::memcmp(data(), other.data(), size_ * sizeof(std::uint16_t)) == 0;
    }
    bool operator!=(const ElementKey& other) const noexcept { return !(*this == other); }

private:
    // Raw inline storage - never placement-new'd/destroyed element-by-element, since
    // uint16_t is trivial; only ever touched via memcpy/std::fill on the range [0, size_).
    std::uint16_t inline_[kInlineCapacity];
    std::uint16_t* heap_ = nullptr; // non-null once spilled to the heap
    std::size_t heapCapacity_ = 0;  // element capacity of *heap_; meaningless while heap_ == nullptr
    std::size_t size_ = 0;

    std::uint16_t* data() noexcept { return heap_ != nullptr ? heap_ : inline_; }
    const std::uint16_t* data() const noexcept { return heap_ != nullptr ? heap_ : inline_; }

    // Ensures storage for exactly `n` elements is available and sets size_ = n, WITHOUT
    // preserving any prior contents - every caller (construction, copy-assignment)
    // immediately overwrites the whole range right after calling this, so there is nothing
    // to preserve. Frees an existing heap buffer whenever `n` fits inline (no reason to keep
    // a heap allocation alive for a key that's shrunk back into the common case), and only
    // reallocates the heap buffer when growing past the current one's capacity.
    void reallocateDiscardingContents(std::size_t n) {
        if (n <= kInlineCapacity) {
            if (heap_ != nullptr) {
                delete[] heap_;
                heap_ = nullptr;
                heapCapacity_ = 0;
            }
        } else if (heap_ == nullptr || heapCapacity_ < n) {
            auto* newBuf = new std::uint16_t[n];
            if (heap_ != nullptr) delete[] heap_;
            heap_ = newBuf;
            heapCapacity_ = n;
        }
        size_ = n;
    }
};

// Describes the shape of a value type's decomposition: how many elements (tree depth) and
// the branching factor (radix) at each level. Shared by every value of that type.
struct KeyShape {
    std::vector<std::uint32_t> radix; // radix[level] = number of distinct child links at that level
    std::size_t depth() const { return radix.size(); }
};

namespace detail {

// Splits a 64-bit unsigned "sortable key" into `chunks` elements of `bits_per_chunk` bits
// each, most-significant chunk first (so element order matches numeric order of the key).
inline ElementKey chunk64(std::uint64_t key, std::size_t chunks, std::size_t bits_per_chunk) {
    ElementKey out(chunks);
    for (std::size_t i = 0; i < chunks; ++i) {
        std::size_t shift = (chunks - 1 - i) * bits_per_chunk;
        std::uint64_t mask = (bits_per_chunk >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << bits_per_chunk) - 1);
        out[i] = static_cast<std::uint16_t>((key >> shift) & mask);
    }
    return out;
}

inline std::uint64_t unchunk64(const ElementKey& elems, std::size_t bits_per_chunk) {
    std::uint64_t key = 0;
    for (std::size_t i = 0; i < elems.size(); ++i) {
        key = (key << bits_per_chunk) | static_cast<std::uint64_t>(elems[i]);
    }
    return key;
}

} // namespace detail

// --- boolean: degenerate 1-element, 2-value case ---

struct BoolCodec {
    static KeyShape shape() { return KeyShape{{2}}; }
    static ElementKey encode(bool v) { return {static_cast<std::uint16_t>(v ? 1 : 0)}; }
    static bool decode(const ElementKey& e) { return e.at(0) != 0; }
};

// --- int64_t: sign-flip trick (two's complement -> monotonic unsigned), 16 x 4-bit elements ---

struct Int64Codec {
    static constexpr std::size_t kChunks = 16;
    static constexpr std::size_t kBitsPerChunk = 4;

    static KeyShape shape() { return KeyShape{std::vector<std::uint32_t>(kChunks, 1u << kBitsPerChunk)}; }

    static ElementKey encode(std::int64_t v) {
        // XOR the sign bit: negative numbers (top bit 1) become < positive numbers (top
        // bit 0) once viewed as unsigned, and two's complement ordering is already
        // monotonic within each sign, so only the sign bit itself needs flipping.
        std::uint64_t bits = static_cast<std::uint64_t>(v) ^ (std::uint64_t{1} << 63);
        return detail::chunk64(bits, kChunks, kBitsPerChunk);
    }

    static std::int64_t decode(const ElementKey& e) {
        std::uint64_t bits = detail::unchunk64(e, kBitsPerChunk) ^ (std::uint64_t{1} << 63);
        return static_cast<std::int64_t>(bits);
    }
};

// --- double: IEEE-754 sortable-bit-pattern trick, 16 x 4-bit elements ---

struct DoubleCodec {
    static constexpr std::size_t kChunks = 16;
    static constexpr std::size_t kBitsPerChunk = 4;

    static KeyShape shape() { return KeyShape{std::vector<std::uint32_t>(kChunks, 1u << kBitsPerChunk)}; }

    static ElementKey encode(double v) {
        // NaN has no well-defined position in a total order (IEEE-754 says NaN compares
        // unequal, unordered, to everything including itself) - PS-Tree's whole model
        // depends on a total order existing, so rather than silently encode NaN's bit
        // pattern as if it were an ordinary sortable value (which would put it somewhere
        // arbitrary and wrong), reject it clearly at the boundary.
        if (std::isnan(v)) {
            throw std::invalid_argument("pstree::DoubleCodec: NaN has no total order, cannot be encoded");
        }
        // -0.0 and +0.0 compare equal under IEEE-754 (v == 0.0 is true for both), but have
        // DIFFERENT bit patterns (sign bit differs) - without this normalization they'd
        // encode to different keys, so a predicate inserted against +0.0 would silently
        // fail to match a query for -0.0 despite being numerically identical values.
        if (v == 0.0) v = 0.0;
        std::uint64_t bits;
        static_assert(sizeof(bits) == sizeof(v));
        std::memcpy(&bits, &v, sizeof(bits));
        // Standard trick: if the sign bit is set (negative), flip all bits (so more
        // negative values, which have larger raw magnitude bits, sort first); if unset
        // (non-negative), flip only the sign bit (so positives sort after all negatives).
        std::uint64_t mask = (bits & (std::uint64_t{1} << 63)) ? ~std::uint64_t{0} : (std::uint64_t{1} << 63);
        return detail::chunk64(bits ^ mask, kChunks, kBitsPerChunk);
    }

    static double decode(const ElementKey& e) {
        std::uint64_t sortable = detail::unchunk64(e, kBitsPerChunk);
        std::uint64_t mask = (sortable & (std::uint64_t{1} << 63)) ? (std::uint64_t{1} << 63) : ~std::uint64_t{0};
        std::uint64_t bits = sortable ^ mask;
        double v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
};

// --- string: characters as elements directly, fixed max length, sentinel for "shorter than" ---
//
// Radix 257 per level: byte values 0-255 plus one reserved "end of string" sentinel
// (value 256) that sorts BEFORE every real byte, so a shorter string's element sequence
// (padded with the sentinel) correctly sorts before any longer string sharing its prefix,
// matching ordinary lexicographic string ordering. Strings longer than max_length are
// truncated at that bound (mirrors nats_sidecar's own existing bounded-string convention,
// e.g. kBetreeStringCount).
class StringCodec {
public:
    explicit StringCodec(std::size_t max_length) : max_length_(max_length) {}

    KeyShape shape() const { return KeyShape{std::vector<std::uint32_t>(max_length_, 257)}; }

    ElementKey encode(std::string_view v) const {
        ElementKey out(max_length_);
        for (std::size_t i = 0; i < max_length_; ++i) {
            // Real bytes are shifted up by one (1..256) so the end-of-string sentinel can
            // occupy 0, the smallest possible element value - this is what makes a
            // shorter string's padded key sort BEFORE a longer string sharing its prefix
            // (e.g. "ab" < "abc"), matching ordinary lexicographic string ordering.
            out[i] = (i < v.size())
                ? static_cast<std::uint16_t>(static_cast<unsigned char>(v[i]) + 1)
                : kEndOfString;
        }
        return out;
    }

    std::string decode(const ElementKey& e) const {
        std::string out;
        out.reserve(e.size());
        for (std::uint16_t elem : e) {
            if (elem == kEndOfString) break;
            out.push_back(static_cast<char>(elem - 1));
        }
        return out;
    }

private:
    static constexpr std::uint16_t kEndOfString = 0;
    std::size_t max_length_;
};

// Adjacent-key stepping: since every ElementKey is a fixed-length sequence of digits in a
// declared per-level radix (a mixed-radix number), "the next/previous representable value"
// is just ordinary mixed-radix increment/decrement with carry/borrow propagation from the
// LEAST significant (last) element - this works identically for every codec above with no
// type-specific logic, which is what lets PS-Tree support strict `>`/`<` by normalizing to
// `>=next(V)`/`<=prev(V)` at the operator layer (see ps_tree.hpp's Op::kGt/kLt) instead of
// needing its own separate tree-wiring for them. Returns nullopt on overflow/underflow -
// i.e. `>` at the largest representable value, or `<` at the smallest, matches nothing.
inline std::optional<ElementKey> nextElementKey(const KeyShape& shape, ElementKey key) {
    for (std::size_t i = key.size(); i-- > 0;) {
        if (static_cast<std::uint32_t>(key[i]) + 1 < shape.radix[i]) {
            key[i] += 1;
            return key;
        }
        key[i] = 0; // carry
    }
    return std::nullopt; // every digit was already at its maximum - no successor
}

inline std::optional<ElementKey> prevElementKey(const KeyShape& shape, ElementKey key) {
    for (std::size_t i = key.size(); i-- > 0;) {
        if (key[i] > 0) {
            key[i] -= 1;
            return key;
        }
        key[i] = static_cast<std::uint16_t>(shape.radix[i] - 1); // borrow
    }
    return std::nullopt; // every digit was already 0 - no predecessor
}

} // namespace pstree
