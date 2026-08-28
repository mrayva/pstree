#pragma once

// PS-Tree (Predicate Space Tree) - originally a direct transcription of the paper's own
// Algorithms 1-3 (leaf-chain boundary partitioning: a range predicate materializes coverage
// on every leaf between its boundary and the domain's edge). That approach is SUPERSEDED here
// for range operators (kGe/kGt/kLe/kLt/kIn) - not a transcription of the paper anymore for
// those - because it has a real, measured scaling problem: inserting a wide, unbounded range
// predicate ("price >= X" with X near the low end of the domain) is O(number of leaves
// currently covered), which grows with how many OTHER predicates have already been inserted
// on the same dimension (more predicates -> a more finely-partitioned, longer leaf chain).
// Measured via mrayva/nats_sidecar's own matching-engine benchmark at K=10,000 synthetic
// subscriptions (30% of which are a bare "price > X" with no other predicate, forcing it to
// be the access predicate): insert throughput fell to ~1.4% of a competing engine's, search
// throughput to ~15% - both degrading far worse than linearly with K.
//
// REPLACEMENT DESIGN: canonical ancestor markers on the existing digit-trie (the classical
// "canonical decomposition" technique for stabbing queries - the multiway/radix
// generalization of a segment tree - applied to the same fixed-length ElementKey digit
// sequence order_key.hpp already produces, most-significant digit first).
//
//   - Equality (kEq) moves to a plain hashmap (ElementKey -> LeafNode*) - it never had the
//     scaling problem (a point predicate is already O(1)) and doesn't need any trie walk.
//   - Range operators attach to O(depth) ANCESTOR nodes instead of O(leaves-covered) leaves.
//     Each InnerNode gains `geMarkers`/`leMarkers` (sorted threshold -> bucket maps - since the
//     encoding is most-significant-digit-first, a digit strictly exceeding a threshold at an
//     earlier level guarantees `>=` regardless of deeper digits, which is exactly why an
//     intermediate-level marker is always a pure prefix/suffix range, never an arbitrary
//     interval) plus a small `boundedMarkers` vector for the one genuinely-bounded case (kIn's
//     single "gap between lo's and hi's subtrees" marker - see insertBetween()).
//   - InsertPredicate(">=V"): walk V's digit path from the root (creating InnerNodes as
//     needed). At every level except the last: if V[level]+1 < radix[level], get-or-create a
//     geMarkers bucket at threshold V[level]+1 on the current node (marks every sibling
//     strictly greater than V[level] as unconditionally covered) and descend into child
//     V[level] to keep refining the "still exactly matches V" case. At the last level:
//     get-or-create a geMarkers bucket at threshold V[depth-1] itself (inclusive - no deeper
//     level to refine further). "<=V" is the exact mirror (leMarkers, guarded by
//     `V[level] > 0` before computing threshold V[level]-1 - `threshold` is uint16_t, and
//     omitting this guard when V[level]==0 would underflow to 65535 and silently corrupt
//     leMarkers). kGt/kLt keep the existing next/prevElementKey normalization to an adjacent
//     key, then kGe/kLe, unchanged.
//   - InsertPredicate(BETWEEN[lo,hi], kIn): lo==hi is a REQUIRED base case (not an
//     optimization) routed to the same equality hashmap as kEq - if lo==hi there is no level
//     where their digits diverge, so "the first level where they differ" is undefined for the
//     general algorithm below. Otherwise: find `lcaLevel` (first level where lo/hi digits
//     diverge); levels before it just descend together (no marker - the whole range agrees on
//     that shared prefix). If lcaLevel is the LAST level (depth-1), there is no deeper level to
//     refine into at all, so the whole range collapses to ONE inclusive boundedMarkers entry
//     [lo[lcaLevel], hi[lcaLevel]] at the lca node itself (a distinct, necessary case from the
//     one below - descending into "child loD" here would create a node with nothing correct to
//     attach to it, since geFrom/leFrom would immediately be called with startLevel==depth and
//     do nothing). Otherwise (lcaLevel < depth-1): one boundedMarkers entry
//     [lo[lcaLevel]+1, hi[lcaLevel]-1] on the shared node (if non-empty) covers everything
//     strictly between lo's and hi's subtrees as a whole; lo's own branch (child lo[lcaLevel])
//     gets the same GE-style decomposition rooted one level deeper (bounded above by hi
//     automatically, since it's entirely inside the lca's lo-side child); hi's branch
//     (child hi[lcaLevel]) gets the LE-style decomposition, symmetrically.
//   - MatchPoint(val): hashmap lookup for exact equality, plus a walk of val's digit path
//     collecting every geMarkers entry with threshold <= val[level] and every leMarkers entry
//     with threshold >= val[level] (both a std::map prefix/suffix scan via upper_bound/
//     lower_bound, not a linear scan) and any boundedMarkers match (linear scan - see the
//     residual-debt note below) at every node visited; stops descending once p[val[level]]
//     doesn't exist (nothing deeper could apply on this exact path - no node was ever created
//     past a point unless some insertion's own path needed it, and this query isn't following
//     any such insertion's path past this point).
//   - DeletePredicate re-runs the identical deterministic walk to find the same buckets and
//     decrement predCounter, throwing (not silently underflowing an unsigned counter) if the
//     expected bucket or inner-node path doesn't exist - deleting a predicate that was never
//     inserted, or double-deleting one. Buckets are DELIBERATELY NOT freed when predCounter
//     reaches zero (unlike an earlier draft of this design, which tried to - see the note on
//     destroy() below for why that's unsafe): PSTDynamic's own per-bucket group state
//     (LeafNode::userData) needs to still be readable/mutable by the CALLER after
//     insertPredicate/deletePredicate returns, and a caller only reads/updates a bucket's
//     userData strictly after seeing it in the returned list - freeing a zero-counter bucket
//     eagerly, inside this call, would leave that access reading freed memory. A zero-counter
//     "zombie" bucket costs a small, bounded amount of memory (mirroring the ORIGINAL leaf-
//     chain design's own deferred "MergeSpaces" stance - see git history - just for a different
//     underlying reason) and is otherwise harmless: MatchEvent finds it via matchPoint() same as
//     any other bucket, and an empty/all-decremented userData structure naturally contributes
//     nothing to a match.
//
// Complexity after this redesign: range insert/delete is O(depth) - a small per-attribute-type
// constant (16 for int64/double, <=128 for string, 1 for bool - see order_key.hpp), independent
// of how many other predicates exist on that dimension. Point query is
// O(depth * log(markers per node) + result-size) instead of exactly O(1), but no longer grows
// with the number of previously-inserted predicates on the wide-range side. Equality stays O(1).
//
// RESIDUAL DEBT, not silently forgotten: `boundedMarkers`' linear scan (both at insert/delete,
// finding-or-creating a matching (lo,hi) pair, and at query time, checking which contain a
// point) is NOT asymptotically optimized - if many kIn/BETWEEN predicates ever shared the same
// lca node, that node's scan would reintroduce an O(K)-shaped cost for that one operator
// specifically. This is out of scope for the fix here because kIn/BETWEEN is (a) not the
// operator the benchmark above found expensive (bare "price > X" access predicates are kGe, not
// kIn) and (b) not reachable at all from mrayva/nats_sidecar's own dialect translation today
// (confirmed: no kBetween references in that repo's pstree_dialect.{hpp,cpp}). If kIn ever
// becomes reachable or performance-critical, this is the place that would need its own
// interval-augmented structure (the same canonical-decomposition idea, one level deeper).
//
// `p[]` (child links) is a std::map<uint16_t,InnerNode*> uniformly for now (a dense
// vector-per-radix hybrid for the small-radix numeric types, keeping the sparse map only for
// the string codec's radix-257 nodes, is real follow-up work - not required for correctness or
// for fixing the measured problem, since even a plain sorted map is already O(log(children at
// this node)) instead of the OLD design's O(leaves) blowup).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pstree/order_key.hpp"

namespace pstree {

// kGe/kEq/kLe/kIn are the paper's own pseudocode operators. kGt/kLt are supported by
// normalizing to kGe/kLe at an adjacent key (order_key.hpp's nextElementKey/prevElementKey),
// exactly as before this redesign - that normalization was never part of the scaling problem
// and needed no changes.
enum class Op {
    kGe, // >=
    kGt, // >
    kEq, // ==
    kLt, // <
    kLe, // <=
    kIn, // BETWEEN [lo, hi], both endpoints inclusive
};

// A predicate-space "bucket": how many currently-inserted low-level Predicates cover this
// exact space, plus an opaque slot for an upper layer's own per-bucket data (PSTDynamic
// attaches its dimension-signature group state here - see pst_dynamic.hpp). Not interpreted
// by PSTree itself. Every bucket is owned by exactly ONE container slot (the equality hashmap,
// or one specific InnerNode's geMarkers/leMarkers/boundedMarkers entry) - unlike the original
// leaf-chain design, buckets are never split or aliased across multiple owners, so there is no
// "which node currently points at this" back-reference bookkeeping needed anymore.
struct LeafNode {
    std::uint64_t predCounter = 0;
    void* userData = nullptr;
};

struct InnerNode {
    std::map<std::uint16_t, InnerNode*> p;                                        // child nodes, by digit
    std::map<std::uint16_t, LeafNode*> geMarkers;                                 // threshold -> bucket: "digit >= threshold" covers
    std::map<std::uint16_t, LeafNode*> leMarkers;                                 // threshold -> bucket: "digit <= threshold" covers
    std::vector<std::tuple<std::uint16_t, std::uint16_t, LeafNode*>> boundedMarkers; // {lo, hi, bucket}: "lo <= digit <= hi" covers
};

// A predicate's single insertion/deletion unit: one attribute (handled by the caller, one
// PSTree per dimension), one operator, one or two values (kIn uses both vals0/vals1 as
// [lo, hi]; every other op uses vals0 only). Unchanged by this redesign.
struct Predicate {
    Op op;
    ElementKey vals0;
    ElementKey vals1; // only meaningful for Op::kIn
};

namespace detail {
struct ElementKeyHash {
    std::size_t operator()(const ElementKey& k) const noexcept {
        std::size_t h = k.size();
        for (auto e : k) {
            h ^= std::hash<std::uint16_t>{}(e) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};
} // namespace detail

class PSTree {
public:
    // Invoked when a bucket is freed (tree destruction only - see the file-level comment on
    // why buckets are never freed early, on decrement). Defaults to a no-op; a PSTree
    // constructed without one behaves exactly as if userData didn't exist at all.
    using DestroyUserData = std::function<void(void*)>;

    explicit PSTree(KeyShape shape, DestroyUserData destroyUserData = {})
        : shape_(std::move(shape)), destroyUserData_(std::move(destroyUserData)) {
        root_ = new InnerNode();
    }

    ~PSTree() { destroy(); }

    PSTree(const PSTree&) = delete;
    PSTree& operator=(const PSTree&) = delete;

    // Returns every bucket now covering `pred` (predCounter already incremented on each) -
    // see the file-level comment for the per-operator decomposition. kGt/kLt at the largest/
    // smallest representable value (no adjacent key to normalize to) correctly match nothing,
    // returning an empty vector without touching the tree.
    std::vector<LeafNode*> insertPredicate(const Predicate& pred) {
        switch (pred.op) {
            case Op::kEq:
                return {insertEq(pred.vals0)};
            case Op::kGe:
                return insertGe(pred.vals0);
            case Op::kGt: {
                auto next = nextElementKey(shape_, pred.vals0);
                if (!next) return {};
                return insertGe(*next);
            }
            case Op::kLe:
                return insertLe(pred.vals0);
            case Op::kLt: {
                auto prev = prevElementKey(shape_, pred.vals0);
                if (!prev) return {};
                return insertLe(*prev);
            }
            case Op::kIn:
                return insertBetween(pred.vals0, pred.vals1);
        }
        return {};
    }

    // Locates every bucket whose predicate space covers `val`.
    std::vector<LeafNode*> matchPoint(const ElementKey& val) const {
        std::vector<LeafNode*> results;
        auto eqIt = equality_.find(val);
        if (eqIt != equality_.end()) results.push_back(eqIt->second);

        const InnerNode* node = root_;
        for (std::size_t level = 0; level < val.size() && node != nullptr; ++level) {
            std::uint16_t d = val[level];
            auto geEnd = node->geMarkers.upper_bound(d);
            for (auto it = node->geMarkers.begin(); it != geEnd; ++it) results.push_back(it->second);
            auto leBegin = node->leMarkers.lower_bound(d);
            for (auto it = leBegin; it != node->leMarkers.end(); ++it) results.push_back(it->second);
            for (auto& entry : node->boundedMarkers) {
                if (std::get<0>(entry) <= d && d <= std::get<1>(entry)) results.push_back(std::get<2>(entry));
            }
            auto childIt = node->p.find(d);
            node = (childIt != node->p.end()) ? childIt->second : nullptr;
        }
        return results;
    }

    // Decrements predCounter on every bucket `pred` covers (throwing, not silently
    // underflowing, if a bucket or inner-node path the deterministic decomposition expects to
    // find doesn't exist - deleting a predicate that was never inserted, or double-deleting
    // one). Returns every bucket touched, in the same shape insertPredicate would have
    // produced for the identical `pred`. Buckets are not freed here - see the file-level
    // comment; call reclaim() with the same `pred` once the caller is done reading every
    // returned bucket's userData, to actually free any that reached zero.
    std::vector<LeafNode*> deletePredicate(const Predicate& pred) {
        switch (pred.op) {
            case Op::kEq:
                return {deleteEq(pred.vals0)};
            case Op::kGe:
                return deleteGe(pred.vals0);
            case Op::kGt: {
                auto next = nextElementKey(shape_, pred.vals0);
                if (!next) return {};
                return deleteGe(*next);
            }
            case Op::kLe:
                return deleteLe(pred.vals0);
            case Op::kLt: {
                auto prev = prevElementKey(shape_, pred.vals0);
                if (!prev) return {};
                return deleteLe(*prev);
            }
            case Op::kIn:
                return deleteBetween(pred.vals0, pred.vals1);
        }
        return {};
    }

    // Frees every zero-predCounter bucket `pred`'s decomposition touches, and prunes any
    // InnerNode left with no children and no markers as a result (walking back up along the
    // same path) - the actual space-reclamation half of what deletePredicate defers. See the
    // file-level comment's "zombie bucket" note for why this MUST be a separate call, invoked
    // only once the caller is completely done reading every bucket deletePredicate returned
    // (its own userData in particular) for THIS predicate - calling it any earlier is exactly
    // the use-after-free deletePredicate itself avoids by not freeing inline.
    //
    // Deliberately tolerant, unlike deletePredicate: re-walks the identical deterministic path,
    // but a missing bucket/child along the way just means "nothing left to reclaim here" (already
    // freed by an earlier reclaim() call, e.g. a sibling predicate from the same kElemOf
    // decomposition sharing a path prefix), not an error - safe to call multiple times, or on a
    // predicate whose buckets never accumulated any zero-counter garbage in the first place.
    void reclaim(const Predicate& pred) {
        switch (pred.op) {
            case Op::kEq:
                reclaimEq(pred.vals0);
                return;
            case Op::kGe:
                reclaimGeTop(pred.vals0);
                return;
            case Op::kGt: {
                auto next = nextElementKey(shape_, pred.vals0);
                if (next) reclaimGeTop(*next);
                return;
            }
            case Op::kLe:
                reclaimLeTop(pred.vals0);
                return;
            case Op::kLt: {
                auto prev = prevElementKey(shape_, pred.vals0);
                if (prev) reclaimLeTop(*prev);
                return;
            }
            case Op::kIn:
                reclaimBetween(pred.vals0, pred.vals1);
                return;
        }
    }

    const KeyShape& shape() const { return shape_; }
    InnerNode* root() const { return root_; }

private:
    KeyShape shape_;
    InnerNode* root_;
    DestroyUserData destroyUserData_;
    std::unordered_map<ElementKey, LeafNode*, detail::ElementKeyHash> equality_;

    std::uint32_t levelRadix(std::size_t level) const { return shape_.radix.at(level); }
    static LeafNode* newBucket() { return new LeafNode(); }

    LeafNode* insertEq(const ElementKey& v) {
        LeafNode*& bucket = equality_[v];
        if (bucket == nullptr) bucket = newBucket();
        bucket->predCounter++;
        return bucket;
    }

    LeafNode* deleteEq(const ElementKey& v) {
        auto it = equality_.find(v);
        if (it == equality_.end() || it->second->predCounter == 0) {
            throw std::logic_error(
                "pstree: DeletePredicate would underflow an equality bucket's predCounter - "
                "deleting a predicate that was never inserted, or double-deleting one");
        }
        it->second->predCounter -= 1;
        return it->second;
    }

    static LeafNode* decrementBucket(std::map<std::uint16_t, LeafNode*>& markers, std::uint16_t threshold) {
        auto it = markers.find(threshold);
        if (it == markers.end() || it->second->predCounter == 0) {
            throw std::logic_error(
                "pstree: DeletePredicate would underflow a bucket's predCounter - deleting a "
                "predicate that was never inserted, or double-deleting one");
        }
        it->second->predCounter -= 1;
        return it->second;
    }

    // GE-style decomposition rooted at an arbitrary (node, level) - used both for a top-level
    // ">=V" (rooted at the tree's actual root, level 0) and for kIn's lo-branch (rooted one
    // level past the lca node). See the file-level comment for the per-level guard/threshold
    // derivation.
    std::vector<LeafNode*> geFrom(InnerNode* startNode, const ElementKey& v, std::size_t startLevel) {
        std::vector<LeafNode*> result;
        InnerNode* node = startNode;
        const std::size_t depth = v.size();
        for (std::size_t level = startLevel; level < depth; ++level) {
            std::uint16_t d = v[level];
            if (level == depth - 1) {
                LeafNode*& bucket = node->geMarkers[d];
                if (bucket == nullptr) bucket = newBucket();
                bucket->predCounter++;
                result.push_back(bucket);
            } else {
                if (static_cast<std::uint32_t>(d) + 1 < levelRadix(level)) {
                    LeafNode*& bucket = node->geMarkers[static_cast<std::uint16_t>(d + 1)];
                    if (bucket == nullptr) bucket = newBucket();
                    bucket->predCounter++;
                    result.push_back(bucket);
                }
                InnerNode*& child = node->p[d];
                if (child == nullptr) child = new InnerNode();
                node = child;
            }
        }
        return result;
    }

    // LE-style mirror of geFrom - guarded by `d > 0` (not `d + 1 < radix`), since it marks
    // siblings strictly LESS than d, and threshold is unsigned (d==0 has no valid "d-1").
    std::vector<LeafNode*> leFrom(InnerNode* startNode, const ElementKey& v, std::size_t startLevel) {
        std::vector<LeafNode*> result;
        InnerNode* node = startNode;
        const std::size_t depth = v.size();
        for (std::size_t level = startLevel; level < depth; ++level) {
            std::uint16_t d = v[level];
            if (level == depth - 1) {
                LeafNode*& bucket = node->leMarkers[d];
                if (bucket == nullptr) bucket = newBucket();
                bucket->predCounter++;
                result.push_back(bucket);
            } else {
                if (d > 0) {
                    LeafNode*& bucket = node->leMarkers[static_cast<std::uint16_t>(d - 1)];
                    if (bucket == nullptr) bucket = newBucket();
                    bucket->predCounter++;
                    result.push_back(bucket);
                }
                InnerNode*& child = node->p[d];
                if (child == nullptr) child = new InnerNode();
                node = child;
            }
        }
        return result;
    }

    // Delete-side mirrors of geFrom/leFrom: find (never create) the same path/buckets a
    // matching insert would have produced, decrementing instead of incrementing. Throws if the
    // deterministic walk expects a bucket or inner-node path that isn't there.
    std::vector<LeafNode*> geFromDelete(InnerNode* startNode, const ElementKey& v, std::size_t startLevel) {
        std::vector<LeafNode*> result;
        InnerNode* node = startNode;
        const std::size_t depth = v.size();
        for (std::size_t level = startLevel; level < depth; ++level) {
            std::uint16_t d = v[level];
            if (level == depth - 1) {
                result.push_back(decrementBucket(node->geMarkers, d));
            } else {
                if (static_cast<std::uint32_t>(d) + 1 < levelRadix(level)) {
                    result.push_back(decrementBucket(node->geMarkers, static_cast<std::uint16_t>(d + 1)));
                }
                auto it = node->p.find(d);
                if (it == node->p.end()) {
                    throw std::logic_error(
                        "pstree: DeletePredicate could not find the expected inner-node path - "
                        "deleting a predicate that was never inserted, or double-deleting one");
                }
                node = it->second;
            }
        }
        return result;
    }

    std::vector<LeafNode*> leFromDelete(InnerNode* startNode, const ElementKey& v, std::size_t startLevel) {
        std::vector<LeafNode*> result;
        InnerNode* node = startNode;
        const std::size_t depth = v.size();
        for (std::size_t level = startLevel; level < depth; ++level) {
            std::uint16_t d = v[level];
            if (level == depth - 1) {
                result.push_back(decrementBucket(node->leMarkers, d));
            } else {
                if (d > 0) {
                    result.push_back(decrementBucket(node->leMarkers, static_cast<std::uint16_t>(d - 1)));
                }
                auto it = node->p.find(d);
                if (it == node->p.end()) {
                    throw std::logic_error(
                        "pstree: DeletePredicate could not find the expected inner-node path - "
                        "deleting a predicate that was never inserted, or double-deleting one");
                }
                node = it->second;
            }
        }
        return result;
    }

    std::vector<LeafNode*> insertGe(const ElementKey& v) { return geFrom(root_, v, 0); }
    std::vector<LeafNode*> insertLe(const ElementKey& v) { return leFrom(root_, v, 0); }
    std::vector<LeafNode*> deleteGe(const ElementKey& v) { return geFromDelete(root_, v, 0); }
    std::vector<LeafNode*> deleteLe(const ElementKey& v) { return leFromDelete(root_, v, 0); }

    static LeafNode* getOrCreateBoundedBucket(InnerNode* node, std::uint16_t lo, std::uint16_t hi) {
        for (auto& entry : node->boundedMarkers) {
            if (std::get<0>(entry) == lo && std::get<1>(entry) == hi) return std::get<2>(entry);
        }
        LeafNode* bucket = newBucket();
        node->boundedMarkers.emplace_back(lo, hi, bucket);
        return bucket;
    }

    static LeafNode* decrementBoundedBucket(InnerNode* node, std::uint16_t lo, std::uint16_t hi) {
        for (auto& entry : node->boundedMarkers) {
            if (std::get<0>(entry) == lo && std::get<1>(entry) == hi) {
                LeafNode* bucket = std::get<2>(entry);
                if (bucket->predCounter == 0) {
                    throw std::logic_error(
                        "pstree: DeletePredicate would underflow a bounded bucket's predCounter - "
                        "deleting a predicate that was never inserted, or double-deleting one");
                }
                bucket->predCounter -= 1;
                return bucket;
            }
        }
        throw std::logic_error(
            "pstree: DeletePredicate could not find the expected bounded marker (kIn) - "
            "deleting a predicate that was never inserted, or double-deleting one");
    }

    // BETWEEN[lo,hi] (kIn) - see the file-level comment for the full derivation, including why
    // lo==hi and "lca is the last level" are both required (not optional) base cases.
    std::vector<LeafNode*> insertBetween(const ElementKey& lo, const ElementKey& hi) {
        if (lo == hi) return {insertEq(lo)};
        const std::size_t depth = lo.size();
        std::size_t lcaLevel = 0;
        while (lcaLevel < depth && lo[lcaLevel] == hi[lcaLevel]) ++lcaLevel;

        InnerNode* node = root_;
        for (std::size_t level = 0; level < lcaLevel; ++level) {
            InnerNode*& child = node->p[lo[level]];
            if (child == nullptr) child = new InnerNode();
            node = child;
        }

        std::uint16_t loD = lo[lcaLevel];
        std::uint16_t hiD = hi[lcaLevel]; // hiD > loD guaranteed: shared prefix through lcaLevel-1, lo < hi overall

        if (lcaLevel == depth - 1) {
            // No deeper level exists to refine "digit == loD"/"digit == hiD" further - the
            // whole range collapses to one inclusive bounded marker at the lca node itself.
            LeafNode* bucket = getOrCreateBoundedBucket(node, loD, hiD);
            bucket->predCounter++;
            return {bucket};
        }

        std::vector<LeafNode*> result;
        if (static_cast<std::uint32_t>(loD) + 1 <= static_cast<std::uint32_t>(hiD) - 1) {
            LeafNode* bucket = getOrCreateBoundedBucket(
                node, static_cast<std::uint16_t>(loD + 1), static_cast<std::uint16_t>(hiD - 1));
            bucket->predCounter++;
            result.push_back(bucket);
        }

        InnerNode*& loChild = node->p[loD];
        if (loChild == nullptr) loChild = new InnerNode();
        auto loResults = geFrom(loChild, lo, lcaLevel + 1);
        result.insert(result.end(), loResults.begin(), loResults.end());

        InnerNode*& hiChild = node->p[hiD];
        if (hiChild == nullptr) hiChild = new InnerNode();
        auto hiResults = leFrom(hiChild, hi, lcaLevel + 1);
        result.insert(result.end(), hiResults.begin(), hiResults.end());

        return result;
    }

    std::vector<LeafNode*> deleteBetween(const ElementKey& lo, const ElementKey& hi) {
        if (lo == hi) return {deleteEq(lo)};
        const std::size_t depth = lo.size();
        std::size_t lcaLevel = 0;
        while (lcaLevel < depth && lo[lcaLevel] == hi[lcaLevel]) ++lcaLevel;

        InnerNode* node = root_;
        for (std::size_t level = 0; level < lcaLevel; ++level) {
            auto it = node->p.find(lo[level]);
            if (it == node->p.end()) {
                throw std::logic_error(
                    "pstree: DeletePredicate could not find the expected inner-node path (kIn) - "
                    "deleting a predicate that was never inserted, or double-deleting one");
            }
            node = it->second;
        }

        std::uint16_t loD = lo[lcaLevel];
        std::uint16_t hiD = hi[lcaLevel];

        if (lcaLevel == depth - 1) {
            return {decrementBoundedBucket(node, loD, hiD)};
        }

        std::vector<LeafNode*> result;
        if (static_cast<std::uint32_t>(loD) + 1 <= static_cast<std::uint32_t>(hiD) - 1) {
            result.push_back(decrementBoundedBucket(
                node, static_cast<std::uint16_t>(loD + 1), static_cast<std::uint16_t>(hiD - 1)));
        }

        auto loIt = node->p.find(loD);
        if (loIt == node->p.end()) {
            throw std::logic_error(
                "pstree: DeletePredicate could not find the expected lo-branch path (kIn) - "
                "deleting a predicate that was never inserted, or double-deleting one");
        }
        auto loResults = geFromDelete(loIt->second, lo, lcaLevel + 1);
        result.insert(result.end(), loResults.begin(), loResults.end());

        auto hiIt = node->p.find(hiD);
        if (hiIt == node->p.end()) {
            throw std::logic_error(
                "pstree: DeletePredicate could not find the expected hi-branch path (kIn) - "
                "deleting a predicate that was never inserted, or double-deleting one");
        }
        auto hiResults = leFromDelete(hiIt->second, hi, lcaLevel + 1);
        result.insert(result.end(), hiResults.begin(), hiResults.end());

        return result;
    }

    // Frees `bucket` (calling destroyUserData_ first) and erases it from `markers` iff its
    // predCounter has reached zero - i.e. no predicate, from this delete or any other still-live
    // one, references it anymore. A missing threshold is silently ignored (see reclaim()'s own
    // tolerance note).
    void freeMarkerIfZero(std::map<std::uint16_t, LeafNode*>& markers, std::uint16_t threshold) {
        auto it = markers.find(threshold);
        if (it == markers.end()) return;
        if (it->second->predCounter != 0) return;
        if (destroyUserData_) destroyUserData_(it->second->userData);
        delete it->second;
        markers.erase(it);
    }

    void freeBoundedMarkerIfZero(InnerNode* node, std::uint16_t lo, std::uint16_t hi) {
        for (auto it = node->boundedMarkers.begin(); it != node->boundedMarkers.end(); ++it) {
            if (std::get<0>(*it) != lo || std::get<1>(*it) != hi) continue;
            LeafNode* bucket = std::get<2>(*it);
            if (bucket->predCounter == 0) {
                if (destroyUserData_) destroyUserData_(bucket->userData);
                delete bucket;
                node->boundedMarkers.erase(it);
            }
            return;
        }
    }

    static bool nodeIsEmpty(const InnerNode* node) {
        return node->p.empty() && node->geMarkers.empty() && node->leMarkers.empty() &&
               node->boundedMarkers.empty();
    }

    void reclaimEq(const ElementKey& v) {
        auto it = equality_.find(v);
        if (it == equality_.end() || it->second->predCounter != 0) return;
        if (destroyUserData_) destroyUserData_(it->second->userData);
        delete it->second;
        equality_.erase(it);
    }

    // Mirrors geFrom's own top-down walk, but recurses to the deepest level FIRST so it can
    // prune a now-fully-empty child (no markers, no children of its own) on the way back up,
    // exactly like a normal post-order tree-node reclaim. Returns whether `node` itself ends up
    // empty afterward, so the CALLER (one level up, or reclaimGeTop for the root) can decide
    // whether the edge leading to `node` should be pruned too - root is never pruned (there is
    // no edge leading to it).
    bool reclaimGe(InnerNode* node, const ElementKey& v, std::size_t level) {
        const std::size_t depth = v.size();
        std::uint16_t d = v[level];
        if (level == depth - 1) {
            freeMarkerIfZero(node->geMarkers, d);
        } else {
            auto childIt = node->p.find(d);
            if (childIt != node->p.end()) {
                if (reclaimGe(childIt->second, v, level + 1)) {
                    delete childIt->second;
                    node->p.erase(childIt);
                }
            }
            if (static_cast<std::uint32_t>(d) + 1 < levelRadix(level)) {
                freeMarkerIfZero(node->geMarkers, static_cast<std::uint16_t>(d + 1));
            }
        }
        return nodeIsEmpty(node);
    }

    bool reclaimLe(InnerNode* node, const ElementKey& v, std::size_t level) {
        const std::size_t depth = v.size();
        std::uint16_t d = v[level];
        if (level == depth - 1) {
            freeMarkerIfZero(node->leMarkers, d);
        } else {
            auto childIt = node->p.find(d);
            if (childIt != node->p.end()) {
                if (reclaimLe(childIt->second, v, level + 1)) {
                    delete childIt->second;
                    node->p.erase(childIt);
                }
            }
            if (d > 0) {
                freeMarkerIfZero(node->leMarkers, static_cast<std::uint16_t>(d - 1));
            }
        }
        return nodeIsEmpty(node);
    }

    void reclaimGeTop(const ElementKey& v) { reclaimGe(root_, v, 0); }
    void reclaimLeTop(const ElementKey& v) { reclaimLe(root_, v, 0); }

    // BETWEEN's own mirror of reclaimGe/reclaimLe - see insertBetween/deleteBetween for the
    // lcaLevel derivation this follows exactly. `reclaimBetweenPrefix` walks the shared
    // lo[0..lcaLevel) descend-only prefix (no marker at those levels, same as insert/delete),
    // recursing to the lca node first so the prefix can be pruned bottom-up same as reclaimGe/Le.
    bool reclaimBetweenAtLca(InnerNode* node, const ElementKey& lo, const ElementKey& hi, std::size_t lcaLevel) {
        const std::size_t depth = lo.size();
        std::uint16_t loD = lo[lcaLevel];
        std::uint16_t hiD = hi[lcaLevel];

        if (lcaLevel == depth - 1) {
            freeBoundedMarkerIfZero(node, loD, hiD);
            return nodeIsEmpty(node);
        }

        if (static_cast<std::uint32_t>(loD) + 1 <= static_cast<std::uint32_t>(hiD) - 1) {
            freeBoundedMarkerIfZero(node, static_cast<std::uint16_t>(loD + 1), static_cast<std::uint16_t>(hiD - 1));
        }

        auto loIt = node->p.find(loD);
        if (loIt != node->p.end()) {
            if (reclaimGe(loIt->second, lo, lcaLevel + 1)) {
                delete loIt->second;
                node->p.erase(loIt);
            }
        }
        auto hiIt = node->p.find(hiD);
        if (hiIt != node->p.end()) {
            if (reclaimLe(hiIt->second, hi, lcaLevel + 1)) {
                delete hiIt->second;
                node->p.erase(hiIt);
            }
        }
        return nodeIsEmpty(node);
    }

    bool reclaimBetweenPrefix(InnerNode* node, const ElementKey& lo, const ElementKey& hi,
                               std::size_t level, std::size_t lcaLevel) {
        if (level == lcaLevel) {
            return reclaimBetweenAtLca(node, lo, hi, lcaLevel);
        }
        std::uint16_t d = lo[level]; // == hi[level] below lcaLevel, by lcaLevel's own definition
        auto childIt = node->p.find(d);
        if (childIt != node->p.end()) {
            if (reclaimBetweenPrefix(childIt->second, lo, hi, level + 1, lcaLevel)) {
                delete childIt->second;
                node->p.erase(childIt);
            }
        }
        return nodeIsEmpty(node);
    }

    void reclaimBetween(const ElementKey& lo, const ElementKey& hi) {
        if (lo == hi) {
            reclaimEq(lo);
            return;
        }
        const std::size_t depth = lo.size();
        std::size_t lcaLevel = 0;
        while (lcaLevel < depth && lo[lcaLevel] == hi[lcaLevel]) ++lcaLevel;
        reclaimBetweenPrefix(root_, lo, hi, 0, lcaLevel);
    }

    void destroy() {
        for (auto& [key, bucket] : equality_) {
            if (destroyUserData_) destroyUserData_(bucket->userData);
            delete bucket;
        }
        equality_.clear();
        destroyInner(root_);
        root_ = nullptr;
    }

    void destroyInner(InnerNode* node) {
        if (node == nullptr) return;
        for (auto& [threshold, bucket] : node->geMarkers) {
            if (destroyUserData_) destroyUserData_(bucket->userData);
            delete bucket;
        }
        for (auto& [threshold, bucket] : node->leMarkers) {
            if (destroyUserData_) destroyUserData_(bucket->userData);
            delete bucket;
        }
        for (auto& entry : node->boundedMarkers) {
            if (destroyUserData_) destroyUserData_(std::get<2>(entry)->userData);
            delete std::get<2>(entry);
        }
        for (auto& [digit, child] : node->p) {
            destroyInner(child);
        }
        delete node;
    }
};

} // namespace pstree
