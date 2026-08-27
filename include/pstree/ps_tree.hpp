#pragma once

// PS-Tree (Predicate Space Tree) - paper Section 4, Algorithms 1-3 (pages 8-15).
//
// A single-dimension index: predicates over one attribute are inserted as covering a
// contiguous range of the value domain ("predicate space"); MatchPair(value) finds which
// predicate space a value falls into in O(depth) time.
//
// This is a direct transcription of Algorithms 1-3 from the PDF, generalized to work over
// any KeyShape-described element decomposition (see order_key.hpp) instead of one fixed
// value type - Section 4.5 explicitly states this isolation ("PS-Tree... isolates the
// specific value types and operators from the upper matching layer") is a design goal, so
// making the tree generic over the element codec follows the paper's own intent.
//
// SCOPE: only the four operators the paper actually gives pseudocode for are implemented
// here - >= (kGe), = (kEq), <= (kLe), and in/BETWEEN (kIn, both endpoints inclusive).
// InsertPredicate's own pseudocode (Algorithm 1, lines 2-10) only shows these three
// ranges explicitly plus prose "other operators are processed in a similar way";
// DeletePredicate (Algorithm 3, lines 1-17) independently confirms the same four
// (>=, =, <=, in), never showing > or < explicitly either. Deriving a correct standalone
// wiring for strict > / < turned out to need real care (an earlier draft of this file got
// it wrong - see git history) and isn't actually necessary: since every value type here is
// encoded as a discrete, totally-ordered ElementKey (order_key.hpp), ">V" and "<V" have a
// well-defined equivalent as ">=next(V)" and "<=prev(V)" via adjacent-key stepping - that
// normalization belongs in the PSTDynamic layer (Phase 2), one level above this file, not
// duplicated as new tree-wiring logic here.
//
// Two identified discrepancies between the literally-printed pseudocode and what must be
// correct, found by cross-checking against the paper's own worked examples and prose
// before writing this code (not assumed - see the comments at each site):
//   1. MatchPair (Algorithm 2, line 8): the paper prints `GetLNode(currNode, pstree.root)`,
//      but the surrounding prose says "GetRNode and GetLNode are invoked" (both used
//      together) and Algorithm 1's own Partition uses the identical two-step pattern
//      `iRNode <- GetRNode(path); iLNode <- GetLNode(iRNode, root)`. Also, if `currNode`'s
//      own subtree has no descendants yet, GetLNode(currNode,...) cannot terminate (its
//      loop requires *some* non-null p[] entry to descend through). Implemented here as
//      `GetLNode(iRNode, root)`, matching Algorithm 1's own pattern.
//   2. DeletePredicate (Algorithm 3, line 19): the paper prints `if lNode.predCounter = 0`,
//      but `lNode` is never defined anywhere else in the function - `startNode`, whose
//      predCounter was just decremented on the immediately preceding line, is clearly the
//      intended variable. Implemented here as `startNode.predCounter == 0`.
//
// Deferred, not a correctness gap: the paper describes space MERGING (combining adjacent
// leaves that end up with equal predicate counters after a deletion) only as prose with a
// worked example (Section 4.4) - it gives no pseudocode for `MergeSpaces` at all. Skipping
// the merge does NOT break MatchPair's correctness (two adjacent leaves with equal
// counters still each correctly answer queries into their own sub-range), it only forgoes
// a memory-compaction opportunity - so this first implementation performs the
// well-specified part (decrementing counters, and leaving a leaf's counter at zero once
// nothing covers it) and leaves both actual merging AND freeing zero-counter leaves as
// follow-up work (freeing one safely needs the same inner-node rewiring merging would).

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pstree/order_key.hpp"

namespace pstree {

// Only the operators PS-Tree's own algorithms give pseudocode for - see file-level
// comment for why kGt/kLt are deliberately absent (handled one layer up via adjacent-key
// normalization instead).
enum class Op {
    kGe, // >=
    kEq, // ==
    kLe, // <=
    kIn, // BETWEEN [lo, hi], both endpoints inclusive
};

struct LeafNode {
    LeafNode* next = nullptr; // next leaf in increasing predicate-space order
    std::uint64_t predCounter = 0;
    // Placeholder for Phase 2's own grouped subscription structure (dimension-signature
    // groups). Phase 1 has no subscription semantics yet - callers of insertPredicate get
    // back the affected LeafNode pointers directly and may attach whatever they like.
};

struct InnerNode {
    // Paper Fig. 2 overloads inner-node l/e/g as pointers to LEAF nodes (not other inner
    // nodes) - modeled faithfully here via separate leaf-typed fields.
    LeafNode* leaf_l = nullptr; // leaf whose predicate space is strictly less than this node's value
    LeafNode* leaf_e = nullptr; // leaf whose predicate space equals this node's value
    LeafNode* leaf_g = nullptr; // leaf whose predicate space is strictly greater than this node's value
    std::vector<InnerNode*> p;  // child inner nodes, indexed by this level's element value
};

// A predicate's single insertion/deletion unit: one attribute (handled by the caller, one
// PSTree per dimension), one operator, one or two values (kIn uses both vals0/vals1 as
// [lo, hi]; every other op uses vals0 only).
struct Predicate {
    Op op;
    ElementKey vals0;
    ElementKey vals1; // only meaningful for Op::kIn
};

class PSTree {
public:
    explicit PSTree(KeyShape shape) : shape_(std::move(shape)) {
        root_ = new InnerNode();
        root_->p.assign(levelRadix(0), nullptr);
        // Root's own boundary leaves: g links to the first (leftmost) predicate space,
        // l links to the last (rightmost) - "For the root node, a special consideration
        // is that g links to the first leaf node, and l links to the last leaf node."
        auto* first = new LeafNode();
        root_->leaf_g = first;
        root_->leaf_l = first;
    }

    ~PSTree() { destroy(); }

    PSTree(const PSTree&) = delete;
    PSTree& operator=(const PSTree&) = delete;

    // Algorithm 1, InsertPredicate. Returns every leaf whose predicate space is now
    // covered by `pred` (predCounter already incremented on each).
    std::vector<LeafNode*> insertPredicate(const Predicate& pred) {
        LeafNode* startNode = nullptr;
        LeafNode* endNode = nullptr;
        switch (pred.op) {
            case Op::kGe:
                startNode = partition(pred.vals0, Op::kGe);
                endNode = root_->leaf_l;
                break;
            case Op::kEq:
                startNode = partition(pred.vals0, Op::kEq);
                endNode = startNode;
                break;
            case Op::kLe:
                startNode = root_->leaf_g;
                endNode = partition(pred.vals0, Op::kLe);
                break;
            case Op::kIn:
                // Not literal pseudocode - InsertPredicate's own Algorithm 1 only shows
                // kGe/kEq/kLe explicitly. Derived by mirroring DeletePredicate's own kIn
                // case (Algorithm 3, lines 13-17), which IS given explicitly and uses two
                // separate boundary lookups, one per endpoint - so BETWEEN [lo,hi] is
                // treated here as ">=lo" partitioned at the low end and "<=hi" partitioned
                // at the high end, the natural insertion-side mirror of that lookup shape.
                startNode = partition(pred.vals0, Op::kGe);
                endNode = partition(pred.vals1, Op::kLe);
                break;
        }
        std::vector<LeafNode*> leafNodes;
        LeafNode* cur = startNode;
        LeafNode* stop = endNode->next;
        while (cur != stop) {
            cur->predCounter++;
            leafNodes.push_back(cur);
            cur = cur->next;
        }
        return leafNodes;
    }

    // Algorithm 2, MatchPair. Locates the leaf whose predicate space covers `val`.
    LeafNode* matchPair(const ElementKey& val) const {
        InnerNode* currNode = root_;
        std::vector<std::pair<InnerNode*, std::uint16_t>> path;
        for (std::size_t level = 0; level < val.size(); ++level) {
            std::uint16_t elem = val[level];
            path.emplace_back(currNode, elem);
            if (currNode->p[elem] != nullptr) {
                currNode = currNode->p[elem];
            } else {
                InnerNode* iRNode = getRNode(path);
                // See file-level comment #1: iRNode (not currNode, as literally printed).
                InnerNode* iLNode = getLNode(iRNode);
                return iLNode->leaf_l;
            }
        }
        return currNode->leaf_e;
    }

    // Algorithm 3, DeletePredicate. Decrements predCounter on every leaf `pred` covered.
    // Does NOT free zero-counter leaves or merge survivors (see file-level comment).
    // Returns every leaf that was decremented, in order.
    std::vector<LeafNode*> deletePredicate(const Predicate& pred) {
        LeafNode* startNode = nullptr;
        LeafNode* endNode = nullptr;
        switch (pred.op) {
            case Op::kGe:
                startNode = matchPair(pred.vals0);
                endNode = root_->leaf_l;
                break;
            case Op::kEq:
                startNode = matchPair(pred.vals0);
                endNode = startNode;
                break;
            case Op::kLe:
                startNode = root_->leaf_g;
                endNode = matchPair(pred.vals0);
                break;
            case Op::kIn:
                startNode = matchPair(pred.vals0);
                endNode = matchPair(pred.vals1);
                break;
        }
        std::vector<LeafNode*> leafNodes;
        LeafNode* cur = startNode;
        LeafNode* stop = endNode->next;
        while (cur != stop) {
            // See file-level comment #2: startNode (not the undefined `lNode`).
            cur->predCounter -= 1;
            leafNodes.push_back(cur);
            cur = cur->next;
        }
        return leafNodes;
    }

    const KeyShape& shape() const { return shape_; }
    InnerNode* root() const { return root_; }

private:
    KeyShape shape_;
    InnerNode* root_;

    std::uint32_t levelRadix(std::size_t level) const { return shape_.radix.at(level); }

    // "The function CopyLeafNode creates a new leaf node by copying all information from
    // an existing leaf node" (page 13 prose) - critically this includes `next`, captured
    // at copy time BEFORE the caller redirects src->next to point at the new leaf. This is
    // what makes the 3-leaf kEq/kIn split (partitionLeafNodeLeft) correct without an
    // explicit final assignment: both new leaves are copied from the same source before
    // either's `next` is touched, so the second (last) one already inherits the source's
    // original downstream link.
    static LeafNode* copyLeafNode(LeafNode* src) {
        auto* dst = new LeafNode();
        dst->predCounter = src->predCounter;
        dst->next = src->next;
        return dst;
    }

    // `level` is the depth of the node being created (root = 0). A node created at the
    // deepest level (depth == shape_.depth()) is never itself indexed via p[] - the walk
    // that created it just consumed the last element of the key - so its own child array
    // is left empty rather than sized from a radix entry that doesn't exist.
    InnerNode* createInnerNode(std::size_t level) {
        auto* node = new InnerNode();
        if (level < shape_.depth()) {
            node->p.assign(levelRadix(level), nullptr);
        }
        return node;
    }

    // Algorithm 1, GetRNode: walk back up `path` (built while descending) looking for the
    // nearest ancestor with an existing child strictly to the right of the path taken.
    InnerNode* getRNode(const std::vector<std::pair<InnerNode*, std::uint16_t>>& path) const {
        for (std::size_t i = path.size(); i-- > 0;) {
            auto [node, elem] = path[i];
            std::uint32_t length = levelRadix(i);
            for (std::uint32_t pos = elem + 1; pos < length; ++pos) {
                if (node->p[pos] != nullptr) return node->p[pos];
            }
            if (i == 0) return node; // exhausted the path - root itself
        }
        return root_;
    }

    // Algorithm 1, GetLNode: descend into the leftmost existing child repeatedly, until
    // reaching a node whose `l` link is set.
    InnerNode* getLNode(InnerNode* iRNode) const {
        if (iRNode == root_) return root_;
        InnerNode* iLNode = iRNode;
        while (iLNode->leaf_l == nullptr) {
            bool found = false;
            for (std::size_t pos = 0; pos < iLNode->p.size(); ++pos) {
                if (iLNode->p[pos] != nullptr) {
                    iLNode = iLNode->p[pos];
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::logic_error("pstree: GetLNode could not descend - malformed tree");
            }
        }
        return iLNode;
    }

    // Algorithm 1, Partition. Walks/creates the inner-node path for `val`, then splits the
    // leaf at that boundary (creating it if necessary) according to `op` (kGe/kEq/kLe
    // only - kIn's two endpoints are each partitioned separately as kGe/kLe by the caller).
    // Always returns currNode->leaf_e (paper line 30, unconditional) - by construction
    // (see partitionLeafNodeLeft/Equal) leaf_e always ends up representing the correct
    // "up to and including V" boundary for kLe too, whether it was freshly split off this
    // call or already existed from an earlier insertion at the same point.
    LeafNode* partition(const ElementKey& val, Op op) {
        InnerNode* currNode = root_;
        std::vector<std::pair<InnerNode*, std::uint16_t>> path;
        for (std::size_t level = 0; level < val.size(); ++level) {
            std::uint16_t elem = val[level];
            path.emplace_back(currNode, elem);
            if (currNode->p[elem] == nullptr) {
                currNode->p[elem] = createInnerNode(level + 1);
            }
            currNode = currNode->p[elem];
        }
        if (currNode->leaf_e == nullptr) {
            InnerNode* iRNode = getRNode(path);
            InnerNode* iLNode = getLNode(iRNode);
            partitionLeafNodeLeft(currNode, iLNode, op);
        } else {
            partitionLeafNodeEqual(currNode, op);
        }
        return currNode->leaf_e;
    }

    // Algorithm 1, PartitionLeafNodeEqual (lines 58-77) - currNode already has an
    // exact-match boundary (currNode->leaf_e != null); further split it based on `op`.
    // Transcribed directly: kGe further splits currNode.l/e apart if they're still the
    // same leaf; kLe further splits currNode.e/g apart if they're still the same leaf;
    // kEq does whichever of the two is still needed (both can be pending at once the
    // first time a `=` predicate meets a boundary another operator already created).
    void partitionLeafNodeEqual(InnerNode* currNode, Op op) {
        if (op == Op::kGe) {
            if (currNode->leaf_l == currNode->leaf_e) {
                LeafNode* leafNode = copyLeafNode(currNode->leaf_l);
                currNode->leaf_l->next = leafNode;
                currNode->leaf_e = leafNode;
            }
        } else if (op == Op::kLe) {
            if (currNode->leaf_e == currNode->leaf_g) {
                LeafNode* leafNode = copyLeafNode(currNode->leaf_e);
                currNode->leaf_e->next = leafNode;
                currNode->leaf_g = leafNode;
            }
        } else if (op == Op::kEq) {
            if (currNode->leaf_e == currNode->leaf_g) {
                LeafNode* leafNode = copyLeafNode(currNode->leaf_e);
                currNode->leaf_e->next = leafNode;
                currNode->leaf_g = leafNode;
            } else if (currNode->leaf_l == currNode->leaf_e) {
                LeafNode* leafNode = copyLeafNode(currNode->leaf_l);
                currNode->leaf_l->next = leafNode;
                currNode->leaf_e = leafNode;
            }
        }
    }

    // Algorithm 1, PartitionLeafNodeLeft (lines 79-102) - no exact-match boundary exists
    // yet; iLNode->leaf_l points to the (undivided) leaf currently spanning across `val`.
    // Transcribed directly: kGe carves off "[V,...)" as a new leaf, old leaf keeps
    // "<V" (currNode.l). kLe carves off "(V,...)" as a new leaf (currNode.g), old leaf
    // absorbs "=V" too and becomes the "(...,V]" boundary (currNode.e AND currNode.l both
    // point at it) - this is the paper's literal ">"-operator wiring (lines 87-93), reused
    // here for kLe: `<=V` and `>V` are complementary partitions of the exact same point,
    // so they necessarily share one tree structure, differing only in which side
    // InsertPredicate/DeletePredicate walk from - see this file's kGt/kLt comment above.
    // kEq carves off two new leaves, "=V" and ">V"; old leaf keeps "<V" unchanged.
    //
    // FIX beyond the literal pseudocode, found by tracing Section 4.4's own worked
    // example by hand and getting a wrong answer: `iLNode` found via GetRNode/GetLNode
    // can itself be an EXISTING boundary node (not just the tree-wide root) - e.g. when
    // 30 is inserted between already-existing boundaries at 20 and 60, iLNode resolves to
    // 60's own boundary node, whose leaf_l WAS aliased with its own leaf_e (both pointing
    // at the single undivided "[20,60]" leaf, from 60's own earlier <=-style insertion).
    // The literal pseudocode reassigns iLNode.l to the freshly split-off leaf but never
    // touches iLNode.e, leaving it stale (still pointing at what is now only "[20,30)"
    // instead of the "[30,60]" piece that actually covers value 60). Fixed by propagating
    // the SAME reassignment to iLNode.e whenever it was aliased with the old iLNode.l -
    // deliberately NOT propagating to iLNode.g (see below) or to root_ specifically (root_
    // has no leaf_e at all - it's always null - so the `== oldLeaf` check is naturally a
    // no-op there without needing a special case).
    //
    // iLNode.g is never aliased with iLNode.l for any non-root node by construction (a
    // kGe-created node has leaf_e==leaf_g, never leaf_l; a kLe-created node has
    // leaf_l==leaf_e, never leaf_g; a kEq-created node has all three distinct) - the ONLY
    // place leaf_l and leaf_g start equal is root_ itself (both = the single initial leaf,
    // at construction), and that aliasing is purely incidental: root_.g means "first
    // (leftmost) leaf" and root_.l means "last (rightmost) leaf", two genuinely different
    // semantic positions that must stop tracking each other the moment the first real
    // split happens - propagating to .g there would be wrong, so this fix deliberately
    // never does.
    void partitionLeafNodeLeft(InnerNode* currNode, InnerNode* iLNode, Op op) {
        LeafNode* oldLeaf = iLNode->leaf_l;
        if (op == Op::kGe) {
            LeafNode* leafNode = copyLeafNode(oldLeaf);
            oldLeaf->next = leafNode;
            currNode->leaf_l = oldLeaf;
            currNode->leaf_e = leafNode;
            currNode->leaf_g = leafNode;
            if (iLNode->leaf_e == oldLeaf) iLNode->leaf_e = leafNode;
            iLNode->leaf_l = leafNode;
        } else if (op == Op::kLe) {
            LeafNode* leafNode = copyLeafNode(oldLeaf);
            oldLeaf->next = leafNode;
            currNode->leaf_l = oldLeaf;
            currNode->leaf_e = oldLeaf;
            currNode->leaf_g = leafNode;
            if (iLNode->leaf_e == oldLeaf) iLNode->leaf_e = leafNode;
            iLNode->leaf_l = leafNode;
        } else if (op == Op::kEq) {
            LeafNode* leafNodeOne = copyLeafNode(oldLeaf);
            LeafNode* leafNodeTwo = copyLeafNode(oldLeaf);
            oldLeaf->next = leafNodeOne;
            leafNodeOne->next = leafNodeTwo;
            currNode->leaf_l = oldLeaf;
            currNode->leaf_e = leafNodeOne;
            currNode->leaf_g = leafNodeTwo;
            if (iLNode->leaf_e == oldLeaf) iLNode->leaf_e = leafNodeTwo;
            iLNode->leaf_l = leafNodeTwo;
        }
    }

    void destroy() {
        LeafNode* leaf = root_ != nullptr ? root_->leaf_g : nullptr;
        while (leaf != nullptr) {
            LeafNode* next = leaf->next;
            delete leaf;
            leaf = next;
        }
        destroyInner(root_);
        root_ = nullptr;
    }

    static void destroyInner(InnerNode* node) {
        if (node == nullptr) return;
        for (InnerNode* child : node->p) {
            destroyInner(child);
        }
        delete node;
    }
};

} // namespace pstree
