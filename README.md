# pstree

A from-scratch C++20 implementation of **PS-Tree** and **PSTDynamic**, the boolean-expression
matching design from ["Efficient Parallel Boolean Expression Matching"](https://doi.org/10.1145/3736756)
(Ji, Yao, Wang, Wei, Jacobsen — ACM Trans. Database Syst. 50, 4, Article 16, 2025).

This is the same research lineage as [a-tree](https://github.com/mrayva/a-tree) and
[be-tree](https://github.com/mrayva/be-tree) - the paper's own benchmarks report PSTDynamic
outperforming both on matching time, index construction time, and memory. No public source exists
for PS-Tree/PSTDynamic/PSTParallel (or their 2018 precursors PSTBloom/PSTHash) anywhere - this is
a full reimplementation from the paper's own pseudocode (Algorithms 1-7), not an adaptation of an
existing codebase.

## Status

**Phase 1 complete**: the PS-Tree index itself (Algorithms 1-3 - `InsertPredicate`, `MatchPair`,
`DeletePredicate`), including strict `>`/`<` support, domain-boundary edge cases, and a randomized
property-based stress test (checked against a brute-force oracle across dozens of seeds and
thousands of operations each) validating the whole insert/delete/match pipeline, not just the
worked examples. Clean under ASan+UBSan with leak detection. PSTDynamic (Algorithms 4-6, the actual
matching engine built on top) and PSTParallel (Algorithm 7, the multicore extension) are not yet
started.

## Design notes and deliberate deviations from the paper

This implementation transcribes the paper's pseudocode directly wherever it's given, but the
pseudocode has real gaps and at least one apparent transcription error, found by cross-checking
against the paper's own worked examples rather than assumed. See the top-of-file comments in
[`include/pstree/ps_tree.hpp`](include/pstree/ps_tree.hpp) and
[`include/pstree/order_key.hpp`](include/pstree/order_key.hpp) for the full detail on each; briefly:

- **Value-type element encoding** (Section 4.5) is generalized rather than copying the paper's own
  irregular per-type bit splits: every fixed-width numeric type (`int64_t`, `double`) is encoded
  via the standard order-preserving "sortable bit pattern" bijection (the same trick radix sort
  uses for signed integers and IEEE-754 floats), then uniformly chunked into 4-bit elements - one
  shared mechanism instead of a bespoke split per type.
- **The four operators the paper's own Algorithms 1 and 3 give pseudocode for** (`>=`, `=`, `<=`,
  `in`/BETWEEN) are implemented directly; strict `>`/`<` are supported too, via adjacent-key
  normalization (`>V` becomes `>=next(V)`) rather than separately-derived tree wiring, since every
  value type here has a well-defined discrete successor (`order_key.hpp`'s
  `nextElementKey`/`prevElementKey` - ordinary mixed-radix increment/decrement, generic across every
  codec).
- **`MatchPair` (Algorithm 2, line 8)**: the paper prints `GetLNode(currNode, ...)`; the surrounding
  prose and Algorithm 1's own identical two-step pattern both point to `GetLNode(iRNode, ...)`
  being correct instead - implemented that way.
- **`DeletePredicate` (Algorithm 3, line 19)**: the paper prints `if lNode.predCounter = 0`, an
  undefined variable - clearly meant to be `startNode`, the variable decremented on the previous
  line.
- **A real, deeper gap in both `PartitionLeafNodeLeft` and `PartitionLeafNodeEqual`**: neither
  function's literal pseudocode updates any node other than the one it's directly operating on when
  splitting a leaf - but `GetLNode`'s own search (invoked by an insertion at some OTHER, unrelated
  value) can permanently wire an already-existing node's `.l` link directly to a leaf that a LATER,
  completely unrelated insertion then splits, leaving that reference stale. Found in two stages:
  first by hand-tracing Section 4.4's own worked example (a `.e` self-alias on the SAME node going
  stale), then - after that fix still left a subtler case unhandled - by a randomized property test
  that found a second instance involving a WHOLLY UNRELATED node's `.l`, wired up by its own earlier
  insertion. Fixed generally, not case-by-case: every `LeafNode` tracks which `InnerNode`s currently
  point to it via `.l` (`l_refs`), so every split can correctly redirect every such reference to the
  new "rightward" piece - see `ps_tree.hpp`'s `setLeafL`/`redirectLeafLRefs` and file-level comment
  #3 for the full reasoning, including why every referrer is guaranteed to need the redirect (not
  just some of them).
- **Space merging is deferred**: the paper describes `MergeSpaces` (combining adjacent leaves that
  end up with equal predicate counters after a deletion) only as prose with a worked example -
  no pseudocode is given anywhere. `DeletePredicate` here does the well-specified part (decrementing
  counters) and leaves both merging and freeing zero-counter leaves as follow-up work - skipping
  them doesn't break `MatchPair`'s correctness, only forgoes a memory-compaction opportunity.

## Building

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Sanitizer build (ASan + UBSan, mirroring be-tree's own CI setup):

```bash
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer" \
  "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined"
cmake --build build-asan
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build-asan --output-on-failure
```

## TODO

- [x] PS-Tree core (`InsertPredicate`/`MatchPair`/`DeletePredicate`), strict `>`/`<`, domain-edge
      cases, randomized stress testing.
- [ ] PSTDynamic (Algorithms 4-6): access-predicate selection, dimension-signature (Bloom filter)
      grouping, `InsertSubscription`/`MatchEvent`/`DeleteSubscription`, own predicate evaluator.
- [ ] `list_valued` attribute support (nats_sidecar's `string_list`/`integer_list` - no direct
      analog in the paper's single-attribute-value-pair model).
- [ ] `nats_sidecar` integration as a third pluggable matching engine.
- [ ] PSTParallel (Algorithm 7) - a much bigger, separate architectural lift, deliberately not
      scoped yet.
- [ ] Space merging / zero-counter leaf reclamation (deferred, see design notes above) - not a
      correctness requirement, but real memory growth under long-running insert/delete churn.
