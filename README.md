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

**Phase 3 complete**: integrated into `mrayva/nats_sidecar` as a third pluggable matching engine
(`engine: pstree`) - see that repo's own README.md for the operator-support/limitations summary.
PSTParallel (Phase 4) remains deliberately not started.

**Independent benchmark against a-tree/be-tree (`nats_sidecar`'s `benchmarks/matching_engine_bench.cpp`,
see that repo's README for the full writeup and numbers) found a real, architectural scaling
limitation, not a bug**: `PSTDynamic::insertSubscription()` must attach a subscription to *every
leaf* its access predicate's range covers (required for `MatchEvent`'s O(1) per-event lookup
contract to hold), which is cheap for an equality/narrow-range access predicate but degrades toward
O(leaf-count) per insertion - and O(group-size) per matching event - when a subscription's *only*
available predicate is a wide, unbounded comparison (`price > X`) and many such subscriptions with
independent thresholds share a dimension. Measured: at K=10,000 synthetic subscriptions (30% of
which are exactly this shape), pstree's insert rate falls to ~1.4% of a-tree's and its search rate
to ~15% of a-tree's - a real, growing-with-K disadvantage, not a fixed constant-factor gap. This
doesn't contradict the paper's own results (likely measured on workloads with narrower/bounded
access predicates) but does mean PSTDynamic as implemented here isn't a good fit for a workload
dominated by independent wide-range "threshold alert" subscriptions.

**Phase 1 complete**: the PS-Tree index itself (Algorithms 1-3 - `InsertPredicate`, `MatchPair`,
`DeletePredicate`), including strict `>`/`<` support, domain-boundary edge cases, and a randomized
property-based stress test (checked against a brute-force oracle across dozens of seeds and
thousands of operations each) validating the whole insert/delete/match pipeline, not just the
worked examples.

**Phase 2 complete**: PSTDynamic (Algorithms 4-6 - `InsertSubscription`, `MatchEvent`,
`DeleteSubscription`), including the access-predicate selectivity heuristic, dimension-signature
(Bloom filter) grouping with real hysteresis on grow/shrink, the full Section 2.1 predicate
language (`<,<=,=,!=,>,>=`, BETWEEN, element-of/not-element-of) with its own evaluator, and a
randomized property-based stress test. Verified against the paper's own Fig. 3 worked example
(all 6 subscriptions - `SelectAccPred` reproduces the paper's own stated access-predicate choice
for every one of them, not just a plausible-looking one) and Section 5.3's event-matching
walkthrough.

Both phases are clean under ASan+UBSan with leak detection, in CI and via ad-hoc extended
multi-seed runs (60+ seeds, up to 3000 operations each) before being considered solid enough to
build on. PSTParallel (Algorithm 7, the multicore extension) is deliberately not started - see the
plan/TODO below.

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

### PSTDynamic (Phase 2) - see `pst_dynamic.hpp`'s own file-level comment for full detail

- **Dimension signatures cover every dimension a subscription has a predicate on, including the
  access predicate's own dimension** - confirmed directly from Fig. 3's own worked example (S1/S2
  have predicates on `{attr1,attr2}`, signature `"110"`; not just the "other" dimension). The
  paper gives no pseudocode for the Bloom filter itself; this implementation uses standard
  Kirsch-Mitzenmacher double hashing and is verified against the worked example at the semantic
  level (which subscriptions land in the same vs. different groups), not by trying to reproduce
  its specific, hash-function-dependent bit strings.
- **Grow/shrink uses two deliberately-separated threshold functions, not the paper's one
  `thresholds[DimSigLen]` array reused for both directions** - reusing one array for both the grow
  check (`Algorithm 4`) and the shrink check (`Algorithm 6`, evaluated at whatever length growth
  just changed it to) risks immediate thrashing unless the thresholds happen to already have
  hysteresis built in, which the paper never specifies. This implementation grows at
  `capacity(len)` and only shrinks once **below** `capacity(len)/4`, giving real hysteresis by
  construction.
- **A real architectural gap found by the randomized stress test, not a hand-traced example this
  time**: PSTDynamic needs to attach its own per-leaf metadata (which subscriptions are grouped
  there) to PS-Tree leaves - but a leaf's identity isn't stable, the same way `predCounter` already
  isn't: a *later, wholly unrelated* subscription's insertion can split a leaf that an *earlier*
  subscription is already grouped under, and nothing propagated that grouping to the new piece.
  Fixed by extending `ps_tree.hpp` itself with a generic, opaque `LeafNode::userData` slot plus
  clone/destroy hooks supplied to `PSTree`'s constructor - `copyLeafNode()` clones `userData`
  exactly when it clones `predCounter`, so whatever PSTDynamic attaches automatically follows
  every split the same way the counter does. This keeps PS-Tree itself fully generic (it never
  interprets what's attached) while making the upper layer's metadata correctly survive a leaf
  split it has no way to observe directly.
- **`kElemOf`'s literal value list is deduplicated by encoded key before building tree
  predicates** - a repeated literal (e.g. `['ee','ee']`) would otherwise insert `kEq(V)` twice,
  returning the same leaf twice in the affected-leaf union and double-counting the subscription in
  its own group. Also caught by the randomized stress test (its random generator can produce
  duplicate literals), not anticipated in advance.
- **`kNe`/`kNotElemOf`, if selected as the (least-bad available) access predicate**, have no
  representable contiguous PS-Tree range at all - handled by inserting `>=` the dimension's
  minimum representable value, i.e. "matches every leaf": correct (never a false negative) but
  forgoes pruning, exactly the outcome Section 2.3's own selectivity ranking anticipates for these
  operators.
- **`list_valued` attributes are not supported** (nats_sidecar's `string_list`/`integer_list`) -
  the paper's model has no analog (an event attribute is always a single value, never a list) -
  real follow-up work for the `nats_sidecar` integration phase, not assumed solved.
- **`kIsNull`/`kIsNotNull` are not part of the paper's model at all** - added for a real caller's
  need (nats_sidecar's own "attribute is/is not present" predicates), evaluated on ABSENCE rather
  than a value comparison. `kIsNull` can never be an access predicate (rejected clearly at insert
  time if it's the only/best option) - there is structurally no way to index "this dimension was
  absent", since `MatchEvent` only ever consults a dimension's tree for events that DO have it.
  `kIsNotNull` CAN be indexed (falls back to "matches every leaf", which is exactly correct here,
  not just safe - see the code comment). A real, subtle bug found by extending the tests for this:
  a subscription's dimension signature must EXCLUDE any dimension used only via `kIsNull` - the
  signature's whole pruning contract assumes "this subscription needs dimension D" means "D must
  be present", which is backwards for `kIsNull` and would incorrectly prune out the exact events
  it's meant to match.

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
- [x] PSTDynamic (Algorithms 4-6): access-predicate selection, dimension-signature (Bloom filter)
      grouping, `InsertSubscription`/`MatchEvent`/`DeleteSubscription`, own predicate evaluator.
- [x] `nats_sidecar` integration as a third pluggable matching engine (`engine: pstree`) - see
      `mrayva/nats_sidecar`'s own README.md/`src/pstree_dialect.hpp` for the DNF-conversion layer
      that bridges PSTDynamic's pure-conjunction model to nats_sidecar's full AND/OR/NOT grammar.
      `list_valued` attribute support (below) stayed unaddressed by design - the integration
      instead rejects any expression referencing one at subscribe time, a real limitation not an
      omission (see nats_sidecar's own README for the full list, including `is null` alone as an
      unindexable access predicate).
- [ ] `list_valued` attribute support (nats_sidecar's `string_list`/`integer_list` - no direct
      analog in the paper's single-attribute-value-pair model) - not pursued; nats_sidecar's
      integration rejects these at subscribe time instead (see above).
- [ ] PSTParallel (Algorithm 7) - a much bigger, separate architectural lift, deliberately not
      scoped yet.
- [ ] Space merging / zero-counter leaf reclamation (deferred, see design notes above) - not a
      correctness requirement, but real memory growth under long-running insert/delete churn.
