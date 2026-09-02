// PSTDynamic correctness tests. The main one is the paper's own Fig. 3 example (6
// subscriptions over 3 integer dimensions) plus Section 5.3's event-matching walkthrough -
// transcribed directly, including verifying SelectAccPred's own choice matches what the
// paper states for every one of the six subscriptions (not assumed - the paper's stated
// final answer, "only S4 matches", is independently re-derived by hand in the plan/commit
// history before being pinned here as a test). Grouping/dimension-signature internals are
// deliberately NOT white-box tested (PSTDynamic exposes no introspection for them, on
// purpose - they're an internal optimization) - every test here checks observable behavior
// (MatchEvent's actual output) instead, which is what the algorithm's public contract
// actually promises.

#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "pstree/pst_dynamic.hpp"

namespace {

int g_failures = 0;

void require(bool cond, const std::string& message) {
    if (!cond) {
        std::cerr << "FAIL: " << message << "\n";
        g_failures++;
    }
}

bool contains(const std::vector<std::uint64_t>& ids, std::uint64_t id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

pstree::SubPredicate P(std::string attr, pstree::CmpOp op, std::vector<pstree::Value> vals) {
    return pstree::SubPredicate{std::move(attr), op, std::move(vals)};
}

std::vector<pstree::AttrSchema> fig3Schema() {
    return {
        {"attr1", pstree::ValueType::kInteger, 0},
        {"attr2", pstree::ValueType::kInteger, 0},
        {"attr3", pstree::ValueType::kInteger, 0},
    };
}

// Fig. 3's own six subscriptions, transcribed directly (page 17, re-read carefully after
// an initial misread of S5/S6 - see git history / session notes).
std::vector<pstree::Subscription> fig3Subscriptions() {
    using pstree::CmpOp;
    std::vector<pstree::Subscription> subs(6);
    subs[0] = {1, {P("attr1", CmpOp::kLt, {std::int64_t{0}}), P("attr2", CmpOp::kBetween, {std::int64_t{1}, std::int64_t{5}})}};
    subs[1] = {2, {P("attr1", CmpOp::kGt, {std::int64_t{8}}), P("attr2", CmpOp::kBetween, {std::int64_t{1}, std::int64_t{5}})}};
    subs[2] = {3, {P("attr2", CmpOp::kBetween, {std::int64_t{1}, std::int64_t{5}}), P("attr3", CmpOp::kLt, {std::int64_t{0}})}};
    subs[3] = {4, {P("attr2", CmpOp::kBetween, {std::int64_t{1}, std::int64_t{5}}), P("attr3", CmpOp::kGt, {std::int64_t{8}})}};
    subs[4] = {5, {P("attr1", CmpOp::kEq, {std::int64_t{2}}), P("attr2", CmpOp::kLt, {std::int64_t{1}}), P("attr3", CmpOp::kGt, {std::int64_t{8}})}};
    subs[5] = {6, {P("attr1", CmpOp::kLt, {std::int64_t{0}}), P("attr2", CmpOp::kGt, {std::int64_t{5}}), P("attr3", CmpOp::kEq, {std::int64_t{6}})}};
    return subs;
}

// "for S1, S2, S3 and S4, the predicate {attr2,in,[1,5]} is selected as the access
// predicate, while for S5 and S6, {attr1,=,2} and {attr3,=,6} are selected as the access
// predicate" (page 17 prose) - verified against selectAccPredIndex() directly, proving the
// static selectivity heuristic reproduces the paper's own stated choices exactly, not just
// SOME plausible-looking choice. selectAccPredIndex is now a PSTDynamic member (it needs
// per-dimension observed-cardinality state - see pst_dynamic.hpp's own comment) rather than a
// free function, but a fresh instance's own cardinality tracker is empty either way: none of
// Fig. 3's subscriptions have two same-tier predicates (kEq/kBetween/kLt/kGt are all
// different tiers here), so opRank alone decides for every one of them regardless - this test
// is unaffected by the cardinality-aware tie-break change.
void test_select_acc_pred_matches_paper() {
    pstree::PSTDynamic pstd(fig3Schema());
    auto subs = fig3Subscriptions();
    require(subs[0].predicates[pstd.selectAccPredIndex(subs[0])].attr == "attr2", "S1 access predicate should be on attr2");
    require(subs[1].predicates[pstd.selectAccPredIndex(subs[1])].attr == "attr2", "S2 access predicate should be on attr2");
    require(subs[2].predicates[pstd.selectAccPredIndex(subs[2])].attr == "attr2", "S3 access predicate should be on attr2");
    require(subs[3].predicates[pstd.selectAccPredIndex(subs[3])].attr == "attr2", "S4 access predicate should be on attr2");
    require(subs[4].predicates[pstd.selectAccPredIndex(subs[4])].attr == "attr1", "S5 access predicate should be on attr1 (the '=' predicate)");
    require(subs[5].predicates[pstd.selectAccPredIndex(subs[5])].attr == "attr3", "S6 access predicate should be on attr3 (the '=' predicate)");
}

// Regression test for a real bug found via `perf`-profiling a downstream benchmark
// (nats_sidecar's K=4000-32000 exchange/symbol set-membership benchmark - see
// pst_dynamic.hpp's own file-level comment for the full story): the paper's raw
// "narrower value list wins" tie-break conflates "few literals" with "selective," which is
// only true when every candidate dimension's REAL domain is similarly sized. A subscription
// referencing a low-cardinality dimension with few literals and a high-cardinality dimension
// with more literals always picked the low-cardinality one under the old rule - which indexes
// almost nothing once subscription counts scale up, since nearly every subscription becomes a
// per-event candidate regardless of which value the event carries.
//
// "narrow" and "wide" mirror the real bug's shape (19-value exchange vs. ~12,000-value
// symbol) at a scale a unit test can afford: 20 warm-up subscriptions teach the tracker that
// "wide" has a much larger real domain (40 distinct values) than "narrow" (2 distinct values,
// same 2 values every time) - then a subscription with a 2-literal "narrow" predicate and a
// 4-literal "wide" predicate should still prefer "wide", because 4/40 is a far smaller
// fraction of its own domain than 2/2 is of narrow's, even though 4 > 2 in raw literal count.
// Checked directly against selectAccPredIndex() (side-effect-free, doesn't require actually
// inserting `sub`) rather than reaching into any other internal state.
void test_select_acc_pred_prefers_higher_cardinality_dimension() {
    using pstree::CmpOp;
    std::vector<pstree::AttrSchema> schema = {
        {"narrow", pstree::ValueType::kString, 8},
        {"wide", pstree::ValueType::kString, 8},
    };
    pstree::PSTDynamic pstd(schema);

    for (int i = 0; i < 20; ++i) {
        pstree::Subscription warmup{
            100 + static_cast<std::uint64_t>(i),
            {P("narrow", CmpOp::kElemOf, {std::string("a"), std::string("b")}),
             P("wide", CmpOp::kElemOf,
               {std::string("w" + std::to_string(i * 2)), std::string("w" + std::to_string(i * 2 + 1))})}};
        pstd.insertSubscription(warmup);
    }

    pstree::Subscription sub{1, {
        P("narrow", CmpOp::kElemOf, {std::string("a"), std::string("b")}),
        P("wide", CmpOp::kElemOf,
          {std::string("w0"), std::string("w1"), std::string("w2"), std::string("w3")}),
    }};
    std::size_t idx = pstd.selectAccPredIndex(sub);
    require(sub.predicates[idx].attr == "wide",
            "wide (40 observed distinct values, 4 literals here) should be preferred over "
            "narrow (2 observed distinct values, 2 literals) despite having MORE literals - "
            "the old raw-width rule would have picked narrow and gotten this backwards");
}

// Insert/delete symmetry under a changing cardinality signal: DeleteSubscription must remove
// exactly the leaf entries InsertSubscription added, even though `observedValues` (and
// therefore what selectAccPredIndex() would return if called again) has changed in between by
// the time delete happens - see pst_dynamic.hpp's own comment on why the chosen index is
// stored, not recomputed. Deliberately checks by RE-INSERTING the same subscription after
// deleting it and confirming it still behaves identically (wrong bookkeeping from a
// mismatched delete would either throw - see deleteSubscription's own logic_error checks - or
// leave stale leaf entries that would make the id "double count" - both are directly
// observable via matchEvent's own output, not internal state).
void test_delete_uses_stored_access_predicate_not_a_recomputed_one() {
    using pstree::CmpOp;
    std::vector<pstree::AttrSchema> schema = {
        {"narrow", pstree::ValueType::kString, 8},
        {"wide", pstree::ValueType::kString, 8},
    };
    pstree::PSTDynamic pstd(schema);

    // sub 1 is inserted FIRST, while both dimensions are still at cardinality 0 - the cold-start
    // fallback (ratio == raw width) applies, so with EQUAL literal counts (1 each) it's a tie,
    // falling through to first-in-subscription-order: "narrow" wins as sub 1's access predicate.
    pstree::Subscription sub1{1, {P("narrow", CmpOp::kElemOf, {std::string("a")}),
                                   P("wide", CmpOp::kElemOf, {std::string("w0")})}};
    pstd.insertSubscription(sub1);

    // Now grow "wide"'s observed cardinality far past "narrow"'s, entirely through OTHER
    // subscriptions - by the time sub 1 is deleted, selectAccPredIndex(sub1) would no longer
    // agree with what was chosen at its own insert time (it would now prefer "wide", since
    // narrow is still stuck at cardinality 1 as the only value ever used for it).
    for (int i = 0; i < 20; ++i) {
        pstree::Subscription warmup{200 + static_cast<std::uint64_t>(i),
                                     {P("wide", CmpOp::kElemOf, {std::string("w" + std::to_string(i + 1))})}};
        pstd.insertSubscription(warmup);
    }
    std::size_t recomputedIdx = pstd.selectAccPredIndex(sub1);
    require(sub1.predicates[recomputedIdx].attr == "wide",
            "sanity check on the test itself: by now selectAccPredIndex WOULD pick a different "
            "predicate than sub1's own original insert-time choice - if it didn't, this test "
            "wouldn't actually be exercising the insert/delete symmetry concern at all");

    // The real check: delete sub1, then confirm an event matching ONLY via "narrow" (the
    // predicate actually chosen at insert time) no longer matches it, cleanly, with no
    // exception and no leftover/duplicate entries. If DeleteSubscription had recomputed and
    // used "wide" instead, it would most likely throw (see deleteSubscription's own
    // "subscription's group missing on delete" check, since sub1 was never actually inserted
    // into wide's own tree) - caught here so a reintroduced bug fails this ONE check cleanly
    // instead of aborting the whole test binary.
    try {
        pstd.deleteSubscription(1);
    } catch (const std::exception& e) {
        require(false, std::string("deleteSubscription(1) threw - likely recomputed a different "
                                    "access predicate than insert time chose: ") + e.what());
        return;
    }
    pstree::Event ev = {{"narrow", pstree::Value(std::string("a"))}, {"wide", pstree::Value(std::string("w0"))}};
    auto matches = pstd.matchEvent(ev);
    require(std::find(matches.begin(), matches.end(), 1u) == matches.end(),
            "deleted subscription must not still match");

    // Re-insert the identical subscription and confirm it matches again cleanly - proves
    // delete's bookkeeping didn't leave the tree in a subtly corrupted state.
    pstd.insertSubscription(sub1);
    matches = pstd.matchEvent(ev);
    require(std::find(matches.begin(), matches.end(), 1u) != matches.end(),
            "re-inserted subscription must match again");
}

// Section 5.3: event {attr2:2, attr3:10} should match ONLY S4 - re-derived independently by
// hand (S1/S2/S5/S6 all require attr1, absent from this event, so none of them can match
// regardless of their other predicates; of the remaining S3/S4, only S4's attr3>8 holds for
// attr3=10). This is a stronger check than trusting the paper's own prose about which LEAF
// a value resolves to (that specific claim in the text looks like a typo - see README) -
// the FINAL answer ("S4 only") is what's actually pinned here.
void test_section_5_3_event_matching() {
    pstree::PSTDynamic pstd(fig3Schema());
    for (auto& sub : fig3Subscriptions()) pstd.insertSubscription(sub);

    pstree::Event event = {{"attr2", pstree::Value(std::int64_t{2})}, {"attr3", pstree::Value(std::int64_t{10})}};
    auto matches = pstd.matchEvent(event);
    require(matches.size() == 1, "exactly one subscription should match {attr2:2, attr3:10}");
    require(contains(matches, 4), "S4 should be the one that matches");
}

// Broader per-subscription coverage: an event tailored to match EACH subscription
// individually (and only that one, where the six are mutually exclusive enough to check),
// exercising every access-predicate dimension and every group at least once.
void test_individual_subscription_matches() {
    pstree::PSTDynamic pstd(fig3Schema());
    for (auto& sub : fig3Subscriptions()) pstd.insertSubscription(sub);

    // S1: attr1<0, attr2 in [1,5]. attr1=-1, attr2=3 -> only S1 (S2 needs attr1>8).
    {
        pstree::Event e = {{"attr1", pstree::Value(std::int64_t{-1})}, {"attr2", pstree::Value(std::int64_t{3})}};
        auto m = pstd.matchEvent(e);
        require(m.size() == 1 && contains(m, 1), "attr1=-1,attr2=3 should match only S1");
    }
    // S2: attr1>8, attr2 in [1,5]. attr1=9, attr2=3.
    {
        pstree::Event e = {{"attr1", pstree::Value(std::int64_t{9})}, {"attr2", pstree::Value(std::int64_t{3})}};
        auto m = pstd.matchEvent(e);
        require(m.size() == 1 && contains(m, 2), "attr1=9,attr2=3 should match only S2");
    }
    // S3: attr2 in [1,5], attr3<0. attr2=3, attr3=-1.
    {
        pstree::Event e = {{"attr2", pstree::Value(std::int64_t{3})}, {"attr3", pstree::Value(std::int64_t{-1})}};
        auto m = pstd.matchEvent(e);
        require(m.size() == 1 && contains(m, 3), "attr2=3,attr3=-1 should match only S3");
    }
    // S5: attr1=2, attr2<1, attr3>8.
    {
        pstree::Event e = {{"attr1", pstree::Value(std::int64_t{2})}, {"attr2", pstree::Value(std::int64_t{0})}, {"attr3", pstree::Value(std::int64_t{9})}};
        auto m = pstd.matchEvent(e);
        require(m.size() == 1 && contains(m, 5), "attr1=2,attr2=0,attr3=9 should match only S5");
    }
    // S6: attr1<0, attr2>5, attr3=6.
    {
        pstree::Event e = {{"attr1", pstree::Value(std::int64_t{-5})}, {"attr2", pstree::Value(std::int64_t{6})}, {"attr3", pstree::Value(std::int64_t{6})}};
        auto m = pstd.matchEvent(e);
        require(m.size() == 1 && contains(m, 6), "attr1=-5,attr2=6,attr3=6 should match only S6");
    }
    // No dimensions present at all -> nothing can match (every subscription needs at least
    // one attribute the event doesn't have).
    {
        pstree::Event e = {};
        auto m = pstd.matchEvent(e);
        require(m.empty(), "an empty event should match nothing");
    }
}

// Native DECIMAL32/64/128/256 support: kDecimal dimension coverage mirroring the per-type
// pattern every other ValueType already gets in this file - equality, all four ordering ops,
// kBetween, kElemOf/kNotElemOf. Values here are already-scaled Int256s (as if rescaled to this
// attribute's own AttrSchema::decimalScale by the caller - pstree itself never rescales, see
// that field's own comment), built from plain int64_t via sign-extension so the test stays
// readable without hand-writing 256-bit literals.
void test_decimal_predicates() {
    using pstree::CmpOp;
    auto dec = [](std::int64_t v) {
        pstree::Int256 r;
        std::uint64_t bits = static_cast<std::uint64_t>(v);
        std::uint64_t fill = (v < 0) ? ~std::uint64_t{0} : std::uint64_t{0};
        r.limb = {bits, fill, fill, fill};
        return r;
    };

    std::vector<pstree::AttrSchema> schema = {
        {"amount", pstree::ValueType::kDecimal, 0, nullptr, 2}, // canonical scale 2 (e.g. cents)
    };
    pstree::PSTDynamic pstd(schema);

    pstree::Subscription eq{1, {P("amount", CmpOp::kEq, {pstree::Value(dec(12345))})}};
    pstree::Subscription lt{2, {P("amount", CmpOp::kLt, {pstree::Value(dec(0))})}};
    pstree::Subscription le{3, {P("amount", CmpOp::kLe, {pstree::Value(dec(-100))})}};
    pstree::Subscription gt{4, {P("amount", CmpOp::kGt, {pstree::Value(dec(1000000))})}};
    pstree::Subscription ge{5, {P("amount", CmpOp::kGe, {pstree::Value(dec(1000000))})}};
    pstree::Subscription between{6, {P("amount", CmpOp::kBetween, {pstree::Value(dec(100)), pstree::Value(dec(200))})}};
    pstree::Subscription elemOf{7, {P("amount", CmpOp::kElemOf,
        {pstree::Value(dec(7)), pstree::Value(dec(9)), pstree::Value(dec(11))})}};
    pstree::Subscription notElemOf{8, {P("amount", CmpOp::kNotElemOf,
        {pstree::Value(dec(7)), pstree::Value(dec(9)), pstree::Value(dec(11))})}};

    for (auto* s : {&eq, &lt, &le, &gt, &ge, &between, &elemOf, &notElemOf}) pstd.insertSubscription(*s);

    auto matchAmount = [&](std::int64_t v) {
        pstree::Event e = {{"amount", pstree::Value(dec(v))}};
        return pstd.matchEvent(e);
    };

    { auto m = matchAmount(12345); require(contains(m, 1), "decimal eq matches 12345"); }
    { auto m = matchAmount(0); require(!contains(m, 1), "decimal eq should not match 0");
      require(!contains(m, 2), "decimal lt(0) is strict, should not match 0 itself"); }
    // Sign-boundary: values differing only in sign, equal magnitude - a real risk spot for a
    // fixed-point two's-complement encoding the way -0.0/+0.0 was for DoubleCodec.
    { auto m = matchAmount(-1); require(contains(m, 2), "decimal lt(0) matches -1 (sign boundary)"); }
    { auto m = matchAmount(1); require(!contains(m, 2), "decimal lt(0) should not match +1"); }
    { auto m = matchAmount(-100); require(contains(m, 3), "decimal le(-100) matches -100 itself"); }
    { auto m = matchAmount(-101); require(contains(m, 3), "decimal le(-100) matches -101"); }
    { auto m = matchAmount(-99); require(!contains(m, 3), "decimal le(-100) should not match -99"); }
    { auto m = matchAmount(1000001); require(contains(m, 4), "decimal gt(1000000) matches 1000001");
      require(contains(m, 5), "decimal ge(1000000) matches 1000001"); }
    { auto m = matchAmount(1000000); require(!contains(m, 4), "decimal gt(1000000) should not match itself");
      require(contains(m, 5), "decimal ge(1000000) matches itself"); }
    { auto m = matchAmount(150); require(contains(m, 6), "decimal between(100,200) matches 150"); }
    { auto m = matchAmount(99); require(!contains(m, 6), "decimal between(100,200) should not match 99"); }
    { auto m = matchAmount(201); require(!contains(m, 6), "decimal between(100,200) should not match 201"); }
    { auto m = matchAmount(9); require(contains(m, 7), "decimal elemOf matches 9");
      require(!contains(m, 8), "decimal notElemOf should not match 9"); }
    { auto m = matchAmount(10); require(!contains(m, 7), "decimal elemOf should not match 10");
      require(contains(m, 8), "decimal notElemOf matches 10"); }
}

// Delete lifecycle: deleting S4 should remove it from future matches without disturbing S3
// (which shares S4's access-predicate leaf and, plausibly, its dimension-signature group).
void test_delete_subscription() {
    pstree::PSTDynamic pstd(fig3Schema());
    for (auto& sub : fig3Subscriptions()) pstd.insertSubscription(sub);

    pstd.deleteSubscription(4);

    pstree::Event event = {{"attr2", pstree::Value(std::int64_t{2})}, {"attr3", pstree::Value(std::int64_t{10})}};
    auto matches = pstd.matchEvent(event);
    require(matches.empty(), "after deleting S4, {attr2:2,attr3:10} should match nothing");

    pstree::Event s3Event = {{"attr2", pstree::Value(std::int64_t{3})}, {"attr3", pstree::Value(std::int64_t{-1})}};
    auto s3Matches = pstd.matchEvent(s3Event);
    require(s3Matches.size() == 1 && contains(s3Matches, 3), "S3 should be unaffected by deleting S4");

    bool threw = false;
    try {
        pstd.deleteSubscription(4);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "deleting an already-deleted subscription id should throw");
}

// kElemOf as the (only, forced) access predicate: decomposed into one kEq per literal
// value, all pointing at the same subscription - verifies both directions (insert makes it
// findable via ANY of the listed values; delete removes it via all of them).
void test_elem_of_access_predicate() {
    std::vector<pstree::AttrSchema> schema = {{"color", pstree::ValueType::kString, 16}};
    pstree::PSTDynamic pstd(schema);
    pstree::Subscription sub{10, {pstree::SubPredicate{"color", pstree::CmpOp::kElemOf,
                                                        {std::string("red"), std::string("green"), std::string("blue")}}}};
    pstd.insertSubscription(sub);

    for (const std::string& color : {"red", "green", "blue"}) {
        pstree::Event e = {{"color", pstree::Value(color)}};
        auto m = pstd.matchEvent(e);
        require(m.size() == 1 && contains(m, 10), "color='" + color + "' should match the elem-of subscription");
    }
    pstree::Event miss = {{"color", pstree::Value(std::string("yellow"))}};
    require(pstd.matchEvent(miss).empty(), "color='yellow' should not match the elem-of subscription");

    pstd.deleteSubscription(10);
    for (const std::string& color : {"red", "green", "blue"}) {
        pstree::Event e = {{"color", pstree::Value(color)}};
        require(pstd.matchEvent(e).empty(), "after delete, color='" + color + "' should match nothing");
    }
}

// kNe as the (only, forced) access predicate: no representable contiguous range, so it
// falls back to "matches every leaf" (see pst_dynamic.hpp) - correctness (not pruning
// efficiency) is what's being checked here.
void test_ne_only_access_predicate_fallback() {
    std::vector<pstree::AttrSchema> schema = {{"status", pstree::ValueType::kInteger, 0}};
    pstree::PSTDynamic pstd(schema);
    pstree::Subscription sub{20, {pstree::SubPredicate{"status", pstree::CmpOp::kNe, {std::int64_t{5}}}}};
    pstd.insertSubscription(sub);

    require(pstd.matchEvent({{"status", pstree::Value(std::int64_t{1})}}).size() == 1, "status=1 (!=5) should match");
    require(pstd.matchEvent({{"status", pstree::Value(std::int64_t{5})}}).empty(), "status=5 should NOT match (!=5 fails)");
    require(pstd.matchEvent({{"status", pstree::Value(std::int64_t{-1000000})}}).size() == 1, "a far-below value should still match");
    require(pstd.matchEvent({{"status", pstree::Value(std::int64_t{1000000})}}).size() == 1, "a far-above value should still match");

    pstd.deleteSubscription(20);
    require(pstd.matchEvent({{"status", pstree::Value(std::int64_t{1})}}).empty(), "after delete, nothing should match");
}

// kIsNotNull as the (only, forced) access predicate: unlike kNe, this is actually the
// CORRECT indexing, not just a safe fallback - see pst_dynamic.hpp's own comment.
void test_is_not_null_only_access_predicate() {
    std::vector<pstree::AttrSchema> schema = {{"discount", pstree::ValueType::kInteger, 0}};
    pstree::PSTDynamic pstd(schema);
    pstree::Subscription sub{30, {pstree::SubPredicate{"discount", pstree::CmpOp::kIsNotNull, {}}}};
    pstd.insertSubscription(sub);

    require(pstd.matchEvent({{"discount", pstree::Value(std::int64_t{0})}}).size() == 1, "discount present (any value) should match");
    require(pstd.matchEvent({{"discount", pstree::Value(std::int64_t{-500})}}).size() == 1, "discount present (negative) should still match");
    require(pstd.matchEvent({}).empty(), "discount absent should not match 'is not null'");

    pstd.deleteSubscription(30);
    require(pstd.matchEvent({{"discount", pstree::Value(std::int64_t{0})}}).empty(), "after delete, nothing should match");
}

// kIsNull cannot be an access predicate at all (see pst_dynamic.hpp's own comment on
// selectAccPredIndex) - a subscription whose ONLY predicate is "is null" must be rejected
// clearly at insert time, not silently accepted as unmatchable.
void test_is_null_only_predicate_rejected() {
    std::vector<pstree::AttrSchema> schema = {{"discount", pstree::ValueType::kInteger, 0}};
    pstree::PSTDynamic pstd(schema);
    pstree::Subscription sub{40, {pstree::SubPredicate{"discount", pstree::CmpOp::kIsNull, {}}}};
    bool threw = false;
    try {
        pstd.insertSubscription(sub);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "a subscription whose only predicate is 'is null' should be rejected at insert time");
}

// kIsNull alongside a normal, indexable predicate: the normal predicate becomes the access
// predicate (kIsNull always ranks worst), and kIsNull is evaluated correctly during the
// final per-subscription check - the realistic pattern ("price>100 and discount is null").
void test_is_null_with_indexable_predicate() {
    std::vector<pstree::AttrSchema> schema = {
        {"price", pstree::ValueType::kInteger, 0},
        {"discount", pstree::ValueType::kInteger, 0},
    };
    pstree::PSTDynamic pstd(schema);
    pstree::Subscription sub{50, {
        pstree::SubPredicate{"price", pstree::CmpOp::kGt, {std::int64_t{100}}},
        pstree::SubPredicate{"discount", pstree::CmpOp::kIsNull, {}},
    }};
    pstd.insertSubscription(sub);

    require(pstd.matchEvent({{"price", pstree::Value(std::int64_t{150})}}).size() == 1,
            "price>100 with discount absent entirely should match");
    require(pstd.matchEvent({{"price", pstree::Value(std::int64_t{150})}, {"discount", pstree::Value(std::int64_t{5})}}).empty(),
            "price>100 but discount present should not match");
    require(pstd.matchEvent({{"price", pstree::Value(std::int64_t{50})}}).empty(),
            "price<=100 should not match regardless of discount");

    pstd.deleteSubscription(50);
    require(pstd.matchEvent({{"price", pstree::Value(std::int64_t{150})}}).empty(), "after delete, nothing should match");
}

// Forces at least one group-reorganization (growing DimSigLen) by inserting more
// subscriptions on the same leaf than the initial threshold allows, then confirms every
// one of them - old and new - still matches/doesn't-match correctly, proving
// ReorganizeGroups preserves correctness across a real length change, not just leaves the
// group count unchanged.
void test_reorganize_groups_preserves_correctness() {
    std::vector<pstree::AttrSchema> schema = {
        {"acc", pstree::ValueType::kInteger, 0}, // shared access-predicate dimension, forces all subs onto one leaf
    };
    // Extra dimensions so different subscriptions get different dimension signatures,
    // exercising real group diversity within the one leaf being stressed.
    for (int i = 0; i < 10; ++i) {
        schema.push_back({"dim" + std::to_string(i), pstree::ValueType::kInteger, 0});
    }
    pstree::PSTDynamic pstd(schema);

    constexpr int kNumSubs = 40; // comfortably past the initial grow threshold (4)
    for (int i = 0; i < kNumSubs; ++i) {
        pstree::Subscription sub;
        sub.id = static_cast<std::uint64_t>(i + 1);
        sub.predicates.push_back(pstree::SubPredicate{"acc", pstree::CmpOp::kEq, {std::int64_t{0}}});
        // Each subscription also requires its own dedicated extra dimension, so it only
        // matches events that include that specific dimension too.
        sub.predicates.push_back(pstree::SubPredicate{"dim" + std::to_string(i % 10), pstree::CmpOp::kEq, {std::int64_t{i}}});
        pstd.insertSubscription(sub);
    }

    for (int i = 0; i < kNumSubs; ++i) {
        pstree::Event e = {{"acc", pstree::Value(std::int64_t{0})}, {"dim" + std::to_string(i % 10), pstree::Value(std::int64_t{i})}};
        auto m = pstd.matchEvent(e);
        require(m.size() == 1 && contains(m, static_cast<std::uint64_t>(i + 1)),
                "subscription " + std::to_string(i + 1) + " should match its own tailored event after reorganization");
    }

    // An event matching the access predicate but with the WRONG dim value for every
    // subscription should match nothing.
    pstree::Event none = {{"acc", pstree::Value(std::int64_t{0})}, {"dim0", pstree::Value(std::int64_t{-1})}};
    require(pstd.matchEvent(none).empty(), "an event satisfying the access predicate but no subscription's own extra predicate should match nothing");
}

// String interning (see pst_dynamic.hpp's StringInternTable): a string well past the OLD
// StringCodec truncation bound (formerly kPstreeStringMaxLen=32 in nats_sidecar's own usage,
// and this schema's own stringMaxLen field is now unused/ignored) must still match exactly -
// interning has no length limit at all, unlike the byte-per-element codec it replaced.
void test_string_interning_no_length_limit() {
    std::vector<pstree::AttrSchema> schema = {{"note", pstree::ValueType::kString, 16}};
    pstree::PSTDynamic pstd(schema);

    std::string longValue(200, 'x'); // far past the old 32-byte cap
    std::string longValueDiffTail = longValue.substr(0, 199) + "y"; // differs only in the last byte
    pstree::Subscription sub{1, {pstree::SubPredicate{"note", pstree::CmpOp::kEq, {longValue}}}};
    pstd.insertSubscription(sub);

    require(pstd.matchEvent({{"note", pstree::Value(longValue)}}).size() == 1,
            "a 200-byte string should match itself exactly");
    require(pstd.matchEvent({{"note", pstree::Value(longValueDiffTail)}}).empty(),
            "a 200-byte string differing only in its last byte must NOT match - the old "
            "codec would have silently confused these past its truncation bound, interning "
            "never does");
}

// Sentinel-miss guarantee at a scale where an id collision would be easy to introduce by
// accident (e.g. an off-by-one in id allocation starting the sentinel and the first real id
// at the same value): insert many distinct strings, then confirm a value NONE of them
// reference matches nothing, for several never-referenced probes, not just one.
void test_string_interning_sentinel_miss_at_scale() {
    std::vector<pstree::AttrSchema> schema = {{"tag", pstree::ValueType::kString, 16}};
    pstree::PSTDynamic pstd(schema);

    constexpr int kNumSubs = 500;
    for (int i = 0; i < kNumSubs; ++i) {
        pstree::Subscription sub;
        sub.id = static_cast<std::uint64_t>(i + 1);
        sub.predicates.push_back(
            pstree::SubPredicate{"tag", pstree::CmpOp::kEq, {std::string("tag-") + std::to_string(i)}});
        pstd.insertSubscription(sub);
    }

    for (int i = 0; i < kNumSubs; ++i) {
        auto m = pstd.matchEvent({{"tag", pstree::Value(std::string("tag-") + std::to_string(i))}});
        require(m.size() == 1 && contains(m, static_cast<std::uint64_t>(i + 1)),
                "tag-" + std::to_string(i) + " should match exactly subscription " + std::to_string(i + 1));
    }
    for (const std::string& probe : {"never-seen", "tag-", "tag-500", "TAG-0", ""}) {
        require(pstd.matchEvent({{"tag", pstree::Value(probe)}}).empty(),
                "never-referenced value '" + probe + "' must match nothing, not collide with a real id");
    }
}

// The ordering-op guardrail (see buildLowLevel's own comment): even though nats_sidecar's own
// shared grammar can never produce an ordering predicate against a string attribute (be-tree's
// parser.y's num_comp_value is integer/float only - confirmed by reading it directly), this
// locks in that invariant at pstree's own API boundary, in case a future caller ever
// constructs a SubPredicate directly, bypassing the dialect layer.
void test_string_ordering_predicate_rejected() {
    std::vector<pstree::AttrSchema> schema = {{"label", pstree::ValueType::kString, 16}};
    pstree::PSTDynamic pstd(schema);

    for (auto op : {pstree::CmpOp::kLt, pstree::CmpOp::kLe, pstree::CmpOp::kGt, pstree::CmpOp::kGe}) {
        pstree::Subscription sub{1, {pstree::SubPredicate{"label", op, {std::string("m")}}}};
        bool threw = false;
        try {
            pstd.insertSubscription(sub);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        require(threw, "an ordering predicate against a string attribute should be rejected");
    }

    pstree::Subscription betweenSub{
        2, {pstree::SubPredicate{"label", pstree::CmpOp::kBetween, {std::string("a"), std::string("z")}}}};
    bool threw = false;
    try {
        pstd.insertSubscription(betweenSub);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "a kBetween predicate against a string attribute should be rejected");
}

// Regression for a real bug caught by test_pst_dynamic_stress.cpp's own randomized fuzzing: the
// ordering-op-vs-string guardrail originally only checked whichever predicate selectAccPredIndex
// chose as the access predicate - too narrow, since string interning applies to ALL of a
// subscription's string-typed predicate values, not just the access predicate's. A NON-access
// ordering predicate on a string dimension would silently compare wrong (non-order-preserving)
// interned integers instead of being rejected. This subscription's "code" predicate (kEq, high
// selectivity) is always chosen as the access predicate over "label" (kLt, lower selectivity per
// opRank) - so without the fix, "label"'s kLt would slip through uncaught.
void test_string_ordering_predicate_rejected_when_not_the_access_predicate() {
    std::vector<pstree::AttrSchema> schema = {
        {"code", pstree::ValueType::kInteger, 0}, {"label", pstree::ValueType::kString, 16}};
    pstree::PSTDynamic pstd(schema);
    pstree::Subscription sub{1, {pstree::SubPredicate{"code", pstree::CmpOp::kEq, {std::int64_t{7}}},
                                  pstree::SubPredicate{"label", pstree::CmpOp::kLt, {std::string("m")}}}};
    bool threw = false;
    try {
        pstd.insertSubscription(sub);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "an ordering predicate on a NON-access string attribute should also be rejected");
}

// The actual point of interning the full-verification path (not just the access predicate's own
// PS-Tree indexing - see pst_dynamic.hpp's insertSubscription/matchEvent comments): a
// subscription with TWO string kElemOf predicates, where only one can be the access predicate,
// must still match/reject correctly via the OTHER one's full-verification check.
void test_two_string_elem_of_predicates_non_access_one_still_correct() {
    std::vector<pstree::AttrSchema> schema = {{"exchange", pstree::ValueType::kString, 16},
                                               {"symbol", pstree::ValueType::kString, 16}};
    pstree::PSTDynamic pstd(schema);
    // Both kElemOf, same tier - "exchange" (narrower list) loses the cardinality tie-break once
    // "symbol" has been observed with more distinct values, but either way, EXACTLY one becomes
    // the access predicate and the other is checked via full verification only.
    pstree::Subscription sub{
        1, {pstree::SubPredicate{"exchange", pstree::CmpOp::kElemOf,
                                  {std::string("A"), std::string("B")}},
            pstree::SubPredicate{"symbol", pstree::CmpOp::kElemOf,
                                  {std::string("AAPL"), std::string("MSFT")}}}};
    pstd.insertSubscription(sub);

    require(pstd.matchEvent({{"exchange", pstree::Value(std::string("A"))},
                             {"symbol", pstree::Value(std::string("AAPL"))}})
                .size() == 1,
            "exchange=A/symbol=AAPL should match (both predicates satisfied)");
    require(pstd.matchEvent({{"exchange", pstree::Value(std::string("A"))},
                             {"symbol", pstree::Value(std::string("GOOG"))}})
                .empty(),
            "exchange=A/symbol=GOOG should NOT match - symbol fails whichever predicate it is "
            "(access or full-verification)");
    require(pstd.matchEvent({{"exchange", pstree::Value(std::string("Z"))},
                             {"symbol", pstree::Value(std::string("AAPL"))}})
                .empty(),
            "exchange=Z/symbol=AAPL should NOT match - exchange fails whichever predicate it is "
            "(access or full-verification)");
}

// Regression for the findAttr-to-index optimization (matchSubscriptionIndexed, see
// predicate.hpp): a subscription's predicates and an event's own pairs can each list the SAME
// set of attributes in a DIFFERENT order from each other and from the schema's own declaration
// order - attrIndex is resolved from the SCHEMA's fixed ordinal, not from either side's own
// listing order, so this must not matter. Schema declares volume, price, symbol (that order);
// the subscription's predicates and the event's own pairs both use a scrambled, different order
// from the schema AND from each other.
void test_indexed_match_independent_of_attribute_order() {
    std::vector<pstree::AttrSchema> schema = {
        {"volume", pstree::ValueType::kInteger, 0},
        {"price", pstree::ValueType::kFloat, 0},
        {"symbol", pstree::ValueType::kString, 16},
    };
    pstree::PSTDynamic pstd(schema);
    // Predicates listed symbol, volume, price - none of these match the schema's own order.
    pstree::Subscription sub{
        1, {pstree::SubPredicate{"symbol", pstree::CmpOp::kEq, {std::string("AAPL")}},
            pstree::SubPredicate{"volume", pstree::CmpOp::kGe, {std::int64_t{100}}},
            pstree::SubPredicate{"price", pstree::CmpOp::kLe, {50.0}}}};
    pstd.insertSubscription(sub);

    // Event pairs listed price, symbol, volume - yet a THIRD order, different from both the
    // schema and the subscription's own predicate order.
    pstree::Event matching = {{"price", pstree::Value(25.0)},
                               {"symbol", pstree::Value(std::string("AAPL"))},
                               {"volume", pstree::Value(std::int64_t{200})}};
    require(pstd.matchEvent(matching).size() == 1, "all three predicates satisfied should match regardless of attribute order anywhere");

    pstree::Event nonMatching = {{"price", pstree::Value(25.0)},
                                  {"symbol", pstree::Value(std::string("MSFT"))},
                                  {"volume", pstree::Value(std::int64_t{200})}};
    require(pstd.matchEvent(nonMatching).empty(), "symbol mismatch should still correctly fail regardless of attribute order");
}

// Regression for matchSubscriptionIndexed's own "attribute absent from this event" path
// (indexed[i] == nullptr) - the indexed equivalent of findAttr() returning nullptr.
void test_indexed_match_missing_event_attribute() {
    std::vector<pstree::AttrSchema> schema = {
        {"volume", pstree::ValueType::kInteger, 0},
        {"symbol", pstree::ValueType::kString, 16},
    };
    pstree::PSTDynamic pstd(schema);
    pstree::Subscription sub{
        1, {pstree::SubPredicate{"volume", pstree::CmpOp::kGe, {std::int64_t{100}}},
            pstree::SubPredicate{"symbol", pstree::CmpOp::kEq, {std::string("AAPL")}}}};
    pstd.insertSubscription(sub);

    // Event has volume but not symbol at all - the subscription's symbol predicate must fail
    // the whole match, exactly like matchSubscription()'s own documented "absent attribute
    // always fails" rule.
    pstree::Event partial = {{"volume", pstree::Value(std::int64_t{500})}};
    require(pstd.matchEvent(partial).empty(), "an event missing an attribute a predicate needs should not match");
}

// Regression for kIsNull/kIsNotNull through matchSubscriptionIndexed specifically (both are
// intercepted before ever reaching matchValue, on both the name-based and indexed paths - this
// confirms the indexed path's own interception is correct, not just inherited by construction).
void test_indexed_match_is_null_is_not_null() {
    std::vector<pstree::AttrSchema> schema = {
        {"code", pstree::ValueType::kInteger, 0},
        {"discount", pstree::ValueType::kInteger, 0},
    };
    pstree::PSTDynamic pstd(schema);
    pstree::Subscription subNotNull{
        1, {pstree::SubPredicate{"code", pstree::CmpOp::kEq, {std::int64_t{1}}},
            pstree::SubPredicate{"discount", pstree::CmpOp::kIsNotNull, {}}}};
    pstd.insertSubscription(subNotNull);

    require(pstd.matchEvent({{"code", pstree::Value(std::int64_t{1})},
                             {"discount", pstree::Value(std::int64_t{5})}})
                .size() == 1,
            "code=1 with discount present should match kIsNotNull");
    require(pstd.matchEvent({{"code", pstree::Value(std::int64_t{1})}}).empty(),
            "code=1 with discount absent should NOT match kIsNotNull");
}

// Direct differential test between the two match functions themselves (predicate.hpp), isolated
// from PSTDynamic entirely - the most direct correctness guarantee for matchSubscriptionIndexed:
// for the SAME logical event/subscription, expressed both ways (name-based Event vs. an indexed
// array), the two functions must agree.
void test_match_subscription_indexed_agrees_with_name_based() {
    pstree::Subscription sub{
        1, {pstree::SubPredicate{"a", pstree::CmpOp::kGe, {std::int64_t{10}}},
            pstree::SubPredicate{"b", pstree::CmpOp::kEq, {std::string("x")}}}};
    for (auto& pred : sub.predicates) pstree::ensurePredicateCachedForInsert(pred);
    // Schema ordinal: a=0, b=1, c=2 (c unused by this subscription, present to exercise a
    // nullptr slot in `indexed` that this subscription never reads).
    sub.predicates[0].attrIndex = 0;
    sub.predicates[1].attrIndex = 1;

    auto check = [&](std::optional<std::int64_t> aVal, std::optional<std::string> bVal, bool expected,
                      const std::string& label) {
        pstree::Event event;
        std::vector<const pstree::Value*> indexed(3, nullptr);
        pstree::Value aStorage, bStorage;
        if (aVal) {
            aStorage = pstree::Value(*aVal);
            event.push_back({"a", aStorage});
            indexed[0] = &aStorage;
        }
        if (bVal) {
            bStorage = pstree::Value(*bVal);
            event.push_back({"b", bStorage});
            indexed[1] = &bStorage;
        }
        bool nameBased = pstree::matchSubscription(event, sub);
        bool indexBased = pstree::matchSubscriptionIndexed(indexed, sub);
        require(nameBased == indexBased, label + ": matchSubscription/matchSubscriptionIndexed disagreed");
        require(nameBased == expected, label + ": unexpected result");
    };

    check(15, std::string("x"), true, "both satisfied");
    check(5, std::string("x"), false, "a fails (below threshold)");
    check(15, std::string("y"), false, "b fails (wrong string)");
    check(std::nullopt, std::string("x"), false, "a absent from event entirely");
    check(15, std::nullopt, false, "b absent from event entirely");
}

// Direct unit test for matchSubscriptionIndexedSkippingAccessPredicate (predicate.hpp) - the
// Phase 3 optimization skipping a subscription's own access predicate at match time, since
// PSTree::matchPoint() already exactly proved it (see that function's own comment for the full
// argument, and PSTDynamic::detail::accessPredicateProvenExactlyByTreeMembership for exactly
// which operators this applies to). Confirms: (1) skipping index i truly ignores predicate i's
// own condition, even when the event's actual value would fail it - proving the skip is real, not
// accidentally still checking; (2) skipping the out-of-range sentinel (size()) behaves identically
// to the full matchSubscriptionIndexed check on every predicate; (3) a single-predicate
// subscription with its sole predicate skipped unconditionally returns true.
void test_match_subscription_indexed_skipping_access_predicate() {
    pstree::Subscription sub{
        1, {pstree::SubPredicate{"a", pstree::CmpOp::kGe, {std::int64_t{10}}},
            pstree::SubPredicate{"b", pstree::CmpOp::kEq, {std::string("x")}}}};
    for (auto& pred : sub.predicates) pstree::ensurePredicateCachedForInsert(pred);
    sub.predicates[0].attrIndex = 0;
    sub.predicates[1].attrIndex = 1;

    pstree::Value aFail(std::int64_t{5});   // fails "a >= 10"
    pstree::Value bFail(std::string("y"));  // fails "b == x"
    pstree::Value aPass(std::int64_t{15});
    pstree::Value bPass(std::string("x"));
    std::vector<const pstree::Value*> indexedBothFail{&aFail, &bFail};
    std::vector<const pstree::Value*> indexedBothPass{&aPass, &bPass};
    std::vector<const pstree::Value*> indexedAFailBPass{&aFail, &bPass};
    std::vector<const pstree::Value*> indexedAPassBFail{&aPass, &bFail};

    // Skipping index 0 (the failing "a") must still return true, since only "b" gets checked.
    require(pstree::matchSubscriptionIndexedSkippingAccessPredicate(indexedAFailBPass, sub, 0),
            "skipping a's own failing predicate must ignore it, b alone passes");
    // Skipping index 1 (the failing "b") must still return true, since only "a" gets checked.
    require(pstree::matchSubscriptionIndexedSkippingAccessPredicate(indexedAPassBFail, sub, 1),
            "skipping b's own failing predicate must ignore it, a alone passes");
    // Skipping a passing predicate changes nothing observable here - still true either way.
    require(pstree::matchSubscriptionIndexedSkippingAccessPredicate(indexedBothPass, sub, 0),
            "both pass regardless of which is skipped");

    // Sentinel (out-of-range index) must behave identically to the full check on every case.
    for (auto* indexed : {&indexedBothFail, &indexedBothPass, &indexedAFailBPass, &indexedAPassBFail}) {
        bool full = pstree::matchSubscriptionIndexed(*indexed, sub);
        bool skipNothing = pstree::matchSubscriptionIndexedSkippingAccessPredicate(
            *indexed, sub, sub.predicates.size());
        require(full == skipNothing, "sentinel skip-index must agree with the full check");
    }
    require(!pstree::matchSubscriptionIndexedSkippingAccessPredicate(indexedBothFail, sub, sub.predicates.size()),
            "sentinel skip-index with both predicates failing must still return false");

    // A single-predicate subscription with its sole (skipped) predicate must unconditionally
    // return true, regardless of the event's actual value on that attribute.
    pstree::Subscription single{2, {pstree::SubPredicate{"a", pstree::CmpOp::kGt, {std::int64_t{1000}}}}};
    for (auto& pred : single.predicates) pstree::ensurePredicateCachedForInsert(pred);
    single.predicates[0].attrIndex = 0;
    pstree::Value aWayBelow(std::int64_t{-999}); // would fail "a > 1000" if actually checked
    std::vector<const pstree::Value*> indexedSingle{&aWayBelow};
    require(pstree::matchSubscriptionIndexedSkippingAccessPredicate(indexedSingle, single, 0),
            "single-predicate subscription with its sole predicate skipped must return true unconditionally");
    require(!pstree::matchSubscriptionIndexed(indexedSingle, single),
            "sanity check: the FULL check on the same data must actually fail (proves the skip is doing real work)");
}

} // namespace

int main() {
    test_select_acc_pred_matches_paper();
    test_select_acc_pred_prefers_higher_cardinality_dimension();
    test_delete_uses_stored_access_predicate_not_a_recomputed_one();
    test_section_5_3_event_matching();
    test_individual_subscription_matches();
    test_decimal_predicates();
    test_delete_subscription();
    test_elem_of_access_predicate();
    test_ne_only_access_predicate_fallback();
    test_is_not_null_only_access_predicate();
    test_is_null_only_predicate_rejected();
    test_is_null_with_indexable_predicate();
    test_reorganize_groups_preserves_correctness();
    test_string_interning_no_length_limit();
    test_string_interning_sentinel_miss_at_scale();
    test_string_ordering_predicate_rejected();
    test_string_ordering_predicate_rejected_when_not_the_access_predicate();
    test_two_string_elem_of_predicates_non_access_one_still_correct();
    test_indexed_match_independent_of_attribute_order();
    test_indexed_match_missing_event_attribute();
    test_indexed_match_is_null_is_not_null();
    test_match_subscription_indexed_agrees_with_name_based();
    test_match_subscription_indexed_skipping_access_predicate();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All pst_dynamic tests passed\n";
    return 0;
}
