// Correctness tests for dim_sig.hpp's Bloom-filter-based dimension signature: two
// signatures built from the SAME dimension set (at the same bit width) must be identical
// (needed for grouping - see pst_dynamic.hpp), and isSubsetOf must correctly implement
// "every dimension a group needs is present in the event", which is what Algorithm 5's
// group-level pruning actually depends on.

#include <iostream>
#include <string>
#include <vector>

#include "pstree/dim_sig.hpp"

namespace {

int g_failures = 0;

void require(bool cond, const std::string& message) {
    if (!cond) {
        std::cerr << "FAIL: " << message << "\n";
        g_failures++;
    }
}

void test_same_dims_same_signature() {
    std::vector<std::string> dimsA = {"attr1", "attr2"};
    std::vector<std::string> dimsB = {"attr1", "attr2"};
    auto sigA = pstree::calculateDimSig(dimsA, 64);
    auto sigB = pstree::calculateDimSig(dimsB, 64);
    require(sigA == sigB, "the same dimension set at the same width must produce an identical signature");
}

void test_different_dims_usually_different_signature() {
    std::vector<std::string> dimsA = {"attr1", "attr2"};
    std::vector<std::string> dimsB = {"attr2", "attr3"};
    auto sigA = pstree::calculateDimSig(dimsA, 256); // wide enough that a collision would be surprising
    auto sigB = pstree::calculateDimSig(dimsB, 256);
    require(!(sigA == sigB), "different dimension sets at a generous width should (almost always) differ");
}

void test_subset_relationship() {
    std::vector<std::string> groupDims = {"attr1", "attr2"};
    std::vector<std::string> supersetDims = {"attr1", "attr2", "attr3"};
    std::vector<std::string> disjointDims = {"attr4", "attr5"};

    auto groupSig = pstree::calculateDimSig(groupDims, 256);
    auto eventSigSuperset = pstree::calculateDimSig(supersetDims, 256);
    auto eventSigDisjoint = pstree::calculateDimSig(disjointDims, 256);
    auto eventSigExact = pstree::calculateDimSig(groupDims, 256);

    require(groupSig.isSubsetOf(eventSigSuperset), "a group's dims should be a subset of an event containing them plus extras");
    require(groupSig.isSubsetOf(eventSigExact), "a group's dims should be a subset of an event with exactly those dims");
    require(!groupSig.isSubsetOf(eventSigDisjoint), "a group's dims should NOT be a subset of a disjoint event");
}

void test_missing_one_dimension_breaks_subset() {
    std::vector<std::string> groupDims = {"attr1", "attr2", "attr3"};
    std::vector<std::string> eventDims = {"attr1", "attr2"}; // missing attr3
    auto groupSig = pstree::calculateDimSig(groupDims, 256);
    auto eventSig = pstree::calculateDimSig(eventDims, 256);
    require(!groupSig.isSubsetOf(eventSig), "missing even one required dimension should break the subset check");
}

void test_empty_dims() {
    std::vector<std::string> empty;
    auto sig = pstree::calculateDimSig(empty, 64);
    auto other = pstree::calculateDimSig(std::vector<std::string>{"anything"}, 64);
    require(sig.isSubsetOf(other), "an empty dimension signature is trivially a subset of anything");
}

} // namespace

int main() {
    test_same_dims_same_signature();
    test_different_dims_usually_different_signature();
    test_subset_relationship();
    test_missing_one_dimension_breaks_subset();
    test_empty_dims();

    if (g_failures > 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All dim_sig tests passed\n";
    return 0;
}
