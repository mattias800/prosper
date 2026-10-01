#include "gpu/recompiler/spirv_fragment_vote_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "../../fixtures/spirv_fragment_vote_fixtures.hpp"
#include <cstdio>

int main() {
    using namespace prosper::gpu;
    namespace f = prosper::test::fragment_votes;
    int failures = 0;
    const auto check = [&](bool ok, const char* name) {
        std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", name);
        failures += !ok;
    };
    for (const auto& fixture : f::fixtures()) {
        const auto unchanged = fixture.words;
        const auto lowered = lower_fragment_votes(fixture.words, fixture.immutable_storage, true);
        const bool admitted = lowered.refusal == FragmentVoteRefusal::None;
        check(admitted == fixture.admitted &&
              lowered.uniform_votes == fixture.uniform_votes && lowered.dead_votes == fixture.dead_votes &&
              fixture.words == unchanged && lowered.words.empty() == !admitted, fixture.name.c_str());
        if (!admitted) {
            check(lowered.refusal == (fixture.inventory_refusal ? FragmentVoteRefusal::UnsupportedWaveOperation :
                                                                FragmentVoteRefusal::UnprovedVote),
                  "refusal is the expected proof/inventory, not malformed input");
            continue;
        }
        check(f::find(lowered.words, 335) == lowered.words.size(), "effective module has no Any");
        check(fragment_spirv_required_subgroup_size(lowered.words) == 0 &&
              fragment_spirv_required_subgroup_reasons(lowered.words) == 0 &&
              fragment_spirv_required_subgroup_features(lowered.words) == 0, "effective contract is subgroup-free");
        size_t copies = 0;
        for (size_t at = 5; at < lowered.words.size(); at += lowered.words[at] >> 16)
            copies += (lowered.words[at] & 0xffffu) == 83;
        check(copies == lowered.uniform_votes + lowered.dead_votes, "Any becomes typed CopyObject, not a Boolean literal");
        for (size_t at = 5; at < fixture.words.size(); at += fixture.words[at] >> 16) {
            if ((fixture.words[at] & 0xffffu) != 335) continue;
            bool exact_copy = false;
            for (size_t effective = 5; effective < lowered.words.size(); effective += lowered.words[effective] >> 16)
                if (lowered.words[effective] == ((4u << 16) | 83u) &&
                    lowered.words[effective + 1] == fixture.words[at + 1] &&
                    lowered.words[effective + 2] == fixture.words[at + 2] &&
                    lowered.words[effective + 3] == fixture.words[at + 4]) exact_copy = true;
            check(exact_copy, "typed CopyObject preserves the exact vote result and predicate IDs");
        }
        check(lowered.words.size() + copies == fixture.words.size() - 4,
              "only vote lengths and two unused capabilities change module size");
    }
    const auto baseline = f::fixtures().front().words;
    check(lower_fragment_votes(f::storage_predicate(), true, false).refusal == FragmentVoteRefusal::UnprovedVote,
          "ordinary robust access is not a defined uniform-load certificate");
    const auto refusal = [&](std::vector<uint32_t> words, FragmentVoteRefusal expected, const char* name) {
        const auto lowered = lower_fragment_votes(words, true, true);
        check(lowered.refusal == expected && lowered.words.empty() &&
              lowered.uniform_votes == 0 && lowered.dead_votes == 0, name);
    };
    auto words = baseline;
    std::vector<uint32_t> metadata;
    f::marker(metadata, "Prosper.FragmentSubgroupSize=64");
    f::insert(words, f::find(words, 71), metadata);
    refusal(words, FragmentVoteRefusal::InconsistentContract, "duplicate size marker refused even when equal");
    words = baseline;
    const auto marker = f::find(words, 330);
    words.erase(words.begin() + marker, words.begin() + marker + (words[marker] >> 16));
    refusal(words, FragmentVoteRefusal::InconsistentContract, "missing marker never defaults to an admission proof");
    words = baseline;
    words[f::find(words, 335) + 3] = 9; // Device instead of Subgroup
    refusal(words, FragmentVoteRefusal::MalformedModule, "wrong vote scope refused");
    words = baseline;
    words[f::find(words, 335)] = (5u << 16) | 334u; // All cannot be smuggled past inventory
    refusal(words, FragmentVoteRefusal::UnsupportedWaveOperation, "another collective refused");
    words = baseline;
    words.back() = 0;
    refusal(words, FragmentVoteRefusal::MalformedModule, "zero word count refused");
    words = baseline;
    words.pop_back();
    refusal(words, FragmentVoteRefusal::MalformedModule, "unclosed function refused");
    std::printf("== %s: %d failures ==\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
