#pragma once
#include <cstdint>
#include <vector>

namespace prosper::gpu {

enum class FragmentWavePolicy { Strict, ProvenVotes };

enum class FragmentVoteRefusal {
    None,
    MalformedModule,
    InconsistentContract,
    UnsupportedWaveOperation,
    UnprovedVote,
};

struct FragmentVoteLowering {
    // Owned effective module; empty on refusal. The captured/source module is never modified.
    std::vector<uint32_t> words;
    FragmentVoteRefusal refusal = FragmentVoteRefusal::MalformedModule;
    uint32_t uniform_votes = 0;
    uint32_t dead_votes = 0;
};

// Input must be a valid guest-generated SPIR-V module, not an unrestricted Vulkan peephole or
// a replacement for spirv-val. Every Any must have a defined
// draw-uniform predicate or a finite known-pure SSA user graph with no control/effect consumer.
// Output-dead branches/loops are deliberately not a rewrite certificate. Strict replay never calls
// this transformation. A caller may certify buffer inputs immutable only after proving that all
// shaders in the pass leave them read-only. Buffer-derived predicates additionally require an
// enabled deterministic robust2 storage-read contract (32-bit aligned range, no rounded padding).
// Only the guest's single runtime-array word-buffer layout is certified; ordinary robust access,
// fixed-array/UBO/push-constant address guesses are not range/definedness authority.
// Word indexes require a non-wrapping stride4 range proof; Float32 word loads require the entry
// point's SignedZeroInfNanPreserve contract, and floating comparisons also require DenormPreserve.
// Raw-bit predicates use integer loads; float-to-bits casts do not certify uniformity (sNaN
// quieting is permitted). External writes/atomics/opaque calls retain their rendezvous obligation.
FragmentVoteLowering lower_fragment_votes(const std::vector<uint32_t>& source,
                                         bool immutable_storage_inputs = false,
                                         bool deterministic_storage_reads = false);

} // namespace prosper::gpu
