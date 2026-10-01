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
    uint32_t neutral_votes = 0;
};

// Input must be a valid guest-generated SPIR-V module, not an unrestricted Vulkan peephole or
// a replacement for spirv-val. Each Any needs an independent certificate: a predicate defined
// uniformly across the intended logical guest wave (including helpers), a finite known-pure SSA
// user graph with no control/effect consumer, or the neutral selection described below. This is
// NOT a claim that a
// per-draw value is uniform across an arbitrary host Vulkan subgroup: fragment subgroups may mix
// commands. The guest's complete-wave scalar EXEC/VCC test is the semantic target; its Any emission
// is an implementation of that target, not unrestricted source-SPIR-V optimization authority.
// A common-entry one-arm selection additionally permits TRUE control only after proving every
// extra operation UB-free and every live merge export stable and unchanged under P=false. P stays
// varying; an output-dead branch/loop alone is NOT that certificate. Strict replay never calls
// this transformation. A caller may certify buffer inputs immutable only after proving that all
// shaders in the pass leave them read-only. Buffer-derived predicates additionally require an
// enabled deterministic robust2 storage-read contract (32-bit aligned range, no rounded padding).
// Only the guest's single runtime-array word-buffer layout is certified; ordinary robust access,
// fixed-array/UBO/push-constant address guesses are not range/definedness authority.
// Word indexes require a non-wrapping stride4 range proof; Float32 word loads require the entry
// point's SignedZeroInfNanPreserve contract, and floating comparisons also require DenormPreserve.
// Raw-bit predicates use integer loads; float-to-bits casts do not certify uniformity (sNaN
// quieting is permitted). External writes/atomics/opaque calls retain their rendezvous obligation.
// Initialized scalar Boolean/Int32 recurrences additionally require a closed, grounded proof of
// EVERY entry-function branch/switch selector. Equal values at matching visits do not imply equal
// values across iterations or bounded addresses; recurrence facts never expand load admission.
FragmentVoteLowering lower_fragment_votes(const std::vector<uint32_t>& source,
                                         bool immutable_storage_inputs = false,
                                         bool deterministic_storage_reads = false);

} // namespace prosper::gpu
