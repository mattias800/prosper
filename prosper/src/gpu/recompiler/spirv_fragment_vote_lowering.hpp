#pragma once
#include <cstddef>
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

inline constexpr const char* fragment_vote_refusal_name(FragmentVoteRefusal refusal) {
    switch (refusal) {
        case FragmentVoteRefusal::None: return "none";
        case FragmentVoteRefusal::MalformedModule: return "malformed-module";
        case FragmentVoteRefusal::InconsistentContract: return "inconsistent-contract";
        case FragmentVoteRefusal::UnsupportedWaveOperation: return "unsupported-wave-operation";
        case FragmentVoteRefusal::UnprovedVote: return "unproved-vote";
    }
    return "unknown";
}

// Observer-only source location of the first vote whose certificate actually failed. An absent
// certificate does not prove that the predicate is varying or undefined. This is a SPIR-V dword
// offset (including the five-word header), NOT an RDNA guest instruction PC.
struct FragmentVoteFailureDetails {
    bool available = false;
    std::size_t source_word = 0;
    uint32_t vote_result_id = 0;
    uint32_t predicate_id = 0;
    uint32_t predicate_opcode = UINT32_MAX; // unavailable, not an invented OpNop
};

// Value-owned diagnostic survives both memoized results and destruction of an uncached result.
// Default construction means the lowering was not attempted, not that it succeeded.
struct FragmentVoteLoweringDiagnostic {
    bool attempted = false;
    FragmentVoteRefusal refusal = FragmentVoteRefusal::MalformedModule;
    FragmentVoteFailureDetails failed_vote;
};

struct FragmentVoteLowering {
    // Owned effective module; empty on refusal. The captured/source module is never modified.
    std::vector<uint32_t> words;
    FragmentVoteRefusal refusal = FragmentVoteRefusal::MalformedModule;
    uint32_t uniform_votes = 0;
    uint32_t dead_votes = 0;
    uint32_t neutral_votes = 0;
    FragmentVoteFailureDetails failed_vote;

    FragmentVoteLoweringDiagnostic diagnostic() const { return {true, refusal, failed_vote}; }
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
