#pragma once
// Diagnostic description of a fragment-vote lowering verdict, sunk from
// gpu/recompiler/spirv_fragment_vote_lowering.hpp (layer inversion: diagnostics
// may not include the recompiler). The lowering itself (FragmentVoteLowering,
// lower_fragment_votes) stays in gpu/; this header carries only the verdict
// vocabulary that diagnostics and tests report on. Namespace stays
// prosper::gpu, so no call-site qualification changes; renaming it is a
// follow-up.
#include <cstddef>
#include <cstdint>

namespace prosper::gpu {

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

} // namespace prosper::gpu
