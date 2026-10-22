// Existing scalar merge-deadness proof and physical mask-width inputs.
#pragma once
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
struct Rdna2Inst;

// What a surviving read of R is allowed to be.
//
// AnyRead     -- R must not be read at all before it is redefined. The original and default proof.
// MaskDomainOnly -- R may be read as 32-bit scalar DATA, but not by a lane-mask consumer. Only
//                meaningful for a VCC half, and only sound because prosper models a VCC half written
//                by a B32 scalar op as scalar data with its own storage: a 32-bit read of that half
//                reproduces the written dword exactly, while a MASK read would need the per-lane
//                predicate the scalar model deliberately does not reconstruct.
//
// The distinction is 32-bit-ness, not explicitness, and that is the whole subtlety: a VOP3
// `v_cndmask_b32 v0, v1, v2, vcc` names VCC explicitly and still reads the mask domain. Every
// mask consumer reads the PAIR, so requiring `scalar_alu_source_words(...) == 1` excludes all of
// them -- VOP3 cndmask, VOP3B carry-in, `s_mov_b64 exec, vcc`, and B64 mask logic alike -- without
// enumerating them. The e32 implicit readers and the vccz/vccnz branches are rejected earlier and
// unconditionally, so they are excluded on both paths.
enum class ScalarMergeProof { AnyRead, MaskDomainOnly };

// Why the proof failed, for the reject message. A liveness proof that reports only "failed" makes
// every widening of it a guess: the first attempt at MaskDomainOnly did not accept the GTA V kernel
// it was written for, and without the blocking pc there was no way to tell whether the walk stopped
// at a mask read, a data read, or an unmodelled opcode -- three different follow-ups.
struct ScalarMergeBlocker {
    uint32_t pc = UINT32_MAX;
    const char* kind = "none";
};

struct ScalarMergeMaskWriteWidths {
    bool carry_pair = true;
    bool implicit_vcc_pair = true;
    bool explicit_vcc_pair = true;
};

bool sgpr_dead_at_merge(const std::vector<Rdna2Inst>& ins, uint32_t target, int R,
                        ScalarMergeProof proof = ScalarMergeProof::AnyRead,
                        ScalarMergeBlocker* blocker = nullptr,
                        // Explicit compare HIGH-word kills require the caller's pair-width
                        // proof. A Wave32 compare always replaces LOW but can leave HIGH live.
                        bool vopc_sgpr_pair_kills = false,
                        // Match physical mask writes to the caller's actual emitter mode.
                        ScalarMergeMaskWriteWidths mask_widths = {});
std::unordered_set<uint32_t> proven_cselect_b64_low_only_pcs(const std::vector<Rdna2Inst>& ins);
bool is_wave64_vcc_lo_scalar_b32_candidate(const Rdna2Inst& in);
std::unordered_set<uint32_t> proven_wave64_vcc_b32_low_only_pcs(const std::vector<Rdna2Inst>& ins,
                                                                bool include_logical = true,
                                                                bool include_high_cselect = false);
}   // namespace prosper::gpu
