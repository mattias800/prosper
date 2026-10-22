// Existing forward-if admission rules.
#include "gpu/recompiler/compiler_choices.hpp"
#include <cstdint>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_cfg_forward_admission.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdarg>
#include <cstdio>

namespace prosper::gpu {
void forward_if_refuse(ForwardIfRefusal* why, const char* format, ...) {
    if (!why) return;
    va_list args;
    va_start(args, format);
    std::vsnprintf(why->text, sizeof why->text, format, args);
    va_end(args);
}

// The divergent loop that owns `pc` as its back-edge, canonical exit or break, if any.
bool div_loop_backedge_at(const std::vector<DivLoop>* loops, uint32_t pc) {
    if (loops)
        for (const auto& L : *loops)
            if (L.backedge_pc == pc) return true;
    return false;
}
bool div_loop_exit_at(const std::vector<DivLoop>* loops, uint32_t pc) {
    if (loops)
        for (const auto& L : *loops)
            if (L.exit_branch_pc == pc) return true;
    return false;
}
const DivLoop* div_loop_break_at(const std::vector<DivLoop>* loops, uint32_t pc) {
    if (loops)
        for (const auto& L : *loops)
            for (uint32_t bp : L.break_pcs)
                if (bp == pc) return &L;
    return nullptr;
}
// `allow_vcc` and `compute_wave_branches` are detect_forward_ifs' parameters of the same name:
// graphics stages pass allow_vcc, compute passes compute_wave_branches.
ForwardIfBranchClass classify_forward_if_branch(const Rdna2Inst& in, bool allow_vcc,
                                                bool compute_wave_branches,
                                                const std::unordered_set<uint32_t>* skip,
                                                const std::vector<DivLoop>* loops,
                                                ForwardIfRefusal* why) {
    if (in.fmt != Rdna2Format::SOPP) return ForwardIfBranchClass::NotACondition;
    switch (in.opcode) {
        case 0x09:   // execnz: only as a loop back-edge
            if (div_loop_backedge_at(loops, in.pc)) return ForwardIfBranchClass::Claimed;
            forward_if_refuse(why, "unclaimed execnz pc=%u", in.pc);
            return ForwardIfBranchClass::Refused;
        case 0x08:   // execz: safe-linearized predication, a loop exit, or a wave-empty test
            if (skip && skip->count(in.pc)) return ForwardIfBranchClass::Claimed;
            if (div_loop_exit_at(loops, in.pc)) return ForwardIfBranchClass::Claimed;
            if (!allow_vcc && !compute_wave_branches) {
                forward_if_refuse(why, "unsupported execz pc=%u", in.pc);
                return ForwardIfBranchClass::Refused;
            }
            return ForwardIfBranchClass::If;
        case 0x06:
        case 0x07:   // vccz / vccnz
            if (div_loop_exit_at(loops, in.pc)) return ForwardIfBranchClass::Claimed;
            if (allow_vcc) return ForwardIfBranchClass::If;
            // Compute VCC is a per-lane bool in the SPIR-V shell. The scalar branch consumes the
            // architectural whole-wave VCCZ/VCCNZ flag. Accept it for the exact guest-wave CFG
            // dispatcher; emit_body must never lower it through a native subgroup vote.
            if (compute_wave_branches) return ForwardIfBranchClass::UniformVccIf;
            forward_if_refuse(why, "unsupported vcc branch pc=%u", in.pc);
            return ForwardIfBranchClass::Refused;
        case 0x04:
        case 0x05:   // scc0 / scc1
            if (div_loop_exit_at(loops, in.pc)) return ForwardIfBranchClass::Claimed;
            if (skip && skip->count(in.pc))
                return ForwardIfBranchClass::Claimed;   // kill-mask branch: linearized
            return ForwardIfBranchClass::If;
        default: return ForwardIfBranchClass::NotACondition;   // s_branch and hints
    }
}

// The forward target of an if candidate, or false when it has none. A target past s_endpgm is an
// early-out only when the code there terminates at once; it is then clamped to end_pc.
bool resolve_forward_if_target(const Rdna2Inst& in, uint32_t end_pc, const uint32_t* code,
                               size_t dwords, uint32_t* target, bool* early,
                               ForwardIfRefusal* why) {
    uint32_t tgt = branch_target(in);
    *early = false;
    if (tgt > end_pc && tgt != UINT32_MAX) {   // possible early-out past s_endpgm:
        if (!code || tgt >= dwords) {
            forward_if_refuse(why, "target out of range pc=%u target=%u", in.pc, tgt);
            return false;
        }
        std::vector<Rdna2Inst> tail;
        rdna2_walk(code + tgt, dwords - tgt, tail);
        bool ends_immediately = false;
        for (const auto& ti : tail) {
            if (ti.is_end) {
                ends_immediately = true;
                break;
            }
            if (ti.fmt == Rdna2Format::SOPP && ti.opcode == 0x00) continue;
            break;
        }
        if (!ends_immediately) {
            forward_if_refuse(why, "nonterminal early target pc=%u target=%u", in.pc, tgt);
            return false;
        }
        tgt = end_pc;
        *early = true;
    }
    if (tgt <= in.pc || tgt > end_pc) {
        forward_if_refuse(why, "nonforward target pc=%u target=%u", in.pc, tgt);
        return false;   // must be forward, within the stream
    }
    *target = tgt;
    return true;
}

// A compute execz region [branch, region_end): its condition is subgroup-uniform, so live scalar
// writes and LDS are safe inside the structured arm. A workgroup barrier is not: different waves
// may take different arms. Scalar writes into VCC_LO/HI also have an implicit mask-domain side
// effect which the scalar scratch model cannot reconstruct; accept those only when the mask is
// provably overwritten before any post-merge implicit read. The captured UE4 kernel writes VCC_LO
// as an integer scratch in the arm and overwrites VCC with a VOPC at the merge, which this
// deliberately narrow proof accepts.
bool compute_execz_region_admissible(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc,
                                     uint32_t region_end, ForwardIfRefusal* why) {
    for (const auto& r : ins) {
        if (r.is_end || r.pc >= region_end) break;
        if (r.pc <= branch_pc) continue;
        if (r.fmt == Rdna2Format::SOPP && r.opcode == 0x0a) {
            forward_if_refuse(why, "execz pc=%u contains barrier pc=%u", branch_pc, r.pc);
            return false;
        }

        uint32_t scalar_width = 0;
        if (r.fmt == Rdna2Format::SOP1)
            scalar_width = r.opcode == 0x04 ? 2u : 1u;
        else if (r.fmt == Rdna2Format::SOP2)
            scalar_width = 1;
        else if (r.fmt == Rdna2Format::SOPK && sopk_writes_scalar_data(r.opcode))
            scalar_width = 1;
        else if (r.fmt == Rdna2Format::SMEM) {
            switch (r.opcode) {
                case 0x0:
                case 0x8: scalar_width = 1; break;
                case 0x1:
                case 0x9: scalar_width = 2; break;
                case 0x2:
                case 0xA: scalar_width = 4; break;
                case 0x3:
                case 0xB: scalar_width = 8; break;
                case 0x4:
                case 0xC: scalar_width = 16; break;
                default: break;
            }
        }
        if (!scalar_width ||
            (r.dst.kind != OperandKind::SGPR && r.dst.kind != OperandKind::Special))
            continue;
        for (int vcc_half = 106; vcc_half <= 107; ++vcc_half) {
            if (vcc_half < r.dst.value || vcc_half >= r.dst.value + (int)scalar_width) continue;
            // PROSPER_VCC_SCALAR_DATA_MERGE=1 widens the proof from "not read" to "not read
            // as a lane mask". The guard this loop implements is about the MASK domain --
            // its own comment says so -- but the test it applies is total liveness, so a
            // half that survives the merge only to be read as a 32-bit scalar dword is
            // rejected even though the scalar model reproduces that dword exactly.
            //
            // Opt-in rather than default because the same physical pair being a mask on one
            // predecessor and data on another has no runtime type tag, and this file already
            // rejects that join deliberately. Keeping it behind a switch also makes the A/B
            // have a lever that can be shown to have moved, which a byte-identical module
            // has already cost this investigation once.
            const bool relax_vcc_scalar_data = compiler_choice(CompilerChoice::VccScalarMerge) != 0;
            const ScalarMergeProof proof = relax_vcc_scalar_data ? ScalarMergeProof::MaskDomainOnly
                                                                 : ScalarMergeProof::AnyRead;
            ScalarMergeBlocker blocker;
            if (!sgpr_dead_at_merge(ins, region_end, vcc_half, proof, &blocker)) {
                forward_if_refuse(
                    why,
                    "execz pc=%u scalar write pc=%u leaves vcc-half s%d live at merge "
                    "pc=%u "
                    "blocked-by pc=%u kind=%s",
                    branch_pc, r.pc, vcc_half, region_end, blocker.pc, blocker.kind);
                return false;
            }
        }
    }
    return true;
}

// A compute vccz/vccnz region whose decision is per-WAVE, so different waves in one workgroup may
// take different arms. Ordinary LDS loads/stores/atomics are valid under that control flow: they
// are per-invocation memory effects and impose no reconvergence requirement. A guest barrier is
// not valid there, and synthesized wave collectives need the CFG dispatcher's common phase rather
// than the compact structured emitter. Keep those synchronized operations fail-visible here; raw
// LDS memory effects can remain in-arm.
bool wave_vcc_region_admissible(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc,
                                uint32_t region_end, ForwardIfRefusal* why) {
    for (const auto& r : ins) {
        if (r.is_end || r.pc >= region_end) break;
        if (r.pc <= branch_pc) continue;
        const bool ds_wave_collective =
            r.fmt == Rdna2Format::DS && (r.opcode == 0x35 || r.opcode == 0x3d || r.opcode == 0x3e ||
                                         r.opcode == kDsOpcodeBpermuteB32);
        if ((r.fmt == Rdna2Format::SOPP && r.opcode == 0x0a) || ds_wave_collective ||
            (r.fmt == Rdna2Format::VOP3 && (r.opcode == 0x365 || r.opcode == 0x366))) {
            forward_if_refuse(why, "vcc branch pc=%u contains synchronized op pc=%u fmt=%d op=0x%x",
                              branch_pc, r.pc, static_cast<int>(r.fmt), r.opcode);
            return false;
        }
    }
    return true;
}
ForwardIfWorkgroupUniform forward_if_workgroup_uniform(const std::vector<Rdna2Inst>& ins,
                                                       const Rdna2Inst& in,
                                                       bool compute_uniform_vcc,
                                                       bool compute_wave_branches) {
    if (compute_uniform_vcc && vcc_branch_is_workgroup_uniform(ins, in.pc))
        return ForwardIfWorkgroupUniform::Vcc;
    if (compute_wave_branches && (in.opcode == 0x04 || in.opcode == 0x05) &&
        scc_branch_is_workgroup_uniform(ins, in.pc))
        return ForwardIfWorkgroupUniform::Scc;
    return ForwardIfWorkgroupUniform::No;
}

}   // namespace prosper::gpu
