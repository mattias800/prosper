// Original scalar CFG case groups; native path exits are retained.
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include <cstdint>
#include "gpu/recompiler/rdna2_cfg_scalar_merge_routes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <optional>

namespace prosper::gpu::detail {
bool CfgScalarMergeRoutes::implicit_reads() {
    // VCC (R = 106/107) has IMPLICIT readers the operand scan can't see: the e32 VOP2 carry /
    // cndmask ops (0x01, 0x28-0x2A) and the vccz/vccnz branches. Flag those as reads up front so
    // the redef checks below are sound for VCC too (which the s_mov carve-out excludes, but the
    // SMEM carve-out needs — compilers love `s_buffer_load_dword vcc_lo, …` scratch loads).
    if (R == 106 || R == 107) {
        if (in.fmt == Rdna2Format::VOP2 &&
            (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2A)))
            return block(in, "vop2-implicit-vcc");
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07))
            return block(in, "vccz-branch");
        // NOTE: v_div_fmas_f32/f64 (VOP3 0x16f/0x170) also read VCC implicitly and are NOT
        // listed above. Neither is emitted by the recompiler today (an unsupported opcode
        // fails the whole recompile), so no liveness question reaches them; add the reader
        // here before one is lowered.
    }
    return true;
}

std::optional<bool> CfgScalarMergeRoutes::step() {
    if (!implicit_reads()) return false;
    switch (in.fmt) {
        case Rdna2Format::SOPK: return sopk();
        case Rdna2Format::SOPP: return sopp();
        case Rdna2Format::SMEM: return smem();
        case Rdna2Format::SOP1:
        case Rdna2Format::SOP2:
        case Rdna2Format::SOPC:
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP2:
        case Rdna2Format::VOP3:
        case Rdna2Format::VOPC:
        case Rdna2Format::DS:
        case Rdna2Format::EXP: return alu();
        case Rdna2Format::MIMG:
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF: return descriptors();
        case Rdna2Format::FLAT:
        case Rdna2Format::VINTRP: return flat_vintrp();
        default: return false;   // memory/branch/unknown: can't bound reads of R -> assume live
    }
}

std::optional<bool> CfgScalarMergeRoutes::sopk() {
    do {
        switch (in.fmt) {
            // Most SOPK instructions remain fail-closed: s_addk/s_mulk/s_cmovk/s_cmpk read or
            // read-modify-write their encoded "dst" while decode exposes no source. MOVK_I32 is a
            // pure one-word definition and WAITCNT_VSCNT is register-transparent; GTA's low-only
            // VCC lifetime crosses both before the actual pair replacement.
            case Rdna2Format::SOPK:
                if (in.opcode == kSopkOpcodeWaitcntVscnt && in.dst.kind == OperandKind::SGPR &&
                    in.dst.value == 125)
                    break;
                if (in.opcode == kSopkOpcodeMovkI32 && in.dst.kind == OperandKind::SGPR) {
                    if (in.dst.value == R) continue;
                    break;
                }
                // Under MaskDomainOnly the whole read-modify-write SOPK family is admissible. Every
                // SOPK operates on ONE dword: s_addk_i32/s_mulk_i32 read and rewrite their encoded
                // destination, s_cmpk_* read it and write SCC, s_cmovk_i32 reads it and may rewrite
                // it. All three are 32-bit scalar-DATA touches of the half; none can consume it as a
                // 64-bit lane mask, which is the only thing this proof needs to exclude.
                //
                // Deliberately does NOT claim the half died, even for the forms that always write
                // it: continuing the walk is the conservative direction, and a later genuine mask
                // read still fails the proof. Being wrong about the kill would be unsound; being
                // silent about it only costs acceptances.
                if (data_read_ok && in.dst.kind == OperandKind::SGPR) break;
                return block(in, "unmodelled-sopk");
            default: break;
        }
        return std::nullopt;
    } while (false);
    return true;
}

std::optional<bool> CfgScalarMergeRoutes::sopp() {
    do {
        switch (in.fmt) {
            case Rdna2Format::SOPP:
                // Hint/sync SOPPs read and write nothing — scan through them (s_waitcnt is ubiquitous
                // after the merge). A barrier is also scalar-register-transparent; it changes only
                // workgroup execution order. For branches, enqueue every possible successor. Unknown
                // scalar control flow remains fail-closed.
                if (sopp_is_noop(in) || in.opcode == 0x0a) break;
                if (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09)) {
                    const int64_t branch_pc =
                        static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
                    const auto branch = branch_pc >= 0
                                            ? pc_to_index.find(static_cast<uint32_t>(branch_pc))
                                            : pc_to_index.end();
                    if (branch == pc_to_index.end()) return false;
                    pending.push_back(branch->second);
                    if (in.opcode != 0x02) {
                        if (index + 1 >= ins.size()) return false;
                        pending.push_back(index + 1);
                    }
                    continue;
                }
                return block(in, "unmodelled-sopp");
            default: break;
        }
        return std::nullopt;
    } while (false);
    return true;
}

}   // namespace prosper::gpu::detail
