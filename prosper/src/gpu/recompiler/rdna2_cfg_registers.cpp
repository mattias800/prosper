#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
int shader_max_vgpr(const std::vector<Rdna2Inst>& ins) {
    int highest = 0;
    for (const auto& in : ins) {
        if (in.is_end) break;
        for (uint32_t source = 0; source < in.n_src; ++source) {
            const uint32_t source_span = rdna2_vgpr_source_span(in, source);
            if (source_span)
                highest =
                    std::max(highest, in.src[source].value + static_cast<int>(source_span) - 1);
        }
        const uint32_t destination_span = rdna2_vgpr_destination_span(in);
        if (destination_span)
            highest = std::max(highest, in.dst.value + static_cast<int>(destination_span) - 1);
    }
    return highest;
}

// Registers WRITTEN in the pc range [lo, hi): candidates for an OpPhi at the loop header. Over-
// approximation is safe (an extra phi for a non-carried value merges equal values). Mirrors emit_alu's
// write targets, INCLUDING multi-register writes (MIMG dmask -> N consecutive VGPRs, SMEM -> N SGPRs) so
// no genuinely-carried register is missed (a missing phi = an undominated use = invalid SPIR-V).
void loop_written_regs(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi,
                       std::set<int>& vregs, std::set<int>& sregs) {
    for (const auto& in : ins) {
        if (in.pc < lo || in.pc >= hi) continue;
        switch (in.fmt) {
            case Rdna2Format::VOP1:
                if (in.opcode == 0x02)
                    sregs.insert(in.dst.value);   // v_readfirstlane -> SGPR
                else if (in.opcode == kVop1OpcodeMovreldB32)   // v_movreld: any observable
                    for (int reg = in.dst.value; reg <= shader_max_vgpr(ins); ++reg)
                        vregs.insert(reg);   // VDST+M0 target
                else
                    vregs.insert(in.dst.value);
                break;
            case Rdna2Format::VOP2:
            case Rdna2Format::VOP3P: vregs.insert(in.dst.value); break;
            case Rdna2Format::VOP3:
                if (in.opcode == 0x360)
                    sregs.insert(in.dst.value);   // v_readlane -> SGPR
                else {
                    for (uint32_t k = 0; k < rdna2_vgpr_write_count(in); ++k)
                        vregs.insert(in.dst.value + (int)k);
                }
                break;   // (writelane: slots, not SSA)
            case Rdna2Format::DS:
                for (uint32_t k = 0; k < rdna2_vgpr_write_count(in); ++k)
                    vregs.insert(in.dst.value + (int)k);
                break;
            case Rdna2Format::MUBUF:
            case Rdna2Format::MTBUF:
            case Rdna2Format::MIMG:
            case Rdna2Format::FLAT:
                for (uint32_t k = 0; k < rdna2_vgpr_write_count(in); ++k)
                    vregs.insert(in.dst.value + (int)k);
                if (const int tfe_status = rdna2_tfe_status_vgpr(in); tfe_status >= 0)
                    vregs.insert(tfe_status);
                break;
            case Rdna2Format::SOP1: sregs.insert(in.dst.value); break;
            case Rdna2Format::SOPK:
                if (sopk_writes_scalar_data(in.opcode)) sregs.insert(in.dst.value);
                break;
            case Rdna2Format::SOP2:
                // s_lshr_b64 -> EXEC is modeled only in the per-lane mask domain. It does not
                // produce scalar SGPR data, so carrying a scalar value through a loop/if merge is
                // both unnecessary and semantically wrong.
                if (in.opcode != 0x21 || (in.dst.value != 126 && in.dst.value != 127)) {
                    uint32_t words = 1;
                    if (in.opcode == 0x0b)
                        words = scalar_write_width(in);
                    else if (in.opcode == 0x1f || in.opcode == 0x21)
                        words = 2;
                    for (uint32_t word = 0; word < words; ++word)
                        sregs.insert(in.dst.value + static_cast<int>(word));
                }
                break;
            case Rdna2Format::SMEM: {   // s_load/s_buffer_load: N consecutive SGPRs
                uint32_t n = 1;
                switch (in.opcode) {
                    case 0x1:
                    case 0x9: n = 2; break;
                    case 0x2:
                    case 0xA: n = 4; break;
                    case 0x3:
                    case 0xB: n = 8; break;
                    case 0x4:
                    case 0xC: n = 16; break;
                }
                for (uint32_t k = 0; k < n; k++) sregs.insert(in.dst.value + (int)k);
                break;
            }
            default: break;   // VOPC/SOPC write VCC/SCC — handled by their own phis
        }
    }
}
}   // namespace prosper::gpu
