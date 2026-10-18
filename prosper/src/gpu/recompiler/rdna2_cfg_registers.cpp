#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
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
        // CONFIDENCE: HIGH -- an image instruction names its further addresses only through the
        // encoding (RDNA2 ISA 70648 MIMG): consecutive after VADDR, or one byte each in the NSA
        // dwords. rdna2_vgpr_source_span charges VADDR alone, so an address register nothing
        // else names was absent from dispatcher storage and from the relay MAY seeding (#5182).
        // Every modelled read shape counts each address; every NSA byte names a register even
        // when the shape is not modelled (zero padding names v0).
        if (in.fmt == Rdna2Format::MIMG) {
            const uint32_t addresses = rdna2_mimg_read_address_count(in);
            for (uint32_t index = 0; index < addresses; ++index)
                highest = std::max(highest, rdna2_mimg_address_vgpr(in, index));
            for (uint32_t byte = 0; byte < 4u * in.mimg_nsa; ++byte)
                highest = std::max(highest, rdna2_mimg_address_vgpr(in, byte + 1u));
        }
    }
    return highest;
}

// Existing phi/dispatcher effect inventory extracted from the capped CFG translation unit.
// Over-approximation allocates/reloads storage only; packet validity still guards actual reads.
void loop_written_regs(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi,
                       std::set<int>& vregs, std::set<int>& sregs, bool owned_packet) {
    for (const auto& in : ins) {
        if (in.pc < lo || in.pc >= hi) continue;
        switch (in.fmt) {
            case Rdna2Format::VINTRP:
                // CONFIDENCE: HIGH — native P1 is a no-op, but every other native VINTRP
                // lowering publishes VDST. Owned P1 also writes partial interpolation state.
                // Omitting a real writer lets an arm-local SSA value escape without a phi.
                if (owned_packet || !rdna2_vintrp_is_p1(in)) vregs.insert(in.dst.value);
                break;
            case Rdna2Format::VOP1:
                if (in.opcode == 0x02)
                    sregs.insert(in.dst.value);        // v_readfirstlane -> SGPR
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
            case Rdna2Format::SOP2:   // the instruction-effects table's widths (#5436)
                for (uint32_t word = 0; word < ir::cfg::sop2_phi_words(in); ++word)
                    sregs.insert(in.dst.value + static_cast<int>(word));
                break;
            case Rdna2Format::SMEM: {   // the SGPRs SDATA may receive (the effects table)
                // A reserved encoding keeps the old conservative one-register clobber.
                const uint32_t n =
                    rdna2_instruction_effects(in).known() ? rdna2_smem_destination_dwords(in) : 1u;
                for (uint32_t k = 0; k < n; k++) sregs.insert(in.dst.value + (int)k);
                break;
            }
            default: break;   // VOPC/SOPC write VCC/SCC, handled by their own phis
        }
    }
}

void if_join_regs(const std::vector<Rdna2Inst>& ins, const ForwardIf& F, uint32_t else_hi,
                  const ir::cfg::StatePlacement& placement, std::set<int>& vgprs,
                  std::set<int>& sgprs) {
    loop_written_regs(ins, F.branch_pc + 1, F.has_else ? F.sb_pc : F.target_pc, vgprs, sgprs);
    if (F.has_else) loop_written_regs(ins, F.target_pc, else_hi, vgprs, sgprs);
    const ir::cfg::IfPlacement* placed = placement.if_at(F.branch_pc);
    ir::cfg::take_placed_if_equal(placed ? &placed->merge : nullptr, ir::cfg::PhiSite::IfMerge,
                                  F.branch_pc, vgprs, sgprs);
}

void loop_join_regs(const std::vector<Rdna2Inst>& ins, uint32_t header_pc, uint32_t backedge_pc,
                    uint32_t exit_branch_pc, const ir::cfg::StatePlacement& placement,
                    std::set<int>& cv, std::set<int>& cs, std::set<int>& condv,
                    std::set<int>& conds) {
    loop_written_regs(ins, header_pc, backedge_pc, cv, cs);
    loop_written_regs(ins, header_pc, exit_branch_pc, condv, conds);
    const ir::cfg::LoopPlacement* placed = placement.loop_at(header_pc);
    ir::cfg::take_placed_if_equal(placed ? &placed->carried : nullptr,
                                  ir::cfg::PhiSite::LoopCarried, header_pc, cv, cs);
    ir::cfg::take_placed_if_equal(placed ? &placed->condition : nullptr,
                                  ir::cfg::PhiSite::LoopCondition, header_pc, condv, conds);
}

// Scalar registers that MAY be overwritten while a loop executes. This is deliberately separate
// from loop_written_regs: mask-pair destinations overwrite physical SGPRs (and therefore descriptor
// provenance) but their values live in sreg_bool rather than the scalar-data SSA domain.
void loop_scalar_may_writes(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi,
                            std::set<int>& sregs) {
    for (const auto& in : ins) {
        if (in.pc < lo || in.pc >= hi) continue;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word)
                sregs.insert(base + static_cast<int>(word));
        });
    }
}

void invalidate_loop_descriptor_provenance(RegState& rs, const std::set<int>& sregs) {
    for (int reg : sregs) {
        rs.sreg_written.insert(reg);
        rs.sreg_input.erase(reg);
        rs.sreg_srt.erase(reg);
        // A loop body that may write this register must not leave a copy alias standing: the alias
        // was established on one iteration's path and says nothing about the next one (#1773).
        rs.sreg_ud_alias.erase(reg);
    }
}
}   // namespace prosper::gpu
