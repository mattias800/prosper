// An EXEC phi may inherit a preheader subset only from a closed, monotonically narrowing loop.
// Keep this code-only proof separate from the carried-value obligations checked by the emitter.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_scalar_condition_marks.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>
#include <set>
#include <unordered_map>

namespace prosper::gpu {
namespace {
bool branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02u && in.opcode <= 0x09u &&
           in.opcode != 0x03u;
}

bool unaccounted_control(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::Unknown) return true;
    if (in.fmt == Rdna2Format::SOP1)
        return in.opcode == kSop1OpcodeSetpcB64 || in.opcode == kSop1OpcodeSwappcB64 ||
               in.opcode == kSop1OpcodeRfeB64;
    if (in.fmt == Rdna2Format::SOPK)
        return in.opcode == kSopkOpcodeCallB64 || in.opcode >= kSopkOpcodeSubvectorLoopBegin;
    // Outside the loop a barrier is an ordinary preceding/succeeding phase boundary. Any other
    // unmodeled scalar flow cannot be assumed unable to enter this header through a hidden edge.
    return in.fmt == Rdna2Format::SOPP && !branch(in) && !sopp_is_noop(in) && !in.is_end &&
           in.opcode != 0x0au;
}

bool plain_cmpx(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode) && in.len_dwords == 1 &&
           !in.has_modifier && !in.has_sdwa && !in.has_dpp;
}

bool writes_exec_word(const Rdna2Inst& in) {
    bool overlaps = false;
    const auto word_span = [&](int base, uint32_t count) {
        overlaps |= base <= 127 && base + static_cast<int>(count) > 126;
    };
    for_each_scalar_write(in, word_span);
    // The architectural B64 span includes the high word even for an odd base; the scalar phi
    // inventory has narrower storage conventions for some operations. It cannot certify EXEC.
    if (in.fmt == Rdna2Format::SOP2) word_span(in.dst.value, rdna2_sop2_dest_dwords(in.opcode));
    return overlaps;
}

bool preserves_exec(const Rdna2Inst& in) {
    if (rdna2_instruction_may_change_exec(in) || writes_exec_word(in)) return false;
    switch (in.fmt) {
        case Rdna2Format::SOP1:
            // Ordinary direct scalar writers only. SETPC/SWAPPC/RFE and relative destinations
            // can escape the region or alias a word not represented by their decoded base.
            return in.opcode == kSop1OpcodeMovB32 || in.opcode == kSop1OpcodeMovB64 ||
                   in.opcode == kSop1OpcodeBcnt1I32B64 || in.opcode == kSop1OpcodeFf1I32B64;
        case Rdna2Format::SOP2: return in.opcode <= 0x35u;
        case Rdna2Format::SOPK: return in.opcode <= kSopkOpcodeMulkI32;
        case Rdna2Format::SOPC: return in.opcode <= 0x11u;
        case Rdna2Format::SOPP: return sopp_is_noop(in);
        case Rdna2Format::SMEM:
            return in.opcode <= kSmemOpcodeLoadDwordX16 || smem_opcode_is_buffer_load(in.opcode);
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP2:
        case Rdna2Format::VOP3:
        case Rdna2Format::VOP3P:
        case Rdna2Format::VOPC:
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF:
        case Rdna2Format::MIMG:
        case Rdna2Format::FLAT: return true;
        default: return false;
    }
}
// SCC has both explicit numeric readers and implicit carry/condition readers. A replacement
// closes only its actual CFG path, and an ADDC/SUBB must read the incoming value before killing it.
bool scc_dead_at_exit(const std::vector<Rdna2Inst>& ins, uint32_t exit) {
    std::unordered_map<uint32_t, std::size_t> indices;
    for (std::size_t i = 0; i < ins.size(); ++i) indices.emplace(ins[i].pc, i);
    const auto start = indices.find(exit);
    if (start == indices.end()) return false;
    std::vector<std::size_t> pending{start->second};
    std::vector<bool> visited(ins.size(), false);
    while (!pending.empty()) {
        const std::size_t index = pending.back();
        pending.pop_back();
        if (visited[index]) continue;
        visited[index] = true;
        const auto& in = ins[index];
        if (in.is_end) continue;
        if (scalar_reads_scc(in) ||
            (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x04u || in.opcode == 0x05u)))
            return false;
        if (scalar_instruction_writes_scc(in)) continue;
        if (unaccounted_control(in)) return false;
        if (branch(in)) {
            const auto target = indices.find(branch_target(in));
            if (target == indices.end()) return false;
            pending.push_back(target->second);
            if (in.opcode == 0x02u) continue;
        }
        if (index + 1u >= ins.size()) return false;
        pending.push_back(index + 1u);
    }
    return true;
}

bool new_loop_admission_is_lane_local(const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                                      const RegState& rs) {
    std::set<int> written, mask_words;
    for (const auto& [base, value] : rs.sreg_bool) {
        (void)value;
        mask_words.insert(base);
        mask_words.insert(base + 1);
    }
    for (const auto& [word, value] : rs.sreg_wave64_mask_half) {
        (void)value;
        mask_words.insert(word);
    }
    // A fresh compare/SDST inside the body changes the source's domain after the snapshot.
    // Inventory every possible pair before checking reads, including overlap and backedge uses.
    for (const auto& in : ins) {
        if (in.pc < loop.header_pc || in.pc > loop.backedge_pc) continue;
        const auto record_mask = [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word)
                mask_words.insert(base + static_cast<int>(word));
        };
        if ((in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) || vop3_writes_mask_sdst(in))
            for_each_scalar_write(in, record_mask);
    }
    bool writes_scc = false;
    for (const auto& in : ins) {
        // READFIRSTLANE/READLANE outside the loop can also supply a per-invocation scalar entry.
        // This proof does not authenticate a broadcast from such an existing approximation.
        if ((in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02u) ||
            (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360u))
            return false;
        if (in.pc < loop.header_pc || in.pc > loop.backedge_pc) continue;
        // Per-invocation loop exits cannot service a cross-lane operation or a wave-mask
        // reduction after other invocations have left. Keep unaudited effects out of this NEW
        // admission; the existing emitter's permissions and warnings are unchanged.
        if (in.has_dpp ||
            (in.fmt == Rdna2Format::VOP1 &&
             (in.opcode == kVop1OpcodeMovreldB32 || in.opcode == 0x43u)) ||
            (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x360u) ||
            (in.fmt == Rdna2Format::SOP1 &&
             (in.opcode == kSop1OpcodeBcnt1I32B64 || in.opcode == kSop1OpcodeFf1I32B64)))
            return false;
        switch (in.fmt) {
            case Rdna2Format::SMEM:
            case Rdna2Format::MTBUF:
            case Rdna2Format::MIMG:
            case Rdna2Format::FLAT: return false;
            case Rdna2Format::MUBUF:
                // The original descriptor-loop consumer is an ordinary read-only DWORD load.
                // Stores, atomics and other forms need their own side-effect ordering proof.
                if (in.opcode < kMubufOpcodeLoadDword || in.opcode > kMubufOpcodeLoadDwordX3 ||
                    in.mubuf_lds || in.mubuf_tfe)
                    return false;
                break;
            default: break;
        }
        for (uint32_t source = 0; source < in.n_src; ++source) {
            const auto& operand = in.src[source];
            if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) continue;
            const uint32_t width = in.fmt == Rdna2Format::MUBUF
                                       ? (source == 1u ? 4u : 1u)
                                       : scalar_alu_source_words(in, source);
            if (width == UINT32_MAX) continue;
            for (uint32_t word = 0; word < width; ++word) {
                const int reg = operand.value + static_cast<int>(word);
                if (reg == 106 || reg == 107 || reg == 126 || reg == 127 ||
                    mask_words.contains(reg))
                    return false;
                const auto value = rs.sreg.find(reg);
                if (value != rs.sreg.end() && rs.lane_local_scalars.contains(value->second))
                    return false;
            }
        }
        const auto record = [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word)
                written.insert(base + static_cast<int>(word));
        };
        for_each_scalar_write(in, record);
        if (in.fmt == Rdna2Format::SOP2) record(in.dst.value, rdna2_sop2_dest_dwords(in.opcode));
        if ((in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
             in.dst.kind == OperandKind::Special) ||
            (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28u && in.opcode <= 0x2au))
            record(106, 2);
        writes_scc |= scalar_instruction_writes_scc(in);
    }
    for (int reg : written)
        if (!sgpr_dead_at_merge(ins, loop.exit_pc, reg, ScalarMergeProof::AnyRead)) return false;
    return !writes_scc || scc_dead_at_exit(ins, loop.exit_pc);
}
}   // namespace

bool structured_loop_only_narrows_exec(const std::vector<Rdna2Inst>& ins, const DivLoop& loop) {
    if (loop.condition != DivLoop::Condition::Exec || !loop.continue_on_set || loop.bottom_tested ||
        loop.direct_exec_breaks || loop.direct_wave_breaks || !loop.break_pcs.empty() ||
        loop.header_pc >= loop.exit_branch_pc || loop.exit_branch_pc >= loop.backedge_pc ||
        loop.exit_pc != loop.backedge_pc + 1u)
        return false;
    uint32_t next = loop.header_pc;
    bool reduction = false;
    for (const auto& in : ins) {
        if (unaccounted_control(in)) return false;
        if (in.pc < loop.header_pc || in.pc > loop.backedge_pc) {
            // A second incoming header edge may bypass the actual emitted preheader. Interior
            // entries are likewise outside this induction, even if a different detector accepts it.
            if (branch(in)) {
                const uint32_t target = branch_target(in);
                if (target >= loop.header_pc && target < loop.exit_pc) return false;
            }
            continue;
        }
        if (in.pc != next || in.len_dwords == 0 || in.is_end || in.synthetic_terminator)
            return false;
        next += in.len_dwords;
        if (in.pc == loop.exit_branch_pc) {
            if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x08u || in.len_dwords != 1 ||
                branch_target(in) != loop.exit_pc)
                return false;
        } else if (in.pc == loop.backedge_pc) {
            if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x02u || in.len_dwords != 1 ||
                branch_target(in) != loop.header_pc)
                return false;
        } else if (plain_cmpx(in)) {
            reduction = true;
        } else if (!preserves_exec(in)) {
            return false;
        }
    }
    return next == loop.exit_pc && reduction;
}

bool narrowing_loop_exec_admission(const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                                   const RegState& entry) {
    return structured_loop_only_narrows_exec(ins, loop) &&
           new_loop_admission_is_lane_local(ins, loop, entry);
}

uint32_t make_narrowing_loop_exec_phi(SpirvCompute& b, RegState& rs, uint32_t preheader,
                                      std::size_t& patch, bool admitted) {
    const uint32_t entry_exec = rs.exec;
    const uint32_t phi = b.emit_phi2(b.t_bool, entry_exec, preheader, patch);
    // CONFIDENCE: HIGH -- induction over the real preheader and every audited backedge: CMPX
    // computes EXEC & comparison, so neither incoming phi value can enable a lane outside entry.
    // The current compute loop leaves per invocation. Do not expose a surviving wave-scalar
    // result, or admit an in-loop cross-lane/side-effect shape, solely through this new DATA proof.
    if (b.is_compute && admitted) rs.exec_subset_of[phi] = entry_exec;
    rs.exec = phi;
    return phi;
}
}   // namespace prosper::gpu
