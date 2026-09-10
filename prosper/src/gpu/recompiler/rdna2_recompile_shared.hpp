#pragma once

// Lifted out of rdna2_to_spirv.cpp's anonymous namespaces so the emit functions that
// operate on them can live in their own translation units. These are INTERNAL to the
// recompiler: nothing outside src/gpu/recompiler/ should include this header.

// rdna2_to_spirv.cpp — see rdna2_to_spirv.hpp. Internal SpirvCompute builder + the VALU translator.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"

namespace prosper::gpu {


// Extend the narrow compiler shape
//
//   s_cbranch_scc* ELSE
//   THEN...
//   s_endpgm
// ELSE:
//   ELSE...
//   s_endpgm
//
// into the existing structured-if/else input. The first s_endpgm is represented as a synthetic
// s_branch to the second end, which is semantically exact: it skips the other terminating arm, while
// the shared SPIR-V shell may still converge solely to publish outputs/return. Keep this deliberately
// conservative: one scalar conditional, adjacent straight-line else arm, and a real terminating end.
// Immediate-end/nop-end tails remain the established early-out shape and are not rewritten.
inline bool extend_terminating_if_else(const uint32_t* code, size_t dwords,
                                       std::vector<Rdna2Inst>& instructions,
                                       size_t* required_dwords = nullptr,
                                       uint32_t* synthetic_branch_pc = nullptr) {
    if (synthetic_branch_pc) *synthetic_branch_pc = UINT32_MAX;
    if (!code || instructions.empty()) return false;
    auto first_end = std::find_if(instructions.begin(), instructions.end(),
                                  [](const Rdna2Inst& in) { return in.is_end; });
    if (first_end == instructions.end()) return false;

    const Rdna2Inst* branch = nullptr;
    for (auto it = instructions.begin(); it != first_end; ++it) {
        if (it->fmt != Rdna2Format::SOPP) continue;
        if (it->opcode == 0x04 || it->opcode == 0x05) {
            if (branch) return false;
            branch = &*it;
        } else if (it->opcode >= 0x02 && it->opcode <= 0x09 && it->opcode != 0x03) {
            return false;
        }
    }
    if (!branch || branch->simm16 <= 0) return false;
    const int64_t target64 = static_cast<int64_t>(branch->pc) + branch->len_dwords + branch->simm16;
    if (target64 < 0 || static_cast<uint64_t>(target64) >= dwords) return false;
    const uint32_t target = static_cast<uint32_t>(target64);
    if (target != first_end->pc + first_end->len_dwords) return false;

    std::vector<Rdna2Inst> tail;
    const size_t tail_dwords = rdna2_walk(code + target, dwords - target, tail);
    if (tail.empty() || !tail.back().is_end) return false;
    bool has_real_else = false;
    for (auto it = tail.begin(); it != tail.end() - 1; ++it) {
        if (it->fmt == Rdna2Format::SOPP) {
            if (it->opcode == 0x00) continue; // padding is harmless but not a real else arm
            if (it->opcode >= 0x02 && it->opcode <= 0x09 && it->opcode != 0x03) return false;
        }
        has_real_else = true;
    }
    if (!has_real_else) return false;

    for (auto& in : tail) in.pc += target;
    const uint32_t merge_pc = tail.back().pc;
    const uint32_t skip_dwords = merge_pc - (first_end->pc + first_end->len_dwords);
    if (!skip_dwords || skip_dwords > static_cast<uint32_t>(INT16_MAX)) return false;
    first_end->fmt = Rdna2Format::SOPP;
    first_end->opcode = 0x02; // s_branch merge_pc: terminates the lexical then arm
    first_end->simm16 = static_cast<int32_t>(skip_dwords);
    first_end->words[0] = 0xbf820000u | (skip_dwords & 0xffffu);
    first_end->is_end = false;
    if (synthetic_branch_pc) *synthetic_branch_pc = first_end->pc;
    instructions.insert(instructions.end(), tail.begin(), tail.end());
    if (required_dwords) *required_dwords = target + tail_dwords;
    return true;
}

inline std::unordered_set<uint32_t> safe_execz_branches(const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> safe;
    // pc of the terminating s_endpgm — a forward execz whose target is here skips straight to the end.
    uint32_t end_pc = 0; bool have_end = false;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; have_end = true; break; }
    for (const auto& br : ins) {
        if (br.fmt != Rdna2Format::SOPP || br.opcode != 0x08 || br.simm16 <= 0) continue;
        const uint32_t target = br.pc + br.len_dwords + (uint32_t)br.simm16;
        bool ok = target > br.pc;
        bool target_found = false;
        for (const auto& in : ins) {
            if (in.pc == target) { target_found = true; break; }
        }
        if (!target_found) ok = false;
        // LOOP EXIT (#1183): if a backward branch that jumps to at-or-before this execz sits inside the
        // block [br, target), then this execz is a data-dependent loop's EXIT, not an if/guard-to-end —
        // even when its target happens to be s_endpgm. Linearizing it here would strand the loop's
        // back-edge as a straight-line reject; instead leave it OUT of `safe` so detect_divergent_loops
        // claims it and emit_divloop reconstructs the structured loop. Back-edges are the shapes that
        // detector recognizes: unconditional s_branch (0x02) or s_cbranch_execnz (0x09).
        bool is_loop_exit = false;
        for (const auto& bb : ins) {
            if (bb.fmt != Rdna2Format::SOPP || (bb.opcode != 0x02 && bb.opcode != 0x09)) continue;
            if (bb.pc <= br.pc || bb.pc >= target) continue;                 // must sit inside this block
            const int64_t bt = static_cast<int64_t>(bb.pc) + bb.len_dwords + bb.simm16;   // signed target
            if (bt <= static_cast<int64_t>(br.pc)) { is_loop_exit = true; break; }
        }
        if (is_loop_exit) continue;
        // Two shapes are safe to linearize (drop the branch, run the block under per-lane EXEC):
        //  * IF/ENDIF rejoining live code (target < end): only EXEC-predicated VGPR writes are safe,
        //    since scalar/VCC/memory writes past the merge would be observed by later code.
        //  * GUARD-TO-END (target == s_endpgm): the block's scalar/SGPR/VCC writes are DEAD (nothing
        //    runs after s_endpgm) and wave-uniform, memory loads are fault-free (robustBufferAccess) into
        //    predicated VGPRs, and memory stores are EXEC-predicated (conditional store). So anything is
        //    safe EXCEPT an EXP export (not EXEC-predicated → would export from inactive lanes).
        const bool guard_to_end = have_end && target >= end_pc;
        for (const auto& in : ins) {
            if (in.pc <= br.pc || in.pc >= target || in.is_end) continue;
            if (in.fmt == Rdna2Format::EXP) { ok = false; break; }   // exports are never EXEC-masked
            if (guard_to_end) continue;                              // dead-at-end / predicated / fault-free
            // IF/ENDIF rejoining live code: safe to linearize iff every write is EXEC-predicated. VGPR ALU
            // (VOP1/2/3) predicates its writes; memory ops (MIMG/MUBUF/DS) predicate their VGPR loads (and
            // are fault-free under robust access) and EXEC-predicate their stores. SGPR/VCC writes (SOP*/
            // VOPC/SMEM) are NOT predicated and would be observed past the merge -> still unsafe.
            // CARVE-OUTS: three op groups inside the "safe" VALU formats have UNPREDICATED scalar side
            // effects (emit_alu writes rs.sreg/rs.vcc/rs.sreg_bool for them, and predicate_write only
            // covers VGPRs) — on hardware the skipped block would have preserved VCC/the SGPR:
            //   VOP1 0x02 v_readfirstlane_b32 (writes an SGPR), VOP2 0x28-0x2A carry ops (write VCC),
            //   VOP3B 0x128-0x12A and 0x30F/0x310/0x319 (write the carry-out SGPR pair/VCC), and
            //   VOP3B v_mad_u64_u32 0x176 (its 65th-bit carry mask also lands in VCC/an SGPR pair
            //   unpredicated).
            const bool scalar_side_effect =
                (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02) ||
                (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2A) ||
                vop3b_fresh_carry_output(in) ||
                (in.fmt == Rdna2Format::VOP3 &&
                 ((in.opcode >= 0x128 && in.opcode <= 0x12A) || in.opcode == 0x176));
            if (!scalar_side_effect &&
                (in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 || in.fmt == Rdna2Format::VOP3 ||
                 in.fmt == Rdna2Format::MIMG || in.fmt == Rdna2Format::MUBUF ||
                 in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::DS ||
                 in.fmt == Rdna2Format::FLAT ||
                 sopp_is_noop(in))) {
                continue;
            }
            if (scalar_side_effect) { ok = false; break; }
            // A pure scalar move (s_mov_b32/b64: no SCC/VCC/memory side effect) whose destination SGPR is
            // DEAD at the merge is also safe to linearize: the unconditional write is overwritten before any
            // later read, so masked-off lanes never observe it. (The tonemap/sRGB divergent-ifs load a scalar
            // constant used only within the block and reset it right after — see shader 033.)
            if (in.fmt == Rdna2Format::SOP1 && (in.opcode == 0x03 || in.opcode == 0x04) &&
                in.dst.kind == OperandKind::SGPR) {
                const bool b64 = (in.opcode == 0x04);
                // dst must be a plain SGPR (s0..s105). SOP1 destinations also decode EXEC/VCC/M0
                // (106/107/124/126/127) as SGPR-kind, but a move into those has wave-wide side effects read
                // implicitly (not via an SGPR operand) — they can never be proven dead, so exclude them.
                const int hi = in.dst.value + (b64 ? 1 : 0);
                if (hi <= 105 && sgpr_dead_at_merge(ins, target, in.dst.value) &&
                    (!b64 || sgpr_dead_at_merge(ins, target, in.dst.value + 1))) continue;
            }
            // A scalar LOAD (s_load/s_buffer_load) inside the block is likewise safe iff every SGPR it
            // writes is DEAD at the merge: the load itself is wave-uniform and fault-free, and a result
            // overwritten before any read is never observed by post-merge code. This is the divergent
            // lighting/fog block's `s_buffer_load_dwordx2 vcc, …` scratch-load shape (DOLL VS, #273);
            // sgpr_dead_at_merge understands VCC's implicit readers, so vcc-targeted loads qualify.
            if (in.fmt == Rdna2Format::SMEM) {
                uint32_t n = 0;
                switch (in.opcode) {
                    case 0x0: case 0x8: n = 1;  break;   case 0x1: case 0x9: n = 2;  break;
                    case 0x2: case 0xA: n = 4;  break;   case 0x3: case 0xB: n = 8;  break;
                    case 0x4: case 0xC: n = 16; break;   default: break;
                }
                if (n) {
                    bool all_dead = true;
                    for (uint32_t k = 0; k < n && all_dead; k++)
                        all_dead = sgpr_dead_at_merge(ins, target, in.dst.value + (int)k);
                    if (all_dead) continue;
                }
            }
            ok = false;
            break;
        }
        if (ok) safe.insert(br.pc);
    }
    return safe;
}
inline const Rdna2Inst* last_scalar_writer(const std::vector<Rdna2Inst>& ins, uint32_t before_pc,
                                    int reg) {
    const Rdna2Inst* result = nullptr;
    for (const auto& in : ins) {
        if (in.is_end || in.pc >= before_pc) break;
        bool writes_reg = false;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            writes_reg |= reg >= base && reg < base + static_cast<int>(width);
        });
        if (writes_reg) result = &in;
    }
    return result;
}

inline bool immediate_operand(const Operand& operand, const Rdna2Inst& in, uint32_t& value) {
    if (operand.kind == OperandKind::Literal) { value = in.literal; return true; }
    if (operand.kind == OperandKind::InlineInt) {
        value = static_cast<uint32_t>(operand.value);
        return true;
    }
    return false;
}

inline bool binary_reg_immediate(const Rdna2Inst& in, int reg, uint32_t& immediate) {
    return (reg_operand(in.src[0], reg) && immediate_operand(in.src[1], in, immediate)) ||
           (reg_operand(in.src[1], reg) && immediate_operand(in.src[0], in, immediate));
}

inline bool ordered_reg_immediate(const Rdna2Inst& in, int reg, uint32_t& immediate) {
    return reg_operand(in.src[0], reg) && immediate_operand(in.src[1], in, immediate);
}

inline bool binary_regs(const Rdna2Inst& in, int a, int b) {
    return (reg_operand(in.src[0], a) && reg_operand(in.src[1], b)) ||
           (reg_operand(in.src[0], b) && reg_operand(in.src[1], a));
}

inline PcrelDispatchInfo detect_pcrel_dispatch(const std::vector<Rdna2Inst>& ins,
                                        const uint32_t* code, size_t dwords,
                                        size_t program_dwords) {
    PcrelDispatchInfo out;
    if (!code || ins.empty()) return out;

    std::unordered_set<uint32_t> instruction_pcs;
    std::vector<uint32_t> branch_targets;
    for (const auto& in : ins) {
        instruction_pcs.insert(in.pc);
        if (in.is_end) continue;
        if (in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
            in.opcode != 0x03)
            branch_targets.push_back(scalar_branch_target(in));
    }

    for (const auto& setpc : ins) {
        if (setpc.is_end || setpc.fmt != Rdna2Format::SOP1 || setpc.opcode != 0x20) continue;
        const int jump_lo = setpc.src[0].value;
        const Rdna2Inst* jump_add = last_scalar_writer(ins, setpc.pc, jump_lo);
        const Rdna2Inst* jump_addc = last_scalar_writer(ins, setpc.pc, jump_lo + 1);
        if (!jump_add || !jump_addc || jump_add->fmt != Rdna2Format::SOP2 ||
            jump_add->opcode != 0x00 || jump_addc->fmt != Rdna2Format::SOP2 ||
            jump_addc->opcode != 0x04 || jump_add->pc >= jump_addc->pc) continue;

        int table_lo = -1;
        for (const Operand& source : jump_add->src) {
            if ((source.kind == OperandKind::SGPR || source.kind == OperandKind::Special) &&
                source.value != jump_lo) table_lo = source.value;
        }
        if (table_lo < 0 || !binary_regs(*jump_add, jump_lo, table_lo) ||
            !binary_regs(*jump_addc, jump_lo + 1, table_lo + 1)) continue;
        const Rdna2Inst* target_getpc = last_scalar_writer(ins, jump_add->pc, jump_lo);
        if (!target_getpc || target_getpc->fmt != Rdna2Format::SOP1 ||
            target_getpc->opcode != 0x1f) continue;

        const Rdna2Inst* table_load = last_scalar_writer(ins, jump_add->pc, table_lo);
        if (!table_load || table_load != last_scalar_writer(ins, jump_add->pc, table_lo + 1) ||
            table_load->fmt != Rdna2Format::SMEM || table_load->opcode != 0x01 ||
            table_load->literal != 0 || table_load->src[0].kind != OperandKind::SGPR) continue;
        const int table_base_lo = table_load->src[0].value;
        const int selector = table_load->src[1].value;

        const Rdna2Inst* table_add = last_scalar_writer(ins, table_load->pc, table_base_lo);
        const Rdna2Inst* table_addc = last_scalar_writer(ins, table_load->pc, table_base_lo + 1);
        if (!table_add || !table_addc || table_add->fmt != Rdna2Format::SOP2 ||
            table_add->opcode != 0x00 || table_addc->fmt != Rdna2Format::SOP2 ||
            table_addc->opcode != 0x04 || table_add->pc >= table_addc->pc) continue;
        uint32_t table_delta = 0, high_zero = 1;
        if (!binary_reg_immediate(*table_add, table_base_lo, table_delta) ||
            !binary_reg_immediate(*table_addc, table_base_lo + 1, high_zero) || high_zero != 0)
            continue;
        const Rdna2Inst* table_getpc = last_scalar_writer(ins, table_add->pc, table_base_lo);
        if (!table_getpc || table_getpc->fmt != Rdna2Format::SOP1 ||
            table_getpc->opcode != 0x1f) continue;

        const Rdna2Inst* shift = last_scalar_writer(ins, table_load->pc, selector);
        if (!shift || shift->fmt != Rdna2Format::SOP2 || shift->opcode != 0x1e) continue;
        uint32_t shift_amount = 0;
        // s_lshl_b32 is ordered: selector << 3 scales the qword index, while 3 << selector
        // is a different program and must not be admitted by the commutative matcher.
        if (!ordered_reg_immediate(*shift, selector, shift_amount) || shift_amount != 3) continue;
        const Rdna2Inst* clamp = last_scalar_writer(ins, shift->pc, selector);
        if (!clamp || clamp->fmt != Rdna2Format::SOP2 || clamp->opcode != 0x07) continue;
        uint32_t selector_max = 0;
        if (!binary_reg_immediate(*clamp, selector, selector_max) || selector_max > 63) continue;

        const Rdna2Inst* adjust = last_scalar_writer(ins, clamp->pc, selector);
        int32_t selector_addend = 0;
        if (adjust && adjust->fmt == Rdna2Format::SOP2 &&
            (adjust->opcode == 0x00 || adjust->opcode == 0x02)) {
            uint32_t addend = 0;
            if (!binary_reg_immediate(*adjust, selector, addend)) continue;
            selector_addend = static_cast<int32_t>(addend);
        } else {
            adjust = nullptr;
        }
        const Rdna2Inst* selector_load = last_scalar_writer(
            ins, adjust ? adjust->pc : clamp->pc, selector);
        if (!selector_load || selector_load->fmt != Rdna2Format::SMEM ||
            selector_load->opcode != 0x08 || selector_load->src[0].kind != OperandKind::SGPR ||
            selector_load->src[1].kind != OperandKind::Special ||
            selector_load->src[1].value != 125) continue;

        const uint64_t table_byte =
            static_cast<uint64_t>(table_getpc->pc + table_getpc->len_dwords) * 4u + table_delta;
        const size_t entry_count = static_cast<size_t>(selector_max) + 1;
        if ((table_byte & 7u) || table_byte / 4 < program_dwords ||
            table_byte / 4 + entry_count * 2 > dwords) continue;

        std::vector<uint32_t> targets;
        targets.reserve(entry_count);
        const int64_t target_pc_byte =
            static_cast<int64_t>(target_getpc->pc + target_getpc->len_dwords) * 4;
        bool table_ok = true;
        for (size_t index = 0; index < entry_count; ++index) {
            const size_t word = static_cast<size_t>(table_byte / 4) + index * 2;
            const int64_t relative = static_cast<int64_t>(
                (static_cast<uint64_t>(code[word + 1]) << 32) | code[word]);
            const int64_t target_byte = target_pc_byte + relative;
            if (target_byte < 0 || (target_byte & 3) ||
                target_byte / 4 > static_cast<int64_t>(UINT32_MAX) ||
                !instruction_pcs.contains(static_cast<uint32_t>(target_byte / 4))) {
                table_ok = false;
                break;
            }
            targets.push_back(static_cast<uint32_t>(target_byte / 4));
        }
        if (!table_ok || targets.empty()) continue;
        const uint32_t merge_pc = *std::max_element(targets.begin(), targets.end());
        if (merge_pc <= setpc.pc) continue;
        for (uint32_t target : targets) if (target < setpc.pc + setpc.len_dwords || target > merge_pc)
            table_ok = false;
        if (!table_ok) continue;

        const std::vector<uint32_t> setup = {
            selector_load->pc,
            adjust ? adjust->pc : UINT32_MAX,
            clamp->pc, shift->pc, table_getpc->pc, table_add->pc, table_addc->pc,
            table_load->pc, target_getpc->pc, jump_add->pc, jump_addc->pc, setpc.pc,
        };
        const uint32_t setup_first = selector_load->pc;
        for (uint32_t target : branch_targets) {
            if (target > setup_first && target <= setpc.pc) { table_ok = false; break; }
        }
        if (!table_ok) continue;

        out.valid = true;
        out.selector_sgpr_base = static_cast<uint32_t>(selector_load->src[0].value);
        out.selector_byte_offset = selector_load->literal;
        out.selector_addend = selector_addend;
        out.selector_max = selector_max;
        out.setpc_pc = setpc.pc;
        out.merge_pc = merge_pc;
        out.required_dwords = static_cast<size_t>(table_byte / 4) + entry_count * 2;
        out.target_pcs = std::move(targets);
        for (uint32_t pc : setup) if (pc != UINT32_MAX) out.setup_pcs.push_back(pc);
        return out;
    }
    return out;
}

inline bool specialize_pcrel_dispatch(std::vector<Rdna2Inst>& ins, const PcrelDispatchInfo& info,
                               uint32_t selected_target) {
    if (!info.valid || std::find(info.target_pcs.begin(), info.target_pcs.end(), selected_target) ==
                           info.target_pcs.end()) return false;
    std::unordered_set<uint32_t> remove(info.setup_pcs.begin(), info.setup_pcs.end());

    // A compiler may jump over an alternate entry prologue before it starts the dispatch setup. Fold
    // only forward unconditional branches wholly contained in that prelude; any external entry into a
    // skipped range makes the specialization unprovable.
    for (const auto& branch : ins) {
        if (branch.pc >= info.setpc_pc || branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x02)
            continue;
        const uint32_t target = scalar_branch_target(branch);
        if (target <= branch.pc || target > info.setpc_pc) return false;
        for (const auto& other : ins) {
            if (other.fmt != Rdna2Format::SOPP || other.pc == branch.pc || other.opcode < 0x02 ||
                other.opcode > 0x09 || other.opcode == 0x03) continue;
            const uint32_t entered = scalar_branch_target(other);
            if (entered > branch.pc + branch.len_dwords && entered < target) return false;
        }
        remove.insert(branch.pc);
        for (const auto& skipped : ins)
            if (skipped.pc >= branch.pc + branch.len_dwords && skipped.pc < target)
                remove.insert(skipped.pc);
    }

    uint32_t selected_end = info.merge_pc;
    for (const auto& in : ins) {
        if (in.pc < selected_target || in.pc >= info.merge_pc) continue;
        if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x02) continue;
        // Internal forward branches and loop back-edges remain in the selected routine and are
        // validated/structured by emit_body. Only the compiler's route terminator jumps to the
        // common merge and can be removed as a now-redundant branch.
        if (scalar_branch_target(in) != info.merge_pc) continue;
        selected_end = in.pc;
        remove.insert(in.pc);
        break;
    }

    std::vector<Rdna2Inst> specialized;
    specialized.reserve(ins.size());
    for (const auto& in : ins) {
        const bool prelude = in.pc < info.setpc_pc;
        const bool selected = in.pc >= selected_target && in.pc < selected_end;
        const bool merge = in.pc >= info.merge_pc;
        if ((prelude || selected || merge) && !remove.contains(in.pc)) specialized.push_back(in);
    }
    if (specialized.empty()) return false;

    // No surviving branch may enter an omitted alternative. The ordinary structurizer performs the
    // remaining detailed CFG checks after this coarse specialization boundary check.
    std::unordered_set<uint32_t> retained;
    uint32_t end_pc = 0;
    for (const auto& in : specialized) retained.insert(in.pc);
    for (const auto& in : specialized) if (in.is_end) { end_pc = in.pc; break; }
    for (const auto& in : specialized) {
        if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03 || in.simm16 < 0) continue;
        const uint32_t target = scalar_branch_target(in);
        // Existing forward-if validation accepts a compiler early-out just beyond the primary
        // S_ENDPGM only after proving that raw target terminates immediately. Preserve that case for
        // the detailed validator; targets into an omitted alternative still fail here.
        if (!retained.contains(target) && target <= end_pc) return false;
    }
    ins = std::move(specialized);
    return true;
}

struct ShaderConstantValue {
    bool known = false;
    uint32_t value = 0;
};

inline ShaderConstantValue shader_constant_operand(
        const std::vector<Rdna2Inst>& ins, size_t block_first, size_t use_index,
        const Rdna2Inst& use, const Operand& operand, uint32_t depth) {
    if (depth > 16) return {};
    if (operand.kind == OperandKind::InlineInt)
        return {true, static_cast<uint32_t>(operand.value)};
    if (operand.kind == OperandKind::Literal)
        return use.has_literal ? ShaderConstantValue{true, use.literal} : ShaderConstantValue{};
    if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special)
        return {};

    const int reg = operand.value;
    for (size_t i = use_index; i-- > block_first;) {
        const Rdna2Inst& writer = ins[i];
        // VCC_LO/HI can temporarily hold ordinary scalar data, as in Astro's exact branch setup,
        // but any intervening vector ALU may replace the architectural VCC pair implicitly. The
        // shared scalar-writer inventory intentionally models that in the Bool domain, so reject it
        // explicitly here instead of walking past a VOPC to an obsolete scalar definition.
        if ((reg == 106 || reg == 107) &&
            (writer.fmt == Rdna2Format::VOP1 || writer.fmt == Rdna2Format::VOP2 ||
             writer.fmt == Rdna2Format::VOPC || writer.fmt == Rdna2Format::VOP3 ||
             writer.fmt == Rdna2Format::VOP3P))
            return {};
        bool writes = false;
        uint32_t width = 0;
        int base = -1;
        for_each_scalar_write(writer, [&](int candidate_base, uint32_t candidate_width) {
            if (reg >= candidate_base &&
                reg < candidate_base + static_cast<int>(candidate_width)) {
                writes = true;
                base = candidate_base;
                width = candidate_width;
            }
        });
        if (!writes) continue;

        // Only one-dword pure scalar data writers participate. A pair write, memory result, lane
        // read, or wave-mask producer is intentionally not a shader-constant proof.
        if (base != reg || width != 1) return {};
        if (writer.fmt == Rdna2Format::SOP1 && writer.opcode == 0x03) { // s_mov_b32
            return shader_constant_operand(
                ins, block_first, i, writer, writer.src[0], depth + 1);
        }
        if (writer.fmt != Rdna2Format::SOP2) return {};

        const ShaderConstantValue lhs = shader_constant_operand(
            ins, block_first, i, writer, writer.src[0], depth + 1);
        const ShaderConstantValue rhs = shader_constant_operand(
            ins, block_first, i, writer, writer.src[1], depth + 1);
        if (!lhs.known || !rhs.known) return {};

        switch (writer.opcode) {
            case 0x00: return {true, lhs.value + rhs.value}; // s_add_u32
            case 0x01: return {true, lhs.value - rhs.value}; // s_sub_u32
            case 0x02: return {true, lhs.value + rhs.value}; // s_add_i32
            case 0x03: return {true, lhs.value - rhs.value}; // s_sub_i32
            case 0x0e: return {true, lhs.value & rhs.value}; // s_and_b32
            case 0x10: return {true, lhs.value | rhs.value}; // s_or_b32
            case 0x12: return {true, lhs.value ^ rhs.value}; // s_xor_b32
            // Shifts take the amount from S1[4:0] -- RDNA2 ISA -- so the `& 31` is required.
            //
            // It is deliberately UNTESTED, because on this host it is untestable: x86 `shl` already
            // masks its count to 5 bits, so an unmasked `lhs.value << rhs.value` yields the SAME
            // answer here (measured: naive `5u << 32` prints 5). The mask is therefore not fixing a
            // wrong result on x86; it is removing undefined behaviour that a compiler is entitled to
            // exploit, and that would diverge on a host whose shift does not wrap. Do not add a
            // regression arm claiming to prove it -- such an arm passes with the mask removed, which
            // makes it a control that cannot fail. The arms below cover only what is observable:
            // that these opcodes fold at all.
            //
            // Opcodes verified against llvm-mc (gfx1030), not against this file's own tables: the
            // decoder that produces the listing you would otherwise check them with is upstream of
            // them, so it cannot check them. #2481 records a mnemonic error that survived three
            // internally consistent anchors and inverted a frontier conclusion.
            case 0x1e: return {true, lhs.value << (rhs.value & 31u)};  // s_lshl_b32
            case 0x20: return {true, lhs.value >> (rhs.value & 31u)};  // s_lshr_b32
            case 0x22: return {true, static_cast<uint32_t>(            // s_ashr_i32
                std::bit_cast<int32_t>(lhs.value) >> (rhs.value & 31u))};
            case 0x26: return {true, lhs.value * rhs.value}; // s_mul_i32
            default: return {};
        }
    }
    // No in-block writer means an entry/user SGPR or other runtime state. Never specialize it.
    return {};
}

inline bool shader_constant_compare(const std::vector<Rdna2Inst>& ins, size_t block_first,
                             size_t compare_index, bool& result) {
    const Rdna2Inst& compare = ins[compare_index];
    if (compare.fmt != Rdna2Format::SOPC || compare.opcode > 0x0b) return false;
    const ShaderConstantValue lhs = shader_constant_operand(
        ins, block_first, compare_index, compare, compare.src[0], 0);
    const ShaderConstantValue rhs = shader_constant_operand(
        ins, block_first, compare_index, compare, compare.src[1], 0);
    if (!lhs.known || !rhs.known) return false;
    const int32_t signed_lhs = std::bit_cast<int32_t>(lhs.value);
    const int32_t signed_rhs = std::bit_cast<int32_t>(rhs.value);

    switch (compare.opcode) {
        case 0x00: result = signed_lhs == signed_rhs; break;
        case 0x01: result = signed_lhs != signed_rhs; break;
        case 0x02: result = signed_lhs >  signed_rhs; break;
        case 0x03: result = signed_lhs >= signed_rhs; break;
        case 0x04: result = signed_lhs <  signed_rhs; break;
        case 0x05: result = signed_lhs <= signed_rhs; break;
        case 0x06: result = lhs.value == rhs.value; break;
        case 0x07: result = lhs.value != rhs.value; break;
        case 0x08: result = lhs.value >  rhs.value; break;
        case 0x09: result = lhs.value >= rhs.value; break;
        case 0x0a: result = lhs.value <  rhs.value; break;
        case 0x0b: result = lhs.value <= rhs.value; break;
        default: return false;
    }
    return true;
}

inline bool scalar_cfg_branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP &&
        sopp_opcode_is_direct_branch(in.opcode);
}

inline void prune_scalar_cfg_reachability(std::vector<Rdna2Inst>& ins) {
    if (ins.empty()) return;
    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);

    std::vector<bool> reachable(ins.size(), false);
    std::vector<size_t> pending{0};
    while (!pending.empty()) {
        const size_t i = pending.back();
        pending.pop_back();
        if (i >= ins.size() || reachable[i]) continue;
        reachable[i] = true;
        const Rdna2Inst& in = ins[i];
        if (in.is_end || (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12)) continue;
        if (scalar_cfg_branch(in)) {
            const auto target = index_by_pc.find(scalar_branch_target(in));
            if (target != index_by_pc.end()) pending.push_back(target->second);
            if (in.opcode == 0x02) continue;
        }
        if (i + 1 < ins.size()) pending.push_back(i + 1);
    }

    std::vector<Rdna2Inst> retained;
    retained.reserve(ins.size());
    for (size_t i = 0; i < ins.size(); ++i)
        if (reachable[i]) retained.push_back(ins[i]);
    // Once a taken branch's omitted arm is gone, its target is often the next retained instruction.
    // Turn that now-redundant edge into a no-op so the ordinary straight-line/structured paths do not
    // need to reconstruct an empty branch region.
    for (size_t i = 0; i + 1 < retained.size(); ++i) {
        Rdna2Inst& in = retained[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x02 &&
            scalar_branch_target(in) == retained[i + 1].pc) {
            in.opcode = 0x00;
            in.simm16 = 0;
            in.words[0] = 0xbf800000u;
        }
    }
    ins = std::move(retained);
}

inline size_t specialize_shader_constant_branches(std::vector<Rdna2Inst>& ins) {
    if (ins.empty()) return 0;
    // An indirect PC transfer can enter code outside the explicit SOPP graph. Keep the whole shader
    // unspecialized rather than treating its lexical successor as the only possible destination.
    if (std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return in.fmt == Rdna2Format::SOP1 &&
                   in.opcode >= 0x20 && in.opcode <= 0x22;
        })) return 0;

    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);

    std::set<size_t> block_starts{0};
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (scalar_cfg_branch(in)) {
            const auto target = index_by_pc.find(scalar_branch_target(in));
            if (target != index_by_pc.end()) block_starts.insert(target->second);
        }
        if ((scalar_cfg_branch(in) || in.is_end ||
             (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12)) &&
            i + 1 < ins.size())
            block_starts.insert(i + 1);
    }

    size_t specialized_count = 0;
    for (size_t i = 1; i < ins.size(); ++i) {
        Rdna2Inst& branch = ins[i];
        if (branch.fmt != Rdna2Format::SOPP ||
            (branch.opcode != 0x04 && branch.opcode != 0x05) || branch.simm16 <= 0)
            continue; // shader-constant SCC only; VCCZ/EXECZ remain runtime wave conditions
        if (!index_by_pc.contains(scalar_branch_target(branch)))
            continue; // do not specialize an exit beyond the decoded instruction graph
        const Rdna2Inst& compare = ins[i - 1];
        if (compare.pc + compare.len_dwords != branch.pc ||
            compare.fmt != Rdna2Format::SOPC)
            continue;
        const auto block = block_starts.upper_bound(i - 1);
        const size_t block_first = block == block_starts.begin() ? 0 : *std::prev(block);
        if (block_first > i - 1) continue;

        bool scc = false;
        if (!shader_constant_compare(ins, block_first, i - 1, scc)) continue;
        const bool taken = branch.opcode == 0x05 ? scc : !scc;
        if (taken) {
            branch.opcode = 0x02; // s_branch: retain the exact immediate target
            branch.words[0] = 0xbf820000u | static_cast<uint16_t>(branch.simm16);
        } else {
            branch.opcode = 0x00; // s_nop 0: retain fallthrough
            branch.simm16 = 0;
            branch.words[0] = 0xbf800000u;
        }
        ++specialized_count;
    }
    if (!specialized_count) return 0;

    // Entry-rooted reachability after replacing proven conditions. Unknown conditional branches
    // retain both successors, so an unresolved resource remains in the instruction stream whenever
    // any runtime path can execute it.
    prune_scalar_cfg_reachability(ins);
    return specialized_count;
}

inline size_t specialize_proven_null_bvh_exits(std::vector<Rdna2Inst>& ins,
                                        const ShaderResourceTable* rt,
                                        uint32_t wave_size) {
    if (!rt || (wave_size != 32 && wave_size != 64) || ins.empty()) return 0;
    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);

    size_t specialized = 0;
    for (const ShaderResource& resource : rt->resources) {
        if (!is_proven_null_bvh(resource)) continue;
        const auto found = index_by_pc.find(resource.fetch_pc);
        if (found == index_by_pc.end()) continue;
        const size_t ray_index = found->second;
        if (ray_index + 5 >= ins.size()) continue;
        const Rdna2Inst& ray = ins[ray_index];
        const Rdna2Inst& wait = ins[ray_index + 1];
        const Rdna2Inst& compare = ins[ray_index + 2];
        const Rdna2Inst& scalar_compare = ins[ray_index + 3];
        const Rdna2Inst& exec_copy = ins[ray_index + 4];
        Rdna2Inst& exit = ins[ray_index + 5];

        // Exact Wave32 no-hit idiom:
        //   image_bvh_intersect_ray vN..vN+3, NULL_BVH
        //   s_waitcnt ...
        //   v_cmp_ne_u32 sM, -1, vN
        //   s_cmp_lg_u32 0, sM
        //   s_mov_b32 exec_lo, vcc_lo
        //   s_cbranch_scc0 EXIT
        // The Wave64 sibling uses the complete two-word mask instead:
        //   v_cmp_ne_u32 s[M:M+1], -1, vN
        //   s_cmp_lg_u64 s[M:M+1], 0
        //   s_mov_b64 exec, vcc
        // The null lowering writes -1 to every ray result for each active lane. The saved compare
        // mask is therefore exactly zero, making SCC zero and the exit unconditional. Requiring the
        // exact width-specific compare and EXEC copy avoids assuming anything about an unobserved
        // mask half. Adjacency and exact register links prevent a nearby unrelated compare from
        // proving the branch.
        const bool ray_shape = ray.fmt == Rdna2Format::MIMG && ray.opcode == 0xe6u &&
            ray.mimg_dmask == 0xfu && ray.dst.kind == OperandKind::VGPR;
        const bool wait_shape = wait.pc == ray.pc + ray.len_dwords &&
            wait.fmt == Rdna2Format::SOPP && wait.opcode == 0x0cu &&
            wait.words[0] == 0xbf8c3f70u; // s_waitcnt vmcnt(0)
        const bool compare_shape = compare.pc == wait.pc + wait.len_dwords &&
            compare.fmt == Rdna2Format::VOPC && compare.opcode == 0xc5u &&
            compare.dst.kind == OperandKind::SGPR &&
            compare.dst.value <= (wave_size == 32 ? 105 : 104) &&
            compare.src[0].kind == OperandKind::InlineInt && compare.src[0].value == -1 &&
            compare.src[1].kind == OperandKind::VGPR && compare.src[1].value == ray.dst.value;
        const bool scalar_shape = scalar_compare.pc == compare.pc + compare.len_dwords &&
            scalar_compare.fmt == Rdna2Format::SOPC &&
            (wave_size == 32
                ? scalar_compare.opcode == 0x07u &&
                  scalar_compare.src[0].kind == OperandKind::InlineInt &&
                  scalar_compare.src[0].value == 0 &&
                  scalar_compare.src[1].kind == OperandKind::SGPR &&
                  scalar_compare.src[1].value == compare.dst.value
                : scalar_compare.opcode == 0x13u &&
                  scalar_compare.src[0].kind == OperandKind::SGPR &&
                  scalar_compare.src[0].value == compare.dst.value &&
                  scalar_compare.src[1].kind == OperandKind::InlineInt &&
                  scalar_compare.src[1].value == 0);
        const bool copy_shape = exec_copy.pc == scalar_compare.pc + scalar_compare.len_dwords &&
            exec_copy.fmt == Rdna2Format::SOP1 &&
            exec_copy.opcode == (wave_size == 32 ? 0x03u : 0x04u) &&
            exec_copy.dst.kind == OperandKind::SGPR && exec_copy.dst.value == 126 &&
            exec_copy.src[0].kind == OperandKind::Special && exec_copy.src[0].value == 106;
        const bool exit_shape = exit.pc == exec_copy.pc + exec_copy.len_dwords &&
            exit.fmt == Rdna2Format::SOPP && exit.opcode == 0x04u && exit.simm16 > 0 &&
            index_by_pc.contains(scalar_branch_target(exit));
        if (!ray_shape || !wait_shape || !compare_shape || !scalar_shape || !copy_shape ||
            !exit_shape)
            continue;

        // Some compiler-generated traversal loops enter with an empty scalar stack, visit one
        // root ray, then pop work written by that ray.  When the dispatch-scoped root is proven
        // null, the exact no-hit branch above reaches the empty-stack test without writing the
        // depth.  That makes the pop/back-edge arm unreachable, which in turn proves that the
        // loop-selected ray sites cannot seed themselves.  Keep this deliberately narrower than
        // ordinary scalar constant propagation: every entry, register relationship, and branch on
        // the first path must match the observed stack idiom.
        auto specialize_empty_stack = [&]() {
            const uint32_t null_exit_pc = scalar_branch_target(exit);
            const auto tail_found = index_by_pc.find(null_exit_pc);
            if (tail_found == index_by_pc.end() || tail_found->second + 1 >= ins.size())
                return false;
            const size_t tail_index = tail_found->second;
            const Rdna2Inst& stack_compare = ins[tail_index];
            Rdna2Inst& stack_exit = ins[tail_index + 1];
            if (stack_compare.fmt != Rdna2Format::SOPC || stack_compare.opcode != 0x07u ||
                stack_compare.src[0].kind != OperandKind::InlineInt ||
                stack_compare.src[0].value != 0 ||
                stack_compare.src[1].kind != OperandKind::SGPR ||
                stack_compare.src[1].value < 0 || stack_compare.src[1].value > 105 ||
                stack_exit.pc != stack_compare.pc + stack_compare.len_dwords ||
                stack_exit.fmt != Rdna2Format::SOPP || stack_exit.opcode != 0x04u ||
                stack_exit.simm16 <= 0 ||
                !index_by_pc.contains(scalar_branch_target(stack_exit)))
                return false;
            // The compiler allocates this scalar stack depth opportunistically (the observed
            // traversal kernels use both s41 and s45). Derive the physical word from the exact
            // empty-stack comparison, then require the initializer and every write check below to
            // agree with it. This broadens only register allocation, not the proven control/data
            // relationship.
            const int stack_reg = stack_compare.src[1].value;
            const uint32_t stack_exit_pc = scalar_branch_target(stack_exit);

            // The first-iteration selector branch targets the block that initializes all four
            // ray results to invalid before the guarded root query.  Requiring this complete
            // eight-instruction prefix prevents a nearby null ray from being mistaken for the
            // traversal root that owns the scalar stack below.
            if (ray_index < 8) return false;
            const size_t root_block = ray_index - 8;
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const Rdna2Inst& init = ins[root_block + lane];
                if (init.fmt != Rdna2Format::VOP1 || init.opcode != 0x01u ||
                    init.dst.kind != OperandKind::VGPR ||
                    init.dst.value != ray.dst.value + static_cast<int>(lane) ||
                    init.src[0].kind != OperandKind::InlineInt || init.src[0].value != -1)
                    return false;
            }
            const Rdna2Inst& mask_copy = ins[root_block + 4];
            const Rdna2Inst& empty_guard = ins[root_block + 5];
            const Rdna2Inst& root_index = ins[root_block + 6];
            const Rdna2Inst& root_nop = ins[root_block + 7];
            const bool mask_copy_shape = wave_size == 32
                ? mask_copy.fmt == Rdna2Format::SOP1 &&
                  mask_copy.opcode == kSop1OpcodeAndSaveexecB32 &&
                  mask_copy.dst.kind == OperandKind::SGPR && mask_copy.dst.value == 106 &&
                  mask_copy.src[0].kind == OperandKind::Special &&
                  mask_copy.src[0].value == 107
                : mask_copy.fmt == Rdna2Format::SOP1 && mask_copy.opcode == 0x24u &&
                  mask_copy.dst.kind == OperandKind::SGPR && mask_copy.dst.value == 106 &&
                  mask_copy.src[0].kind == OperandKind::SGPR;
            if (!mask_copy_shape ||
                empty_guard.fmt != Rdna2Format::SOPP || empty_guard.opcode != 0x08u ||
                scalar_branch_target(empty_guard) != scalar_compare.pc ||
                root_index.fmt != Rdna2Format::VOP1 || root_index.opcode != 0x01u ||
                root_index.dst.kind != OperandKind::VGPR ||
                root_index.dst.value != ray.dst.value ||
                root_index.src[0].kind != OperandKind::SGPR ||
                root_nop.fmt != Rdna2Format::SOPP || root_nop.opcode != 0x00u)
                return false;

            size_t selector_branch_index = SIZE_MAX;
            size_t stack_init_index = SIZE_MAX;
            bool selector_scc = false;
            for (size_t i = root_block; i-- > 2;) {
                const Rdna2Inst& branch = ins[i];
                if (branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x05u ||
                    scalar_branch_target(branch) != ins[root_block].pc ||
                    ins[i - 1].pc + ins[i - 1].len_dwords != branch.pc ||
                    ins[i - 1].fmt != Rdna2Format::SOPC)
                    continue;
                for (size_t j = i - 1; j-- > 1;) {
                    const Rdna2Inst& selector_init = ins[j];
                    if (selector_init.fmt != Rdna2Format::SOP1 ||
                        selector_init.opcode != 0x03u ||
                        selector_init.dst.kind != OperandKind::SGPR ||
                        selector_init.src[0].kind != OperandKind::InlineInt ||
                        selector_init.src[0].value != 37 ||
                        root_index.src[0].value != selector_init.dst.value)
                        continue;
                    size_t candidate_stack_init = SIZE_MAX;
                    if (wave_size == 32) {
                        const Rdna2Inst& stack_init = ins[j - 1];
                        if (stack_init.pc + stack_init.len_dwords == selector_init.pc &&
                            stack_init.fmt == Rdna2Format::SOP1 &&
                            stack_init.opcode == 0x03u &&
                            stack_init.dst.kind == OperandKind::SGPR &&
                            stack_init.dst.value == stack_reg &&
                            stack_init.src[0].kind == OperandKind::InlineInt &&
                            stack_init.src[0].value == 0)
                            candidate_stack_init = j - 1;
                    } else {
                        // The Wave64 sibling schedules independent scalar address setup between
                        // `stack=0` and `selector=37`. Accept only the nearest in-block write to the
                        // derived stack register, and require it to be the exact zero initializer.
                        for (size_t k = j; k-- > 0;) {
                            bool writes_candidate = false;
                            for_each_scalar_write(ins[k], [&](int base, uint32_t width) {
                                writes_candidate |= base <= stack_reg &&
                                    stack_reg < base + static_cast<int>(width);
                            });
                            if (!writes_candidate) {
                                if (scalar_cfg_branch(ins[k]) || ins[k].is_end) break;
                                continue;
                            }
                            const Rdna2Inst& stack_init = ins[k];
                            if (stack_init.fmt == Rdna2Format::SOP1 &&
                                stack_init.opcode == 0x03u &&
                                stack_init.dst.kind == OperandKind::SGPR &&
                                stack_init.dst.value == stack_reg &&
                                stack_init.src[0].kind == OperandKind::InlineInt &&
                                stack_init.src[0].value == 0)
                                candidate_stack_init = k;
                            break;
                        }
                    }
                    if (candidate_stack_init == SIZE_MAX) continue;
                    bool has_branch = false;
                    for (size_t k = candidate_stack_init + 1; k < i; ++k)
                        has_branch |= scalar_cfg_branch(ins[k]);
                    if (has_branch ||
                        !shader_constant_compare(ins, j, i - 1, selector_scc) ||
                        !selector_scc)
                        continue;
                    selector_branch_index = i;
                    stack_init_index = candidate_stack_init;
                    break;
                }
                if (selector_branch_index != SIZE_MAX) break;
            }
            if (selector_branch_index == SIZE_MAX) return false;

            auto writes_stack = [&](const Rdna2Inst& instruction) {
                bool writes = false;
                for_each_scalar_write(instruction, [&](int base, uint32_t width) {
                    writes |= base <= stack_reg &&
                        stack_reg < base + static_cast<int>(width);
                }, /*wave32_one_word_masks=*/wave_size == 32);
                return writes;
            };
            // Only the selector setup, root/no-hit block, and empty-stack comparison are reachable
            // before the proven exits.  None may alter the initialized stack depth.
            for (size_t i = stack_init_index + 1; i <= selector_branch_index; ++i)
                if (writes_stack(ins[i])) return false;
            for (size_t i = root_block; i <= ray_index + 5; ++i)
                if (writes_stack(ins[i])) return false;
            if (writes_stack(stack_compare)) return false;

            const uint32_t proof_begin = ins[stack_init_index].pc;
            // An indirect transfer or an external edge into the middle of the proof could bypass
            // the zero initializer.  Re-entry at the initializer itself is harmless: it resets the
            // invariant before the selector is evaluated again.
            for (const Rdna2Inst& instruction : ins) {
                if (instruction.fmt == Rdna2Format::SOP1 &&
                    instruction.opcode >= 0x20u && instruction.opcode <= 0x22u)
                    return false;
                if (!scalar_cfg_branch(instruction)) continue;
                const uint32_t target = scalar_branch_target(instruction);
                const bool source_inside = instruction.pc >= proof_begin &&
                    instruction.pc < stack_exit_pc;
                if (!source_inside && target > proof_begin && target < stack_exit_pc)
                    return false;
            }

            Rdna2Inst& selector_branch = ins[selector_branch_index];
            selector_branch.opcode = 0x02u;
            selector_branch.words[0] = 0xbf820000u |
                static_cast<uint16_t>(selector_branch.simm16);
            stack_exit.opcode = 0x02u;
            stack_exit.words[0] = 0xbf820000u |
                static_cast<uint16_t>(stack_exit.simm16);
            return true;
        };

        (void)specialize_empty_stack();
        exit.opcode = 0x02;
        exit.words[0] = 0xbf820000u | static_cast<uint16_t>(exit.simm16);
        ++specialized;
    }
    if (specialized) prune_scalar_cfg_reachability(ins);
    return specialized;
}

inline bool valid_scalar_pair_base(int base) {
    return base >= 0 && !(base & 1) &&
        (base <= 104 || base == 106 ||
         (base >= 108 && base <= 122) || base == 126);
}

inline bool scalar_pair_operand(const Operand& operand, int& base) {
    if ((operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) ||
        !valid_scalar_pair_base(operand.value))
        return false;
    base = operand.value;
    return true;
}

inline bool valid_buffer_resource_base(const Operand& operand) {
    if (operand.kind != OperandKind::SGPR || operand.value < 0 ||
        (operand.value & 3))
        return false;
    return operand.value <= 100 ||
        (operand.value >= 108 && operand.value <= 120);
}

inline bool zero_record_load_shape(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::MUBUF &&
        in.opcode == kMubufOpcodeLoadDword && in.len_dwords == 2u &&
        in.dst.kind == OperandKind::VGPR && in.dst.value >= 0 &&
        in.src[0].kind == OperandKind::VGPR &&
        valid_buffer_resource_base(in.src[1]) &&
        in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0 &&
        !in.mubuf_glc && !in.mubuf_dlc && !in.mubuf_lds && !in.mubuf_tfe;
}

inline bool scalar_instruction_writes_anything(const Rdna2Inst& in) {
    bool writes = false;
    for_each_scalar_write(in, [&](int, uint32_t) { writes = true; });
    return writes;
}

inline bool dynamic_vgpr_destination(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32;
}

inline bool writes_vgpr(const Rdna2Inst& in, int vgpr) {
    if (dynamic_vgpr_destination(in)) return true;
    const uint32_t words = rdna2_vgpr_write_count(in);
    if (words && in.dst.kind == OperandKind::VGPR && in.dst.value <= vgpr &&
        vgpr < in.dst.value + static_cast<int>(words))
        return true;
    return rdna2_tfe_status_vgpr(in) == vgpr;
}

// Prove that every lane active at `and_index` was also active when `load_index` wrote zero. The
// scalar compiler idioms in the live shader elect one lane and later restore a saved mask; tracking
// only subset lineage is enough and deliberately cannot prove an unrelated mask or an expanding
// EXEC write. Conditional skips inside the interval must contain no proof-relevant writes so the
// lexical transfer below represents both paths.
inline bool zero_load_reaches_and_under_exec_subset(const std::vector<Rdna2Inst>& ins,
                                             size_t load_index, size_t and_index,
                                             const std::unordered_map<uint32_t, size_t>& index_by_pc) {
    const int zero_vgpr = ins[load_index].dst.value;

    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& branch = ins[i];
        if (!scalar_cfg_branch(branch)) continue;
        const auto target = index_by_pc.find(scalar_branch_target(branch));
        if (target == index_by_pc.end()) return false;

        const bool source_inside = i > load_index && i < and_index;
        // The compare and branch are part of the proof too: an external edge to either would
        // bypass the zero-reaching definition just as surely as one into the transfer interval.
        const bool target_enters_after_load = target->second > load_index &&
            target->second <= and_index + 2u;
        if (!source_inside && target_enters_after_load) return false;
        if (!source_inside) continue;
        if (branch.opcode == kSoppOpcodeBranch || target->second <= i ||
            target->second > and_index)
            return false;
        for (size_t skipped = i + 1; skipped < target->second; ++skipped) {
            const Rdna2Inst& candidate = ins[skipped];
            if (rdna2_instruction_may_change_exec(candidate) ||
                scalar_instruction_writes_anything(candidate) ||
                writes_vgpr(candidate, zero_vgpr))
                return false;
        }
    }

    std::array<bool, 128> mask_subset{};
    bool exec_subset = true; // the load's EXEC is the reference set

    auto pair_is_subset = [&](const Operand& operand) {
        if (operand.value == 126 &&
            (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special))
            return exec_subset;
        int base = -1;
        return scalar_pair_operand(operand, base) &&
            mask_subset[base] && mask_subset[base + 1];
    };

    for (size_t i = load_index + 1; i < and_index; ++i) {
        const Rdna2Inst& in = ins[i];
        if (writes_vgpr(in, zero_vgpr)) return false;

        bool derived_mask = false;
        int mask_dst = -1;
        bool next_exec_subset = exec_subset;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64) {
            if (in.dst.value == 126) {
                next_exec_subset = pair_is_subset(in.src[0]);
                if (!next_exec_subset) return false;
            } else if (in.dst.value == 127) {
                // A B64 destination pair must be even-aligned. Reject the invalid EXEC_HI-rooted
                // packet here rather than letting it carry proof state through this exact analysis.
                return false;
            } else if (pair_is_subset(in.src[0])) {
                if (!valid_scalar_pair_base(in.dst.value))
                    return false;
                derived_mask = true;
                mask_dst = in.dst.value;
            }
        }
        if (in.fmt == Rdna2Format::SOP1 &&
            in.opcode == kSop1OpcodeAndSaveexecB64) {
            // The destination receives OLD_EXEC and the new EXEC is OLD_EXEC & src.
            if (!valid_scalar_pair_base(in.dst.value))
                return false;
            derived_mask = exec_subset;
            mask_dst = in.dst.value;
            next_exec_subset = exec_subset;
        } else if (rdna2_instruction_may_change_exec(in)) {
            // CMPX only removes lanes from the current mask. Every other unhandled EXEC writer may
            // activate a lane whose destination was preserved by the zero-record load.
            if (!(in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)) &&
                !(in.fmt == Rdna2Format::SOP1 &&
                  (in.opcode == kSop1OpcodeMovB64 ||
                   in.opcode == kSop1OpcodeAndSaveexecB64)))
                return false;
        }

        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word) {
                const int reg = base + static_cast<int>(word);
                if (reg < 0 || reg >= 128) continue;
                mask_subset[reg] = false;
            }
        });
        if (derived_mask && mask_dst >= 0 && mask_dst + 1 < 128) {
            mask_subset[mask_dst] = true;
            mask_subset[mask_dst + 1] = true;
        }
        exec_subset = next_exec_subset;
    }
    return exec_subset;
}

inline size_t specialize_zero_record_execz_exits(std::vector<Rdna2Inst>& ins,
                                          const ShaderResourceTable* rt,
                                          uint32_t wave_size) {
    if (!rt || wave_size != 64 || ins.empty()) return 0;
    if (std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return (in.fmt == Rdna2Format::SOP1 &&
                    in.opcode >= kSop1OpcodeSetpcB64 &&
                    in.opcode <= kSop1OpcodeRfeB64) ||
                   (in.fmt == Rdna2Format::SOPK &&
                    (in.opcode == kSopkOpcodeCallB64 ||
                     in.opcode == kSopkOpcodeSubvectorLoopBegin ||
                     in.opcode == kSopkOpcodeSubvectorLoopEnd));
        }))
        return 0;

    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);
    std::set<uint32_t> specialized_branches;

    for (const ShaderResource& resource : rt->resources) {
        if (!is_zero_record_raw_buffer(resource)) continue;
        // Translation resolves an instruction-scoped buffer through the table's first matching
        // fetch PC. A later zero marker must not override an earlier ordinary resource here.
        if (rt->by_fetch_pc(resource.fetch_pc) != &resource) continue;
        const auto load_found = index_by_pc.find(resource.fetch_pc);
        if (load_found == index_by_pc.end()) continue;
        const size_t load_index = load_found->second;
        const Rdna2Inst& load = ins[load_index];
        if (!zero_record_load_shape(load)) continue;

        for (size_t i = load_index + 1; i + 2 < ins.size(); ++i) {
            const Rdna2Inst& bit_and = ins[i];
            const Rdna2Inst& compare = ins[i + 1];
            const Rdna2Inst& branch = ins[i + 2];
            if (writes_vgpr(bit_and, load.dst.value)) break;
            const bool and_shape = bit_and.fmt == Rdna2Format::VOP2 &&
                bit_and.opcode == kVop2OpcodeAndB32 &&
                bit_and.len_dwords == 1u && !bit_and.has_sdwa &&
                !bit_and.has_modifier && !bit_and.has_dpp &&
                bit_and.dst.kind == OperandKind::VGPR &&
                bit_and.src[0].kind == OperandKind::InlineInt &&
                bit_and.src[0].value == 7 &&
                bit_and.src[1].kind == OperandKind::VGPR &&
                bit_and.src[1].value == load.dst.value;
            if (!and_shape) continue;
            const bool compare_shape = compare.pc == bit_and.pc + bit_and.len_dwords &&
                compare.fmt == Rdna2Format::VOPC &&
                compare.opcode == kVopcOpcodeCmpxEqU32 &&
                compare.len_dwords == 1u && !compare.has_sdwa &&
                !compare.has_modifier && !compare.has_dpp &&
                compare.src[0].kind == OperandKind::InlineInt &&
                compare.src[0].value == 5 &&
                compare.src[1].kind == OperandKind::VGPR &&
                compare.src[1].value == bit_and.dst.value;
            const bool branch_shape = branch.pc == compare.pc + compare.len_dwords &&
                branch.fmt == Rdna2Format::SOPP &&
                branch.opcode == kSoppOpcodeCbranchExecz && branch.simm16 > 0 &&
                index_by_pc.contains(scalar_branch_target(branch));
            if (!compare_shape || !branch_shape ||
                !zero_load_reaches_and_under_exec_subset(
                    ins, load_index, i, index_by_pc))
                continue;
            specialized_branches.insert(branch.pc);
            break;
        }
    }

    for (uint32_t pc : specialized_branches) {
        Rdna2Inst& branch = ins[index_by_pc.at(pc)];
        branch.opcode = kSoppOpcodeBranch;
        branch.words[0] = 0xbf820000u | static_cast<uint16_t>(branch.simm16);
    }
    if (!specialized_branches.empty()) prune_scalar_cfg_reachability(ins);
    return specialized_branches.size();
}

inline uint64_t shader_program_hash(const uint32_t* code, size_t dwords) {
    uint64_t hash = 1469598103934665603ull;
    const auto* bytes = reinterpret_cast<const uint8_t*>(code);
    for (size_t i = 0; i < dwords * sizeof(uint32_t); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

}  // namespace prosper::gpu
