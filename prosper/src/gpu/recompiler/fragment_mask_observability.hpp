#pragma once

#include "gpu/recompiler/rdna2_cfg_support.hpp"

#include <bitset>

namespace prosper::gpu {

struct FragmentMaskObservability {
    bool admitted = false;
    const char* reason = "unproved-mask-observability";
    uint32_t blocker_pc = UINT32_MAX;
};

// A projected scalar pair retains only participating invocations' bits. Follow every alias of that
// projection, including across loop backedges, and refuse consumers that could observe other bits.
// `derived` is MAY provenance; `bounded` is MUST zero outside the initial participating set. Keep
// both: AND with EXEC bounds a derived value, but a later NOT can make its omitted bits nonzero.
// Called after fragment_loop_mask validates the direct CFG and physical write lifetime. This is an
// admission proof, not a replacement for emitter domain/SSA and subgroup validation.
inline FragmentMaskObservability prove_fragment_mask_observability(
        const std::vector<Rdna2Inst>& ins, int pair_base, uint32_t seed_pc,
        uint32_t header_pc) {
    using Bits = std::bitset<128>;
    struct State {
        Bits derived, bounded;
        bool uncertain_scc = false;
        bool operator==(const State&) const = default;
    };
    struct Fact { bool derived = false, bounded = false; };
    auto reject = [](const char* reason, uint32_t pc) {
        return FragmentMaskObservability{false, reason, pc};
    };
    if (ins.empty()) return reject("empty-mask-observability-stream", seed_pc);
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t i = 0; i < ins.size(); ++i) by_pc.emplace(ins[i].pc, i);
    if (!by_pc.contains(seed_pc)) return reject("missing-mask-seed", seed_pc);
    const auto scalar = [](const Operand& op) {
        return op.kind == OperandKind::SGPR || op.kind == OperandKind::Special;
    };
    const auto overlap = [](int a, uint32_t width, int pair) {
        return width && a <= pair + 1 && static_cast<int64_t>(a) + width > pair;
    };
    const auto logic = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOP2 && in.opcode >= 0x0f &&
               in.opcode <= 0x1d && (in.opcode & 1u);
    };
    const auto logical_fact = [](uint32_t op, Fact a, Fact b) {
        Fact out{a.derived || b.derived, false};
        if (op == 0x0f) out.bounded = a.bounded || b.bounded;       // AND
        if (op == 0x11 || op == 0x13) out.bounded = a.bounded && b.bounded;
        if (op == 0x15) out.bounded = a.bounded;                    // A & ~B
        // ORN2/NAND/NOR/XNOR may set omitted bits even from two bounded inputs.
        return out;
    };
    std::vector<State> incoming(ins.size());
    std::vector<bool> reachable(ins.size(), false);
    incoming.front().bounded.set(126); // Entry EXEC defines the participating set.
    reachable.front() = true;
    std::vector<size_t> pending{0};
    while (!pending.empty()) {
        const size_t index = pending.back();
        pending.pop_back();
        const Rdna2Inst& in = ins[index];
        State state = incoming[index];
        if (in.is_end) continue;
        const auto fact = [&](const Operand& op) {
            if (op.kind == OperandKind::InlineInt) return Fact{false, op.value == 0};
            if (!scalar(op) || op.value < 0 || op.value >= 128) return Fact{};
            return Fact{state.derived.test(op.value), state.bounded.test(op.value)};
        };
        const auto publish = [&](int base, Fact value) {
            if (base < 0 || base > 126) return;
            state.derived.set(base, value.derived);
            state.bounded.set(base, value.bounded);
        };
        const bool mov = in.fmt == Rdna2Format::SOP1 && in.opcode == 0x04;
        const bool invert = in.fmt == Rdna2Format::SOP1 && in.opcode == 0x08;
        const bool wqm = in.fmt == Rdna2Format::SOP1 && in.opcode == 0x0a;
        const bool select = in.fmt == Rdna2Format::SOP2 && in.opcode == 0x0b;
        const bool saveexec = in.fmt == Rdna2Format::SOP1 &&
            ((in.opcode >= 0x24 && in.opcode <= 0x2b) ||
             in.opcode == 0x37 || in.opcode == 0x38);
        const bool pointwise = logic(in) || mov || invert || select;
        const bool cndmask = in.fmt == Rdna2Format::VOP3 && in.opcode == 0x101;
        // Eager seeding must not change a prefix consumer's choice of representation. These
        // descriptor/address and ordinary B32 paths read the real scalar words without preferring
        // sreg_bool. All other raw reads must prove themselves below, including QUADMASK, U64
        // compares, reductions and WRITELANE, whose emitters may select the Bool view first.
        const bool raw_prefix_data =
            in.fmt == Rdna2Format::MIMG || in.fmt == Rdna2Format::MUBUF ||
            in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::SMEM ||
            in.fmt == Rdna2Format::FLAT ||
            (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 &&
             in.dst.value >= 0 && in.dst.value <= 105) ||
            (in.fmt == Rdna2Format::SOP2 && in.opcode <= kSop2OpcodeCselectB32 &&
             in.dst.value >= 0 && in.dst.value <= 105) ||
            (in.fmt == Rdna2Format::SOPC && in.opcode <= 0x0d) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode >= 0x03 && in.opcode <= 0x0e) ||
            (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x01);
        const bool branch = in.fmt == Rdna2Format::SOPP &&
                            sopp_opcode_is_direct_branch(in.opcode);
        const bool scc_read =
            (branch && (in.opcode == 0x04 || in.opcode == 0x05)) ||
            (in.fmt == Rdna2Format::SOP2 &&
             (in.opcode == 0x04 || in.opcode == 0x05 || in.opcode == 0x0a || select)) ||
            (in.fmt == Rdna2Format::SOP1 && (in.opcode == 0x05 || in.opcode == 0x06)) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == 0x02);
        if (state.uncertain_scc && scc_read)
            return reject("derived-mask-scc-consumer", in.pc);
        for (uint32_t source = 0; source < in.n_src; ++source)
            if (state.uncertain_scc && scalar(in.src[source]) && in.src[source].value == 253)
                return reject("derived-mask-scc-source", in.pc);
        if (branch && (in.opcode == 0x06 || in.opcode == 0x07) &&
            state.derived.test(106) && !state.bounded.test(106))
            return reject("unbounded-derived-vcc-branch", in.pc);
        if (branch && (in.opcode == 0x08 || in.opcode == 0x09) &&
            state.derived.test(126) && !state.bounded.test(126))
            return reject("unbounded-derived-exec-branch", in.pc);

        // Check each remaining scalar touch using the existing physical-width read inventory.
        // A one-instruction walk has no later value to kill, so its ordinary destination handling
        // cannot hide another read. Only exact, admitted pointwise mask operands are sanitized.
        Rdna2Inst read_check = in;
        for (uint32_t source = 0; source < in.n_src; ++source) {
            const bool mask_operand = pointwise || saveexec || (cndmask && source == 2);
            if (!mask_operand || !scalar(in.src[source])) continue;
            bool exact = true;
            for (int base = 0; base < 127; ++base)
                if (state.derived.test(base) && overlap(in.src[source].value, 2, base) &&
                    in.src[source].value != base) exact = false;
            if (exact) read_check.src[source] = {OperandKind::InlineInt, 0};
        }
        for (int base = 0; base < 127; ++base) {
            if (!state.derived.test(base)) continue;
            // The original scalar words remain real data before the promoted loop. Other aliases
            // deliberately get no scalar-data exception, even if an AND has bounded their bits.
            if (base == pair_base && in.pc < header_pc && raw_prefix_data) continue;
            if (branch) continue; // Wave and SCC predicates were checked above.
            if (in.fmt == Rdna2Format::VINTRP) continue; // VGPR input plus implicit M0, never SGPR pair.
            const uint32_t implicit_width = scalar_implicit_destination_read_width(in);
            if (in.fmt == Rdna2Format::SOPK && implicit_width) {
                // The shared liveness walk declines most SOPK opcodes without inspecting their
                // destination. Their complete scalar input is nevertheless the decoded one-word
                // implicit destination read (CMPK/ADDK/MULK/CMOVK/SETREG).
                if (overlap(in.dst.value, implicit_width, base))
                    return reject("derived-mask-data-or-reduction", in.pc);
                continue;
            }
            if (in.fmt == Rdna2Format::SMEM &&
                overlap(in.src[0].value, in.opcode >= 0x08 ? 4u : 2u, base))
                return reject("derived-mask-data-or-reduction", in.pc);
            for (int word = 0; word < 2; ++word)
                if (!sgpr_dead_at_merge(std::vector<Rdna2Inst>{read_check}, in.pc, base + word))
                    return reject(wqm ? "derived-mask-wqm" : "derived-mask-data-or-reduction", in.pc);
        }

        const Fact a = fact(in.src[0]), b = fact(in.src[1]);
        const Fact old_exec{state.derived.test(126), state.bounded.test(126)};
        Fact output;
        if (logic(in)) output = logical_fact(in.opcode, a, b);
        else if (mov) output = a;
        else if (invert) output = {a.derived, false};
        else if (select) output = {a.derived || b.derived, a.bounded && b.bounded};
        Fact new_exec;
        if (saveexec) {
            // SAVEEXEC's ANDN2/ORN2 spelling reverses the operands relative to SOP2.
            const uint32_t op = in.opcode == 0x37 ? 0x15 : in.opcode == 0x38 ? 0x17
                                                : 0x0f + 2 * (in.opcode - 0x24);
            new_exec = (in.opcode == 0x27 || in.opcode == 0x28)
                ? logical_fact(op, a, old_exec) : logical_fact(op, old_exec, a);
        }
        if ((pointwise && in.dst.value == 126 && output.derived && !output.bounded) ||
            (saveexec && new_exec.derived && !new_exec.bounded))
            return reject("unbounded-derived-exec-write", in.pc);
        if ((pointwise && output.derived) || (saveexec && old_exec.derived)) {
            const int dst = in.dst.value;
            if (dst < 0 || (dst & 1) || (dst > 104 && dst != 106 && dst != 126))
                return reject("unsupported-derived-mask-destination", in.pc);
        }
        bool partial_derived_write = false;
        const auto clear_write = [&](int written, uint32_t width) {
            for (int base = 0; base < 127; ++base) {
                if (!overlap(written, width, base)) continue;
                const bool covers = written <= base &&
                    static_cast<int64_t>(written) + width >= base + 2;
                if (state.derived.test(base) && !covers) partial_derived_write = true;
                if (covers) state.derived.reset(base);
                state.bounded.reset(base);
            }
        };
        for_each_scalar_write(in, clear_write);
        // The generic scalar writer inventory omits architectural VCC and implicit EXEC writes.
        // They still clobber overlapping saved-pair bounds (including non-derived odd aliases).
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) && in.dst.value == 106)
            clear_write(106, 2);
        if (saveexec || (in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)))
            clear_write(126, 2);
        if (partial_derived_write) return reject("partial-derived-mask-write", in.pc);
        if (pointwise) publish(in.dst.value, output);
        if (saveexec) {
            publish(in.dst.value, old_exec);
            publish(126, new_exec);
        }
        if (in.fmt == Rdna2Format::VOPC) {
            if (vopc_is_cmpx(in.opcode)) publish(126, {false, old_exec.bounded});
            else publish(in.dst.value, {false, old_exec.bounded});
        }
        if (vop3_writes_mask_sdst(in)) publish(in.sdst.value, {false, old_exec.bounded});
        if (wqm) {
            // Existing WQM lowering is outside this proof. It cannot establish a fresh bound.
            publish(in.dst.value, {false, false});
        } else if (!pointwise && !saveexec && in.fmt != Rdna2Format::VOPC &&
                   rdna2_instruction_may_change_exec(in)) {
            state.bounded.reset(126);
        }
        // Query the shared inventory with a register outside every physical operand span. A false
        // result means the inventory itself is unknown, not a real read. Such an instruction must
        // not preserve a saved-EXEC bound merely because it precedes the first derived alias use.
        // SOPK comparisons have an explicit known destination-read span; VINTRP has only VGPR/M0.
        const bool known_sopk = in.fmt == Rdna2Format::SOPK && in.opcode <= 0x10;
        if (!pointwise && !saveexec && !wqm && !branch && !known_sopk &&
            in.fmt != Rdna2Format::VINTRP &&
            !sgpr_dead_at_merge(std::vector<Rdna2Inst>{in}, in.pc, 1000))
            state.bounded.reset();
        if (logic(in) || invert)
            state.uncertain_scc = output.derived && !output.bounded;
        else if (saveexec)
            state.uncertain_scc = new_exec.derived && !new_exec.bounded;
        else if (in.fmt == Rdna2Format::SOPC ||
                 (in.fmt == Rdna2Format::SOPK && in.opcode >= 0x03 && in.opcode <= 0x0e) ||
                 (in.fmt == Rdna2Format::SOP2 && in.opcode <= 0x09))
            state.uncertain_scc = false;
        if (in.pc == seed_pc) publish(pair_base, {true, false});

        std::vector<size_t> successors;
        if (branch) {
            const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
            if (target < 0 || target > UINT32_MAX || !by_pc.contains(static_cast<uint32_t>(target)))
                return reject("invalid-observability-branch", in.pc);
            successors.push_back(by_pc.at(static_cast<uint32_t>(target)));
        }
        if ((!branch || in.opcode != 0x02) && index + 1 < ins.size())
            successors.push_back(index + 1);
        for (size_t next : successors) {
            State joined = state;
            if (reachable[next]) {
                joined.derived |= incoming[next].derived;
                joined.bounded &= incoming[next].bounded;
                joined.uncertain_scc |= incoming[next].uncertain_scc;
            }
            if (!reachable[next] || !(joined == incoming[next])) {
                incoming[next] = joined;
                reachable[next] = true;
                pending.push_back(next);
            }
        }
    }
    return {true, "derived-mask-bits-unobservable", UINT32_MAX};
}

} // namespace prosper::gpu
