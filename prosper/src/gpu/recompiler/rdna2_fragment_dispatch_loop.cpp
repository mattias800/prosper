// Only the private dispatch continuation consumes this finite-loop proof.
// It supplies neither an existing classifier verdict nor scalar DATA authority.
#include "gpu/recompiler/rdna2_fragment_dispatch_loop.hpp"

#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <algorithm>
#include <vector>

namespace prosper::gpu {
bool fragment_dispatch_interior_loops_are_finite(const std::vector<Rdna2Inst>& selected,
                                                 uint32_t early_out_pc, uint32_t gate_pc) {
    const auto target = [](const Rdna2Inst& in) {
        return int64_t(in.pc) + in.len_dwords + in.simm16;
    };
    const auto branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
    };
    const auto writes = [](const Rdna2Inst& in, int word) {
        bool overlap = false;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            overlap |= base <= word && word < base + static_cast<int>(width);
        });
        return overlap;
    };
    const auto interior = [&](const Rdna2Inst& in) {
        return in.pc > early_out_pc && in.pc < gate_pc;
    };
    if (std::none_of(selected.begin(), selected.end(), [&](const auto& in) {
            return interior(in) && branch(in) && target(in) <= in.pc;
        }))
        return true;
    // Recognizing a counted CFG alone proves no progress. Restrict this new permit
    // to one top-tested, unsigned/signed LT loop with a dominating zero initializer,
    // an invariant 32-bit bound and an unconditional unit increment on every latch path.
    const auto loop = detect_counted_loop(selected);
    if (!loop.found || !loop.exit_on_scc0 || loop.exit_branch_pc == loop.backedge_pc ||
        loop.header_pc <= early_out_pc || loop.backedge_pc >= gate_pc || loop.exit_pc > gate_pc)
        return false;
    const Rdna2Inst* compare = nullptr;
    for (const auto& in : selected) {
        if (in.pc < loop.header_pc || in.pc >= loop.exit_branch_pc) continue;
        if (sopp_is_noop(in)) continue;
        if (compare || in.fmt != Rdna2Format::SOPC || (in.opcode != 0x04 && in.opcode != 0x0a) ||
            in.n_src != 2)
            return false;
        compare = &in;
    }
    if (!compare || compare->src[0].kind != OperandKind::SGPR || compare->src[0].value < 0 ||
        compare->src[0].value > 104)
        return false;
    const int counter = compare->src[0].value;
    const auto& bound = compare->src[1];
    if (!(bound.kind == OperandKind::InlineInt ||
          (bound.kind == OperandKind::Literal && compare->has_literal) ||
          (bound.kind == OperandKind::SGPR && bound.value >= 0 && bound.value <= 104 &&
           bound.value != counter)))
        return false;
    const Rdna2Inst* initializer = nullptr;
    const Rdna2Inst* step = nullptr;
    for (const auto& in : selected) {
        if (in.pc < loop.header_pc) {
            if (writes(in, counter)) initializer = &in;
            continue;
        }
        if (in.pc > loop.backedge_pc) break;
        if (rdna2_escapes_decoded_effects(in)) return false;
        if (bound.kind == OperandKind::SGPR && writes(in, bound.value)) return false;
        if (!writes(in, counter)) continue;
        if (step || in.pc <= loop.exit_branch_pc || in.pc >= loop.backedge_pc ||
            in.fmt != Rdna2Format::SOP2 || (in.opcode != 0 && in.opcode != 2) ||
            in.dst.kind != OperandKind::SGPR || in.dst.value != counter || in.n_src != 2 ||
            in.src[0].kind != OperandKind::SGPR || in.src[0].value != counter ||
            in.src[1].kind != OperandKind::InlineInt || in.src[1].value != 1)
            return false;
        step = &in;
    }
    if (!step || !initializer || initializer->pc <= early_out_pc ||
        initializer->fmt != Rdna2Format::SOP1 || initializer->opcode != kSop1OpcodeMovB32 ||
        initializer->dst.kind != OperandKind::SGPR || initializer->dst.value != counter ||
        initializer->src[0].kind != OperandKind::InlineInt || initializer->src[0].value != 0)
        return false;
    for (const auto& in : selected) {
        if (!branch(in)) continue;
        const int64_t to = target(in);
        if (interior(in) && to <= in.pc && in.pc != loop.backedge_pc)
            return false; // the shared detector excludes EXEC/VCC and waterfall backedges
        if (in.pc < loop.header_pc && to > initializer->pc && to <= loop.backedge_pc)
            return false; // no entry edge skips the authentic induction initializer
        if (in.pc < loop.header_pc || in.pc > loop.backedge_pc || in.pc == loop.exit_branch_pc ||
            in.pc == loop.backedge_pc)
            continue;
        if (to == loop.exit_pc) continue; // an earlier exit cannot add iterations
        if (to <= in.pc || to <= loop.exit_branch_pc || to > step->pc)
            return false; // no inner cycle or path to the latch skips the unit increment
    }
    // For signed LT, a nonpositive bound exits immediately; a positive bound is at
    // most INT32_MAX. For unsigned LT, the bound is at most UINT32_MAX. Starting at
    // zero and adding one reaches either bound before any signed/unsigned wrap.
    return true;
}
} // namespace prosper::gpu
