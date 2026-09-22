#pragma once

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/fragment_mask_observability.hpp"

#include <array>

namespace prosper::gpu {

struct FragmentLoopMaskProof {
    bool admitted = false;
    const char* reason = "unproved";
    uint32_t blocker_pc = UINT32_MAX;
    std::array<uint32_t, 2> definition_pc{UINT32_MAX, UINT32_MAX};
};

// A deliberately bounded bridge into a structured loop's Bool PHIs. Both scalar words must be
// defined before the first branch, remain unchanged up to the header, and have a mask-only lifetime
// thereafter. Seed the Bool immediately after the later definition, before any branch, so an arm
// bypassing the loop also retains the original predicate. This proves register uses, not scalar-load
// emission or subgroup support: the caller still needs actual scalar SSA values, exact fragment
// lane identity, and must remove scalar PHIs for the promoted pair. No game/address specialization.
inline FragmentLoopMaskProof prove_fragment_loop_mask(
        const std::vector<Rdna2Inst>& ins, int pair_base, uint32_t header_pc) {
    FragmentLoopMaskProof proof;
    auto decline = [&](const char* reason, uint32_t pc) {
        proof.reason = reason;
        proof.blocker_pc = pc;
        return proof;
    };
    if (pair_base < 0 || pair_base > 104 || (pair_base & 1))
        return decline("ordinary-aligned-pair-required", header_pc);

    std::unordered_map<uint32_t, size_t> index_for_pc;
    for (size_t i = 0; i < ins.size(); ++i) {
        if (!ins[i].len_dwords || !index_for_pc.emplace(ins[i].pc, i).second ||
            (i && ins[i].pc != ins[i - 1].pc + ins[i - 1].len_dwords))
            return decline("invalid-instruction-layout", ins[i].pc);
    }
    const auto header = index_for_pc.find(header_pc);
    if (header == index_for_pc.end()) return decline("missing-loop-header", header_pc);
    const auto overlaps = [&](int base, uint32_t width) {
        return width && base <= pair_base + 1 &&
               static_cast<int64_t>(base) + width > pair_base;
    };
    const auto scalar_operand = [](const Operand& operand) {
        return operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special;
    };
    const auto b64_logic = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOP2 && in.opcode >= 0x0f &&
               in.opcode <= 0x1d && (in.opcode & 1u);
    };
    const auto mask_source = [&](const Rdna2Inst& in, uint32_t source) {
        if (b64_logic(in) ||
            (in.fmt == Rdna2Format::SOP2 && in.opcode == kSop2OpcodeCselectB64))
            return source < 2;
        if (in.fmt == Rdna2Format::SOP1 && source == 0) {
            return in.opcode == kSop1OpcodeMovB64 || in.opcode == 0x08 ||
                   (in.opcode >= kSop1OpcodeAndSaveexecB64 &&
                    in.opcode <= kSop1OpcodeXnorSaveexecB64) ||
                   in.opcode == kSop1OpcodeAndn1SaveexecB64 ||
                   in.opcode == kSop1OpcodeOrn1SaveexecB64;
        }
        return in.fmt == Rdna2Format::VOP3 && in.opcode == 0x101 && source == 2;
    };
    const auto known_mask = [&](const Operand& operand) {
        // MOV/CSELECT can also emit scalar data and erase the destination's Bool. A physical
        // VCC pair is not proof of its representation: scalar writes can leave rs.vcc absent.
        // The eagerly seeded pair, EXEC, and inline integers always retain a Bool view;
        // inline_int_mask_bit projects every signed inline value using the exact fragment lane.
        return operand.kind == OperandKind::InlineInt ||
               (operand.kind == OperandKind::SGPR && operand.value == pair_base) ||
               (scalar_operand(operand) && operand.value == 126);
    };
    const auto mask_writer = [&](const Rdna2Inst& in, int base, uint32_t width) {
        if (base != pair_base || width != 2 || !scalar_write_is_b64_mask(in, base))
            return false;
        if (b64_logic(in)) return true; // This emitter has no scalar-data fallback.
        if (in.fmt == Rdna2Format::SOP2 && in.opcode == kSop2OpcodeCselectB64)
            return known_mask(in.src[0]) && known_mask(in.src[1]);
        if (in.fmt == Rdna2Format::SOP1) {
            // MOV also has a scalar spelling. Certify the mask spelling explicitly.
            if (in.opcode == kSop1OpcodeMovB64 || in.opcode == 0x08)
                return known_mask(in.src[0]);
            return mask_source(in, 0); // SAVEEXEC always publishes the old EXEC mask.
        }
        return in.fmt == Rdna2Format::VOPC || vop3_writes_mask_sdst(in);
    };

    bool branch_seen = false, mask_read_seen = false, mask_write_seen = false;
    std::vector<Rdna2Inst> sanitized = ins;
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) {
            if (i < header->second) return decline("entry-ends-before-header", in.pc);
            continue;
        }
        if (in.fmt == Rdna2Format::Unknown)
            return decline("unknown-instruction", in.pc);
        // Relative SGPR operations can touch a register not named by a decoded operand. Indirect
        // control and subvector calls likewise escape the direct CFG used by the liveness walk.
        if ((in.fmt == Rdna2Format::SOP1 &&
             ((in.opcode >= kSop1OpcodeSetpcB64 && in.opcode <= kSop1OpcodeRfeB64) ||
              (in.opcode >= 0x2e && in.opcode <= 0x31) || in.opcode == 0x49)) ||
            (in.fmt == Rdna2Format::SOPK &&
             (in.opcode == kSopkOpcodeCallB64 ||
              in.opcode == kSopkOpcodeSubvectorLoopBegin ||
              in.opcode == kSopkOpcodeSubvectorLoopEnd)))
            return decline("unmodelled-control-or-relative-sgpr", in.pc);
        if (in.fmt == Rdna2Format::SOPP) {
            if (sopp_opcode_is_direct_branch(in.opcode)) {
                const int64_t target = static_cast<int64_t>(in.pc) +
                                       in.len_dwords + in.simm16;
                if (target < 0 || target > UINT32_MAX ||
                    !index_for_pc.contains(static_cast<uint32_t>(target)))
                    return decline("invalid-branch-target", in.pc);
                if (i < header->second && target <= in.pc)
                    return decline("prefix-branch-rewinds", in.pc);
                if (i >= header->second && target < header_pc)
                    return decline("mask-lifetime-reenters-prefix", in.pc);
                branch_seen = true;
            } else if (!sopp_is_noop(in) && in.opcode != kSoppOpcodeBarrier) {
                return decline("unmodelled-control-flow", in.pc);
            }
        }

        bool invalid_write = false;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            if (!overlaps(base, width)) return;
            if (i < header->second) {
                const bool scalar_definition =
                    (in.fmt == Rdna2Format::SMEM && scalar_write_width(in) != 0) ||
                    (in.fmt == Rdna2Format::SOP1 &&
                     in.opcode == kSop1OpcodeMovB32 &&
                     (in.src[0].kind == OperandKind::InlineInt ||
                      in.src[0].kind == OperandKind::Literal)) ||
                    (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeMovkI32);
                if (branch_seen || !scalar_definition) { invalid_write = true; return; }
                for (int word = 0; word < 2; ++word) {
                    if (pair_base + word < base ||
                        pair_base + word >= static_cast<int64_t>(base) + width) continue;
                    if (proof.definition_pc[word] != UINT32_MAX) invalid_write = true;
                    else proof.definition_pc[word] = in.pc;
                }
            } else {
                if (!mask_writer(in, base, width)) { invalid_write = true; return; }
                mask_write_seen = true;
                if (in.dst.kind == OperandKind::SGPR && in.dst.value == pair_base)
                    sanitized[i].dst = {OperandKind::SGPR, 125};
                if (vop3_writes_mask_sdst(in) && in.sdst.value == pair_base)
                    sanitized[i].sdst = {OperandKind::SGPR, 125};
            }
        });
        if (invalid_write)
            return decline(i < header->second ? "prefix-pair-overwrite-or-unproved-definition"
                                             : "non-mask-or-partial-pair-write", in.pc);
        if (i < header->second) continue;
        if (proof.definition_pc[0] == UINT32_MAX || proof.definition_pc[1] == UINT32_MAX)
            return decline("pair-not-defined-in-entry-prefix", header_pc);

        // sgpr_dead_at_merge already accounts for descriptor and implicit destination reads. Its
        // raw-SMEM SBASE check covers two words; buffered loads consume four, so check that span
        // here too. Replace only admitted exact mask operands; all remaining touches must be dead.
        if (in.fmt == Rdna2Format::SMEM &&
            overlaps(in.src[0].value, in.opcode >= 0x08 ? 4u : 2u))
            return decline("scalar-memory-address-read", in.pc);
        for (uint32_t source = 0; source < in.n_src; ++source) {
            if (in.src[source].kind == OperandKind::SGPR &&
                in.src[source].value == pair_base && mask_source(in, source)) {
                sanitized[i].src[source] = {OperandKind::InlineInt, 0};
                mask_read_seen = true;
            }
        }
    }
    if (!mask_read_seen) return decline("no-mask-consumer", header_pc);
    if (!mask_write_seen) return decline("no-loop-carried-mask-write", header_pc);
    for (int word = 0; word < 2; ++word) {
        ScalarMergeBlocker blocker;
        if (!sgpr_dead_at_merge(sanitized, header_pc, pair_base + word,
                                ScalarMergeProof::AnyRead, &blocker))
            return decline("scalar-or-unmodelled-read", blocker.pc);
    }
    const auto observability = prove_fragment_mask_observability(
        ins, pair_base, std::max(proof.definition_pc[0], proof.definition_pc[1]), header_pc);
    if (!observability.admitted)
        return decline(observability.reason, observability.blocker_pc);
    proof.admitted = true;
    proof.reason = "entry-defined-mask-only-lifetime";
    return proof;
}

} // namespace prosper::gpu
