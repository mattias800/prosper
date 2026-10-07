// A finite vector address never grants a loaded pointer without its exact reaching x2 load.
#include "gpu/recompiler/rdna2_loaded_scalar_global.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <optional>
#include <set>

namespace prosper::gpu {
namespace {
using Domain = std::optional<std::set<uint32_t>>;

bool straight_original_body(const std::vector<Rdna2Inst>& ins) {
    if (ins.empty() || !ins.back().is_end) return false;
    uint64_t next = 0;
    for (size_t index = 0; index < ins.size(); ++index) {
        const auto& in = ins[index];
        if (in.pc != next || !in.len_dwords || in.synthetic_terminator ||
            in.fmt == Rdna2Format::Unknown || rdna2_instruction_may_write_memory(in) ||
            (in.is_end && index + 1 != ins.size())) return false;
        next += in.len_dwords;
        if (next > UINT32_MAX) return false;
        // Both earlier joins and later backedges invalidate a lexical reaching definition.
        // AMD RDNA2 ISA 12.5.1: message9 requests GS parameter-cache space. It neither
        // writes guest global memory nor branches; this is not a GS launch/entry proof.
        const bool gs_cache_request = in.fmt == Rdna2Format::SOPP &&
                                     in.opcode == 0x10u && in.simm16 == 9;
        // Other messages, debug branches and traps remain outside this straight-line proof.
        if (in.fmt == Rdna2Format::SOPP && !in.is_end &&
            !gs_cache_request &&
            in.opcode != 0u && in.opcode != 0x0cu && in.opcode != 0x20u &&
            in.opcode != 0x21u && in.opcode != 0x22u) return false;
        if (in.fmt == Rdna2Format::SOP1 &&
            ((in.opcode >= kSop1OpcodeSetpcB64 && in.opcode <= kSop1OpcodeRfeB64) ||
             (in.opcode >= 0x30u && in.opcode <= 0x33u) ||
             in.opcode == kSop1OpcodeMovrelsd2B32)) return false;
        if (in.fmt == Rdna2Format::SOPK &&
            (in.opcode == kSopkOpcodeCallB64 ||
             in.opcode == kSopkOpcodeSubvectorLoopBegin ||
             in.opcode == kSopkOpcodeSubvectorLoopEnd)) return false;
        if (in.fmt == Rdna2Format::VOP1 &&
            in.opcode == kVop1OpcodeMovreldB32) return false;
    }
    return true;
}

bool writes_pair(const Rdna2Inst& in, uint32_t first) {
    bool written = false;
    const auto overlaps = [first](int base, uint32_t width) {
        return base >= 0 && uint64_t(base) < uint64_t(first) + 2u &&
               uint64_t(base) + width > first;
    };
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        written |= overlaps(base, width);
    });
    // Storage emission inventories only supported ALU destinations. A code-side pointer
    // exclusion additionally uses the decoder's conservative pair width for unknown SOP2.
    if (in.fmt == Rdna2Format::SOP2)
        written |= overlaps(in.dst.value, rdna2_sop2_dest_dwords(in.opcode));
    return written;
}

bool whole_valu(const Rdna2Inst& in) {
    return !in.has_modifier && !in.has_dpp && !in.clamp && !in.omod &&
        (!in.has_sdwa || (in.sdwa_dst_sel == 6u && in.sdwa_src0_sel == 6u &&
                         in.sdwa_src1_sel == 6u && !in.sdwa_src0_sext &&
                         !in.sdwa_src1_sext)) &&
        !in.src_neg[0] && !in.src_neg[1] && !in.src_abs[0] && !in.src_abs[1];
}

Domain operand_domain(const Rdna2Inst& in, const Operand& op,
                       const std::array<Domain, 256>& regs) {
    if (op.kind == OperandKind::VGPR && op.value >= 0 && op.value < 256)
        return regs[op.value];
    if (op.kind == OperandKind::InlineInt) return std::set<uint32_t>{uint32_t(op.value)};
    if (op.kind == OperandKind::InlineFloat)
        return std::set<uint32_t>{std::bit_cast<uint32_t>(inline_float_value(op.value))};
    if (op.kind == OperandKind::Literal && in.has_literal)
        return std::set<uint32_t>{in.literal};
    return {};
}

Domain valu_domain(const Rdna2Inst& in, const std::array<Domain, 256>& regs) {
    if (!whole_valu(in)) return {};
    auto first = operand_domain(in, in.src[0], regs);
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == 1u) return first;
    if (in.fmt == Rdna2Format::VOP2 && in.opcode == 1u) {
        const auto second = operand_domain(in, in.src[1], regs);
        if (!first || !second) return {};
        first->insert(second->begin(), second->end());
        return first->size() <= 16u ? first : Domain{};
    }
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == 8u && first) {
        std::set<uint32_t> converted;
        for (uint32_t bits : *first) {
            const double value = std::bit_cast<float>(bits);
            if (!std::isfinite(value) || value < -2147483648.0 || value >= 2147483648.0)
                return {};
            converted.insert(uint32_t(int32_t(std::trunc(value))));
        }
        return converted;
    }
    if (in.fmt == Rdna2Format::VOP2 && in.opcode == 0x1au && first) {
        const auto second = operand_domain(in, in.src[1], regs);
        if (!second) return {};
        std::set<uint32_t> shifted;
        for (uint32_t shift : *first)
            for (uint32_t value : *second) shifted.insert(value << (shift & 31u));
        if (shifted.size() <= 16u) return shifted;
    }
    return {};
}

bool completes_after(const std::vector<Rdna2Inst>& ins, size_t index, bool scalar) {
    // This initial contract permits only hints between the load and full completion.
    // AMD RDNA2 ISA 4.4/12.5: scalar loads may complete out of order, so partial
    // LGKM counts cannot prove this result. Segment2 GLOBAL loads use the VM counter.
    while (++index < ins.size()) {
        const auto& in = ins[index];
        if (in.fmt != Rdna2Format::SOPP || in.is_end || in.len_dwords != 1u) return false;
        if (in.opcode == 0u || in.opcode == 0x20u || in.opcode == 0x21u) continue;
        if (in.opcode != 0x0cu) return false;
        const uint32_t imm = uint16_t(in.simm16);
        return scalar ? ((imm >> 8u) & 63u) == 0u
                      : (((imm >> 10u) & 48u) | (imm & 15u)) == 0u;
    }
    return false;
}

std::optional<LoadedScalarGlobalRead> readpoint(
        const std::vector<Rdna2Inst>& ins, size_t global_index, const Domain& offsets) {
    const auto& global = ins[global_index];
    if (!offsets || offsets->empty() || global.fmt != Rdna2Format::FLAT ||
        global.len_dwords != 2u || global.n_src != 2u ||
        global.flat_segment != 2u || global.opcode != 0x0eu || global.flat_lds ||
        global.flat_glc || global.flat_slc || global.flat_dlc ||
        global.src[0].kind != OperandKind::VGPR || global.src[0].value < 0 ||
        global.src[0].value >= 256 || global.src[1].kind != OperandKind::SGPR ||
        global.src[1].value < 0 || global.src[1].value > 104 ||
        global.dst.kind != OperandKind::VGPR || global.dst.value < 0 ||
        global.dst.value > 252 || !completes_after(ins, global_index, false)) return {};
    const uint32_t base = static_cast<uint32_t>(global.src[1].value);
    size_t parent_index = global_index;
    while (parent_index && !writes_pair(ins[parent_index - 1u], base)) --parent_index;
    if (!parent_index) return {};
    --parent_index;
    const auto& parent = ins[parent_index];
    if (parent.fmt != Rdna2Format::SMEM || parent.opcode != kSmemOpcodeLoadDwordX2 ||
        parent.len_dwords != 2u || parent.n_src != 2u ||
        parent.dst.kind != OperandKind::SGPR || parent.dst.value != int(base) ||
        parent.src[0].kind != OperandKind::SGPR || parent.src[0].value < 0 ||
        parent.src[0].value > 104 || parent.src[1].kind != OperandKind::Special ||
        parent.src[1].value != 125 || (parent.literal & 3u) || int32_t(parent.literal) < 0 ||
        !completes_after(ins, parent_index, true)) return {};
    const uint32_t entry = static_cast<uint32_t>(parent.src[0].value);
    for (size_t index = 0; index < parent_index; ++index)
        if (writes_pair(ins[index], entry)) return {};
    uint64_t low = UINT64_MAX, high = 0;
    for (uint32_t offset : *offsets) {
        const int64_t effective = int64_t(offset) + int32_t(global.literal);
        if (effective < 0 || effective > UINT32_MAX || (effective & 3)) return {};
        low = std::min(low, uint64_t(effective));
        high = std::max(high, uint64_t(effective));
    }
    const uint64_t required = high - low + 16u;
    if (required > 32u) return {};
    return LoadedScalarGlobalRead{parent.pc, global.pc, entry, base,
        static_cast<uint32_t>(global.src[0].value), parent.literal,
        static_cast<uint32_t>(low), required <= 16u ? 16u : 32u, int32_t(global.literal)};
}
} // namespace

std::vector<LoadedScalarGlobalRead> rdna2_loaded_scalar_global_reads(
        const std::vector<Rdna2Inst>& original) {
    if (!straight_original_body(original)) return {};
    std::array<Domain, 256> regs{};
    std::vector<LoadedScalarGlobalRead> result;
    for (size_t index = 0; index < original.size(); ++index) {
        const auto& in = original[index];
        // Definitions under an old EXEC cannot initialize lanes newly enabled by a later mask.
        if (rdna2_instruction_may_change_exec(in)) for (auto& reg : regs) reg.reset();
        if (in.fmt == Rdna2Format::FLAT && in.src[0].kind == OperandKind::VGPR &&
            in.src[0].value >= 0 && in.src[0].value < 256)
            if (auto proof = readpoint(original, index, regs[in.src[0].value]))
                result.push_back(*proof);
        const auto value = valu_domain(in, regs);
        const uint32_t writes = rdna2_vgpr_write_count(in);
        if (in.dst.kind == OperandKind::VGPR && in.dst.value >= 0)
            for (uint32_t word = 0; word < writes && uint64_t(in.dst.value) + word < 256u; ++word)
                regs[in.dst.value + word] = writes == 1u ? value : Domain{};
        if (const int tfe = rdna2_tfe_status_vgpr(in); tfe >= 0 && tfe < 256) regs[tfe].reset();
    }
    return result;
}
} // namespace prosper::gpu
