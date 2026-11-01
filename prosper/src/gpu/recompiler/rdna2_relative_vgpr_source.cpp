// AMD ISA 70648 section 3.6.1: an out-of-range vector source reads current VGPR0.
#include "gpu/recompiler/rdna2_relative_vgpr_source.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_relative_index_values.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <optional>
#include <set>
#include <algorithm>
#include <vector>

namespace prosper::gpu {
int allocated_vgpr_max(const std::vector<Rdna2Inst>& instructions, uint32_t allocation) {
    return std::max(shader_max_vgpr(instructions),
                    allocation ? static_cast<int>(allocation - 1) : -1);
}

void add_allocated_vgprs(std::set<int>& registers, uint32_t allocation) {
    for (uint32_t reg = 0; reg < allocation; ++reg) registers.insert(static_cast<int>(reg));
}
void add_allocated_relative_write(const Rdna2Inst& in, uint32_t allocation, std::set<int>& writes,
                                  std::set<int>& reads) {
    if (!allocation || in.fmt != Rdna2Format::VOP1 || in.opcode != kVop1OpcodeMovreldB32) return;
    for (int reg = in.dst.value; reg < static_cast<int>(allocation); ++reg) {
        writes.insert(reg);
        reads.insert(reg); // Every unselected destination keeps its current value.
    }
}

bool allocated_vgpr_inventory_fits(const std::vector<Rdna2Inst>& instructions,
                                   uint32_t allocation) {
    return !allocation || shader_max_vgpr(instructions) < static_cast<int>(allocation);
}

bool allocated_relative_write_needs_dispatcher(const std::vector<Rdna2Inst>& instructions) {
    return std::any_of(instructions.begin(), instructions.end(), [](const auto& in) {
        return in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32;
    });
}

bool compute_vgpr_allocation_is_canonical(uint32_t words, uint32_t rsrc1, uint32_t wave_size) {
    if (!words) return true;
    if (rsrc1 == UINT32_MAX || (wave_size != 32 && wave_size != 64) || words > 256) return false;
    // COMPUTE_PGM_RSRC1 encoding, AMDGPU ABI: four VGPRs per Wave64 block, eight per Wave32.
    return words == ((rsrc1 & 63u) + 1u) * (wave_size == 64 ? 4u : 8u);
}

std::optional<std::set<int>> relative_vgpr_sources(const SpirvCompute& b, const RegState& rs,
                                                   const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::VOP1 || in.opcode != 0x43 || in.src[0].kind != OperandKind::VGPR ||
        in.src[0].value < 0 || in.src[0].value > 255)
        return std::nullopt;
    const auto m0 = rs.sreg.find(124);
    if (m0 == rs.sreg.end()) return std::nullopt;
    const auto offsets = relative_index_values(b, rs, m0->second, rs.exec);
    const uint32_t limit = b.compute_allocated_vgprs;
    std::set<int> result;
    if (offsets && !offsets->empty()) {
        for (uint32_t offset : *offsets) {
            const uint64_t reg = uint64_t(in.src[0].value) + offset;
            if (limit)
                result.insert(reg < limit ? static_cast<int>(reg) : 0);
            else if (reg <= 255)
                result.insert(static_cast<int>(reg));
            else
                return std::nullopt;
        }
    } else if (limit) {
        result.insert(0); // Every index beyond the allocation selects this genuine source.
        for (uint32_t r = static_cast<uint32_t>(in.src[0].value); r < limit; ++r)
            result.insert(static_cast<int>(r));
    } else
        return std::nullopt;
    return result;
}

bool lower_allocated_movrels(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, uint32_t& result,
                             bool& ok) {
    const uint32_t limit = b.compute_allocated_vgprs;
    if (!limit) return false;
    const auto m0 = rs.sreg.find(124);
    if (m0 == rs.sreg.end()) return false;
    const auto sources = relative_vgpr_sources(b, rs, in);
    if (!sources) {
        ok = false;
        return true;
    }
    // Strict attempts retain the ordinary reader's storage-domain checks too. WRITELANE
    // erases per-lane DATA even when old source marks still describe the earlier clean value.
    for (int reg : *sources) {
        (void)operand_bits(b, rs, in, {OperandKind::VGPR, reg}, &ok);
        if (!ok) return true;
    }
    const auto zero = rs.vreg.find(0);
    const uint32_t fallback = zero != rs.vreg.end() && zero->second ? zero->second : b.uconst(0);
    uint32_t selected = b.uconst(0);
    const uint32_t base = static_cast<uint32_t>(in.src[0].value);
    for (const auto& [reg, value] : rs.vreg) {
        if (!value || reg < static_cast<int>(base) || reg >= static_cast<int>(limit)) continue;
        selected =
            b.sel(b.ucmp(Op_IEqual, m0->second, b.uconst(uint32_t(reg) - base)), value, selected);
    }
    result = base >= limit ? fallback
                           : b.sel(b.ucmp(Op_ULessThan, m0->second, b.uconst(limit - base)),
                                   selected, fallback);
    return true;
}
} // namespace prosper::gpu
