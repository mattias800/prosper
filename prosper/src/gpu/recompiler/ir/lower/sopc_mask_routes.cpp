// sopc_mask_routes.cpp — SOPC routes that compare wave masks rather than scalar data.
// Relocated verbatim from emit_alu's SOPC case by tools/refactor/extract_case_routes.py (ADR 0012).
// These routes still emit SPIR-V directly; they are the mask-domain and proof-carrying shapes the
// IR does not model yet. Each returns nullopt when the instruction is not its shape.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

std::optional<bool> SopcLowering::route_wave32_mask_compare() {
    // Scalar compare -> SCC (read by s_cselect / s_cbranch_scc). eq/lg are bitwise (sign-agnostic);
    // the ordered compares are signed for i32 (0x02-0x05), unsigned for u32 (0x08-0x0b).
    // Wave32 compilers also compare the physical VCC_LO dword with zero. The mask has no
    // per-invocation integer representation, but EQ/LG and the bounded unsigned forms below
    // reduce exactly to none/any across the guest wave.
    if (allow_wave && b.is_compute && b.wave_size == 32) {
        auto zero = [](const Operand& operand) {
            return operand.kind == OperandKind::InlineInt && operand.value == 0;
        };
        auto mask = [&](const Operand& operand) -> uint32_t {
            if (operand.kind != OperandKind::SGPR &&
                operand.kind != OperandKind::Special) return 0;
            auto found = rs.sreg_bool.find(operand.value);
            return found == rs.sreg_bool.end() ? 0 : found->second;
        };
        const uint32_t first_mask = mask(in.src[0]);
        const uint32_t second_mask = mask(in.src[1]);
        const bool mask_first = first_mask && zero(in.src[1]) &&
            (in.opcode == 0x06 || in.opcode == 0x07 || in.opcode == 0x08 ||
             in.opcode == 0x0b);
        const bool mask_second = second_mask && zero(in.src[0]) &&
            (in.opcode == 0x06 || in.opcode == 0x07 || in.opcode == 0x09 ||
             in.opcode == 0x0a);
        if (mask_first || mask_second) {
            const uint32_t any = b.native_subgroup_size
                ? b.native_wave_any(mask_first ? first_mask : second_mask)
                : b.guest_wave_any(mask_first ? first_mask : second_mask);
            const bool invert = in.opcode == 0x06 ||
                (mask_first ? in.opcode == 0x0b : in.opcode == 0x09);
            rs.scc = invert ? b.logical_not(any) : any;
            return true;
        }
    }
    return std::nullopt;
}

std::optional<bool> SopcLowering::route_b64_mask_compare() {
    //
    // A B64 compare may consume EXEC, VCC, or a saved wave mask. Those values intentionally
    // have no scalar-data representation: one SPIR-V bool represents this invocation's bit.
    // At a wave-uniform fragment site, reduce the per-lane mismatch across the enforced
    // 64-lane subgroup. This is the exact SCC result of s_cmp_eq/lg_u64 and, unlike reading
    // VCC_LO as uint data, also preserves masks built by vector comparisons and saveexec.
    if ((in.opcode == 0x12 || in.opcode == 0x13) && allow_wave && b.is_fragment) {
        auto mask = [&](const Operand& o) -> uint32_t {
            if (o.value == 106 || o.value == 107) return rs.vcc;
            if (o.value == 126 || o.value == 127) return rs.exec;
            if (o.kind == OperandKind::SGPR) {
                auto it = rs.sreg_bool.find(o.value);
                if (it != rs.sreg_bool.end()) return it->second;
            }
            if (o.kind == OperandKind::InlineInt)
                return inline_int_mask_bit(b, o.value);
            return 0;
        };
        const uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
        if (m0 && m1) {
            const uint32_t mismatch = b.bsel(m0, b.logical_not(m1), m1);
            const uint32_t different = b.fragment_wave_any(mismatch);
            rs.scc = in.opcode == 0x12 ? b.logical_not(different) : different;
            return true;
        }
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir
