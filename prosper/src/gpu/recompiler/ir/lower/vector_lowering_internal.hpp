// vector_lowering_internal.hpp — the vector-ALU lowering classes shared by ir/lower/vop*.cpp.
//
// A VALU instruction is a pipeline rather than a route chain: DPP routes that own the whole
// instruction, then SRC0's DPP transform, SDWA/abs/neg source modifiers, the operation itself,
// SDWA destination placement, OMOD/CLAMP, and finally the EXEC-predicated write of VDST. The
// stages were moved verbatim out of emit_alu; each returns what emit_alu returned when it ends the
// instruction early, or nullopt to continue. Their order is the old code's order and is
// load-bearing. Only the operation stage goes through the IR (ir/ops/vector_alu.hpp), and only for
// the opcodes that have a definition.
#pragma once

#include <cstdint>
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vector_alu_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_dpp_row_shr.hpp"
#include "gpu/recompiler/rdna2_dpp_row_ror8.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"

#include <optional>

namespace prosper::gpu::ir {

class Vop2Lowering : LoweringContext {
public:
    explicit Vop2Lowering(const LoweringContext& context)
        : LoweringContext(context), vreg(rs.vreg), vcc(rs.vcc) {}
    bool lower();

private:
    // emit_alu's locals of the same names.
    decltype(RegState::vreg)& vreg;
    uint32_t& vcc;
    uint32_t a = 0, c = 0, old_d = 0;
    uint32_t dpp_active = 0;
    uint32_t row_result = 0;
    bool packed_f16 = false;
    bool integer_sdwa = false;

    std::optional<bool> route_dpp_row_shr_in_place();
    std::optional<bool> route_vadd_dpp_partial_row();
    std::optional<bool> stage_dpp_source();
    std::optional<bool> stage_source_modifiers();
    std::optional<bool> compute(uint32_t& d);
    bool compute_float(uint32_t& d);
    bool compute_f16(uint32_t& d);
    uint32_t half_source(uint32_t raw, const Operand& operand, uint8_t sel, int k);
    std::optional<bool> stage_sdwa_destination(uint32_t& d);
    std::optional<bool> stage_output_modifiers(uint32_t& d);
};

} // namespace prosper::gpu::ir
