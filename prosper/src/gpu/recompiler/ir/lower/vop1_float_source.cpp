// Existing VOP1 float-input stage, mechanically extracted without changing its arithmetic.
#include "gpu/recompiler/ir/lower/vop1_float_source.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <optional>
namespace prosper::gpu::ir {
namespace {
struct SourceContext {
    SpirvCompute& b;
    const Rdna2Inst& in;
    uint32_t& a;
};
struct Vop1FloatSourceLowering : SourceContext {
    std::optional<bool> stage_source_modifiers();
};
std::optional<bool> Vop1FloatSourceLowering::stage_source_modifiers() {
    // SDWA float source modifiers on src0 (abs then neg). v_cvt_f32_f16 applies them after
    // selecting and unpacking the requested half below; applying them to the packed u32 as
    // though it were an f32 changes both the value and the selected sign bit.
    if (in.opcode != 0x0B) {
        if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
        if (in.src_neg[0]) a = b.fbin(Op_FSub, b.uconst(0), a);
    }
    return std::nullopt;
}

} // namespace
void lower_vop1_float_source(SpirvCompute& b, const Rdna2Inst& in, uint32_t& a) {
    Vop1FloatSourceLowering stage{{b, in, a}};
    (void)stage.stage_source_modifiers();
}
} // namespace prosper::gpu::ir
