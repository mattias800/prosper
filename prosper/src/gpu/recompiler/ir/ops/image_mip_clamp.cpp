// One unsigned clamp definition for evaluation and emission. Keep the old levels - 1 expression:
// materialization/admission remains a caller fact, including its proof that the count is present.
// Clamp the selector to the levels the host image actually has.
//
// The half that is certain: an out-of-range `Lod` operand is UNDEFINED in
// SPIR-V, so some clamp is mandatory whatever the hardware does.
//
// The half that is inference: RDNA 2 ISA doc 70648 gives the image
// descriptor BASE_LEVEL/LAST_LEVEL fields as the addressable level range,
// and prosper reads `declared_mip_levels` off exactly that pair -- but the
// guide's MIMG section does not spell out the saturating behaviour for an
// out-of-range IMAGE_LOAD_MIP operand, and no live capture here exercises
// one. Saturating to the last level is the reading consistent with the
// field's purpose; returning zeros would be the other candidate.
// CONFIDENCE: MED -- revisit if a title is measured issuing an
// out-of-range mip.

#include "gpu/recompiler/ir/ops/image_mip_clamp.hpp"

#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"

#include <array>

namespace prosper::gpu::ir {

ImageMipClampBlock build_image_mip_clamp(uint32_t materialized_mip_levels) {
    ImageMipClampBlock out;
    Builder ir(out.block);
    const Value requested = ir.input(Type::U32);
    out.lod = ir.umin(requested, ir.c32(materialized_mip_levels - 1u));
    return out;
}

Known eval_image_mip_clamp(Known requested, uint32_t materialized_mip_levels) {
    const auto built = build_image_mip_clamp(materialized_mip_levels);
    const std::array<Known, 1> inputs{requested};
    return evaluate(built.block, inputs)[built.lod.index];
}

uint32_t emit_image_mip_clamp(SpirvCompute& b, uint32_t requested,
                              uint32_t materialized_mip_levels) {
    const auto built = build_image_mip_clamp(materialized_mip_levels);
    const std::array<uint32_t, 1> inputs{requested};
    return emit_spirv(b, built.block, inputs)[built.lod.index];
}

} // namespace prosper::gpu::ir
