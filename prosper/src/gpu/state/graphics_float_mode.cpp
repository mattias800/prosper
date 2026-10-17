// GC10.3 has RSRC1 banks for VS, merged ES/GS and merged LS/HS. Register facts are
// independently published in AMD's gc_10_3_0_offset.h and gc_10_3_0_sh_mask.h.
#include "gpu/state/graphics_float_mode.hpp"
#include "gpu/recompiler/graphics_float_mode.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/pm4/vgt_shader_stages.hpp"
#include <cstdint>
namespace prosper::gpu {
GraphicsFloatMode graphics_float_mode(const GpuState& state, GraphicsHardwareStage stage) {
    namespace P = prosper::agc::Pm4;
    uint32_t reg = P::SPI_SHADER_PGM_RSRC1_VS;
    if (stage == GraphicsHardwareStage::Geometry) reg = P::SPI_SHADER_PGM_RSRC1_GS;
    if (stage == GraphicsHardwareStage::Hull) reg = P::SPI_SHADER_PGM_RSRC1_HS;
    const auto found = state.sh.find(reg);
    if (found == state.sh.end()) return {};
    auto result = graphics_float_mode_from_rsrc1(found->second);
    // FP16_OVFL is HS[30], VS/GS[31]; FLOAT_MODE and IEEE happen to share positions.
    const auto shift = stage == GraphicsHardwareStage::Hull ? 30u : 31u;
    result.fp16_overflow = {true, (found->second & (1u << shift)) != 0};
    return result;
}
GraphicsFloatMode vertex_graphics_float_mode(const GpuState& state) {
    const auto found = state.cx.find(prosper::agc::Pm4::VGT_SHADER_STAGES_EN);
    if (found == state.cx.end()) return {};
    const VgtShaderStages stages{found->second};
    // NGG VS-only and merged ES/GS both execute under RSRC1_GS. A fetch-prolog
    // continuation is part of that same hardware program and shares its launch mode.
    return graphics_float_mode(state, stages.primgen_enabled() || stages.gs_enabled()
                                          ? GraphicsHardwareStage::Geometry
                                          : GraphicsHardwareStage::Vertex);
}
} // namespace prosper::gpu
