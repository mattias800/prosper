#pragma once
#include "gpu/execute/gpu_execute.hpp"

namespace prosper::gpu {
// This is actual selected native-module lineage, not same-content/address cache authority.
// Only the normal checked compiler can mint it after selecting its genuine module while the
// exact ordered source is live. Keeping it afterward owns that historical selection, not a
// permission to inspect current guest bytes. It is per draw, never stored in a resident cache.
class NativeGraphicsStageCompilation {
    friend SharedShaderWords recompile_graphics_shader_cached_shared(
        ShaderProgramStage, const uint32_t*, size_t, const ShaderResourceTable*,
        const PixelInputMapping*, const PixelSystemInputMapping*, uint64_t*, bool, uint32_t, bool,
        const SharedShaderAnalysis&, FragmentFloatMode, FloatTransportConfig, FragmentFloatFlags,
        FragmentLaunchRsrc1, RefusedShaderSource*, const CheckedGraphicsSource*,
        std::shared_ptr<const NativeGraphicsStageCompilation>*, FragmentExportFormats);
    const GraphicsReadSource source_;
    const SharedShaderWords module_;
    const ShaderProgramStage stage_;
    const uint64_t read_point_identity_;
    NativeGraphicsStageCompilation(GraphicsReadSource source, SharedShaderWords module,
                                   ShaderProgramStage stage, uint64_t identity)
        : source_(std::move(source)), module_(std::move(module)), stage_(stage),
          read_point_identity_(identity) {}

public:
    const GraphicsReadSource& source() const { return source_; }
    const SharedShaderWords& module() const { return module_; }
    ShaderProgramStage stage() const { return stage_; }
    uint64_t read_point_identity() const { return read_point_identity_; }
    bool matches(const GraphicsReadSource&, const SharedShaderWords&) const;
};
} // namespace prosper::gpu
