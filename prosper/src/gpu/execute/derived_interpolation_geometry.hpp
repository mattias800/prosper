// Immutable generated GS residence follows actual producing code generations, not draw entries.
#pragma once
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <memory>

namespace prosper::gpu {
struct ShaderCodeAnalysis;
struct DerivedInterpolationGeometryStats {
    uint64_t compile_calls = 0, cache_hits = 0, retired = 0;
};
DerivedInterpolationGeometryStats derived_interpolation_geometry_stats();

// The module is device-independent SOURCE. Enabled-device/launch and selected-module checks remain
// per draw; this cache never returns a producing permission or retains either source generation.
// Residence is process-wide: realizer workers produce the owner, and the render thread's helper
// plan/collector caches key on it, so one profile must yield one generation across workers.
std::shared_ptr<const std::vector<uint32_t>> acquire_derived_interpolation_geometry(
    const std::shared_ptr<const ShaderCodeAnalysis>& fragment,
    const std::shared_ptr<const std::vector<uint32_t>>& vertex, const FragmentInterpolationLayout&,
    bool capture_vertex_position, bool synthesize_rect, FloatTransportConfig);
} // namespace prosper::gpu
