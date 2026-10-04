#pragma once
#include "gpu/state/raster_launch_facts.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct GpuState;
struct RenderState;
struct RasterQuadInputs;
class FragmentScalarBank;
class OrderedScalarBankReadPoint;
class NativeGraphicsStageCompilation;
// Immutable original-PS/native-geometry association, not a native FS module/reflection or live
// read permission. Only the actual execution producer can seal it; lower-level consumers check
// the association without depending on HLE, executor implementation, or a supplied Boolean.
class OriginalFragmentDrawProducer {
    friend std::shared_ptr<const OriginalFragmentDrawProducer>
    seal_original_fragment_draw_producer(const OrderedScalarBankReadPoint&, const GpuState&,
                                         uint64_t,
                                         std::shared_ptr<const NativeGraphicsStageCompilation>,
                                         std::shared_ptr<const FragmentScalarBank>);
    const std::shared_ptr<const NativeGraphicsStageCompilation> vertex_;
    const std::shared_ptr<const FragmentScalarBank> bank_;
    const RasterLaunchFacts launch_;
    const FragmentFloatMode float_mode_;
    const FragmentFloatFlags float_flags_;
    const FragmentLaunchRsrc1 launch_rsrc1_;
    OriginalFragmentDrawProducer(std::shared_ptr<const NativeGraphicsStageCompilation>,
                                 std::shared_ptr<const FragmentScalarBank>, const RenderState&);

public:
    bool matches(const RasterQuadInputs&) const;
    uint64_t read_point_identity() const;
    bool matches_modules(const std::shared_ptr<const std::vector<uint32_t>>& vertex,
                         std::span<const uint32_t> geometry,
                         std::span<const uint32_t> fragment) const;
};
} // namespace prosper::gpu
