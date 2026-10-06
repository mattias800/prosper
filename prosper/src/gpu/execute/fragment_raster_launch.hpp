#pragma once
#include "gpu/pm4/command_processor.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/state/fragment_entry_facts.hpp"
#include "gpu/state/raster_launch_facts.hpp"
#include <memory>

namespace prosper::gpu {
struct DrawItem;
struct OperationRealizationFailure;
struct GraphicsRawSnapshotContext;
struct ShaderCodeAnalysis;
struct RasterQuadInputs;
struct FragmentPacketVgprRequirements;
class OrderedScalarBankReadPoint;
class FragmentScalarBank;

// The actual realizer retains this observation beside its selected modules. It is NOT incoming
// EXEC, a helper/coverage conversion, an LDS allocation, a virtual scheduling permission or a
// backend-completion certificate. Those distinct facts cannot be supplied by a public ready bit.
class FragmentRasterLaunchSource {
public:
    const RasterLaunchFacts& launch() const { return launch_; }
    const FragmentEntryFacts& entry() const { return entry_; }
    const FragmentInterpolationLayout& original_parameters() const { return parameters_; }
    // Privately captured complete original-body proof, never a public empty-module shortcut.
    bool pending_original_has_no_external_effects() const { return pending_original_proved_; }
    bool matches(const RasterQuadInputs&) const;

private:
    friend bool realize_draw_item(const GpuState&, const GpuState::Draw*, uint32_t, uint32_t, bool,
                                  DrawItem&, OperationRealizationFailure*, bool, const char* const*,
                                  const GraphicsRawSnapshotContext*,
                                  std::shared_ptr<const OrderedScalarBankReadPoint>,
                                  std::shared_ptr<const FragmentScalarBank>);
    // Only the actual realizer may associate its selected analysis and modules with this draw.
    // A public inputs flag, a caller-provided analysis or a necessary memoized predicate cannot
    // mint this owner. The immutable observation is retained through normal draw completion.
    static void bind(const GpuState&, const RasterLaunchFacts&,
                     std::shared_ptr<const ShaderCodeAnalysis>, RasterQuadInputs&);
    // Stable immutable representation of an ACTUALLY selected empty stage, not a dummy shader.
    // An independently proved private observation is still mandatory for any pending original PS.
    static std::shared_ptr<const std::vector<uint32_t>> selected_empty_words();
    FragmentRasterLaunchSource(const RasterQuadInputs&, bool pending_original_proved);
    // Draw/completion ownership only. The code/profile cache must copy the required profile and
    // keep its existing weak original-generation ledger; it must not retain this draw owner.
    std::shared_ptr<const std::vector<uint32_t>> raw_, vertex_, geometry_, fragment_;
    std::shared_ptr<const FragmentPacketVgprRequirements> requirements_;
    RasterLaunchFacts launch_{};
    FragmentEntryFacts entry_{};
    FragmentInterpolationLayout parameters_{};
    PixelInputMapping pixel_{};
    PixelSystemInputMapping system_{};
    FloatTransportConfig transport_{};
    FragmentFloatMode float_mode_{};
    FragmentFloatFlags float_flags_{};
    FragmentLaunchRsrc1 rsrc1_{};
    bool has_pixel_ = false, has_system_ = false;
    bool generated_geometry_ = false, owned_wave_pending_ = false;
    bool pending_original_proved_ = false;
};

// Necessary cold original-body/raster facts for a DISTINCT deferred PS carrier. This does not
// mint an observation, bypass the selected source owner, or authorize a backend draw by itself.
bool fragment_raster_pending_original(const GpuState&, const RasterLaunchFacts&,
                                      const std::shared_ptr<const ShaderCodeAnalysis>&,
                                      PixelSystemInputMapping);
}   // namespace prosper::gpu
