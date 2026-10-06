#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace prosper::gpu {
// The reference model for the device composer, not a live-launch certificate or a CPU shipping
// assembly path. Its inputs name normalized genuine quads; deriving workitem-valid from raster
// observations and admitting this virtual scheduling policy are separate draw-bound proofs.
// No append order, helper flag, public Boolean or zero storage grants those proofs.
struct FragmentQuadCompositionInput {
    uint32_t record = 0, primitive = 0;
    int32_t x = 0, y = 0; // even 2x2 origin, including genuine negative edge-helper coordinates
    uint32_t sample = 0, layer = 0, view = 0;
    uint8_t backing = 0, live = 0, export_candidates = 0;
};
struct FragmentQuadCompositionProfile {
    // Original SC_SHADER_CONTROL[6:5]: 0=no region break, 1=8, 2=16, 3=32 pixels. These constrain
    // the candidate grouping; they do not prove its within-region order matches AMD hardware.
    uint32_t wave_break = 0;
};
struct FragmentQuadComposedWave {
    std::array<uint32_t, 16> records{};   // UINT32_MAX in unused quads, never a padding helper
    std::array<uint32_t, 16> primitives{};
    uint32_t quad_count = 0;
    uint64_t backing = 0, live = 0, export_candidates = 0;
    uint16_t primitive_transitions = 0;   // Q1..Q15; Q0 is implicitly the first primitive
};
struct FragmentQuadComposition {
    std::vector<FragmentQuadComposedWave> waves;   // EMPTY on any malformed original input
    std::string rejection;
};
// Canonical (view,layer,sample,region,primitive,y,x) ordering is independent of collector append
// order. Never split a genuine quad. Spill after 16 quads and before an overlapping pixel scope;
// spill creates another wave, not a dropped quad or a shader-count admission limit. The accepted
// split-scope normalization still runs BEFORE this component; duplicate normalized keys refuse.
FragmentQuadComposition
    fragment_quad_composition_reference(std::span<const FragmentQuadCompositionInput>,
                                        FragmentQuadCompositionProfile);

// Pure field packing for the actual owned virtual parameter allocation. The mandatory post-user
// SGPR's BC flag is NOT M0[31]. Unknown BC observation yields absence, not a convenient zero.
// The caller must separately own the parameter offset and derive BC from the genuine wave/raster
// contract. Neither this byte packer nor a packed value initializes an actual guest register.
std::optional<uint32_t> fragment_quad_system_word(const FragmentQuadComposedWave&,
                                                  uint16_t parameter_offset,
                                                  std::optional<bool> bc_optimize);
}   // namespace prosper::gpu
