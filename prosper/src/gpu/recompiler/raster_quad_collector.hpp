#pragma once
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/fragment_packet_vgpr_requirements.hpp"
#include "gpu/state/raster_launch_facts.hpp"
#include "gpu/state/fragment_entry_facts.hpp"
#include "gpu/execute/fragment_packet_preparation.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace prosper::gpu {
// An owned real draw's input contract, not initialized guest registers or a logical Wave64.
// All selected modules are pinned; raw_code owns the same analysis version used by the PS compiler.
struct RasterQuadInputs {
    std::shared_ptr<const std::vector<uint32_t>> source_vs, source_gs, source_fs, raw_code;
    std::shared_ptr<const FragmentPacketVgprRequirements> vgpr_requirements;
    PixelInputMapping pixel_inputs{};
    PixelSystemInputMapping system_inputs{};
    FragmentInterpolationLayout interpolation{};
    RasterLaunchFacts launch{};
    bool has_pixel_inputs = false, has_system_inputs = false;
    bool raw_matches_producing_source = false;
    // Typed deferred owned-wave PS input collection: no native producing PS module exists yet.
    // This grants neither initialized guest registers nor export/attachment authority.
    bool owned_wave_pending = false;
    bool generated_interpolation_geometry = false;
    FloatTransportConfig float_transport{};
    FragmentEntryFacts entry{};
    FragmentFloatMode float_mode{};
    FragmentFloatFlags float_flags{};
    FragmentLaunchRsrc1 launch_rsrc1{};
    FragmentPacketResources ps_resources{};
};

enum class RasterQuadFieldKind : uint8_t { Interpolant, Parameter, SystemInterpolation };
struct RasterQuadField {
    RasterQuadFieldKind kind = RasterQuadFieldKind::Interpolant;
    uint32_t index = 0, selector = 0, words = 4;
    // Words are host-observed input values, not an all-path definition proof for guest registers.
    bool guest_initialization_proved = false;
};
inline constexpr uint32_t kRasterQuadMagic = 0x51554144u;
inline constexpr uint32_t kRasterQuadBufferHeaderWords = 4;
// Each quad has four lanes in Vulkan's specified quad-index order. Fixed lane words:
// HelperInvocation, coverage-available, SampleMask[0], FrontFacing, PrimitiveId, FragCoord.xyzw.
// Coverage is unavailable for helpers: their observed mask is never interpreted as guest coverage.
inline constexpr uint32_t kRasterQuadLaneFixedWords = 9;
struct RasterQuadCollector {
    std::vector<RasterQuadField> fields;
    uint32_t lane_words = 0, record_words = 0, max_quads = 0;
    std::string rejection;
};
std::vector<uint32_t> build_raster_quad_collector(const RasterQuadInputs& inputs,
    uint32_t max_quads, RasterQuadCollector& contract);
// A completed device write still needs this bounded, transactional wire/provenance check. On
// refusal no records are returned. Same-origin scopes with disjoint nonhelper masks retain their
// separate raw records; overlapping masks remain an unproved supported-domain refusal. Neither
// coordinate equality nor append order establishes guest invocation or wave membership.
std::string decode_raster_quad_records(const RasterQuadCollector& collector,
    const uint32_t* words, size_t word_count, uint32_t primitive_count,
    std::vector<std::vector<uint32_t>>& quads);

// Whole completed collection, or a named refusal with no records. Private scratch effects are not
// guest attachment effects. A consumer must still choose a guest wave policy and initialize every
// required guest input explicitly; these are host-realized inputs, not PS5 helper/packing authority.
struct RasterQuadResult {
    std::shared_ptr<const FragmentPacketPreparation> packet_preparation;
    bool attempted = false, complete = false;
    uint64_t source_submit = 0, draw_index = 0, command_order = 0;
    uint32_t width = 0, height = 0, lane_words = 0;
    RasterLaunchFacts launch{};
    // Established by this producer's actual pipeline, not inferred guest launch defaults. Helpers'
    // coverage remains unavailable; coverage never establishes post-shader/depth export eligibility.
    bool host_raster_domain_available = false;
    uint32_t host_sample_count = 0, host_sample_index = 0, host_layer = 0, host_view_index = 0;
    uint32_t host_instance_count = 0, host_first_instance = 0;
    bool guest_export_eligibility_available = false;
    std::vector<RasterQuadField> fields;
    std::vector<std::vector<uint32_t>> quads;
    std::shared_ptr<const std::vector<uint32_t>> vertex_source, geometry_source, collector_source;
    struct BufferSnapshot { uint32_t binding = 0; std::vector<uint32_t> words; };
    std::vector<BufferSnapshot> vertex_buffers;
    std::string rejection;
};
struct RasterQuadCollection {
    std::shared_ptr<const RasterQuadInputs> inputs;
    uint32_t max_quads = 1024;
    mutable std::mutex mutex;
    RasterQuadResult result;
};
} // namespace prosper::gpu
