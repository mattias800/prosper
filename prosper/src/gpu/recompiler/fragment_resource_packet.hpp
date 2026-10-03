#pragma once
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <span>

namespace prosper::gpu {

// This is an explicitly owned execution input, NOT a conversion of RasterQuadInputs or physical
// USER_DATA observations into guest registers. Every readpoint names the original decoded PC.
struct FragmentPacketBufferRead {
    uint32_t pc = UINT32_MAX;
    std::array<uint32_t, 4> descriptor{};
    std::vector<uint32_t> words; // complete immutable byte-addressed V# backing, DWORD aligned
};
struct FragmentPacketParameter {
    uint32_t primitive = 0, attribute = 0, channel = 0;
    uint32_t p0 = 0, p10 = 0, p20 = 0;
};
struct FragmentPacketParameterCache {
    bool available = false;
    uint32_t m0 = 0; // actual expected {0,new_prim_mask[15:1],lds_param_offset[15:0]}
    std::array<uint32_t, 16> quad_primitive{};
    std::vector<FragmentPacketParameter> parameters;
    // Optional genuinely supplied guest M0 entry. Otherwise an original guest writer is required.
    bool entry_m0_available = false;
    uint32_t entry_m0 = 0;
};
struct FragmentPacketImageMip {
    uint32_t width = 0, height = 0;
    std::vector<std::array<uint32_t, 4>> texels; // owned R32G32B32A32_SFLOAT words
};
struct FragmentPacketImageRead {
    uint32_t pc = UINT32_MAX;
    std::array<uint32_t, 8> descriptor{};
    std::array<uint32_t, 4> sampler{};
    std::vector<FragmentPacketImageMip> mips;
};
struct FragmentPacketDeviceContract {
    // Supplied by the executing device owner AFTER feature enablement. A compiler's host device
    // name, ambient FloatControls verdict, or copied host bytes cannot establish these facts.
    uint64_t device_identity = 0;
    bool shader_int64_enabled = false;
    bool rgba32_sfloat_sampled = false;
};
struct FragmentResourcePacket {
    FragmentInvocationPacket invocation;
    FragmentLaunchRsrc1 launch_rsrc1{}; // complete ORIGINAL launch word, not a host mode verdict
    FragmentPacketParameterCache parameter_cache;
    std::vector<FragmentPacketBufferRead> buffers;
    std::vector<FragmentPacketImageRead> images;
    FragmentPacketDeviceContract device;
};

enum class FragmentPacketRuntimeFailure : uint32_t {
    None = 0, DescriptorMismatch = 1, M0Mismatch = 2, NonFinite = 3,
    FiniteOverflow = 4, InterpolationNotExact = 5,
    SampleCoordinateDomain = 6, SampleLodDomain = 7,
    UndefinedVgpr = 9, // 8 reserved for the coordinated special-F32 service variant
};
// New resource variants append [magic, first-failing-PC, reason] for EACH logical worker after
// the old raw EXP records. No worker exits early. Sticky failure is not a substituted guest value.
inline constexpr uint32_t kFragmentResourceStatusMagic = 0x52504b31u; // RPK1
inline constexpr uint32_t kFragmentResourceStatusWords = 3;
struct FragmentResourcePacketProgram {
    FragmentPacketProgram packet;
    std::vector<FragmentPacketImageRead> images; // binding 16+i, complete immutable upload plan
    FragmentPacketDeviceContract device;
    uint32_t status_offset = 0;
    FragmentLaunchRsrc1 launch_rsrc1{};
    FragmentFloatMode float_mode{};
    FragmentFloatFlags float_flags{};
    std::vector<uint32_t> runtime_failure_pcs; // exact original service PCs, never caller guesses
};
struct FragmentResourcePacketResult {
    std::vector<uint32_t> exports; // EMPTY on ANY malformed/failing worker or readback failure
    std::string rejection;
    uint32_t lane = UINT32_MAX, pc = UINT32_MAX;
    uint32_t vgpr = UINT32_MAX;
    FragmentPacketRuntimeFailure failure = FragmentPacketRuntimeFailure::None;
};

FragmentResourcePacketProgram recompile_fragment_resource_packet(
    const FragmentResourcePacket& packet,
    RecompileDiagnosticContext diagnostic = {RecompileDiagnosticStage::Fragment, 0});
// Production transactional consumer. Only a complete, successfully synchronized and invalidated
// readback from the SAME enabled device can publish exports. It never consumes partial statuses.
FragmentResourcePacketResult decode_fragment_resource_packet(
    const FragmentResourcePacketProgram& program, std::span<const uint32_t> readback,
    bool completion_and_host_availability, uint64_t executing_device_identity);

} // namespace prosper::gpu
