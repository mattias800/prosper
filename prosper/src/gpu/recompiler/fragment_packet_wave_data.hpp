#pragma once
#include "gpu/recompiler/fragment_resource_packet.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <memory>

namespace prosper::gpu {
// Explicit owned host-dispatch ABI, NOT a guest raster scheduling/entry ABI. One workgroup owns
// one caller-assigned logical64 wave. Code/profile is immutable; no wave value is a cache key.
inline constexpr uint32_t kPacketWaveInputMagic = 0x57494e31u; // WIN1
inline constexpr uint32_t kPacketWaveOutputMagic = 0x574f5531u; // WOU1
inline constexpr uint32_t kPacketWaveTableMagic = 0x57415631u; // WAV1
inline constexpr uint32_t kPacketWaveHeaderWords = 4;
inline constexpr uint32_t kPacketWaveAuthorityMagic = 0x57415431u; // WAT1
inline constexpr uint32_t kPacketWaveAuthorityHeaderWords = 6;
inline constexpr uint32_t kPacketWaveOutputPrefix = 64 * 2;
struct FragmentPacketWavePlacement {
    uint32_t input_base = 0, output_base = 0; // DWORD offsets, not byte addresses
};
struct PacketWaveDataLayout {
    struct Buffer {
        uint32_t pc = 0, offset = 0, words = 0, descriptor = 0;
    };
    struct Parameter {
        uint32_t pc = 0, offset = 0, attribute = 0, channel = 0;
    };
    std::vector<uint32_t> vgprs, sgprs, scalar_offsets, scalar_available_offsets;
    std::vector<Buffer> buffers;
    std::vector<Parameter> parameters;
    std::vector<uint32_t> image_descriptor_offsets, sampler_offsets;
    uint32_t expected_m0 = 0, entry_m0 = 0, entry_m0_available_offset = 0;
    uint32_t input_words = 0, output_words = 0;
    bool entry_m0_available = false;
    // Explicit alternate entry ABI. False preserves the owned WAT1 compiler and wire bytes.
    bool gpu_capacity = false;
};
struct FragmentPacketKernel {
    FragmentResourcePacketProgram program; // cached SOURCE and per-wave status/read-site schema
    PacketWaveDataLayout layout;
    std::vector<uint32_t> guest_code;
    std::vector<Rdna2Inst> instructions; // one immutable original-program inventory
    FragmentPacketQuadTopology topology = FragmentPacketQuadTopology::Unknown;
    FloatTransportConfig transport{};
};
struct FragmentPacketWaveBatch;
// Private validated dispatcher ownership, uploaded readonly at binding2 and retained through
// completion. Binding0 routing/data may be faulted after packing; this plane may NOT be changed.
// It is not guest wave-composition authority or a defense against a hostile host uploader.
class FragmentPacketWaveAuthority {
public:
    const std::vector<uint32_t>& words() const { return words_; }
    bool matches(const std::shared_ptr<const FragmentPacketKernel>&,
                 std::span<const FragmentPacketWavePlacement>, size_t input_words,
                 size_t output_words) const;

private:
    FragmentPacketWaveAuthority(std::shared_ptr<const FragmentPacketKernel> owner,
                                std::vector<uint32_t> words)
        : kernel_(std::move(owner)), words_(std::move(words)) {}
    friend FragmentPacketWaveBatch
        pack_fragment_packet_waves(std::shared_ptr<const FragmentPacketKernel>,
                                   std::span<const FragmentResourcePacket>,
                                   std::span<const FragmentPacketWavePlacement>);
    const std::shared_ptr<const FragmentPacketKernel> kernel_;
    const std::vector<uint32_t> words_;
};
struct FragmentPacketWaveBatch {
    std::shared_ptr<const FragmentPacketKernel> kernel; // retained cached-code owner
    std::shared_ptr<const FragmentPacketWaveAuthority> authority;
    std::vector<uint32_t> input_words, output_words;
    std::vector<FragmentPacketWavePlacement> placements;
    // Complete immutable binding upload owner. Initial domain shares these views across waves;
    // independent per-wave image binding/indexing is NOT inferred from equal dimensions.
    std::vector<FragmentPacketImageRead> images;
    uint64_t device_identity = 0;
    std::string rejection;
};
struct FragmentPacketWaveResult {
    std::vector<std::vector<uint32_t>> exports; // EMPTY if ANY wave fails
    std::vector<std::vector<FragmentPacketArchitecturalLane>> architectural_exports;
    std::string rejection;
    uint32_t wave = UINT32_MAX, lane = UINT32_MAX, pc = UINT32_MAX;
};
FragmentPacketKernel recompile_fragment_packet_kernel(const FragmentResourcePacket& schema,
                                                      RecompileDiagnosticContext diagnostic = {
                                                          RecompileDiagnosticStage::Fragment, 0});
// No compiler call. Every wave must carry the exact original code and genuinely supplied scalar,
// mask, launch and resource facts. Shape mismatch/presence/ownership failures are named refusals.
FragmentPacketWaveBatch
pack_fragment_packet_waves(std::shared_ptr<const FragmentPacketKernel>,
                           std::span<const FragmentResourcePacket>,
                           std::span<const FragmentPacketWavePlacement> placements = {});
FragmentPacketWaveResult decode_fragment_packet_waves(const FragmentPacketWaveBatch&,
                                                      std::span<const uint32_t>,
                                                      bool completion_and_host_availability,
                                                      uint64_t executing_device_identity);
} // namespace prosper::gpu
