#pragma once
#include "gpu/recompiler/fragment_packet_wave_data.hpp"
#include "gpu/recompiler/raster_quad_collector.hpp"
#include <array>

namespace prosper::gpu {
// GPU-produced counts are not the CPU-owned WAT1 wave list. This separate domain retains an
// immutable capacity/placement certificate, while collection supplies the actual wave count.
// Nothing in this certificate initializes a guest input or establishes raster scheduling.
inline constexpr uint32_t kFragmentDrawAuthorityMagic = 0x57415432u; // WAT2
inline constexpr uint32_t kFragmentDrawInputMagic = 0x57415632u;     // WAV2
inline constexpr uint32_t kFragmentDrawHeaderWords = 12;
inline constexpr uint32_t kFragmentDrawAuthorityWords = 10;
inline constexpr uint32_t kFragmentDrawIndirectWord = 5;
inline constexpr uint32_t kFragmentDrawCommitHeaderWords = 4;
inline constexpr uint32_t kFragmentDrawPixelProbeLimit = 32;
enum class FragmentDrawFailure : uint32_t {
    None = 0,
    CollectionHeader = 1,
    CollectionOverflow = 2,
    CollectionRecord = 3,
    HelperEntryUnavailable = 4,
    WorkerCompletion = 5,
    GuestRuntime = 6,
    GuestExport = 7,
    DuplicatePixel = 8,
    PixelIndexCapacity = 9,
    OccupiedExportUnavailable = 10,
};

class FragmentDrawCapacity {
public:
    const std::shared_ptr<const FragmentPacketKernel>& kernel() const { return kernel_; }
    const std::array<uint32_t, kFragmentDrawAuthorityWords>& authority() const { return words_; }
    const RasterQuadCollector& collector() const { return collector_; }
    const std::vector<FragmentPacketExportSite>& export_sites() const { return export_sites_; }
    bool matches_collector(const RasterQuadCollector&) const;
    uint32_t max_quads() const { return words_[6]; }
    uint32_t max_waves() const { return words_[1]; }
    uint32_t input_words() const { return words_[2]; }
    uint32_t output_words() const { return words_[3]; }
    uint32_t input_span() const { return words_[4]; }
    uint32_t output_span() const { return words_[5]; }
    uint32_t collector_words() const { return words_[7]; }
    uint32_t pixel_slots() const { return words_[8]; }
    uint32_t commit_words() const { return words_[9]; }
    FragmentPacketWavePlacement placement(uint32_t wave) const {
        if (wave >= max_waves()) return {UINT32_MAX, UINT32_MAX};
        return {kFragmentDrawHeaderWords + wave * input_span(), wave * output_span()};
    }

private:
    FragmentDrawCapacity(std::shared_ptr<const FragmentPacketKernel> kernel,
                         std::array<uint32_t, kFragmentDrawAuthorityWords> words,
                         RasterQuadCollector collector)
        : kernel_(std::move(kernel)), words_(words), collector_(std::move(collector)),
          export_sites_(kernel_->program.packet.export_sites) {}
    friend std::shared_ptr<const FragmentDrawCapacity>
    fragment_draw_capacity(std::shared_ptr<const FragmentPacketKernel>, const RasterQuadCollector&,
                           std::string&);
    const std::shared_ptr<const FragmentPacketKernel> kernel_;
    const std::array<uint32_t, kFragmentDrawAuthorityWords> words_;
    const RasterQuadCollector collector_;
    // Copied original-site inventory belongs to the immutable capacity/code owner, not GPU data.
    const std::vector<FragmentPacketExportSite> export_sites_;
};

// Cold code/profile verification only. LegacyRaw is never attachment-authoritative, including
// a forged 14-word output shape. Original ISA sites and the emitted policy marker must agree.
bool fragment_draw_architectural_exports_match(const FragmentPacketKernel&);

std::shared_ptr<const FragmentDrawCapacity>
fragment_draw_capacity(std::shared_ptr<const FragmentPacketKernel>, const RasterQuadCollector&,
                       std::string& rejection);
// Cached original PS, with the distinct uniformly guarded GPU-capacity entry. Default WAT1
// compilation and its public owned-wave pack/decode API remain unchanged.
FragmentPacketKernel recompile_fragment_packet_capacity_kernel(
    const FragmentResourcePacket&,
    RecompileDiagnosticContext = {RecompileDiagnosticStage::Fragment, 0});
}   // namespace prosper::gpu
