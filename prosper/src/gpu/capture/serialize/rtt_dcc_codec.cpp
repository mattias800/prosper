// The v79 tail retains a descriptor association, never a host pointer or an inferred clear.
#include "gpu/capture/serialize/rtt_dcc_codec.hpp"
#include "gpu/capture/serialize/capture_codecs.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {
namespace {
bool valid(const GpuCaptureRttDccMetadata& m, std::string& error) {
    if (m == GpuCaptureRttDccMetadata{}) return true;
    if (!m.metadata_addr || !m.metadata_bytes || m.metadata_bytes > max_blob_bytes() ||
        m.metadata_addr > UINT64_MAX - m.metadata_bytes || m.num_components > 4 ||
        (m.num_components ? (!m.guest_width || !m.guest_height)
                          : (m.alpha_is_on_msb || m.guest_width || m.guest_height))) {
        error = "invalid RTT DCC metadata association";
        return false;
    }
    return true;
}
} // namespace
bool write_rtt_dcc_metadata(Writer& w, const std::vector<GpuCaptureRttSeed>& seeds,
                             std::string& error) {
    w.u32(static_cast<uint32_t>(seeds.size()));
    for (const auto& seed : seeds) {
        const auto& m = seed.dcc_metadata;
        if (!valid(m, error)) return false;
        w.u64(m.metadata_addr);
        w.u64(m.metadata_bytes);
        w.u32(m.num_components);
        w.u32(m.alpha_is_on_msb);
        w.u32(m.guest_width);
        w.u32(m.guest_height);
    }
    return true;
}
bool read_rtt_dcc_metadata(Reader& r, std::vector<GpuCaptureRttSeed>& seeds,
                            std::string& error) {
    uint32_t count = 0;
    if (!r.u32(count) || count != seeds.size()) {
        error = "invalid RTT DCC metadata count";
        return false;
    }
    for (auto& seed : seeds) {
        auto& m = seed.dcc_metadata;
        uint32_t alpha = 0;
        if (!r.u64(m.metadata_addr) || !r.u64(m.metadata_bytes) ||
            !r.u32(m.num_components) || !r.u32(alpha) || !r.u32(m.guest_width) ||
            !r.u32(m.guest_height)) return false;
        if (alpha > 1) {
            error = "invalid RTT DCC alpha convention";
            return false;
        }
        m.alpha_is_on_msb = alpha != 0;
        if (!valid(m, error)) return false;
    }
    return true;
}
} // namespace prosper::gpu
