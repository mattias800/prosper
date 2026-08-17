#include "capture_manifest.hpp"

#include "build_revision.hpp"   // revision this binary was compiled from

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace prosper::screenshot {

namespace {
// One publisher (the guest thread), one reader (the sampling thread). `g_guest_state` is stored
// last with release ordering and loaded first with acquire ordering, which is what makes the fault
// fields and the detail buffer safe to read without a lock.
std::atomic<uint32_t> g_guest_state{0};   // 0 running, 1 returned, 2 faulted
std::atomic<int> g_guest_kind{0};
std::atomic<uint64_t> g_guest_fault_rip{0};
std::atomic<uint64_t> g_guest_fault_addr{0};
// `trap_detail()` builds its text in a 256-byte buffer, so this holds every detail prosper can
// currently produce; snprintf truncates rather than overruns if that ever grows.
char g_guest_detail[256]{};
} // namespace

const char* guest_run_state_name(GuestRunState state) {
    switch (state) {
        case GuestRunState::Returned: return "returned";
        case GuestRunState::Faulted:  return "faulted";
        case GuestRunState::Running:  break;
    }
    return "running";
}

void publish_guest_outcome(int kind, uint64_t fault_rip, uint64_t fault_addr, const char* detail) {
    std::snprintf(g_guest_detail, sizeof g_guest_detail, "%s", detail ? detail : "");
    g_guest_kind.store(kind, std::memory_order_relaxed);
    g_guest_fault_rip.store(fault_rip, std::memory_order_relaxed);
    g_guest_fault_addr.store(fault_addr, std::memory_order_relaxed);
    g_guest_state.store(kind == 0 ? 1u : 2u, std::memory_order_release);
}

GuestOutcome read_guest_outcome() {
    GuestOutcome outcome;
    const uint32_t state = g_guest_state.load(std::memory_order_acquire);
    if (state == 0) return outcome;
    outcome.state = state == 1 ? GuestRunState::Returned : GuestRunState::Faulted;
    outcome.kind = g_guest_kind.load(std::memory_order_relaxed);
    outcome.fault_rip = g_guest_fault_rip.load(std::memory_order_relaxed);
    outcome.fault_addr = g_guest_fault_addr.load(std::memory_order_relaxed);
    outcome.detail.assign(g_guest_detail, strnlen(g_guest_detail, sizeof g_guest_detail));
    return outcome;
}

RunVerdict decide_run_verdict(bool assertions_failed, const GuestOutcome& guest,
                              bool allow_guest_fault) {
    if (guest.state == GuestRunState::Faulted && !allow_guest_fault) return {"GUEST-FAULT", 1};
    if (assertions_failed) return {"FAILED", 1};
    if (guest.state == GuestRunState::Faulted) return {"GUEST-FAULT-ALLOWED", 0};
    return {"ok", 0};
}

void normalize_capture_rgba(CaptureSource source, std::vector<uint8_t>& pixels) {
    // A republished guest scanout is normalized like a composited frame and for the same reason:
    // both reach the desktop through the OPAQUE swapchain, so neither one's alpha attenuates the
    // visible RGB. Only RawScanout — read straight out of guest memory with no renderer publication
    // behind it — stays byte-for-byte guest evidence.
    if (source == CaptureSource::RawScanout) return;
    for (size_t offset = 3; offset < pixels.size(); offset += 4)
        pixels[offset] = 255;
}

PixelContentMetrics measure_pixel_content_rgba(const std::vector<uint8_t>& pixels) {
    std::unordered_set<uint32_t> colors;
    colors.reserve(std::min<size_t>(pixels.size() / 4, 65536));
    uint64_t nonblack = 0;
    for (size_t i = 0; i + 3 < pixels.size(); i += 4) {
        const uint32_t alpha = pixels[i + 3];
        const uint32_t red = (static_cast<uint32_t>(pixels[i]) * alpha + 127) / 255;
        const uint32_t green = (static_cast<uint32_t>(pixels[i + 1]) * alpha + 127) / 255;
        const uint32_t blue = (static_cast<uint32_t>(pixels[i + 2]) * alpha + 127) / 255;
        const uint32_t rgb = (red << 16) | (green << 8) | blue;
        colors.insert(rgb);
        nonblack += rgb != 0;
    }
    return {static_cast<uint32_t>(colors.size()), nonblack};
}

namespace {
uint8_t cell_luma(const std::vector<uint8_t>& pixels, uint32_t width, uint32_t height,
                  uint32_t cell_x, uint32_t cell_y, uint32_t grid_width, uint32_t grid_height) {
    const uint32_t x0 = cell_x * width / grid_width;
    const uint32_t x1 = std::max(x0 + 1, (cell_x + 1) * width / grid_width);
    const uint32_t y0 = cell_y * height / grid_height;
    const uint32_t y1 = std::max(y0 + 1, (cell_y + 1) * height / grid_height);
    uint64_t sum = 0;
    uint64_t count = 0;
    for (uint32_t y = y0; y < std::min(y1, height); ++y) {
        for (uint32_t x = x0; x < std::min(x1, width); ++x) {
            const size_t offset = (static_cast<size_t>(y) * width + x) * 4;
            // Integer Rec. 601 luma after compositing the presented RGBA over black. Some
            // renderer failures preserve stale RGB under zero alpha, which is not visible output.
            const uint32_t luma = 299u * pixels[offset] + 587u * pixels[offset + 1] +
                                  114u * pixels[offset + 2];
            sum += (luma * pixels[offset + 3] + 127) / 255;
            ++count;
        }
    }
    return count ? static_cast<uint8_t>((sum / count + 500) / 1000) : 0;
}
} // namespace

PerceptualHashes perceptual_hashes_rgba(const std::vector<uint8_t>& pixels,
                                        uint32_t width, uint32_t height) {
    PerceptualHashes result;
    if (!width || !height || pixels.size() < static_cast<size_t>(width) * height * 4)
        return result;

    uint8_t average_cells[64]{};
    uint32_t sum = 0;
    for (uint32_t y = 0; y < 8; ++y) {
        for (uint32_t x = 0; x < 8; ++x) {
            const uint8_t value = cell_luma(pixels, width, height, x, y, 8, 8);
            average_cells[y * 8 + x] = value;
            sum += value;
        }
    }
    const uint32_t mean = sum / 64;
    for (uint32_t i = 0; i < 64; ++i)
        if (average_cells[i] > mean) result.average |= uint64_t{1} << i;

    for (uint32_t y = 0; y < 8; ++y) {
        uint8_t row[9]{};
        for (uint32_t x = 0; x < 9; ++x)
            row[x] = cell_luma(pixels, width, height, x, y, 9, 8);
        for (uint32_t x = 0; x < 8; ++x)
            if (row[x] > row[x + 1]) result.difference |= uint64_t{1} << (y * 8 + x);
    }
    return result;
}

std::array<uint8_t, kPerceptualLumaCells>
perceptual_luma16x9_rgba(const std::vector<uint8_t>& pixels,
                         uint32_t width, uint32_t height) {
    std::array<uint8_t, kPerceptualLumaCells> result{};
    if (!width || !height || pixels.size() < static_cast<size_t>(width) * height * 4)
        return result;
    for (uint32_t y = 0; y < 9; ++y)
        for (uint32_t x = 0; x < 16; ++x)
            result[y * 16 + x] = cell_luma(pixels, width, height, x, y, 16, 9);
    return result;
}

CaptureClassification CaptureTracker::observe(const CaptureObservation& observation,
                                               const std::vector<uint8_t>& pixels) {
    CaptureClassification result;
    if (have_previous_) {
        result.source_advanced = observation.source != previous_.source ||
                                 observation.source_seq != previous_.source_seq;
        result.pixel_identical = observation.width == previous_.width &&
                                 observation.height == previous_.height &&
                                 pixels == previous_pixels_;
        if (!result.source_advanced)
            result.stale_seconds = std::max(0.0, observation.elapsed_seconds - last_advance_seconds_);
    }
    if (!have_previous_ || result.source_advanced) {
        distinct_source_frames_++;
        last_advance_seconds_ = observation.elapsed_seconds;
    }
    if (!have_previous_ || !result.pixel_identical) {
        pixel_distinct_frames_++;
        last_pixel_change_seconds_ = observation.elapsed_seconds;
    } else {
        result.pixel_stale_seconds =
            std::max(0.0, observation.elapsed_seconds - last_pixel_change_seconds_);
    }
    // Deliberately Rendered ONLY. `--require-composited-frame` is enforced off this counter, and its
    // contract is "prosper composited at least one of these frames" — a republished guest scanout
    // is precisely the case it must keep rejecting.
    if (observation.source == CaptureSource::Rendered) rendered_samples_++;
    if (observation.source == CaptureSource::GuestScanout) guest_scanout_samples_++;
    max_stale_seconds_ = std::max(max_stale_seconds_, result.stale_seconds);
    max_pixel_stale_seconds_ =
        std::max(max_pixel_stale_seconds_, result.pixel_stale_seconds);
    max_frame_seq_ = std::max(max_frame_seq_, observation.frame_seq);
    max_present_count_ = std::max(max_present_count_, observation.present_count);
    previous_ = observation;
    previous_pixels_ = pixels;
    have_previous_ = true;
    return result;
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const unsigned char c : value) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char escaped[7];
                std::snprintf(escaped, sizeof escaped, "\\u%04x", static_cast<unsigned>(c));
                out += escaped;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

const char* capture_source_name(CaptureSource source) {
    switch (source) {
        case CaptureSource::Rendered:     return "composited";
        case CaptureSource::GuestScanout: return "guest_scanout";
        case CaptureSource::RawScanout:   break;
    }
    return "raw_scanout";
}

std::string manifest_run_json(const CaptureRunConfig& c) {
    std::ostringstream line;
    line << "{\"type\":\"run\",\"schema\":1"
         << ",\"title\":\"" << json_escape(c.title) << "\""
         // The revision this BINARY was compiled from, not the revision of the checkout it is run
         // from. Those differ whenever a lane checks out new work and measures without rebuilding,
         // and nothing else in the artifact can tell them apart: the source tree looks current, the
         // run succeeds, and the result is a confident measurement of the wrong build (instrument
         // trap 81). Recording it here makes every manifest self-describing, including archived ones
         // nobody can re-run. `tools/revision/check_build_revision.py` is the active check.
         << ",\"build_revision\":\"" << json_escape(prosper::embedded_build_revision()) << "\""
         << ",\"timestamp\":\"" << json_escape(c.timestamp) << "\""
         << ",\"output_dir\":\"" << json_escape(c.output_dir) << "\""
         << ",\"input_route\":\"" << json_escape(c.input_route) << "\""
         << ",\"capture_mode\":\"" << (c.time_mode ? "wall_seconds" : "rendered_frames") << "\""
         << ",\"seconds\":" << std::fixed << std::setprecision(6) << c.seconds
         << ",\"every\":" << c.every << ",\"requested\":" << c.requested
         << ",\"warmup_ms\":" << c.warmup_ms
         << ",\"warmup_submits\":" << c.warmup_submits
         << ",\"render_every\":\"" << json_escape(c.render_every) << "\""
         << ",\"render_every_for_ms\":\"" << json_escape(c.render_every_for_ms) << "\""
         << ",\"render_scale\":\"" << json_escape(c.render_scale) << "\""
         << ",\"render_target_dim\":\"" << json_escape(c.render_target_dim) << "\""
         << ",\"render_resource_dim\":\"" << json_escape(c.render_resource_dim) << "\""
         << ",\"assertions\":{\"min_distinct_frames\":" << c.min_distinct_frames
         << ",\"max_stale_seconds\":" << std::fixed << std::setprecision(6) << c.max_stale_seconds
         << ",\"min_pixel_distinct_frames\":" << c.min_pixel_distinct_frames
         << ",\"max_pixel_stale_seconds\":" << std::fixed << std::setprecision(6)
         << c.max_pixel_stale_seconds
         << ",\"require_composited_frame\":" << (c.require_composited_frame ? "true" : "false")
         << ",\"min_present_count\":" << c.min_present_count
         << ",\"min_frame_seq\":" << c.min_frame_seq
         << ",\"allow_guest_fault\":" << (c.allow_guest_fault ? "true" : "false")
         << ",\"required_crc32\":";
    if (c.required_crc32_set)
        line << "\"" << std::hex << std::setw(8) << std::setfill('0') << c.required_crc32
             << std::dec << "\"";
    else
        line << "null";
    line << "}}";
    return line.str();
}

std::string manifest_sample_json(int index, const std::string& png_path,
                                 const CaptureObservation& o,
                                 const CaptureClassification& c,
                                 const std::string& input_route) {
    std::ostringstream line;
    line << "{\"type\":\"sample\",\"schema\":1,\"index\":" << index
         << ",\"png\":\"" << json_escape(png_path) << "\""
         << ",\"elapsed_seconds\":" << std::fixed << std::setprecision(6) << o.elapsed_seconds
         << ",\"source\":\"" << capture_source_name(o.source) << "\""
         << ",\"source_seq\":" << o.source_seq
         << ",\"frame_seq\":" << o.frame_seq
         << ",\"present_count\":" << o.present_count
         << ",\"front_index\":" << o.front_index
         << ",\"width\":" << o.width << ",\"height\":" << o.height
         << ",\"pixel_crc32\":\"" << std::hex << std::setw(8) << std::setfill('0')
         << o.pixel_crc32 << std::dec << "\""
         << ",\"distinct_rgb_colors\":" << o.distinct_rgb_colors
         << ",\"nonblack_rgb_pixels\":" << o.nonblack_rgb_pixels
         << ",\"average_hash\":\"" << std::hex << std::setw(16) << std::setfill('0')
         << o.average_hash << "\""
         << ",\"difference_hash\":\"" << std::setw(16) << o.difference_hash << "\""
         << ",\"luma16x9\":\"";
    for (const uint8_t value : o.luma16x9)
        line << std::setw(2) << static_cast<unsigned>(value);
    line << std::dec << "\""
         << ",\"source_advanced\":" << (c.source_advanced ? "true" : "false")
         << ",\"pixel_identical\":" << (c.pixel_identical ? "true" : "false")
         << ",\"stale_seconds\":" << std::fixed << std::setprecision(6) << c.stale_seconds
         << ",\"pixel_stale_seconds\":" << std::fixed << std::setprecision(6)
         << c.pixel_stale_seconds
         << ",\"input_route\":\"" << json_escape(input_route) << "\"}";
    return line.str();
}

std::string manifest_summary_json(int saved, int requested, bool timed_out,
                                  const CaptureTracker& tracker, const RunVerdict& verdict,
                                  const GuestOutcome& guest, bool allow_guest_fault) {
    std::ostringstream line;
    line << "{\"type\":\"summary\",\"schema\":1,\"saved\":" << saved
         << ",\"requested\":" << requested
         << ",\"timed_out\":" << (timed_out ? "true" : "false")
         << ",\"distinct_source_frames\":" << tracker.distinct_source_frames()
         << ",\"pixel_distinct_frames\":" << tracker.pixel_distinct_frames()
         << ",\"rendered_samples\":" << tracker.rendered_samples()
         << ",\"guest_scanout_samples\":" << tracker.guest_scanout_samples()
         << ",\"max_stale_seconds\":" << std::fixed << std::setprecision(6)
         << tracker.max_stale_seconds()
         << ",\"max_pixel_stale_seconds\":" << std::fixed << std::setprecision(6)
         << tracker.max_pixel_stale_seconds()
         << ",\"max_frame_seq\":" << tracker.max_frame_seq()
         << ",\"max_present_count\":" << tracker.max_present_count()
         // The guest's own terminal state, so a batch consumer can filter runs whose guest died
         // without re-reading the run log. Addresses are hex strings, not JSON numbers: a 64-bit
         // guest address does not survive a double-typed JSON parser.
         << ",\"guest_state\":\"" << guest_run_state_name(guest.state) << "\""
         << ",\"guest_kind\":" << guest.kind
         << ",\"guest_detail\":\"" << json_escape(guest.detail) << "\"";
    if (guest.state == GuestRunState::Faulted) {
        char address[32];
        std::snprintf(address, sizeof address, "0x%llx",
                      static_cast<unsigned long long>(guest.fault_rip));
        line << ",\"guest_fault_rip\":\"" << address << "\"";
        std::snprintf(address, sizeof address, "0x%llx",
                      static_cast<unsigned long long>(guest.fault_addr));
        line << ",\"guest_fault_addr\":\"" << address << "\"";
    } else {
        line << ",\"guest_fault_rip\":null,\"guest_fault_addr\":null";
    }
    line << ",\"allow_guest_fault\":" << (allow_guest_fault ? "true" : "false")
         << ",\"status\":\"" << verdict.status << "\""
         << ",\"exit_code\":" << verdict.exit_code << "}";
    return line.str();
}

} // namespace prosper::screenshot
