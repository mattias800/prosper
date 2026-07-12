#include "capture_manifest.hpp"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace prosper::screenshot {

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
    if (observation.source == CaptureSource::Rendered) rendered_samples_++;
    max_stale_seconds_ = std::max(max_stale_seconds_, result.stale_seconds);
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
    return source == CaptureSource::Rendered ? "composited" : "raw_scanout";
}

std::string manifest_run_json(const CaptureRunConfig& c) {
    std::ostringstream line;
    line << "{\"type\":\"run\",\"schema\":1"
         << ",\"title\":\"" << json_escape(c.title) << "\""
         << ",\"timestamp\":\"" << json_escape(c.timestamp) << "\""
         << ",\"output_dir\":\"" << json_escape(c.output_dir) << "\""
         << ",\"input_route\":\"" << json_escape(c.input_route) << "\""
         << ",\"capture_mode\":\"" << (c.time_mode ? "wall_seconds" : "rendered_frames") << "\""
         << ",\"seconds\":" << std::fixed << std::setprecision(6) << c.seconds
         << ",\"every\":" << c.every << ",\"requested\":" << c.requested
         << ",\"warmup_ms\":" << c.warmup_ms
         << ",\"warmup_submits\":" << c.warmup_submits
         << ",\"render_every\":\"" << json_escape(c.render_every) << "\""
         << ",\"render_scale\":\"" << json_escape(c.render_scale) << "\""
         << ",\"render_target_dim\":\"" << json_escape(c.render_target_dim) << "\""
         << ",\"render_resource_dim\":\"" << json_escape(c.render_resource_dim) << "\""
         << ",\"assertions\":{\"min_distinct_frames\":" << c.min_distinct_frames
         << ",\"max_stale_seconds\":" << std::fixed << std::setprecision(6) << c.max_stale_seconds
         << ",\"require_composited_frame\":" << (c.require_composited_frame ? "true" : "false")
         << ",\"min_present_count\":" << c.min_present_count
         << ",\"min_frame_seq\":" << c.min_frame_seq
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
         << ",\"source_advanced\":" << (c.source_advanced ? "true" : "false")
         << ",\"pixel_identical\":" << (c.pixel_identical ? "true" : "false")
         << ",\"stale_seconds\":" << std::fixed << std::setprecision(6) << c.stale_seconds
         << ",\"input_route\":\"" << json_escape(input_route) << "\"}";
    return line.str();
}

std::string manifest_summary_json(int saved, int requested, bool timed_out,
                                  const CaptureTracker& tracker, int exit_code) {
    std::ostringstream line;
    line << "{\"type\":\"summary\",\"schema\":1,\"saved\":" << saved
         << ",\"requested\":" << requested
         << ",\"timed_out\":" << (timed_out ? "true" : "false")
         << ",\"distinct_source_frames\":" << tracker.distinct_source_frames()
         << ",\"rendered_samples\":" << tracker.rendered_samples()
         << ",\"max_stale_seconds\":" << std::fixed << std::setprecision(6)
         << tracker.max_stale_seconds()
         << ",\"max_frame_seq\":" << tracker.max_frame_seq()
         << ",\"max_present_count\":" << tracker.max_present_count()
         << ",\"exit_code\":" << exit_code << "}";
    return line.str();
}

} // namespace prosper::screenshot
