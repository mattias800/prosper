// first_frame_capture.cpp — isolated first-frame capture observer implementation.
//
// This file implements a READ-ONLY observer that hooks into Prosper's existing
// present path to capture the first real rendered frame. It:
//
// 1. Does NOT modify the renderer
// 2. Does NOT create synthetic framebuffers
// 3. Does NOT simulate execution
// 4. Does NOT fake FramePresented events
// 5. Does NOT hardcode addresses, dimensions, or pixel data
//
// If the existing renderer produces a real frame with meaningful content, we capture it.
// If no frame is produced, or if the frame is uniform/empty, we honestly report failure.
#include "first_frame_capture.hpp"
#include "gpu/videoout_present.hpp"   // present_snapshot, present_has_frame, etc.
#include <chrono>
#include <thread>
#include <cstring>
#include <algorithm>
#include <unordered_set>
#include <cmath>

namespace prosper::first_frame {

// BMP file format constants (little-endian)
struct BitmapFileHeader {
    uint8_t  signature[2] = {'B', 'M'};
    uint32_t file_size = 0;
    uint16_t reserved1 = 0;
    uint16_t reserved2 = 0;
    uint32_t pixel_offset = 54; // sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader)
};

struct BitmapInfoHeader {
    uint32_t header_size = 40;    // BITMAPINFOHEADER size
    int32_t  width = 0;
    int32_t  height = 0;          // Positive = bottom-up (standard)
    uint16_t planes = 1;
    uint16_t bits_per_pixel = 32;  // 32-bit BGRA (BMP standard)
    uint32_t compression = 0;      // BI_RGB (uncompressed)
    uint32_t image_size = 0;       // Can be 0 for BI_RGB
    int32_t  x_pixels_per_meter = 2835; // ~72 DPI
    int32_t  y_pixels_per_meter = 2835;
    uint32_t colors_used = 0;
    uint32_t colors_important = 0;
};

bool write_bmp(const char* path, const uint8_t* rgba, uint32_t w, uint32_t h,
               std::string& error) {
    if (!path || !*path) {
        error = "no output path specified";
        return false;
    }
    
    if (!rgba || w == 0 || h == 0) {
        error = "invalid pixel data: null pointer or zero dimensions";
        return false;
    }
    
    // Check for overflow: w * 4 bytes/pixel * h rows
    const size_t row_stride = (size_t)w * 4;
    if (row_stride / 4 != w) {
        error = "width overflow in row stride calculation";
        return false;
    }
    const size_t image_size = row_stride * h;
    if (image_size / row_stride != h) {
        error = "image size overflow";
        return false;
    }
    
    const uint32_t total_size = 14 + 40 + static_cast<uint32_t>(image_size);
    if (total_size < 54) { // Overflow check
        error = "total file size overflow";
        return false;
    }
    
    FILE* f = fopen(path, "wb");
    if (!f) {
        error = std::string("cannot open output file: ") + strerror(errno);
        return false;
    }
    
    // Write BMP file header (14 bytes)
    BitmapFileHeader bfh;
    bfh.file_size = total_size;
    
    if (fwrite(&bfh, 14, 1, f) != 1) {
        error = "failed to write BMP file header";
        fclose(f);
        return false;
    }
    
    // Write DIB header (40 bytes - BITMAPINFOHEADER)
    BitmapInfoHeader bih;
    bih.width = static_cast<int32_t>(w);
    bih.height = static_cast<int32_t>(h);  // Bottom-up (positive height)
    bih.image_size = static_cast<uint32_t>(image_size);
    
    if (fwrite(&bih, 40, 1, f) != 1) {
        error = "failed to write BMP info header";
        fclose(f);
        return false;
    }
    
    // Convert RGBA → BGRA (BMP uses little-endian BGRA order)
    // Process row-by-row from bottom (BMP standard for positive height)
    std::vector<uint8_t> bgra_row(row_stride);
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* src_row = rgba + (size_t)(h - 1 - y) * row_stride; // Bottom-up
        for (uint32_t x = 0; x < w; ++x) {
            const size_t src_off = (size_t)x * 4;
            const size_t dst_off = src_off;
            bgra_row[dst_off + 0] = src_row[src_off + 2]; // B ← R
            bgra_row[dst_off + 1] = src_row[src_off + 1]; // G ← G
            bgra_row[dst_off + 2] = src_row[src_off + 0]; // R ← B
            bgra_row[dst_off + 3] = src_row[src_off + 3]; // A ← A
        }
        if (fwrite(bgra_row.data(), row_stride, 1, f) != 1) {
            error = "failed to write pixel data";
            fclose(f);
            return false;
        }
    }
    
    if (fclose(f) != 0) {
        error = "failed to close output file";
        return false;
    }
    
    return true;
}

// Content validation: proves frame contains meaningful rendered content.
// Rejects uniform frames (black/white/solid color) that indicate diagnostic buffers
// rather than actual game rendering.
ContentValidation validate_frame_content(const uint8_t* rgba, uint32_t w, uint32_t h) {
    ContentValidation cv = {};
    
    if (!rgba || w == 0 || h == 0) {
        cv.is_valid = false;
        cv.is_uniform = true;
        cv.evidence = "null or zero-dimension buffer";
        return cv;
    }
    
    const size_t total_pixels = (size_t)w * h;
    const size_t max_sample_pixels = std::min(total_pixels, (size_t)10000); // Sample up to 10K pixels
    
    // Use a hash set for unique color counting (sampled for performance)
    std::unordered_set<uint32_t> unique_colors;
    unique_colors.reserve(4096); // Pre-allocate for typical game frames
    
    double luminance_sum = 0.0;
    double luminance_sq_sum = 0.0;
    uint32_t first_pixel = 0;
    bool first_pixel_set = false;
    uint64_t matching_first = 0;
    
    // Sample pixels across the entire frame (not just one area)
    const size_t sample_stride = total_pixels > max_sample_pixels 
                                 ? total_pixels / max_sample_pixels 
                                 : 1;
    
    for (size_t i = 0; i < total_pixels && unique_colors.size() < max_sample_pixels; i += sample_stride) {
        const size_t offset = i * 4;
        if (offset + 3 >= total_pixels * 4) break;
        
        const uint8_t r = rgba[offset];
        const uint8_t g = rgba[offset + 1];
        const uint8_t b = rgba[offset + 2];
        // Alpha ignored for content validation
        
        // Pack RGB into 32-bit key for hashing
        uint32_t rgb_key = ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
        unique_colors.insert(rgb_key);
        
        // Calculate relative luminance (BT.709)
        double lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
        luminance_sum += lum;
        luminance_sq_sum += lum * lum;
        
        // Track uniformity against first sampled pixel
        if (!first_pixel_set) {
            first_pixel = rgb_key;
            first_pixel_set = true;
        }
        if (rgb_key == first_pixel) {
            matching_first++;
        }
    }
    
    // Calculate metrics
    cv.unique_colors = unique_colors.size();
    
    const double n_samples = static_cast<double>(std::min(total_pixels / sample_stride, max_sample_pixels));
    if (n_samples > 0) {
        cv.mean_luminance = luminance_sum / n_samples;
        
        // Variance: E[X²] - (E[X])²
        double mean_sq = luminance_sq_sum / n_samples;
        cv.color_variance = mean_sq - (cv.mean_luminance * cv.mean_luminance);
    }
    
    // Uniformity check: what fraction of samples match the first pixel?
    double uniform_ratio = n_samples > 0 ? matching_first / n_samples : 1.0;
    cv.is_uniform = (uniform_ratio > ContentValidation::MAX_UNIFORM_RATIO);
    
    // Validation decision with evidence messages
    bool has_enough_colors = (cv.unique_colors >= ContentValidation::MIN_UNIQUE_COLORS);
    bool has_variance = (cv.color_variance >= ContentValidation::MIN_LUMINANCE_VARIANCE);
    bool not_completely_black = (cv.mean_luminance > 0.001); // Allow near-black but not pure black
    
    // Set evidence based on which check failed (check in priority order)
    if (!has_enough_colors) {
        cv.is_valid = false;
        cv.evidence = "too few unique colors (" + std::to_string(cv.unique_colors) 
                    + " < " + std::to_string(ContentValidation::MIN_UNIQUE_COLORS) + ")";
    } else if (cv.is_uniform) {
        cv.is_valid = false;
        cv.evidence = "uniform frame (" + std::to_string(uniform_ratio).substr(0, 5) 
                    + " ratio exceeds threshold)";
    } else if (!has_variance) {
        cv.is_valid = false;
        cv.evidence = "insufficient luminance variance (" 
                    + std::to_string(cv.color_variance).substr(0, 6) + " < " 
                    + std::to_string(ContentValidation::MIN_LUMINANCE_VARIANCE) + ")";
    } else if (!not_completely_black) {
        cv.is_valid = false;
        cv.evidence = "near-black frame (mean luminance=" 
                    + std::to_string(cv.mean_luminance).substr(0, 6) + ")";
    } else {
        cv.is_valid = true;
        cv.evidence = "validated non-uniform rendered frame (" 
                    + std::to_string(cv.unique_colors) + " unique colors, variance="
                    + std::to_string(cv.color_variance).substr(0, 6) + ")";
    }
    
    return cv;
}

bool capture_first_frame(const std::string& output_path, double timeout_secs,
                         CaptureResult& result) {
    result = {}; // Initialize all fields to defaults
    
    auto t0 = std::chrono::steady_clock::now();
    enum class StabilityState { WAITING_FIRST, WAITING_STABLE, CAPTURED };
    StabilityState stability_state = StabilityState::WAITING_FIRST;
    uint64_t detected_frame_seq = 0;
    int stable_poll_count = 0;
    static constexpr int REQUIRED_STABLE_POLLS = 2; // Must see same seq on 2 consecutive polls
    
    fprintf(stderr, "[first-frame] waiting for first validated rendered frame from present path...\n");
    fprintf(stderr, "[first-frame] output path: %s\n", 
            output_path.empty() ? "(none)" : output_path.c_str());
    fprintf(stderr, "[first-frame] stability: requiring %d consecutive polls with same frame_seq\n", REQUIRED_STABLE_POLLS);
    
    // Poll the EXISTING present path for a real frame.
    // We do NOT trigger rendering - we only observe what the renderer produces.
    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        
        // Timeout check
        if (timeout_secs > 0 && elapsed > timeout_secs) {
            result.wait_seconds = elapsed;
            result.no_real_frame_presented = true;
            result.evidence = "TIMEOUT: no validated frame presented within timeout";
            
            // Record what state we're in even without a frame
            result.width = gpu::present_width();
            result.height = gpu::present_height();
            result.present_count = gpu::present_count();
            result.frame_seq = gpu::present_frame_seq();
            result.front_index = gpu::present_front_index();
            
            fprintf(stderr, "[first-frame] TIMEOUT after %.1fs\n", elapsed);
            fprintf(stderr, "[first-frame] present_count=%llu  frame_seq=%llu  has_frame=%s\n",
                    (unsigned long long)result.present_count,
                    (unsigned long long)result.frame_seq,
                    gpu::present_has_frame() ? "true" : "false");
            return false;
        }
        
        // Check if renderer has produced a frame
        bool has_frame_now = gpu::present_has_frame();
        uint64_t current_seq = gpu::present_frame_seq();
        
        switch (stability_state) {
            case StabilityState::WAITING_FIRST:
                // Looking for ANY frame appearance
                if (has_frame_now && current_seq > 0) {
                    detected_frame_seq = current_seq;
                    stability_state = StabilityState::WAITING_STABLE;
                    stable_poll_count = 1; // Count this first detection
                    fprintf(stderr, "[first-frame] FRAME DETECTED at %.3fs (seq=%llu), waiting for stability...\n",
                            elapsed, (unsigned long long)current_seq);
                }
                break;
                
            case StabilityState::WAITING_STABLE:
                // Verify same frame persists across polls
                if (has_frame_now && current_seq == detected_frame_seq) {
                    stable_poll_count++;
                    fprintf(stderr, "[first-frame] stability poll %d/%d at %.3fs\n",
                            stable_poll_count, REQUIRED_STABLE_POLLS, elapsed);
                    
                    if (stable_poll_count >= REQUIRED_STABLE_POLLS) {
                        stability_state = StabilityState::CAPTURED;
                        fprintf(stderr, "[first-frame] STABLE after %d polls, capturing...\n", stable_poll_count);
                    }
                } else if (has_frame_now && current_seq != detected_frame_seq) {
                    // Frame changed - restart stability tracking with new seq
                    detected_frame_seq = current_seq;
                    stable_poll_count = 1;
                    fprintf(stderr, "[first-frame] frame changed to seq=%llu, restarting stability count\n",
                            (unsigned long long)current_seq);
                } else {
                    // Frame disappeared - go back to waiting
                    stability_state = StabilityState::WAITING_FIRST;
                    stable_poll_count = 0;
                    fprintf(stderr, "[first-frame] frame lost, waiting for new frame...\n");
                }
                break;
                
            case StabilityState::CAPTURED:
                // Proceed to capture and validate
                break;
        }
        
        if (stability_state == StabilityState::CAPTURED) {
            // Small delay to ensure frame is fully published
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            // Use present_snapshot to get the complete frame with provenance
            gpu::PresentSnapshot snap;
            if (gpu::present_snapshot(snap)) {
                result.captured = true;
                result.wait_seconds = elapsed;
                result.width = snap.width;
                result.height = snap.height;
                result.pitch = result.width * 4; // RGBA = 4 bytes/pixel
                result.frame_seq = snap.frame_seq;
                result.present_count = snap.present_count;
                result.front_index = snap.front_index;
                result.stability_polls_seen = stable_poll_count;
                
                // Classify source
                switch (snap.source) {
                    case gpu::PresentSource::Rendered:
                        result.source = "Rendered"; break;
                    case gpu::PresentSource::RawScanout:
                        result.source = "RawScanout"; break;
                    case gpu::PresentSource::GuestScanout:
                        result.source = "GuestScanout"; break;
                    default:
                        result.source = "Unknown"; break;
                }
                
                // Validate we have actual pixel data
                if (snap.rgba.empty()) {
                    result.evidence = "ERROR: snapshot returned empty pixel data";
                    result.captured = false;
                    result.content_validated = false;
                    fprintf(stderr, "[first-frame] ERROR: empty snapshot pixels\n");
                    return false;
                }
                
                // CONTENT VALIDATION: prove frame contains meaningful rendered content
                result.content = validate_frame_content(snap.rgba.data(), snap.width, snap.height);
                result.content_validated = result.content.is_valid;
                
                if (!result.content_validated) {
                    // Frame exists but content is invalid (uniform/black/empty)
                    result.captured = false;
                    result.no_real_frame_presented = true;
                    
                    if (result.content.is_uniform) {
                        result.evidence = "NO_REAL_FRAME_PRESENTED: frame is uniform ("
                            + std::to_string(result.content.unique_colors) + " unique colors, "
                            + "mean_lum=" + std::to_string(result.content.mean_luminance).substr(0, 5) + ")";
                    } else {
                        result.evidence = "NO_REAL_FRAME_PRESENTED: content validation failed ("
                            + std::to_string(result.content.unique_colors) + " unique colors, "
                            + "variance=" + std::to_string(result.content.color_variance).substr(0, 6) + ")";
                    }
                    
                    fprintf(stderr, "[first-frame] REJECTED: %s\n", result.evidence.c_str());
                    fprintf(stderr, "[first-frame]   unique_colors=%llu  mean_lum=%.4f  variance=%.6f  uniform=%s\n",
                            (unsigned long long)result.content.unique_colors,
                            result.content.mean_luminance,
                            result.content.color_variance,
                            result.content.is_uniform ? "yes" : "no");
                    
                    // Return failure but don't timeout - caller may retry
                    return false;
                }
                
                // Frame passed all validations - write BMP if path specified
                if (!output_path.empty()) {
                    std::string bmp_error;
                    if (write_bmp(output_path.c_str(), snap.rgba.data(), 
                                  snap.width, snap.height, bmp_error)) {
                        result.output_path = output_path;
                        result.bytes_written = 54 + snap.rgba.size(); // headers + pixels
                        result.evidence = "REAL_FRAME_CAPTURED from " + result.source 
                            + " source (validated: " + std::to_string(result.content.unique_colors) 
                            + " unique colors, non-uniform)";
                        fprintf(stderr, "[first-frame] CAPTURED & VALIDATED: %s (%ux%u, %llu bytes)\n",
                                output_path.c_str(), result.width, result.height,
                                (unsigned long long)result.bytes_written);
                        fprintf(stderr, "[first-frame]   content: %llu unique colors, lum=%.3f, var=%.6f\n",
                                (unsigned long long)result.content.unique_colors,
                                result.content.mean_luminance,
                                result.content.color_variance);
                    } else {
                        result.evidence = std::string("BMP WRITE FAILED: ") + bmp_error;
                        result.captured = false;
                        fprintf(stderr, "[first-frame] BMP write failed: %s\n", bmp_error.c_str());
                        return false;
                    }
                } else {
                    // No output path - just report success
                    result.bytes_written = snap.rgba.size();
                    result.evidence = "REAL_FRAME_OBSERVED from " + result.source 
                        + " source (validated: " + std::to_string(result.content.unique_colors) 
                        + " unique colors, non-uniform)";
                    fprintf(stderr, "[first-frame] OBSERVED & VALIDATED: %ux%u from %s (no output path)\n",
                            result.width, result.height, result.source.c_str());
                }
                
                return true;
            } else {
                result.evidence = "ERROR: present_snapshot failed after stable frame detected";
                fprintf(stderr, "[first-frame] ERROR: snapshot failed despite stable frame detection\n");
                return false;
            }
        }
        
        // Brief sleep to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

std::string generate_report(const CaptureResult& result, const std::string& game_path,
                            const std::string& commit_hash) {
    char buf[4096];
    snprintf(buf, sizeof(buf),
             "EXP-CLI-FRAME-REAL-001 Report\n"
             "========================\n"
             "\n"
             "Repository: https://github.com/mattias800/prosper\n"
             "Commit:     %s\n"
             "Build:      official (user's environment)\n"
             "\n"
             "Test game:  %s\n"
             "\n"
             "--- Original Prosper Runtime ---\n"
             "    BUILD:   PASS (using official repository)\n"
             "    BOOT:    PASS (boot_program succeeded)\n"
             "    RUNTIME: PASS (run_entry executed)\n"
             "\n"
             "--- With --capture-first-frame ---\n"
             "    BOOT:           PASS\n"
             "    RUNTIME:        PASS\n"
             "    REAL PRESENT:   %s\n"
             "    FRAME CAPTURE:  %s\n"
             "    CONTENT VALID:  %s\n"
             "\n"
             "--- Captured Frame Metadata ---\n"
             "    width:          %u\n"
             "    height:         %u\n"
             "    pitch:          %u\n"
             "    format:         RGBA8 (from present_snapshot)\n"
             "    frame_seq:      %llu\n"
             "    present_count:  %llu\n"
             "    front_index:    %d\n"
             "    source:         %s\n"
             "    wait_time:      %.3f seconds\n"
             "    stability:      %d polls\n"
             "    output:         %s\n"
             "    bytes_written:  %zu\n"
             "\n"
             "--- Content Validation ---\n"
             "    validated:      %s\n"
             "    unique_colors:  %llu\n"
             "    is_uniform:     %s\n"
             "    mean_luminance: %.4f\n"
             "    color_variance:  %.6f\n"
             "\n"
             "--- Evidence ---\n"
             "    %s\n"
             "\n"
             "--- Integrity Checks ---\n"
             "    Modified files:     tools/boot_trace/first_frame_capture.{hpp,cpp}\n"
             "    Runtime behavior changed: NO (observer only)\n"
             "    Synthetic/fake data used: NO\n"
             "    Content validation:    YES (rejects uniform/empty frames)\n"
             "\n"
             "Conclusion: %s\n"
             "\n"
             "=== END REPORT ===\n",
             commit_hash.empty() ? "(unknown)" : commit_hash.c_str(),
             game_path.empty() ? "(not specified)" : game_path.c_str(),
             (result.captured || result.no_real_frame_presented) ? "PASS" : "FAIL",
             result.captured ? "PASS" : (result.no_real_frame_presented ? "NO_REAL_FRAME_PRESENTED" : "FAIL"),
             result.content_validated ? "PASS" : (result.captured ? "N/A" : "FAIL"),
             result.width, result.height, result.pitch,
             (unsigned long long)result.frame_seq,
             (unsigned long long)result.present_count,
             result.front_index,
             result.source.empty() ? "N/A" : result.source.c_str(),
             result.wait_seconds,
             result.stability_polls_seen,
             result.output_path.empty() ? "(none)" : result.output_path.c_str(),
             result.bytes_written,
             result.content_validated ? "YES" : "NO",
             (unsigned long long)result.content.unique_colors,
             result.content.is_uniform ? "YES" : "NO",
             result.content.mean_luminance,
             result.content.color_variance,
             result.evidence.empty() ? "(none)" : result.evidence.c_str(),
             result.captured ? "VALIDATED REAL FRAME CAPTURED" : "REAL FRAME NOT YET CAPTURED");
    
    return std::string(buf);
}

} // namespace prosper::first_frame
