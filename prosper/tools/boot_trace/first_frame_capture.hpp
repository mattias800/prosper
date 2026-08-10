// first_frame_capture.hpp — isolated first-frame capture observer for Prosper CLI.
//
// Hooks into the EXISTING present path (videoout_present.cpp) to capture the first
// real rendered frame when --capture-first-frame is enabled. This is a READ-ONLY observer:
// it does not modify, replace, or simulate any rendering. If no real frame is produced,
// it reports NO_REAL_FRAME_PRESENTED with evidence.
//
// Usage: Integrated into boot_trace via --capture-first-frame [output_path]
#pragma once
#include <string>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <chrono>

namespace prosper::first_frame {

// Content validation result - proves frame contains meaningful rendered content
struct ContentValidation {
    bool is_valid = false;          // Does frame pass content validation?
    bool is_uniform = false;        // Is frame a single solid color (black/white/etc)?
    uint64_t unique_colors = 0;     // Number of distinct RGBA pixel values
    double mean_luminance = 0.0;    // Average brightness (0.0 = black, 1.0 = white)
    double color_variance = 0.0;     // Pixel-to-pixel variance (0.0 = uniform)
    std::string evidence;           // Human-readable explanation of validation result
    
    // Validation thresholds (configurable for testing)
    static constexpr uint64_t MIN_UNIQUE_COLORS = 16;      // Reject < 16 unique colors
    static constexpr double MAX_UNIFORM_RATIO = 0.999;     // Reject > 99.9% same color
    static constexpr double MIN_LUMINANCE_VARIANCE = 0.001; // Minimum variance threshold
};

// Frame capture result - records what actually happened (evidence, not fabrication)
struct CaptureResult {
    bool captured = false;           // Was a real frame captured?
    bool no_real_frame_presented = false;  // Explicit: no frame was available
    bool content_validated = false; // Did frame pass content validation?
    
    // Frame metadata (all from actual runtime, zero if no frame)
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pitch = 0;             // width * 4 (RGBA)
    uint64_t frame_seq = 0;         // From present_frame_seq()
    uint64_t present_count = 0;     // From present_count()
    int front_index = -1;           // Buffer index
    
    // Source classification (from PresentSource enum)
    std::string source;             // "Rendered", "RawScanout", "GuestScanout", or "None"
    
    // Content validation metrics
    ContentValidation content;      // Proves frame has meaningful rendered content
    
    // Output info
    std::string output_path;        // Where BMP was written (empty if not written)
    size_t bytes_written = 0;
    
    // Timing
    double wait_seconds = 0;        // How long we waited for the frame
    
    // Stability tracking
    int stability_polls_seen = 0;   // How many polls showed this frame seq
    
    // Evidence fields
    std::string evidence;           // Human-readable evidence string
};

// Write RGBA pixel data as uncompressed BMP (simplest format, widely compatible).
// Returns true on success. Does not fabricate pixel data.
bool write_bmp(const char* path, const uint8_t* rgba, uint32_t w, uint32_t h,
               std::string& error);

// Wait for and capture the first real frame from the existing present path.
// This OBSERVES ONLY - it does not trigger, modify, or fake rendering.
// 
// Parameters:
//   output_path   - Where to write the BMP (empty = no file, just report)
//   timeout_secs  - Maximum time to wait (0 = no timeout)
//   result        - Output: complete capture evidence
//
// Returns true if capture succeeded (real frame obtained and written).
// Returns false if no frame was available (result.no_real_frame_presented = true).
bool capture_first_frame(const std::string& output_path, double timeout_secs,
                         CaptureResult& result);

// Generate EXP-CLI-FRAME-REAL-001 compliant report string.
std::string generate_report(const CaptureResult& result, const std::string& game_path,
                            const std::string& commit_hash);

} // namespace prosper::first_frame
