#pragma once

#include <cstdint>

// Test-side mirror of the sceVideodec2 guest structs, shared by every test that drives the
// Videodec2 HLE handlers the way a title does (test_videodec2_decode, test_videodec2_mf).
namespace prosper::test::vdec {

// The guest-facing structs, mirrored from the handler's own definitions. static_assert on each size
// so a silent divergence makes this file fail to compile rather than exercise a different ABI.
struct VdecConfig {
    uint64_t size; uint32_t resource, codec, profile, max_level;
    int32_t max_width, max_height, max_dpb; uint32_t input_depth;
    uint64_t compute_queue, affinity; int32_t priority;
    uint8_t optimize, check_memory, reserved0, reserved1; uint64_t extra;
};
struct VdecInput { uint64_t size, data, data_size, pts, dts, attached; };
struct VdecFrame { uint64_t size, data, data_size; uint8_t accepted, pad[7]; };
struct VdecOutput {
    uint64_t size; uint8_t valid, error, pictures, discarded;
    uint32_t codec, width, pitch, height; uint64_t frame, frame_size;
    uint32_t format, pitch_bytes;
};
static_assert(sizeof(VdecConfig) == 72, "test VdecConfig must mirror the HLE layout");
static_assert(sizeof(VdecInput) == 48, "test VdecInput must mirror the HLE layout");
static_assert(sizeof(VdecFrame) == 32, "test VdecFrame must mirror the HLE layout");
static_assert(sizeof(VdecOutput) == 56, "test VdecOutput must mirror the HLE layout");

// --- libSceVdecsw ABI structures (#3669) --------------------------------------------------------
struct VdecswConfig { VdecConfig base; uint64_t extra2; };
struct VdecswFrame { uint64_t size, data, data_size; };
struct VdecswSyncInput { uint64_t size, rsvd0, rsvd1; };
struct VdecswOutput {
    uint64_t size; uint8_t valid, error, pictures, pictureCount;
    uint32_t codec, width, pitch, height; uint64_t frame, frame_size;
    uint32_t format, pitch_bytes;
};
static_assert(sizeof(VdecswConfig) == 80, "test VdecswConfig must mirror the HLE layout");
static_assert(sizeof(VdecswFrame) == 24, "test VdecswFrame must mirror the HLE layout");
static_assert(sizeof(VdecswSyncInput) == 24, "test VdecswSyncInput must mirror the HLE layout");
static_assert(sizeof(VdecswOutput) == 56, "test VdecswOutput must mirror the HLE layout");

} // namespace prosper::test::vdec
