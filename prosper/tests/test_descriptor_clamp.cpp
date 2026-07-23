// GPU descriptor-bound INVARIANT guard.
//
// The entire GPU guest-input memory-safety argument (verified by a systematic 16-hunt sweep of the
// command processor, PM4 decode, executor, compute-dispatch, RDNA2->SPIR-V recompiler, and tile/detile
// swizzle kernels — overnight 2026-07-23) rests on ONE load-bearing fact: `decode_image_descriptor`
// clamps a guest T#'s width/height to <= 16384 and depth to <= 8192, and `decode_buffer_descriptor`
// clamps size_bytes to 32 bits. Downstream code multiplies these into host allocation / copy / detile
// bounds in uint64_t and rejects anything over UINT32_MAX / kMaxComputeImageBytes; that rejection is
// SOUND only because the dimensions can never exceed the masked maxima.
//
// These decoders were untested for the max case. A future refactor that widened a field mask (e.g.
// width to 15 bits) or changed a `+1` bias would silently break the <=16384 invariant and turn every
// "already bounded upstream" rule-out into a real integer-overflow OOB — with NO test catching it.
// This drift guard pins the exact ceilings (constructed from all-ones / max-field descriptor blobs) so
// such a change fails loudly here first. Pure decode; no guest memory or GPU state involved.

#include "../src/gpu/agc_shader_layout.hpp"

#include <cstdint>
#include <cstdio>

using namespace prosper::gpu;

namespace {
int g_fail = 0;
void check(bool ok, const char* msg) {
    std::printf("  %s %s\n", ok ? "[ok]  " : "[FAIL]", msg);
    if (!ok) g_fail++;
}
}  // namespace

int main() {
    std::printf("== test_descriptor_clamp (GPU descriptor bound invariant) ==\n");

    // 1. An all-ones T# drives every image field to its maximum. width/height are 14-bit fields with a
    //    +1 bias, so the ceiling is exactly 0x3FFF + 1 == 16384 (never 16385, never a wrap); format is
    //    a 9-bit field (<= 511). These are the exact bounds gpu_executor.cpp / live_compute.cpp assume.
    {
        uint32_t t[8];
        for (uint32_t& w : t) w = 0xFFFFFFFFu;
        const DecodedImageDescriptor d = decode_image_descriptor(t);
        check(d.width == 16384u, "all-ones T#: width == 16384 (14-bit field + 1 ceiling)");
        check(d.height == 16384u, "all-ones T#: height == 16384 (14-bit field + 1 ceiling)");
        check(d.width <= 16384u && d.height <= 16384u,
              "all-ones T#: within the downstream <= 16384 bound");
        check(d.format <= 511u, "all-ones T#: format masked to 9 bits (<= 511)");
    }

    // 2. A 3D SRV (type 10, non-UAV) with the maximum 13-bit depth field: depth == 0x1FFF + 1 == 8192.
    {
        uint32_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        t[3] = 10u << 28;   // Type == 10 (3D)
        t[4] = 0x1FFFu;     // depth_or_last = max 13-bit value
        // t[5] bit0 == 0 -> a 3D SRV (not a UAV view) -> depth = depth_or_last + 1
        const DecodedImageDescriptor d = decode_image_descriptor(t);
        check(d.depth == 8192u, "3D SRV max: depth == 8192 (13-bit field + 1 ceiling)");
        check(d.depth <= 8192u, "3D SRV max: within the <= 8192 bound");
    }

    // 3. A V# with the maximum stride (14-bit) and num_records (full 32-bit): num_records * stride is
    //    computed in uint64_t and clamped to 0xFFFFFFFF, so the 32-bit product can never wrap to a small
    //    value that would slip under the caller's `size_bytes > 0x10000000` plausibility guard.
    {
        uint32_t v[4] = {0, 0, 0, 0};
        v[1] = 0x3FFFu << 16;   // stride = max 14-bit (0x3FFF)
        v[2] = 0xFFFFFFFFu;     // num_records = max 32-bit
        const DecodedBufferDescriptor d = decode_buffer_descriptor(v);
        check(d.stride == 0x3FFFu, "all-ones V#: stride is a 14-bit field (0x3FFF)");
        check(d.size_bytes == 0xFFFFFFFFu,
              "V# max stride*records: size_bytes clamped to 0xFFFFFFFF (uint64 no-wrap)");
    }

    std::printf(g_fail ? "FAILED (%d)\n" : "PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
