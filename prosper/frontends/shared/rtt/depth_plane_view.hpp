// Which sampled views of a guest depth plane read the value the renderer's retained depth image
// holds, so a consumer may be served that image instead of the plane's guest bytes.
#pragma once

#include "gpu/resources/shader_resources.hpp"

#include <cstdint>

namespace prosper::frontend {

// The renderer never writes rendered depth back to guest memory, so a depth plane's guest bytes
// hold whatever was there before the pass -- usually the clear. A one-component view reads the
// retained D32 image's depth aspect as one of:
//   Float -- a sampled float. FLOAT32 views a Z32 plane; UNORM16 is how GFX10 samples a Z16 plane.
//            Both return the stored depth in [0, 1], the value the depth aspect returns (the D32
//            copy is finer than a guest Z16 plane, never coarser). UE4's volumetric fog reads its
//            16-bit shadow atlas this way; served the guest clear, it found no occluder anywhere
//            and lit every froxel (Kena, KENA_STATUS.md).
//   Bits  -- UINT32: the raw bits of a Z32 plane, copied rather than sampled.
// Anything else is not a depth reading and keeps the consumer's ordinary path.
enum class DepthPlaneRead : uint8_t { None, Float, Bits };

struct DepthPlaneView {
    DepthPlaneRead read = DepthPlaneRead::None;
    // Bytes per texel of the guest plane this view reads: 2 for a Z16 plane, 4 for Z32. 0 = none.
    uint32_t texel_bytes = 0;
};

constexpr DepthPlaneView depth_plane_view(prosper::gpu::DataFormat format, uint32_t components) {
    if ((components ? components : 1u) != 1u) return {};
    switch (format) {
        case prosper::gpu::DataFormat::Float32: return {DepthPlaneRead::Float, 4u};
        case prosper::gpu::DataFormat::Unorm16: return {DepthPlaneRead::Float, 2u};
        case prosper::gpu::DataFormat::Uint32: return {DepthPlaneRead::Bits, 4u};
        default: return {};
    }
}

// Whether a retained plane may serve a view that reads `view_bytes` per texel. `plane_bytes` is
// the guest plane's width as the pass that attached it described it (DB_Z_INFO.FORMAT; 0 = no pass
// described it). The widths must match. Retained planes are never evicted and are preferred over
// colour targets at the same address, so without this a UNORM16 view over a Z32 plane, or an
// R16_UNORM texture later placed at a recycled depth address, would be served depth. An
// undescribed plane is taken as four bytes, the size guest_depth_plane_bytes already assumes:
// FLOAT32 and UINT32 views keep the behaviour that predates this check, and no UNORM16 view is
// admitted on a guess.
constexpr bool depth_plane_admits_view(uint32_t plane_bytes, uint32_t view_bytes) {
    return view_bytes && (plane_bytes ? plane_bytes : 4u) == view_bytes;
}

}   // namespace prosper::frontend
