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
enum class DepthPlaneView : uint8_t { None, Float, Bits };

constexpr DepthPlaneView depth_plane_view(prosper::gpu::DataFormat format, uint32_t components) {
    if ((components ? components : 1u) != 1u) return DepthPlaneView::None;
    switch (format) {
        case prosper::gpu::DataFormat::Float32:
        case prosper::gpu::DataFormat::Unorm16: return DepthPlaneView::Float;
        case prosper::gpu::DataFormat::Uint32: return DepthPlaneView::Bits;
        default: return DepthPlaneView::None;
    }
}

} // namespace prosper::frontend
