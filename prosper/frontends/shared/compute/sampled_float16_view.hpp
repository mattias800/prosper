// Whether live compute samples a guest FP16 texture in its native format (R16F, RG16F, RGBA16F)
// or through the historical RGBA8 UNORM conversion.
#pragma once

#include <cstdint>

namespace prosper::frontend {

struct SampledFloat16Source {
    uint32_t components = 4;            // the T#'s channel count
    bool renderer_owned = false;        // the live renderer holds this texture's pixels
    bool renderer_narrow_match = false; // ...as R16F for one channel or RG16F for two
    bool imported = false;              // bound in place from the renderer's own image
    bool volume = false;                // a 3D texture
};

// The RGBA8 conversion clamps every channel to [0, 1] and keeps 8 bits of it. It stays for
// ordinary guest-backed 2D RGBA16F only because native RGBA16F sampling was measured 7x slower in
// Astro Bot's full-resolution composite on RADV. That is a cost decision, not a correctness one:
// a guest-backed RGBA16F holding HDR colour (> 1) or signed data is still clamped, a known
// correctness loss tracked as #4738. Renderer imports and 3D RGBA16F
// sample natively, because narrowing a volume discarded its HDR range.
//
// One- and two-channel FP16 is data, not colour: a signed depth-of-field circle of confusion, a
// velocity, a luminance. Converted, a negative value becomes 0. UE4's Diaphragm DOF keeps its
// per-tile foreground CoC (negative) in RG16F tiles; through the conversion every tile read "no
// blur", the gather passes wrote zero and the foreground stayed sharp (Kena, KENA_STATUS.md).
// RG16F is no wider than RGBA8, so the RGBA16F cost does not apply. A renderer-owned narrow
// texture samples natively only when the renderer holds it in that same narrow format, because
// its bytes then come straight from the renderer's copy.
// Three-channel optimal images are not universally supported; those keep the portable expansion.
constexpr bool sample_float16_natively(const SampledFloat16Source& source) {
    if (source.components == 3) return false;
    if (source.components == 4) return source.imported || source.volume;
    if (source.renderer_owned) return source.renderer_narrow_match;
    return source.components == 1 || source.components == 2;
}

} // namespace prosper::frontend
