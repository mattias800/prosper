// Which sampled resource is AvPlayer's NV12 chroma plane (#2005).
//
// The failure this guards is silent by construction: a chroma plane the renderer does not recognise
// goes down the legacy narrow coverage path, which broadcasts its first byte to every channel. The
// resulting picture keeps correct luma, detail and geometry and only its colour collapses onto a
// single green<->magenta axis, so no draw census, colour count or non-black metric can detect it.
// The descriptor-level classification is therefore what has to be asserted.
#include "shared/media/avplayer_plane_policy.hpp"

#include "gpu/texture/guest_texture_layout.hpp"

#include <cstdio>
#include <vector>

using prosper::frontend::AvpChromaReason;
using prosper::frontend::classify_avplayer_chroma_plane;
using prosper::gpu::DataFormat;
using prosper::gpu::ResourceClass;
using prosper::gpu::ShaderResource;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

namespace {

// R-Type Delta (PPSA26414), measured live on the default route: AvPlayer stages a 1920x1080 NV12
// frame at a 2048-byte physical pitch, and the title declares BOTH planes as DIM=2D_ARRAY with a
// single layer (img_dim 5, depth 1) rather than DIM=2D.
constexpr uint64_t kLumaAddr   = 0x2051400600ull;
constexpr uint32_t kPitch      = 2048;
constexpr uint32_t kHeight     = 1080;
constexpr uint64_t kChromaAddr = kLumaAddr + uint64_t{kPitch} * kHeight;   // 0x205161c600
constexpr uint32_t kNv12Bytes  = kPitch * kHeight + kPitch * (kHeight / 2);

ShaderResource rtype_luma_plane() {
    ShaderResource r;
    r.cls = ResourceClass::Texture;
    r.format = DataFormat::Unorm8;
    r.num_components = 1;
    r.gpu_addr = kLumaAddr;
    r.width = kPitch;             // the title builds the luma T# with pitch as its width
    r.height = kHeight;
    r.depth = 1;
    r.img_dim = 5;                // 2D_ARRAY, one layer
    r.tile_mode = 0;              // linear
    r.swizzle[0] = 4; r.swizzle[1] = 0; r.swizzle[2] = 0; r.swizzle[3] = 1;   // (R,0,0,1)
    return r;
}

ShaderResource rtype_chroma_plane() {
    ShaderResource r;
    r.cls = ResourceClass::Texture;
    r.format = DataFormat::Unorm8;
    r.num_components = 2;         // interleaved U/V
    r.gpu_addr = kChromaAddr;
    r.width = kPitch / 2;         // 1024 RG8 texels == 2048 bytes
    r.height = kHeight / 2;
    r.depth = 1;
    r.img_dim = 5;                // 2D_ARRAY, one layer
    r.tile_mode = 0;
    r.swizzle[0] = 4; r.swizzle[1] = 5; r.swizzle[2] = 0; r.swizzle[3] = 1;   // (R,G,0,1)
    return r;
}

// Tales of Graces f Remastered (PPSA19991), measured live on the default route with
// PROSPER_AVPCHROMA_LOG=1: the title drives sceVideodec2 and stages the decoded 1920x1088 NV12 into
// two textures of its OWN, which do not touch. The luma plane ends at 0x2078643000 and the chroma
// plane begins 0x111000 bytes later, so every adjacency form fails and the whole opening movie took
// the coverage broadcast (#2731).
constexpr uint64_t kTogfLumaAddr   = 0x2078423000ull;
constexpr uint64_t kTogfChromaAddr = 0x2078754000ull;   // luma_end + 0x111000
constexpr uint32_t kTogfPitch      = 2048;
constexpr uint32_t kTogfLumaHeight = 1088;

ShaderResource togf_luma_plane() {
    ShaderResource r;
    r.cls = ResourceClass::Texture;
    r.format = DataFormat::Unorm8;
    r.num_components = 1;
    r.gpu_addr = kTogfLumaAddr;
    r.width = kTogfPitch;
    r.height = kTogfLumaHeight;
    r.depth = 1;
    r.img_dim = 1;
    r.tile_mode = 0;
    r.swizzle[0] = 4; r.swizzle[1] = 0; r.swizzle[2] = 0; r.swizzle[3] = 1;   // (R,0,0,1)
    return r;
}

ShaderResource togf_chroma_plane() {
    ShaderResource r;
    r.cls = ResourceClass::Texture;
    r.format = DataFormat::Unorm8;
    r.num_components = 2;
    r.gpu_addr = kTogfChromaAddr;
    r.width = kTogfPitch / 2;          // 1024 RG8 texels == 2048 bytes
    r.height = kTogfLumaHeight / 2;    // 544
    r.depth = 1;
    r.img_dim = 1;
    r.tile_mode = 0;
    r.swizzle[0] = 4; r.swizzle[1] = 5; r.swizzle[2] = 0; r.swizzle[3] = 1;   // (R,G,0,1)
    return r;
}

// Sonic Origins (PPSA05325), measured live on the default route: the same sceVideodec2 staging, but
// GPU-TILED. Both planes are SW_64KB_S (tile_mode 9) one-layer 2D arrays, and the pair is exactly
// adjacent -- in the TILED size. A 3840x2160 8-bit surface is 0x870000 bytes tiled against 0x7e9000
// tight, and it is the tiled figure that lands on the chroma plane's address. (#2731)
constexpr uint64_t kSonicLumaAddr   = 0x2047870000ull;
constexpr uint64_t kSonicChromaAddr = 0x20480e0000ull;
constexpr uint32_t kSonicTileMode   = 9;   // SW_64KB_S

ShaderResource sonic_luma_plane() {
    ShaderResource r;
    r.cls = ResourceClass::Texture;
    r.format = DataFormat::Unorm8;
    r.num_components = 1;
    r.gpu_addr = kSonicLumaAddr;
    r.width = 3840;
    r.height = 2160;
    r.depth = 1;
    r.img_dim = 5;
    r.tile_mode = kSonicTileMode;
    r.swizzle[0] = 4; r.swizzle[1] = 0; r.swizzle[2] = 0; r.swizzle[3] = 1;   // (R,0,0,1)
    return r;
}

ShaderResource sonic_chroma_plane() {
    ShaderResource r;
    r.cls = ResourceClass::Texture;
    r.format = DataFormat::Unorm8;
    r.num_components = 2;
    r.gpu_addr = kSonicChromaAddr;
    r.width = 1920;
    r.height = 1080;
    r.depth = 1;
    r.img_dim = 5;
    r.tile_mode = kSonicTileMode;
    r.swizzle[0] = 4; r.swizzle[1] = 5; r.swizzle[2] = 0; r.swizzle[3] = 1;   // (R,G,0,1)
    return r;
}

auto classify(const ShaderResource& r, const std::vector<ShaderResource>& table) {
    return classify_avplayer_chroma_plane(r, r.width, r.height, table);
}

} // namespace

int main() {
    // ---- 1. The live R-Type descriptors, with AvPlayer's exact pitch provenance registered. ----
    // This is the arm that fails when the classifier demands DIM=2D: the plane is real, its bytes
    // are correct, and it would silently take the coverage broadcast.
    prosper::gpu::register_guest_linear_texture_layout(kLumaAddr, kNv12Bytes, kPitch);
    {
        const std::vector<ShaderResource> table{rtype_luma_plane(), rtype_chroma_plane()};
        const auto v = classify(table[1], table);
        CHECK(v.match);
        CHECK(v.reason == AvpChromaReason::MatchedRegisteredPitch);
        CHECK(v.row_bytes == kPitch);
        CHECK(v.registered_pitch == kPitch);

        // The luma plane is never a chroma plane, whatever else is true of it.
        CHECK(!classify(table[0], table).match);
    }

    // ---- 2. A one-layer 2D-array plane is recognised without any pitch registry either, through
    // the adjacent-sibling luma plane (the captured/replayed and title-staged case). ----
    prosper::gpu::unregister_guest_linear_texture_layout(kLumaAddr);
    {
        const std::vector<ShaderResource> table{rtype_luma_plane(), rtype_chroma_plane()};
        const auto v = classify(table[1], table);
        CHECK(v.match);
        CHECK(v.reason == AvpChromaReason::MatchedAdjacentLumaPlane);
        CHECK(v.sibling_luma_addr == kLumaAddr);
    }

    // ---- 3. The historical DIM=2D shape keeps working unchanged. ----
    {
        ShaderResource luma = rtype_luma_plane();   luma.img_dim = 1;
        ShaderResource chroma = rtype_chroma_plane(); chroma.img_dim = 1;
        const std::vector<ShaderResource> table{luma, chroma};
        const auto v = classify(chroma, table);
        CHECK(v.match);
        CHECK(v.reason == AvpChromaReason::MatchedAdjacentLumaPlane);
    }

    // ---- 4. A REAL multi-layer array is still rejected: its slices are not one contiguous plane,
    // so neither the pitch nor the adjacency reasoning holds. ----
    {
        ShaderResource chroma = rtype_chroma_plane();
        chroma.depth = 6;
        chroma.layer_stride_bytes = kPitch * (kHeight / 2);
        const std::vector<ShaderResource> table{rtype_luma_plane(), chroma};
        const auto v = classify(chroma, table);
        CHECK(!v.match);
        CHECK(v.reason == AvpChromaReason::NotNarrowRg8Plane);
    }
    {
        ShaderResource chroma = rtype_chroma_plane();
        chroma.layer_mip_offset_bytes = 4096;   // a selected level inside a layered allocation
        const std::vector<ShaderResource> table{rtype_luma_plane(), chroma};
        CHECK(!classify(chroma, table).match);
    }
    {
        // depth == 0 on an array descriptor is NOT "a single layer": the decoder emits zero when
        // LAST_ARRAY < BASE_ARRAY, i.e. an inverted, malformed array range. It must fail visibly
        // rather than be treated as one plane.
        ShaderResource chroma = rtype_chroma_plane();
        chroma.depth = 0;
        const std::vector<ShaderResource> table{rtype_luma_plane(), chroma};
        const auto v = classify(chroma, table);
        CHECK(!v.match);
        CHECK(v.reason == AvpChromaReason::NotNarrowRg8Plane);
        // ... and the same malformed range on the SIBLING luma plane must not rescue a chroma plane
        // through the adjacency route either.
        ShaderResource luma = rtype_luma_plane();
        luma.depth = 0;
        const std::vector<ShaderResource> table2{luma, rtype_chroma_plane()};
        CHECK(!classify(table2[1], table2).match);
    }

    // ---- 5. The test stays narrower than "any RG8 texture": established game paths rely on the
    // coverage broadcast for their own two-channel surfaces. ----
    {
        ShaderResource other = rtype_chroma_plane();
        other.swizzle[2] = 6; other.swizzle[3] = 7;   // ordinary identity-ish DST_SEL
        const std::vector<ShaderResource> table{rtype_luma_plane(), other};
        const auto v = classify(other, table);
        CHECK(!v.match);
        CHECK(v.reason == AvpChromaReason::UnrecognisedSwizzle);
    }
    {
        ShaderResource tiled = rtype_chroma_plane();
        tiled.tile_mode = 27;                          // a GPU-tiled surface is not a staged plane
        const std::vector<ShaderResource> table{rtype_luma_plane(), tiled};
        CHECK(!classify(tiled, table).match);
    }
    {
        // No sibling and no registry: an isolated RG8 surface must not be claimed as chroma.
        const std::vector<ShaderResource> table{rtype_chroma_plane()};
        const auto v = classify(table[0], table);
        CHECK(!v.match);
        CHECK(v.reason == AvpChromaReason::NoSiblingLumaPlane);
    }
    {
        // A sibling luma plane of the wrong height (not 2:1) is a different surface.
        ShaderResource luma = rtype_luma_plane();
        luma.height = kHeight / 4;
        const std::vector<ShaderResource> table{luma, rtype_chroma_plane()};
        CHECK(!classify(table[1], table).match);
    }

    // ---- 6. The 64 KiB-boundary form of adjacency, which a title-staged pair can take. ----
    {
        ShaderResource luma = rtype_luma_plane();
        ShaderResource chroma = rtype_chroma_plane();
        const uint64_t luma_end = luma.gpu_addr + uint64_t{kPitch} * kHeight;
        chroma.gpu_addr = (luma_end + 0xffffull) & ~uint64_t{0xffffull};
        CHECK(chroma.gpu_addr != luma_end);            // the case is genuinely exercised
        const std::vector<ShaderResource> table{luma, chroma};
        const auto v = classify(chroma, table);
        CHECK(v.match);
        CHECK(v.reason == AvpChromaReason::MatchedAdjacentLumaPlane);
    }

    // #2034: the same one-layer-2D-array predicate now gates the native R8 luma upload in
    // live_renderer.cpp, not only the chroma classifier. R-Type Delta declares its AvPlayer
    // LUMA plane as dim=5/depth=1, which the old `img_dim == 1u` test missed -- so every movie
    // frame was CPU-expanded to RGBA8 and moved 4x the bytes it needed.
    {
        ShaderResource luma = rtype_luma_plane();
        CHECK(prosper::frontend::avp_plane_is_one_layer_2d(luma));   // dim=5, depth=1
        ShaderResource plain = luma; plain.img_dim = 1;
        CHECK(prosper::frontend::avp_plane_is_one_layer_2d(plain));  // the historical dim=1 case

        // A REAL array must still fail, or the fast path would upload one layer of many and
        // silently drop the rest. Each rejection is asserted on its own so a predicate that
        // stopped checking one of them cannot hide behind the others.
        ShaderResource deep = luma;  deep.depth = 2;
        CHECK(!prosper::frontend::avp_plane_is_one_layer_2d(deep));
        ShaderResource strided = luma; strided.layer_stride_bytes = 0x1000;
        CHECK(!prosper::frontend::avp_plane_is_one_layer_2d(strided));
        ShaderResource mipped = luma;  mipped.layer_mip_offset_bytes = 0x40;
        CHECK(!prosper::frontend::avp_plane_is_one_layer_2d(mipped));
    }

    // ---- 7. #2731: a title that stages the two planes in SEPARATE allocations. Memory adjacency
    // is evidence of an NV12 pair, never a requirement for one -- what makes the pair is the
    // geometry, the shared physical pitch and the co-binding. Without this the whole decoded-video
    // path collapses Cr onto Cb on every title that does its own staging. ----
    {
        const std::vector<ShaderResource> table{togf_luma_plane(), togf_chroma_plane()};
        const auto v = classify(table[1], table);
        // The gap is genuinely exercised: neither adjacency form can be reached from here.
        const uint64_t luma_end = kTogfLumaAddr + uint64_t{kTogfPitch} * kTogfLumaHeight;
        CHECK(luma_end != kTogfChromaAddr);
        CHECK(((luma_end + 0xffffull) & ~uint64_t{0xffffull}) != kTogfChromaAddr);
        CHECK(v.match);
        CHECK(v.reason == AvpChromaReason::MatchedSeparateLumaPlane);
        CHECK(v.sibling_luma_addr == kTogfLumaAddr);
        CHECK(v.resolved_pitch == kTogfPitch);

        // The luma plane of that same pair is still never a chroma plane.
        CHECK(!classify(table[0], table).match);
    }
    {
        // An ADJACENT pair still reports the stronger verdict, so the log keeps telling the two
        // staging routes apart rather than flattening them into one.
        ShaderResource chroma = togf_chroma_plane();
        chroma.gpu_addr = kTogfLumaAddr + uint64_t{kTogfPitch} * kTogfLumaHeight;
        const std::vector<ShaderResource> table{togf_luma_plane(), chroma};
        const auto v = classify(chroma, table);
        CHECK(v.match);
        CHECK(v.reason == AvpChromaReason::MatchedAdjacentLumaPlane);
    }
    {
        // Two planes of one picture are two DISJOINT ranges. A candidate that overlaps its
        // supposed luma plane is one allocation read two ways, and must not be claimed.
        ShaderResource chroma = togf_chroma_plane();
        chroma.gpu_addr = kTogfLumaAddr + uint64_t{kTogfPitch} * 4u;   // inside the luma plane
        const std::vector<ShaderResource> table{togf_luma_plane(), chroma};
        const auto v = classify(chroma, table);
        CHECK(!v.match);
        CHECK(v.reason == AvpChromaReason::NoSiblingLumaPlane);
    }
    {
        // The sibling luma plane's own DST_SEL is deliberately not constrained: GFX10 does not tie
        // it to the component count, so the identity (R,G,B,A) remap is an ordinary descriptor for
        // a one-component surface and must not disqualify a real plane pair.
        ShaderResource luma = togf_luma_plane();
        luma.swizzle[0] = 4; luma.swizzle[1] = 5; luma.swizzle[2] = 6; luma.swizzle[3] = 7;
        const std::vector<ShaderResource> table{luma, togf_chroma_plane()};
        const auto v = classify(table[1], table);
        CHECK(v.match);
        CHECK(v.reason == AvpChromaReason::MatchedSeparateLumaPlane);
    }
    {
        // The pitch relation still has to hold across the gap: a separate allocation is not a
        // licence to pair with any single-channel surface in the table.
        ShaderResource luma = togf_luma_plane();
        luma.linear_row_pitch_bytes = kTogfPitch + 256u;
        const std::vector<ShaderResource> table{luma, togf_chroma_plane()};
        CHECK(!classify(table[1], table).match);
    }
    {
        ShaderResource luma = togf_luma_plane();
        luma.height = kTogfLumaHeight - 2u;                    // no longer 2:1 against 544 rows
        const std::vector<ShaderResource> table{luma, togf_chroma_plane()};
        CHECK(!classify(table[1], table).match);
    }

    // ---- 8. #2731: a GPU-TILED plane pair. A tiled surface has no row pitch, so none of the pitch
    // reasoning applies -- but its padded TILED size does land exactly on the second plane, which is
    // the strongest evidence a pair can carry, so this route keeps adjacency as a requirement. ----
    {
        // The tiled size is load-bearing: the tight w*h figure misses the chroma plane entirely.
        CHECK(prosper::gpu::tiled_surface_bytes(3840, 2160, kSonicTileMode, 0, 1u) ==
              kSonicChromaAddr - kSonicLumaAddr);
        CHECK(kSonicLumaAddr + uint64_t{3840} * 2160 != kSonicChromaAddr);

        const std::vector<ShaderResource> table{sonic_luma_plane(), sonic_chroma_plane()};
        const auto v = classify(table[1], table);
        CHECK(v.match);
        CHECK(v.reason == AvpChromaReason::MatchedAdjacentLumaPlane);
        CHECK(v.sibling_luma_addr == kSonicLumaAddr);

        CHECK(!classify(table[0], table).match);
    }
    {
        // Two planes of one picture share a tile mode. A luma plane in a different swizzle is a
        // different surface, and its size arithmetic would not even be comparable.
        ShaderResource luma = sonic_luma_plane();
        luma.tile_mode = 27;                                   // SW_64KB_R_X
        const std::vector<ShaderResource> table{luma, sonic_chroma_plane()};
        const auto v = classify(table[1], table);
        CHECK(!v.match);
        CHECK(v.reason == AvpChromaReason::NoSiblingLumaPlane);
    }
    {
        // The tiled route does NOT inherit the separate-allocation licence: with no adjacency there
        // is no pitch to corroborate the pair, so a detached tiled RG8 surface stays unclaimed.
        ShaderResource chroma = sonic_chroma_plane();
        chroma.gpu_addr = kSonicChromaAddr + 0x200000ull;
        const std::vector<ShaderResource> table{sonic_luma_plane(), chroma};
        const auto v = classify(chroma, table);
        CHECK(!v.match);
        CHECK(v.reason == AvpChromaReason::NoSiblingLumaPlane);
    }
    {
        // A tiled 2-channel surface with no luma partner at all is an ordinary game texture and
        // must keep the historical coverage broadcast.
        const std::vector<ShaderResource> table{sonic_chroma_plane()};
        const auto v = classify(table[0], table);
        CHECK(!v.match);
        CHECK(v.reason == AvpChromaReason::NoSiblingLumaPlane);
    }

    if (!failures) std::printf("avplayer_plane_policy: OK\n");
    return failures ? 1 : 0;
}
