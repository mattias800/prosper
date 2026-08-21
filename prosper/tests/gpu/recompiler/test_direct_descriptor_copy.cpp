// test_direct_descriptor_copy — a DIRECT user-data descriptor that the shader stages into a
// different SGPR range with s_mov_b32 before the image op must still resolve to its resource.
//
// Provenance for an indirect (s_load-derived) descriptor survives a copy: `sreg_srt` is propagated
// through s_mov_b32, so `by_srt_offset` still keys it. A DIRECT descriptor -- one that lives in
// user data and is keyed by `sgpr_base` -- had no equivalent, so the copy destroyed its only key:
// `by_sgpr_base(SRSRC)` is guarded by `!sreg_range_written(...)`, and the eight moves make that
// guard false. The image op then resolved to nothing and the draw was dropped (#1773).
//
// Live evidence: Earthion (PPSA28061) pixel shader 0x4101c1f00 stages s[9:16] into s[20:27] with
// eight s_mov_b32 and samples through s[20:27]. Its `image_sample` words are reproduced verbatim in
// the "exact live encoding" arm below.
//
// Pure (no Vulkan), so it runs in CI.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <cstdio>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

// SOP1 s_mov_b32 sD, sS.  [31:23]=opcode class, [22:16]=SDST, [15:8]=OP(0x03), [7:0]=SSRC0.
constexpr uint32_t s_mov_b32(uint32_t dst, uint32_t src) {
    return 0xBE800300u | (dst << 16) | src;
}

// MIMG word1: [7:0]=VADDR, [15:8]=VDATA, [20:16]=SRSRC/4, [25:21]=SSAMP/4.
constexpr uint32_t mimg_word1(uint32_t vaddr, uint32_t vdata, uint32_t srsrc, uint32_t ssamp) {
    return vaddr | (vdata << 8) | ((srsrc / 4u) << 16) | ((ssamp / 4u) << 21);
}

constexpr uint32_t kImageSampleDmask15 = 0xF0800F08u;   // image_sample, DMASK=0xf
constexpr uint32_t kEndpgm             = 0xBF810000u;

// A direct T# occupying user-data s[8:15], sampled through the S# at s[16:19].
ShaderResourceTable direct_texture_table() {
    ShaderResourceTable rt;
    ShaderResource texture{};
    texture.cls = ResourceClass::Texture;
    texture.binding = 4;
    texture.img_dim = 1;
    texture.width = texture.height = 4;
    texture.sgpr_base = 8;              // DIRECT provenance: the descriptor IS user data dw8..15
    texture.sampler_sgpr_base = 16;
    rt.resources.push_back(texture);
    return rt;
}

ComputeShaderConfig direct_texture_config() {
    ComputeShaderConfig config;
    config.user_sgprs.resize(20);       // s0..s19 are entry-time user data
    return config;
}

bool recompiles(const std::vector<uint32_t>& code) {
    const ShaderResourceTable rt = direct_texture_table();
    return !recompile_compute(code.data(), code.size(), &rt, direct_texture_config()).empty();
}

}  // namespace

int main() {
    printf("== test_direct_descriptor_copy ==\n");

    // --- Positive control, constructed independently of the copy case ------------------------
    // The same resource table and the same image op, reading the descriptor where it actually
    // lives. This proves the harness can EXPRESS a resolving direct-T# image_sample at all; a
    // copy-arm failure is only evidence about the copy if this arm passes.
    const std::vector<uint32_t> direct = {
        kImageSampleDmask15, mimg_word1(0, 0, /*srsrc=*/8, /*ssamp=*/16),
        kEndpgm,
    };
    CHECK(recompiles(direct),
          "control: a direct user-data T# read in place (SRSRC=s[8:15]) resolves and recompiles");

    // --- The defect: the identical descriptor, staged through s_mov_b32 ----------------------
    // s[20:27] <- s[8:15], then sample through s[20:27]. Nothing about the descriptor changed;
    // only the register range holding it did.
    std::vector<uint32_t> copied;
    for (uint32_t i = 0; i < 8; ++i) copied.push_back(s_mov_b32(20 + i, 8 + i));
    copied.push_back(kImageSampleDmask15);
    copied.push_back(mimg_word1(0, 0, /*srsrc=*/20, /*ssamp=*/16));
    copied.push_back(kEndpgm);
    CHECK(recompiles(copied),
          "a direct user-data T# copied with eight s_mov_b32 still resolves through the copy");

    // Earthion's whole staging shape (program 0x4101c1f00), not merely its image_sample words. The
    // structure is what makes this arm worth having: the shader stages descriptor A out of user data
    // dw9..16 into s[20:27], and THEN reuses s[12:19] for descriptor B out of dw1..8 -- so by the
    // time the first sample executes, five of A's ORIGIN words (s12..s16) have been overwritten.
    //
    // A synthesized copy that does not reuse the source range passes even when the resolver wrongly
    // re-checks `!sreg_written` on the origin; this one does not. That difference is the whole point
    // of pinning the real shape: the first version of this test had only the synthesized arm and the
    // verbatim image_sample words, and it went green against a rule that still declined the live
    // shader.
    //
    // pc  0..7   s_mov_b32 s20..s27, s9..s16    ; stage descriptor A
    // pc  8..15  s_mov_b32 s12..s19, s1..s8     ; stage descriptor B, clobbering A's origin words
    // pc 16      image_sample v3, v[6:7], s[20:27], s[28:31]   ; A -- exact live encoding
    // pc 18      image_sample v0, v[6:7], s[12:19], s[28:31]   ; B
    ShaderResourceTable live_rt;
    { ShaderResource a{};
      a.cls = ResourceClass::Texture; a.binding = 4; a.img_dim = 1;
      a.width = a.height = 4; a.sgpr_base = 9; a.sampler_sgpr_base = 28;
      live_rt.resources.push_back(a); }
    { ShaderResource bres{};
      bres.cls = ResourceClass::Texture; bres.binding = 5; bres.img_dim = 1;
      bres.width = bres.height = 4; bres.sgpr_base = 1; bres.sampler_sgpr_base = 28;
      live_rt.resources.push_back(bres); }
    ComputeShaderConfig live_config;
    live_config.user_sgprs.resize(32);
    std::vector<uint32_t> live;
    for (uint32_t i = 0; i < 8; ++i) live.push_back(s_mov_b32(20 + i, 9 + i));
    for (uint32_t i = 0; i < 8; ++i) live.push_back(s_mov_b32(12 + i, 1 + i));
    live.push_back(0xF0800708u);        // image_sample, DMASK=0x7  (verbatim from the live capture)
    live.push_back(0x00E50306u);        // v3, v[6:7], s[20:27], s[28:31]
    live.push_back(0xF0800708u);
    live.push_back(mimg_word1(6, 0, /*srsrc=*/12, /*ssamp=*/28));
    live.push_back(kEndpgm);
    CHECK(!recompile_compute(live.data(), live.size(), &live_rt, live_config).empty(),
          "Earthion's whole staging shape recompiles, with A's origin words reused for B");

    // --- Negative arms: a copy that is not a faithful whole-descriptor move must NOT resolve --
    // One word taken from somewhere else. Admitting this would bind a descriptor the shader never
    // assembled, which is worse than declining: it renders with silently wrong texels.
    std::vector<uint32_t> mixed = copied;
    mixed[7] = s_mov_b32(27, 3);        // s27 <- s3 instead of s15
    CHECK(!recompiles(mixed),
          "a copy whose last word comes from an unrelated SGPR is still rejected");

    // A non-consecutive copy: the range is complete but permuted, so it is a different descriptor.
    std::vector<uint32_t> permuted = copied;
    permuted[0] = s_mov_b32(20, 9);
    permuted[1] = s_mov_b32(21, 8);
    CHECK(!recompiles(permuted),
          "a permuted copy of the same eight words is still rejected");

    // A partial copy: only the first four words are staged, the rest is whatever s[24:27] held.
    std::vector<uint32_t> partial;
    for (uint32_t i = 0; i < 4; ++i) partial.push_back(s_mov_b32(20 + i, 8 + i));
    partial.push_back(kImageSampleDmask15);
    partial.push_back(mimg_word1(0, 0, /*srsrc=*/20, /*ssamp=*/16));
    partial.push_back(kEndpgm);
    CHECK(!recompiles(partial),
          "a copy covering only half the descriptor is still rejected");

    // The source range must itself still be entry-time user data. Overwrite one source word after
    // the copy reads it -- no: overwrite it BEFORE, so the copy stages a computed value.
    std::vector<uint32_t> clobbered_source;
    clobbered_source.push_back(0xBE880381u);   // s_mov_b32 s8, 1  -- s8 is no longer user data
    for (uint32_t i = 0; i < 8; ++i) clobbered_source.push_back(s_mov_b32(20 + i, 8 + i));
    clobbered_source.push_back(kImageSampleDmask15);
    clobbered_source.push_back(mimg_word1(0, 0, /*srsrc=*/20, /*ssamp=*/16));
    clobbered_source.push_back(kEndpgm);
    CHECK(!recompiles(clobbered_source),
          "a copy whose source word was overwritten before the copy is still rejected");

    printf("%s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
