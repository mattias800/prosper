// rdna2_emit_image_lod.cpp -- IMAGE_GET_LOD (MIMG 0x60) in the graphics emitter, moved out of
// rdna2_emit_alu.cpp (its line cap) when the NSA form was admitted (#4808).
#include "gpu/recompiler/rdna2_alu_support.hpp"

namespace prosper::gpu {

// RDNA2 returns {sampler-clamped LOD, raw LOD}; SPIR-V's OpImageQueryLod returns the same pair.
// House of the Dead 2's Unity scene shaders use the ordinary 2D form in fragment programs, and the
// NSA form `image_get_lod v7, [v32, v17], ...`, whose second coordinate is words[2] byte 0
// (mimg_get_lod_nsa_two_addresses). Every unverified dimension, Table 100 control, address shape
// and output component stays fail-visible rather than guessed.
bool emit_image_get_lod(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                        const ShaderResource& res, bool uint_texture) {
    constexpr uint32_t kDim2d = 1u;   // SQ_RSRC_IMG_2D (MIMG DIM and T# TYPE alike)
    const bool nsa = mimg_get_lod_nsa_two_addresses(in);
    if (!b.is_fragment || res.cls != ResourceClass::Texture || in.mimg_dim != kDim2d ||
        res.img_dim != kDim2d || (in.len_dwords != 2u && !nsa) ||
        mimg_get_lod_has_unmodeled_controls(in) || !(in.mimg_dmask & 0x3u) ||
        (in.mimg_dmask & ~0x3u) || res.unnormalized || res.depth_compare ||
        !b.declare_texture(res.binding, Dim_2D, uint_texture))
        return false;
    const auto vread = [&](int r) {
        const auto it = rs.vreg.find(r);
        return it == rs.vreg.end() ? b.uconst(0) : it->second;
    };
    const int second = nsa ? static_cast<int>(in.words[2] & 0xffu) : in.src[0].value + 1;
    uint32_t out[2];
    b.image_get_lod_2d(res.binding, vread(in.src[0].value), vread(second), out);
    int written = 0;
    for (uint32_t component = 0; component < 2; ++component) {
        if (!(in.mimg_dmask & (1u << component))) continue;
        const uint32_t old = vreg_old(b, rs, in.dst.value + written);
        rs.vreg[in.dst.value + written] = out[component];
        predicate_write(b, rs, in.dst.value + written, old);
        ++written;
    }
    return true;
}

}   // namespace prosper::gpu
