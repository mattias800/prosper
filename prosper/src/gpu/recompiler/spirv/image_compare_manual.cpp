// Manual comparison sampling moved from inline SpirvCompute bodies.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>

namespace prosper::gpu {
uint32_t SpirvCompute::image_compare_address_coord(uint32_t value, uint32_t extent, uint32_t mode,
                                                   uint32_t& outside_border) {
    outside_border = bfalse();
    if (mode == 0u) return sbin(Op_SMod, value, extent);
    if (mode == 1u) {
        const uint32_t period = sbin(Op_IMul, extent, uconst(2));
        const uint32_t wrapped = sbin(Op_SMod, value, period);
        const uint32_t reflected = sbin(Op_ISub, sbin(Op_ISub, period, uconst(1)), wrapped);
        return sel(scmp(Op_SGreaterThanEqual, wrapped, extent), reflected, wrapped);
    }
    const uint32_t maximum = sbin(Op_ISub, extent, uconst(1));
    if (mode >= 6u) {
        outside_border =
            lor(scmp(Op_SLessThan, value, uconst(0)), scmp(Op_SGreaterThanEqual, value, extent));
    }
    return sext2(Glsl_SMin, sext2(Glsl_SMax, value, uconst(0)), maximum);
}

void SpirvCompute::image_compare_mix_taps(
    const bool& coord_arrayed, bool& arrayed, uint32_t& layer_bits, uint32_t& img,
    uint32_t& dref_bits, uint32_t& compare_func, const bool& uses_border, uint32_t& border_compare,
    const uint32_t& x0, const uint32_t& y0, uint32_t& x0_border, uint32_t& y0_border,
    const uint32_t& x1, uint32_t& x1_border, const uint32_t& y1, uint32_t& y1_border,
    const uint32_t& fx, const uint32_t& fy, uint32_t*& out) {
    auto compare_fetch = [&](uint32_t tx, uint32_t ty, uint32_t outside_border) {
        // #325: follow the DECLARATION, as the sampled branch above does. A caller that
        // reached an arrayed binding without supplying a slice still needs three
        // components, and reads the base one.
        uint32_t coord = id();
        if (coord_arrayed)
            put(code, Op_CompositeConstruct,
                {t_v3u_fetch(), coord, tx, ty,
                 arrayed ? i2u(cvt_f2i(bcf(layer_bits))) : uconst(0)});
        else
            put(code, Op_CompositeConstruct, {t_v2u(), coord, tx, ty});
        uint32_t texel = id();
        put(code, Op_ImageFetch, {t_v4f, texel, img, coord, ImgOp_Lod, uconst(0)});
        uint32_t depth = id();
        put(code, Op_CompositeExtract, {t_f32, depth, texel, 0});
        const uint32_t fetched = image_dref_compare_bits(depth, dref_bits, compare_func);
        return uses_border ? sel(outside_border, border_compare, fetched) : fetched;
    };
    auto tap_border = [&](uint32_t x_border, uint32_t y_border) {
        return uses_border ? lor(x_border, y_border) : bfalse();
    };
    const uint32_t c00 = compare_fetch(x0, y0, tap_border(x0_border, y0_border));
    const uint32_t c10 = compare_fetch(x1, y0, tap_border(x1_border, y0_border));
    const uint32_t c01 = compare_fetch(x0, y1, tap_border(x0_border, y1_border));
    const uint32_t c11 = compare_fetch(x1, y1, tap_border(x1_border, y1_border));
    auto lerp = [&](uint32_t a, uint32_t b, uint32_t t) {
        return fbin(Op_FAdd, a, fbin(Op_FMul, fbin(Op_FSub, b, a), t));
    };
    const uint32_t bits = lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy);
    out[0] = out[1] = out[2] = out[3] = bits;
}

void SpirvCompute::image_compare_nearest(const bool& coord_arrayed, uint32_t& u_bits,
                                         uint32_t& v_bits, bool& arrayed, uint32_t& layer_bits,
                                         uint32_t& si, uint32_t& dref_bits, uint32_t& compare_func,
                                         uint32_t*& out) {
    uint32_t coord = id();
    if (coord_arrayed)
        put(code, Op_CompositeConstruct,
            {t_v3f(), coord, bcf(u_bits), bcf(v_bits), arrayed ? bcf(layer_bits) : fconstf(0.0f)});
    else
        put(code, Op_CompositeConstruct, {t_v2f(), coord, bcf(u_bits), bcf(v_bits)});
    uint32_t res = id();
    put(code, Op_ImageSampleExplicitLod, {t_v4f, res, si, coord, ImgOp_Lod, fconstf(0.0f)});
    uint32_t depth = id();
    put(code, Op_CompositeExtract, {t_f32, depth, res, 0});
    const uint32_t bits = image_dref_compare_bits(depth, dref_bits, compare_func);
    out[0] = out[1] = out[2] = out[3] = bits;
}

void SpirvCompute::image_sample_dref_manual_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                               uint32_t dref_bits, uint32_t compare_func,
                                               bool linear_filter, uint32_t addr_u, uint32_t addr_v,
                                               uint32_t border_color_type, uint32_t out[4],
                                               bool arrayed, uint32_t layer_bits) {
        // Consult the DECLARATION, not just the caller: a binding declared arrayed needs
        // three components whatever opcode reached it, and a caller that did not supply a
        // slice reads the base one (#325). Hoisted so BOTH the sampled and the manual-fetch
        // branches below use it -- they disagreed, and only one was fixed.
    const bool coord_arrayed =
        arrayed || (tex_binding_arrayed.count(binding) && tex_binding_arrayed[binding]);

    uint32_t si = id();
    put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
    if (linear_filter) {
        if (!declared_image_query) {
            put(caps, Op_Capability, {Cap_ImageQuery});
            declared_image_query = true;
        }
        uint32_t img = id();
        put(code, Op_Image, {tex_binding_img[binding], img, si});
        // #325: the DECLARATION, not the caller -- the third of three uses in this function,
        // and the one the first pass missed. Query-size arity is validated exactly, so an
        // ivec2 against an Arrayed image is invalid SPIR-V.
        uint32_t size = id();
        put(code, Op_ImageQuerySizeLod, {coord_arrayed ? t_v3i() : t_v2i(), size, img, uconst(0)});
        uint32_t w_i = id();
        put(code, Op_CompositeExtract, {t_i32, w_i, size, 0});
        uint32_t h_i = id();
        put(code, Op_CompositeExtract, {t_i32, h_i, size, 1});
        const uint32_t w = i2u(w_i), h = i2u(h_i);

        // Normalized linear filtering addresses texel centers at n + 0.5. Convert to that
        // footprint, then address each tap with the decoded S#. Match the backend's current
        // Gen5 contract (#1400): 0=repeat, 1=mirrored repeat, 2..5=clamp-to-edge, 6..7=border.
        // The finer mirror-once/half-border distinctions in 3..5 remain deliberately folded by
        // both paths until the shared sampler decoder grows them.
        const uint32_t half = bcu(fconstf(0.5f));
        const uint32_t x = fbin(Op_FSub, fbin(Op_FMul, u_bits, cvt_i2f(w)), half);
        const uint32_t y = fbin(Op_FSub, fbin(Op_FMul, v_bits, cvt_i2f(h)), half);
        const uint32_t fx = fext1(Glsl_Fract, x), fy = fext1(Glsl_Fract, y);
        const uint32_t bx = cvt_f2i(fext1(Glsl_Floor, x));
        const uint32_t by = cvt_f2i(fext1(Glsl_Floor, y));
        auto address_coord = [&](uint32_t value, uint32_t extent, uint32_t mode,
                                 uint32_t& outside_border) {
            return image_compare_address_coord(value, extent, mode, outside_border);
        };
        uint32_t x0_border, x1_border, y0_border, y1_border;
        const uint32_t x0 = address_coord(bx, w, addr_u, x0_border);
        const uint32_t y0 = address_coord(by, h, addr_v, y0_border);
        const uint32_t x1 = address_coord(sbin(Op_IAdd, bx, uconst(1)), w, addr_u, x1_border);
        const uint32_t y1 = address_coord(sbin(Op_IAdd, by, uconst(1)), h, addr_v, y1_border);

        const bool uses_border = addr_u >= 6u || addr_v >= 6u;
        uint32_t border_compare = 0;
        if (uses_border) {
            // The sampled depth is R: transparent/opaque black and unsupported custom-register
            // border colors all contribute 0, while opaque white contributes 1. This mirrors the
            // backend sampler's custom-border fallback.
            const float border_depth = border_color_type == 2u ? 1.0f : 0.0f;
            border_compare =
                image_dref_compare_bits(fconstf(border_depth), dref_bits, compare_func);
        }

        image_compare_mix_taps(coord_arrayed, arrayed, layer_bits, img, dref_bits, compare_func,
                               uses_border, border_compare, x0, y0, x0_border, y0_border, x1,
                               x1_border, y1, y1_border, fx, fy, out);
        return;
    }

    image_compare_nearest(coord_arrayed, u_bits, v_bits, arrayed, layer_bits, si, dref_bits,
                          compare_func, out);
}
}   // namespace prosper::gpu
