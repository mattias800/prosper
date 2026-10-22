// Storage-image declarations, reads and predicated writes.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>

namespace prosper::gpu {
void SpirvCompute::declare_storage_image(uint32_t binding, uint32_t dim, bool arrayed, bool ms,
                                         bool float_texel, uint32_t image_format, bool packed_r11) {
    if (dim == Dim_1D) {   // SPIR-V: Dim=1D needs Sampled1D; storage 1D also needs Image1D
        declare_cap_sampled1d();
        declare_cap_image1d();
    }
    if (ms && !declared_ms) {
        put(caps, Op_Capability, {Cap_StorageImageMultisample});
        declared_ms = true;
    }
    if (ms && arrayed && !declared_msarray) {
        put(caps, Op_Capability, {Cap_ImageMSArray});
        declared_msarray = true;
    }
    if ((image_format == ImgFmt_R8ui || image_format == ImgFmt_R16ui ||
         image_format == ImgFmt_Rgba8ui) &&
        !declared_storage_extended) {
        put(caps, Op_Capability, {Cap_StorageImageExtendedFormats});
        declared_storage_extended = true;
    }
    uint32_t key = stg_key(dim, arrayed, ms, float_texel, image_format);
    if (!stg_img_type.count(key)) {
        uint32_t ti = id();
        put(types, Op_TypeImage,
            {ti, float_texel ? t_f32 : t_u32, dim, 0, arrayed ? 1u : 0u, ms ? 1u : 0u,
             Img_Sampled_Storage, image_format});
        stg_img_type[key] = ti;
    }
    if (stg_img_var.count(binding)) return;
    uint32_t t_ptr = id();
    put(types, Op_TypePointer, {t_ptr, SC_UniformConstant, stg_img_type[key]});
    uint32_t v = id();
    put(types, Op_Variable, {t_ptr, v, SC_UniformConstant});
    put(deco, Op_Decorate, {v, Dec_DescriptorSet, desc_set});
    put(deco, Op_Decorate, {v, Dec_Binding, binding});
    stg_img_binding_type[binding] = stg_img_type[key];
    stg_img_var[binding] = v;
    stg_img_float[binding] = float_texel;
    stg_img_format[binding] = image_format;
    stg_img_packed_r11[binding] = packed_r11;
}

void SpirvCompute::image_read(uint32_t binding, uint32_t dim, bool arrayed, uint32_t ncoord,
                              const uint32_t* coords, uint32_t out[4], bool ms, uint32_t sample) {
    if (stg_img_format[binding] == ImgFmt_Unknown && !declared_read_wo_fmt) {
        put(caps, Op_Capability, {Cap_StorageImageReadWithoutFormat});
        declared_read_wo_fmt = true;
    }
    const bool float_texel = stg_img_float[binding];
    const uint32_t vector_type = float_texel ? t_v4f : t_v4u();
    uint32_t img = id();
    put(code, Op_Load, {stg_img_binding_type[binding], img, stg_img_var[binding]});
    uint32_t coord = stg_coord(ncoord, coords);
    uint32_t res = id();
    if (ms)
        put(code, Op_ImageRead, {vector_type, res, img, coord, ImgOp_Sample, sample});
    else
        put(code, Op_ImageRead, {vector_type, res, img, coord});
    if (stg_img_packed_r11[binding]) {
        uint32_t packed = id();
        put(code, Op_CompositeExtract, {t_u32, packed, res, 0});
        out[0] = unpack_ufloat(packed, 0, 11);
        out[1] = unpack_ufloat(packed, 11, 11);
        out[2] = unpack_ufloat(packed, 22, 10);
        out[3] = bcu(fconstf(1.0f));
        return;
    }
    for (uint32_t c = 0; c < 4; c++) {
        uint32_t e = id();
        put(code, Op_CompositeExtract, {float_texel ? t_f32 : t_u32, e, res, c});
        out[c] = float_texel ? bcu(e) : e;
    }
}

void SpirvCompute::image_write(uint32_t binding, uint32_t dim, bool arrayed, uint32_t ncoord,
                               const uint32_t* coords, const uint32_t vals[4], bool predicated,
                               uint32_t pred) {
    if (stg_img_format[binding] == ImgFmt_Unknown && !declared_write_wo_fmt) {
        put(caps, Op_Capability, {Cap_StorageImageWriteWithoutFormat});
        declared_write_wo_fmt = true;
    }
    const bool float_texel = stg_img_float[binding];
    uint32_t img = id();
    put(code, Op_Load, {stg_img_binding_type[binding], img, stg_img_var[binding]});
    uint32_t coord = stg_coord(ncoord, coords);
    uint32_t texel = id();
    if (stg_img_packed_r11[binding]) {
        const uint32_t r = pack_ufloat(vals[0], 6);
        const uint32_t g = ibin(Op_ShiftLeftLogical, pack_ufloat(vals[1], 6), uconst(11));
        const uint32_t b = ibin(Op_ShiftLeftLogical, pack_ufloat(vals[2], 5), uconst(22));
        const uint32_t packed = ibin(Op_BitwiseOr, ibin(Op_BitwiseOr, r, g), b);
        put(code, Op_CompositeConstruct, {t_v4u(), texel, packed, uconst(0), uconst(0), uconst(0)});
    } else if (float_texel)
        put(code, Op_CompositeConstruct,
            {t_v4f, texel, bcf(vals[0]), bcf(vals[1]), bcf(vals[2]), bcf(vals[3])});
    else {
        // Vulkan's narrow integer storage conversion may saturate a u32 value that does not fit
        // the target, while the guest image-store contract discards the high bits. Make that
        // conversion explicit so typed R8ui/R16ui stay identical to the raw CPU pack fallback
        // for arbitrary shader values, not only values loaded from another narrow surface.
        const uint32_t width_mask =
            (stg_img_format[binding] == ImgFmt_R8ui || stg_img_format[binding] == ImgFmt_Rgba8ui)
                ? 0xffu
            : stg_img_format[binding] == ImgFmt_R16ui ? 0xffffu
                                                      : 0u;
        uint32_t narrowed[4] = {vals[0], vals[1], vals[2], vals[3]};
        if (width_mask)
            for (uint32_t c = 0; c < 4; ++c)
                narrowed[c] = ibin(Op_BitwiseAnd, vals[c], uconst(width_mask));
        put(code, Op_CompositeConstruct,
            {t_v4u(), texel, narrowed[0], narrowed[1], narrowed[2], narrowed[3]});
    }
    if (!predicated) {
        put(code, Op_ImageWrite, {img, coord, texel});
        return;
    }
    uint32_t then = id(), merge = id();
    put(code, Op_SelectionMerge, {merge, 0});
    put(code, Op_BranchConditional, {pred, then, merge});
    put(code, Op_Label, {then});
    cur_block = then;
    put(code, Op_ImageWrite, {img, coord, texel});
    put(code, Op_Branch, {merge});
    put(code, Op_Label, {merge});
    cur_block = merge;
}
}   // namespace prosper::gpu
