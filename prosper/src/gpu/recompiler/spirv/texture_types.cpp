// Sampled-image type construction and texture declarations.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>

namespace prosper::gpu {
uint32_t SpirvCompute::sampled_image_type(uint32_t dim, bool is_uint, bool arrayed, bool depth) {
    const uint32_t key = sampled_image_key(dim, is_uint, arrayed, depth);
    auto it = tex_simg_type.find(key);
    if (it != tex_simg_type.end()) return it->second;
    // A SAMPLED Dim=1D image needs Sampled1D just as a storage one does, and this path used
    // to omit it -- spirv-val: "Operand 3 of TypeImage requires one of these capabilities:
    // Sampled1D". It stayed invisible while every 1D binding reaching here was refused
    // upstream; #657 admits Sonic Frontiers' query-only 1D binding, which is what surfaced it.
    // NOT Image1D: that capability is for a 1D STORAGE image, and this type has Sampled=1.
    if (dim == Dim_1D) declare_cap_sampled1d();
    uint32_t ti = id();
    put(types, Op_TypeImage,
        {ti, is_uint ? t_u32 : t_f32, dim, depth ? 1u : 0u, arrayed ? 1u : 0u, 0, 1,
         0});   // sampled, Sampled=1
    uint32_t si = id();
    put(types, Op_TypeSampledImage, {si, ti});
    tex_simg_type[key] = si;
    tex_img_type[key] = ti;
    return si;
}

bool SpirvCompute::declare_texture(uint32_t binding, uint32_t dim, bool is_uint, bool arrayed,
                                   bool depth) {
    const uint32_t key = sampled_image_key(dim, is_uint, arrayed, depth);
    uint32_t simg = sampled_image_type(dim, is_uint, arrayed, depth);
    if (tex_var.count(binding)) return tex_binding_key[binding] == key;
    uint32_t t_ptr = id();
    put(types, Op_TypePointer, {t_ptr, SC_UniformConstant, simg});
    uint32_t v = id();
    put(types, Op_Variable, {t_ptr, v, SC_UniformConstant});
    put(deco, Op_Decorate, {v, Dec_DescriptorSet, desc_set});
    put(deco, Op_Decorate, {v, Dec_Binding, binding});
    tex_var[binding] = v;
    tex_binding_simg[binding] = simg;
    tex_binding_img[binding] = tex_img_type[key];
    tex_binding_uint[binding] = is_uint;
    tex_binding_key[binding] = key;
    tex_binding_arrayed[binding] = arrayed;
    return true;
}
}   // namespace prosper::gpu
