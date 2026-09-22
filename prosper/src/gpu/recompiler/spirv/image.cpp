// Generated move, not new code: `SpirvCompute`'s method bodies, lifted out of
// gpu/recompiler/rdna2_to_spirv_internal.hpp. outline_methods.py checked each body byte for byte.

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

void SpirvCompute::set_tap(uint32_t a, uint32_t bb, uint32_t c, uint32_t d) {
        uint32_t v = id(); putv(code, Op_CompositeConstruct, {t_v4f, v, bcf(a), bcf(bb), bcf(c), bcf(d)});
        tap_vec = v;
    }

bool SpirvCompute::tex_is_arrayed(uint32_t binding) {
        auto it = tex_binding_arrayed.find(binding);
        return it != tex_binding_arrayed.end() && it->second;
    }

uint32_t SpirvCompute::tex_coord_uv(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                    uint32_t layer_bits) {
        uint32_t c = id();
        if (tex_binding_arrayed.count(binding) && tex_binding_arrayed[binding])
            put(code, Op_CompositeConstruct,
                {t_v3f(), c, bcf(u_bits), bcf(v_bits),
                 layer_bits ? bcf(layer_bits) : fconstf(0.0f)});
        else
            put(code, Op_CompositeConstruct, {t_v2f(), c, bcf(u_bits), bcf(v_bits)});
        return c;
    }

void SpirvCompute::unpack_texture_result(uint32_t binding, uint32_t result, uint32_t out[4]) {
        const bool is_uint = tex_binding_uint[binding];
        const uint32_t scalar_type = is_uint ? t_u32 : t_f32;
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t e = id(); put(code, Op_CompositeExtract, {scalar_type, e, result, c});
            out[c] = is_uint ? e : bcu(e);
        }
    }

void SpirvCompute::image_sample_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
        uint32_t res   = id();
        if (is_fragment) put(code, Op_ImageSampleImplicitLod, {texture_vec4(binding), res, si, coord});
        else             put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Lod, fconstf(0.0f)});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_sample_grad_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                              uint32_t dsdx_bits, uint32_t dtdx_bits,
                              uint32_t dsdy_bits, uint32_t dtdy_bits, uint32_t out[4]) {
        uint32_t si = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
        uint32_t grad_x = id(); put(code, Op_CompositeConstruct,
                                    {t_v2f(), grad_x, bcf(dsdx_bits), bcf(dtdx_bits)});
        uint32_t grad_y = id(); put(code, Op_CompositeConstruct,
                                    {t_v2f(), grad_y, bcf(dsdy_bits), bcf(dtdy_bits)});
        uint32_t res = id(); put(code, Op_ImageSampleExplicitLod,
                                 {texture_vec4(binding), res, si, coord,
                                  ImgOp_Grad, grad_x, grad_y});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_sample_3d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t w_bits, uint32_t out[4]) {
        uint32_t simg  = tex_binding_simg[binding];
        uint32_t si    = id(); put(code, Op_Load, {simg, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v3f(), coord, bcf(u_bits), bcf(v_bits), bcf(w_bits)});
        uint32_t res   = id();
        if (is_fragment) put(code, Op_ImageSampleImplicitLod, {texture_vec4(binding), res, si, coord});
        else             put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Lod, fconstf(0.0f)});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_sample_lod_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t lod_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
        uint32_t res   = id(); put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Lod, bcf(lod_bits)});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_sample_2d_array(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                               uint32_t layer_bits, uint32_t out[4]) {
        uint32_t si = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        // #2998: PROSPER_FORCE_LAYER=<n> substitutes a constant array layer, and SAYS SO, which is
        // the point -- it separates "the shader is given the wrong slice" from "the slice it asks
        // for holds the wrong content", two failures that look identical on screen. A probe that
        // could not show its own lever moved would leave a null meaning nothing.
        //
        // This forces guest-visible state, so its output illustrates an investigation and is never
        // acceptance evidence for a rendered frame.
        const int forced_layer = forced_array_layer();
        uint32_t layer_use = bcf(layer_bits);
        if (forced_layer >= 0) {
            layer_use = fconstf((float)forced_layer);
            announce_forced_layer(binding, forced_layer);
        }
        uint32_t coord = id(); put(code, Op_CompositeConstruct,
                                   {t_v3f(), coord, bcf(u_bits), bcf(v_bits), layer_use});
        uint32_t res = id();
        if (is_fragment)
            put(code, Op_ImageSampleImplicitLod, {texture_vec4(binding), res, si, coord});
        else
            put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord,
                                                  ImgOp_Lod, fconstf(0.0f)});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_sample_lod_2d_array(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                   uint32_t layer_bits, uint32_t lod_bits, uint32_t out[4]) {
        uint32_t si = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        // #2998: PROSPER_FORCE_LAYER applies HERE TOO. Covering only the implicit-LOD sampler is
        // the scope error this session already made once and had to withdraw: the census puts most
        // of this title's array events on other opcodes, so a probe on one helper produces a null
        // that means nothing about the rest.
        uint32_t lod_layer = bcf(layer_bits);
        if (forced_array_layer() >= 0) {
            lod_layer = fconstf((float)forced_array_layer());
            announce_forced_layer(binding, forced_array_layer());
        }
        uint32_t coord = id(); put(code, Op_CompositeConstruct,
                                   {t_v3f(), coord, bcf(u_bits), bcf(v_bits), lod_layer});
        uint32_t res = id(); put(code, Op_ImageSampleExplicitLod,
                                 {texture_vec4(binding), res, si, coord,
                                  ImgOp_Lod, bcf(lod_bits)});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_sample_lod_3d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t w_bits,
                             uint32_t lod_bits, uint32_t out[4]) {
        uint32_t simg  = tex_binding_simg[binding];
        uint32_t si    = id(); put(code, Op_Load, {simg, si, tex_var[binding]});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v3f(), coord, bcf(u_bits), bcf(v_bits), bcf(w_bits)});
        uint32_t res   = id(); put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Lod, bcf(lod_bits)});
        unpack_texture_result(binding, res, out);
    }

uint32_t SpirvCompute::image_dref_compare_bits(uint32_t depth, uint32_t dref_bits, uint32_t compare_func) {
        if (compare_func == 0u) return bcu(fconstf(0.0f));   // NEVER
        if (compare_func == 7u) return bcu(fconstf(1.0f));   // ALWAYS
        uint32_t op;
        switch (compare_func) {
            case 1u: op = Op_FOrdLessThan;         break;
            case 2u: op = Op_FOrdEqual;            break;
            case 3u: op = Op_FOrdLessThanEqual;    break;
            case 4u: op = Op_FOrdGreaterThan;      break;
            case 5u: op = Op_FOrdNotEqual;         break;
            default: op = Op_FOrdGreaterThanEqual; break;   // 6 GEQUAL
        }
        uint32_t cmp = id(); put(code, op, {t_bool, cmp, bcf(dref_bits), depth});
        uint32_t result = id();
        put(code, Op_Select, {t_f32, result, cmp, fconstf(1.0f), fconstf(0.0f)});
        return bcu(result);
    }

void SpirvCompute::image_sample_bias_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t bias_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
        uint32_t res   = id();
        if (is_fragment) put(code, Op_ImageSampleImplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Bias, bcf(bias_bits)});
        else             put(code, Op_ImageSampleExplicitLod, {texture_vec4(binding), res, si, coord, ImgOp_Lod, fconstf(0.0f)});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_gather_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                   uint32_t comp, uint32_t out[4], uint32_t layer_bits) {
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits, layer_bits);
        uint32_t res   = id(); put(code, Op_ImageGather, {texture_vec4(binding), res, si, coord, uconst(comp)});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_gather_offset_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t comp,
                                uint32_t off_bits, uint32_t out[4], uint32_t layer_bits) {
        if (!declared_gather_ext) { put(caps, Op_Capability, {Cap_ImageGatherExtended}); declared_gather_ext = true; }
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits, layer_bits);
        // signed 6-bit texel offsets. NOTE: bfe_s takes SPIR-V IDs — raw integers here (the #296
        // original) emitted OpBitFieldSExtract with invalid operand IDs; never caught live because
        // the only gather4_lz_o user (DOLL's FXAA PS) still rejected upstream on its execz region.
        uint32_t ox    = bfe_s(off_bits, uconst(0), uconst(6)), oy = bfe_s(off_bits, uconst(8), uconst(6));
        uint32_t offv  = id(); put(code, Op_CompositeConstruct, {t_v2i(), offv, bcs(ox), bcs(oy)});
        uint32_t res   = id(); put(code, Op_ImageGather, {texture_vec4(binding), res, si, coord, uconst(comp), ImgOp_Offset, offv});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_sample_lz_offset_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                   uint32_t off_bits, uint32_t out[4]) {
        if (!declared_image_query) { put(caps, Op_Capability, {Cap_ImageQuery}); declared_image_query = true; }
        uint32_t si   = id(); put(code, Op_Load,  {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t img  = id(); put(code, Op_Image, {tex_binding_img[binding], img, si});
        // #325: OpImageQuerySizeLod on an Arrayed 2D image yields ivec3 (w, h, layers). Asking
        // for ivec2 is invalid SPIR-V; the width and height are still components 0 and 1.
        const uint32_t q_type = tex_is_arrayed(binding) ? t_v3i() : t_v2i();
        uint32_t size = id(); put(code, Op_ImageQuerySizeLod, {q_type, size, img, uconst(0)});
        uint32_t w_i  = id(); put(code, Op_CompositeExtract, {t_i32, w_i, size, 0});
        uint32_t h_i  = id(); put(code, Op_CompositeExtract, {t_i32, h_i, size, 1});
        uint32_t ox = bfe_s(off_bits, uconst(0), uconst(6)), oy = bfe_s(off_bits, uconst(8), uconst(6));   // signed 6-bit texel offsets
        uint32_t du = fbin(Op_FDiv, cvt_i2f(ox), cvt_i2f(i2u(w_i)));
        uint32_t dv = fbin(Op_FDiv, cvt_i2f(oy), cvt_i2f(i2u(h_i)));
        image_sample_lod_2d(binding, fbin(Op_FAdd, u_bits, du), fbin(Op_FAdd, v_bits, dv), uconst(0), out);
    }

void SpirvCompute::image_get_resinfo(uint32_t binding, uint32_t dim, uint32_t lod_bits, uint32_t out[4]) {
        if (!declared_image_query) { put(caps, Op_Capability, {Cap_ImageQuery}); declared_image_query = true; }
        const uint32_t simg = tex_binding_simg[binding];
        const uint32_t image_type = tex_binding_img[binding];
        uint32_t si = id();  put(code, Op_Load,  {simg, si, tex_var[binding]});
        uint32_t img = id(); put(code, Op_Image, {image_type, img, si});
        out[0] = out[1] = out[2] = uconst(1);
        if (dim == Dim_1D) {
            uint32_t size = id(); put(code, Op_ImageQuerySizeLod, {t_i32, size, img, bcs(lod_bits)});
            out[0] = i2u(size);
        } else {
            // #325: an Arrayed 2D image queries as ivec3, its third component being the layer
            // COUNT -- which is exactly what GET_RESINFO's third result means for a 2D_ARRAY T#, so
            // reporting it is right rather than merely legal. Dim_Cube queries as ivec2 (width, height).
            const bool arrayed_2d = dim == Dim_2D && tex_is_arrayed(binding);
            const bool is_2d_or_cube = (dim == Dim_2D && !arrayed_2d) || dim == Dim_Cube;
            const uint32_t size_type = is_2d_or_cube ? t_v2i() : t_v3i();
            uint32_t size = id(); put(code, Op_ImageQuerySizeLod, {size_type, size, img, bcs(lod_bits)});
            const uint32_t components = is_2d_or_cube ? 2u : 3u;
            for (uint32_t c = 0; c < components; c++) {
                uint32_t value = id(); put(code, Op_CompositeExtract, {t_i32, value, size, c});
                out[c] = i2u(value);
            }
        }
        uint32_t levels = id(); put(code, Op_ImageQueryLevels, {t_i32, levels, img});
        out[3] = i2u(levels);
    }

void SpirvCompute::image_get_lod_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t out[2]) {
        if (!declared_image_query) { put(caps, Op_Capability, {Cap_ImageQuery}); declared_image_query = true; }
        const uint32_t simg = tex_binding_simg[binding];
        uint32_t si = id(); put(code, Op_Load, {simg, si, tex_var[binding]});
        uint32_t coord = tex_coord_uv(binding, u_bits, v_bits);
        uint32_t lod = id(); put(code, Op_ImageQueryLod, {t_v2f(), lod, si, coord});
        for (uint32_t component = 0; component < 2; ++component) {
            uint32_t value = id();
            put(code, Op_CompositeExtract, {t_f32, value, lod, component});
            out[component] = bcu(value);
        }
    }

uint32_t SpirvCompute::t_v3u_fetch() {
        if (t_v3u) return t_v3u;   // compute/vertex shells already declare a uvec3 for built-ins
        if (!t_v3u_cache2) { t_v3u_cache2 = id(); put(types, Op_TypeVector, {t_v3u_cache2, t_u32, 3}); }
        return t_v3u_cache2;
    }

void SpirvCompute::image_fetch_2d(uint32_t binding, uint32_t x_bits, uint32_t y_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load,  {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t img   = id(); put(code, Op_Image, {tex_binding_img[binding], img, si});
        // #325: an Arrayed image needs a three-component fetch coordinate whatever opcode got
        // here. Layer 0 is the base slice a graphics IMAGE_LOAD used to read through the old
        // base-slice 2D view, so this preserves that behaviour rather than inventing one.
        uint32_t coord = id();
        if (tex_is_arrayed(binding))
            put(code, Op_CompositeConstruct, {t_v3u_fetch(), coord, x_bits, y_bits, uconst(0)});
        else
            put(code, Op_CompositeConstruct, {t_v2u(), coord, x_bits, y_bits});
        uint32_t res   = id(); put(code, Op_ImageFetch, {texture_vec4(binding), res, img, coord, ImgOp_Lod, uconst(0)});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_fetch_2d_lod(uint32_t binding, uint32_t x_bits, uint32_t y_bits,
                            uint32_t lod_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load,  {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t img   = id(); put(code, Op_Image, {tex_binding_img[binding], img, si});
        uint32_t coord = id();
        if (tex_is_arrayed(binding))
            put(code, Op_CompositeConstruct, {t_v3u_fetch(), coord, x_bits, y_bits, uconst(0)});
        else
            put(code, Op_CompositeConstruct, {t_v2u(), coord, x_bits, y_bits});
        uint32_t res   = id();
        put(code, Op_ImageFetch,
            {texture_vec4(binding), res, img, coord, ImgOp_Lod, lod_bits});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_fetch_2d_array(uint32_t binding, uint32_t x_bits, uint32_t y_bits,
                              uint32_t layer_bits, uint32_t out[4]) {
        uint32_t si    = id(); put(code, Op_Load, {tex_binding_simg[binding], si, tex_var[binding]});
        uint32_t img   = id(); put(code, Op_Image, {tex_binding_img[binding], img, si});
        uint32_t coord = id(); put(code, Op_CompositeConstruct,
                                   {t_v3u_fetch(), coord, x_bits, y_bits, layer_bits});
        uint32_t res   = id(); put(code, Op_ImageFetch,
                                   {texture_vec4(binding), res, img, coord,
                                    ImgOp_Lod, uconst(0)});
        unpack_texture_result(binding, res, out);
    }

void SpirvCompute::image_fetch_3d(uint32_t binding, uint32_t x_bits, uint32_t y_bits, uint32_t z_bits, uint32_t out[4]) {
        uint32_t simg  = tex_binding_simg[binding];
        uint32_t t_img3 = tex_binding_img[binding];   // OpImage's result type must be the pair's Image type
        uint32_t si    = id(); put(code, Op_Load,  {simg, si, tex_var[binding]});
        uint32_t img   = id(); put(code, Op_Image, {t_img3, img, si});
        uint32_t coord = id(); put(code, Op_CompositeConstruct, {t_v3u_fetch(), coord, x_bits, y_bits, z_bits});
        uint32_t res   = id(); put(code, Op_ImageFetch, {texture_vec4(binding), res, img, coord, ImgOp_Lod, uconst(0)});
        unpack_texture_result(binding, res, out);
    }

uint32_t SpirvCompute::stg_coord(uint32_t n, const uint32_t* c) {
        if (n == 1) return c[0];
        // 2D reuses the shared uvec2 helper (also used by texelFetch) so a shader mixing a texture
        // image_load and a 2-coord storage image doesn't emit a duplicate OpTypeVector %uint 2.
        if (n == 2) { uint32_t v = id(); put(code, Op_CompositeConstruct, {t_v2u(), v, c[0], c[1]}); return v; }
        // uvec3: reuse the compute shell's uvec3 (t_v3u, declared for gl_GlobalInvocationID) if present —
        // a second OpTypeVector %uint 3 would be an illegal duplicate non-aggregate type.
        if (!t_v3u_c) {
            if (t_v3u) t_v3u_c = t_v3u;
            else { t_v3u_c = id(); put(types, Op_TypeVector, {t_v3u_c, t_u32, 3}); }
        }
        uint32_t v = id(); put(code, Op_CompositeConstruct, {t_v3u_c, v, c[0], c[1], c[2]}); return v;
    }

uint32_t SpirvCompute::image_atomic_u32(uint16_t opcode, uint32_t binding, uint32_t ncoord,
                              const uint32_t* coords, uint32_t value, bool predicated,
                              uint32_t pred, uint32_t fallback) {
        // 2D and 2D_ARRAY R32_UINT atomics. `arrayed` is implied by ncoord==3 -- the gate below
        // only reaches here for SQ_DIM_2D (ncoord 2) and SQ_DIM_2D_ARRAY (ncoord 3, x/y/layer).
        const bool arrayed = ncoord == 3;
        if (ncoord != 2 && ncoord != 3) return fallback;
        if (!t_ptr_img_u32) {
            t_ptr_img_u32 = id();
            put(types, Op_TypePointer, {t_ptr_img_u32, SC_Image, t_u32});
        }
        if (!declared_image_query) {
            put(caps, Op_Capability, {Cap_ImageQuery});
            declared_image_query = true;
        }
        const uint32_t image = id();
        put(code, Op_Load, {stg_img_binding_type[binding], image, stg_img_var[binding]});
        const uint32_t size = id();
        // An arrayed image's OpImageQuerySize yields (width, height, layers); a plain 2D yields
        // (width, height). Querying the wrong arity is a SPIR-V validity error rather than a silent
        // miscompile, so spv_validate catches a mistake here.
        put(code, Op_ImageQuerySize, {arrayed ? t_v3i() : t_v2i(), size, image});
        const uint32_t width_i = id(), height_i = id();
        put(code, Op_CompositeExtract, {t_i32, width_i, size, 0});
        put(code, Op_CompositeExtract, {t_i32, height_i, size, 1});
        uint32_t in_bounds = ucmp(Op_ULessThan, coords[0], i2u(width_i));
        in_bounds = land(in_bounds, ucmp(Op_ULessThan, coords[1], i2u(height_i)));
        if (arrayed) {
            // The layer bound is NOT optional. Vulkan leaves an out-of-bounds image atomic
            // undefined -- robust image access does not cover atomics -- and the 2D case's own
            // comment records that RADV can spend seconds in one before resetting the GPU. An
            // unbounded layer index would be exactly that, so the layer is bounded from the image's
            // own query rather than from a descriptor field the shader cannot see.
            const uint32_t layers_i = id();
            put(code, Op_CompositeExtract, {t_i32, layers_i, size, 2});
            in_bounds = land(in_bounds, ucmp(Op_ULessThan, coords[2], i2u(layers_i)));
        }
        const uint32_t active = predicated ? land(pred, in_bounds) : in_bounds;
        auto emit = [&]() {
            const uint32_t coord = stg_coord(ncoord, coords);
            const uint32_t pointer = id();
            put(code, Op_ImageTexelPointer,
                {t_ptr_img_u32, pointer, stg_img_var[binding], coord, uconst(0)});
            const uint32_t result = id();
            put(code, opcode,
                {t_u32, result, pointer, uconst(Scope_Device),
                 uconst(MemSem_ImageAcqRel), value});
            return result;
        };
        const uint32_t entry = cur_block;
        const uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {active, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        const uint32_t result = emit();
        const uint32_t then_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        return emit_phi_2way(t_u32, result, then_end, fallback, entry);
    }

}  // namespace prosper::gpu
