// Pure textual relocation by tools/refactor/extract_case_routes.py; no new image semantics.
#include "gpu/recompiler/ir/lower/image_storage_routes.hpp"
#include "gpu/recompiler/ir/lower/image_store_values_lowering.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_mimg_store.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/storage_dst_sel.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
#include <cstdlib>
namespace prosper::gpu::ir {
// NOLINTBEGIN(concurrency-mt-unsafe): the existing cached diagnostic choice retains its timing.
bool ImageStorageRoutes::lower() {
    const bool is_ld = in.opcode == 0x00;
    const bool is_zero_mip_st = in.opcode == 0x09;
    const bool is_st = in.opcode == 0x08 || is_zero_mip_st;
    // The backend exposes one materialized mip, but that alone is not permission to
    // discard a guest operand. IMAGE_STORE_MIP is admitted only for the exact captured
    // 2D packet whose mip VGPR was proven zero at this use and whose descriptor declares
    // one uncompressed base level. The live dmask-f packet is exact to the captured
    // fmt60 Uint8x4 resource; other formats and dynamic/nonzero/general or DCC-backed
    // forms remain fail-visible.
    if (is_zero_mip_st &&
        (!rdna2_mimg_zero_mip_shape(in) || !res->proven_zero_mip || res->img_dim != SQ_DIM_2D ||
         res->sample_count != 1u || res->declared_mip_levels != 1u || res->in_mip_tail ||
         res->compression_enabled ||
         (in.mimg_dmask == 0xfu &&
          (res->format != DataFormat::Uint8 || res->num_components != 4u)))) {
        ok = false;
        return true;
    }
    // Opcodes verified with llvm-mc; add/positive-control matches the guest's own word0 (#2275).
    uint16_t spirv_atomic = 0;
    switch (in.opcode) {
        case 0x0f: spirv_atomic = Op_AtomicExchange; break;
        case 0x11: spirv_atomic = Op_AtomicIAdd; break;
        case 0x12: spirv_atomic = Op_AtomicISub; break;
        case 0x14: spirv_atomic = Op_AtomicSMin; break;
        case 0x15: spirv_atomic = Op_AtomicUMin; break;
        case 0x16: spirv_atomic = Op_AtomicSMax; break;
        case 0x17: spirv_atomic = Op_AtomicUMax; break;
        case 0x18: spirv_atomic = Op_AtomicAnd; break;
        case 0x19: spirv_atomic = Op_AtomicOr; break;
        case 0x1a: spirv_atomic = Op_AtomicXor; break;
        default: break;
    }
    const bool is_atomic = spirv_atomic != 0;
    if (!is_ld && !is_st && !is_atomic) {
        ok = false;
        return true;
    }
    uint32_t dim, ncoord;
    bool arrayed = false, ms = false;
    switch (
        in.mimg_dim) {   // SQ_RSRC dim -> SPIR-V Dim + coord count (+ array layer / MSAA sample)
        case 0:
            dim = Dim_1D;
            ncoord = 1;
            break;   // 1D
        case 1:
            dim = Dim_2D;
            ncoord = 2;
            break;   // 2D
        case 2:
            dim = Dim_3D;
            ncoord = 3;
            break;   // 3D
        case 4:
            dim = Dim_1D;
            ncoord = 2;
            arrayed = true;
            break;   // 1D_ARRAY (x, layer)
        case 5:
            dim = Dim_2D;
            ncoord = 3;
            arrayed = true;
            break;   // 2D_ARRAY (x, y, layer)
        case 6:
            dim = Dim_2D;
            ncoord = 2;
            ms = true;
            break;   // 2D_MSAA (x, y) + sample index
        case 7:
            dim = Dim_2D;
            ncoord = 3;
            arrayed = true;
            ms = true;
            break;   // 2D_MSAA_ARRAY (x,y,layer)+sample
        default: ok = false; return true;   // cube storage images deferred
    }
    if (ms && (is_st || is_atomic)) {
        ok = false;
        return true;
    }
    const uint32_t components = res->num_components ? res->num_components : 1;
    // The live Astro Bot visibility image is an ordinary 2D R32_UINT surface. Keep the
    // first atomic implementation exact and fail-visible for every other image shape;
    // atomics require a typed integer image in Vulkan/SPIR-V rather than Format=Unknown.
    // 2D and 2D_ARRAY. The array layer was added for #2265: Sonic Racing: CrossWorlds
    // issues IMAGE_ATOMIC_ADD at dim=2D_ARRAY on a default launch, and the storage
    // load/store siblings had already been generalised over the layer while the atomic
    // had not. `res->depth` is the layer COUNT for an arrayed view, so it is required to
    // be exactly 1 only in the non-arrayed case.
    const bool atomic_2d_array = in.mimg_dim == SQ_DIM_2D_ARRAY && arrayed && !ms;
    // #2265: the RESOURCE half is now shader_resource_supports_atomic_image_buffer,
    // shared with the descriptor validator and the backend materialization. Those three
    // had drifted -- #2272 widened this gate over the array layer and left the other
    // two on the single-layer clause, so this lowering emitted a StorageBuffer binding
    // that its own validator then rejected as WrongType and the dispatch was skipped
    // every frame. The INSTRUCTION half stays here, because only this site sees it.
    if (is_atomic && ((in.mimg_dim != SQ_DIM_2D && !atomic_2d_array) || ms || in.len_dwords < 2 ||
                      in.mimg_unorm || in.mimg_d16 || in.mimg_dmask != 1u ||
                      arrayed != (res->img_dim == SQ_DIM_2D_ARRAY) ||
                      !shader_resource_supports_atomic_image_buffer(*res))) {
        ok = false;
        return true;
    }
    // Same-binary control for the one-layer 2D descriptor / 2D_ARRAY instruction
    // extension. Sample once per process: changing the mode within a shader-cache
    // lifetime would otherwise reuse a module compiled under the other policy.
    static const bool single_layer_array_native_disabled =
        std::getenv("PROSPER_NO_NATIVE_SINGLE_LAYER_ARRAY_STORAGE") != nullptr;
    const bool single_layer_array_from_2d =
        dim == Dim_2D && arrayed && !ms && res->img_dim == 1 && res->depth == 1;
    const bool single_layer_array_native_allowed =
        !single_layer_array_from_2d || (b.is_compute && !single_layer_array_native_disabled);
    const bool native_2d_storage =
        shader_resource_uses_native_2d_storage_image(*res, dim == Dim_2D, arrayed, ms) &&
        single_layer_array_native_allowed;
    const bool native_uint_2d_storage =
        shader_resource_uses_native_uint_2d_storage_image(*res, dim == Dim_2D, arrayed, ms) &&
        single_layer_array_native_allowed;
    const bool ordinary_3d = in.mimg_dim == SQ_DIM_3D && res->img_dim == 2 && res->depth &&
                             !arrayed && !ms && !res->depth_compare;
    const uint32_t native_support_bit =
        ordinary_3d ? native_storage_3d_format_support_bit(res->format, components)
                    : native_storage_format_support_bit(res->format, components);
    // Vulkan permits B10G11R11 storage conversion, but RADV stores finite values with
    // a systematic downward bias while the guest conversion is round-to-nearest-even
    // (#1790). Keep the exact shader-side R32ui pack authoritative in ordinary/single-
    // layer-arrayed 2D and 3D, even when the device advertises native typed storage.
    // Other float formats retain the device-gated native path below.
    const bool packed_r11 = b.packed_r11_storage && (native_2d_storage || ordinary_3d) && !ms &&
                            !is_atomic && res->format == DataFormat::Float10_11_11 &&
                            components == 3;
    const bool native_float = !packed_r11 && (native_2d_storage || ordinary_3d) &&
                              native_float_storage_image_supported(
                                  res->format, components, res->srgb,
                                  (b.native_storage_format_support & native_support_bit) != 0);
    const bool native_uint = !packed_r11 && !is_atomic &&
                             (native_uint_2d_storage || (ordinary_3d && res->sample_count == 1)) &&
                             native_uint_storage_image_supported(
                                 res->format, components, res->srgb,
                                 (b.native_storage_format_support & native_support_bit) != 0);
    const bool compute_atomic_buffer = is_atomic && b.is_compute;
    if (compute_atomic_buffer) {
        if (!b.declare_compute_atomic_image_buffer(res->binding)) {
            ok = false;
            return true;
        }
    } else {
        const bool native_r32ui = is_atomic;
        const uint32_t native_uint_format = res->format == DataFormat::Uint32   ? ImgFmt_R32ui
                                            : res->format == DataFormat::Uint16 ? ImgFmt_R16ui
                                            : components == 4                   ? ImgFmt_Rgba8ui
                                                                                : ImgFmt_R8ui;
        b.declare_storage_image(res->binding, dim, arrayed, ms, native_float,
                                (native_r32ui || packed_r11) ? ImgFmt_R32ui
                                : native_uint                ? native_uint_format
                                                             : ImgFmt_Unknown,
                                packed_r11);
    }
    // Non-NSA coords follow VADDR; NSA coord0=VADDR, coord k>=1 is byte (k-1) of
    // words[2..3] (dword2=addr1..4, dword3=addr5..8), verified via llvm-mc gfx1010.
    auto coord_vgpr = [&](uint32_t k) { return mimg_address_vgpr(rs, in, k); };
    uint32_t coords[3] = {0, 0, 0};
    for (uint32_t k = 0; k < ncoord; k++) coords[k] = vread(coord_vgpr(k));
    uint32_t sample =
        ms ? vread(coord_vgpr(ncoord)) : 0;   // MSAA sample index = coord after the spatial ones
    if (is_ld) {
        uint32_t out[4];
        b.image_read(res->binding, dim, arrayed, ncoord, coords, out, ms, sample);
        // T# DST_SEL, forward (#4274): see storage_dst_sel.hpp.
        int channel[4];
        uint32_t constant[4];
        if (!storage_load_selects(res->swizzle, res->format, channel, constant)) {
            ok = false;
            return true;
        }
        uint32_t routed_out[4];
        ir::lower_image_storage_load_selects(b, out, channel, constant, routed_out);
        if (!write_mimg_results(routed_out)) {
            ok = false;
            return true;
        }
    } else if (is_st) {
        // image_store: gather the VDATA VGPRs selected by dmask into an RGBA texel (channels
        // absent from dmask store as 0). Under a narrowed EXEC (e.g. a grid-tail bounds check),
        // the write is EXEC-predicated so inactive lanes don't write out-of-range texels.
        uint32_t vals[4] = {b.uconst(0), b.uconst(0), b.uconst(0), b.uconst(0)};
        if (!mimg_store_components(b, in, d16_float_resource, uint_texture, vread, vals)) {
            ok = false;
            return true;
        }
        // T# DST_SEL, inverse (#4120): stored channel c receives the VDATA component whose
        // selector names c. See storage_dst_sel.hpp for the evidence and refusals.
        int source[4];
        if (!storage_store_sources(res->swizzle, components, source)) {
            ok = false;
            return true;
        }
        uint32_t routed[4];
        ir::lower_image_storage_store_selects(b, vals, source, routed);
        b.image_write(res->binding, dim, arrayed, ncoord, coords, routed, rs.exec_narrowed,
                      rs.exec);
    } else {
        // IMAGE_ATOMIC_SWAP/ADD reads its operand from VDATA. GLC=1 overwrites that VGPR
        // with the pre-operation texel; GLC=0 leaves VDATA unchanged. The helper's phi
        // preserves the old register value for EXEC-inactive lanes.
        const int vd = in.dst.value;
        const uint32_t old = vreg_old(b, rs, vd);
        const uint16_t atomic_op = spirv_atomic;
        uint32_t result;
        if (compute_atomic_buffer) {
            // #2265: the backend stages an arrayed view as layers tightly packed in
            // layer-major order (live_compute.cpp detiles each slice from its physical
            // stride into w*h*4 of linear), so the flat index is
            // (z*height + y)*width + x. The layer MUST be in the bound as well as the
            // index: Vulkan leaves an out-of-bounds image atomic undefined, robust
            // buffer access does not cover atomics, and the 2D path's own comment
            // records that RADV can spend seconds in one before resetting the device.
            //
            // `width`, `height` and `depth` are baked into the module as OpConstants,
            // which is only safe because all three are part of the shader cache key, so
            // a module compiled for one extent can never be reused against a smaller
            // one: `gpu_executor.cpp` sets `compiled.width`/`height` under
            // `atomic_extent` (an R32_UINT single-component storage image -- exactly
            // this case) and `compiled.depth`/`img_dim` under `storage_image`, and
            // `ShaderResourceCompileKey` has a defaulted member-wise `operator==`. Check
            // that still holds before baking a fourth quantity in here.
            const uint32_t row = arrayed ? b.ibin(Op_IAdd, coords[1],
                                                  b.ibin(Op_IMul, coords[2], b.uconst(res->height)))
                                         : coords[1];
            const uint32_t index =
                b.ibin(Op_IAdd, coords[0], b.ibin(Op_IMul, row, b.uconst(res->width)));
            uint32_t active = b.land(b.ucmp(Op_ULessThan, coords[0], b.uconst(res->width)),
                                     b.ucmp(Op_ULessThan, coords[1], b.uconst(res->height)));
            if (arrayed)
                active = b.land(active, b.ucmp(Op_ULessThan, coords[2],
                                               b.uconst(res->depth ? res->depth : 1u)));
            if (rs.exec_narrowed) active = b.land(rs.exec, active);
            result =
                b.cbuf_atomic_rtn(atomic_op, index, vread(vd), res->binding, true, active, old);
        } else {
            result = b.image_atomic_u32(atomic_op, res->binding, ncoord, coords, vread(vd),
                                        rs.exec_narrowed, rs.exec, old);
        }
        if (in.mimg_glc) rs.vreg[vd] = result;
    }
    return true;
}
// NOLINTEND(concurrency-mt-unsafe)
}   // namespace prosper::gpu::ir
