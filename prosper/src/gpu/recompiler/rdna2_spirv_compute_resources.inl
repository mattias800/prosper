#ifndef PROSPER_GPU_RECOMPILER_SPIRV_COMPUTE_MEMBER_SCOPE
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"   // IWYU pragma: export
#else
#ifndef PROSPER_GPU_RECOMPILER_RDNA2_SPIRV_COMPUTE_RESOURCES_INL
#define PROSPER_GPU_RECOMPILER_RDNA2_SPIRV_COMPUTE_RESOURCES_INL
// Class-scope fragment: preserve the mechanically moved member bytes.
// clang-format off

    // Combined image+sampler support (MIMG image_sample/image_load). Float and unsigned-integer
    // textures need distinct SPIR-V image types even at the same dimension: normalized/float T#s
    // return float bits, while UINT T#s return raw integer channel values. Each texture is still a
    // COMBINED_IMAGE_SAMPLER UniformConstant at its binding.
    std::unordered_map<uint32_t, uint32_t> tex_var;       // binding -> combined-sampler OpVariable id
    std::unordered_map<uint32_t, bool> tex_binding_arrayed;  // binding -> declared with Arrayed=1
    std::unordered_map<uint32_t, uint32_t> tex_simg_type; // (Dim | UINT flag) -> OpTypeSampledImage id
    std::unordered_map<uint32_t, uint32_t> tex_img_type;  // same key -> its OpTypeImage id
    std::unordered_map<uint32_t, uint32_t> tex_binding_simg;
    std::unordered_map<uint32_t, uint32_t> tex_binding_img;
    std::unordered_map<uint32_t, bool> tex_binding_uint;
    std::unordered_map<uint32_t, uint32_t> tex_binding_key;
    uint32_t t_v3f_cache = 0;
    uint32_t t_v3f() { if (!t_v3f_cache) { t_v3f_cache = id(); put(types, Op_TypeVector, {t_v3f_cache, t_f32, 3}); } return t_v3f_cache; }
    uint32_t t_v3i_cache = 0;
    uint32_t t_v3i() { if (!t_v3i_cache) { t_v3i_cache = id(); put(types, Op_TypeVector, {t_v3i_cache, t_i32, 3}); } return t_v3i_cache; }
    // Separate flags on purpose: the sampled path needs Sampled1D ONLY, so a shared flag would
    // let a sampled 1D image suppress Image1D for a storage 1D image declared afterwards.
    void declare_cap_sampled1d() {
        if (!declared_sampled1d) { put(caps, Op_Capability, {Cap_Sampled1D}); declared_sampled1d = true; }
    }
    void declare_cap_image1d() {
        if (!declared_image1d) { put(caps, Op_Capability, {Cap_Image1D}); declared_image1d = true; }
    }
    static uint32_t sampled_image_key(uint32_t dim, bool is_uint, bool arrayed, bool depth) {
        return dim | (is_uint ? 0x100u : 0u) | (arrayed ? 0x200u : 0u) |
               (depth ? 0x400u : 0u);
    }
    uint32_t sampled_image_type(uint32_t dim, bool is_uint = false, bool arrayed = false,
                                bool depth = false);
    // Declare (idempotently) a combined image+sampler of SPIR-V `dim` at descriptor-set 0, `binding`.
    bool declare_texture(uint32_t binding, uint32_t dim = Dim_2D, bool is_uint = false,
                         bool arrayed = false, bool depth = false);
    // The sampling coordinate for `binding`, with the array layer appended when the binding was
    // DECLARED arrayed. #325: BC arrays use the resource-wide T# shape, so a non-array
    // instruction reaching one gets layer zero. Layered Float32 graphics bindings may instead
    // declare a 2D base-slice view for an ordinary 2D instruction; the renderer follows that
    // declaration when choosing the actual view.
    bool tex_is_arrayed(uint32_t binding);
    uint32_t tex_coord_uv(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                          uint32_t layer_bits = 0);
    uint32_t texture_vec4(uint32_t binding) {
        return tex_binding_uint[binding] ? t_v4u() : t_v4f;
    }
    void unpack_texture_result(uint32_t binding, uint32_t result, uint32_t out[4]);
    // image_sample 2D: sample the combined sampler at `binding` with (u,v) float-BITS coords; fills
    // out[0..3] with the RGBA result components as raw VGPR bits. Implicit LOD is only legal in the
    // Fragment execution model (#151) — the compute/vertex shells have no derivatives, so there we
    // sample at explicit LOD 0 (what a non-pixel-shader image_sample resolves to without gradients).
    void image_sample_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t out[4]);
    // image_sample_d 2D: the guest supplies explicit normalized-coordinate derivatives in the
    // modifier-first vaddr slots. Preserve both vectors with the SPIR-V Grad image operand; using
    // implicit fragment-quad derivatives selects the wrong mip at seams and for uniform coordinates.
    void image_sample_grad_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                              uint32_t dsdx_bits, uint32_t dtdx_bits,
                              uint32_t dsdy_bits, uint32_t dtdy_bits, uint32_t out[4]);
    // image_sample 3D: (u,v,w) float-BITS coords -> RGBA. Uses the Dim_3D sampled image; same
    // implicit-LOD-only-in-Fragment rule as image_sample_2d.
    void image_sample_3d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t w_bits, uint32_t out[4]);
    // image_sample_l / _lz: sample with an EXPLICIT LOD (lod_bits float). Stage-agnostic (no derivatives).
    void image_sample_lod_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t lod_bits, uint32_t out[4]);
    // A real 2D-array explicit-LOD sample. The layer is the third float coordinate; keeping it in
    // SPIR-V makes reflection require a matching 2D-array view instead of silently sampling layer 0.
    // IMAGE_SAMPLE from a 2D_ARRAY texture. The implicit-LOD sibling of the _lod_ form below, and
    // the distinction is not cosmetic: in a fragment stage the LOD comes from quad derivatives, so
    // lowering a plain SAMPLE through the explicit-LOD helper would pin every textured surface to
    // the base level. Outside a fragment stage there are no derivatives, so LOD 0 is the only
    // legal choice -- which is exactly what image_sample_2d already does for the non-array case.
    // PROSPER_FORCE_LAYER=<n>: the constant array layer to substitute, or -1 for off. A malformed
    // or empty value DISABLES the probe rather than forcing layer 0 -- a typo must cost a
    // measurement, never produce a wrong one silently.
    // Announce a substitution, from whichever sampler performed it. Both must announce or the
    // probe produces exactly the null it exists to prevent: a title that samples only through the
    // explicit-LOD helper would otherwise get full substitution with no output at all.
    static void announce_forced_layer(uint32_t binding, int layer) {
        static std::atomic<uint32_t> said{0};
        if (said.fetch_add(1, std::memory_order_relaxed) < 4u) {
            fprintf(stderr, "[force-layer] binding=%u substituted constant layer %d\n",
                    binding, layer);
            fflush(stderr);
        }
    }
    static int forced_array_layer() {
        return static_cast<int>(compiler_choice(CompilerChoice::ForcedArrayLayer));
    }
    void image_sample_2d_array(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                               uint32_t layer_bits, uint32_t out[4]);
    void image_sample_lod_2d_array(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                   uint32_t layer_bits, uint32_t lod_bits, uint32_t out[4]);
    // image_sample_lz from a 3D texture: explicit LOD (usually 0) on a (u,v,w) coord. Stage-agnostic.
    void image_sample_lod_3d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t w_bits,
                             uint32_t lod_bits, uint32_t out[4]);
    // Compare one sampled/fetched depth value against DREF and return raw float bits containing 0/1.
    uint32_t image_dref_compare_bits(uint32_t depth, uint32_t dref_bits, uint32_t compare_func);
    uint32_t image_compare_address_coord(uint32_t value, uint32_t extent, uint32_t mode,
                                         uint32_t& outside_border);
    void image_compare_mix_taps(const bool& coord_arrayed, bool& arrayed, uint32_t& layer_bits,
                                uint32_t& img, uint32_t& dref_bits, uint32_t& compare_func,
                                const bool& uses_border, uint32_t& border_compare,
                                const uint32_t& x0, const uint32_t& y0, uint32_t& x0_border,
                                uint32_t& y0_border, const uint32_t& x1, uint32_t& x1_border,
                                const uint32_t& y1, uint32_t& y1_border, const uint32_t& fx,
                                const uint32_t& fy, uint32_t*& out);
    void image_compare_nearest(const bool& coord_arrayed, uint32_t& u_bits, uint32_t& v_bits,
                               bool& arrayed, uint32_t& layer_bits, uint32_t& si,
                               uint32_t& dref_bits, uint32_t& compare_func, uint32_t*& out);
    // IMAGE_SAMPLE_C_LZ on a plain 2D texture (or the backend's base-slice 2D_ARRAY fallback),
    // lowered as a MANUAL depth compare. The hardware path
    // (compareEnable sampler over a depth-format view) needs backend machinery the color-texture path
    // lacks (see the render-runner S# note). NEAREST retains the exact single sampled tap. LINEAR uses
    // four level-0 fetches, compares each independently, then bilinearly weights the booleans — the
    // compare-before-filter order required for 2x2 PCF. Vulkan compare semantics: result =
    // (reference OP stored); the SQ S# compare enum
    // (WORD0 [14:12]) matches VkCompareOp order — 0 NEVER, 1 LESS, 2 EQUAL, 3 LEQUAL, 4 GREATER,
    // 5 NOTEQUAL, 6 GEQUAL, 7 ALWAYS. CONFIDENCE: MED (standard AMD sampler enum ordering; Blue
    // Prince renders visually-correct shadows with it — the live A/B on #1271).
    void image_sample_dref_manual_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                     uint32_t dref_bits, uint32_t compare_func, bool linear_filter,
                                     uint32_t addr_u, uint32_t addr_v, uint32_t border_color_type,
                                     uint32_t out[4], bool arrayed = false,
                                     uint32_t layer_bits = 0);
    // image_sample_b 2D: implicit-LOD sample with an LOD BIAS (bias_bits float). Bias only means
    // anything with implicit LOD (fragment derivatives); outside the fragment stage the op resolves
    // like the other samples there — explicit LOD 0 (bias dropped, matching image_sample_2d's rule).
    void image_sample_bias_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t bias_bits, uint32_t out[4]);
    // image_gather4_lz 2D: OpImageGather of component `comp` (0..3) — the 2x2 footprint's four texels
    // of one channel, in the DX/GL gather order ((0,1),(1,1),(1,0),(0,0)), which the AMD gather4
    // result order matches. Gather always samples the base level (== the _lz behavior). out[0..3] =
    // the four gathered values as raw bits.
    void image_gather_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t comp,
                         uint32_t out[4], uint32_t layer_bits = 0);
    // image_gather4_lz_o 2D: gather with the MIMG _o per-pixel OFFSET operand. The offset VGPR packs
    // signed 6-bit TEXEL offsets (x = bits[5:0], y = bits[13:8] — AMD RDNA2 ISA "offset" packing, the
    // same fields image_sample_*_o uses). SPIR-V's dynamic Offset image operand requires the
    // ImageGatherExtended capability (the offset VGPR is not a compile-time constant in our SSA model,
    // even though shaders typically v_mov an immediate into it).
    bool declared_gather_ext = false;
    uint32_t t_v2i_cache = 0;
    uint32_t t_v2i() { if (!t_v2i_cache) { t_v2i_cache = id(); put(types, Op_TypeVector, {t_v2i_cache, t_i32, 2}); } return t_v2i_cache; }
    void image_gather_offset_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t comp,
                                uint32_t off_bits, uint32_t out[4], uint32_t layer_bits = 0);
    // image_sample_lz_o 2D: explicit-LOD-0 sample with the MIMG _o packed texel offset (x = bits[5:0],
    // y = bits[13:8], signed 6-bit — same packing as gather4_lz_o). Vulkan forbids the dynamic Offset
    // image operand on OpImageSample* (it is gather-only), so fold the texel offset into the
    // NORMALIZED coordinate instead: u' = u + ox/width, v' = v + oy/height (exact — the hardware adds
    // the integer offset to the unnormalized coordinate before filtering, and at LOD 0
    // (u + ox/W)*W == u*W + ox). The level-0 size comes from OpImageQuerySizeLod (Cap ImageQuery).
    bool declared_image_query = false;
    void image_sample_lz_offset_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                   uint32_t off_bits, uint32_t out[4]);
    // image_sample_o 2D on a ONE-level texture: the same level-0 fold, then an implicit-LOD sample
    // in a fragment stage (explicit LOD 0 elsewhere, as image_sample_2d). Only exact while the
    // texture has a single level -- the caller guarantees it (#4808).
    void image_sample_offset_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits,
                                uint32_t off_bits, uint32_t out[4]);
    // image_get_resinfo: query a sampled image's dimensions at the integer LOD carried in VADDR.
    // RDNA returns {width,height,depth-or-layers,mip-levels}; absent spatial axes are 1. The result is
    // integer data in ordinary VGPRs, so the signed SPIR-V query result is bitcast back to raw u32.
    void image_get_resinfo(uint32_t binding, uint32_t dim, uint32_t lod_bits, uint32_t out[4]);
    // image_get_lod: return the sampler-clamped and raw implicit LOD for a hypothetical 2D sample.
    // SPIR-V defines OpImageQueryLod's x/y results in the same order as RDNA2's VDATA[0]/[1], so the
    // two raw FP32 values can be copied directly to the guest VGPRs. Like the hardware instruction,
    // this consumes screen-space derivatives and is therefore fragment-only.
    void image_get_lod_2d(uint32_t binding, uint32_t u_bits, uint32_t v_bits, uint32_t out[2]);
    // 2-component uint vector (integer texel coordinates for OpImageFetch).
    uint32_t t_v2u_cache = 0;
    uint32_t t_v2u() { if (!t_v2u_cache) { t_v2u_cache = id(); put(types, Op_TypeVector, {t_v2u_cache, t_u32, 2}); } return t_v2u_cache; }
    uint32_t t_v3u_cache2 = 0;
    uint32_t t_v3u_fetch();
    // image_load 2D (image_load): texelFetch the image at the combined sampler's `binding` with INTEGER
    // (x,y) coords (raw VGPR bits). OpImage strips the sampler; OpImageFetch at explicit LOD 0.
    void image_fetch_2d(uint32_t binding, uint32_t x_bits, uint32_t y_bits, uint32_t out[4]);

    // Exact sampled IMAGE_LOAD representation for guest 2D_MSAA: the host image is single-sample
    // 2D-array, with each guest sample plane in one layer. The guest's explicit sample coordinate is
    // therefore the third integer coordinate; LOD remains zero because the host view has one level.
    // IMAGE_LOAD_MIP lowering: the guest's own mip selector reaches OpImageFetch's Lod operand.
    // Identical to image_fetch_2d except that the level is a value rather than the constant zero,
    // so it is valid only against a binding whose image really carries that many levels (#3048).
    void image_fetch_2d_lod(uint32_t binding, uint32_t x_bits, uint32_t y_bits,
                            uint32_t lod_bits, uint32_t out[4]);

    void image_fetch_2d_array(uint32_t binding, uint32_t x_bits, uint32_t y_bits,
                              uint32_t layer_bits, uint32_t out[4]);

    // image_load from a 3D texture (integer texel fetch through the combined sampler — DOLL's
    // color-grade LUT, #273): OpImage strips the sampler; OpImageFetch with (x,y,z) integer coords.
    void image_fetch_3d(uint32_t binding, uint32_t x_bits, uint32_t y_bits, uint32_t z_bits, uint32_t out[4]);

    // --- STORAGE images (MIMG image_load / image_store WITHOUT a sampler; compute copy/blit) ---
    // Integer/packed formats use a UINT-sampled OpTypeImage. The portable host path converts between
    // raw VGPR channels and guest texels; the exact R11G11B10 fallback instead packs one R32ui word
    // in the shader. Four-channel UNORM8/FLOAT16/FLOAT32 use a FLOAT-sampled image instead:
    // Vulkan then performs exactly the descriptor's normalized/float load-store conversion and the
    // host can stage the guest texels at their native width. VGPRs remain raw u32 bits; float image
    // components are bitcast at this boundary. Both paths use Format=Unknown and therefore require
    // the read/write-without-format capabilities.
    uint32_t t_v4u_cache = 0;
    uint32_t t_v4u() { if (!t_v4u_cache) { t_v4u_cache = id(); put(types, Op_TypeVector, {t_v4u_cache, t_u32, 4}); } return t_v4u_cache; }
    uint32_t t_v3u_c = 0;   // integer coordinate vector type (uvec3); 2D reuses the shared t_v2u()
    std::unordered_map<uint32_t, uint32_t> stg_img_type;   // dimension/type/format key -> OpTypeImage
    std::unordered_map<uint32_t, uint32_t> stg_img_binding_type; // binding -> declared OpTypeImage
    std::unordered_map<uint32_t, uint32_t> stg_img_var;    // binding -> storage-image OpVariable id
    std::unordered_map<uint32_t, bool> stg_img_float;      // binding -> float (rather than uint) texels
    std::unordered_map<uint32_t, uint32_t> stg_img_format; // binding -> SPIR-V Image Format
    std::unordered_map<uint32_t, bool> stg_img_packed_r11; // binding -> R11G11B10 packed in R32ui
    bool declared_read_wo_fmt = false, declared_write_wo_fmt = false,
         declared_storage_extended = false, declared_sampled1d = false,
         declared_image1d = false,
         declared_ms = false, declared_msarray = false;
    static uint32_t stg_key(uint32_t dim, bool arrayed, bool ms, bool float_texel,
                            uint32_t image_format) {
        return dim | (arrayed ? 0x100u : 0u) | (ms ? 0x200u : 0u) |
               (float_texel ? 0x400u : 0u) | (image_format << 11);
    }
    // Declare (idempotently) a storage image of SPIR-V `dim` (arrayed = layer in the coord; ms =
    // multisampled) at set 0, `binding`. Float and uint sampled types are keyed separately.
    void declare_storage_image(uint32_t binding, uint32_t dim, bool arrayed = false,
                               bool ms = false, bool float_texel = false,
                               uint32_t image_format = ImgFmt_Unknown, bool packed_r11 = false);
    // Build the integer coordinate operand from `n` raw-bit VGPR coords (u32 texel indices, incl. the
    // array layer as the last component for arrayed images). n=1 -> scalar u32; 2 -> uvec2; 3 -> uvec3.
    uint32_t stg_coord(uint32_t n, const uint32_t* c);
    // image_load: OpImageRead the storage image at `binding` (dim gives the coord count); fills out[0..3]
    // with the RGBA texel components as raw VGPR bits (uint sampled type -> no bitcast needed).
    // OOB CONTRACT: like a buffer load (which relies on robustBufferAccess), the read is issued for ALL
    // invocations — including EXEC-inactive lanes whose coordinate may be out of range (e.g. a grid-tail
    // bounds check narrowed EXEC). The loaded value for such a lane is discarded by write-back predication
    // (predicate_write) at the call site, so it is harmless — PROVIDED the device enables IMAGE ROBUSTNESS
    // (robustImageAccess / VK_EXT_image_robustness: OOB image reads return 0), the image analogue of the
    // robustBufferAccess this recompiler already depends on for buffer loads. Every device prosper
    // executes this on acquires it through frontends/shared/device/image_robustness.hpp, and a device
    // that CANNOT declines the storage-image path rather than running it undefined (#3531 -- until
    // then the compute backend's own device silently did neither). The store side is instead
    // EXEC-predicated (no robust "harmless" OOB write).
    void image_read(uint32_t binding, uint32_t dim, bool arrayed, uint32_t ncoord,
                    const uint32_t* coords, uint32_t out[4], bool ms = false, uint32_t sample = 0);
    // image_store: OpImageWrite raw-bit VGPR components vals[0..3] as a uvec4 texel to the storage image.
    // When `predicated` (narrowed EXEC), the write is wrapped in a selection merge on `pred` (the per-lane
    // EXEC bool) so inactive lanes do not write — a real conditional store, like cbuf_store. (Image OOB is
    // not covered by robustBufferAccess, so guarding matters: it also skips a lane's write when EXEC is off
    // e.g. a grid-tail bounds check.)
    void image_write(uint32_t binding, uint32_t dim, bool arrayed, uint32_t ncoord,
                     const uint32_t* coords, const uint32_t vals[4], bool predicated = false,
                     uint32_t pred = 0);

    // An integer atomic on an R32_UINT 2D storage image. OpImageTexelPointer consumes the descriptor
    // variable directly (not a loaded image object) and yields an Image-storage pointer suitable for
    // the SPIR-V atomic. Vulkan leaves an out-of-bounds image atomic undefined (robust image access
    // does not cover atomics), and RADV can spend seconds in such an atomic before resetting the GPU.
    // Query the bound view and put BOTH the texel pointer and atomic behind an explicit bounds/EXEC
    // branch. A skipped lane leaves VDATA unchanged, matching the no-operation EXEC fallback.
    uint32_t image_atomic_u32(uint16_t opcode, uint32_t binding, uint32_t ncoord,
                              const uint32_t* coords, uint32_t value, bool predicated,
                              uint32_t pred, uint32_t fallback);

    // buffer element pointer: base[ gid.x*stride + k ]
    uint32_t elem_ptr(uint32_t bufvar, uint32_t k);
    // Load one float from the input buffer and return it as raw bits (VGPR value).
    uint32_t load_input(uint32_t k) {
        uint32_t p = elem_ptr(v_in, k), r = id();
        put(code, Op_Load, {raw_input_words ? t_u32 : t_f32, r, p});
        return raw_input_words ? r : bcu(r);
    }
    uint32_t load_owned_packet_word(uint32_t absolute_word) {
        // This entry owns raw-u32 binding0. A float cross-bitcast/native input envelope would be a
        // different contract; do not make this a generic buffer helper with a fallback binding.
        uint32_t pointer = id(), result = id();
        put(code, Op_AccessChain, {t_ptr_sb_f32, pointer, v_in, uconst(0), absolute_word});
        put(code, Op_Load, {t_u32, result, pointer});
        return result;
    }
    // The storage-buffer variable for a descriptor `binding`. N-buffer model: each distinct constant/
    // vertex buffer the shader reads is bound at its own descriptor binding (the executor assigns them),
    // so multiple constant buffers (e.g. Unity's per-draw transform vs per-frame) don't collapse onto one.
    // Bindings 2/3 keep mapping to v_cbuf/v_cbuf1 for the compute-shell + legacy 2-slot callers; any other
    // binding uses cbuf_var[] (declared by declare_cbufs from the resource table). Falls back to v_cbuf.
    uint32_t buf_for_binding(uint32_t binding);
    // Logical-addressing SPIR-V cannot bitcast a pointer into the ordinary runtime-u32 Block to a
    // u64 pointer. Declare a second, aliased StorageBuffer variable at the SAME descriptor binding.
    // Vulkan binds both declarations to one VkBuffer; ArrayStride=8 gives the qword atomics their
    // natural record index. The narrow caller excludes descriptor arrays.
    uint32_t u64_buf_for_binding(uint32_t binding);
    void mark_cbuf_coherent(uint32_t binding);
    void device_uniform_release_barrier();
    // Load one dword (raw bits) from the constant/vertex buffer at descriptor `binding` at dword index
    // `idx` (SMEM). The 1-arg form keeps the legacy slot convention (0 -> binding 2, 1 -> binding 3).
    // THE only place a storage-buffer element pointer is built. There were FOUR identical
    // OpAccessChain sites -- cbuf_load_impl, cbuf_store's unpredicated and predicated branches, and
    // cbuf_atomic_rtn -- and a table-indexed binding needs the leading selector in every one of them.
    //
    // #2474 claimed cbuf_load_impl was "the single place a buffer access chain is built" and used that
    // as its correctness argument. It was false, and the review that caught it found three sites; the
    // fourth turned up while fixing it. For a table-indexed binding the pointee is `OpTypeArray %Block N`,
    // so an unfixed site does not merely read the wrong slot: `uconst(0)` consumes the ARRAY index and
    // `idx` is then a member index into the Block, which has one member. So idx==0 yields a pointer to
    // the runtime array against a declared `t_ptr_sb_u32` (result-type mismatch) and idx!=0 is
    // out-of-range -- an invalid module either way. Loud rather than silent, but only where something
    // validates; on this path nothing does yet, so it surfaces as a failed pipeline and a dropped draw.
    //
    // Hence one function: the sites cannot drift apart again, and a fifth caller inherits the selector
    // rather than a stale comment promising there is only one.
    uint32_t cbuf_element_ptr(uint32_t buf, uint32_t binding, uint32_t idx,
                              uint32_t* array_valid = nullptr);
    uint32_t cbuf_load_impl(uint32_t idx, uint32_t binding, bool coherent_access);
    uint32_t cbuf_load(uint32_t idx, uint32_t binding = 2, bool coherent_access = false);
    struct U64PairAdd {
        uint32_t lo = 0;
        uint32_t hi = 0;
        uint32_t overflow = 0;
    };
    U64PairAdd add_u64_pair_u32(uint32_t lo, uint32_t hi, uint32_t addend);
    uint32_t u64_pair_ule(uint32_t lhs_lo, uint32_t lhs_hi,
                          uint32_t rhs_lo, uint32_t rhs_hi);
    void declare_indirect_pointer_descriptor_capture();
    void capture_indirect_pointer_descriptor_source(
            uint32_t record_index, uint32_t root_lo, uint32_t descriptor_word1,
            bool predicated, uint32_t predicate);
    uint32_t relocated_indirect_carrier_dword(uint32_t selected_byte,
                                               uint32_t valid);
    // Relocate one proven guest GLOBAL dword through the version-2 carrier. Segment byte_count is
    // the exact guest interval; carrier_bytes includes only the physical zero padding needed for a
    // safe unaligned dword join. An invalid, ambiguous, overflowing, or padding-only address first
    // selects carrier byte zero for memory safety and then returns architectural zero.
    uint32_t relocated_indirect_load_dword(uint32_t address_lo, uint32_t address_hi,
                                            uint32_t immediate_byte_offset);
    // DescriptorRange differs from StaticFootprint in one essential way: adjacent source records
    // may describe adjacent or overlapping guest intervals, but an address derived from record A
    // must never borrow record B's authority. Match the captured producer identity against one
    // record-directory entry first, prove the final dword lies in that exact record, and only then
    // translate it through the segment named by that record. All arithmetic is an explicit u32 pair;
    // this path intentionally does not require ShaderInt64.
    uint32_t relocated_indirect_descriptor_load_dword(
            uint32_t address_lo, uint32_t address_hi,
            uint32_t immediate_byte_offset);
    uint32_t cbuf_load_zero_padded_tail(uint32_t binding,
                                        StorageBufferTailSemantic semantic,
                                        bool coherent_access = false) {
        auto [candidate, inserted] = cbuf_zero_pad_candidates.emplace(binding, semantic);
        if (!inserted && candidate->second != semantic)
            cbuf_ordinary_accesses.insert(binding);
        return cbuf_load_impl(uconst(0), binding, coherent_access);
    }
    // Store one dword `value` to the buffer at descriptor `binding` at dword index `idx` (MUBUF store).
    // When `predicated`, the store is wrapped in a selection merge on `pred` (the per-lane EXEC bool) so
    // inactive lanes do not write — a real conditional store, not a select of a loaded old value.
    void cbuf_store(uint32_t idx, uint32_t value, uint32_t binding, bool predicated, uint32_t pred,
                    bool coherent_access = false);
    // Returning atomic RMW on a descriptor-backed storage buffer. BUFFER_ATOMIC_* writes memory and
    // returns the pre-operation value in VDATA. Inactive EXEC lanes neither access the buffer nor
    // clobber VDATA, hence the predicated path joins the old destination through OpPhi.
    uint32_t cbuf_atomic_rtn(uint32_t op, uint32_t idx, uint32_t value, uint32_t binding,
                             bool predicated, uint32_t pred, uint32_t fallback);
    // RDNA2 BUFFER_ATOMIC_FMIN/FMAX and DS_MIN/MAX_F32 use floating min/max bit semantics, but
    // operate on memory for which SPIR-V 1.3 has no core floating-point min/max atomic. Keep the CAS
    // loop and all selection arithmetic in integer space so host-driver float controls cannot alter
    // the guest result. Signaling NaNs are quieted and propagated before ordinary MINNUM/MAXNUM
    // selection. COMPUTE_PGM_RSRC1.FP32_DENORM controls whether subnormal operands compare as
    // signed zero. MIN/MAX still publishes the selected operand's original, non-flushed bits.
    uint32_t atomic_fminmax_bits(uint32_t resident, uint32_t value, bool is_min);
    uint32_t cbuf_atomic_fminmax_rtn(uint32_t idx, uint32_t value, uint32_t binding,
                                     bool is_min, bool predicated, uint32_t pred,
                                     uint32_t fallback);
    // One true 64-bit RMW for the exact naturally-strided guest contract. The caller supplies the
    // SPIR-V atomic opcode and original qword record index, and range-checks it before entry.
    uint32_t cbuf_atomic_x2_rtn(uint32_t op, uint32_t index, uint32_t value, uint32_t binding,
                                uint32_t pred, uint32_t fallback);
    // RADV currently hangs/reset-poisons the device on Astro Bot's compute R32_UINT image atomic.
    // Compute lowers that exact 2D resource through a detiled storage-buffer view instead. The live
    // backend recognizes the reflected atomic buffer over a StorageImage resource, detiles before
    // dispatch, and tiles the result back afterwards. Graphics retains the native image-atomic path.
    // Bindings this function itself declared, so a REPEAT use of one is a hit rather than a
    // failure. A shader with two image atomics on the same image called here twice: the first
    // declared cbuf_var[binding], and the second was refused by the `cbuf_var.count(binding)` guard
    // -- which rejected the whole shader for the crime of using its own image twice. Measured on
    // Sonic Racing: CrossWorlds, where the resource table has NO entry at the binding in question,
    // so declare_cbufs cannot have been the declarer (#2265).
    //
    // Only OUR bindings are reusable. A binding declared by declare_cbufs from a
    // ConstantBuffer/VertexBuffer resource is still refused: the types happen to match, which is
    // exactly what would make aliasing onto someone else's buffer silent rather than loud.
    std::set<uint32_t> atomic_img_buf_bindings;
    bool declare_compute_atomic_image_buffer(uint32_t binding);
// clang-format on
#endif
#endif
