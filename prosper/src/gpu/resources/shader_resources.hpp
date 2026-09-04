// shader_resources.hpp — the resource-binding CONTRACT between the front-half (which knows the game's
// real GPU resources) and the recompiler back-half (which must translate the shader's memory ops
// correctly). This is the seam that unblocks the format-dependent memory instructions:
//   * s_buffer_load_*   — uniforms / constant buffers (multi-buffer, beyond the single-cbuf model)
//   * buffer_load_format_* (MUBUF) — vertex attribute fetch (needs the descriptor data format)
//   * tbuffer_load/store_format_* (MTBUF) — typed buffers (the instruction supplies its data format)
//   * image_sample / image_load (MIMG) — texture reads (needs texture format + sampler)
//
// Why a contract: to translate `buffer_load_format_xyzw` the recompiler must know the attribute's
// DATA FORMAT (float32 vs unorm8 vs …) to emit the right conversion — and that format lives in the
// V#/T# descriptor the game builds, which the FRONT-HALF can read from the shader's user_data / SRT
// and the game's bound resources. So the recompiler is *parameterized* by a ShaderResourceTable the
// front-half fills. See docs/RESOURCE_BINDING.md for the model, the descriptor-provenance mechanism,
// and the staged implementation plan.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace prosper::gpu {

// Element data format decoded from gfx1030's combined V#/T#/MTBUF format fields. Determines the
// conversion a format load / texture sample must emit. `Float32` is the common case
// for positions and most attributes — and for our raw-32-bit-VGPR model it is a *no-op reinterpret*,
// so float32 format loads reduce to raw dword loads. The rest require real conversion.
enum class DataFormat : uint32_t {
    Unknown = 0,
    Float32, Uint32, Sint32,
    Float16, Unorm16, Snorm16, Uint16, Sint16,
    Unorm8,  Snorm8,  Uint8,  Sint8,
    // Block-compressed texture formats (4x4-texel blocks). BC1/2/3/4/5/7 are decoded to RGBA8 on
    // upload (bc_decode, #121/#290); BC6H (HDR) and the SNORM BC4/5 variants are recognized but
    // still skipped (gen5_image_format flags them). data_format_bytes() returns 0 for these (a
    // per-COMPONENT byte size is meaningless for a block format; use
    // Gen5ImageFormatInfo::bytes_per_block).
    Bc1, Bc2, Bc3, Bc4, Bc5, Bc6, Bc7,
    // Packed 32-bit R11G11B10 unsigned-float (GFX10 IMG_FMT 36 "10_11_11_FLOAT", DX
    // R11G11B10_FLOAT, VK B10G11R11_UFLOAT_PACK32): R = bits[10:0] (5e6m), G = bits[21:11]
    // (5e6m), B = bits[31:22] (5e5m), no sign bits, no alpha. UE4's scene-color render-target
    // format (#294). data_format_bytes() returns 0 (packed — per-component size is meaningless;
    // the texel is 4 bytes, carried by Gen5ImageFormatInfo::bytes_per_block).
    Float10_11_11,
    // Packed 32-bit R10G10B10A2 UNORM. GFX10 names its high-to-low storage layout
    // "2_10_10_10_UNORM" (IMG_FMT 50); logical R/G/B occupy the low/middle 10-bit fields and
    // alpha occupies the high 2 bits. Sampled uploads unpack it to RGBA8.
    Unorm2_10_10_10,
    // Vertex-fetch variants of the same packed 32-bit R10G10B10A2 layout. These stay distinct so
    // buffer_load_format_* can apply the descriptor's signed/normalized conversion contract while
    // texture uploads keep their existing Unorm2_10_10_10 path. Per-component byte size is 0 for
    // every packed-word format; the fetch recompiler handles their bit fields explicitly.
    Snorm2_10_10_10,
    Uint2_10_10_10,
    Sint2_10_10_10,
    // Integer fields converted to floating-point without normalization. GFX10's USCALED/SSCALED
    // buffer formats are valid vertex/typed-buffer inputs (for example 8_8_USCALED format 16), but
    // are not sampled-image formats in Prosper's current upload contract. Keep these at the end so
    // existing capture enum values remain stable.
    Uscaled8, Sscaled8, Uscaled16, Sscaled16,
};

// Address source for a const-fold-resolved graphics buffer fetch. Automatic preserves the legacy
// recompiler heuristic; the other values are proofs produced by the dynamic fetch walk. In an NGG
// merged shader the same MUBUF VADDR register can be selected from vertex_id or instance_id by a
// wave-uniform v_cndmask, so replacing every such fetch with gl_VertexIndex is not equivalent.
enum class VertexFetchIndexMode : uint32_t {
    Automatic = 0,
    Shader = 1,
    Vertex = 2,
    Instance = 3,
};

// How a descriptor-array element is selected. Keep the selector shape separate from the declared
// array arity: `table_index_count` describes the Vulkan binding, while this enum describes the
// guest value that chooses one element at execution time.
enum class BufferTableSelectorMode : uint32_t {
    None = 0,
    UserSgprIndex = 1,
    DynamicSbufferByteOffset = 2,
};

// One concrete V# in a runtime-selected buffer descriptor table. The raw words are the input-side
// descriptor identity; the normalized fields are the exact backing contract used by capture,
// dependency analysis, and the Vulkan backend. Replay fills host_data from the entry's captured
// blob while preserving gpu_addr as its logical guest identity.
struct ShaderBufferTableEntry {
    std::array<uint32_t, 4> vsharp{};
    uint64_t gpu_addr = 0;
    uint32_t size = 0;
    uint32_t stride = 0;
    uint8_t* host_data = nullptr;
    uint64_t host_data_size = 0;
};

// Decode an RDNA2 (GFX10/PS5) combined seven-bit buffer FORMAT field, used by both V# descriptors
// and MTBUF instructions, into the recompiler's data-format contract. Unknown values stay explicit.
void rdna2_buffer_format(uint32_t fmt, DataFormat* out_fmt, uint32_t* out_components);

// How many bytes one component of `format` occupies (0 for Unknown and block-compressed formats).
uint32_t data_format_bytes(DataFormat f);

// Formats whose storage-image conversion can be delegated exactly to a native Vulkan float image.
// Loads return float32 components (with Vulkan's standard missing-channel defaults) and stores accept
// float32 components, matching the PS5 image instruction's VGPR contract; the backing texels remain
// at their native byte width. Three-channel optimal images are deliberately limited to packed
// R11G11B10 because ordinary RGB8/RGB16/RGB32 storage support is not portable.
constexpr bool native_float_storage_image(DataFormat format, uint32_t components, bool srgb) {
    return !srgb &&
           (((components == 1 || components == 2 || components == 4) &&
             (format == DataFormat::Unorm8 || format == DataFormat::Float16 ||
              format == DataFormat::Float32)) ||
            (components == 3 && format == DataFormat::Float10_11_11) ||
            (components == 4 && format == DataFormat::Unorm2_10_10_10));
}

// Native typed storage is only valid when the physical device advertises storage-image support
// for the selected VkFormat. Keep the feature input abstract so the fallback policy can be tested
// without requiring a particular Vulkan device or exposing Vulkan types in this shared header.
constexpr bool native_float_storage_image_supported(DataFormat format, uint32_t components,
                                                    bool srgb, bool storage_image_feature) {
    return storage_image_feature && native_float_storage_image(format, components, srgb);
}

// Narrow unsigned integer storage has the same exact byte contract as the guest surface:
// Vulkan zero-extends narrow texels on load and discards the high bits on store. Keeping these
// images typed avoids the portable RGBA32_UINT interchange representation (16 bytes per texel)
// without changing any shader-visible value. The 16-bit arm is important for GTA V's layered depth
// copies: the raw fallback expands a six-face 512x512 Uint16 cube from 3 MiB to 24 MiB for every
// single-face dispatch. RGBA8_UINT is equally exact and avoids expanding GTA V's 4K transition
// target from 33 MiB to 127 MiB. Keep integer 3D separate so no unqueried dimensional contract is
// implied.
constexpr bool native_uint_storage_image(DataFormat format, uint32_t components, bool srgb) {
    return !srgb &&
           ((components == 1 &&
             (format == DataFormat::Uint32 || format == DataFormat::Uint16 ||
              format == DataFormat::Uint8)) ||
            (components == 4 && format == DataFormat::Uint8));
}

constexpr bool native_uint_storage_image_supported(DataFormat format, uint32_t components,
                                                   bool srgb, bool storage_image_feature) {
    return storage_image_feature && native_uint_storage_image(format, components, srgb);
}

// Stable, Vulkan-independent bits used to carry per-format storage-image capabilities from the
// device-owning frontend into the compute recompiler. Zero means the semantic format is not a native
// typed-storage candidate; otherwise each exact VkFormat candidate has its own bit.
constexpr uint32_t native_storage_format_support_bit(DataFormat format, uint32_t components) {
    if (format == DataFormat::Unorm8)
        return components == 1 ? 1u << 0 : components == 2 ? 1u << 1
             : components == 4 ? 1u << 2 : 0u;
    if (format == DataFormat::Float16)
        return components == 1 ? 1u << 3 : components == 2 ? 1u << 4
             : components == 4 ? 1u << 5 : 0u;
    if (format == DataFormat::Float32)
        return components == 1 ? 1u << 6 : components == 2 ? 1u << 7
             : components == 4 ? 1u << 8 : 0u;
    if (format == DataFormat::Float10_11_11 && components == 3) return 1u << 9;
    // Bits 10..19 retain their capture-stable meaning as the 3D mirrors of bits 0..9.
    if (format == DataFormat::Uint32 && components == 1) return 1u << 20;
    if (format == DataFormat::Uint8 && components == 1) return 1u << 21;
    if (format == DataFormat::Uint16 && components == 1) return 1u << 22;
    if (format == DataFormat::Uint8 && components == 4) return 1u << 23;
    if (format == DataFormat::Unorm2_10_10_10 && components == 4) return 1u << 24;
    return 0;
}

// 3D optimal images have a dimension-specific Vulkan support query. Keep their capabilities in
// the upper half of the same replay-stable mask so a device that supports typed 2D storage but not
// the corresponding 3D image never compiles a shader the live backend cannot bind.
constexpr uint32_t native_storage_3d_format_support_bit(DataFormat format,
                                                        uint32_t components) {
    const uint32_t format_bit = native_storage_format_support_bit(format, components);
    return (format_bit & ((1u << 10) - 1u)) ? format_bit << 10 : 0u;
}

constexpr uint32_t kNativeStorageFormatSupportMask = (1u << 25) - 1u;

// IEEE-754 binary16 -> binary32 (handles subnormals, +/-inf, NaN). Used by the texture upload path to
// convert a sampled Float16 surface to the RGBA8 the backend uploads (#290). Pure + testable.
float half_to_float(uint16_t h);

// IEEE-754 binary32 -> binary16, round-to-nearest-even. This is the inverse conversion needed when
// a format-free storage-image write targets an R16_FLOAT/R16G16B16A16_FLOAT guest surface.
uint16_t float_to_half(float f);

// Narrow one normalized 16-bit channel to the RGBA8 upload path with nearest-value scaling. Reading
// only one byte is not an approximation: it makes a monotonic R16 gradient wrap every 256 source
// values, which appears as repeated contour bands in sampled attenuation ramps (#1186).
uint8_t unorm16_to_unorm8(uint16_t value);

// Unsigned small-float components of a packed Float10_11_11 texel (#294). Both share binary16's
// 5-bit exponent (bias 15) with a shortened mantissa (6 bits for the 11-bit R/G, 5 for the 10-bit B)
// and NO sign bit — so a left-shift of the mantissa into a half's 10-bit field is an exact
// widening (subnormals scale identically, inf/NaN preserved). Pure + testable.
float f11_to_float(uint16_t v);   // low 11 bits used
float f10_to_float(uint16_t v);   // low 10 bits used
uint16_t float_to_f11(float f);   // negative clamps to zero; round-to-nearest-even
uint16_t float_to_f10(float f);   // negative clamps to zero; round-to-nearest-even

// GFX10_FORMAT_2_10_10_10_UNORM packed texel -> RGBA8, with nearest UNORM scaling. Pure + testable.
void unorm2_10_10_10_to_rgba8(uint32_t packed, uint8_t rgba[4]);

enum class ResourceClass : uint32_t {
    ConstantBuffer,  // read by s_buffer_load_* (scalar, uniform across the wave)
    VertexBuffer,    // read by buffer_load_format_* (per-lane attribute fetch)
    Texture,         // read by image_sample / image_load (sampled image + sampler)
    Sampler,         // paired with a Texture for image_sample
    StorageImage,    // read/written by image_load / image_store WITHOUT a sampler (compute copy/blit).
                     // Bound as a Vulkan STORAGE_IMAGE; img_dim gives 1D/2D/3D. Integer/packed formats
                     // use uint texels plus host conversion; native four-channel float/UNORM formats
                     // use float texels and Vulkan's descriptor conversion.
};

// Derived, dispatch-owned binding for a generic indirect-pointer relocation proof. The scalar
// marker is cache/capture metadata only: authority is re-derived from the raw shader, launch, source
// records, and version-2 carrier at every compile boundary.
struct IndirectPointerRelocationBinding {
    uint32_t carrier_version = 0;
    uint32_t proof_schema = 0;
    uint32_t binding_bytes = 0;
    uint32_t record_count = 0;
    uint32_t segment_count = 0;
    uint32_t segment_directory_byte_offset = 0;
    uint64_t proof_fingerprint = 0;

    bool operator==(const IndirectPointerRelocationBinding&) const = default;
};

// One resource a shader accesses. FILLED BY THE FRONT-HALF from the game's real descriptors (the
// V#/T#/S# words in the shader's user_data / SRT, resolved against the game's bound resources and
// guest memory). CONSUMED BY THE RECOMPILER: it uses `format`/`num_components` to emit correct
// conversions and records `binding`; and BY THE PIPELINE: it binds `size` bytes at `gpu_addr`
// (unified guest memory) to descriptor-set 0, `binding`.
struct ShaderResource {
    ResourceClass cls           = ResourceClass::ConstantBuffer;
    DataFormat    format        = DataFormat::Float32;   // for format loads / textures
    uint32_t      num_components = 1;                    // 1..4
    uint32_t      binding       = 0;                     // Vulkan descriptor-set-0 binding

    uint64_t      gpu_addr      = 0;                     // base of the backing bytes in guest memory
    uint32_t      size          = 0;                     // byte size of the backing region
    uint32_t      stride        = 0;                     // element stride (vertex/structured buffers)

    // Descriptor identity — how the recompiler maps a memory op back to this resource. There are two
    // provenance modes (see RESOURCE_BINDING.md); a resource sets whichever matches how the shader
    // gets its descriptor, leaving the other 0xFFFFFFFF:
    //   * srt_offset — INDIRECT: the shader loads the V# with `s_load_dwordx4` from this byte offset
    //     within its user_data/SRT. The recompiler tags the load's dest SGPRs and resolves by offset.
    //   * sgpr_base  — DIRECT: the driver places the V# straight in the user-data SGPRs starting at
    //     this SGPR index (Sony "direct" resources — e.g. vertex-buffer descriptors). The recompiler
    //     resolves a memory op by matching its SRSRC/SBASE SGPR to this index (no in-shader load).
    uint32_t      srt_offset    = 0xFFFFFFFFu;
    uint32_t      sgpr_base     = 0xFFFFFFFFu;

    // TABLE-INDEXED provenance (#2412, stage 2 of the runtime-selected-descriptor lift). A fifth,
    // deliberately SEPARATE shape rather than a variation on the four above.
    //
    // The others all answer "where did this descriptor come from" -- an SRT offset, an SGPR index, a
    // fetch pc. This one cannot: the shader loads a descriptor from a table at an index that is only
    // known on the GPU (GTA V derives it from EXEC after `s_and_saveexec_b64`), so the descriptor is
    // identified by *a table plus a runtime index*, not by an origin. Folding it into `srt_offset`
    // would be the tempting shortcut and would erase exactly the distinction later validation has to
    // see: `srt_offset` promises "the descriptor is AT this offset", while this promises "the
    // descriptor is one OF these, selected later".
    //
    // `table_index_count` is the number of array elements the binding declares -- 0 means this resource
    // is not table-indexed and the stride below is inert, so existing resources are unaffected by
    // construction. When non-zero, `binding` names the ARRAY and the shader supplies the element.
    //
    // Producers must supply the complete concrete payload below. Reflection, validation, SPIR-V
    // emission and the live compute backend all agree on this exact arity; a partial lift is rejected
    // rather than silently binding element zero.
    uint32_t      table_index_count = 0;

    // Byte stride between consecutive descriptors in the guest table. 16 for a V#, 32 for a T#; kept
    // explicit rather than derived from `cls` because the guest chooses the packing and a table of
    // V#s addressed with a 32-byte stride is a real shape (padded slots) that must not be silently
    // re-packed.
    uint32_t      table_entry_stride = 0;

    // Which USER SGPR carries the descriptor index for a table-indexed binding, or 0xFFFFFFFF when the
    // index is not available as a user SGPR (#2412 stage 4c). Same shape as the flat-load path's
    // `flat_base_sgpr`, and resolved the same way: the emitter reads it through a push constant.
    //
    // This covers the case where the guest passes the index in user data. It deliberately does NOT
    // cover GTA V's own case, where the index is derived from EXEC on the GPU
    // (`s_mov_b64 vcc, exec` -> `s_buffer_load_dwordx4 ..., vcc_lo`) and therefore has no user-SGPR
    // home -- that needs the const-fold to name the register holding the computed value, which is the
    // remaining step. Keeping the two apart matters: a wrong index reads a valid descriptor from the
    // wrong slot, which renders confidently wrong content rather than failing.
    uint32_t      table_index_sgpr = 0xFFFFFFFFu;

    // Selector contract for this array. A user-SGPR selector names `table_index_sgpr`; a dynamic
    // scalar-buffer selector names the exact descriptor-load instruction in `table_load_pc` and
    // interprets its runtime scalar value as a byte offset through `table_entry_stride`.
    BufferTableSelectorMode table_selector_mode = BufferTableSelectorMode::None;
    uint32_t      table_load_pc = 0xFFFFFFFFu;

    // Concrete buffer descriptors available to this dispatch. `table_index_count` remains the
    // declared descriptor-set arity and is intentionally not derived from this payload; every
    // production boundary requires exact equality so a partially populated table fails visibly.
    std::vector<ShaderBufferTableEntry> table_entries;

    // Exact input-side origin for a DIRECT four-dword V# in the PM4 SH register file. This is
    // runtime realization provenance only; it does not affect descriptor binding or shader-cache
    // identity. The front half sets an absolute SPI_SHADER_USER_DATA_* register when it can prove
    // that all four descriptor words came from one entry user-data range. A shader-loaded/modified
    // descriptor can still be a valid runtime resource but has no single raw SH origin.
    static constexpr uint32_t kDirectVSharpOriginUnavailable = 0xFFFFFFFFu;
    static constexpr uint32_t kDirectVSharpOriginAmbiguous = 0xFFFFFFFEu;
    uint32_t direct_vsharp_sh_register_base = kDirectVSharpOriginUnavailable;

    //   * fetch_pc — PER-FETCH: the pc of the exact buffer/tbuffer format instruction this descriptor was
    //     resolved for. A single SRSRC SGPR is reloaded with a DIFFERENT V# per vertex attribute (position,
    //     uv, color, …), so keying only by sgpr_base collapses them to the first. MUBUF may fall back to
    //     metadata provenance; MTBUF requires this validated live-V# identity so FORMAT=INVALID cannot be
    //     resurrected through an older metadata resource. 0xFFFFFFFF = unset.
    uint32_t      fetch_pc      = 0xFFFFFFFFu;
    VertexFetchIndexMode fetch_index_mode = VertexFetchIndexMode::Automatic;

    // BVH descriptor BOX_GROW (bits 62:55). IMAGE_BVH_INTERSECT_RAY expands box intervals by
    // this many 2^-24 increments; zero is exact/no growth. Meaningful only for the per-fetch raw
    // ConstantBuffer view created from a validated BVH descriptor.
    uint32_t      bvh_box_grow  = 0;

    // BVH descriptor BOX_SORT_EN (bit 63). When set, IMAGE_BVH_INTERSECT_RAY returns box-child
    // pointers in increasing intersection-time order instead of physical node order. This changes
    // generated SPIR-V and therefore participates in compile-cache and capture identity.
    bool          bvh_sort_enabled = false;

    // FLAT-window (#1171) provenance: for a general flat_load whose 64-bit source pointer lives in the
    // consecutive user SGPRs s[flat_base_sgpr : +1], the executor binds the containing guest allocation
    // as this SSBO (keyed by the load's fetch_pc) and the emitter lowers the load to an indexed read at
    // (address - base). Kept SEPARATE from sgpr_base so the direct-descriptor SGPR routing does not
    // divert the base pointer into sreg_input; the emitter reads it straight from the push constants.
    // 0xFFFFFFFF = not a flat window.
    uint32_t      flat_base_sgpr = 0xFFFFFFFFu;

    // Texture-only (cls == Texture). img_dim mirrors the MIMG dim field (1D=0, 2D=1, 3D=2, ...).
    // width/height/depth are the complete base-level extent (depth is >1 for 3D only), used by image
    // queries, uploads, and image_load/texelFetch. sampler_sgpr_base = the paired sampler's S# base SGPR (SSAMP); with a Vulkan
    // COMBINED_IMAGE_SAMPLER the sampler is baked into the same `binding`, so this is provenance for a
    // future image/sampler split.
    uint32_t      img_dim           = 1;
    uint32_t      width             = 0;
    uint32_t      height            = 0;
    uint32_t      depth             = 1;
    // Guest sample count. 2D_MSAA image_load is represented on the host as a single-sample 2D-array
    // image with one exact sample plane per layer, while this field retains guest descriptor identity.
    uint32_t      sample_count      = 1;
    uint32_t      tile_mode         = 0;                  // T# GFX10 TileMode; drives auto-detile of a sampled surface
    // Byte distance between rows of a linear sampled image. Zero means derive it from the live
    // descriptor/backing: exact HLE-producer provenance wins, other guest images use the GFX10
    // sampled-image alignment, and ordinary host fixtures are tight. Captures persist the resolved
    // value so replay keeps every layout exact.
    uint32_t      linear_row_pitch_bytes = 0;
    // A packed mip-tail view shares the allocation's first 4/64 KiB block. gpu_addr remains the
    // shared block base; the backend applies mip_tail_offset and preserves sibling levels on writes.
    // T#-declared mip-chain length relative to the selected base level (last_level - base_level + 1,
    // WORD3 [19:16]/[15:12]). 1 = single level (the historical behavior). The backend uses this to
    // bound generated-mip uploads (#1272) — it never invents levels a T# does not declare.
    uint32_t      declared_mip_levels = 1;
    // Placement provenance for the WHOLE allocation this view selects from, not just the selected
    // level. `declared_mip_levels` says how many levels the T# declares; these four say where each
    // of them lives, because tiled_mip_level_layout needs the allocation's own level-zero element
    // extent and its final mip -- neither of which is recoverable from the fields above once
    // gpu_addr/width/height have been shifted to the selected base level. Zero element extent means
    // "not modelled"; every consumer must fail closed on it (#3048).
    uint32_t      mip_chain_element_width  = 0;
    uint32_t      mip_chain_element_height = 0;
    uint32_t      mip_chain_bytes_per_block = 0;
    uint32_t      mip_chain_max_level      = 0;   // effective MAX_MIP of the allocation
    uint32_t      mip_chain_base_level     = 0;   // the T#'s BASE_LEVEL for this view
    bool          in_mip_tail       = false;
    uint32_t      mip_tail_offset   = 0;
    uint32_t      mip_tail_bytes    = 0;
    uint32_t      mip_tail_x        = 0;
    uint32_t      mip_tail_y        = 0;
    // Selected view inside a multi-layer thin-2D allocation. Zero preserves the historical tightly
    // packed selected-level representation. When nonzero, gpu_addr is the first selected slice's
    // allocation base and each layer's level begins at layer*stride + layer_mip_offset; packed-tail
    // levels instead use mip_tail_x/y within each slice base.
    uint32_t      layer_stride_bytes = 0;
    uint32_t      layer_mip_offset_bytes = 0;
    // Instruction-scoped proof for IMAGE_LOAD_MIP / IMAGE_STORE_MIP. This is compile semantics,
    // not descriptor metadata: only the exact fetch_pc whose dimension-specific mip VGPR has a
    // same-block plain-v_mov zero proof may set it. Cache/capture identity must preserve the marker
    // so a zero-specialized module can never be reused for a dynamic or nonzero mip.
    bool          proven_zero_mip   = false;
    bool          srgb              = false;              // T# is a gamma-encoded (sRGB) surface — sample with sRGB->linear (#263)
    uint32_t      sampler_sgpr_base = 0xFFFFFFFFu;

    // Sampler state decoded from the PAIRED S# (Texture only). The backend previously hardcoded a
    // LINEAR + clamp-to-edge sampler for every texture, which blurs point-sampled content (pixel art
    // gets an outline halo on every texel) and ignores the game's real wrap modes. Honoring the S#
    // fixes that for pixel-art titles AND keeps linear-filtered art correct. Defaults preserve the old
    // LINEAR/clamp behavior for any texture whose S# we cannot resolve.
    //   mag/min/mip_filter: 0 = point/nearest, 1 = bilinear/linear (SQ_IMG_SAMP XY_*_FILTER / MIP_FILTER).
    //   addr_uvw:           Gen5 SQ_TEX CLAMP enum per axis (0=wrap,1=mirror,2=clamp-last-texel,6/7=border).
    uint32_t      mag_filter        = 1;
    uint32_t      min_filter        = 1;
    uint32_t      mip_filter        = 0;
    uint32_t      addr_uvw[3]       = {2, 2, 2};

    // Remaining SQ_IMG_SAMP fields (#262). Defaults reproduce the current Vulkan sampler exactly, so a
    // texture whose S# we cannot resolve — and every render test that fills a ShaderResource directly —
    // is byte-identical. Applied where valid on the current color combined-image-sampler path; the three
    // that need extra machinery are gated separately: depth compare is lowered manually,
    // unnormalized coordinates are scaled in the recompiler while retaining the ordinary sampler,
    // and anisotropy requires the device feature.
    //   border_color_type:  WORD3[31:30] SQ_TEX_BORDER_COLOR (0=transparent-black,1=opaque-black,
    //                       2=opaque-white,3=register/custom). Only affects CLAMP_TO_BORDER wrap.
    //   min_lod/max_lod:    WORD1 [11:0]/[23:12], unsigned u4.8 (raw/256.0). LOD clamp.
    //   lod_bias:           WORD2 [13:0], signed s5.8 (sign-extended raw/256.0). mip LOD bias.
    //   max_aniso_ratio:    WORD0 [11:9] enum (maxAnisotropy = 1<<ratio). NEEDS the samplerAnisotropy
    //                       device feature — decoded only.
    //   depth_compare_func: WORD0 [14:12] SQ compare enum for shadow/PCF samplers (VkCompareOp order).
    //                       APPLIED in-shader by the recompiler's manual-dref c_lz lowering (#1271);
    //                       the hardware compareEnable-sampler path still needs a depth-format view
    //                       (see the render-runner sampler note).
    //   unnormalized:       WORD0 [15] FORCE_UNNORMALIZED. The recompiler converts only spatial
    //                       texel coordinates/gradients to normalized coordinates so wrap, LOD and
    //                       array-layer semantics remain unchanged.
    uint32_t      border_color_type = 0;
    float         min_lod           = 0.0f;
    float         max_lod           = 0.0f;
    float         lod_bias          = 0.0f;
    uint32_t      max_aniso_ratio   = 0;
    uint32_t      depth_compare_func = 0;
    bool          depth_compare      = false;  // MIMG IMAGE_SAMPLE_C* use
    uint32_t      unnormalized      = 0;

    // T# DST_SEL channel swizzle (SQ_SEL enum per channel: 0=0,1=1,4=R,5=G,6=B,7=A). Applied as a Vulkan
    // component-mapping on the sampled view so a non-identity surface (e.g. BGRA order, or an alpha-only
    // mask) reads correctly. Default = identity (R,G,B,A). NOT applied on the narrow R->RGBA replication
    // path (that already broadcasts coverage to every channel).
    uint32_t      swizzle[4]        = {4, 5, 6, 7};

    // GFX10 image-compression state decoded from T# WORD6/7. A DCC-compressed base allocation cannot
    // be interpreted as ordinary tiled texels: metadata_addr names the separate compression-control
    // surface. These fields are preserved through captures even though software DCC decode is not yet
    // implemented, so live/replay diagnostics remain truthful instead of hiding corrupt source bytes.
    uint32_t      max_uncompressed_block_size = 0;
    uint32_t      max_compressed_block_size   = 0;
    bool          meta_pipe_aligned           = false;
    bool          write_compress_enabled      = false;
    bool          compression_enabled         = false;
    bool          alpha_is_on_msb             = false;
    bool          color_transform             = false;
    uint64_t      metadata_addr               = 0;

    // Replay-only DCC backing. Captures retain the exact control-surface span separately from the
    // compressed base allocation. Production tables leave these zero/null and read guest memory by
    // metadata_addr; a future software decoder can consume either source without changing identity.
    uint64_t      dcc_metadata_size           = 0;
    uint8_t*      dcc_metadata_host_data      = nullptr;
    uint64_t      dcc_metadata_host_data_size = 0;

    // Replay-only owned backing. `gpu_addr` remains the captured logical guest address so render-target
    // identity/alias checks stay faithful. Graphics reads from host_data; compute may update it before a
    // later operation consumes the same version. Production tables leave both fields zero and use guest memory.
    uint8_t*       host_data        = nullptr;
    uint64_t       host_data_size   = 0;
    // Bytes of the SAME allocation that lie immediately BELOW `host_data`: `host_data - k` is the
    // guest byte at `gpu_addr - k` for every k in [1, host_data_prefix_bytes]. Zero means the
    // backing starts exactly at `gpu_addr`, which is what every host-built fixture and every
    // diagnostic replacement produces.
    //
    // It exists because a tiled GFX10 mip chain stores level zero LAST -- the shared tail block
    // first, then the remaining levels smallest-to-largest -- so a descriptor selecting level zero
    // names the HIGHEST address in its own allocation and every other level lives below it. Replay
    // sets this from the resource's offset inside its capture blob, whose bytes are the guest bytes
    // at those addresses by construction; production tables leave it zero and read guest memory by
    // address instead (#3202). Anything that repoints `host_data` at a different buffer must reset
    // it: a stale prefix would authorize a read before the start of the new allocation.
    uint64_t       host_data_prefix_bytes = 0;

    // Instruction-scoped record-count proof for GTA V's exact linear stride-8 qword-atomic V#. Appended
    // after the historical aggregate-initialized fields so positional test fixtures remain stable.
    // OOB_SELECT is not otherwise retained by this normalized resource, so emission, cache identity,
    // capture, and per-dispatch revalidation carry the proof explicitly.
    uint32_t       atomic_x2_record_count = 0;

    // Exact upper pointer word for key-less scalar-SMEM raw-pointer resources. The normalized
    // Base48/stride view cannot retain bits 48..63 even though later shader code may consume them
    // while rebuilding a V#. UINT32_MAX means this resource did not come from that raw-pointer path.
    uint32_t       scalar_raw_pointer_word_hi = UINT32_MAX;

    // Dispatch-scoped proof for GTA V 0x413ce6000's pc153 scalar descriptor-table lookup. The
    // front half admits either the complete zero-descriptor chain, or only the one in-bounds SOFFSET
    // (record 4) plus wholly OOB offsets, then keeps the exact selected V# words here. UINT32_MAX
    // means this is an ordinary resource. The marker lives only on pc153; pc156/158 remain ordinary
    // target bindings (including ordinary zero-record bindings in the all-zero mode).
    uint32_t       selected_sbuffer_soffset = UINT32_MAX;
    std::array<uint32_t, 4> selected_sbuffer_words{};

    // Derived authority for a dispatch-owned indirect-buffer shadow. A validated program contract
    // may append bounded pointee slots to host_data and lower only its exact indirect access sites.
    // These fields never promise that arbitrary FLAT addresses may index the shadow.
    uint32_t       indirect_buffer_contract_tag = 0;
    uint32_t       indirect_buffer_binding_bytes = 0;
    uint32_t       indirect_buffer_slot_count = 0;
    uint32_t       indirect_buffer_header_bytes = 0;
    uint32_t       indirect_buffer_slot_bytes = 0;

    // Version-2 carrier metadata stays separate from the fixed-slot v1 marker above. Reusing those
    // fields would make a generic relocation enter the legacy title-specific validator before its
    // independent proof could be re-established.
    IndirectPointerRelocationBinding indirect_pointer_relocation{};

    // Exact RDNA2 S_BUFFER M_SIZE in dwords. Zero means this resource has no scalar-buffer bound
    // metadata. Kept separate from `size`: the ordinary V# footprint is NUM_RECORDS*STRIDE, but
    // scalar-buffer addresses advance four bytes and use NUM_RECORDS only as their dword bound.
    // Appended so historical positional aggregate initializers retain their field mapping.
    uint32_t scalar_buffer_dword_count = 0;
};

// Decode the exact SQ_IMG_SAMP state consumed by one MIMG instruction. Metadata describes a
// texture's usual paired sampler, but shaders may load or patch a different S# before the sample.
// In that case instruction-time descriptor folding is authoritative.
inline void apply_sampler_descriptor(ShaderResource& resource, const uint32_t sampler[4]) {
    resource.mag_filter  = ((sampler[2] >> 20) & 0x3u) ? 1u : 0u;
    resource.min_filter  = ((sampler[2] >> 22) & 0x3u) ? 1u : 0u;
    resource.mip_filter  = ((sampler[2] >> 26) & 0x3u) ? 1u : 0u;
    resource.addr_uvw[0] = (sampler[0] >> 0) & 0x7u;
    resource.addr_uvw[1] = (sampler[0] >> 3) & 0x7u;
    resource.addr_uvw[2] = (sampler[0] >> 6) & 0x7u;
    resource.max_aniso_ratio    = (sampler[0] >> 9)  & 0x7u;
    resource.depth_compare_func = (sampler[0] >> 12) & 0x7u;
    resource.unnormalized       = (sampler[0] >> 15) & 0x1u;
    resource.min_lod            = static_cast<float>(sampler[1] & 0xFFFu) / 256.0f;
    resource.max_lod            = static_cast<float>((sampler[1] >> 12) & 0xFFFu) / 256.0f;
    int32_t bias14              = static_cast<int32_t>(sampler[2] & 0x3FFFu);
    if (bias14 & 0x2000) bias14 -= 0x4000;
    resource.lod_bias           = static_cast<float>(bias14) / 256.0f;
    resource.border_color_type  = (sampler[3] >> 30) & 0x3u;
}

// Attach one exact MIMG use to an otherwise identical metadata resource. The use may carry a live
// S# that differs from the metadata slot's paired sampler; retaining the old normalized sampler
// while publishing the new fetch PC creates a self-contradictory resource record.
inline bool attach_image_use(ShaderResource& resource, uint32_t use_pc,
                             const uint32_t* sampler) {
    if (resource.fetch_pc != 0xFFFFFFFFu && resource.fetch_pc != use_pc) return false;
    resource.fetch_pc = use_pc;
    if (resource.cls == ResourceClass::Texture && sampler)
        apply_sampler_descriptor(resource, sampler);
    return true;
}

inline bool is_gta5_selected_sbuffer_marker_candidate(const ShaderResource& resource) {
    return resource.selected_sbuffer_soffset != UINT32_MAX;
}

inline constexpr uint32_t kGtaSelectedSbufferRecord4Soffset = 480u;
inline constexpr uint32_t kGtaSelectedSbufferZeroChainSoffset = UINT32_MAX - 1u;
inline constexpr uint32_t kGtaSelectedSbufferAllOobSoffset = UINT32_MAX - 2u;
inline constexpr uint32_t kGtaSelectedSbufferNullRecord4Soffset = UINT32_MAX - 3u;

inline bool is_gta5_selected_sbuffer_descriptor(const ShaderResource& resource) {
    if (resource.cls != ResourceClass::ConstantBuffer || resource.fetch_pc != 153u)
        return false;
    if (resource.selected_sbuffer_soffset == kGtaSelectedSbufferZeroChainSoffset)
        return resource.format == DataFormat::Unknown && resource.num_components == 0u &&
               resource.gpu_addr == 0u && resource.size == 0u && resource.stride == 0u &&
               resource.srt_offset == UINT32_MAX && resource.sgpr_base == UINT32_MAX &&
               resource.host_data == nullptr && resource.host_data_size == 0u &&
               std::all_of(resource.selected_sbuffer_words.begin(),
                           resource.selected_sbuffer_words.end(),
                           [](uint32_t word) { return word == 0u; });
    if (resource.selected_sbuffer_soffset == kGtaSelectedSbufferAllOobSoffset)
        return resource.gpu_addr > 0x10000u && resource.stride == 120u &&
               resource.size == 600u &&
               std::all_of(resource.selected_sbuffer_words.begin(),
                           resource.selected_sbuffer_words.end(),
                           [](uint32_t word) { return word == 0u; });
    if (resource.selected_sbuffer_soffset == kGtaSelectedSbufferNullRecord4Soffset)
        return resource.gpu_addr > 0x10000u && resource.stride == 120u &&
               resource.size == 600u &&
               std::all_of(resource.selected_sbuffer_words.begin(),
                           resource.selected_sbuffer_words.end(),
                           [](uint32_t word) { return word == 0u; });
    return resource.gpu_addr > 0x10000u && resource.stride == 120u &&
           resource.size == 600u &&
           resource.selected_sbuffer_soffset == kGtaSelectedSbufferRecord4Soffset;
}

// A one-layer guest 2D-array descriptor is byte-identical to an ordinary 2D image, and some guest
// shaders deliberately address it with a non-arrayed DIM=2D instruction. Keep that established
// base-slice contract explicit and shared between recompilation and the live backend. This predicate
// identifies an actually ordinary, non-MSAA shader view; the separate storage-only predicate below
// covers a real depth-one arrayed view. The resource's descriptor shape remains part of cache
// identity, so this does not alias unrelated 2D/array resources.
constexpr bool shader_resource_uses_ordinary_2d_image(
    const ShaderResource& resource,
    bool shader_2d,
    bool shader_arrayed,
    bool shader_multisampled) {
    return shader_2d && !shader_arrayed && !shader_multisampled &&
           resource.depth == 1 && !resource.depth_compare &&
           (resource.img_dim == 1 || resource.img_dim == 5);
}

// Vulkan permits a VK_IMAGE_VIEW_TYPE_2D_ARRAY view with one array layer. That view is not an
// ordinary 2D shader type, but its sole subresource has the same texel bytes and transfer extent as
// the non-arrayed base-slice view above. Admit it only for native typed STORAGE: the producer keeps
// its reflected layer coordinate and one-layer array view, while a later ordinary sampled image may
// receive the exact retained result through a distinct one-layer VkImage copy. Multi-layer, MSAA,
// depth, and descriptor/view mismatches remain unsupported and fail visibly.
constexpr bool shader_resource_uses_native_2d_storage_image(
    const ShaderResource& resource,
    bool shader_2d,
    bool shader_arrayed,
    bool shader_multisampled) {
    return shader_resource_uses_ordinary_2d_image(
               resource, shader_2d, shader_arrayed, shader_multisampled) ||
           (shader_2d && shader_arrayed && !shader_multisampled &&
            resource.img_dim == 5 && resource.depth == 1 &&
            !resource.depth_compare);
}

// Exact unsigned integer storage also admits a real multi-layer 2D-array view. The live backend
// already stages every physical slice independently and creates an array image with the reflected
// layer count; keeping this separate from the float predicate prevents an unrelated format from
// silently broadening its dimensional contract. A shader-declared non-arrayed view still uses the
// ordinary/base-slice rule above.
constexpr bool shader_resource_uses_native_uint_2d_storage_image(
    const ShaderResource& resource,
    bool shader_2d,
    bool shader_arrayed,
    bool shader_multisampled) {
    return shader_resource_uses_native_2d_storage_image(
               resource, shader_2d, shader_arrayed, shader_multisampled) ||
           (shader_2d && shader_arrayed && !shader_multisampled &&
            resource.img_dim == 5 && resource.depth >= 1 &&
            !resource.depth_compare);
}

// The RESOURCE half of "may this R32_UINT storage image be lowered to a detiled linear atomic SSBO"
// (the RADV image-atomic workaround: RADV hangs/reset-poisons the device on a compute R32_UINT image
// atomic, so compute reaches the image through a buffer view instead).
//
// This lives in one place because it has already drifted twice. #2293 was titled "the image-atomic
// opcode list existed in THREE places and all three had to agree"; #2272 then generalised the
// LOWERING gate over the array layer and left the descriptor validator and the backend
// materialization on the single-layer clause, so a shader compiled into a binding its own validator
// rejected -- Sonic Racing: CrossWorlds' full-screen dispatch, skipped every frame (#2265). Callers
// add their own INSTRUCTION-side conditions (dmask, unorm, length); this answers only the question
// that every site was answering separately and differently.
//
// `depth` is the layer COUNT for an arrayed view, so it is pinned to 1 only in the non-arrayed case;
// dim 5 is SQ 2D_ARRAY. A layered resource additionally requires the caller to stage every layer --
// see shader_resource_atomic_image_layers below, and note that the slice stride is NOT
// width*height*bpe for a tiled surface.
constexpr bool shader_resource_supports_atomic_image_buffer(const ShaderResource& resource) {
    return resource.cls == ResourceClass::StorageImage &&
           resource.format == DataFormat::Uint32 &&
           (resource.num_components ? resource.num_components : 1u) == 1u &&
           (resource.img_dim == 1u ? resource.depth == 1u
                                   : (resource.img_dim == 5u && resource.depth >= 1u)) &&
           !resource.depth_compare && !resource.in_mip_tail &&
           !resource.compression_enabled && resource.width && resource.height;
}

// Layer count to stage for such a resource: an arrayed view carries `depth` layers, a plain 2D one
// carries exactly one regardless of what `depth` happens to hold.
constexpr uint32_t shader_resource_atomic_image_layers(const ShaderResource& resource) {
    return resource.img_dim == 5u ? (resource.depth ? resource.depth : 1u) : 1u;
}

// A null BVH is represented by a bounded host-owned marker. The front-half creates this only after
// proving that a mapped zero pointer feeds the complete descriptor inside an EXEC-guarded region.
// The exact shape survives capture/replay without adding a capture-format field and cannot alias a
// title allocation because gpu_addr is zero.
inline bool is_proven_null_bvh(const ShaderResource& resource) {
    return resource.cls == ResourceClass::ConstantBuffer &&
           resource.format == DataFormat::Uint32 && resource.num_components == 1u &&
           resource.gpu_addr == 0 && resource.size == 256u && resource.stride == 0u &&
           resource.fetch_pc != 0xFFFFFFFFu && resource.host_data != nullptr &&
           resource.host_data_size >= resource.size;
}

// Exact-PC marker for a fully-known buffer descriptor with NUM_RECORDS=0. Such a descriptor has
// architectural zero-read/drop-write behavior for the admitted bounded scalar, raw, format-load, and
// atomic operations, regardless of its base, so it deliberately has no guest or host backing. The
// unusual Unknown/zero-component shape keeps it distinct from an ordinary explicit null buffer and
// survives capture/replay without adding a serialized descriptor field.
inline bool is_zero_record_raw_buffer(const ShaderResource& resource) {
    return resource.cls == ResourceClass::ConstantBuffer &&
           resource.format == DataFormat::Unknown && resource.num_components == 0u &&
           resource.gpu_addr == 0 && resource.size == 0u && resource.stride == 0u &&
           resource.srt_offset == 0xFFFFFFFFu && resource.sgpr_base == 0xFFFFFFFFu &&
           resource.fetch_pc != 0xFFFFFFFFu && resource.host_data == nullptr &&
           resource.host_data_size == 0u;
}

// Exact-PC marker for GTA V's provenance-backed optional-null RAW dword load. It deliberately uses
// a different structural shape from the NUM_RECORDS=0 marker: the latter may drop stores/atomics,
// while this convention is load-only. Existing serialized resource fields therefore preserve the
// semantic distinction across capture/replay without a capture-format extension. The marker uses
// sampler_sgpr_base because that field is serialized for every resource but is otherwise inert for
// a ConstantBuffer; 0xFFFFFFFE is not a valid scalar-register base.
inline constexpr uint32_t kGtaOptionalBufferTableBytes = 272u;
inline constexpr uint32_t kGtaOptionalBufferPointerOffset = 0x58u;
inline constexpr uint32_t kGtaOptionalBufferStride = 4u;
inline constexpr uint32_t kGtaOptionalBufferStrideWord = kGtaOptionalBufferStride << 16u;
inline constexpr uint32_t kGtaOptionalBufferConfigWord = 0x00016204u;
inline constexpr uint32_t kGtaOptionalBufferTgidSgpr = 15u;
inline constexpr uint32_t kGtaOptionalBufferLocalSize = 64u;
inline constexpr uint32_t kOptionalNullRawLoadMarkerSamplerBase = 0xFFFFFFFEu;

inline bool is_optional_null_raw_load_buffer(const ShaderResource& resource) {
    return resource.cls == ResourceClass::ConstantBuffer &&
           resource.format == DataFormat::Uint32 && resource.num_components == 1u &&
           resource.gpu_addr == 0 && resource.size == 0u && resource.stride == 4u &&
           resource.srt_offset == 0xFFFFFFFFu && resource.sgpr_base == 0xFFFFFFFFu &&
           resource.sampler_sgpr_base == kOptionalNullRawLoadMarkerSamplerBase &&
           resource.flat_base_sgpr == 0xFFFFFFFFu &&
           resource.fetch_pc != 0xFFFFFFFFu && resource.host_data == nullptr &&
           resource.host_data_size == 0u;
}

// Exact-PC no-backing marker for GTA V's proven-null output-store region. UINT32_MAX is outside the
// V#'s 14-bit STRIDE domain and is serialized by every capture version, keeping this semantic
// distinct from both an ordinary base-zero descriptor and the NUM_RECORDS=0 marker without a format
// bump. Only the front half may manufacture this shape after proving the complete EXEC guard.
inline constexpr uint32_t kProvenNullGuardedRawStoreStride = UINT32_MAX;
inline bool is_proven_null_guarded_raw_store(const ShaderResource& resource) {
    return resource.cls == ResourceClass::ConstantBuffer &&
           resource.format == DataFormat::Unknown && resource.num_components == 0u &&
           resource.gpu_addr == 0 && resource.size == 0u &&
           resource.stride == kProvenNullGuardedRawStoreStride &&
           resource.srt_offset == 0xFFFFFFFFu && resource.sgpr_base == 0xFFFFFFFFu &&
           resource.fetch_pc != 0xFFFFFFFFu && resource.host_data == nullptr &&
           resource.host_data_size == 0u;
}

// GTA V's workgroup-list kernels read an optional output/work pointer from the mapped dispatch table
// at s0:s1+0x20, then build a launch-sized V# with 1024-byte stride. A zero pointer is an application
// convention, not NUM_RECORDS=0, so preserve a distinct exact-PC marker and retain the complete
// 40-byte source-table witness for capture/replay revalidation. UINT32_MAX-1 lies outside the V#'s
// 14-bit stride domain and cannot alias the guarded-store marker above.
inline constexpr uint32_t kGtaNullableOutputPointerOffset = 0x20u;
inline constexpr uint32_t kGtaNullableOutputWitnessBytes = 0x28u;
inline constexpr uint32_t kGtaNullableOutputStride = 1024u;
inline constexpr uint32_t kGtaNullableOutputStrideWord =
    kGtaNullableOutputStride << 16u;
// The first retained route used 57 groups; later gameplay uses the same exact kernels with 63.
// Keep 57 as the fixture count, but production validation relates the live descriptor count to
// entry s7 and the dispatch grid instead of treating one observed launch as program identity.
inline constexpr uint32_t kGtaNullableOutputFixtureRecordCount = 57u;
inline constexpr uint32_t kGtaNullableOutputMaxRecordCount =
    0x10000000u / kGtaNullableOutputStride;
inline constexpr uint32_t kGtaNullableOutputConfigWord = 0x00016204u;
inline constexpr uint32_t kGtaNullableOutputLocalSize = 256u;
inline constexpr uint32_t kGtaNullableOutputFixtureThreads =
    kGtaNullableOutputFixtureRecordCount * kGtaNullableOutputLocalSize;
inline constexpr uint32_t kGtaNullableOutputUserSgpr8 = 0x08000200u;
// The process kernel extracts bits 30:28 as a three-bit work selector at pc2. Routed gameplay
// exercised all eight selector values with every other entry-s8 bit unchanged.
inline constexpr uint32_t kGtaNullableOutputProcessSelectorMask = 0x70000000u;
inline constexpr uint32_t kProvenNullNullableRawBufferStride = UINT32_MAX - 1u;

inline bool is_nullable_raw_buffer_marker_candidate(const ShaderResource& resource) {
    return resource.stride == kProvenNullNullableRawBufferStride;
}

inline bool is_proven_null_nullable_raw_buffer(const ShaderResource& resource) {
    const bool valid_host_witness = resource.host_data
        ? resource.host_data_size >= kGtaNullableOutputWitnessBytes
        : resource.host_data_size == 0u;
    return resource.cls == ResourceClass::ConstantBuffer &&
           resource.format == DataFormat::Unknown && resource.num_components == 0u &&
           resource.gpu_addr > 0x10000u &&
           resource.size == kGtaNullableOutputWitnessBytes &&
           is_nullable_raw_buffer_marker_candidate(resource) &&
           resource.srt_offset == 0xFFFFFFFFu && resource.sgpr_base == 0xFFFFFFFFu &&
           resource.fetch_pc != 0xFFFFFFFFu && valid_host_witness;
}

// Which resource class an image instruction can accept. RDNA2 splits MIMG by operation, not by
// descriptor: image_store and every integer image atomic are read-modify-write and therefore can
// only target a StorageImage, an ordinary sample can only use a Texture, and IMAGE_LOAD (0x00) /
// IMAGE_GET_RESINFO (0x0e) read a descriptor that may legitimately be either. Keeping the three
// cases in one enum means the lookup and the post-lookup validation cannot drift apart -- the
// project has already paid for that drift once, when a widened storage classifier met an
// un-widened acceptance test and made resolution reject HARDER than before the fix (#2275).
enum class ImageResourceRequirement : uint32_t {
    SampledOnly,    // ordinary image_sample/gather: Texture
    StorageOnly,    // image_store + integer image atomics: StorageImage
    Either,         // IMAGE_LOAD / IMAGE_GET_RESINFO: whichever the front half classified it as
};

// Single acceptance predicate shared by the class-filtered lookups and by the recompiler's
// post-resolution validation, so "which classes may this op use" is stated exactly once.
inline bool image_resource_class_satisfies(ResourceClass cls, ImageResourceRequirement requirement) {
    switch (requirement) {
        case ImageResourceRequirement::SampledOnly: return cls == ResourceClass::Texture;
        case ImageResourceRequirement::StorageOnly: return cls == ResourceClass::StorageImage;
        case ImageResourceRequirement::Either:
            return cls == ResourceClass::Texture || cls == ResourceClass::StorageImage;
    }
    return false;
}

// The set of resources a shader uses. The front-half builds it from the shader's user_data; the
// recompiler consults it while translating memory ops and the pipeline binds from it. Pure data.
struct ShaderResourceTable {
    std::vector<ShaderResource> resources;
    // Exact dispatch contracts may materialize immutable host-only resources. ShaderResource keeps
    // raw pointers for the renderer ABI, so the table owns those allocations separately; table
    // copies retain stable backing through shared ownership.
    std::vector<std::shared_ptr<std::vector<uint8_t>>> owned_host_data;
    // Diagnostic replacements (gpu_replay --override-resource) kept SEPARATE from owned_host_data,
    // which is not merely an ownership list: `table_owns()` uses membership as a discriminator
    // between "renderer-built shadow, re-validate against live guest memory" and "replay-owned,
    // self-contained snapshot" (rdna2_indirect_buffer_shadow.cpp). A diagnostic override IS a
    // self-contained snapshot, so putting it in owned_host_data marks it as the opposite and can
    // make a dispatch decline for a reason caused by the override MECHANISM rather than the
    // substituted bytes -- a void comparison generated by the tool that exists to avoid them.
    // Same copy semantics, deliberately no consumers.
    std::vector<std::shared_ptr<std::vector<uint8_t>>> owned_diagnostic_data;
    // Graphics-only draw ABI input used by the portable NGG shell. Hardware packs consecutive
    // vertex/instance invocations into guest waves; flattening InstanceIndex therefore needs the
    // submitted number of vertices per instance. Zero keeps standalone shader fixtures compatible.
    uint32_t vertices_per_instance = 0;

    // Resolve the resource whose descriptor originates at `srt_offset` (indirect/`s_load` provenance);
    // nullptr if none. Deterministic; first match wins.
    const ShaderResource* by_srt_offset(uint32_t srt_offset) const;
    // Resolve the resource whose descriptor lives at SGPR `sgpr` (direct/user-data provenance);
    // nullptr if none.
    const ShaderResource* by_sgpr_base(uint32_t sgpr) const;
    // Same, but restricted to a resource class. A single SGPR can hold different descriptors at different
    // points (e.g. s8 = a constant-buffer V# for an early s_buffer_load, then a vertex-buffer V# after a
    // dynamic reload for a later buffer_load_format). The instruction type implies the class, so filtering
    // by class disambiguates without tracking per-instruction reloads. nullptr if none.
    const ShaderResource* by_sgpr_base_cls(uint32_t sgpr, ResourceClass cls) const;
    // Resolve the vertex buffer for the fetch instruction at `pc` (per-fetch provenance — disambiguates an
    // SRSRC SGPR reloaded with a different V# per attribute). nullptr if none.
    const ShaderResource* by_fetch_pc(uint32_t pc) const;

    // CLASS-FILTERED lookups for an image op (MIMG). The three plain lookups above are first-match
    // wins AND class-blind, which is safe only for a caller that can use whatever class comes back.
    // An image op cannot: it needs a Texture or a StorageImage, and one key can legitimately carry
    // several resources of different classes -- a live vertex stage has been observed with a
    // ConstantBuffer and two VertexBuffers all at sgpr_base 8. Taking the first hit and discarding
    // it on class (which is what the MIMG resolver did) makes an image resource behind a
    // buffer-class one at the same key UNREACHABLE, and, worse, a wrong-class hit on the SRT route
    // also suppressed the SGPR fallback because `res` was non-null when that fallback was tested.
    // Filtering inside the lookup removes both failure modes: the search continues past a
    // wrong-class entry instead of stopping on it. See #3126 / #1634.
    const ShaderResource* image_by_srt_offset(uint32_t srt_offset,
                                              ImageResourceRequirement requirement) const;
    const ShaderResource* image_by_sgpr_base(uint32_t sgpr,
                                             ImageResourceRequirement requirement) const;
    const ShaderResource* image_by_fetch_pc(uint32_t pc,
                                            ImageResourceRequirement requirement) const;
    // Resolve by assigned Vulkan binding (the pipeline's lookup); nullptr if none.
    const ShaderResource* by_binding(uint32_t binding) const;
};

// Validate the generic runtime-selected buffer-array representation. Scalar resources are valid
// only with an inert table payload; array resources require a coherent selector and one raw,
// normalized entry per declared element. This does not require host backing -- live resources use
// guest memory -- but any provided host backing must cover the complete normalized entry.
bool valid_shader_buffer_table_contract(const ShaderResource& resource);

// Descriptor interface reflected from generated SPIR-V. This deliberately models only descriptor
// classes emitted by prosper's recompiler; I/O variables and inactive declarations are excluded.
enum class SpirvDescriptorKind : uint32_t {
    Unknown,
    StorageBuffer,
    CombinedImageSampler,
    StorageImage,
};

enum class SpirvShaderStage : uint32_t {
    Vertex = 0,
    Fragment = 4,
    Compute = 5,
    Unknown = 0xFFFFFFFFu,
};

// Scalar numeric class carried by an OpTypeImage's Sampled Type. Vulkan requires a storage-image
// view's numeric class to agree with this type even when the SPIR-V Image Format is Unknown. Keep
// Unknown explicit: guessing float from a normalized guest format is exactly the undefined binding
// that #1713 exposed.
enum class SpirvImageNumericClass : uint32_t {
    Unknown = 0,
    Float,
    Uint,
    Sint,
};

// Semantic carried by the strict two-byte -> one-dword storage-buffer materialization contract.
// It is explicit because Uint16 and Float16 share the same physical bytes but produce different
// guest VGPR values; a cached module or replay marker must never infer one from the other.
enum class StorageBufferTailSemantic : uint32_t {
    None = 0,
    Uint16,
    Float16,
};

// SPIR-V Image Format operand used by the exact one-word storage fallback. Keep this public so the
// recompiler, reflection, live backend, and their contract tests cannot drift through magic values.
constexpr uint32_t kSpirvImageFormatR32ui = 33;
constexpr uint32_t kSpirvImageFormatRgba8ui = 32;
constexpr uint32_t kSpirvImageFormatR16ui = 38;
constexpr uint32_t kSpirvImageFormatR8ui = 39;

// A declared descriptor-array length that reflection could not read (#2412). Distinct from 0, which
// means exactly `OpTypeRuntimeArray` -- a length deliberately supplied at bind time and therefore
// compatible with any table size.
//
// The distinction is load-bearing rather than tidy. Validation treats 0 as permissive, so folding
// "unresolved" into it made a DECODE FAILURE the most permissive value in the space, inside the check
// whose entire purpose is to stop silent acceptance: a shader declaring 16 entries whose length did not
// resolve was accepted against a table of 8, and then indexes past the set. An `OpSpecConstant` length
// reaches this -- valid SPIR-V, and `OpSpecConstant` is not among the opcodes this pass resolves.
inline constexpr uint32_t kDescriptorArityUnknown = 0xFFFFFFFFu;

struct SpirvDescriptorBinding {
    uint32_t variable_id = 0;
    uint32_t set = 0;
    uint32_t binding = 0;
    SpirvDescriptorKind kind = SpirvDescriptorKind::Unknown;
    SpirvShaderStage stage = SpirvShaderStage::Unknown;
    // Minimum byte range proven by constant access-chain indices. `dynamic_access` means a larger
    // runtime range may be addressed, so validation cannot derive an upper bound from SPIR-V alone.
    uint64_t required_bytes = 0;
    bool dynamic_access = false;
    // Data access, distinct from merely loading the descriptor object. For buffers these follow
    // pointer loads/stores; for images they follow OpImageRead/sample and OpImageWrite. Backends use
    // the per-binding result to avoid seeding write-only outputs and reading back read-only inputs.
    bool readable = false;
    bool writable = false;
    // Coordinate contract for sampled images. Normalized sampling (OpImageSample*/Gather) may bind a
    // uniformly render-scaled image directly; texel-space access (OpImageFetch/Read) and image-size
    // queries require the descriptor's exact declared extent. A binding can use both, in which case
    // exact extent wins.
    bool normalized_sampling = false;
    bool texel_access = false;
    // Sampled-image component type encoded by SPIR-V. UNORM formats return this float type for both
    // normalized sample/gather operations and integer-coordinate OpImageFetch.
    bool sampled_float = false;
    // Storage-image sampled type encoded by SPIR-V. This is the backend's authoritative choice
    // between a native float VkFormat and the portable raw-uvec4 conversion path, including replay.
    bool storage_float = false;
    // Raw SPIR-V Image Format operand. Format=R32ui distinguishes exact single-word storage
    // fallbacks from the wider formatless uvec4 contract during offline replay.
    uint32_t storage_image_format = 0;
    // Exact OpTypeImage shape. Backends use this instead of guessing from the guest T#: a DIM=5
    // packet may deliberately compile either as the historical base-slice 2D fallback or as a real
    // 2D-array image whose layer coordinate must match a VK_IMAGE_VIEW_TYPE_2D_ARRAY view. It also
    // distinguishes a guest cube descriptor compiled as an ordinary 2D face view.
    uint32_t image_dim = UINT32_MAX;
    bool image_arrayed = false;
    bool image_multisampled = false;
    // OpTypeImage Depth. IMAGE_SAMPLE_C* currently lowers to an in-shader comparison over an
    // ordinary color sampler, so this remains false even when ShaderResource::depth_compare says
    // that the guest instruction performs a comparison. Backends must follow the SPIR-V type here.
    bool image_depth = false;
    // The descriptor is reached by an OpAtomic*. Compute uses this to recognize the deliberately
    // buffer-backed view of an exact R32_UINT StorageImage (the RADV image-atomic workaround).
    bool atomic_access = false;
    // Exact Sampled Type numeric class. `sampled_float` / `storage_float` remain as convenient,
    // backwards-compatible predicates; this field prevents a signed/unsigned integer image from
    // collapsing into the same false boolean at the Vulkan binding boundary.
    SpirvImageNumericClass image_numeric_class = SpirvImageNumericClass::Unknown;
    // A generated module may prove that every access to this storage-buffer binding is an exact
    // one-record scalar Uint16/Float16 format-load contract. Vulkan storage buffers are arrays of u32 in our
    // portable ABI, so that two-byte guest record is materialized as one zero-padded dword. These
    // values are non-zero only when a strict Prosper.OpModuleProcessed marker survives reflection;
    // ordinary/raw accesses never receive the exception.
    uint64_t zero_pad_logical_bytes = 0;
    uint64_t zero_pad_binding_bytes = 0;
    StorageBufferTailSemantic zero_pad_semantic = StorageBufferTailSemantic::None;

    // Number of descriptors the binding declares: 1 for an ordinary binding, N for a fixed-size array,
    // and 0 for an `OpTypeRuntimeArray` of descriptors whose length is supplied at bind time (#2412).
    //
    // Distinct from `required_bytes` on purpose. Reflection folds an array index into the same offset
    // arithmetic it uses for byte offsets inside a buffer, because it treats `TypeKind::Array`
    // uniformly -- so an array of descriptors reports one binding with a nonsense byte requirement.
    // This field is what lets validation tell "eight descriptors" from "one descriptor, 128 bytes in".
    //
    // DELIBERATELY LAST, and anything added later must also go last. Reflection builds this struct with
    // a POSITIONAL aggregate initializer (`shader_resources.cpp`, `SpirvDescriptorBinding descriptor{
    // var, si->second, ... }`) that names no field, so a field inserted anywhere above silently shifts
    // every initializer after it by one -- `readable` receives `descriptor_count`'s slot and so on down.
    // It compiles without a warning. Measured: inserting this field after `dynamic_access` turned 0 test
    // failures into 15. Appending leaves the initializer's positions untouched and this member falls to
    // its default initializer, which is what an ordinary single-descriptor binding must report anyway.
    uint32_t descriptor_count = 1;
};

struct StorageBufferMaterializationPlan {
    uint64_t logical_bytes = 0;
    uint64_t binding_bytes = 0;
    StorageBufferTailSemantic semantic = StorageBufferTailSemantic::None;
    bool zero_padded_tail = false;
    bool valid = false;
};

// Validate an explicitly-carried scalar S_BUFFER bound and return the complete byte span its dword
// addresses may read. Ordinary resources return `size`; malformed scalar metadata returns zero.
uint64_t shader_resource_buffer_binding_bytes(const ShaderResource& resource);

// Derive the exact host binding range from reflected shader semantics plus the runtime V#. Admitted
// expansions are the read-only one-record Uint16/Float16 FORMAT tail (zero-padded 2 -> 4 bytes), and
// a read-only scalar S_BUFFER whose dword-addressable span exceeds its ordinary V# byte footprint.
// Any marker/runtime mismatch fails closed. Ordinary buffers keep logical==binding==resource.size.
StorageBufferMaterializationPlan plan_storage_buffer_materialization(
    const SpirvDescriptorBinding& descriptor,
    const ShaderResource& resource);

// Deterministically seed a planned binding. The destination is cleared before the exact logical
// source bytes are copied, so a padded tail can never borrow bytes from the following guest object.
bool materialize_storage_buffer_bytes(
    const StorageBufferMaterializationPlan& plan,
    const uint8_t* source,
    uint64_t source_bytes,
    uint8_t* destination,
    uint64_t destination_bytes);

enum class DescriptorIssueCode : uint32_t {
    MalformedSpirv,
    StageMismatch,
    SetMismatch,
    MissingBinding,
    DuplicateBinding,
    WrongType,
    InvalidAddress,
    InvalidBufferMetadata,
    InvalidImageMetadata,
    UndersizedBuffer,
    UnusedRuntimeBinding,
    // The SPIR-V declares an ARRAY of descriptors at this binding while the resource table supplies a
    // single one, or the reverse (#2412).
    //
    // Added BEFORE arrays are made to work, deliberately. Today an array binding reflects as one binding
    // with a `required_bytes` computed as though the array index were a byte offset, and is then compared
    // against a scalar `ShaderResource` -- and no existing code covers that, so the mismatch validates
    // SILENTLY. Introducing the code first turns the rest of this work into "make this stop firing",
    // which is a lever that can be watched, instead of "change reflection and hope validation still means
    // something". A layer whose job is catching mis-binding must not be the layer that fails quietly.
    ArrayBindingArityMismatch,
};

struct DescriptorValidationIssue {
    DescriptorIssueCode code = DescriptorIssueCode::MalformedSpirv;
    bool error = true;
    uint32_t set = 0;
    uint32_t binding = 0;
    SpirvDescriptorKind expected = SpirvDescriptorKind::Unknown;
    SpirvDescriptorKind actual = SpirvDescriptorKind::Unknown;
    uint64_t required_bytes = 0;
    uint64_t available_bytes = 0;

    // Descriptor counts for `ArrayBindingArityMismatch`, and zero for every other code (#2412).
    //
    // These exist rather than reusing the byte slots above, which was the first attempt: those two are
    // rendered for EVERY issue by `gpu_executor.cpp` (the `[descriptor]` and `[compute-descriptor]`
    // dumps) and are mixed into the diagnostic dedupe hash for error issues, so an eight-element array
    // against a scalar printed `required=8 available=0` -- a byte count that was really a descriptor
    // count, in a line a human reads while debugging the very stages this check exists to guard.
    //
    // APPEND-ONLY, like `SpirvDescriptorBinding`: this struct is brace-initialised positionally at every
    // `report.issues.push_back({...})` site, so a field inserted above silently shifts them all. #2462.
    uint32_t shader_count = 0;
    uint32_t runtime_count = 0;
};

struct DescriptorValidationReport {
    std::vector<SpirvDescriptorBinding> descriptors;
    std::vector<DescriptorValidationIssue> issues;
    bool ok() const;
};

// Reflection remains complete when a valid module disagrees with its runtime table (for example an
// undersized binding). Consumers that only need the module's statically-used binding set may still
// use descriptors in that case; malformed SPIR-V is the one fail-closed condition where the list can
// be partial.
bool spirv_descriptor_reflection_complete(const DescriptorValidationReport& report);

// Reflect the statically-used descriptor interface and validate it against one stage's runtime
// table. `expected_set`/`expected_stage` catch stage visibility mistakes (VS=set 0, PS=set 1).
// Unused runtime resources are warnings; every other issue rejects strict mode.
DescriptorValidationReport validate_spirv_descriptor_interface(
    const std::vector<uint32_t>& spirv,
    const ShaderResourceTable* runtime,
    uint32_t expected_set,
    SpirvShaderStage expected_stage,
    bool report_unused = true);

// Locate one binding in the reflected, statically-used descriptor interface. Runtime tables can
// retain descriptor candidates recovered while folding the guest shader even when the final SPIR-V
// does not reference them; consumers should not materialize those unused candidates.
const SpirvDescriptorBinding* find_spirv_descriptor_binding(
    const DescriptorValidationReport& report, uint32_t set, uint32_t binding);

const char* spirv_descriptor_kind_name(SpirvDescriptorKind kind);
const char* descriptor_issue_name(DescriptorIssueCode code);

} // namespace prosper::gpu
