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
#include <cstdint>
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
            (components == 3 && format == DataFormat::Float10_11_11));
}

// Native typed storage is only valid when the physical device advertises storage-image support
// for the selected VkFormat. Keep the feature input abstract so the fallback policy can be tested
// without requiring a particular Vulkan device or exposing Vulkan types in this shared header.
constexpr bool native_float_storage_image_supported(DataFormat format, uint32_t components,
                                                    bool srgb, bool storage_image_feature) {
    return storage_image_feature && native_float_storage_image(format, components, srgb);
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
    return 0;
}

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

    //   * fetch_pc — PER-FETCH: the pc of the exact buffer/tbuffer format instruction this descriptor was
    //     resolved for. A single SRSRC SGPR is reloaded with a DIFFERENT V# per vertex attribute (position,
    //     uv, color, …), so keying only by sgpr_base collapses them to the first. MUBUF may fall back to
    //     metadata provenance; MTBUF requires this validated live-V# identity so FORMAT=INVALID cannot be
    //     resurrected through an older metadata resource. 0xFFFFFFFF = unset.
    uint32_t      fetch_pc      = 0xFFFFFFFFu;
    VertexFetchIndexMode fetch_index_mode = VertexFetchIndexMode::Automatic;

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
    // that need extra machinery (depth-compare, unnormalized, anisotropy) are decoded but NOT applied yet
    // (see the render-runner sampler site for why each is gated).
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
    //   unnormalized:       WORD0 [15] FORCE_UNNORMALIZED. NEEDS strict validity (no mips, equal filters,
    //                       clamp addressing, minLod=maxLod=0) — decoded only.
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
};

// A statically proven missing BVH descriptor is materialized as this exact host-owned resource.
// Guest address zero cannot name a real BVH allocation, so the recompiler may replace the guarded
// IMAGE_BVH_INTERSECT_RAY use with a compact deterministic no-hit result. Keeping the marker derived
// from serialized resource fields also preserves capture/replay without changing the capture ABI.
inline bool is_bvh_no_hit_fallback(const ShaderResource& resource) {
    return resource.cls == ResourceClass::ConstantBuffer &&
           resource.format == DataFormat::Uint32 && resource.num_components == 1u &&
           resource.gpu_addr == 0 && resource.size == 256u && resource.stride == 0u &&
           resource.fetch_pc != 0xFFFFFFFFu && resource.host_data != nullptr &&
           resource.host_data_size >= resource.size;
}

// The set of resources a shader uses. The front-half builds it from the shader's user_data; the
// recompiler consults it while translating memory ops and the pipeline binds from it. Pure data.
struct ShaderResourceTable {
    std::vector<ShaderResource> resources;
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
    // Resolve by assigned Vulkan binding (the pipeline's lookup); nullptr if none.
    const ShaderResource* by_binding(uint32_t binding) const;
};

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

// SPIR-V Image Format operand used by the exact one-word storage fallback. Keep this public so the
// recompiler, reflection, live backend, and their contract tests cannot drift through magic values.
constexpr uint32_t kSpirvImageFormatR32ui = 33;

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
};

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
};

struct DescriptorValidationReport {
    std::vector<SpirvDescriptorBinding> descriptors;
    std::vector<DescriptorValidationIssue> issues;
    bool ok() const;
};

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
