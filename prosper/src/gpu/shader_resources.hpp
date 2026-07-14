// shader_resources.hpp — the resource-binding CONTRACT between the front-half (which knows the game's
// real GPU resources) and the recompiler back-half (which must translate the shader's memory ops
// correctly). This is the seam that unblocks the format-dependent memory instructions:
//   * s_buffer_load_*   — uniforms / constant buffers (multi-buffer, beyond the single-cbuf model)
//   * buffer_load_format_* (MUBUF) — vertex attribute fetch (needs the attribute's data format)
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

// Element data format, decoded from a V#/T# descriptor's DFMT (data format) + NFMT (number format).
// Determines the conversion a format load / texture sample must emit. `Float32` is the common case
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
};

// How many bytes one component of `format` occupies (0 for Unknown and block-compressed formats).
uint32_t data_format_bytes(DataFormat f);

// IEEE-754 binary16 -> binary32 (handles subnormals, +/-inf, NaN). Used by the texture upload path to
// convert a sampled Float16 surface to the RGBA8 the backend uploads (#290). Pure + testable.
float half_to_float(uint16_t h);

// Unsigned small-float components of a packed Float10_11_11 texel (#294). Both share binary16's
// 5-bit exponent (bias 15) with a shortened mantissa (6 bits for the 11-bit R/G, 5 for the 10-bit B)
// and NO sign bit — so a left-shift of the mantissa into a half's 10-bit field is an exact
// widening (subnormals scale identically, inf/NaN preserved). Pure + testable.
float f11_to_float(uint16_t v);   // low 11 bits used
float f10_to_float(uint16_t v);   // low 10 bits used

// GFX10_FORMAT_2_10_10_10_UNORM packed texel -> RGBA8, with nearest UNORM scaling. Pure + testable.
void unorm2_10_10_10_to_rgba8(uint32_t packed, uint8_t rgba[4]);

enum class ResourceClass : uint32_t {
    ConstantBuffer,  // read by s_buffer_load_* (scalar, uniform across the wave)
    VertexBuffer,    // read by buffer_load_format_* (per-lane attribute fetch)
    Texture,         // read by image_sample / image_load (sampled image + sampler)
    Sampler,         // paired with a Texture for image_sample
    StorageImage,    // read/written by image_load / image_store WITHOUT a sampler (compute copy/blit).
                     // Bound as a Vulkan STORAGE_IMAGE; img_dim gives 1D/2D/3D. Our raw-32-bit-VGPR
                     // model uses a uint-sampled storage image (Format=Unknown + read/write-without-
                     // format), so texels move bit-exact (format reinterpretation lives in the descriptor).
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

    //   * fetch_pc — PER-FETCH: the pc of the exact buffer_load_format_* instruction this descriptor was
    //     resolved for. A single SRSRC SGPR is reloaded with a DIFFERENT V# per vertex attribute (position,
    //     uv, color, …), so keying only by sgpr_base collapses them to the first. The recompiler resolves a
    //     vertex fetch by its instruction pc first (exact), falling back to sgpr_base. 0xFFFFFFFF = unset.
    uint32_t      fetch_pc      = 0xFFFFFFFFu;

    // Texture-only (cls == Texture). img_dim mirrors the MIMG dim field (1D=0, 2D=1, 3D=2, ...).
    // width/height are for image_load/texelFetch + unnormalized addressing (unused by normalized
    // image_sample). sampler_sgpr_base = the paired sampler's S# base SGPR (SSAMP); with a Vulkan
    // COMBINED_IMAGE_SAMPLER the sampler is baked into the same `binding`, so this is provenance for a
    // future image/sampler split.
    uint32_t      img_dim           = 1;
    uint32_t      width             = 0;
    uint32_t      height            = 0;
    uint32_t      tile_mode         = 0;                  // T# GFX10 TileMode; drives auto-detile of a sampled surface
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
    //   depth_compare_func: WORD0 [14:12] SQ compare enum for shadow/PCF samplers. NEEDS a depth/shadow
    //                       sampling path (our images are color UNORM) — decoded only.
    //   unnormalized:       WORD0 [15] FORCE_UNNORMALIZED. NEEDS strict validity (no mips, equal filters,
    //                       clamp addressing, minLod=maxLod=0) — decoded only.
    uint32_t      border_color_type = 0;
    float         min_lod           = 0.0f;
    float         max_lod           = 0.0f;
    float         lod_bias          = 0.0f;
    uint32_t      max_aniso_ratio   = 0;
    uint32_t      depth_compare_func = 0;
    uint32_t      unnormalized      = 0;

    // T# DST_SEL channel swizzle (SQ_SEL enum per channel: 0=0,1=1,4=R,5=G,6=B,7=A). Applied as a Vulkan
    // component-mapping on the sampled view so a non-identity surface (e.g. BGRA order, or an alpha-only
    // mask) reads correctly. Default = identity (R,G,B,A). NOT applied on the narrow R->RGBA replication
    // path (that already broadcasts coverage to every channel).
    uint32_t      swizzle[4]        = {4, 5, 6, 7};

    // Replay-only owned backing. `gpu_addr` remains the captured logical guest address so render-target
    // identity/alias checks stay faithful. Graphics reads from host_data; compute may update it before a
    // later operation consumes the same version. Production tables leave both fields zero and use guest memory.
    uint8_t*       host_data        = nullptr;
    uint64_t       host_data_size   = 0;
};

// The set of resources a shader uses. The front-half builds it from the shader's user_data; the
// recompiler consults it while translating memory ops and the pipeline binds from it. Pure data.
struct ShaderResourceTable {
    std::vector<ShaderResource> resources;

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

const char* spirv_descriptor_kind_name(SpirvDescriptorKind kind);
const char* descriptor_issue_name(DescriptorIssueCode code);

} // namespace prosper::gpu
