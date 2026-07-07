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
    // 10/11-bit packed and block-compressed formats are added as the target needs them.
};

// How many bytes one component of `format` occupies (0 for Unknown).
uint32_t data_format_bytes(DataFormat f);

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
    uint32_t      sampler_sgpr_base = 0xFFFFFFFFu;
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

} // namespace prosper::gpu
