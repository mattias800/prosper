// gpu_execute.hpp — the stage-independent core of the GPU executor (Stage A of docs/GPU_EXECUTOR_DESIGN.md).
//
// Turns a folded GpuState (exactly what agc_driver_submit_dcb produces via run_command_buffer) into a
// rendered frame: extract the RDNA2 render-state, recompile the vertex+pixel shaders straight from their
// SHADER_PGM addresses, resolve fixed-function state to Vulkan-ready values, and invoke a caller-supplied
// render backend. It is deliberately **Vulkan-agnostic** — the backend is a std::function — so this lives
// in prosper_core (which does not link Vulkan) while the live-device renderer is supplied by whoever has a
// device (the app/HLE, or tests via render_runner.h). agc_driver_submit_dcb calls this with the live
// renderer once the device is wired; tests call it with the offscreen renderer to verify the spine.
#pragma once
#include "command_processor.hpp"   // GpuState
#include "render_state.hpp"        // extract_render_state / resolve_pipeline_state / ResolvedPipelineState
#include "pm4_registers.hpp"        // CB_COLOR_CONTROL operation decode
#include <cstring>                 // memcpy: aliasing-safe index-buffer fingerprint loads
#include "rdna2_to_spirv.hpp"      // recompile_vertex / recompile_fragment
#include "shader_resources.hpp"    // ShaderResourceTable
#include "agc_shader_layout.hpp"   // DecodedBufferDescriptor (DynFetch)
#include "videoout_present.hpp"    // PresentFrameOrigin: a rendered frame carries its provenance
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <memory>
#include <vector>
#include <set>

extern "C" const void* prosper_agc_shader_header_for_code(uint64_t code_addr);
extern "C" const void* prosper_agc_fused_back_header_for_front(uint64_t front_code_addr);
extern "C" uint64_t prosper_agc_shader_continuation_for_code(uint64_t code_addr);

namespace prosper::gpu {

struct ShaderResourceTable;   // fwd (shader_resources.hpp); passed to the backend so it can bind resources

// Generation-scoped host-readability guard used while a synchronous GPU submit is active. Positive
// mapping ranges are cached for that submit only and discarded before guest code can reuse an address.
bool guest_readable(uint64_t address, uint32_t bytes);

using SharedShaderWords = std::shared_ptr<const std::vector<uint32_t>>;

// One realized draw of a submit: recompiled VS+PS SPIR-V, the draw's OWN resolved fixed-function
// state, the two stages' resource tables (so the backend can bind the constant/vertex buffers +
// textures the shaders declare, reading their bytes from 1:1-mapped guest memory), and its vertex
// count. execute_gpustate() emits one per realized draw; the backend records ALL of them into ONE
// render pass (clear once, then draw each with its own pipeline + descriptors) so a multi-draw submit
// — e.g. Unity's background + composite, whose per-draw masks/blends/shaders differ — composites
// correctly instead of collapsing onto a single draw. The tables may be null (color-only shaders).
struct DrawItem {
    std::vector<uint32_t> vs, gs, fs;                 // recompiled/generated SPIR-V
    // The live path can retain warm-cache shader modules by shared ownership instead of copying the
    // same SPIR-V words twice per draw (cache -> DrawItem -> BackendDraw). Capture/replay and direct
    // tests keep using the owned vectors above. Consumers must use the accessors so both forms are
    // equivalent; a shared value is immutable and remains valid if its cache entry is evicted.
    SharedShaderWords vs_shared, fs_shared;
    uint64_t vs_guest_addr = 0, fs_guest_addr = 0;    // diagnostic/source identity in guest VA space
    // Some GFX10 vertex programs install a fetch prolog at vs_guest_addr and transfer to a
    // separately allocated NGG main program. Capture retains that raw continuation as well so
    // diagnostic replay can recompile the same complete architectural program.
    uint64_t vs_chain_guest_addr = 0;
    // Content-addressed raw RDNA2 versions owned by a materialized capture. Live draw items leave
    // these unset; capture assigns them from the guest addresses above and replay restores them.
    uint32_t vs_raw_shader_index = 0xFFFFFFFFu;
    uint32_t fs_raw_shader_index = 0xFFFFFFFFu;
    uint32_t vs_chain_raw_shader_index = 0xFFFFFFFFu;
    uint32_t vertex_lds_dwords = 0;
    // Exact fixed-function graphics ABI used to produce the stored modules. Raw-shader replay must
    // use these values as a pair: shader bytes alone cannot reconstruct fragment system VGPRs or
    // PARAM linkage after the guest register state has been folded away.
    PixelInputMapping pixel_inputs{};
    PixelSystemInputMapping system_inputs{};
    bool has_pixel_inputs = false;
    bool has_system_inputs = false;
    // Process-unique identities supplied by the exact shader-recompile cache. Zero means the
    // shader came from an external/replay path, so persistent backend caches must compare words.
    uint64_t vs_identity = 0, fs_identity = 0;
    ResolvedPipelineState ps;                         // THIS draw's fixed-function state
    std::shared_ptr<ShaderResourceTable> vrt, prt;    // may be null
    uint32_t vertex_count = 3;
    uint32_t instance_count = 1;
    // #1256: the RAW draw-packet state recorded BEFORE realization — the DrawIndexAuto/DrawIndex
    // index_count (vcount_hint) and whether the draw was indexed. For a non-indexed draw the realized
    // vertex_count must equal raw_draw_count (see resolve_nonindexed_vertex_count / #1163); the capture
    // persists these so gpu_replay --inspect-only can flag a decode/realization divergence offline.
    uint32_t raw_draw_count = 0;
    bool raw_indexed = false;
    uint64_t raw_draw_modifier = 0;
    // GE_INDX_OFFSET at this draw. Vulkan consumes it as firstVertex for non-indexed draws and as
    // vertexOffset for indexed draws, preserving the hardware gl_VertexIndex contract.
    int32_t vertex_offset = 0;
    // Indexed draw (sceAgcDcbDrawIndex): the guest index buffer, fetched from 1:1-mapped memory and
    // widened to 32-bit. Non-empty -> the backend must render with vkCmdDrawIndexed (gl_VertexIndex
    // then IS the fetched index, which the recompiled VS uses for its storage-buffer vertex fetch);
    // empty -> plain vkCmdDraw(vertex_count).
    std::vector<uint32_t> indices;
    // Render-to-texture (#167): the CB_COLOR0_BASE this draw renders INTO. The game renders its scene
    // into a color target then samples that same address as a texture in a later composite pass; the
    // live renderer caches each submit's rendered pixels under this address and injects them when a
    // subsequent draw samples a texture at a matching base (otherwise the sample reads empty guest
    // memory — the scene RT is never populated on the CPU side — and the frame is a black composite).
    uint64_t color0_base = 0;
    uint32_t color0_width = 0, color0_height = 0;
    uint64_t color1_base = 0;
    uint32_t color1_width = 0, color1_height = 0;
    struct ColorTargetBinding {
        uint64_t base = 0;
        uint32_t width = 0, height = 0;
    };
    std::array<ColorTargetBinding, kColorTargetCount> color_targets{};
    uint64_t draw_index = 0;
    uint64_t command_order = 0;

    // A shared value WINS over the owned vector. Assigning `vs`/`fs` on an item that already carries
    // a shared value therefore has NO effect — the original shader keeps rendering, and the mistake
    // reads as "my substitution changed nothing" rather than as an error (#1434; it cost real time in
    // the #1427 investigation, and #1396 fixed the same class of confusion for the replay probes).
    // Substitute through set_vs()/set_fs(), which drop the shared value and the cache identity so
    // persistent backend caches compare the new words instead of hitting the old entry. `gs` has no
    // shared form today; if one is ever added it needs the same accessor/setter pair, or this bug
    // returns through the geometry stage.
    const std::vector<uint32_t>& vs_words() const { return vs_shared ? *vs_shared : vs; }
    const std::vector<uint32_t>& gs_words() const { return gs; }
    const std::vector<uint32_t>& fs_words() const { return fs_shared ? *fs_shared : fs; }

    void set_vs(std::vector<uint32_t> words) {
        vs = std::move(words); vs_shared.reset(); vs_identity = 0;
    }
    void set_fs(std::vector<uint32_t> words) {
        fs = std::move(words); fs_shared.reset(); fs_identity = 0;
    }
    // True when an owned vector was assigned while a shared value still shadows it — the silent
    // failure above. Diagnostics assert on this rather than rendering the wrong shader quietly.
    bool has_shadowed_shader() const {
        return (vs_shared && !vs.empty()) || (fs_shared && !fs.empty());
    }
};

// The pluggable Vulkan backend: render the submit's draw items into one image and return W*H*4 RGBA8
// pixels (or {} on failure). Empty list -> {} (nothing to draw).
using RenderFn = std::function<std::vector<uint8_t>(const std::vector<DrawItem>& items)>;

// Safe guest-address readability probe: a page-touching pipe probe on Linux/macOS and VirtualQuery
// on Windows let callers test a guest pointer without risking a host fault. Implemented in
// gpu_executor.cpp; shared by the executor's const-eval and HLE diagnostic probes.
bool guest_readable(uint64_t addr, uint32_t bytes);
// True only when the complete guest range is currently mapped writable. DMA_DATA uses this before
// copying into a guest-provided destination so a read-only mapping cannot turn a malformed packet
// into a host fault.
bool guest_writable(uint64_t addr, uint32_t bytes);

// PROSPER_DBG can encounter the same failed shader pair millions of times in a draw loop. Return
// true for the first and power-of-two occurrences so diagnostics retain recurrence/count evidence
// without letting repeated failures bury the unique recompiler rejection that caused them.
bool should_log_recompile_reject(uint64_t es_addr, uint64_t ps_addr,
                                 size_t vs_words, size_t gs_words, size_t fs_words,
                                 uint64_t* occurrence);

// Convert the untrusted byte size published by an AGC shader header into the exact instruction
// span that dynamic descriptor folding may probe and decode. Exposed so the safety/work bound has
// direct regression coverage instead of being inferred from whether a large guest range is mapped.
size_t dynamic_fold_shader_dwords(uint32_t shader_size_bytes);

// One resolved bindless-dynamic vertex fetch from the wave-uniform scalar const-fold in
// gpu_executor.cpp: the exact fetch instruction (pc), its SRSRC SGPR, and the V# live in that SGPR
// at that instruction. Exposed (with resolve_dynamic_fetch) so the fold's scalar-ALU semantics are
// unit-testable; production callers stay inside gpu_executor.cpp.
struct DynFetch {
    uint32_t fetch_pc; int srsrc; DecodedBufferDescriptor desc; uint32_t desc_v3;
    // Entry user-data dword that supplied the first raw V# word, proven only when all four live
    // descriptor words remain an unchanged consecutive seed (possibly through scalar moves) and
    // the fetch adds no instruction/SOFFSET byte transform to the normalized descriptor base.
    // UINT32_MAX means the descriptor was loaded, patched, or otherwise has no single SH origin.
    uint32_t direct_user_data_index = UINT32_MAX;
    // True when the V# came from the user-data SEED fallback (never s_loaded/patched-tracked). A
    // seed entry must NOT shadow a metadata-described direct vertex buffer at the same SGPRs: the
    // by_fetch_pc dyn path models the element address as gl_VertexIndex*stride (per-attribute
    // patched V#s fold their in-record offset into the base), while a single direct V# needs the
    // faithful VADDR/inst-offset address — shadowing it collapses every attribute onto offset 0.
    bool from_seed = false;
    // Descriptor before the instruction's constant OFFSET/SOFFSET is folded into `desc.base`.
    // Graphics VertexBuffers need the shifted descriptor for their special vertex-index path;
    // compute ConstantBuffers keep this original descriptor and apply those terms exactly once.
    DecodedBufferDescriptor unshifted_desc;
    // MTBUF takes its type from the instruction's combined gfx1030 BUF_FMT, not V# dword3.
    // UINT32_MAX identifies ordinary MUBUF, whose descriptor remains authoritative.
    uint32_t instruction_format = UINT32_MAX;
    // Proven VADDR source at this fetch. The scalar fold also follows the wave-uniform mask that
    // selects NGG's merged-stage VGPR inputs, allowing the recompiler to distinguish vertex_id from
    // instance_id instead of applying one gl_VertexIndex shortcut to both.
    VertexFetchIndexMode index_mode = VertexFetchIndexMode::Automatic;
};

// One descriptor use recovered by the same const-fold (#294): shaders may load their T#/S#/V#
// descriptors with `s_load_dwordx4/x8 sN, s[ptr:ptr+1], <imm>` from a resource table whose pointer
// sits in the user-data SGPRs, or consume a V# placed directly in their entry user SGPRs. They then
// use those descriptors through image_sample SRSRC/SSAMP or s_buffer_load SBASE.
// The recompiler tags such a load's dest SGPRs with the load IMMEDIATE (sreg_srt) and resolves the
// consumer via by_srt_offset(imm) — so `key` here is exactly that immediate, and build_stage_table
// turns each use into a ShaderResource with srt_offset = key.
struct SrtUse {
    int kind = 0;                    // 0 = texture, 1 = constant buffer, 2 = BVH buffer,
                                     // 3 = proven guarded null BVH
    uint32_t key = 0;                // the s_load immediate byte offset (== emit_alu's sreg_srt tag);
                                     // 0xFFFFFFFF = key-less (direct entry V#, register-SOFFSET, or
                                     // negative-imm load; resolve by the exact instruction pc instead)
    std::array<uint32_t, 8> t8{};    // T# dwords as loaded (kind 0)
    std::array<uint32_t, 4> v4{};    // V# dwords as loaded (kind 1)
    std::array<uint32_t, 4> bvh4{};  // BVH descriptor dwords live at IMAGE_BVH_INTERSECT_RAY (kind 2)
    uint32_t instruction_format = UINT32_MAX; // MTBUF BUF_FMT override for kind 1
    // Fully-known raw/untyped MUBUF V# whose NUM_RECORDS is zero. This covers ordinary RAW
    // loads/stores and the emitter-supported 32-bit atomic set; typed and unsupported atomic shapes
    // remain fail-closed. It is distinct from the scalar-SMEM raw-pointer convention, where a zero
    // decoded size means "unbounded" and required_size supplies the proven access span. Only
    // resolve_dynamic_fetch may set this after decoding all four live descriptor words at the exact
    // consuming instruction.
    bool zero_record_raw = false;
    bool has_samp = false;
    std::array<uint32_t, 4> s4{};    // paired S# dwords (kind 0, when the SSAMP load also resolved)
    // Minimum byte span needed by this scalar-buffer consumer, measured from V#.Base. Some PS5
    // scalar descriptors carry usable base bits but do not expose a conventional bounded V# size;
    // the exact observed access span is sufficient for their pc-keyed upload.
    uint32_t required_size = 0;      // kind 1 only
    // PER-USE pc provenance (#273 — DOLL's title-composite image_sample_b): the pc of the consuming
    // image op. The load-immediate key model breaks when the same immediate appears against two
    // different table pointers (a key-0 EUD sharp colliding with a key-0 table T#) or when the load
    // has no usable key; keying the TEXTURE use by its exact instruction (ShaderResource::fetch_pc,
    // the same per-instruction provenance the vertex fetches use) is unambiguous.
    uint32_t use_pc = 0xFFFFFFFFu;   // exact consuming pc for key-less texture/buffer uses
    // The consuming image op requires a STORAGE image (image_store 0x08, image_store_mip 0x09, or
    // an image atomic such as IMAGE_ATOMIC_SWAP 0x0f), not a sampled texture. Only meaningful for
    // kind 0.
    bool is_storage_image = false;
    // IMAGE_LOAD_MIP / IMAGE_STORE_MIP have one more address operand than their base-level
    // siblings. The current Vulkan compute backend materializes one mip only, so the fold may
    // specialize that operand away only after proving its exact VGPR was written in the same basic
    // block by a plain v_mov_b32 from a known-zero wave-uniform SGPR.
    bool proven_zero_mip = false;
    // The consuming MIMG opcode is a comparison/depth sample (IMAGE_SAMPLE_C*). This is a
    // property of the use, not merely the S# compare function: NEVER is a valid compare op.
    bool is_depth_compare = false;
};

// Materialization half of the IMAGE_*_MIP specialization contract. Kept observable so regression
// tests can mutate the exact DCC/single-level gate used by live resource construction.
bool shader_resource_allows_zero_mip_specialization(
    const SrtUse& use, const DecodedImageDescriptor& descriptor,
    const DecodedImageView& view);
std::vector<DynFetch> resolve_dynamic_fetch(const uint32_t* code, size_t dwords,
                                            const uint32_t* user_sgprs, uint32_t nsgpr,
                                            uint32_t user_sgpr_base,
                                            std::vector<SrtUse>* srt_uses = nullptr,
                                            uint32_t pcrel_dispatch_target = UINT32_MAX,
                                            const PcrelDispatchInfo* pcrel_dispatch = nullptr,
                                            const uint32_t* system_sgprs = nullptr,
                                            uint32_t nsystem_sgprs = 0);

// Add instruction-provenance compute buffer resources to a metadata-built table. This is the exact
// buffer-discovery path used by realize_compute_dispatches; it is exposed so tests can assert the
// final resource identities instead of manually rebuilding a lookalike table. Returned SrtUses also
// contain image uses, which the production caller materializes with image-specific view handling.
std::vector<SrtUse> add_compute_buffer_resources(ShaderResourceTable& table,
                                                 const uint32_t* code, size_t dwords,
                                                 const uint32_t* user_sgprs, uint32_t nsgpr,
                                                 uint32_t linear_local_x = 0,
                                                 uint32_t linear_threads_x = 0,
                                                 uint32_t tgid_x_sgpr = UINT32_MAX);

// Apply the exact dispatch-scoped resource-path specialization used by the live compute executor.
// The report makes the production decision observable to tests and diagnostics: callers can verify
// which raw PCs disappeared and which instruction-scoped resources were removed before translation.
// A proven-null marker is retained even after its fetch disappears because recompile_compute repeats
// the same proof from raw shader bytes; dropping that marker would silently undo the specialization.
struct ComputeResourcePathSpecializationReport {
    size_t proven_null_exits = 0;
    size_t shader_constant_branches = 0;
    size_t removed_resources = 0;
    std::vector<uint32_t> removed_pcs;
};
ComputeResourcePathSpecializationReport specialize_compute_resource_paths(
    std::vector<Rdna2Inst>& instructions, ShaderResourceTable& resources,
    uint32_t wave_size);

// The dynamic descriptor fold and shader-cache key builder both walk immutable shader instructions
// on every draw. Cache only the decoded instructions, validating the complete consumed byte range on
// every hit. Concrete SGPR values and descriptor-table memory remain per-draw inputs to the fold.
struct ShaderDecodeCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bypasses = 0;
    uint64_t invalidations = 0;
    uint64_t evictions = 0;
    uint64_t entries = 0;
    uint64_t bytes = 0;
};
ShaderDecodeCacheStats shader_decode_cache_stats();
void clear_shader_decode_cache();

// Immutable shader analysis shared by resource folding, interpolation discovery, and recompile-key
// construction. Hits validate the complete code/table span byte-for-byte before reusing its bounds
// and PC-relative dispatch metadata.
struct ShaderAnalysisCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bypasses = 0;
    uint64_t invalidations = 0;
    uint64_t evictions = 0;
    uint64_t entries = 0;
    uint64_t bytes = 0;
};
ShaderAnalysisCacheStats shader_analysis_cache_stats();
void clear_shader_analysis_cache();

// PROSPER_DYNTRACE_FAIL support (gpu_executor.cpp): while true, resolve_dynamic_fetch traces its
// walk and build_stage_table dumps the user-data SGPR blocks. realize_draw_item sets it around a
// replay of a FAILED vertex-stage resource build, so the diagnostic captures exactly the failing
// draw without needing the shader's address up front (the UI draws it targets are rare/phase-bound).
extern bool g_dyntrace_force;

// True when failed-shader tracing is enabled for this exact program. With no address filter this
// preserves the historical behavior of tracing every distinct failed program; the optional filter
// keeps a long boot's diagnostic volume bounded to one known shader.
bool dyntrace_failed_shader_enabled(uint64_t code_addr);

// Assign each resource in `t` a descriptor binding from `first`: constant/vertex buffers first, then
// textures / storage images — never on binding 2 or 3 (the recompiler's two hardwired cbufs) so a
// texture-first shader can't collide two descriptor types at one binding (#157). Exposed for testing.
void assign_convention_bindings(ShaderResourceTable& t, uint32_t first);

// Join the resource contracts for a separately-installed vertex-fetch prolog and main shader.
// Per-instruction provenance in the main table is rebased to its linked PC, then the complete table
// receives one collision-free VS binding layout. Null inputs are accepted.
std::shared_ptr<ShaderResourceTable> merge_vertex_chain_resource_tables(
    const std::shared_ptr<ShaderResourceTable>& prolog,
    const std::shared_ptr<ShaderResourceTable>& main,
    uint32_t main_pc_offset);

// Build a shader stage's resource table from the folded GpuState: look up the registered shader header
// by its bound code address, read its user-data SGPR block from the sh register file, decode the V#/T#/S#
// descriptors, and assign bindings matching the recompiler+backend convention (constant buffer -> binding
// 2, vertex buffer -> binding 3, textures -> binding 4+). Returns null if the stage has no shader header
// or no resources. `draw_vertex_count` bounds dynamic descriptors whose V# publishes zero records;
// indexed draws are grown to their decoded max-index range later in realize_draw_item. Implemented in
// gpu_executor.cpp (needs the AGC registry + descriptor decode).
std::shared_ptr<ShaderResourceTable> build_stage_table(const GpuState& st, uint64_t code_addr,
                                                       bool is_ps, uint32_t draw_vertex_count = 0);

// PROSPER_COMPUTELOG diagnostic: resolve every skipped DispatchDirect packet's compute shader and
// AGC resource table from its retained register snapshot. PROSPER_COMPUTELOG_DIM=WxH restricts output
// to dispatches referencing an image of that size (for example the Messenger 1024x32 grading LUT).
void diagnose_compute_dispatches(const GpuState& st, uint64_t submit_no);

struct ComputeLaunchDimensions {
    uint32_t threads_x = 0, threads_y = 0, threads_z = 0;
    uint32_t local_x = 1, local_y = 1, local_z = 1;
    uint32_t groups_x = 0, groups_y = 0, groups_z = 0;
};

// Resolve sceAgcCbDispatch dimensions using the dispatch modifier. USE_THREAD_DIMENSIONS=1 means
// the API values are total threads and must be ceil-divided by COMPUTE_NUM_THREAD_*; when clear,
// they are already hardware/Vulkan workgroup counts. Zero local-size registers fall back to one.
ComputeLaunchDimensions resolve_compute_launch(const GpuState::Dispatch& dispatch);

// A small set of exact, structurally recognized compute programs can execute directly against
// their guest backing. Classification is based on the raw RDNA2 instruction stream, never a title
// address; an unrecognized or partially matching program always uses the ordinary Vulkan backend.
enum class ComputeCpuFastPath : uint8_t {
    None,
    FillSgprUvec4,
};

ComputeCpuFastPath classify_compute_cpu_fast_path(const uint32_t* code, size_t dwords);

// Read the exact bound compute program from a dispatch's retained register snapshot, falling back to
// the submit state for hand-built records without a snapshot. Capture selectors use this before DMA-
// ordered realization, so a target program remains discoverable even when the pre-realized list is empty.
uint64_t compute_dispatch_code_addr(const GpuState& submit,
                                    const GpuState::Dispatch& dispatch);

struct ComputeItem {
    std::vector<uint32_t> spirv;
    std::vector<uint32_t> user_sgprs;
    std::shared_ptr<ShaderResourceTable> resources;
    ComputeLaunchDimensions launch;
    uint64_t code_addr = 0;
    uint64_t dispatch_index = 0;
    uint64_t submit_no = 0;
    uint64_t command_order = 0;
    uint32_t required_subgroup_size = 0;
    ComputeCpuFastPath cpu_fast_path = ComputeCpuFastPath::None;
    // Capture v39 retains the raw compute program and every semantic launch/recompiler input. The
    // stored SPIR-V remains the default replay artifact; --recompile-raw may rebuild it with the
    // current translator and replay device's optional format/subgroup capabilities.
    uint32_t raw_shader_index = 0xFFFFFFFFu;
    ComputeShaderConfig recompile_config{};
    bool recompile_config_available = false;
};

enum class SubmitOperationKind : uint8_t { Draw, Dispatch, DmaCopy };
struct SubmitOperation {
    SubmitOperationKind kind = SubmitOperationKind::Draw;
    size_t index = 0;
    uint64_t command_order = 0;
};

// Offline captures keep guest addresses as stable identities while their bytes live in owned host
// storage. This replay-facing DMA item binds both views so ordered execution can mutate the exact
// resource instances consumed by later draws/dispatches and invalidate renderer caches by guest VA.
struct ReplayDmaCopy {
    uint64_t dst = 0, src = 0;
    uint32_t bytes = 0, sels = 0;
    uint64_t command_order = 0;
    uint64_t packet_addr = 0;
    uint8_t* destination_data = nullptr;
    const uint8_t* source_data = nullptr;
    uint64_t destination_size = 0;
    uint64_t source_size = 0;
};

enum class ShaderProgramStage : uint8_t { Vertex, Fragment, Compute };

// Graphics shaders are commonly submitted dozens of times per frame with different guest backing
// addresses but the same code and descriptor interface. Cache the deterministic RDNA2 -> SPIR-V
// result by shader bytes plus the resource fields the recompiler actually consumes. Runtime resource
// state (addresses, sizes, dimensions, sampler state, and host backing) deliberately stays out of the
// key; the backend reads it from each draw's current ShaderResourceTable.
struct ShaderRecompileCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bypasses = 0;
    uint64_t evictions = 0;
    uint64_t entries = 0;
    uint64_t bytes = 0;
    double compile_ms = 0.0;
};
std::vector<uint32_t> recompile_graphics_shader_cached(ShaderProgramStage stage,
                                                       const uint32_t* code, size_t dwords,
                                                       const ShaderResourceTable* resources = nullptr,
                                                       const PixelInputMapping* pixel_inputs = nullptr,
                                                       const PixelSystemInputMapping* system_inputs = nullptr,
                                                       uint64_t* cache_identity = nullptr,
                                                       bool fragment_wave32 = false,
                                                       uint32_t vertex_lds_dwords = 0,
                                                       bool vertex_capture_position = false);
SharedShaderWords recompile_graphics_shader_cached_shared(
    ShaderProgramStage stage, const uint32_t* code, size_t dwords,
    const ShaderResourceTable* resources = nullptr,
    const PixelInputMapping* pixel_inputs = nullptr,
    const PixelSystemInputMapping* system_inputs = nullptr,
    uint64_t* cache_identity = nullptr,
    bool fragment_wave32 = false,
    uint32_t vertex_lds_dwords = 0,
    bool vertex_capture_position = false);
// Compute uses the same bounded content-addressed cache as graphics. Launch geometry that changes
// generated SPIR-V participates in the key; per-dispatch push-constant values deliberately do not.
std::vector<uint32_t> recompile_compute_shader_cached(
    const uint32_t* code, size_t dwords, const ShaderResourceTable* resources,
    const ComputeShaderConfig& config, uint64_t* cache_identity = nullptr,
    RecompileDiagnosticContext diagnostic = {RecompileDiagnosticStage::Compute, 0});
// Report the final live consequence once per guest program address. Returns true only for the first
// report so the caller can keep its adjacent resource-table dump on the same distinct-address gate.
bool report_compute_recompile_skip_once(RecompileDiagnosticContext diagnostic);
SharedShaderWords recompile_vertex_chain_cached_shared(
    const uint32_t* prolog, size_t prolog_dwords,
    const uint32_t* main, size_t main_dwords,
    const ShaderResourceTable* resources = nullptr,
    const PixelInputMapping* pixel_inputs = nullptr,
    uint64_t* cache_identity = nullptr,
    uint32_t vertex_lds_dwords = 0,
    bool capture_position = false);
ShaderRecompileCacheStats shader_recompile_cache_stats();
void clear_shader_recompile_cache();

FragmentInterpolationLayout fragment_interpolation_layout_cached(
    const uint32_t* code, size_t dwords,
    const PixelSystemInputMapping* system_inputs = nullptr,
    const PixelInputMapping* pixel_inputs = nullptr);

struct DrawRealizationPhaseStats {
    uint64_t draws = 0;
    double table_ms = 0.0;
    double shader_ms = 0.0;
};
void record_draw_realization_phases(double table_ms, double shader_ms);
DrawRealizationPhaseStats draw_realization_phase_stats();

struct StageTablePhaseStats {
    uint64_t calls = 0;
    double metadata_ms = 0.0;
    double dynamic_fold_ms = 0.0;
    double resources_ms = 0.0;
};
void record_stage_table_phases(double metadata_ms, double dynamic_fold_ms, double resources_ms);
StageTablePhaseStats stage_table_phase_stats();

struct ParallelDrawRealizationStats {
    uint64_t batches = 0;
    uint64_t semantic_draws = 0;
    uint64_t worker_threads = 0;  // sum of participating threads, including the submit thread
    double wall_ms = 0.0;
};
ParallelDrawRealizationStats parallel_draw_realization_stats();

// APPEND ONLY, and update kMaxRealizationFailureReason below. The value is serialized as a raw byte
// into .prgcap, and both the in-memory validator and the reader bound-check it against that maximum.
enum class RealizationFailureReason : uint8_t {
    None,
    Unknown,
    MissingProgram,
    ShaderRecompile,
    DescriptorContract,
    NoEffect,
    ZeroVertices,
    Filtered,
    // The three exits below are reached in the ordered-submit path BEFORE realize_draw_item runs, so
    // they can never carry a shader/pipeline diagnostic. They are distinct reasons rather than
    // Unknown because "nothing was attempted, and here is why" is the answer an investigation needs
    // (#1636) — collapsing them to Unknown is indistinguishable from the reason being lost.
    RetainedDrawNotSelected,   // the retained-submit policy did not select this draw index
    IndirectArguments,         // indirect DRAW arguments could not be resolved. Dispatches
                               // have the same failure but realize_retained_compute still
                               // returns a bare false for it, so they remain Unknown.
    IndirectDependencies,      // an indirect operation's producer had not landed for this submit
};
inline constexpr RealizationFailureReason kMaxRealizationFailureReason =
    RealizationFailureReason::IndirectDependencies;

// Capture-facing facts collected at the exact point an operation is dropped. These contain no raw
// shader bytes; gpu_capture reads those through its fault-safe, size-bounded memory reader.
struct ShaderRealizationDiagnostic {
    ShaderProgramStage stage = ShaderProgramStage::Vertex;
    uint64_t program_addr = 0;
    std::shared_ptr<ShaderResourceTable> resources;
    RecompileCoverage coverage;
    bool recompiled = false;
    uint32_t descriptor_issue_count = 0;
    uint32_t first_descriptor_issue = 0xFFFFFFFFu;
    // Failed compute recompilation depends on the complete dispatch ABI, not only the raw program
    // and its resource table. Graphics stages leave this unavailable.
    ComputeShaderConfig recompile_config{};
    bool recompile_config_available = false;
};

struct OperationRealizationFailure {
    SubmitOperationKind kind = SubmitOperationKind::Draw;
    size_t index = 0;
    uint64_t command_order = 0;
    RealizationFailureReason reason = RealizationFailureReason::None;
    bool pipeline_present = false;
    ResolvedPipelineState pipeline;
    uint64_t color0_base = 0;
    uint32_t color0_width = 0;
    uint32_t color0_height = 0;
    uint64_t color1_base = 0;
    uint32_t color1_width = 0;
    uint32_t color1_height = 0;
    std::array<DrawItem::ColorTargetBinding, kColorTargetCount> color_targets{};
    uint32_t vertex_count = 0;
    ComputeLaunchDimensions compute_launch;
    std::vector<ShaderRealizationDiagnostic> stages;
};

using LiveComputeFn = std::function<bool(const std::vector<ComputeItem>& items)>;

// Behavior-neutral ordered-boundary observer for the live compute-authority census (#1854).  The
// executor reports only architectural submit positions and exact guest ranges it already owns;
// it never asks the observer whether work should execute.  Draw ranges are deliberately unknown
// until resource consumption is proven precisely enough to make a narrower claim.
enum class ComputeAuthorityBoundaryKind : uint8_t {
    SubmitBegin,
    Draw,
    // Post-realization evidence for the conservative Draw event above. One event names one exact
    // shader-resource backing range; DrawResourceEnd closes that draw and records whether it was
    // realized. These are diagnostic-only and never relax the pre-realization fail-closed boundary.
    DrawResource,
    DrawResourceEnd,
    Dma,
    OrderedMemoryEffect,
    Capture,
    SubmitEnd,
    // A compute operation that could not reach the live backend's exact finalized-resource
    // observations, or an exact CPU-fast write range. Unknown range fails closed; a known range is
    // treated as an ordered compute memory effect.
    Compute,
};
struct ComputeAuthorityBoundary {
    ComputeAuthorityBoundaryKind kind = ComputeAuthorityBoundaryKind::SubmitBegin;
    uint64_t submit_no = 0;
    uint64_t command_order = 0;
    uint64_t address = 0;
    uint64_t bytes = 0;
    bool range_known = false;
    uint32_t binding = UINT32_MAX;
    uint32_t resource_class = UINT32_MAX;
    bool draw_realized = false;
};
// Exact post-realization shader-resource evidence for one draw, followed by DrawResourceEnd. This
// planner does not claim that pre-realization descriptor/index/shader reads are covered; callers use
// it only to audit which concrete draw resources overlap an already fail-closed boundary.
std::vector<ComputeAuthorityBoundary> compute_authority_draw_resource_boundaries(
    const DrawItem& item, uint64_t submit_no);
using ComputeAuthorityBoundaryObserver =
    std::function<void(const ComputeAuthorityBoundary& boundary)>;
void set_compute_authority_boundary_observer(ComputeAuthorityBoundaryObserver observer);
void notify_compute_authority_boundary(const ComputeAuthorityBoundary& boundary);

// Guest GPU writes can change backing memory represented by a persistent host-side image. Backends
// register one observer so guest-memory-producing backends can invalidate overlapping cached surfaces
// without making prosper_core depend on Vulkan.
using GuestGpuWriteObserver = std::function<void(uint64_t addr, uint64_t size)>;
void set_guest_gpu_write_observer(GuestGpuWriteObserver observer);
void notify_guest_gpu_write(uint64_t addr, uint64_t size);
// A backend can prove that a dispatched write reproduced the exact guest bytes while renderer-owned
// aliases at the same address may still hold divergent state. Notify those alias owners without
// dirtying guest-byte watches or the in-submit mutation journal.
void notify_guest_gpu_write_preserving_bytes(uint64_t addr, uint64_t size);

// A validation snapshot is meaningful only inside one synchronous execute_ordered_items call.
// It lets a backend prove that no retained GPU operation wrote a resource between graphics spans;
// CPU writes and later submits deliberately remain outside this proof and must use exact validation.
constexpr size_t kGuestGpuWriteJournalCapacity = 4096;
struct GuestGpuWriteSnapshot {
    uint64_t submit_serial = 0;
    size_t write_count = 0;
};
enum class GuestGpuWriteQuery {
    Unchanged,
    Overlap,
    Unknown,
};
GuestGpuWriteSnapshot guest_gpu_write_snapshot();
GuestGpuWriteQuery guest_gpu_writes_since(const GuestGpuWriteSnapshot& snapshot,
                                           uint64_t addr, uint64_t size);

// Whether the per-submit write journal is armed ON THIS THREAD (#2289). Diagnostic only.
//
// guest_gpu_writes_since() returns Unknown for two structurally different reasons and a caller
// cannot tell them apart: the journal is not armed here at all, or it IS armed but the snapshot
// carries a different submit serial. The second is not a defect -- the journal is intra-submit by
// construction, so a snapshot taken in an earlier submit can only ever answer Unknown. Conflating
// them is how a cross-submit cost gets mistaken for missing instrumentation.
//
// The journal is thread_local, so this answers for the calling thread and nothing else.
bool guest_gpu_write_tracking_active();

// Register the synchronous live compute backend. execute_compute_dispatches realizes every retained
// dispatch from its state snapshot and invokes the backend in stream order.
void set_submit_compute(LiveComputeFn fn);
bool have_submit_compute();

// Live render-target query (#590): the compute backend must not read a sampled input from raw guest
// memory when the LIVE RENDERER owns that surface's current pixels (an RTT color target — raw memory
// is then empty/stale, the Dead Cells 642x362 lesson). The live renderer registers this; the compute
// backend imports an immutable CPU snapshot for sampled bindings. Writable bindings seed from the
// snapshot, publish their result to guest backing, and notify the renderer to invalidate its copy.
using LiveTargetQueryFn = std::function<bool(uint64_t gpu_addr)>;
void set_live_target_query(LiveTargetQueryFn fn);
bool is_live_render_target(uint64_t gpu_addr);
enum class LiveTargetPixelFormat : uint8_t { Rgba8Unorm, Rgba16Float, R11G11B10Float };
struct LiveTargetSnapshot {
    uint32_t width = 0, height = 0;
    LiveTargetPixelFormat format = LiveTargetPixelFormat::Rgba8Unorm;
    std::shared_ptr<const std::vector<uint8_t>> pixels;
};
using LiveTargetReaderFn = std::function<bool(uint64_t gpu_addr, LiveTargetSnapshot& snapshot)>;
void set_live_target_reader(LiveTargetReaderFn fn);
bool read_live_render_target(uint64_t gpu_addr, LiveTargetSnapshot& snapshot);

// Direct GPU import of a renderer-owned color target (#1095, phase 2 of #1091). Once graphics and
// compute share one VkDevice, a dispatch that SAMPLES a renderer-owned surface can read the
// renderer's persistent image in place instead of forcing a GPU->CPU readback and an immediate
// re-upload of the very same pixels to the very same device.
//
// AUTHORITY RULE: the import is offered only while the persistent Vulkan image is current. An
// ordered GPU readback may leave an equivalent CPU snapshot beside it; that mirror does not make the
// image stale. CPU-newer publications clear gpu_valid, and guest GPU writes invalidate the cache
// entry before import (#780), so neither can expose an obsolete image through this contract.
//
// The image is BORROWED: the caller must not destroy it, must restore `layout` before returning, and
// must call release_live_render_target_image() to drop the pin that keeps the renderer's cache from
// evicting the entry mid-dispatch. Handles stay opaque so this layer keeps no Vulkan dependency.
struct LiveTargetImageImport {
    enum class Kind : uint8_t { Color, Depth };
    uint32_t width = 0, height = 0;
    LiveTargetPixelFormat format = LiveTargetPixelFormat::Rgba8Unorm;
    Kind kind = Kind::Color;
    // Opaque VkFormat for depth imports. Color imports keep using `format`, whose deliberately
    // small enum is part of the renderer/compute numeric-compatibility contract.
    uint32_t native_format = 0;
    void* image = nullptr;    // VkImage owned by the live renderer
    void* device = nullptr;   // VkDevice it belongs to; the caller must be running on that device
    uint32_t layout = 0;      // VkImageLayout the renderer left it in -- restore it after the dispatch
    // Explicit image-creation contract: a sampled import is not otherwise guaranteed to carry
    // VK_IMAGE_USAGE_TRANSFER_DST_BIT, which the compute-result mirror requires.
    bool transfer_dst = false;
    bool valid() const { return image && device && width && height; }
};
struct LiveTargetImageRequest {
    uint32_t width = 0, height = 0;
    uint32_t render_scale = 1;
    bool allow_depth = false;
    // True only when reflection proves this descriptor is read exclusively through normalized
    // sample/gather operations. Integer image fetch/read must retain the exact declared extent.
    bool normalized_sampling = false;
};
using LiveTargetImageImportFn = std::function<bool(
    uint64_t gpu_addr, const LiveTargetImageRequest& request, LiveTargetImageImport& import)>;
using LiveTargetImageReleaseFn = std::function<void(uint64_t gpu_addr)>;
struct LiveTargetImageWrite {
    uint64_t gpu_addr = 0;
    uint32_t width = 0, height = 0;
    LiveTargetPixelFormat format = LiveTargetPixelFormat::Rgba8Unorm;
    bool valid() const { return gpu_addr && width && height; }
};
using LiveTargetImageWrittenFn = std::function<void(const LiveTargetImageWrite& write)>;
void set_live_target_image_importer(LiveTargetImageImportFn import_fn,
                                    LiveTargetImageReleaseFn release_fn);
void set_live_target_image_written_notifier(LiveTargetImageWrittenFn written_fn);
bool import_live_render_target_image(uint64_t gpu_addr, const LiveTargetImageRequest& request,
                                     LiveTargetImageImport& import);
void release_live_render_target_image(uint64_t gpu_addr);
// Publish that a borrowed renderer image received the completed compute result. The caller may also
// have written the exact guest bytes for alias correctness; the renderer processes that normal
// invalidation first, then re-authorizes only this exact image identity and discards an older CPU
// readback mirror.
void notify_live_render_target_image_written(const LiveTargetImageWrite& write);
// Shared Vulkan device (#1091 phase 1). live_compute historically created its own VkInstance/VkDevice,
// so a renderer-owned image could never be bound to a dispatch and had to round-trip through host
// memory. The live renderer publishes its already-created device here; the compute backend ADOPTS it
// when it is present and feature-adequate, and otherwise creates its own exactly as before (compute
// must keep working with no renderer at all -- see tests/test_game_compute.cpp).
//
// Handles are opaque so this layer keeps no Vulkan dependency; both sides cast to the real types.
// An adopted context is BORROWED: the consumer owns none of these objects and must not destroy them.
// Graphics and compute execute strictly sequentially on one thread (execute_ordered_gpustate), so a
// shared queue needs no cross-thread synchronization.
struct SharedVulkanContext {
    void* instance = nullptr;
    void* physical = nullptr;
    void* device = nullptr;
    void* queue = nullptr;
    uint32_t queue_family = UINT32_MAX;
    // Features the compute backend requires; when false it must decline the shared device.
    bool storage_image_read_without_format = false;
    bool storage_image_write_without_format = false;
    // Exact compute-wave acceleration. These describe features ENABLED on the borrowed device, not
    // merely physical support. The recompiler opts in only when the requested guest wave is inside
    // this range, full compute subgroups can be required, and both vote and arithmetic subgroup
    // operations are available.
    bool compute_subgroup_size_control = false;
    bool compute_full_subgroups = false;
    bool compute_subgroup_vote = false;
    bool compute_subgroup_arithmetic = false;
    uint32_t min_compute_subgroup_size = 0;
    uint32_t max_compute_subgroup_size = 0;
    // Runtime-selected descriptors (#2412). Same contract as the wave fields above: this is a capability
    // ENABLED on the borrowed device, not merely one the physical device supports. The compute backend
    // must not re-query the physical device for it, because whether the features were requested at device
    // creation is the renderer's decision -- a physically capable device whose owner declined them cannot
    // execute an indexed descriptor array, and asking the driver would answer the wrong question.
    bool descriptor_indexing = false;
    uint32_t max_compute_workgroup_subgroups = 0;
    uint32_t max_compute_workgroup_size_x = 0;
    uint32_t max_compute_workgroup_invocations = 0;
    // Vulkan-independent mask from native_storage_format_support_bit() and its 3D counterpart. The
    // renderer queries dimension-specific optimal-tiling STORAGE_IMAGE support before publishing
    // its physical device.
    uint32_t native_storage_format_support = 0;
    bool compute_queue_supported = false;
    // Present unification (#1270): when present_capable, prosper-app may create its window surface on
    // `instance`, its swapchain on `device`, and present on `present_queue` -- then blit the renderer's
    // front-buffer image straight to the swapchain with no CPU round-trip. present_queue_shared means the
    // present queue aliases the render queue, so both threads must serialize submits through
    // shared_render_submit_mutex(). false/nullptr on a headless build (prosper-app then uses its own
    // separate present device + CPU pixels, exactly as before).
    bool present_capable = false;
    void* present_queue = nullptr;
    bool present_queue_shared = false;
    bool valid() const { return instance && physical && device && queue && queue_family != UINT32_MAX; }
};
// Select the exact native subgroup contract only when the renderer device can actually be adopted
// by compute and every required-subgroup workgroup bound is satisfied. Single-wave workgroups are
// the default proven fast path; the caller may allow a multi-wave program based on a structural
// shader proof or the explicit experimental opt-in. `disabled` is the diagnostic/compatibility
// opt-out.
uint32_t select_native_compute_subgroup_size(const SharedVulkanContext& context,
                                             const ComputeShaderConfig& config,
                                             bool allow_multiwave, bool disabled);
void set_shared_vulkan_context(const SharedVulkanContext& context);
SharedVulkanContext shared_vulkan_context();

// Present unification (#1270): when prosper-app presents on the SAME VkQueue the renderer/compute submit
// on (present_queue_shared), the renderer's guest thread and the app's present thread would call
// vkQueueSubmit/vkQueueWaitIdle/vkQueuePresentKHR on one queue concurrently -- undefined without external
// synchronization. This mutex serializes those host CALLS (not GPU waits, which don't touch the queue).
// It is a no-op until the app adopts the shared queue and calls set_shared_present_active(true): every
// renderer/compute submit site takes it only when shared_present_active() is true, so the headless
// test/screenshot path pays a single relaxed atomic load and no lock. GPU-side render->present ordering
// is handled by a semaphore, independently of this mutex.
std::mutex& shared_present_submit_mutex();
void set_shared_present_active(bool active);
bool shared_present_active();

// Present unification (#1270): true once prosper-app has adopted the render device and is consuming the
// renderer's front-buffer image directly (present_blit). While true the renderer publishes the front image
// via present_blit_publish and SKIPS the CPU readback+reupload of the scanout buffer. Distinct from
// shared_present_active: GPU present can run on a dedicated present queue (no shared-queue mutex needed).
// Stays false in every headless/test/screenshot process, so their CPU present path is byte-for-byte
// unchanged. Set once by prosper-app after a successful adoption; never cleared mid-run.
void set_gpu_present_active(bool active);
bool gpu_present_active();

// Ordered memory producers need the same authoritative storage version as live compute. A source
// may begin inside a target, so the renderer validates the complete requested byte range instead of
// exposing an unbounded pointer into its cache.
enum class LiveTargetByteReadResult : uint8_t { NotFound, Success, InvalidRange };
using LiveTargetByteRangeReaderFn = std::function<LiveTargetByteReadResult(
    uint64_t gpu_addr, uint32_t bytes, std::vector<uint8_t>& output)>;
void set_live_target_byte_range_reader(LiveTargetByteRangeReaderFn fn);
LiveTargetByteReadResult read_live_render_target_bytes(uint64_t gpu_addr, uint32_t bytes,
                                                       std::vector<uint8_t>& output);
std::vector<ComputeItem> realize_compute_dispatches(const GpuState& st,
                                                     uint64_t submit_no = 0,
                                                     std::vector<OperationRealizationFailure>* failures = nullptr);
bool execute_compute_dispatches(const GpuState& st, uint64_t submit_no = 0);
// Execute retained dispatches and address-backed DMA copies in PM4 order when graphics rendering is
// intentionally skipped or unavailable. Draw operations are omitted, but still delimit ordering.
bool execute_nonrender_submit_work(const GpuState& st, uint64_t submit_no = 0);
std::vector<SubmitOperation> plan_submit_operations(const GpuState& st);

// PROSPER_PROVENANCE_DIM=WxH: inspect sampled images of that size and report overlapping
// color, compute, DMA_DATA, and WRITE_DATA events with both observation and PM4 ordering.
// PROSPER_PROVENANCE_MIN_DRAWS=N limits expensive descriptor resolution to large target submits.
void diagnose_resource_provenance(const GpuState& st, uint64_t submit_no);

// PROSPER_DESCRIPTOR_VALIDATE=warn|strict|poison. Reflect the generated stage module and compare
// every statically-used descriptor with its runtime table before Vulkan sees the draw. Strict rejects
// errors; warn and poison report and continue (the live backend applies poison substitutions).
bool validate_runtime_descriptor_contract(const char* stage_name,
                                           const std::vector<uint32_t>& spirv,
                                           const ShaderResourceTable* runtime,
                                           uint32_t expected_set,
                                           SpirvShaderStage expected_stage);

// Same, with the mode string supplied by the caller instead of read here.
//
// For per-draw callers only, and the reason is measured rather than assumed. The form above reads
// getenv on EVERY call and is invoked twice per draw, so the read is the entire cost of the fast
// path: when the variable is unset the function returns on the very next line. On this Windows host
// one missing getenv costs 1.26 us (86 environ entries, ~15 ns each on a linear scan), so at the
// 2,179 draws/submit recorded beside the poison_R hoist in live_renderer.cpp that is 2 x 2,179 x
// 1.26 us = 5.49 ms/submit spent discovering that a diagnostic is switched off. The renderer's own
// build_resources partition measured this leaf at 5.77 ms/submit -- a 5% agreement with a number
// derived independently, which is what established that the leaf is the read and not the work.
//
// `mode` must come from a live getenv hoisted to submit scope, NOT from PROSPER_ENV_VALUE:
// test_gpu_capture_render.cpp and test_shader_resources.cpp arm PROSPER_DESCRIPTOR_VALIDATE at
// runtime, and a process-lifetime cache would leave their second write unobserved -- the #2214
// defect that `cached_env_arming_logic` gates. Passing nullptr means "unset", i.e. validation off.
bool validate_runtime_descriptor_contract(const char* stage_name,
                                           const std::vector<uint32_t>& spirv,
                                           const ShaderResourceTable* runtime,
                                           uint32_t expected_set,
                                           SpirvShaderStage expected_stage,
                                           const char* mode);

// Byte size of one index element for a GpuState::index_type (the last SetIndexType value).
// 0 -> 16-bit, 1 -> 32-bit, exactly Kyty's index_type_and_size switch (GraphicsRender.cpp:4724) and
// the hardware VGT_INDEX_TYPE encoding; 0 is also the reset default, matching this title, which never
// emits SetIndexType yet packs its quad index streams 12 bytes apart (6 x 2-byte indices) — live
// capture confirms 16-bit. CONFIDENCE: HIGH for 0/1; any other value is unseen -> returns 0 and the
// caller falls back to a non-indexed draw (loudly), rather than mis-reading the buffer.
inline uint32_t index_elem_bytes(uint32_t index_type) {
    return index_type == 0 ? 2u : index_type == 1 ? 4u : 0u;
}

// Detect an UNANNOUNCED 32-bit index buffer (#304). DOLL's UE4 Slate/UMG quad index buffers are
// 32-bit, but the title never calls sceAgcDcbSetIndexSize and programs no VGT_INDEX_TYPE register,
// so index_type defaults to 16-bit — which misreads each 32-bit index as TWO 16-bit ones (the low
// half = the real index, the high half = 0), collapsing a quad's [0,1,2,2,1,3] to a degenerate
// [0,0,1,0,2,0]. The fingerprint is unmistakable and cheap: read the same buffer as 16-bit and as
// 32-bit for `n` entries; a real 32-bit buffer has EVERY odd 16-bit word (the zero high halves) == 0
// AND every 32-bit value small (< 0x10000) and not all-zero. Genuine 16-bit index buffers (DOLL's
// scene/text meshes; every Messenger quad, e.g. [0,1,2,2,3,0]) have a non-zero odd word and are
// rejected, so this never reinterprets a real 16-bit buffer. `p16` reads from the 16-bit (elem=2)
// address; `p32` from the recomputed 32-bit address — for a DrawIndexOffset they differ, but both
// land on a quad-periodic region so the fingerprint holds on either. CONFIDENCE: HIGH.
inline bool index_buffer_is_unannounced_32bit(const uint16_t* p16, const uint32_t* p32, uint32_t n) {
    if (n < 2) return false;                         // need at least one odd word to test
    bool odd_zero = true, all_small = true, any_nonzero = false, has_odd = false;
    // memcpy loads, NOT typed derefs: p16/p32 view the SAME guest bytes, and reading one object
    // through both element types is strict-aliasing UB — Apple Clang 21 at -O2 proved it and
    // compiled the caller into ud2 (found by the macOS port; Linux GCC happened to tolerate it).
    // Fixed-size memcpy compiles to the same single loads without the aliasing assumption.
    for (uint32_t i = 0; i < n; i++) {
        if (i & 1) {
            has_odd = true;
            uint16_t w; memcpy(&w, (const char*)p16 + 2u * i, 2);
            if (w != 0) odd_zero = false;
        }
        uint32_t d; memcpy(&d, (const char*)p32 + 4u * i, 4);
        if (d >= 0x10000u) all_small = false;
        if (d != 0) any_nonzero = true;
    }
    return has_odd && odd_zero && all_small && any_nonzero;
}

// #1163: choose a NON-INDEXED draw's vertex count. A DrawIndexAuto packet's count (draw_count) is the
// AUTHORITATIVE hardware vertex count — the GPU draws exactly that many vertices with auto indices
// 0..draw_count-1. The bound vertex buffer's record count (vb_records = size/stride) is ONLY a fallback for
// a draw that supplied no count at all; it must NEVER override a real count. Historically (#60) the executor
// folded a whole submit into one item and applied the FIRST draw's index_count to it, so a multi-draw mesh
// rendered only that first (small) count -> a degenerate sliver; substituting the larger VB record count
// recovered the geometry. The per-draw executor now gives every draw its OWN decoded count (R_DRAW_INDEX_-
// AUTO -> index_count), so that workaround is obsolete, and substituting the VB record count is actively WRONG when
// a title binds a SHARED vertex POOL: GTA V's Scaleform UI submits per-draw counts of 3/6/30 against one
// fixed ~4096-byte pool, so vb_records = pool/stride is 146/1024/512 — sweeping the whole pool rasterizes
// hundreds of spurious triangles from stale/zero pool data. That inflated the stencil masks' coverage past
// their EQUAL==2 clip (GTA #1163's black menu wedges) and painted the same phantom vertex tail that earlier
// read as an "alpha-0 compositing" artifact. Zero-count non-indexed draws are already skipped as no-ops
// (#400) before this runs, so draw_count is non-zero in practice and the vb_records fallback is a guard.
inline uint32_t resolve_nonindexed_vertex_count(uint32_t draw_count, uint32_t vb_records) {
    return draw_count ? draw_count : vb_records;
}

struct ColorStateTraceConfig {
    bool enabled = false;
    bool filter_by_dimension = false;
    uint32_t width = 0, height = 0;
};

// Parse the process-wide diagnostic switch once. Draw realization is a hot path even when tracing is
// disabled, so its ordinary per-draw cost is only a cached boolean check -- never getenv or sscanf.
inline const ColorStateTraceConfig& color_state_trace_config() {
    static const ColorStateTraceConfig config = [] {
        ColorStateTraceConfig parsed;
        const char* setting = std::getenv("PROSPER_COLORSTATETRACE");
        if (!setting || !*setting) return parsed;
        if (std::strcmp(setting, "1") == 0 || std::strcmp(setting, "all") == 0) {
            parsed.enabled = true;
            return parsed;
        }

        char trailing = 0;
        if (std::sscanf(setting, "%ux%u%c", &parsed.width, &parsed.height, &trailing) == 2 &&
            parsed.width && parsed.height) {
            parsed.enabled = true;
            parsed.filter_by_dimension = true;
            return parsed;
        }
        std::fprintf(stderr,
                     "[color-state] invalid PROSPER_COLORSTATETRACE='%s' "
                     "(expected 1, all, or WxH)\n",
                     setting);
        return ColorStateTraceConfig{};
    }();
    return config;
}

// PROSPER_COLORSTATETRACE=1|all|WxH: one raw-to-resolved color/depth state record per matching draw.
// This deliberately runs before the no-effect fast path so a zero raw mask remains observable even
// when shader/resource realization is skipped. It is diagnostic-only and does not mutate draw state.
inline void trace_color_state_if_requested(const RenderState& rs,
                                           const ResolvedPipelineState& ps) {
    const ColorStateTraceConfig& config = color_state_trace_config();
    if (!config.enabled) return;

    const ColorStateTraceSnapshot snapshot = snapshot_color_state_trace(rs, ps);
    if (config.filter_by_dimension &&
        !color_state_trace_matches_dimension(snapshot, config.width, config.height))
        return;

    // Draw realization may run on worker threads. Keep each snapshot contiguous so target rows do
    // not become associated with another draw's raw register line in a busy live trace.
    static std::mutex trace_mutex;
    std::lock_guard lock(trace_mutex);
    std::fprintf(stderr,
                 "[color-state] es=0x%llx ps=0x%llx "
                 "cb-control=%d:%08x mode=%u target-mask=%d:%08x "
                 "shader-mask=%d:%08x\n",
                 static_cast<unsigned long long>(snapshot.es_addr),
                 static_cast<unsigned long long>(snapshot.ps_addr),
                 snapshot.has_cb_color_control, snapshot.cb_color_control,
                 snapshot.cb_color_mode, snapshot.has_cb_target_mask,
                 snapshot.cb_target_mask, snapshot.has_cb_shader_mask,
                 snapshot.cb_shader_mask);
    for (uint32_t slot = 0; slot < snapshot.color_targets.size(); ++slot) {
        const auto& target = snapshot.color_targets[slot];
        if (!target.base && !target.raw_format && !target.resolved_format &&
            !target.resolved_write_mask)
            continue;
        std::fprintf(stderr,
                     "[color-state]   color%u=0x%llx %ux%u raw-format=%u "
                     "resolved-format=%u resolved-cwm=%x\n",
                     slot, static_cast<unsigned long long>(target.base),
                     target.width, target.height, target.raw_format,
                     target.resolved_format, target.resolved_write_mask);
    }
    std::fprintf(stderr,
                 "[color-state]   depth=%d:%ux%u raw-size=%08x test=%d write=%d "
                 "z=0x%llx/0x%llx s=0x%llx/0x%llx htile=0x%llx stencil=%d\n",
                 snapshot.has_depth_extent, snapshot.depth_width, snapshot.depth_height,
                 snapshot.raw_depth_size_xy, ps.depth_test_enable, ps.depth_write_enable,
                 static_cast<unsigned long long>(snapshot.depth_read_base),
                 static_cast<unsigned long long>(snapshot.depth_write_base),
                 static_cast<unsigned long long>(snapshot.stencil_read_base),
                 static_cast<unsigned long long>(snapshot.stencil_write_base),
                 static_cast<unsigned long long>(snapshot.htile_data_base), ps.stencil_enable);
}

// Realize ONE draw of `ds` (a register snapshot or the folded state) into a DrawItem: recompile the
// VS+PS, resolve fixed-function state, and — for an indexed draw — fetch the guest index buffer.
// `draw` is the PM4 draw record (index count + indexed/index_addr); null means "no record" (hand-built
// states) and renders vcount_hint vertices non-indexed. Returns false (and leaves `out` untouched) if
// the draw is a no-op (no PGM bound, recompile failed, or no color/depth/stencil effect). Shared by the default
// (folded-state, one item) and PROSPER_PERDRAW (per-draw) paths so their per-draw handling is identical.
// `hoisted_validate_mode` (#2287): a POINTER to a PROSPER_DESCRIPTOR_VALIDATE value the caller
// already read once for the whole submit, or nullptr to read it here as before. The extra
// indirection is what distinguishes "the caller hoisted it and it was unset" (non-null pointer to a
// null string) from "the caller did not hoist" (null pointer) -- a plain `const char*` cannot say
// both, and defaulting the ambiguous case to "off" would silently disable validation for every
// caller that had not been converted.
//
// Why it is worth plumbing rather than reading here: this function is called once per draw from
// parallel_draw_worker_execute, and on Windows getenv takes a process-wide lock on every call, so
// two reads per draw across every worker is contention rather than a per-call constant -- and it
// gets worse with more workers, which is the opposite of what parallelising the loop is for.
// #2285 removed the equivalent pair from the serial build_bds path and measured the unit at
// 1.26 us/call, 5.49 ms/submit at the 2,179 draws/submit this title reaches in the FMV phase.
inline bool realize_draw_item(const GpuState& ds, const GpuState::Draw* draw, uint32_t vcount_hint,
                              uint32_t max_shader_dwords, bool log, DrawItem& out,
                              OperationRealizationFailure* failure = nullptr,
                              bool retain_shared_shader_words = false,
                              const char* const* hoisted_validate_mode = nullptr) {
    RenderState rs = extract_render_state(ds);
    if (failure) {
        *failure = {};
        failure->kind = SubmitOperationKind::Draw;
        failure->pipeline_present = true;
        failure->pipeline = resolve_pipeline_state(rs);
        failure->color0_base = rs.color0_base;
        failure->color0_width = rs.color0_width;
        failure->color0_height = rs.color0_height;
        failure->color1_base = rs.color1_base;
        failure->color1_width = rs.color1_width;
        failure->color1_height = rs.color1_height;
        for (uint32_t slot = 0; slot < failure->color_targets.size(); ++slot) {
            failure->color_targets[slot].base = rs.color_targets[slot].base;
            failure->color_targets[slot].width = rs.color_targets[slot].width;
            failure->color_targets[slot].height = rs.color_targets[slot].height;
        }
        failure->color_targets[0] = {
            failure->color0_base, failure->color0_width, failure->color0_height};
        failure->color_targets[1] = {
            failure->color1_base, failure->color1_width, failure->color1_height};
        failure->vertex_count = vcount_hint;
    }
    auto add_stage_diagnostic = [&](ShaderProgramStage stage, uint64_t addr,
                                    const std::shared_ptr<ShaderResourceTable>& resources,
                                    const std::vector<uint32_t>& spirv) {
        if (!failure) return;
        ShaderRealizationDiagnostic diagnostic;
        diagnostic.stage = stage;
        diagnostic.program_addr = addr;
        diagnostic.resources = resources;
        diagnostic.recompiled = !spirv.empty();
        if (!spirv.empty()) {
            const uint32_t expected_set = stage == ShaderProgramStage::Vertex ? 0u : 1u;
            const SpirvShaderStage expected_stage = stage == ShaderProgramStage::Vertex
                ? SpirvShaderStage::Vertex : SpirvShaderStage::Fragment;
            const DescriptorValidationReport report = validate_spirv_descriptor_interface(
                spirv, resources.get(), expected_set, expected_stage, true);
            diagnostic.descriptor_issue_count = static_cast<uint32_t>(report.issues.size());
            auto issue = std::find_if(report.issues.begin(), report.issues.end(),
                                      [](const auto& candidate) { return candidate.error; });
            if (issue == report.issues.end() && !report.issues.empty()) issue = report.issues.begin();
            if (issue != report.issues.end())
                diagnostic.first_descriptor_issue = static_cast<uint32_t>(issue->code);
        }
        failure->stages.push_back(std::move(diagnostic));
    };
    if (!rs.es_addr || !rs.ps_addr) {
        if (failure) {
            failure->reason = RealizationFailureReason::MissingProgram;
            add_stage_diagnostic(ShaderProgramStage::Vertex, rs.es_addr, {}, {});
            add_stage_diagnostic(ShaderProgramStage::Fragment, rs.ps_addr, {}, {});
        }
        if (log) fprintf(stderr, "[exec] skip draw: no PGM bound (es=0x%llx ps=0x%llx "
                                 "color0=0x%llx/%ux%u)\n",
                         (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr,
                         (unsigned long long)rs.color0_base, rs.color0_width, rs.color0_height);
        return false;
    }
    const ResolvedPipelineState resolved_pipeline = failure
        ? failure->pipeline : resolve_pipeline_state(rs);
    trace_color_state_if_requested(rs, resolved_pipeline);
    // CB_TARGET_MASK/CB_SHADER_MASK are upstream hardware gates: a fragment export can narrow these
    // masks, but cannot enable a component that they already disabled.  When fixed-function state also
    // has no depth/stencil side effect, shader/resource realization cannot change the no-op verdict.
    // Astro's late FMV/world-map submits contain hundreds of these draws, so rejecting them here avoids
    // building resource tables, walking RDNA programs, and copying cached SPIR-V merely to discard them.
    const bool preexport_color_effect = std::any_of(
        resolved_pipeline.color_targets.begin(), resolved_pipeline.color_targets.end(),
        [](const auto& target) { return target.write_mask != 0; });
    if (!preexport_color_effect && !has_depth_stencil_side_effect(resolved_pipeline) &&
        !getenv("PROSPER_FORCE_COLORWRITE") && !getenv("PROSPER_NO_EARLY_NO_EFFECT")) {
        if (failure) {
            failure->reason = RealizationFailureReason::NoEffect;
            // Keep the bound program addresses in captures without performing shader analysis.  A
            // no-effect operation does not need compiled modules or descriptor tables for replay.
            add_stage_diagnostic(ShaderProgramStage::Vertex, rs.es_addr, {}, {});
            add_stage_diagnostic(ShaderProgramStage::Fragment, rs.ps_addr, {}, {});
        }
        if (log) fprintf(stderr, "[exec] skip draw early: no color/depth/stencil effect "
                                "cb_target_mask=0x%x cb_color_control=0x%x color0_fmt=%u "
                                "color0=0x%llx/%ux%u es=0x%llx ps=0x%llx order=%llu\n",
                         rs.cb_target_mask, rs.cb_color_control,
                         resolved_pipeline.color0_format,
                         (unsigned long long)rs.color0_base, rs.color0_width, rs.color0_height,
                         (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr,
                         (unsigned long long)(draw ? draw->command_order : 0));
        return false;
    }
    const auto* fused_back = static_cast<const AgcShaderHeader*>(
        prosper_agc_fused_back_header_for_front(rs.es_addr));
    const bool phase_timing = getenv("PROSPER_RENDER_TIMING") != nullptr;
    const auto table_start = phase_timing
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto* vertex_header = static_cast<const AgcShaderHeader*>(
        prosper_agc_shader_header_for_code(rs.es_addr));
    uint64_t chain_addr = prosper_agc_shader_continuation_for_code(rs.es_addr);
    if (!chain_addr && rs.gs_addr && rs.gs_addr != rs.es_addr) chain_addr = rs.gs_addr;
    const auto* chain_header = chain_addr
        ? static_cast<const AgcShaderHeader*>(prosper_agc_shader_header_for_code(chain_addr))
        : nullptr;
    auto bounded_shader_dwords = [&](uint64_t address, const AgcShaderHeader* header) -> size_t {
        if (!address || !header || !header->shader_size) return 0;
        const size_t dwords = std::min<size_t>(max_shader_dwords, header->shader_size / sizeof(uint32_t));
        if (!dwords || dwords > UINT32_MAX / sizeof(uint32_t) ||
            !guest_readable(address, static_cast<uint32_t>(dwords * sizeof(uint32_t))))
            return 0;
        return dwords;
    };
    const size_t vertex_dwords = bounded_shader_dwords(rs.es_addr, vertex_header);
    const VertexPrologInfo vertex_prolog = rdna2_vertex_prolog_info(
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(rs.es_addr)), vertex_dwords);
    const size_t chain_dwords = vertex_prolog.valid
        ? bounded_shader_dwords(chain_addr, chain_header) : 0;
    const bool vertex_chain = vertex_prolog.valid && chain_dwords != 0;
    // GFX10 encodes graphics LDS in 512-byte units.  The vertex recompiler's portable NGG model
    // represents one live guest lane and therefore needs the exact allocation as per-invocation
    // Function memory; an absent/zero register deliberately leaves graphics DS fail-closed.
    uint32_t vertex_lds_dwords = 0;
    const auto vertex_lds_state = ds.sh.find(prosper::agc::Pm4::SPI_SHADER_PGM_RSRC2_GS);
    if (vertex_lds_state != ds.sh.end()) {
        const uint32_t encoded =
            (vertex_lds_state->second >>
             prosper::agc::Pm4::SPI_SHADER_PGM_RSRC2_GS_LDS_SIZE_SHIFT) &
            prosper::agc::Pm4::SPI_SHADER_PGM_RSRC2_GS_LDS_SIZE_MASK;
        vertex_lds_dwords = std::min(encoded * 128u, 16384u);
    }

    // A proven fetch-prolog chain is linked explicitly.  Other fused programs retain the newer
    // fail-closed AGC association and compile the registered back-half body directly.
    const uint64_t fused_back_addr = fused_back && fused_back->type == 6 && fused_back->code
        ? static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fused_back->code))
        : rs.es_addr;
    const uint64_t vs_program_addr = vertex_chain ? rs.es_addr : fused_back_addr;
    std::shared_ptr<ShaderResourceTable> vrt = build_stage_table(
        ds, vertex_chain ? rs.es_addr : vs_program_addr, false, vcount_hint);
    std::shared_ptr<ShaderResourceTable> chain_vrt;
    if (vertex_chain) {
        const size_t prolog_resource_count = vrt ? vrt->resources.size() : 0;
        chain_vrt = build_stage_table(ds, chain_addr, false, vcount_hint);
        vrt = merge_vertex_chain_resource_tables(vrt, chain_vrt,
                                                  static_cast<uint32_t>(vertex_prolog.prefix_dwords));
        if (getenv("PROSPER_DBG")) {
            static std::set<std::pair<uint64_t, uint64_t>> logged;
            if (logged.emplace(rs.es_addr, chain_addr).second)
                fprintf(stderr,
                        "[vertex-chain] prolog=0x%llx words=%zu setpc=%u main=0x%llx words=%zu "
                        "resources=%zu+%zu lds=%u dwords\n",
                        (unsigned long long)rs.es_addr, vertex_dwords, vertex_prolog.setpc_pc,
                        (unsigned long long)chain_addr, chain_dwords,
                        prolog_resource_count,
                        chain_vrt ? chain_vrt->resources.size() : 0,
                        vertex_lds_dwords);
            }
    }
    std::shared_ptr<ShaderResourceTable> prt = build_stage_table(ds, rs.ps_addr, true, vcount_hint);
    const auto table_done = phase_timing
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // PROSPER_RTLOG: correlate this draw's render-target address (CB_COLOR0_BASE) with the addresses of
    // the textures it SAMPLES. If a sampled texture's base equals some draw's color0_base, that surface is
    // a GPU render target (the game renders into it then samples it) -> render-to-texture (#83/#101).
    if (getenv("PROSPER_RTLOG")) {
        fprintf(stderr, "[rt] color0=0x%llx %ux%u cf=0x%x nt=%u cs=%u",
                (unsigned long long)rs.color0_base, rs.color0_width, rs.color0_height,
                rs.color0_format, rs.color0_number_type, rs.color0_comp_swap);
        if (prt) for (const auto& r : prt->resources)
            if (r.cls == ResourceClass::Texture)
                fprintf(stderr, " tex=0x%llx(%ux%u f%u c%u)", (unsigned long long)r.gpu_addr, r.width, r.height,
                        (unsigned)r.format, r.num_components);
        if (vrt) for (const auto& r : vrt->resources)
            if (r.cls == ResourceClass::Texture)
                fprintf(stderr, " vtex=0x%llx(%ux%u f%u)", (unsigned long long)r.gpu_addr, r.width, r.height, (unsigned)r.format);
        fprintf(stderr, "\n");
    }
    PixelInputMapping pixel_inputs;
    pixel_inputs.controls = rs.ps_input_cntl;
    pixel_inputs.valid_mask = rs.ps_input_cntl_valid_mask;
    bool interpolants_from_metadata = false;
    if (!pixel_inputs.valid_mask || pixel_inputs.ambiguous_passthrough_mask()) {
        const auto* producer = vertex_chain ? chain_header : static_cast<const AgcShaderHeader*>(
            prosper_agc_shader_header_for_code(vs_program_addr));
        const auto* pixel = static_cast<const AgcShaderHeader*>(
            prosper_agc_shader_header_for_code(rs.ps_addr));
        const AgcPixelInputControls derived = derive_agc_pixel_input_controls(producer, pixel);
        PixelInputMapping metadata_inputs;
        metadata_inputs.controls = derived.controls;
        metadata_inputs.valid_mask = derived.valid_mask;
        metadata_inputs.passthrough_mask = derived.passthrough_mask;
        pixel_inputs = resolve_pixel_input_mapping(
            pixel_inputs, metadata_inputs, &interpolants_from_metadata);
    }
    if (getenv("PROSPER_INTERPLOG")) {
        fprintf(stderr, "[interp] es=0x%llx ps=0x%llx source=%s valid=%08x ena=%08x addr=%08x",
                (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr,
                interpolants_from_metadata ? "metadata" : "registers",
                pixel_inputs.valid_mask, rs.ps_input_ena, rs.ps_input_addr);
        for (uint32_t i = 0; i < pixel_inputs.controls.size(); ++i)
            if (pixel_inputs.valid_mask & (1u << i))
                fprintf(stderr, " i%u=%08x", i, pixel_inputs.controls[i]);
        fprintf(stderr, "\n");
    }
    const PixelInputMapping* pixel_input_ptr = pixel_inputs.valid_mask ? &pixel_inputs : nullptr;
    PixelSystemInputMapping system_inputs{rs.ps_input_ena, rs.ps_input_addr};
    const PixelSystemInputMapping* system_input_ptr =
        (system_inputs.ena || system_inputs.addr) ? &system_inputs : nullptr;
    const FragmentInterpolationLayout interpolation = fragment_interpolation_layout_cached(
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(rs.ps_addr)),
        max_shader_dwords, system_input_ptr, pixel_input_ptr);
    const bool capture_vertex_position = getenv("PROSPER_GEOM_PROBE") != nullptr &&
                                         !interpolation.requires_geometry;
    uint64_t vs_identity = 0, fs_identity = 0;
    SharedShaderWords vs_shared, fs_shared;
    std::vector<uint32_t> vs, fs;
    if (retain_shared_shader_words) {
        if (vertex_chain)
            vs_shared = recompile_vertex_chain_cached_shared(
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(rs.es_addr)), vertex_dwords,
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(chain_addr)), chain_dwords,
                vrt.get(), pixel_input_ptr, &vs_identity, vertex_lds_dwords,
                capture_vertex_position);
        else
            vs_shared = recompile_graphics_shader_cached_shared(
                ShaderProgramStage::Vertex, (const uint32_t*)(uintptr_t)vs_program_addr,
                max_shader_dwords, vrt.get(), pixel_input_ptr, nullptr, &vs_identity,
                false, vertex_lds_dwords, capture_vertex_position);
        fs_shared = recompile_graphics_shader_cached_shared(
            ShaderProgramStage::Fragment, (const uint32_t*)(uintptr_t)rs.ps_addr,
            max_shader_dwords, prt.get(), pixel_input_ptr, system_input_ptr, &fs_identity,
            rs.ps_wave32);
    } else {
        if (vertex_chain) {
            const SharedShaderWords linked = recompile_vertex_chain_cached_shared(
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(rs.es_addr)), vertex_dwords,
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(chain_addr)), chain_dwords,
                vrt.get(), pixel_input_ptr, &vs_identity, vertex_lds_dwords,
                capture_vertex_position);
            if (linked) vs = *linked;
        } else {
            vs = recompile_graphics_shader_cached(
                ShaderProgramStage::Vertex, (const uint32_t*)(uintptr_t)vs_program_addr,
                max_shader_dwords, vrt.get(), pixel_input_ptr, nullptr, &vs_identity,
                false, vertex_lds_dwords, capture_vertex_position);
        }
        fs = recompile_graphics_shader_cached(
            ShaderProgramStage::Fragment, (const uint32_t*)(uintptr_t)rs.ps_addr,
            max_shader_dwords, prt.get(), pixel_input_ptr, system_input_ptr, &fs_identity,
            rs.ps_wave32);
    }
    // CB_COLOR_CONTROL.DCC_DECOMPRESS interprets the bound AGC metadata helper, rather than its
    // ordinary fragment-color export. The operation bits can remain folded into a later graphics
    // submit even though the guest's utility sequence has restored normal mode by then, so MODE alone
    // is not a sufficient discriminator: doing that replaced Astro's post-process and scanout shaders.
    // Match the descriptor-free clear-RG helper program emitted by AGC as well. This is program content,
    // not a title-specific address, and keeps normal shaders fail-visible when stale operation bits leak.
    static constexpr uint32_t kDccDecompressHelperProgram[] = {
        0x7e000280u, 0xf8001803u, 0x00000000u, 0xbf810000u,
    };
    const auto* raw_fragment = reinterpret_cast<const uint32_t*>(
        static_cast<uintptr_t>(rs.ps_addr));
    const bool dcc_helper_program = raw_fragment && max_shader_dwords >=
        std::size(kDccDecompressHelperProgram) &&
        std::equal(std::begin(kDccDecompressHelperProgram),
                   std::end(kDccDecompressHelperProgram), raw_fragment);
    const bool dcc_decompress = dcc_helper_program &&
        PM4_FIELD(rs.cb_color_control, CB_COLOR_CONTROL, MODE) ==
            prosper::agc::Pm4::CB_COLOR_CONTROL_MODE_DCC_DECOMPRESS;
    if (dcc_decompress) {
        static constexpr uint32_t kNullExportProgram[] = {
            0xf8001890u, 0x00000000u, 0xbf810000u,
        };
        static const std::vector<uint32_t> kNullExportFragment =
            recompile_fragment(kNullExportProgram, std::size(kNullExportProgram));
        fs_shared.reset();
        fs = kNullExportFragment;
        fs_identity = 0;
    }
    const std::vector<uint32_t>& vs_words = vs_shared ? *vs_shared : vs;
    const std::vector<uint32_t>& fs_words = fs_shared ? *fs_shared : fs;
    std::vector<uint32_t> gs;
    if (interpolation.requires_geometry && interpolation.valid) {
        // Geometry `Triangles` accepts list, strip, and fan input assembly. Points/lines cannot
        // provide the three AMD vertex parameters and remain fail-visible.
        const bool triangle_topology = resolved_pipeline.topology >= 3u &&
                                       resolved_pipeline.topology <= 5u;
        if (triangle_topology)
            gs = recompile_interpolation_geometry(
                interpolation, getenv("PROSPER_GEOM_PROBE") != nullptr);
    }
    if (phase_timing) {
        const auto shader_done = std::chrono::steady_clock::now();
        record_draw_realization_phases(
            std::chrono::duration<double, std::milli>(table_done - table_start).count(),
            std::chrono::duration<double, std::milli>(shader_done - table_done).count());
    }
    add_stage_diagnostic(ShaderProgramStage::Vertex,
                         vertex_chain ? rs.es_addr : vs_program_addr, vrt, vs_words);
    // Retain the separately allocated main program in failed-operation captures. Without this second
    // address a replay can inspect only the fetch prolog and cannot reproduce/debug the linked stage.
    if (vertex_chain)
        add_stage_diagnostic(ShaderProgramStage::Vertex, chain_addr, vrt, vs_words);
    add_stage_diagnostic(ShaderProgramStage::Fragment, rs.ps_addr, prt, fs_words);
    if (const char* dd = getenv("PROSPER_VS_DUMP")) {   // diag: dump successful VS SPIR-V + raw RDNA2 for inspection
        static int nd = 0;
        if (nd < 3 && !vs_words.empty()) {
            char fn[512];
            snprintf(fn, sizeof fn, "%s/vs_%d_%llx.spv", dd, nd, (unsigned long long)vs_program_addr);
            if (FILE* f = fopen(fn, "wb")) { fwrite(vs_words.data(), 4, vs_words.size(), f); fclose(f); }
            snprintf(fn, sizeof fn, "%s/vs_%d_%llx.bin", dd, nd, (unsigned long long)vs_program_addr);
            if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)vs_program_addr, 1, 4096, f); fclose(f); }
            // Also dump the paired PS raw RDNA2 (the recompile-guard fixture needs both stages, #228).
            snprintf(fn, sizeof fn, "%s/ps_%d_%llx.bin", dd, nd, (unsigned long long)rs.ps_addr);
            if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)rs.ps_addr, 1, 4096, f); fclose(f); }
            nd++;
        }
    }
    if (vs_words.empty() || fs_words.empty() || (interpolation.requires_geometry && gs.empty())) {
        if (failure) failure->reason = RealizationFailureReason::ShaderRecompile;
        // PROSPER_DYNTRACE_FAIL=1: replay the FAILED vertex stage's resource build with the
        // dynamic-fetch walk trace + user-data block dump forced on (once per distinct VS), so the
        // failing draw's exact seeding/s_load chain is captured without knowing its address up front.
        if (vs_words.empty() && dyntrace_failed_shader_enabled(vs_program_addr)) {
            static std::set<uint64_t> traced;
            if (traced.insert(vs_program_addr).second) {
                fprintf(stderr, "[dynfail] replaying VS 0x%llx resource build with trace:\n",
                        (unsigned long long)vs_program_addr);
                g_dyntrace_force = true;
                (void)build_stage_table(ds, vs_program_addr, false, vcount_hint);
                g_dyntrace_force = false;
            }
        }
        // Same replay for a FAILED pixel stage (#273 — the PS-side descriptor-resolution walls).
        if (fs_words.empty() && dyntrace_failed_shader_enabled(rs.ps_addr)) {
            static std::set<uint64_t> traced_ps;
            if (traced_ps.insert(rs.ps_addr).second) {
                fprintf(stderr, "[dynfail] replaying PS 0x%llx resource build with trace:\n",
                        (unsigned long long)rs.ps_addr);
                g_dyntrace_force = true;
                (void)build_stage_table(ds, rs.ps_addr, true, vcount_hint);
                g_dyntrace_force = false;
            }
        }
        uint64_t reject_occurrence = 0;
        if (getenv("PROSPER_DBG") &&
            should_log_recompile_reject(rs.es_addr, rs.ps_addr,
                                        vs_words.size(), gs.size(), fs_words.size(),
                                        &reject_occurrence))
            fprintf(stderr, "[exec-recompile-reject] es=0x%llx gs-pgm=0x%llx hs-pgm=0x%llx ps=0x%llx vs=%zu gs=%zu fs=%zu "
                            "prim=%u topo=%u ena=%08x addr=%08x order=%llu occurrence=%llu\n",
                    (unsigned long long)rs.es_addr, (unsigned long long)rs.gs_addr,
                    (unsigned long long)rs.hs_addr, (unsigned long long)rs.ps_addr,
                    vs_words.size(), gs.size(), fs_words.size(), rs.prim_type, resolved_pipeline.topology,
                    rs.ps_input_ena, rs.ps_input_addr,
                    (unsigned long long)(draw ? draw->command_order : 0),
                    (unsigned long long)reject_occurrence);
        if (const char* dd = getenv("PROSPER_SHADER_DUMP")) {
            for (auto [tag, addr] : {std::pair{"vs", vs_program_addr}, std::pair{"ps", rs.ps_addr}}) {
                if (!addr || !guest_readable(addr, sizeof(uint32_t))) continue;
                const size_t dump_dwords = rdna2_recompile_code_span(
                    reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(addr)),
                    max_shader_dwords);
                const size_t dump_bytes = dump_dwords * sizeof(uint32_t);
                if (!dump_bytes || !guest_readable(addr, static_cast<uint32_t>(dump_bytes))) continue;
                char fn[512];
                snprintf(fn, sizeof fn, "%s/exec_%s_%llx.bin", dd, tag,
                         (unsigned long long)addr);
                if (FILE* f = fopen(fn, "wb")) {
                    fwrite((const void*)(uintptr_t)addr, 1, dump_bytes, f);
                    fclose(f);
                }
            }
        }
        if (log) {
            fprintf(stderr, "[exec] skip draw: recompile failed (vs=%zu gs=%zu fs=%zu; order=%llu "
                            "es=0x%llx ps=0x%llx color0=0x%llx/%ux%u "
                            "depth=%d/%d/op%u clear=%d/%g base=0x%llx/0x%llx "
                            "stencil=%d clear=%d/%u base=0x%llx/0x%llx)\n",
                    vs_words.size(), gs.size(), fs_words.size(),
                    (unsigned long long)(draw ? draw->command_order : 0),
                    (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr,
                    (unsigned long long)rs.color0_base, rs.color0_width, rs.color0_height,
                    (int)rs.z_enable, (int)rs.z_write_enable, rs.zfunc,
                    (int)rs.depth_clear_enable, rs.depth_clear_value,
                    (unsigned long long)rs.depth_read_base,
                    (unsigned long long)rs.depth_write_base,
                    (int)rs.stencil_enable, (int)rs.stencil_clear_enable, rs.stencil_clear_value,
                    (unsigned long long)rs.stencil_read_base,
                    (unsigned long long)rs.stencil_write_base);
            for (auto [tag, addr] : {std::pair{"vs", vs_program_addr}, std::pair{"ps", rs.ps_addr}}) {
                RecompileCoverage c = recompile_coverage((const uint32_t*)(uintptr_t)addr, max_shader_dwords);
                fprintf(stderr, "[exec]   %s coverage: total=%u alu=%u exp=%u tabledep=%u unsupported=%u "
                                "first_bad fmt=%d op=0x%x\n", tag, c.total, c.alu, c.exports,
                        c.table_dependent, c.unsupported, c.first_bad_fmt, c.first_bad_op);
            }
        }
        return false;
    }
    // One read per submit when the caller hoisted it, one per call otherwise -- never one per draw
    // per stage on a parallel worker (#2287). The read stays LIVE either way, deliberately not
    // PROSPER_ENV_VALUE: test_shader_resources.cpp and test_gpu_capture_render.cpp arm this variable
    // at runtime, and a process-lifetime cache would make those arms vacuous rather than failing --
    // the #2214 defect that check_cached_env.py gates.
    const char* const validate_mode = hoisted_validate_mode ? *hoisted_validate_mode
                                                            : getenv("PROSPER_DESCRIPTOR_VALIDATE");
    if (!validate_runtime_descriptor_contract("VS", vs_words, vrt.get(), 0, SpirvShaderStage::Vertex,
                                              validate_mode) ||
        !validate_runtime_descriptor_contract("PS", fs_words, prt.get(), 1, SpirvShaderStage::Fragment,
                                              validate_mode)) {
        if (failure) failure->reason = RealizationFailureReason::DescriptorContract;
        if (log) fprintf(stderr, "[exec] skip draw: strict descriptor contract failed "
                                "(es=0x%llx ps=0x%llx color0=0x%llx)\n",
                         (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr,
                         (unsigned long long)rs.color0_base);
        return false;
    }
    ResolvedPipelineState ps = resolved_pipeline;
    // EXP.EN is the final per-component gate after CB_TARGET_MASK and CB_SHADER_MASK. Vulkan exposes
    // the same preservation semantics through colorWriteMask: disabled attachment components retain
    // their old values even though the fragment output itself is a full vec4.
    const uint32_t exp_mask = fragment_color_export_mask(
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(rs.ps_addr)), max_shader_dwords);
    for (uint32_t slot = 0; slot < ps.color_targets.size(); ++slot)
        ps.color_targets[slot].write_mask &= (exp_mask >> (slot * 4u)) & 0xFu;
    ps.color_write_mask = ps.color_targets[0].write_mask;
    ps.color1_write_mask = ps.color_targets[1].write_mask;
    // Color-disabled draws are not necessarily no-ops. Depth prepasses and stencil mask writers
    // deliberately set CB_TARGET_MASK=0, then later color draws consume their DS result. Dropping
    // those writers made The Messenger clear stencil to 0 and then test for bits 1/2 that could never
    // be produced (#520). Skip only when the draw has no observable color OR depth/stencil effect.
    const bool ds_effect = has_depth_stencil_side_effect(ps);
    // A non-zero hardware write mask is itself an observable color effect. Synthetic callers and
    // legacy captures may omit CB_COLOR_INFO and rely on the backend's established RGBA8 fallback;
    // requiring a decoded format here would incorrectly discard those otherwise-valid draws.
    const bool color_effect = std::any_of(
        ps.color_targets.begin(), ps.color_targets.end(),
        [](const auto& target) { return target.write_mask != 0; });
    if (!color_effect && !ds_effect && !getenv("PROSPER_FORCE_COLORWRITE")) {
        if (failure) failure->reason = RealizationFailureReason::NoEffect;
        if (log) fprintf(stderr, "[exec] skip draw: no color/depth/stencil effect cb_target_mask=0x%x cb_color_control=0x%x color0_fmt=%u\n",
                         rs.cb_target_mask, rs.cb_color_control, ps.color0_format);
        return false;
    }
    // PROSPER_FORCE_COLORWRITE: diagnostic — render color_write_mask==0 draws anyway (force mask to RGBA).
    // The cutscene submits ~66 draws/frame that resolve to mask==0; if that is a mis-decode (not a genuine
    // depth-only pass), rendering them reveals whether they are the cutscene content.
    if (ps.color_write_mask == 0 && getenv("PROSPER_FORCE_COLORWRITE")) ps.color_write_mask = 0xf;
    // PROSPER_DRAWDIAG: per-RENDERED-draw geometry/position/texture — to LOCATE specific draws (e.g. the
    // cutscene caption text: small indexed quads, bottom viewport, blended, sampling a font atlas).
    if (getenv("PROSPER_DRAWDIAG")) {
        uint32_t ic = (draw && draw->indexed) ? draw->index_count : vcount_hint;
        fprintf(stderr, "[draw] idx=%u vp=%d y=%.0f h=%.0f blend=%d cwm=0x%x "
                        "target=0x%x shader=0x%x exp=0x%x es=0x%llx ps=0x%llx", ic,
                ps.has_viewport, ps.viewport_y, ps.viewport_h, ps.blend_enable, ps.color_write_mask,
                rs.cb_target_mask, rs.cb_shader_mask, exp_mask,
                (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr);
        if (prt) for (const auto& r : prt->resources)
            if (r.cls == ResourceClass::Texture)
                fprintf(stderr, " tex=0x%llx(%ux%u f%u)", (unsigned long long)r.gpu_addr, r.width, r.height, (unsigned)r.format);
        fprintf(stderr, "\n");
    }
    uint32_t vertex_count = vcount_hint ? vcount_hint : 3u;
    // The bound vertex buffer's record count (size/stride) — bounds an indexed draw's vertex range, and
    // for a NON-indexed draw is often the truer count: a draw record's index_count can be a low/stale
    // value for these NGG draws (4 of ~20 verts -> a degenerate sliver), while the VB's record count is
    // the whole mesh. A shader fetching past a real vertex reads 0 under robustBufferAccess -> a
    // degenerate, clipped vertex, so a slightly-generous count is harmless.
    uint32_t vb_entries = 0;
    if (vrt) for (const auto& r : vrt->resources)
        if (r.cls == ResourceClass::VertexBuffer && r.stride)
            vb_entries = std::max(vb_entries, r.size / r.stride);
    if (vb_entries > 65536u) vb_entries = 65536u;   // sanity cap: don't stall llvmpipe on an over-sized VB
    // Indexed draw (sceAgcDcbDrawIndex): fetch the real index data from guest memory (1:1-mapped) and
    // hand it to the backend, which renders it with vkCmdDrawIndexed. This replaced the old "4-record
    // VB -> TRIANGLE_FAN" heuristic (issue #64): the sprite quads that heuristic guessed at are in fact
    // DrawIndex packets with a 6-entry index list ([0,1,2, 2,3,0]-style two-triangle quads), so the
    // real indices + the real decoded VGT_PRIMITIVE_TYPE topology render every 4-vertex mesh correctly,
    // whatever its vertex order. Unknown element size or an unreadable buffer falls back (loudly) to a
    // non-indexed draw of the hint count instead of reading garbage.
    static constexpr uint32_t kMaxIndices = 1u << 20;   // sanity cap (largest seen live: 0x61e)
    if (draw && draw->indexed && draw->index_addr && draw->index_count) {
        uint32_t esz = index_elem_bytes(ds.index_type);
        uint32_t n = std::min(draw->index_count, kMaxIndices);
        uint64_t index_addr = draw->index_addr;
        // Auto-detect a 32-bit index buffer that the guest never announced (#304). DOLL's UE4 Slate/UMG
        // quads use 32-bit index buffers but the title never calls sceAgcDcbSetIndexSize and sets no
        // VGT_INDEX_TYPE register, so index_type defaults to 16-bit — misreading each 32-bit index as
        // two 16-bit ones. The fingerprint is unmistakable: a 32-bit index buffer read as 16-bit has
        // every ODD 16-bit word (the zero high half of a small index) == 0, while reading it as 32-bit
        // yields small, valid indices. Genuine 16-bit buffers (DOLL's scene/text meshes, all of the
        // Messenger's quads) have non-zero odd words and are left untouched. When detected, use the
        // 32-bit element size AND recompute a DrawIndexOffset's address at that stride (index_base +
        // index_offset*4) so it lands on the correct quad. CONFIDENCE: HIGH — the banner index buffer
        // decodes to a clean [0,1,2,2,1,3] quad this way vs a degenerate [0,0,1,0,2,0] as 16-bit.
        if (esz == 2 && n >= 2) {
            uint64_t addr32 = draw->from_offset ? (draw->index_base + (uint64_t)draw->index_offset * 4u)
                                                : draw->index_addr;
            if (guest_readable(draw->index_addr, n * 2u) && guest_readable(addr32, n * 4u) &&
                index_buffer_is_unannounced_32bit((const uint16_t*)(uintptr_t)draw->index_addr,
                                                  (const uint32_t*)(uintptr_t)addr32, n)) {
                esz = 4; index_addr = addr32;
                if (log) fprintf(stderr, "[exec] indexed draw: auto-detected 32-bit index buffer "
                                 "(unannounced) at 0x%llx (was 16-bit 0x%llx)\n",
                                 (unsigned long long)addr32, (unsigned long long)draw->index_addr);
            }
        }
        if (esz == 0) {
            if (log) fprintf(stderr, "[exec] indexed draw: UNKNOWN index_type=%u — falling back to non-indexed\n",
                             ds.index_type);
        } else if (!guest_readable(index_addr, n * esz)) {
            if (log) fprintf(stderr, "[exec] indexed draw: index buffer 0x%llx (%u x %uB) unreadable — "
                             "falling back to non-indexed\n",
                             (unsigned long long)index_addr, n, esz);
        } else {
            out.indices.resize(n);
            uint32_t max_index = 0;
            if (esz == 2) {
                const uint16_t* src = (const uint16_t*)(uintptr_t)index_addr;
                for (uint32_t i = 0; i < n; i++) { out.indices[i] = src[i]; max_index = std::max(max_index, out.indices[i]); }
            } else {
                const uint32_t* src = (const uint32_t*)(uintptr_t)index_addr;
                for (uint32_t i = 0; i < n; i++) { out.indices[i] = src[i]; max_index = std::max(max_index, out.indices[i]); }
            }
            // Vertex count = the indexed range (max index + 1). The INDEX BUFFER is authoritative for how
            // many vertices the draw touches — do NOT clamp down to the bound VB's record count: the
            // bindless per-glyph fetch resolves a tiny per-glyph V# (num_records=4 = one glyph), so clamping
            // to it would drop every glyph but the first (#257). The VB is grown below to span this range;
            // a truly-unmapped tail still degrades safely (safe_copy stops at the mapping edge -> zero).
            vertex_count = max_index + 1;
            // Sanity-cap the VALUE-derived vertex range (#461). kMaxIndices already caps the index COUNT,
            // but a single garbage/torn 32-bit index VALUE — an announced 32-bit index buffer skips the
            // <0x10000 fingerprint, and the index buffer is read from guest memory another thread may be
            // freeing/rewriting — would inflate vertex_count to hundreds of millions and force a multi-GB
            // VB upload (OOM / llvmpipe stall / crash on the submit thread). A real single-draw mesh is far
            // below this ceiling; anything above it is garbage, so clamp (this only shrinks a garbage draw,
            // never a legitimate one; the VB-grow below is 64-bit-safe + capped regardless).
            if (vertex_count > kMaxIndices) {
                if (log) fprintf(stderr, "[exec] indexed draw: max_index %u exceeds sanity cap %u — clamped "
                                 "(garbage/torn indices?)\n", max_index, kMaxIndices);
                vertex_count = kMaxIndices;
            }
        }
    }
    // A genuinely empty draw (vcount_hint == 0 — engines emit 0-vertex DrawIndexAuto/DrawIndexOffset as
    // no-ops) that resolved NO indices above must render nothing, exactly as it does on hardware. Do NOT
    // fall through: `vertex_count = vcount_hint ? vcount_hint : 3` already fabricated 3, and the vb_records
    // fallback in resolve_nonindexed_vertex_count() below would then sweep the ENTIRE residual vertex pool
    // (0..vb_records-1) of whatever geometry the last-bound VB still holds — turning a no-op into a phantom
    // triangle or a full-VB draw of stale geometry composited into the frame (#400). Skipping here keeps
    // that fallback a pure zero-count guard; a non-zero DrawIndexAuto count is authoritative (see #1163).
    if (vcount_hint == 0 && out.indices.empty()) {
        if (failure) failure->reason = RealizationFailureReason::ZeroVertices;
        if (log) fprintf(stderr, "[exec] skip draw: zero vertex count (no-op draw)\n");
        return false;
    }
    if (out.indices.empty()) vertex_count = resolve_nonindexed_vertex_count(vcount_hint, vb_entries);
    // PS5 RectList (primitive 7; standard AMD RectList is 17) consumes three procedural vertices but
    // covers the rectangle's synthesized fourth corner. Vulkan has no rectangle-list topology. The
    // Blasphemous 2 clear shader explicitly computes all four clip-space corners from VertexIndex, has
    // no vertex-buffer inputs, and submits count=3; invoke index 3 and render the four results as a
    // triangle strip. Restrict the expansion to that observed no-VB form: a general VB-backed RectList
    // needs post-VS fourth-vertex synthesis and must not speculatively fetch a fourth input record.
    const bool rect_list = rs.prim_type == 7u || rs.prim_type == 17u;
    if (rect_list && out.indices.empty() && vertex_count == 3u && vb_entries == 0u) {
        vertex_count = 4u;
        if (log) fprintf(stderr, "[exec] RectList: expanded procedural 3-vertex rectangle to 4-vertex strip\n");
    }
    // Bindless per-glyph vertex fetch (#257): the fetch-shader patches a SMALL per-glyph V# (num_records=4
    // = one glyph's 4 corners, size=304). But the draw indexes ALL vertices (gl_VertexIndex 0..N-1) out of
    // the CONTIGUOUS vertex pool that begins at that base — so uploading only num_records*stride bytes
    // leaves every vertex past the first glyph reading out-of-bounds (robustBufferAccess 0), collapsing the
    // whole caption to the first glyph. Grow each vertex buffer to cover the draw's full vertex range so
    // gl_VertexIndex reads the real per-vertex data. (Correct in general: a VB must span the drawn range.)
    if (vrt) for (auto& r : vrt->resources)
        if (r.cls == ResourceClass::VertexBuffer && r.stride) {
            // 64-bit to avoid a uint32 overflow (vertex_count up to 1M x a 14-bit stride overflows 32-bit),
            // and cap the grown size to the same 256 MB plausibility ceiling the V# decode uses — a VB this
            // large is never real, and an unbounded r.size drives a multi-GB upload (#461). Vertices past
            // the real backing degrade safely (safe_copy stops at the mapping edge -> robust-0).
            // A positive GE_INDX_OFFSET selects a later range in the same shared vertex pool. The
            // capture/backend buffer must include that prefix because gl_VertexIndex includes the
            // Vulkan firstVertex/vertexOffset value before the translated shader fetches the V#.
            uint64_t addressed_vertices = vertex_count;
            const int32_t vertex_offset = static_cast<int32_t>(rs.ge_indx_offset);
            if (vertex_offset > 0) addressed_vertices += static_cast<uint32_t>(vertex_offset);
            uint64_t need = addressed_vertices * r.stride;
            if (need > 0x10000000ull) need = 0x10000000ull;   // 256 MB cap
            if ((uint64_t)r.size < need) r.size = (uint32_t)need;
        }
    // PROSPER_CAPTION_DIAG: for the caption text mesh (a draw whose PS samples the 2048x1024 R8 font
    // atlas), dump everything needed to see WHY its geometry collapses — vertex count, the bound vertex
    // buffers (base/stride/size), the index range, and the VS bytecode (for offline llvm-mc disasm of the
    // position export). Once per distinct VS. #102 follow-up (font atlas decodes; glyphs land nowhere).
    // PROSPER_ONLY_ATLAS: render ONLY the caption text draw (samples the 2048x1024 font atlas), skipping
    // every other draw — so the caption geometry renders alone onto a clear frame. With TESTPS this shows
    // whether the caption rasterizes at all (magenta glyphs) or is culled/degenerate/clipped. #257.
    if (getenv("PROSPER_ONLY_ATLAS")) {
        bool sa = false;
        if (prt) for (const auto& r : prt->resources)
            if (r.cls == ResourceClass::Texture && r.width == 2048 && r.height == 1024) sa = true;
        if (!sa) {
            if (failure) failure->reason = RealizationFailureReason::Filtered;
            return false;
        }
        // PROSPER_ONLY_IC=<n>: further restrict to draws whose index count == n (isolate one atlas mesh,
        // e.g. the main glyph batch idx=1566 vs a fullscreen atlas draw).
        if (const char* ic_s = getenv("PROSPER_ONLY_IC")) {
            uint32_t want = (uint32_t)atoi(ic_s);
            uint32_t ic = (draw && draw->indexed) ? draw->index_count : vcount_hint;
            if (ic != want) {
                if (failure) failure->reason = RealizationFailureReason::Filtered;
                return false;
            }
        }
    }
    if (getenv("PROSPER_CAPTION_DIAG") && prt) {
        bool samples_atlas = false;
        for (const auto& r : prt->resources)
            if (r.cls == ResourceClass::Texture && r.width == 2048 && r.height == 1024) samples_atlas = true;
        static std::set<uint64_t> seen_vs;
        if (samples_atlas && seen_vs.insert(rs.es_addr).second) {
            uint32_t maxi = 0; for (uint32_t v : out.indices) maxi = std::max(maxi, v);
            fprintf(stderr, "[caption] es=0x%llx vertex_count=%u indices=%zu max_index=%u vb_entries=%u\n",
                    (unsigned long long)rs.es_addr, vertex_count, out.indices.size(), maxi, vb_entries);
            if (vrt) for (const auto& r : vrt->resources) {
                fprintf(stderr, "[caption]   %s binding=%u base=0x%llx stride=%u size=%u fmt=%u nc=%u\n",
                        r.cls == ResourceClass::VertexBuffer ? "VB" :
                        r.cls == ResourceClass::ConstantBuffer ? "CB" : "TEX",
                        r.binding, (unsigned long long)r.gpu_addr, r.stride, r.size,
                        (unsigned)r.format, r.num_components);
                // Raw first 3 records as floats — reveals whether the fetched position/attr data is valid or
                // NaN/degenerate (a stride-0 buffer reads record 0 for every vertex -> collapse). #257.
                if (r.cls == ResourceClass::VertexBuffer && r.gpu_addr > 0x10000) {
                    uint32_t st = r.stride ? r.stride : 16;
                    for (int rec = 0; rec < 3; rec++) {
                        const float* f = (const float*)(uintptr_t)(r.gpu_addr + (uint64_t)rec * st);
                        const uint32_t* u = (const uint32_t*)(uintptr_t)(r.gpu_addr + (uint64_t)rec * st);
                        fprintf(stderr, "[caption]       rec%d: %.3f %.3f %.3f %.3f  (raw %08x %08x %08x %08x)\n",
                                rec, f[0], f[1], f[2], f[3], u[0], u[1], u[2], u[3]);
                    }
                }
                // Constant buffers = the transform matrices / uniforms. Dump as vec4 rows (the collapse may
                // be a mis-resolved MVP -> valid glyph coords transform off-screen / to w<=0). #257.
                if (r.cls == ResourceClass::ConstantBuffer && r.gpu_addr > 0x10000) {
                    uint32_t nvec = r.size / 16; if (nvec > 24) nvec = 24;
                    for (uint32_t v = 0; v < nvec; v++) {
                        const float* f = (const float*)(uintptr_t)(r.gpu_addr + (uint64_t)v * 16);
                        fprintf(stderr, "[caption]       cb[%u]: %.4f %.4f %.4f %.4f\n", v, f[0], f[1], f[2], f[3]);
                    }
                }
            }
            fflush(stderr);
            if (const char* dd = getenv("PROSPER_FRAME_DIR")) {
                char fn[512]; snprintf(fn, sizeof fn, "%s/caption_vs_%llx.bin", dd, (unsigned long long)rs.es_addr);
                if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)rs.es_addr, 1, 8192, f); fclose(f); }
                // The RECOMPILED VS SPIR-V — disassemble offline (spirv-dis) to trace the gl_Position export
                // op-by-op against the RDNA2 source and find the mis-modeled op / bad matrix input. #257.
                snprintf(fn, sizeof fn, "%s/caption_recompiled_%llx.spv", dd, (unsigned long long)rs.es_addr);
                if (FILE* f = fopen(fn, "wb")) { fwrite(vs_words.data(), 4, vs_words.size(), f); fclose(f); }
                // The recompiled PIXEL shader — the text is invisible because the PS discards every fragment
                // (SDF alpha-test). Dump it (spirv-dis) to inspect the discard condition + atlas sample. #257.
                snprintf(fn, sizeof fn, "%s/caption_ps_%llx.spv", dd, (unsigned long long)rs.ps_addr);
                if (FILE* f = fopen(fn, "wb")) { fwrite(fs_words.data(), 4, fs_words.size(), f); fclose(f); }
                snprintf(fn, sizeof fn, "%s/caption_ps_raw_%llx.bin", dd, (unsigned long long)rs.ps_addr);
                if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)rs.ps_addr, 1, 8192, f); fclose(f); }
            }
        }
    }
    out.vs = std::move(vs); out.gs = std::move(gs); out.fs = std::move(fs);
    out.vs_shared = std::move(vs_shared); out.fs_shared = std::move(fs_shared);
    out.vs_guest_addr = vertex_chain ? rs.es_addr : vs_program_addr;
    out.fs_guest_addr = rs.ps_addr;
    out.vs_chain_guest_addr = vertex_chain ? chain_addr : 0;
    out.vertex_lds_dwords = vertex_lds_dwords;
    out.has_pixel_inputs = pixel_input_ptr != nullptr;
    if (pixel_input_ptr) out.pixel_inputs = *pixel_input_ptr;
    out.has_system_inputs = system_input_ptr != nullptr;
    if (system_input_ptr) out.system_inputs = *system_input_ptr;
    out.vs_identity = vs_identity; out.fs_identity = fs_identity; out.ps = ps;
    out.vrt = std::move(vrt); out.prt = std::move(prt); out.vertex_count = vertex_count;
    // #1256: record the raw draw-packet state (pre-realization) so a capture can be checked offline for
    // realization divergence. vcount_hint is the DrawIndexAuto/DrawIndex index_count decoded from the guest.
    out.raw_draw_count = vcount_hint; out.raw_indexed = (draw && draw->indexed);
    out.raw_draw_modifier = draw ? draw->modifier : 0;
    out.vertex_offset = draw && draw->has_vertex_offset_override
        ? draw->indirect_vertex_offset
        : static_cast<int32_t>(rs.ge_indx_offset);
    // The draw record is authoritative even in folded mode: register state may change after the
    // last draw, while IT_NUM_INSTANCES belongs to the draw at the moment it executes.
    out.instance_count = draw ? draw->instance_count : ds.num_instances;
    out.color0_base = rs.color0_base;   // render-to-texture: the target this draw writes into (#167)
    out.color0_width = rs.color0_width; out.color0_height = rs.color0_height; // per-target extent (#526)
    out.color1_base = rs.color1_base;
    out.color1_width = rs.color1_width; out.color1_height = rs.color1_height;
    for (uint32_t slot = 0; slot < out.color_targets.size(); ++slot) {
        out.color_targets[slot].base = rs.color_targets[slot].base;
        out.color_targets[slot].width = rs.color_targets[slot].width;
        out.color_targets[slot].height = rs.color_targets[slot].height;
    }
    // Preserve direct/synthetic callers that still populate only the named aliases.
    out.color_targets[0] = {out.color0_base, out.color0_width, out.color0_height};
    out.color_targets[1] = {out.color1_base, out.color1_width, out.color1_height};
    return true;
}

// Live submits with hundreds of immutable draw snapshots can realize those snapshots concurrently.
// The implementation owns a bounded persistent worker set, writes one result slot per semantic draw,
// and compacts those slots in original PM4 order. `attempted` distinguishes an ineligible batch from
// a batch in which every draw was legitimately filtered out, so callers only take the serial fallback
// in the former case.
std::vector<DrawItem> realize_gpustate_draws_parallel(
    const GpuState& st, uint32_t max_shader_dwords, bool log,
    bool retain_shared_shader_words, bool* attempted = nullptr);

// Recompile + resolve a GpuState's draws and render them via `render`. Default: ONE item from the folded
// end state, realized as the submit's LAST draw — the folded register file IS the state at the last
// draw, so that is the one draw record (count + index buffer) coherent with it. (The pre-#64 code paired
// the end state with draws[0]'s count instead, and needed a "4-record VB -> TRIANGLE_FAN" heuristic to
// paper over the mismatch; the title composite is in fact the submit's last draw, a 6-index DrawIndex
// quad, which now renders through the real indexed path.) PROSPER_PERDRAW=1: ONE item per draw, each
// realized from ITS OWN register snapshot (Draw::state), so per-draw masks/blends/shaders composite
// correctly — the path for multi-geometry scenes (opt-in until the AGC context-log section semantics
// that stage duplicate register writes are fully RE'd; see docs/REAL_FRAMES_FINDINGS.md).
// `max_shader_dwords` bounds the recompiler's walk (it stops at S_ENDPGM).
// vp_scale_{x,y}: scale each draw's guest viewport by this factor. The guest programs PA_CL_VPORT in
// full present-resolution pixels; when we render into a reduced-resolution framebuffer (PROSPER_RENDER_SCALE)
// the viewport must shrink by the same ratio, or a full-res viewport into a small framebuffer clips the
// image to its bottom-left corner. Default 1.0 (full-res / tests: no change).
inline std::vector<DrawItem> realize_gpustate_draws(const GpuState& st,
                                                    uint32_t max_shader_dwords = 0x10000,
                                                    float vp_scale_x = 1.0f,
                                                    float vp_scale_y = 1.0f,
                                                    std::vector<OperationRealizationFailure>* failures = nullptr,
                                                    bool retain_shared_shader_words = false,
                                                    bool allow_parallel = true) {
    if (failures) failures->clear();
    if (st.draws.empty()) return {};
    // PROSPER_EXECLOG: just the per-draw bail-point/skip logs, without PROSPER_GFXLOG's per-packet
    // firehose (which is GBs over a minutes-long run) — for "which draws skip and why" surveys (#319).
    const bool log = getenv("PROSPER_GFXLOG") != nullptr ||
                     getenv("PROSPER_EXECLOG") != nullptr;   // bail-point visibility (why no frame?)
    // Render each draw from its OWN register snapshot when the submit has MULTIPLE draws — a real
    // multi-geometry scene (the game's in-game/cutscene submits carry 8-11 distinct draws with per-draw
    // shaders/textures/blends). Folding those to just the last draw drops the rest and the frame comes out
    // as the bare clear; per-draw rendering makes the intro cutscene's real content appear (black scene +
    // its geometry) instead of a blank blue clear. A single-draw submit stays folded (nothing to composite).
    // Overridable: PROSPER_PERDRAW forces per-draw always, PROSPER_FOLDED forces the old single-item path.
    // CONFIDENCE: MED — verified multi-draw scenes render their content per-draw vs. a blank clear folded.
    static const bool force_perdraw = getenv("PROSPER_PERDRAW") != nullptr;
    static const bool force_folded  = getenv("PROSPER_FOLDED") != nullptr;
    const bool perdraw = force_perdraw ||
                         (!force_folded && (st.draws.size() > 1 || !st.dispatches.empty()));
    std::vector<DrawItem> items;
    if (perdraw) {
        bool parallel_attempted = false;
        if (allow_parallel && !failures)
            items = realize_gpustate_draws_parallel(
                st, max_shader_dwords, log, retain_shared_shader_words,
                &parallel_attempted);
        if (!parallel_attempted) {
            for (size_t i = 0; i < st.draws.size(); i++) {
                DrawItem it;
                OperationRealizationFailure failure;
                if (realize_draw_item(st.state_at_draw(i), &st.draws[i], st.draws[i].index_count,
                                      max_shader_dwords, log, it,
                                      failures ? &failure : nullptr,
                                      retain_shared_shader_words)) {
                    it.draw_index = i;
                    it.command_order = st.draws[i].command_order;
                    items.push_back(std::move(it));
                } else if (failures) {
                    failure.index = i;
                    failure.command_order = st.draws[i].command_order;
                    failures->push_back(std::move(failure));
                }
            }
        }
    } else {
        // A forced folded capture still needs an explanation for every earlier semantic draw that
        // the execution policy omits. Diagnose those draws without adding them to the realized list.
        if (failures && st.draws.size() > 1) {
            for (size_t i = 0; i + 1 < st.draws.size(); ++i) {
                DrawItem ignored;
                OperationRealizationFailure failure;
                const bool would_realize = realize_draw_item(
                    st.state_at_draw(i), &st.draws[i], st.draws[i].index_count,
                    max_shader_dwords, log, ignored, &failure,
                    retain_shared_shader_words);
                if (would_realize) failure.reason = RealizationFailureReason::Filtered;
                failure.index = i;
                failure.command_order = st.draws[i].command_order;
                failures->push_back(std::move(failure));
            }
        }
        // Default: render the submit's last draw from the folded end state as a single item.
        DrawItem it;
        const GpuState::Draw& last = st.draws.back();
        OperationRealizationFailure failure;
        if (realize_draw_item(st, &last, last.index_count, max_shader_dwords, log, it,
                              failures ? &failure : nullptr,
                              retain_shared_shader_words)) {
            it.draw_index = st.draws.size() - 1;
            it.command_order = last.command_order;
            items.push_back(std::move(it));
        } else if (failures) {
            failure.index = st.draws.size() - 1;
            failure.command_order = last.command_order;
            failures->push_back(std::move(failure));
        }
    }
    if (getenv("PROSPER_DRAWLOG")) { fprintf(stderr, "[exec] draws=%zu perdraw=%d -> %zu item(s): raw index_counts=[",
        st.draws.size(), (int)perdraw, items.size());
        for (size_t i = 0; i < st.draws.size(); i++) fprintf(stderr, "%s%u%s", i?",":"", st.draws[i].index_count,
                                                             st.draws[i].indexed ? "i" : "");
        fprintf(stderr, "] items:");
        for (auto& it : items) fprintf(stderr, " (vcount=%u nidx=%zu topo=%u mask=0x%x)",
                                       it.vertex_count, it.indices.size(), it.ps.topology, it.ps.color_write_mask);
        fprintf(stderr, "\n"); fflush(stderr); }
    if (items.empty()) return {};
    // Scale each item's guest viewport and scissor to the actual (possibly reduced-resolution)
    // framebuffer. State that was never programmed keeps the backend's full-target defaults.
    if (vp_scale_x != 1.0f || vp_scale_y != 1.0f)
        for (auto& it : items)
            scale_resolved_render_area(it.ps, vp_scale_x, vp_scale_y);
    if (log) fprintf(stderr, "[exec] rendering %zu draw item(s) (of %zu draws)\n", items.size(), st.draws.size());
    return items;
}

inline std::vector<uint8_t> execute_gpustate(const GpuState& st, const RenderFn& render,
                                             uint32_t max_shader_dwords = 0x10000,
                                             float vp_scale_x = 1.0f, float vp_scale_y = 1.0f) {
    if (!render) return {};
    std::vector<DrawItem> items = realize_gpustate_draws(
        st, max_shader_dwords, vp_scale_x, vp_scale_y);
    if (items.empty()) return {};
    return render(items);
}

// --- Live submit renderer registry (Stage A wiring; implemented in gpu_executor.cpp) --------------------
// The live renderer additionally receives the target width/height (from videoout) so it can size its
// attachments. Registered by whoever owns a persistent Vulkan device — the runtime binary at startup, or a
// test — so prosper_core itself stays Vulkan-free (this just stores a std::function). Same DrawItem-list
// shape as RenderFn, plus (w,h).
struct RenderedFrame {
    std::shared_ptr<const std::vector<uint8_t>> storage;
    // Where these pixels came from, carried to the present layer so a consumer can tell a frame
    // prosper composited from the guest's own display buffer republished verbatim (#1968, #2044).
    // Defaults to Composited, which is what every producer other than the renderer's last-resort
    // guest-scanout branch means.
    PresentFrameOrigin origin = PresentFrameOrigin::Composited;

    RenderedFrame() = default;
    RenderedFrame(std::vector<uint8_t> pixels)
        : storage(std::make_shared<const std::vector<uint8_t>>(std::move(pixels))) {}
    explicit RenderedFrame(std::shared_ptr<const std::vector<uint8_t>> pixels)
        : storage(std::move(pixels)) {}

    bool empty() const { return !storage || storage->empty(); }
    size_t size() const { return storage ? storage->size() : 0; }
    const uint8_t* data() const { return storage ? storage->data() : nullptr; }
    const std::vector<uint8_t>& bytes() const {
        static const std::vector<uint8_t> empty;
        return storage ? *storage : empty;
    }
};

using LiveRenderFn = std::function<RenderedFrame(const std::vector<DrawItem>& items,
                                                  uint32_t width, uint32_t height)>;

struct OrderedSubmitResult {
    RenderedFrame frame;
    size_t render_spans = 0;
    bool compute_executed = false;
};
OrderedSubmitResult execute_ordered_items(const std::vector<SubmitOperation>& operations,
                                          const std::vector<DrawItem>& draws,
                                          const std::vector<ComputeItem>& computes,
                                          const std::vector<GpuState::DmaCopy>& dma_copies,
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height);
OrderedSubmitResult execute_ordered_items(const std::vector<SubmitOperation>& operations,
                                          const std::vector<DrawItem>& draws,
                                          const std::vector<ComputeItem>& computes,
                                          const std::vector<ReplayDmaCopy>& dma_copies,
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height);
// Compatibility overload for capture replay and tests whose timelines contain only draws/dispatches.
OrderedSubmitResult execute_ordered_items(const std::vector<SubmitOperation>& operations,
                                          const std::vector<DrawItem>& draws,
                                          const std::vector<ComputeItem>& computes,
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height);

// Register (or clear, with {}) the live render backend that agc_driver_submit_dcb uses on each submit.
void set_submit_renderer(LiveRenderFn fn);
bool have_submit_renderer();

struct LiveRenderPhase {
    bool first_span = true;
    bool final_span = true;
    // The next ordered operation reads render-target bytes on the CPU. Persistent Vulkan targets
    // must synchronously read back this span instead of deferring their authoritative pixels.
    bool authoritative_readback = false;

    bool allows_deferred_scanout_readback() const {
        return !final_span && !authoritative_readback;
    }
};
LiveRenderPhase live_render_phase();

// Invoke the registered live backend directly with already-realized draws. Used by the local capture
// replayer; normal guest execution enters through execute_and_present(). Returns {} when unregistered.
std::vector<uint8_t> render_submit_items(const std::vector<DrawItem>& items,
                                         uint32_t width, uint32_t height);
bool execute_compute_items(const std::vector<ComputeItem>& items);

// Render a folded GpuState at (width,height) via the registered live renderer and hand the frame to the
// present path (present_write_frame) when publish is true. Returns true iff a frame was produced and
// published. With publish=false the renderer still executes all GPU work, but scanout remains unchanged.
// A no-op
// returning false when there is no renderer registered or the state has no draws — so it is inert on the
// game path until the runtime wires a device, yet fully exercised by tests. Stage A of GPU_EXECUTOR_DESIGN.
bool execute_and_present(const GpuState& st, uint32_t width, uint32_t height,
                         bool publish = true);

// Execute retained graphics and compute work in PM4 order. Graphics spans share the frontend's
// persistent render-target cache, and only the final span is handed to the present path.
bool execute_ordered_and_present(const GpuState& st, uint32_t width, uint32_t height,
                                 uint64_t submit_no = 0, bool publish = true);

// The extent contract between the live renderer and the publish gate (#1986). Both *_and_present
// functions publish a frame only when its byte count is exactly width*height*4 and silently discard
// anything else, so inside such a submit a rendered pass of a different extent is not a present
// source at all — the renderer must not hand one back, or the frame is dropped with the guest still
// submitting and nothing saying why (Sonic Frontiers, #1968).
//
// That contract holds ONLY for a submit destined for the gate. render_submit_items — gpu_replay's
// ordered-prefix inspection (--draw / --draw-steps / --through-operation) and the renderer's own
// tests — deliberately consumes the LAST PASS AT ITS OWN EXTENT and infers the extent from the
// returned byte count afterwards, which is a different and equally deliberate contract (#1330,
// #526). A scope rather than a callback parameter because that signature is shared by every
// consumer, including replay-owned renderers; the callback is invoked synchronously on the
// submitting thread, so the state is thread-local. Nesting is counted, so an inner scope cannot
// end an outer one.
class PresentSubmitScope {
public:
    PresentSubmitScope();
    ~PresentSubmitScope();
    PresentSubmitScope(const PresentSubmitScope&) = delete;
    PresentSubmitScope& operator=(const PresentSubmitScope&) = delete;
};

// True while this thread is inside a submit whose frame will reach the publish gate.
bool present_submit_in_progress();

// The 64 KiB Global Data Share. Compute shaders reach it through the internal binding installed when
// a kernel decodes a `ds_*` instruction with the GDS flag; the command processor reaches it here,
// because DMA_DATA can name a GDS OFFSET rather than a guest address as its endpoint (Sony's own
// parameter is `dstAddressOrOffset`, and `sceAgcDcbAtomicGds` is part of the same API surface).
// Returns the backing and its size so a caller can bounds-check an offset without assuming 64 KiB.
uint8_t* compute_gds_backing();
size_t compute_gds_size();

} // namespace prosper::gpu
