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
#include <cstring>                 // memcpy: aliasing-safe index-buffer fingerprint loads
#include "rdna2_to_spirv.hpp"      // recompile_vertex / recompile_fragment
#include "shader_resources.hpp"    // ShaderResourceTable
#include "agc_shader_layout.hpp"   // DecodedBufferDescriptor (DynFetch)
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>
#include <set>

extern "C" const void* prosper_agc_shader_header_for_code(uint64_t code_addr);

namespace prosper::gpu {

struct ShaderResourceTable;   // fwd (shader_resources.hpp); passed to the backend so it can bind resources

// One realized draw of a submit: recompiled VS+PS SPIR-V, the draw's OWN resolved fixed-function
// state, the two stages' resource tables (so the backend can bind the constant/vertex buffers +
// textures the shaders declare, reading their bytes from 1:1-mapped guest memory), and its vertex
// count. execute_gpustate() emits one per realized draw; the backend records ALL of them into ONE
// render pass (clear once, then draw each with its own pipeline + descriptors) so a multi-draw submit
// — e.g. Unity's background + composite, whose per-draw masks/blends/shaders differ — composites
// correctly instead of collapsing onto a single draw. The tables may be null (color-only shaders).
struct DrawItem {
    std::vector<uint32_t> vs, fs;                     // recompiled SPIR-V
    ResolvedPipelineState ps;                         // THIS draw's fixed-function state
    std::shared_ptr<ShaderResourceTable> vrt, prt;    // may be null
    uint32_t vertex_count = 3;
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
    uint64_t draw_index = 0;
    uint64_t command_order = 0;
};

// The pluggable Vulkan backend: render the submit's draw items into one image and return W*H*4 RGBA8
// pixels (or {} on failure). Empty list -> {} (nothing to draw).
using RenderFn = std::function<std::vector<uint8_t>(const std::vector<DrawItem>& items)>;

// Safe guest-address readability probe: write() to /dev/null returns EFAULT for an unmapped source
// (Linux; always-true on Windows), so callers can test a guest pointer without risking a SIGSEGV.
// Implemented in gpu_executor.cpp; shared by the executor's const-eval and HLE diagnostic probes.
bool guest_readable(uint64_t addr, uint32_t bytes);

// One resolved bindless-dynamic vertex fetch from the wave-uniform scalar const-fold in
// gpu_executor.cpp: the exact fetch instruction (pc), its SRSRC SGPR, and the V# live in that SGPR
// at that instruction. Exposed (with resolve_dynamic_fetch) so the fold's scalar-ALU semantics are
// unit-testable; production callers stay inside gpu_executor.cpp.
struct DynFetch {
    uint32_t fetch_pc; int srsrc; DecodedBufferDescriptor desc; uint32_t desc_v3;
    // True when the V# came from the user-data SEED fallback (never s_loaded/patched-tracked). A
    // seed entry must NOT shadow a metadata-described direct vertex buffer at the same SGPRs: the
    // by_fetch_pc dyn path models the element address as gl_VertexIndex*stride (per-attribute
    // patched V#s fold their in-record offset into the base), while a single direct V# needs the
    // faithful VADDR/inst-offset address — shadowing it collapses every attribute onto offset 0.
    bool from_seed = false;
};

// One descriptor-TABLE use recovered by the same const-fold (#294): UE4 shaders load their T#/S#/V#
// descriptors with `s_load_dwordx4/x8 sN, s[ptr:ptr+1], <imm>` from a resource table whose pointer
// sits in the user-data SGPRs, then consume them (image_sample SRSRC/SSAMP, s_buffer_load SBASE).
// The recompiler tags such a load's dest SGPRs with the load IMMEDIATE (sreg_srt) and resolves the
// consumer via by_srt_offset(imm) — so `key` here is exactly that immediate, and build_stage_table
// turns each use into a ShaderResource with srt_offset = key.
struct SrtUse {
    int kind = 0;                    // 0 = texture (t8, + s4 sampler when resolved), 1 = constant buffer (v4)
    uint32_t key = 0;                // the s_load immediate byte offset (== emit_alu's sreg_srt tag);
                                     // 0xFFFFFFFF = key-less (register-SOFFSET / negative-imm load — the
                                     // recompiler then resolves the use by its instruction pc instead)
    std::array<uint32_t, 8> t8{};    // T# dwords as loaded (kind 0)
    std::array<uint32_t, 4> v4{};    // V# dwords as loaded (kind 1)
    bool has_samp = false;
    std::array<uint32_t, 4> s4{};    // paired S# dwords (kind 0, when the SSAMP load also resolved)
    // PER-USE pc provenance (#273 — DOLL's title-composite image_sample_b): the pc of the consuming
    // image op. The load-immediate key model breaks when the same immediate appears against two
    // different table pointers (a key-0 EUD sharp colliding with a key-0 table T#) or when the load
    // has no usable key; keying the TEXTURE use by its exact instruction (ShaderResource::fetch_pc,
    // the same per-instruction provenance the vertex fetches use) is unambiguous.
    uint32_t use_pc = 0xFFFFFFFFu;   // kind 0 only (cbufs keep the key model)
    // The consuming image op WRITES the image (image_store — MIMG op 0x8): the resource must be a
    // STORAGE image, not a sampled texture (#590 — the recompiler's storage path requires
    // ResourceClass::StorageImage). Only meaningful for kind 0.
    bool is_store = false;
};
std::vector<DynFetch> resolve_dynamic_fetch(const uint32_t* code, size_t dwords,
                                            const uint32_t* user_sgprs, uint32_t nsgpr,
                                            uint32_t user_sgpr_base,
                                            std::vector<SrtUse>* srt_uses = nullptr);

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

// PROSPER_DYNTRACE_FAIL support (gpu_executor.cpp): while true, resolve_dynamic_fetch traces its
// walk and build_stage_table dumps the user-data SGPR blocks. realize_draw_item sets it around a
// replay of a FAILED vertex-stage resource build, so the diagnostic captures exactly the failing
// draw without needing the shader's address up front (the UI draws it targets are rare/phase-bound).
extern bool g_dyntrace_force;

// Assign each resource in `t` a descriptor binding from `first`: constant/vertex buffers first, then
// textures / storage images — never on binding 2 or 3 (the recompiler's two hardwired cbufs) so a
// texture-first shader can't collide two descriptor types at one binding (#157). Exposed for testing.
void assign_convention_bindings(ShaderResourceTable& t, uint32_t first);

// Build a shader stage's resource table from the folded GpuState: look up the registered shader header
// by its bound code address, read its user-data SGPR block from the sh register file, decode the V#/T#/S#
// descriptors, and assign bindings matching the recompiler+backend convention (constant buffer -> binding
// 2, vertex buffer -> binding 3, textures -> binding 4+). Returns null if the stage has no shader header
// or no resources. Implemented in gpu_executor.cpp (needs the AGC registry + descriptor decode).
std::shared_ptr<ShaderResourceTable> build_stage_table(const GpuState& st, uint64_t code_addr, bool is_ps);

// PROSPER_COMPUTELOG diagnostic: resolve every skipped DispatchDirect packet's compute shader and
// AGC resource table from its retained register snapshot. PROSPER_COMPUTELOG_DIM=WxH restricts output
// to dispatches referencing an image of that size (for example the Messenger 1024x32 grading LUT).
void diagnose_compute_dispatches(const GpuState& st, uint64_t submit_no);

struct ComputeLaunchDimensions {
    uint32_t threads_x = 0, threads_y = 0, threads_z = 0;
    uint32_t local_x = 1, local_y = 1, local_z = 1;
    uint32_t groups_x = 0, groups_y = 0, groups_z = 0;
};

// Convert sceAgcCbDispatch's API thread counts into hardware/Vulkan workgroup counts using the
// dispatch's retained COMPUTE_NUM_THREAD_* register snapshot. Zero local-size registers fall back
// to one so malformed/incomplete state never divides by zero.
ComputeLaunchDimensions resolve_compute_launch(const GpuState::Dispatch& dispatch);

struct ComputeItem {
    std::vector<uint32_t> spirv;
    std::shared_ptr<ShaderResourceTable> resources;
    ComputeLaunchDimensions launch;
    uint64_t code_addr = 0;
    uint64_t dispatch_index = 0;
    uint64_t submit_no = 0;
    uint64_t command_order = 0;
};

enum class SubmitOperationKind : uint8_t { Draw, Dispatch };
struct SubmitOperation {
    SubmitOperationKind kind = SubmitOperationKind::Draw;
    size_t index = 0;
    uint64_t command_order = 0;
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
                                                       const PixelSystemInputMapping* system_inputs = nullptr);
ShaderRecompileCacheStats shader_recompile_cache_stats();
void clear_shader_recompile_cache();

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

enum class RealizationFailureReason : uint8_t {
    None,
    Unknown,
    MissingProgram,
    ShaderRecompile,
    DescriptorContract,
    NoEffect,
    ZeroVertices,
    Filtered,
};

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
    uint32_t vertex_count = 0;
    ComputeLaunchDimensions compute_launch;
    std::vector<ShaderRealizationDiagnostic> stages;
};

using LiveComputeFn = std::function<bool(const std::vector<ComputeItem>& items)>;

// Guest GPU writes can change backing memory represented by a persistent host-side image. Backends
// register one observer so guest-memory-producing backends can invalidate overlapping cached surfaces
// without making prosper_core depend on Vulkan.
using GuestGpuWriteObserver = std::function<void(uint64_t addr, uint64_t size)>;
void set_guest_gpu_write_observer(GuestGpuWriteObserver observer);
void notify_guest_gpu_write(uint64_t addr, uint64_t size);

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

// Register the synchronous live compute backend. execute_compute_dispatches realizes every retained
// dispatch from its state snapshot and invokes the backend in stream order.
void set_submit_compute(LiveComputeFn fn);
bool have_submit_compute();

// Live render-target query (#590): the compute backend must not read a sampled input from raw guest
// memory when the LIVE RENDERER owns that surface's current pixels (an RTT color target — raw memory
// is then empty/stale, the Dead Cells 642x362 lesson). The live renderer registers this; the compute
// backend skips such dispatches loudly. Cross-device RTT sharing is the follow-up.
using LiveTargetQueryFn = std::function<bool(uint64_t gpu_addr)>;
void set_live_target_query(LiveTargetQueryFn fn);
bool is_live_render_target(uint64_t gpu_addr);
std::vector<ComputeItem> realize_compute_dispatches(const GpuState& st,
                                                     uint64_t submit_no = 0,
                                                     std::vector<OperationRealizationFailure>* failures = nullptr);
bool execute_compute_dispatches(const GpuState& st, uint64_t submit_no = 0);
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

// Realize ONE draw of `ds` (a register snapshot or the folded state) into a DrawItem: recompile the
// VS+PS, resolve fixed-function state, and — for an indexed draw — fetch the guest index buffer.
// `draw` is the PM4 draw record (index count + indexed/index_addr); null means "no record" (hand-built
// states) and renders vcount_hint vertices non-indexed. Returns false (and leaves `out` untouched) if
// the draw is a no-op (no PGM bound, recompile failed, or no color/depth/stencil effect). Shared by the default
// (folded-state, one item) and PROSPER_PERDRAW (per-draw) paths so their per-draw handling is identical.
inline bool realize_draw_item(const GpuState& ds, const GpuState::Draw* draw, uint32_t vcount_hint,
                              uint32_t max_shader_dwords, bool log, DrawItem& out,
                              OperationRealizationFailure* failure = nullptr) {
    RenderState rs = extract_render_state(ds);
    if (failure) {
        *failure = {};
        failure->kind = SubmitOperationKind::Draw;
        failure->pipeline_present = true;
        failure->pipeline = resolve_pipeline_state(rs);
        failure->color0_base = rs.color0_base;
        failure->color0_width = rs.color0_width;
        failure->color0_height = rs.color0_height;
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
        if (log) fprintf(stderr, "[exec] skip draw: no PGM bound (es=0x%llx ps=0x%llx)\n",
                         (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr);
        return false;
    }
    const bool phase_timing = getenv("PROSPER_RENDER_TIMING") != nullptr;
    const auto table_start = phase_timing
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::shared_ptr<ShaderResourceTable> vrt = build_stage_table(ds, rs.es_addr, false);
    std::shared_ptr<ShaderResourceTable> prt = build_stage_table(ds, rs.ps_addr, true);
    const auto table_done = phase_timing
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // PROSPER_RTLOG: correlate this draw's render-target address (CB_COLOR0_BASE) with the addresses of
    // the textures it SAMPLES. If a sampled texture's base equals some draw's color0_base, that surface is
    // a GPU render target (the game renders into it then samples it) -> render-to-texture (#83/#101).
    if (getenv("PROSPER_RTLOG")) {
        fprintf(stderr, "[rt] color0=0x%llx", (unsigned long long)rs.color0_base);
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
    if (!pixel_inputs.valid_mask) {
        const auto* producer = static_cast<const AgcShaderHeader*>(
            prosper_agc_shader_header_for_code(rs.es_addr));
        const auto* pixel = static_cast<const AgcShaderHeader*>(
            prosper_agc_shader_header_for_code(rs.ps_addr));
        const AgcPixelInputControls derived = derive_agc_pixel_input_controls(producer, pixel);
        if (derived.valid_mask) {
            pixel_inputs.controls = derived.controls;
            pixel_inputs.valid_mask = derived.valid_mask;
            interpolants_from_metadata = true;
        }
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
    std::vector<uint32_t> vs = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, (const uint32_t*)(uintptr_t)rs.es_addr,
        max_shader_dwords, vrt.get(), pixel_input_ptr);
    std::vector<uint32_t> fs = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, (const uint32_t*)(uintptr_t)rs.ps_addr,
        max_shader_dwords, prt.get(), nullptr, system_input_ptr);
    if (phase_timing) {
        const auto shader_done = std::chrono::steady_clock::now();
        record_draw_realization_phases(
            std::chrono::duration<double, std::milli>(table_done - table_start).count(),
            std::chrono::duration<double, std::milli>(shader_done - table_done).count());
    }
    add_stage_diagnostic(ShaderProgramStage::Vertex, rs.es_addr, vrt, vs);
    add_stage_diagnostic(ShaderProgramStage::Fragment, rs.ps_addr, prt, fs);
    if (const char* dd = getenv("PROSPER_VS_DUMP")) {   // diag: dump successful VS SPIR-V + raw RDNA2 for inspection
        static int nd = 0;
        if (nd < 3 && !vs.empty()) {
            char fn[512];
            snprintf(fn, sizeof fn, "%s/vs_%d_%llx.spv", dd, nd, (unsigned long long)rs.es_addr);
            if (FILE* f = fopen(fn, "wb")) { fwrite(vs.data(), 4, vs.size(), f); fclose(f); }
            snprintf(fn, sizeof fn, "%s/vs_%d_%llx.bin", dd, nd, (unsigned long long)rs.es_addr);
            if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)rs.es_addr, 1, 4096, f); fclose(f); }
            // Also dump the paired PS raw RDNA2 (the recompile-guard fixture needs both stages, #228).
            snprintf(fn, sizeof fn, "%s/ps_%d_%llx.bin", dd, nd, (unsigned long long)rs.ps_addr);
            if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)rs.ps_addr, 1, 4096, f); fclose(f); }
            nd++;
        }
    }
    if (vs.empty() || fs.empty()) {
        if (failure) failure->reason = RealizationFailureReason::ShaderRecompile;
        // PROSPER_DYNTRACE_FAIL=1: replay the FAILED vertex stage's resource build with the
        // dynamic-fetch walk trace + user-data block dump forced on (once per distinct VS), so the
        // failing draw's exact seeding/s_load chain is captured without knowing its address up front.
        if (vs.empty() && getenv("PROSPER_DYNTRACE_FAIL")) {
            static std::set<uint64_t> traced;
            if (traced.insert(rs.es_addr).second) {
                fprintf(stderr, "[dynfail] replaying VS 0x%llx resource build with trace:\n",
                        (unsigned long long)rs.es_addr);
                g_dyntrace_force = true;
                (void)build_stage_table(ds, rs.es_addr, false);
                g_dyntrace_force = false;
            }
        }
        // Same replay for a FAILED pixel stage (#273 — the PS-side descriptor-resolution walls).
        if (fs.empty() && getenv("PROSPER_DYNTRACE_FAIL")) {
            static std::set<uint64_t> traced_ps;
            if (traced_ps.insert(rs.ps_addr).second) {
                fprintf(stderr, "[dynfail] replaying PS 0x%llx resource build with trace:\n",
                        (unsigned long long)rs.ps_addr);
                g_dyntrace_force = true;
                (void)build_stage_table(ds, rs.ps_addr, true);
                g_dyntrace_force = false;
            }
        }
        if (log) {
            fprintf(stderr, "[exec] skip draw: recompile failed (vs=%zu fs=%zu; order=%llu "
                            "es=0x%llx ps=0x%llx color0=0x%llx/%ux%u "
                            "depth=%d/%d/op%u clear=%d/%g base=0x%llx/0x%llx "
                            "stencil=%d clear=%d/%u base=0x%llx/0x%llx)\n",
                    vs.size(), fs.size(),
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
            for (auto [tag, addr] : {std::pair{"vs", rs.es_addr}, std::pair{"ps", rs.ps_addr}}) {
                RecompileCoverage c = recompile_coverage((const uint32_t*)(uintptr_t)addr, max_shader_dwords);
                fprintf(stderr, "[exec]   %s coverage: total=%u alu=%u exp=%u tabledep=%u unsupported=%u "
                                "first_bad fmt=%d op=0x%x\n", tag, c.total, c.alu, c.exports,
                        c.table_dependent, c.unsupported, c.first_bad_fmt, c.first_bad_op);
                if (const char* dd = getenv("PROSPER_SHADER_DUMP")) {
                    // 64 KB: a large UE4 post-process PS (~1500 instrs) far exceeds the old 4 KB
                    // window, which truncated the very control-flow tail the reject is about (#319).
                    char fn[512]; snprintf(fn, sizeof fn, "%s/exec_%s_%llx.bin", dd, tag, (unsigned long long)addr);
                    if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)addr, 1, 0x10000, f); fclose(f); }
                }
            }
        }
        return false;
    }
    if (!validate_runtime_descriptor_contract("VS", vs, vrt.get(), 0, SpirvShaderStage::Vertex) ||
        !validate_runtime_descriptor_contract("PS", fs, prt.get(), 1, SpirvShaderStage::Fragment)) {
        if (failure) failure->reason = RealizationFailureReason::DescriptorContract;
        if (log) fprintf(stderr, "[exec] skip draw: strict descriptor contract failed "
                                "(es=0x%llx ps=0x%llx color0=0x%llx)\n",
                         (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr,
                         (unsigned long long)rs.color0_base);
        return false;
    }
    ResolvedPipelineState ps = resolve_pipeline_state(rs);
    // Color-disabled draws are not necessarily no-ops. Depth prepasses and stencil mask writers
    // deliberately set CB_TARGET_MASK=0, then later color draws consume their DS result. Dropping
    // those writers made The Messenger clear stencil to 0 and then test for bits 1/2 that could never
    // be produced (#520). Skip only when the draw has no observable color OR depth/stencil effect.
    const bool ds_effect = has_depth_stencil_side_effect(ps);
    if (ps.color_write_mask == 0 && !ds_effect && !getenv("PROSPER_FORCE_COLORWRITE")) {
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
        fprintf(stderr, "[draw] idx=%u vp=%d y=%.0f h=%.0f blend=%d cwm=0x%x es=0x%llx", ic,
                ps.has_viewport, ps.viewport_y, ps.viewport_h, ps.blend_enable, ps.color_write_mask,
                (unsigned long long)rs.es_addr);
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
    // fall through: `vertex_count = vcount_hint ? vcount_hint : 3` already fabricated 3, and the
    // vb_entries override below would then sweep the ENTIRE residual vertex pool (0..vb_entries-1) of
    // whatever geometry the last-bound VB still holds — turning a no-op into a phantom triangle or a
    // full-VB draw of stale geometry composited into the frame (#400). The vb_entries "truer count"
    // override exists to correct a LOW/stale count, never to synthesize one for a zero-count draw.
    if (vcount_hint == 0 && out.indices.empty()) {
        if (failure) failure->reason = RealizationFailureReason::ZeroVertices;
        if (log) fprintf(stderr, "[exec] skip draw: zero vertex count (no-op draw)\n");
        return false;
    }
    if (out.indices.empty() && vb_entries > vertex_count) vertex_count = vb_entries;
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
            uint64_t need = (uint64_t)vertex_count * r.stride;
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
                        r.binding, (unsigned long long)r.gpu_addr, r.stride, r.size, (unsigned)r.format, r.num_components);
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
                if (FILE* f = fopen(fn, "wb")) { fwrite(vs.data(), 4, vs.size(), f); fclose(f); }
                // The recompiled PIXEL shader — the text is invisible because the PS discards every fragment
                // (SDF alpha-test). Dump it (spirv-dis) to inspect the discard condition + atlas sample. #257.
                snprintf(fn, sizeof fn, "%s/caption_ps_%llx.spv", dd, (unsigned long long)rs.ps_addr);
                if (FILE* f = fopen(fn, "wb")) { fwrite(fs.data(), 4, fs.size(), f); fclose(f); }
                snprintf(fn, sizeof fn, "%s/caption_ps_raw_%llx.bin", dd, (unsigned long long)rs.ps_addr);
                if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)rs.ps_addr, 1, 8192, f); fclose(f); }
            }
        }
    }
    out.vs = std::move(vs); out.fs = std::move(fs); out.ps = ps;
    out.vrt = std::move(vrt); out.prt = std::move(prt); out.vertex_count = vertex_count;
    out.color0_base = rs.color0_base;   // render-to-texture: the target this draw writes into (#167)
    out.color0_width = rs.color0_width; out.color0_height = rs.color0_height; // per-target extent (#526)
    return true;
}

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
                                                    std::vector<OperationRealizationFailure>* failures = nullptr) {
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
        for (size_t i = 0; i < st.draws.size(); i++) {
            DrawItem it;
            OperationRealizationFailure failure;
            if (realize_draw_item(st.state_at_draw(i), &st.draws[i], st.draws[i].index_count,
                                  max_shader_dwords, log, it, failures ? &failure : nullptr)) {
                it.draw_index = i;
                it.command_order = st.draws[i].command_order;
                items.push_back(std::move(it));
            } else if (failures) {
                failure.index = i;
                failure.command_order = st.draws[i].command_order;
                failures->push_back(std::move(failure));
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
                    max_shader_dwords, log, ignored, &failure);
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
                              failures ? &failure : nullptr)) {
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
    // Scale each item's guest viewport to the actual (possibly reduced-resolution) framebuffer. Skip if
    // 1.0 (full-res) or if this draw has no guest viewport (the backend then uses the full-target default).
    if (vp_scale_x != 1.0f || vp_scale_y != 1.0f)
        for (auto& it : items)
            if (it.ps.has_viewport) {
                it.ps.viewport_x *= vp_scale_x; it.ps.viewport_w *= vp_scale_x;
                it.ps.viewport_y *= vp_scale_y; it.ps.viewport_h *= vp_scale_y;
            }
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
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height);

// Register (or clear, with {}) the live render backend that agc_driver_submit_dcb uses on each submit.
void set_submit_renderer(LiveRenderFn fn);
bool have_submit_renderer();

struct LiveRenderPhase {
    bool first_span = true;
    bool final_span = true;
};
LiveRenderPhase live_render_phase();

// Invoke the registered live backend directly with already-realized draws. Used by the local capture
// replayer; normal guest execution enters through execute_and_present(). Returns {} when unregistered.
std::vector<uint8_t> render_submit_items(const std::vector<DrawItem>& items,
                                         uint32_t width, uint32_t height);
bool execute_compute_items(const std::vector<ComputeItem>& items);

// Render a folded GpuState at (width,height) via the registered live renderer and hand the frame to the
// present path (present_write_frame). Returns true iff a frame was produced and presented. A no-op
// returning false when there is no renderer registered or the state has no draws — so it is inert on the
// game path until the runtime wires a device, yet fully exercised by tests. Stage A of GPU_EXECUTOR_DESIGN.
bool execute_and_present(const GpuState& st, uint32_t width, uint32_t height);

// Execute retained graphics and compute work in PM4 order. Graphics spans share the frontend's
// persistent render-target cache, and only the final span is handed to the present path.
bool execute_ordered_and_present(const GpuState& st, uint32_t width, uint32_t height,
                                 uint64_t submit_no = 0);

} // namespace prosper::gpu
