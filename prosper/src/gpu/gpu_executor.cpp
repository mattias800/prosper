// gpu_executor.cpp — the live-submit half of the GPU executor (Stage A of docs/GPU_EXECUTOR_DESIGN.md).
//
// Holds the process-wide live render backend and drives it on each AGC submit. This is deliberately the
// ONLY place the executor touches process-global state; execute_gpustate() itself (gpu_execute.hpp) stays
// pure. No Vulkan here — the backend is a std::function injected by whoever owns a device (the runtime
// binary at startup, or a test via render_runner.h), so prosper_core links this without Vulkan.
#include "gpu_execute.hpp"
#include "gpu_capture.hpp"
#include "gpu_timeline.hpp"
#include "capture_compute_policy.hpp"
#include "videoout_present.hpp"   // present_write_frame
#include "agc_shader_layout.hpp"  // AgcShaderHeader + build_shader_resources
#include "pm4_registers.hpp"      // SPI_SHADER_USER_DATA_* offsets
#include "rdna2_decode.hpp"       // rdna2_walk (for the vertex-fetch const-eval)
#include "rdna2_to_spirv.hpp"     // recompile_compute
#include "writer_provenance.hpp"
#include "../host/guest_memory_map.hpp"
#include "../host/guest_write_watch.hpp"
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <bitset>
#include <condition_variable>
#include <filesystem>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif
#endif

// Look up a registered AGC shader header by its bound code address (hle_agc.cpp). Layout-compatible
// with gpu::AgcShaderHeader (file_header@0, user_data@0x08, code@0x10, type@0x5a).
extern "C" const void* prosper_agc_shader_header_for_code(uint64_t code_addr);
// #305 instrument (hle_agc.cpp): every registered shader bound to one code address, oldest first.
extern "C" size_t prosper_agc_shader_headers_for_code(uint64_t code_addr, const void** out,
                                                      size_t max);
extern "C" size_t prosper_agc_shader_count();
extern "C" const void* prosper_agc_shader_at(size_t index);

namespace prosper::gpu {

bool should_log_recompile_reject(uint64_t es_addr, uint64_t ps_addr,
                                 size_t vs_words, size_t gs_words, size_t fs_words,
                                 uint64_t* occurrence) {
    using Key = std::tuple<uint64_t, uint64_t, size_t, size_t, size_t>;
    static std::mutex mutex;
    static std::map<Key, uint64_t> counts;
    std::lock_guard<std::mutex> lock(mutex);
    uint64_t& count = counts[{es_addr, ps_addr, vs_words, gs_words, fs_words}];
    if (count != UINT64_MAX) ++count;
    if (occurrence) *occurrence = count;
    return count != 0 && (count & (count - 1)) == 0;
}

namespace {
LiveRenderFn g_live;   // empty until the runtime/test registers a device-backed renderer
LiveComputeFn g_compute;   // synchronous compute backend, registered with the live Vulkan frontend
GuestGpuWriteObserver g_guest_gpu_write_observer;
ComputeAuthorityBoundaryObserver g_compute_authority_boundary_observer;
std::mutex g_compute_authority_boundary_mutex;
std::atomic<bool> g_compute_authority_boundary_enabled{false};
thread_local LiveRenderPhase g_live_phase;
std::array<uint8_t, 64 * 1024> g_compute_gds{};

struct GuestGpuWriteRange {
    uint64_t addr = 0;
    uint64_t size = 0;
};
struct GuestGpuWriteJournal {
    bool active = false;
    bool overflowed = false;
    uint64_t submit_serial = 0;
    std::vector<GuestGpuWriteRange> writes;
};
thread_local GuestGpuWriteJournal g_guest_gpu_writes;
std::atomic<uint64_t> g_next_guest_gpu_submit_serial{0};
std::atomic<uint64_t> g_next_shader_analysis_identity{0};

void dispatch_compute_authority_boundary(const ComputeAuthorityBoundary& boundary) {
    // Default path: one relaxed load and no mutex/std::function copy when the opt-in diagnostic is
    // absent. This hook sits on every ordered draw, so making an unarmed census measurable would
    // defeat the performance investigation it exists to support.
    if (!g_compute_authority_boundary_enabled.load(std::memory_order_relaxed)) return;
    ComputeAuthorityBoundaryObserver observer;
    {
        std::lock_guard lock(g_compute_authority_boundary_mutex);
        observer = g_compute_authority_boundary_observer;
    }
    if (observer) observer(boundary);
}

void notify_compute_authority_range(ComputeAuthorityBoundaryKind kind,
                                    uint64_t submit_no, uint64_t command_order,
                                    uint64_t address, uint64_t bytes) {
    const bool known = address != 0 && bytes != 0 &&
        address <= UINT64_MAX - (bytes - 1);
    dispatch_compute_authority_boundary(
        {kind, submit_no, command_order, address, bytes, known});
}

void notify_compute_authority_unknown(ComputeAuthorityBoundaryKind kind,
                                      uint64_t submit_no, uint64_t command_order = 0) {
    dispatch_compute_authority_boundary(
        {kind, submit_no, command_order, 0, 0, false});
}

void notify_compute_authority_draw_resources(const DrawItem& item, uint64_t submit_no) {
    if (!g_compute_authority_boundary_enabled.load(std::memory_order_relaxed)) return;
    for (const ComputeAuthorityBoundary& boundary :
         compute_authority_draw_resource_boundaries(item, submit_no))
        dispatch_compute_authority_boundary(boundary);
}

void notify_compute_authority_draw_unrealized(uint64_t submit_no,
                                              uint64_t command_order) {
    if (!g_compute_authority_boundary_enabled.load(std::memory_order_relaxed)) return;
    dispatch_compute_authority_boundary({
        ComputeAuthorityBoundaryKind::DrawResourceEnd,
        submit_no,
        command_order,
        0,
        0,
        false,
        UINT32_MAX,
        UINT32_MAX,
        false,
    });
}

bool ranges_overlap(uint64_t a, uint64_t a_size, uint64_t b, uint64_t b_size) {
    if (!a_size || !b_size) return false;
    return a <= b ? b - a < a_size : a - b < b_size;
}

class GuestGpuWriteSubmitScope {
public:
    GuestGpuWriteSubmitScope() {
        // Ordered execution is synchronous and non-recursive. If that contract is ever broken,
        // force conservative validation rather than silently losing the outer write history.
        if (g_guest_gpu_writes.active) {
            nested_ = true;
            g_guest_gpu_writes.overflowed = true;
            return;
        }
        g_guest_gpu_writes.active = true;
        g_guest_gpu_writes.overflowed = false;
        g_guest_gpu_writes.submit_serial =
            g_next_guest_gpu_submit_serial.fetch_add(1, std::memory_order_relaxed) + 1;
        g_guest_gpu_writes.writes.clear();
    }
    ~GuestGpuWriteSubmitScope() {
        if (!nested_) g_guest_gpu_writes.active = false;
    }

private:
    bool nested_ = false;
};

// Read a 32-dword user-data SGPR block from a stage's register file. `base` = the stage's
// SPI_SHADER_USER_DATA_*_0 register offset; absent registers read as 0. 32 (not 16) because NGG merged
// shaders place descriptors in the extended user SGPRs s16..s31 (e.g. vertex buffers at s16/s18).
static constexpr uint32_t kUserSgprs = 32;
void read_user_sgprs(const std::unordered_map<uint32_t, uint32_t>& sh, uint32_t base, uint32_t out[kUserSgprs]) {
    for (uint32_t i = 0; i < kUserSgprs; i++) { auto it = sh.find(base + i); out[i] = it == sh.end() ? 0u : it->second; }
}

} // namespace (guest_readable below has external linkage — declared in gpu_execute.hpp, shared with
  // the HLE diagnostic probes that chase raw guest pointers)

// PROSPER_DYNTRACE_FAIL support: while true, resolve_dynamic_fetch traces its walk and
// build_stage_table dumps the user-data blocks, regardless of the PROSPER_DYNTRACE/RESDUMP
// envs. Set (and cleared) by realize_draw_item's failure replay — the submit path is serialized
// by the HLE submit mutex, so a plain global is safe there.
bool g_dyntrace_force = false;

bool dyntrace_failed_shader_enabled(uint64_t code_addr) {
    if (!std::getenv("PROSPER_DYNTRACE_FAIL")) return false;
    const char* filter = std::getenv("PROSPER_DYNTRACE_FAIL_ADDR");
    if (!filter) return true;

    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(filter, &end, 16);
    return errno == 0 && end != filter && *end == '\0' && parsed == code_addr;
}

namespace {

struct ShaderResourceCompileKey {
    uint32_t cls = 0;
    uint32_t format = 0;
    uint32_t num_components = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t img_dim = 0;
    uint32_t sample_count = 1;
    bool in_mip_tail = false;
    bool compression_enabled = false;
    uint32_t binding = 0;
    uint32_t stride = 0;
    uint32_t one_record_tail_semantic = 0;
    uint32_t srt_offset = 0;
    uint32_t sgpr_base = 0;
    uint32_t fetch_pc = 0;
    uint32_t fetch_index_mode = 0;
    uint32_t flat_base_sgpr = 0;
    uint32_t bvh_box_grow = 0;
    bool null_bvh = false;
    bool srgb = false;
    bool depth_compare = false;
    uint32_t depth_compare_func = 0;
    uint32_t mag_filter = 0;
    uint32_t addr_u = 0;
    uint32_t addr_v = 0;
    uint32_t border_color_type = 0;
    bool normalize_unnormalized_coordinates = false;

    bool operator==(const ShaderResourceCompileKey&) const = default;
};

struct ShaderCompileKey {
    ShaderProgramStage stage = ShaderProgramStage::Vertex;
    bool has_resource_table = false;
    bool force_position_w = false;
    bool capture_position = false;
    bool has_pixel_inputs = false;
    PixelInputMapping pixel_inputs{};
    bool has_system_inputs = false;
    PixelSystemInputMapping system_inputs{};
    bool fragment_wave32 = false;
    bool has_pcrel_dispatch = false;
    uint32_t pcrel_dispatch_target = UINT32_MAX;
    // Compute modules also depend on launch ABI shape. User SGPR VALUES are push constants and stay
    // runtime data; only their count changes declarations. Exact thread extents matter because a
    // partial final workgroup emits a literal invocation guard into SPIR-V.
    bool has_compute_config = false;
    uint32_t compute_user_sgpr_count = 0;
    uint32_t compute_local_x = 1, compute_local_y = 1, compute_local_z = 1;
    bool compute_exact_thread_extent = false;
    uint32_t compute_threads_x = 0, compute_threads_y = 0, compute_threads_z = 0;
    uint32_t compute_wave_size = 64;
    uint32_t compute_tidig_comp_cnt = 0;
    bool compute_tgid_x_en = false, compute_tgid_y_en = false, compute_tgid_z_en = false;
    bool compute_tg_size_en = false;
    uint32_t compute_lds_bytes = 0;
    uint32_t compute_native_subgroup_size = 0;
    uint32_t compute_native_storage_format_support = 0;
    bool compute_packed_r11_storage = true;
    uint32_t vertex_lds_dwords = 0;
    uint32_t vertices_per_instance = 0;
    // Aliases ShaderCodeAnalysis::code and keeps that immutable analysis alive. Warm lookups used to
    // copy and re-hash the complete raw program for every draw before reaching the shader cache.
    std::shared_ptr<const std::vector<uint32_t>> code;
    uint64_t code_hash = 0;
    // Optional main program reached by a separately-installed vertex-fetch prolog. It remains a
    // distinct immutable analysis so cache identity covers both allocations without constructing a
    // transient concatenated buffer on every warm draw.
    std::shared_ptr<const std::vector<uint32_t>> chain_code;
    uint64_t chain_code_hash = 0;
    std::vector<ShaderResourceCompileKey> resources;
    size_t cached_hash = 0;

    bool operator==(const ShaderCompileKey& other) const {
        const bool same_code = code == other.code ||
            (code && other.code && *code == *other.code);
        const bool same_chain_code = chain_code == other.chain_code ||
            (chain_code && other.chain_code && *chain_code == *other.chain_code);
        return stage == other.stage &&
               has_resource_table == other.has_resource_table &&
               force_position_w == other.force_position_w &&
               capture_position == other.capture_position &&
               has_pixel_inputs == other.has_pixel_inputs &&
               pixel_inputs == other.pixel_inputs &&
               has_system_inputs == other.has_system_inputs &&
               system_inputs == other.system_inputs &&
               fragment_wave32 == other.fragment_wave32 &&
               has_pcrel_dispatch == other.has_pcrel_dispatch &&
               pcrel_dispatch_target == other.pcrel_dispatch_target &&
               has_compute_config == other.has_compute_config &&
               compute_user_sgpr_count == other.compute_user_sgpr_count &&
               compute_local_x == other.compute_local_x &&
               compute_local_y == other.compute_local_y &&
               compute_local_z == other.compute_local_z &&
               compute_exact_thread_extent == other.compute_exact_thread_extent &&
               compute_threads_x == other.compute_threads_x &&
               compute_threads_y == other.compute_threads_y &&
               compute_threads_z == other.compute_threads_z &&
               compute_wave_size == other.compute_wave_size &&
               compute_tidig_comp_cnt == other.compute_tidig_comp_cnt &&
               compute_tgid_x_en == other.compute_tgid_x_en &&
               compute_tgid_y_en == other.compute_tgid_y_en &&
               compute_tgid_z_en == other.compute_tgid_z_en &&
               compute_tg_size_en == other.compute_tg_size_en &&
               compute_lds_bytes == other.compute_lds_bytes &&
               compute_native_subgroup_size == other.compute_native_subgroup_size &&
               compute_native_storage_format_support ==
                   other.compute_native_storage_format_support &&
               compute_packed_r11_storage == other.compute_packed_r11_storage &&
               vertex_lds_dwords == other.vertex_lds_dwords &&
               vertices_per_instance == other.vertices_per_instance &&
               resources == other.resources && same_code && same_chain_code;
    }
};

static uint64_t hash_mix(uint64_t hash, uint64_t value) {
    // FNV-1a over fixed-width values. Equality still compares the full key, so collisions are benign.
    for (unsigned i = 0; i < 8; ++i) {
        hash ^= static_cast<uint8_t>(value >> (i * 8));
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint64_t hash_shader_code(const std::vector<uint32_t>& code) {
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t word : code) hash = hash_mix(hash, word);
    return hash;
}

struct ShaderCompileKeyHash {
    static size_t compute(const ShaderCompileKey& key) {
        uint64_t hash = 1469598103934665603ull;
        hash = hash_mix(hash, static_cast<uint32_t>(key.stage));
        hash = hash_mix(hash, key.has_resource_table);
        hash = hash_mix(hash, key.force_position_w);
        hash = hash_mix(hash, key.capture_position);
        hash = hash_mix(hash, key.has_pixel_inputs);
        if (key.has_pixel_inputs) {
            hash = hash_mix(hash, key.pixel_inputs.valid_mask);
            hash = hash_mix(hash, key.pixel_inputs.passthrough_mask);
            for (uint32_t control : key.pixel_inputs.controls)
                hash = hash_mix(hash, control);
        }
        hash = hash_mix(hash, key.has_system_inputs);
        if (key.has_system_inputs) {
            hash = hash_mix(hash, key.system_inputs.ena);
            hash = hash_mix(hash, key.system_inputs.addr);
        }
        hash = hash_mix(hash, key.fragment_wave32);
        hash = hash_mix(hash, key.has_pcrel_dispatch);
        if (key.has_pcrel_dispatch) hash = hash_mix(hash, key.pcrel_dispatch_target);
        hash = hash_mix(hash, key.has_compute_config);
        if (key.has_compute_config) {
            hash = hash_mix(hash, key.compute_user_sgpr_count);
            hash = hash_mix(hash, key.compute_local_x);
            hash = hash_mix(hash, key.compute_local_y);
            hash = hash_mix(hash, key.compute_local_z);
            hash = hash_mix(hash, key.compute_exact_thread_extent);
            hash = hash_mix(hash, key.compute_threads_x);
            hash = hash_mix(hash, key.compute_threads_y);
            hash = hash_mix(hash, key.compute_threads_z);
            hash = hash_mix(hash, key.compute_wave_size);
            hash = hash_mix(hash, key.compute_tidig_comp_cnt);
            hash = hash_mix(hash, key.compute_tgid_x_en);
            hash = hash_mix(hash, key.compute_tgid_y_en);
            hash = hash_mix(hash, key.compute_tgid_z_en);
            hash = hash_mix(hash, key.compute_tg_size_en);
            hash = hash_mix(hash, key.compute_lds_bytes);
            hash = hash_mix(hash, key.compute_native_subgroup_size);
            hash = hash_mix(hash, key.compute_native_storage_format_support);
            hash = hash_mix(hash, key.compute_packed_r11_storage);
        }
        hash = hash_mix(hash, key.vertex_lds_dwords);
        hash = hash_mix(hash, key.vertices_per_instance);
        hash = hash_mix(hash, key.code ? key.code->size() : 0u);
        hash = hash_mix(hash, key.code_hash);
        hash = hash_mix(hash, key.chain_code ? key.chain_code->size() : 0u);
        hash = hash_mix(hash, key.chain_code_hash);
        hash = hash_mix(hash, key.resources.size());
        for (const auto& resource : key.resources) {
            hash = hash_mix(hash, resource.cls);
            hash = hash_mix(hash, resource.format);
            hash = hash_mix(hash, resource.num_components);
            hash = hash_mix(hash, resource.width);
            hash = hash_mix(hash, resource.height);
            hash = hash_mix(hash, resource.depth);
            hash = hash_mix(hash, resource.img_dim);
            hash = hash_mix(hash, resource.sample_count);
            hash = hash_mix(hash, resource.in_mip_tail);
            hash = hash_mix(hash, resource.compression_enabled);
            hash = hash_mix(hash, resource.binding);
            hash = hash_mix(hash, resource.stride);
            hash = hash_mix(hash, resource.one_record_tail_semantic);
            hash = hash_mix(hash, resource.srt_offset);
            hash = hash_mix(hash, resource.sgpr_base);
            hash = hash_mix(hash, resource.fetch_pc);
            hash = hash_mix(hash, resource.fetch_index_mode);
            hash = hash_mix(hash, resource.flat_base_sgpr);
            hash = hash_mix(hash, resource.bvh_box_grow);
            hash = hash_mix(hash, resource.null_bvh);
            hash = hash_mix(hash, resource.srgb);
            hash = hash_mix(hash, resource.depth_compare);
            hash = hash_mix(hash, resource.depth_compare_func);
            hash = hash_mix(hash, resource.mag_filter);
            hash = hash_mix(hash, resource.addr_u);
            hash = hash_mix(hash, resource.addr_v);
            hash = hash_mix(hash, resource.border_color_type);
            hash = hash_mix(hash, resource.normalize_unnormalized_coordinates);
        }
        return static_cast<size_t>(hash);
    }

    size_t operator()(const ShaderCompileKey& key) const { return key.cached_hash; }
};

struct CachedShader {
    SharedShaderWords spirv;
    uint64_t identity = 0;
    uint64_t last_use = 0;
    uint64_t bytes = 0;
};

struct ShaderCache {
    std::mutex mutex;
    std::unordered_map<ShaderCompileKey, CachedShader, ShaderCompileKeyHash> entries;
    ShaderRecompileCacheStats stats;
    uint64_t use_counter = 0;
    uint64_t next_identity = 1;
};

ShaderCache& shader_cache() {
    static ShaderCache cache;
    return cache;
}

struct DecodedShader {
    std::vector<uint32_t> code;
    std::vector<Rdna2Inst> instructions;
    std::vector<Rdna2Inst> shader_constant_instructions;
    size_t source_dwords = 0;
    bool shader_constant_specialized = false;
    bool terminated = false;
    uint64_t bytes = 0;
};

struct DecodedShaderEntry {
    std::shared_ptr<const DecodedShader> shader;
    uint64_t last_use = 0;
};

struct ShaderDecodeCache {
    std::mutex mutex;
    std::unordered_map<uintptr_t, DecodedShaderEntry> entries;
    ShaderDecodeCacheStats stats;
    uint64_t use_counter = 0;
};

struct ShaderCodeAnalysis {
    std::vector<uint32_t> code;
    uint64_t code_hash = 0;
    PcrelDispatchInfo pcrel_dispatch;
    uint64_t identity = 0;
    size_t source_dwords = 0;
    bool bounded_span = false;
    uint64_t bytes = 0;
};

struct ShaderCodeAnalysisEntry {
    std::shared_ptr<const ShaderCodeAnalysis> analysis;
    uint64_t last_use = 0;
};

struct ShaderAnalysisCache {
    std::mutex mutex;
    std::unordered_map<uintptr_t, ShaderCodeAnalysisEntry> entries;
    ShaderAnalysisCacheStats stats;
    uint64_t use_counter = 0;
};

struct InterpolationCacheKey {
    uint64_t analysis_identity = 0;
    PixelSystemInputMapping system_inputs{};
    bool has_system_inputs = false;
    uint32_t passthrough_mask = 0;

    bool operator==(const InterpolationCacheKey&) const = default;
};

struct InterpolationCacheKeyHash {
    size_t operator()(const InterpolationCacheKey& key) const {
        uint64_t hash = 1469598103934665603ull;
        hash = hash_mix(hash, key.analysis_identity);
        hash = hash_mix(hash, key.has_system_inputs);
        if (key.has_system_inputs) {
            hash = hash_mix(hash, key.system_inputs.ena);
            hash = hash_mix(hash, key.system_inputs.addr);
        }
        hash = hash_mix(hash, key.passthrough_mask);
        return static_cast<size_t>(hash);
    }
};

struct CachedInterpolationLayout {
    FragmentInterpolationLayout layout;
    uint64_t last_use = 0;
};

struct InterpolationCache {
    std::mutex mutex;
    std::unordered_map<InterpolationCacheKey, CachedInterpolationLayout,
                       InterpolationCacheKeyHash> entries;
    uint64_t use_counter = 0;
};

struct StageFoldProfileEntry {
    uint64_t code = 0;
    uint32_t user_base = 0;
    uint64_t calls = 0;
    uint64_t instructions = 0;
    uint64_t dynamic_fetches = 0;
    uint64_t srt_uses = 0;
    uint64_t code_dwords = 0;
    uint64_t guest_probes = 0;
    double total_ms = 0.0;
    double decode_ms = 0.0;
    double guest_probe_ms = 0.0;
    double max_ms = 0.0;
};

struct StageFoldProfileKey {
    uint64_t code = 0;
    uint32_t user_base = 0;
    bool operator==(const StageFoldProfileKey&) const = default;
};

struct StageFoldProfileKeyHash {
    size_t operator()(const StageFoldProfileKey& key) const {
        const uint64_t mixed = key.code ^ (static_cast<uint64_t>(key.user_base) << 48);
        return static_cast<size_t>(mixed ^ (mixed >> 32));
    }
};

struct StageFoldProfiler {
    std::mutex mutex;
    std::unordered_map<StageFoldProfileKey, StageFoldProfileEntry,
                       StageFoldProfileKeyHash> window;
    uint64_t calls = 0;
};

StageFoldProfiler& stage_fold_profiler() {
    static StageFoldProfiler profiler;
    return profiler;
}

void record_stage_fold_profile(uint64_t code, uint32_t user_base, size_t code_dwords,
                               size_t instructions, size_t dynamic_fetches, size_t srt_uses,
                               uint64_t guest_probes, double elapsed_ms, double decode_ms,
                               double guest_probe_ms) {
    static const uint64_t interval = [] {
        const char* value = std::getenv("PROSPER_STAGE_FOLD_PROFILE_CALLS");
        if (!value || !*value) return 4096ull;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        return end != value && parsed > 0 ? parsed : 4096ull;
    }();
    StageFoldProfiler& profiler = stage_fold_profiler();
    std::lock_guard lock(profiler.mutex);
    StageFoldProfileEntry& entry = profiler.window[{code, user_base}];
    entry.code = code;
    entry.user_base = user_base;
    ++entry.calls;
    entry.instructions += instructions;
    entry.dynamic_fetches += dynamic_fetches;
    entry.srt_uses += srt_uses;
    entry.code_dwords += code_dwords;
    entry.guest_probes += guest_probes;
    entry.total_ms += elapsed_ms;
    entry.decode_ms += decode_ms;
    entry.guest_probe_ms += guest_probe_ms;
    entry.max_ms = std::max(entry.max_ms, elapsed_ms);
    if (++profiler.calls < interval) return;

    std::vector<StageFoldProfileEntry> ranked;
    ranked.reserve(profiler.window.size());
    for (const auto& item : profiler.window) ranked.push_back(item.second);
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.total_ms > b.total_ms;
    });
    std::fprintf(stderr, "[stage-fold-profile] calls=%llu shaders=%zu top-by-total-ms:\n",
                 (unsigned long long)profiler.calls, ranked.size());
    for (size_t i = 0; i < std::min<size_t>(ranked.size(), 12); ++i) {
        const StageFoldProfileEntry& item = ranked[i];
        const double calls = static_cast<double>(item.calls);
        std::fprintf(stderr,
                     "[stage-fold-profile] code=0x%llx base=%u calls=%llu total=%.3f avg=%.3f "
                     "max=%.3f decode=%.3f probe=%.3f body=%.3f dw/call=%.1f ins/call=%.1f "
                     "probes/call=%.1f dyn/call=%.1f srt/call=%.1f\n",
                     (unsigned long long)item.code, item.user_base,
                     (unsigned long long)item.calls, item.total_ms, item.total_ms / calls,
                     item.max_ms, item.decode_ms / calls, item.guest_probe_ms / calls,
                     (item.total_ms - item.decode_ms - item.guest_probe_ms) / calls,
                     item.code_dwords / calls, item.instructions / calls,
                     item.guest_probes / calls, item.dynamic_fetches / calls, item.srt_uses / calls);
    }
    profiler.window.clear();
    profiler.calls = 0;
}

ShaderDecodeCache& shader_decode_cache() {
    static ShaderDecodeCache cache;
    return cache;
}

ShaderAnalysisCache& shader_analysis_cache() {
    static ShaderAnalysisCache cache;
    return cache;
}

InterpolationCache& interpolation_cache() {
    static InterpolationCache cache;
    return cache;
}

uint64_t shader_decode_cache_limit_bytes() {
    constexpr uint64_t default_bytes = 64ull * 1024 * 1024;
    const char* value = getenv("PROSPER_SHADER_DECODE_CACHE_MB");
    if (!value || !*value) return default_bytes;
    char* end = nullptr;
    const unsigned long long mib = strtoull(value, &end, 10);
    if (end == value || *end != '\0') return default_bytes;
    return std::min<uint64_t>(mib, 1024ull) * 1024 * 1024;
}

std::shared_ptr<const DecodedShader> decode_shader_cached(const uint32_t* code, size_t dwords) {
    auto decode = [&] {
        auto result = std::make_shared<DecodedShader>();
        result->source_dwords = dwords;
        std::vector<Rdna2Inst> decoded;
        const size_t consumed = rdna2_walk(code, dwords, decoded);
        result->code.assign(code, code + consumed);
        if (!decoded.empty()) {
            const Rdna2Inst& last = decoded.back();
            result->terminated = last.is_end || last.fmt == Rdna2Format::Unknown ||
                                 last.len_dwords == 0;
        }
        // The fold ignores most vector/control/export instructions unless the decoder reports an SGPR
        // destination (the conservative unknown-value invalidation). Scalar lane spills are the exception:
        // retain their spill VGPR writes too, so an ordinary VGPR write invalidates any saved lane values.
        std::set<int> scalar_spill_vgprs;
        std::set<int> fetch_vaddr_vgprs;
        for (const Rdna2Inst& instruction : decoded) {
            if ((instruction.fmt == Rdna2Format::MUBUF ||
                 instruction.fmt == Rdna2Format::MTBUF) &&
                instruction.src[0].kind == OperandKind::VGPR)
                fetch_vaddr_vgprs.insert(instruction.src[0].value);
            if (instruction.fmt == Rdna2Format::VOP3) {
                if (instruction.opcode == 0x361 && instruction.dst.kind == OperandKind::VGPR)
                    scalar_spill_vgprs.insert(instruction.dst.value);       // v_writelane_b32
                else if (instruction.opcode == 0x360 && instruction.src[0].kind == OperandKind::VGPR)
                    scalar_spill_vgprs.insert(instruction.src[0].value);    // v_readlane_b32
            }
        }
        auto retain_fold_instructions = [&](const std::vector<Rdna2Inst>& source,
                                            std::vector<Rdna2Inst>& retained) {
            retained.reserve(source.size());
            for (const Rdna2Inst& instruction : source) {
                if (instruction.is_end) break;
                const bool scalar_spill = instruction.fmt == Rdna2Format::VOP3 &&
                                          (instruction.opcode == 0x360 || instruction.opcode == 0x361);
                const bool vector_index_select =
                    ((instruction.fmt == Rdna2Format::VOP3 && instruction.opcode == 0x101) ||
                     (instruction.fmt == Rdna2Format::VOP2 && instruction.opcode == 0x01)) &&
                    instruction.dst.kind == OperandKind::VGPR &&
                    fetch_vaddr_vgprs.contains(instruction.dst.value);
                // Index provenance begins at the hardware ABI VGPRs and is killed by any later shader
                // computation of a register used as VADDR. Keep only writes to actual fetch-address
                // registers; retaining every VALU instruction would make the otherwise-small scalar fold
                // walk large UE shaders in full. This is what distinguishes DQ's first v5=vertex_id fetch
                // from its later v5=3*vertex_id+1 packed-attribute fetch.
                const bool fetch_vaddr_write = instruction.dst.kind == OperandKind::VGPR &&
                                               fetch_vaddr_vgprs.contains(instruction.dst.value);
                const bool scalar_spill_invalidation = instruction.dst.kind == OperandKind::VGPR &&
                                                       scalar_spill_vgprs.contains(instruction.dst.value);
                const bool fold_format = instruction.fmt == Rdna2Format::SOP1 ||
                                         instruction.fmt == Rdna2Format::SOP2 ||
                                         instruction.fmt == Rdna2Format::SOPC ||
                                         instruction.fmt == Rdna2Format::SOPK ||
                                         instruction.fmt == Rdna2Format::SOPP ||
                                         instruction.fmt == Rdna2Format::SMEM ||
                                         instruction.fmt == Rdna2Format::MIMG ||
                                         instruction.fmt == Rdna2Format::MUBUF ||
                                         instruction.fmt == Rdna2Format::MTBUF;
                if (fold_format || scalar_spill || vector_index_select || fetch_vaddr_write ||
                    scalar_spill_invalidation || instruction.dst.kind == OperandKind::SGPR)
                    retained.push_back(instruction);
            }
        };
        // Retain only instructions that can affect fold state or emit a descriptor use, preserving
        // their original order and PCs. Prove shader-constant branches against the FULL decoded
        // stream first: the compact fold stream intentionally omits most VALU, including implicit
        // VCC writers that must invalidate a scalar-data proof through s106:s107.
        retain_fold_instructions(decoded, result->instructions);
        std::vector<Rdna2Inst> shader_constant_decoded = decoded;
        result->shader_constant_specialized =
            rdna2_specialize_shader_constant_branches(shader_constant_decoded) != 0;
        if (result->shader_constant_specialized)
            retain_fold_instructions(shader_constant_decoded,
                                     result->shader_constant_instructions);
        result->bytes = static_cast<uint64_t>(result->code.size()) * sizeof(uint32_t) +
                        static_cast<uint64_t>(result->instructions.size() +
                                              result->shader_constant_instructions.size()) *
                            sizeof(Rdna2Inst);
        return result;
    };

    if (!code || !dwords) return decode();
    auto& cache = shader_decode_cache();
    if (getenv("PROSPER_NO_SHADER_DECODE_CACHE")) {
        std::lock_guard lock(cache.mutex);
        ++cache.stats.bypasses;
        return decode();
    }

    const uintptr_t address = reinterpret_cast<uintptr_t>(code);
    {
        std::lock_guard lock(cache.mutex);
        auto found = cache.entries.find(address);
        if (found != cache.entries.end()) {
            const auto& cached = found->second.shader;
            const bool compatible_length = cached->terminated || cached->source_dwords == dwords;
            const uint64_t code_bytes = static_cast<uint64_t>(cached->code.size()) * sizeof(uint32_t);
            const bool readable = code_bytes == 0 ||
                                  (code_bytes <= UINT32_MAX && guest_readable(address, (uint32_t)code_bytes));
            if (compatible_length && cached->code.size() <= dwords && readable &&
                (code_bytes == 0 || memcmp(code, cached->code.data(), (size_t)code_bytes) == 0)) {
                ++cache.stats.hits;
                found->second.last_use = ++cache.use_counter;
                return cached;
            }
            cache.stats.bytes -= cached->bytes;
            cache.entries.erase(found);
            ++cache.stats.invalidations;
        }
        ++cache.stats.misses;
    }

    auto decoded = decode();
    std::lock_guard lock(cache.mutex);
    // Another cold worker may have populated this address while decode ran. Replace it only after
    // removing its accounted bytes; returned shared_ptrs remain valid independently of the map entry.
    auto concurrent = cache.entries.find(address);
    if (concurrent != cache.entries.end()) {
        cache.stats.bytes -= concurrent->second.shader->bytes;
        cache.entries.erase(concurrent);
    }
    constexpr size_t max_entries = 4096;
    const uint64_t limit = shader_decode_cache_limit_bytes();
    while (!cache.entries.empty() &&
           (cache.entries.size() >= max_entries || cache.stats.bytes + decoded->bytes > limit)) {
        auto oldest = cache.entries.begin();
        for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
            if (it->second.last_use < oldest->second.last_use) oldest = it;
        cache.stats.bytes -= oldest->second.shader->bytes;
        cache.entries.erase(oldest);
        ++cache.stats.evictions;
    }
    if (decoded->bytes <= limit && max_entries != 0) {
        cache.entries[address] = {decoded, ++cache.use_counter};
        cache.stats.bytes += decoded->bytes;
    }
    cache.stats.entries = cache.entries.size();
    return decoded;
}

uint64_t shader_analysis_cache_limit_bytes() {
    constexpr uint64_t default_bytes = 64ull * 1024 * 1024;
    const char* value = getenv("PROSPER_SHADER_ANALYSIS_CACHE_MB");
    if (!value || !*value) return default_bytes;
    char* end = nullptr;
    const unsigned long long mib = strtoull(value, &end, 10);
    if (end == value || *end != '\0') return default_bytes;
    return std::min<uint64_t>(mib, 1024ull) * 1024 * 1024;
}

std::shared_ptr<const ShaderCodeAnalysis> analyze_shader_code_cached(const uint32_t* code,
                                                                      size_t dwords) {
    auto analyze = [&] {
        auto result = std::make_shared<ShaderCodeAnalysis>();
        result->identity = ++g_next_shader_analysis_identity;
        result->source_dwords = dwords;
        const size_t span = rdna2_recompile_code_span(code, dwords);
        result->bounded_span = span < dwords;
        if (code && span) result->code.assign(code, code + span);
        result->code_hash = hash_shader_code(result->code);
        result->pcrel_dispatch = rdna2_pcrel_dispatch_info(code, dwords);
        result->bytes = static_cast<uint64_t>(result->code.size()) * sizeof(uint32_t) +
                        static_cast<uint64_t>(result->pcrel_dispatch.target_pcs.size()) *
                            sizeof(uint32_t) +
                        static_cast<uint64_t>(result->pcrel_dispatch.setup_pcs.size()) *
                            sizeof(uint32_t);
        return result;
    };

    if (!code || !dwords) return analyze();
    auto& cache = shader_analysis_cache();
    if (getenv("PROSPER_NO_SHADER_ANALYSIS_CACHE")) {
        std::lock_guard lock(cache.mutex);
        ++cache.stats.bypasses;
        return analyze();
    }

    const uintptr_t address = reinterpret_cast<uintptr_t>(code);
    // Copy the immutable candidate under the mutex, then validate guest bytes without it. Dense draw
    // realization performs these exact checks from several workers; holding one global cache mutex
    // across guest_readable + memcmp otherwise serializes every warm lookup.
    for (;;) {
        std::shared_ptr<const ShaderCodeAnalysis> cached;
        {
            std::lock_guard lock(cache.mutex);
            auto found = cache.entries.find(address);
            if (found == cache.entries.end()) {
                ++cache.stats.misses;
                break;
            }
            cached = found->second.analysis;
        }
        const bool compatible_length = cached->bounded_span
            ? cached->code.size() <= dwords : cached->source_dwords == dwords;
        const uint64_t code_bytes = static_cast<uint64_t>(cached->code.size()) * sizeof(uint32_t);
        const bool readable = code_bytes == 0 ||
                              (code_bytes <= UINT32_MAX &&
                               guest_readable(address, static_cast<uint32_t>(code_bytes)));
        const bool matches = compatible_length && readable &&
            (code_bytes == 0 ||
             memcmp(code, cached->code.data(), static_cast<size_t>(code_bytes)) == 0);

        std::lock_guard lock(cache.mutex);
        auto found = cache.entries.find(address);
        if (found == cache.entries.end() || found->second.analysis != cached)
            continue; // Evicted/replaced while unlocked: validate the current version instead.
        if (matches) {
            ++cache.stats.hits;
            found->second.last_use = ++cache.use_counter;
            return cached;
        }
        cache.stats.bytes -= cached->bytes;
        cache.entries.erase(found);
        ++cache.stats.invalidations;
        ++cache.stats.misses;
        break;
    }

    auto analysis = analyze();
    std::lock_guard lock(cache.mutex);
    // Another cold worker may have populated this address while analysis ran. Keep exact byte
    // accounting when replacing it; each returned shared_ptr remains independently valid.
    auto concurrent = cache.entries.find(address);
    if (concurrent != cache.entries.end()) {
        cache.stats.bytes -= concurrent->second.analysis->bytes;
        cache.entries.erase(concurrent);
    }
    constexpr size_t max_entries = 4096;
    const uint64_t limit = shader_analysis_cache_limit_bytes();
    while (!cache.entries.empty() &&
           (cache.entries.size() >= max_entries || cache.stats.bytes + analysis->bytes > limit)) {
        auto oldest = cache.entries.begin();
        for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
            if (it->second.last_use < oldest->second.last_use) oldest = it;
        cache.stats.bytes -= oldest->second.analysis->bytes;
        cache.entries.erase(oldest);
        ++cache.stats.evictions;
    }
    if (analysis->bytes <= limit && max_entries != 0) {
        cache.entries[address] = {analysis, ++cache.use_counter};
        cache.stats.bytes += analysis->bytes;
    }
    cache.stats.entries = cache.entries.size();
    return analysis;
}

uint64_t shader_cache_limit_bytes() {
    constexpr uint64_t default_bytes = 128ull * 1024 * 1024;
    const char* value = getenv("PROSPER_SHADER_CACHE_MB");
    if (!value || !*value) return default_bytes;
    char* end = nullptr;
    const unsigned long long mib = strtoull(value, &end, 10);
    if (end == value || *end != '\0') return default_bytes;
    return std::min<uint64_t>(mib, 4096ull) * 1024 * 1024;
}

uint64_t shader_cache_entry_bytes(const ShaderCompileKey& key, const std::vector<uint32_t>& spirv) {
    return static_cast<uint64_t>(key.code ? key.code->size() : 0u) * sizeof(uint32_t) +
           static_cast<uint64_t>(key.chain_code ? key.chain_code->size() : 0u) * sizeof(uint32_t) +
           static_cast<uint64_t>(key.resources.size()) * sizeof(ShaderResourceCompileKey) +
           (key.has_pixel_inputs ? sizeof(PixelInputMapping) : 0u) +
           (key.has_system_inputs ? sizeof(PixelSystemInputMapping) : 0u) +
           static_cast<uint64_t>(spirv.size()) * sizeof(uint32_t);
}

struct PcrelDispatchSelection {
    PcrelDispatchInfo dispatch;
    const ShaderResource* resource = nullptr;
    uint32_t raw_selector = 0;
    uint32_t target = UINT32_MAX;
    bool readable = false;
};

PcrelDispatchSelection select_pcrel_dispatch(const uint32_t* code, size_t dwords,
                                              const ShaderResourceTable* resources,
                                              const ShaderCodeAnalysis* analysis = nullptr) {
    PcrelDispatchSelection selection;
    if (!code || !dwords || !resources) return selection;
    selection.dispatch = analysis ? analysis->pcrel_dispatch
                                  : rdna2_pcrel_dispatch_info(code, dwords);
    if (!selection.dispatch.valid) return selection;

    selection.resource = resources->by_sgpr_base_cls(
        selection.dispatch.selector_sgpr_base, ResourceClass::ConstantBuffer);
    if (!selection.resource ||
        selection.dispatch.selector_byte_offset > selection.resource->size ||
        selection.resource->size - selection.dispatch.selector_byte_offset <
            sizeof(selection.raw_selector)) return selection;

    if (selection.resource->host_data &&
        selection.dispatch.selector_byte_offset <= selection.resource->host_data_size &&
        selection.resource->host_data_size - selection.dispatch.selector_byte_offset >=
            sizeof(selection.raw_selector)) {
        memcpy(&selection.raw_selector,
               selection.resource->host_data + selection.dispatch.selector_byte_offset,
               sizeof(selection.raw_selector));
        selection.readable = true;
    } else if (selection.resource->gpu_addr <=
               UINT64_MAX - selection.dispatch.selector_byte_offset) {
        const uint64_t address =
            selection.resource->gpu_addr + selection.dispatch.selector_byte_offset;
        if (guest_readable(address, sizeof(selection.raw_selector))) {
            memcpy(&selection.raw_selector,
                   reinterpret_cast<const void*>(static_cast<uintptr_t>(address)),
                   sizeof(selection.raw_selector));
            selection.readable = true;
        }
    }
    if (!selection.readable) return selection;

    const uint32_t adjusted = selection.raw_selector +
                              static_cast<uint32_t>(selection.dispatch.selector_addend);
    const uint32_t index = std::min(adjusted, selection.dispatch.selector_max);
    if (index < selection.dispatch.target_pcs.size())
        selection.target = selection.dispatch.target_pcs[index];
    return selection;
}

ShaderCompileKey make_shader_compile_key(ShaderProgramStage stage, const uint32_t* code, size_t dwords,
                                         const ShaderResourceTable* resources,
                                         const PixelInputMapping* pixel_inputs,
                                         const PixelSystemInputMapping* system_inputs,
                                         const uint32_t* chain_code = nullptr,
                                         size_t chain_dwords = 0,
                                         uint32_t vertex_lds_dwords = 0,
                                         const ComputeShaderConfig* compute_config = nullptr,
                                         bool fragment_wave32 = false,
                                         bool capture_position = false) {
    ShaderCompileKey key;
    key.stage = stage;
    key.vertex_lds_dwords = stage == ShaderProgramStage::Vertex
        ? std::min(vertex_lds_dwords, 16384u) : 0u;
    key.vertices_per_instance = stage == ShaderProgramStage::Vertex && resources
        ? resources->vertices_per_instance : 0u;
    key.has_resource_table = resources != nullptr;
    key.force_position_w = getenv("PROSPER_FORCE_W") != nullptr;
    key.capture_position = stage == ShaderProgramStage::Vertex && capture_position;
    key.has_pixel_inputs = stage != ShaderProgramStage::Compute && pixel_inputs != nullptr;
    if (key.has_pixel_inputs) key.pixel_inputs = *pixel_inputs;
    key.has_system_inputs = stage == ShaderProgramStage::Fragment && system_inputs != nullptr;
    if (key.has_system_inputs) key.system_inputs = *system_inputs;
    key.fragment_wave32 = stage == ShaderProgramStage::Fragment && fragment_wave32;
    key.has_compute_config = stage == ShaderProgramStage::Compute && compute_config;
    if (key.has_compute_config) {
        key.compute_user_sgpr_count = static_cast<uint32_t>(compute_config->user_sgprs.size());
        key.compute_local_x = compute_config->local_x;
        key.compute_local_y = compute_config->local_y;
        key.compute_local_z = compute_config->local_z;
        key.compute_exact_thread_extent = compute_config->exact_thread_extent;
        key.compute_threads_x = compute_config->threads_x;
        key.compute_threads_y = compute_config->threads_y;
        key.compute_threads_z = compute_config->threads_z;
        key.compute_wave_size = compute_config->wave_size;
        key.compute_tidig_comp_cnt = compute_config->tidig_comp_cnt;
        key.compute_tgid_x_en = compute_config->tgid_x_en;
        key.compute_tgid_y_en = compute_config->tgid_y_en;
        key.compute_tgid_z_en = compute_config->tgid_z_en;
        key.compute_tg_size_en = compute_config->tg_size_en;
        key.compute_lds_bytes = compute_config->lds_bytes;
        key.compute_native_subgroup_size = compute_config->native_subgroup_size;
        key.compute_native_storage_format_support =
            compute_config->native_storage_format_support;
        key.compute_packed_r11_storage = compute_config->packed_r11_storage;
    }
    const std::shared_ptr<const ShaderCodeAnalysis> analysis =
        code && dwords ? analyze_shader_code_cached(code, dwords) : nullptr;
    if (stage == ShaderProgramStage::Fragment && code && dwords && resources) {
        const PcrelDispatchSelection selection =
            select_pcrel_dispatch(code, dwords, resources, analysis.get());
        const PcrelDispatchInfo& dispatch = selection.dispatch;
        if (dispatch.valid) {
            if (selection.target != UINT32_MAX) {
                key.has_pcrel_dispatch = true;
                key.pcrel_dispatch_target = selection.target;
            }
            if (getenv("PROSPER_DBG")) {
                static std::mutex dispatch_log_mutex;
                static std::set<uintptr_t> dispatch_logged;
                std::lock_guard lock(dispatch_log_mutex);
                if (dispatch_logged.insert(reinterpret_cast<uintptr_t>(code)).second) {
                    fprintf(stderr,
                            "[pcrel-dispatch] code=%p selector=s%u+0x%x resource=%s readable=%u "
                            "raw=%u target=%u resources=%zu\n",
                            static_cast<const void*>(code), dispatch.selector_sgpr_base,
                            dispatch.selector_byte_offset,
                            selection.resource ? "found" : "missing", selection.readable,
                            selection.raw_selector, key.pcrel_dispatch_target,
                            resources->resources.size());
                    for (const auto& candidate : resources->resources)
                        fprintf(stderr,
                                "[pcrel-dispatch]   cls=%u binding=%u sgpr=%u srt=%u addr=%llx "
                                "size=%u host=%llu\n",
                                static_cast<unsigned>(candidate.cls), candidate.binding,
                                candidate.sgpr_base, candidate.srt_offset,
                                static_cast<unsigned long long>(candidate.gpu_addr), candidate.size,
                                static_cast<unsigned long long>(candidate.host_data_size));
                }
            }
        }
    }
    if (analysis) {
        // Most shaders end at S_ENDPGM. A compiler-generated s_getpc_b64 V# may instead address an
        // embedded lookup table after ENDPGM; retain that proven tail so cached recompilation sees the
        // same blob as the direct path and table contents participate in the cache identity.
        key.code = std::shared_ptr<const std::vector<uint32_t>>(analysis, &analysis->code);
        key.code_hash = analysis->code_hash;
    }
    if (stage == ShaderProgramStage::Vertex && chain_code && chain_dwords) {
        const std::shared_ptr<const ShaderCodeAnalysis> chain_analysis =
            analyze_shader_code_cached(chain_code, chain_dwords);
        if (chain_analysis) {
            key.chain_code = std::shared_ptr<const std::vector<uint32_t>>(
                chain_analysis, &chain_analysis->code);
            key.chain_code_hash = chain_analysis->code_hash;
        }
    }
    if (resources) {
        key.resources.reserve(resources->resources.size());
        for (const auto& resource : resources->resources) {
            const bool texture = resource.cls == ResourceClass::Texture;
            const bool storage_image = resource.cls == ResourceClass::StorageImage;
            const bool manual_compare = texture && resource.depth_compare;
            const bool normalize_unnormalized = texture && resource.unnormalized &&
                !std::getenv("PROSPER_NO_UNNORMALIZED_COORD_NORMALIZE");
            const bool atomic_extent = storage_image &&
                resource.format == DataFormat::Uint32 && resource.num_components == 1;
            ShaderResourceCompileKey compiled;
            compiled.cls = static_cast<uint32_t>(resource.cls);
            compiled.format = static_cast<uint32_t>(resource.format);
            compiled.num_components = resource.num_components;
            compiled.width = (atomic_extent || normalize_unnormalized) ? resource.width : 0u;
            compiled.height = (atomic_extent || normalize_unnormalized) ? resource.height : 0u;
            compiled.depth = (storage_image || normalize_unnormalized) ? resource.depth : 0u;
            compiled.img_dim = (texture || storage_image) ? resource.img_dim : 0u;
            compiled.sample_count = (texture || storage_image) ? resource.sample_count : 1u;
            compiled.in_mip_tail = storage_image && resource.in_mip_tail;
            compiled.compression_enabled = storage_image && resource.compression_enabled;
            compiled.binding = resource.binding;
            compiled.stride = resource.stride;
            const bool one_record_tail = resource.num_components == 1u &&
                resource.stride == 2u && resource.size == 2u;
            compiled.one_record_tail_semantic = static_cast<uint32_t>(
                one_record_tail && resource.format == DataFormat::Uint16
                    ? StorageBufferTailSemantic::Uint16
                    : one_record_tail && resource.format == DataFormat::Float16
                        ? StorageBufferTailSemantic::Float16
                        : StorageBufferTailSemantic::None);
            compiled.srt_offset = resource.srt_offset;
            compiled.sgpr_base = resource.sgpr_base;
            compiled.fetch_pc = resource.fetch_pc;
            compiled.fetch_index_mode = static_cast<uint32_t>(resource.fetch_index_mode);
            compiled.flat_base_sgpr = resource.flat_base_sgpr;
            compiled.bvh_box_grow = resource.bvh_box_grow;
            compiled.null_bvh = is_proven_null_bvh(resource);
            compiled.srgb = storage_image && resource.srgb;
            compiled.depth_compare = (manual_compare || storage_image) && resource.depth_compare;
            compiled.depth_compare_func = manual_compare ? resource.depth_compare_func : 0u;
            compiled.mag_filter = manual_compare ? resource.mag_filter : 0u;
            compiled.addr_u = manual_compare ? resource.addr_uvw[0] : 0u;
            compiled.addr_v = manual_compare ? resource.addr_uvw[1] : 0u;
            compiled.border_color_type = manual_compare ? resource.border_color_type : 0u;
            compiled.normalize_unnormalized_coordinates = normalize_unnormalized;
            key.resources.push_back(compiled);
        }
    }
    key.cached_hash = ShaderCompileKeyHash::compute(key);
    return key;
}

std::vector<uint32_t> compile_graphics_shader(ShaderProgramStage stage, const ShaderCompileKey& key,
                                              const ShaderResourceTable* resources) {
    const uint32_t* code = !key.code || key.code->empty() ? nullptr : key.code->data();
    const size_t code_size = key.code ? key.code->size() : 0u;
    if (stage == ShaderProgramStage::Vertex)
        // Geometry probe: decorate gl_Position only when the caller proved the VS is the last
        // pre-rasterization stage. Generated interpolation geometry stages own XFB themselves.
        return key.chain_code
            ? recompile_vertex_chain(code, code_size, key.chain_code->data(),
                                     key.chain_code->size(), resources,
                                     key.has_pixel_inputs ? &key.pixel_inputs : nullptr,
                                     key.capture_position,
                                     key.vertex_lds_dwords)
            : recompile_vertex(code, code_size, resources,
                               key.has_pixel_inputs ? &key.pixel_inputs : nullptr,
                               key.capture_position,
                               key.vertex_lds_dwords);
    if (stage == ShaderProgramStage::Fragment) {
        const FragmentInterpolationLayout interpolation = fragment_interpolation_layout(
            code, code_size,
            key.has_system_inputs ? &key.system_inputs : nullptr,
            key.has_pixel_inputs ? &key.pixel_inputs : nullptr);
        return recompile_fragment(code, code_size, resources,
                                  key.has_system_inputs ? &key.system_inputs : nullptr,
                                  key.has_pcrel_dispatch ? key.pcrel_dispatch_target : UINT32_MAX,
                                  &interpolation, key.fragment_wave32);
    }
    return {};
}

void maybe_dump_successful_shader(ShaderProgramStage stage, const ShaderCompileKey& key,
                                  const std::vector<uint32_t>& spirv) {
    const char* directory = getenv("PROSPER_SHADER_DUMP_SUCCESS");
    if (!directory || !*directory || !key.code || key.code->empty() || spirv.empty()) return;

    const uint64_t spirv_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(spirv.data()), spirv.size() * sizeof(uint32_t));
    const uint64_t raw_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(key.code->data()), key.code->size() * sizeof(uint32_t));
    const uint64_t chain_hash = key.chain_code && !key.chain_code->empty()
        ? gpu_capture_hash(reinterpret_cast<const uint8_t*>(key.chain_code->data()),
                           key.chain_code->size() * sizeof(uint32_t))
        : 0;
    static std::mutex dump_mutex;
    static std::set<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>> dumped;
    std::lock_guard lock(dump_mutex);
    if (!dumped.emplace(static_cast<uint32_t>(stage), spirv_hash, raw_hash, chain_hash).second)
        return;

    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        fprintf(stderr, "[shader-dump] cannot create %s: %s\n", directory, ec.message().c_str());
        return;
    }
    const char* tag = stage == ShaderProgramStage::Vertex ? "vs" :
                      stage == ShaderProgramStage::Fragment ? "ps" : "cs";
    char raw_path[1024], chain_path[1024], spirv_path[1024];
    snprintf(raw_path, sizeof(raw_path), "%s/success_%s_%016llx_%016llx.bin", directory, tag,
             static_cast<unsigned long long>(spirv_hash),
             static_cast<unsigned long long>(raw_hash));
    snprintf(chain_path, sizeof(chain_path), "%s/success_%s_%016llx_%016llx_main_%016llx.bin",
             directory, tag, static_cast<unsigned long long>(spirv_hash),
             static_cast<unsigned long long>(raw_hash),
             static_cast<unsigned long long>(chain_hash));
    snprintf(spirv_path, sizeof(spirv_path), "%s/success_%s_%016llx_%016llx.spv", directory, tag,
             static_cast<unsigned long long>(spirv_hash),
             static_cast<unsigned long long>(raw_hash));
    FILE* raw = fopen(raw_path, "wb");
    FILE* chain = chain_hash ? fopen(chain_path, "wb") : nullptr;
    FILE* translated = fopen(spirv_path, "wb");
    const size_t raw_bytes = key.code->size() * sizeof(uint32_t);
    const size_t chain_bytes = chain_hash ? key.chain_code->size() * sizeof(uint32_t) : 0;
    const size_t spirv_bytes = spirv.size() * sizeof(uint32_t);
    const bool ok = raw && translated &&
        fwrite(key.code->data(), 1, raw_bytes, raw) == raw_bytes &&
        (!chain_hash || (chain && fwrite(key.chain_code->data(), 1, chain_bytes, chain) ==
                                     chain_bytes)) &&
        fwrite(spirv.data(), 1, spirv_bytes, translated) == spirv_bytes;
    if (raw) fclose(raw);
    if (chain) fclose(chain);
    if (translated) fclose(translated);
    fprintf(stderr,
            "[shader-dump] %s spv=%016llx raw=%016llx main=%016llx words=%zu+%zu/%zu "
            "result=%s\n", tag,
            static_cast<unsigned long long>(spirv_hash),
            static_cast<unsigned long long>(raw_hash),
            static_cast<unsigned long long>(chain_hash), key.code->size(),
            key.chain_code ? key.chain_code->size() : 0u, spirv.size(),
            ok ? "written" : "failed");
}

SharedShaderWords cache_compiled_graphics_shader(ShaderProgramStage stage, ShaderCompileKey key,
                                                  const ShaderResourceTable* resources,
                                                  uint64_t* cache_identity) {
    if (cache_identity) *cache_identity = 0;
    if (getenv("PROSPER_NO_SHADER_CACHE")) {
        auto& cache = shader_cache();
        {
            std::lock_guard lock(cache.mutex);
            ++cache.stats.bypasses;
        }
        auto spirv = std::make_shared<const std::vector<uint32_t>>(
            compile_graphics_shader(stage, key, resources));
        maybe_dump_successful_shader(stage, key, *spirv);
        return spirv;
    }

    auto& cache = shader_cache();
    std::lock_guard lock(cache.mutex);
    auto found = cache.entries.find(key);
    if (found != cache.entries.end()) {
        ++cache.stats.hits;
        found->second.last_use = ++cache.use_counter;
        if (cache_identity) *cache_identity = found->second.identity;
        maybe_dump_successful_shader(stage, key, *found->second.spirv);
        return found->second.spirv;
    }

    const auto start = std::chrono::steady_clock::now();
    auto spirv = std::make_shared<const std::vector<uint32_t>>(
        compile_graphics_shader(stage, key, resources));
    maybe_dump_successful_shader(stage, key, *spirv);
    const auto end = std::chrono::steady_clock::now();
    ++cache.stats.misses;
    cache.stats.compile_ms += std::chrono::duration<double, std::milli>(end - start).count();

    constexpr size_t max_entries = 4096;
    const uint64_t bytes = shader_cache_entry_bytes(key, *spirv);
    const uint64_t limit = shader_cache_limit_bytes();
    while (!cache.entries.empty() &&
           (cache.entries.size() >= max_entries || cache.stats.bytes + bytes > limit)) {
        auto oldest = cache.entries.begin();
        for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
            if (it->second.last_use < oldest->second.last_use) oldest = it;
        cache.stats.bytes -= oldest->second.bytes;
        cache.entries.erase(oldest);
        ++cache.stats.evictions;
    }
    if (bytes <= limit && max_entries != 0) {
        CachedShader value;
        value.spirv = spirv;
        value.identity = cache.next_identity++;
        value.last_use = ++cache.use_counter;
        value.bytes = bytes;
        if (cache_identity) *cache_identity = value.identity;
        cache.stats.bytes += bytes;
        cache.entries.emplace(std::move(key), std::move(value));
    }
    cache.stats.entries = cache.entries.size();
    return spirv;
}

} // namespace

ShaderDecodeCacheStats shader_decode_cache_stats() {
    auto& cache = shader_decode_cache();
    std::lock_guard lock(cache.mutex);
    ShaderDecodeCacheStats stats = cache.stats;
    stats.entries = cache.entries.size();
    return stats;
}

void clear_shader_decode_cache() {
    auto& cache = shader_decode_cache();
    std::lock_guard lock(cache.mutex);
    cache.entries.clear();
    cache.stats = {};
    cache.use_counter = 0;
}

ShaderAnalysisCacheStats shader_analysis_cache_stats() {
    auto& cache = shader_analysis_cache();
    std::lock_guard lock(cache.mutex);
    ShaderAnalysisCacheStats stats = cache.stats;
    stats.entries = cache.entries.size();
    return stats;
}

void clear_shader_analysis_cache() {
    {
        auto& cache = shader_analysis_cache();
        std::lock_guard lock(cache.mutex);
        cache.entries.clear();
        cache.stats = {};
        cache.use_counter = 0;
    }
    {
        auto& cache = interpolation_cache();
        std::lock_guard lock(cache.mutex);
        cache.entries.clear();
        cache.use_counter = 0;
    }
}

FragmentInterpolationLayout fragment_interpolation_layout_cached(
        const uint32_t* code, size_t dwords,
        const PixelSystemInputMapping* system_inputs,
        const PixelInputMapping* pixel_inputs) {
    const auto analysis = analyze_shader_code_cached(code, dwords);
    if (!analysis || getenv("PROSPER_NO_SHADER_ANALYSIS_CACHE"))
        return fragment_interpolation_layout(code, dwords, system_inputs, pixel_inputs);

    InterpolationCacheKey key;
    // Use the immutable analysis version, without retaining its shader-byte allocation beyond the
    // analysis cache's own memory bound. A same-address shader mutation receives a new identity.
    key.analysis_identity = analysis->identity;
    key.has_system_inputs = system_inputs != nullptr;
    if (system_inputs) key.system_inputs = *system_inputs;
    key.passthrough_mask = pixel_inputs ? pixel_inputs->effective_passthrough_mask() : 0u;
    auto& cache = interpolation_cache();
    std::lock_guard lock(cache.mutex);
    auto found = cache.entries.find(key);
    if (found != cache.entries.end()) {
        found->second.last_use = ++cache.use_counter;
        return found->second.layout;
    }
    const FragmentInterpolationLayout layout =
        fragment_interpolation_layout(code, dwords, system_inputs, pixel_inputs);
    constexpr size_t max_entries = 4096;
    while (cache.entries.size() >= max_entries && !cache.entries.empty()) {
        auto oldest = cache.entries.begin();
        for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
            if (it->second.last_use < oldest->second.last_use) oldest = it;
        cache.entries.erase(oldest);
    }
    cache.entries.emplace(std::move(key), CachedInterpolationLayout{layout, ++cache.use_counter});
    return layout;
}

SharedShaderWords recompile_graphics_shader_cached_shared(
        ShaderProgramStage stage, const uint32_t* code, size_t dwords,
        const ShaderResourceTable* resources, const PixelInputMapping* pixel_inputs,
        const PixelSystemInputMapping* system_inputs, uint64_t* cache_identity,
        bool fragment_wave32, uint32_t vertex_lds_dwords,
        bool vertex_capture_position) {
    ShaderCompileKey key = make_shader_compile_key(stage, code, dwords, resources, pixel_inputs,
                                                   system_inputs, nullptr, 0,
                                                   vertex_lds_dwords, nullptr,
                                                   fragment_wave32, vertex_capture_position);
    return cache_compiled_graphics_shader(stage, std::move(key), resources, cache_identity);
}

SharedShaderWords recompile_vertex_chain_cached_shared(
        const uint32_t* prolog, size_t prolog_dwords,
        const uint32_t* main, size_t main_dwords,
        const ShaderResourceTable* resources, const PixelInputMapping* pixel_inputs,
        uint64_t* cache_identity, uint32_t vertex_lds_dwords,
        bool capture_position) {
    ShaderCompileKey key = make_shader_compile_key(
        ShaderProgramStage::Vertex, prolog, prolog_dwords, resources, pixel_inputs, nullptr,
        main, main_dwords, vertex_lds_dwords, nullptr, false, capture_position);
    if (!key.chain_code) {
        if (cache_identity) *cache_identity = 0;
        return {};
    }
    return cache_compiled_graphics_shader(ShaderProgramStage::Vertex, std::move(key), resources,
                                          cache_identity);
}

std::vector<uint32_t> recompile_graphics_shader_cached(
        ShaderProgramStage stage, const uint32_t* code, size_t dwords,
        const ShaderResourceTable* resources, const PixelInputMapping* pixel_inputs,
        const PixelSystemInputMapping* system_inputs, uint64_t* cache_identity,
        bool fragment_wave32, uint32_t vertex_lds_dwords,
        bool vertex_capture_position) {
    SharedShaderWords words = recompile_graphics_shader_cached_shared(
        stage, code, dwords, resources, pixel_inputs, system_inputs, cache_identity,
        fragment_wave32, vertex_lds_dwords, vertex_capture_position);
    return words ? *words : std::vector<uint32_t>{};
}

std::vector<uint32_t> recompile_compute_shader_cached(
        const uint32_t* code, size_t dwords, const ShaderResourceTable* resources,
        const ComputeShaderConfig& config, uint64_t* cache_identity) {
    if (cache_identity) *cache_identity = 0;
    ShaderCompileKey key = make_shader_compile_key(
        ShaderProgramStage::Compute, code, dwords, resources, nullptr, nullptr,
        nullptr, 0, 0, &config);
    auto compile = [&] {
        const uint32_t* owned_code = !key.code || key.code->empty() ? nullptr : key.code->data();
        const size_t owned_dwords = key.code ? key.code->size() : 0u;
        return recompile_compute(owned_code, owned_dwords, resources, config);
    };
    if (getenv("PROSPER_NO_SHADER_CACHE")) {
        auto& cache = shader_cache();
        {
            std::lock_guard lock(cache.mutex);
            ++cache.stats.bypasses;
        }
        std::vector<uint32_t> spirv = compile();
        maybe_dump_successful_shader(ShaderProgramStage::Compute, key, spirv);
        return spirv;
    }

    auto& cache = shader_cache();
    std::lock_guard lock(cache.mutex);
    auto found = cache.entries.find(key);
    if (found != cache.entries.end()) {
        ++cache.stats.hits;
        found->second.last_use = ++cache.use_counter;
        if (cache_identity) *cache_identity = found->second.identity;
        maybe_dump_successful_shader(ShaderProgramStage::Compute, key, *found->second.spirv);
        return *found->second.spirv;
    }

    const auto start = std::chrono::steady_clock::now();
    auto spirv = std::make_shared<const std::vector<uint32_t>>(compile());
    maybe_dump_successful_shader(ShaderProgramStage::Compute, key, *spirv);
    const auto end = std::chrono::steady_clock::now();
    ++cache.stats.misses;
    cache.stats.compile_ms += std::chrono::duration<double, std::milli>(end - start).count();

    constexpr size_t max_entries = 4096;
    const uint64_t bytes = shader_cache_entry_bytes(key, *spirv);
    const uint64_t limit = shader_cache_limit_bytes();
    while (!cache.entries.empty() &&
           (cache.entries.size() >= max_entries || cache.stats.bytes + bytes > limit)) {
        auto oldest = cache.entries.begin();
        for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
            if (it->second.last_use < oldest->second.last_use) oldest = it;
        cache.stats.bytes -= oldest->second.bytes;
        cache.entries.erase(oldest);
        ++cache.stats.evictions;
    }
    if (bytes <= limit && max_entries != 0) {
        CachedShader value;
        value.spirv = spirv;
        value.identity = cache.next_identity++;
        value.last_use = ++cache.use_counter;
        value.bytes = bytes;
        if (cache_identity) *cache_identity = value.identity;
        cache.stats.bytes += bytes;
        cache.entries.emplace(std::move(key), std::move(value));
    }
    cache.stats.entries = cache.entries.size();
    return *spirv;
}

ShaderRecompileCacheStats shader_recompile_cache_stats() {
    auto& cache = shader_cache();
    std::lock_guard lock(cache.mutex);
    ShaderRecompileCacheStats result = cache.stats;
    result.entries = cache.entries.size();
    return result;
}

void clear_shader_recompile_cache() {
    auto& cache = shader_cache();
    std::lock_guard lock(cache.mutex);
    cache.entries.clear();
    cache.stats = {};
    cache.use_counter = 0;
}

thread_local DrawRealizationPhaseStats g_draw_realization_phases;

void record_draw_realization_phases(double table_ms, double shader_ms) {
    ++g_draw_realization_phases.draws;
    g_draw_realization_phases.table_ms += table_ms;
    g_draw_realization_phases.shader_ms += shader_ms;
}

DrawRealizationPhaseStats draw_realization_phase_stats() {
    return g_draw_realization_phases;
}

thread_local StageTablePhaseStats g_stage_table_phases;

void record_stage_table_phases(double metadata_ms, double dynamic_fold_ms, double resources_ms) {
    ++g_stage_table_phases.calls;
    g_stage_table_phases.metadata_ms += metadata_ms;
    g_stage_table_phases.dynamic_fold_ms += dynamic_fold_ms;
    g_stage_table_phases.resources_ms += resources_ms;
}

StageTablePhaseStats stage_table_phase_stats() {
    return g_stage_table_phases;
}

namespace {

struct GuestReadableCacheState {
    bool enabled = getenv("PROSPER_NO_GUEST_READ_CACHE") == nullptr;
    bool active = false;
    host::GuestReadableRangeCache persistent_ranges;
    host::GuestReadableRangeCache submit_ranges;
    uint64_t calls = 0, hits = 0, os_probes = 0;
};
thread_local GuestReadableCacheState g_guest_readable_cache;

bool guest_range_cache_hit(uint64_t begin, uint64_t end) {
    if (!g_guest_readable_cache.enabled) return false;
    if (g_guest_readable_cache.active) ++g_guest_readable_cache.calls;

    // Completion-label writes are folded on the guest draw thread before/after a renderer submit
    // scope. On Windows, falling through to VirtualQuery for every 8-byte label read made Astro's
    // otherwise headless frame loop take ~16 seconds per flip. The kernel-memory HLE already owns a
    // generation-guarded registry of fully committed readable guest mappings; reuse it here on every
    // call, not only while a renderer scope happens to be active. Sparse/lazy mappings are deliberately
    // absent from that registry and retain the OS probe/commit path below.
    g_guest_readable_cache.persistent_ranges.sync_generation(host::guest_mapping_generation());
    bool hit = g_guest_readable_cache.persistent_ranges.contains(begin, end) ||
               (g_guest_readable_cache.active &&
                g_guest_readable_cache.submit_ranges.contains(begin, end));
    if (!hit) {
        host::GuestReadableRange mapping{};
        if (host::guest_readable_mapping_containing(begin, end, mapping)) {
            g_guest_readable_cache.persistent_ranges.insert(mapping.begin, mapping.end);
            hit = true;
        }
    }
    if (hit && g_guest_readable_cache.active) ++g_guest_readable_cache.hits;
    return hit;
}

void cache_guest_readable_range(uint64_t begin, uint64_t end,
                                uint64_t query_begin, uint64_t query_end) {
    if (!g_guest_readable_cache.enabled || !g_guest_readable_cache.active || begin >= end) return;
    g_guest_readable_cache.submit_ranges.insert(begin, end);
    host::GuestReadableRange mapping{};
    if (host::guest_readable_mapping_containing(query_begin, query_end, mapping))
        g_guest_readable_cache.persistent_ranges.insert(mapping.begin, mapping.end);
}

struct GuestReadableSubmitScope {
    GuestReadableSubmitScope() {
        g_guest_readable_cache.active = g_guest_readable_cache.enabled;
        g_guest_readable_cache.calls = 0;
        g_guest_readable_cache.hits = 0;
        g_guest_readable_cache.os_probes = 0;
        g_guest_readable_cache.submit_ranges.clear();
        if (g_guest_readable_cache.active)
            g_guest_readable_cache.persistent_ranges.sync_generation(
                host::guest_mapping_generation());
    }
    ~GuestReadableSubmitScope() {
        g_guest_readable_cache.active = false;
    }
};

} // namespace

// Readability probe (guest memory is 1:1-mapped, but a mis-decoded address could be unmapped),
// so guarded derefs on the render/submit thread don't risk a SIGSEGV. NOTE: /dev/null does NOT
// work for this — the kernel's null_write returns count without ever touching the source buffer,
// so the old probe reported EVERY address >= 0x1000 "readable" and all guards built on it were
// no-ops (verified empirically on this project's WSL kernel). A pipe write actually imports the
// user pages and returns EFAULT for unmapped memory. Readability is page-granular: probe one byte
// in each page the range touches, draining after every write so the pipe can never fill. Windows
// uses VirtualQuery for the same guard. Dynamic fetch resolution now runs in the native
// live renderer too, so an always-true probe turns a guest null descriptor-table pointer into a
// host access violation (#688).
#ifndef _WIN32
static int g_probe_pipe[2] = {-1, -1};
// One-time pipe creation via a C++11 magic static (thread-safe). guest_readable is shared with
// the multi-threaded HLE pointer probes: an unguarded `if (fd < 0) pipe2(...)` lazy init let two
// first-callers each pipe2 into the array — a torn pair (write-end of pipe B, read-end of pipe A)
// never drains, fills, and EAGAINs: VALID memory reported unreadable, silently. (PR #61 review.)
static bool make_probe_pipe() {
#ifdef __linux__
    return pipe2(g_probe_pipe, O_CLOEXEC | O_NONBLOCK) == 0;
#else   // Darwin/BSD: no pipe2 — pipe + fcntl (still inside the once-only magic static, so no race)
    if (pipe(g_probe_pipe) != 0) return false;
    for (int fd : g_probe_pipe) {
        if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) return false;
        if (fcntl(fd, F_SETFL, O_NONBLOCK) != 0) return false;
    }
    return true;
#endif
}
static bool probe_pipe_ok() {
    static const bool ok = make_probe_pipe();
    return ok;
}
static bool probe_byte(uint64_t a) {
    ssize_t w = write(g_probe_pipe[1], (const void*)(uintptr_t)a, 1);
    if (w == 1) { char c; (void)!read(g_probe_pipe[0], &c, 1); return true; }
    return false;   // EFAULT: unmapped (EAGAIN can't happen — we drain after every byte)
}
bool guest_readable(uint64_t a, uint32_t n) {
    if (a < 0x1000 || n == 0) return false;
    if (a + n < a) return false;   // wrap
    const uint64_t end = a + n;
    if (guest_range_cache_hit(a, end)) return true;
    if (!probe_pipe_ok()) return false;
    uint64_t last_page = (a + n - 1) & ~0xfffull;
    for (uint64_t p = a & ~0xfffull; p <= last_page; p += 0x1000) {
        if (g_guest_readable_cache.active) ++g_guest_readable_cache.os_probes;
        if (!probe_byte(p < a ? a : p)) return false;
    }
    cache_guest_readable_range(a & ~0xfffull, last_page + 0x1000, a, end);
    return true;
}
#else
extern "C" int prosper_try_commit_dmem(uint64_t addr, uint64_t len, int write);

bool guest_readable(uint64_t a, uint32_t n) {
    if (a < 0x1000 || n == 0 || a + n < a) return false;
    const uint64_t end = a + n;
    if (guest_range_cache_hit(a, end)) return true;
    uint64_t cursor = a;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (g_guest_readable_cache.active) ++g_guest_readable_cache.os_probes;
        if (!VirtualQuery((const void*)(uintptr_t)cursor, &mbi, sizeof(mbi))) return false;
        const DWORD blocked = PAGE_NOACCESS | PAGE_GUARD;
        if (mbi.State != MEM_COMMIT) {
            if (!prosper_try_commit_dmem(cursor, end - cursor, 0)) return false;
            if (g_guest_readable_cache.active) ++g_guest_readable_cache.os_probes;
            if (!VirtualQuery((const void*)(uintptr_t)cursor, &mbi, sizeof(mbi))) return false;
        }
        if (mbi.State != MEM_COMMIT || (mbi.Protect & blocked)) return false;
        const DWORD readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                               PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
        if (!(mbi.Protect & readable)) return false;
        const uint64_t region_end = (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize;
        if (region_end <= cursor) return false;
        const uint64_t query_end = std::min(end, region_end);
        cache_guest_readable_range((uint64_t)(uintptr_t)mbi.BaseAddress, region_end,
                                   cursor, query_end);
        cursor = query_end;
    }
    return true;
}
#endif

bool guest_writable(uint64_t a, uint32_t n) {
    if (a < 0x1000 || n == 0 || a + n < a) return false;
    const uint64_t end = a + n;
#ifdef _WIN32
    uint64_t cursor = a;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery((const void*)(uintptr_t)cursor, &mbi, sizeof(mbi))) return false;
        if (mbi.State != MEM_COMMIT) {
            if (!prosper_try_commit_dmem(cursor, end - cursor, 1) ||
                !VirtualQuery((const void*)(uintptr_t)cursor, &mbi, sizeof(mbi))) return false;
        }
        const DWORD blocked = PAGE_NOACCESS | PAGE_GUARD;
        const DWORD writable = PAGE_READWRITE | PAGE_WRITECOPY |
                               PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
        if (mbi.State != MEM_COMMIT || (mbi.Protect & blocked) || !(mbi.Protect & writable))
            return false;
        const uint64_t region_end = (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize;
        if (region_end <= cursor) return false;
        cursor = std::min(end, region_end);
    }
    return true;
#elif defined(__APPLE__)
    uint64_t cursor = a;
    while (cursor < end) {
        mach_vm_address_t region = cursor;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info{};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        const kern_return_t result = mach_vm_region(
            mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64,
            reinterpret_cast<vm_region_info_t>(&info), &count, &object);
        if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
        if (result != KERN_SUCCESS || cursor < region || !(info.protection & VM_PROT_WRITE))
            return false;
        if (region > UINT64_MAX - size || region + size <= cursor) return false;
        cursor = std::min(end, static_cast<uint64_t>(region + size));
    }
    return true;
#else
    FILE* maps = fopen("/proc/self/maps", "re");
    if (!maps) return false;
    uint64_t cursor = a;
    char line[512];
    while (cursor < end && fgets(line, sizeof line, maps)) {
        unsigned long long begin = 0, finish = 0;
        char perms[5] = {};
        if (sscanf(line, "%llx-%llx %4s", &begin, &finish, perms) != 3 || finish <= cursor)
            continue;
        if (begin > cursor || perms[1] != 'w' || finish <= begin) {
            fclose(maps);
            return false;
        }
        cursor = std::min(end, static_cast<uint64_t>(finish));
    }
    fclose(maps);
    return cursor == end;
#endif
}

// Registered AGC headers publish the shader blob size in bytes. Dynamic descriptor folding used to
// ignore it and hand the decoder a fixed 0x4000-dword window, allowing the walk to read up to 64 KiB
// past a short shader. Retain that historical 64 KiB ceiling as a WORK bound too: CreateShader accepts
// guest metadata, so a corrupt multi-gigabyte shader_size must not become a page-probe/decode/OOM budget.
// Truncate a non-dword-aligned tail rather than reading one byte beyond the blob, and refuse the bounded
// span if it is not wholly readable. A zero result safely disables only the optional fold;
// metadata-described resources remain available to build_stage_table.
size_t dynamic_fold_shader_dwords(uint32_t shader_size_bytes) {
    constexpr size_t kMaxDynamicFoldDwords = 0x4000;
    return std::min<size_t>(shader_size_bytes / sizeof(uint32_t), kMaxDynamicFoldDwords);
}

size_t registered_shader_dwords(const AgcShaderHeader& header, uint64_t code_addr) {
    const size_t dwords = dynamic_fold_shader_dwords(header.shader_size);
    const uint32_t bounded_bytes = static_cast<uint32_t>(dwords * sizeof(uint32_t));
    if (!bounded_bytes || !guest_readable(code_addr, bounded_bytes)) return 0;
    return dwords;
}

// A null BVH marker is only safe when the static ray instruction is inside a real EXEC region. Prove
// the narrow compiler shape `EXEC writer; s_cbranch_execz MERGE; ... ray ...; MERGE`, and reject any
// external branch edge into the region. This is intentionally stronger than merely observing an
// earlier forward branch: the prior no-hit experiment did not establish dominance and could hide an
// unguarded missing descriptor.
bool guarded_bvh_use(const std::vector<Rdna2Inst>& instructions, uint32_t use_pc) {
    auto changes_exec = [](const Rdna2Inst& in) {
        if (in.fmt == Rdna2Format::VOPC)
            return (in.opcode >= 0x10 && in.opcode <= 0x1f) ||
                   (in.opcode >= 0x90 && in.opcode <= 0x9f) ||
                   (in.opcode >= 0xd0 && in.opcode <= 0xdf);
        return in.fmt == Rdna2Format::SOP1 &&
               ((in.opcode >= 0x24 && in.opcode <= 0x2b) ||
                in.opcode == 0x37 || in.opcode == 0x38 ||
                in.opcode == 0x3c || in.opcode == 0x40 || in.opcode == 0x44);
    };
    auto is_branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP &&
               (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09));
    };
    auto target = [](const Rdna2Inst& in) -> int64_t {
        return static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
    };

    for (size_t i = 1; i < instructions.size(); ++i) {
        const Rdna2Inst& guard = instructions[i];
        if (guard.fmt != Rdna2Format::SOPP || guard.opcode != 0x08 ||
            guard.simm16 <= 0 || guard.pc >= use_pc)
            continue;
        const int64_t merge64 = target(guard);
        if (merge64 <= static_cast<int64_t>(use_pc) || merge64 > UINT32_MAX)
            continue;
        const uint32_t merge = static_cast<uint32_t>(merge64);
        const Rdna2Inst& exec_writer = instructions[i - 1];
        if (exec_writer.pc + exec_writer.len_dwords != guard.pc ||
            !changes_exec(exec_writer))
            continue;
        if (std::none_of(instructions.begin(), instructions.end(),
                         [&](const Rdna2Inst& in) { return in.pc == merge; }))
            continue;

        bool external_entry = false;
        for (const Rdna2Inst& branch : instructions) {
            if (!is_branch(branch) || &branch == &guard) continue;
            const int64_t branch_target = target(branch);
            const bool target_inside = branch_target > static_cast<int64_t>(guard.pc) &&
                                       branch_target < static_cast<int64_t>(merge);
            const bool source_inside = branch.pc > guard.pc && branch.pc < merge;
            if (target_inside && !source_inside) { external_entry = true; break; }
        }
        if (!external_entry) return true;
    }
    return false;
}

// --- Bindless-dynamic vertex-fetch resolution (const-fold the scalar setup) ---------------------------
// This game's NGG vertex shader loads its vertex-buffer V# from a descriptor table at a RUNTIME-computed
// offset (e.g. `s_load_dwordx4 s[8:11], s[24:25], vcc_hi` where `vcc_hi = (s64<<4)&0x1f0` and
// `s64 = *[s26:27]`). The recompiler resolves descriptors by a STATIC provenance key, so it can't match a
// computed offset. We const-fold the wave-uniform scalar setup here: seed the concrete user-data SGPR
// values, interpret the scalar ALU + scalar loads (reading the 1:1-mapped guest memory), and snapshot the
// V# each descriptor load produces AT LOAD TIME (before the shader's later dynamic stride patch). The
// result maps each buffer_load_format's SRSRC SGPR -> its decoded V#, which build_stage_table emits as a
// VertexBuffer keyed by sgpr_base so the recompiler's by_sgpr_base() resolves it. Uniform-scalar only: any
// value that would depend on a VGPR/lane is left unknown (the op's dest becomes unknown), so we never
// fabricate a per-lane-dependent descriptor. CONFIDENCE: MED (covers this game's fetch-shader shape).
// External linkage (DynFetch + declaration in gpu_execute.hpp) so the fold is unit-testable.
std::vector<DynFetch>
resolve_dynamic_fetch(const uint32_t* code, size_t dwords, const uint32_t* user_sgprs, uint32_t nsgpr,
                      uint32_t user_sgpr_base, std::vector<SrtUse>* srt_uses,
                      uint32_t pcrel_dispatch_target,
                      const PcrelDispatchInfo* pcrel_dispatch,
                      const uint32_t* system_sgprs, uint32_t nsystem_sgprs) {
    using FoldClock = std::chrono::steady_clock;
    static const bool profile_fold = std::getenv("PROSPER_STAGE_FOLD_PROFILE") != nullptr;
    const auto fold_start = profile_fold ? FoldClock::now() : FoldClock::time_point{};
    const size_t srt_before = srt_uses ? srt_uses->size() : 0;
    uint64_t guest_probe_calls = 0;
    double guest_probe_ms = 0.0;
    std::vector<DynFetch> out;
    const auto decoded = decode_shader_cached(code, dwords);
    const auto decode_done = profile_fold ? FoldClock::now() : FoldClock::time_point{};
    std::vector<Rdna2Inst> specialized;
    const std::vector<Rdna2Inst>* fold_instructions = &decoded->instructions;
    if (pcrel_dispatch_target != UINT32_MAX) {
        specialized = decoded->instructions;
        const PcrelDispatchInfo dispatch = pcrel_dispatch
            ? *pcrel_dispatch : rdna2_pcrel_dispatch_info(code, dwords);
        if (!rdna2_specialize_pcrel_dispatch(specialized, dispatch,
                                             pcrel_dispatch_target)) {
            // The fragment recompiler will reject the same unprovable specialization. Do not walk all
            // alternatives here: doing so would fabricate resource provenance for code that cannot run.
            specialized.clear();
        }
        fold_instructions = &specialized;
    }
    // The cache specializes the full decoded stream before compacting it for the scalar fold. A
    // PC-relative dispatch is already a separate explicit specialization and keeps its historical
    // filtered stream here; combining the two requires proving the selected full-stream CFG first.
    if (fold_instructions == &decoded->instructions && decoded->shader_constant_specialized)
        fold_instructions = &decoded->shader_constant_instructions;
    const auto& ins = *fold_instructions;

    auto readable = [&](uint64_t addr, uint32_t bytes) {
        if (!profile_fold) return guest_readable(addr, bytes);
        const auto start = FoldClock::now();
        const bool result = guest_readable(addr, bytes);
        guest_probe_ms += std::chrono::duration<double, std::milli>(
            FoldClock::now() - start).count();
        ++guest_probe_calls;
        return result;
    };

    // PROSPER_DYNTRACE traces the whole const-fold walk; PROSPER_DYNTRACE_ADDR=<hex code addr>
    // narrows it to ONE shader (a full run otherwise traces every draw's walk — unusable volume).
    // g_dyntrace_force: set by the PROSPER_DYNTRACE_FAIL failure-replay path (gpu_execute.hpp) so
    // the walk of a shader that just FAILED to recompile is traced without knowing its address.
    bool trc = g_dyntrace_force || getenv("PROSPER_DYNTRACE") != nullptr;
    if (trc && !g_dyntrace_force)
        if (const char* fa = getenv("PROSPER_DYNTRACE_ADDR"))
            trc = strtoull(fa, nullptr, 16) == (uint64_t)(uintptr_t)code;
    // A full scalar-fold trace is intentionally verbose. Live shaders can rebuild their stage table
    // thousands of times per scene, so permit a targeted diagnostic run to capture the first matching
    // fold without turning every subsequent frame into gigabytes of duplicate logging.
    if (trc && !g_dyntrace_force && getenv("PROSPER_DYNTRACE_ONCE")) {
        static std::mutex once_mx;
        static std::set<const uint32_t*> traced;
        std::lock_guard<std::mutex> lk(once_mx);
        trc = traced.insert(code).second;
    }
    // Scalar operands encode at most 128 SGPRs. Fixed register files avoid hundreds of tiny hash/tree
    // allocations per submit while retaining exactly the same known/unknown state model.
    constexpr size_t kFoldSgprs = 128;
    std::array<uint32_t, kFoldSgprs> val{};                 // concrete SGPR values
    std::bitset<kFoldSgprs> val_known;
    // Entry-user-data identity attached to an otherwise ordinary scalar value. This is narrower
    // than descriptor provenance: it only proves that s_mov_b32/s_mov_b64 copies reassembled
    // consecutive entry dwords, without arithmetic or a load changing any word.
    std::array<uint32_t, kFoldSgprs> val_seed_origin{};
    std::bitset<kFoldSgprs> val_seed_origin_known;
    // Descriptor provenance attached to the CURRENT scalar value. Unlike the load-time snapshots
    // below, this follows s_mov shuffles and is cleared by arithmetic/data writes. A key-less value
    // uses exact consuming-pc provenance, matching the recompiler after scalar spills.
    std::array<uint32_t, kFoldSgprs> val_srt_key{};
    std::bitset<kFoldSgprs> val_srt_key_known;
    std::unordered_map<uint32_t, uint32_t> scalar_spill_slots; // (VGPR << 6) | lane -> scalar value
    std::array<std::array<uint32_t, 4>, kFoldSgprs> descr{}; // load-time V# snapshots by base SGPR
    std::bitset<kFoldSgprs> descr_known;
    // Descriptor-TABLE provenance (#294): for each snapshotted 4/8-dword s_load, the load's IMMEDIATE
    // byte offset — the recompiler's sreg_srt/by_srt_offset key. 0xFFFFFFFF = not provenance-usable
    // (register-SOFFSET or negative-immediate load, which emit_alu doesn't tag).
    std::array<uint32_t, kFoldSgprs> descr_key{};
    std::bitset<kFoldSgprs> descr_key_known;
    std::array<std::array<uint32_t, 8>, kFoldSgprs> descr8{};  // load-time T# snapshots by base SGPR
    std::bitset<kFoldSgprs> descr8_known;
    std::array<uint32_t, kFoldSgprs> descr8_key{};
    std::bitset<kFoldSgprs> descr8_key_known;
    // Exact null-pointer dataflow for guarded BVHs. Each mapped zero qword load receives a unique
    // origin; failed dereferences and scalar address/descriptor ALU retain that origin. A null BVH is
    // published only when all four live descriptor words carry the SAME origin, so an unrelated
    // unknown/zero SGPR cannot turn an unresolved ray instruction into a synthetic result.
    std::array<uint32_t, kFoldSgprs> null_chain_origin{};
    std::bitset<kFoldSgprs> null_chain_known;
    uint32_t next_null_chain_origin = 0;
    // SGPRs overwritten by an s_load since seeding — the seed-V# MUBUF fallback below must not use a
    // stale user-data snapshot once the register was RELOADED from memory (ALU patches deliberately
    // don't count: descriptor snapshots are load-time semantics, pre-patch, like `descr`).
    std::bitset<kFoldSgprs> reloaded;
    enum class FoldMask : uint8_t { Unknown, None, All };
    std::array<FoldMask, kFoldSgprs> mask_state{};
    constexpr size_t kFoldVgprs = 256;
    std::array<VertexFetchIndexMode, kFoldVgprs> vector_index_mode{};
    const bool explicit_ngg_index_provenance = user_sgpr_base == 8;
    if (explicit_ngg_index_provenance) {
        // GFX10 NGG merged VS/GS ABI: the VS inputs follow five GS VGPRs.
        vector_index_mode[5] = VertexFetchIndexMode::Vertex;
        vector_index_mode[8] = VertexFetchIndexMode::Instance;
    }
    int scc = -1;   // tracked SCC (-1 unknown): set by s_cmp_*, consumed by s_cselect (the format patch's tail)
    // The SPI loads the user-data block starting at shader SGPR `user_sgpr_base` (s0..s7 are NGG system
    // SGPRs). So user-data block index k lands in shader SGPR (user_sgpr_base + k).
    auto valid_reg = [](int r) { return r >= 0 && r < (int)kFoldSgprs; };
    auto set_value = [&](int r, uint32_t v) {
        if (valid_reg(r)) {
            val[(size_t)r] = v;
            val_known.set((size_t)r);
            val_srt_key_known.reset((size_t)r);
            val_seed_origin_known.reset((size_t)r);
            null_chain_known.reset((size_t)r);
            mask_state[(size_t)r] = FoldMask::Unknown;
        }
    };
    auto forget = [&](int r) {
        if (valid_reg(r)) {
            val_known.reset((size_t)r);
            val_srt_key_known.reset((size_t)r);
            val_seed_origin_known.reset((size_t)r);
            null_chain_known.reset((size_t)r);
            mask_state[(size_t)r] = FoldMask::Unknown;
        }
    };
    for (uint32_t i = 0; system_sgprs && i < nsystem_sgprs && i < kFoldSgprs; ++i)
        set_value(static_cast<int>(i), system_sgprs[i]);
    for (uint32_t i = 0; i < nsgpr; i++) {
        const int reg = (int)(user_sgpr_base + i);
        set_value(reg, user_sgprs[i]);
        if (valid_reg(reg)) {
            val_seed_origin[(size_t)reg] = i;
            val_seed_origin_known.set((size_t)reg);
        }
    }
    if (explicit_ngg_index_provenance) {
        // Match recompile_vertex's merged GS/ES ABI model: s3[7:0] is the active ES-vertex
        // count and s3[15:8] the GS-primitive count.  The Vulkan vertex shell represents one
        // active ES vertex and no GS primitive, so s3=1.  Fetch prologues use this value to
        // choose and patch V# descriptors before their MUBUF loads; leaving it unknown made the
        // dynamic resource walk drop otherwise valid scene-geometry fetches even though the
        // translator itself already compiled the same prologue with s3=1.
        set_value(3, 1u);
    }

    // A direct sharp lives in the initial user-data SGPR block rather than arriving through an
    // s_load. It remains a usable load-time descriptor only while none of its SGPRs has subsequently
    // been reloaded from memory. ALU descriptor patches deliberately retain the same pre-patch
    // snapshot semantics as `descr`/`descr8` and the seed-V# fallback below.
    auto untouched_seed_range = [&](int first, int count) {
        if (!user_sgprs || first < (int)user_sgpr_base ||
            first + count > (int)(user_sgpr_base + nsgpr)) return false;
        for (int r = first; r < first + count; r++)
            if (!valid_reg(r) || reloaded.test((size_t)r)) return false;
        return true;
    };

    auto known = [&](int r, uint32_t& v) {
        if (!valid_reg(r) || !val_known.test((size_t)r)) return false;
        v = val[(size_t)r];
        return true;
    };
    auto null_origin = [&](int r) -> uint32_t {
        return valid_reg(r) && null_chain_known.test((size_t)r)
            ? null_chain_origin[(size_t)r] : 0u;
    };
    auto mark_null_origin = [&](int r, uint32_t origin) {
        if (!origin || !valid_reg(r)) return;
        null_chain_origin[(size_t)r] = origin;
        null_chain_known.set((size_t)r);
    };
    auto operand_null_origin = [&](const Operand& operand) -> uint32_t {
        if ((operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
            valid_reg(operand.value))
            return null_origin(operand.value);
        return 0u;
    };
    auto unchanged_seed_range = [&](int first, int count) {
        if (!untouched_seed_range(first, count)) return false;
        for (int r = first; r < first + count; ++r) {
            uint32_t current = 0;
            if (!known(r, current) ||
                current != user_sgprs[r - (int)user_sgpr_base]) return false;
        }
        return true;
    };
    auto consecutive_seed_copy_range = [&](int first, int count) {
        if (!user_sgprs || count <= 0 || !valid_reg(first) || !valid_reg(first + count - 1) ||
            !val_seed_origin_known.test((size_t)first)) return false;
        const uint32_t first_origin = val_seed_origin[(size_t)first];
        if (first_origin + (uint32_t)count > nsgpr) return false;
        for (int k = 0; k < count; ++k) {
            const int reg = first + k;
            uint32_t current = 0;
            if (!val_seed_origin_known.test((size_t)reg) ||
                val_seed_origin[(size_t)reg] != first_origin + (uint32_t)k ||
                !known(reg, current) || current != user_sgprs[first_origin + (uint32_t)k])
                return false;
        }
        return true;
    };
    // Resolve an ALU source operand to a concrete value (SGPR / inline int / literal / a vcc Special).
    // vcc_lo/hi (106/107) are written by ALU dsts as SGPR 106/107 but read back as Special operands with
    // the same field value, so map them onto the same val[] keys. Other Specials (EXEC/M0/...) stay unknown.
    auto srcval = [&](const Operand& o, uint32_t& v) -> bool {
        switch (o.kind) {
            case OperandKind::SGPR:      return known(o.value, v);
            case OperandKind::InlineInt: v = (uint32_t)o.value; return true;
            case OperandKind::Special:   return (o.value == 106 || o.value == 107) ? known(o.value, v) : false;
            case OperandKind::Literal:   v = 0; return false;   // literal is in in.literal; handled per-op
            default: return false;
        }
    };
    auto srcmask = [&](const Operand& o) -> FoldMask {
        if (o.kind == OperandKind::Special && (o.value == 126 || o.value == 127))
            return FoldMask::All;       // EXEC is full at this fetch-prologue boundary
        if (o.kind == OperandKind::InlineInt)
            return o.value == 0 ? FoldMask::None
                 : o.value == -1 ? FoldMask::All : FoldMask::Unknown;
        if ((o.kind == OperandKind::SGPR || o.kind == OperandKind::Special) &&
            valid_reg(o.value))
            return mask_state[(size_t)o.value];
        return FoldMask::Unknown;
    };

    for (const auto& in : ins) {
        if (in.is_end) break;
        const bool scalar_spill = in.fmt == Rdna2Format::VOP3 &&
                                  (in.opcode == 0x360 || in.opcode == 0x361);
        const bool vector_select =
            (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x101) ||
            (in.fmt == Rdna2Format::VOP2 && in.opcode == 0x01);
        // Buffer instructions read VADDR before writing their payload destination. Some prologues
        // intentionally reuse the same VGPR for both (Messenger's v0 attribute fetch), so retain the
        // source provenance before the generic destination-write invalidation below.
        VertexFetchIndexMode fetch_index_mode_before_write = VertexFetchIndexMode::Automatic;
        if (explicit_ngg_index_provenance &&
            (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) &&
            in.src[0].kind == OperandKind::VGPR && in.src[0].value >= 0 &&
            in.src[0].value < static_cast<int>(kFoldVgprs))
            fetch_index_mode_before_write = vector_index_mode[(size_t)in.src[0].value];
        if (!scalar_spill && in.dst.kind == OperandKind::VGPR) {
            // Any ordinary write replaces the whole vector result. Drop scalar values previously
            // packed into that VGPR's lanes rather than restoring stale descriptor words later.
            for (uint32_t lane = 0; lane < 64; ++lane)
                scalar_spill_slots.erase(((uint32_t)in.dst.value << 6) | lane);
            // Address selectors are scalar VGPR values; ordinary VALU writes replace their exact
            // destination. Multi-dword memory results are payload here, not later ABI selectors.
            if (explicit_ngg_index_provenance && !vector_select && in.dst.value >= 0 &&
                in.dst.value < static_cast<int>(kFoldVgprs)) {
                vector_index_mode[(size_t)in.dst.value] = VertexFetchIndexMode::Shader;
            }
        }
        switch (in.fmt) {
            case Rdna2Format::SOP1:
                if (in.opcode == 0x03) {                        // s_mov_b32
                    uint32_t v, source_key = 0;
                    const uint32_t source_null_origin = operand_null_origin(in.src[0]);
                    const bool source_key_known =
                        in.src[0].kind == OperandKind::SGPR && valid_reg(in.src[0].value) &&
                        val_srt_key_known.test((size_t)in.src[0].value) &&
                        (source_key = val_srt_key[(size_t)in.src[0].value], true);
                    uint32_t source_origin = 0;
                    const bool source_origin_known =
                        in.src[0].kind == OperandKind::SGPR && valid_reg(in.src[0].value) &&
                        val_seed_origin_known.test((size_t)in.src[0].value) &&
                        (source_origin = val_seed_origin[(size_t)in.src[0].value], true);
                    if (in.src[0].kind == OperandKind::Literal ? (v = in.literal, true) : srcval(in.src[0], v)) {
                        set_value(in.dst.value, v);
                        if (source_key_known && valid_reg(in.dst.value)) {
                            val_srt_key[(size_t)in.dst.value] = source_key;
                            val_srt_key_known.set((size_t)in.dst.value);
                        }
                        if (source_origin_known && valid_reg(in.dst.value)) {
                            val_seed_origin[(size_t)in.dst.value] = source_origin;
                            val_seed_origin_known.set((size_t)in.dst.value);
                        }
                    } else forget(in.dst.value);
                    mark_null_origin(in.dst.value, source_null_origin);
                } else if (in.opcode == 0x34) {                 // s_abs_i32
                    // This is a 32-bit destination. Treating every unmodeled SOP1 as a possible
                    // 64-bit pair write erased the untouched adjacent SGPR; Astro Bot keeps the
                    // high half of a descriptor-table pointer there immediately before an x8 load.
                    // Compute abs in unsigned arithmetic so INT_MIN remains 0x80000000 without C++
                    // signed-overflow undefined behaviour. Arithmetic deliberately drops seed/SRT
                    // provenance through set_value().
                    uint32_t v = 0;
                    const bool source_known = in.src[0].kind == OperandKind::Literal
                        ? (v = in.literal, true) : srcval(in.src[0], v);
                    if (source_known)
                        set_value(in.dst.value, static_cast<int32_t>(v) < 0 ? 0u - v : v);
                    else
                        forget(in.dst.value);
                } else if ((in.opcode == 0x1c || in.opcode == 0x1d) &&
                           in.dst.kind == OperandKind::SGPR) {  // s_bitset{0,1}_b32
                    // These are in-place read/modify/write operations: SDST is both the old value
                    // and destination, while SRC0 is only the bit index. Astro sets the descriptor's
                    // SORT bit after extracting a zero BVH base, so preserve the exact null-root
                    // provenance while applying that constant field patch.
                    uint32_t old_value = 0, bit_index = 0;
                    const uint32_t old_null_origin = null_origin(in.dst.value);
                    const bool old_known = known(in.dst.value, old_value);
                    const bool bit_known = in.src[0].kind == OperandKind::Literal
                        ? (bit_index = in.literal, true) : srcval(in.src[0], bit_index);
                    if (old_known && bit_known) {
                        const uint32_t bit = 1u << (bit_index & 31u);
                        set_value(in.dst.value, in.opcode == 0x1c
                            ? old_value & ~bit : old_value | bit);
                    } else {
                        forget(in.dst.value);
                    }
                    mark_null_origin(in.dst.value, old_null_origin);
                } else if (in.opcode == 0x04 &&                 // s_mov_b64
                           in.dst.kind == OperandKind::SGPR &&
                           in.src[0].kind == OperandKind::SGPR) {
                    // Capture both source lanes before touching either destination: source and
                    // destination pairs may overlap. Only an entirely known SGPR pair is modeled;
                    // special/inline sources and partial pairs retain the previous fail-closed erase.
                    std::array<uint32_t, 2> source_values{};
                    std::array<uint32_t, 2> source_keys{};
                    std::array<uint32_t, 2> source_origins{};
                    std::array<uint32_t, 2> source_null_origins{};
                    std::array<bool, 2> source_key_known{};
                    std::array<bool, 2> source_origin_known{};
                    bool source_known = true;
                    for (int k = 0; k < 2; ++k) {
                        const int src = in.src[0].value + k;
                        source_known &= known(src, source_values[(size_t)k]);
                        source_key_known[(size_t)k] = valid_reg(src) &&
                            val_srt_key_known.test((size_t)src);
                        if (source_key_known[(size_t)k])
                            source_keys[(size_t)k] = val_srt_key[(size_t)src];
                        source_origin_known[(size_t)k] = valid_reg(src) &&
                            val_seed_origin_known.test((size_t)src);
                        if (source_origin_known[(size_t)k])
                            source_origins[(size_t)k] = val_seed_origin[(size_t)src];
                        source_null_origins[(size_t)k] = null_origin(src);
                    }
                    for (int k = 0; k < 2; ++k) {
                        const int dst = in.dst.value + k;
                        if (!source_known) {
                            forget(dst);
                            continue;
                        }
                        set_value(dst, source_values[(size_t)k]);
                        if (source_key_known[(size_t)k] && valid_reg(dst)) {
                            val_srt_key[(size_t)dst] = source_keys[(size_t)k];
                            val_srt_key_known.set((size_t)dst);
                        }
                        if (source_origin_known[(size_t)k] && valid_reg(dst)) {
                            val_seed_origin[(size_t)dst] = source_origins[(size_t)k];
                            val_seed_origin_known.set((size_t)dst);
                        }
                        mark_null_origin(dst, source_null_origins[(size_t)k]);
                    }
                } else if (in.dst.kind == OperandKind::SGPR) {
                    // Not a modeled scalar move -> the dest is unknown. Erase the PAIR: 64-bit SOP1 ops
                    // (s_getpc_b64, s_and/or/xor/not_b64, s_*_saveexec_b64, …) write
                    // S[dst:dst+1], so leaving a stale "known" val[dst+1] let a later instruction fold a
                    // confidently-wrong 64-bit base/offset -> a wrong V#/T# read from the wrong guest
                    // address (#460). Over-erasing dst+1 for a 32-bit SOP1 only loses a fold opportunity
                    // (never fabricates a value) — matching the SOP2 s_bfe_u64 pair-erase.
                    forget(in.dst.value);
                    forget(in.dst.value + 1);
                }
                // Several SOP1 ops write SCC (s_abs_i32, s_not_b32, s_and_saveexec_*, …). Only the moves
                // (s_mov_b32 0x03 / s_mov_b64 0x04) are known not to — anything else invalidates the
                // tracked SCC, or a later s_cselect folds with a stale compare result.
                if (in.opcode != 0x03 && in.opcode != 0x04) scc = -1;
                break;
            case Rdna2Format::SOP2: {
                // s_cselect_b64 is commonly used to make an all-lanes/zero mask for the following
                // NGG v_cndmask ABI selector. It does not produce ordinary scalar data, but SCC and
                // both source masks are wave-uniform here, so retain the exact All/None result.
                if (in.opcode == 0x0B) {
                    const FoldMask m0 = srcmask(in.src[0]);
                    const FoldMask m1 = srcmask(in.src[1]);
                    const FoldMask result = scc < 0 ? FoldMask::Unknown : (scc ? m0 : m1);
                    forget(in.dst.value);
                    forget(in.dst.value + 1);
                    if (valid_reg(in.dst.value)) mask_state[(size_t)in.dst.value] = result;
                    if (trc)
                        fprintf(stderr,
                                "[dyntrace]   SOP2 pc=%u s_cselect_b64 dst=s%d scc=%d mask=%u\n",
                                in.pc, in.dst.value, scc, (unsigned)result);
                    break;
                }
                uint32_t a, c; bool ka, kc;
                ka = (in.src[0].kind == OperandKind::Literal) ? (a = in.literal, true) : srcval(in.src[0], a);
                kc = (in.src[1].kind == OperandKind::Literal) ? (c = in.literal, true) : srcval(in.src[1], c);
                int d = in.dst.value; bool ok = ka && kc; uint32_t r = 0;
                uint32_t hi64 = 0; bool wrote_pair = false;
                int next_scc = scc;
                const uint32_t null0 = operand_null_origin(in.src[0]);
                const uint32_t null1 = operand_null_origin(in.src[1]);
                // s_addc_u32 remains derived from the same pointer chain even when its concrete
                // carry is unknown; the taint is provenance, not a claim about the folded value.
                const bool null_chain_opcode =
                    in.opcode <= 0x04 || in.opcode == 0x0E || in.opcode == 0x10 ||
                    in.opcode == 0x12 || in.opcode == 0x1E || in.opcode == 0x20 ||
                    in.opcode == 0x26 || (in.opcode >= 0x2E && in.opcode <= 0x31) ||
                    in.opcode == 0x27 || in.opcode == 0x1F || in.opcode == 0x21 ||
                    in.opcode == 0x29;
                uint32_t propagated_null_origin = 0;
                if (null_chain_opcode) {
                    if (null0 && null1 && null0 == null1) propagated_null_origin = null0;
                    else if (null0 && !null1 && kc) propagated_null_origin = null0;
                    else if (null1 && !null0 && ka) propagated_null_origin = null1;
                }
                if (ok) switch (in.opcode) {
                    case 0x00: {                                           // s_add_u32
                        const uint64_t sum = static_cast<uint64_t>(a) + c;
                        r = static_cast<uint32_t>(sum);
                        next_scc = static_cast<int>(sum >> 32);
                        break;
                    }
                    case 0x01:                                             // s_sub_u32
                        r = a - c; next_scc = a < c; break;
                    case 0x02: {                                           // s_add_i32
                        r = a + c;
                        next_scc = ((~(a ^ c) & (a ^ r)) >> 31) != 0;
                        break;
                    }
                    case 0x03:                                             // s_sub_i32
                        r = a - c; next_scc = (((a ^ c) & (a ^ r)) >> 31) != 0; break;
                    case 0x04: {                                           // s_addc_u32
                        if (scc < 0) { ok = false; break; }
                        const uint64_t sum = static_cast<uint64_t>(a) + c +
                                             static_cast<uint32_t>(scc);
                        r = static_cast<uint32_t>(sum);
                        next_scc = static_cast<int>(sum >> 32);
                        break;
                    }
                    case 0x0E: r = a & c; next_scc = r != 0; break;         // s_and_b32
                    case 0x10: r = a | c; next_scc = r != 0; break;         // s_or_b32
                    case 0x12: r = a ^ c; next_scc = r != 0; break;         // s_xor_b32
                    case 0x1E:                                             // s_lshl_b32
                        r = a << (c & 31); next_scc = r != 0; break;
                    case 0x20:                                             // s_lshr_b32
                        r = a >> (c & 31); next_scc = r != 0; break;
                    case 0x26: r = a * c; break;                            // s_mul_i32
                    case 0x2E: case 0x2F: case 0x30: case 0x31: {           // s_lshl{1,2,3,4}_add_u32
                        const uint32_t shift = in.opcode - 0x2Du;
                        const uint64_t sum = (static_cast<uint64_t>(a) << shift) + c;
                        r = static_cast<uint32_t>(sum);
                        next_scc = sum >= (uint64_t{1} << 32);
                        break;
                    }
                    case 0x27: { uint32_t off = c & 0x1f, wid = (c >> 16) & 0x7f;   // s_bfe_u32
                                 r = wid == 0 ? 0 : (wid >= 32 ? (a >> off) : ((a >> off) & ((1u << wid) - 1)));
                                 next_scc = r != 0; break; }
                    case 0x0A:   // s_cselect_b32: dst = SCC ? src0 : src1 (the vertex-fetch format patch's tail)
                        if (scc < 0) ok = false; else r = scc ? a : c;
                        break;
                    case 0x1F: case 0x21: {  // s_lshl_b64 / s_lshr_b64
                        uint32_t ahi = 0;
                        bool khi = false;
                        if (in.src[0].kind == OperandKind::SGPR)
                            khi = known(in.src[0].value + 1, ahi);
                        else if (in.src[0].kind == OperandKind::InlineInt) {
                            ahi = in.src[0].value < 0 ? UINT32_MAX : 0u;
                            khi = true;
                        } else if (in.src[0].kind == OperandKind::Literal) {
                            ahi = 0;
                            khi = true;
                        }
                        if (!khi) { ok = false; wrote_pair = true; break; }
                        const uint64_t src64 = static_cast<uint64_t>(a) |
                                               (static_cast<uint64_t>(ahi) << 32);
                        const uint32_t shift = c & 63u;
                        const uint64_t result = in.opcode == 0x1Fu
                            ? src64 << shift : src64 >> shift;
                        r = static_cast<uint32_t>(result);
                        hi64 = static_cast<uint32_t>(result >> 32);
                        wrote_pair = true;
                        next_scc = result != 0;
                        break;
                    }
                    case 0x29: {  // s_bfe_u64: dst[63:0] = bitfield of src0[63:0] (format patch reads a small field)
                        uint32_t off = c & 0x3f, wid = (c >> 16) & 0x7f, ahi = 0;
                        // src0's high dword (RDNA2 ISA 64-bit scalar-operand rules, #155): the next SGPR
                        // of the pair; SIGN-extension of an integer inline constant (-1..-16 read as a
                        // 64-bit operand are all-ones in the high dword); or 0 only for a 32-bit literal
                        // (zero-extended). Inline FLOAT constants read as 64-bit doubles (a different bit
                        // pattern entirely) — srcval() already leaves those unknown, so they never reach
                        // here. Only an SGPR operand may index val[value+1] — a literal's `value` is not
                        // an SGPR number. And an UNTRACKED high dword must not silently fold as 0: if the
                        // field reaches bits >= 32 the result is unknown.
                        bool khi;
                        if (in.src[0].kind == OperandKind::SGPR)           khi = known(in.src[0].value + 1, ahi);
                        else if (in.src[0].kind == OperandKind::InlineInt) { ahi = in.src[0].value < 0 ? 0xFFFFFFFFu : 0u; khi = true; }
                        else                                               { ahi = 0; khi = true; }   // 32-bit literal
                        if (!khi && wid != 0 && off + wid > 32) { ok = false; wrote_pair = true; break; }
                        uint64_t src64 = (uint64_t)a | ((uint64_t)ahi << 32);
                        uint64_t res = wid == 0 ? 0 : (wid >= 64 ? (src64 >> off) : ((src64 >> off) & (((uint64_t)1 << wid) - 1)));
                        r = (uint32_t)res; hi64 = (uint32_t)(res >> 32); wrote_pair = true;
                        next_scc = res != 0; break;
                    }
                    default: ok = false; break;                            // SCC-dependent / unmodeled -> unknown
                }
                if (trc)   // unfiltered like the SMEM/MUBUF traces (one shader walk — volume is bounded)
                    fprintf(stderr, "[dyntrace]   SOP2 pc=%u op=0x%x dst=s%d src0=%d(k%d) src1=%d(k%d) ok=%d r=0x%x\n",
                            in.pc, in.opcode, d, in.src[0].value, ka, in.src[1].value, kc, ok, r);
                scc = ok ? next_scc : -1;
                if (ok) { set_value(d, r); if (wrote_pair) set_value(d + 1, hi64); }
                // A 64-bit-dst op invalidates BOTH dwords even when its source was unknown (the
                // opcode switch may not have reached the point that marks wrote_pair).
                else    { forget(d); if (wrote_pair || in.opcode == 0x1F || in.opcode == 0x21 ||
                                         in.opcode == 0x29) forget(d + 1); }
                mark_null_origin(d, propagated_null_origin);
                if (wrote_pair || in.opcode == 0x1F || in.opcode == 0x21 || in.opcode == 0x29)
                    mark_null_origin(d + 1, propagated_null_origin);
                break;
            }
            case Rdna2Format::SOPC: {   // scalar compare -> SCC (feeds the format patch's s_cselect)
                uint32_t a, c;
                bool ka = (in.src[0].kind == OperandKind::Literal) ? (a = in.literal, true) : srcval(in.src[0], a);
                bool kc = (in.src[1].kind == OperandKind::Literal) ? (c = in.literal, true) : srcval(in.src[1], c);
                if (ka && kc) switch (in.opcode) {
                    case 0x00: scc = (a == c); break;                            // s_cmp_eq_i32
                    case 0x01: scc = (a != c); break;                            // s_cmp_lg_i32
                    case 0x02: scc = ((int32_t)a >  (int32_t)c); break;          // s_cmp_gt_i32
                    case 0x03: scc = ((int32_t)a >= (int32_t)c); break;          // s_cmp_ge_i32
                    case 0x04: scc = ((int32_t)a <  (int32_t)c); break;          // s_cmp_lt_i32
                    case 0x05: scc = ((int32_t)a <= (int32_t)c); break;          // s_cmp_le_i32
                    case 0x06: scc = (a == c); break;                            // s_cmp_eq_u32
                    case 0x07: scc = (a != c); break;                            // s_cmp_lg_u32
                    case 0x08: scc = (a >  c); break;                            // s_cmp_gt_u32
                    case 0x09: scc = (a >= c); break;                            // s_cmp_ge_u32
                    case 0x0A: scc = (a <  c); break;                            // s_cmp_lt_u32
                    case 0x0B: scc = (a <= c); break;                            // s_cmp_le_u32
                    default: scc = -1; break;
                } else scc = -1;
                break;
            }
            case Rdna2Format::SMEM: {
                // SBASE (src[0]) is a 2-dword pointer (s_load, op<8) or a 4-dword V# (s_buffer_load, op>=8).
                // Address = base + immediate OFFSET (in.literal) + SOFFSET register value (decoded here from
                // words[1][31:25]; the shared decoder doesn't expose it). Dword count from the opcode.
                uint32_t n = 0; bool is_buffer = in.opcode >= 8;
                switch (in.opcode & 7) { case 0: n = 1; break; case 1: n = 2; break; case 2: n = 4; break;
                                         case 3: n = 8; break; case 4: n = 16; break; default: n = 0; }
                int sbase = in.src[0].value, sdst = in.dst.value;
                const uint32_t null_base_lo = null_origin(sbase);
                const uint32_t null_base_hi = null_origin(sbase + 1);
                const uint32_t null_base_origin =
                    null_base_lo && null_base_lo == null_base_hi ? null_base_lo : 0u;
                uint32_t soff_field = (in.words[1] >> 25) & 0x7Fu;         // SOFFSET SGPR (125 = null)
                uint32_t soff_val = 0; bool soff_ok = true;
                if (soff_field < 106) soff_ok = known((int)soff_field, soff_val);          // SGPR
                else if (soff_field == 106 || soff_field == 107) soff_ok = known((int)soff_field, soff_val); // vcc lo/hi
                else if (soff_field == 125) soff_val = 0;                   // SGPR_NULL -> const-0 offset (ok)
                else soff_ok = false;                                       // m0(124)/exec(126,127)/reserved:
                                                                            // untracked -> mark UNKNOWN, not 0. The
                                                                            // old `else soff_val=0` claimed ok=true
                                                                            // for these, snapshotting a descriptor
                                                                            // from base+imm+0 instead of +m0/exec
                                                                            // -> wrong V#/T#, silently (#398).
                // Descriptor-table use (#294): an s_buffer_load's SBASE is a V# — if that V# was
                // snapshotted from a table load, report it as a ConstantBuffer use keyed by its load
                // immediate (matching the recompiler's sreg_srt tag). Recorded BEFORE the dest write
                // below (SBASE and SDST ranges may overlap).
                SrtUse pending_srt_use;
                bool have_pending_srt_use = false;
                if (srt_uses && is_buffer) {
                    bool current_known = true, current_key_known = true;
                    uint32_t current_key = 0;
                    for (int k = 0; k < 4; ++k) {
                        if (!valid_reg(sbase + k) || !val_known.test((size_t)(sbase + k)))
                            current_known = false;
                        if (!valid_reg(sbase + k) || !val_srt_key_known.test((size_t)(sbase + k))) {
                            current_key_known = false;
                        } else if (k == 0) {
                            current_key = val_srt_key[(size_t)sbase];
                        } else if (val_srt_key[(size_t)(sbase + k)] != current_key) {
                            current_key_known = false;
                        }
                    }
                    if (current_known) {
                        pending_srt_use.kind = 1;
                        // A V# may live directly in the entry user SGPRs without an AGC sharp or
                        // preceding s_load (Astro's title PS uses s[24:27] this way). Its four live
                        // dwords are still exact; only the table-offset provenance is absent. Publish
                        // that descriptor by the unambiguous consuming PC, matching direct MIMG/MUBUF
                        // discovery, instead of forcing the recompiler onto an unbound fallback cbuf.
                        pending_srt_use.key = current_key_known ? current_key : 0xFFFFFFFFu;
                        for (int k = 0; k < 4; ++k)
                            pending_srt_use.v4[(size_t)k] = val[(size_t)(sbase + k)];
                        pending_srt_use.use_pc = in.pc;
                        have_pending_srt_use = true;
                    } else if (valid_reg(sbase) && descr_known.test((size_t)sbase) &&
                               descr_key_known.test((size_t)sbase)) {
                        pending_srt_use.kind = 1;
                        pending_srt_use.key = descr_key[(size_t)sbase];
                        pending_srt_use.v4 = descr[(size_t)sbase];
                        pending_srt_use.use_pc = in.pc;
                        have_pending_srt_use = true;
                    }
                }
                for (uint32_t k = 0; k < n; k++)
                    if (valid_reg(sdst + (int)k)) reloaded.set((size_t)(sdst + (int)k));
                uint64_t base = 0; bool base_ok;
                if (is_buffer) { uint32_t b0, b1; base_ok = known(sbase, b0) && known(sbase + 1, b1);
                                 base = ((uint64_t)b0 | ((uint64_t)b1 << 32)) & 0xFFFFFFFFFFFFull; }   // V#.Base48
                else { uint32_t p0, p1; base_ok = known(sbase, p0) && known(sbase + 1, p1);
                       base = (uint64_t)p0 | ((uint64_t)p1 << 32); }        // raw pointer
                if (trc) fprintf(stderr, "[dyntrace] SMEM op=0x%x %s sdst=s%d sbase=s%d base=0x%llx base_ok=%d "
                                 "soff_field=%u soff_val=0x%x soff_ok=%d imm=0x%x n=%u\n", in.opcode,
                                 is_buffer ? "bufload" : "load", sdst, sbase, (unsigned long long)base, base_ok,
                                 soff_field, soff_val, soff_ok, in.literal, n);
                if (n == 0 || !base_ok || !soff_ok) {
                    // A wave-derived scalar SOFFSET (for example v_readfirstlane -> VCC_LO) is
                    // deliberately not concrete in this CPU-side fold. The V# can still be exact,
                    // though: publish its full bounded buffer by consuming-PC provenance and let
                    // the SPIR-V recompiler evaluate the dynamic dword index at runtime. The stage
                    // resource builder below accepts this zero-required-size use only when V# itself
                    // carries a conventional valid size, so an unknown offset never invents a range
                    // or gets folded to zero.
                    if (have_pending_srt_use && n != 0 && base_ok && !soff_ok)
                        srt_uses->push_back(pending_srt_use);
                    for (uint32_t k = 0; k < n; k++) {
                        forget(sdst + (int)k);
                        mark_null_origin(sdst + (int)k, null_base_origin);
                    }
                    break;
                }
                // in.literal is the SIGN-EXTENDED 21-bit immediate (#149) — add it as signed so a
                // negative offset subtracts from the base instead of wrapping to a huge address.
                const int64_t byte_off = (int64_t)(int32_t)in.literal + (int64_t)soff_val;
                uint64_t addr = (base + (uint64_t)byte_off) & ~3ull;
                if (have_pending_srt_use) {
                    const int64_t required = byte_off + (int64_t)n * 4;
                    if (required > 0 && required <= (int64_t)UINT32_MAX) {
                        pending_srt_use.required_size = (uint32_t)required;
                        srt_uses->push_back(pending_srt_use);
                    }
                }
                // AGC scalar-pointer user data can carry aperture/tag bits above the title's usable
                // GPU VA. Real GFX10 S_LOAD addresses are canonicalized by the memory system; a raw
                // host dereference is not. Prefer the exact 64-bit address, then (only when it is
                // unreadable) try the architectural Base48 and the PS5 process' exercised 40-bit VA
                // aperture. Every candidate must be mapped for the complete load, so this never turns
                // an unreadable pointer into an unchecked dereference. The 4-byte alignment matches
                // scalar-memory dword addressing (SharpEmu's evaluator applies the same alignment).
                bool addr_readable = is_buffer ? false : readable(addr, n * 4);
                if (!is_buffer && !addr_readable) {
                    for (uint64_t mask : {0xFFFFFFFFFFFFull, 0xFFFFFFFFFFull}) {
                        const uint64_t candidate = (((base & mask) + (uint64_t)byte_off) & ~3ull);
                        if (candidate != addr && readable(candidate, n * 4)) {
                            if (trc) fprintf(stderr, "[dyntrace]   canonical S_LOAD addr 0x%llx -> 0x%llx (mask=0x%llx)\n",
                                             (unsigned long long)addr, (unsigned long long)candidate,
                                             (unsigned long long)mask);
                            addr = candidate;
                            addr_readable = true;
                            break;
                        }
                    }
                }
                if (is_buffer) addr_readable = readable(addr, n * 4);
                if (!addr_readable) { if (trc) fprintf(stderr, "[dyntrace]   addr 0x%llx unreadable\n", (unsigned long long)addr);
                                      for (uint32_t k = 0; k < n; k++) {
                                          forget(sdst + (int)k);
                                          mark_null_origin(sdst + (int)k, null_base_origin);
                                      }
                                      break; }
                if (trc && writer_provenance_enabled()) {
                    const auto writer = last_guest_write_overlap(addr, n * 4);
                    if (writer) {
                        fprintf(stderr,
                                "[dyntrace]   latest GPU writer kind=%s seq=%llu "
                                "range=[0x%llx,+0x%llx) submit=%llu item=%llu order=%llu "
                                "identity=0x%llx\n",
                                guest_writer_kind_name(writer->kind),
                                (unsigned long long)writer->sequence,
                                (unsigned long long)writer->addr,
                                (unsigned long long)writer->size,
                                (unsigned long long)writer->submit,
                                (unsigned long long)writer->item,
                                (unsigned long long)writer->order,
                                (unsigned long long)writer->identity);
                    } else {
                        fprintf(stderr,
                                "[dyntrace]   no recorded GPU writer overlaps [0x%llx,+0x%x)\n",
                                (unsigned long long)addr, n * 4);
                    }
                }
                const uint32_t* mem = (const uint32_t*)(uintptr_t)addr;
                const bool imm_only = (soff_field == 125) && (int32_t)in.literal >= 0;   // SGPR_NULL soffset
                for (uint32_t k = 0; k < n; k++) set_value(sdst + (int)k, mem[k]);
                if (!is_buffer && n == 2 && mem[0] == 0 && mem[1] == 0) {
                    uint32_t origin = ++next_null_chain_origin;
                    if (!origin) origin = ++next_null_chain_origin;
                    mark_null_origin(sdst, origin);
                    mark_null_origin(sdst + 1, origin);
                }
                if ((n == 4 || n == 8) && valid_reg(sdst) && valid_reg(sdst + (int)n - 1)) {
                    const uint32_t key = (imm_only && !is_buffer) ? in.literal : 0xFFFFFFFFu;
                    for (uint32_t k = 0; k < n; ++k) {
                        val_srt_key[(size_t)(sdst + (int)k)] = key;
                        val_srt_key_known.set((size_t)(sdst + (int)k));
                    }
                }
                // Provenance key: the recompiler tags an IMMEDIATE-only descriptor load's dest SGPRs
                // with the load immediate (sreg_srt = in.literal); register-SOFFSET / negative loads
                // are not tagged, so mark those snapshots key-less.
                // A 4-dword load is a V# candidate — snapshot it now (before any later stride patch) so a
                // vertex fetch using these SGPRs resolves to the descriptor as loaded. An 8-dword load is
                // a T# candidate (image_sample SRSRC), snapshotted the same way (#294).
                if (n == 4 && valid_reg(sdst)) {
                              descr[(size_t)sdst] = { mem[0], mem[1], mem[2], mem[3] };
                              descr_known.set((size_t)sdst);
                              // only s_load (not s_buffer_load) dests get the recompiler's sreg_srt tag
                              descr_key[(size_t)sdst] = (imm_only && !is_buffer)
                                  ? in.literal : 0xFFFFFFFFu;
                              descr_key_known.set((size_t)sdst);
                }
                if (n == 8 && valid_reg(sdst)) {
                              descr8[(size_t)sdst] = {
                                  mem[0], mem[1], mem[2], mem[3], mem[4], mem[5], mem[6], mem[7] };
                              descr8_known.set((size_t)sdst);
                              descr8_key[(size_t)sdst] = (imm_only && !is_buffer)
                                  ? in.literal : 0xFFFFFFFFu;
                              descr8_key_known.set((size_t)sdst);
                              // SGPR loads are typeless: a later scalar buffer load may consume the
                              // first four words of this eight-dword result as a V#. Keep both views;
                              // only an actual buffer consumer reports the V# candidate.
                              descr[(size_t)sdst] = { mem[0], mem[1], mem[2], mem[3] };
                              descr_known.set((size_t)sdst);
                              descr_key[(size_t)sdst] = descr8_key[(size_t)sdst];
                              descr_key_known.set((size_t)sdst);
                }
                if (n == 16) {
                    // NGG back halves commonly fetch a compact stage-data block containing a mix of
                    // one 8-dword T# and 4-dword V# descriptors with one s_load_dwordx16. SGPR loads
                    // are typeless, so retain both aligned interpretations and let the eventual MIMG
                    // or MUBUF consumer select the appropriate one. The load's one byte-offset key
                    // cannot distinguish the resources inside the block, so retain key-less
                    // descriptor provenance; each consumer is resolved by its exact instruction pc.
                    for (uint32_t first = 0; first < n; first += 8) {
                        const int base_reg = sdst + static_cast<int>(first);
                        if (!valid_reg(base_reg) || !valid_reg(base_reg + 7)) continue;
                        descr8[(size_t)base_reg] = {
                            mem[first], mem[first + 1], mem[first + 2], mem[first + 3],
                            mem[first + 4], mem[first + 5], mem[first + 6], mem[first + 7] };
                        descr8_known.set((size_t)base_reg);
                        descr8_key[(size_t)base_reg] = 0xFFFFFFFFu;
                        descr8_key_known.set((size_t)base_reg);
                    }
                    for (uint32_t first = 0; first < n; first += 4) {
                        const int base_reg = sdst + static_cast<int>(first);
                        if (!valid_reg(base_reg) || !valid_reg(base_reg + 3)) continue;
                        descr[(size_t)base_reg] = {
                            mem[first], mem[first + 1], mem[first + 2], mem[first + 3] };
                        descr_known.set((size_t)base_reg);
                        descr_key[(size_t)base_reg] = 0xFFFFFFFFu;
                        descr_key_known.set((size_t)base_reg);
                    }
                }
                break;
            }
            case Rdna2Format::MIMG: {
                // IMAGE_BVH_INTERSECT_RAY consumes a four-dword BVH descriptor, not an eight-dword
                // image T#. Preserve the words live at this exact instruction and materialize the
                // acceleration-structure bytes as a raw read-only SSBO for software lowering.
                if (in.opcode == 0xE6u) {
                    if (srt_uses) {
                        const int bbase = in.src[1].value;
                        std::array<uint32_t, 4> live_bvh{};
                        bool live_known = valid_reg(bbase) && valid_reg(bbase + 3);
                        for (int k = 0; live_known && k < 4; ++k)
                            live_known &= known(bbase + k, live_bvh[(size_t)k]);
                        const bool snapshot_provenance = valid_reg(bbase) &&
                            descr_known.test((size_t)bbase);
                        const bool seed_provenance = !snapshot_provenance &&
                            (untouched_seed_range(bbase, 4) ||
                             consecutive_seed_copy_range(bbase, 4));
                        const DecodedBvhDescriptor d = live_known
                            ? decode_bvh_descriptor(live_bvh.data()) : DecodedBvhDescriptor{};
                        const bool plausible = live_known && (snapshot_provenance || seed_provenance) &&
                            d.type == 8u && d.base > 0x10000u && d.size_bytes != 0;
                        uint32_t proven_null_origin = valid_reg(bbase) ? null_origin(bbase) : 0u;
                        for (int k = 1; proven_null_origin && k < 4; ++k)
                            if (null_origin(bbase + k) != proven_null_origin)
                                proven_null_origin = 0;
                        const bool guarded_null = !plausible && proven_null_origin &&
                            guarded_bvh_use(ins, in.pc);
                        if (trc) {
                            fprintf(stderr,
                                    "[dyntrace] BVH pc=%u srsrc=s%d snapshot=%d seed=%d "
                                    "live_known=%d bvh=",
                                    in.pc, bbase, snapshot_provenance, seed_provenance, live_known);
                            if (live_known) {
                                for (uint32_t word : live_bvh) fprintf(stderr, "%08x ", word);
                            } else {
                                fprintf(stderr, "<unknown> ");
                            }
                            if (plausible) {
                                fprintf(stderr, "-> base=0x%llx size=0x%llx type=%u tri_mode=%u grow=%u",
                                        (unsigned long long)d.base,
                                        (unsigned long long)d.size_bytes, d.type,
                                        d.triangle_return_mode, d.box_grow);
                            } else if (guarded_null) {
                                fprintf(stderr, "<guarded-null origin=%u>", proven_null_origin);
                            } else if (live_known) {
                                fprintf(stderr, "<insufficient-provenance>");
                            }
                            fputc('\n', stderr);
                        }
                        if (plausible) {
                            SrtUse u;
                            u.kind = 2;
                            u.key = 0xFFFFFFFFu;
                            u.bvh4 = live_bvh;
                            u.use_pc = in.pc;
                            srt_uses->push_back(u);
                        } else if (guarded_null) {
                            SrtUse u;
                            u.kind = 3;
                            u.key = 0xFFFFFFFFu;
                            u.use_pc = in.pc;
                            srt_uses->push_back(u);
                        }
                    }
                    break;
                }
                // Descriptor-table use (#294): an image op's SRSRC (src[1]) is an 8-dword T#; if it was
                // snapshotted from a table load, report it as a Texture use — with the paired SSAMP
                // (src[2]) S# when that 4-dword load also resolved. VGPR-only dest: no SGPR state.
                // Key-less snapshots (register-SOFFSET loads) are reported too (#273): the use carries
                // its instruction pc, which the recompiler resolves via ShaderResource::fetch_pc when
                // the immediate-key model fails or collides.
                if (srt_uses) {
                    const int tbase = in.src[1].value;
                    const int samp_base = in.src[2].value;
                    const bool have_t8 = valid_reg(tbase) && descr8_known.test((size_t)tbase);
                    const bool have_key = valid_reg(tbase) && descr8_key_known.test((size_t)tbase);
                    // MIMG itself proves that its eight SRSRC words are a T#. A table snapshot or
                    // direct user-data range establishes provenance, but the RESOURCE MUST use the
                    // words live at this instruction: modeled scalar patches are architectural, and
                    // an unmodeled write makes one word unknown and therefore rejects. Falling back
                    // to descr8's load-time bytes after either case would bind a stale texture.
                    std::array<uint32_t, 8> live_t8{};
                    bool live_t8_known = true;
                    for (int k = 0; k < 8; ++k)
                        live_t8_known &= known(tbase + k, live_t8[(size_t)k]);
                    const bool seed_provenance = !have_t8 &&
                        (untouched_seed_range(tbase, 8) ||
                         consecutive_seed_copy_range(tbase, 8));
                    bool plausible_seed = true;
                    if (seed_provenance && live_t8_known) {
                        const DecodedImageDescriptor d = decode_image_descriptor(live_t8.data());
                        plausible_seed = d.base > 0x10000 && d.width && d.height &&
                            d.width <= 16384 && d.height <= 16384 &&
                            d.type >= 8 && d.type <= 15;
                    }
                    const std::array<uint32_t, 8>* t8 =
                        live_t8_known && (have_t8 || (seed_provenance && plausible_seed))
                            ? &live_t8 : nullptr;
                    uint32_t tkey = 0xFFFFFFFFu;
                    if (t8 && have_key) {
                        uint32_t common_key = 0;
                        bool common_key_known = true;
                        for (int k = 0; k < 8; ++k) {
                            const size_t reg = (size_t)(tbase + k);
                            if (!valid_reg(tbase + k) || !val_srt_key_known.test(reg)) {
                                common_key_known = false;
                                break;
                            }
                            if (k == 0) common_key = val_srt_key[reg];
                            else if (val_srt_key[reg] != common_key) {
                                common_key_known = false;
                                break;
                            }
                        }
                        if (common_key_known) tkey = common_key;
                    }
                    const bool from_seed = t8 && seed_provenance;
                    if (trc) {
                        fprintf(stderr, "[dyntrace] MIMG pc=%u op=0x%x srsrc=s%d ssamp=s%d "
                                        "have_t8=%d seed_t8=%d key=0x%x t8=",
                                in.pc, in.opcode, tbase, samp_base, have_t8, from_seed, tkey);
                        if (t8) {
                            for (uint32_t word : *t8) fprintf(stderr, "%08x ", word);
                            const DecodedImageDescriptor td = decode_image_descriptor(t8->data());
                            fprintf(stderr,
                                    "-> base=0x%llx %ux%ux%u type=%u fmt=%u tile=%u mip=%u:%u",
                                    (unsigned long long)td.base, td.width, td.height, td.depth,
                                    td.type, td.format, td.tile_mode, td.base_level, td.max_mip);
                        } else {
                            fprintf(stderr, "<unknown>");
                        }
                        fputc('\n', stderr);
                    }
                    if (t8) {
                        SrtUse u; u.kind = 0; u.t8 = *t8;
                        u.key = tkey;
                        u.use_pc = in.pc;
                        u.is_storage_image = in.opcode == 0x08 || in.opcode == 0x0f ||
                                             in.opcode == 0x11;
                        u.is_depth_compare = (in.opcode >= 0x28 && in.opcode <= 0x2f) ||
                                             (in.opcode >= 0x38 && in.opcode <= 0x3f) ||
                                             (in.opcode >= 0x58 && in.opcode <= 0x5f);
                        const bool have_s4 = valid_reg(samp_base) &&
                                             descr_known.test((size_t)samp_base);
                        const bool seed_s4 = !have_s4 && untouched_seed_range(samp_base, 4);
                        bool live_s4_known = true;
                        for (int k = 0; k < 4; ++k)
                            live_s4_known &= known(samp_base + k, u.s4[(size_t)k]);
                        if (live_s4_known && (have_s4 || seed_s4)) {
                            // Like the T#, sampler words are read live. This avoids retaining a stale
                            // x16 load snapshot if scalar code patches or invalidates the paired S#.
                            u.has_samp = true;
                        }
                        srt_uses->push_back(u);
                    }
                }
                break;
            }
            case Rdna2Format::MUBUF:
            case Rdna2Format::MTBUF: {
                const bool is_mtbuf = in.fmt == Rdna2Format::MTBUF;
                // Buffer stores, raw loads/stores, and supported atomics need a kind-1 resource use.
                // Format loads are intentionally handled only by DynFetch below: it snapshots the V#
                // live at the instruction and resolves by exact pc, avoiding a duplicate stale SRT use.
                const bool raw_buffer_use = !is_mtbuf &&
                    ((in.opcode >= 0x08 && in.opcode <= 0x0F) ||
                     (in.opcode >= 0x1C && in.opcode <= 0x1F));
                const bool format_store_use = in.opcode >= 0x04 && in.opcode <= 0x07;
                const bool atomic_buffer_use = in.opcode == 0x38; // buffer_atomic_umax
                if (srt_uses && (format_store_use || raw_buffer_use || atomic_buffer_use)) {
                    const int srsrc = in.src[1].value;
                    std::array<uint32_t, 4> current{};
                    bool current_known = true;
                    for (int k = 0; k < 4; ++k)
                        current_known &= known(srsrc + k, current[(size_t)k]);
                    const bool loaded_provenance = valid_reg(srsrc) &&
                                                   descr_known.test((size_t)srsrc);
                    if (current_known) {
                        // MUBUF/MTBUF itself is definitive that its four SRSRC words are a V#. Publish
                        // the values LIVE at the consumer whenever the scalar fold knows all of them.
                        // This also covers a direct descriptor whose NUM_RECORDS/stride is patched by
                        // modeled scalar ALU: such a patch intentionally breaks entry-seed provenance,
                        // but does not make the now-concrete V# any less valid. An unmodeled write makes
                        // a word unknown and still fails closed; the shape/range checks below reject
                        // malformed concrete values before any resource is emitted.
                        SrtUse u; u.kind = 1; u.v4 = current; u.key = 0xFFFFFFFFu; u.use_pc = in.pc;
                        if (is_mtbuf) u.instruction_format = in.mtbuf_format;
                        if (loaded_provenance) {
                            uint32_t common_key = 0;
                            bool have_common_key = true;
                            for (int k = 0; k < 4; ++k) {
                                const size_t r = (size_t)(srsrc + k);
                                if (!valid_reg(srsrc + k) || !val_srt_key_known.test(r)) {
                                    have_common_key = false;
                                    break;
                                }
                                if (k == 0) common_key = val_srt_key[r];
                                else if (val_srt_key[r] != common_key) {
                                    have_common_key = false;
                                    break;
                                }
                            }
                            // Rewrites clear the recompiler's complete sreg_srt tag, so the live V#
                            // must then resolve through this consuming instruction's exact pc.
                            if (have_common_key) u.key = common_key;
                        }
                        DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
                        DataFormat inst_format = DataFormat::Unknown;
                        uint32_t inst_components = 0;
                        if (is_mtbuf)
                            rdna2_buffer_format(in.mtbuf_format, &inst_format, &inst_components);
                        // FORMAT=INVALID is the architectural unbound-resource marker. MTBUF's
                        // instruction format controls conversion, but does not turn an unbound V#
                        // into a valid resource.
                        const bool descriptor_bound = !is_mtbuf ||
                            (((u.v4[3] >> 12) & 0x7Fu) != 0);
                        const bool format_supported = !format_store_use ||
                            (is_mtbuf ? (descriptor_bound && inst_format != DataFormat::Unknown &&
                                         inst_components != 0)
                                      : (d.format != DataFormat::Unknown && d.num_components != 0 &&
                                         !d.forbid_unknown_fallback));
                        // Byte-addressed raw/atomic V#s validly use stride zero: NUM_RECORDS is bytes.
                        // Typed format stores retain the strided record requirement.
                        const bool stride_supported = !format_store_use || d.stride != 0;
                        if (d.base > 0x10000 && d.size_bytes != 0 &&
                            d.size_bytes <= 0x10000000u && stride_supported && format_supported) {
                            if (trc) fprintf(stderr,
                                             "[dyntrace] MUBUF pc=%u live V# s%d base=0x%llx "
                                             "stride=%u size=%u\n",
                                             in.pc, srsrc, (unsigned long long)d.base,
                                             d.stride, d.size_bytes);
                            srt_uses->push_back(u);
                        }
                    }
                }
                // buffer_load_format_* (vertex fetch): opcodes 0..3. Resolve the SRSRC (src[1]) SGPR to the
                // V# most-recently loaded into it.
                if (in.opcode <= 3) {
                    int srsrc = in.src[1].value;
                    // Prefer the FETCH-TIME V#: the fetch shader patches the descriptor's format field (v[3])
                    // between load and fetch — so read the CURRENT SGPR values (which the interpreter has
                    // tracked through the patch, incl. the s_cselect tail) to get the real data format (e.g.
                    // UNORM8 for a packed vertex color, vs the load-time Unknown). Fall back to the load-time
                    // snapshot if the patched dwords aren't fully known.
                    uint32_t vv[4]; bool k0 = known(srsrc, vv[0]), k1 = known(srsrc + 1, vv[1]),
                                         k2 = known(srsrc + 2, vv[2]), k3 = known(srsrc + 3, vv[3]);
                    bool patched = k0 && k1 && k2 && k3;
                    const bool have_descr = valid_reg(srsrc) && descr_known.test((size_t)srsrc);
                    // Fold the fetch's CONSTANT byte offset into the emitted V# base (#273 item 1, the
                    // "solid banner" bug): the recompiler's per-fetch (by_fetch_pc) address model is
                    // exactly gl_VertexIndex*stride from the resolved base — it assumes the attribute's
                    // in-record byte offset is already IN the base. Unity-style fetch shaders satisfy
                    // that by patching each attribute's V# base; DOLL's UE4 Slate VS instead uses ONE
                    // un-patched V# and carries each attribute's offset in the MUBUF SOFFSET register
                    // (+ the 12-bit inst offset). Without the fold, all four Slate attributes (pos, uv,
                    // material-uv, color) read the position bytes -> the loading-banner widget rendered
                    // as a solid bar. The walk knows the SOFFSET value (it computed it from the attr-spec
                    // words), so add soffset+inst_offset to the descriptor base; an UNKNOWN soffset keeps
                    // the un-offset base (previous behavior). CONFIDENCE: HIGH (fetch-time values traced
                    // live; Messenger's fetches carry SOFFSET=0 so they are byte-identical).
                    uint32_t soff = 0; bool soff_known = true;
                    if (in.src[2].kind == OperandKind::Special && in.src[2].value == 125)     // SGPR_NULL -> 0
                        soff = 0;
                    else if (!srcval(in.src[2], soff))
                        soff_known = false;                          // real but untracked SOFFSET (#398) — see below
                    const uint32_t inst_off = in.literal & 0xFFFu;
                    const uint32_t fetch_off = soff + inst_off;
                    auto with_off = [&](DecodedBufferDescriptor d) {
                        d.base += fetch_off;
                        // size_bytes stays num_records*stride: the hardware bound is INDEX < num_records
                        // (record granularity), so from the offset base the last record's attribute still
                        // lies within (num_records-1)*stride + fetch_off + attr bytes — trimming the size
                        // by fetch_off cut the LAST vertex's attribute off the upload (guarded reads made
                        // it zeros -> a collapsed final vertex).
                        return d;
                    };
                    auto append_fetch = [&](DecodedBufferDescriptor d, uint32_t desc_v3,
                                            bool from_seed = false) {
                        if (is_mtbuf && ((desc_v3 >> 12) & 0x7Fu) == 0) {
                            if (trc) fprintf(stderr,
                                "[dyntrace]   MTBUF pc=%u unbound V# v3=0x%x -> unresolved\n",
                                in.pc, desc_v3);
                            return;
                        }
                        // decode_buffer_descriptor deliberately rejects packed formats whose selector
                        // or conversion semantics are unsupported. Do not let build_stage_table's
                        // legacy Unknown->Float32 fallback resurrect that descriptor as four raw dwords.
                        if (!is_mtbuf && d.forbid_unknown_fallback) {
                            if (trc) fprintf(stderr,
                                "[dyntrace]   MUBUF pc=%u packed V# v3=0x%x unsupported -> unresolved\n",
                                in.pc, desc_v3);
                            return;
                        }
                        DynFetch fetch{ in.pc, srsrc, with_off(d), desc_v3 };
                        fetch.from_seed = from_seed;
                        // A legal instruction/SOFFSET transform changes the normalized base without
                        // changing the four raw seed dwords.  That transformed resource no longer has
                        // byte-identical descriptor identity, so it cannot name the entry V# as a
                        // direct raw witness (doing so would manufacture a mismatch in #1853).
                        if (fetch_off == 0 && consecutive_seed_copy_range(srsrc, 4))
                            fetch.direct_user_data_index =
                                val_seed_origin[static_cast<size_t>(srsrc)];
                        fetch.unshifted_desc = d;
                        if (is_mtbuf) fetch.instruction_format = in.mtbuf_format;
                        fetch.index_mode = fetch_index_mode_before_write;
                        if (trc)
                            fprintf(stderr,
                                    "[dyntrace]   fetch pc=%u -> base=0x%llx stride=%u "
                                    "num_records=%u size=%u fmt=%u nc=%u index=%u\n",
                                    in.pc, (unsigned long long)fetch.desc.base, fetch.desc.stride,
                                    fetch.desc.num_records, fetch.desc.size_bytes,
                                    (unsigned)fetch.desc.format, fetch.desc.num_components,
                                    (unsigned)fetch.index_mode);
                        out.push_back(fetch);
                    };
                    if (trc) fprintf(stderr, "[dyntrace] MUBUF fetch pc=%u op=0x%x SRSRC=s%d patched=%d (k=%d%d%d%d v3=0x%x) have_descr=%d off=+0x%x soff_known=%d\n",
                                     in.pc, in.opcode, srsrc, patched, k0, k1, k2, k3,
                                     k3 ? vv[3] : 0, have_descr, fetch_off, (int)soff_known);
                    // A real (non-NULL) SOFFSET the fold cannot resolve would silently collapse fetch_off's
                    // in-record component to 0 — every attribute reads base+inst_off (the "solid banner"
                    // collapse this fold was written to fix) or a wrong descriptor address. Leave the fetch
                    // UNRESOLVED (a loud recompile-coverage miss) rather than fabricating offset 0 (#398).
                    if (!soff_known) {
                        if (trc) fprintf(stderr, "[dyntrace]   MUBUF pc=%u SOFFSET untracked -> fetch left unresolved (not folded to 0)\n", in.pc);
                        break;
                    }
                    // Per-fetch: record THIS fetch's live V# (the SRSRC SGPR is reloaded with a different V#
                    // per vertex attribute — position, uv, color…). Keyed by the fetch's pc so the recompiler
                    // resolves each buffer_load_format to the descriptor as loaded at that instruction.
                    if (patched) {
                        append_fetch(decode_buffer_descriptor(vv), vv[3]);
                    } else if (have_descr) {
                        append_fetch(decode_buffer_descriptor(descr[(size_t)srsrc].data()),
                                     descr[(size_t)srsrc][3]);
                    }
                    else if (srsrc >= (int)user_sgpr_base && srsrc + 4 <= (int)(user_sgpr_base + nsgpr) &&
                             valid_reg(srsrc + 3) && !reloaded.test((size_t)srsrc) &&
                             !reloaded.test((size_t)(srsrc + 1)) &&
                             !reloaded.test((size_t)(srsrc + 2)) &&
                             !reloaded.test((size_t)(srsrc + 3))) {
                        // SEED fallback (#294): the SRSRC V# was placed directly in the user-data SGPRs
                        // by the driver (never s_loaded — so no `descr` snapshot) and the shader's
                        // stride/format patch left the CURRENT dwords partially unknown (its s_cselect
                        // condition reads an NGG system SGPR we don't model). Use the SEED values — the
                        // same load-time/pre-patch semantics as the `descr` fallback above. Refused if
                        // any of the 4 SGPRs was RELOADED from memory since seeding (a stale seed then
                        // no longer describes the register). DOLL's scene-geometry VS fetches resolve
                        // through exactly this path. CONFIDENCE: MED (patch-ignoring, like `descr`).
                        const uint32_t sv[4] = { user_sgprs[srsrc - (int)user_sgpr_base],
                                                 user_sgprs[srsrc - (int)user_sgpr_base + 1],
                                                 user_sgprs[srsrc - (int)user_sgpr_base + 2],
                                                 user_sgprs[srsrc - (int)user_sgpr_base + 3] };
                        DecodedBufferDescriptor d = decode_buffer_descriptor(sv);
                        // Plausibility: only emit a real-looking V# (mirrors the direct-resource guard).
                        if (d.base > 0x10000 && d.size_bytes != 0 && d.size_bytes <= 0x10000000u &&
                            (is_mtbuf || !d.forbid_unknown_fallback)) {
                            if (trc) fprintf(stderr, "[dyntrace]   MUBUF pc=%u seed-V# fallback SRSRC=s%d base=0x%llx\n",
                                             in.pc, srsrc, (unsigned long long)d.base);
                            append_fetch(d, sv[3], /*from_seed=*/true);
                        }
                    }
                }
                break;
            }
            case Rdna2Format::VOP2:
                // The compact e32 v_cndmask form uses VCC implicitly. Unity's NGG fetch prologue
                // reissues this selector before each attribute load; treating the write as ordinary
                // shader arithmetic loses the vertex-id proof after the first position fetch and
                // collapses Evergate's later UV/color attributes.
                if (explicit_ngg_index_provenance && in.opcode == 0x01 &&
                    in.dst.kind == OperandKind::VGPR) {
                    const FoldMask mask = mask_state[106]; // implicit VCC_LO/HI pair
                    const Operand& selected = mask == FoldMask::All ? in.src[1] : in.src[0];
                    VertexFetchIndexMode mode = VertexFetchIndexMode::Shader;
                    if (mask != FoldMask::Unknown && selected.kind == OperandKind::VGPR &&
                        selected.value >= 0 && selected.value < static_cast<int>(kFoldVgprs))
                        mode = vector_index_mode[(size_t)selected.value];
                    if (in.dst.value >= 0 && in.dst.value < static_cast<int>(kFoldVgprs))
                        vector_index_mode[(size_t)in.dst.value] = mode;
                    if (trc)
                        fprintf(stderr,
                                "[dyntrace]   VOP2 pc=%u cndmask v%d mask=%u selected=v%d index=%u\n",
                                in.pc, in.dst.value, (unsigned)mask, selected.value, (unsigned)mode);
                }
                break;
            case Rdna2Format::VOP3:
                if (explicit_ngg_index_provenance && in.opcode == 0x101 &&
                    in.dst.kind == OperandKind::VGPR) {
                    const FoldMask mask = srcmask(in.src[2]);
                    const Operand& selected = mask == FoldMask::All ? in.src[1] : in.src[0];
                    VertexFetchIndexMode mode = VertexFetchIndexMode::Shader;
                    if (mask != FoldMask::Unknown && selected.kind == OperandKind::VGPR &&
                        selected.value >= 0 && selected.value < static_cast<int>(kFoldVgprs))
                        mode = vector_index_mode[(size_t)selected.value];
                    if (in.dst.value >= 0 && in.dst.value < static_cast<int>(kFoldVgprs))
                        vector_index_mode[(size_t)in.dst.value] = mode;
                    if (trc)
                        fprintf(stderr,
                                "[dyntrace]   VOP3 pc=%u cndmask v%d mask=%u selected=v%d index=%u\n",
                                in.pc, in.dst.value, (unsigned)mask, selected.value, (unsigned)mode);
                }
                // Scalar spill slots used by large UE shaders: v_writelane_b32 packs wave-uniform
                // SGPR values into fixed VGPR lanes, then v_readlane_b32 restores them later. The
                // recompiler deliberately drops SRT tags on restore, so recovered descriptors use
                // key-less, exact-pc provenance.
                if ((in.opcode == 0x361 || in.opcode == 0x360) &&
                    in.src[1].kind == OperandKind::InlineInt &&
                    in.src[1].value >= 0 && in.src[1].value < 64) {
                    const uint32_t slot =
                        ((uint32_t)(in.opcode == 0x361 ? in.dst.value : in.src[0].value) << 6) |
                        (uint32_t)in.src[1].value;
                    if (in.opcode == 0x361) {                         // v_writelane_b32 vDST, sSRC, lane
                        uint32_t v;
                        if (srcval(in.src[0], v)) scalar_spill_slots[slot] = v;
                        else scalar_spill_slots.erase(slot);
                    } else {                                          // v_readlane_b32 sDST, vSRC, lane
                        auto it = scalar_spill_slots.find(slot);
                        if (it == scalar_spill_slots.end()) {
                            forget(in.dst.value);
                        } else {
                            set_value(in.dst.value, it->second);
                            if (valid_reg(in.dst.value)) {
                                val_srt_key[(size_t)in.dst.value] = 0xFFFFFFFFu;
                                val_srt_key_known.set((size_t)in.dst.value);
                            }
                        }
                    }
                }
                break;
            case Rdna2Format::SOPK:
                // s_cmpk_* / s_addk_i32 write SCC (only s_movk/s_version/s_cmovk/s_mulk don't); this
                // interpreter doesn't model SOPK, so ANY SOPK conservatively invalidates the tracked SCC —
                // a stale SCC consumed by a later s_cselect would fabricate a confidently-wrong V# patch.
                scc = -1;
                if (in.dst.kind == OperandKind::SGPR) forget(in.dst.value);
                break;
            default:
                // Remaining formats (SOPP, VALU, memory, …) don't write SCC, so the tracked SCC survives.
                if (in.dst.kind == OperandKind::SGPR) forget(in.dst.value);   // unmodeled scalar write -> unknown
                break;
        }
    }
    if (profile_fold)
        record_stage_fold_profile(
            (uint64_t)(uintptr_t)code, user_sgpr_base, decoded->code.size(), ins.size(), out.size(),
            srt_uses ? srt_uses->size() - srt_before : 0, guest_probe_calls,
            std::chrono::duration<double, std::milli>(FoldClock::now() - fold_start).count(),
            std::chrono::duration<double, std::milli>(decode_done - fold_start).count(),
            guest_probe_ms);
    return out;
}

// Bound the common compiler-generated linear clear kernel without trusting an oversized/formatless
// V#. Its first instruction forms GlobalInvocationId.x from TGID.x and TID.x, then an idxen-only
// single-component store writes literal zero at that descriptor record. Since every supported numeric
// format represents zero identically, Uint32 is a lossless backing view even when FORMAT is INVALID.
// This recognizes the complete four-instruction kernel, not a general "large buffer" fallback: any
// other value, address shape, or side effect stays rejected.
static uint32_t linear_dispatch_raw_store_size(const uint32_t* code, size_t dwords,
                                               uint32_t fetch_pc,
                                               const DecodedBufferDescriptor& descriptor,
                                               uint32_t local_x, uint32_t threads_x,
                                               uint32_t tgid_x_sgpr) {
    if (!code || !dwords || !local_x || !threads_x || tgid_x_sgpr == UINT32_MAX ||
        !descriptor.stride || (local_x & (local_x - 1u)) != 0)
        return 0;
    uint32_t local_shift = 0;
    for (uint32_t width = local_x; width > 1; width >>= 1) ++local_shift;

    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    if (instructions.size() != 4) return 0;
    const Rdna2Inst* index_writer = &instructions[0];
    const Rdna2Inst* zero_writer = &instructions[1];
    const Rdna2Inst* access = &instructions[2];
    const Rdna2Inst* end = &instructions[3];
    if (access->pc != fetch_pc || index_writer->pc != 0 || zero_writer->pc != 2 ||
        end->pc != 5 || end->fmt != Rdna2Format::SOPP || end->opcode != 0x001 ||
        access->fmt != Rdna2Format::MUBUF || access->opcode != 0x004 ||
        access->dst.kind != OperandKind::VGPR || access->dst.value != 1 ||
        access->src[0].kind != OperandKind::VGPR || access->src[0].value != 0 ||
        access->src[2].kind != OperandKind::InlineInt || access->src[2].value != 0 ||
        (access->literal & 0x3FFFu) != 0x2000u ||
        zero_writer->fmt != Rdna2Format::VOP1 || zero_writer->opcode != 0x001 ||
        zero_writer->dst.kind != OperandKind::VGPR || zero_writer->dst.value != 1 ||
        zero_writer->src[0].kind != OperandKind::InlineInt || zero_writer->src[0].value != 0 ||
        index_writer->fmt != Rdna2Format::VOP3 || index_writer->opcode != 0x346 ||
        index_writer->dst.kind != OperandKind::VGPR || index_writer->dst.value != 0 ||
        index_writer->src[0].kind != OperandKind::SGPR ||
        static_cast<uint32_t>(index_writer->src[0].value) != tgid_x_sgpr ||
        index_writer->src[1].kind != OperandKind::InlineInt ||
        static_cast<uint32_t>(index_writer->src[1].value) != local_shift ||
        index_writer->src[2].kind != OperandKind::VGPR || index_writer->src[2].value != 0)
        return 0;

    const uint64_t required =
        static_cast<uint64_t>(threads_x - 1u) * descriptor.stride + sizeof(uint32_t);
    constexpr uint32_t kMaxProvenLinearStore = 16u << 20;
    if (!required || required > descriptor.size_bytes || required > kMaxProvenLinearStore)
        return 0;
    return static_cast<uint32_t>(required);
}

std::vector<SrtUse> add_compute_buffer_resources(ShaderResourceTable& table,
                                                 const uint32_t* code, size_t dwords,
                                                 const uint32_t* user_sgprs, uint32_t nsgpr,
                                                 uint32_t linear_local_x,
                                                 uint32_t linear_threads_x,
                                                 uint32_t tgid_x_sgpr) {
    std::vector<SrtUse> srt_uses;
    const std::vector<DynFetch> direct_fetches = resolve_dynamic_fetch(
        code, dwords, user_sgprs, nsgpr, /*user_sgpr_base*/0, &srt_uses);

    // A format-load resource has one identity: the descriptor live at its exact instruction pc.
    // Keep its original base because the compute ConstantBuffer address path applies the MUBUF
    // OFFSET/SOFFSET itself; `fetch.desc` is shifted for graphics' special vertex-index path.
    for (const auto& fetch : direct_fetches) {
        const DecodedBufferDescriptor& d = fetch.unshifted_desc;
        if (fetch.instruction_format != UINT32_MAX &&
            ((fetch.desc_v3 >> 12) & 0x7Fu) == 0)
            continue;
        DataFormat format = d.format;
        uint32_t components = d.num_components;
        if (fetch.instruction_format != UINT32_MAX)
            rdna2_buffer_format(fetch.instruction_format, &format, &components);
        const uint32_t resource_size = d.size_bytes;
        if (d.base <= 0x10000 || resource_size == 0 || resource_size > 0x10000000u ||
            format == DataFormat::Unknown || !components ||
            (fetch.instruction_format == UINT32_MAX && d.forbid_unknown_fallback))
            continue;
        bool mapped = false;
        for (auto& r0 : table.resources) {
            if (r0.cls != ResourceClass::ConstantBuffer || r0.gpu_addr != d.base ||
                r0.size != resource_size || r0.stride != d.stride || r0.format != format ||
                r0.num_components != components)
                continue;
            if (r0.fetch_pc == 0xFFFFFFFFu) r0.fetch_pc = fetch.fetch_pc;
            if (r0.fetch_pc == fetch.fetch_pc) { mapped = true; break; }
        }
        if (mapped) continue;
        ShaderResource r;
        r.cls = ResourceClass::ConstantBuffer;
        r.format = format;
        r.num_components = components;
        r.gpu_addr = d.base;
        r.size = resource_size;
        r.stride = d.stride;
        r.fetch_pc = fetch.fetch_pc;
        table.resources.push_back(r);
    }

    // Oversized FORMAT=INVALID store descriptors are deliberately absent from the generic fold.
    // Recover only the exact zero-clear kernel proven above, taking its live direct V# from the
    // user-data SGPRs and bounding the upload to this dispatch's one-dimensional invocation extent.
    // A nonzero FORMAT stays on the generic width-aware path: treating an R8/R16 zero as Uint32 would
    // overwrite adjacent components even though zero itself has the same bit pattern.
    uint32_t proven_linear_store_pc = UINT32_MAX;
    if (linear_local_x && linear_threads_x && tgid_x_sgpr != UINT32_MAX && user_sgprs) {
        std::vector<Rdna2Inst> instructions;
        rdna2_walk(code, dwords, instructions);
        if (instructions.size() == 4 && instructions[2].src[1].kind == OperandKind::SGPR) {
            const uint32_t srsrc = static_cast<uint32_t>(instructions[2].src[1].value);
            if (srsrc + 4u <= nsgpr) {
                const DecodedBufferDescriptor d = decode_buffer_descriptor(user_sgprs + srsrc);
                const uint32_t raw_format = (user_sgprs[srsrc + 3] >> 12) & 0x7Fu;
                const uint32_t resource_size = linear_dispatch_raw_store_size(
                    code, dwords, instructions[2].pc, d, linear_local_x, linear_threads_x,
                    tgid_x_sgpr);
                const bool already_materialized = std::any_of(
                    table.resources.begin(), table.resources.end(), [&](const ShaderResource& r) {
                        return r.cls == ResourceClass::ConstantBuffer && r.gpu_addr == d.base &&
                               r.fetch_pc == instructions[2].pc;
                    });
                if (raw_format == 0 && d.base > 0x10000 && resource_size &&
                    !already_materialized) {
                    ShaderResource r;
                    r.cls = ResourceClass::ConstantBuffer;
                    r.format = DataFormat::Uint32;
                    r.num_components = 1;
                    r.gpu_addr = d.base;
                    r.size = resource_size;
                    r.stride = d.stride;
                    r.fetch_pc = instructions[2].pc;
                    table.resources.push_back(r);
                    proven_linear_store_pc = instructions[2].pc;
                }
            }
        }
    }

    std::set<uint64_t> seen;
    for (const auto& u : srt_uses) {
        if (u.kind == 3) {
            const uint64_t dk = 0x8000000300000000ull | u.use_pc;
            if (!seen.insert(dk).second) continue;
            alignas(256) static std::array<uint8_t, 256> null_bvh{};
            ShaderResource r;
            r.cls = ResourceClass::ConstantBuffer;
            r.format = DataFormat::Uint32;
            r.num_components = 1;
            r.size = static_cast<uint32_t>(null_bvh.size());
            r.fetch_pc = u.use_pc;
            r.host_data = null_bvh.data();
            r.host_data_size = null_bvh.size();
            table.resources.push_back(r);
            continue;
        }
        if (u.kind == 2) {
            const DecodedBvhDescriptor d = decode_bvh_descriptor(u.bvh4.data());
            if (d.type != 8u || d.base <= 0x10000u || d.size_bytes == 0 ||
                d.size_bytes > 0x10000000u || d.size_bytes > UINT32_MAX ||
                !d.triangle_return_mode || d.box_node_64b || d.sort_enabled)
                continue;
            const uint64_t dk = 0x8000000200000000ull | u.use_pc;
            if (!seen.insert(dk).second) continue;
            bool mapped = false;
            for (auto& r0 : table.resources) {
                if (r0.cls != ResourceClass::ConstantBuffer || r0.gpu_addr != d.base ||
                    r0.size != static_cast<uint32_t>(d.size_bytes) ||
                    r0.bvh_box_grow != d.box_grow)
                    continue;
                if (r0.fetch_pc == 0xFFFFFFFFu) r0.fetch_pc = u.use_pc;
                if (r0.fetch_pc == u.use_pc) { mapped = true; break; }
            }
            if (mapped) continue;
            ShaderResource r;
            r.cls = ResourceClass::ConstantBuffer;
            r.format = DataFormat::Uint32;
            r.num_components = 1;
            r.gpu_addr = d.base;
            r.size = static_cast<uint32_t>(d.size_bytes);
            r.fetch_pc = u.use_pc;
            r.bvh_box_grow = d.box_grow;
            table.resources.push_back(r);
            continue;
        }
        if (u.kind != 1) continue;
        if (u.use_pc == proven_linear_store_pc) continue;
        if (u.instruction_format != UINT32_MAX && ((u.v4[3] >> 12) & 0x7Fu) == 0)
            continue;
        const bool exact_mtbuf = u.instruction_format != UINT32_MAX;
        // Keep an exact alias for every consumer pc, including an otherwise keyed table load. The
        // arbitrary-CFG compute dispatcher persists scalar values across basic blocks, but descriptor
        // identity is compile-time provenance and is intentionally not stored in those Function
        // variables. Exact aliases therefore remain the only unambiguous lookup after a block join.
        const uint64_t dk = 0x8000000100000000ull | u.use_pc;
        if (!seen.insert(dk).second) continue;
        bool clash = exact_mtbuf || u.key == 0xFFFFFFFFu;
        if (!clash)
            for (const auto& r0 : table.resources)
                if (r0.srt_offset == u.key) { clash = true; break; }

        const DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
        if (d.base <= 0x10000 || d.size_bytes == 0 || d.size_bytes > 0x10000000u) continue;
        if (clash && !exact_mtbuf) {
            bool piggybacked = false;
            for (auto& r0 : table.resources) {
                if (r0.cls != ResourceClass::ConstantBuffer || r0.gpu_addr != d.base ||
                    r0.size != d.size_bytes)
                    continue;
                if (r0.fetch_pc == 0xFFFFFFFFu) r0.fetch_pc = u.use_pc;
                piggybacked = r0.fetch_pc == u.use_pc;
                if (piggybacked) break;
            }
            if (piggybacked) continue;
        }
        ShaderResource r;
        r.cls = ResourceClass::ConstantBuffer;
        if (u.instruction_format != UINT32_MAX) {
            rdna2_buffer_format(u.instruction_format, &r.format, &r.num_components);
            if (r.format == DataFormat::Unknown || r.num_components == 0) continue;
        } else {
            r.format = d.format;
            r.num_components = d.num_components ? d.num_components : 1;
        }
        r.gpu_addr = d.base;
        r.size = d.size_bytes;
        r.stride = d.stride;
        r.srt_offset = clash ? 0xFFFFFFFFu : u.key;
        r.fetch_pc = u.use_pc;
        table.resources.push_back(r);
    }
    return srt_uses;
}

namespace { constexpr uint32_t kPsBindingBase = 32; }

// Preserve the exact resource-path proof across the decoded analysis and raw-byte translation
// passes. Ordinary instruction-scoped resources disappear with their consumers, but the dispatch-
// scoped null marker must survive because the raw translator independently repeats the proof.
ComputeResourcePathSpecializationReport specialize_compute_resource_paths(
        std::vector<Rdna2Inst>& instructions, ShaderResourceTable& resources,
        uint32_t wave_size) {
    ComputeResourcePathSpecializationReport report;
    std::vector<uint32_t> original_pcs;
    original_pcs.reserve(instructions.size());
    for (const Rdna2Inst& instruction : instructions)
        original_pcs.push_back(instruction.pc);

    report.proven_null_exits = rdna2_specialize_proven_null_bvh_paths(
        instructions, &resources, wave_size);
    if (!report.proven_null_exits) return report;
    report.shader_constant_branches =
        rdna2_specialize_shader_constant_branches(instructions);

    std::unordered_set<uint32_t> live_pcs;
    live_pcs.reserve(instructions.size());
    for (const Rdna2Inst& instruction : instructions)
        live_pcs.insert(instruction.pc);
    for (uint32_t pc : original_pcs)
        if (!live_pcs.contains(pc)) report.removed_pcs.push_back(pc);

    const size_t resources_before = resources.resources.size();
    std::erase_if(resources.resources, [&](const ShaderResource& resource) {
        // The raw-byte translator repeats resource-path specialization. Keep its dispatch-scoped
        // null proof even though the marked IMAGE_BVH instruction disappeared from this analysis
        // stream; all ordinary instruction-scoped resources can be pruned with their consumer.
        return resource.fetch_pc != 0xFFFFFFFFu &&
               !is_proven_null_bvh(resource) &&
               !live_pcs.contains(resource.fetch_pc);
    });
    report.removed_resources = resources_before - resources.resources.size();
    return report;
}

// Assign each resource its OWN descriptor binding, starting at `first` (0/1 reserved). The N-buffer
// model: the shader reads several distinct constant buffers (Unity's per-draw transform, per-frame,
// …) + vertex buffers + textures, and each must land at a separate binding so they don't collapse.
// The recompiler declares a storage buffer (cbuf/vbuf) or image sampler (texture) at each binding and
// resolves an s_buffer_load/image_sample to its resource's binding via provenance.
//
// The VS and PS tables are bound together in ONE descriptor set by the live renderer, so the two
// stages' binding ranges MUST be disjoint (VS keeps 2.., PS starts at kPsBindingBase). Within a
// stage, constant/vertex BUFFERS are assigned first (from `first`), then TEXTURES / storage images —
// but never on binding 2 or 3, which the recompiler's declare_cbufs always occupies with its two
// hardwired storage-buffer cbufs (v_cbuf/v_cbuf1). A shader whose FIRST resource is a texture used to
// land it on binding 2, declaring BOTH a combined image sampler AND that storage buffer at one
// binding (two descriptor types -> layout-creation failure, the draw disappears) (#157). Buffers-
// first keeps the common cbufs-first shaders' bindings byte-identical (cbufs 2/3, textures 4+).
// External linkage (declared in gpu_execute.hpp) so the binding policy is unit-testable.
void assign_convention_bindings(ShaderResourceTable& t, uint32_t first) {
    uint32_t next = first;
    for (auto& r : t.resources)
        if (r.cls == ResourceClass::ConstantBuffer || r.cls == ResourceClass::VertexBuffer)
            r.binding = next++;
    uint32_t tex_next = next > first + 2 ? next : first + 2;   // reserve the two hardwired cbuf slots
    for (auto& r : t.resources)
        if (r.cls != ResourceClass::ConstantBuffer && r.cls != ResourceClass::VertexBuffer)
            r.binding = tex_next++;
}

std::shared_ptr<ShaderResourceTable> merge_vertex_chain_resource_tables(
        const std::shared_ptr<ShaderResourceTable>& prolog,
        const std::shared_ptr<ShaderResourceTable>& main,
        uint32_t main_pc_offset) {
    if (!prolog && !main) return nullptr;
    auto merged = std::make_shared<ShaderResourceTable>();
    if (prolog) {
        merged->resources = prolog->resources;
        merged->vertices_per_instance = prolog->vertices_per_instance;
    }
    if (main) {
        if (!merged->vertices_per_instance)
            merged->vertices_per_instance = main->vertices_per_instance;
        merged->resources.reserve(merged->resources.size() + main->resources.size());
        for (ShaderResource resource : main->resources) {
            if (resource.fetch_pc != 0xFFFFFFFFu) {
                if (resource.fetch_pc > UINT32_MAX - main_pc_offset) return nullptr;
                resource.fetch_pc += main_pc_offset;
            }
            merged->resources.push_back(std::move(resource));
        }
    }
    assign_convention_bindings(*merged, 2u);
    return merged;
}

std::shared_ptr<ShaderResourceTable> build_stage_table(const GpuState& st, uint64_t code_addr,
                                                       bool is_ps, uint32_t draw_vertex_count) {
    if (!code_addr) return nullptr;
    const auto* hdr = (const AgcShaderHeader*)prosper_agc_shader_header_for_code(code_addr);
    if (!hdr) return nullptr;
    using StageClock = std::chrono::steady_clock;
    const bool phase_timing = getenv("PROSPER_RENDER_TIMING") != nullptr;
    const auto metadata_start = phase_timing ? StageClock::now() : StageClock::time_point{};
    namespace P = prosper::agc::Pm4;
    const bool log = getenv("PROSPER_GFXLOG") != nullptr;
    const size_t shader_dwords = registered_shader_dwords(*hdr, code_addr);

    // The V#/T# descriptors live in the stage's user-data SGPR block. The pixel stage uses PS user
    // data; the vertex/geometry stage under NGG merges the ES program's descriptors into GS user data.
    // Try the stage's expected base, then the alternates — the exact stage-merge layout varies, so use
    // whichever base actually yields resources.
    uint32_t bases[3];
    if (is_ps) { bases[0] = P::SPI_SHADER_USER_DATA_PS_0; bases[1] = P::SPI_SHADER_USER_DATA_GS_0; bases[2] = P::SPI_SHADER_USER_DATA_VS_0; }
    else       { bases[0] = P::SPI_SHADER_USER_DATA_GS_0; bases[1] = P::SPI_SHADER_USER_DATA_VS_0; bases[2] = P::SPI_SHADER_USER_DATA_PS_0; }

    // Per-shader user-data RANGE: the shader blob's "specials" block declares which DWORD range of
    // the stage's USER_DATA register block holds this shader's SGPR-visible user data
    // (user_data_range_start/end — the range SetSource programs; e.g. DOLL's UE4 Slate VS declares
    // [0,8) matching its 8-dword {V#, ptr, ptr} block). Header sharp/direct offsets are relative to
    // range_start, so seed the SGPR block from USER_DATA_<stage>_<range_start>. Every shader
    // observed live (DOLL + Messenger) declares start=0, so this is currently behavior-identical —
    // LATENT support for a start!=0 shader, guarded back to 0 on insane metadata.
    // CONFIDENCE: LOW on start!=0 semantics (no live example yet); zero risk for start==0.
    uint32_t range_start = 0;
    if (hdr->specials && guest_readable((uint64_t)(uintptr_t)hdr->specials, sizeof(AgcShaderSpecials))) {
        const uint32_t s = hdr->specials->user_data_range_start;
        const uint32_t e = hdr->specials->user_data_range_end;
        if (s < kUserSgprs && e > s && e <= 2 * kUserSgprs) range_start = s;
        if (log && range_start) {   // once per shader: non-zero ranges are the rare/interesting case
            static std::set<uint64_t> logged;
            if (logged.insert(code_addr).second)
                fprintf(stderr, "[agc] %s 0x%llx user_data_range=[%u,%u) -> seeding from USER_DATA_%u\n",
                        is_ps ? "PS" : "VS", (unsigned long long)code_addr, s, e, range_start);
        }
    }

    // PROSPER_UD_TAIL_ALIGN (#305 A/B) — FALSIFIED, retained as a documented negative result.
    //
    // The hypothesis was that a stage's user data is the TAIL of the block the pipeline programmed:
    // on Nikoderiko the failing stages' declared descriptors do land on clean guest pointers exactly
    // `programmed - user_data_range_end` dwords above USER_DATA_*_0, and the stages that resolve
    // today are exactly those where `programmed == range_end`. Two independent measurements kill it:
    //
    //  * SPI_SHADER_PGM_RSRC2_GS.USER_SGPR equals `user_data_range_end` for every stage measured
    //    (8/8, 12/12, 20/20, ...). That field is the count of user SGPRs the hardware loads, starting
    //    at USER_DATA_GS_0 — so a stage with range_end=12 physically cannot see GS_12..GS_31, and no
    //    tail alignment can be what the guest intended. It also confirms the existing seeding base
    //    and the 8 leading system SGPRs of the merged-stage ABI are correct.
    //  * A live A/B with the semantics-derived prefix raised Nikoderiko's exec-recompile rejects
    //    well above the unmodified baseline rather than clearing them.
    //
    // Keep the switch so the measurement is reproducible; it must stay off. CONFIDENCE: HIGH that
    // tail alignment is wrong.
    static const char* const tail_mode = std::getenv("PROSPER_UD_TAIL_ALIGN");
    if (tail_mode) {
        const uint32_t e = (hdr->specials &&
                            guest_readable((uint64_t)(uintptr_t)hdr->specials,
                                           sizeof(AgcShaderSpecials)))
                               ? hdr->specials->user_data_range_end : 0u;
        const uint32_t rsrc2 = [&] {
            const auto it = st.sh.find(is_ps ? P::SPI_SHADER_PGM_RSRC2_PS : P::SPI_SHADER_PGM_RSRC2_GS);
            return it == st.sh.end() ? 0u : it->second;
        }();
        uint32_t user_sgprs = (rsrc2 >> P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_SHIFT) &
                              P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_MASK;
        if (!is_ps)
            user_sgprs |= ((rsrc2 >> P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_MSB_SHIFT) &
                           P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_MSB_MASK) << 5;
        // Two candidate sources for the size of the block that precedes this stage's own user data:
        // the hardware user-SGPR count the pipeline programmed, and the shader's own declared vertex
        // input count (one 4-dword V# per input, the fetch block). Report both, drive with one.
        const uint32_t sem_prefix = is_ps ? 0u : hdr->num_input_semantics * 4u;
        const uint32_t rsrc2_prefix = (user_sgprs >= e) ? user_sgprs - e : UINT32_MAX;
        const bool use_sem = tail_mode[0] == 's';
        const uint32_t prefix = use_sem ? sem_prefix : rsrc2_prefix;
        if (log || g_dyntrace_force) {
            static std::set<uint64_t> logged;
            if (logged.insert(code_addr).second)
                fprintf(stderr,
                        "[udtail] %s 0x%llx rsrc2=0x%08x user_sgprs=%u range_end=%u num_in_sem=%u "
                        "prefix_rsrc2=%d prefix_sem=%u mode=%s\n",
                        is_ps ? "PS" : "VS", (unsigned long long)code_addr, rsrc2, user_sgprs, e,
                        hdr->num_input_semantics, (int)rsrc2_prefix, sem_prefix,
                        use_sem ? "sem" : "rsrc2");
        }
        if (e && prefix != UINT32_MAX && prefix + e <= kUserSgprs) range_start = prefix;
    }

    // PROSPER_RESDUMP: raw dump of the user-data struct + SGPR block per base, so the EUD layout
    // (which sharps have offset_dw>=16, and where the EUD pointer sits) can be read empirically.
    bool resdump = getenv("PROSPER_RESDUMP") != nullptr;
    if (resdump)   // PROSPER_RESDUMP_ADDR=<hex code addr>: narrow the dump to one shader
        if (const char* fa = getenv("PROSPER_RESDUMP_ADDR"))
            resdump = strtoull(fa, nullptr, 16) == code_addr;
    if (g_dyntrace_force) resdump = true;   // failure replay: always dump the failing stage's blocks
    if (resdump) {
        const AgcShaderUserData* ud = hdr->user_data;
        fprintf(stderr, "[resdump] %s code=0x%llx type=%u ud=%p range_start=%u (end=%u)\n",
                is_ps ? "PS" : "VS", (unsigned long long)code_addr, hdr->type, (const void*)ud,
                range_start, hdr->specials ? (uint32_t)hdr->specials->user_data_range_end : 0u);
        if (ud) {
            fprintf(stderr, "[resdump]   eud_size_dw=%u srt_size_dw=%u direct_count=%u sharp_counts={%u,%u,%u,%u}\n",
                    ud->eud_size_dw, ud->srt_size_dw, ud->direct_resource_count,
                    ud->sharp_resource_count[0], ud->sharp_resource_count[1],
                    ud->sharp_resource_count[2], ud->sharp_resource_count[3]);
            for (int cat = 0; cat < 4; cat++) {
                const AgcShaderSharp* sh = ud->sharp_resource_offset[cat];
                if (!sh || !ud->sharp_resource_count[cat]) continue;
                fprintf(stderr, "[resdump]   sharp[%d] offset_dw:", cat);
                for (uint16_t s = 0; s < ud->sharp_resource_count[cat] && s < 12; s++)
                    fprintf(stderr, " %u%s", sh[s].offset_dw(), sh[s].empty() ? "(empty)" : "");
                fprintf(stderr, "\n");
            }
            if (ud->direct_resource_offset && ud->direct_resource_count) {
                fprintf(stderr, "[resdump]   direct offset_dw:");
                for (uint16_t t2 = 0; t2 < ud->direct_resource_count && t2 < 16; t2++)
                    fprintf(stderr, " [%u]=%u", t2, ud->direct_resource_offset[t2]);
                fprintf(stderr, "\n");
            }
        }
        for (uint32_t base : bases) {
            uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, base, sgprs);
            fprintf(stderr, "[resdump] %s code=0x%llx sgprs@0x%x:",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr, base);
            for (uint32_t i = 0; i < kUserSgprs; i++) fprintf(stderr, " %08x", sgprs[i]);
            fprintf(stderr, "\n");
        }
        // USER-DATA POINTER MAP. A stage's descriptor-table pointers are ordinary 64-bit guest
        // addresses sitting in consecutive user-data dwords, and every bindless descriptor chain
        // starts by dereferencing one. When a stage fails to resolve its V#/T#, the first question
        // is whether the seeded block even CONTAINS the pointers the shader loads from — the AGC
        // header's direct-resource offsets and the shader's own SBASE registers both name dword
        // positions, so a block whose readable pointers sit elsewhere is not the block that shader
        // ran with. Report every dword pair that is a mapped guest address, plus the readability of
        // each header-declared direct offset, so that question is answered by data rather than by
        // eye. Probing is bounded to the 32-dword window and uses the fault-safe readability probe.
        for (uint32_t base : bases) {
            uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, base, sgprs);
            fprintf(stderr, "[udmap] %s code=0x%llx base=0x%x readable-ptr dwords:",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr, base);
            bool any = false;
            for (uint32_t i = 0; i + 1 < kUserSgprs; i++) {
                const uint64_t candidate =
                    (uint64_t)sgprs[i] | ((uint64_t)sgprs[i + 1] << 32);
                if (candidate <= 0x10000 || !guest_readable(candidate, 8)) continue;
                fprintf(stderr, " [%u]=0x%llx", i, (unsigned long long)candidate);
                any = true;
            }
            if (!any) fprintf(stderr, " none");
            fprintf(stderr, "\n");
        }
        // A stage pointer may carry aperture/tag bits above the title's usable GPU VA, exactly as the
        // scalar fold's S_LOAD canonicalization assumes. Accept a dword pair as a pointer when the
        // raw 64-bit value or either canonical form is mapped, so a tagged-but-valid table pointer is
        // never reported as unmapped.
        const auto pointer_is_mapped = [](uint64_t value) {
            for (uint64_t mask : {~uint64_t{0}, uint64_t{0xFFFFFFFFFFFF}, uint64_t{0xFFFFFFFFFF}}) {
                const uint64_t candidate = value & mask;
                if (candidate > 0x10000 && guest_readable(candidate, 8)) return true;
            }
            return false;
        };
        if (const AgcShaderUserData* ud = hdr->user_data) {
            uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, bases[0] + range_start, sgprs);
            fprintf(stderr, "[udmap] %s code=0x%llx declared direct:",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr);
            for (uint16_t t = 0; t < ud->direct_resource_count && t < 16; t++) {
                const uint32_t off = ud->direct_resource_offset[t];
                if (off == 0xFFFFu || off + 1 >= kUserSgprs) continue;
                const uint64_t candidate =
                    (uint64_t)sgprs[off] | ((uint64_t)sgprs[off + 1] << 32);
                fprintf(stderr, " [%u]@dw%u=0x%llx(%s)", t, off,
                        (unsigned long long)candidate,
                        pointer_is_mapped(candidate) ? "readable" : "UNMAPPED");
            }
            // Self-validating half: search the 32-dword window for the ONE seed offset at which
            // EVERY declared direct pointer becomes a mapped guest address. If that offset equals
            // the shader's raw user_data_range_start, the block is simply seeded from the wrong
            // register and the metadata already said so; if no offset works, the block genuinely
            // does not contain this shader's pointers and the seeding origin is not the defect.
            uint32_t implied = UINT32_MAX;
            uint32_t declared_pointers = 0;
            for (uint16_t t = 0; t < ud->direct_resource_count && t < 16; t++)
                if (ud->direct_resource_offset[t] != 0xFFFFu) ++declared_pointers;
            for (uint32_t seed = 0; declared_pointers && seed < kUserSgprs; ++seed) {
                uint32_t probe[kUserSgprs];
                read_user_sgprs(st.sh, bases[0] + seed, probe);
                bool all_mapped = true;
                for (uint16_t t = 0; all_mapped && t < ud->direct_resource_count && t < 16; t++) {
                    const uint32_t off = ud->direct_resource_offset[t];
                    if (off == 0xFFFFu) continue;
                    if (off + 1 >= kUserSgprs) { all_mapped = false; break; }
                    const uint64_t value =
                        (uint64_t)probe[off] | ((uint64_t)probe[off + 1] << 32);
                    all_mapped = pointer_is_mapped(value);
                }
                if (all_mapped) { implied = seed; break; }
            }
            const uint32_t raw_start = hdr->specials ? hdr->specials->user_data_range_start : 0u;
            const uint32_t raw_end = hdr->specials ? hdr->specials->user_data_range_end : 0u;
            fprintf(stderr, " | specials raw=[%u,%u) seeded=%u implied=",
                    raw_start, raw_end, range_start);
            if (implied == UINT32_MAX) fprintf(stderr, "none");
            else fprintf(stderr, "%u", implied);
            fprintf(stderr, "\n");
        }
        // #305 instrument: the code address alone does not identify a shader. CreateShader's registry
        // is append-only and the header lookup returns the FIRST registration bound to an address, so
        // a guest that re-registers a different shader over a recycled code allocation resolves to the
        // OLDEST layout while the register block holds the newest bind. Enumerate every candidate and
        // report, for each, whether ALL of its declared direct offsets land on mapped pointers in the
        // block this draw actually has — the candidate that does is the layout the guest programmed.
        {
            const void* cands[16] = {};
            const size_t total =
                prosper_agc_shader_headers_for_code(code_addr, cands, std::size(cands));
            uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, bases[0] + range_start, sgprs);
            fprintf(stderr, "[udcand] %s code=0x%llx registrations=%zu\n",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr, total);
            for (size_t ci = 0; ci < total && ci < std::size(cands); ci++) {
                const auto* ch = static_cast<const AgcShaderHeader*>(cands[ci]);
                // This loop exists to enumerate EXTRA registrations at one code address — i.e. the
                // recycled-allocation case — so its entries are the ones most likely to point into
                // a blob the guest has already freed. Probe every hop, exactly as the whole-registry
                // scan below does.
                if (!ch || !guest_readable((uint64_t)(uintptr_t)ch, sizeof(AgcShaderHeader)))
                    continue;
                const bool sp_ok = ch->specials && guest_readable((uint64_t)(uintptr_t)ch->specials,
                                                                  sizeof(AgcShaderSpecials));
                fprintf(stderr, "[udcand]   #%zu hdr=%p type=%u range=[%u,%u) direct:", ci,
                        (const void*)ch, ch->type,
                        sp_ok ? (uint32_t)ch->specials->user_data_range_start : 0u,
                        sp_ok ? (uint32_t)ch->specials->user_data_range_end : 0u);
                bool all = true, any = false;
                const AgcShaderUserData* cud = ch->user_data;
                if (cud && !guest_readable((uint64_t)(uintptr_t)cud, sizeof(AgcShaderUserData)))
                    cud = nullptr;
                uint16_t cud_count = 0;
                const uint16_t* cud_offsets = nullptr;
                if (cud) {
                    cud_count = cud->direct_resource_count;
                    cud_offsets = cud->direct_resource_offset;
                    if (!cud_offsets || !cud_count || cud_count > 64 ||
                        !guest_readable((uint64_t)(uintptr_t)cud_offsets,
                                        cud_count * (uint32_t)sizeof(uint16_t)))
                        cud = nullptr;
                }
                if (cud) {
                    for (uint16_t t = 0; t < cud_count && t < 16; t++) {
                        const uint32_t off = cud_offsets[t];
                        if (off == 0xFFFFu) continue;
                        any = true;
                        if (off + 1 >= kUserSgprs) { all = false; fprintf(stderr, " [%u]@dw%u=OOB", t, off); continue; }
                        const uint64_t v = (uint64_t)sgprs[off] | ((uint64_t)sgprs[off + 1] << 32);
                        const bool ok = pointer_is_mapped(v);
                        all = all && ok;
                        fprintf(stderr, " [%u]@dw%u=0x%llx(%s)", t, off, (unsigned long long)v,
                                ok ? "readable" : "UNMAPPED");
                    }
                    fprintf(stderr, " sharps={%u,%u,%u,%u}", cud->sharp_resource_count[0],
                            cud->sharp_resource_count[1], cud->sharp_resource_count[2],
                            cud->sharp_resource_count[3]);
                }
                fprintf(stderr, " -> %s\n", !any ? "no-direct" : (all ? "FITS-BLOCK" : "misfit"));
            }
            // Whole-registry search: the block this draw programmed is a fact; the header is an
            // inference from the PGM register. If the resolved header does not fit the block but
            // some OTHER registered shader's declared layout does — and its declared size equals the
            // contiguous extent this draw's own bind freshly wrote — then the PGM register, not the
            // register block, is what is stale. `fresh` is that extent, from the write-provenance
            // instrument (dwords sharing the newest write order, from dw0 up).
            uint32_t fresh = 0;
            if (prosper::gpu::udprov_enabled()) {
                uint64_t newest = 0;
                for (uint32_t i = 0; i < kUserSgprs; i++) {
                    const auto it = st.sh_prov.find(bases[0] + i);
                    if (it != st.sh_prov.end())
                        newest = std::max(newest, it->second & ~GpuState::kProvIndirect);
                }
                for (uint32_t i = 0; i < kUserSgprs; i++) {
                    const auto it = st.sh_prov.find(bases[0] + i);
                    if (it == st.sh_prov.end() ||
                        (it->second & ~GpuState::kProvIndirect) != newest) break;
                    fresh = i + 1;
                }
            }
            const size_t registry = prosper_agc_shader_count();
            fprintf(stderr, "[udcand]   block: fresh_extent=%u registry=%zu fitting:", fresh, registry);
            size_t fits = 0;
            for (size_t si = 0; si < registry; si++) {
                const auto* sh = static_cast<const AgcShaderHeader*>(prosper_agc_shader_at(si));
                // This walks the WHOLE registry (thousands of entries), and every pointer in it is
                // guest-owned metadata that may point into a blob the guest has since freed. Probe
                // each hop before dereferencing it — the resolved-header path above only ever
                // touches one entry, this one touches all of them.
                if (!sh || !guest_readable((uint64_t)(uintptr_t)sh, sizeof(AgcShaderHeader)))
                    continue;
                if (!sh->user_data ||
                    !guest_readable((uint64_t)(uintptr_t)sh->user_data, sizeof(AgcShaderUserData)))
                    continue;
                const uint16_t direct_count = sh->user_data->direct_resource_count;
                const uint16_t* direct_offsets = sh->user_data->direct_resource_offset;
                if (!direct_offsets || !direct_count || direct_count > 64 ||
                    !guest_readable((uint64_t)(uintptr_t)direct_offsets,
                                    direct_count * (uint32_t)sizeof(uint16_t)))
                    continue;
                if (!sh->specials ||
                    !guest_readable((uint64_t)(uintptr_t)sh->specials, sizeof(AgcShaderSpecials)))
                    continue;
                const uint32_t end = sh->specials->user_data_range_end;
                if (!fresh || end != fresh) continue;   // must match what this bind actually wrote
                bool all = true, any = false;
                for (uint16_t t = 0; all && t < direct_count && t < 16; t++) {
                    const uint32_t off = direct_offsets[t];
                    if (off == 0xFFFFu) continue;
                    any = true;
                    if (off + 1 >= kUserSgprs) { all = false; break; }
                    all = pointer_is_mapped((uint64_t)sgprs[off] |
                                            ((uint64_t)sgprs[off + 1] << 32));
                }
                if (!any || !all) continue;
                if (fits++ < 8)
                    fprintf(stderr, " 0x%llx(type=%u,end=%u)",
                            (unsigned long long)(uintptr_t)sh->code, sh->type, end);
            }
            if (!fits) fprintf(stderr, " none");
            fprintf(stderr, " total=%zu\n", fits);
        }
        // PROSPER_UDPROV (#305): write provenance for the block this stage was seeded from, plus the
        // program-address registers that named the stage. "The block holds a previous pipeline's
        // user data" is a claim about WHEN each dword was last written relative to the draw — values
        // alone cannot express it. Print, per dword, the value and the command_order of its last
        // write (i = indirect path, d = direct), the orders of the PGM registers that selected this
        // shader, and the draw's own order.
        if (prosper::gpu::udprov_enabled()) {
            // order + path + SOURCE. `q` is the queue origin the packet folded under
            // (0=unknown/graphics, 1=Dcb, 2=Acb, 3=DcbFinal), `f` the top-level fold (submit
            // stream) id, `j` the sceAgcDcbJump recursion depth. A write whose q/f differs from
            // the draw's own is one that did not arrive in this submit's inline position.
            const auto prov = [&](uint32_t reg) -> std::string {
                const auto it = st.sh_prov.find(reg);
                if (it == st.sh_prov.end()) return "never";
                char buf[64];
                const auto sit = st.sh_prov_src.find(reg);
                const uint64_t src = sit == st.sh_prov_src.end() ? 0 : sit->second;
                snprintf(buf, sizeof(buf), "%c%llu/q%u,f%llu,j%u",
                         (it->second & GpuState::kProvIndirect) ? 'i' : 'd',
                         (unsigned long long)(it->second & ~GpuState::kProvIndirect),
                         (unsigned)(src & 0xFFu), (unsigned long long)(src >> 16),
                         (unsigned)((src >> 8) & 0xFFu));
                return buf;
            };
            const auto shv = [&](uint32_t reg) {
                const auto it = st.sh.find(reg);
                return it == st.sh.end() ? 0u : it->second;
            };
            fprintf(stderr,
                    "[udprov] %s code=0x%llx draw_order=%llu pgm: ES_LO=0x%x@%s ES_HI=0x%x@%s "
                    "GS_LO=0x%x@%s GS_HI=0x%x@%s PS_LO=0x%x@%s\n",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr,
                    (unsigned long long)st.command_order,
                    shv(P::SPI_SHADER_PGM_LO_ES), prov(P::SPI_SHADER_PGM_LO_ES).c_str(),
                    shv(P::SPI_SHADER_PGM_HI_ES), prov(P::SPI_SHADER_PGM_HI_ES).c_str(),
                    shv(P::SPI_SHADER_PGM_LO_GS), prov(P::SPI_SHADER_PGM_LO_GS).c_str(),
                    shv(P::SPI_SHADER_PGM_HI_GS), prov(P::SPI_SHADER_PGM_HI_GS).c_str(),
                    shv(P::SPI_SHADER_PGM_LO_PS), prov(P::SPI_SHADER_PGM_LO_PS).c_str());
            {
                const auto cxv = [&](uint32_t reg) {
                    const auto it = st.cx.find(reg);
                    return it == st.cx.end() ? 0u : it->second;
                };
                fprintf(stderr,
                        "[udprov]   stages: HS_LO=0x%x@%s LS_LO=0x%x@%s RSRC2_GS=0x%x@%s "
                        "VGT_SHADER_STAGES_EN=0x%x\n",
                        shv(P::SPI_SHADER_PGM_LO_HS), prov(P::SPI_SHADER_PGM_LO_HS).c_str(),
                        shv(P::SPI_SHADER_PGM_LO_LS), prov(P::SPI_SHADER_PGM_LO_LS).c_str(),
                        shv(P::SPI_SHADER_PGM_RSRC2_GS), prov(P::SPI_SHADER_PGM_RSRC2_GS).c_str(),
                        cxv(P::VGT_SHADER_STAGES_EN));
            }
            for (uint32_t bi = 0; bi < std::size(bases); ++bi) {
                fprintf(stderr, "[udprov]   base=0x%x:", bases[bi]);
                for (uint32_t i = 0; i < kUserSgprs; i++) {
                    const uint32_t reg = bases[bi] + i;
                    fprintf(stderr, " dw%u=0x%08x@%s", i, shv(reg), prov(reg).c_str());
                }
                fprintf(stderr, "\n");
            }
        }
        // ALL set sh registers (sorted) — finds where the user-data SGPRs actually landed, including
        // any at unexpected offsets (a wrong indirect-register decode would scatter them).
        fprintf(stderr, "[resdump]   %zu sh regs set; lowest 48 offsets:", st.sh.size());
        std::vector<uint32_t> keys; for (auto& kv : st.sh) if (kv.second) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        for (size_t i = 0; i < keys.size() && i < 48; i++) fprintf(stderr, " 0x%x", keys[i]);
        fprintf(stderr, "\n");
    }

    // Bindless-dynamic buffer fetch: const-fold the scalar setup to resolve each buffer_load_format's
    // V#, then emit it as a VertexBuffer keyed by its SRSRC SGPR so the recompiler's by_sgpr_base resolves
    // it. Despite the historical vertex-fetch name, pixel shaders use the same instructions for UE4
    // structured/material buffers. Dropping the PS results leaves those loads with no storage-buffer
    // binding, so the recompiler falls back to its legacy binding 2 and reads zeros (#719).
    // The SAME const-fold also recovers descriptor-TABLE uses for BOTH stages (#294): UE4 shaders
    // s_load their T#/S#/V# descriptors from a table pointer in the user-data SGPRs and consume them
    // via image_sample / s_buffer_load — srt_uses reports each with its load-immediate key, which
    // becomes the resource's srt_offset (the recompiler's by_srt_offset provenance).
    std::vector<DynFetch> dyn_vb;
    std::vector<SrtUse> srt_uses;
    // Build the primary metadata table before dynamic folding so a proven PS dispatch can read its
    // direct selector resource. The fold then follows the same selected arm as recompilation.
    const uint32_t user_sgpr_base = is_ps ? 0u : 8u;
    uint32_t primary_sgprs[kUserSgprs];
    read_user_sgprs(st.sh, bases[0] + range_start, primary_sgprs);
    ShaderResourceTable primary_resources =
        build_shader_resources(*hdr, primary_sgprs, kUserSgprs, user_sgpr_base);
    PcrelDispatchSelection dispatch_selection;
    std::shared_ptr<const ShaderCodeAnalysis> shader_analysis;
    if (is_ps) {
        shader_analysis = analyze_shader_code_cached(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)), shader_dwords);
        dispatch_selection = select_pcrel_dispatch(
            (const uint32_t*)(uintptr_t)code_addr, shader_dwords, &primary_resources,
            shader_analysis.get());
    }
    const auto metadata_done = phase_timing ? StageClock::now() : StageClock::time_point{};
    if (is_ps) {
        dyn_vb = resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, shader_dwords,
                                       primary_sgprs, kUserSgprs, 0, &srt_uses,
                                       dispatch_selection.target, &dispatch_selection.dispatch);
    } else {
        // NGG merged VS/GS: s0..s7 are system SGPRs, user data starts at s8 (confirmed by matching the
        // shader's s[8:11]/s[24:25] descriptor pointers to the register file at GS_0+offset).
        uint32_t system_sgprs[2] = {};
        uint32_t system_count = 0;
        if (hdr->type == 6) { // fused GS back: s[0:1] points at the driver stage-data table
            const auto sh_value = [&](uint32_t reg) {
                const auto found = st.sh.find(reg);
                return found == st.sh.end() ? 0u : found->second;
            };
            system_sgprs[0] = sh_value(P::SPI_SHADER_USER_DATA_ADDR_LO_GS);
            system_sgprs[1] = sh_value(P::SPI_SHADER_USER_DATA_ADDR_HI_GS);
            system_count = (system_sgprs[0] || system_sgprs[1]) ? 2u : 0u;
        }
        dyn_vb = resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, shader_dwords,
                                       primary_sgprs, kUserSgprs, 8, &srt_uses,
                                       UINT32_MAX, nullptr, system_sgprs, system_count);
        if (getenv("PROSPER_GFXLOG") || getenv("PROSPER_RESDUMP")) {
            fprintf(stderr, "[dynvb] VS resolved %zu dynamic vertex-fetch descriptor(s):\n", dyn_vb.size());
            for (auto& kv : dyn_vb) {
                const auto& d = kv.desc;
                fprintf(stderr, "[dynvb]   pc=%u SRSRC s%d -> base=0x%llx stride=%u num_records=%u size=%u fmt=%u nc=%u\n",
                        kv.fetch_pc, kv.srsrc, (unsigned long long)d.base, d.stride, d.num_records, d.size_bytes,
                        (unsigned)d.format, d.num_components);
                if (guest_readable(d.base, 64)) {   // raw dwords of vertex 0 (to read packed formats) + floats
                    const uint32_t* u = (const uint32_t*)(uintptr_t)d.base;
                    fprintf(stderr, "[dynvb]     v3fmt=0x%x  raw v0: %08x %08x %08x %08x\n",
                            kv.desc_v3, u[0], u[1], u[2], u[3]);
                    // Print the first 3 vertex records (the DrawIndexAuto count=3 triangle) as floats, so the
                    // actual on/off-screen span can be computed offline. Each record is `stride` bytes.
                    for (int rec = 0; rec < 3 && d.stride; rec++) {
                        uint64_t a = d.base + (uint64_t)rec * d.stride;
                        if (!guest_readable(a, 16)) break;
                        const float* f = (const float*)(uintptr_t)a;
                        fprintf(stderr, "[dynvb]     rec%d: %.3f %.3f %.3f %.3f\n", rec, f[0], f[1], f[2], f[3]);
                    }
                }
            }
        }
    }

    const auto fold_done = phase_timing ? StageClock::now() : StageClock::time_point{};
    auto record_phases = [&] {
        if (!phase_timing) return;
        const auto resources_done = StageClock::now();
        record_stage_table_phases(
            std::chrono::duration<double, std::milli>(metadata_done - metadata_start).count(),
            std::chrono::duration<double, std::milli>(fold_done - metadata_done).count(),
            std::chrono::duration<double, std::milli>(resources_done - fold_done).count());
    };

    // NGG VS/GS loads user data at shader s8 (s0..s7 = system SGPRs); PS at s0. The resource sgpr_base
    // (an s_buffer_load/image_sample's SBASE/SRSRC register) is in that shader-SGPR space.
    for (size_t base_index = 0; base_index < std::size(bases); ++base_index) {
        const uint32_t base = bases[base_index];
        ShaderResourceTable t;
        if (base_index == 0) {
            t = std::move(primary_resources);
        } else {
            uint32_t sgprs[kUserSgprs];
            read_user_sgprs(st.sh, base + range_start, sgprs);
            t = build_shader_resources(*hdr, sgprs, kUserSgprs, user_sgpr_base);
        }
        // Add the const-fold-resolved dynamic buffers, keyed by their SRSRC SGPR so the
        // recompiler's by_sgpr_base() resolves each buffer_load_format. The V#'s data format is patched
        // at runtime by the fetch shader (so the load-time snapshot reads Unknown) — default to Float32
        // (a raw 32-bit-per-component fetch, correct for float attributes like positions).
        for (auto& kv : dyn_vb) {
            // Only ABI-proven vertex/instance fetches use the special address path that replaces
            // VADDR with a built-in index and therefore needs OFFSET/SOFFSET folded into the bound
            // base. A shader-computed VADDR keeps the instruction's complete address expression,
            // including idxen+offen's second VGPR, so bind the original base for that mode.
            const auto& d = kv.index_mode == VertexFetchIndexMode::Shader
                          ? kv.unshifted_desc : kv.desc;
            if (kv.instruction_format != UINT32_MAX &&
                ((kv.desc_v3 >> 12) & 0x7Fu) == 0)
                continue;
            // Belt-and-suspenders: resolve_dynamic_fetch filters these before emitting DynFetch, but
            // never allow a deliberately rejected packed descriptor to reach the generic Float32
            // fallback if another producer constructs a DynFetch in the future.
            if (kv.instruction_format == UINT32_MAX && d.forbid_unknown_fallback) continue;
            // A SEED-fallback entry must not shadow a metadata-described DIRECT vertex buffer at the
            // same SGPRs (see DynFetch::from_seed): the direct resource resolves the fetch through
            // the faithful address path, which is the correct model for a single un-patched V#.
            if (kv.from_seed && kv.instruction_format == UINT32_MAX) {
                bool direct_exists = false;
                for (const auto& r0 : t.resources)
                    if (r0.cls == ResourceClass::VertexBuffer && r0.sgpr_base == (uint32_t)kv.srsrc)
                        { direct_exists = true; break; }
                if (direct_exists) continue;
            }
            ShaderResource r;
            // A pixel-stage buffer_load_format_* is a structured/material-buffer read, not a
            // vertex attribute fetch. Keeping it as ConstantBuffer preserves the instruction's
            // computed VADDR/stride address; labeling it VertexBuffer makes the recompiler take
            // the gl_VertexIndex shortcut and rejects valid stride-2 uint16 tables (#719).
            r.cls           = is_ps ? ResourceClass::ConstantBuffer : ResourceClass::VertexBuffer;
            if (kv.instruction_format != UINT32_MAX) {
                rdna2_buffer_format(kv.instruction_format, &r.format, &r.num_components);
                if (r.format == DataFormat::Unknown || r.num_components == 0) continue;
            } else {
                r.format = (d.format == DataFormat::Unknown) ? DataFormat::Float32 : d.format;
                r.num_components = d.num_components ? d.num_components : 4;
            }
            r.gpu_addr      = d.base;
            if (d.size_bytes) {
                r.size = d.size_bytes;
            } else if (is_ps) {
                // A pixel-stage buffer_load_format_* retains its real VADDR/stride addressing as a
                // ConstantBuffer. Fragment/material indices are unrelated to submitted vertex count,
                // so applying the VS draw-derived bound here can truncate an access that the previous
                // compatibility allocation covered. Keep that allocation for PS until the fold can
                // prove an access range, and surface the uncertainty below.
                r.size = d.stride ? d.stride * 4u : 128u;
            } else {
                const bool packed_word =
                    r.format == DataFormat::Float10_11_11 ||
                    r.format == DataFormat::Unorm2_10_10_10 ||
                    r.format == DataFormat::Snorm2_10_10_10 ||
                    r.format == DataFormat::Uint2_10_10_10 ||
                    r.format == DataFormat::Sint2_10_10_10;
                const uint64_t element_bytes = packed_word ? 4u :
                    static_cast<uint64_t>(data_format_bytes(r.format)) * r.num_components;
                const uint64_t record_bytes = d.stride ? d.stride : element_bytes;
                const uint64_t draw_bytes = record_bytes * draw_vertex_count;
                r.size = static_cast<uint32_t>(std::min<uint64_t>(draw_bytes, 0x10000000ull));
            }
            r.stride        = d.stride;
            r.sgpr_base     = kv.srsrc;           // DIRECT provenance = the fetch's SRSRC SGPR (fallback)
            r.fetch_pc      = kv.fetch_pc;        // PER-FETCH provenance = the exact fetch instruction
            r.fetch_index_mode = kv.index_mode;
            r.srt_offset    = 0xFFFFFFFFu;
            // The dynamic fold may resolve a descriptor loaded or patched inside the shader. Only
            // an unchanged four-dword entry seed has one exact raw PM4 SH source. Keep the primary
            // entry range that actually fed the fold; do not search other stage bases by whether
            // their decoded output happens to equal this normalized resource (#1853).
            if (kv.direct_user_data_index <= kUserSgprs - 4u) {
                r.direct_vsharp_sh_register_base =
                    bases[0] + range_start + kv.direct_user_data_index;
            } else {
                r.direct_vsharp_sh_register_base =
                    ShaderResource::kDirectVSharpOriginAmbiguous;
            }
            if (!d.size_bytes && is_ps) {
                static std::mutex zero_ps_mx;
                static std::set<std::tuple<uint64_t, uint32_t, uint32_t>> zero_ps_seen;
                std::lock_guard<std::mutex> lk(zero_ps_mx);
                if (zero_ps_seen.emplace(code_addr, kv.fetch_pc, kv.desc_v3).second)
                    std::fprintf(stderr,
                                 "[dynvb] PS code=0x%llx fetch pc=%u has zero-record V#; "
                                 "VADDR range is not derivable from draw vertices, preserving "
                                 "compatibility size=%u\n",
                                 (unsigned long long)code_addr, kv.fetch_pc, r.size);
            }
            if (kv.instruction_format == UINT32_MAX && d.format == DataFormat::Unknown) {
                static std::mutex unknown_mx;
                static std::set<std::tuple<uint64_t, bool, uint32_t, uint32_t>> unknown_seen;
                std::lock_guard<std::mutex> lk(unknown_mx);
                if (unknown_seen.emplace(code_addr, is_ps, kv.fetch_pc, kv.desc_v3).second)
                    std::fprintf(stderr,
                                 "[dynvb] %s code=0x%llx fetch pc=%u has unknown V# format 0x%x; "
                                 "using Float32x4 (draw_vertices=%u size=%u)\n",
                                 is_ps ? "PS" : "VS", (unsigned long long)code_addr,
                                 kv.fetch_pc, (kv.desc_v3 >> 12) & 0x7fu,
                                 draw_vertex_count, r.size);
            }
            t.resources.push_back(r);
        }
        // Descriptor-TABLE resources (#294): one ShaderResource per distinct table use, keyed by the
        // s_load immediate (srt_offset) so the recompiler's sreg_srt/by_srt_offset provenance resolves
        // the consuming image_sample / s_buffer_load. Never shadow an existing resource at the same
        // srt_offset (the EUD-sharp path may already have emitted it — first match wins in
        // by_srt_offset, and two DIFFERENT tables reusing one immediate would be ambiguous anyway).
        {
            std::set<uint64_t> srt_seen;
            for (const auto& u : srt_uses) {
                // Dedupe: a KEYED cbuf use per key (the s_buffer_load resolves by key); texture and
                // key-less buffer uses per CONSUMING INSTRUCTION (#273 — several image ops may share
                // one key, or have none; a key-less V# fetch resolves by its pc).
                // Distinct namespaces: pc keys must never collide with byte-offset keys.
                const bool exact_mtbuf = u.kind == 1 && u.instruction_format != UINT32_MAX;
                uint64_t dk = (u.kind == 0 || u.key == 0xFFFFFFFFu || exact_mtbuf)
                                  ? (0x8000000000000000ull | ((uint64_t)(uint32_t)u.kind << 32) | u.use_pc)
                                  : ((uint64_t)(uint32_t)u.kind << 32) | u.key;
                if (!srt_seen.insert(dk).second) continue;
                bool clash = exact_mtbuf || u.key == 0xFFFFFFFFu;
                if (!clash)
                    for (const auto& r0 : t.resources) if (r0.srt_offset == u.key) { clash = true; break; }
                if (u.kind == 1) {                       // constant buffer / structured-buffer V#
                    DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
                    if (u.instruction_format != UINT32_MAX &&
                        ((u.v4[3] >> 12) & 0x7Fu) == 0)
                        continue;
                    if (d.base <= 0x10000) continue;
                    uint32_t resource_size = d.size_bytes;
                    uint32_t resource_stride = d.stride;
                    if (resource_size == 0 || resource_size > 0x10000000u) {
                        // Scalar SMEM only needs V#.Base plus the consuming load's exact byte span.
                        // DOLL uses base-valid sharps whose other words do not describe a conventional
                        // bounded buffer. Cap the pc-keyed upload to the observed access; the renderer's
                        // safe copy preserves its usual zero-fill behavior for an unavailable guest page.
                        if (u.key != 0xFFFFFFFFu || u.required_size == 0 ||
                            u.required_size > (1u << 20)) continue;
                        resource_size = u.required_size;
                        resource_stride = 0;
                    } else if (resource_size < u.required_size) {
                        resource_size = u.required_size;
                    }
                    // A keyed use whose key already resolves keeps the existing resource; a key-less
                    // (or key-clashed) use still needs a pc-provenance entry — piggyback the pc onto
                    // an existing resource describing the SAME buffer, else create one (#273).
                    if (clash && !exact_mtbuf) {
                        bool piggybacked = false;
                        for (auto& r0 : t.resources)
                            if ((r0.cls == ResourceClass::ConstantBuffer || r0.cls == ResourceClass::VertexBuffer) &&
                                r0.gpu_addr == d.base && r0.size == resource_size && r0.stride == resource_stride) {
                                if (r0.fetch_pc == 0xFFFFFFFFu && r0.cls == ResourceClass::ConstantBuffer)
                                    r0.fetch_pc = u.use_pc;
                                piggybacked = r0.fetch_pc == u.use_pc || r0.cls == ResourceClass::VertexBuffer;
                                if (piggybacked) break;
                            }
                        if (piggybacked) continue;
                    }
                    ShaderResource r;
                    r.cls = ResourceClass::ConstantBuffer;
                    if (u.instruction_format != UINT32_MAX) {
                        rdna2_buffer_format(u.instruction_format, &r.format, &r.num_components);
                        if (r.format == DataFormat::Unknown || r.num_components == 0) continue;
                    } else {
                        r.format = d.format;
                        r.num_components = d.num_components ? d.num_components : 1;
                    }
                    r.gpu_addr = d.base; r.size = resource_size; r.stride = resource_stride;
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;
                    if (clash) r.fetch_pc = u.use_pc;    // pc-only provenance (key-less/collided V#)
                    if (log) fprintf(stderr, "[srt] %s cbuf key=0x%x pc=%u base=0x%llx size=%u\n", is_ps ? "PS" : "VS",
                                     u.key, u.use_pc, (unsigned long long)d.base, resource_size);
                    t.resources.push_back(r);
                } else {                                  // sampled texture or storage image (T# [+ paired S#])
                    DecodedImageDescriptor d = decode_image_descriptor(u.t8.data());
                    if (g_dyntrace_force)
                        fprintf(stderr, "[dynfail] tex use pc=%u key=0x%x base=0x%llx %ux%u fmt=%u tile=%u\n",
                                u.use_pc, u.key, (unsigned long long)d.base, d.width, d.height,
                                d.format, d.tile_mode);
                    if (d.base == 0 || d.width == 0 || d.height == 0 || d.depth == 0 ||
                        !valid_image_type(d.type) ||
                        d.base_array != 0 ||
                        d.width > 16384 || d.height > 16384) continue;       // garbage/degenerate T#
                    // A previous use already produced a resource for this SAME selected view (address +
                    // extent): don't duplicate the binding/upload — give it this use's pc provenance
                    // if it has none yet (#273). If it already carries a DIFFERENT use's pc, fall
                    // through and create a second resource for this pc (fetch_pc holds one pc; a
                    // sample whose pc has no mapping would stay unresolved).
                    const ResourceClass wanted = u.is_storage_image
                        ? ResourceClass::StorageImage : ResourceClass::Texture;
                    Gen5ImageFormatInfo fi;
                    if (!gen5_image_format(d.format, &fi)) {
                        if (wanted == ResourceClass::StorageImage) continue;
                        // Same policy as build_shader_resources: the normal per-target renderer can
                        // bind this as RGBA8 for RTT injection; legacy single-target mode skips it.
                        static const bool rtt_bind = getenv("PROSPER_RTT") != nullptr ||
                                                     getenv("PROSPER_RTT_PERTARGET") != nullptr;
                        if (!rtt_bind) continue;
                        fi.format = DataFormat::Unorm8; fi.num_components = 4; fi.bytes_per_block = 4;
                        fi.block_width = fi.block_height = 1; fi.srgb = false; fi.snorm = false;
                    }
                    const bool is_bcn = fi.block_width > 1;
                    if (is_bcn && fi.snorm) continue;   // signed BCn (SNORM / BC6H SF16): decode not wired
                    const DecodedImageView view = image_base_level_view(d, fi);
                    if (!view.supported) {
                        warn_unsupported_image_view(d);
                        continue;
                    }
                    const uint32_t img_dim = image_type_to_dim(d.type);
                    {
                        bool mapped = false;
                        for (auto& r0 : t.resources)
                            if (r0.cls == wanted && r0.gpu_addr == view.base &&
                                r0.width == view.width && r0.height == view.height &&
                                r0.depth == d.depth && r0.format == fi.format &&
                                r0.img_dim == img_dim && r0.sample_count == d.sample_count &&
                                r0.depth_compare == u.is_depth_compare &&
                                r0.in_mip_tail == view.in_mip_tail &&
                                r0.mip_tail_offset == (view.in_mip_tail ? view.mip_offset : 0) &&
                                r0.mip_tail_x == view.mip_tail_x &&
                                r0.mip_tail_y == view.mip_tail_y &&
                                r0.layer_stride_bytes == view.layer_stride &&
                                r0.layer_mip_offset_bytes == view.layer_mip_offset) {
                                if (r0.fetch_pc == 0xFFFFFFFFu) { r0.fetch_pc = u.use_pc; mapped = true; break; }
                                if (r0.fetch_pc == u.use_pc)    { mapped = true; break; }
                            }
                        if (mapped) continue;
                    }
                    ShaderResource r;
                    r.cls = wanted;
                    r.format = fi.format; r.num_components = fi.num_components;
                    r.gpu_addr = view.base; r.width = view.width; r.height = view.height; r.depth = d.depth;
                    r.sample_count = d.sample_count;
                    r.declared_mip_levels = d.sample_count > 1u ? 1u :
                        (d.last_level >= d.base_level ?
                            (uint32_t)(d.last_level - d.base_level) + 1u : 1u);
                    r.tile_mode = d.tile_mode; r.srgb = fi.srgb;
                    r.in_mip_tail = view.in_mip_tail;
                    r.mip_tail_offset = view.in_mip_tail
                        ? static_cast<uint32_t>(view.mip_offset) : 0;
                    r.mip_tail_bytes = view.mip_tail_bytes;
                    r.mip_tail_x = view.mip_tail_x;
                    r.mip_tail_y = view.mip_tail_y;
                    r.layer_stride_bytes = static_cast<uint32_t>(view.layer_stride);
                    r.layer_mip_offset_bytes = static_cast<uint32_t>(view.layer_mip_offset);
                    r.max_uncompressed_block_size = d.max_uncompressed_block_size;
                    r.max_compressed_block_size = d.max_compressed_block_size;
                    const bool shifted_view = view.mip_offset != 0 || view.in_mip_tail;
                    r.meta_pipe_aligned = shifted_view ? false : d.meta_pipe_aligned;
                    r.write_compress_enabled = shifted_view ? false : d.write_compress_enabled;
                    r.compression_enabled = shifted_view ? false : d.compression_enabled;
                    r.alpha_is_on_msb = d.alpha_is_on_msb;
                    r.color_transform = d.color_transform;
                    r.metadata_addr = shifted_view ? 0 : d.metadata_addr;
                    // T# TYPE -> MIMG dim (GFX10: 9=2D, 10=3D, 11=CUBE, 13=2D_ARRAY); a cube
                    // uploads as six vertically-stacked faces (#273 — see agc_shader_layout).
                    r.img_dim = img_dim;
                    r.depth_compare = u.is_depth_compare;
                    r.swizzle[0] = d.dst_sel[0]; r.swizzle[1] = d.dst_sel[1];
                    r.swizzle[2] = d.dst_sel[2]; r.swizzle[3] = d.dst_sel[3];
                    const uint64_t backing_bytes_per_sample = is_bcn
                        ? static_cast<uint64_t>((view.width + 3) / 4) * ((view.height + 3) / 4) * d.depth * fi.bytes_per_block
                        : static_cast<uint64_t>(view.width) * view.height * d.depth * fi.bytes_per_block;
                    if (!d.sample_count ||
                        backing_bytes_per_sample > UINT32_MAX / d.sample_count) continue;
                    const uint64_t backing_bytes = backing_bytes_per_sample * d.sample_count;
                    if (!backing_bytes || backing_bytes > UINT32_MAX) continue;
                    r.size = static_cast<uint32_t>(backing_bytes);
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;   // ambiguous/absent key: pc-only provenance
                    r.fetch_pc   = u.use_pc;                       // per-instruction provenance (#273)
                    if (wanted == ResourceClass::Texture && u.has_samp) {
                        // Paired S# (same SQ_IMG_SAMP decode as the sharp path). Storage operations do
                        // not consume a sampler even when their SSAMP bits alias known user SGPRs.
                        const uint32_t* sm = u.s4.data();
                        r.mag_filter  = ((sm[2] >> 20) & 0x3u) ? 1u : 0u;
                        r.min_filter  = ((sm[2] >> 22) & 0x3u) ? 1u : 0u;
                        r.mip_filter  = ((sm[2] >> 26) & 0x3u) ? 1u : 0u;
                        r.addr_uvw[0] = (sm[0] >> 0) & 0x7u;
                        r.addr_uvw[1] = (sm[0] >> 3) & 0x7u;
                        r.addr_uvw[2] = (sm[0] >> 6) & 0x7u;
                        r.max_aniso_ratio    = (sm[0] >> 9)  & 0x7u;
                        r.depth_compare_func = (sm[0] >> 12) & 0x7u;
                        r.unnormalized       = (sm[0] >> 15) & 0x1u;
                        r.min_lod            = (float)( sm[1]        & 0xFFFu) / 256.0f;
                        r.max_lod            = (float)((sm[1] >> 12) & 0xFFFu) / 256.0f;
                        int32_t bias14       = (int32_t)(sm[2] & 0x3FFFu);
                        if (bias14 & 0x2000) bias14 -= 0x4000;
                        r.lod_bias           = (float)bias14 / 256.0f;
                        r.border_color_type  = (sm[3] >> 30) & 0x3u;
                    }
                    if (log) fprintf(stderr, "[srt] %s %s key=0x%x %ux%u fmt=%u base=0x%llx tile=%u samp=%d\n",
                                     is_ps ? "PS" : "VS",
                                     wanted == ResourceClass::StorageImage ? "storage-image" : "tex",
                                     u.key, d.width, d.height, d.format,
                                     (unsigned long long)d.base, d.tile_mode,
                                     (int)(wanted == ResourceClass::Texture && u.has_samp));
                    t.resources.push_back(r);
                }
            }
        }
        // Metadata-described DIRECT buffers came from this exact candidate user-data range.
        // Dynamic-fold resources already carry either their proven primary seed or the explicit
        // ambiguous sentinel and are therefore never overwritten here.
        for (auto& resource : t.resources) {
            if (resource.direct_vsharp_sh_register_base !=
                    ShaderResource::kDirectVSharpOriginUnavailable ||
                resource.srt_offset != 0xFFFFFFFFu ||
                resource.sgpr_base == 0xFFFFFFFFu ||
                (resource.cls != ResourceClass::ConstantBuffer &&
                 resource.cls != ResourceClass::VertexBuffer))
                continue;
            if (resource.sgpr_base < user_sgpr_base ||
                resource.sgpr_base - user_sgpr_base > kUserSgprs - 4u) {
                resource.direct_vsharp_sh_register_base =
                    ShaderResource::kDirectVSharpOriginAmbiguous;
                continue;
            }
            resource.direct_vsharp_sh_register_base = base + range_start +
                (resource.sgpr_base - user_sgpr_base);
        }
        if (t.resources.empty()) continue;
        t.vertices_per_instance = is_ps ? 0u : draw_vertex_count;
        assign_convention_bindings(t, is_ps ? kPsBindingBase : 2u);
        if (log) {
            fprintf(stderr, "[restab] %s code=0x%llx base=0x%x -> %zu resources:\n",
                    is_ps ? "PS" : "VS", (unsigned long long)code_addr, base, t.resources.size());
            for (auto& r : t.resources) {
                fprintf(stderr, "[restab]   cls=%u binding=%u addr=0x%llx size=%u %ux%u fmt=%u stride=%u\n",
                        (unsigned)r.cls, r.binding, (unsigned long long)r.gpu_addr, r.size,
                        r.width, r.height, (unsigned)r.format, r.stride);
                if (r.cls == ResourceClass::ConstantBuffer && guest_readable(r.gpu_addr, 32)) {
                    const float* f = (const float*)(uintptr_t)r.gpu_addr;
                    fprintf(stderr, "[restab]     cbuf@0 floats: %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f\n",
                            f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
                    if (r.size >= 64 && guest_readable(r.gpu_addr, 64))
                        fprintf(stderr, "[restab]     cbuf@0 8..15:   %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f\n",
                                f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]);
                    if (r.size >= 0x150 && guest_readable(r.gpu_addr + 0x110, 64)) {
                        const float* g = (const float*)(uintptr_t)(r.gpu_addr + 0x110);
                        // Full 4x4 projection matrix at 0x110 (16 floats). Row 3's w element (g[15]) must be
                        // ~1.0 for clip.w to be nonzero — a zero here collapses the perspective divide.
                        fprintf(stderr, "[restab]     mtx@0x110 r0: %.4f %.4f %.4f %.4f\n", g[0], g[1], g[2], g[3]);
                        fprintf(stderr, "[restab]     mtx@0x110 r1: %.4f %.4f %.4f %.4f\n", g[4], g[5], g[6], g[7]);
                        fprintf(stderr, "[restab]     mtx@0x110 r2: %.4f %.4f %.4f %.4f\n", g[8], g[9], g[10], g[11]);
                        fprintf(stderr, "[restab]     mtx@0x110 r3: %.4f %.4f %.4f %.4f  <- g[15]=clip.w src\n", g[12], g[13], g[14], g[15]);
                    }
                }
            }
        }
        auto result = std::make_shared<ShaderResourceTable>(std::move(t));
        record_phases();
        return result;
    }
    if (log) fprintf(stderr, "[restab] %s code=0x%llx -> no resources in any user-data base\n",
                     is_ps ? "PS" : "VS", (unsigned long long)code_addr);
    record_phases();
    return nullptr;
}

bool validate_runtime_descriptor_contract(const char* stage_name,
                                           const std::vector<uint32_t>& spirv,
                                           const ShaderResourceTable* runtime,
                                           uint32_t expected_set,
                                           SpirvShaderStage expected_stage) {
    const char* mode = getenv("PROSPER_DESCRIPTOR_VALIDATE");
    if (!mode || !*mode || !strcmp(mode, "off") || !strcmp(mode, "0")) return true;

    DescriptorValidationReport report = validate_spirv_descriptor_interface(
        spirv, runtime, expected_set, expected_stage, true);
    const bool verbose = !strcmp(mode, "all");
    if (!report.issues.empty() || verbose) {
        uint64_t hash = 1469598103934665603ull;
        for (uint32_t word : spirv) { hash ^= word; hash *= 1099511628211ull; }
        static std::set<uint64_t> logged;
        uint64_t key = hash ^ (static_cast<uint64_t>(expected_set) << 56);
        auto mix = [&](uint64_t value) { key ^= value; key *= 1099511628211ull; };
        for (const auto& issue : report.issues) {
            mix(static_cast<uint32_t>(issue.code)); mix(issue.binding); mix(issue.set);
            mix(static_cast<uint32_t>(issue.actual));
            // Contract errors with different proven ranges are distinct. Warning-only unused
            // resources are not: their guest address/size can change every draw and must not flood
            // a long diagnostic run with the same module/binding warning.
            if (issue.error) { mix(issue.required_bytes); mix(issue.available_bytes); }
        }
        if (logged.insert(key).second) {
            fprintf(stderr, "[descriptor] %s module=%016llx used=%zu runtime=%zu result=%s mode=%s\n",
                    stage_name, (unsigned long long)hash, report.descriptors.size(),
                    runtime ? runtime->resources.size() : 0, report.ok() ? "accept" : "reject", mode);
            for (const auto& d : report.descriptors)
                fprintf(stderr, "[descriptor]   set=%u binding=%u type=%s required=%llu%s\n",
                        d.set, d.binding, spirv_descriptor_kind_name(d.kind),
                        (unsigned long long)d.required_bytes, d.dynamic_access ? "+dynamic" : "");
            for (const auto& issue : report.issues)
                fprintf(stderr, "[descriptor]   %s: %s set=%u binding=%u expected=%s actual=%s "
                                "required=%llu available=%llu\n",
                        issue.error ? "ERROR" : "warn", descriptor_issue_name(issue.code),
                        issue.set, issue.binding, spirv_descriptor_kind_name(issue.expected),
                        spirv_descriptor_kind_name(issue.actual),
                        (unsigned long long)issue.required_bytes,
                        (unsigned long long)issue.available_bytes);
            if (runtime) for (const auto& r : runtime->resources)
                fprintf(stderr, "[descriptor]   runtime binding=%u cls=%u addr=0x%llx size=%u "
                                "stride=%u fmt=%u comps=%u srt=0x%x sgpr=%u pc=%u\n",
                        r.binding, (unsigned)r.cls, (unsigned long long)r.gpu_addr, r.size,
                        r.stride, (unsigned)r.format, r.num_components,
                        r.srt_offset, r.sgpr_base, r.fetch_pc);
        }
    }
    return strcmp(mode, "strict") != 0 || report.ok();
}

ComputeLaunchDimensions resolve_compute_launch(const GpuState::Dispatch& d) {
    namespace P = prosper::agc::Pm4;
    ComputeLaunchDimensions out;
    const GpuState* ds = d.state.get();
    auto reg = [&](uint32_t off) {
        if (!ds) {
            return 0u;
        }
        auto it = ds->sh.find(off);
        return it == ds->sh.end() ? 0u : it->second;
    };
    // The workgroup local size is the NUM_THREAD_FULL field [15:0]; masking prevents a nonzero
    // NUM_THREAD_PARTIAL ([31:16]) from being folded into the dimension (#911).
    auto num_thread = [&](uint32_t off) {
        return (reg(off) >> P::COMPUTE_NUM_THREAD_FULL_SHIFT) & P::COMPUTE_NUM_THREAD_FULL_MASK;
    };
    out.local_x = num_thread(P::COMPUTE_NUM_THREAD_X);
    out.local_y = num_thread(P::COMPUTE_NUM_THREAD_Y);
    out.local_z = num_thread(P::COMPUTE_NUM_THREAD_Z);
    if (!out.local_x) out.local_x = 1;
    if (!out.local_y) out.local_y = 1;
    if (!out.local_z) out.local_z = 1;
    auto groups = [](uint32_t threads, uint32_t local) {
        return threads ? 1u + (threads - 1u) / local : 0u;
    };
    const bool use_thread_dimensions = ((d.modifier >>
        P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_SHIFT) &
        P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_MASK) != 0;
    if (use_thread_dimensions) {
        out.threads_x = d.threads_x;
        out.threads_y = d.threads_y;
        out.threads_z = d.threads_z;
        out.groups_x = groups(out.threads_x, out.local_x);
        out.groups_y = groups(out.threads_y, out.local_y);
        out.groups_z = groups(out.threads_z, out.local_z);
    } else {
        out.groups_x = d.threads_x;
        out.groups_y = d.threads_y;
        out.groups_z = d.threads_z;
        auto total_threads = [](uint32_t group_count, uint32_t local) {
            return static_cast<uint32_t>(std::min<uint64_t>(
                static_cast<uint64_t>(group_count) * local, UINT32_MAX));
        };
        out.threads_x = total_threads(out.groups_x, out.local_x);
        out.threads_y = total_threads(out.groups_y, out.local_y);
        out.threads_z = total_threads(out.groups_z, out.local_z);
    }
    // PROSPER_MAX_DISPATCH_GROUPS=N — DIAGNOSTIC ONLY, default off. Caps the workgroup count of every
    // dispatch. This deliberately computes WRONG results (a clamped kernel processes part of its
    // domain), so it is never a correctness or progression mode; it exists to test whether an
    // enormous dispatch is what loses the Vulkan device (#1742 -> #1743). Without it, Astro Bot
    // reaches 752,646 workgroups in one dispatch and the device is lost permanently a few submits
    // later; with it, the same route can be run to the same point with the size bounded.
    static const uint32_t group_cap = [] {
        const char* value = std::getenv("PROSPER_MAX_DISPATCH_GROUPS");
        return value && *value ? static_cast<uint32_t>(strtoul(value, nullptr, 0)) : 0u;
    }();
    if (group_cap) {
        auto clamp_groups = [&](uint32_t& groups, uint32_t& threads, uint32_t local) {
            if (groups <= group_cap) return;
            static std::atomic<int> reported{0};
            if (reported.fetch_add(1) < 16)
                std::fprintf(stderr, "[dispatch-cap] clamping %u workgroups to %u\n",
                             groups, group_cap);
            groups = group_cap;
            threads = static_cast<uint32_t>(std::min<uint64_t>(
                static_cast<uint64_t>(groups) * local, UINT32_MAX));
        };
        clamp_groups(out.groups_x, out.threads_x, out.local_x);
        clamp_groups(out.groups_y, out.threads_y, out.local_y);
        clamp_groups(out.groups_z, out.threads_z, out.local_z);
    }
    return out;
}

ComputeCpuFastPath classify_compute_cpu_fast_path(const uint32_t* code, size_t dwords) {
    // Compiler-generated 16-byte buffer fill:
    //   record = (tgid.x << 6) + tid.x; buffer[record] = {s4, s5, s6, s7}
    // This exact stream is used by Dead Cells and very frequently by Astro Bot. Keep the match
    // byte-exact so a different address calculation, descriptor, component mask, or predicate can
    // never be mistaken for a fill.
    static constexpr uint32_t kFillSgprUvec4[] = {
        0xd7460004u, 0x04010c08u, 0x7e000204u, 0x7e020205u, 0x7e040206u,
        0x7e060207u, 0xe01c2000u, 0x80000004u, 0xbf810000u,
    };
    if (code && dwords >= std::size(kFillSgprUvec4) &&
        std::equal(std::begin(kFillSgprUvec4), std::end(kFillSgprUvec4), code))
        return ComputeCpuFastPath::FillSgprUvec4;
    return ComputeCpuFastPath::None;
}

uint64_t compute_dispatch_code_addr(const GpuState& submit, const GpuState::Dispatch& dispatch) {
    namespace P = prosper::agc::Pm4;
    const GpuState& state = dispatch.state ? *dispatch.state : submit;
    auto reg = [&](uint32_t offset) {
        const auto it = state.sh.find(offset);
        return it == state.sh.end() ? 0u : it->second;
    };
    return (static_cast<uint64_t>(reg(P::COMPUTE_PGM_LO)) << 8) |
           (static_cast<uint64_t>(reg(P::COMPUTE_PGM_HI) & 0xffu) << 40);
}

std::vector<ComputeItem> realize_compute_dispatches(
    const GpuState& st, uint64_t submit_no,
    std::vector<OperationRealizationFailure>* failures) {
    if (failures) failures->clear();
    if (st.dispatches.empty()) return {};
    namespace P = prosper::agc::Pm4;
    auto rd = [](const std::unordered_map<uint32_t, uint32_t>& regs, uint32_t off) {
        auto it = regs.find(off);
        return it == regs.end() ? 0u : it->second;
    };

    std::vector<ComputeItem> items;
    items.reserve(st.dispatches.size());
    for (size_t dispatch_index = 0; dispatch_index < st.dispatches.size(); dispatch_index++) {
        const auto& dispatch = st.dispatches[dispatch_index];
        const GpuState& ds = dispatch.state ? *dispatch.state : st;
        const uint64_t code_addr = compute_dispatch_code_addr(st, dispatch);
        OperationRealizationFailure failure;
        failure.kind = SubmitOperationKind::Dispatch;
        failure.index = dispatch_index;
        failure.command_order = dispatch.command_order;
        failure.compute_launch = resolve_compute_launch(dispatch);
        auto record_failure = [&](RealizationFailureReason reason,
                                  const std::shared_ptr<ShaderResourceTable>& resources,
                                  const std::vector<uint32_t>& spirv,
                                  const ComputeShaderConfig* recompile_config = nullptr) {
            if (!failures) return;
            failure.reason = reason;
            ShaderRealizationDiagnostic stage;
            stage.stage = ShaderProgramStage::Compute;
            stage.program_addr = code_addr;
            stage.resources = resources;
            stage.recompiled = !spirv.empty();
            if (recompile_config) {
                stage.recompile_config = *recompile_config;
                stage.recompile_config_available = true;
            }
            if (!spirv.empty()) {
                const DescriptorValidationReport diagnostic_report =
                    validate_spirv_descriptor_interface(spirv, resources.get(), 0,
                                                        SpirvShaderStage::Compute, true);
                stage.descriptor_issue_count =
                    static_cast<uint32_t>(diagnostic_report.issues.size());
                auto issue = std::find_if(diagnostic_report.issues.begin(),
                                          diagnostic_report.issues.end(),
                                          [](const auto& candidate) { return candidate.error; });
                if (issue == diagnostic_report.issues.end() && !diagnostic_report.issues.empty())
                    issue = diagnostic_report.issues.begin();
                if (issue != diagnostic_report.issues.end())
                    stage.first_descriptor_issue = static_cast<uint32_t>(issue->code);
            }
            failure.stages.push_back(std::move(stage));
            failures->push_back(std::move(failure));
        };
        const auto* header = static_cast<const AgcShaderHeader*>(
            prosper_agc_shader_header_for_code(code_addr));
        if (!header || !code_addr || !guest_readable(code_addr, sizeof(uint32_t))) {
            record_failure(RealizationFailureReason::MissingProgram, {}, {});
            static std::set<uint64_t> logged;
            if (logged.insert(code_addr).second)
                std::fprintf(stderr, "[compute] skip unregistered/unreadable program 0x%llx\n",
                             (unsigned long long)code_addr);
            continue;
        }
        const size_t shader_dwords = registered_shader_dwords(*header, code_addr);

        uint32_t range_start = 0;
        if (header->specials && guest_readable((uint64_t)(uintptr_t)header->specials,
                                               sizeof(AgcShaderSpecials))) {
            const uint32_t start = header->specials->user_data_range_start;
            const uint32_t end = header->specials->user_data_range_end;
            if (start < kUserSgprs && end > start && end <= 2 * kUserSgprs) range_start = start;
        }
        uint32_t sgprs[kUserSgprs] = {};
        read_user_sgprs(ds.sh, P::COMPUTE_USER_DATA_0 + range_start, sgprs);
        const uint32_t rsrc2 = rd(ds.sh, P::COMPUTE_PGM_RSRC2);
        auto field = [&](uint32_t shift, uint32_t mask) { return (rsrc2 >> shift) & mask; };
        const uint32_t user_count = field(P::COMPUTE_PGM_RSRC2_USER_SGPR_SHIFT,
                                          P::COMPUTE_PGM_RSRC2_USER_SGPR_MASK);
        const uint32_t compute_wave_size = ((dispatch.modifier >>
            P::COMPUTE_DISPATCH_INITIATOR_CS_W32_EN_SHIFT) &
            P::COMPUTE_DISPATCH_INITIATOR_CS_W32_EN_MASK) ? 32u : 64u;
        const ComputeLaunchDimensions launch = resolve_compute_launch(dispatch);
        const bool tgid_x_en = field(P::COMPUTE_PGM_RSRC2_TGID_X_EN_SHIFT,
                                     P::COMPUTE_PGM_RSRC2_TGID_X_EN_MASK) != 0;
        const bool tgid_y_en = field(P::COMPUTE_PGM_RSRC2_TGID_Y_EN_SHIFT,
                                     P::COMPUTE_PGM_RSRC2_TGID_Y_EN_MASK) != 0;
        const bool tgid_z_en = field(P::COMPUTE_PGM_RSRC2_TGID_Z_EN_SHIFT,
                                     P::COMPUTE_PGM_RSRC2_TGID_Z_EN_MASK) != 0;
        // System SGPRs follow the user SGPR block. Only provide dispatch geometry to the bounded
        // linear-store recognizer when TGID.x is the only enabled group id and fits in the
        // push-constant SGPR block. Extra Y/Z groups merely repeat the exact same zero stores because
        // the proven kernel cannot observe them; all other oversized/formatless descriptors stay rejected.
        const bool linear_store_proof_context =
            user_count < kUserSgprs && tgid_x_en && !tgid_y_en && !tgid_z_en;
        auto table = std::make_shared<ShaderResourceTable>(
            build_shader_resources(*header, sgprs, kUserSgprs, 0));
        // Descriptor-TABLE uses (#590, mirroring the graphics fold in build_stage_table): UE4 compute
        // kernels s_load their V#/T# descriptors from tables pointed to by the user-data SGPRs and
        // consume them via s_buffer_load / image ops. build_shader_resources only sees the DIRECT
        // sharp/user-data layout, so those table descriptors were thrown away — the const-fold walk
        // recovers them here with the same keyed (srt_offset) / per-use (fetch_pc) provenance the
        // recompiler resolves. Compute-specific delta: an image_store use becomes a STORAGE image
        // (the recompiler's storage path requires ResourceClass::StorageImage), a sampled use a
        // Texture. Never shadow an existing resource at the same srt_offset (first-match-wins).
        {
            const std::vector<SrtUse> srt_uses = add_compute_buffer_resources(
                *table, (const uint32_t*)(uintptr_t)code_addr, shader_dwords,
                sgprs, kUserSgprs,
                linear_store_proof_context ? launch.local_x : 0,
                linear_store_proof_context ? launch.threads_x : 0,
                linear_store_proof_context ? user_count : UINT32_MAX);
            std::set<uint64_t> srt_seen;
            for (const auto& u : srt_uses) {
                if (u.kind != 0) continue;                 // buffers were materialized by the shared helper
                // Texture/storage uses are instruction-specific even when their table keys repeat.
                const uint64_t dk = 0x8000000000000000ull | u.use_pc;
                if (!srt_seen.insert(dk).second) continue;
                bool clash = u.key == 0xFFFFFFFFu;
                if (!clash)
                    for (const auto& r0 : table->resources)
                        if (r0.srt_offset == u.key) { clash = true; break; }
                {                                         // T# — storage image (store) or sampled texture
                    DecodedImageDescriptor d = decode_image_descriptor(u.t8.data());
                    if (d.base == 0 || d.width == 0 || d.height == 0 || d.depth == 0 ||
                        !valid_image_type(d.type) ||
                        d.base_array != 0 ||
                        d.width > 16384 || d.height > 16384) continue;   // garbage/degenerate T#
                    Gen5ImageFormatInfo fi;
                    const bool mapped_fmt = gen5_image_format(d.format, &fi);
                    // Unknown sampled formats cannot be decoded. Unknown storage formats may still
                    // recompile (format-free SPIR-V), but remain explicitly Unknown so the live backend
                    // rejects them instead of silently treating arbitrary bytes as RGBA8.
                    if (!mapped_fmt && !u.is_storage_image) continue;
                    if (mapped_fmt && fi.block_width > 1 && fi.snorm) continue;   // signed BCn: not wired
                    const DecodedImageView view = mapped_fmt
                        ? image_base_level_view(d, fi)
                        : DecodedImageView{d.base, d.width, d.height, 0, false, 0, 0, 0,
                                           0, 0, d.base_level == 0};
                    if (!view.supported) {
                        warn_unsupported_image_view(d);
                        continue;
                    }
                    const ResourceClass wanted = u.is_storage_image ? ResourceClass::StorageImage
                                                                   : ResourceClass::Texture;
                    const DataFormat view_format = mapped_fmt ? fi.format : DataFormat::Unknown;
                    const uint32_t img_dim = image_type_to_dim(d.type);
                    {
                        bool mapped = false;
                        for (auto& r0 : table->resources)
                            if (r0.cls == wanted && r0.gpu_addr == view.base &&
                                r0.width == view.width && r0.height == view.height &&
                                r0.depth == d.depth && r0.format == view_format &&
                                r0.img_dim == img_dim && r0.sample_count == d.sample_count &&
                                r0.depth_compare == u.is_depth_compare &&
                                r0.in_mip_tail == view.in_mip_tail &&
                                r0.mip_tail_offset == (view.in_mip_tail ? view.mip_offset : 0) &&
                                r0.mip_tail_x == view.mip_tail_x &&
                                r0.mip_tail_y == view.mip_tail_y &&
                                r0.layer_stride_bytes == view.layer_stride &&
                                r0.layer_mip_offset_bytes == view.layer_mip_offset) {
                                if (r0.fetch_pc == 0xFFFFFFFFu) { r0.fetch_pc = u.use_pc; mapped = true; break; }
                                if (r0.fetch_pc == u.use_pc)    { mapped = true; break; }
                            }
                        if (mapped) continue;
                    }
                    ShaderResource r;
                    r.cls = wanted;
                    if (mapped_fmt) {
                        r.format = fi.format; r.num_components = fi.num_components;
                        const bool is_bcn = fi.block_width > 1;
                        const uint64_t bytes_per_sample = is_bcn
                            ? static_cast<uint64_t>((view.width + 3) / 4) * ((view.height + 3) / 4) * d.depth * fi.bytes_per_block
                            : static_cast<uint64_t>(view.width) * view.height * d.depth * fi.bytes_per_block;
                        if (!d.sample_count || bytes_per_sample > UINT32_MAX / d.sample_count)
                            continue;
                        const uint64_t bytes = bytes_per_sample * d.sample_count;
                        if (!bytes || bytes > UINT32_MAX) continue;
                        r.size = static_cast<uint32_t>(bytes);
                        r.srgb = fi.srgb;
                    } else {
                        const uint64_t bytes_per_sample =
                            static_cast<uint64_t>(d.width) * d.height * d.depth * 4;
                        if (!d.sample_count || bytes_per_sample > UINT32_MAX / d.sample_count)
                            continue;
                        const uint64_t bytes = bytes_per_sample * d.sample_count;
                        if (!bytes || bytes > UINT32_MAX) continue;
                        r.format = DataFormat::Unknown; r.num_components = 4;
                        r.size = static_cast<uint32_t>(bytes);
                    }
                    r.gpu_addr = view.base; r.width = view.width; r.height = view.height; r.depth = d.depth;
                    r.sample_count = d.sample_count;
                    r.tile_mode = d.tile_mode;
                    r.declared_mip_levels = d.sample_count > 1u ? 1u :
                        (d.last_level >= d.base_level ?
                            (uint32_t)(d.last_level - d.base_level) + 1u : 1u);
                    r.in_mip_tail = view.in_mip_tail;
                    r.mip_tail_offset = view.in_mip_tail
                        ? static_cast<uint32_t>(view.mip_offset) : 0;
                    r.mip_tail_bytes = view.mip_tail_bytes;
                    r.mip_tail_x = view.mip_tail_x;
                    r.mip_tail_y = view.mip_tail_y;
                    r.layer_stride_bytes = static_cast<uint32_t>(view.layer_stride);
                    r.layer_mip_offset_bytes = static_cast<uint32_t>(view.layer_mip_offset);
                    r.max_uncompressed_block_size = d.max_uncompressed_block_size;
                    r.max_compressed_block_size = d.max_compressed_block_size;
                    const bool shifted_view = view.mip_offset != 0 || view.in_mip_tail;
                    r.meta_pipe_aligned = shifted_view ? false : d.meta_pipe_aligned;
                    r.write_compress_enabled = shifted_view ? false : d.write_compress_enabled;
                    r.compression_enabled = shifted_view ? false : d.compression_enabled;
                    r.alpha_is_on_msb = d.alpha_is_on_msb;
                    r.color_transform = d.color_transform;
                    r.metadata_addr = shifted_view ? 0 : d.metadata_addr;
                    r.img_dim = img_dim;
                    r.depth_compare = u.is_depth_compare;
                    r.swizzle[0] = d.dst_sel[0]; r.swizzle[1] = d.dst_sel[1];
                    r.swizzle[2] = d.dst_sel[2]; r.swizzle[3] = d.dst_sel[3];
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;
                    r.fetch_pc = u.use_pc;               // per-use pc provenance (the image op)
                    if (u.has_samp) {
                        const uint32_t* sm = u.s4.data();
                        r.mag_filter  = ((sm[2] >> 20) & 0x3u) ? 1u : 0u;
                        r.min_filter  = ((sm[2] >> 22) & 0x3u) ? 1u : 0u;
                        r.mip_filter  = ((sm[2] >> 26) & 0x3u) ? 1u : 0u;
                        r.addr_uvw[0] = (sm[0] >> 0) & 0x7u;
                        r.addr_uvw[1] = (sm[0] >> 3) & 0x7u;
                        r.addr_uvw[2] = (sm[0] >> 6) & 0x7u;
                        r.max_aniso_ratio    = (sm[0] >> 9) & 0x7u;
                        r.depth_compare_func = (sm[0] >> 12) & 0x7u;
                        r.unnormalized       = (sm[0] >> 15) & 0x1u;
                        r.min_lod            = static_cast<float>(sm[1] & 0xFFFu) / 256.0f;
                        r.max_lod            = static_cast<float>((sm[1] >> 12) & 0xFFFu) / 256.0f;
                        int32_t bias14        = static_cast<int32_t>(sm[2] & 0x3FFFu);
                        if (bias14 & 0x2000) bias14 -= 0x4000;
                        r.lod_bias           = static_cast<float>(bias14) / 256.0f;
                        r.border_color_type  = (sm[3] >> 30) & 0x3u;
                    }
                    table->resources.push_back(r);
                }
            }
        }
        // A direct compute "constant buffer" can actually be an SRT address pair consumed by
        // s_load_dword[xN]. Its adjacent SGPRs are not V# NUM_RECORDS/format words, so decoding them
        // as a four-dword V# can produce a nonsense one-byte bound. Size these pointer-backed tables
        // from the shader's immediate s_load accesses, but only when that exact guest range is mapped.
        for (auto& resource : table->resources) {
            if (resource.cls != ResourceClass::ConstantBuffer ||
                resource.sgpr_base == 0xFFFFFFFFu || !resource.gpu_addr)
                continue;
            const uint32_t required = rdna2_sload_required_bytes(
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                shader_dwords, resource.sgpr_base);
            if (required > resource.size && required <= 0x10000000u &&
                guest_readable(resource.gpu_addr, required)) {
                if (std::getenv("PROSPER_DBG"))
                    std::fprintf(stderr,
                                 "[compute-sload-range] s%u addr=0x%llx decoded=%u inferred=%u\n",
                                 resource.sgpr_base, (unsigned long long)resource.gpu_addr,
                                 resource.size, required);
                resource.size = required;
            }
        }
        // FLAT-window resources (#1171): a general flat_load reads a byte/dword from a raw 64-bit guest
        // pointer held in user SGPRs (not a V# descriptor, so build/add above never saw it). Resolve each
        // such load to its base SGPR pair, read the base pointer from THIS dispatch's user SGPRs, bind the
        // containing guest allocation as an SSBO, and key it by the load's pc so the recompiler lowers the
        // load to an indexed read at (address - base). Pushed before assign_convention_bindings so each
        // window gets a distinct binding; an unmapped base leaves the load unresolved (fail-visible).
        {
            // Bound the analysis by the ACTUAL user-SGPR count, not kUserSgprs (32). The emit indexes the
            // push-constant block, which is sized to config.user_sgprs.size() = min(user_count, kUserSgprs);
            // a base resolved in [user_count, 31] would make the emit's load_push_constant index past that
            // block (spirv-val would fail-visible, but keep the analysis and push-constant widths consistent).
            const uint32_t fw_user_count = std::min<uint32_t>(
                (rd(ds.sh, P::COMPUTE_PGM_RSRC2) >> P::COMPUTE_PGM_RSRC2_USER_SGPR_SHIFT) &
                    P::COMPUTE_PGM_RSRC2_USER_SGPR_MASK,
                kUserSgprs);
            const FlatLoadAnalysis fla = analyze_flat_loads(
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                shader_dwords, fw_user_count);
            // 256 MiB cap on the per-dispatch copy. TODO(#1183): the decode loop reads only base..base+N;
            // size the window from that access footprint (or the destination image extent) once the loop
            // executes, instead of the whole containing-allocation remainder.
            constexpr uint32_t kFlatWindowMax = 0x10000000u;
            for (const auto& fl : fla.loads) {
                if (fl.base_sgpr < 0 || static_cast<uint32_t>(fl.base_sgpr) + 1u >= fw_user_count) continue;
                const uint64_t base = static_cast<uint64_t>(sgprs[fl.base_sgpr]) |
                                      (static_cast<uint64_t>(sgprs[fl.base_sgpr + 1]) << 32);
                host::GuestReadableRange range{};
                if (!host::guest_readable_mapping_containing(base, base + 1, range) || range.end <= base)
                    continue;                                  // unmapped base -> stay fail-visible
                const uint64_t avail = range.end - base;
                ShaderResource w;
                w.cls = ResourceClass::ConstantBuffer;         // read-only SSBO (declare_cbufs + cbuf_load)
                w.gpu_addr = base;
                w.size = static_cast<uint32_t>(std::min<uint64_t>(avail, kFlatWindowMax));
                w.fetch_pc = fl.load_pc;
                w.flat_base_sgpr = static_cast<uint32_t>(fl.base_sgpr);
                table->resources.push_back(w);
                if (std::getenv("PROSPER_DBG"))
                    std::fprintf(stderr, "[flat-window] pc=%u base=s%d addr=0x%llx size=%u\n",
                                 fl.load_pc, fl.base_sgpr, (unsigned long long)base, w.size);
            }
        }
        bool native_multiwave_wave_work = false;
        {
            std::vector<Rdna2Inst> decoded;
            rdna2_walk(reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                       shader_dwords, decoded);
            // Keep dispatch-scoped resource discovery and translation on the same specialized
            // instruction stream. A proven-null BVH can collapse only the exact no-hit exit and a
            // fully matched empty-stack traversal cycle; shader-byte constant folding may then
            // remove any remaining unreachable arm.
            // Drop only instruction-scoped resources whose consumers disappeared. Direct resources
            // (fetch_pc == UINT32_MAX) remain available to every surviving use of their SGPR/SRT key.
            std::vector<Rdna2Inst> resource_paths = decoded;
            const ComputeResourcePathSpecializationReport path_report =
                specialize_compute_resource_paths(resource_paths, *table, compute_wave_size);
            if (std::getenv("PROSPER_DBG") && path_report.proven_null_exits) {
                std::fprintf(stderr,
                             "[compute-resource-specialization] code=0x%llx null-exits=%zu constants=%zu "
                             "removed-resources=%zu removed-pcs=",
                             static_cast<unsigned long long>(code_addr),
                             path_report.proven_null_exits,
                             path_report.shader_constant_branches,
                             path_report.removed_resources);
                for (size_t index = 0; index < path_report.removed_pcs.size(); ++index)
                    std::fprintf(stderr, "%s%u", index ? "," : "",
                                 path_report.removed_pcs[index]);
                std::fprintf(stderr, "\n");
            }
            assign_convention_bindings(*table, 2);
            native_multiwave_wave_work = compute_shader_prefers_native_multiwave(
                decoded, reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                shader_dwords);
            const bool uses_gds = std::any_of(decoded.begin(), decoded.end(), [](const auto& in) {
                return in.fmt == Rdna2Format::DS && in.ds_gds &&
                       (in.opcode == 0x0d || in.opcode == 0x3d || in.opcode == 0x3e);
            });
            if (uses_gds) {
                ShaderResource gds;
                gds.cls = ResourceClass::ConstantBuffer;
                gds.format = DataFormat::Uint32;
                gds.num_components = 1;
                gds.binding = kComputeInternalGdsBinding;
                gds.size = static_cast<uint32_t>(g_compute_gds.size());
                gds.stride = 4;
                gds.host_data = g_compute_gds.data();
                gds.host_data_size = g_compute_gds.size();
                table->resources.push_back(gds);
            }
        }

        ComputeShaderConfig config;
        config.user_sgprs.assign(sgprs, sgprs + std::min(user_count, kUserSgprs));
        config.local_x = launch.local_x;
        config.local_y = launch.local_y;
        config.local_z = launch.local_z;
        config.exact_thread_extent = ((dispatch.modifier >>
            P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_SHIFT) &
            P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_MASK) != 0;
        config.threads_x = launch.threads_x;
        config.threads_y = launch.threads_y;
        config.threads_z = launch.threads_z;
        config.wave_size = compute_wave_size;
        const SharedVulkanContext shared_vulkan = shared_vulkan_context();
        const bool shared_compute_adoptable = shared_vulkan.valid() &&
            shared_vulkan.compute_queue_supported &&
            shared_vulkan.storage_image_read_without_format &&
            shared_vulkan.storage_image_write_without_format;
        config.native_storage_format_support = shared_compute_adoptable
            ? shared_vulkan.native_storage_format_support : 0;
        const uint32_t replay_native_storage_format_support =
            config.native_storage_format_support;
        config.packed_r11_storage =
            std::getenv("PROSPER_NO_PACKED_R11_STORAGE") == nullptr;
        // Keep the stored capture module portable by compiling capture-bound dispatches through
        // device-independent storage paths (raw uvec4 or exact packed R32ui), so optional format
        // support never becomes an artifact ABI. Capture v39 also retains the raw shader and
        // semantic launch ABI, allowing --recompile-raw to reconstruct a device-specific module.
        static const bool timeline_capture_requested =
            std::getenv("PROSPER_GPU_TIMELINE_CAPTURE") != nullptr;
        static const bool timeline_capture_after_compute_gated =
            timeline_capture_requested && gpu_timeline_capture_is_after_compute_gated();
        const bool timeline_capture_after_compute_armed =
            timeline_capture_after_compute_gated &&
            gpu_timeline_capture_after_compute_gate_armed();
        const bool timeline_capture_bound = timeline_capture_requires_portable_compute(
            timeline_capture_requested, timeline_capture_after_compute_gated,
            timeline_capture_after_compute_armed);
        const bool capture_bound = std::getenv("PROSPER_GPU_CAPTURE") ||
            timeline_capture_bound ||
            interactive_gpu_capture_armed() || interactive_capture_bundle_active();
        if (capture_bound)
            config.native_storage_format_support = 0;
        // Full exact-size subgroups let the translator assign guest local coordinates in
        // SubgroupId/SubgroupLocalInvocationId order. That avoids assuming any relationship between
        // Vulkan's implementation-defined LocalInvocationIndex order and subgroup lane order while
        // still making each native subgroup exactly one RDNA wave. Capture v37 records this exact
        // module's required-subgroup/full-subgroups pipeline contract for faithful replay.
        // Repeated scratch-emulated wave scans/votes are a structural exception to the conservative
        // multi-wave default: the exact subgroup shell removes their workgroup barriers while
        // preserving one native subgroup per guest wave. Keep the environment switch as an explicit
        // experiment for every other multi-wave shape; this automatic path is shader-address/title
        // independent and remains subject to all device and workgroup bounds below.
        config.native_subgroup_size = select_native_compute_subgroup_size(
            shared_vulkan, config,
            native_multiwave_wave_work || getenv("PROSPER_NATIVE_COMPUTE_MULTIWAVE") != nullptr,
            getenv("PROSPER_NO_NATIVE_COMPUTE_SUBGROUP") != nullptr);
        config.tgid_x_en = tgid_x_en;
        config.tgid_y_en = tgid_y_en;
        config.tgid_z_en = tgid_z_en;
        config.tg_size_en = field(P::COMPUTE_PGM_RSRC2_TG_SIZE_EN_SHIFT,
                                  P::COMPUTE_PGM_RSRC2_TG_SIZE_EN_MASK) != 0;
        config.tidig_comp_cnt = field(P::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_SHIFT,
                                      P::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_MASK);
        config.lds_bytes = field(P::COMPUTE_PGM_RSRC2_LDS_SIZE_SHIFT,
                                 P::COMPUTE_PGM_RSRC2_LDS_SIZE_MASK) * 512u;

        ComputeItem item;
        item.spirv = recompile_compute_shader_cached(
            (const uint32_t*)(uintptr_t)code_addr, 0x10000, table.get(), config);
        item.user_sgprs = config.user_sgprs;
        item.required_subgroup_size = config.native_subgroup_size;
        item.cpu_fast_path = classify_compute_cpu_fast_path(
            reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)), shader_dwords);
        item.recompile_config = config;
        // Capture-bound SPIR-V deliberately uses the device-independent storage path, but v39 raw
        // replay needs the capability mask that a normal live dispatch would have used.
        item.recompile_config.native_storage_format_support =
            replay_native_storage_format_support;
        item.recompile_config_available = true;
        item.resources = std::move(table);
        item.launch = launch;
        item.code_addr = code_addr;
        item.dispatch_index = dispatch_index;
        item.submit_no = submit_no;
        item.command_order = dispatch.command_order;
        if (item.spirv.empty()) {
            record_failure(RealizationFailureReason::ShaderRecompile, item.resources, item.spirv,
                           &config);
            // PROSPER_DYNTRACE_FAIL=1: replay the FAILED compute program's resource build with the
            // const-fold walk trace + user-data dump forced on (once per distinct program) — the
            // compute analog of the graphics VS/PS fail-replay (gpu_execute.hpp). Compute dispatches
            // only run build_shader_resources (the DIRECT front-half table) and NEVER the const-fold
            // resolve_dynamic_fetch that graphics stages use, so a bindless storage-image U# / T#
            // (image_store SRSRC, image_sample) can't be recovered by pc-provenance for a dispatch.
            // This trace reveals whether the failing dispatch's descriptors ARE const-fold-resolvable
            // (i.e. loaded via a foldable s_load chain) — the ground truth for #590 before extending
            // the compute resource build. Read-only: it does not change what the executor binds.
            if (dyntrace_failed_shader_enabled(code_addr)) {
                static std::set<uint64_t> traced_cs;
                if (traced_cs.insert(code_addr).second) {
                    std::fprintf(stderr,
                                 "[dynfail] replaying PRE-SPECIALIZATION raw COMPUTE 0x%llx "
                                 "resource build with trace:\n",
                                 (unsigned long long)code_addr);
                    std::fprintf(stderr, "[dynfail]   compute user-data SGPRs (s0..s%u):\n", kUserSgprs - 1);
                    std::fprintf(stderr,
                                 "[dynfail]   launch groups=%ux%ux%u threads=%ux%ux%u local=%ux%ux%u "
                                 "user_sgprs=%u tgid=%u/%u/%u\n",
                                 launch.groups_x, launch.groups_y, launch.groups_z,
                                 launch.threads_x, launch.threads_y, launch.threads_z,
                                 launch.local_x, launch.local_y, launch.local_z, user_count,
                                 (unsigned)config.tgid_x_en, (unsigned)config.tgid_y_en,
                                 (unsigned)config.tgid_z_en);
                    for (uint32_t i = 0; i < kUserSgprs; i += 4)
                        std::fprintf(stderr, "[dynfail]     s%-2u: %08x %08x %08x %08x\n", i,
                                     sgprs[i], sgprs[i + 1], sgprs[i + 2], sgprs[i + 3]);
                    std::vector<SrtUse> cs_uses;
                    g_dyntrace_force = true;
                    resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, shader_dwords,
                                          sgprs, kUserSgprs, 0, &cs_uses);
                    g_dyntrace_force = false;
                    std::fprintf(stderr,
                                 "[dynfail]   pre-specialization raw const-fold recovered %zu "
                                 "descriptor use(s):\n",
                                 cs_uses.size());
                    for (const auto& u : cs_uses) {
                        if (u.kind == 0) {
                            std::fprintf(stderr,
                                         "[dynfail]     TEX/IMG(t8) key=0x%x use_pc=%u "
                                         "t8=%08x:%08x:%08x:%08x:%08x:%08x:%08x:%08x\n",
                                         u.key, u.use_pc, u.t8[0], u.t8[1], u.t8[2], u.t8[3],
                                         u.t8[4], u.t8[5], u.t8[6], u.t8[7]);
                        } else if (u.kind == 1) {
                            const DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
                            std::fprintf(stderr,
                                         "[dynfail]     BUF(v4) key=0x%x use_pc=%u "
                                         "v4=%08x:%08x:%08x:%08x base=0x%llx stride=%u "
                                         "records=%u size=%u required=%u\n",
                                         u.key, u.use_pc, u.v4[0], u.v4[1], u.v4[2], u.v4[3],
                                         (unsigned long long)d.base, d.stride, d.num_records,
                                         d.size_bytes, u.required_size);
                        } else {
                            const DecodedBvhDescriptor d = decode_bvh_descriptor(u.bvh4.data());
                            std::fprintf(stderr,
                                         "[dynfail]     BVH(bvh4) use_pc=%u "
                                         "bvh4=%08x:%08x:%08x:%08x base=0x%llx size=%llu "
                                         "type=%u tri_mode=%u box64=%u sort=%u grow=%u\n",
                                         u.use_pc, u.bvh4[0], u.bvh4[1], u.bvh4[2], u.bvh4[3],
                                         (unsigned long long)d.base,
                                         (unsigned long long)d.size_bytes, d.type,
                                         (unsigned)d.triangle_return_mode,
                                         (unsigned)d.box_node_64b, (unsigned)d.sort_enabled,
                                         (unsigned)d.box_grow);
                        }
                    }
                }
            }
            static std::set<uint64_t> logged;
            if (logged.insert(code_addr).second) {
                std::fprintf(stderr, "[compute] skip unsupported program 0x%llx\n",
                             (unsigned long long)code_addr);
                // PROSPER_SHADER_DUMP=<dir>: write the failed COMPUTE program's raw bytes for offline
                // shader_inspect, mirroring the graphics VS/PS dump (gpu_execute.hpp). The graphics
                // path was the only dumper, so a failing dispatch's CFG could not be mapped offline.
                // Keep the established 64 KiB diagnostic window: some compiler-generated branches
                // jump thousands of dwords forward even when the first rejection is near the entry.
                // Bound the raw fwrite to the bytes the decoder actually PROVED readable
                // (registered_shader_dwords is guest_readable-checked and capped at 0x4000 dwords ==
                // 64 KiB), not a fixed 0x10000 — a short shader at the tail of its mapping would
                // otherwise over-read past the mapped page into a SIGSEGV inside the dump (#1209).
                if (const char* dd = getenv("PROSPER_SHADER_DUMP")) {
                    const size_t dump_bytes = std::min(shader_dwords * sizeof(uint32_t), size_t(0x10000));
                    char fn[512];
                    snprintf(fn, sizeof fn, "%s/exec_cs_%llx.bin", dd,
                             (unsigned long long)code_addr);
                    if (FILE* f = fopen(fn, "wb")) {
                        fwrite(reinterpret_cast<const void*>(static_cast<uintptr_t>(code_addr)),
                               1, dump_bytes, f);
                        fclose(f);
                    } else if (getenv("PROSPER_DBG")) {
                        std::fprintf(stderr, "[shader-dump] cannot open %s: %s\n",
                                     fn, std::strerror(errno));
                    }
                }
            }
            continue;
        }
        const DescriptorValidationReport report = validate_spirv_descriptor_interface(
            item.spirv, item.resources.get(), 0, SpirvShaderStage::Compute, false);
        if (!report.ok()) {
            record_failure(RealizationFailureReason::DescriptorContract, item.resources, item.spirv,
                           &config);
            static std::set<uint64_t> logged;
            if (logged.insert(code_addr).second) {
                std::fprintf(stderr, "[compute] skip invalid descriptor contract for program 0x%llx\n",
                             (unsigned long long)code_addr);
                if (std::getenv("PROSPER_DBG")) for (const auto& issue : report.issues) {
                    std::fprintf(stderr,
                                 "[compute-descriptor] %s binding=%u expected=%s actual=%s required=%llu available=%llu error=%d\n",
                                 descriptor_issue_name(issue.code), issue.binding,
                                 spirv_descriptor_kind_name(issue.expected),
                                 spirv_descriptor_kind_name(issue.actual),
                                 (unsigned long long)issue.required_bytes,
                                 (unsigned long long)issue.available_bytes, issue.error ? 1 : 0);
                    if (item.resources) for (const auto& resource : item.resources->resources)
                        if (resource.binding == issue.binding)
                            std::fprintf(stderr,
                                         "[compute-resource] binding=%u class=%u addr=0x%llx size=%llu host=%zu stride=%u fmt=%u comps=%u dims=%ux%ux%u srt=0x%x sgpr=%u pc=%u\n",
                                         resource.binding, static_cast<unsigned>(resource.cls),
                                         (unsigned long long)resource.gpu_addr,
                                         (unsigned long long)resource.size, resource.host_data_size,
                                         resource.stride, static_cast<unsigned>(resource.format),
                                         resource.num_components, resource.width, resource.height,
                                         resource.depth, resource.srt_offset, resource.sgpr_base,
                                         resource.fetch_pc);
                }
            }
            continue;
        }
        // Image bindings (sampled textures + storage images) execute through the live backend's
        // image paths (#590, live_compute.cpp); shapes it cannot bind correctly are skipped there,
        // loudly and per-item, without aborting the rest of the batch.
        items.push_back(std::move(item));
    }
    return items;
}

bool execute_compute_dispatches(const GpuState& st, uint64_t submit_no) {
    if (!g_compute) return false;
    std::vector<ComputeItem> items = realize_compute_dispatches(st, submit_no);
    return !items.empty() && g_compute(items);
}

struct OrderedGpustateCaptureTrace {
    std::vector<DrawItem> draws;
    std::vector<ComputeItem> computes;
    std::vector<OperationRealizationFailure> failures;
    PendingGpuCapture* pending_capture = nullptr;
};

static OrderedSubmitResult execute_ordered_gpustate(
    const GpuState& st, uint32_t width, uint32_t height, uint64_t submit_no,
    const LiveRenderFn& render, const LiveComputeFn& compute,
    OrderedGpustateCaptureTrace* capture_trace = nullptr,
    const std::vector<DrawItem>* eager_draws = nullptr);

bool execute_nonrender_submit_work(const GpuState& st, uint64_t submit_no) {
    if (st.dma_copies.empty() && (!g_compute || st.dispatches.empty())) return false;
    GuestReadableSubmitScope guest_readable_scope;
    notify_compute_authority_unknown(
        ComputeAuthorityBoundaryKind::SubmitBegin, submit_no);
    // Compute-only submits never reach execute_ordered_and_present(), but they are exactly where an
    // unsupported dispatch can disappear before there is a realized ComputeItem to select.  Give
    // the environment capture path the same semantic pre-submit hook as rendering submits so
    // PROSPER_GPU_CAPTURE_COMPUTE_ADDR can retain the raw program, table snapshots, and explicit
    // realization failure.  Interactive/F9 captures remain draw-triggered: begin_requested_gpu_capture
    // deliberately does not consume an armed interactive request when semantic_draw_count is zero.
    std::unique_ptr<PendingGpuCapture> pending_capture;
    std::vector<SubmitOperation> capture_operations;
    const bool has_indirect_dispatch = std::any_of(
        st.dispatches.begin(), st.dispatches.end(),
        [](const GpuState::Dispatch& dispatch) { return dispatch.indirect; });
    // Preserve the established exact-trace path for ordinary no-DMA compute submits, and extend it
    // to DMA-backed indirect consumers whose arguments cannot be realized until after the copy.
    const bool can_defer_capture = st.dma_copies.empty() || has_indirect_dispatch;
    if (const char* capture_path = std::getenv("PROSPER_GPU_CAPTURE");
        capture_path && *capture_path) {
        capture_operations = plan_submit_operations(st);
        pending_capture = begin_requested_gpu_capture(
            {}, {}, capture_operations, present_width(), present_height(), &st, submit_no,
            static_cast<uint64_t>(st.draws.size()), nullptr, can_defer_capture);
    }
    snapshot_pending_gpu_capture_compute_gds(
        pending_capture.get(), g_compute_gds.data(), g_compute_gds.size());
    OrderedGpustateCaptureTrace capture_trace;
    capture_trace.pending_capture = pending_capture.get();
    const OrderedSubmitResult result = execute_ordered_gpustate(
        st, 0, 0, submit_no, {}, g_compute,
        pending_capture && can_defer_capture ? &capture_trace : nullptr);
    if (pending_capture) {
        std::string error;
        notify_compute_authority_unknown(
            ComputeAuthorityBoundaryKind::Capture, submit_no);
        if (!finish_requested_gpu_capture(
                std::move(pending_capture), {}, error,
                can_defer_capture ? &capture_trace.draws : nullptr,
                can_defer_capture ? &capture_trace.computes : nullptr,
                can_defer_capture ? &capture_operations : nullptr,
                can_defer_capture ? &st : nullptr,
                can_defer_capture ? &capture_trace.failures : nullptr))
            std::fprintf(stderr, "[gpucap] write failed: %s\n", error.c_str());
    }
    notify_compute_authority_unknown(
        ComputeAuthorityBoundaryKind::SubmitEnd, submit_no);
    return !st.dma_execution_rejected &&
           (result.compute_executed || !st.dma_copies.empty() ||
            !st.ordered_memory_effects.empty());
}

void diagnose_compute_dispatches(const GpuState& st, uint64_t submit_no) {
    const char* enabled = getenv("PROSPER_COMPUTELOG");
    const char* dim_env = getenv("PROSPER_COMPUTELOG_DIM");
    if ((!enabled || !*enabled) && (!dim_env || !*dim_env)) return;

    uint32_t want_w = 0, want_h = 0;
    if (dim_env && *dim_env && sscanf(dim_env, "%ux%u", &want_w, &want_h) != 2) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[compute] invalid PROSPER_COMPUTELOG_DIM='%s' (expected WxH)\n", dim_env);
        }
        want_w = want_h = 0;
    }

    namespace P = prosper::agc::Pm4;
    auto rd = [](const std::unordered_map<uint32_t, uint32_t>& regs, uint32_t off) {
        auto it = regs.find(off); return it == regs.end() ? 0u : it->second;
    };
    size_t matched = 0;
    for (size_t i = 0; i < st.dispatches.size(); ++i) {
        const auto& d = st.dispatches[i];
        const GpuState& ds = d.state ? *d.state : st;
        const ComputeLaunchDimensions launch = resolve_compute_launch(d);
        const uint64_t code_addr = compute_dispatch_code_addr(st, d);
        const auto* hdr = static_cast<const AgcShaderHeader*>(prosper_agc_shader_header_for_code(code_addr));

        uint32_t range_start = 0;
        if (hdr && hdr->specials && guest_readable((uint64_t)(uintptr_t)hdr->specials,
                                                   sizeof(AgcShaderSpecials))) {
            uint32_t s = hdr->specials->user_data_range_start;
            uint32_t e = hdr->specials->user_data_range_end;
            if (s < kUserSgprs && e > s && e <= 2 * kUserSgprs) range_start = s;
        }

        ShaderResourceTable table;
        uint32_t sgprs[kUserSgprs] = {};
        if (hdr) {
            read_user_sgprs(ds.sh, P::COMPUTE_USER_DATA_0 + range_start, sgprs);
            table = build_shader_resources(*hdr, sgprs, kUserSgprs, 0);
            assign_convention_bindings(table, 2);
        }

        // A compute shader can carry only an inline direct type-1 V# and no sharp descriptors. Dump
        // its metadata and bound SGPRs once per program if resource decoding still returns empty.
        // This turns the next unsupported layout into a reproducible decode problem instead of
        // another blind `resources=0` investigation.
        if (hdr && table.resources.empty() && enabled && *enabled) {
            static std::set<uint64_t> logged_empty;
            if (logged_empty.insert(code_addr).second) {
                const AgcShaderUserData* ud = hdr->user_data;
                fprintf(stderr, "[compute] empty-resource metadata code=0x%llx type=%u ud=%p",
                        (unsigned long long)code_addr, hdr->type, (const void*)ud);
                if (ud) {
                    fprintf(stderr, " eud=%u srt=%u direct_count=%u sharp={%u,%u,%u,%u}",
                            ud->eud_size_dw, ud->srt_size_dw, ud->direct_resource_count,
                            ud->sharp_resource_count[0], ud->sharp_resource_count[1],
                            ud->sharp_resource_count[2], ud->sharp_resource_count[3]);
                }
                fprintf(stderr, "\n[compute]   user_sgprs:");
                for (uint32_t s = 0; s < kUserSgprs; ++s) fprintf(stderr, " %08x", sgprs[s]);
                fprintf(stderr, "\n");

                if (ud && ud->direct_resource_offset && ud->direct_resource_count) {
                    fprintf(stderr, "[compute]   direct offsets:");
                    for (uint16_t t = 0; t < ud->direct_resource_count && t < 16; ++t)
                        fprintf(stderr, " [%u]=%u", t, ud->direct_resource_offset[t]);
                    fprintf(stderr, "\n");
                    const uint32_t reg = ud->direct_resource_count > 1 ? ud->direct_resource_offset[1] : 0xffffu;
                    if (reg != 0xffffu && reg + 4 <= kUserSgprs) {
                        const DecodedBufferDescriptor d = decode_buffer_descriptor(&sgprs[reg]);
                        fprintf(stderr, "[compute]   type1 V# reg=%u base=0x%llx stride=%u records=%u "
                                        "size=%u fmt=%u comps=%u\n",
                                reg, (unsigned long long)d.base, d.stride, d.num_records, d.size_bytes,
                                (unsigned)d.format, d.num_components);
                    }
                }
            }
        }

        bool dim_match = !want_w || !want_h;
        if (!dim_match) {
            for (const auto& r : table.resources)
                if (r.width == want_w && r.height == want_h) { dim_match = true; break; }
        }
        if (!dim_match) continue;
        matched++;

        uint64_t code_hash = 1469598103934665603ull;
        if (code_addr && guest_readable(code_addr, 4096)) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(code_addr));
            for (size_t n = 0; n < 4096; ++n) { code_hash ^= p[n]; code_hash *= 1099511628211ull; }
        } else {
            code_hash = 0;
        }
        fprintf(stderr,
                "[compute] submit=%llu dispatch=%zu threads=%ux%ux%u local=%ux%ux%u "
                "groups=%ux%ux%u modifier=0x%llx "
                "code=0x%llx hash4k=%016llx header=%s resources=%zu\n",
                (unsigned long long)submit_no, i,
                launch.threads_x, launch.threads_y, launch.threads_z,
                launch.local_x, launch.local_y, launch.local_z,
                launch.groups_x, launch.groups_y, launch.groups_z,
                (unsigned long long)d.modifier, (unsigned long long)code_addr,
                (unsigned long long)code_hash, hdr ? "yes" : "no", table.resources.size());
        for (const auto& r : table.resources) {
            fprintf(stderr,
                    "[compute]   cls=%u binding=%u addr=0x%llx size=%u dims=%ux%u "
                    "fmt=%u comps=%u tile=%u sgpr=%u srt=0x%x\n",
                    (unsigned)r.cls, r.binding, (unsigned long long)r.gpu_addr, r.size,
                    r.width, r.height, (unsigned)r.format, r.num_components, r.tile_mode,
                    r.sgpr_base, r.srt_offset);
        }
    }

    if (want_w && want_h && !st.dispatches.empty() && matched == 0 && enabled && enabled[0] == 'a') {
        fprintf(stderr, "[compute] submit=%llu dispatches=%zu: no resource matched %ux%u\n",
                (unsigned long long)submit_no, st.dispatches.size(), want_w, want_h);
    }
}

void diagnose_resource_provenance(const GpuState& st, uint64_t submit_no) {
    const char* dim_env = getenv("PROSPER_PROVENANCE_DIM");
    if (!dim_env || !*dim_env) return;

    uint32_t want_w = 0, want_h = 0;
    if (sscanf(dim_env, "%ux%u", &want_w, &want_h) != 2 || !want_w || !want_h) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[provenance] invalid PROSPER_PROVENANCE_DIM='%s' (expected WxH)\n", dim_env);
        }
        return;
    }
    const size_t min_draws = [] {
        const char* e = getenv("PROSPER_PROVENANCE_MIN_DRAWS");
        return e ? static_cast<size_t>(strtoull(e, nullptr, 0)) : size_t{0};
    }();

    struct ColorWrite {
        uint64_t submit = 0;
        uint64_t draw_submit = 0;
        size_t draw = 0;
        uint64_t vs = 0, ps = 0;
        uint32_t width = 0, height = 0;
        GpuState::Draw draw_record{};
    };
    static std::unordered_map<uint64_t, ColorWrite> last_color_write;
    static std::set<uint64_t> recorded_color_ranges;
    static uint64_t draw_submit_ordinal = 0;
    const uint64_t this_draw_submit = st.draws.empty() ? draw_submit_ordinal : draw_submit_ordinal++;

    const bool inspect_consumers = st.draws.size() >= min_draws;
    for (size_t i = 0; i < st.draws.size(); ++i) {
        const GpuState& ds = st.draws[i].state ? *st.draws[i].state : st;
        const RenderState rs = extract_render_state(ds);

        // Query before recording this draw's target: a feedback draw should resolve to the preceding
        // writer, not identify itself as its own producer.
        if (inspect_consumers && rs.ps_addr) {
            auto prt = build_stage_table(ds, rs.ps_addr, true, st.draws[i].index_count);
            if (prt) for (const auto& r : prt->resources) {
                if (r.width != want_w || r.height != want_h) continue;
                const uint64_t resource_size = r.size ? r.size :
                    static_cast<uint64_t>(r.width) * r.height * 4;
                auto writer = last_guest_write_overlap(r.gpu_addr, resource_size);
                if (writer) {
                    fprintf(stderr,
                            "[provenance]   latest-recorded-overlap kind=%s seq=%llu "
                            "range=[0x%llx,+0x%llx) "
                            "submit=%llu item=%llu order=%llu identity=0x%llx dims=%ux%u\n",
                            guest_writer_kind_name(writer->kind),
                            (unsigned long long)writer->sequence,
                            (unsigned long long)writer->addr,
                            (unsigned long long)writer->size,
                            (unsigned long long)writer->submit,
                            (unsigned long long)writer->item,
                            (unsigned long long)writer->order,
                            (unsigned long long)writer->identity,
                            writer->width, writer->height);
                } else {
                    fprintf(stderr,
                            "[provenance]   no recorded color/compute/DMA/WRITE_DATA overlap for "
                            "[0x%llx,+0x%llx)\n",
                            (unsigned long long)r.gpu_addr,
                            (unsigned long long)resource_size);
                }
                auto it = last_color_write.find(r.gpu_addr);
                if (it == last_color_write.end()) {
                    fprintf(stderr,
                            "[provenance] consumer submit=%llu draw=%zu ps=0x%llx samples "
                            "addr=0x%llx dims=%ux%u draw_submit=%llu order=%llu: "
                            "no prior color-target write\n",
                            (unsigned long long)submit_no, i, (unsigned long long)rs.ps_addr,
                            (unsigned long long)r.gpu_addr, r.width, r.height,
                            (unsigned long long)this_draw_submit,
                            (unsigned long long)st.draws[i].command_order);
                } else {
                    const ColorWrite& w = it->second;
                    fprintf(stderr,
                            "[provenance] consumer submit=%llu draw=%zu ps=0x%llx samples "
                            "addr=0x%llx dims=%ux%u draw_submit=%llu order=%llu: "
                            "last color write submit=%llu "
                            "draw_submit=%llu draw=%zu target_extent=%ux%u "
                            "vs=0x%llx ps=0x%llx\n",
                            (unsigned long long)submit_no, i, (unsigned long long)rs.ps_addr,
                            (unsigned long long)r.gpu_addr, r.width, r.height,
                            (unsigned long long)this_draw_submit,
                            (unsigned long long)st.draws[i].command_order,
                            (unsigned long long)w.submit,
                            (unsigned long long)w.draw_submit, w.draw, w.width, w.height,
                            (unsigned long long)w.vs, (unsigned long long)w.ps);
                    static std::set<uint64_t> probed;
                    if (probed.insert(r.gpu_addr).second && w.draw_record.state) {
                        DrawItem producer;
                        bool realized = realize_draw_item(*w.draw_record.state, &w.draw_record,
                                                         w.draw_record.index_count, 0x10000, true, producer);
                        fprintf(stderr,
                                "[provenance] producer-realize addr=0x%llx result=%s extent=%ux%u "
                                "items-target=0x%llx\n",
                                (unsigned long long)r.gpu_addr, realized ? "success" : "dropped",
                                producer.color0_width, producer.color0_height,
                                (unsigned long long)producer.color0_base);
                    }
                }
            }
        }

        if (rs.color0_base) {
            last_color_write[rs.color0_base] = {
                submit_no, this_draw_submit, i, rs.es_addr, rs.ps_addr,
                rs.color0_width, rs.color0_height, st.draws[i]
            };
            // The exact-address map above retains every latest color writer. The generic overlap
            // history needs only one representative event per target range; recording every draw
            // adds millions of mutex/hash operations during Dead Cells' submit-heavy startup.
            if (recorded_color_ranges.insert(rs.color0_base).second) {
                const uint64_t bytes = static_cast<uint64_t>(rs.color0_width) * rs.color0_height * 4;
                record_guest_write(GuestWriterKind::ColorTarget, rs.color0_base, bytes,
                                   submit_no, i, st.draws[i].command_order, rs.ps_addr,
                                   rs.color0_width, rs.color0_height);
            }
        }
        if (rs.color1_base) {
            last_color_write[rs.color1_base] = {
                submit_no, this_draw_submit, i, rs.es_addr, rs.ps_addr,
                rs.color1_width, rs.color1_height, st.draws[i]
            };
            if (recorded_color_ranges.insert(rs.color1_base).second) {
                const uint64_t bytes = static_cast<uint64_t>(rs.color1_width) * rs.color1_height * 4;
                record_guest_write(GuestWriterKind::ColorTarget, rs.color1_base, bytes,
                                   submit_no, i, st.draws[i].command_order, rs.ps_addr,
                                   rs.color1_width, rs.color1_height);
            }
        }
    }
}

std::vector<SubmitOperation> plan_submit_operations(const GpuState& st) {
    std::vector<SubmitOperation> operations;
    operations.reserve(st.draws.size() + st.dispatches.size() + st.dma_copies.size());
    for (size_t i = 0; i < st.draws.size(); ++i)
        operations.push_back({SubmitOperationKind::Draw, i, st.draws[i].command_order});
    for (size_t i = 0; i < st.dispatches.size(); ++i)
        operations.push_back({SubmitOperationKind::Dispatch, i, st.dispatches[i].command_order});
    for (size_t i = 0; i < st.dma_copies.size(); ++i)
        operations.push_back({SubmitOperationKind::DmaCopy, i, st.dma_copies[i].command_order});
    std::stable_sort(operations.begin(), operations.end(), [](const auto& a, const auto& b) {
        return a.command_order < b.command_order;
    });
    return operations;
}

namespace {

template <typename DmaCopyRecord, typename ExecuteDma>
OrderedSubmitResult execute_ordered_items_impl(const std::vector<SubmitOperation>& operations,
                                               const std::vector<DrawItem>& draws,
                                               const std::vector<ComputeItem>& computes,
                                               const std::vector<DmaCopyRecord>& dma_copies,
                                               const LiveRenderFn& render,
                                               const LiveComputeFn& compute,
                                               uint32_t width, uint32_t height,
                                               ExecuteDma&& execute_dma) {
    GuestGpuWriteSubmitScope guest_gpu_write_scope;
    std::unordered_map<size_t, size_t> draw_by_index, compute_by_index;
    for (size_t i = 0; i < draws.size(); ++i) draw_by_index[draws[i].draw_index] = i;
    for (size_t i = 0; i < computes.size(); ++i)
        compute_by_index[static_cast<size_t>(computes[i].dispatch_index)] = i;

    enum class ExecutableKind : uint8_t { Draw, Dispatch, DmaCopy };
    struct ExecutableOperation {
        ExecutableKind kind;
        size_t item;
        uint64_t command_order;
    };
    std::vector<ExecutableOperation> executable;
    bool explicit_dma_operations = false;
    for (const auto& operation : operations) {
        if (operation.kind == SubmitOperationKind::Draw) {
            auto it = draw_by_index.find(operation.index);
            if (it != draw_by_index.end())
                executable.push_back({ExecutableKind::Draw, it->second, operation.command_order});
        } else if (operation.kind == SubmitOperationKind::Dispatch) {
            auto it = compute_by_index.find(operation.index);
            if (it != compute_by_index.end())
                executable.push_back({ExecutableKind::Dispatch, it->second, operation.command_order});
        } else {
            explicit_dma_operations = true;
            if (operation.index < dma_copies.size())
                executable.push_back({ExecutableKind::DmaCopy, operation.index,
                                      operation.command_order});
        }
    }
    // Compatibility for pre-v14 callers whose operation list predates the DMA kind.
    if (!explicit_dma_operations)
        for (size_t i = 0; i < dma_copies.size(); ++i)
            executable.push_back({ExecutableKind::DmaCopy, i, dma_copies[i].command_order});
    std::stable_sort(executable.begin(), executable.end(), [](const auto& a, const auto& b) {
        return a.command_order < b.command_order;
    });

    size_t total_spans = 0;
    bool in_draw_span = false;
    for (const auto& operation : executable) {
        if (operation.kind == ExecutableKind::Draw) {
            if (!in_draw_span) ++total_spans;
            in_draw_span = true;
        } else {
            in_draw_span = false;
        }
    }

    OrderedSubmitResult result;
    std::vector<DrawItem> span;
    auto flush_span = [&](bool authoritative_readback = false) {
        if (span.empty() || !render) return;
        LiveRenderPhase saved = g_live_phase;
        g_live_phase = {result.render_spans == 0, result.render_spans + 1 == total_spans,
                        authoritative_readback};
        RenderedFrame rendered = render(span, width, height);
        g_live_phase = saved;
        if (!rendered.empty()) result.frame = std::move(rendered);
        span.clear();
        ++result.render_spans;
    };
    for (const auto& operation : executable) {
        if (operation.kind == ExecutableKind::Draw) {
            span.push_back(draws[operation.item]);
        } else if (operation.kind == ExecutableKind::Dispatch) {
            flush_span();
            result.compute_executed |= compute && compute({computes[operation.item]});
        } else {
            flush_span(true);
            execute_dma(dma_copies[operation.item]);
        }
    }
    flush_span();
    return result;
}

} // namespace

OrderedSubmitResult execute_ordered_items(const std::vector<SubmitOperation>& operations,
                                          const std::vector<DrawItem>& draws,
                                          const std::vector<ComputeItem>& computes,
                                          const std::vector<GpuState::DmaCopy>& dma_copies,
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height) {
    return execute_ordered_items_impl(
        operations, draws, computes, dma_copies, render, compute, width, height,
        [](const GpuState::DmaCopy& copy) {
            std::vector<uint8_t> current_source;
            const LiveTargetByteReadResult source_result = read_live_render_target_bytes(
                copy.src, copy.bytes, current_source);
            if (source_result == LiveTargetByteReadResult::InvalidRange) {
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    std::fprintf(stderr,
                                 "[agc] DMA_DATA live-target source range invalid: src=0x%llx bytes=%u\n",
                                 static_cast<unsigned long long>(copy.src), copy.bytes);
                return;
            }
            execute_ordered_dma_copy(
                copy, source_result == LiveTargetByteReadResult::Success
                          ? current_source.data() : nullptr);
        });
}

OrderedSubmitResult execute_ordered_items(const std::vector<SubmitOperation>& operations,
                                          const std::vector<DrawItem>& draws,
                                          const std::vector<ComputeItem>& computes,
                                          const std::vector<ReplayDmaCopy>& dma_copies,
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height) {
    return execute_ordered_items_impl(
        operations, draws, computes, dma_copies, render, compute, width, height,
        [](const ReplayDmaCopy& copy) {
            const uint32_t source_selector = (copy.sels >> 8u) & 0xffu;
            const uint32_t destination_selector = copy.sels & 0xffu;
            const bool source_gds = source_selector == 1u;
            const bool destination_gds = destination_selector == 1u;
            if (source_gds) {
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    std::fprintf(stderr,
                                 "[gpu_replay] DMA_DATA GDS source is unsupported: "
                                 "src=0x%llx dst=0x%llx bytes=%u sels=0x%x\n",
                                 static_cast<unsigned long long>(copy.src),
                                 static_cast<unsigned long long>(copy.dst), copy.bytes,
                                 copy.sels);
                return;
            }
            const bool address_source = (copy.sels & kDmaDataAddressSource) != 0 ||
                                        copy.src > UINT32_MAX;
            if (destination_gds && (source_selector != 3u || !address_source)) {
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    std::fprintf(stderr,
                                 "[gpu_replay] DMA_DATA memory-to-GDS selector form is "
                                 "unsupported: src=0x%llx dst=0x%llx bytes=%u sels=0x%x\n",
                                 static_cast<unsigned long long>(copy.src),
                                 static_cast<unsigned long long>(copy.dst), copy.bytes,
                                 copy.sels);
                return;
            }
            std::vector<uint8_t> current_source;
            const LiveTargetByteReadResult source_result = read_live_render_target_bytes(
                copy.src, copy.bytes, current_source);
            if (source_result == LiveTargetByteReadResult::InvalidRange) {
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    std::fprintf(stderr,
                                 "[gpu_replay] DMA_DATA live-target source range invalid: src=0x%llx bytes=%u\n",
                                 static_cast<unsigned long long>(copy.src), copy.bytes);
                return;
            }
            const uint8_t* source = source_result == LiveTargetByteReadResult::Success
                ? current_source.data() : copy.source_data;
            const uint64_t source_size = source_result == LiveTargetByteReadResult::Success
                ? current_source.size() : copy.source_size;
            if (!copy.destination_data || !source ||
                copy.bytes > copy.destination_size || copy.bytes > source_size)
                return;
            std::memmove(copy.destination_data, source, copy.bytes);
            if (!destination_gds) notify_guest_gpu_write(copy.dst, copy.bytes);
        });
}

OrderedSubmitResult execute_ordered_items(const std::vector<SubmitOperation>& operations,
                                          const std::vector<DrawItem>& draws,
                                          const std::vector<ComputeItem>& computes,
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height) {
    return execute_ordered_items(operations, draws, computes,
                                 std::vector<GpuState::DmaCopy>{},
                                 render, compute, width, height);
}

namespace {

struct DrawRealizationBatch {
    using Callback = void (*)(void*, size_t);
    using DrawCallback = void (*)(void*, size_t, size_t);

    void* context = nullptr;
    Callback begin = nullptr;
    DrawCallback draw = nullptr;
    Callback end = nullptr;
    size_t count = 0;
    size_t worker_participants = 0;
    std::atomic<size_t> next{0};
    std::atomic<size_t> remaining{0};
    std::mutex done_mutex;
    std::condition_variable done_cv;
    std::mutex error_mutex;
    std::exception_ptr error;
};

// One process-lifetime worker set avoids creating dozens of host threads for every guest submit.
// The pool deliberately supports one batch at a time; callers are serialized at run() even when
// independent submit threads arrive concurrently. The submit thread participates as slot zero;
// each persistent worker owns a stable measurement slot.
class DrawRealizationPool {
public:
    explicit DrawRealizationPool(size_t workers) {
        workers_.reserve(workers);
        try {
            for (size_t i = 0; i < workers; ++i)
                workers_.emplace_back([this, i] { worker_main(i); });
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
                ++generation_;
            }
            work_cv_.notify_all();
            for (auto& worker : workers_) if (worker.joinable()) worker.join();
            throw;
        }
    }

    ~DrawRealizationPool() {
        std::lock_guard run_lock(run_mutex_);
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        work_cv_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }

    size_t worker_count() const { return workers_.size(); }

    void run(const std::shared_ptr<DrawRealizationBatch>& batch, size_t workers_to_use) {
        std::lock_guard run_lock(run_mutex_);
        workers_to_use = std::min(workers_to_use, workers_.size());
        batch->worker_participants = workers_to_use;
        batch->remaining.store(workers_to_use, std::memory_order_relaxed);
        {
            std::lock_guard lock(mutex_);
            active_ = batch;
            ++generation_;
        }
        work_cv_.notify_all();

        run_participant(batch, 0);
        if (workers_to_use) {
            std::unique_lock lock(batch->done_mutex);
            batch->done_cv.wait(lock, [&] {
                return batch->remaining.load(std::memory_order_acquire) == 0;
            });
        }
        // Do not retain the caller-owned context through the process-lifetime pool. A worker that
        // woke for this generation but was not selected may still hold its own shared batch reference;
        // clearing the pool reference is safe after every selected worker has completed.
        {
            std::lock_guard lock(mutex_);
            if (active_ == batch) active_.reset();
        }
        if (batch->error) std::rethrow_exception(batch->error);
    }

private:
    static void remember_error(const std::shared_ptr<DrawRealizationBatch>& batch) {
        std::lock_guard lock(batch->error_mutex);
        if (!batch->error) batch->error = std::current_exception();
    }

    static void run_participant(const std::shared_ptr<DrawRealizationBatch>& batch,
                                size_t measurement_slot) {
        bool began = false;
        try {
            if (batch->begin) batch->begin(batch->context, measurement_slot);
            began = true;
            constexpr size_t kGrain = 4;
            for (;;) {
                const size_t first = batch->next.fetch_add(kGrain, std::memory_order_relaxed);
                if (first >= batch->count) break;
                const size_t last = std::min(first + kGrain, batch->count);
                for (size_t draw = first; draw < last; ++draw)
                    batch->draw(batch->context, draw, measurement_slot);
            }
        } catch (...) {
            remember_error(batch);
        }
        if (began && batch->end) {
            try { batch->end(batch->context, measurement_slot); }
            catch (...) { remember_error(batch); }
        }
    }

    void worker_main(size_t worker_index) {
        uint64_t seen_generation = 0;
        for (;;) {
            std::shared_ptr<DrawRealizationBatch> batch;
            {
                std::unique_lock lock(mutex_);
                work_cv_.wait(lock, [&] {
                    return stopping_ || generation_ != seen_generation;
                });
                if (stopping_) return;
                seen_generation = generation_;
                batch = active_;
            }
            if (!batch || worker_index >= batch->worker_participants) continue;
            run_participant(batch, worker_index + 1);
            if (batch->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard lock(batch->done_mutex);
                batch->done_cv.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex run_mutex_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::shared_ptr<DrawRealizationBatch> active_;
    uint64_t generation_ = 0;
    bool stopping_ = false;
};

size_t configured_draw_realization_threads() {
    static const size_t threads = [] {
        const unsigned hardware = std::thread::hardware_concurrency();
        size_t result = std::min<size_t>(hardware ? hardware : 4u, 8u);
        if (const char* value = std::getenv("PROSPER_DRAW_REALIZE_THREADS")) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);
            if (end != value && *end == '\0') result = std::clamp<size_t>(parsed, 1u, 32u);
        }
        return std::max<size_t>(result, 1u);
    }();
    return threads;
}

size_t parallel_draw_minimum() {
    static const size_t minimum = [] {
        size_t result = 32;
        if (const char* value = std::getenv("PROSPER_DRAW_REALIZE_MIN_DRAWS")) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);
            if (end != value && *end == '\0') result = std::clamp<size_t>(parsed, 2u, 4096u);
        }
        return result;
    }();
    return minimum;
}

bool parallel_draw_diagnostic_active(bool log) {
    if (log || std::getenv("PROSPER_SERIAL_DRAW_REALIZE")) return true;
    static constexpr const char* diagnostics[] = {
        "PROSPER_RTLOG", "PROSPER_INTERPLOG", "PROSPER_RESDUMP",
        "PROSPER_DYNTRACE_FAIL", "PROSPER_DRAWDIAG", "PROSPER_ONLY_ATLAS",
        "PROSPER_CAPTION_DIAG", "PROSPER_VS_DUMP", "PROSPER_SHADER_DUMP",
        "PROSPER_SHADER_DUMP_SUCCESS", "PROSPER_DRAWLOG", "PROSPER_DBG",
        "PROSPER_STAGE_FOLD_PROFILE", "PROSPER_DESCRIPTOR_VALIDATE",
        "PROSPER_NO_SHADER_CACHE",
    };
    for (const char* name : diagnostics)
        if (std::getenv(name)) return true;
    return false;
}

DrawRealizationPool& draw_realization_pool() {
    static DrawRealizationPool pool(configured_draw_realization_threads() - 1);
    return pool;
}

struct ParallelDrawSlot {
    DrawItem item;
    bool realized = false;
};

struct ParallelWorkerMeasurement {
    DrawRealizationPhaseStats draw_before;
    StageTablePhaseStats table_before;
    DrawRealizationPhaseStats draw_delta;
    StageTablePhaseStats table_delta;
    uint64_t readable_calls = 0;
    uint64_t readable_hits = 0;
    uint64_t readable_os_probes = 0;
    std::unique_ptr<GuestReadableSubmitScope> readable_scope;
};

struct ParallelDrawContext {
    const GpuState* state = nullptr;
    uint32_t max_shader_dwords = 0;
    bool retain_shared_shader_words = false;
    bool parent_readable_active = false;
    std::vector<ParallelDrawSlot> slots;
    std::vector<ParallelWorkerMeasurement> measurements;
};

void parallel_draw_worker_begin(void* opaque, size_t worker_slot) {
    auto& context = *static_cast<ParallelDrawContext*>(opaque);
    auto& measurement = context.measurements[worker_slot];
    if (worker_slot != 0)
        measurement.readable_scope = std::make_unique<GuestReadableSubmitScope>();
    measurement.draw_before = draw_realization_phase_stats();
    measurement.table_before = stage_table_phase_stats();
}

void parallel_draw_worker_execute(void* opaque, size_t draw_index, size_t) {
    auto& context = *static_cast<ParallelDrawContext*>(opaque);
    const GpuState& state = *context.state;
    const GpuState::Draw& draw = state.draws[draw_index];
    ParallelDrawSlot& slot = context.slots[draw_index];
    slot.realized = realize_draw_item(
        state.state_at_draw(draw_index), &draw, draw.index_count,
        context.max_shader_dwords, false, slot.item, nullptr,
        context.retain_shared_shader_words);
    if (slot.realized) {
        slot.item.draw_index = draw_index;
        slot.item.command_order = draw.command_order;
    }
}

void parallel_draw_worker_end(void* opaque, size_t worker_slot) {
    auto& context = *static_cast<ParallelDrawContext*>(opaque);
    auto& measurement = context.measurements[worker_slot];
    const DrawRealizationPhaseStats draw_after = draw_realization_phase_stats();
    const StageTablePhaseStats table_after = stage_table_phase_stats();
    measurement.draw_delta = {
        draw_after.draws - measurement.draw_before.draws,
        draw_after.table_ms - measurement.draw_before.table_ms,
        draw_after.shader_ms - measurement.draw_before.shader_ms,
    };
    measurement.table_delta = {
        table_after.calls - measurement.table_before.calls,
        table_after.metadata_ms - measurement.table_before.metadata_ms,
        table_after.dynamic_fold_ms - measurement.table_before.dynamic_fold_ms,
        table_after.resources_ms - measurement.table_before.resources_ms,
    };
    if (worker_slot != 0) {
        measurement.readable_calls = g_guest_readable_cache.calls;
        measurement.readable_hits = g_guest_readable_cache.hits;
        measurement.readable_os_probes = g_guest_readable_cache.os_probes;
        measurement.readable_scope.reset();
    }
}

struct ParallelDrawStatsState {
    std::mutex mutex;
    ParallelDrawRealizationStats totals;
};

ParallelDrawStatsState& parallel_draw_stats_state() {
    static ParallelDrawStatsState state;
    return state;
}

} // namespace

ParallelDrawRealizationStats parallel_draw_realization_stats() {
    auto& state = parallel_draw_stats_state();
    std::lock_guard lock(state.mutex);
    return state.totals;
}

std::vector<DrawItem> realize_gpustate_draws_parallel(
        const GpuState& st, uint32_t max_shader_dwords, bool log,
        bool retain_shared_shader_words, bool* attempted) {
    if (attempted) *attempted = false;
    const size_t thread_count = configured_draw_realization_threads();
    if (thread_count < 2 || st.draws.size() < parallel_draw_minimum() ||
        parallel_draw_diagnostic_active(log))
        return {};
    if (attempted) *attempted = true;

    DrawRealizationPool& pool = draw_realization_pool();
    const size_t worker_count = std::min(pool.worker_count(), thread_count - 1);
    ParallelDrawContext context;
    context.state = &st;
    context.max_shader_dwords = max_shader_dwords;
    context.retain_shared_shader_words = retain_shared_shader_words;
    context.parent_readable_active = g_guest_readable_cache.active;
    context.slots.resize(st.draws.size());
    context.measurements.resize(worker_count + 1);

    auto batch = std::make_shared<DrawRealizationBatch>();
    batch->context = &context;
    batch->begin = parallel_draw_worker_begin;
    batch->draw = parallel_draw_worker_execute;
    batch->end = parallel_draw_worker_end;
    batch->count = st.draws.size();
    const auto begin = std::chrono::steady_clock::now();
    pool.run(batch, worker_count);
    const auto end = std::chrono::steady_clock::now();

    // The parent thread's phase/readability counters feed the existing submit timing report. Worker
    // thread-local deltas must be merged explicitly so parallel work is not mislabeled as "other".
    for (size_t i = 1; i < context.measurements.size(); ++i) {
        const auto& measurement = context.measurements[i];
        g_draw_realization_phases.draws += measurement.draw_delta.draws;
        g_draw_realization_phases.table_ms += measurement.draw_delta.table_ms;
        g_draw_realization_phases.shader_ms += measurement.draw_delta.shader_ms;
        g_stage_table_phases.calls += measurement.table_delta.calls;
        g_stage_table_phases.metadata_ms += measurement.table_delta.metadata_ms;
        g_stage_table_phases.dynamic_fold_ms += measurement.table_delta.dynamic_fold_ms;
        g_stage_table_phases.resources_ms += measurement.table_delta.resources_ms;
        if (context.parent_readable_active) {
            g_guest_readable_cache.calls += measurement.readable_calls;
            g_guest_readable_cache.hits += measurement.readable_hits;
            g_guest_readable_cache.os_probes += measurement.readable_os_probes;
        }
    }

    {
        auto& stats = parallel_draw_stats_state();
        std::lock_guard lock(stats.mutex);
        ++stats.totals.batches;
        stats.totals.semantic_draws += st.draws.size();
        stats.totals.worker_threads += worker_count + 1;
        stats.totals.wall_ms +=
            std::chrono::duration<double, std::milli>(end - begin).count();
    }

    std::vector<DrawItem> items;
    items.reserve(st.draws.size());
    for (auto& slot : context.slots)
        if (slot.realized) items.push_back(std::move(slot.item));
    return items;
}

namespace {
enum class RetainedSubmitKind : uint8_t { Draw, Dispatch, DmaCopy, ParserStall, MemoryEffect };
struct RetainedSubmitOperation {
    RetainedSubmitKind kind;
    size_t index;
    uint64_t command_order;
};

bool use_per_draw_policy(const GpuState& st) {
    static const bool force_perdraw = getenv("PROSPER_PERDRAW") != nullptr;
    static const bool force_folded = getenv("PROSPER_FOLDED") != nullptr;
    return force_perdraw || (!force_folded && (st.draws.size() > 1 || !st.dispatches.empty()));
}

bool retained_draw_selected(const GpuState& st, size_t index) {
    return use_per_draw_policy(st) || index + 1 == st.draws.size();
}

bool resolve_indirect_draw_arguments(const GpuState& submit, const GpuState::Draw& source,
                                     GpuState::Draw& resolved) {
    resolved = source;
    if (!source.indirect) return true;
    constexpr uint32_t kArgumentBytes = 5u * sizeof(uint32_t);
    if (!source.indirect_args_addr || (source.indirect_args_addr & 3u) ||
        !guest_readable(source.indirect_args_addr, kArgumentBytes)) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr, "[agc] indexed indirect draw skipped: unreadable arguments at 0x%llx\n",
                         static_cast<unsigned long long>(source.indirect_args_addr));
        return false;
    }
    uint32_t args[5] = {};
    std::memcpy(args, reinterpret_cast<const void*>(source.indirect_args_addr), sizeof(args));
    const uint32_t index_count = args[0];
    const uint32_t instance_count = args[1];
    const uint32_t first_index = args[2];
    const int32_t vertex_offset = static_cast<int32_t>(args[3]);
    const uint32_t first_instance = args[4];
    constexpr uint32_t kMaxIndirectCount = 1u << 20;
    if (!index_count || !instance_count) return false;  // hardware no-op
    if (index_count > kMaxIndirectCount || instance_count > kMaxIndirectCount ||
        first_instance != 0 || !source.index_base) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr,
                         "[agc] indexed indirect draw skipped: count=%u instances=%u first=%u "
                         "vertex_offset=%d first_instance=%u index_base=0x%llx\n",
                         index_count, instance_count, first_index, vertex_offset, first_instance,
                         static_cast<unsigned long long>(source.index_base));
        return false;
    }
    const GpuState& draw_state = source.state ? *source.state : submit;
    const uint64_t element_bytes = index_elem_bytes(draw_state.index_type);
    if (!element_bytes) return false;
    if (first_index > (UINT64_MAX - source.index_base) / element_bytes) return false;
    resolved.index_count = index_count;
    resolved.instance_count = instance_count;
    resolved.indexed = true;
    resolved.index_offset = first_index;
    resolved.index_addr = source.index_base + static_cast<uint64_t>(first_index) * element_bytes;
    resolved.from_offset = true;
    resolved.indirect_vertex_offset = vertex_offset;
    resolved.has_vertex_offset_override = true;
    resolved.indirect = false;
    if (std::getenv("PROSPER_INDIRECTLOG")) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 256)
            std::fprintf(stderr,
                         "[agc-indirect] draw args=0x%llx count=%u instances=%u first=%u "
                         "vertex_offset=%d index_base=0x%llx\n",
                         static_cast<unsigned long long>(source.indirect_args_addr), index_count,
                         instance_count, first_index, vertex_offset,
                         static_cast<unsigned long long>(source.index_base));
    }
    return true;
}

bool resolve_indirect_dispatch_arguments(const GpuState::Dispatch& source,
                                         GpuState::Dispatch& resolved) {
    resolved = source;
    if (!source.indirect) return true;
    constexpr uint32_t kArgumentBytes = 3u * sizeof(uint32_t);
    if (!source.indirect_args_addr || (source.indirect_args_addr & 3u) ||
        !guest_readable(source.indirect_args_addr, kArgumentBytes)) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr, "[agc] indirect dispatch skipped: unreadable arguments at 0x%llx\n",
                         static_cast<unsigned long long>(source.indirect_args_addr));
        return false;
    }
    uint32_t args[3] = {};
    std::memcpy(args, reinterpret_cast<const void*>(source.indirect_args_addr), sizeof(args));
    if (!args[0] || !args[1] || !args[2]) return false;  // hardware no-op
    resolved.threads_x = args[0];
    resolved.threads_y = args[1];
    resolved.threads_z = args[2];
    resolved.indirect = false;
    const ComputeLaunchDimensions launch = resolve_compute_launch(resolved);
    if (!launch.groups_x || !launch.groups_y || !launch.groups_z) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr,
                         "[agc] indirect dispatch skipped: dimensions=%ux%ux%u resolve to "
                         "%ux%ux%u workgroups\n",
                         args[0], args[1], args[2], launch.groups_x, launch.groups_y,
                         launch.groups_z);
        return false;
    }
    if (std::getenv("PROSPER_INDIRECTLOG")) {
        static std::atomic<int> logged{0};
        const uint64_t code_addr = source.state
            ? compute_dispatch_code_addr(*source.state, source) : 0;
        if (logged.fetch_add(1) < 256)
            std::fprintf(stderr,
                         "[agc-indirect] dispatch args=0x%llx code=0x%llx "
                         "dims=%ux%ux%u groups=%ux%ux%u\n",
                         static_cast<unsigned long long>(source.indirect_args_addr),
                         static_cast<unsigned long long>(code_addr),
                         args[0], args[1], args[2], launch.groups_x, launch.groups_y,
                         launch.groups_z);
    }
    return true;
}

// `failure`, when supplied, is filled on every false return — mirroring realize_retained_compute.
// The two exits above realize_draw_item name themselves, because nothing was attempted there and a
// shader/pipeline diagnostic cannot exist; the third forwards whatever realize_draw_item determined
// (#1636). Callers that do not want a diagnostic keep passing nothing.
bool realize_retained_draw(const GpuState& st, size_t index, float scale_x, float scale_y,
                           DrawItem& item, OperationRealizationFailure* failure = nullptr) {
    const auto note = [&](RealizationFailureReason reason) {
        // An out-of-range index has no planned operation to attach to, and a record whose identity
        // matches nothing fails validate_failure_diagnostics for the WHOLE capture. Report nothing
        // rather than poison the capture; the caller still sees false.
        if (!failure || index >= st.draws.size()) return false;
        *failure = {};
        failure->kind = SubmitOperationKind::Draw;
        failure->index = index;
        failure->command_order = st.draws[index].command_order;
        failure->reason = reason;
        return false;
    };
    if (index >= st.draws.size() || !retained_draw_selected(st, index))
        return note(RealizationFailureReason::RetainedDrawNotSelected);
    const bool per_draw = use_per_draw_policy(st);
    const GpuState& draw_state = per_draw ? st.state_at_draw(index) : st;
    GpuState::Draw draw;
    if (!resolve_indirect_draw_arguments(st, st.draws[index], draw))
        return note(RealizationFailureReason::IndirectArguments);
    const bool log = getenv("PROSPER_GFXLOG") != nullptr || getenv("PROSPER_EXECLOG") != nullptr;
    if (!realize_draw_item(draw_state, &draw, draw.index_count, 0x10000, log, item,
                           failure, true)) {
        // realize_draw_item resets and fills the record, including pipeline/targets/extent, but has
        // no notion of which retained operation it belongs to.
        if (failure) {
            failure->kind = SubmitOperationKind::Draw;
            failure->index = index;
            failure->command_order = draw.command_order;
            if (failure->reason == RealizationFailureReason::None)
                failure->reason = RealizationFailureReason::Unknown;
        }
        return false;
    }
    item.draw_index = index;
    item.command_order = draw.command_order;
    if (scale_x != 1.0f || scale_y != 1.0f)
        scale_resolved_render_area(item.ps, scale_x, scale_y);
    return true;
}

bool realize_retained_compute(const GpuState& st, size_t index, uint64_t submit_no,
                              ComputeItem& item,
                              OperationRealizationFailure* failure = nullptr) {
    if (index >= st.dispatches.size()) return false;
    // DMA-bearing submits are uncommon. A one-dispatch state keeps the mature realization path
    // intact while ensuring it runs only after every preceding ordered producer has landed.
    GpuState one = st.dispatches[index].state ? *st.dispatches[index].state : st;
    one.dispatches.clear();
    GpuState::Dispatch dispatch;
    if (!resolve_indirect_dispatch_arguments(st.dispatches[index], dispatch)) return false;
    one.dispatches.push_back(std::move(dispatch));
    std::vector<OperationRealizationFailure> failures;
    std::vector<ComputeItem> realized = realize_compute_dispatches(
        one, submit_no, failure ? &failures : nullptr);
    if (realized.empty()) {
        if (failure && !failures.empty()) {
            *failure = std::move(failures.front());
            failure->index = index;
            failure->command_order = st.dispatches[index].command_order;
        }
        return false;
    }
    item = std::move(realized.front());
    item.dispatch_index = index;
    return true;
}
} // namespace

std::vector<ComputeAuthorityBoundary> compute_authority_draw_resource_boundaries(
        const DrawItem& item, uint64_t submit_no) {
    std::vector<ComputeAuthorityBoundary> boundaries;
    const size_t vertex_resources = item.vrt ? item.vrt->resources.size() : 0;
    const size_t fragment_resources = item.prt ? item.prt->resources.size() : 0;
    boundaries.reserve((vertex_resources + fragment_resources) * 2 + 1);
    const auto append_table = [&](const ShaderResourceTable* table) {
        if (!table) return;
        for (const ShaderResource& resource : table->resources) {
            // Replay-owned bytes do not read the live guest address. Ordinary live resources have
            // no host_data and are reported with the same conservative footprint capture uses.
            if (resource.host_data || !resource.gpu_addr) continue;
            const uint64_t bytes = gpu_capture_resource_footprint(resource);
            const bool known = bytes != 0 &&
                resource.gpu_addr <= UINT64_MAX - (bytes - 1);
            boundaries.push_back({
                ComputeAuthorityBoundaryKind::DrawResource,
                submit_no,
                item.command_order,
                resource.gpu_addr,
                bytes,
                known,
                resource.binding,
                static_cast<uint32_t>(resource.cls),
                false,
            });
            const uint64_t metadata_bytes =
                gpu_capture_dcc_metadata_footprint(resource);
            if (resource.dcc_metadata_host_data || !resource.metadata_addr ||
                !metadata_bytes)
                continue;
            const bool metadata_known =
                resource.metadata_addr <= UINT64_MAX - (metadata_bytes - 1);
            boundaries.push_back({
                ComputeAuthorityBoundaryKind::DrawResource,
                submit_no,
                item.command_order,
                resource.metadata_addr,
                metadata_bytes,
                metadata_known,
                resource.binding,
                static_cast<uint32_t>(resource.cls),
                false,
            });
        }
    };
    append_table(item.vrt.get());
    append_table(item.prt.get());
    boundaries.push_back({
        ComputeAuthorityBoundaryKind::DrawResourceEnd,
        submit_no,
        item.command_order,
        0,
        0,
        false,
        UINT32_MAX,
        UINT32_MAX,
        true,
    });
    return boundaries;
}

static OrderedSubmitResult execute_ordered_gpustate(const GpuState& st, uint32_t width,
                                                     uint32_t height, uint64_t submit_no,
                                                     const LiveRenderFn& render,
                                                     const LiveComputeFn& compute,
                                                     OrderedGpustateCaptureTrace* capture_trace,
                                                     const std::vector<DrawItem>* eager_draws) {
    GuestGpuWriteSubmitScope guest_gpu_write_scope;
    if (st.dma_execution_rejected) {
        for (const GpuState::Dispatch& dispatch : st.dispatches)
            notify_compute_authority_unknown(
                ComputeAuthorityBoundaryKind::Compute,
                submit_no, dispatch.command_order);
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr,
                         "[agc] ordered DMA submit not executed: unsupported eager/deferred "
                         "guest-memory dependency\n");
        return {};
    }
    std::vector<RetainedSubmitOperation> executable;
    executable.reserve(st.draws.size() + st.dispatches.size() + st.dma_copies.size() +
                       st.parser_stalls.size() + st.ordered_memory_effects.size());
    for (size_t i = 0; i < st.draws.size(); ++i)
        if (retained_draw_selected(st, i))
            executable.push_back({RetainedSubmitKind::Draw, i, st.draws[i].command_order});
    for (size_t i = 0; i < st.dispatches.size(); ++i)
        executable.push_back({RetainedSubmitKind::Dispatch, i, st.dispatches[i].command_order});
    for (size_t i = 0; i < st.dma_copies.size(); ++i)
        executable.push_back({RetainedSubmitKind::DmaCopy, i, st.dma_copies[i].command_order});
    for (size_t i = 0; i < st.parser_stalls.size(); ++i)
        executable.push_back({RetainedSubmitKind::ParserStall, i,
                              st.parser_stalls[i].command_order});
    for (size_t i = 0; i < st.ordered_memory_effects.size(); ++i)
        executable.push_back({RetainedSubmitKind::MemoryEffect, i,
                              st.ordered_memory_effects[i].cmd.stream_order});
    std::stable_sort(executable.begin(), executable.end(), [](const auto& a, const auto& b) {
        return a.command_order < b.command_order;
    });

    std::unordered_map<size_t, size_t> eager_draw_by_index;
    if (eager_draws)
        for (size_t i = 0; i < eager_draws->size(); ++i)
            eager_draw_by_index[static_cast<size_t>((*eager_draws)[i].draw_index)] = i;

    size_t total_spans = 0;
    bool in_draw_span = false;
    for (const auto& operation : executable) {
        if (render && operation.kind == RetainedSubmitKind::Draw) {
            if (!in_draw_span) ++total_spans;
            in_draw_span = true;
        } else {
            in_draw_span = false;
        }
    }

    uint32_t full_width = present_width(), full_height = present_height();
    const float scale_x = full_width ? static_cast<float>(width) / full_width : 1.0f;
    const float scale_y = full_height ? static_cast<float>(height) / full_height : 1.0f;
    OrderedSubmitResult result;
    bool producer_epoch_ok = true;
    bool indirect_dependencies_ok = true;
    bool final_callback_sent = false;
    std::vector<DrawItem> span;
    auto flush_span = [&](bool authoritative_readback = false) {
        if (span.empty() || !render) return;
        LiveRenderPhase saved = g_live_phase;
        const bool final_span = result.render_spans + 1 == total_spans;
        g_live_phase = {result.render_spans == 0, final_span, authoritative_readback};
        RenderedFrame rendered = render(span, width, height);
        g_live_phase = saved;
        if (!rendered.empty()) result.frame = std::move(rendered);
        span.clear();
        ++result.render_spans;
        final_callback_sent |= final_span;
    };

    for (const auto& operation : executable) {
        switch (operation.kind) {
            case RetainedSubmitKind::Draw: {
                if (!render) break;
                if (st.draws[operation.index].indirect &&
                    (!indirect_dependencies_ok || !producer_epoch_ok)) {
                    if (capture_trace) {
                        capture_trace->failures.push_back({
                            SubmitOperationKind::Draw, operation.index,
                            operation.command_order,
                            RealizationFailureReason::IndirectDependencies});
                    }
                    break;
                }
                // Resource realization can inspect guest descriptor/backing state.  No exact draw
                // read range has yet been proven at this seam, so an armed authority census must
                // fail closed before that first possible consumer rather than after rendering.
                notify_compute_authority_unknown(
                    ComputeAuthorityBoundaryKind::Draw,
                    submit_no, operation.command_order);
                DrawItem item;
                bool realized = false;
                OperationRealizationFailure failure;
                bool failure_known = false;
                if (eager_draws) {
                    const auto found = eager_draw_by_index.find(operation.index);
                    if (found != eager_draw_by_index.end()) {
                        item = (*eager_draws)[found->second];
                        realized = true;
                    }
                } else {
                    realized = realize_retained_draw(
                        st, operation.index, scale_x, scale_y, item,
                        capture_trace ? &failure : nullptr);
                    failure_known = capture_trace != nullptr;
                }
                if (realized) {
                    notify_compute_authority_draw_resources(item, submit_no);
                    if (capture_trace) {
                        snapshot_pending_gpu_capture_draw_resource(
                            capture_trace->pending_capture, item, {}, &st);
                        capture_trace->draws.push_back(item);
                    }
                    span.push_back(std::move(item));
                } else {
                    notify_compute_authority_draw_unrealized(
                        submit_no, operation.command_order);
                    if (!capture_trace) break;
                    // The reason used to die here: this path simply broke, and gpu_capture later
                    // synthesized an empty Unknown record for the unrealized operation (#1636).
                    if (failure_known) {
                        failure.command_order = operation.command_order;
                        capture_trace->failures.push_back(std::move(failure));
                    } else {
                        // Eager path: realize_gpustate_draws DOES have a failures out-parameter,
                        // but this submit's call site (:6023) passes nullptr, so the reason was
                        // dropped in that earlier pass and Unknown is the honest answer here rather
                        // than a guess. Not a simple plumb-through to fix: requesting failures also
                        // takes the serial path (gpu_execute.hpp:1650 gates parallel realization on
                        // `!failures`), so it trades draw-realization throughput for diagnostics.
                        // Tracked in #1643.
                        capture_trace->failures.push_back({
                            SubmitOperationKind::Draw, operation.index,
                            operation.command_order, RealizationFailureReason::Unknown});
                    }
                }
                break;
            }
            case RetainedSubmitKind::Dispatch: {
                flush_span();
                const bool indirect = st.dispatches[operation.index].indirect;
                if (indirect && (!indirect_dependencies_ok || !producer_epoch_ok)) {
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Compute,
                        submit_no, operation.command_order);
                    if (capture_trace) {
                        capture_trace->failures.push_back({
                            SubmitOperationKind::Dispatch, operation.index,
                            operation.command_order,
                            RealizationFailureReason::IndirectDependencies});
                    }
                    producer_epoch_ok = false;
                    break;
                }
                if (!compute) {
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Compute,
                        submit_no, operation.command_order);
                    if (capture_trace) {
                        capture_trace->failures.push_back({
                            SubmitOperationKind::Dispatch, operation.index,
                            operation.command_order, RealizationFailureReason::Unknown});
                    }
                    producer_epoch_ok = false;
                    break;
                }
                ComputeItem item;
                OperationRealizationFailure failure;
                if (realize_retained_compute(
                        st, operation.index, submit_no, item,
                        capture_trace ? &failure : nullptr)) {
                    const bool executed = capture_trace
                        ? compute({item}) : compute({std::move(item)});
                    if (capture_trace && executed)
                        capture_trace->computes.push_back(std::move(item));
                    if (capture_trace && !executed) {
                        capture_trace->failures.push_back({
                            SubmitOperationKind::Dispatch, operation.index,
                            operation.command_order, RealizationFailureReason::Unknown});
                    }
                    result.compute_executed |= executed;
                    producer_epoch_ok &= executed;
                } else if (capture_trace && failure.reason != RealizationFailureReason::None) {
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Compute,
                        submit_no, operation.command_order);
                    capture_trace->failures.push_back(std::move(failure));
                    producer_epoch_ok = false;
                } else {
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Compute,
                        submit_no, operation.command_order);
                    producer_epoch_ok = false;
                }
                break;
            }
            case RetainedSubmitKind::DmaCopy: {
                flush_span(true);
                const GpuState::DmaCopy& copy = st.dma_copies[operation.index];
                // Source and destination are distinct ordered consumers: a disjoint source must not
                // hide an overlapping destination (or vice versa).
                const bool source_gds = ((copy.sels >> 8u) & 0xffu) == 1u;
                const bool destination_gds = (copy.sels & 0xffu) == 1u;
                if (!source_gds)
                    notify_compute_authority_range(
                        ComputeAuthorityBoundaryKind::Dma, submit_no,
                        operation.command_order, copy.src, copy.bytes);
                // Selector byte 1 names a GDS offset, not guest memory. The source still consumes a
                // guest/render-target range, while the destination is ordered by this executor's
                // shared GDS backing and must not manufacture a guest range at (for example) 0x24.
                if (!destination_gds)
                    notify_compute_authority_range(
                        ComputeAuthorityBoundaryKind::Dma, submit_no,
                        operation.command_order, copy.dst, copy.bytes);
                std::vector<uint8_t> current_source;
                const LiveTargetByteReadResult source_result = read_live_render_target_bytes(
                    copy.src, copy.bytes, current_source);
                if (source_result == LiveTargetByteReadResult::InvalidRange) {
                    static std::atomic<int> warned{0};
                    if (warned.fetch_add(1) < 24)
                        std::fprintf(stderr,
                                     "[agc] DMA_DATA live-target source range invalid: src=0x%llx bytes=%u\n",
                                     static_cast<unsigned long long>(copy.src), copy.bytes);
                    producer_epoch_ok = false;
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Dma, submit_no,
                        operation.command_order);
                    break;
                }
                const bool executed = execute_ordered_dma_copy(
                    copy, source_result == LiveTargetByteReadResult::Success
                              ? current_source.data() : nullptr);
                producer_epoch_ok &= executed;
                if (!executed)
                    notify_compute_authority_unknown(
                        ComputeAuthorityBoundaryKind::Dma, submit_no,
                        operation.command_order);
                break;
            }
            case RetainedSubmitKind::ParserStall:
                flush_span();
                // Once an argument-producing epoch fails, a later empty/redundant stall cannot
                // make the stale bytes trustworthy again. Keep the submit poisoned until it ends.
                indirect_dependencies_ok &= producer_epoch_ok;
                producer_epoch_ok = true;
                break;
            case RetainedSubmitKind::MemoryEffect:
                flush_span();
                {
                    const GpuState::MemoryEffect& effect =
                        st.ordered_memory_effects[operation.index];
                    const Pm4Command& command = effect.cmd;
                    bool exact = false;
                    switch (command.kind) {
                        case Pm4Command::Kind::ReleaseMem:
                            notify_compute_authority_range(
                                ComputeAuthorityBoundaryKind::OrderedMemoryEffect,
                                submit_no, operation.command_order, command.rel_addr,
                                command.rel_data_sel == 1u ? 4u : 8u);
                            exact = command.rel_addr != 0;
                            break;
                        case Pm4Command::Kind::EventWrite:
                            notify_compute_authority_range(
                                ComputeAuthorityBoundaryKind::OrderedMemoryEffect,
                                submit_no, operation.command_order, command.event_addr, 8u);
                            exact = command.event_addr != 0;
                            break;
                        case Pm4Command::Kind::WriteData:
                            if (command.wd_valid) {
                                notify_compute_authority_range(
                                    ComputeAuthorityBoundaryKind::OrderedMemoryEffect,
                                    submit_no, operation.command_order, command.wd_addr,
                                    static_cast<uint64_t>(command.wd_num) * 4u);
                                exact = command.wd_addr != 0 && command.wd_num != 0;
                            }
                            break;
                        case Pm4Command::Kind::DmaData: {
                            // Selector byte 1 names the 64 KiB GDS offset domain, not guest VA.
                            // Every other valid destination is an exact guest write range.  A
                            // An asserted address source (or a legacy >32-bit source) is observed
                            // separately; numeric width alone is insufficient for new HLE packets.
                            const bool source_gds = ((command.dd_sels >> 8u) & 0xffu) == 1u;
                            const bool destination_gds = (command.dd_sels & 0xffu) == 1u;
                            const bool address_source =
                                (command.dd_sels & kDmaDataAddressSource) != 0 ||
                                command.dd_src > UINT32_MAX;
                            if (!source_gds && address_source) {
                                notify_compute_authority_range(
                                    ComputeAuthorityBoundaryKind::Dma, submit_no,
                                    operation.command_order, command.dd_src,
                                    command.dd_bytes);
                                exact = command.dd_bytes != 0;
                            }
                            if (!destination_gds) {
                                notify_compute_authority_range(
                                    ComputeAuthorityBoundaryKind::Dma, submit_no,
                                    operation.command_order, command.dd_dst,
                                    command.dd_bytes);
                                exact = exact ||
                                    (command.dd_dst != 0 && command.dd_bytes != 0);
                            }
                            break;
                        }
                        default:
                            break;
                    }
                    if (!exact)
                        notify_compute_authority_unknown(
                            ComputeAuthorityBoundaryKind::OrderedMemoryEffect,
                            submit_no, operation.command_order);
                    execute_ordered_memory_effect(effect);
                }
                break;
        }
    }
    flush_span();
    // A semantic draw record can fail only when lazily realized at its ordered position. If that
    // record was counted as a later span, the last successful callback was intentionally marked
    // intermediate. Send an empty terminal callback so the frontend can recover cached scanout,
    // close timing state, and publish exactly once without re-rendering any draw.
    if (render && result.render_spans && !final_callback_sent) {
        LiveRenderPhase saved = g_live_phase;
        g_live_phase = {false, true, false};
        RenderedFrame rendered = render({}, width, height);
        g_live_phase = saved;
        if (!rendered.empty()) result.frame = std::move(rendered);
    }
    return result;
}

void set_submit_renderer(LiveRenderFn fn) { g_live = std::move(fn); }
bool have_submit_renderer()               { return static_cast<bool>(g_live); }
uint8_t* compute_gds_backing()            { return g_compute_gds.data(); }
size_t   compute_gds_size()               { return g_compute_gds.size(); }
void set_submit_compute(LiveComputeFn fn) { g_compute = std::move(fn); }
bool have_submit_compute()                { return static_cast<bool>(g_compute); }
void notify_compute_authority_boundary(
        const ComputeAuthorityBoundary& boundary) {
    dispatch_compute_authority_boundary(boundary);
}
void set_compute_authority_boundary_observer(
        ComputeAuthorityBoundaryObserver observer) {
    std::lock_guard lock(g_compute_authority_boundary_mutex);
    g_compute_authority_boundary_observer = std::move(observer);
    g_compute_authority_boundary_enabled.store(
        static_cast<bool>(g_compute_authority_boundary_observer),
        std::memory_order_release);
}

static LiveTargetQueryFn g_live_target_query;   // registered by the live renderer (#590)
void set_live_target_query(LiveTargetQueryFn fn) { g_live_target_query = std::move(fn); }
bool is_live_render_target(uint64_t gpu_addr) {
    return g_live_target_query && g_live_target_query(gpu_addr);
}
static LiveTargetReaderFn g_live_target_reader;
void set_live_target_reader(LiveTargetReaderFn fn) { g_live_target_reader = std::move(fn); }

static LiveTargetImageImportFn g_live_target_image_import;
static LiveTargetImageReleaseFn g_live_target_image_release;
static LiveTargetImageWrittenFn g_live_target_image_written;
void set_live_target_image_importer(LiveTargetImageImportFn import_fn,
                                    LiveTargetImageReleaseFn release_fn) {
    g_live_target_image_import = std::move(import_fn);
    g_live_target_image_release = std::move(release_fn);
}
void set_live_target_image_written_notifier(LiveTargetImageWrittenFn written_fn) {
    g_live_target_image_written = std::move(written_fn);
}
bool import_live_render_target_image(uint64_t gpu_addr, const LiveTargetImageRequest& request,
                                     LiveTargetImageImport& import) {
    import = LiveTargetImageImport{};
    if (!g_live_target_image_import) return false;
    if (!g_live_target_image_import(gpu_addr, request, import)) {
        import = LiveTargetImageImport{};
        return false;
    }
    if (!import.valid()) {
        // The importer pinned the entry before returning true; drop that pin rather than leaking a
        // permanently un-evictable cache entry if a future importer breaks the contract.
        release_live_render_target_image(gpu_addr);
        import = LiveTargetImageImport{};
        return false;
    }
    return true;
}
void release_live_render_target_image(uint64_t gpu_addr) {
    if (g_live_target_image_release) g_live_target_image_release(gpu_addr);
}
void notify_live_render_target_image_written(const LiveTargetImageWrite& write) {
    if (g_live_target_image_written && write.valid()) g_live_target_image_written(write);
}
static SharedVulkanContext g_shared_vulkan;
void set_shared_vulkan_context(const SharedVulkanContext& context) { g_shared_vulkan = context; }
SharedVulkanContext shared_vulkan_context() { return g_shared_vulkan; }

uint32_t select_native_compute_subgroup_size(const SharedVulkanContext& context,
                                             const ComputeShaderConfig& config,
                                             bool allow_multiwave, bool disabled) {
    const bool adoptable = context.valid() && context.compute_queue_supported &&
        context.storage_image_read_without_format &&
        context.storage_image_write_without_format;
    if (disabled || !adoptable ||
        !context.compute_subgroup_size_control || !context.compute_full_subgroups ||
        !context.compute_subgroup_vote || !context.compute_subgroup_arithmetic ||
        !context.max_compute_workgroup_subgroups || !context.max_compute_workgroup_size_x ||
        !context.max_compute_workgroup_invocations || !config.local_x || !config.local_y ||
        !config.local_z || (config.wave_size != 32u && config.wave_size != 64u) ||
        config.wave_size < context.min_compute_subgroup_size ||
        config.wave_size > context.max_compute_subgroup_size)
        return 0;

    // The native shader declares a flattened LocalSize=(guest X*Y*Z,1,1), then reconstructs the
    // guest 3D local/global IDs from SubgroupId/SubgroupLocalInvocationId. Besides avoiding any
    // implementation-defined lane ordering, this makes Vulkan's REQUIRE_FULL_SUBGROUPS X-dimension
    // rule explicit. Keep the multiplication and maxComputeWorkgroupSubgroups bound overflow-safe.
    const uint64_t xy = static_cast<uint64_t>(config.local_x) * config.local_y;
    if (xy > UINT64_MAX / config.local_z) return 0;
    const uint64_t local_invocations = xy * config.local_z;
    const uint64_t subgroup_capacity = static_cast<uint64_t>(config.wave_size) *
        context.max_compute_workgroup_subgroups;
    // One guest wave removes the portable shell without adding inter-subgroup coordinate recovery.
    // Multi-wave kernels can be faster or slower depending on their LDS/barrier shape, so retain
    // the portable default until a diagnostic run opts into the exact experimental contract.
    if (local_invocations % config.wave_size != 0 ||
        (!allow_multiwave && local_invocations != config.wave_size) ||
        local_invocations > subgroup_capacity ||
        local_invocations > context.max_compute_workgroup_size_x ||
        local_invocations > context.max_compute_workgroup_invocations ||
        local_invocations > UINT32_MAX)
        return 0;
    return config.wave_size;
}

// Present unification (#1270): see gpu_execute.hpp. The atomic gates the lock so the common (headless /
// non-shared / app-not-yet-adopted) path pays only a single acquire load and takes no lock. Set true
// exactly once, by prosper-app, after it has adopted the shared queue for present and before its first
// present submit; never cleared mid-run (the app owns the shared device for the process lifetime once
// adopted). acquire/release ordering publishes the adoption's writes to the guest thread.
static std::atomic<bool> g_shared_present_active{false};
std::mutex& shared_present_submit_mutex() {
    static std::mutex m;
    return m;
}
void set_shared_present_active(bool active) {
    g_shared_present_active.store(active, std::memory_order_release);
}
bool shared_present_active() {
    return g_shared_present_active.load(std::memory_order_acquire);
}
static std::atomic<bool> g_gpu_present_active{false};
void set_gpu_present_active(bool active) {
    g_gpu_present_active.store(active, std::memory_order_release);
}
bool gpu_present_active() {
    return g_gpu_present_active.load(std::memory_order_acquire);
}
bool read_live_render_target(uint64_t gpu_addr, LiveTargetSnapshot& snapshot) {
    snapshot = {};
    return g_live_target_reader && g_live_target_reader(gpu_addr, snapshot);
}
static LiveTargetByteRangeReaderFn g_live_target_byte_range_reader;
void set_live_target_byte_range_reader(LiveTargetByteRangeReaderFn fn) {
    g_live_target_byte_range_reader = std::move(fn);
}
LiveTargetByteReadResult read_live_render_target_bytes(uint64_t gpu_addr, uint32_t bytes,
                                                       std::vector<uint8_t>& output) {
    output.clear();
    if (!g_live_target_byte_range_reader)
        return LiveTargetByteReadResult::NotFound;
    return g_live_target_byte_range_reader(gpu_addr, bytes, output);
}

void set_guest_gpu_write_observer(GuestGpuWriteObserver observer) {
    g_guest_gpu_write_observer = std::move(observer);
}
void notify_guest_gpu_write(uint64_t addr, uint64_t size) {
    if (!addr || !size) return;
    // Page-protection watches observe guest CPU stores, but device/DMA writes can mutate the same
    // direct-memory pages without a CPU protection fault. Mark the virtual range dirty as part of the
    // existing authoritative GPU-write notification so cross-submit texture/compute caches never trust
    // an Unchanged watch over bytes written by the GPU. This may run after a host-mirrored write and
    // is deliberately idempotent.
    prosper::host::guest_write_watch_notify_gpu_write(addr, size);
    if (g_guest_gpu_writes.active) {
        if (g_guest_gpu_writes.writes.size() < kGuestGpuWriteJournalCapacity)
            g_guest_gpu_writes.writes.push_back({addr, size});
        else
            g_guest_gpu_writes.overflowed = true;
    }
    if (g_guest_gpu_write_observer) g_guest_gpu_write_observer(addr, size);
}
void notify_guest_gpu_write_preserving_bytes(uint64_t addr, uint64_t size) {
    if (!addr || !size) return;
    // The observer owns renderer-resident aliases (color/depth targets and their CPU snapshots),
    // which may differ from the exact guest bytes even when a compute result does not. Guest-memory
    // caches, page watches, and the submit journal remain valid because the caller proved that those
    // bytes were not modified.
    if (g_guest_gpu_write_observer) g_guest_gpu_write_observer(addr, size);
}
GuestGpuWriteSnapshot guest_gpu_write_snapshot() {
    if (!g_guest_gpu_writes.active || g_guest_gpu_writes.overflowed) return {};
    return {g_guest_gpu_writes.submit_serial, g_guest_gpu_writes.writes.size()};
}
GuestGpuWriteQuery guest_gpu_writes_since(const GuestGpuWriteSnapshot& snapshot,
                                           uint64_t addr, uint64_t size) {
    if (!snapshot.submit_serial || !g_guest_gpu_writes.active ||
        snapshot.submit_serial != g_guest_gpu_writes.submit_serial ||
        g_guest_gpu_writes.overflowed ||
        snapshot.write_count > g_guest_gpu_writes.writes.size())
        return GuestGpuWriteQuery::Unknown;
    for (size_t i = snapshot.write_count; i < g_guest_gpu_writes.writes.size(); ++i) {
        const auto& write = g_guest_gpu_writes.writes[i];
        if (ranges_overlap(addr, size, write.addr, write.size))
            return GuestGpuWriteQuery::Overlap;
    }
    return GuestGpuWriteQuery::Unchanged;
}
LiveRenderPhase live_render_phase()       { return g_live_phase; }
std::vector<uint8_t> render_submit_items(const std::vector<DrawItem>& items,
                                         uint32_t width, uint32_t height) {
    if (!g_live) return {};
    RenderedFrame frame = g_live(items, width, height);
    return frame.storage ? *frame.storage : std::vector<uint8_t>{};
}
bool execute_compute_items(const std::vector<ComputeItem>& items) {
    return g_compute && !items.empty() && g_compute(items);
}

bool execute_ordered_and_present(const GpuState& st, uint32_t width, uint32_t height,
                                 uint64_t submit_no, bool publish) {
    if ((!g_live && !g_compute && st.dma_copies.empty()) ||
        (st.draws.empty() && st.dispatches.empty() && st.dma_copies.empty())) return false;
    // Guest allocations referenced by a GPU submit must remain mapped until that submit completes.
    // Reuse positive page/VirtualQuery results only inside this synchronous execution window; the
    // scope is discarded before guest code can submit a later mapping generation.
    GuestReadableSubmitScope guest_readable_scope;
    notify_compute_authority_unknown(
        ComputeAuthorityBoundaryKind::SubmitBegin, submit_no);
    using TimingClock = std::chrono::steady_clock;
    const bool timing_enabled = std::getenv("PROSPER_RENDER_TIMING") != nullptr;
    const auto timing_start = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    const ShaderRecompileCacheStats shader_before = timing_enabled
        ? shader_recompile_cache_stats() : ShaderRecompileCacheStats{};
    const ShaderDecodeCacheStats decode_before = timing_enabled
        ? shader_decode_cache_stats() : ShaderDecodeCacheStats{};
    const DrawRealizationPhaseStats phases_before = timing_enabled
        ? draw_realization_phase_stats() : DrawRealizationPhaseStats{};
    const StageTablePhaseStats table_phases_before = timing_enabled
        ? stage_table_phase_stats() : StageTablePhaseStats{};
    const ParallelDrawRealizationStats parallel_before = timing_enabled
        ? parallel_draw_realization_stats() : ParallelDrawRealizationStats{};
    uint32_t fw = present_width(), fh = present_height();
    float sx = fw ? (float)width / (float)fw : 1.0f;
    float sy = fh ? (float)height / (float)fh : 1.0f;
    const bool has_ordered_dma = !st.dma_copies.empty();
    const bool has_indirect = std::any_of(st.draws.begin(), st.draws.end(),
                                         [](const auto& draw) { return draw.indirect; }) ||
                              std::any_of(st.dispatches.begin(), st.dispatches.end(),
                                         [](const auto& dispatch) { return dispatch.indirect; });
    const bool needs_ordered_realization = has_ordered_dma || has_indirect || !st.dispatches.empty();
    // Draws without DMA/indirect arguments remain safe to prepare in parallel. Compute resources,
    // however, are always realized at their ordered position: a preceding dispatch in the same
    // submit can write a pointer or descriptor consumed by the next dispatch (Astro Bot's BVH root
    // is one such dependency). Pre-realizing every compute snapshots stale guest bytes.
    const bool can_eagerly_realize_draws = !has_ordered_dma && !has_indirect;
    std::vector<DrawItem> draws = can_eagerly_realize_draws && g_live && width && height
        ? realize_gpustate_draws(st, 0x10000, sx, sy, nullptr, true)
        : std::vector<DrawItem>{};
    const ShaderRecompileCacheStats shader_after = timing_enabled
        ? shader_recompile_cache_stats() : ShaderRecompileCacheStats{};
    const ShaderDecodeCacheStats decode_after = timing_enabled
        ? shader_decode_cache_stats() : ShaderDecodeCacheStats{};
    const DrawRealizationPhaseStats phases_after = timing_enabled
        ? draw_realization_phase_stats() : DrawRealizationPhaseStats{};
    const StageTablePhaseStats table_phases_after = timing_enabled
        ? stage_table_phase_stats() : StageTablePhaseStats{};
    const ParallelDrawRealizationStats parallel_after = timing_enabled
        ? parallel_draw_realization_stats() : ParallelDrawRealizationStats{};
    const auto timing_draws_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    std::vector<ComputeItem> computes = !needs_ordered_realization && g_compute
        ? realize_compute_dispatches(st, submit_no) : std::vector<ComputeItem>{};
    const auto timing_compute_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};

    auto operations = plan_submit_operations(st);
    const auto timing_plan_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    // DMA-bearing submits are captured from semantic state so v14 can retain both endpoint backings
    // even though live execution deliberately realizes their consumers only at ordered positions.
    auto pending_capture = begin_requested_gpu_capture(
        draws, computes, operations, width, height, &st, submit_no,
        static_cast<uint64_t>(st.draws.size()), nullptr,
        /*defer_materialization=*/needs_ordered_realization);
    snapshot_pending_gpu_capture_compute_gds(
        pending_capture.get(), g_compute_gds.data(), g_compute_gds.size());
    OrderedGpustateCaptureTrace capture_trace;
    capture_trace.pending_capture = pending_capture.get();
    OrderedSubmitResult result = needs_ordered_realization
        ? execute_ordered_gpustate(st, width, height, submit_no, g_live, g_compute,
                                   pending_capture ? &capture_trace : nullptr,
                                   can_eagerly_realize_draws ? &draws : nullptr)
        : execute_ordered_items(operations, draws, computes, g_live, g_compute, width, height);
    const auto timing_backend_done = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    const std::vector<uint8_t>& px = result.frame.bytes();

    if (pending_capture) {
        std::string error;
        notify_compute_authority_unknown(
            ComputeAuthorityBoundaryKind::Capture, submit_no);
        const std::vector<DrawItem>* capture_draws = needs_ordered_realization
            ? &capture_trace.draws : &draws;
        const std::vector<ComputeItem>* capture_computes = needs_ordered_realization
            ? &capture_trace.computes : &computes;
        if (!finish_requested_gpu_capture(std::move(pending_capture), px, error,
                                          capture_draws, capture_computes, &operations, &st,
                                          needs_ordered_realization ? &capture_trace.failures : nullptr))
            std::fprintf(stderr, "[gpucap] write failed: %s\n", error.c_str());
    }
    notify_compute_authority_unknown(
        ComputeAuthorityBoundaryKind::SubmitEnd, submit_no);
    const bool frame_ready = px.size() == static_cast<size_t>(width) * height * 4;
    const bool presented = frame_ready && publish;
    if (presented) present_write_frame(result.frame.storage, width, height);
    if (timing_enabled) {
        const auto timing_done = TimingClock::now();
        auto ms = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        struct TimingTotals {
            uint64_t submits = 0, draws = 0, dispatches = 0, render_spans = 0;
            uint64_t shader_hits = 0, shader_misses = 0, shader_bypasses = 0;
            uint64_t decode_hits = 0, decode_misses = 0, decode_invalidations = 0;
            uint64_t readable_calls = 0, readable_hits = 0, readable_os_probes = 0;
            uint64_t parallel_batches = 0, parallel_draws = 0, parallel_threads = 0;
            double realize_draws = 0, realize_compute = 0, plan = 0, backend = 0, publish = 0;
            double table_build = 0, shader_lookup = 0, shader_compile = 0;
            double table_metadata = 0, table_dynamic_fold = 0, table_resources = 0;
            double parallel_wall = 0;
        };
        static TimingTotals totals;
        static TimingTotals window;
        auto accumulate = [&](TimingTotals& timing) {
            timing.submits++;
            timing.draws += draws.size();
            timing.dispatches += computes.size();
            timing.render_spans += result.render_spans;
            timing.shader_hits += shader_after.hits - shader_before.hits;
            timing.shader_misses += shader_after.misses - shader_before.misses;
            timing.shader_bypasses += shader_after.bypasses - shader_before.bypasses;
            timing.decode_hits += decode_after.hits - decode_before.hits;
            timing.decode_misses += decode_after.misses - decode_before.misses;
            timing.decode_invalidations += decode_after.invalidations - decode_before.invalidations;
            timing.readable_calls += g_guest_readable_cache.calls;
            timing.readable_hits += g_guest_readable_cache.hits;
            timing.readable_os_probes += g_guest_readable_cache.os_probes;
            timing.parallel_batches += parallel_after.batches - parallel_before.batches;
            timing.parallel_draws += parallel_after.semantic_draws - parallel_before.semantic_draws;
            timing.parallel_threads += parallel_after.worker_threads - parallel_before.worker_threads;
            timing.parallel_wall += parallel_after.wall_ms - parallel_before.wall_ms;
            timing.realize_draws += ms(timing_start, timing_draws_ready);
            timing.realize_compute += ms(timing_draws_ready, timing_compute_ready);
            timing.plan += ms(timing_compute_ready, timing_plan_ready);
            timing.backend += ms(timing_plan_ready, timing_backend_done);
            timing.publish += ms(timing_backend_done, timing_done);
            timing.table_build += phases_after.table_ms - phases_before.table_ms;
            timing.table_metadata += table_phases_after.metadata_ms - table_phases_before.metadata_ms;
            timing.table_dynamic_fold += table_phases_after.dynamic_fold_ms -
                                         table_phases_before.dynamic_fold_ms;
            timing.table_resources += table_phases_after.resources_ms - table_phases_before.resources_ms;
            timing.shader_lookup += phases_after.shader_ms - phases_before.shader_ms;
            timing.shader_compile += shader_after.compile_ms - shader_before.compile_ms;
        };
        accumulate(totals);
        accumulate(window);
        if (totals.submits % 25 == 0) {
            const double n = static_cast<double>(totals.submits);
            const double total = totals.realize_draws + totals.realize_compute + totals.plan +
                                 totals.backend + totals.publish;
            std::fprintf(stderr,
                         "[render-timing] submits=%llu draws=%llu dispatches=%llu spans=%llu "
                         "avg_ms: total=%.2f "
                         "realize_draws=%.2f realize_compute=%.2f plan=%.2f backend=%.2f publish=%.2f\n",
                         (unsigned long long)totals.submits, (unsigned long long)totals.draws,
                         (unsigned long long)totals.dispatches,
                         (unsigned long long)totals.render_spans, total / n,
                         totals.realize_draws / n, totals.realize_compute / n, totals.plan / n,
                         totals.backend / n, totals.publish / n);
            const double wn = static_cast<double>(window.submits);
            const double window_total = window.realize_draws + window.realize_compute + window.plan +
                                        window.backend + window.publish;
            std::fprintf(stderr,
                         "[render-window] submits=%llu avg_items: draws=%.1f dispatches=%.1f "
                         "spans=%.1f shaders: hit=%.1f miss=%.1f bypass=%.1f "
                         "decode: hit=%.1f miss=%.1f invalid=%.1f "
                         "readable: hit=%.1f/%.1f os=%.1f "
                         "avg_ms: total=%.2f realize_draws=%.2f tables=%.2f shader_lookup=%.2f "
                         "shader_compile=%.2f table_parts: metadata=%.2f fold=%.2f resources=%.2f "
                         "parallel: batches=%.2f draws=%.1f threads=%.1f wall=%.2f "
                         "realize_compute=%.2f plan=%.2f backend=%.2f publish=%.2f\n",
                         (unsigned long long)window.submits, window.draws / wn,
                         window.dispatches / wn, window.render_spans / wn,
                         window.shader_hits / wn, window.shader_misses / wn,
                         window.shader_bypasses / wn, window.decode_hits / wn,
                         window.decode_misses / wn, window.decode_invalidations / wn,
                         window.readable_hits / wn, window.readable_calls / wn,
                         window.readable_os_probes / wn, window_total / wn,
                         window.realize_draws / wn, window.table_build / wn,
                         window.shader_lookup / wn, window.shader_compile / wn,
                         window.table_metadata / wn, window.table_dynamic_fold / wn,
                         window.table_resources / wn,
                         window.parallel_batches / wn, window.parallel_draws / wn,
                         window.parallel_batches
                             ? static_cast<double>(window.parallel_threads) /
                                   static_cast<double>(window.parallel_batches)
                             : 0.0,
                         window.parallel_wall / wn,
                         window.realize_compute / wn, window.plan / wn, window.backend / wn,
                         window.publish / wn);
            window = {};
        }
    }
    return presented;
}

bool execute_and_present(const GpuState& st, uint32_t width, uint32_t height, bool publish) {
    if (!g_live || st.draws.empty() || !width || !height) return false;
    // Bind the target dimensions and defer to the pure core, which recompiles the shaders from their
    // SHADER_PGM addresses and resolves fixed-function state before calling back into the live renderer.
    // Scale the guest viewport to our framebuffer: `width`/`height` are the render target (reduced by
    // PROSPER_RENDER_SCALE), while the guest programs its viewport in full present-resolution pixels — so
    // without this a 1/N render shows only the bottom-left 1/N of the frame.
    uint32_t fw = present_width(), fh = present_height();
    float sx = fw ? (float)width  / (float)fw : 1.0f;
    float sy = fh ? (float)height / (float)fh : 1.0f;
    std::vector<DrawItem> items = realize_gpustate_draws(
        st, 0x10000, sx, sy, nullptr, true);
    if (items.empty()) return false;
    std::vector<SubmitOperation> operations;
    operations.reserve(items.size());
    for (const auto& item : items)
        operations.push_back({SubmitOperationKind::Draw,
                              static_cast<size_t>(item.draw_index), item.command_order});
    auto pending = begin_requested_gpu_capture(items, {}, operations, width, height);
    RenderedFrame rendered = g_live(items, width, height);
    if (pending) {
        std::string error;
        if (!finish_requested_gpu_capture(std::move(pending), rendered.bytes(), error))
            std::fprintf(stderr, "[gpucap] write failed: %s\n", error.c_str());
    }
    if (rendered.size() != static_cast<size_t>(width) * height * 4) return false;
    if (!publish) return false;
    present_write_frame(rendered.storage, width, height);
    return true;
}

} // namespace prosper::gpu
