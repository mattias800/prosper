// gpu_executor.cpp — the live-submit half of the GPU executor (Stage A of docs/GPU_EXECUTOR_DESIGN.md).
//
// Holds the process-wide live render backend and drives it on each AGC submit. This is deliberately the
// ONLY place the executor touches process-global state; execute_gpustate() itself (gpu_execute.hpp) stays
// pure. No Vulkan here — the backend is a std::function injected by whoever owns a device (the runtime
// binary at startup, or a test via render_runner.h), so prosper_core links this without Vulkan.
#include "gpu_execute.hpp"
#include "gpu_capture.hpp"
#include "videoout_present.hpp"   // present_write_frame
#include "agc_shader_layout.hpp"  // AgcShaderHeader + build_shader_resources
#include "pm4_registers.hpp"      // SPI_SHADER_USER_DATA_* offsets
#include "rdna2_decode.hpp"       // rdna2_walk (for the vertex-fetch const-eval)
#include "rdna2_to_spirv.hpp"     // recompile_compute
#include "writer_provenance.hpp"
#include "../host/guest_memory_map.hpp"
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <bitset>
#include <filesystem>
#include <iterator>
#include <mutex>
#include <set>
#include <tuple>
#include <vector>
#include <unordered_map>
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

namespace prosper::gpu {
namespace {
LiveRenderFn g_live;   // empty until the runtime/test registers a device-backed renderer
LiveComputeFn g_compute;   // synchronous compute backend, registered with the live Vulkan frontend
GuestGpuWriteObserver g_guest_gpu_write_observer;
thread_local LiveRenderPhase g_live_phase;

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

namespace {

struct ShaderResourceCompileKey {
    uint32_t cls = 0;
    uint32_t format = 0;
    uint32_t num_components = 0;
    uint32_t binding = 0;
    uint32_t stride = 0;
    uint32_t srt_offset = 0;
    uint32_t sgpr_base = 0;
    uint32_t fetch_pc = 0;

    bool operator==(const ShaderResourceCompileKey&) const = default;
};

struct ShaderCompileKey {
    ShaderProgramStage stage = ShaderProgramStage::Vertex;
    bool has_resource_table = false;
    bool force_position_w = false;
    bool has_pixel_inputs = false;
    PixelInputMapping pixel_inputs{};
    bool has_system_inputs = false;
    PixelSystemInputMapping system_inputs{};
    bool has_pcrel_dispatch = false;
    uint32_t pcrel_dispatch_target = UINT32_MAX;
    std::vector<uint32_t> code;
    std::vector<ShaderResourceCompileKey> resources;

    bool operator==(const ShaderCompileKey&) const = default;
};

static uint64_t hash_mix(uint64_t hash, uint64_t value) {
    // FNV-1a over fixed-width values. Equality still compares the full key, so collisions are benign.
    for (unsigned i = 0; i < 8; ++i) {
        hash ^= static_cast<uint8_t>(value >> (i * 8));
        hash *= 1099511628211ull;
    }
    return hash;
}

struct ShaderCompileKeyHash {
    size_t operator()(const ShaderCompileKey& key) const {
        uint64_t hash = 1469598103934665603ull;
        hash = hash_mix(hash, static_cast<uint32_t>(key.stage));
        hash = hash_mix(hash, key.has_resource_table);
        hash = hash_mix(hash, key.force_position_w);
        hash = hash_mix(hash, key.has_pixel_inputs);
        if (key.has_pixel_inputs) {
            hash = hash_mix(hash, key.pixel_inputs.valid_mask);
            for (uint32_t control : key.pixel_inputs.controls)
                hash = hash_mix(hash, control);
        }
        hash = hash_mix(hash, key.has_system_inputs);
        if (key.has_system_inputs) {
            hash = hash_mix(hash, key.system_inputs.ena);
            hash = hash_mix(hash, key.system_inputs.addr);
        }
        hash = hash_mix(hash, key.has_pcrel_dispatch);
        if (key.has_pcrel_dispatch) hash = hash_mix(hash, key.pcrel_dispatch_target);
        hash = hash_mix(hash, key.code.size());
        for (uint32_t word : key.code) hash = hash_mix(hash, word);
        hash = hash_mix(hash, key.resources.size());
        for (const auto& resource : key.resources) {
            hash = hash_mix(hash, resource.cls);
            hash = hash_mix(hash, resource.format);
            hash = hash_mix(hash, resource.num_components);
            hash = hash_mix(hash, resource.binding);
            hash = hash_mix(hash, resource.stride);
            hash = hash_mix(hash, resource.srt_offset);
            hash = hash_mix(hash, resource.sgpr_base);
            hash = hash_mix(hash, resource.fetch_pc);
        }
        return static_cast<size_t>(hash);
    }
};

struct CachedShader {
    std::vector<uint32_t> spirv;
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
    size_t source_dwords = 0;
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
        for (const Rdna2Inst& instruction : decoded) {
            if (instruction.fmt != Rdna2Format::VOP3) continue;
            if (instruction.opcode == 0x361 && instruction.dst.kind == OperandKind::VGPR)
                scalar_spill_vgprs.insert(instruction.dst.value);       // v_writelane_b32
            else if (instruction.opcode == 0x360 && instruction.src[0].kind == OperandKind::VGPR)
                scalar_spill_vgprs.insert(instruction.src[0].value);    // v_readlane_b32
        }
        // Retain only instructions that can affect fold state or emit a descriptor use, preserving
        // their original order and PCs.
        result->instructions.reserve(decoded.size());
        for (const Rdna2Inst& instruction : decoded) {
            if (instruction.is_end) break;
            const bool scalar_spill = instruction.fmt == Rdna2Format::VOP3 &&
                                      (instruction.opcode == 0x360 || instruction.opcode == 0x361);
            const bool scalar_spill_invalidation = instruction.dst.kind == OperandKind::VGPR &&
                                                   scalar_spill_vgprs.contains(instruction.dst.value);
            const bool fold_format = instruction.fmt == Rdna2Format::SOP1 ||
                                     instruction.fmt == Rdna2Format::SOP2 ||
                                     instruction.fmt == Rdna2Format::SOPC ||
                                     instruction.fmt == Rdna2Format::SOPK ||
                                     instruction.fmt == Rdna2Format::SMEM ||
                                     instruction.fmt == Rdna2Format::MIMG ||
                                     instruction.fmt == Rdna2Format::MUBUF;
            if (fold_format || scalar_spill || scalar_spill_invalidation ||
                instruction.dst.kind == OperandKind::SGPR)
                result->instructions.push_back(instruction);
        }
        result->bytes = static_cast<uint64_t>(result->code.size()) * sizeof(uint32_t) +
                        static_cast<uint64_t>(result->instructions.size()) * sizeof(Rdna2Inst);
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
    return static_cast<uint64_t>(key.code.size()) * sizeof(uint32_t) +
           static_cast<uint64_t>(key.resources.size()) * sizeof(ShaderResourceCompileKey) +
           (key.has_pixel_inputs ? sizeof(PixelInputMapping) : 0u) +
           (key.has_system_inputs ? sizeof(PixelSystemInputMapping) : 0u) +
           static_cast<uint64_t>(spirv.size()) * sizeof(uint32_t);
}

ShaderCompileKey make_shader_compile_key(ShaderProgramStage stage, const uint32_t* code, size_t dwords,
                                         const ShaderResourceTable* resources,
                                         const PixelInputMapping* pixel_inputs,
                                         const PixelSystemInputMapping* system_inputs) {
    ShaderCompileKey key;
    key.stage = stage;
    key.has_resource_table = resources != nullptr;
    key.force_position_w = getenv("PROSPER_FORCE_W") != nullptr;
    key.has_pixel_inputs = stage == ShaderProgramStage::Vertex && pixel_inputs != nullptr;
    if (key.has_pixel_inputs) key.pixel_inputs = *pixel_inputs;
    key.has_system_inputs = stage == ShaderProgramStage::Fragment && system_inputs != nullptr;
    if (key.has_system_inputs) key.system_inputs = *system_inputs;
    if (stage == ShaderProgramStage::Fragment && code && dwords && resources) {
        const PcrelDispatchInfo dispatch = rdna2_pcrel_dispatch_info(code, dwords);
        if (dispatch.valid) {
            const ShaderResource* resource = resources->by_sgpr_base_cls(
                dispatch.selector_sgpr_base, ResourceClass::ConstantBuffer);
            uint32_t raw_selector = 0;
            bool readable = false;
            if (resource && dispatch.selector_byte_offset <= resource->size &&
                resource->size - dispatch.selector_byte_offset >= sizeof(raw_selector)) {
                if (resource->host_data && dispatch.selector_byte_offset <= resource->host_data_size &&
                    resource->host_data_size - dispatch.selector_byte_offset >= sizeof(raw_selector)) {
                    memcpy(&raw_selector, resource->host_data + dispatch.selector_byte_offset,
                           sizeof(raw_selector));
                    readable = true;
                } else if (resource->gpu_addr <= UINT64_MAX - dispatch.selector_byte_offset) {
                    const uint64_t address = resource->gpu_addr + dispatch.selector_byte_offset;
                    if (guest_readable(address, sizeof(raw_selector))) {
                        memcpy(&raw_selector, reinterpret_cast<const void*>(static_cast<uintptr_t>(address)),
                               sizeof(raw_selector));
                        readable = true;
                    }
                }
            }
            if (readable) {
                const uint32_t adjusted = raw_selector + static_cast<uint32_t>(dispatch.selector_addend);
                const uint32_t index = std::min(adjusted, dispatch.selector_max);
                if (index < dispatch.target_pcs.size()) {
                    key.has_pcrel_dispatch = true;
                    key.pcrel_dispatch_target = dispatch.target_pcs[index];
                }
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
                            dispatch.selector_byte_offset, resource ? "found" : "missing", readable,
                            raw_selector, key.pcrel_dispatch_target, resources->resources.size());
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
    if (code && dwords) {
        // Most shaders end at S_ENDPGM. A compiler-generated s_getpc_b64 V# may instead address an
        // embedded lookup table after ENDPGM; retain that proven tail so cached recompilation sees the
        // same blob as the direct path and table contents participate in the cache identity.
        const size_t span = rdna2_recompile_code_span(code, dwords);
        key.code.assign(code, code + span);
    }
    if (resources) {
        key.resources.reserve(resources->resources.size());
        for (const auto& resource : resources->resources) {
            key.resources.push_back({
                static_cast<uint32_t>(resource.cls), static_cast<uint32_t>(resource.format),
                resource.num_components, resource.binding, resource.stride,
                resource.srt_offset, resource.sgpr_base, resource.fetch_pc,
            });
        }
    }
    return key;
}

std::vector<uint32_t> compile_graphics_shader(ShaderProgramStage stage, const ShaderCompileKey& key,
                                              const ShaderResourceTable* resources) {
    const uint32_t* code = key.code.empty() ? nullptr : key.code.data();
    if (stage == ShaderProgramStage::Vertex)
        return recompile_vertex(code, key.code.size(), resources,
                                key.has_pixel_inputs ? &key.pixel_inputs : nullptr);
    if (stage == ShaderProgramStage::Fragment)
        return recompile_fragment(code, key.code.size(), resources,
                                  key.has_system_inputs ? &key.system_inputs : nullptr,
                                  key.has_pcrel_dispatch ? key.pcrel_dispatch_target : UINT32_MAX);
    return {};
}

void maybe_dump_successful_shader(ShaderProgramStage stage, const ShaderCompileKey& key,
                                  const std::vector<uint32_t>& spirv) {
    const char* directory = getenv("PROSPER_SHADER_DUMP_SUCCESS");
    if (!directory || !*directory || key.code.empty() || spirv.empty()) return;

    const uint64_t spirv_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(spirv.data()), spirv.size() * sizeof(uint32_t));
    const uint64_t raw_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(key.code.data()), key.code.size() * sizeof(uint32_t));
    static std::mutex dump_mutex;
    static std::set<std::tuple<uint32_t, uint64_t, uint64_t>> dumped;
    std::lock_guard lock(dump_mutex);
    if (!dumped.emplace(static_cast<uint32_t>(stage), spirv_hash, raw_hash).second) return;

    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        fprintf(stderr, "[shader-dump] cannot create %s: %s\n", directory, ec.message().c_str());
        return;
    }
    const char* tag = stage == ShaderProgramStage::Vertex ? "vs" :
                      stage == ShaderProgramStage::Fragment ? "ps" : "cs";
    char raw_path[1024], spirv_path[1024];
    snprintf(raw_path, sizeof(raw_path), "%s/success_%s_%016llx_%016llx.bin", directory, tag,
             static_cast<unsigned long long>(spirv_hash),
             static_cast<unsigned long long>(raw_hash));
    snprintf(spirv_path, sizeof(spirv_path), "%s/success_%s_%016llx_%016llx.spv", directory, tag,
             static_cast<unsigned long long>(spirv_hash),
             static_cast<unsigned long long>(raw_hash));
    FILE* raw = fopen(raw_path, "wb");
    FILE* translated = fopen(spirv_path, "wb");
    const size_t raw_bytes = key.code.size() * sizeof(uint32_t);
    const size_t spirv_bytes = spirv.size() * sizeof(uint32_t);
    const bool ok = raw && translated &&
        fwrite(key.code.data(), 1, raw_bytes, raw) == raw_bytes &&
        fwrite(spirv.data(), 1, spirv_bytes, translated) == spirv_bytes;
    if (raw) fclose(raw);
    if (translated) fclose(translated);
    fprintf(stderr, "[shader-dump] %s spv=%016llx raw=%016llx words=%zu/%zu result=%s\n", tag,
            static_cast<unsigned long long>(spirv_hash),
            static_cast<unsigned long long>(raw_hash), key.code.size(), spirv.size(),
            ok ? "written" : "failed");
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

std::vector<uint32_t> recompile_graphics_shader_cached(ShaderProgramStage stage,
                                                       const uint32_t* code, size_t dwords,
                                                       const ShaderResourceTable* resources,
                                                       const PixelInputMapping* pixel_inputs,
                                                       const PixelSystemInputMapping* system_inputs,
                                                       uint64_t* cache_identity) {
    if (cache_identity) *cache_identity = 0;
    ShaderCompileKey key = make_shader_compile_key(stage, code, dwords, resources, pixel_inputs,
                                                   system_inputs);
    if (getenv("PROSPER_NO_SHADER_CACHE")) {
        auto& cache = shader_cache();
        {
            std::lock_guard lock(cache.mutex);
            ++cache.stats.bypasses;
        }
        std::vector<uint32_t> spirv = compile_graphics_shader(stage, key, resources);
        maybe_dump_successful_shader(stage, key, spirv);
        return spirv;
    }

    auto& cache = shader_cache();
    std::lock_guard lock(cache.mutex);
    auto found = cache.entries.find(key);
    if (found != cache.entries.end()) {
        ++cache.stats.hits;
        found->second.last_use = ++cache.use_counter;
        if (cache_identity) *cache_identity = found->second.identity;
        maybe_dump_successful_shader(stage, key, found->second.spirv);
        return found->second.spirv;
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<uint32_t> spirv = compile_graphics_shader(stage, key, resources);
    maybe_dump_successful_shader(stage, key, spirv);
    const auto end = std::chrono::steady_clock::now();
    ++cache.stats.misses;
    cache.stats.compile_ms += std::chrono::duration<double, std::milli>(end - start).count();

    constexpr size_t max_entries = 4096;
    const uint64_t bytes = shader_cache_entry_bytes(key, spirv);
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
    bool active = false;
    host::GuestReadableRangeCache persistent_ranges;
    host::GuestReadableRangeCache submit_ranges;
    uint64_t calls = 0, hits = 0, os_probes = 0;
};
thread_local GuestReadableCacheState g_guest_readable_cache;

bool guest_range_cache_hit(uint64_t begin, uint64_t end) {
    if (!g_guest_readable_cache.active) return false;
    ++g_guest_readable_cache.calls;
    const bool hit = g_guest_readable_cache.persistent_ranges.contains(begin, end) ||
                     g_guest_readable_cache.submit_ranges.contains(begin, end);
    if (hit) ++g_guest_readable_cache.hits;
    return hit;
}

void cache_guest_readable_range(uint64_t begin, uint64_t end,
                                uint64_t query_begin, uint64_t query_end) {
    if (!g_guest_readable_cache.active || begin >= end) return;
    g_guest_readable_cache.submit_ranges.insert(begin, end);
    host::GuestReadableRange mapping{};
    if (host::guest_readable_mapping_containing(query_begin, query_end, mapping))
        g_guest_readable_cache.persistent_ranges.insert(mapping.begin, mapping.end);
}

struct GuestReadableSubmitScope {
    GuestReadableSubmitScope() {
        g_guest_readable_cache.active = getenv("PROSPER_NO_GUEST_READ_CACHE") == nullptr;
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
                      uint32_t user_sgpr_base, std::vector<SrtUse>* srt_uses) {
    using FoldClock = std::chrono::steady_clock;
    static const bool profile_fold = std::getenv("PROSPER_STAGE_FOLD_PROFILE") != nullptr;
    const auto fold_start = profile_fold ? FoldClock::now() : FoldClock::time_point{};
    const size_t srt_before = srt_uses ? srt_uses->size() : 0;
    uint64_t guest_probe_calls = 0;
    double guest_probe_ms = 0.0;
    std::vector<DynFetch> out;
    const auto decoded = decode_shader_cached(code, dwords);
    const auto decode_done = profile_fold ? FoldClock::now() : FoldClock::time_point{};
    const auto& ins = decoded->instructions;

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
    // Scalar operands encode at most 128 SGPRs. Fixed register files avoid hundreds of tiny hash/tree
    // allocations per submit while retaining exactly the same known/unknown state model.
    constexpr size_t kFoldSgprs = 128;
    std::array<uint32_t, kFoldSgprs> val{};                 // concrete SGPR values
    std::bitset<kFoldSgprs> val_known;
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
    // SGPRs overwritten by an s_load since seeding — the seed-V# MUBUF fallback below must not use a
    // stale user-data snapshot once the register was RELOADED from memory (ALU patches deliberately
    // don't count: descriptor snapshots are load-time semantics, pre-patch, like `descr`).
    std::bitset<kFoldSgprs> reloaded;
    int scc = -1;   // tracked SCC (-1 unknown): set by s_cmp_*, consumed by s_cselect (the format patch's tail)
    // The SPI loads the user-data block starting at shader SGPR `user_sgpr_base` (s0..s7 are NGG system
    // SGPRs). So user-data block index k lands in shader SGPR (user_sgpr_base + k).
    auto valid_reg = [](int r) { return r >= 0 && r < (int)kFoldSgprs; };
    auto set_value = [&](int r, uint32_t v) {
        if (valid_reg(r)) {
            val[(size_t)r] = v;
            val_known.set((size_t)r);
            val_srt_key_known.reset((size_t)r);
        }
    };
    auto forget = [&](int r) {
        if (valid_reg(r)) {
            val_known.reset((size_t)r);
            val_srt_key_known.reset((size_t)r);
        }
    };
    for (uint32_t i = 0; i < nsgpr; i++)
        set_value((int)(user_sgpr_base + i), user_sgprs[i]);

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

    for (const auto& in : ins) {
        if (in.is_end) break;
        const bool scalar_spill = in.fmt == Rdna2Format::VOP3 &&
                                  (in.opcode == 0x360 || in.opcode == 0x361);
        if (!scalar_spill && in.dst.kind == OperandKind::VGPR) {
            // Any ordinary write replaces the whole vector result. Drop scalar values previously
            // packed into that VGPR's lanes rather than restoring stale descriptor words later.
            for (uint32_t lane = 0; lane < 64; ++lane)
                scalar_spill_slots.erase(((uint32_t)in.dst.value << 6) | lane);
        }
        switch (in.fmt) {
            case Rdna2Format::SOP1:
                if (in.opcode == 0x03) {                        // s_mov_b32
                    uint32_t v, source_key = 0;
                    const bool source_key_known =
                        in.src[0].kind == OperandKind::SGPR && valid_reg(in.src[0].value) &&
                        val_srt_key_known.test((size_t)in.src[0].value) &&
                        (source_key = val_srt_key[(size_t)in.src[0].value], true);
                    if (in.src[0].kind == OperandKind::Literal ? (v = in.literal, true) : srcval(in.src[0], v)) {
                        set_value(in.dst.value, v);
                        if (source_key_known && valid_reg(in.dst.value)) {
                            val_srt_key[(size_t)in.dst.value] = source_key;
                            val_srt_key_known.set((size_t)in.dst.value);
                        }
                    } else forget(in.dst.value);
                } else if (in.dst.kind == OperandKind::SGPR) {
                    // Not the modeled s_mov_b32 -> the dest is unknown. Erase the PAIR: 64-bit SOP1 ops
                    // (s_mov_b64, s_getpc_b64, s_and/or/xor/not_b64, s_*_saveexec_b64, …) write
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
                uint32_t a, c; bool ka, kc;
                ka = (in.src[0].kind == OperandKind::Literal) ? (a = in.literal, true) : srcval(in.src[0], a);
                kc = (in.src[1].kind == OperandKind::Literal) ? (c = in.literal, true) : srcval(in.src[1], c);
                int d = in.dst.value; bool ok = ka && kc; uint32_t r = 0;
                uint32_t hi64 = 0; bool wrote_pair = false;
                if (ok) switch (in.opcode) {
                    case 0x00: case 0x02: r = a + c; break;                 // s_add_u32 / s_add_i32
                    case 0x01: case 0x03: r = a - c; break;                 // s_sub_u32 / s_sub_i32
                    case 0x0E: r = a & c; break;                            // s_and_b32
                    case 0x10: r = a | c; break;                            // s_or_b32
                    case 0x12: r = a ^ c; break;                            // s_xor_b32
                    case 0x1E: r = a << (c & 31); break;                    // s_lshl_b32
                    case 0x20: r = a >> (c & 31); break;                    // s_lshr_b32
                    case 0x26: r = a * c; break;                            // s_mul_i32
                    case 0x31: r = (a << 4) + c; break;                     // s_lshl4_add_u32
                    case 0x27: { uint32_t off = c & 0x1f, wid = (c >> 16) & 0x7f;   // s_bfe_u32
                                 r = wid == 0 ? 0 : (wid >= 32 ? (a >> off) : ((a >> off) & ((1u << wid) - 1))); break; }
                    case 0x0A:   // s_cselect_b32: dst = SCC ? src0 : src1 (the vertex-fetch format patch's tail)
                        if (scc < 0) ok = false; else r = scc ? a : c;
                        break;
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
                        r = (uint32_t)res; hi64 = (uint32_t)(res >> 32); wrote_pair = true; break;
                    }
                    default: ok = false; break;                            // SCC-dependent / unmodeled -> unknown
                }
                if (trc)   // unfiltered like the SMEM/MUBUF traces (one shader walk — volume is bounded)
                    fprintf(stderr, "[dyntrace]   SOP2 pc=%u op=0x%x dst=s%d src0=%d(k%d) src1=%d(k%d) ok=%d r=0x%x\n",
                            in.pc, in.opcode, d, in.src[0].value, ka, in.src[1].value, kc, ok, r);
                // Every SOP2 ALU op except s_cselect writes SCC to a value we don't track here — invalidate so a
                // later s_cselect only trusts SCC set by an immediately-preceding s_cmp.
                if (in.opcode != 0x0A) scc = -1;
                if (ok) { set_value(d, r); if (wrote_pair) set_value(d + 1, hi64); }
                // A 64-bit-dst op (s_bfe_u64) invalidates BOTH dwords even when its sources were
                // unknown (the opcode switch never ran, so wrote_pair may still be false).
                else    { forget(d); if (wrote_pair || in.opcode == 0x29) forget(d + 1); }
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
                    if (current_known && current_key_known) {
                        pending_srt_use.kind = 1;
                        pending_srt_use.key = current_key;
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
                    for (uint32_t k = 0; k < n; k++) forget(sdst + (int)k);
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
                                      for (uint32_t k = 0; k < n; k++) forget(sdst + (int)k); break; }
                const uint32_t* mem = (const uint32_t*)(uintptr_t)addr;
                const bool imm_only = (soff_field == 125) && (int32_t)in.literal >= 0;   // SGPR_NULL soffset
                for (uint32_t k = 0; k < n; k++) set_value(sdst + (int)k, mem[k]);
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
                break;
            }
            case Rdna2Format::MIMG: {
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
                    const std::array<uint32_t, 8>* t8 = have_t8
                        ? &descr8[(size_t)tbase] : nullptr;
                    std::array<uint32_t, 8> seed_t8{};
                    const uint32_t tkey = have_key
                        ? descr8_key[(size_t)tbase] : 0xFFFFFFFFu;
                    bool from_seed = false;
                    if (!t8 && untouched_seed_range(tbase, 8)) {
                        for (int k = 0; k < 8; k++)
                            seed_t8[k] = user_sgprs[tbase - (int)user_sgpr_base + k];
                        DecodedImageDescriptor d = decode_image_descriptor(seed_t8.data());
                        // A direct T# has no s_load provenance key, so it is resolved by this MIMG's
                        // exact pc. Reject obviously non-image user data before creating that mapping.
                        if (d.base > 0x10000 && d.width && d.height &&
                            d.width <= 16384 && d.height <= 16384 &&
                            d.type >= 8 && d.type <= 15) {
                            t8 = &seed_t8;
                            from_seed = true;
                        }
                    }
                    if (trc) fprintf(stderr, "[dyntrace] MIMG pc=%u srsrc=s%d ssamp=s%d have_t8=%d seed_t8=%d key=0x%x t8[0]=0x%x\n",
                                     in.pc, tbase, samp_base, have_t8, from_seed,
                                     tkey, t8 ? (*t8)[0] : 0u);
                    if (t8) {
                        SrtUse u; u.kind = 0; u.t8 = *t8;
                        u.key = tkey;
                        u.use_pc = in.pc;
                        u.is_store = in.opcode == 0x08;   // image_store: a STORAGE-image use (#590)
                        if (valid_reg(samp_base) && descr_known.test((size_t)samp_base)) {
                            u.has_samp = true;
                            u.s4 = descr[(size_t)samp_base];
                        } else if (untouched_seed_range(samp_base, 4)) {
                            // Direct S# paired with the direct/table T#. Samplers have no address or
                            // extent field to validate; block membership plus reload invalidation is
                            // the conservative provenance check.
                            for (int k = 0; k < 4; k++)
                                u.s4[k] = user_sgprs[samp_base - (int)user_sgpr_base + k];
                            u.has_samp = true;
                        }
                        srt_uses->push_back(u);
                    }
                }
                break;
            }
            case Rdna2Format::MUBUF: {
                // Descriptor-table V# use (#273 — the title post-chain PSes' per-lane structured-buffer
                // fetches): a buffer_load_format_* / buffer_load_dword* whose SRSRC V# was s_loaded from
                // a keyed table slot. Report it as a kind-1 (buffer) use so the consuming instruction
                // resolves via its sreg_srt tag -> by_srt_offset — the same key model the s_buffer_loads
                // use. (The VS vertex-fetch path below is unchanged: by_fetch_pc still wins there.)
                if (srt_uses && (in.opcode <= 3 || (in.opcode >= 0x0C && in.opcode <= 0x0F))) {
                    int srsrc = in.src[1].value;
                    if (valid_reg(srsrc) && descr_known.test((size_t)srsrc)) {
                        // Key-less snapshots (an s_buffer_load-fetched V# — a structured-buffer
                        // descriptor stored INSIDE a constant buffer, DOLL's title post PSes) carry
                        // the consuming instruction's pc instead; the recompiler resolves those via
                        // ShaderResource::fetch_pc with the faithful (non-vertex-index) address path.
                        SrtUse u; u.kind = 1; u.v4 = descr[(size_t)srsrc];
                        u.key = descr_key_known.test((size_t)srsrc)
                            ? descr_key[(size_t)srsrc] : 0xFFFFFFFFu;
                        u.use_pc = in.pc;
                        srt_uses->push_back(u);
                    } else if (in.opcode >= 0x0C && untouched_seed_range(srsrc, 4)) {
                        // Direct RAW V#: an untyped MUBUF can consume a V# placed in the initial
                        // user-data SGPRs without any preceding s_load. This is not a vertex-format
                        // fetch, so the dyn_vb path below never reports it. Preserve the exact V# as
                        // a pc-keyed buffer view; treating the same eight SGPRs as a plausible T# can
                        // otherwise make raw MUBUF resolve to a texture (stride 0), dropping idxen.
                        SrtUse u; u.kind = 1; u.key = 0xFFFFFFFFu; u.use_pc = in.pc;
                        for (int k = 0; k < 4; ++k)
                            u.v4[k] = user_sgprs[srsrc - (int)user_sgpr_base + k];
                        DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
                        if (d.base > 0x10000 && d.size_bytes != 0 &&
                            d.size_bytes <= 0x10000000u && d.stride != 0) {
                            if (trc) fprintf(stderr,
                                             "[dyntrace] raw MUBUF pc=%u direct V# s%d base=0x%llx "
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
                    if (patched)          out.push_back({ in.pc, srsrc, with_off(decode_buffer_descriptor(vv)), vv[3] });
                    else if (have_descr)
                        out.push_back({ in.pc, srsrc,
                                        with_off(decode_buffer_descriptor(descr[(size_t)srsrc].data())),
                                        descr[(size_t)srsrc][3] });
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
                        if (d.base > 0x10000 && d.size_bytes != 0 && d.size_bytes <= 0x10000000u) {
                            if (trc) fprintf(stderr, "[dyntrace]   MUBUF pc=%u seed-V# fallback SRSRC=s%d base=0x%llx\n",
                                             in.pc, srsrc, (unsigned long long)d.base);
                            out.push_back({ in.pc, srsrc, with_off(d), sv[3], /*from_seed=*/true });
                        }
                    }
                }
                break;
            }
            case Rdna2Format::VOP3:
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

namespace { constexpr uint32_t kPsBindingBase = 32; }

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

std::shared_ptr<ShaderResourceTable> build_stage_table(const GpuState& st, uint64_t code_addr, bool is_ps) {
    if (!code_addr) return nullptr;
    const auto* hdr = (const AgcShaderHeader*)prosper_agc_shader_header_for_code(code_addr);
    if (!hdr) return nullptr;
    using StageClock = std::chrono::steady_clock;
    const bool phase_timing = getenv("PROSPER_RENDER_TIMING") != nullptr;
    const auto metadata_start = phase_timing ? StageClock::now() : StageClock::time_point{};
    namespace P = prosper::agc::Pm4;
    const bool log = getenv("PROSPER_GFXLOG") != nullptr;

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
            fprintf(stderr, "[resdump]   sgprs@0x%x:", base);
            for (uint32_t i = 0; i < kUserSgprs; i++) fprintf(stderr, " %08x", sgprs[i]);
            fprintf(stderr, "\n");
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
    const auto metadata_done = phase_timing ? StageClock::now() : StageClock::time_point{};
    if (is_ps) {
        uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, bases[0] + range_start, sgprs);
        dyn_vb = resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, 0x4000,
                                       sgprs, kUserSgprs, 0, &srt_uses);
    } else {
        uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, bases[0] + range_start, sgprs);
        // NGG merged VS/GS: s0..s7 are system SGPRs, user data starts at s8 (confirmed by matching the
        // shader's s[8:11]/s[24:25] descriptor pointers to the register file at GS_0+offset).
        dyn_vb = resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, 0x4000,
                                       sgprs, kUserSgprs, 8, &srt_uses);
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
    const uint32_t user_sgpr_base = is_ps ? 0u : 8u;
    for (uint32_t base : bases) {
        uint32_t sgprs[kUserSgprs]; read_user_sgprs(st.sh, base + range_start, sgprs);
        ShaderResourceTable t = build_shader_resources(*hdr, sgprs, kUserSgprs, user_sgpr_base);
        // Add the const-fold-resolved dynamic buffers, keyed by their SRSRC SGPR so the
        // recompiler's by_sgpr_base() resolves each buffer_load_format. The V#'s data format is patched
        // at runtime by the fetch shader (so the load-time snapshot reads Unknown) — default to Float32
        // (a raw 32-bit-per-component fetch, correct for float attributes like positions).
        for (auto& kv : dyn_vb) {
            const auto& d = kv.desc;
            // A SEED-fallback entry must not shadow a metadata-described DIRECT vertex buffer at the
            // same SGPRs (see DynFetch::from_seed): the direct resource resolves the fetch through
            // the faithful address path, which is the correct model for a single un-patched V#.
            if (kv.from_seed) {
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
            r.format        = (d.format == DataFormat::Unknown) ? DataFormat::Float32 : d.format;
            r.num_components = d.num_components ? d.num_components : 4;
            r.gpu_addr      = d.base;
            r.size          = d.size_bytes ? d.size_bytes : (d.stride ? d.stride * 4 : 128);
            r.stride        = d.stride;
            r.sgpr_base     = kv.srsrc;           // DIRECT provenance = the fetch's SRSRC SGPR (fallback)
            r.fetch_pc      = kv.fetch_pc;        // PER-FETCH provenance = the exact fetch instruction
            r.srt_offset    = 0xFFFFFFFFu;
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
                uint64_t dk = (u.kind == 0 || u.key == 0xFFFFFFFFu)
                                  ? (0x8000000000000000ull | ((uint64_t)(uint32_t)u.kind << 32) | u.use_pc)
                                  : ((uint64_t)(uint32_t)u.kind << 32) | u.key;
                if (!srt_seen.insert(dk).second) continue;
                bool clash = u.key == 0xFFFFFFFFu;       // key-less: never resolvable by srt_offset
                if (!clash)
                    for (const auto& r0 : t.resources) if (r0.srt_offset == u.key) { clash = true; break; }
                if (u.kind == 1) {                       // constant buffer / structured-buffer V#
                    DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
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
                    if (clash) {
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
                    r.format = d.format; r.num_components = d.num_components ? d.num_components : 1;
                    r.gpu_addr = d.base; r.size = resource_size; r.stride = resource_stride;
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;
                    if (clash) r.fetch_pc = u.use_pc;    // pc-only provenance (key-less/collided V#)
                    if (log) fprintf(stderr, "[srt] %s cbuf key=0x%x pc=%u base=0x%llx size=%u\n", is_ps ? "PS" : "VS",
                                     u.key, u.use_pc, (unsigned long long)d.base, resource_size);
                    t.resources.push_back(r);
                } else {                                  // texture (T# [+ paired S#])
                    DecodedImageDescriptor d = decode_image_descriptor(u.t8.data());
                    if (g_dyntrace_force)
                        fprintf(stderr, "[dynfail] tex use pc=%u key=0x%x base=0x%llx %ux%u fmt=%u tile=%u\n",
                                u.use_pc, u.key, (unsigned long long)d.base, d.width, d.height,
                                d.format, d.tile_mode);
                    if (d.base == 0 || d.width == 0 || d.height == 0 ||
                        d.width > 16384 || d.height > 16384) continue;       // garbage/degenerate T#
                    // A previous use already produced a resource for this SAME selected view (address +
                    // extent): don't duplicate the binding/upload — give it this use's pc provenance
                    // if it has none yet (#273). If it already carries a DIFFERENT use's pc, fall
                    // through and create a second resource for this pc (fetch_pc holds one pc; a
                    // sample whose pc has no mapping would stay unresolved).
                    Gen5ImageFormatInfo fi;
                    if (!gen5_image_format(d.format, &fi)) {
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
                            if (r0.cls == ResourceClass::Texture && r0.gpu_addr == view.base &&
                                r0.width == view.width && r0.height == view.height &&
                                r0.depth == d.depth && r0.format == fi.format && r0.img_dim == img_dim) {
                                if (r0.fetch_pc == 0xFFFFFFFFu) { r0.fetch_pc = u.use_pc; mapped = true; break; }
                                if (r0.fetch_pc == u.use_pc)    { mapped = true; break; }
                            }
                        if (mapped) continue;
                    }
                    ShaderResource r;
                    r.cls = ResourceClass::Texture;
                    r.format = fi.format; r.num_components = fi.num_components;
                    r.gpu_addr = view.base; r.width = view.width; r.height = view.height; r.depth = d.depth;
                    r.tile_mode = d.tile_mode; r.srgb = fi.srgb;
                    r.max_uncompressed_block_size = d.max_uncompressed_block_size;
                    r.max_compressed_block_size = d.max_compressed_block_size;
                    const bool shifted_view = view.mip_offset != 0;
                    r.meta_pipe_aligned = shifted_view ? false : d.meta_pipe_aligned;
                    r.write_compress_enabled = shifted_view ? false : d.write_compress_enabled;
                    r.compression_enabled = shifted_view ? false : d.compression_enabled;
                    r.alpha_is_on_msb = d.alpha_is_on_msb;
                    r.color_transform = d.color_transform;
                    r.metadata_addr = shifted_view ? 0 : d.metadata_addr;
                    // T# TYPE -> MIMG dim (GFX10: 9=2D, 10=3D, 11=CUBE, 13=2D_ARRAY); a cube
                    // uploads as six vertically-stacked faces (#273 — see agc_shader_layout).
                    r.img_dim = img_dim;
                    r.swizzle[0] = d.dst_sel[0]; r.swizzle[1] = d.dst_sel[1];
                    r.swizzle[2] = d.dst_sel[2]; r.swizzle[3] = d.dst_sel[3];
                    const uint64_t backing_bytes = is_bcn
                        ? static_cast<uint64_t>((view.width + 3) / 4) * ((view.height + 3) / 4) * d.depth * fi.bytes_per_block
                        : static_cast<uint64_t>(view.width) * view.height * d.depth * fi.bytes_per_block;
                    if (!backing_bytes || backing_bytes > UINT32_MAX) continue;
                    r.size = static_cast<uint32_t>(backing_bytes);
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;   // ambiguous/absent key: pc-only provenance
                    r.fetch_pc   = u.use_pc;                       // per-instruction provenance (#273)
                    if (u.has_samp) {                     // paired S# (same SQ_IMG_SAMP decode as the sharp path)
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
                    if (log) fprintf(stderr, "[srt] %s tex key=0x%x %ux%u fmt=%u base=0x%llx tile=%u samp=%d\n",
                                     is_ps ? "PS" : "VS", u.key, d.width, d.height, d.format,
                                     (unsigned long long)d.base, d.tile_mode, (int)u.has_samp);
                    t.resources.push_back(r);
                }
            }
        }
        if (t.resources.empty()) continue;
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
    out.local_x = reg(P::COMPUTE_NUM_THREAD_X);
    out.local_y = reg(P::COMPUTE_NUM_THREAD_Y);
    out.local_z = reg(P::COMPUTE_NUM_THREAD_Z);
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
    return out;
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
        const uint64_t code_addr = (static_cast<uint64_t>(rd(ds.sh, P::COMPUTE_PGM_LO)) << 8) |
                                   (static_cast<uint64_t>(rd(ds.sh, P::COMPUTE_PGM_HI) & 0xffu) << 40);
        OperationRealizationFailure failure;
        failure.kind = SubmitOperationKind::Dispatch;
        failure.index = dispatch_index;
        failure.command_order = dispatch.command_order;
        failure.compute_launch = resolve_compute_launch(dispatch);
        auto record_failure = [&](RealizationFailureReason reason,
                                  const std::shared_ptr<ShaderResourceTable>& resources,
                                  const std::vector<uint32_t>& spirv) {
            if (!failures) return;
            failure.reason = reason;
            ShaderRealizationDiagnostic stage;
            stage.stage = ShaderProgramStage::Compute;
            stage.program_addr = code_addr;
            stage.resources = resources;
            stage.recompiled = !spirv.empty();
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

        uint32_t range_start = 0;
        if (header->specials && guest_readable((uint64_t)(uintptr_t)header->specials,
                                               sizeof(AgcShaderSpecials))) {
            const uint32_t start = header->specials->user_data_range_start;
            const uint32_t end = header->specials->user_data_range_end;
            if (start < kUserSgprs && end > start && end <= 2 * kUserSgprs) range_start = start;
        }
        uint32_t sgprs[kUserSgprs] = {};
        read_user_sgprs(ds.sh, P::COMPUTE_USER_DATA_0 + range_start, sgprs);
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
            std::vector<SrtUse> srt_uses;
            resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, 0x4000,
                                  sgprs, kUserSgprs, /*user_sgpr_base*/0, &srt_uses);
            std::set<uint64_t> srt_seen;
            for (const auto& u : srt_uses) {
                // Dedupe: keyed cbuf uses per key; texture/storage and key-less uses per consuming pc
                // (distinct namespaces so pc keys never collide with byte-offset keys).
                uint64_t dk = (u.kind == 0 || u.key == 0xFFFFFFFFu)
                                  ? (0x8000000000000000ull | ((uint64_t)(uint32_t)u.kind << 32) | u.use_pc)
                                  : ((uint64_t)(uint32_t)u.kind << 32) | u.key;
                if (!srt_seen.insert(dk).second) continue;
                bool clash = u.key == 0xFFFFFFFFu;
                if (!clash)
                    for (const auto& r0 : table->resources)
                        if (r0.srt_offset == u.key) { clash = true; break; }
                if (u.kind == 1) {                       // constant/structured-buffer V#
                    DecodedBufferDescriptor d = decode_buffer_descriptor(u.v4.data());
                    if (d.base <= 0x10000 || d.size_bytes == 0 || d.size_bytes > 0x10000000u) continue;
                    if (clash) {
                        bool piggybacked = false;
                        for (auto& r0 : table->resources)
                            if (r0.cls == ResourceClass::ConstantBuffer &&
                                r0.gpu_addr == d.base && r0.size == d.size_bytes) {
                                if (r0.fetch_pc == 0xFFFFFFFFu) r0.fetch_pc = u.use_pc;
                                piggybacked = r0.fetch_pc == u.use_pc;
                                if (piggybacked) break;
                            }
                        if (piggybacked) continue;
                    }
                    ShaderResource r;
                    r.cls = ResourceClass::ConstantBuffer;
                    r.format = d.format; r.num_components = d.num_components ? d.num_components : 1;
                    r.gpu_addr = d.base; r.size = d.size_bytes; r.stride = d.stride;
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;
                    if (clash) r.fetch_pc = u.use_pc;    // pc-only provenance (key-less/collided V#)
                    table->resources.push_back(r);
                } else {                                  // T# — storage image (store) or sampled texture
                    DecodedImageDescriptor d = decode_image_descriptor(u.t8.data());
                    if (d.base == 0 || d.width == 0 || d.height == 0 ||
                        d.width > 16384 || d.height > 16384) continue;   // garbage/degenerate T#
                    Gen5ImageFormatInfo fi;
                    const bool mapped_fmt = gen5_image_format(d.format, &fi);
                    // Unknown sampled formats cannot be decoded. Unknown storage formats may still
                    // recompile (format-free SPIR-V), but remain explicitly Unknown so the live backend
                    // rejects them instead of silently treating arbitrary bytes as RGBA8.
                    if (!mapped_fmt && !u.is_store) continue;
                    if (mapped_fmt && fi.block_width > 1 && fi.snorm) continue;   // signed BCn: not wired
                    const DecodedImageView view = mapped_fmt
                        ? image_base_level_view(d, fi)
                        : DecodedImageView{d.base, d.width, d.height, 0, d.base_level == 0};
                    if (!view.supported) {
                        warn_unsupported_image_view(d);
                        continue;
                    }
                    const ResourceClass wanted = u.is_store ? ResourceClass::StorageImage
                                                           : ResourceClass::Texture;
                    const DataFormat view_format = mapped_fmt ? fi.format : DataFormat::Unknown;
                    const uint32_t img_dim = image_type_to_dim(d.type);
                    {
                        bool mapped = false;
                        for (auto& r0 : table->resources)
                            if (r0.cls == wanted && r0.gpu_addr == view.base &&
                                r0.width == view.width && r0.height == view.height &&
                                r0.depth == d.depth && r0.format == view_format &&
                                r0.img_dim == img_dim) {
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
                        const uint64_t bytes = is_bcn
                            ? static_cast<uint64_t>((view.width + 3) / 4) * ((view.height + 3) / 4) * d.depth * fi.bytes_per_block
                            : static_cast<uint64_t>(view.width) * view.height * d.depth * fi.bytes_per_block;
                        if (!bytes || bytes > UINT32_MAX) continue;
                        r.size = static_cast<uint32_t>(bytes);
                        r.srgb = fi.srgb;
                    } else {
                        const uint64_t bytes = static_cast<uint64_t>(d.width) * d.height * d.depth * 4;
                        if (!bytes || bytes > UINT32_MAX) continue;
                        r.format = DataFormat::Unknown; r.num_components = 4;
                        r.size = static_cast<uint32_t>(bytes);
                    }
                    r.gpu_addr = view.base; r.width = view.width; r.height = view.height; r.depth = d.depth;
                    r.tile_mode = d.tile_mode;
                    r.max_uncompressed_block_size = d.max_uncompressed_block_size;
                    r.max_compressed_block_size = d.max_compressed_block_size;
                    const bool shifted_view = view.mip_offset != 0;
                    r.meta_pipe_aligned = shifted_view ? false : d.meta_pipe_aligned;
                    r.write_compress_enabled = shifted_view ? false : d.write_compress_enabled;
                    r.compression_enabled = shifted_view ? false : d.compression_enabled;
                    r.alpha_is_on_msb = d.alpha_is_on_msb;
                    r.color_transform = d.color_transform;
                    r.metadata_addr = shifted_view ? 0 : d.metadata_addr;
                    r.img_dim = img_dim;
                    r.swizzle[0] = d.dst_sel[0]; r.swizzle[1] = d.dst_sel[1];
                    r.swizzle[2] = d.dst_sel[2]; r.swizzle[3] = d.dst_sel[3];
                    r.srt_offset = clash ? 0xFFFFFFFFu : u.key;
                    r.fetch_pc = u.use_pc;               // per-use pc provenance (the image op)
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
                0x4000, resource.sgpr_base);
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
        assign_convention_bindings(*table, 2);

        const uint32_t rsrc2 = rd(ds.sh, P::COMPUTE_PGM_RSRC2);
        auto field = [&](uint32_t shift, uint32_t mask) { return (rsrc2 >> shift) & mask; };
        ComputeShaderConfig config;
        const uint32_t user_count = field(P::COMPUTE_PGM_RSRC2_USER_SGPR_SHIFT,
                                          P::COMPUTE_PGM_RSRC2_USER_SGPR_MASK);
        config.user_sgprs.assign(sgprs, sgprs + std::min(user_count, kUserSgprs));
        const ComputeLaunchDimensions launch = resolve_compute_launch(dispatch);
        config.local_x = launch.local_x;
        config.local_y = launch.local_y;
        config.local_z = launch.local_z;
        config.exact_thread_extent = ((dispatch.modifier >>
            P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_SHIFT) &
            P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_MASK) != 0;
        config.threads_x = launch.threads_x;
        config.threads_y = launch.threads_y;
        config.threads_z = launch.threads_z;
        config.wave_size = ((dispatch.modifier >>
            P::COMPUTE_DISPATCH_INITIATOR_CS_W32_EN_SHIFT) &
            P::COMPUTE_DISPATCH_INITIATOR_CS_W32_EN_MASK) ? 32u : 64u;
        config.tgid_x_en = field(P::COMPUTE_PGM_RSRC2_TGID_X_EN_SHIFT,
                                 P::COMPUTE_PGM_RSRC2_TGID_X_EN_MASK) != 0;
        config.tgid_y_en = field(P::COMPUTE_PGM_RSRC2_TGID_Y_EN_SHIFT,
                                 P::COMPUTE_PGM_RSRC2_TGID_Y_EN_MASK) != 0;
        config.tgid_z_en = field(P::COMPUTE_PGM_RSRC2_TGID_Z_EN_SHIFT,
                                 P::COMPUTE_PGM_RSRC2_TGID_Z_EN_MASK) != 0;
        config.tg_size_en = field(P::COMPUTE_PGM_RSRC2_TG_SIZE_EN_SHIFT,
                                  P::COMPUTE_PGM_RSRC2_TG_SIZE_EN_MASK) != 0;
        config.tidig_comp_cnt = field(P::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_SHIFT,
                                      P::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_MASK);
        config.lds_bytes = field(P::COMPUTE_PGM_RSRC2_LDS_SIZE_SHIFT,
                                 P::COMPUTE_PGM_RSRC2_LDS_SIZE_MASK) * 512u;

        ComputeItem item;
        item.spirv = recompile_compute((const uint32_t*)(uintptr_t)code_addr, 0x10000,
                                       table.get(), config);
        if (!item.spirv.empty() && getenv("PROSPER_SHADER_DUMP_SUCCESS")) {
            const ShaderCompileKey dump_key = make_shader_compile_key(
                ShaderProgramStage::Compute,
                reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(code_addr)),
                0x10000, table.get(), nullptr, nullptr);
            maybe_dump_successful_shader(ShaderProgramStage::Compute, dump_key, item.spirv);
        }
        item.resources = std::move(table);
        item.launch = launch;
        item.code_addr = code_addr;
        item.dispatch_index = dispatch_index;
        item.submit_no = submit_no;
        item.command_order = dispatch.command_order;
        if (item.spirv.empty()) {
            record_failure(RealizationFailureReason::ShaderRecompile, item.resources, item.spirv);
            // PROSPER_DYNTRACE_FAIL=1: replay the FAILED compute program's resource build with the
            // const-fold walk trace + user-data dump forced on (once per distinct program) — the
            // compute analog of the graphics VS/PS fail-replay (gpu_execute.hpp). Compute dispatches
            // only run build_shader_resources (the DIRECT front-half table) and NEVER the const-fold
            // resolve_dynamic_fetch that graphics stages use, so a bindless storage-image U# / T#
            // (image_store SRSRC, image_sample) can't be recovered by pc-provenance for a dispatch.
            // This trace reveals whether the failing dispatch's descriptors ARE const-fold-resolvable
            // (i.e. loaded via a foldable s_load chain) — the ground truth for #590 before extending
            // the compute resource build. Read-only: it does not change what the executor binds.
            if (getenv("PROSPER_DYNTRACE_FAIL")) {
                static std::set<uint64_t> traced_cs;
                if (traced_cs.insert(code_addr).second) {
                    std::fprintf(stderr, "[dynfail] replaying COMPUTE 0x%llx resource build with trace:\n",
                                 (unsigned long long)code_addr);
                    std::fprintf(stderr, "[dynfail]   compute user-data SGPRs (s0..s%u):\n", kUserSgprs - 1);
                    for (uint32_t i = 0; i < kUserSgprs; i += 4)
                        std::fprintf(stderr, "[dynfail]     s%-2u: %08x %08x %08x %08x\n", i,
                                     sgprs[i], sgprs[i + 1], sgprs[i + 2], sgprs[i + 3]);
                    std::vector<SrtUse> cs_uses;
                    g_dyntrace_force = true;
                    resolve_dynamic_fetch((const uint32_t*)(uintptr_t)code_addr, 0x4000,
                                          sgprs, kUserSgprs, 0, &cs_uses);
                    g_dyntrace_force = false;
                    std::fprintf(stderr, "[dynfail]   const-fold recovered %zu descriptor use(s):\n",
                                 cs_uses.size());
                    for (const auto& u : cs_uses)
                        std::fprintf(stderr, "[dynfail]     %s key=0x%x use_pc=%u dw0=0x%x\n",
                                     u.kind == 0 ? "TEX/IMG(t8)" : "BUF(v4)", u.key, u.use_pc,
                                     u.kind == 0 ? u.t8[0] : u.v4[0]);
                }
            }
            static std::set<uint64_t> logged;
            if (logged.insert(code_addr).second) {
                std::fprintf(stderr, "[compute] skip unsupported program 0x%llx\n",
                             (unsigned long long)code_addr);
                // PROSPER_SHADER_DUMP=<dir>: write the failed COMPUTE program's raw bytes for offline
                // shader_inspect, mirroring the graphics VS/PS dump (gpu_execute.hpp). The graphics
                // path was the only dumper, so a failing dispatch's CFG could not be mapped offline.
                // Bounded like the graphics dump (64 KB covers the largest observed shaders); shrink
                // to the readable prefix instead of faulting on a short final mapping.
                if (const char* dd = getenv("PROSPER_SHADER_DUMP")) {
                    size_t n = 0x10000;
                    while (n >= 0x1000 && !guest_readable(code_addr, (uint32_t)n)) n >>= 1;
                    if (n >= 0x1000) {
                        char fn[512];
                        snprintf(fn, sizeof fn, "%s/exec_cs_%llx.bin", dd,
                                 (unsigned long long)code_addr);
                        if (FILE* f = fopen(fn, "wb")) {
                            fwrite((const void*)(uintptr_t)code_addr, 1, n, f);
                            fclose(f);
                        }
                    }
                }
            }
            continue;
        }
        const DescriptorValidationReport report = validate_spirv_descriptor_interface(
            item.spirv, item.resources.get(), 0, SpirvShaderStage::Compute, false);
        if (!report.ok()) {
            record_failure(RealizationFailureReason::DescriptorContract, item.resources, item.spirv);
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

static OrderedSubmitResult execute_ordered_gpustate(const GpuState& st, uint32_t width,
                                                     uint32_t height, uint64_t submit_no,
                                                     const LiveRenderFn& render,
                                                     const LiveComputeFn& compute);

bool execute_nonrender_submit_work(const GpuState& st, uint64_t submit_no) {
    if (st.dma_copies.empty() && (!g_compute || st.dispatches.empty())) return false;
    GuestReadableSubmitScope guest_readable_scope;
    const OrderedSubmitResult result = execute_ordered_gpustate(
        st, 0, 0, submit_no, {}, g_compute);
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
    auto pgm_addr = [&](const GpuState& ds) {
        uint32_t lo = rd(ds.sh, P::COMPUTE_PGM_LO), hi = rd(ds.sh, P::COMPUTE_PGM_HI);
        return (static_cast<uint64_t>(lo) << 8) | (static_cast<uint64_t>(hi & 0xffu) << 40);
    };

    size_t matched = 0;
    for (size_t i = 0; i < st.dispatches.size(); ++i) {
        const auto& d = st.dispatches[i];
        const GpuState& ds = d.state ? *d.state : st;
        const ComputeLaunchDimensions launch = resolve_compute_launch(d);
        const uint64_t code_addr = pgm_addr(ds);
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
            auto prt = build_stage_table(ds, rs.ps_addr, true);
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
    operations.reserve(st.draws.size() + st.dispatches.size());
    for (size_t i = 0; i < st.draws.size(); ++i)
        operations.push_back({SubmitOperationKind::Draw, i, st.draws[i].command_order});
    for (size_t i = 0; i < st.dispatches.size(); ++i)
        operations.push_back({SubmitOperationKind::Dispatch, i, st.dispatches[i].command_order});
    std::stable_sort(operations.begin(), operations.end(), [](const auto& a, const auto& b) {
        return a.command_order < b.command_order;
    });
    return operations;
}

OrderedSubmitResult execute_ordered_items(const std::vector<SubmitOperation>& operations,
                                          const std::vector<DrawItem>& draws,
                                          const std::vector<ComputeItem>& computes,
                                          const std::vector<GpuState::DmaCopy>& dma_copies,
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height) {
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
    for (const auto& operation : operations) {
        if (operation.kind == SubmitOperationKind::Draw) {
            auto it = draw_by_index.find(operation.index);
            if (it != draw_by_index.end())
                executable.push_back({ExecutableKind::Draw, it->second, operation.command_order});
        } else {
            auto it = compute_by_index.find(operation.index);
            if (it != compute_by_index.end())
                executable.push_back({ExecutableKind::Dispatch, it->second, operation.command_order});
        }
    }
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
            const GpuState::DmaCopy& copy = dma_copies[operation.item];
            std::vector<uint8_t> current_source;
            const LiveTargetByteReadResult source_result = read_live_render_target_bytes(
                copy.src, copy.bytes, current_source);
            if (source_result == LiveTargetByteReadResult::InvalidRange) {
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    std::fprintf(stderr,
                                 "[agc] DMA_DATA live-target source range invalid: src=0x%llx bytes=%u\n",
                                 static_cast<unsigned long long>(copy.src), copy.bytes);
                continue;
            }
            execute_ordered_dma_copy(
                copy, source_result == LiveTargetByteReadResult::Success
                          ? current_source.data() : nullptr);
        }
    }
    flush_span();
    return result;
}

OrderedSubmitResult execute_ordered_items(const std::vector<SubmitOperation>& operations,
                                          const std::vector<DrawItem>& draws,
                                          const std::vector<ComputeItem>& computes,
                                          const LiveRenderFn& render,
                                          const LiveComputeFn& compute,
                                          uint32_t width, uint32_t height) {
    return execute_ordered_items(operations, draws, computes, {}, render, compute, width, height);
}

namespace {
enum class RetainedSubmitKind : uint8_t { Draw, Dispatch, DmaCopy, MemoryEffect };
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

bool realize_retained_draw(const GpuState& st, size_t index, float scale_x, float scale_y,
                           DrawItem& item) {
    if (index >= st.draws.size() || !retained_draw_selected(st, index)) return false;
    const bool per_draw = use_per_draw_policy(st);
    const GpuState& draw_state = per_draw ? st.state_at_draw(index) : st;
    const GpuState::Draw& draw = st.draws[index];
    const bool log = getenv("PROSPER_GFXLOG") != nullptr || getenv("PROSPER_EXECLOG") != nullptr;
    if (!realize_draw_item(draw_state, &draw, draw.index_count, 0x10000, log, item)) return false;
    item.draw_index = index;
    item.command_order = draw.command_order;
    if ((scale_x != 1.0f || scale_y != 1.0f) && item.ps.has_viewport) {
        item.ps.viewport_x *= scale_x; item.ps.viewport_w *= scale_x;
        item.ps.viewport_y *= scale_y; item.ps.viewport_h *= scale_y;
    }
    return true;
}

bool realize_retained_compute(const GpuState& st, size_t index, uint64_t submit_no,
                              ComputeItem& item) {
    if (index >= st.dispatches.size()) return false;
    // DMA-bearing submits are uncommon. A one-dispatch state keeps the mature realization path
    // intact while ensuring it runs only after every preceding ordered producer has landed.
    GpuState one = st.dispatches[index].state ? *st.dispatches[index].state : st;
    one.dispatches.clear();
    one.dispatches.push_back(st.dispatches[index]);
    std::vector<ComputeItem> realized = realize_compute_dispatches(one, submit_no);
    if (realized.empty()) return false;
    item = std::move(realized.front());
    item.dispatch_index = index;
    return true;
}
} // namespace

static OrderedSubmitResult execute_ordered_gpustate(const GpuState& st, uint32_t width,
                                                     uint32_t height, uint64_t submit_no,
                                                     const LiveRenderFn& render,
                                                     const LiveComputeFn& compute) {
    GuestGpuWriteSubmitScope guest_gpu_write_scope;
    if (st.dma_execution_rejected) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr,
                         "[agc] ordered DMA submit not executed: unsupported eager/deferred "
                         "guest-memory dependency\n");
        return {};
    }
    std::vector<RetainedSubmitOperation> executable;
    executable.reserve(st.draws.size() + st.dispatches.size() + st.dma_copies.size() +
                       st.ordered_memory_effects.size());
    for (size_t i = 0; i < st.draws.size(); ++i)
        if (retained_draw_selected(st, i))
            executable.push_back({RetainedSubmitKind::Draw, i, st.draws[i].command_order});
    for (size_t i = 0; i < st.dispatches.size(); ++i)
        executable.push_back({RetainedSubmitKind::Dispatch, i, st.dispatches[i].command_order});
    for (size_t i = 0; i < st.dma_copies.size(); ++i)
        executable.push_back({RetainedSubmitKind::DmaCopy, i, st.dma_copies[i].command_order});
    for (size_t i = 0; i < st.ordered_memory_effects.size(); ++i)
        executable.push_back({RetainedSubmitKind::MemoryEffect, i,
                              st.ordered_memory_effects[i].cmd.stream_order});
    std::stable_sort(executable.begin(), executable.end(), [](const auto& a, const auto& b) {
        return a.command_order < b.command_order;
    });

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
                DrawItem item;
                if (realize_retained_draw(st, operation.index, scale_x, scale_y, item))
                    span.push_back(std::move(item));
                break;
            }
            case RetainedSubmitKind::Dispatch: {
                flush_span();
                if (!compute) break;
                ComputeItem item;
                if (realize_retained_compute(st, operation.index, submit_no, item))
                    result.compute_executed |= compute({std::move(item)});
                break;
            }
            case RetainedSubmitKind::DmaCopy: {
                flush_span(true);
                const GpuState::DmaCopy& copy = st.dma_copies[operation.index];
                std::vector<uint8_t> current_source;
                const LiveTargetByteReadResult source_result = read_live_render_target_bytes(
                    copy.src, copy.bytes, current_source);
                if (source_result == LiveTargetByteReadResult::InvalidRange) {
                    static std::atomic<int> warned{0};
                    if (warned.fetch_add(1) < 24)
                        std::fprintf(stderr,
                                     "[agc] DMA_DATA live-target source range invalid: src=0x%llx bytes=%u\n",
                                     static_cast<unsigned long long>(copy.src), copy.bytes);
                    break;
                }
                execute_ordered_dma_copy(
                    copy, source_result == LiveTargetByteReadResult::Success
                              ? current_source.data() : nullptr);
                break;
            }
            case RetainedSubmitKind::MemoryEffect:
                flush_span();
                execute_ordered_memory_effect(st.ordered_memory_effects[operation.index]);
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
void set_submit_compute(LiveComputeFn fn) { g_compute = std::move(fn); }
bool have_submit_compute()                { return static_cast<bool>(g_compute); }

static LiveTargetQueryFn g_live_target_query;   // registered by the live renderer (#590)
void set_live_target_query(LiveTargetQueryFn fn) { g_live_target_query = std::move(fn); }
bool is_live_render_target(uint64_t gpu_addr) {
    return g_live_target_query && g_live_target_query(gpu_addr);
}
static LiveTargetReaderFn g_live_target_reader;
void set_live_target_reader(LiveTargetReaderFn fn) { g_live_target_reader = std::move(fn); }
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
    if (g_guest_gpu_writes.active) {
        if (g_guest_gpu_writes.writes.size() < kGuestGpuWriteJournalCapacity)
            g_guest_gpu_writes.writes.push_back({addr, size});
        else
            g_guest_gpu_writes.overflowed = true;
    }
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
    uint32_t fw = present_width(), fh = present_height();
    float sx = fw ? (float)width / (float)fw : 1.0f;
    float sy = fh ? (float)height / (float)fh : 1.0f;
    const bool has_ordered_dma = !st.dma_copies.empty();
    // A DMA-bearing submit realizes consumers at their ordered position below. Pre-realizing here
    // would snapshot old indices/descriptors/shader bytes before the copy updates their backing.
    std::vector<DrawItem> draws = !has_ordered_dma && g_live && width && height
        ? realize_gpustate_draws(st, 0x10000, sx, sy) : std::vector<DrawItem>{};
    const ShaderRecompileCacheStats shader_after = timing_enabled
        ? shader_recompile_cache_stats() : ShaderRecompileCacheStats{};
    const ShaderDecodeCacheStats decode_after = timing_enabled
        ? shader_decode_cache_stats() : ShaderDecodeCacheStats{};
    const DrawRealizationPhaseStats phases_after = timing_enabled
        ? draw_realization_phase_stats() : DrawRealizationPhaseStats{};
    const StageTablePhaseStats table_phases_after = timing_enabled
        ? stage_table_phase_stats() : StageTablePhaseStats{};
    const auto timing_draws_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    std::vector<ComputeItem> computes = !has_ordered_dma && g_compute
        ? realize_compute_dispatches(st, submit_no) : std::vector<ComputeItem>{};
    const auto timing_compute_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};

    auto operations = plan_submit_operations(st);
    const auto timing_plan_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    // Capture formats through v13 have no DMA operation record. Pass that fact into selection so a
    // selected unsupported submit is claimed-and-failed, never silently skipped in favor of a later
    // match. Direct/timeline capture rejects again at the central GpuState boundary.
    auto pending_capture = begin_requested_gpu_capture(
        draws, computes, operations, width, height, has_ordered_dma,
        static_cast<uint64_t>(st.draws.size()));
    OrderedSubmitResult result = has_ordered_dma
        ? execute_ordered_gpustate(st, width, height, submit_no, g_live, g_compute)
        : execute_ordered_items(operations, draws, computes, g_live, g_compute, width, height);
    const auto timing_backend_done = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    const std::vector<uint8_t>& px = result.frame.bytes();

    if (pending_capture) {
        std::string error;
        if (!finish_requested_gpu_capture(std::move(pending_capture), px, error))
            std::fprintf(stderr, "[gpucap] write failed: %s\n", error.c_str());
    }
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
            double realize_draws = 0, realize_compute = 0, plan = 0, backend = 0, publish = 0;
            double table_build = 0, shader_lookup = 0, shader_compile = 0;
            double table_metadata = 0, table_dynamic_fold = 0, table_resources = 0;
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
    std::vector<DrawItem> items = realize_gpustate_draws(st, 0x10000, sx, sy);
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
