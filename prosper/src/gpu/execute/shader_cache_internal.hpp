#pragma once

// Types and small helpers shared between gpu_executor.cpp and the shader compile/decode/analysis
// caches being split out of it (#3516). Internal to src/gpu/execute.
//
// The large cache entry points -- decode_shader_cached, analyze_shader_code_cached,
// select_pcrel_dispatch -- are deliberately NOT here: their bodies stay in the .cpp and callers reach
// them by declaration. Promoting a 177-line function into a header parsed by every includer is the
// opposite of what this split is for.

// gpu_executor.cpp — the live-submit half of the GPU executor (Stage A of docs/GPU_EXECUTOR_DESIGN.md).
//
// Holds the process-wide live render backend and drives it on each AGC submit. This is deliberately the
// ONLY place the executor touches process-global state; execute_gpustate() itself (gpu_execute.hpp) stays
// pure. No Vulkan here — the backend is a std::function injected by whoever owns a device (the runtime
// binary at startup, or a test via render_runner.h), so prosper_core links this without Vulkan.
#include "gpu/resources/fold_control_plan.hpp"
#include "gpu/capture/fold_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "diagnostics/env_submit.hpp"
#include "gpu/diagnostics/watch_list.hpp"   // strict 0x-only watch parsing (shared with the RTT watch)
#include "gpu/diagnostics/diag_ratelimit.hpp"   // first-N-then-powers-of-two report throttling
#include "diagnostics/env_numeric.hpp"   // #3267: a typo must not silently drop an operator-set cap
#include <cstdint>
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/timeline/gpu_timeline.hpp"
#include "gpu/capture/capture_compute_policy.hpp"
#include "gpu/diagnostics/compute_parent_walk.hpp"
#include "gpu/diagnostics/shader_dump_filter.hpp"  // PROSPER_SHADER_DUMP_PROGRAM address filter
#include "gpu/diagnostics/compute_tree_watch.hpp"
#include "gpu/present/videoout_present.hpp"   // present_write_frame
#include "gpu/agc/agc_shader_layout.hpp"  // AgcShaderHeader + build_shader_resources
#include "gpu/resources/mip_chain_plan.hpp"  // shader_resource_compute_mip_chain_levels (#3048)
#include "gpu/pm4/pm4_registers.hpp"      // SPI_SHADER_USER_DATA_* offsets
#include "gpu/recompiler/rdna2_decode.hpp"       // rdna2_walk (for the vertex-fetch const-eval)
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"     // recompile_compute
#include "gpu/capture/writer_provenance.hpp"
#include "host/memory/guest_memory_map.hpp"
#include "host/memory/guest_memory_query.hpp"
#include "host/memory/guest_write_watch.hpp"
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <condition_variable>
#include <filesystem>
#include <iterator>
#include <map>
#include <mutex>
#include <shared_mutex>
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

struct ShaderResourceCompileKey {
    uint32_t cls = 0;
    uint32_t format = 0;
    uint32_t num_components = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t img_dim = 0;
    uint32_t sample_count = 1;
    uint32_t declared_mip_levels = 1;
    // #3048: how many levels the compute backend materializes for this resource. The emitted module
    // bakes it in (the IMAGE_LOAD_MIP LOD clamp is a literal), so two descriptors that differ only
    // in their allocation-wide mip placement must not share a compiled module.
    uint32_t materialized_mip_levels = 1;
    bool in_mip_tail = false;
    bool compression_enabled = false;
    bool proven_zero_mip = false;
    uint32_t binding = 0;
    uint32_t stride = 0;
    uint32_t one_record_tail_semantic = 0;
    uint32_t atomic_x2_record_count = 0;
    uint32_t scalar_buffer_dword_count = 0;
    bool scalar_buffer_contract_valid = true;
    uint32_t srt_offset = 0;
    uint32_t sgpr_base = 0;
    uint32_t fetch_pc = 0;
    uint32_t fetch_index_mode = 0;
    uint32_t table_index_count = 0;
    uint32_t table_entry_stride = 0;
    uint32_t table_index_sgpr = UINT32_MAX;
    uint32_t table_selector_mode = 0;
    uint32_t table_load_pc = UINT32_MAX;
    bool table_contract_valid = true;
    uint32_t flat_base_sgpr = 0;
    uint32_t bvh_box_grow = 0;
    bool bvh_sort_enabled = false;
    bool null_bvh = false;
    bool zero_record_raw = false;
    bool optional_null_raw_load = false;
    bool proven_null_guarded_raw_store = false;
    bool proven_null_nullable_raw_buffer = false;
    bool gta5_cf9200_no_backing = false;
    uint32_t selected_sbuffer_soffset = UINT32_MAX;
    std::array<uint32_t, 4> selected_sbuffer_words{};
    uint32_t indirect_buffer_contract_tag = 0;
    uint32_t indirect_buffer_binding_bytes = 0;
    uint32_t indirect_buffer_slot_count = 0;
    uint32_t indirect_buffer_header_bytes = 0;
    uint32_t indirect_buffer_slot_bytes = 0;
    uint32_t indirect_pointer_carrier_version = 0;
    uint32_t indirect_pointer_proof_schema = 0;
    uint32_t indirect_pointer_binding_bytes = 0;
    uint32_t indirect_pointer_record_count = 0;
    uint32_t indirect_pointer_segment_count = 0;
    uint32_t indirect_pointer_segment_directory_byte_offset = 0;
    uint64_t indirect_pointer_proof_fingerprint = 0;
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
    uint32_t compute_pgm_rsrc1 = kDefaultComputePgmRsrc1;
    uint32_t compute_tidig_comp_cnt = 0;
    bool compute_tgid_x_en = false, compute_tgid_y_en = false, compute_tgid_z_en = false;
    bool compute_tg_size_en = false;
    uint32_t compute_lds_bytes = 0;
    uint32_t compute_native_subgroup_size = 0;
    uint32_t compute_native_storage_format_support = 0;
    bool compute_storage_buffer_int64_atomics = false;
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
    // Diagnostic identity. All-default in production (PROSPER_CFG_TRIP_BOUND unset), which leaves
    // every key byte-identical to what it was before this field existed -- so caching behaviour is
    // provably unchanged unless the diagnostic is armed. When it IS armed, the emitted module depends
    // on the program's ADDRESS and on the selector state, neither of which any other key field can
    // see: the rest of the key is code bytes and launch shape, so a target and a non-target sharing
    // a body would otherwise share one compiled module and defeat the targeting.
    uint64_t trip_bound_program_address = 0;
    ComputeTripBoundSettings trip_bound{};
    size_t cached_hash = 0;

    bool operator==(const ShaderCompileKey& other) const {
        const bool same_code = code == other.code ||
            (code && other.code && *code == *other.code);
        const bool same_chain_code = chain_code == other.chain_code ||
            (chain_code && other.chain_code && *chain_code == *other.chain_code);
        return stage == other.stage &&
               trip_bound_program_address == other.trip_bound_program_address &&
               trip_bound.bound == other.trip_bound.bound &&
               trip_bound.only_program == other.trip_bound.only_program &&
               trip_bound.only_phase == other.trip_bound.only_phase &&
               trip_bound.only_ordinal == other.trip_bound.only_ordinal &&
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
               compute_pgm_rsrc1 == other.compute_pgm_rsrc1 &&
               compute_tidig_comp_cnt == other.compute_tidig_comp_cnt &&
               compute_tgid_x_en == other.compute_tgid_x_en &&
               compute_tgid_y_en == other.compute_tgid_y_en &&
               compute_tgid_z_en == other.compute_tgid_z_en &&
               compute_tg_size_en == other.compute_tg_size_en &&
               compute_lds_bytes == other.compute_lds_bytes &&
               compute_native_subgroup_size == other.compute_native_subgroup_size &&
               compute_native_storage_format_support ==
                   other.compute_native_storage_format_support &&
               compute_storage_buffer_int64_atomics ==
                   other.compute_storage_buffer_int64_atomics &&
               compute_packed_r11_storage == other.compute_packed_r11_storage &&
               vertex_lds_dwords == other.vertex_lds_dwords &&
               vertices_per_instance == other.vertices_per_instance &&
               resources == other.resources && same_code && same_chain_code;
    }
};

inline uint64_t hash_mix(uint64_t hash, uint64_t value) {
    // FNV-1a over fixed-width values. Equality still compares the full key, so collisions are benign.
    for (unsigned i = 0; i < 8; ++i) {
        hash ^= static_cast<uint8_t>(value >> (i * 8));
        hash *= 1099511628211ull;
    }
    return hash;
}

inline uint64_t hash_shader_code(const std::vector<uint32_t>& code) {
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t word : code) hash = hash_mix(hash, word);
    return hash;
}

struct ShaderCompileKeyHash {
    template <bool WordHash>
    static uint64_t mix(uint64_t hash, uint64_t value) {
        if constexpr (!WordHash) return hash_mix(hash, value);
        // In-process bucket selection only. Every semantic field and the full equality check
        // remain authoritative; shader-code hashes and persisted identities are unchanged.
        return hash ^ (value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
    }

    template <bool WordHash>
    static size_t compute_impl(const ShaderCompileKey& key) {
        constexpr auto hash_mix = [](uint64_t state, uint64_t value) {
            return mix<WordHash>(state, value);
        };
        uint64_t hash = 1469598103934665603ull;
        hash = hash_mix(hash, static_cast<uint32_t>(key.stage));
        hash = hash_mix(hash, key.trip_bound_program_address);
        hash = hash_mix(hash, key.trip_bound.bound);
        hash = hash_mix(hash, key.trip_bound.only_program);
        hash = hash_mix(hash, key.trip_bound.only_phase);
        hash = hash_mix(hash, key.trip_bound.only_ordinal);
        hash = hash_mix(hash, key.has_resource_table);
        hash = hash_mix(hash, key.force_position_w);
        hash = hash_mix(hash, key.capture_position);
        hash = hash_mix(hash, key.has_pixel_inputs);
        if (key.has_pixel_inputs) {
            hash = hash_mix(hash, key.pixel_inputs.valid_mask);
            hash = hash_mix(hash, key.pixel_inputs.passthrough_mask);
            hash = hash_mix(hash, key.pixel_inputs.consumed_mask);   // #2945
            hash = hash_mix(hash, key.pixel_inputs.consumed_known);
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
            hash = hash_mix(hash, key.compute_pgm_rsrc1);
            hash = hash_mix(hash, key.compute_tidig_comp_cnt);
            hash = hash_mix(hash, key.compute_tgid_x_en);
            hash = hash_mix(hash, key.compute_tgid_y_en);
            hash = hash_mix(hash, key.compute_tgid_z_en);
            hash = hash_mix(hash, key.compute_tg_size_en);
            hash = hash_mix(hash, key.compute_lds_bytes);
            hash = hash_mix(hash, key.compute_native_subgroup_size);
            hash = hash_mix(hash, key.compute_native_storage_format_support);
            hash = hash_mix(hash, key.compute_storage_buffer_int64_atomics);
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
            hash = hash_mix(hash, resource.declared_mip_levels);
            hash = hash_mix(hash, resource.materialized_mip_levels);
            hash = hash_mix(hash, resource.in_mip_tail);
            hash = hash_mix(hash, resource.compression_enabled);
            hash = hash_mix(hash, resource.proven_zero_mip);
            hash = hash_mix(hash, resource.binding);
            hash = hash_mix(hash, resource.stride);
            hash = hash_mix(hash, resource.one_record_tail_semantic);
            hash = hash_mix(hash, resource.atomic_x2_record_count);
            hash = hash_mix(hash, resource.scalar_buffer_dword_count);
            hash = hash_mix(hash, resource.scalar_buffer_contract_valid);
            hash = hash_mix(hash, resource.srt_offset);
            hash = hash_mix(hash, resource.sgpr_base);
            hash = hash_mix(hash, resource.fetch_pc);
            hash = hash_mix(hash, resource.fetch_index_mode);
            hash = hash_mix(hash, resource.table_index_count);
            hash = hash_mix(hash, resource.table_entry_stride);
            hash = hash_mix(hash, resource.table_index_sgpr);
            hash = hash_mix(hash, resource.table_selector_mode);
            hash = hash_mix(hash, resource.table_load_pc);
            hash = hash_mix(hash, resource.table_contract_valid);
            hash = hash_mix(hash, resource.flat_base_sgpr);
            hash = hash_mix(hash, resource.bvh_box_grow);
            hash = hash_mix(hash, resource.bvh_sort_enabled);
            hash = hash_mix(hash, resource.null_bvh);
            hash = hash_mix(hash, resource.zero_record_raw);
            hash = hash_mix(hash, resource.optional_null_raw_load);
            hash = hash_mix(hash, resource.proven_null_guarded_raw_store);
            hash = hash_mix(hash, resource.proven_null_nullable_raw_buffer);
            hash = hash_mix(hash, resource.gta5_cf9200_no_backing);
            hash = hash_mix(hash, resource.selected_sbuffer_soffset);
            for (const uint32_t word : resource.selected_sbuffer_words)
                hash = hash_mix(hash, word);
            hash = hash_mix(hash, resource.indirect_buffer_contract_tag);
            hash = hash_mix(hash, resource.indirect_buffer_binding_bytes);
            hash = hash_mix(hash, resource.indirect_buffer_slot_count);
            hash = hash_mix(hash, resource.indirect_buffer_header_bytes);
            hash = hash_mix(hash, resource.indirect_buffer_slot_bytes);
            hash = hash_mix(hash, resource.indirect_pointer_carrier_version);
            hash = hash_mix(hash, resource.indirect_pointer_proof_schema);
            hash = hash_mix(hash, resource.indirect_pointer_binding_bytes);
            hash = hash_mix(hash, resource.indirect_pointer_record_count);
            hash = hash_mix(hash, resource.indirect_pointer_segment_count);
            hash = hash_mix(
                hash, resource.indirect_pointer_segment_directory_byte_offset);
            hash = hash_mix(hash, resource.indirect_pointer_proof_fingerprint);
            hash = hash_mix(hash, resource.srgb);
            hash = hash_mix(hash, resource.depth_compare);
            hash = hash_mix(hash, resource.depth_compare_func);
            hash = hash_mix(hash, resource.mag_filter);
            hash = hash_mix(hash, resource.addr_u);
            hash = hash_mix(hash, resource.addr_v);
            hash = hash_mix(hash, resource.border_color_type);
            hash = hash_mix(hash, resource.normalize_unnormalized_coordinates);
        }
        if constexpr (WordHash) {
            // Avalanche once after the field walk, including upper bits in bucket selection.
            hash ^= hash >> 33;
            hash *= 0xff51afd7ed558ccdull;
            hash ^= hash >> 33;
            hash *= 0xc4ceb9fe1a85ec53ull;
            hash ^= hash >> 33;
        }
        return static_cast<size_t>(hash);
    }

    static size_t compute(const ShaderCompileKey& key) {
        // A cache must use one hash policy throughout its lifetime, even if diagnostics change.
        static const bool word_hash = std::getenv("PROSPER_NO_SHADER_KEY_WORD_HASH") == nullptr;
        return word_hash ? compute_impl<true>(key) : compute_impl<false>(key);
    }

    size_t operator()(const ShaderCompileKey& key) const { return key.cached_hash; }
};

struct CachedShader {
    SharedShaderWords spirv;
    uint64_t identity = 0;
    mutable std::atomic<uint64_t> last_use{0};
    uint64_t bytes = 0;
    bool writes_trip_witness = false;

    CachedShader() = default;
    CachedShader(const CachedShader& other)
        : spirv(other.spirv), identity(other.identity),
          last_use(other.last_use.load(std::memory_order_relaxed)),
          bytes(other.bytes), writes_trip_witness(other.writes_trip_witness) {}
    CachedShader& operator=(const CachedShader& other) {
        if (this != &other) {
            spirv = other.spirv;
            identity = other.identity;
            last_use.store(other.last_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
            bytes = other.bytes;
            writes_trip_witness = other.writes_trip_witness;
        }
        return *this;
    }
};

struct ShaderCache {
    std::shared_mutex mutex;
    std::unordered_map<ShaderCompileKey, CachedShader, ShaderCompileKeyHash> entries;
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> bypasses{0};
    std::atomic<uint64_t> compute_witness_analyses{0};
    std::atomic<uint64_t> use_counter{0};
    ShaderRecompileCacheStats stats;
    uint64_t next_identity = 1;
};

inline bool fold_control_cache_enabled() {
    static const bool enabled = std::getenv("PROSPER_NO_FOLD_CONTROL_CACHE") == nullptr;
    return enabled;
}

struct DecodedShader {
    FoldControlPlan control_plan;
    FoldControlPlan shader_constant_control_plan;
    std::vector<uint32_t> code;
    std::vector<Rdna2Inst> instructions;
    std::vector<Rdna2Inst> shader_constant_instructions;
    // Full-stream inventory: specialization may remove a spill but cannot introduce one.
    std::bitset<256> scalar_spill_written_vgprs;
    size_t source_dwords = 0;
    bool shader_constant_specialized = false;
    bool terminated = false;
    uint64_t bytes = 0;
};

struct DecodedShaderEntry {
    std::shared_ptr<const DecodedShader> shader;
    uint64_t last_use = 0;
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

struct PcrelDispatchSelection {
    PcrelDispatchInfo dispatch;
    const ShaderResource* resource = nullptr;
    uint32_t raw_selector = 0;
    uint32_t target = UINT32_MAX;
    bool readable = false;
};

// Reuse only allocation storage: all resource semantics are reconstructed on every lookup.
// Moving the slot out gives each active call its own vector, including nested construction.
// Cached keys never return storage here; their ownership remains with the cache.
class ShaderKeyResourceScratch {
public:
    explicit ShaderKeyResourceScratch(ShaderCompileKey& key) : key_(key) {}
    ShaderKeyResourceScratch(const ShaderKeyResourceScratch&) = delete;
    ShaderKeyResourceScratch& operator=(const ShaderKeyResourceScratch&) = delete;
    ~ShaderKeyResourceScratch() { recycle(key_.resources); }

    static std::vector<ShaderResourceCompileKey> acquire() {
        return enabled() ? std::move(slot()) : std::vector<ShaderResourceCompileKey>{};
    }

    void prepare_for_cache() {
        // A smaller miss must not transfer an oversized scratch allocation into the cache,
        // whose existing byte ledger charges resource elements, not spare scratch capacity.
        // Do this potentially throwing allocation before any eviction/accounting mutation.
        if (key_.resources.capacity() > key_.resources.size()) {
            std::vector<ShaderResourceCompileKey> compact(key_.resources.begin(),
                                                        key_.resources.end());
            recycle(key_.resources);
            key_.resources = std::move(compact);
        }
    }

private:
    static bool enabled() {
        static const bool value = std::getenv("PROSPER_NO_SHADER_KEY_SCRATCH") == nullptr;
        return value;
    }
    static std::vector<ShaderResourceCompileKey>& slot() {
        thread_local std::vector<ShaderResourceCompileKey> storage;
        return storage;
    }
    static void recycle(std::vector<ShaderResourceCompileKey>& resources) {
        // Limit idle storage per thread, independently of cache and active-call allocations.
        constexpr size_t max_capacity = 65536 / sizeof(ShaderResourceCompileKey);
        if (!enabled() || resources.capacity() > max_capacity) return;
        auto& storage = slot();
        if (resources.capacity() > storage.capacity()) {
            resources.clear();
            storage = std::move(resources);
        }
    }
    ShaderCompileKey& key_;
};
struct ConsumedAttributeMaskCache {
    std::mutex mutex;
    std::unordered_map<uint64_t, uint32_t> masks;
};

}  // namespace prosper::gpu
