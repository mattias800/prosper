// ngg_live_draw.cpp -- see ngg_live_draw.hpp.
#include "gpu/execute/ngg_live_draw.hpp"

#include "gpu/execute/shader_cache_internal.hpp"

#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/ngg_subgroup_abi.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace prosper::gpu {

namespace {

namespace P = prosper::agc::Pm4;

constexpr size_t kStageEntries = 64;
constexpr size_t kDrawEntries = 128;
constexpr uint32_t kMaxWaves = 4;

uint64_t mix(uint64_t hash, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i)
        hash = (hash ^ ((value >> (8 * i)) & 0xffu)) * 1099511628211ull;
    return hash;
}

uint64_t interpolation_hash(const FragmentInterpolationLayout& layout) {
    uint64_t hash = 1469598103934665603ull;
    for (const auto& locations : layout.parameter_locations)
        for (uint32_t location : locations) hash = mix(hash, location);
    for (uint32_t location : layout.system_locations) hash = mix(hash, location);
    hash = mix(hash, layout.attribute_mask);
    hash = mix(hash, layout.smooth_mask);
    hash = mix(hash, layout.passthrough_mask);
    hash = mix(hash, layout.flat_mask);
    hash = mix(hash, (layout.requires_geometry ? 1u : 0u) | (layout.valid ? 2u : 0u));
    return hash;
}

// The resource table's half of the stage key. The per-resource part is built by the SAME code as
// the ordinary shader cache's key (append_shader_resource_compile_keys), with the shell's stage
// (compute) and its program (the linked chain), so the two partitions cannot drift: every
// data-dependent admission the emitter reads -- snapshot sizes and validity, scalar-buffer and
// table contracts, null markers -- is in it, and no address or content byte is.
struct ResourceKey {
    bool present = false;
    uint32_t vertices_per_instance = 0;
    std::vector<std::pair<uint32_t, uint32_t>> owned_raw, owned_nested;
    std::vector<ShaderResourceCompileKey> resources;
    bool operator==(const ResourceKey&) const = default;
};

ResourceKey resource_key(const ShaderResourceTable* table,
                         const std::shared_ptr<const std::vector<uint32_t>>& program) {
    ResourceKey key;
    if (!table) return key;
    key.present = true;
    key.vertices_per_instance = table->vertices_per_instance;
    key.owned_raw = table->owned_raw_snapshot_requirements;
    key.owned_nested = table->owned_nested_snapshot_requirements;
    append_shader_resource_compile_keys(ShaderProgramStage::Compute, *table, program,
                                        key.resources);
    return key;
}

std::vector<uint32_t> pixel_input_shape(const PixelInputMapping* mapping) {
    std::vector<uint32_t> out;
    if (!mapping) return out;
    out.assign(mapping->controls.begin(), mapping->controls.end());
    out.insert(out.end(), {mapping->valid_mask, mapping->passthrough_mask, mapping->consumed_mask,
                           mapping->consumed_known ? 1u : 0u});
    return out;
}

// Everything a compiled stage depends on besides W (see the header's CACHING note).
struct StageKey {
    std::vector<uint32_t> program;   // the linked chain, compared exactly
    ResourceKey resources;
    std::vector<uint32_t> pixel_inputs;   // pixel_input_shape()
    uint32_t user_sgprs = 0, lds_granules = 0, layer_slices = 0, depth_slice_fanout = 0;
    uint8_t topology = 0, route = 0, float_transport = 0;
    bool native_wave64 = false, provoking_vertex_last = false, layer_from_pos1 = false;
    bool count_violations = false, interpolation = false, user_data_address = false;
    uint64_t interpolation_layout = 0;
    bool operator==(const StageKey&) const = default;
};

struct StageKeyHash {
    size_t operator()(const StageKey& key) const {
        uint64_t hash = 1469598103934665603ull;
        for (uint32_t word : key.program) hash = mix(hash, word);
        for (const ShaderResourceCompileKey& r : key.resources.resources)
            hash = mix(hash, (uint64_t{r.binding} << 32) ^ r.fetch_pc ^ (uint64_t{r.cls} << 48));
        for (uint32_t word : key.pixel_inputs) hash = mix(hash, word);
        hash = mix(hash, (uint64_t{key.user_sgprs} << 32) | key.layer_slices);
        hash = mix(hash, key.lds_granules ^ (uint64_t{key.depth_slice_fanout} << 32));
        hash = mix(hash, key.interpolation_layout);
        hash = mix(
            hash,
            key.topology | (key.route << 8) | (key.float_transport << 16) |
                (uint64_t{key.native_wave64} << 24) | (uint64_t{key.provoking_vertex_last} << 25) |
                (uint64_t{key.layer_from_pos1} << 26) | (uint64_t{key.count_violations} << 27) |
                (uint64_t{key.interpolation} << 28) | (uint64_t{key.user_data_address} << 29));
        return static_cast<size_t>(hash);
    }
};

struct StageEntry {
    std::mutex compile;   // held while one W compiles; other entries proceed in parallel
    std::array<std::shared_ptr<const NggSubgroupStages>, kMaxWaves + 1> stages{};
    std::array<std::string, kMaxWaves + 1> refusal{};   // cached refusal per W
    std::array<bool, kMaxWaves + 1> attempted{};
    uint64_t last_use = 0;
};

struct DrawKey {
    const StageEntry* stages = nullptr;   // pinned by the stage cache entry, see `stage_owner`
    uint32_t vertices = 0, instances = 0;
    uint8_t topology = 0;
    std::array<uint32_t, 7> limits{};
    std::vector<uint32_t> push_constants;
    // An indexed draw's index VALUES (#3135 P6): the plan, and so every launch record, depends on
    // them. Keyed by a hash computed once per draw, with the decoded vector itself held by
    // reference (never copied into the key); two keys whose hashes match are compared exactly, so a
    // collision is never a hit. An indexed draw never keys equal to a non-indexed one (`indexed`).
    bool indexed = false;
    uint64_t index_hash = 0;
    std::shared_ptr<const std::vector<uint32_t>> indices;
    auto tie() const {
        return std::tie(stages, vertices, instances, topology, limits, push_constants, indexed,
                        index_hash);
    }
    bool operator<(const DrawKey& other) const {
        const auto a = tie(), b = other.tie();
        if (a < b) return true;
        if (b < a || indices == other.indices) return false;
        if (!indices || !other.indices) return !indices;
        return *indices < *other.indices;
    }
};

struct DrawEntry {
    std::shared_ptr<StageEntry> stage_owner;
    std::shared_ptr<const NggSubgroupDraw> draw;
    uint64_t last_use = 0;
};

struct Cache {
    std::mutex mutex;
    uint64_t clock = 0;
    std::unordered_map<StageKey, std::shared_ptr<StageEntry>, StageKeyHash> stages;
    std::map<DrawKey, DrawEntry> draws;
    NggLiveDrawCacheStats stats;
};

Cache& cache() {
    static Cache instance;
    return instance;
}

uint64_t last_use_of(const std::shared_ptr<StageEntry>& entry) {
    return entry->last_use;
}
uint64_t last_use_of(const DrawEntry& entry) {
    return entry.last_use;
}

template <typename Map>
void evict(Map& map, size_t bound, uint64_t* evictions) {
    while (map.size() > bound) {
        auto oldest = map.begin();
        for (auto it = std::next(map.begin()); it != map.end(); ++it)
            if (last_use_of(it->second) < last_use_of(oldest->second)) oldest = it;
        map.erase(oldest);
        if (evictions) ++*evictions;
    }
}

// The rule a compiler refusal ("reason=<name> ...") names, as a string that lives for the process.
const char* intern_reason(const std::string& text) {
    static std::mutex mutex;
    static std::set<std::string> names;
    std::string name = "ngg-compile-refused";
    const size_t at = text.find("reason=");
    if (at != std::string::npos) {
        const size_t begin = at + 7;
        const size_t end = text.find(' ', begin);
        name = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (name.empty()) name = "ngg-compile-refused";
    }
    const std::lock_guard lock(mutex);
    return names.insert(std::move(name)).first->c_str();
}

uint32_t reg(const RegisterFile& file, uint32_t offset, const char* name, const char** missing) {
    const auto it = file.find(offset);
    if (it == file.end()) {
        if (missing && !*missing) *missing = name;
        return 0;
    }
    return it->second;
}

}   // namespace

NggDrawRegisters read_ngg_draw_registers(const GpuState& state, uint32_t primitive_type) {
    NggDrawRegisters r;
    const char** m = &r.missing;
    r.vgt_shader_stages_en = reg(state.cx, P::VGT_SHADER_STAGES_EN, "VGT_SHADER_STAGES_EN", m);
    r.vgt_gs_onchip_cntl = reg(state.cx, P::VGT_GS_ONCHIP_CNTL, "VGT_GS_ONCHIP_CNTL", m);
    r.ge_cntl = reg(state.uc, P::GE_CNTL, "GE_CNTL", m);
    r.ge_max_output_per_subgroup =
        reg(state.cx, P::GE_MAX_OUTPUT_PER_SUBGROUP, "GE_MAX_OUTPUT_PER_SUBGROUP", m);
    r.vgt_gs_max_vert_out = reg(state.cx, P::VGT_GS_MAX_VERT_OUT, "VGT_GS_MAX_VERT_OUT", m);
    r.vgt_esgs_ring_itemsize =
        reg(state.cx, P::VGT_ESGS_RING_ITEMSIZE, "VGT_ESGS_RING_ITEMSIZE", m);
    r.spi_shader_pgm_rsrc2_gs =
        reg(state.sh, P::SPI_SHADER_PGM_RSRC2_GS, "SPI_SHADER_PGM_RSRC2_GS", m);
    r.vgt_gs_out_prim_type = reg(state.cx, P::VGT_GS_OUT_PRIM_TYPE, "VGT_GS_OUT_PRIM_TYPE", m);
    // Reset value 0 when absent.
    r.vgt_gs_instance_cnt = reg(state.cx, P::VGT_GS_INSTANCE_CNT, nullptr, nullptr);
    r.pa_su_sc_mode_cntl = reg(state.cx, P::PA_SU_SC_MODE_CNTL, nullptr, nullptr);
    r.pa_cl_vs_out_cntl = reg(state.cx, P::PA_CL_VS_OUT_CNTL, nullptr, nullptr);
    r.pa_cl_clip_cntl = reg(state.cx, P::PA_CL_CLIP_CNTL, nullptr, nullptr);
    r.spi_ps_input_ena = reg(state.cx, P::SPI_PS_INPUT_ENA, nullptr, nullptr);
    r.primitive_type = primitive_type;
    return r;
}

std::shared_ptr<const std::vector<uint32_t>> ngg_linked_chain(const uint32_t* prolog,
                                                              size_t prefix_dwords,
                                                              const uint32_t* main,
                                                              size_t main_dwords) {
    // No prolog: an NGG VS that is its own primitive shader runs as one program (#3135 P7).
    if (!prolog != !prefix_dwords || !main || !main_dwords) return nullptr;
    const size_t main_span = rdna2_recompile_code_span(main, main_dwords);
    if (!main_span) return nullptr;
    std::vector<uint32_t> words;
    if (prolog) words.assign(prolog, prolog + prefix_dwords);
    words.insert(words.end(), main, main + main_span);
    static std::mutex mutex;
    static std::map<std::vector<uint32_t>,
                    std::pair<std::shared_ptr<const std::vector<uint32_t>>, uint64_t>>
        chains;
    static uint64_t clock = 0;
    const std::lock_guard lock(mutex);
    auto& slot = chains[words];
    if (!slot.first) slot.first = std::make_shared<const std::vector<uint32_t>>(words);
    slot.second = ++clock;
    while (chains.size() > 16u) {
        auto oldest = chains.begin();
        for (auto it = std::next(chains.begin()); it != chains.end(); ++it)
            if (it->second.second < oldest->second.second) oldest = it;
        chains.erase(oldest);
    }
    return chains[words].first;
}

bool read_ngg_user_data(const GpuState& state, uint32_t count, std::vector<uint32_t>* words) {
    words->assign(count, 0u);
    for (uint32_t k = 0; k < count; ++k) {
        const auto it = state.sh.find(P::SPI_SHADER_USER_DATA_GS_0 + k);
        if (it == state.sh.end()) return false;
        (*words)[k] = it->second;
    }
    return true;
}

bool read_ngg_user_data_address(const GpuState& state, uint32_t words[2]) {
    const auto lo = state.sh.find(P::SPI_SHADER_USER_DATA_ADDR_LO_GS);
    const auto hi = state.sh.find(P::SPI_SHADER_USER_DATA_ADDR_HI_GS);
    if (lo == state.sh.end() || hi == state.sh.end() || (!lo->second && !hi->second)) return false;
    words[0] = lo->second;
    words[1] = hi->second;
    return true;
}

bool ngg_program_reads_user_data_address(const std::shared_ptr<const std::vector<uint32_t>>& linked,
                                         uint32_t user_sgprs) {
    if (!linked || linked->empty()) return false;
    // Keyed by the shared program (ngg_linked_chain hands out one copy per content), which the
    // entry also owns, so an address is never reused for other words while it is cached.
    static std::mutex mutex;
    static std::map<std::pair<std::shared_ptr<const std::vector<uint32_t>>, uint32_t>, bool> cache;
    {
        const std::lock_guard lock(mutex);
        if (const auto found = cache.find({linked, user_sgprs}); found != cache.end())
            return found->second;
    }
    std::vector<Rdna2Inst> ins;
    rdna2_walk(linked->data(), linked->size(), ins);
    NggSubgroupAbiLaunch launch;
    launch.user_sgprs = user_sgprs;
    const bool reads = analyze_ngg_subgroup_abi(ins, launch).reason == "ngg-abi-read-s0-s1";
    const std::lock_guard lock(mutex);
    if (cache.size() >= 64u) cache.clear();
    cache[{linked, user_sgprs}] = reads;
    return reads;
}

NggLiveDrawResult realize_ngg_live_draw(const NggLiveDrawInput& input,
                                        const NggHostCapabilities& host) {
    NggLiveDrawResult result;
    const NggDrawAdmission admission = admit_ngg_draw(input.registers, input.facts, host);
    result.applies = admission.applies;
    result.strip = admission.shape.topology == NggInputTopology::TriangleStrip;
    result.indexed = admission.shape.indices != nullptr;
    if (!admission.applies) return result;
    const auto refuse = [&](const char* reason, std::string detail = {}) {
        result.refusal = reason;
        result.detail = std::move(detail);
        result.draw.reset();
        return result;
    };
    if (!admission.ok()) return refuse(admission.refusal);
    if (!input.user_data_complete || input.user_data.size() != admission.user_sgprs)
        return refuse("ngg-user-data-unavailable");
    if (!input.linked || input.linked->empty()) return refuse("ngg-program-unavailable");
    // s0:s1 costs two push words, so it is supplied only to a program that reads it (#4735
    // review): a program with 31-32 user SGPRs that never touches s0:s1 keeps its admission.
    const bool supply_address =
        input.user_data_address_known &&
        ngg_program_reads_user_data_address(input.linked, admission.user_sgprs);
    if (supply_address && admission.user_sgprs + 2u > kNggShellMaxPushWords)
        return refuse("ngg-user-sgpr-count");
    const bool interpolation = input.interpolation.requires_geometry;
    if (interpolation && !input.interpolation.valid) return refuse("ngg-interpolation-invalid");

    StageKey key;
    key.program = *input.linked;
    key.resources = resource_key(input.resources, input.linked);
    key.pixel_inputs = pixel_input_shape(input.pixel_inputs);
    key.user_sgprs = admission.user_sgprs;
    key.lds_granules = admission.lds_granules;
    key.layer_slices = admission.layer_slices;
    key.depth_slice_fanout = admission.depth_slice_fanout;
    key.topology = static_cast<uint8_t>(admission.topology);
    key.route = static_cast<uint8_t>(admission.route);
    key.float_transport = static_cast<uint8_t>(input.float_transport.profile);
    key.native_wave64 = admission.native_wave64;
    key.provoking_vertex_last = admission.provoking_vertex_last;
    key.layer_from_pos1 = admission.layer_from_pos1;
    key.count_violations = admission.count_violations;
    key.interpolation = interpolation;
    key.interpolation_layout = interpolation ? interpolation_hash(input.interpolation) : 0u;
    key.user_data_address = supply_address;

    NggSubgroupDrawRequest request;
    request.resources = input.resources;
    request.shell.rsrc2_gs_lds_size = admission.lds_granules;
    request.shell.user_sgprs = admission.user_sgprs;
    request.shell.native_wave64 = admission.native_wave64;
    request.shell.user_data_address_known = supply_address;
    request.limits = admission.limits;
    request.shape = admission.shape;
    request.raster.topology = admission.topology;
    request.raster.provoking_vertex_last = admission.provoking_vertex_last;
    request.raster.layer_from_pos1 = admission.layer_from_pos1;
    request.raster.layer_slices = admission.layer_slices;
    request.raster.route = admission.route;
    request.raster.pixel_inputs = input.pixel_inputs;
    request.raster.float_transport = input.float_transport;
    request.raster.count_violations = admission.count_violations;
    // A depth array is replayed per slice; the base draw is the slice-0 replay, so a consumer that
    // knows nothing of the replay draws slice 0's primitives only, never the others into it.
    if (admission.depth_slice_fanout) request.raster.layer_select = true;
    if (admission.route == NggLayerRoute::InterpolationGeometry)
        request.raster.reserved_locations = input.interpolation.attribute_mask;
    if (interpolation) {
        const FragmentInterpolationLayout layout = input.interpolation;
        const FloatTransportConfig transport = input.float_transport;
        request.interpolation_geometry = [layout, transport](const NggRasterCommitInterface& i) {
            return recompile_interpolation_geometry(layout, false, false, transport, false,
                                                    i.layer_location);
        };
    }
    request.push_constants = input.user_data;
    if (supply_address)
        request.push_constants.insert(request.push_constants.end(), input.user_data_address,
                                      input.user_data_address + 2);
    request.diagnostic = {RecompileDiagnosticStage::Vertex, input.program_address};

    request.linked_code = key.program.data();
    request.dwords = key.program.size();
    Cache& c = cache();
    std::shared_ptr<StageEntry> entry;
    {
        const std::lock_guard lock(c.mutex);
        auto& slot = c.stages[key];
        if (!slot) slot = std::make_shared<StageEntry>();
        slot->last_use = ++c.clock;
        entry = slot;
        evict(c.stages, kStageEntries, &c.stats.stage_evictions);
    }

    // The draw cache: a repeated shape with the same push-constant words reuses its description.
    DrawKey draw_key;
    draw_key.stages = entry.get();
    draw_key.vertices = admission.shape.vertex_count;
    draw_key.instances = admission.shape.instance_count;
    draw_key.topology = static_cast<uint8_t>(admission.shape.topology);
    draw_key.limits = {admission.limits.es_verts_per_subgroup,
                       admission.limits.gs_prims_per_subgroup,
                       admission.limits.prim_group_size,
                       admission.limits.vert_group_size,
                       admission.limits.max_out_verts_per_subgroup,
                       admission.limits.gs_max_vert_out,
                       admission.limits.esgs_item_size};
    draw_key.push_constants = request.push_constants;
    draw_key.indexed = admission.shape.indices != nullptr;
    if (admission.shape.indices) {
        draw_key.index_hash = 1469598103934665603ull;
        for (uint32_t index : *admission.shape.indices)
            draw_key.index_hash = (draw_key.index_hash ^ index) * 1099511628211ull;
        draw_key.indices = admission.shape.indices;
    }
    {
        const std::lock_guard lock(c.mutex);
        const auto found = c.draws.find(draw_key);
        if (found != c.draws.end() && found->second.stage_owner == entry) {
            found->second.last_use = ++c.clock;
            ++c.stats.draw_hits;
            c.stats.strip_draws += result.strip ? 1u : 0u;
            c.stats.indexed_draws += result.indexed ? 1u : 0u;
            result.draw = found->second.draw;
            result.depth_slice_count = admission.depth_slice_fanout;
            result.depth_first_slice = admission.depth_first_slice;
            return result;
        }
    }

    const auto stages_for = [&](uint32_t waves,
                                std::string* why) -> std::shared_ptr<const NggSubgroupStages> {
        if (!waves || waves > kMaxWaves) {
            if (why) *why = "reason=ngg-draw-stages-unavailable";
            return nullptr;
        }
        const std::lock_guard compile_lock(entry->compile);
        if (!entry->attempted[waves]) {
            entry->stages[waves] =
                compile_ngg_subgroup_stages(request, waves, &entry->refusal[waves]);
            entry->attempted[waves] = true;
            const std::lock_guard lock(c.mutex);
            ++c.stats.stage_compiles;
        } else {
            const std::lock_guard lock(c.mutex);
            ++c.stats.stage_hits;
        }
        if (!entry->stages[waves] && why) *why = entry->refusal[waves];
        return entry->stages[waves];
    };
    std::string why;
    auto draw = assemble_ngg_subgroup_draw(request, stages_for, &why);
    if (!draw) return refuse(intern_reason(why), why);
    if (const char* device = ngg_device_refusal(*draw, host)) return refuse(device);
    {
        const std::lock_guard lock(c.mutex);
        ++c.stats.draw_assemblies;
        DrawEntry stored;
        stored.stage_owner = entry;
        stored.draw = draw;
        stored.last_use = ++c.clock;
        c.draws[std::move(draw_key)] = std::move(stored);
        evict(c.draws, kDrawEntries, nullptr);
    }
    if (result.strip || result.indexed) {
        const std::lock_guard lock(c.mutex);
        c.stats.strip_draws += result.strip ? 1u : 0u;
        c.stats.indexed_draws += result.indexed ? 1u : 0u;
    }
    result.draw = std::move(draw);
    // The per-slice replay shares this one description: each slice's item selects its layer at
    // draw time (DrawItem::ngg_layer_select), so nothing here is copied per slice.
    result.depth_slice_count = admission.depth_slice_fanout;
    result.depth_first_slice = admission.depth_first_slice;
    return result;
}

NggLiveDrawCacheStats ngg_live_draw_cache_stats() {
    Cache& c = cache();
    const std::lock_guard lock(c.mutex);
    return c.stats;
}

void reset_ngg_live_draw_cache_for_test() {
    Cache& c = cache();
    const std::lock_guard lock(c.mutex);
    c.draws.clear();
    c.stages.clear();
    c.stats = {};
}

}   // namespace prosper::gpu
