#include "gpu/execute/ordered_graphics_read_point_internal.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/original_graphics_stage_effects.hpp"
#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/execute/registered_graphics_source_internal.hpp"
#include "gpu/state/fragment_entry_observation.hpp"
#include "host/memory/guest_memory_topology.hpp"

#include <algorithm>
#include <atomic>
#include <tuple>

namespace prosper::gpu {
namespace {
bool coupled_read_only(const GraphicsReadSource& source, ShaderProgramStage stage) {
    const auto& effects =
        stage == ShaderProgramStage::Fragment ? source.fragment_effects : source.vertex_effects;
    if (!source.words || !source.decoded || !effects || !source.native_analysis ||
        !source.native_analysis->belongs_to(source.decoded) ||
        source.words.get() != &source.decoded->code || source.words.owner_before(source.decoded) ||
        source.decoded.owner_before(source.words) || source.words.owner_before(effects) ||
        effects.owner_before(source.words))
        return false;
    const uint32_t index = stage == ShaderProgramStage::Fragment ? 1u : 0u;
    return effects.get() == &source.decoded->original_effects[index] &&
           effects->source_words == source.words.get() && effects->stage == stage &&
           effects->known_read_only();
}
bool ordinary_draw(const GpuState& state, const RenderState& render) {
    if (!render.es_addr || !render.ps_addr || render.gs_addr || render.hs_addr) return false;
    for (const auto [address, type] :
         {std::pair{render.es_addr, 2u}, std::pair{render.ps_addr, 1u}}) {
        const auto* header =
            static_cast<const AgcShaderHeader*>(prosper_agc_shader_header_for_code(address));
        if (!header || header->type != type || prosper_agc_shader_continuation_for_code(address) ||
            prosper_agc_fused_back_header_for_front(address))
            return false;
    }
    // Additional fixed-function stage/streamout enable controls need explicit authority before
    // this capture can issue. Source addresses alone do not prove they are disabled.
    (void)state;
    return false; // closed until the actual retained enable-state domain is implemented
}
bool capture_outputs(const GpuState& state, const RenderState& render,
                     const prosper::GuestMappingLease& lease,
                     std::vector<prosper::GuestDirectAllocation>& outputs) {
    namespace P = prosper::agc::Pm4;
    const auto reg = [&](uint32_t offset) {
        const auto found = state.cx.find(offset);
        return found == state.cx.end() ? 0u : found->second;
    };
    for (uint32_t slot = 0; slot < render.color_targets.size(); ++slot) {
        const auto& target = render.color_targets[slot];
        for (const auto [low, high] :
             {std::pair{P::CB_COLOR0_CMASK + slot * 0xfu, P::CB_COLOR0_CMASK_BASE_EXT + slot},
              std::pair{P::CB_COLOR0_FMASK + slot * 0xfu, P::CB_COLOR0_FMASK_BASE_EXT + slot},
              std::pair{P::CB_COLOR0_DCC_BASE + slot * 0xfu, P::CB_COLOR0_DCC_BASE_EXT + slot}})
            if ((uint64_t(reg(low)) << 8u) | (uint64_t(reg(high) & 0xffu) << 40u))
                return false; // full independent metadata write layouts are not proved here
        if (!target.base) continue;
        const auto extent = color_target_physical_bytes(target);
        if (!extent) return false;
        const auto origin = prosper::guest_memory_direct_allocation(lease, target.base, extent);
        if (!origin.identity || origin.physical_begin >= origin.physical_end) return false;
        outputs.push_back(origin); // FULL original allocation, not only addressed pixels
    }
    for (const uint64_t address :
         {render.depth_read_base, render.depth_write_base, render.stencil_read_base,
          render.stencil_write_base, render.htile_data_base})
        if (address) return false; // native DS/HTILE physical write footprints remain unproved
    return true;
}
std::shared_ptr<const std::vector<prosper::GuestDirectAllocation>>
merge_outputs(const std::shared_ptr<const std::vector<prosper::GuestDirectAllocation>>& previous,
              const std::vector<prosper::GuestDirectAllocation>& additions) {
    std::shared_ptr<std::vector<prosper::GuestDirectAllocation>> changed;
    for (const auto& addition : additions) {
        const auto& current = changed ? *changed : *previous;
        if (std::any_of(current.begin(), current.end(), [&](const auto& prior) {
                return prior.identity == addition.identity &&
                       prior.physical_begin == addition.physical_begin &&
                       prior.physical_end == addition.physical_end;
            }))
            continue;
        if (!changed)
            changed = std::make_shared<std::vector<prosper::GuestDirectAllocation>>(*previous);
        changed->push_back(addition);
    }
    // Common repeated targets share the same immutable origin set; copy only on a new origin.
    return changed ? changed : previous;
}
}   // namespace

void OrderedGraphicsReadPointIssuer::record_queued_draw(const GpuState& state,
                                                        const DrawItem& draw) {
    ++scalar_covered_draws_;
    if (!scalar_effects_known_) return;
    const auto render = extract_render_state(state);
    if (!ordinary_draw(state, render) || draw.vs_guest_addr != render.es_addr ||
        draw.fs_guest_addr != render.ps_addr || draw.vs_chain_guest_addr || !draw.gs.empty() ||
        !coupled_read_only(draw.original_vs_source, ShaderProgramStage::Vertex) ||
        !coupled_read_only(draw.original_ps_source, ShaderProgramStage::Fragment)) {
        scalar_effects_known_ = false;
        return;
    }
    prosper::GuestMappingLease lease;
    std::vector<prosper::GuestDirectAllocation> outputs;
    if (!capture_outputs(state, render, lease, outputs)) {
        scalar_effects_known_ = false;
        return;
    }
    scalar_outputs_ = merge_outputs(scalar_outputs_, outputs);
    auto retained = std::make_shared<OrderedScalarBankReadPoint::PendingDraw>();
    retained->previous = scalar_effects_;
    retained->command_order = draw.command_order;
    retained->draw_index = draw.draw_index;
    for (const auto [source, address, fragment] :
         {std::tuple{&draw.original_vs_source, render.es_addr, false},
          std::tuple{&draw.original_ps_source, render.ps_addr, true}})
        retained->sources.push_back({address, source->words,
                                     fragment ? source->fragment_effects : source->vertex_effects,
                                     source->packet_requirements, source->decoded,
                                     source->native_analysis, source->registered_header, fragment});
    scalar_effects_ = std::move(retained);
}

std::shared_ptr<const OrderedScalarBankReadPoint>
OrderedGraphicsReadPointIssuer::issue_scalar(uint64_t submit, uint64_t order, const GpuState& state,
                                             const std::vector<DrawItem>& pending,
                                             std::string& refusal) const {
    refusal.clear();
    const auto reject =
        [&](const char* reason) -> std::shared_ptr<const OrderedScalarBankReadPoint> {
        refusal = reason;
        return {};
    };
    if (!submit || !dependencies_ok || !baseline.known)
        return reject("scalar-bank-ordered-dependency-unknown");
    if (!scalar_effects_known_ || scalar_covered_draws_ != pending.size() ||
        (!pending.empty() &&
         (!scalar_effects_ || scalar_effects_->command_order != pending.back().command_order ||
          scalar_effects_->draw_index != pending.back().draw_index)))
        return reject("scalar-bank-prior-queued-effects-unproved");
    const auto version = execution_.current_version();
    const auto before = graphics_producer_status();
    if (!version || !before.known || before.pending || before.failures != baseline.failures)
        return reject("scalar-bank-published-producer-not-current");
    const auto render = extract_render_state(state);
    if (!ordinary_draw(state, render))
        return reject("scalar-bank-current-stage-or-enable-domain-unproved");
    auto vertex = registered_graphics_read_source(render.es_addr);
    auto fragment = registered_graphics_read_source(render.ps_addr);
    if (!coupled_read_only(vertex, ShaderProgramStage::Vertex) ||
        !coupled_read_only(fragment, ShaderProgramStage::Fragment))
        return reject("scalar-bank-complete-original-effects-unproved");
    prosper::GuestMappingLease lease;
    std::vector<prosper::GuestDirectAllocation> current_outputs;
    if (!capture_outputs(state, render, lease, current_outputs))
        return reject("scalar-bank-current-attachment-footprints-unproved");
    const auto outputs = merge_outputs(scalar_outputs_, current_outputs);
    FragmentEntryFacts entry;
    observe_fragment_entry(state, true, entry);
    if (!entry.observed || !entry.canonical() || !entry.rsrc2_available)
        return reject("scalar-bank-current-entry-unavailable");
    const auto after = graphics_producer_status();
    if (!after.known || after.pending || after.failures != baseline.failures ||
        execution_.current_version() != version)
        return reject("scalar-bank-producer-changed-during-issue");
    std::vector<OrderedScalarBankReadPoint::Source> sources;
    for (const auto [source, address, pixel] :
         {std::tuple{&vertex, render.es_addr, false}, std::tuple{&fragment, render.ps_addr, true}})
        sources.push_back({address, source->words,
                           pixel ? source->fragment_effects : source->vertex_effects,
                           source->packet_requirements, source->decoded,
                           source->native_analysis, source->registered_header, pixel});
    static std::atomic<uint64_t> next_identity{1};
    const uint64_t identity = next_identity.fetch_add(1, std::memory_order_relaxed);
    if (!identity) return reject("scalar-bank-identity-exhausted");
    return std::shared_ptr<const OrderedScalarBankReadPoint>(new OrderedScalarBankReadPoint(
        identity, submit, order, baseline.failures, state, std::move(entry), std::move(sources),
        outputs, scalar_effects_, epoch_, epoch_->sequence.load(std::memory_order_acquire),
        version));
}
}   // namespace prosper::gpu
