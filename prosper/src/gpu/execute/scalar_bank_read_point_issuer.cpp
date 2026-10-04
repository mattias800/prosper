#include "gpu/execute/ordered_graphics_read_point_internal.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/original_graphics_stage_effects.hpp"
#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/execute/registered_graphics_source_internal.hpp"
#include "gpu/execute/native_graphics_source_lineage.hpp"
#include "gpu/execute/fragment_scalar_bank.hpp"
#include "gpu/recompiler/original_fragment_producer.hpp"
#include "gpu/recompiler/original_graphics_draw_effects.hpp"
#include "gpu/state/fragment_entry_observation.hpp"
#include "host/memory/guest_memory_topology.hpp"

#include <algorithm>
#include <atomic>
#include <tuple>

namespace prosper::gpu {
namespace {
const char* coupled_read_only_gap(const GraphicsReadSource& source, uint64_t address,
                                  ShaderProgramStage stage) {
    const auto& effects =
        stage == ShaderProgramStage::Fragment ? source.fragment_effects : source.vertex_effects;
    const auto* header = source.header_snapshot.get();
    const uint32_t type = stage == ShaderProgramStage::Fragment ? 1u : 2u;
    if (!source.words || !source.decoded) return "original-source-unavailable";
    if (!source.registered_header || !header) return "registered-header-unavailable";
    if (header->type != type) return "registered-stage-mismatch";
    if (reinterpret_cast<uint64_t>(header->code) != address)
        return "registered-code-address-mismatch";
    if (!header->shader_size || (header->shader_size & 3u) ||
        source.decoded->source_dwords != header->shader_size / sizeof(uint32_t) ||
        source.words->size() != source.decoded->source_dwords)
        return "registered-complete-code-extent-unproved";
    if (prosper_agc_shader_continuation_for_code(address) ||
        prosper_agc_fused_back_header_for_front(address))
        return "registered-continuation-or-fused-stage-unproved";
    if (!source.native_analysis || !source.native_analysis->belongs_to(source.decoded))
        return "original-native-analysis-association-unproved";
    if (!effects || source.words.get() != &source.decoded->code ||
        source.words.owner_before(source.decoded) || source.decoded.owner_before(source.words) ||
        source.words.owner_before(effects) || effects.owner_before(source.words))
        return "original-effects-owner-association-unproved";
    const uint32_t index = stage == ShaderProgramStage::Fragment ? 1u : 0u;
    if (effects.get() != &source.decoded->original_effects[index] ||
        effects->source_words != source.words.get() || effects->stage != stage)
        return "original-effects-stage-association-unproved";
    if (!effects->known_read_only())
        return effects->rejection.empty() ? "original-effects-architectural-end-unproved"
                                          : effects->rejection.c_str();
    return nullptr;
}
bool coupled_read_only(const GraphicsReadSource& source, uint64_t address,
                       ShaderProgramStage stage) {
    return !coupled_read_only_gap(source, address, stage);
}
bool ordinary_draw(const GpuState& state, const RenderState& render) {
    if (!render.es_addr || !render.ps_addr || render.gs_addr || render.hs_addr) return false;
    namespace P = prosper::agc::Pm4;
    // AMD gc_10_3_0 context fields: all-zero STAGES_EN proves ordinary VS_REAL, with
    // LS/HS/ES/GS, dynamic/dispatched draw, NGG and wave-ID modes off. Other known settings
    // remain a named initial-domain refusal, not an assertion they write memory. Presence is
    // required: a missing register is not an observed zero or a hardware reset observation.
    for (uint32_t offset : {P::VGT_SHADER_STAGES_EN, P::VGT_STRMOUT_CONFIG,
                            P::VGT_STRMOUT_BUFFER_CONFIG, P::DB_DEPTH_CONTROL, P::DB_RENDER_CONTROL,
                            P::DB_RENDER_OVERRIDE, P::DB_RENDER_OVERRIDE2}) {
        const auto observed = state.cx.find(offset);
        if (observed == state.cx.end() || observed->second) return false;
    }
    const auto color_control = state.cx.find(P::CB_COLOR_CONTROL);
    return color_control != state.cx.end() && PM4_FIELD(color_control->second, CB_COLOR_CONTROL,
                                                        MODE) == P::CB_COLOR_CONTROL_MODE_NORMAL;
}
bool capture_outputs(const GpuState& state, const RenderState& render,
                     const prosper::GuestMappingLease& lease,
                     std::vector<prosper::GuestDirectAllocation>& outputs) {
    namespace P = prosper::agc::Pm4;
    if (!state.cx.count(P::CB_TARGET_MASK) || !state.cx.count(P::CB_SHADER_MASK)) return false;
    for (uint32_t slot = 0; slot < render.color_targets.size(); ++slot) {
        const auto& target = render.color_targets[slot];
        const bool enabled =
            ((render.cb_target_mask & render.cb_shader_mask) >> (slot * 4u)) & 0xfu;
        if (!target.base) {
            if (enabled) return false;
            continue;
        }
        for (const auto [low, high] :
             {std::pair{P::CB_COLOR0_CMASK + slot * 0xfu, P::CB_COLOR0_CMASK_BASE_EXT + slot},
              std::pair{P::CB_COLOR0_FMASK + slot * 0xfu, P::CB_COLOR0_FMASK_BASE_EXT + slot},
              std::pair{P::CB_COLOR0_DCC_BASE + slot * 0xfu, P::CB_COLOR0_DCC_BASE_EXT + slot}}) {
            const auto lo = state.cx.find(low), hi = state.cx.find(high);
            if (lo == state.cx.end() || hi == state.cx.end()) return false;
            if ((uint64_t(lo->second) << 8u) | (uint64_t(hi->second & 0xffu) << 40u))
                return false; // full independent metadata write layouts are not proved here
        }
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

bool draw_requires_original_scalar_bank(const GpuState& state) {
    const auto render = extract_render_state(state);
    // Scheduling hint ONLY: do not inspect shader bytes before their ordered operation or add
    // another warm cache validation. Missing physical Wave32 metadata conservatively routes
    // ordered. The actual issuer/manifest/seal remains the sole bank admission observation.
    return render.ps_addr &&
           (!render.ps_raster_launch.ps_in_control_available || !render.ps_wave32);
}

void OrderedGraphicsReadPointIssuer::record_queued_draw(const GpuState& state, const DrawItem& draw,
                                                        uint64_t submit) {
    ++scalar_covered_draws_;
    if (!scalar_effects_known_) return;
    const auto render = extract_render_state(state);
    if (!ordinary_draw(state, render) || draw.vs_guest_addr != render.es_addr ||
        draw.fs_guest_addr != render.ps_addr || draw.vs_chain_guest_addr || !draw.gs.empty() ||
        !coupled_read_only(draw.original_vs_source, render.es_addr, ShaderProgramStage::Vertex) ||
        !coupled_read_only(draw.original_ps_source, render.ps_addr, ShaderProgramStage::Fragment) ||
        !draw.original_graphics_effects ||
        !draw.original_graphics_effects->matches_draw(submit, draw.command_order, draw.vs_shared,
                                                      draw.fs_shared, draw.gs, draw.fs_words()) ||
        !draw.native_vs_source || draw.native_vs_source->stage() != ShaderProgramStage::Vertex ||
        !draw.native_vs_source->matches(draw.original_vs_source, draw.vs_shared) ||
        (draw.fragment_draw_inputs && draw.fragment_draw_inputs->original_fragment_producer
             ? !draw.fragment_draw_inputs->original_fragment_producer->matches(
                   *draw.fragment_draw_inputs)
             : !draw.native_ps_source ||
                   draw.native_ps_source->stage() != ShaderProgramStage::Fragment ||
                   !draw.native_ps_source->matches(draw.original_ps_source, draw.fs_shared))) {
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
        retained->sources.push_back(
            {address, source->words, fragment ? source->fragment_effects : source->vertex_effects,
             source->packet_requirements, source->decoded, source->native_analysis,
             source->registered_header, source->header_snapshot, fragment});
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
    for (const auto [source, address, stage, name] :
         {std::tuple{&vertex, render.es_addr, ShaderProgramStage::Vertex, "vs"},
          std::tuple{&fragment, render.ps_addr, ShaderProgramStage::Fragment, "ps"}})
        if (const char* gap = coupled_read_only_gap(*source, address, stage)) {
            // Cached code facts name the ORIGINAL stage/PC gap without rescanning warm code.
            refusal = std::string("scalar-bank-") + name + ":" + gap;
            return {};
        }
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
                           source->packet_requirements, source->decoded, source->native_analysis,
                           source->registered_header, source->header_snapshot, pixel});
    static std::atomic<uint64_t> next_identity{1};
    const uint64_t identity = next_identity.fetch_add(1, std::memory_order_relaxed);
    if (!identity) return reject("scalar-bank-identity-exhausted");
    return std::shared_ptr<const OrderedScalarBankReadPoint>(new OrderedScalarBankReadPoint(
        identity, submit, order, baseline.failures, state, std::move(entry), std::move(sources),
        outputs, scalar_effects_, epoch_, epoch_->sequence.load(std::memory_order_acquire),
        version));
}
OrderedScalarDrawInputs prepare_ordered_scalar_draw(OrderedGraphicsReadPointIssuer& issuer,
                                                    uint64_t submit, uint64_t order,
                                                    const GpuState& state,
                                                    const std::vector<DrawItem>& pending) {
    OrderedScalarDrawInputs result;
    std::string refusal;
    result.point = issuer.issue_scalar(submit, order, state, pending, refusal);
    const auto render = extract_render_state(state);
    if (result.point) {
        const auto requirements = result.point->packet_requirements(render.ps_addr);
        if (!render.ps_wave32 && requirements && requirements->scalar_reads.has_smem) {
            prosper::GuestMappingLease lease;
            result.bank =
                seal_fragment_scalar_bank(*result.point, render.ps_addr, order, lease, refusal);
        }
    }
    if (!refusal.empty()) {
        static std::atomic<uint64_t> refusals{0};
        const auto count = refusals.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((count & (count - 1)) == 0)
            std::fprintf(
                stderr,
                "[scalar-bank-refusal] count=%llu submit=%llu order=%llu ps=0x%llx reason=%s\n",
                static_cast<unsigned long long>(count), static_cast<unsigned long long>(submit),
                static_cast<unsigned long long>(order),
                static_cast<unsigned long long>(render.ps_addr), refusal.c_str());
    }
    return result;
}
}   // namespace prosper::gpu
