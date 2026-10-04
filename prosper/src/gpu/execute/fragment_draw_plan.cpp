#include "gpu/execute/fragment_draw_plan.hpp"
#include "gpu/execute/fragment_raster_launch.hpp"
#include "gpu/execute/fragment_raster_launch_collection.hpp"
#include "gpu/execute/fragment_raster_contract.hpp"
#include "gpu/execute/fragment_raster_program.hpp"
#include "gpu/execute/fragment_scalar_bank.hpp"
#include "gpu/recompiler/original_fragment_producer.hpp"
#include "gpu/recompiler/rdna2_waitcnt.hpp"
#include <array>
#include <bit>
#include <map>
#include <optional>

namespace prosper::gpu {
namespace {
bool producing_owner(const RasterQuadInputs& in, const FragmentPacketPreparation& prepared) {
    if (prepared.inputs.get() == &in && in.original_fragment_producer &&
        in.original_fragment_producer->matches(in))
        return prepared.vgpr_requirements && prepared.vgpr_requirements == in.vgpr_requirements &&
               in.vgpr_requirements->source_words == in.raw_code.get() &&
               in.vgpr_requirements->rejection.empty();
    const bool pending_original = in.owned_wave_pending && in.source_fs && in.source_fs->empty() &&
                                  in.launch_source && prepared.launch_source == in.launch_source &&
                                  in.launch_source->matches(in);
    return prepared.inputs.get() == &in && in.raw_matches_producing_source && in.raw_code &&
           !in.raw_code->empty() && in.raw_code->size() <= 4096 && in.source_vs &&
           !in.source_vs->empty() && in.source_fs && (!in.source_fs->empty() || pending_original) &&
           prepared.vgpr_requirements && prepared.vgpr_requirements == in.vgpr_requirements &&
           in.vgpr_requirements->source_words == in.raw_code.get() &&
           in.vgpr_requirements->rejection.empty();
}
uint32_t user_presence(const FragmentPacketPreparation& prepared) {
    uint32_t presence = 0;
    for (const auto& [reg, value] : prepared.initial_user_sgprs) {
        (void)value;
        if (reg < 32) presence |= uint32_t(1) << reg;
    }
    return presence;
}
bool launch_owned(const RasterQuadInputs& in, const FragmentPacketPreparation& prepared) {
    return in.entry.observed && in.entry.canonical() && in.entry.rsrc2_available &&
           prepared.user_sgpr_count_available && prepared.user_sgpr_count <= 32 &&
           in.launch.canonical() && in.launch.ps_in_control_available &&
           in.launch.baryc_cntl_available && in.launch.input_ena_available &&
           in.launch.input_addr_available && in.float_mode.canonical() && in.float_mode.available &&
           in.float_flags.canonical() && in.float_flags.available && in.launch_rsrc1.canonical() &&
           in.launch_rsrc1.available && in.float_transport.explicit_nonfinite32();
}
bool input_free_layout(const RasterQuadInputs& in) {
    const auto& layout = in.interpolation;
    if (!layout.valid || layout.requires_geometry || layout.attribute_mask || layout.smooth_mask ||
        layout.passthrough_mask || layout.flat_mask)
        return false;
    for (const auto& attribute : layout.parameter_locations)
        for (uint32_t location : attribute)
            if (location != FragmentInterpolationLayout::kUnusedLocation) return false;
    for (uint32_t location : layout.system_locations)
        if (location != FragmentInterpolationLayout::kUnusedLocation) return false;
    return true;
}
bool resource_free(const RasterQuadInputs& in) {
    return in.ps_resources.observed && in.ps_resources.table && in.ps_resources.rejection.empty() &&
           in.ps_resources.table->resources.empty() && in.ps_resources.host_backing_owned.empty();
}
bool scalar_bank_program(const RasterQuadInputs& in) {
    return in.vgpr_requirements && in.vgpr_requirements->scalar_reads.has_smem;
}
bool canonical_packet_owner(const RasterQuadInputs& in) {
    return in.raw_code && in.vgpr_requirements && !in.raw_code.owner_before(in.vgpr_requirements) &&
           !in.vgpr_requirements.owner_before(in.raw_code) &&
           in.vgpr_requirements->source_words == in.raw_code.get() &&
           in.vgpr_requirements->masks.source_words == in.raw_code.get() &&
           in.vgpr_requirements->scalar_reads.source_words == in.raw_code.get();
}
// Replay currently has one complete uncompressed color event. Other genuine EXP controls are
// recipe gaps, not corrupt output or a general Architectural EXP restriction. Name the original
// site BEFORE the broader packing proof can hide the reason behind an entry/composition gap.
std::optional<uint32_t> attachment_export_recipe_gap(const std::vector<Rdna2Inst>& instructions) {
    uint32_t sites = 0, end_pc = UINT32_MAX;
    for (const auto& in : instructions) {
        if (in.is_end) end_pc = in.pc;
        if (in.fmt == Rdna2Format::EXP &&
            (++sites > 1 || in.exp_target != 0 || in.exp_en != 15 || in.exp_compr ||
             !(in.words[0] & (1u << 11)) || !(in.words[0] & (1u << 12))))
            return in.pc;
    }
    return sites ? std::nullopt : std::optional<uint32_t>{end_pc};
}
// A FIRST recipe proof, not a title/opcode allowlist in the general packet compiler. This narrow
// fragment is insensitive to inter-quad placement and cannot observe private scratch workers.
// Wider recipes must establish the extra composition/input/resource facts, not pretend this is
// full Wave64. Full masks may come from actual dynamic user words; their values are checked at
// instantiation, never baked into code or inferred from full observed host coverage.
bool packing_unobservable(const std::vector<Rdna2Inst>& instructions, uint32_t code_words,
                          uint32_t presence, std::vector<FragmentDrawFullMask>& full_masks,
                          std::span<const FragmentPacketScalarReadSite> scalar_sites = {}) {
    std::array<std::optional<FragmentDrawMaskWord>, 106> scalars;
    std::array<bool, 106> scalar_defined{}, scalar_ready{};
    std::array<bool, 256> vectors{};
    for (uint32_t reg = 0; reg < 32; ++reg)
        if (presence & (uint32_t(1) << reg)) {
            scalars[reg] = FragmentDrawMaskWord{true, reg};
            scalar_defined[reg] = scalar_ready[reg] = true;
        }
    auto word = [&](const Operand& source, const Rdna2Inst& in,
                    bool vector_source) -> std::optional<FragmentDrawMaskWord> {
        if (source.kind == OperandKind::SGPR && source.value >= 0 && source.value < 106) {
            if (!scalar_defined[source.value] || !scalar_ready[source.value]) return {};
            if (scalars[source.value]) return scalars[source.value];
            // A completed scalar read defines a value, but not a known full EXEC mask.
            return vector_source ? std::optional{FragmentDrawMaskWord{}} : std::nullopt;
        }
        if (source.kind == OperandKind::InlineInt)
            return FragmentDrawMaskWord{false, static_cast<uint32_t>(source.value)};
        if (source.kind == OperandKind::Literal && in.has_literal)
            return FragmentDrawMaskWord{false, in.literal};
        if (source.kind == OperandKind::InlineFloat)
            return FragmentDrawMaskWord{false,
                                        std::bit_cast<uint32_t>(inline_float_value(source.value))};
        if (vector_source && source.kind == OperandKind::VGPR && source.value >= 0 &&
            source.value < 256 && vectors[source.value])
            return FragmentDrawMaskWord{};   // definition proof only; never used as a scalar value
        return {};
    };
    std::optional<std::pair<FragmentDrawMaskWord, FragmentDrawMaskWord>> exec_mask;
    const auto require_full_exec = [&](uint32_t pc) {
        if (!exec_mask) return false;
        full_masks.push_back({pc, exec_mask->first, exec_mask->second});
        return true;
    };
    bool ended = false;
    uint32_t exports = 0;
    for (const auto& in : instructions) {
        if (ended || in.synthetic_terminator || in.has_modifier || in.has_sdwa || in.has_dpp ||
            in.pc >= code_words || !in.len_dwords || in.len_dwords > code_words - in.pc)
            return false;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64) {
            std::optional<FragmentDrawMaskWord> lo, hi;
            if (in.src[0].kind == OperandKind::SGPR && in.src[0].value >= 0 &&
                in.src[0].value < 105) {
                if (!scalar_defined[in.src[0].value] || !scalar_defined[in.src[0].value + 1] ||
                    !scalar_ready[in.src[0].value] || !scalar_ready[in.src[0].value + 1])
                    return false;
                lo = scalars[in.src[0].value];
                hi = scalars[in.src[0].value + 1];
            } else if (in.src[0].kind == OperandKind::InlineInt) {
                lo = FragmentDrawMaskWord{false, static_cast<uint32_t>(in.src[0].value)};
                hi = FragmentDrawMaskWord{false, in.src[0].value < 0 ? UINT32_MAX : 0};
            } else
                return false;
            if (in.dst.kind == OperandKind::SGPR && in.dst.value == 126) {
                exec_mask = lo && hi ? std::optional{std::pair{*lo, *hi}} : std::nullopt;
            } else if (in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 &&
                       in.dst.value < 105) {
                scalars[in.dst.value] = lo;
                scalars[in.dst.value + 1] = hi;
                scalar_defined[in.dst.value] = scalar_defined[in.dst.value + 1] = true;
                scalar_ready[in.dst.value] = scalar_ready[in.dst.value + 1] = true;
            } else
                return false;
        } else if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 &&
                   in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value < 106) {
            const auto value = word(in.src[0], in, false);
            if (!value && !(in.src[0].kind == OperandKind::SGPR && in.src[0].value >= 0 &&
                            in.src[0].value < 106 && scalar_defined[in.src[0].value] &&
                            scalar_ready[in.src[0].value]))
                return false;
            scalars[in.dst.value] = value;
            scalar_defined[in.dst.value] = scalar_ready[in.dst.value] = true;
        } else if (in.fmt == Rdna2Format::SMEM) {
            const auto site = std::find_if(scalar_sites.begin(), scalar_sites.end(),
                                           [&](const auto& value) { return value.pc == in.pc; });
            if (site == scalar_sites.end() || site->opcode != in.opcode ||
                site->byte_offset != in.literal || in.dst.kind != OperandKind::SGPR ||
                (site->words != 1 && site->words != 2 && site->words != 4) || in.dst.value < 0 ||
                uint32_t(in.dst.value) > scalars.size() - site->words)
                return false;
            for (uint32_t word = 0; word < site->words; ++word) {
                const auto reg = in.dst.value + word;
                scalars[reg].reset();
                scalar_defined[reg] = true;
                scalar_ready[reg] = false;
            }
        } else if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0c) {
            const auto immediate = uint16_t(in.simm16);
            if (rdna2_waitcnt_execution_gap(immediate)) return false;
            if (decode_rdna2_waitcnt(immediate).drains_scalar_reads())
                scalar_ready = scalar_defined;
        } else if (in.fmt == Rdna2Format::VOP1 && in.opcode == 1 &&
                   in.dst.kind == OperandKind::VGPR && in.dst.value >= 0 && in.dst.value < 256) {
            if (!word(in.src[0], in, true) || !require_full_exec(in.pc)) return false;
            vectors[in.dst.value] =
                true;   // the retained full-mask requirements cover ALL64 writers
        } else if (in.fmt == Rdna2Format::EXP && in.exp_target == 0 && in.exp_en == 15 &&
                   !in.exp_compr && (in.words[0] & (1u << 11)) && (in.words[0] & (1u << 12))) {
            if (++exports != 1 || !require_full_exec(in.pc)) return false;
            for (const auto& source : in.src)
                if (source.kind != OperandKind::VGPR || source.value < 0 || source.value >= 256 ||
                    !vectors[source.value])
                    return false;
        } else if (in.fmt == Rdna2Format::SOPP && in.is_end && in.opcode == 1) {
            ended = in.pc + in.len_dwords == code_words;
            if (!ended) return false;
        } else if (!(in.fmt == Rdna2Format::SOPP && in.opcode == 0))
            return false;
    }
    return ended && exports == 1 && !full_masks.empty();
}
std::vector<uint32_t> profile_key(const RasterQuadInputs& in,
                                  const FragmentPacketPreparation& prepared,
                                  FragmentPacketDeviceContract device, uint32_t max_quads) {
    if (!producing_owner(in, prepared) || !launch_owned(in, prepared) ||
        (!scalar_bank_program(in) && !resource_free(in)) ||
        (!input_free_layout(in) && (!device.raster || prepared.launch_source != in.launch_source ||
                                    !in.launch_source || !in.launch_source->matches(in))))
        return {};
    std::vector<uint32_t> key{max_quads,
                              prepared.user_sgpr_count,
                              user_presence(prepared),
                              in.entry.rsrc2,
                              in.launch_rsrc1.value,
                              in.float_mode.value,
                              uint32_t(in.float_flags.ieee_mode),
                              uint32_t(in.float_flags.dx10_clamp),
                              uint32_t(in.float_transport.profile),
                              uint32_t(FragmentPacketExportObservation::Architectural),
                              uint32_t(device.device_identity),
                              uint32_t(device.device_identity >> 32),
                              uint32_t(device.shader_int64_enabled),
                              uint32_t(device.rgba32_sfloat_sampled),
                              uint32_t(scalar_bank_program(in)),
                              uint32_t(!scalar_bank_program(in) && resource_free(in))};
    key.push_back(bool(device.raster));
    if (device.raster) {
        const auto& raster = *device.raster;
        key.insert(key.end(),
                   {uint32_t(raster.geometry_shader_enabled), raster.max_vertex_output_components,
                    raster.max_geometry_input_components, raster.max_geometry_output_components,
                    raster.max_geometry_total_output_components,
                    raster.max_geometry_output_vertices, raster.max_geometry_shader_invocations,
                    raster.max_fragment_input_components});
        // Only cold profile/shape facts enter the key, never dynamic USER_DATA values. A different
        // native varying/GS profile cannot reuse a complete-coefficient collection module.
        for (const auto* module : {in.source_vs.get(), in.source_gs.get(), in.source_fs.get()}) {
            const uint64_t identity = reinterpret_cast<uintptr_t>(module);
            key.insert(key.end(), {uint32_t(identity), uint32_t(identity >> 32)});
        }
        key.insert(key.end(), {in.launch.input_ena, in.launch.input_addr,
                               in.interpolation.attribute_mask, in.interpolation.smooth_mask,
                               in.interpolation.flat_mask, in.interpolation.passthrough_mask,
                               uint32_t(in.generated_interpolation_geometry)});
        for (const auto& locations : in.interpolation.parameter_locations)
            key.insert(key.end(), locations.begin(), locations.end());
        key.insert(key.end(), in.interpolation.system_locations.begin(),
                   in.interpolation.system_locations.end());
        key.insert(key.end(),
                   {uint32_t(in.owned_wave_pending),
                    uint32_t(in.launch.sc_shader_control_available), in.launch.sc_shader_control,
                    uint32_t(in.launch.sc_mode_cntl_0_available), in.launch.sc_mode_cntl_0,
                    uint32_t(in.launch.sc_mode_cntl_1_available), in.launch.sc_mode_cntl_1,
                    uint32_t(in.launch.sc_aa_config_available), in.launch.sc_aa_config,
                    uint32_t(in.launch.db_shader_control_available), in.launch.db_shader_control,
                    in.launch.coverage.available});
        key.insert(key.end(), in.launch.coverage.words.begin(), in.launch.coverage.words.end());
    }
    return key;
}
struct SourceProfileKey {
    std::weak_ptr<const std::vector<uint32_t>> source;
    const std::vector<uint32_t>* words = nullptr;
    std::vector<uint32_t> profile;
    bool operator<(const SourceProfileKey& other) const {
        if (source.owner_before(other.source)) return true;
        if (other.source.owner_before(source)) return false;
        if (words != other.words)
            return std::less<const std::vector<uint32_t>*>{}(words, other.words);
        return profile < other.profile;
    }
};
}   // namespace

FragmentDrawProgramPlan compile_fragment_draw_program(const RasterQuadInputs& in,
                                                      const FragmentPacketPreparation& prepared,
                                                      FragmentPacketDeviceContract device,
                                                      uint32_t max_quads,
                                                      RecompileDiagnosticContext diagnostic) {
    ++fragment_draw_cache_stats().program_compile_calls;
    FragmentDrawProgramPlan result;
    result.source_generations->remember(in.raw_code);
    if (device.raster)
        result.raster_module_generations = {in.source_vs, in.source_gs, in.source_fs};
    const auto refuse = [&](const std::string& reason) {
        FragmentDrawProgramPlan failed;
        failed.source_generations = result.source_generations;
        failed.raster_module_generations = result.raster_module_generations;
        failed.device = device;
        failed.rejection = reason;
        return failed;
    };
    if (!producing_owner(in, prepared)) return refuse("fragment-draw-producing-owner-unavailable");
    if (!launch_owned(in, prepared)) {
        if (device.raster) {
            if (!in.float_transport.explicit_nonfinite32())
                return refuse("fragment-draw-enabled-float-transport-unavailable");
            if (!in.float_mode.available || !in.float_mode.canonical())
                return refuse("fragment-draw-original-float-mode-unavailable");
            if (!in.float_flags.available || !in.float_flags.canonical())
                return refuse("fragment-draw-original-float-flags-unavailable");
            if (!in.launch_rsrc1.available || !in.launch_rsrc1.canonical())
                return refuse("fragment-draw-original-rsrc1-unavailable");
            if (!prepared.user_sgpr_count_available)
                return refuse("fragment-draw-original-user-prefix-count-unavailable");
        }
        return refuse("fragment-draw-original-launch-unavailable");
    }
    if (!device.device_identity || !device.shader_int64_enabled)
        return refuse("fragment-draw-enabled-device-unavailable");
    const bool scalar_bank = scalar_bank_program(in);
    if (!scalar_bank && !resource_free(in))
        return refuse("fragment-draw-resource-lease-unimplemented");
    const auto& scalar_reads = in.vgpr_requirements->scalar_reads;
    if (scalar_bank &&
        (scalar_reads.source_words != in.raw_code.get() || !scalar_reads.rejection.empty() ||
         scalar_reads.sites.empty() || scalar_reads.sites.size() > 64))
        return refuse("fragment-draw-scalar-bank-original-site-schema-unproved");
    if (scalar_bank && !input_free_layout(in))
        return refuse("fragment-draw-parameter-system-entry-recipe-unimplemented");
    result.user_prefix_count = prepared.user_sgpr_count;
    result.user_prefix_presence = user_presence(prepared);
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(in.raw_code->data(), in.raw_code->size(), instructions);
    if (const auto pc = attachment_export_recipe_gap(instructions))
        return refuse("fragment-draw-attachment-export-recipe-unimplemented:pc=" +
                      std::to_string(*pc));
    std::vector<FragmentDrawRasterInput> raster_inputs;
    const auto raster_recipe = [&]() {
        if (!device.raster || !in.launch_source || prepared.launch_source != in.launch_source ||
            !in.launch_source->matches(in))
            return refuse("fragment-draw-draw-bound-launch-source-unavailable");
        const auto original =
            fragment_raster_program(instructions, static_cast<uint32_t>(in.raw_code->size()),
                                    in.system_inputs, result.user_prefix_presence);
        if (!original.rejection.empty()) return refuse(original.rejection);
        if (const auto* gap = fragment_raster_workitem_gap(in.launch)) return refuse(gap);
        auto collection = compile_fragment_raster_launch_collection(in, device, max_quads);
        if (!collection.rejection.empty()) return refuse(collection.rejection);
        result.raster_collection =
            std::make_shared<const FragmentRasterLaunchCollection>(std::move(collection));
        result.raster_module_generations = {in.source_vs, in.source_gs, in.source_fs};
        result.entry_recipe = FragmentDrawEntryRecipe::DrawBoundRasterSystemAndQuadMasks;
        result.scheduling = FragmentDrawScheduling::QuadLocalObservationalEquivalence;
        result.raster_profile = in.launch;
        result.device = device;
        result.collect = result.raster_collection->collector;
        result.collector = result.raster_collection->shape;
        for (const auto& row : original.positions)
            raster_inputs.push_back({row.reg, row.collector_word});
        return result;
    };
    const bool old_recipe =
        (scalar_bank || !in.source_fs->empty()) && input_free_layout(in) &&
        packing_unobservable(instructions, static_cast<uint32_t>(in.raw_code->size()),
                             result.user_prefix_presence, result.full_masks,
                             scalar_bank ? std::span{scalar_reads.sites}
                                         : std::span<const FragmentPacketScalarReadSite>{});
    if (!old_recipe) {
        result.full_masks.clear();
        if (scalar_bank) return refuse("fragment-draw-entry-and-composition-recipe-unproved");
        if (!device.raster)
            return refuse(input_free_layout(in)
                              ? "fragment-draw-entry-and-composition-recipe-unproved"
                              : "fragment-draw-parameter-system-entry-recipe-unimplemented");
        auto raster = raster_recipe();
        if (!raster.rejection.empty()) return raster;
    } else {
        result.collect = build_raster_quad_collector(in, max_quads, result.collector);
        if (result.collect.empty()) {
            result.rejection = result.collector.rejection;
            return result;
        }
        if (!result.collector.fields.empty())
            return refuse("fragment-draw-parameter-system-entry-recipe-unimplemented");
    }
    FragmentResourcePacket schema;
    if (scalar_bank) schema.scalar_bank_sites = scalar_reads.sites;
    auto& invocation = schema.invocation;
    invocation.guest_code = *in.raw_code;
    // This is the code-generation AND completed-consumer policy, selected before the original
    // PS is compiled. A 14-word stride alone never upgrades a LegacyRaw observation to authority.
    invocation.export_observation = FragmentPacketExportObservation::Architectural;
    invocation.slots_available.fill(true);   // owned execution storage, NOT guest initial EXEC
    invocation.quad_topology = FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
    invocation.float_mode = in.float_mode;
    invocation.float_flags = in.float_flags;
    invocation.float_transport = in.float_transport;
    if (!raster_inputs.empty()) {
        invocation.exec_available = true;   // values supplied only by the draw-bound GPU assembler
        for (const auto& row : raster_inputs) {
            FragmentPacketVgpr column;
            column.reg = row.reg;
            column.available_mask = 0;   // dynamic per-lane validity, never initial zero authority
            invocation.vgprs.push_back(column);
        }
    }
    // Values are absent compile-time placeholders in a dynamically loaded schema. No initial
    // mask, VGPR, system SGPR, M0, coefficient, helper or resource word is manufactured here.
    for (uint32_t reg = 0; reg < 32; ++reg)
        if (result.user_prefix_presence & (uint32_t(1) << reg))
            invocation.sgprs.emplace_back(reg, 0);
    schema.launch_rsrc1 = in.launch_rsrc1;
    schema.entry_facts = in.entry;
    schema.entry_facts.user_data = {};
    schema.entry_facts.user_data_available = 0;
    schema.device = device;
    auto kernel = std::make_shared<const FragmentPacketKernel>(
        recompile_fragment_packet_capacity_kernel(schema, diagnostic));
    if (kernel->program.packet.spirv.empty()) {
        result.rejection = kernel->program.packet.rejection;
        return result;
    }
    if (!fragment_draw_architectural_exports_match(*kernel))
        return refuse("fragment-draw-architectural-exp-required");
    result.capacity = fragment_draw_capacity(kernel, result.collector, result.rejection,
                                             std::move(raster_inputs));
    if (!result.capacity) return result;
    result.count = build_fragment_draw_count(*result.capacity, result.collector);
    result.assemble = build_fragment_draw_assembly(*result.capacity, result.collector);
    result.validate = build_fragment_draw_validation(*result.capacity, result.collector);
    result.replay = build_fragment_draw_replay(*result.capacity, result.collector);
    if (result.count.empty() || result.assemble.empty() || result.validate.empty() ||
        result.replay.empty())
        return refuse("fragment-draw-transaction-source-unavailable");
    result.replay_shared = std::make_shared<const std::vector<uint32_t>>(result.replay);
    result.rsrc2 = in.entry.rsrc2;
    result.float_mode = in.float_mode;
    result.float_flags = in.float_flags;
    result.launch_rsrc1 = in.launch_rsrc1;
    result.transport = in.float_transport;
    result.device = device;
    return result;
}

std::shared_ptr<const FragmentDrawProgramPlan>
cached_fragment_draw_program(const RasterQuadInputs& in, const FragmentPacketPreparation& prepared,
                             FragmentPacketDeviceContract device, uint32_t max_quads,
                             RecompileDiagnosticContext diagnostic) {
    static thread_local std::map<std::vector<uint32_t>,
                                 std::shared_ptr<const FragmentDrawProgramPlan>>
        cache;
    // The canonical immutable owner/profile alias avoids copying or comparing original code on
    // warm hits. Cold/new generations still join the copied code-only cache; no V#/VA/payload
    // belongs in either key. Weak aliases cannot keep their own retired generation alive.
    static thread_local std::map<SourceProfileKey, std::weak_ptr<const FragmentDrawProgramPlan>>
        aliases;
    auto profile = profile_key(in, prepared, device, max_quads);
    const bool canonical = canonical_packet_owner(in);
    const auto same_modules = [&](const FragmentDrawProgramPlan& resident) {
        if (!device.raster) return true;
        const std::array<std::shared_ptr<const std::vector<uint32_t>>, 3> modules{
            in.source_vs, in.source_gs, in.source_fs};
        for (size_t index = 0; index < modules.size(); ++index) {
            const auto& retained = resident.raster_module_generations[index];
            if (!modules[index] || retained.lock().get() != modules[index].get() ||
                retained.owner_before(modules[index]) || modules[index].owner_before(retained))
                return false;
        }
        return true;
    };
    SourceProfileKey alias{in.raw_code, in.raw_code.get(), profile};
    if (canonical && !profile.empty())
        if (const auto found = aliases.find(alias); found != aliases.end())
            if (auto resident = found->second.lock()) {
                if (resident->source_live() && same_modules(*resident)) {
                    ++fragment_draw_cache_stats().program_hits;
                    return resident;
                }
            }
    auto key = std::move(profile);
    if (!key.empty()) key.insert(key.end(), in.raw_code->begin(), in.raw_code->end());
    if (!key.empty()) {
        const auto found = cache.find(key);
        if (found != cache.end()) {
            if (same_modules(*found->second)) {
                found->second->source_generations->remember(in.raw_code);
                if (canonical) aliases.insert_or_assign(std::move(alias), found->second);
                ++fragment_draw_cache_stats().program_hits;
                return found->second;
            }
            cache.erase(found);   // a reused address never stands in for the selected module owner
        }
    }
    retire_dead_fragment_draw_entries(cache, fragment_draw_cache_stats().program_retired);
    for (auto item = aliases.begin(); item != aliases.end();)
        if (item->first.source.expired() || item->second.expired())
            item = aliases.erase(item);
        else
            ++item;
    auto compiled = compile_fragment_draw_program(in, prepared, device, max_quads, diagnostic);
    auto result = std::make_shared<const FragmentDrawProgramPlan>(std::move(compiled));
    if (!key.empty()) {
        // All live source/profile generations stay warm, including named refused profiles.
        // No cached payload retains its own weakly tracked source owner. Completion leases
        // may outlive retirement without causing recompilation of another live generation.
        cache.emplace(std::move(key), result);
        if (canonical) aliases.insert_or_assign(std::move(alias), result);
    }
    return result;
}

FragmentDrawTransaction instantiate_fragment_draw_transaction(
    std::shared_ptr<const FragmentDrawProgramPlan> program,
    std::shared_ptr<const RasterQuadInputs> in, const FragmentPacketPreparation& prepared,
    uint32_t width, uint32_t height, uint32_t primitive_count, uint64_t executing_device_identity) {
    const auto refuse = [&](const char* reason) { return FragmentDrawTransaction(reason); };
    if (!program || !program->rejection.empty() || !program->capacity)
        return refuse("fragment-draw-program-unavailable");
    if (program->entry_recipe == FragmentDrawEntryRecipe::DrawBoundRasterSystemAndQuadMasks && in) {
        const std::array<std::shared_ptr<const std::vector<uint32_t>>, 3> modules{
            in->source_vs, in->source_gs, in->source_fs};
        for (size_t index = 0; index < modules.size(); ++index) {
            const auto& retained = program->raster_module_generations[index];
            if (!modules[index] || retained.lock().get() != modules[index].get() ||
                retained.owner_before(modules[index]) || modules[index].owner_before(retained))
                return refuse("fragment-draw-producing-raster-module-generation-mismatch");
        }
    }
    if (!in || !producing_owner(*in, prepared) || !launch_owned(*in, prepared) ||
        (!program->requires_scalar_bank() && !resource_free(*in)) ||
        (program->entry_recipe == FragmentDrawEntryRecipe::OwnedUserPrefixAndShaderDefinedMasks
             ? (!input_free_layout(*in) ||
                (!program->requires_scalar_bank() && in->source_fs->empty()))
             : (!in->launch_source || prepared.launch_source != in->launch_source ||
                !in->launch_source->matches(*in) || in->launch != program->raster_profile ||
                fragment_raster_workitem_gap(in->launch))) ||
        (!(canonical_packet_owner(*in) && program->source_generations->owns(in->raw_code)) &&
         *in->raw_code != program->capacity->kernel()->guest_code) ||
        prepared.user_sgpr_count != program->user_prefix_count ||
        user_presence(prepared) != program->user_prefix_presence ||
        in->entry.rsrc2 != program->rsrc2 || in->float_mode != program->float_mode ||
        in->float_flags != program->float_flags || in->launch_rsrc1 != program->launch_rsrc1 ||
        in->float_transport != program->transport)
        return refuse("fragment-draw-producing-profile-mismatch");
    if (!executing_device_identity || executing_device_identity != program->device.device_identity)
        return refuse("fragment-draw-executing-device-mismatch");
    if (program->requires_scalar_bank() &&
        (!in->scalar_bank ||
         !in->scalar_bank->matches(in->raw_code, in->vgpr_requirements, in->entry)))
        return refuse("fragment-draw-scalar-bank-producing-owner-mismatch");
    if (!width || !height || width > 8192 || height > 8192 || !primitive_count)
        return refuse("fragment-draw-raster-extent-invalid");
    std::array<std::optional<uint32_t>, 32> words;
    for (const auto& [reg, value] : prepared.initial_user_sgprs) {
        if (reg >= prepared.user_sgpr_count || words[reg] ||
            !(in->entry.user_data_available & (uint32_t(1) << reg)) ||
            value != in->entry.user_data[reg])
            return refuse("fragment-draw-user-prefix-owner-mismatch");
        words[reg] = value;
    }
    const auto evaluate = [&](FragmentDrawMaskWord word) -> std::optional<uint32_t> {
        if (!word.user_word) return word.word;
        return word.word < words.size() ? words[word.word] : std::nullopt;
    };
    for (const auto& mask : program->full_masks) {
        const auto lo = evaluate(mask.lo), hi = evaluate(mask.hi);
        if (!lo || !hi || *lo != UINT32_MAX || *hi != UINT32_MAX)
            return refuse("fragment-draw-observable-mask-composition-unproved");
    }
    const auto& layout = program->capacity->kernel()->layout;
    std::vector<uint32_t> entry_words(4 + layout.input_words,
                                      0);   // storage, never guest zero authority
    entry_words[0] = primitive_count;
    entry_words[1] = width;
    entry_words[2] = height;
    entry_words[3] = kFragmentDrawInputMagic;
    for (size_t index = 0; index < layout.sgprs.size(); ++index) {
        const auto reg = layout.sgprs[index];
        if (reg >= words.size() || !words[reg])
            return refuse("fragment-draw-user-prefix-word-unavailable");
        entry_words[4 + layout.scalar_offsets[index]] = *words[reg];
        entry_words[4 + layout.scalar_available_offsets[index]] = 1;
    }
    auto bank = program->requires_scalar_bank() ? in->scalar_bank : nullptr;
    return FragmentDrawTransaction(std::move(program), std::move(in), std::move(entry_words), width,
                                   height, primitive_count, executing_device_identity,
                                   std::move(bank));
}
}   // namespace prosper::gpu
