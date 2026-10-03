#include "gpu/execute/fragment_draw_plan.hpp"
#include <array>
#include <bit>
#include <map>
#include <optional>

namespace prosper::gpu {
namespace {
bool producing_owner(const RasterQuadInputs& in, const FragmentPacketPreparation& prepared) {
    return prepared.inputs.get() == &in && in.raw_matches_producing_source && in.raw_code &&
           !in.raw_code->empty() && in.raw_code->size() <= 4096 && in.source_vs &&
           !in.source_vs->empty() && in.source_fs && !in.source_fs->empty() &&
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
// A FIRST recipe proof, not a title/opcode allowlist in the general packet compiler. This narrow
// fragment is insensitive to inter-quad placement and cannot observe private scratch workers.
// Wider recipes must establish the extra composition/input/resource facts, not pretend this is
// full Wave64. Full masks may come from actual dynamic user words; their values are checked at
// instantiation, never baked into code or inferred from full observed host coverage.
bool packing_unobservable(const std::vector<Rdna2Inst>& instructions, uint32_t code_words,
                          uint32_t presence, std::vector<FragmentDrawFullMask>& full_masks) {
    std::array<std::optional<FragmentDrawMaskWord>, 106> scalars;
    std::array<bool, 256> vectors{};
    for (uint32_t reg = 0; reg < 32; ++reg)
        if (presence & (uint32_t(1) << reg)) scalars[reg] = FragmentDrawMaskWord{true, reg};
    auto word = [&](const Operand& source, const Rdna2Inst& in,
                    bool vector_source) -> std::optional<FragmentDrawMaskWord> {
        if (source.kind == OperandKind::SGPR && source.value >= 0 && source.value < 106)
            return scalars[source.value];
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
    bool exec_defined = false, ended = false;
    uint32_t exports = 0;
    for (const auto& in : instructions) {
        if (ended || in.synthetic_terminator || in.has_modifier || in.has_sdwa || in.has_dpp ||
            in.pc >= code_words || !in.len_dwords || in.len_dwords > code_words - in.pc)
            return false;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64) {
            std::optional<FragmentDrawMaskWord> lo, hi;
            if (in.src[0].kind == OperandKind::SGPR && in.src[0].value >= 0 &&
                in.src[0].value < 105) {
                lo = scalars[in.src[0].value];
                hi = scalars[in.src[0].value + 1];
            } else if (in.src[0].kind == OperandKind::InlineInt) {
                lo = FragmentDrawMaskWord{false, static_cast<uint32_t>(in.src[0].value)};
                hi = FragmentDrawMaskWord{false, in.src[0].value < 0 ? UINT32_MAX : 0};
            }
            if (!lo || !hi) return false;
            if (in.dst.kind == OperandKind::SGPR && in.dst.value == 126) {
                full_masks.push_back({in.pc, *lo, *hi});
                exec_defined = true;
            } else if (in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 &&
                       in.dst.value < 105) {
                scalars[in.dst.value] = lo;
                scalars[in.dst.value + 1] = hi;
            } else
                return false;
        } else if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 &&
                   in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value < 106) {
            const auto value = word(in.src[0], in, false);
            if (!value) return false;
            scalars[in.dst.value] = value;
        } else if (in.fmt == Rdna2Format::VOP1 && in.opcode == 1 && exec_defined &&
                   in.dst.kind == OperandKind::VGPR && in.dst.value >= 0 && in.dst.value < 256) {
            if (!word(in.src[0], in, true)) return false;
            vectors[in.dst.value] =
                true;   // the retained full-mask requirements cover ALL64 writers
        } else if (in.fmt == Rdna2Format::EXP && exec_defined && in.exp_target == 0 &&
                   in.exp_en == 15 && !in.exp_compr && (in.words[0] & (1u << 11)) &&
                   (in.words[0] & (1u << 12))) {
            if (++exports != 1) return false;
            for (const auto& source : in.src)
                if (source.kind != OperandKind::VGPR || source.value < 0 || source.value >= 256 ||
                    !vectors[source.value])
                    return false;
        } else if (in.fmt == Rdna2Format::SOPP && in.is_end && in.opcode == 1) {
            ended = in.pc + in.len_dwords == code_words;
            if (!ended) return false;
        } else if (!(in.fmt == Rdna2Format::SOPP &&
                     (in.opcode == 0 || (in.opcode == 0x0c && in.simm16 == 0))))
            return false;
    }
    return ended && exports == 1 && !full_masks.empty();
}
std::vector<uint32_t> profile_key(const RasterQuadInputs& in,
                                  const FragmentPacketPreparation& prepared,
                                  FragmentPacketDeviceContract device, uint32_t max_quads) {
    if (!producing_owner(in, prepared) || !launch_owned(in, prepared) || !resource_free(in) ||
        !input_free_layout(in))
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
                              uint32_t(device.device_identity),
                              uint32_t(device.device_identity >> 32),
                              uint32_t(device.shader_int64_enabled),
                              uint32_t(device.rgba32_sfloat_sampled)};
    key.insert(key.end(), in.raw_code->begin(), in.raw_code->end());
    return key;
}
}   // namespace

FragmentDrawProgramPlan compile_fragment_draw_program(const RasterQuadInputs& in,
                                                      const FragmentPacketPreparation& prepared,
                                                      FragmentPacketDeviceContract device,
                                                      uint32_t max_quads,
                                                      RecompileDiagnosticContext diagnostic) {
    FragmentDrawProgramPlan result;
    const auto refuse = [&](const char* reason) {
        FragmentDrawProgramPlan failed;
        failed.rejection = reason;
        return failed;
    };
    if (!producing_owner(in, prepared)) return refuse("fragment-draw-producing-owner-unavailable");
    if (!launch_owned(in, prepared)) return refuse("fragment-draw-original-launch-unavailable");
    if (!device.device_identity || !device.shader_int64_enabled)
        return refuse("fragment-draw-enabled-device-unavailable");
    if (!resource_free(in)) return refuse("fragment-draw-resource-lease-unimplemented");
    if (!input_free_layout(in))
        return refuse("fragment-draw-parameter-system-entry-recipe-unimplemented");
    result.user_prefix_count = prepared.user_sgpr_count;
    result.user_prefix_presence = user_presence(prepared);
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(in.raw_code->data(), in.raw_code->size(), instructions);
    if (!packing_unobservable(instructions, static_cast<uint32_t>(in.raw_code->size()),
                              result.user_prefix_presence, result.full_masks))
        return refuse("fragment-draw-entry-and-composition-recipe-unproved");
    result.collect = build_raster_quad_collector(in, max_quads, result.collector);
    if (result.collect.empty()) {
        result.rejection = result.collector.rejection;
        return result;
    }
    if (!result.collector.fields.empty())
        return refuse("fragment-draw-parameter-system-entry-recipe-unimplemented");
    FragmentResourcePacket schema;
    auto& invocation = schema.invocation;
    invocation.guest_code = *in.raw_code;
    invocation.slots_available.fill(true);   // owned execution storage, NOT guest initial EXEC
    invocation.quad_topology = FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
    invocation.float_mode = in.float_mode;
    invocation.float_flags = in.float_flags;
    invocation.float_transport = in.float_transport;
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
    // Architectural EXP and demanded absent-state support must be accepted upstream. Until then
    // this builder deliberately refuses; LegacyRaw output cannot authorize attachment publication.
    auto kernel = std::make_shared<const FragmentPacketKernel>(
        recompile_fragment_packet_capacity_kernel(schema, diagnostic));
    if (kernel->program.packet.spirv.empty()) {
        result.rejection = kernel->program.packet.rejection;
        return result;
    }
    if (kernel->program.status_offset != 64 * 14)
        return refuse("fragment-draw-architectural-exp-required");
    result.capacity = fragment_draw_capacity(kernel, result.collector, result.rejection);
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
    struct Cache {
        std::map<std::vector<uint32_t>, std::shared_ptr<const FragmentDrawProgramPlan>> entries;
        size_t bytes = 0;
    };
    static thread_local Cache cache;
    auto key = profile_key(in, prepared, device, max_quads);
    if (!key.empty()) {
        const auto found = cache.entries.find(key);
        if (found != cache.entries.end()) return found->second;
    }
    auto compiled = compile_fragment_draw_program(in, prepared, device, max_quads, diagnostic);
    size_t bytes = sizeof(compiled) + key.capacity() * sizeof(uint32_t) +
                   compiled.full_masks.capacity() * sizeof(FragmentDrawFullMask) +
                   compiled.collector.fields.capacity() * sizeof(RasterQuadField) +
                   compiled.collector.rejection.capacity() + compiled.rejection.capacity() + 2 +
                   256;
    for (const auto* words : {&compiled.collect, &compiled.count, &compiled.assemble,
                              &compiled.validate, &compiled.replay})
        bytes += words->capacity() * sizeof(uint32_t);
    if (compiled.capacity) {
        const auto& kernel = *compiled.capacity->kernel();
        bytes += sizeof(kernel) + kernel.guest_code.capacity() * sizeof(uint32_t) +
                 kernel.instructions.capacity() * sizeof(Rdna2Inst) +
                 kernel.program.packet.spirv.capacity() * sizeof(uint32_t) +
                 kernel.program.packet.input_words.capacity() * sizeof(uint32_t) +
                 kernel.program.packet.output_words.capacity() * sizeof(uint32_t) +
                 kernel.program.packet.vgpr_failure_sites.capacity() *
                     sizeof(FragmentPacketProgram::VgprFailureSite) +
                 kernel.program.packet.rejection.capacity() + 1 +
                 kernel.program.runtime_failure_pcs.capacity() * sizeof(uint32_t) +
                 sizeof(FragmentDrawCapacity) +
                 compiled.capacity->collector().fields.capacity() * sizeof(RasterQuadField) +
                 compiled.capacity->collector().rejection.capacity() + 1;
        for (const auto* offsets :
             {&kernel.layout.vgprs, &kernel.layout.sgprs, &kernel.layout.scalar_offsets,
              &kernel.layout.scalar_available_offsets, &kernel.layout.image_descriptor_offsets,
              &kernel.layout.sampler_offsets})
            bytes += offsets->capacity() * sizeof(uint32_t);
        bytes += kernel.layout.buffers.capacity() * sizeof(PacketWaveDataLayout::Buffer) +
                 kernel.layout.parameters.capacity() * sizeof(PacketWaveDataLayout::Parameter);
    }
    constexpr size_t budget = 16 * 1024 * 1024;
    if (bytes > budget) {
        compiled = {};
        compiled.rejection = "fragment-draw-program-cache-budget";
        bytes = sizeof(compiled) + key.capacity() * sizeof(uint32_t);
    }
    auto result = std::make_shared<const FragmentDrawProgramPlan>(std::move(compiled));
    if (!key.empty()) {
        // Shared completion owners survive eviction. No per-draw hot global mutex or compile.
        // A failed profile is cached too, so an honest named gap cannot become a compile loop.
        if (cache.entries.size() >= 16 || cache.bytes > budget - bytes) {
            cache.entries.clear();
            cache.bytes = 0;
        }
        cache.entries.emplace(std::move(key), result);
        cache.bytes += bytes;
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
    if (!in || !producing_owner(*in, prepared) || !launch_owned(*in, prepared) ||
        !resource_free(*in) || !input_free_layout(*in) ||
        *in->raw_code != program->capacity->kernel()->guest_code ||
        prepared.user_sgpr_count != program->user_prefix_count ||
        user_presence(prepared) != program->user_prefix_presence ||
        in->entry.rsrc2 != program->rsrc2 || in->float_mode != program->float_mode ||
        in->float_flags != program->float_flags || in->launch_rsrc1 != program->launch_rsrc1 ||
        in->float_transport != program->transport)
        return refuse("fragment-draw-producing-profile-mismatch");
    if (!executing_device_identity || executing_device_identity != program->device.device_identity)
        return refuse("fragment-draw-executing-device-mismatch");
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
    return FragmentDrawTransaction(std::move(program), std::move(in), std::move(entry_words), width,
                                   height, primitive_count, executing_device_identity);
}
}   // namespace prosper::gpu
