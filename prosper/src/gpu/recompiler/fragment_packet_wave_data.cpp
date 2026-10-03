#include "gpu/recompiler/fragment_packet_services.hpp"
#include "gpu/recompiler/fragment_packet_vgpr_requirements.hpp"

namespace prosper::gpu {
namespace {
constexpr uint32_t max_batch_words = 16 * 1024 * 1024;
bool same_image_shape(const FragmentPacketImageRead& a, const FragmentPacketImageRead& b) {
    if (a.pc != b.pc || a.mips.size() != b.mips.size()) return false;
    for (size_t i = 0; i < a.mips.size(); ++i)
        if (a.mips[i].width != b.mips[i].width || a.mips[i].height != b.mips[i].height)
            return false;
    return true;
}
bool same_image_binding(const FragmentPacketImageRead& a, const FragmentPacketImageRead& b) {
    if (!same_image_shape(a, b) || a.descriptor != b.descriptor || a.sampler != b.sampler)
        return false;
    for (size_t i = 0; i < a.mips.size(); ++i)
        if (a.mips[i].texels != b.mips[i].texels) return false;
    return true;
}
bool nonoverlapping(std::vector<std::pair<uint32_t, uint32_t>> spans) {
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); ++i)
        if (spans[i].first < spans[i - 1].second) return false;
    return true;
}
} // namespace
FragmentPacketKernel recompile_fragment_packet_kernel(const FragmentResourcePacket& prototype,
                                                      RecompileDiagnosticContext diagnostic) {
    FragmentPacketKernel result;
    const auto reject = [&](const std::string& reason) {
        result.program.packet.rejection = reason;
        log_recompile_diagnostic(diagnostic, "fragment-packet-wave-reject", "terminal", "reason=%s",
                                 reason.c_str());
        std::fprintf(stderr, "[fragment-packet-wave-reject] reason=%s\n", reason.c_str());
        return result;
    };
    if (prototype.invocation.guest_code.empty() || prototype.invocation.guest_code.size() > 4096 ||
        prototype.invocation.vgprs.size() > 256 || prototype.invocation.sgprs.size() > 106 ||
        prototype.buffers.size() > 64 || prototype.images.size() > 16 ||
        prototype.parameter_cache.parameters.size() > 16 * 32 * 4)
        return reject("packet-wave-schema-budget");
    uint64_t owned_words = 0;
    for (const auto& buffer : prototype.buffers) owned_words += buffer.words.size();
    for (const auto& image : prototype.images) {
        if (image.mips.size() > 16) return reject("packet-wave-schema-budget");
        for (const auto& mip : image.mips) owned_words += uint64_t(mip.texels.size()) * 4;
    }
    if (owned_words > 1048576) return reject("packet-wave-schema-budget");
    result.guest_code = prototype.invocation.guest_code;
    rdna2_walk(result.guest_code.data(), result.guest_code.size(), result.instructions);
    auto schema = prototype;
    const auto requirements =
        fragment_packet_vgpr_requirements(result.guest_code, result.instructions);
    if (!requirements.rejection.empty()) return reject(requirements.rejection);
    // Storage schema only. Absence selects runtime instrumentation; none of these placeholders
    // is an executable entry value or a promise that a later wave supplies that word.
    std::set<uint32_t> supplied;
    for (auto& column : schema.invocation.vgprs) {
        supplied.insert(column.reg);
        column.available_mask = 0;
    }
    for (uint32_t reg = 0; reg < 256; ++reg)
        if (requirements.storage.test(reg) && !supplied.contains(reg)) {
            FragmentPacketVgpr placeholder;
            placeholder.reg = reg;
            placeholder.available_mask = 0;
            schema.invocation.vgprs.push_back(placeholder);
        }
    auto& program = result.program;
    program.images = schema.images;
    program.device = schema.device;
    program.launch_rsrc1 = schema.launch_rsrc1;
    program.entry_facts = schema.entry_facts;
    program.float_mode = schema.invocation.float_mode;
    program.float_flags = schema.invocation.float_flags;
    result.transport = schema.invocation.float_transport;
    result.topology = schema.invocation.quad_topology;
    PacketResourceServices services{schema, program};
    program.packet =
        recompile_fragment_packet_impl(schema.invocation, diagnostic, &services, &result.layout);
    if (program.packet.spirv.empty()) return result;
    for (const auto& in : result.instructions)
        if (in.fmt == Rdna2Format::SMEM || in.fmt == Rdna2Format::MIMG ||
            in.fmt == Rdna2Format::VINTRP ||
            (in.fmt == Rdna2Format::VOP2 && (in.opcode == 3 || in.opcode == 8)) ||
            (in.fmt == Rdna2Format::VOP1 && packet_special_f32_opcode(in.opcode)))
            program.runtime_failure_pcs.push_back(in.pc);
    // Cache only code-generation shapes, not prototype bytes, descriptors, M0 or input buffers.
    program.packet.input_words.clear();
    program.launch_rsrc1 = {};
    program.entry_facts = {};
    for (auto& image : program.images) {
        image.descriptor = {};
        image.sampler = {};
        for (auto& mip : image.mips) mip.texels.clear();
    }
    return result;
}
FragmentPacketWaveBatch
pack_fragment_packet_waves(std::shared_ptr<const FragmentPacketKernel> owner,
                           std::span<const FragmentResourcePacket> waves,
                           std::span<const FragmentPacketWavePlacement> placements) {
    FragmentPacketWaveBatch result;
    const auto reject = [&](const char* reason) {
        FragmentPacketWaveBatch refused;
        refused.rejection = reason;
        std::fprintf(stderr, "[fragment-packet-wave-reject] reason=%s\n", reason);
        return refused;
    };
    if (!owner || owner->program.packet.spirv.empty() || !owner->program.packet.rejection.empty())
        return reject("packet-wave-kernel-unavailable");
    const auto& kernel = *owner;
    const auto& layout = kernel.layout;
    if (waves.empty() || waves.size() > 4096 ||
        (!placements.empty() && placements.size() != waves.size()))
        return reject("packet-wave-count-invalid");
    if (!layout.input_words || layout.input_words > max_batch_words - 2 ||
        layout.output_words != kernel.program.packet.output_words.size() ||
        layout.output_words > max_batch_words - kPacketWaveOutputPrefix)
        return reject("packet-wave-layout-invalid");
    const uint32_t table_end = 4 + static_cast<uint32_t>(waves.size()) * 2;
    const uint32_t input_span = layout.input_words + 2;
    const uint32_t output_span = layout.output_words + kPacketWaveOutputPrefix;
    uint64_t input_end = table_end, output_end = 0;
    std::vector<std::pair<uint32_t, uint32_t>> input_spans, output_spans;
    for (size_t wave = 0; wave < waves.size(); ++wave) {
        FragmentPacketWavePlacement place;
        if (placements.empty()) {
            if (input_end + input_span > max_batch_words ||
                output_end + output_span > max_batch_words)
                return reject("packet-wave-storage-budget");
            place = {static_cast<uint32_t>(input_end), static_cast<uint32_t>(output_end)};
        } else
            place = placements[wave];
        const uint64_t in_end = uint64_t(place.input_base) + input_span;
        const uint64_t out_end = uint64_t(place.output_base) + output_span;
        if (place.input_base < table_end || in_end > max_batch_words || out_end > max_batch_words)
            return reject("packet-wave-base-range-invalid");
        input_spans.emplace_back(place.input_base, static_cast<uint32_t>(in_end));
        output_spans.emplace_back(place.output_base, static_cast<uint32_t>(out_end));
        input_end = std::max(input_end, in_end);
        output_end = std::max(output_end, out_end);
        result.placements.push_back(place);
    }
    if (!nonoverlapping(input_spans) || !nonoverlapping(output_spans))
        return reject("packet-wave-base-overlap");
    // Bounds are established BEFORE allocation or copying. Guards are observable, not raw EXP.
    result.input_words.resize(static_cast<size_t>(input_end), 0xa11ce55u);
    result.output_words.resize(static_cast<size_t>(output_end), 0xdecafbad);
    result.input_words[0] = kPacketWaveTableMagic;
    result.input_words[1] = static_cast<uint32_t>(waves.size());
    result.input_words[2] = static_cast<uint32_t>(input_end);
    result.input_words[3] = static_cast<uint32_t>(output_end);
    for (size_t wave = 0; wave < waves.size(); ++wave) {
        const auto& input = waves[wave];
        const auto& invocation = input.invocation;
        if (invocation.guest_code != kernel.guest_code)
            return reject("packet-wave-original-code-association-mismatch");
        if (!invocation.mask_state_available ||
            !std::all_of(invocation.slots_available.begin(), invocation.slots_available.end(),
                         [](bool v) { return v; }) ||
            std::any_of(invocation.export_enabled.begin(), invocation.export_enabled.end(),
                        [](uint8_t v) { return v > 1; }))
            return reject("packet-wave-invocation-state-unavailable");
        if (invocation.float_mode != kernel.program.float_mode ||
            invocation.float_flags != kernel.program.float_flags ||
            invocation.float_transport != kernel.transport ||
            invocation.quad_topology != kernel.topology ||
            input.parameter_cache.entry_m0_available != layout.entry_m0_available)
            return reject("packet-wave-code-generation-profile-mismatch");
        if (input.device.device_identity != kernel.program.device.device_identity ||
            input.device.shader_int64_enabled != kernel.program.device.shader_int64_enabled ||
            input.device.rgba32_sfloat_sampled != kernel.program.device.rgba32_sfloat_sampled ||
            !input.entry_facts.canonical())
            return reject("packet-wave-producing-device-or-state-mismatch");
        uint32_t failure_pc = UINT32_MAX;
        if (const auto* gap = packet_resource_preflight(input, kernel.instructions, failure_pc))
            return reject(gap);
        std::map<uint32_t, const FragmentPacketVgpr*> columns;
        for (const auto& column : invocation.vgprs)
            if (!columns.emplace(column.reg, &column).second ||
                std::find(layout.vgprs.begin(), layout.vgprs.end(), column.reg) ==
                    layout.vgprs.end())
                return reject("packet-wave-vgpr-schema-invalid");
        std::map<uint32_t, uint32_t> scalars;
        for (const auto& [reg, value] : invocation.sgprs)
            if (!scalars.emplace(reg, value).second)
                return reject("packet-wave-sgpr-schema-invalid");
        if (scalars.size() != layout.sgprs.size())
            return reject("packet-wave-sgpr-presence-unavailable");
        for (const auto reg : layout.sgprs)
            if (!scalars.contains(reg)) return reject("packet-wave-sgpr-presence-unavailable");
        const auto place = result.placements[wave];
        result.input_words[4 + wave * 2] = place.input_base;
        result.input_words[5 + wave * 2] = place.output_base;
        result.input_words[place.input_base] = kPacketWaveInputMagic;
        result.input_words[place.input_base + 1] = static_cast<uint32_t>(wave);
        const auto base = place.input_base + 2;
        std::fill_n(result.input_words.begin() + base, layout.input_words,
                    0);   // placeholders have availability0
        std::fill_n(result.output_words.begin() + place.output_base, output_span, 0);
        const uint32_t n = static_cast<uint32_t>(layout.vgprs.size());
        const auto stride = kernel.program.packet.input_stride;
        const bool validity = stride == 2 * n + 4;
        for (uint32_t lane = 0; lane < 64; ++lane) {
            for (uint32_t column = 0; column < n; ++column) {
                const auto found = columns.find(layout.vgprs[column]);
                if (found != columns.end()) {
                    result.input_words[base + lane * stride + column] = found->second->words[lane];
                    if (validity)
                        result.input_words[base + lane * stride + n + column] =
                            (found->second->available_mask >> lane) & 1u;
                }
            }
            const auto state = base + lane * stride + n * (validity ? 2u : 1u);
            result.input_words[state] = (invocation.exec_mask >> lane) & 1u;
            result.input_words[state + 1] = (invocation.vcc_mask >> lane) & 1u;
            result.input_words[state + 2] = invocation.scc;
            result.input_words[state + 3] = invocation.export_enabled[lane];
        }
        for (size_t scalar = 0; scalar < layout.sgprs.size(); ++scalar) {
            result.input_words[base + layout.scalar_offsets[scalar]] =
                scalars.at(layout.sgprs[scalar]);
            result.input_words[base + layout.scalar_available_offsets[scalar]] = 1;
        }
        result.input_words[base + layout.expected_m0] = input.parameter_cache.m0;
        result.input_words[base + layout.entry_m0] = input.parameter_cache.entry_m0;
        result.input_words[base + layout.entry_m0_available_offset] =
            input.parameter_cache.entry_m0_available ? 1 : 0;
        for (const auto& buffer : layout.buffers) {
            const auto found = std::find_if(input.buffers.begin(), input.buffers.end(),
                                            [&](const auto& b) { return b.pc == buffer.pc; });
            if (found == input.buffers.end() || found->words.size() != buffer.words)
                return reject("packet-wave-buffer-shape-mismatch");
            std::copy(found->words.begin(), found->words.end(),
                      result.input_words.begin() + base + buffer.offset);
            std::copy(found->descriptor.begin(), found->descriptor.end(),
                      result.input_words.begin() + base + buffer.descriptor);
        }
        for (const auto& parameter : layout.parameters)
            for (uint32_t lane = 0; lane < 64; ++lane) {
                const auto primitive = input.parameter_cache.quad_primitive[lane / 4];
                const auto found =
                    std::find_if(input.parameter_cache.parameters.begin(),
                                 input.parameter_cache.parameters.end(), [&](const auto& p) {
                                     return p.primitive == primitive &&
                                            p.attribute == parameter.attribute &&
                                            p.channel == parameter.channel;
                                 });
                if (found == input.parameter_cache.parameters.end())
                    return reject("packet-parameter-tuple-unavailable");
                const auto offset = base + parameter.offset + lane * 3;
                result.input_words[offset] = found->p0;
                result.input_words[offset + 1] = found->p10;
                result.input_words[offset + 2] = found->p20;
            }
        if (input.images.size() != kernel.program.images.size())
            return reject("packet-wave-image-shape-mismatch");
        for (size_t slot = 0; slot < kernel.program.images.size(); ++slot) {
            const auto& shape = kernel.program.images[slot];
            const auto found =
                std::find_if(input.images.begin(), input.images.end(),
                             [&](const auto& image) { return image.pc == shape.pc; });
            if (found == input.images.end() || !same_image_shape(*found, shape))
                return reject("packet-wave-image-shape-mismatch");
            if (!wave)
                result.images.push_back(*found);
            else if (!same_image_binding(*found, result.images[slot]))
                return reject("packet-wave-independent-image-binding-unimplemented");
            std::copy(found->descriptor.begin(), found->descriptor.end(),
                      result.input_words.begin() + base + layout.image_descriptor_offsets[slot]);
            std::copy(found->sampler.begin(), found->sampler.end(),
                      result.input_words.begin() + base + layout.sampler_offsets[slot]);
        }
    }
    result.device_identity = kernel.program.device.device_identity;
    result.kernel = std::move(owner);
    return result;
}
FragmentPacketWaveResult decode_fragment_packet_waves(const FragmentPacketWaveBatch& batch,
                                                      std::span<const uint32_t> words,
                                                      bool completed, uint64_t device) {
    FragmentPacketWaveResult result;
    const auto reject = [&](const char* reason) {
        result.exports.clear();
        result.rejection = reason;
        std::fprintf(stderr, "[fragment-packet-wave-reject] reason=%s wave=%u lane=%u pc=%u\n",
                     reason, result.wave, result.lane, result.pc);
        return result;
    };
    if (!batch.kernel || !batch.rejection.empty() || batch.placements.empty() ||
        words.size() != batch.output_words.size())
        return reject("packet-wave-batch-abi-invalid");
    if (!completed || device != batch.device_identity)
        return reject("packet-wave-completion-or-device-unproved");
    const auto& kernel = *batch.kernel;
    std::vector<bool> owned(words.size());
    std::vector<std::vector<uint32_t>> validated;
    for (uint32_t wave = 0; wave < batch.placements.size(); ++wave) {
        result.wave = wave;
        const auto base = batch.placements[wave].output_base;
        const auto count = uint64_t(kernel.layout.output_words) + kPacketWaveOutputPrefix;
        if (uint64_t(base) + count > words.size()) return reject("packet-wave-batch-abi-invalid");
        for (uint32_t offset = 0; offset < count; ++offset) {
            if (owned[base + offset]) return reject("packet-wave-batch-abi-invalid");
            owned[base + offset] = true;
        }
        for (uint32_t lane = 0; lane < 64; ++lane)
            if (words[base + lane * 2] != kPacketWaveOutputMagic ||
                words[base + lane * 2 + 1] != wave)
                return reject("packet-wave-metadata-or-completion-invalid");
        const auto decoded = decode_fragment_resource_packet(
            kernel.program,
            words.subspan(base + kPacketWaveOutputPrefix, kernel.layout.output_words), completed,
            device);
        if (!decoded.rejection.empty()) {
            result.lane = decoded.lane;
            result.pc = decoded.pc;
            return reject(decoded.rejection.c_str());
        }
        validated.push_back(decoded.exports);
    }
    for (size_t offset = 0; offset < words.size(); ++offset)
        if (!owned[offset] && words[offset] != batch.output_words[offset])
            return reject("packet-wave-output-guard-corrupted");
    result.wave = UINT32_MAX;
    result.exports =
        std::move(validated);   // publish ONLY after all statuses and holes are validated
    return result;
}
}   // namespace prosper::gpu
