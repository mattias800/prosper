#include "gpu/recompiler/fragment_packet_services.hpp"

namespace prosper::gpu {
FragmentResourcePacketProgram recompile_fragment_resource_packet(const FragmentResourcePacket& input,
    RecompileDiagnosticContext diagnostic) {
    FragmentResourcePacketProgram result;
    result.images = input.images;
    result.device = input.device;
    result.launch_rsrc1 = input.launch_rsrc1;
    result.entry_facts = input.entry_facts;
    result.float_mode = input.invocation.float_mode;
    result.float_flags = input.invocation.float_flags;
    PacketResourceServices services{input, result};
    result.packet = recompile_fragment_packet_impl(input.invocation, diagnostic, &services);
    if (result.packet.spirv.empty()) { result.images.clear(); result.status_offset = 0; }
    else {
        std::vector<Rdna2Inst> ins;
        rdna2_walk(input.invocation.guest_code.data(), input.invocation.guest_code.size(), ins);
        for (const auto& in : ins)
            if (in.fmt == Rdna2Format::SMEM || in.fmt == Rdna2Format::MIMG ||
                in.fmt == Rdna2Format::VINTRP ||
                (in.fmt == Rdna2Format::VOP2 && (in.opcode == 3 || in.opcode == 8)) ||
                (in.fmt == Rdna2Format::VOP1 && packet_special_f32_opcode(in.opcode)))
                result.runtime_failure_pcs.push_back(in.pc);
    }
    return result;
}

FragmentResourcePacketResult decode_fragment_resource_packet(const FragmentResourcePacketProgram& program,
    std::span<const uint32_t> words, bool completed, uint64_t executing_device) {
    FragmentResourcePacketResult result;
    const auto reject = [&](const char* reason) {
        result.rejection = reason;
        std::fprintf(stderr, "[fragment-resource-packet-reject] reason=%s lane=%u pc=%u failure=%u\n",
            reason, result.lane, result.pc, static_cast<uint32_t>(result.failure));
        return result;
    };
    if (!completed) return reject("packet-completion-or-host-availability-unproved");
    if (!program.device.device_identity || executing_device != program.device.device_identity ||
        !program.device.shader_int64_enabled || (!program.images.empty() && !program.device.rgba32_sfloat_sampled))
        return reject("packet-executing-enabled-device-mismatch");
    const uint64_t export_count = uint64_t(program.packet.exports_per_lane) * 64 * kFragmentPacketExportWords;
    if (program.packet.spirv.empty() || !program.packet.rejection.empty() ||
        !program.packet.exports_per_lane || program.packet.exports_per_lane > 64 ||
        program.status_offset != export_count || words.size() != export_count + 64 * kFragmentResourceStatusWords ||
        program.packet.output_words.size() != words.size()) return reject("packet-status-abi-invalid");
    // Validate the ENTIRE status array before returning even one record. Deterministic lowest
    // failing logical lane, then its sticky first guest-PC/reason. A later success cannot launder it.
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto offset = program.status_offset + lane * kFragmentResourceStatusWords;
        const auto pc = words[offset + 1], reason = words[offset + 2];
        if (words[offset] != kFragmentResourceStatusMagic ||
            reason >
                static_cast<uint32_t>(FragmentPacketRuntimeFailure::SpecialNanOrNegativeRoot) ||
            ((reason == 0) != (pc == UINT32_MAX)) ||
            (reason &&
             std::find(program.runtime_failure_pcs.begin(), program.runtime_failure_pcs.end(),
                       pc) == program.runtime_failure_pcs.end()))
            return reject("packet-status-record-invalid");
        if (reason && result.failure == FragmentPacketRuntimeFailure::None) {
            result.lane = lane; result.pc = pc; result.failure = static_cast<FragmentPacketRuntimeFailure>(reason);
        }
    }
    if (result.failure != FragmentPacketRuntimeFailure::None) {
        switch (result.failure) {
            case FragmentPacketRuntimeFailure::DescriptorMismatch: return reject("packet-runtime-descriptor-mismatch");
            case FragmentPacketRuntimeFailure::M0Mismatch: return reject("packet-runtime-m0-mismatch");
            case FragmentPacketRuntimeFailure::NonFinite: return reject("packet-runtime-nonfinite-f32-unimplemented");
            case FragmentPacketRuntimeFailure::FiniteOverflow: return reject("packet-runtime-f32-overflow-unimplemented");
            case FragmentPacketRuntimeFailure::InterpolationNotExact: return reject("packet-runtime-interpolation-exactness-unproved");
            case FragmentPacketRuntimeFailure::SampleCoordinateDomain: return reject("packet-runtime-sample-coordinate-domain-unimplemented");
            case FragmentPacketRuntimeFailure::SampleLodDomain: return reject("packet-runtime-sample-lod-domain-unimplemented");
            case FragmentPacketRuntimeFailure::SpecialNanOrNegativeRoot:
                return reject("packet-runtime-special-f32-nan-or-negative-root-unimplemented");
            default: return reject("packet-status-record-invalid");
        }
    }
    for (uint32_t offset = 0; offset < program.status_offset; offset += kFragmentPacketExportWords) {
        if (!words[offset]) {
            for (uint32_t c = 1; c < kFragmentPacketExportWords; ++c)
                if (words[offset + c]) return reject("packet-unreached-export-invalid");
            continue;
        }
        if (words[offset] != 1 || words[offset + 1] > 1 || words[offset + 2] > 1 ||
            words[offset + 4] > 15 || words[offset + 5] || words[offset + 6] > 1 || words[offset + 7] > 1 ||
            (words[offset + 3] >= kFragmentColorOutputs && words[offset + 3] != 8 && words[offset + 3] != 9))
            return reject("packet-export-record-invalid");
        if ((words[offset + 3] == 8 && (!words[offset + 4] || (words[offset + 4] & ~5u))) ||
            (words[offset + 3] == 9 && words[offset + 4])) return reject("packet-export-record-invalid");
        for (uint32_t c = 0; c < 4; ++c)
            if (!(words[offset + 4] & (1u << c)) && words[offset + 8 + c])
                return reject("packet-disabled-export-payload-invalid");
    }
    result.exports.assign(words.begin(), words.begin() + program.status_offset);
    return result;
}
} // namespace prosper::gpu
