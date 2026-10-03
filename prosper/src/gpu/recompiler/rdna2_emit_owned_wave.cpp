// Owned logical-wave memory and event helpers, shared by the existing ALU/CFG emitters.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
bool emit_owned_raw_window(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, uint32_t n,
                           bool& ok) {
    const auto packet_window = b.packet_raw_windows.find(in.pc);
    if (packet_window != b.packet_raw_windows.end()) {
        const auto& [certificate, first_word] = packet_window->second;
        const auto offset = rs.sreg.find(int(certificate.offset_sgpr));
        if (b.guest_wave_event_domain != GuestWaveEventDomain::Owned64 ||
            !b.has_workgroup_execution() || !b.raw_input_words || b.wave_size != 64u ||
            b.local_count != 64u || offset == rs.sreg.end() || certificate.bytes != n * 4u ||
            in.src[0].value != int(certificate.base_sgpr) ||
            in.src[1].value != int(certificate.offset_sgpr) ||
            int32_t(in.literal) != certificate.immediate || in.dst.value < 0 ||
            in.dst.value + int(n) > 106) {
            ok = false;
            return true;
        }
        // The full-code certificate and complete owned interval were admitted before
        // emission. Keep the runtime selector; never fold it, repair a pointer, clip the
        // bound, issue a robust-OOB access, or reinterpret this as descriptor words.
        const uint32_t relative = b.ibin(Op_ISub, offset->second, b.uconst(certificate.offset_min));
        const uint32_t index = b.ibin(Op_IAdd, b.uconst(first_word),
                                      b.ibin(Op_ShiftRightLogical, relative, b.uconst(2u)));
        for (uint32_t word = 0; word < n; ++word) {
            const int destination = in.dst.value + int(word);
            rs.sreg[destination] =
                b.load_owned_packet_word(word ? b.ibin(Op_IAdd, index, b.uconst(word)) : index);
            rs.sreg_srt.erase(destination);
            rs.sreg_bool.erase(destination);
            rs.sreg_bool_narrowed.erase(destination);
            rs.sreg_bool_b32.erase(destination);
        }
        return true; // scalar loads and destinations ignore EXEC; SCC is unchanged
    }
    return false;
}
bool owned_or_portable_readfirstlane_enabled(const SpirvCompute& b) {
    return b.portable_readfirstlane_shader ||
           b.guest_wave_event_domain == GuestWaveEventDomain::Owned64;
}
bool owned_or_portable_readfirstlane_candidate(const SpirvCompute& b, const Rdna2Inst& in) {
    const bool owned = b.guest_wave_event_domain == GuestWaveEventDomain::Owned64 &&
                       b.has_workgroup_execution() && b.local_count == 64;
    return (owned || (b.portable_readfirstlane_shader && b.is_compute)) && b.wave_size == 64 &&
           !b.native_subgroup_size && b.local_count > 0 && in.fmt == Rdna2Format::VOP1 &&
           in.opcode == 0x02 && in.src[0].kind == OperandKind::VGPR && in.dst.value >= 0 &&
           in.dst.value <= (owned ? 105 : 107) && !in.has_sdwa && !in.has_dpp && !in.src_abs[0] &&
           !in.src_neg[0] && !in.clamp && !in.omod;
}
} // namespace prosper::gpu
