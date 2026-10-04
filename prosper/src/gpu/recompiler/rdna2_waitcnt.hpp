#pragma once
#include <cstdint>

namespace prosper::gpu {
// RDNA2 SOPP S_WAITCNT's unsigned SIMM16, not the signed branch displacement stored by
// Rdna2Inst. AMD's RDNA2 ISA section 12.5 and machine-readable OPR_WAITCNT agree on these
// fields. Bit 7 has no meaning in either inspected definition: retain it, do not invent one.
// https://gpuopen.com/download/rdna2-shader-instruction-set-architecture.pdf
// https://gpuopen.com/machine-readable-isa/
struct Rdna2Waitcnt {
    uint8_t vmcnt, expcnt, lgkmcnt;
    uint16_t unmodeled_bits;

    constexpr bool drains_scalar_reads() const { return lgkmcnt == 0; }
    constexpr bool drains_vector_reads() const { return vmcnt == 0; }
    constexpr bool drains_exports() const { return expcnt == 0; }
};

constexpr Rdna2Waitcnt decode_rdna2_waitcnt(uint16_t immediate) {
    return {uint8_t((immediate & 15u) | ((immediate >> 10u) & 48u)),
            uint8_t((immediate >> 4u) & 7u), uint8_t((immediate >> 8u) & 63u),
            uint16_t(immediate & 0x80u)};
}

// Canonical WAIT has no register read/write effects even with a partial counter threshold.
// This fact preserves code-owned origins/MUST masks; it grants NO memory or export completion.
constexpr bool rdna2_waitcnt_effects_known(uint16_t immediate) {
    return decode_rdna2_waitcnt(immediate).unmodeled_bits == 0;
}

// The current owned packet proof has pending sets, not an exact outstanding-count/order model.
// Zero drains its OWN set; the maximum grants no drain. Intermediate positive thresholds remain
// an explicit domain gap, never an excuse to clear all results. No ISA-invalid bit-7 claim.
constexpr const char* rdna2_waitcnt_execution_gap(uint16_t immediate) {
    const auto wait = decode_rdna2_waitcnt(immediate);
    if (wait.unmodeled_bits) return "packet-waitcnt-noncanonical-bits-unimplemented";
    if ((wait.vmcnt != 0 && wait.vmcnt != 63) || (wait.expcnt != 0 && wait.expcnt != 7) ||
        (wait.lgkmcnt != 0 && wait.lgkmcnt != 63))
        return "packet-waitcnt-partial-threshold-unimplemented";
    return nullptr;
}
}   // namespace prosper::gpu
