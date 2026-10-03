#pragma once
#include "gpu/recompiler/fragment_resource_packet.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
struct PacketF32Result {
    uint32_t bits = 0, valid = 0, exact = 0;
    uint32_t nonfinite = 0, overflow = 0;
};
// Integer-only finite F32 operations. ADD aligns with explicit sticky saturation for ALL finite
// exponent distances; MUL carries the complete 48-bit product. Nonfinite/overflow are statuses.
// This is not a hardware float-control assumption and does not implement exception state/NaNs.
PacketF32Result packet_f32_add(SpirvCompute&, uint32_t a, uint32_t b, FragmentFloatMode);
PacketF32Result packet_f32_mul(SpirvCompute&, uint32_t a, uint32_t b, FragmentFloatMode);
// Correctly rounded integer software implementations within AMD's published 1-ULP bounds,
// not a bit-identical model of its approximation unit. These opcode-specific paths always
// sign-preserving flush denorms, independently of FLOAT_MODE's ordinary denormal controls.
// nonfinite denotes unresolved NaN payload/negative-root semantics, not supported infinities.
PacketF32Result packet_f32_special(SpirvCompute&, uint32_t a, uint32_t opcode, FragmentFloatMode);
bool packet_special_f32_opcode(uint32_t opcode);

struct PacketResourceServices {
    const FragmentResourcePacket& input;
    FragmentResourcePacketProgram& output;
    std::map<uint32_t, uint32_t> buffer_offsets;
    std::map<uint32_t, uint32_t> parameter_offsets;
    uint32_t failure_var = 0, failure_pc_var = 0, m0_var = 0;
    void begin(SpirvCompute&);
    // 0 => existing integer emitter, 1 => handled, -1 => transactional emission refusal.
    int emit(SpirvCompute&, RegState&, const Rdna2Inst&);
    void finish(SpirvCompute&);
    void fail(SpirvCompute&, uint32_t condition, uint32_t pc, FragmentPacketRuntimeFailure);
};
const char* packet_resource_instruction_gap(const Rdna2Inst&);
const char* packet_resource_preflight(const FragmentResourcePacket&, const std::vector<Rdna2Inst>&,
                                     uint32_t& failure_pc);
FragmentPacketProgram recompile_fragment_packet_impl(const FragmentInvocationPacket&,
    RecompileDiagnosticContext, PacketResourceServices*);
} // namespace prosper::gpu
