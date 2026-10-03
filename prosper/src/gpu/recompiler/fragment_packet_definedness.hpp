#pragma once
#include "gpu/recompiler/fragment_packet_vgpr_requirements.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
inline constexpr const char* kPacketVgprValidityMarker = "Prosper.GuestFragmentPacket.VgprValidity=VGP1";
// Packet-only validity, independent of RegState's allocated/placeholder value bank. Function
// variables persist each physical worker's LOGICAL lane fact across dispatcher branches/services.
struct PacketVgprDefinedness {
    const FragmentPacketVgprRequirements& requirements;
    std::map<int, uint32_t> valid_vars;
    uint32_t failure_pc = 0, failure_reg = 0, failure_kind = 0;
    uint32_t peer_valid = 0, peer_pc = 0, peer_reg = 0, peer_scratch_base = 0;
    void begin(SpirvCompute&, const std::map<int, uint32_t>& columns);
    void instruction(SpirvCompute&, const RegState&, const Rdna2Inst&);
    void publish_peer(SpirvCompute&, uint32_t pending);
    void consume_peer(SpirvCompute&, uint32_t pending, uint32_t index);
    void finish(SpirvCompute&, uint32_t status_offset);
    void fail(SpirvCompute&, uint32_t condition, uint32_t pc, uint32_t reg, uint32_t kind);
};
// Extracted common phase, shared by legacy compute and packet routes. Null definedness preserves
// the original instruction/word stream; the packet extension adds metadata beside existing values.
bool emit_cfg_readlane_phase(SpirvCompute&, PacketVgprDefinedness*, uint32_t pending_var,
    uint32_t source_var, uint32_t selector_var, uint32_t destination_var,
    const std::set<int>& destinations, const std::map<int, uint32_t>& scalar_vars,
    const std::map<int, uint32_t>& mask_vars, const std::map<int, uint32_t>& mask_half_vars,
    uint32_t vcc_var);
} // namespace prosper::gpu
