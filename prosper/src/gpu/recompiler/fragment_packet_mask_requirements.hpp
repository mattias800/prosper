#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <string>
#include <vector>

namespace prosper::gpu {
struct FragmentInvocationPacket;
inline constexpr uint8_t kPacketInitialExec = 1, kPacketInitialVcc = 2, kPacketInitialScc = 4;
// Code facts only. Definitions are instruction-order MUST facts, never allocation, supplied
// values, branch pruning, or permission to fabricate a missing architectural entry word.
struct FragmentPacketMaskRequirements {
    const std::vector<uint32_t>* source_words = nullptr;   // exact immutable original-program owner
    struct Entry {
        uint32_t pc = 0;
        uint8_t defined = 0;   // complete EXEC/VCC pairs and SCC, BEFORE this instruction
    };
    uint8_t demanded = 0;
    std::array<uint32_t, 3> first_read_pc{UINT32_MAX, UINT32_MAX, UINT32_MAX};
    std::vector<Entry> entries;
    std::string rejection;
    bool defined_before(uint32_t pc, uint8_t state) const;
    uint64_t retained_bytes() const;
};
FragmentPacketMaskRequirements fragment_packet_mask_requirements(const std::vector<uint32_t>&,
                                                                 const std::vector<Rdna2Inst>&);
uint8_t fragment_packet_initial_mask_availability(const FragmentInvocationPacket&);
const char* fragment_packet_missing_initial_mask(const FragmentPacketMaskRequirements&,
                                                 uint8_t available, uint32_t& pc);
}   // namespace prosper::gpu
