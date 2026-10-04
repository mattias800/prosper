#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <string>
#include <vector>

namespace prosper::gpu {
// Exact original-program demands, not a descriptor, guest read permission or backing lease.
// Each origin names an architectural entry SGPR; only a later canonical PS user-prefix
// observation can establish that it is an actually supplied user word rather than a system word.
struct FragmentPacketScalarReadSite {
    uint32_t pc = 0, opcode = 0, words = 0, byte_offset = 0;
    std::array<uint32_t, 4> entry_words{};
    bool operator==(const FragmentPacketScalarReadSite&) const = default;
};
struct FragmentPacketScalarReadRequirements {
    const std::vector<uint32_t>* source_words = nullptr;
    bool has_smem = false;
    std::vector<FragmentPacketScalarReadSite> sites;
    std::string rejection;
    uint64_t retained_bytes() const;
};
// The immutable analysis producer supplies its full original decoded stream, never compacted or
// native32-folded instructions. This initial origin envelope is straight-line SGPR MOV forwarding.
FragmentPacketScalarReadRequirements
fragment_packet_scalar_read_requirements(const std::vector<uint32_t>&,
                                         const std::vector<Rdna2Inst>&);
} // namespace prosper::gpu
