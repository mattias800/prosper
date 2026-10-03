#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/fragment_packet_exports.hpp"
#include "gpu/recompiler/fragment_packet_mask_requirements.hpp"
#include <bitset>
#include <map>
#include <string>
#include <vector>

namespace prosper::gpu {
enum class FragmentPacketVgprRead : uint32_t {
    Direct = 1,
    InterpolationPrevious = 2,
    SelectedPeer = 3,
    RawExport = 4,
};
struct FragmentPacketVgprAccess {
    uint32_t reg = 0;
    FragmentPacketVgprRead kind = FragmentPacketVgprRead::Direct;
};
// Immutable code facts, not launch values or an EXEC/coverage proof. A structural writer only
// establishes a possible definition: runtime per-lane validity still protects every actual read.
struct FragmentPacketVgprRequirements {
    const std::vector<uint32_t>* source_words = nullptr; // pinned by the analysis/capsule owner
    std::bitset<256> storage, possible_entry;
    std::map<uint32_t, std::vector<FragmentPacketVgprAccess>> reads;
    std::string rejection;
    FragmentPacketMaskRequirements masks;
    uint64_t retained_bytes() const;
};
FragmentPacketVgprRequirements fragment_packet_vgpr_requirements(
    const std::vector<uint32_t>& code, const std::vector<Rdna2Inst>& instructions,
    FragmentPacketExportObservation = FragmentPacketExportObservation::LegacyRaw);
FragmentPacketVgprRequirements fragment_packet_vgpr_requirements(const std::vector<uint32_t>& code);
} // namespace prosper::gpu
