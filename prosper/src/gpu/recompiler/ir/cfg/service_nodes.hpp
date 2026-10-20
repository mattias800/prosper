// service_nodes.hpp -- step 11 groundwork: selected dispatcher services as pure CFG data.
// Selection remains with the dispatcher. These records neither recognize instructions nor emit
// SPIR-V; the shadow observer projects existing selections onto block ends (see AGENTS.md).
#pragma once

#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace prosper::gpu::ir::cfg {

// The dispatcher's 21 block-ending selections, in the order of its block_* declarations.
// Auxiliary PacketRawMasks publication is not a block-ending selection.
enum class ServiceKind : uint8_t {
    Mbcnt,
    Append,
    LdsStore,
    LdsFminmax,
    Swizzle,
    Bpermute,
    FragmentDppMinRowShr,
    ComputeDppRowShr,
    ComputeDppRowRor8,
    ComputeDppQuadOr,
    ComputeDppAddRowMask,
    MaskFfbh,
    MaskReduction,
    PacketWqm,
    Readlane,
    Readfirstlane,
    MaskZeroCompare,
    ExecSavedMaskCompare,
    SavedMaskPairCompare,
    VopcMaskZeroCompare,
    B64MaskSccVote,
    Count,
};

inline constexpr size_t kServiceKindCount = static_cast<size_t>(ServiceKind::Count);
const char* service_kind_name(ServiceKind kind);

struct ServiceEvent {
    uint32_t pc = 0;
    ServiceKind kind = ServiceKind::Mbcnt;
    bool operator==(const ServiceEvent&) const = default;
};

// An existing per-block selection. dispatch is the old dispatch-group ordinal; block_pc names
// the original block, before any native forward-chain fusion.
struct BlockServiceSelection {
    uint32_t dispatch = 0;
    uint32_t block_pc = 0;
    ServiceEvent event;
};

struct ServiceNode {
    uint32_t block = kNone;
    uint32_t dispatch = 0;
    ServiceEvent event;
    bool operator==(const ServiceNode&) const = default;
};

// The separate oracle: the dispatcher's outer service pointers after its tail assignments.
struct DispatchServiceEvent {
    uint32_t dispatch = 0;
    ServiceEvent event;
    bool operator==(const DispatchServiceEvent&) const = default;
};

struct ServiceNodes {
    std::vector<ServiceNode> nodes;   // observed dispatch-group/block order, not sorted by pc
    std::string difference;   // a selection not at its CFG block's service-ending instruction
};

ServiceNodes build_service_nodes(const GuestCfg& cfg, std::span<const Rdna2Inst> ins,
                                 std::span<const BlockServiceSelection> selections);

// Checks the independent dispatcher event stream in order, including dispatch identity and kind.
// A malformed node projection is also a difference; nothing consumes either answer in emission.
std::string service_nodes_difference(const ServiceNodes& model,
                                     std::span<const DispatchServiceEvent> dispatcher);

}   // namespace prosper::gpu::ir::cfg
