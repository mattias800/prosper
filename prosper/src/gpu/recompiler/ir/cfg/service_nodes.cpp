// service_nodes.cpp -- pure selected-service projection and its ordered shadow comparison.
#include "gpu/recompiler/ir/cfg/service_nodes.hpp"
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <string>

namespace prosper::gpu::ir::cfg {

const char* service_kind_name(ServiceKind kind) {
    static constexpr const char* names[] = {
        "mbcnt",
        "append",
        "lds-store",
        "lds-fminmax",
        "swizzle",
        "bpermute",
        "fragment-dpp-min-row-shr",
        "compute-dpp-row-shr",
        "compute-dpp-row-ror8",
        "compute-dpp-quad-or",
        "compute-dpp-add-row-mask",
        "mask-ffbh",
        "mask-reduction",
        "packet-wqm",
        "readlane",
        "readfirstlane",
        "mask-zero-compare",
        "exec-saved-mask-compare",
        "saved-mask-pair-compare",
        "vopc-mask-zero-compare",
        "b64-mask-scc-vote",
    };
    static_assert(std::size(names) == kServiceKindCount);
    const size_t index = static_cast<size_t>(kind);
    return index < std::size(names) ? names[index] : "invalid";
}

ServiceNodes build_service_nodes(const GuestCfg& cfg, std::span<const Rdna2Inst> ins,
                                 std::span<const BlockServiceSelection> selections) {
    ServiceNodes out;
    for (const auto& selected : selections) {
        const uint32_t block = cfg.block_at_pc(selected.block_pc);
        const std::string where = "pc " + std::to_string(selected.event.pc) + " (" +
                                  service_kind_name(selected.event.kind) + ")";
        std::string error;
        if (static_cast<size_t>(selected.event.kind) >= kServiceKindCount)
            error = "invalid kind at " + where;
        else if (block == kNone)
            error = "no CFG block at pc " + std::to_string(selected.block_pc) + " for " + where;
        else if (cfg.blocks[block].last >= ins.size() ||
                 ins[cfg.blocks[block].last].pc != selected.event.pc ||
                 cfg.blocks[block].service_event != selected.event.pc)
            error = "not a CFG service block end: " + where;
        else if (!out.nodes.empty() && selected.dispatch < out.nodes.back().dispatch)
            error = "dispatch order goes backwards at " + where;
        else if (!out.nodes.empty() && selected.dispatch == out.nodes.back().dispatch)
            error =
                "multiple block-ending services in dispatch " + std::to_string(selected.dispatch);
        if (!error.empty() && out.difference.empty()) out.difference = error;
        out.nodes.push_back({block, selected.dispatch, selected.event});
    }
    return out;
}

std::string service_nodes_difference(const ServiceNodes& model,
                                     std::span<const DispatchServiceEvent> dispatcher) {
    if (!model.difference.empty()) return model.difference;
    const size_t common = std::min(model.nodes.size(), dispatcher.size());
    for (size_t i = 0; i < common; ++i) {
        const ServiceNode& node = model.nodes[i];
        const DispatchServiceEvent& own = dispatcher[i];
        if (node.dispatch != own.dispatch || node.event != own.event)
            return "event " + std::to_string(i) + ": CFG dispatch " +
                   std::to_string(node.dispatch) + " pc " + std::to_string(node.event.pc) + " " +
                   service_kind_name(node.event.kind) + ", dispatcher dispatch " +
                   std::to_string(own.dispatch) + " pc " + std::to_string(own.event.pc) + " " +
                   service_kind_name(own.event.kind);
    }
    if (model.nodes.size() != dispatcher.size())
        return "events: CFG " + std::to_string(model.nodes.size()) + ", dispatcher " +
               std::to_string(dispatcher.size());
    return {};
}

}   // namespace prosper::gpu::ir::cfg
