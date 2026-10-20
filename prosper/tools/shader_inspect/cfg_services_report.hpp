// cfg_services_report.hpp -- the observe-only selected-service census for shader_inspect.
#pragma once

#include "gpu/recompiler/ir/cfg/service_census.hpp"
#include "gpu/recompiler/ir/cfg/service_nodes.hpp"

#include <array>
#include <cstdio>

namespace prosper::tools {

struct CfgServicesReport {
    gpu::ir::cfg::ServiceCapture capture;

    ~CfgServicesReport() {
        using namespace gpu::ir::cfg;
        size_t equal = 0, complete = 0, nodes = 0, events = 0, pending = 0;
        size_t empty = 0, groups = 0, blocks = 0, splits = 0, auxiliary = 0;
        std::array<size_t, kServiceKindCount> kinds{};
        for (const auto& record : capture.records()) {
            const bool same = record.difference.empty();
            equal += same;
            complete += record.complete;
            nodes += record.model.nodes.size();
            events += record.dispatcher.size();
            pending += record.pending_selections;
            empty += record.model.nodes.empty() && record.dispatcher.empty();
            groups += record.groups;
            blocks += record.blocks;
            splits += record.split_sites;
            auxiliary += record.auxiliary_raw_mask_sites;
            std::printf("cfg-services entry=%u verdict=%s complete=%u blocks=%zu groups=%zu "
                        "splits=%zu nodes=%zu events=%zu pending=%zu auxiliary-raw-masks=%zu",
                        record.entry_pc, same ? "equal" : "differ",
                        static_cast<unsigned>(record.complete), record.blocks, record.groups,
                        record.split_sites, record.model.nodes.size(), record.dispatcher.size(),
                        record.pending_selections, record.auxiliary_raw_mask_sites);
            if (!same) std::printf(" difference=\"%s\"", record.difference.c_str());
            std::printf("\n");
            for (const auto& node : record.model.nodes)
                if (static_cast<size_t>(node.event.kind) < kinds.size())
                    ++kinds[static_cast<size_t>(node.event.kind)];
        }
        for (size_t k = 0; k < kinds.size(); ++k)
            std::printf("cfg-services-kind kind=%s nodes=%zu\n",
                        service_kind_name(static_cast<ServiceKind>(k)), kinds[k]);
        std::printf("cfg-services-end regions=%zu equal=%zu differ=%zu complete=%zu partial=%zu "
                    "empty=%zu groups=%zu blocks=%zu splits=%zu nodes=%zu events=%zu pending=%zu "
                    "auxiliary-raw-masks=%zu\n",
                    capture.records().size(), equal, capture.records().size() - equal, complete,
                    capture.records().size() - complete, empty, groups, blocks, splits, nodes,
                    events, pending, auxiliary);
    }
};

}   // namespace prosper::tools
