#include "gpu_dependency_graph.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace prosper::gpu {
namespace {

struct Writer {
    uint32_t operation = 0;
    uint64_t addr = 0;
    uint64_t size = 0;
};

uint64_t span_end(uint64_t addr, uint64_t size) {
    return size > std::numeric_limits<uint64_t>::max() - addr
        ? std::numeric_limits<uint64_t>::max() : addr + size;
}

bool overlaps(uint64_t a_addr, uint64_t a_size, uint64_t b_addr, uint64_t b_size) {
    return a_addr < span_end(b_addr, b_size) && b_addr < span_end(a_addr, a_size);
}

uint64_t resource_size(const ShaderResource& resource) {
    if (resource.host_data_size) return resource.host_data_size;
    if (resource.size) return resource.size;
    if (resource.width && resource.height)
        return static_cast<uint64_t>(resource.width) * resource.height * 4;
    return 0;
}

void append_accesses(const ShaderResourceTable* table, const char* stage,
                     std::vector<GpuDependencyAccess>& accesses) {
    if (!table) return;
    for (const auto& resource : table->resources) {
        const uint64_t size = resource_size(resource);
        if (!resource.gpu_addr || !size || resource.cls == ResourceClass::Sampler) continue;
        accesses.push_back({resource.gpu_addr, size, resource.width, resource.height,
                            resource.binding, resource.cls, stage});
    }
}

} // namespace

bool build_gpu_dependency_graph(const GpuReplayFrame& replay,
                                GpuDependencyGraph& graph, std::string& error) {
    graph = {};
    error.clear();
    std::unordered_map<uint64_t, const DrawItem*> draws;
    std::unordered_map<uint64_t, const ComputeItem*> computes;
    for (const auto& draw : replay.items) draws.emplace(draw.draw_index, &draw);
    for (const auto& compute : replay.computes) computes.emplace(compute.dispatch_index, &compute);

    std::vector<Writer> writers;
    graph.nodes.reserve(replay.operations.size());
    for (size_t operation_index = 0; operation_index < replay.operations.size(); ++operation_index) {
        const auto& operation = replay.operations[operation_index];
        graph.nodes.push_back({static_cast<uint32_t>(operation_index), operation.kind,
                               operation.source_index, operation.command_order, operation.realized});
        if (!operation.realized) continue;

        std::vector<GpuDependencyAccess> reads;
        std::vector<std::pair<uint64_t, uint64_t>> writes;
        if (operation.kind == SubmitOperationKind::Draw) {
            auto it = draws.find(operation.source_index);
            if (it == draws.end()) {
                error = "realized draw operation has no materialized item";
                return false;
            }
            const DrawItem& draw = *it->second;
            append_accesses(draw.vrt.get(), "vs", reads);
            append_accesses(draw.prt.get(), "ps", reads);
            if (draw.color0_base && draw.color0_width && draw.color0_height)
                writes.push_back({draw.color0_base,
                                  static_cast<uint64_t>(draw.color0_width) * draw.color0_height * 4});
        } else {
            auto it = computes.find(operation.source_index);
            if (it == computes.end()) {
                error = "realized dispatch operation has no materialized item";
                return false;
            }
            append_accesses(it->second->resources.get(), "cs", reads);
            for (const auto& access : reads) writes.push_back({access.addr, access.size});
        }

        for (const auto& access : reads) {
            auto producer = std::find_if(writers.rbegin(), writers.rend(), [&](const Writer& writer) {
                return overlaps(access.addr, access.size, writer.addr, writer.size);
            });
            if (producer == writers.rend()) {
                auto leaf = std::find_if(graph.external_leaves.begin(), graph.external_leaves.end(),
                    [&](const GpuDependencyLeaf& candidate) {
                        return candidate.access.addr == access.addr && candidate.access.size == access.size;
                    });
                if (leaf == graph.external_leaves.end())
                    graph.external_leaves.push_back({access, {static_cast<uint32_t>(operation_index)}, UINT32_MAX});
                else if (leaf->consumer_operations.empty() ||
                         leaf->consumer_operations.back() != operation_index)
                    leaf->consumer_operations.push_back(static_cast<uint32_t>(operation_index));
            } else
                graph.edges.push_back({producer->operation, static_cast<uint32_t>(operation_index), access});
        }
        for (const auto& write : writes)
            if (write.first && write.second)
                writers.push_back({static_cast<uint32_t>(operation_index), write.first, write.second});
    }
    for (auto& leaf : graph.external_leaves) {
        const uint32_t first_consumer = leaf.consumer_operations.front();
        auto future = std::find_if(writers.begin(), writers.end(), [&](const Writer& writer) {
            return writer.operation >= first_consumer &&
                   overlaps(leaf.access.addr, leaf.access.size, writer.addr, writer.size);
        });
        if (future != writers.end()) leaf.first_future_writer = future->operation;
    }
    return true;
}

} // namespace prosper::gpu
