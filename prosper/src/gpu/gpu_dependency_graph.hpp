#pragma once

#include "gpu_capture.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {

struct GpuDependencyAccess {
    uint64_t addr = 0;
    uint64_t size = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t binding = 0;
    ResourceClass resource_class = ResourceClass::ConstantBuffer;
    std::string stage;
};

struct GpuDependencyNode {
    uint32_t operation_index = 0;
    SubmitOperationKind kind = SubmitOperationKind::Draw;
    uint64_t source_index = 0;
    uint64_t command_order = 0;
    bool realized = false;
};

struct GpuDependencyEdge {
    uint32_t producer_operation = 0;
    uint32_t consumer_operation = 0;
    GpuDependencyAccess access;
};

struct GpuDependencyLeaf {
    GpuDependencyAccess access;
    std::vector<uint32_t> consumer_operations;
    uint32_t first_future_writer = UINT32_MAX;
};

struct GpuDependencyGraph {
    std::vector<GpuDependencyNode> nodes;
    std::vector<GpuDependencyEdge> edges;
    std::vector<GpuDependencyLeaf> external_leaves;
};

bool build_gpu_dependency_graph(const GpuReplayFrame& replay,
                                GpuDependencyGraph& graph, std::string& error);

} // namespace prosper::gpu
