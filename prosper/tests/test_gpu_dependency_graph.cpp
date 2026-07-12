#include "../src/gpu/gpu_dependency_graph.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

static ShaderResource resource(uint64_t addr, uint64_t bytes, uint32_t binding,
                               ResourceClass cls = ResourceClass::Texture) {
    ShaderResource result;
    result.gpu_addr = addr;
    result.size = static_cast<uint32_t>(bytes);
    result.host_data_size = bytes;
    result.binding = binding;
    result.cls = cls;
    return result;
}

int main() {
    std::printf("== test_gpu_dependency_graph ==\n");
    GpuReplayFrame replay;

    DrawItem producer;
    producer.draw_index = 0;
    producer.command_order = 10;
    producer.color0_base = 0x2000;
    producer.color0_width = 8;
    producer.color0_height = 8;
    producer.prt = std::make_shared<ShaderResourceTable>();
    producer.prt->resources.push_back(resource(0x1000, 0x100, 4));

    ComputeItem compute;
    compute.dispatch_index = 2;
    compute.command_order = 20;
    compute.resources = std::make_shared<ShaderResourceTable>();
    compute.resources->resources.push_back(resource(0x3000, 0x80, 2, ResourceClass::ConstantBuffer));

    DrawItem consumer;
    consumer.draw_index = 1;
    consumer.command_order = 30;
    consumer.color0_base = 0x5000;
    consumer.color0_width = 8;
    consumer.color0_height = 8;
    consumer.prt = std::make_shared<ShaderResourceTable>();
    consumer.prt->resources = {
        resource(0x2000, 0x100, 4),
        resource(0x3040, 0x20, 5, ResourceClass::ConstantBuffer),
        resource(0x4000, 0x40, 6),
    };

    replay.items = {producer, consumer};
    replay.computes = {compute};
    replay.operations = {
        {SubmitOperationKind::Draw, 0, 10, true},
        {SubmitOperationKind::Dispatch, 2, 20, true},
        {SubmitOperationKind::Draw, 1, 30, true},
        {SubmitOperationKind::Draw, 9, 40, false},
    };

    GpuDependencyGraph graph;
    std::string error;
    CHECK(build_gpu_dependency_graph(replay, graph, error), "mixed operation graph builds");
    CHECK(graph.nodes.size() == 4 && !graph.nodes[3].realized,
          "all semantic operations remain visible, including missing work");
    CHECK(graph.edges.size() == 2 && graph.edges[0].producer_operation == 0 &&
          graph.edges[0].consumer_operation == 2 && graph.edges[1].producer_operation == 1 &&
          graph.edges[1].consumer_operation == 2,
          "consumer resolves latest overlapping draw and compute writers");
    CHECK(graph.external_leaves.size() == 3,
          "resources with no earlier in-capsule writer remain explicit external leaves");
    CHECK(graph.external_leaves[0].access.addr == 0x1000 &&
          graph.external_leaves[1].access.addr == 0x3000 &&
          graph.external_leaves[2].access.addr == 0x4000 &&
          graph.external_leaves[0].consumer_operations[0] == 0,
          "external leaf identities preserve operation order");
    CHECK(graph.external_leaves[0].first_future_writer == UINT32_MAX &&
          graph.external_leaves[1].first_future_writer == 1,
          "read-before-write leaves identify the first in-capsule future writer");

    replay.operations[3].realized = true;
    CHECK(!build_gpu_dependency_graph(replay, graph, error) &&
          error.find("no materialized item") != std::string::npos,
          "graph rejects a falsely realized operation");

    if (fails) return 1;
    std::printf("== PASS ==\n");
    return 0;
}
