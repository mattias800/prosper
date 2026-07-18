#include "../src/gpu/agc_shader_layout.hpp"
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
    producer.color0_base = 0x2800;
    producer.color0_width = 8;
    producer.color0_height = 8;
    producer.color1_base = 0x2000;
    producer.color1_width = 8;
    producer.color1_height = 8;
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
    ReplayDmaCopy dma;
    dma.src = 0x6000; dma.dst = 0x4000; dma.bytes = 0x40; dma.command_order = 25;
    replay.dma_copies = {dma};
    replay.operations = {
        {SubmitOperationKind::Draw, 0, 10, true},
        {SubmitOperationKind::Dispatch, 2, 20, true},
        {SubmitOperationKind::DmaCopy, 0, 25, true},
        {SubmitOperationKind::Draw, 1, 30, true},
        {SubmitOperationKind::Draw, 9, 40, false},
    };

    GpuDependencyGraph graph;
    std::string error;
    CHECK(build_gpu_dependency_graph(replay, graph, error), "mixed operation graph builds");
    CHECK(graph.nodes.size() == 5 && graph.nodes[2].kind == SubmitOperationKind::DmaCopy &&
          !graph.nodes[4].realized,
          "all semantic operations remain visible, including missing work");
    CHECK(graph.edges.size() == 3 && graph.edges[0].producer_operation == 0 &&
          graph.edges[0].consumer_operation == 3 && graph.edges[1].producer_operation == 1 &&
          graph.edges[1].consumer_operation == 3 && graph.edges[2].producer_operation == 2 &&
          graph.edges[2].consumer_operation == 3,
          "consumer resolves latest overlapping MRT1, compute, and DMA writers");
    CHECK(graph.external_leaves.size() == 3,
          "resources with no earlier in-capsule writer remain explicit external leaves");
    CHECK(graph.external_leaves[0].access.addr == 0x1000 &&
          graph.external_leaves[1].access.addr == 0x3000 &&
          graph.external_leaves[2].access.addr == 0x6000 &&
          graph.external_leaves[0].consumer_operations[0] == 0,
          "external leaf identities preserve operation order");
    CHECK(graph.external_leaves[0].first_future_writer == UINT32_MAX &&
          graph.external_leaves[1].first_future_writer == 1,
          "read-before-write leaves identify the first in-capsule future writer");

    // #636: dependency analysis consumes the realized resource table, so descriptor-looking scalar
    // user data must never appear as a read/write access unless an instruction actually uses it.
    uint32_t scalar_sgprs[32] = {};
    scalar_sgprs[0] = 0x00100000u;
    scalar_sgprs[2] = 16u;
    scalar_sgprs[3] = (2u << 12) | 0xFACu;
    AgcShaderUserData scalar_user_data{};
    AgcShaderHeader scalar_header{};
    scalar_header.file_header = 0x34333231u;
    scalar_header.version = 0x18;
    scalar_header.type = 0;
    scalar_header.user_data = &scalar_user_data;
    ComputeItem scalar_compute;
    scalar_compute.dispatch_index = 636;
    scalar_compute.command_order = 636;
    scalar_compute.resources = std::make_shared<ShaderResourceTable>(
        build_shader_resources(scalar_header, scalar_sgprs, 32));
    GpuReplayFrame scalar_replay;
    scalar_replay.computes = {scalar_compute};
    scalar_replay.operations = {
        {SubmitOperationKind::Dispatch, 636, 636, true},
    };
    GpuDependencyGraph scalar_graph;
    CHECK(build_gpu_dependency_graph(scalar_replay, scalar_graph, error) &&
              scalar_compute.resources->resources.empty() && scalar_graph.edges.empty() &&
              scalar_graph.external_leaves.empty(),
          "#636: dependency graph ignores unconsumed descriptor-looking scalar arguments");

    replay.operations[4].realized = true;
    CHECK(!build_gpu_dependency_graph(replay, graph, error) &&
          error.find("no materialized item") != std::string::npos,
          "graph rejects a falsely realized operation");

    if (fails) return 1;
    std::printf("== PASS ==\n");
    return 0;
}
