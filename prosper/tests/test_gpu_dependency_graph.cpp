#include "../src/gpu/agc_shader_layout.hpp"
#include "../src/gpu/gpu_dependency_graph.hpp"
#include "../src/gpu/vk_translate.hpp"

#include <cstdio>
#include <initializer_list>
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

static void emit(std::vector<uint32_t>& spv, uint16_t op,
                 std::initializer_list<uint32_t> words) {
    spv.push_back((static_cast<uint32_t>(words.size() + 1) << 16) | op);
    spv.insert(spv.end(), words.begin(), words.end());
}

static std::vector<uint32_t> storage_image_copy_spirv() {
    // Binding 5 is read-only and binding 6 is independently write-only.
    std::vector<uint32_t> spv = {0x07230203u, 0x00010000u, 0, 24, 0};
    emit(spv, 15, {5, 20, 0x6e69616d, 0});
    emit(spv, 21, {1, 32, 0});
    emit(spv, 25, {2, 1, 1, 0, 0, 0, 2, 0});
    emit(spv, 32, {3, 0, 2});
    emit(spv, 59, {3, 4, 0}); emit(spv, 59, {3, 5, 0});
    emit(spv, 71, {4, 34, 0}); emit(spv, 71, {4, 33, 5});
    emit(spv, 71, {5, 34, 0}); emit(spv, 71, {5, 33, 6});
    emit(spv, 61, {2, 6, 4}); emit(spv, 61, {2, 7, 5});
    emit(spv, 98, {8, 9, 6, 10});
    emit(spv, 99, {7, 10, 9});
    return spv;
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
    producer.ps.color_write_mask = 0xf;
    producer.ps.color1_write_mask = 0xf;
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

    ComputeItem reflected_compute;
    reflected_compute.dispatch_index = 637;
    reflected_compute.command_order = 637;
    reflected_compute.spirv = storage_image_copy_spirv();
    reflected_compute.resources = std::make_shared<ShaderResourceTable>();
    ShaderResource reflected_src = resource(0x7000, 4, 5, ResourceClass::StorageImage);
    reflected_src.width = reflected_src.height = 1;
    reflected_src.format = DataFormat::Unorm8;
    reflected_src.num_components = 4;
    ShaderResource reflected_dst = reflected_src;
    reflected_dst.binding = 6;
    reflected_dst.gpu_addr = 0x8000;
    reflected_compute.resources->resources = {reflected_src, reflected_dst};
    GpuReplayFrame reflected_replay;
    reflected_replay.computes = {reflected_compute};
    reflected_replay.operations = {
        {SubmitOperationKind::Dispatch, 637, 637, true},
    };
    GpuDependencyGraph reflected_graph;
    CHECK(build_gpu_dependency_graph(reflected_replay, reflected_graph, error) &&
              reflected_graph.external_leaves.size() == 1 &&
              reflected_graph.external_leaves[0].access.addr == reflected_src.gpu_addr &&
              reflected_graph.external_leaves[0].access.format == DataFormat::Unorm8 &&
              reflected_graph.external_leaves[0].access.num_components == 4 &&
              reflected_graph.external_leaves[0].first_future_writer == UINT32_MAX,
          "reflected compute graph keeps the read-only input and does not invent a temporal "
          "read of the write-only output");
    GpuCaptureRttSeed exact_seed;
    exact_seed.guest_addr = reflected_src.gpu_addr;
    exact_seed.width = exact_seed.height = 1;
    exact_seed.format = GpuCaptureColorFormat::Rgba8Unorm;
    exact_seed.rgba.resize(4);
    CHECK(gpu_dependency_rtt_seed_matches(reflected_graph.external_leaves[0].access, exact_seed),
          "an exact live RTT address, extent, and format closes an image dependency");
    GpuCaptureRttSeed wrong_extent = exact_seed;
    wrong_extent.width = 2;
    GpuCaptureRttSeed wrong_format = exact_seed;
    wrong_format.format = GpuCaptureColorFormat::Rgba16Float;
    GpuDependencyAccess srgb_access = reflected_graph.external_leaves[0].access;
    srgb_access.srgb = true;
    CHECK(!gpu_dependency_rtt_seed_matches(reflected_graph.external_leaves[0].access,
                                           wrong_extent) &&
              !gpu_dependency_rtt_seed_matches(reflected_graph.external_leaves[0].access,
                                               wrong_format) &&
              !gpu_dependency_rtt_seed_matches(srgb_access, exact_seed),
          "address-only RTT matches cannot hide extent, format, or sRGB reinterpretation");

    GpuReplayFrame image_overlap_replay;
    DrawItem image_writer;
    image_writer.draw_index = 70;
    image_writer.command_order = 70;
    image_writer.color0_base = 0x9000;
    image_writer.color0_width = image_writer.color0_height = 8;
    image_writer.ps.color_write_mask = 0xf;
    DrawItem interior_reader;
    interior_reader.draw_index = 71;
    interior_reader.command_order = 71;
    interior_reader.prt = std::make_shared<ShaderResourceTable>();
    ShaderResource interior = resource(0x9040, 4, 4, ResourceClass::Texture);
    interior.width = interior.height = 1;
    interior_reader.prt->resources = {interior};
    image_overlap_replay.items = {image_writer, interior_reader};
    image_overlap_replay.operations = {
        {SubmitOperationKind::Draw, 70, 70, true},
        {SubmitOperationKind::Draw, 71, 71, true},
    };
    GpuDependencyGraph image_overlap_graph;
    CHECK(build_gpu_dependency_graph(image_overlap_replay, image_overlap_graph, error) &&
              image_overlap_graph.edges.empty() &&
              image_overlap_graph.external_leaves.size() == 1 &&
              image_overlap_graph.external_leaves[0].first_future_writer == UINT32_MAX,
          "an interior byte overlap does not invent an exact-base RTT image dependency");

    GpuReplayFrame masked_target_replay;
    DrawItem masked_writer = image_writer;
    masked_writer.draw_index = 72;
    masked_writer.command_order = 72;
    masked_writer.ps.color_write_mask = 0;
    DrawItem masked_reader = interior_reader;
    masked_reader.draw_index = 73;
    masked_reader.command_order = 73;
    masked_reader.prt = std::make_shared<ShaderResourceTable>(*interior_reader.prt);
    masked_reader.prt->resources[0].gpu_addr = masked_writer.color0_base;
    masked_target_replay.items = {masked_writer, masked_reader};
    masked_target_replay.operations = {
        {SubmitOperationKind::Draw, 72, 72, true},
        {SubmitOperationKind::Draw, 73, 73, true},
    };
    GpuDependencyGraph masked_target_graph;
    CHECK(build_gpu_dependency_graph(masked_target_replay, masked_target_graph, error) &&
              masked_target_graph.edges.empty() &&
              masked_target_graph.external_leaves.size() == 1 &&
              masked_target_graph.external_leaves[0].access.addr == masked_writer.color0_base,
          "a programmed color target with a zero write mask is not a dependency producer");

    // A fixed-function MODE=RESOLVE consumes color0 and writes color1 even though color1 has no
    // shader export/write mask. Asterix's final composite samples that destination in the same
    // submit; treating the resolve as an ordinary draw made the graph call it an external leaf.
    DrawItem resolve_source;
    resolve_source.draw_index = 80;
    resolve_source.command_order = 80;
    resolve_source.color0_base = 0xa000;
    resolve_source.color0_width = resolve_source.color0_height = 8;
    resolve_source.ps.color_write_mask = 0xf;
    DrawItem resolve;
    resolve.draw_index = 81;
    resolve.command_order = 81;
    resolve.color0_base = resolve_source.color0_base;
    resolve.color0_width = resolve.color0_height = 8;
    resolve.color1_base = 0xb000;
    resolve.color1_width = resolve.color1_height = 8;
    resolve.ps.cb_resolve = true;
    resolve.ps.color0_format = static_cast<uint32_t>(VkFormat::R16G16B16A16_SFLOAT);
    resolve.ps.color_write_mask = 0xf;
    resolve.ps.color1_write_mask = 0;
    resolve.prt = std::make_shared<ShaderResourceTable>();
    resolve.prt->resources.push_back(resource(0xc000, 0x100, 34));
    DrawItem resolve_consumer;
    resolve_consumer.draw_index = 82;
    resolve_consumer.command_order = 82;
    resolve_consumer.prt = std::make_shared<ShaderResourceTable>();
    resolve_consumer.prt->resources.push_back(resource(resolve.color1_base, 0x100, 34));
    resolve_consumer.prt->resources[0].width = 8;
    resolve_consumer.prt->resources[0].height = 8;
    GpuReplayFrame resolve_replay;
    resolve_replay.items = {resolve_source, resolve, resolve_consumer};
    resolve_replay.operations = {
        {SubmitOperationKind::Draw, 80, 80, true},
        {SubmitOperationKind::Draw, 81, 81, true},
        {SubmitOperationKind::Draw, 82, 82, true},
    };
    GpuDependencyGraph resolve_graph;
    CHECK(build_gpu_dependency_graph(resolve_replay, resolve_graph, error) &&
              resolve_graph.edges.size() == 2 &&
              resolve_graph.edges[0].producer_operation == 0 &&
              resolve_graph.edges[0].consumer_operation == 1 &&
              resolve_graph.edges[0].access.addr == resolve.color0_base &&
              resolve_graph.edges[0].access.stage == "resolve-src" &&
              resolve_graph.edges[1].producer_operation == 1 &&
              resolve_graph.edges[1].consumer_operation == 2 &&
              resolve_graph.edges[1].access.addr == resolve.color1_base &&
              resolve_graph.external_leaves.empty(),
          "fixed-function resolve reads color0, writes color1, and ignores shader resources");

    GpuReplayFrame external_resolve_replay;
    external_resolve_replay.items = {resolve};
    external_resolve_replay.operations = {
        {SubmitOperationKind::Draw, 81, 81, true},
    };
    GpuDependencyGraph external_resolve_graph;
    GpuCaptureRttSeed resolve_seed;
    resolve_seed.guest_addr = resolve.color0_base;
    resolve_seed.width = resolve.color0_width;
    resolve_seed.height = resolve.color0_height;
    resolve_seed.format = GpuCaptureColorFormat::Rgba16Float;
    CHECK(build_gpu_dependency_graph(external_resolve_replay, external_resolve_graph, error) &&
              external_resolve_graph.external_leaves.size() == 1 &&
              external_resolve_graph.external_leaves[0].access.addr == resolve.color0_base &&
              external_resolve_graph.external_leaves[0].access.size == 8u * 8u * 8u &&
              external_resolve_graph.external_leaves[0].access.format == DataFormat::Float16 &&
              external_resolve_graph.external_leaves[0].access.num_components == 4 &&
              gpu_dependency_rtt_seed_matches(
                  external_resolve_graph.external_leaves[0].access, resolve_seed),
          "an external resolve source retains exact extent and format for RTT seed closure");

    GpuReplayFrame ordinary_replay = resolve_replay;
    ordinary_replay.items[1].ps.cb_resolve = false;
    GpuDependencyGraph ordinary_graph;
    CHECK(build_gpu_dependency_graph(ordinary_replay, ordinary_graph, error) &&
              ordinary_graph.external_leaves.size() == 2 &&
              ordinary_graph.external_leaves[0].access.addr == 0xc000 &&
              ordinary_graph.external_leaves[1].access.addr == resolve.color1_base,
          "the same zero-MRT1 state without MODE=RESOLVE does not invent a destination write");

    DrawItem table_producer;
    table_producer.draw_index = 90;
    table_producer.command_order = 90;
    table_producer.color0_base = 0xd000;
    table_producer.color0_width = table_producer.color0_height = 1;
    table_producer.ps.color_write_mask = 0xf;
    DrawItem table_consumer;
    table_consumer.draw_index = 91;
    table_consumer.command_order = 91;
    table_consumer.vrt = std::make_shared<ShaderResourceTable>();
    ShaderResource descriptor_array;
    descriptor_array.cls = ResourceClass::ConstantBuffer;
    descriptor_array.format = DataFormat::Uint32;
    descriptor_array.num_components = 1;
    descriptor_array.binding = 2;
    descriptor_array.table_index_count = 2;
    descriptor_array.table_entry_stride = 120;
    descriptor_array.table_selector_mode =
        BufferTableSelectorMode::DynamicSbufferByteOffset;
    descriptor_array.table_load_pc = 202;
    for (uint64_t address : {0xd000ull, 0xe000ull}) {
        ShaderBufferTableEntry entry;
        entry.gpu_addr = address;
        entry.size = 4;
        entry.stride = 4;
        entry.vsharp = {
            static_cast<uint32_t>(address),
            static_cast<uint32_t>(address >> 32u) | (entry.stride << 16u),
            1u,
            0x00014204u,
        };
        descriptor_array.table_entries.push_back(entry);
    }
    table_consumer.vrt->resources.push_back(descriptor_array);
    GpuReplayFrame table_replay;
    table_replay.items = {table_producer, table_consumer};
    table_replay.operations = {
        {SubmitOperationKind::Draw, 90, 90, true},
        {SubmitOperationKind::Draw, 91, 91, true},
    };
    GpuDependencyGraph table_graph;
    CHECK(build_gpu_dependency_graph(table_replay, table_graph, error) &&
              table_graph.edges.size() == 1 &&
              table_graph.edges[0].producer_operation == 0 &&
              table_graph.edges[0].consumer_operation == 1 &&
              table_graph.edges[0].access.addr == 0xd000 &&
              table_graph.external_leaves.size() == 1 &&
              table_graph.external_leaves[0].access.addr == 0xe000,
          "dependency closure enumerates every concrete descriptor-array backing");

    GpuReplayFrame short_table_replay = table_replay;
    short_table_replay.items[1].vrt = std::make_shared<ShaderResourceTable>(
        *table_replay.items[1].vrt);
    short_table_replay.items[1].vrt->resources[0].table_entries.pop_back();
    CHECK(!build_gpu_dependency_graph(short_table_replay, table_graph, error) &&
              error == "resource has an invalid buffer descriptor-table dependency contract",
          "same-resource mutation: dependency analysis rejects a short descriptor-array payload");

    replay.operations[4].realized = true;
    CHECK(!build_gpu_dependency_graph(replay, graph, error) &&
          error.find("no materialized item") != std::string::npos,
          "graph rejects a falsely realized operation");

    if (fails) return 1;
    std::printf("== PASS ==\n");
    return 0;
}
