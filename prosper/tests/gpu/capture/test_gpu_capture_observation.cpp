// Thin capture reports retain normalized bytes/descriptors without granting executable raw
// provenance. Exercise the actual common-v69 codec and actual CLI, not an imitation parser.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_dependency_graph.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>
#ifndef _WIN32
#include <sys/wait.h>
#endif

using namespace prosper::gpu;

static_assert(!std::is_copy_constructible_v<GpuCaptureObservation>);
static_assert(std::is_move_constructible_v<GpuCaptureObservation>);
static_assert(!std::is_convertible_v<GpuCaptureObservation, GpuReplayFrame>);
static_assert(!std::is_assignable_v<GpuReplayFrame&, GpuCaptureObservation>);
template <class T>
concept PublishesItems = requires(T& value) { value.items; };
template <class T>
concept PublishesFrame = requires(T& value) { value.normalized_; };
static_assert(!PublishesItems<GpuCaptureObservation> && !PublishesFrame<GpuCaptureObservation>);
static_assert(
    !std::is_invocable_v<decltype(build_gpu_dependency_graph), const GpuCaptureObservation&,
                         GpuDependencyGraph&, std::string&>);

namespace {
std::vector<uint32_t> stored_stage(uint32_t stage) {
    std::
        vector<uint32_t>
            words = {0x07230203, 0x00010000, 0,          5,          0,
                     0x0003000e, 0,          1,   // Logical GLSL450 memory model
                     0x0005000f, stage,      1,          0x6e69616d, 0,   // main entry point
                     0x00020013, 2,   // void
                     0x00030021, 3,          2,   // function type
                     0x00050036, 2,          1,          0,          3,
                     0x000200f8, 4,          0x000100fd, 0x00010038};
    if (stage == 4) words.insert(words.begin() + 13, {0x00030010, 1, 7});
    return words;
}
GpuCaptureFile capsule(bool complete = false) {
    GpuCaptureFile capture;
    capture.format_version = 69;
    capture.metadata.title_id = "PPSA00000";
    capture.metadata.width = capture.metadata.height = 8;
    capture.metadata.renderer_env = {{"PROSPER_GPU_CAPTURE_METADATA_ONLY", "1"}};
    GpuCaptureRawShaderVersion raw;
    raw.words = complete ? std::vector<uint32_t>{0xbf810000} : std::vector<uint32_t>{0xbf800000};
    raw.has_endpgm = complete;
    raw.content_hash = gpu_capture_hash(reinterpret_cast<const uint8_t*>(raw.words.data()), 4);
    capture.raw_shader_versions.push_back(raw);
    GpuCapturedDraw draw;
    draw.vs = stored_stage(0);
    draw.fs = stored_stage(4);
    draw.vs_raw_shader_index = 0;
    draw.vertex_count = draw.raw_draw_count = 3;
    draw.draw_index = 7;
    draw.command_order = 10;
    capture.draws.push_back(draw);
    capture.operations = {{SubmitOperationKind::Draw, 7, 10, true}};
    return capture;
}
std::vector<uint8_t> common69_bytes(const GpuCaptureFile& capture) {
    std::vector<uint8_t> bytes;
    std::string error;
    EXPECT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    if (bytes.size() < 16) return {};
    // No owned stages: v70's final extension is count + one absent-owner byte per draw.
    EXPECT_EQ(bytes[8], 70);
    for (const auto& draw : capture.draws) EXPECT_FALSE(draw.owned_waves);
    bytes.resize(bytes.size() - 4 - capture.draws.size());
    bytes[8] = 69;
    return bytes;
}
GpuCapturedResource buffer(uint64_t address, uint32_t size, uint32_t binding) {
    GpuCapturedResource resource;
    resource.resource.cls = ResourceClass::ConstantBuffer;
    resource.resource.gpu_addr = address;
    resource.resource.size = size;
    resource.resource.binding = binding;
    resource.captured_size = size;
    return resource;
}
std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), {}};
}
std::string quote(const std::string& value) {
#ifdef _WIN32
    return "\"" + value + "\"";
#else
    std::string result = "'";
    for (char ch : value) result += ch == '\'' ? "'\"'\"'" : std::string(1, ch);
    return result + "'";
#endif
}
struct CliResult {
    int code;
    std::string output;
};
CliResult cli(const std::filesystem::path& input, const std::string& arguments,
              const std::filesystem::path& log) {
    const char* executable = std::getenv("PROSPER_GPU_REPLAY_TEST_BINARY");
    if (!executable) {
        ADD_FAILURE() << "CMake must bind the actual gpu_replay executable";
        return {-1, {}};
    }
    const auto command = quote(executable) + " " + arguments + " " + quote(input.string()) + " > " +
                         quote(log.string()) + " 2>&1";
    int status = std::system(command.c_str());
#ifndef _WIN32
    status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    return {status, read_text(log)};
}
std::filesystem::path write_common69(const GpuCaptureFile& capture, const char* name) {
    const auto path = prosper_test::test_scratch_path(name);
    const auto bytes = common69_bytes(capture);
    GpuCaptureFile decoded;
    std::string error;
    EXPECT_TRUE(deserialize_gpu_capture(bytes, decoded, error)) << error;
    EXPECT_EQ(decoded.format_version, 69);
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    EXPECT_TRUE(file.good());
    return path;
}
}   // namespace

#ifndef PROSPER_OBSERVATION_CLI_TESTS
TEST(GpuCaptureObservation, Common69ReportsDoNotGrantRawExecution) {
    const auto bytes = common69_bytes(capsule());
    GpuCaptureFile capture;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, capture, error)) << error;
    ASSERT_EQ(capture.format_version, 69);
    GpuCaptureObservation observation;
    ASSERT_TRUE(materialize_gpu_capture_observation(capture, observation, error)) << error;
    GpuDependencyGraph graph;
    EXPECT_TRUE(build_gpu_capture_observation_graph(observation, graph, error)) << error;
    EXPECT_EQ(graph.nodes.size(), 1);
    GpuReplayFrame executable;
    EXPECT_FALSE(materialize_gpu_replay(capture, executable, error));
    EXPECT_EQ(error, "logical-wave replay original decode unavailable stage=vs pc=0");
}

TEST(GpuCaptureObservation, CompleteEndRetainsStrictAdmission) {
    auto capture = capsule(true);
    GpuReplayFrame executable;
    std::string error;
    ASSERT_TRUE(materialize_gpu_replay(capture, executable, error)) << error;
    EXPECT_EQ(executable.items.size(), 1);
    EXPECT_EQ(executable.items[0].vs_raw_shader_index, 0);
}

TEST(GpuCaptureObservation, BadPresentBlobIsNotOmittedPayload) {
    auto capture = capsule(true);
    capture.draws[0].vrt.present = true;
    auto resource = buffer(0x1000, 16, 0);
    resource.blob_index = 9;
    capture.draws[0].vrt.resources.push_back(resource);
    GpuCaptureObservation observation;
    GpuReplayFrame executable;
    std::string error;
    EXPECT_FALSE(materialize_gpu_capture_observation(capture, observation, error));
    EXPECT_EQ(error, "resource references an invalid capture blob");
    EXPECT_FALSE(materialize_gpu_replay(capture, executable, error));
    EXPECT_EQ(error, "resource references an invalid capture blob");
    GpuDependencyGraph graph;
    EXPECT_FALSE(build_gpu_capture_observation_graph(observation, graph, error));
}

TEST(GpuCaptureObservation, EarlierStrictRefusalPrecedesLaterTableError) {
    auto capture = capsule();
    auto later = capture.draws[0];
    later.draw_index = 8;
    later.vrt.present = true;
    auto resource = buffer(0x1000, 16, 0);
    resource.blob_index = 9;
    later.vrt.resources.push_back(resource);
    capture.draws.push_back(later);
    GpuReplayFrame executable;
    std::string error;
    EXPECT_FALSE(materialize_gpu_replay(capture, executable, error));
    EXPECT_EQ(error, "logical-wave replay original decode unavailable stage=vs pc=0");
    GpuCaptureObservation observation;
    EXPECT_FALSE(materialize_gpu_capture_observation(capture, observation, error));
    EXPECT_EQ(error, "resource references an invalid capture blob")
        << "This later error must reach normalization, not fail upfront validation";
}

TEST(GpuCaptureObservation, GraphPreservesDrawDmaOverlapAndMovedOwner) {
    auto capture = capsule();
    auto& producer = capture.draws[0];
    producer.draw_index = 0;
    producer.color0_base = 0x2000;
    producer.color0_width = producer.color0_height = 8;
    producer.ps.color_write_mask = 0xf;
    auto consumer = producer;
    consumer.draw_index = 1;
    consumer.command_order = 30;
    consumer.color0_base = 0x4000;
    consumer.prt.present = true;
    auto image = buffer(0x2000, 256, 4);
    image.resource.cls = ResourceClass::Texture;
    image.resource.width = image.resource.height = 8;
    consumer.prt.resources = {image, buffer(0x3004, 8, 5)};
    capture.draws.push_back(consumer);
    GpuCapturedDmaCopy dma;
    dma.src = 0x5000;
    dma.dst = 0x3000;
    dma.bytes = 16;
    dma.command_order = 20;
    capture.dma_copies.push_back(dma);
    capture.operations = {{SubmitOperationKind::Draw, 0, 10, true},
                          {SubmitOperationKind::DmaCopy, 0, 20, true},
                          {SubmitOperationKind::Draw, 1, 30, true}};
    GpuCaptureObservation observation;
    std::string error;
    ASSERT_TRUE(materialize_gpu_capture_observation(capture, observation, error)) << error;
    capture = {};   // Observation owns all normalized metadata independently.
    auto moved = std::move(observation);
    GpuDependencyGraph graph;
    EXPECT_FALSE(build_gpu_capture_observation_graph(observation, graph, error));
    ASSERT_TRUE(build_gpu_capture_observation_graph(moved, graph, error)) << error;
    ASSERT_EQ(graph.nodes.size(), 3);
    ASSERT_EQ(graph.edges.size(), 2);
    EXPECT_EQ(graph.edges[0].producer_operation, 0);
    EXPECT_EQ(graph.edges[0].consumer_operation, 2);
    EXPECT_EQ(graph.edges[0].access.addr, 0x2000);
    EXPECT_EQ(graph.edges[1].producer_operation, 1);
    EXPECT_EQ(graph.edges[1].access.addr, 0x3004);
    EXPECT_EQ(graph.edges[1].access.size, 8);
    ASSERT_EQ(graph.external_leaves.size(), 1);
    EXPECT_EQ(graph.external_leaves[0].access.addr, 0x5000);
    EXPECT_EQ(graph.external_leaves[0].access.size, 16);
}

TEST(GpuCaptureObservation, GraphRetainsSerializedRelocationLogicalAndSegmentBytes) {
    auto capture = capsule(true);
    capture.draws.clear();
    std::array<uint8_t, 8> pointee{1, 2, 3, 4, 5, 6, 7, 8};
    const auto address = reinterpret_cast<uint64_t>(pointee.data());
    std::array<uint8_t, 16> source_bytes{};
    std::memcpy(source_bytes.data(), &address, sizeof(address));
    ShaderResource source;
    source.cls = ResourceClass::ConstantBuffer;
    source.format = DataFormat::Uint32;
    source.num_components = 1;
    source.gpu_addr = 0x7000;
    source.size = source_bytes.size();
    source.stride = 16;
    const auto& layout = kIndirectPointerStaticFootprintLayout;
    const std::array<uint32_t, 4> witnesses{kIndirectPointerProofSchema, 0x1234, 0x5678,
                                            layout.tag ^ kIndirectPointerProofSchema ^ 0x1234 ^
                                                0x5678};
    const std::array<IndirectBufferRelocationRecord, 1> records{{{0, address, 8}}};
    std::shared_ptr<std::vector<uint8_t>> owner;
    IndirectBufferRelocationInfo info;
    ASSERT_TRUE(build_indirect_buffer_relocation(source, source_bytes.data(), layout, records,
                                                 witnesses, owner, info));
    GpuCapturedResource carrier;
    carrier.resource = source;
    carrier.internal_bytes = *owner;
    carrier.captured_size = owner->size();
    GpuCapturedCompute compute;
    compute.dispatch_index = 2;
    compute.command_order = 10;
    compute.raw_shader_index = 0;
    compute.recompile_config_available = true;
    compute.recompile_config.local_x = compute.recompile_config.local_y =
        compute.recompile_config.local_z = 1;
    compute.recompile_config.threads_x = compute.recompile_config.threads_y =
        compute.recompile_config.threads_z = 1;
    compute.launch.groups_x = compute.launch.groups_y = compute.launch.groups_z = 1;
    compute.launch.local_x = compute.launch.local_y = compute.launch.local_z = 1;
    compute.launch.threads_x = compute.launch.threads_y = compute.launch.threads_z = 1;
    compute.resources.present = true;
    compute.resources.resources = {carrier};
    capture.computes = {compute};
    capture.operations = {{SubmitOperationKind::Dispatch, 2, 10, true}};
    // A genuine manifest retains internal carrier bytes but explicitly omits guest blobs.
    capture.blobs.push_back({0, 0, gpu_capture_hash(nullptr, 0), {}});
    std::string error;
    const auto bytes = common69_bytes(capture);
    ASSERT_TRUE(deserialize_gpu_capture(bytes, capture, error)) << error;
    ASSERT_EQ(capture.computes[0]
                  .resources.resources[0]
                  .resource.indirect_pointer_relocation.carrier_version,
              0);
    GpuCaptureObservation observation;
    ASSERT_TRUE(materialize_gpu_capture_observation(capture, observation, error)) << error;
    capture = {};
    owner.reset();
    pointee.fill(0);   // Reporting depends on captured bytes, never the live pointee.
    GpuDependencyGraph graph;
    ASSERT_TRUE(build_gpu_capture_observation_graph(observation, graph, error)) << error;
    ASSERT_EQ(graph.external_leaves.size(), 2);
    EXPECT_EQ(graph.external_leaves[0].access.addr, 0x7000);
    EXPECT_EQ(graph.external_leaves[0].access.size, 16);
    EXPECT_EQ(graph.external_leaves[1].access.addr, address);
    EXPECT_EQ(graph.external_leaves[1].access.size, 8);
}

TEST(GpuCaptureObservation, DefaultObservationCannotProduceAReportGraph) {
    GpuCaptureObservation observation;
    GpuDependencyGraph graph;
    std::string error;
    EXPECT_FALSE(build_gpu_capture_observation_graph(observation, graph, error));
    EXPECT_EQ(error, "capture observation is unavailable");
}

#else
TEST(GpuCaptureObservationCli, Common69TerminalReportsUseActualTool) {
    const auto input = write_common69(capsule(), "thin69.prgcap");
    const auto log = prosper_test::test_scratch_path("report.log");
    for (const std::string arguments : {"--inspect-only", "--validate", "--graph"}) {
        const auto result = cli(input, arguments, log);
        EXPECT_EQ(result.code, 0) << arguments << "\n" << result.output;
        if (arguments == "--inspect-only")
            EXPECT_NE(result.output.find("draw[7]"), std::string::npos);
        if (arguments == "--validate")
            EXPECT_NE(result.output.find("result=accept"), std::string::npos);
        if (arguments == "--graph")
            EXPECT_NE(result.output.find("operations=1"), std::string::npos);
    }
    const auto json = prosper_test::test_scratch_path("graph.json");
    const auto result = cli(input, "--graph-json " + quote(json.string()), log);
    EXPECT_EQ(result.code, 0) << result.output;
    EXPECT_NE(read_text(json).find("\"operation_count\": 1"), std::string::npos);
}

TEST(GpuCaptureObservationCli, MixedExecutableCommandsKeepStrictRefusal) {
    const auto input = write_common69(capsule(), "thin69.prgcap");
    const auto log = prosper_test::test_scratch_path("mixed.log");
    const auto output = prosper_test::test_scratch_path("sentinel.spv");
    {
        std::ofstream sentinel(output);
        sentinel << "retained sentinel";
    }
    for (const std::string extra : std::vector<std::string>{
             "--recompile-raw", "--prepend " + quote(input.string()),
             "--dump-shader 0:vs " + quote(output.string()), "--allow-mismatch"}) {
        const auto result = cli(input, "--inspect-only " + extra, log);
        EXPECT_EQ(result.code, 2) << extra << "\n" << result.output;
        EXPECT_NE(
            result.output.find("logical-wave replay original decode unavailable stage=vs pc=0"),
            std::string::npos)
            << result.output;
    }
    EXPECT_EQ(read_text(output), "retained sentinel");
    const auto complete = write_common69(capsule(true), "complete69.prgcap");
    const auto positive = cli(complete, "--inspect-only --allow-mismatch", log);
    EXPECT_EQ(positive.code, 0) << positive.output;
    EXPECT_EQ(positive.output.find("execution admission not evaluated"), std::string::npos);
}

TEST(GpuCaptureObservationCli, OwnedBlobOffsetAndScalarSpanReachOriginalFormatterAndGraph) {
    auto capture = capsule();
    auto resource = buffer(0x8008, 32, 3);
    resource.resource.fetch_pc = 12;
    resource.resource.scalar_buffer_dword_count = 8;
    resource.blob_index = 0;
    resource.blob_offset = 8;
    capture.draws[0].vrt.present = true;
    capture.draws[0].vrt.resources = {resource};
    GpuCaptureBlob blob;
    blob.guest_addr = 0x8000;
    blob.bytes.assign(48, 0xaa);
    for (uint8_t i = 0; i < 32; ++i) blob.bytes[8 + i] = i + 1;
    blob.bytes_read = blob.bytes.size();
    blob.content_hash = gpu_capture_hash(blob.bytes.data(), blob.bytes.size());
    char bounded_hash[17];
    std::snprintf(bounded_hash, sizeof(bounded_hash), "%016llx",
                  static_cast<unsigned long long>(gpu_capture_hash(blob.bytes.data() + 8, 32)));
    capture.blobs = {blob};
    const auto input = write_common69(capture, "payload69.prgcap");
    const auto log = prosper_test::test_scratch_path("payload.log");
    const auto result = cli(input, "--inspect-only", log);
    EXPECT_EQ(result.code, 0) << result.output;
    EXPECT_NE(result.output.find(std::string("captured=32 nz=32 hash=") + bounded_hash),
              std::string::npos)
        << result.output;
    const auto graph = cli(input, "--graph", log);
    EXPECT_EQ(graph.code, 0) << graph.output;
    EXPECT_NE(graph.output.find("addr=0000000000008008 bytes=32"), std::string::npos)
        << graph.output;
}

TEST(GpuCaptureObservationCli, MissingStoredStagesAreUnavailableInsteadOfPass) {
    auto capture = capsule();
    capture.draws[0].vs.clear();
    auto input = write_common69(capture, "missing-vs69.prgcap");
    auto result = cli(input, "--validate", prosper_test::test_scratch_path("missing.log"));
    EXPECT_EQ(result.code, 1) << result.output;
    EXPECT_NE(result.output.find("VS descriptor validation unavailable: missing stored stage"),
              std::string::npos)
        << result.output;
    capture.draws.clear();
    capture.raw_shader_versions.clear();
    GpuCapturedCompute compute;
    compute.dispatch_index = 2;
    capture.computes = {compute};
    capture.operations = {{SubmitOperationKind::Dispatch, 2, 10, true}};
    input = write_common69(capture, "missing-cs69.prgcap");
    result = cli(input, "--validate", prosper_test::test_scratch_path("missing-cs.log"));
    EXPECT_EQ(result.code, 1) << result.output;
    EXPECT_NE(result.output.find("CS descriptor validation unavailable: missing stored stage"),
              std::string::npos)
        << result.output;
    EXPECT_EQ(result.output.find("result=accept"), std::string::npos) << result.output;
}

TEST(GpuCaptureObservationCli, MissingAndCapturedPendingFragmentStagesStayUnavailable) {
    auto capture = capsule();
    capture.draws[0].fs.clear();
    const auto input = write_common69(capture, "missing-ps69.prgcap");
    const auto log = prosper_test::test_scratch_path("missing-ps.log");
    auto result = cli(input, "--validate", log);
    EXPECT_EQ(result.code, 1) << result.output;
    EXPECT_NE(result.output.find("PS descriptor validation unavailable: missing stored stage"),
              std::string::npos)
        << result.output;
    EXPECT_EQ(result.output.find(" PS descriptors="), std::string::npos) << result.output;

    capture = capsule(true);
    capture.draws[0].fs.clear();
    capture.draws[0].fs_raw_shader_index = 0;
    auto owner = std::make_shared<GraphicsOwnedWaveDraw>();
    owner->fragment_pending = true;
    owner->fragment_code =
        std::make_shared<const std::vector<uint32_t>>(capture.raw_shader_versions[0].words);
    PacketRawWaveWindow window;
    window.load_pc = 0;
    window.guest_base = window.guest_begin = 0x1000;
    window.words = {1, 2, 3, 4};
    owner->fragment_windows = {window};
    capture.draws[0].owned_waves = owner;
    // Structurally retained input bytes are observation metadata, not an executable shader proof.
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    GpuCaptureFile decoded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, decoded, error)) << error;
    ASSERT_EQ(decoded.format_version, 70);
    ASSERT_TRUE(decoded.draws[0].owned_waves);
    ASSERT_TRUE(decoded.draws[0].owned_waves->fragment_pending);
    ASSERT_TRUE(decoded.draws[0].fs.empty());
    const auto pending = prosper_test::test_scratch_path("pending-ps70.prgcap");
    {
        std::ofstream file(pending, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        ASSERT_TRUE(file.good());
    }
    result = cli(pending, "--validate", log);
    EXPECT_EQ(result.code, 1) << result.output;
    EXPECT_NE(
        result.output.find("PS descriptor validation unavailable: captured pending stored stage"),
        std::string::npos)
        << result.output;
    EXPECT_EQ(result.output.find(" PS descriptors="), std::string::npos) << result.output;
}

TEST(GpuCaptureObservationCli, DescriptorValidationHasARealNegativeControl) {
    auto capture = capsule();
    capture.draws[0].vs = stored_stage(4);   // A fragment entry cannot satisfy the VS interface.
    const auto input = write_common69(capture, "wrong-stage69.prgcap");
    const auto result = cli(input, "--validate", prosper_test::test_scratch_path("invalid.log"));
    EXPECT_EQ(result.code, 1) << result.output;
    EXPECT_NE(result.output.find("result=reject"), std::string::npos) << result.output;
}

#endif
