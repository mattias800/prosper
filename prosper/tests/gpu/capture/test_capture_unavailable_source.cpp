// #3807: a draw buffer the live renderer's source gate refuses is captured as a placeholder.
//
// Kena (PPSA01802) binds a whole-range V# -- NUM_RECORDS 0xffffffff -- at 0xf00000000000, past the top
// of the 47-bit user address space. The live draw path finds nothing mapped there and binds its
// all-zero fallback; the F9 capture used to plan the declared 4 GiB as guest bytes, hit the 1 GiB
// per-resource ceiling and lose the whole frame. These arms pin the replacement contract end to end
// through the real collector, codec and materializer: the capture asks the renderer's own gate,
// records what the renderer did, writes no blob for it, and replay binds the same fallback by the
// record. The ceiling still fires for a source the gate admits. No Vulkan device or game is involved.
//
// The mutation that turns the first arm red: drop the `unavailable_sources->contains(&r)` skip in
// collect_intervals (capture_collect.cpp). The 4 GiB range is then planned again and the capture
// fails with the per-resource ceiling error, exactly the live Kena failure.
#include "gpu/capture/capture_source_gate.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "shared/live/buffer_source_gate.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint64_t kSyntheticWholeRange = 0xf00000000000ull;   // Kena's binding 32/33 address
constexpr uint32_t kWholeRange = 0xffffffffu;
constexpr uint64_t kRealBuffer = 0x100000ull;

// Restores the process-global probe even when an assertion returns early.
struct ProbeScope {
    explicit ProbeScope(CaptureBufferSourceProbe probe) {
        set_gpu_capture_buffer_source_probe(std::move(probe));
    }
    ~ProbeScope() { set_gpu_capture_buffer_source_probe({}); }
};

ShaderResource whole_range_constant_buffer(uint32_t binding) {
    ShaderResource r{};
    r.cls = ResourceClass::ConstantBuffer;
    r.format = DataFormat::Unknown;
    r.num_components = 0;
    r.binding = binding;
    r.gpu_addr = kSyntheticWholeRange;
    r.size = kWholeRange;
    r.srt_offset = 0x40;
    return r;
}

ShaderResource small_constant_buffer(uint32_t binding) {
    ShaderResource r{};
    r.cls = ResourceClass::ConstantBuffer;
    r.format = DataFormat::Float32;
    r.num_components = 4;
    r.binding = binding;
    r.gpu_addr = kRealBuffer;
    r.size = 64;
    r.srt_offset = 0x50;
    return r;
}

DrawItem draw_with(std::vector<ShaderResource> resources) {
    DrawItem draw;
    draw.vs = {0x07230203, 1, 2};
    draw.fs = {0x07230203, 3};
    draw.prt = std::make_shared<ShaderResourceTable>();
    draw.prt->resources = std::move(resources);
    draw.vertex_count = draw.raw_draw_count = 3;
    draw.ps.topology = 3;
    draw.ps.color_write_mask = 0xf;
    draw.color0_base = 0x2000;
    draw.color0_width = draw.color0_height = 16;
    draw.draw_index = 0;
    draw.command_order = 1;
    return draw;
}

struct Capture {
    bool ok = false;
    std::string error;
    GpuCaptureFile file;
    bool synthetic_read = false;   // the reader was asked for the unmapped range
};

Capture capture(const DrawItem& draw) {
    Capture result;
    std::vector<uint8_t> real(64);
    for (size_t i = 0; i < real.size(); ++i) real[i] = static_cast<uint8_t>(0x40 + i);
    auto reader = [&](uint64_t addr, uint8_t* dst, size_t bytes) -> size_t {
        if (addr >= kSyntheticWholeRange) {
            result.synthetic_read = true;
            return 0;
        }
        if (addr < kRealBuffer || addr - kRealBuffer >= real.size()) return 0;
        const size_t offset = static_cast<size_t>(addr - kRealBuffer);
        const size_t n = std::min(bytes, real.size() - offset);
        std::memcpy(dst, real.data() + offset, n);
        return n;
    };
    GpuCaptureMetadata meta;
    meta.width = meta.height = 16;
    meta.revision = "unavailable-source-test";
    result.ok = capture_draw_items({draw}, meta, reader, result.file, result.error);
    return result;
}

const GpuCapturedResource* captured_binding(const GpuCaptureFile& file, uint32_t binding) {
    for (const auto& resource : file.draws.at(0).prt.resources)
        if (resource.resource.binding == binding) return &resource;
    return nullptr;
}

const ShaderResource* replayed_binding(const GpuReplayFrame& replay, uint32_t binding) {
    for (const auto& resource : replay.items.at(0).prt->resources)
        if (resource.binding == binding) return &resource;
    return nullptr;
}

// The renderer's probe as the live frontend registers it, against a fake mapping table that holds
// exactly the real buffer. The REAL gate function, not a copy of its rule.
bool fake_tracked_call = false;
ProsperRendererTrackedMappingResult fake_tracked(uint64_t) {
    fake_tracked_call = true;
    return ProsperRendererTrackedMappingResult::Untracked;
}
int fake_reserved_state(uint64_t addr) {
    return addr >= kRealBuffer && addr < kRealBuffer + 0x10000 ? 2 : 0;
}
CaptureBufferSourceProbe live_shaped_probe() {
    return [](uint64_t addr) {
        return prosper::frontend::capture_buffer_source_unavailable(addr, fake_tracked,
                                                                     fake_reserved_state);
    };
}

}   // namespace

TEST(CaptureUnavailableSource, WholeRangeDescriptorOverNothingMappedIsAPlaceholderNotABlob) {
    const ProbeScope probe(live_shaped_probe());
    const auto result = capture(draw_with({whole_range_constant_buffer(33), small_constant_buffer(2)}));
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_FALSE(result.synthetic_read) << "no guest byte of the unmapped range may be read";

    const auto* placeholder = captured_binding(result.file, 33);
    const auto* real = captured_binding(result.file, 2);
    ASSERT_TRUE(placeholder && real);
    EXPECT_TRUE(placeholder->source_unavailable);
    EXPECT_EQ(placeholder->blob_index, 0xFFFFFFFFu);
    EXPECT_TRUE(valid_source_unavailable_record(*placeholder));
    EXPECT_EQ(placeholder->resource.gpu_addr, kSyntheticWholeRange);
    EXPECT_EQ(placeholder->resource.size, kWholeRange);   // the descriptor is recorded unchanged
    EXPECT_FALSE(real->source_unavailable);
    ASSERT_NE(real->blob_index, 0xFFFFFFFFu) << "a mapped binding still owns its bytes";

    uint64_t blob_bytes = 0;
    for (const auto& blob : result.file.blobs) blob_bytes += blob.bytes.size();
    EXPECT_LE(blob_bytes, 64u) << "nothing of the 4 GiB declaration reaches a blob";
    EXPECT_EQ(result.file.format_version, 74u);

    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(result.file, bytes, error)) << error;
    ASSERT_GT(bytes.size(), 12u);
    EXPECT_EQ(bytes[8], 74u);
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    ASSERT_TRUE(captured_binding(loaded, 33) && captured_binding(loaded, 2));
    EXPECT_TRUE(captured_binding(loaded, 33)->source_unavailable);
    EXPECT_FALSE(captured_binding(loaded, 2)->source_unavailable);

    GpuReplayFrame replay;
    ASSERT_TRUE(materialize_gpu_replay(loaded, replay, error)) << error;
    const auto* replayed_placeholder = replayed_binding(replay, 33);
    const auto* replayed_real = replayed_binding(replay, 2);
    ASSERT_TRUE(replayed_placeholder && replayed_real);
    EXPECT_TRUE(replayed_placeholder->replay_source_unavailable);
    EXPECT_EQ(replayed_placeholder->host_data, nullptr);
    EXPECT_EQ(replayed_placeholder->gpu_addr, kSyntheticWholeRange);
    EXPECT_FALSE(replayed_real->replay_source_unavailable);
    ASSERT_NE(replayed_real->host_data, nullptr);
    EXPECT_EQ(replayed_real->host_data[0], 0x40u);
}

// "Replay binds the same thing": the draw path's gate must refuse the replayed placeholder by its
// record, even in a process where the address IS mapped. Mutation: drop the `recorded_unavailable`
// branch in classify_recorded_buffer_source and the second expectation goes red.
TEST(CaptureUnavailableSource, ReplayGateRefusesThePlaceholderByItsRecordNotByTheReplayProcess) {
    auto mapped = [](uint64_t) { return ProsperRendererTrackedMappingResult::AuthoritativeTracked; };
    auto committed = [](uint64_t) { return 2; };
    EXPECT_FALSE(prosper::frontend::classify_recorded_buffer_source(
                     false, false, kSyntheticWholeRange, true, mapped, committed)
                     .unavailable)
        << "control: without the record, a mapped address is admitted";
    EXPECT_TRUE(prosper::frontend::classify_recorded_buffer_source(
                    true, false, kSyntheticWholeRange, true, mapped, committed)
                    .unavailable);
    // And the live decision the capture asked is the same gate the renderer runs.
    fake_tracked_call = false;
    EXPECT_TRUE(prosper::frontend::capture_buffer_source_unavailable(
        kSyntheticWholeRange, fake_tracked, fake_reserved_state));
    EXPECT_FALSE(fake_tracked_call) << "the capture asks the authoritative lookup, not the cache";
    EXPECT_FALSE(prosper::frontend::capture_buffer_source_unavailable(
        kRealBuffer, fake_tracked, fake_reserved_state));
}

// The per-resource ceiling is not weakened: a whole-range descriptor over a source the gate ADMITS
// has real bytes behind it, and the capture still refuses to own 4 GiB of them.
TEST(CaptureUnavailableSource, MappedOversizedSourceStillFailsClosedOnTheCeiling) {
    const ProbeScope probe([](uint64_t) { return false; });
    const auto result = capture(draw_with({whole_range_constant_buffer(33)}));
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("per-resource"), std::string::npos) << result.error;
    EXPECT_NE(result.error.find("binding=33"), std::string::npos) << result.error;
}

TEST(CaptureUnavailableSource, WithoutARegisteredProbeCaptureKeepsItsHistoricalFailClosedShape) {
    set_gpu_capture_buffer_source_probe({});
    const auto result = capture(draw_with({whole_range_constant_buffer(32)}));
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("per-resource"), std::string::npos) << result.error;
}

TEST(CaptureUnavailableSource, CaptureWithoutAPlaceholderIsStillWrittenAsV71) {
    const ProbeScope probe(live_shaped_probe());
    const auto result = capture(draw_with({small_constant_buffer(2)}));
    ASSERT_TRUE(result.ok) << result.error;
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(result.file, bytes, error)) << error;
    EXPECT_EQ(bytes[8], 71u);
}

// Images never consult the draw-buffer gate, so a texture over an unmapped address keeps the
// ordinary contract (here: it plans its range and the reader supplies nothing).
TEST(CaptureUnavailableSource, ImagesAreNeverPlaceholders) {
    const ProbeScope probe([](uint64_t) { return true; });
    ShaderResource texture{};
    texture.cls = ResourceClass::Texture;
    texture.format = DataFormat::Unorm8;
    texture.num_components = 4;
    texture.binding = 4;
    texture.img_dim = 1;
    texture.width = texture.height = 2;
    texture.size = 16;
    texture.gpu_addr = kSyntheticWholeRange;
    texture.linear_row_pitch_bytes = 8;
    const auto result = capture(draw_with({texture}));
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_FALSE(captured_binding(result.file, 4)->source_unavailable);
    EXPECT_NE(captured_binding(result.file, 4)->blob_index, 0xFFFFFFFFu);
}

// A corrupt or hand-edited file cannot turn a real resource into zeros by setting one byte, and
// the mark is a strict boolean. The v74 tail is the last thing in the file: one byte per draw-table
// resource, vrt first then prt, so the last two bytes here are the marks of bindings 33 and 2.
TEST(CaptureUnavailableSource, CodecRejectsAMarkOnABackedResourceAndNonBooleanMarks) {
    const ProbeScope probe(live_shaped_probe());
    const auto result = capture(draw_with({whole_range_constant_buffer(33), small_constant_buffer(2)}));
    ASSERT_TRUE(result.ok) << result.error;
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(result.file, bytes, error)) << error;
    ASSERT_EQ(bytes[bytes.size() - 2], 1u);
    ASSERT_EQ(bytes[bytes.size() - 1], 0u);

    auto backed = bytes;
    backed.back() = 1;
    GpuCaptureFile rejected;
    EXPECT_FALSE(deserialize_gpu_capture(backed, rejected, error));
    EXPECT_EQ(error, "invalid unavailable-source placeholder");

    auto non_boolean = bytes;
    non_boolean[bytes.size() - 2] = 2;
    EXPECT_FALSE(deserialize_gpu_capture(non_boolean, rejected, error));
    EXPECT_EQ(error, "invalid unavailable-source mark");

    // The writer refuses the same lie, and refuses a compute placeholder outright.
    auto forged = result.file;
    forged.draws[0].prt.resources[1].source_unavailable = true;
    EXPECT_FALSE(serialize_gpu_capture(forged, bytes, error));
    EXPECT_EQ(error, "invalid unavailable-source placeholder");
    GpuCaptureFile compute = result.file;
    compute.draws.clear();
    compute.operations.clear();
    GpuCapturedCompute dispatch;
    dispatch.resources.present = true;
    GpuCapturedResource placeholder;
    placeholder.resource = whole_range_constant_buffer(1);
    placeholder.source_unavailable = true;
    dispatch.resources.resources.push_back(placeholder);
    compute.computes.push_back(dispatch);
    EXPECT_FALSE(serialize_gpu_capture(compute, bytes, error));
    EXPECT_EQ(error, "compute resource cannot be an unavailable-source placeholder");
}
