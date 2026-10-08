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
#include "fixtures/ngg_merged_lut_fixture.hpp"
#include "fixtures/test_data.h"
#include "gpu/capture/capture_source_gate.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/capture/serialize/ngg_subgroup_codec.hpp"
#include "gpu/execute/ngg_subgroup_draw.hpp"
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
    const auto result =
        capture(draw_with({whole_range_constant_buffer(33), small_constant_buffer(2)}));
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
    auto mapped = [](uint64_t) {
        return ProsperRendererTrackedMappingResult::AuthoritativeTracked;
    };
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
    EXPECT_FALSE(prosper::frontend::capture_buffer_source_unavailable(kRealBuffer, fake_tracked,
                                                                      fake_reserved_state));
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
    const auto result =
        capture(draw_with({whole_range_constant_buffer(33), small_constant_buffer(2)}));
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

// ---- Tail composition (#3807 with #4704 and #3135 P5) ---------------------------------------------
// Each conditional tail is present exactly when the file's version reaches it, in version order:
// v72 merged-NGG, v73 depth bounds, v74 source marks. These arms build the expected bytes from the
// next-lower file plus an independently written tail, so a v72 or v73 capture with no placeholder
// stays byte-identical to what it was before v74 existed, and a v74 capture is that file with the
// header raised and the later tails appended.
namespace {

std::vector<uint32_t> stored_stage(uint32_t stage) {
    std::vector<uint32_t> words{
        0x07230203, 0x00010000, 0, 5,          0, 0x0003000e, 0, 1,   // Logical GLSL450
        0x0005000f, stage,      1, 0x6e69616d, 0, 0x00020013, 2, 0x00030021, 3,         2,
        0x00050036, 2,          1, 0,          3, 0x000200f8, 4, 0x000100fd, 0x00010038};
    if (stage == 4) words.insert(words.begin() + 13, {0x00030010, 1, 7});
    return words;
}

std::shared_ptr<const NggSubgroupDraw> kena_ngg() {
    using namespace prosper::test;
    static const auto linked = ngg::kena_linked(tests_root(__FILE__) / "data");
    static const ShaderResourceTable table = ngg::kena_resources(4);
    NggSubgroupDrawRequest request;
    request.linked_code = linked.data();
    request.dwords = linked.size();
    request.resources = &table;
    request.shell.rsrc2_gs_lds_size = ngg_rsrc2_gs_lds_size(ngg::kKenaRsrc2Gs);
    request.shell.user_sgprs = ngg::kKenaUserSgprs;
    request.limits = ngg::kena_limits();
    request.shape = {NggInputTopology::TriangleStrip, 4, 1, 0};
    request.raster.topology = NggOutputTopology::TriangleList;
    request.raster.layer_from_pos1 = true;
    request.raster.layer_slices = 32;
    request.raster.route = NggLayerRoute::ForwardingGeometry;
    request.push_constants = {1, 2, 3, 4, 5, 6, 7, 8};
    return build_ngg_subgroup_draw(request);
}

// Two draws; draw 1 binds the whole-range buffer with no blob, marked or not.
GpuCaptureFile composed(bool ngg, bool bounded, bool placeholder) {
    GpuCaptureFile capture;
    capture.metadata.title_id = "PPSA00000";
    capture.metadata.width = capture.metadata.height = 8;
    for (uint32_t i = 0; i < 2; ++i) {
        GpuCapturedDraw draw;
        draw.vs = stored_stage(0);
        draw.fs = stored_stage(4);
        draw.vertex_count = draw.raw_draw_count = 3;
        draw.draw_index = 7 + i;
        draw.command_order = 10 + i;
        if (i == 0 && ngg) draw.ngg_subgroup = kena_ngg();
        if (i == 1) {
            if (bounded) {
                draw.ps.depth_bounds_enable = true;
                draw.ps.depth_bounds_min = 0.25f;
                draw.ps.depth_bounds_max = 0.75f;
            }
            GpuCapturedResource buffer;
            buffer.resource = whole_range_constant_buffer(33);
            buffer.captured_size = kWholeRange;
            buffer.source_unavailable = placeholder;
            draw.prt.present = true;
            draw.prt.resources.push_back(buffer);
        }
        capture.draws.push_back(draw);
        capture.operations.push_back(
            {SubmitOperationKind::Draw, draw.draw_index, draw.command_order, true});
    }
    return capture;
}

std::vector<uint8_t> serialized(const GpuCaptureFile& capture) {
    std::vector<uint8_t> bytes;
    std::string error;
    EXPECT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    return bytes;
}

std::vector<uint8_t> with_version(std::vector<uint8_t> bytes, uint32_t version) {
    for (unsigned i = 0; i < 4; ++i) bytes.at(8 + i) = uint8_t(version >> (8u * i));
    return bytes;
}

std::vector<uint8_t> concat(std::vector<uint8_t> a, const std::vector<uint8_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// Independent oracles for each tail, written field by field.
std::vector<uint8_t> ngg_tail(const std::shared_ptr<const NggSubgroupDraw>& first) {
    Writer w;
    w.u32(2);
    EXPECT_TRUE(write_ngg_subgroup_draw(w, first));
    EXPECT_TRUE(write_ngg_subgroup_draw(w, nullptr));
    return w.data;
}
std::vector<uint8_t> depth_bounds_tail(bool bounded) {
    Writer w;
    w.u32(2);
    w.u8(0);
    w.f32(0.0f);
    w.f32(1.0f);   // draw 0: disabled, default bounds
    w.u8(bounded ? 1u : 0u);
    w.f32(bounded ? 0.25f : 0.0f);
    w.f32(bounded ? 0.75f : 1.0f);
    return w.data;
}
std::vector<uint8_t> source_mark_tail() {
    Writer w;
    w.u32(1);   // draw 0 binds nothing; draw 1's one prt resource
    w.u8(1);
    return w.data;
}

}   // namespace

TEST(CaptureTailComposition, V72AndV73WithoutAPlaceholderAreTheirPreV74Bytes) {
    const auto ngg = kena_ngg();
    ASSERT_TRUE(ngg);
    const auto v71 = serialized(composed(false, false, false));
    ASSERT_GT(v71.size(), 12u);
    EXPECT_EQ(v71[8], 71u);
    const auto v72 = serialized(composed(true, false, false));
    EXPECT_EQ(v72, concat(with_version(v71, 72), ngg_tail(ngg)));
    EXPECT_EQ(serialized(composed(false, true, false)),
              concat(concat(with_version(v71, 73), ngg_tail(nullptr)), depth_bounds_tail(true)));
    EXPECT_EQ(serialized(composed(true, true, false)),
              concat(with_version(v72, 73), depth_bounds_tail(true)));
}

TEST(CaptureTailComposition, V74IsTheLowerFileWithItsLaterTailsAppendedInVersionOrder) {
    const auto v71 = serialized(composed(false, false, false));
    const auto v72 = serialized(composed(true, false, false));
    const auto v73 = serialized(composed(false, true, false));
    const auto v73_ngg = serialized(composed(true, true, false));
    // Only a placeholder: the v71 body, a v72 tail with no description, a v73 tail with the test
    // disabled on every draw, then the marks.
    EXPECT_EQ(
        serialized(composed(false, false, true)),
        concat(concat(concat(with_version(v71, 74), ngg_tail(nullptr)), depth_bounds_tail(false)),
               source_mark_tail()));
    EXPECT_EQ(serialized(composed(true, false, true)),
              concat(concat(with_version(v72, 74), depth_bounds_tail(false)), source_mark_tail()));
    EXPECT_EQ(serialized(composed(false, true, true)),
              concat(with_version(v73, 74), source_mark_tail()));
    EXPECT_EQ(serialized(composed(true, true, true)),
              concat(with_version(v73_ngg, 74), source_mark_tail()));
}

TEST(CaptureTailComposition, DepthBoundsAndAPlaceholderRoundTripAndReplayTogether) {
    const auto capture = composed(true, true, true);
    const auto bytes = serialized(capture);
    ASSERT_GT(bytes.size(), 12u);
    EXPECT_EQ(bytes[8], 74u);
    GpuCaptureFile loaded;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    EXPECT_EQ(loaded.format_version, 74u);
    ASSERT_EQ(loaded.draws.size(), 2u);
    EXPECT_TRUE(loaded.draws[0].ngg_subgroup);
    EXPECT_FALSE(loaded.draws[0].ps.depth_bounds_enable);
    EXPECT_TRUE(loaded.draws[1].ps.depth_bounds_enable);
    EXPECT_EQ(loaded.draws[1].ps.depth_bounds_min, 0.25f);
    EXPECT_EQ(loaded.draws[1].ps.depth_bounds_max, 0.75f);
    ASSERT_EQ(loaded.draws[1].prt.resources.size(), 1u);
    EXPECT_TRUE(loaded.draws[1].prt.resources[0].source_unavailable);

    GpuReplayFrame replay;
    ASSERT_TRUE(materialize_gpu_replay(loaded, replay, error)) << error;
    ASSERT_EQ(replay.items.size(), 2u);
    EXPECT_TRUE(replay.items[1].ps.depth_bounds_enable);
    EXPECT_EQ(replay.items[1].ps.depth_bounds_min, 0.25f);
    EXPECT_EQ(replay.items[1].ps.depth_bounds_max, 0.75f);
    ASSERT_TRUE(replay.items[1].prt);
    ASSERT_EQ(replay.items[1].prt->resources.size(), 1u);
    EXPECT_TRUE(replay.items[1].prt->resources[0].replay_source_unavailable);
    EXPECT_EQ(replay.items[1].prt->resources[0].host_data, nullptr);

    // A v73 reader-era file (the same capture without the placeholder) still reads its bounds.
    GpuCaptureFile v73;
    ASSERT_TRUE(deserialize_gpu_capture(serialized(composed(true, true, false)), v73, error))
        << error;
    EXPECT_EQ(v73.format_version, 73u);
    EXPECT_TRUE(v73.draws[1].ps.depth_bounds_enable);
    EXPECT_FALSE(v73.draws[1].prt.resources[0].source_unavailable);
}
