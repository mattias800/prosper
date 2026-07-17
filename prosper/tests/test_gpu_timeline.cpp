#include "../src/gpu/gpu_timeline.hpp"
#include "../src/gpu/gpu_capture_bundle.hpp"
#include "../src/gpu/pm4_registers.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

static bool write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(out);
}

int main() {
    std::printf("== test_gpu_timeline ==\n");
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto base = std::filesystem::temp_directory_path() /
        ("prosper-gpu-timeline-" + std::to_string(nonce));
    const auto good = base.string() + ".prgtl";
    const auto truncated = base.string() + "-truncated.prgtl";
    const auto corrupt = base.string() + "-corrupt.prgtl";

    GpuTimelineWriter writer;
    GpuTimelineMetadata metadata{"revision-1", "PPSA15552", "@scripts/dead-cells/route.pad"};
    std::string error;
    CHECK(writer.open(good, metadata, error), "writer creates a versioned timeline");
    GpuTimelineSubmit first;
    first.submit_no = 10; first.process_command_order = 400; first.first_command_order = 390;
    first.last_command_order = 400; first.color0_base = 0x12340000; first.draw_count = 3;
    first.dispatch_count = 1; first.color0_width = 642; first.color0_height = 362;
    first.dma_copy_count = 2;
    first.dma_copies.push_back({0x500000, 0x600000, 64, 0, 395, 0x1000});
    first.dma_copies.push_back({0x500040, 0x600040, 32, 3, 398, 0x1100});
    first.capture_incomplete = false;
    first.target_spans.push_back({0, 1, 642, 362});
    first.target_spans.push_back({1, 2, 738, 420});
    GpuTimelineSubmit invalid_spans = first;
    invalid_spans.target_spans[1].first_draw = 2;
    CHECK(!writer.append_submit(invalid_spans, error) &&
          error.find("invalid target spans") != std::string::npos,
          "writer rejects noncontiguous target-span coverage");
    error.clear();
    GpuTimelineDepthSurface first_depth;
    first_depth.depth_read_base = first_depth.depth_write_base = 0x700000;
    first_depth.stencil_read_base = first_depth.stencil_write_base = 0x830000;
    first_depth.htile_data_base = 0x820000; first_depth.db_depth_view = 0x11;
    first_depth.db_depth_size_xy = 0x01690281; first_depth.db_z_info = 0xa0000003;
    first_depth.db_stencil_info = 0x20000001; first_depth.target_width = 642;
    first_depth.target_height = 362; first_depth.draw_count = 3;
    first_depth.depth_test_count = 3; first_depth.depth_write_count = 2;
    first_depth.compare_mask = 1u << 3;
    first_depth.backing_hash_mask = 7; first_depth.depth_backing_hash = 0x1111;
    first_depth.stencil_backing_hash = 0x2222; first_depth.htile_backing_hash = 0x3333;
    first_depth.backing_writer_kind = 3; first_depth.backing_writer_sequence = 9;
    first_depth.backing_writer_addr = 0x820000; first_depth.backing_writer_size = 0x10000;
    first_depth.backing_writer_order = 380; first_depth.backing_writer_identity = 0xabcdef;
    first.depth_surfaces.push_back(first_depth);
    CHECK(writer.append_submit(first, error), "writer appends a submit record");
    GpuTimelinePresent present;
    present.present_count = 7; present.latest_submit_no = 10; present.buffer_index = 2;
    present.flip_arg = -5; present.width = 1920; present.height = 1080;
    CHECK(writer.append_present(present, error), "writer appends a present record");
    GpuTimelineDetail detail;
    detail.submit_no = 10; detail.capture_path = "/tmp/dead-cells-submit-10.prgcap";
    detail.semantic_draw_count = 3; detail.semantic_dispatch_count = 1;
    detail.realized_draw_count = 2; detail.realized_dispatch_count = 1;
    detail.operation_count = 4; detail.missing_operation_count = 1;
    detail.shader_version_count = 5; detail.resource_version_count = 3; detail.resource_bytes = 4096;
    CHECK(writer.append_detail(detail, error), "writer appends a detailed-capture link");
    GpuTimelineProducer producer;
    producer.consumer_submit_no = 10; producer.consumer_operation = 19;
    producer.future_writer_operation = 28; producer.resource_addr = 0x700000;
    producer.resource_size = 642ull * 362 * 4; producer.resource_width = 642;
    producer.resource_height = 362; producer.resolved = true;
    producer.producer_submit_no = 9; producer.producer_draw_index = 13;
    producer.producer_command_order = 380; producer.producer_target_addr = 0x700000;
    producer.producer_width = 642; producer.producer_height = 362;
    producer.first_writer_kind = GpuTimelineWriterKind::Graphics;
    producer.history_first_submit_no = 2; producer.history_first_draw_index = 7;
    producer.history_first_command_order = 80; producer.history_write_count = 31;
    producer.history_submit_count = 8; producer.history_window_first_submit_no = 1;
    producer.lifetime_truncated = true; producer.history_window_truncated = true;
    producer.first_color_has_clear = true; producer.first_color_clear_word0 = 0x11223344;
    producer.first_color_clear_word1 = 0x55667788; producer.first_color_control = 0x60;
    producer.first_color_control_mode = 6; producer.first_target_mask = 0xf;
    producer.first_color_format = 10;
    CHECK(writer.append_producer(producer, error), "writer appends a prior-producer identity");
    GpuTimelineSubmit second = first;
    second.submit_no = 11; second.draw_count = 4; second.dispatch_count = 0;
    second.target_spans[1].draw_count = 3;
    CHECK(writer.append_submit(second, error), "writer appends a second submit record");
    CHECK(writer.flush(error), "writer flushes explicitly");
    writer.close();

    GpuTimelineFile timeline;
    CHECK(read_gpu_timeline(good, timeline, error), "reader accepts a complete timeline");
    CHECK(timeline.metadata.revision == metadata.revision &&
          timeline.metadata.title_id == metadata.title_id &&
          timeline.metadata.input_route == metadata.input_route,
          "metadata round-trips");
    CHECK(timeline.version == 8 && timeline.submits.size() == 2 && timeline.presents.size() == 1 &&
          timeline.details.size() == 1 && timeline.producers.size() == 1,
          "version and record counts round-trip");
    CHECK(timeline.submits[0].sequence == 1 && timeline.presents[0].sequence == 2 &&
          timeline.details[0].sequence == 3 && timeline.producers[0].sequence == 4 &&
          timeline.submits[1].sequence == 5,
          "global record ordering round-trips");
    CHECK(timeline.submits[0].color0_base == 0x12340000 &&
          timeline.submits[0].color0_width == 642 && timeline.submits[0].color0_height == 362,
          "target identity and extent round-trip");
    CHECK(timeline.submits[0].dma_copy_count == 2 &&
          !timeline.submits[0].capture_incomplete &&
          timeline.submits[0].dma_copies.size() == 2 &&
          timeline.submits[0].dma_copies[0].src == 0x600000 &&
          timeline.submits[0].dma_copies[1].dst == 0x500040 &&
          timeline.submits[0].dma_copies[1].command_order == 398,
          "ordered-DMA identities and exact command order round-trip");
    CHECK(timeline.submits[0].depth_surfaces.size() == 1 &&
          timeline.submits[0].depth_surfaces[0].htile_data_base == 0x820000 &&
          timeline.submits[0].depth_surfaces[0].db_depth_size_xy == 0x01690281 &&
          timeline.submits[0].depth_surfaces[0].depth_write_count == 2 &&
          timeline.submits[0].depth_surfaces[0].backing_hash_mask == 7 &&
          timeline.submits[0].depth_surfaces[0].htile_backing_hash == 0x3333 &&
          timeline.submits[0].depth_surfaces[0].backing_writer_sequence == 9 &&
          timeline.submits[0].depth_surfaces[0].backing_writer_identity == 0xabcdef,
          "native timeline depth identity, HTILE programming, and use counts round-trip");
    CHECK(timeline.submits[0].target_spans.size() == 2 &&
          timeline.submits[0].target_spans[1].first_draw == 1 &&
          timeline.submits[0].target_spans[1].draw_count == 2 &&
          timeline.submits[0].target_spans[1].width == 738 &&
          !timeline.submits[0].target_spans_truncated,
          "target-extent spans round-trip without run-local addresses");
    GpuTimelineSelector selector;
    selector.min_submit_no = 1; selector.target_width = 738; selector.target_height = 420;
    selector.target_min_draw = 1; selector.target_max_draw = 2;
    selector.min_draws = selector.max_draws = 3;
    selector.min_dispatches = selector.max_dispatches = 1;
    CHECK(gpu_timeline_submit_matches(timeline.submits[0], selector),
          "shared selector matches target position plus draw/dispatch counts");
    selector.target_min_draw = selector.target_max_draw = 0;
    CHECK(!gpu_timeline_submit_matches(timeline.submits[0], selector),
          "shared selector rejects a target outside the requested draw range");
    selector.target_min_draw = 1; selector.target_max_draw = 2;
    selector.min_dispatches = selector.max_dispatches = 2;
    CHECK(!gpu_timeline_submit_matches(timeline.submits[0], selector),
          "shared selector rejects an otherwise identical dispatch-count mismatch");
    selector.min_dispatches = selector.max_dispatches = 1;
    timeline.submits[0].target_spans_truncated = true;
    CHECK(!gpu_timeline_submit_matches(timeline.submits[0], selector),
          "shared selector refuses an incomplete target signature");
    timeline.submits[0].target_spans_truncated = false;
    CHECK(timeline.presents[0].buffer_index == 2 && timeline.presents[0].flip_arg == -5,
          "signed present fields round-trip");
    CHECK(timeline.details[0].submit_no == 10 && timeline.details[0].missing_operation_count == 1 &&
          timeline.details[0].shader_version_count == 5 && timeline.details[0].resource_bytes == 4096,
          "detailed-capture identity and version statistics round-trip");
    CHECK(timeline.producers[0].resolved && timeline.producers[0].consumer_operation == 19 &&
          timeline.producers[0].producer_submit_no == 9 &&
          timeline.producers[0].producer_command_order == 380 &&
          timeline.producers[0].resource_width == 642 &&
          timeline.producers[0].history_first_submit_no == 2 &&
          timeline.producers[0].history_write_count == 31 &&
          timeline.producers[0].first_writer_kind == GpuTimelineWriterKind::Graphics &&
          timeline.producers[0].lifetime_truncated &&
          timeline.producers[0].history_window_truncated &&
          timeline.producers[0].first_color_has_clear &&
          timeline.producers[0].first_color_clear_word1 == 0x55667788,
          "prior-producer resource and writer identities round-trip");

    std::ifstream input(good, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    CHECK(bytes.size() > 16, "test timeline has framed records");
    std::vector<uint8_t> truncated_bytes(bytes.begin(), bytes.end() - 5);
    CHECK(write_bytes(truncated, truncated_bytes), "created a truncated-tail fixture");
    GpuTimelineFile partial;
    CHECK(read_gpu_timeline(truncated, partial, error), "reader recovers complete records from a truncated tail");
    CHECK(partial.truncated_tail && partial.submits.size() == 1 && partial.presents.size() == 1 &&
          partial.details.size() == 1 && partial.producers.size() == 1,
          "truncated final submit is ignored explicitly");

    std::vector<uint8_t> corrupt_bytes = bytes;
    corrupt_bytes[corrupt_bytes.size() - 12] ^= 0x80;
    CHECK(write_bytes(corrupt, corrupt_bytes), "created a checksum-corrupt fixture");
    GpuTimelineFile rejected;
    CHECK(!read_gpu_timeline(corrupt, rejected, error) &&
          error.find("checksum mismatch") != std::string::npos,
          "reader rejects corruption instead of treating it as truncation");

    const auto runtime = base.string() + "-runtime.prgtl";
    const auto runtime_capture = base.string() + "-submit-42.prgcap";
    const auto predecessor_capture = base.string() + "-submit-41.prgcap";
    const auto bundle_capture = base.string() + "-submits-41-42.prgbundle";
#ifdef _WIN32
    _putenv_s("PROSPER_GPU_TIMELINE", runtime.c_str());
    _putenv_s("PROSPER_CAPTURE_TITLE", "runtime-title");
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE", runtime_capture.c_str());
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "42");
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR", predecessor_capture.c_str());
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE", bundle_capture.c_str());
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_DEPTH", "2");
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM", "642x362");
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_CHECKPOINT_EVERY", "1");
#else
    setenv("PROSPER_GPU_TIMELINE", runtime.c_str(), 1);
    setenv("PROSPER_CAPTURE_TITLE", "runtime-title", 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE", runtime_capture.c_str(), 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "42", 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR", predecessor_capture.c_str(), 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE", bundle_capture.c_str(), 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_DEPTH", "2", 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM", "642x362", 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_CHECKPOINT_EVERY", "1", 1);
#endif
    GpuState prestart_state;
    prestart_state.command_order = 105;
    prestart_state.dma_copies.push_back({0x200000, 0x100000000ull, 16, 0, 104, 0});
    begin_gpu_timeline_submit(40);
    record_gpu_timeline_submit(prestart_state, 40);
    GpuState predecessor_state;
    predecessor_state.cx[prosper::agc::Pm4::CB_COLOR0_ATTRIB2] =
        ((642u - 1u) << 14) | (362u - 1u);
    predecessor_state.cx[prosper::agc::Pm4::DB_DEPTH_CONTROL] = 0x36;
    predecessor_state.cx[prosper::agc::Pm4::DB_Z_READ_BASE] = 0x7000;
    predecessor_state.cx[prosper::agc::Pm4::DB_Z_WRITE_BASE] = 0x7000;
    predecessor_state.cx[prosper::agc::Pm4::DB_HTILE_DATA_BASE] = 0x8200;
    predecessor_state.cx[prosper::agc::Pm4::DB_DEPTH_SIZE_XY] = 0x01690281;
    predecessor_state.draws.push_back({3});
    predecessor_state.command_order = 120;
    begin_gpu_timeline_submit(41);
    record_gpu_timeline_submit(predecessor_state, 41);
    GpuCaptureBundle checkpoint_bundle;
    CHECK(read_gpu_capture_bundle(bundle_capture, checkpoint_bundle, error) &&
          checkpoint_bundle.submits.size() == 1 &&
          checkpoint_bundle.submits[0].submit_index == 41,
          "bundle checkpoint preserves predecessor history before the endpoint exists");
    begin_gpu_timeline_submit(42);
    record_gpu_timeline_present(9, 1, 77, 1920, 1080); // a PM4 flip can precede submit completion
    GpuState runtime_state;
    runtime_state.command_order = 123;
    record_gpu_timeline_submit(runtime_state, 42);
    close_gpu_timeline();
    GpuTimelineFile runtime_timeline;
    CHECK(read_gpu_timeline(runtime, runtime_timeline, error), "runtime hooks produce an inspectable timeline");
    CHECK(runtime_timeline.presents.size() == 1 && runtime_timeline.submits.size() == 3 &&
          runtime_timeline.presents[0].latest_submit_no == 42,
          "a flip during folding is associated with its active submit");
    CHECK(runtime_timeline.submits[0].dma_copy_count == 1 &&
          !runtime_timeline.submits[0].capture_incomplete &&
          runtime_timeline.submits[0].dma_copies.size() == 1 &&
          runtime_timeline.submits[0].dma_copies[0].src == 0x100000000ull &&
          runtime_timeline.submits[0].dma_copies[0].dst == 0x200000 &&
          runtime_timeline.submits[0].first_command_order == 104 &&
          runtime_timeline.submits[0].last_command_order == 104,
          "runtime compact timeline exposes replayable ordered DMA identities");
    CHECK(runtime_timeline.submits[1].depth_surfaces.size() == 1 &&
          runtime_timeline.submits[1].depth_surfaces[0].depth_read_base == 0x700000 &&
          runtime_timeline.submits[1].depth_surfaces[0].htile_data_base == 0x820000 &&
          runtime_timeline.submits[1].depth_surfaces[0].depth_test_count == 1,
          "runtime timeline extracts compact DS programming without realizing resources");
    CHECK(runtime_timeline.submits[1].target_spans.size() == 1 &&
          runtime_timeline.submits[1].target_spans[0].first_draw == 0 &&
          runtime_timeline.submits[1].target_spans[0].draw_count == 1 &&
          runtime_timeline.submits[1].target_spans[0].width == 642 &&
          runtime_timeline.submits[1].target_spans[0].height == 362,
          "runtime timeline records compact semantic target spans");
    CHECK(runtime_timeline.details.size() == 3 && runtime_timeline.details[0].submit_no == 41 &&
          runtime_timeline.details[0].capture_path.find(bundle_capture + "#submit=41") == 0 &&
          runtime_timeline.details[1].capture_path == predecessor_capture &&
          runtime_timeline.details[2].submit_no == 42 &&
          runtime_timeline.details[2].capture_path == runtime_capture &&
          std::filesystem::exists(predecessor_capture) && std::filesystem::exists(runtime_capture),
          "exact submit selection writes and links producer-time predecessor and consumer capsules");
    GpuCaptureBundle runtime_bundle;
    CHECK(read_gpu_capture_bundle(bundle_capture, runtime_bundle, error) &&
          runtime_bundle.submits.size() == 2 && runtime_bundle.submits[0].submit_index == 41 &&
          runtime_bundle.submits[1].submit_index == 42,
          "exact submit selection writes an ordered same-run capture bundle");
    CHECK(runtime_bundle.submits.front().submit_index == 41,
          "bundle start target excludes earlier nonmatching submits and includes the first writer");

    const auto compat_v2 = base.string() + "-compat-v2.prgtl";
    GpuTimelineWriter compat_writer;
    GpuTimelineSubmit legacy_first = first;
    legacy_first.depth_surfaces.clear(); legacy_first.target_spans.clear();
    legacy_first.dma_copies.clear();
    legacy_first.dma_copy_count = 0; legacy_first.capture_incomplete = false;
    CHECK(compat_writer.open(compat_v2, metadata, error) && compat_writer.append_submit(legacy_first, error),
          "created a detail-free compatibility timeline");
    compat_writer.close();
    std::ifstream compat_input(compat_v2, std::ios::binary);
    std::vector<uint8_t> version1_bytes((std::istreambuf_iterator<char>(compat_input)), {});
    version1_bytes[8] = 1; version1_bytes[9] = version1_bytes[10] = version1_bytes[11] = 0;
    const auto version1 = base.string() + "-version1.prgtl";
    CHECK(write_bytes(version1, version1_bytes), "created a version-1 compatibility fixture");
    GpuTimelineFile version1_timeline;
    CHECK(read_gpu_timeline(version1, version1_timeline, error) && version1_timeline.version == 1 &&
          version1_timeline.submits.size() == 1,
          "reader remains backward-compatible with version-1 semantic timelines");

    std::error_code ec;
    std::filesystem::remove(good, ec);
    std::filesystem::remove(truncated, ec);
    std::filesystem::remove(corrupt, ec);
    std::filesystem::remove(runtime, ec);
    std::filesystem::remove(runtime_capture, ec);
    std::filesystem::remove(predecessor_capture, ec);
    std::filesystem::remove(bundle_capture, ec);
    std::filesystem::remove(compat_v2, ec);
    std::filesystem::remove(version1, ec);
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
