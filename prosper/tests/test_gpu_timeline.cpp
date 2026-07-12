#include "../src/gpu/gpu_timeline.hpp"

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
    CHECK(writer.append_producer(producer, error), "writer appends a prior-producer identity");
    GpuTimelineSubmit second = first;
    second.submit_no = 11; second.draw_count = 4; second.dispatch_count = 0;
    CHECK(writer.append_submit(second, error), "writer appends a second submit record");
    CHECK(writer.flush(error), "writer flushes explicitly");
    writer.close();

    GpuTimelineFile timeline;
    CHECK(read_gpu_timeline(good, timeline, error), "reader accepts a complete timeline");
    CHECK(timeline.metadata.revision == metadata.revision &&
          timeline.metadata.title_id == metadata.title_id &&
          timeline.metadata.input_route == metadata.input_route,
          "metadata round-trips");
    CHECK(timeline.version == 3 && timeline.submits.size() == 2 && timeline.presents.size() == 1 &&
          timeline.details.size() == 1 && timeline.producers.size() == 1,
          "version and record counts round-trip");
    CHECK(timeline.submits[0].sequence == 1 && timeline.presents[0].sequence == 2 &&
          timeline.details[0].sequence == 3 && timeline.producers[0].sequence == 4 &&
          timeline.submits[1].sequence == 5,
          "global record ordering round-trips");
    CHECK(timeline.submits[0].color0_base == 0x12340000 &&
          timeline.submits[0].color0_width == 642 && timeline.submits[0].color0_height == 362,
          "target identity and extent round-trip");
    CHECK(timeline.presents[0].buffer_index == 2 && timeline.presents[0].flip_arg == -5,
          "signed present fields round-trip");
    CHECK(timeline.details[0].submit_no == 10 && timeline.details[0].missing_operation_count == 1 &&
          timeline.details[0].shader_version_count == 5 && timeline.details[0].resource_bytes == 4096,
          "detailed-capture identity and version statistics round-trip");
    CHECK(timeline.producers[0].resolved && timeline.producers[0].consumer_operation == 19 &&
          timeline.producers[0].producer_submit_no == 9 &&
          timeline.producers[0].producer_command_order == 380 &&
          timeline.producers[0].resource_width == 642,
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
#ifdef _WIN32
    _putenv_s("PROSPER_GPU_TIMELINE", runtime.c_str());
    _putenv_s("PROSPER_CAPTURE_TITLE", "runtime-title");
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE", runtime_capture.c_str());
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "42");
#else
    setenv("PROSPER_GPU_TIMELINE", runtime.c_str(), 1);
    setenv("PROSPER_CAPTURE_TITLE", "runtime-title", 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE", runtime_capture.c_str(), 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "42", 1);
#endif
    begin_gpu_timeline_submit(42);
    record_gpu_timeline_present(9, 1, 77, 1920, 1080); // a PM4 flip can precede submit completion
    GpuState runtime_state;
    runtime_state.command_order = 123;
    record_gpu_timeline_submit(runtime_state, 42);
    close_gpu_timeline();
    GpuTimelineFile runtime_timeline;
    CHECK(read_gpu_timeline(runtime, runtime_timeline, error), "runtime hooks produce an inspectable timeline");
    CHECK(runtime_timeline.presents.size() == 1 && runtime_timeline.submits.size() == 1 &&
          runtime_timeline.presents[0].latest_submit_no == 42,
          "a flip during folding is associated with its active submit");
    CHECK(runtime_timeline.details.size() == 1 && runtime_timeline.details[0].submit_no == 42 &&
          runtime_timeline.details[0].capture_path == runtime_capture &&
          std::filesystem::exists(runtime_capture),
          "exact submit selection writes and links one bounded detail capsule");

    const auto compat_v2 = base.string() + "-compat-v2.prgtl";
    GpuTimelineWriter compat_writer;
    CHECK(compat_writer.open(compat_v2, metadata, error) && compat_writer.append_submit(first, error),
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
    std::filesystem::remove(compat_v2, ec);
    std::filesystem::remove(version1, ec);
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
