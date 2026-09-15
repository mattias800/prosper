// Real live renderer -> backend statistics -> ordered pending spans -> serialized F8 records.
// The separator only creates an ordered graphics boundary; every pixel and upload is produced by
// the registered Vulkan renderer. No BackendResourceReuseStats/RendererTimingRecord is fabricated.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/present/videoout_present.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "host/memory/guest_write_watch.hpp"
#ifdef __linux__
#include "host/image/exec_image.hpp"
#endif
#include "shared/live/live_renderer.hpp"
#include "shared/perf/performance_capture.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
int failures = 0;
void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}
constexpr uint32_t Width = 32, Height = 32, PresentWidth = 64, PresentHeight = 64;
constexpr size_t Words = 2048, Bytes = Words * sizeof(uint32_t);
using Hle8Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t, uint64_t);

void positions(std::vector<uint32_t>& source, bool visible) {
    constexpr std::array<float, 8> xy{-1,-1, -1,1, 1,1, 1,-1};
    for (size_t i = 0; i < xy.size(); ++i)
        source[i] = std::bit_cast<uint32_t>(xy[i] + (visible ? 0.f : 10.f));
}

DrawItem make_draw(std::vector<uint32_t>& source, std::vector<uint8_t>& target) {
    const uint32_t vertex[]{0x7e060280u,0x7e0802f2u,0xe0042000u,0x80020100u,
                            0xf80008cfu,0x04030201u,0xbf810000u};
    const uint32_t fragment[]{0x7e000280u,0x7e0202f2u,0x7e040280u,0x7e0602f2u,
                              0xf800180fu,0x03020100u,0xbf810000u};
    DrawItem item;
    item.vrt = std::make_shared<ShaderResourceTable>();
    ShaderResource buffer{};
    buffer.cls = ResourceClass::VertexBuffer;
    buffer.format = DataFormat::Float32;
    buffer.num_components = 2;
    buffer.binding = 3;
    buffer.stride = 8;
    buffer.sgpr_base = 8;
    buffer.gpu_addr = reinterpret_cast<uintptr_t>(source.data());
    buffer.size = Bytes;
    buffer.host_data = reinterpret_cast<uint8_t*>(source.data());
    buffer.host_data_size = Bytes;
    item.vrt->resources.push_back(buffer);
    item.vs = recompile_vertex(vertex, std::size(vertex), item.vrt.get());
    item.fs = recompile_fragment(fragment, std::size(fragment));
    item.vertex_count = 4;
    item.indices = {0,1,2, 2,3,0};
    item.ps.topology = 3;
    item.ps.color_write_mask = 0xf;
    item.ps.has_clear_color = true;
    item.ps.clear_color[0] = item.ps.clear_color[1] = 0;
    item.ps.clear_color[2] = item.ps.clear_color[3] = 1;
    item.color0_base = reinterpret_cast<uintptr_t>(target.data());
    item.color0_width = Width;
    item.color0_height = Height;
    return item;
}

bool solid(const std::vector<uint8_t>& rgba, bool green) {
    if (rgba.size() != Width * Height * 4u) return false;
    for (size_t i = 0; i < rgba.size(); i += 4)
        if (rgba[i] > 8 || rgba[i + (green ? 1 : 2)] < 247 ||
            rgba[i + (green ? 2 : 1)] > 8 || rgba[i + 3] < 247) return false;
    return true;
}

void sample_packed_texture(DrawItem& draw, uint64_t address, uint32_t extent = 2) {
    const uint32_t fragment[]{
        0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u};
    ShaderResource texture{};
    texture.cls = ResourceClass::Texture;
    texture.format = DataFormat::Unorm2_10_10_10;
    texture.num_components = 4;
    texture.binding = 4;
    texture.sgpr_base = 8;
    texture.img_dim = 1;
    texture.width = texture.height = extent;
    texture.depth = 1;
    texture.size = extent * extent * 4u;
    texture.gpu_addr = address;
    draw.prt = std::make_shared<ShaderResourceTable>();
    draw.prt->resources.push_back(texture);
    draw.fs = recompile_fragment(fragment, std::size(fragment), draw.prt.get());
}

bool solid_red(const std::vector<uint8_t>& rgba) {
    if (rgba.size() != Width * Height * 4u) return false;
    for (size_t i = 0; i < rgba.size(); i += 4)
        if (rgba[i] < 247 || rgba[i + 1] > 8 || rgba[i + 2] > 8 || rgba[i + 3] < 247)
            return false;
    return true;
}

uint64_t integer(const std::string& row, const char* field) {
    const std::string key = std::string("\"") + field + "\":";
    const size_t at = row.find(key);
    if (at == std::string::npos || row.find(key, at + key.size()) != std::string::npos) {
        check(false, field);
        return UINT64_MAX;
    }
    uint64_t value = 0;
    const char* begin = row.data() + at + key.size();
    const auto parsed = std::from_chars(begin, row.data() + row.size(), value);
    const bool valid = parsed.ec == std::errc{} && parsed.ptr != begin &&
        parsed.ptr != row.data() + row.size() && (*parsed.ptr == ',' || *parsed.ptr == '}');
    check(valid, field);
    return valid ? value : UINT64_MAX;
}
double milliseconds(const std::string& row, const char* field) {
    const std::string key = std::string("\"") + field + "\":";
    const size_t at = row.find(key);
    if (at == std::string::npos || row.find(key, at + key.size()) != std::string::npos) {
        check(false, field);
        return -1;
    }
    const char* begin = row.c_str() + at + key.size();
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    const bool valid = end != begin && (*end == ',' || *end == '}') &&
        std::isfinite(value) && value >= 0;
    check(valid, field);
    return valid ? value : -1;
}
} // namespace

int range_capture() {
    namespace fs = std::filesystem;
    uint64_t address = 0;
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    check(map && unmap &&
          map(reinterpret_cast<uint64_t>(&address),0x10000,0x2,0,
              reinterpret_cast<uint64_t>("buffer-range-capture"),0) == 0 && address,
          "real guest-readable mapping supplies overlapping inputs");
    if (!address) return 1;
    std::vector<uint32_t> source(Words,0);
    positions(source,true);
    std::memcpy(reinterpret_cast<void*>(address),source.data(),Bytes);
    std::memcpy(reinterpret_cast<void*>(address+1024),source.data(),32);
    std::vector<uint8_t> target(Width*Height*4u), changed_target(target.size());
    auto draw = make_draw(source,target);
    auto& resource = draw.vrt->resources[0];
    resource.host_data = nullptr; resource.host_data_size = 0; resource.gpu_addr = address;
    auto second = draw;
    second.vrt = std::make_shared<ShaderResourceTable>(*draw.vrt);
    second.vrt->resources[0].gpu_addr += 1024; second.draw_index = 1;
    auto third = draw, fourth = second; third.draw_index = 2; fourth.draw_index = 3;
    auto& capture = prosper::perf::interactive_performance_capture();
    capture.cancel();
    const fs::path directory = "render_buffer_range_capture_test";
    fs::create_directories(directory);
    const auto start = prosper::perf::monotonic_now_ns();
    prosper::perf::ProcessSample sample; sample.monotonic_ns = start;
    capture.observe_sample(sample);
    const auto armed = capture.arm(directory.string(),"TEST","live range statistics","test",
                                   start,std::chrono::system_clock::now());
    check(armed.ok,"F8 records real range-sharing backend calls");
    LiveRenderFn live = [](const std::vector<DrawItem>& items,uint32_t w,uint32_t h) {
        return RenderedFrame(render_submit_items(items,w,h));
    };
    LiveComputeFn separator = [](const std::vector<ComputeItem>&) { return true; };
    const std::vector<SubmitOperation> operations{
        {SubmitOperationKind::Draw,0,10},{SubmitOperationKind::Draw,1,11},
        {SubmitOperationKind::Dispatch,0,20},
        {SubmitOperationKind::Draw,2,30},{SubmitOperationKind::Draw,3,31}};
    const auto split = execute_ordered_items(operations,{draw,second,third,fourth},
                                            {ComputeItem{}},live,separator,Width,Height);
    check(split.render_spans == 2 && solid(split.frame.bytes(),true),
          "two ordered live spans render both overlapping vertex inputs");
    positions(source,false);
    std::memcpy(reinterpret_cast<void*>(address),source.data(),32);
    std::memcpy(reinterpret_cast<void*>(address+1024),source.data(),32);
    draw.color0_base = second.color0_base = reinterpret_cast<uintptr_t>(changed_target.data());
    check(solid(render_submit_items({draw,second},Width,Height),false),
          "next semantic submit sees changed guest vertices");
    sample.monotonic_ns = std::max<uint64_t>(prosper::perf::monotonic_now_ns(),start+5'000'000'000ull);
    capture.observe_sample(sample);
    prosper::perf::CaptureOutcome outcome;
    check(capture.take_outcome(outcome) && outcome.ok && outcome.renderer_records == 2,
          "actual live callbacks publish exactly two F8 submit records");
    std::ifstream input(outcome.path);
    size_t records = 0;
    for (std::string row; std::getline(input,row);) {
        if (row.find("\"type\":\"renderer\"") == std::string::npos) continue;
        const uint64_t callbacks = records++ == 0 ? 2 : 1;
        check(integer(row,"callbacks") == callbacks &&
              integer(row,"buffer_range_uploads") == callbacks &&
              integer(row,"buffer_range_bindings") == 2*callbacks &&
              integer(row,"buffer_range_upload_bytes") == (Bytes+1024)*callbacks &&
              integer(row,"buffer_range_bound_bytes") == 2*Bytes*callbacks &&
              integer(row,"buffer_upload_bytes") == (Bytes+1024)*callbacks,
              "F8 accumulates union copies once across ordered spans and resets on next submit");
        check(milliseconds(row,"res_buffer_range_plan_ms") > 0 &&
              milliseconds(row,"setup_resources_ms") >= milliseconds(row,"res_buffer_range_plan_ms"),
              "measured planner time reaches F8 inside enclosing resource setup");
    }
    check(records == 2,"both published rows were checked");
    capture.cancel(); set_submit_renderer({}); present_reset();
    unmap(address,0x10000,0,0,0,0);
    fs::remove_all(directory);
    return failures ? 1 : 0;
}

#ifdef __linux__
int small_texture_watch(bool unsupported, bool control) {
    using prosper::host::guest_write_watch_stats;
    constexpr size_t SourceBytes = 64u << 10;
    // Install the actual fault handler before HLE mapping registers the source's physical identity.
    check(unsetenv("PROSPER_FAULT_NO_ONSTACK") == 0,
          "production write-watch signal path uses the alternate stack");
    prosper::install_trap_handler();
    prosper::host::guest_write_watch_set_fault_onstack(!unsupported);
    uint64_t address = 0;
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    check(map && unmap &&
          map(reinterpret_cast<uint64_t>(&address), SourceBytes, 0x2, 0,
              reinterpret_cast<uint64_t>("stable-small-texture"), 0) == 0 && address,
          "real mapped 64KiB source provides watchable guest backing");
    if (!address) return 1;
    auto* words = reinterpret_cast<uint32_t*>(address);
    std::fill_n(words, SourceBytes / 4, 0xc00ffc00u);
    std::vector<uint32_t> positions_source(Words, 0);
    positions(positions_source, true);
    std::vector<uint8_t> target(Width * Height * 4u);
    auto draw = make_draw(positions_source, target);
    sample_packed_texture(draw, address, 128);
    const auto baseline = guest_write_watch_stats();
    auto green = [&] {
        check(solid(render_submit_items({draw}, Width, Height), true),
              "unchanged mapped texture renders green");
    };
    green(); // Cold decode has no stability history.
    for (unsigned i = 0; i < 3; ++i) {
        check(guest_write_watch_stats().create_attempts == baseline.create_attempts,
              "cold source and insufficient stability do not attempt registration");
        green();
    }
    check(guest_write_watch_stats().create_attempts == baseline.create_attempts,
          "three exact matches authorize only the next acquisition's attempt");
    green(); // Arm before this acquisition's exact comparison.
    const auto promoted = guest_write_watch_stats();
    if (control) {
        check(promoted.create_attempts == baseline.create_attempts,
              "small-watch control preserves comparison-only admission");
    } else if (unsupported) {
        check(promoted.create_attempts > baseline.create_attempts &&
              promoted.registrations == baseline.registrations &&
              promoted.create_no_mapping > baseline.create_no_mapping,
              "unsupported fault handling refuses registration after a real attempt");
    } else {
        check(promoted.registrations == baseline.registrations + 1,
              "repeated exact matches promote precisely one source watch");
    }
    green();
    const auto reused = guest_write_watch_stats();
    check((reused.unchanged > promoted.unchanged) == (!unsupported && !control),
          "only a successfully registered watch supplies unchanged authority");
    // No manual write notification: the actual CPU store must take the production watch fault.
    std::fill_n(words, SourceBytes / 4, 0xc00003ffu);
    if (!unsupported && !control)
        check(guest_write_watch_stats().faults > reused.faults,
              "raw CPU mutation reaches the production fault handler");
    check(solid_red(render_submit_items({draw}, Width, Height)),
          "mutation invalidates the promoted texture and renders new red pixels");
    check(solid_red(render_submit_items({draw}, Width, Height)),
          "refreshed texture remains red on subsequent reuse");
    if (!unsupported && !control)
        check(guest_write_watch_stats().dirty > reused.dirty,
              "renderer observes the dirty source before reusing decoded pixels");
    check(unmap(address, SourceBytes, 0, 0, 0, 0) == 0, "source unmaps through the HLE");
    set_submit_renderer({});
    present_reset();
    return failures ? 1 : 0;
}
#endif

int main(int argc, char** argv) {
    using namespace prosper::gpu;
    namespace fs = std::filesystem;
    const bool expect_snapshot_move = argc == 2 &&
        std::strcmp(argv[1], "--source-snapshot-expect-move") == 0;
    const bool source_snapshot = expect_snapshot_move ||
        (argc == 2 && std::strcmp(argv[1], "--source-snapshot") == 0);
    // This is the real live renderer with CPU color readback, as in draw_program_skip_render.
    // Only color-target retention is disabled; immutable buffer residency must remain enabled.
    const char* no_targets = std::getenv("PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS");
    if (!no_targets || std::strcmp(no_targets, "1") != 0) {
        check(false, "CTest must set PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1 before startup");
        return 1;
    }
    prosper::register_builtin_hle();
    present_reset();
    std::array<std::vector<uint8_t>, 3> scanouts;
    for (auto& pixels : scanouts) pixels.resize(PresentWidth * PresentHeight * 4u);
    struct VideoBuffer { const void* data; const void* metadata; const void* reserved[2]; };
    std::array<VideoBuffer, 3> buffers{};
    for (size_t i = 0; i < buffers.size(); ++i) buffers[i].data = scanouts[i].data();
    auto open = prosper::Hle::lookup(prosper::nid_hash("sceVideoOutOpen"));
    auto attributes = reinterpret_cast<Hle8Fn>(prosper::Hle::lookup("PjS5uASwcV8"));
    auto register_buffers = prosper::Hle::lookup("rKBUtgRrtbk");
    uint8_t attribute_bytes[0x50]{};
    const uint64_t video = open ? open(0,0,0,0,0,0) : 0;
    if (attributes)
        attributes(reinterpret_cast<uint64_t>(attribute_bytes), 0x8000000000000000ull,
                   0, PresentWidth, PresentHeight, 0,0,0);
    const uint64_t registered = register_buffers
        ? register_buffers(video,0,0,reinterpret_cast<uint64_t>(buffers.data()), buffers.size(),
                           reinterpret_cast<uint64_t>(attribute_bytes)) : UINT64_MAX;
    check(open && attributes && register_buffers && registered == 0 &&
              present_width() == PresentWidth && present_height() == PresentHeight,
          "actual VideoOut registration provides a distinct presentation extent");
    if (failures) return 1;

    prosper::frontend::register_live_renderer("", false);
#ifdef __linux__
    if (argc == 2 && std::strcmp(argv[1], "--small-watch") == 0)
        return small_texture_watch(false, false);
    if (argc == 2 && std::strcmp(argv[1], "--small-watch-unsupported") == 0)
        return small_texture_watch(true, false);
    if (argc == 2 && std::strcmp(argv[1], "--small-watch-control") == 0)
        return small_texture_watch(false, true);
#endif
    if (argc == 2 && std::strcmp(argv[1], "--range-sharing") == 0) return range_capture();
    std::vector<uint32_t> source(Words, 0);
    std::vector<uint8_t> target(Width * Height * 4u, 0);
    positions(source, true);
    DrawItem draw = make_draw(source, target);
    uint64_t texture_address = 0;
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    if (source_snapshot) {
        auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
        check(map && unmap &&
                  map(reinterpret_cast<uint64_t>(&texture_address), 0x10000, 0x2, 0,
                      reinterpret_cast<uint64_t>("snapshot-capture-test"), 0) == 0 && texture_address,
              "snapshot capture maps real guest-readable encoded texture sources");
        if (!texture_address) return 1;
        // Two independent encoded sources establish two actual handoffs across ordered spans.
        // Host-backed replay inputs are deliberately absent: they cannot enter this guest cache.
        std::fill_n(reinterpret_cast<uint32_t*>(texture_address), 8, 0xc00ffc00u);
        sample_packed_texture(draw, texture_address);
    }
    check(!draw.vs.empty() && !draw.fs.empty(), "real readonly vertex-fetch shaders compile");
    if (failures) return 1;

    const fs::path directory = source_snapshot
        ? (std::getenv("PROSPER_NO_TEXTURE_SOURCE_SNAPSHOT_MOVE")
            ? "render_source_snapshot_copy_capture_test" : "render_source_snapshot_move_capture_test")
        : "render_buffer_capture_test";
    std::error_code ec;
    fs::remove_all(directory, ec);
    fs::create_directories(directory, ec);
    check(!ec, "isolated capture directory is available");
    auto& capture = prosper::perf::interactive_performance_capture();
    capture.cancel();
    const uint64_t start = prosper::perf::monotonic_now_ns();
    prosper::perf::ProcessSample sample;
    sample.monotonic_ns = start;
    capture.observe_sample(sample);
    const auto armed = capture.arm(directory.string(), "TEST", "live buffer statistics",
                                   "test", start, std::chrono::system_clock::now());
    check(armed.ok && capture.detailed_timing_active(), "actual F8 singleton starts detailed capture");
    if (!armed.ok) return 1;

    std::vector<LiveRenderPhase> observed_phases;
    LiveRenderFn live = [&](const std::vector<DrawItem>& items, uint32_t w, uint32_t h) {
        observed_phases.push_back(live_render_phase());
        return RenderedFrame(render_submit_items(items, w, h));
    };
    unsigned separators = 0;
    LiveComputeFn separator = [&](const std::vector<ComputeItem>&) {
        ++separators;
        return true; // Only separates actual draw spans; no GPU/CPU statistics are fabricated.
    };
    const std::vector<SubmitOperation> operations{
        {SubmitOperationKind::Draw,0,10}, {SubmitOperationKind::Dispatch,0,20},
        {SubmitOperationKind::Draw,1,30}};
    DrawItem second_draw = draw;
    second_draw.draw_index = 1; // Ordered operations resolve by guest draw index, not vector position.
    if (source_snapshot) sample_packed_texture(second_draw, texture_address + 16);
    const auto split = execute_ordered_items(operations, {draw,second_draw}, {ComputeItem{}},
                                            live, separator, Width, Height);
    check(separators == 1 && split.render_spans == 2 && observed_phases.size() == 2 &&
              observed_phases[0].first_span && !observed_phases[0].final_span &&
              !observed_phases[1].first_span && observed_phases[1].final_span,
          "actual ordered executor brackets two renderer calls into one semantic record");
    check(solid(split.frame.bytes(), true), "cold and unchanged spans actually draw the green quad");
    if (source_snapshot)
        std::fill_n(reinterpret_cast<uint32_t*>(texture_address), 4, 0xc00003ffu);
    else
        positions(source, false); // Same source/identity/extent, changed guest-visible vertex positions.
    // A fresh colour destination starts from the blue clear. Reusing the previous destination
    // would correctly seed its prior green image, so offscreen geometry would leave green pixels.
    std::vector<uint8_t> changed_target(Width * Height * 4u, 0);
    draw.color0_base = reinterpret_cast<uintptr_t>(changed_target.data());
    const auto changed = render_submit_items({draw}, Width, Height);
    check(source_snapshot ? solid_red(changed) : solid(changed, false),
          "changed guest input actually changes the rendered pixels");
    const auto unchanged = render_submit_items({draw}, Width, Height);
    check(source_snapshot ? solid_red(unchanged) : solid(unchanged, false),
          "unchanged refreshed input still draws the correct result");

    // Drive only the process-sampler clock; never sleep or assert a speed threshold. Detailed
    // records came from actual renderer calls while the collector was armed.
    sample.monotonic_ns = std::max<uint64_t>(prosper::perf::monotonic_now_ns(), start + 5'000'000'000ull);
    capture.observe_sample(sample);
    prosper::perf::CaptureOutcome outcome;
    check(capture.take_outcome(outcome) && outcome.ok && outcome.renderer_records == 3 &&
              outcome.renderer_dropped == 0,
          "two-span cold submit and two later submits produce exactly three complete F8 records");
    std::ifstream input(outcome.path);
    std::vector<std::string> records;
    for (std::string line; std::getline(input, line);)
        if (line.find("\"type\":\"renderer\"") != std::string::npos) records.push_back(line);
    check(records.size() == 3, "the published capture contains the actual three renderer records");
    if (records.size() == 3 && source_snapshot) {
        constexpr std::array<uint64_t,3> snapshot_bytes{32,16,0};
        // The explicit negative-control arm keeps move expectations even when the runtime
        // switch restores copying. Pixel checks still run normally and must remain correct.
        const bool copy = !expect_snapshot_move &&
            std::getenv("PROSPER_NO_TEXTURE_SOURCE_SNAPSHOT_MOVE") != nullptr;
        for (size_t i = 0; i < records.size(); ++i) {
            const auto& r = records[i];
            check(integer(r, "callbacks") == (i == 0 ? 2u : 1u),
                  "snapshot counters aggregate two callbacks once and reset at the next submit");
            check(integer(r, "frontend_tex_source_snapshot_copied_bytes") ==
                      (copy ? snapshot_bytes[i] : 0),
                  "F8 records exact admitted snapshot copy bytes, including the copy control");
            check(integer(r, "frontend_tex_source_snapshot_transferred_bytes") ==
                      (copy ? 0 : snapshot_bytes[i]),
                  "F8 records exact admitted snapshot owner transfers and no work on cache hits");
            const double handoff_ms = milliseconds(r, "frontend_tex_source_snapshot_handoff_ms");
            check(i == 2 ? handoff_ms == 0 : handoff_ms > 0,
                  "real handoff timing reaches F8 and resets for the unchanged cached submit");
        }
    } else if (records.size() == 3) {
        constexpr std::array<uint64_t,3> copied{Bytes,Bytes,0};
        constexpr std::array<uint64_t,3> hits{1,0,1};
        constexpr std::array<uint64_t,3> admitted{Bytes,0,0};
        constexpr std::array<uint64_t,3> refreshed{0,Bytes,0};
        for (size_t i = 0; i < records.size(); ++i) {
            const auto& r = records[i];
            const double resident_ms = milliseconds(r, "res_buffer_resident_ms");
            check(resident_ms >= 0 && (i != 0 || resident_ms > 0),
                  "actual cold resident work reaches F8's separate timer, without a speed threshold");
            check(integer(r,"callbacks") == (i == 0 ? 2u : 1u), "pending callback count resets per semantic submit");
            check(integer(r,"buffer_upload_bytes") == copied[i], "F8 copied bytes match real cold/refresh/warm work");
            check(integer(r,"buffer_resident_hits") == hits[i], "F8 unchanged hits reach the semantic record once");
            check(integer(r,"buffer_resident_compared_bytes") == Bytes, "F8 comparison spans survive every backend boundary");
            check(integer(r,"buffer_resident_reused_bytes") == hits[i]*Bytes, "F8 reused bytes exclude changed refreshes");
            check(integer(r,"buffer_resident_admitted_bytes") == admitted[i], "F8 admission is counted only on the cold span");
            check(integer(r,"buffer_resident_refreshed_bytes") == refreshed[i], "F8 refresh is counted after old work completes");
            check(integer(r,"buffer_resident_declined_bytes") == 0 &&
                      integer(r,"buffer_resident_ineligible_bytes") == 0,
                  "the actual readonly fixture is admitted rather than silently falling back");
        }
    }
    capture.cancel();
    set_submit_renderer({});
    present_reset();
    if (texture_address) unmap(texture_address, 0x10000, 0, 0, 0, 0);
    fs::remove_all(directory, ec);
    return failures ? 1 : 0;
}
