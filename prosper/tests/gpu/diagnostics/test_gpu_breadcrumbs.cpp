// test_gpu_breadcrumbs.cpp -- the breadcrumb core and emitter, with no GPU.
//
// WHAT THIS PROVES AND WHAT IT CANNOT. The verdict arithmetic is tested on synthetic marker values,
// and the emitter is tested against RECORDING functions substituted for the Vulkan entry points. That
// proves the emitter writes exactly the values and offsets the resolver expects. It proves NOTHING about
// whether a real AMD or NV driver executes those writes before a hang: stock lavapipe exposes neither
// marker extension, so no CI job can. That half is a local run on real hardware (see the PR).
#include "gpu/diagnostics/gpu_breadcrumbs.hpp"
#include "gpu/diagnostics/gpu_breadcrumbs_vk.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else { std::printf("  [ok]   %s\n", m); } } while (0)

namespace {

BreadcrumbSite site(uint64_t submit, uint32_t pass, uint32_t index, uint64_t program = 0xABC000,
                    uint64_t pipeline = 0x1234, BreadcrumbKind kind = BreadcrumbKind::draw) {
    BreadcrumbSite s;
    s.kind = kind; s.submit_no = submit; s.pass_local_index = pass; s.draw_index = index;
    s.program_addr = program; s.pipeline_hash = pipeline;
    return s;
}

// Record `n` consecutive draws (index 0..n-1) and return their ids, which are 1..n on a fresh table.
std::vector<uint32_t> record(BreadcrumbTable& table, int n) {
    std::vector<uint32_t> ids;
    for (int i = 0; i < n; ++i) ids.push_back(table.begin_site(site(7, 2, static_cast<uint32_t>(i))));
    return ids;
}

void ids_and_ring() {
    BreadcrumbTable table;
    const auto ids = record(table, 3);
    CHECK(ids[0] == 1 && ids[1] == 2 && ids[2] == 3, "ids start at 1 and count up");
    CHECK((ids[2] & kBreadcrumbAfterBit) == 0, "an id never carries the after bit");
    BreadcrumbSite out;
    CHECK(table.lookup(2, out) && out.draw_index == 1 && out.submit_no == 7 && out.pass_local_index == 2,
          "a site's identity round-trips through the table");
    CHECK(!table.lookup(0, out) && !table.lookup(99, out), "id 0 and an unissued id resolve to nothing");
    CHECK(!table.lookup(kBreadcrumbAfterBit | 1, out), "an after-bit value is not an id");

    // Overwrite the ring: a marker older than the ring must be reported missing, never mis-resolved to
    // the NEWER site that reused its slot.
    BreadcrumbTable big;
    for (uint32_t i = 0; i < kBreadcrumbRingSites + 5; ++i) big.begin_site(site(1, 0, i));
    CHECK(!big.lookup(1, out), "a site older than the ring is no longer found");
    CHECK(big.lookup(kBreadcrumbRingSites + 5, out) && out.draw_index == kBreadcrumbRingSites + 4,
          "the newest site is found");
    CHECK(big.sites_recorded() == kBreadcrumbRingSites + 5, "the recorded count is not capped by the ring");
}

void verdict_arithmetic() {
    BreadcrumbTable table;
    record(table, 10);
    using State = BreadcrumbVerdict::State;

    BreadcrumbVerdict v = resolve_breadcrumbs(table, 0, 0);
    CHECK(v.state == State::nothing_recorded, "no markers: nothing recorded");

    // Started 5, finished 4: exactly one site in flight, and it is site 5 (draw index 4).
    v = resolve_breadcrumbs(table, 5, breadcrumb_after_value(4));
    CHECK(v.state == State::in_flight && v.window_sites == 1, "started 5 / finished 4 is a window of ONE");
    CHECK(v.first_unfinished_known && v.first_unfinished.draw_index == 4,
          "...and the suspect is site 5, draw index 4");
    CHECK(v.last_finished_known && v.last_finished.draw_index == 3, "...with site 4 the last finished");

    // Started 8, finished 4: sites 5..8 were in flight together, so the verdict is a window of four and
    // names the FIRST unfinished, not a single culprit it cannot prove.
    v = resolve_breadcrumbs(table, 8, breadcrumb_after_value(4));
    CHECK(v.state == State::in_flight && v.window_sites == 4, "started 8 / finished 4 is a window of FOUR");
    CHECK(v.first_unfinished.draw_index == 4 && v.last_started.draw_index == 7,
          "...from site 5 through site 8");

    // Nothing finished yet: the window starts at the very first site.
    v = resolve_breadcrumbs(table, 3, 0);
    CHECK(v.state == State::in_flight && v.window_sites == 3 && v.first_unfinished.draw_index == 0 &&
              v.last_finished_id == 0 && !v.last_finished_known,
          "started 3 / finished none: window from site 1, no last-finished");

    v = resolve_breadcrumbs(table, 6, breadcrumb_after_value(6));
    CHECK(v.state == State::between_sites && v.last_finished.draw_index == 5,
          "started == finished: the GPU stopped OUTSIDE instrumented work");

    v = resolve_breadcrumbs(table, 3, breadcrumb_after_value(5));
    CHECK(v.state == State::inconsistent, "a site finished that never started is inconsistent, not a suspect");

    // Polarity: a before slot holding an after-value, or an after slot holding a plain id, is absent.
    v = resolve_breadcrumbs(table, breadcrumb_after_value(5), 5);
    CHECK(v.state == State::nothing_recorded, "wrong-polarity values are treated as absent, not trusted");

    // A site the ring no longer holds is reported as unknown rather than invented.
    BreadcrumbTable small;
    for (uint32_t i = 0; i < kBreadcrumbRingSites + 10; ++i) small.begin_site(site(1, 0, i));
    v = resolve_breadcrumbs(small, 5, breadcrumb_after_value(4));
    CHECK(v.state == State::in_flight && !v.first_unfinished_known,
          "a marker older than the ring resolves to an unknown site");
    CHECK(format_breadcrumb_verdict(v, 5, breadcrumb_after_value(4)).find("no longer in the ring") !=
              std::string::npos,
          "...and the report says so");
}

// The id counter wraps from kBreadcrumbMaxId back to 1, after which the NEWEST id is numerically the
// SMALLEST. Comparing ids with `>` or std::max then reads a healthy run as corrupt and picks the OLD
// checkpoint as the latest. Started three below the maximum, so the sequence crosses the wrap:
// ids M-2, M-1, M, 1, 2 for draw indexes 0..4.
void wraparound() {
    constexpr uint32_t M = kBreadcrumbMaxId;
    BreadcrumbTable table(M - 2);
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < 5; ++i) ids.push_back(table.begin_site(site(3, 0, i)));
    CHECK(ids[0] == M - 2 && ids[2] == M && ids[3] == 1 && ids[4] == 2,
          "the counter wraps from the maximum id to 1, never issuing 0");
    using State = BreadcrumbVerdict::State;

    // Started id 2 (draw 4), finished id M-1 (draw 1): sites M, 1, 2 are in flight, a window of THREE
    // even though 2 < M-1 numerically.
    BreadcrumbVerdict v = resolve_breadcrumbs(table, 2, breadcrumb_after_value(M - 1));
    CHECK(v.state == State::in_flight && v.window_sites == 3,
          "across the wrap, started 2 / finished M-1 is a window of THREE, not corruption");
    CHECK(v.first_unfinished_known && v.first_unfinished.draw_index == 2 &&
              v.last_started.draw_index == 4,
          "...from the site with id M (draw 2) through id 2 (draw 4)");

    v = resolve_breadcrumbs(table, 1, breadcrumb_after_value(M));
    CHECK(v.state == State::in_flight && v.window_sites == 1 && v.first_unfinished.draw_index == 3,
          "started 1 / finished M: exactly one site in flight, the first after the wrap");

    v = resolve_breadcrumbs(table, M - 1, breadcrumb_after_value(2));
    CHECK(v.state == State::inconsistent,
          "a finished marker far AHEAD of the started one is still inconsistent across the wrap");

    // A checkpoint SET spanning the wrap: the latest before is id 1 (after M), not M.
    const uint32_t values[] = {breadcrumb_before_value(M), breadcrumb_before_value(1),
                               breadcrumb_after_value(M - 1), breadcrumb_after_value(M)};
    uint32_t started = 0, finished = 0;
    reduce_checkpoint_values(values, 4, started, finished);
    CHECK(started == 1 && finished == breadcrumb_after_value(M),
          "checkpoint reduction picks the latest id around the ring, not the numerically greatest");
}

void checkpoint_reduction() {
    const uint32_t values[] = {breadcrumb_before_value(7), breadcrumb_after_value(5),
                               breadcrumb_before_value(6), breadcrumb_after_value(4), 0};
    uint32_t started = 99, finished = 99;
    reduce_checkpoint_values(values, 5, started, finished);
    CHECK(started == 7 && finished == breadcrumb_after_value(5),
          "a set of reached checkpoints reduces to its greatest before and greatest after");
    reduce_checkpoint_values(nullptr, 0, started, finished);
    CHECK(started == 0 && finished == 0, "an empty set reduces to nothing");
}

void report_text() {
    BreadcrumbTable table;
    table.begin_site(site(41, 3, 0, 0xDEAD0000, 0xBEEF, BreadcrumbKind::dispatch));
    table.begin_site(site(42, 4, 17, 0x1122334400ull, 0x99, BreadcrumbKind::draw));
    const BreadcrumbVerdict v = resolve_breadcrumbs(table, 2, breadcrumb_after_value(1));
    const std::string text = format_breadcrumb_verdict(v, 2, breadcrumb_after_value(1));
    CHECK(text.find("[gpu-breadcrumb]") == 0, "the report carries the greppable tag");
    CHECK(text.find("ONE site") != std::string::npos && text.find("draw submit=42 local=4 index=17") !=
              std::string::npos && text.find("program=0x1122334400") != std::string::npos &&
              text.find("pipeline=0x99") != std::string::npos,
          "it names the suspect's submit, pass, index, program and pipeline");
    CHECK(text.find("last finished: dispatch submit=41") != std::string::npos,
          "...and the last finished site, with its kind");
    CHECK(text.find("started=0x2") != std::string::npos && text.find("finished=0x80000001") != std::string::npos,
          "...and the raw markers it was derived from");
}

// ---- the emitter against recording entry points ----------------------------------------------------

struct Recorded {
    std::vector<uint32_t> amd_values;
    std::vector<VkDeviceSize> amd_offsets;
    std::vector<VkPipelineStageFlagBits> amd_stages;
    std::vector<uint32_t> checkpoints;
    std::vector<uint32_t> nv_reported;   // what the mock queue "returns" after a loss
    std::vector<VkPipelineStageFlagBits> nv_stages;   // stage per entry; BOTTOM_OF_PIPE when absent
};
Recorded rec;

VKAPI_ATTR void VKAPI_CALL mock_write_marker(VkCommandBuffer, VkPipelineStageFlagBits stage, VkBuffer,
                                             VkDeviceSize offset, uint32_t marker) {
    rec.amd_stages.push_back(stage);
    rec.amd_offsets.push_back(offset);
    rec.amd_values.push_back(marker);
}
VKAPI_ATTR void VKAPI_CALL mock_set_checkpoint(VkCommandBuffer, const void* marker) {
    rec.checkpoints.push_back(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(marker)));
}
VKAPI_ATTR void VKAPI_CALL mock_get_checkpoint_data(VkQueue, uint32_t* count, VkCheckpointDataNV* data) {
    if (!data) { *count = static_cast<uint32_t>(rec.nv_reported.size()); return; }
    for (uint32_t i = 0; i < *count && i < rec.nv_reported.size(); ++i)
    {
        data[i].pCheckpointMarker =
            reinterpret_cast<void*>(static_cast<uintptr_t>(rec.nv_reported[i]));
        data[i].stage = i < rec.nv_stages.size() ? rec.nv_stages[i]
                                                  : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }
}
VKAPI_ATTR VkResult VKAPI_CALL mock_fault_info(VkDevice, VkDeviceFaultCountsEXT* counts,
                                               VkDeviceFaultInfoEXT* info) {
    counts->addressInfoCount = 1;
    counts->vendorInfoCount = 1;
    if (!info) return VK_SUCCESS;
    std::snprintf(info->description, sizeof info->description, "mock page fault");
    info->pAddressInfos[0].addressType = VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT;
    info->pAddressInfos[0].reportedAddress = 0xCAFE000;
    info->pAddressInfos[0].addressPrecision = 0x1000;
    info->pVendorInfos[0].vendorFaultCode = 0x77;
    info->pVendorInfos[0].vendorFaultData = 0x5;
    std::snprintf(info->pVendorInfos[0].description, sizeof info->pVendorInfos[0].description, "mock vendor");
    return VK_SUCCESS;
}

const VkCommandBuffer kCmd = reinterpret_cast<VkCommandBuffer>(uintptr_t{0x1000});
const VkQueue kQueue = reinterpret_cast<VkQueue>(uintptr_t{0x2000});
const VkDevice kDev = reinterpret_cast<VkDevice>(uintptr_t{0x9});     // the device configure() arms
const VkDevice kOtherDev = reinterpret_cast<VkDevice>(uintptr_t{0x7});
const VkBuffer kBuffer = reinterpret_cast<VkBuffer>(uintptr_t{0x3000});

void inactive_emitter_does_nothing() {
    breadcrumb_table().reset();
    rec = {};
    BreadcrumbEmitter emitter;
    CHECK(!emitter.active() && emitter.begin(kDev, kCmd, site(1, 0, 0)) == 0,
          "an unconfigured emitter is inactive and begin() returns id 0");
    emitter.end(kCmd, 0);
    CHECK(rec.amd_values.empty() && rec.checkpoints.empty() && breadcrumb_table().sites_recorded() == 0,
          "...and records nothing: no write, no table entry");
    const std::string hint = emitter.report_device_loss(kDev, kQueue);
    CHECK(hint.find("not armed") != std::string::npos && hint.find("PROSPER_GPU_BREADCRUMBS=1") != std::string::npos,
          "...and on a device loss says it was not armed and how to arm it, instead of staying silent");
}

void amd_path() {
    breadcrumb_table().reset();
    rec = {};
    uint32_t slots[2] = {0, 0};
    BreadcrumbDispatch d;
    d.cmd_write_buffer_marker = mock_write_marker;
    d.get_device_fault_info = mock_fault_info;
    BreadcrumbEmitter emitter;
    emitter.configure(BreadcrumbMode::amd_buffer_marker, d, kDev,
                      kBuffer, slots, /*device_fault_enabled=*/true);

    const uint32_t a = emitter.begin(kDev, kCmd, site(5, 1, 0));
    emitter.end(kCmd, a);
    const uint32_t b = emitter.begin(kDev, kCmd, site(5, 1, 1));
    emitter.end(kCmd, b);
    const uint32_t c = emitter.begin(kDev, kCmd, site(5, 1, 2));
    CHECK(a == 1 && b == 2 && c == 3, "three sites get ids 1, 2, 3");
    CHECK(rec.amd_values.size() == 5, "begin, end, begin, end, begin write five markers");
    CHECK(rec.amd_values[0] == 1 && rec.amd_values[1] == breadcrumb_after_value(1) &&
              rec.amd_values[4] == 3,
          "values are the id before and the id with the after bit after");
    CHECK(rec.amd_offsets[0] == 0 && rec.amd_offsets[1] == 4 && rec.amd_offsets[2] == 0 &&
              rec.amd_offsets[3] == 4,
          "before markers go to slot 0 (offset 0), after markers to slot 1 (offset 4)");
    CHECK(rec.amd_stages[0] == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT &&
              rec.amd_stages[1] == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
          "before is written at TOP_OF_PIPE, after at BOTTOM_OF_PIPE");

    // The "GPU" got as far as the third site's start and the second site's end, then stopped.
    slots[0] = 3;
    slots[1] = breadcrumb_after_value(2);
    const auto read = emitter.read_back(kQueue);
    CHECK(read.ok && read.started == 3 && read.finished == breadcrumb_after_value(2),
          "the marker slots read back as written");
    const std::string report = emitter.report_device_loss(kDev, kQueue);
    CHECK(report.find("ONE site") != std::string::npos && report.find("index=2") != std::string::npos,
          "the device-loss report names the third draw as the one in flight");
    CHECK(report.find("[gpu-fault] driver: mock page fault") != std::string::npos &&
              report.find("reported=0xcafe000") != std::string::npos &&
              report.find("vendor code=0x77") != std::string::npos,
          "VK_EXT_device_fault's account follows the verdict");
    emitter.end(kCmd, 0);
    CHECK(rec.amd_values.size() == 5, "end(0) is a no-op");
}

void nv_path() {
    breadcrumb_table().reset();
    rec = {};
    BreadcrumbDispatch d;
    d.cmd_set_checkpoint = mock_set_checkpoint;
    d.get_queue_checkpoint_data = mock_get_checkpoint_data;
    BreadcrumbEmitter emitter;
    emitter.configure(BreadcrumbMode::nv_checkpoint, d, kDev,
                      VK_NULL_HANDLE, nullptr, false);

    const uint32_t a = emitter.begin(kDev, kCmd, site(9, 0, 10, 0x500, 0x6));
    emitter.end(kCmd, a);
    const uint32_t b = emitter.begin(kDev, kCmd, site(9, 0, 11, 0x600, 0x7));
    CHECK(rec.checkpoints.size() == 3 && rec.checkpoints[0] == 1 &&
              rec.checkpoints[1] == breadcrumb_after_value(1) && rec.checkpoints[2] == 2,
          "NV writes the same marker values as checkpoints");

    // The queue reports what it reached: the first site finished, the second only started.
    rec.nv_reported = {breadcrumb_before_value(b), breadcrumb_after_value(a), breadcrumb_before_value(a)};
    const std::string report = emitter.report_device_loss(kDev, kQueue);
    CHECK(report.find("ONE site") != std::string::npos && report.find("index=11") != std::string::npos &&
              report.find("program=0x600") != std::string::npos,
          "the reached checkpoints resolve to the second draw as the one in flight");
    CHECK(report.find("[gpu-fault]") == std::string::npos, "no fault section when VK_EXT_device_fault is off");

    // An "after" checkpoint seen only at TOP_OF_PIPE means the command processor got past the draw,
    // not that it finished, so the first site is still open: two sites are in flight, not one.
    rec.nv_reported = {breadcrumb_before_value(b), breadcrumb_after_value(a)};
    rec.nv_stages = {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    CHECK(emitter.report_device_loss(kDev, kQueue).find("WINDOW of 2 sites") != std::string::npos,
          "an after checkpoint short of BOTTOM_OF_PIPE does not close its site: both are in flight");
    rec.nv_stages.clear();

    // A second device in the process (a private compute device) is not the armed one: it records
    // nothing through this emitter and its loss reports that, instead of reading the armed device's
    // checkpoint list through the other device's entry points.
    const size_t recorded = rec.checkpoints.size();
    CHECK(emitter.begin(kOtherDev, kCmd, site(9, 0, 12)) == 0 && rec.checkpoints.size() == recorded,
          "begin for a device that is not the armed one records nothing");
    CHECK(emitter.report_device_loss(kOtherDev, kQueue).find("not the armed one") != std::string::npos,
          "a loss on the other device says its work carried no markers");

    rec.nv_reported.clear();
    CHECK(emitter.report_device_loss(kDev, kQueue).find("NO marker reached memory") != std::string::npos,
          "an empty checkpoint list reports that nothing was recorded, not a suspect");
}

void support_selection() {
    BreadcrumbDeviceSupport s;
    CHECK(s.chosen() == BreadcrumbMode::off, "a device advertising neither extension arms nothing");
    CHECK(s.note_extension("VK_KHR_swapchain") == nullptr && s.chosen() == BreadcrumbMode::off,
          "an unrelated extension is ignored");
    CHECK(s.note_extension(VK_AMD_BUFFER_MARKER_EXTENSION_NAME) != nullptr &&
              s.chosen() == BreadcrumbMode::amd_buffer_marker,
          "VK_AMD_buffer_marker selects the AMD path");
    CHECK(s.note_extension(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME) != nullptr &&
              s.chosen() == BreadcrumbMode::nv_checkpoint,
          "when both are advertised NV checkpoints win (no buffer needed)");
}

} // namespace

int main() {
    std::printf("== ids and ring ==\n");               ids_and_ring();
    std::printf("== verdict arithmetic ==\n");         verdict_arithmetic();
    std::printf("== id wraparound ==\n");              wraparound();
    std::printf("== checkpoint reduction ==\n");       checkpoint_reduction();
    std::printf("== report text ==\n");                report_text();
    std::printf("== inactive emitter ==\n");           inactive_emitter_does_nothing();
    std::printf("== AMD buffer marker path ==\n");     amd_path();
    std::printf("== NV checkpoint path ==\n");         nv_path();
    std::printf("== extension selection ==\n");        support_selection();
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
