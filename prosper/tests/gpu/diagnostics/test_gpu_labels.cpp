// test_gpu_labels.cpp -- command labels against recording debug-utils entry points. No GPU.
//
// Proves WHAT is recorded: the label text, its colour class, and begin/end pairing. It cannot prove
// how RenderDoc or RGP display the result; that is a local look at a real capture (see the PR).
#include "gpu/diagnostics/gpu_breadcrumbs.hpp"
#include "gpu/diagnostics/gpu_labels_vk.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else { std::printf("  [ok]   %s\n", m); } } while (0)

namespace {

struct Event {
    bool begin = false;
    std::string name;
    float color[4] = {};
};
std::vector<Event> events;

VKAPI_ATTR void VKAPI_CALL mock_begin(VkCommandBuffer, const VkDebugUtilsLabelEXT* label) {
    Event e;
    e.begin = true;
    e.name = label->pLabelName ? label->pLabelName : "";
    for (int i = 0; i < 4; ++i) e.color[i] = label->color[i];
    events.push_back(e);
}
VKAPI_ATTR void VKAPI_CALL mock_end(VkCommandBuffer) { events.push_back(Event{}); }

const VkCommandBuffer kCmd = reinterpret_cast<VkCommandBuffer>(uintptr_t{0x1000});

BreadcrumbSite make_site(BreadcrumbKind kind, uint64_t submit, uint32_t local, uint32_t index) {
    BreadcrumbSite s;
    s.kind = kind; s.submit_no = submit; s.pass_local_index = local; s.draw_index = index;
    s.program_addr = 0xABC000; s.pipeline_hash = 0x1234;
    return s;
}

void label_text_and_pairing() {
    events.clear();
    LabelDispatch d;
    d.cmd_begin = mock_begin;
    d.cmd_end = mock_end;
    const BreadcrumbSite draw = make_site(BreadcrumbKind::draw, 42, 4, 17);
    gpu_label_begin(d, kCmd, draw);
    gpu_label_end(d, kCmd);
    CHECK(events.size() == 2 && events[0].begin && !events[1].begin, "a label is a begin then an end");
    CHECK(events[0].name == "draw submit=42 local=4 index=17 program=0xabc000 pipeline=0x1234",
          "the label names the submit, pass-local offset, index, program and pipeline");
    CHECK(events[0].name == format_breadcrumb_site(draw),
          "...and is exactly the text a breadcrumb verdict uses for the same site");
}

void shared_identity_with_verdicts() {
    BreadcrumbTable table;
    const BreadcrumbSite draw = make_site(BreadcrumbKind::draw, 9, 1, 3);
    const uint32_t id = table.begin_site(draw);
    const BreadcrumbVerdict v = resolve_breadcrumbs(table, id, 0);
    const std::string verdict = format_breadcrumb_verdict(v, id, 0);
    BreadcrumbSite stored;
    CHECK(table.lookup(id, stored), "the site is in the table");
    CHECK(verdict.find(format_breadcrumb_site(stored)) != std::string::npos,
          "a verdict line contains the same site text a label for that site carries");
}

void colours_separate_draws_from_dispatches() {
    events.clear();
    LabelDispatch d;
    d.cmd_begin = mock_begin;
    d.cmd_end = mock_end;
    gpu_label_begin(d, kCmd, make_site(BreadcrumbKind::draw, 1, 0, 0));
    gpu_label_begin(d, kCmd, make_site(BreadcrumbKind::dispatch, 1, 0, 0));
    CHECK(events.size() == 2 && events[0].name.rfind("draw", 0) == 0 && events[1].name.rfind("dispatch", 0) == 0,
          "kinds are spelled out in the text");
    // Exact values, not an ordering: an ordering test passes when the two classes' greens are swapped
    // so long as each stays on the right side of its blue, which is the mutation that proved it weak.
    CHECK(events[0].color[0] == 0.25f && events[0].color[1] == 0.55f && events[0].color[2] == 0.95f,
          "a draw is blue (0.25, 0.55, 0.95)");
    CHECK(events[1].color[0] == 0.25f && events[1].color[1] == 0.80f && events[1].color[2] == 0.35f,
          "a dispatch is green (0.25, 0.80, 0.35)");
    CHECK(events[0].color[3] == 1.0f && events[1].color[3] == 1.0f, "labels are opaque");
}

void unusable_dispatch_is_a_no_op() {
    events.clear();
    LabelDispatch none;
    CHECK(!none.usable(), "a table with no entry points is unusable");
    gpu_label_begin(none, kCmd, make_site(BreadcrumbKind::draw, 1, 0, 0));
    gpu_label_end(none, kCmd);
    LabelDispatch half;
    half.cmd_begin = mock_begin;
    CHECK(!half.usable(), "a table missing the end entry point is unusable (an unbalanced begin is worse than none)");
    gpu_label_begin(half, kCmd, make_site(BreadcrumbKind::draw, 1, 0, 0));
    CHECK(events.empty(), "nothing is recorded through an unusable table");
    gpu_label_begin(LabelDispatch{mock_begin, mock_end}, VK_NULL_HANDLE, make_site(BreadcrumbKind::draw, 1, 0, 0));
    CHECK(events.empty(), "a null command buffer is ignored");
}

void null_device_resolves_to_nothing() {
    CHECK(!label_dispatch_for(VK_NULL_HANDLE).usable(), "no device, no entry points");
    CHECK(!gpu_label_begin(VK_NULL_HANDLE, kCmd, make_site(BreadcrumbKind::draw, 1, 0, 0)),
          "the device-form begin reports that no label was opened, so no end is recorded");
}

} // namespace

int main() {
    std::printf("== label text and pairing ==\n");        label_text_and_pairing();
    std::printf("== shared identity with verdicts ==\n"); shared_identity_with_verdicts();
    std::printf("== draw vs dispatch colours ==\n");      colours_separate_draws_from_dispatches();
    std::printf("== unusable dispatch ==\n");             unusable_dispatch_is_a_no_op();
    std::printf("== null device ==\n");                   null_device_resolves_to_nothing();
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
