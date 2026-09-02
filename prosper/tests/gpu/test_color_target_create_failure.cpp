// test_color_target_create_failure -- #3180: a failed color-target vkCreateImage must say so.
//
// `render_draw_pass_rgba` creates its color attachments at six sites: a primary and a
// budget-miss-fallback create for slot 0, for slot 1, and for MRT slots 2..7. Every one of them
// already guarded the resulting HANDLE before using it, so unlike #3045's two sites none fed a
// null image into a Vulkan call. They still discarded the `VkResult`, which left the pass dropping
// silently: an empty frame and nothing in the run log, so a device refusing a 4K attachment and an
// empty draw list produced identical evidence.
//
// Each arm here arms ONE site, renders a configuration that reaches it, and requires the run log to
// name that site and the real VkResult. Without the fix nothing is printed and every log assertion
// fails; the "pass is dropped" assertions alone would pass on unfixed code, which is exactly why
// they are not the contract this test exists for.
//
// The failure is injected rather than provoked, for the reason #3045 recorded: these are ordinary
// 2D color attachments at the test's own 64x64 extent, so no real device can be made to refuse one
// on demand.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/state/render_state.hpp"
#include "fixtures/render_runner.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

using namespace prosper::gpu;
using prosper::test::RenderColorTargetCreateSite;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

static bool has(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

// Run `body` with stderr redirected to a scratch file and return everything it printed. stderr is
// not restored; every assertion in this test reports on stdout.
template <typename Body>
static std::string capture_stderr(const char* scratch, Body&& body) {
    std::fflush(stderr);
    if (!std::freopen(scratch, "w+", stderr)) {
        printf("  [FAIL] cannot redirect stderr\n"); fails++; return {};
    }
    body();
    std::fflush(stderr);
    std::string text;
    if (FILE* f = std::fopen(scratch, "rb")) {
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
        std::fclose(f);
    }
    std::remove(scratch);
    return text;
}

int main() {
    printf("== test_color_target_create_failure (#3180) ==\n");

    // Must precede every render: persistent_color_target_limit() memoizes the environment on its
    // first call. A zero budget makes every persistent target miss and take its FALLBACK create,
    // which is the only way the three fallback sites are reachable at all.
    set_env("PROSPER_BACKEND_TARGET_CACHE_MB", "0");

    const uint32_t W = 64, H = 64;
    #include "../tools/boot_trace/refvs.inc"
    const std::vector<uint32_t> vs(kRefVs, kRefVs + sizeof(kRefVs) / 4);
    // Solid red, exported to MRT0. Inline consts: 0xF2 = 1.0f, 0x80 = 0.0f.
    static const uint32_t kRedPs[] = {0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,
                                      0xF800180Fu, 0x03020100u, 0xBF810000u};
    const std::vector<uint32_t> fs = recompile_fragment(kRedPs, sizeof(kRedPs) / 4, nullptr);
    CHECK(!vs.empty() && !fs.empty(), "fullscreen VS + solid-red PS available");
    if (vs.empty() || fs.empty()) { printf("FAILED (%d)\n", fails); return 1; }

    ResolvedPipelineState opaque{};
    opaque.topology = 3 /*VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST*/;
    opaque.color_write_mask = 0xF;
    const float black[4] = {0, 0, 0, 1};

    // One render, configured to reach `site`'s creation. `persistent` arms the budget-miss path.
    struct Arm {
        RenderColorTargetCreateSite site;
        const char* name;          // as it must appear in the log
        uint32_t color_count;      // 1 -> slot 0 only, 2 -> +slot 1, 3 -> +MRT slot 2
        bool persistent;           // set the identities that route through the budget check
        uint64_t id_base;          // distinct per arm so the target cache cannot answer from before
    };
    const Arm arms[] = {
        {RenderColorTargetCreateSite::Slot0,             "slot0",               1, false, 0},
        {RenderColorTargetCreateSite::Slot0Fallback,     "slot0-fallback",      1, true,  0x1100},
        {RenderColorTargetCreateSite::Slot1,             "slot1",               2, false, 0},
        {RenderColorTargetCreateSite::Slot1Fallback,     "slot1-fallback",      2, true,  0x2200},
        {RenderColorTargetCreateSite::SlotExtra,         "slot-extra",          3, false, 0},
        {RenderColorTargetCreateSite::SlotExtraFallback, "slot-extra-fallback", 3, true,  0x3300},
    };

    auto run = [&](const Arm& arm, bool inject, std::vector<uint8_t>* pixels) -> std::string {
        char scratch[128];
        std::snprintf(scratch, sizeof scratch, "test_color_target_create_failure_%s_%d.log",
                      arm.name, inject ? 1 : 0);
        return capture_stderr(scratch, [&] {
            prosper::test::BackendDraw d;
            d.vs = vs; d.fs = fs; d.ps = &opaque; d.vcount = 3;
            prosper::test::BackendColorTarget target{};
            if (arm.persistent) {
                target.persistent_id = arm.id_base + 1;
                target.persistent_id1 = arm.id_base + 2;
                target.persistent_id_slots[2] = arm.id_base + 3;
            }
            prosper::test::BackendMrtOutputs outputs;
            outputs.color_count = arm.color_count;
            if (inject) prosper::test::inject_render_color_target_create_failure_once(arm.site);
            *pixels = prosper::test::render_draws_rgba(
                {d}, W, H, nullptr, black, false, arm.persistent ? &target : nullptr,
                nullptr, black, nullptr, nullptr, true, &outputs);
        });
    };

    for (const Arm& arm : arms) {
        printf("  -- site %s --\n", arm.name);

        // Control FIRST: the same configuration with nothing armed must render and stay silent.
        // Without it, an arm that reached the site by accident (or never reached it, and printed
        // from some other site) would be indistinguishable from one that worked.
        std::vector<uint8_t> control_px;
        const std::string control_log = run(arm, false, &control_px);
        char msg[192];
        std::snprintf(msg, sizeof msg,
                      "%s: an uninjected run renders a frame", arm.name);
        CHECK(control_px.size() == static_cast<size_t>(W) * H * 4, msg);
        std::snprintf(msg, sizeof msg,
                      "%s: an uninjected run logs no create failure", arm.name);
        CHECK(!has(control_log, "[color-target-create-failed]"), msg);

        std::vector<uint8_t> failed_px;
        const std::string failure_log = run(arm, true, &failed_px);

        // THE CONTRACT. Unfixed code prints nothing at all, so all three of these go red.
        std::string expect_site = std::string("[color-target-create-failed] site=") + arm.name + " ";
        std::snprintf(msg, sizeof msg, "%s: the failure names ITS OWN site", arm.name);
        CHECK(has(failure_log, expect_site.c_str()), msg);
        std::snprintf(msg, sizeof msg,
                      "%s: the failure reports the real VkResult, not just 'a null handle'",
                      arm.name);
        // VK_ERROR_OUT_OF_DEVICE_MEMORY == -2, what the injector simulates.
        CHECK(has(failure_log, "vkCreateImage result=-2"), msg);
        std::snprintf(msg, sizeof msg, "%s: the failure reports the requested extent", arm.name);
        CHECK(has(failure_log, "extent=64x64"), msg);

        // The pre-existing control flow is unchanged: the pass is still dropped, not continued
        // with a null image. (True on unfixed code too -- stated so a future change that made the
        // report by CONTINUING past the failure would be caught.)
        std::snprintf(msg, sizeof msg, "%s: the pass is still dropped, not continued", arm.name);
        CHECK(failed_px.empty(), msg);
    }

    // Site selectivity and one-shot semantics: arming slot1 must not fire at slot0, and the arm
    // must be spent by the render that consumed it. Without selectivity every arm above would pass
    // on whichever site happened to run first.
    {
        std::vector<uint8_t> px;
        const std::string log = run(arms[0] /* slot0 config, color_count 1 */, false, &px);
        (void)log;
        prosper::test::inject_render_color_target_create_failure_once(
            RenderColorTargetCreateSite::Slot1);
        std::vector<uint8_t> unaffected;
        const std::string slot0_log = capture_stderr(
            "test_color_target_create_failure_selectivity.log", [&] {
                prosper::test::BackendDraw d;
                d.vs = vs; d.fs = fs; d.ps = &opaque; d.vcount = 3;
                prosper::test::BackendMrtOutputs outputs;   // color_count 1 -> slot 1 never created
                unaffected = prosper::test::render_draws_rgba(
                    {d}, W, H, nullptr, black, false, nullptr, nullptr, black, nullptr, nullptr,
                    true, &outputs);
            });
        CHECK(unaffected.size() == static_cast<size_t>(W) * H * 4,
              "a slot-1 arm does not fire at the slot-0 site");
        CHECK(!has(slot0_log, "[color-target-create-failed]"),
              "...and prints nothing while it stays armed");
        // Still armed: disarm it so it cannot leak into anything later.
        CHECK(prosper::test::consume_render_color_target_create_failure(
                  RenderColorTargetCreateSite::Slot1),
              "the unfired arm is still pending and consumes exactly once");
        CHECK(!prosper::test::consume_render_color_target_create_failure(
                  RenderColorTargetCreateSite::Slot1),
              "...and is spent after one consume");
        CHECK(!prosper::test::consume_render_color_target_create_failure(
                  RenderColorTargetCreateSite::None),
              "the None site never fires");
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
