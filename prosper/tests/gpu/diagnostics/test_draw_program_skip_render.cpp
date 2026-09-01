// PROSPER_SKIP_DRAW_PROGRAM, end to end through the REAL live renderer.
//
// test_draw_program_skip pins the selector's logic. This pins the WIRING: that the live renderer
// actually consults it, that a named program's draws are withheld from the GPU, and — the half that
// a logic test cannot show — that an unnamed draw submitted in the same process still renders.
//
// The wiring is the part that can be silently wrong. The renderer already carries three per-draw
// substitution switches (REFVS, TESTPS, FS_SPV) whose exact-match gates fail closed, plus a
// draw_index skip; a program skip placed in the wrong branch, or short-circuited by the shared-words
// accessor trap in DrawItem (#1434), would leave every draw rendering while the log claimed
// otherwise. So the assertion is on the rendered pixels and on the selector's own skip counter, in
// one process, with one submit per arm.
#include "gpu/diagnostics/draw_program_skip.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/present/videoout_present.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

namespace {

constexpr uint32_t W = 64, H = 64;
// A guest presentation surface LARGER than the render extent, exactly as test_gpu_capture_render
// does: render_submit_items then consumes the last pass at its own extent rather than under a
// present-extent contract it was never given.
constexpr uint32_t PRESENT_W = 128, PRESENT_H = 128;
// Two program addresses in the shape a real title uses. Only the first is named by the selector.
constexpr uint64_t kDeclinedVs = 0x5006c6a00ull;
constexpr uint64_t kKeptVs     = 0x5006c7100ull;

using Hle8Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t, uint64_t);

// A fullscreen-ish triangle from constants, and a solid magenta pixel shader — the same synthetic
// pair the renderer's own PROSPER_RENDER_TESTPS override uses, so neither stage depends on a
// resource table, a vertex buffer or guest memory being mapped.
DrawItem make_draw(uint64_t vs_addr) {
    static const uint32_t vs_rdna[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u,
        0xBF810000u,
    };
    static const uint32_t magenta_ps[] = {
        0x7E0002F2u, 0x7E020280u, 0x7E0402F2u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    DrawItem item;
    item.vs = recompile_vertex(vs_rdna, std::size(vs_rdna));
    item.fs = recompile_fragment(magenta_ps, std::size(magenta_ps), nullptr);
    item.vertex_count = 3;
    item.ps.topology = 3;              // triangle list
    item.ps.color_write_mask = 0xF;
    item.color0_base = 0x200000;
    item.color0_width = W;
    item.color0_height = H;
    item.vs_guest_addr = vs_addr;
    item.fs_guest_addr = vs_addr + 0x1000ull;
    return item;
}

// The magenta the test PS writes, sampled at the frame centre.
bool centre_is_magenta(const std::vector<uint8_t>& rgba) {
    if (rgba.size() != static_cast<size_t>(W) * H * 4) return false;
    const uint8_t* px = &rgba[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
    return px[0] > 0xC0 && px[1] < 0x40 && px[2] > 0xC0;
}

}  // namespace

int main() {
    std::printf("== test_draw_program_skip_render ==\n");

    // Armed BEFORE the first render: the selector is a process singleton read once, which is what
    // makes a run's log unambiguous. A test that armed it later would be asserting on a disarmed
    // selector and passing for the wrong reason.
    // BOTH variables come from the ctest ENVIRONMENT property, not from a setenv here, and that is
    // deliberate rather than incidental:
    //
    //   * PROSPER_SKIP_DRAW_PROGRAM is read once into a function-local static on first use. Arming
    //     it from the environment the process starts with removes the ordering argument entirely.
    //   * PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS is read through PROSPER_ENV_VALUE, which CACHES.
    //     A test that setenv'd it would trip cached_env_arming_logic — correctly, because a cached
    //     read that ran first would make every pixel assertion below vacuous while printing [ok].
    //
    // The renderer keeps intermediate colour targets GPU-resident and defers readback by default, so
    // without that opt-out a pass's pixels are simply not on the CPU for this test to assert on. It
    // does not change the thing under test: the decline happens while the backend draw list is being
    // assembled, upstream of where the pixels end up living. It does mean the pixel arms below cover
    // the CPU-readback path only; the selector's effect on the persistent-target path is covered by
    // the skip counter, which is path-independent.
    const char* skip_spec = std::getenv("PROSPER_SKIP_DRAW_PROGRAM");
    const char* no_persistent = std::getenv("PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS");
    if (!skip_spec || !no_persistent) {
        std::printf("  [FAIL] run me through ctest, or with:\n"
                    "         PROSPER_SKIP_DRAW_PROGRAM=0x5006c6a00 "
                    "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1 %s\n",
                    "test_draw_program_skip_render");
        return 1;
    }
    CHECK(std::strcmp(skip_spec, "0x5006c6a00") == 0,
          "the environment names the program this test expects to be declined");

    // The live renderer publishes through the present layer, so it needs a guest presentation
    // surface to publish INTO. Without one it renders and then reports "Vulkan render FAILED",
    // which would make every arm below vacuously true for a reason that has nothing to do with the
    // selector -- the exact shape the positive control exists to catch.
    prosper::register_builtin_hle();
    present_reset();
    auto open = prosper::Hle::lookup(prosper::nid_hash("sceVideoOutOpen"));
    auto set_buffer_attribute2 = reinterpret_cast<Hle8Fn>(prosper::Hle::lookup("PjS5uASwcV8"));
    auto register_buffers2 = prosper::Hle::lookup("rKBUtgRrtbk");
    std::vector<uint8_t> scanout0(static_cast<size_t>(PRESENT_W) * PRESENT_H * 4u);
    std::vector<uint8_t> scanout1(static_cast<size_t>(PRESENT_W) * PRESENT_H * 4u);
    std::vector<uint8_t> scanout2(static_cast<size_t>(PRESENT_W) * PRESENT_H * 4u);
    uint8_t scanout_attr[0x50]{};
    struct VideoBuffer { const void* data; const void* metadata; const void* reserved[2]; };
    VideoBuffer scanouts[3] = {
        {scanout0.data(), nullptr, {nullptr, nullptr}},
        {scanout1.data(), nullptr, {nullptr, nullptr}},
        {scanout2.data(), nullptr, {nullptr, nullptr}},
    };
    const uint64_t video_handle = open ? open(0, 0, 0, 0, 0, 0) : 0;
    if (set_buffer_attribute2)
        set_buffer_attribute2(reinterpret_cast<uint64_t>(scanout_attr), 0x8000000000000000ull,
                              0, PRESENT_W, PRESENT_H, 0, 0, 0);
    const uint64_t register_result = register_buffers2
        ? register_buffers2(video_handle, 0, 0, reinterpret_cast<uint64_t>(scanouts), 3,
                            reinterpret_cast<uint64_t>(scanout_attr))
        : UINT64_MAX;
    CHECK(open && set_buffer_attribute2 && register_buffers2 && register_result == 0 &&
              present_width() == PRESENT_W && present_height() == PRESENT_H,
          "the test registers a presentation surface for the live renderer to publish into");

    prosper::frontend::register_live_renderer(".", false);

    const DrawItem kept = make_draw(kKeptVs);
    CHECK(!kept.vs.empty() && !kept.fs.empty(), "the synthetic VS/PS pair recompiles to SPIR-V");

    // The named program FIRST, before anything has rendered. Order matters: the present layer serves
    // the last retained frame when a submit publishes nothing, so a declined submit run after a
    // successful one shows the PREVIOUS frame's pixels and looks like it rendered. That is real
    // behaviour and one of the documented limits of this instrument — it is asserted here in the one
    // arrangement where the retained frame cannot supply the answer.
    const std::vector<uint8_t> declined_pixels = render_submit_items({make_draw(kDeclinedVs)}, W, H);
    CHECK(draw_program_skip_selector().armed() && draw_program_skip_selector().size() == 1,
          "the selector armed from the environment on exactly one program");
    CHECK(draw_program_skip_selector().skipped_total() == 1,
          "the named program's draw is declined exactly once");
    CHECK(!centre_is_magenta(declined_pixels),
          "the declined draw paints nothing -- its pixels never reach the target");

    // POSITIVE CONTROL, and the arm that makes the one above mean anything: the same shaders, the
    // same submit shape, one different program address. Without it, "the named program produced no
    // pixels" is equally consistent with this environment producing no pixels for anything — which
    // is precisely how a skip instrument reports a clean result it never earned.
    const std::vector<uint8_t> kept_pixels = render_submit_items({kept}, W, H);
    CHECK(centre_is_magenta(kept_pixels),
          "a draw whose program is NOT named renders through the live backend");
    CHECK(draw_program_skip_selector().skipped_total() == 1,
          "the unnamed draw was not declined");

    // A mixed submit: naming a program must decline THAT draw, not the batch. This separates a
    // working selector from one that poisons the whole submit — which is how a renderer-side
    // rejection already behaves, and would look identical in a screenshot.
    const std::vector<uint8_t> mixed_pixels =
        render_submit_items({make_draw(kDeclinedVs), kept}, W, H);
    CHECK(draw_program_skip_selector().skipped_total() == 2,
          "the mixed submit declines only the named draw");
    CHECK(centre_is_magenta(mixed_pixels),
          "the unnamed draw in a mixed submit still renders");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
