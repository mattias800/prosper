// test_render_diagnostic_paths — #3248: run the env-gated render diagnostics at least once, so the
// validation layer can see them.
//
// WHY THIS EXISTS, and it is not really about the assertions below.
//
// `PROSPER_GEOM_PROBE` and `PROSPER_DRAW_ISO` record Vulkan commands that NO ctest case executed.
// So `tools/vkval` — the only check in the tree that sees spec misuse — was blind to them for a
// second, entirely separate reason on top of synchronization validation being off: the code never
// ran while the layer was loaded. Both were misusing Vulkan, and both misled their reader as a
// result:
//
//   * the geometry probe bound transform feedback around a draw whose last pre-rasterization stage
//     is not xfb-decorated (VUID-vkCmdBeginTransformFeedbackEXT-None-04128), captured nothing, and
//     printed "transform feedback wrote 0 vertices (draw produced no primitives)" — a WRONG answer
//     about a draw that did produce primitives;
//   * the draw-isolation re-render replayed each draw's pipeline and descriptors without any of the
//     dynamic state the main loop sets (VUID-vkCmdDraw-None-07831 / -07832), so the pass naming the
//     culprit draw was not the pass being isolated.
//
// The assertions here pin the arming decisions and prove the paths executed. The MISUSE itself is
// caught by running this test under `vk_validation_scan.py`, which is now possible at all because
// these paths run.
//
// The diagnostics are armed by ctest, NOT by this file. PROSPER_GEOM_PROBE is read through
// PROSPER_ENV_VALUE -- a one-shot cached read -- so a setenv() here would be a vacuous arm on any
// build where something reads it first, and `cached_env_arming_logic` refuses that pattern by name.
// Setting it in the test's ctest ENVIRONMENT puts it in place before the process starts, which is
// the only spelling that is correct regardless of read order. Running the binary directly therefore
// arms nothing, and the check below fails loudly rather than passing on an unexercised path.

#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"
#include "gpu/diagnostics/geometry_probe_arming.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using prosper::test::BackendDraw;
using prosper::test::FrameResource;

static int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::printf("  [FAIL] %s\n", message); ++failures; } \
    else               { std::printf("  [ok]   %s\n", message); } \
} while (0)

namespace {

std::vector<BackendDraw> one_triangle() {
    BackendDraw draw;
    draw.vs.assign(kTriVertSpv, kTriVertSpv + sizeof(kTriVertSpv) / 4);
    draw.fs.assign(kTriFragSpv, kTriFragSpv + sizeof(kTriFragSpv) / 4);
    draw.vcount = 3;
    for (uint32_t set = 0; set < 2; ++set) {
        FrameResource b2; b2.binding = 2; b2.set = set; draw.R.push_back(std::move(b2));
        FrameResource b3; b3.binding = 3; b3.set = set; draw.R.push_back(std::move(b3));
    }
    return {std::move(draw)};
}

uint64_t armed() {
    return prosper::test::backend_geom_probe_armed_count().load(std::memory_order_relaxed);
}
uint64_t undeclared() {
    return prosper::test::backend_geom_probe_undeclared_count().load(std::memory_order_relaxed);
}
uint64_t iso_passes() {
    return prosper::test::backend_draw_iso_pass_count().load(std::memory_order_relaxed);
}

// SPIR-V instruction word: (word_count << 16) | opcode. OpExecutionMode is 16, and its operands are
// <entry point id> <mode>, so the whole instruction is three words.
//
// HAND-BUILT on purpose. A positive control drawn from the same source as the null tests the
// discriminator and not the domain; the domain question here is "would this predicate recognise a
// module that IS decorated", and asking the recompiler to produce one would inherit whatever the
// recompiler believes about the decoration. These bytes believe nothing.
constexpr uint32_t kExecutionModeXfb = 11u;
constexpr uint32_t kExecutionModeOriginUpperLeft = 7u;
const uint32_t kXfbModule[] = {
    0x07230203u, 0x00010000u, 0x00080001u, 0x0000000Au, 0x00000000u,   // header
    0x00030010u, 0x00000004u, kExecutionModeXfb,                        // OpExecutionMode %4 Xfb
};
const uint32_t kPlainModule[] = {
    0x07230203u, 0x00010000u, 0x00080001u, 0x0000000Au, 0x00000000u,
    0x00030010u, 0x00000004u, kExecutionModeOriginUpperLeft,
};

}  // namespace

int main() {
    std::printf("== test_render_diagnostic_paths ==\n");

    // --- Arm 1: the arming predicate itself (no device, no render) ------------------------------
    CHECK(prosper::gpu::spirv_declares_xfb_capture(kXfbModule,
                                                   sizeof(kXfbModule) / sizeof(uint32_t)),
          "a module declaring OpExecutionMode Xfb is recognised (hand-built positive control)");
    CHECK(!prosper::gpu::spirv_declares_xfb_capture(kPlainModule,
                                                    sizeof(kPlainModule) / sizeof(uint32_t)),
          "a module with a different execution mode is not mistaken for a capture");
    CHECK(!prosper::gpu::spirv_declares_xfb_capture(nullptr, 0),
          "no words is a refusal, not a capture");
    {
        // Malformed input must refuse rather than loop or read past the end: the predicate decides
        // whether an instrument may trust its own output, so unreadable means no.
        const uint32_t zero_length[] = {0x07230203u, 0, 0, 0, 0, 0x00000010u};
        CHECK(!prosper::gpu::spirv_declares_xfb_capture(zero_length,
                                                        sizeof(zero_length) / sizeof(uint32_t)),
              "a zero-length instruction refuses instead of looping");
        const uint32_t overrun[] = {0x07230203u, 0, 0, 0, 0, 0x00FF0010u, 4, kExecutionModeXfb};
        CHECK(!prosper::gpu::spirv_declares_xfb_capture(overrun,
                                                        sizeof(overrun) / sizeof(uint32_t)),
              "an instruction running past the end refuses instead of reading out of bounds");
        const uint32_t not_spirv[] = {0xDEADBEEFu, 0, 0, 0, 0, 0x00030010u, 4, kExecutionModeXfb};
        CHECK(!prosper::gpu::spirv_declares_xfb_capture(not_spirv,
                                                        sizeof(not_spirv) / sizeof(uint32_t)),
              "a module without the SPIR-V magic is refused before it is walked");
    }

    // The render extent, and the arming ctest must agree with it: PROSPER_ISO_AT names the pixel
    // whose first lit submit triggers isolation, and that has to be inside this frame.
    const uint32_t W = 32, H = 32;
    const char* const iso_at = std::getenv("PROSPER_ISO_AT");
    const std::string want_iso_at = std::to_string(W / 2) + "," + std::to_string(H / 2);
    const bool armed_by_ctest =
        std::getenv("PROSPER_GEOM_PROBE") && std::getenv("PROSPER_DRAW_ISO") &&
        iso_at && want_iso_at == iso_at;
    CHECK(armed_by_ctest,
          "the diagnostics are armed (run this through ctest: it sets PROSPER_GEOM_PROBE, "
          "PROSPER_DRAW_ISO and PROSPER_ISO_AT in the test ENVIRONMENT)");
    if (!armed_by_ctest) {
        std::printf("== FAILED ==\n");
        return 1;
    }
    std::error_code frame_dir_error;
    const char* const frame_dir_env = std::getenv("PROSPER_FRAME_DIR");
    const std::filesystem::path frame_dir =
        frame_dir_env ? frame_dir_env : "render_diagnostic_paths_frames";
    std::filesystem::create_directories(frame_dir, frame_dir_error);

    const uint64_t before_armed = armed(), before_undeclared = undeclared();
    const uint64_t before_iso = iso_passes();
    const std::vector<uint8_t> pixels = prosper::test::render_draws_rgba(one_triangle(), W, H);
    CHECK(pixels.size() == static_cast<size_t>(W) * H * 4,
          "the diagnostic-armed pass still rendered a full frame");

    // --- Arm 2: the geometry probe refuses a shader it cannot capture ---------------------------
    // kTriVertSpv is hand-written fixture SPIR-V with no Xfb execution mode, which is exactly the
    // case that used to arm anyway. Both counters are checked: "did not arm" alone would also be
    // satisfied by a device with no VK_EXT_transform_feedback, which is a different reason and
    // would make this arm pass without testing anything.
    std::printf("  geom-probe: armed +%llu, refused-undeclared +%llu\n",
                static_cast<unsigned long long>(armed() - before_armed),
                static_cast<unsigned long long>(undeclared() - before_undeclared));
    CHECK(undeclared() > before_undeclared,
          "the probe recognised that the draw's shader declares no transform-feedback capture");
    CHECK(armed() == before_armed,
          "and refused to arm, rather than capturing nothing and calling it 'no primitives'");

    // --- Arm 3: the isolation re-render actually ran ---------------------------------------------
    // One baseline pass (kill = -1) plus one pass per draw. This proves the path executed, which is
    // what makes it visible to the validation layer at all; the dynamic state it now records is
    // asserted by `vk_validation_scan.py`, because no pixel or counter on a conformant driver can
    // tell a draw with undefined viewport state from a correct one.
    std::printf("  draw-iso re-render passes: %llu\n",
                static_cast<unsigned long long>(iso_passes() - before_iso));
    CHECK(iso_passes() == before_iso + 2,
          "draw isolation re-rendered the submit once as a baseline and once per killed draw");

    std::filesystem::remove_all(frame_dir, frame_dir_error);
    std::printf("== %s ==\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
