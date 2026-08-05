// test_present_extent — the publish extent contract (#1986).
//
// The defect this guards: the live renderer chose the frame it hands to the publish gate by target
// IDENTITY alone (the flipped front buffer > any registered scanout > the last non-empty RGBA8 pass
// of the submit), and the gate publishes only a frame of exactly width*height*4 bytes. Sonic
// Frontiers (#1968) renders internally at 1920x1080 and presents at 3840x2160, so post-intro submits
// in which no pass targeted a scanout returned an internal pass — 1920x1080, 1024x1024 and 3840x3072
// were all observed in one run — and every frame was discarded. 13,028 submits and 8,223 folded draws
// produced no publish and no explanation.
//
// The second half was worse: the net that holds the last good scanout across VideoOut buffer rotation
// (Bendy and the Ink Machine, PPSA27616) tested `!empty()` rather than the publish predicate, so the
// first wrong-extent frame OVERWROTE the retained good frame and the recovery branch became
// unreachable for the rest of the process. That is why the wall was permanent, not a flicker.
//
// Both halves are pure functions of byte counts, so they are asserted here with no Vulkan device, no
// renderer registration and no guest. `want_bytes == 0` (no contract) is exercised as its own arm
// because it is a real path — gpu_replay's ordered-prefix inspection and the render_submit_items
// tests consume the last pass at its own extent — not a degenerate one.
#include "present_extent.hpp"

#include <cstdio>
#include <string_view>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); ++fails; } \
                              else         { std::printf("  [ok]   %s\n", msg); } } while (0)

namespace {

using prosper::frontend::PresentSourceChoice;
using prosper::frontend::RetainedFrameAction;
using prosper::frontend::present_frame_publishable;
using prosper::frontend::present_source_demoted;
using prosper::frontend::present_source_name;
using prosper::frontend::retained_frame_action;
using prosper::frontend::select_present_source;

constexpr size_t bytes_for(size_t w, size_t h) { return w * h * 4; }

// The measured Frontiers configuration: a 3840x2160 present, and the three extents that were actually
// selected as its present source over one run.
constexpr size_t k4k     = bytes_for(3840, 2160);
constexpr size_t k1080p  = bytes_for(1920, 1080);
constexpr size_t k1024sq = bytes_for(1024, 1024);
constexpr size_t k4kTall = bytes_for(3840, 3072);

} // namespace

int main() {
    std::printf("== test_present_extent ==\n");

    // ---- selection, under the contract -------------------------------------------------------
    // The exact regression. Before the fix this returned Last and the caller threw the frame away.
    CHECK(select_present_source(0, 0, k1080p, k4k) == PresentSourceChoice::None,
          "a submit whose only RGBA8 pass is smaller than the present extent selects no source");
    CHECK(select_present_source(0, 0, k1024sq, k4k) == PresentSourceChoice::None &&
              select_present_source(0, 0, k4kTall, k4k) == PresentSourceChoice::None,
          "every wrong extent measured on Frontiers is refused, over- as well as under-sized");
    CHECK(select_present_source(k4k, 0, k1080p, k4k) == PresentSourceChoice::Front,
          "the flipped front buffer is selected when it matches the present extent");
    CHECK(select_present_source(0, k4k, k1080p, k4k) == PresentSourceChoice::Vo,
          "a registered scanout pass is selected over a wrong-extent last pass");
    // Preference, not merely rejection: a correctly sized lower-priority candidate must win.
    CHECK(select_present_source(k1080p, k4k, 0, k4k) == PresentSourceChoice::Vo,
          "a correctly sized scanout pass beats a wrong-extent front buffer");
    CHECK(select_present_source(k1080p, k1024sq, k4k, k4k) == PresentSourceChoice::Last,
          "a correctly sized last pass beats wrong-extent front and scanout passes");
    CHECK(select_present_source(k4k, k4k, k4k, k4k) == PresentSourceChoice::Front,
          "identity priority still decides among candidates that all fit");
    CHECK(select_present_source(0, 0, 0, k4k) == PresentSourceChoice::None,
          "a submit that rendered no RGBA8 pass selects no source");

    // ---- selection, with no contract (gpu_replay prefix inspection, render_submit_items) ------
    CHECK(select_present_source(0, 0, bytes_for(16, 16), 0) == PresentSourceChoice::Last,
          "with no extent contract the last pass is still returned at its own extent");
    CHECK(select_present_source(bytes_for(8, 8), bytes_for(4, 4), bytes_for(2, 2), 0) ==
              PresentSourceChoice::Front,
          "with no extent contract the historical identity priority is unchanged");
    CHECK(select_present_source(0, bytes_for(4, 4), bytes_for(2, 2), 0) == PresentSourceChoice::Vo,
          "with no extent contract a scanout pass still outranks the last pass");
    CHECK(select_present_source(0, 0, 0, 0) == PresentSourceChoice::None,
          "with no extent contract an empty submit still selects no source");

    // ---- demotion: the one new semantic path, and the only one with no other witness ---------
    // Unreachable on current master (a VO/front-buffer pass has its extent pinned to the present
    // extent before it renders), so these assert the tripwire that would fire if that pin were
    // relaxed — not live behaviour. Stated here so the next reader does not mistake them for it.
    CHECK(present_source_demoted(PresentSourceChoice::Vo, k1080p, k4k) &&
              present_source_demoted(PresentSourceChoice::Last, k1080p, 0) &&
              present_source_demoted(PresentSourceChoice::Last, 0, k1080p),
          "passing over a non-empty higher-priority candidate on extent counts as a demotion");
    CHECK(!present_source_demoted(PresentSourceChoice::Front, k4k, k4k) &&
              !present_source_demoted(PresentSourceChoice::Vo, 0, k4k) &&
              !present_source_demoted(PresentSourceChoice::Last, 0, 0) &&
              !present_source_demoted(PresentSourceChoice::None, k1080p, k1080p),
          "choosing the top available candidate, or none at all, is not a demotion");

    // ---- the retained-frame net -------------------------------------------------------------
    // The permanence bug: a non-empty wrong-extent frame must NOT be retained, and the retained good
    // frame must be served in its place.
    CHECK(retained_frame_action(k1080p, k4k, k4k) == RetainedFrameAction::ServeRetained,
          "a wrong-extent frame is never retained and the retained 4K frame is served instead");
    CHECK(retained_frame_action(k4k, k1080p, k4k) == RetainedFrameAction::StoreCurrent,
          "a correctly sized frame is published and replaces a stale retained frame");
    // Bendy's own case, which the net exists for: nothing rendered this frame, so serve the last one.
    CHECK(retained_frame_action(0, k4k, k4k) == RetainedFrameAction::ServeRetained,
          "an empty frame during scanout rotation serves the retained frame rather than black");
    CHECK(retained_frame_action(k1080p, 0, k4k) == RetainedFrameAction::NoUsableFrame,
          "a wrong-extent frame with nothing retained yields no usable frame, not a substitution");
    CHECK(retained_frame_action(k1080p, k1024sq, k4k) == RetainedFrameAction::NoUsableFrame,
          "a wrong-extent retained frame cannot rescue a wrong-extent current frame");
    CHECK(retained_frame_action(0, 0, k4k) == RetainedFrameAction::NoUsableFrame,
          "an empty frame with nothing retained yields no usable frame");
    // With no contract the net keeps its historical emptiness semantics exactly.
    CHECK(retained_frame_action(bytes_for(16, 16), bytes_for(64, 64), 0) ==
              RetainedFrameAction::StoreCurrent,
          "with no extent contract any non-empty frame is still retained");
    CHECK(retained_frame_action(0, bytes_for(64, 64), 0) == RetainedFrameAction::ServeRetained,
          "with no extent contract an empty frame still serves the retained one");
    CHECK(retained_frame_action(0, 0, 0) == RetainedFrameAction::NoUsableFrame,
          "with no extent contract and nothing retained there is no usable frame");

    // Retaining and publishing must be the SAME predicate; that identity is the fix.
    CHECK(present_frame_publishable(k4k, k4k) && !present_frame_publishable(k1080p, k4k) &&
              !present_frame_publishable(0, k4k) && present_frame_publishable(k1080p, 0) &&
              !present_frame_publishable(0, 0),
          "publishability is non-emptiness plus the extent, and non-emptiness alone without one");

    CHECK(present_source_name(PresentSourceChoice::Front) == std::string_view("px_front") &&
              present_source_name(PresentSourceChoice::Vo) == std::string_view("px_vo") &&
              present_source_name(PresentSourceChoice::Last) == std::string_view("px_last") &&
              present_source_name(PresentSourceChoice::None) == std::string_view("none"),
          "each candidate keeps the name the renderer's provenance diagnostics already used");

    std::printf(fails ? "test_present_extent: %d FAILURE(S)\n" : "test_present_extent: all ok\n",
                fails);
    return fails ? 1 : 0;
}
