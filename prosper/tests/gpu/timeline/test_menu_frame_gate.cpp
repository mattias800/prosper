#include "gpu/timeline/menu_capture_policy.hpp"

#include <array>
#include <cstdio>
#include <limits>
#include <vector>

using namespace prosper::gpu;

static int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::printf("FAIL: %s\n", message); ++failures; } \
} while (0)

int main() {
    constexpr uint32_t width = 100, height = 100;
    const MenuFrameGateSpec gate{
        .width = width, .height = height,
        .menu = {5, 60, 45, 90},
        .logo = {5, 5, 45, 35},
        .center_x = 50, .center_y = 50,
        .bright_above = 170, .center_at_most = 48,
        .menu_bright_min = 1000, .logo_bright_min = 1000,
    };
    MenuFrameGateSpec parsed;
    CHECK(parse_menu_frame_gate_spec(
              "100x100:5,60,45,90:5,5,45,35:50,50:170,48,1000,1000", parsed) &&
              parsed.valid() && parsed.menu.x0 == gate.menu.x0 &&
              parsed.logo.y1 == gate.logo.y1 &&
              parsed.menu_bright_min == gate.menu_bright_min,
          "the configured ROI and thresholds must parse exactly");
    CHECK(!parse_menu_frame_gate_spec(
              "100x100:5,60,45,90:5,5,45,35:50,50:170,48,1000,1000extra", parsed) &&
              !parsed.valid(),
          "trailing input cannot silently change a capture gate");
    CHECK(!parse_menu_frame_gate_spec(
              "100x100:5,60,45,90:5,5,101,35:50,50:170,48,1000,1000", parsed) &&
              !parsed.valid(),
          "an out-of-bounds ROI cannot arm a capture");
    CHECK(!parse_menu_frame_gate_spec(
              "4294967296x100:5,60,45,90:5,5,45,35:50,50:170,48,1000,1000", parsed) &&
              !parsed.valid(),
          "numeric overflow cannot wrap into an apparently valid ROI");
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4, 0);
    auto paint = [&](MenuFrameRoi roi, uint32_t count) {
        for (uint32_t y = roi.y0; y < roi.y1; ++y)
            for (uint32_t x = roi.x0; x < roi.x1; ++x) {
                uint8_t* pixel = pixels.data() + (static_cast<size_t>(y) * width + x) * 4;
                const uint8_t value = count ? 255 : 0;
                if (count) --count;
                pixel[0] = pixel[1] = pixel[2] = value;
                pixel[3] = 255;
            }
    };

    CHECK(classify_menu_frame(pixels, width, height, gate).verdict ==
              MenuFrameGateVerdict::NotMenu, "black frame cannot arm the menu gate");
    paint(gate.menu, 1001);
    CHECK(classify_menu_frame(pixels, width, height, gate).verdict ==
              MenuFrameGateVerdict::MenuWithoutLogo,
          "a menu-positive frame can report a missing wordmark");
    paint(gate.logo, 1001);
    auto result = classify_menu_frame(pixels, width, height, gate);
    CHECK(result.verdict == MenuFrameGateVerdict::MenuWithLogo &&
              result.menu_bright == 1001 && result.logo_bright == 1001,
          "menu and wordmark must be classified from the same pixels");

    paint(gate.menu, 1000);
    CHECK(classify_menu_frame(pixels, width, height, gate).verdict ==
              MenuFrameGateVerdict::NotMenu,
          "threshold equality cannot silently count as menu-positive");
    paint(gate.menu, 1001);
    pixels[(static_cast<size_t>(gate.center_y) * width + gate.center_x) * 4] = 255;
    CHECK(classify_menu_frame(pixels, width, height, gate).verdict ==
              MenuFrameGateVerdict::NotMenu,
          "full-white or bright-center images cannot arm a menu capture");
    pixels[(static_cast<size_t>(gate.center_y) * width + gate.center_x) * 4] = 0;
    CHECK(classify_menu_frame(pixels, width + 1, height, gate).verdict ==
              MenuFrameGateVerdict::InvalidImage,
          "a resized frame cannot inherit another extent's ROI proof");
    CHECK(classify_menu_frame(std::span<const uint8_t>(pixels).first(pixels.size() - 1),
                              width, height, gate).verdict ==
              MenuFrameGateVerdict::InvalidImage,
          "a short publication cannot be inspected as a complete image");
    auto invalid = gate;
    invalid.logo.x1 = width + 1;
    CHECK(classify_menu_frame(pixels, width, height, invalid).verdict ==
              MenuFrameGateVerdict::InvalidImage,
          "out-of-bounds ROI must refuse before reading pixels");
    CHECK(menu_capture_exact_candidate(941, 941, 941, true) &&
              !menu_capture_exact_candidate(941, 940, 941, true) &&
              !menu_capture_exact_candidate(942, 941, 941, true) &&
              !menu_capture_exact_candidate(941, 941, 940, true) &&
              !menu_capture_exact_candidate(941, 941, 941, false) &&
              !menu_capture_exact_candidate(0, 0, 0, true),
          "capsule write and policy acceptance require the same live source identity");

    MenuCapturePolicy policy({.max_candidates = 2, .max_total_candidate_bytes = 100,
                              .max_wait_ms = 1000, .max_after_menu_ms = 100}, 0);
    policy.published(934, true, MenuFrameGateVerdict::NotMenu, true, 10);
    CHECK(policy.phase() == MenuCapturePhase::WaitingForMenu &&
              policy.attempted_candidates() == 0,
          "a pre-menu PS match cannot spend the first candidate");
    policy.published(940, true, MenuFrameGateVerdict::MenuWithLogo, true, 20);
    CHECK(policy.phase() == MenuCapturePhase::Armed &&
              policy.attempted_candidates() == 0,
          "the first menu-positive image arms only a later candidate");
    CHECK(policy.begin_candidate(941, 40, 21), "the first post-menu candidate is bounded");
    policy.published(999, true, MenuFrameGateVerdict::MenuWithLogo, true, 22);
    policy.published(941, false, MenuFrameGateVerdict::MenuWithLogo, true, 23);
    CHECK(policy.phase() == MenuCapturePhase::Pending,
          "another or unknown producer cannot certify the selected candidate");
    policy.published(941, true, MenuFrameGateVerdict::NotMenu, true, 24);
    CHECK(policy.phase() == MenuCapturePhase::Armed &&
              policy.attempted_candidates() == 1,
          "a menu-negative selected publication consumes only its own attempt");
    CHECK(policy.begin_candidate(942, 50, 25), "a later exact candidate can be retried");
    policy.published(942, true, MenuFrameGateVerdict::MenuWithoutLogo, true, 26);
    CHECK(policy.phase() == MenuCapturePhase::Accepted &&
              policy.attempted_candidates() == 2 && policy.copied_candidate_bytes() == 90,
          "accept only the menu-positive image of the exact selected submit");

    MenuCapturePolicy budget({.max_candidates = 2, .max_total_candidate_bytes = 100,
                              .max_wait_ms = 1000, .max_after_menu_ms = 100}, 0);
    budget.published(1, true, MenuFrameGateVerdict::MenuWithLogo, true, 10);
    CHECK(!budget.begin_candidate(2, 101, 11) &&
              budget.refusal() == MenuCaptureRefusal::ByteBudget,
          "over-budget candidate must refuse before any copy is authorized");
    MenuCapturePolicy no_menu({.max_wait_ms = 100}, 0);
    no_menu.tick(101);
    CHECK(no_menu.refusal() == MenuCaptureRefusal::WaitExpired,
          "a permanently absent menu must terminate with an explicit reason");
    MenuCapturePolicy expired({.max_wait_ms = 1000, .max_after_menu_ms = 100}, 0);
    expired.published(1, true, MenuFrameGateVerdict::MenuWithLogo, true, 10);
    expired.tick(111);
    CHECK(expired.refusal() == MenuCaptureRefusal::CandidateExpired,
          "the post-menu candidate interval is bounded");
    MenuCapturePolicy no_present({.max_candidates = 1}, 0);
    no_present.published(1, true, MenuFrameGateVerdict::MenuWithLogo, true, 1);
    CHECK(no_present.begin_candidate(2, 8, 2), "candidate can arm before publication");
    no_present.missing_presentation(2, 3);
    CHECK(no_present.refusal() == MenuCaptureRefusal::MissingPresentation,
          "a skipped selected publication cannot be replaced by a retained image");
    MenuCapturePolicy submit_zero({}, 0);
    submit_zero.published(0, false, MenuFrameGateVerdict::MenuWithLogo, true, 1);
    CHECK(submit_zero.phase() == MenuCapturePhase::WaitingForMenu,
          "live source submit zero is reserved for unknown and cannot arm a capture");

    MenuSourceCensusPolicy source_retry({.max_distinct_sources = 3, .max_wait_ms = 100});
    CHECK(!source_retry.observe(11, 10, false, 1) &&
              source_retry.distinct_retained_sources() == 0,
          "a non-menu image cannot spend the source-census budget");
    CHECK(!source_retry.observe(11, 10, true, 2) &&
              source_retry.phase() == MenuSourceCensusPhase::WaitingForNewSource &&
              source_retry.distinct_retained_sources() == 1,
          "a retained first menu image must not certify current draw state");
    for (uint64_t callback = 13; callback < 30; ++callback)
        CHECK(!source_retry.observe(callback, 0, true, callback) &&
                  !source_retry.observe(callback, 10, true, callback) &&
                  source_retry.distinct_retained_sources() == 1 &&
                  source_retry.phase() == MenuSourceCensusPhase::WaitingForNewSource,
              "unknown and repeated retained publications cannot exhaust source opportunities");
    CHECK(source_retry.observe(30, 30, true, 30) &&
              source_retry.phase() == MenuSourceCensusPhase::Exact &&
              source_retry.distinct_retained_sources() == 1,
          "a later exact menu publication provides the source-draw census");
    CHECK(!source_retry.observe(31, 31, true, 31) &&
              source_retry.distinct_retained_sources() == 1,
          "the exact census runs only once");
    MenuSourceCensusPolicy source_limit({.max_distinct_sources = 2, .max_wait_ms = 100});
    CHECK(!source_limit.observe(10, 0, true, 0) &&
              !source_limit.observe(11, 10, true, 1) &&
              !source_limit.observe(12, 9, true, 2) &&
              source_limit.phase() == MenuSourceCensusPhase::AttemptLimit,
          "only distinct known retained sources exhaust the bounded opportunity budget");
    MenuSourceCensusPolicy source_expiry({.max_distinct_sources = 3, .max_wait_ms = 100});
    CHECK(!source_expiry.observe(10, 9, true, 0), "retained image starts the census clock");
    source_expiry.tick(101);
    CHECK(source_expiry.phase() == MenuSourceCensusPhase::WaitExpired &&
              !source_expiry.observe(11, 11, true, 102),
          "a late exact image cannot revive an expired census");

    const MenuWriterEvidence scene{.draw_index = 10, .ps_addr = 0xabc,
        .target_addr = 0x123, .width = 3200, .height = 1800, .format = 1,
        .raw_write_mask = 7, .realized_exact = true, .realized_write_mask = 7};
    const MenuWriterEvidence screen{.draw_index = 11, .ps_addr = 0xdef,
        .target_addr = 0x456, .width = 3840, .height = 2160, .format = 1,
        .raw_write_mask = 15, .realized_exact = true, .realized_write_mask = 15};
    const auto choose = [](std::span<const MenuWriterEvidence> draws) {
        return menu_capture_select_target(draws, 0xabc, 3200, 1800, 3840, 2160);
    };
    const std::array chain{scene, screen};
    CHECK(choose(chain).rejection == MenuTargetReject::None &&
              choose(chain).target_addr == scene.target_addr &&
              choose(chain).draw_index == scene.draw_index,
          "same-submit admitted scene followed by screen draws yields a run-local candidate");
    auto last_screen = screen;
    last_screen.draw_index = 12;
    const std::array stale_program{scene, MenuWriterEvidence{.draw_index = 11,
        .ps_addr = 0xfff, .target_addr = 0x987, .width = 3200, .height = 1800,
        .format = 1, .raw_write_mask = 7, .realized_exact = true,
        .realized_write_mask = 7}, last_screen};
    CHECK(choose(stale_program).rejection == MenuTargetReject::WrongSceneProgram,
          "an old PS address cannot select an earlier scene when the final scene differs");
    const std::array no_screen{scene};
    CHECK(choose(no_screen).rejection == MenuTargetReject::MissingScreen,
          "a scene draw without a later same-submit screen writer is not a menu chain");
    const std::array interrupted{scene, screen, MenuWriterEvidence{.draw_index = 12,
        .ps_addr = 0xeee, .target_addr = 0x777, .width = 100, .height = 100,
        .format = 1, .raw_write_mask = 15, .realized_exact = true,
        .realized_write_mask = 15}};
    CHECK(choose(interrupted).rejection == MenuTargetReject::OtherWriterAfterScene,
          "another admitted target size after the scene breaks the simple chain");
    auto disabled = scene;
    disabled.realized_write_mask = 0;
    const std::array disabled_chain{disabled, screen};
    CHECK(choose(disabled_chain).rejection == MenuTargetReject::UnrealizedFinalScene,
          "a semantic final scene writer missing admission cannot select an earlier target");
    disabled.realized_write_mask = 8;
    const std::array disjoint_mask_chain{disabled, screen};
    CHECK(choose(disjoint_mask_chain).rejection == MenuTargetReject::UnrealizedFinalScene,
          "semantic and realized masks must overlap for an admitted color-write candidate");
    disabled.draw_index = 11;
    const std::array missing_final_scene{scene, disabled, last_screen};
    CHECK(choose(missing_final_scene).rejection == MenuTargetReject::UnrealizedFinalScene &&
              choose(missing_final_scene).draw_index == disabled.draw_index,
          "a later failed semantic scene writer invalidates an earlier admitted scene");
    auto missing_screen = screen;
    missing_screen.realized_exact = false;
    const std::array missing_screen_writer{scene, missing_screen};
    CHECK(choose(missing_screen_writer).rejection == MenuTargetReject::UnrealizedLaterWriter,
          "a later semantic screen writer without matching realization is ambiguous");
    CHECK(menu_capture_target_matches_request(choose(chain), 0) &&
              menu_capture_target_matches_request(choose(chain), scene.target_addr) &&
              !menu_capture_target_matches_request(choose(chain), 0x999) &&
              !menu_capture_target_matches_request(choose(disabled_chain), 0),
          "an explicit address is an assertion on the run-local source, never a substitute");
    CHECK(menu_declared_range_relation(100, 20, 110, 20) ==
              MenuDeclaredRangeRelation::Overlap &&
              menu_declared_range_relation(100, 20, 120, 20) ==
              MenuDeclaredRangeRelation::Disjoint &&
              menu_declared_range_relation(100, 20, 0, 20) ==
              MenuDeclaredRangeRelation::Unknown &&
              menu_declared_range_relation(UINT64_MAX - 4, 8, 100, 20) ==
              MenuDeclaredRangeRelation::Unknown,
          "dependency metadata cannot label missing or overflowing declared ranges disjoint");
    enum class StubOperationKind { Draw, Dispatch };
    struct StubOperation {
        StubOperationKind kind;
        size_t index;
        uint64_t command_order;
    };
    const std::array ordered{StubOperation{StubOperationKind::Dispatch, 0, 8},
                             StubOperation{StubOperationKind::Draw, 360, 9}};
    const std::span<const StubOperation> ordered_span(ordered);
    CHECK(menu_capture_selected_draw_position(ordered_span, StubOperationKind::Draw, 360, 9) == 1 &&
              menu_capture_selected_draw_position(ordered_span, StubOperationKind::Draw, 360, 10) ==
                  std::numeric_limits<size_t>::max() &&
              menu_capture_selected_draw_position(ordered_span, StubOperationKind::Draw, 0, 8) ==
                  std::numeric_limits<size_t>::max(),
          "a prior effect is temporal evidence only when the exact selected draw exists later");

    std::printf("menu_frame_gate: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
