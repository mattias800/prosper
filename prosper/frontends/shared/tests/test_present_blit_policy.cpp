#include "shared/present/present_blit_policy.hpp"
#include <gtest/gtest.h>
#include "shared/present/compute_scanout.hpp"
#include "diagnostics/perf/perf_alarm_rules.hpp"   // kPresentSlotTroubleReasons (header-only use)

#include <cstdio>
#include <cstring>

using prosper::frontend::present_blit_wait_completed;
using prosper::frontend::present_blit_has_new_flip;
using prosper::frontend::present_source_is_newer;

#define CHECK(cond) EXPECT_TRUE(cond)

TEST(PresentBlitPolicy, Contract) {
    CHECK(present_blit_wait_completed(VK_SUCCESS));

    // #1303: a timeout or device error must not publish a slot whose blit may still be in flight.
    CHECK(!present_blit_wait_completed(VK_TIMEOUT));
    CHECK(!present_blit_wait_completed(VK_ERROR_DEVICE_LOST));

    // Several graphics submits may build one guest frame. Only the first pre-flip image and a
    // newly flipped image are publishable; repeated submits at the same flip stay GPU-local.
    CHECK(present_blit_has_new_flip(UINT64_MAX, 0));
    CHECK(!present_blit_has_new_flip(0, 0));
    CHECK(present_blit_has_new_flip(0, 1));
    CHECK(!present_blit_has_new_flip(42, 42));
    CHECK(present_blit_has_new_flip(UINT64_MAX, UINT64_MAX));

    // The first source is always displayable, including the pre-flip identity zero. Thereafter,
    // duplicate or stale CPU/GPU representations must not overwrite the newest frame.
    CHECK(present_source_is_newer(false, 0, 0));
    CHECK(present_source_is_newer(false, 99, 3));
    CHECK(!present_source_is_newer(true, 7, 6));
    CHECK(!present_source_is_newer(true, 7, 7));
    CHECK(present_source_is_newer(true, 7, 8));

    // #3915: every renderer outcome that sends a flip to the CPU fallback is a named decline, and
    // the three that do not (published, already published, no consumer) are not.
    using prosper::frontend::GpuPresentOutcome;
    using prosper::frontend::gpu_present_outcome_is_decline;
    CHECK(!gpu_present_outcome_is_decline(GpuPresentOutcome::Published));
    CHECK(!gpu_present_outcome_is_decline(GpuPresentOutcome::SameFlip));
    CHECK(!gpu_present_outcome_is_decline(GpuPresentOutcome::Inactive));
    for (int o = static_cast<int>(GpuPresentOutcome::CaptureNeedsCpu);
         o < static_cast<int>(GpuPresentOutcome::Count); ++o) {
        CHECK(gpu_present_outcome_is_decline(static_cast<GpuPresentOutcome>(o)));
        CHECK(prosper::frontend::gpu_present_outcome_name(static_cast<GpuPresentOutcome>(o))[0] != '?');
    }

    // #3891: present-slot-trouble matches decline slots by NAME (the diagnostics layer cannot
    // include this header). A renamed outcome would silence that alarm, so pin each spelling here.
    for (const char* reason : prosper::diagnostics::perf::kPresentSlotTroubleReasons) {
        bool named = false;
        for (int o = 0; o < static_cast<int>(GpuPresentOutcome::Count); ++o)
            named |= std::strcmp(reason, prosper::frontend::gpu_present_outcome_name(
                                             static_cast<GpuPresentOutcome>(o))) == 0 &&
                     gpu_present_outcome_is_decline(static_cast<GpuPresentOutcome>(o));
        CHECK(named);
    }

    // #3915: which compute results may carry a GPU-present mirror.
    using namespace prosper::frontend;
    ComputeScanoutCandidate c;
    c.gpu_present_active = true; c.registered_scanout = true; c.exact_full_result = true;
    c.width = 3840; c.height = 2160; c.linear_bytes = 3840ull * 2160 * 4; c.shared_device = true;
    CHECK(compute_scanout_eligible(c) == ComputeScanoutEligibility::Eligible);
    { auto d = c; d.gpu_present_active = false;
      CHECK(compute_scanout_eligible(d) == ComputeScanoutEligibility::Inactive); }
    { auto d = c; d.registered_scanout = false;
      CHECK(compute_scanout_eligible(d) == ComputeScanoutEligibility::NotScanout); }
    { auto d = c; d.exact_full_result = false;
      CHECK(compute_scanout_eligible(d) == ComputeScanoutEligibility::NotExactResult); }
    { auto d = c; d.linear_bytes = 3840ull * 2160 * 8;   // an 8-byte texel format
      CHECK(compute_scanout_eligible(d) == ComputeScanoutEligibility::WrongTexelSize); }
    { auto d = c; d.shared_device = false;
      CHECK(compute_scanout_eligible(d) == ComputeScanoutEligibility::DeviceMismatch); }

    // #3915: when a committed mirror may stand in for the flipped buffer.
    ComputeScanoutPresentInputs in;
    in.present_extent_bytes = in.display_bytes = 3840ull * 2160 * 4;
    in.committed = true; in.watch_state = 0;
    in.mirror_width = in.front_width = 3840; in.mirror_height = in.front_height = 2160;
    in.mirror_tile_mode = in.front_scanout_tile_mode = 27;
    CHECK(compute_scanout_present_decision(in) == ComputeScanoutPresent::Publish);
    { auto d = in; d.have_selected_pixels = true;
      CHECK(compute_scanout_present_decision(d) == ComputeScanoutPresent::RendererSource); }
    { auto d = in; d.renderer_scanout = true;
      CHECK(compute_scanout_present_decision(d) == ComputeScanoutPresent::RendererSource); }
    // #3924 review: ANY renderer entry at the front address, a pixel-less tombstone included, makes
    // the CPU path keep the previous frame, so the mirror must not stand in -- even when everything
    // else (a fresh, watched, matching mirror) would publish.
    { auto d = in; d.renderer_owns_front = true;
      CHECK(compute_scanout_present_decision(d) == ComputeScanoutPresent::RendererOwnsTarget); }
    { auto d = in; d.present_extent_bytes = 1920ull * 1080 * 4;   // PROSPER_RENDER_SCALE=2
      CHECK(compute_scanout_present_decision(d) == ComputeScanoutPresent::ScaledPresent); }
    { auto d = in; d.committed = false;
      CHECK(compute_scanout_present_decision(d) == ComputeScanoutPresent::Absent); }
    { auto d = in; d.front_width = 1920;
      CHECK(compute_scanout_present_decision(d) == ComputeScanoutPresent::ExtentMismatch); }
    { auto d = in; d.front_scanout_tile_mode = 0;
      CHECK(compute_scanout_present_decision(d) == ComputeScanoutPresent::TileMismatch); }
    { auto d = in; d.watch_state = 1;
      CHECK(compute_scanout_present_decision(d) == ComputeScanoutPresent::Stale); }
    { auto d = in; d.watch_state = 2;
      CHECK(compute_scanout_present_decision(d) == ComputeScanoutPresent::Unwatched); }

}
