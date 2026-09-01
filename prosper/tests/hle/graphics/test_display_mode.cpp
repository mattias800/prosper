// test_display_mode -- #3017: prosper must DERIVE the display it advertises to the guest from the
// host's real mode, not hardcode 1920x1080 @ 59.94 Hz.
//
// Two halves, and the second is the one that matters:
//
//   Part A is the pure derivation (display_mode.hpp) against SYNTHESIZED host capabilities. The
//   load-bearing arms are the ones where two DIFFERENT hosts must produce two DIFFERENT advertised
//   modes. A test that only checked "1080p host -> 1080p59.94" would pass just as happily against
//   the hardcoded constants this replaces, so it would prove nothing about deriving.
//
//   Part B drives the real HLE surface and asserts the two places the issue names -- what the title
//   READS (sceVideoOutGetResolutionStatus) and what actually PACES it (the vblank period the
//   kevent pump schedules on) -- BOTH follow the one synthesized host. Before #3017 those were
//   independent constants that agreed only by hand, so this is the arm that pins them together.
//
// Part B must run in its own process: the advertised mode resolves ONCE on first use (deliberately
// -- a period that moved mid-run would step the guest's vblank count sideways), so the environment
// has to be set before any VideoOut call. test_videoout.cpp covers the complementary case, the
// untouched `legacy` default, in a process where these variables are unset.

#include "hle/graphics/display_mode.hpp"
#include "hle/dispatch/dispatch.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

using namespace prosper;
using namespace prosper::hle::graphics;

// The accessor hle_kernel_time.cpp's vblank kevent pump schedules on (#3024). Reading it here is
// what proves the PACING half moved, not merely the status struct the title reads.
extern "C" uint64_t prosper_vo_vblank_period_ns();

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int main() {
    printf("== test_display_mode ==\n");

    // ---- Part A: the pure derivation -------------------------------------------------------
    printf("-- parsing --\n");
    {
        HostDisplayMode host;
        CHECK(parse_host_display_mode("1920x1080@59.94", host) &&
              host.width == 1920 && host.height == 1080 &&
              host.refresh_hz > 59.93 && host.refresh_hz < 59.95,
              "a well-formed host mode parses to its three fields");

        HostDisplayMode integral;
        CHECK(parse_host_display_mode("3840x2160@120", integral) &&
              integral.width == 3840 && integral.height == 2160 &&
              integral.refresh_hz > 119.99 && integral.refresh_hz < 120.01,
              "an integral refresh rate parses without a fractional part");

        // A malformed value must leave the caller on the documented fallback rather than half-
        // applying: the same rule the capture triggers follow -- a typo costs the feature, never a
        // wrong answer.
        HostDisplayMode untouched;
        const char* malformed[] = { "", "1920x1080", "1920@59.94", "1920x1080@", "x1080@59.94",
                                    "1920x1080@59.94junk", "1920x1080@60.", "0x1080@60",
                                    "1920x0@60", "1920x1080@0" };
        bool all_rejected = true;
        for (const char* text : malformed)
            all_rejected &= !parse_host_display_mode(text, untouched);
        CHECK(all_rejected, "every malformed host mode is rejected");
        CHECK(untouched.width == 0 && untouched.height == 0 && untouched.refresh_hz == 0.0,
              "a rejected parse does not partially overwrite the caller's mode");

        DisplayModePolicy policy = DisplayModePolicy::host;
        CHECK(!parse_display_mode_policy("hosty", policy) && policy == DisplayModePolicy::host,
              "an unknown policy name is rejected and leaves the caller's choice alone");
        CHECK(parse_display_mode_policy("legacy", policy) && policy == DisplayModePolicy::legacy,
              "\"legacy\" parses");
        CHECK(parse_display_mode_policy("host", policy) && policy == DisplayModePolicy::host,
              "\"host\" parses");
        CHECK(parse_display_mode_policy("host-high-refresh", policy) &&
              policy == DisplayModePolicy::host_high_refresh,
              "\"host-high-refresh\" parses");
    }

    printf("-- the default is exactly what prosper advertised before #3017 --\n");
    {
        // The conservative default, and the whole reason deriving is opt-in: a title that works
        // today cannot be changed by this code existing.
        const HostDisplayMode big{ 3840, 2160, 119.88 };
        const AdvertisedDisplayMode legacy = select_display_mode(DisplayModePolicy::legacy, big);
        CHECK(legacy.width == 1920 && legacy.height == 1080,
              "legacy advertises 1920x1080 even on a 4K host");
        CHECK(legacy.refresh_enum == 3 && legacy.vblank_period_ns == 16683350,
              "legacy advertises refresh enum 3 with the 59.94 Hz period");
        CHECK(!legacy.derived, "legacy reports itself as not derived");

        // An unknown host is the headless / boot_trace / unit-test case and MUST land on the same
        // fallback, not on a zeroed or partially-filled mode.
        const AdvertisedDisplayMode unknown = select_display_mode(DisplayModePolicy::host, {});
        CHECK(unknown.width == 1920 && unknown.height == 1080 && unknown.refresh_enum == 3 &&
              unknown.vblank_period_ns == 16683350 && !unknown.derived,
              "an unknown host falls back to 1920x1080 @ 59.94 and says it did not derive");
    }

    printf("-- DERIVATION: a different host must produce a different advertised mode --\n");
    {
        // This block is the point of the whole test. Each arm pairs a synthesized host with an
        // answer the pre-#3017 constants could not have produced.
        const AdvertisedDisplayMode uhd = select_display_mode(
            DisplayModePolicy::host, { 3840, 2160, 60.0 });
        CHECK(uhd.width == 3840 && uhd.height == 2160,
              "a 4K host is advertised 3840x2160, not the hardcoded 1080p");
        CHECK(uhd.derived, "a 4K host reports a derived mode");

        const AdvertisedDisplayMode hd = select_display_mode(
            DisplayModePolicy::host, { 1920, 1080, 60.0 });
        CHECK(hd.width == 1920 && hd.height == 1080,
              "a 1080p host is advertised 1920x1080");
        CHECK(uhd.width != hd.width && uhd.height != hd.height,
              "two hosts of different SIZE yield two different advertised resolutions");

        const AdvertisedDisplayMode small = select_display_mode(
            DisplayModePolicy::host, { 1366, 768, 60.0 });
        CHECK(small.width == 1280 && small.height == 720,
              "a host too small for 1080p drops to the 720p output mode, not to the host's own "
              "non-PS5 resolution");

        // A nominal "60.000" host maps onto the 59.94 enumerant -- the rate a PS5 calls 60 -- and
        // must NOT reach 119.88 even though the tolerance exists.
        CHECK(hd.refresh_enum == 3 && hd.vblank_period_ns == 16683350,
              "a 60.000 Hz host maps to the 59.94 enumerant");

        // A 50 Hz host is a real configuration, and telling it 59.94 is precisely the lie #3017 is
        // about. This arm cannot pass against a hardcoded refresh constant.
        const AdvertisedDisplayMode pal = select_display_mode(
            DisplayModePolicy::host, { 1920, 1080, 50.0 });
        CHECK(pal.refresh_enum == 2 && pal.vblank_period_ns == 20000000,
              "a 50 Hz host is advertised the 50 Hz enumerant and a 20 ms period");
        CHECK(pal.refresh_enum != hd.refresh_enum &&
              pal.vblank_period_ns != hd.vblank_period_ns,
              "two hosts of different REFRESH yield two different enumerants AND two different "
              "vblank periods");
    }

    printf("-- the evidence floor gates the enumerant prosper cannot yet defend --\n");
    {
        const HostDisplayMode fast{ 3840, 2160, 120.0 };
        // `host` must not reach 119.88: its enumerant value rests on no primary evidence, and no
        // title here has been checked for flip-rate-dependent logic (the charter records a title
        // whose in-game time advances PER FLIP, which at double the rate runs at double speed).
        const AdvertisedDisplayMode conservative = select_display_mode(DisplayModePolicy::host, fast);
        CHECK(conservative.refresh_enum == 3 && conservative.vblank_period_ns == 16683350,
              "`host` on a 120 Hz display still advertises 59.94 -- the LOW-evidence enumerant is "
              "not reachable by deriving alone");
        CHECK(conservative.width == 3840 && conservative.height == 2160,
              "...while the resolution half, which needs no enumerant, still derives");

        const AdvertisedDisplayMode opted_in =
            select_display_mode(DisplayModePolicy::host_high_refresh, fast);
        CHECK(opted_in.refresh_enum == 6 && opted_in.vblank_period_ns == 8341675,
              "`host-high-refresh` reaches the 119.88 enumerant and its period");
        CHECK(opted_in.vblank_period_ns * 2 == 16683350,
              "the 119.88 period is exactly half the 59.94 period (the two enumerants are "
              "consistent with each other)");

        // Never advertise a rate the host cannot present: the pacing grid would then honour it and
        // wake the guest faster than the panel can show.
        const AdvertisedDisplayMode slow_panel =
            select_display_mode(DisplayModePolicy::host_high_refresh, { 1920, 1080, 60.0 });
        CHECK(slow_panel.refresh_enum == 3,
              "even under host-high-refresh, a 60 Hz host is never told 119.88");

        // A host slower than every enumerant we will advertise keeps the fallback refresh, but its
        // resolution still derives: the two halves fail independently.
        const AdvertisedDisplayMode very_slow = select_display_mode(
            DisplayModePolicy::host, { 3840, 2160, 30.0 });
        CHECK(very_slow.refresh_enum == 3 && very_slow.vblank_period_ns == 16683350,
              "a 30 Hz host keeps the fallback refresh (no enumerant is at or below it)");
        CHECK(very_slow.width == 3840 && very_slow.height == 2160,
              "...and still gets its derived resolution");
    }

    printf("-- every advertised enumerant carries a self-consistent period --\n");
    {
        bool consistent = true;
        for (const RefreshMode& mode : kRefreshModes) {
            const double implied_hz = 1e9 / (double)mode.period_ns;
            consistent &= implied_hz > mode.hz * 0.999 && implied_hz < mode.hz * 1.001;
        }
        CHECK(consistent,
              "each refresh row's period_ns is the reciprocal of its own hz -- so the status "
              "struct and the pacing grid cannot describe different rates");
    }

    // ---- Part B: the live HLE surface, end to end ------------------------------------------
    printf("-- INTEGRATION: both enforcement points follow one synthesized host --\n");
    {
        // Set BEFORE the first VideoOut call: the mode resolves once, on first use.
        set_env("PROSPER_DISPLAY_MODE", "host-high-refresh");
        set_env("PROSPER_HOST_DISPLAY_MODE", "3840x2160@119.88");

        register_builtin_hle();
        auto open = Hle::lookup(nid_hash("sceVideoOutOpen"));
        auto res  = Hle::lookup(nid_hash("sceVideoOutGetResolutionStatus"));
        CHECK(open && res, "the VideoOut entry points are registered");
        if (open && res) {
            const uint64_t handle = open(0, 0, 0, 0, 0, 0);
            uint8_t rs[0x30];
            memset(rs, 0xEE, sizeof rs);
            CHECK(res(handle, (uint64_t)(uintptr_t)rs, 0, 0, 0, 0) == 0,
                  "GetResolutionStatus succeeds on a live handle");

            // (1) what the title READS.
            CHECK(*(uint32_t*)(rs + 0x00) == 3840 && *(uint32_t*)(rs + 0x04) == 2160,
                  "the advertised resolution is the host's 3840x2160, NOT the hardcoded 1920x1080");
            CHECK(*(uint32_t*)(rs + 0x08) == 3840 && *(uint32_t*)(rs + 0x0c) == 2160,
                  "the pane resolution follows the same derived answer");
            CHECK(*(uint64_t*)(rs + 0x10) == 6,
                  "the advertised refresh enumerant is the host's 119.88, NOT the hardcoded 3");

            // (2) what actually PACES it -- the accessor hle_kernel_time.cpp's vblank pump reads.
            CHECK(prosper_vo_vblank_period_ns() == 8341675,
                  "the vblank grid's period is the 119.88 Hz period, NOT the hardcoded 16683350");

            // The invariant the issue asks for in so many words: changing only one of the two would
            // give the title a number it cannot act on, or pace it in a mode it does not believe it
            // is in. Assert they came from the same row rather than merely both being right.
            const uint64_t advertised_enum = *(uint64_t*)(rs + 0x10);
            const RefreshMode* row = nullptr;
            for (const RefreshMode& candidate : kRefreshModes)
                if (candidate.enumerant == advertised_enum) row = &candidate;
            CHECK(row && row->period_ns == prosper_vo_vblank_period_ns(),
                  "the reported enumerant and the pacing period are the SAME table row");
        }
    }

    printf("%s: %d failure(s)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
