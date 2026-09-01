#pragma once
// The one place prosper decides WHICH DISPLAY it tells the guest it is attached to.
//
// Before #3017 this decision was two unrelated hardcoded constants in hle_graphics.cpp: a
// 1920x1080 / refresh-enumerant-3 pair written into SceVideoOutResolutionStatus, and a separate
// 16683350 ns vblank period driving both sceVideoOutGetVblankStatus's count and
// sceVideoOutWaitVblank's wake boundary (and, since #3024, the kernel's vblank kevent pump). They
// agreed only because somebody kept two literals in step by hand. A title that reads the advertised
// mode and paces to it therefore sat at 59.94 Hz on any host, and a 4K host was told it was 1080p.
//
// The rule this header exists to enforce is the charter's: DERIVE the answer, never hardcode it.
// So one `select_display_mode()` resolves ONE AdvertisedDisplayMode, and both the status struct and
// the pacing grid read their fields off that single object. They can no longer disagree, because
// there is no longer a second place for them to disagree in.
//
// Pure and dependency-free on purpose: no SDL, no Vulkan, no guest memory. The host's real mode
// arrives as plain data (the frontend queries SDL and publishes it), which is what lets the whole
// derivation be unit-tested against SYNTHESIZED host capabilities -- including the arm that proves
// a different host produces a different advertised mode, rather than one constant swapped for
// another.

#include <cstdint>
#include <string_view>

namespace prosper::hle::graphics {

// How much evidence stands behind a refresh enumerant's VALUE -- not behind the rate existing.
// SceVideoOutRefreshRate is an ENUM, so advertising a rate means asserting the integer Sony
// assigned it. Getting that integer wrong is worse than not offering the rate at all: the title
// reads a number it may act on as a different rate entirely. Per the charter's evidence hierarchy,
// each row records what it rests on and the selection policy carries a floor, so an unevidenced
// enumerant can never be reached by accident.
enum class ModeEvidence {
    low,      // a hypothesis; reachable only when explicitly asked for
    medium,   // agreement among independently written secondary implementations
    high,     // prosper's own live-title evidence
};

struct RefreshMode {
    uint64_t enumerant;    // SceVideoOutRefreshRate value written at ResolutionStatus + 0x10
    uint64_t period_ns;    // the vblank period THIS rate implies -- the pacing grid's period
    double   hz;
    ModeEvidence evidence;
};

// The refresh rates prosper is willing to advertise, ascending by rate.
//
// Deliberately minimal: one row per rate a real host display can actually be. Every row is a
// liability (a wrong enumerant is a wrong answer the guest acts on), so rates no desktop monitor
// runs at are omitted even where the enumerant is guessable.
//
//   59.94  CONFIDENCE: HIGH   -- enumerant 3 is what prosper has advertised since the VideoOut
//                                bring-up and what every booting title has read; its period is the
//                                16683350 ns grid hle_graphics.cpp and the #3024 kevent pump share.
//   50.00  CONFIDENCE: MED    -- enumerant 2. Secondary-implementation agreement only; no title
//                                observed reading it here. Reachable under `host` because a 50 Hz
//                                host is a real configuration and telling it 59.94 is the lie this
//                                issue is about.
//  119.88  CONFIDENCE: LOW    -- enumerant 6. This is the rate #3017 actually wants and the one
//                                prosper can least defend: no primary evidence pins the integer,
//                                and no 120 Hz-capable title has been checked here for
//                                flip-rate-dependent logic (the charter records The Plucky Squire
//                                advancing in-game time PER FLIP, which at double the rate runs at
//                                double speed). Reachable ONLY via the explicit
//                                host-high-refresh policy, never by deriving from the host alone.
inline constexpr RefreshMode kRefreshModes[] = {
    {  2, 20000000,  50.00, ModeEvidence::medium },
    {  3, 16683350,  59.94, ModeEvidence::high   },
    {  6,  8341675, 119.88, ModeEvidence::low    },
};

// The documented fallback, and today's behaviour exactly: 1080p at 59.94 Hz. Used whenever the
// host mode is unknown, unparseable, or matches no enumerant we are willing to advertise.
inline constexpr uint32_t kFallbackWidth  = 1920;
inline constexpr uint32_t kFallbackHeight = 1080;
inline constexpr uint64_t kFallbackRefreshEnum = 3;
inline constexpr uint64_t kFallbackPeriodNs    = 16683350;

// The output resolutions a PS5 actually scans out. The host's desktop resolution is NOT passed
// through: a title told it is attached to a 1366x768 panel is being told something no PS5 ever
// reports. We pick the largest REAL PS5 output mode the host can display instead, which keeps the
// answer derived from the host while staying inside the set of modes titles are built for.
struct Resolution { uint32_t width, height; };
inline constexpr Resolution kOutputResolutions[] = {
    { 1280,  720 },
    { 1920, 1080 },
    { 3840, 2160 },
};

// What the host reports about its display. All-zero / non-positive means "unknown", which is the
// normal case for headless runs, boot_trace, and the unit tests -- and must land on the fallback.
struct HostDisplayMode {
    uint32_t width = 0;
    uint32_t height = 0;
    double   refresh_hz = 0.0;

    constexpr bool known() const {
        return width > 0 && height > 0 && refresh_hz > 0.0;
    }
};

// The resolved answer. Every VideoOut surface that describes the display reads its fields off one
// of these, so the status struct and the vblank grid share a single origin by construction.
struct AdvertisedDisplayMode {
    uint32_t width = kFallbackWidth;
    uint32_t height = kFallbackHeight;
    uint64_t refresh_enum = kFallbackRefreshEnum;
    uint64_t vblank_period_ns = kFallbackPeriodNs;
    // True only when the host's real mode was consulted AND used. False means this is the
    // documented 59.94/1080p fallback, whatever the reason -- which is what the boot log prints.
    bool derived = false;
};

// PROSPER_DISPLAY_MODE. Default is `legacy`: byte-identical to prosper's pre-#3017 behaviour, so a
// title that works today cannot be changed by this code existing. Deriving is opt-in because the
// advertised mode is something titles ACT on -- resolution selection and frame pacing both hang
// off it -- so it needs per-title verification rather than a global flip.
enum class DisplayModePolicy {
    legacy,             // always 1920x1080 @ 59.94 (enum 3). The default.
    host,               // derive from the host, among enumerants of MEDIUM evidence or better.
    host_high_refresh,  // additionally allow LOW-evidence high-rate enumerants (119.88).
};

constexpr const char* display_mode_policy_name(DisplayModePolicy policy) {
    switch (policy) {
    case DisplayModePolicy::legacy: return "legacy";
    case DisplayModePolicy::host: return "host";
    case DisplayModePolicy::host_high_refresh: return "host-high-refresh";
    }
    return "legacy";
}

// A malformed value selects nothing and returns false, so the caller keeps the conservative
// default rather than silently picking a policy nobody asked for (the same rule the capture
// triggers follow: a typo costs you the feature, never a wrong measurement).
constexpr bool parse_display_mode_policy(std::string_view text, DisplayModePolicy& policy) {
    if (text == "legacy") policy = DisplayModePolicy::legacy;
    else if (text == "host") policy = DisplayModePolicy::host;
    else if (text == "host-high-refresh") policy = DisplayModePolicy::host_high_refresh;
    else return false;
    return true;
}

namespace detail {

// Minimal, locale-independent "<uint>" and "<double>" scanners. std::stoul/strtod would pull in
// locale and errno behaviour this header has no reason to inherit, and constexpr parsing keeps the
// whole seam usable in static_asserts.
constexpr bool scan_uint(std::string_view& text, uint32_t& out) {
    if (text.empty()) return false;
    uint64_t value = 0;
    size_t digits = 0;
    while (digits < text.size() && text[digits] >= '0' && text[digits] <= '9') {
        value = value * 10 + (uint64_t)(text[digits] - '0');
        if (value > 0xFFFFFFFFull) return false;
        ++digits;
    }
    if (digits == 0) return false;
    text.remove_prefix(digits);
    out = (uint32_t)value;
    return true;
}

constexpr bool scan_double(std::string_view& text, double& out) {
    uint32_t whole = 0;
    if (!scan_uint(text, whole)) return false;
    double value = (double)whole;
    if (!text.empty() && text.front() == '.') {
        text.remove_prefix(1);
        double scale = 0.1;
        size_t digits = 0;
        while (digits < text.size() && text[digits] >= '0' && text[digits] <= '9') {
            value += (double)(text[digits] - '0') * scale;
            scale *= 0.1;
            ++digits;
        }
        if (digits == 0) return false;   // "60." is malformed, not 60
        text.remove_prefix(digits);
    }
    out = value;
    return true;
}

}  // namespace detail

// Parse the frontend's published host mode: "<width>x<height>@<hz>", e.g. "3840x2160@119.88".
// Anything else fails and leaves `mode` untouched, which lands the caller on the fallback.
constexpr bool parse_host_display_mode(std::string_view text, HostDisplayMode& mode) {
    HostDisplayMode parsed;
    if (!detail::scan_uint(text, parsed.width)) return false;
    if (text.empty() || text.front() != 'x') return false;
    text.remove_prefix(1);
    if (!detail::scan_uint(text, parsed.height)) return false;
    if (text.empty() || text.front() != '@') return false;
    text.remove_prefix(1);
    if (!detail::scan_double(text, parsed.refresh_hz)) return false;
    if (!text.empty()) return false;              // trailing junk is malformed, not ignorable
    if (!parsed.known()) return false;
    mode = parsed;
    return true;
}

// The largest PS5 output resolution the host can actually display. A host smaller than 720p still
// gets 720p: that is the floor a PS5 scans out, and shrinking below it would advertise a mode no
// title expects.
constexpr Resolution select_output_resolution(const HostDisplayMode& host) {
    Resolution best = kOutputResolutions[0];
    for (const Resolution& candidate : kOutputResolutions)
        if (candidate.width <= host.width && candidate.height <= host.height)
            best = candidate;
    return best;
}

// The fastest enumerant the host can actually present, at or below its reported rate, subject to
// the policy's evidence floor.
//
// The 0.5% tolerance is what maps a host reporting a nominal "60.000" onto the 59.94 enumerant --
// the rate a PS5 calls 60. It is deliberately far too tight to let 119.88 through on a 60 Hz host
// (119.88 > 60.3), because advertising a rate the host cannot present would be a lie the pacing
// grid then has to honour: the guest would be woken at 119.88 Hz on a panel that cannot show it.
constexpr const RefreshMode* select_refresh_mode(const HostDisplayMode& host,
                                                 ModeEvidence min_evidence) {
    const RefreshMode* best = nullptr;
    for (const RefreshMode& candidate : kRefreshModes) {
        if (candidate.evidence < min_evidence) continue;
        if (candidate.hz > host.refresh_hz * 1.005) continue;
        if (!best || candidate.hz > best->hz) best = &candidate;
    }
    return best;
}

// Resolve the one mode prosper advertises. Both halves the issue names -- what the title READS
// (SceVideoOutResolutionStatus) and what actually PACES it (the vblank grid's period) -- come out
// of this single call, which is the structural half of the fix.
constexpr AdvertisedDisplayMode select_display_mode(DisplayModePolicy policy,
                                                    const HostDisplayMode& host) {
    AdvertisedDisplayMode mode;   // the documented 1080p @ 59.94 fallback
    if (policy == DisplayModePolicy::legacy) return mode;
    if (!host.known()) return mode;

    const ModeEvidence floor = policy == DisplayModePolicy::host_high_refresh
                                   ? ModeEvidence::low
                                   : ModeEvidence::medium;
    const RefreshMode* refresh = select_refresh_mode(host, floor);
    // A host slower than every enumerant we will advertise (a 30 Hz panel) keeps the fallback
    // refresh. Its resolution is still derived: the two halves fail independently, and there is no
    // reason to lie about both because we could not answer one.
    const Resolution resolution = select_output_resolution(host);
    mode.width = resolution.width;
    mode.height = resolution.height;
    if (refresh) {
        mode.refresh_enum = refresh->enumerant;
        mode.vblank_period_ns = refresh->period_ns;
    }
    mode.derived = true;
    return mode;
}

}  // namespace prosper::hle::graphics
