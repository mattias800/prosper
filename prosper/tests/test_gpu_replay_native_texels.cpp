// Native-typed readback formatting (tools/gpu_replay/native_texels.hpp), #3765.
//
// The property under test is the one an RGBA8 rendering destroys. `inspect_rtt_seed` clamps to
// 0..1 and scales to 0..255, so an exposure scalar of 10 and one of 100 both arrive as 255 --
// indistinguishable, and therefore useless for the question they were read to answer (#2790:
// is Sonic Frontiers' band excessive HDR, or excessive exposure?). Every assertion below is
// chosen so that it FAILS if the value is routed through a normalized 8-bit conversion.
#include "../tools/gpu_replay/native_texels.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper::gpu::replay_tool;
using F = prosper::gpu::LiveTargetPixelFormat;

static int failures = 0;

static void check(const char* what, bool ok) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

static bool has(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

// Just the decoded `value=` fields, joined. Comparing whole reports lets an arm pass on the
// `raw=` bits alone: under a full RGBA8 clamp both 10 and 100 decode to the same text, and a
// review found the "distinguishable" arm below still green because the raw words differ. The
// property under test is about the DECODED value, so compare only that.
static std::string values_only(const std::string& report) {
    std::string out;
    size_t at = 0;
    while ((at = report.find("value=", at)) != std::string::npos) {
        at += 6;
        const size_t end = report.find('\n', at);
        out += report.substr(at, end == std::string::npos ? end : end - at);
        out += ';';
    }
    return out;
}

static std::vector<uint8_t> f32_bytes(const std::vector<float>& values) {
    std::vector<uint8_t> out(values.size() * 4);
    for (size_t i = 0; i < values.size(); ++i) std::memcpy(out.data() + i * 4, &values[i], 4);
    return out;
}

int main() {
    std::printf("gpu_replay native texel readback\n");

    // The discriminating case: three values an 8-bit normalized conversion collapses together.
    // 0.5 -> 128, and BOTH 10 and 100 -> 255. Here they must remain distinct and exact.
    {
        const auto bytes = f32_bytes({0.5f, 10.0f});
        const std::string a = format_native_texels(F::R32Float, bytes.data(), bytes.size(), 2, 1, 8);
        const auto bytes2 = f32_bytes({0.5f, 100.0f});
        const std::string b = format_native_texels(F::R32Float, bytes2.data(), bytes2.size(), 2, 1, 8);
        check("0.5 is reported exactly", has(a, "value=0.5"));
        check("10 is reported exactly", has(a, "value=10"));
        check("100 is reported exactly", has(b, "value=100"));
        // The whole point: 10 and 100 must not DECODE identically, as they would at 8 bits.
        // Compare values only -- `a != b` was true even under the clamp, on the raw bits.
        check("10 and 100 decode distinguishably", values_only(a) != values_only(b));
        check("raw bits accompany the value (0x41200000 == 10.0f)", has(a, "raw=41200000"));
    }

    // Values outside [0,1] must survive rather than clamp, in both directions.
    {
        const auto bytes = f32_bytes({-3.25f, 65504.0f});
        const std::string s = format_native_texels(F::R32Float, bytes.data(), bytes.size(), 2, 1, 8);
        check("a negative value is preserved, not clamped to 0", has(s, "value=-3.25"));
        check("a large value is preserved, not clamped to 1", has(s, "value=65504"));
    }

    // The three non-finite cases are named, not spelled by the platform's printf.
    {
        const float nan_v = std::nanf(""), pinf = INFINITY, ninf = -INFINITY;
        const auto bytes = f32_bytes({nan_v, pinf, ninf, 0.0f});
        const std::string s = format_native_texels(F::R32Float, bytes.data(), bytes.size(), 4, 1, 8);
        check("NaN is named", has(s, "value=nan"));
        check("+inf is named", has(s, "value=+inf"));
        check("-inf is named", has(s, "value=-inf"));
        check("zero is still zero", has(s, "value=0\n") || has(s, "value=0 "));
    }

    // A half-float target decodes through the same path.
    {
        std::vector<uint8_t> bytes(2);
        const uint16_t half_two = 0x4000;  // 2.0 in IEEE binary16
        std::memcpy(bytes.data(), &half_two, sizeof half_two);
        const std::string s = format_native_texels(F::R16Float, bytes.data(), bytes.size(), 1, 1, 8);
        check("R16Float 0x4000 decodes to 2", has(s, "value=2") && has(s, "raw=4000"));
    }

    // Never silently empty: an unhandled enumerator and a byte-count mismatch each SAY so,
    // because an empty string here reads exactly like "the target was all zeros".
    {
        const auto bytes = f32_bytes({1.0f});
        const std::string wrong = format_native_texels(F::R32Float, bytes.data(), bytes.size(), 4, 1, 8);
        check("a byte-count mismatch is reported", has(wrong, "wants") && has(wrong, "readback has"));
        const std::string none = format_native_texels(F::R32Float, nullptr, 0, 2, 1, 8);
        check("a null readback is reported", !none.empty() && has(none, "readback has 0"));
    }

    // Bounded output: a large target prints a few texels and says how many it withheld.
    {
        std::vector<uint8_t> bytes(64 * 4, 0);
        const std::string s = format_native_texels(F::R32Float, bytes.data(), bytes.size(), 8, 8, 4);
        check("output is bounded to max_texels", has(s, "texel[3]") && !has(s, "texel[4]"));
        check("the withheld count is stated", has(s, "60 more texel(s) not shown"));
    }

    // An UNHANDLED enumerator must say so. The coverage arm below cannot see this: it passes
    // whenever every current format is handled, which stays true if the "not handled here"
    // message is deleted -- verified by mutation, that deletion survived until this arm existed.
    // The realistic case is a new enumerator added later without touching the switch, so the
    // probe is an out-of-range value (well defined: the enum has a fixed uint8_t underlying type).
    {
        const auto bogus = static_cast<F>(200);
        const std::string s = format_native_texels(bogus, nullptr, 0, 1, 1, 8);
        check("an unhandled format enumerator is reported, not silently empty",
              !s.empty() && has(s, "not handled here") && has(s, "200"));
        check("...and its layout reports zero bytes", native_texel_layout(bogus).bytes == 0);
    }

    // Texel COORDINATES must be row-major over the real width. Nothing asserted this: every
    // fixture above is a single row, where (t % width, t / width) and (t, 0) agree.
    {
        const auto bytes = f32_bytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        const std::string s = format_native_texels(F::R32Float, bytes.data(), bytes.size(), 3, 2, 8);
        check("coordinates are row-major on a multi-row target",
              has(s, "texel[3] (0,1)") && has(s, "texel[5] (2,1)"));
        check("...and the value follows the coordinate", has(s, "texel[4] (1,1) raw=40a00000 value=5"));
    }

    // MULTI-COMPONENT formats must advance per component. Every arm above is single-component,
    // so collapsing all component offsets to 0 -- and even applying a wrong unorm scale --
    // survived the whole suite.
    {
        std::vector<uint8_t> rgba(16);
        const float comps[4] = {0.25f, 4.0f, -1.5f, 8.0f};
        for (int i = 0; i < 4; ++i) std::memcpy(rgba.data() + i * 4, &comps[i], 4);
        const std::string s = format_native_texels(F::Rgba32Float, rgba.data(), rgba.size(), 1, 1, 8);
        check("Rgba32Float decodes each component at its own offset",
              has(s, "value=0.25,4,-1.5,8"));
        std::vector<uint8_t> half(8);
        const uint16_t halves[4] = {0x3C00, 0x4000, 0x4200, 0x4400};  // 1, 2, 3, 4
        for (int i = 0; i < 4; ++i) std::memcpy(half.data() + i * 2, &halves[i], 2);
        const std::string h = format_native_texels(F::Rgba16Float, half.data(), half.size(), 1, 1, 8);
        check("Rgba16Float decodes each component at its own offset", has(h, "value=1,2,3,4"));
        check("...and prints per-component raw words", has(h, "raw=3c00,4000,4200,4400"));
        const uint8_t unorm[4] = {0, 51, 204, 255};
        const std::string u = format_native_texels(F::Rgba8Unorm, unorm, sizeof unorm, 1, 1, 8);
        // 255 is the divisor, not 256: under /256 these read 0.19921875 / 0.796875 / 0.99609375.
        check("Rgba8Unorm normalizes by 255, not 256",
              has(u, "value=0,0.200000003,0.800000012,1") && !has(u, "0.99609375"));
    }

    // R32Uint is an integer channel: it must print the integer, not a normalized float.
    {
        const uint32_t big = 4294967295u;
        std::vector<uint8_t> bytes(4);
        std::memcpy(bytes.data(), &big, 4);
        const std::string s = format_native_texels(F::R32Uint, bytes.data(), bytes.size(), 1, 1, 8);
        check("R32Uint prints the integer value", has(s, "value=4294967295"));
    }

    // R11G11B10Float packs three channels in one word: R low, G middle, B high. A wrong channel
    // order survived every arm. One `raw=` for the whole packed word is deliberate.
    {
        // 1.0, 2.0, 0.5 in 11/11/10-bit float, matching gpu_replay.cpp's existing convention.
        const uint32_t packed = 0x702003c0u;
        std::vector<uint8_t> bytes(4);
        std::memcpy(bytes.data(), &packed, 4);
        const std::string s = format_native_texels(F::R11G11B10Float, bytes.data(), bytes.size(), 1, 1, 8);
        check("R11G11B10Float decodes R,G,B in that order", has(s, "value=1,2,0.5"));
        check("...and prints the packed word once", has(s, "raw=702003c0"));
    }

    // Layout coverage: every enumerator the executor can hand back must be handled, or the
    // formatter silently degrades to "not handled here" on a real target.
    {
        const F all[] = {F::Rgba8Unorm, F::Rgba16Float, F::R11G11B10Float, F::R8Unorm,
                         F::R32Uint, F::R32Float, F::Rg8Unorm, F::Rgba32Float,
                         F::Rg16Float, F::R16Float};
        // Calling the formatter, not just reading the layout table: a loop that only checks
        // `layout.bytes != 0` reads like format coverage and exercises none of the decode --
        // the same "looks like coverage" shape already caught one level up in this file.
        bool every = true;
        for (F f : all) {
            const NativeTexelLayout layout = native_texel_layout(f);
            if (!layout.bytes) { every = false; continue; }
            const std::vector<uint8_t> zero(layout.bytes * 2, 0);
            const std::string s =
                format_native_texels(f, zero.data(), zero.size(), 2, 1, 8);
            every = every && has(s, "texel[0] (0,0)") && has(s, "texel[1] (1,0)") &&
                    has(s, layout.name) && !has(s, "not handled here");
        }
        check("every LiveTargetPixelFormat enumerator formats two texels", every);
    }

    // CHOSEN POINTS of a larger target (#3765's follow-up): the first-N report cannot reach the
    // inside of a defect region on a big buffer. Values are the linear index, so a wrong row
    // stride or a swapped x/y produces a different, checkable number.
    {
        std::vector<float> values(4 * 3);
        for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i) + 0.5f;
        const auto bytes = f32_bytes(values);
        const std::vector<NativeTexelPoint> points = {{3, 2}, {1, 0}, {0, 1}, {4, 0}, {0, 3}};
        const std::string s =
            format_native_texels_at(F::R32Float, bytes.data(), bytes.size(), 4, 3, points);
        check("a point report names how many points it was asked for", has(s, "at 5 point(s)"));
        check("(3,2) is linear index 11 on a 4-wide target", has(s, "texel[11] (3,2)") &&
              has(s, "value=11.5"));
        check("(1,0) and (0,1) are not confused", has(s, "texel[1] (1,0)") &&
              has(s, "texel[4] (0,1)") && has(s, "value=4.5"));
        check("a point past the width is reported, not dropped",
              has(s, "(4,0) is outside the 4x3 extent"));
        check("a point past the height is reported, not dropped",
              has(s, "(0,3) is outside the 4x3 extent"));
        check("points past the first-N limit are reachable", !has(s, "more texel(s) not shown"));
        const std::string wrong =
            format_native_texels_at(F::R32Float, bytes.data(), bytes.size() - 4, 4, 3, points);
        check("a point report on a short readback says so",
              has(wrong, "wants 48 bytes, readback has 44"));
        const std::string bogus = format_native_texels_at(static_cast<F>(200), bytes.data(),
                                                          bytes.size(), 4, 3, points);
        check("a point report on an unhandled format says so", has(bogus, "not handled here"));
    }

    std::printf("%s\n", failures ? "FAILURES PRESENT" : "all passed");
    return failures ? 1 : 0;
}
