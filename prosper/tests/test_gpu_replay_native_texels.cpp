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
        // The whole point: 10 and 100 must not render identically, as they would at 8 bits.
        check("10 and 100 are distinguishable", a != b);
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

    // Layout coverage: every enumerator the executor can hand back must be handled, or the
    // formatter silently degrades to "not handled here" on a real target.
    {
        const F all[] = {F::Rgba8Unorm, F::Rgba16Float, F::R11G11B10Float, F::R8Unorm,
                         F::R32Uint, F::R32Float, F::Rg8Unorm, F::Rgba32Float,
                         F::Rg16Float, F::R16Float};
        bool every = true;
        for (F f : all) every = every && native_texel_layout(f).bytes != 0;
        check("every LiveTargetPixelFormat enumerator has a layout", every);
    }

    std::printf("%s\n", failures ? "FAILURES PRESENT" : "all passed");
    return failures ? 1 : 0;
}
