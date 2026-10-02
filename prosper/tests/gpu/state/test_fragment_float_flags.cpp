#include "gpu/state/render_state.hpp"
#include "gpu/pm4/pm4_registers.hpp"

#include <cstdio>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

namespace {
unsigned checks = 0;
unsigned failures = 0;

void check(bool condition, const char* message, unsigned mode = 0, unsigned flags = 0) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "[FAIL] %s (FLOAT_MODE=0x%02x flags=%u)\n",
                     message, mode, flags);
    }
}
}

int main() {
    // Register absence and an observed zero are independent producing inputs.
    const auto absent = extract_render_state(GpuState{});
    check(absent.ps_float_flags == FragmentFloatFlags{}, "absent launch flags are unknown");
    check(absent.ps_float_flags.canonical(), "absent launch flags are canonical");
    check(!absent.ps_float_mode.available, "absent launch FLOAT_MODE is unknown");
    check(absent.ps_launch_rsrc1==FragmentLaunchRsrc1{}, "absent full word remains unavailable");

    for (unsigned mode = 0; mode < 256; ++mode) {
        for (unsigned flags = 0; flags < 4; ++flags) {
            GpuState launch;
            // Literal published positions: FLOAT_MODE [19:12], DX10_CLAMP 21,
            // DEBUG_MODE 22 and IEEE_MODE 23. Expected fields never use PM4_FIELD.
            launch.sh[P::SPI_SHADER_PGM_RSRC1_PS] =
                (mode << 12) | ((flags & 1u) << 21) | ((flags & 2u) << 22) |
                (1u << 22) | 0x3fu;
            const auto decoded = extract_render_state(launch);
            const FragmentFloatFlags expected{true, (flags & 2u) != 0, (flags & 1u) != 0};
            check(decoded.ps_float_flags == expected, "both launch flags survive extraction",
                  mode, flags);
            check(decoded.ps_float_flags.canonical(), "observed flags are canonical", mode, flags);
            check(decoded.ps_float_mode == FragmentFloatMode{true, static_cast<uint8_t>(mode)},
                  "launch flags do not contaminate FLOAT_MODE", mode, flags);
            check(decoded.ps_launch_rsrc1==FragmentLaunchRsrc1{true,launch.sh.at(P::SPI_SHADER_PGM_RSRC1_PS)},
                  "complete actual word survives independently of derived policy fields",mode,flags);
        }
    }

    GpuState observed_zero;
    observed_zero.sh[P::SPI_SHADER_PGM_RSRC1_PS] = 0;
    const auto zero = extract_render_state(observed_zero);
    check(zero.ps_float_flags == FragmentFloatFlags{true, false, false},
          "observed zero is known flags-clear");
    check(zero.ps_float_mode == FragmentFloatMode{true, 0}, "observed zero has known FLOAT_MODE");
    check(zero.ps_launch_rsrc1==FragmentLaunchRsrc1{true,0}, "observed zero is available raw evidence");
    for (uint32_t word : {0u,1u<<29,UINT32_MAX}) {
        GpuState launch; launch.sh[P::SPI_SHADER_PGM_RSRC1_PS]=word;
        check(extract_render_state(launch).ps_launch_rsrc1==FragmentLaunchRsrc1{true,word},
              "non-policy and future-policy bits survive byte-exactly as evidence");
    }
    check(!(FragmentLaunchRsrc1{false,1}).canonical(), "unavailable raw payload rejects");

    // Direct compiler/capture callers may know the flags without knowing FLOAT_MODE.
    const FragmentFloatFlags independently_known{true, true, true};
    check(independently_known.canonical() && !FragmentFloatMode{}.available,
          "flag authority is independent of mode authority");
    check(!(FragmentFloatFlags{false, true, false}).canonical(), "unknown IEEE payload rejects");
    check(!(FragmentFloatFlags{false, false, true}).canonical(), "unknown DX10 payload rejects");
    check(!(FragmentFloatFlags{false, true, true}).canonical(), "both unknown payloads reject");
    std::printf("fragment_float_flags: %u checks, %u failures; 1024 actual register extractions\n",
                checks, failures);
    return failures ? 1 : 0;
}
