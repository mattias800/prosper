// test_float_controls — every recompiled GUEST module must declare SignedZeroInfNanPreserve.
//
// WHY THIS IS A CONTRACT AND NOT A NICETY (#3479). RDNA2 VALU arithmetic defines Inf, NaN and
// signed zero exactly, and guest programs rely on it. The case that found this: Unity's `sign()`
// idiom synthesises +Inf with integer shifts (((1<<8)-1) << 23 == 0x7F800000), multiplies the
// operand by it, then clamps to [0,1] and truncates. Vulkan does NOT guarantee Inf/NaN preservation
// by default, so a module without this execution mode lets the host driver compile under no-Inf
// assumptions. Measured on Windows/NVIDIA: that multiply yields 0 rather than +Inf, so sign()
// answers 0 everywhere, PPSA02664's colour-grading LUT builder writes an all-black 1024x32 LUT, and
// every graded pixel of the scene composites to black while the UI -- composited after the grade --
// stays pixel-perfect. RADV preserves Inf, so the identical build renders the identical frame
// correctly on Linux/AMD.
//
// That asymmetry is exactly why the guard here is STRUCTURAL rather than an execution assertion.
// An "Inf * x == Inf" kernel passes on RADV and on Mesa lavapipe (what CI uses) whether or not the
// mode is declared, because those implementations preserve Inf anyway -- so it could never go red
// where this project's automated checks run. Asserting on the declaration itself fails on every
// platform when the declaration is removed, which is the property a regression arm needs.
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {
int failures = 0;
int checks = 0;

// Counts assertions ACTUALLY EXECUTED. A deleted block leaves `failures` at zero, which is
// indistinguishable from a passing run; the count is the falsifiable half.
void check(bool ok, const char* label) {
    ++checks;
    if (ok) { std::printf("  [ok]   %s\n", label); return; }
    ++failures;
    std::printf("  [FAIL] %s\n", label);
}

constexpr uint32_t kOpExtension = 10;
constexpr uint32_t kOpEntryPoint = 15;
constexpr uint32_t kOpExecutionMode = 16;
constexpr uint32_t kOpCapability = 17;
// SPV_KHR_float_controls numbers its five capabilities consecutively -- 4464 DenormPreserve,
// 4465 DenormFlushToZero, 4466 SignedZeroInfNanPreserve, 4467 RoundingModeRTE, 4468
// RoundingModeRTZ -- and both this test and the emitter first carried 4467. The test passed:
// an assertion that shares the emitter's constant cannot detect a wrong constant, only a wrong
// presence. `spv_validate` (spirv-val, which shares nothing with either) is what caught it, and
// still covers that axis; this literal is checked against the published SPIR-V grammar.
constexpr uint32_t kCapSignedZeroInfNanPreserve = 4466;
constexpr uint32_t kExecModeSignedZeroInfNanPreserve = 4461;

struct ModuleFacts {
    bool parsed = false;
    bool capability = false;
    bool extension = false;
    bool execution_mode_32 = false;
    bool execution_mode_names_entry = false;
};

// Walk the module header rather than linking SPIRV-Tools: the three declarations all live in the
// prologue, and a hand walk keeps this test CPU-only and dependency-free.
ModuleFacts scan(const std::vector<uint32_t>& spirv) {
    ModuleFacts facts;
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return facts;
    facts.parsed = true;
    uint32_t entry_function = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t words = spirv[offset] >> 16;
        const uint32_t opcode = spirv[offset] & 0xffffu;
        if (!words || words > spirv.size() - offset) { facts.parsed = false; break; }
        if (opcode == kOpCapability && words == 2 &&
            spirv[offset + 1] == kCapSignedZeroInfNanPreserve)
            facts.capability = true;
        if (opcode == kOpExtension && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            if (std::memchr(text, '\0', bytes) &&
                std::strcmp(text, "SPV_KHR_float_controls") == 0)
                facts.extension = true;
        }
        if (opcode == kOpEntryPoint && words >= 4 && !entry_function)
            entry_function = spirv[offset + 2];
        if (opcode == kOpExecutionMode && words == 4 &&
            spirv[offset + 2] == kExecModeSignedZeroInfNanPreserve) {
            if (spirv[offset + 3] == 32) facts.execution_mode_32 = true;
            if (entry_function && spirv[offset + 1] == entry_function)
                facts.execution_mode_names_entry = true;
        }
        offset += words;
    }
    return facts;
}

void expect_declared(const std::vector<uint32_t>& spirv, const char* stage) {
    const ModuleFacts facts = scan(spirv);
    const std::string prefix(stage);
    check(facts.parsed, (prefix + ": recompiled to a walkable SPIR-V module").c_str());
    check(facts.capability, (prefix + ": declares OpCapability SignedZeroInfNanPreserve").c_str());
    check(facts.extension, (prefix + ": declares OpExtension SPV_KHR_float_controls").c_str());
    check(facts.execution_mode_32,
          (prefix + ": declares SignedZeroInfNanPreserve for 32-bit float").c_str());
    check(facts.execution_mode_names_entry,
          (prefix + ": the execution mode names this module's OpEntryPoint function").c_str());
}
} // namespace

int main() {
    std::printf("== test_float_controls ==\n");

    // Assembled with llvm-mc -mcpu=gfx1030, as elsewhere in this directory.
    // v_add_f32 v0, v0, v1 | v_mul_f32 v0, v0, v2 | s_endpgm
    const uint32_t compute_code[] = { 0x06000300u, 0x10000500u, 0xBF810000u };
    expect_declared(recompile_valu(compute_code, std::size(compute_code),
                                   /*num_inputs=*/3, /*out_vgpr=*/0),
                    "compute");

    // v_mov_b32 v0,0 | v_mov_b32 v1,1.0 | v_mov_b32 v2,0 | v_mov_b32 v3,1.0 | exp mrt0 | s_endpgm
    const uint32_t fragment_code[] = {
        0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,
        0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    expect_declared(recompile_fragment(fragment_code, std::size(fragment_code)), "fragment");

    // v_mov_b32 v0,0 | v_mov_b32 v1,1 | exp pos0 done | s_endpgm
    const uint32_t vertex_code[] = {
        0x7E000280u, 0x7E020281u, 0xF80008CFu, 0x01010000u, 0xBF810000u,
    };
    expect_declared(recompile_vertex(vertex_code, std::size(vertex_code)), "vertex");

    std::printf("== %s (%d checks, %d failures) ==\n",
                failures ? "FAIL" : "PASS", checks, failures);
    // A run that executed nothing is a failure, not a pass: see the comment on `checks`.
    if (checks < 15) {
        std::printf("== FAIL: expected at least 15 executed checks, ran %d ==\n", checks);
        return 1;
    }
    return failures ? 1 : 0;
}
