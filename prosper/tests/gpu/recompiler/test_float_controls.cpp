// test_float_controls — a recompiled GUEST module declares SignedZeroInfNanPreserve when, and only
// when, a device owner has published that the device can execute it.
//
// WHY THE DECLARATION IS A CONTRACT AND NOT A NICETY (#3479). RDNA2 VALU arithmetic defines Inf, NaN
// and signed zero exactly, and guest programs rely on it. The case that found this: Unity's `sign()`
// idiom synthesises +Inf with integer shifts (((1<<8)-1) << 23 == 0x7F800000), multiplies the
// operand by it, then clamps to [0,1] and truncates. Vulkan does NOT guarantee Inf/NaN preservation
// by default, so a module without this execution mode lets the host driver compile under no-Inf
// assumptions. Measured on Windows/NVIDIA: that multiply yields 0 rather than +Inf, so sign()
// answers 0 everywhere, PPSA02664's colour-grading LUT builder writes an all-black 1024x32 LUT, and
// every graded pixel of the scene composites to black while the UI -- composited after the grade --
// stays pixel-perfect. RADV preserves Inf, so the identical build renders the identical frame
// correctly on Linux/AMD.
//
// WHY THE GATE IS A CONTRACT TOO (#3561). Declaring a capability the device does not satisfy makes
// the module INVALID, not merely unoptimised: VUID-VkShaderModuleCreateInfo-pCode-08740 wants
// shaderSignedZeroInfNanPreserveFloat{16,32,64} == VK_TRUE and -08742 wants Vulkan 1.2 or
// VK_KHR_shader_float_controls. The first revision declared it unconditionally and asserted in a
// comment that a device reporting VK_FALSE would make "pipeline creation fail loudly". It does not:
// CI's validation scan reported both VUIDs x475 across four binaries WHILE ALL 464 TESTS PASSED.
// So the negative arms below are not symmetry for its own sake -- they are the only thing that can
// tell a gate that was written from a gate that was not, and the failure they guard is silent.
//
// THE GUARD IS STRUCTURAL, NOT AN EXECUTION ASSERTION, and that is the important judgement here.
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
void check(bool ok, const std::string& label) {
    ++checks;
    if (ok) { std::printf("  [ok]   %s\n", label.c_str()); return; }
    ++failures;
    std::printf("  [FAIL] %s\n", label.c_str());
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

// The four entry points that call declare_float_controls(). Every stage is rebuilt inside each arm:
// the gate is read at EMIT time, so a module carried over from a previous arm would answer the
// previous arm's question.
struct StageModules {
    std::vector<uint32_t> compute, fragment, vertex, geometry;
};

StageModules emit_every_stage() {
    StageModules modules;

    // Assembled with llvm-mc -mcpu=gfx1030, as elsewhere in this directory.
    // v_add_f32 v0, v0, v1 | v_mul_f32 v0, v0, v2 | s_endpgm
    static const uint32_t compute_code[] = { 0x06000300u, 0x10000500u, 0xBF810000u };
    modules.compute = recompile_valu(compute_code, std::size(compute_code),
                                     /*num_inputs=*/3, /*out_vgpr=*/0);

    // v_mov_b32 v0,0 | v_mov_b32 v1,1.0 | v_mov_b32 v2,0 | v_mov_b32 v3,1.0 | exp mrt0 | s_endpgm
    static const uint32_t fragment_code[] = {
        0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,
        0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    modules.fragment = recompile_fragment(fragment_code, std::size(fragment_code));

    // v_mov_b32 v0,0 | v_mov_b32 v1,1 | exp pos0 done | s_endpgm
    static const uint32_t vertex_code[] = {
        0x7E000280u, 0x7E020281u, 0xF80008CFu, 0x01010000u, 0xBF810000u,
    };
    modules.vertex = recompile_vertex(vertex_code, std::size(vertex_code));

    // The fourth entry point, and the one the first revision of this test missed. The generated
    // interpolation geometry stage declares float controls from its own begin*(), twenty lines
    // before Cap_Geometry rather than after Cap_Shader like the other three, so it is the stage
    // most likely to drift. A PS reading attribute 0 through BOTH a P0 parameter fetch and smooth
    // interpolation is what makes the layout require the generated stage at all.
    static const uint32_t mixed_ps[] = {
        0xc80e0002u, 0xc8110002u, 0xf800000fu, 0x03030303u, 0xbf810000u,
    };
    const FragmentInterpolationLayout layout =
        fragment_interpolation_layout(mixed_ps, std::size(mixed_ps));
    modules.geometry = recompile_interpolation_geometry(layout);
    return modules;
}

void expect_declared(const std::vector<uint32_t>& spirv, const std::string& stage) {
    const ModuleFacts facts = scan(spirv);
    check(facts.parsed, stage + ": recompiled to a walkable SPIR-V module");
    check(facts.capability, stage + ": declares OpCapability SignedZeroInfNanPreserve");
    check(facts.extension, stage + ": declares OpExtension SPV_KHR_float_controls");
    check(facts.execution_mode_32,
          stage + ": declares SignedZeroInfNanPreserve for 32-bit float");
    check(facts.execution_mode_names_entry,
          stage + ": the execution mode names this module's OpEntryPoint function");
}

// The negative arm asserts all THREE declarations are absent, not just the capability. They fail
// independently against different VUIDs -- the capability against -08740 (the device property) and
// the extension against -08742 (the API version) -- so a gate that suppressed only one would still
// emit a module the device must reject, and an arm checking only one could not tell.
void expect_not_declared(const std::vector<uint32_t>& spirv, const std::string& stage) {
    const ModuleFacts facts = scan(spirv);
    check(facts.parsed, stage + ": still recompiles to a walkable SPIR-V module");
    check(!facts.capability, stage + ": no OpCapability SignedZeroInfNanPreserve");
    check(!facts.extension, stage + ": no OpExtension SPV_KHR_float_controls");
    check(!facts.execution_mode_32 && !facts.execution_mode_names_entry,
          stage + ": no SignedZeroInfNanPreserve execution mode");
}

void assert_all_stages(const char* arm, bool declared) {
    std::printf("-- %s --\n", arm);
    const StageModules modules = emit_every_stage();
    const std::string prefix(arm);
    auto expect = declared ? expect_declared : expect_not_declared;
    expect(modules.compute, prefix + " compute");
    expect(modules.fragment, prefix + " fragment");
    expect(modules.vertex, prefix + " vertex");
    expect(modules.geometry, prefix + " geometry");
}
} // namespace

int main() {
    std::printf("== test_float_controls ==\n");

    // ARM 1 -- a device that can execute the mode. What the live renderer publishes on NVIDIA and
    // on RADV, and the configuration the #3479 fix depends on.
    reset_float_controls_support_for_test();
    publish_float_controls_support(/*signed_zero_inf_nan_preserve_float32=*/true,
                                   /*declaration_permitted_by_api=*/true);
    check(signed_zero_inf_nan_preserve_declared(),
          "gate: a device that preserves Inf and permits the extension declares it");
    assert_all_stages("supported:", /*declared=*/true);

    // ARM 2 -- VUID-...-08740. The device reports shaderSignedZeroInfNanPreserveFloat32 == VK_FALSE.
    // Nothing may be declared: the module would be invalid, and -- measured, not assumed -- drivers
    // execute such a module anyway, so no run would report it.
    reset_float_controls_support_for_test();
    publish_float_controls_support(/*signed_zero_inf_nan_preserve_float32=*/false,
                                   /*declaration_permitted_by_api=*/true);
    check(!signed_zero_inf_nan_preserve_declared(),
          "gate: shaderSignedZeroInfNanPreserveFloat32 == VK_FALSE suppresses the declaration");
    assert_all_stages("property-false:", /*declared=*/false);

    // ARM 3 -- VUID-...-08742. The device preserves Inf but is below Vulkan 1.2 with
    // VK_KHR_shader_float_controls not enabled, so the extension may not be named on it at all.
    // This is the half that four CI binaries actually tripped: their fixture creates 1.1 devices.
    reset_float_controls_support_for_test();
    publish_float_controls_support(/*signed_zero_inf_nan_preserve_float32=*/true,
                                   /*declaration_permitted_by_api=*/false);
    check(!signed_zero_inf_nan_preserve_declared(),
          "gate: an API version that cannot name the extension suppresses the declaration");
    assert_all_stages("api-forbids:", /*declared=*/false);

    // ARM 4 -- the neutral state. No device owner has published, which is where every offline
    // caller sits: tools, CPU-only tests, an emitter invoked before a device exists. The error
    // direction is deliberate -- an unmeasured device loses the Inf guarantee rather than being
    // handed a module it may not legally compile.
    reset_float_controls_support_for_test();
    check(!signed_zero_inf_nan_preserve_declared(),
          "gate: nothing is declared until some device owner has published");
    assert_all_stages("unpublished:", /*declared=*/false);

    // ARM 5 -- the AND. Two device owners, one of which cannot take the mode; one emitted module
    // has to be legal on both, so the weaker device decides regardless of publication order.
    reset_float_controls_support_for_test();
    publish_float_controls_support(true, true);
    publish_float_controls_support(false, true);
    check(!signed_zero_inf_nan_preserve_declared(),
          "gate: a second, weaker device revokes a declaration the first device allowed");
    reset_float_controls_support_for_test();
    publish_float_controls_support(false, true);
    publish_float_controls_support(true, true);
    check(!signed_zero_inf_nan_preserve_declared(),
          "gate: and it revokes it in the other publication order too");

    std::printf("== %s (%d checks, %d failures) ==\n",
                failures ? "FAIL" : "PASS", checks, failures);
    // A run that executed nothing is a failure, not a pass: see the comment on `checks`.
    constexpr int kExpectedChecks = 74;
    if (checks < kExpectedChecks) {
        std::printf("== FAIL: expected at least %d executed checks, ran %d ==\n",
                    kExpectedChecks, checks);
        return 1;
    }
    return failures ? 1 : 0;
}
