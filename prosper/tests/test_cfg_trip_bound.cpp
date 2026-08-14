// test_cfg_trip_bound — the EXEC-driven pointer-chasing loop, and the diagnostic that bounds it.
//
// WHY THIS SHAPE. GTA V's compute program 0x413dc6700 hangs the GPU into a driver reset, and its
// whole body contains exactly one backward branch. Disassembled from a live run, that loop is:
//
//     pc88  v_mov_b32     v2, s22          ; remember the current depth
//     pc89  v_cmpx_ne_u32 exec, 0, v1      ; EXEC = (index != 0)   <-- the ONLY thing that can exit
//     pc90  s_cbranch_execz -> 98          ; leave once every lane is done
//     pc91  buffer_load_dword v1, v1, ...  ; follow the link
//     pc93  s_add_i32     s22, s22, 1      ; depth++
//     pc95  v_bfe_u32     v1, v1, 3, 27    ; next index out of the record
//     pc97  s_branch      -> 88
//
// Nothing inside the loop writes EXEC except the `v_cmpx`, so termination rests entirely on two
// things being lowered correctly: `v_cmpx` must narrow EXEC (not VCC — on RDNA2 the cmpx forms write
// EXEC, and the e32 encoding's destination field still reads as VCC), and `s_cbranch_execz` must be
// a CROSS-LANE test that becomes true only when every lane has dropped out. Get either wrong and the
// loop cannot end — which is exactly what a two-second dispatch followed by a device reset looks
// like from outside.
//
// The kernel below is a HAND-BUILT positive instance of that class, deliberately not derived from
// the capture, the recompiler, or anything else that produced the observation under test. Its body
// decrements the index instead of chasing a buffer, so the guest's data is out of the picture and
// only the control-flow lowering is on trial: lane i must run exactly i iterations and report depth
// i. That separates "our lowering cannot exit this shape" from "the guest's table is cyclic", which
// no measurement taken on the live program can do, because there both are true at once.
//
// Then, with the shape proven, the same kernel exercises PROSPER_CFG_TRIP_BOUND — the diagnostic
// that caps an emitted back edge. Bounding a loop that terminates ANYWAY is what makes this test
// safe to run in CI: a broken bound yields wrong depths, never a hung queue.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "compute_runner.h"

#include <cmath>
#include <set>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Verified instruction-by-instruction against tools/shader_inspect before being embedded here; the
// decoded listing is reproduced beside each word so a future reader can re-check it without
// reassembling. v1 arrives as a float (the harness loads storage-buffer floats into v0..vN) and
// leaves as one, so the walk is bracketed by converts.
static const uint32_t kExecWalk[] = {
    0x7e020f01u,  //  0: v_cvt_u32_f32  v1, v1          ; index = (uint)input[1]
    0xbe960380u,  //  1: s_mov_b32      s22, 0          ; depth = 0
    0xbeea047eu,  //  2: s_mov_b64      s[106:107], exec; save the incoming mask (as the guest does)
    0x7e040216u,  //  3: v_mov_b32      v2, s22         ; LOOP HEAD: publish the current depth
    0x7daa0280u,  //  4: v_cmpx_ne_u32  exec, 0, v1     ; narrow EXEC to lanes still walking
    0xbf880003u,  //  5: s_cbranch_execz -> 9           ; the only exit
    0x4a0202c1u,  //  6: v_add_nc_u32   v1, -1, v1      ; one step along an acyclic chain
    0x81168116u,  //  7: s_add_i32      s22, s22, 1     ; depth++
    0xbf82fffau,  //  8: s_branch       -> 3            ; back edge
    0xbefe046au,  //  9: s_mov_b64      exec, s[106:107]; restore, so the epilogue store is live
    0x7e040d02u,  // 10: v_cvt_f32_u32  v2, v2
    0xbf810000u,  // 11: s_endpgm
};

// PROSPER_CFG_TRIP_BOUND* are read fresh on every call (see compute_trip_bound_settings), which is
// what lets one process arm, compile, disarm and compile again.
static void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    // MinGW's UCRT has no setenv/unsetenv; _putenv_s with an empty value removes the variable.
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

int main() {
    printf("== test_cfg_trip_bound ==\n");
    set_env("PROSPER_CFG_TRIP_BOUND", nullptr);
    set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", nullptr);
    set_env("PROSPER_CFG_TRIP_BOUND_PHASE", nullptr);

    constexpr uint32_t kLanes = 128;
    const size_t kWords = sizeof(kExecWalk) / sizeof(kExecWalk[0]);

    // Lane i walks a chain of length i. Expected depth is therefore i, and the value is DISTINCT per
    // lane on purpose: a single shared expectation would be satisfied by a lowering that ignores the
    // per-lane mask entirely and runs every lane for the same number of iterations.
    std::vector<float> input(kLanes * 2, 0.0f);
    for (uint32_t lane = 0; lane < kLanes; ++lane) input[lane * 2 + 1] = static_cast<float>(lane);

    const std::vector<uint32_t> unbounded = recompile_valu(kExecWalk, kWords, 2, 2);
    CHECK(!unbounded.empty() && unbounded[0] == 0x07230203u,
          "the EXEC-walk loop recompiles to a SPIR-V module");
    if (unbounded.empty()) { printf("== FAIL: recompile produced nothing ==\n"); return 1; }

    const std::vector<float> walked =
        prosper::test::run_compute(unbounded, input, kLanes, kLanes);
    CHECK(walked.size() == kLanes, "the EXEC-walk loop runs to completion on real Vulkan");

    uint32_t wrong = 0;
    uint32_t first_wrong_lane = kLanes;
    for (uint32_t lane = 0; lane < kLanes && walked.size() == kLanes; ++lane) {
        if (std::fabs(walked[lane] - static_cast<float>(lane)) > 0.5f) {
            if (first_wrong_lane == kLanes) first_wrong_lane = lane;
            ++wrong;
        }
    }
    if (wrong && first_wrong_lane < kLanes)
        printf("  first wrong lane=%u got=%g expected=%u\n", first_wrong_lane,
               walked[first_wrong_lane], first_wrong_lane);
    CHECK(wrong == 0,
          "v_cmpx narrows EXEC and s_cbranch_execz exits only when every lane has dropped out "
          "(lane i walks exactly i steps)");

    // --- the diagnostic ------------------------------------------------------------------------
    // Disarmed, the emitter must be inert: not "close enough", byte-identical. This is the property
    // that lets the bound exist in shipped code at all.
    const std::vector<uint32_t> again = recompile_valu(kExecWalk, kWords, 2, 2);
    CHECK(again == unbounded, "recompilation is deterministic with the bound unset");

    constexpr uint32_t kBound = 4;
    const uint64_t kSyntheticProgram = 0x413dc6700ull;   // the address the live investigation targets
    set_env("PROSPER_CFG_TRIP_BOUND", std::to_string(kBound).c_str());

    // recompile_valu has no diagnostic context, so its program address is 0. Selecting a DIFFERENT
    // program must therefore leave this module untouched — the negative arm for targeting.
    set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", "0x413dc6700");
    const std::vector<uint32_t> other_program = recompile_valu(kExecWalk, kWords, 2, 2);
    CHECK(other_program == unbounded,
          "a bound aimed at another program leaves this module byte-identical");

    // The settings struct is what the shader cache key mixes in. Assert it reports the selector
    // faithfully, because a key built from a struct that under-reports would silently share one
    // compiled module between a targeted and an untargeted program.
    {
        const ComputeTripBoundSettings armed = compute_trip_bound_settings();
        CHECK(armed.bound == kBound && armed.only_program == kSyntheticProgram &&
                  armed.only_phase == ComputeTripBoundSettings::kAllPhases,
              "compute_trip_bound_settings reports the complete armed selector state");
    }

    // Aimed at EVERY program — and this kernel is still not bounded, which is the documented scope
    // and is asserted here rather than assumed. PROSPER_CFG_TRIP_BOUND covers the CFG DISPATCHER's
    // back edge only; the two structured loop emitters are untouched. The distinction is the whole
    // reason the bound is usable as evidence: a null result from a structured-loop program means
    // "not measured", never "measured and did not run away". An earlier revision of the emitter's
    // comment claimed all three emitters were covered while calling the helper from one, and that
    // claim would have turned a structurally-unmeasurable program into a clean bill of health.
    set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", nullptr);
    const std::vector<uint32_t> bounded = recompile_valu(kExecWalk, kWords, 2, 2);
    CHECK(!bounded.empty(), "the bounded kernel still recompiles");
    CHECK(bounded == unbounded,
          "a structured loop is NOT bounded — the diagnostic covers only the CFG dispatcher");

    const std::vector<float> still_walked =
        prosper::test::run_compute(bounded, input, kLanes, kLanes);
    uint32_t truncated = 0;
    for (uint32_t lane = 0; lane < kLanes && still_walked.size() == kLanes; ++lane)
        if (std::fabs(still_walked[lane] - static_cast<float>(lane)) > 0.5f) ++truncated;
    CHECK(still_walked.size() == kLanes && truncated == 0,
          "and it therefore still reports every lane's true depth while armed");

    // --- the dispatcher path, where the bound DOES apply -----------------------------------------
    //
    // Two overlapping, non-nested EXEC loops: a valid reducible machine CFG that the narrow pattern
    // structurizer declines, so emit_body falls back to emit_cfg_state_machine. This is the shape
    // GTA V's 0x413dc6700 is lowered through (it reaches role=terminal in the structurizer), which is
    // why the bound reaches it there and not in the walk kernel above.
    //
    // Asserted on the emitted MODULE rather than by execution, deliberately. The property under test
    // is "the bound reached this emitter's back edge", and comparing armed against unarmed bytes
    // settles that exactly. Running it could not add anything and would make a lowering defect
    // present as a hung queue in CI instead of a failed assertion.
    static const uint32_t kDispatcherLoops[] = {
        0xBE800380u,  //  0: s_mov_b32      s0, 0
        0x7E020284u,  //  1: v_mov_b32      v1, 4
        0x7DA20200u,  //  2: A: v_cmpx_lt_u32 exec, s0, v1
        0xBF880008u,  //  3: s_cbranch_execz -> 12
        0x7DA20200u,  //  4: B: v_cmpx_lt_u32 exec, s0, v1
        0xBF880008u,  //  5: s_cbranch_execz -> 14
        0x4a0202c1u,  //  6: v_add_nc_u32   v1, -1, v1
        0x7E040200u,  //  7: v_mov_b32      v2, s0
        0x81008100u,  //  8: s_add_i32      s0, s0, 1
        0xBF82FFF8u,  //  9: s_branch       -> 2      ; A back edge
        0x81008100u,  // 10: s_add_i32      s0, s0, 1
        0xBF82FFF8u,  // 11: s_branch       -> 4      ; B back edge, overlaps A without nesting
        0x7E040280u,  // 12: v_mov_b32      v2, 0
        0x7E040280u,  // 13: v_mov_b32      v2, 0
        0xBF810000u,  // 14: s_endpgm
    };
    const size_t kDispatcherWords = sizeof(kDispatcherLoops) / sizeof(kDispatcherLoops[0]);

    set_env("PROSPER_CFG_TRIP_BOUND", nullptr);
    const std::vector<uint32_t> dispatcher_plain =
        recompile_valu(kDispatcherLoops, kDispatcherWords, 2, 2);
    CHECK(!dispatcher_plain.empty(), "the overlapping-EXEC-loop kernel recompiles unarmed");

    set_env("PROSPER_CFG_TRIP_BOUND", std::to_string(kBound).c_str());
    const std::vector<uint32_t> dispatcher_bounded =
        recompile_valu(kDispatcherLoops, kDispatcherWords, 2, 2);
    CHECK(!dispatcher_bounded.empty(), "the overlapping-EXEC-loop kernel recompiles armed");
    CHECK(!dispatcher_plain.empty() && dispatcher_bounded != dispatcher_plain,
          "arming the bound changes the CFG dispatcher's emitted module");

    // THE ARM THAT DISCRIMINATES. "The module changed" and "the module got bigger" do NOT: a
    // reviewer deleted the portable back-edge cap, rebuilt, and both still passed, because the trip
    // counter and the witness stores grow the module on their own. An assertion satisfied by the
    // instrumentation surrounding a fix, rather than by the fix, is exactly the announced-but-inert
    // shape this whole diagnostic exists to catch — so assert at the site.
    //
    // The back edge is an OpBranchConditional whose merge-block target and continue target are named
    // by the enclosing OpLoopMerge. Capped, its CONDITION is the result of the trip comparison;
    // uncapped it is the raw `active` value. Requiring an OpULessThan whose result feeds a
    // conditional branch means deleting the cap fails this even though the counter still exists.
    // Anchored on the one thing ONLY apply_trip_bound emits: an unsigned comparison against the
    // bound's literal value. `trip_var` and its initialising store live outside that helper, so
    // deleting the back-edge call leaves them (and the module still differs, and still grows) while
    // this comparison disappears — which is precisely the mutation that slipped through before.
    //
    // Following the condition all the way to the OpBranchConditional was the first attempt and is
    // wrong: the portable dispatcher routes it through a Function variable, so a store/load pair
    // breaks any def-use walk that does not model memory. Matching the comparison against the bound
    // CONSTANT keeps the anchor at the emitter's own site without needing that machinery.
    auto compares_against_bound = [](const std::vector<uint32_t>& m, uint32_t bound) {
        if (m.size() < 5) return false;
        constexpr uint16_t OpConstant = 43, OpULessThan = 176;
        std::set<uint32_t> bound_constants;
        for (size_t word = 5; word < m.size();) {
            const uint32_t op = m[word] & 0xffffu, len = m[word] >> 16;
            if (len == 0 || word + len > m.size()) break;
            if (op == OpConstant && len == 4 && m[word + 3] == bound)
                bound_constants.insert(m[word + 2]);
            else if (op == OpULessThan && len >= 5 && bound_constants.count(m[word + 4]))
                return true;
            word += len;
        }
        return false;
    };
    CHECK(!compares_against_bound(dispatcher_plain, kBound),
          "unarmed, the dispatcher module contains no comparison against the bound");
    CHECK(compares_against_bound(dispatcher_bounded, kBound),
          "armed, the dispatcher compares the trip counter against the bound "
          "(deleting the cap fails HERE, where module size and inequality do not)");

    // Independent of the scan above: the bound's VALUE must reach the emitter, not merely its
    // armed/disarmed state. Two different bounds must produce two different modules.
    set_env("PROSPER_CFG_TRIP_BOUND", std::to_string(kBound + 1).c_str());
    const std::vector<uint32_t> dispatcher_other_bound =
        recompile_valu(kDispatcherLoops, kDispatcherWords, 2, 2);
    CHECK(dispatcher_other_bound != dispatcher_bounded &&
              compares_against_bound(dispatcher_other_bound, kBound + 1),
          "the bound's value reaches the emitted comparison, not just its armed state");
    set_env("PROSPER_CFG_TRIP_BOUND", std::to_string(kBound).c_str());

    // Targeting again, now on the emitter that actually honours it: a bound aimed elsewhere must
    // leave even a dispatcher module untouched. Without this arm, "arming changes the module" would
    // be satisfied by a bound that ignored PROSPER_CFG_TRIP_BOUND_PROGRAM entirely.
    set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", "0x413dc6700");
    const std::vector<uint32_t> dispatcher_other =
        recompile_valu(kDispatcherLoops, kDispatcherWords, 2, 2);
    CHECK(dispatcher_other == dispatcher_plain,
          "a bound aimed at another program leaves the dispatcher module byte-identical");
    set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", nullptr);

    // --- through the real cache, in BOTH orderings ------------------------------------------------
    //
    // Everything above calls recompile_valu, which never consults the shader cache — so none of it
    // can see the defect the cache key fix addresses. The key is blind to a program's ADDRESS while
    // the selector chooses by address, so two programs with identical bodies collide, and WHICH one
    // compiles first decides who gets the wrong module. Both orderings are exercised because a key
    // that omitted the address would still pass one of them by luck.
    {
        ComputeShaderConfig config{};
        config.local_x = 64; config.local_y = 1; config.local_z = 1;
        config.wave_size = 64;
        config.tgid_x_en = true;
        ShaderResourceTable table{};
        const uint64_t kTarget = 0x413dc6700ull, kOther = 0x413ce3400ull;

        set_env("PROSPER_CFG_TRIP_BOUND", std::to_string(kBound).c_str());
        set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", "0x413dc6700");

        auto compile_as = [&](uint64_t program) {
            return recompile_compute_shader_cached(
                kDispatcherLoops, kDispatcherWords, &table, config, nullptr,
                {RecompileDiagnosticStage::Compute, program});
        };

        clear_shader_recompile_cache();
        const std::vector<uint32_t> target_first_target = compile_as(kTarget);
        const std::vector<uint32_t> target_first_other  = compile_as(kOther);

        clear_shader_recompile_cache();
        const std::vector<uint32_t> other_first_other   = compile_as(kOther);
        const std::vector<uint32_t> other_first_target  = compile_as(kTarget);

        CHECK(!target_first_target.empty() && !other_first_other.empty(),
              "both programs recompile through the cached entry point");
        CHECK(target_first_target != target_first_other,
              "target and non-target get DIFFERENT modules when the target compiles first");
        CHECK(other_first_other != other_first_target,
              "target and non-target get different modules when the non-target compiles first");
        CHECK(target_first_target == other_first_target,
              "the target's module does not depend on compile order");
        CHECK(target_first_other == other_first_other,
              "the non-target's module does not depend on compile order");
        CHECK(compares_against_bound(target_first_target, kBound) &&
                  !compares_against_bound(target_first_other, kBound),
              "through the cache, only the SELECTED program's back edge is capped");
        set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", nullptr);
        clear_shader_recompile_cache();
    }

    // Disarming must restore the original module exactly. Without this, "arming changes the module"
    // above could be satisfied by any nondeterminism in the emitter rather than by the bound.
    set_env("PROSPER_CFG_TRIP_BOUND", nullptr);
    const std::vector<uint32_t> disarmed = recompile_valu(kExecWalk, kWords, 2, 2);
    CHECK(disarmed == unbounded, "disarming restores the byte-identical unbounded module");

    printf(fails ? "== FAILURES: %d ==\n" : "== all passed (%d failures) ==\n", fails);
    return fails ? 1 : 0;
}
