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
#include "../frontends/shared/trip_bound_witness.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "compute_runner.h"

#include <algorithm>
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

    // A PHASE SELECTOR IS REQUIRED. One witness record cannot describe two phases -- their dispatch
    // ordinals index different tables -- so arming without one emits nothing at all. Assert that
    // before anything else, because every arm below depends on the phase being set and would
    // otherwise pass for the wrong reason.
    set_env("PROSPER_CFG_TRIP_BOUND", std::to_string(kBound).c_str());
    set_env("PROSPER_CFG_TRIP_BOUND_PHASE", nullptr);
    CHECK(recompile_valu(kExecWalk, kWords, 2, 2) == unbounded,
          "a bound with no phase selector emits nothing (the record could not be coherent)");
    set_env("PROSPER_CFG_TRIP_BOUND_PHASE", "0");

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
                  armed.only_phase == 0u,
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
        0xBF880009u,  //  3: s_cbranch_execz -> 13
        0x7DA20200u,  //  4: B: v_cmpx_lt_u32 exec, s0, v1
        0xBF880009u,  //  5: s_cbranch_execz -> 15
        0x4a0202c1u,  //  6: v_add_nc_u32   v1, -1, v1
        0x7E040200u,  //  7: v_mov_b32      v2, s0      ; publishes the iteration reached
        0x81008100u,  //  8: s_add_i32      s0, s0, 1
        0xBF82FFF8u,  //  9: s_branch       -> 2        ; A back edge
        0x81008100u,  // 10: s_add_i32      s0, s0, 1
        0xBF82FFF8u,  // 11: s_branch       -> 4        ; B back edge, overlaps A without nesting
        0x7E040280u,  // 12: v_mov_b32      v2, 0
        0x7E040280u,  // 13: v_mov_b32      v2, 0
        0xBEFE04C1u,  // 14: s_mov_b64      exec, -1    ; restore, so the epilogue store is live
        0x7e040d02u,  // 15: v_cvt_f32_u32  v2, v2
        0xBF810000u,  // 16: s_endpgm
    };
    const size_t kDispatcherWords = sizeof(kDispatcherLoops) / sizeof(kDispatcherLoops[0]);

    set_env("PROSPER_CFG_TRIP_BOUND", nullptr);
    set_env("PROSPER_CFG_TRIP_BOUND_PHASE", nullptr);
    const std::vector<uint32_t> dispatcher_plain =
        recompile_valu(kDispatcherLoops, kDispatcherWords, 2, 2);
    CHECK(!dispatcher_plain.empty(), "the overlapping-EXEC-loop kernel recompiles unarmed");

    set_env("PROSPER_CFG_TRIP_BOUND", std::to_string(kBound).c_str());
    set_env("PROSPER_CFG_TRIP_BOUND_PHASE", "0");
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
    // Exact def-use walk, valid ONLY where the condition reaches the branch without passing through
    // memory -- i.e. the direct/native dispatcher. SPIR-V is in SSA dominance order, so one forward
    // pass resolves it.
    auto condition_reaches_branch = [](const std::vector<uint32_t>& m, uint32_t bound) {
        if (m.size() < 5) return false;
        constexpr uint16_t OpConstant = 43, OpULessThan = 176, OpLogicalAnd = 167,
                           OpBranchConditional = 250;
        std::set<uint32_t> bound_constants, derived;
        for (size_t word = 5; word < m.size();) {
            const uint32_t op = m[word] & 0xffffu, len = m[word] >> 16;
            if (len == 0 || word + len > m.size()) break;
            if (op == OpConstant && len == 4 && m[word + 3] == bound)
                bound_constants.insert(m[word + 2]);
            else if (op == OpULessThan && len >= 5 && bound_constants.count(m[word + 4]))
                derived.insert(m[word + 2]);
            // OpLogicalAnd is `{result-type, result, a, b}` -- length FIVE. An earlier revision
            // required >= 6 here, so this branch never fired and the walk silently reported "the
            // condition never reaches a branch" for every module. A predicate that cannot return
            // true is not a strict test, it is a broken one.
            else if (op == OpLogicalAnd && len >= 5 &&
                     (derived.count(m[word + 3]) || derived.count(m[word + 4])))
                derived.insert(m[word + 2]);
            else if (op == OpBranchConditional && len >= 4 && derived.count(m[word + 1]))
                return true;
            word += len;
        }
        return false;
    };

    // Does the module publish witness slot `slot` with atomic opcode `opcode`? The slot arrives as an
    // OpConstant feeding the AccessChain that the atomic then targets.
    auto witness_uses = [](const std::vector<uint32_t>& m, uint32_t slot, uint16_t opcode) {
        if (m.size() < 5) return false;
        constexpr uint16_t OpConstant = 43, OpAccessChain = 65;
        std::set<uint32_t> slot_constants, slot_pointers;
        for (size_t word = 5; word < m.size();) {
            const uint32_t op = m[word] & 0xffffu, len = m[word] >> 16;
            if (len == 0 || word + len > m.size()) break;
            if (op == OpConstant && len == 4 && m[word + 3] == slot)
                slot_constants.insert(m[word + 2]);
            else if (op == OpAccessChain && len >= 5 && slot_constants.count(m[word + len - 1]))
                slot_pointers.insert(m[word + 2]);
            else if (op == opcode && len >= 4 && slot_pointers.count(m[word + 3]))
                return true;
            word += len;
        }
        return false;
    };

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
          "armed, the dispatcher emits a comparison against the bound");

    // EXECUTION, because emitting a comparison is not the same as that comparison CONTROLLING the
    // loop. A reviewer changed `return land(keep_going, under_bound)` to `return keep_going` --
    // counter, comparison and witness all still emitted, cap completely inert -- and every
    // module-shaped assertion above still passed. Only running it can tell the difference.
    //
    // The fixture terminates on its own (s0 rises while v1 falls, so `s0 < v1` fails on the third
    // pass), which is what makes this safe in CI: an inert cap yields a wrong VALUE, never a hung
    // queue. Unbounded, every lane reports 1 -- the last iteration index reached at pc7. A bound of
    // 4 dispatcher iterations cuts the run short of that.
    std::vector<float> dispatcher_input(kLanes * 2, 0.0f);
    set_env("PROSPER_CFG_TRIP_BOUND", nullptr);
    const std::vector<float> ran_free =
        prosper::test::run_compute(dispatcher_plain, dispatcher_input, kLanes, kLanes);
    set_env("PROSPER_CFG_TRIP_BOUND", std::to_string(kBound).c_str());
    set_env("PROSPER_CFG_TRIP_BOUND_PHASE", "0");
    std::vector<uint32_t> witness_gds;
    const std::vector<float> ran_capped =
        prosper::test::run_compute(dispatcher_bounded, dispatcher_input, kLanes, kLanes,
                                   {}, {}, nullptr, 64, nullptr, &witness_gds);

    const bool free_ok = ran_free.size() == kLanes &&
        std::all_of(ran_free.begin(), ran_free.end(),
                    [](float v) { return std::fabs(v - 1.0f) < 0.5f; });
    CHECK(free_ok, "unbounded, the overlapping-EXEC-loop fixture runs to its natural end (every "
                   "lane reports 1)");
    const bool capped_differs = ran_capped.size() == kLanes && ran_free.size() == kLanes &&
        std::any_of(ran_capped.begin(), ran_capped.end(),
                    [](float v) { return std::fabs(v - 1.0f) >= 0.5f; });
    if (ran_capped.size() == kLanes && ran_free.size() == kLanes)
        printf("  dispatcher fixture: unbounded lane0=%g  bounded lane0=%g\n",
               ran_free[0], ran_capped[0]);
    CHECK(capped_differs,
          "the cap CONTROLS the back edge: bounding truncates the run and changes the result "
          "(an inert cap fails HERE, where an emitted-but-unused comparison does not)");

    // The witness is now a real bound buffer (binding 127), so what the shader wrote is readable
    // rather than inferred. Before this, the armed module was dispatched with that binding absent
    // from the pipeline layout -- VUID-VkComputePipelineCreateInfo-layout-07988 and
    // VUID-vkCmdDispatch-None-08114 -- so every result above was undefined behaviour that happened
    // to print "passed". Reading the record back is what turns the execution arm into evidence.
    const size_t hit_slot = kComputeTripWitnessDword;
    const bool witness_readable = witness_gds.size() > hit_slot + 4;
    CHECK(witness_readable && witness_gds[hit_slot] == 1u,
          "the device-side witness records a HIT when the cap is reached");
    if (witness_readable)
        printf("  witness: hit=%u phase=%u trips=%u range=%u..%u\n", witness_gds[hit_slot],
               witness_gds[hit_slot + 1], witness_gds[hit_slot + 2], witness_gds[hit_slot + 3],
               witness_gds[hit_slot + 4]);
    CHECK(witness_readable && witness_gds[hit_slot + 2] >= kBound,
          "and the trip count it reports reaches the bound");
    CHECK(witness_readable && witness_gds[hit_slot + 3] <= witness_gds[hit_slot + 4],
          "with a dispatch-range whose atomic minimum does not exceed its atomic maximum");

    // Independent of the scan above: the bound's VALUE must reach the emitter, not merely its
    // armed/disarmed state. Two different bounds must produce two different modules.
    set_env("PROSPER_CFG_TRIP_BOUND", std::to_string(kBound + 1).c_str());
    set_env("PROSPER_CFG_TRIP_BOUND_PHASE", "0");
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
        set_env("PROSPER_CFG_TRIP_BOUND_PHASE", "0");

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

    // --- the DIRECT / native-subgroup dispatcher -------------------------------------------------
    //
    // A separate emitter with its own back edge, and removing its apply_trip_bound call was shown to
    // leave every assertion above passing: nothing here reaches it, because recompile_valu never sets
    // a native subgroup size. `direct_dispatch` requires one, so this arm goes through the cached
    // compute entry point with the size set.
    //
    // The direct path passes the capped condition STRAIGHT into emit_condbranch with no Function
    // variable in between, so unlike the portable path a def-use walk to the OpBranchConditional is
    // exact here and needs no memory modelling. That is why the two paths are checked differently:
    // the shape of each lowering decides what can honestly be asserted about it.
    {
        ComputeShaderConfig direct{};
        direct.local_x = 64; direct.local_y = 1; direct.local_z = 1;
        direct.wave_size = 64;
        direct.tgid_x_en = true;
        direct.native_subgroup_size = 64;
        ShaderResourceTable table{};
        const uint64_t kTarget = 0x413dc6700ull;

        auto compile_direct = [&] {
            clear_shader_recompile_cache();
            return recompile_compute_shader_cached(
                kDispatcherLoops, kDispatcherWords, &table, direct, nullptr,
                {RecompileDiagnosticStage::Compute, kTarget});
        };

        set_env("PROSPER_CFG_TRIP_BOUND", nullptr);
        const std::vector<uint32_t> direct_plain = compile_direct();
        set_env("PROSPER_CFG_TRIP_BOUND", std::to_string(kBound).c_str());
        set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", "0x413dc6700");
        set_env("PROSPER_CFG_TRIP_BOUND_PHASE", "0");
        const std::vector<uint32_t> direct_bounded = compile_direct();

        CHECK(!direct_plain.empty() && !direct_bounded.empty(),
              "the native-subgroup dispatcher compiles both unarmed and armed");
        CHECK(!condition_reaches_branch(direct_plain, kBound),
              "unarmed, no branch in the native-subgroup module is gated on the bound comparison");
        CHECK(condition_reaches_branch(direct_bounded, kBound),
              "armed, the native-subgroup back edge BRANCHES on the bound comparison "
              "(removing the direct apply_trip_bound call fails HERE)");
        set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", nullptr);
        set_env("PROSPER_CFG_TRIP_BOUND_PHASE", nullptr);
        clear_shader_recompile_cache();
    }

    // The witness's reduction DIRECTION is part of the contract: field +3 is a minimum and +4 a
    // maximum, and swapping them still produced a plausible-looking record that every other
    // assertion accepted. Pin each opcode to its slot.
    CHECK(witness_uses(dispatcher_bounded, kComputeTripWitnessDword + 3, 237 /*OpAtomicUMin*/),
          "the low end of the dispatch range is published with an atomic MINIMUM");
    CHECK(witness_uses(dispatcher_bounded, kComputeTripWitnessDword + 4, 239 /*OpAtomicUMax*/),
          "the high end of the dispatch range is published with an atomic MAXIMUM");
    CHECK(!witness_uses(dispatcher_bounded, kComputeTripWitnessDword + 3, 239) &&
              !witness_uses(dispatcher_bounded, kComputeTripWitnessDword + 4, 237),
          "and neither slot carries the other's reduction");

    // --- emission RESULT vs. selector INTENT --------------------------------------------------
    //
    // The host decides whether to read and clear five guest-visible GDS dwords. Gating that on the
    // selectors is wrong: a program the selectors accept can emit nothing -- a structured loop, or a
    // phase ordinal the program does not have -- and the host would then report whatever guest data
    // occupied those dwords as a measurement, and destroy it.
    //
    // The predicate reads the COMPILED MODULE. An earlier revision kept a process-global set of
    // addresses that had ever emitted, which cannot express this contract: it was monotonic, so the
    // SAME address recompiled under a phase it does not have still answered "instrumented". The
    // same-address sequence below is exactly that case, and it is the arm that revision failed.
    {
        ComputeShaderConfig config{};
        config.local_x = 64; config.local_y = 1; config.local_z = 1;
        config.wave_size = 64;
        config.tgid_x_en = true;
        ShaderResourceTable table{};
        const uint64_t kAddress = 0x900000001ull;

        // Cache-WARM on purpose. An earlier revision cleared the cache before every compilation,
        // so the phase-0 -> phase-99 sequence was two cold compiles and proved nothing about the
        // cache -- while the implementation and the PR both claimed hits were covered. The trip-bound
        // identity is part of ShaderCompileKey, so changing the phase must MISS and recompile; the
        // repeats around it must HIT and keep their answer.
        auto compile_at = [&](const uint32_t* code, size_t words, uint64_t address) {
            return recompile_compute_shader_cached(code, words, &table, config, nullptr,
                                                   {RecompileDiagnosticStage::Compute, address});
        };

        set_env("PROSPER_CFG_TRIP_BOUND", std::to_string(kBound).c_str());
        set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", nullptr);
        clear_shader_recompile_cache();

        set_env("PROSPER_CFG_TRIP_BOUND_PHASE", "0");
        const std::vector<uint32_t> emitted =
            compile_at(kDispatcherLoops, kDispatcherWords, kAddress);
        CHECK(spirv_writes_trip_witness(emitted),
              "the bounded dispatcher module writes the witness (cold)");
        CHECK(spirv_writes_trip_witness(compile_at(kDispatcherLoops, kDispatcherWords, kAddress)),
              "and an emitting CACHE HIT still writes it");

        // SAME address, cache left warm, phase the program does not have.
        set_env("PROSPER_CFG_TRIP_BOUND_PHASE", "99");
        const std::vector<uint32_t> not_emitted =
            compile_at(kDispatcherLoops, kDispatcherWords, kAddress);
        CHECK(!spirv_writes_trip_witness(not_emitted),
              "the SAME address under a phase it does not have writes no witness, warm "
              "(a process-history emission record fails HERE)");
        CHECK(!spirv_writes_trip_witness(compile_at(kDispatcherLoops, kDispatcherWords, kAddress)),
              "and the non-emitting CACHE HIT still writes none");

        set_env("PROSPER_CFG_TRIP_BOUND_PHASE", "0");
        CHECK(!spirv_writes_trip_witness(compile_at(kExecWalk, kWords, kAddress)),
              "a structured-loop program writes no witness even with every selector satisfied");

        set_env("PROSPER_CFG_TRIP_BOUND_PHASE", nullptr);
        set_env("PROSPER_CFG_TRIP_BOUND", nullptr);
        clear_shader_recompile_cache();
        CHECK(!spirv_writes_trip_witness(compile_at(kDispatcherLoops, kDispatcherWords, kAddress)),
              "and disarmed, the same program writes none either");

        // IDENTITY, not coincidence. The predicate must recognise the witness RESOURCE -- the
        // variable decorated set 0 / binding 127 -- not merely an atomic at the numeric slot index.
        // Both arms mutate the REAL emitted module by one word, so they cannot pass for a reason
        // unrelated to the property under test.
        {
            std::vector<uint32_t> wrong_binding = emitted;
            constexpr uint32_t OpDecorate = 71, DecorationBinding = 33;
            bool rebound = false;
            for (size_t word = 5; word < wrong_binding.size();) {
                const uint32_t op = wrong_binding[word] & 0xffffu, len = wrong_binding[word] >> 16;
                if (!len) break;
                if (op == OpDecorate && len == 4 &&
                    wrong_binding[word + 2] == DecorationBinding &&
                    wrong_binding[word + 3] == kComputeInternalGdsBinding) {
                    wrong_binding[word + 3] = kComputeInternalGdsBinding - 1u;
                    rebound = true;
                    break;
                }
                word += len;
            }
            CHECK(rebound, "the emitted module decorates a binding-127 variable (arm precondition)");
            CHECK(rebound && !spirv_writes_trip_witness(wrong_binding),
                  "moving ONLY the binding decoration off 127 is no longer the witness "
                  "(the predicate resolves the resource, not the slot number)");

            std::vector<uint32_t> bad_magic = emitted;
            bad_magic[0] = 0u;
            CHECK(!spirv_writes_trip_witness(bad_magic),
                  "and a module with a corrupt header is refused rather than scanned");

            // THE ARM THAT ISOLATES THE BASE CHECK, and the reason it is separate from the one
            // above. Moving the binding decoration off 127 leaves the module with NO witness
            // variable, which the predicate's early "no such variable" exit already rejects -- so
            // that arm passes whether or not the access chain's base is ever examined. I verified
            // this by deleting the base check: the wrong-binding arm still passed.
            //
            // The false positive the predicate must actually refuse is a module that HAS a
            // binding-127 variable and performs the witness-slot atomic through a DIFFERENT one --
            // an ordinary storage buffer indexed at the same element number. Build exactly that by
            // repointing the witness access chain's base, one word, in the real emitted module.
            constexpr uint32_t OpDecorateOp = 71, OpVariableOp = 59, OpConstantOp = 43,
                               OpAccessChainOp = 65;
            constexpr uint32_t DecorationBindingId = 33, DecorationDescriptorSetId = 34;
            std::set<uint32_t> bound127, set0, slot_ids, variables;
            for (size_t word = 5; word < emitted.size();) {
                const uint32_t op = emitted[word] & 0xffffu, len = emitted[word] >> 16;
                if (!len) break;
                if (op == OpDecorateOp && len == 4) {
                    if (emitted[word + 2] == DecorationBindingId &&
                        emitted[word + 3] == kComputeInternalGdsBinding) bound127.insert(emitted[word + 1]);
                    if (emitted[word + 2] == DecorationDescriptorSetId && emitted[word + 3] == 0u)
                        set0.insert(emitted[word + 1]);
                } else if (op == OpVariableOp && len >= 4) {
                    variables.insert(emitted[word + 2]);
                } else if (op == OpConstantOp && len == 4 &&
                           emitted[word + 3] == kComputeTripWitnessDword) {
                    slot_ids.insert(emitted[word + 2]);
                }
                word += len;
            }
            uint32_t witness_var = 0;
            for (uint32_t id : bound127) if (set0.count(id)) witness_var = id;
            uint32_t other_var = 0;
            for (uint32_t id : variables) if (id != witness_var) { other_var = id; break; }

            std::vector<uint32_t> rebased = emitted;
            bool repointed = false;
            for (size_t word = 5; word < rebased.size();) {
                const uint32_t op = rebased[word] & 0xffffu, len = rebased[word] >> 16;
                if (!len) break;
                if (op == OpAccessChainOp && len >= 5 && rebased[word + 3] == witness_var &&
                    slot_ids.count(rebased[word + len - 1])) {
                    rebased[word + 3] = other_var;
                    repointed = true;
                }
                word += len;
            }
            CHECK(witness_var && other_var && repointed,
                  "the emitted module has a binding-127 variable, another variable, and a "
                  "witness-slot access chain (arm precondition)");
            CHECK(repointed && !spirv_writes_trip_witness(rebased),
                  "an atomic at the witness SLOT through a different variable is not the witness "
                  "(deleting the access-chain base check fails HERE, where wrong-binding does not)");
        }
    }

    // Guest GDS must be byte-identical after an instrumented dispatch. The witness buffer is ONE
    // persistent allocation shared by every dispatch, so proving the instrumented program does not
    // read GDS says nothing about the program that runs next and does.
    {
        uint8_t* gds = compute_gds_backing();
        uint32_t* slots = reinterpret_cast<uint32_t*>(gds) + kComputeTripWitnessDword;
        const uint32_t sentinel[kComputeTripWitnessDwordCount] = {
            0xA1A1A1A1u, 0xB2B2B2B2u, 0xC3C3C3C3u, 0xD4D4D4D4u, 0xE5E5E5E5u};

        prosper::gpu::ComputeItem item{};
        item.code_addr = 0x900000003ull;

        // Not instrumented: the scope must not touch a single dword.
        for (uint32_t i = 0; i < kComputeTripWitnessDwordCount; ++i) slots[i] = sentinel[i];
        item.trip_witness_instrumented = false;
        { prosper::frontend::TripBoundWitnessScope scope(item); }
        bool untouched = true;
        for (uint32_t i = 0; i < kComputeTripWitnessDwordCount; ++i)
            untouched &= slots[i] == sentinel[i];
        CHECK(untouched, "an uninstrumented dispatch leaves the guest's GDS dwords alone");

        // Instrumented: seeded during the dispatch, restored exactly afterwards.
        item.trip_witness_instrumented = true;
        bool seeded_min_is_identity = false, seeded_hit_cleared = false;
        {
            prosper::frontend::TripBoundWitnessScope scope(item);
            seeded_min_is_identity = slots[3] == UINT32_MAX;
            seeded_hit_cleared = slots[0] == 0;
        }
        CHECK(seeded_hit_cleared && seeded_min_is_identity,
              "an instrumented dispatch starts from a cleared flag and an atomic-min identity");
        bool restored = true;
        for (uint32_t i = 0; i < kComputeTripWitnessDwordCount; ++i)
            restored &= slots[i] == sentinel[i];
        CHECK(restored,
              "and the guest's original GDS dwords are byte-identical afterwards, so a later GDS "
              "consumer is unaffected");
    }

    // Disarming must restore the original module exactly. Without this, "arming changes the module"
    // above could be satisfied by any nondeterminism in the emitter rather than by the bound.
    set_env("PROSPER_CFG_TRIP_BOUND", nullptr);
    const std::vector<uint32_t> disarmed = recompile_valu(kExecWalk, kWords, 2, 2);
    CHECK(disarmed == unbounded, "disarming restores the byte-identical unbounded module");

    printf(fails ? "== FAILURES: %d ==\n" : "== all passed (%d failures) ==\n", fails);
    return fails ? 1 : 0;
}
