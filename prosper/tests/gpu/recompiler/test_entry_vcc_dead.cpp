// test_entry_vcc_dead — the CFG dispatcher's `missing-entry-vcc` gate, and the one case it may
// stop refusing (#3231).
//
// THE GATE. `emit_cfg_state_machine` stores one value into `vcc_var` before its dispatcher loop.
// When the caller's `RegState` carries no VCC value — because the guest recycled the physical
// s[106:107] pair as ordinary scalar data — there is nothing honest to store, and persisting
// `false` would hand a fabricated mask to whatever reads VCC next. That is silent-wrong, not a
// crash, so the dispatcher refuses the whole region:
//
//     if (!initial.vcc && !proven_wave32_masks && !entry_vcc_dead)
//         return reject_cfg(ins.front().pc, "missing-entry-vcc");
//
// THE RELAXATION. `entry_vcc_dead` is new. Block 0 runs first, exactly once, with every invocation
// active, so a stored entry value is observable only until block 0 overwrites it. If block 0
// DEFINES the whole VCC pair before anything in it can read VCC, the entry value is dead on every
// path and storing `false` invents nothing. Astro Bot's world-map lighting consumer
// `0x500571000` is exactly that shape and was declined on all 830 of its dispatches.
//
// THE ROUTE IS THE POINT, twice over.
//
// 1. Reaching the gate at all needs `initial.vcc` to be ABSENT, and a whole-program compute region
//    cannot be: `recompile_compute` seeds `rs.vcc = b.bfalse()`. Nor is a plain scalar overwrite of
//    the pair enough — in Wave64 `record_scalar_write` leaves `rs.vcc` alone, because the physical
//    pair's data/mask domain is decided by the block-entry MUST analysis instead. The value goes
//    missing exactly where the subject loses it: on the way OUT of a dispatcher region, where
//    `emit_cfg_state_machine` rebuilds its caller's state with `load_state()` and republishes VCC
//    only if the terminal block's `wave64_b64_mask_in` still contains s106. So every program here is
//    BARRIER-PHASED with a proven terminal guard and TWO barriers, like the subject: phase A retires
//    the pair to scalar data and exits through the dispatcher, and phase B — the region under test —
//    then begins with no VCC value at all. Control A is the minimal pair that proves it: change the
//    one dword `s_mov_b32 vcc_lo, 5` to `s_mov_b32 s0, 5` and the same program compiles, because the
//    gate is never consulted.
//
// 2. The tail must lower through the CFG DISPATCHER and nothing else, or an arm's verdict could
//    come from the structurizer instead. Each tail is portable compute (`native_subgroup_size`
//    stays 0), contains one structured forward SCC if, and contains a `v_readlane_b32` whose source
//    is not a writelane spill array — `portable_compute_cfg_readlane` in `emit_body`, the one
//    dispatcher entry with no fallback after a clean reject. Asserted rather than assumed: every
//    compiling arm must contain an `OpSwitch`, which only the dispatcher emits.
//
// Every arm reports independently instead of returning at the first failure, so a mutation run can
// see which arms moved.
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static bool has_opcode(const std::vector<uint32_t>& spv, uint32_t opcode) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t wc = spv[i] >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if ((spv[i] & 0xffffu) == opcode) return true;
        i += wc;
    }
    return false;
}

static bool has(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

constexpr uint32_t kOpSwitch = 251;

// Distinct non-zero program addresses per arm: reject reasons are keyed by address and a zero
// address is not recorded at all, so sharing one would let a later arm overwrite an earlier verdict.
static std::vector<uint32_t> compile(const uint32_t* code, size_t dwords, uint64_t program) {
    ComputeShaderConfig config;   // portable Wave64 compute: native_subgroup_size stays 0
    return recompile_compute(code, dwords, nullptr, config,
                             {RecompileDiagnosticStage::Compute, program});
}

// ---------------------------------------------------------------------------------------------
// Encodings, all verified against a decode of the subject program's own dwords.
//
//   0xBEEA0385  s_mov_b32 vcc_lo, 5              SOP1  op=0x03 sdst=106 ssrc0=inline 5
//   0xBE800385  s_mov_b32 s0, 5                  the same instruction into an ordinary SGPR
//   0xBF068004  s_cmp_eq_u32 s4, 0               SOPC  op=0x06
//   0xBF84NNNN  s_cbranch_scc0 +NNNN             SOPP  op=0x04
//   0xBF8A0000  s_barrier                        SOPP  op=0x0a
//   0xBF810000  s_endpgm                         SOPP  op=0x01
//   0x7D860080  v_cmp_le_u32_e32 vcc, 0, v0      VOPC  op=0xc3 dst=vcc  (the subject's own define
//                                                shape: pc 322 is `7d86300c`, the same opcode)
//   0x02060300  v_cndmask_b32_e32 v3, v0, v1     VOP2  op=0x01 — an IMPLICIT VCC read
//   0x7E040281  v_mov_b32 v2, 1
//   0xd7600002,0x00010100  v_readlane_b32 s2, v0, 0
//   0x4A020000  v_add_nc_u32 v1, s0, v0
// ---------------------------------------------------------------------------------------------

int main() {
    printf("== test_entry_vcc_dead ==\n");

    // ---------------------------------------------------------------------------------------------
    // Every program below is the same nineteen-dword skeleton and differs only inside PHASE B. The
    // shape is the subject's, reduced: a proven terminal scalar guard, then TWO barriers, so there
    // are three regions.
    //
    //   pc0   s_mov_b32 s0, 5             scalar prefix (the guard's uniformity proof needs one)
    //   pc1   s_cmp_eq_u32 s4, 0
    //   pc2   s_cbranch_scc0 -> s_endpgm  the proven terminal guard
    //   pc3   s_barrier                   boundary 1
    //   -- PHASE A: reaches the dispatcher, and RETIRES the VCC pair --
    //   pc4   s_mov_b32 vcc_lo, 5         the physical pair becomes ordinary scalar data
    //   pc5   s_cmp_eq_u32 s4, 0
    //   pc6   s_cbranch_scc0 -> pc8       one structured forward if
    //   pc7   v_mov_b32 v2, 1
    //   pc8   v_readlane_b32 s2, v0, 0    the portable-readlane dispatcher route
    //   pc10  s_barrier                   boundary 2
    //   -- PHASE B: the region under test --
    //   pc11  <the variable>
    //   ...   s_cmp / s_cbranch / v_mov / v_readlane / v_add_nc_u32
    //   pc18  s_endpgm
    //
    // Phase A is what makes the gate reachable, and it has to be a DISPATCHER region rather than a
    // straight-line one. `emit_cfg_state_machine` hands its caller back a state rebuilt by
    // `load_state()`, which only republishes VCC when the terminal block's Wave64 MUST set still
    // calls s106 a mask. `s_mov_b32 vcc_lo, 5` takes it out of that set, so phase B's region begins
    // with no VCC value at all — exactly the state the subject reaches across its own two barriers.
    // ---------------------------------------------------------------------------------------------

    // THE ARM. Phase B's entry block is pc11..pc13, and `v_cmp_le_u32 vcc, 0, v0` at pc11 defines
    // the whole pair before anything in that block reads VCC. The entry value is dead, so the
    // region must compile.
    const uint32_t define_before_read[] = {
        0xBE800385u,                            // pc0:  s_mov_b32 s0, 5
        0xBF068004u,                            // pc1:  s_cmp_eq_u32 s4, 0
        0xBF84000Fu,                            // pc2:  s_cbranch_scc0 -> pc18
        0xBF8A0000u,                            // pc3:  s_barrier
        0xBEEA0385u,                            // pc4:  s_mov_b32 vcc_lo, 5
        0xBF068004u,                            // pc5:  s_cmp_eq_u32 s4, 0
        0xBF840001u,                            // pc6:  s_cbranch_scc0 -> pc8
        0x7E040281u,                            // pc7:  v_mov_b32 v2, 1
        0xd7600002u, 0x00010100u,               // pc8:  v_readlane_b32 s2, v0, 0
        0xBF8A0000u,                            // pc10: s_barrier
        0x7D860080u,                            // pc11: v_cmp_le_u32_e32 vcc, 0, v0   THE DEFINE
        0xBF068004u,                            // pc12: s_cmp_eq_u32 s4, 0
        0xBF840001u,                            // pc13: s_cbranch_scc0 -> pc15
        0x7E040281u,                            // pc14: v_mov_b32 v2, 1
        0xd7600002u, 0x00010100u,               // pc15: v_readlane_b32 s2, v0, 0
        0x4A020000u,                            // pc17: v_add_nc_u32 v1, s0, v0
        0xBF810000u,                            // pc18: s_endpgm
    };

    // CONTROL A — the minimal pair. ONE dword differs: phase A writes s0 instead of vcc_lo, so the
    // pair is never recycled, phase B inherits a live entry VCC and the gate is never consulted.
    // Without it the arm's PASS could come from the skeleton, the phase split or the readlane route
    // rather than from the new rule, and the arm would be a tautology wearing a compile.
    const uint32_t entry_vcc_never_retired[] = {
        0xBE800385u, 0xBF068004u, 0xBF84000Fu, 0xBF8A0000u,
        0xBE800385u,                            // pc4:  s_mov_b32 s0, 5   <-- the only change
        0xBF068004u, 0xBF840001u, 0x7E040281u,
        0xd7600002u, 0x00010100u, 0xBF8A0000u,
        0x7D860080u, 0xBF068004u, 0xBF840001u, 0x7E040281u,
        0xd7600002u, 0x00010100u, 0x4A020000u, 0xBF810000u,
    };

    // THE NEGATIVE ARM, and it is the load-bearing one. One instruction is inserted AHEAD of the
    // define: `v_cndmask_b32_e32` consumes VCC implicitly, so the entry value is live and the region
    // must still be refused. Without this arm the change is indistinguishable from deleting the gate.
    const uint32_t read_before_define[] = {
        0xBE800385u, 0xBF068004u,
        0xBF840010u,                            // pc2:  s_cbranch_scc0 -> pc19
        0xBF8A0000u, 0xBEEA0385u, 0xBF068004u, 0xBF840001u, 0x7E040281u,
        0xd7600002u, 0x00010100u, 0xBF8A0000u,
        0x02060300u,                            // pc11: v_cndmask_b32_e32 v3, v0, v1  THE READ
        0x7D860080u,                            // pc12: v_cmp_le_u32_e32 vcc, 0, v0   (too late)
        0xBF068004u,                            // pc13: s_cmp_eq_u32 s4, 0
        0xBF840001u,                            // pc14: s_cbranch_scc0 -> pc16
        0x7E040281u,                            // pc15: v_mov_b32 v2, 1
        0xd7600002u, 0x00010100u,               // pc16: v_readlane_b32 s2, v0, 0
        0x4A020000u,                            // pc18: v_add_nc_u32 v1, s0, v0
        0xBF810000u,                            // pc19: s_endpgm
    };

    // SECOND NEGATIVE ARM — a b32 write of vcc_lo is NOT a define. Half the pair would still carry
    // the entry value, so the region stays refused even though its entry block does "write VCC"
    // before reading it. A rule that accepted any write to s106 would pass the arm above and fail
    // this one.
    const uint32_t half_pair_write_only[] = {
        0xBE800385u, 0xBF068004u, 0xBF84000Fu, 0xBF8A0000u,
        0xBEEA0385u, 0xBF068004u, 0xBF840001u, 0x7E040281u,
        0xd7600002u, 0x00010100u, 0xBF8A0000u,
        0xBEEA0385u,                            // pc11: s_mov_b32 vcc_lo, 5   (b32, not the pair)
        0xBF068004u, 0xBF840001u, 0x7E040281u,
        0xd7600002u, 0x00010100u, 0x4A020000u, 0xBF810000u,
    };

    // THIRD NEGATIVE ARM — the define must be in the ENTRY block, not merely somewhere in the
    // region. Here phase B's entry block is only the compare and its branch; the `v_cmp` sits in the
    // fall-through successor, so the taken edge leaves the entry block without passing a define.
    const uint32_t define_after_entry_block[] = {
        0xBE800385u, 0xBF068004u, 0xBF84000Fu, 0xBF8A0000u,
        0xBEEA0385u, 0xBF068004u, 0xBF840001u, 0x7E040281u,
        0xd7600002u, 0x00010100u, 0xBF8A0000u,
        0xBF068004u,                            // pc11: s_cmp_eq_u32 s4, 0     entry block starts
        0xBF840001u,                            // pc12: s_cbranch_scc0 -> pc14 entry block ends
        0x7D860080u,                            // pc13: v_cmp_le_u32 vcc, 0, v0  (successor block)
        0x7E040281u,                            // pc14: v_mov_b32 v2, 1
        0xd7600002u, 0x00010100u,               // pc15: v_readlane_b32 s2, v0, 0
        0x4A020000u,                            // pc17: v_add_nc_u32 v1, s0, v0
        0xBF810000u,                            // pc18: s_endpgm
    };

    const auto arm = compile(define_before_read,
                             std::size(define_before_read), 0x32310001ull);
    const auto control_a = compile(entry_vcc_never_retired,
                                   std::size(entry_vcc_never_retired), 0x32310002ull);
    const auto negative = compile(read_before_define,
                                  std::size(read_before_define), 0x32310003ull);
    const auto half_pair = compile(half_pair_write_only,
                                   std::size(half_pair_write_only), 0x32310004ull);
    const auto late_define = compile(define_after_entry_block,
                                     std::size(define_after_entry_block), 0x32310005ull);

    // The control first: an arm whose control has not been read is a verdict with no denominator.
    CHECK(!control_a.empty(),
          "control A: the same skeleton compiles when the entry VCC was never retired");
    CHECK(has_opcode(control_a, kOpSwitch),
          "control A: it lowered through the CFG DISPATCHER (the module contains an OpSwitch)");

    CHECK(!arm.empty(),
          "#3231: an entry block that DEFINES the VCC pair before reading it compiles");
    CHECK(has_opcode(arm, kOpSwitch),
          "#3231: and it too lowered through the CFG dispatcher, not the structurizer");

    CHECK(negative.empty(),
          "#3231: an entry block that READS VCC before defining it is still refused");
    const std::string negative_reason = last_terminal_reject_reason(0x32310003ull);
    CHECK(has(negative_reason, "missing-entry-vcc"),
          "#3231: and it is THIS gate that refused it, not some other reject on the way");

    CHECK(half_pair.empty(),
          "#3231: a b32 write of vcc_lo alone does not make the entry pair dead");
    CHECK(has(last_terminal_reject_reason(0x32310004ull), "missing-entry-vcc"),
          "#3231: the half-pair refusal is this gate as well");

    CHECK(late_define.empty(),
          "#3231: a define in a SUCCESSOR block does not make the entry value dead");
    CHECK(has(last_terminal_reject_reason(0x32310005ull), "missing-entry-vcc"),
          "#3231: the successor-define refusal is this gate as well");

    if (fails) {
        printf("  [info] arm reject reason:           '%s'\n",
               last_terminal_reject_reason(0x32310001ull).c_str());
        printf("  [info] control A reject reason:     '%s'\n",
               last_terminal_reject_reason(0x32310002ull).c_str());
        printf("  [info] negative reject reason:      '%s'\n", negative_reason.c_str());
        printf("  [info] half-pair reject reason:     '%s'\n",
               last_terminal_reject_reason(0x32310004ull).c_str());
        printf("  [info] late-define reject reason:   '%s'\n",
               last_terminal_reject_reason(0x32310005ull).c_str());
        printf("== FAIL: %d ==\n", fails);
        return 1;
    }
    printf("== PASS ==\n");
    return 0;
}
