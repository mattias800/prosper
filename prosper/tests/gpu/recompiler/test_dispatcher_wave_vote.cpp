// test_dispatcher_wave_vote — the CFG dispatcher must lower a fragment exec/vcc branch as a WAVE
// VOTE, the same way the structured route already does.
//
// Why this exists (#3573). `s_cbranch_execz` / `execnz` / `vccz` / `vccnz` are SCALAR branches: the
// condition is a property of the whole wave — is the 64-bit EXEC/VCC mask zero — not of the current
// lane's own bit. prosper knows this and implements three separate exact-wave reductions for it. The
// CFG dispatcher used none of them for graphics stages and branched on the invocation's own bit.
//
// The sharpest form of the defect is not the absolute claim but the INCONSISTENCY: the SAME fragment
// shader was lowered two different ways depending on which route its CFG happened to take — a wave
// vote when `detect_forward_ifs` structured it, a lane-local bit when it fell through to the
// dispatcher. Two lowerings of one instruction means one of them is wrong independent of any argument
// about which, and that is what these arms pin.
//
// Why it matters beyond tidiness is recorded rather than speculated: a guard-only predicate was
// falsified on DOLL's FXAA (#273), and `RECOMPILER_REMAINING.md` § Ruled out carries four rejected
// attempts at relaxing the fragment wave64 vote. A scalar written inside a branch-guarded block is
// observable by the whole wave on hardware and by only the entering invocations under a lane-local
// branch.
//
// The VERTEX stage is deliberately NOT covered here and still takes the lane bit — see the comment at
// the dispatcher's graphics arm. None of the three existing reductions can serve it: two are
// compute-only (workgroup barriers / a compute-only gate) and the third pins a fragment-stage backend
// contract. That gap is real and is left open on #3573.
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>
#include <cstdio>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static uint32_t count_opcode(const std::vector<uint32_t>& spv, uint32_t opcode) {
    if (spv.size() < 5) return 0;
    uint32_t n = 0;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t wc = spv[i] >> 16u;
        if (wc == 0 || i + wc > spv.size()) break;
        if ((spv[i] & 0xffffu) == opcode) ++n;
        i += wc;
    }
    return n;
}

constexpr uint32_t kOpSwitch = 251;
constexpr uint32_t kOpGroupNonUniformAny = 335;

int main() {
    printf("== test_dispatcher_wave_vote ==\n");

    // A fragment CFG the narrow structurizer rejects, which is what routes a GRAPHICS stage to the
    // dispatcher. The entry condition is `complex_graphics_cfg && cf_rejected`, where
    // complex_graphics_cfg needs MORE THAN TWO branches -- so a simple forward if, or even a single
    // loop, is structured instead and never reaches the arm under test.
    //
    // Shape: an irreducible region. pc1 jumps INTO the middle of the region that pc7's back-edge
    // closes, so the region has two entries and no structurizer can nest it.
    //
    //   pc0  v_cmp_le_u32 vcc, 0, v0        divergent VCC -- lanes disagree
    //   pc1  s_cbranch_vccz  -> pc6         THE INSTRUCTION UNDER TEST (scalar branch on VCC)
    //   pc2  s_cmp_eq_u32 s4, 0             back-edge target
    //   pc3  s_cbranch_scc0  -> pc8
    //   pc4  s_mov_b32 s0, 5
    //   pc5  s_branch        -> pc8
    //   pc6  s_mov_b32 s0, 7                entered from OUTSIDE the back-edge region
    //   pc7  s_branch        -> pc2         back-edge
    //   pc8  export, endpgm
    const uint32_t irreducible_vccz_ps[] = {
        0x7D860080u,              // pc0  v_cmp_le_u32 vcc, 0, v0
        0xBF860004u,              // pc1  s_cbranch_vccz -> pc6
        0xBF068004u,              // pc2  s_cmp_eq_u32 s4, 0
        0xBF840004u,              // pc3  s_cbranch_scc0 -> pc8
        0xBE800385u,              // pc4  s_mov_b32 s0, 5
        0xBF820002u,              // pc5  s_branch -> pc8
        0xBE800387u,              // pc6  s_mov_b32 s0, 7
        0xBF82FFFAu,              // pc7  s_branch -> pc2   (back-edge)
        0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,
        0xF800180Fu, 0x03020100u, 0xBF810000u,
    };

    const std::vector<uint32_t> frag =
        recompile_fragment(irreducible_vccz_ps, std::size(irreducible_vccz_ps));
    CHECK(!frag.empty(), "the irreducible vccz fragment recompiles");
    if (frag.empty()) { printf("== FAIL: %d ==\n", fails + 1); return 1; }

    // THE CONTROL, and it is load-bearing. If this program did not actually take the dispatcher
    // route, the arm below would be asserting something about the structurizer instead -- which
    // already voted, and would pass while proving nothing about the change.
    CHECK(count_opcode(frag, kOpSwitch) >= 1,
          "CONTROL: this program lowered through the CFG DISPATCHER (the module has an OpSwitch), "
          "so the arm below is about the dispatcher and not the structurizer");

    CHECK(count_opcode(frag, kOpGroupNonUniformAny) >= 1,
          "#3573: the dispatcher lowers a FRAGMENT vcc branch as a wave vote "
          "(OpGroupNonUniformAny), not as this invocation's own EXEC bit");

    // The vote is only exact when the subgroup IS the guest wave, so taking it obliges the module to
    // declare the width it needs. Without this the emitted vote could be over a narrower subgroup --
    // a wrong answer with no reject, which is worse than the lane-local form it replaces.
    CHECK(fragment_spirv_required_subgroup_size(frag) == 64u,
          "#3573: ...and the module declares the exact 64-lane subgroup that vote requires, "
          "so the backend enforces it or skips the draw rather than voting over half a wave");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
