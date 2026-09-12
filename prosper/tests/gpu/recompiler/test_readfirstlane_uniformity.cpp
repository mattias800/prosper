// test_readfirstlane_uniformity — a value prosper models as LANE-LOCAL must not satisfy the
// wave-uniformity proof that lets a fragment scalar branch skip its wave vote.
//
// Why this exists (#3596). `s_cbranch_vccz` is a SCALAR branch: the condition is whether the whole
// wave's VCC mask is zero. prosper votes for it with `OpGroupNonUniformAny` unless it can prove VCC is
// identical in every lane, in which case `any(P)` is `P` and the vote is pure cost.
//
// That proof asked "are both compare operands scalar-register sourced?" — and `v_readfirstlane_b32`
// writes an SGPR while producing THIS lane's value. prosper says so in place: its lowering is annotated
// SPECULATIVE(confidence: med) precisely because the per-lane scalar model has no cross-lane reduction.
// So a VCC derived from a readfirstlane result passed the proof, the vote was skipped, and the branch
// took the invocation's own bit — wrong pixels, no reject, no diagnostic. That is the exact defect the
// wave-vote work exists to prevent, reappearing through the escape meant to optimise it.
//
// The fix taints the SSA VALUE rather than the register, so it survives scalar copies (`s_mov_b32`
// lowers as `d = a`) and clears itself on overwrite.
//
// WHAT THIS TEST DOES NOT CLAIM. It does not establish that the proof is now sound in general — see
// the residual recorded in #3596. It establishes that the one demonstrated counterexample no longer
// passes, and it is built as a PAIR so it cannot pass by the escape simply never firing.
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
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
    printf("== test_readfirstlane_uniformity ==\n");

    // An irreducible fragment CFG -- pc1 branches INTO the region the back-edge closes, giving it two
    // entries -- so the narrow structurizer rejects and the program routes to the CFG dispatcher, which
    // is where the escape lives. Four branches, because the graphics dispatcher entry requires more
    // than two.
    //
    // EXEC is restored before the compare (`s_mov_b64 exec, -1`), because the live uniformity marker is
    // only set for a compare that ran with EXEC not narrowed, and a dispatch case always ENTERS
    // narrowed. Without that instruction neither variant below would fire the escape and the pair would
    // compare nothing.
    //
    // The two programs differ in ONE instruction: how `s0` is produced.
    auto build = [](uint32_t producer_lo, uint32_t producer_hi, bool two_dword) {
        std::vector<uint32_t> k;
        k.push_back(producer_lo);                       // pc0  s0 = <producer>
        if (two_dword) k.push_back(producer_hi);
        k.push_back(0xBE810380u);                       //      s_mov_b32 s1, 0
        k.push_back(0xBEFE04C1u);                       //      s_mov_b64 exec, -1
        k.push_back(0xD4C2006Au); k.push_back(0x00000200u);  // v_cmp_eq_u32_e64 vcc, s0, s1
        k.push_back(0xBF860004u);                       //      s_cbranch_vccz -> +4
        k.push_back(0xBF068004u);                       //      s_cmp_eq_u32 s4, 0
        k.push_back(0xBF840004u);                       //      s_cbranch_scc0 -> +4
        k.push_back(0xBE800385u);                       //      s_mov_b32 s0, 5
        k.push_back(0xBF820002u);                       //      s_branch -> +2
        k.push_back(0xBE800387u);                       //      s_mov_b32 s0, 7
        k.push_back(0xBF82FFFAu);                       //      s_branch back-edge
        k.push_back(0x7E0202F2u); k.push_back(0x7E040280u); k.push_back(0x7E0602F2u);
        k.push_back(0xF800180Fu); k.push_back(0x03020100u); k.push_back(0xBF810000u);
        return k;
    };

    // CONTROL: `s0` is an immediate. Genuinely wave-uniform, so the escape SHOULD fire.
    const std::vector<uint32_t> imm = build(0xBE800380u, 0u, false);   // s_mov_b32 s0, 0
    // THE CASE: `s0` comes from v_readfirstlane_b32 s0, v0 -- lane-local under prosper's model.
    const std::vector<uint32_t> rfl = build(0x7E000501u, 0u, false);   // v_readfirstlane_b32 s0, v0

    const std::vector<uint32_t> spv_imm = recompile_fragment(imm.data(), imm.size());
    const std::vector<uint32_t> spv_rfl = recompile_fragment(rfl.data(), rfl.size());

    CHECK(!spv_imm.empty() && !spv_rfl.empty(), "both variants recompile");
    if (spv_imm.empty() || spv_rfl.empty()) { printf("== FAIL: %d ==\n", fails + 1); return 1; }

    // CONTROLS FIRST. If either program failed to reach the dispatcher, the comparison below would be
    // about routing rather than about the uniformity proof.
    CHECK(count_opcode(spv_imm, kOpSwitch) >= 1 && count_opcode(spv_rfl, kOpSwitch) >= 1,
          "CONTROL: both variants lower through the CFG dispatcher");
    CHECK(count_opcode(spv_imm, kOpGroupNonUniformAny) == 0,
          "CONTROL: the immediate-sourced compare IS wave-uniform, so the escape fires and no vote "
          "is emitted -- without this the arm below could pass on a build that never escapes");

    CHECK(count_opcode(spv_rfl, kOpGroupNonUniformAny) >= 1,
          "#3596: a VCC derived from v_readfirstlane_b32 does NOT satisfy the proof, so the wave vote "
          "is emitted rather than the branch taking this invocation's own bit");
    CHECK(fragment_spirv_required_subgroup_size(spv_rfl) == 64u,
          "#3596: ...and the module declares the exact subgroup that vote needs");

    // The pair is the discriminator: the two programs differ in exactly one instruction, and they
    // disagree about the vote. Either arm alone could be satisfied by a build that always votes or one
    // that never does.
    CHECK(count_opcode(spv_rfl, kOpGroupNonUniformAny) >
              count_opcode(spv_imm, kOpGroupNonUniformAny),
          "#3596: the two differ ONLY in how s0 is produced, and they disagree about the vote -- so "
          "this pair tests the uniformity proof rather than the routing");

    // The taint must survive a scalar COPY, because it is keyed on the SSA value and `s_mov_b32`
    // lowers as `d = a`. Same program with the readfirstlane result laundered through s2.
    {
        std::vector<uint32_t> via_copy;
        via_copy.push_back(0x7E040501u);            // v_readfirstlane_b32 s2, v0
        via_copy.push_back(0xBE800302u);            // s_mov_b32 s0, s2      <- launder
        const std::vector<uint32_t> tail = build(0xBF800000u, 0u, false);  // s_nop as the producer slot
        via_copy.insert(via_copy.end(), tail.begin() + 1, tail.end());
        const std::vector<uint32_t> spv_copy = recompile_fragment(via_copy.data(), via_copy.size());
        CHECK(!spv_copy.empty() && count_opcode(spv_copy, kOpSwitch) >= 1,
              "CONTROL: the laundered variant also recompiles and reaches the dispatcher");
        CHECK(count_opcode(spv_copy, kOpGroupNonUniformAny) >= 1,
              "#3596: laundering the lane-local value through an s_mov_b32 does not clean it -- the "
              "taint is on the SSA value, and a scalar move copies that value");
    }

    // Two controls for the laundering arm, because "it voted" is not by itself evidence that the
    // READFIRSTLANE caused the vote -- a longer program, or an extra scalar move, could in principle
    // do it. Both of these are the laundering shape MINUS the thing under test, and both must escape.
    {
        // (a) the same two-instruction launder with a CLEAN source: no readfirstlane anywhere.
        std::vector<uint32_t> clean_launder;
        clean_launder.push_back(0xBE820380u);       // s_mov_b32 s2, 0
        clean_launder.push_back(0xBE800302u);       // s_mov_b32 s0, s2     <- same launder shape
        const std::vector<uint32_t> tail_a = build(0xBF800000u, 0u, false);
        clean_launder.insert(clean_launder.end(), tail_a.begin() + 1, tail_a.end());
        const std::vector<uint32_t> spv_a =
            recompile_fragment(clean_launder.data(), clean_launder.size());
        CHECK(!spv_a.empty() && count_opcode(spv_a, kOpSwitch) >= 1,
              "CONTROL: the clean two-move launder recompiles and reaches the dispatcher");
        CHECK(count_opcode(spv_a, kOpGroupNonUniformAny) == 0,
              "CONTROL: laundering a CLEAN scalar through the same two moves still escapes -- so the "
              "vote in the arm above is caused by the readfirstlane, not by the launder shape");

        // (b) readfirstlane PRESENT but not feeding the compare. Its result goes to s2 and is never
        //     read; the compare still sources the immediate-defined s0.
        std::vector<uint32_t> unused_rfl;
        unused_rfl.push_back(0x7E040501u);          // v_readfirstlane_b32 s2, v0   (result unused)
        unused_rfl.push_back(0xBE800380u);          // s_mov_b32 s0, 0
        const std::vector<uint32_t> tail_b = build(0xBF800000u, 0u, false);
        unused_rfl.insert(unused_rfl.end(), tail_b.begin() + 1, tail_b.end());
        const std::vector<uint32_t> spv_b =
            recompile_fragment(unused_rfl.data(), unused_rfl.size());
        CHECK(!spv_b.empty() && count_opcode(spv_b, kOpSwitch) >= 1,
              "CONTROL: the unused-readfirstlane variant recompiles and reaches the dispatcher");
        CHECK(count_opcode(spv_b, kOpGroupNonUniformAny) == 0,
              "CONTROL: a readfirstlane whose result never reaches the compare does NOT suppress the "
              "escape -- the taint follows the VALUE into the compare, it is not a program-wide flag");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
