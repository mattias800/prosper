// #5039: a word defined before a loop and accumulated inside it stays defined in the lanes the loop
// reads it from, when every back edge keeps EXEC within the loop's preheader EXEC.
//
// The fixture is test_fragment_loop_mask_exec's nested loop: v8 is zeroed under the outer loop's
// EXEC, then incremented and compared inside an inner loop that narrows EXEC through a VOPC mask
// (`s_mov_b64 exec, s[92:93]`, `s_andn2_b64 exec, s[92:93], vcc`) rather than a CMPX, and that
// leaves through an interior VCCZ break. The walk proves the inner header EXEC within the outer
// owner of v8 by induction (rdna2_loop_exec_subset.cpp): assumed at the header, checked at the back
// edge. Each refused twin differs in one place:
//   - the back edge widens EXEC: the inner loop's final S_ANDN2 starts from the outer entry's saved
//     EXEC (s[88:89]) instead of the inner mask, so the induction step fails;
//   - the definition happens on only one preheader path: a scalar branch may skip the v8 zeroing.
// The positive arm carries the same two inserted words with the branch replaced by S_NOP, so every
// arm has the same PCs and branch offsets.
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;

constexpr uint32_t kSkipDefinition = 0xbf850001u;   // s_cbranch_scc1 +1: skips v_mov_b32 v8, 0
constexpr uint32_t kNop = 0xbf800000u;
constexpr uint32_t kInnerAndn2 = 0x8afe6a5cu;   // s_andn2_b64 exec, s[92:93], vcc
constexpr uint32_t kWidenedAndn2 = 0x8afe6a58u;   // s_andn2_b64 exec, s[88:89], vcc

std::vector<uint32_t> program(bool skip_definition = false, bool widen_back_edge = false) {
    return {
        0xbed40380u,
        0xbed50380u,
        0xbed8047eu,   // s84 = s85 = 0; s[88:89] = entry EXEC
        0xd7650007u,
        0x000100c1u,
        0xd7660007u,
        0x00020ec1u,   // v7 = lane id
        0x36120e83u,
        0x4a121281u,
        0x36160e81u,
        0x7e140280u,   // v9 = (v7&3)+1, v11 = v7&1, v10 = 0
        0xbf88001bu,   // outer header: s_cbranch_execz -> exit
        0xbf068080u,   // s_cmp_eq_u32 0, 0
        skip_definition ? kSkipDefinition : kNop,
        0x7e100280u,   // v_mov_b32 v8, 0: the definition made before the inner loop
        0xd4c2005cu,
        0x0002150bu,
        0xbefe045cu,
        0xbed6047eu,   // EXEC = parity lanes
        0xbf88000du,   // inner header: s_cbranch_execz -> inner exit
        0xd4c6005cu,
        0x00010109u,
        0x8ac47e54u,
        0xbeea045cu,
        0x88d4445cu,
        0xbefe045cu,
        0xbf860006u,   // s_cbranch_vccz: interior break
        0x4a101081u,   // v_add_nc_u32 v8, 1, v8: accumulated inside the loop
        0x7d8c1308u,   // v_cmp_ge_u32 vcc, v8, v9: the fresh predicate reading it
        widen_back_edge ? kWidenedAndn2 : kInnerAndn2,
        0x8aea5c54u,
        0x88d46a5cu,
        0xbf89fff2u,   // inner back edge: s_cbranch_execnz -> inner header
        0xbefe0458u,
        0x4a141481u,
        0xd4c1006au,
        0x0001050au,
        0xbefe046au,
        0xbf89ffe4u,   // outer back edge: s_cbranch_execnz -> outer header
        0xbefe0458u,
        0x87ea5854u,
        0xd5010000u,
        0x01a9e480u,
        0x7e0202f2u,
        0x7e040280u,
        0x7e0602f2u,
        0xf800080fu,
        0x03020100u,
        0xbf810000u,
    };
}

struct Result {
    std::vector<uint32_t> module;
    std::vector<std::pair<std::string, std::string>> rejects;
};

Result compile(const std::vector<uint32_t>& code, uint64_t address) {
    TerminalRejectCapture capture;
    auto module = recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX,
                                     nullptr, false, {RecompileDiagnosticStage::Fragment, address});
    return {std::move(module), capture.take()};
}

// Match the reason and, when supplied, the instruction detail in the same terminal refusal.
bool rejected_with(const Result& result, const std::string& reason,
                   const std::string& detail = {}) {
    return std::any_of(result.rejects.begin(), result.rejects.end(), [&](const auto& reject) {
        return (reject.first == reason || reject.second.find(reason) != std::string::npos) &&
               reject.second.find(detail) != std::string::npos;
    });
}

}   // namespace

TEST(LoopExecSubset, ADefinitionBeforeTheLoopSurvivesItsAccumulation) {
    const auto result = compile(program(), 0x50390001u);
    EXPECT_FALSE(result.module.empty()) << testing::PrintToString(result.rejects);
}

TEST(LoopExecSubset, ABackEdgeThatWidensExecIsRefused) {
    const auto result = compile(program(false, true), 0x50390002u);
    EXPECT_TRUE(result.module.empty());
    EXPECT_TRUE(rejected_with(result, "loop-exec-subset-unproved"))
        << testing::PrintToString(result.rejects);
}

TEST(LoopExecSubset, ADefinitionOnOnlyOnePreheaderPathIsRefused) {
    const auto result = compile(program(true, false), 0x50390003u);
    EXPECT_TRUE(result.module.empty());
    EXPECT_TRUE(rejected_with(result, "fabricated-fresh-mask-source", "pc=28 "))
        << testing::PrintToString(result.rejects);
}
