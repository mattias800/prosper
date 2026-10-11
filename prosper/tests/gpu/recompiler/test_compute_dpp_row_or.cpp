// #4891: refused OR row reductions leave compute producer masks unwritten.
// Execute the production portable dispatcher with an independent instruction-order CPU oracle;
// native modules have separate compilation/structure checks, not simulated subgroup execution.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/dpp_row_or.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t kLanes = 256;
using Words = std::array<uint32_t, kLanes>;
using prosper::test::DppRowCfgCase;

uint32_t count(const std::vector<uint32_t>& module, uint32_t opcode) {
    uint32_t result = 0;
    for (size_t at = 5; at < module.size();) {
        const uint32_t size = module[at] >> 16u;
        if (!size || size > module.size() - at) return 0;
        result += (module[at] & 0xffffu) == opcode;
        at += size;
    }
    return result;
}

Words values() {
    Words result{};
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        result[lane] = (1u << (lane % 16u)) | ((lane / 16u + 1u) << 24u);
    return result;
}

void row(Words& words, uint32_t wave, uint32_t shift, bool masked) {
    const auto previous = words;
    for (uint32_t lane = wave * 64u; lane < (wave + 1u) * 64u; ++lane) {
        if (lane % 16u < shift || (masked && (lane % 5u == 0 || (lane - shift) % 5u == 0)))
            continue;
        words[lane] = previous[lane] | previous[lane - shift];
    }
}

Words expected(std::initializer_list<uint32_t> shifts, bool masked = false) {
    auto words = values();
    for (uint32_t shift : shifts)
        for (uint32_t wave = 0; wave < 4; ++wave) row(words, wave, shift, masked);
    return words;
}

Words expected_cfg(DppRowCfgCase shape) {
    auto words = values();
    constexpr uint32_t trips[] = {0, 3, 1, 4};
    for (uint32_t wave = 0; wave < 4; ++wave) {
        if (shape == DppRowCfgCase::Mixed) {
            for (uint32_t shift : {1u, 2u, 4u, 8u}) row(words, wave, shift, false);
        } else if (shape == DppRowCfgCase::LoopAndCompletedPeer) {
            for (uint32_t iteration = 0; iteration < trips[wave]; ++iteration) {
                row(words, wave, 1u, true);
                row(words, wave, 2u, true);
            }
        } else if (wave % 2u == 0) {
            row(words, wave, 1u, false);
            row(words, wave, 2u, false);
        } else {
            row(words, wave, 1u, false);
            row(words, wave, 4u, false);
            row(words, wave, 8u, false);
        }
    }
    return words;
}

std::vector<uint32_t> compile_dispatcher(const std::vector<uint32_t>& words, uint32_t native = 0,
                                         bool phase = false) {
    SpirvCompute b;
    static uint64_t next_address = 0x48910000u;
    b.diagnostic = {RecompileDiagnosticStage::Compute, next_address++};
    b.native_subgroup_size = native;
    b.begin(10u, nullptr, kLanes, 1u, 1u, 64u, 0u, true, true);
    RegState state;
    state.exec = b.btrue();
    state.vcc = b.bfalse();
    state.scc = b.bfalse();
    state.max_vgpr = 10;
    const uint32_t wave = b.ibin(Op_ShiftRightLogical, b.linear_localid, b.uconst(6u));
    state.vreg[0] = b.ibin(Op_BitwiseAnd, wave, b.uconst(1u));
    state.vreg[1] =
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, b.uconst(1u),
                      b.ibin(Op_BitwiseAnd, b.linear_localid, b.uconst(15u))),
               b.ibin(Op_ShiftLeftLogical,
                      b.ibin(Op_IAdd, b.ibin(Op_ShiftRightLogical, b.linear_localid, b.uconst(4u)),
                             b.uconst(1u)),
                      b.uconst(24u)));
    state.vreg[2] = b.uconst(1u);
    state.vreg[3] =
        b.sel(b.ucmp(Op_IEqual, wave, b.uconst(0u)), b.uconst(0u),
              b.sel(b.ucmp(Op_IEqual, wave, b.uconst(1u)), b.uconst(3u),
                    b.sel(b.ucmp(Op_IEqual, wave, b.uconst(2u)), b.uconst(1u), b.uconst(4u))));
    state.vreg[4] =
        b.sel(b.ucmp(Op_IEqual, b.ibin(Op_UMod, b.linear_localid, b.uconst(5u)), b.uconst(0u)),
              b.uconst(0u), b.uconst(1u));
    state.vreg[5] = b.uconst(1u);
    std::vector<Rdna2Inst> ins;
    EXPECT_EQ(rdna2_walk(words.data(), words.size(), ins), words.size());
    const auto no_export = [](RegState&, const Rdna2Inst&) { return false; };
    const bool ok = phase ? emit_body(b, state, ins, {}, nullptr, true, false, no_export,
                                      words.data(), words.size(), nullptr, true, 0, true)
                          : emit_cfg_state_machine(b, state, ins, {}, nullptr, true, false,
                                                   no_export, words.data(), words.size());
    if (!ok) return {};
    const auto result = state.vreg.find(1);
    if (result == state.vreg.end()) return {};
    b.store_output_word(result->second, 1u, 0u);
    return b.finish();
}

void execute(const std::vector<uint32_t>& words, const Words& wanted, bool phase = false) {
    const auto module = compile_dispatcher(words, 0u, phase);
    ASSERT_FALSE(module.empty());
    EXPECT_GT(count(module, Op_Switch), 0u);
    EXPECT_GE(count(module, Op_ControlBarrier), 2u);
    EXPECT_EQ(count(module, Op_GroupNonUniformShuffle), 0u)
        << "portable workers must consume the tagged common scratch phase";
    bpermute_oracle::Interpreter vm(module);
    const auto actual =
        vm.run_buffers(kLanes, {{1u, std::vector<uint32_t>(kLanes, 0xdeadbeefu)}}, 1u);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(actual, std::vector<uint32_t>(wanted.begin(), wanted.end()));
}

bool ordinary(const Rdna2Inst& in, bool visible = true, bool spill = false) {
    SpirvCompute b;
    b.begin(3u);
    RegState state;
    state.exec = b.btrue();
    state.vreg[1] = b.uconst(0x80000001u);
    if (spill) {
        state.vgpr_lane_slots[1][0] = b.uconst(7u);
        state.vgpr_lane_mask_slots[1][0] = b.btrue();
    }
    bool ok = true;
    EXPECT_TRUE(emit_alu(b, state, in, ok, true, nullptr, false, nullptr, visible));
    return ok;
}

TEST(ComputeDppRowOr, DecoderRetainsTheOwnedContractAndMaximumControl) {
    const auto words = prosper::test::dpp_row_or_program({1u});
    const auto in = rdna2_decode_one(words.data(), words.size());
    EXPECT_EQ(in.opcode, 0x1cu);
    EXPECT_TRUE(in.has_dpp);
    EXPECT_FALSE(in.has_modifier);
    EXPECT_FALSE(in.dpp_bound_ctrl);
    EXPECT_EQ(in.dpp_ctrl, 0x111u);
    EXPECT_EQ(in.dpp_row_mask, 15u);
    EXPECT_EQ(in.dpp_bank_mask, 15u);
    EXPECT_EQ(in.dst.value, 1);
    EXPECT_EQ(in.src[0].value, 1);
    EXPECT_EQ(in.src[1].value, 1);
    const auto max = prosper::test::dpp_row_max_program({1u});
    EXPECT_TRUE(ordinary(rdna2_decode_one(max.data(), max.size())));
}

TEST(ComputeDppRowOr, OrdinaryComputeAdmitsAllStridesAndBothWaveWidths) {
    for (uint32_t shift = 1; shift <= 15; ++shift) {
        auto words = prosper::test::dpp_row_or_program({shift});
        words.insert(words.begin(), 0x7e020300u);   // genuine production local-id input
        for (uint32_t wave : {32u, 64u}) {
            for (uint32_t native : {0u, wave}) {
                ComputeShaderConfig config;
                config.wave_size = wave;
                config.native_subgroup_size = native;
                config.local_x = 128u;
                SCOPED_TRACE(testing::Message() << shift << '/' << wave << '/' << native);
                EXPECT_FALSE(
                    recompile_compute(words.data(), words.size(), nullptr, config).empty());
            }
        }
    }
}

TEST(ComputeDppRowOr, OrdinaryEmissionRequiresAWaveVisiblePoint) {
    const auto words = prosper::test::dpp_row_or_program({1u});
    const auto in = rdna2_decode_one(words.data(), words.size());
    EXPECT_FALSE(ordinary(in, false));
    EXPECT_TRUE(ordinary(in));
}

TEST(ComputeDppRowOr, UnownedMasksModifiersAndRegisterShapesRemainClosed) {
    const auto words = prosper::test::dpp_row_or_program({1u});
    const auto in = rdna2_decode_one(words.data(), words.size());
    for (uint32_t mutation = 0; mutation < 11u; ++mutation) {
        auto other = in;
        switch (mutation) {
            case 0: other.dst.value = 2; break;
            case 1: other.src[0].value = 2; break;
            case 2: other.src[1].value = 2; break;
            case 3: other.dpp_bound_ctrl = true; break;
            case 4: other.dpp_row_mask = 7; break;
            case 5: other.dpp_bank_mask = 7; break;
            case 6: other.has_modifier = true; break;
            case 7: other.has_sdwa = true; break;
            case 8: other.src_neg[0] = true; break;
            case 9: other.clamp = true; break;
            case 10: other.omod = 1; break;
        }
        EXPECT_FALSE(ordinary(other)) << mutation;
    }
    auto inactive_fetch = words;
    inactive_fetch[1] |= 1u << 18u;
    EXPECT_FALSE(ordinary(rdna2_decode_one(inactive_fetch.data(), inactive_fetch.size())));
}

TEST(ComputeDppRowOr, PortableValuesKeepEveryStrideInsideItsRow) {
    for (uint32_t shift = 1; shift <= 15; ++shift) {
        SCOPED_TRACE(shift);
        execute(prosper::test::dpp_row_or_program({shift}), expected({shift}));
    }
}

TEST(ComputeDppRowOr, PortableLadderUsesInstructionOrderSnapshots) {
    execute(prosper::test::dpp_row_or_program({1u, 2u, 4u, 8u}), expected({1u, 2u, 4u, 8u}));
}

TEST(ComputeDppRowOr, PortableInactiveSourcesAndDestinationsPreserveValues) {
    // Select EXEC from independent v4/v5, leaving input values in v1 unchanged.
    auto words = prosper::test::dpp_row_or_program({1u, 2u, 4u, 8u});
    words.insert(words.begin(), 0x7da40b04u);
    words.insert(words.end() - 1, 0xbefe04c1u);
    execute(words, expected({1u, 2u, 4u, 8u}, true));
}

TEST(ComputeDppRowOr, PortableDistinctStaticSitesKeepTheirOwnWaves) {
    const auto shape = DppRowCfgCase::DivergentSites;
    execute(prosper::test::dpp_row_or(prosper::test::dpp_row_cfg_program(shape)),
            expected_cfg(shape));
}

TEST(ComputeDppRowOr, PortableLoopsKeepCompletedPeersAndIndependentExecSafe) {
    const auto shape = DppRowCfgCase::LoopAndCompletedPeer;
    execute(prosper::test::dpp_row_or(prosper::test::dpp_row_cfg_program(shape)),
            expected_cfg(shape));
}

TEST(ComputeDppRowOr, LaterBarrierPhaseRetainsItsScratchPlanes) {
    const auto shape = DppRowCfgCase::LaterBarrierPhase;
    execute(prosper::test::dpp_row_or(prosper::test::dpp_row_cfg_program(shape)),
            expected_cfg(shape), true);
}

TEST(ComputeDppRowOr, NativeAndPortableDispatchersKeepTheirParticipationRoutes) {
    const auto words = prosper::test::dpp_row_or(
        prosper::test::dpp_row_cfg_program(DppRowCfgCase::DivergentSites));
    for (uint32_t native : {0u, 64u}) {
        const auto module = compile_dispatcher(words, native);
        ASSERT_FALSE(module.empty()) << native;
        EXPECT_GT(count(module, Op_Switch), 0u);
        EXPECT_EQ(count(module, Op_GroupNonUniformShuffle) != 0, native != 0);
    }
}

TEST(ComputeDppRowOr, ScalarSpillCannotEnterTheDispatcherRowPhase) {
    for (uint32_t native : {0u, 64u}) {
        for (bool mask : {false, true}) {
            for (bool restored : {false, true}) {
                for (bool spill_source : {false, true}) {
                    std::vector<uint32_t> words{0xbf820000u, 0xd7610000u | (spill_source ? 1u : 8u),
                                                mask ? 0x0001007eu : 0x00010081u};
                    if (restored) words.push_back(0xbf820000u);
                    const auto row_words = prosper::test::dpp_row_or_program({1u});
                    words.insert(words.end(), row_words.begin(), row_words.end());
                    SCOPED_TRACE(testing::Message() << native << '/' << mask << '/' << restored
                                                    << '/' << spill_source);
                    EXPECT_EQ(compile_dispatcher(words, native).empty(), spill_source);
                }
            }
        }
    }
}

TEST(ComputeDppRowOr, ScalarSpillCannotReplaceAnOrdinarySourceWord) {
    const auto words = prosper::test::dpp_row_or_program({1u});
    const auto in = rdna2_decode_one(words.data(), words.size());
    EXPECT_FALSE(ordinary(in, true, true));
    EXPECT_TRUE(ordinary(in));
}

TEST(ComputeDppRowOr, IndependentValuesDistinguishAddAndCrossRowFetch) {
    const auto input = values();
    const auto wanted = expected({1u});
    EXPECT_NE(wanted[1], input[1] + input[0]);
    EXPECT_EQ(wanted[16], input[16]);
    EXPECT_NE(wanted[16], input[16] | input[15]);
}
}  // namespace
