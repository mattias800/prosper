// #4891: refused OR row reductions leave compute producer masks unwritten.
// Execute the production portable dispatcher with an independent instruction-order CPU oracle;
// native modules have separate compilation/structure checks, not simulated subgroup execution.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/dpp_row_max.hpp"
#include "fixtures/dpp_row_or.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/resources/shader_resources.hpp"
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

std::vector<uint32_t> ngg_row_program(uint32_t opcode, bool barrier) {
    auto words = prosper::test::dpp_row_max_program({1u, 2u, 4u, 8u});
    for (size_t at = 0; at + 1u < words.size(); at += 2u)
        words[at] = (words[at] & 0x81ffffffu) | (opcode << 25u);
    words.insert(words.begin(), 0x7e020300u);   // v_mov_b32 v1,v0: genuine launch input
    if (barrier) words.insert(words.begin() + 1, 0xbf8a0000u);
    words.insert(words.end() - 1, {0xf8000941u, 0x00000001u});   // EXP PRIM,v1
    return words;
}

TEST(ComputeDppRowOr, PortableNggIntegerRowsUseWorkgroupPhasesWithoutScalarBranches) {
    for (uint32_t opcode : {0x25u, 0x14u, 0x1cu}) {
        for (bool barrier : {false, true}) {
            for (bool native : {false, true}) {
                SCOPED_TRACE(testing::Message() << opcode << '/' << barrier << '/' << native);
                const auto words = ngg_row_program(opcode, barrier);
                const auto module = recompile_ngg_exports_for_test(
                    words.data(), words.size(), 10, 0, nullptr, 4, 0, {}, true, true, native);
                ASSERT_FALSE(module.empty());
                EXPECT_EQ(count(module, Op_GroupNonUniformShuffle) != 0, native);
                if (!native) {
                    EXPECT_GT(count(module, Op_Switch), 0u);
                    EXPECT_GE(count(module, Op_ControlBarrier), 2u);
                }
            }
        }
    }
}

TEST(ComputeDppRowOr, NggIntegerRowSourceSpillsStayRefusedInBothProfiles) {
    for (uint32_t opcode : {0x25u, 0x14u, 0x1cu}) {
        for (bool barrier : {false, true}) {
            for (bool native : {false, true}) {
                for (bool spill_source : {false, true}) {
                    auto words = ngg_row_program(opcode, barrier);
                    words[words.size() - 2u] =
                        2u;   // independent EXP sink cannot hide a row refusal
                    words.insert(words.begin() + 1,
                                 {0xd7610000u | (spill_source ? 1u : 8u), 0x00010081u});
                    SCOPED_TRACE(testing::Message() << opcode << '/' << barrier << '/' << native
                                                    << '/' << spill_source);
                    const uint64_t address = 0x49590000u + (opcode << 8u) + (barrier << 2u) +
                                             (native << 1u) + spill_source;
                    EXPECT_EQ(recompile_ngg_exports_for_test(
                                  words.data(), words.size(), 10, 0, nullptr, 4, 0,
                                  {RecompileDiagnosticStage::Compute, address}, true, true, native)
                                  .empty(),
                              spill_source);
                    if (spill_source) {
                        const auto reason = last_terminal_reject_reason(address);
                        EXPECT_NE(reason.find("pc=" + std::to_string(barrier ? 4u : 3u)),
                                  std::string::npos)
                            << reason;
                    }
                }
            }
        }
    }
}

TEST(ComputeDppRowOr, NggIntegerRowsAcceptCurrentOrdinaryOverwritesOfHistoricalSpills) {
    const std::vector<std::vector<uint32_t>> replacements{
        {0xd7610001u, 0x00010081u, 0x7e020283u}, // full ordinary overwrite after a spill
        {0xbefe04c1u, 0xd7610001u, 0x00010081u, 0xbf068181u, 0xbf850002u, 0x7e020300u, 0xbf820001u,
         0x7e020303u}, // both forward arms overwrite from genuine VGPRs
        {0xd7610001u, 0x00010081u, 0x7e020300u}, // full overwrite after the guest barrier
    };
    for (uint32_t opcode : {0x25u, 0x14u, 0x1cu}) {
        for (size_t shape = 0; shape < replacements.size(); ++shape) {
            for (bool barrier : {false, true}) {
                for (bool native : {false, true}) {
                    auto words = ngg_row_program(opcode, barrier);
                    // The positive consumes the row result, including any later alias reload.
                    const size_t at = shape == 1u && barrier ? 2u : 1u;
                    auto replacement = replacements[shape];
                    if (shape == 2u && barrier) {
                        words.erase(words.begin() + 1);
                        replacement.insert(replacement.begin() + 2, 0xbf8a0000u);
                    }
                    words.insert(words.begin() + at, replacement.begin(), replacement.end());
                    SCOPED_TRACE(testing::Message()
                                 << opcode << '/' << shape << '/' << barrier << '/' << native);
                    const uint64_t address =
                        0x4959a000u + (opcode << 8u) + (shape << 4u) + (barrier << 1u) + native;
                    EXPECT_FALSE(recompile_ngg_exports_for_test(
                                     words.data(), words.size(), 10, 0, nullptr, 4, 0,
                                     {RecompileDiagnosticStage::Compute, address}, true, true,
                                     native)
                                     .empty())
                        << last_terminal_reject_reason(address);
                }
            }
        }
    }
}

TEST(ComputeDppRowOr, NggIntegerRowsAcceptCompleteExplicitIntegerProducers) {
    for (uint32_t producer : {0x0bu, 0x11u, 0x12u, 0x13u, 0x14u, 0x16u, 0x18u, 0x1au, 0x1bu, 0x1cu,
                              0x1du, 0x1eu, 0x25u, 0x26u, 0x27u}) {
        for (bool barrier : {false, true}) {
            for (bool native : {false, true}) {
                auto words = ngg_row_program(0x1cu, false);
                std::vector<uint32_t> replacement{0xd7610001u, 0x00010081u};
                if (barrier) replacement.push_back(0xbf8a0000u);
                replacement.push_back((producer << 25u) | 0x00020100u);   // v1 = op(v0,v0)
                words.insert(words.begin() + 1, replacement.begin(), replacement.end());
                SCOPED_TRACE(testing::Message() << producer << '/' << barrier << '/' << native);
                EXPECT_FALSE(recompile_ngg_exports_for_test(words.data(), words.size(), 10, 0,
                                                            nullptr, 4, 0, {}, true, true, native)
                                 .empty());
            }
        }
    }
}

TEST(ComputeDppRowOr, NggRowsAcceptWholeWordUnaryFloatAndExtendedProducers) {
    for (const std::vector<uint32_t>& producer : {
             std::vector<uint32_t>{0x7e026f00u},   // NOT_B32
             std::vector<uint32_t>{0x7e027100u},   // BFREV_B32
             std::vector<uint32_t>{0x06020100u},   // F32 ADD
             std::vector<uint32_t>{0xd5030001u, 0x00020100u},   // VOP3 F32 ADD
             std::vector<uint32_t>{0xd5690001u, 0x00020100u},   // VOP3 MUL_LO_U32
             std::vector<uint32_t>{0xd5780001u, 0x02020100u},   // VOP3 XOR3 with explicit zero
             std::vector<uint32_t>{0xd7110001u, 0x00020100u},   // PACK_B32_F16 writes both halves
             std::vector<uint32_t>{0xd8d480e4u, 0x01000000u},   // identity DS swizzle from v0
             std::vector<uint32_t>{0xd7610003u, 0x00010081u, 0x7e060300u, 0xd8d480e4u, 0x01000003u},
             std::vector<uint32_t>{0xd7610000u, 0x00010081u, 0xd8340000u, 0x00000605u, 0xd8b40000u,
                                   0x01000605u},   // unused DATA1=0 is not a v0 read
         }) {
        for (bool barrier : {false, true}) {
            for (bool native : {false, true}) {
                auto words = ngg_row_program(0x1cu, false);
                std::vector<uint32_t> replacement{0xd7610001u, 0x00010081u};
                if (barrier) replacement.push_back(0xbf8a0000u);
                replacement.insert(replacement.end(), producer.begin(), producer.end());
                words.insert(words.begin() + 1, replacement.begin(), replacement.end());
                SCOPED_TRACE(testing::Message() << producer[0] << '/' << barrier << '/' << native);
                EXPECT_FALSE(recompile_ngg_exports_for_test(words.data(), words.size(), 10, 0,
                                                            nullptr, 4, 0, {}, true, true, native)
                                 .empty());
            }
        }
    }
}

TEST(ComputeDppRowOr, NggImageRowsRequireEveryConsecutiveAndNsaAddressWord) {
    ShaderResourceTable resources;
    ShaderResource texture{};
    texture.cls = ResourceClass::Texture;
    texture.binding = 2;
    texture.img_dim = 1;
    texture.width = texture.height = 4;
    texture.format = DataFormat::Float32;
    texture.num_components = 1;
    texture.sgpr_base = 8;
    resources.resources.push_back(texture);
    for (bool nsa : {false, true}) {
        for (bool barrier : {false, true}) {
            for (uint32_t shape = 0; shape < 4; ++shape) {
                auto words = ngg_row_program(0x25u, false);
                const bool valid = shape != 1u;
                if (!valid) words[words.size() - 2u] = 2u;   // independent genuine sink
                std::vector<uint32_t> prefix;
                if (shape == 1u || shape == 2u)
                    prefix.insert(prefix.end(), {0xd7610006u, 0x00010081u});
                if (shape == 3u) prefix.insert(prefix.end(), {0xd7610008u, 0x00010081u});
                if (barrier) prefix.push_back(0xbf8a0000u);
                if (shape == 2u) prefix.push_back(0x7e0c0280u);   // full genuine coordinate zero
                prefix.insert(prefix.end(), {nsa ? 0xf000010au : 0xf0000108u, 0x00020105u});
                if (nsa) prefix.push_back(6u);   // [v5,v6], with v6 in the extra NSA byte
                words.insert(words.begin() + 1, prefix.begin(), prefix.end());
                const uint64_t address = 0x4959e000u + (nsa << 5u) + (barrier << 4u) + shape;
                SCOPED_TRACE(testing::Message() << nsa << '/' << barrier << '/' << shape);
                EXPECT_EQ(recompile_ngg_exports_for_test(
                              words.data(), words.size(), 10, 0, &resources, 4, 0,
                              {RecompileDiagnosticStage::Compute, address}, true, true, false)
                              .empty(),
                          !valid);
                if (!valid) {
                    const auto reason = last_terminal_reject_reason(address);
                    EXPECT_NE(reason.find("pc=" + std::to_string(1u + prefix.size())),
                              std::string::npos)
                        << reason;
                }
            }
        }
    }
}

TEST(ComputeDppRowOr, NggPackedStorageImageRowsCannotHideSecondCoordinateSpills) {
    ShaderResourceTable resources;
    ShaderResource image{};
    image.cls = ResourceClass::StorageImage;
    image.binding = 2;
    image.img_dim = 1;
    image.width = image.height = 4;
    image.format = DataFormat::Uint32;
    image.num_components = 1;
    image.sgpr_base = 8;
    resources.resources.push_back(image);
    for (bool barrier : {false, true}) {
        for (uint32_t shape = 0; shape < 4; ++shape) {
            auto words = ngg_row_program(0x25u, false);
            const bool valid = shape >= 2u;
            if (!valid) words[words.size() - 2u] = 2u;   // independent genuine sink
            std::vector<uint32_t> prefix{shape == 3u ? 0xd7610008u : 0xd7610006u, 0x00010081u};
            if (barrier) prefix.push_back(0xbf8a0000u);
            if (shape == 2u) prefix.push_back(0x7e0c0280u);   // full genuine coordinate zero
            prefix.insert(prefix.end(), {0xf0000108u, shape == 0u ? 0x40020105u : 0x00020105u});
            words.insert(words.begin() + 1, prefix.begin(), prefix.end());
            const uint64_t address = 0x4959f000u + (barrier << 4u) + shape;
            SCOPED_TRACE(testing::Message() << barrier << '/' << shape);
            EXPECT_EQ(recompile_ngg_exports_for_test(
                          words.data(), words.size(), 10, 0, &resources, 4, 0,
                          {RecompileDiagnosticStage::Compute, address}, true, true, false)
                          .empty(),
                      !valid);
            if (!valid) {
                const auto reason = last_terminal_reject_reason(address);
                EXPECT_NE(reason.find("pc=" + std::to_string(1u + prefix.size())),
                          std::string::npos)
                    << reason;
            }
        }
    }
}

TEST(ComputeDppRowOr, NggRowsRequireCurrentOrdinarySourceDependencies) {
    const std::vector<std::vector<uint32_t>> replacements{
        {},   // direct live spill source
        {0xd8d480e4u, 0x04000003u},   // relay the unresolved source through v4
        {0x7e060300u},   // fully replace source v3 from genuine v0
        {0xbefe04c1u, 0xbf068181u, 0xbf850001u, 0x7e060300u},   // only one arm replaces v3
        {0xbefe04c1u, 0xbf068181u, 0xbf850002u, 0x7e060300u, 0xbf820001u, 0x7e060300u},
        {0x7da40500u, 0x7e060300u, 0xbefe04c1u},   // narrowed replacement leaves spilled peers
    };
    for (size_t shape = 0; shape < replacements.size(); ++shape) {
        for (bool destination_spill : {false, true}) {
            for (bool barrier : {false, true}) {
                auto words = ngg_row_program(0x25u, false);
                const bool valid = shape == 2u || shape == 4u;
                if (!valid) words[words.size() - 2u] = 2u;   // authentic independent sink
                std::vector<uint32_t> prefix;
                if (destination_spill) prefix.insert(prefix.end(), {0xd7610001u, 0x00010081u});
                prefix.insert(prefix.end(), {0xd7610003u, 0x00010081u});
                if (barrier) prefix.push_back(0xbf8a0000u);
                prefix.insert(prefix.end(), replacements[shape].begin(), replacements[shape].end());
                prefix.insert(prefix.end(), {0xd8d480e4u, shape == 1u ? 0x01000004u : 0x01000003u});
                words.insert(words.begin() + 1, prefix.begin(), prefix.end());
                const uint64_t address =
                    0x4959d000u + (shape << 4u) + (destination_spill << 1u) + barrier;
                SCOPED_TRACE(testing::Message()
                             << shape << '/' << destination_spill << '/' << barrier);
                EXPECT_EQ(recompile_ngg_exports_for_test(
                              words.data(), words.size(), 10, 0, nullptr, 4, 0,
                              {RecompileDiagnosticStage::Compute, address}, true, true, false)
                              .empty(),
                          !valid);
                if (!valid) {
                    const auto reason = last_terminal_reject_reason(address);
                    EXPECT_NE(reason.find("pc=" + std::to_string(1u + prefix.size())),
                              std::string::npos)
                        << reason;
                }
            }
        }
    }
}

TEST(ComputeDppRowOr, NggAddRowsKeepPartialJoinedAndRespilledSourcesRefused) {
    const std::vector<std::vector<uint32_t>> replacements{
        {0x7da40500u, 0x7e020283u, 0xbefe04c1u},   // narrowed ordinary write then full restore
        {0xbefe04c1u, 0xbf068181u, 0xbf850001u, 0x7e020283u},   // only one forward arm overwrites
        {0x7e020283u, 0xd7610001u, 0x00010081u},   // current generation becomes a spill again
        {0x64020100u},   // v_add_f16 preserves the unresolved high half
        {0x56020100u},   // v_fmac_f32 consumes the unresolved old destination
        {0x7e021500u},   // plain v_cvt_f16_f32 preserves the high half
        {0x7e02a900u},   // plain v_rcp_f16 preserves the high half
        {0xd7030001u, 0x00020100u},   // VOP3 v_add_nc_u16 preserves the opposite half
        {0xcc210001u, 0x02020100u},   // VOP3P v_fma_mixlo_f16 preserves the high half
        {0xd51f0001u, 0x00020100u},   // VOP3 v_mac_f32 consumes the old destination
    };
    for (size_t shape = 0; shape < replacements.size(); ++shape) {
        for (bool barrier : {false, true}) {
            for (bool native : {false, true}) {
                // Native partial-overwrite lifetime checks retain their separate preexisting route.
                if ((shape == 0u || shape >= 3u) && native) continue;
                auto words = ngg_row_program(0x25u, barrier);
                words[words.size() - 2u] = 2u;
                std::vector<uint32_t> prefix{0xd7610001u, 0x00010081u};
                prefix.insert(prefix.end(), replacements[shape].begin(), replacements[shape].end());
                words.insert(words.begin() + (barrier && shape < 3u ? 2 : 1), prefix.begin(),
                             prefix.end());
                SCOPED_TRACE(testing::Message() << shape << '/' << barrier << '/' << native);
                const uint64_t address = 0x4959b000u + (shape << 4u) + (barrier << 1u) + native;
                EXPECT_TRUE(recompile_ngg_exports_for_test(
                                words.data(), words.size(), 10, 0, nullptr, 4, 0,
                                {RecompileDiagnosticStage::Compute, address}, true, true, native)
                                .empty());
                const auto reason = last_terminal_reject_reason(address);
                const size_t row_pc = 1u + prefix.size() + barrier;
                EXPECT_NE(reason.find("pc=" + std::to_string(row_pc)), std::string::npos) << reason;
            }
        }
    }
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

namespace {
std::vector<uint32_t> phase_relay_program(bool before_barrier) {
    std::vector<uint32_t> words{0xd7610003u, 0x00010081u};   // live v3 scalar lane slot
    if (!before_barrier) words.push_back(0xbf8a0000u);
    words.insert(words.end(), {0xd8d480e4u, 0x04000003u});   // identity DS v4 <- v3
    if (before_barrier) words.push_back(0xbf8a0000u);
    words.insert(words.end(), {0x380808fau, 0xff011104u, 0xf8000941u, 2u, 0xbf810000u});
    return words;
}

void check_phase_relay(const std::vector<uint32_t>& words, uint64_t address, bool genuine,
                       uint32_t row_pc) {
    const auto module = recompile_ngg_exports_for_test(
        words.data(), words.size(), 10, 0, nullptr, 4, 0,
        {RecompileDiagnosticStage::Compute, address}, true, true, false);
    EXPECT_EQ(module.empty(), !genuine);
    if (!genuine) {
        const auto reason = last_terminal_reject_reason(address);
        EXPECT_NE(reason.find("dpp-row-or-source-unresolved"), std::string::npos) << reason;
        EXPECT_NE(reason.find("pc=" + std::to_string(row_pc)), std::string::npos) << reason;
    }
}
}   // namespace

TEST(ComputeDppRowOr, NggLiveSpillRelayBeforeBarrierStaysUnowned) {
    check_phase_relay(phase_relay_program(true), 0x4959b001u, false, 5u);
}

TEST(ComputeDppRowOr, NggLiveSpillRelayAfterBarrierStaysUnowned) {
    check_phase_relay(phase_relay_program(false), 0x4959b002u, false, 5u);
}

TEST(ComputeDppRowOr, NggLiveSpillRelayWithoutBarrierStaysUnowned) {
    auto words = phase_relay_program(true);
    words.erase(words.begin() + 4);
    check_phase_relay(words, 0x4959b003u, false, 4u);
}

TEST(ComputeDppRowOr, NggGenuineSourceReplacementBeforeRelayExportsRow) {
    auto words = phase_relay_program(true);
    words.insert(words.begin() + 2, 0x7e060300u);   // v3 <- genuine v0 before DS
    words[words.size() - 2] = 4u;
    check_phase_relay(words, 0x4959b004u, true, 6u);
}

TEST(ComputeDppRowOr, NggGenuineRelayReplacementAfterBarrierExportsRow) {
    auto words = phase_relay_program(true);
    words.insert(words.begin() + 5, 0x7e080300u);   // v4 <- genuine v0 after barrier
    words[words.size() - 2] = 4u;
    check_phase_relay(words, 0x4959b005u, true, 6u);
}

TEST(ComputeDppRowOr, NggLaterSourceReplacementCannotAuthenticateEarlierRelay) {
    auto words = phase_relay_program(true);
    words.insert(words.begin() + 4, 0x7e060300u);   // expires v3, never defines v4
    check_phase_relay(words, 0x4959b006u, false, 6u);
}

TEST(ComputeDppRowOr, NggRepeatedBarriersCannotAuthenticateEarlierRelay) {
    auto words = phase_relay_program(true);
    words.insert(words.begin() + 5, 0xbf8a0000u);
    check_phase_relay(words, 0x4959b007u, false, 6u);
}

TEST(ComputeDppRowOr, NggRelayReplacementAfterBarrierRequiresEveryFullExecPath) {
    const std::vector<std::vector<uint32_t>> replacements{
        {0xbefe04c1u, 0xbf068181u, 0xbf850001u, 0x7e080300u},   // one arm only
        {0xbefe04c1u, 0xbf068181u, 0xbf850002u, 0x7e080300u, 0xbf820001u, 0x7e080300u},
        {0x7da40500u, 0x7e080300u, 0xbefe04c1u},   // only narrowed EXEC writes
    };
    for (size_t shape = 0; shape < replacements.size(); ++shape) {
        auto words = phase_relay_program(true);
        const bool genuine = shape == 1u;
        words.insert(words.begin() + 5, replacements[shape].begin(), replacements[shape].end());
        if (genuine) words[words.size() - 2] = 4u;
        SCOPED_TRACE(shape);
        check_phase_relay(words, 0x4959b008u + shape, genuine,
                          5u + static_cast<uint32_t>(replacements[shape].size()));
    }
}
