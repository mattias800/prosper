// Architectural hint admission must reach the actual dispatcher without changing logical64
// sinks or lending support to reserved encodings, CLAUSE, or the independent LegacyRaw policy.
#include "fixtures/fragment_packet_exports_fixture.hpp"
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/fragment_packet_mask_requirements.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {
using namespace prosper::gpu;
namespace f = prosper::test::fragment_packet_exports;
void retain(const std::vector<uint32_t>& words, const char* scenario) {
    const auto* root = std::getenv("PROSPER_ARCHITECTURAL_EXP_SPV_DIRECTORY");
    if (words.empty() || !root || !*root) return;
    const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
    ASSERT_NE(test, nullptr);
    std::error_code error;
    std::filesystem::create_directories(root, error);
    ASSERT_FALSE(error) << error.message();
    std::ofstream file(std::filesystem::path(root) /
                           (std::string(test->name()) + "_" + scenario + ".spv"),
                       std::ios::binary);
    file.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
    file.close();
    ASSERT_TRUE(bool(file)) << scenario;
}
FragmentPacketProgram compile(const FragmentInvocationPacket& input, const char* scenario) {
    auto p = recompile_fragment_packet(input);
    retain(p.spirv, scenario);
    return p;
}
std::vector<uint32_t> execute(const FragmentPacketProgram& p) {
    bpermute_oracle::Interpreter vm(p.spirv);
    auto words = vm.run_packet(p.input_words, p.output_words);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return words;
}
TEST(FragmentPacketExports, ScratchAbsentInactiveIsUnobservedNotZeroAndLegacyRemainsRaw) {
    for (uint32_t wave : {0u, 1u, 2u}) {
        auto input = f::scratch(wave).invocation;
        auto p = compile(input, wave == 0 ? "nonzero" : wave == 1 ? "present_zero" : "distinct");
        ASSERT_FALSE(p.spirv.empty()) << p.rejection;
        const auto words = execute(p);
        const auto expected = f::scratch_records(wave);
        ASSERT_GE(words.size(), expected.size());
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), words.begin()));
        const auto decoded = decode_fragment_packet(p, words, true);
        ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
        EXPECT_TRUE(decoded.exports.empty());
        ASSERT_EQ(decoded.architectural_exports.size(), 64u);
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const auto& out = decoded.architectural_exports[lane];
            ASSERT_EQ(out.events.size(), 1u);
            EXPECT_EQ(out.events[0].source_words[0].has_value(), f::active(lane));
            EXPECT_EQ(out.colors[0][0].has_value(), f::active(lane));
            EXPECT_EQ(out.commit_eligible, f::active(lane) && lane != 40);
            if (f::active(lane)) {
                ASSERT_TRUE(out.colors[0][0].has_value());
                EXPECT_EQ(out.colors[0][0]->bits, f::value(wave));
            }
        }
        input.export_observation = FragmentPacketExportObservation::LegacyRaw;
        auto raw = compile(input, wave == 0 ? "legacy0" : wave == 1 ? "legacy1" : "legacy2");
        ASSERT_FALSE(raw.spirv.empty());
        const auto refused = decode_fragment_packet(raw, execute(raw), true);
        EXPECT_EQ(refused.rejection, "packet-vgpr-raw-export-unavailable");
        EXPECT_TRUE(refused.exports.empty());
    }
}
TEST(FragmentPacketExports, EntirelyInactiveMayHaveNoPayloadButKeepsControl) {
    const auto p = compile(f::scratch(0, true).invocation, "inactive");
    ASSERT_FALSE(p.spirv.empty());
    const auto words = execute(p);
    const auto expected = f::scratch_records(0, true);
    ASSERT_GE(words.size(), expected.size());
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), words.begin()));
    const auto decoded = decode_fragment_packet(p, words, true);
    ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
    ASSERT_EQ(decoded.architectural_exports.size(), 64u);
    for (const auto& lane : decoded.architectural_exports) {
        ASSERT_EQ(lane.events.size(), 1u);
        EXPECT_TRUE(lane.terminal_done);
        EXPECT_TRUE(lane.valid_mask_available);
        EXPECT_FALSE(lane.valid_mask);
        EXPECT_FALSE(lane.commit_eligible);
        EXPECT_FALSE(lane.events[0].source_words[0].has_value());
    }
}
TEST(FragmentPacketExports, CanonicalPrefetchPreservesArchitecturalPacketAndResourceSinks) {
    for (uint32_t wave : {0u, 1u}) {
        SCOPED_TRACE(wave); // nonzero and genuinely present zero are distinct inputs
        auto expected = f::scratch_records(wave);
        for (uint32_t lane = 0; lane < 64; ++lane) expected[lane * 14 + 12] = 2;
        for (uint32_t mode : {1u, 2u, 3u}) {
            SCOPED_TRACE(mode);
            auto input = f::scratch(wave);
            input.invocation.guest_code.insert(input.invocation.guest_code.begin(),
                                               0xbfa00000u | mode);
            const auto check_program = [&](const FragmentPacketProgram& program) {
                ASSERT_FALSE(program.spirv.empty()) << program.rejection;
                EXPECT_EQ(program.export_observation,
                          FragmentPacketExportObservation::Architectural);
                EXPECT_EQ(program.demanded_initial_masks, kPacketInitialExec);
                const auto words = execute(program);
                ASSERT_GE(words.size(), expected.size());
                EXPECT_TRUE(std::equal(expected.begin(), expected.end(), words.begin()));
                const auto decoded = decode_fragment_packet(program, words, true);
                ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
                ASSERT_EQ(decoded.architectural_exports.size(), 64u);
                for (uint32_t lane = 0; lane < 64; ++lane) {
                    const auto& out = decoded.architectural_exports[lane];
                    ASSERT_EQ(out.events.size(), 1u);
                    EXPECT_EQ(out.events[0].site.pc, 2u);
                    EXPECT_EQ(out.colors[0][0].has_value(), f::active(lane));
                    EXPECT_EQ(out.commit_eligible, f::active(lane) && lane != 40);
                    if (f::active(lane)) {
                        ASSERT_TRUE(out.colors[0][0].has_value());
                        EXPECT_EQ(out.colors[0][0]->bits, f::value(wave));
                    }
                }
            };
            const auto scenario = std::to_string(wave) + "_" + std::to_string(mode);
            const auto packet = compile(input.invocation, ("prefetch_packet_" + scenario).c_str());
            check_program(packet);
            const auto resource = recompile_fragment_resource_packet(input);
            retain(resource.packet.spirv, ("prefetch_resource_" + scenario).c_str());
            check_program(resource.packet);
            ASSERT_FALSE(resource.packet.spirv.empty()) << resource.packet.rejection;
            const auto decoded = decode_fragment_resource_packet(
                resource, execute(resource.packet), true, input.device.device_identity);
            ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
            EXPECT_EQ(decoded.architectural_exports.size(), 64u);
        }
    }
}
TEST(FragmentPacketExports, ReservedPrefetchAndClauseRefuseAtTheActualArchitecturalDispatcher) {
    for (bool resource_route : {false, true}) {
        SCOPED_TRACE(resource_route);
        auto input = f::scratch();
        input.invocation.guest_code.insert(input.invocation.guest_code.begin(), 0xbfa00003u);
        const auto compile_route = [&] {
            return resource_route ? recompile_fragment_resource_packet(input).packet
                                  : recompile_fragment_packet(input.invocation);
        };
        ASSERT_FALSE(compile_route().spirv.empty()) << "same original with canonical PREFETCH3";
        for (uint32_t word : {0xbfa00000u, 0xbfa00004u, 0xbfa00103u, 0xbfa08003u, 0xbfa10000u,
                              0xbfa10001u, 0xbfa1ffffu}) {
            SCOPED_TRACE(word);
            input.invocation.guest_code[0] = word;   // only the first original instruction differs
            const auto refused = compile_route();
            EXPECT_EQ(refused.rejection, (word & 0xffff0000u) == 0xbfa00000u
                                             ? "packet-prefetch-mode-unimplemented"
                                             : "packet-clause-unimplemented");
            EXPECT_TRUE(refused.spirv.empty());
            EXPECT_TRUE(refused.input_words.empty());
            EXPECT_TRUE(refused.output_words.empty());
        }
    }
}
TEST(FragmentPacketExports, ArchitecturalPrefetchDoesNotExpandLegacyRawPacketOrResourcePolicy) {
    for (bool resource_route : {false, true}) {
        SCOPED_TRACE(resource_route);
        auto input = f::scratch();
        input.invocation.export_observation = FragmentPacketExportObservation::LegacyRaw;
        input.invocation.guest_code.insert(input.invocation.guest_code.begin(), 0xbf800000u);
        const auto compile_route = [&] {
            return resource_route ? recompile_fragment_resource_packet(input).packet
                                  : recompile_fragment_packet(input.invocation);
        };
        ASSERT_FALSE(compile_route().spirv.empty()) << "same LegacyRaw original with a NOP";
        for (uint32_t word : {0xbfa00001u, 0xbfa00002u, 0xbfa00003u, 0xbfa00000u, 0xbfa00004u,
                              0xbfa00103u, 0xbfa08003u, 0xbfa10001u}) {
            SCOPED_TRACE(word);
            input.invocation.guest_code[0] = word;
            const auto refused = compile_route();
            EXPECT_EQ(refused.rejection, "packet-control-unimplemented");
            EXPECT_TRUE(refused.spirv.empty());
            EXPECT_TRUE(refused.input_words.empty());
            EXPECT_TRUE(refused.output_words.empty());
        }
    }
}
TEST(FragmentPacketExports, OnlyDemandedInitialMasksSupplyArchitecturalAuthority) {
    auto input = f::scratch().invocation;
    auto vertex = input;
    vertex.stage = GraphicsPacketStage::Vertex;
    const auto wrong_stage = recompile_fragment_packet(vertex);
    EXPECT_TRUE(wrong_stage.spirv.empty());
    EXPECT_EQ(wrong_stage.rejection, "packet-architectural-export-stage-unimplemented")
        << "the explicit PS observation contract cannot admit vertex exports";
    input.mask_state_available = false;
    input.exec_available = true;
    input.vcc_mask = UINT64_MAX; // deliberately absent, NOT a value used to seed the compiler
    input.scc = true;
    const auto exec_only = compile(input, "only_exec_present");
    ASSERT_FALSE(exec_only.spirv.empty()) << exec_only.rejection;
    EXPECT_EQ(exec_only.initial_mask_availability, kPacketInitialExec);
    EXPECT_EQ(exec_only.demanded_initial_masks, kPacketInitialExec);
    const auto first_words = execute(exec_only);
    const auto first_expected = f::scratch_records(0);
    ASSERT_GE(first_words.size(), first_expected.size());
    EXPECT_TRUE(std::equal(first_expected.begin(), first_expected.end(), first_words.begin()));
    input.exec_available = false;
    EXPECT_EQ(recompile_fragment_packet(input).rejection, "packet-entry-exec-unavailable");
    input.guest_code.insert(input.guest_code.begin(), 0xbefe0414u);   // real entry-independent EXEC
    const auto writer = compile(input, "genuine_exec_writer");
    ASSERT_FALSE(writer.spirv.empty()) << writer.rejection;
    EXPECT_EQ(writer.initial_mask_availability, 0u);
    EXPECT_EQ(writer.demanded_initial_masks, 0u);
    auto expected = f::scratch_records(0);
    for (uint32_t lane = 0; lane < 64; ++lane) expected[lane * 14 + 12] = 2;
    const auto words = execute(writer);
    ASSERT_GE(words.size(), expected.size());
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), words.begin()));
    const auto decoded = decode_fragment_packet(writer, words, true);
    ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
    ASSERT_EQ(decoded.architectural_exports.size(), 64u);
    ASSERT_TRUE(decoded.architectural_exports[63].colors[0][0].has_value());
    EXPECT_EQ(decoded.architectural_exports[63].colors[0][0]->bits, f::value(0));
}
TEST(FragmentPacketExports, ActiveReadIgnoresHostEligibilityAndLaterVm) {
    for (bool missing : {false, true}) {
        const auto p =
            compile(f::missing_active(missing).invocation, missing ? "absent" : "supplied");
        ASSERT_FALSE(p.spirv.empty());
        const auto decoded = decode_fragment_packet(p, execute(p), true);
        if (missing) {
            EXPECT_EQ(decoded.rejection, "packet-vgpr-architectural-export-unavailable");
            EXPECT_EQ(decoded.lane, 40u);
            EXPECT_EQ(decoded.pc, 0u);
            EXPECT_TRUE(decoded.architectural_exports.empty());
        } else {
            ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
            ASSERT_EQ(decoded.architectural_exports.size(), 64u);
            ASSERT_TRUE(decoded.architectural_exports[40].colors[0][0].has_value());
            EXPECT_EQ(decoded.architectural_exports[40].colors[0][0]->bits, 40u);
            EXPECT_FALSE(decoded.architectural_exports[40].commit_eligible);
        }
    }
}
TEST(FragmentPacketExports, InactiveSelectedPeerAndWqmReactivationStillDemandRealWords) {
    for (uint32_t family : {0u, 1u})
        for (bool missing : {false, true}) {
            const auto p = compile((family ? f::wqm(missing) : f::peer(missing)).invocation,
                                   family    ? missing ? "wqm_absent" : "wqm_good"
                                   : missing ? "peer_absent"
                                             : "peer_good");
            ASSERT_FALSE(p.spirv.empty()) << p.rejection;
            const auto decoded = decode_fragment_packet(p, execute(p), true);
            if (missing) {
                EXPECT_EQ(decoded.rejection, family ? "packet-vgpr-architectural-export-unavailable"
                                                    : "packet-vgpr-selected-peer-unavailable");
                EXPECT_EQ(decoded.pc, family ? 1u : 0u);
                EXPECT_TRUE(decoded.architectural_exports.empty());
            } else {
                ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
                ASSERT_EQ(decoded.architectural_exports.size(), 64u);
                ASSERT_TRUE(decoded.architectural_exports[63].colors[0][0].has_value());
                EXPECT_EQ(decoded.architectural_exports[63].colors[0][0]->bits,
                          family ? 0x7700003fu : 0x87000028u);
                EXPECT_EQ(decoded.architectural_exports[60].colors[0][0].has_value(), bool(family));
            }
        }
}
TEST(FragmentPacketExports, ChannelAccumulationNullAndLastVmAreNotFinalExec) {
    const auto p = compile(f::multiple().invocation, "multiple");
    ASSERT_FALSE(p.spirv.empty()) << p.rejection;
    auto words = execute(p);
    const auto decoded = decode_fragment_packet(p, words, true);
    ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
    ASSERT_EQ(decoded.architectural_exports.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto& out = decoded.architectural_exports[lane];
        ASSERT_EQ(out.events.size(), 4u);
        EXPECT_EQ(out.events[0].site.pc, 0u);
        EXPECT_EQ(out.events[1].site.pc, 4u);
        EXPECT_EQ(out.events[2].site.pc, 6u);
        EXPECT_EQ(out.events[3].site.pc, 10u);
        EXPECT_TRUE(out.events[3].exec);
        EXPECT_EQ(out.valid_mask, f::active(lane));
        EXPECT_EQ(out.commit_eligible, f::active(lane) && lane != 40);
        ASSERT_TRUE(out.colors[2][0].has_value());
        EXPECT_EQ(out.colors[2][0]->bits, 0x80010000u + lane);
        ASSERT_TRUE(out.colors[2][1].has_value());
        EXPECT_EQ(out.colors[2][1]->bits, 0x80020000u + lane);
        EXPECT_EQ(out.colors[2][2].has_value(), f::active(lane));
        if (f::active(lane)) {
            ASSERT_TRUE(out.colors[2][2].has_value());
            EXPECT_EQ(out.colors[2][2]->bits, 0x80030000u + lane);
        }
        EXPECT_FALSE(out.colors[2][3].has_value());
        for (const auto& source : out.events[2].source_words) EXPECT_FALSE(source.has_value());
    }
    ASSERT_GE(words.size(), 64u * 56u);
    words[63 * 56 + 12] ^= 1;
    EXPECT_EQ(decode_fragment_packet(p, words, true).rejection,
              "packet-architectural-export-record-invalid");
}
TEST(FragmentPacketExports, PendingWordsAndEveryExecWriterNeedActualCompletion) {
    EXPECT_EQ(recompile_fragment_packet(f::multiple(false).invocation).rejection,
              "packet-export-exec-overwrite-before-wait");   // historical unsafe snapshot refused
    for (uint32_t family = 0; family < 7; ++family) {
        const auto unsafe = f::pending_write(family, false).invocation;
        const auto refused = recompile_fragment_packet(unsafe);
        if (family < 5) {
            EXPECT_EQ(refused.rejection, family == 0 ? "packet-export-source-overwrite-before-wait"
                                                     : "packet-export-exec-overwrite-before-wait");
            EXPECT_TRUE(refused.spirv.empty());
        } else {
            ASSERT_FALSE(refused.spirv.empty()) << refused.rejection;
            retain(refused.spirv, family == 5 ? "vcc_not_exec" : "disjoint_word");
            EXPECT_TRUE(decode_fragment_packet(refused, execute(refused), true).rejection.empty());
        }
        const auto good = compile(f::pending_write(family, true).invocation,
                                  ("wait_writer_" + std::to_string(family)).c_str());
        ASSERT_FALSE(good.spirv.empty()) << good.rejection;
        const auto words = execute(good);
        const auto expected = f::pending_records(family, 0);
        ASSERT_GE(words.size(), expected.size());
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), words.begin()));
        const auto decoded = decode_fragment_packet(good, words, true);
        ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
        ASSERT_EQ(decoded.architectural_exports.size(), 64u);
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const auto& events = decoded.architectural_exports[lane].events;
            ASSERT_EQ(events.size(), 2u);
            EXPECT_EQ(events[0].exec, f::active(lane));
            if (f::active(lane)) {
                ASSERT_TRUE(events[0].source_words[0].has_value());
                EXPECT_EQ(*events[0].source_words[0], 0x87000000u + lane);
            }
            const bool on = family == 3   ? f::active(lane) && lane != 40
                            : family == 4 ? lane < 4 || (lane >= 28 && lane < 36) ||
                                                (lane >= 40 && lane < 44) || lane >= 60
                                          : f::active(lane);
            EXPECT_EQ(events[1].exec, on);
            EXPECT_EQ(events[1].source_words[0].has_value(), on);
            if (on) {
                ASSERT_TRUE(events[1].source_words[0].has_value());
                EXPECT_EQ(*events[1].source_words[0],
                          family == 0 ? f::value(0) : 0x87000000u + lane);
            }
        }
    }
    auto done = f::pending_write(0, false).invocation;
    done.guest_code[0] |= 1u << 11;   // DONE alone cannot protect the pending payload overwrite
    EXPECT_EQ(recompile_fragment_packet(done).rejection,
              "packet-export-source-overwrite-before-wait");
    auto unrelated = f::pending_write(0, true).invocation;
    unrelated.guest_code[2] = 0xbf8c0070u;   // LGKM0/VM0 do not complete an EXPCNT7 request
    EXPECT_EQ(recompile_fragment_packet(unrelated).rejection,
              "packet-export-source-overwrite-before-wait");
}
TEST(FragmentPacketExports, EveryConditionalPathAndPackedPhysicalWordMustStayStable) {
    for (bool scc : {false, true}) {
        EXPECT_EQ(recompile_fragment_packet(f::pending_join(false, scc).invocation).rejection,
                  "packet-export-source-overwrite-before-wait");
        const auto good = compile(f::pending_join(true, scc).invocation,
                                  scc ? "both_paths_one" : "both_paths_zero");
        ASSERT_FALSE(good.spirv.empty()) << good.rejection;
        const auto decoded = decode_fragment_packet(good, execute(good), true);
        ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
        ASSERT_EQ(decoded.architectural_exports.size(), 64u);
        for (uint32_t lane = 0; lane < 64; ++lane)
            if (f::active(lane)) {
                ASSERT_EQ(decoded.architectural_exports[lane].events.size(), 2u);
                ASSERT_TRUE(
                    decoded.architectural_exports[lane].events[0].source_words[0].has_value());
                EXPECT_EQ(*decoded.architectural_exports[lane].events[0].source_words[0],
                          0x87000000u + lane);
                ASSERT_TRUE(decoded.architectural_exports[lane].colors[0][0].has_value());
                EXPECT_EQ(decoded.architectural_exports[lane].colors[0][0]->bits, f::value(0));
            }
    }
    for (uint32_t reg : {6u, 7u}) {
        auto input = f::compressed(12).invocation;
        input.guest_code[0] &= ~((1u << 11) | (1u << 12));
        input.guest_code.pop_back();
        f::fp::vmov(input.guest_code, reg, 0);
        f::exp(input.guest_code, 9, 0, 0, true, true);
        input.guest_code.push_back(0xbf810000u);
        const auto actual = compile(input, reg == 6 ? "packed_pending" : "packed_disjoint");
        if (reg == 6) {
            EXPECT_EQ(actual.rejection, "packet-export-source-overwrite-before-wait");
            EXPECT_TRUE(actual.spirv.empty());
        } else {
            ASSERT_FALSE(actual.spirv.empty()) << actual.rejection;
            const auto decoded = decode_fragment_packet(actual, execute(actual), true);
            ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
            ASSERT_EQ(decoded.architectural_exports.size(), 64u);
            ASSERT_TRUE(decoded.architectural_exports[63].colors[3][2].has_value());
            EXPECT_EQ(decoded.architectural_exports[63].colors[3][2]->bits, 63u);
            ASSERT_TRUE(decoded.architectural_exports[63].colors[3][3].has_value());
            EXPECT_EQ(decoded.architectural_exports[63].colors[3][3]->bits, 0x7fffu);
        }
    }
}
TEST(FragmentPacketExports, NumericSaveexecKeepsCurrentWordsOldMaskAndNewSccDistinct) {
    for (uint32_t scenario = 0; scenario < f::numeric_saveexec_cases.size(); ++scenario) {
        const auto input = f::numeric_saveexec(scenario);
        const auto p = compile(input.invocation, f::numeric_saveexec_cases[scenario].name);
        ASSERT_FALSE(p.spirv.empty()) << p.rejection;
        EXPECT_EQ(p.demanded_initial_masks, kPacketInitialExec);
        const auto words = execute(p);
        const auto expected = f::numeric_saveexec_records(scenario, 0);
        ASSERT_GE(words.size(), expected.size());
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), words.begin()));
        const auto decoded = decode_fragment_packet(p, words, true);
        ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
        ASSERT_EQ(decoded.architectural_exports.size(), 64u);
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const auto& output = decoded.architectural_exports[lane];
            EXPECT_EQ(output.colors[0][0].has_value(), f::active(lane));
            if (f::active(lane)) {
                ASSERT_TRUE(output.colors[0][0].has_value());
                ASSERT_TRUE(output.colors[0][1].has_value());
                ASSERT_TRUE(output.colors[0][2].has_value());
                EXPECT_EQ(output.colors[0][1]->bits, 0x80000001u);
                EXPECT_EQ(output.colors[0][2]->bits, 0x80000101u);
                EXPECT_EQ(output.colors[0][0]->bits, f::numeric_saveexec_cases[scenario].new_nonzero
                                                         ? f::value(0)
                                                         : 0x6247f000u);
            }
        }
    }
    for (uint32_t absent : {22u, 23u}) {
        auto missing = f::numeric_saveexec(0);
        std::erase_if(missing.invocation.sgprs,
                      [=](const auto& word) { return word.first == absent; });
        const auto p = recompile_fragment_packet(missing.invocation);
        EXPECT_TRUE(p.spirv.empty());
        EXPECT_EQ(p.rejection, "packet-sgpr-read-before-definition")
            << "original numeric MOV at PC1 must not borrow its absent s" << absent
            << " from old entry s20:21 or the saved Bool; this is not an AND-site guard fault";
    }
    for (uint32_t absent : {20u, 21u}) {
        auto missing = f::pending_write(2, true);
        std::erase_if(missing.invocation.sgprs,
                      [=](const auto& word) { return word.first == absent; });
        const auto p = recompile_fragment_packet(missing.invocation);
        EXPECT_TRUE(p.spirv.empty());
        EXPECT_EQ(p.rejection, "packet-sgpr-read-before-definition")
            << "the original waited AND_SAVEEXEC at PC3 needs both real source words";
    }
}
TEST(FragmentPacketExports, CompressedReadsTwoPhysicalWordsAndKeepsRaw16BitChannels) {
    for (uint32_t en : {0u, 3u, 12u, 15u}) {
        const auto input = f::compressed(en).invocation;
        const auto p = compile(input, en == 0 ? "none" : en == 3 ? "rg" : en == 12 ? "ba" : "rgba");
        ASSERT_FALSE(p.spirv.empty()) << p.rejection;
        const auto decoded = decode_fragment_packet(p, execute(p), true);
        ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
        ASSERT_EQ(decoded.architectural_exports.size(), 64u);
        for (uint32_t lane = 0; lane < 64; ++lane)
            for (uint32_t channel = 0; channel < 4; ++channel) {
                const auto& component = decoded.architectural_exports[lane].colors[3][channel];
                const bool observed = f::active(lane) && (en & (1u << channel));
                EXPECT_EQ(component.has_value(), observed);
                if (observed) {
                    ASSERT_TRUE(component.has_value());
                    const uint32_t packed = (channel < 2 ? 0x12348000u : 0x7fff0000u) + lane;
                    EXPECT_EQ(component->bits, (packed >> ((channel & 1u) * 16)) & 0xffffu);
                    EXPECT_EQ(component->width, 16u);
                }
            }
        auto invalid = input;
        invalid.guest_code[0] = (invalid.guest_code[0] & ~15u) | 1u;
        EXPECT_EQ(recompile_fragment_packet(invalid).rejection,
                  "packet-compressed-export-channel-mask-invalid");
        invalid = input;
        invalid.guest_code[0] = (invalid.guest_code[0] & ~(63u << 4)) | (8u << 4);
        EXPECT_EQ(recompile_fragment_packet(invalid).rejection,
                  "packet-compressed-depth-export-unimplemented");
    }
}
TEST(FragmentPacketExports, ObservationSchemaAndCompletedWholeWaveAreImmutable) {
    auto p = compile(f::scratch().invocation, "schema");
    ASSERT_FALSE(p.spirv.empty());
    auto words = execute(p);
    ASSERT_GE(words.size(), 2u * 14u);
    ASSERT_EQ(p.export_sites.size(), 1u);
    EXPECT_FALSE(decode_fragment_packet(p, words, true).architectural_exports.empty());
    EXPECT_TRUE(decode_fragment_packet(p, words, false).architectural_exports.empty());
    auto mutated = p;
    mutated.export_sites[0].vm = 0;
    EXPECT_EQ(decode_fragment_packet(mutated, words, true).rejection,
              "packet-export-observation-schema-mismatch");
    words[1 * 14 + 13] = 1;
    EXPECT_EQ(decode_fragment_packet(p, words, true).rejection,
              "packet-export-observation-mask-invalid");
    words = execute(p);
    ASSERT_GE(words.size(), 2u * 14u);
    words[1 * 14 + 8] = 1;
    EXPECT_EQ(decode_fragment_packet(p, words, true).rejection,
              "packet-unobserved-export-payload-invalid");
    auto no_vm = f::scratch().invocation;
    no_vm.guest_code[1] &= ~(1u << 12);
    const auto vm = compile(no_vm, "no_vm");
    EXPECT_EQ(decode_fragment_packet(vm, execute(vm), true).rejection,
              "packet-export-terminal-or-valid-mask-unproved");
    no_vm = f::scratch().invocation;
    no_vm.guest_code[1] &= ~(1u << 11);
    const auto done = compile(no_vm, "no_done");
    EXPECT_EQ(decode_fragment_packet(done, execute(done), true).rejection,
              "packet-export-terminal-or-valid-mask-unproved");
}
TEST(FragmentPacketExports, WideImageDestinationCannotOverwritePendingHighestPhysicalWord) {
    const auto refused = recompile_fragment_resource_packet(f::pending_image(false));
    EXPECT_EQ(refused.packet.rejection, "packet-export-source-overwrite-before-wait");
    EXPECT_TRUE(refused.packet.spirv.empty());
    const auto input = f::pending_image(true);
    const auto good = recompile_fragment_resource_packet(input);
    retain(good.packet.spirv, "highest_image_word_wait");
    ASSERT_FALSE(good.packet.spirv.empty()) << good.packet.rejection;
    bpermute_oracle::Interpreter vm(good.packet.spirv);
    for (uint32_t image = 0; image < good.images.size(); ++image)
        for (const auto& mip : good.images[image].mips)
            vm.sampled_images[16 + image].push_back({mip.width, mip.height, mip.texels});
    const auto words = vm.run_packet(good.packet.input_words, good.packet.output_words);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    const auto decoded =
        decode_fragment_resource_packet(good, words, true, input.device.device_identity);
    ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
    ASSERT_EQ(decoded.architectural_exports.size(), 64u);
    const uint32_t sampled[]{0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u};
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto& out = decoded.architectural_exports[lane];
        ASSERT_EQ(out.events.size(), 2u);
        EXPECT_EQ(out.events[0].source_words[0].has_value(), f::active(lane));
        if (f::active(lane)) {
            ASSERT_TRUE(out.events[0].source_words[0].has_value());
            EXPECT_EQ(*out.events[0].source_words[0], f::value(0) + lane);
            for (uint32_t channel = 0; channel < 4; ++channel) {
                ASSERT_TRUE(out.colors[0][channel].has_value());
                EXPECT_EQ(out.colors[0][channel]->bits, sampled[channel]);
            }
        }
    }
}
TEST(FragmentPacketExports, CachedModeAndEveryLateWaveFailureBlockTypedPublication) {
    auto prototype = f::scratch();
    const auto k =
        std::make_shared<const FragmentPacketKernel>(recompile_fragment_packet_kernel(prototype));
    ASSERT_FALSE(k->program.packet.spirv.empty()) << k->program.packet.rejection;
    retain(k->program.packet.spirv, "cached");
    std::vector<FragmentResourcePacket> waves{f::scratch(0), f::scratch(1), f::scratch(2)};
    const auto run = [&](const FragmentPacketWaveBatch& batch) {
        auto words = batch.output_words;
        for (uint32_t wave : {2u, 0u, 1u}) {
            bpermute_oracle::Interpreter vm(k->program.packet.spirv);
            words = vm.run_buffers(
                64, {{0, batch.input_words}, {1, words}, {2, batch.authority->words()}}, 1, wave);
            EXPECT_TRUE(vm.error.empty()) << vm.error;
        }
        return words;
    };
    auto batch = pack_fragment_packet_waves(k, waves, f::placements(*k));
    ASSERT_TRUE(batch.rejection.empty()) << batch.rejection;
    ASSERT_EQ(batch.placements.size(), 3u);
    auto words = run(batch);
    auto decoded =
        decode_fragment_packet_waves(batch, words, true, prototype.device.device_identity);
    ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
    ASSERT_EQ(decoded.architectural_exports.size(), 3u);
    EXPECT_TRUE(decoded.exports.empty());
    for (uint32_t wave = 0; wave < 3; ++wave) {
        ASSERT_EQ(decoded.architectural_exports[wave].size(), 64u);
        const auto start = batch.placements[wave].output_base + 128;
        const auto expected = f::scratch_records(wave);
        ASSERT_GE(words.size(), start + expected.size());
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), words.begin() + start));
        ASSERT_TRUE(decoded.architectural_exports[wave][63].colors[0][0].has_value());
        EXPECT_EQ(decoded.architectural_exports[wave][63].colors[0][0]->bits, f::value(wave));
    }
    ASSERT_GE(words.size(), batch.placements[2].output_base + 128 +
                                k->program.packet.vgpr_status_offset + 64 * 4);
    words[batch.placements[2].output_base + 128 + k->program.packet.vgpr_status_offset + 63 * 4 +
          1] = 1;
    words[batch.placements[2].output_base + 128 + k->program.packet.vgpr_status_offset + 63 * 4 +
          2] = 1;
    words[batch.placements[2].output_base + 128 + k->program.packet.vgpr_status_offset + 63 * 4 +
          3] = 4;
    decoded = decode_fragment_packet_waves(batch, words, true, prototype.device.device_identity);
    EXPECT_EQ(decoded.wave, 2u);
    EXPECT_EQ(decoded.pc, 1u);
    EXPECT_TRUE(decoded.architectural_exports.empty());
    auto wrong = waves;
    wrong[2].invocation.export_observation = FragmentPacketExportObservation::LegacyRaw;
    EXPECT_EQ(pack_fragment_packet_waves(k, wrong).rejection,
              "packet-wave-code-generation-profile-mismatch");
}
TEST(FragmentPacketExports, ReactivatedP2OldDestinationRemainsAReadNotExportExemption) {
    for (bool missing : {false, true}) {
        const auto input = f::previous_destination(missing);
        const auto p = recompile_fragment_resource_packet(input);
        retain(p.packet.spirv, missing ? "old_absent" : "old_supplied");
        ASSERT_FALSE(p.packet.spirv.empty()) << p.packet.rejection;
        const auto words = execute(p.packet);
        const auto decoded =
            decode_fragment_resource_packet(p, words, true, input.device.device_identity);
        if (missing) {
            EXPECT_EQ(decoded.rejection, "packet-runtime-vgpr-read-before-definition");
            EXPECT_EQ(decoded.lane, 60u);
            EXPECT_EQ(decoded.pc, 3u);
            EXPECT_EQ(decoded.vgpr, 10u);
            EXPECT_TRUE(decoded.architectural_exports.empty());
        } else {
            ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
            ASSERT_EQ(decoded.architectural_exports.size(), 64u);
            for (uint32_t lane = 0; lane < 64; ++lane) {
                EXPECT_EQ(decoded.architectural_exports[lane].colors[0][0].has_value(), lane >= 60);
                if (lane >= 60) {
                    ASSERT_TRUE(decoded.architectural_exports[lane].colors[0][0].has_value());
                    EXPECT_EQ(decoded.architectural_exports[lane].colors[0][0]->bits, 0x3f800000u);
                }
            }
        }
    }
}
}   // namespace
