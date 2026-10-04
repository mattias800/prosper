// #4370: original per-counter WAITs must not lend readiness to another counter's results.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/fragment_packet_exports_fixture.hpp"
#include "gpu/recompiler/fragment_packet_export_timing.hpp"
#include "gpu/recompiler/fragment_packet_mask_requirements.hpp"
#include "gpu/recompiler/fragment_packet_scalar_reads.hpp"
#include "gpu/recompiler/fragment_packet_services.hpp"
#include "gpu/recompiler/rdna2_waitcnt.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace exports = prosper::test::fragment_packet_exports;
namespace resources = prosper::test::fragment_resource_packet;
constexpr uint32_t Lgkm0 = 0xbf8cc07fu, Vm0 = 0xbf8c3f70u, Exp0 = 0xbf8cff0fu;
constexpr uint32_t NoDrain = 0xbf8cff7fu;

std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    return ins;
}
void retain(const std::vector<uint32_t>& module, const std::string& scenario) {
    const auto* root = std::getenv("PROSPER_PACKET_WAITCNT_SPV_DIRECTORY");
    if (module.empty() || !root || !*root) return;
    const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
    ASSERT_NE(test, nullptr);
    std::error_code error;
    std::filesystem::create_directories(root, error);
    ASSERT_FALSE(error) << error.message();
    std::ofstream file(std::filesystem::path(root) /
                           (std::string(test->name()) + "_" + scenario + ".spv"),
                       std::ios::binary);
    file.write(reinterpret_cast<const char*>(module.data()), std::streamsize(module.size() * 4));
    file.close();
    ASSERT_TRUE(bool(file));
}
void raw_resource_sink(const FragmentResourcePacket& input, const std::string& scenario) {
    const auto program = recompile_fragment_resource_packet(input);
    ASSERT_FALSE(program.packet.spirv.empty()) << program.packet.rejection;
    retain(program.packet.spirv, scenario);
    bpermute_oracle::Interpreter vm(program.packet.spirv);
    for (uint32_t image = 0; image < program.images.size(); ++image)
        for (const auto& mip : program.images[image].mips)
            vm.sampled_images[16 + image].push_back({mip.width, mip.height, mip.texels});
    const auto words = vm.run_packet(program.packet.input_words, program.packet.output_words);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    const auto result =
        decode_fragment_resource_packet(program, words, true, input.device.device_identity);
    ASSERT_TRUE(result.rejection.empty()) << result.rejection;
    EXPECT_EQ(result.exports, resources::expected_chain(input, false));
}
void resource_refusal(const FragmentResourcePacket& input, const char* reason, uint32_t pc) {
    uint32_t actual_pc = UINT32_MAX;
    const auto* gap =
        packet_resource_preflight(input, decode(input.invocation.guest_code), actual_pc);
    ASSERT_NE(gap, nullptr);
    EXPECT_STREQ(gap, reason);
    EXPECT_EQ(actual_pc, pc);
    const auto result = recompile_fragment_resource_packet(input);
    EXPECT_EQ(result.packet.rejection, reason);
    EXPECT_TRUE(result.packet.spirv.empty());
    EXPECT_TRUE(result.packet.input_words.empty());
    EXPECT_TRUE(result.packet.output_words.empty());
}
void resource_wait_refusal(const FragmentResourcePacket& input, const char* reason, uint32_t pc) {
    const auto ins = decode(input.invocation.guest_code);
    const auto found =
        std::find_if(ins.begin(), ins.end(), [pc](const auto& in) { return in.pc == pc; });
    ASSERT_NE(found, ins.end());
    ASSERT_EQ(found->fmt, Rdna2Format::SOPP);
    ASSERT_EQ(found->opcode, 0x0cu);
    // This whole-program instruction gate precedes the resource pending-state transfer.
    EXPECT_STREQ(packet_resource_instruction_gap(*found), reason);
    const auto result = recompile_fragment_resource_packet(input);
    EXPECT_EQ(result.packet.rejection, reason);
    EXPECT_TRUE(result.packet.spirv.empty());
    EXPECT_TRUE(result.packet.input_words.empty());
    EXPECT_TRUE(result.packet.output_words.empty());
}
void export_sink(const FragmentInvocationPacket& input, const std::vector<uint32_t>& expected,
                 const std::string& scenario) {
    const auto program = recompile_fragment_packet(input);
    ASSERT_FALSE(program.spirv.empty()) << program.rejection;
    retain(program.spirv, scenario);
    bpermute_oracle::Interpreter vm(program.spirv);
    const auto words = vm.run_packet(program.input_words, program.output_words);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_GE(words.size(), expected.size());
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), words.begin()));
    EXPECT_TRUE(decode_fragment_packet(program, words, true).rejection.empty());
}
void export_refusal(const FragmentInvocationPacket& input, const char* reason, uint32_t pc) {
    uint32_t actual_pc = UINT32_MAX;
    const auto* gap = fragment_packet_export_timing_gap(decode(input.guest_code), actual_pc);
    ASSERT_NE(gap, nullptr);
    EXPECT_STREQ(gap, reason);
    EXPECT_EQ(actual_pc, pc);
    const auto result = recompile_fragment_packet(input);
    EXPECT_EQ(result.rejection, reason);
    EXPECT_TRUE(result.spirv.empty());
}

TEST(FragmentPacketWaitcnt, SplitVmcntAndSixBitLgkmcntAreDecodedIndependently) {
    struct Rail {
        uint16_t immediate;
        uint8_t vm, exp, lgkm;
    };
    for (const auto rail :
         {Rail{0, 0, 0, 0}, Rail{0xc07f, 63, 7, 0}, Rail{0x3f70, 0, 7, 63}, Rail{0xff0f, 63, 0, 63},
          Rail{0xff7f, 63, 7, 63}, Rail{0x5013, 19, 1, 16}, Rail{0xa025, 37, 2, 32}}) {
        SCOPED_TRACE(rail.immediate);
        const auto value = decode_rdna2_waitcnt(rail.immediate);
        EXPECT_EQ(value.vmcnt, rail.vm);
        EXPECT_EQ(value.expcnt, rail.exp);
        EXPECT_EQ(value.lgkmcnt, rail.lgkm);
        EXPECT_EQ(value.unmodeled_bits, 0);
        EXPECT_EQ(value.drains_scalar_reads(), rail.lgkm == 0);
        EXPECT_EQ(value.drains_vector_reads(), rail.vm == 0);
        EXPECT_EQ(value.drains_exports(), rail.exp == 0);
    }
    for (uint16_t immediate : {0u, 0xc07fu, 0x3f70u, 0xff0fu, 0xff7fu})
        EXPECT_EQ(rdna2_waitcnt_execution_gap(immediate), nullptr);
    for (uint16_t immediate : {0xc17fu, 0x3f71u, 0xff1fu}) {
        EXPECT_TRUE(rdna2_waitcnt_effects_known(immediate));
        EXPECT_STREQ(rdna2_waitcnt_execution_gap(immediate),
                     "packet-waitcnt-partial-threshold-unimplemented");
    }
    const auto noncanonical = decode_rdna2_waitcnt(0xc0ffu);
    EXPECT_EQ(noncanonical.unmodeled_bits, 0x80u);
    EXPECT_FALSE(rdna2_waitcnt_effects_known(0xc0ffu));
    EXPECT_STREQ(rdna2_waitcnt_execution_gap(0xc0ffu),
                 "packet-waitcnt-noncanonical-bits-unimplemented");
}

TEST(FragmentPacketWaitcnt, CanonicalWaitEffectsPreserveOriginsButCannotSupplyEntryMasks) {
    for (uint32_t wait : {Lgkm0, Vm0, Exp0, NoDrain, 0xbf8cc17fu}) {
        const std::vector<uint32_t> code{wait,        0xf4200500u, 0xfa000010u, 0x7e020214u,
                                         0xf8001801u, 1u,          0xbf810000u};
        const auto ins = decode(code);
        const auto scalar = fragment_packet_scalar_read_requirements(code, ins);
        ASSERT_TRUE(scalar.rejection.empty()) << scalar.rejection;
        ASSERT_EQ(scalar.sites.size(), 1u);
        EXPECT_EQ(scalar.sites[0].pc, 1u);
        EXPECT_EQ(scalar.sites[0].entry_words, (std::array<uint32_t, 4>{0, 1, 2, 3}));
        const auto masks = fragment_packet_mask_requirements(code, ins);
        ASSERT_TRUE(masks.rejection.empty()) << masks.rejection;
        EXPECT_EQ(masks.demanded, kPacketInitialExec);
        EXPECT_EQ(masks.first_read_pc[0], 3u);
        uint32_t missing_pc = UINT32_MAX;
        EXPECT_STREQ(fragment_packet_missing_initial_mask(masks, 0, missing_pc),
                     "packet-entry-exec-unavailable");
        EXPECT_EQ(missing_pc, 3u);
    }
}

TEST(FragmentPacketWaitcnt, OriginalScalarAndImageSinksNeedTheirOwnZeroCounter) {
    auto input = resources::chain();
    input.invocation.guest_code[3] = Lgkm0;
    input.invocation.guest_code[input.images[0].pc + 2] = Vm0;
    raw_resource_sink(input, "independent_zero_counters");
    for (uint32_t wait : {Vm0, Exp0, NoDrain}) {
        auto wrong = input;
        wrong.invocation.guest_code[3] = wait;
        resource_refusal(wrong, "packet-smem-result-read-before-wait", 8u);
    }
    for (uint32_t wait : {Lgkm0, Exp0, NoDrain}) {
        auto wrong = input;
        wrong.invocation.guest_code[wrong.images[0].pc + 2] = wait;
        resource_refusal(wrong, "packet-image-result-read-before-wait", 17u);
    }
    auto wrong = input;
    wrong.invocation.guest_code[3] = 0xbf8cc17fu;
    resource_wait_refusal(wrong, "packet-waitcnt-partial-threshold-unimplemented", 3u);
    wrong = input;
    wrong.invocation.guest_code[3] = 0xbf8cc0ffu;
    resource_wait_refusal(wrong, "packet-waitcnt-noncanonical-bits-unimplemented", 3u);
}

TEST(FragmentPacketWaitcnt, ImageCompletionIsRequiredOnBothConditionalPredecessors) {
    for (bool scc : {false, true}) {
        auto input = resources::chain();
        input.invocation.scc = scc;
        input.invocation.guest_code[3] = Lgkm0;
        const auto wait_pc = input.images[0].pc + 2;
        input.invocation.guest_code[wait_pc] = 0xbf840002u;
        input.invocation.guest_code.insert(input.invocation.guest_code.begin() + wait_pc + 1,
                                           {Vm0, 0xbf820001u, Vm0});
        raw_resource_sink(input, scc ? "both_image_paths_one" : "both_image_paths_zero");
        input.invocation.guest_code[wait_pc + 3] = Lgkm0;
        resource_refusal(input, "packet-image-result-read-before-wait", 20u);
    }
}

TEST(FragmentPacketWaitcnt, ScalarCompletionIsRequiredOnBothConditionalPredecessors) {
    for (bool scc : {false, true}) {
        auto input = resources::chain();
        input.invocation.scc = scc;
        input.invocation.guest_code[3] = 0xbf840002u;
        input.invocation.guest_code.insert(input.invocation.guest_code.begin() + 4,
                                           {Lgkm0, 0xbf820001u, Lgkm0});
        input.images[0].pc += 3;
        input.invocation.guest_code[input.images[0].pc + 2] = Vm0;
        raw_resource_sink(input, scc ? "both_scalar_paths_one" : "both_scalar_paths_zero");
        input.invocation.guest_code[6] = Vm0;
        resource_refusal(input, "packet-smem-result-read-before-wait", 11u);
    }
}

TEST(FragmentPacketWaitcnt, SimultaneousScalarAndImageResultsNeedBothIndependentDrains) {
    auto input = resources::chain();
    input.parameter_cache = {};
    input.buffers[0].pc = 0;
    input.images[0].pc = 2;
    input.invocation.vgprs[0].words.fill(0x3e000000u);
    input.invocation.vgprs[1].words.fill(0x3e000000u);
    // Original S_BUFFER_LOAD_DWORDX4 and SAMPLE_LZ are both outstanding BEFORE either WAIT.
    // The image coordinates use actual supplied VGPRs, not the still-pending scalar result.
    input.invocation.guest_code = {0xf4280400u, 0xfa000000u, 0xf09c0f08u, 0x00611400u,
                                   Lgkm0,       Vm0,         0x7e300210u, 0xf800180fu,
                                   0x17161514u, 0xf8001801u, 24u,         0xbf810000u};
    const auto check = [&](const FragmentResourcePacket& packet, const char* scenario) {
        const auto program = recompile_fragment_resource_packet(packet);
        ASSERT_FALSE(program.packet.spirv.empty()) << program.packet.rejection;
        retain(program.packet.spirv, scenario);
        bpermute_oracle::Interpreter vm(program.packet.spirv);
        ASSERT_EQ(program.images.size(), 1u);
        for (const auto& mip : program.images[0].mips)
            vm.sampled_images[16].push_back({mip.width, mip.height, mip.texels});
        const auto words = vm.run_packet(program.packet.input_words, program.packet.output_words);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        const auto result =
            decode_fragment_resource_packet(program, words, true, packet.device.device_identity);
        ASSERT_TRUE(result.rejection.empty()) << result.rejection;
        ASSERT_EQ(result.exports.size(), 64u * 24u);
        const uint32_t first[]{1, 1, 1,           0,           15,          0,
                               1, 1, 0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u};
        const uint32_t second[]{1, 1, 1, 0, 1, 0, 1, 1, 0x3f000000u, 0, 0, 0};
        for (uint32_t lane = 0; lane < 64u; ++lane) {
            SCOPED_TRACE(lane);
            for (uint32_t word = 0; word < 12u; ++word) {
                EXPECT_EQ(result.exports[lane * 24u + word], first[word]);
                EXPECT_EQ(result.exports[lane * 24u + 12u + word], second[word]);
            }
        }
    };
    check(input, "scalar_then_vector");
    std::swap(input.invocation.guest_code[4], input.invocation.guest_code[5]);
    check(input, "vector_then_scalar");
    input.invocation.guest_code[4] = input.invocation.guest_code[5] = Lgkm0;
    resource_refusal(input, "packet-image-result-read-before-wait", 7u);
    input.invocation.guest_code[4] = input.invocation.guest_code[5] = Vm0;
    resource_refusal(input, "packet-smem-result-read-before-wait", 6u);
}

TEST(FragmentPacketWaitcnt, ExportWordsAndExecAreDrainedOnlyByExpcntZero) {
    for (uint32_t family : {0u, 1u, 2u, 3u, 4u}) {
        auto input = exports::pending_write(family, true).invocation;
        input.guest_code[2] = Exp0;
        export_sink(input, exports::pending_records(family, 0),
                    "export_wait_" + std::to_string(family));
        for (uint32_t wait : {Lgkm0, Vm0, NoDrain}) {
            input.guest_code[2] = wait;
            export_refusal(input,
                           family == 0 ? "packet-export-source-overwrite-before-wait"
                                       : "packet-export-exec-overwrite-before-wait",
                           3u);
        }
    }
}

TEST(FragmentPacketWaitcnt, ExportCompletionCannotBeBorrowedAcrossConditionalArms) {
    for (bool scc : {false, true}) {
        auto input = exports::pending_join(true, scc).invocation;
        input.guest_code[3] = input.guest_code[5] = Exp0;
        const auto program = recompile_fragment_packet(input);
        ASSERT_FALSE(program.spirv.empty()) << program.rejection;
        retain(program.spirv, scc ? "both_export_paths_one" : "both_export_paths_zero");
        bpermute_oracle::Interpreter vm(program.spirv);
        const auto words = vm.run_packet(program.input_words, program.output_words);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        const auto result = decode_fragment_packet(program, words, true);
        ASSERT_TRUE(result.rejection.empty()) << result.rejection;
        ASSERT_EQ(result.architectural_exports.size(), 64u);
        for (uint32_t lane : {0u, 31u, 32u, 40u, 63u}) {
            const auto& events = result.architectural_exports[lane].events;
            ASSERT_EQ(events.size(), 2u);
            ASSERT_TRUE(events[0].source_words[0].has_value());
            ASSERT_TRUE(events[1].source_words[0].has_value());
            EXPECT_EQ(*events[0].source_words[0], 0x87000000u + lane);
            EXPECT_EQ(*events[1].source_words[0], exports::value(0));
        }
        input.guest_code[5] = Lgkm0;
        export_refusal(input, "packet-export-source-overwrite-before-wait", 6u);
    }
}

TEST(FragmentPacketWaitcnt, ExportExecLifetimeMustBeDrainedOnEveryConditionalPredecessor) {
    for (bool scc : {false, true}) {
        auto input = exports::pending_join(true, scc).invocation;
        input.guest_code[3] = input.guest_code[5] = Exp0;
        input.guest_code[6] =
            0xbefe0414u;   // same-valued genuine EXEC replacement STILL writes EXEC
        const auto program = recompile_fragment_packet(input);
        ASSERT_FALSE(program.spirv.empty()) << program.rejection;
        retain(program.spirv, scc ? "both_exec_paths_one" : "both_exec_paths_zero");
        bpermute_oracle::Interpreter vm(program.spirv);
        const auto words = vm.run_packet(program.input_words, program.output_words);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        const auto result = decode_fragment_packet(program, words, true);
        ASSERT_TRUE(result.rejection.empty()) << result.rejection;
        ASSERT_EQ(result.architectural_exports.size(), 64u);
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const auto& events = result.architectural_exports[lane].events;
            ASSERT_EQ(events.size(), 2u);
            for (const auto& event : events) {
                EXPECT_EQ(event.exec, exports::active(lane));
                EXPECT_EQ(event.source_words[0].has_value(), exports::active(lane));
                if (exports::active(lane)) {
                    ASSERT_TRUE(event.source_words[0].has_value());
                    EXPECT_EQ(*event.source_words[0], 0x87000000u + lane);
                }
            }
        }
        input.guest_code[5] = Vm0;
        export_refusal(input, "packet-export-exec-overwrite-before-wait", 6u);
    }
}

TEST(FragmentPacketWaitcnt, ArchitecturalWaitAdmissionDoesNotWidenDefaultRawPackets) {
    auto input = exports::scratch().invocation;
    input.guest_code.insert(input.guest_code.begin() + 1, NoDrain);
    // No operation precedes this maximum-threshold WAIT with an outstanding result. It grants
    // no completion, but need not reject a safe Architectural original program.
    const auto program = recompile_fragment_packet(input);
    ASSERT_FALSE(program.spirv.empty()) << program.rejection;
    retain(program.spirv, "no_outstanding_results");
    bpermute_oracle::Interpreter vm(program.spirv);
    const auto words = vm.run_packet(program.input_words, program.output_words);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    const auto decoded = decode_fragment_packet(program, words, true);
    ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
    ASSERT_EQ(decoded.architectural_exports.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto& events = decoded.architectural_exports[lane].events;
        ASSERT_EQ(events.size(), 1u);
        EXPECT_EQ(events[0].site.pc, 2u);
        EXPECT_EQ(events[0].source_words[0].has_value(), exports::active(lane));
        if (exports::active(lane)) {
            ASSERT_TRUE(events[0].source_words[0].has_value());
            EXPECT_EQ(*events[0].source_words[0], exports::value(0));
        }
    }
    for (uint32_t wait : {NoDrain, Exp0}) {
        input.guest_code[1] = wait;
        auto raw = input;
        raw.export_observation = FragmentPacketExportObservation::LegacyRaw;
        const auto refused = recompile_fragment_packet(raw);
        EXPECT_TRUE(refused.spirv.empty());
        EXPECT_EQ(refused.rejection, "packet-control-unimplemented");
    }
    input.guest_code[1] = 0xbf8cff1fu;
    export_refusal(input, "packet-waitcnt-partial-threshold-unimplemented", 1u);
    input.guest_code[1] = 0xbf8cff8fu;
    export_refusal(input, "packet-waitcnt-noncanonical-bits-unimplemented", 1u);
}
}   // namespace
