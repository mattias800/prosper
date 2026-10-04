// Genuine original instructions and emitted integer sinks; no native subgroup/device execution.
#include "gpu/recompiler/fragment_packet_quad_swizzle.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {
using namespace prosper::gpu;
FragmentInvocationPacket packet(uint32_t wait = 0xbf8cc07fu, bool wqm = true) {
    FragmentInvocationPacket input;
    input.slots_available.fill(true);
    input.exec_available = true;
    input.exec_mask = 0x8421842184218421ull;
    input.export_observation = FragmentPacketExportObservation::Architectural;
    input.quad_topology = FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
    input.export_enabled.fill(1);
    FragmentPacketVgpr source;
    source.reg = 0;
    source.available_mask = UINT64_MAX;
    for (uint32_t lane = 0; lane < 64; ++lane) source.words[lane] = 0x41230000u + 17u * lane;
    input.vgprs.push_back(source);
    input.guest_code = {0xbe94047eu}; // save the actual sparse initial EXEC
    if (wqm) input.guest_code.push_back(0xbefe0a7eu);
    // Every output selects quad lane3, VDST==ADDR: all old peer values must be snapshotted first.
    input.guest_code.insert(input.guest_code.end(), {0xd8d480ffu, 0x00000000u, wait, 0xbefe0414u,
                                                     0xf800180fu, 0x00000000u, 0xbf810000u});
    return input;
}
void retain(const FragmentPacketProgram& program, const char* name) {
    const auto* root = std::getenv("PROSPER_PACKET_QUAD_SWIZZLE_SPV_DIRECTORY");
    if (!root || !*root) return;
    std::error_code error;
    std::filesystem::create_directories(root, error);
    ASSERT_FALSE(error) << error.message();
    std::ofstream file(std::filesystem::path(root) / (std::string(name) + ".spv"),
                       std::ios::binary);
    file.write(reinterpret_cast<const char*>(program.spirv.data()),
               std::streamsize(program.spirv.size() * sizeof(uint32_t)));
    file.close();
    ASSERT_TRUE(bool(file));
}
void evaluate(const FragmentInvocationPacket& input, const char* name, bool absent = false) {
    const auto program = recompile_fragment_packet(input);
    ASSERT_FALSE(program.spirv.empty()) << program.rejection;
    ASSERT_NE(program.vgpr_status_offset, UINT32_MAX);
    retain(program, name);
    for (size_t pc = 5; pc < program.spirv.size();) {
        const auto count = program.spirv[pc] >> 16, op = program.spirv[pc] & 0xffffu;
        ASSERT_NE(count, 0u);
        ASSERT_LE(count, program.spirv.size() - pc);
        EXPECT_NE(op, 345u) << "owned logical64 must not use native subgroup shuffle";
        pc += count;
    }
    bpermute_oracle::Interpreter vm(program.spirv);
    const auto words = vm.run_packet(program.input_words, program.output_words);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    const auto result = decode_fragment_packet(program, words, true);
    if (absent) {
        EXPECT_EQ(result.rejection, "packet-quad-selected-source-exec-or-value-unavailable");
        EXPECT_EQ(result.kind, 5u);
        EXPECT_TRUE(result.architectural_exports.empty());
        return;
    }
    ASSERT_TRUE(result.rejection.empty()) << result.rejection;
    ASSERT_EQ(result.architectural_exports.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto& output = result.architectural_exports[lane];
        const bool live = (0x8421842184218421ull >> lane) & 1u;
        ASSERT_EQ(output.events.size(), 1u);
        EXPECT_EQ(output.events[0].exec, live);
        EXPECT_EQ(output.valid_mask, live);
        EXPECT_EQ(output.commit_eligible, live);
        for (uint32_t channel = 0; channel < 4; ++channel) {
            EXPECT_EQ(output.events[0].source_words[channel].has_value(), live);
            if (live) {
                ASSERT_TRUE(output.events[0].source_words[channel].has_value());
                EXPECT_EQ(*output.events[0].source_words[channel],
                          0x41230000u + 17u * ((lane & ~3u) | 3u));
            }
        }
    }
}
}   // namespace

TEST(PacketQuadSwizzle, SavedExecWqmGenuineHelpersAliasAndUpper32ProduceOriginalWords) {
    reset_float_controls_support_for_test();
    evaluate(packet(), "saved_exec_wqm_quad_helper_alias");
}
TEST(PacketQuadSwizzle, MissingHelperAndInactiveSelectedSourceNeverBecomeZeroAuthority) {
    reset_float_controls_support_for_test();
    auto missing = packet();
    missing.vgprs[0].available_mask &= ~(uint64_t{1} << 63);
    evaluate(missing, "absent_helper63", true);
    evaluate(packet(0xbf8cc07fu, false), "inactive_selected_source", true);
}
TEST(PacketQuadSwizzle, CounterSpecificCompletionCannotBeBorrowedFromVmOrExport) {
    const auto safe = recompile_fragment_packet(packet());
    ASSERT_FALSE(safe.spirv.empty()) << safe.rejection;
    for (uint32_t wait : {0xbf8c3f70u, 0xbf8cff0fu, 0xbf8cff7fu}) {
        const auto refused = recompile_fragment_packet(packet(wait));
        EXPECT_TRUE(refused.spirv.empty());
        EXPECT_EQ(refused.rejection, "packet-quad-swizzle-result-read-before-lgkm-wait:pc=6");
    }
}
TEST(PacketQuadSwizzle, AllPathWaitAndOverwriteRemainSeparateLoadBearingGuards) {
    auto input = packet();
    // Conditional branch bypasses only the genuine LGKM wait. Both structural arms matter even
    // with an actual SCC value choosing the safe arm in this original submission.
    input.scc_available = true;
    input.scc = true;
    input.guest_code.insert(input.guest_code.begin() + 4, 0xbf840001u);
    const auto bypass = recompile_fragment_packet(input);
    EXPECT_TRUE(bypass.spirv.empty());
    EXPECT_EQ(bypass.rejection, "packet-quad-swizzle-result-read-before-lgkm-wait:pc=7");
    input = packet();
    input.guest_code.insert(input.guest_code.begin() + 4,
                            0x7e000280u);   // overwrite v0 before wait
    const auto overwrite = recompile_fragment_packet(input);
    EXPECT_TRUE(overwrite.spirv.empty());
    EXPECT_EQ(overwrite.rejection, "packet-quad-swizzle-result-overwrite-before-lgkm-wait:pc=4");
}
TEST(PacketQuadSwizzle, TopologyModesAndLegacyPolicyRemainExplicitNotBroadDsAdmission) {
    auto input = packet();
    input.quad_topology = FragmentPacketQuadTopology::Unknown;
    EXPECT_EQ(recompile_fragment_packet(input).rejection, "packet-quad-topology-unavailable:pc=2");
    input = packet();
    input.export_observation = FragmentPacketExportObservation::LegacyRaw;
    EXPECT_EQ(recompile_fragment_packet(input).rejection, "packet-ds-op-unimplemented:pc=2");
    for (uint32_t bits : {0u, 0x10000u, 0x20000u, 0x4000u, 0x6000u}) {
        input = packet();
        input.guest_code[2] = 0xd8d40000u | bits;
        const auto refused = recompile_fragment_packet(input);
        EXPECT_TRUE(refused.spirv.empty());
        EXPECT_FALSE(refused.rejection.empty());
    }
}
