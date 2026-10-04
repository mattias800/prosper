// Project-owned composition inputs and original integer PS sinks. This is a reference-model
// regression, not live host-coverage -> guest EXEC authority or AMD hardware-packing evidence.
#include "gpu/recompiler/fragment_quad_composition.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {
using namespace prosper::gpu;
FragmentQuadCompositionInput quad(uint32_t record, int32_t x, int32_t y, uint32_t primitive = 10,
                                  uint8_t live = 1, uint8_t export_candidates = 1) {
    return {record, primitive, x, y, 0, 0, 0, 15, live, export_candidates};
}
std::vector<FragmentQuadCompositionInput> grid(uint32_t count) {
    std::vector<FragmentQuadCompositionInput> result;
    for (uint32_t index = 0; index < count; ++index)
        result.push_back(quad(index, int32_t((index % 4) * 2), int32_t((index / 4) * 2), 10,
                              uint8_t(1u << (index % 4)), index == 5 ? 0 : 15));
    return result;
}
void retain(const FragmentPacketProgram& program, const char* name) {
    const auto* root = std::getenv("PROSPER_FRAGMENT_COMPOSITION_SPV_DIRECTORY");
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
} // namespace

TEST(FragmentQuadComposition, PartialLiveQuadsKeepGenuineHelperBackingAndIndependentExports) {
    const auto result = fragment_quad_composition_reference(grid(16), {});
    ASSERT_TRUE(result.rejection.empty()) << result.rejection;
    ASSERT_EQ(result.waves.size(), 1u);
    const auto& wave = result.waves[0];
    EXPECT_EQ(wave.quad_count, 16u);
    EXPECT_EQ(wave.backing, UINT64_MAX);
    EXPECT_EQ(wave.live, 0x8421842184218421ull);
    EXPECT_EQ(wave.export_candidates, 0xffffffffff0fffffull);
    EXPECT_EQ(wave.primitive_transitions, 0u);
    for (uint32_t index = 0; index < 16; ++index) {
        EXPECT_EQ(wave.records[index], index);
        EXPECT_EQ(wave.primitives[index], 10u);
    }
}
TEST(FragmentQuadComposition, CollectorAppendOrderCannotSelectLogicalLanes) {
    auto inputs = grid(33);
    const auto first = fragment_quad_composition_reference(inputs, {});
    std::reverse(inputs.begin(), inputs.end());
    const auto second = fragment_quad_composition_reference(inputs, {});
    ASSERT_TRUE(first.rejection.empty());
    ASSERT_TRUE(second.rejection.empty());
    ASSERT_EQ(first.waves.size(), 3u);
    ASSERT_EQ(second.waves.size(), first.waves.size());
    for (size_t wave = 0; wave < first.waves.size(); ++wave) {
        EXPECT_EQ(first.waves[wave].records, second.waves[wave].records);
        EXPECT_EQ(first.waves[wave].live, second.waves[wave].live);
        EXPECT_EQ(first.waves[wave].backing, second.waves[wave].backing);
    }
    const auto& last = first.waves[2];
    EXPECT_EQ(last.quad_count, 1u);
    EXPECT_EQ(last.backing, 15u);
    EXPECT_EQ(last.live, 1u);
    EXPECT_EQ(last.records[0], 32u);
    for (uint32_t index = 1; index < 16; ++index) {
        EXPECT_EQ(last.records[index], UINT32_MAX);
        EXPECT_EQ(last.primitives[index], UINT32_MAX);
    }
}
TEST(FragmentQuadComposition, EveryPhysicalWaveBreakKeepsQuadAlignedRegions) {
    for (uint32_t mode = 1; mode <= 3; ++mode) {
        const int32_t boundary = int32_t(4u << mode);
        const std::vector input{quad(0, -2, 0), quad(1, 0, 0), quad(2, boundary - 2, 0),
                                quad(3, boundary, 0)};
        const auto result = fragment_quad_composition_reference(input, {mode});
        ASSERT_TRUE(result.rejection.empty());
        ASSERT_EQ(result.waves.size(), 3u);
        EXPECT_EQ(result.waves[0].records[0], 0u); // floor, not signed truncation toward zero
        EXPECT_EQ(result.waves[1].records[0], 1u);
        EXPECT_EQ(result.waves[1].records[1], 2u);
        EXPECT_EQ(result.waves[2].records[0], 3u);
    }
    const std::vector input{quad(0, -2, 0), quad(1, 0, 0), quad(2, 32, 0)};
    const auto no_break = fragment_quad_composition_reference(input, {0});
    ASSERT_TRUE(no_break.rejection.empty());
    ASSERT_EQ(no_break.waves.size(), 1u);
    EXPECT_EQ(no_break.waves[0].quad_count, 3u);
    EXPECT_EQ(fragment_quad_composition_reference(input, {4}).rejection,
              "fragment-quad-wave-break-encoding-unimplemented");
}
TEST(FragmentQuadComposition, PrimitiveTransitionsAndSystemWordDoNotInventBcOrM0) {
    const std::vector input{quad(0, 0, 0, 20), quad(1, 2, 0, 10), quad(2, 4, 0, 10),
                            quad(3, 6, 0, 30)};
    const auto result = fragment_quad_composition_reference(input, {});
    ASSERT_TRUE(result.rejection.empty());
    ASSERT_EQ(result.waves.size(), 1u);
    const auto& wave = result.waves[0];
    EXPECT_EQ(wave.records[0], 1u);
    EXPECT_EQ(wave.records[1], 2u);
    EXPECT_EQ(wave.records[2], 0u);
    EXPECT_EQ(wave.records[3], 3u);
    EXPECT_EQ(wave.primitive_transitions, 6u); // Q2 and Q3, Q0 implicit
    EXPECT_FALSE(fragment_quad_system_word(wave, 0x1234, {}).has_value());
    const auto ordinary = fragment_quad_system_word(wave, 0x1234, false);
    const auto optimized = fragment_quad_system_word(wave, 0x1234, true);
    ASSERT_TRUE(ordinary.has_value());
    ASSERT_TRUE(optimized.has_value());
    EXPECT_EQ(*ordinary, 0x00061234u);
    EXPECT_EQ(*optimized, 0x80061234u);
    EXPECT_EQ(*optimized & 0x7fffffffu, *ordinary)
        << "only the ORIGINAL guest clearing BC may turn its system word into M0";
}
TEST(FragmentQuadComposition, RealCollisionsSpillRatherThanDropOrInventCollisionIds) {
    const std::vector input{quad(0, 0, 0, 10), quad(1, 2, 0, 10), quad(2, 0, 0, 20),
                            quad(3, 4, 0, 20)};
    const auto result = fragment_quad_composition_reference(input, {});
    ASSERT_TRUE(result.rejection.empty());
    ASSERT_EQ(result.waves.size(), 2u);
    EXPECT_EQ(result.waves[0].quad_count, 2u);
    EXPECT_EQ(result.waves[1].quad_count, 2u);
    EXPECT_EQ(result.waves[0].records[0], 0u);
    EXPECT_EQ(result.waves[0].records[1], 1u);
    EXPECT_EQ(result.waves[1].records[0], 2u);
    EXPECT_EQ(result.waves[1].records[1], 3u);
}
TEST(FragmentQuadComposition, DifferentSampleLayerAndViewNeverShareAQuadWave) {
    auto input = grid(4);
    input[1].sample = 1;
    input[2].layer = 1;
    input[3].view = 1;
    const auto result = fragment_quad_composition_reference(input, {});
    ASSERT_TRUE(result.rejection.empty());
    ASSERT_EQ(result.waves.size(), 4u);
    for (const auto& wave : result.waves) EXPECT_EQ(wave.quad_count, 1u);
}
TEST(FragmentQuadComposition, MalformedNormalizedInputNeverPublishesAPartialWave) {
    auto input = grid(17);
    for (uint32_t mutation = 0; mutation < 6; ++mutation) {
        auto bad = input;
        switch (mutation) {
            case 0: bad.back().backing = 7; break;
            case 1: bad.back().live = 16; break;
            case 2: bad.back().export_candidates = 16; break;
            case 3: bad.back().live = 0; break;
            case 4: bad.back().x = 1; break;
            case 5: bad.back().record = bad.front().record; break;
        }
        const auto result = fragment_quad_composition_reference(bad, {});
        EXPECT_FALSE(result.rejection.empty());
        EXPECT_TRUE(result.waves.empty());
    }
    input.back().x = input.front().x;
    input.back().y = input.front().y;
    const auto duplicate = fragment_quad_composition_reference(input, {});
    EXPECT_EQ(duplicate.rejection, "fragment-quad-normalized-origin-duplicate");
    EXPECT_TRUE(duplicate.waves.empty());
}
TEST(FragmentQuadComposition, OriginalSavedExecWqmUsesHelpersThenRestoresLiveVm) {
    reset_float_controls_support_for_test();
    const auto composed = fragment_quad_composition_reference(grid(16), {});
    ASSERT_TRUE(composed.rejection.empty());
    ASSERT_EQ(composed.waves.size(), 1u);
    const auto& wave = composed.waves[0];
    FragmentInvocationPacket packet;
    packet.slots_available.fill(true); // physical execution workers, not 64 live pixels
    packet.exec_available = true;
    packet.exec_mask = wave.live;
    packet.quad_topology = FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
    packet.export_observation = FragmentPacketExportObservation::Architectural;
    for (uint32_t lane = 0; lane < 64; ++lane)
        packet.export_enabled[lane] = uint8_t((wave.export_candidates >> lane) & 1u);
    FragmentPacketVgpr actual;
    actual.reg = 0;
    actual.available_mask = wave.backing;
    for (uint32_t lane = 0; lane < 64; ++lane) actual.words[lane] = 0x73100000u + lane * 17u;
    packet.vgprs.push_back(actual);
    // Save the genuine incoming EXEC, expand complete logical quads, consume original helper
    // data, restore the ORIGINAL live mask, and read an actual now-inactive helper at lane60.
    // No incoming VCC/SCC, destination initial value, fullmask MOV or zero helper is supplied.
    packet.guest_code = {0xbe94047eu, 0xbefe0a7eu, 0x7e020300u, 0x7e040214u,
                         0x7e060215u, 0xbefe0414u, 0xd760001eu, 257u | (188u << 9),
                         0x7e08021eu, 0xf800180fu, 0x04030201u, 0xbf810000u};
    const auto program = recompile_fragment_packet(packet);
    ASSERT_FALSE(program.spirv.empty()) << program.rejection;
    retain(program, "partial_live_saved_exec_wqm_helper_restore");
    bpermute_oracle::Interpreter vm(program.spirv);
    const auto words = vm.run_packet(program.input_words, program.output_words);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    const auto result = decode_fragment_packet(program, words, true);
    ASSERT_TRUE(result.rejection.empty()) << result.rejection;
    ASSERT_EQ(result.architectural_exports.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto& out = result.architectural_exports[lane];
        const bool live = (0x8421842184218421ull >> lane) & 1u;
        ASSERT_EQ(out.events.size(), 1u);
        EXPECT_EQ(out.events[0].exec, live);
        EXPECT_EQ(out.valid_mask, live);
        EXPECT_EQ(out.commit_eligible, live && (lane < 20 || lane >= 24));
        for (uint32_t channel = 0; channel < 4; ++channel)
            EXPECT_EQ(out.events[0].source_words[channel].has_value(), live);
        if (live) {
            for (uint32_t channel = 0; channel < 4; ++channel)
                ASSERT_TRUE(out.events[0].source_words[channel].has_value());
            EXPECT_EQ(*out.events[0].source_words[0], 0x73100000u + lane * 17u);
            EXPECT_EQ(*out.events[0].source_words[1], 0x84218421u);
            EXPECT_EQ(*out.events[0].source_words[2], 0x84218421u);
            EXPECT_EQ(*out.events[0].source_words[3], 0x731003fcu)
                << "genuine EXEC-off helper60 was initialized by WQM, not fabricated at entry";
        }
    }
    auto absent = packet;
    absent.vgprs[0].available_mask &= ~(uint64_t{1} << 60);
    const auto bad = recompile_fragment_packet(absent);
    ASSERT_FALSE(bad.spirv.empty()) << bad.rejection;
    retain(bad, "missing_actual_helper60");
    bpermute_oracle::Interpreter bad_vm(bad.spirv);
    const auto bad_words = bad_vm.run_packet(bad.input_words, bad.output_words);
    ASSERT_TRUE(bad_vm.error.empty()) << bad_vm.error;
    const auto refused = decode_fragment_packet(bad, bad_words, true);
    EXPECT_FALSE(refused.rejection.empty());
    EXPECT_TRUE(refused.architectural_exports.empty());
}
