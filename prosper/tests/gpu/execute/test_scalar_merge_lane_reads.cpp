// Lane reads define one scalar word even when their decoded primary destination names a VGPR.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "fixtures/compute_runner.h"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>
namespace {
using namespace prosper::gpu;
using Words = std::vector<uint32_t>;
constexpr uint32_t end = 0xbf810000u, provided = 0x3c003c00u;
constexpr uint64_t address = 0x5521cc9300ull;
enum class Mutation { None, MissingSpill, OldSelector, LiveHigh };
Words program(Mutation mutation) {
    const bool missing = mutation == Mutation::MissingSpill;
    Words words{0xbf068014u,
                0xbf840003u,
                0xbe980389u,
                missing ? 0xd761003cu : 0xbf800000u,
                missing ? (42u | (128u << 9)) : 0xbf800000u,
                0xcc0e4004u,
                0x1c0c5501u,
                missing ? 0xbf800000u : 0xd761003cu,
                missing ? 0xbf800000u : (42u | (128u << 9)),
                0xd7600018u,
                316u | ((mutation == Mutation::OldSelector ? 24u : 128u) << 9),
                mutation == Mutation::LiveHigh ? 0x7e000219u : 0x7e000218u,
                end};
    return words;
}
std::vector<Rdna2Inst> decode(const Words& words) {
    std::vector<Rdna2Inst> result;
    EXPECT_EQ(rdna2_walk(words.data(), words.size(), result), words.size());
    return result;
}
Words compile(bool native, Mutation mutation = Mutation::None) {
    return recompile_with_arbitrary_placeholders({RecompileDiagnosticStage::Compute, address}, [&] {
        SpirvCompute b;
        b.begin(4, nullptr, 64, 1, 1, 64, 21, true, true);
        b.diagnostic = {RecompileDiagnosticStage::Compute, address};
        b.native_subgroup_size = native ? 64 : 0;
        ComputeShaderConfig config;
        config.local_x = config.wave_size = 64;
        config.native_subgroup_size = native ? 64 : 0;
        config.user_sgprs.resize(21);
        prosper::test::complete_compute_launch(config);
        prosper::test::bind_checked_compute_launch(b, config);
        RegState rs;
        rs.exec = b.btrue();
        rs.vcc = rs.scc = b.bfalse();
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.sreg[20] = b.load_push_constant(20);
        rs.sreg[42] = b.uconst(provided);
        rs.sreg_bool[24] = constant_mask_bit(b, 0x12345678u, 0xabcdef01u);
        rs.vreg[1] = b.uconst(0x40004000u);
        rs.vreg[3] = b.uconst(0x38003800u);
        const auto words = program(mutation);
        if (!emit_body(
                b, rs, decode(words), {}, nullptr, true, false,
                [](RegState&, const Rdna2Inst&) { return true; }, words.data(), words.size()))
            return Words{};
        b.store_output_word(rs.vreg.at(0), 2, 0);
        b.store_output_word(rs.vreg.at(4), 2, 1);
        return b.finish();
    });
}
Words expected() {
    Words result;
    for (unsigned lane = 0; lane < 64; ++lane) {
        result.push_back(provided);
        result.push_back(0x41004100u);
    }
    return result;
}
void gpu(bool native) {
    const auto module = compile(native);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    for (uint32_t branch : {0u, 1u}) {
        Words push(21);
        push[20] = branch;
        const auto floats =
            prosper::test::run_compute(module, {0.0f}, 64, 128, {}, {}, nullptr, 64, nullptr,
                                       nullptr, nullptr, native ? 64 : 0, nullptr, push);
        Words actual;
        for (float value : floats) actual.push_back(std::bit_cast<uint32_t>(value));
        EXPECT_EQ(actual, expected());
    }
}
}   // namespace
TEST(ScalarMergeLaneReads, KnownLaneDefinitionsKillTheirActualScalarWord) {
    for (const auto& words : {Words{0xd7600018u, 316u | (128u << 9), 0x7e000218u, end},
                              Words{0x7e300501u, 0x7e000218u, end}}) {
        const auto ins = decode(words);
        ASSERT_FALSE(ins.empty());
        EXPECT_TRUE(rdna2_lane_read_writes_sgpr(ins.front()));
        EXPECT_EQ(scalar_write_width(ins.front()), 1u);
        EXPECT_TRUE(sgpr_dead_at_merge(ins, 0, 24));
    }
}
TEST(ScalarMergeLaneReads, RealSpillReplacementExecutesBothArmsInTheInterpreter) {
    for (bool native : {false, true})
        for (uint32_t branch : {0u, 1u}) {
            SCOPED_TRACE(native);
            SCOPED_TRACE(branch);
            const auto module = compile(native);
            ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
            bpermute_oracle::Interpreter vm(module);
            vm.native_subgroup_width = 64;
            vm.half_rte_preserve_denorm = true;
            vm.supplied_push_constants = Words(21);
            vm.supplied_push_constants[20] = branch;
            const auto actual =
                vm.run_buffers(64, {{0, Words(64)}, {1, Words(128, 0xdeadbeefu)}}, 1);
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            EXPECT_EQ(actual, expected());
        }
}
TEST(ScalarMergeLaneReads, MissingSpillCannotSupplyTheReplacement) {
    for (bool native : {false, true}) EXPECT_TRUE(compile(native, Mutation::MissingSpill).empty());
}
TEST(ScalarMergeLaneReads, SelectorReadsTheOldDestinationBeforeReplacingIt) {
    for (bool native : {false, true}) EXPECT_TRUE(compile(native, Mutation::OldSelector).empty());
}
TEST(ScalarMergeLaneReads, OneWordReplacementCannotRetireLiveHigh) {
    for (bool native : {false, true}) EXPECT_TRUE(compile(native, Mutation::LiveHigh).empty());
}
TEST(ScalarMergeLaneReads, AnOrdinaryVgprMoveDoesNotKillTheScalarWord) {
    const auto ins = decode({0x7e300281u, 0x7e000218u, end});
    EXPECT_FALSE(rdna2_lane_read_writes_sgpr(ins.front()));
    EXPECT_FALSE(sgpr_dead_at_merge(ins, 0, 24));
}
TEST(ScalarMergeLaneReads, PortableWave64ExecutesOnTheGpu) {
    gpu(false);
}
TEST(ScalarMergeLaneReads, RequiredWave64ExecutesOnTheGpu) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 unavailable";
    gpu(true);
}
