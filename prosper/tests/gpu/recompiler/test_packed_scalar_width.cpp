// #5497: packed operands occupy one scalar word. Charging their unwritten sibling
// poisons genuine arithmetic and makes the following fresh predicate refuse.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;

std::vector<uint32_t> program() {
    return {
        0xbeaa03ffu, 0x3c003c00u, // s42 = packed (1, 1); s43 is never written
        0x7e0202ffu, 0x40004000u, // v1 = packed (2, 2)
        0x7e0602ffu, 0x38003800u, // v3 = packed (0.5, 0.5)
        0xcc0e4004u, 0x1c0c5501u, // v4 = pk_fma(v1, s42, v3)
        0x7d840904u,              // VCC = (v4 == v4); reads the packed result
        0x7e0c0281u,              // v6 = 1
        0x020a0c80u,              // v5 = VCC ? v6 : 0
        0xe0742000u, 0x80020400u, // output[lane] = v[4:5]
        0xbf810000u,
    };
}

ComputeShaderConfig config() {
    ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = 64;
    c.user_sgprs.resize(16, 0);
    prosper::test::complete_compute_launch(c);
    return c;
}

ShaderResourceTable resources() {
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 8;
    output.sgpr_base = 8;
    ShaderResourceTable table;
    table.resources.push_back(output);
    return table;
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address) {
    const auto table = resources();
    return recompile_compute(code.data(), code.size(), &table, config(),
                             {RecompileDiagnosticStage::Compute, address});
}
} // namespace

TEST(PackedScalarWidth, FreshPredicateAndPackedValuesExecuteWithoutAnUnusedSibling) {
    const uint64_t address = 0x5497500100ull;
    const auto module = compile(program(), address);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.half_rte_preserve_denorm = true;
    vm.push_constants = config().user_sgprs;
    const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(128, 0xdeadbeefu)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 128u);
    for (std::size_t lane = 0; lane < 64; ++lane) {
        EXPECT_EQ(out[lane * 2], 0x41004100u) << "packed 2.5, lane " << lane;
        EXPECT_EQ(out[lane * 2 + 1], 1u) << "fresh predicate, lane " << lane;
    }
}

TEST(PackedScalarWidth, AnActuallyMissingPackedScalarStillRefuses) {
    auto code = program();
    code[0] = 0xbeaa0310u; // s42 = possible unsupplied scratch offset s16
    code[1] = 0xbf800000u; // retain the original two-word producer footprint
    const uint64_t address = 0x5497500200ull;
    EXPECT_TRUE(compile(code, address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("fabricated-fresh-mask-source pc=8"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

TEST(PackedScalarWidth, AMissingLaneSpillCannotSupplyThePackedSource) {
    auto code = program();
    code[0] = 0xd760002au; // s42 = readlane(v60, 0), without a matching spill
    code[1] = 316u | (128u << 9u);
    const uint64_t address = 0x5497500300ull;
    EXPECT_TRUE(compile(code, address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("mode=unresolved-operand pc=0"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

TEST(PackedScalarWidth, EveryKnownPackedSourceUsesTheCanonicalPhysicalWidth) {
    Rdna2Inst in;
    in.fmt = Rdna2Format::VOP3P;
    uint32_t checked = 0;
    for (uint32_t opcode = 0; opcode < 128; ++opcode) {
        in.opcode = opcode;
        const auto& effects = rdna2_instruction_effects(in);
        if (!effects.describes_sources()) continue;
        for (uint32_t source = 0; source < 3; ++source) {
            if (!effects.ssrc_dwords[source]) continue;
            EXPECT_EQ(scalar_source_mark_words(in, source, 64), 1u) << opcode << '/' << source;
            EXPECT_EQ(scalar_source_mark_words(in, source, 32), 1u) << opcode << '/' << source;
            ++checked;
        }
    }
    EXPECT_GT(checked, 50u);
    in.opcode = 0x7f; // unclassified forms retain the conservative pair
    EXPECT_EQ(scalar_source_mark_words(in, 0, 64), 2u);
}

namespace {
std::vector<uint32_t> mask_program() {
    return {
        0x7e0602ffu, 0x00030003u, // v3 = packed integer (3, 3)
        0x7d8360f9u, 0x86069400u, // s[20:21] = v0 < 48
        0x7d8600f9u, 0x06869690u, // s[22:23] = 16 <= v0
        0x87941614u,              // Bool-only AND: lanes 16..47
        0xcc094004u, 0x0c0d2004u, // v4 = packed_mad(s4, 16, v3); no mask DATA read
        0x7e0a0281u,              // v5 = 1
        0x7d840904u,              // fresh predicate reads packed result
        0x020a0a80u,              // v5 = VCC ? 1 : 0
        0xd5010006u, 0x00520a80u, // v6 = saved-mask s[20:21] ? v5 : 0
        0xe0742000u, 0x80020400u, // output[lane].xy = v[4:5]
        0xe0702008u, 0x80020600u, // output[lane].z = v6
        0xbf810000u,
    };
}

ComputeShaderConfig mask_config(bool dispatcher) {
    ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = 64;
    c.user_sgprs.resize(16, 0);
    c.user_sgprs[4] = 1;
    c.exact_thread_extent = true;
    c.threads_x = 48;
    c.threads_y = c.threads_z = 1;
    c.force_exchange_dispatcher = dispatcher;
    return c;
}

std::vector<uint32_t> mask_module(const std::vector<uint32_t>& code, uint64_t address,
                                  bool dispatcher) {
    auto c = mask_config(dispatcher);
    auto table = resources();
    table.resources[0].stride = 12;
    return recompile_compute(code.data(), code.size(), &table, c,
                             {RecompileDiagnosticStage::Compute, address});
}

void expect_mask_words(bool dispatcher) {
    const uint64_t address = dispatcher ? 0x5497500500ull : 0x5497500400ull;
    const auto module = mask_module(mask_program(), address, dispatcher);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = mask_config(dispatcher).user_sgprs;
    const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(192, 0xdeadbeefu)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 192u);
    for (std::size_t lane = 0; lane < 64; ++lane) {
        EXPECT_EQ(out[lane * 3], lane < 48 ? 0x00030013u : 0xdeadbeefu) << lane;
        EXPECT_EQ(out[lane * 3 + 1], lane < 48 ? 1u : 0xdeadbeefu) << lane;
        EXPECT_EQ(out[lane * 3 + 2], lane < 48 ? (lane >= 16 ? 1u : 0u) : 0xdeadbeefu) << lane;
    }
}
} // namespace

TEST(PackedScalarWidth, NumericVccHighDoesNotReadTheUnwrittenTtmpBesideIt) {
    auto code = program();
    code[0] = 0xbeeb03ffu;       // VCC_HI = packed (1, 1); TTMP0 stays absent
    code[7] = 0x1c0cd701u;      // packed FMA reads VCC_HI in place of s42
    code.insert(code.begin() + 2, 0xbeec033cu); // unrelated TTMP0 = missing s60
    const uint64_t address = 0x5497500600ull;
    const auto module = compile(code, address);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.half_rte_preserve_denorm = true;
    vm.push_constants = config().user_sgprs;
    const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(128, 0xdeadbeefu)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 128u);
    for (std::size_t lane = 0; lane < 64; ++lane) {
        EXPECT_EQ(out[lane * 2], 0x41004100u) << lane;
        EXPECT_EQ(out[lane * 2 + 1], 1u) << lane;
    }
}

TEST(PackedScalarWidth, PartialLaunchBoolMaskSurvivesAnUnrelatedPackedRead) {
    expect_mask_words(false);
}

TEST(PackedScalarWidth, PartialLaunchDispatcherPreservesTheSameBoolMask) {
    expect_mask_words(true);
}

TEST(PackedScalarWidth, MissingPackedHighDataCannotBorrowAnotherMask) {
    auto code = mask_program();
    code[8] = (code[8] & ~0x1ffu) | 25u; // s25 is not part of either genuine mask
    const uint64_t address = 0x5497500700ull;
    EXPECT_TRUE(mask_module(code, address, false).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("fabricated-fresh-mask-source pc=10"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

TEST(PackedScalarWidth, MissingNumericVccHighStillCannotSupplyPackedData) {
    auto code = program();
    code[0] = 0xbeeb0310u; // VCC_HI = possible unsupplied scratch offset s16
    code[1] = 0xbf800000u;
    code[7] = 0x1c0cd701u;
    const uint64_t address = 0x5497500800ull;
    EXPECT_TRUE(compile(code, address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("fabricated-fresh-mask-source pc=8"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}
