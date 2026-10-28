// A carry write replaces SAVEEXEC's numeric words and Bool mirror, including across a CFG edge.
// Returning a prior lane mask can choose the wrong lanes or publish the wrong numeric value.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/native_execution_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
using namespace prosper::gpu;
constexpr std::array<uint32_t, 3> carries{0x28, 0x29, 0x2a};
constexpr uint32_t replacement = 0x13579bdfu;
struct Options {
    uint32_t opcode = 0x28, half = 1, active = 40;
    bool output = false, split = false, replace = false, mask_read = false, literal_input = false;
};
std::vector<uint32_t> program(Options o) {
    std::vector<uint32_t> code{0xbe9404c1u}; // s[20:21]=-1
    if (o.active != 64) code.insert(code.end(), {0x7d880080u | o.active, 0xbefe046au});
    code.insert(code.end(),
                {0xbeea2414u, 0xbefe04c1u}); // save old EXEC in VCC, restore full EXEC
    if (o.literal_input) code.push_back(0xbeea04c1u); // genuine numeric VCC=-1 before the carry
    const uint32_t a = o.opcode == 0x28   ? (o.output ? 193u : 128u)
                       : o.opcode == 0x29 ? (o.output ? 128u : 129u)
                                          : (o.output ? 130u : 128u);
    const uint32_t c = o.opcode == 0x28   ? (o.output ? 129u : 128u)
                       : o.opcode == 0x29 ? (o.output ? 130u : 128u)
                                          : (o.output ? 128u : 129u);
    code.push_back(0x7e040200u | c); // v2 is the second arithmetic source
    code.push_back((o.opcode << 25u) | (3u << 17u) | (2u << 9u) | a);
    if (o.split) code.insert(code.end(), {0xbf068004u, 0xbf840001u, 0x7e080280u});
    if (o.replace) code.insert(code.end(), {0xbe8003ffu | ((106u + o.half) << 16u), replacement});
    if (o.mask_read)
        code.insert(code.end(), {0xd5010001u, 128u | (129u << 9u) | (106u << 18u)});
    else
        code.push_back(0x7e020200u | (106u + o.half));
    code.insert(code.end(), {0xe0702000u, 0x80020100u});
    if (o.split) {
        // The existing project-owned irreducible tail requires a real dispatcher, not an if fold.
        code.insert(code.end(), {0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u, 0x7d840100u,
                                 0xbf870001u, 0xbf82fffdu, 0x7e040d02u});
    }
    code.push_back(0xbf810000u);
    return code;
}
std::vector<uint32_t> compile(Options o) {
    const auto code = program(o);
    const auto resources = prosper::test::arbitrary_placeholder::table();
    auto config = prosper::test::native_execution_launch();
    config.user_sgprs.resize(12);
    return recompile_compute(code.data(), code.size(), &resources, config);
}
bool dispatcher(const std::vector<uint32_t>& module) {
    for (size_t at = 5; at < module.size();) {
        const uint32_t width = module[at] >> 16u;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 65535u) == 251) return true;
        at += width;
    }
    return false;
}
uint32_t expected(Options o) {
    return o.replace ? replacement : o.output ? (o.mask_read ? 1u : UINT32_MAX) : 0u;
}
void values(Options o) {
    SCOPED_TRACE(o.opcode);
    SCOPED_TRACE(o.half);
    SCOPED_TRACE(o.active);
    SCOPED_TRACE(o.output);
    const auto module = compile(o);
    ASSERT_FALSE(module.empty());
    if (o.split) ASSERT_TRUE(dispatcher(module));
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants.resize(12);
    const auto result = vm.run(64, std::vector<uint32_t>(64, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(result, std::vector<uint32_t>(64, expected(o)));
}
}   // namespace

TEST(NativeVccCarryReplacement, EveryImplicitCarryReplacesBothNumericWords) {
    for (uint32_t op : carries)
        for (bool output : {false, true})
            for (uint32_t half : {0u, 1u})
                for (uint32_t active : {40u, 64u})
                    values({.opcode = op, .half = half, .active = active, .output = output});
}
TEST(NativeVccCarryReplacement, DispatcherReloadCannotResurrectTheSavedWords) {
    for (uint32_t op : carries)
        for (bool output : {false, true})
            for (uint32_t half : {0u, 1u})
                values({.opcode = op, .half = half, .output = output, .split = true});
}
TEST(NativeVccCarryReplacement, LiteralDataCannotSurviveTheImplicitMaskWrite) {
    for (uint32_t op : carries)
        for (bool output : {false, true})
            for (uint32_t half : {0u, 1u})
                for (bool split : {false, true})
                    values({.opcode = op,
                            .half = half,
                            .output = output,
                            .split = split,
                            .literal_input = true});
}
TEST(NativeVccCarryReplacement, ExplicitNumericReplacementWinsAfterTheCarry) {
    for (uint32_t op : carries)
        for (uint32_t half : {0u, 1u}) values({.opcode = op, .half = half, .replace = true});
}
TEST(NativeVccCarryReplacement, ExplicitMaskReadersUseTheCurrentCarry) {
    for (uint32_t op : carries)
        for (bool output : {false, true})
            for (bool split : {false, true})
                values({.opcode = op, .output = output, .split = split, .mask_read = true});
}
TEST(NativeVccCarryReplacement, BothWordsExecuteOnARequiredWave64Subgroup) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 compute unavailable";
    for (uint32_t op : carries) {
        for (bool output : {false, true}) {
            for (uint32_t half : {0u, 1u}) {
                for (bool split : {false, true}) {
                    for (bool literal : {false, true}) {
                        const Options o{.opcode = op,
                                        .half = half,
                                        .output = output,
                                        .split = split,
                                        .literal_input = literal};
                        const auto module = compile(o);
                        ASSERT_FALSE(module.empty());
                        if (split) ASSERT_TRUE(dispatcher(module));
                        std::vector<uint32_t> result;
                        prosper::test::run_compute(module, {0.0f}, 64, 1, {},
                                                   std::vector<uint32_t>(64, 0xdeadbeefu), &result,
                                                   64, nullptr, nullptr, nullptr, 64);
                        EXPECT_EQ(result, std::vector<uint32_t>(64, expected(o)));
                    }
                }
            }
        }
    }
}

TEST(NativeVccCarryReplacement, Wave32CarryPreservesTheIndependentHighWord) {
    const auto resources = prosper::test::arbitrary_placeholder::table();
    auto config = prosper::test::native_execution_launch();
    config.wave_size = config.native_subgroup_size = config.local_x = 32;
    config.user_sgprs.resize(12);
    prosper::test::complete_compute_launch(config);
    for (uint32_t opcode : carries) {
        const uint32_t code[]{0xbeeb03ffu,
                              replacement,
                              0x7d8800a0u,
                              0x7e040280u,
                              (opcode << 25u) | (3u << 17u) | (2u << 9u) | 128u,
                              0x7e02026bu,
                              0xe0702000u,
                              0x80020100u,
                              0xbf810000u};
        const auto module = recompile_compute(code, std::size(code), &resources, config);
        ASSERT_FALSE(module.empty());
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 32;
        vm.push_constants.resize(12);
        const auto result = vm.run(32, std::vector<uint32_t>(32, 0xdeadbeefu));
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(result, std::vector<uint32_t>(32, replacement));
    }
}
