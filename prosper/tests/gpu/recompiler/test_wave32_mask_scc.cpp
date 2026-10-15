// Pins the dispatcher's Wave32 mask-intersection SCC proof: a real native vote must remain
// available to its branch, while missing words or an incomplete producing launch cannot
// authenticate that condition. Losing either distinction rejects valid code or branches on
// fabricated state. These compile-time tests exercise emitted SPIR-V, not a device interpreter.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;

std::vector<uint32_t> program() {
    return {
        0xbe950380u,
        0xbe990381u,
        0xbe9a0382u,
        0x7e0a0300u, // real s21/s25/s26/v5 inputs
        0xbe9b037eu,                                     // s27 = EXEC_LO
        0x7d8a0a15u,                                     // compare VCC_LO, s21, v5
        0x7d8a0af9u,
        0x0686eb19u,                        // compare s107, s25, v5
        0x7d8a0af9u,
        0x06869c1au,                        // compare s28, s26, v5
        0x876a6a1bu,
        0x876a6b6au,
        0x876a1c6au,            // intersect the three masks
        0xbf840001u,
        0xbf800000u,                         // SCC0 branch, fallthrough work
        // The existing reduced dispatcher tail keeps the native vote in the real CFG route.
        0xbe840380u,
        0xbe800380u,
        0x7e000280u,
        0x7e020300u,
        0xd7610013u,
        0x00014a7eu,
        0xd7610013u,
        0x0001507fu,
        0xd760000eu,
        0x00014b13u,
        0xd760000fu,
        0x00015113u,
        0xbefe040eu,
        0xe00c2000u,
        0x80020400u,
        0x7db900f9u,
        0x86050007u,
        0x7d020200u,
        0xbf860006u,
        0xbf0a8204u,
        0x360000fdu,
        0xbf840001u,
        0x81008100u,
        0x81008100u,
        0xbf82fff4u,
        0xbf810000u,
    };
}

struct Compilation {
    std::vector<uint32_t> module;
    std::string reason;
};

enum class Launch { Complete, Unknown, Partial, MismatchedThreadWord };

Compilation compile(const std::vector<uint32_t>& code, uint32_t native_width = 32,
                    Launch launch = Launch::Complete) {
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 32;
    config.native_subgroup_size = native_width;
    config.user_sgprs.resize(1);   // genuine entry s0 for the conditional definition controls
    prosper::test::complete_compute_launch(config);
    if (launch == Launch::Unknown)
        config.dispatch_launch = {};
    else if (launch == Launch::Partial)
        config.dispatch_launch.initiator |= 2u;   // actual PARTIAL_TG_EN, not normalized geometry
    else if (launch == Launch::MismatchedThreadWord)
        config.dispatch_launch.num_thread[0] = 32;   // normalized local size remains 64
    ShaderResourceTable table;
    ShaderResource resource;
    resource.cls = ResourceClass::VertexBuffer;
    resource.binding = 3;
    resource.sgpr_base = 8;
    resource.stride = 16;
    resource.format = DataFormat::Float32;
    resource.num_components = 4;
    table.resources.push_back(resource);
    static uint64_t next_address = 0x5051000100ull;
    const auto address = next_address++;
    auto module = recompile_compute(code.data(), code.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

bool has_opcode(const std::vector<uint32_t>& module, uint32_t opcode) {
    for (std::size_t at = 5; at < module.size();) {
        const uint32_t width = module[at] >> 16;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 0xffffu) == opcode) return true;
        at += width;
    }
    return false;
}

void actual_native_condition(const Compilation& result) {
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_TRUE(has_opcode(result.module, 251)) << "OpSwitch must own the real CFG route";
    EXPECT_TRUE(has_opcode(result.module, 335)) << "OpGroupNonUniformAny must compute wave SCC";
}

void missing_source(const Compilation& result, uint32_t pc) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("fabricated-fresh-mask-source"), std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << result.reason;
}
}   // namespace

TEST(Wave32MaskScc, NativeIntersectionPublishesAnActualCondition) {
    actual_native_condition(compile(program()));
}

TEST(Wave32MaskScc, MissingComparedScalarCannotBecomeAMaskCondition) {
    auto code = program();
    code[1] = 0xbf800000u;   // remove only s25's actual producer
    missing_source(compile(code), 6);
}

TEST(Wave32MaskScc, MissingComparedVectorCannotBecomeAMaskCondition) {
    auto code = program();
    // Replace only v5's actual producer by a relay of s30, which no launch loads: hard on every
    // route. (A never-written v5 would be arbitrary under #5185 on every route, #5039.)
    code[3] = 0x7e0a021eu;   // v_mov_b32 v5, s30
    missing_source(compile(code), 5);
}

TEST(Wave32MaskScc, AOnePathWordRemainsMissingAtTheCompare) {
    auto code = program();
    code[1] = 0xbf800000u;
    code.insert(code.begin() + 4, {0xbf060000u, 0xbf850001u, 0xbe990381u});
    missing_source(compile(code), 9);
}

TEST(Wave32MaskScc, BothPathsCanSupplyTheActualComparedWord) {
    auto code = program();
    code[1] = 0xbf800000u;
    code.insert(code.begin() + 4,
                {0xbf060000u, 0xbf850002u, 0xbe990381u, 0xbf820001u, 0xbe990382u});
    actual_native_condition(compile(code));
}

TEST(Wave32MaskScc, APortableProfileDoesNotBorrowTheNativeVote) {
    const auto result = compile(program(), 0);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("poisoned-scc"), std::string::npos) << result.reason;
    EXPECT_NE(result.reason.find("pc=13 "), std::string::npos) << result.reason;
}

TEST(Wave32MaskScc, AnOlderPoisonedSccCannotPoisonAnIndependentNativeVote) {
    auto code = program();
    code.insert(code.begin() + 4, 0xbf068028u);   // s_cmp_eq_u32 absent s40, 0
    actual_native_condition(compile(code));
}

TEST(Wave32MaskScc, AVectorRelayCannotAuthenticateAnAbsentScalarSource) {
    auto code = program();
    code[1] = 0xbf800000u;
    code.insert(code.begin() + 4,
                {0x7e3c0228u, 0x7e32051eu});   // v30 = s40; s25 = readfirstlane(v30)
    missing_source(compile(code), 8);
    code.insert(code.begin() + 4, 0xbea80381u);   // genuine s40 = 1 before the identical relay
    actual_native_condition(compile(code));
}

TEST(Wave32MaskScc, AMismatchedNativeWidthCannotSupplyTheGuestVote) {
    const auto result = compile(program(), 64);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("poisoned-scc"), std::string::npos) << result.reason;
    EXPECT_NE(result.reason.find("pc=13 "), std::string::npos) << result.reason;
}

TEST(Wave32MaskScc, NormalizedGeometryCannotSupplyUnknownProducingWords) {
    const auto result = compile(program(), 32, Launch::Unknown);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("poisoned-scc"), std::string::npos) << result.reason;
    EXPECT_NE(result.reason.find("pc=13 "), std::string::npos) << result.reason;
}

TEST(Wave32MaskScc, PartialThreadGroupsCannotSupplyAllGuestMaskBits) {
    const auto result = compile(program(), 32, Launch::Partial);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("poisoned-scc"), std::string::npos) << result.reason;
    EXPECT_NE(result.reason.find("pc=13 "), std::string::npos) << result.reason;
}

TEST(Wave32MaskScc, AChangedThreadWordCannotBorrowTheNormalizedLaunch) {
    const auto result = compile(program(), 32, Launch::MismatchedThreadWord);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("poisoned-scc"), std::string::npos) << result.reason;
    EXPECT_NE(result.reason.find("pc=13 "), std::string::npos) << result.reason;
}
