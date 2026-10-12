// #4970: a typed direct EXEC spill half must not grant numeric ballot authority from host
// width alone. Exercise the public CFG dispatcher and its real DATA sink; unknown/partial raw
// launches stay refused, while both physical halves keep their exact values on complete launches.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/native_execution_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;

struct Options {
    bool old_data = false;
    bool high = false;
    bool asymmetric = false;
    uint32_t s4 = 0;
    uint32_t s5 = 0;
};

struct Program {
    std::vector<uint32_t> words;
    uint32_t read_pc = 0;
};

Program program(Options o = {}) {
    Program p{{0xbe840380u | o.s4, 0xbe850380u | o.s5, 0x7e020280u, 0x7e060280u}};
    // Actual s4/s5 and v1 definitions supply the unchanged SCC edges and irreducible tail.
    // v3=0 precedes any narrowed EXEC, so the observable result has no absent old destination.
    if (o.asymmetric)
        p.words.insert(p.words.end(), {0x7d8800a8u, 0xbefe046au}); // EXEC=(local x < 40)
    p.words.push_back(0xbe9e047eu); // saved complete Bool, unused as numeric DATA
    if (o.old_data) p.words.insert(p.words.end(), {0xbe8c0387u, 0xd7610014u, 0x0001380cu});
    // The final write of v20[28] is directly from the CURRENT EXEC physical half. An earlier
    // numeric 7 in the same slot must not be mistaken for this mask generation's DATA word.
    p.words.insert(p.words.end(), {0xd7610014u, o.high ? 0x0001387fu : 0x0001387eu});
    if (o.asymmetric)
        p.words.push_back(0xbefe04c1u); // reconverge before the public dispatcher DATA read
    p.words.insert(p.words.end(), {0xbf068004u, 0xbf840001u, 0x7e040281u});
    p.read_pc = static_cast<uint32_t>(p.words.size());
    p.words.insert(p.words.end(), {0xd760000eu, 0x00013914u}); // s14=v20[28]
    if (!o.old_data) p.words.insert(p.words.end(), {0xbf068005u, 0xbf840001u, 0x7e040282u});
    p.words.insert(p.words.end(), {0x4a06000eu, 0xe0702000u, 0x80020300u}); // out[x]=s14+x
    // Existing project-owned irreducible shape forces the actual CFG dispatcher. Every current
    // ordinary source is initialized; branch offsets and the original tail remain unchanged.
    p.words.insert(p.words.end(), {0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u, 0x7d840100u,
                                   0xbf870001u, 0xbf82fffdu, 0x7e040d02u, 0xbf810000u});
    return p;
}

ShaderResourceTable table() {
    ShaderResource out;
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.stride = 4;
    out.sgpr_base = 8;
    ShaderResourceTable t;
    t.resources.push_back(out);
    return t;
}

struct Compilation {
    std::vector<uint32_t> module;
    std::string reason;
};

Compilation compile(const Program& p, const ComputeShaderConfig& config) {
    static uint64_t address = 0x4970000000ull;
    const uint64_t diagnostic_address = address++;
    const auto resources = table();
    auto module = recompile_compute(p.words.data(), p.words.size(), &resources, config,
                                    {RecompileDiagnosticStage::Compute, diagnostic_address});
    return {std::move(module), last_terminal_reject_reason(diagnostic_address)};
}

bool has_opcode(const std::vector<uint32_t>& module, uint32_t opcode) {
    for (size_t at = 5; at < module.size();) {
        const size_t width = module[at] >> 16u;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 0xffffu) == opcode) return true;
        at += width;
    }
    return false;
}

void values(Options o) {
    const auto result = compile(program(o), prosper::test::native_execution_launch());
    ASSERT_FALSE(result.module.empty()) << result.reason;
    ASSERT_TRUE(has_opcode(result.module, 251u)) << "the public CFG dispatcher must produce it";
    ASSERT_TRUE(has_opcode(result.module, 339u)) << "the physical DATA read must emit a ballot";
    bpermute_oracle::Interpreter vm(result.module);
    vm.native_subgroup_width = 64; // explicit complete-control oracle contract, never inferred
    const auto out = vm.run(64, std::vector<uint32_t>(64, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 64u);
    const uint32_t half = o.asymmetric && o.high ? 0xffu : UINT32_MAX;
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(out[lane], lane + half)
            << "lane=" << lane << " high=" << o.high << " old_data=" << o.old_data;
}

void refuses(const ComputeShaderConfig& config) {
    for (bool old_data : {false, true})
        for (bool high : {false, true}) {
            const auto p = program({.old_data = old_data, .high = high});
            // Same words, genuine full-launch control. An unrelated refusal cannot authenticate
            // the raw-launch negative; that negative is compile-only, with no invented CPU lanes.
            const auto control = compile(p, prosper::test::native_execution_launch());
            ASSERT_FALSE(control.module.empty()) << control.reason;
            const auto result = compile(p, config);
            EXPECT_TRUE(result.module.empty());
            // The CFG dispatcher records its generic unresolved-operand label rather than the
            // emitter's stage-reject tag. Pin this exact READLANE and its PC, with the same-word
            // complete-launch control above, so a later unrelated refusal cannot pass this test.
            EXPECT_NE(result.reason.find("cfg-recompile-reject mode=unresolved-operand"),
                      std::string::npos)
                << result.reason;
            EXPECT_NE(result.reason.find("words=d760000e,00013914"), std::string::npos)
                << result.reason;
            EXPECT_NE(result.reason.find("pc=" + std::to_string(p.read_pc) + " "),
                      std::string::npos)
                << result.reason;
        }
}

class MissingRawHalfReadLaneField : public ::testing::TestWithParam<uint32_t> {};
}   // namespace

TEST(NativeHalfReadLaneLaunch, BothLowHalfSccEdgesKeepTheCurrentNumericWord) {
    for (bool old_data : {false, true})
        for (uint32_t s4 : {0u, 1u})
            for (uint32_t s5 : {0u, 1u}) values({.old_data = old_data, .s4 = s4, .s5 = s5});
}

TEST(NativeHalfReadLaneLaunch, BothHighHalfSccEdgesKeepTheCurrentNumericWord) {
    for (bool old_data : {false, true})
        for (uint32_t s4 : {0u, 1u})
            for (uint32_t s5 : {0u, 1u})
                values({.old_data = old_data, .high = true, .s4 = s4, .s5 = s5});
}

TEST(NativeHalfReadLaneLaunch, AsymmetricPhysicalHalvesKeepTheirOwnValuesAfterReconvergence) {
    for (bool old_data : {false, true})
        for (bool high : {false, true})
            for (uint32_t s4 : {0u, 1u})
                for (uint32_t s5 : {0u, 1u})
                    values({.old_data = old_data,
                            .high = high,
                            .asymmetric = true,
                            .s4 = s4,
                            .s5 = s5});
}

TEST(NativeHalfReadLaneLaunch, UnknownRawLaunchCannotGrantPhysicalWordAuthority) {
    auto c = prosper::test::native_execution_launch();
    c.dispatch_launch = {};
    refuses(c);
}

TEST_P(MissingRawHalfReadLaneField, EveryProducingFieldIsRequired) {
    auto c = prosper::test::native_execution_launch();
    c.dispatch_launch.known &= ~GetParam();
    refuses(c);
}

INSTANTIATE_TEST_SUITE_P(NativeHalfReadLaneLaunch, MissingRawHalfReadLaneField,
                         ::testing::Values(ComputeDispatchLaunchState::kInitiator,
                                           ComputeDispatchLaunchState::kNumThreadX,
                                           ComputeDispatchLaunchState::kNumThreadY,
                                           ComputeDispatchLaunchState::kNumThreadZ,
                                           ComputeDispatchLaunchState::kDimX,
                                           ComputeDispatchLaunchState::kDimY,
                                           ComputeDispatchLaunchState::kDimZ));

TEST(NativeHalfReadLaneLaunch, PartialGroupFlagCannotGrantPhysicalWordAuthority) {
    auto c = prosper::test::native_execution_launch();
    c.dispatch_launch.initiator |= 2u;
    refuses(c);
}

TEST(NativeHalfReadLaneLaunch, ShortFinalExtentCannotBorrowCompletePhysicalWords) {
    auto c = prosper::test::native_execution_launch();
    c.exact_thread_extent = true;
    c.threads_x = 33;
    refuses(c);
}

TEST(NativeHalfReadLaneLaunch, NormalizedExtentCannotDisagreeWithItsRawProducer) {
    auto c = prosper::test::native_execution_launch();
    c.threads_x = 128;
    refuses(c);
}

TEST(NativeHalfReadLaneLaunch, ACheckedWave32LaunchCannotSupplyWave64Halves) {
    auto c = prosper::test::native_execution_launch();
    c.dispatch_launch.initiator |= 1u << 15u;
    refuses(c);
}

TEST(NativeHalfReadLaneLaunch, AShorterHostSubgroupCannotAuthenticatePhysicalHalves) {
    auto c = prosper::test::native_execution_launch();
    c.native_subgroup_size = 32;
    for (bool high : {false, true}) {
        const auto p = program({.high = high});
        ASSERT_FALSE(compile(p, prosper::test::native_execution_launch()).module.empty());
        const auto result = compile(p, c);
        EXPECT_TRUE(result.module.empty());
        EXPECT_FALSE(result.reason.empty());
    }
}

TEST(NativeHalfReadLaneLaunch, TheCpuOracleRejectsUndeclaredNativeParticipation) {
    const auto result = compile(program(), prosper::test::native_execution_launch());
    ASSERT_FALSE(result.module.empty()) << result.reason;
    bpermute_oracle::Interpreter missing_width(result.module);
    EXPECT_TRUE(missing_width.run(64, std::vector<uint32_t>(64)).empty());
    EXPECT_FALSE(missing_width.error.empty());
}
