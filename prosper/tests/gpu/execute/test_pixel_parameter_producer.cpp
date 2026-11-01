// The shipping draw producer must key native parameter authority on the actual RSRC2 word.
// A warm accepted draw must not supply missing authority or borrow another user-SGPR prefix.
#include "fixtures/native_pixel_parameter.hpp"
#include "fixtures/ps_pull_model_fixture.hpp"
#include <gtest/gtest.h>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::native_pixel_parameter;
namespace programs = prosper::test::ps_pull;
programs::Owners* owner(uint32_t count) {
    prosper::register_builtin_hle();
    const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    uint64_t address = 0;
    if (!map ||
        map(reinterpret_cast<uint64_t>(&address), 0x10000, 2, 0,
            reinterpret_cast<uint64_t>("pixel-parameter-programs"), 0) != 0 ||
        address < 0x100000000ull || address % alignof(programs::Owners))
        return nullptr;
    // Registry-owned code remains immutable for process lifetime across all cache arms.
    auto& result = *std::construct_at(reinterpret_cast<programs::Owners*>(address));
    if (!programs::register_program(result.vs, true,
                                    programs::vertex_words(programs::ClipW::One)) ||
        !programs::register_program(result.ps, false, fixture::program(count)))
        return nullptr;
    return &result;
}
bool realize(programs::Owners& owner, const PixelParameterLaunch& launch, DrawItem& out,
             OperationRealizationFailure* failure = nullptr) {
    namespace p = prosper::agc::Pm4;
    GpuState state;
    for (const auto* program : {&owner.vs, &owner.ps})
        for (const auto& reg : program->registers) state.sh[reg.offset] = reg.value;
    state.uc[p::VGT_PRIMITIVE_TYPE] = 4;
    state.cx[p::CB_TARGET_MASK] = state.cx[p::CB_SHADER_MASK] = 15;
    state.cx[p::SPI_PS_IN_CONTROL] = 1;
    state.cx[p::SPI_PS_INPUT_CNTL_0] = 0;
    state.cx[p::SPI_PS_INPUT_ENA] = state.cx[p::SPI_PS_INPUT_ADDR] = 1u << 8;
    if (launch.available) state.sh[p::SPI_SHADER_PGM_RSRC2_PS] = launch.value;
    GpuState::Draw packet;
    packet.index_count = 3;
    packet.instance_count = 1;
    packet.command_order = 1;
    state.draws.push_back(packet);
    return realize_draw_item(state, &state.draws[0], 3, 64, false, out, failure, true);
}
} // namespace

TEST(PixelParameterProducer, WarmCacheCannotBorrowAnotherPrefixOrMissingRegister) {
    auto* code = owner(30);
    ASSERT_NE(code, nullptr);
    DrawItem first, warm, wrong, unknown;
    const auto launch = fixture::mapping().parameter_launch;
    ASSERT_TRUE(realize(*code, launch, first));
    ASSERT_TRUE(realize(*code, launch, warm));
    EXPECT_EQ(first.system_inputs.parameter_launch, launch);
    EXPECT_EQ(first.fs_identity, warm.fs_identity);
    EXPECT_EQ(first.fs_words(), warm.fs_words());
    ASSERT_FALSE(first.fs_words().empty());
    OperationRealizationFailure failure;
    EXPECT_FALSE(realize(*code, fixture::mapping(31).parameter_launch, wrong, &failure));
    EXPECT_EQ(failure.system_inputs.parameter_launch, fixture::mapping(31).parameter_launch);
    failure = {};
    EXPECT_FALSE(realize(*code, {}, unknown, &failure));
    EXPECT_TRUE(failure.has_system_inputs && failure.fragment_retry_config_available);
    EXPECT_EQ(failure.system_inputs.parameter_launch, PixelParameterLaunch{});
    EXPECT_TRUE(unknown.fs_words().empty());
}

TEST(PixelParameterProducer, ObservedZeroIsKnownAndAbsentZeroIsNot) {
    auto* code = owner(0);
    ASSERT_NE(code, nullptr);
    DrawItem known, unknown;
    ASSERT_TRUE(realize(*code, {true, 0}, known));
    EXPECT_EQ(known.system_inputs.parameter_launch, (PixelParameterLaunch{true, 0}));
    EXPECT_FALSE(realize(*code, {}, unknown));
}
