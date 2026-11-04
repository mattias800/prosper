// Mutation: omit unconditional scalar loads from the entry-word lifetime inventory.
// The automatic word serves interpolation; its later numerical value must come from real memory.
#include "fixtures/native_fragment_spirv_oracle.hpp"
#include "fixtures/native_parameter_load.hpp"
#include "fixtures/native_pixel_parameter.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_native_parameter_base.hpp"
#include "gpu/recompiler/float_transport_config.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
namespace input = prosper::test::native_pixel_parameter;
namespace load = prosper::test::native_parameter_load;
auto compile(const load::Words& code, uint32_t parameter, bool smooth, bool wave32,
             const prosper::gpu::ShaderResourceTable& resources) {
    return input::compile(code, input::mapping(parameter, smooth), wave32,
                          {prosper::gpu::FloatTransportProfile::ExplicitNonFinite32}, &resources);
}
void witness(uint32_t parameter, uint32_t observed) {
    for (bool wave32 : {false, true})
        for (bool smooth : {false, true})
            for (bool buffer : {false, true}) {
                SCOPED_TRACE(parameter);
                SCOPED_TRACE(buffer);
                const auto resources = load::resources(buffer);
                const auto result = compile(load::program(parameter, smooth, buffer, observed),
                                            parameter, smooth, wave32, resources);
                ASSERT_FALSE(result.module.empty()) << result.reason;
                EXPECT_GT(prosper::gpu::arbitrary_placeholder_reads(result.module).total(), 0u);
                for (float red : {0.125f, 0.625f})
                    for (float green : {0.25f, 0.75f}) {
                        const auto r = std::bit_cast<uint32_t>(red);
                        const auto g = std::bit_cast<uint32_t>(green);
                        const auto actual = prosper::test::native_fragment_words(
                            result.module, {{0, {r, 0, 0, 0x3f800000u}}},
                            {{15, {0x3f000000u, 0x3f000000u, 0, 0x3f800000u}}}, wave32 ? 32 : 64,
                            {{{1, 2}, load::payload(green, observed)}});
                        EXPECT_EQ(actual, (std::array<uint32_t, 4>{r, g, r, 0x3f800000u}));
                    }
                EXPECT_THROW(prosper::test::native_fragment_words(
                                 result.module, {{0, {0x3f000000u, 0, 0, 0x3f800000u}}},
                                 {{15, {0x3f000000u, 0x3f000000u, 0, 0x3f800000u}}},
                                 wave32 ? 32 : 64),
                             std::runtime_error);
            }
}
} // namespace

TEST(NativeParameterLoad, TheExistingMoveControlAlreadyObservesSuppliedMemory) {
    witness(30, 20);
}

TEST(NativeParameterLoad, ACompleteLoadEndsTheEntryWordLifetimeIncludingItsLastWord) {
    witness(20, 20);
    witness(23, 23);
}

TEST(NativeParameterLoad, ALoadOutsideTheWordCannotRetireIt) {
    const auto code = load::program(24, false, false, 24);
    EXPECT_FALSE(prosper::gpu::native_parameter_base_plan(input::mapping(24).parameter_launch,
                                                          code.data(), code.size()));
    const auto resources = load::resources();
    for (bool wave32 : {false, true}) {
        const auto result = compile(code, 24, false, wave32, resources);
        EXPECT_TRUE(result.module.empty());
        EXPECT_NE(result.reason.find("origin=launch-input word=s24"), std::string::npos)
            << result.reason;
    }
}

TEST(NativeParameterLoad, TheEntryServiceDoesNotSupplyMissingMemory) {
    for (bool buffer : {false, true})
        for (bool wave32 : {false, true})
            EXPECT_TRUE(
                compile(load::program(20, false, buffer), 20, false, wave32, {}).module.empty());
}

TEST(NativeParameterLoad, ALoadAfterControlCannotBorrowAnUnconditionalEntryLifetime) {
    auto code = load::program();
    code.insert(code.begin() + 2, 0xbf840000u); // a scalar-control boundary before the load
    EXPECT_FALSE(prosper::gpu::native_parameter_base_plan(input::mapping(20).parameter_launch,
                                                          code.data(), code.size()));
}
