// Dependency repair activates ordinary input guards too. It needs the real PS entry service
// independently of Arbitrary admission; otherwise a genuine automatic M0 input stays missing.
#include "fixtures/native_fragment_spirv_oracle.hpp"
#include "fixtures/native_pixel_parameter.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::native_pixel_parameter;
fixture::Words program() {
    auto code = fixture::program();
    code[3] = 0x7e0e02f0u; // v7 = 0.5, so no unloaded source requests Arbitrary admission
    code[6] = 0x7d840f07u; // compare the genuinely initialized v7 with itself
    code.erase(code.begin() + 4, code.begin() + 6); // remove the unrelated READLANE relay
    return code;
}
fixture::Words repaired(const fixture::Words& code, const PixelSystemInputMapping& inputs) {
    unsigned attempts = 0;
    const auto result = recompile_with_arbitrary_placeholders({}, [&] {
        ++attempts;
        if (!allow_scalar_merge_dependency_repair()) return fixture::Words{};
        EXPECT_TRUE(scalar_dependency_repair_active());
        EXPECT_TRUE(placeholder_recompile_guards_active());
        EXPECT_FALSE(arbitrary_placeholder_retry_active());
        return fixture::compile(code, inputs).module;
    });
    EXPECT_EQ(attempts, 2u);
    return result;
}
} // namespace

TEST(ScalarDependencyParameter, GuardedRepairExecutesTheRealInputWithoutArbitraryAdmission) {
    const auto module = repaired(program(), fixture::mapping());
    ASSERT_FALSE(module.empty());
    EXPECT_EQ(arbitrary_placeholder_reads(module).total(), 0u);
    for (const uint32_t red : {0x3e000000u, 0x3f200000u}) {
        const auto actual = prosper::test::native_fragment_words(
            module, {{0, {red, 0, 0, 0x3f800000u}}},
            {{15, {0x3f000000u, 0x3f000000u, 0, 0x3f800000u}}}, 64);
        EXPECT_EQ(actual, (std::array<uint32_t, 4>{red, 0x3f000000u, red, 0x3f800000u}));
    }
}

TEST(ScalarDependencyParameter, UnknownAndWrongSystemPrefixesStayHardDuringRepair) {
    auto unknown = fixture::mapping();
    unknown.parameter_launch = {};
    EXPECT_TRUE(repaired(program(), unknown).empty());
    EXPECT_TRUE(repaired(program(), fixture::mapping(31)).empty());
}

TEST(ScalarDependencyParameter, NumericalM0AndUnreplacedSystemWordsStayHardDuringRepair) {
    auto code = program();
    code[7] = 0x7e12027cu; // the old v9=s30 numerical observation now reads M0 instead
    EXPECT_TRUE(repaired(code, fixture::mapping()).empty());
    code = program();
    code[1] = 0xbf800000u; // no complete overwrite before the later numerical s30 read
    EXPECT_TRUE(repaired(code, fixture::mapping()).empty());
}
