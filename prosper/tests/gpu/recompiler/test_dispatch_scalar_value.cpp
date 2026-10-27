// A dispatcher reload must retain computed scalar DATA independently of its placeholder mark.
// Lane 0 is enabled in every execution arm: its v6 is written before READLANE selects it.
// Other lanes need not initialize v6. Dropping s20 after adding 7 silently stores 0 on the GPU.
#include "fixtures/compute_runner.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
using namespace prosper::gpu;
std::vector<uint32_t> program(bool complete = false, bool predicate = false) {
    // llvm-mc gfx1030, Wave64: CMPX; v6=0; restore EXEC; READLANE s15,v6,0;
    // s20=s15+7; constant SCC; conditional diamond; v1=s20; END.
    std::vector<uint32_t> code{0x7daa0080u, 0x7e0c0280u, 0xbefe04c1u, 0xd760000fu,
                               0x00010106u, 0x8014870fu, 0xbf068080u, 0xbf840001u,
                               0xbf800000u, 0x7e020214u, 0xbf810000u};
    if (predicate) code.insert(code.begin() + 9, 0x7d840014u); // CMP_EQ VCC,s20,v0
    if (complete) code.insert(code.begin(), 0x7e0c0280u); // v6=0 before narrowing
    return code;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code) {
    return recompile_valu(code.data(), code.size(), 1, 1, nullptr, 0, kDefaultComputePgmRsrc1, true,
                          64, 64);
}
void expect_seven(const std::vector<uint32_t>& code, const std::vector<float>& input) {
    const auto module = compile(code);
    ASSERT_FALSE(module.empty());
    const auto output = prosper::test::run_compute(module, input, 64, 64);
    ASSERT_EQ(output.size(), 64u) << "A Vulkan device was found; dispatch must execute";
    std::vector<uint32_t> bits(output.size());
    std::memcpy(bits.data(), output.data(), bits.size() * sizeof(uint32_t));
    EXPECT_EQ(bits, std::vector<uint32_t>(64, 7));
}
bool device_available() {
    return prosper::test::default_compute_subgroup_properties().size != 0;
}
} // namespace

TEST(DispatchScalarValue, APartialVectorSourceRetainsItsComputedScalarAcrossTheSplit) {
    if (!device_available()) GTEST_SKIP() << "no Vulkan compute device";
    std::vector<float> input(64, 1.0f);
    expect_seven(program(), input);
    // Lane 0 remains enabled; inactive other lanes do not supply READLANE's chosen word.
    for (unsigned lane : {1u, 31u, 32u, 63u}) input[lane] = 0.0f;
    expect_seven(program(), input);
}

TEST(DispatchScalarValue, ACompleteSourceRetainsTheSameComputedValue) {
    if (!device_available()) GTEST_SKIP() << "no Vulkan compute device";
    expect_seven(program(true), std::vector<float>(64, 1.0f));
}

TEST(DispatchScalarValue, AStraightScalarRelayRetainsTheSameValue) {
    if (!device_available()) GTEST_SKIP() << "no Vulkan compute device";
    auto code = program();
    code[7] = 0xbf800000u; // replace the conditional split with NOP
    expect_seven(code, std::vector<float>(64, 1.0f));
}

TEST(DispatchScalarValue, RetainedDataDoesNotEraseItsMaskInputObligation) {
    EXPECT_TRUE(compile(program(false, true)).empty());
}

TEST(DispatchScalarValue, ACompleteSourceCanStillProduceAFreshPredicate) {
    if (!device_available()) GTEST_SKIP() << "no Vulkan compute device";
    expect_seven(program(true, true), std::vector<float>(64, 1.0f));
}
