// Routing retains source ids and emits constants only at their existing channel position;
// unsigned D16 extracts numeric halves, whereas float D16 widens the packed half value.
#include "gpu/recompiler/ir/lower/image_store_values_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace {
TEST(IrLowerImageStoreValues, LoadSelectsReuseIdsAndFormatConstants) {
    SpirvCompute b;
    b.begin(4, nullptr, 1, 1, 1, 1, 0, true, true);
    const uint32_t input[] = {b.load_input(0), b.load_input(1), b.load_input(2), b.load_input(3)};
    const int channel[] = {3, -1, 1, -1};
    const uint32_t constant[] = {0, 1, 0, 0x3f800000};
    uint32_t result[4]{};
    const auto before = b.code.size();
    lower_image_storage_load_selects(b, input, channel, constant, result);
    EXPECT_EQ(result[0], input[3]);
    EXPECT_EQ(result[2], input[1]);
    EXPECT_EQ(result[1], b.uconst(1));
    EXPECT_EQ(result[3], b.uconst(0x3f800000));
    EXPECT_EQ(b.code.size(), before);
}
TEST(IrLowerImageStoreValues, StoreSelectsReuseIdsAndZeroAnAbsentSource) {
    SpirvCompute b;
    b.begin(4, nullptr, 1, 1, 1, 1, 0, true, true);
    const uint32_t input[] = {b.load_input(0), b.load_input(1), b.load_input(2), b.load_input(3)};
    const int source[] = {2, 0, -1, 1};
    uint32_t result[4]{};
    const auto before = b.code.size();
    lower_image_storage_store_selects(b, input, source, result);
    EXPECT_EQ(result[0], input[2]);
    EXPECT_EQ(result[1], input[0]);
    EXPECT_EQ(result[2], b.uconst(0));
    EXPECT_EQ(result[3], input[1]);
    EXPECT_EQ(b.code.size(), before);
}
TEST(IrLowerImageStoreValues, RawStoreWordIsTheExistingIdWithoutExtraEmission) {
    SpirvCompute b;
    b.begin(1, nullptr, 1, 1, 1, 1, 0, true, true);
    const uint32_t word = b.load_input(0);
    const auto before = b.code.size();
    EXPECT_EQ(lower_image_store_raw_word(b, word), word);
    EXPECT_EQ(b.code.size(), before);
}
TEST(IrLowerImageStoreValues, HalfWordsKeepUnsignedAndFloatingInterpretationsDistinct) {
    for (bool integer : {false, true})
        for (uint32_t half : {0u, 1u}) {
            SpirvCompute b;
            b.begin(1, nullptr, 1, 1, 1, 1, 0, true, true);
            b.store_output_word(lower_image_store_d16_word(b, b.load_input(0), half, integer), 1,
                                0);
            bpermute_oracle::Interpreter vm(b.finish());
            vm.half_rte_preserve_denorm = true;
            const auto result = vm.run_buffers(1, {{0, {0xc0003c00u}}, {1, {0}}}, 1);
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            const uint32_t expected =
                integer ? (half ? 0xc000u : 0x3c00u) : (half ? 0xc0000000u : 0x3f800000u);
            EXPECT_EQ(result, (std::vector<uint32_t>{expected}));
        }
}
} // namespace
