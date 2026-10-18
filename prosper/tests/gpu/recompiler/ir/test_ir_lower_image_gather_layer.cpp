// A missing array layer is a zero SPIR-V-id sentinel, never a generated numeric zero. A present
// layer must keep its original address slot and raw id; scaling applies only to spatial axes.
#include "gpu/recompiler/ir/lower/image_gather_layer_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
TEST(IrLowerImageGatherLayer, AbsentLayerIsAnUnreadZeroIdSentinel) {
    for (bool offset : {false, true}) {
        LoweringFixture f;
        uint32_t calls = 0;
        const auto size = f.code_words();
        EXPECT_EQ(lower_image_gather_layer(f.b, false, offset,
                                           [&](uint32_t) {
                                               ++calls;
                                               return f.b.load_input(0);
                                           }),
                  0u);
        EXPECT_EQ(calls, 0u);
        EXPECT_EQ(f.code_words(), size);
    }
}
TEST(IrLowerImageGatherLayer, ArrayReadsExactlyItsLayerSlotOnce) {
    for (bool offset : {false, true}) {
        LoweringFixture f;
        std::vector<uint32_t> slots, ids;
        const auto result = lower_image_gather_layer(f.b, true, offset, [&](uint32_t slot) {
            slots.push_back(slot);
            ids.push_back(f.b.load_input(slot));
            return ids.back();
        });
        EXPECT_EQ(slots, (std::vector<uint32_t>{offset ? 3u : 2u}));
        ASSERT_EQ(ids.size(), 1u);
        EXPECT_EQ(result, ids[0]);
        EXPECT_NE(result, 0u);
    }
}
TEST(IrLowerImageGatherLayer, ExistingLayerWordIsReturnedWithoutExtraEmission) {
    for (bool offset : {false, true}) {
        LoweringFixture f;
        const auto id = f.b.load_input(0);
        const auto size = f.code_words();
        EXPECT_EQ(lower_image_gather_layer(f.b, true, offset, [&](uint32_t) { return id; }), id);
        EXPECT_EQ(f.code_words(), size);
    }
}
} // namespace
