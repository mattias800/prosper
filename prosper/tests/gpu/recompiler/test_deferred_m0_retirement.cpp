// #5497: deferred M0 retirement must authenticate literals and every legacy M0 read.
#include "gpu/recompiler/rdna2_cfg_m0_retirement.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <initializer_list>
#include <vector>
namespace {
using namespace prosper::gpu;
std::vector<Rdna2Inst> instructions(std::initializer_list<uint32_t> words) {
    const std::vector<uint32_t> code(words);
    std::vector<Rdna2Inst> result;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), result), code.size());
    return result;
}
}   // namespace
TEST(DeferredM0Prefix, APlainScalarLoadCanPrecedeTheIndependentOverwrite) {
    const auto ins = instructions({0xf4080406u, 0xfa0000a0u, 0xbefc0380u});
    ASSERT_EQ(ins.size(), 2u);
    ASSERT_EQ(ins.front().fmt, Rdna2Format::SMEM);
    ASSERT_EQ(ins.front().opcode, 0x02u);
    EXPECT_TRUE(deferred_waterfall_m0_retirement(ins, 0, 3));
}
TEST(DeferredM0Prefix, AStoreCannotHideAnImplicitSdataReadAcrossM0) {
    const auto ins = instructions({0xf4441ec6u, 0xfa0000a0u, 0xbefc0380u});
    ASSERT_EQ(ins.size(), 2u);
    ASSERT_EQ(ins.front().fmt, Rdna2Format::SMEM);
    ASSERT_EQ(ins.front().opcode, 0x11u);
    ASSERT_EQ(ins.front().dst.value, 123);
    EXPECT_FALSE(deferred_waterfall_m0_retirement(ins, 0, 3));
}

TEST(DeferredM0Prefix, LiteralMadmkAndLiteralOverwriteReadBackExactly) {
    const auto ins = instructions({0x40500904u, 0x40000000u, 0xbefc03ffu, 0x12345678u});
    ASSERT_EQ(ins.size(), 2u);
    ASSERT_TRUE(ins[0].has_literal);
    ASSERT_EQ(ins[0].literal, 0x40000000u);
    ASSERT_TRUE(ins[1].has_literal);
    EXPECT_TRUE(deferred_waterfall_m0_retirement(ins, 0, 3));
    const auto wide = instructions({0xd5410006u, 0x1bc8c8ffu, 0xc0400000u, 0xbefc0380u});
    ASSERT_EQ(wide.size(), 2u);
    ASSERT_EQ(wide[0].len_dwords, 3u);
    ASSERT_TRUE(wide[0].has_literal);
    EXPECT_TRUE(deferred_waterfall_m0_retirement(wide, 0, 3));
}
TEST(DeferredM0Prefix, ParsedOperandsAndLiteralMetadataMustMatchTheEncoding) {
    const auto original = instructions({0x40500904u, 0x40000000u, 0xbefc0380u});
    ASSERT_TRUE(deferred_waterfall_m0_retirement(original, 0, 3));
    for (int mutation = 0; mutation < 5; ++mutation) {
        auto ins = original;
        switch (mutation) {
            case 0: ins[0].src[0].value = 124; break;
            case 1: ins[0].opcode = 0x21; break;
            case 2: ins[0].has_literal = false; break;
            case 3: ins[0].len_dwords = 1; break;
            case 4: ins[1].dst.value = 123; break;
        }
        EXPECT_FALSE(deferred_waterfall_m0_retirement(ins, 0, 3)) << mutation;
    }
}
TEST(DeferredM0Prefix, ScalarPairAndLoadAddressCannotHideM0Reads) {
    for (const auto words : {std::initializer_list<uint32_t>{0xbe82047bu, 0xbefc0380u},
                             {0xf408043eu, 0xfa0000a0u, 0xbefc0380u}}) {
        const auto ins = instructions(words);
        EXPECT_FALSE(deferred_waterfall_m0_retirement(ins, 0, 3));
    }
}
TEST(DeferredM0Prefix, AWriteToM0BeforeTheIndependentKillIsObservable) {
    const auto ins = instructions({0x807c8180u, 0xbefc0380u}); // s_add_u32 M0, 0, 1
    ASSERT_EQ(ins.front().fmt, Rdna2Format::SOP2);
    ASSERT_EQ(ins.front().opcode, 0u);
    ASSERT_EQ(ins.front().dst.value, 124);
    EXPECT_FALSE(deferred_waterfall_m0_retirement(ins, 0, 3));
}
TEST(DeferredM0Prefix, DiscontinuousOrSyntheticInstructionsCannotRetireM0) {
    const auto original = instructions({0xbf800000u, 0xbefc0380u});
    ASSERT_TRUE(deferred_waterfall_m0_retirement(original, 0, 3));
    for (int mutation = 0; mutation < 3; ++mutation) {
        auto ins = original;
        if (mutation == 0) ++ins[1].pc;
        if (mutation == 1) ins[0].synthetic_terminator = true;
        if (mutation == 2) ins[0].is_end = true;
        EXPECT_FALSE(deferred_waterfall_m0_retirement(ins, 0, 3)) << mutation;
    }
}
