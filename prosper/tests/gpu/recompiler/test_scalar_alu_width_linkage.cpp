// A declaration-only ALU caller must link against the effects-table object. Keeping the
// definition inline in cfg_support made optimized Windows app links depend on another
// translation unit accidentally emitting a weak definition (#5153).
#include <gtest/gtest.h>

#include <cstdint>

#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

TEST(ScalarAluWidthLinkage, DeclarationOnlyCallerUsesTheTableDefinition) {
    prosper::gpu::Rdna2Inst in{};
    in.fmt = prosper::gpu::Rdna2Format::VOP1;
    in.opcode = 0x0fu;   // v_cvt_f32_f64 reads a pair.
    EXPECT_EQ(prosper::gpu::scalar_alu_source_words(in, 0), 2u);
    in.fmt = prosper::gpu::Rdna2Format::VOP3;
    in.opcode = 0x103u;   // v_add_f32 reads two single words, no SRC2.
    EXPECT_EQ(prosper::gpu::scalar_alu_source_words(in, 0), 1u);
    EXPECT_EQ(prosper::gpu::scalar_alu_source_words(in, 2), UINT32_MAX);
}
