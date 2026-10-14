// test_scalar_source_words_link: scalar_alu_source_words links for a caller that sees only its
// declaration (#5153).
//
// rdna2_alu_support.hpp declares the function non-inline for callers such as rdna2_emit_alu.cpp
// that never include the header which used to hold its inline definition. After #5150 shrank the
// body, every call site inlined it, no object emitted a standalone copy, and MinGW's static link
// failed with an undefined reference. This file deliberately includes only the decoder and declares
// the function itself -- exactly the shape of those callers -- so it fails to link if the definition
// ever becomes inline-only again.
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <iterator>
#include <vector>

namespace prosper::gpu {
uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source);
}   // namespace prosper::gpu

using namespace prosper::gpu;

TEST(ScalarSourceWordsLink, CallableFromADeclarationOnlyTranslationUnit) {
    // v_bcnt_u32_b32 v7, s9, v7: two B32 sources, one SGPR each (#5148).
    const uint32_t code[] = {0xd7640007u, 0x00020e09u};
    std::vector<Rdna2Inst> decoded;
    ASSERT_EQ(rdna2_walk(code, std::size(code), decoded), std::size(code));
    ASSERT_EQ(decoded.size(), 1u);
    EXPECT_EQ(scalar_alu_source_words(decoded[0], 0), 1u);
}
