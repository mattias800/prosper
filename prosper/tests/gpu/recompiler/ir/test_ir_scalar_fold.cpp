// The constant folder (rdna2_recompile_shared.hpp) evaluates scalar ops through their IR
// definitions (#2514). Expected values here are worked by hand from the ISA, so a definition or an
// operand wiring that disagrees with the ISA fails here even though fold and emitter agree.
#include "gpu/recompiler/ir/scalar_encoding.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"

#include <gtest/gtest.h>

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {
// Folds the SOPC compare that ends `code`; nullopt when the folder cannot prove it.
std::optional<bool> fold(std::vector<uint32_t> code) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    bool result = false;
    if (!shader_constant_compare(ins, 0, ins.size() - 1, result)) return std::nullopt;
    return result;
}
std::vector<uint32_t> mov(uint32_t sgpr, uint32_t value) {
    return {sop1(0x03, sgpr, kLiteral), value};
}
std::vector<uint32_t> operator+(std::vector<uint32_t> a, const std::vector<uint32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}
// s_cmp_eq_u32 s22, literal
std::vector<uint32_t> is(uint32_t value) { return {sopc(0x06, 22, kLiteral), value}; }
} // namespace

TEST(IrScalarFold, Sop2ValuesFoldThroughTheirDefinitions) {
    const auto ab = mov(20, 5) + mov(21, 7);
    EXPECT_EQ(fold(ab + std::vector{sop2(0x01, 22, 20, 21)} + is(0xfffffffeu)), true);   // 5 - 7
    EXPECT_EQ(fold(ab + std::vector{sop2(0x01, 22, 20, 21)} + is(2u)), false);
    // Folds the old eleven-opcode switch never covered.
    EXPECT_EQ(fold(mov(20, 0xffffffffu) + mov(21, 1) + std::vector{sop2(0x06, 22, 20, 21)} +
                   is(0xffffffffu)),
              true);   // s_min_i32(-1, 1)
    EXPECT_EQ(fold(mov(20, 0xabcd1234u) + mov(21, (8u << 16) | 4u) +
                   std::vector{sop2(0x27, 22, 20, 21)} + is(0x23u)),
              true);   // s_bfe_u32: 8 bits from bit 4
    EXPECT_EQ(fold(mov(20, 1) + mov(21, 33) + std::vector{sop2(0x1e, 22, 20, 21)} + is(2u)),
              true);   // s_lshl_b32 masks the amount to 5 bits
}

TEST(IrScalarFold, Sop1AndSopkValuesFold) {
    EXPECT_EQ(fold(mov(20, 5) + std::vector{sop1(0x07, 22, 20)} + is(~5u)), true);   // s_not_b32
    EXPECT_EQ(fold(std::vector{sopk(0x00, 22, 0xffff)} + is(0xffffffffu)), true);   // s_movk_i32
    EXPECT_EQ(fold(mov(22, 0x7fffffffu) + std::vector{sopk(0x0f, 22, 1)} + is(0x80000000u)),
              true);   // s_addk_i32 reads its SDST
}

TEST(IrScalarFold, SccReadersAndRuntimeInputsDoNotFold) {
    EXPECT_EQ(fold(mov(20, 5) + mov(21, 7) + std::vector{sop2(0x0a, 22, 20, 21)} + is(5u)),
              std::nullopt);   // s_cselect_b32 reads SCC
    EXPECT_EQ(fold(mov(21, 7) + std::vector{sop2(0x00, 22, 20, 21)} + is(12u)),
              std::nullopt);   // s20 is an entry SGPR
}

TEST(IrScalarFold, ComparesFoldThroughTheirDefinitions) {
    EXPECT_EQ(fold(mov(20, 0xffffffffu) + std::vector{sopc(0x04, 20, kInline0)}), true);   // -1 < 0
    EXPECT_EQ(fold(mov(20, 0xffffffffu) + std::vector{sopc(0x0a, 20, kInline0)}), false);  // u32
    EXPECT_EQ(fold(mov(20, 5) + std::vector{sopc(0x0d, 20, kInline0)}), true);   // s_bitcmp1 bit 0
    EXPECT_EQ(fold(mov(20, 5) + std::vector{sopc(0x0c, 20, kInline0 + 1)}), true);   // bit 1 clear
}
