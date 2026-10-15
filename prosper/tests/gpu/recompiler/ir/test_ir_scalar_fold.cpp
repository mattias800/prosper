// The constant folder (rdna2_recompile_shared.hpp) evaluates scalar ops through their IR
// definitions (#2514), within its pre-IR domain. Expected values here are worked by hand from the
// ISA, so a definition or an operand wiring that disagrees with the ISA fails here even though fold
// and emitter agree.
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
    EXPECT_EQ(fold(ab + std::vector{sop2(0x26, 22, 20, 21)} + is(35u)), true);   // s_mul_i32
    EXPECT_EQ(fold(mov(20, 0x80000000u) + mov(21, 33) + std::vector{sop2(0x22, 22, 20, 21)} +
                   is(0xc0000000u)),
              true);   // s_ashr_i32 masks the amount to 5 bits and keeps the sign
    EXPECT_EQ(fold(mov(20, 1) + mov(21, 33) + std::vector{sop2(0x1e, 22, 20, 21)} + is(2u)),
              true);   // s_lshl_b32
}

// ADR 0012 step 1 keeps the folder's pre-IR domain (ScalarOpDef::fold): defined operations outside
// it are emitted through the same definitions but never specialise a branch.
TEST(IrScalarFold, DefinitionsOutsideTheFoldDomainDoNotFold) {
    EXPECT_EQ(fold(mov(20, 0xffffffffu) + mov(21, 1) + std::vector{sop2(0x06, 22, 20, 21)} +
                   is(0xffffffffu)),
              std::nullopt);   // s_min_i32
    EXPECT_EQ(fold(mov(20, 5) + std::vector{sop1(0x07, 22, 20)} + is(~5u)), std::nullopt);
    EXPECT_EQ(fold(std::vector{sopk(0x00, 22, 0xffff)} + is(0xffffffffu)), std::nullopt);
    EXPECT_EQ(fold(mov(20, 5) + std::vector{sopc(0x0d, 20, kInline0)}), std::nullopt);   // bitcmp1
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
    EXPECT_EQ(fold(mov(20, 7) + std::vector{sopc(0x03, 20, kInline0 + 7)}), true);   // ge_i32 tie
}
