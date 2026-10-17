// The binary64 relations (F64Lt .. F64UnordGe over U64 words): hand-worked IEEE binary64 truth,
// Block type rules, the lazy Float64/DenormPreserve declaration, and every opcode's SPIR-V run per
// lane in the interpreter against its own evaluation.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/eval_float64.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "support/death_test.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <set>
#include <span>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;

namespace {

constexpr uint64_t kPosZero = 0, kNegZero = 0x8000000000000000ull;
constexpr uint64_t kOne = 0x3ff0000000000000ull, kNegOne = 0xbff0000000000000ull;
constexpr uint64_t kOneUlp = 0x3ff0000000000001ull;   // differs from 1.0 in the low word only
constexpr uint64_t kPosInf = 0x7ff0000000000000ull, kNegInf = 0xfff0000000000000ull;
constexpr uint64_t kQnan = 0x7ff8000000000000ull, kSnan = 0x7ff0000000000001ull;
constexpr uint64_t kNegQnan = 0xfff8000000000000ull;
constexpr uint64_t kMinSub = 1, kNegMinSub = 0x8000000000000001ull;
constexpr uint64_t kMaxSub = 0x000fffffffffffffull, kMinNormal = 0x0010000000000000ull;
constexpr uint64_t kMax = 0x7fefffffffffffffull;

constexpr Opcode kOps[] = {Opcode::F64Lt,      Opcode::F64Eq,      Opcode::F64Le,
                           Opcode::F64Gt,      Opcode::F64Lg,      Opcode::F64Ge,
                           Opcode::F64UnordLt, Opcode::F64UnordEq, Opcode::F64UnordLe,
                           Opcode::F64UnordGt, Opcode::F64UnordNe, Opcode::F64UnordGe};

// AMD 70648 12.9 Table 61 and IEEE 754: truth of lt eq le gt lg ge, then unordered
// lt eq le gt ne ge. Worked by hand from the binary64 encodings, never from eval.
struct Row {
    uint64_t a, b;
    const char* truth;
};
constexpr Row kRows[] = {
    {kPosZero, kNegZero, "011001011001"},     // +0 == -0
    {kOne, kQnan, "000000111111"},            // a NaN: ordered false, unordered true
    {kQnan, kQnan, "000000111111"},
    {kSnan, kOne, "000000111111"},            // a signalling NaN is a NaN too
    {kNegQnan, kNegInf, "000000111111"},
    {kNegInf, kPosInf, "101010101010"},
    {kMax, kPosInf, "101010101010"},          // infinity is the extreme of its sign
    {kPosInf, kPosInf, "011001011001"},
    {kMinSub, kPosZero, "000111000111"},      // a preserved subnormal is not zero
    {kNegMinSub, kPosZero, "101010101010"},
    {kMaxSub, kMinNormal, "101010101010"},    // the subnormal/normal boundary
    {kOne, kOneUlp, "101010101010"},          // only the low word decides
    {kNegOne, kNegOne, "011001011001"},
    {kNegOne, kOne, "101010101010"},
};

TEST(IrCoreFloat64, RelationsAreHandWorkedIeeeBinary64) {
    for (const Row& r : kRows)
        for (size_t i = 0; i < std::size(kOps); ++i)
            EXPECT_EQ(evaluate_float64(kOps[i], r.a, r.b), Known64(r.truth[i] == '1'))
                << opcode_info(kOps[i]).name << " " << std::hex << r.a << " " << r.b;
    EXPECT_FALSE(evaluate_float64(Opcode::U64Lt, 0, 1)) << "not a binary64 opcode";
}

TEST(IrCoreFloat64, OperandTypesAreEnforcedAndResultsAreBool) {
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            ir.f64lt(ir.c32(1), ir.c32(2));
        },
        "argument type mismatch");
    for (const Opcode op : kOps) {
        EXPECT_EQ(opcode_info(op).result, Type::Bool);
        EXPECT_EQ(opcode_info(op).arity, 2u);
        EXPECT_TRUE(is_float64_opcode(op));
    }
    EXPECT_FALSE(is_float64_opcode(Opcode::U64Ge));
}

TEST(IrCoreFloat64, AFloat64RelationEvaluatesThroughTheWideView) {
    Block block;
    Builder ir(block);
    const Value alo = ir.input(Type::U32), ahi = ir.input(Type::U32);
    const Value blo = ir.input(Type::U32), bhi = ir.input(Type::U32);
    const Value lt = ir.f64lt(ir.pack_u64(alo, ahi), ir.pack_u64(blo, bhi));
    const std::array<Known, 4> in = {0u, 0x3ff00000u, 1u, 0x3ff00000u};
    EXPECT_EQ(evaluate(block, in)[lt.index], Known(1u));
    const std::array<Known, 4> unknown = {0u, std::nullopt, 1u, 0x3ff00000u};
    EXPECT_FALSE(evaluate(block, unknown)[lt.index]);
}

// One module per case, as test_ir_core_integer64: lane l computes `block` on inputs base[i] ^ l.
std::vector<uint32_t> run_per_lane(const Block& block, Value result, std::span<const uint32_t> base,
                                   std::vector<uint32_t>* words) {
    SpirvCompute b;
    static uint64_t next_address = 0x64f40000u;
    b.diagnostic = {RecompileDiagnosticStage::Compute, next_address++};
    b.begin(10u, nullptr, 64u, 1u, 1u, 64u, 0u, true, true);
    std::vector<uint32_t> inputs;
    for (const uint32_t word : base)
        inputs.push_back(b.ibin(Op_BitwiseXor, b.linear_localid, b.uconst(word)));
    const auto ids = emit_spirv(b, block, inputs);
    b.store_output_word(b.sel(ids[result.index], b.uconst(1), b.uconst(0)), 1u, 0u);
    const auto module = b.finish();
    if (words) *words = module;
    bpermute_oracle::Interpreter vm(module);
    auto out = vm.run_buffers(64u, {{1u, std::vector<uint32_t>(64u, 0xdeadbeefu)}}, 1u);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    out.resize(64u);
    return out;
}

bool has_instruction(std::span<const uint32_t> module, uint32_t opcode,
                     std::span<const uint32_t> operands) {
    for (size_t at = 5; at < module.size();) {
        const uint32_t length = module[at] >> 16;
        if (!length) return false;
        if ((module[at] & 0xffffu) == opcode && length == operands.size() + 1 &&
            std::equal(operands.begin(), operands.end(), module.begin() + at + 1))
            return true;
        at += length;
    }
    return false;
}

TEST(IrCoreFloat64, EveryOpcodeRunsLikeItsEvaluationPerLane) {
    // High words around the binary64 classes; the lane xor lands in the low bits of every word.
    const uint32_t highs[] = {0u,          0x80000000u, 0x3ff00000u, 0xbff00000u,
                              0x7ff00000u, 0xfff00000u, 0x7ff80000u, 0x000fffffu};
    const uint32_t lows[] = {0u, 0xffffffc0u};
    std::set<Opcode> seen;
    for (const Opcode opcode : kOps) {
        Block block;
        Builder ir(block);
        const Value a_lo = ir.input(Type::U32), a_hi = ir.input(Type::U32);
        const Value c_lo = ir.input(Type::U32), c_hi = ir.input(Type::U32);
        const Value result = block.append(
            {opcode, Type::Bool, 0, {ir.pack_u64(a_lo, a_hi), ir.pack_u64(c_lo, c_hi)}});
        seen.insert(opcode);
        for (const uint32_t x : highs)
            for (const uint32_t y : highs)
                for (const uint32_t lo : lows) {
                    // Equal high words half the time, so the low words decide those cases.
                    const std::array<uint32_t, 4> base = {lo, x, 0u, (lo ? x : y)};
                    const auto run = run_per_lane(block, result, base, nullptr);
                    ASSERT_EQ(run.size(), 64u);
                    for (uint32_t lane = 0; lane < 64u; ++lane) {
                        const std::array<Known, 4> in = {base[0] ^ lane, base[1] ^ lane,
                                                         base[2] ^ lane, base[3] ^ lane};
                        const Known want = evaluate(block, in)[result.index];
                        ASSERT_TRUE(want.has_value());
                        ASSERT_EQ(run[lane], *want)
                            << opcode_info(opcode).name << " lane " << lane << std::hex << " in "
                            << *in[1] << ':' << *in[0] << " " << *in[3] << ':' << *in[2];
                    }
                }
    }
    EXPECT_EQ(seen.size(), std::size(kOps));
}

// Float64 is declared only by a module that uses a binary64 relation, and that module also asks
// for DenormPreserve 64 (one SPV_KHR_float_controls extension); a U64-only module declares none.
TEST(IrCoreFloat64, Float64AndDenormPreserveAreDeclaredOnlyWhenUsed) {
    const auto module_for = [](bool float64) {
        Block block;
        Builder ir(block);
        const Value a = ir.pack_u64(ir.input(Type::U32), ir.input(Type::U32));
        const Value c = ir.pack_u64(ir.input(Type::U32), ir.input(Type::U32));
        const Value result = float64 ? ir.f64lt(a, c) : ir.u64lt(a, c);
        std::vector<uint32_t> words;
        const std::array<uint32_t, 4> base = {0u, 0x3ff00000u, 0u, 0x40000000u};
        run_per_lane(block, result, base, &words);
        return words;
    };
    const auto with = module_for(true), without = module_for(false);
    const uint32_t float64_cap[] = {10u}, denorm_cap[] = {4464u}, int64_cap[] = {11u};
    EXPECT_TRUE(has_instruction(with, 17u, float64_cap));
    EXPECT_TRUE(has_instruction(with, 17u, denorm_cap));
    EXPECT_TRUE(has_instruction(with, 17u, int64_cap));
    EXPECT_FALSE(has_instruction(without, 17u, float64_cap));
    EXPECT_FALSE(has_instruction(without, 17u, denorm_cap));
    EXPECT_TRUE(has_instruction(without, 17u, int64_cap));
    bool mode = false;
    size_t extensions = 0;
    for (size_t at = 5; at < with.size(); at += with[at] >> 16) {
        if ((with[at] & 0xffffu) == 16u && (with[at] >> 16) == 4u && with[at + 2] == 4459u &&
            with[at + 3] == 64u)
            mode = true;   // OpExecutionMode %main DenormPreserve 64
        extensions += (with[at] & 0xffffu) == 10u;
    }
    EXPECT_TRUE(mode);
    EXPECT_EQ(extensions, 1u);
}

// The interpreter refuses a binary64 subnormal when the module did not ask Vulkan to keep it, so
// the per-lane runs above prove the declaration reached the module, not just that values matched.
TEST(IrCoreFloat64, InterpreterRefusesSubnormalsWithoutDenormPreserve) {
    Block block;
    Builder ir(block);
    const Value a = ir.pack_u64(ir.input(Type::U32), ir.input(Type::U32));
    const Value c = ir.pack_u64(ir.input(Type::U32), ir.input(Type::U32));
    const Value result = ir.f64gt(a, c);
    std::vector<uint32_t> words;
    const std::array<uint32_t, 4> base = {1u, 0u, 0u, 0u};   // lane 0: min subnormal > +0
    const auto run = run_per_lane(block, result, base, &words);
    ASSERT_EQ(run.size(), 64u);
    EXPECT_EQ(run[0], 1u);
    // The same module with the mode rewritten to DenormFlushToZero 64 (4460) must refuse.
    for (size_t at = 5; at < words.size(); at += words[at] >> 16)
        if ((words[at] & 0xffffu) == 16u && words[at + 2] == 4459u) words[at + 2] = 4460u;
    bpermute_oracle::Interpreter vm(words);
    vm.run_buffers(64u, {{1u, std::vector<uint32_t>(64u, 0u)}}, 1u);
    EXPECT_NE(vm.error.find("DenormPreserve"), std::string::npos) << vm.error;
}

}   // namespace
