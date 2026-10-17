// The 64-bit word opcodes (PackU64, U64Lo, U64Hi, U64Eq..U64Ge): hand-worked evaluation, Block
// type rules, the 32-bit view of a U64 value, and every opcode's SPIR-V run per lane in the integer
// interpreter against its own evaluation.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/eval_integer64.hpp"
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

constexpr uint64_t kMinI64 = 0x8000000000000000ull, kMaxI64 = 0x7fffffffffffffffull;

TEST(IrCoreInteger64, PackAndUnpackAreHandWorked) {
    EXPECT_EQ(evaluate_integer64(Opcode::PackU64, 0x89abcdefu, 0x01234567u),
              Known64(0x0123456789abcdefull));
    EXPECT_EQ(evaluate_integer64(Opcode::PackU64, 0xffffffffu, 0u), Known64(0xffffffffull))
        << "the low word is not sign-extended into the high one";
    EXPECT_EQ(evaluate_integer64(Opcode::U64Lo, 0x0123456789abcdefull, 0), Known64(0x89abcdefu));
    EXPECT_EQ(evaluate_integer64(Opcode::U64Hi, 0x0123456789abcdefull, 0), Known64(0x01234567u));
    EXPECT_EQ(evaluate_integer64(Opcode::U64Hi, kMinI64, 0), Known64(0x80000000u))
        << "a logical, not arithmetic, shift";
}

TEST(IrCoreInteger64, RelationsAreUnsignedOverAllSixtyFourBits) {
    struct Row {
        uint64_t a, b;
        const char* truth;   // eq ne lt le gt ge
    };
    for (const Row& r : {Row{0x0000000100000000ull, 0x00000000ffffffffull, "010011"},   // high only
                         Row{0x0000000500000007ull, 0x0000000500000007ull, "100101"},
                         Row{0x0000000580000000ull, 0x000000057fffffffull, "010011"},   // low only
                         Row{kMinI64, kMaxI64, "010011"}, Row{0, ~0ull, "011100"}}) {
        const Opcode ops[] = {Opcode::U64Eq, Opcode::U64Ne, Opcode::U64Lt,
                              Opcode::U64Le, Opcode::U64Gt, Opcode::U64Ge};
        for (size_t i = 0; i < std::size(ops); ++i)
            EXPECT_EQ(evaluate_integer64(ops[i], r.a, r.b), Known64(r.truth[i] == '1'))
                << opcode_info(ops[i]).name << " " << std::hex << r.a << " " << r.b;
    }
}

// pack_u64_signed_order flips the sign bit, so unsigned order on the keys is signed order.
TEST(IrCoreInteger64, SignedOrderKeysOrderTwosComplementValues) {
    Block block;
    Builder ir(block);
    const Value alo = ir.input(Type::U32), ahi = ir.input(Type::U32);
    const Value blo = ir.input(Type::U32), bhi = ir.input(Type::U32);
    const Value lt =
        ir.u64lt(ir.pack_u64_signed_order(alo, ahi), ir.pack_u64_signed_order(blo, bhi));
    struct Row {
        uint64_t a, b;
        bool lt;
    };
    for (const Row& r :
         {Row{kMinI64, kMaxI64, true}, Row{kMaxI64, kMinI64, false}, Row{~0ull, 0, true},
          Row{0, ~0ull, false}, Row{0xfffffffeffffffffull, 0xffffffff00000000ull, true},
          Row{kMinI64, 0, true}, Row{kMinI64 + 1, kMinI64, false}}) {
        const std::array<Known, 4> in = {uint32_t(r.a), uint32_t(r.a >> 32), uint32_t(r.b),
                                         uint32_t(r.b >> 32)};
        EXPECT_EQ(evaluate(block, in)[lt.index], Known(r.lt)) << std::hex << r.a << " " << r.b;
    }
}

TEST(IrCoreInteger64, AU64ValueHasNoThirtyTwoBitViewButKeepsItsWideOne) {
    Block block;
    Builder ir(block);
    const Value lo = ir.input(Type::U32), hi = ir.input(Type::U32);
    const Value packed = ir.pack_u64(lo, hi);
    const Value back = ir.u64hi(packed);
    const std::array<Known, 2> in = {0x11111111u, 0x22222222u};
    const auto narrow = evaluate(block, in);
    EXPECT_FALSE(narrow[packed.index]);
    EXPECT_EQ(narrow[back.index], Known(0x22222222u));
    const std::array<Known64, 2> wide_in = {0x11111111u, 0x22222222u};
    EXPECT_EQ(evaluate_wide(block, wide_in)[packed.index], Known64(0x2222222211111111ull));
    const std::array<Known64, 2> too_wide = {0x100000000ull, 0u};
    EXPECT_FALSE(evaluate_wide(block, too_wide)[packed.index]) << "a U32 input above 32 bits";
    const std::array<Known, 2> unknown = {0x11111111u, std::nullopt};
    EXPECT_FALSE(evaluate(block, unknown)[back.index]);
}

TEST(IrCoreInteger64, OperandTypesAreEnforced) {
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            ir.u64lt(ir.c32(1), ir.c32(2));
        },
        "argument type mismatch");
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            const Value wide = ir.pack_u64(ir.c32(1), ir.c32(2));
            ir.iadd(wide, ir.c32(1));
        },
        "argument type mismatch");
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            const Value wide = ir.pack_u64(ir.c32(1), ir.c32(2));
            ir.pack_u64(wide, ir.c32(0));
        },
        "argument type mismatch");
    EXPECT_EQ(opcode_info(Opcode::PackU64).result, Type::U64);
    EXPECT_EQ(opcode_info(Opcode::U64Lo).result, Type::U32);
    EXPECT_EQ(opcode_info(Opcode::U64Ge).result, Type::Bool);
}

// One module per case: lane l computes `block` on inputs base[i] ^ l (the lane in the low bits, so
// nothing can fold), and stores the result word (a Bool as 0/1).
std::vector<uint32_t> run_per_lane(const Block& block, Value result,
                                   std::span<const uint32_t> base) {
    SpirvCompute b;
    static uint64_t next_address = 0x64640000u;
    b.diagnostic = {RecompileDiagnosticStage::Compute, next_address++};
    b.begin(10u, nullptr, 64u, 1u, 1u, 64u, 0u, true, true);
    std::vector<uint32_t> inputs;
    for (const uint32_t word : base)
        inputs.push_back(b.ibin(Op_BitwiseXor, b.linear_localid, b.uconst(word)));
    const auto ids = emit_spirv(b, block, inputs);
    uint32_t word = ids[result.index];
    if (block.at(result).type == Type::Bool) word = b.sel(word, b.uconst(1), b.uconst(0));
    b.store_output_word(word, 1u, 0u);
    bpermute_oracle::Interpreter vm(b.finish());
    auto out = vm.run_buffers(64u, {{1u, std::vector<uint32_t>(64u, 0xdeadbeefu)}}, 1u);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    out.resize(64u);
    return out;
}

TEST(IrCoreInteger64, EveryOpcodeRunsLikeItsEvaluationPerLane) {
    const uint32_t words[] = {0u,          1u,          0x7fffffffu, 0x80000000u,
                              0xffffffffu, 0x12345678u, 0xffffffc0u, 0x0000003fu};
    std::set<Opcode> seen;
    for (uint32_t op = uint32_t(Opcode::PackU64); op <= uint32_t(Opcode::U64Ge); ++op) {
        const Opcode opcode = static_cast<Opcode>(op);
        Block block;
        Builder ir(block);
        const Value a_lo = ir.input(Type::U32), a_hi = ir.input(Type::U32);
        const Value c_lo = ir.input(Type::U32), c_hi = ir.input(Type::U32);
        const Value a = ir.pack_u64(a_lo, a_hi);
        Value result = a;
        if (opcode == Opcode::U64Lo) result = ir.u64lo(a);
        if (opcode == Opcode::U64Hi) result = ir.u64hi(a);
        if (opcode >= Opcode::U64Eq) {
            const Value c = ir.pack_u64(c_lo, c_hi);
            result = block.append({opcode, Type::Bool, 0, {a, c}});
        }
        // A packed word leaves through both halves, so the store reads all 64 bits.
        const Value lo = opcode == Opcode::PackU64 ? ir.u64lo(a) : result;
        const Value hi = opcode == Opcode::PackU64 ? ir.u64hi(a) : result;
        for (const Inst& inst : block.insts()) seen.insert(inst.op);
        for (const uint32_t x : words)
            for (const uint32_t y : words) {
                // High words equal half the time, so the low words decide those cases.
                const std::array<uint32_t, 4> base = {x, y, x ^ y, (x & 1u) ? y : x};
                for (const Value stored : {lo, hi}) {
                    const auto run = run_per_lane(block, stored, base);
                    ASSERT_EQ(run.size(), 64u);
                    for (uint32_t lane = 0; lane < 64u; ++lane) {
                        const std::array<Known, 4> in = {base[0] ^ lane, base[1] ^ lane,
                                                         base[2] ^ lane, base[3] ^ lane};
                        const Known want = evaluate(block, in)[stored.index];
                        ASSERT_TRUE(want.has_value());
                        ASSERT_EQ(run[lane], *want)
                            << opcode_info(opcode).name << " lane " << lane << std::hex
                            << " in "
                            // NOLINTNEXTLINE(bugprone-unchecked-optional-access): all inputs set above.
                            << *in[0] << " " << *in[1] << " " << *in[2] << " " << *in[3];
                    }
                    if (lo == hi) break;
                }
            }
    }
    for (uint32_t op = uint32_t(Opcode::PackU64); op <= uint32_t(Opcode::U64Ge); ++op)
        EXPECT_TRUE(seen.contains(static_cast<Opcode>(op))) << opcode_info(Opcode(op)).name;
}

}   // namespace
