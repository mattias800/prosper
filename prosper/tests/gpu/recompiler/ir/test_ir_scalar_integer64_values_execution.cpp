// Existing scalar-pair numeric values: direct per-lane definition execution and genuine uniform
// SALU programs. The latter execute with partial EXEC, then restore it before low/high/SCC reads.
#include "gpu/recompiler/ir/scalar_integer64_value_harness.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/ir/ops/scalar_integer64_values.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/ir/eval_integer64.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
uint32_t expected_word(const ScalarInteger64Def& def, const Scalar64Case& x, uint32_t output) {
    const uint64_t result = scalar64_reference(def, x);
    if (output < 2) return output ? uint32_t(result >> 32) : uint32_t(result);
    if (rdna2_writes_scc(def.fmt, def.opcode) == Rdna2Answer::Yes) return result != 0;
    return x.scc;
}
std::vector<uint32_t> outputs(const ScalarInteger64Def& def) {
    const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
    std::vector<uint32_t> out;
    out.reserve(effects.sdst_dwords + 1);
    for (uint32_t word = 0; word < effects.sdst_dwords; ++word) out.push_back(word);
    out.push_back(2);
    return out;
}
void check_real(const ScalarInteger64Def& def, const Scalar64Case& x, uint32_t wave,
                uint32_t output, uint32_t form, bool restore = true) {
    const auto in = lane_inputs(0x5350);
    const auto run =
        run_vector_op_per_lane(scalar64_program(def, x, output, form, restore), in, wave);
    if (def.fmt == Rdna2Format::SOP2 && def.opcode == kSop2OpcodeBfeU64 &&
        ((scalar64_form_case(x, form).other_lo >> 16) & 127) == 0) {
        // The unchanged helper selects zero after an unused shift by 64. This strict VM
        // rejects that intermediate eagerly; retain the boundary rather than alter the oracle.
        ASSERT_EQ(run.error, "poison shift");
        return;
    }
    ASSERT_TRUE(run.error.empty()) << run.error << " op " << def.opcode << " output " << output;
    ASSERT_EQ(run.d.size(), in.lanes());
    const auto expected = expected_word(def, scalar64_form_case(x, form), output);
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        ASSERT_EQ(run.d[lane], restore || in.active[lane] ? expected : in.old_d[lane])
            << " lane " << lane << " wave " << wave << " op " << def.opcode;
}
}   // namespace
TEST(IrScalarInteger64ValuesExecution, EveryDefinitionRunsPerLaneLikeEvalAndIndependentMath) {
    for (const auto& def : scalar_integer64_value_defs())
        for (uint32_t seed = 0; seed < 6; ++seed) {
            SCOPED_TRACE(rdna2_instruction_effects(def.fmt, def.opcode).name);
            SpirvCompute b;
            b.begin(10, nullptr, 64, 1, 1, 64, 0, true, true);
            std::array<uint32_t, 5> ids{};
            for (uint32_t i = 0; i < 4; ++i)
                ids[i] =
                    b.ibin(Op_BitwiseXor, b.linear_localid, b.uconst(0x5350u * (seed + 1 + i)));
            const bool bfe = def.fmt == Rdna2Format::SOP2 && def.opcode == kSop2OpcodeBfeU64;
            if (bfe) ids[2] = b.ibin(Op_BitwiseOr, ids[2], b.uconst(1u << 16));
            ids[4] = b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, b.linear_localid, b.uconst(1)),
                            b.uconst(0));
            const auto result = emit_scalar_integer64_value(
                b, def, [&](Scalar64Source s) { return ids[size_t(s)]; });
            Block block;
            Builder ir(block);
            std::vector<Scalar64Source> slots;
            const auto shape = def.build(ir, Scalar64Inputs(ir, slots, true));
            uint32_t lo = result.first, hi = result.second;
            if (block.at(shape.first).type == Type::U64) {
                lo = b.u64_lo(lo);
                hi = b.u64_hi(result.first);
            } else if (block.at(shape.first).type == Type::Bool)
                lo = b.sel(lo, b.uconst(1), b.uconst(0));
            if (!hi) hi = b.uconst(0);
            const auto active = b.ucmp(
                Op_INotEqual, b.ibin(Op_BitwiseAnd, b.linear_localid, b.uconst(3)), b.uconst(0));
            b.store_output_word(lo, 2, 0, active);
            b.store_output_word(hi, 2, 1, active);
            bpermute_oracle::Interpreter vm(b.finish());
            const auto out = vm.run_buffers(64, {{1, std::vector<uint32_t>(128, 0xdeadbeef)}}, 1);
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            ASSERT_EQ(out.size(), 128u);
            for (uint32_t lane = 0; lane < 64; ++lane) {
                Scalar64Case x{lane ^ (0x5350u * (seed + 1)), lane ^ (0x5350u * (seed + 2)),
                               lane ^ (0x5350u * (seed + 3)), lane ^ (0x5350u * (seed + 4)),
                               (lane & 1) != 0};
                if (bfe) x.other_lo |= 1u << 16;
                const auto eval = eval_scalar_integer64_value(def, scalar64_inputs(x));
                ASSERT_TRUE(eval.first);
                const uint64_t reference = scalar64_reference(def, x);
                if (shape.second.valid()) ASSERT_TRUE(eval.second);
                const uint64_t evaluated = *eval.first | (eval.second.value_or(0) << 32);
                ASSERT_EQ(evaluated, reference) << " lane " << lane;
                ASSERT_EQ(out[size_t(2) * lane], (lane & 3) ? uint32_t(reference) : 0xdeadbeef);
                ASSERT_EQ(out[size_t(2) * lane + 1],
                          (lane & 3) ? uint32_t(reference >> 32) : 0xdeadbeef);
            }
        }
}
TEST(IrScalarInteger64ValuesExecution, RealUniformScalarWordsBothHalvesAndSccInBothWaves) {
    const auto cases = scalar64_cases();
    for (const auto& def : scalar_integer64_value_defs())
        for (uint32_t wave : {64u, 32u})
            for (const auto& x : cases)
                for (uint32_t output : outputs(def)) check_real(def, x, wave, output, 0);
}
TEST(IrScalarInteger64ValuesExecution, SourceDestinationAliasesKeepBothOriginalWords) {
    const auto cases = scalar64_cases();
    for (const auto& def : scalar_integer64_value_defs())
        for (uint32_t form : {1u, 2u})
            for (uint32_t wave : {64u, 32u})
                for (uint32_t output : outputs(def))
                    check_real(def, cases[form + 2], wave, output, form);
}
TEST(IrScalarInteger64ValuesExecution, VectorReadbackRemainsPredicatedAndScalarResultDoesNot) {
    const auto x = scalar64_cases()[5];
    for (const auto& def : scalar_integer64_value_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t output : outputs(def)) check_real(def, x, wave, output, 0, false);
}
TEST(IrScalarInteger64ValuesExecution, HandWorkedIsaCasesAreIndependentOfEval) {
    struct Row {
        Rdna2Format fmt;
        uint32_t op;
        Scalar64Case x;
        uint64_t want;
    };
    using F = Rdna2Format;
    // 70648: S_BFE width/offset fields are seven/six bits; FLBIT counts leading zero bits;
    // BITREPLICATE copies each source bit twice. Equality compares both unsigned raw words.
    for (const auto& r :
         {Row{F::SOP2, 0x29, {0x89abcdef, 0x01234567, 0x100004, 0, false}, 0xbcde},
          Row{F::SOP2, 0x29, {0x89abcdef, 0x01234567, 0x200020, 0, false}, 0x01234567},
          Row{F::SOP2, 0x29, {0x89abcdef, 0x01234567, 0x24001c, 0, false}, 0x012345678ull},
          Row{F::SOP2, 0x29, {0x76543210, 0xfedcba98, 0x7f003f, 0, false}, 1},
          Row{F::SOP2, 0x29, {0xffffffff, 0xffffffff, 0, 0, false}, 0},
          Row{F::SOP2, 0x29, {0xffffffff, 0xffffffff, 0x7f0000, 0, false}, ~uint64_t(0)},
          Row{F::SOP1, 0x10, {0xffffffff, 0xffffffff, 0, 0, false}, 64},
          Row{F::SOP1, 0x10, {0x80000000, 1, 0, 0, false}, 2},
          Row{F::SOP1, 0x16, {0, 0, 0, 0, false}, 0xffffffff},
          Row{F::SOP1, 0x16, {1, 0, 0, 0, false}, 63},
          Row{F::SOP1, 0x16, {0x80000000, 0, 0, 0, false}, 32},
          Row{F::SOP1, 0x16, {0xffffffff, 1, 0, 0, false}, 31},
          Row{F::SOP1, 0x16, {0, 0x80000000, 0, 0, false}, 0},
          Row{F::SOP1, 0x3b, {1, 0, 0, 0, false}, 3},
          Row{F::SOP1, 0x3b, {0x80000000, 0, 0, 0, false}, 0xc000000000000000ull},
          Row{F::SOP1, 0x3b, {0x00010001, 0, 0, 0, false}, 0x0000000300000003ull},
          Row{F::SOPC, 0x12, {7, 5, 7, 5, false}, 1},
          Row{F::SOPC, 0x12, {7, 5, 7, 6, false}, 0},
          Row{F::SOPC, 0x13, {7, 5, 7, 6, false}, 1},
          Row{F::SOPC, 0x13, {7, 5, 7, 5, false}, 0},
          Row{F::SOP2, 0xb, {1, 2, 3, 4, false}, 0x0000000400000003ull},
          Row{F::SOP2, 0xb, {1, 2, 3, 4, true}, 0x0000000200000001ull}}) {
        const auto* def = find_scalar_integer64_value_op(r.fmt, r.op);
        ASSERT_NE(def, nullptr);
        const auto got = eval_scalar_integer64_value(*def, scalar64_inputs(r.x));
        ASSERT_TRUE(got.first);
        ASSERT_EQ(*got.first | (got.second.value_or(0) << 32), r.want);
        ASSERT_EQ(scalar64_reference(*def, r.x), r.want);
        for (uint32_t output : outputs(*def)) check_real(*def, r.x, 64, output, 0);
    }
}
TEST(IrScalarInteger64ValuesExecution, EffectsDomainDoesNotInventScalarOrMaskOperations) {
    ASSERT_EQ(scalar_integer64_value_defs().size(), 7u);
    std::set<std::pair<Rdna2Format, uint32_t>> seen;
    for (const auto& def : scalar_integer64_value_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        ASSERT_TRUE(e.known());
        EXPECT_TRUE(seen.insert({def.fmt, def.opcode}).second);
        EXPECT_EQ(e.m0_relative, Rdna2M0Relative::None);
        EXPECT_EQ(e.exec, Rdna2ExecEffect::None);
        EXPECT_GE(e.ssrc_dwords[0], 1);
        EXPECT_LE(e.sdst_dwords, 2);
        EXPECT_EQ(find_scalar_integer64_value_op(def.fmt, def.opcode), &def);
    }
    EXPECT_EQ(find_scalar_integer64_value_op(Rdna2Format::SOP1, 0x15), nullptr);
    EXPECT_EQ(find_scalar_integer64_value_op(Rdna2Format::SOP1, 0x17), nullptr);
    EXPECT_EQ(find_scalar_integer64_value_op(Rdna2Format::SOP2, 0x24), nullptr);
}
TEST(IrScalarInteger64ValuesExecution, RequiredUnknownInputsStayUnknown) {
    for (const auto& def : scalar_integer64_value_defs()) {
        Block block;
        Builder ir(block);
        std::vector<Scalar64Source> slots;
        const auto shape = def.build(ir, Scalar64Inputs(ir, slots, true));
        const std::set<Scalar64Source> used(slots.begin(), slots.end());
        for (const auto source : used) {
            auto in = scalar64_inputs(scalar64_cases()[5]);
            in[size_t(source)] = std::nullopt;
            const auto out = eval_scalar_integer64_value(def, in);
            EXPECT_FALSE(out.first && (!shape.second.valid() || out.second))
                << "unknown required source has no complete result";
        }
    }
}
TEST(IrScalarInteger64ValuesExecution, ZeroWidthBfeRetainsTheStrictVmBoundary) {
    const auto* def = find_scalar_integer64_value_op(Rdna2Format::SOP2, kSop2OpcodeBfeU64);
    ASSERT_NE(def, nullptr);
    const Scalar64Case zero{0x89abcdef, 0x01234567, 0, 0, false};
    const auto value = eval_scalar_integer64_value(*def, scalar64_inputs(zero));
    EXPECT_EQ(value.first, Known64(0));
    const auto in = lane_inputs(0x5350);
    const auto run = run_vector_op_per_lane(scalar64_program(*def, zero, 0), in);
    EXPECT_EQ(run.error, "poison shift");
    Scalar64Case nonzero = zero;
    nonzero.other_lo = 0x00100004;
    check_real(*def, nonzero, 64, 0, 0);
}
TEST(IrScalarInteger64ValuesExecution, BfePrimitiveBoundsTypesAndRawMsbZeroStayExplicit) {
    EXPECT_EQ(evaluate_integer64(Opcode::U64Bfe, 0xffffffffffffffffull, 63, 127), Known64(1));
    EXPECT_EQ(evaluate_integer64(Opcode::U64Bfe, 0x0123456789abcdefull, 0, 64),
              Known64(0x0123456789abcdefull));
    EXPECT_EQ(evaluate_integer64(Opcode::U64Bfe, 1, 0, 0), Known64(0));
    EXPECT_FALSE(evaluate_integer64(Opcode::U64Bfe, 1, 64, 1));
    EXPECT_FALSE(evaluate_integer64(Opcode::U64Bfe, 1, 0, 0x100000000ull));
    EXPECT_EQ(evaluate_op(Opcode::FindUMsbU32, 0, 0, 0, 0), Known(UINT32_MAX));
    EXPECT_EQ(evaluate_op(Opcode::FindUMsbU32, 0, 1, 0, 0), Known(0));
    EXPECT_EQ(evaluate_op(Opcode::FindUMsbU32, 0, 0x80000000, 0, 0), Known(31));
    EXPECT_EQ(evaluate_op(Opcode::BoolNot, 0, 0, 0, 0), Known(1));
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block block;
            Builder ir(block);
            ir.u64bfe(ir.c32(1), ir.c32(0), ir.c32(1));
        },
        "argument type mismatch");
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block block;
            Builder ir(block);
            const auto wide = ir.pack_u64(ir.c32(1), ir.c32(0));
            ir.u64bfe(wide, wide, ir.c32(1));
        },
        "argument type mismatch");
}
TEST(IrScalarInteger64ValuesExecution, InlineAndLiteralSourcesKeepExistingExtension) {
    const auto x = scalar64_cases()[5];
    for (const auto& def : scalar_integer64_value_defs())
        for (uint32_t form : scalar64_forms(def))
            if (form >= 3)
                for (uint32_t wave : {64u, 32u})
                    for (uint32_t output : outputs(def)) check_real(def, x, wave, output, form);
}
// GLSL.std.450 instruction75 explicitly returns all bits set for zero. The scalar
// FLBIT route keeps its old OR1 safeguard; the raw IR primitive still models the builtin itself.
TEST(IrScalarInteger64ValuesExecution, RawFindUMsbZeroOneAndHighBitRunAsSpecified) {
    Block unknown_block;
    Builder unknown_ir(unknown_block);
    const auto unknown_result = unknown_ir.find_umsb_u32(unknown_ir.input(Type::U32));
    const Known unknown_input[] = {std::nullopt};
    EXPECT_FALSE(evaluate(unknown_block, unknown_input)[unknown_result.index]);
    struct Row {
        uint32_t input, expected;
    };
    for (const auto row : {Row{0, UINT32_MAX}, Row{1, 0}, Row{0x80000000u, 31},
                           Row{0xffffffffu, 31}, Row{0x01000000u, 24}}) {
        Block block;
        Builder ir(block);
        const auto value = ir.find_umsb_u32(ir.input(Type::U32));
        SpirvCompute b;
        b.begin(10, nullptr, 64, 1, 1, 64, 0, true, true);
        const uint32_t input[] = {b.uconst(row.input)};
        const auto ids = emit_spirv(b, block, input);
        b.store_output_word(ids[value.index], 1, 0);
        bpermute_oracle::Interpreter vm(b.finish());
        const auto out = vm.run_buffers(64, {{1, std::vector<uint32_t>(64, 0xdeadbeef)}}, 1);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(out.size(), 64u);
        for (const auto word : out) ASSERT_EQ(word, row.expected);
    }
}
TEST(IrScalarInteger64ValuesExecution, ExportsAllNumericReadbackAndAliasForms) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const auto root = compare_witness_root("scalar-integer64-values");
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error);
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const auto dump = compare_witness_dump("scalar-integer64-values");
    ASSERT_TRUE(fs::create_directories(dump));
    const auto cases = scalar64_cases();
    size_t count = 0, executed = 0, strict_zero_boundary = 0;
    for (const auto& def : scalar_integer64_value_defs())
        for (uint32_t form : scalar64_forms(def))
            for (uint32_t output : outputs(def))
                for (uint32_t wave : {64u, 32u}) {
                    const auto x = cases[(form % 3) + 2];
                    const auto words = scalar64_program(def, x, output, form);
                    check_real(def, x, wave, output, form);
                    const auto lp = lane_program(words, 64, wave);
                    const uint64_t address = 0x5350000000ull + count * 0x100;
                    write_compare_witness_program(dump, address, count, lp);
                    const auto stem =
                        dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
                    std::vector<uint32_t> saved_code(lp.code.size());
                    std::ifstream raw(stem.string() + ".bin", std::ios::binary);
                    raw.read(reinterpret_cast<char*>(saved_code.data()),
                             static_cast<std::streamsize>(saved_code.size() * sizeof(uint32_t)));
                    ASSERT_EQ(raw.gcount(),
                              static_cast<std::streamsize>(lp.code.size() * sizeof(uint32_t)));
                    ASSERT_EQ(saved_code, lp.code);
                    const auto saved = read_refused_shader_inputs(stem.string() + ".inputs");
                    ASSERT_TRUE(saved.incomplete.empty()) << saved.incomplete;
                    const auto replay = replay_refused_shader_inputs(saved_code, saved);
                    ASSERT_TRUE(replay.produced) << replay.reject;
                    ASSERT_FALSE(replay.spirv.empty());
                    std::ofstream spv(stem.string() + ".spv", std::ios::binary);
                    spv.write(reinterpret_cast<const char*>(replay.spirv.data()),
                              static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
                    spv.close();
                    ASSERT_TRUE(spv.good());
                    const auto in = lane_inputs(0x5350);
                    std::vector<uint32_t> buffer;
                    for (const auto* a : {&in.s0, &in.s1, &in.old_d})
                        buffer.insert(buffer.end(), a->begin(), a->end());
                    for (const bool active : in.active) buffer.push_back(active ? 1u : 0u);
                    bpermute_oracle::Interpreter vm(replay.spirv);
                    const auto out = vm.run_buffers(64, {{3, buffer}}, 3);
                    const auto effective = scalar64_form_case(x, form);
                    if (def.fmt == Rdna2Format::SOP2 && def.opcode == kSop2OpcodeBfeU64 &&
                        ((effective.other_lo >> 16) & 127) == 0) {
                        ASSERT_EQ(vm.error, "poison shift");
                        ++strict_zero_boundary;
                    } else {
                        ASSERT_TRUE(vm.error.empty()) << vm.error;
                        ASSERT_GE(out.size(), 64u);
                        const auto expected = expected_word(def, effective, output);
                        for (size_t lane = 0; lane < 64; ++lane) ASSERT_EQ(out[lane], expected);
                        ++executed;
                    }
                    ++count;
                }
    EXPECT_EQ(count, 228u);
    EXPECT_EQ(executed, 222u);
    EXPECT_EQ(strict_zero_boundary, 6u);
}
