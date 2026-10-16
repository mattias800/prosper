// AMD 70648 12.12 byte permutation: emitted SPIR-V vs shared definitions,
// with three independent per-lane inputs, old D independent of S2, and partial EXEC in both waves.
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <system_error>
#include "gpu/recompiler/ir/ops/vector_permute.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
uint32_t isa_permute(uint32_t high, uint32_t low, uint32_t control) {
    uint32_t out = 0;
    for (uint32_t byte = 0; byte < 4; ++byte) {
        const uint32_t selector = (control >> (8 * byte)) & 255;
        uint32_t value;
        if (selector < 8)
            value = ((selector < 4 ? low : high) >> (8 * (selector & 3))) & 255;
        else
            switch (selector) {
                case 8: value = ((low >> 15) & 1) * 255; break;
                case 9: value = ((low >> 31) & 1) * 255; break;
                case 10: value = ((high >> 15) & 1) * 255; break;
                case 11: value = ((high >> 31) & 1) * 255; break;
                case 12: value = 0; break;
                default: value = 255; break;
            }
        out |= value << (8 * byte);
    }
    return out;
}
LaneInputs ternary_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const uint32_t edges[] = {0u, 1u, 31u, 32u, 33u, 0x7fffffffu, 0x80000000u, 0xffffffffu};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        seed = seed * 1103515245u + 12345u;
        uint32_t value = lane < 16 ? edges[(lane * 3 + 5) % 8] : seed;
        if ((lane & 1u) == 0) value &= 0x07070707u;
        if (value == in.old_d[lane]) value ^= 1;
        in.s2.push_back(value);
    }
    return in;
}
std::vector<uint32_t> words(uint32_t opcode) {
    return {0xd4000000u | (opcode << 16) | kVdst,
            (256u + kVsrc0) | ((256u + kVsrc1) << 9) | ((256u + kVsrc2) << 18)};
}
}   // namespace
namespace {
std::vector<uint32_t> variant_words(uint32_t opcode, uint32_t form) {
    auto code = words(opcode);
    for (uint32_t source = 0; source < 3; ++source)
        if (form == 4 || form == source + 1) {
            const uint32_t shift = 9 * source;
            code[1] = (code[1] & ~(0x1ffu << shift)) | ((128u + 7 + source * 2) << shift);
        }
    return code;
}
}   // namespace
TEST(IrVectorPermuteExecution, EveryDefinitionMatchesEachLaneInBothWaves) {
    for (const auto& def : vector_permute_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(def.fmt, def.opcode).name);
        const auto code = words(def.opcode);
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        EXPECT_EQ(decoded[0].src[2].kind, OperandKind::VGPR);
        EXPECT_EQ(decoded[0].src[2].value, int(kVsrc2));
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) {
                const auto in = ternary_inputs(seed * 5249);
                const auto run = run_vector_op_per_lane(code, in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto want =
                        eval_ternary_integer_op(def, {in.s0[lane], in.s1[lane], in.s2[lane]});
                    if (!want) {
                        ADD_FAILURE() << "unknown lane " << lane;
                        continue;
                    }
                    EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
                        << "lane " << lane << " wave " << wave;
                }
            }
    }
}
TEST(IrVectorPermuteExecution, HarnessSeesAllThreeInputsAndInactiveDestinations) {
    auto in = ternary_inputs(43);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s2[lane] = 0;
        for (uint32_t byte = 0; byte < 4; ++byte)
            in.s2[lane] |= uint32_t((lane + byte) & 7) << (8 * byte);
        if (in.s2[lane] == in.old_d[lane]) in.s2[lane] ^= 1;
    }
    const auto code = words(0x344);
    const auto run = run_vector_op_per_lane(code, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> active;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane]) {
            active.insert(run.d[lane]);
            EXPECT_EQ(run.d[lane], isa_permute(in.s0[lane], in.s1[lane], in.s2[lane]));
            EXPECT_NE(in.s2[lane], in.old_d[lane]);
        } else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    EXPECT_GT(active.size(), 16u);
    EXPECT_GE(inactive, 2u);
    in.s2.pop_back();
    const auto bad = run_vector_op_per_lane(code, in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_EQ(bad.error, "S2 lane count mismatch");
}
TEST(IrVectorPermuteExecution, HandWorkedIsaByteOrderSignSourcesAndConstantFills) {
    struct Example {
        uint32_t a, b, c, want;
    };
    // Selectors0..3 read S1 low-to-high;4..7 read S0 low-to-high.8..11 replicate
    // S1 bit15, S1 bit31, S0 bit15 and S0 bit31.12 fills0;13..255 fill255.
    for (auto x : {Example{0x11223344, 0x55667788, 0x03020100, 0x55667788},
                   Example{0x11223344, 0x55667788, 0x07060504, 0x11223344},
                   Example{0x11223344, 0x55667788, 0x00010203, 0x88776655},
                   Example{0x11223344, 0x55667788, 0x04000501, 0x44883377},
                   Example{0x11223344, 0x55667788, 0x0c0c0c0c, 0},
                   Example{0x11223344, 0x55667788, 0x0d0d0d0d, 0xffffffff},
                   Example{0x11223344, 0x55667788, 0xfffefd0d, 0xffffffff},
                   Example{0x11223344, 0x55667788, 0x0b0a0908, 0},
                   Example{0x80ff8000, 0x7fff7f00, 0x0b0a0908, 0xffff0000},
                   Example{0x7fff7f00, 0x80ff8000, 0x0b0a0908, 0x0000ffff},
                   Example{0x80000000, 0x00008000, 0x0b0a0908, 0xff0000ff},
                   Example{0x11223344, 0x55667788, 0x0d0c0800, 0xff000088},
                   Example{0, 0x00008000, 0x08080808, 0xffffffff},
                   Example{0, 0x80000000, 0x09090909, 0xffffffff},
                   Example{0x00008000, 0, 0x0a0a0a0a, 0xffffffff},
                   Example{0x80000000, 0, 0x0b0b0b0b, 0xffffffff}}) {
        const auto* def = find_vector_permute_op(Rdna2Format::VOP3, 0x344);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_ternary_integer_op(*def, {x.a, x.b, x.c}), Known(x.want));
        auto in = ternary_inputs(13);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.b);
        std::fill(in.s2.begin(), in.s2.end(), x.c);
        const auto run = run_vector_op_per_lane(words(0x344), in);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
    }
}
TEST(IrVectorPermuteExecution, EverySelectorInEveryDestinationByteWithBothSignBitsAndWaveSizes) {
    const auto* def = find_vector_permute_op(Rdna2Format::VOP3, 0x344);
    ASSERT_NE(def, nullptr);
    for (uint32_t wave : {64u, 32u})
        for (uint32_t batch = 0; batch < 4; ++batch)
            for (bool flip : {false, true}) {
                auto in = ternary_inputs(71 + batch);
                std::fill(in.active.begin(), in.active.end(), 1);
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    in.s2[lane] = (batch * 64u + static_cast<uint32_t>(lane)) * 0x01010101u;
                    if (flip) {
                        in.s0[lane] ^= 0x80808080;
                        in.s1[lane] ^= 0x80808080;
                    }
                }
                const auto run = run_vector_op_per_lane(words(0x344), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto want = isa_permute(in.s0[lane], in.s1[lane], in.s2[lane]);
                    EXPECT_EQ(run.d[lane], want);
                    EXPECT_EQ(
                        eval_ternary_integer_op(*def, {in.s0[lane], in.s1[lane], in.s2[lane]}),
                        Known(want));
                }
            }
}
TEST(IrVectorPermuteExecution, ExistingClampRefusalRemains) {
    const auto in = ternary_inputs(151);
    for (const auto& def : vector_permute_defs()) {
        const auto good = run_vector_op_per_lane(words(def.opcode), in);
        ASSERT_TRUE(good.error.empty()) << good.error;
        auto code = words(def.opcode);
        code[0] |= 1u << 15;
        const auto bad = run_vector_op_per_lane(code, in);
        EXPECT_TRUE(bad.d.empty());
        EXPECT_TRUE(bad.error.starts_with("refused:")) << bad.error;
    }
}
TEST(IrVectorPermuteExecution, EffectsAndSourceSlotsAgreeWithTheDefinitionDomain) {
    std::set<Opcode> seen;
    for (const auto& def : vector_permute_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_FALSE(e.accumulates);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 3u);
        Block block;
        Builder ir(block);
        std::vector<uint32_t> sources;
        def.build(ir, TernaryInputs(ir, sources));
        EXPECT_EQ(sources.size(), block.input_count());
        EXPECT_EQ(std::set<uint32_t>(sources.begin(), sources.end()),
                  (std::set<uint32_t>{0, 1, 2}));
        for (const auto& inst : block.insts()) seen.insert(inst.op);
    }
    for (auto op : {Opcode::BfeU, Opcode::IAdd, Opcode::IMul, Opcode::And, Opcode::ULt, Opcode::UGt,
                    Opcode::IEq, Opcode::Select, Opcode::Or, Opcode::Shl})
        EXPECT_TRUE(seen.contains(op));
    EXPECT_EQ(find_vector_permute_op(Rdna2Format::VOP2, 0x344), nullptr);
    EXPECT_EQ(find_vector_permute_op(Rdna2Format::VOP3, 0x142), nullptr);
}
TEST(IrVectorPermuteExecution, UnknownDemandedSourceWordsRemainUnknown) {
    const auto* def = find_vector_permute_op(Rdna2Format::VOP3, 0x344);
    ASSERT_NE(def, nullptr);
    EXPECT_FALSE(eval_ternary_integer_op(*def, {std::nullopt, 0x55667788, 0x07060504}));
    EXPECT_FALSE(eval_ternary_integer_op(*def, {0x11223344, std::nullopt, 0x03020100}));
    EXPECT_FALSE(eval_ternary_integer_op(*def, {0x11223344, 0x55667788, std::nullopt}));
}

TEST(IrVectorPermuteExecution, InlineFormsPreserveOperandPositionAndMasking) {
    const auto* def = find_vector_permute_op(Rdna2Format::VOP3, 0x344);
    ASSERT_NE(def, nullptr);
    for (uint32_t form = 1; form < 5; ++form)
        for (uint32_t wave : {64u, 32u}) {
            const auto in = ternary_inputs(81 + form);
            const auto run = run_vector_op_per_lane(variant_words(def->opcode, form), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                const uint32_t a = form == 4 || form == 1 ? 7 : in.s0[lane];
                const uint32_t b = form == 4 || form == 2 ? 9 : in.s1[lane];
                const uint32_t c = form == 4 || form == 3 ? 11 : in.s2[lane];
                const auto want = isa_permute(a, b, c);
                EXPECT_EQ(eval_ternary_integer_op(*def, {a, b, c}), Known(want));
                EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
            }
        }
}

// The recorded corpus did not observe this family's output. Export every existing plain form
// using exactly the execution harness's program and resource/launch bindings, without touching it.
TEST(IrVectorPermuteExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "permute-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    ASSERT_TRUE(fs::create_directories(root));
    const fs::path dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directory(dump));
    size_t count = 0;
    for (const auto& def : vector_permute_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t form = 0; form < 5; ++form) {
                const auto program = lane_program(variant_words(def.opcode, form), 64, wave, true);
                const uint64_t address = 0x5249000000ull + count * 0x100;
                const fs::path stem =
                    dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
                std::ofstream bin(stem.string() + ".bin", std::ios::binary);
                bin.write(reinterpret_cast<const char*>(program.code.data()),
                          static_cast<std::streamsize>(program.code.size() * sizeof(uint32_t)));
                bin.close();
                ASSERT_TRUE(bin.good());
                const fs::path inputs = stem.string() + ".inputs";
                ASSERT_TRUE(write_refused_compute_inputs(inputs, address, &program.table,
                                                         program.config,
                                                         "synthetic-witness (no live refusal)"));
                const auto read = read_refused_shader_inputs(inputs);
                ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
                const auto replay = replay_refused_shader_inputs(program.code, read);
                ASSERT_TRUE(replay.produced) << replay.reject << " op " << def.opcode;
                EXPECT_TRUE(replay.incomplete.empty()) << replay.incomplete;
                ++count;
            }
    EXPECT_EQ(count, 10u);
}
