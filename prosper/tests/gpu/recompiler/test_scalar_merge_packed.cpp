// Packed ALU must charge its actual scalar operands without keeping unrelated masks live.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "fixtures/compute_runner.h"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <bit>
#include <cstdint>
#include <vector>

namespace {
using namespace prosper::gpu;
using Words = std::vector<uint32_t>;
constexpr uint32_t end = 0xbf810000u;
constexpr uint64_t replacement = 0xfffffffffffffffeull;
constexpr uint32_t untouched = 0xdeadbeefu;
Words program(bool read_old_mask = false) {
    Words code{
        0xbf068014u, 0xbf840001u,
        0xbe980389u, // conditional LOW write retires only one Bool edge
        0xcc0e4004u, 0x1c0c5501u, // v4=pk_fma(v1,s42,v3), before the complete replacement
        0xbe9804c2u, // s[24:25]=-2 replaces both physical words and their predicate
        0xd5010000u, 129u | (137u << 9u) | (24u << 18u),
        end,
    };
    if (read_old_mask) code[4] = (code[4] & ~(0x1ffu << 9u)) | (24u << 9u);
    return code;
}
std::vector<Rdna2Inst> decode(const Words& words) {
    std::vector<Rdna2Inst> ins;
    EXPECT_EQ(rdna2_walk(words.data(), words.size(), ins), words.size());
    return ins;
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    explicit Emitter(bool native) {
        b.begin(4, nullptr, 64, 1, 1, 64, 21, true, true);
        b.diagnostic = {RecompileDiagnosticStage::Compute, 0x5521cc8000ull};
        b.native_subgroup_size = native ? 64 : 0;
        ComputeShaderConfig config;
        config.local_x = config.wave_size = 64;
        config.native_subgroup_size = native ? 64 : 0;
        prosper::test::complete_compute_launch(config);
        prosper::test::bind_checked_compute_launch(b, config);
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.scc = b.bfalse();
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.sreg[20] = b.load_push_constant(20);
        rs.sreg_bool[24] = constant_mask_bit(b, 0x12345678u, 0xabcdef01u);
        rs.sreg[42] = b.uconst(0x3c003c00u);
        rs.vreg[1] = b.uconst(0x40004000u);
        rs.vreg[3] = b.uconst(0x38003800u);
    }
    bool emit(const Words& words) {
        return emit_body(
            b, rs, decode(words), {}, nullptr, true, false,
            [](RegState&, const Rdna2Inst&) { return true; }, words.data(), words.size());
    }
    Words finish() {
        b.store_output_word(rs.vreg.at(0), 2, 0);
        b.store_output_word(rs.vreg.at(4), 2, 1);
        return b.finish();
    }
};
Words compile(bool native, bool read_old_mask = false) {
    return recompile_with_arbitrary_placeholders(
        {RecompileDiagnosticStage::Compute, 0x5521cc8000ull}, [&] {
            Emitter e(native);
            if (!e.emit(program(read_old_mask))) return Words{};
            return e.finish();
        });
}
Words expected() {
    Words result;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        result.push_back(replacement & (uint64_t{1} << lane) ? 9u : 1u);
        result.push_back(0x41004100u); // packed (2.5,2.5)
    }
    return result;
}
void gpu(bool native) {
    for (uint32_t branch : {0u, 1u}) {
        SCOPED_TRACE(branch);
        const auto module = compile(native);
        ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x5521cc8000ull);
        Words push(21);
        push[20] = branch;
        const auto output = prosper::test::run_compute(
            module, std::vector<float>(64, std::bit_cast<float>(branch)), 64, 128, {}, {}, nullptr,
            64, nullptr, nullptr, nullptr, native ? 64 : 0, nullptr, push);
        Words actual;
        for (float value : output) actual.push_back(std::bit_cast<uint32_t>(value));
        EXPECT_EQ(actual, expected());
    }
}
} // namespace

TEST(ScalarMergePacked, CompleteReplacementExecutesBothArmsInTheInterpreter) {
    for (bool native : {false, true})
        for (uint32_t branch : {0u, 1u}) {
            SCOPED_TRACE(native);
            SCOPED_TRACE(branch);
            const auto module = compile(native);
            ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x5521cc8000ull);
            bpermute_oracle::Interpreter vm(module);
            vm.native_subgroup_width = 64;
            vm.half_rte_preserve_denorm = true;
            vm.supplied_push_constants = Words(21);
            vm.supplied_push_constants[20] = branch;
            const auto actual =
                vm.run_buffers(64, {{0, Words(64, branch)}, {1, Words(128, untouched)}}, 1);
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            EXPECT_EQ(actual, expected());
        }
}

TEST(ScalarMergePacked, PackedReadBeforeReplacementKeepsTheMissingMaskLive) {
    for (bool native : {false, true}) {
        EXPECT_TRUE(compile(native, true).empty());
        EXPECT_NE(last_terminal_reject_reason(0x5521cc8000ull)
                      .find("structured-saved-mask-domain-missing-edge"),
                  std::string::npos);
    }
}

TEST(ScalarMergePacked, OnlyTheActualPackedScalarWordIsRead) {
    const Words words{0xcc0e4004u, 0x1c0c5501u, end};
    const auto ins = decode(words);
    EXPECT_FALSE(sgpr_dead_at_merge(ins, 0, 42));
    EXPECT_TRUE(sgpr_dead_at_merge(ins, 0, 43)) << "the packed source occupies one word";
    EXPECT_TRUE(sgpr_dead_at_merge(ins, 0, 24));
}

TEST(ScalarMergePacked, AnUnknownPackedOpcodeIsNotTransparent) {
    auto ins = decode({0xcc0e4004u, 0x1c0c5501u, end});
    ASSERT_EQ(ins.size(), 2u);
    ins.front().opcode = 0x7f;
    ScalarMergeBlocker blocker;
    EXPECT_FALSE(sgpr_dead_at_merge(ins, 0, 24, ScalarMergeProof::AnyRead, &blocker));
    EXPECT_STREQ(blocker.kind, "unmodelled-vop3p");
}

TEST(ScalarMergePacked, PortableWave64ExecutesOnTheGpu) {
    gpu(false);
}

TEST(ScalarMergePacked, RequiredWave64ExecutesOnTheGpu) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 compute unavailable";
    gpu(true);
}
