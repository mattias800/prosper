// #5497: a loop's current Bool mask need not have reconstructible physical SGPR halves.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_scalar_sources.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;

std::vector<uint32_t> loop_program(uint32_t opcode = 0x128, bool dispatcher = false) {
    std::vector<uint32_t> program{
        0x7e040300u,                  // v2 = original lane, output index
        0x7e080300u,                  // v4 = lane, loop counter
        0x7e020280u,                  // v1 = 0, accumulated carry/borrow
        0xbe84047eu,                  // s[4:5] = EXEC
        0xd4c1000eu,
        0x00014104u,    // loop: v_cmp_lt_u32 s[14:15], v4, 32
        0xd4006a01u | (opcode << 16u),
        opcode == 0x129 ? 0x00390101u : 0x003a0280u, // accumulate +/- carry from s[14:15]
        0xd4c1006au,
        0x00018104u,    // v_cmp_lt_u32 vcc, v4, 64
        0x87fe6a7eu,                  // EXEC &= vcc
        0xbf880002u,                  // s_cbranch_execz exit
        0x4a080881u,                  // v4 += 1
        0xbf82fff6u,                  // s_branch loop
        0xbefe0404u,                  // exit: EXEC = s[4:5]
        0xe0702000u,
        0x80020102u,    // output[v2] = v1
        0xbf810000u,
    };
    if (dispatcher) {
        // Retain the compare's Bool through a genuine dispatcher reload before the carry.
        // Both scalar paths preserve the mask; neither needs its physical SGPR halves.
        program.insert(program.begin() + 6, {0xbf068006u, 0xbf840001u, 0x7e0a0280u});
        program[16] = 0xbf82fff3u;
    }
    return program;
}

ComputeShaderConfig config() {
    ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = 64;
    c.user_sgprs.resize(16, 0);
    prosper::test::complete_compute_launch(c);
    return c;
}

ShaderResourceTable resources() {
    ShaderResource out;
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.stride = 4;
    out.sgpr_base = 8;
    ShaderResourceTable table;
    table.resources.push_back(out);
    return table;
}

bool has_op(const std::vector<uint32_t>& module, uint32_t opcode) {
    for (size_t at = 5; at < module.size();) {
        const uint32_t width = module[at] >> 16u;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 65535u) == opcode) return true;
        at += width;
    }
    return false;
}
void check_values(uint32_t opcode, bool dispatcher = false) {
    const auto program = loop_program(opcode, dispatcher);
    auto c = config();
    c.force_exchange_dispatcher = dispatcher;
    const auto table = resources();
    const auto module = recompile_compute(program.data(), program.size(), &table, c,
                                          {RecompileDiagnosticStage::Compute, 0x5497000100ull});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x5497000100ull);
    ASSERT_TRUE(has_op(module, 246)); // OpLoopMerge: the real loop remains present
    ASSERT_EQ(has_op(module, 251),
              dispatcher); // distinguish native regression from exact controls
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = c.user_sgprs;
    const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const uint32_t bits = lane < 32 ? 32u - lane : 0u;
        EXPECT_EQ(out[lane], opcode == 0x128 ? bits : 0u - bits) << "lane " << lane;
    }
}

void refuses(const std::vector<uint32_t>& program, uint64_t address) {
    const auto c = config();
    const auto table = resources();
    EXPECT_TRUE(recompile_compute(program.data(), program.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("reason=fabricated-fresh-mask-source pc=6"), std::string::npos) << reason;
}
} // namespace

TEST(NativeLoopCarryMask, CurrentBoolFeedsCarryInsideExeczLoop) {
    check_values(0x128);
}
TEST(NativeLoopCarryMask, CurrentBoolFeedsBorrowInsideExeczLoop) {
    check_values(0x129);
}
TEST(NativeLoopCarryMask, CurrentBoolFeedsReverseBorrowInsideExeczLoop) {
    check_values(0x12a);
}

TEST(NativeLoopCarryMask, DispatcherExecutesCurrentCarryMask) {
    check_values(0x128, true);
}
TEST(NativeLoopCarryMask, DispatcherExecutesCurrentBorrowMask) {
    check_values(0x129, true);
}
TEST(NativeLoopCarryMask, DispatcherExecutesCurrentReverseBorrowMask) {
    check_values(0x12a, true);
}

TEST(NativeLoopCarryMask, NeverWrittenMaskStillRefuses) {
    auto program = loop_program();
    program[7] = (program[7] & ~(511u << 18u)) | (30u << 18u); // no producer for s[30:31]
    refuses(program, 0x5497000200ull);
}

TEST(NativeLoopCarryMask, MaskPhysicalWordsRemainDataInOtherOperands) {
    auto program = loop_program();
    program[7] =
        (program[7] & ~511u) | 14u; // SRC0 needs s14's physical DATA, even with a live Bool
    refuses(program, 0x5497000300ull);
}

TEST(NativeLoopCarryMask, MergePoisonStillChargesEitherMaskWord) {
    const auto program = loop_program();
    const auto carry = rdna2_decode_one(program.data() + 6, 2);
    RegState rs;
    rs.vector_wave_size = 64;
    rs.vreg[1] = 1;
    rs.sreg_bool[14] = 2;
    rs.unavailable_saved_mask_words = {14, 15};
    EXPECT_FALSE(scalar_source_marks(rs, carry).placeholder);
    for (const int word : {14, 15}) {
        rs.sreg_merge_placeholder = {word};
        EXPECT_TRUE(scalar_source_marks(rs, carry).hard) << "poisoned mask word " << word;
    }
}
