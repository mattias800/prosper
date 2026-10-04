// A later read-only FLAT/GLOBAL must not turn descriptor assembly into numeric raw data.
// Actual scalar observers still need backing, and uncertain memory effects still block snapshots.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> result;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), result), code.size());
    EXPECT_FALSE(result.empty());
    return result;
}

std::vector<uint32_t> descriptor_patch() {
    return {
        0xf4080200u, 0x28000000u, // x4 s[8:11], entry s[0:1], register offset s20
        0x870bff0bu, 0xfff80000u, // s_and_b32 s11,s11,literal: descriptor format patch
        0xe00c2000u, 0x80020000u, // buffer_load_format_xyzw v[0:3],v0,s[8:11],0 idxen
    };
}

std::vector<uint32_t> with_flat(std::vector<uint32_t> code, uint32_t opcode,
                                uint32_t segment = 2u) {
    code.push_back(0xdc000000u | (opcode << 18u) | (segment << 14u));
    code.push_back(0x087d0009u); // v[8:11],v9,NULL: independent address, overlaps output
    code.push_back(0xbf810000u);
    return code;
}

class RawWideFlatReadEffects : public ::testing::TestWithParam<uint32_t> {};
} // namespace

TEST_P(RawWideFlatReadEffects, DescriptorPatchRemainsDescriptorOnly) {
    auto control = descriptor_patch();
    control.push_back(0xbf810000u);
    EXPECT_TRUE(rdna2_raw_wide_data_loads(decode(control)).empty())
        << "The same descriptor patch without FLAT calibrates descriptor-only routing";
    for (uint32_t segment : {0u, 2u}) {
        SCOPED_TRACE(segment);
        const auto instructions = decode(with_flat(descriptor_patch(), GetParam(), segment));
        ASSERT_EQ(instructions[2].fmt, Rdna2Format::MUBUF);
        ASSERT_EQ(instructions[2].src[1].value, 8);
        ASSERT_EQ(instructions[3].fmt, Rdna2Format::FLAT);
        EXPECT_FALSE(rdna2_may_write_guest_memory(instructions[3]));
        EXPECT_TRUE(rdna2_raw_wide_data_loads(instructions).empty())
            << "A separate read cannot invalidate the descriptor source";
    }
}

TEST_P(RawWideFlatReadEffects, NumericObservationStillNeedsBacking) {
    const auto instructions = decode(with_flat({
        0xf4080200u, 0x28000000u, // same x4 raw load
        0x7e000c09u,             // v_cvt_f32_u32 v0,s9: actual numeric observer
    }, GetParam()));
    EXPECT_EQ(rdna2_raw_wide_data_loads(instructions), std::vector<uint32_t>{0u});
    EXPECT_EQ(rdna2_proven_raw_register_wide_data_loads(instructions),
              std::vector<uint32_t>{0u})
        << "Only the existing unchanged-entry scalar-offset proof can grant backing";
}

INSTANTIATE_TEST_SUITE_P(DocumentedLoads, RawWideFlatReadEffects,
                        ::testing::Values(0x08u, 0x09u, 0x0au, 0x0bu,
                                          0x0cu, 0x0du, 0x0eu, 0x0fu));

TEST(RawWideFlatReadEffects, StoresAtomicsAndUnknownOperationsRemainConservative) {
    for (uint32_t opcode : {0x07u, 0x10u, 0x1cu, 0x30u, 0x7fu}) {
        SCOPED_TRACE(opcode);
        const auto instructions = decode(with_flat(descriptor_patch(), opcode));
        EXPECT_TRUE(rdna2_may_write_guest_memory(instructions[3]));
        EXPECT_EQ(rdna2_raw_wide_data_loads(instructions), std::vector<uint32_t>{0u})
            << "No alias proof permits a descriptor snapshot across a potential writer";
    }
}

TEST(RawWideFlatReadEffects, ReadOnlyGlobalDoesNotInvalidateLatchedNumericParent) {
    const std::vector<uint32_t> parent{
        0xf4080007u, 0xfa000000u, // x4 s[0:3], entry s[14:15], immediate 0
        0xbe8e0380u,             // later reuse of s14
        0x8f6b8403u,             // vcc_hi = numeric s3 << 4
        0x7e000c03u,             // v_cvt_f32_u32 v0,s3: genuine numeric observation
    };
    const auto read_only = decode(with_flat(parent, 0x0eu));
    EXPECT_EQ(rdna2_raw_wide_data_loads(read_only), std::vector<uint32_t>{0u});
    EXPECT_TRUE(rdna2_proven_raw_immediate_wide_data_loads(read_only).empty())
        << "Later pointer reuse remains excluded from the old full-program entry proof";
    EXPECT_EQ(rdna2_owned_raw_wide_data_loads(read_only), std::vector<uint32_t>{0u})
        << "The existing owned read-point proof preserves the original numeric observation";
    const auto writer = decode(with_flat(parent, 0x1cu));
    EXPECT_TRUE(rdna2_owned_raw_wide_data_loads(writer).empty())
        << "Owned snapshots still require the complete no-guest-write proof";
}

// #4422: the raw-wide snapshot guard and the ISA-pinned writer classifier answer the same
// question, so they must agree for every opcode of every memory family. The guard used to keep
// its own four-opcode MIMG reader list and called image_sample_l a writer.
TEST(RawWideFlatReadEffects, GuestWriteGuardAgreesWithTheIsaClassifier) {
    Rdna2Inst instruction;
    for (const auto format : {Rdna2Format::MIMG, Rdna2Format::MUBUF, Rdna2Format::MTBUF,
                              Rdna2Format::SMEM, Rdna2Format::FLAT}) {
        instruction.fmt = format;
        for (uint32_t opcode = 0; opcode < 0x100u; ++opcode) {
            instruction.opcode = opcode;
            EXPECT_EQ(rdna2_may_write_guest_memory(instruction),
                      rdna2_instruction_may_write_memory(instruction))
                << "format=" << static_cast<int>(format) << " opcode=0x" << std::hex << opcode;
        }
    }
    instruction.fmt = Rdna2Format::MIMG;
    instruction.opcode = 0x24u;   // image_sample_l: Kena's vertex programs read through it
    EXPECT_FALSE(rdna2_may_write_guest_memory(instruction));
    instruction.opcode = 0x01u;   // image_load_mip
    EXPECT_FALSE(rdna2_may_write_guest_memory(instruction));
    instruction.opcode = 0x08u;   // image_store
    EXPECT_TRUE(rdna2_may_write_guest_memory(instruction));
    instruction.opcode = 0x0fu;   // image_atomic_swap
    EXPECT_TRUE(rdna2_may_write_guest_memory(instruction));
}

// #4422, Kena's NGG vertex shape: a register-offset (VCC_LO) x4 descriptor load whose words are
// only consumed as a descriptor, followed later by an image SAMPLE. A sample cannot invalidate the
// dispatch-time descriptor snapshot, so the load keeps the established exact-PC descriptor route;
// a real image STORE in the same position still forces backing.
TEST(RawWideFlatReadEffects, LaterImageSampleDoesNotForceRegisterOffsetDescriptorBacking) {
    const auto program = [](uint32_t image_word0) {
        return std::vector<uint32_t>{
            0x8f6a8404u,               // s_lshl_b32 vcc_lo, s4, 4
            0xf4080201u, 0xd4000000u,  // s_load_dwordx4 s[8:11], s[2:3], vcc_lo
            0xbf8cc07fu,               // s_waitcnt lgkmcnt(0)
            0x880b040bu,               // s_or_b32 s11, s11, s4: a descriptor patch, as Kena's
            0xe0002000u, 0x80020100u,  // buffer_load_format_x v1, v0, s[8:11], 0 idxen
            image_word0, 0x00c20007u,  // image_sample_l / image_store (Kena's operand word)
            0xbf810000u};              // s_endpgm
    };
    const auto decode = [](const std::vector<uint32_t>& words) {
        std::vector<Rdna2Inst> out;
        rdna2_walk(words.data(), words.size(), out);
        return out;
    };
    const auto sample = decode(program(0xf0900308u));
    const auto store = decode(program(0xf0200308u));
    ASSERT_GE(sample.size(), 7u);
    ASSERT_GE(store.size(), 7u);
    ASSERT_EQ(sample[3].fmt, Rdna2Format::SOP2);
    ASSERT_EQ(sample[3].opcode, 0x10u) << "s_or_b32";
    ASSERT_EQ(sample[3].dst.value, 11);
    ASSERT_EQ(sample[1].fmt, Rdna2Format::SMEM);
    ASSERT_EQ(sample[1].opcode, 0x2u);
    ASSERT_EQ(sample[1].src[1].kind, OperandKind::Special);
    ASSERT_EQ(sample[1].src[1].value, 106);
    ASSERT_EQ(sample[5].fmt, Rdna2Format::MIMG);
    ASSERT_EQ(sample[5].opcode, 0x24u) << "image_sample_l";
    ASSERT_EQ(store[5].fmt, Rdna2Format::MIMG);
    ASSERT_EQ(store[5].opcode, 0x08u) << "image_store";
    EXPECT_TRUE(rdna2_raw_wide_data_loads(sample).empty());
    EXPECT_EQ(rdna2_raw_wide_data_loads(store), std::vector<uint32_t>{1u})
        << "a genuine writer still keeps a register-offset descriptor patch fail-visible";
}

// #4425 review: the lane reads write an SGPR whose decoded operand kind is VGPR. cannot_write_vcc
// is shared by the raw-wide VCC walk and the #590 uniform-VCCZ walk, which has no other backstop.
TEST(RawWideFlatReadEffects, LaneReadsIntoVccAreVccWriters) {
    const auto one = [](std::vector<uint32_t> words) {
        std::vector<Rdna2Inst> out;
        rdna2_walk(words.data(), words.size(), out);
        return out.at(0);
    };
    const auto firstlane_vcc = one({0x7ed40500u});           // v_readfirstlane_b32 vcc_lo, v0
    const auto firstlane_vcc_hi = one({0x7ed60500u});        // v_readfirstlane_b32 vcc_hi, v0
    const auto firstlane_s5 = one({0x7e0a0500u});            // v_readfirstlane_b32 s5, v0
    const auto readlane_vcc = one({0xd760006au, 0x00010100u});   // v_readlane_b32 vcc_lo, v0, 0
    const auto readlane_s5 = one({0xd7600005u, 0x00010100u});    // v_readlane_b32 s5, v0, 0
    const auto mov = one({0x7e020300u});                     // v_mov_b32 v1, v0
    ASSERT_EQ(firstlane_vcc.fmt, Rdna2Format::VOP1);
    ASSERT_EQ(firstlane_vcc.opcode, 0x02u);
    ASSERT_EQ(firstlane_vcc.dst.value, 106);
    ASSERT_EQ(firstlane_vcc_hi.dst.value, 107);
    ASSERT_EQ(firstlane_s5.dst.value, 5);
    ASSERT_EQ(readlane_vcc.fmt, Rdna2Format::VOP3);
    ASSERT_EQ(readlane_vcc.opcode, 0x360u);
    ASSERT_EQ(readlane_vcc.dst.value, 106);
    ASSERT_EQ(readlane_s5.dst.value, 5);
    EXPECT_FALSE(cannot_write_vcc(firstlane_vcc));
    EXPECT_FALSE(cannot_write_vcc(firstlane_vcc_hi));
    EXPECT_FALSE(cannot_write_vcc(readlane_vcc));
    EXPECT_TRUE(cannot_write_vcc(firstlane_s5));
    EXPECT_TRUE(cannot_write_vcc(readlane_s5));
    EXPECT_TRUE(cannot_write_vcc(mov));
}
