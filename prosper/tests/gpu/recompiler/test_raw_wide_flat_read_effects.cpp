// A later read-only FLAT/GLOBAL must not turn descriptor assembly into numeric raw data.
// Actual scalar observers still need backing, and uncertain memory effects still block snapshots.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
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

TEST(RawWideFlatReadEffects, OtherMemoryFamilyPoliciesRemainUnchanged) {
    Rdna2Inst instruction;
    instruction.fmt = Rdna2Format::MIMG;
    instruction.opcode = 0x01u;
    EXPECT_FALSE(rdna2_instruction_may_write_memory(instruction));
    EXPECT_TRUE(rdna2_may_write_guest_memory(instruction))
        << "This change does not broaden the raw-wide policy for other image readers";
    instruction.opcode = 0x2fu;
    EXPECT_FALSE(rdna2_may_write_guest_memory(instruction));
    instruction.opcode = 0x08u;
    EXPECT_TRUE(rdna2_may_write_guest_memory(instruction));
}
