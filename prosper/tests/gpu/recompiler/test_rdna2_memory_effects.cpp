// Pins ADR 0038 step 8a's unused architectural facts. A missing read/short write would make a
// future transfer unsound; these tests do not change or assert backend admission/definedness.
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include "gpu/recompiler/ir/effects/operand_effects.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>

using namespace prosper::gpu;
namespace {
#include "memory_opcode_oracle.inc"
#include "memory_width_oracle.inc"
using A = Rdna2MemoryAccess;
using D = Rdna2MemoryData;
using L = Rdna2MemoryAddress;
using H = Rdna2MemoryHalf;

Rdna2Inst decoded(Rdna2Format fmt, uint32_t op) {
    uint32_t words[2] = {};
    switch (fmt) {
        case Rdna2Format::DS: words[0] = 0xd8000000u | (op << 18); break;
        case Rdna2Format::MUBUF: words[0] = 0xe0000000u | (op << 18); break;
        case Rdna2Format::MTBUF:
            words[0] = 0xe8000000u | ((op & 7) << 16);
            words[1] = (op >> 3) << 21;
            break;
        case Rdna2Format::MIMG: words[0] = 0xf0000000u | ((op & 127) << 18) | (op >> 7); break;
        case Rdna2Format::FLAT: words[0] = 0xdc000000u | (op << 18); break;
        case Rdna2Format::EXP: words[0] = 0xf8000000u; break;
        case Rdna2Format::VINTRP: words[0] = 0xc8000000u | (op << 16); break;
        default: return {};
    }
    return rdna2_decode_one(words, 2);
}

const Rdna2MemoryEffects& mem(Rdna2Format fmt, uint32_t op) {
    return rdna2_instruction_effects(fmt, op).memory;
}
}   // namespace

TEST(Rdna2MemoryEffects, EveryEncodingHasAnExplicitEntry) {
    const std::pair<Rdna2Format, uint32_t> formats[] = {
        {Rdna2Format::DS, 256},   {Rdna2Format::MUBUF, 256}, {Rdna2Format::MTBUF, 16},
        {Rdna2Format::MIMG, 256}, {Rdna2Format::FLAT, 128},  {Rdna2Format::EXP, 1},
        {Rdna2Format::VINTRP, 4},
    };
    for (const auto& [fmt, end] : formats) {
        ASSERT_TRUE(rdna2_effects_format_is_tabulated(fmt));
        auto range = rdna2_effects_opcode_range(fmt);
        ASSERT_EQ(range.first, 0u);
        ASSERT_EQ(range.end, end);
        for (uint32_t op = range.first; op < range.end; ++op) {
            const auto in = decoded(fmt, op);
            EXPECT_EQ(in.fmt, fmt);
            EXPECT_EQ(in.opcode, op);
            const auto& e = rdna2_instruction_effects(fmt, op);
            EXPECT_NE(e.status, Rdna2EffectsStatus::Unknown) << int(fmt) << ':' << op;
            EXPECT_FALSE(e.known());
            EXPECT_FALSE(e.describes_sources());
            EXPECT_FALSE(e.describes_vector_result());
        }
        EXPECT_EQ(rdna2_instruction_effects(fmt, range.end).status, Rdna2EffectsStatus::Unknown);
    }
}

TEST(Rdna2MemoryEffects, NamedRowsMatchTheIndependentLlvmSweep) {
    std::set<std::pair<Rdna2Format, uint32_t>> expected;
    for (const auto& op : kMemoryOpcodes) {
        expected.insert({op.fmt, op.opcode});
        const auto& e = rdna2_instruction_effects(op.fmt, op.opcode);
        ASSERT_NE(e.name, nullptr) << op.name;
        EXPECT_STREQ(e.name, op.name);
        EXPECT_TRUE(e.status == Rdna2EffectsStatus::Memory ||
                    e.status == Rdna2EffectsStatus::Unmodeled)
            << op.name;
    }
    std::size_t found = 0;
    for (const auto& row : rdna2_instruction_effects_rows()) {
        if (row.effects.status != Rdna2EffectsStatus::Memory &&
            row.effects.status != Rdna2EffectsStatus::Unmodeled)
            continue;
        ++found;
        EXPECT_EQ(expected.count({row.fmt, row.opcode}), 1u) << row.effects.name;
    }
    EXPECT_EQ(found, expected.size());
    EXPECT_EQ(found, 418u);
}

TEST(Rdna2MemoryEffects, FixedRegisterWidthsMatchLlvmRegisterClasses) {
    for (const auto& op : kMemoryWidths) {
        const auto& e = rdna2_instruction_effects(op.fmt, op.op);
        SCOPED_TRACE(e.name);
        ASSERT_TRUE(e.memory.described);
        const auto& m = e.memory;
        EXPECT_EQ(m.vsrc_dwords[0], op.v0);
        EXPECT_EQ(m.vsrc_dwords[1], op.v1);
        EXPECT_EQ(m.vsrc_dwords[2], op.v2);
        EXPECT_EQ(m.data_read_dwords, op.read);
        EXPECT_EQ(m.data_write_dwords, op.write);
        EXPECT_EQ(m.write_if_glc, m.access == A::Atomic && op.fmt != Rdna2Format::DS);
        if (op.fmt == Rdna2Format::MUBUF || op.fmt == Rdna2Format::MTBUF) {
            EXPECT_EQ(m.status, m.access == A::Load);
            const bool lds_load =
                op.fmt == Rdna2Format::MUBUF && (op.op == 0 || (op.op >= 8 && op.op <= 12));
            EXPECT_EQ(m.lds_redirect, lds_load);
        }
    }
}

TEST(Rdna2MemoryEffects, UndocumentedLlvmFormsNeverClaimNoEffect) {
    unsigned count = 0;
    for (const auto& row : rdna2_instruction_effects_rows()) {
        if (row.effects.status != Rdna2EffectsStatus::Unmodeled) continue;
        ++count;
        EXPECT_NE(row.effects.name, nullptr);
        EXPECT_FALSE(row.effects.memory.described);
        EXPECT_FALSE(row.effects.known());
    }
    EXPECT_EQ(count, 47u);   // 31 DS src2 and 16 image CD forms absent from ISA 70648
}

TEST(Rdna2MemoryEffects, DsHasSeparateAddressDataAndReturnedWords) {
    EXPECT_EQ(mem(Rdna2Format::DS, 0x0e).access, A::Store);
    EXPECT_EQ(mem(Rdna2Format::DS, 0x2e).access, A::Atomic);
    EXPECT_EQ(mem(Rdna2Format::DS, 0x76).data_write_dwords, 2);
    EXPECT_EQ(mem(Rdna2Format::DS, 0x77).data_write_dwords, 4);
    EXPECT_EQ(mem(Rdna2Format::DS, 0x10).vsrc_dwords[2], 1);
    EXPECT_EQ(mem(Rdna2Format::DS, 0x50).vsrc_dwords[2], 2);
    EXPECT_EQ(mem(Rdna2Format::DS, 0x35).access, A::CrossLane);
    EXPECT_FALSE(mem(Rdna2Format::DS, 0x35).gds_reads_m0);
    EXPECT_TRUE(mem(Rdna2Format::DS, 0x36).gds_reads_m0);
    for (uint32_t op : {0x19u, 0x3du, 0x3eu, 0x3fu, 0xb0u, 0xb1u})
        EXPECT_NE(mem(Rdna2Format::DS, op).implicit_reads & kRdna2ImplicitM0, 0);
    EXPECT_EQ(mem(Rdna2Format::DS, 0x14).implicit_reads, 0);
    EXPECT_EQ(mem(Rdna2Format::DS, 0xa2).result_half, H::Low);
    EXPECT_EQ(mem(Rdna2Format::DS, 0xa3).result_half, H::High);
    EXPECT_EQ(mem(Rdna2Format::DS, 0xa3).element_bits, 8);
}

TEST(Rdna2MemoryEffects, BufferAddressAndStatusRemainControlDependent) {
    for (auto fmt : {Rdna2Format::MUBUF, Rdna2Format::MTBUF}) {
        const auto& load = mem(fmt, 0);
        EXPECT_EQ(load.address, L::Buffer);   // OFFEN + IDXEN address words, even when both zero
        EXPECT_EQ(load.ssrc_dwords[1], 4);   // V#
        EXPECT_EQ(load.ssrc_dwords[2], 1);   // SOFFSET when it names a register
        EXPECT_TRUE(load.status);
        EXPECT_EQ(load.lds_redirect, fmt == Rdna2Format::MUBUF);
    }
    EXPECT_EQ(mem(Rdna2Format::MUBUF, 0x81).data_write_dwords, 1);   // two packed halves
    EXPECT_EQ(mem(Rdna2Format::MUBUF, 0x83).data_write_dwords, 2);
    EXPECT_EQ(mem(Rdna2Format::MUBUF, 0x51).data_read_dwords, 4);
    EXPECT_EQ(mem(Rdna2Format::MUBUF, 0x51).data_write_dwords, 2);
    EXPECT_EQ(mem(Rdna2Format::MUBUF, 0x21).result_half, H::High);
    EXPECT_EQ(mem(Rdna2Format::MUBUF, 0x08).element_bits, 8);
    EXPECT_EQ(mem(Rdna2Format::MUBUF, 0x0a).element_bits, 16);
    EXPECT_EQ(mem(Rdna2Format::MUBUF, 0x71).access, A::CacheControl);
    // ISA 8.1.9 allows LDS only for FORMAT_X and the five ordinary single-word loads.
    for (uint32_t op : {0u, 8u, 9u, 10u, 11u, 12u})
        EXPECT_TRUE(mem(Rdna2Format::MUBUF, op).lds_redirect);
    for (uint32_t op : {1u, 0x0du, 0x20u, 0x26u, 0x80u})
        EXPECT_FALSE(mem(Rdna2Format::MUBUF, op).lds_redirect);
    for (auto fmt : {Rdna2Format::MUBUF, Rdna2Format::MTBUF})
        EXPECT_FALSE(mem(fmt, 4).status);   // stores have no buffer TFE response
    EXPECT_FALSE(mem(Rdna2Format::MUBUF, 0x30).status);   // neither do buffer atomics
}

TEST(Rdna2MemoryEffects, FlatKeepsTheSegmentAndRawDataFieldDistinctions) {
    const auto& load = mem(Rdna2Format::FLAT, 0x0d);
    EXPECT_EQ(load.address, L::Flat);
    EXPECT_EQ(load.flat_segments, 7);   // flat, scratch, global: different address widths
    EXPECT_TRUE(load.lds_redirect);   // only legal for scratch/global
    EXPECT_EQ(mem(Rdna2Format::FLAT, 0x30).flat_segments, 5);   // no scratch atomic
    EXPECT_EQ(mem(Rdna2Format::FLAT, 0x34).flat_segments, 4);   // global CSUB only
    EXPECT_EQ(mem(Rdna2Format::FLAT, 0x16).address, L::FlatAddtid);
    EXPECT_EQ(mem(Rdna2Format::FLAT, 0x17).access, A::Store);
    EXPECT_EQ(mem(Rdna2Format::FLAT, 0x51).data_read_dwords, 4);
    EXPECT_EQ(mem(Rdna2Format::FLAT, 0x51).data_write_dwords, 2);
}

TEST(Rdna2MemoryEffects, ImageHasDmaskPackingAddressAndDescriptorShapes) {
    const auto& load = mem(Rdna2Format::MIMG, 0);
    EXPECT_EQ(load.address, L::Image);
    EXPECT_EQ(load.data, D::ImageDmask);
    EXPECT_EQ(load.ssrc_dwords[1], 8);   // four with R128; BVH always uses four
    EXPECT_TRUE(load.status);   // TFE/LWE status follows packed data, never a data component
    EXPECT_FALSE(load.image_sampler);
    EXPECT_EQ(mem(Rdna2Format::MIMG, 1).address, L::ImageMip);
    EXPECT_EQ(mem(Rdna2Format::MIMG, 2).data, D::ImagePacked);
    EXPECT_EQ(mem(Rdna2Format::MIMG, 0xe).address, L::ImageResinfo);
    EXPECT_EQ(mem(Rdna2Format::MIMG, 0x10).data, D::ImageAtomicCompare);
    EXPECT_TRUE(mem(Rdna2Format::MIMG, 0x10).write_if_glc);
    EXPECT_EQ(mem(Rdna2Format::MIMG, 0x40).data, D::ImageGather);
    const auto& sample = mem(Rdna2Format::MIMG, 0x3b);
    EXPECT_EQ(sample.ssrc_dwords[2], 4);
    EXPECT_TRUE(sample.image_sampler);
    EXPECT_TRUE(sample.image_offset);
    EXPECT_TRUE(sample.image_compare);
    EXPECT_TRUE(sample.image_derivatives);
    EXPECT_TRUE(sample.image_tail);
    EXPECT_FALSE(sample.image_bias);
    EXPECT_TRUE(mem(Rdna2Format::MIMG, 0xa2).image_g16);
    EXPECT_EQ(mem(Rdna2Format::MIMG, 0x80).address, L::ImageMsaa);
    EXPECT_EQ(mem(Rdna2Format::MIMG, 0xe6).address, L::ImageBvh32);
    EXPECT_EQ(mem(Rdna2Format::MIMG, 0xe7).address, L::ImageBvh64);
    EXPECT_EQ(mem(Rdna2Format::MIMG, 0xe6).ssrc_dwords[1], 4);
    EXPECT_EQ(mem(Rdna2Format::MIMG, 0xe6).data_write_dwords, 4);
}

TEST(Rdna2MemoryEffects, VintrpMovIsASelectorAndP2ReadsTheAccumulator) {
    EXPECT_EQ(mem(Rdna2Format::VINTRP, 0).vsrc_dwords[0], 1);
    EXPECT_FALSE(mem(Rdna2Format::VINTRP, 0).reads_old_vdst);
    EXPECT_TRUE(mem(Rdna2Format::VINTRP, 1).reads_old_vdst);
    EXPECT_EQ(mem(Rdna2Format::VINTRP, 2).vsrc_dwords[0], 0);
    for (uint32_t op : {0u, 1u, 2u}) {
        EXPECT_EQ(mem(Rdna2Format::VINTRP, op).data_write_dwords, 1);
        EXPECT_EQ(mem(Rdna2Format::VINTRP, op).implicit_reads,
                  kRdna2ImplicitM0 | kRdna2ImplicitExec);
    }
    EXPECT_EQ(mem(Rdna2Format::EXP, 0).address, L::Export);
    EXPECT_EQ(mem(Rdna2Format::EXP, 0).access, A::Export);
    EXPECT_EQ(mem(Rdna2Format::EXP, 0).data_write_dwords, 0);
}

TEST(Rdna2MemoryEffects, Vop3InterpolationKeepsTheAttributeOutOfRegisterReads) {
    for (uint32_t op : {0x200u, 0x201u, 0x202u, 0x342u, 0x343u, 0x35au}) {
        const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, op);
        const auto& m = e.memory;
        EXPECT_EQ(e.status, Rdna2EffectsStatus::Valu);
        ASSERT_TRUE(m.described);
        EXPECT_EQ(m.access, A::Interpolate);
        EXPECT_EQ(m.vsrc_dwords[0], 0);   // attribute immediate, never s0/v0
        EXPECT_EQ(m.ssrc_dwords[0], 0);
        EXPECT_EQ(m.implicit_reads, kRdna2ImplicitExec | kRdna2ImplicitM0);
        EXPECT_EQ(m.data_write_dwords, 1);
        EXPECT_EQ(m.reads_old_vdst, op == 0x201);
    }
    EXPECT_EQ(mem(Rdna2Format::VOP3, 0x202).vsrc_dwords[1], 0);   // MOV selector
    EXPECT_EQ(mem(Rdna2Format::VOP3, 0x201).vsrc_dwords[1], 1);
    EXPECT_EQ(mem(Rdna2Format::VOP3, 0x343).vsrc_dwords[2], 1);   // explicit partial result
    EXPECT_EQ(mem(Rdna2Format::VOP3, 0x35a).vsrc_dwords[2], 1);
    EXPECT_FALSE(mem(Rdna2Format::VOP3, 0x35a).reads_old_vdst);
    EXPECT_EQ(mem(Rdna2Format::VOP3, 0x342).element_bits, 16);
}
