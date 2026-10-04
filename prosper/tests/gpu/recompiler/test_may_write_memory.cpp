// test_may_write_memory — checks rdna2_instruction_may_write_memory against the ISA opcode
// tables, exhaustively per memory family.
//
// A missed writer is silent and dangerous: a descriptor whose backing the shader rewrote keeps
// resolving to stale bytes. An over-reported writer only costs an unresolved descriptor.
// This test pins the current conservative policy exactly:
//   - The ISA confirms every listed reader is a pure read;
//   - Deliberately conservative readers (such as D16 loads or unassigned SMEM 0x05-0x07/0x0D-0x0F)
//     match the implementation's classification;
//   - Admitting any further reader requires updating this table.
//
// Oracle: AMD RDNA2 ISA 70648 opcode tables, cross-checked against the llvm-mc-established
// sets cited by rdna2_decode.cpp. MIMG Table 100: loads 0x00-0x05, get_resinfo 0x0E,
// sample/gather/get_lod 0x20-0x6F (its invalid holes stay writers), msaa_load 0x80,
// BVH intersects 0xE6/0xE7; stores 0x08-0x0B and atomics 0x0F-0x1F are writers.
// MUBUF Table 98: format loads 0x00-0x03, raw untyped loads 0x08-0x0F;
// MTBUF/FLAT/SMEM ranges as coded. No device is created.
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iomanip>

using namespace prosper::gpu;

namespace {

// [ISA Table 100] Unassigned slots inside the 0x20-0x6F sample/gather/get_lod range. An
// unlisted encoding can only cost an unresolved descriptor, never a stale binding, so these
// stay writers; the list is pinned exactly so a silent widening reddens.
bool isMimgHole(uint32_t op) {
    switch (op) {
        case 0x42u:
        case 0x43u:
        case 0x4Au:
        case 0x4Bu:
        case 0x52u:
        case 0x53u:
        case 0x5Au:
        case 0x5Bu:
        case 0x62u:
        case 0x63u:
        case 0x64u:
        case 0x65u:
        case 0x66u:
        case 0x67u: return true;
        default: return false;
    }
}

// [ISA Table 100] Opcodes architecturally guaranteed to never write guest memory.
bool isMimgRead(uint32_t op) {
    if (op <= 0x05u || op == 0x0Eu || op == 0x80u || op == 0xE6u || op == 0xE7u) return true;
    return op >= 0x20u && op <= 0x6Fu && !isMimgHole(op);
}

Rdna2Inst mem_inst(Rdna2Format fmt, uint32_t op) {
    Rdna2Inst in{};
    in.fmt = fmt;
    in.opcode = op;
    return in;
}

}  // namespace

TEST(Rdna2MayWriteMemory, MimgReadersAndWritersMatchTheIsaTable) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        EXPECT_EQ(rdna2_instruction_may_write_memory(mem_inst(Rdna2Format::MIMG, op)),
                  !isMimgRead(op))
            << "MIMG op=0x" << std::hex << op;
    }
}

TEST(Rdna2MayWriteMemory, MimgAnchorsOnLlvmMcPackets) {
    // gfx1030 words from the decode suite: image_gather4_lz 0x47 reads, image_store 0x08 and
    // image_atomic_swap 0x0F write.
    const auto decode2 = [](uint32_t w0, uint32_t w1) {
        const uint32_t words[2] = {w0, w1};
        return rdna2_decode_one(words, 2);
    };

    const Rdna2Inst dec_gather = decode2(0xF11C0108u, 0x00070103u);
    ASSERT_EQ(dec_gather.fmt, Rdna2Format::MIMG);
    ASSERT_EQ(dec_gather.opcode, 0x47u);
    EXPECT_FALSE(rdna2_instruction_may_write_memory(dec_gather));

    const Rdna2Inst dec_store = decode2(0xF0200108u, 0x00020009u);
    ASSERT_EQ(dec_store.fmt, Rdna2Format::MIMG);
    ASSERT_EQ(dec_store.opcode, 0x08u);
    EXPECT_TRUE(rdna2_instruction_may_write_memory(dec_store));

    const Rdna2Inst dec_swap = decode2(0xF03C2108u, 0x00000900u);
    ASSERT_EQ(dec_swap.fmt, Rdna2Format::MIMG);
    ASSERT_EQ(dec_swap.opcode, 0x0Fu);
    EXPECT_TRUE(rdna2_instruction_may_write_memory(dec_swap));
}

TEST(Rdna2MayWriteMemory, MubufLoadRangesAreReadsAndTheRestWrites) {
    // [ISA Table 98] format loads 0x00-0x03 and raw untyped loads 0x08-0x0F (ubyte..dwordx4) read;
    // format stores, raw stores and the D16/cache-invalidate forms (incl. 0x80+) stay writers.
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const bool read = op <= 0x03u || (op >= 0x08u && op <= 0x0Fu);
        EXPECT_EQ(rdna2_instruction_may_write_memory(mem_inst(Rdna2Format::MUBUF, op)), !read)
            << "MUBUF op=0x" << std::hex << op;
    }
}

TEST(Rdna2MayWriteMemory, MtbufFlatSmemRangesAreExact) {
    // MTBUF opcode is 4 bits in decoder (0x0..0xF).
    for (uint32_t op = 0; op <= 0x0Fu; ++op) {
        EXPECT_EQ(rdna2_instruction_may_write_memory(mem_inst(Rdna2Format::MTBUF, op)), op > 0x03u)
            << "MTBUF op=0x" << std::hex << op;
    }

    // FLAT opcode is 7 bits in decoder (0x00..0x7F).
    for (uint32_t op = 0; op <= 0x7Fu; ++op) {
        const bool flat_read = op >= 0x08u && op <= 0x0Fu;
        EXPECT_EQ(rdna2_instruction_may_write_memory(mem_inst(Rdna2Format::FLAT, op)), !flat_read)
            << "FLAT op=0x" << std::hex << op;
    }

    // SMEM opcode is 8 bits (0x00..0xFF).
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        EXPECT_EQ(rdna2_instruction_may_write_memory(mem_inst(Rdna2Format::SMEM, op)), op >= 0x10u)
            << "SMEM op=0x" << std::hex << op;
    }
}

// Smoke check: Non-descriptor formats are outside this predicate's contract (which classifies
// guest memory reached through descriptors). LDS/GDS and export side-effects are classified elsewhere.
TEST(Rdna2MayWriteMemory, NonDescriptorInstructionsAreOutsidePredicateContract) {
    for (Rdna2Format fmt :
         {Rdna2Format::SOP1, Rdna2Format::SOP2, Rdna2Format::SOPK, Rdna2Format::SOPC,
          Rdna2Format::SOPP, Rdna2Format::VOP1, Rdna2Format::VOP2, Rdna2Format::VOPC,
          Rdna2Format::VOP3, Rdna2Format::VOP3P, Rdna2Format::VINTRP, Rdna2Format::Unknown}) {
        EXPECT_FALSE(rdna2_instruction_may_write_memory(mem_inst(fmt, 0u)))
            << "fmt=" << static_cast<int>(fmt);
    }
}
