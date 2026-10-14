// test_rdna2_instruction_effects — pins the shared RDNA2 instruction-effects table (#5146).
//
// The table replaces private opcode lists that had drifted apart: an SCC-reader list matched
// s_mov_b32 where it meant s_cmov_b32, a destination-width list wrote one SGPR for s_ashr_i64 and
// the WREXEC pair, and two copies of one width function disagreed about VOPC. Every one of those
// was silent -- a missed SCC reader drops the wave vote a consumer needs, and an under-counted
// destination lets an analysis believe a clobbered register still holds its old value.
//
// What is pinned here:
//   * COVERAGE: every opcode the decoder can produce in a tabulated format has an entry, named or
//     explicitly reserved. A new decodable opcode without an entry turns this red; so does deleting
//     an entry, because reserved slots are listed, not inferred from gaps.
//   * IDENTITY: the named opcodes are exactly the ones llvm-mc -mcpu=gfx1030 disassembles, under
//     the same mnemonic (an oracle independent of prosper; the arrays below were produced by
//     disassembling every encoding of each opcode space).
//   * EFFECTS: the SCC readers/writers and the SGPR-pair destinations against the ISA 70648
//     tables (chapter 5 "Sets SCC?" columns and the chapter 12 pseudo-code), listed independently.
//   * REGRESSIONS for the old-list bugs the table exposed, each against the old wrong answer.
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include "gpu/recompiler/indirect/rdna2_indirect_scalar_writes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace prosper::gpu;

namespace {

struct LlvmOpcode {
    uint32_t opcode;
    const char* name;
};

// llvm-mc -arch=amdgcn -mcpu=gfx1030 -disassemble over each whole opcode space (SOPP with
// SIMM16 = 0, since several SOPP forms reject a nonzero immediate).
constexpr LlvmOpcode kLlvmSop2[] = {
    {0x00, "s_add_u32"},         {0x01, "s_sub_u32"},         {0x02, "s_add_i32"},
    {0x03, "s_sub_i32"},         {0x04, "s_addc_u32"},        {0x05, "s_subb_u32"},
    {0x06, "s_min_i32"},         {0x07, "s_min_u32"},         {0x08, "s_max_i32"},
    {0x09, "s_max_u32"},         {0x0a, "s_cselect_b32"},     {0x0b, "s_cselect_b64"},
    {0x0e, "s_and_b32"},         {0x0f, "s_and_b64"},         {0x10, "s_or_b32"},
    {0x11, "s_or_b64"},          {0x12, "s_xor_b32"},         {0x13, "s_xor_b64"},
    {0x14, "s_andn2_b32"},       {0x15, "s_andn2_b64"},       {0x16, "s_orn2_b32"},
    {0x17, "s_orn2_b64"},        {0x18, "s_nand_b32"},        {0x19, "s_nand_b64"},
    {0x1a, "s_nor_b32"},         {0x1b, "s_nor_b64"},         {0x1c, "s_xnor_b32"},
    {0x1d, "s_xnor_b64"},        {0x1e, "s_lshl_b32"},        {0x1f, "s_lshl_b64"},
    {0x20, "s_lshr_b32"},        {0x21, "s_lshr_b64"},        {0x22, "s_ashr_i32"},
    {0x23, "s_ashr_i64"},        {0x24, "s_bfm_b32"},         {0x25, "s_bfm_b64"},
    {0x26, "s_mul_i32"},         {0x27, "s_bfe_u32"},         {0x28, "s_bfe_i32"},
    {0x29, "s_bfe_u64"},         {0x2a, "s_bfe_i64"},         {0x2c, "s_absdiff_i32"},
    {0x2e, "s_lshl1_add_u32"},   {0x2f, "s_lshl2_add_u32"},   {0x30, "s_lshl3_add_u32"},
    {0x31, "s_lshl4_add_u32"},   {0x32, "s_pack_ll_b32_b16"}, {0x33, "s_pack_lh_b32_b16"},
    {0x34, "s_pack_hh_b32_b16"}, {0x35, "s_mul_hi_u32"},      {0x36, "s_mul_hi_i32"},
};
constexpr LlvmOpcode kLlvmSop1[] = {
    {0x03, "s_mov_b32"},
    {0x04, "s_mov_b64"},
    {0x05, "s_cmov_b32"},
    {0x06, "s_cmov_b64"},
    {0x07, "s_not_b32"},
    {0x08, "s_not_b64"},
    {0x09, "s_wqm_b32"},
    {0x0a, "s_wqm_b64"},
    {0x0b, "s_brev_b32"},
    {0x0c, "s_brev_b64"},
    {0x0d, "s_bcnt0_i32_b32"},
    {0x0e, "s_bcnt0_i32_b64"},
    {0x0f, "s_bcnt1_i32_b32"},
    {0x10, "s_bcnt1_i32_b64"},
    {0x11, "s_ff0_i32_b32"},
    {0x12, "s_ff0_i32_b64"},
    {0x13, "s_ff1_i32_b32"},
    {0x14, "s_ff1_i32_b64"},
    {0x15, "s_flbit_i32_b32"},
    {0x16, "s_flbit_i32_b64"},
    {0x17, "s_flbit_i32"},
    {0x18, "s_flbit_i32_i64"},
    {0x19, "s_sext_i32_i8"},
    {0x1a, "s_sext_i32_i16"},
    {0x1b, "s_bitset0_b32"},
    {0x1c, "s_bitset0_b64"},
    {0x1d, "s_bitset1_b32"},
    {0x1e, "s_bitset1_b64"},
    {0x1f, "s_getpc_b64"},
    {0x20, "s_setpc_b64"},
    {0x21, "s_swappc_b64"},
    {0x22, "s_rfe_b64"},
    {0x24, "s_and_saveexec_b64"},
    {0x25, "s_or_saveexec_b64"},
    {0x26, "s_xor_saveexec_b64"},
    {0x27, "s_andn2_saveexec_b64"},
    {0x28, "s_orn2_saveexec_b64"},
    {0x29, "s_nand_saveexec_b64"},
    {0x2a, "s_nor_saveexec_b64"},
    {0x2b, "s_xnor_saveexec_b64"},
    {0x2c, "s_quadmask_b32"},
    {0x2d, "s_quadmask_b64"},
    {0x2e, "s_movrels_b32"},
    {0x2f, "s_movrels_b64"},
    {0x30, "s_movreld_b32"},
    {0x31, "s_movreld_b64"},
    {0x34, "s_abs_i32"},
    {0x37, "s_andn1_saveexec_b64"},
    {0x38, "s_orn1_saveexec_b64"},
    {0x39, "s_andn1_wrexec_b64"},
    {0x3a, "s_andn2_wrexec_b64"},
    {0x3b, "s_bitreplicate_b64_b32"},
    {0x3c, "s_and_saveexec_b32"},
    {0x3d, "s_or_saveexec_b32"},
    {0x3e, "s_xor_saveexec_b32"},
    {0x3f, "s_andn2_saveexec_b32"},
    {0x40, "s_orn2_saveexec_b32"},
    {0x41, "s_nand_saveexec_b32"},
    {0x42, "s_nor_saveexec_b32"},
    {0x43, "s_xnor_saveexec_b32"},
    {0x44, "s_andn1_saveexec_b32"},
    {0x45, "s_orn1_saveexec_b32"},
    {0x46, "s_andn1_wrexec_b32"},
    {0x47, "s_andn2_wrexec_b32"},
    {0x49, "s_movrelsd_2_b32"},
};
constexpr LlvmOpcode kLlvmSopk[] = {
    {0x00, "s_movk_i32"},           {0x01, "s_version"},
    {0x02, "s_cmovk_i32"},          {0x03, "s_cmpk_eq_i32"},
    {0x04, "s_cmpk_lg_i32"},        {0x05, "s_cmpk_gt_i32"},
    {0x06, "s_cmpk_ge_i32"},        {0x07, "s_cmpk_lt_i32"},
    {0x08, "s_cmpk_le_i32"},        {0x09, "s_cmpk_eq_u32"},
    {0x0a, "s_cmpk_lg_u32"},        {0x0b, "s_cmpk_gt_u32"},
    {0x0c, "s_cmpk_ge_u32"},        {0x0d, "s_cmpk_lt_u32"},
    {0x0e, "s_cmpk_le_u32"},        {0x0f, "s_addk_i32"},
    {0x10, "s_mulk_i32"},           {0x12, "s_getreg_b32"},
    {0x13, "s_setreg_b32"},         {0x15, "s_setreg_imm32_b32"},
    {0x16, "s_call_b64"},           {0x17, "s_waitcnt_vscnt"},
    {0x18, "s_waitcnt_vmcnt"},      {0x19, "s_waitcnt_expcnt"},
    {0x1a, "s_waitcnt_lgkmcnt"},    {0x1b, "s_subvector_loop_begin"},
    {0x1c, "s_subvector_loop_end"},
};
constexpr LlvmOpcode kLlvmSopc[] = {
    {0x00, "s_cmp_eq_i32"},  {0x01, "s_cmp_lg_i32"},  {0x02, "s_cmp_gt_i32"},
    {0x03, "s_cmp_ge_i32"},  {0x04, "s_cmp_lt_i32"},  {0x05, "s_cmp_le_i32"},
    {0x06, "s_cmp_eq_u32"},  {0x07, "s_cmp_lg_u32"},  {0x08, "s_cmp_gt_u32"},
    {0x09, "s_cmp_ge_u32"},  {0x0a, "s_cmp_lt_u32"},  {0x0b, "s_cmp_le_u32"},
    {0x0c, "s_bitcmp0_b32"}, {0x0d, "s_bitcmp1_b32"}, {0x0e, "s_bitcmp0_b64"},
    {0x0f, "s_bitcmp1_b64"}, {0x12, "s_cmp_eq_u64"},  {0x13, "s_cmp_lg_u64"},
};
constexpr LlvmOpcode kLlvmSopp[] = {
    {0x00, "s_nop"},
    {0x01, "s_endpgm"},
    {0x02, "s_branch"},
    {0x03, "s_wakeup"},
    {0x04, "s_cbranch_scc0"},
    {0x05, "s_cbranch_scc1"},
    {0x06, "s_cbranch_vccz"},
    {0x07, "s_cbranch_vccnz"},
    {0x08, "s_cbranch_execz"},
    {0x09, "s_cbranch_execnz"},
    {0x0a, "s_barrier"},
    {0x0b, "s_setkill"},
    {0x0c, "s_waitcnt"},
    {0x0d, "s_sethalt"},
    {0x0e, "s_sleep"},
    {0x0f, "s_setprio"},
    {0x10, "s_sendmsg"},
    {0x11, "s_sendmsghalt"},
    {0x12, "s_trap"},
    {0x13, "s_icache_inv"},
    {0x14, "s_incperflevel"},
    {0x15, "s_decperflevel"},
    {0x16, "s_ttracedata"},
    {0x17, "s_cbranch_cdbgsys"},
    {0x18, "s_cbranch_cdbguser"},
    {0x19, "s_cbranch_cdbgsys_or_user"},
    {0x1a, "s_cbranch_cdbgsys_and_user"},
    {0x1b, "s_endpgm_saved"},
    {0x1e, "s_endpgm_ordered_ps_done"},
    {0x1f, "s_code_end"},
    {0x20, "s_inst_prefetch"},
    {0x21, "s_clause"},
    {0x22, "s_wait_idle"},
    {0x23, "s_waitcnt_depctr"},
    {0x24, "s_round_mode"},
    {0x25, "s_denorm_mode"},
    {0x28, "s_ttracedata_imm"},
};

const char* fmt_name(Rdna2Format fmt) {
    switch (fmt) {
        case Rdna2Format::SOP2: return "SOP2";
        case Rdna2Format::SOP1: return "SOP1";
        case Rdna2Format::SOPK: return "SOPK";
        case Rdna2Format::SOPC: return "SOPC";
        case Rdna2Format::SOPP: return "SOPP";
        default: return "other";
    }
}

// A one-dword instruction of `fmt` with opcode `op`, built from the encoding layout the decoder
// classifies on (rdna2_decode_one), plus a trailing literal in case the encoding wants one.
Rdna2Inst decode(Rdna2Format fmt, uint32_t op, uint32_t ssrc0 = 2u) {
    uint32_t code[2] = {0, 0x1234u};
    switch (fmt) {
        case Rdna2Format::SOP2:
            code[0] = 0x80000000u | (op << 23) | (4u << 16) | (6u << 8) | ssrc0;
            break;
        case Rdna2Format::SOP1: code[0] = 0xBE800000u | (4u << 16) | (op << 8) | ssrc0; break;
        case Rdna2Format::SOPK: code[0] = 0xB0000000u | (op << 23) | (4u << 16) | 0x10u; break;
        case Rdna2Format::SOPC: code[0] = 0xBF000000u | (op << 16) | (6u << 8) | ssrc0; break;
        case Rdna2Format::SOPP: code[0] = 0xBF800000u | (op << 16); break;
        default: break;
    }
    return rdna2_decode_one(code, 2);
}

const Rdna2Format kTabulated[] = {Rdna2Format::SOP2, Rdna2Format::SOP1, Rdna2Format::SOPK,
                                  Rdna2Format::SOPC, Rdna2Format::SOPP};

std::vector<LlvmOpcode> llvm_set(Rdna2Format fmt) {
    switch (fmt) {
        case Rdna2Format::SOP2: return {std::begin(kLlvmSop2), std::end(kLlvmSop2)};
        case Rdna2Format::SOP1: return {std::begin(kLlvmSop1), std::end(kLlvmSop1)};
        case Rdna2Format::SOPK: return {std::begin(kLlvmSopk), std::end(kLlvmSopk)};
        case Rdna2Format::SOPC: return {std::begin(kLlvmSopc), std::end(kLlvmSopc)};
        case Rdna2Format::SOPP: return {std::begin(kLlvmSopp), std::end(kLlvmSopp)};
        default: return {};
    }
}

bool in_set(std::initializer_list<uint32_t> set, uint32_t op) {
    for (uint32_t member : set)
        if (member == op) return true;
    return false;
}

}   // namespace

// ---- Coverage --------------------------------------------------------------------------------

TEST(Rdna2InstructionEffects, OpcodeSpaceIsExactlyWhatTheDecoderProduces) {
    for (Rdna2Format fmt : kTabulated) {
        const uint32_t space = rdna2_effects_opcode_space(fmt);
        ASSERT_GT(space, 0u) << fmt_name(fmt);
        for (uint32_t op = 0; op < space; ++op) {
            const Rdna2Inst in = decode(fmt, op);
            EXPECT_EQ(in.fmt, fmt) << fmt_name(fmt) << " op=" << op;
            EXPECT_EQ(in.opcode, op) << fmt_name(fmt) << " op=" << op;
        }
    }
    // The first opcode past each space belongs to another format: the space is not too small.
    EXPECT_NE(decode(Rdna2Format::SOP2, 0x60).fmt, Rdna2Format::SOP2);
    EXPECT_NE(decode(Rdna2Format::SOPK, 0x1d).fmt, Rdna2Format::SOPK);
}

TEST(Rdna2InstructionEffects, EveryDecodableOpcodeHasAnEntry) {
    for (Rdna2Format fmt : kTabulated) {
        EXPECT_TRUE(rdna2_effects_format_is_tabulated(fmt)) << fmt_name(fmt);
        for (uint32_t op = 0; op < rdna2_effects_opcode_space(fmt); ++op)
            EXPECT_NE(rdna2_instruction_effects(fmt, op).status, Rdna2EffectsStatus::Unknown)
                << fmt_name(fmt) << " op=0x" << std::hex << op
                << " decodes but has no entry: add a named row or a reserved range";
    }
}

TEST(Rdna2InstructionEffects, RowsAreUniqueAndWellFormed) {
    std::set<std::pair<int, uint32_t>> seen;
    for (const Rdna2InstructionEffectsRow& row : rdna2_instruction_effects_rows()) {
        EXPECT_TRUE(seen.insert({static_cast<int>(row.fmt), row.opcode}).second)
            << fmt_name(row.fmt) << " op=0x" << std::hex << row.opcode << " has two rows";
        EXPECT_LT(row.opcode, rdna2_effects_opcode_space(row.fmt))
            << fmt_name(row.fmt) << " op=0x" << std::hex << row.opcode << " cannot be decoded";
        const Rdna2InstructionEffects& e = row.effects;
        if (e.status == Rdna2EffectsStatus::Known) {
            EXPECT_NE(e.name, nullptr);
            // The SCC read kind and the implicit-read bit are one fact stated twice.
            EXPECT_EQ(e.scc_read != Rdna2SccRead::None, (e.implicit_reads & kRdna2ImplicitScc) != 0)
                << e.name;
            EXPECT_EQ(e.exec != Rdna2ExecEffect::None,
                      (e.implicit_writes & kRdna2ImplicitExec) != 0)
                << e.name;
        } else {
            EXPECT_EQ(e.status, Rdna2EffectsStatus::Reserved);
            EXPECT_EQ(e.name, nullptr);
        }
        // The lookup returns the row.
        EXPECT_EQ(rdna2_instruction_effects(row.fmt, row.opcode).status, e.status);
    }
}

TEST(Rdna2InstructionEffects, NamedOpcodesAreExactlyLlvmGfx1030) {
    for (Rdna2Format fmt : kTabulated) {
        const std::vector<LlvmOpcode> llvm = llvm_set(fmt);
        std::set<uint32_t> llvm_ops;
        for (const LlvmOpcode& entry : llvm) {
            llvm_ops.insert(entry.opcode);
            const Rdna2InstructionEffects& e = rdna2_instruction_effects(fmt, entry.opcode);
            ASSERT_TRUE(e.known()) << fmt_name(fmt) << " op=0x" << std::hex << entry.opcode
                                   << " is " << entry.name << " in llvm-mc";
            EXPECT_STREQ(e.name, entry.name)
                << fmt_name(fmt) << " op=0x" << std::hex << entry.opcode;
        }
        for (uint32_t op = 0; op < rdna2_effects_opcode_space(fmt); ++op)
            if (!llvm_ops.contains(op))
                EXPECT_EQ(rdna2_instruction_effects(fmt, op).status, Rdna2EffectsStatus::Reserved)
                    << fmt_name(fmt) << " op=0x" << std::hex << op << " is invalid in llvm-mc";
    }
}

TEST(Rdna2InstructionEffects, UntabulatedLookupsAreUnknownNotNoEffect) {
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::VOP3, 0x16d).status,
              Rdna2EffectsStatus::Unknown);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::Unknown, 0).status,
              Rdna2EffectsStatus::Unknown);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x1000).status,
              Rdna2EffectsStatus::Unknown);
    EXPECT_EQ(rdna2_writes_scc(Rdna2Format::Unknown, 0), Rdna2Answer::Unknown);
    EXPECT_EQ(rdna2_reads_scc_implicitly(Rdna2Format::Unknown, 0), Rdna2Answer::Unknown);
    // A reserved scalar encoding is not a known non-writer either.
    EXPECT_EQ(rdna2_writes_scc(Rdna2Format::SOPC, 0x10), Rdna2Answer::Unknown);
    EXPECT_EQ(rdna2_reads_scc_implicitly(Rdna2Format::SOP2, 0x0c), Rdna2Answer::Unknown);
    // VALU, memory and export formats never touch SCC implicitly (ISA 3.8 / 3.9).
    for (Rdna2Format fmt :
         {Rdna2Format::VOP1, Rdna2Format::VOP2, Rdna2Format::VOPC, Rdna2Format::VOP3,
          Rdna2Format::VOP3P, Rdna2Format::SMEM, Rdna2Format::MIMG, Rdna2Format::EXP}) {
        EXPECT_EQ(rdna2_writes_scc(fmt, 0x10), Rdna2Answer::No);
        EXPECT_EQ(rdna2_reads_scc_implicitly(fmt, 0x10), Rdna2Answer::No);
    }
}

// ---- Effects against the ISA -------------------------------------------------------------------

TEST(Rdna2InstructionEffects, SccWritersMatchIsaTables) {
    // ISA 70648 5.3-5.8 "Sets SCC? y" and the 12.x pseudo-code (SCC = ...).
    const auto writers = [](Rdna2Format fmt, uint32_t op) {
        switch (fmt) {
            case Rdna2Format::SOP2:   // all but cselect, bfm, mul_i32, the packs and mul_hi
                return op <= 0x09 || (op >= 0x0e && op <= 0x23) || (op >= 0x27 && op <= 0x2a) ||
                       op == 0x2c || (op >= 0x2e && op <= 0x31);
            case Rdna2Format::SOP1:   // not, wqm, bcnt, quadmask, abs, every saveexec/wrexec
                return (op >= 0x07 && op <= 0x0a) || (op >= 0x0d && op <= 0x10) ||
                       (op >= 0x24 && op <= 0x2d) || op == 0x34 || (op >= 0x37 && op <= 0x3a) ||
                       (op >= 0x3c && op <= 0x47);
            case Rdna2Format::SOPK: return op >= 0x03 && op <= 0x0f;   // cmpk, addk
            case Rdna2Format::SOPC: return true;
            default: return false;
        }
    };
    for (Rdna2Format fmt : kTabulated)
        for (const LlvmOpcode& entry : llvm_set(fmt))
            EXPECT_EQ(rdna2_writes_scc(fmt, entry.opcode),
                      writers(fmt, entry.opcode) ? Rdna2Answer::Yes : Rdna2Answer::No)
                << entry.name;
}

TEST(Rdna2InstructionEffects, SccReadersMatchIsaTables) {
    // Data: s_addc/s_subb carry-in, s_cselect (ISA 5.4), s_cmov/s_cmovk (5.4). Branch: 12.5.
    for (Rdna2Format fmt : kTabulated) {
        for (const LlvmOpcode& entry : llvm_set(fmt)) {
            const uint32_t op = entry.opcode;
            const bool data = (fmt == Rdna2Format::SOP2 && in_set({0x04, 0x05, 0x0a, 0x0b}, op)) ||
                              (fmt == Rdna2Format::SOP1 && in_set({0x05, 0x06}, op)) ||
                              (fmt == Rdna2Format::SOPK && op == 0x02);
            const bool branch = fmt == Rdna2Format::SOPP && in_set({0x04, 0x05}, op);
            EXPECT_EQ(rdna2_instruction_effects(fmt, op).scc_read, data     ? Rdna2SccRead::Data
                                                                   : branch ? Rdna2SccRead::Branch
                                                                            : Rdna2SccRead::None)
                << entry.name;
        }
    }
}

TEST(Rdna2InstructionEffects, DestinationWidthsMatchIsaPseudoCode) {
    // D.u64 / D.i64 writers (ISA 12.x pseudo-code). Every other named SOP1/SOP2 writes one SGPR.
    const auto pair = [](Rdna2Format fmt, uint32_t op) {
        if (fmt == Rdna2Format::SOP2)
            return in_set({0x0b, 0x0f, 0x11, 0x13, 0x15, 0x17, 0x19, 0x1b, 0x1d, 0x1f, 0x21, 0x23,
                           0x25, 0x29, 0x2a},
                          op);
        if (fmt == Rdna2Format::SOP1)
            return in_set({0x04, 0x06, 0x08, 0x0a, 0x0c, 0x1c, 0x1e, 0x1f, 0x21, 0x2d, 0x2f, 0x31,
                           0x3b},
                          op) ||
                   (op >= 0x24 && op <= 0x2b) || (op >= 0x37 && op <= 0x3a);
        return fmt == Rdna2Format::SOPK && op == 0x16;   // s_call_b64
    };
    const auto none = [](Rdna2Format fmt, uint32_t op) {
        if (fmt == Rdna2Format::SOP1) return op == 0x20 || op == 0x22;   // s_setpc, s_rfe
        if (fmt ==
            Rdna2Format::SOPK)   // version, cmpk, setreg, waitcnt: SDST is a source or unused
            return op == 0x01 || (op >= 0x03 && op <= 0x0e) || op == 0x13 || op == 0x15 ||
                   (op >= 0x17 && op <= 0x1a);
        return fmt == Rdna2Format::SOPC || fmt == Rdna2Format::SOPP;
    };
    for (Rdna2Format fmt : kTabulated)
        for (const LlvmOpcode& entry : llvm_set(fmt))
            EXPECT_EQ(rdna2_instruction_effects(fmt, entry.opcode).sdst_dwords,
                      pair(fmt, entry.opcode)   ? 2u
                      : none(fmt, entry.opcode) ? 0u
                                                : 1u)
                << entry.name;
}

TEST(Rdna2InstructionEffects, ExecAndM0Effects) {
    for (uint32_t op = 0x24; op <= 0x2b; ++op)
        EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, op).exec, Rdna2ExecEffect::SaveExec);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x39).exec, Rdna2ExecEffect::WrExec);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x47).exec, Rdna2ExecEffect::WrExec);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x2e).m0_relative,
              Rdna2M0Relative::Source);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x30).m0_relative,
              Rdna2M0Relative::Destination);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x49).m0_relative,
              Rdna2M0Relative::Both);
    EXPECT_TRUE(rdna2_instruction_effects(Rdna2Format::SOPP, 0x06).implicit_reads &
                kRdna2ImplicitVcc);
    EXPECT_TRUE(rdna2_instruction_effects(Rdna2Format::SOPP, 0x08).implicit_reads &
                kRdna2ImplicitExec);
    // A SOPP never writes a register.
    for (const LlvmOpcode& entry : kLlvmSopp)
        EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOPP, entry.opcode).implicit_writes, 0u)
            << entry.name;
}

// ---- Regressions: each of these was answered wrongly by a private list -------------------------

// The two copies of conservative_scalar_write_width "disagreed" about a VOPC destination (1 SGPR
// in the descriptor-range proof, 2 in the pointer analysis). Both were right for the one wave size
// their own proof requires, so the shared copy takes the wave size; charging a Wave32 proof a pair
// refuses its exact shape. Both copies charged s_call_b64 one SGPR, but it writes a pair, and
// both charged a Wave64 VOP3B carry-out one SGPR, but it is a pair too.
TEST(Rdna2InstructionEffectsRegression, ConservativeWidthChargesEveryPairDestination) {
    Rdna2Inst compare;
    compare.fmt = Rdna2Format::VOPC;
    compare.opcode = 0xc1;   // v_cmp_lt_u32, e64 form writing s[8:9] (Wave64) or s8 (Wave32)
    compare.dst = {OperandKind::SGPR, 8};
    EXPECT_EQ(conservative_scalar_write_width(compare, IndirectWave::Wave64), 2u);
    EXPECT_EQ(conservative_scalar_write_width(compare, IndirectWave::Wave32), 1u);
    const Rdna2Inst call = decode(Rdna2Format::SOPK, 0x16);
    EXPECT_EQ(conservative_scalar_write_width(call, IndirectWave::Wave32), 2u)
        << "s_call_b64 writes a pair";
    const Rdna2Inst movk = decode(Rdna2Format::SOPK, 0x00);
    EXPECT_EQ(conservative_scalar_write_width(movk, IndirectWave::Wave64), 1u);
    // So a Wave64 compare writing s[8:9] clobbers a live range starting at s9; in Wave32 it does not.
    compare.pc = 2;
    EXPECT_FALSE(scalar_range_unchanged({compare}, 1u, 3u, 9u, 1u, IndirectWave::Wave64));
    EXPECT_TRUE(scalar_range_unchanged({compare}, 1u, 3u, 9u, 1u, IndirectWave::Wave32));
    // A Wave64 v_add_co_u32's carry-out s[4:5] clobbers s5.
    Rdna2Inst carry;
    carry.fmt = Rdna2Format::VOP3;
    carry.opcode = kVop3OpcodeAddCoU32;
    carry.pc = 2;
    carry.dst = {OperandKind::VGPR, 0};
    carry.sdst = {OperandKind::SGPR, 4};
    EXPECT_FALSE(scalar_range_unchanged({carry}, 1u, 3u, 5u, 1u, IndirectWave::Wave64));
    EXPECT_TRUE(scalar_range_unchanged({carry}, 1u, 3u, 5u, 1u, IndirectWave::Wave32));
}
