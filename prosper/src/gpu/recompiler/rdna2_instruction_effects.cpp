// rdna2_instruction_effects.cpp — the per-opcode effects table. See the header for the contract.
//
// Every row names its source. "ISA 12.x" is the per-format opcode table of AMD doc 70648, which
// gives each instruction's pseudo-code (what D, SCC, EXEC and M0 receive); "ISA 5.x" is the
// chapter-5 summary tables with their "Sets SCC?" column. The opcode numbers and the reserved
// holes were cross-checked against `llvm-mc -mcpu=gfx1030 -disassemble` over each whole opcode
// space; where the two sources differ the row says so and carries a CONFIDENCE mark.
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <array>
#include <cstddef>

namespace prosper::gpu {
namespace {

using F = Rdna2Format;
using Row = Rdna2InstructionEffectsRow;

constexpr uint8_t kScc = kRdna2ImplicitScc;
constexpr uint8_t kVcc = kRdna2ImplicitVcc;
constexpr uint8_t kExec = kRdna2ImplicitExec;
constexpr uint8_t kM0 = kRdna2ImplicitM0;

// One named instruction. `dst`/`s0`/`s1` are dword counts; `reads`/`writes` are implicit registers.
constexpr Rdna2InstructionEffects op(const char* name, uint8_t dst, uint8_t s0, uint8_t s1,
                                     uint8_t reads = 0, uint8_t writes = 0,
                                     Rdna2SccRead scc_read = Rdna2SccRead::None,
                                     Rdna2ExecEffect exec = Rdna2ExecEffect::None,
                                     Rdna2M0Relative relative = Rdna2M0Relative::None) {
    Rdna2InstructionEffects e;
    e.status = Rdna2EffectsStatus::Known;
    e.name = name;
    e.scc_read = scc_read;
    e.implicit_reads = reads;
    e.implicit_writes = writes;
    e.sdst_dwords = dst;
    e.ssrc_dwords[0] = s0;
    e.ssrc_dwords[1] = s1;
    e.m0_relative = relative;
    e.exec = exec;
    return e;
}
// Common shapes.
constexpr Rdna2InstructionEffects alu(const char* name, uint8_t dst, uint8_t s0, uint8_t s1,
                                      bool writes_scc) {
    return op(name, dst, s0, s1, 0, writes_scc ? kScc : 0);
}
constexpr Rdna2InstructionEffects scc_data(const char* name, uint8_t dst, uint8_t s0, uint8_t s1,
                                           bool writes_scc = false) {
    return op(name, dst, s0, s1, kScc, writes_scc ? kScc : 0, Rdna2SccRead::Data);
}
constexpr Rdna2InstructionEffects saveexec(const char* name, uint8_t width) {
    return op(name, width, width, 0, kExec, kExec | kScc, Rdna2SccRead::None,
              Rdna2ExecEffect::SaveExec);
}
constexpr Rdna2InstructionEffects wrexec(const char* name, uint8_t width) {
    return op(name, width, width, 0, kExec, kExec | kScc, Rdna2SccRead::None,
              Rdna2ExecEffect::WrExec);
}
constexpr Rdna2InstructionEffects compare(const char* name, uint8_t s0, uint8_t s1) {
    return op(name, 0, s0, s1, 0, kScc);
}
constexpr Rdna2InstructionEffects plain(const char* name) {
    return op(name, 0, 0, 0);
}
constexpr Rdna2InstructionEffects reserved() {
    Rdna2InstructionEffects e;
    e.status = Rdna2EffectsStatus::Reserved;
    return e;
}

// ---- SOP2: ISA 12.1; SCC column from ISA 5.3/5.5/5.7 (tables "Sets SCC?"). CONFIDENCE: HIGH --
// every row's pseudo-code states D's width and whether SCC is assigned.
constexpr Row kSop2[] = {
    {F::SOP2, 0x00, alu("s_add_u32", 1, 1, 1, true)},
    {F::SOP2, 0x01, alu("s_sub_u32", 1, 1, 1, true)},
    {F::SOP2, 0x02, alu("s_add_i32", 1, 1, 1, true)},
    {F::SOP2, 0x03, alu("s_sub_i32", 1, 1, 1, true)},
    {F::SOP2, 0x04, scc_data("s_addc_u32", 1, 1, 1, true)},   // D = S0 + S1 + SCC
    {F::SOP2, 0x05, scc_data("s_subb_u32", 1, 1, 1, true)},   // D = S0 - S1 - SCC
    {F::SOP2, 0x06, alu("s_min_i32", 1, 1, 1, true)},
    {F::SOP2, 0x07, alu("s_min_u32", 1, 1, 1, true)},
    {F::SOP2, 0x08, alu("s_max_i32", 1, 1, 1, true)},
    {F::SOP2, 0x09, alu("s_max_u32", 1, 1, 1, true)},
    {F::SOP2, 0x0a, scc_data("s_cselect_b32", 1, 1, 1)},   // D = SCC ? S0 : S1; SCC untouched
    {F::SOP2, 0x0b, scc_data("s_cselect_b64", 2, 2, 2)},
    {F::SOP2, 0x0e, alu("s_and_b32", 1, 1, 1, true)},
    {F::SOP2, 0x0f, alu("s_and_b64", 2, 2, 2, true)},
    {F::SOP2, 0x10, alu("s_or_b32", 1, 1, 1, true)},
    {F::SOP2, 0x11, alu("s_or_b64", 2, 2, 2, true)},
    {F::SOP2, 0x12, alu("s_xor_b32", 1, 1, 1, true)},
    {F::SOP2, 0x13, alu("s_xor_b64", 2, 2, 2, true)},
    {F::SOP2, 0x14, alu("s_andn2_b32", 1, 1, 1, true)},
    {F::SOP2, 0x15, alu("s_andn2_b64", 2, 2, 2, true)},
    {F::SOP2, 0x16, alu("s_orn2_b32", 1, 1, 1, true)},
    {F::SOP2, 0x17, alu("s_orn2_b64", 2, 2, 2, true)},
    {F::SOP2, 0x18, alu("s_nand_b32", 1, 1, 1, true)},
    {F::SOP2, 0x19, alu("s_nand_b64", 2, 2, 2, true)},
    {F::SOP2, 0x1a, alu("s_nor_b32", 1, 1, 1, true)},
    {F::SOP2, 0x1b, alu("s_nor_b64", 2, 2, 2, true)},
    {F::SOP2, 0x1c, alu("s_xnor_b32", 1, 1, 1, true)},
    {F::SOP2, 0x1d, alu("s_xnor_b64", 2, 2, 2, true)},
    {F::SOP2, 0x1e, alu("s_lshl_b32", 1, 1, 1, true)},
    {F::SOP2, 0x1f, alu("s_lshl_b64", 2, 2, 1, true)},   // the shift count is S1.u[5:0]
    {F::SOP2, 0x20, alu("s_lshr_b32", 1, 1, 1, true)},
    {F::SOP2, 0x21, alu("s_lshr_b64", 2, 2, 1, true)},
    {F::SOP2, 0x22, alu("s_ashr_i32", 1, 1, 1, true)},
    {F::SOP2, 0x23, alu("s_ashr_i64", 2, 2, 1, true)},
    {F::SOP2, 0x24, alu("s_bfm_b32", 1, 1, 1, false)},
    {F::SOP2, 0x25, alu("s_bfm_b64", 2, 1, 1, false)},   // width and offset are B32
    {F::SOP2, 0x26, alu("s_mul_i32", 1, 1, 1, false)},
    {F::SOP2, 0x27, alu("s_bfe_u32", 1, 1, 1, true)},
    {F::SOP2, 0x28, alu("s_bfe_i32", 1, 1, 1, true)},
    {F::SOP2, 0x29, alu("s_bfe_u64", 2, 2, 1, true)},   // S1 holds offset/width: B32
    {F::SOP2, 0x2a, alu("s_bfe_i64", 2, 2, 1, true)},
    {F::SOP2, 0x2c, alu("s_absdiff_i32", 1, 1, 1, true)},
    {F::SOP2, 0x2e, alu("s_lshl1_add_u32", 1, 1, 1, true)},
    {F::SOP2, 0x2f, alu("s_lshl2_add_u32", 1, 1, 1, true)},
    {F::SOP2, 0x30, alu("s_lshl3_add_u32", 1, 1, 1, true)},
    {F::SOP2, 0x31, alu("s_lshl4_add_u32", 1, 1, 1, true)},
    {F::SOP2, 0x32, alu("s_pack_ll_b32_b16", 1, 1, 1, false)},
    {F::SOP2, 0x33, alu("s_pack_lh_b32_b16", 1, 1, 1, false)},
    {F::SOP2, 0x34, alu("s_pack_hh_b32_b16", 1, 1, 1, false)},
    {F::SOP2, 0x35, alu("s_mul_hi_u32", 1, 1, 1, false)},
    {F::SOP2, 0x36, alu("s_mul_hi_i32", 1, 1, 1, false)},
};

// ---- SOP1: ISA 12.3; SCC column from ISA 5.5/5.7 Table 14. CONFIDENCE: HIGH unless noted.
constexpr Row kSop1[] = {
    {F::SOP1, 0x03, alu("s_mov_b32", 1, 1, 0, false)},
    {F::SOP1, 0x04, alu("s_mov_b64", 2, 2, 0, false)},
    // if (SCC) D = S0. Both D words may change, and when SCC is 0 the old D survives.
    {F::SOP1, 0x05, scc_data("s_cmov_b32", 1, 1, 0)},
    {F::SOP1, 0x06, scc_data("s_cmov_b64", 2, 2, 0)},
    {F::SOP1, 0x07, alu("s_not_b32", 1, 1, 0, true)},
    {F::SOP1, 0x08, alu("s_not_b64", 2, 2, 0, true)},
    {F::SOP1, 0x09, alu("s_wqm_b32", 1, 1, 0, true)},
    {F::SOP1, 0x0a, alu("s_wqm_b64", 2, 2, 0, true)},
    {F::SOP1, 0x0b, alu("s_brev_b32", 1, 1, 0, false)},
    {F::SOP1, 0x0c, alu("s_brev_b64", 2, 2, 0, false)},
    {F::SOP1, 0x0d, alu("s_bcnt0_i32_b32", 1, 1, 0, true)},
    {F::SOP1, 0x0e, alu("s_bcnt0_i32_b64", 1, 2, 0, true)},
    {F::SOP1, 0x0f, alu("s_bcnt1_i32_b32", 1, 1, 0, true)},
    {F::SOP1, 0x10, alu("s_bcnt1_i32_b64", 1, 2, 0, true)},
    {F::SOP1, 0x11, alu("s_ff0_i32_b32", 1, 1, 0, false)},
    {F::SOP1, 0x12, alu("s_ff0_i32_b64", 1, 2, 0, false)},
    {F::SOP1, 0x13, alu("s_ff1_i32_b32", 1, 1, 0, false)},
    {F::SOP1, 0x14, alu("s_ff1_i32_b64", 1, 2, 0, false)},
    {F::SOP1, 0x15, alu("s_flbit_i32_b32", 1, 1, 0, false)},
    {F::SOP1, 0x16, alu("s_flbit_i32_b64", 1, 2, 0, false)},
    {F::SOP1, 0x17, alu("s_flbit_i32", 1, 1, 0, false)},
    {F::SOP1, 0x18, alu("s_flbit_i32_i64", 1, 2, 0, false)},
    {F::SOP1, 0x19, alu("s_sext_i32_i8", 1, 1, 0, false)},
    {F::SOP1, 0x1a, alu("s_sext_i32_i16", 1, 1, 0, false)},
    // D[S0[4:0]] = 0/1 (B64: D[S0[5:0]]): a read-modify-write of D whose source is a bit index.
    {F::SOP1, 0x1b, alu("s_bitset0_b32", 1, 1, 0, false)},
    {F::SOP1, 0x1c, alu("s_bitset0_b64", 2, 1, 0, false)},
    {F::SOP1, 0x1d, alu("s_bitset1_b32", 1, 1, 0, false)},
    {F::SOP1, 0x1e, alu("s_bitset1_b64", 2, 1, 0, false)},
    {F::SOP1, 0x1f, alu("s_getpc_b64", 2, 0, 0, false)},   // D = PC + 4; no source
    {F::SOP1, 0x20, alu("s_setpc_b64", 0, 2, 0, false)},   // PC = S0; D is not written
    {F::SOP1, 0x21, alu("s_swappc_b64", 2, 2, 0, false)},   // D = PC + 4; PC = S0
    {F::SOP1, 0x22, alu("s_rfe_b64", 0, 2, 0, false)},   // PC = S0 (trap return)
    {F::SOP1, 0x24, saveexec("s_and_saveexec_b64", 2)},
    {F::SOP1, 0x25, saveexec("s_or_saveexec_b64", 2)},
    {F::SOP1, 0x26, saveexec("s_xor_saveexec_b64", 2)},
    {F::SOP1, 0x27, saveexec("s_andn2_saveexec_b64", 2)},
    {F::SOP1, 0x28, saveexec("s_orn2_saveexec_b64", 2)},
    {F::SOP1, 0x29, saveexec("s_nand_saveexec_b64", 2)},
    {F::SOP1, 0x2a, saveexec("s_nor_saveexec_b64", 2)},
    {F::SOP1, 0x2b, saveexec("s_xnor_saveexec_b64", 2)},
    {F::SOP1, 0x2c, alu("s_quadmask_b32", 1, 1, 0, true)},
    {F::SOP1, 0x2d, alu("s_quadmask_b64", 2, 2, 0, true)},
    // SGPR[D] = SGPR[S0 + M0] / SGPR[D + M0] = SGPR[S0]. The widths are those of the moved value;
    // which registers are touched depends on M0 at run time.
    {F::SOP1, 0x2e,
     op("s_movrels_b32", 1, 1, 0, kM0, 0, Rdna2SccRead::None, Rdna2ExecEffect::None,
        Rdna2M0Relative::Source)},
    {F::SOP1, 0x2f,
     op("s_movrels_b64", 2, 2, 0, kM0, 0, Rdna2SccRead::None, Rdna2ExecEffect::None,
        Rdna2M0Relative::Source)},
    {F::SOP1, 0x30,
     op("s_movreld_b32", 1, 1, 0, kM0, 0, Rdna2SccRead::None, Rdna2ExecEffect::None,
        Rdna2M0Relative::Destination)},
    {F::SOP1, 0x31,
     op("s_movreld_b64", 2, 2, 0, kM0, 0, Rdna2SccRead::None, Rdna2ExecEffect::None,
        Rdna2M0Relative::Destination)},
    {F::SOP1, 0x34, alu("s_abs_i32", 1, 1, 0, true)},
    {F::SOP1, 0x37, saveexec("s_andn1_saveexec_b64", 2)},
    {F::SOP1, 0x38, saveexec("s_orn1_saveexec_b64", 2)},
    {F::SOP1, 0x39, wrexec("s_andn1_wrexec_b64", 2)},   // EXEC = ~S0 & EXEC; D = EXEC
    {F::SOP1, 0x3a, wrexec("s_andn2_wrexec_b64", 2)},
    {F::SOP1, 0x3b, alu("s_bitreplicate_b64_b32", 2, 1, 0, false)},
    // The B32 forms read and write EXEC_LO only.
    {F::SOP1, 0x3c, saveexec("s_and_saveexec_b32", 1)},
    {F::SOP1, 0x3d, saveexec("s_or_saveexec_b32", 1)},
    {F::SOP1, 0x3e, saveexec("s_xor_saveexec_b32", 1)},
    {F::SOP1, 0x3f, saveexec("s_andn2_saveexec_b32", 1)},
    {F::SOP1, 0x40, saveexec("s_orn2_saveexec_b32", 1)},
    {F::SOP1, 0x41, saveexec("s_nand_saveexec_b32", 1)},
    {F::SOP1, 0x42, saveexec("s_nor_saveexec_b32", 1)},
    {F::SOP1, 0x43, saveexec("s_xnor_saveexec_b32", 1)},
    {F::SOP1, 0x44, saveexec("s_andn1_saveexec_b32", 1)},
    {F::SOP1, 0x45, saveexec("s_orn1_saveexec_b32", 1)},
    {F::SOP1, 0x46, wrexec("s_andn1_wrexec_b32", 1)},
    {F::SOP1, 0x47, wrexec("s_andn2_wrexec_b32", 1)},
    // SGPR[D + M0[25:16]] = SGPR[S0 + M0[9:0]].
    {F::SOP1, 0x49,
     op("s_movrelsd_2_b32", 1, 1, 0, kM0, 0, Rdna2SccRead::None, Rdna2ExecEffect::None,
        Rdna2M0Relative::Both)},
};

// ---- SOPK: ISA 12.2 and Table 15. The SDST field is a destination for movk/cmovk/addk/mulk/
// getreg/call and the subvector loops, and a SOURCE for cmpk, setreg and the waitcnt forms.
constexpr Row kSopk[] = {
    {F::SOPK, 0x00, alu("s_movk_i32", 1, 0, 0, false)},
    {F::SOPK, 0x01, plain("s_version")},
    {F::SOPK, 0x02, scc_data("s_cmovk_i32", 1, 0, 0)},   // if (SCC) D = signext(SIMM16)
    {F::SOPK, 0x03, compare("s_cmpk_eq_i32", 0, 0)},
    {F::SOPK, 0x04, compare("s_cmpk_lg_i32", 0, 0)},
    {F::SOPK, 0x05, compare("s_cmpk_gt_i32", 0, 0)},
    {F::SOPK, 0x06, compare("s_cmpk_ge_i32", 0, 0)},
    {F::SOPK, 0x07, compare("s_cmpk_lt_i32", 0, 0)},
    {F::SOPK, 0x08, compare("s_cmpk_le_i32", 0, 0)},
    {F::SOPK, 0x09, compare("s_cmpk_eq_u32", 0, 0)},
    {F::SOPK, 0x0a, compare("s_cmpk_lg_u32", 0, 0)},
    {F::SOPK, 0x0b, compare("s_cmpk_gt_u32", 0, 0)},
    {F::SOPK, 0x0c, compare("s_cmpk_ge_u32", 0, 0)},
    {F::SOPK, 0x0d, compare("s_cmpk_lt_u32", 0, 0)},
    {F::SOPK, 0x0e, compare("s_cmpk_le_u32", 0, 0)},
    {F::SOPK, 0x0f, alu("s_addk_i32", 1, 0, 0, true)},   // D += SIMM16; SCC = signed overflow
    {F::SOPK, 0x10, alu("s_mulk_i32", 1, 0, 0, false)},
    {F::SOPK, 0x12, alu("s_getreg_b32", 1, 0, 0, false)},
    // Writes a hardware register from the SDST-field SGPR. Table 15 lists no SCC update.
    // CONFIDENCE: MED -- the hardware register id is SIMM16 data, and no row of the ISA says what
    // a write to HW_REG_STATUS (which holds SCC) does; no guest has been observed doing it.
    {F::SOPK, 0x13, plain("s_setreg_b32")},
    {F::SOPK, 0x15, plain("s_setreg_imm32_b32")},
    {F::SOPK, 0x16, alu("s_call_b64", 2, 0, 0, false)},   // D = PC + 4; PC += SIMM16 * 4 + 4
    {F::SOPK, 0x17, plain("s_waitcnt_vscnt")},
    {F::SOPK, 0x18, plain("s_waitcnt_vmcnt")},
    {F::SOPK, 0x19, plain("s_waitcnt_expcnt")},
    {F::SOPK, 0x1a, plain("s_waitcnt_lgkmcnt")},
    // ISA 12.2 pseudo-code: begin writes D0 = EXEC_LO or EXEC_HI and may clear EXEC_HI; end may
    // write EXEC_LO/EXEC_HI from D0 and D0 from EXEC_LO. Both read the SDST field as the saved
    // half. CONFIDENCE: MED -- the D write is conditional on EXEC, which this records as a write.
    {F::SOPK, 0x1b,
     op("s_subvector_loop_begin", 1, 0, 0, kExec, kExec, Rdna2SccRead::None,
        Rdna2ExecEffect::SubvectorLoop)},
    {F::SOPK, 0x1c,
     op("s_subvector_loop_end", 1, 0, 0, kExec, kExec, Rdna2SccRead::None,
        Rdna2ExecEffect::SubvectorLoop)},
    // 0x1D-0x1F are not SOPK: those encodings decode as SOP1/SOPC/SOPP first.
};

// ---- SOPC: ISA 12.4. Every compare writes SCC and nothing else. CONFIDENCE: HIGH.
constexpr Row kSopc[] = {
    {F::SOPC, 0x00, compare("s_cmp_eq_i32", 1, 1)},
    {F::SOPC, 0x01, compare("s_cmp_lg_i32", 1, 1)},
    {F::SOPC, 0x02, compare("s_cmp_gt_i32", 1, 1)},
    {F::SOPC, 0x03, compare("s_cmp_ge_i32", 1, 1)},
    {F::SOPC, 0x04, compare("s_cmp_lt_i32", 1, 1)},
    {F::SOPC, 0x05, compare("s_cmp_le_i32", 1, 1)},
    {F::SOPC, 0x06, compare("s_cmp_eq_u32", 1, 1)},
    {F::SOPC, 0x07, compare("s_cmp_lg_u32", 1, 1)},
    {F::SOPC, 0x08, compare("s_cmp_gt_u32", 1, 1)},
    {F::SOPC, 0x09, compare("s_cmp_ge_u32", 1, 1)},
    {F::SOPC, 0x0a, compare("s_cmp_lt_u32", 1, 1)},
    {F::SOPC, 0x0b, compare("s_cmp_le_u32", 1, 1)},
    {F::SOPC, 0x0c, compare("s_bitcmp0_b32", 1, 1)},
    {F::SOPC, 0x0d, compare("s_bitcmp1_b32", 1, 1)},
    {F::SOPC, 0x0e, compare("s_bitcmp0_b64", 2, 1)},
    {F::SOPC, 0x0f, compare("s_bitcmp1_b64", 2, 1)},
    {F::SOPC, 0x12, compare("s_cmp_eq_u64", 2, 2)},
    {F::SOPC, 0x13, compare("s_cmp_lg_u64", 2, 2)},
};

// ---- SOPP: ISA 12.5. No SOPP instruction writes an SGPR or SCC. CONFIDENCE: HIGH unless noted.
constexpr Row kSopp[] = {
    {F::SOPP, 0x00, plain("s_nop")},
    {F::SOPP, 0x01, plain("s_endpgm")},
    {F::SOPP, 0x02, plain("s_branch")},
    {F::SOPP, 0x03, plain("s_wakeup")},
    {F::SOPP, 0x04, op("s_cbranch_scc0", 0, 0, 0, kScc, 0, Rdna2SccRead::Branch)},
    {F::SOPP, 0x05, op("s_cbranch_scc1", 0, 0, 0, kScc, 0, Rdna2SccRead::Branch)},
    {F::SOPP, 0x06, op("s_cbranch_vccz", 0, 0, 0, kVcc)},
    {F::SOPP, 0x07, op("s_cbranch_vccnz", 0, 0, 0, kVcc)},
    {F::SOPP, 0x08, op("s_cbranch_execz", 0, 0, 0, kExec)},
    {F::SOPP, 0x09, op("s_cbranch_execnz", 0, 0, 0, kExec)},
    {F::SOPP, 0x0a, plain("s_barrier")},
    {F::SOPP, 0x0b, plain("s_setkill")},
    {F::SOPP, 0x0c, plain("s_waitcnt")},
    {F::SOPP, 0x0d, plain("s_sethalt")},
    {F::SOPP, 0x0e, plain("s_sleep")},
    {F::SOPP, 0x0f, plain("s_setprio")},
    // "The S_SENDMSG instruction encodes the message type in M0, and can also send data from the
    // SIMM16 field and in some cases from EXEC" (ISA 12.5). CONFIDENCE: MED for the EXEC read,
    // which depends on the message.
    {F::SOPP, 0x10, op("s_sendmsg", 0, 0, 0, kM0 | kExec)},
    {F::SOPP, 0x11, op("s_sendmsghalt", 0, 0, 0, kM0 | kExec)},
    {F::SOPP, 0x12, plain("s_trap")},
    {F::SOPP, 0x13, plain("s_icache_inv")},
    {F::SOPP, 0x14, plain("s_incperflevel")},
    {F::SOPP, 0x15, plain("s_decperflevel")},
    {F::SOPP, 0x16, op("s_ttracedata", 0, 0, 0, kM0)},   // sends M0 to the thread trace
    {F::SOPP, 0x17, plain("s_cbranch_cdbgsys")},
    {F::SOPP, 0x18, plain("s_cbranch_cdbguser")},
    {F::SOPP, 0x19, plain("s_cbranch_cdbgsys_or_user")},
    {F::SOPP, 0x1a, plain("s_cbranch_cdbgsys_and_user")},
    {F::SOPP, 0x1b, plain("s_endpgm_saved")},
    {F::SOPP, 0x1e, plain("s_endpgm_ordered_ps_done")},
    {F::SOPP, 0x1f, plain("s_code_end")},
    {F::SOPP, 0x20, plain("s_inst_prefetch")},
    {F::SOPP, 0x21, plain("s_clause")},
    // CONFIDENCE: MED -- llvm-mc disassembles 0x22 as s_wait_idle for gfx1030, but ISA 12.5 skips
    // opcode 34. Either way it has no register effect.
    {F::SOPP, 0x22, plain("s_wait_idle")},
    {F::SOPP, 0x23, plain("s_waitcnt_depctr")},
    {F::SOPP, 0x24, plain("s_round_mode")},
    {F::SOPP, 0x25, plain("s_denorm_mode")},
    {F::SOPP, 0x28, plain("s_ttracedata_imm")},
};

// Encodings the decoder accepts but RDNA2 does not assign, listed EXPLICITLY rather than inferred
// from the gaps between named rows: deleting a named row must leave its slot Unknown, so the
// coverage test notices, instead of quietly turning it into a reserved encoding. Each range is
// confirmed twice: the ISA 12.x opcode table skips it, and llvm-mc -mcpu=gfx1030 rejects it.
struct ReservedRange {
    Rdna2Format fmt;
    uint32_t first;
    uint32_t last;
};
constexpr ReservedRange kReserved[] = {
    {F::SOP2, 0x0c, 0x0d},
    {F::SOP2, 0x2b, 0x2b},
    {F::SOP2, 0x2d, 0x2d},
    {F::SOP2, 0x37, 0x5f},   // the SALU-float SOP2 block (0x40+) exists on gfx11 only
    {F::SOP1, 0x00, 0x02},
    {F::SOP1, 0x23, 0x23},
    {F::SOP1, 0x32, 0x33},
    {F::SOP1, 0x35, 0x36},
    {F::SOP1, 0x48, 0x48},
    {F::SOP1, 0x4a, 0xff},
    {F::SOPK, 0x11, 0x11},
    {F::SOPK, 0x14, 0x14},
    {F::SOPC, 0x10, 0x11},   // s_setvskip / s_set_gpr_idx_on are pre-RDNA encodings
    {F::SOPC, 0x14, 0x7f},
    {F::SOPP, 0x1c, 0x1d},
    {F::SOPP, 0x26, 0x27},
    {F::SOPP, 0x29, 0x7f},
};

// The decoder's opcode field widths, minus encodings another format claims first
// (rdna2_decode_one: SOPP, SOPC, SOP1 and SOPK are carved out of the SOP2 prefix).
// SOP2: bits[29:23]; 0x60-0x7F are the SOPK/SOP1/SOPC/SOPP prefixes.
constexpr uint32_t kSop2Space = 0x60;
// SOP1: bits[15:8].
constexpr uint32_t kSop1Space = 0x100;
// SOPK: bits[27:23]; 0x1D/0x1E/0x1F are the SOP1/SOPC/SOPP prefixes.
constexpr uint32_t kSopkSpace = 0x1d;
// SOPC and SOPP: bits[22:16].
constexpr uint32_t kSopcSpace = 0x80;
constexpr uint32_t kSoppSpace = 0x80;

constexpr size_t kMaxSpace = 0x100;

struct FormatTable {
    std::array<Rdna2InstructionEffects, kMaxSpace> by_opcode{};
};

// Dense lookup for one format: its named rows plus its explicit reserved ranges. Every other slot
// stays Unknown.
template <size_t N>
constexpr FormatTable dense(Rdna2Format fmt, const Row (&rows)[N]) {
    FormatTable table;
    for (const ReservedRange& range : kReserved)
        if (range.fmt == fmt)
            for (uint32_t opcode = range.first; opcode <= range.last; ++opcode)
                table.by_opcode[opcode] = reserved();
    for (const Row& row : rows) table.by_opcode[row.opcode] = row.effects;
    return table;
}

constexpr FormatTable kSop2Table = dense(F::SOP2, kSop2);
constexpr FormatTable kSop1Table = dense(F::SOP1, kSop1);
constexpr FormatTable kSopkTable = dense(F::SOPK, kSopk);
constexpr FormatTable kSopcTable = dense(F::SOPC, kSopc);
constexpr FormatTable kSoppTable = dense(F::SOPP, kSopp);

const FormatTable* table_for(Rdna2Format fmt) {
    switch (fmt) {
        case F::SOP2: return &kSop2Table;
        case F::SOP1: return &kSop1Table;
        case F::SOPK: return &kSopkTable;
        case F::SOPC: return &kSopcTable;
        case F::SOPP: return &kSoppTable;
        default: return nullptr;
    }
}

std::vector<Row> all_rows() {
    std::vector<Row> rows;
    auto append = [&](const auto& list) {
        for (const Row& row : list) rows.push_back(row);
    };
    append(kSop2);
    append(kSop1);
    append(kSopk);
    append(kSopc);
    append(kSopp);
    for (const ReservedRange& range : kReserved)
        for (uint32_t opcode = range.first; opcode <= range.last; ++opcode)
            rows.push_back({range.fmt, opcode, reserved()});
    return rows;
}

const Rdna2InstructionEffects kUnknown{};

// Formats with no scalar-ALU encoding. None of them reads or writes SCC except through an explicit
// operand (ISA 70648 3.8: "Most scalar ALU instructions set the Scalar Condition Code"; VALU
// compares write VCC/EXEC instead, 3.9).
bool never_touches_scc(Rdna2Format fmt) {
    switch (fmt) {
        case F::SMEM:
        case F::VOP2:
        case F::VOP1:
        case F::VOPC:
        case F::VOP3:
        case F::VOP3P:
        case F::VINTRP:
        case F::DS:
        case F::MUBUF:
        case F::MTBUF:
        case F::MIMG:
        case F::FLAT:
        case F::EXP: return true;
        default: return false;
    }
}

}   // namespace

const Rdna2InstructionEffects& rdna2_instruction_effects(Rdna2Format fmt, uint32_t opcode) {
    const FormatTable* table = table_for(fmt);
    if (!table || opcode >= kMaxSpace) return kUnknown;
    return table->by_opcode[opcode];
}

bool rdna2_effects_format_is_tabulated(Rdna2Format fmt) {
    return table_for(fmt) != nullptr;
}

uint32_t rdna2_effects_opcode_space(Rdna2Format fmt) {
    switch (fmt) {
        case F::SOP2: return kSop2Space;
        case F::SOP1: return kSop1Space;
        case F::SOPK: return kSopkSpace;
        case F::SOPC: return kSopcSpace;
        case F::SOPP: return kSoppSpace;
        default: return 0;
    }
}

std::span<const Rdna2InstructionEffectsRow> rdna2_instruction_effects_rows() {
    static const std::vector<Row> rows = all_rows();
    return rows;
}

Rdna2Answer rdna2_writes_scc(Rdna2Format fmt, uint32_t opcode) {
    if (never_touches_scc(fmt)) return Rdna2Answer::No;
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(fmt, opcode);
    if (!e.known()) return Rdna2Answer::Unknown;
    return (e.implicit_writes & kScc) ? Rdna2Answer::Yes : Rdna2Answer::No;
}

Rdna2Answer rdna2_reads_scc_implicitly(Rdna2Format fmt, uint32_t opcode) {
    if (never_touches_scc(fmt)) return Rdna2Answer::No;
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(fmt, opcode);
    if (!e.known()) return Rdna2Answer::Unknown;
    return e.scc_read != Rdna2SccRead::None ? Rdna2Answer::Yes : Rdna2Answer::No;
}

Rdna2Answer rdna2_reads_scc_as_data(Rdna2Format fmt, uint32_t opcode) {
    if (never_touches_scc(fmt)) return Rdna2Answer::No;
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(fmt, opcode);
    if (!e.known()) return Rdna2Answer::Unknown;
    return e.scc_read == Rdna2SccRead::Data ? Rdna2Answer::Yes : Rdna2Answer::No;
}

bool rdna2_operand_reads_scc(const Rdna2Inst& in) {
    for (uint32_t source = 0; source < in.n_src && source < 4; ++source)
        if (in.src[source].kind == OperandKind::Special && in.src[source].value == 253) return true;
    return false;
}

bool rdna2_program_reads_scc(const std::vector<Rdna2Inst>& ins) {
    for (const Rdna2Inst& in : ins)
        if (rdna2_operand_reads_scc(in) ||
            rdna2_reads_scc_implicitly(in.fmt, in.opcode) != Rdna2Answer::No)
            return true;
    return false;
}

}   // namespace prosper::gpu
