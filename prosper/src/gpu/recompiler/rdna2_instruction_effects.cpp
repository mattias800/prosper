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
    e.ssrc_dwords[2] = 0;
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
// A VALU instruction described by its source widths only.
constexpr Rdna2InstructionEffects valu(const char* name, uint8_t s0, uint8_t s1, uint8_t s2) {
    Rdna2InstructionEffects e;
    e.status = Rdna2EffectsStatus::SourcesOnly;
    e.name = name;
    e.ssrc_dwords[0] = s0;
    e.ssrc_dwords[1] = s1;
    e.ssrc_dwords[2] = s2;
    return e;
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

// ---- VALU source widths: VOP2 (ISA 12.7), VOP1 (12.8), VOPC (12.9) and VOP3 (12.12; VOP2 and
// VOP1 promote to VOP3 at +0x100 and +0x180, VOPC keeps its number and decodes as VOPC). Only the
// name and the width of each source are described (status SourcesOnly).
//
// Derived by two independent routes, which were required to agree:
//   * llvm-mc -mcpu=gfx1030 -mattr=+wavefrontsize64 -disassemble over every encoding with SGPR
//     sources s2/s6/s10: the register class LLVM prints for each operand (s[2:3] is a pair);
//   * the ISA 70648 pseudo-code: a source is a pair when it is read as S<n>.u64/i64/f64/b64/d or
//     by a bit index above 31, or when a compare's name ends in _F64/_I64/_U64 (CLASS_F64: S0).
// They differed on 9 rows, all resolved in LLVM's favour: V_RSQ_F64 (ISA erratum, its pseudo-code
// reads S0.f16), V_QSAD_PK_U16_U8, V_MQSAD_PK_U16_U8 and V_MQSAD_U32_U8 (S0[55:24]-style bit
// ranges), and the lane-mask SRC2 of v_cndmask_b32 and the three carry-in forms, which the ISA
// leaves untyped and Wave64 reads as a pair. CONFIDENCE: HIGH.
//
// PS5 guest code also uses gfx10.1-era encodings that gfx1030 and ISA 70648 dropped, so the rows
// are the UNION of llvm-mc gfx1030 and gfx1013. From gfx1013: v_mac_f32 (VOP2 0x1F / VOP3 0x11F),
// v_madmk_f32 / v_madak_f32 (VOP2 0x20 / 0x21, about 14,000 uses in the local shader corpus),
// v_mad_f32 (VOP3 0x141, Kena's merged-GS prolog) and v_mul_lo_i32 (VOP3 0x16B). From gfx1030:
// v_dot2c_f32_f16 and v_dot4c_i32_i8. Where both define an opcode the widths agree, and the scalar
// formats above are identical in both.
// A VGPR-only source (v_readlane_b32's SRC0) is listed with its width; an attribute field that is
// not a register operand (the v_interp forms) is 0. A literal constant (v_madak's K) is not a source.
constexpr Row kVop2[] = {
    {F::VOP2, 0x001, valu("v_cndmask_b32", 1, 1, 0)},
    {F::VOP2, 0x002, valu("v_dot2c_f32_f16", 1, 1, 0)},
    {F::VOP2, 0x003, valu("v_add_f32", 1, 1, 0)},
    {F::VOP2, 0x004, valu("v_sub_f32", 1, 1, 0)},
    {F::VOP2, 0x005, valu("v_subrev_f32", 1, 1, 0)},
    {F::VOP2, 0x006, valu("v_fmac_legacy_f32", 1, 1, 0)},
    {F::VOP2, 0x007, valu("v_mul_legacy_f32", 1, 1, 0)},
    {F::VOP2, 0x008, valu("v_mul_f32", 1, 1, 0)},
    {F::VOP2, 0x009, valu("v_mul_i32_i24", 1, 1, 0)},
    {F::VOP2, 0x00a, valu("v_mul_hi_i32_i24", 1, 1, 0)},
    {F::VOP2, 0x00b, valu("v_mul_u32_u24", 1, 1, 0)},
    {F::VOP2, 0x00c, valu("v_mul_hi_u32_u24", 1, 1, 0)},
    {F::VOP2, 0x00d, valu("v_dot4c_i32_i8", 1, 1, 0)},
    {F::VOP2, 0x00f, valu("v_min_f32", 1, 1, 0)},
    {F::VOP2, 0x010, valu("v_max_f32", 1, 1, 0)},
    {F::VOP2, 0x011, valu("v_min_i32", 1, 1, 0)},
    {F::VOP2, 0x012, valu("v_max_i32", 1, 1, 0)},
    {F::VOP2, 0x013, valu("v_min_u32", 1, 1, 0)},
    {F::VOP2, 0x014, valu("v_max_u32", 1, 1, 0)},
    {F::VOP2, 0x016, valu("v_lshrrev_b32", 1, 1, 0)},
    {F::VOP2, 0x018, valu("v_ashrrev_i32", 1, 1, 0)},
    {F::VOP2, 0x01a, valu("v_lshlrev_b32", 1, 1, 0)},
    {F::VOP2, 0x01b, valu("v_and_b32", 1, 1, 0)},
    {F::VOP2, 0x01c, valu("v_or_b32", 1, 1, 0)},
    {F::VOP2, 0x01d, valu("v_xor_b32", 1, 1, 0)},
    {F::VOP2, 0x01e, valu("v_xnor_b32", 1, 1, 0)},
    {F::VOP2, 0x01f, valu("v_mac_f32", 1, 1, 0)},
    {F::VOP2, 0x020, valu("v_madmk_f32", 1, 1, 0)},
    {F::VOP2, 0x021, valu("v_madak_f32", 1, 1, 0)},
    {F::VOP2, 0x025, valu("v_add_nc_u32", 1, 1, 0)},
    {F::VOP2, 0x026, valu("v_sub_nc_u32", 1, 1, 0)},
    {F::VOP2, 0x027, valu("v_subrev_nc_u32", 1, 1, 0)},
    {F::VOP2, 0x028, valu("v_add_co_ci_u32", 1, 1, 0)},
    {F::VOP2, 0x029, valu("v_sub_co_ci_u32", 1, 1, 0)},
    {F::VOP2, 0x02a, valu("v_subrev_co_ci_u32", 1, 1, 0)},
    {F::VOP2, 0x02b, valu("v_fmac_f32", 1, 1, 0)},
    {F::VOP2, 0x02c, valu("v_fmamk_f32", 1, 1, 0)},
    {F::VOP2, 0x02d, valu("v_fmaak_f32", 1, 1, 0)},
    {F::VOP2, 0x02f, valu("v_cvt_pkrtz_f16_f32", 1, 1, 0)},
    {F::VOP2, 0x032, valu("v_add_f16", 1, 1, 0)},
    {F::VOP2, 0x033, valu("v_sub_f16", 1, 1, 0)},
    {F::VOP2, 0x034, valu("v_subrev_f16", 1, 1, 0)},
    {F::VOP2, 0x035, valu("v_mul_f16", 1, 1, 0)},
    {F::VOP2, 0x036, valu("v_fmac_f16", 1, 1, 0)},
    {F::VOP2, 0x037, valu("v_fmamk_f16", 1, 1, 0)},
    {F::VOP2, 0x038, valu("v_fmaak_f16", 1, 1, 0)},
    {F::VOP2, 0x039, valu("v_max_f16", 1, 1, 0)},
    {F::VOP2, 0x03a, valu("v_min_f16", 1, 1, 0)},
    {F::VOP2, 0x03b, valu("v_ldexp_f16", 1, 1, 0)},
    {F::VOP2, 0x03c, valu("v_pk_fmac_f16", 1, 1, 0)},
};
constexpr Row kVop1[] = {
    {F::VOP1, 0x001, valu("v_mov_b32", 1, 0, 0)},
    {F::VOP1, 0x002, valu("v_readfirstlane_b32", 1, 0, 0)},
    {F::VOP1, 0x003, valu("v_cvt_i32_f64", 2, 0, 0)},
    {F::VOP1, 0x004, valu("v_cvt_f64_i32", 1, 0, 0)},
    {F::VOP1, 0x005, valu("v_cvt_f32_i32", 1, 0, 0)},
    {F::VOP1, 0x006, valu("v_cvt_f32_u32", 1, 0, 0)},
    {F::VOP1, 0x007, valu("v_cvt_u32_f32", 1, 0, 0)},
    {F::VOP1, 0x008, valu("v_cvt_i32_f32", 1, 0, 0)},
    {F::VOP1, 0x00a, valu("v_cvt_f16_f32", 1, 0, 0)},
    {F::VOP1, 0x00b, valu("v_cvt_f32_f16", 1, 0, 0)},
    {F::VOP1, 0x00c, valu("v_cvt_rpi_i32_f32", 1, 0, 0)},
    {F::VOP1, 0x00d, valu("v_cvt_flr_i32_f32", 1, 0, 0)},
    {F::VOP1, 0x00e, valu("v_cvt_off_f32_i4", 1, 0, 0)},
    {F::VOP1, 0x00f, valu("v_cvt_f32_f64", 2, 0, 0)},
    {F::VOP1, 0x010, valu("v_cvt_f64_f32", 1, 0, 0)},
    {F::VOP1, 0x011, valu("v_cvt_f32_ubyte0", 1, 0, 0)},
    {F::VOP1, 0x012, valu("v_cvt_f32_ubyte1", 1, 0, 0)},
    {F::VOP1, 0x013, valu("v_cvt_f32_ubyte2", 1, 0, 0)},
    {F::VOP1, 0x014, valu("v_cvt_f32_ubyte3", 1, 0, 0)},
    {F::VOP1, 0x015, valu("v_cvt_u32_f64", 2, 0, 0)},
    {F::VOP1, 0x016, valu("v_cvt_f64_u32", 1, 0, 0)},
    {F::VOP1, 0x017, valu("v_trunc_f64", 2, 0, 0)},
    {F::VOP1, 0x018, valu("v_ceil_f64", 2, 0, 0)},
    {F::VOP1, 0x019, valu("v_rndne_f64", 2, 0, 0)},
    {F::VOP1, 0x01a, valu("v_floor_f64", 2, 0, 0)},
    {F::VOP1, 0x020, valu("v_fract_f32", 1, 0, 0)},
    {F::VOP1, 0x021, valu("v_trunc_f32", 1, 0, 0)},
    {F::VOP1, 0x022, valu("v_ceil_f32", 1, 0, 0)},
    {F::VOP1, 0x023, valu("v_rndne_f32", 1, 0, 0)},
    {F::VOP1, 0x024, valu("v_floor_f32", 1, 0, 0)},
    {F::VOP1, 0x025, valu("v_exp_f32", 1, 0, 0)},
    {F::VOP1, 0x027, valu("v_log_f32", 1, 0, 0)},
    {F::VOP1, 0x02a, valu("v_rcp_f32", 1, 0, 0)},
    {F::VOP1, 0x02b, valu("v_rcp_iflag_f32", 1, 0, 0)},
    {F::VOP1, 0x02e, valu("v_rsq_f32", 1, 0, 0)},
    {F::VOP1, 0x02f, valu("v_rcp_f64", 2, 0, 0)},
    {F::VOP1, 0x031, valu("v_rsq_f64", 2, 0, 0)},
    {F::VOP1, 0x033, valu("v_sqrt_f32", 1, 0, 0)},
    {F::VOP1, 0x034, valu("v_sqrt_f64", 2, 0, 0)},
    {F::VOP1, 0x035, valu("v_sin_f32", 1, 0, 0)},
    {F::VOP1, 0x036, valu("v_cos_f32", 1, 0, 0)},
    {F::VOP1, 0x037, valu("v_not_b32", 1, 0, 0)},
    {F::VOP1, 0x038, valu("v_bfrev_b32", 1, 0, 0)},
    {F::VOP1, 0x039, valu("v_ffbh_u32", 1, 0, 0)},
    {F::VOP1, 0x03a, valu("v_ffbl_b32", 1, 0, 0)},
    {F::VOP1, 0x03b, valu("v_ffbh_i32", 1, 0, 0)},
    {F::VOP1, 0x03c, valu("v_frexp_exp_i32_f64", 2, 0, 0)},
    {F::VOP1, 0x03d, valu("v_frexp_mant_f64", 2, 0, 0)},
    {F::VOP1, 0x03e, valu("v_fract_f64", 2, 0, 0)},
    {F::VOP1, 0x03f, valu("v_frexp_exp_i32_f32", 1, 0, 0)},
    {F::VOP1, 0x040, valu("v_frexp_mant_f32", 1, 0, 0)},
    {F::VOP1, 0x042, valu("v_movreld_b32", 1, 0, 0)},
    {F::VOP1, 0x043, valu("v_movrels_b32", 1, 0, 0)},
    {F::VOP1, 0x044, valu("v_movrelsd_b32", 1, 0, 0)},
    {F::VOP1, 0x048, valu("v_movrelsd_2_b32", 1, 0, 0)},
    {F::VOP1, 0x050, valu("v_cvt_f16_u16", 1, 0, 0)},
    {F::VOP1, 0x051, valu("v_cvt_f16_i16", 1, 0, 0)},
    {F::VOP1, 0x052, valu("v_cvt_u16_f16", 1, 0, 0)},
    {F::VOP1, 0x053, valu("v_cvt_i16_f16", 1, 0, 0)},
    {F::VOP1, 0x054, valu("v_rcp_f16", 1, 0, 0)},
    {F::VOP1, 0x055, valu("v_sqrt_f16", 1, 0, 0)},
    {F::VOP1, 0x056, valu("v_rsq_f16", 1, 0, 0)},
    {F::VOP1, 0x057, valu("v_log_f16", 1, 0, 0)},
    {F::VOP1, 0x058, valu("v_exp_f16", 1, 0, 0)},
    {F::VOP1, 0x059, valu("v_frexp_mant_f16", 1, 0, 0)},
    {F::VOP1, 0x05a, valu("v_frexp_exp_i16_f16", 1, 0, 0)},
    {F::VOP1, 0x05b, valu("v_floor_f16", 1, 0, 0)},
    {F::VOP1, 0x05c, valu("v_ceil_f16", 1, 0, 0)},
    {F::VOP1, 0x05d, valu("v_trunc_f16", 1, 0, 0)},
    {F::VOP1, 0x05e, valu("v_rndne_f16", 1, 0, 0)},
    {F::VOP1, 0x05f, valu("v_fract_f16", 1, 0, 0)},
    {F::VOP1, 0x060, valu("v_sin_f16", 1, 0, 0)},
    {F::VOP1, 0x061, valu("v_cos_f16", 1, 0, 0)},
    {F::VOP1, 0x062, valu("v_sat_pk_u8_i16", 1, 0, 0)},
    {F::VOP1, 0x063, valu("v_cvt_norm_i16_f16", 1, 0, 0)},
    {F::VOP1, 0x064, valu("v_cvt_norm_u16_f16", 1, 0, 0)},
    {F::VOP1, 0x065, valu("v_swap_b32", 1, 0, 0)},
    {F::VOP1, 0x068, valu("v_swaprel_b32", 1, 0, 0)},
};
constexpr Row kVopc[] = {
    {F::VOPC, 0x000, valu("v_cmp_f_f32", 1, 1, 0)},
    {F::VOPC, 0x001, valu("v_cmp_lt_f32", 1, 1, 0)},
    {F::VOPC, 0x002, valu("v_cmp_eq_f32", 1, 1, 0)},
    {F::VOPC, 0x003, valu("v_cmp_le_f32", 1, 1, 0)},
    {F::VOPC, 0x004, valu("v_cmp_gt_f32", 1, 1, 0)},
    {F::VOPC, 0x005, valu("v_cmp_lg_f32", 1, 1, 0)},
    {F::VOPC, 0x006, valu("v_cmp_ge_f32", 1, 1, 0)},
    {F::VOPC, 0x007, valu("v_cmp_o_f32", 1, 1, 0)},
    {F::VOPC, 0x008, valu("v_cmp_u_f32", 1, 1, 0)},
    {F::VOPC, 0x009, valu("v_cmp_nge_f32", 1, 1, 0)},
    {F::VOPC, 0x00a, valu("v_cmp_nlg_f32", 1, 1, 0)},
    {F::VOPC, 0x00b, valu("v_cmp_ngt_f32", 1, 1, 0)},
    {F::VOPC, 0x00c, valu("v_cmp_nle_f32", 1, 1, 0)},
    {F::VOPC, 0x00d, valu("v_cmp_neq_f32", 1, 1, 0)},
    {F::VOPC, 0x00e, valu("v_cmp_nlt_f32", 1, 1, 0)},
    {F::VOPC, 0x00f, valu("v_cmp_tru_f32", 1, 1, 0)},
    {F::VOPC, 0x010, valu("v_cmpx_f_f32", 1, 1, 0)},
    {F::VOPC, 0x011, valu("v_cmpx_lt_f32", 1, 1, 0)},
    {F::VOPC, 0x012, valu("v_cmpx_eq_f32", 1, 1, 0)},
    {F::VOPC, 0x013, valu("v_cmpx_le_f32", 1, 1, 0)},
    {F::VOPC, 0x014, valu("v_cmpx_gt_f32", 1, 1, 0)},
    {F::VOPC, 0x015, valu("v_cmpx_lg_f32", 1, 1, 0)},
    {F::VOPC, 0x016, valu("v_cmpx_ge_f32", 1, 1, 0)},
    {F::VOPC, 0x017, valu("v_cmpx_o_f32", 1, 1, 0)},
    {F::VOPC, 0x018, valu("v_cmpx_u_f32", 1, 1, 0)},
    {F::VOPC, 0x019, valu("v_cmpx_nge_f32", 1, 1, 0)},
    {F::VOPC, 0x01a, valu("v_cmpx_nlg_f32", 1, 1, 0)},
    {F::VOPC, 0x01b, valu("v_cmpx_ngt_f32", 1, 1, 0)},
    {F::VOPC, 0x01c, valu("v_cmpx_nle_f32", 1, 1, 0)},
    {F::VOPC, 0x01d, valu("v_cmpx_neq_f32", 1, 1, 0)},
    {F::VOPC, 0x01e, valu("v_cmpx_nlt_f32", 1, 1, 0)},
    {F::VOPC, 0x01f, valu("v_cmpx_tru_f32", 1, 1, 0)},
    {F::VOPC, 0x020, valu("v_cmp_f_f64", 2, 2, 0)},
    {F::VOPC, 0x021, valu("v_cmp_lt_f64", 2, 2, 0)},
    {F::VOPC, 0x022, valu("v_cmp_eq_f64", 2, 2, 0)},
    {F::VOPC, 0x023, valu("v_cmp_le_f64", 2, 2, 0)},
    {F::VOPC, 0x024, valu("v_cmp_gt_f64", 2, 2, 0)},
    {F::VOPC, 0x025, valu("v_cmp_lg_f64", 2, 2, 0)},
    {F::VOPC, 0x026, valu("v_cmp_ge_f64", 2, 2, 0)},
    {F::VOPC, 0x027, valu("v_cmp_o_f64", 2, 2, 0)},
    {F::VOPC, 0x028, valu("v_cmp_u_f64", 2, 2, 0)},
    {F::VOPC, 0x029, valu("v_cmp_nge_f64", 2, 2, 0)},
    {F::VOPC, 0x02a, valu("v_cmp_nlg_f64", 2, 2, 0)},
    {F::VOPC, 0x02b, valu("v_cmp_ngt_f64", 2, 2, 0)},
    {F::VOPC, 0x02c, valu("v_cmp_nle_f64", 2, 2, 0)},
    {F::VOPC, 0x02d, valu("v_cmp_neq_f64", 2, 2, 0)},
    {F::VOPC, 0x02e, valu("v_cmp_nlt_f64", 2, 2, 0)},
    {F::VOPC, 0x02f, valu("v_cmp_tru_f64", 2, 2, 0)},
    {F::VOPC, 0x030, valu("v_cmpx_f_f64", 2, 2, 0)},
    {F::VOPC, 0x031, valu("v_cmpx_lt_f64", 2, 2, 0)},
    {F::VOPC, 0x032, valu("v_cmpx_eq_f64", 2, 2, 0)},
    {F::VOPC, 0x033, valu("v_cmpx_le_f64", 2, 2, 0)},
    {F::VOPC, 0x034, valu("v_cmpx_gt_f64", 2, 2, 0)},
    {F::VOPC, 0x035, valu("v_cmpx_lg_f64", 2, 2, 0)},
    {F::VOPC, 0x036, valu("v_cmpx_ge_f64", 2, 2, 0)},
    {F::VOPC, 0x037, valu("v_cmpx_o_f64", 2, 2, 0)},
    {F::VOPC, 0x038, valu("v_cmpx_u_f64", 2, 2, 0)},
    {F::VOPC, 0x039, valu("v_cmpx_nge_f64", 2, 2, 0)},
    {F::VOPC, 0x03a, valu("v_cmpx_nlg_f64", 2, 2, 0)},
    {F::VOPC, 0x03b, valu("v_cmpx_ngt_f64", 2, 2, 0)},
    {F::VOPC, 0x03c, valu("v_cmpx_nle_f64", 2, 2, 0)},
    {F::VOPC, 0x03d, valu("v_cmpx_neq_f64", 2, 2, 0)},
    {F::VOPC, 0x03e, valu("v_cmpx_nlt_f64", 2, 2, 0)},
    {F::VOPC, 0x03f, valu("v_cmpx_tru_f64", 2, 2, 0)},
    {F::VOPC, 0x080, valu("v_cmp_f_i32", 1, 1, 0)},
    {F::VOPC, 0x081, valu("v_cmp_lt_i32", 1, 1, 0)},
    {F::VOPC, 0x082, valu("v_cmp_eq_i32", 1, 1, 0)},
    {F::VOPC, 0x083, valu("v_cmp_le_i32", 1, 1, 0)},
    {F::VOPC, 0x084, valu("v_cmp_gt_i32", 1, 1, 0)},
    {F::VOPC, 0x085, valu("v_cmp_ne_i32", 1, 1, 0)},
    {F::VOPC, 0x086, valu("v_cmp_ge_i32", 1, 1, 0)},
    {F::VOPC, 0x087, valu("v_cmp_t_i32", 1, 1, 0)},
    {F::VOPC, 0x088, valu("v_cmp_class_f32", 1, 1, 0)},
    {F::VOPC, 0x089, valu("v_cmp_lt_i16", 1, 1, 0)},
    {F::VOPC, 0x08a, valu("v_cmp_eq_i16", 1, 1, 0)},
    {F::VOPC, 0x08b, valu("v_cmp_le_i16", 1, 1, 0)},
    {F::VOPC, 0x08c, valu("v_cmp_gt_i16", 1, 1, 0)},
    {F::VOPC, 0x08d, valu("v_cmp_ne_i16", 1, 1, 0)},
    {F::VOPC, 0x08e, valu("v_cmp_ge_i16", 1, 1, 0)},
    {F::VOPC, 0x08f, valu("v_cmp_class_f16", 1, 1, 0)},
    {F::VOPC, 0x090, valu("v_cmpx_f_i32", 1, 1, 0)},
    {F::VOPC, 0x091, valu("v_cmpx_lt_i32", 1, 1, 0)},
    {F::VOPC, 0x092, valu("v_cmpx_eq_i32", 1, 1, 0)},
    {F::VOPC, 0x093, valu("v_cmpx_le_i32", 1, 1, 0)},
    {F::VOPC, 0x094, valu("v_cmpx_gt_i32", 1, 1, 0)},
    {F::VOPC, 0x095, valu("v_cmpx_ne_i32", 1, 1, 0)},
    {F::VOPC, 0x096, valu("v_cmpx_ge_i32", 1, 1, 0)},
    {F::VOPC, 0x097, valu("v_cmpx_t_i32", 1, 1, 0)},
    {F::VOPC, 0x098, valu("v_cmpx_class_f32", 1, 1, 0)},
    {F::VOPC, 0x099, valu("v_cmpx_lt_i16", 1, 1, 0)},
    {F::VOPC, 0x09a, valu("v_cmpx_eq_i16", 1, 1, 0)},
    {F::VOPC, 0x09b, valu("v_cmpx_le_i16", 1, 1, 0)},
    {F::VOPC, 0x09c, valu("v_cmpx_gt_i16", 1, 1, 0)},
    {F::VOPC, 0x09d, valu("v_cmpx_ne_i16", 1, 1, 0)},
    {F::VOPC, 0x09e, valu("v_cmpx_ge_i16", 1, 1, 0)},
    {F::VOPC, 0x09f, valu("v_cmpx_class_f16", 1, 1, 0)},
    {F::VOPC, 0x0a0, valu("v_cmp_f_i64", 2, 2, 0)},
    {F::VOPC, 0x0a1, valu("v_cmp_lt_i64", 2, 2, 0)},
    {F::VOPC, 0x0a2, valu("v_cmp_eq_i64", 2, 2, 0)},
    {F::VOPC, 0x0a3, valu("v_cmp_le_i64", 2, 2, 0)},
    {F::VOPC, 0x0a4, valu("v_cmp_gt_i64", 2, 2, 0)},
    {F::VOPC, 0x0a5, valu("v_cmp_ne_i64", 2, 2, 0)},
    {F::VOPC, 0x0a6, valu("v_cmp_ge_i64", 2, 2, 0)},
    {F::VOPC, 0x0a7, valu("v_cmp_t_i64", 2, 2, 0)},
    {F::VOPC, 0x0a8, valu("v_cmp_class_f64", 2, 1, 0)},
    {F::VOPC, 0x0a9, valu("v_cmp_lt_u16", 1, 1, 0)},
    {F::VOPC, 0x0aa, valu("v_cmp_eq_u16", 1, 1, 0)},
    {F::VOPC, 0x0ab, valu("v_cmp_le_u16", 1, 1, 0)},
    {F::VOPC, 0x0ac, valu("v_cmp_gt_u16", 1, 1, 0)},
    {F::VOPC, 0x0ad, valu("v_cmp_ne_u16", 1, 1, 0)},
    {F::VOPC, 0x0ae, valu("v_cmp_ge_u16", 1, 1, 0)},
    {F::VOPC, 0x0b0, valu("v_cmpx_f_i64", 2, 2, 0)},
    {F::VOPC, 0x0b1, valu("v_cmpx_lt_i64", 2, 2, 0)},
    {F::VOPC, 0x0b2, valu("v_cmpx_eq_i64", 2, 2, 0)},
    {F::VOPC, 0x0b3, valu("v_cmpx_le_i64", 2, 2, 0)},
    {F::VOPC, 0x0b4, valu("v_cmpx_gt_i64", 2, 2, 0)},
    {F::VOPC, 0x0b5, valu("v_cmpx_ne_i64", 2, 2, 0)},
    {F::VOPC, 0x0b6, valu("v_cmpx_ge_i64", 2, 2, 0)},
    {F::VOPC, 0x0b7, valu("v_cmpx_t_i64", 2, 2, 0)},
    {F::VOPC, 0x0b8, valu("v_cmpx_class_f64", 2, 1, 0)},
    {F::VOPC, 0x0b9, valu("v_cmpx_lt_u16", 1, 1, 0)},
    {F::VOPC, 0x0ba, valu("v_cmpx_eq_u16", 1, 1, 0)},
    {F::VOPC, 0x0bb, valu("v_cmpx_le_u16", 1, 1, 0)},
    {F::VOPC, 0x0bc, valu("v_cmpx_gt_u16", 1, 1, 0)},
    {F::VOPC, 0x0bd, valu("v_cmpx_ne_u16", 1, 1, 0)},
    {F::VOPC, 0x0be, valu("v_cmpx_ge_u16", 1, 1, 0)},
    {F::VOPC, 0x0c0, valu("v_cmp_f_u32", 1, 1, 0)},
    {F::VOPC, 0x0c1, valu("v_cmp_lt_u32", 1, 1, 0)},
    {F::VOPC, 0x0c2, valu("v_cmp_eq_u32", 1, 1, 0)},
    {F::VOPC, 0x0c3, valu("v_cmp_le_u32", 1, 1, 0)},
    {F::VOPC, 0x0c4, valu("v_cmp_gt_u32", 1, 1, 0)},
    {F::VOPC, 0x0c5, valu("v_cmp_ne_u32", 1, 1, 0)},
    {F::VOPC, 0x0c6, valu("v_cmp_ge_u32", 1, 1, 0)},
    {F::VOPC, 0x0c7, valu("v_cmp_t_u32", 1, 1, 0)},
    {F::VOPC, 0x0c8, valu("v_cmp_f_f16", 1, 1, 0)},
    {F::VOPC, 0x0c9, valu("v_cmp_lt_f16", 1, 1, 0)},
    {F::VOPC, 0x0ca, valu("v_cmp_eq_f16", 1, 1, 0)},
    {F::VOPC, 0x0cb, valu("v_cmp_le_f16", 1, 1, 0)},
    {F::VOPC, 0x0cc, valu("v_cmp_gt_f16", 1, 1, 0)},
    {F::VOPC, 0x0cd, valu("v_cmp_lg_f16", 1, 1, 0)},
    {F::VOPC, 0x0ce, valu("v_cmp_ge_f16", 1, 1, 0)},
    {F::VOPC, 0x0cf, valu("v_cmp_o_f16", 1, 1, 0)},
    {F::VOPC, 0x0d0, valu("v_cmpx_f_u32", 1, 1, 0)},
    {F::VOPC, 0x0d1, valu("v_cmpx_lt_u32", 1, 1, 0)},
    {F::VOPC, 0x0d2, valu("v_cmpx_eq_u32", 1, 1, 0)},
    {F::VOPC, 0x0d3, valu("v_cmpx_le_u32", 1, 1, 0)},
    {F::VOPC, 0x0d4, valu("v_cmpx_gt_u32", 1, 1, 0)},
    {F::VOPC, 0x0d5, valu("v_cmpx_ne_u32", 1, 1, 0)},
    {F::VOPC, 0x0d6, valu("v_cmpx_ge_u32", 1, 1, 0)},
    {F::VOPC, 0x0d7, valu("v_cmpx_t_u32", 1, 1, 0)},
    {F::VOPC, 0x0d8, valu("v_cmpx_f_f16", 1, 1, 0)},
    {F::VOPC, 0x0d9, valu("v_cmpx_lt_f16", 1, 1, 0)},
    {F::VOPC, 0x0da, valu("v_cmpx_eq_f16", 1, 1, 0)},
    {F::VOPC, 0x0db, valu("v_cmpx_le_f16", 1, 1, 0)},
    {F::VOPC, 0x0dc, valu("v_cmpx_gt_f16", 1, 1, 0)},
    {F::VOPC, 0x0dd, valu("v_cmpx_lg_f16", 1, 1, 0)},
    {F::VOPC, 0x0de, valu("v_cmpx_ge_f16", 1, 1, 0)},
    {F::VOPC, 0x0df, valu("v_cmpx_o_f16", 1, 1, 0)},
    {F::VOPC, 0x0e0, valu("v_cmp_f_u64", 2, 2, 0)},
    {F::VOPC, 0x0e1, valu("v_cmp_lt_u64", 2, 2, 0)},
    {F::VOPC, 0x0e2, valu("v_cmp_eq_u64", 2, 2, 0)},
    {F::VOPC, 0x0e3, valu("v_cmp_le_u64", 2, 2, 0)},
    {F::VOPC, 0x0e4, valu("v_cmp_gt_u64", 2, 2, 0)},
    {F::VOPC, 0x0e5, valu("v_cmp_ne_u64", 2, 2, 0)},
    {F::VOPC, 0x0e6, valu("v_cmp_ge_u64", 2, 2, 0)},
    {F::VOPC, 0x0e7, valu("v_cmp_t_u64", 2, 2, 0)},
    {F::VOPC, 0x0e8, valu("v_cmp_u_f16", 1, 1, 0)},
    {F::VOPC, 0x0e9, valu("v_cmp_nge_f16", 1, 1, 0)},
    {F::VOPC, 0x0ea, valu("v_cmp_nlg_f16", 1, 1, 0)},
    {F::VOPC, 0x0eb, valu("v_cmp_ngt_f16", 1, 1, 0)},
    {F::VOPC, 0x0ec, valu("v_cmp_nle_f16", 1, 1, 0)},
    {F::VOPC, 0x0ed, valu("v_cmp_neq_f16", 1, 1, 0)},
    {F::VOPC, 0x0ee, valu("v_cmp_nlt_f16", 1, 1, 0)},
    {F::VOPC, 0x0ef, valu("v_cmp_tru_f16", 1, 1, 0)},
    {F::VOPC, 0x0f0, valu("v_cmpx_f_u64", 2, 2, 0)},
    {F::VOPC, 0x0f1, valu("v_cmpx_lt_u64", 2, 2, 0)},
    {F::VOPC, 0x0f2, valu("v_cmpx_eq_u64", 2, 2, 0)},
    {F::VOPC, 0x0f3, valu("v_cmpx_le_u64", 2, 2, 0)},
    {F::VOPC, 0x0f4, valu("v_cmpx_gt_u64", 2, 2, 0)},
    {F::VOPC, 0x0f5, valu("v_cmpx_ne_u64", 2, 2, 0)},
    {F::VOPC, 0x0f6, valu("v_cmpx_ge_u64", 2, 2, 0)},
    {F::VOPC, 0x0f7, valu("v_cmpx_t_u64", 2, 2, 0)},
    {F::VOPC, 0x0f8, valu("v_cmpx_u_f16", 1, 1, 0)},
    {F::VOPC, 0x0f9, valu("v_cmpx_nge_f16", 1, 1, 0)},
    {F::VOPC, 0x0fa, valu("v_cmpx_nlg_f16", 1, 1, 0)},
    {F::VOPC, 0x0fb, valu("v_cmpx_ngt_f16", 1, 1, 0)},
    {F::VOPC, 0x0fc, valu("v_cmpx_nle_f16", 1, 1, 0)},
    {F::VOPC, 0x0fd, valu("v_cmpx_neq_f16", 1, 1, 0)},
    {F::VOPC, 0x0fe, valu("v_cmpx_nlt_f16", 1, 1, 0)},
    {F::VOPC, 0x0ff, valu("v_cmpx_tru_f16", 1, 1, 0)},
};
constexpr Row kVop3[] = {
    {F::VOP3, 0x101, valu("v_cndmask_b32", 1, 1, 2)},
    {F::VOP3, 0x103, valu("v_add_f32", 1, 1, 0)},
    {F::VOP3, 0x104, valu("v_sub_f32", 1, 1, 0)},
    {F::VOP3, 0x105, valu("v_subrev_f32", 1, 1, 0)},
    {F::VOP3, 0x106, valu("v_fmac_legacy_f32", 1, 1, 0)},
    {F::VOP3, 0x107, valu("v_mul_legacy_f32", 1, 1, 0)},
    {F::VOP3, 0x108, valu("v_mul_f32", 1, 1, 0)},
    {F::VOP3, 0x109, valu("v_mul_i32_i24", 1, 1, 0)},
    {F::VOP3, 0x10a, valu("v_mul_hi_i32_i24", 1, 1, 0)},
    {F::VOP3, 0x10b, valu("v_mul_u32_u24", 1, 1, 0)},
    {F::VOP3, 0x10c, valu("v_mul_hi_u32_u24", 1, 1, 0)},
    {F::VOP3, 0x10f, valu("v_min_f32", 1, 1, 0)},
    {F::VOP3, 0x110, valu("v_max_f32", 1, 1, 0)},
    {F::VOP3, 0x111, valu("v_min_i32", 1, 1, 0)},
    {F::VOP3, 0x112, valu("v_max_i32", 1, 1, 0)},
    {F::VOP3, 0x113, valu("v_min_u32", 1, 1, 0)},
    {F::VOP3, 0x114, valu("v_max_u32", 1, 1, 0)},
    {F::VOP3, 0x116, valu("v_lshrrev_b32", 1, 1, 0)},
    {F::VOP3, 0x118, valu("v_ashrrev_i32", 1, 1, 0)},
    {F::VOP3, 0x11a, valu("v_lshlrev_b32", 1, 1, 0)},
    {F::VOP3, 0x11b, valu("v_and_b32", 1, 1, 0)},
    {F::VOP3, 0x11c, valu("v_or_b32", 1, 1, 0)},
    {F::VOP3, 0x11d, valu("v_xor_b32", 1, 1, 0)},
    {F::VOP3, 0x11e, valu("v_xnor_b32", 1, 1, 0)},
    {F::VOP3, 0x11f, valu("v_mac_f32", 1, 1, 0)},
    {F::VOP3, 0x125, valu("v_add_nc_u32", 1, 1, 0)},
    {F::VOP3, 0x126, valu("v_sub_nc_u32", 1, 1, 0)},
    {F::VOP3, 0x127, valu("v_subrev_nc_u32", 1, 1, 0)},
    {F::VOP3, 0x128, valu("v_add_co_ci_u32", 1, 1, 2)},
    {F::VOP3, 0x129, valu("v_sub_co_ci_u32", 1, 1, 2)},
    {F::VOP3, 0x12a, valu("v_subrev_co_ci_u32", 1, 1, 2)},
    {F::VOP3, 0x12b, valu("v_fmac_f32", 1, 1, 0)},
    {F::VOP3, 0x12f, valu("v_cvt_pkrtz_f16_f32", 1, 1, 0)},
    {F::VOP3, 0x132, valu("v_add_f16", 1, 1, 0)},
    {F::VOP3, 0x133, valu("v_sub_f16", 1, 1, 0)},
    {F::VOP3, 0x134, valu("v_subrev_f16", 1, 1, 0)},
    {F::VOP3, 0x135, valu("v_mul_f16", 1, 1, 0)},
    {F::VOP3, 0x136, valu("v_fmac_f16", 1, 1, 0)},
    {F::VOP3, 0x139, valu("v_max_f16", 1, 1, 0)},
    {F::VOP3, 0x13a, valu("v_min_f16", 1, 1, 0)},
    {F::VOP3, 0x13b, valu("v_ldexp_f16", 1, 1, 0)},
    {F::VOP3, 0x140, valu("v_fma_legacy_f32", 1, 1, 1)},
    {F::VOP3, 0x141, valu("v_mad_f32", 1, 1, 1)},
    {F::VOP3, 0x142, valu("v_mad_i32_i24", 1, 1, 1)},
    {F::VOP3, 0x143, valu("v_mad_u32_u24", 1, 1, 1)},
    {F::VOP3, 0x144, valu("v_cubeid_f32", 1, 1, 1)},
    {F::VOP3, 0x145, valu("v_cubesc_f32", 1, 1, 1)},
    {F::VOP3, 0x146, valu("v_cubetc_f32", 1, 1, 1)},
    {F::VOP3, 0x147, valu("v_cubema_f32", 1, 1, 1)},
    {F::VOP3, 0x148, valu("v_bfe_u32", 1, 1, 1)},
    {F::VOP3, 0x149, valu("v_bfe_i32", 1, 1, 1)},
    {F::VOP3, 0x14a, valu("v_bfi_b32", 1, 1, 1)},
    {F::VOP3, 0x14b, valu("v_fma_f32", 1, 1, 1)},
    {F::VOP3, 0x14c, valu("v_fma_f64", 2, 2, 2)},
    {F::VOP3, 0x14d, valu("v_lerp_u8", 1, 1, 1)},
    {F::VOP3, 0x14e, valu("v_alignbit_b32", 1, 1, 1)},
    {F::VOP3, 0x14f, valu("v_alignbyte_b32", 1, 1, 1)},
    {F::VOP3, 0x150, valu("v_mullit_f32", 1, 1, 1)},
    {F::VOP3, 0x151, valu("v_min3_f32", 1, 1, 1)},
    {F::VOP3, 0x152, valu("v_min3_i32", 1, 1, 1)},
    {F::VOP3, 0x153, valu("v_min3_u32", 1, 1, 1)},
    {F::VOP3, 0x154, valu("v_max3_f32", 1, 1, 1)},
    {F::VOP3, 0x155, valu("v_max3_i32", 1, 1, 1)},
    {F::VOP3, 0x156, valu("v_max3_u32", 1, 1, 1)},
    {F::VOP3, 0x157, valu("v_med3_f32", 1, 1, 1)},
    {F::VOP3, 0x158, valu("v_med3_i32", 1, 1, 1)},
    {F::VOP3, 0x159, valu("v_med3_u32", 1, 1, 1)},
    {F::VOP3, 0x15a, valu("v_sad_u8", 1, 1, 1)},
    {F::VOP3, 0x15b, valu("v_sad_hi_u8", 1, 1, 1)},
    {F::VOP3, 0x15c, valu("v_sad_u16", 1, 1, 1)},
    {F::VOP3, 0x15d, valu("v_sad_u32", 1, 1, 1)},
    {F::VOP3, 0x15e, valu("v_cvt_pk_u8_f32", 1, 1, 1)},
    {F::VOP3, 0x15f, valu("v_div_fixup_f32", 1, 1, 1)},
    {F::VOP3, 0x160, valu("v_div_fixup_f64", 2, 2, 2)},
    {F::VOP3, 0x164, valu("v_add_f64", 2, 2, 0)},
    {F::VOP3, 0x165, valu("v_mul_f64", 2, 2, 0)},
    {F::VOP3, 0x166, valu("v_min_f64", 2, 2, 0)},
    {F::VOP3, 0x167, valu("v_max_f64", 2, 2, 0)},
    {F::VOP3, 0x168, valu("v_ldexp_f64", 2, 1, 0)},
    {F::VOP3, 0x169, valu("v_mul_lo_u32", 1, 1, 0)},
    {F::VOP3, 0x16a, valu("v_mul_hi_u32", 1, 1, 0)},
    {F::VOP3, 0x16b, valu("v_mul_lo_i32", 1, 1, 0)},
    {F::VOP3, 0x16c, valu("v_mul_hi_i32", 1, 1, 0)},
    {F::VOP3, 0x16d, valu("v_div_scale_f32", 1, 1, 1)},
    {F::VOP3, 0x16e, valu("v_div_scale_f64", 2, 2, 2)},
    {F::VOP3, 0x16f, valu("v_div_fmas_f32", 1, 1, 1)},
    {F::VOP3, 0x170, valu("v_div_fmas_f64", 2, 2, 2)},
    {F::VOP3, 0x171, valu("v_msad_u8", 1, 1, 1)},
    {F::VOP3, 0x172, valu("v_qsad_pk_u16_u8", 2, 1, 2)},
    {F::VOP3, 0x173, valu("v_mqsad_pk_u16_u8", 2, 1, 2)},
    {F::VOP3, 0x174, valu("v_trig_preop_f64", 2, 1, 0)},
    {F::VOP3, 0x175, valu("v_mqsad_u32_u8", 2, 1, 0)},
    {F::VOP3, 0x176, valu("v_mad_u64_u32", 1, 1, 2)},
    {F::VOP3, 0x177, valu("v_mad_i64_i32", 1, 1, 2)},
    {F::VOP3, 0x178, valu("v_xor3_b32", 1, 1, 1)},
    {F::VOP3, 0x181, valu("v_mov_b32", 1, 0, 0)},
    {F::VOP3, 0x183, valu("v_cvt_i32_f64", 2, 0, 0)},
    {F::VOP3, 0x184, valu("v_cvt_f64_i32", 1, 0, 0)},
    {F::VOP3, 0x185, valu("v_cvt_f32_i32", 1, 0, 0)},
    {F::VOP3, 0x186, valu("v_cvt_f32_u32", 1, 0, 0)},
    {F::VOP3, 0x187, valu("v_cvt_u32_f32", 1, 0, 0)},
    {F::VOP3, 0x188, valu("v_cvt_i32_f32", 1, 0, 0)},
    {F::VOP3, 0x18a, valu("v_cvt_f16_f32", 1, 0, 0)},
    {F::VOP3, 0x18b, valu("v_cvt_f32_f16", 1, 0, 0)},
    {F::VOP3, 0x18c, valu("v_cvt_rpi_i32_f32", 1, 0, 0)},
    {F::VOP3, 0x18d, valu("v_cvt_flr_i32_f32", 1, 0, 0)},
    {F::VOP3, 0x18e, valu("v_cvt_off_f32_i4", 1, 0, 0)},
    {F::VOP3, 0x18f, valu("v_cvt_f32_f64", 2, 0, 0)},
    {F::VOP3, 0x190, valu("v_cvt_f64_f32", 1, 0, 0)},
    {F::VOP3, 0x191, valu("v_cvt_f32_ubyte0", 1, 0, 0)},
    {F::VOP3, 0x192, valu("v_cvt_f32_ubyte1", 1, 0, 0)},
    {F::VOP3, 0x193, valu("v_cvt_f32_ubyte2", 1, 0, 0)},
    {F::VOP3, 0x194, valu("v_cvt_f32_ubyte3", 1, 0, 0)},
    {F::VOP3, 0x195, valu("v_cvt_u32_f64", 2, 0, 0)},
    {F::VOP3, 0x196, valu("v_cvt_f64_u32", 1, 0, 0)},
    {F::VOP3, 0x197, valu("v_trunc_f64", 2, 0, 0)},
    {F::VOP3, 0x198, valu("v_ceil_f64", 2, 0, 0)},
    {F::VOP3, 0x199, valu("v_rndne_f64", 2, 0, 0)},
    {F::VOP3, 0x19a, valu("v_floor_f64", 2, 0, 0)},
    {F::VOP3, 0x1a0, valu("v_fract_f32", 1, 0, 0)},
    {F::VOP3, 0x1a1, valu("v_trunc_f32", 1, 0, 0)},
    {F::VOP3, 0x1a2, valu("v_ceil_f32", 1, 0, 0)},
    {F::VOP3, 0x1a3, valu("v_rndne_f32", 1, 0, 0)},
    {F::VOP3, 0x1a4, valu("v_floor_f32", 1, 0, 0)},
    {F::VOP3, 0x1a5, valu("v_exp_f32", 1, 0, 0)},
    {F::VOP3, 0x1a7, valu("v_log_f32", 1, 0, 0)},
    {F::VOP3, 0x1aa, valu("v_rcp_f32", 1, 0, 0)},
    {F::VOP3, 0x1ab, valu("v_rcp_iflag_f32", 1, 0, 0)},
    {F::VOP3, 0x1ae, valu("v_rsq_f32", 1, 0, 0)},
    {F::VOP3, 0x1af, valu("v_rcp_f64", 2, 0, 0)},
    {F::VOP3, 0x1b1, valu("v_rsq_f64", 2, 0, 0)},
    {F::VOP3, 0x1b3, valu("v_sqrt_f32", 1, 0, 0)},
    {F::VOP3, 0x1b4, valu("v_sqrt_f64", 2, 0, 0)},
    {F::VOP3, 0x1b5, valu("v_sin_f32", 1, 0, 0)},
    {F::VOP3, 0x1b6, valu("v_cos_f32", 1, 0, 0)},
    {F::VOP3, 0x1b7, valu("v_not_b32", 1, 0, 0)},
    {F::VOP3, 0x1b8, valu("v_bfrev_b32", 1, 0, 0)},
    {F::VOP3, 0x1b9, valu("v_ffbh_u32", 1, 0, 0)},
    {F::VOP3, 0x1ba, valu("v_ffbl_b32", 1, 0, 0)},
    {F::VOP3, 0x1bb, valu("v_ffbh_i32", 1, 0, 0)},
    {F::VOP3, 0x1bc, valu("v_frexp_exp_i32_f64", 2, 0, 0)},
    {F::VOP3, 0x1bd, valu("v_frexp_mant_f64", 2, 0, 0)},
    {F::VOP3, 0x1be, valu("v_fract_f64", 2, 0, 0)},
    {F::VOP3, 0x1bf, valu("v_frexp_exp_i32_f32", 1, 0, 0)},
    {F::VOP3, 0x1c0, valu("v_frexp_mant_f32", 1, 0, 0)},
    {F::VOP3, 0x1c2, valu("v_movreld_b32", 1, 0, 0)},
    {F::VOP3, 0x1c3, valu("v_movrels_b32", 1, 0, 0)},
    {F::VOP3, 0x1c4, valu("v_movrelsd_b32", 1, 0, 0)},
    {F::VOP3, 0x1c8, valu("v_movrelsd_2_b32", 1, 0, 0)},
    {F::VOP3, 0x1d0, valu("v_cvt_f16_u16", 1, 0, 0)},
    {F::VOP3, 0x1d1, valu("v_cvt_f16_i16", 1, 0, 0)},
    {F::VOP3, 0x1d2, valu("v_cvt_u16_f16", 1, 0, 0)},
    {F::VOP3, 0x1d3, valu("v_cvt_i16_f16", 1, 0, 0)},
    {F::VOP3, 0x1d4, valu("v_rcp_f16", 1, 0, 0)},
    {F::VOP3, 0x1d5, valu("v_sqrt_f16", 1, 0, 0)},
    {F::VOP3, 0x1d6, valu("v_rsq_f16", 1, 0, 0)},
    {F::VOP3, 0x1d7, valu("v_log_f16", 1, 0, 0)},
    {F::VOP3, 0x1d8, valu("v_exp_f16", 1, 0, 0)},
    {F::VOP3, 0x1d9, valu("v_frexp_mant_f16", 1, 0, 0)},
    {F::VOP3, 0x1da, valu("v_frexp_exp_i16_f16", 1, 0, 0)},
    {F::VOP3, 0x1db, valu("v_floor_f16", 1, 0, 0)},
    {F::VOP3, 0x1dc, valu("v_ceil_f16", 1, 0, 0)},
    {F::VOP3, 0x1dd, valu("v_trunc_f16", 1, 0, 0)},
    {F::VOP3, 0x1de, valu("v_rndne_f16", 1, 0, 0)},
    {F::VOP3, 0x1df, valu("v_fract_f16", 1, 0, 0)},
    {F::VOP3, 0x1e0, valu("v_sin_f16", 1, 0, 0)},
    {F::VOP3, 0x1e1, valu("v_cos_f16", 1, 0, 0)},
    {F::VOP3, 0x1e2, valu("v_sat_pk_u8_i16", 1, 0, 0)},
    {F::VOP3, 0x1e3, valu("v_cvt_norm_i16_f16", 1, 0, 0)},
    {F::VOP3, 0x1e4, valu("v_cvt_norm_u16_f16", 1, 0, 0)},
    {F::VOP3, 0x200, valu("v_interp_p1_f32", 0, 1, 0)},
    {F::VOP3, 0x201, valu("v_interp_p2_f32", 0, 1, 0)},
    {F::VOP3, 0x202, valu("v_interp_mov_f32", 0, 0, 0)},
    {F::VOP3, 0x2ff, valu("v_lshlrev_b64", 1, 2, 0)},
    {F::VOP3, 0x300, valu("v_lshrrev_b64", 1, 2, 0)},
    {F::VOP3, 0x301, valu("v_ashrrev_i64", 1, 2, 0)},
    {F::VOP3, 0x303, valu("v_add_nc_u16", 1, 1, 0)},
    {F::VOP3, 0x304, valu("v_sub_nc_u16", 1, 1, 0)},
    {F::VOP3, 0x305, valu("v_mul_lo_u16", 1, 1, 0)},
    {F::VOP3, 0x307, valu("v_lshrrev_b16", 1, 1, 0)},
    {F::VOP3, 0x308, valu("v_ashrrev_i16", 1, 1, 0)},
    {F::VOP3, 0x309, valu("v_max_u16", 1, 1, 0)},
    {F::VOP3, 0x30a, valu("v_max_i16", 1, 1, 0)},
    {F::VOP3, 0x30b, valu("v_min_u16", 1, 1, 0)},
    {F::VOP3, 0x30c, valu("v_min_i16", 1, 1, 0)},
    {F::VOP3, 0x30d, valu("v_add_nc_i16", 1, 1, 0)},
    {F::VOP3, 0x30e, valu("v_sub_nc_i16", 1, 1, 0)},
    {F::VOP3, 0x30f, valu("v_add_co_u32", 1, 1, 0)},
    {F::VOP3, 0x310, valu("v_sub_co_u32", 1, 1, 0)},
    {F::VOP3, 0x311, valu("v_pack_b32_f16", 1, 1, 0)},
    {F::VOP3, 0x312, valu("v_cvt_pknorm_i16_f16", 1, 1, 0)},
    {F::VOP3, 0x313, valu("v_cvt_pknorm_u16_f16", 1, 1, 0)},
    {F::VOP3, 0x314, valu("v_lshlrev_b16", 1, 1, 0)},
    {F::VOP3, 0x319, valu("v_subrev_co_u32", 1, 1, 0)},
    {F::VOP3, 0x340, valu("v_mad_u16", 1, 1, 1)},
    {F::VOP3, 0x342, valu("v_interp_p1ll_f16", 0, 1, 0)},
    {F::VOP3, 0x343, valu("v_interp_p1lv_f16", 0, 1, 1)},
    {F::VOP3, 0x344, valu("v_perm_b32", 1, 1, 1)},
    {F::VOP3, 0x345, valu("v_xad_u32", 1, 1, 1)},
    {F::VOP3, 0x346, valu("v_lshl_add_u32", 1, 1, 1)},
    {F::VOP3, 0x347, valu("v_add_lshl_u32", 1, 1, 1)},
    {F::VOP3, 0x34b, valu("v_fma_f16", 1, 1, 1)},
    {F::VOP3, 0x351, valu("v_min3_f16", 1, 1, 1)},
    {F::VOP3, 0x352, valu("v_min3_i16", 1, 1, 1)},
    {F::VOP3, 0x353, valu("v_min3_u16", 1, 1, 1)},
    {F::VOP3, 0x354, valu("v_max3_f16", 1, 1, 1)},
    {F::VOP3, 0x355, valu("v_max3_i16", 1, 1, 1)},
    {F::VOP3, 0x356, valu("v_max3_u16", 1, 1, 1)},
    {F::VOP3, 0x357, valu("v_med3_f16", 1, 1, 1)},
    {F::VOP3, 0x358, valu("v_med3_i16", 1, 1, 1)},
    {F::VOP3, 0x359, valu("v_med3_u16", 1, 1, 1)},
    {F::VOP3, 0x35a, valu("v_interp_p2_f16", 0, 1, 1)},
    {F::VOP3, 0x35e, valu("v_mad_i16", 1, 1, 1)},
    {F::VOP3, 0x35f, valu("v_div_fixup_f16", 1, 1, 1)},
    {F::VOP3, 0x360, valu("v_readlane_b32", 1, 1, 0)},
    {F::VOP3, 0x361, valu("v_writelane_b32", 1, 1, 0)},
    {F::VOP3, 0x362, valu("v_ldexp_f32", 1, 1, 0)},
    {F::VOP3, 0x363, valu("v_bfm_b32", 1, 1, 0)},
    {F::VOP3, 0x364, valu("v_bcnt_u32_b32", 1, 1, 0)},
    {F::VOP3, 0x365, valu("v_mbcnt_lo_u32_b32", 1, 1, 0)},
    {F::VOP3, 0x366, valu("v_mbcnt_hi_u32_b32", 1, 1, 0)},
    {F::VOP3, 0x368, valu("v_cvt_pknorm_i16_f32", 1, 1, 0)},
    {F::VOP3, 0x369, valu("v_cvt_pknorm_u16_f32", 1, 1, 0)},
    {F::VOP3, 0x36a, valu("v_cvt_pk_u16_u32", 1, 1, 0)},
    {F::VOP3, 0x36b, valu("v_cvt_pk_i16_i32", 1, 1, 0)},
    {F::VOP3, 0x36d, valu("v_add3_u32", 1, 1, 1)},
    {F::VOP3, 0x36f, valu("v_lshl_or_b32", 1, 1, 1)},
    {F::VOP3, 0x371, valu("v_and_or_b32", 1, 1, 1)},
    {F::VOP3, 0x372, valu("v_or3_b32", 1, 1, 1)},
    {F::VOP3, 0x373, valu("v_mad_u32_u16", 1, 1, 1)},
    {F::VOP3, 0x375, valu("v_mad_i32_i16", 1, 1, 1)},
    {F::VOP3, 0x376, valu("v_sub_nc_i32", 1, 1, 0)},
    {F::VOP3, 0x377, valu("v_permlane16_b32", 1, 1, 1)},
    {F::VOP3, 0x378, valu("v_permlanex16_b32", 1, 1, 1)},
    {F::VOP3, 0x37f, valu("v_add_nc_i32", 1, 1, 0)},
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
    {F::VOP2, 0x000, 0x000},
    {F::VOP2, 0x00e, 0x00e},
    {F::VOP2, 0x015, 0x015},
    {F::VOP2, 0x017, 0x017},
    {F::VOP2, 0x019, 0x019},
    {F::VOP2, 0x022, 0x024},
    {F::VOP2, 0x02e, 0x02e},
    {F::VOP2, 0x030, 0x031},
    {F::VOP2, 0x03d, 0x03d},
    {F::VOP1, 0x000, 0x000},
    {F::VOP1, 0x009, 0x009},
    {F::VOP1, 0x01b, 0x01f},
    {F::VOP1, 0x026, 0x026},
    {F::VOP1, 0x028, 0x029},
    {F::VOP1, 0x02c, 0x02d},
    {F::VOP1, 0x030, 0x030},
    {F::VOP1, 0x032, 0x032},
    {F::VOP1, 0x041, 0x041},
    {F::VOP1, 0x045, 0x047},
    {F::VOP1, 0x049, 0x04f},
    {F::VOP1, 0x066, 0x067},
    {F::VOP1, 0x069, 0x0ff},
    {F::VOPC, 0x040, 0x07f},
    {F::VOPC, 0x0af, 0x0af},
    {F::VOPC, 0x0bf, 0x0bf},
    {F::VOP3, 0x100, 0x100},
    {F::VOP3, 0x102, 0x102},
    {F::VOP3, 0x10d, 0x10e},
    {F::VOP3, 0x115, 0x115},
    {F::VOP3, 0x117, 0x117},
    {F::VOP3, 0x119, 0x119},
    {F::VOP3, 0x120, 0x124},
    {F::VOP3, 0x12c, 0x12e},
    {F::VOP3, 0x130, 0x131},
    {F::VOP3, 0x137, 0x138},
    {F::VOP3, 0x13c, 0x13f},
    {F::VOP3, 0x161, 0x163},
    {F::VOP3, 0x179, 0x180},
    {F::VOP3, 0x182, 0x182},
    {F::VOP3, 0x189, 0x189},
    {F::VOP3, 0x19b, 0x19f},
    {F::VOP3, 0x1a6, 0x1a6},
    {F::VOP3, 0x1a8, 0x1a9},
    {F::VOP3, 0x1ac, 0x1ad},
    {F::VOP3, 0x1b0, 0x1b0},
    {F::VOP3, 0x1b2, 0x1b2},
    {F::VOP3, 0x1c1, 0x1c1},
    {F::VOP3, 0x1c5, 0x1c7},
    {F::VOP3, 0x1c9, 0x1cf},
    {F::VOP3, 0x1e5, 0x1ff},
    {F::VOP3, 0x203, 0x2fe},
    {F::VOP3, 0x302, 0x302},
    {F::VOP3, 0x306, 0x306},
    {F::VOP3, 0x315, 0x318},
    {F::VOP3, 0x31a, 0x33f},
    {F::VOP3, 0x341, 0x341},
    {F::VOP3, 0x348, 0x34a},
    {F::VOP3, 0x34c, 0x350},
    {F::VOP3, 0x35b, 0x35d},
    {F::VOP3, 0x367, 0x367},
    {F::VOP3, 0x36c, 0x36c},
    {F::VOP3, 0x36e, 0x36e},
    {F::VOP3, 0x370, 0x370},
    {F::VOP3, 0x374, 0x374},
    {F::VOP3, 0x379, 0x37e},
    {F::VOP3, 0x380, 0x3ff},
};

// The decoder's opcode spaces, minus encodings another format claims first (rdna2_decode_one).
// SOP2: bits[29:23]; 0x60-0x7F are the SOPK/SOP1/SOPC/SOPP prefixes.
constexpr Rdna2OpcodeRange kSop2Range{0, 0x60};
// SOP1: bits[15:8].
constexpr Rdna2OpcodeRange kSop1Range{0, 0x100};
// SOPK: bits[27:23]; 0x1D/0x1E/0x1F are the SOP1/SOPC/SOPP prefixes.
constexpr Rdna2OpcodeRange kSopkRange{0, 0x1d};
// SOPC and SOPP: bits[22:16].
constexpr Rdna2OpcodeRange kSopcRange{0, 0x80};
constexpr Rdna2OpcodeRange kSoppRange{0, 0x80};
// VOP2: bits[30:25]; 0x3E/0x3F are the VOPC/VOP1 prefixes.
constexpr Rdna2OpcodeRange kVop2Range{0, 0x3e};
// VOP1 and VOPC: 8-bit opcode fields (a VOPC e64 compare keeps its e32 number).
constexpr Rdna2OpcodeRange kVop1Range{0, 0x100};
constexpr Rdna2OpcodeRange kVopcRange{0, 0x100};
// VOP3: bits[25:16]; 0x000-0x0FF are VOPC e64 and decode as VOPC.
constexpr Rdna2OpcodeRange kVop3Range{0x100, 0x400};

template <size_t N>
struct FormatTable {
    std::array<Rdna2InstructionEffects, N> by_opcode{};
};

// Dense lookup for one format: its named rows plus its explicit reserved ranges. Every other slot
// stays Unknown.
template <size_t N, size_t R>
constexpr FormatTable<N> dense(Rdna2Format fmt, const Row (&rows)[R]) {
    FormatTable<N> table;
    for (const ReservedRange& range : kReserved)
        if (range.fmt == fmt)
            for (uint32_t opcode = range.first; opcode <= range.last; ++opcode)
                table.by_opcode[opcode] = reserved();
    for (const Row& row : rows) table.by_opcode[row.opcode] = row.effects;
    return table;
}

constexpr auto kSop2Table = dense<0x100>(F::SOP2, kSop2);
constexpr auto kSop1Table = dense<0x100>(F::SOP1, kSop1);
constexpr auto kSopkTable = dense<0x100>(F::SOPK, kSopk);
constexpr auto kSopcTable = dense<0x100>(F::SOPC, kSopc);
constexpr auto kSoppTable = dense<0x100>(F::SOPP, kSopp);
constexpr auto kVop2Table = dense<0x100>(F::VOP2, kVop2);
constexpr auto kVop1Table = dense<0x100>(F::VOP1, kVop1);
constexpr auto kVopcTable = dense<0x100>(F::VOPC, kVopc);
constexpr auto kVop3Table = dense<0x400>(F::VOP3, kVop3);

Rdna2OpcodeRange range_for(Rdna2Format fmt) {
    switch (fmt) {
        case F::SOP2: return kSop2Range;
        case F::SOP1: return kSop1Range;
        case F::SOPK: return kSopkRange;
        case F::SOPC: return kSopcRange;
        case F::SOPP: return kSoppRange;
        case F::VOP2: return kVop2Range;
        case F::VOP1: return kVop1Range;
        case F::VOPC: return kVopcRange;
        case F::VOP3: return kVop3Range;
        default: return {};
    }
}

template <size_t N>
const Rdna2InstructionEffects* lookup(const FormatTable<N>& table, uint32_t opcode) {
    return opcode < N ? &table.by_opcode[opcode] : nullptr;
}

const Rdna2InstructionEffects* entry_for(Rdna2Format fmt, uint32_t opcode) {
    const Rdna2OpcodeRange range = range_for(fmt);
    if (opcode < range.first || opcode >= range.end) return nullptr;
    switch (fmt) {
        case F::SOP2: return lookup(kSop2Table, opcode);
        case F::SOP1: return lookup(kSop1Table, opcode);
        case F::SOPK: return lookup(kSopkTable, opcode);
        case F::SOPC: return lookup(kSopcTable, opcode);
        case F::SOPP: return lookup(kSoppTable, opcode);
        case F::VOP2: return lookup(kVop2Table, opcode);
        case F::VOP1: return lookup(kVop1Table, opcode);
        case F::VOPC: return lookup(kVopcTable, opcode);
        case F::VOP3: return lookup(kVop3Table, opcode);
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
    append(kVop2);
    append(kVop1);
    append(kVopc);
    append(kVop3);
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
    const Rdna2InstructionEffects* entry = entry_for(fmt, opcode);
    return entry ? *entry : kUnknown;
}

bool rdna2_effects_format_is_tabulated(Rdna2Format fmt) {
    return range_for(fmt).end != 0;
}

Rdna2OpcodeRange rdna2_effects_opcode_range(Rdna2Format fmt) {
    return range_for(fmt);
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

// Declared in rdna2_decode.hpp, where its callers already look for it.
uint32_t rdna2_sop2_dest_dwords(uint32_t opcode) {
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(Rdna2Format::SOP2, opcode);
    return e.known() ? e.sdst_dwords : 2u;   // fail closed: a reserved encoding counts as a pair
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
