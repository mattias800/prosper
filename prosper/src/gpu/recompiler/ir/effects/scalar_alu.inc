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
    // PC = S0; D is not written
    {F::SOP1, 0x20, control(alu("s_setpc_b64", 0, 2, 0, false), Rdna2ControlFlow::Indirect)},
    // D = PC + 4; PC = S0
    {F::SOP1, 0x21, control(alu("s_swappc_b64", 2, 2, 0, false), Rdna2ControlFlow::IndirectCall)},
    // PC = S0 (trap return)
    {F::SOP1, 0x22, control(alu("s_rfe_b64", 0, 2, 0, false), Rdna2ControlFlow::Indirect)},
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
    // D = PC + 4; PC += SIMM16 * 4 + 4
    {F::SOPK, 0x16, control(alu("s_call_b64", 2, 0, 0, false), Rdna2ControlFlow::Call)},
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
    {F::SOPP, 0x01, control(plain("s_endpgm"), Rdna2ControlFlow::End)},
    {F::SOPP, 0x02, control(plain("s_branch"), Rdna2ControlFlow::Jump)},
    {F::SOPP, 0x03, plain("s_wakeup")},
    {F::SOPP, 0x04,
     control(op("s_cbranch_scc0", 0, 0, 0, kScc, 0, Rdna2SccRead::Branch),
             Rdna2ControlFlow::BranchScc0)},
    {F::SOPP, 0x05,
     control(op("s_cbranch_scc1", 0, 0, 0, kScc, 0, Rdna2SccRead::Branch),
             Rdna2ControlFlow::BranchScc1)},
    {F::SOPP, 0x06, control(op("s_cbranch_vccz", 0, 0, 0, kVcc), Rdna2ControlFlow::BranchVccz)},
    {F::SOPP, 0x07, control(op("s_cbranch_vccnz", 0, 0, 0, kVcc), Rdna2ControlFlow::BranchVccnz)},
    {F::SOPP, 0x08, control(op("s_cbranch_execz", 0, 0, 0, kExec), Rdna2ControlFlow::BranchExecz)},
    {F::SOPP, 0x09,
     control(op("s_cbranch_execnz", 0, 0, 0, kExec), Rdna2ControlFlow::BranchExecnz)},
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
    {F::SOPP, 0x12, control(plain("s_trap"), Rdna2ControlFlow::Trap)},
    {F::SOPP, 0x13, plain("s_icache_inv")},
    {F::SOPP, 0x14, plain("s_incperflevel")},
    {F::SOPP, 0x15, plain("s_decperflevel")},
    {F::SOPP, 0x16, op("s_ttracedata", 0, 0, 0, kM0)},   // sends M0 to the thread trace
    {F::SOPP, 0x17, control(plain("s_cbranch_cdbgsys"), Rdna2ControlFlow::DebugBranch)},
    {F::SOPP, 0x18, control(plain("s_cbranch_cdbguser"), Rdna2ControlFlow::DebugBranch)},
    {F::SOPP, 0x19, control(plain("s_cbranch_cdbgsys_or_user"), Rdna2ControlFlow::DebugBranch)},
    {F::SOPP, 0x1a, control(plain("s_cbranch_cdbgsys_and_user"), Rdna2ControlFlow::DebugBranch)},
    {F::SOPP, 0x1b, control(plain("s_endpgm_saved"), Rdna2ControlFlow::End)},
    {F::SOPP, 0x1e, control(plain("s_endpgm_ordered_ps_done"), Rdna2ControlFlow::End)},
    {F::SOPP, 0x1f, control(plain("s_code_end"), Rdna2ControlFlow::End)},
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

