// ---- VALU: VOP2 (ISA 12.7), VOP1 (12.8), VOPC (12.9) and VOP3 (12.12; VOP2 and VOP1 promote to
// VOP3 at +0x100 and +0x180, VOPC keeps its number and decodes as VOPC). Status Valu: the name,
// the source widths, the vector-result fields and the implicit reads are described.
//
// VECTOR RESULT. valu() is the common shape: one VGPR written lane-wise from the explicit operands.
// Every other shape is stated by a wrapper, and each was derived from the ISA pseudo-code and
// cross-checked against llvm-mc (#5143, #5146 step 4):
//   * wide(n): VDST is n VGPRs -- the register class llvm-mc prints for VDST (v[4:5]), which agrees
//     with the pseudo-code's D.u64/D.f64/D.d on every row; 0 for v_readlane/v_readfirstlane, whose
//     VDST field names an SGPR.
//   * half: D is one 16-bit value -- the mnemonic's first type token (the D type in AMD naming) is
//     16-bit, which agrees with the pseudo-code's D.f16/D.u16/D.i16/D.f16_lo. Not packed results
//     (v_pk_*, *_pk_*, v_cvt_pk*, v_pack_b32_f16; v_sat_pk_u8_i16 is "D.u32 = {16'b0, ...}"), and
//     not v_sad_u16 (D.u32) or v_interp_p1ll/p1lv_f16 (D.f32). v_fma_mix* are VOP3P.
//   * accumulate: the pseudo-code reads D on its right-hand side ("D.f32 = S0.f32 * S1.f32 +
//     D.f32"): v_mac_f32, v_fmac_f32, v_fmac_legacy_f32, v_fmac_f16, v_pk_fmac_f16, v_dot2c_f32_f16
//     and v_dot4c_i32_i8, in every encoding that exists.
//   * flow(...): v_readfirstlane/v_readlane (ReadLane; readfirstlane reads EXEC to pick the lane),
//     v_writelane ("Ignores exec mask"), v_permlane16/x16 (gather within a row), v_mbcnt_lo/hi
//     (ThreadMask of the lane's own index), v_swap_b32, the VOP3 v_interp_* forms (LDS parameters
//     at M0). v_cndmask_b32 and the three carry-in forms read VCC implicitly only in their VOP2
//     (e32/SDWA/DPP) encoding -- VOP3 names the mask as SRC2 -- and v_div_fmas_f32/f64 read VCC in
//     VOP3 (ISA 12.12: "VCC is used as the carry/borrow input").
//   * relative: ISA 6.6 Table 25, VGPR[D+M0] / VGPR[S0+M0]: v_movreld/movrels/movrelsd/movrelsd_2
//     in VOP1 and VOP3, v_swaprel_b32 in VOP1.
//   * swaps: v_swap_b32 and v_swaprel_b32 also write SRC0's VGPR ("swap two VGPRs", ISA 6.6).
//   * no_result: v_nop, v_pipeflush and v_clrexcp (VOP1 0x00/0x1B/0x41 and VOP3 0x180/0x19B/0x1C1).
//     PR 2 listed these six slots as reserved because its llvm-mc sweep gave them SGPR sources,
//     which llvm-mc rejects for an operand-less instruction; with no operands every one of them
//     disassembles, and ISA 12.8 names all three.
//
// SOURCE WIDTHS
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
// A VGPR-only source is listed with its width (v_readlane_b32's SRC0; v_mqsad_u32_u8's SRC2 is
// the 128-bit VReg_128 llvm-mc requires -- an SGPR there is not encodable, and charging one the
// quad only fails closed); an attribute field that is
// not a register operand (the v_interp forms) is 0. A literal constant (v_madak's K) is not a source.
constexpr Row kVop2[] = {
    {F::VOP2, 0x001, flow(valu("v_cndmask_b32", 1, 1, 0), Rdna2LaneFlow::LaneWise, kVcc)},
    {F::VOP2, 0x002, accumulate(valu("v_dot2c_f32_f16", 1, 1, 0))},
    {F::VOP2, 0x003, valu("v_add_f32", 1, 1, 0)},
    {F::VOP2, 0x004, valu("v_sub_f32", 1, 1, 0)},
    {F::VOP2, 0x005, valu("v_subrev_f32", 1, 1, 0)},
    {F::VOP2, 0x006, accumulate(valu("v_fmac_legacy_f32", 1, 1, 0))},
    {F::VOP2, 0x007, valu("v_mul_legacy_f32", 1, 1, 0)},
    {F::VOP2, 0x008, valu("v_mul_f32", 1, 1, 0)},
    {F::VOP2, 0x009, valu("v_mul_i32_i24", 1, 1, 0)},
    {F::VOP2, 0x00a, valu("v_mul_hi_i32_i24", 1, 1, 0)},
    {F::VOP2, 0x00b, valu("v_mul_u32_u24", 1, 1, 0)},
    {F::VOP2, 0x00c, valu("v_mul_hi_u32_u24", 1, 1, 0)},
    {F::VOP2, 0x00d, accumulate(valu("v_dot4c_i32_i8", 1, 1, 0))},
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
    {F::VOP2, 0x01f, accumulate(valu("v_mac_f32", 1, 1, 0))},
    {F::VOP2, 0x020, valu("v_madmk_f32", 1, 1, 0)},
    {F::VOP2, 0x021, valu("v_madak_f32", 1, 1, 0)},
    {F::VOP2, 0x025, valu("v_add_nc_u32", 1, 1, 0)},
    {F::VOP2, 0x026, valu("v_sub_nc_u32", 1, 1, 0)},
    {F::VOP2, 0x027, valu("v_subrev_nc_u32", 1, 1, 0)},
    {F::VOP2, 0x028, flow(valu("v_add_co_ci_u32", 1, 1, 0), Rdna2LaneFlow::LaneWise, kVcc)},
    {F::VOP2, 0x029, flow(valu("v_sub_co_ci_u32", 1, 1, 0), Rdna2LaneFlow::LaneWise, kVcc)},
    {F::VOP2, 0x02a, flow(valu("v_subrev_co_ci_u32", 1, 1, 0), Rdna2LaneFlow::LaneWise, kVcc)},
    {F::VOP2, 0x02b, accumulate(valu("v_fmac_f32", 1, 1, 0))},
    {F::VOP2, 0x02c, valu("v_fmamk_f32", 1, 1, 0)},
    {F::VOP2, 0x02d, valu("v_fmaak_f32", 1, 1, 0)},
    {F::VOP2, 0x02f, valu("v_cvt_pkrtz_f16_f32", 1, 1, 0)},
    {F::VOP2, 0x032, half(valu("v_add_f16", 1, 1, 0))},
    {F::VOP2, 0x033, half(valu("v_sub_f16", 1, 1, 0))},
    {F::VOP2, 0x034, half(valu("v_subrev_f16", 1, 1, 0))},
    {F::VOP2, 0x035, half(valu("v_mul_f16", 1, 1, 0))},
    {F::VOP2, 0x036, accumulate(half(valu("v_fmac_f16", 1, 1, 0)))},
    {F::VOP2, 0x037, half(valu("v_fmamk_f16", 1, 1, 0))},
    {F::VOP2, 0x038, half(valu("v_fmaak_f16", 1, 1, 0))},
    {F::VOP2, 0x039, half(valu("v_max_f16", 1, 1, 0))},
    {F::VOP2, 0x03a, half(valu("v_min_f16", 1, 1, 0))},
    {F::VOP2, 0x03b, half(valu("v_ldexp_f16", 1, 1, 0))},
    {F::VOP2, 0x03c, accumulate(valu("v_pk_fmac_f16", 1, 1, 0))},
};
constexpr Rdna2InstructionEffects f32_input_sensitive_round(const char* name) {
    auto result = valu(name, 1, 0, 0);
    result.f32_rounding_input_sensitive = true;
    return result;
}
constexpr Row kVop1[] = {
    {F::VOP1, 0x000, no_result("v_nop")},
    {F::VOP1, 0x001, valu("v_mov_b32", 1, 0, 0)},
    {F::VOP1, 0x002,
     flow(wide(valu("v_readfirstlane_b32", 1, 0, 0), 0), Rdna2LaneFlow::ReadLane, kExec)},
    {F::VOP1, 0x003, valu("v_cvt_i32_f64", 2, 0, 0)},
    {F::VOP1, 0x004, wide(valu("v_cvt_f64_i32", 1, 0, 0), 2)},
    {F::VOP1, 0x005, valu("v_cvt_f32_i32", 1, 0, 0)},
    {F::VOP1, 0x006, valu("v_cvt_f32_u32", 1, 0, 0)},
    {F::VOP1, 0x007, valu("v_cvt_u32_f32", 1, 0, 0)},
    {F::VOP1, 0x008, valu("v_cvt_i32_f32", 1, 0, 0)},
    {F::VOP1, 0x00a, half(valu("v_cvt_f16_f32", 1, 0, 0))},
    {F::VOP1, 0x00b, valu("v_cvt_f32_f16", 1, 0, 0)},
    {F::VOP1, 0x00c, valu("v_cvt_rpi_i32_f32", 1, 0, 0)},
    {F::VOP1, 0x00d, valu("v_cvt_flr_i32_f32", 1, 0, 0)},
    {F::VOP1, 0x00e, valu("v_cvt_off_f32_i4", 1, 0, 0)},
    {F::VOP1, 0x00f, valu("v_cvt_f32_f64", 2, 0, 0)},
    {F::VOP1, 0x010, wide(valu("v_cvt_f64_f32", 1, 0, 0), 2)},
    {F::VOP1, 0x011, valu("v_cvt_f32_ubyte0", 1, 0, 0)},
    {F::VOP1, 0x012, valu("v_cvt_f32_ubyte1", 1, 0, 0)},
    {F::VOP1, 0x013, valu("v_cvt_f32_ubyte2", 1, 0, 0)},
    {F::VOP1, 0x014, valu("v_cvt_f32_ubyte3", 1, 0, 0)},
    {F::VOP1, 0x015, valu("v_cvt_u32_f64", 2, 0, 0)},
    {F::VOP1, 0x016, wide(valu("v_cvt_f64_u32", 1, 0, 0), 2)},
    {F::VOP1, 0x017, wide(valu("v_trunc_f64", 2, 0, 0), 2)},
    {F::VOP1, 0x018, wide(valu("v_ceil_f64", 2, 0, 0), 2)},
    {F::VOP1, 0x019, wide(valu("v_rndne_f64", 2, 0, 0), 2)},
    {F::VOP1, 0x01a, wide(valu("v_floor_f64", 2, 0, 0), 2)},
    {F::VOP1, 0x01b, no_result("v_pipeflush")},
    {F::VOP1, 0x020, valu("v_fract_f32", 1, 0, 0)},
    {F::VOP1, 0x021, valu("v_trunc_f32", 1, 0, 0)},
    {F::VOP1, 0x022, f32_input_sensitive_round("v_ceil_f32")},
    {F::VOP1, 0x023, valu("v_rndne_f32", 1, 0, 0)},
    {F::VOP1, 0x024, f32_input_sensitive_round("v_floor_f32")},
    {F::VOP1, 0x025, valu("v_exp_f32", 1, 0, 0)},
    {F::VOP1, 0x027, valu("v_log_f32", 1, 0, 0)},
    {F::VOP1, 0x02a, valu("v_rcp_f32", 1, 0, 0)},
    {F::VOP1, 0x02b, valu("v_rcp_iflag_f32", 1, 0, 0)},
    {F::VOP1, 0x02e, valu("v_rsq_f32", 1, 0, 0)},
    {F::VOP1, 0x02f, wide(valu("v_rcp_f64", 2, 0, 0), 2)},
    {F::VOP1, 0x031, wide(valu("v_rsq_f64", 2, 0, 0), 2)},
    {F::VOP1, 0x033, valu("v_sqrt_f32", 1, 0, 0)},
    {F::VOP1, 0x034, wide(valu("v_sqrt_f64", 2, 0, 0), 2)},
    {F::VOP1, 0x035, valu("v_sin_f32", 1, 0, 0)},
    {F::VOP1, 0x036, valu("v_cos_f32", 1, 0, 0)},
    {F::VOP1, 0x037, valu("v_not_b32", 1, 0, 0)},
    {F::VOP1, 0x038, valu("v_bfrev_b32", 1, 0, 0)},
    {F::VOP1, 0x039, valu("v_ffbh_u32", 1, 0, 0)},
    {F::VOP1, 0x03a, valu("v_ffbl_b32", 1, 0, 0)},
    {F::VOP1, 0x03b, valu("v_ffbh_i32", 1, 0, 0)},
    {F::VOP1, 0x03c, valu("v_frexp_exp_i32_f64", 2, 0, 0)},
    {F::VOP1, 0x03d, wide(valu("v_frexp_mant_f64", 2, 0, 0), 2)},
    {F::VOP1, 0x03e, wide(valu("v_fract_f64", 2, 0, 0), 2)},
    {F::VOP1, 0x03f, valu("v_frexp_exp_i32_f32", 1, 0, 0)},
    {F::VOP1, 0x040, valu("v_frexp_mant_f32", 1, 0, 0)},
    {F::VOP1, 0x041, no_result("v_clrexcp")},
    {F::VOP1, 0x042, relative(valu("v_movreld_b32", 1, 0, 0), Rdna2M0Relative::Destination)},
    {F::VOP1, 0x043, relative(valu("v_movrels_b32", 1, 0, 0), Rdna2M0Relative::Source)},
    {F::VOP1, 0x044, relative(valu("v_movrelsd_b32", 1, 0, 0), Rdna2M0Relative::Both)},
    {F::VOP1, 0x048, relative(valu("v_movrelsd_2_b32", 1, 0, 0), Rdna2M0Relative::Both)},
    {F::VOP1, 0x050, half(valu("v_cvt_f16_u16", 1, 0, 0))},
    {F::VOP1, 0x051, half(valu("v_cvt_f16_i16", 1, 0, 0))},
    {F::VOP1, 0x052, half(valu("v_cvt_u16_f16", 1, 0, 0))},
    {F::VOP1, 0x053, half(valu("v_cvt_i16_f16", 1, 0, 0))},
    {F::VOP1, 0x054, half(valu("v_rcp_f16", 1, 0, 0))},
    {F::VOP1, 0x055, half(valu("v_sqrt_f16", 1, 0, 0))},
    {F::VOP1, 0x056, half(valu("v_rsq_f16", 1, 0, 0))},
    {F::VOP1, 0x057, half(valu("v_log_f16", 1, 0, 0))},
    {F::VOP1, 0x058, half(valu("v_exp_f16", 1, 0, 0))},
    {F::VOP1, 0x059, half(valu("v_frexp_mant_f16", 1, 0, 0))},
    {F::VOP1, 0x05a, half(valu("v_frexp_exp_i16_f16", 1, 0, 0))},
    {F::VOP1, 0x05b, half(valu("v_floor_f16", 1, 0, 0))},
    {F::VOP1, 0x05c, half(valu("v_ceil_f16", 1, 0, 0))},
    {F::VOP1, 0x05d, half(valu("v_trunc_f16", 1, 0, 0))},
    {F::VOP1, 0x05e, half(valu("v_rndne_f16", 1, 0, 0))},
    {F::VOP1, 0x05f, half(valu("v_fract_f16", 1, 0, 0))},
    {F::VOP1, 0x060, half(valu("v_sin_f16", 1, 0, 0))},
    {F::VOP1, 0x061, half(valu("v_cos_f16", 1, 0, 0))},
    {F::VOP1, 0x062, valu("v_sat_pk_u8_i16", 1, 0, 0)},
    {F::VOP1, 0x063, half(valu("v_cvt_norm_i16_f16", 1, 0, 0))},
    {F::VOP1, 0x064, half(valu("v_cvt_norm_u16_f16", 1, 0, 0))},
    {F::VOP1, 0x065, swaps(flow(valu("v_swap_b32", 1, 0, 0), Rdna2LaneFlow::Swap))},
    {F::VOP1, 0x068, swaps(relative(valu("v_swaprel_b32", 1, 0, 0), Rdna2M0Relative::Both))},
};
