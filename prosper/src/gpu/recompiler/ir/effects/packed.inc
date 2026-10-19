// ---- VOP3P: ISA 12.10. Every source is one 32-bit register (a packed pair of halves for the v_pk_*
// forms); llvm-mc prints each as one SGPR. Results: the v_pk_* forms write both halves of VDST, the
// dot forms and v_fma_mix_f32 a full 32-bit word, and v_fma_mixlo/mixhi_f16 one half ("result
// stored in low 16 bits of destination", ISA 12.10; the other half is not produced). The rows are
// the UNION of llvm-mc gfx1030 and gfx1013; gfx1013 has no v_dot* forms. CONFIDENCE: HIGH.
constexpr Row kVop3p[] = {
    {F::VOP3P, 0x000, valu("v_pk_mad_i16", 1, 1, 1)},
    {F::VOP3P, 0x001, valu("v_pk_mul_lo_u16", 1, 1, 0)},
    {F::VOP3P, 0x002, valu("v_pk_add_i16", 1, 1, 0)},
    {F::VOP3P, 0x003, valu("v_pk_sub_i16", 1, 1, 0)},
    {F::VOP3P, 0x004, valu("v_pk_lshlrev_b16", 1, 1, 0)},
    {F::VOP3P, 0x005, valu("v_pk_lshrrev_b16", 1, 1, 0)},
    {F::VOP3P, 0x006, valu("v_pk_ashrrev_i16", 1, 1, 0)},
    {F::VOP3P, 0x007, valu("v_pk_max_i16", 1, 1, 0)},
    {F::VOP3P, 0x008, valu("v_pk_min_i16", 1, 1, 0)},
    {F::VOP3P, 0x009, valu("v_pk_mad_u16", 1, 1, 1)},
    {F::VOP3P, 0x00a, valu("v_pk_add_u16", 1, 1, 0)},
    {F::VOP3P, 0x00b, valu("v_pk_sub_u16", 1, 1, 0)},
    {F::VOP3P, 0x00c, valu("v_pk_max_u16", 1, 1, 0)},
    {F::VOP3P, 0x00d, valu("v_pk_min_u16", 1, 1, 0)},
    {F::VOP3P, 0x00e, valu("v_pk_fma_f16", 1, 1, 1)},
    {F::VOP3P, 0x00f, valu("v_pk_add_f16", 1, 1, 0)},
    {F::VOP3P, 0x010, valu("v_pk_mul_f16", 1, 1, 0)},
    {F::VOP3P, 0x011, valu("v_pk_min_f16", 1, 1, 0)},
    {F::VOP3P, 0x012, valu("v_pk_max_f16", 1, 1, 0)},
    {F::VOP3P, 0x013, valu("v_dot2_f32_f16", 1, 1, 1)},   // gfx1030 only
    {F::VOP3P, 0x014, valu("v_dot2_i32_i16", 1, 1, 1)},   // gfx1030 only
    {F::VOP3P, 0x015, valu("v_dot2_u32_u16", 1, 1, 1)},   // gfx1030 only
    {F::VOP3P, 0x016, valu("v_dot4_i32_i8", 1, 1, 1)},   // gfx1030 only
    {F::VOP3P, 0x017, valu("v_dot4_u32_u8", 1, 1, 1)},   // gfx1030 only
    {F::VOP3P, 0x018, valu("v_dot8_i32_i4", 1, 1, 1)},   // gfx1030 only
    {F::VOP3P, 0x019, valu("v_dot8_u32_u4", 1, 1, 1)},   // gfx1030 only
    {F::VOP3P, 0x020, valu("v_fma_mix_f32", 1, 1, 1)},
    {F::VOP3P, 0x021, half(valu("v_fma_mixlo_f16", 1, 1, 1))},
    {F::VOP3P, 0x022, half(valu("v_fma_mixhi_f16", 1, 1, 1))},
};
