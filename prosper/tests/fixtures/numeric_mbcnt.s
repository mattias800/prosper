// Encoding references for the project-owned numeric MBCNT execution fixtures.
// Assemble with llvm-mc -triple=amdgcn -mcpu=gfx1030 -show-encoding.
// Semantics: AMD RDNA2 ISA 70648 section 12.12, instructions 869/870.
v_mbcnt_lo_u32_b32 v7, -1, 0
v_mbcnt_hi_u32_b32 v7, -1, v7
v_mbcnt_lo_u32_b32 v10, 3, 5
v_mbcnt_hi_u32_b32 v10, 0x80000001, 5
v_cvt_f32_u32 v8, v7
v_mov_b32_dpp v2, v8 quad_perm:[0,0,0,0]
v_cvt_f32_u32 v0, v7
v_mul_f32 v0, 0x3b808081, v0
v_cvt_f32_u32 v1, v10
v_mul_f32 v1, 0x3b808081, v1
v_mov_b32 v3, 1.0
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
v_cvt_u32_f32 v1, v0
v_mov_b32 v2, 37
v_cmp_lt_u32_e64 vcc_lo, v1, 16
s_and_saveexec_b64 s[8:9], vcc
v_mbcnt_lo_u32_b32 v2, 3, v1
s_mov_b64 exec, s[8:9]
v_cvt_f32_u32 v0, v2
s_endpgm
s_mov_b32 s6, 0xffffffe0
v_mov_b32 v1, -1
v_mbcnt_lo_u32_b32 v0, v1, s6
v_cvt_f32_u32 v0, v0
s_endpgm
s_mov_b32 s4, 0xaaaaaaaa
v_cvt_u32_f32 v1, v0
v_mbcnt_lo_u32_b32 v0, s4, 5
v_mbcnt_hi_u32_b32 v0, v1, 5
v_mbcnt_lo_u32_b32 v0, 1.0, 5
v_mbcnt_hi_u32_b32 v0, 1.0, 5
v_cvt_f32_u32 v0, v0
s_endpgm
