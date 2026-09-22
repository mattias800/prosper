// Assemble with llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1030
// -mattr=+wavefrontsize64,-wavefrontsize32 -show-encoding; the words are checked into
// tests/gpu/execute/test_fragment_partial_wave_exec.cpp.
//
// A Wave64 pixel shader that reports, per pixel, what its own wave looked like -- in a form whose
// correctness can be checked WITHOUT knowing which pixels the host grouped (#3464):
//   R = my rank among the launched lanes (MBCNT over the entry EXEC)            / 255
//   G = the number of launched lanes (V_BCNT over both halves of the entry EXEC) / 255
//   B = flags / 255: bit0 my own bit is set in the saved entry EXEC (the ballot/lane-id
//       read-back GTA V uses to restore its non-helper lanes); bit1 exactly one lane has
//       rank 0 (V_BCNT over a VOPC-written SGPR pair); bit2 a vote over that pair is true
//   A = 1.0 when the HIGH dword of the entry EXEC is zero, else 0.5
.text
s_mov_b64 s[0:1], exec
v_bcnt_u32_b32 v5, s0, 0
v_bcnt_u32_b32 v5, s1, v5
v_mbcnt_lo_u32_b32 v4, s0, 0
v_mbcnt_hi_u32_b32 v4, s1, v4
v_cmp_eq_u32_e64 s[8:9], 0, v4
v_bcnt_u32_b32 v7, s8, 0
v_bcnt_u32_b32 v7, s9, v7
v_cmp_eq_u32 vcc, 1, v7
v_cndmask_b32_e64 v8, 0, 2, vcc
s_cmp_lg_u64 s[8:9], 0
s_cselect_b32 s7, 4, 0
v_cndmask_b32_e64 v6, 0, 1, s[0:1]
v_or3_b32 v6, v6, v8, s7
s_cmp_eq_u32 s1, 0
s_cselect_b32 s3, 1.0, 0.5
v_cvt_f32_u32 v0, v4
v_mul_f32 v0, 0x3b808081, v0
v_cvt_f32_u32 v1, v5
v_mul_f32 v1, 0x3b808081, v1
v_cvt_f32_u32 v2, v6
v_mul_f32 v2, 0x3b808081, v2
v_mov_b32 v3, s3
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
