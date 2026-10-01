// Project-owned WQM packet references. AMD RDNA2 ISA 70648 section 12.3, ops 9/10.
// llvm-mc -triple=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize64,-wavefrontsize32 -show-encoding
s_wqm_b32 s20, 1
s_wqm_b32 s20, 1.0
s_wqm_b32 s20, 0x80000001
s_wqm_b64 s[20:21], -2
s_wqm_b64 s[20:21], 0x80000001
s_wqm_b64 s[20:21], s[4:5]
s_wqm_b64 s[4:5], s[4:5]
s_mov_b32 s4, 0x80000000
s_mov_b32 s5, 1
s_cmp_eq_u32 0, 1
s_cmp_eq_u32 0, 0
s_mov_b32 s30, scc
v_mov_b32 v0, s20
v_mov_b32 v0, s21
v_mov_b32 v0, s30
s_mov_b64 s[20:21], exec
s_wqm_b32 s21, 1
s_mov_b32 s20, 2
s_bcnt1_i32_b64 s30, s[20:21]
s_mov_b64 exec, 0
s_mov_b64 exec, -1
s_branch 1
s_nop 0
s_endpgm
s_wqm_b64 s[4:5], 0
s_wqm_b64 s[4:5], 15
s_wqm_b64 s[4:5], -1
s_wqm_b64 s[4:5], -16
s_wqm_b64 exec, 15
s_mov_b64 exec, s[4:5]
s_not_b64 exec, s[4:5]
s_mov_b64 s[8:9], s[4:5]
s_and_saveexec_b64 s[20:21], s[4:5]
s_cselect_b64 exec, s[4:5], 0
s_and_b64 exec, s[4:5], -1
v_cndmask_b32_e64 v0, 7, 1, s[4:5]
v_add_co_ci_u32_e64 v0, vcc, 1, 2, s[4:5]
