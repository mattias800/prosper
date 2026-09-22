// Assemble as fragment_loop_mask.s. Inner loop takes zero iterations; upper-half seed must survive.
// Blue independently witnesses covered upper-half lanes; red reports whether the seed was correct.
.text
s_mov_b32 s84, 0
s_mov_b32 s85, -1
s_mov_b64 s[88:89], exec
v_mbcnt_lo_u32_b32 v7, -1, 0
v_mbcnt_hi_u32_b32 v7, -1, v7
v_and_b32 v9, 3, v7
v_add_nc_u32 v9, 1, v9
v_and_b32 v11, 1, v7
v_mov_b32 v10, 2
outer:
s_cbranch_execz done
v_mov_b32 v8, 0
v_cmp_eq_u32_e64 s[92:93], v11, v10
s_mov_b64 exec, s[92:93]
s_mov_b64 s[86:87], exec
inner:
s_cbranch_execz inner_done
v_cmp_ge_u32_e64 s[92:93], v9, 0
s_andn2_b64 s[68:69], s[84:85], exec
s_mov_b64 vcc, s[92:93]
s_or_b64 s[84:85], s[92:93], s[68:69]
s_mov_b64 exec, s[92:93]
s_cbranch_vccz inner_done
v_add_nc_u32 v8, 1, v8
v_cmp_ge_u32 vcc, v8, v9
s_andn2_b64 exec, s[92:93], vcc
s_andn2_b64 vcc, s[84:85], s[92:93]
s_or_b64 s[84:85], s[92:93], vcc
s_cbranch_execnz inner
inner_done:
s_mov_b64 exec, s[88:89]
v_add_nc_u32 v10, 1, v10
v_cmp_lt_u32 vcc, v10, 2
s_mov_b64 exec, vcc
s_cbranch_execnz outer
done:
s_mov_b64 exec, s[88:89]
s_and_b64 vcc, s[84:85], s[88:89]
v_cndmask_b32 v12, 0, 1, vcc
v_cmp_ge_u32 vcc, v7, 32
v_cndmask_b32 v2, 0, 1.0, vcc
v_cndmask_b32 v13, 0, 1, vcc
v_cmp_eq_u32 vcc, v12, v13
v_cndmask_b32 v0, 0, 1.0, vcc
v_mov_b32 v1, 1.0
v_mov_b32 v3, 1.0
exp mrt0 v0, v1, v2, v3 done
s_endpgm
