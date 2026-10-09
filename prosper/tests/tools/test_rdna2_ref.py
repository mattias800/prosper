"""Known-answer tests for the independent RDNA2 reference interpreter (tools/rdna2_ref).

Every expected value below was computed by hand from the AMD "RDNA 2" ISA reference guide
(doc 70648), not by running the interpreter; each test cites the section it checks. The
instruction words are assembled by the small encoders in this file from the ISA's ch. 13
field layouts, so the tests do not depend on the decoder's own field extraction logic for
construction. All data is synthetic.
"""

import shutil
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from rdna2_ref import decode as D  # noqa: E402
from rdna2_ref import llvm_check as L  # noqa: E402
from rdna2_ref import machine as M  # noqa: E402

VCC, EXEC, LIT = 106, 126, 255
SCC_SRC = 253


def V(n):
    """Operand code of VGPR n."""
    return 256 + n


def IC(k):
    """Operand code of inline integer constant k (0..64)."""
    return 128 + k


def NEG(k):
    """Operand code of inline integer constant -k (1..16)."""
    return 192 + k


# ---- encoders (ISA ch. 13 field layouts) -----------------------------------------------
def sop2(op, sdst, s0, s1):
    return [(0b10 << 30) | (op << 23) | (sdst << 16) | (s1 << 8) | s0]


def sop1(op, sdst, s0):
    return [0xBE800000 | (sdst << 16) | (op << 8) | s0]


def sopc(op, s0, s1):
    return [0xBF000000 | (op << 16) | (s1 << 8) | s0]


def sopp(op, simm=0):
    return [0xBF800000 | (op << 16) | (simm & 0xFFFF)]


def sopk(op, sdst, simm):
    return [0xB0000000 | (op << 23) | (sdst << 16) | (simm & 0xFFFF)]


def vop1(op, vdst, src0, lit=None):
    return [0x7E000000 | (vdst << 17) | (op << 9) | src0] + ([lit] if lit is not None else [])


def vop2(op, vdst, src0, vsrc1):
    return [(op << 25) | (vdst << 17) | (vsrc1 << 9) | src0]


def vopc(op, src0, vsrc1):
    return [0x7C000000 | (op << 17) | (vsrc1 << 9) | src0]


def vop3(op, vdst, s0, s1=0, s2=0, sdst=0):
    return [0xD4000000 | (op << 16) | (sdst << 8) | vdst, s0 | (s1 << 9) | (s2 << 18)]


def smem(op, sbase, sdata, offset, soffset=125):
    return [0xF4000000 | (op << 18) | (sbase // 2) | (sdata << 6), offset | (soffset << 25)]


END = sopp(1)
M64 = (1 << 64) - 1


def code(*parts):
    flat = [w for p in parts for w in p]
    return struct.pack(f"<{len(flat)}I", *flat)


def run(*parts, wave=64, exec_=None, sgpr=None, vgpr=None, scc=0, mem=None, steps=100_000):
    """Assemble and run; returns (Wave, result dict). EXEC defaults to all lanes."""
    w = M.Wave(size=wave)
    w.exec = w.lane_mask if exec_ is None else exec_
    w.scc = scc
    for k, v in (sgpr or {}).items():
        w.sgpr[k] = v
    for (n, lane), v in (vgpr or {}).items():
        w.vgpr[n][lane] = v
    for addr, data in (mem or {}).items():
        w.mem.write(addr, data)
    res = M.Machine(w, code(*parts, END)).run(steps)
    assert res["status"] == "ended", res
    return w, res


def pair(w, n):
    return w.sgpr[n] | (w.sgpr[n + 1] << 32)


# ---- opcode tables (ISA tables 64-83): spot values ---------------------------------------
def test_table_spot_values():
    assert D.SOP2[28] == "S_XNOR_B32" and D.SOP2[4] == "S_ADDC_U32"
    assert D.SOP1[36] == "S_AND_SAVEEXEC_B64" and D.SOP1[3] == "S_MOV_B32"
    assert D.SOPP[8] == "S_CBRANCH_EXECZ" and D.SOPC[18] == "S_CMP_EQ_U64"
    assert D.VOP3[864] == "V_READLANE_B32" and D.VOP3[865] == "V_WRITELANE_B32"
    assert D.VOP2[40] == "V_ADD_CO_CI_U32" and D.VOPC[194] == "V_CMP_EQ_U32"
    assert D.SMEM[2] == "S_LOAD_DWORDX4" and D.VOP3[783] == "V_ADD_CO_U32"


# ---- SOP2 arithmetic (ISA 12.1, S_ADD_U32 etc.) -----------------------------------------
def test_s_add_u32_carry():  # D = S0 + S1, SCC = carry out
    w, _ = run(sop2(0, 0, 1, IC(1)), sgpr={1: 0xFFFFFFFF})
    assert w.sgpr[0] == 0 and w.scc == 1


def test_s_add_u32_no_carry():
    w, _ = run(sop2(0, 0, 1, 2), sgpr={1: 5, 2: 7}, scc=1)
    assert w.sgpr[0] == 12 and w.scc == 0


def test_s_sub_u32_borrow():  # SCC = borrow (S1 > S0)
    w, _ = run(sop2(1, 0, 1, 2), sgpr={1: 3, 2: 5})
    assert w.sgpr[0] == 0xFFFFFFFE and w.scc == 1
    w, _ = run(sop2(1, 0, 1, 2), sgpr={1: 5, 2: 3}, scc=1)
    assert w.sgpr[0] == 2 and w.scc == 0


def test_s_add_i32_overflow():  # SCC = signed overflow
    w, _ = run(sop2(2, 0, 1, IC(1)), sgpr={1: 0x7FFFFFFF})
    assert w.sgpr[0] == 0x80000000 and w.scc == 1


def test_s_addc_chain_64bit_add():
    # (hi=1, lo=0xFFFFFFFF) + (hi=0, lo=1) = 0x2_00000000: lo 0 carry 1, hi 1+0+1 = 2
    w, _ = run(
        sop2(0, 4, 0, 2),
        sop2(4, 5, 1, 3),
        sgpr={0: 0xFFFFFFFF, 1: 1, 2: 1, 3: 0},
    )
    assert (w.sgpr[4], w.sgpr[5], w.scc) == (0, 2, 0)


def test_s_addc_uses_scc_in():
    w, _ = run(sop2(4, 0, 1, 2), sgpr={1: 10, 2: 20}, scc=1)
    assert w.sgpr[0] == 31 and w.scc == 0


def test_s_subb_chain_64bit_sub():
    # (hi=2, lo=0) - (hi=0, lo=1): lo = 0xFFFFFFFF borrow 1; hi = 2 - 0 - 1 = 1, no borrow
    w, _ = run(sop2(1, 4, 0, 2), sop2(5, 5, 1, 3), sgpr={0: 0, 1: 2, 2: 1, 3: 0})
    assert (w.sgpr[4], w.sgpr[5], w.scc) == (0xFFFFFFFF, 1, 0)


def test_s_min_max_scc_selects_s0():
    w, _ = run(sop2(6, 0, 1, IC(1)), sgpr={1: 0xFFFFFFFF})  # min_i32(-1, 1) = -1, S0 chosen
    assert w.sgpr[0] == 0xFFFFFFFF and w.scc == 1
    w, _ = run(sop2(7, 0, 1, IC(1)), sgpr={1: 0xFFFFFFFF})  # min_u32(0xFFFFFFFF, 1) = 1
    assert w.sgpr[0] == 1 and w.scc == 0


def test_s_mul_i32():
    w, _ = run(sop2(38, 0, NEG(3), IC(5)))  # -3 * 5 = -15
    assert w.sgpr[0] == 0xFFFFFFF1


def test_s_bfe_u32_and_bfm():
    # S1 = width 8 << 16 | offset 8: (0xABCD1234 >> 8) & 0xFF = 0x12
    w, _ = run(sop2(39, 0, 1, 2), sgpr={1: 0xABCD1234, 2: 0x00080008})
    assert w.sgpr[0] == 0x12 and w.scc == 1
    w, _ = run(sop2(36, 0, IC(4), IC(8)))  # ((1<<4)-1) << 8
    assert w.sgpr[0] == 0xF00


# ---- SOP2 logic, 32 and 64-bit, pair boundary (ISA 12.1 S_AND_B64 etc.) ------------------
def test_s_and_b64_pair():
    w, _ = run(
        sop2(15, 4, 0, 2),
        sgpr={0: 0xFFFF0000, 1: 0xFFFF0000, 2: 0xFFFFFFFF, 3: 0x0F0F0F0F},
    )
    assert pair(w, 4) == 0x0F0F0000_FFFF0000 and w.scc == 1


def test_s_andn2_b64_operand_order():  # D = S0 & ~S1
    w, _ = run(
        sop2(21, 4, 0, 2),
        sgpr={0: 0xFFFFFFFF, 1: 0xFFFFFFFF, 2: 0xFFFFFFFF, 3: 0},
    )
    assert pair(w, 4) == 0xFFFFFFFF_00000000


def test_s_orn2_b32_operand_order():  # D = S0 | ~S1
    w, _ = run(sop2(22, 0, 1, 2), sgpr={1: 0xF, 2: 0xFFFFFFF0})
    assert w.sgpr[0] == 0xF
    w, _ = run(sop2(22, 0, 1, 2), sgpr={1: 0x0, 2: 0xFFFFFFF0})
    assert w.sgpr[0] == 0xF


def test_s_nand_nor_xnor_b32():
    a, b = 0xFF00FF00, 0x0FF00FF0
    s = {1: a, 2: b}
    assert run(sop2(24, 0, 1, 2), sgpr=s)[0].sgpr[0] == 0xF0FFF0FF  # ~(a & b)
    assert run(sop2(26, 0, 1, 2), sgpr=s)[0].sgpr[0] == 0x000F000F  # ~(a | b)
    assert run(sop2(28, 0, 1, 2), sgpr=s)[0].sgpr[0] == 0x0F0F0F0F  # ~(a ^ b)


def test_s_and_b32_scc_is_nonzero():
    w, _ = run(sop2(14, 0, 1, 2), sgpr={1: 0xF0, 2: 0x0F}, scc=1)
    assert w.sgpr[0] == 0 and w.scc == 0


def test_s_shifts_b64():
    s = {0: 0x80000001, 1: 0}
    w, _ = run(sop2(31, 4, 0, IC(4)), sgpr=s)  # lshl 4 -> 0x8_00000010
    assert pair(w, 4) == 0x0000000800000010
    w, _ = run(sop2(33, 4, 0, IC(4)), sgpr=s)  # lshr 4 -> 0x08000000
    assert pair(w, 4) == 0x08000000
    w, _ = run(sop2(35, 4, 0, IC(4)), sgpr={0: 0, 1: 0x80000000})  # ashr 4
    assert pair(w, 4) == 0xF8000000_00000000


def test_s_cselect_b64_scc0_and_scc1():
    s = {2: 0x11, 3: 0x22, 4: 0x33, 5: 0x44}
    w, _ = run(sop2(11, VCC, 2, 4), sgpr=s, scc=0)  # SCC=0 -> S1 = s[4:5]
    assert pair(w, VCC) == 0x44_00000033
    w, _ = run(sop2(11, VCC, 2, 4), sgpr=s, scc=1)  # SCC=1 -> S0 = s[2:3]
    assert pair(w, VCC) == 0x22_00000011


# ---- SOP1 / SOPK -------------------------------------------------------------------------
def test_s_mov_b64_inline_minus_one_and_literal():
    w, _ = run(sop1(4, 0, NEG(1)))  # inline -1 sign-extends to 64 bits
    assert pair(w, 0) == M64
    w, _ = run(sop1(3, 0, LIT) + [0xDEADBEEF])
    assert w.sgpr[0] == 0xDEADBEEF


def test_s_not_b64():
    w, _ = run(sop1(8, 4, 0), sgpr={0: 0xFFFFFFFF, 1: 0})
    assert pair(w, 4) == 0xFFFFFFFF_00000000 and w.scc == 1


def test_s_bitset1_b64_high_half():
    w, _ = run(sop1(30, 0, IC(33)))  # bit 33 of s[0:1]
    assert (w.sgpr[0], w.sgpr[1]) == (0, 2)


def test_s_and_saveexec_b64():
    w, _ = run(
        sop1(36, 0, 2),
        sgpr={2: 0xFF00FF00, 3: 0xFF00FF00},
        exec_=0xF0F0F0F0_F0F0F0F0,
    )
    assert w.exec == 0xF000F000_F000F000 and pair(w, 0) == 0xF0F0F0F0_F0F0F0F0 and w.scc == 1
    w, _ = run(sop1(36, 0, 2), sgpr={2: 0x0F0F0F0F, 3: 0x0F0F0F0F}, exec_=0xF0F0F0F0_F0F0F0F0)
    assert w.exec == 0 and w.scc == 0


def test_s_andn2_and_andn1_saveexec():
    # exec = 0xFF, S0 = 0xFF0: ANDN2 -> S0 & ~EXEC = 0xF00; ANDN1 -> ~S0 & EXEC = 0x0F
    w, _ = run(sop1(39, 0, 2), sgpr={2: 0xFF0, 3: 0}, exec_=0xFF)
    assert w.exec == 0xF00 and pair(w, 0) == 0xFF
    w, _ = run(sop1(55, 0, 2), sgpr={2: 0xFF0, 3: 0}, exec_=0xFF)
    assert w.exec == 0x0F


def test_s_movk_and_cmpk():
    w, _ = run(sopk(0, 0, 0xFFFE))  # s_movk_i32 sign-extends -2
    assert w.sgpr[0] == 0xFFFFFFFE
    w, _ = run(sopk(13, 0, 0xFFFF), sgpr={0: 0xFFFFFFFE})  # cmpk_lt_u32: simm zero-extended
    assert w.scc == 0
    w, _ = run(sopk(7, 0, 1), sgpr={0: 0xFFFFFFFE})  # cmpk_lt_i32: -2 < 1
    assert w.scc == 1


# ---- SOPC ---------------------------------------------------------------------------------
def test_s_cmp_32_signed_unsigned():
    assert run(sopc(0, 1, 2), sgpr={1: 7, 2: 7})[0].scc == 1  # eq_i32
    assert run(sopc(4, 1, IC(0)), sgpr={1: 0xFFFFFFFF})[0].scc == 1  # lt_i32: -1 < 0
    assert run(sopc(10, 1, IC(0)), sgpr={1: 0xFFFFFFFF})[0].scc == 0  # lt_u32


def test_s_cmp_u64_pair_boundary():
    s = {0: 5, 1: 1, 2: 5, 3: 2}  # low halves equal, high halves differ
    assert run(sopc(18, 0, 2), sgpr=s)[0].scc == 0  # eq_u64
    assert run(sopc(19, 0, 2), sgpr=s)[0].scc == 1  # lg_u64


def test_s_bitcmp():
    assert run(sopc(13, 1, IC(4)), sgpr={1: 0x10})[0].scc == 1  # bitcmp1 bit 4
    assert run(sopc(12, 1, IC(4)), sgpr={1: 0x10})[0].scc == 0  # bitcmp0 bit 4


# ---- program flow (ISA ch. 4) -----------------------------------------------------------
def _skip_mov(branch_op, **init):
    # branch +1 skips "s_mov_b32 s0, 1"
    w, _ = run(sopp(branch_op, 1), sop1(3, 0, IC(1)), **init)
    return w.sgpr[0]


def test_cbranch_execz_exec_zero_and_nonzero():
    assert _skip_mov(8, exec_=0) == 0  # taken: mov skipped
    assert _skip_mov(8, exec_=1 << 63) == 1  # not taken: mov executes


def test_cbranch_execnz_scc_vcc():
    assert _skip_mov(9, exec_=4) == 0
    assert _skip_mov(4, scc=0) == 0 and _skip_mov(4, scc=1) == 1  # scc0
    assert _skip_mov(5, scc=1) == 0 and _skip_mov(5, scc=0) == 1  # scc1
    assert _skip_mov(6, sgpr={VCC: 0, VCC + 1: 0}) == 0  # vccz taken
    assert _skip_mov(7, sgpr={VCC: 0, VCC + 1: 1}) == 0  # vccnz taken (high half only)


def _loop(n):
    return run(
        sop1(3, 1, IC(0)),
        sopc(0, 0, IC(0)),
        sopp(5, 4),
        sop2(0, 1, 1, IC(1)),
        sop2(1, 0, 0, IC(1)),
        sopc(1, 0, IC(0)),
        sopp(5, -4),
        sgpr={0: n},
    )


@pytest.mark.parametrize("n,steps", [(0, 4), (1, 8), (4, 20)])
def test_counted_loop_iterations_and_trace_length(n, steps):
    w, res = _loop(n)
    assert w.sgpr[1] == n and res["steps"] == steps


def test_step_limit_reported():
    w = M.Wave()
    res = M.Machine(w, code(sopp(2, -1))).run(50)  # s_branch to itself
    assert res["status"] == "step-limit" and res["steps"] == 50


# ---- unsupported is never skipped ----------------------------------------------------------
def test_unsupported_stops_and_reports_pc():
    w = M.Wave()
    w.exec = M.M64
    prog = code(sop1(3, 0, IC(1)), vop2(3, 0, IC(0), 1), sop1(3, 2, IC(1)), END)  # v_add_f32
    res = M.Machine(w, prog).run()
    assert res["status"] == "unsupported:v_add_f32" and res["pc"] == 4
    assert w.sgpr[0] == 1 and w.sgpr[2] == 0  # the later mov did not run


def test_unmapped_scalar_load_faults():
    w = M.Wave()
    res = M.Machine(w, code(smem(0, 2, 4, 0), END)).run()
    assert res["status"].startswith("fault:")


# ---- SMEM ---------------------------------------------------------------------------------
def test_s_load_dwordx2_and_offset():
    base = 0x1000
    data = struct.pack("<4I", 1, 2, 0xDEADBEEF, 0xCAFEF00D)
    w, _ = run(
        smem(1, 2, 4, 8),
        sgpr={2: base, 3: 0},
        mem={base: data},
    )
    assert (w.sgpr[4], w.sgpr[5]) == (0xDEADBEEF, 0xCAFEF00D)


# ---- VALU: EXEC masking (ISA ch. 6) -----------------------------------------------------
def test_v_mov_exec_zero_single_lane_all():
    w, _ = run(vop1(1, 0, IC(7)), exec_=0)
    assert all(x == 0 for x in w.vgpr[0])
    w, _ = run(vop1(1, 0, IC(7)), exec_=1 << 5)
    assert [i for i, x in enumerate(w.vgpr[0]) if x] == [5]
    w, _ = run(vop1(1, 0, IC(7)))
    assert all(x == 7 for x in w.vgpr[0])


def test_v_mov_literal():
    w, _ = run(vop1(1, 3, LIT, 0x12345678))
    assert w.vgpr[3][63] == 0x12345678


def test_v_add_nc_u32_masked_lanes():
    vg = {(0, i): i for i in range(64)} | {(1, i): 10 for i in range(64)}
    w, _ = run(vop2(37, 2, V(0), 1), vgpr=vg, exec_=0b101)
    assert w.vgpr[2][0] == 10 and w.vgpr[2][2] == 12 and w.vgpr[2][1] == 0


def test_v_lshlrev_shift_amount_is_src0():
    w, _ = run(vop2(26, 1, IC(4), 0), vgpr={(0, 0): 1})  # v1 = v0 << 4
    assert w.vgpr[1][0] == 16


def test_v_cndmask_vcc_pair_wave64():
    vg = {(0, i): 100 + i for i in range(64)} | {(1, i): 200 + i for i in range(64)}
    w, _ = run(
        vop2(1, 2, V(0), 1),
        vgpr=vg,
        sgpr={VCC: 0b10, VCC + 1: 0b10},  # lanes 1 and 33 select src1
    )
    assert w.vgpr[2][0] == 100 and w.vgpr[2][1] == 201
    assert w.vgpr[2][33] == 233 and w.vgpr[2][32] == 132


def test_v_cndmask_vop3_sgpr_pair():
    vg = {(0, 0): 5, (1, 0): 9, (0, 1): 6, (1, 1): 8}
    w, _ = run(vop3(257, 2, V(0), V(1), 4), vgpr=vg, sgpr={4: 0b01, 5: 0})
    assert w.vgpr[2][0] == 9 and w.vgpr[2][1] == 6


# ---- VALU compares ---------------------------------------------------------------------
def test_v_cmp_eq_u32_writes_vcc_and_respects_exec():
    vg = {(0, i): i for i in range(64)} | {(1, i): 5 for i in range(64)}
    w, _ = run(vopc(194, V(0), 1), vgpr=vg)
    assert w.vcc == 1 << 5
    w, _ = run(vopc(194, V(0), 1), vgpr=vg, exec_=0)
    assert w.vcc == 0
    w, _ = run(vopc(194, V(0), 1), vgpr=vg, exec_=M64 & ~(1 << 5))
    assert w.vcc == 0  # inactive lanes write 0


def test_v_cmpx_gt_u32_writes_exec():
    vg = {(0, i): i for i in range(64)} | {(1, i): 3 for i in range(64)}
    w, _ = run(vopc(212, V(0), 1), vgpr=vg)  # lanes with v0 > 3
    assert w.exec == M64 & ~0xF


def test_v_cmp_lt_i64_vop3_pair_destination():
    vg = {(0, 0): 0xFFFFFFFF, (1, 0): 0xFFFFFFFF, (0, 1): 1, (1, 1): 0, (2, 0): 0, (3, 0): 0}
    vg |= {(2, 1): 0, (3, 1): 0}
    w, _ = run(vop3(161, 4, V(0), V(2)), vgpr=vg)  # s[4:5] = v[0:1] < v[2:3] (signed)
    assert pair(w, 4) == 1  # lane 0: -1 < 0; lane 1: 1 < 0 false; others 0 < 0 false


# ---- lane ops (ISA ch. 6.6, VOP3 readlane / writelane / mbcnt) ----------------------------
def test_v_readfirstlane_follows_exec():
    vg = {(0, 3): 33, (0, 4): 44, (0, 0): 99}
    w, _ = run(vop1(2, 7, V(0)), vgpr=vg, exec_=0b11000)
    assert w.sgpr[7] == 33
    w, _ = run(vop1(2, 7, V(0)), vgpr=vg, exec_=0)  # EXEC = 0 reads lane 0
    assert w.sgpr[7] == 99


def test_v_readlane_lane_over_31_and_wave32_wrap():
    w, _ = run(vop3(864, 0, V(0), 1), vgpr={(0, 40): 0xABC, (0, 8): 0x888}, sgpr={1: 40})
    assert w.sgpr[0] == 0xABC
    w, _ = run(  # CONFIDENCE: MED - wave32 lane index uses its low 5 bits
        vop3(864, 0, V(0), 1), vgpr={(0, 8): 0x888}, sgpr={1: 40}, wave=32
    )
    assert w.sgpr[0] == 0x888


def test_v_readlane_ignores_exec():
    w, _ = run(vop3(864, 0, V(0), 1), vgpr={(0, 9): 0x77}, sgpr={1: 9}, exec_=0)
    assert w.sgpr[0] == 0x77


def test_v_writelane_leaves_other_lanes_and_ignores_exec():
    vg = {(0, i): 0x55 for i in range(64)}
    w, _ = run(vop3(865, 0, 2, 1), vgpr=vg, sgpr={1: 3, 2: 0x99}, exec_=0)
    assert w.vgpr[0][3] == 0x99
    assert [i for i, x in enumerate(w.vgpr[0]) if x != 0x55] == [3]


def test_v_mbcnt_lane_id_idiom():
    # v_mbcnt_lo(-1, 0); v_mbcnt_hi(-1, v0) = lane id for every lane (ISA mbcnt description)
    w, _ = run(vop3(869, 0, NEG(1), IC(0)), vop3(870, 0, NEG(1), V(0)))
    assert w.vgpr[0] == list(range(64))


def test_v_mbcnt_respects_source_mask():
    # src0 = 0b1010 (lanes 1,3): lane 5 sees two set bits below it; lane 2 sees one.
    w, _ = run(vop3(869, 0, 1, IC(0)), sgpr={1: 0b1010})
    assert w.vgpr[0][5] == 2 and w.vgpr[0][2] == 1 and w.vgpr[0][0] == 0


# ---- VALU carry chain -------------------------------------------------------------------
def test_v_add_co_u32_then_ci_64bit_add_per_lane():
    # lane 0: 0x1_FFFFFFFF + 1 = 0x2_00000000; lane 1: 0x0_00000001 + 1 = 2
    vg = {(0, 0): 0xFFFFFFFF, (1, 0): 1, (0, 1): 1, (1, 1): 0, (2, 0): 1, (3, 0): 0}
    vg |= {(2, 1): 1, (3, 1): 0}  # v0 = a_lo, v1 = a_hi, v2 = b_lo, v3 = b_hi
    w, _ = run(
        vop3(783, 4, V(0), V(2), sdst=VCC),  # v4 = v0 + v2, VCC = carry
        vop2(40, 5, V(1), 3),  # v5 = v1 + v3 + VCC
        vgpr=vg,
    )
    assert (w.vgpr[4][0], w.vgpr[5][0]) == (0, 2)  # 0x1_FFFFFFFF + 1
    assert (w.vgpr[4][1], w.vgpr[5][1]) == (2, 0)  # 1 + 1
    assert w.vcc == 0  # last carry-out (the high-half add) is clear in both lanes


# ---- the Kena spill/restore shape (#4714 / #4749 family), computed on paper ------------------
def test_kena_exec_spill_through_writelane_readlane():
    """EXEC saved to s[100:101], spilled to lanes 1 and 0 of v20, s[100:101] reused for an SMEM
    load, then restored with v_readlane and s_mov_b64 exec, s[100:101].

    Paper result: EXEC = 0x00FF00AA_12345678 (s100 = low half from lane 0, s101 = high half
    from lane 1); v20 lane 0 = 0x12345678, lane 1 = 0x00FF00AA, all other lanes 0.
    """
    orig = 0x00FF00AA_12345678
    base = 0x4000
    w, res = run(
        sop1(4, 100, EXEC),  # s_mov_b64 s[100:101], exec
        sop1(4, EXEC, IC(0)),  # s_mov_b64 exec, 0 (writelane/readlane must ignore this)
        vop3(865, 20, 101, IC(1)),  # v_writelane_b32 v20, s101, 1
        vop3(865, 20, 100, IC(0)),  # v_writelane_b32 v20, s100, 0
        smem(1, 2, 100, 0),  # s_load_dwordx2 s[100:101], s[2:3], 0   (SGPR reused)
        vop3(864, 100, V(20), IC(0)),  # v_readlane_b32 s100, v20, 0
        vop3(864, 101, V(20), IC(1)),  # v_readlane_b32 s101, v20, 1
        sop1(4, EXEC, 100),  # s_mov_b64 exec, s[100:101]
        exec_=orig,
        sgpr={2: base, 3: 0},
        mem={base: struct.pack("<2I", 0xDEADBEEF, 0xCAFEF00D)},
    )
    assert w.exec == orig
    assert (w.sgpr[100], w.sgpr[101]) == (0x12345678, 0x00FF00AA)
    assert w.vgpr[20][0] == 0x12345678 and w.vgpr[20][1] == 0x00FF00AA
    assert sum(1 for x in w.vgpr[20] if x) == 2


# ---- decode cross-check against LLVM (third reference) -----------------------------------
CONTROL = bytes(
    [0xFF, 0x03, 0xEB, 0xBE, 0x03, 0x00, 0x00, 0x00, 0x80, 0x02, 0x00, 0x7E, 0x00, 0x01, 0x00, 0x5E]
    + [
        0x00,
        0x00,
        0x80,
        0xBF,
        0x0F,
        0x1C,
        0x00,
        0xF8,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x81,
        0xBF,
    ]
)


def test_control_decodes_to_six_instructions():
    insns, err = D.decode_program(CONTROL)
    assert err is None
    assert [(i.pc, i.size, i.mnemonic) for i in insns] == [
        (0, 8, "s_mov_b32"),
        (8, 4, "v_mov_b32"),
        (12, 4, "v_cvt_pkrtz_f16_f32"),
        (16, 4, "s_nop"),
        (20, 8, "exp"),
        (28, 4, "s_endpgm"),
    ]


needs_llvm = pytest.mark.skipif(shutil.which("wsl") is None, reason="LLVM only via WSL Ubuntu")


@needs_llvm
def test_llvm_gfx1030_agrees_on_control():
    llvm = L.run_llvm(CONTROL, "gfx1030")
    assert len(llvm.insns) == 6 and llvm.invalid == 0
    res = L.compare(CONTROL, llvm)
    assert res["disagreements"] == [] and res["mnemonic_checked"] == 6


@needs_llvm
def test_llvm_gfx900_reports_invalid_on_control():
    llvm = L.run_llvm(CONTROL, "gfx900")
    assert llvm.invalid > 0
    assert L.compare(CONTROL, llvm)["disagreements"]


# ---- Mutation proofs ---------------------------------------------------------------------
# Each semantic family has a proof that goes red when deliberately broken.
def test_mutation_proof_andn2_swapped_operands(monkeypatch):
    """Mutating ANDN2 to swap operands (e.g. ~S0 & S1 instead of S0 & ~S1) fails."""
    # Under correct semantics: S0=0xFFFFFFFF_FFFFFFFF, S1=0xFFFFFFFF_00000000 -> S0 & ~S1 = 0x00000000_FFFFFFFF
    # Under swapped operands: ~S0 & S1 = 0
    w, _ = run(
        sop2(21, 4, 0, 2),
        sgpr={0: 0xFFFFFFFF, 1: 0xFFFFFFFF, 2: 0, 3: 0xFFFFFFFF},
    )
    assert pair(w, 4) == 0x00000000_FFFFFFFF

    # Break it by swapping operands:
    monkeypatch.setitem(M.BIN_LOGIC, "ANDN2", lambda a, b: ~a & b)
    w_mut, _ = run(
        sop2(21, 4, 0, 2),
        sgpr={0: 0xFFFFFFFF, 1: 0xFFFFFFFF, 2: 0, 3: 0xFFFFFFFF},
    )
    assert pair(w_mut, 4) != 0x00000000_FFFFFFFF  # Goes red under the mutant!


def test_mutation_proof_b64_drops_high_half(monkeypatch):
    """Mutating write_s64 to drop the high half fails the b64 ops test."""

    def broken_write_s64(self, idx, v):
        self.write_s32(idx, v & M.M32)
        # Drop high half write: self.write_s32(idx + 1, 0)
        self.write_s32(idx + 1, 0)

    monkeypatch.setattr(M.Wave, "write_s64", broken_write_s64)
    with pytest.raises(AssertionError):
        test_s_and_b64_pair()


def test_mutation_proof_writelane_respects_exec(monkeypatch):
    """Mutating v_writelane to mask by EXEC fails when EXEC is 0."""
    orig_valu = M.Machine.valu

    def broken_valu(self, insn):
        if insn.name == "V_WRITELANE_B32":
            # Mutant: incorrectly apply EXEC mask to writelane
            lane = self.w.src_scalar(insn.fields["src1"], 32, self.literal(insn)) & (
                self.w.size - 1
            )
            if (self.w.exec >> lane) & 1:
                self.w.vgpr[insn.fields["vdst"]][lane] = self.w.src_scalar(
                    insn.fields["src0"], 32, self.literal(insn)
                )
            return
        return orig_valu(self, insn)

    monkeypatch.setattr(M.Machine, "valu", broken_valu)
    with pytest.raises(AssertionError):
        test_v_writelane_leaves_other_lanes_and_ignores_exec()
