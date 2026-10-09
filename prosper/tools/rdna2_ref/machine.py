"""Wave state and instruction semantics for an independent RDNA2 scalar/EXEC-subset interpreter.

Semantics are written from the AMD "RDNA 2" Instruction Set Architecture Reference Guide
(doc 70648) only: ch. 3 (scalar registers, EXEC, VCC, SCC, inline constants), ch. 4 (program
flow), ch. 5 (scalar ALU), ch. 6 (vector ALU), ch. 7 (scalar memory), ch. 12 (per-instruction
pseudocode) and ch. 13 (encodings). Nothing is taken from prosper's recompiler.

Policy: an instruction outside the modelled subset raises ``Unsupported`` and stops the
program. It is never skipped. Every semantic with residual doubt carries ``CONFIDENCE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import decode as D

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1

SGPR_VCC_LO, SGPR_VCC_HI = 106, 107
SGPR_M0 = 124
SGPR_NULL = 125
SGPR_EXEC_LO, SGPR_EXEC_HI = 126, 127


class Unsupported(Exception):
    """The instruction is outside the modelled subset (never skipped)."""

    def __init__(self, mnemonic: str, why: str = ""):
        super().__init__(mnemonic)
        self.mnemonic = mnemonic
        self.why = why


class Fault(Exception):
    """A modelled hardware fault, e.g. a scalar load from unmapped memory."""


def s32(x: int) -> int:
    """Reinterpret the low 32 bits as signed."""
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def s64(x: int) -> int:
    """Reinterpret the low 64 bits as signed."""
    x &= M64
    return x - (1 << 64) if x >> 63 else x


def sext(x: int, bits: int) -> int:
    """Sign-extend the low ``bits`` bits of ``x``."""
    x &= (1 << bits) - 1
    return x - (1 << bits) if x >> (bits - 1) else x


class Memory:
    """Sparse byte-addressable memory image for scalar loads."""

    def __init__(self) -> None:
        self.bytes: dict[int, int] = {}

    def write(self, addr: int, data: bytes) -> None:
        """Write ``data`` at ``addr``."""
        for i, b in enumerate(data):
            self.bytes[addr + i] = b

    def read(self, addr: int, n: int) -> bytes:
        """Read ``n`` bytes; any unwritten byte is a Fault (never fabricated)."""
        out = bytearray()
        for i in range(n):
            if addr + i not in self.bytes:
                raise Fault(f"unmapped read at {addr + i:#x}")
            out.append(self.bytes[addr + i])
        return bytes(out)


# Inline float constants as 32-bit patterns (ISA ch. 3 source-operand table, codes 240-248).
FLOAT_CONSTS = {
    240: 0x3F000000,  # 0.5
    241: 0xBF000000,  # -0.5
    242: 0x3F800000,  # 1.0
    243: 0xBF800000,  # -1.0
    244: 0x40000000,  # 2.0
    245: 0xC0000000,  # -2.0
    246: 0x40800000,  # 4.0
    247: 0xC0800000,  # -4.0
    248: 0x3E22F983,  # 1/(2*PI)
}


@dataclass
class Wave:
    """Architectural state of one wave."""

    size: int = 64
    mem: Memory = field(default_factory=Memory)
    sgpr: list = field(default_factory=lambda: [0] * 128)
    vgpr: list = field(default_factory=list)
    scc: int = 0
    pc: int = 0
    halted: bool = False

    def __post_init__(self) -> None:
        if self.size not in (32, 64):
            raise ValueError("wave size must be 32 or 64")
        self.vgpr = [[0] * self.size for _ in range(256)]

    # -- special registers ---------------------------------------------------
    @property
    def lane_mask(self) -> int:
        """All-ones mask of the wave width."""
        return (1 << self.size) - 1

    @property
    def exec(self) -> int:
        """EXEC as a lane mask (wave32 uses EXEC_LO only)."""
        return (self.sgpr[SGPR_EXEC_LO] | (self.sgpr[SGPR_EXEC_HI] << 32)) & self.lane_mask

    @exec.setter
    def exec(self, v: int) -> None:
        self.sgpr[SGPR_EXEC_LO] = v & M32
        if self.size == 64:
            self.sgpr[SGPR_EXEC_HI] = (v >> 32) & M32

    @property
    def vcc(self) -> int:
        """VCC as a lane mask (wave32 uses VCC_LO only)."""
        return (self.sgpr[SGPR_VCC_LO] | (self.sgpr[SGPR_VCC_HI] << 32)) & self.lane_mask

    @vcc.setter
    def vcc(self, v: int) -> None:
        self.sgpr[SGPR_VCC_LO] = v & M32
        if self.size == 64:
            self.sgpr[SGPR_VCC_HI] = (v >> 32) & M32

    # -- scalar register file ------------------------------------------------
    def read_s32(self, idx: int) -> int:
        """Read one 32-bit scalar register (0-127 range of the SSRC encoding)."""
        if idx == SGPR_NULL:
            return 0
        if 0 <= idx <= 127:
            return self.sgpr[idx]
        raise Unsupported("sgpr-index", str(idx))

    def write_s32(self, idx: int, v: int) -> None:
        """Write one 32-bit scalar register; NULL discards the write."""
        if idx == SGPR_NULL:
            return
        if 0 <= idx <= 127:
            self.sgpr[idx] = v & M32
            return
        raise Unsupported("sgpr-dst", str(idx))

    def read_s64(self, idx: int) -> int:
        """Read the pair s[idx:idx+1] (little-endian halves)."""
        return self.read_s32(idx) | (self.read_s32(idx + 1) << 32)

    def write_s64(self, idx: int, v: int) -> None:
        """Write the pair s[idx:idx+1]; both halves are written."""
        self.write_s32(idx, v & M32)
        self.write_s32(idx + 1, (v >> 32) & M32)

    # -- source operands -----------------------------------------------------
    def src_scalar(self, code: int, bits: int, lit: int | None) -> int:
        """Value of an SSRC-style operand of width 32 or 64 (section 3, operand table)."""
        if code <= 127:
            return self.read_s64(code) if bits == 64 else self.read_s32(code)
        if code == 128:
            return 0
        if 129 <= code <= 192:
            return code - 128
        if 193 <= code <= 208:
            v = -(code - 192)
            return (v & M64) if bits == 64 else (v & M32)
        if code in FLOAT_CONSTS:
            # CONFIDENCE: MED (64-bit integer use of a float inline constant is unusual)
            return FLOAT_CONSTS[code]
        if code == 251:  # VCCZ
            return int(self.vcc == 0)
        if code == 252:  # EXECZ
            return int(self.exec == 0)
        if code == 253:
            return self.scc
        if code == 255:
            if lit is None:
                raise Unsupported("literal-missing")
            # CONFIDENCE: MED for 64-bit ops: the 32-bit literal is zero-extended here.
            return lit & M32
        raise Unsupported("operand", f"code {code}")

    def src_lane(self, code: int, lane: int, lit: int | None) -> int:
        """32-bit VALU source operand for ``lane`` (codes 256-511 are VGPRs)."""
        if code >= 256:
            return self.vgpr[code - 256][lane]
        return self.src_scalar(code, 32, lit)

    def src_lane64(self, code: int, lane: int, lit: int | None) -> int:
        """64-bit VALU source operand: VGPR pair v[n:n+1] or SGPR pair / inline constant."""
        if code >= 256:
            n = code - 256
            return self.vgpr[n][lane] | (self.vgpr[n + 1][lane] << 32)
        return self.src_scalar(code, 64, lit)

    def write_lanemask(self, sdst: int, mask: int) -> None:
        """Write a lane mask to an SGPR pair (wave64) or one SGPR (wave32)."""
        if self.size == 64:
            self.write_s64(sdst, mask & M64)
        else:
            self.write_s32(sdst, mask & M32)

    def read_lanemask(self, code: int) -> int:
        """Read a lane-mask operand (carry-in / cndmask selector) from an SGPR pair."""
        if code > 127:
            if code == 128:
                return 0
            if code == 193:  # inline -1: all lanes
                return self.lane_mask
            raise Unsupported("lanemask-operand", str(code))
        v = self.read_s64(code) if self.size == 64 else self.read_s32(code)
        return v & self.lane_mask


# ---------------------------------------------------------------------------
# Scalar ALU semantics (ch. 5, instruction pseudocode in ch. 12.1-12.5)
# ---------------------------------------------------------------------------

BIN_LOGIC = {
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "XOR": lambda a, b: a ^ b,
    "ANDN2": lambda a, b: a & ~b,  # S0 & ~S1
    "ORN2": lambda a, b: a | ~b,  # S0 | ~S1
    "NAND": lambda a, b: ~(a & b),
    "NOR": lambda a, b: ~(a | b),
    "XNOR": lambda a, b: ~(a ^ b),
}

# *_SAVEEXEC: D = EXEC; EXEC = f(S0, EXEC); SCC = (EXEC != 0). ANDN1 negates S0, ANDN2
# negates EXEC (ch. 12 SOP1 pseudocode).
SAVEEXEC = {
    "AND": lambda s, e: s & e,
    "OR": lambda s, e: s | e,
    "XOR": lambda s, e: s ^ e,
    "ANDN2": lambda s, e: s & ~e,
    "ORN2": lambda s, e: s | ~e,
    "NAND": lambda s, e: ~(s & e),
    "NOR": lambda s, e: ~(s | e),
    "XNOR": lambda s, e: ~(s ^ e),
    "ANDN1": lambda s, e: ~s & e,
    "ORN1": lambda s, e: ~s | e,
}

CMP_I = {
    "EQ": lambda a, b: a == b,
    "LG": lambda a, b: a != b,
    "GT": lambda a, b: a > b,
    "GE": lambda a, b: a >= b,
    "LT": lambda a, b: a < b,
    "LE": lambda a, b: a <= b,
}

# VOPC comparison codes: operation index within a family (Table 79).
VCMP = {
    0: lambda a, b: False,  # F
    1: lambda a, b: a < b,
    2: lambda a, b: a == b,
    3: lambda a, b: a <= b,
    4: lambda a, b: a > b,
    5: lambda a, b: a != b,
    6: lambda a, b: a >= b,
    7: lambda a, b: True,  # T
}

# Lane-wise integer ops: (S0, S1, S2) -> result (low 32 bits kept by the caller).
LANE_FN = {
    "V_NOP": lambda a, b, c: 0,
    "V_MOV_B32": lambda a, b, c: a,
    "V_NOT_B32": lambda a, b, c: ~a,
    "V_BFREV_B32": lambda a, b, c: int(format(a, "032b")[::-1], 2),
    "V_CNDMASK_B32": lambda a, b, c: a,  # selector handled by the caller
    "V_ADD_NC_U32": lambda a, b, c: a + b,
    "V_SUB_NC_U32": lambda a, b, c: a - b,
    "V_SUBREV_NC_U32": lambda a, b, c: b - a,
    "V_AND_B32": lambda a, b, c: a & b,
    "V_OR_B32": lambda a, b, c: a | b,
    "V_XOR_B32": lambda a, b, c: a ^ b,
    "V_XNOR_B32": lambda a, b, c: ~(a ^ b),
    # REV shifts: shift amount is S0, value is S1 (ch. 12 VOP2).
    "V_LSHLREV_B32": lambda a, b, c: b << (a & 31),
    "V_LSHRREV_B32": lambda a, b, c: b >> (a & 31),
    "V_ASHRREV_I32": lambda a, b, c: s32(b) >> (a & 31),
    "V_MIN_U32": lambda a, b, c: min(a, b),
    "V_MAX_U32": lambda a, b, c: max(a, b),
    "V_MIN_I32": lambda a, b, c: min(s32(a), s32(b)),
    "V_MAX_I32": lambda a, b, c: max(s32(a), s32(b)),
    "V_MUL_LO_U32": lambda a, b, c: a * b,
    "V_MUL_HI_U32": lambda a, b, c: (a * b) >> 32,
    "V_MUL_U32_U24": lambda a, b, c: (a & 0xFFFFFF) * (b & 0xFFFFFF),
    "V_MUL_I32_I24": lambda a, b, c: sext(a, 24) * sext(b, 24),
    "V_ADD3_U32": lambda a, b, c: a + b + c,
    "V_OR3_B32": lambda a, b, c: a | b | c,
    "V_XOR3_B32": lambda a, b, c: a ^ b ^ c,
    "V_AND_OR_B32": lambda a, b, c: (a & b) | c,
    "V_LSHL_OR_B32": lambda a, b, c: (a << (b & 31)) | c,
    "V_LSHL_ADD_U32": lambda a, b, c: (a << (b & 31)) + c,
    "V_ADD_LSHL_U32": lambda a, b, c: (a + b) << (c & 31),
    # Bit-field ops (ch. 12 VOP3). CONFIDENCE: MED: pseudocode read from the instruction text.
    "V_BFE_U32": lambda a, b, c: (a >> (b & 31)) & ((1 << (c & 31)) - 1),
    "V_BFE_I32": lambda a, b, c: (
        sext((a >> (b & 31)) & ((1 << (c & 31)) - 1), c & 31) if c & 31 else 0
    ),
    "V_BFI_B32": lambda a, b, c: (a & b) | (~a & c),
    "V_BCNT_U32_B32": lambda a, b, c: bin(a).count("1") + b,
    "V_BFM_B32": lambda a, b, c: ((1 << (a & 31)) - 1) << (b & 31),
    "V_ALIGNBIT_B32": lambda a, b, c: ((a << 32) | b) >> (c & 31),
    "V_MAD_U32_U24": lambda a, b, c: (a & 0xFFFFFF) * (b & 0xFFFFFF) + c,
    "V_MAD_I32_I24": lambda a, b, c: sext(a, 24) * sext(b, 24) + s32(c),
}

CARRY_OPS = {
    # name: (add?, reversed operands?)
    "V_ADD_CO_CI_U32": (True, False),
    "V_SUB_CO_CI_U32": (False, False),
    "V_SUBREV_CO_CI_U32": (False, True),
    "V_ADD_CO_U32": (True, False),
    "V_SUB_CO_U32": (False, False),
    "V_SUBREV_CO_U32": (False, True),
}

SOPP_NOPS = {
    "S_NOP",
    "S_WAITCNT",
    "S_SLEEP",
    "S_SETPRIO",
    "S_ICACHE_INV",
    "S_INST_PREFETCH",
    "S_CLAUSE",
    "S_WAITCNT_DEPCTR",
}

SMEM_DW = {
    "S_LOAD_DWORD": 1,
    "S_LOAD_DWORDX2": 2,
    "S_LOAD_DWORDX4": 4,
    "S_LOAD_DWORDX8": 8,
    "S_LOAD_DWORDX16": 16,
    "S_BUFFER_LOAD_DWORD": 1,
    "S_BUFFER_LOAD_DWORDX2": 2,
    "S_BUFFER_LOAD_DWORDX4": 4,
    "S_BUFFER_LOAD_DWORDX8": 8,
    "S_BUFFER_LOAD_DWORDX16": 16,
}


def is_supported(insn: D.Insn) -> bool:
    """Static check: is this instruction's mnemonic in the modelled subset?"""
    n = insn.name
    if n is None:
        return False
    f = insn.fmt
    if f == "SOPP":
        return (
            n in SOPP_NOPS
            or n == "S_ENDPGM"
            or n
            in (
                "S_BRANCH",
                "S_CBRANCH_SCC0",
                "S_CBRANCH_SCC1",
                "S_CBRANCH_VCCZ",
                "S_CBRANCH_VCCNZ",
                "S_CBRANCH_EXECZ",
                "S_CBRANCH_EXECNZ",
            )
        )
    return n in _SUPPORTED_NAMES


def _scalar_names() -> set:
    names = set()
    for n in D.SOP2.values():
        if n[2:].rsplit("_", 1)[0] in BIN_LOGIC or n in (
            "S_ADD_U32",
            "S_SUB_U32",
            "S_ADD_I32",
            "S_SUB_I32",
            "S_ADDC_U32",
            "S_SUBB_U32",
            "S_MIN_I32",
            "S_MIN_U32",
            "S_MAX_I32",
            "S_MAX_U32",
            "S_CSELECT_B32",
            "S_CSELECT_B64",
            "S_LSHL_B32",
            "S_LSHR_B32",
            "S_ASHR_I32",
            "S_LSHL_B64",
            "S_LSHR_B64",
            "S_ASHR_I64",
            "S_BFM_B32",
            "S_BFM_B64",
            "S_MUL_I32",
            "S_BFE_U32",
            "S_BFE_I32",
            "S_BFE_U64",
            "S_BFE_I64",
            "S_ABSDIFF_I32",
            "S_LSHL1_ADD_U32",
            "S_LSHL2_ADD_U32",
            "S_LSHL3_ADD_U32",
            "S_LSHL4_ADD_U32",
            "S_PACK_LL_B32_B16",
            "S_PACK_LH_B32_B16",
            "S_PACK_HH_B32_B16",
            "S_MUL_HI_U32",
            "S_MUL_HI_I32",
        ):
            names.add(n)
    for n in D.SOP1.values():
        if n.endswith(("_SAVEEXEC_B64", "_SAVEEXEC_B32")):
            if n[2 : n.rindex("_SAVEEXEC")] in SAVEEXEC:
                names.add(n)
        elif n in (
            "S_MOV_B32",
            "S_MOV_B64",
            "S_CMOV_B32",
            "S_CMOV_B64",
            "S_NOT_B32",
            "S_NOT_B64",
            "S_BREV_B32",
            "S_BREV_B64",
            "S_BCNT0_I32_B32",
            "S_BCNT0_I32_B64",
            "S_BCNT1_I32_B32",
            "S_BCNT1_I32_B64",
            "S_FF0_I32_B32",
            "S_FF0_I32_B64",
            "S_FF1_I32_B32",
            "S_FF1_I32_B64",
            "S_SEXT_I32_I8",
            "S_SEXT_I32_I16",
            "S_BITSET0_B32",
            "S_BITSET1_B32",
            "S_BITSET0_B64",
            "S_BITSET1_B64",
            "S_ABS_I32",
            "S_WQM_B32",
            "S_WQM_B64",
        ):
            names.add(n)
    for n in D.SOPC.values():
        p = n.split("_")
        if n in ("S_CMP_EQ_U64", "S_CMP_LG_U64") or n.startswith("S_BITCMP"):
            names.add(n)
        elif len(p) == 4 and p[1] == "CMP" and p[2] in CMP_I and p[3] in ("I32", "U32"):
            names.add(n)
    for n in D.SOPK.values():
        if n in (
            "S_MOVK_I32",
            "S_CMOVK_I32",
            "S_ADDK_I32",
            "S_MULK_I32",
            "S_VERSION",
            "S_WAITCNT_VSCNT",
            "S_WAITCNT_VMCNT",
            "S_WAITCNT_EXPCNT",
            "S_WAITCNT_LGKMCNT",
        ) or (n.startswith("S_CMPK_") and n.split("_")[2] in CMP_I):
            names.add(n)
    names.update(SMEM_DW)
    return names


def _vector_names() -> set:
    names = set(LANE_FN) | set(CARRY_OPS)
    names |= {
        "V_READFIRSTLANE_B32",
        "V_READLANE_B32",
        "V_WRITELANE_B32",
        "V_MBCNT_LO_U32_B32",
        "V_MBCNT_HI_U32_B32",
    }
    for table in (D.VOPC,):
        for n in table.values():
            if n.split("_")[-1] in ("I32", "U32", "I64", "U64"):
                names.add(n)
    return names


_SUPPORTED_NAMES = _scalar_names() | _vector_names()


class Machine:
    """Executes decoded RDNA2 instructions against a ``Wave``."""

    def __init__(self, wave: Wave, code: bytes):
        self.w = wave
        self.code = code
        self._cache: dict[int, D.Insn] = {}
        self.trace_len = 0

    # -- fetch -----------------------------------------------------------
    def fetch(self, pc: int) -> D.Insn:
        """Decode (and cache) the instruction at byte offset ``pc``."""
        i = self._cache.get(pc)
        if i is None:
            try:
                i = D.decode_one(self.code, pc)
            except D.DecodeError as exc:
                raise Unsupported("decode-error", str(exc)) from exc
            self._cache[pc] = i
        return i

    @staticmethod
    def literal(insn: D.Insn) -> int | None:
        """The trailing 32-bit literal dword of ``insn``, if the encoding carries one."""
        base = 8 if insn.fmt in ("VOP3", "VOP3B", "SMEM") else 4
        return insn.words[-1] if insn.size > base else None

    # -- run -------------------------------------------------------------
    def run(self, step_limit: int = 100_000) -> dict:
        """Run until s_endpgm, an unsupported instruction, a fault, or the step limit."""
        w = self.w
        steps = 0
        while not w.halted:
            if steps >= step_limit:
                return {"status": "step-limit", "steps": steps, "pc": w.pc}
            if w.pc >= len(self.code):
                return {"status": "fell-off-end", "steps": steps, "pc": w.pc}
            try:
                self.step(self.fetch(w.pc))
            except Unsupported as exc:
                return {
                    "status": f"unsupported:{exc.mnemonic}",
                    "steps": steps,
                    "pc": w.pc,
                    "why": exc.why,
                }
            except Fault as exc:
                return {"status": f"fault:{exc}", "steps": steps, "pc": w.pc}
            steps += 1
            self.trace_len = steps
        return {"status": "ended", "steps": steps, "pc": w.pc}

    # -- dispatch ----------------------------------------------------------
    def step(self, insn: D.Insn) -> None:
        """Execute one instruction and advance the PC."""
        if insn.name is None:
            raise Unsupported(insn.mnemonic)
        w = self.w
        next_pc = w.pc + insn.size
        fmt = insn.fmt
        if fmt == "SOP2":
            self.sop2(insn)
        elif fmt == "SOP1":
            self.sop1(insn)
        elif fmt == "SOPC":
            self.sopc(insn)
        elif fmt == "SOPK":
            self.sopk(insn)
        elif fmt == "SOPP":
            next_pc = self.sopp(insn, next_pc)
        elif fmt == "SMEM":
            self.smem(insn)
        elif fmt in ("VOP1", "VOP2", "VOPC", "VOP3", "VOP3B"):
            self.valu(insn)
        else:
            raise Unsupported(insn.mnemonic)
        if not w.halted:
            w.pc = next_pc

    def _ss(self, insn, key, bits):
        return self.w.src_scalar(insn.fields[key], bits, self.literal(insn))

    # -- SOP2 ----------------------------------------------------------------
    def sop2(self, insn: D.Insn) -> None:
        """SOP2 (section 12.1)."""
        w, n = self.w, insn.name
        sdst = insn.fields["sdst"]
        is64 = n.endswith(("_B64", "_U64", "_I64"))
        # Shift / BFE-64 / BFM forms take a 32-bit second operand even when D, S0 are 64-bit.
        if n in ("S_LSHL_B64", "S_LSHR_B64", "S_ASHR_I64", "S_BFE_U64", "S_BFE_I64"):
            a, b = self._ss(insn, "ssrc0", 64), self._ss(insn, "ssrc1", 32)
        elif n == "S_BFM_B64":
            a, b = self._ss(insn, "ssrc0", 32), self._ss(insn, "ssrc1", 32)
        else:
            bits = 64 if is64 else 32
            a, b = self._ss(insn, "ssrc0", bits), self._ss(insn, "ssrc1", bits)
        mask = M64 if is64 else M32

        def put(v):
            if is64:
                w.write_s64(sdst, v & M64)
            else:
                w.write_s32(sdst, v & M32)

        def nz(r):
            w.scc = int(r != 0)

        if n == "S_ADD_U32":
            r = a + b
            put(r)
            w.scc = int(r > M32)
        elif n == "S_SUB_U32":
            put(a - b)
            w.scc = int(b > a)
        elif n in ("S_ADD_I32", "S_SUB_I32"):
            r = s32(a) + s32(b) if n == "S_ADD_I32" else s32(a) - s32(b)
            put(r)
            w.scc = int(not -(1 << 31) <= r < (1 << 31))
        elif n == "S_ADDC_U32":
            r = a + b + w.scc
            put(r)
            w.scc = int(r > M32)
        elif n == "S_SUBB_U32":
            carry = w.scc
            put(a - b - carry)
            w.scc = int(b + carry > a)  # borrow out
        elif n in ("S_MIN_I32", "S_MAX_I32", "S_MIN_U32", "S_MAX_U32"):
            x, y = (s32(a), s32(b)) if n.endswith("I32") else (a, b)
            pick_a = (x <= y) if n.startswith("S_MIN") else (x >= y)
            put(a if pick_a else b)
            w.scc = int(pick_a)  # SCC = 1 when S0 was selected
        elif n in ("S_CSELECT_B32", "S_CSELECT_B64"):
            put(a if w.scc else b)  # no flag update
        elif n[2:].rsplit("_", 1)[0] in BIN_LOGIC and n[-3:] in ("B32", "B64"):
            r = BIN_LOGIC[n[2:].rsplit("_", 1)[0]](a, b) & mask
            put(r)
            nz(r)
        elif n == "S_LSHL_B32":
            r = (a << (b & 31)) & M32
            put(r)
            nz(r)
        elif n == "S_LSHR_B32":
            r = a >> (b & 31)
            put(r)
            nz(r)
        elif n == "S_ASHR_I32":
            r = (s32(a) >> (b & 31)) & M32
            put(r)
            nz(r)
        elif n == "S_LSHL_B64":
            r = (a << (b & 63)) & M64
            put(r)
            nz(r)
        elif n == "S_LSHR_B64":
            r = a >> (b & 63)
            put(r)
            nz(r)
        elif n == "S_ASHR_I64":
            r = (s64(a) >> (b & 63)) & M64
            put(r)
            nz(r)
        elif n == "S_BFM_B32":
            put(((1 << (a & 31)) - 1) << (b & 31))
        elif n == "S_BFM_B64":
            put((((1 << (a & 63)) - 1) << (b & 63)) & M64)
        elif n == "S_MUL_I32":
            put(s32(a) * s32(b))
        elif n in ("S_BFE_U32", "S_BFE_I32", "S_BFE_U64", "S_BFE_I64"):
            lim = 63 if n.endswith("64") else 31
            off, wid = b & lim, (b >> 16) & 0x7F
            r = (a >> off) & ((1 << wid) - 1)
            if "_I" in n and wid:
                r = sext(r, wid) & mask
            put(r)
            nz(r)
        elif n == "S_ABSDIFF_I32":
            r = abs(s32(a) - s32(b)) & M32
            put(r)
            nz(r)
        elif n in ("S_LSHL1_ADD_U32", "S_LSHL2_ADD_U32", "S_LSHL3_ADD_U32", "S_LSHL4_ADD_U32"):
            r = (a << int(n[6])) + b
            put(r)
            w.scc = int(r > M32)
        elif n == "S_PACK_LL_B32_B16":
            put((a & 0xFFFF) | ((b & 0xFFFF) << 16))
        elif n == "S_PACK_LH_B32_B16":
            put((a & 0xFFFF) | (b & 0xFFFF0000))
        elif n == "S_PACK_HH_B32_B16":
            put((a >> 16) | (b & 0xFFFF0000))
        elif n == "S_MUL_HI_U32":
            put((a * b) >> 32)
        elif n == "S_MUL_HI_I32":
            put((s32(a) * s32(b)) >> 32)
        else:
            raise Unsupported(insn.mnemonic)

    # -- SOP1 ----------------------------------------------------------------
    def sop1(self, insn: D.Insn) -> None:
        """SOP1 (section 12.2), including the *_SAVEEXEC family."""
        w, n = self.w, insn.name
        sdst = insn.fields["sdst"]
        if n.endswith(("_SAVEEXEC_B64", "_SAVEEXEC_B32")):
            fam = n[2 : n.rindex("_SAVEEXEC")]
            if fam not in SAVEEXEC:
                raise Unsupported(insn.mnemonic)
            wide = n.endswith("B64")
            if wide != (w.size == 64):
                raise Unsupported(insn.mnemonic, "width differs from wave size")
            s0 = self._ss(insn, "ssrc0", 64 if wide else 32)
            old = w.exec
            if wide:
                w.write_s64(sdst, old)
            else:
                w.write_s32(sdst, old)
            new = SAVEEXEC[fam](s0, old) & w.lane_mask
            w.exec = new
            w.scc = int(new != 0)
            return
        is64 = n.endswith(("_B64", "_I64"))
        width = 64 if is64 else 32
        mask = M64 if is64 else M32

        def put(v):
            if is64:
                w.write_s64(sdst, v & M64)
            else:
                w.write_s32(sdst, v & M32)

        if n in ("S_MOV_B32", "S_MOV_B64"):
            put(self._ss(insn, "ssrc0", width))
        elif n in ("S_CMOV_B32", "S_CMOV_B64"):
            v = self._ss(insn, "ssrc0", width)
            if w.scc:
                put(v)
        elif n in ("S_NOT_B32", "S_NOT_B64"):
            r = ~self._ss(insn, "ssrc0", width) & mask
            put(r)
            w.scc = int(r != 0)
        elif n in ("S_BREV_B32", "S_BREV_B64"):
            v = self._ss(insn, "ssrc0", width)
            put(int(format(v, f"0{width}b")[::-1], 2))
        elif n.startswith(("S_BCNT0_I32", "S_BCNT1_I32")):
            ones = bin(self._ss(insn, "ssrc0", width)).count("1")
            r = ones if "BCNT1" in n else width - ones
            w.write_s32(sdst, r)
            w.scc = int(r != 0)
        elif n.startswith(("S_FF0_I32", "S_FF1_I32")):
            v = self._ss(insn, "ssrc0", width)
            if "FF0" in n:
                v = ~v & mask
            w.write_s32(sdst, (v & -v).bit_length() - 1 if v else M32)
        elif n in ("S_SEXT_I32_I8", "S_SEXT_I32_I16"):
            v = self._ss(insn, "ssrc0", 32)
            w.write_s32(sdst, sext(v, 8 if n.endswith("I8") else 16))
        elif n.startswith(("S_BITSET0", "S_BITSET1")):
            # D[S0[4:0]] = 0/1 (read-modify-write of the destination)
            bit = self._ss(insn, "ssrc0", 32) & (63 if is64 else 31)
            cur = w.read_s64(sdst) if is64 else w.read_s32(sdst)
            cur = (cur | (1 << bit)) if "BITSET1" in n else (cur & ~(1 << bit))
            put(cur & mask)
        elif n == "S_ABS_I32":
            r = abs(s32(self._ss(insn, "ssrc0", 32))) & M32
            put(r)
            w.scc = int(r != 0)
        elif n in ("S_WQM_B32", "S_WQM_B64"):
            v = self._ss(insn, "ssrc0", width)
            r = 0
            for q in range(0, width, 4):
                if (v >> q) & 0xF:
                    r |= 0xF << q
            put(r)
            w.scc = int(r != 0)
        else:
            raise Unsupported(insn.mnemonic)

    # -- SOPC ----------------------------------------------------------------
    def sopc(self, insn: D.Insn) -> None:
        """SOPC compares set SCC (section 12.3)."""
        w, n = self.w, insn.name
        if n in ("S_CMP_EQ_U64", "S_CMP_LG_U64"):
            a, b = self._ss(insn, "ssrc0", 64), self._ss(insn, "ssrc1", 64)
            w.scc = int((a == b) if n.endswith("EQ_U64") else (a != b))
            return
        if n.startswith("S_BITCMP"):
            wide = n.endswith("B64")
            a = self._ss(insn, "ssrc0", 64 if wide else 32)
            bit = self._ss(insn, "ssrc1", 32) & (63 if wide else 31)
            w.scc = int(((a >> bit) & 1) == (0 if "BITCMP0" in n else 1))
            return
        p = n.split("_")  # S CMP EQ I32
        if len(p) == 4 and p[1] == "CMP" and p[2] in CMP_I and p[3] in ("I32", "U32"):
            a, b = self._ss(insn, "ssrc0", 32), self._ss(insn, "ssrc1", 32)
            if p[3] == "I32":
                a, b = s32(a), s32(b)
            w.scc = int(CMP_I[p[2]](a, b))
            return
        raise Unsupported(insn.mnemonic)

    # -- SOPK ----------------------------------------------------------------
    def sopk(self, insn: D.Insn) -> None:
        """SOPK 16-bit-immediate forms (section 12.4)."""
        w, n, f = self.w, insn.name, insn.fields
        imm = sext(f["simm16"], 16)
        sdst = f["sdst"]
        if n == "S_MOVK_I32":
            w.write_s32(sdst, imm)
        elif n == "S_CMOVK_I32":
            if w.scc:
                w.write_s32(sdst, imm)
        elif n.startswith("S_CMPK_") and n.split("_")[2] in CMP_I:
            c, t = n.split("_")[2:4]
            a = w.read_s32(sdst)
            if t == "I32":
                a, b = s32(a), imm
            else:
                b = f["simm16"]  # unsigned compares zero-extend SIMM16
            w.scc = int(CMP_I[c](a, b))
        elif n == "S_ADDK_I32":
            r = s32(w.read_s32(sdst)) + imm
            w.write_s32(sdst, r)
            w.scc = int(not -(1 << 31) <= r < (1 << 31))
        elif n == "S_MULK_I32":
            w.write_s32(sdst, s32(w.read_s32(sdst)) * imm)
        elif n in (
            "S_VERSION",
            "S_WAITCNT_VSCNT",
            "S_WAITCNT_VMCNT",
            "S_WAITCNT_EXPCNT",
            "S_WAITCNT_LGKMCNT",
        ):
            pass
        else:
            raise Unsupported(insn.mnemonic)

    # -- SOPP ----------------------------------------------------------------
    def sopp(self, insn: D.Insn, next_pc: int) -> int:
        """SOPP program flow (ch. 4): branch target = next_pc + sext(simm16) * 4."""
        w, n = self.w, insn.name
        target = next_pc + sext(insn.fields["simm16"], 16) * 4
        if n == "S_ENDPGM":
            w.halted = True
            return next_pc
        if n in SOPP_NOPS:
            return next_pc  # state-neutral for this model
        conds = {
            "S_BRANCH": True,
            "S_CBRANCH_SCC0": w.scc == 0,
            "S_CBRANCH_SCC1": w.scc == 1,
            "S_CBRANCH_VCCZ": w.vcc == 0,
            "S_CBRANCH_VCCNZ": w.vcc != 0,
            "S_CBRANCH_EXECZ": w.exec == 0,
            "S_CBRANCH_EXECNZ": w.exec != 0,
        }
        if n in conds:
            return target if conds[n] else next_pc
        raise Unsupported(insn.mnemonic)

    # -- SMEM ----------------------------------------------------------------
    def smem(self, insn: D.Insn) -> None:
        """Scalar memory loads against the memory image (ch. 7)."""
        w, n, f = self.w, insn.name, insn.fields
        if n not in SMEM_DW:
            raise Unsupported(insn.mnemonic)
        cnt = SMEM_DW[n]
        off = sext(f["offset"], 21)  # signed byte offset (S_LOAD / S_BUFFER_LOAD)
        if f["soffset"] != SGPR_NULL:
            off += w.read_s32(f["soffset"])  # CONFIDENCE: MED (SOFFSET add when not NULL)
        if n.startswith("S_BUFFER"):
            # base address = V#[47:0]. CONFIDENCE: LOW for bounds; no range check is applied.
            base = w.read_s64(f["sbase"]) & ((1 << 48) - 1)
        else:
            base = w.read_s64(f["sbase"])
        raw = w.mem.read((base + off) & M64, 4 * cnt)
        for i in range(cnt):
            w.write_s32(f["sdata"] + i, int.from_bytes(raw[4 * i : 4 * i + 4], "little"))

    # -- VALU ----------------------------------------------------------------
    def valu(self, insn: D.Insn) -> None:
        """Vector ALU subset (ch. 6/12): every lane effect is masked by EXEC."""
        w, f, n, fmt = self.w, insn.fields, insn.name, insn.fmt
        lit = self.literal(insn)
        if fmt in ("VOP3", "VOP3B"):
            if f["clamp"] or f["omod"] or f["neg"] or f["abs"]:
                raise Unsupported(insn.mnemonic, "VOP3 modifier")
            base = f["base"]
            s0, s1, s2 = f["src0"], f["src1"], f["src2"]
        elif fmt == "VOP2":
            base, s0, s1, s2 = "VOP2", f["src0"], f["vsrc1"] + 256, None
        elif fmt == "VOP1":
            base, s0, s1, s2 = "VOP1", f["src0"], None, None
        else:
            base, s0, s1, s2 = "VOPC", f["src0"], f["vsrc1"] + 256, None
        ex = w.exec
        lanes = [i for i in range(w.size) if (ex >> i) & 1]
        if base == "VOPC":
            self._vcmp(insn, s0, s1, lit, lanes)
            return
        # Lane ops that do not follow EXEC for their selection.
        if n == "V_READFIRSTLANE_B32":
            lane = (ex & -ex).bit_length() - 1 if ex else 0  # EXEC = 0 reads lane 0
            val = w.vgpr[s0 - 256][lane] if s0 >= 256 else w.src_scalar(s0, 32, lit)
            w.write_s32(f["vdst"], val)
            return
        if n == "V_READLANE_B32":
            if s0 < 256:
                raise Unsupported(insn.mnemonic, "src0 must be a VGPR")
            lane = w.src_scalar(s1, 32, lit) & (w.size - 1)
            w.write_s32(f["vdst"], w.vgpr[s0 - 256][lane])
            return
        if n == "V_WRITELANE_B32":
            # D[lane] = S0; EXEC is ignored. CONFIDENCE: HIGH (ch. 12 pseudocode: no EXEC mask).
            lane = w.src_scalar(s1, 32, lit) & (w.size - 1)
            w.vgpr[f["vdst"]][lane] = w.src_scalar(s0, 32, lit)
            return
        if n in CARRY_OPS:
            add, rev = CARRY_OPS[n]
            has_ci = "_CI_" in n
            if fmt == "VOP2":
                cin_mask, to_vcc = (w.vcc if has_ci else 0), True
            else:
                cin_mask, to_vcc = (w.read_lanemask(s2) if has_ci else 0), False
            outmask = 0
            for ln in lanes:
                a, b = w.src_lane(s0, ln, lit), w.src_lane(s1, ln, lit)
                if rev:
                    a, b = b, a
                cin = (cin_mask >> ln) & 1
                if add:
                    r = a + b + cin
                    co = int(r > M32)
                else:
                    r = a - b - cin
                    co = int(b + cin > a)
                w.vgpr[f["vdst"]][ln] = r & M32
                outmask |= co << ln
            # Lanes outside EXEC write 0 to the lane-mask output. CONFIDENCE: MED.
            if to_vcc:
                w.vcc = outmask
            else:
                w.write_lanemask(f["sdst"], outmask)
            return
        if n in ("V_MBCNT_LO_U32_B32", "V_MBCNT_HI_U32_B32"):
            for ln in lanes:
                if n.endswith("LO_U32_B32"):
                    m = (1 << min(ln, 32)) - 1
                else:
                    m = (1 << max(ln - 32, 0)) - 1
                cnt = bin(w.src_lane(s0, ln, lit) & m & M32).count("1")
                w.vgpr[f["vdst"]][ln] = (cnt + w.src_lane(s1, ln, lit)) & M32
            return
        fn = LANE_FN.get(n)
        if fn is None:
            raise Unsupported(insn.mnemonic)
        for ln in lanes:
            a = w.src_lane(s0, ln, lit)
            b = w.src_lane(s1, ln, lit) if s1 is not None else 0
            c = w.src_lane(s2, ln, lit) if s2 is not None else 0
            if n == "V_CNDMASK_B32":
                sel = ((w.vcc if fmt == "VOP2" else w.read_lanemask(s2)) >> ln) & 1
                r = b if sel else a
            else:
                r = fn(a, b, c)
            w.vgpr[f["vdst"]][ln] = r & M32

    def _vcmp(self, insn, s0, s1, lit, lanes) -> None:
        """Integer compares: VOPC writes VCC, VOP3 an SGPR pair; CMPX writes EXEC."""
        w, f, n = self.w, insn.fields, insn.name
        parts = n.split("_")  # V CMP[X] LT I32
        ty = parts[-1]
        if ty not in ("I32", "U32", "I64", "U64"):
            raise Unsupported(insn.mnemonic)
        opi = insn.op & 7  # CONFIDENCE: HIGH (Table 81: each family is 8 compare codes wide)
        wide, signed = ty.endswith("64"), ty.startswith("I")
        res = 0
        for ln in lanes:
            if wide:
                a, b = w.src_lane64(s0, ln, lit), w.src_lane64(s1, ln, lit)
                if signed:
                    a, b = s64(a), s64(b)
            else:
                a, b = w.src_lane(s0, ln, lit), w.src_lane(s1, ln, lit)
                if signed:
                    a, b = s32(a), s32(b)
            if VCMP[opi](a, b):
                res |= 1 << ln
        if parts[1] == "CMPX":
            # CONFIDENCE: MED: CMPX writes EXEC; no SGPR-pair destination is written.
            w.exec = res
        elif insn.fmt == "VOPC":
            w.vcc = res
        else:
            w.write_lanemask(f["vdst"], res)  # VOP3 compares: SDST is the VDST field [7:0]
