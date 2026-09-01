#!/usr/bin/env python3
"""Find shaders where prosper's emitted SPIR-V does not honour what the guest instruction declared.

The question this answers, for any capture, with no per-title knowledge:

    does the recompiled shader still address the image the way the guest asked?

A GCN/RDNA image instruction carries its own DIM field, so the guest states the addressing it wants
in the machine code itself. If the emitted SPIR-V declares a different image shape, the difference is
silently dropped coordinates -- which renders as *plausible but wrong* output rather than as an
error, and is therefore expensive to find by eye. #325 is one instance: a 2D_ARRAY sample lowered to
2D drops the array layer, so every surface samples slice 0. Tomb Raider I-III Remastered textures its
whole world from one 256-slice array and rendered flat for exactly that reason -- found by hand over
several hours, and mechanically detectable in seconds.

Usage:
    scan.py <capture.prgcap|.prgbundle> [more...] [--gpu-replay PATH] [--json] [--self-test]

Exit codes: 0 = scanned, no mismatch. 1 = mismatches found. 2 = could not scan (see below).

It refuses to report "clean" without having parsed something. A run that dumps no shader, or whose
disassembler is missing, exits 2 and says so rather than printing zero findings -- a silent scanner
is indistinguishable from a clean codebase, and that failure has cost this project real time.
"""
import argparse, json, os, re, shutil, struct, subprocess, sys, tempfile

# MIMG DIM field (SQ_RSRC): the encoding prosper's own decoder uses, rdna2_decode.cpp `(w >> 3) & 0x7`.
DIM_NAMES = {0: "1D", 1: "2D", 2: "3D", 3: "CUBE",
             4: "1D_ARRAY", 5: "2D_ARRAY", 6: "2D_MSAA", 7: "2D_MSAA_ARRAY"}
DIM_ARRAYED = {4, 5, 7}
DIM_3D = {2}
MIMG_ENCODING = 0b111100          # dword0[31:26]
S_ENDPGM = 0xBF810000             # SOPP encoding of s_endpgm -- terminates the walk, as it terminates
                                   # the real program (rdna2_decode.cpp's S_ENDPGM).


class Step:
    """One decoded instruction's length and (if it is one) MIMG identity. See `rdna2_instr_len`."""
    __slots__ = ("length", "is_mimg", "opcode", "dim", "is_end", "is_unknown")

    def __init__(self, length, is_mimg=False, opcode=None, dim=None, is_end=False, is_unknown=False):
        self.length, self.is_mimg, self.opcode, self.dim = length, is_mimg, opcode, dim
        self.is_end, self.is_unknown = is_end, is_unknown


def _sop_has_literal(w, nsrc):
    """A scalar source operand field == 255 selects a trailing 32-bit literal constant dword."""
    if (w & 0xFF) == 0xFF:
        return True
    if nsrc >= 2 and ((w >> 8) & 0xFF) == 0xFF:
        return True
    return False


def rdna2_instr_len(words, i):
    """Decode the LENGTH (in dwords) of the instruction at `words[i]`, plus its MIMG identity if any.

    #3040: `decode_mimg_sites` used to test every dword for the MIMG top-6-bit encoding regardless of
    what instruction it actually belonged to, so an operand or trailing-literal dword whose top bits
    happened to alias the encoding (0b111100) read as an instruction start -- a phantom finding for a
    MIMG that was never in the program. The fix is to walk real instruction boundaries: decode each
    instruction's length from its format, exactly as the guest's own wavefront would, and only ever
    test a dword for MIMG-ness when it is actually the first dword of an instruction.

    This mirrors the length computation in `rdna2_decode_one` / `rdna2_walk`
    (`src/gpu/recompiler/rdna2_decode.cpp`) -- every format bucket that instruction actually
    dispatches on, so this walk lands on the same instruction boundaries the real recompiler does. It
    intentionally ports only the LENGTH rule, not full operand decode (register fields, DPP/SDWA
    sub-modes, condition codes): that is all a boundary-accurate walk needs, and it is a small,
    self-contained piece of the ~1,200-line decoder, not a port of the whole thing.

    Being a second, independent implementation of that dispatch is itself a drift risk -- if
    `rdna2_decode_one`'s length rules change, this walk needs a matching update or it can silently
    fall out of sync. #3040's own suggested fix (teaching `shader_histo` to emit real per-instruction
    MIMG sites and having `scan.py` consume that) removes the duplication entirely and remains the
    more robust long-term direction; this is the length-aware fix within the existing Python-only
    tool, not a replacement for that suggestion.

    RDNA2 ISA reference: AMD document 70648 ("RDNA 2" Instruction Set Architecture: Reference Guide).
    """
    n = len(words)
    w = words[i]
    max_dwords = n - i

    if (w & 0x80000000) == 0:
        # Vector ALU group (bit31 == 0): VOP1 (0x7E prefix), VOPC (0x7C prefix), else VOP2.
        is_vop2 = (w & 0xFE000000) != 0x7E000000 and (w & 0xFE000000) != 0x7C000000
        src0 = w & 0x1FF
        # A VOP src0 field selects an extra dword four ways: 0xFF = trailing literal; 0xF9 = SDWA,
        # 0xFA = DPP16, 0xE9/0xEA = DPP8 = a control word. All make the instruction 2 dwords.
        if src0 in (0xF9, 0xFA, 0xE9, 0xEA):
            length = 2 if max_dwords >= 2 else 1
        else:
            lit = (src0 == 0xFF)
            if is_vop2:
                # The six K-carrying VOP2 mul-adds embed a mandatory 32-bit literal K:
                # v_madmk_f32(0x20) v_madak_f32(0x21) v_fmamk_f32(0x2C) v_fmaak_f32(0x2D)
                # v_fmamk_f16(0x37) v_fmaak_f16(0x38).
                if ((w >> 25) & 0x3F) in (0x20, 0x21, 0x2C, 0x2D, 0x37, 0x38):
                    lit = True
            length = (2 if max_dwords >= 2 else 1) if lit else 1
        return Step(min(length, max_dwords))

    if (w & 0xC0000000) == 0x80000000:
        # Scalar group (bits[31:30] == 10): SOPP/SOPC/SOP1/SOPK carved out before the SOP2 default.
        if (w & 0xFF800000) == 0xBF800000:
            return Step(min(1, max_dwords), is_end=(w == S_ENDPGM))
        if (w & 0xFF800000) == 0xBF000000:      # SOPC
            lit = _sop_has_literal(w, 2)
        elif (w & 0xFF800000) == 0xBE800000:    # SOP1
            lit = _sop_has_literal(w, 1)
        elif (w & 0xF0000000) == 0xB0000000:    # SOPK, except S_SETREG_IMM32_B32 (opcode 21) whose
            lit = ((w >> 23) & 0x1F) == 21       # 32-bit register data trails as a mandatory literal.
        else:                                    # SOP2
            lit = _sop_has_literal(w, 2)
        length = (2 if max_dwords >= 2 else 1) if lit else 1
        return Step(min(length, max_dwords))

    # Remaining top-bit patterns (bits[31:30] == 11) dispatch on dword0[31:26].
    bucket = w >> 26
    if bucket == 0x32:                                          # VINTRP
        return Step(min(1, max_dwords))
    if bucket in (0x33, 0x34, 0x35):                             # VOP3P, VOP3 (old-gen/RDNA2)
        # 2 dwords, plus a trailing literal when any of dword1's three 9-bit src fields is 0xFF.
        if max_dwords < 2:
            return Step(min(1, max_dwords))
        d1 = words[i + 1]
        lit = ((d1 & 0x1FF) == 0xFF or ((d1 >> 9) & 0x1FF) == 0xFF or ((d1 >> 18) & 0x1FF) == 0xFF)
        length = 3 if (lit and max_dwords >= 3) else 2
        return Step(min(length, max_dwords))
    if bucket in (0x36, 0x37, 0x38, 0x3A, 0x3D, 0x3E):           # DS, FLAT, MUBUF, MTBUF, SMEM, EXP
        return Step(min(2, max_dwords) if max_dwords >= 2 else 1)
    if bucket == MIMG_ENCODING:                                  # 0x3C -- MIMG
        # 2 dwords, plus NSA (Non-Sequential Address) extra dwords holding the split address VGPRs.
        # dword0[2:1] gives the extra-dword count (0..3), so total length is 2..5 dwords. Opcode and
        # DIM are both dword0-only fields, so they are exact even when the buffer ends mid-instruction.
        extra = (w >> 1) & 0x3
        opcode = ((w & 1) << 7) | ((w >> 18) & 0x7F)
        dim = (w >> 3) & 0x7
        return Step(min(2 + extra, max_dwords), is_mimg=True, opcode=opcode, dim=dim)
    return Step(min(1, max_dwords), is_unknown=True)


def decode_mimg_sites(raw: bytes):
    """(dword_index, opcode, dim) for every MIMG instruction found by a LENGTH-AWARE walk of `raw`.

    Each dword is visited only once, as either the first dword of an instruction (tested for MIMG-
    ness) or an operand/literal dword of the instruction that precedes it (never tested) -- see
    `rdna2_instr_len`. The walk stops at `s_endpgm` or an undecodable encoding, exactly as
    `rdna2_walk` does, since nothing after either is reachable guest code.
    """
    n = len(raw) // 4
    if not n:
        return []
    words = struct.unpack(f"<{n}I", raw[:n * 4])
    out = []
    pc = 0
    while pc < n:
        step = rdna2_instr_len(words, pc)
        if step.is_mimg:
            out.append((pc, step.opcode, step.dim))
        if step.length <= 0:
            break                                # safety: never loop on a zero-length step
        pc += step.length
        if step.is_end or step.is_unknown:
            break
    return out


def decode_mimg(raw: bytes):
    """Every MIMG instruction in a raw RDNA2 shader, as (opcode, dim).

    A length-aware walk (see `decode_mimg_sites`) can only ever test a dword for the MIMG encoding
    when it is genuinely the first dword of an instruction, so it can neither invent an MIMG from an
    operand/literal dword (#3040) nor let a phantom match swallow a real one that follows it (#3039;
    both directions are exercised in `self_test`).
    """
    return [(op, dim) for _, op, dim in decode_mimg_sites(raw)]


def parse_spirv_images(dis: str):
    """Emitted image types as (dim, arrayed), plus the arity of each sample/fetch coordinate."""
    types, arities = [], []
    for ln in dis.splitlines():
        m = re.search(r"= OpTypeImage \S+ (\w+) \d+ (\d+) (\d+) \d+ \w+", ln)
        if m:
            types.append((m.group(1), int(m.group(2))))
    vec = {}
    for ln in dis.splitlines():
        m = re.search(r"(%\w+) = OpTypeVector \S+ (\d+)", ln)
        if m:
            vec[m.group(1)] = int(m.group(2))
    comp = {}
    for ln in dis.splitlines():
        m = re.search(r"(%\w+) = OpCompositeConstruct (%\w+)", ln)
        if m:
            comp[m.group(1)] = vec.get(m.group(2), 0)
    for ln in dis.splitlines():
        m = re.search(r"= OpImage(?:Sample|Fetch)\w* \S+ %\w+ (%\w+)", ln)
        if m:
            arities.append(comp.get(m.group(1), 0))
    return types, arities


def classify(mimg, types, arities):
    """The product claim: guest sampled a shape the emitted SPIR-V cannot address.

    `types` is every OpTypeImage in the module, so `any()` here is deliberately permissive -- a
    module that declares one arrayed image clears every arrayed sample in it. That is the honest
    bound for a module-wide parse: without per-binding attribution we cannot say WHICH image a
    given MIMG resolved to, and claiming otherwise would report a shader that is actually correct.
    It costs recall, never soundness of a clean result.
    """
    emitted_arrayed = any(a for _, a in types)
    emitted_3d = any(d == "3D" for d, _ in types)
    arity = max(arities) if arities else 0
    out, seen = [], set()
    for opcode, dim in mimg:
        if dim in DIM_ARRAYED and not emitted_arrayed:
            key = ("array", dim)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(opcode=f"0x{opcode:02x}", guest_dim=DIM_NAMES[dim],
                            emitted="no arrayed image type", klass="array-layer-dropped",
                            detail="#325", max_coord_arity=arity))
        elif dim in DIM_3D and not emitted_3d:
            key = ("3d", dim)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(opcode=f"0x{opcode:02x}", guest_dim=DIM_NAMES[dim],
                            emitted="no 3D image type", klass="volume-coordinate-dropped",
                            detail="", max_coord_arity=arity))
    return out


def spirv_dis_cmd():
    """The disassembler, as an argv prefix.

    `PROSPER_SPIRV_DIS` overrides it, and a `.py` value is run with this interpreter. That override
    is what lets --self-test be hermetic ON EVERY PLATFORM: the first version shipped a `/bin/sh`
    shim on PATH, which is unrunnable on Windows/MinGW, so the suite went red there while passing
    locally. A PATH shim also cannot be made portable -- Windows resolves executables by extension.
    """
    override = os.environ.get("PROSPER_SPIRV_DIS")
    if override:
        return [sys.executable, override] if override.endswith(".py") else [override]
    return ["spirv-dis"]


def replay_cmd(replay):
    """gpu_replay, as an argv prefix.

    A `.py` value is run with this interpreter. In production `replay` is a real executable and this
    is the identity; the `.py` arm exists so --self-test can drive a stub. Windows `CreateProcess`
    cannot execute a `.py` by path the way a POSIX shebang can, and getting this wrong is not merely
    a red CI square: the ten `rc == 2` arms would still PASS, because a stub that cannot start looks
    exactly like a capture that could not be scanned. Half the suite would be vacuously green on
    Windows.
    """
    return [sys.executable, replay] if str(replay).endswith(".py") else [replay]


def run(cmd, **kw):
    """Every child gets a timeout. A hung gpu_replay on a shared GPU otherwise hangs the scan
    silently, and the reflex is to kill the scanner -- which loses the partial result too."""
    kw.setdefault("timeout", 300)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, **kw)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"timed out after {kw['timeout']}s")


def enumerate_shaders(replay, capture):
    """Unique (draw, stage, hash) triples, so one shader compiled into many draws is dumped once."""
    r = run(replay_cmd(replay) + ["--inspect-only", capture])
    if r.returncode != 0:
        return None, f"gpu_replay --inspect-only failed ({r.returncode}): {r.stderr.strip()[:200]}"
    seen, out = set(), []
    for ln in r.stdout.splitlines():
        m = re.match(r"^draw\[(\d+)\] ", ln)
        if not m:
            continue
        draw = int(m.group(1))
        for stage in ("vs", "fs"):
            h = re.search(rf"\b{stage}=\d+/(\w+)", ln)
            if not h or h.group(1) in seen:
                continue
            seen.add(h.group(1))
            out.append((draw, stage, h.group(1)))
    # Compute dispatches carry MIMG ops too, and the recompiler's array handling has historically
    # differed between the two stages -- for a long time array sampling worked ONLY in compute -- so
    # a scanner that looked at graphics alone would have reported the healthy half.
    for ln in r.stdout.splitlines():
        m = re.match(r"^compute\[(\d+)\] ", ln)
        if not m:
            continue
        h = re.search(r"shader=\d+/(\w+)", ln)
        key = h.group(1) if h else f"cs{m.group(1)}"
        if key in seen:
            continue
        seen.add(key)
        out.append((int(m.group(1)), "cs", key))
    if not out:
        return None, "this capture has no draws or dispatches carrying shaders"
    return out, None


def materialize(replay, path, tmp):
    """A .prgbundle is many submits; --inspect-only wants one .prgcap. Extract the last submit.

    The F9 frame grab writes bundles, and that grab is schedulable headlessly
    (PROSPER_GRAB_BUNDLE_AFTER_MS), which is what makes "one capture per title, then scan" cheap.
    Refusing bundles would have left the scanner unable to read the captures the project actually
    produces.
    """
    if not path.endswith(".prgbundle"):
        return path, None
    r = run(replay_cmd(replay) + ["--bundle", path, os.path.join(tmp, "b.bmp")])
    subs = re.findall(r"bundle-submit=(\d+)", r.stdout + r.stderr)
    if not subs:
        return None, f"no submits found in bundle ({r.returncode}): {r.stderr.strip()[:160]}"
    out = os.path.join(tmp, os.path.basename(path) + ".prgcap")
    e = run(replay_cmd(replay) + ["--bundle", path, "--bundle-extract-submit", subs[-1], out])
    if not os.path.exists(out):
        return None, f"could not extract submit {subs[-1]}: {e.stderr.strip()[:160]}"
    return out, None


def scan_capture(replay, capture, tmp):
    findings, examined, skipped = [], 0, 0
    skip_reasons = []
    capture, err = materialize(replay, capture, tmp)
    if capture is None:
        return None, err
    shaders, err = enumerate_shaders(replay, capture)
    if shaders is None:
        return None, err
    bmp = os.path.join(tmp, "sink.bmp")
    for draw, stage, sh in shaders:
        raw_p = os.path.join(tmp, f"{sh}.raw")
        spv_p = os.path.join(tmp, f"{sh}.spv")
        if stage == "cs":
            a = run(replay_cmd(replay) + ["--dump-compute-raw", str(draw), raw_p, capture, bmp])
            b = run(replay_cmd(replay) + ["--dump-compute", str(draw), spv_p, capture, bmp])
        else:
            a = run(replay_cmd(replay) + ["--dump-realized-shader", f"{draw}:{stage}", raw_p, capture, bmp])
            b = run(replay_cmd(replay) + ["--dump-shader", f"{draw}:{stage}", spv_p, capture, bmp])
        # The child's exit code, not the file's existence. `tmp` is per-capture now, but within one
        # capture two draws can share a shader hash, so a failed dump could still find the previous
        # draw's file sitting there and be scored as a successful read of the wrong shader.
        if a.returncode != 0 or b.returncode != 0 or \
                not (os.path.exists(raw_p) and os.path.exists(spv_p)):
            skipped += 1
            skip_reasons.append(f"draw[{draw}] {stage}: dump exit {a.returncode}/{b.returncode}")
            continue
        dis = run(spirv_dis_cmd() + [spv_p])
        if dis.returncode != 0:
            skipped += 1
            skip_reasons.append(f"draw[{draw}] {stage}: spirv-dis exit {dis.returncode}")
            continue
        raw = open(raw_p, "rb").read()
        if not raw:
            # An empty ISA dump is a shader we did NOT read. Counting it as examined is how a
            # capture that yielded nothing still reports a clean scan.
            skipped += 1
            skip_reasons.append(f"draw[{draw}] {stage}: empty ISA dump")
            continue
        mimg = decode_mimg(raw)
        if not mimg:
            examined += 1
            continue
        types, arities = parse_spirv_images(dis.stdout)
        examined += 1
        for f in classify(mimg, types, arities):
            f.update(capture=os.path.basename(capture), draw=draw, stage=stage, shader=sh)
            findings.append(f)
    return dict(examined=examined, skipped=skipped, findings=findings,
                skip_reasons=skip_reasons), None


SELF_TEST_STUB = r"""#!/usr/bin/env python3
# Stand-in for gpu_replay: enough surface for scan.py's end-to-end arms, with per-capture
# behaviour driven by the capture's own filename so one run can exercise good and barren captures.
import sys, os, struct
a = sys.argv[1:]
cap = next((x for x in a if x.endswith(".prgcap")), "")
tag = os.path.basename(cap).split(".")[0]
if "--inspect-only" in a:
    if tag == "empty":
        sys.exit(0)
    if tag == "broken":
        sys.stderr.write("inspect failed\n"); sys.exit(3)
    print("draw[0] vs=1/aaaa fs=2/bbbb")
    sys.exit(0)
for flag in ("--dump-realized-shader", "--dump-shader"):
    if flag in a:
        out = a[a.index(flag) + 2]
        if tag == "nodump":
            sys.exit(2)                       # writes nothing, and SAYS so
        if tag == "partial":
            open(out, "wb").write(b"\x00\x00\x00\x00")
            sys.exit(2)                       # writes a file AND fails: only the exit code tells
        if tag == "silentzero":
            sys.exit(0)                       # writes nothing and claims success
        if flag == "--dump-realized-shader":
            if tag == "emptyisa":
                open(out, "wb").write(b"")    # exits 0 having written nothing readable
                sys.exit(0)
            # one MIMG: 2D_ARRAY sample, plus its operand dword
            open(out, "wb").write(struct.pack("<II", 0xF0800028, 0x00000000))
        else:
            arrayed = "1" if tag == "good" else "0"
            open(out, "w").write("; SPIR-V\n%1 = OpTypeImage %float 2D 0 " + arrayed + " 0 1 Unknown\n")
        sys.exit(0)
sys.exit(0)
"""


def _run_self_test_case(tmp, script, captures):
    """Drive scan.py end-to-end against the stub. Returns (returncode, stdout, stderr)."""
    stub = os.path.join(tmp, "stub_replay.py")
    with open(stub, "w") as f:
        f.write(SELF_TEST_STUB)
    os.chmod(stub, 0o755)
    # A `spirv-dis` shim, so the suite is hermetic: it runs identically on a host without the
    # Vulkan toolchain, and cannot be quietly reduced to "spirv-dis missing -> exit 2" -- which is
    # what made four of these arms pass for the wrong reason the first time they were written.
    # Named through PROSPER_SPIRV_DIS rather than dropped on PATH, because a PATH shim is inherently
    # POSIX-only and turned the Windows/MinGW job red.
    shim = os.path.join(tmp, "spirv_dis_shim.py")
    with open(shim, "w") as f:
        f.write("import sys\nsys.stdout.write(open(sys.argv[1]).read())\n")
    env = dict(os.environ, PROSPER_SPIRV_DIS=shim)
    paths = []
    for name in captures:
        q = os.path.join(tmp, f"{name}.prgcap")
        open(q, "wb").write(b"\0")
        paths.append(q)
    r = subprocess.run([sys.executable, script, "--gpu-replay", stub] + paths,
                       capture_output=True, text=True, env=env)
    return r.returncode, r.stdout, r.stderr


def self_test():
    """Every check below is written to FAIL under a mutation of the thing it claims to cover.

    The previous suite did not. It built its MIMG fixture out of `MIMG_ENCODING`, so mutating that
    constant mutated the fixture too and the check passed either way -- a same-source positive
    control, which tests the discriminator and never the domain. It also never touched the
    classifier, the exit contract, or either parser's caller, so nine separate mutations (including
    `DIM_ARRAYED = {4, 7}`, which drops the entire #325 class this tool exists to find) all still
    printed PASSED.
    """
    ok = True

    def check(cond, label):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}]   {label}")
        ok = ok and cond

    # --- MIMG decode. Literal words, computed by hand, NOT built from the constants under test.
    # 0xF0800028: encoding 0b111100, opcode 0x20, DIM 5. 0xF0800029 sets the opcode MSB -> 0xa0.
    # 0xF4800028 is encoding 0b111101 and must decode to nothing.
    mimg2 = struct.pack("<II", 0xF0800028, 0)
    check(decode_mimg(mimg2) == [(0x20, 5)], "literal 0xF0800028 decodes as opcode 0x20 / DIM 5")
    check(decode_mimg(struct.pack("<II", 0xF0800029, 0)) == [(0xa0, 5)],
          "opcode MSB (bit 0) is reconstructed -- 0xF0800029 is opcode 0xa0, not 0x20")
    check(decode_mimg(struct.pack("<II", 0xF4800028, 0)) == [],
          "encoding 0b111101 is NOT MIMG (pins MIMG_ENCODING against an off-by-one)")
    check(decode_mimg(struct.pack("<I", 0x12345678)) == [], "a non-MIMG word decodes to nothing")
    # A real 2-dword MIMG's SECOND dword (VADDR/SRSRC/SSAMP operand fields) is consumed as part of
    # that SAME instruction by the length-aware walk, not re-tested as a new instruction start --
    # #3040. This is the load-bearing behavior change from the pre-#3040 walk, which tested every
    # dword unconditionally and would have reported TWO hits here.
    check(decode_mimg(struct.pack("<II", 0xF0800028, 0xF0800028)) == [(0x20, 5)],
          "a real MIMG's own dword1 is consumed by that instruction, not read as a second one (#3040)")
    # The exact sequence review of #3039 used: a literal aliasing the encoding, then a REAL
    # 2D_ARRAY sample. The real one must survive the phantom, AND (#3040) the phantom itself must be
    # gone: `0x7E0002FF` is `v_mov_b32 v0, <literal>` (VOP1, src0=0xFF forces a trailing literal), so
    # the length-aware walk consumes dwords 0-1 as ONE instruction and never tests dword1 in
    # isolation -- unlike the pre-#3040 walk, which read dword1 as its own MIMG candidate (DIM 0).
    swallow = struct.pack("<III", 0x7E0002FF, 0xF0000000, 0xF0800028)
    check(decode_mimg(swallow) == [(0x20, 5)],
          "a real MIMG immediately after an aliasing literal is found, and ONLY it is (#3039 + #3040)")
    check(decode_mimg(b"") == [] and decode_mimg(b"\x01\x02") == [], "short/empty input is safe")

    # --- #3040's own positive control, built BY HAND rather than from `MIMG_ENCODING`/`DIM_ARRAYED`
    # (a same-source control tests the discriminator, not the domain -- see CLAUDE.md). This is a
    # single real instruction -- `v_mov_b32 v0, 0xF0000028` (VOP1, forced literal) -- and NO MIMG
    # instruction anywhere in it. The literal's value is hand-picked so ITS OWN top 6 bits alias the
    # MIMG encoding (0b111100) with DIM=5 (2D_ARRAY, an arrayed shape) sitting in bits [5:3]: exactly
    # the pattern a dword-by-dword scan mistakes for `image_sample dim:2D_ARRAY`. Before #3040, this
    # decoded as one MIMG finding and classify() reported a phantom `array-layer-dropped` defect
    # against a "shader" that samples no image at all.
    phantom_literal = struct.pack("<II", 0x7E0002FF, 0xF0000028)
    check(decode_mimg(phantom_literal) == [],
          "a literal operand aliasing an ARRAYED MIMG encoding is not read as an instruction (#3040)")
    check(classify(decode_mimg(phantom_literal), [("2D", 0)], []) == [],
          "...and so the full pipeline reports no phantom array-layer-dropped finding either")

    # --- NSA (Non-Sequential Address) length: a real MIMG can be 2-5 dwords (dword0[2:1] gives the
    # extra-dword count), and each extra dword must be skipped too, not just dword1. Here the NSA
    # instruction's own extra dwords are themselves crafted to alias the MIMG encoding -- if the walk
    # advanced by a fixed 2 dwords instead of reading the NSA field, it would misread them as two
    # more (phantom) MIMG instructions, and would then desync and miss the real one that follows.
    nsa_dim1 = 0xF0000000 | (1 << 1) | (1 << 3)          # MIMG, NSA=1 extra dword, DIM=1 (plain 2D)
    nsa = struct.pack("<IIII", nsa_dim1, 0xF0800028, 0xF0800028, 0xF0800028)
    check(decode_mimg_sites(nsa) == [(0, 0x00, 1), (3, 0x20, 5)],
          "NSA's extra dwords (dword0[2:1]) are skipped, not read as instructions, and the real MIMG "
          "that follows is found at the correct offset (#3040)")

    # --- SPIR-V parse, both directions.
    arrayed = "%1 = OpTypeImage %float 2D 0 1 0 1 Unknown"
    plain = "%1 = OpTypeImage %float 2D 0 0 0 1 Unknown"
    check(parse_spirv_images(arrayed)[0] == [("2D", 1)], "OpTypeImage arrayed flag read as 1")
    check(parse_spirv_images(plain)[0] == [("2D", 0)], "OpTypeImage arrayed flag read as 0")
    dis = ("%v3 = OpTypeVector %float 3\n"
           "%c = OpCompositeConstruct %v3 %a %b %d\n"
           "%r = OpImageSampleImplicitLod %v4float %s %c")
    check(parse_spirv_images(dis)[1] == [3], "sample coordinate arity read from its vector type")

    # --- The classifier: the actual product claim, in BOTH directions.
    A = [("2D", 1)]        # an arrayed image was declared
    P = [("2D", 0)]        # only a plain 2D image was declared
    check(len(classify([(0x20, 5)], P, [])) == 1,
          "DIM 5 against a non-arrayed module IS reported (the #325 class)")
    check(classify([(0x20, 5)], A, []) == [],
          "DIM 5 against an arrayed module is NOT reported")
    check(len(classify([(0x20, 4)], P, [])) == 1 and len(classify([(0x20, 7)], P, [])) == 1,
          "DIM 4 and DIM 7 are arrayed too")
    check(classify([(0x20, 1)], P, []) == [], "plain 2D against a 2D module is not a finding")
    check(len(classify([(0x00, 2)], P, [])) == 1 and classify([(0x00, 2)], [("3D", 0)], []) == [],
          "the 3D class fires and can also say no")
    check(len(classify([(0x20, 5), (0x21, 5), (0x22, 5)], P, [])) == 1,
          "repeated hits of one class collapse to a single finding per shader")
    check(classify([], P, []) == [], "a shader with no MIMG yields nothing")

    # --- The exit contract, end to end. These are what make a clean result mean anything.
    script = os.path.abspath(__file__)
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err = _run_self_test_case(tmp, script, ["good"])
        # The positive count matters as much as the exit code. Every `rc == 2` arm below would pass
        # if the stub could not START -- a stub that never runs is indistinguishable from a capture
        # that could not be scanned -- so at least one arm has to assert that the stub really
        # enumerated shaders. Without this, half the suite is vacuously green on any platform where
        # launching the stub fails, which is exactly what happened on Windows/MinGW.
        check(rc == 0 and "examined=2" in out,
              "a capture whose module IS arrayed exits 0 AND reports its 2 examined shaders")

        rc, out, err = _run_self_test_case(tmp, script, ["bad"])
        check(rc == 1 and "array-layer-dropped" in out, "a real mismatch exits 1 and is printed")

        rc, out, err = _run_self_test_case(tmp, script, ["empty"])
        check(rc == 2, "a capture with no shaders exits 2, not 0")

        rc, out, err = _run_self_test_case(tmp, script, ["broken"])
        check(rc == 2, "a capture gpu_replay could not inspect exits 2")

        rc, out, err = _run_self_test_case(tmp, script, ["nodump"])
        check(rc == 2 and "zero examined" in err,
              "dumps that exit non-zero are NOT counted as examined")

        rc, out, err = _run_self_test_case(tmp, script, ["emptyisa"])
        check(rc == 2, "a zero-byte ISA dump is NOT an examined shader (it exits 0, so only its "
                       "emptiness distinguishes it)")

        # The finding that made the old exit contract unsound: one good capture certifying a run.
        rc, out, err = _run_self_test_case(tmp, script, ["good", "empty"])
        check(rc == 2, "one readable capture does NOT certify a barren one alongside it")

        rc, out, err = _run_self_test_case(tmp, script, ["bad", "nodump"])
        check(rc == 2, "a barren capture outranks findings elsewhere (2 beats 1)")

        # Cross-capture contamination: `nodump` writes nothing, so it must contribute no findings
        # even when a previous capture in the SAME run wrote files under the same shader hashes.
        _, alone, _ = _run_self_test_case(tmp, script, ["bad"])
        finding_lines = lambda out: [ln for ln in out.splitlines() if "guest " in ln]
        for bad_tag in ("nodump", "partial", "silentzero"):
            _, pair, _ = _run_self_test_case(tmp, script, ["bad", bad_tag])
            attributed = [ln for ln in finding_lines(pair) if ln.startswith(bad_tag)]
            check(len(finding_lines(pair)) == len(finding_lines(alone)) and not attributed,
                  f"'{bad_tag}' contributes no findings of its own")
        # These three pin the two halves of the contamination fix INDEPENDENTLY, which the first
        # version did not: `partial` writes a file and exits non-zero, so ONLY the return-code check
        # rejects it; `silentzero` exits 0 having written nothing, so ONLY the per-capture temp
        # directory stops it inheriting the previous capture's file under the same shader hash.
        rc, _, err = _run_self_test_case(tmp, script, ["partial"])
        check(rc == 2, "a dump that writes a file but exits non-zero is not an examined shader")
        rc, _, err = _run_self_test_case(tmp, script, ["silentzero"])
        check(rc == 2, "a dump that exits 0 having written nothing is not an examined shader")

    print("== self-test PASSED ==" if ok else "== self-test FAILED ==")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="*")
    ap.add_argument("--gpu-replay", default="./build-linux/gpu_replay")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.captures:
        ap.error("give at least one capture, or --self-test")
    if not os.environ.get("PROSPER_SPIRV_DIS") and not shutil.which("spirv-dis"):
        print("scan: spirv-dis not on PATH -- cannot read emitted shaders, so a clean result would\n"
              "      be meaningless. Run inside the toolchain container.", file=sys.stderr)
        return 2
    if not os.path.exists(a.gpu_replay):
        print(f"scan: no gpu_replay at {a.gpu_replay} (--gpu-replay to point elsewhere)", file=sys.stderr)
        return 2

    all_f, total_examined, total_skipped, failed, per_capture = [], 0, 0, [], []
    for cap in a.captures:
        # One temp dir PER CAPTURE. Sharing one dir keyed only on shader hash let a capture whose
        # dumps all failed be scored against the PREVIOUS capture's files -- findings reported
        # against bytes the tool never read.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                res, err = scan_capture(a.gpu_replay, cap, tmp)
            except Exception as e:                       # noqa: BLE001 -- must not exit 1
                res, err = None, f"{type(e).__name__}: {e}"
        if res is None:
            failed.append((cap, err))
            per_capture.append(dict(capture=cap, examined=0, skipped=0, error=err))
            continue
        total_examined += res["examined"]
        total_skipped += res["skipped"]
        all_f += res["findings"]
        per_capture.append(dict(capture=cap, examined=res["examined"], skipped=res["skipped"],
                                findings=len(res["findings"]), skip_reasons=res["skip_reasons"]))

    # A capture that yielded no examined shader was NOT scanned, whatever the other captures did.
    # The old global `total_examined == 0` guard let one readable capture certify a whole run.
    barren = [c["capture"] for c in per_capture if c["examined"] == 0]

    if a.json:
        print(json.dumps(dict(examined=total_examined, skipped=total_skipped,
                              captures=per_capture, barren=barren,
                              failed=[{"capture": c, "error": e} for c, e in failed],
                              findings=all_f), indent=2))
    else:
        for c, e in failed:
            print(f"could not scan {c}: {e}", file=sys.stderr)
        for f in all_f:
            print(f"{f['capture']}  draw[{f['draw']}] {f['stage']}  guest {f['guest_dim']} "
                  f"op={f['opcode']} -> {f['emitted']}  [{f['klass']} {f['detail']}] "
                  f"coord_arity={f['max_coord_arity']}")
        print()
        for c in per_capture:
            note = f"  ERROR {c['error']}" if c.get("error") else ""
            print(f"  {os.path.basename(c['capture'])}: examined={c['examined']} "
                  f"not-readable={c['skipped']} findings={c.get('findings', 0)}{note}")
        print(f"\nshaders examined: {total_examined}   not readable: {total_skipped}   "
              f"captures unscannable: {len(failed)}")
        print(f"shaders with a mismatch: {len(all_f)}")
        if all_f:
            print("\nConfirm each finding before acting on it -- classify() is a module-wide match (any\n"
                  "declared image of the wrong shape clears every guest sample of that shape in the\n"
                  "module, not just the one it actually resolved to), so a finding can still be a false\n"
                  "positive for THAT sample even though the ISA walk itself is length-aware (#3040) and\n"
                  "cannot invent an instruction from an operand or literal dword.")
        if barren:
            print(f"\nNOT A CLEAN RESULT -- {len(barren)} capture(s) yielded zero examined shaders:",
                  file=sys.stderr)
            for b in barren:
                print(f"  {b}", file=sys.stderr)

    if failed or barren:
        return 2
    return 1 if all_f else 0


if __name__ == "__main__":
    sys.exit(main())
