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
    scan.py <capture.prgcap|.prgbundle> [more...] [--gpu-replay PATH] [--mimg-decoder PATH]
            [--json] [--self-test]

Exit codes: 0 = scanned, no mismatch. 1 = mismatches found. 2 = could not scan (see below).

The ISA walk is prosper's OWN RDNA2 decoder, reached through `shader_inspect --mimg-sites` (#3184).
Nothing in this file decodes an instruction: a second implementation of a decoding rule is the one
that drifts, and it drifts silently, which is the failure this tool exists to make impossible.

It refuses to report "clean" without having parsed something. A run that dumps no shader, or whose
disassembler or decoder is missing, exits 2 and says so rather than printing zero findings -- a
silent scanner is indistinguishable from a clean codebase, and that failure has cost this project
real time.
"""
import argparse, json, os, re, shutil, struct, subprocess, sys, tempfile

# MIMG DIM values (SQ_RSRC), as reported by prosper's decoder in `Rdna2Inst::mimg_dim`. This is a
# naming and classification table, not a decode: nothing here reads an instruction word. Which bits
# hold DIM, where the opcode's split MSB lives and how long an instruction is are all stated once,
# in `src/gpu/recompiler/rdna2_decode.cpp`, and reach this tool through `--mimg-sites` (#3184).
DIM_NAMES = {0: "1D", 1: "2D", 2: "3D", 3: "CUBE",
             4: "1D_ARRAY", 5: "2D_ARRAY", 6: "2D_MSAA", 7: "2D_MSAA_ARRAY"}
DIM_ARRAYED = {4, 5, 7}
DIM_3D = {2}


class DecoderUnavailable(Exception):
    """The MIMG census could not be obtained. Never confusable with an empty census (see below)."""


def decoder_cmd(decoder):
    """The MIMG-site decoder, as an argv prefix. A `.py` value is run with this interpreter.

    Same shape as `replay_cmd`, and for the same Windows reason: `CreateProcess` cannot execute a
    `.py` by path the way a POSIX shebang can.
    """
    return [sys.executable, decoder] if str(decoder).endswith(".py") else [decoder]


def decode_mimg_sites(decoder, path):
    """(dword_index, opcode, dim) for every MIMG instruction in the raw RDNA2 stream at `path`.

    The walk is prosper's OWN decoder -- `shader_inspect --mimg-sites`, which is `rdna2_walk` over
    `rdna2_decode_one` (`src/gpu/recompiler/rdna2_decode.cpp`) and nothing else. There is no second
    implementation of the RDNA2 length rules in this file, and that is the point (#3184).

    What used to be here was a Python port of `rdna2_decode_one`'s format/length dispatch: enough of
    it to tell a real instruction boundary from an operand or trailing-literal dword whose top six
    bits happen to alias the MIMG encoding. It was correct -- 11,266 real shaders out of the local
    dump library, carrying 199,521 MIMG sites, decode identically through both -- but correct is not
    the property that matters for a copy. The copy is the one nobody updates when the decoder learns
    a new literal-forcing opcode or a wider NSA field, and it fails by *silently* returning a
    plausible instruction stream, which is the failure this whole tool exists to make impossible.
    The same folder had already drifted this way once, in the other direction: shader_histo's
    hand-maintained format-name table lost track of `Rdna2Format` when VOP3P was added (#3229).

    Raises `DecoderUnavailable` rather than returning [] when the census could not be taken. A
    scanner whose entire product is a CLEAN result must never let "this shader has no image
    instruction" and "the decoder did not run" reduce to the same value -- so the decoder prints a
    `mimg-sites-end` sentinel and this refuses any output without one, whatever the exit code said.

    The sentinel also carries the decoder's OWN site count, and this cross-checks it against the
    number of lines actually parsed. Without that, the remaining silent failure is on this side of
    the pipe rather than the far side: a site line the regex stops matching -- a widened field, a
    renamed key -- would be dropped while the sentinel still arrived, and an under-count reads as a
    cleaner shader. Two independent statements of the same number make that loud instead.
    """
    r = run(decoder_cmd(decoder) + [path, "--mimg-sites"])
    sites, claimed = [], None
    for ln in r.stdout.splitlines():
        m = re.match(r"^mimg-site pc=(\d+) op=0x([0-9a-fA-F]+) dim=(\d+)$", ln)
        if m:
            sites.append((int(m.group(1)), int(m.group(2), 16), int(m.group(3))))
            continue
        m = re.match(r"^mimg-sites-end .*\bsites=(\d+)\b", ln)
        if m:
            claimed = int(m.group(1))
    if r.returncode != 0 or claimed is None:
        why = "no mimg-sites-end sentinel" if claimed is None else "exit " + str(r.returncode)
        raise DecoderUnavailable(f"mimg decoder {why}: {(r.stderr or r.stdout).strip()[:160]}")
    if claimed != len(sites):
        raise DecoderUnavailable(f"mimg decoder reported sites={claimed} but {len(sites)} site "
                                 f"line(s) parsed -- the census and its own count disagree")
    return sites


def decode_mimg(decoder, path):
    """Every MIMG instruction in a raw RDNA2 shader file, as (opcode, dim)."""
    return [(op, dim) for _, op, dim in decode_mimg_sites(decoder, path)]


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


def scan_capture(replay, decoder, capture, tmp):
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
        if os.path.getsize(raw_p) == 0:
            # An empty ISA dump is a shader we did NOT read. Counting it as examined is how a
            # capture that yielded nothing still reports a clean scan.
            skipped += 1
            skip_reasons.append(f"draw[{draw}] {stage}: empty ISA dump")
            continue
        try:
            mimg = decode_mimg(decoder, raw_p)
        except DecoderUnavailable as e:
            # An unreadable census is a shader we did NOT examine, exactly like an unreadable dump.
            # The decoder also refuses a dump that is not a whole number of dwords, which the old
            # in-process walk silently truncated instead -- a partial dword is a truncated dump, and
            # a scanner that reports on one is reporting on bytes it does not have.
            skipped += 1
            skip_reasons.append(f"draw[{draw}] {stage}: {e}")
            continue
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


def _run_self_test_case(tmp, script, decoder, captures):
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
    r = subprocess.run([sys.executable, script, "--gpu-replay", stub,
                        "--mimg-decoder", decoder] + paths,
                       capture_output=True, text=True, env=env)
    return r.returncode, r.stdout, r.stderr


def self_test(decoder):
    """Every check below is written to FAIL under a mutation of the thing it claims to cover.

    The previous suite did not. It built its MIMG fixture out of the encoding constant it was
    checking, so mutating that constant mutated the fixture too and the check passed either way -- a
    same-source positive control, which tests the discriminator and never the domain. It also never
    touched the classifier, the exit contract, or either parser's caller, so nine separate mutations
    (including `DIM_ARRAYED = {4, 7}`, which drops the entire #325 class this tool exists to find)
    all still printed PASSED.

    Since #3184 the decode arms are no longer hermetic, and that is the improvement rather than a
    regression: the walk they exercise is prosper's REAL decoder, reached through
    `shader_inspect --mimg-sites`, so a change to `rdna2_decode.cpp`'s length dispatch now reddens
    this suite. It could not before -- the Python port was a separate implementation, and a suite
    that only tested the copy is exactly how the copy drifts unnoticed. A missing decoder is
    reported as exit 2 (could not test), never as a pass.
    """
    if not decoder or not os.path.exists(decoder):
        print(f"self-test: no shader_inspect at {decoder!r}. That binary IS the ISA walk under\n"
              f"           test, so this suite cannot run without it and must not report a pass.\n"
              f"           Build it (cmake --build <dir> --target shader_inspect) and pass\n"
              f"           --mimg-decoder <path>.", file=sys.stderr)
        return 2
    ok = True

    def check(cond, label):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}]   {label}")
        ok = ok and cond

    with tempfile.TemporaryDirectory() as fixtures:
        # Hand-built ISA streams, decoded THE WAY scan_capture decodes a real dump: written to a
        # file and handed to prosper's own decoder. The words below are computed by hand, never
        # assembled from a constant this file also uses to judge the answer.
        def sites(words):
            path = os.path.join(fixtures, "fixture.bin")
            with open(path, "wb") as f:
                f.write(words)
            return decode_mimg_sites(decoder, path)

        def mimg(words):
            return [(op, dim) for _, op, dim in sites(words)]

        def refused(words):
            try:
                mimg(words)
            except DecoderUnavailable:
                return True
            return False

        # --- MIMG identity. 0xF0800028: encoding 0b111100, opcode 0x20, DIM 5. 0xF0800029 sets the
        # opcode MSB -> 0xa0. 0xF4800028 is encoding 0b111101 and must decode to nothing.
        check(mimg(struct.pack("<II", 0xF0800028, 0)) == [(0x20, 5)],
              "literal 0xF0800028 decodes as opcode 0x20 / DIM 5")
        check(mimg(struct.pack("<II", 0xF0800029, 0)) == [(0xa0, 5)],
              "opcode MSB (bit 0) is reconstructed -- 0xF0800029 is opcode 0xa0, not 0x20")
        check(mimg(struct.pack("<II", 0xF4800028, 0)) == [],
              "encoding 0b111101 is NOT MIMG (pins the encoding against an off-by-one)")
        check(mimg(struct.pack("<I", 0x12345678)) == [], "a non-MIMG word decodes to nothing")
        # A real 2-dword MIMG's SECOND dword (VADDR/SRSRC/SSAMP operand fields) is consumed as part
        # of that SAME instruction by the length-aware walk, not re-tested as a new instruction
        # start -- #3040. This is the load-bearing behavior of a real walk: a scan that tested every
        # dword unconditionally would report TWO hits here.
        check(mimg(struct.pack("<II", 0xF0800028, 0xF0800028)) == [(0x20, 5)],
              "a real MIMG's own dword1 is consumed by that instruction, not read as a second one")
        # The exact sequence review of #3039 used: a literal aliasing the encoding, then a REAL
        # 2D_ARRAY sample. The real one must survive the phantom, AND the phantom itself must be
        # gone: `0x7E0002FF` is `v_mov_b32 v0, <literal>` (VOP1, src0=0xFF forces a trailing
        # literal), so the walk consumes dwords 0-1 as ONE instruction and never tests dword1 in
        # isolation -- unlike a dword-by-dword scan, which reads dword1 as its own MIMG candidate.
        check(mimg(struct.pack("<III", 0x7E0002FF, 0xF0000000, 0xF0800028)) == [(0x20, 5)],
              "a real MIMG immediately after an aliasing literal is found, and ONLY it is (#3039)")

        # --- #3040's positive control, built BY HAND rather than from `DIM_ARRAYED` (a same-source
        # control tests the discriminator, not the domain -- see CLAUDE.md). This is a single real
        # instruction -- `v_mov_b32 v0, 0xF0000028` (VOP1, forced literal) -- and NO MIMG
        # instruction anywhere in it. The literal's value is hand-picked so ITS OWN top 6 bits alias
        # the MIMG encoding (0b111100) with DIM=5 (2D_ARRAY, an arrayed shape) in bits [5:3]:
        # exactly the pattern a dword-by-dword scan mistakes for `image_sample dim:2D_ARRAY`. Such a
        # scan decodes one MIMG here and classify() reports a phantom `array-layer-dropped` defect
        # against a "shader" that samples no image at all.
        phantom_literal = struct.pack("<II", 0x7E0002FF, 0xF0000028)
        check(mimg(phantom_literal) == [],
              "a literal operand aliasing an ARRAYED MIMG encoding is not read as an instruction")
        check(classify(mimg(phantom_literal), [("2D", 0)], []) == [],
              "...and so the full pipeline reports no phantom array-layer-dropped finding either")

        # --- NSA (Non-Sequential Address) length: a real MIMG can be 2-5 dwords (dword0[2:1] gives
        # the extra-dword count), and each extra dword must be skipped too, not just dword1. Here
        # the NSA instruction's own extra dwords are themselves crafted to alias the MIMG encoding
        # -- if the walk advanced by a fixed 2 dwords instead of reading the NSA field, it would
        # misread them as two more (phantom) MIMG instructions, then desync and miss the real one
        # that follows. Reverting `rdna2_decode.cpp`'s `2 + extra` to a flat `2` reddens this.
        nsa_dim1 = 0xF0000000 | (1 << 1) | (1 << 3)     # MIMG, NSA=1 extra dword, DIM=1 (plain 2D)
        check(sites(struct.pack("<IIII", nsa_dim1, 0xF0800028, 0xF0800028, 0xF0800028))
              == [(0, 0x00, 1), (3, 0x20, 5)],
              "NSA's extra dwords (dword0[2:1]) are skipped, not read as instructions, and the real "
              "MIMG that follows is found at the correct dword offset")

        # --- A stream the decoder cannot read is REFUSED, not answered with an empty census. This
        # is the half that keeps a clean result honest, and it is a real behavior change from the
        # in-process walk, which truncated a partial dword and returned []. A partial dword is a
        # truncated dump; a scanner that reports on one is reporting on bytes it does not have.
        check(refused(b""), "a zero-byte ISA dump is refused, not decoded as 'no MIMG'")
        check(refused(b"\x01\x02"), "half a dword is refused, not silently truncated")

        # --- The sentinel contract, with a stub decoder. `mimg-sites-end` is what separates "this
        # shader has no image instruction" -- the answer that certifies a clean scan -- from "the
        # decoder died before printing anything". Without this arm, a decoder that crashed on
        # startup would certify every shader in a run as clean.
        fixture = os.path.join(fixtures, "fixture.bin")
        with open(fixture, "wb") as f:
            f.write(struct.pack("<II", 0xF0800028, 0))
        silent = os.path.join(fixtures, "silent_decoder.py")
        with open(silent, "w") as f:
            f.write("import sys\nsys.exit(0)\n")          # exits 0 having printed nothing
        truncated = os.path.join(fixtures, "truncated_decoder.py")
        with open(truncated, "w") as f:                     # prints a site, then dies before the end
            f.write("import sys\nprint('mimg-site pc=0 op=0x20 dim=5')\nsys.exit(0)\n")
        undercount = os.path.join(fixtures, "undercount_decoder.py")
        with open(undercount, "w") as f:                    # sentinel claims 5, one line parseable
            f.write("import sys\n"
                    "print('mimg-site pc=0 op=0x20 dim=5')\n"
                    "print('mimg-site pc=2 op=0xZZ dim=5')\n"
                    "print('mimg-sites-end dwords=8 consumed=8 instructions=2 sites=5 endpgm=1')\n")
        for stub, label in ((silent, "a decoder that exits 0 having printed nothing"),
                            (truncated, "a decoder that prints sites but no mimg-sites-end"),
                            (undercount, "a census whose sentinel count disagrees with its lines")):
            raised = False
            try:
                decode_mimg_sites(stub, fixture)
            except DecoderUnavailable:
                raised = True
            check(raised, f"{label} is refused, not read as a census")

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
        rc, out, err = _run_self_test_case(tmp, script, decoder, ["good"])
        # The positive count matters as much as the exit code. Every `rc == 2` arm below would pass
        # if the stub could not START -- a stub that never runs is indistinguishable from a capture
        # that could not be scanned -- so at least one arm has to assert that the stub really
        # enumerated shaders. Without this, half the suite is vacuously green on any platform where
        # launching the stub fails, which is exactly what happened on Windows/MinGW.
        check(rc == 0 and "examined=2" in out,
              "a capture whose module IS arrayed exits 0 AND reports its 2 examined shaders")

        rc, out, err = _run_self_test_case(tmp, script, decoder, ["bad"])
        check(rc == 1 and "array-layer-dropped" in out, "a real mismatch exits 1 and is printed")

        rc, out, err = _run_self_test_case(tmp, script, decoder, ["empty"])
        check(rc == 2, "a capture with no shaders exits 2, not 0")

        rc, out, err = _run_self_test_case(tmp, script, decoder, ["broken"])
        check(rc == 2, "a capture gpu_replay could not inspect exits 2")

        rc, out, err = _run_self_test_case(tmp, script, decoder, ["nodump"])
        check(rc == 2 and "zero examined" in err,
              "dumps that exit non-zero are NOT counted as examined")

        rc, out, err = _run_self_test_case(tmp, script, decoder, ["emptyisa"])
        check(rc == 2, "a zero-byte ISA dump is NOT an examined shader (it exits 0, so only its "
                       "emptiness distinguishes it)")

        # The finding that made the old exit contract unsound: one good capture certifying a run.
        rc, out, err = _run_self_test_case(tmp, script, decoder, ["good", "empty"])
        check(rc == 2, "one readable capture does NOT certify a barren one alongside it")

        rc, out, err = _run_self_test_case(tmp, script, decoder, ["bad", "nodump"])
        check(rc == 2, "a barren capture outranks findings elsewhere (2 beats 1)")

        # Cross-capture contamination: `nodump` writes nothing, so it must contribute no findings
        # even when a previous capture in the SAME run wrote files under the same shader hashes.
        _, alone, _ = _run_self_test_case(tmp, script, decoder, ["bad"])
        finding_lines = lambda out: [ln for ln in out.splitlines() if "guest " in ln]
        for bad_tag in ("nodump", "partial", "silentzero"):
            _, pair, _ = _run_self_test_case(tmp, script, decoder, ["bad", bad_tag])
            attributed = [ln for ln in finding_lines(pair) if ln.startswith(bad_tag)]
            check(len(finding_lines(pair)) == len(finding_lines(alone)) and not attributed,
                  f"'{bad_tag}' contributes no findings of its own")
        # These three pin the two halves of the contamination fix INDEPENDENTLY, which the first
        # version did not: `partial` writes a file and exits non-zero, so ONLY the return-code check
        # rejects it; `silentzero` exits 0 having written nothing, so ONLY the per-capture temp
        # directory stops it inheriting the previous capture's file under the same shader hash.
        rc, _, err = _run_self_test_case(tmp, script, decoder, ["partial"])
        check(rc == 2, "a dump that writes a file but exits non-zero is not an examined shader")
        rc, _, err = _run_self_test_case(tmp, script, decoder, ["silentzero"])
        check(rc == 2, "a dump that exits 0 having written nothing is not an examined shader")

    print("== self-test PASSED ==" if ok else "== self-test FAILED ==")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="*")
    ap.add_argument("--gpu-replay", default="./build-linux/gpu_replay")
    # The MIMG walk is prosper's own decoder, invoked out of process (#3184). There is deliberately
    # no in-process fallback: a fallback would be the duplicate implementation this replaced, and an
    # untested one, so a missing decoder must stop the scan rather than quietly downgrade it.
    ap.add_argument("--mimg-decoder", default="./build-linux/shader_inspect",
                    help="shader_inspect binary, which supplies the real RDNA2 MIMG walk")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test(a.mimg_decoder)
    if not a.captures:
        ap.error("give at least one capture, or --self-test")
    if not os.environ.get("PROSPER_SPIRV_DIS") and not shutil.which("spirv-dis"):
        print("scan: spirv-dis not on PATH -- cannot read emitted shaders, so a clean result would\n"
              "      be meaningless. Run inside the toolchain container.", file=sys.stderr)
        return 2
    if not os.path.exists(a.gpu_replay):
        print(f"scan: no gpu_replay at {a.gpu_replay} (--gpu-replay to point elsewhere)", file=sys.stderr)
        return 2
    if not os.path.exists(a.mimg_decoder):
        print(f"scan: no shader_inspect at {a.mimg_decoder} -- that binary IS the ISA walk, so\n"
              f"      without it this tool cannot find a single image instruction and a clean\n"
              f"      result would be meaningless. Build it (--target shader_inspect), or point\n"
              f"      --mimg-decoder at it.", file=sys.stderr)
        return 2

    all_f, total_examined, total_skipped, failed, per_capture = [], 0, 0, [], []
    for cap in a.captures:
        # One temp dir PER CAPTURE. Sharing one dir keyed only on shader hash let a capture whose
        # dumps all failed be scored against the PREVIOUS capture's files -- findings reported
        # against bytes the tool never read.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                res, err = scan_capture(a.gpu_replay, a.mimg_decoder, cap, tmp)
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
