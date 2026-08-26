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


def decode_mimg(raw: bytes):
    """Every MIMG instruction in a raw RDNA2 shader, as (opcode, dim)."""
    n = len(raw) // 4
    if not n:
        return []
    words = struct.unpack(f"<{n}I", raw[:n * 4])
    out = []
    for w in words:
        if (w >> 26) != MIMG_ENCODING:
            continue
        # opcode spans a split field, exactly as rdna2_decode.cpp reconstructs it.
        opcode = ((w & 1) << 7) | ((w >> 18) & 0x7F)
        out.append((opcode, (w >> 3) & 0x7))
    return out


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


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def enumerate_shaders(replay, capture):
    """Unique (draw, stage, hash) triples, so one shader compiled into many draws is dumped once."""
    r = run([replay, "--inspect-only", capture])
    if r.returncode == 0 and not r.stdout.strip():
        r = subprocess.run([replay, "--inspect-only", capture], capture_output=True, text=True)
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
    r = run([replay, "--bundle", path, os.path.join(tmp, "b.bmp")])
    subs = re.findall(r"bundle-submit=(\d+)", r.stdout + r.stderr)
    if not subs:
        return None, f"no submits found in bundle ({r.returncode}): {r.stderr.strip()[:160]}"
    out = os.path.join(tmp, os.path.basename(path) + ".prgcap")
    e = run([replay, "--bundle", path, "--bundle-extract-submit", subs[-1], out])
    if not os.path.exists(out):
        return None, f"could not extract submit {subs[-1]}: {e.stderr.strip()[:160]}"
    return out, None


def scan_capture(replay, capture, tmp):
    findings, examined, skipped = [], 0, 0
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
            run([replay, "--dump-compute-raw", str(draw), raw_p, capture, bmp])
            run([replay, "--dump-compute", str(draw), spv_p, capture, bmp])
        else:
            run([replay, "--dump-realized-shader", f"{draw}:{stage}", raw_p, capture, bmp])
            run([replay, "--dump-shader", f"{draw}:{stage}", spv_p, capture, bmp])
        if not (os.path.exists(raw_p) and os.path.exists(spv_p)):
            skipped += 1
            continue
        dis = run(["spirv-dis", spv_p])
        if dis.returncode != 0:
            skipped += 1
            continue
        mimg = decode_mimg(open(raw_p, "rb").read())
        if not mimg:
            examined += 1
            continue
        types, arities = parse_spirv_images(dis.stdout)
        emitted_arrayed = any(a for _, a in types)
        emitted_3d = any(d == "3D" for d, _ in types)
        examined += 1
        for opcode, dim in mimg:
            if dim in DIM_ARRAYED and not emitted_arrayed:
                findings.append(dict(capture=os.path.basename(capture), draw=draw, stage=stage,
                                     shader=sh, opcode=f"0x{opcode:02x}",
                                     guest_dim=DIM_NAMES[dim], emitted="no arrayed image type",
                                     klass="array-layer-dropped", detail="#325",
                                     max_coord_arity=max(arities) if arities else 0))
                break
            if dim in DIM_3D and not emitted_3d:
                findings.append(dict(capture=os.path.basename(capture), draw=draw, stage=stage,
                                     shader=sh, opcode=f"0x{opcode:02x}",
                                     guest_dim=DIM_NAMES[dim], emitted="no 3D image type",
                                     klass="volume-coordinate-dropped", detail="",
                                     max_coord_arity=max(arities) if arities else 0))
                break
    return dict(examined=examined, skipped=skipped, findings=findings), None


def self_test():
    """Prove each decoder fires and each can also say NO -- a matcher that only ever agrees is void."""
    ok = True

    def check(cond, label):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}]   {label}")
        ok = ok and cond

    # A hand-built MIMG word: encoding 0b111100, opcode 0x20 (image_sample), DIM=5 (2D_ARRAY).
    w = (MIMG_ENCODING << 26) | ((0x20 & 0x7F) << 18) | (5 << 3)
    check(decode_mimg(struct.pack("<I", w)) == [(0x20, 5)], "MIMG decode recovers opcode 0x20 / DIM 5")
    check(decode_mimg(struct.pack("<I", 0x12345678)) == [], "a non-MIMG word decodes to nothing")
    arrayed = "%1 = OpTypeImage %float 2D 0 1 0 1 Unknown"
    plain = "%1 = OpTypeImage %float 2D 0 0 0 1 Unknown"
    check(parse_spirv_images(arrayed)[0] == [("2D", 1)], "OpTypeImage arrayed flag read as 1")
    check(parse_spirv_images(plain)[0] == [("2D", 0)], "OpTypeImage arrayed flag read as 0")
    dis = ("%v3 = OpTypeVector %float 3\n"
           "%c = OpCompositeConstruct %v3 %a %b %d\n"
           "%r = OpImageSampleImplicitLod %v4float %s %c")
    check(parse_spirv_images(dis)[1] == [3], "sample coordinate arity read from its vector type")
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
    if not shutil.which("spirv-dis"):
        print("scan: spirv-dis not on PATH -- cannot read emitted shaders, so a clean result would\n"
              "      be meaningless. Run inside the toolchain container.", file=sys.stderr)
        return 2
    if not os.path.exists(a.gpu_replay):
        print(f"scan: no gpu_replay at {a.gpu_replay} (--gpu-replay to point elsewhere)", file=sys.stderr)
        return 2

    all_f, total_examined, total_skipped, failed = [], 0, 0, []
    with tempfile.TemporaryDirectory() as tmp:
        for cap in a.captures:
            res, err = scan_capture(a.gpu_replay, cap, tmp)
            if res is None:
                failed.append((cap, err))
                continue
            total_examined += res["examined"]
            total_skipped += res["skipped"]
            all_f += res["findings"]

    if a.json:
        print(json.dumps(dict(examined=total_examined, skipped=total_skipped,
                              failed=[{"capture": c, "error": e} for c, e in failed],
                              findings=all_f), indent=2))
    else:
        for c, e in failed:
            print(f"could not scan {c}: {e}", file=sys.stderr)
        for f in all_f:
            print(f"{f['capture']}  draw[{f['draw']}] {f['stage']}  guest {f['guest_dim']} "
                  f"op={f['opcode']} -> {f['emitted']}  [{f['klass']} {f['detail']}] "
                  f"coord_arity={f['max_coord_arity']}")
        print(f"\nshaders examined: {total_examined}   not readable: {total_skipped}   "
              f"captures unscannable: {len(failed)}")
        print(f"mismatches: {len(all_f)}")
        if total_examined == 0:
            print("\nNOTHING WAS EXAMINED -- this is not a clean result. Check the paths above.",
                  file=sys.stderr)

    if total_examined == 0:
        return 2
    return 1 if all_f else 0


if __name__ == "__main__":
    sys.exit(main())
