"""Command line for the independent RDNA2 reference interpreter.

Usage:
  python -m prosper.tools.rdna2_ref.cli disasm FILE.bin
  python -m prosper.tools.rdna2_ref.cli run FILE.bin [--wave 32|64] [--steps N]
  python -m prosper.tools.rdna2_ref.cli report --corpus {fw,eboot,census,census-refused}
        --path DIR [--store DIR] [--wave 32|64] [--llvm] [--sample N]

``report`` prints counts only (programs, closed, unsupported mnemonics, decode cross-check
disagreements) and never names, hashes or paths of the corpus members. Corpora are local-only.
A run starts from EXEC = all ones and zeroed registers with an empty memory image, so a
scalar load faults (reported separately as ``fault``); it is a coverage probe, not a
behavioural verdict.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from . import corpus, llvm_check
from . import decode as D
from . import machine as M


def run_program(code: bytes, wave: int = 64, steps: int = 100_000) -> tuple[dict, M.Wave]:
    """Run ``code`` from a clean wave with EXEC = all ones."""
    w = M.Wave(size=wave)
    w.exec = w.lane_mask
    res = M.Machine(w, code).run(steps)
    return res, w


def static_unsupported(code: bytes) -> tuple[list[str], str | None]:
    """Mnemonics of decodable instructions outside the subset, plus any decode error."""
    insns, err = D.decode_program(code)
    return [i.mnemonic for i in insns if not M.is_supported(i)], err


def summarize(progs: list[bytes], wave: int, use_llvm: bool, sample: int) -> dict:
    """Coverage numbers for one corpus."""
    out: dict = {"programs": len(progs)}
    static_closed = decode_err = 0
    status = collections.Counter()
    first_unsup = collections.Counter()
    static_unsup_progs = collections.Counter()
    for code in progs:
        bad, err = static_unsupported(code)
        decode_err += err is not None
        static_closed += (not bad) and err is None
        for m in set(bad):
            static_unsup_progs[m] += 1
        res, _ = run_program(code, wave)
        s = res["status"]
        status[s.split(":")[0] if s.startswith("fault") else s] += 1
        if s.startswith("unsupported:"):
            first_unsup[s] += 1
    out["static_closed"] = static_closed
    out["decode_errors"] = decode_err
    out["run_ended"] = status.get("ended", 0)
    out["run_status"] = dict(status.most_common(12))
    out["top10_unsupported_first_hit"] = first_unsup.most_common(10)
    out["top10_unsupported_programs_containing"] = static_unsup_progs.most_common(10)
    if use_llvm:
        total = programs_bad = checked = invalid = 0
        samples = []
        for code in progs:
            r = llvm_check.compare(code, llvm_check.run_llvm(code))
            total += len(r["disagreements"])
            programs_bad += bool(r["disagreements"])
            checked += r["mnemonic_checked"]
            invalid += r["llvm_invalid"]
            for d in r["disagreements"]:
                if len(samples) < sample:
                    samples.append(d)
        out["llvm"] = {
            "disagreements": total,
            "programs_with_disagreement": programs_bad,
            "mnemonics_checked": checked,
            "llvm_invalid_instructions": invalid,
            "sample": samples,
        }
    return out


def load(args) -> list[bytes]:
    """Select the corpus named on the command line."""
    path = Path(args.path)
    if args.corpus == "fw":
        return corpus.firmware_programs(path, Path(args.store) if args.store else None)
    if args.corpus == "eboot":
        return corpus.eboot_programs(path)
    return corpus.census_programs(path, refused=args.corpus == "census-refused")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("disasm")
    d.add_argument("file")
    r = sub.add_parser("run")
    r.add_argument("file")
    r.add_argument("--wave", type=int, default=64, choices=(32, 64))
    r.add_argument("--steps", type=int, default=100_000)
    p = sub.add_parser("report")
    p.add_argument("--corpus", required=True, choices=("fw", "eboot", "census", "census-refused"))
    p.add_argument("--path", required=True)
    p.add_argument("--store")
    p.add_argument("--wave", type=int, default=64, choices=(32, 64))
    p.add_argument("--llvm", action="store_true")
    p.add_argument("--sample", type=int, default=5)
    a = ap.parse_args(argv)
    if a.cmd == "disasm":
        insns, err = D.decode_program(Path(a.file).read_bytes())
        for i in insns:
            print(f"{i.pc:06x}: {i.mnemonic}{'' if M.is_supported(i) else '  [unsupported]'}")
        if err:
            print(f"decode error: {err}")
        return 0
    if a.cmd == "run":
        res, w = run_program(Path(a.file).read_bytes(), a.wave, a.steps)
        print(res)
        print(f"exec={w.exec:#x} vcc={w.vcc:#x} scc={w.scc}")
        return 0
    if a.llvm and not llvm_check.llvm_available():
        print("--llvm: llvm-mc/llvm-objdump not on PATH and no WSL to run them in", file=sys.stderr)
        return 2
    summary = summarize(load(a), a.wave, a.llvm, a.sample)
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
