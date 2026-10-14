"""The local all-stage shader corpus: harvest refused-shader dumps, survey a build, report flips.

Live runs write `refused_shaders_<UTC>_<n>/` directories (#4808) holding each refused program as
`<cs|ps|vs|vsmain>_<addr>_<hash>.bin`, most with a `.inputs` sidecar that records the live compiler
inputs. They end up scattered across run and capture directories. This module gathers them into one
content-addressed corpus, runs one `shader_inspect` build over every program, and compares two
builds' answers per program, so a recompiler change that flips a real game shader between accepted
and refused is visible before it merges. `recompile_corpus.py harvest|survey|flips|compare` is the
command line; everything here is importable for tests.

The corpus is game content and is never committed; only this code and the manifest format are.
"""

import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdna2_ref import decode as rdna2_decode  # noqa: E402

CORPUS_FORMAT = "prosper-shader-corpus/1"
RESULTS_FORMAT = "prosper-shader-corpus-results/1"
PREFIX_STAGE = {"cs": "compute", "ps": "fragment", "vs": "vertex", "vsmain": "vertex"}
DUMP_DIR = re.compile(r"^refused_shaders_(\d{8})-(\d{6})Z")
DUMP_FILE = re.compile(r"^(cs|ps|vsmain|vs)_([0-9a-f]+)_([0-9a-f]+)\.bin$")
TITLE = re.compile(r"(?<![A-Z0-9])((?:PPSA|CUSA)\d{5})(?![0-9])")
# A dump's parent may be a run directory whose logs name the dump being run (`<id>-app0`).
APP0 = re.compile(r"((?:PPSA|CUSA)\d{5})-app0")
LOG_NAMES = (
    "run.log",
    "launch.json",
    "plan.json",
    "app.json",
    "finish.json",
    "child-processes.json",
    "capture.log",
)
LOG_BYTES = 256 * 1024
SIBLING_LIMIT = 400
# Enum order of Rdna2Format (src/gpu/recompiler/rdna2_decode.hpp); reject lines print the index.
FORMATS = (
    "SOP2 SOP1 SOPK SOPC SOPP SMEM VOP2 VOP1 VOPC VOP3 VINTRP DS MUBUF MTBUF MIMG FLAT EXP VOP3P"
).split()
ACCEPTED, REFUSED = "accepted", "refused"
# The changes listed by group, and whose refusal names the group: the build that refuses.
GROUPED = (
    ("newly-refused", "candidate"),
    ("refused-earlier", "candidate"),
    ("newly-accepted", "base"),
)


# ---- harvest --------------------------------------------------------------------------------


def find_dump_dirs(roots, excludes=()):
    """Every `refused_shaders_*` directory under `roots`, never descending into `excludes`,
    hidden directories or the dumps themselves. Symlinks are not followed."""
    excluded = {Path(e).resolve() for e in excludes}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for here, dirs, _files in os.walk(root, followlinks=False):
            keep = []
            for d in dirs:
                path = Path(here) / d
                if DUMP_DIR.match(d):
                    yield path
                elif not d.startswith(".") or d in (".claude", "worktrees"):
                    if path.resolve() not in excluded and d not in ("node_modules", "_deps"):
                        keep.append(d)
            dirs[:] = keep


def is_build_dir(path: Path) -> bool:
    """ctest writes refused-shader dumps of its own fixtures into the build directory. Those are
    test programs, not game shaders, so a dump whose parent is a CMake build tree is skipped."""
    return (path / "CMakeCache.txt").exists()


def dump_date(dump: Path) -> str:
    m = DUMP_DIR.match(dump.name)
    if m:
        d, t = m.groups()
        return f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}Z"
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(dump.stat().st_mtime))


def title_hints(dump: Path) -> tuple[str | None, str]:
    """(title id, how it was found) for one dump directory: its own path first, then the names of
    files beside it and one level up (frame grabs carry the id), then the run's own logs, which
    name the `<id>-app0` dump they ran. None when nothing names a title."""
    m = TITLE.findall(str(dump))
    if m:
        return m[-1], "path"
    votes: Counter = Counter()
    for level in (dump.parent, dump.parent.parent):
        try:
            names = os.listdir(level)
        except OSError:
            continue
        if len(names) > SIBLING_LIMIT:
            continue
        for name in names:
            votes.update(TITLE.findall(name))
        for name in LOG_NAMES:
            try:
                with open(level / name, "rb") as f:
                    votes.update(APP0.findall(f.read(LOG_BYTES).decode("latin-1")))
            except OSError:
                pass
        if votes:
            return votes.most_common(1)[0][0], "run-files"
    return None, "none"


def live_rejects(dump: Path) -> dict:
    """{file name: live reject text} from the dump's index.txt."""
    out = {}
    try:
        text = (dump / "index.txt").read_text(errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        f = re.search(r"\bfile=(\S+)", line)
        r = re.search(r'\breject="([^"]*)"', line)
        if f and r:
            out[f.group(1)] = r.group(1)
    return out


def sidecar_schema(data: bytes) -> int | None:
    """The schema number of a `.inputs` sidecar (`PRRSINP\\0` then a little-endian u32)."""
    if len(data) >= 12 and data[:8] == b"PRRSINP\0":
        return int.from_bytes(data[8:12], "little")
    return None


def load_manifest(corpus: Path) -> dict:
    path = corpus / "manifest.json"
    if not path.exists():
        return {"format": CORPUS_FORMAT, "programs": {}}
    manifest = json.loads(path.read_text())
    if manifest.get("format") != CORPUS_FORMAT or not isinstance(manifest.get("programs"), dict):
        raise ValueError(f"{path} is not a {CORPUS_FORMAT} manifest")
    return manifest


def harvest(corpus: Path, roots, excludes=(), log=print) -> dict:
    """Copy every dumped program (and its sidecar) into `corpus/<stage>/<prefix>_<key>.bin`, keyed
    by the content of both files, and merge their sources into `corpus/manifest.json`. Sources are
    only read: nothing found is moved or deleted. Re-running adds new dumps and new sources."""
    manifest = load_manifest(corpus)
    programs = manifest["programs"]
    known_sources = {s["path"] for p in programs.values() for s in p["sources"]}
    stats = Counter()
    for dump in sorted(find_dump_dirs(roots, excludes)):
        if is_build_dir(dump.parent):
            stats["skipped ctest dumps"] += 1
            continue
        stats["dumps"] += 1
        title, how = title_hints(dump)
        date = dump_date(dump)
        rejects = live_rejects(dump)
        for blob in sorted(dump.iterdir()):
            m = DUMP_FILE.match(blob.name)
            if not m or str(blob) in known_sources:
                continue
            prefix = m.group(1)
            code = blob.read_bytes()
            sidecar = blob.with_suffix(".inputs")
            inputs = sidecar.read_bytes() if sidecar.exists() else None
            key = hashlib.sha256(code + b"\0inputs\0" + (inputs or b"")).hexdigest()[:24]
            stage = PREFIX_STAGE[prefix]
            name = f"{stage}/{prefix}_{key}.bin"
            entry = programs.get(name)
            if entry is None:
                (corpus / stage).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(blob, corpus / name)
                if inputs is not None:
                    shutil.copyfile(sidecar, (corpus / name).with_suffix(".inputs"))
                entry = programs[name] = {
                    "stage": stage,
                    "prefix": prefix,
                    "code_sha256": hashlib.sha256(code).hexdigest(),
                    "inputs_sha256": hashlib.sha256(inputs).hexdigest() if inputs else None,
                    "inputs_schema": sidecar_schema(inputs) if inputs else None,
                    "sources": [],
                }
                stats["new programs"] += 1
            source = {
                "path": str(blob),
                "date": date,
                "title": title,
                "title_from": how,
                "live_reject": rejects.get(blob.name, ""),
            }
            entry["sources"].append(source)
            stats["new sources"] += 1
    infer_titles(programs)
    stats["sources still without a title"] = sum(
        1 for p in programs.values() for s in p["sources"] if s["title"] is None
    )
    for p in programs.values():
        p["titles"] = sorted({s["title"] for s in p["sources"] if s["title"]})
    manifest["generated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    for k, v in sorted(stats.items()):
        log(f"{k}: {v}")
    return manifest


def infer_titles(programs: dict) -> None:
    """A dump with no title in its path or run files takes the title most of its programs carry
    in other, titled dumps (shader code is title-specific; shared engine code is outvoted)."""
    by_code: dict = defaultdict(Counter)
    for p in programs.values():
        for s in p["sources"]:
            if s["title"]:
                by_code[p["code_sha256"]][s["title"]] += 1
    by_dump: dict = defaultdict(Counter)
    untitled = []
    for p in programs.values():
        for s in p["sources"]:
            if s["title"] is None:
                dump = str(Path(s["path"]).parent)
                by_dump[dump].update(by_code.get(p["code_sha256"], Counter()))
                untitled.append(s)
    for s in untitled:
        votes = by_dump[str(Path(s["path"]).parent)]
        if votes:
            s["title"], s["title_from"] = votes.most_common(1)[0][0], "shared-code"


# ---- one build over the corpus --------------------------------------------------------------


class InspectorError(Exception):
    """shader_inspect could not run at all, as opposed to running and refusing a shader."""


def capabilities(inspect: str) -> dict:
    """What this build's shader_inspect can do, read from its own usage text."""
    try:
        done = subprocess.run([inspect], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as err:
        raise InspectorError(f"{inspect} did not run: {err}") from err
    usage = done.stdout + done.stderr
    if "usage:" not in usage:
        raise InspectorError(f"{inspect} printed no usage text")
    return {
        "inputs": "--inputs" in usage,
        "inputs_spirv": bool(re.search(r"--inputs <[^>]*>\s*\[--spirv-out", usage)),
    }


def strip_reject(line: str) -> str:
    """The reject text without the `[kind]` brackets and the trailing diagnostic context, so a
    `--stage` line and a live/replay record compare as the same text."""
    line = line.strip()
    m = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
    if m:
        line = f"{m.group(1)} {m.group(2)}"
    return re.sub(r"\s+stage=\S+\s+program=\S+\s+role=\S+\s*$", "", line).strip()


def opcode_name(words_text: str | None, fmt: str | None, op: str | None, code: bytes | None, pc):
    """The ISA mnemonic of the refused instruction, from the reject's own words when present,
    else from the program at the reported dword PC, else `FMT:op`."""
    candidates = []
    if words_text:
        try:
            ws = [int(w, 16) for w in words_text.split(",") if w]
            candidates.append(b"".join(w.to_bytes(4, "little") for w in ws))
        except ValueError:
            pass
    if code is not None and pc is not None and 0 <= pc * 4 < len(code):
        candidates.append(code[pc * 4 :])
    for data in candidates:
        try:
            insn = rdna2_decode.decode_one(data, 0)
        except (rdna2_decode.DecodeError, IndexError, KeyError, ValueError):
            continue
        if insn.name:
            return insn.name
    if fmt is not None and op is not None:
        name = FORMATS[int(fmt)] if fmt.isdigit() and int(fmt) < len(FORMATS) else f"fmt{fmt}"
        return f"{name}:{op}"
    return None


def classify(reject: str, code: bytes | None = None) -> dict:
    """{reason, pc, opcode} for one reject text. The reason is the reject kind plus its `mode=` and
    `reason=` fields; a wrapper's `inner=` reject supplies the PC and opcode it names."""
    text = strip_reject(reject) if reject else ""
    if not text:
        return {"reason": None, "pc": None, "opcode": None}
    inner = text.split(" inner=", 1)
    outer, detail = inner[0], (inner[1] if len(inner) == 2 else inner[0])
    kind = outer.split()[0]
    parts = [kind]
    for field in ("mode", "reason"):
        m = re.search(rf"\b{field}=(\S+)", outer)
        if m:
            parts.append(f"{field}={m.group(1)}")
    if len(parts) == 1:  # free text: keep its words, drop its numbers
        parts = [re.sub(r"\b(0x[0-9a-f]+|\d+)\b", "N", outer)[:96]]
    if len(inner) == 2:
        m = re.search(r"\bmode=(\S+)", detail)
        if m:
            parts.append(f"inner-mode={m.group(1)}")
    pc = re.search(r"\bpc=(\d+)", detail)
    pc = int(pc.group(1)) if pc else None
    words = re.search(r"\bwords=([0-9a-f,]+)", detail)
    fmt = re.search(r"\bfmt=(\d+)", detail)
    op = re.search(r"\bop=(0x[0-9a-f]+)", detail)
    return {
        "reason": " ".join(parts),
        "pc": pc,
        "opcode": opcode_name(
            words.group(1) if words else None,
            fmt.group(1) if fmt else None,
            op.group(1) if op else None,
            code,
            pc,
        ),
    }


def _run(cmd, env, timeout):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def replay_answer(inspect, caps, blob: Path, sidecar: Path, env, timeout, code) -> dict:
    """The build's answer on the recorded live inputs (`--inputs`)."""
    with tempfile.TemporaryDirectory() as tmp:
        spv = Path(tmp) / "replay.spv"
        cmd = [inspect, str(blob), "--inputs", str(sidecar)]
        if caps["inputs_spirv"]:
            cmd += ["--spirv-out", str(spv)]
        done = _run(cmd, env, timeout)
        if done is None:
            return {"verdict": "error", "detail": "timeout"}
        text = done.stdout + done.stderr
        if done.returncode == 2:
            # The build cannot read this sidecar's schema (or the file): no answer, not a refusal.
            last = text.strip().splitlines()[-1:] or [""]
            return {"verdict": "unreadable", "detail": last[0][-160:]}
        m = re.search(
            r'^inputs-replay status=(\w+) reject="([^"]*)"(?: why=(.*?))?(?: spirv_dwords=\d+)?$',
            text,
            re.M,
        )
        if done.returncode not in (0, 6, 7) or not m:
            return {"verdict": "error", "detail": f"exit {done.returncode}"}
        status, reject, why = m.group(1), m.group(2), m.group(3)
        verdict = {"ok": ACCEPTED, "rejected": REFUSED}.get(status, "incomplete")
        out = {"verdict": verdict, "reject": reject.strip(), "detail": (why or "").strip()}
        out.update(classify(out["reject"], code) if verdict == REFUSED else {})
        out["spirv_sha256"] = _sha(spv) if verdict == ACCEPTED else None
        return out


def stage_answer(inspect, stage: str, blob: Path, env, timeout, code) -> dict:
    """The build's answer without live inputs: `--stage compute`, or the `--synthetic-table`
    compile probe for vertex/fragment (table-less graphics answers are almost all `undetermined`).
    Every build of shader_inspect has both, so this answer always compares across builds."""
    with tempfile.TemporaryDirectory() as tmp:
        spv = Path(tmp) / "stage.spv"
        cmd = [inspect, str(blob), "--stage", stage]
        if stage != "compute":
            cmd += ["--synthetic-table", "--spirv-out", str(spv)]
        done = _run(cmd, env, timeout)
        if done is None:
            return {"verdict": "error", "detail": "timeout"}
        text = done.stdout + done.stderr
        status = re.search(r"^stage-recompile stage=\S+ status=(\S+)", text, re.M)
        ok_exits = (4, 5) if stage != "compute" else (0, 1, 3)
        if done.returncode not in ok_exits or not status:
            return {"verdict": "error", "detail": f"exit {done.returncode}"}
        status = status.group(1)
        accepted = status in ("ok", "ok-synthetic-table")
        terminal = [ln for ln in text.splitlines() if "role=terminal" in ln]
        out = {
            "verdict": ACCEPTED if accepted else REFUSED,
            "status": status,
            "reject": strip_reject(terminal[-1]) if terminal and not accepted else "",
        }
        out.update(classify(out["reject"], code) if not accepted else {})
        out["spirv_sha256"] = _sha(spv) if accepted and stage != "compute" else None
        return out


def _env() -> dict:
    # PROSPER_DBG only logs (it is what prints the reject lines); it never varies the output.
    return {**os.environ, "PROSPER_DBG": "1"}


def survey_stage(inspect, corpus: Path, name: str, entry: dict, timeout=300) -> dict:
    blob = corpus / name
    return stage_answer(inspect, entry["stage"], blob, _env(), timeout, blob.read_bytes())


def survey_replay(inspect, caps, corpus: Path, name: str, timeout=300) -> dict | None:
    blob = corpus / name
    sidecar = blob.with_suffix(".inputs")
    if not (sidecar.exists() and caps["inputs"]):
        return None
    return replay_answer(inspect, caps, blob, sidecar, _env(), timeout, blob.read_bytes())


def survey(corpus: Path, inspect: str, jobs: int = 8, log=print) -> dict:
    """Run one build over every program in the manifest. Returns the results document."""
    manifest = load_manifest(corpus)
    programs = manifest["programs"]
    if not programs:
        raise InspectorError(f"no programs in {corpus}/manifest.json; run harvest first")
    caps = capabilities(inspect)
    # The input-free answer depends only on the stage and the code, and one program recurs across
    # runs with differing sidecars, so it is computed once per distinct (stage, code).
    first: dict = {}
    for name, e in sorted(programs.items()):
        first.setdefault((e["stage"], e["code_sha256"]), name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        stage_jobs = {
            pool.submit(survey_stage, inspect, corpus, n, programs[n]): key
            for key, n in first.items()
        }
        replay_jobs = {pool.submit(survey_replay, inspect, caps, corpus, n): n for n in programs}
        by_code = {stage_jobs[f]: f.result() for f in stage_jobs}
        results = {}
        for i, fut in enumerate(concurrent.futures.as_completed(replay_jobs), 1):
            name, e = replay_jobs[fut], programs[replay_jobs[fut]]
            results[name] = {"stage": by_code[(e["stage"], e["code_sha256"])]}
            if fut.result() is not None:
                results[name]["replay"] = fut.result()
            if i % 500 == 0:
                log(f"  {i}/{len(replay_jobs)}")
    if all(r["stage"]["verdict"] == "error" for r in results.values()):
        raise InspectorError(f"{inspect} gave no answer for any program")
    return {
        "format": RESULTS_FORMAT,
        "inspect": str(inspect),
        "capabilities": caps,
        "programs": results,
    }


# ---- comparing two builds -------------------------------------------------------------------


def decided(base: dict, cand: dict) -> tuple[str, dict, dict]:
    """The answer pair a comparison stands on: the live-inputs replay when BOTH builds could read
    the sidecar, else the input-free `--stage` answer, which every build has."""
    rb, rc = base.get("replay"), cand.get("replay")
    if rb and rc and "unreadable" not in (rb["verdict"], rc["verdict"]):
        return "replay", rb, rc
    return "stage", base["stage"], cand["stage"]


def change_of(a: dict, b: dict) -> str | None:
    va, vb = a["verdict"], b["verdict"]
    if va == ACCEPTED and vb == REFUSED:
        return "newly-refused"
    if va == REFUSED and vb == ACCEPTED:
        return "newly-accepted"
    if va != vb:
        return "verdict-changed"
    # A build that cannot write the module has no hash: that is "not compared", never a change.
    ha, hb = a.get("spirv_sha256"), b.get("spirv_sha256")
    if va == ACCEPTED and ha and hb and ha != hb:
        return "output-changed"
    if va == REFUSED and (a.get("reason"), a.get("opcode")) != (b.get("reason"), b.get("opcode")):
        # Both refuse, for a different reason. A refusal at an EARLIER pc is a new check firing
        # before the old stopping point -- on input-free answers that is usually the only sign of
        # a new guard, because both builds still stop at the first table-dependent instruction.
        pa, pb = a.get("pc"), b.get("pc")
        if pa is not None and pb is not None and pb != pa:
            return "refused-earlier" if pb < pa else "refused-later"
        return "refusal-moved"
    return None


def flips(manifest: dict, base: dict, cand: dict) -> dict:
    """Per-program changes between two results documents, plus groups of the two verdict flips
    keyed by (reason, opcode) of the refusing side, each with its per-title program counts."""
    programs = manifest["programs"]
    rows, modes, outputs = [], Counter(), Counter()
    for name in sorted(base["programs"].keys() & cand["programs"].keys()):
        mode, a, b = decided(base["programs"][name], cand["programs"][name])
        modes[mode] += 1
        if a["verdict"] == b["verdict"] == ACCEPTED:
            hashed = bool(a.get("spirv_sha256") and b.get("spirv_sha256"))
            outputs["compared" if hashed else "not compared (a build wrote no module)"] += 1
        change = change_of(a, b)
        if change:
            entry = programs.get(name, {})
            rows.append(
                {
                    "program": name,
                    "stage": entry.get("stage"),
                    "titles": entry.get("titles", []),
                    "mode": mode,
                    "change": change,
                    "base": a,
                    "candidate": b,
                }
            )
    groups = {}
    for kind, side in GROUPED:
        g: dict = {}
        for r in rows:
            if r["change"] != kind:
                continue
            key = (r[side].get("reason") or "?", r[side].get("opcode") or "-")
            cell = g.setdefault(
                key,
                {
                    "reason": key[0],
                    "opcode": key[1],
                    "entries": 0,
                    "codes": set(),
                    "titles": defaultdict(set),
                    "stages": Counter(),
                    "examples": [],
                },
            )
            # A program recurs across runs with different sidecars: count distinct code, so a title
            # that was run fifty times does not outweigh one run of another title.
            code = programs.get(r["program"], {}).get("code_sha256", r["program"])
            cell["entries"] += 1
            if code not in cell["codes"]:
                cell["stages"][r["stage"]] += 1
                if len(cell["examples"]) < 3:
                    cell["examples"].append({"program": r["program"], "pc": r[side].get("pc")})
            cell["codes"].add(code)
            for t in r["titles"] or ["unknown"]:
                cell["titles"][t].add(code)
        for cell in g.values():
            cell["programs"] = len(cell.pop("codes"))
            cell["titles"] = {t: len(c) for t, c in cell["titles"].items()}
        groups[kind] = sorted(
            g.values(), key=lambda c: (-len(c["titles"]), -c["programs"], c["reason"], c["opcode"])
        )
    return {
        "compared": sum(modes.values()),
        "modes": dict(modes),
        "only_base": len(base["programs"].keys() - cand["programs"].keys()),
        "only_candidate": len(cand["programs"].keys() - base["programs"].keys()),
        "accepted_outputs": dict(outputs),
        "changes": dict(Counter(r["change"] for r in rows)),
        "changes_by_mode": {
            m: dict(Counter(r["change"] for r in rows if r["mode"] == m)) for m in sorted(modes)
        },
        "groups": {k: [{**c, "stages": dict(c["stages"])} for c in v] for k, v in groups.items()},
        "rows": rows,
    }


def format_flips(report: dict, base_name: str, cand_name: str, top: int = 25) -> list[str]:
    out = [
        f"{base_name} -> {cand_name}: {report['compared']} programs compared "
        f"({', '.join(f'{v} by {k}' for k, v in sorted(report['modes'].items()))})",
        "changes: "
        + (", ".join(f"{k} {v}" for k, v in sorted(report["changes"].items())) or "none"),
    ]
    if report.get("accepted_outputs"):
        out.append(
            "accepted by both, SPIR-V: "
            + ", ".join(f"{v} {k}" for k, v in sorted(report["accepted_outputs"].items()))
        )
    for mode, changes in sorted(report.get("changes_by_mode", {}).items()):
        if changes:
            out.append(f"  by {mode}: " + ", ".join(f"{k} {v}" for k, v in sorted(changes.items())))
    names = {"candidate": cand_name, "base": base_name}
    for kind, side in GROUPED:
        side = names[side]
        groups = report["groups"][kind]
        if not groups:
            continue
        out.append(
            f"{kind}: {sum(g['programs'] for g in groups)} programs in {len(groups)} "
            f"groups (reason and opcode as {side} refuses), ranked by titles touched"
        )
        for g in groups[:top]:
            titles = ", ".join(
                f"{t} {n}" for t, n in sorted(g["titles"].items(), key=lambda kv: (-kv[1], kv[0]))
            )
            stages = "/".join(f"{s} {n}" for s, n in sorted(g["stages"].items()))
            out.append(
                f"  {len(g['titles'])} titles  {g['programs']:4d} programs  "
                f"{g['opcode']:<20} {g['reason']}  [{stages}]  ({titles})"
            )
        if len(groups) > top:
            out.append(f"  ... {len(groups) - top} more groups in the JSON report")
    return out


def summarize_corpus(manifest: dict) -> list[str]:
    programs = manifest["programs"].values()
    by_stage = Counter(p["stage"] for p in programs)
    with_inputs = Counter(p["stage"] for p in programs if p.get("inputs_sha256"))
    codes = {p["code_sha256"] for p in programs}
    titles = Counter(t for p in programs for t in (p.get("titles") or ["unknown"]))
    return [
        f"{sum(by_stage.values())} corpus entries ({len(codes)} distinct programs), "
        + ", ".join(
            f"{s} {n} ({with_inputs[s]} with sidecar)" for s, n in sorted(by_stage.items())
        ),
        f"{len([t for t in titles if t != 'unknown'])} titles: "
        + ", ".join(f"{t} {n}" for t, n in sorted(titles.items(), key=lambda kv: (-kv[1], kv[0]))),
    ]
