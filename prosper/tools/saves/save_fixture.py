#!/usr/bin/env python3
"""Capture, verify and seed local save fixtures so a test run can start from a post-setup state.

Newer titles force first-boot steps on every fresh run (EULAs scrolled to the end, initial
settings, language and brightness prompts), and every pad route has to re-navigate them. A fixture
is a saved copy of BOTH prosper save roots taken after those steps:

    PROSPER_SAVEDATA_DIR -> SaveDataMemory slots (the whole save path for Unity titles)
    PROSPER_SAVE0        -> the /savedata0 file mount

Fixtures are derived from game content, so they are NEVER committed. They live under
`$PROSPER_SAVE_FIXTURES` (default `~/prosper-saves`) as `<TITLE_ID>/<state>/`:

    fixture.json   metadata (title id, content version, prosper SHA, date, route, note, tree sha256)
    savedata/      copy of the PROSPER_SAVEDATA_DIR tree
    save0/         copy of the PROSPER_SAVE0 tree

What may be committed is `manifest.json` beside this file: it names fixtures and pins the sha256
and content version a route expects, so a route can declare the fixture it needs without shipping
it. `pin` writes an entry (sha256, content version and the capture `--note`, which is therefore
public); `seed` and `verify` refuse a fixture that disagrees with it.

Commands:

    capture <TITLE_ID> <state> --savedata-dir D --save0-dir D [--route PAD] [--note TEXT]
    seed    <TITLE_ID> <state> <run-dir> [--env] [--strict-version]
    list
    verify  [<TITLE_ID> [<state>]]
    pin     <TITLE_ID> <state>

`seed` ALWAYS copies into fresh `<run-dir>/savedata` and `<run-dir>/save0` and refuses a run
directory that already holds files: a stale or mutated save is a known trap (a resumed stale save
turned a whole Unbound run black and read as a render defect), so the fixture itself is only ever
read. `seed` prints `export` lines for a shell, or a JSON object with `--env`; warnings go to
stderr so the output stays safe to `eval`. When the dump's content version differs from the one
the fixture was captured against it warns loudly (an error with `--strict-version`).

Exit status: 0 success, 1 a fixture problem (missing, tampered, mismatched), 2 a usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROSPER_ROOT = HERE.parent.parent
DEFAULT_MANIFEST = HERE / "manifest.json"
SCHEMA_VERSION = 1
METADATA_NAME = "fixture.json"
# Subdirectory names inside a fixture, and inside a seeded run directory (the same names, so a
# seeded run dir looks like a fixture without its metadata).
SAVEDATA = "savedata"
SAVE0 = "save0"
ROOT_NAMES = (SAVEDATA, SAVE0)
ENV_SAVEDATA = "PROSPER_SAVEDATA_DIR"
ENV_SAVE0 = "PROSPER_SAVE0"

_TITLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,31}$")
_STATE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class FixtureError(Exception):
    """A fixture is missing, tampered with, or disagrees with what the caller expects."""


# --------------------------------------------------------------------------------------------
# locations
# --------------------------------------------------------------------------------------------

def fixtures_root() -> Path:
    override = os.environ.get("PROSPER_SAVE_FIXTURES")
    if override:
        return Path(override).expanduser()
    return Path.home() / "prosper-saves"


def manifest_path() -> Path:
    override = os.environ.get("PROSPER_SAVE_MANIFEST")
    return Path(override).expanduser() if override else DEFAULT_MANIFEST


def check_names(title: str, state: str) -> None:
    """Reject names that could escape the fixture root (they become path components)."""
    if not _TITLE_RE.fullmatch(title):
        raise FixtureError(f"invalid title id {title!r}")
    if not _STATE_RE.fullmatch(state) or state in (".", ".."):
        raise FixtureError(f"invalid state name {state!r}")


def fixture_dir(title: str, state: str) -> Path:
    check_names(title, state)
    return fixtures_root() / title / state


def title_from_dump(dump: str) -> str:
    """`PPSA24651-app0` (or a path ending in it) -> `PPSA24651`."""
    match = re.match(r"^([A-Za-z0-9]+)-app0$", os.path.basename(os.path.normpath(dump)))
    if not match:
        raise FixtureError(f"cannot derive a title id from dump {dump!r}")
    return match.group(1)


def _git_common_dir() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=PROSPER_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def default_dump_root() -> Path:
    """Where `<TITLE_ID>-app0` dumps live: same rule as snapshot.py's GAME_ROOT."""
    env = os.environ.get("PROSPER_GAME_ROOT")
    if env:
        return Path(env)
    common = _git_common_dir()
    return Path(common).parent if common else PROSPER_ROOT.parent


def read_content_version(title: str, dump_root: Path | None = None) -> str | None:
    """`contentVersion` from the dump's sce_sys/param.json, or None when it cannot be read."""
    root = dump_root if dump_root is not None else default_dump_root()
    param = Path(root) / f"{title}-app0" / "sce_sys" / "param.json"
    try:
        data = json.loads(param.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = data.get("contentVersion") if isinstance(data, dict) else None
    return str(version).strip() if version else None


def prosper_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROSPER_ROOT,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


# --------------------------------------------------------------------------------------------
# tree hashing and copying
# --------------------------------------------------------------------------------------------

def _walk(root: Path):
    """Yield (kind, relative posix path, absolute path) for a tree, sorted, rejecting symlinks."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        base = Path(dirpath)
        for name in dirnames + filenames:
            path = base / name
            if path.is_symlink():
                raise FixtureError(f"symlink in save tree is not allowed: {path}")
        for name in dirnames:
            yield "D", (base / name).relative_to(root).as_posix(), base / name
        for name in sorted(filenames):
            yield "F", (base / name).relative_to(root).as_posix(), base / name


def tree_sha256(base: Path) -> tuple[str, int, int]:
    """Hash `base/savedata` and `base/save0` (path, size and content of every entry).

    File modes and timestamps are deliberately ignored: they change on copy and say nothing about
    the save. Empty directories are hashed because a title may test for their existence.
    Returns (hex digest, file count, total bytes).
    """
    digest = hashlib.sha256()
    files = 0
    total = 0
    for name in ROOT_NAMES:
        digest.update(f"root {name}\n".encode())
        root = base / name
        if not root.is_dir():
            raise FixtureError(f"missing directory {root}")
        for kind, rel, path in _walk(root):
            if kind == "D":
                digest.update(f"D {rel}\n".encode())
                continue
            file_hash = hashlib.sha256()
            size = 0
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    file_hash.update(chunk)
                    size += len(chunk)
            digest.update(f"F {rel}\0{size}\0{file_hash.hexdigest()}\n".encode())
            files += 1
            total += size
    return digest.hexdigest(), files, total


def _copy_tree(src: Path, dst: Path) -> None:
    if src.is_symlink():
        raise FixtureError(f"save root is a symlink: {src}")
    for _kind, _rel, _path in _walk(src):   # rejects symlinks before anything is copied
        pass
    shutil.copytree(src, dst, symlinks=False)


# --------------------------------------------------------------------------------------------
# metadata and manifest
# --------------------------------------------------------------------------------------------

def load_metadata(title: str, state: str) -> dict:
    meta_path = fixture_dir(title, state) / METADATA_NAME
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FixtureError(
            f"no fixture {title}/{state} under {fixtures_root()} (capture it with "
            f"`save_fixture.py capture {title} {state} ...`)") from None
    except (OSError, ValueError) as exc:
        raise FixtureError(f"unreadable metadata {meta_path}: {exc}") from exc
    if not isinstance(meta, dict) or "tree_sha256" not in meta:
        raise FixtureError(f"malformed metadata {meta_path}")
    return meta


def fixture_exists(title: str, state: str) -> bool:
    return (fixture_dir(title, state) / METADATA_NAME).is_file()


def load_manifest() -> dict:
    path = manifest_path()
    if not path.is_file():
        return {"fixtures": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FixtureError(f"unreadable manifest {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("fixtures"), dict):
        raise FixtureError(f"malformed manifest {path}: expected a top-level 'fixtures' object")
    return data


def manifest_entry(title: str, state: str) -> dict | None:
    return load_manifest()["fixtures"].get(title, {}).get(state)


def check_against_manifest(title: str, state: str, actual_sha: str, meta: dict) -> None:
    """A fixture the committed manifest pins must match it. No entry means no expectation."""
    entry = manifest_entry(title, state)
    if entry is None:
        return
    if entry.get("sha256") != actual_sha:
        raise FixtureError(
            f"{title}/{state} does not match the committed manifest: tree sha256 "
            f"{actual_sha[:16]}... but the manifest pins {str(entry.get('sha256'))[:16]}...")
    pinned = entry.get("content_version")
    if pinned and pinned != meta.get("content_version"):
        raise FixtureError(
            f"{title}/{state} content version {meta.get('content_version')!r} but the manifest "
            f"pins {pinned!r}")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------------------------

def _relativize_route(route: str) -> str:
    """Store a route relative to the checkout when it lives there, so metadata names no host path."""
    try:
        return Path(route).resolve().relative_to(PROSPER_ROOT.parent).as_posix()
    except (ValueError, OSError):
        return route


def capture(title: str, state: str, savedata_dir: Path, save0_dir: Path, route: str | None = None,
            note: str = "", force: bool = False, dump_root: Path | None = None) -> dict:
    """Copy both save roots into a new fixture and write its metadata. Returns the metadata."""
    target = fixture_dir(title, state)
    for label, src in (("--savedata-dir", savedata_dir), ("--save0-dir", save0_dir)):
        if not Path(src).is_dir():
            raise FixtureError(f"{label} {src} is not a directory")
    if target.exists() and not force:
        raise FixtureError(f"fixture {title}/{state} already exists (use --force to replace it)")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{state}.", dir=target.parent))
    try:
        _copy_tree(Path(savedata_dir), staging / SAVEDATA)
        _copy_tree(Path(save0_dir), staging / SAVE0)
        sha, files, size = tree_sha256(staging)
        if files == 0:
            raise FixtureError("both save directories are empty: nothing to capture")
        meta = {
            "schema_version": SCHEMA_VERSION,
            "title_id": title,
            "state": state,
            "description": note,
            "content_version": read_content_version(title, dump_root),
            "prosper_sha": prosper_sha(),
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "route": _relativize_route(route) if route else None,
            "tree_sha256": sha,
            "file_count": files,
            "total_bytes": size,
        }
        write_json(staging / METADATA_NAME, meta)
        if target.exists():
            shutil.rmtree(target)
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return meta


def verify_fixture(title: str, state: str) -> dict:
    """Recompute the tree hash and compare it with the metadata and the manifest."""
    meta = load_metadata(title, state)
    actual, _files, _size = tree_sha256(fixture_dir(title, state))
    if actual != meta["tree_sha256"]:
        raise FixtureError(
            f"{title}/{state} was modified after capture: tree sha256 {actual[:16]}... but its "
            f"metadata records {meta['tree_sha256'][:16]}...")
    check_against_manifest(title, state, actual, meta)
    return meta


def version_warning(title: str, state: str, meta: dict, dump_root: Path | None) -> str | None:
    """A loud message when the dump's content version differs from the fixture's, else None."""
    captured = meta.get("content_version")
    current = read_content_version(title, dump_root)
    if captured is None or current is None:
        return (f"WARNING: cannot compare content versions for {title}/{state} (fixture "
                f"{captured!r}, dump {current!r}); the save may not match this dump")
    if captured != current:
        return (f"WARNING: CONTENT VERSION MISMATCH for {title}/{state}: the fixture was captured "
                f"against {captured} but the dump is {current}. The save may be rejected or "
                f"mis-parsed, and a route recorded against it may diverge.")
    return None


def _refuse_store_overlap(run_dir: Path) -> None:
    """Refuse a run dir inside, equal to, or an ancestor of the fixture store.

    Seeding into the store would write inside a fixture (breaking `verify` and every later seed)
    or leave a stray directory that `list` reports as BROKEN. Paths are resolved through symlinks
    so a link into the store is caught too.
    """
    resolved = Path(os.path.realpath(run_dir))
    store = Path(os.path.realpath(fixtures_root()))
    if resolved == store or store in resolved.parents or resolved in store.parents:
        raise FixtureError(
            f"run dir {run_dir} overlaps the fixture store {store}: seed into a fresh directory "
            f"outside it")


def seed(title: str, state: str, run_dir: Path, dump_root: Path | None = None,
         strict_version: bool = False, warn=None) -> dict:
    """Copy a fixture into fresh `<run_dir>/savedata` and `<run_dir>/save0`.

    Returns {"PROSPER_SAVEDATA_DIR": ..., "PROSPER_SAVE0": ...}. The fixture is only ever read:
    what is verified is the COPY that will be handed to the guest, so a fixture edited between
    check and copy cannot slip through. A run directory that already holds files is refused.
    """
    warn = warn or (lambda text: print(text, file=sys.stderr))
    meta = load_metadata(title, state)
    message = version_warning(title, state, meta, dump_root)
    if message:
        if strict_version:
            raise FixtureError(message.replace("WARNING: ", ""))
        warn(message)
    run_dir = Path(run_dir)
    _refuse_store_overlap(run_dir)
    dests = {name: run_dir / name for name in ROOT_NAMES}
    for dest in dests.values():
        if dest.is_symlink():
            raise FixtureError(f"{dest} is a symlink: seed only fills real fresh directories")
        if dest.exists() and any(dest.iterdir()):
            raise FixtureError(f"{dest} is not empty: seed only fills fresh directories")
    run_dir.mkdir(parents=True, exist_ok=True)
    src = fixture_dir(title, state)
    try:
        for name, dest in dests.items():
            if dest.exists():
                dest.rmdir()
            _copy_tree(src / name, dest)
        actual, _files, _size = tree_sha256(run_dir)
        if actual != meta["tree_sha256"]:
            raise FixtureError(
                f"{title}/{state} is corrupt or was modified after capture: seeded tree sha256 "
                f"{actual[:16]}... but its metadata records {meta['tree_sha256'][:16]}...")
        check_against_manifest(title, state, actual, meta)
    except BaseException:
        for dest in dests.values():
            shutil.rmtree(dest, ignore_errors=True)
        raise
    return {ENV_SAVEDATA: str(dests[SAVEDATA]), ENV_SAVE0: str(dests[SAVE0])}


def pin(title: str, state: str) -> Path:
    """Record a verified local fixture in the committed manifest. Returns the manifest path."""
    meta = verify_fixture(title, state)
    manifest = load_manifest()
    manifest["fixtures"].setdefault(title, {})[state] = {
        "sha256": meta["tree_sha256"],
        "content_version": meta.get("content_version"),
        "description": meta.get("description", ""),
    }
    path = manifest_path()
    write_json(path, manifest)
    return path


def list_fixtures() -> list[dict]:
    """Every local fixture, plus manifest entries that have no local copy (status MISSING)."""
    rows = []
    seen = set()
    root = fixtures_root()
    if root.is_dir():
        for title_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for state_dir in sorted(p for p in title_dir.iterdir() if p.is_dir()):
                if state_dir.name.startswith("."):
                    continue
                try:
                    meta = load_metadata(title_dir.name, state_dir.name)
                except FixtureError as exc:
                    rows.append({"title": title_dir.name, "state": state_dir.name,
                                 "status": "BROKEN", "detail": str(exc)})
                    continue
                seen.add((title_dir.name, state_dir.name))
                rows.append({"title": title_dir.name, "state": state_dir.name, "status": "local",
                             "meta": meta})
    for title, states in sorted(load_manifest()["fixtures"].items()):
        for state, entry in sorted(states.items()):
            if (title, state) not in seen:
                rows.append({"title": title, "state": state, "status": "MISSING", "meta": entry})
    return rows


# --------------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------------

def cmd_capture(args) -> int:
    meta = capture(args.title, args.state, Path(args.savedata_dir), Path(args.save0_dir),
                   route=args.route, note=args.note, force=args.force,
                   dump_root=Path(args.dump_root) if args.dump_root else None)
    print(f"captured {args.title}/{args.state}: {meta['file_count']} files, "
          f"{meta['total_bytes']} bytes, content version {meta['content_version']}, "
          f"sha256 {meta['tree_sha256'][:16]}...")
    if meta["content_version"] is None:
        print("WARNING: no content version recorded (dump param.json not found); pass "
              "--dump-root or set PROSPER_GAME_ROOT so seed can detect a version mismatch",
              file=sys.stderr)
    return 0


def cmd_seed(args) -> int:
    env = seed(args.title, args.state, Path(args.run_dir),
               dump_root=Path(args.dump_root) if args.dump_root else None,
               strict_version=args.strict_version)
    if args.env:
        print(json.dumps(env, indent=2, sort_keys=True))
    else:
        for key in (ENV_SAVEDATA, ENV_SAVE0):
            print(f"export {key}={shlex.quote(env[key])}")
    return 0


def cmd_list(args) -> int:
    rows = list_fixtures()
    if not rows:
        print(f"no fixtures under {fixtures_root()}")
        return 0
    for row in rows:
        meta = row.get("meta", {})
        if row["status"] == "BROKEN":
            print(f"{row['title']}/{row['state']}  BROKEN  {row['detail']}")
            continue
        sha = str(meta.get("tree_sha256") or meta.get("sha256") or "")[:12]
        print(f"{row['title']}/{row['state']}  {row['status']}  version={meta.get('content_version')}"
              f"  sha256={sha}  {meta.get('description', '')}")
    return 0


def cmd_verify(args) -> int:
    if args.title and args.state:
        targets = [(args.title, args.state)]
    else:
        targets = [(r["title"], r["state"]) for r in list_fixtures()
                   if r["status"] != "MISSING" and (not args.title or r["title"] == args.title)]
        if args.title and not targets:
            raise FixtureError(f"no fixtures for {args.title} under {fixtures_root()}")
    rc = 0
    for title, state in targets:
        try:
            verify_fixture(title, state)
            print(f"{title}/{state}: OK")
        except FixtureError as exc:
            print(f"{title}/{state}: FAIL {exc}", file=sys.stderr)
            rc = 1
    return rc


def cmd_pin(args) -> int:
    path = pin(args.title, args.state)
    print(f"pinned {args.title}/{args.state} in {path}")
    print("note: the description is written to the committed manifest and is public; it must "
          "not contain paths, names or anything derived from game content", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="copy both save roots into a new fixture")
    cap.add_argument("title")
    cap.add_argument("state")
    cap.add_argument("--savedata-dir", required=True, help="the PROSPER_SAVEDATA_DIR to capture")
    cap.add_argument("--save0-dir", required=True, help="the PROSPER_SAVE0 to capture")
    cap.add_argument("--route", help="the pad script that produced this state")
    cap.add_argument("--note", default="", help="one line: what state this is")
    cap.add_argument("--force", action="store_true", help="replace an existing fixture")
    cap.add_argument("--dump-root", help="directory holding <TITLE_ID>-app0 (default: auto)")
    cap.set_defaults(func=cmd_capture)

    sd = sub.add_parser("seed", help="copy a fixture into fresh per-run directories")
    sd.add_argument("title")
    sd.add_argument("state")
    sd.add_argument("run_dir")
    sd.add_argument("--env", action="store_true", help="print a JSON object instead of exports")
    sd.add_argument("--strict-version", action="store_true",
                    help="fail instead of warning on a content version mismatch")
    sd.add_argument("--dump-root", help="directory holding <TITLE_ID>-app0 (default: auto)")
    sd.set_defaults(func=cmd_seed)

    ls = sub.add_parser("list", help="list local fixtures and manifest entries")
    ls.set_defaults(func=cmd_list)

    ver = sub.add_parser("verify", help="check fixtures against their recorded sha256")
    ver.add_argument("title", nargs="?")
    ver.add_argument("state", nargs="?")
    ver.set_defaults(func=cmd_verify)

    pn = sub.add_parser("pin", help="record a verified fixture in the committed manifest")
    pn.add_argument("title")
    pn.add_argument("state")
    pn.set_defaults(func=cmd_pin)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FixtureError as exc:
        print(f"save_fixture: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
