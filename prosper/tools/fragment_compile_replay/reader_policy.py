"""Reviewed source policy for compiler resource/ambient read seams.

This is intentionally conservative: candidate agents must audit changed compiler sources, route
new reads through the explicit gateways/transcript, and update the SHA inventory in the same PR.
The verified verdict is embedded at BUILD time and checked BEFORE offline compiler entry. Merely
changing a source/config identity cannot authorize an unaudited memory or environment reader.
"""
import hashlib
import json
from pathlib import Path
import re
import sys


def inventory(root):
    # Conservative prosper/src guard (*.cpp, *.hpp, *.h, *.inc): an already-called helper can add a reader without
    # changing its caller. This is a maintenance boundary, NOT an assertion that every source
    # is read by one invocation or that external runtime/toolchain behavior is certified.
    paths = set()
    for extension in ("*.cpp", "*.hpp", "*.h", "*.inc"):
        paths.update((root / "prosper/src").rglob(extension))
    if not paths or len(paths) > 512:
        raise ValueError("reader source count")
    result = {}
    for path in sorted(paths):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("reader source ownership/budget")
        # Normalize checkout newline policy only; all semantic source bytes remain guarded.
        source = path.read_bytes().replace(b"\r\n", b"\n")
        result[path.relative_to(root).as_posix()] = hashlib.sha256(source).hexdigest()
    return result


def verify(root):
    actual = inventory(root)
    policy = Path(__file__).with_name("reader_policy.json")
    if policy.stat().st_size > 128 * 1024:
        raise ValueError("reader policy size")
    expected = json.loads(policy.read_text(encoding="utf-8"))
    if not isinstance(expected, dict) or len(expected) > 512:
        raise ValueError("reader policy inventory")
    for name, digest in expected.items():
        if (not isinstance(name, str) or len(name) > 512 or
                not name.startswith("prosper/src/") or ".." in name.split("/") or
                any(ord(c) < 32 for c in name) or "\\" in name or
                not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("reader policy entry")
    if actual != expected:
        differences = [("changed", name) for name in sorted(actual.keys() & expected.keys())
                       if actual[name] != expected[name]]
        differences += [("missing", name) for name in sorted(expected.keys() - actual.keys())]
        differences += [("added", name) for name in sorted(actual.keys() - expected.keys())]
        for kind, name in differences[:32]:
            print(f"reader policy {kind}: {name}", file=sys.stderr)
        if len(differences) > 32:
            print(f"reader policy: {len(differences) - 32} further paths omitted (bounded)", file=sys.stderr)
        raise ValueError("reader-policy-source-mismatch (re-audit required)")


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("reader policy arguments")
        verify(Path(sys.argv[1]).resolve(strict=True))
        print("fragment compiler reader policy: verified")
    except Exception:
        print("fragment compiler reader policy: UNVERIFIED (re-audit changed read seams)", file=sys.stderr)
        sys.exit(2)
