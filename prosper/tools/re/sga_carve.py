#!/usr/bin/env python3
"""Carve RDNA2 shader payloads out of a Rockstar SGA/AWC shader archive.

WHY THIS EXISTS, AND WHY IT CARVES RATHER THAN PARSES
-----------------------------------------------------
A title's shipped shader archive is the only place a live GPU program can be given a NAME. prosper
sees programs by guest address (`0x413dc6700`); the archive knows them as
`CS_RaytraceReflectionLightPass`. Matching one to the other turns "which dispatch is this?" from a
research question into a lookup, and it is the difference between debugging an anonymous 903-dword
blob and debugging a named lighting pass.

The obvious approach is to parse the container. That was tried first and abandoned on evidence: the
payload length is NOT stored adjacent to the payload (probed every offset from -4 to -96 as u32, u16
and dword-count across 600 known members -- best hit rate 3/600), payloads sit at arbitrary byte
alignment rather than dword alignment, and the header's fields did not survive contact. Reversing the
whole container is possible but expensive, and it is not what the question needs.

Carving is better here for a reason specific to this data: **the payload's own format is
self-delimiting and we already own a verified decoder for it.** `shader_inspect` stops at `s_endpgm`
and reports how many dwords it consumed -- so given a start offset it finds the end by itself, with no
container knowledge at all. That turns the problem into "find the starts", and the starts are
strongly marked: 98.8% of members begin with `s_inst_prefetch` (`0xbfa0....`), which is the GFX10
prologue.

So this tool finds candidate starts by that signature, hands each to `shader_inspect` for validation
and length, and takes the answer from the decoder rather than from a guessed struct. That also means
it inherits the decoder's correctness instead of forking a second, divergent one.

VERIFICATION IS THE POINT, NOT AN EXTRA
---------------------------------------
Carving invites false confidence: it always produces *something*. `--verify` therefore requires the
run to reproduce a directory of known-good members byte-for-byte, and reports any it missed. Without
that arm a carve result is a pile of blobs nobody should trust. Run it with `--verify` whenever an
extraction of the same archive is available; a carve that cannot reproduce known answers is a bug in
this tool, not a discovery about the archive.

KNOWN LIMIT, STATED RATHER THAN HIDDEN
--------------------------------------
The `0xbfa0....` prologue filter captures 98.8% of members, not all. The remainder start with other
encodings (`0xbefc....`, `0x7e00....`, SMEM forms) and will be missed unless `--extra-prologues` is
passed, which widens the filter at the cost of more candidates to validate. The summary always prints
the filter's coverage against the known set so the shortfall is visible rather than assumed away.
"""

import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# GFX10 prologue: s_inst_prefetch is SOPP opcode 0x20, so the dword's high half is 0xbfa0.
DEFAULT_PROLOGUES = (0xBFA0,)
# Other observed first-dword high halves, in descending frequency across a 4,864-member extraction.
EXTRA_PROLOGUES = (0xBEFC, 0x7E00, 0xF428, 0xF420, 0xF424, 0x7E10)

# An identifier-like ASCII run. Names sit a few hundred bytes ahead of their payload; see --name-window.
NAME_RE = re.compile(rb"[A-Za-z][A-Za-z0-9_]{5,63}")

INSPECT_RE = re.compile(
    r"bytes=(\d+)\s+dwords=(\d+)\s+consumed=(\d+)\s+instructions=(\d+)\s+endpgm=(\d+)")
COVERAGE_RE = re.compile(r"total=(\d+)\s+alu=(\d+)\s+exp=(\d+)\s+table=(\d+)\s+unsupported=(\d+)")


class Inspector:
    """Validate a candidate by decoding it, reusing prosper's own verified RDNA2 decoder.

    Reimplementing instruction-length decoding in Python would create a second decoder that can
    disagree with the one the emulator actually uses -- and a carve validated by the wrong decoder is
    worse than no carve, because its output looks authoritative. Shelling out keeps exactly one
    source of truth.
    """

    def __init__(self, binary: Path, slice_bytes: int, worker: int = 0):
        self.binary = binary
        self.slice_bytes = slice_bytes
        # One scratch file PER WORKER. Sharing one path across threads would let workers overwrite
        # each other's candidate between the write and the exec, producing decodes attributed to the
        # wrong offset -- a corruption that reads as a discovery rather than as a bug.
        self._dir = Path(tempfile.mkdtemp(prefix="sga_carve_"))
        self._tmp = self._dir / f"candidate_{worker}.bin"

    def decode(self, data: bytes, offset: int):
        """Return (consumed_dwords, instructions, unsupported) or None if this is not a shader.

        The slice grows adaptively. Handing the decoder a fixed 256 KiB per candidate costs about a
        gigabyte of pointless I/O across a full archive and dominates the runtime; starting small and
        growing ONLY when the decoder consumed everything it was given -- the one case where the
        answer might have been truncated -- is both faster and no less correct.
        """
        size = 8192
        while True:
            self._tmp.write_bytes(data[offset:offset + size])
            try:
                out = subprocess.run([str(self.binary), str(self._tmp)], capture_output=True,
                                     text=True, timeout=30).stdout
            except (subprocess.TimeoutExpired, OSError):
                return None
            m = INSPECT_RE.search(out)
            if not m:
                return None
            consumed, instructions, endpgm = int(m.group(3)), int(m.group(4)), int(m.group(5))
            truncated = consumed * 4 >= min(size, len(data) - offset)
            if endpgm and consumed > 0 and not truncated:
                cov = COVERAGE_RE.search(out)
                return consumed, instructions, int(cov.group(5)) if cov else -1
            if size >= self.slice_bytes or offset + size >= len(data):
                return None
            size = min(size * 4, self.slice_bytes)

    def cleanup(self):
        try:
            self._tmp.unlink()
            self._dir.rmdir()
        except OSError:
            pass


def candidate_offsets(data: bytes, prologues) -> list:
    """Every byte offset whose dword has one of the prologue high halves.

    Byte offsets, not dword offsets: members are stored at arbitrary alignment in this container
    (measured uniform across o%4 on 200 located members), so an aligned-only scan silently misses
    roughly three quarters of them. That exact mistake produced a 1,365-vs-5,698 undercount before it
    was caught, so the scan is unaligned by construction.
    """
    hits = []
    for hi in prologues:
        lo_byte, hi_byte = hi & 0xFF, (hi >> 8) & 0xFF
        start = 0
        needle = bytes((lo_byte, hi_byte))
        while True:
            i = data.find(needle, start)
            if i < 2:
                if i < 0:
                    break
                start = i + 1
                continue
            hits.append(i - 2)   # the pattern is the dword's HIGH half
            start = i + 1
    return sorted(set(h for h in hits if h >= 0))


def nearest_name(data: bytes, payload_start: int, window: int) -> str:
    """The last identifier-like string before the payload.

    Measured on 300 known members: every one has its own name within 4 KiB before its payload, at
    distances 338..514. The nearest preceding identifier is therefore a good default guess, but it is
    a HEURISTIC -- the container also stores field and semantic names -- so callers should treat a
    name as a label to verify, not as ground truth.
    """
    lo = max(0, payload_start - window)
    best = None
    for m in NAME_RE.finditer(data[lo:payload_start]):
        best = m.group().decode("ascii", "replace")
    return best or ""


def load_known(known_dir: Path):
    """sha256 -> sorted list of filenames, over every .bin under a directory."""
    known = {}
    for f in sorted(known_dir.rglob("*.bin")):
        try:
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError:
            continue
        known.setdefault(digest, []).append(f.name)
    return known


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path, help="the .awc / SGA shader archive")
    ap.add_argument("--shader-inspect", type=Path, required=True,
                    help="path to prosper's shader_inspect binary (supplies the verified decoder)")
    ap.add_argument("--out", type=Path, help="write carved payloads here (omit to only report)")
    ap.add_argument("--known", type=Path,
                    help="directory of already-extracted .bin members, for NEW/KNOWN classification")
    ap.add_argument("--verify", action="store_true",
                    help="require every --known member to be reproduced; exit non-zero if not")
    ap.add_argument("--extra-prologues", action="store_true",
                    help="also scan the rarer first-dword forms (wider, slower, higher recall)")
    ap.add_argument("--name-window", type=int, default=4096,
                    help="how far back to look for a name (default 4096)")
    ap.add_argument("--slice-bytes", type=int, default=262144,
                    help="bytes handed to the decoder per candidate (default 256 KiB)")
    ap.add_argument("--min-dwords", type=int, default=8,
                    help="reject anything shorter; guards against decoding noise (default 8)")
    ap.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 2)),
                    help="parallel decoder processes; this run is spawn-bound, not CPU-bound")
    args = ap.parse_args()

    if not args.shader_inspect.is_file() or not os.access(args.shader_inspect, os.X_OK):
        print(f"error: shader_inspect not executable: {args.shader_inspect}", file=sys.stderr)
        return 2
    data = args.archive.read_bytes()
    magic = data[:4]
    print(f"archive {args.archive.name}: {len(data):,} bytes, magic={magic!r}")

    prologues = DEFAULT_PROLOGUES + (EXTRA_PROLOGUES if args.extra_prologues else ())
    cands = candidate_offsets(data, prologues)
    print(f"candidate starts: {len(cands):,} (prologues {' '.join(f'0x{p:04x}' for p in prologues)})")

    known = load_known(args.known) if args.known else {}
    if known:
        print(f"known members: {sum(len(v) for v in known.values()):,} files, {len(known):,} unique")

    local = threading.local()
    pool = []

    def decode_one(off):
        insp = getattr(local, "insp", None)
        if insp is None:
            insp = Inspector(args.shader_inspect, args.slice_bytes, len(pool))
            local.insp = insp
            pool.append(insp)
        return off, insp.decode(data, off)

    carved, seen_digests = [], {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            decoded = list(ex.map(decode_one, cands))
        for off, res in decoded:
            # EVERY candidate is evaluated. An earlier revision skipped any candidate falling inside
            # an already-accepted payload, on the theory that a shader can contain the prologue
            # pattern mid-stream. That reasoning is sound but the implementation was not: a single
            # false positive that decodes far past its true end swallows every real member inside its
            # span, and the --verify arm caught it doing exactly that -- 1,098 known members missed in
            # one run. Overlap is resolved AFTER the fact, where a wrong guess costs a label rather
            # than a member.
            if res is None:
                continue
            dwords, instructions, unsupported = res
            if dwords < args.min_dwords:
                continue
            payload = data[off:off + dwords * 4]
            consumed_until = off + len(payload)
            digest = hashlib.sha256(payload).hexdigest()
            entry = {
                "offset": off, "bytes": len(payload), "dwords": dwords,
                "instructions": instructions, "unsupported": unsupported, "sha256": digest,
                "name": nearest_name(data, off, args.name_window),
                "known": digest in known,
            }
            if digest not in seen_digests:
                seen_digests[digest] = entry
                carved.append(entry)
    finally:
        for insp in pool:
            insp.cleanup()

    # Resolve overlap by REPORTING rather than by dropping. A payload wholly inside another is
    # usually a mid-stream prologue in a real shader, but it can also be the real member and the
    # container the false positive -- and only the known-set comparison can tell, so both are kept
    # and labelled.
    spans = sorted((c["offset"], c["offset"] + c["bytes"], c) for c in carved)
    for i, (lo, hi, c) in enumerate(spans):
        c["contained_in"] = next((o for o, e, _ in spans[:i] if o <= lo and hi <= e), None)
    contained = [c for c in carved if c["contained_in"] is not None]

    new = [c for c in carved if not c["known"]]
    print(f"\ncarved: {len(carved):,} unique payloads "
          f"({len(contained):,} lie wholly inside another -- likely mid-stream prologues)")
    if known:
        reproduced = sum(1 for d in known if d in seen_digests)
        missed = [d for d in known if d not in seen_digests]
        print(f"  reproduced known members: {reproduced:,} of {len(known):,}")
        print(f"  NEW (not in --known)    : {len(new):,}")
        if missed:
            print(f"  MISSED known members    : {len(missed):,}")
            for d in missed[:10]:
                print(f"      {known[d][0]}")
            if len(missed) > 10:
                print(f"      ... and {len(missed)-10} more")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for c in carved:
            tag = "known" if c["known"] else "new"
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", c["name"])[:80] or "unnamed"
            path = args.out / f"{tag}_{c['offset']:08x}_{safe}_{c['dwords']}dw.bin"
            path.write_bytes(data[c["offset"]:c["offset"] + c["bytes"]])
        print(f"\nwrote {len(carved):,} payloads to {args.out}")

    if new:
        print("\nfirst NEW payloads (offset, dwords, unsupported ops, nearest name):")
        for c in new[:20]:
            print(f"   @0x{c['offset']:08x} {c['dwords']:6d}dw unsupported={c['unsupported']:<4} {c['name']}")

    if args.verify:
        if not known:
            print("\nerror: --verify requires --known", file=sys.stderr)
            return 2
        if missed:
            print(f"\nVERIFY FAILED: {len(missed)} known member(s) not reproduced", file=sys.stderr)
            return 1
        print("\nVERIFY OK: every known member reproduced byte-for-byte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
