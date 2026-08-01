#!/usr/bin/env python3
"""Duplicate-export census: which NIDs two LINKED modules both export, per title (#1635).

The loader builds its global export table with `exports.emplace(nid, addr)`, and emplace is a no-op on
an existing key — so a NID exported by two modules is silently aliased to whichever links first. The
loser stays mapped, its init_array still runs, and sceKernelDlsym consults the handle's own table
first (#147), so the same NID can resolve to two different addresses depending on which handle asks.

This answers "is that firing today?" statically, with no boot and no device.

    python3 tools/re/dup_exports.py [dump-root]

It scans only the modules prosper actually LINKS (the fixed named list in boot_program.cpp) — scanning
every .prx present would report collisions between modules that are never loaded together, which is a
different and much noisier question.

CAVEAT, learned the hard way: dedupe by RESOLVED FILE. prosper's list carries two spellings of
Il2CppUserAssemblies, and a case-insensitive resolver maps both to the same file, reporting it as ~300
self-duplicates. That artifact was ~95% of the first run of this census — the instrument, not the
subject. Anything that looks like a module colliding with itself is that bug.

Result at the time of writing: 30 dumps scanned, 7 with duplicates, 41 aliased NIDs — all among
modules linked BY NAME (PSNCommon+PSNCore on six titles, libfmod+libfmodstudio 16 NIDs on one,
AkMotion/AkSoundEngine/AkVorbisHwAccelerator one NID three ways).
"""
import subprocess, pathlib, collections, sys

SELF_DUMP = str(pathlib.Path(__file__).resolve().parents[2] / "build-linux" / "self_dump")
def _default_root():
    """Walk up looking for a directory that actually contains dumps.

    parents[3] is wrong from a git worktree: the tree lives under .claude/worktrees/<name>/ while the
    dumps sit in the main checkout. Searching upward finds either. If nothing is found we say so and
    exit non-zero rather than printing a confident "0 dumps scanned" — a census that reports nothing
    because it looked in the wrong place is indistinguishable from one that found nothing.
    """
    here = pathlib.Path(__file__).resolve()
    for cand in list(here.parents):
        if any(cand.glob("PPSA*-app0")): return cand
    return None

if len(sys.argv) > 1:
    ROOT = pathlib.Path(sys.argv[1])
else:
    ROOT = _default_root()
    if ROOT is None:
        sys.exit("dup_exports: no PPSA*-app0 dumps found above this file; pass a dump root explicitly")
if not any(ROOT.glob("PPSA*-app0")):
    sys.exit(f"dup_exports: {ROOT} contains no PPSA*-app0 dumps")

# The fixed named list prosper links (boot_program.cpp), plus the eboot itself.
LINKED = ["eboot.bin",
          "Media/Modules/Il2CppUserAssemblies.prx", "Media/Modules/Il2cppUserAssemblies.prx",
          "Media/Modules/PS5Util.prx",
          "Media/Plugins/AkMotion.prx", "Media/Plugins/AkSoundEngine.prx",
          "Media/Plugins/AkVorbisHwAccelerator.prx", "Media/Plugins/CommonDialog.prx",
          "Media/Plugins/libfmod.prx", "Media/Plugins/libfmodstudio.prx",
          "Media/Plugins/PSNCommon.prx", "Media/Plugins/PSNCore.prx", "Media/Plugins/PSN.prx",
          "Media/Plugins/SaveData.prx",
          "sce_module/libc.prx", "sce_module/libSceNpCppWebApi.prx"]

def exports(path):
    try:
        r = subprocess.run([SELF_DUMP, str(path), "--symbols"], capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    if r.returncode != 0: return None
    out, seen = set(), False
    for line in r.stdout.splitlines():
        if line.startswith("[EXPORTS]"): seen = True; continue
        if not seen: continue
        p = line.split()
        if len(p) >= 4 and p[0].startswith("0x"): out.add(p[2])
    return out

def ci_find(root, rel):
    """Dumps vary in case; resolve case-insensitively like the loader does."""
    cur = root
    for part in rel.split("/"):
        if not cur.is_dir(): return None
        m = [c for c in cur.iterdir() if c.name.lower() == part.lower()]
        if not m: return None
        cur = m[0]
    return cur if cur.is_file() else None

total_dumps = total_dups = dumps_with_dups = 0
for dump in sorted(ROOT.glob("PPSA*-app0")):
    mods, seen_files = {}, set()
    for rel in LINKED:
        f = ci_find(dump, rel)
        if not f: continue
        # Dedupe by RESOLVED FILE. prosper's list carries two spellings of Il2CppUserAssemblies, and a
        # case-insensitive resolver maps both to the same file — counting it as 296 self-duplicates.
        # That artifact was ~95% of the first census run; the instrument, not the subject.
        key = f.resolve()
        if key in seen_files: continue
        seen_files.add(key)
        e = exports(f)
        if e is None:
            print(f"  !! {dump.name}: self_dump FAILED on {rel} — census incomplete for this title",
                  file=sys.stderr)
            continue
        # An empty set is a real answer (18 eboots export nothing); None is a tool failure. Conflating
        # them would let a broken self_dump report a title as clean.
        mods[rel] = e
    if not mods: continue
    total_dumps += 1
    owners = collections.defaultdict(list)
    for rel, nids in mods.items():
        for n in nids: owners[n].append(rel)
    dups = {n: o for n, o in owners.items() if len(o) > 1}
    if dups:
        dumps_with_dups += 1
        total_dups += len(dups)
        pairs = collections.Counter(tuple(sorted(o)) for o in dups.values())
        print(f"\n=== {dump.name}: {len(dups)} duplicated NID(s) across {len(mods)} linked modules")
        for combo, cnt in pairs.most_common():
            print(f"    {cnt:5d} NIDs shared by: {' + '.join(combo)}")
            ex = [n for n, o in dups.items() if tuple(sorted(o)) == combo][:3]
            print(f"          e.g. {', '.join(ex)}")
    else:
        print(f"{dump.name}: clean ({len(mods)} linked modules)")
print(f"\n==== {total_dumps} dumps scanned, {dumps_with_dups} with duplicates, {total_dups} duplicated NIDs total ====")
