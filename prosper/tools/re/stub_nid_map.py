#!/usr/bin/env python3
"""stub_nid_map — which imported NID does the stub at address X resolve to?

This is the INVERSE of `nid_gate_scan.py`. That tool starts from a NID and finds the call sites that
reach it; this one starts from the module's code and names, for every PLT-ish stub in it, the import
behind it. The two questions look interchangeable and are not: reading a guest call chain goes the
other way round. A disassembled function shows `call 0x13d810` and nothing else — the target is a
`jmp *[rip+d]` thunk with no symbol, `objdump`/`readelf` cannot decode the Sony relocation that
binds its slot, and the NID never appears as a literal anywhere near the call. So the only route
from "this function calls four things in sequence" to "it calls CreateHandle, GetGameInfo, then
DestroyHandle" was to guess a NID, scan for it, and check whether a reported site matched the
address you were staring at — once per candidate, against a library that may export hundreds.

    NID -> dynsym index -> JMPREL jump slot -> `jmp *[rip+d]` stub -> stub entry point

is exactly the resolution `nid_gate_scan` already performs, so this walks it once for *every*
import and inverts the result. It shares that module's Image/flatten/scan_code, which means a
module this can read is a module that tool can read, and neither can silently disagree with the
other about what is imported.

Worked example (#2186). *Earthion*'s only `sceNpTrophy2GetGroupInfoArray` call site sits in a
six-step init chain whose steps are anonymous addresses. Mapping the stubs named step 4 as
`sceNpTrophy2GetGameInfo` — already answering an error — which proved the call site under
investigation is unreachable, and turned a boot A/B that would have reported a misleading null into
a positive control that fires.

Usage:
    stub_nid_map.py <module|app0-dir> [--names <PS5-libs-dir>] [--addr 0x13d810 ...]

    <module>    a SELF/PRX/eboot.bin (flattened in memory) or an already-flat ELF. Given a
                directory, every `eboot.bin` and `*.prx`/`*.sprx` under it is mapped.
    --names     `../PS5-3.20_Libs` — symbolication only. An unknown NID is still listed, so a
                missing library dump costs labels, never coverage.
    --addr      print only these stub addresses. **An address with no row is reported explicitly
                as `NOT A STUB`** rather than omitted: a silent omission reads identically to
                "this call does not go through an import", which is the wrong conclusion to hand
                someone reading a call chain, and the one they cannot detect.

Output is `stub_va<TAB>NID<TAB>name<TAB>library`, ascending by address.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nid_gate_scan as G                                            # noqa: E402


def invert(img, xref):
    """{stub entry VA: (nid, sym_index)} for every import reached through a `jmp *[rip+d]` stub.

    Keyed by ADDRESS, deliberately. A single import owns more than one entry point whenever its
    stub carries a CET prologue — a caller then lands 4 bytes before the `jmp` — so a NID-keyed map
    would have to pick one and would answer "not a stub" for the address the caller actually names.
    The address is the unique thing a reader has in hand; the NID is what they are missing.

    An import whose slot is reached only by a direct `call *[rip+d]` has no stub and is correctly
    absent: there is no stub address to name.

    Split from `stub_map` so it can be driven with fakes — the inversion is the whole logic, and
    building a synthetic module with a dynsym, a JMPREL table and decodable stub bytes would test
    the fixture more than the function.
    """
    stubs_of_slot = {}
    for slot, refs in xref.items():
        for site, kind in refs:
            if kind == "jmp*":
                stubs_of_slot.setdefault(slot, []).append(site)

    out = {}
    for nid, idx in img.imported_nids():
        for slot in img.jump_slots(idx):
            for site in stubs_of_slot.get(slot, []):
                for entry in G.stub_entry_points(img, site):
                    out[entry] = (nid, idx)
    return out


def stub_map(path):
    """`invert()` over a module on disk (SELF/PRX/eboot flattened in memory, or a flat ELF)."""
    img = G.Image(G.flatten(path))
    return invert(img, G.scan_code(img))


def format_rows(smap, wanted, label, names):
    """Output lines for one module.

    `wanted` empty means "every stub, ascending". Otherwise only those addresses — and an address
    with no row yields an explicit `NOT A STUB` line. That branch is the reason this is a separate
    function: dropping the address silently is the obvious simplification, and it produces output
    in which "this call does not go through an import" and "you asked about the wrong address" are
    the same thing (namely nothing), with the reader having no way to tell which they got.
    """
    lines = []
    for va in (wanted or sorted(smap)):
        hit = smap.get(va)
        if hit is None:
            if wanted:
                lines.append("0x%x\t-\tNOT A STUB in %s\t-" % (va, label))
            continue
        nid, _idx = hit
        fn, lib = names.get(nid, ("?", "?"))
        lines.append("0x%x\t%s\t%s\t%s" % (va, nid, fn, lib))
    return lines


def modules_under(path):
    if os.path.isfile(path):
        return [path]
    found = []
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f == "eboot.bin" or f.endswith((".prx", ".sprx")):
                found.append(os.path.join(root, f))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("module")
    ap.add_argument("--names", help="PS5-3.20_Libs directory, for NID -> name symbolication")
    ap.add_argument("--addr", action="append", default=[],
                    help="only report these stub addresses (repeatable; hex 0x… or decimal)")
    args = ap.parse_args()

    names = G.load_nid_names(args.names) if args.names else {}
    wanted = [int(a, 0) for a in args.addr]

    rc = 0
    mods = modules_under(args.module)
    for mod in mods:
        try:
            smap = stub_map(mod)
        except Exception as exc:                                     # noqa: BLE001 — report, go on
            print("%s\tunreadable: %s" % (mod, exc), file=sys.stderr)
            rc = 2
            continue
        if len(mods) > 1:
            print("### %s" % mod)
        for line in format_rows(smap, wanted, os.path.basename(mod), names):
            print(line)
    return rc


if __name__ == "__main__":
    sys.exit(main())
