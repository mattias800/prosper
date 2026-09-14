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
import struct
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


# ---------------------------------------------------- imports patched out of the module in place

# A lazy PLT entry on these modules is 16 bytes: `jmp *[rip+disp32]` (6) then the lazy-binding tail
# `push <reloc index>` (5) and `jmp <resolver>` (5). Offsets of the tail within the entry.
PLT_PUSH_OFF = 6
PLT_JMP_OFF = 11
PLT_ENTRY = 16


def jmprel_index_to_symbol(img):
    """{reloc index: symbol index} over JMPREL, in table order.

    The reloc index is what a lazy PLT entry pushes before jumping to the resolver, so this is what
    ties an orphaned tail back to the import whose stub used to sit in front of it. `jump_slots`
    already walks this table; it just throws the index away because it is matching on the symbol.
    """
    out = {}
    for tv, sv in [(0x17, 0x2), (0x61000029, 0x6100002D)]:
        va, sz = img.tags.get(tv), img.tags.get(sv)
        if va is None or sz is None:
            continue
        base = img.foff(va)
        if base is None:
            continue
        for i in range(sz // 24):
            off = base + i * 24
            if off + 24 > len(img.raw):
                break
            _r_off, r_info, _r_add = struct.unpack_from("<QQq", img.raw, off)
            if (r_info & 0xFFFFFFFF) == G.JMP_SLOT:
                out[i] = r_info >> 32
    return out


def patched_scan(img, sym_of=None):
    """Import stubs whose `jmp *[rip+disp32]` head was overwritten, leaving the original tail.

    Returns (status, rows): status is "no-plt" | "uncalibrated" | "clean" | "patched",
    rows is [(entry VA, reloc index, symbol index or None, how)], ascending.

    The status exists because "I did not judge this module" and "I judged it clean" must
    not be the same answer -- the same reason `format_rows` prints NOT A STUB rather than
    omitting a row. A reader sweeping a corpus would otherwise count a refusal as a pass,
    and 14 modules in the local corpus are in exactly that state.

    WHY NOT "imported but has no stub": that is the obvious test and it over-reports. An import
    reached only by a direct `call *[rip+disp32]` legitimately has no stub (see `invert`), and so
    does one the module never calls. Neither is tampering, and a detector that cannot tell them
    apart would tell people their dump is modified when it is not.

    What IS evidence is a 16-byte entry that still carries `push <reloc index>; jmp <resolver>`
    while its leading `jmp *[rip+disp32]` is gone. That tail is lazy-binding code no compiler emits
    on its own and nothing else jumps to; its presence without a head means something wrote over the
    head in place. Observed on Uncharted (PPSA05684), whose libScePlayGo entries for
    scePlayGoGetLocus and scePlayGoOpen were replaced with short hand-written stubs -- so the calls
    never reached prosper at all, and the resulting assertion read exactly like an HLE defect (see
    the issue this function was written for).

    The resolver is not assumed: every candidate tail votes with its own `jmp` target and only the
    majority target is accepted as the module's resolver. That is what keeps an incidental
    `push imm32; jmp rel32` elsewhere in the text from being read as a PLT entry.
    """
    cand = []
    push_vas = []
    for seg_va, seg_off, seg_len, flags in img.segs:
        if not (flags & 1):          # PF_X only: the PLT lives in executable memory
            continue
        blob = img.raw[seg_off:seg_off + seg_len]
        start = 0
        while True:
            i = blob.find(b"\x68", start)
            if i < 0 or i + 10 > len(blob):
                break
            start = i + 1
            if blob[i + 5] != 0xE9 or i < PLT_PUSH_OFF:
                continue
            reloc = struct.unpack_from("<I", blob, i + 1)[0]
            rel = struct.unpack_from("<i", blob, i + 6)[0]
            entry_va = seg_va + i - PLT_PUSH_OFF
            resolver = seg_va + i + 10 + rel
            head_intact = blob[i - PLT_PUSH_OFF] == 0xFF and blob[i - PLT_PUSH_OFF + 1] == 0x25
            cand.append((entry_va, reloc, resolver, head_intact))
            push_vas.append(seg_va + i)

    if not cand:
        return "no-plt", []
    votes = {}
    for _va, _r, resolver, _ok in cand:
        votes[resolver] = votes.get(resolver, 0) + 1
    # No `if not votes` guard here: `votes` is built from `cand` and is empty exactly when `cand` is,
    # which the return above has already handled. A second guard would be dead code, and adding one
    # was how the previous attempt at this looked fixed while the suite still aborted on a mutation --
    # the thing that actually needed fixing was the test helper, not this function.
    resolver = max(votes, key=lambda k: votes[k])

    if sym_of is None:
        sym_of = jmprel_index_to_symbol(img)

    # THREE guards stand between a byte pattern and telling somebody their dump is tampered with.
    # Each closes a way a *clean* module can be accused, which is the expensive direction here; all
    # three were measured free on this corpus (61 eboots, 261 firmware .sprx and 726 game .prx: 0 grid
    # violations, 0 strays surviving the vote, one flagged module).
    #
    # (1) THE GRID. Entries are `PLT_ENTRY` apart with reloc indices running alongside, so every real
    #     entry shares one value of `va - PLT_ENTRY*reloc`. Take the MODE of that over the surviving
    #     candidates and drop anything off it. Fitting from the first and last candidate instead lets
    #     a single stray become an endpoint and silently disable the geometry pass entirely.
    accepted = [(va, reloc, ok) for va, reloc, res, ok in cand if res == resolver]
    bases = {}
    for va, reloc, _ok in accepted:
        key = va - PLT_ENTRY * reloc
        bases[key] = bases.get(key, 0) + 1
    base = max(bases, key=lambda k: bases[k])
    on_grid = [(va, reloc, ok) for va, reloc, ok in accepted if va - PLT_ENTRY * reloc == base]

    # (2) CALIBRATION. Every judgement here is "this entry does not look like the intact ones", so
    #     the module has to contain at least one intact entry to compare against. A PLT whose entries
    #     simply have a shape this code does not know -- an `endbr64` prologue shifting the `push`
    #     off byte 6, say -- would otherwise present as ENTIRELY patched, which is the loudest
    #     possible false accusation. No local module has such a PLT, and neither lld nor GNU ld emits
    #     one, but "unrecognised" and "overwritten" must not be the same answer.
    if not on_grid or not any(ok for _va, _r, ok in on_grid):
        return "uncalibrated", []

    out = []
    seen_reloc = {}
    for entry_va, reloc, head_intact in on_grid:
        seen_reloc[reloc] = entry_va
        if head_intact or reloc not in sym_of:
            continue
        out.append((entry_va, reloc, sym_of.get(reloc), "tail"))

    # Second signal, for the case the first cannot reach: a patcher that overwrote the TAIL as well
    # leaves no `push` to find, so such an entry is invisible above -- and on the dump this was
    # written for that was precisely the entry that broke the title, while five noisier ones were
    # caught. Any reloc inside the observed span with no entry on the grid had its whole entry
    # overwritten.
    #
    # The span bound keeps this honest: extrapolating past the observed entries would invent slots
    # for imports that may legitimately have none (one reached only by a direct `call *[rip+disp32]`),
    # which is the over-report this function exists to avoid.
    #
    # (3) OTHER-REGION CANDIDATES. A slot that holds a candidate which merely lost the resolver vote
    #     is a PLT entry belonging to a second region, not an overwritten one. Without this, a module
    #     with two lazy PLTs reports the losing region's interleaved slots as patched.
    #     Compared by the PUSH's address rather than the derived entry VA: an entry whose shape this
    #     code does not recognise carries its push at a different offset, so a VA comparison misses it
    #     and the slot is then reported as overwritten. Neither real whole-entry slot on the one
    #     flagged dump contains a 0x68 byte, so this costs nothing there.
    mine = set(seen_reloc.values())
    other_pushes = [pv for pv, (va, _r, _res, _ok) in zip(push_vas, cand) if va not in mine]
    if len(seen_reloc) >= 2:
        r0, r1 = min(seen_reloc), max(seen_reloc)
        for reloc in range(r0, r1 + 1):
            if reloc in seen_reloc or reloc not in sym_of:
                continue
            va = base + PLT_ENTRY * reloc
            if img.foff(va) is None:
                continue
            if any(va <= pv < va + PLT_ENTRY for pv in other_pushes):
                continue
            out.append((va, reloc, sym_of.get(reloc), "whole-entry"))
    rows = sorted(set(out))
    return ("patched" if rows else "clean"), rows


def patched_import_stubs(img, sym_of=None):
    """The rows from `patched_scan`, for callers that only care whether anything was found."""
    return patched_scan(img, sym_of)[1]


def format_patched(img, names, label, sym_of=None):
    """Report lines for one module, naming which of the four outcomes it got.

    A refusal must not print as a pass. `uncalibrated` means the module has a lazy PLT whose entry
    shape this code does not recognise, so it has nothing intact to compare against and declines to
    judge; a reader sweeping a corpus has to be able to count those separately from clean ones.
    """
    status, rows = patched_scan(img, sym_of)
    if status == "no-plt":
        # Two different things reach here, and only one of them is "nothing to check". A module with
        # no imports at all genuinely has nothing; a module whose imports are bound without a lazy
        # PLT (`-z now`) has plenty, and this analysis simply cannot see them -- 22 of the local
        # corpus's modules are in that state, PS5Util.prx among them with 7 jump slots. Calling that
        # "nothing to check" is the same error as printing a refusal as a pass.
        if jmprel_index_to_symbol(img) if sym_of is None else sym_of:
            return ["%s: NOT JUDGED -- this module binds imports without a lazy PLT, which this "
                    "analysis cannot inspect" % label]
        return ["%s: no imports bound through a PLT in this module -- nothing to check" % label]
    if status == "uncalibrated":
        return ["%s: NOT JUDGED -- lazy PLT found, but no entry in it has the expected "
                "`jmp *[rip+disp32]` head, so there is nothing intact to compare against" % label]
    if status == "clean":
        return ["%s: no import stub has been overwritten in place" % label]
    nid_of = {idx: nid for nid, idx in img.imported_nids()}
    lines = ["%s: %d import stub(s) OVERWRITTEN IN PLACE -- calls to these never reach the HLE layer"
             % (label, len(rows))]
    for va, reloc, sym, how in rows:
        nid = nid_of.get(sym) if sym is not None else None
        fn, lib = names.get(nid, ("?", "?")) if nid else ("?", "?")
        lines.append("0x%x\treloc=%d\t%s\t%s\t%s\t[%s]" % (va, reloc, nid or "-", fn, lib, how))
    return lines


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
    ap.add_argument("--patched", action="store_true",
                    help="instead of the stub map, report import stubs OVERWRITTEN IN PLACE -- a "
                         "third-party replacement patched into the module, which the module-path "
                         "policy cannot see because there is no extra module to refuse")
    args = ap.parse_args()

    names = G.load_nid_names(args.names) if args.names else {}
    wanted = [int(a, 0) for a in args.addr]
    if args.patched and wanted:
        # Refused rather than ignored. --addr asks "what is at this address"; --patched answers
        # "which entries were overwritten", and silently dropping the filter would hand back a
        # module-wide verdict to somebody who asked about three addresses.
        ap.error("--addr does not apply to --patched: --patched reports the whole module")

    rc = 0
    mods = modules_under(args.module)
    for mod in mods:
        try:
            if args.patched:
                img = G.Image(G.flatten(mod))
                lines = format_patched(img, names, os.path.basename(mod))
            else:
                lines = format_rows(stub_map(mod), wanted, os.path.basename(mod), names)
        except Exception as exc:                                     # noqa: BLE001 — report, go on
            print("%s\tunreadable: %s" % (mod, exc), file=sys.stderr)
            rc = 2
            continue
        if len(mods) > 1:
            print("### %s" % mod)
        for line in lines:
            print(line)
    return rc


if __name__ == "__main__":
    sys.exit(main())
