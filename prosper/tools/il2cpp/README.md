# IL2CPP symbolication for Unity/IL2CPP PS5 titles

Recover C# method names + addresses from an IL2CPP title so you can breakpoint
managed logic (scene load, boot state machines) during emulator debugging.

## Overview

These tools turn a PlayStation 5 SELF/PRX module into a form
[Il2CppDumper](https://github.com/Perfare/Il2CppDumper) accepts, then map addresses back to C#
method names — offline from a shell, or in-process while prosper runs (see *Runtime symbolication*
below).

    PRX module -> prx_to_elf.py -> flattened ELF -> Il2CppDumper -> script.json -> resolve.py -> method names

| Tool | Purpose | Input | Output |
| --- | --- | --- | --- |
| `prx_to_elf.py` | Flatten a PRX into a loadable ELF | `*.prx` | `*.elf` |
| `resolve.py` | Name an address, or emit prosper's runtime symbol table | `script.json` + addresses | Method names / `*.symtab` |

## Prerequisites

- Python 3, standard library only (`struct`, `json`, `bisect`, `re`) — no pip dependencies.
- Il2CppDumper and a dotnet runtime. It is not preinstalled; step 2 below fetches it.
- An **unencrypted** dump carrying `Media/Modules/` and `Media/Metadata/`. These tools do not
  decrypt anything: an encrypted PRX cannot be flattened, and that is out of scope by design.

## Recipe
1. Flatten the compiled-C# PRX (unencrypted in the dump) into a loadable ELF:

       python3 prx_to_elf.py <dump>/Media/Modules/Il2CppUserAssemblies.prx /tmp/asm.elf

2. Get Il2CppDumper if you don't have it (it is NOT preinstalled). Only a net8
   runtime ships in the WSL image, and no `unzip`, so grab the net7 build and
   force roll-forward:

       cd /tmp
       curl -sSL https://github.com/Perfare/Il2CppDumper/releases/download/v6.7.46/Il2CppDumper-net7-v6.7.46.zip -o d.zip
       python3 -c "import zipfile; zipfile.ZipFile('d.zip').extractall('dumper')"

3. Run it (needs dotnet-sdk). The ELF starts at vaddr 0 so it prompts
   "input il2cpp dump address" — answer `0` to force the registration auto-scan.
   Pipe the answers in (headless has no console; the tool still throws a harmless
   `Cannot read keys` at the very end AFTER all outputs are written — ignore it):

       cd /tmp/dumper && DOTNET_ROLL_FORWARD=LatestMajor \
         sh -c 'printf "0\n0\n0\n" | dotnet Il2CppDumper.dll /tmp/asm.elf \
           <dump>/Media/Metadata/global-metadata.dat /tmp/out'

4. `out/dump.cs` lists every method with a `// RVA: 0x...` comment; `out/script.json`
   has the machine-readable `{Address, Name}` list. The runtime address is
   `module load base + RVA`, and the base is not a guess: `boot_program.cpp:129` maps
   `Media/Modules/Il2cppUserAssemblies.prx` at `BOOT_IL2CPP` (`boot_program.hpp:21`,
   currently `0x440000000`), so RVA == script.json Address == a prosper
   `Il2cpp+0x<offset>` label. Read the constant rather than the literal — the eboot's
   base already moved once (#825) and every diagnostic that had hard-coded the old one
   went on printing plausible offsets that landed nowhere.
   Worked example, PPSA02664: `SCS.UserManagement.SCSUserManager$$DoFirstLogin` is at RVA
   `0x2140D0`, so it runs at `0x4402140D0`. (That RVA is title-specific — in PPSA24651 the
   same offset is 0xf0 into `SonicBloom.Koreo.Koreographer$$GetMusicSampleTime`. Both
   measured 2026-08-17 with Il2CppDumper v6.7.46.)

5. Symbolicate addresses / btrace chains with `resolve.py`:

       python3 resolve.py /tmp/out/script.json il+0x1764ce2 0x11e63c
       # il+0x1764ce2  ->  Unity.PSN.PS5.Aysnc.WorkerThread$$RunProc  (+0xa2)
       echo '[btrace] ... chain=il+0xde92e9,il+0xde9159' | python3 resolve.py /tmp/out/script.json -

   (`Aysnc` is not a typo in this README: that is how Unity's own PSN plugin spells the
   namespace. Measured 2026-08-17 with Il2CppDumper v6.7.46 — the only match for
   `WorkerThread$$RunProc` is `Unity.PSN.PS5.Aysnc.WorkerThread$$RunProc` in BOTH
   PPSA24651 and PPSA02664. This line used to read `Async`, which resolves to nothing.
   `0x1764ce2` is PPSA24651's RVA for it; in PPSA02664 that offset has no managed method.)

   NOTE: prosper's `[btrace]` validated unwinder must accept INDIRECT call sites
   (`0xFF` at v-2/-3/-6/-7), not just `0xE8` (call rel32) — IL2CPP dispatches
   managed methods indirectly, so an 0xE8-only filter drops every managed frame.

## Runtime symbolication — naming the method during the run (#2551)

Steps 1–5 are offline: they answer "what method is at this address" *after* the fact.
prosper can also do it **in-process**, so a guest backtrace names the C# method instead
of printing an address somebody has to go and correlate:

```
[app] guest backtrace: 0x440de92e9 (Il2cpp+0xde92e9 System.Threading.WaitHandle$$WaitOneNative+0xd9)
```

Two commands, both one-off per title:

```bash
python3 resolve.py /tmp/out/script.json --emit-symtab ~/PPSA24651.symtab   # from step 4's script.json
export PROSPER_IL2CPP_SYMBOLS=~/PPSA24651.symtab                          # then run prosper normally
```

`--emit-symtab` writes a flat `<hex-rva> <name>` table from resolve.py's **own** loader, and
`src/host/symbols/il2cpp_symbols.cpp` reads that. script.json itself is deliberately *not* parsed at
runtime — it is 36.8 MB for PPSA24651 against 5.6 MB for the symtab, and a second JSON parser
would be a re-derivation of a mapping resolve.py already gets right. The ctest case
`il2cpp_symtab_agreement` drives both CLIs over one synthesized corpus and requires them to agree
address-for-address.

**Coverage is exactly one function: `describe_code_address()`** (`src/host/image/exec_image_linux.cpp`,
`exec_image_win.cpp`). That is the shared guest-address label, so every consumer of it is
symbolicated at once — the app's `[app] guest backtrace` after a fault, `tools/screenshot`'s, and
the Windows `[thread-trace]`. Diagnostics that call `guest_module_name()`/`guest_module_offset()`
directly (`hle_agc`'s `ra=`, `hle_file`'s MISS attribution, `boot_trace`'s fault backtrace) still
print bare offsets; pipe those through `resolve.py` as before. Extending them is a one-line change
each and was left undone deliberately rather than touched blind.

### What a failed resolution looks like — four distinct answers, never silence

A resolver that answers "nothing" when it means "I did not run" is this project's most expensive
recurring defect, so the four outcomes are spelled differently:

| situation | label |
| --- | --- |
| `PROSPER_IL2CPP_SYMBOLS` unset | `Il2cpp+0xde92e9` — unchanged, no claim made |
| set, but the file could not be loaded | `Il2cpp+0xde92e9 <il2cpp-symbols-unavailable>` |
| loaded, no managed method covers the address | `Il2cpp+0xde92e9 <no-managed-method>` |
| loaded and covered | `Il2cpp+0xde92e9 System.Threading.WaitHandle$$WaitOneNative+0xd9` |

Every load also prints one line to stderr, on success **and** failure, so "how many symbols do I
actually have" is never inferred from the absence of names:

```
[il2cpp-sym] loaded 87851 symbols from ~/PPSA24651.symtab (window=0x8000)
[il2cpp-sym] NOT LOADED from ~/typo.symtab: cannot open file (guest addresses stay unsymbolicated)
```

Pointing the variable at a raw `script.json` is **refused**, naming the header it wanted, rather
than being read as a table with zero symbols.

`<no-managed-method>` is a real and common answer, not a failure. Of 36 live IL2CPP addresses
sampled from a running PPSA24651 boot, 26 were in the il2cpp *runtime* — the VM's own native C++,
compiled into the same PRX **below** the managed-method region, since PPSA24651's lowest managed
method is at RVA `0x1e3e00` — and `resolve.py` reports exactly the same thing for them.

Read that sample as what it is: a live sanity check, not the agreement evidence. Those 26 sit below
the first record, so neither implementation could have named anything there and no disagreement was
expressible; only the remaining **10** addresses were in the covered region at all. The evidence
that the two lookups agree is the `il2cpp_symtab_agreement` ctest case above, which synthesizes its
own corpus, probes 656 addresses, checks that the probe set contains **both** outcomes before
believing the agreement, and carries a planted-disagreement arm proving the comparator can report
one.

### Limits

- **One module.** Only the IL2CPP aperture is symbolicated; the Unity player eboot and the system
  PRXs are stripped C++ and stay bare offsets. Use `tools/guest_bt/` when you want a whole *stack*
  rather than a single address — it unwinds guest threads through the HLE stub boundary and already
  consumes a `script.json` for managed names.
- **Nearest-preceding, windowed at `0x8000`.** script.json gives method starts, never lengths, so an
  address more than 32 KiB past a start is reported as `<no-managed-method>` rather than attributed.
  The window is declared in the symtab header and read from it, so resolve.py remains the one place
  it is defined.
- **The symtab is derived from the gitignored dump — do not commit one** (same rule as `dump.cs` /
  `script.json`). `tests/data/il2cpp_symtab_fixture.symtab` is hand-written and contains no dumper
  output.
- IL2CPP layout is version-dependent; a symtab is only valid for the exact module it was dumped
  from. Nothing detects a mismatched pair, so a stale symtab yields confident wrong names.

## Tool reference

### prx_to_elf.py

    python3 prx_to_elf.py <in.prx> <out.elf> [--sections]

Flattens an unencrypted PS5 SELF/PRX into an `ET_DYN` ELF whose `p_offset == p_vaddr`, which is
what makes an RVA and a file offset the same number downstream.

1. Read the SELF segment table; data segments are the ones carrying flag bit `0x800`, and a
   segment's program-header index is `flags >> 20`.
2. Copy each `PT_LOAD`'s bytes to `p_vaddr` in a flat buffer, then copy the embedded ELF's 64-byte
   header over the front.
3. Rewrite `e_type = ET_DYN`, `e_phoff = 0x40`, and clear `e_shoff` / `e_shnum` / `e_shstrndx`.
4. Rewrite every program header with `p_offset := p_vaddr`.

| Field | In the module | Flattened | Why |
| --- | --- | --- | --- |
| `e_type` | varies | `ET_DYN` (3) | position-independent image |
| `e_phoff` | embedded offset | `0x40` | header is copied to the front |
| `e_shoff` | past-EOF offset | `0` | the table it describes is not in this file |
| `e_shnum` | `0`, or `43`–`48` | `0` | ditto |
| `e_shstrndx` | `41`–`46` | `SHN_UNDEF` (0) | ditto |

Those stale ranges are what stripped PS5 modules actually carry, measured across 22 title modules.
**All three must be cleared together** — either survivor makes binutils reject the whole file with
`file format not recognized`. `--sections` is covered in the next section.

### resolve.py

    python3 resolve.py <script.json> [--base <addr>] [--emit-symtab <out>] <address...>
    python3 resolve.py <script.json> -            # read addresses from stdin

| Option | Effect |
| --- | --- |
| `--base <addr>` | Subtract `<addr>` from bare-hex inputs, so absolute runtime addresses resolve |
| `--emit-symtab <out>` | Write the flat runtime symbol table prosper reads via `PROSPER_IL2CPP_SYMBOLS` |
| `-` | Read tokens from stdin; scrapes `il+0x…` / `eb+0x…` / `0x…` out of a pasted log line |

| Address form | Example | Read as |
| --- | --- | --- |
| bare hex | `0x2140d0` | an RVA, or an absolute address when `--base` is given |
| `il+0x…` | `il+0x1764ce2` | an RVA in the IL2CPP module (already what `[btrace]` prints) |
| `eb+0x…` | `eb+0xada254` | the Unity eboot — reported as native, never guessed at |

Lookup is **nearest-preceding within `NEAREST_WINDOW` (`0x8000`)**: sort the `script.json` records by
address, binary-search for the last method start at or below the query, and report it only if the
query is strictly less than `0x8000` past that start. script.json gives starts but never lengths, so
some bound is unavoidable; anything further out is reported as no managed method rather than
attributed to a method it is probably not in. The window is written into the emitted symtab header
so prosper reads it from here instead of keeping a second copy that can drift.

## What binutils can and cannot do with the flattened ELF

The image has program headers but **no section header table**, so `e_shoff`, `e_shnum` and
`e_shstrndx` are all zero (`prx_to_elf.py`; regression: `test_prx_to_elf.py`, ctest
`il2cpp_prx_to_elf_header`). That is enough for binutils to *accept* the file — `objdump -f`,
`objdump -p` and `objdump -T` all work, and `-T` is genuinely useful: it lists the module's Sony
NID dynamic symbols. It is **not** enough for `objdump -d`, which disassembles sections and finds
none.

**`--sections` fixes that (#2154).** It synthesizes one section per `PT_LOAD` plus a `.shstrtab`,
with `sh_addr == sh_offset == p_vaddr` — which follows directly from the flattening, so nothing is
derived or guessed; each section is the segment restated in the form objdump reads. `objdump -d`
then disassembles with real virtual addresses and resolves call targets, instead of the old
`objdump -D -b binary -m i386:x86-64 --start-address=…` workaround that throws every
program-header-derived address away.

```
python3 prx_to_elf.py <module>.prx out.elf --sections
objdump -d --start-address=0x2140d0 --stop-address=0x214100 out.elf
```

It is **opt-in, and the default output is byte-identical** to what it was — verified on
`Il2cppUserAssemblies.prx` with `cmp`, and asserted by `test_prx_to_elf.py`. Il2CppDumper reads
program headers and the dynamic segment and does not need sections, but its ELF reader does consult
them on some paths and that has not been re-run end to end, so the dump path is left untouched. Use
`--sections` for disassembly; leave it off for a dump.

### Ruled out
- **"binutils also needs `e_ident[EI_OSABI]`/`[EI_ABIVERSION]` cleared" (#2016's own note) — false.**
  binutils 2.46 accepts the flattened image with the module's `0x09 0x02` (FreeBSD, ABI 2) intact,
  as long as `e_shnum` **and** `e_shstrndx` are both 0. A/B over all four combinations on the
  `PPSA24651` IL2CPP module: `(shnum=14, shstrndx=43)` REJECT, `(0, 43)` REJECT, `(14, 0)` REJECT,
  `(0, 0)` ACCEPT — and `(0, 0)` with OSABI cleared is also ACCEPT, i.e. OSABI changes nothing.
  Zeroing only `e_shnum`, as #2016 suggested, would not have made binutils accept the file. No
  `--gnu-compat` flag is needed (#2016 / PR #2155).

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `objdump`: `file format not recognized` | Reading the raw `.prx`, whose stale `e_shoff`/`e_shnum`/`e_shstrndx` describe a table that is not there | Flatten it with `prx_to_elf.py` first |
| `objdump -d` prints nothing | No section header table in the default output — it disassembles sections and finds none | Regenerate with `--sections` |
| Il2CppDumper throws `Cannot read keys` | Headless run with no console, thrown after every output is written | Ignore it; check `/tmp/out` for `dump.cs` and `script.json` |
| Il2CppDumper finds no methods | Wrong or mismatched `global-metadata.dat`, or a dumper too old for this title's IL2CPP generation | Check the metadata path; try a newer Il2CppDumper release |
| Every address resolves to nothing | Absolute addresses fed in without `--base`, so each one lands far past the last method | Pass `--base 0x440000000`, or subtract the base yourself and feed RVAs |
| A known-good method reports no managed method | The query is more than `0x8000` past the nearest start, or it is genuinely in the il2cpp runtime's own native C++ below the managed region | Confirm against `dump.cs`; see *Limits* above |
| Runtime symbolication prints `<il2cpp-symbols-unavailable>` | `PROSPER_IL2CPP_SYMBOLS` is set but the file could not be read, or points at a raw `script.json` | Read the `[il2cpp-sym]` stderr line — it names the reason; regenerate with `--emit-symtab` |

Two quick checks that the flattened ELF is sound:

    objdump -f out.elf     # ELF64, ET_DYN
    objdump -p out.elf     # program headers, with p_offset == p_vaddr
    python3 -c "import json;print(len(json.load(open('/tmp/out/script.json'))['ScriptMethod']),'methods')"

## Testing

    python3 test_prx_to_elf.py         # ctest: il2cpp_prx_to_elf_header
    python3 test_il2cpp_tools.py       # ctest: il2cpp_resolve_lookup
    python3 test_symtab_agreement.py   # ctest: il2cpp_symtab_agreement

| Suite | What it pins |
| --- | --- |
| `test_prx_to_elf.py` | The written ELF header read back at its gABI offsets, `--sections` synthesis, and byte-identity of the default output |
| `test_il2cpp_tools.py` | `resolve.py`'s loader and nearest-preceding lookup: exact and interior hits, both sides of the `0x8000` window edge, empty and field-missing `script.json` |
| `test_symtab_agreement.py` | `--emit-symtab` against prosper's in-process resolver, address-for-address, with a planted-disagreement arm |

All three are pure Python over synthesized fixtures — no game dump, no build artifacts, no network.

## gdb caveat (prosper)
prosper installs a SIGSEGV handler for lazy memory commit. Under gdb, ALWAYS use

    handle SIGSEGV SIGILL SIGBUS nostop noprint pass

so recoverable guest faults reach prosper's handler. Otherwise gdb stops on
normal lazy-commit faults and corrupts the run — a healthy default boot then
appears to "crash" at a null deref when it actually does not.

Do NOT commit dumper output (dump.cs / script.json / il2cpp.h) — it is derived
from the gitignored game dump.
