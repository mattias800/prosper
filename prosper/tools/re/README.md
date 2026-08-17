# Guest binary reference workflow

Use these tools to map an unsymbolicated guest address without running a debugger. All addresses printed
by the tools are image-relative; add the module's load base only when comparing them with a live trace.

## Find an exported function

PS5 dynamic symbols use NIDs rather than plaintext names. Once a name has been hashed with
`prosper::nid_hash` (or obtained from a runtime lookup), query its module directly:

```bash
./build-linux/self_dump /path/to/module.prx --find-symbol vXRp9zVGPzU
```

The result distinguishes imports from exports and prints the export RVA. `--symbols` prints every import
NID and every defined export with its RVA, size, module, and library context.

## Find who calls an IMPORTED Sony function

This is the question that starts most HLE contract work — "who calls this, and what does the guest do
with the result?" — and it is how #1592 was root-caused: the guest's own `test eax,eax` / `jns` around
`scePadGetHandle` proved the contract prosper was violating.

A PS5 module never calls an import directly. The call goes to a PLT stub, the stub jumps through a GOT
slot, and the loader fills that slot at bind time. `self_dump --import-slots` prints the slot
(`DT_JMPREL`/`DT_RELA` `r_offset`), which used to be hand-parsed out of the relocation table:

```bash
./build-linux/self_dump <DUMP_ROOT>/PPSA20052-app0/eboot.bin \
    --import-slots --names ../PS5-3.20_Libs | grep scePadGetHandle
```

```text
0x000007f9b18 u1GRHp+oWoY  libScePad    scePadGetHandle    JUMP_SLOT
```

Then flatten the module and walk the two references out from that slot. **The slot's only direct
reference is the PLT stub** — the callers reference the *stub*, so it is two `xref` queries, not one:

```bash
python3 tools/il2cpp/prx_to_elf.py <DUMP_ROOT>/PPSA20052-app0/eboot.bin ~/work/eboot.elf
python3 tools/re/xref.py ~/work/eboot.elf to 0x7f9b18      # slot  -> PLT stub
python3 tools/re/xref.py ~/work/eboot.elf to 0x60c8e0      # stub  -> the call sites
```

```text
code references to 0x7f9b18: 1 (0 write, 0 read, 0 address-taken)
   [x ] jmp*    at 0x60c8e0  (in func 0x60c3e0)            # the PLT stub

code references to 0x60c8e0: 9 (0 write, 0 read, 0 address-taken)
   [x ] call    at 0x7fc74  (in func 0x7fa30)
   [x ] call    at 0x7fd5c  (in func 0x7fa30)              # the #1592 site
   [x ] call    at 0x800d9  (in func 0x7fa30)
   ...
```

`tools/re/edis.py` then disassembles a call site to read the guest's own test of the return value.
This is the worked known answer for `--import-slots`: `0x7fd5c`, `0x800d9` and the containing function
`0x7fa30` are exactly what #1592 recorded by hand.

Two things to know before believing an answer here. **`--import-slots` reports `GLOB_DAT` and `64`
relocations as well as `JUMP_SLOT`** — those are the same function address stored in a vtable or a
static initialiser rather than the call path, so a busy title lists several times more slots than it has
imports; `grep JUMP_SLOT` when you want the call site. And a **zero from `--import-slots` always states
its own cause and exits 3**, so "no output" is never ambiguous between "no slots" and "nothing parsed".

## Find readers and writers of a slot

Flatten the SELF/PRX first, then query the address with `xref.py`:

```bash
python3 tools/il2cpp/prx_to_elf.py /path/to/eboot.bin /tmp/eboot.elf
python3 tools/re/xref.py /tmp/eboot.elf to 0x2031d10
```

The `to` query reports Sony relative-relocation data pointers, direct calls, RIP-relative loads/stores,
indirect calls/jumps, and the RIP-relative **immediate byte/dword stores and byte compares**
(`storeb` = `c6 05` mov-byte-imm, `stored` = `c7 05` mov-dword-imm, `cmpb` = `80 3d` cmp-byte-imm)
through the requested slot. The immediate forms are how the *writer* of a 1-byte state flag is found —
it is not a register store, so `objdump`/`readelf` cannot name it. For example, the Bendy Agc
suspend-point SAFE flag (`#1195`) is set by three `storeb` sites that a plain reg-store scan misses:

```text
python3 tools/re/xref.py /tmp/eboot.elf to 0x20698a5
   storeb at 0x109a6e4  (in func 0x109a680)   # mov BYTE [rip+..],1
   cmpb   at 0x1530385  (in func 0x1530320)   # the watchdog's flag test
```

`from ADDRESS` lists references made by the function window containing that address; `reloc ADDRESS`
restricts output to relative data relocations.

### How to read a zero — and the CLI contract that makes a zero readable

`xref.py`'s arguments are **positional**: `xref.py <module.elf> {to|from|reloc|imm} <argument>`.
There are no option flags. `xref.py --help` prints that usage; anything that is not one of the four
modes is **refused**, loudly and with a non-zero exit.

That is a contract, not a nicety, because the one answer this tool must never give by accident is a
confident zero. `xref.py <elf> --addr 0x…` used to clear the argument-count check, parse the module,
decode half a million reference sites, then fall off the end of the mode chain and exit **0 with no
output at all** — indistinguishable from "nothing references this address", which is a conclusion
strong enough to redirect an investigation (#2399; the reporter fell back to a hand-rolled byte scan
and had to retract its false positives, #2396).

So the exit status means exactly one of two things:

| exit | meaning |
| --- | --- |
| `0` | the query ran; the printed counts **are** the answer, zero included |
| `2` | refused — nothing was searched, and no number printed is a result |

Every query prints, before its answer, how many references the decoder found **across the whole
module**. That number is what separates "this address has no references" from "this run found
nothing at all", and it is printed on hits and misses alike so a saved transcript carries its own
validity check. A zero result says so on its own line, and an address lying outside every `PT_LOAD`
(checked against `p_memsz`, so a `.bss` flag byte is *inside*) is called out as the likeliest cause —
usually a runtime address nobody rebased to an image-relative one (#1659).

Two zeros are still worth nothing until you have done one more thing: a string of 22 bytes or fewer
needs `imm` (see below), and an address you cannot find at all needs the load base subtracted.

## Find who builds a SHORT string — `to` cannot, and its zero is misleading

A short string may have **no address to reference**. Clang materialises a `std::string` built from a
literal of 22 bytes or fewer directly into the small-buffer with `movabs`/`mov` **immediates**, and the
`.rodata` copy of that literal is then left with zero references of every kind — no RIP-relative `lea`,
no relocation, nothing. `to <literal VA>` answers **0** for a string the guest asks for on every boot,
which reads as "the guest cannot be requesting this". Use `imm` for that question:

```bash
python3 tools/re/xref.py /tmp/eboot.elf imm 'ui/ui_startup'
```

```text
inline-immediate constructions of 'ui/ui_startup' (13 bytes): 1
   0x587628..0x587632  (in func 0x5875b0)  0x587628[0:8], 0x587632[5:13]
literal copies in non-executable segments: 1
   0xdd593b: 0 code refs, 0 data relocations
```

The two sites are the overlapping 8-byte immediates the compiler emits for a 13-byte string
(`[0:8]` then `[5:13]`), and the second line is the trap itself: the literal is unreferenced, so every
other query in this file reports nothing for it. Sonic Origins (`PPSA05325`, #1905) is the worked
example — the package is requested on every boot from exactly this one site.

The report is deliberately conservative. Only 8- and 4-byte windows form a candidate, at most a 3-byte
tail may be completed by a smaller store, and matches cluster only while consecutive hits are no more
than ~96 bytes apart. Single-byte matches are constant in a dense instruction stream, so admitting them
into the cover would let a stretch of code "build" any string — on this eboot that turns the one real
answer above into **96**. A needle shorter than 4 bytes is **refused** rather than answered, because that
floor cannot exist below the needle's own length. A string that exists only as a referenced literal
correctly reports **0** constructions — check both numbers, not either alone.

Two precision limits are visible in the output and are not designed away. Clustering is single-linkage
over *consecutive* matches, so a cluster can span more than ~96 bytes overall — the printed
`first..last` range is what bounds it. And a `<= 3`-byte tail window is accepted anywhere inside the
cluster, with no positional relation to the strong stores, so a tail-completed hit is only as precise as
its 4- and 8-byte windows; the `[lo:hi]` sizes in the output are how you tell the two cases apart.

For The Messenger, the example above identifies both sides of Unity's runtime IL2CPP API table:

```text
store at eboot+0x148276c   # writes dlsym("il2cpp_init")
call* at eboot+0x1473a6f  # invokes the populated slot
```

This slot is zero in the flat image and has no ELF relocation. Treating every indirect RIP-relative call
as a static GOT import would send the investigation to the wrong layer.

## Disassemble guest code at an address

Once `xref.py` (or a live trace) points at an interesting address, read the instructions there with
`edis.py`. The flattened module ELFs have no section header table, so `objdump -d`/`readelf` reject them
outright; `edis.py` resolves the byte at a guest VADDR through the module's PT_LOAD program headers and
disassembles it in Intel syntax with the real guest addresses:

```bash
python3 tools/il2cpp/prx_to_elf.py /path/to/eboot.bin /tmp/eboot.elf
python3 tools/re/edis.py /tmp/eboot.elf 0x109a680 0x70
```

```text
 109a6e4:  c6 05 ba f1 fc 00 01   mov  BYTE PTR [rip+0xfcf1ba],0x1   # 0x20698a5
 109a6eb:  89 05 af f1 fc 00      mov  DWORD PTR [rip+0xfcf1af],eax  # 0x20698a0
```

objdump computes the RIP-relative targets (the `# 0x…` comments), so `edis.py` shows exactly what the
code at a site does. It complements `xref.py to <slot>`, which lists the *sites* that reference a slot
(including the `storeb`/`stored`/`cmpb` immediate byte/dword store + compare forms): find the writers with
`xref.py`, then read each one with `edis.py`. Unlike `tools/dbg/dis.sh`, which disassembles a pre-extracted
`/tmp/text.bin` for one title, `edis.py` takes any flattened module ELF directly, so the same command works
for every title. Requires `objdump` (binutils) on PATH.

## `nid_gate_scan.py` — does a title *branch on* an imported function's return value?

Before changing what an HLE handler returns, the question that bounds the blast radius is not "who
imports this NID?" — it is "who **inspects the result**, and how?" Those are very different sets: a
title that calls the import and drops `eax` cannot be affected by any answer.

```bash
# does this dump gate on sceSysmoduleIsLoaded's answer, and against which errno?
python3 tools/re/nid_gate_scan.py <DUMP_ROOT>/PPSA04263-app0 --nid fMP5NHUOaMk --const 0x805a1001 -v
```

```text
    call at 0x197a22a id=0xb4     -> const     cmp eax,0x805a1001
    call at 0x25e3dc4 id=0xa8     -> nonzero   test eax,eax
eboot.bin                                                ok const=2 nonzero=14
```

It resolves NID → dynsym index → JMPREL jump slot → the `jmp *[rip+d]` stub → every `call rel32`
that reaches it, then disassembles the bytes after each call and classifies what happens to the
result. Pass a module or a whole `app0` directory (every `eboot.bin` and `*.prx`/`*.sprx` under it).
Flattening happens in memory, so there is no `prx_to_elf.py` step.

| bucket | meaning |
|---|---|
| `const` | some reachable path compares the result against `--const` |
| `nonzero` | gated on zero/non-zero, and the value is then dead on **every** reachable path — the only bucket that says this site cannot tell one non-zero answer from another |
| `gate-open` | gated, but an arm left the scan's reach (returned, spilled, indirect branch, unmapped, budget) — **not** cleared; read it by hand |
| `alu-gate` | folds the result into flags (`and eax,mask` …) and branches on that |
| `other-cmp` | compares against some other immediate |
| `forward` | never gated here and still live — returned, spilled, or tail-jumped; read it by hand |
| `ignored` | dead before any read: this call site cannot be affected |
| `undecodable` | objdump could not decode the window — a **void** sample, not a negative one |

**The error arm is read.** The scan forks at every branch the value survives and explores both
successors, so a site that tests the result generically and *then* const-compares it inside its error
arm lands in `const`. That case used to land in `nonzero` — indistinguishable from a site that cannot
care — which is why `nonzero` now carries the stronger meaning above and `gate-open` exists to hold
everything the walk could not finish. `--no-follow-arms` restores the old single-window walk, and is
there only to reproduce numbers published before this existed; it under-reports `const`, measurably
(PPSA08804's `sceSaveDataTransferringMountPs4` site: `nonzero` without, `const` with).

**Read the output as a bound, not a census.** It classifies the *compare*, never the branch target,
so it tells you which titles *can* change behaviour, not which do; site discovery follows one stub
level and one call deep, so a count is a lower bound; and the walk is intra-module, so a value handed
to a caller is `gate-open` rather than cleared. `gate-open=0 forward=0 undecodable=0` across a corpus
is worth quoting for the opposite reason — together they say every site was resolved.

The classifier tracks the result as a small register taint set rather than watching `eax` alone,
because the ordinary compiler output is `mov ecx,eax; xor eax,eax; test ecx,ecx` — the `xor` is the
*enclosing* function's return value, and an `eax`-only reader therefore reports a live gate as
`ignored`, i.e. it errs in the reassuring direction. That, an unconditional `jmp` being walked
through into the next basic block, and a mask-test being called `ignored` are all regression cases in
`test_nid_gate_scan.py` (ctest: `re_nid_gate_classifier`). Set `TMPDIR` to keep its scratch file off
`/tmp`.

Requires **GNU binutils** `objdump`: it disassembles a raw blob with `-b binary`, which LLVM's
`objdump` does not support. That matters on macOS, where Xcode's LLVM build *is* `objdump` and answers
`--version` happily while rejecting every actual decode — so the tool probes with a real one-byte
disassembly rather than trusting the binary's presence, accepts Homebrew's `gobjdump`, and honours
`$OBJDUMP`. The ctest skips cleanly when no capable objdump exists.

### `--all-nids` — which Sony answers does this title depend on *at all*?

`--nid` asks about one function. `--all-nids` inverts the question and classifies **every** import in
one pass, which is the bound a *registered-but-mismodelled return value* investigation needs and
cannot get any other way. An absence check (the `PROSPER_PROGRESS_UNIMPL` table) cannot see a handler
that is registered and answers wrongly — the call happens, so nothing is missing. A runtime
return-value histogram says what prosper returned, not whether the guest looked. This says which call
sites *can* be affected by an answer, so the ones that cannot are struck off before any boot.

```bash
python3 tools/re/nid_gate_scan.py <DUMP_ROOT>/PPSA05325-app0/eboot.bin \
    --all-nids --names ../PS5-3.20_Libs --min-gated 1
```

Rows sort by gated call sites descending, so the head of the table is libc, and the trailing `#`
block is the part to read first:

```text
Ovb2dSJOAuE  strcmp                libSceLibcInternal  sites=981 gated=979 gate-open=66 nonzero=911 …
…
fMP5NHUOaMk  sceSysmoduleIsLoaded  libSceSysmodule     sites=5   gated=5   const=1 gate-open=1 nonzero=3
…
# <path>: 536 imported NIDs are called; 254 shown at --min-gated=1
#   254 gated, 192 ignored-only (cannot matter), 90 neither (no gate, and >=1 site not cleared)
#   241 rows carry >=1 site the scan could not resolve (gate-open/forward/undecodable) -- read
#   those by hand
#   site buckets: alu-gate=517 const=1 forward=2013 gate-open=1053 ignored=18270 nonzero=2133
#                 other-cmp=313 undecodable=8
```

**Not-gated is not the same as cleared, and neither is gated.** `ignored` is the only bucket that
means an answer cannot matter at that site, and `nonzero` the only one that means it cannot matter
*which* non-zero answer; `forward` and `gate-open` explicitly need a look by hand and `undecodable`
is a void sample. A two-way "called / gated" split invites the reading that everything below the cut
is struck off — and the `gate-open` rows show the same trap above the cut, since a row can gate on
the result and still be unresolved. That is why the summary prints the not-resolved row count
separately from the not-gated one, and the site-bucket totals underneath both.

`--names` points at the gitignored PS5 firmware `genstub.py` library dump and is **symbolication
only** — an unlabelled NID is still scanned and still reported, so a missing dump costs names, never
coverage; a directory with no `sprx_dlsym` lines is rejected rather than silently producing an
all-`?` table. `--min-gated N` hides imports with fewer than N gated sites; the summary always states
the unfiltered total, so a filtered table cannot be mistaken for the whole one either.

The enumerator accepts a dynsym name only when exactly 11 characters precede the first `#`, searching
no further than the name's own NUL, because every relaxation of that rule fails silently — a C++
mangled name pulled in as a phantom "import" reports zero call sites and reads as a clean negative, a
dropped real import shrinks a census whose whole purpose is exhaustiveness, and an unclamped `#`
search lets a name with no `#` borrow the next strtab string's. `test_nid_gate_scan.py` pins all
three with a synthetic symbol table. Note the rule does **not** separate imports from exports (PS5
exports carry the same shape) — what removes exports is the JMPREL/call-site filter, since an export
has no jump slot bound to it.

## `stub_nid_map.py` — which import does the stub at address X resolve to?

The inverse of `nid_gate_scan.py`, and the direction you need when **reading a call chain** rather
than chasing one NID. A disassembled guest function shows `call 0x13d810` and nothing more: the
target is an unsymbolicated `jmp *[rip+d]` thunk, `objdump`/`readelf` cannot decode the Sony
relocation binding its slot, and the NID appears nowhere near the call. Without this, going from
"this function calls four things in sequence" to their names means guessing a NID, scanning for it,
and checking whether a reported site matches the address you are looking at — once per candidate.

```bash
# name every stub in the module
python3 tools/re/stub_nid_map.py <DUMP_ROOT>/PPSA28061-app0/eboot.bin --names ../PS5-3.20_Libs

# or just the ones a call chain targets
python3 tools/re/stub_nid_map.py <DUMP_ROOT>/PPSA28061-app0/eboot.bin --names ../PS5-3.20_Libs \
    --addr 0x13d750 --addr 0x13d810 --addr 0x13d820
0x13d750  Gz1rmUZpROM  sceNpTrophy2CreateHandle        libSceNpTrophy2
0x13d810  4IzqhhUQ3nk  sceNpTrophy2GetGameInfo         libSceNpTrophy2
0x13d820  +PDSI6WgPRc  sceNpTrophy2GetGroupInfoArray   libSceNpTrophy2
```

It walks `nid_gate_scan`'s own resolution (NID → dynsym index → JMPREL jump slot → stub → entry
points) for *every* import and inverts it, sharing that module's `Image`/`flatten`/`scan_code` so the
two tools cannot silently disagree about what a module imports.

Two behaviours are deliberate. The map is keyed by **address**, because a CET-prologue stub is
entered four bytes before its `jmp` and a NID-keyed map would answer "not a stub" for the address the
caller actually names. And a `--addr` you ask about that has no row is reported as **`NOT A STUB`**
rather than omitted — silently dropping it makes "this call does not go through an import" and "you
named the wrong address" the same empty output, and the reader cannot tell which they got.
`test_stub_nid_map.py` (ctest: `re_stub_nid_map`) pins both, each arm naming the mutation it kills.

Worked example (#2186): *Earthion*'s only `sceNpTrophy2GetGroupInfoArray` call site sits in a
six-step trophy-init chain of anonymous addresses. Mapping the stubs named step 4 as
`sceNpTrophy2GetGameInfo`, which already answers an error and short-circuits the chain — so the call
site under investigation is unreachable, and a boot A/B that would otherwise have reported a
misleading null could be turned into a positive control that fires.

## `hle_handler_map.py` — which Sony NIDs share ONE prosper handler?

`--all-nids` above says which Sony answers a title *depends* on. This is the other axis of the same
cross, and without it the first one cannot be acted on.

A runtime return-value histogram (`tools/hle_calls --values`) keys on the **prosper handler symbol**,
not on the Sony function. When one handler is registered for several Sony entry points, all of them
collapse into a single row — typically reading `0x0` — and **a mismodelled answer for one is
indistinguishable from a correct answer for all of them.** No runtime instrument can recover that:
by the time a call is counted, the collapse has already happened. It has to be read out of the
registration tables.

```bash
# 1. which imports does the title branch on?
python3 tools/re/nid_gate_scan.py <DUMP_ROOT>/PPSA05325-app0/eboot.bin \
    --all-nids --names ../PS5-3.20_Libs > gated.txt

# 2. which of those are answered by a handler that also answers something else?
python3 tools/re/hle_handler_map.py --platform linux --names ../PS5-3.20_Libs --gated gated.txt
```

Measured on *Sonic Origins* (`PPSA05325`) at the time of writing: of 247 gated rows, **36** sit on a
handler that answers more than one Sony entry point — so 36 `--values` rows cannot be trusted on
their own, while the other 135 registered rows have a handler to themselves. The largest collapses
in the tree are `k_attr_noop` (**20** Sony names), `s_ok` (**14**) and `font_ok` (**12**).

### Reading a zero from it

Every run prints a coverage block *before* any table, because "no collapses" and "I parsed nothing"
must never be the same output:

```text
#   registration APIs (from dispatch.hpp class Hle): register_fn, register_placeholder
#   wrappers discovered:  4 macro (R RN RN_SUBMIT RN_SUBMIT_NAMED)
#                         2 lambda (R reg)
#   registration sites claimed:                      1156  [direct=237 macro(R)=692 …]
#   distinct NIDs registered:                        1153
#   sites claimed but NID not a literal:             17  (listed below; not a coverage gap)
#   sites UNCLAIMED (a missed shape):                0
```

| exit | meaning |
| --- | --- |
| `0` | the scan ran; the printed counts **are** the answer, zero included |
| `2` | refused — nothing was parsed, and no number printed is a result |
| `3` | the scan ran but is **incomplete**: a registration site was unclaimed, or the reconciliation below disagreed. Every table is a lower bound |

`sites UNCLAIMED` is the load-bearing line. It is produced by a deliberately over-broad detector that
is independent of the shape list — it counts *mentions* of a registration API or of a discovered
wrapper, while the shape list *consumes* them — so a shape the parser does not know surfaces as a
non-zero residual instead of vanishing.

### The shapes are discovered, not hardcoded

prosper almost never calls `Hle::register_fn` directly: there are file-local `#define R(...)` /
`RN(...)` / `RN_SUBMIT(...)` / `RN_SUBMIT_NAMED(...)` macros and file-local lambdas. A fixed pattern
list is exactly what made the first version of this measurement a scratch script, so:

* the **API list** is read out of the `class Hle` declaration in `dispatch.hpp` — a third
  registration API is picked up the day it is declared;
* the **wrapper list** is every macro, lambda **or free function** whose body forwards one of its own
  parameters into a registration call, found by a fixpoint so a macro wrapping another macro is
  caught too;
* the discriminator that keeps `register_service_hle()` — a zero-parameter function whose body is 300
  registration calls — from being mistaken for a forwarder is that a wrapper must forward a
  **parameter** into the NID slot. Without it the census comes back empty.

### It is reconciled against the binary, not trusted

A parser is the wrong kind of authority here, so `hle_registry_dump` (built from
`tools/re/hle_registry_dump.cpp`) runs the real `register_builtin_hle()` and prints the registry
prosper actually builds — no boot, no game dump, no GPU, because registration is pure setup:

```bash
./build-linux/hle_registry_dump registry.tsv
python3 tools/re/hle_handler_map.py --platform linux --registry registry.tsv \
    --names ../PS5-3.20_Libs
```

A NID the binary registers that the parser never produced is, by definition, a registration shape the
parser cannot read. `ctest -R re_hle_handler_map` runs both this and the parser's own unit tests.

The handler-address column is an **upper** bound on collapse only: an identical-code-folding linker
can give two distinct `{ return 0; }` handlers one address, so equal addresses do not prove a shared
handler — unequal addresses do prove a distinct one.

### Two things that moved the number, both of which look like nothing

* **`--platform` is REQUIRED, and that is the point.** `hle_kernel_mem.cpp` defines `register_kernel_mem_hle()` **twice**,
  ~3,200 lines apart, in the two arms of one `#if defined(__linux__) || defined(__APPLE__)`. A
  line-based extraction counts both arms, so every kernel-memory handler looks registered for two
  Sony names and gets promoted to "shared". Five handlers that answer exactly one Sony function each
  — `k_dmem_size`, `k_virtual_query`, `k_alloc_dmem`, `k_mtypeprotect`, `k_mprotect` — were counted
  as collapses this way. The tool evaluates the conditionals for one platform and prints how many
  lines it skipped. There is no default: a tool whose default silently answers for one platform is
  how this class of error starts, so the reader has to name the arm.
* **Distinct Sony NAMES, not registration SITES.** `scePthreadAttrSetaffinity` is registered to
  `k_attr_noop` at both `hle_kernel.cpp:4539` and `:4588`. That is one Sony entry point; counting the
  lines inflates every handler with a duplicate.

Together those two are the whole difference between the **41** rows this measurement was first
published with (#2070) and the **36** the tool reports: platform-blind, per-site counting reproduces
41 exactly, and the 5 extra rows are the `hle_kernel_mem.cpp` handlers listed above. `s_ok` is
likewise **14**, not 15 — the 15th, `sceNpCheckCallback`, is registered to `s_ok` only in the
`#else` (Windows) arm at `hle_service.cpp:4596`; on Linux that NID goes to a real handler, and the
two arms *swap* a name rather than adding one, so the total is 14 on every platform.

### Library attribution is a definition, and it is printed as one

The by-library table uses the PS5 3.20 firmware file names **verbatim** and never rolls up a
"libc"/"non-libc" subtotal, because that rollup is the one place this measurement has been misread:
two extractions that appeared to disagree (13 vs 37) were the same rows, differing only over whether
`libkernel` counted as "libc". Attribution comes from the same `--names` loader `nid_gate_scan.py`
uses — the first `libSceXxx.c` in sorted order that exports the NID — so the two tools cannot
disagree about a library. Note this is a *Sony* attribution, not a prosper-file one:
`sceSysmoduleIsLoaded` is `libSceSysmodule` even though prosper registers it in
`src/hle/hle_kernel_time.cpp`.

## `pak_index.py` — turn a UE4 `.pak` byte offset into an asset name

A UE4 title on PS5 streams content through the Ampr/APR async-read path, and `PROSPER_FILELOG=1`
logs every request:

```text
[apr] read-submit id=1 …/game-ps5.pak -> dst=0x…(guest) off=0x89a1be4 size=3080 got=3080 OK
```

That records *how much* was read and *whether it succeeded* — but not **what** was read. In a
stripped shipping eboot the read stream is often the only readable statement of the engine's
loading state, and those offsets are raw byte positions in a multi-gigabyte container.

`pak_index.py` parses the pak's own index (footer → encoded entries → full directory index) and
maps offsets back to asset paths, so the APR stream becomes a load trace:

```bash
# what is at this offset?
python3 tools/re/pak_index.py GAME.pak --resolve 0x89a1be4

# decode a whole run into the assets it loaded, in order
python3 tools/re/pak_index.py GAME.pak --log run.log --distinct
python3 tools/re/pak_index.py GAME.pak --log run.log --summary

# does the title even ship this map?
python3 tools/re/pak_index.py GAME.pak --list '*/Maps/*.umap'
```

It is offline and read-only — no boot, no GPU, and it works against a log captured earlier.

**Why it is worth reaching for.** On PPSA19244 (The Oregon Trail) the open question was why UE4's
base pass drew nothing. One `--distinct` pass over a retained `PROSPER_FILELOG=1` log answered it
without another run: of the 481 `.umap` files in the container the title loads exactly one,
`TheOregonTrail/Content/Maps/L_GameloftSplash.umap` (a 12 KB splash level), and never opens
`L_Main` or `L_StartMap`. An empty base pass is the *correct* output for that world — so the draw
count was never the defect. Use this before concluding anything about missing geometry: confirm
which content is actually resident first.

Limits: unencrypted index only, pak index versions 10-11 (the v10+ encoded-entry + full-directory
layout). IoStore `.utoc`/`.ucas` containers are a different format and are not handled. Run
`--self-test` to check the entry decoder, the offset lookup, and the log parser without a pak.

## `gstr.py` — name a stripped function by the strings it references

`xref.py from <addr>` mostly prints `lea -> 0x…` / `load -> 0x…` into read-only data, and on a
stripped native title those data addresses are the *only* way to identify a function. Feed them to
`gstr.py` and the function names itself:

```bash
python3 tools/re/xref.py /tmp/eboot.elf from 0x2121d9e \
  | awk '/^   (lea|load) /{print $3}' \
  | python3 tools/re/gstr.py /tmp/eboot.elf - --strings-only
0x700bd18      W"nothreadtimeout"
0x705fd1e      W"GPU has hung or crashed!"
0x718c6ca      W"GameThread timed out waiting for RenderThread after %.02f secs"
```

— so the guest frame at `eboot+0x2121d9e` is UE4's `GameThreadWaitForTask`, i.e. that thread is
blocked on the render-command fence, not on the async loader. Three frames identified this way
(`Create{Vertex,Index,Structured}Buffer_RenderThread`, `FlushRHIThreadFlushResourcesTotal`,
`r.RHICmdAsyncRHIThreadDispatch`) turned a 90-thread `guest_bt` dump on `PPSA07809` into a named
GameThread → RenderThread → RHIThread wait chain (#1982).

`strings`/`objdump`/`readelf` cannot do this: `prx_to_elf.py` writes no section header table, so they
reject the image, and none of them maps a VADDR to a file offset. `gstr.py` resolves the address
through the image's `PT_LOAD` headers and auto-detects UTF-16LE (UE4's `TCHAR`) as well as ASCII.
Addresses are image-relative, like every other tool here. An address that holds no string prints
`bytes <hex>` and one outside every `PT_LOAD` prints `<unmapped>`, so a wrong address is visibly
wrong rather than silently empty.
