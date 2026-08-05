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
| `const` | compares the result against `--const` |
| `nonzero` | `test eax,eax` / `cmp eax,0` → conditional branch |
| `alu-gate` | folds the result into flags (`and eax,mask` …) and branches on that |
| `other-cmp` | compares against some other immediate |
| `forward` | still live when the window ends — returned, spilled, or tail-jumped; read it by hand |
| `ignored` | dead before any read: this call site cannot be affected |
| `undecodable` | objdump could not decode the window — a **void** sample, not a negative one |

**Read the output as a bound, not a census.** It classifies the *compare*, never the branch target,
so it tells you which titles *can* change behaviour, not which do; and site discovery follows one
stub level and one call deep, so a count is a lower bound. `forward=0` across a corpus is worth
quoting for the opposite reason — together with `undecodable=0` it shows every window decoded.

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
