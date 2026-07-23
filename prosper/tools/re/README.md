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
