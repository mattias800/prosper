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
and indirect calls/jumps through the requested slot. `from ADDRESS` lists references made by the function
window containing that address; `reloc ADDRESS` restricts output to relative data relocations.

For The Messenger, the example above identifies both sides of Unity's runtime IL2CPP API table:

```text
store at eboot+0x148276c   # writes dlsym("il2cpp_init")
call* at eboot+0x1473a6f  # invokes the populated slot
```

This slot is zero in the flat image and has no ELF relocation. Treating every indirect RIP-relative call
as a static GOT import would send the investigation to the wrong layer.
