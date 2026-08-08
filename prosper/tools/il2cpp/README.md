# IL2CPP symbolication for Unity/IL2CPP PS5 titles

Recover C# method names + addresses from an IL2CPP title so you can breakpoint
managed logic (scene load, boot state machines) during emulator debugging.

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
   `module load base + RVA`. Both PPSA24651 (The Messenger) and PPSA02664 load the
   IL2CPP PRX at `0x440000000`, so RVA == script.json Address == a prosper
   `[btrace] il+0x<offset>` frame. `DoFirstLogin` (RVA `0x2140D0`) → `0x4402140D0`.

5. Symbolicate addresses / btrace chains with `resolve.py`:

       python3 resolve.py /tmp/out/script.json il+0x1764ce2 0x11e63c
       # il+0x1764ce2  ->  Unity.PSN.PS5.Async.WorkerThread$$RunProc (+0xa2)
       echo '[btrace] ... chain=il+0xde92e9,il+0xde9159' | python3 resolve.py /tmp/out/script.json -

   NOTE: prosper's `[btrace]` validated unwinder must accept INDIRECT call sites
   (`0xFF` at v-2/-3/-6/-7), not just `0xE8` (call rel32) — IL2CPP dispatches
   managed methods indirectly, so an 0xE8-only filter drops every managed frame.

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

## gdb caveat (prosper)
prosper installs a SIGSEGV handler for lazy memory commit. Under gdb, ALWAYS use

    handle SIGSEGV SIGILL SIGBUS nostop noprint pass

so recoverable guest faults reach prosper's handler. Otherwise gdb stops on
normal lazy-commit faults and corrupts the run — a healthy default boot then
appears to "crash" at a null deref when it actually does not.

Do NOT commit dumper output (dump.cs / script.json / il2cpp.h) — it is derived
from the gitignored game dump.
