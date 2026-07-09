# IL2CPP symbolication for Unity/IL2CPP PS5 titles

Recover C# method names + addresses from an IL2CPP title so you can breakpoint
managed logic (scene load, boot state machines) during emulator debugging.

## Recipe
1. Flatten the compiled-C# PRX (unencrypted in the dump) into a loadable ELF:

       python3 prx_to_elf.py <dump>/Media/Modules/Il2CppUserAssemblies.prx /tmp/asm.elf

2. Run Il2CppDumper (needs dotnet-sdk). The ELF starts at vaddr 0 so it prompts
   "input il2cpp dump address" — answer `0` to force the registration auto-scan:

       printf '0\n0\n0\n' | dotnet Il2CppDumper.dll /tmp/asm.elf \
           <dump>/Media/Metadata/global-metadata.dat /tmp/out

3. `out/dump.cs` lists every method with a `// RVA: 0x...` comment. The runtime
   address is `module load base + RVA`. For PPSA02664 the PRX loads at
   `0x440000000` (verified via gdb `info proc mappings`), so `DoFirstLogin`
   (RVA `0x2140D0`) is at `0x4402140D0`.

## gdb caveat (prosper)
prosper installs a SIGSEGV handler for lazy memory commit. Under gdb, ALWAYS use

    handle SIGSEGV SIGILL SIGBUS nostop noprint pass

so recoverable guest faults reach prosper's handler. Otherwise gdb stops on
normal lazy-commit faults and corrupts the run — a healthy default boot then
appears to "crash" at a null deref when it actually does not.

Do NOT commit dumper output (dump.cs / script.json / il2cpp.h) — it is derived
from the gitignored game dump.
