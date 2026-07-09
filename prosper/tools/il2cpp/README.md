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

## gdb caveat (prosper)
prosper installs a SIGSEGV handler for lazy memory commit. Under gdb, ALWAYS use

    handle SIGSEGV SIGILL SIGBUS nostop noprint pass

so recoverable guest faults reach prosper's handler. Otherwise gdb stops on
normal lazy-commit faults and corrupts the run — a healthy default boot then
appears to "crash" at a null deref when it actually does not.

Do NOT commit dumper output (dump.cs / script.json / il2cpp.h) — it is derived
from the gitignored game dump.
