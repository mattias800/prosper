# guest_bt — symbolicated GUEST-thread backtraces for prosper

Answer *"what is this guest thread actually doing / waiting for?"* with a real call stack — including
**managed C# method names** for IL2CPP/Unity titles — for any live or frozen prosper process.

**It is not IL2CPP-only.** The managed-symbol step is optional; everything else — the re-sectioned
`.eh_frame` unwind, the HLE stub bridge, and per-thread reporting — works on any title. On a
**stripped native C++ UE4** title (PPSA19244, no `script.json`, no symbols at all) it walked all
**68** guest threads and reported them under their **real engine thread names**, because the guest's
own `pthread_setname` values survive: `RenderThread 1`, `RTHeartBeat 1`, `FAsyncPurge`,
`FAPREventQueueL`, `TaskGraphThread` ×21, `PoolThread 0..8`, `HttpManagerThre`, `OnlineAsyncTask`.
Guest frames show as bare `eboot+offset`, but the thread name plus the prosper-side wait function
(`k_cond_wait` / `k_cond_timedwait` / `k_usleep` / `k_eq_wait`) is usually enough to answer "which
subsystem is stuck": on that title it isolated the **two** threads of 68 still executing guest code
in a boot that had otherwise gone completely quiet. Reach for it on native titles, not just Unity.

## Why gdb alone can't do this

Two things defeat gdb's native unwinder on a prosper guest thread:

1. **No frame pointers.** Unity's engine and IL2CPP-compiled C# are built `-fomit-frame-pointer`, so
   gdb's fallback `rbp`-chain walk produces garbage. Correct unwinding needs each module's DWARF CFI
   (`.eh_frame`) — but `tools/il2cpp/prx_to_elf.py` flattens the guest modules **without a section
   header table** (it targets Il2CppDumper, which only reads program headers), so gdb finds no
   `.eh_frame` section and never engages its DWARF unwinder.
2. **The HLE stub boundary.** When guest code calls a Sony library function, it calls a prosper-
   *synthesized* import stub in the `BOOT_STUB` region (`0x600000000`) which swaps `%fs` and tail-calls
   the C++ HLE. That stub has no CFI at all, so even a CFI-aware unwinder dies at it with a bogus
   `saved-rip = 0` — exactly the frame between the C++ HLE and the guest caller.

`guest_bt` fixes both: it re-sections the flattened module ELFs so gdb gets real `.eh_frame` (and a
`.symtab` of managed method names), and it teaches gdb to step *through* the stub with a custom
unwinder derived from the swap-stub's fixed stack layout.

## Pieces

| file | role |
|---|---|
| `mk_sym_elf.py` | add a section header table (`.text`, `.eh_frame` located via `PT_GNU_EH_FRAME`, and an optional `.symtab` from an Il2CppDumper `script.json`) to a flattened module ELF |
| `guest_bt_gdb.py` | gdb plugin: `add-symbol-file` each module at its runtime base, register the stub unwinder, expose `guest-bt [thread]` / `guest-bt-all` |
| `guest_bt.py` | driver: parse a `PROSPER_INITLOG` module map, build/cache the sectioned ELFs, run gdb with the plugin |

## Recipe

1. Run the title once with the module map (and, for managed names, capture nothing extra — the map is
   all `guest_bt.py` needs):

       PROSPER_INITLOG=1 PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
         ./screenshot <dump>-app0 ... 2> run.log

2. (IL2CPP titles, once) build a managed symbol map — see `tools/il2cpp/README.md` — and keep the
   resulting `script.json`.

3. Backtrace a live/frozen instance:

       python3 tools/guest_bt/guest_bt.py --pid $(pgrep -f <TITLEID> | head -1) \
           --initlog run.log --il2cpp-script /path/script.json --all

   `--thread GameUpdate` (id or name; kernel names are truncated to 15 chars, substring-matched) limits
   it to one thread; omit `--all`/`--thread` to backtrace the currently-selected thread.

The sectioned ELFs are cached under `tools/guest_bt/.cache/` (override with `--cache`), so repeat runs
against the same title are instant.

## Example (Terminator 2D, PPSA25872, a thread blocked in an HLE wait)

```
#1  prosper::k_wait_on_address(...)
#2  0x60000ba9d in ??? ()                              <- HLE stub (bridged by the custom unwinder)
#7  System.Threading.WaitHandle$$WaitOneNative
#8  System.Threading.WaitHandle$$InternalWaitOne
#9  Unity.PSN.PS5.Async.WorkerThread$$RunProc          <- named managed frame
#10 System.Threading.ExecutionContext$$RunInternal
#16 prosper::thread_trampoline(void*)
```

## The stub-unwind rule (how the bridge works)

The `%fs`-swap stub (`emit_swap_stub` in `src/host/exec_image_linux.cpp`) pushes five qwords
(`r11` + the four re-pushed stack args) and then `call`s the HLE. So at the stub's return site the
guest caller's frame is a fixed distance away:

    caller_sp = stub_frame_sp + 0x30      # 5 pushes (0x28) + the call's return slot
    caller_pc = *(stub_frame_sp + 0x28)   # the guest return address the stub will `ret` to

and every callee-saved register (`rbp rbx r12-r15`) is passed through unchanged — the stub only
clobbers `rax/r10/r11`. If prosper's stub codegen ever changes this push sequence, update the offsets
in `guest_bt_gdb.py._StubUnwinder` (and this note) together.

## Limits

- Managed (IL2CPP, `0x440000000..`) frames get method names; the Unity **player** eboot
  (`0x410000000`) and system PRXs are stripped C++, so those frames show as bare `module+offset`.
  The same is true of an entirely native title — no managed frames means no method names, but the
  walk, the thread names, and the prosper-side wait frames are all still there (see above).
- The unwind is only as good as the modules' `.eh_frame`; a frame whose module ships no CFI ends the
  walk. `mk_sym_elf.py` reports which modules had CFI.
- Linux/`%fs`-swap stubs only. The Windows/macOS stub shapes differ; the rule would need porting.
