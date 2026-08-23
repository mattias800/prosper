# Metaphor: ReFantazio (`PPSA20800`) — status

Atlus GFD engine + CRIWARE middleware, AGC SDK 12, 71 GB dump. Tracker
[#2876](https://github.com/mattias800/prosper/issues/2876).

**Rung 0.** The title boots in ~90 ms, loads its GFD assets, opens two audio ports, initialises
save data, compute and fonts, and publishes 63 frames at 34.8 fps — and every one of those frames
is black. Its primary thread then dies of a divide-by-zero about five seconds in, so every later
sample is stale by construction.

That is a different place from where it was on 2026-08-22, when it produced **zero** frames and
died in its CRI Mana movie thread ([#2934](https://github.com/mattias800/prosper/issues/2934)).
That crash is fixed; the section below is the record of what it was, because the shape of it is
worth keeping.

## What #2934 actually was — a 4 GiB in-place byte-swap of the guest's own heap

The symptom was a SIGSEGV in the thread named `CRI Server Mana`, at `eboot+0x1221b60`
(`mov eax,[rdi] ; mfence ; ret`, an atomic word read) with `rdi = 0x20000000e0c5df73` — an address
that is not a guest pointer at all. The caller is the CRI server's list walk at `eboot+0x11cd7c1`,
which iterates two intrusive lists whose heads live at `eboot+0x2d71838` (the "stop" list) and
`eboot+0x2d71850` (the "playing" list).

The register that named the defect was `r13 = 0x200fddc5e0` — the list node itself, a perfectly
good pointer. `*r13` held `0x20000000e0c5dd0f`, which is `0x200fddc5e0` with **each 32-bit half
byte-reversed**. Every pointer in that object was in the same state, including `[node+0x18] =
0x040000003818d712`, the byte-reversed form of the list head `0x412d71838`. The whole doubly-linked
list was internally consistent *in the reversed domain*, so nothing had overwritten it with foreign
data: something had byte-reversed live memory in place.

Three measurements settled it:

1. **`PROSPER_POLLWATCH`** (added by this work) sampled two slots 48 KiB apart every millisecond.
   Both held correct values (`0x1a3`, `0x412524998`) and both flipped to their reversed forms
   (`0xa3010000`, `0x400000098495212`) inside the same 1 ms sample at **t ≈ 94 ms** — one bulk
   event during guest init, not a gradual corruption, and nearly seven seconds before the fault.
2. **`PROSPER_HWWATCH_ABS=0x200fde8000`** named the writer: `eboot+0x10d04b5`, the store half of an
   AVX2 loop at `eboot+0x10d0360` — `vmovdqu` / `vpshufb` against the mask at `0x19a8c40` (which is
   `03 02 01 00 07 06 05 04 …`, the 8-in-32 byte-reverse shuffle) with a `movbe` scalar tail. That
   function is `swap32(dst, src, byteLen)`, the engine's big-endian resource converter.
3. **`PROSPER_HWBP=0x10d0360 PROSPER_HWBP_ARGS=1`** printed its arguments:
   `rdi=0x20004d1b80 rsi=0x20004d1b80 rdx=0xffffffff`. Destination equals source, so it is the
   in-place path, and the byte length is **`(uint32)-1`** — `0x3FFFFFFF` dwords, ~4 GiB, all of it
   inside the guest's own 10 GiB Ampr AMM pool at `0x2000000000`.

`0xffffffff` is an error return. The caller is `eboot+0x10d1e00`, a read-with-endian-swap helper
whose own error paths leave `r15d = -1` and then fall into the swap call at `eboot+0x10d1e96`
anyway, passing `-1` as a length. Probing the reader struct at the moment of the call
(`PROSPER_HWBP_PROBE` on `r14`) showed `[+0x10] = 0x10003` (mode 3 = memory stream, flags = 1),
`[+0x18] = 0` (cursor), `[+0x1c] = 0` (**size**), `[+0x20] = 0` (**base pointer**). The reader had
no buffer at all, so it reported EOF, returned -1, and the caller swapped -1 bytes.

Upstream of that, the 16-byte header buffer read back as all zeros, failed the `'GFS0'` magic test
at `eboot+0x10d1b6e`, and was itself byte-swapped as a presumed big-endian header — the loader's
normal handling of a big-endian GFS file, applied to nothing.

### Why the resource was empty: prosper's direct-memory pool started outside the budget it advertised

Immediately before the empty reader, the memory log shows:

```text
[memhle] alloc_dmem len=0x400000 align=0x100000 type=0xb in [0x0,0x400000000) -> ENOMEM
```

`sceKernelGetDirectMemorySize()` answered `0x400000000` (16 GiB), and the pool held exactly that
much — but at physical offsets `[0x10000000, 0x410000000)`, because `kDmemBase` was `0x10000000`.
A guest search window is expressed in the offset space the advertised size defines, and this guest
partitions that space **to the byte**:

| request | length | where it landed |
| --- | --- | --- |
| Ampr AMM pool (`sceAmprAmmGiveDirectMemory`) | `0x27F000000` | `[0x10000000, 0x28F000000)` |
| Ampr command buffer | `0x400000` | `[0x28F000000, 0x28F400000)` |
| `sceKernelAllocateMainDirectMemory` | `0x180000000` | `[0x28F400000, 0x40F400000)` |
| the one that failed | `0x400000` | — |

`0x400000000 - 0x180000000 - 0x400000 - 0xC00000 = 0x27F000000`: the guest sized its first pool so
that exactly 12 MiB of the advertised budget would be left over. Twelve megabytes *were* free — at
`[0x40F400000, 0x410000000)`, entirely above the `0x400000000` window the guest asked in, because
prosper's base offset had pushed the tail of the pool past the end of the size prosper itself had
reported. There is no environment lever that helps: shifting `PROSPER_DMEM_BUDGET_MB` moves both
the advertised size and the pool, so the same `kDmemBase` bytes fall off the end whatever the
budget is.

**Fix:** `kDmemBase` is now `0x10000` — 64 KiB. Small enough that the unreachable tail is
negligible, non-zero so a successful allocation never returns physical offset 0, and a multiple of
the Windows allocation granularity, which review established is load-bearing: the Windows arm maps
its section at `phys - kDmemBase`, and `map_section_view`'s `MapViewOfFileEx` fallback depends on
that offset staying congruent to `phys` mod 64 KiB. The pool now lives inside the offset space the
guest is told it has. With it, the 4 MiB request succeeds at `phys=0x3ff600000`, the resource loads, and the
`swap32(-1)` never happens.

This is a cross-title defect that happened to be fatal here: any guest that partitions the whole
advertised direct-memory budget loses its last `kDmemBase` bytes.

## Where it stops now

Three blockers, in the order they bite.

1. **SIGFPE in the guest's primary thread at `eboot+0x10019f6`**, about 5 s in — `div r14d` with
   `r14d == 0`. The enclosing function is `eboot+0x1001970`, and RTTI names it: it is slot 3 of
   **`fw::font::Font_PS5`**'s vtable (typeinfo `N2fw4font8Font_PS5E` at `0x237078f`), the routine
   that expands an 8-bit glyph coverage bitmap into a white RGBA8 texture. `r14d` is
   **bytes-per-pixel**, resolved through two lookup tables from the FORMAT field of the target
   image's Gen5 T# — `(dword[image+0x64] >> 20) & 0x1ff` — and every stage of that resolution
   defaults to **0** on an unrecognised value. The healthy answer is 4. The `div` runs *before*
   both zero-dimension guards, so a zero glyph width or height is not what kills it.
   **The lead:** the four `libSceFont` entry points this path calls —
   `sceFontSelectRendererFt` (`Xx974EW-QFY`), `sceFontRenderSurfaceSetScissor` (`vRxf4d0ulPs`),
   `sceFontGetCharGlyphMetrics` (`L97d+3OgMlE`) and `sceFontRenderCharGlyphImage` (`3G4zhgKuxE8`)
   — are all **unregistered** and all answer 0, and they are the last four
   `[prosper] unimplemented:` lines before the fault. `sceFontRenderSurfaceInit` and
   `sceFontCreateRendererWithEdition` on the same path *are* registered
   (`src/hle/util/hle_font.cpp:332`, `:311`), so this is a partial library rather than an absent
   one. Full derivation, the T# layout and the one-run experiment that would settle it are on
   [#2951](https://github.com/mattias800/prosper/issues/2951).
   Once it dies, `tools/screenshot` reports `guest=faulted` and every sample after that moment is
   stale (`max-pixel-stale=115.0s`), so the run's 24 identical PNGs are one frame served 24 times.
2. **The 63 frames it does publish are black** — 6,093 of 6,137 sampled pixels pure black, the rest
   near-black (`0x161314`, `0x150e05`). 5 distinct of 63 published. Whether this is the composite
   defect of [#2932](https://github.com/mattias800/prosper/issues/2932) or a title-specific absence
   is **not established**: the guest dies before enough frames exist to characterise it. Tracked as
   [#2952](https://github.com/mattias800/prosper/issues/2952).
3. **A host-side null dereference in prosper's own renderer, on the guest job thread
   `job_render_0`** — inside a `std::unordered_map` lookup in `render_draw_pass_rgba`
   (`tests/fixtures/render_runner.h:6169`, whose `persistent_texture_images` is an unsynchronised
   function-local static). **Deterministic under `boot_trace`: 5 of 5 runs**, at two distinct
   instructions that both resolve to libstdc++'s `_M_equals`. **0 of 2 under `tools/screenshot`**,
   which is why the rung evidence above exists at all. Name the frontend in any claim about this
   one (instrument trap 127). [#2953](https://github.com/mattias800/prosper/issues/2953).

## Reproduction route

Frontend `tools/screenshot`, default route, no pad input, Linux / AMD Radeon 8060S (RADV
STRIX_HALO):

```bash
PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct SDL_VIDEODRIVER=offscreen \
timeout -k 10 300 ./prosper/build-linux/screenshot <DUMP_ROOT>/PPSA20800-app0 \
    --seconds 5 --count 24 --timeout 170 --out <OUT_DIR> \
    --allow-guest-fault --no-stop-after-guest-fault
```

The outer bound and `--timeout` are deliberately different values: `--timeout` cannot bound a boot
(instrument trap 214), so a run that prints neither `[shot] done:` nor `[shot] timeout` was killed
from outside and its silence is not evidence.

## Ruled out

One line per hypothesis this work killed, so nobody re-derives it at full cost.

- **prosper's live Vulkan renderer is not what byte-reversed the heap.** A matched A/B with
  `PROSPER_RENDER` unset reproduces the corruption byte-for-byte — same addresses, same reversed
  values, same `CRI Server Mana` fault. A GPU write-back with a channel-order swap would have been
  an exact fit for an 8-in-32 reversal, and it is not the mechanism. #2934.
- **prosper did not write the reversed bytes at all.** `grep -rn 'bswap\|byteswap\|__builtin_bswap\|htonl\|htobe32\|be32toh\|std::byteswap'`
  returns **zero** matches over the whole `prosper` tree — and the wider scope is the point, because
  `prosper/src` alone would not have established it: the live renderer is in
  `tests/fixtures/render_runner.h`, outside `src/`. (Widened during review; the only hits anywhere
  are vendored SDL3 headers under a build directory.) There is no byte-swapping code in the
  emulator, so the writer was always going to be guest code — which is what `PROSPER_HWWATCH_ABS`
  then confirmed. #2934.
- **The four unregistered NIDs called before the fault are not the cause.** `sceHttp2CreateTemplate`
  (`+wCt7fCijgk`), `sceAmprCommandBufferGetNumCommands` (`gzndltBEzWc`),
  `sce::Json::InitParameter2`'s constructor (`GvGvswb0v34`) and `sceRtcGetDayOfWeek`
  (`CyIK-i4XdgQ`) are all still unregistered and still answer 0 on the fixed run, which reaches
  asset loading, audio, save data, compute and fonts. They were the leads on the tracker; none of
  them was it. #2934.
- **The refused `ampr push-map` is not the cause either.** Both `sceAmprCommandBufferSetBuffer`
  calls still log `FAILED (every page of the range has a VMA)` on the fixed run — the guest mapped
  those VAs itself moments earlier, so the refusal is the correct no-clobber outcome — and the title
  now gets six seconds and 63 frames further with them unchanged. #2934.
- **`PROSPER_DMEM_BUDGET_MB` cannot substitute for the `kDmemBase` fix.** The guest derives its
  search window's end from `sceKernelGetDirectMemorySize()`, so raising or lowering the advertised
  budget moves the window and the pool together and leaves exactly the same `kDmemBase`-sized tail
  unreachable. Arithmetic, not a run: worked through above. #2934.
- **`PROSPER_NO_BACKEND_PERSISTENT_TEXTURES=1` was never tested against the CRI fault.** A run that
  appeared to falsify the fix under that lever was executed against a **stale `screenshot` binary**
  — only the `boot_trace` target had been rebuilt. Recorded because the arm looked like a clean
  negative and was not; rebuild every frontend, not the one you are about to run.

## Instruments this work added

- **`PROSPER_POLLWATCH="0xADDR[,...]"`** (`src/host/image/exec_image_linux.cpp`) — a host thread
  samples up to six guest 64-bit slots every millisecond and prints every value change with an
  elapsed time. It names no writer, but it bounds an event in time for the cost of one read per
  slot per millisecond, and it perturbs nothing. That is what the exact instruments could not do
  here: the mprotect page-watch costs a SIGSEGV plus a single-step SIGTRAP per write to the whole
  4 KiB page, and on this title's heap a boot that normally faults at 7 s had not reached the fault
  after 240 s.
- **`PROSPER_PAGEWATCH=0xADDR`** (`src/hle/memory/hle_kernel_mem.cpp`) — arms prosper's existing
  mprotect write-watch as soon as a guest mapping covers the address. The watch already existed;
  its only arm site was the AGC fence builder, and this title issues no `ReleaseMem` at all, so
  `PROSPER_WATCH_ABS` was unarmable and its silence was indistinguishable from "no writer".
