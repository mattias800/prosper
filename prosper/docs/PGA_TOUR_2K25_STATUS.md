# PGA TOUR 2K25 (`PPSA17952`) — bring-up status

**Rung 0** as of 2026-08-22. The guest boots, links, loads assets and submits real GPU work, but no
frame the live renderer publishes is anything other than black, and the process dies on a worker
thread before a title screen. First bring-up session; nothing about this title existed in the repo
before it.

| | |
| --- | --- |
| Engine | Unity **6000.0.20f1** (Unity 6) / IL2CPP |
| SDK | **9.00** (`sdkVersion 0x0900000000000000`) — below the SDK ≥ 13 post-submit-visibility gate |
| Content | `UP1001-PPSA17952_00-PGATOUR2K25GBL00`, `contentVersion 01.026.000` |
| Video out | 3840x2160, `hdr-display-enabled=1`, `gfx-threading-mode=4`, native gfx jobs |
| Modules | `Media/Modules/{Il2cppUserAssemblies,PS5Util}.prx`, five `Media/Plugins/*.prx` (Photon Voice, Burst, Opus, spatializer), `sce_module/{libc,libSceFace,libSceFaceTracker,libSceJobManager,libSceNpCppWebApi,libScePfs}.prx` |

## Reproduction route

No pad route exists yet — the title dies before any menu, so there is nothing to drive.

```bash
PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_BOOTPHASE=1 \
PROSPER_SAVE0=<WORK>/save0 SDL_VIDEODRIVER=offscreen \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA17952-app0 \
    --seconds 7 --count 20 --out <WORK>/shots --timeout 300
```

Frontend: `tools/screenshot`, default route, no pad input. Outer `timeout 480` differs from the
inner `--timeout 300` deliberately (instrument trap 214), and `--seconds 7` is prime (trap 211).

## What works

Boot is fast and clean — `BOOT_COMPLETE` at **+437 ms**, ten modules linked and mapped. The engine
then reaches deep into its own startup: it streams `Media/data.unity3d` through the **Ampr** path
(`[apr] read-submit … OK`), opens `StreamingAssets/ContentArchives/archive_dependencies.bin` and the
DOTS `StreamingAssets/EntityScenes/scene_info.bin`, brings up the live Vulkan compute backend, and
submits real draws — `CB_COLOR_CONTROL.MODE=0` alone is counted past 1024 draws in a 7 s window.

So neither the loader, the IL2CPP path, the asset layer nor the recompiler is the frontier here.

## The first blocker (fixed in this session)

The title died at **1.2 s on every boot** with `SIGSEGV addr=0x4 rip=Il2cpp+0x6a112`, immediately
after printing its own diagnosis:

```
ERROR...
 PSN is an old version that cannot be used by the current player runtime.
 Please update the PSN native module and any associated managed assemblies to the latest versions
```

That message is not about a missing module. Disassembling `Il2cppUserAssemblies.prx` names the
mechanism exactly:

```
6a09a:  mov  rax,[rip+…]            # global A
6a0ae:  cmp  rax,-1
6a0b2:  je   0x6a12f                ; -1 => not initialized, skip
6a0b4:  mov  rbx,[rip+…]            # global B = the plugin-args pointer
6a0bb:  test eax,0xfffffff0
6a0c0:  je   0x6a0dd                ; argc has no high bits => ERROR path
6a0c2:  cmp  DWORD PTR [rbx],0x10   ; argp->size    == 0x10
6a0c7:  cmp  DWORD PTR [rbx+4],0x200; argp->version == 0x200
6a0d0:  mov  rax,[rbx+8]            ; argp->callback
…
6a112:  mov  esi,DWORD PTR [rbx+0x4]   ; <-- faults: rbx is NULL
```

`xref.py` finds exactly **one** writer of global B, a two-instruction setter that stores `rdi` and
`rsi` — i.e. the module's `module_start(size_t argc, const void *argp)` arguments. prosper started
that module with `(0, NULL)`, so both globals were zero: `A != -1` so no skip, `A & ~0xF == 0` so the
error branch, and then the error branch's own third `printf` reads `argp->version` off a NULL `argp`.

prosper already had the descriptor this handshake wants — `{size=0x10, version=0x200, cb=0}` in
`src/host/image/exec_image_linux.cpp`, commented as *"the descriptor Sony's PSN/SaveData plugin
module_start validates"*. It was simply not registered for this module, because on every title seen
until now the Unity PSN package shipped as a separate `PSN.prx`. **With IL2CPP the package's native
half can be linked straight into `Il2cppUserAssemblies.prx` instead**, and then it is that module's
`module_start` which performs the handshake. Registering `{BOOT_IL2CPP, BOOT_PSNCORE}` in
`set_module_start_param_ranges` fixes it.

The blast radius is one function per module: every module prosper links has `DT_INIT` at `image+0x10`
and an **empty** `DT_INIT_ARRAY` (measured with `PROSPER_INITLOG=1` across all ten of this title's
modules), so the range selects the real `module_start` and never a C++ static constructor.

## The current blocker

With the handshake fixed the title runs ~6x longer and reaches GPU submission, then dies on a
**worker thread named `Background Job.`**:

```
[prosper] WORKER-THREAD FAULT: sig=11 addr=(nil) rip=0x41142d5e0 (image+0x142d5e0)
[prosper]   insn bytes @rip: 41 0f b6 14 0b   (movzx edx,BYTE PTR [r11+rcx*1])   r11=0 rcx=0
```

`eboot+0x142d5e0` is an **HTTP response-header parser** — it scans for `0x09..0x0d`/`0x20`
whitespace, then `:` (0x3a), then CR/LF — and it is walking a NULL buffer. Immediately before the
fault the log shows the whole libSceHttp2 request lifecycle answered by the dispatcher's
unregistered-NID default of `0`:

| NID | Name | prosper today |
| --- | --- | --- |
| `+wCt7fCijgk` | `sceHttp2CreateTemplate` | 0 |
| `mmyOCxQMVYQ` | `sceHttp2CreateRequestWithURL` | 0 |
| `N4UfjvWJsMw` | `sceHttp2CreateCookieBox` | 0 |
| `nrPfOE8TQu0` | `sceHttp2AddRequestHeader` | 0 |
| `rbqZig38AT8` | `sceHttp2SendRequest` | 0 |
| `-rdXUi2XW90` | `sceHttp2GetAllResponseHeaders` | 0 |
| `9XYJwCf3lEA` | `sceHttp2GetStatusCode` | 0 |
| `o0DBQpFE13o` | `sceHttp2GetResponseContentLength` | 0 |

`0` is `SCE_OK`. So `sceHttp2SendRequest` reports that a request the emulator never sent
**succeeded**, and `sceHttp2GetAllResponseHeaders` reports that it returned headers while writing
nothing to its out-parameters. The guest then parses the NULL it was handed. This is precisely the
silent-success-stub failure the charter names, and `src/hle/net/hle_http.cpp` already states the
right principle for this library family in its first comment: *"Network requests remain offline, but
parsing is local and deterministic: returning success without filling this structure makes callers
dereference stale pointer fields."*

The fix is to implement libSceHttp2's lifecycle honestly — local resource creation genuinely
succeeds, setters genuinely record state, and anything that would need a network answer **reports
failure instead of `SCE_OK`**, leaving out-parameters untouched. Tracked as #2894.

**The reason this was deferred no longer holds — see `## Ruled out`.** This paragraph used to say
the v2 error encodings were not derivable from anything on hand. They are, and the answer changes
what an implementer must do: **libSceHttp2's facility is `0x817b____`, not v1's `0x8043____`**, so
prosper's existing libSceHttp constants are *not* reusable across the two even though the low code
bytes are shared. `<DUMP_ROOT>/sprx/` ships `libSceHttp.sprx` and `libSceHttp2.sprx` as plain ELF
(entropy 5.417 / 5.403 bits/byte), and this title's own error classifier at `eboot+0x142ddf0`
confirms the facility independently.

The behavioural finding matters more than the constant. **Any non-zero return is handled
gracefully; only `0` crashes this title.** The classifier's sole zero-returning path is a literal
`test eax,eax`, all 33 targets of its `0x817b1064..0x817b1084` jump table return non-zero, and at
the send site a non-zero classification makes the caller store the error and *return* without ever
reaching the response parser above. So the fault at `eboot+0x142d5e0` is caused specifically by the
false `SCE_OK` — the title was always prepared to be told the request failed. Picking the exact
constant is a correctness question, not a safety one.

For the send path specifically, the real library does not return an HTTP error at all: on a connect
failure it propagates the **raw libSceNet error**, pinned by `0x80410124` being special-cased as
*not* a failure at libSceHttp2's `sceNetConnect` site (`0x24` = 36 = `EINPROGRESS`), which fixes the
encoding as `0x80410100 | BSD errno`. That makes `ENETUNREACH` the honest offline answer, and it is
consistent with what prosper already tells this guest through NetCtl: `sceNetCtlGetState` writes
`SCE_NET_CTL_STATE_DISCONNECTED` and `sceNetCtlGetInfo` returns `SCE_NET_CTL_ERROR_NOT_CONNECTED`
(`hle_service.cpp:4795`, `:4830`). Full derivation, per-NID export map and argument shapes are on
#2894; start there rather than re-deriving.

Other unimplemented NIDs seen on the same boot, none of which is implicated in the fault:
`sceNpAuthCreateAsyncRequest`, `sceNpAuthGetAuthorizationCodeV3`, `sceNpAuthWaitAsync`,
`sceNpRegisterNpReachabilityStateCallback`, `sceNpSessionSignalingInitialize`,
`sceShareRegisterContentEventCallback`, `sce::Json::InitParameter2::InitParameter2`,
`sce::Json::InitParameterRtti2::setAllocatorRtti`, and one libkernel NID (`tU5e3f9gSiU`) that is
**not present in FW 3.20** and so cannot be named from the reference library set.

## Rendering

Not yet assessable, and deliberately not called a defect. Every frame captured is pure black — 1
distinct colour over 3840x2160 — and that is not an artifact of the sampling interval: a second run
at `--seconds 1` (30 requested, `--allow-guest-fault --no-stop-after-guest-fault`) captured 3 frames
before the fault and all 3 are uniformly black. The renderer says why:

```
[rtt] GUEST SCANOUT #1: no present source and no renderer target at the flipped buffer 0x0
[rtt] PRESENT SOURCE EXTENT MISMATCH #1: no pass produced a 3840x2160 present source
[render] frame 0: Vulkan render FAILED (3840x2160)
```

The guest flips a buffer whose address is `0x0`. Whether that is an independent renderer problem or
another consequence of the aborted startup cannot be separated until the title survives past the
HTTP fault, so no hypothesis is recorded here yet.

## Ruled out

- **"The correct Sony error values for the libSceHttp2 facility are not derivable from the dump,
  from `../PS5-3.20_Libs/`, or from prosper's existing `0x8043xxxx` constants, so implementing the
  lifecycle honestly would require fabricating one."** Falsified 2026-09-03 (#2894, PR #3295 for
  the v1 sibling). They are derivable, from two independent primary sources that agree. The dump's
  own `sprx/` directory ships `libSceHttp2.sprx` as a **plain ELF** — entropy 5.403 bits/byte, so
  nothing is decrypted to read it — and every export's shared stack-guard prologue returns
  `0x817b1076`, dating the facility as **`0x817b____`**, *not* v1's `0x8043____`. This title's own
  error classifier at `eboot+0x142ddf0` compares against `0x817b1220` and dispatches
  `0x817b1064..0x817b1084` through a 33-entry jump table, confirming the facility from the guest
  side. The claim was not merely incomplete but load-bearing in the wrong direction: it said the
  work could not be started, when in fact the send path does not use an HTTP error at all — the
  library propagates the raw libSceNet error, whose encoding (`0x80410100 | BSD errno`) is pinned by
  `0x80410124` = `EINPROGRESS` being treated as a non-failure at the connect site. Independently
  re-derived from the bytes by a second reader before this row was written.
- **"Any error prosper invents here risks being read as data, so returning one is hazardous."**
  Falsified 2026-09-03 by enumerating all 33 jump-table targets: they return five distinct non-zero
  categories and **none** reaches the classifier's zero-return `ret`, which is reachable only for an
  input of exactly `0`. So every non-zero value is classified as a failure and handled the same way,
  and the caller returns before touching the response parser. A wrong constant here is wrong, not
  dangerous — which removes the safety argument for deferring the whole lifecycle.
- **"The `only_if_imported` filter (#2870/#2890) drops `sce_module/libSceNpCppWebApi.prx` for this
  title, and that is what makes the PSN handshake fail."** Falsified 2026-08-22 on `af481db2`:
  `PROSPER_INITLOG=1` lists it as **module 8, base `0x4d0000000`** — it is linked, mapped and
  started. Something among the non-candidate modules vouches for it. The PSN failure was the
  `module_start` descriptor above and had nothing to do with module selection.
- **"`PROSPER_NULL_PAGE=1` gets this title past the worker-thread fault."** Falsified 2026-08-22:
  with it set, the same thread still dies, `0x23` bytes further along at `eboot+0x142d603` — the
  null-backed read is re-executed and the *next* access in the same header parser faults. The switch
  is a bounded diagnostic (its own comment says *"Diagnostic, NOT a fix"*), and what it established
  here is that the cascade is short and local to one parser, which is what pointed at the NULL
  header buffer rather than at broken heap state.
- **"The unimplemented libkernel NID `tU5e3f9gSiU` is the cause of the 1.2 s death."** Falsified
  2026-08-22: it is still unimplemented and still returns `0` after the `module_start` fix, and the
  crash is gone. It was adjacency, not causation — worth recording because it was the only
  unimplemented NID in the entire pre-fix boot, which made it look conclusive.

## Not yet investigated

- Whether a title screen sits directly behind the HTTP fault, or behind further walls. The faulting
  thread is a Unity job thread, so it is not obvious that the main thread would reach a menu even if
  the job survived; this has not been tested and should not be assumed either way.
- The SDK-9 submit race (#2219). This title is pre-13 like *ArcRunner* (SDK 10) and *Crisis Core*,
  so `PROSPER_POST_SUBMIT_VISIBILITY=1` is a known lever if an unexplained mid-boot death appears
  later. It has **not** been needed so far and was not exercised — the two deaths seen here are both
  fully explained above.
