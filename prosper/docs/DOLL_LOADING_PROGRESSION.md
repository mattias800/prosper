# DOLL (PPSA17942, "Dragon Quest VII Reimagined") — why it never leaves the loading screen

> **RESOLVED 2026-07-10 (issue #306, branch `fix/issue-306-frontend-progression`).** The gate was
> exactly the §8 Lead-A hypothesis: the online init chain's success-with-garbage-out. Making the
> chain answer as an honest OFFLINE, SIGNED-OUT console (`hle_service.cpp`) drops the wall:
> `sceNpGetOnlineId` → `SCE_NP_ERROR_SIGNED_OUT` (0x80550006), Np state callback delivers
> SIGNED_OUT once via `sceNpCheckCallback`, NetCtl DISCONNECTED delivery default-on,
> `sceNetCtlGetInfo` → NOT_CONNECTED, ErrorDialog real lifecycle. **The instant the Np SIGNED_OUT
> state lands, the front-end resumes: title-screen assets stream in (T_Title_Clouds_*),
> pakchunk1/2 open, a ~1,000-chunk PlayGo locus sweep runs, the system+language save is created
> and committed (`SystemSaveData999.dat`, `LanguageSaveData998.dat`), Videodec2/Ajm init fires,
> and the game reaches an INTERACTIVE title screen** — a `PROSPER_PAD_SCRIPT` drive navigates
> title → save-slot menu (reads `GameSaveData*.dat` through `/savedata0`) → a content-load burst
> (~930 pak reads) → a much richer screen (draws/frame 100 → 356). The per-frame
> `sceErrorDialogUpdateStatus`/`sceNetCtlCheckCallback` pumps and the `sceGameUpdateCreateRequest`
> follow-ups never re-appear — the flow took its offline path (no patch check attempted, exactly
> like a disconnected retail console). **Next gate:** a MallocBinned3 heap-corruption fatal
> (`Canary was 0x3, should be 0x1`) during the menu-driven content load, present in every
> input-driven run including pre-#306-fix builds — tracked separately. The sections below are the
> original diagnosis, kept for the method.

**Verdict (2026-07-10, original diagnosis): STALLED — a correctness wall, not llvmpipe slowness.** The game reaches a fully-alive,
60 fps engine steady state within ~6 seconds of boot (all asset IO complete), renders its
`DOLLLoadingScreen` UMG widget forever, and never advances its boot flow. Wall-clock speed is not
a factor: with the renderer OFF the guest experiences a full 60 fps for 7+ minutes of game time
(25,000+ flips) and still never progresses, performs **zero** file IO after second 6, and never
enqueues a level-load task. Something in its boot-flow state machine never transitions.
CONFIDENCE: HIGH on the verdict. The missing piece is narrowed (by the runs below, incl. a direct
NetCtl-callback-delivery experiment that came back NEGATIVE) to the front-end boot flow's
**"content/patch/entitlement ready" gate** — the online / GameUpdate / EntitlementAccess init
chain returns success-with-garbage-out instead of an honest offline failure, so the flow can't
take its offline path to the title. CONFIDENCE MED on the exact subsystem, pending a trace of the
flow target eboot+0x124beb0.

Diagnosed 2026-07-10 on `diag/doll-loading-progression` (= master e116cd4) with the new
`PROSPER_PROGRESS` heartbeat + `PROSPER_PROGRESS_UNIMPL` call-count dump (this branch). All runs:
WSL2 ext4 fast path (`/root/PPSA17942-app0`), `PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1` — recorded as
run, but `PROSPER_GUEST_FS` was the then-current Linux opt-in gate for guest `%fs` TLS and has been
default-on since #825 (2026-07-17); today only `PROSPER_NULL_PAGE=1` is needed here (#2095).
Tracked as **issue #306** (`bug`, `area:ue4`).

## 1. The steady state, measured (run 1: no renderer, 420 s)

`[progress]` heartbeat every 5 s (submits / draws / dispatches / flips are cumulative):

| t | submits | draws_cum | dispatches | flips |
|------|---------|-----------|------------|-------|
| 60s | 4,054 | 334,258 | 100,801 | 3,565 |
| 180s | 12,319 | 1,059,829 | 315,961 | 10,737 |
| 300s | 20,587 | 1,787,135 | 531,212 | 17,912 |
| 415s | 28,531 | 2,485,159 | 737,822 | 24,799 |

- **Perfectly linear**: ~59.8 flips/s (vsync-paced 60 fps), ~69 submits/s, ~100 draws +
  ~1,500 compute dispatches per frame, for the entire run. The engine is not degrading, not
  loading, not leaking — it is parked in a steady frame loop rendering the same UI.
- **All IO happens in the first 6 seconds**: 781 file opens, 5,430 APR reads — 100% in minute 0,
  zero in minutes 1–6 (bucketed FILELOG). The last file activity (t=6 s) is the game
  **enumerating `movies/cutscene/` and `movies/ui/`** and opening the four
  `fms_ui_systemsettingcamera0*.usm` — it never opens `sms_opening_*.usm` (the opening movie).
- draws_last oscillates 100↔12–27: the loading widget is animating (the screen is live, not
  frozen).

## 2. Every thread is parked (run 4: gdb all-thread dump at t=180 s)

`tools/dbg/thr232.py` dump, 80+ threads, all healthy and ALL idle:

- GameThread (main): sleeping in the per-frame throttle (`clock_nanosleep`, no guest frames on
  the sampled stack) — ticking, not blocked.
- FAsyncLoadingThread: parked on its zenaphore — **async loading fully drained** (same as the
  session-4/5 finding in `UE4_APR_IOSTORE_BRINGUP.md`).
- IoDispatcher / IoService / 20+ TaskGraph + pool threads: parked in eq/cond waits.
- Game-side workers **DollLevelPreloader / SaveLoadUpdate / DLCDataUpdate / ShareUpdate**: parked
  in the generic pool-worker wait — named workers that were never given work.
- **The whole CRI middleware stack is up and idle**: CRI Server Manager, CRI FS File Access ×2,
  CRI FS Data Dec, CRI FS Memory F, CRI Audio Output, 12× CRI MPV Worker, CriManaDecodeTh —
  all in CRI's own sleep loops. FMediaTicker (UE media framework) idle.
- RenderThread / RHIThread / AgcSubmission / AgcInterrupt / AgcCleanup: normal frame-loop waits.
- OnlineAsyncTask + HttpManagerThread: parked (OnlineAsyncTaskManager has no completing work).

Nothing is deadlocked in a wait-primitive sense; the game's *flow state machine* simply never
transitions. This is scene non-activation at the game-flow level, exactly the shape of the
(pre-drawbuilder) #232 wall, now one screen further along.

## 3. What the game polls every frame (the smoking gun)

`PROSPER_PROGRESS_UNIMPL` (periodic `dump_call_log`) at t=241 s of run 2 — per-call counts of
UNIMPLEMENTED NIDs (deduped first-seen logging hides these; the counts do not lie):

| calls | NID | name (PS5 3.20 libs) | note |
|-------|-----|----------------------|------|
| 14,191 | libSceNetCtl `iQw3iQPhvUQ` | **sceNetCtlCheckCallback** | exactly once per frame |
| 14,191 | libSceErrorDialog `WWiGuh9XfgQ` | **sceErrorDialogUpdateStatus** | exactly once per frame |
| 227,380 | libSceAgc `bbFueFP+J4k` | sceAgcDcbSetPredication | per-draw, render-side |
| 113,690 | libSceAgc `xSAR0LTcRKM` | sceAgcDcbJump | per-draw, render-side |
| 113,690 | libSceAgc `w6Dj1VJt5qY` | sceAgcSetPacketPredication | per-draw, render-side |
| 1,079 | libkernel `rqwFKI4PAiM` | sceKernelAprWaitCommandBuffer | load-phase only |
| 951 | libkernel `6xVpy0Fdq+I` | _sigprocmask | |
| 50 | libScePosix `Xs9hdiD7sAA` | pthread_setschedparam | |
| 3 | libSceNpEntitlementAccess `jO8DM8oyego` | sceNpEntitlementAccessInitialize | **retried 3×** |
| 1 each | see §5 | one-shot init calls | |

The game **pumps `sceNetCtlCheckCallback` every frame, forever** (from the SystemEventGatherer
thread, run 6). This looked like a network-state spin, so it was tested directly
(`PROSPER_NETCTL_CB` experiment, runs 7 + 9): deliver the registered callback with
`DISCONNECTED` on first CheckCallback, plus `sceNetCtlGetState → DISCONNECTED`.

**Result: the callback delivers cleanly (correct guest-%fs handling, no crash) and the game
consumes it — `sceNetCtlGetState` is then called EXACTLY ONCE and never again — but the flow
still does not advance** (no new IO, no pad-open change, no new NIDs, identical steady state).
So the per-frame CheckCallback pump is the **generic PS callback pump** (it re-pumps every
registered callback each frame), NOT a spin on network state, and NetCtl delivery — while correct
infrastructure the game needed — is **not the gate by itself**. This is the key negative result
of the investigation: it rules out the most obvious per-frame signal and points the gate deeper
into the online/update/entitlement flow (below).

The surrounding one-shot init calls draw the picture of a boot flow stuck at its
**network/update/entitlement check step**:

- `sceGameUpdateInitialize` fired once — but `sceGameUpdateCreateRequest`/`sceGameUpdateCheck`
  (the calls that would actually perform the update check; full surface in
  `../PS5-3.20_Libs/libSceGameUpdate.c`) were **never called**. The update check started
  initializing and then waited on a precondition that never arrived.
- The eboot contains UE4's PatchCheck/InstallBundleManager subsystem
  (`EInstallBundleManagerPatchCheckResult::*`, "Could not download patch data. Please check your
  internet connection, and try again.") — the standard console boot-flow gate that DQ7R runs
  before the title screen.
- `sceNpRegisterStateCallbackA` (once): the Np sign-in state callback is registered and likewise
  never delivered (our `sceNpCheckCallback` is a return-0 no-op).
- `sceNpGetOnlineId` (once): unimplemented → returned **0 = success with a garbage out-struct**
  (the recurring success+garbage-out class) — a signed-OUT console must return the signed-out
  error here; "success" tells the game a user is signed in and can push the flow into online
  branches that then wait on Http/WebApi machinery that goes nowhere.
- `sceHttpInit`/`sceHttpCreateTemplate`/`sceHttp2Init`/`sceSslInit`/`sceNpWebApi2Initialize`/
  `sceNpCppWebApi ×4`/`sceNetResolverCreate`/`sceNetPoolCreate`: each once, all returning
  0-with-garbage — the whole online transport is a house of cards of fake handles.
- `sceNpEntitlementAccessInitialize` tried **3 times** (the only retried one-shot) — something
  keeps re-initializing entitlement access because the follow-up never works (DLC check; the
  parked DLCDataUpdate worker is its consumer).

`sceErrorDialogUpdateStatus` pumping per-frame is the standard per-frame dialog pump
(no `sceErrorDialogOpen`-class NID ever fires, so no dialog was actually opened — it is
housekeeping, not a stuck modal).

## 4. Ruled out by A/B or measurement

- **llvmpipe / render throughput (the "SLOW" hypothesis)**: with `PROSPER_RENDER=1` the guest is
  throttled hard by synchronous llvmpipe (see §7 numbers) — but with the renderer OFF the game
  runs a true 60 fps and *still* never progresses. Progression is not wall-clock-, IO- or
  GPU-bound. A faster GPU changes nothing about this wall.
- **Asset streaming**: all reads complete in 6 s on ext4; nothing is ever read again. Not
  IO-bound, not Oodle (containers are uncompressed), not a stuck async read (FAsyncLoadingThread
  drained; the #115/#208 completion machinery works — 5,430 APR reads served cleanly).
- **Movies**: `PROSPER_DENY_SUBSTR=.usm` (every movie ENOENT, run 2) changes nothing — same
  steady state, same per-frame pumps. The stall is upstream of movie playback (the opening movie
  is never even opened; only the 4 settings-screen UI movies get existence-probed at t=4 s).
- **Input** (corrected mid-investigation, then A/B tested): the game DOES open and poll the pad —
  `scePadOpen` at t=0 and `scePadReadState` at ~220 calls/s (visible only under `PROSPER_PADLOG`;
  the pad HLE is implemented, so its absence from the unimplemented log proves nothing). The
  neutral backend reports connected=0, so a "reconnect controller / press any button" gate was a
  live suspect. **Run 8a (`PROSPER_PAD_PRESS=1` — connected=1, CROSS held all run) does NOT
  advance the flow** (56,000+ pad reads with buttons=0x4000, zero IO after t=6 s). Not an
  input/controller gate.
- **NetCtl network state** (runs 7 + 9, `PROSPER_NETCTL_CB=1`): delivering the DISCONNECTED
  callback + GetState does NOT advance the flow either (§3). The most obvious per-frame signal is
  ruled out.
- **PlayGo / SaveData / Trophy2 / Share**: implemented with real contracts in the session-5 work
  (`hle_service.cpp`); the save-data flow completes end-to-end (files written). Not the gate.
- **GPU fence/EOP delivery**: 28,500 submits complete, flips fire, no FRenderCommandFence
  timeout, no watchdog abort — the #232/#236/#278 fixes hold.

## 5. One-shot unimplemented calls (complete list, run 1)

libSceAgc: dbOlWdppb4o (50×), qj7QZpgr9Uw (20×) [driver-info-class, unnamed];
libSceAjm: sceAjmInitialize, sceAjmModuleRegister; libSceAmpr: 4fgtGfXDrFc;
libSceCoredump: sceCoredumpRegisterCoredumpHandler; libSceGameUpdate: sceGameUpdateInitialize;
libSceHttp: sceHttpInit, sceHttpCreateTemplate; libSceHttp2: sceHttp2Init;
libSceJson2: 6 NIDs (parser objects for the online config);
libSceNet: sceNetPoolCreate (3×), sceNetResolverCreate;
libSceNetCtl: sceNetCtlRegisterCallback, sceNetCtlGetInfo, sceNetCtlCheckCallback (per-frame);
libSceNpCppWebApi: 4 NIDs; libSceNpEntitlementAccess: sceNpEntitlementAccessInitialize (3×);
libSceNpGameIntent: sceNpGameIntentInitialize; libSceNpManager: sceNpRegisterStateCallbackA,
sceNpGetOnlineId; libSceNpWebApi2: sceNpWebApi2Initialize, sceNpWebApi2PushEventCreateHandle;
libSceSsl: sceSslInit (2×); libSceVideodec2: sceVideodec2QueryComputeMemoryInfo (2×);
libSceVoiceQoS: sceVoiceQoSInit;
libkernel: sceKernelAio{InitializeImpl,InitializeParam,SubmitReadCommands,WaitRequest,
DeleteRequest} (the CRI FS IO path — will matter the moment a movie actually plays),
sceKernelAddAmprEvent, sceKernelAprWaitCommandBuffer (1,079×), _sigprocmask (951×);
libScePosix: pthread_setschedparam (50×).

## 6. Who makes the per-frame calls (runs 5/6: gdb probes)

- The per-frame tick object (`this` of eboot+0x5044740, vtable eboot+0x8a46708, `*0x288` target =
  **eboot+0x124beb0**) is completely static across frames — no RTTI typename resolvable; the flow
  state lives deeper (flow5.py).
- Caller-chain histogram of all unimplemented calls at steady state (flow6.py, breakpoint on
  `prosper_on_unimpl`, 40 hits):
  - The dominant chains are the **libSceAgc trio** (SetPredication ×2 / SetPacketPredication /
    DcbJump — import idx 1624/1592/1498) from four consecutive call sites in one guest function
    (eboot+0x2212144/57/64/6c ← eboot+0x2219bd6|0x2219d36 ← 0x2c61ae6 ← 0x2c841ea ← 0x2c8401a ←
    thread-runner) — the **RHIThread's command-list translate loop**. These are render-side
    (ignored predication/jump packets — a rendering-correctness item, not a progression gate).
  - `sceNetCtlCheckCallback` is pumped from **eboot+0x27ebb05 — the SystemEventGatherer thread's
    gather loop** (run-4 dump shows that thread's wait at eboot+0x27eba63). The delivered
    callback would fire on that thread and post state the GameThread flow consumes.
  - `sceErrorDialogUpdateStatus` is called from a non-eboot module (no eboot RAs on its stack).

## 7. Render-path throughput (run 3: PROSPER_RENDER=1, llvmpipe)

For completeness (the game is STALLED, so throughput never lets it "finish" — but it quantifies
the render cost for when the wall falls). `execute_and_present` is **synchronous on the guest
submit thread**, so each presented frame blocks the whole guest:

| t | flips | presents | submits |
|------|-------|----------|---------|
| 69 s | 34 | 33 | 39 |
| 88 s | 38 | 38 | 44 |

- **~0.43 flips/s with the renderer on** vs **~59.8/s off** — a ~140× slowdown; each llvmpipe
  present (reduced-res draw list) costs ~2 s and stalls the guest for its duration. This is the
  documented synchronous-llvmpipe cost, not a new bug. It confirms a real-GPU Vulkan path (vs
  llvmpipe) is the unlock for *watchable* frame rate — but it is orthogonal to the progression
  wall (§1: the game does not progress even at a free-running 60 fps with the renderer off).
- The renderer presents DOLL's real backbuffer compose (the loading screen) every frame; 42 BMPs
  captured to `/root/doll_r3_frames/` (never committed).

## 8. The highest-leverage next step

**Two parallel tracks. NetCtl delivery is DONE-and-insufficient (proven), so the lead is the
online/update/entitlement honesty pass + a flow-target trace.**

Tested and insufficient (kept as gated experiment `PROSPER_NETCTL_CB=1`, `hle_service.cpp`):
NetCtl callback delivery (`sceNetCtlRegisterCallback`/`CheckCallback`/`GetState` as an offline
console). It delivers cleanly and is consumed, but does not move the wall — see §3. Land it as
correct infra, but it is not the gate.

Lead A — **make the whole online/update/entitlement init chain honest** (one focused
`area:ue4` PR, shared-infra files):
1. `sceNpGetOnlineId` → signed-out error (PS4: 0x80550006 SCE_NP_ERROR_SIGNED_OUT; verify PS5
   value) instead of success+garbage; `sceNpRegisterStateCallbackA`/`sceNpCheckCallback` →
   deliver SIGNED_OUT once (same guest-%fs callback discipline the NetCtl experiment proves
   works).
2. `sceNpEntitlementAccessInitialize` (retried 3× — a real "keeps failing" signal) and
   `sceGameUpdateInitialize`/`CreateRequest`/`Check`: return honest FAILURE or a valid
   "no patch required / no entitlement, offline" result per their real ABI, so UE4's
   `InstallBundleManager` PatchCheck resolves to `NoPatchRequired`/`NoLoggedInUser` and the flow
   takes its offline path rather than waiting on fake Http/WebApi handles.
3. If dismissing produces the "Could not download patch data..." dialog, give `sceErrorDialog`
   (or CommonDialog) the minimal Initialize/Open→Status FINISHED lifecycle so it can be
   dismissed, then drive the confirm with `PROSPER_PAD_SCRIPT` (#202).

Lead B — **trace the flow gate directly** (no guessing): the front-end tick calls through
`vtable*0x288 = eboot+0x124beb0` (run 5). Break there / disassemble it and the predicate it
returns to; it decides "advance to title / begin level load." Whatever boolean or async result it
polls names the exact subsystem — feed Lead A from it. `tools/dbg/flow5.py` + `flow6.py` are the
starting probes.

Verification for either: re-run run 1's heartbeat — the wall has fallen when IO resumes after
t=6 s (front-end map load), the pad-open path changes, or `sms_opening_en.usm` finally opens.
The next surfaces are already visible (§5: Videodec2 + sceKernelAio for the opening movie, then
pad → title menu → `PROSPER_PAD_SCRIPT` press Start → first level).

## Appendix: how to reproduce

```
# heartbeat + call-count boot (no renderer, full speed):
timeout 420 env PROSPER_NULL_PAGE=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 \
  PROSPER_PROGRESS=5 PROSPER_PROGRESS_UNIMPL=1 \
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/doll.log 2>&1
# scripts: scripts/diag/run{1..5}*.sh on diag/doll-loading-progression
# thread dump: gdb -p <pid> -batch -x tools/dbg/thr232.py
# flow probe:  gdb -p <pid> -batch -x tools/dbg/flow5.py
```
