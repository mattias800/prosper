# Graphics & Audio bring-up (the M4/M5 frontier)

Status as of the boot reaching multithreaded graphics/audio initialization. The game now runs its
**entire non-graphics runtime** — IL2CPP init, C# startup, the runtime main flow, splash/message
dialog, the flip/render loop scaffolding — and enters GPU + audio setup. This doc is the blueprint
for turning that into actual rendered frames.

## How far the boot gets

```
loader → CRT → C++ ctors → il2cpp_init (GC, metadata, type system) → runtime startup
  → sceSystemServiceHideSplashScreen → sceMsgDialog (auto-dismissed)
  → GRAPHICS init: libSceAgc (GPU command build) + libSceAgcDriver (submission)
  → libSceVideoOut (display / flip / vsync event queues)
  → AUDIO init: libSceAmpr
  → [BLOCKED] multithreaded null-object derefs in graphics/audio worker threads
```

The block is the **headless limit**: our placeholder graphics/audio objects are zeroed, so worker
threads eventually read a null sub-object pointer out of them and dereference it (e.g.
`eboot+0x3b5ea6` `[null+0x30]`, `eboot+0x149c99c` `[null+0x18]`). Zeroed placeholders no longer
suffice — the game needs **real object graphs**, i.e. the actual graphics/audio subsystems.

### The terminal fault is unimplemented libSceAgc, NOT a C++ locale bug (verified 2026-07-04)

**This corrects two earlier mis-diagnoses in this file's history** (first "rune facet never set during
static init", then "std::ctype locale facet array left zero"). Both were wrong: the table-lookup
*instruction shape* at `eboot+0x3b5ea0` (`movzwl 0x2e(%rdi,%rcx,2)`) merely *resembles* `std::ctype`
classification. The object it reads actually comes from an **unimplemented libSceAgc initializer**.

The last log line before the crash is Unity's own `todo: void GfxDevicePS5SharedData::CreateWorkload()`
— i.e. we are inside Unity's PS5 GPU-device setup. The crash **site varies across runs** (multithreaded
graphics workers): `eboot+0x3b5ea6` (`addr=0x30`), `eboot+0x149c99c` (`addr=0x18`), … — all null-field
derefs of zeroed graphics objects, not one deterministic path.

Verified chain for the `CreateWorkload` object (gdb, `break run_entry` first; `pltm` + the unimpl log):

- In `eboot+0x14dd*`'s fn: `obj = r14+0x48` (`lea 0x48(%r14),%r12` @`+0x14dda75`), then immediately
  `call 0x4003ae7d0(obj)` — the object's initializer.
- `0x4003ae7d0` is a **PLT stub**: `jmp *[0x401d95858]`. At runtime that GOT slot holds `0x600004180`,
  which is one of **our unimplemented-import stubs** (`mov $idx,%edi; movabs $prosper_on_unimpl; jmp`).
- `pltm eboot.bin known_names.txt 1d95858` → NID **`+kSrjIVxKFE`**, and the boot log shows
  `unimplemented: libSceAgc::+kSrjIVxKFE -> returning 0`. So `obj`'s initializer is an **unimplemented
  libSceAgc function** that does nothing.
- The caller then passes that *same* `obj` (`r12`, stored at `[rbp-0x198]`) to `0x4003a7b60`, which
  walks `obj+0x38` and reads `[obj+0x40]` as a pointer → it's null (never set by the stubbed init) →
  SIGSEGV. `PROSPER_FAULTMEM` confirms the whole `obj` region is zero at the fault.
- `_Getpctype` works and is called, but only for an unrelated inline `isspace`-style loop
  (`eboot+0x82e893`, `test $0x144`) — it is NOT part of this path.

**Conclusion:** the blocker is the **libSceAgc GPU object graph**. `CreateWorkload` (and sibling
graphics init) call ~24 libSceAgc functions — `23LRUSvYu1M`, `BfBDZGbti7A`, `+kSrjIVxKFE`,
`H7uZqCoNuWk`, `vRoArM9zaIk`, … (full list via `boot_trace`) — **all unimplemented, all returning 0**,
so the GPU objects they should build stay null and the graphics workers deref null. This is squarely
the M4/M5 libSceAgc→Vulkan work below; there is no locale/libc gap to fix. Faking these objects with
plausible-looking fields would be "limping to graphics" (violates correctness-first) — they need the
real AGC object model, reverse-engineered from the call args (`PROSPER_GFXLOG`) + AGC semantics.

**Tooling:** `PROSPER_FAULTMEM=1 ./build-linux/boot_trace <dump>` dumps every GP register + 4 qwords of
guest memory at each pointer-looking one, at fault time on the stopped thread (reliable; live gdb
breakpoints race in this multithreaded, signal-scheduled guest). `pltm eboot.bin known_names.txt
<got_off>` maps a GOT slot to its import NID/name.

**AGC call tracing (RE bootstrap for M4).** All 28 libSceAgc/AgcDriver NIDs the game calls are routed
through per-NID logging thunks (`glog_thunk<I>` in `hle_graphics.cpp`; behaviour unchanged — still
returns 0). `PROSPER_GFXLOG=1 ./build-linux/boot_trace <dump>` now emits a **self-describing** line per
call: `libSceAgc::<NID>  from eboot+0x<callsite>  a0..a5`. ~148 AGC calls fire before the fault.
Call-frequency profile of one run (NID ×count — the hot ones are the AGC command/descriptor ops to
understand first; args' high addresses are heap/GPU-VA that vary per run, but NIDs + callsites are
stable):
- `f3dg2CSgRKY` ×36 — hottest; a per-op/per-command call.
- `d-6uF9sZDIU` ×25, `ZvwO9euwYzc` ×25 — next hottest, paired.
- `TRO721eVt4g` ×5 — `CreateWorkload` per-object init (`eboot+0x14e6661`, `a0=a3=obj, a4=obj-0x48`);
  `obj` is the very object whose null field later faults, so this call (or `+kSrjIVxKFE` ×3, the
  initializer at `eboot+0x3ae7d0`) is what should populate it.
- device/context level (once each): `23LRUSvYu1M`, `BfBDZGbti7A`, `XlNp7jzGiPo`, `MM4IZSEYytQ`.
Next: map each hot NID + arg pattern to the AGC API to build the real object model, then implement the
initializers to construct valid GPU objects (correctness-first — no plausible-looking fakes).

## What's already in place (headless bring-up, correctness-first)

- **Unified GPU memory (lazy)** — `exec_image_linux.cpp` fault handler backs any unmapped page in
  the GPU-VA window `[4 GiB, 64 GiB)` with a real zeroed page on demand and retries. This models the
  PS5's unified CPU/GPU memory (GPU VAs are real RAM). Low-address null derefs stay fatal. It got the
  boot past the format-table fault at GPU VA `0x100000000` into audio init. **Contents are zero until
  the driver layer is real** — a documented placeholder, not faked output.
- **libSceVideoOut** (`hle_graphics.cpp`) — `Open`→handle, `SubmitFlip` increments a flip counter,
  `GetFlipStatus` reports it (correct 0x40 struct — do NOT over-write, it smashes the guest canary),
  `IsFlipPending`→false, event/flip machinery no-op. Simulated flip completion so the render loop
  advances.
- **libSceAgc** getters that return dereferenced objects → stable zeroed singletons.
- **event queues** (`hle_kernel_time.cpp`) — valid queue objects; `WaitEqueue` yields + reports no
  events.
- **Diagnostics** — `PROSPER_GFXLOG` logs AgcDriver call args. Sampled-texture isolation accepts
  `PROSPER_TESTTEX_DRAW=N`, `PROSPER_TESTTEX_BINDING=B`, and `PROSPER_TESTTEX=zero|checker`; both RGBA8 and
  renderer-owned RGBA16F RTT inputs are replaced in their native format. `PROSPER_TESTTEX_FILTER=linear|point`
  uses the same draw/binding selectors for sampler-only A/B tests. `PROSPER_DUMP_RTGROUPS_ADDR=0x...` scopes
  `PROSPER_DUMP_RTGROUPS` to one target during a long run. `PROSPER_SHADER_DUMP_SUCCESS=DIR` captures both the
  exact raw RDNA2 bytes and translated SPIR-V for successfully recompiled graphics and compute shaders. These are
  diagnostics, not renderer policy; record an unmodified output before interpreting an override.

## Build environment

- `libvulkan.so.1` (1.3.275) **is present**; Vulkan **headers are not** (`apt install libvulkan-dev`
  or vendor `vulkan/`). No SDL2/X11 → plan **headless/offscreen Vulkan** first (WSL has no display).
- Graphics libs are sparsely documented; most NIDs don't resolve to names — reverse-engineer from
  call args (`PROSPER_GFXLOG`) and `build-linux/tools/pltm` (maps a module GOT offset → NID).

## ✔ USER-DATA PROBE (2026-07-04): zero-varying fault pair still has PS resource bindings

Follow-up to PR #14's fork between "legitimate zero-varying pipeline, bad resident flag" and
"`[+0xc0]` is fed by shader resource bindings." `PROSPER_PIPETRACE` now logs the decoded
`ShaderUserData` resource-binding table: direct resource offsets, `eud_size_dw`/`srt_size_dw`, and
the four sharp-resource arrays (`sharp[0]` texture, `sharp[2]` sampler, `sharp[3]` storage buffer).

One traced WSL2 boot of the faulting pair showed:

```
CreateInterpolantMapping gs=...c0620 ps=...c0dd0 -> mapping 0 interpolants
  gs user_data: direct_count=11 sharp_counts={0,0,0,0}
  ps user_data: direct_count=11 sharp_counts={0,0,0,1}
    sharp[3] storage: slot0={off=0000,size=1}
```

So the semantics conclusion still holds: this is a genuine zero-varying pair. But the pixel shader is
not resource-empty: it has one storage-buffer sharp binding. That makes hypothesis (2) the better next
thread: the pipeline reflection table `[pipeline+0xc0]` is likely populated from shader resource
bindings/user-data as well as, or instead of, interpolants for this path. The next probe should catch
the Unity builder that folds `ShaderUserData` into `[pipeline+0xc0]/[+0xe0]` and compare the expected
storage binding against the records seen by predicate `eboot+0xd58710`. Chasing `[obj+0x1a0]` first is
lower signal until the resource-binding reflection path is ruled out.

## ✔ SEMANTICS PROBE (2026-07-04): metadata is surfaced CORRECTLY — the "mis-relocation" bet is rejected

Agent-2's fastest-to-confirm bet was that our shader I/O semantics are empty/mis-relocated, making the
interpolant mapping short as a symptom. **Rejected, conclusively, via raw blob bytes** (PROSPER_PIPETRACE
now logs semantic counts/arrays + the raw `0x50..0x5f` header region):

- Shaders that HAVE varyings read correctly: `num_input_semantics=1` (`raw[0x50]=01 00 00 00`),
  `num_output_semantics=1` (`raw[0x56]=01 00`), and `output_semantics[0].semantic=0x0f` (=15) — exactly
  the value consumed by the observed VS→PS linkage. So our offsets (in u32@0x50, out u16@0x56, arrays@0x30/0x38)
  and relocation are correct.
- The **faulting pipelines** use `gs` (num_out=0, `raw[0x56]=00 00`) + `ps` (num_in=0) — genuinely
  **zero-varying shaders** (position-only/blit). All 3 `CreateInterpolantMapping` calls therefore map 0
  interpolants — a correct consequence, not a surfacing bug.

**Conclusion:** the empty pipeline-reflection table `[pipeline+0xc0]` is NOT caused by missing/mangled
semantics. Either these are legitimately 0-varying pipelines where the reader at `0xba6e08` should
tolerate a null companion (and our divergence is that the "resident" flag `[obj+0x1a0]` never gets set),
or `[+0xc0]` is populated from **resource bindings** (textures/samplers/cbuffers via the shader's
`user_data` table) rather than interpolant semantics. Next probe target: capture `[pipeline+0xc0]`/
`[+0xe0]` contents directly (the reflection records) and check the shader `user_data`/resource tables,
not the semantics. `PROSPER_PIPETRACE` retained (semantic + raw-header logging).

## ✔ SKIP PROBE (2026-07-04): the null companion is systemic, not a single gap → STOP hunting per-object writers

Env-gated diagnostic `PROSPER_SKIP_NULL_COMPANION` (exec_image_linux.cpp; default off): at the reader
`eboot+0xba6e08`, log the object's state and redirect RIP to the reader's own skip label
`eboot+0xba6e40` (where its type/flag branches already land), continuing as if the companion weren't
needed. A probe to reveal the *shape* past the fault — NOT a fix (companions are still not real).

Result of one run: **4 bounded skips** (distinct objects, all `[+0x140]=0`; resident-flag low byte
`[+0x1a0]&0xff == 0` on every one — none processed), then the boot advances **past `0xba6e08` to a
NEW null-object deref at `eboot+0x95c823`** (`mov (%r14),%rdi; mov (%rdi),%rax` with `[r14]`→null).

**Interpretation (per the agreed decision tree): systemic, not a narrow gap.** It is NOT an infinite
cascade of the same deref (bounded at the collection size = 4), and it does NOT reach a frame — it
lands on the *next* null GfxDevice object. So the pipeline-residency subsystem simply never ran: every
pipeline lacks a companion and the resident flag, and past that lies another null object of the same
"placeholder is zeroed" kind. **Chasing per-object `[+0x140]` writers is unproductive — skipping one
only reveals the next null.** The productive direction is systemic: drive Unity's deferred
processing/residency pass (the "deferred to a submit/flip/frame boundary we never pump" theme — no
`SubmitFlip` fires before the fault), which ties into the swapchain scaffolding (`prosper_vo_*`) and
the back-half pipeline realization. Next: find what triggers the residency pass and whether pumping
submit/flip fires it, rather than reconstructing per-object writers.

## 🔎 THE [+0x140] WRITER CHAIN (2026-07-04) — host-side RE, found the constructor + gate, root is a pipeline-reflection predicate

Agent-2 (correctly) reassigned this to the front-half: the companion is built pre-submit, upstream of
any GpuState. Traced the writer/constructor of the null `[obj+0x140]` and its gating condition:

**The writer** — `eboot+0x15aef7d..0x15aef83`:
```
g = *[0x1ff28c8]                       ; factory/device singleton (global)
[r13+0x140] = g->vtable[0x88](g, 0, payload)   ; create the GPU companion  <-- THE WRITER
```
The sibling just above (`+0x138`) is the same call with arg1=1 (`g->vtable[0x88](g,1,..)`) — so
`+0x138`/`+0x140` are a pair of companions (two shader stages) created from the same factory.

**The factory is fine (not the cause).** `[0x1ff28c8]` is a lazy Meyers singleton constructed locally
at `eboot+0xc33769`: `operator new(...)` (call `0x809830`) + vtable `0x1d6b8f8` stored at `(obj)`,
then `[0x1ff28c8]=obj`. **No AGC/gfx call gates it** — our heap works, so `g` is non-null at runtime.
So `+0x140` is null NOT because the factory is missing.

**The proximate gate** — the writer block is skipped unless an optional local is present:
```
0x15aef69:  mov -0x200(%rbp),%rcx ; test %rcx,%rcx ; je 0x15aef8a   (skip the +0x140 create)
```
`[rbp-0x200]` is the "present" flag of an optional{payload@-0x210, present@-0x200}, zero-initialized
and filled only if a predicate passes:
```
0x15aee82:  call 0xd58710 ; test %eax,%eax ; je 0x15aef30   (skip filling the optional)
```
(The `+0x138` companion has the identical structure gated by `[rbp-0x1e0]` via the same predicate at
`0x15aec7a`.)

**The root predicate `eboot+0xd58710`** decides whether each companion is created. It reads
`[pipeline+0xe0]` (if 0 → early-out) and iterates the record array at `[pipeline+0xc0]` (0x20-byte
records, checking byte fields at `+0x02`/`+0x22`) — i.e. it scans the pipeline's **reflection/binding
table** and returns non-zero when a matching entry exists. For our 3 resources the second call returns
0 → the `+0x140` optional stays empty → the companion is never created → `+0x140` null → the
`0xba6e08` reader faults.

**So the true root is upstream of the writer: the pipeline's reflection collection at `[pipeline+0xc0]`
is empty/short for these resources**, so `0xd58710` reports "no second-stage companion needed" and the
writer legitimately skips it — but the *reader* at `0xba6e08` derefs it anyway. That `[+0xc0]`
reflection table is populated from the shader register/semantic data that flows through
`CreatePrimState`/`CreateInterpolantMapping` (which we implement into stack scratch, then Unity folds
in). **Next probe (joint):** capture `[pipeline+0xc0]`/`[pipeline+0xe0]` for the faulting objects and
compare against what our `CreateInterpolantMapping` produced — a short/empty interpolant set would
explain the empty reflection table. This is the remaining thread to the true root cause.

## ✔ libSceVideoOut real 1080p60 + swapchain scaffolding (2026-07-04) — and the [+0x140] gate is NOT the display surface

Implemented the 5 previously-unimplemented VideoOut NIDs with real, self-consistent values (resolved
through firmware symbols, guest call sites, and PS4-inherited public contracts): `Nv8c-Kb+DUM` sceVideoOutIsOutputSupported, `PjS5uASwcV8`
sceVideoOutSetBufferAttribute2, `rKBUtgRrtbk` sceVideoOutRegisterBuffers2, `utPrVdxio-8`
sceVideoOutGetOutputStatus, `w0hLuNarQxY` sceVideoOutConfigureOutput. Plus fixed GetResolutionStatus
(was all-zero → now 1920×1080@59.94Hz) and added GetVblankStatus (advancing counter) +
GetDeviceCapabilityInfo (SDR). All output-struct writes are size-exact.

**Verified the game requests exactly what we advertise:** `SetBufferAttribute2` arrives with
`width=0x780 (1920)`, `height=0x438 (1080)`, `pixel_format=0x8000000000000000` (PS5 A8R8G8B8 sRGB);
`RegisterBuffers2` registers **3** framebuffers (triple-buffered). Recorded them in a display-buffer
registry (swapchain scaffolding, `prosper_vo_buffer_count/_display_width/_height/_format/_buffer_addr`)
that the back-half present path turns into swapchain images. test_videoout (27 tests total).

**Hypothesis result (agent-2 asked): the display surface is NOT what gates the `[+0x140]` companion.**
With all 5 VideoOut calls returning real 1080p60 values + buffers registered, the boot still faults at
**the same `eboot+0xba6e08`** (unchanged). So VideoOut is eliminated as the suspect — Unity allocates
the pipeline companion independent of display-surface availability. The residency interception belongs
in the back-half (GpuState/register-context stage), as concluded below.

Note: `GetOutputStatus` currently traces + returns success without writing its output struct — its
exact layout/size is unconfirmed and the game tolerates a no-write return (boot reaches the same
downstream fault). Left non-writing rather than risk a wrong-size write (stack-canary smash).

## ✔ BOUNDED TRACE ANSWER (2026-07-04): the pipeline object is Unity-INTERNAL — no HLE hook exists

Back-half asked one bounded question: does the fault object `r15` (or its `[+0x18]`/`[+0x40]`
packed-register sub-object) ever appear as an arg / return / write of any AGC call we implement
(`CreateShader`, `CreatePrimState`, `CreateInterpolantMapping`, or the register-context builders)?

**Answer: NO.** Logged every construction call's pointer args (`PROSPER_PIPETRACE`) and diffed against
the fault object in the same run (`r15=0x7d622562c300`, sub-object `0x7d62256120f0`):
- `CreateShader` (×36+): header/code blobs live in `0x7d6252dc…`; `*dst` writes into the eboot BSS
  shader registry (`0x402048…`) or a stack slot — never `r15`'s heap region.
- `CreatePrimState` / `CreateInterpolantMapping`: write register `{offset,value}` pairs into **stack
  scratch** (`a0/a1 = 0x7d6255dfb…`); shaders are `0x7d6252dc…`.
- The only `0x7d6225…`-region pointers logged are the AGC context (`…5cb378`) and register-context
  builder args — none in `r15`'s (`…62c300`) or the sub-object's (`…6120f0`) region.

So the register *content* originates from our calls (Unity harvests it into stack scratch), but the
**pipeline object and its `[+0x140]` GPU companion are constructed by Unity's own resource manager** —
there is no game→AGC call that takes/returns/writes them, hence no front-half HLE hook. Per the
back-half's decision tree, pipeline residency will be intercepted at the GpuState/register-context
stage in the back-half; the front-half is off the hook for this object. `PROSPER_PIPETRACE` (logs the
raw pointer args of the shader/pipeline construction calls) is retained for future seams.

## ⚠ RE UPDATE (2026-07-04): the 0xba6e08 object is a Unity PIPELINE object, and NO resource-creation call feeds [+0x140]

Investigating the resource-layer integration (gpu_resources.hpp contract) turned up evidence that
**contradicts the "hook one AGC resource-creation NID → store handle at [obj+0x140]" plan** — flagging
for the back-half agent before writing integration code (per "flag it rather than edit silently").

Findings (via the multi-spec/deref `PROSPER_PEEK`, see below):
- **No AGC/Gnm resource-creation call fires before the fault.** The complete graphics-call set in the
  run is register setup (`23LRUSvYu1M`/`BfBDZGbti7A`/`H7uZqCoNuWk`/register-indirect triplets), the
  register-context builders (now implemented), and `Zw7uUVPulbw`. The only unimplemented graphics
  calls at fault are 5 pre-graphics `libSceVideoOut` queries. There is no texture/buffer/`createResource`
  call with a `(gpu_addr, size, width, height, format)` shape anywhere in the trace.
- **`Zw7uUVPulbw` is a red herring** — disassembly of its consumer (`eboot+0x14dfb04`) shows a
  GPU-timing/profiler loop (computes frametimes: `×0xf4240`, reciprocal-divide, `vcvtsi2ss`→float,
  stored to a ring at `rbx+0xc40`). It does not gate resource upload.
- **The fault object is a Unity pipeline/material object, not a texture/buffer.** `PROSPER_PEEK` of the
  faulting `r15` (`0x…e2c300`, game-heap): `[+0]=0x2b`, `[+8]=0xf`, `[+0x20]=7`, `[+0x28]=0x19`,
  `[+0x18]==[+0x40]`→ a shared sub-object holding **packed register-like fields** (`[+0x10]=0x28a7…000e`,
  `[+0x18]=0xbba2…002b`). `[+0x140]` (the GPU-backing companion, deref'd as `[+0x08]`/`[+0x40]` byte)
  is null; `[+0x520]/[+0x530]` (an array ptr/count) are 0. It sits in a **3-element collection**
  (outer `rbx`: array@`+0x78`, count@`+0x88`=3).
- The deref is **residency-gated**: `0xba6e08` is reached only for resources whose remapped type is in
  mask `bt 0xc8220` (specific formats) AND whose flag `[+0x1a0]==0` ("not resident"). So this is a
  GPU-residency check that assumes an upload/creation step already populated `[+0x140]`.

**Interpretation:** Unity's resource-upload/creation path (which on PS5 would allocate GPU memory +
build a descriptor and set `[+0x140]`) either never ran or was no-op'd by our stubs — but it is *not*
a direct game→AGC `createResource` call we can hook. Populating `[+0x140]` with a fabricated handle to
"advance the boot" would be a correctness-first violation (a fake that moves the fault deeper).
**Proposed next step (needs back-half input):** trace Unity's texture/pipeline upload path to find the
real creation site (likely built from the shader/pipeline AGC calls already implemented +
direct-memory allocation), then wire `resource_create()` there. The `ResourceDesc` contract looks
right for the *eventual* texture case; this specific object is a pipeline object whose companion may
warrant a distinct `ResourceKind`. Left unresolved pending that trace rather than guess-fabricated.

Tooling: `PROSPER_PEEK` now takes multiple `;`-separated specs and a `*pre+off` one-level pointer
chase (`[[reg+pre]+off]`), for classifying linked object graphs at fault time.

## ▶ NEXT FRONTIER (2026-07-04): past all AGC HLE → Unity GPU-resource residency (needs real backing)

After CreateShader + the CommandProcessor submit path (below), the boot advances through the entire
AGC command frontend and now faults **with ZERO unimplemented libSceAgc calls remaining** (only 5
pre-graphics libSceVideoOut queries stay stubbed, tolerated). The command buffer executes:
`sceAgcDriverSubmitDcb` → `gpu::run_command_buffer` folds it into a GpuState (verified: "SubmitDcb #1:
71 dwords -> 12 packets applied"). Unity then proceeds to load `unity default resources`.

**The fault: `eboot+0xba6e08`**, a Unity `GfxDevice` routine (`eboot+0xba6720`, reached via the
`0xd3xxxx`/`0xd4xxxx`/`0x15fxxxx` GfxDevice chain) iterating a resource list. For a resource whose
type is in a specific mask (`bt 0xc8220`) and whose flag `[obj+0x1a0]==0`, it reads a GPU-backing
pointer `[obj+0x140]` and dereferences `[that+0x08]` → SIGSEGV at addr 0x8 because `[obj+0x140]` is
null. `PROSPER_PEEK="r15:0x140,0x1a0,0x520,0x530"` at fault shows the object (a **game-heap**
allocation `0x…e2c300`, i.e. Unity's own, not one of our zeroed singletons) is only partially
initialized: `[+0]=0x2b`, `[+0x140]=0`, `[+0x1a0]=0`, `[+0x520]=0` (array ptr), `[+0x530]=0` (count).

**Diagnosis:** this is no longer an AGC-HLE gap — it is the **AGC→Vulkan resource-backing boundary**.
Unity created a resource object but its GPU-side backing (`+0x140`) was never populated, because our
AGC resource path returns handles/zeros without constructing real GPU objects. Pushing past this
means the real GPU resource layer (textures/buffers/render targets backed by Vulkan), which is the
M4/M5 work in the "Recommended implementation order" below — largely the back-half (render_state /
vk_translate / command_processor → live Vulkan resources) now fed by real submitted command buffers.
Fabricating a `+0x140` object would be a correctness-first violation (a fake that moves the fault
deeper), so this is the point to build the real resource backend rather than stub further.

New diagnostic: `PROSPER_PEEK="rN:0xoff,0xoff,…"` (exec_image_linux.cpp) reads arbitrary offsets off a
register at fault time — for classifying large objects past FAULTMEM's 0x20-byte window.

## ✅ THE BOOT BLOCKER — RESOLVED (2026-07-04): the "source" is the Shader; CreateShader was the gap

**Resolution (supersedes the "SDK-gated / parked" conclusion below).** The null register-source
global `[eboot+0x2048c60]` was never libSceAgc-private state: it is **field +0x10 of a 0x28-byte
shader-registry slot at `0x2048c50`** — one of ~30 slots for Unity's built-in shaders, registered
at graphics init by `eboot+0x14bc002..` → `eboot+0x14e74c0(slot, shader_elf, flag)`. That function
parses a **shader ELF embedded in eboot rodata** (`e_machine=0xe0` EM_AMDGPU; sections
`.shader_header` / `.shader_text`) and calls `sceAgcCreateShader(&slot->shader, header, code)`
(NID `f3dg2CSgRKY`, via the arg-validating wrapper `eboot+0x3ae120`). Our stub returned 0 without
writing `*dst`, so every slot's shader stayed null. The earlier static-scan conclusion "no eboot
code writes the global" was the classic computed-addressing blind spot (same failure mode as the
RGCTX hunt): the writer stores through `slot+0x10`, never through the literal address.

The **register source object IS the Shader**: `SetSource` (eboot+0x3af400) reads `[src+0x08]`
(user_data), `[src+0x28]` (specials), `[src+0x5a]` (type) — the layout established by the guest's own
accesses and captured header bytes. The "classify table" ships inside each shader blob; nothing is fabricated.

**Implemented in `hle_agc.cpp` (all real semantics, layout-verified via eboot disassembly and captures):**
- `f3dg2CSgRKY` **sceAgcCreateShader** — relocates the header's self-relative pointers in place,
  binds the code pointer, patches the leading `SPI/COMPUTE_PGM_LO/HI` sh-register pair (all five
  stage pairs, beyond the initially observed ES/PS pair), guards double-relocation, writes `*dst`, and registers
  the shader in a host-side registry (`prosper_agc_shader_count()`) for the AGC→Vulkan pipeline.
- `V++UgBtQhn0` **sceAgcGetDataPacketPayloadAddress** — called from *inside* eboot's static AGC
  code (register-bank prepare, `eboot+0x3af040`): the returned payload becomes the register bank
  `[sub+0x10]`/`[sub+0x18]`. The banks live in the game's own Dcb data packets.
- `n2fD4A+pb+g` **sceAgcCbSetShRegisterRangeDirect** — IT_SET_SH_REG range packet (+ the marker
  NOP the real library emits).
- `D9sr1xGUriE` **sceAgcCreatePrimState**, `HV4j+E0MBHE` **sceAgcCreateInterpolantMapping** —
  pipeline registers derived from the bound shaders' specials/semantics using generalized semantic
  matching rather than a hard-coded identity layout.

**Result:** all 36 built-in shaders register (`PROSPER_GFXLOG` shows `pgm_patched=1` on each), the
whole `CreateWorkload` register-context chain (`0x3b5ea6` → `0x3b1562` → `0x3b1533` → `0x3afcff`)
completes, and **no unimplemented libSceAgc call remains in the boot**. The boot now faults much
later at `eboot+0xba6e08` (addr=0x8, non-AGC backtrace via `0xd3xxxx`/`0x15fxxxx`) — the next,
separate frontier. Note: the game passes AGC interface version **13**; layouts are verified against
this title rather than inferred from earlier-generation material.

Tooling added: `build-linux/imgdump <module> <out.img>` dumps a module's flat image for offline
`objdump -D -b binary -m i386:x86-64` disassembly (how the registry writer was found).

## (Historical, disproved by #641) `+kSrjIVxKFE` context-init theory

> **Correction (2026-07-13):** the authoritative PS5 3.20 symbol map identifies
> `+kSrjIVxKFE` as `sceAgcDcbPushMarker`, not a register-context constructor. Its first argument is
> a live DCB and its second argument is the marker label. The temporary `g_agc_ctx_init` handler
> described below corrupted that DCB by clearing three 0x70-byte regions on every marker. Removing
> it and emitting a correctly framed marker packet lets Blasphemous 2 render its studio logos and
> title and continue through the EULA. See #641 and `docs/AGC_TRACE.md`. The following section is
> retained only as the reasoning trail that produced the obsolete workaround; none of its
> conclusions about the import's identity or ownership are current.

The boot faults at `eboot+0x3b5ea6` inside a GPU register-setting routine. Full chain, traced under gdb:

- During `GfxDevicePS5` graphics init (a `CreateWorkload`-style fn at `eboot+0x14dd900`), the game
  computes its **register context = device+0x48** (embedded object) and calls `+kSrjIVxKFE(context)`
  as the very first operation on it (`eboot+0x14dda7c` → thunk `eboot+0x3ae7d0` → PLT `eboot+0x19b4730`
  → GOT `+0x1d95858`).
- That GOT slot resolves to **our stub** (`0x600003140`) which tail-jumps to `glog_thunk<14>` —
  i.e. `+kSrjIVxKFE` is `kAgcNids[14]`, currently an **observe-only logger that returns 0**.
- So the context is never initialized: it stays fully zeroed. `[context+0x08]` (the register-index→
  hardware-slot *classify table*), `[context+0x10]`/`[context+0x18]` (the two register-bank output
  buffers) are all null.
- The following register-set loop (`eboot+0x3afb90`, reached via thunk `eboot+0x3a7b60`) calls the
  classifier `eboot+0x3b5ea0`: `classify(table=[context+8], sel, key) = (key < table.limit16[sel]) ?
  table.subarray[sel][key] : 0x7fff`, where the table has 16-bit `limit[sel]` at `+0x2e` and
  `subarray*[sel]` at `+0x08`. With `table==NULL` it reads `[0x30]` → SIGSEGV at addr `0x30`.

**Disproved proposed fix:** implement `+kSrjIVxKFE` as an AGC register-context constructor: allocate the two
register banks, install `[context+0x10]`/`[context+0x18]`, and install a valid classify table at
`[context+0x08]` mapping (register-set selector, SDK register index) → hardware slot. The register
offsets for that mapping are exactly the independently verified tables now stored in
`agc_reg_defaults.cpp`. Note: this AGC code is **statically linked into eboot** — only the leaf
SDK entrypoints like `+kSrjIVxKFE` are imports (PLT/GOT), which is why implementing that one import
unblocks the whole internal register path.

(Superseded theories, for the record: this is NOT a `std::ctype`/`std::locale` facet issue — the
classifier is hit exactly once, not thousands of times — and NOT the `GetRegisterDefaults2` result;
wiring real RegisterDefaults did not move the fault, confirming the context table is installed by
`+kSrjIVxKFE`, not read from `GetRegisterDefaults2` here.)

### Obsolete context-object interpretation

The investigation incorrectly interpreted `+kSrjIVxKFE(context)` as the constructor for the register
context embedded at **device+0x48**. The
context holds an array of **0x70-byte register-set sub-objects at context+0x38** (index 0..2 = the
cx/sh/uc sets): the setter thunks (`eboot+0x3a7aa0/0x3a7b20/0x3a7b60`) and getters all compute
`sub[sel] = (context+0x38) + sel*0x70` (`eboot+0x3b0210`: `rax = rdi + sel*0x70`). Each sub-object:
- `[sub+0x00]` → an owner/state object (chain: `[[sub+0]+0x28]+0x10/0x18`) — **must be non-null**
- `[sub+0x08]` → the register classify table (see above)
- `[sub+0x10]`/`[sub+0x18]` → the two register-bank output buffers
- `[sub+0x32]` (u16), `[sub+0x68]` (flags byte)

### Historical Stage 1 workaround (removed by #641)

`+kSrjIVxKFE` was temporarily implemented in `hle_graphics.cpp` to install a zeroed classify table into each
sub-object's `[sub+0x08]`. With all per-selector limits = 0 the classifier returns `0x7fff` for every
register, so the register-set loops skip every write and never touch the null banks. **Result: the
fault moved from `eboot+0x3b5ea6` to `eboot+0x3b1562`** — a getter that derefs `[sub[0]+0x00]`
(still null) → `[null+0x28]`. At the time this appeared to confirm the theory, but #641 proved that
the handler merely changed live DCB contents and moved the symptom; it was never a valid context
initialization boundary.

### Deeper mechanism (RE'd 2026-07-04) — the "source" object supersedes the direct-table stopgap

Going past `0x3b1562` revealed the real wiring, which **supersedes** stage-1's approach of writing
`[sub+0x08]` directly. The eboot function `SetSource(sub, src)` at **eboot+0x3af400** owns these
fields:
```
[sub+0x00] = src                     ; the "source"/state object (NOT the context back-pointer)
if (src == 0) { [sub+0x08] = 0; return }   ; <-- null source => null table => the boot fault
[sub+0x08] = [src+0x08]              ; the classify table is COPIED FROM the source object
[sub+0x30] = [src+0x5a]; [sub+0x34] = [ [src+0x08] + 0x28 ]; [sub+0x38/0x3c] from [src+0x28]+0x14/0x16
```
So `[sub+0x08]` (the table) is not ours to set — it is pulled from a **source object** `src`, and
`src` is currently **null** because the AGC call that creates it is stubbed. A sibling flush
(eboot+0x3af440) uses `peek(table,idx) = [[table+0x00]+idx*2]` (eboot+0x3b5e90) and treats a
non-`0xffff` result as "sub exhausted" → resets the sub-object. Some sub-objects DO get a valid
(host-allocated) source in other calls, so at least one create-source HLE returns non-null; the
faulting `sub[0]` gets null.

**Root of the chain (RE'd 2026-07-04):** the `src` passed to `SetSource` is a single **global**,
`[eboot+0x2048c60]`, read (never written) at eboot+0x149a54a and eboot+0x14ddb22 and handed to the
sub-object setup (`eboot+0x3a72c0`). That global is **null**, and it is:
- NOT set by any relocation — the eboot's highest reloc offset is `0x1f4e160`, below `0x2048c60`
  (verified with a Module-parser probe over all 51,475 relocs); it lives in zero-init `.bss`.
- NOT written by any eboot code (objdump over the whole image: only the two reads reference it).
- NOT an exported symbol, and its address is never taken (`lea`) — so nothing external is handed a
  pointer to it either.

Conclusion: `[0x2048c60]` is **libSceAgc.prx's private global**, which the real libSceAgc populates
with its register-source object during its own init. Because prosper HLE-stubs libSceAgc instead of
loading a real `.prx`, that init never runs and the global stays null. This is the true root of the
`0x3b5ea6` boot blocker.

**STAGE 2 (real fix), two options:**
1. *Preferred, needs data we don't have yet:* the AGC SDK headers / a real libSceAgc.prx, to know the
   exact register-source object layout and the init entrypoint. Then implement that init in our
   libSceAgc HLE to build+install the object.
2. *Reconstruct it ourselves:* build a source object whose `[src+0x08]` is a populated register-map
   table (peek array @ `+0x00`, subarray ptrs @ `+0x08`, u16 limits @ `+0x2e`) from
   `agc_reg_defaults.cpp`, plus the `[src+0x28]`/`[src+0x5a]` register-count fields, and install its
   pointer into the guest global `0x2048c60` from an early AGC-init HLE. Feasible (we know the guest
   base, so the VA is writable) but the object layout must be fully RE'd first, and the hardcoded
   global VA is title-specific — acceptable as a stepping stone but flagged as such.

The current stage-1 direct-`[sub+0x08]` write is a stopgap that `SetSource` later overwrites; it
stands only as a documented WIP checkpoint.

### Diagnostic result (2026-07-04): the graphics init is a chain of libSceAgc objects → SDK-gated

A throwaway probe installed an empty-but-structurally-valid source object into the guest global and
re-ran the boot. The fault moved only slightly — `0x3b1562 → 0x3b1533` — to the **same class** of
null-deref (`[sub[esi]+0]` → `[+0x28]`) on the next sub-object/path. Conclusion: the AGC graphics
init is a *chain* of libSceAgc-internal object installations; each empty scaffold reveals the next of
the same kind. Building this chain correctly requires the real libSceAgc object layouts, i.e. **the
AGC SDK headers** (or a real libSceAgc.prx). Continuing to hand-fabricate the object graph would
violate correctness-first (endless chain of fakes that never reaches real rendering), so stage 2 is
**parked pending the SDK headers**. Independent, provably-correct graphics-pipeline work (AGC command
decode, shader recompiler, Vulkan backend — all unit-testable in isolation) proceeds meanwhile.

## Ruled out

Cross-title falsifications for the **present / publish path** (what reaches the screen once passes
have rendered). One line each: the dead hypothesis, the evidence, the link. Extend this rather than
re-deriving — and read it before forming a hypothesis about a frozen, black, or missing frame.

- **A frozen frame with a live guest is not necessarily a guest or a draw problem — check the publish
  gate first.** Sonic Frontiers (PPSA03831) looked dead from t≈140 s with `frame_seq` frozen at 2,081,
  while the guest went on to flip 283 more times and prosper accepted 13,028 further submits and folded
  8,223 more draws. The renderer was selecting a present source by target *identity* and the publisher
  accepts only `w*h*4` bytes, so correctly rendered passes at the wrong extent (1920x1080, 1024x1024,
  3840x3072 against a requested 3840x2160) were silently dropped. Nothing logged it. #1986 / #1990.
- **Removing the publish wall does not restore blacked-out content — the two are separate causes.**
  With the extent contract in place Frontiers publishes again and the composited frame is *still*
  uniformly black (`distinct_rgb_colors == 1`, `nonblack_rgb_pixels == 0`) for the remaining 140 s of a
  300 s arm. This kills #1968 §6 (that the content going black shortly *before* the wall shared the
  wall's cause). The open question is unchanged and is #1968 §5: why no post-intro pass targets the
  flipped VideoOut buffer. #1990.
- **A climbing publish counter does not distinguish "rendering fresh frames" from "re-serving one
  retained frame" — instrument the two branches, do not infer.** On the same Frontiers arm, `frame_seq`
  reaching 5,499 was compatible with either, and the two send the next investigation to opposite places.
  The renderer's `fresh=`/`retained=` totals (carried on every `[rtt] PRESENT SOURCE EXTENT MISMATCH`
  line) settle it: between shortfall #2048 and #4096 **fresh grew by 1 (141 → 142) while retained grew
  by 2,048 (1,109 → 3,157)**. So post-wall Frontiers publishes **one retained black frame, re-served
  thousands of times**, and the last *fresh* 4K composite prosper produced was itself already black.
  That is the surface to investigate: not "why is the served frame black" but "why did the 4K composite
  go black, and then stop being produced at all". #1990.
- **`pixel_crc32` for a black frame is not a fingerprint.** `666f7b3f` is just "black 3840x2160" and
  recurs on three unrelated titles, so it identifies a resolution, not a title or a defect. Do not use
  a black-frame hash as an oracle or to claim two titles share a cause. #1990.
- **"No pass targets the flipped VideoOut buffer" is not necessarily a defect in pass selection — some
  titles never draw into their scanout at all, and the frame is in the buffer anyway.** This kills
  #1968 §5's framing (that a post-intro change makes Frontiers stop targeting the scanout). A
  `PROSPER_PASS_LOG` census over a full Sonic Frontiers boot records **`vo=1` on 0 of ~3,700 passes**,
  in every phase — including the intro, which prosper composites *correctly*. `PROSPER_DUMP_PERSISTENT`
  agrees from the other side: `scanout=MISS` on every submit, no `g_rtt` entry at the flipped address
  (`0x200a160000` / `0x200c140000`), and that address is not the base of a single pass. What prosper
  publishes during the intro is an internal RTT (`0x20851c0000`) that happens to be CPU-materialized
  and happens to hold the movie; when that stops, selection has nothing left, which is the whole of the
  "black composite". So the question to ask of a scanout that no pass writes is not "which pass was
  mis-selected" but **"what is in the guest's flipped buffer"** — for this title, the finished frame.
  #1968.
- **A registered scanout's bytes are swizzled, and every raw-scanout read took them literally.**
  `videoout_copy_*` memcpy'd guest display memory while `VideoOutBufferSnapshot::tiling_mode` had
  recorded the layout all along, so a TILE-mode title's `raw_scanout` present was horizontal-band
  noise. Two things not to re-derive: (1) a `PROSPER_DUMP_SCANOUT` dump that looks like bands is a
  **real, tiled** frame, not an empty one — Frontiers' bands de-swizzle under SW_64KB_R_X into an exact
  SEGA logo and an exact intro shot, while `distinct_rgb_colors` in the low thousands reads as
  "content" for something that is not an image; (2) a `width*height*4` read of a tiled surface is
  **short** — the tail past the nominal end holds real texels of the last block row, so truncating it
  drops part of the bottom-right (measured: the bottom-right 1024x48 of a 3840x2160 scanout loses half
  its non-black pixels). #1968.
- **"Not all its bytes are zero" is NOT evidence that the guest wrote a buffer, and a present path must
  never treat it as such.** It holds only for a freshly zeroed allocation. A title that re-registers a
  scanout over **reused** memory passes it with whatever the previous owner left there, while prosper's
  render-target map misses precisely *because* the address is new — so the test that was meant to
  protect a title publishes stale garbage in place of the retained good frame, which is #1990's failure
  class through a new door. This cost #2026 a revert (#2044). The workable test is differential:
  fingerprint the buffer when the guest registers it and require the contents to have *changed* since
  (`videoout_read_front_linear`). Note the corollary — authorship is not brightness: a frame the guest
  deliberately clears to opaque black is authored, and publishing black is then correct. #1968 / #2044.
- **A page-mapped probe is not proof that a buffer owns the bytes past its nominal end.**
  `gpu::guest_readable` answers "are these pages mapped", so an adjacent unrelated mapping passes it,
  and two separately allocated scanouts that merely land far apart satisfy a registration-stride test
  as well. At 3840x2160 that combination licenses a ~240 KiB silent over-read into another object,
  surfacing as garbage texels in the bottom-right block row. Use
  `host::guest_readable_mapping_containing` — one registered mapping spanning the whole range — *in
  addition to* the stride, because each covers the other's blind spot. #1968 / #2044.
- **Sonic Frontiers' post-intro black frame is not a present-path defect — the guest's own display
  buffer is black, and prosper is now showing exactly that.** With the flipped buffer published, a
  720 s default launch reports `[rtt] GUEST SCANOUT … publish` with ordinals reaching **2,048** (one
  fresh read per guest flip) against **9** `PRESENT SOURCE EXTENT MISMATCH` lines, all of them during
  boot with `fresh=0 retained=0`. That distinction is the whole point and it is easy to get wrong:
  `rgb_nonblack=0, distinct_rgb=1` **cannot** tell "publishing black" from "declining and freezing on
  the last logo", because it ignores alpha and both read identically — the *counters* settle it. Every
  post-140 s frame is a fresh read of the guest buffer, so the buffer really is black; the SEGA logo,
  Cyber Space intro, Sonic Team logo and middleware credits from the same run are exact 4K frames. Do
  not look for a lost render target, a mis-selected pass or a dropped publish for this symptom: the
  remaining blocker is upstream guest progression. #1968 / #2023.

## Recommended implementation order

1. **Real unified memory.** Make GPU allocations CPU/GPU-VA *aliased*: when the guest maps direct
   memory, back it so the GPU VA it later uses equals a valid CPU address (single physical page seen
   at both). This replaces the lazy zeroed-page placeholder with real, coherent memory — the
   prerequisite for everything else. Trace the libSceAgc memory-map calls to learn the GPU-VA scheme.
2. **libSceVideoOut swapchain.** Real window/offscreen images + a Vulkan swapchain (or headless
   render target); `SubmitFlip` presents the current buffer.
3. **libSceAgc → Vulkan.** Decode the AGC command buffers (draw/dispatch/state) and translate to
   Vulkan command buffers. This is the largest piece.
4. **RDNA2 shader recompiler.** Translate the game's GCN/RDNA2 shader ISA (or the AGC shader blobs)
   to SPIR-V. The other large piece.
5. **libSceAmpr audio.** Mix/output via a host audio backend (or a null sink first).

## Testing

Everything above must stay behind programmatic checks (agentic-first). Current suite (7 tests):
module parse, NID hashing, trap, boot (asserts the GC stop-the-world runs), setjmp, HLE registration,
SceKernelStat layout. Add: unified-memory aliasing test, AGC command-decode unit tests, a boot check
that asserts the first `SubmitFlip`.
