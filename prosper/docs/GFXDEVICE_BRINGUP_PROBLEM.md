# The GfxDevice bring-up wall — problem brief (for a fresh reviewer)

**Status:** open, blocking. **Audience:** an agent who has *not* lived this investigation and needs
full context. **Ask:** a fresh framework / a second pair of eyes on why Unity's GfxDevice object
graph comes up systematically null in our environment. Three mechanism-hypotheses have missed; we
think the way forward is *ground-truth observation*, not a fourth guess (see §8).

---

## 1. One-paragraph summary

We run the retail PS5 build of **The Messenger** (Unity 2022.3.32f1, IL2CPP) natively on Linux via
HLE (no CPU emulation). The boot gets all the way **through** IL2CPP runtime init and into Unity's
**GfxDevice** graphics bring-up, then dies during load of *"unity default resources"* — **before it
ever issues a real draw**. The proximate cause is that Unity's GfxDevice **CPU-side object graph is
systematically null**: pipeline/resource objects are allocated but their sub-object slots (GPU
companions, etc.) are never populated, and the engine eventually calls a virtual method on a null
object and crashes. Our whole AGC→Vulkan rendering pipeline is *built and proven offscreen* — the
blocker is upstream, in Unity's own CPU-side device/resource construction, not in our renderer.

---

## 2. Project context (minimum you need)

- **prosper** = a PS5→PC user-space compatibility layer ("Proton for PS5"). The PS5 is x86-64, so the
  guest game code runs **natively**; we translate the OS/ABI (FreeBSD-ish + Sony `libkernel`), the
  library ABI (Sony NID-linked modules, HLE'd), and the GPU (AGC → Vulkan).
- **Target:** `PPSA24651` = The Messenger, a 2D Unity/IL2CPP game (a deliberately small/favorable
  first title). Segments are unencrypted, which is what makes this possible.
- **Boot pipeline that already works (all `ctest`-verified):** SELF/ELF loader → multi-module link →
  NID import binding → host mmap + import-trap HLE dispatch → libkernel/libc/services → **IL2CPP init
  completes** (GC + 14-thread job pool, stop-the-world suspension solved via real async exception
  delivery) → guest `main()` runs → **Unity GfxDevice bring-up**, where we are stuck.
- **Graphics stack we've BUILT and verified (offscreen, on Vulkan/llvmpipe):**
  - `pm4_decode` → `command_processor` (folds a real submitted `Dcb` into a `GpuState`: register files
    + draw list) → `render_state` / `resolve_pipeline_state` → `vk_translate` → a real
    `VkGraphicsPipeline` (topology/blend/depth/write-mask all pixel-verified).
  - **RDNA2 → SPIR-V recompiler**: ~52 ALU ops + convert/compare/select/bitfield/pack + **divergent
    control flow** (EXEC per-lane predication, `saveexec`/restore, safe forward `s_cbranch_execz`).
    ~65% instruction coverage over the game's 41 real embedded shaders (`recompile_coverage`).
  - The full `GpuState → recompiled shaders → VkPipeline → frame` spine renders real pixels in tests.
  - **AGC command frontend is complete**: `sceAgcCreateShader` (relocates the embedded RDNA2 shader
    ELFs, registers all 36 shaders) + `sceAgcDriverSubmitDcb`. **Zero unimplemented `libSceAgc` calls
    remain in the boot.**

  So: the renderer is ready. It is **not** the thing that's broken.

---

## 3. The problem, precisely

During GfxDevice bring-up (loading *unity default resources*, **pre-render-loop**), the boot faults.
Investigation shows the fault is one symptom of a **systematically-null GfxDevice object graph**:
discrete GPU/pipeline sub-objects that Unity expects to have been constructed are all null.

### 3a. The original fault
`eboot+0xba6e08` — a Unity GfxDevice routine (`eboot+0xba6720`) iterates a small collection of
pipeline/resource objects. For an object whose type is in mask `bt 0xc8220` **and** whose flag
`[obj+0x1a0]` (low byte) `== 0` ("not resident/processed"), it reads a GPU-backing companion pointer
`[obj+0x140]`, then dereferences `[that+0x08]` → **SIGSEGV at address 0x8** because `[obj+0x140]` is
null.

### 3b. It is a chain, not a single gap (skip-probe result)
We built two env-gated diagnostics (default off; they are *probes*, never fixes):
- `PROSPER_SKIP_NULL_COMPANION`: at `0xba6e08`, log the object and redirect RIP to the reader's own
  skip label so it continues as if no companion were needed. → **4 bounded skips** (4 pipelines, all
  `[+0x140]=0`, all resident-flag low byte 0), then a **new** null fault at `eboot+0x95c823`.
- `PROSPER_NULL_PAGE`: back low-address **read** faults with a shared read-only zero page (write- and
  call-through-null are left unbacked — those are the real endpoints). → **one page (page 0) backs
  *every* null field read**, the reader proceeds on zeros, and execution terminates at a
  **call-through-null**: a virtual method invoked on a null object (`rdi=0`). Backtrace:
  `eboot 0x1485851 → 0x14795a2 → 0xaee233 → 0xb06a2d → 0xb05cd9 → [call *method]`. **Never reaches a
  frame.**

Interpretation: this is **not** a narrow validation gap you can patch at one site. A whole subsystem
that should populate the GfxDevice object graph did not run, so nulls are everywhere and skipping one
only reveals the next.

### 3c. The game has not drawn yet
The only GPU submission before the fault is a **0-draw state submit** (register/context setup, no
draw). So we are stuck in **CPU-side engine construction**, before any real rendering work exists.

---

## 4. What we know for certain (survived every hypothesis)

1. Four pipeline objects are **unprocessed**: `[+0x140]=0` (null companion) and `[+0x1a0]` low byte
   `=0` (not resident/processed) on all of them.
2. The companion `[+0x140]` is built by a routine **gated by a predicate at `eboot+0xd58710`** that
   scans a per-pipeline table at `[pipeline+0xc0]` / `[+0xe0]` and returns 0 for our pipelines → the
   companion is never created.
3. The fault is a **lazy-init during asset load, pre-render-loop**.
4. **Device init *succeeds*.** Unity's AGC device-init is a success/error chain; our stubs return
   `0`=success, so Unity believes the device initialized fine. (See §5, hypothesis 3.)
5. **Zero unimplemented `libSceAgc` calls** fire in the boot. If a resource/pipeline object is built
   by an AGC call, it's a call we *do* implement — as a no-op / return-0 — not a missing one.
6. The GfxDevice sub-object graph is **systematically null** (one zero page satisfies every null
   field read), terminating in a **virtual call on a null object**.

---

## 5. Investigation log — three hypotheses that MISSED (important: don't re-run these)

This is the most valuable context: three plausible mechanism-hypotheses were each disproved with
evidence. The pattern — reasoning *about* Unity's internals from the fault site — has not worked.

### Hypothesis 1 — "hook the AGC resource-creation call / find the `[+0x140]` writer" ❌
- Traced a `[+0x140]` writer at `eboot+0x15aef7d` (`[r13+0x140] = g->vtable[0x88](g, 0, payload)`),
  gated by the `0xd58710` reflection-worklist predicate.
- **Disproved:** the faulting object `r15` is **not** the `0x15ae` pipeline that was traced — runtime
  peek showed `r15+0xe0` is a pointer (a shared sub-object), i.e. a *sibling type*. The real writer
  for `r15` was never found, and per-object writer-hunting was abandoned (skipping one null only
  reveals the next — see §3b).
- Also established: **no** game→AGC `createResource`-shaped call `(gpu_addr,size,w,h,format)` fires
  anywhere in the trace. The pipeline object + its companion are constructed by Unity's *own*
  resource manager, not a hookable Sony call.

### Hypothesis 2 — "the residency pass is GPU-completion-event-driven; our headless equeue never fires it" ❌
- There is an event-drain loop at `eboot+0x14dfb04` (`while(WaitEqueue()) GetEqContextId()…`), and our
  equeue is a headless stub, so the theory was: completions never arrive → the completion-driven
  residency pass never runs.
- **Disproved:** `0x14dfb04` is a **completion-timing profiler** (frametime math `×0xf4240`,
  reciprocal-divide → a ring buffer), running **off-thread**, **not on the fault path**. Delivering a
  completion event would not build the companions. Wrong trigger.

### Hypothesis 3 — "a wrong device-init capability/format query steers Unity onto a degenerate/null device" ❌
- Unity has device-init failure branches; a stubbed "unsupported"/0 could make it build a zeroed
  placeholder device (which *would* explain a systematically-null graph).
- **Disproved by audit:** device init is a success/error chain; our stubs return `0`=success, so Unity
  thinks init succeeded. `GetDeviceCapabilityInfo` is **never called**; the only stubbed device query
  is HDR tone-map luminance (benign); `IsOutputSupported → 1`. There is **no wrong device query**
  steering a degenerate path. The device *succeeds* — yet its sub-objects were never built. Model
  doesn't fit.
- Reference cross-check is thin: **Kyty** only implements a shallow `GraphicsInit`; not much to diff.

### Also ruled out
- **Display surface** is not the gate: we implemented real `libSceVideoOut` (1920×1080@59.94, 3
  buffers, real flip/vblank status) and the boot faults **identically**.
- **Shader semantics are not mangled:** raw-byte trace confirmed our `CreateShader` relocation is
  correct (a shader with varyings reads `num_out=1`, `output_semantics[0].semantic=0x0f`, exactly what
  Kyty's VS→PS linkage expects). The faulting pair are genuinely **zero-varying** shaders
  (position-only / blit / clear) — so the empty reflection worklist is a *correct* consequence for
  them, not a surfacing bug. (One PS does have a single `sharp[3]` storage-buffer binding.)

---

## 6. Object-layout & address reference (raw observations)

**The fault object `r15`** (a game-heap allocation; layout partially understood, interpretations have
shifted across probes — treat offsets as observations, not confirmed semantics):
- `[+0]=0x2b`, `[+8]=0xf`, `[+0x20]=7`, `[+0x28]=0x19`
- `[+0x18] == [+0x40]` → a shared sub-object holding packed register-like fields
  (`[+0x10]=0x28a7…000e`, `[+0x18]=0xbba2…002b`)
- `[+0xc0]` / `[+0xe0]` → the table scanned by predicate `0xd58710`. Early reads suggested
  `[+0xc0]`=array-ptr / `[+0xe0]`=count; the skip probe later showed small integers here
  (`[+0xc0]=0x7`, `[+0xe0]`=a pointer, `[+0x138]=0x7`). **Layout here is not firmly pinned.**
- `[+0x140]` = **null GPU companion** (the thing dereferenced as `[+8]` → the crash)
- `[+0x1a0]` = resident/processed flag; **low byte** is what the reader tests (0 = not resident).
  (High bytes were non-zero on 2 of 4 objects — `0x9b838200`, `0xffffff00` — but the low byte is 0.)
- `[+0x520]` / `[+0x530]` = an array ptr / count, both 0
- Part of a small collection (outer object: array@`+0x78`, count@`+0x88` = 3; skip probe hit 4).

**Key code addresses (all `eboot+`):**
| Address | What it is |
|---|---|
| `0xba6720` | GfxDevice routine iterating the pipeline/resource collection |
| `0xba6e08` | the reader that derefs null `[obj+0x140]` → original SIGSEGV (addr 0x8) |
| `0xd58710` | reflection-worklist predicate gating companion creation (returns 0 for our objects) |
| `0x15aef7d`/`0x15aef83` | a `[+0x140]` companion writer (`g->vtable[0x88]`) — but for a *sibling* object type |
| `0x14dfb04` | completion-timing profiler loop (red herring — hypothesis 2) |
| `0x95c823` | next null fault after skipping the 4 companions |
| `0x1485851 → 0x14795a2 → 0xaee233 → 0xb06a2d → 0xb05cd9` | backtrace to the terminal call-through-null (`rdi=0`) under the null-page probe |

---

## 7. Diagnostic tooling available (Linux host; all env-gated, default off)

- `PROSPER_PEEK="rN:off,off;rM:*pre+off"` — dump arbitrary offsets off registers at fault time,
  including a one-level pointer chase (`[[reg+pre]+off]`). For classifying object graphs.
- `PROSPER_PIPETRACE` — logs raw pointer args + shader semantic/user-data of the pipeline/shader
  construction calls (`CreateShader`, `CreatePrimState`, `CreateInterpolantMapping`).
- `PROSPER_SKIP_NULL_COMPANION` — at `0xba6e08`, skip the null-companion deref (RIP redirect) and log.
- `PROSPER_NULL_PAGE` — back low-address read faults with a shared zero page; leaves write/call-through
  -null unbacked (the real endpoints). Reveals the whole null-read chain.
- `PROSPER_FAULTMEM` / `PROSPER_FAULTLOG` — fault-time register/memory dump + logging.
- `tools/boot_trace` — links all modules, boots headless, prints the unimplemented-call trace +
  register state + module-classified rbp backtrace on fault.
- `shader_histo` / `recompile_coverage` — RDNA2 recompiler coverage over the real shaders.
- Source of most of the above: `src/host/exec_image_linux.cpp` (fault handler + probes),
  `src/hle/hle_agc.cpp` + `hle_graphics.cpp` (AGC/graphics HLE). See also `docs/GRAPHICS.md` (the
  full front-half RE log) and `docs/ROADMAP.md` §"Current status".

---

## 8. Current plan — stop guessing the mechanism, observe it

Three misses all came from *inferring* Unity's internals. The proposed next step is **ground truth**:

> **Watchpoint the null slot.** Pick the *earliest* null sub-object slot in the chain (not the deep
> `0x95c823` one). Put a **write-watchpoint** on its address (hardware debug register `DR0–3`, or
> `mprotect` the page + catch the write — same fault machinery as the probes) and replay the boot.
> Report: (a) does *anything* ever write that slot? (b) if yes — from what PC/function, storing what,
> and what were the last few HLE calls before it? (c) if nothing ever writes it — what is the last HLE
> call before the slot is *read* as null?

That single observation distinguishes the remaining possibilities **with data instead of a guess**:
- a write happens but stores null / early-returns → that's the builder; see *why* it produced null;
- nothing ever writes it → the builder never ran → correlate the expected-write moment against the
  HLE call trace to find what *should* have triggered it (tests "an AGC call we no-op should populate
  it");
- a write comes from a completion/readback path → then (and only then) does "needs real GPU execution"
  have evidence.

### A hypothesis we explicitly do NOT endorse yet
"These sub-objects are built by **GPU-execution side-effects** we don't perform, so this needs real
command-buffer/compute execution (M4/M5)." We're skeptical: GPU execution writes GPU *memory*, not
Unity's **CPU-side** object slots (those are `new`'d by CPU code); and the only submit so far has
**0 draws**, so there's nothing to execute. Implementing real GPU execution is a large effort and,
against this specific fault, would be a fourth unvalidated guess. The watchpoint would provide
evidence before committing to it.

---

## 9. Specific questions for the fresh reviewer

1. **Is the watchpoint the right ground-truth probe, or is there a better one?** (e.g. is there a way
   to identify the *constructor* of these GfxDevice sub-objects from the object layout / vtables
   rather than the fault site?)
2. **What CPU-side mechanism builds a batch of GfxDevice sub-objects in Unity's PS5 backend**, and what
   would make it silently produce nulls in our environment (device *reports* success)? Fresh ideas on
   the class of divergence welcome — we may be over-fixated on "one wrong answer."
3. **Is there a lower-level divergence we've overlooked** — memory allocation (a GPU/direct-memory
   alloc that "succeeds" but returns an address Unity later treats as unusable), an alignment/size
   contract, a returned handle we zero, a `this`-pointer we never set?
4. **Strategic:** given three misses on a Wine/Proton-class CPU-side bring-up wall, is it worth
   continuing to drill *this* path, or should we bank the (proven) renderer and demonstrate graphics by
   driving the game's *real recompiled shaders* + a real/constructed command buffer through our
   verified `GpuState → frame` spine, decoupled from the Unity-init morass? (This is a human decision;
   input welcome.)

**Nothing is lost either way** — the renderer is staged and verified. The moment this wall falls and
real draws flow through `run_command_buffer`, the downstream pipeline is ready to render them.
