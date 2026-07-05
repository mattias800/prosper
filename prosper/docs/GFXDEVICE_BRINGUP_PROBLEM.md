# The GfxDevice bring-up wall — problem brief (for a fresh reviewer)

**Status:** open, blocking. **Audience:** an agent who has *not* lived this investigation and needs
full context. **Ask:** a fresh framework / a second pair of eyes on why Unity's GfxDevice object
graph comes up systematically null in our environment. Four mechanism-hypotheses have missed; the
latest watchpoint result changes the question from "who builds the missing companion?" to "why does
the reader require a companion that this pipeline class legitimately lacks?" (see §8).

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
7. **The companion slot is not built late.** `PROSPER_WATCH_COMPANION` showed the faulting
   `[obj+0x140]` slot is written many times, but every observed write stores zero from reset/memset
   paths on both the reader thread and a worker thread. The object tag stays `[+0]=0x2b`, so this is
   the same live object being zeroed, not a non-null builder that races the reader. No non-null
   companion write was observed.

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
- **Late writer / ordering race** is not the gate: the companion watchpoint saw only zero writes, not
  a delayed non-null construction. The `[+0x140]` companion is absent by design for this object shape.

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
- `PROSPER_WATCH_COMPANION` — at the first `0xba6e08` null read, arm a page-protection
  write-watchpoint on `[obj+0x140]`, single-step writer faults, and log writer PC/thread plus the
  post-write slot value. Its first result: only zero writes from reset/memset paths.
- `PROSPER_FAULTMEM` / `PROSPER_FAULTLOG` — fault-time register/memory dump + logging.
- `tools/boot_trace` — links all modules, boots headless, prints the unimplemented-call trace +
  register state + module-classified rbp backtrace on fault.
- `shader_histo` / `recompile_coverage` — RDNA2 recompiler coverage over the real shaders.
- Source of most of the above: `src/host/exec_image_linux.cpp` (fault handler + probes),
  `src/hle/hle_agc.cpp` + `hle_graphics.cpp` (AGC/graphics HLE). See also `docs/GRAPHICS.md` (the
  full front-half RE log) and `docs/ROADMAP.md` §"Current status".

---

## 8. Current plan — the watchpoint answered; decide whether to keep chasing this wall

Four misses all came from *inferring* Unity's internals. The watchpoint supplied the requested ground
truth and refuted the last ordering theory: `[obj+0x140]` is zero-initialized/reset, never populated
with a real companion. Combined with the `0xd58710` predicate result, this means the companion is
legitimately absent for these zero-varying pipelines.

The remaining question is now narrower and harder:

> Why does the `0xba6e08` reader take the companion-required path for a pipeline whose companion is
> legitimately absent?

The plausible next probes are reader-side, not writer-side:
- Watch `[obj+0x1a0]` and the type/mask inputs to prove whether a "processed/no-companion-needed" bit
  is never set, is reset to zero, or is not the real gate.
- Disassemble/trace `0xba6720..0xba6e40` enough to prove the exact branch contract before changing
  any state. A local RIP skip is still only a diagnostic; it is not a correctness fix.
- Compare the faulting zero-varying pipeline against a non-faulting pipeline that does get a
  companion, focusing on the reflection records consumed by `0xd58710` and the reader's type remap.

Strategically, this path now has uncertain ROI. The renderer and shader path are already verified
offscreen; absent new reference evidence for the Unity PS5 reader contract, parking `0xba6e08` as a
well-characterized Unity-init wall and redirecting effort to present/swapchain, recompiler coverage,
or a real-shader demonstration frame is defensible.

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

---

## 2026-07-05 correction — fresh 3-agent RE (ground truth via eboot x86-64 disassembly)

A three-investigator pass (read-only x86 disassembly of the flat eboot image via `imgdump` +
`objdump`) **corrected two load-bearing errors** in the analysis above and reframed the fix.

**(A) `eboot+0xd58710` was MISIDENTIFIED.** It is NOT "the predicate gating companion creation" (the
basis of Hypothesis 1 and the `[+0xc0]/[+0xe0]` "reflection table" reading). It is Unity's generic
**`SafeBinaryRead::Transfer`** — asset-deserialization TypeTree field reader. Proof (ground truth):
it references the string `"SafeBinaryRead::BeginTransfer name mismatch, name='%s' oldBaseTypeName='%s'"`
at `eboot+0x1b95eb2`, and has **4,955 direct call sites**. The `[+0x140]` it feeds (via `0x15aef7d`)
is a **Mesh** GPU vertex/index buffer (`m_BakedTriangleCollisionMesh` etc.) on a *different* object
type than the crash object. ⟹ **The "empty shader semantics → empty reflection table → no companion"
theory is dead.** Surfacing shader semantics will NOT fix `0xba6e08`. (The stale `hle_agc.cpp` comment
that encoded this has been corrected.)

**(B) The reader `eboot+0xba6720` is a GfxDevice GC / DEFERRED-RELEASE DRAIN pass**, not a render or
residency pass. It walks a work-list at `[container+0x78]`/count `[+0x88]`, and per pipeline `r15`
decides keep-vs-release; the "skip" target `0xba6e40` calls the object's Release/destructor
`eboot+0xb9ca80` (refcount at `[+0x514]`) and removes it from the list. The exact fault gate:
```
ba6dd1 type_id=[rax+0x1e4c]; category=remap_table[0x1ca3b80][type_id]
ba6de1 cmp $0x13 / bt $0xc8220   ; category ∈ {5,9,15,18,19} (NOTE: 15, not 13 — the brief's 0xc8220 bit)
ba6df0 cmpb $0,0x1a0(%r15) / jne ; UNLESS the processed-flag [+0x1a0] is set → release path (safe)
ba6dfa mov 0x140(%r15),%rsi      ; else read GPU companion (NULL for us)
ba6e08 mov 0x8(%rsi),%rax        ; SIGSEGV
```
Remap table @ `0x1ca3b80`: type_id 8→cat5, 11→cat9, 17→cat15 are the only in-range companion-required
types, so **r15 ∈ {type_id 8, 11, 17}**.

**(C) `[pipeline+0x1a0]` has no setter for this class.** A full-image scan (only 17 byte-flag sites
touch `+0x1a0`) found it is zero-initialized in the constructor, only propagated by the move-ctor, and
read by the GC drain — in BOTH environments. So the engine does NOT rely on `[+0x1a0]` to keep these
pipelines safe; **safety comes from the companion `[+0x140]` existing.** On a real PS5 this drain path
runs without faulting ⟹ `[+0x140]` is normally non-null ⟹ the companion is built eagerly (writers exist
in the `eboot 0xb3xxxx` GfxDevice module).

**Conclusion / corrected fix direction.** The root cause is a **missing GPU companion** `[pipeline+0x140]`
for category-{5,9,15} GfxDevice pipelines — an **AGC/GPU-resource gap in prosper's HLE**, consistent
with the "we reimplement the absent, undocumented `libSceAgc` and must reconstruct the GPU objects it
builds" nature of this project. The correct fix is to make that companion get built (implement whichever
AGC/graphics call the `0xb3xxxx` builder depends on that we currently stub/no-op); setting `[+0x1a0]=1`
is a crash-avoidance shim only (changes GC keep/release semantics, not what the game does). **Next
concrete step:** trace the `0xb3xxxx` `[+0x140]` companion-builder — what invokes it at pipeline
creation, and which AGC/memory call it needs — to name the exact API to implement.

### 2026-07-05 (later) — companion-builder trace + k3 probe: REFUTED, and why static/watchpoint stall

- The follow-up trace found the earlier "companion writer in `0xb3xxxx`" hint was a **misattribution**
  (offset `0x140` is reused across dozens of unrelated classes — `std::vector` members, move-ctors,
  int arrays; static byte-slicing on `mov …,0x140` cannot disambiguate the crash object's writer).
- It named **`k3GhuSNmBLU`** (a return-0 thunk that "fires once just before 'unity default resources'
  load") as the top companion-builder suspect. **Tested and REFUTED** (gated probe
  `g_agc_k3_companion_probe`, `hle_graphics.cpp`): (1) returning a shape-valid companion → fault
  unchanged; (2) also writing it into `a0[+0x140]` (k3's arg) → fault unchanged. So k3 neither returns
  nor out-params the companion for the faulting pipeline `r15`.
- **Why the reliable probes stall:** the `PROSPER_WATCH_COMPANION` watchpoint arms on the *read* (the
  fault site), by which time `r15`'s construction — and its `[+0x140]` write — already happened; it only
  ever catches later memset/zero writes. And static search can't find the writer (offset reuse).

**Revised next reliable probe (not yet done):** arm a `[obj+0x140]` write-watchpoint on `r15`'s
**constructor** (backtrace `eboot+0x1478fc9 → 0x1485851`), not on the reader — i.e. identify the
allocation that becomes a category-{5,9,15} pipeline and watch its `[+0x140]` from birth. That
definitively catches the real builder (or proves it never runs) and yields the writer PC to disassemble
→ the missing input → the API. This is front-half (`exec_image_linux.cpp`) work of moderate size.

**Strategic note:** this wall has now survived 4 prior hypotheses + a fresh 4-agent pass (which still
delivered real value: corrected two wrong premises, pinned the exact mechanism, refuted the leading
suspect). ROI on continued drilling is uncertain. Higher-confidence parallel tracks: (a) the cross-engine
recompiler generality check on the new UE4 title (PPSA17942); (b) Path B — drive the game's real
recompiled shaders through the verified `GpuState → frame` spine for a demonstrable game-shader frame.

### 2026-07-05 (later still) — per-pipeline ground truth + two corrections + the reliable entry probe

`PROSPER_SKIP_NULL_COMPANION` dumped the 4 faulting pipelines' fields (skipping each → the documented
cascade to a **new** null fault at `eboot+0x95c823`, confirming this is a *systematically-null GfxDevice
subsystem*, not a single-companion gap):
```
#1 r15=…62c300 [+0xc0]=7 [+0xe0]=<ptr> [+0x138]=7 [+0x140]=0 [+0x1a0]=0x00000000
#2 r15=…add4c0 [+0xc0]=4 [+0xe0]=<ptr> [+0x138]=4 [+0x140]=0 [+0x1a0]=0xe8de52_00
#3 r15=…c5da00 [+0xc0]=7 [+0xe0]=<ptr> [+0x138]=7 [+0x140]=0 [+0x1a0]=0x……0c02_00
#4 r15=…d5e1c0 [+0xc0]=6 [+0xe0]=<ptr> [+0x138]=6 [+0x140]=0 [+0x1a0]=0xbfb989_00
```
**Correction 1 — `[+0x1a0]` HAS a runtime writer.** #2/#3/#4 show non-zero *upper* bytes; only the low
"processed" byte is 0. The earlier "no setter exists" (static-scan) conclusion was wrong — the writer
uses `lea`/SIB addressing invisible to a `disp32(base)` grep. So a real pass writes `+0x1a0` and just
never sets the low byte. **This is the single best runtime probe target.**
**Correction 2 — `[+0xe0]` is a valid heap pointer** (not a count) and `[+0xc0]==[+0x138]` is a small
count (4–7). The companion at `[+0x140]` may be built by walking `[+0xe0][0..count]`.

**THE reliable entry probe for a focused next session (two-pass; boot is deterministic):**
1. Pass 1 (done): `PROSPER_SKIP_NULL_COMPANION` → the r15 addresses above.
2. Pass 2 (to build): arm a write-watchpoint on `[r15+0x1a0]` (and `[r15+0x140]`) from BEFORE construction
   — either a hardware debug-register breakpoint (perf_event_open / self-ptrace) on the now-known address,
   or page-protect the r15 page once mapped. Log the writer PC(s). That reveals the pass that
   partial-inits `+0x1a0` (Correction 1) → disassemble it → why it stops short of the low byte / the
   companion. First verify the r15 addresses are stable across runs (the high mmap bits may vary even if
   the boot order is deterministic; if so, key the watch off the owning mapping + offset).
   **VERIFIED 2026-07-05: r15 addresses are NOT deterministic** across runs (run A `…0d62c300`, run B
   `…2ee2c300` — high mmap bits AND the relative offset differ, from the multithreaded boot). So the
   simple "hardcode r15, watch it in pass 2" approach is out. The reliable probe must HOOK THE
   ALLOCATOR/CONSTRUCTOR and watch by object *type*: catch every object the gfx-init at `0x1471fe0`
   builds (or whose ctor writes `[+0x1e4c]` = a category-{5,9,15} type_id), arm a `[+0x1a0]`/`[+0x140]`
   write-watchpoint on each from birth, log the writer PC. Multi-step in-run instrumentation — a focused
   effort, not a quick probe.

**Honest status:** this is a subsystem-level null cascade (likely a whole GfxDevice-construction pass
that doesn't run in our env), not a one-API fix. It plausibly needs the two-pass watchpoint above and/or
Unity-PS5-backend reference material. Materially advanced, but a fix is not close by autonomous drilling.
