# PPSA02664 black gameplay world — target validation & investigation plan

**Status:** DRAFT for review (multiple agents). Tracks #755; investigates #320. No writer instrumentation
is proposed as ready; the watchpoint material is a **conditional toolbox appendix** (§5), not an
evidence-selected Phase 1. Revised after review by Mira Voss (2026-07-15), whose Phase-0A discipline
falsified the original premise (below) before any tooling was built.

> **Current status (2026-07-15) — this is a frozen METHOD/DIAGNOSTIC PR.** Phase −1 (target identity)
> was subsequently completed: the actual Mt. Eternal gameplay scene is content-verified, and the world
> IS rendered but is lost during final composition in offline replay. The **evolving diagnosis and all
> results now live on issue #320**, not in this document (a durable doc must not become a moving
> hypothesis log). This PR contains only two diagnostics — `PROSPER_FS_SPV_MATCH` (fail-closed
> content-gated fragment override) and `PROSPER_DUMP_RTGROUPS_RGBA` (alpha-preserving per-target dump).
> **Neither constitutes a gameplay root-cause claim or a fix**; the current root cause is unresolved and
> gated on a live/replay state comparison (see #320).

## Governing invariant (adopt this first)

**Scene identity must be established from rendered *semantic content* before any shader/resource
evidence is attributed to the user-visible failure.** Submit ordinals, guest VAs, and draw-count /
shader-hash "signatures" are **run-local selectors**, not scene identity. This doc exists partly because
that invariant was violated once already (§1).

---

## 1. Phase 0A result — a *narration* false-positive (the original premise is FALSIFIED)

An earlier localization claimed the black gameplay world was caused by the world material shader
`fs2949` premultiplying its output by a vertex-color alpha of 0. A per-draw alpha-only A/B **disproved
that capsule as evidence for the gameplay world**:

**Experiment.** Retain the real `fs2949` + textures + constants + blend + geometry; substitute **only**
the vertex-alpha multiplier (`fs2949` SPIR-V `%508 = vcolor.w` → constant `1.0`), applied **only** to
draws whose recompiled fragment SPIR-V exactly matches `fs2949` (an exact-content match gate, so the
composite/base draws are untouched).

**Result.**
- Baseline (real `fs2949`, alpha 0): the `ce610000` float-target group renders `px_nonzero=0` → black.
- Alpha→1: the group renders `px_nonzero=842526 rgb_nonblack=336636` and composites through — but what
  appears is **overlapping intro narration text** ("A young…", "Alex was not aware, but at the time the
  Radaxian region was under threat from Janken the Great and his minions", "A long time ago, on the
  distant planet Aries…"), **not** the Mt. Eternal world. The float target is text-only (no
  sky/platforms/sprites).

**Conclusion.** The captured capsule (submit 10000 of the run-local 78-draw signature) is the **intro
narration crawl**. There, `fs2949` premultiplying by vertex alpha with alpha-0 is **correct**: each
narration page fades in/out, so a page not currently shown legitimately has alpha 0. Forcing them all
opaque overlaps every page. This **falsifies** using that capsule as evidence for the black gameplay
world — it is not merely a weaker claim; there is now **no positive evidence** in this investigation
that the gameplay-world failure involves vertex alpha, this draw group, or consumed-resource writer
tracing at all.

**Proof boundary.** The A/B verifies the *current emitted-SPIR-V* path and identifies the scene as
narration. It does **not** establish raw-RDNA2 ↔ SPIR-V translation fidelity — but that question is
moot for rejecting this capsule as gameplay-world evidence.

**This capsule is now a negative / control case:** any eventual fix must **preserve narration fading**
and must **not** make inactive narration pages overlap. Do **not** treat "this group renders
`px_nonzero>0`" as a success gate — that would encode the incorrect overlapping-narration override.

### Reproducibility of the A/B
- Emulator revision: the diagnostic commit on branch `fix/issue-320-render-oom` (the
  `PROSPER_FS_SPV_MATCH` commit), based on master `f71a847`. The A/B was run from that committed
  implementation (not uncommitted local state).
- Title/build identity: `PPSA02664`. *(Exact game executable/build revision not separately recorded —
  TODO if a future run needs byte-level title identity.)*
- Capsule: `levelscene.prgcap`, size 14021048 bytes, content sha1[:16] `09ce839ff344d327`
  (retained under the working scratch dir; not committed — game-derived). Native-speed `boot_trace`
  timeline capture at submit floor 10000, MIN/MAX draws 77/79 → **12 realized draws**.
- Selected/substituted draws: draw indices **2, 3, 4, 5** (the `fs2949`/`vs10004` group targeting the
  `ce610000` B10G11R11 float target); fragment `fs=2949` hash `9166015cb3155d4f`, vertex `vs=10004`
  hash `7130ae8c793a9804`. The gate logs `[fs-match] file override applied to draw#N` per substitution,
  so the count (4) is derivable from the run log.
- Output hashes (full-frame RGB, **sha1[:8]**): baseline `117df2e3`; alpha→1 `2a97a2a2`; alpha→1
  `ce610000` group `3903ce98`. Fail-closed control (bad match path) reproduces the baseline `117df2e3`
  (proves the gate applies **no** override on an invalid match, never a global fallback).
- Retained artifacts (scratch, not committed — game imagery): `ab_a1_final.png` (overlapping narration
  text), `ab_a1_ce610000.png` (float-target group, pre-composite).
- Gate: `PROSPER_FS_SPV=<fs2949_alpha1.spv> PROSPER_FS_SPV_MATCH=<fs2949.spv>` — exact SPIR-V content
  match; only the `fs2949` draws are substituted (composite `fs557` / base `fs496` untouched). The gate
  is committed in this branch (see §5); it **fails closed** if the match file is missing/short/
  unaligned/unreadable (no silent global fallback).

---

## 2. What is known / not known now

- **Known:** in the *current translated* `fs2949` path, the consumed vertex-alpha multiplier controls
  visibility for the narration group; the narration scene is identified; alpha 0 there is legitimate.
- **Not known (re-opened):** the **black gameplay world** ("Well, that's fine for today…" Mt. Eternal,
  observed only in a slow *live* run) is **uncharacterized**. It has not been captured or analyzed. Its
  cause is unattributed — it could be a required world draw with zero alpha, the empty-RTT-producer
  class, geometry/descriptor/dataflow, composition, or something else.
- The native 200 s timeline (submit ≤24352) reached only the narration; the world scene is at a later
  guest-time point (live reached it ~360 s of wall time).

---

## 3. Phase −1 — Target identity (the new leading gate; do this before anything else)

Capture and *prove* the actual gameplay-world scene before attributing any evidence:

1. Capture the frame that shows `"Well, that's fine for today…"` **and** the black gameplay world.
2. Retain: a screenshot + output hash; exact title revision + executable/build identity; route state;
   guest time; submit/capsule identity; and capture bounds.
3. **Prove the capture is the gameplay world from its rendered composition** (expected world/UI
   content), not from a familiar shader hash or draw-count signature.
4. Identify which target/group is *expected* to contain the world geometry, and **demonstrate the
   missing contribution there** (which draw(s), which target, what the composite samples).
5. Only then inspect that failing draw's geometry, material inputs, producer graph, and resource
   lifetime.

---

## 4. Investigation branches (selected by Phase −1 evidence, not pre-committed)

After the correct scene is captured, branch from *its* evidence:
- **Only if** that scene independently shows a zero-alpha input on a *required* world draw → an
  alpha-only A/B (as in §1) and then producer/lifetime classification.
- Otherwise the leading branch may be **empty-RTT producer provenance** (`gpu_replay --graph` /
  timeline resource versions — phrased as "no producer within retained provenance scope," never "GPU
  ruled out" from a selected-submit graph), **geometry**, **descriptor/dataflow**, or **composition**.

Producer/lifetime tracing and any write-watch tooling are **downstream** of a defect being localized on
the correct scene — they are not started against the narration range (its alpha is legitimate).

---

## 5. Conditional toolbox appendix (NOT an evidence-selected Phase 1)

Two small pieces exist/were designed; both are conditional on a future defect actually calling for them:

- **`PROSPER_FS_SPV_MATCH`** (committed in this branch): restrict `PROSPER_FS_SPV` to draws whose
  recompiled fragment SPIR-V exactly matches a given file — a per-draw shader A/B substitution that
  avoids hitting draws with different descriptor contracts. General-purpose; used for §1.
- **Consumed-resource write-tracing design** (deferred): if a *future* correct-scene defect turns out to
  be a consumed resource with a bad value, catching its writer needs care because Unity's dynamic VB
  address is run-local and the existing watchpoint machinery (`exec_image_linux.cpp`) is
  calling-thread-owned, single-global-fd, `LEN_8`, per-thread-DR-limited. Options: per-thread perf-watch
  fd table (MB3 pattern) *or* page-protection write-log, chosen from measured lifetime/thread behavior;
  capture-derived selector, not runtime semantic recognition. This is retained as a generic method, not
  a committed plan for #320.

---

## 6. Reproduction & gotchas
Native-speed `boot_trace` for captures (recipe in #755). `PROSPER_RENDER_SCALE` and
`PROSPER_PUMP_USEREV` destabilize this title's early boot; `PROSPER_NO_BLEND` breaks the live composite;
never `pkill` in this env; addresses/submit ordinals are run-local (see §Governing invariant).
