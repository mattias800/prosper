# Where prosper's render architecture costs more than it needs to (2026-09-26)

A comparison study. prosper's render path was measured this session (see
`RENDERER_ARCHITECTURE_GAPS_2026_09_25.md`); this document records what a *different* set of
answers to the same problems looks like, so that prosper can decide which of its own decisions are
load-bearing. Everything here is re-derived and described in prosper's terms. No code, type,
comment or test was copied from anywhere, and nothing here names another project: the technical
content stands on its own, and the reader who needs provenance should ask rather than assume.

**Read the caveats in `## Confidence` before acting on any of it.** This is one reading of two
unmerged pull requests by a third party. It is a source of *hypotheses about our own architecture*,
which is the only thing an external implementation is ever evidence for.

## The four differences that matter

### 1. A sampled render target need not leave the GPU

prosper serves a render target that a later draw samples from a **CPU-side cache**, so the image is
copied back before the sampling draw can be recorded. Measured: 68.5% of the backend call on the
shipped frontend, and it forces one of every two flushes.

The alternative is to key images by **guest address range rather than by role**. "Render target"
and "texture" become two *bindings on the same `VkImage`*, not two resources with a transfer
between them. The producer writes it; the consumer looks up the same address and gets a view of the
same allocation. There is nothing to copy because there is only one object.

What that costs instead: a `vkCmdEndRendering` and one image barrier, recorded into the *same*
command buffer. Per-image cached `{stage, access, layout}` elides the barrier when nothing changed.
The simultaneous case — a target sampled while still bound — is expressible by putting both the
descriptor and the attachment in `GENERAL` with the union of accesses.

**The question for prosper is whether its readback is a data-availability problem or a
hazard-tracking problem.** If a sampling draw cannot be recorded into the same command buffer as
its producer, the CPU copy is a symptom of that, not the cause.

### 2. Fence-per-submit is not the only way to know when work finished

prosper creates a `VkFence` per submission, submits, and blocks in `vkWaitForFences`. It uses no
timeline semaphores (`grep` returns nothing; they have been core since Vulkan 1.2 and prosper has
been on 1.4 since #3418).

The alternative: one timeline semaphore for the whole renderer. `vkQueueSubmit` takes a **null
fence**, signals the next tick, and returns it without waiting. Then:

- **Command-buffer recycling** compares a recorded tick against a cheaply-polled counter, and
  *grows the pool* when nothing is free rather than blocking on one.
- **Resource lifetime** is a queue of tick-stamped callbacks, drained per draw. Freeing a cached
  image or a staging allocation is one of these. Reclamation becomes continuous and never waits.
- **Blocking becomes guest-driven** — an explicit guest GPU-sync, or predication carrying a wait —
  rather than a property of how prosper submits.

Note the ordering dependency: § 1 must come first. A timeline semaphore cannot remove a wait whose
purpose is to deliver bytes the next draw is about to read.

### 3. Detiling on the GPU removes a second, independent reason to round-trip

Tiling and detiling can be compute shaders over device-local buffers rather than CPU loops. This
matters for sequencing: **if prosper reads pixels back partly so the CPU can detile them, then
fixing the image cache alone will not remove the round trip.** Establish which it is before
planning § 1, because the answer changes the size of the work.

### 4. A switch should describe the HOST, never the guest

Measured on prosper, 2026-09-25: the GPU/render layer reads **501** distinct `PROSPER_*`
environment variables — 77 diagnostics, 108 A/B opt-outs, **316 behaviour-changing**.

The comparison point is not "fewer". It is a different *kind*. In the implementation studied, the
whole project has **7**, and every one is an opt-out from a capability the emulator **detected** on
the host device — "use `shaderFloat16` if this GPU has it, unless told not to". **Not one selects
between two ways of emulating the guest.**

That is the distinction to apply to prosper's 316, one at a time:

- a **host-capability** opt-out is legitimate and permanent;
- a **diagnostic** is legitimate and should be named so;
- a **guest-behaviour selector** is a recorded *unresolved question*. Its real cost is not the
  branch: it is that no later fix can compound across it, because there is no single path to fix.

## Why prosper's wins do not compound, stated as a mechanism

This is the most useful thing in the study, and it is structural rather than cultural.

prosper recovers shader resource addresses by **static dataflow analysis over machine registers** —
`src/gpu/recompiler/indirect/rdna2_indirect_pointer_descriptor_range.cpp`, 1,106 lines of
VGPR/SGPR live-range analysis reconstructing, at recompile time, the address a shader will
dereference at run time.

The alternative is **partial evaluation of the shader's own address computation**: lift the shader
to a typed SSA IR, extract the sub-graph that computes each descriptor once per shader, cache it,
and re-execute that sub-graph on the CPU per draw against the user-data registers and a
guest-memory reader. The shader is the specification; the host interprets the slice of it that
produces descriptors.

The consequence is the entire point:

| | extension point | who benefits |
|---|---|---|
| register dataflow analysis | write an analysis for this shape | that shape |
| partial evaluation of the IR | teach the evaluator one more IR opcode | **every shader that ever uses it** |

**That is why prosper has `src/gpu/recompiler/gta5/` (1,837 lines of title-specific contracts) and
an implementation built the other way has no equivalent.** There is nowhere in that design to put
"for this shader, the descriptor is at address X". Not because its authors are more disciplined —
because the architecture gives a title-shaped fix nowhere to land, and the general fix is the
cheapest one available.

Two supporting properties, both cheap and both adoptable independently of the above:

- **Decode tables complete over the encoding space, not over what a title used.** Where a variant
  set is the cross-product of orthogonal modifier bits, enumerate the product and decompose the
  modifiers, rather than adding encodings as titles demand them. Adding one then costs what adding
  all of them costs, so there is no cheaper narrow option.
- **A failure taxonomy instead of a failure.** When the host cannot model something, classify it —
  which opcode, which operand, permanent vs transient vs stubbed — and report once per shader hash.
  One run then enumerates every gap with a named next action, instead of serialising bring-up one
  abort at a time. This is compatible with the charter's rule that an unsupported op is a FATAL
  gap: that rule governs what must eventually be implemented, this governs how a run behaves on the
  way there.

## The cheapest thing to adopt, and it is mechanically checkable

**A title id may appear in a comment, naming the capture that motivated a rule. It may never appear
in the rule's condition.** A grep for title ids outside comments under `src/` should return zero.
The rule itself is written in terms of format, extent, tile mode, sample count — properties any
title can present.

That single convention is most of what keeps a title-shaped line count at zero, and it costs
nothing to start.

## Confidence

- **HIGH** that prosper has no timeline semaphore, no buffer-device-address path, 501 render-layer
  switches, and a 1,106-line register-dataflow address recovery. Those are direct reads of this
  tree and are restated from this session's own measurements.
- **MED** on the external descriptions. They come from reading two **unmerged** pull requests by a
  third party; one is in a conflicting state and carries no human review. They are evidence about
  an *approach*, not about a shipped result.
- **LOW** on any performance claim attributed to that approach. Nothing in this document was
  measured by prosper. Their figures are theirs, on different hardware and a different driver.
- **A single PR cannot establish that someone else's wins compound.** What it can establish is that
  the title-keying which would *prevent* compounding is absent — which is a claim about
  possibility, not about outcome.
- **One sobering counter-datum, recorded because it cuts against the thesis:** even with none of
  the readback cost described in § 1, that implementation still needed periodic non-blocking
  flushes to keep the GPU fed. **Removing readback is necessary, not sufficient.** Do not expect
  § 1 alone to produce pipelining.

## What this does NOT license

A rewrite. Each of §§ 1-4 is separately testable against prosper's own tree, and § 3 in particular
may already be true here. The next step is to answer them about prosper — starting with whether its
detiling is on the CPU, because that answer changes the shape of § 1.
