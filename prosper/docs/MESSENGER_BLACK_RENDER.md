# Messenger black-render investigation

This is the canonical status for two visible Messenger failures:

- GitHub #300: the first gameplay level submits frames but its scene content is black.
- GitHub #299: the save-game list is not visible.

GitHub issues track what remains and own live findings. This document defines what evidence is current,
which conclusions are obsolete, and how the next investigation must be run. Update it when a finding
changes the investigation boundary; put run-by-run detail on the relevant issue.

## Current conclusion (2026-07-11)

There is **no verified current-master root cause** for either symptom.

The old project frontier, "the vertex shader cannot recompile because bindless vertex fetch is unresolved",
is solved. Dynamic fetch const-folding, per-instruction resource provenance, EUD descriptor recovery, and
both shader stages now execute. `NEXT_STEP_VERTEX_FETCH.md` is a historical implementation record.

#300 contains useful measurements, but it also contains mutually incompatible conclusions: texture decode,
alpha, transform collapse, render-target propagation, and a missing vertex-color binding were each called
the root cause and later overturned. Those tests were run while the graphics stack changed quickly. They
cannot be combined into one proof unless repeated on the same commit with the same captured frame.

The newest focused diagnostic, `772c44a` on `origin/fix/messenger-vertex-color-binding`, found that the
previously absent binding-9 color vertex buffer had become present with real packed-white data after the
EUD/resource fixes. The level was still black. That commit is based on `9b5bf27`, 27 commits behind
`origin/master` at the time of this update, so it narrows history but is not current-master verification.

#299 has only a shared-root hypothesis. It has not been shown whether the save-list rows are absent from
Unity's draw stream, dropped during realization, or rendered with bad inputs. SaveData persistence and
filesystem behavior changed recently (#389, #392), and the diagnostic run above reported flaky save-mount
behavior. Treat #299 as an independent HLE/UI investigation until a capture proves otherwise.

## Evidence policy

Every new finding posted to #299 or #300 must include:

- exact git commit and whether local changes were present;
- title dump ID, savedata directory/seed, input route, and relevant environment flags;
- pad-read, flip, submit, pass, and draw identifiers as applicable;
- hashes of shader code, resolved resource table, referenced input bytes, and output image/pass;
- the observation, the narrower conclusion it supports, and alternatives it does not distinguish.

Avoid "definitive" or "root cause" until a proposed fix changes the failing replay and a regression test
fails before the fix and passes after it. A diagnostic override proves only the boundary it directly changes.

## Required baseline

Re-establish both symptoms independently on `origin/master`:

1. Use a unique, explicitly recorded `PROSPER_SAVEDATA_DIR` and state whether it starts empty or seeded.
2. Record the exact inline, frame-anchored `PROSPER_PAD_SCRIPT` value. File-loaded routes are not on
   current master yet; do not cite a branch-only `scripts/<title>/reach-*.pad` path as the input.
3. Do not rely on `PROSPER_DET_CLOCK` or pad-read screenshot checkpoints; #302 confirmed those later
   implementations did not merge with PR #301.
4. Detect the reached state with a semantic/content metric, not elapsed time or one screenshot hash.
5. Repeat until the success rate and any alternate states are quantified.

Do not begin another graphics hypothesis cycle until this baseline is reliable enough to identify the same
failing submit/pass across runs. If savedata or guest state prevents that, fix the reproducer first.

## Tooling milestones

### 1. Immutable GPU capture/replay (#514)

Capture a failing draw-carrying frame locally: ordered draws, render-target transitions, decoded GPU state,
shader code, resource tables, referenced VB/CB/texture bytes, initial RT state, and alias/dependency metadata.
Replay it without loading Unity and require live-versus-replay state and output hashes to match. Captures
contain game data and remain gitignored; only minimized synthetic fixtures may be committed.

This converts a multi-minute, nondeterministic boot experiment into a revision-locked test that runs in
seconds and can be bisected, probed, and compared.

### 2. Strict shader/resource contract (#515)

Validate the descriptor interface declared by generated SPIR-V against the runtime `ShaderResourceTable`:
set/binding, descriptor class, stage visibility, guest address, byte range, stride, format, and statically
known accessed offsets. Diagnostic mode must reject missing, ambiguous, wrong-type, or undersized resources
before Vulkan robust-buffer behavior silently turns them into zeros.

### 3. Structured per-draw probes

On a replayed draw, expose vertex position/color/UV, raw texture samples, constant-buffer factors,
pre-blend fragment output, and final attachment output in a standard report. Existing switches such as
`PROSPER_RENDER_TESTPS`, `PROSPER_TESTTEX`, `PROSPER_CBUFLOG`, and draw isolation are useful primitives;
the goal is one repeatable probe rather than another collection of live-only A/B runs.

## Decision boundary

Do not start a persistent render-target-manager rewrite, another whole-stack audit, or an HLE workaround
based only on the historical #300 comments. Let the current baseline and immutable replay identify the
first bad contract. Once identified, implement the real behavior, derive a synthetic regression where
possible, then add a late-state local snapshot/checkpoint guard for the repaired Messenger path.
