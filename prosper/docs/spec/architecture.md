---
kind: spec
status: accepted
owner: area:infra
last-verified: 2026-10-05 61557f29
---

# Architecture: target shape and cross-cutting rules

What prosper is built to look like, and the rules that hold across every layer. Accepted rules are
policy today; proposed rules describe the target and become binding only when their ADR is
accepted. The layer order itself is in `layers.md` and the frame invariants in `performance.md`.
How the components work today is `docs/architecture/ARCHITECTURE.md`; the move plan with its
measurements is `docs/architecture/ARCHITECTURE_TARGET_TREE.md`.

## Target shape

The guest's x86-64 code runs natively, so everything below is about the platform underneath it,
never about the CPU. Each layer borrows the pattern that proved itself in an established project:
Wine for the API surface over a core, DXVK and vkd3d-proton for GPU submission, the yuzu/Ryujinx
and shadPS4 recompilers for shader translation, and Dolphin's FIFO player for replay testing. They
are verification references for structure only; no code, types or prose are taken from them.

```
 guest x86-64 (native, unmodified)
        | NID calls / syscalls
 +------v-----------------------------------------------------------------+
 | hle/<library>/   API surface: argument decoding, return codes, NIDs     |
 +------------------------------------------------------------------------+
 | guest/           guest semantics, written once for every host (ADR 0003)|
 |                  abi, tls, fault, image, memory, sync, scheduling, APR   |
 +------------------------------------------------------------------------+
 | gpu/             decode -> state -> resources -> execute; recompiler as  |
 |                  a pure function (ADR 0006)                              |
 +------------------------------------------------------------------------+
 | Vulkan backend   device, submission, pipeline cache (ADR 0004 moves it   |
 |                  out of tests/)                                          |
 +------------------------------------------------------------------------+
 | host/platform/   OS primitives only: vm, futex, thread, fiber, clock,    |
 |                  file, fault install -- one interface, one file per OS   |
 +------------------------------------------------------------------------+
 frontends/   app, screenshot, replay: consume the stack; nothing depends on them
 diagnostics  observe-only, cross-cutting: perf alarms, capture, timeline
```

## Direction not yet proposed as a rule

These follow from the shape above and the frame invariants, and each needs its own ADR before it
becomes a rule: one canonical resource identity per guest allocation with page-granular write
tracking, so several guest mappings of one allocation share one host resource (`PERF-P5`); a single wait and synchronisation model
in `guest/` that never assumes the host thread entering an HLE call is the one that returns from it
(fiber titles, #3615 / #3638 / #3623); a submit worker that records frame N+1 while frame N executes
(`PERF-P1`, `PERF-P6`); a typed GPU command representation between PM4 decode and the backend; and
an SSA IR inside the shader recompiler, migrated one instruction family at a time.

Deliberately not part of the target: a CPU translation layer or relinker (the guest runs natively),
a virtual interface on every boundary (implementations are chosen at build time, and there is no
LTO to remove the cost of a cross-unit call), a per-title behaviour database (see `TITLE-1`), and a
second GPU backend before the command representation exists.

## Rules

### ARCH-1 -- the guest runs natively; prosper translates the platform, not the CPU

prosper MUST NOT add a CPU emulator, a binary translator or an image relinker. Guest code executes
as loaded; work goes into the ABI, the libraries and the GPU underneath it.
Status: accepted
Enforcement: review: (a design boundary, not a code pattern; a translator has no single signature a scan could match)

### PLAT-1 -- host-platform conditionals live only in the OS seam

A host-platform `#if` (`_WIN32`, `__linux__`, `__APPLE__`, ...) MUST NOT appear in `src/hle`,
`src/loader`, `src/self` or `src/gpu`. OS behaviour is reached through an interface under
`src/host/platform/` with one backend file per OS (`docs/architecture/HOST_PLATFORM_SEAM.md`).
Existing sites are baselined and migrate when touched, each move in its own commit.
Status: accepted
Enforcement: ratchet:platform-ifdef, adr:0002

### PLAT-2 -- a platform arm holds primitives, never a copy of another arm's logic

Portable logic is written once. A platform arm MUST NOT be a stub or reduced copy of another arm.
Status: accepted
Enforcement: ratchet:platform-stub, review: (the ratchet sees only handlers named *_stub; a reduced copy under another name is invisible to it)

### TITLE-1 -- no title ids in shared-code conditions

Shared code MUST NOT branch on a title id. Behaviour one title needs is a general rule the evidence
supports, derived from what the guest presents. Naming the evidence in a comment is fine; tests
may use title ids as fixture data.
Status: accepted
Enforcement: ratchet:title-id

### TITLE-2 -- no title-named modules

No directory or module in shared code is named after a title. The existing one
(`src/gpu/recompiler/gta5/`) may shrink, never grow, and is generalised into properties of the
shader and its data.
Status: accepted
Enforcement: ratchet:title-dir

### CFG-1 -- runtime switches are read through the env cache and classified

A `PROSPER_*` switch is read through `src/diagnostics/env_cache.hpp`, never a new raw `getenv`.
Each new switch is classified in its PR as host-capability, diagnostic, or guest-behaviour
selector; a selector has an issue whose resolution sets the default and deletes the switch.
Status: accepted
Enforcement: ratchet:getenv, review: (the classification is a statement of intent no scan can derive)

### SIZE-1 -- no file grows past the line cap

A source file stays under the ratchet's line cap. Files already over it may shrink, never grow;
split with `tools/refactor/split_file.py`, which proves the split moved bytes rather than changed them.
Status: accepted
Enforcement: ratchet:file-size

### FAIL-1 -- an unsupported operation fails visibly

An unknown NID, shader opcode, storage format or PM4 packet the guest exercises is logged with its
identity and counted; it MUST NOT be skipped silently. A reject path is a backstop marked
`CONFIDENCE: LOW` with an issue, never a resolution.
Status: accepted
Enforcement: runtime:dropped-draws, runtime:skipped-dispatches, runtime:unimplemented-hle-calls

### ENT-1 -- ownership queries answer from local inventory only

Entitlement and add-content APIs answer from content declared and present in the dump, derived in
one place that every library exposing the question calls. They MUST NOT return "owned"
unconditionally, and MUST report every piece of content the dump does contain.
Status: accepted
Enforcement: review: (the distinction between deriving and hardcoding is semantic; each change needs the charter's free-SKU mutation arm, CLAUDE.md "Entitlement and add-content APIs")

### DIAG-1 -- diagnostics observe; they never change what the guest sees

Code in `src/diagnostics/` and every switch classified as diagnostic MUST NOT alter guest-visible
results. A switch that does is a guest-behaviour selector and is classified as one (`CFG-1`).
Status: accepted
Enforcement: review: (whether a value reaches the guest is a dataflow question; the producer/printer split is guarded separately by tools/env/check_diag_gates.py)

### HLE-1 -- hle/ is the API surface; engines live in guest/

A handler decodes arguments, calls an engine, and encodes the return code. Engine state and
behaviour shared by several libraries (APR's command buffers, executor and completions) live once
in `src/guest/`.
Status: proposed (adr:0003)
Enforcement: adr:0003

### GPU-1 -- the shader recompiler is a pure function

The recompiler takes shader bytes, the register state it reads, and a target description, and
returns SPIR-V plus a binding layout. It reads no environment switch, owns no cache and no device,
and includes nothing from `gpu/execute`, `gpu/state`, the backend or a frontend. Caching and
diagnostics wrap it from outside.
Status: proposed (adr:0006)
Enforcement: adr:0006

### VER-1 -- renderer changes are replayed against recorded frames before merge

A change to the GPU path is replayed against a corpus of recorded `.prgbundle` frames and their
output compared with recorded goldens before it merges.
Status: proposed (adr:0005)
Enforcement: adr:0005

### HLE-2 -- library coverage is generated, not estimated

The registered-versus-exported state of every reimplemented library is generated from the
registration code and reported per library, and the count of unregistered NIDs a title calls is
reported by the always-on alarm on every run.
Status: proposed (adr:0007)
Enforcement: runtime:unimplemented-hle-calls, adr:0007
