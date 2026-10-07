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

## Runtime and performance direction

Each of these is a proposed ADR with a rule here or in `performance.md`: pipelined GPU submission
retiring guest-visible effects in stream order (ADR 0009, `PERF-P8`); one canonical resource per
guest allocation validated by page tracking (ADR 0010, `PERF-P9`); a typed GPU command
representation with deterministic passes (ADR 0011, `GPU-2`); an SSA IR inside the recompiler
(ADR 0012, `GPU-3`); one guest sync and scheduling model with no thread-identity assumptions
(ADR 0013, `SYNC-1`); pipeline compilation off the submit thread from a persistent cache (ADR 0014,
`PERF-P7`); declarative per-library HLE export tables (ADR 0015, `HLE-3`); and a release gate on the
reference workloads (ADR 0016, `PERF-G1`). Engineering-infrastructure proposals follow the same pattern: logging
channels and typed configuration (ADR 0017, `CFG-2`), typed guest pointers (ADR 0018, `HLE-4`),
fuzzed parsers (ADR 0019, `VER-2`), per-library conformance suites (ADR 0020, `VER-3`) and a
bug-report bundle (ADR 0021, `OPS-1`). Profile-guided and AI-assisted optimisation are deferred
until those instruments exist (`performance.md`).

Deliberately not part of the target: a CPU translation layer or relinker (the guest runs natively),
a virtual interface on every boundary (implementations are chosen at build time, and there is no
LTO to remove the cost of a cross-unit call), a per-title behaviour database (see `TITLE-1`), a
separate runtime orchestration layer (frame pacing belongs to submission, ADR 0009), and a second
GPU backend before the command representation exists.

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
supports, derived from what the guest presents, or isolated code with its measurement
(`src/gpu/recompiler/gta5/`), as the charter states. Naming the evidence in a comment is fine;
tests may use title ids as fixture data.
Status: accepted
Enforcement: ratchet:title-id

### TITLE-2 -- no title-named modules

No new title-named directory or module is added to shared code. The existing isolated one
(`src/gpu/recompiler/gta5/`) may shrink, never grow; target-tree move 10 proposes generalising it into
properties of the shader and its data.
Status: accepted
Enforcement: ratchet:title-dir

### CFG-1 -- runtime switches are read through the env cache and classified

A `PROSPER_*` switch is read through `src/diagnostics/env_cache.hpp`, never a new raw `getenv`.
Each new switch is classified in its PR as host-capability, diagnostic, or guest-behaviour
selector; a selector has an issue whose resolution sets the default and deletes the switch.
Status: accepted
Enforcement: ratchet:getenv, ci:switch-registry, review: (the registry checks that a class exists; whether it is true is a statement of intent no scan can derive)

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

### GPU-2 -- the backend consumes typed operations, never PM4

PM4 decode produces an ordered list of typed operations per submit, each with its resolved state
and guest accesses. Execution consumes only that list, and optimisation is a set of pure passes over
it, each replay-tested for unchanged guest-visible output.
Status: proposed (adr:0011)
Enforcement: adr:0011

### GPU-3 -- the recompiler lowers through an SSA IR

RDNA2 instructions lower into an IR on which passes run before SPIR-V is emitted, one instruction
family at a time, each family gated on byte-identical SPIR-V over a recorded corpus.
Status: proposed (adr:0012)
Enforcement: adr:0012

### SYNC-1 -- one wait model, no host-thread identity across a fiber switch

Every blocking guest wait goes through one model that names what it waits for. No state keyed by
host thread is assumed to survive an HLE call that can switch fibers.
Status: proposed (adr:0013)
Enforcement: runtime:hle-blocking-wait, adr:0013

### HLE-3 -- each library declares its exports in one table

A reimplemented library lists the functions it implements in one declaration table of names,
handlers and the argument semantics a C++ type cannot carry; NIDs are derived from names, so no
firmware symbol data is committed. Registration and a typed relay trace of every call's decoded
arguments are generated from the table. The Windows calling-convention bridge is already generated
by `emit_sysv_to_ms_bridge` and stays so.
Status: proposed (adr:0015)
Enforcement: adr:0015

### CFG-2 -- diagnostics are log channels; settings are typed configuration

Diagnostic output goes through named channels with levels, enabled by one variable. Host
capabilities and user settings are declared once in a typed schema, read from file, flags and
environment, and validated at start-up. Only temporary selectors remain individual switches.
Status: proposed (adr:0017)
Enforcement: adr:0017

### HLE-4 -- guest memory crosses the HLE boundary as a typed pointer

A handler argument that refers to guest memory is a `GuestPtr<T>` or `GuestSpan<T>`, which compiles
to a raw access in release builds and is checked against the guest memory map in diagnostic builds.
Status: proposed (adr:0018)
Enforcement: adr:0018

### VER-2 -- every parser of untrusted bytes is fuzzed

SELF/ELF, PM4, RDNA2 decode and capture deserialisation each have a coverage-guided fuzz target
under sanitizers, seeded from synthetic inputs only, and every crash it finds becomes a regression
test before the fix merges.
Status: proposed (adr:0019)
Enforcement: adr:0019

### VER-3 -- each library has a conformance suite that names its evidence

HLE tests live per library, and each states the evidence its expectation rests on: a trace of the
real guest, a published contract, firmware symbol data, or guest disassembly.
Status: proposed (adr:0020)
Enforcement: adr:0020

### OPS-1 -- one command produces a complete bug report

`--report`, and any fatal fault, writes a versioned bundle with build, host, driver, configuration,
alarms, unimplemented calls, recent log lines and backtraces, and never game bytes or absolute host
paths.
Status: proposed (adr:0021)
Enforcement: adr:0021

### HLE-5 -- no host exception unwinds into guest frames

Code in `src/{hle,loader,self,gpu}` reports failure through return codes and the logging abort path.
A `throw` there is allowed only when it is caught before leaving prosper code, with a comment saying
where.
Status: accepted
Enforcement: ratchet:host-throw, adr:0022

### LOCAL-1 -- everything derived from a title stays on the user's machine

Game content is never committed or uploaded, and neither are captures and capsules, replay
corpora, pipeline-cache files, fuzz inputs taken from dumps, or bug-report bundles carrying game
bytes. What a PR may commit is prosper's own data about a run -- hashes, counts, timings, pipeline
keys -- and screenshots under the charter's screenshot rules. How title-derived shader programs in
test fixtures are treated is decided in ADR 0025 before this rule is accepted.
Status: proposed (adr:0025)
Enforcement: adr:0025, review: (whether a file is derived from game content is a question about its origin, which no scan of its bytes can answer; the gitignore covers dumps, not derivatives)

### TITLE-3 -- isolated title code is registered with its measurement and exit

Every isolated title-specific unit in shared code is listed with the mechanism it implements, the
measurement and issue that justify it, and the condition under which it is removed; the list and the
tree match in both directions.
Status: proposed (adr:0024)
Enforcement: ratchet:title-dir, adr:0024

### VER-4 -- an evidence run records its switches

Screenshot and snapshot runs record every `PROSPER_*` switch set, and a run counts as acceptance
evidence only if each is a host-capability or diagnostic switch; a selector disqualifies it.
Status: proposed (adr:0023)
Enforcement: adr:0023

### GPU-5 -- a guest Wave64 program runs natively, by proof or exact emulation, or is refused visibly

On a host without a native 64-lane subgroup for the stage, a Wave64 program runs only when its
width independence is proven or an exact emulation route admits it; otherwise it is refused with a
`[wave64-unsupported]` line. An approximate vote is only a default-OFF selector. Until the ADR is
accepted the only admitting route is proof (`ProvenVotes`); the emulation routes do not exist yet.
Violated today: the compute Wave64 scalar-pair projection does not consult the fabricated-zero
mark, so a synthetic zero can stand in for a lane mask on the default path (#4714).
Status: proposed (adr:0028)
Enforcement: adr:0028
