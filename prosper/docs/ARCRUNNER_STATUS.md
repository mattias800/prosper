# ArcRunner (`PPSA21406`) — status and evidence

Unreal Engine 4.27-plus. **Rung 0 — renderer bring-up, with no real visible game graphics yet.**
The long-lived compatibility index is [#1817](https://github.com/mattias800/prosper/issues/1817),
and the primary allocator, barrier, and intro-movie investigation is
[#1226](https://github.com/mattias800/prosper/issues/1226).

Read `## Ruled out` before forming a hypothesis. The allocator investigation has already separated
the shape of the terminal corruption from its possible authors, and one early combined-arm run was
invalidated by its own incomplete census before the corrected run answered the narrow question.

## Current state

The title boots through UE4 initialisation and its `.pak` load, installs the renderer, and submits
real GPU work, but no game frame is ever composited. The earlier async-compute submit ABI failure is
fixed. Between about 8 and 14 seconds the guest's `RenderThread 1` instead faults at one of two
already-recorded sibling sites.

The usual terminal chain is:

```text
WORKER-THREAD FAULT: sig=11 addr=0x30016000 rip=eboot+0x127e751
insn @rip: 48 8b 01            (mov rax,[rcx])   rcx=0x30016000
[fault] thread='RenderThread 1' on-guest-TCB=NO(host-%fs leak?)
```

`0x30016000` is far below the guest arena (`0x2000000000`–`0x9fc0000000`) and is read out of a
structure whose neighbouring fields are ordinary guest pointers. It is the misaligned dereference
of the tagged free-list value `0x2000000001`, not a separate poison value written directly by a
known PM4 path.

The same fault site reproduces under three guest-argument combinations. One important apparatus
detail is that `PROSPER_GUEST_ARGS=-force-gfx-direct` lets the signal kill the process with exit 139
and prints **no** worker-fault report, while empty `PROSPER_GUEST_ARGS`, or `gdb` with `SIGSEGV`
passed through, reports the same fault completely. Use the empty-argument route when fault evidence
is required.

Static import attribution now identifies what the sibling `eboot+0x117811f` path was trying to do.
The guest walks the object references retained by each submitted command-allocation page, reads a
side-command descriptor from `item+0x98`, opens a `sceAgcDcbSetPredication` window, and then passes
the descriptor's `{target, dword count}` to `sceAgcDcbJump` before marking that jump predicated. The
fault is the guest's first dereference of that descriptor, **before** Prosper receives the Jump:

```text
1178064: mov r14, [rax+0x98]   ; retained item -> side-command descriptor
117811a: call SetPredication   ; begin
117811f: mov rcx, [r14]        ; fault: r14 == 0
1178122: mov r8d, [r14+8]
117812d: call DcbJump          ; target=[r14], num_dw=[r14+8]
```

This excludes a silent null-Jump fallback as a legitimate fix. It does not yet say whether the
retained item is stale/corrupted or structurally valid with only its side segment absent.

No screenshot belongs in the compatibility record yet. The latest content-selective capture
retained exactly two non-alpha-only frames: frame 12 was a uniform solid-yellow clear and frame 50
was a uniform solid-white clear. Full inspection found no geometry, text, or game imagery in either.

## Reproduction

Build and run inside the `ps5ys` distrobox, with `TMPDIR` on disk rather than `/tmp`:

```bash
cd prosper
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1 \
  ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0
```

The combined diagnostic arm used for the most recent narrow experiment is:

```bash
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1 \
PROSPER_INIT_SUPPRESS=ptr PROSPER_REL1_FORGE_SUPPRESS_ALL=1 \
PROSPER_FORGE_TRIP=1 PROSPER_PRESENT_NZLOG=1 \
  ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0
```

`PROSPER_REL1_FORGE_SUPPRESS_ALL` is a default-off diagnostic, not a title fix. It suppresses every
exact REL1 `forges_freelist_ptr` candidate, including live paired fences. A valid run must prove the
lever with an exact terminal `FORGE-DECISION-TOTALS` census where `candidates == suppressed` and
`landed == 0`; malformed values print `NOT ARMED`.

## Combined init-plus-forge suppression experiment

The first arm, on `dfd89f3f`, was **provisional and void for causality**. Init suppression reached
sparse ordinal 4,096 and forge detail reached at least ordinal 64, but the only totals snapshot was
candidate 1. The worker-fault path calls `_exit(90)`, so no `atexit` summary can run, and the original
first-plus-every-256 reporting cadence could not prove that every later candidate was suppressed.

The corrected diagnostic reports every candidate through ordinal 256 and then every 256th. Its
schedule is mutation-tested: ordinals 1, 64, 127, and 256 report, 257 is quiet, and 512 reports. A
second bounded title run on `fb3daaa4` independently reached `INIT-SUPPRESS #1024` and ended with the
exact final line:

```text
FORGE-DECISION-TOTALS candidates=20 suppressed=20 landed=0 mode=all
```

That run presented frames 0 through 52, did not hit the `0x30016000` terminal chain, and instead
faulted at the other known sibling site:

```text
WORKER-THREAD FAULT: sig=11 addr=(nil) rip=eboot+0x117811f
r14=0 r15=0x2420e48230
```

The result settles only the intervention's narrow necessity question: the specific
`0x2000000001 -> 0x30016000` terminal chain does **not** survive when neither known label write
lands. Suppressing both writes is not a title fix. The all-mode arm deliberately drops live fences,
so it neither isolates the init write from the fence write nor attributes the sibling null-object
fault.

Durable run records are in the #1226 comments for the
[CPU seam](https://github.com/mattias800/prosper/issues/1226#issuecomment-5164678099),
[provisional first arm](https://github.com/mattias800/prosper/issues/1226#issuecomment-5164771700),
[census correction](https://github.com/mattias800/prosper/issues/1226#issuecomment-5164799214), and
[valid corrected arm](https://github.com/mattias800/prosper/issues/1226#issuecomment-5164904408).

## Ruled out

Do not re-derive these without contradictory new evidence.

| Hypothesis | Evidence that killed it | Ref |
| --- | --- | --- |
| The `0x30016000` "POOLSHIFT byte-shifted pool pointer" is a **structurally different artifact** from the `0x2000000001` free-list value — the two shapes this issue tracked as separate legs since #1249 | They are the same shape one dereference apart. A qword that is a pointer with its **low bit set** misreads, when dereferenced, to the target qword shifted down by 8: `*(0x2000000001) = 0x30016000`, byte-for-byte the value the terminal fault dereferences. The pop at `eboot+0x127e751` both **returns** that node — the guest's own `FMallocBinned3 Attempt to free/realloc an unrecognized block 2000000001` fatal — and **stores** the misread as the next head, which is the SIGSEGV. Confirms the session-10 "the byte-shift is a READ ARTIFACT" note with a measurement. **This is about the shape, not the author** — see the next row. | #1226, #1754 |
| **The paired 4-byte `DMA_DATA` immediate-zero init is the "first half" of the damage**, systematically destroying live `FFreeBlock::NextFreeBlock` links (#1754 left this at `CONFIDENCE: MED`) | Not systematic. Whole-run totals with `PROSPER_INIT_TRIP=1`: **n=1024, overptr=926, member=10, both=1** — 90% of these inits do overwrite pointer-shaped content, but only ~1% of destinations are on an idx=1 chain the walk models, and exactly **one** init in the run is both. The overwritten pointer is almost always the label pool's own stale link in a block the guest holds. Two cautions: `member` positives appear **late**, so any single sample before ordinal ~512 reads 0 and must not be quoted as the run (this is how the first draft of this row got it wrong); and the scope is *not on the idx=1 chains the walk models*, not *on no list anywhere*. The walk is positive-controlled per sample by `mb3_freelist_selftest()` for both arming and traversal. `CONFIDENCE: LOW` on any causal role for the init. | #1226 |
| A prosper **PM4 write path** stores `0x30016000` into guest memory | `PROSPER_WRITE_TRAP=0x30016000` (checks `RELEASE_MEM` sel 1/2/3, `EVENT_WRITE`, `WRITE_DATA`, both `DMA_DATA` forms): **0 hits** across a full faulting run, positive-controlled by arming `0x1` alongside it (that control reaches ordinal 2,048 in the same run). The value is produced by the guest's own misaligned read, above. Three limits on how far this negative reaches: PM4 paths only (compute writeback, the Vulkan backend and the HLEs are outside it); the scan is 4-byte-strided from each payload base, so an *unaligned* poison inside a `DMA_DATA` **copy** is not seen; and a copy is scanned only to its first 4 KiB. | #1226, #1754 |
| A prosper GPU write targets the `0x3001600000` page the poison decodes to | `PROSPER_PROVENANCE_ADDR=0x3001600000:0x10000` reports zero overlapping writes across a full faulting run; `PROSPER_POOLSHIFT=1` is also 0. The page is an ordinary 64 KiB guest `sceKernelBatchMap` mapping, consecutive in the same series as the earlier-reported `0x30015f0000`. | #1226, #1754 |
| The renderer's fold latency (the guest outrunning our deferred label writes) is causal | `PROSPER_RENDER=0` faults at the same site with the same value, about 6 seconds in instead of about 21 seconds. | #1226, #1754 |
| Suppressing the forging fence — or both known label writes — fixes the underlying allocator corruption | The fence-only arm removes the terminal `0x30016000` and moves the fault to `0x2400100024001`, two pops farther along the same walk. The first combined run at `dfd89f3f` was **PROVISIONAL/VOID for causality**: its forge population was at least 64 but its only totals snapshot was candidate 1. After the terminal census was fixed and mutation-tested, the corrected arm at `fb3daaa4` independently reached `INIT-SUPPRESS #1024` and ended with the exact final line **`candidates=20 suppressed=20 landed=0`**. It presented frames 0–52, then faulted at the other already-known sibling site `eboot+0x117811f` with `r14=0` (`addr=(nil)`), not at the `0x30016000` pop. Thus removing both writes is not a title fix, but the valid arm settles its narrow necessity question: the specific terminal `0x2000000001 -> 0x30016000` chain does **not** survive when neither known write lands. It does not isolate init from fence, nor attribute the sibling fault, because all-mode deliberately drops live fences. Content-selective capture retained exactly two non-alpha-only images; both were inspected and are single-colour clears (frame 12 solid yellow, frame 50 solid white), with no game imagery. | #1226, #1754 |

## Next discriminators

1. Attribute the `eboot+0x117811f` sibling null-object fault without using the combined suppression
   arm as a title fix. First prove whether the failure exists on current master in an ordinary
   unsuppressed run, then use the existing generic fault-memory peek to recover the original retained
   item saved at `rbp-0xa8` and its allocation-page metadata at `rbp-0x70`.
2. Isolate the init and REL1 fence interventions only with arms whose independent lever witnesses and
   terminal populations are complete. Do not infer authorship from a moved terminal fault alone.
3. Revisit the per-queue barrier model and intro-movie path from #1226 only after checking their
   current master reproductions; neither is settled by the allocator arm.
4. Capture and inspect a real game image before advancing the tracker or adding an ArcRunner
   screenshot to `COMPATIBILITY.md`.
