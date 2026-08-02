# AGC command-packet sizes — the contract, the audit, and the instrument

**Status: current.** Read this before changing any `begin_packet(a0, N, …)` count in
`src/hle/hle_agc.cpp`, before implementing a new AGC builder, and before concluding that a title's
command-buffer churn is the guest's own behaviour.

## The contract

prosper's AGC builders **append into the guest's own command buffer**. The guest owns that memory,
decides how large it is, and — this is the part that makes packet size an ABI fact rather than an
implementation detail — **reserves space for a sequence of packets before building any of them**,
using the sizes it was compiled against.

So a builder that emits **more dwords than the real AGC function** overruns a reservation the guest
made in good faith. Emitting **fewer** is safe: it wastes reserved space and nothing else. The
asymmetry matters, because it means the audit only ever has to prove an upper bound.

## Why this is invisible in almost every title

A guest can learn a packet's size two ways, and only one of them can be corrected at runtime:

* **It calls `sceAgc*GetSize`.** prosper answers, the guest reserves exactly what prosper will
  write, and the pair is self-consistent **at any size** — even where prosper's encoding differs
  from the real packet. Such a title cannot be broken by a wrong size, only by a wrong *answer*.
* **It has the size inlined** from the AGC headers at compile time. prosper is never asked and
  cannot be told. This title is exposed to every millimetre of difference.

That is the whole shape of the defect class: **silent in every title that asks, fatal in the one
that does not.** It is also why waiting for the next crash is a bad strategy — the next affected
title need not crash at all. #1748 is the fatal end of the spectrum (Asterix & Obelix - Babylon
Mission reserves a 16-dword submit epilogue, prosper's `RELEASE_MEM` was 9 dwords instead of 8, and
the resulting per-submit command-buffer chunk churn drained 16 GiB of direct memory in 125 s). The
quiet end looks like nothing but **allocator traffic and low frame rate** — 466 fresh ~600 KiB
chunks per second, in a title with memory to spare, is a performance defect wearing no label.

## The instrument

Two gated diagnostics in `hle_agc.cpp` (`AgcDcb::allocate_dw` / `begin_packet`), off by default and
unable to alter a single write:

```bash
PROSPER_DCBFULL=1   # every Dcb "buffer full" callback: sub-op, dwords needed, dwords available
PROSPER_DCBWIN=N    # every packet appended into a window of <= N dwords (N=1 selects the default 64)
```

**Read `avail`, not the window size.** For a segmented Dcb, `cursor_down - bottom` is the current
segment, not a reservation, and reading it as one produces false positives:

| `[dcbfull]` shape | meaning |
| --- | --- |
| `avail=0` | the buffer is genuinely exhausted — ordinary growth or a segment refill. Not a defect. |
| `avail>0` | space was deliberately left and prosper asked for **more than fits**. `need - avail` is the overrun, in dwords. |

#1748's signature was `need=9 avail=8` — overrun of exactly 1 — repeated once per submit.
`PROSPER_DCBWIN` then names the packets sharing that window, and a tightly sized window *is* the
guest's own size table for the packets in it: Babylon's 16-dword epilogue holding `AcquireMem` +
`EopAction` is what proved `RELEASE_MEM` must be 8 and not 9.

## The builder table

`dw` is what prosper emits, **measured** (not read off the source) by `test_agc_getsize`, which
drives every builder into a Dcb and records the cursor advance.

The "hardware" column is a **reference, not an authority**, and the confidence column says so per
row. Most AGC entry points are not a bare PM4 packet: several are library operations that may expand
to more than one packet or to none, and prosper's payloads are its own encoding (the
register-indirect forms carry a pointer + count rather than the register values; `ReleaseMem` carries
a `#312` build snapshot in its spare slot). **Do not shrink a builder on the published PM4 size
alone** — that is how a working title gets broken to satisfy a table. Only a guest reservation
(via `PROSPER_DCBWIN`) or guest disassembly settles a row.

| builder (`sceAgc…`) | prosper dw | PM4 packet it stands for | published dw | confidence | evidence |
| --- | --- | --- | --- | --- | --- |
| `DcbAcquireMem` / `AcbAcquireMem` | 8 | `ACQUIRE_MEM` (GFX10, with `GCR_CNTL`) | 8 | HIGH | all 7 payload slots are hardware fields; Babylon's 16-dword epilogue |
| `CbReleaseMem` (end-of-pipe action) | 8 | `RELEASE_MEM` | 8 | HIGH | Babylon's 16-dword epilogue (#1748), fixed from 9 |
| `DcbDmaData` / `AcbDmaData` | 9 | `DMA_DATA` | 7 | LOW | carries a 2-dword `#312` build snapshot; **no title evidence** |
| `DcbWaitRegMem` / `AcbWaitRegMem` | 9 | `WAIT_REG_MEM` | 7 | LOW | payload holds a poll interval + compare fn; **no title evidence** |
| `DcbJump` | 5 | `INDIRECT_BUFFER` | 4 | LOW | call-with-length, carries an explicit dword count (#319) |
| `DcbDrawIndex` | 7 | `DRAW_INDEX_2` | 6 | LOW | payload carries a 64-bit AGC draw modifier |
| `DcbDrawIndexAuto` | 7 | `DRAW_INDEX_AUTO` | 3 | LOW | same modifier; not a bare PM4 draw |
| `DcbDrawIndexOffset` | 3 | `DRAW_INDEX_OFFSET_2` | 5 | LOW | prosper emits **fewer** — safe direction |
| `DcbDrawIndexIndirect` | 4 | `DRAW_INDEX_INDIRECT` | 6 | LOW | fewer — safe direction |
| `CbDispatch` | 6 | `DISPATCH_DIRECT` | 5 | LOW | no title evidence |
| `Dcb/AcbDispatchIndirect` | 4 | `DISPATCH_INDIRECT` | 3–4 | LOW | no title evidence |
| `Dcb/AcbEventWrite` | 4 | `EVENT_WRITE` (address form) | 4 | MED | address-carrying form (#132) |
| `DcbWriteData` | 5 + n | `WRITE_DATA` | 4 + n | LOW | one dword over; no title evidence |
| `DcbSetIndexBuffer` | 3 | `INDEX_BASE` | 3 | MED | — |
| `DcbSetIndexCount` | 2 | `INDEX_BUFFER_SIZE` | 2 | MED | — |
| `DcbSetIndexSize` | 2 | `INDEX_TYPE` | 2 | MED | real `IT_INDEX_TYPE` packet |
| `DcbSetNumInstances` | 2 | `NUM_INSTANCES` | 2 | MED | real `IT_NUM_INSTANCES` packet |
| `DcbStallCommandBufferParser` | 2 | (AGC-level) | — | LOW | — |
| `DcbSet{Sh,Cx,Uc}RegisterDirect` | 3 | `SET_*_REG` (1 value) | 3 | HIGH | real `IT_SET_*_REG` packets (#395) |
| `DcbSet{Sh,Cx,Uc}RegistersIndirect` | 4 | `SET_*_REG` (n values) | 2 + n | LOW | prosper stores a pointer + count instead; smaller for n>2 — safe direction |
| `CbSetShRegisterRangeDirect` | n + 4 | `SET_SH_REG` (n values) | 2 + n | LOW | prosper prepends a 2-dword NOP marker `0x6875000d`, claimed to mirror the real library |
| `CbSetShRegistersDirect` | Σ(run + 2) | `SET_SH_REG` per run | Σ(run + 2) | MED | coalesces adjacent offsets — never larger than one packet per register |
| `DcbSetFlip` | 6 | (AGC-level) | — | LOW | — |
| `DcbSetPredication` | 4 | `SET_PREDICATION` | 3 | LOW | no title evidence (#319) |
| `DcbSetBaseIndirectArgs` | 4 | `SET_BASE` | 4 | MED | — |
| `DcbWaitUntilSafeForRendering` | 7 | (AGC-level, flip wait) | — | LOW | — |
| `Dcb/AcbPushMarker` | 1 + ⌈(len+1)/4⌉ | `NOP` with payload | same | HIGH | marker text is the payload (#641) |
| `Dcb/AcbPopMarker` | 2 | `NOP` | 2 | MED | — |
| `DcbResetQueue` | 2 | (AGC-level) | — | LOW | — |
| `CbNop` | n (1 ⇒ type-2 filler) | `NOP` | n | HIGH | `GetSize` implemented and asserted |

**Nothing in this table was changed on the strength of the "published dw" column.** The rows that
differ and have no title evidence are recorded as open, not fixed; see *Open* below.

## `GetSize` coverage — the half that can be fixed without hardware knowledge

libSceAgc 3.20 exports **65** `sceAgc*GetSize` functions. Before #1756 prosper answered **6**; the
other 59 fell through to the generic unimplemented path, which returns **0**.

Zero is the worst possible answer. A guest that sizes its buffer from `GetSize` reserves *nothing*
and the matching builder then overruns it — the same failure as #1748 but guaranteed rather than
off-by-one, and the failure #1137 already hit once ("stubbed to 0 the guest allocates a 128-byte
buffer and the builders overflow it → corrupt submit → RAGE fatal"), fixed then for the four NIDs
that title happened to call.

| | count | disposition |
| --- | --- | --- |
| `GetSize` exports in libSceAgc 3.20 | 65 | |
| answered by prosper before #1756 | 6 | Jump, AcquireMem ×2, EopAction, Rewind, Nop |
| **builder present, `GetSize` missing** | **26** | **fixed by #1756** — each returns the dword count prosper's own builder emits |
| builder absent as well | 33 | left alone: a size would be a guess, and the builder call fails anyway |

The fix needs no hardware size and cannot be wrong in the dangerous direction: the guest reserves
exactly what prosper will write. `test_agc_getsize` measures every builder and asserts its `GetSize`
equals that emission, so the two cannot drift apart; the sizes are single `constexpr` constants used
by both sides.

Three size-carrying builders are **deliberately excluded**: `DcbWriteData`,
`CbSetShRegisterRangeDirect` and `CbSetShRegistersDirect`. Their `GetSize` takes a count argument
whose position is not established (`CbNopGetSize` puts it in `a0`, but that is one data point), and
`CbSetShRegistersDirect`'s emission is data-dependent — it coalesces adjacent register offsets, so
its size is not a function of the count at all.

## Import is not call

An import scan over the 30-title local corpus reports these `GetSize` NIDs as imported by 11 titles
each — but those 11 import essentially the **entire** libSceAgc stub table, which is what a wholesale
static link looks like. **Do not read an import as a call.** The one asymmetry that is meaningful:
`sceAgcCbSetShRegisterRangeDirectGetSize` (`bxGoVxpdSPQ`) is imported by **22 of 30** titles,
including the Unity titles that import only a handful of AGC NIDs — a selective import. Call counts,
not imports, are what `PROSPER_PROGRESS=N PROSPER_PROGRESS_UNIMPL=1` measures.

## Ruled out

| Hypothesis | Evidence that killed it | Where |
| --- | --- | --- |
| A small `cursor_down - bottom` window at a buffer-full event means prosper overran a reservation | False positive on segmented Dcbs: The Pathless shows 58 such events, all with `avail=0` and `cursor_up == cursor_down` — the segment was simply exhausted. `avail > 0` is the real discriminator | #1756 |
| The 59 unimplemented `GetSize` NIDs are all live defects because 11 titles import them | Import is not call; those 11 titles import the whole stub table | #1756 |

## Open

* The `LOW`-confidence rows above where prosper is **larger** than the reference — `DmaData` (9 vs
  7), `WaitRegMem` (9 vs 7), `Jump` (5 vs 4), `WriteData` (5+n vs 4+n), `SetPredication` (4 vs 3),
  `CbDispatch` (6 vs 5), `DrawIndex`/`DrawIndexAuto`. Each needs a title that reserves tightly around
  it before it can be changed. `PROSPER_DCBWIN` over a new title is the cheapest way to get that.
* `w1KFAHVqpaU` is `sceAgcCbBranch` in the 3.20 export table, but prosper registers it as the DOLL
  "final buffer" submit variant (#232), derived from live disassembly and load-bearing. `CbBranch`
  also has a `GetSize`, which implies it is a *builder*. Recorded, not changed — a title calling it
  as a branch would currently reach the submit path instead.
* The three excluded size-carrying `GetSize` functions.
