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

**Reading it correctly took two corrections, both forced by the corpus.** Get these wrong and the
probe manufactures defects:

1. **Read `avail`, not the window size.** For a segmented Dcb, `cursor_down - bottom` is the current
   segment, not a reservation. The Pathless produces 58 buffer-full events on windows of 1-7 dwords
   — every one with `avail=0` and the cursor exactly at the limit, i.e. an exhausted segment.
2. **`avail>0` is only meaningful when `reserved=0`.** `available_dw()` subtracts the guest's own
   `reserved_dw`, so a title holding a reserve back reaches `avail>0, need>avail` at the end of
   *every* buffer, for whatever packet arrives next. Corpus-wide, **every** repeated `avail>0` triple
   carries `reserved=18`, and the `avail` value tracks the packet's own size rather than any
   per-packet budget — `SetIndexBuffer need=3 avail=2`, `EventWrite need=4 avail=2`,
   `WaitRegMem need=9 avail=7`, `WriteData need=69 avail=25`. prosper is not over on every builder by
   a different amount; the buffer is simply ending.

| `[dcbfull]` shape | meaning |
| --- | --- |
| `avail=0` | buffer exhausted — ordinary growth or a segment refill. Not a defect. |
| `avail>0`, `reserved>0` | the unreserved remainder ran out. Ordinary. Not a defect. |
| `avail>0`, `reserved=0`, **same triple repeating** | space was deliberately left and prosper asked for **more than fits**. `need - avail` is the overrun. **This is the signature.** |

#1748's signature was `need=9 avail=8 reserved=0` — overrun of exactly 1 — once per submit.
`PROSPER_DCBWIN` then names the packets sharing that window, and a tightly sized window *is* the
guest's own size table for the packets in it: Babylon's 16-dword epilogue holding `AcquireMem` +
`EopAction` is what proved `RELEASE_MEM` must be 8 and not 9.

## What the probe got wrong before it got anything right

The most reusable result in this document is not a packet size. It is that **the first reading rule
this probe shipped with would have reported seven builders as oversized, and every one was an
artefact** — including one carrying the *exact arithmetic* of the real defect:

| sub-op | reported as | why it was noise |
| --- | --- | --- |
| `SetCxRegsIndirect` / `SetShRegsIndirect` | `need=4 avail=3` → over by 1 | **The same shape as #1748's genuine off-by-one.** Three titles produce it |
| `SetIndexBuffer` | `need=3 avail=2` | |
| `EventWrite` | `need=4 avail=2` | |
| `WaitRegMem` | `need=9 avail=7` | |
| `WriteData` | `need=69 avail=25` | |
| `SET_SH_REG` (op-carried) | `need=18 avail=2`, `need=30 avail=1` | |

All of them carry `reserved=18`. `available_dw()` subtracts the guest's own `reserved_dw`, so a title
holding a reserve reaches *space left, but not enough* at the end of **every** buffer, for whatever
packet arrives next — and the leftover tracks each packet's own size, which is why the "overrun"
appeared to be a different amount on every builder. Read naively that says prosper is oversized on
seven builders by seven different amounts, which is not credible; the buffer is simply ending.

Two further corrections the same instrument forced on its own operator, both recorded as instrument
traps 55 and 56 in `GAME_COMPAT_ORCHESTRATION.md`:

* a **window** is the current segment of a segmented Dcb, not a reservation — The Pathless produces
  58 buffer-full events on windows of 1–7 dwords, every one with the cursor exactly at the limit;
* the census's per-title counts were `[dcbfull]` **log lines**, which cap at 512 and then sample, so
  four titles reporting "512–514" were reporting the cap. Every rate derived from them was void.

The rule that survives is narrow: **`avail>0` AND `reserved=0` AND the same triple repeating**, and
it is *sufficient, not necessary* (see the census). A probe whose first three readings were all wrong
is worth more documented than deleted — the failure modes are the transferable part.

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
| `DcbDmaData` / `AcbDmaData` | **7** | `DMA_DATA` | 7 | HIGH | was 9; the 2 extra were the `#312` build snapshot, and the remaining 6 payload dwords are exactly the hardware field list. **Shrunk by #1756** |
| `DcbWaitRegMem` / `AcbWaitRegMem` | 9 | `WAIT_REG_MEM64` | 9 | HIGH | **not oversized — this row was published wrong twice.** Enumerating fields on both sides: hardware is header + `ENGINE/FUNCTION`, `POLL_ADDRESS_LO/HI`, `REFERENCE` (2), `MASK` (2), `POLL_INTERVAL` = 9; prosper is header + addr (2), mask (2), reference (2), compare_function, interval = 9. Delta **0** |
| `DcbJump` | 5 | `INDIRECT_BUFFER` | 4 | MED | +1 is a **predication flag slot** that `sceAgcSetPacketPredication` writes into this packet afterwards (#319) — live per-packet state, not a snapshot, so it needs a home before the packet can shrink |
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

## Which builders are genuinely oversized

The question this audit exists to answer, and the one implementing `GetSize` does **not** answer:
`GetSize` makes a guest that *asks* agree with prosper's version of the packet; only shrinking the
builder makes prosper agree with the hardware. Babylon died of the second problem.

Of the 30 builders, three emitted more dwords than the packet they stand for with no explanation
other than prosper's own payload. They are not one class:

| builder | was | is | verdict |
| --- | --- | --- | --- |
| `DmaData` (Dcb + Acb) | 9 | **7** | **Genuinely oversized, and fixed.** Identical to `RELEASE_MEM`: 2 of the 9 were the `#312` build-time snapshot of the destination qword, and the remaining 6 payload dwords are *exactly* the hardware `DMA_DATA` field list (dst lo/hi, srcOrImm lo/hi, numBytes, selectors). Unlike `RELEASE_MEM` there is no spare slot at the hardware size, so the snapshot leaves the packet entirely. |
| `WaitRegMem` (Dcb + Acb) | 9 | 9 | **Not oversized — delta 0. My flag was wrong twice before this, and each wrong figure was published.** "9 vs 7" compared it against the 32-bit `WAIT_REG_MEM` when the sub-op is `R_WAIT_MEM_64` and the guest supplies a 64-bit mask and reference; "9 vs 8" was a second guess at the same row. Enumerate instead: hardware `WAIT_REG_MEM64` spends header + `ENGINE/FUNCTION`, `POLL_ADDRESS_LO`, `POLL_ADDRESS_HI`, `REFERENCE`×2, `MASK`×2, `POLL_INTERVAL` = **9**; prosper spends header + addr×2, mask×2, reference×2, `compare_function`, interval = **9**. Eight payload scalars on both sides. **Closed, and off the open list.** |
| `Jump` | 5 | 5 | **Oversized by 1, but the extra dword is load-bearing.** `cmd[4]` is a predication flag that `sceAgcSetPacketPredication` writes into this packet *after* it is built (#319) — live per-packet state, not a diagnostic snapshot. Shrinking to the hardware `INDIRECT_BUFFER` size needs that state rehoused (a side table keyed by packet address, as the fence journal already does), which is a design change wanting its own evidence. **Open.** |

So: **one of the three was a latent Babylon and is now closed; one is not oversized at all and my
flag was wrong twice; one is real but cannot be fixed by deletion.** Both wrong flags came from
comparing packet *names* rather than enumerating fields — enumerate them. The remaining table rows where prosper is larger
(`WriteData` +1, `SetPredication` +1, `CbDispatch` +1, the two draws) have neither an identifiable
private payload nor title evidence, and stay open.

### The hazard the `DmaData` shrink carries, and why a draw-count A/B cannot see it

`kDwDmaData` is not only the builder's size. `dma_patch_recover_header` — the #1124 fix that rebuilds
a DMA header Alex Kidd (`PPSA02664`, rung 6) clobbers to zero — **stamps a header and probes
readability** with it. A hard-coded 9 left there while the builder emits 7 writes a 9-dword header
over a 7-dword allocation: the command-processor walk steps two dwords into the *next* packet and
desyncs the remainder of the submit, and `decode_pm4`'s historical `npl >= 8` arm then reads
`dd_build_pre` out of whatever follows. The `guest_readable(cmd, 9*4)` probe would also start failing
near a buffer end, silently disabling the recovery and restoring #1124's "0 DMA copies".

Both literals now follow the constant. The lesson generalises: **a packet size is not local to its
builder** — anything that stamps, probes or validates that packet must read the same constant.

The hazard is **measured, not predicted**. Building exactly that broken intermediate — builder at 7,
`dma_patch_recover_header` still stamping 9 — and booting Alex Kidd:

| build | `#1124` recovery fires | `SHORT FOLD` | draws/submit at t=45 s |
| --- | --- | --- | --- |
| 9-dword builder (before) | 8 (the report's cap) | **0** | 30 |
| 7-dword builder, literals moved (**this change**) | 8 | **0** | 30 |
| 7-dword builder, `#1124` left at 9 (**the hazard**) | 8 | **32** (the report's cap) | 30 |

```text
[agc] SHORT FOLD SubmitDcb #1: guest declared 126 dwords, decode stopped after 120 (95.2%) at stream
      dword 0x00000020 — the remaining 6 dwords, INCLUDING ANY BIND, were never applied
```

From the very first submit, and on nearly every submit after it. Note the last column: **the draw
count is 30 in all three builds**, so a draw-count A/B — the check I had been running — cannot see
this at all. It also doubles as the positive control for the `SHORT FOLD` grep: silent on the two
correct builds *because* it fires 32 times on the broken one, not because it cannot fire.

### What the `DmaData` shrink was checked against

There was no failure to make return — no title in the corpus was observably harmed by the 2 extra
dwords — so the check is that the change is **null**, not that a defect reappears. Same build, same
routes, only `kDwDmaData` and its two payload writes differing, `PROSPER_PROGRESS` series at 45–65 s:

| title | 9-dword | 7-dword |
| --- | --- | --- |
| *Alex Kidd* `PPSA02664` (rung 6, exercises #1124) | 30 draws/submit, 0 short folds, recovery at its cap | 30, 0, at its cap |
| *Dragon Quest VII* `PPSA17942` | 96, 96, 91, 87, 94, 91 draws/submit | **identical series** |
| *The Messenger* `PPSA24651` | 11 | 11 |
| *Dead Cells* `PPSA15552` | 173 | 170 (its own frame-to-frame band is 169–173) |
| *GTA V* `PPSA04263` | 8 | 8 |

That is the expected result: the removed dwords were read by exactly one consumer,
`dma_build_pre_changed`, behind the default-off `PROSPER_GENERATION_GUARD`. That leg now **announces
itself as inert** rather than returning a quiet `false`, because an opt-in check that has silently
stopped running is indistinguishable from one that ran and found nothing. `prosper_label_hist_dma_built`
still records the same init event out-of-band, so #312's label-history legs are untouched, and
`decode_pm4` still reads the 9-dword form in captures recorded before this change.

## `GetSize` coverage — the half that can be fixed without hardware knowledge

libSceAgc 3.20 exports **65** `sceAgc*GetSize` functions. Before #1756 prosper answered **6**; the
other 59 fell through to the generic unimplemented path, which returns **0**. This change adds **22**,
bringing the answered total to **28**.

Zero is the worst possible answer. A guest that sizes its buffer from `GetSize` reserves *nothing*
and the matching builder then overruns it — the same failure as #1748 but guaranteed rather than
off-by-one, and the failure #1137 already hit once ("stubbed to 0 the guest allocates a 128-byte
buffer and the builders overflow it → corrupt submit → RAGE fatal"), fixed then for the four NIDs
that title happened to call.

| | count | disposition |
| --- | --- | --- |
| `GetSize` exports in libSceAgc 3.20 | 65 | |
| answered by prosper **before** #1756 | 6 | Jump, AcquireMem ×2, EopAction, Rewind, Nop |
| **added by #1756** | **22** | each returns the dword count prosper's own builder emits |
| answered by prosper **after** #1756 | 28 | 6 + 22 |
| builder present but still unanswered | 3 | the size-carrying trio below — argument position unknown |
| builder absent as well | 34 | left alone: a size would be a guess, and the builder call fails anyway |

(Count the delta and the total separately: **22 NIDs are added**, bringing the answered total from 6
to 28. `git diff 928b21f5..HEAD -- prosper/src/hle/hle_agc.cpp | grep -c '^+ *RN(".*get_size'` is 22,
and the counter-arm — reverting only those registrations — fails exactly 22 assertions.)

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

## Corpus census (30 titles, 45 s headless boot each)

Run with the shipping build and the corrected signature above.

**No title in the corpus produced the signature.** Zero events with `avail>0, reserved=0` in any of
the 30 dumps.

That silence is worth something only because the rule was shown to speak: reintroducing the #1748
defect (`RELEASE_MEM` back to 9 dwords) and re-running Asterix & Obelix - Babylon Mission produces
**256** signature events among the first 512 reported (the report caps at 512 and then samples), all
on one triple — `ReleaseMem/EopAction need=9 avail=8 reserved=0`. The rule fires on the known defect
and is silent everywhere else.

**Do not read that as "no title has an overrun".** The signature is *sufficient, not necessary*, and
it has two known blind spots:

* **A reserve masks it.** An overrun that happens while `reserved_dw > 0` is classified as the
  ordinary end-of-buffer condition — and the corpus is dominated by titles that hold `reserved=18`
  permanently, for which this rule can never fire at all. Those titles are unmeasured, not cleared.
* **Only reservations that end at `cursor_down` are visible.** The probe hangs off `allocate_dw`'s
  failure path, so it sees an overrun only when the packet does not fit the Dcb. A guest that
  pre-computes offsets *inside* a large buffer and expects each packet at a known position is
  corrupted silently, with no event of any kind.

Both are why the per-builder table keeps its open rows rather than treating this census as closure.

The runs are also 45 s headless boots on one implicit route with no renderer, so they cover each
title's startup and whatever it reaches unattended — not its gameplay.

**Command-buffer churn** — and a correction to how it was first reported here. The original figures
came from counting `[dcbfull]` **log lines**, which the report caps at 512 and then samples. Bendy,
GTA V, Terminator and Babylon all sat at 512-514 lines: that was **the cap, not a count**, and the
"11.4/s" rates derived from it were meaningless (instrument trap 51, in this project's own list).

The probe now keeps an exact tally and prints an `armed` banner plus a running
`seen=<packets> full=<events>` line every 2^14 packets, so a zero is self-evidencing and a rate is a
rate. Re-measured, 45 s headless boot each:

| title | fps | packets built | buffer-full events | rate |
| --- | --- | --- | --- | --- |
| *Syberia: Remastered* `PPSA30140` | ~1.9 | 311,296 | **0** | **0.0/s** |
| *Dragon Quest VII* `PPSA17942` | ~12 | 2,326,528 | 74 | 1.6/s |
| *GTA V* `PPSA04263` | — | 81,920 | 588 | 13.1/s |
| *Bendy and the Ink Machine* `PPSA27616` | ~8 | 4,472,832 | 13,368 | **297/s** |
| *Asterix & Obelix* `PPSA30490`, **after** #1748 | ~124 | 3,375,104 | 16,545 | 368/s |

What this does and does not establish:

* **Syberia's zero is established.** The banner proves the probe was armed, and 311,296 packets
  produced no buffer-full event at all. Its 1.9 fps is not command-buffer churn.
* **"No slow title pays this cost" was wrong**, and the capped numbers hid it. Bendy runs at 297/s —
  the same order as Babylon *after* its fix (368/s). High buffer-full traffic is therefore **not**
  the defect signature: Babylon at 368/s reclaims its chunks and allocates 23 blocks in a run, while
  the same title before #1748 allocated 1053 and died. What distinguishes them is whether the chunks
  come back, which `[memhle] alloc_main_dmem` measures directly and this probe does not.
* Whether **Bendy** reclaims is unmeasured and worth one `PROSPER_MEMLOG` run.

The same cap limits the overrun census below: a title whose log hit 512 `[dcbfull]` lines had the
remainder truncated, so "no signature events" is established only for titles that stayed under the
cap — Syberia among them. For Bendy, GTA V, Terminator and Babylon it is **not established**, and
re-running them against the exact tally is the cheap way to close it.

## Called and unimplemented

Measured with `PROSPER_PROGRESS_UNIMPL` (call counts, not imports) over the same runs — the whole
`libSceAgc` **and** `libSceAgcDriver` surface the corpus actually calls and prosper does not
implement. Callers completed once all 29 bootable dumps had run:

| NID | function | called by |
| --- | --- | --- |
| `MlEw1feXcjg` | `sceAgcQueueEndOfPipeActionPatchData` | `PPSA04263` |
| `vuSXe69VILM` | `sceAgcDcbGetLodStats` | `PPSA21564` |
| `Ikfdt-rIqCE` | *(not in the 3.20 export table)* | `PPSA04263` |
| `F0Y42t-3e18` | `sceAgcDriverInitResourceRegistration` | `PPSA15552`, `PPSA26414`, `PPSA28061` |
| `pWLG7WOpVcw` | `sceAgcDriverUnregisterResource` | `PPSA15552` |
| `F0ZXt5q0ZTA` | *(not in the 3.20 export table)* | `PPSA15552`, `PPSA28061` |
| `U9ueyEhSkF4` | *(not in the 3.20 export table)* | `PPSA15552`, `PPSA26414`, `PPSA28061` |
| `JQc0956gCf0` | *(not in the 3.20 export table)* | `PPSA28061` |

`sceAgcQueueEndOfPipeActionPatchData` is the notable one: prosper implements its sibling
`…PatchAddress` (`0fWWK5uG9rQ`) and not this, so GTA V builds an end-of-pipe action, patches its
address, and has its **value** patch silently dropped. Second: `sceAgcDriverInitResourceRegistration`
and `sceAgcDriverUnregisterResource` are called by **Dead Cells**, a rung-6 title with a snapshot
guard — prosper answers the *query* half of resource registration (#539, #660, both real defects
when they returned stack residue) but not `Init` or `Unregister`. All tracked on #1768. Four of the
eight are absent from the 3.20 export table, as `T6xuVw0KUJo` was on #1748: the corpus contains
titles built against SDK versions that dump does not cover, so "not in the DB" is not "not real".

**None of the 22 `GetSize` functions this change implements is called by any title in the corpus.**
That fix is therefore *preventive*, not the repair of an observed failure — stated plainly because
the distinction matters. Its justification is that the same zero-answer has already been fatal twice
(#1137's RAGE buffer overflow, #1748's leak), each time found by a crash in the one title that
happened to ask, and each time fixed for that title's NIDs alone.

## Ruled out

| Hypothesis | Evidence that killed it | Where |
| --- | --- | --- |
| A small `cursor_down - bottom` window at a buffer-full event means prosper overran a reservation | False positive on segmented Dcbs: The Pathless shows 58 such events, all with `avail=0` and `cursor_up == cursor_down` — the segment was simply exhausted | #1756 |
| `avail>0` at a buffer-full event means prosper overran a reservation | False positive whenever the guest holds a reserve: every repeated `avail>0` triple in the corpus carries `reserved=18`, across sub-ops whose `avail` tracks their own size (`need=3 avail=2` … `need=69 avail=25`). `reserved=0` is required | #1756 |
| `SetShRegsIndirect` / `SetCxRegsIndirect` are one dword too large — three titles reach `need=4 avail=3` | Same artefact: all three carry `reserved=18`, and prosper's 4-dword encoding (count + 64-bit pointer) cannot fit in 3 anyway, so a shrink would need a different encoding and its own evidence | #1756 |
| Syberia is slow because of #1748-style command-buffer churn | **Falsified for Syberia**, whose probe was armed (banner in the log) and saw **0** buffer-full events in 311,296 packets. **Not established for Bendy**, which the corrected tally puts at **297/s** — the same order as Babylon *after* its fix (368/s), so buffer-full rate is not itself the defect signature; whether Bendy reclaims its chunks is unmeasured | #1756 |
| The 59 unimplemented `GetSize` NIDs are all live defects because 11 titles import them | Import is not call; those 11 titles import the whole libSceAgc stub table, and no corpus title calls any of them | #1756 |

## Open

* The rows still larger than the reference after the `DmaData` shrink — `Jump` (+1, the predication
  slot that needs rehousing),
  `WriteData` (5+n vs 4+n), `SetPredication` (4 vs 3), `CbDispatch` (6 vs 5),
  `DrawIndex`/`DrawIndexAuto`. None has an identifiable private payload to delete, so each needs a
  title that reserves tightly around it before it can be changed. `PROSPER_DCBWIN` over a new title
  is the cheapest way to get that.
* **Two rows are larger than the reference only for small `n`, and the table's "safe direction" note
  hides it.** `SetXxRegistersIndirect` is a fixed 4 dwords against `2 + n`, so it is *larger* at
  **n ≤ 1** and smaller from n = 3 up — and n = 1 is exactly the arithmetic of the `need=4 avail=3`
  events three titles produce on that same builder. Those events are classified as artefacts because
  all three carry `reserved=18` (see *Ruled out*), which settles the classification but **not** the
  size question. `CbSetShRegisterRangeDirect` is `n + 4` against `2 + n` — larger by 2 for **every**
  n, because of the 2-dword NOP marker prosper prepends on a secondary implementation's claim that
  the real library does the same. Both want a tight-window title, and the register path is where
  Nikoderiko's #305 already lives.
* `sceAgcQueueEndOfPipeActionPatchData` and `sceAgcDcbGetLodStats` are called and unimplemented
  (see the table above).
* `w1KFAHVqpaU` is `sceAgcCbBranch` in the 3.20 export table, but prosper registers it as the DOLL
  "final buffer" submit variant (#232), derived from live disassembly and load-bearing. `CbBranch`
  also has a `GetSize`, which implies it is a *builder*. Recorded, not changed — a title calling it
  as a branch would currently reach the submit path instead.
* The three excluded size-carrying `GetSize` functions.
