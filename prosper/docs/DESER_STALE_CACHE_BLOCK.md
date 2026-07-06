# DESER CRASH ROOT CAUSE — stale CachedReader block, NOT a typetree/descriptor bug (2026-07-06)

**TL;DR: the deserializer is fine. The crash reads correct-parse-state over WRONG FILE BYTES: Unity's
64KB CachedReader window is positioned/bounded for file block 2 (0x20000) but still holds block 7's
(0x70000) data — the block-2 `pread` is NEVER issued (verified by strace). Every "misaligned parse /
MatrixParameter byteSize 12 / descriptor field-count 0" observation is downstream garbage-in.**

## Evidence chain (all reproducible, artifacts in /tmp)

1. **The `r8` array walked by the generic reader (`0xd4cec0`) is NOT a typetree node array — it is
   `unity_builtin_extra`'s object-info table `{pathID, byteStart, byteSize}`** (24-byte entries).
   Verified against UnityPy for all entries in `/tmp/prosper_ttnodes.bin` (dump base file-off 0x6000):
   e.g. `{0x3b00, 0x76658, 0x4310}` = Hidden/CubeBlur pathid 15104, raw size 17168 — exact match.
   The `div` at `0xd4cef7` computes **`byteStart / 0x10000` = cache-BLOCK index** (elemSize 0x10000/
   0x100000 are Unity CachedReader block sizes), not an element count.

2. **The crash-time object is `Hidden/BlitCopy` (pathid 0x42, byteStart 0x27a10, size 0x18f4) — not
   CubeBlur.** Driver hit #6's `r8` → BlitCopy's table entry, and the reader window math matches on all
   three axes: `end−base = 0x9304 = BlitCopy_end(0x29304) − block2_base(0x20000)`;
   `cursor−base = 0x7a20 = BlitCopy_start − 0x20000 + 0x10` (16 bytes into the object).
   Hits #4/#5 = Sprites/Mask (0x716d0), UI/Default (0x737f8) — both block 7, windows match the same
   model (`base ↔ block base`, `end ↔ object end`).

3. **The buffer holds block 7, byte-for-byte:** `/tmp/prosper_buf_6.bin` (37636 B) is IDENTICAL to
   `file[0x70000..0x79304)` (37636/37636 bytes; vs 2415/37636 for `file[0x20000..)`). So the parser —
   correctly positioned at "BlitCopy+0x10, expecting a property-name length" — reads CubeBlur's
   mid-object bytes (ref offset ~5064: MatrixParameter `{m_Type=0,m_RowCount=4,pad}` = `00 04 00 00`)
   and gets **0x400**. At the CORRECT offset 0x27a20 the bytes are `08 00 00 00 "_MainTex"` — a sane
   length-prefixed string. BlitCopy parses clean under UnityPy, like every other shader.

4. **strace (the kill shot):** full boot under
   `strace -f -y -e trace=openat,read,pread64,lseek,close` (log `/tmp/claude2_strace.log`, crash
   reproduced). The COMPLETE I/O on unity_builtin_extra is:
   `openat` → `pread64(65536, offset=0)` → `pread64(65536, offset=458752 /*0x70000*/)` → `close`.
   **A read at 0x20000 (block 2) is never issued** — not even a failed one. Our `f_pread`/`f_read`/
   `f_lseek` HLE are thin passthroughs, so this is not a host-I/O bug: the guest never asks.

## What this means for the parse-level findings

- "MatrixParameter read 12B not 16B", "cursor drift", "descriptor field-count [r14+0x30]==0 skip
  branch" — these describe the parser's behavior over stale bytes. The parse state (start of a Shader
  object) never legitimately reaches MatrixParameter; the 0x400 IS CubeBlur's MatrixParameter tail
  bytes, sitting at the stale position by coincidence. No descriptor/typetree/static-init fix can help,
  because the input bytes are from the wrong file offset.
- The earlier reference checks stay valid and now cohere: the file data is well-formed, thousands of
  reads parse fine (they're all within correctly-fetched blocks), and the fault is deterministic
  (same load order → same stale block).

## The actual open question (the real next frontier)

**Why does Unity's cacher skip the block-2 fetch?** It updated the reader's bounds/cursor for block 2
(the window math above proves that) but not the data. Facts to constrain it:
- unity_builtin_extra is on the sync path ("path … not considered suitable for apr reads flags:0x0" —
  note: "apr" = AMPR async page reads; `libSceAmpr::8aI7R7WaOlc/a8uLzYY--tM/N-FSPA4S3nI` ARE called
  once each and stubbed to 0 — worth implementing/failing honestly, but this file didn't use it).
- Both successful preads ran on ONE thread (strace tid 461421, same tid that opened the file); the
  close came from another (461474). All 6 driver hits share one tid per run.
- Candidate mechanisms, ranked: (a) Unity FileCacherRead block-lookup wrongly HITS for block 2
  (multi-block LRU state corrupted / compare fed a wrong value under our env); (b) the fetch is
  dispatched to an async-read worker that never runs it, while the requester's wait is satisfied
  spuriously (our sync/equeue layer — same producer/consumer class as the render-loop deadlock);
  (c) allocator/pool overlap corrupting the cacher's bookkeeping fields.

**Decisive captures (small):** HWBP on the guest call site of the block-2 fetch decision — or simplest:
log every guest `pread/read/lseek` with a guest backtrace (gate on the builtin_extra fd) and HWWATCH the
cacher's block-index field / the buffer base `..601920` to see who last wrote it and who consulted it
before skipping the fetch. Compare the fetch call flow for block 7 (works) vs block 2 (skipped).

## Repro/artifacts
- `/tmp/claude2_strace.log`, `/tmp/claude2_boot.{out,err}` — strace boot (crash reproduced under strace).
- `/tmp/prosper_buf_6.bin` vs `file[0x70000..0x79304)` — byte-identity check.
- `/tmp/prosper_ttnodes.bin` — object-info table dump; decode with pathIDs above.
- UnityPy cross-check: all 43 builtin shaders parse; object table byteStart/size match the r8 entries 1:1.
