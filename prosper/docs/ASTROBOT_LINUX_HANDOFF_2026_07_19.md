# Astro Bot native-Linux graphics handoff (2026-07-19)

> **Draft handoff — documentation only. Do not merge yet.** Tracks #1054 and continues the visual
> correctness work after #825. The functional PC-relative scalar-table fix still belongs on this branch.

This document hands the active Astro Bot (`PPSA21564`) graphics investigation from the Windows/WSL
host to a native Linux development machine. It records the exact merged frontier, the currently proven
failure, local-only evidence, dead ends and diagnostic caveats, a concrete implementation direction,
and the acceptance checks for the real goal.

The active focused tracker item is [#1054](https://github.com/mattias800/ps5ys/issues/1054). The
historical umbrella [#825](https://github.com/mattias800/ps5ys/issues/825) is closed because control-flow
progression through the first-level hub was achieved on Linux and Windows. That closure did **not** mean
the native title graphics were complete. The current goal is stricter: obtain genuine nonblack Astro
graphics from the normal Linux headless renderer, then validate the Linux app and cross-check Windows
headless/app behavior.

## Executive status

- Start from remote `master`. At final handoff publication it is
  `f13b90cf88834a9ce963a04a74715aab0fada41b`, including PR #1030 plus the subsequently merged
  compute-shader and AudioIn work in PRs #1053 and #1055.
- The handoff branch is `fix/issue-1054-astro-pcrel`. It is already pushed and claimed on #1054.
- Astro reaches `title_controller_ship` and the title level starts on the established Linux diagnostic
  route. Guest execution, input scripting, VideoOut flips, and substantial GPU submission all advance.
- The current raw/composited headless title captures are still entirely black. This is not a screenshot
  timing problem: manifests show thousands of presents and hundreds of composited frame publications
  while every pixel remains black.
- PR #1030 removed the immediately preceding pixel-shader resource failure at PC 137. The main title PS
  `0x5002af200` now reaches PC 221 and rejects `s_getpc_b64 s[0:1]`.
- PC 221 is **not** an indirect branch. It is the first of six compiler-generated constructions of the
  same PC-relative 64-byte scalar constant buffer. The table is loaded with
  `s_buffer_load_dwordx4` using a dynamically computed scalar byte offset.
- A live `/proc/<pid>/mem` probe proves that the table begins at `code + 0x15f0`, 20 bytes after the
  reported instruction blob ends at `code + 0x15dc`. The 64-byte payload is present and readable in the
  guest mapping.
- prosper already folds the analogous PC-relative **MUBUF** embedded-table idiom and already retains its
  proven post-`s_endpgm` tail in shader cache keys. It does not recognize or fold this **SMEM scalar
  buffer-load** form. Extending that narrowly proven mechanism is the next implementation, not adding a
  generic `s_getpc_b64` no-op.
- No functional #1054 fix has been written on this branch yet. This draft is deliberately a handoff,
  not a claim that the shader or black output is fixed.

## Repository and coordination state

At the time this handoff was written:

- `origin/master`: `f13b90c` (`Merge pull request #1055 ... audioin-core`)
- active branch: `fix/issue-1054-astro-pcrel`
- active issue: #1054, labeled `bug`, `area:infra`, `in-progress`, and `agent:astro-title`
- no other open PR carries the `agent:astro-title` label
- open PR #1052 (`perf(executor): reuse analyzed shader content hashes`) is green and clean, and touches:
  - `prosper/src/gpu/gpu_executor.cpp`
  - `prosper/tests/test_shader_recompile_cache.cpp`

The branch was rebased onto that exact master tip immediately before publication. Fetch again before
editing because this repository moves frequently. If #1052 has merged, rebase the handoff branch and
inspect its cache-key/span changes before implementing #1054. The expected core #1054 work is mostly in
`rdna2_to_spirv.cpp`, but cache-span regression coverage naturally overlaps
`test_shader_recompile_cache.cpp`.

Do not create a duplicate issue or branch. Continue #1054 on this branch and update the existing draft
PR. This repository is an agent project; the owner explicitly requested that agents not request reviews
from real people. Perform exact-head author verification and use CI, focused negative controls, and
programmatic image evidence. Keep the PR draft until it contains a real fix and its required renderer
verification.

## What is already working

The original #825 bring-up is extensive and should not be restarted:

- SELF/ELF load and linking, guest `main`, worker threads, fibers, VideoOut, input, save data, and the
  native ASOBI engine boot path work far beyond the issue's original fiber crash.
- PR #837 reached the loading screen.
- PR #842 advanced through title/world-map control flow.
- PR #852 reached the title with native Windows video.
- PR #876 established scripted Linux and Windows progression through the first-level hub.
- PRs #922 and #961 restored/rendered the opening video surface.
- PR #884 fixed Windows directory enumeration needed by Astro cinematics.
- PR #970 fixed compute raw-buffer provenance across CFG joins for the Astro title path.
- PR #1023 implemented the RDNA2 unsigned-byte conversion operations exposed by Astro.
- PR #1030 recovered direct scalar-buffer resources, removing the title PS's PC 137 rejection.
- The wider RDNA2 audit in PR #889 and subsequent shader fixes are on master. Treat current code and
  the AMD RDNA 2 ISA guide as authoritative; do not revive old assumptions from pre-audit logs.

The route documentation and committed app screenshots are in:

- `scripts/astrobot/README.md`
- `scripts/astrobot/reach-first-level.pad`
- `scripts/astrobot/reach-first-level-windows.pad`
- `docs/screenshots/issue-825-astrobot-linux-sony-presents.png`
- `docs/screenshots/issue-825-astrobot-windows-sony-presents.png`
- `docs/screenshots/issue-825-astrobot-windows-title.png`

Those screenshots prove frontend/video/control-flow milestones. They do not prove that the current
headless title render is correct.

## Current proven rendering failure

The strongest exact-head diagnostic for PR #1030 used:

```sh
PROSPER_GUEST_FS=1 \
PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_NO_COMPUTE=1 \
PROSPER_AVP_SYNTH_FRAMES=120 \
PROSPER_PAD_SCRIPT=@scripts/astrobot/reach-first-level.pad \
PROSPER_PAD_SCRIPT_LOG=1 \
PROSPER_DBG=1 \
PROSPER_RENDER_SCALE=4 \
  ./build-linux/screenshot /path/to/PPSA21564-app0 \
    --output-dir /path/to/output --count 1 --seconds 1 --warmup-seconds 78
```

This was a **control-flow/shader diagnostic only**:

- `PROSPER_NO_COMPUTE=1` deliberately disables guest compute.
- `PROSPER_AVP_SYNTH_FRAMES=120` is synthetic video, not decoder acceptance.
- `PROSPER_RENDER_SCALE=4` reduces resolution.

It reached:

```text
LevelDocument Loaded: title_controller_ship [title]
GAME: Level has started: title_controller_ship
```

and recorded:

- present count: 3995
- PC 137 rejects for title PS `0x5002af200`: 0
- PC 221 rejects for that PS: 6
- PC 221 word: `0xbe801f00` (`s_getpc_b64 s[0:1]`)
- screenshot source: `raw_scanout`
- `source_seq=3995`, `frame_seq=0`
- `distinct_rgb_colors=1`, `nonblack_rgb_pixels=0`
- CRC32 `08ed2210` (black 3840x2160 scanout)

The earlier 60-image Linux run is also useful evidence. It began with raw scanout samples and later
published composited frames. Its summary reported:

- 60/60 requested images saved
- 60 distinct source publications
- 57 rendered/composited samples
- maximum composited `frame_seq=621`
- maximum present count 5551
- all retained frames black (`nonblack_rgb_pixels=0`, one distinct RGB color)
- pixel content remained identical for the final 56.14 seconds

Therefore the black result is not explained by a dead flip loop or by sampling only before compositor
publication. Real shader/draw failures remain.

The diagnostic log is noisy: a long `PROSPER_DBG=1` run printed roughly 260,000 repeated
`[exec-recompile-reject]` lines. Use a bounded run or filter/narrow diagnostics; do not repeat that
volume unnecessarily.

## PR #1030: the completed predecessor

PR #1030 merged as `9c63435`. Its branch exact head was
`b5f883ae11ce4f0ed92100f8f8f27909b156d537`.

The title PS instruction at dword PC 137 is:

```text
s_buffer_load_dwordx2 s[0:1], s[24:27], vcc_lo offset:4
```

The V# was directly placed in entry SGPRs, while the dynamic offset was held in VCC_LO. The resource
fold knew the V# value but previously published an `SrtUse` only when it also had an SRT key. The fix
allows a fully known entry V# to be published as a keyless exact-consumer-PC resource. A production-path
regression verifies resource creation, binding, and compute recompilation, and a negative control showed
that reverting the production condition makes both new assertions fail.

Verification recorded on the exact PR head:

- Linux build: pass
- Linux ctest: 121/121 pass
- native Windows focused `agc_shader_create` and `dynfetch_fold`: 2/2 pass
- hosted Linux, Linux App, Windows MinGW, Windows App, macOS, and macOS App: pass
- Messenger snapshot: 29/29 qualifying and structural/nonblack, 8 pixel changes
- Evergate: 9/9 qualifying and structural/nonblack, 8 pixel changes
- Dead Cells: 44/44 qualifying and structural/nonblack, 43 pixel changes
- Blasphemous II: 98/98 qualifying and structural/nonblack, 97 pixel changes

The final merge combined #1030 with later master shader changes (#1041 and #1044). Those merges were
textually clean, but the full four-title snapshot matrix was not rerun on merge commit `9c63435`.
Run current-master/current-branch guards before publishing the next renderer change.

## Exact PC-relative scalar-table evidence

The failed raw shader is the pixel shader at guest address `0x5002af200`.

The original failure dump is 5596 bytes / 1399 dwords and ends at:

```text
pc=1394 len=2 EXP
pc=1396 len=2 EXP
pc=1398 len=1 s_endpgm
```

There are six `s_getpc_b64 s[0:1]` sites at dword PCs:

```text
221, 349, 402, 530, 583, 710
```

Each computes the **same** absolute table base relative to the next instruction, exactly matching the
AMD ISA's `D.u64 = PC + 4` contract:

| getpc PC | next-PC byte | added byte literal | resolved byte offset |
|---:|---:|---:|---:|
| 221 | `222*4 = 0x378` | `0x1278` | `0x15f0` |
| 349 | `350*4 = 0x578` | `0x1078` | `0x15f0` |
| 402 | `403*4 = 0x64c` | `0x0fa4` | `0x15f0` |
| 530 | `531*4 = 0x84c` | `0x0da4` | `0x15f0` |
| 583 | `584*4 = 0x920` | `0x0cd0` | `0x15f0` |
| 710 | `711*4 = 0xb1c` | `0x0ad4` | `0x15f0` |

The first sequence is:

```text
pc=221  s_getpc_b64 s[0:1]
pc=222  s_add_u32  s0, 0x1278, s0
pc=224  s_addc_u32 s1, 0, s1
pc=225  s_mov_b32  s2, 64
pc=226  s_mov_b32  s3, 0x10005004
...
pc=233  s_buffer_load_dwordx4 s[4:7], s[0:3], s107
...
pc=241  s_buffer_load_dwordx4 s[8:11], s[0:3], s106
```

The scalar offsets are computed by preceding scalar bitfield/multiply operations and select rows from
the 64-byte buffer. Other branches repeat the same descriptor construction and use the same pool.

### Live guest-memory proof

A one-off WSL diagnostic started the renderer, then read 8 KiB at guest address `0x5002af200` through
the running process's `/proc/<pid>/mem`. This did not modify the emulator. The probe did not reach the
title marker during its 110-second bound, so it is **memory-layout evidence only**, not a new progression
run.

Facts from the dump:

- full dump SHA-256:
  `CD5D7FA63E730F2860EE31F50ADA314FE32F43A941F093E9188155CF1B2F4AA6`
- the first 5596 bytes are byte-for-byte equal to the existing failed shader dump
- failed shader dump SHA-256:
  `C51E9F8CD8D652728721EF6B868E548098ABF0E274B59BC26060ABBEE641ED6D`
- `s_endpgm` finishes at byte `0x15dc`
- bytes `0x15dc..0x15ef` are a 20-byte post-program/alignment region
- the table starts at byte `0x15f0`
- the table is exactly 16 dwords / 64 bytes

Table words:

```text
00000000 00000000 3f800000 00000000
3f800000 00000000 00000000 00000000
3f800000 3f800000 00000000 00000000
00000000 3f800000 00000000 00000000
```

Interpreted as floats:

```text
0 0 1 0
1 0 0 0
1 1 0 0
0 1 0 0
```

The local-only artifacts on the originating Windows machine are:

```text
%USERPROFILE%\Downloads\AstroBot-Linux-handoff-tail-probe.bin
%USERPROFILE%\Downloads\AstroBot-Linux-handoff-tail-probe.log
%USERPROFILE%\Downloads\AstroBot-shaders-current\exec_ps_5002af200.bin
%USERPROFILE%\Downloads\AstroBot-Linux-1030-title-diag.log
%USERPROFILE%\Downloads\AstroBot-Linux-1030-title-diag\
%USERPROFILE%\Downloads\AstroBot-linux-60-20260719-183308\
```

These contain title-derived data and remain gitignored/local-only. Do not commit them. Transfer the
8 KiB probe privately to the Linux host if useful, or reproduce the bounded memory probe there.

## AMD primary-source contracts

Use the AMD RDNA 2 ISA guide linked from `CLAUDE.md` (document 70648), not a secondary emulator, for
instruction semantics.

Relevant contracts from the guide:

- `S_GETPC_B64`: destination receives the byte address of the next instruction (`D.u64 = PC + 4`).
- SMEM reads consecutive dwords into SGPRs without format conversion.
- for scalar buffer reads, the descriptor contributes base, stride, and `num_records`.
- the effective scalar address is descriptor base + instruction OFFSET + SOFFSET.
- SOFFSET is an unsigned byte offset; the two low address bits are ignored for dword alignment.
- for scalar buffer operations the instruction offset must be positive.
- stride participates in bounds checking, not address multiplication.

KytyPS5 is available as a secondary behavioral reference but was neither needed nor consulted for this
failure. The live Astro bytes plus AMD semantics fully establish the current table shape. Do not copy
code, tests, comments, or prose from another emulator.

## Why the current recompiler rejects

The existing implementation is deliberately fail-closed:

- `detect_pcrel_tables` in `src/gpu/rdna2_to_spirv.cpp` recognizes a proven
  `s_getpc_b64 -> add/addc -> V# -> MUBUF raw load` embedded-table idiom.
- It records `MUBUF instruction PC -> copied table dwords` in
  `RegState::mubuf_pcrel_tables`.
- MUBUF emission folds the runtime index to a bounded SPIR-V select chain, returning zero out of bounds.
- the fold runs before external-resource gating, because self-contained shader data needs no Vulkan
  descriptor.
- `rdna2_recompile_code_span` includes only the proven table tail, so cache ownership and identity retain
  exactly the bytes that affect generated SPIR-V.
- `s_getpc_b64` is accepted only when the prepass found such a table; otherwise it rejects rather than
  allowing a PC value to leak into unmodeled address math.

Astro's title PS uses the same address construction but consumes it with SMEM
`s_buffer_load_dwordx4`, not MUBUF. The current prepass's SMEM case only kills overwritten scalar facts;
it never records a PC-relative scalar table. `RegState::mubuf_pcrel_tables` stays empty, so the SOP1
handler rejects at the first `s_getpc_b64` (PC 221).

This explains both the exact failure and why generic support is unsafe: making `s_getpc_b64` a no-op
would leave later address arithmetic operating on fabricated zeros and could emit plausible but wrong
constants.

## Recommended implementation shape

This is a proposed design, not yet code. Preserve fail-visible behavior outside the fully proven idiom.

1. Generalize PC-relative table analysis so it can return both:
   - current MUBUF table loads (`instruction PC -> table dwords`)
   - scalar-buffer SMEM table loads (`instruction PC -> table dwords`)

   A small result struct is clearer than adding another output parameter to the existing unordered map.
   Keep the code-blob required span alongside or continue reporting it through `required_dwords`.

2. In the scalar-analysis walk, inspect SMEM **before** killing its destination facts. Recognize only
   buffer-load opcodes `0x8..0xC`, initially only those exercised and tested. Require:
   - SBASE low word has a live `s_getpc_b64`-derived byte offset
   - the matching high word is proven by the paired `s_addc_u32 ... + 0`
   - descriptor word 2 (`num_records`) is a known, nonzero, dword-aligned constant
   - table base and length are dword aligned
   - the table length is narrowly bounded (the existing 1024-byte ceiling is reasonable)
   - no branch enters between the newest contributing fact and the load, using the existing entry
     soundness rule
   - the required tail is within the caller-provided readable code window before copying bytes

   Descriptor word 3 need not affect address calculation per AMD, but recognizing the exact observed
   `0x10005004` form can be an additional conservative gate if desired. Do not make the implementation
   title-address-specific.

3. Store each proven scalar load's table by its exact instruction PC. Multiple loads and all six Astro
   setup sites may point to the same copied 16-dword payload.

4. In SMEM emission, check the proven scalar-table map **before** `allow_smem` and external resource
   binding. For a matched load:
   - obtain the tracked SOFFSET scalar SSA value using the existing register-offset logic
   - compute `(SOFFSET + instruction_offset) >> 2`
   - for each returned dword, select the corresponding literal-pool word with the same bounded-select
     pattern already used by MUBUF folding
   - return zero for out-of-bounds indices, matching the bounded V# contract
   - write the results to consecutive `rs.sreg` entries
   - clear any stale `rs.sreg_srt` descriptor provenance on those data destinations

5. Accept `s_getpc_b64` when **any** proven PC-relative table consumer exists, not only when the MUBUF
   map is nonempty. Continue erasing the architectural PC pair instead of fabricating host/guest
   addresses; the proven consumer is folded independently.

6. Ensure `rdna2_recompile_code_span` includes byte `0x15f0 + 64 = 0x1630` for the live shape when its
   caller supplies a sufficiently large readable window. This is critical:
   - the owning cache copy must contain the literal pool
   - table bytes must participate in the shader cache key
   - failed/successful shader dumps should retain the proven tail
   - changing a table word must invalidate/recompile rather than hit stale SPIR-V

7. Do **not** broaden `registered_shader_dwords` or the dynamic descriptor fold to scan arbitrary bytes
   after AGC `shader_size`. `tests/test_agc_shader.cpp` intentionally proves that descriptor discovery
   cannot inspect a valid-looking fetch outside the registered header size. The graphics recompiler is
   already called with a bounded readable maximum window and its span analysis can retain a narrowly
   proven tail. Keep the generic AGC metadata safety contract intact.

8. Keep arbitrary `s_getpc_b64`, PC-derived external pointers, unbounded tables, non-buffer SMEM forms,
   unsupported address arithmetic, and untracked SOFFSET values rejected.

## Required focused tests

Add tests that prove behavior rather than mere decode acceptance:

1. A compact synthetic program with:
   - `s_getpc_b64`
   - `s_add_u32` + `s_addc_u32`
   - `num_records=64`
   - a descriptor configuration word
   - a tracked scalar byte offset
   - `s_buffer_load_dwordx4`
   - an observable scalar-to-vector/output use
   - `s_endpgm`, alignment gap, and a 16-dword table tail

   Execute the generated SPIR-V and assert selected table contents, not just nonempty output.

2. Use at least two offsets (or two loads) so a broken implementation that always chooses table element
   zero cannot pass.

3. Assert `rdna2_recompile_code_span` includes the alignment gap and full table.

4. Recompile through the graphics shader cache, mutate a table word, and assert a cache miss plus changed
   SPIR-V/output. This proves the tail is in the owning key.

5. Negative controls should reject:
   - a bare `s_getpc_b64` with no proven table consumer
   - omitted/truncated table tail
   - an out-of-bound/unreasonably large `num_records`
   - a broken high-half `s_addc` chain
   - an unsupported/untracked scalar offset if it reaches emission

6. Run a production negative control before publishing: temporarily disable only the new scalar-table
   match and verify the new positive test and live shader fail at the expected frontier; restore it and
   verify both pass.

The existing MUBUF PC-relative tests around T12 in `test_rdna2_to_spirv.cpp` and the cache tests in
`test_shader_recompile_cache.cpp` are the closest templates. Extend the project-owned mechanism; do not
paste title shader bytes into the repository.

## Native Linux execution plan

### 1. Establish a clean native baseline

```sh
git fetch origin --prune
git switch fix/issue-1054-astro-pcrel
git rebase origin/master

cd prosper
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-linux -j"$(nproc)"
ctest --test-dir build-linux --output-on-failure
```

Confirm Vulkan uses the intended hardware device, not llvmpipe, unless software rendering is an explicit
A/B. Record GPU, driver, Vulkan implementation, and video decode path in the eventual PR evidence.

### 2. Reproduce the exact shader frontier cheaply

For rapid control-flow iteration, the documented synthetic/no-compute route is acceptable:

```sh
cd prosper
PROSPER_GUEST_FS=1 \
PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_NO_COMPUTE=1 \
PROSPER_AVP_SYNTH_FRAMES=120 \
PROSPER_PAD_SCRIPT=@scripts/astrobot/reach-first-level.pad \
PROSPER_PAD_SCRIPT_LOG=1 \
PROSPER_DBG=1 \
  timeout 180 ./build-linux/boot_trace /path/to/PPSA21564-app0 \
  2>&1 | tee /tmp/astro-title-diag.log
```

Bound or filter debug logging. Prove the route reaches `title_controller_ship`, PC 221 disappears, and
record the **next** exact `[recompile-reject]` PC/opcode. A successful fix to #1054 is not the final goal
if another stage/op remains.

### 3. Run the real Linux path

Final correctness must remove the diagnostic shortcuts:

- omit `PROSPER_NO_COMPUTE`
- omit `PROSPER_AVP_SYNTH_FRAMES` on a host with working VA-API decode
- keep `PROSPER_RENDER_SCALE=1`
- keep `PROSPER_RENDER_EVERY=1`
- do not use a timed sampling skip for acceptance

Start with headless `screenshot`, then the Linux SDL app. The exact CLI options may evolve; consult
`tools/screenshot/README.md` and `scripts/astrobot/README.md` on current master.

### 4. Linux visual acceptance

A genuine Linux headless success requires all of the following in one unmodified route:

- expected guest checkpoint markers (title, main menu, or first level as targeted)
- real renderer enabled and guest compute enabled
- real Linux video decode where the captured phase depends on video
- no forced guest-state patch or synthetic visual substitute
- saved PNG from the normal screenshot frontend
- manifest reports advancing source/presents
- `nonblack_rgb_pixels > 0`
- more than one distinct RGB color
- visible title/scene content rather than a diagnostic clear/test pattern
- no exercised shader/GPU operation left silently skipped

Store representative images in the user's normal Downloads folder during iteration. When the draft PR
becomes a real game-progress PR, attach representative direct screenshots and caption title, platform,
frontend, route, and checkpoint as required by `CLAUDE.md`.

### 5. Regression checks after renderer changes

At minimum:

```sh
ctest --test-dir build-linux --output-on-failure
python3 tools/snapshot/snapshot.py check
```

Run all configured known-title guards (Messenger, Evergate, Dead Cells, Blasphemous II), inspect their
retained images, and record exact counts. Use `tools/verify-pr.ps1 renderer` only from a clean pushed
head if the native Linux host can run PowerShell; otherwise perform and document equivalent commands.

### 6. Cross-platform sequence after Linux is genuinely nonblack

Only after the normal Linux headless result is real:

1. Linux SDL app at the same checkpoint.
2. Windows headless with the normal Windows route.
3. Windows app with Media Foundation/DXVA, synthetic fallback unset.
4. Compare manifests and screenshots, then isolate frontend-only differences from shared renderer
   failures.

Windows acceptance should log `decoder=hardware DXVA MFT` for the real video path. The existing Windows
title screenshot is historical evidence, not proof that the current merged renderer matches Linux.

## Diagnostic caveats and rejected shortcuts

- A synthetic AVPlayer frame is useful for reaching later guest code but is not video-decode evidence.
- `PROSPER_NO_COMPUTE=1` is useful for shader/control-flow diagnostics but cannot be used for graphics
  acceptance; Astro has real compute producers in its composition chain.
- `PROSPER_RENDER_SCALE>1` and `PROSPER_RENDER_EVERY>1` intentionally reduce work and are not final visual
  correctness settings.
- A rising present count does not prove pixels changed. Use manifest pixel metrics.
- `source=raw_scanout` with `frame_seq=0` can still show that flips advance while no composited renderer
  frame exists. It is not nonblack graphics evidence.
- A black capture is diagnostic evidence only and must not be attached or described as progress.
- Do not make unknown GPU operations return success/no-op. The charter treats every exercised unsupported
  shader/GPU operation as the next fatal gap to implement.
- Do not hard-code shader address `0x5002af200`, table offset `0x15f0`, or Astro-specific data in
  production. Recognize the general compiler idiom with bounded evidence.
- Do not read arbitrary post-header guest memory in dynamic descriptor folding. Retain only the exact
  literal-pool span proven by the PC-relative analysis in the already bounded recompiler window.

## Likely later shader frontiers

Earlier logs have contained MIMG opcode `0x27` variants and VOP3/V_MBCNT-related failures after other
shader fixes. The current title PS itself contains MIMG opcode `0x27` near PC 1367. Recent master changes
have already implemented many operations, so do not assume an old log's order remains current. After
each fix, rerun a bounded live diagnostic and take the next exact first failure from current master.

The #1054 scalar table is the only currently proven immediate blocker. Avoid speculative implementation
of later operations until the live frontier advances.

## Handoff checklist

- [ ] Fetch/rebase against current remote master and inspect newly merged GPU work.
- [ ] Confirm #1054 and this branch still own the work; do not duplicate.
- [ ] Build and run full tests natively on Linux.
- [ ] Reproduce PC 221 on the diagnostic route.
- [ ] Implement narrowly proven PC-relative SMEM scalar-table folding.
- [ ] Add execution, span/cache, and fail-closed negative tests.
- [ ] Demonstrate negative control.
- [ ] Confirm PC 221 is gone and record the next live frontier.
- [ ] Continue implementing every exercised GPU gap until real Linux pixels appear.
- [ ] Run all known-title snapshot guards after each render-affecting change.
- [ ] Capture genuine nonblack Linux headless PNGs and manifest evidence with normal settings.
- [ ] Validate the Linux app.
- [ ] Cross-check Windows headless and Windows app; fix frontend discrepancies.
- [ ] Add direct screenshots to the progress PR when there is visible progress.
- [ ] Keep the draft PR owned through verification and merge; do not leave a straggler.

## Definition of done for the broader goal

Do not close the broader visual task merely because #1054 recompiles. Completion requires current,
direct evidence of:

1. genuine nonblack Astro Bot graphics in Linux headless from the normal renderer;
2. raw screenshot/manifest evidence stored and inspected;
3. equivalent Linux app output;
4. Windows headless/app cross-checks;
5. any frontend discrepancy identified and fixed or explicitly tracked with exact evidence;
6. applicable tests, all-title renderer guards, hosted CI, screenshots, and the final PR merged.

Until those conditions hold, keep advancing the live failure frontier.

## Ruled out

One line per falsified hypothesis, the evidence that killed it, and the issue/PR. Extend this rather
than re-deriving a dead answer at full cost.

- **"A timeline submit with `dmas=0` contains no GDS counter reset."** False for timeline versions
  through v9. The timeline derived its DMA census exclusively from `GpuState::dma_copies`, the
  ordered-execution vector for address-backed copies. Immediate fills, rejected packets, and GDS
  offset destinations never entered that vector, so Astro's reset packets were invisible even when
  decoded and executed. Timeline v10 adds an execution-neutral raw `DMA_DATA` journal with the exact
  PM4 order/operands/selectors plus an uncapped original count and explicit truncation flag. Older
  retained timelines cannot answer reset-versus-dispatch ordering; recapture that question with v10.

- **"Astro Bot's unbounded indirect dispatch (#1742) is caused by the dropped GDS counter resets."**
  False, and measured rather than argued. The guest does reset GDS counters with `DMA_DATA` packets
  whose destination selector names a GDS offset, and prosper *was* discarding every one of them —
  that is a real defect, fixed in #1750 / PR #1751, restoring **343 writes per routed run where 0
  landed before**. It changes nothing here: with the lever demonstrably moved, `groups_x` still runs
  1, 1, 2, 3, 4, 5, 7, 15, 6711, 26783, 40167, 66927. Whatever accumulates that count is elsewhere.
- **"The count is something prosper derives or inflates."** False. `0x500571000` is an *indirect*
  dispatch; `resolve_indirect_dispatch_arguments` reads the value verbatim from guest memory at
  `0x5074063c0` (an address stable across runs), and that memory already holds the growing series.
- **"A CP-DMA / WRITE_DATA writes the count."** False. `PROSPER_DMA_WATCH_DST=0x5074063c0` (on master)
  over a routed run reports **zero** hits. The sole writer is compute program `0x5006eac00` — every
  write `compute-buffer`, nothing else — established with `PROSPER_PROVENANCE_ADDR`, which is **added
  in PR #1752 and is not on master**; until that lands, reproduce this with that branch.
- **"#1743's 87,551 failing dispatches are 87,551 independent failures."** False. They are one
  `VK_ERROR_DEVICE_LOST` and its casualties: for `0x5006eac00`, 329 successes at submits 1..2341 and
  249 failures at 2349..4333, with **zero successes after the first failure**. The device is lost once
  and never recovers. Reproduce the failure reason on master with
  `PROSPER_COMPUTELOG_CODE=0x5006eac00`, which un-gags the `if (trace)` failure paths.
- **"The device loss has some cause other than the dispatch size."** False, and this is what makes
  #1742 upstream of #1743. Capping the workgroup count prevents it outright: unclamped, the device is
  lost at ~submit 2345 after 329 successes; with the cap the same route reaches submit 3373 with
  **458 successes and zero device losses**. The cap is `PROSPER_MAX_DISPATCH_GROUPS=N`, also **added in
  PR #1752 and not on master** — it is a diagnostic that computes wrong results and must never be used
  as a fix.
