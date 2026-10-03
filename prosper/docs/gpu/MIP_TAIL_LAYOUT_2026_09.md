# Separable CPU mip-tail layout conversion

Tracking: [#3407](https://github.com/mattias800/prosper/issues/3407).

The remaining six-layer Float16x4 mode-27 writebacks in Sonic were **packed mip tails**.
The scalar control now records their tail coordinates explicitly: all 266 targeted writeback rows
have `in-tail=1`, with each physical layer sharing a 64 KiB block. Broadening ordinary native-array
GPU retile would miss this path, which deliberately excludes shared mip-tail blocks.

The selected-level CPU walker previously evaluated both coordinate contributions to every address
bit for every texel. XOR separability lets it evaluate x once per column and y once per row instead.
The change benefits both mip-tail reads and writes across supported formats and titles.

## Contract and regression guards

`sw64kb_level_copy` uses a bounded 256-entry `uint16_t` stack table, without allocation or retained
state. Wider callers retain the scalar equation. Global coordinates, unsigned wrap, block selection,
whole-texel bounds checks and copy order are unchanged. Writes preserve neighboring mip bytes;
truncated reads keep the existing zero-fill behavior. Modes 9/24/27 and 1/2/4/8/16-byte elements
use the same existing patterns. No new Vulkan feature, cache budget or deferred guest publication
is involved. `PROSPER_NO_SEPARABLE_MIP_TAIL=1` restores the original equation in both directions.

`test_mip_tail` compares selected-level writes and reads with the unchanged full-surface placement
oracle, checking complete poisoned destination/guard buffers. Seven pipe-count processes each
compare **56,937,558 actual bytes** across optimized and scalar arms. Cases cover nonzero origins,
truncated/empty backing, neighboring mip preservation, widths 255/256/257, and unsigned coordinate
wrap. The scalar switch needs a fresh process because its value is cached. Python registration is
optional; the direct C++ oracle still runs without Python. Windows emits binary output and accepts
CRLF in diagnostic markers.

A deliberately corrupted oracle is rejected. More substantially, compiling a production mutant
with the new x-offset entries replaced by zero produces **273 placement failures**. Disabling only
the optimization correctly preserves output; its performance consequence is measured below rather
than enforced with a noisy CI timing threshold.

Validation: **508/508** full tests passed before the Windows-only harness adjustments; the final
capture head passed **8/8** focused tests. A strict scan passed **4/4** selected tests (three Vulkan
execution tests and the CPU oracle) with zero validation messages, after positive layer-insertion
and deliberate write-after-write hazard controls. Standalone ASan/UBSan checks passed.

Run these commands in the project's complete C++20/CMake/Ninja toolchain with Vulkan and SPIR-V
tools available; Python and Pillow are also needed for the capture helpers. `<DUMP_ROOT>` must
contain `PPSA24651-app0` for the recorded full test configuration, plus the Sonic/GTA dumps for
game captures. A build without those dependencies/fixtures does not reproduce the 508-test claim.

```sh
export PROSPER_GAME_ROOT=<DUMP_ROOT>
cmake -S prosper -B prosper/build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release -DPROSPER_APP=ON \
  -DGAME_DUMP="$PROSPER_GAME_ROOT/PPSA24651-app0"
cmake --build prosper/build-linux -j8
ctest --test-dir prosper/build-linux --no-tests=error --output-on-failure
python3 prosper/tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux \
  --sync --allowlist /dev/null --ctest-arg=-R \
  '--ctest-arg=^(gpu_retile_array16|live_compute_host_read_barrier|storage_readonly|gpu_mip_tail_oracle)$' \
  --ctest-arg=--no-tests=error
```

## Isolated operation measurement

Five alternating scalar/separable pairs use the same Release `test_mip_tail` executable. Each arm
performs 3,000 tile/read pairs of a 64×64 image with eight-byte elements in a 64 KiB mode-27 block.
Median elapsed time is **829.255 ms scalar / 55.106 ms separable**. This is an isolated operation,
not a game FPS measurement. Original per-arm results are in `microbench.json`.

```sh
PROSPER_NO_SEPARABLE_MIP_TAIL=1 prosper/build-linux/test_mip_tail --bench
prosper/build-linux/test_mip_tail --bench
```

## Comparable game captures

All four runs use source `f291cb301f598c8926ac4c192431ab4c4be2744e`, based on main
`87451e9144cea9df3441a7b75b9d1dccc180bf91`, and app SHA-256
`ab285c52754f133a89bc2b1267e0cae37aceca5ae235bd568367a3a1facb429e`.
Later documentation changes do not change this measured executable.

Normal Linux `prosper-app`, windowed immediate presentation, native/default rendering, fresh save
and cache directories per run. Sonic uses `scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad`;
GTA uses `scripts/gta5/reach-performance-story.pad`. The sole normalized environment difference
within each pair is `PROSPER_NO_SEPARABLE_MIP_TAIL=1` in the scalar arm. Capture source, executable,
route and helper hashes stay fixed. No foreign build/game/profiler workload was observed by the
sampled census. F8 starts at 300 seconds; F9 follows at 330 seconds for Sonic and 310 for GTA;
stop requests follow 380/360 seconds. Every run exits 0 with complete F8/F9 records, no dropped
F8 details, and successful logged compute phases.

| Observation | Sonic scalar | Sonic separable | GTA scalar | GTA separable |
| --- | ---: | ---: | ---: | ---: |
| Measured F8 seconds | 5.015864 | 5.009935 | 5.018213 | 5.020185 |
| Host presentations / s | 7.376595 | 7.784532 | 6.177498 | 6.175071 |
| Guest flips / s | 7.376595 | 7.784532 | 6.177498 | 6.175071 |
| Mean host presentation interval, ms | 135.564 | 128.460 | 161.878 | 161.941 |
| Average CPU cores | 1.445 | 1.456 | 1.477 | 1.471 |
| RSS range, MiB | 4494.8–4495.0 | 4570.3–4570.4 | 9158.6–9158.9 | 9523.9–9524.1 |
| Renderer total, ms | 1963.339 | 1947.360 | 2661.489 | 2676.842 |
| Frontend texture preparation, ms | 361.974 | 297.671 | 693.228 | 682.020 |
| Enclosing resource preparation, ms | 784.204 | 735.974 | 1444.131 | 1441.517 |
| Renderer wait / device time, ms | 635.059 / 516.383 | 652.539 / 531.645 | 539.692 / 350.489 | 543.907 / 345.535 |
| Compute total, ms | 1612.679 | 1638.014 | 1188.949 | 1215.735 |
| Compute device time, ms | 360.681 | 386.689 | 323.849 | 334.321 |

Rates use each capture's actual post-trigger sample span. Newly rendered frames are **unavailable**:
host presentations and guest flips do not prove fresh rendered frames. The interval is the reciprocal
mean rate, not an individual frame-time percentile. Nested CPU/GPU timers must not be added together;
renderer record counts are not queue-submission counts. RSS is reported as observed, not as a proven
memory improvement or a physical Vulkan residency cap.

### Sonic target

Program `0xb1f346a293f11e9a` has **37/39** completed F8 dispatch records in the scalar/separable arms.
Its mean complete dispatch cost falls **4.055611 → 2.206192 ms**, a **45.6% reduction in this pair**.
This includes setup, waits, writeback and cleanup. Mean setup falls 1.209189 → 0.423812 ms; mean
writeback falls 1.445970 → 0.258369 ms. Mean shader GPU time rises 0.965882 → 1.089410 ms;
the complete cost already includes that increase.

The separately gated image trace has **266/273** rows across seven matching tail bindings.
Its layout total falls **48.344 → 3.971 ms**, or **0.181744 → 0.014546 ms per image write**.
Complete image-writeback totals are 54.252/9.282 ms. These rows describe 38/39 repeated binding
sets, whereas F8 contains 37/39 program records: the gate boundaries differ. Keep these populations
separate; do not substitute the image trace totals into the F8 phase accounting. Recorded guest
byte spans include physical shared mip blocks, not a count of bytes actually overwritten.

The same helper also serves reads, and frontend texture preparation is lower in this pair as shown
above; that enclosing timer is not isolated attribution to a single call site. Total compute time
is slightly higher while more dispatches occur. The higher host presentation rate in this one pair
is **not a stable or newly-rendered FPS gain**.

Both F9 images retain the known black world and the same HUD elements (#2790), with different timer
values (00:51.53/00:52.80) and counter colors. They are not pixel-identical replays or evidence that
the missing world renders correctly. The longstanding title-screen issue (#2206) is separate.

### GTA compatibility and profiling context

GTA has **zero mip-tail writeback rows** in either separately gated image trace. The expected bank scene, lighting and HUD remain
visible with ordinary pose changes. Presentation rate is effectively unchanged; no GTA speedup is
established. This is compatibility evidence for a shared change, not a demonstrated target-path win.

Separate 20-second profiles begin at elapsed 250 seconds and finish before 295 seconds. CPU sampling
uses `perf record -F 199 -g`; folded stacks and flame graphs are retained. Sonic's inclusive mip-walker
cycle-weight share is **1.335%/0.166%** scalar/separable, including both wrappers and the outlined
`sw64kb_level_copy` specializations. These are sampled cycle weights, not sample counts. No such
symbol was sampled in GTA; that absence alone is not proof that no mip-tail read occurred.

The supported `radeontop` GPU column averages **20.5%/15.0% Sonic** and **14.5%/12.0% GTA** across
20 observations per arm. These earlier profile windows provide context and are not the F8 timer
population. Unsupported device counters are not used, and no zero-lost-samples claim is made.

Main audio output `17/1/active` starts/ends at 0/0 and 1/1 observed shortfall calls in Sonic,
and 1/1 in both GTA arms. Callback counters advance with no new shortfalls in covering bands
60–240, 295–307 and 345–355 seconds. GTA's separate early output `18/1/active` ends at 36/39
shortfalls; it has no late coverage. These are per-output/open-generation SDL demand observations,
not physical XRUN or audible-quality measurements. Audio buffering/mixing is unchanged (#3435).

Only GTA scalar logs one `submit=-4`, after `[app] shutting down`. The existing closing gate can
return that code without calling the driver; this log cannot independently distinguish gate from
driver return. The other three runs do not log that submit error. No pre-shutdown submit=-4 is observed.

## Remaining scope and retained artifacts

#3407 remains open. The costly unchanged buffer-result scans are a separate unresolved tradeoff:
current 256 MiB residency cannot fit the large optional baselines, and the previously rejected
worker-pool/extra-baseline experiments should not be repeated without new evidence. See
[the recorded investigation](https://github.com/mattias800/prosper/issues/3407#issuecomment-5584213103).
GPU-retiled images still require synchronous guest publication; further skips need #1854's authority
contract. The enabled windows still spend **639.511 ms Sonic / 362.273 ms GTA** in compute writeback,
and **735.974/1,441.517 ms** in enclosing renderer resource preparation. These are next attribution
priorities, not projected savings. Coordinate VMA work with #3399 and keep fresh-frame lineage
(#3486) and audio (#3435) separate.

Private originals, source archive and all reproduction helpers are retained under
`<EVIDENCE_ROOT>/issue-3407-mip-tail-20260913/`. Inside its `prosper/build-linux/evidence/`:

- `tail-{off,on}-{sonic,gta}/`: run/env/census manifests, original F8/F9/BMP, PNG inspection copy,
  exact image/buffer/phase rows, per-output audio, `cpu.data`, folded stacks, SVG and GPU log.
- `tail-{sonic,gta}-comparison.json`: full standard summaries and every compute program (zero
  omitted groups), plus separately counted image-layout groups.
- `capture-identities.json`, `capture-audit.json`, `verification.json`, original test/validation
  logs, `microbench.json`, and `snapshot-{capture,child,validation,report}.py`.

Restore the archived source at the capture head and copy the helpers into its ignored
`prosper/build-linux/evidence/`. The retained capture driver freezes fresh directories, hashes,
normalizes environment differences and rejects concurrent test/build/profiler processes:

```sh
python3 prosper/build-linux/evidence/snapshot-capture.py disabled sonic tail-off
python3 prosper/build-linux/evidence/snapshot-capture.py enabled sonic tail-on
python3 prosper/build-linux/evidence/snapshot-capture.py disabled gta tail-off
python3 prosper/build-linux/evidence/snapshot-capture.py enabled gta tail-on
python3 prosper/build-linux/evidence/snapshot-report.py \
  prosper/build-linux/evidence/tail-off-sonic prosper/build-linux/evidence/tail-on-sonic
```

The retained driver contains private dump/container paths; adapt them and keep equivalent inputs
before running on another machine. For each original `.prperf`, the ordinary reader also works:
`python3 prosper/tools/perf/performance_capture_report.py <capture.prperf> --json`.
F8 SHA-256 identities follow; original BMP hashes are in `capture-identities.json`.

| Run | F8 SHA-256 |
| --- | --- |
| tail-off-sonic | `4600b2a2168edb4d47c593c75d41c1eaf3e8999c86c46ecbb62a463eb8937bd7` |
| tail-on-sonic | `c211d3243cf0fbe5172ee778e0c04bfa9b16563f51ae3e04f5291c04328be1f0` |
| tail-off-gta | `9ef2c78ad60210d73447f09298fd86b5e559fd21dc42f14fba24ddc064c32304` |
| tail-on-gta | `4d99ed25756b713f50003b9b437eaf74609dc4542ead566d8daa13322e019917` |
