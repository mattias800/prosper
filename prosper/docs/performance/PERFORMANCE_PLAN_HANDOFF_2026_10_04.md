# Cross-title performance plan: recovery checkpoint, 2026-10-04

This is a **draft recovery handoff**, prepared at the owner's request with approximately 4% of
Codex usage remaining. It preserves the original [#3873 plan](https://github.com/mattias800/prosper/issues/3873)
and the smallest next steps. It does not enable async submission, increase a cache default, qualify
a new app, or claim the remaining tests ran. Read the issue comments chronologically; later
wrap-up comments supersede earlier proposals. Check current remote main and tracker comments
again before resuming, because this document deliberately records a dated state.

## Where the original plan stands

| Original item | Checkpoint | Work remaining |
| --- | --- | --- |
| [#3948 async submission](https://github.com/mattias800/prosper/issues/3948) | Measurement/Stage 0 and opt-in Stage 2 are merged. **Default remains OFF.** Historical GTA collapse investigation, normal-app A/B observations and apparatus limitations are documented. | Qualify an actual latest-main normal app; finish the original fourteen-title OFF/ON visual inventory and the larger repeated current-cohort GTA/Sonic A/B. Preserve rendering holds and refuse unsupported default claims. |
| [#3951 raw x4/x8 scalar backing](https://github.com/mattias800/prosper/issues/3951) | Original backing work landed through #3987, #4104 and #4270. Broader original scalar-buffer/Wave64 integration #4374 also merged on October 4. | Do not restart the completed original backing task. Normal-game effects of the newly merged integration still need current-app evidence; Windows/Kena work is a separate lane. |
| [#3892 renderer state/image-branch work](https://github.com/mattias800/prosper/issues/3892) | Original registration-owned state and image-branch work completed. | No unfinished original unit is claimed here. Consult later issue comments for separate follow-ups. |
| [#3891 performance alarms](https://github.com/mattias800/prosper/issues/3891) | Original accepted alarm proposals and review nits completed. | Read raw alarms and exit summaries on every new run; propose a new rule only for an actual uncovered engine problem. |
| GTA render-target entry ceiling | Requested six-arm comparison completed, native frames inspected, raw evidence archived, result posted, and #4381 merged the narrow falsification into GTA status. **Count default remains 256.** | No accepted count-lift performance win. Startup/other-title benefits remain open; this completed cohort does not become a current-main measurement after later code changes. |

Astro Bot is explicitly **after the original plan**, not a parallel priority. The owner supplied a
title-screen oracle in [tracker #1809](https://github.com/mattias800/prosper/issues/1809).
Read that tracker's comments and retrieve the oracle before investigating; the optional rendering
work has not been completed by this checkpoint.

## The immediate blocker was fixture construction, and its correction is merged

Latest remote main observed after the near-ready fix was
**ee9dcb1e1e2c695426acb25575e7c6c320971d1d**. Its parent integration baseline was
**3fe269ef0d035a46baee750af30b6a0004d4f7b0**, the squash of #4374: 70 files changed in the
original graphics-source/scalar-bank path. This is a production-code delta from the earlier
performance cohorts, not a documentation-only revision update.

The normal GCC Release app built on 3fe269ef. Nine selected CPU test targets registered exactly
70 cases. The actual first results were **42 PASS / 28 FAIL / 0 SKIP**, CTest exit **8**; the
build exited **0**. Certification and app qualification stopped. All 28 failures occurred during
fixture setup before the intended scalar-bank/ordered-read contracts could be exercised:

- One original-stage-effects case and twelve scalar-bank cases stopped at shader registration.
- Fifteen ordered-read cases stopped at an assertion that a heap object must be above 4 GiB.
- Non-PIE Linux binaries can allocate small heap objects below 4 GiB. Supplying those absolute
  values in SDK-relocatable fields is not a valid substitute for a genuine self-relative blob.

[#4382](https://github.com/mattias800/prosper/pull/4382) corrected exactly two fixture paths with
header-first contiguous blobs, forward field-relative pointers, process-lifetime retained owners,
and exact relocated user/register/code/PGM/header checks. It changed no production code, shared
ps-pull layout, CMake flags, dependency, or semantic ownership/refusal assertion. The three affected
targets contain 38 cases; the other 32 selected cases remained in the same run.

The actual corrected results on author head **49d6d6874746db8bf84edaaae40a550bb8f9b4f1** were
**70 PASS / 0 FAIL / 0 SKIP**, with build/discovery/CTest exits **[0, 0, 0]**. Only those three
targets rebuilt; build wall time was 20.96 s and CTest command wall time 2.11 s. Root independently
counted all 70 unique passed rows; the raw log SHA256 is
**44a504ddc355283fd3b5770e96a495cd54d85917a0eb388e1fd293b2d6549bf3**.
The failed cohort and immutable failed test binaries were retained, not overwritten or rerun.

One [registered independent APPROVED review](https://github.com/mattias800/prosper/pull/4382#pullrequestreview-5405489407)
bound the exact author head. The unmodified merge gate actually returned **1**, with 10 passing,
0 failing, 4 pending and 1 skipped check. The active owner CI-wait waiver permitted the protected
merge; those pending checks were not reported green. Squash **ee9dcb1e** has the same Git tree as
the reviewed author head: **de71ee71735dfcfa29565622c0d82b3904776ec2**. Landed fixture bytes were
independently checked and the author/reviewer worktrees were removed.

**This does not qualify a current app.** The retained new app still embeds 3fe269ef, while the
producing worktree/build was last frozen at author head 49d6d687. The latter build contains
corrected test binaries and a refreshed revision archive, but its app was deliberately not relinked
during fixture validation. Never label it ee9dcb1e or launch a mismatched source/app pair.

## Retained app/source cohorts

| Actual app revision | Immutable app SHA256 | Interpretation |
| --- | --- | --- |
| e291f7060042d4b5b36f8eed52ea228f6dbf9081 | d43a39b78d9282072482847247fb3b15d1251d2d1493eda5093111453dd95528 | Independently qualified normal app used for the partial Messenger OFF run and another lane's historical Silent Hill compatibility observations. Historical after #4374. |
| 3fe269ef0d035a46baee750af30b6a0004d4f7b0 | 058abdfe6a6c539c75386c0815e97c29f5790e59794f40c4b402eb6c23631df1 | Built normal app retained after the failed initial 70-case qualification. Full ELF source identity is 12c688a6d0fc7aa7b80d9368a08d3d439c4ea7994ded61618b3fa28a211bc84a. **Not accepted as a completed app qualification.** |
| 525cb71c0c7dda1d7404184b1c1a093d14641a28 | d6823e50040a29a8997cf3e6848190c2ca481abd4a46e0fb4e32812408d314ef | Qualified normal app for the completed Sonic pair and GTA count matrix. Its CPU correctness cohort was 26 PASS / 0 SKIP. Historical and not pooled with 3fe. |

Earlier shipping-986 evidence also remains archived. Retained apps keep the basename
**prosper-app**. Installed provider inventory and ldd output do not establish which driver was
actually loaded by a future game; obtain that runtime proof again.

## What the completed measurements establish

### GTA target-cache count: complete, no default lift supported

Read the full [#3873 result](https://github.com/mattias800/prosper/issues/3873#issuecomment-5978333004)
and `prosper/docs/games/GTA5_STATUS.md` before proposing another count experiment. Six 840-second
normal-app arms used **256, 512, 1024, 1024, 512, 256** entries, deferred waits OFF, texture cap
1024 MiB, simulated heap 8192 MiB, and target-byte budget 2048 MiB. Each owned app/runner ended
with actual exit 0 and drained. Root inspected all 18 native menu/intermediate/bank frames.

The preregistered common bank population was 600–780 s; primary unfiltered and capture-free
populations were identical in all six arms. Capture intervals were independently admitted from
actual native events and bounded file-completion proxies; unknown write-start timing was retained.

| Ordered arm | Entry setting | Guest-flip / host-present events per second |
| --- | --- | --- |
| A1 | 256 | 2.404111 |
| A2 | 512 | 2.533779 |
| A3 | 1024 | 2.051207 |
| A4 | 1024 | 2.363218 |
| A5 | 512 | 2.578634 |
| A6 | 256 | 2.511207 |

These are event rates, **not proven fresh-rendered-frame FPS**. Every selected bank window had
zero target evictions and zero shader/pipeline creation; three fragment draw losses plus one
compute skip per flip remained. Higher counts retained more memory but did not restore the
historical 9–10 event/s behavior in this cohort. The 1024 setting encountered the approximately
2048 MiB byte ceiling before the entry ceiling. Unequal I/O pressure, two observations per count,
and unproved equal guest state prevent a causal benefit/regression claim.

The narrow hypothesis **ongoing warm target eviction churn explains this low bank rate** is ruled
out for the selected windows. Cold-start/other-title benefits remain open. Shutdown-gate refusal
messages were retained; source inspection identified a pre-driver VK_ERROR_DEVICE_LOST sentinel,
so they were not promoted into an established hardware device-loss event. #4381 landed this
falsification and its limits in the GTA status document.

### Async GTA and Sonic: useful observations, no accepted default-on win

- Historical GTA's collapse investigation reproduced approximately 3.433 events/s in an OFF arm
  under compiler load, versus quiet OFF 9.378 / ON 9.201. This supports host pressure as a
  mechanism, but does not prove the exact cause of the old 3.6-FPS interval on #3948.
- The earlier shipping-986 GTA OFF/ON/ON/OFF bank observations were 2.519318, 2.580901,
  2.445593 and 2.591568 events/s. The two ordered comparisons disagreed; I/O was asymmetric,
  and there was no accepted async performance win. See the
  [actual cohort](https://github.com/mattias800/prosper/issues/3948#issuecomment-5976895616).
- Sonic's shipping-986 380-second pair reached menu captures only. Gameplay populations were
  NO_DATA; mixed whole-run rates cannot be renamed gameplay throughput.
- The later shipping-525 1600-second Sonic OFF/ON pair reached menu and gameplay HUD/clock
  captures. The world remained unrecognizable in both arms: dark upper region and pale lower
  region/glowing lines. Async ON alone repairing that world defect was falsified; this was not
  established as an ON-only regression. Stationary event rates 1.618456 / 1.601637 and moving
  rates 1.509879 / 1.533889, asymmetric pressure, one ordered pair and the rendering HOLD do
  not support a default change. See the
  [full Sonic result](https://github.com/mattias800/prosper/issues/3948#issuecomment-5977560549)
  and `prosper/docs/games/SONIC_FRONTIERS_STATUS.md` (#4380).

Never pool these different source/app cohorts. A correctness change that exposes a GTA problem
is not automatically wrong; preserve the real defect and distinguish correctness cost, existing
bug exposure, and a demonstrated new regression. Stage 1 pipelined compute remains deferred:
the old measured ceiling was at most about 5% of wall time for a large implementation cost.

## Messenger partial run and reviewed visibility-helper successor

The first e291 Messenger OFF compatibility arm stopped at **265.881 s** when a KWin Match
response exceeded the helper's 8192-character stdout allowance. The app shut down cleanly:
app exit 0, runner/launcher exit 1, all owned processes drained. This was an apparatus failure,
not a demonstrated guest fault or completed guard.

Root inspected requested/actual native captures 500/500, 2000/2001 and 15000/15000: title prompt,
selectable Play/Options menu, and full-colour village/player/world/health/flame HUD. Requested
18000 was unreached; the required pre-cutoff witness and second gameplay image were missing.
The owner played during this OFF arm. Exact actions/times/state are unknown, so it is interactive
compatibility evidence with script plus owner input, not a controlled performance arm. Its save
files, native frames, raw log and failure receipt were durably archived.

A later private reproduction recorded a 122017-character typed Match response with a foreign
game's irrelevant icon-data pixels. Historic failing stdout bytes were not saved. Independent
validation used a semantically equivalent 121993-character reconstruction, not an exact-byte
replay of the failed query.

The independently APPROVED private successor changes **only** WindowsRunner.Match's stdout
allowance to 4194304 Python characters. Other stdout/stderr bounds, at-most-16 typed matches,
0.4-second per-query / 2-second phase budgets, exact owned PID/birth/executable/UUID checks,
normal/non-minimized/current-desktop/viewport checks and cleanup remain unchanged. This is a
post-capture character limit, not a hard streaming byte-allocation limit. Fifteen pure controls
passed; source review alone is not a runtime grant.

| Private helper | Exact approved SHA256 |
| --- | --- |
| launch-visible-compatibility.py | fb8763ebeb5c4854a3df1d653f88989ee83bf3c4bfd7a37db095f6d53883450a |
| compatibility-visible-run.py | f1dd0f792030f1dd829111a919d0bf07e9d70b562d5b3a7d178f9490ba1deb33 |
| async_launch_proof.py | 352e326b8ec8a4284201746eb9a36626aeb5e4a1544a7414a6b7c76a97ad2466 |
| wayland_window_witness.py | 476ccdeeeb46cdb312053a3f229613df2634f34a17cceaf1c33db5151ff3fde4 |

Preserve the original helper and failed run unchanged. Fresh complete OFF and ON must use the
same reviewed successor, distinct unused output directories, fresh saves/caches, and a fresh
native desktop binding. Do not rerun the known-overflow original helper.

## Original fourteen-title visual inventory: pending current-app qualification

Exactly thirteen automatic gameplay profiles plus manual GRIS belong to this original inventory.
Do not add duplicate Evergate-title/Blue-Prince-title companion profiles or Terminator's boot-only
profile and then claim fifteen gameplay titles. Requested flip ordinals below are **calibration
proposals**; they do not establish the current render phase. Later requested captures may coalesce
to a different actual flip; retain actuals.jsonl and actual_fX_atY filenames.

| Title / ID | Profile | Normal route under prosper/scripts | Proposed seconds / requested flips |
| --- | --- | --- | --- |
| The Messenger / PPSA24651 | messenger-scene | messenger/reach-first-level.pad | 320 / 500,2000,15000,18000 |
| Evergate / PPSA01885 | evergate-gameplay | evergate/reach-first-gameplay-realtime.pad | 230 / 500,6000,9000,12000 |
| Dead Cells / PPSA15552 | dead-cells-gameplay | dead-cells/reach-first-gameplay-full-render.pad | 320 / 500,1000,12000,15000 |
| Blasphemous 2 / PPSA13579 | blasphemous2-gameplay | blasphemous2/reach-first-gameplay.pad | 540 / 500,2000,4000,12000,24000,27000 |
| Blue Prince / PPSA25009 | blue-prince-hall | none | 810 / 500,2000,4500,15000,18000 |
| Alex Kidd in Miracle World DX / PPSA02664 | alexkidd-gameplay | ppsa02664/reach-first-gameplay.pad | 150 / 500,1000,2000,4500,6000 |
| Rugrats: Adventures in Gameland / PPSA23396 | rugrats-gameplay | rugrats/reach-gameplay.pad | 220 / 2000,6000,9000 |
| Greak: Memories of Azur / PPSA02849 | greak-gameplay | greak/reach-gameplay.pad | 250 / 1000,9000,12000 |
| New Joe & Mac: Caveman Ninja / PPSA02801 | joe-mac-gameplay | joe-mac/reach-gameplay.pad | 190 / 2000,3000,4500,6000,9000 |
| Asterix & Obelix: Slap Them All! / PPSA08576 | asterix-gameplay | asterix/reach-gameplay.pad | 190 / 2000,3000,4500,6000,9000 |
| Summer Sports Games / PPSA03416 | summer-sports-gameplay | summer-sports/reach-gameplay.pad | 140 / 2000,3000,4500 |
| Space Adventure Cobra — The Awakening / PPSA17337 | cobra-gameplay | cobra/reach-title-or-gameplay.pad | 210 / 1500,3000,4500,6000,9000 |
| Worms Armageddon: Anniversary Edition / PPSA20052 | worms-armageddon-gameplay | worms-armageddon/reach-training-gameplay.pad | 190 / 500,2000,4000,6000,8000 |
| GRIS / PPSA09804 | gris-gameplay | gris/reach-first-gameplay.pad | 250 / 500,2000,6000,12000 |

All are proposed OFF then ON, full rendering scale/cadence 1, target count 256, texture cap
1024 MiB, simulated heap 8192 MiB and target-byte budget 2048 MiB. Blue Prince uses the retained
IME-autokey route; no-autokey supplementation is a different visual cohort. GRIS uses the
documented manual normal route; do not substitute the old automated gris-gameplay guard.
Compatibility captures disable automatic F8. Performance capture runs have a separate protocol.

For each arm, inspect a real selectable menu and at least two actual gameplay images, with HUD
where the game has one. GRIS has no conventional HUD. Logos, narration, movies, results, game-over
screens and tutorial-only images do not establish the intended playable phase. Retain a wrong or
unreached checkpoint as NO_DATA/HOLD and calibrate the smallest needed selector/route adjustment;
do not silently extend a deadline or treat a historical ordinal as a current oracle. Cobra's
historical sky seam/HUD/partial-model rendering HOLD remains a hold until current images resolve it.

## Smallest next steps after restoring quota

1. Read CLAUDE.md, LOCAL.md, root/nested AGENTS.md, CONTRIBUTING.md and this handoff; inspect
   current #3873/#3948 comments. Fetch current remote main **in your own worktree** and run
   `python3 prosper/tools/session_start.py`. Do not reset the mostly-unused shared checkout.
2. Verify the private durable capsule and extract into a new private directory, never over an
   active worktree. Its local recovery guide supplies exact paths, archive hashes, manifests,
   command plans, grant schemas and owned PID/birth receipts. Git/source snapshots and app bytes
   are historical; their original absolute paths are not proof of current identity.
3. Finish **one actual latest-main normal app qualification**. The producing tree was frozen at
   49d6; the last remote main ee9d has identical tree bytes. If the only newer changes are this
   fixture correction/documentation, inspect that delta and carry the exact 70-case evidence;
   do not blindly rerun it. Refresh the real app's embedded revision, run meaningful revision
   verification/certification, and independently verify full ELF revision/source identity,
   configuration, actual compile/input closure and immutable app bytes. New relevant runtime
   changes require appropriate validation instead of the old carry-forward assumption.
4. Keep normal GCC Release, real SDL audio/pad, FFmpeg/VAAPI and diagnostics-OFF settings. Do not
   disable the newly merged scalar-bank route, use a synthetic backend, or turn a failed/pending
   result into a passing manifest. Preserve the initial failed 70-case cohort and old apps.
5. Qualify fresh native desktop/env/window binding, bind the exact reviewed helper hashes and
   actual app/source, and start fresh Messenger OFF/ON. The 3fe preparation contains inert
   candidate data only: no app qualification, executable grant or completed current guard.
6. Continue the fourteen-title pairs above, recording native menu/gameplay/HUD judgments,
   unsupported draws/dispatches, deferred-path exercise, actual alarms, missing phases and owned
   cleanup. Treat ON with no eligible deferred work as unproved exercise, not a benefit.
7. Announce and obtain the root-coordinated quiet measurement interval before the larger repeated
   GTA/Sonic A/B (the pending campaign intended three repeated pairs per title). Use the canonical
   measurement wrapper, same actual latest-qualified app for both arms, immediate presentation,
   fresh isolated run state and simulated budgets. Keep ordered pairs and source cohorts separate.
   Inspect native menu/gameplay frames before admitting populations. Do not turn Sonic's known
   rendering HOLD into a healthy performance claim or repeatedly escalate a failed route.
8. Compare preregistered phase/capture-free populations, raw event counters, losses, device-time
   coverage, memory and host pressure. A console FPS string alone is insufficient; distinguish
   guest flips, host presentations and proven fresh delivered frames. Sum overlapping device
   observations only within their actual documented contract, not as a budget partition.
9. Keep deferred submission default OFF unless both compatibility and meaningful repeated evidence
   support enabling it. Record falsified hypotheses in the relevant Ruled out section, put the
   measured conclusion/limits in the issues, and land each justified code/default unit in its own
   independently reviewed PR. Do not enable the switch merely to close the plan.
10. Only after the original plan reaches a justified checkpoint, resume optional Astro Bot using
    tracker #1809's hardware oracle. The broader performance roadmap can wait for restored quota.

## Scheduling, merge and cleanup contracts on resume

- Every game run shows the real Prosper window. Serialize **performance measurements** through
  the machine-local canonical measure.sh. Compatibility runs may coexist after the orchestrator
  explicitly releases a measurement interval; the latest interval was released after the GTA
  count matrix. There is no active root-owned game at this checkpoint.
- Announced quiet measurements park other GPU/game/Vulkan work and heavy builds/tests. Never kill
  foreign work. Signal only a launched PID whose recorded birth identity still matches; verify
  `ps -eo pid,etime,comm | grep prosper-app` after every run. Never rename prosper-app for A/B.
- Read every raw perf-alarm line and exit summary. Missing data, unsupported rendering losses,
  shutdown sentinel refusals and unverified quiescence remain visible. The compositor-query
  allowance bug is not an uncovered engine alarm rule.
- Current owner policy: independent review on the exact head and resolved blockers remain
  mandatory; run the unmodified pr_merge_gate.py and retain actual results. The owner currently
  waives waiting for CI. Check the actual remote/fork tip and protect the merge with the expected
  head. Never report waived failures/pending checks green. Never merge any draft; external authors
  must publish their own drafts. No Claude-Session trailers or badges.
- Watch published yuriolive PRs for a new head and assign one independent security/correctness
  review. At the checkpoint #4331 and #4189 still have their previously rejected unchanged heads;
  #4323 is a draft. Do not duplicate unchanged reviews/pings. #4189's author said revisions are
  planned; #4331 already received one inactivity check-in. Leave them open.
- The incoming author, independent reviewer and builder completed #4382 and are parked. Other
  CPU/Outer-Wilds lanes were already parked; Silent Hill was asked to finish its existing historical
  result handoff and park. No new large jobs, game arms or app qualification were started after
  the owner's quota-checkpoint request.
- Keep builds on the RAM drive while working, monitor available RAM, temporary storage and container quota, and
  remove only owned obsolete builds/worktrees **after verified durable recovery**. Shared SDL
  build/source dependencies have protected consumers; never delete them as blanket tmp cleanup.
  The private guide names the protected paths and describes restored build-cache limitations.

## Durable evidence map

Machine-local paths and desktop/process details stay in the private recovery capsule. Publicly
share only this document, PRs/issues and portable commands/placeholders; do not commit game dumps,
raw machine captions, build directories or private evidence archives.

| Durable historical archive | Verified archive SHA256 | Scope |
| --- | --- | --- |
| root-shipping986-async-checkpoint-20261004-v1 | e636eeeabe140a5cdb2120005f1e48b83c0733ef17d43d48a9302926a345328d | Earlier GTA ABBA, menu-only Sonic observations and their qualification/receipts; 11209 members verified. |
| root-shipping525-Sonic-checkpoint-20261004-v1 | 776a40f5ff55081c700f369a9f857d21980ec6ed88c453ed6d240a0d1c8795a0 | Qualified 525 source/app and Sonic pair; 5657 members verified. |
| root-shipping525-GTA-count-checkpoint-20261004-v1 | e6288501c8ad6d67790987a6543446175284f51b47a8aeed0e52c72c55af5c16 | Completed six-arm count data, all native frames and root derivations; 9336 members verified. |
| root-e291-Messenger-partial-helper-checkpoint-20261004-v1 | 7e10533ff8717c60197920c120b625169dce504640f2cf38fd4c7806445e39c5 | e291 qualification/source, partial Messenger/save files, helper successor review, #4381 receipts; 2918 members verified. |
| root-shipping3fe-failed-CPU70-checkpoint-20261004-v1 | 1d3ee054ae6d628ba8c237dbe6d549a2d06ad067655f359db68b7cad81b43f3d | 3fe source, failed 70-case raw results, immutable built app and three failed test binaries; 2538 members verified. |

The final quota-recovery capsule adds corrected 70-case evidence, #4382 review/merge/cleanup
receipts, this draft, frozen producing-source/build recovery material, approved compatibility
helpers, inert next-arm plans, the local state ledger and a detailed recovery guide. Its actual
manifest/hash is stored beside the private archive after completion; this draft does not invent
an archive completion before that verification happens. Installed system/compiler dependencies
must be checked against their retained pins on the restored machine; archives are not a claim of
a fully portable environment or runtime driver qualification.
