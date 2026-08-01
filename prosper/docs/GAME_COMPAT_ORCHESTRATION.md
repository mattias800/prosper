# Game compatibility orchestration

This is the durable playbook and current handoff for concurrent game-compatibility work. It is written for an
orchestrating agent that delegates one bounded investigation to each subagent, integrates real progress, and keeps
the shared repository, GPU, evidence, issues, and pull requests coherent.

Read the repository-root `CLAUDE.md` before this document. `CLAUDE.md` remains authoritative for filesystem,
verification, evidence, review, privacy, and release policy. This document explains how to apply those rules when
several game agents work at once.

The current-state sections below are dated **2026-07-31**. Update them whenever an investigation materially moves,
a PR merges, or ownership changes. Do not let this document replace issue comments: issues are the durable evidence
log, while this document is the map that helps the next orchestrator find and interpret that evidence.

## Current checkpoint

- Repository: `mattias800/prosper` (renamed from `mattias800/ps5ys`).
- Remote branch: `master`.
- Exact master at this update: `ede320b6` (2026-07-31 session; eleven merges).
- Active title lanes: Astro Bot, Dragon Quest VII, The Plucky Squire, Alex Kidd, and The Pathless.
- GPU state: sharing is the default; see the scheduling rules below.

**Read the falsification record before planning anything.** The three lanes handed over on `81a9548e` each had
their assigned premise **falsified** during the 2026-07-31 session, including both items the previous handoff
called its highest-value work. Re-running them wastes a session:

| Previous "highest-value" item | Outcome |
|---|---|
| Astro op221 R11G11B10 storage lowering | **FALSIFIED.** Stored (packed R32ui) and current-raw (native typed float) take provably different backend paths yet produce byte-identical output `28b5ea9a3fbd2ca9`. |
| Dragon Quest binding-36 sampling | **EXONERATED entirely.** LUT contents, tile27 detile, DCC, 10:10:10:2 unpack, 3D coordinate wiring, dimensional opcode, dmask, swizzle, address modes, sampler filtering, and draw90's shader translation are all ruled out. |
| Plucky pc915 barrier-spanning branch | **PROVEN workgroup-uniform and merged** (#1572). |

Two lanes were added because the inherited allocation was skewed to the hardest titles while easier ground sat
idle: **Alex Kidd** (PPSA02664) and **The Pathless** (PPSA01826). Alex Kidd reached the gameplay rung in one
session. Prefer breadth when a hard lane stalls — overall library progress matters more than any single title.

### Tooling hazards that cost real time this session

1. **Build inside the `ps5ys` distrobox.** The host has only the Vulkan *loader*, not the headers. A host
   configure **succeeds while silently omitting** `gpu_replay`, the offscreen graphics harness, and every Vulkan
   **execution** test, logging just `Vulkan not found`. A green host `ctest` is therefore not evidence that
   execution tests ran — confirm the binaries exist and report test names, not an exit code. Wipe the build dir
   when switching; a stale cache keeps Vulkan disabled.
2. **`shader_inspect` was misreporting ~96% of known-good shaders** (109 of 114, across all three stages) as
   `rejected`, because a raw dump carries no descriptors. Fixed in #1575 — it now reports
   `undetermined-no-resource-table` with exit 3. Any shader verdict taken from it before `37dfe752` is suspect.
3. **`-DGAME_DUMP` against a non-Messenger title fails 3 dump-parameterized tests** (`module_loads_eboot`,
   `boot_reaches_first_syscall`, `real_shader_render`). This is expected, not a regression — current master is
   **162/162** when configured against `PPSA24651`. Tracked as #1573.

### The instrument-not-the-subject list — read this before believing any surprising measurement

**Every entry below is a phantom defect that came from the measuring apparatus, not the subject.**
Several cost hours; one cost two sessions. This is the single highest-value page in this document.

*Maintenance:* **append, never renumber** — an existing number may already be cited from an issue or a
commit message. Deliberately no total is stated here: a restated count goes stale the moment a lane
appends, and it already had (the header read "Fourteen" against a 17-row table). If you need the
count, read the last row's number.

| # | Instrument | How it lied |
|---|---|---|
| 1 | `shader_inspect` | Reported **109 of 114 known-good shaders** as `rejected` — a raw dump has no descriptors, so table-dependent MIMG/MUBUF/MTBUF (and SMEM in graphics) correctly refuse. Fixed #1575. |
| 2 | `PROSPER_FS_TAP` | Exports `(dst, dst+1, dst+2, dst+3)`, so taps with different `dst` compare **different registers**. Manufactured a "~14x amplification". |
| 3 | Snapshot guards | `blue-prince-hall` and `terminator-boot` "FAILED" on **fully-rendered healthy frames** whose routes had drifted to another scene/animation phase. A control run on master reproduced both. |
| 4 | `sgpr:106` | Is `VCC_LO` **scratch**, not a uniform — a static constant-buffer read cannot resolve its value at point of use. |
| 5 | `--through-operation N` | Renders the **last executed draw target**, so adjacent prefixes show *different surfaces*. #1486's founding "corruption boundary" was 1920x1080 vs 3840x2160. Fixed by #1580's `target=`. |
| 6 | `git checkout <file>` | Restores from the **index** — it is not an undo for a temporary edit, and silently wipes uncommitted work. |
| 7 | A stale `in-progress` label | Read as occupancy. #1284's claim had been dead six days (no open PR, no activity); CLAUDE.md expires claims at 24 h. |
| 8 | An orchestrator's instruction | "If it is a missing capability, implement it" — all three opcodes were **already implemented**; following it would have duplicated a working `image_gather4_lz`. |
| 9 | `[compute] skip unsupported program` | **Deduped per `code_addr`**: one line means "failed at least once", NOT "always fails". A program running 763 times and failing once looks identical to a permanently blocked one. |
| 10 | `PROSPER_COMPUTELOG_CODE=<addr>` | Correctly protects a realtime route from ~13k submits of logging, and thereby **suppresses `execute` lines for every other program**. Right for one question, silently wrong for the next asked of the same log. |
| 11 | SHA reachability after a **squash** merge | `git merge-base --is-ancestor` reports "unmerged" for work that *is* on master. Verify by **content**, not by SHA. |
| 12 | `PROSPER_REGWATCH` **joined to** `PROSPER_EXECLOG` by stream adjacency | The two timestamp at **different pipeline stages** — REGWATCH at command-processor *decode*, EXECLOG at *realization* — and the phases interleave per submit. Read by log order, #1606 said **1,049 of 1,050** suppressed draws resolved against a *future* register write. Clean, compelling, and entirely false. Fixed by #1633's `order=`. |
| 13 | A **print-capped** diagnostic read as a frequency | `[agc] WaitRegMem … NOT satisfied` prints the first 40 then every 1024th (`ln < 40 \|\| (ln & 1023) == 0`); `[agc] out-of-range indirect reg write dropped` stops after **4**; `[agc] indexed indirect draw skipped` after 24. #1606 called the `WaitRegMem` volume "the loudest signal, dozens per second"; the true rate is **~1.5/s** (see below). A line count from these is a cap, not a rate. Same trap as #9's dedupe, different mechanism. |

| 14 | A **decoded-draw census read without the render phase** | Draw counts per frame are phase-dependent by two orders of magnitude *within one title*. Blue Prince (rung 3) decodes **7-13 draws/frame on its own menus** and **1,500-3,200 in its 3D scene**. So Nikoderiko's 53/frame on a title/EULA screen looks exactly like #1641's "implausibly few" signature and is simply **normal for a 2D UI screen**. Reporting it would have sent a lane chasing nothing. |
| 15 | "is this dword pair a mapped pointer?" | The user-data window is 32 dwords and holds many live pointers, so **several** seed offsets satisfy "every declared descriptor is readable". On #305 a shifted seed made all declared pointers land cleanly on **9 of 9** stages — and the hardware field that bounds the window (`SPI_SHADER_PGM_RSRC2_GS.USER_SGPR`) proved the stage cannot see there at all. A live A/B then raised rejects 118-141 -> 521. A numeric fit over a pointer-shaped predicate is weak evidence; find the register that bounds the search space. |
| 16 | One diagnostic **label** covering two packet kinds | #305's bind trace emitted `[bind] DRAW` from all five per-item snapshot sites in `GpuState::apply` — and **two of those are compute dispatches**. A dispatch never consumes `SPI_SHADER_PGM_LO_ES`, so a bind/draw agreement statistic computed over those lines mixes in events that cannot agree by construction. Graphics-queue dispatches read the *graphics* register file, so they print plausible `es_lo` and are **indistinguishable from draws in the log** — a re-run, not a re-parse. **Re-measured after relabelling** (a fresh boot — the two figures are not the same run: 434,239 bind packets vs 193,397), the contaminated "871,648 of 876,217 (99.5%)" became **300,404 of 300,404 (100%)**, so the residue had been dispatches. Tag every emitter with the kind it actually observed. |
| 17 | **Asymmetric** exhaustion of per-site caps | A cousin of #13. When one side of a paired trace caps (bind lines, one counter at 4,000) and the other runs on **five independent counters**, the first side dies during the pre-title load while the second keeps emitting; a "does each draw follow a bind?" analysis then reported a 41% mismatch that was pure counter exhaustion. Bound the analysis to the region where **every** stream is still logging. Related: `command_order` is **per-submit and resets**, so ordering events globally by it interleaves submits and manufactures impossible pairs. |
| 18 | `VAR=1 $unquoted_var cmd` | The shell expands `$unquoted_var` **after** it has decided which word is the command, so an expansion like `PROSPER_NO_PLUGIN_AUTOLINK=1` becomes the **command name**, not an assignment. The arm never runs, exits 127, and — with stdout redirected to a per-arm log — leaves an empty file that reads exactly like a legitimate "no output" result. #1609's founding 14,666-vs-18,167 "20% regression" came from an A/B whose OFF arm had never executed. Build env into the command directly, or use `env VAR=1 cmd`. |
| 19 | `$HOME` read as private scratch | Every lane on this box shares one `$HOME` (410 files at the time of writing). Generic scratch names — `pr-body.md`, `ab-result.md`, `app.log`, `apr.log` — **already collide across lanes**. Writing one destroys another agent's work in flight, and reading one silently answers your question with someone else's data. Title-scope every scratch file (`tales-pr-body.md`, `tales-rate-*.log`). The tell that saved it here: the Write tool **refused to overwrite a file it had not read**. A tool declining to clobber is a safety net, not an obstacle — an agent that "fixes" it by deleting the file and retrying destroys the thing the guard was protecting. |
| 20 | An **empty CI rollup** read as "queued" | GitHub does **not run checks on a conflicting branch at all**, so a PR whose base has moved reports `pass=0 fail=0 pending=0` with `mergeStateStatus=DIRTY` — permanently, until it is rebased. That is byte-identical to the "checks have not registered yet" state, and #1656 sat in it while master moved twice. Same shape as #18's empty log: **an absent signal mimicking a pending one**. Read `mergeable`/`mergeStateStatus` before concluding anything from a check count, and never wait on zeros. |
| 21 | The **same draw census, now read without the resident *content*** — trap 14 firing a second time, on the lane that recorded it | #1641 spent a session treating "~23 draws/frame is implausibly few for a 4K UE4 title" as the anomaly. Decoding the title's own asset reads against its pak index showed it loads **exactly one of the 481 maps** in the container — `L_GameloftSplash.umap`, **7,870 bytes** — and never opens `L_Main` or `L_StartMap`. 23 draws is a **complete and correct** deferred frame over a splash world. Worse, the premise was invalid in *both* directions: `L_Main.uexp` is **616 bytes**, so this title's front end is UMG/Slate and a *working* title screen would also decode few 3D draws. Nothing was ever missing. |
| 22 | A **runtime `eboot+0x…` offset** | Several diagnostics subtract the literal `0x400000000` while the eboot maps at `BOOT_EBOOT = 0x410000000`, so the printed offset is **`0x10000000` too high** — wrong but entirely plausible, and every downstream `edis.py`/`xref.py`/`PROSPER_BP` lookup then fails in a way that looks like a code problem. `boot_trace.cpp:97` uses the right constant, so two conventions coexist under one label. Filed as **#1659**. Tell: an `eboot+0x1…` offset **past the module image size** is a labelling artifact, not a wild pointer. |
| 23 | A **call-count table read as evidence of *what* is being waited on** | `PROSPER_PROGRESS_UNIMPL`'s per-NID counts showed **1,300,185,430 `select()` calls in 144 s** on PPSA19244 while every other socket call stayed a frozen singleton — a real, liveness-controlled measurement. It was then read as "the guest is blocked polling a socket", and a shared-substrate design decision (implement a loopback socket backing) was nearly escalated on it. Reading the **arguments** took one `PROSPER_STUBDUMP` lookup plus one gdb breakpoint and showed `select(nfds=0, NULL, NULL, NULL, {3,0})` — the portable **`sleep()`** idiom, no descriptor involved. The thread was trying to sleep 3 s; the 0-returning stub made it a no-op. Not a socket poll, not the stall — a CPU tax. A count answers *"is something spinning?"*; it can **never** answer *"on what?"*, and the function's name is not evidence. |
| 24 | A log whose **emission gate** is narrower than the population you reason over | `[udmap]` (#305) prints a stage's user-data map **only under failure replay**, so every row it emits is a stage that ALREADY FAILED. Its rows therefore sample failures, not stages — and a row showing readable pointers is a stage that failed for a *different* reason, **not a passing stage**. Building a pass/fail table from it silently answers a question it was never sampling. Same category error as #9 and #13 (a count that answers "did this ever happen?" but not "how often?"), one level up: here the **rows** are the biased sample, not the counts. Before treating any log's rows as a population, read the condition it is emitted under — grep the `if` above the `fprintf`, do not infer it from the message text. |
| 25 | `gh pr merge --delete-branch` run **from a worktree** | After the server-side merge succeeds, `gh` tries to check out the base branch locally and dies with `fatal: 'master' is already used by worktree at …` — which is unavoidable from a worktree, since master is checked out in the main tree. **The merge lands; the branch survives; the exit status does not cleanly separate the two.** Two lanes hit this in one session, both on their own merge. Read as "the merge failed" it invites a retry against an already-merged PR; read as "it all worked" it leaves a stale remote branch that later reads as an unfinished lane (and doubles as a false claim-lock, per the claim rules). Verify the two halves **separately**: `gh pr view N --json state,mergeCommit` for the merge, `git ls-remote origin 'refs/heads/<branch>'` for the branch, then `git push origin --delete <branch>` explicitly. |
| 26 | A **commit-SHA-pinned asset URL** in a PR body or issue | Screenshots referenced as `raw.githubusercontent.com/<org>/<repo>/<sha>/…` break silently once that SHA is **orphaned** — which a rebase does immediately and a squash merge does at merge time. The images render perfectly while the PR is open and are dead weeks later, when nobody can reconstruct what they showed. Pin progression evidence to a **branch that survives** (`…/master/assets/screenshots/…`) once the file is on it, and re-point any SHA URL after the final rebase. Cousin of #11: both are the same mistake of treating a SHA as a stable name across a history rewrite. |
| 27 | **A confident document header with no observation behind it** | Three parked `reach-gameplay.pad` drafts (Joe & Mac, Asterix, Summer Sports) carried three *different* title-specific rationales — "Options is a fallback for a start-button title prompt", "the title screen is reached with no input at all", "Cross confirms the default selector entries" — over **byte-identical input sequences** (`md5` of the non-comment lines matched across all three). Three different findings cannot come from one identical button mash, so the prose was generated, not observed. The genuine routes were in the tracking issues the whole time. |
| 28 | A **distinct-colour count quoted across two different render scales** | Triage issue #1593 recorded Joe & Mac gameplay at **444,000-468,000** distinct colours; profiling the same route through `tools/snapshot` measured **at most 79,167**, a 5.6x gap that reads exactly like a severe colour regression. Both are correct. Triage captures run the `screenshot` frontend at native 1920x1080; every snapshot guard runs at `scale: 4`, so the frame is **480x270 = 129,600 pixels** and cannot exceed that many colours by construction. A 3840x2160 title (*Summer Sports Games*) yields 960x540 at the same scale, so even `dims` differs between titles. **Never lift a `min_colors` threshold from a triage issue**, and read `dims` off the profile instead of assuming the 1080p value. |
| 29 | **Ranking candidate guard windows by metric stability** | Choosing a content-guard window by "tightest colour spread / highest self-SSIM" is the obvious automation, and on *Summer Sports Games* it confidently selects the settled span from 150 s to the end of the route: 48,201-48,489 colours, 1.0000 coverage, **SSIM 1.000 against itself** — the best-scoring window in the entire run by every available metric. It is the static `JAVELIN THROW` **standings overlay**, not gameplay. A guard adopted there would have passed forever while the athletics scene collapsed. Stability is a property of *held* frames, and menus, results screens and pause overlays are the most held frames a title has, so the metric is actively biased toward them. Rank windows to shortlist; **decide by opening the images**. Same family as the earlier note that a diagnostic clear can out-score real content on distinct colours. |
| 30 | **A wall-clock guard window placed in a route that keeps moving** | A content guard samples by wall-clock seconds, but in a route that drives the player *forward* the scene at second N is a function of ~N seconds of accumulated progress, so ordinary run-to-run variance moves the **scene**, not just the timing. Joe & Mac's movement route passed `verify` twice and a `check` twice, then the third run drifted onto a level transition plus two death fades and failed 19 structural matches against 24 required — no build change, same route. Replacing it with the **stationary** route inverted the problem: the player is killed every ~22 s but respawns to the *same* opening screen, so a 73-frame window spanning three lives is one visual state and tolerates the deaths (69 matches against 55 required, worst-frame SSIM 0.97 instead of 0.81). The more impressive route made the worse guard. Prefer a route whose guarded span is **self-restoring** — an idle screen, a respawn loop, a held camera — over one whose span is a trajectory, and treat two green runs as far too small a sample to call a moving window stable. |
| 31 | The **first `grep` hit for a register**, when it has more than one consumer | Tracing why Astro's draws resolve `cwm=0`, `CB_COLOR_CONTROL.MODE` is read at `render_state.cpp:143` — where it feeds **only** the gated `PROSPER_MSAA_LOG` diagnostic. Reasoning from that site says MODE does not participate in mask resolution, which **inverts the conclusion**. The site that matters is ~650 lines later, where `MODE == DISABLE` explicitly zeroes `color_write_mask`, `color1_write_mask` and every `color_targets[].write_mask`. Same symbol, two consumers, opposite implications — and nothing at the first site announces that it is not the one. **Find the consumer on the path you are reasoning about**: for a render-state question that is where the pipeline state is built, not where a diagnostic prints. |
| 32 | **`std::mismatch` read as a failure count** | The R11G11B10 exactness sweep (#1681) printed its first difference with `std::mismatch`, which returns only the **first** differing element, while the assertion compared the **whole vector**. The printout is byte-identical whether 1 texel differs or 92 — so "the same single texel fails on both machines" was a number the instrument never produced, and the supporting argument built on it ("a genuinely wrong encoding would fail broadly rather than at a single value") had no evidence behind it. The real count was **92**. Cousin of #9/#13/#24: an instrument that answers *"did this happen?"* being read as *"how often?"* — here at the level of a single printed element rather than a rate. **When an assertion compares a collection, the diagnostic must summarise the collection**: count the differences, classify them, and report the classification, because the breadth of a failure is usually the thing that selects between explanations. |
| 33 | **A skipped draw's clear colour read as a rendered pixel** | Six assertions in `test_recompiled_fragment` and one in `test_recompiled_shaders` "failed a rendered-pixel comparison" on lavapipe (#1681). No comparison happened: the shaders require an exact 64-lane fragment subgroup, lavapipe is fixed at 8, and prosper **correctly refused the draw** (`[render] skip draw: fragment shader requires subgroup size 64 (device range 8..8)`, 15 times). The target simply kept its BLUE clear, and a clear colour is a perfectly plausible-looking pixel value — nothing about `center=(0,0,255,255)` announces that no draw ran. The neighbouring assertions in the *same files* already gated on `supports_fragment_wave64_vote`; these did not. **A pixel assertion is only evidence if the draw executed** — assert the capability, or assert the clear, but never read back a target without knowing which of the two you are looking at. Note the failure direction: a fail-visible skip is designed to be loud, and it was — on **stderr**, while the assertions printed to stdout, so the two never appeared together in the ctest summary. |
| 34 | **A check of the ANSWER read as a check of the RULE** — and a shape shared by several entries here | Deriving an NV12 frame size (#1687), the rule "`w*h*3/2` and `wh + (wh+1)/2` agree" took three attempts: "identical for every reachable input", then "agree whenever `w*h` is even", then the correct one. **Two** were wrong, each in a smaller way; the third was right. The truth is **exact iff BOTH dimensions are even** — independently re-derived three times, most widely at 200x200 (40,000 pairs), never over-reporting — an even *product* is not sufficient and is actively a trap, since `1920x1081` has an even product and the derivation is **960 bytes short**. The error survived three rounds of competent review because every size anyone checked (`1920x1088`, `1920x1080`, `640x480`) is **both-even**, so each spot check confirmed the true rule *and* the wrong one equally. **The cases in front of you are not a sample; they are a coincidence until you show they discriminate.** An exhaustive grid settled it in one command and could have been run first. The general form: verifying the *answer* on the examples you already have tells you nothing about the *rule* — you must ask what input would distinguish your rule from its near-miss, and check that one. Named instances rather than a fraction: an A/B whose arms agree when they must differ (the tell under #10), a pre-registered decision rule that cannot separate its two hypotheses, a vacuous assertion that holds whether or not the event occurred, #33's pixel assertion on a draw that never ran, and this one. **Every one of those PASSED**, and that is the mechanism of concealment, not a coincidence: a green result is **self-authenticating**, so a check that *could not have failed* is indistinguishable from a check that genuinely passed, and both get filed as confirmation. A failing check gets investigated; a vacuous one gets believed. **So the question to ask of your own test before you write down what it showed is not "did it pass?" but "what result would have falsified it?"** — if there is no answer, the test established nothing, however green. Note where it bit: this is normally reached for around *instruments*, and here it was plain arithmetic — nobody thinks to apply the discipline to a formula. |
| 35 | A **`.bmp` and a same-named `.prgbundle` assumed to be one F9 grab** | The frontend writes a grab's bundle through a `.tmp` and its screenshot separately, so **killing the app mid-grab leaves a `.bmp` whose same-named `.prgbundle` is from a previous grab entirely**. A Dragon Quest VII investigation was handed `dq7_grab_001.*` and `dq7_grab_002.*` as "two frames of the same scene in two states" — the highest-value comparison available. `stat` showed 001's pair 0.2 s apart, but `dq7_grab_002.prgbundle` **51 minutes older than `dq7_grab_002.bmp`, from a different boot**; the screenshot's real partner was the abandoned `frame_grab_002.prgbundle.tmp` written 0.7 s before it (and it loads fine). The mismatched pair was a 1920x1080/1519-draw **scene** submit against a 3840x2160/97-draw **composite** submit, which would have manufactured a large, entirely artefactual structural difference in exactly the comparison the investigation was told to trust. Names and adjacent ordinals are not provenance: **`stat` the artifacts and confirm sub-second pairing before treating a screenshot as evidence about a bundle** — and prefer a `.tmp` with the right mtime over a completed file with the wrong one. **Fixed at the source (#1693 / #1694), and the replacement rule is exact:** a grab claims BOTH names when it is armed, from one timestamp and at one collision suffix, so **the two artifacts of one grab always carry the identical full stem, suffix included** — the suffix is a property of the capture, not of a file. So pair on the **whole stem**, never on the timestamp: `…-210000-123.bmp` and `…-210000-123-2.prgbundle` are two *different* grabs despite an identical title and millisecond, and reading them as a pair re-creates this exact trap with more confidence than before. The `stat` habit remains correct for artifacts captured before that landed — `captures/frame_grab_001.*` in the main checkout is exactly that shape — and for anything else named by ordinal. |
| 36 | **`snapshot.py verify` passing, on a scene the title regenerates randomly** | `verify` builds its structural references from **both** of its own two runs and then asks whether those runs match that combined set — so a scene that is *different every run* still passes, because each run matches the half of the reference set it contributed. Worms Armageddon's Quickstart generates its terrain per match: two fresh-save runs produced a pirate/treasure map and a fairytale/castle map, each scoring 70/70 against the combined references and looking like a healthy `CONTENT-STABLE` baseline. Measured **across** runs instead — run B's frames against run A's references **only** — the same window scores at most **0.6567**, with **0 of 70** clearing the 0.85 floor. Such a guard would be adopted green and fail on the very first `check`, whose third random map matches neither. The tell is available before any capture: the 16x9 SSIM is highly layout-sensitive on this title (the *same* map mirrored scores 0.4595), so any terrain variation is fatal. **Before adopting a window, score one run against the other run's references alone** — verify's own output cannot answer this, and 'it passed verify twice' is not evidence of stability (cousin of #30, where two green runs were also too small a sample). The fix here was a route change, not a threshold change: Training levels are authored fixed maps and cross-run scoring on them gives 0.9612-1.0000, 47/47. |
| 37 | **A key injected into `prosper-app` that the app never receives — silently** | Validating the F9 frame grab (#1693) on this box, `xdotool key F9` reached a correctly named, correctly focused XWayland window and produced **nothing**. Two readings were available and both were wrong: "F9 is broken" and "my change broke F9". The controls settle it: injected **Esc quits the app** and injected **Pause pauses** it, while **F10 and F11 — which this change does not touch — are lost exactly like F9**, and `Pause` and `F10` are literally the *same command* in the same handler. So XTEST on this host does not deliver **function keys** to the window; nothing about the app is implicated. `ydotool` (uinput, `ydotool key 67:1 67:0` for F9) does deliver them, and the whole F9 path then worked first time. **Two rules:** a key-injection experiment must carry a positive control through the *same* path — end the run with an injected `Esc` and require the app to exit, or the negative result is void rather than negative — and `2>/dev/null` on the injector hides the commonest failure of all (the first attempt ran `xdotool` **inside** the distrobox, where it is not installed, and the redirect swallowed `command not found`). |
| 38 | **A driver-version split, read as a driver bug** | Two 3D storage-volume assertions in `test_game_compute` failed on Mesa 25.2.8 lavapipe and passed on Mesa 26.1.4 lavapipe **and** on RADV (#1690). Both driver *classes* agreed with each other and disagreed with one version, which is about as clean a version signal as this list ever gets — so the working hypothesis was an upstream lavapipe defect, and the proposed resolution was a version-conditioned skip. It was **prosper's own fixture**: `image_copy_3d` prepared a third address register and called itself 3D while leaving MIMG.DIM encoded as 2D, so the recompiler faithfully emitted `OpTypeImage … 2D` with two coordinates while the backend bound the `VK_IMAGE_VIEW_TYPE_3D` view its 3D *resource* descriptor calls for. A 3D view on a descriptor whose module declares Dim 2D is **undefined** in Vulkan, so all three results are conformant and the "version axis" was three drivers each resolving one piece of UB their own way. Encoding DIM=3D passes everywhere. **The rule this teaches is about what a cross-implementation split licenses you to conclude: it localises the disagreement, it does not assign the fault.** Two implementations agreeing does not make them right — UB is agreed on far more often than it is diagnosed — so before attributing a split to the odd one out, ask what *you* are asking all of them to do, and whether it is defined. The cheap discriminator was available the whole time and settles it in one run: a **standalone reproduction of the same operation** with none of prosper's code in the path (own instance, device, images, hand-written kernel). It passed on 25.2.8, which exonerates the driver and points the search back inside — the inverse of the usual instrument trap, since here the *subject* was fine and prosper's own input was malformed. Two corroborations were also sitting in the tree, unread: `fill_3d`, in the same test file, already encoded `0xF0200F10` as `dim:3D` and passed on 25.2.8, and `test_rdna2_decode.cpp:309` pins the same DIM field — bits [5:3] = 2 — against an `llvm-mc gfx1010` capture, though on a different literal (`0xf0000f12`: another opcode, plus an NSA bit), so compare the field rather than the constant when following that pointer. Cousin of #34 — a green result on two of three configurations is self-authenticating, and "it works on the good drivers" is exactly what correct code and undefined behaviour look like alike. Note the cost of the wrong reading: the proposed skip would have removed **484 executed assertions** from CI permanently, annotated as a driver limitation. |

**Detecting the generated-prose trap (#27), because it generalises past `.pad` files:**

When several artifacts claim *independent* findings, **diff their substance, not their prose**. Identical
bodies under differing explanations means the explanations were written to sound like observations:

```bash
# strip the commentary and compare only the load-bearing content
for d in joe-mac asterix summer-sports; do grep -v '^#' "$d/reach-gameplay.pad" | md5sum; done
```

Seconds of work invalidated three documents. The same check applies to route scripts, per-title status
docs, issue comments and PR bodies — anywhere parallel artifacts assert parallel results.

**It also happens in source comments, which is where it does the most damage.** "It happens in `.pad`
files" understates the trap; a second lane hit it the same day in `hle_service.cpp`. Recording the
first live confirmation of `VdecConfig` (#1687), the comment claimed the guest's values were
"individually meaningful at **every** offset" and that a wrong field order "**could not**" produce
them. Checking the pairs one at a time instead of asserting the general claim: `profile=100` /
`max_level=41` cannot swap (41 is not a `profile_idc`, 100 is not a `level_idc`), `max_dpb` /
`input_depth` cannot (depth would go negative), width/height cannot (the movie would be portrait) —
and `resource` / `codec` are **both `1`** in that title, so that one pair is not discriminated at all
and still rests on the commit that introduced it. The sweeping version was written first and read as
established fact.

A doc carries a `docs/` path and some ambient suspicion; a source comment sitting above a
`static_assert` reads as verified by construction, nothing marks it as inference, and every future
reader inherits it. So when a comment records evidence, **enumerate what the evidence forces and name
what it does not** — "HIGH except this pair, which is MED, and a non-AVC title settles it in one line
of log" is a comment the next reader can act on. A general confidence claim is one they cannot check.

This is **worse than a stale `UNVALIDATED` banner**, and for a specific reason: a banner tells the reader
to check. A confident, specific, plausible header tells the reader **not to bother**, so it survives every
subsequent reading. The remedy is not to flip a banner but to **add accurate provenance where a confident
claim has none** — hold a route header to the `scripts/greak/reach-gameplay.pad` standard, which cites
*observed* phase timings ("the title screen appears at roughly 112 s on a Linux hardware-Vulkan boot")
rather than asserted behaviour. A claim marked ✅ on a run nobody can reproduce is the same defect one
level up.

**Working rules that follow:**

- **A decoded-draw census is meaningless without knowing the render phase, and needs a positive control.**
  Menu, loading, and 2D-UI phases legitimately decode single-digit to low-double-digit draws per frame, so a
  low count is evidence of nothing on its own. Count **per sample interval**, not per run — a whole-run
  average describes neither regime (Blue Prince's is 622.7, which is neither its 10 nor its 2,200) — and
  calibrate against a title **known to render the thing you claim is missing**. Best of all is one control
  exhibiting *both* regimes in a single run, which rules out per-title decode differences as the
  explanation for the gap. Then confirm the phase by **opening the frames**: #1641's Nikoderiko row was
  only interpretable once the images showed samples 10-11 carrying ~160k distinct colours while 3-9 were
  black, which proved the census was tracking real workload rather than drifting.
- **Build a timing measurement that cannot lie quietly.** Everything above is "this instrument lied"; this is
  the counterpart — the shape of a measurement that reports its own invalidity instead of averaging it away.
  Seven lanes share one GPU, so vigilance does not scale: a run *will* eventually overlap someone else's
  capture, and a contended sample looks exactly like a real regression. Three properties, all cheap:
  1. **Self-lock** (`flock` on a lockfile) so a second copy of the driver refuses to start rather than racing
     the first. Two waiters queued behind the same "GPU free" condition both fire when it clears.
  2. **Build strictly before measuring, never concurrently.** A `-j8` compile landing on top of a run costs
     more throughput than whatever is being measured. Put the build inside the driver, ahead of the loop.
  3. **Discard, don't average.** Sample the foreign-process count **before and after** each run; if either is
     nonzero, **rename the log to `DISCARD-…`** instead of dropping it. Contention then becomes *visible in
     the record* rather than invisible in the mean, and the discarded run stays auditable afterwards instead
     of surviving as an unexplained outlier.
  4. **Alternate the arms, then check the residuals for cycle correlation** before attributing any
     difference to the treatment. Running all of arm A then all of arm B lets a *time-varying* confound
     land entirely on one arm and become a phantom effect. Alternating forces it onto both, which makes it
     detectable: in #1609's 12-run experiment the slowest run was **cycle 2 in all four title × arm
     combinations**, so the residual variance tracked *when* a run happened, not *which arm* it was. That is
     not an argument that the gap is noise — it is a demonstration, from the same data that would otherwise
     have produced the phantom. If the extremes line up by cycle rather than by arm, the treatment is not
     the explanation.
  Report **whole-run and steady-state (tail-window) rates separately**: a one-off boot cost and a per-frame
  cost are indistinguishable in a whole-run average, because a run that reaches steady state one second later
  and then performs identically still shows a lower average. Report min/max spread and an explicit
  **overlap verdict** across arms — if the arms overlap, say so plainly and the question is closed.
- **Joining two instruments requires an explicit shared ordinal, never stream adjacency.** If two diagnostics
  are emitted at different pipeline stages, their interleaving in the log is an artifact of *when each stage
  ran*, not of when the events happened. Print a common ordinal (`command_order`) on both and join on it. This
  is why the `order=` field exists on the `[exec] skip draw early` line — **it is not redundant with the
  register values on the same line, and removing it silently re-opens the phantom above.**
- **A guard keyed on a threshold has an expiry date and gives no signal at expiry.** Key it on the thing
  itself. #1675 added a check that CI must register at least 160 tests, so a configure that quietly lost
  Vulkan would fail instead of reporting a smaller green suite. It read as the careful part of the change
  and it was decoration with a delay: CI has 138 non-Vulkan tests, the suite only grows, and once ~22 more
  exist, losing all 27 Vulkan tests still clears 160. The guard would have stopped protecting the thing it
  was added for, at some unremarkable future commit, silently — the same shape as the defect it was
  guarding against. Replaced with a check that three **named** Vulkan-gated tests are registered: they
  cannot exist unless Vulkan was found, and that stays true at any suite size. The general form — **a
  threshold derived from today's numbers decays as the system grows; assert the property, not a count that
  happens to imply it today** — and the corollary: verify a guard fires in **both** directions, because
  one that cannot fail is worse than none.
- **Before recording that something cannot be measured, check the tools you are claiming cannot measure
  it.** This is nastier than a wrong fact and it is self-sealing. A wrong claim about the *subject* gets
  tested by the next measurement; a wrong claim about the *instrument* stops the measurement from
  happening, so nothing ever contradicts it. #1635 shipped "severity is unknowable — `self_dump` has no
  import listing" into a PR body. `self_dump` prints `[IMPORTS BY LIBRARY]`, documented in
  `tools/re/README.md`, and answering the question took ten minutes once someone looked: 21 aliased NIDs,
  **0** imported by anything. An "unanswerable" is a claim about a tool, and claims about tools are the
  cheapest of all to verify — run `--help`, grep the README, read the source. The same session produced
  the mirror image in #1675: CI's green tick was believed to mean "the suite passed" when it meant "the
  138 tests that got registered passed", and the 32 missing ones — ~1,822 assertions — were invisible
  because nothing reports a test that was never registered.
- **A fix that makes an instrument lie more convincingly is worse than the bug.** This one is about
  *repairs*, not measurements, and it is the inverse of everything else on this page. #1659's own fix
  briefly introduced it: the diagnostics were widened to resolve any guest module, but several kept a
  hard-coded `eboot+` label. Before the change an Il2Cpp frame printed an offset *larger than the image* —
  the exact tell the entry below tells you to look for. After it, the same frame printed a small,
  plausible, in-range offset under the **wrong module name**, with no tell at all. The fault was
  unchanged; only the symptom was removed. Any correction that deletes a symptom without deleting the
  fault produces a **quieter** failure, and quiet failures are the expensive ones. When you fix an
  instrument, ask what its wrongness used to look like and confirm you removed the wrongness rather than
  the appearance of it. Corollary for tests: a test that exercises the *helper* you added will not catch
  this — it has to render through the real formatter and assert the text a reader would see.
- **A `<module>+0x<rva>` offset larger than the module's image is a labelling artifact, not a wild pointer.**
  Guest module bases are fixed constants in `src/host/boot_program.hpp`, and they *move*: #825 relocated the
  eboot from `0x400000000` to `0x410000000` so Astro Bot's direct-memory mapping would stop aliasing code.
  Diagnostics that had hard-coded the old literal kept printing `eboot+0x…` against it, so every offset was
  the real RVA **plus `0x10000000`** — an offset past the end of a 161 MB image, and one that no longer
  round-tripped through `PROSPER_BP` (which adds the *mapped* base). #1659 converged every printer onto
  `prosper::guest_module_name()` / `guest_module_offset()`; if you are hand-computing a base anywhere, that
  is the bug. Two corollaries worth keeping: an address below the lowest module base is **data, not code**
  (that region is now a DMEM aperture), and a single wide range test labelled every module in it "eboot",
  so a wrong *binary* is as likely as a wrong offset. Static tools are unaffected — `xref.py`, `edis.py` and
  `prx_to_elf.py` compute from ELF `p_vaddr`/file offsets and never add a runtime base, so RVAs recorded
  from disassembly stay trustworthy.
- **Before quoting a diagnostic's line count as a rate, grep for its cap.** These sites use a
  `static std::atomic<int>` counter or a per-key dedupe set; check for one before concluding "this fires N
  times" or "this is rare". When the site prints its own counter — `WaitRegMem #%d` *is* `ln` — the true
  total is free: the highest `#N` in the log bounds it, and a run whose maximum stays under the cap has
  printed *every* occurrence. That read turned #1606's "dozens per second" into 18 and 21 events for two
  entire runs.

- **Before concluding that geometry is missing, establish what content is actually resident.** A draw
  count describes the frame; it says nothing about whether the world has anything in it. For a
  UE4/pak title this is one offline command against a log you already have —
  `tools/re/pak_index.py GAME.pak --log run.log --distinct` resolves the `[apr] read-submit` byte
  offsets to asset names, so "which maps/blueprints/widgets did the guest load, and where did
  loading stop?" is answered with no boot and no GPU. Run it *first*. On #1641 it dissolved a
  session's worth of GPU investigation in one pass.
- **Read an unimplemented call's *arguments* before inferring what it is doing.** This is cheap and
  almost never done. `PROSPER_STUBDUMP=1` prints one line per import — `[stub] #877 off=0x148e0
  libScePosix::T8fER+tIGgk` — and each stub lives at `0x600000000 + off`, so a gdb breakpoint at the
  **stub entry** catches the guest's argument registers fully intact (the stub has not clobbered
  anything yet; note the generic unimplemented path *does* overwrite `rdi`/`rsi`/`rdx` by the time it
  reaches `prosper_on_unimpl`, so break on the stub, not the C++ handler):

  ```bash
  PROSPER_STUBDUMP=1 … boot_trace <dump> 2>&1 | grep '<NID>'      # -> off=0x148e0
  gdb -p $(pgrep -x boot_trace) -batch -ex 'set pagination off' \
      -ex 'break *0x6000148e0' -ex continue -ex 'info registers rdi rsi rdx rcx r8'
  ```

  The stub table is per-run — re-read it, never reuse an address. On PPSA19244 this converted
  "1.3 billion socket polls" into "a 3-second sleep that returns instantly", and stopped a
  substantial piece of shared work from being scoped against a wrong premise.
- **A trap you have already recorded is the one most likely to catch you.** Trap 21 is trap 14, hit
  by the lane that wrote trap 14 down, because the rule had been internalised as "check the render
  *phase*" when the general form is "check the *denominator* — phase, resident content, and engine
  architecture — before treating a low count as a defect."
- Prefer experiments that **detect their own invalidity** — e.g. render two adjacent operations and assert both
  return the same resolution, so a surface switch invalidates the comparison instead of producing a phantom.
- **Open the image.** Believe neither a metric win nor a metric failure without looking. A diagnostic clear can
  out-score real content on colour count; a healthy frame can fail SSIM on animation phase.
- A measurement that is **right for one question can be silently wrong for the next one asked of the same data**.
- Never re-derive a bisect boundary without confirming both sides render the **same target**.
- Check whether a later comment or merged PR already **superseded** an inherited premise before acting on it
  (Bendy's "CPU software-detile dominates" had been overtaken by #1179/#1190/#1269 in its own issue).
- Write `Refs #NN`, not `Fixes #NN`, when a PR only partially addresses an issue — a parenthetical qualifier does
  **not** stop GitHub's auto-close, and #1554 was closed that way despite a comment saying to keep it open.

### State only what is true at the moment and scope you state it

**A statement should only assert what is true at the moment and scope it is made. A line that announces an
*intention* in the grammar of a *fact* is indistinguishable from a result — and both humans and agents act on it
as one.** A confident statement reads identically whether or not it is earned, so nothing announces itself when
it stops being true.

Instances, all the same shape:

- The F9 arming line announced a filename it could not guarantee. A grab that aborted then left a `.bmp` beside a
  same-named `.prgbundle` from a different boot 51 minutes earlier, and a lane was handed the two as a matched
  pair. Fixed by naming the title at arm time and the real path per artifact after each file exists (#1693).
- A PR body asserted a `CONFIDENCE: HIGH` the source had already retracted; merging would have written the
  withdrawn version into the permanent record.
- An issue title asserted a mechanism that had been falsified, and kept sending agents at it for a day.

The practical form:

- If you cannot guarantee it when you write it, **state what you do know** and emit the fact when you have it.
- If a claim rests on three examples, **say "verified on three cases"** — not "for every input".
- A document, title, or comment that outlives its reasoning becomes **an assertion nobody is checking**.
  Retitling a falsified issue and correcting a stale PR body are correctness work, not tidying.

Same family as trap #34 (a check of the *answer* read as a check of the *rule*) and the source-comment case
under the generated-prose trap above: those are claims whose **scope** outran their evidence, this is claims
whose **moment** outran theirs. In both the grammar is what does the damage — nothing in a confident sentence
distinguishes the earned version from the unearned one.

### A test seam that pins a policy makes its env-var A/B silently void

An A/B was run by setting `PROSPER_ULT_RETURN_SUCCESS=1` to force libSceUlt's mutex to a no-op, expecting the
mutual-exclusion test to fail. **It passed**, which read as "the test is too weak to detect a broken lock" — a
believable and entirely wrong conclusion. The test calls `ult_set_return_success_for_test(false)` in its own
`main()`, deliberately, so that the environment cannot decide whether it passes. That seam overrode the variable,
so the "no-op" arm was never a no-op; both arms ran the same real mutex. Re-run with the pin removed, the
original test caught the broken lock immediately (27,984 of 40,000 increments surviving).

**The rule: any A/B driven by an environment variable is void until you prove the variable actually took effect.**
Have the run print the policy it is operating under, or assert the arm's expected *behaviour* before trusting its
result. This is the same shape as trap #10 (a switch that is right for one question and silently wrong for the
next) and trap #7 (a stale label read as state): a control that legitimately exists for one purpose silently
neutralises a measurement made for another.

Two corollaries worth keeping:

- The tell is an A/B where **the arms agree when they must differ**. Treat that as an instrument failure first,
  not a finding. A negative control that cannot fail is not a control.
- A test that pins its own configuration is still the right design — it stops the environment deciding the
  verdict. The defect is using that same variable as the A/B lever. Give the A/B its own lever, or remove the pin
  for the experiment and say so.

### A host function is not a valid stand-in for guest code — and only Windows will tell you

A test for libSceUlt's ulthreads used an ordinary host C++ function as the ulthread entry. It passed on
Linux and macOS and failed on **Windows MinGW only**, for three hours, on the single assertion that read the
entry's argument back out.

The guest is **System V AMD64**; prosper enters guest code through `prosper_call_guest_on_stack`, which
marshals into SysV (first argument in `%rdi`). On Linux and macOS the host ABI *is* System V, so a host
function is an accurate model and the test is sound by accident. On Windows the host is **Microsoft x64**,
where the first argument arrives in `%rcx` — so the untagged host function read a garbage argument and
returned a garbage status **while still executing correctly and touching its stack**, which is what made it
look like it worked. The product was never wrong; the test was. `win_thread_trampoline` already documents
this exact hazard for the production path ("a bare `pthread_create(entry, arg)` mis-passes the arg (MS x64
vs SysV)").

**The rule: any test that supplies a function for prosper to call *as guest code* must tag it with the guest
convention**, `__attribute__((sysv_abi))` on x86-64, not leave it at the host default. Prefer the tag over an
`#ifdef` around the body: "this function uses the guest's calling convention" is true on every platform and
the attribute is simply redundant where the native ABI already matches. Keep such entries plain leaf
functions — no destructors, no exceptions — because the attribute conflicts with MinGW's SEH-based C++
unwinding, which is why `PROSPER_SYSV_ABI` is empty for the HLE handlers (see `dispatch.hpp`).

The generalisable half: **a green Linux run says nothing about ABI-boundary code.** Guest entries, callbacks
the guest invokes, and anything reached through the call-guest shims are the cases where the dev platform
cannot fail and Windows is the only place the bug is visible.

### Registering a NID with an error return can be worse than leaving it unregistered

A trap of the same family as the list above, but in the HLE surface rather than an instrument. It cost a
merged defect (#1618, introduced by #1614), so it belongs beside them.

The dispatcher's default for an **unresolved** NID is `return 0` (`hle/dispatch.cpp` `prosper_on_unimpl`).
That default is safe for a function whose contract returns a **value**, and unsafe for one that returns a
**status** — which is exactly why fail-visible registration is usually an improvement. The trap is that the
inverse is equally true, and easy to miss: **an error sentinel is unsafe for a value-returning contract,
because such a signature has no error channel and the sentinel is read as data.**

`sceUltWaitingQueueResourcePoolGetWorkAreaSize` returns a `size_t` in `rax` which the guest passes straight
to `malloc`. Registering it to return `SCE_KERNEL_ERROR_ENOSYS` therefore asked for a 2.0 GiB allocation
where the unregistered default had produced a harmless `malloc(0)` — the #544/#660 class, reintroduced by
the very change meant to prevent it.

**The rule:** before registering a NID purely to make it visible, establish from the call site whether its
contract returns a status or a value. If it returns a value and the real one is unknown, the dispatcher's
`0` — or better, an honest computed value — is correct, and the visibility belongs in the log line, not the
return. Disassembling one call site is enough: `call …; mov [rbp-N],rax; …; call malloc` settles it in
seconds, and no amount of reasoning about the function's *name* substitutes for it.

### Proving "the guest never touches X" — pick the exhaustive instrument, not the plausible one

Recurring question when deciding whether prosper may write bookkeeping into a guest-owned struct: *does guest
code ever read inside it?* Answering it for libSceUlt (#1603) produced one sound instrument and one tempting
unsound one; the difference is worth reusing.

**Unsound:** identify the functions that operate on the owning C++ class by the displacements they use
(`[reg+0x398]`, `[reg+0x410]`, …), then look for accesses inside the member's range. Displacement sets are not
identities — `memcpy`'s bulk AVX stores and every function's own `[rsp+0x...]` stack frame match them. This
reported **1,361** then **707** hits, all false, and tightening the heuristic eventually matched *nothing*,
which is equally uninformative. A heuristic that can be tuned from "everything" to "nothing" is measuring the
tuning, not the subject.

**Sound:** enumerate the objects' actual addresses, then scan **every** RIP-relative reference in the whole
executable segment and ask which land at a non-zero offset from one. That is exhaustive over the only
addressing mode that can reach a static object, so zero hits is a proof rather than an absence of evidence. It
found 16 static Ult objects and **zero** interior references — every reference a `lea` of the base feeding an
Ult call.

Generalising: prefer the instrument whose **negative result is exhaustive over a closed set** (all
RIP-relative references) to one that pattern-matches an **open set** (all functions that might alias a
pointer). When only the open-set instrument exists, its negative result is a hypothesis, not a finding.

### Cross-title: the colour-state registers are decoded correctly — start elsewhere

Two lanes independently established this, and both results killed a live hypothesis in a *different*
lane. Recorded here rather than in either lane because the next agent to see a zero colour write mask
will reach for the same explanation.

**`MODE` decode is not remapped.** A tempting hypothesis on Astro Bot (#1459) was that prosper mis-maps
the `CB_COLOR_CONTROL` `MODE` enum — its live trace never once observed `MODE=1 (CB_NORMAL)`, only modes
0 and 6, which is not what an ordinary renderer looks like. **Blue Prince falsifies it**: that title
writes `MODE=1 CB_NORMAL` and renders correctly through the same decode. Any remapping hypothesis now
has to explain Blue Prince too, which no simple remap does.

**`CB_COLOR_CONTROL` tracking is not lossy.** An Oregon Trail investigation measured the indirect
register path directly: **~12,000 observed `CB_COLOR_CONTROL` writes**, `MODE=1` dominant at roughly
**7:1**, and — joined on **each draw's own `command_order`** — **zero mismatches across all 1,050
suppressed draws**. The register prosper believes is in force at a draw is the register the guest wrote.

**So "prosper is misreading MODE" starts from a losing position.** When a draw resolves `cwm=0`, the
productive question is **who wrote the zero and when**, not whether the decode is right.

**The join is the trap in that second result, and it is worth its own line.** The same measurement read
the *other* way — joining `PROSPER_REGWATCH` against `PROSPER_EXECLOG` by **stream adjacency** rather
than by `command_order` — reported that 1,049 of 1,050 suppressed draws resolved against a **future**
register write. Clean, compelling, and entirely false: the two instruments timestamp at different
pipeline stages (REGWATCH at command-processor *decode*, EXECLOG at *realization*) and the phases
interleave per submit. See instrument-trap 12; #1633's `order=` is what makes the honest join possible.

## The orchestration contract

### Orchestrator responsibilities

The orchestrator owns integration and scheduling, not every line of investigation. It should:

1. Read `CLAUDE.md`, this document, the title status document, and the active issue before assigning work.
2. Fetch `origin/master` and give every subagent a private worktree and a fresh branch from the exact remote head.
3. Assign one title or one precisely bounded shared-infrastructure question per subagent.
4. Prevent duplicated hypotheses by requiring agents to read the current issue evidence before running anything.
5. Keep each agent on an evidence ladder: retained offline artifact first, bounded live capture only when the artifact
   cannot answer the question, implementation only after a generic behavioral contract is identified.
6. Decide GPU scheduling. Ordinary correctness replays may overlap; reserve exclusivity only when measurements,
   memory pressure, or observed interference require it.
7. Review every proposed diff itself, including assumptions, scope, tests, `git diff --check`, and exact base/head.
8. Open short-lived PRs for proven progress, wait for every applicable CI job, confirm release publication skipped on
   an ordinary PR, then merge when authorized.
9. Keep issues and PR descriptions self-contained. Post exact commands, hashes, conclusions, and falsified hypotheses.
10. Update compatibility docs and representative screenshots when a title reaches a new visible checkpoint.
11. Rebase or restart long-lived investigation branches from current master before they drift.
12. Stop stale processes and keep evidence off the repository and off RAM-backed `/tmp`.

The current user explicitly authorized the orchestrator/subagent arrangement as pair programming and allowed the
independent-review step to be skipped for PRs produced by this coordinated work. The orchestrator still has to inspect
the complete diff, verify the exact head, wait for all CI, and merge deliberately. Treat this as engagement-specific
authorization: if a future user has not granted it, follow the independent-review default in `CLAUDE.md`.

### Subagent responsibilities

Every game subagent should receive and follow this contract:

- Work only in the assigned private worktree and worktree-local build directory.
- Start from the exact remote master SHA supplied by the orchestrator; report branch, head, merge-base, and status.
- Read the title issue and status docs before running a new experiment.
- State one falsifiable question before each run and the outcome that would distinguish the competing explanations.
- Prefer immutable F9/timeline/capsule evidence and offline `gpu_replay` over repeated game boots.
- Put captures, logs, screenshots, raw shaders, and scratch analysis under `~/`, never in git and never in `/tmp`.
- Never publish private absolute host paths. Public text uses `~/`, `<REPO_ROOT>`, `<WORKTREE>`, and `<DUMP_ROOT>`.
- Do not make title-address special cases. Shared GPU/recompiler behavior needs a generic contract and regression tests.
- Do not silently skip unsupported guest behavior. Retain fail-visible rejection until semantics are proven.
- Do not edit after the orchestrator asks for a freeze or design review.
- Report meaningful progress early: evidence paths/hashes, issue comment links, exact next blocker, and whether a code
  change is ready. Avoid holding a useful generic fix for days while exploring a later title-specific blocker.
- Commit and push a focused branch when the orchestrator approves implementation. The orchestrator owns PR creation,
  CI gating, merge, and cross-agent integration unless explicitly delegated otherwise.
- A visual milestone requires an unmodified frontend screenshot. Diagnostic substitutions may illustrate a hypothesis
  but are labeled diagnostic and never presented as progression evidence.

### Worktree and branch protocol

Never build or edit in the shared checkout. A typical assignment begins with:

```bash
git fetch origin master
git worktree add .claude/worktrees/<agent-title> \
  -b investigate/<title-question> origin/master
cd .claude/worktrees/<agent-title>
git status --short --branch
git rev-parse HEAD
```

Each worktree gets its own build directory. Put compiler temporaries on real disk:

```bash
mkdir -p prosper/build-linux/tmpdir
TMPDIR="$PWD/prosper/build-linux/tmpdir" \
  cmake -S prosper -B prosper/build-linux -DGAME_DUMP=<DUMP_ROOT>/<TITLE>-app0
TMPDIR="$PWD/prosper/build-linux/tmpdir" \
  cmake --build prosper/build-linux -j6
```

Before publishing, fetch master, inspect divergence, synchronize when needed, rerun the relevant checks on the exact
head, and report:

```text
branch / head / merge-base / worktree status
focused build and test commands with pass counts
artifact or live evidence hashes
git diff --check result
known limitations and deliberately unaffected paths
```

Do not keep coding on a branch whose useful part has already merged. Create a fresh branch from the new master for the
next hypothesis.

### GPU scheduling

Exclusive GPU ownership is **not** the default.

Safe to overlap in normal circumstances:

- `shader_inspect`, graph generation, disassembly, capture inspection, and CPU analysis;
- short deterministic correctness replays where timing is irrelevant;
- game-list or no-game frontend tests that only render light ImGui content;
- builds and non-Vulkan tests.

**Omitting `PROSPER_RENDER` does NOT make a run GPU-free.** This has been read as "no GPU" in
briefs and in lane planning, and it is wrong. `tools/boot_trace/boot_trace.cpp:285` gates only the
**live renderer** on `PROSPER_RENDER`; a few lines earlier, `:277` unconditionally calls
`prosper::frontend::register_live_compute()`, which executes the title's compute dispatches through
Vulkan. A "CPU-only" `boot_trace` run of PPSA19244 executed **~135,000 dispatches** over 310 s — a
real, sustained GPU consumer that any peer measuring frame time would have felt.

The genuinely Vulkan-free path is **`PROSPER_NO_COMPUTE=1`**, which selects the progression-only
no-op backend at `boot_trace.cpp:270-276` instead. It retains semantic dispatches and mutates no
guest GPU resources, so it is sound for CPU-side questions — asset/file I/O, pad, HLE call counts,
guest backtraces, boot progression — but **not** for anything whose result depends on compute
output. Validate it per investigation rather than assuming: on PPSA19244 the `NO_COMPUTE` arm was
confirmed to reach the same steady state (same ~23 draws/submit, same asset set, same stall) as the
compute-enabled arm before any conclusion was drawn from it.

**A "sustained all-clear" needs a NUMBER, and 120 s is too short.** Waiting for a peer's GPU work to
finish by checking that all five process names (`prosper-app boot_trace gpu_replay screenshot
screenshot_snap`, plus `/proc/<pid>/cwd` to attribute them) are clear is correct, but a *point-in-time*
check is worthless and a short window is nearly as bad: a lane running back-to-back captures has gaps
between its runs, and a window that fits inside one reports a false all-clear. Measured 2026-08-01: a
waiter requiring **120 s continuously clear** declared SUSTAINED ALL-CLEAR at 12:18:33 and the routes
lane's `screenshot_snap` was back **within seconds** — the window had fitted inside one inter-run gap.

Require **300 s minimum continuously clear** against a lane doing sequential capture runs, reset the
streak on **any** busy sample, and log what reset it so the wait is auditable rather than a guess.

**A before/after peer count is the right pattern and has a blind spot — know which one you are
relying on.** Sampling the foreign-process count immediately before and after a run, and marking the
result `DISCARD` if either is nonzero, makes contention *visible in the record* instead of invisible
in a mean. But a peer that starts **and finishes entirely inside** your run reads zero at both ends,
so "0 before, 0 after" is evidence of a clean run, not proof of one. It is strictly weaker the longer
your run is and the shorter the peer's. The gate width and the before/after check protect different
things: the **gate** decides whether you started into an inter-run gap, the **before/after** decides
whether contention was still present at the boundaries. Neither subsumes the other, so use both, and
for a genuine timing claim add a mid-run sample rather than trusting the endpoints.

**Scope this discipline to the claims it actually protects.** It exists for *timing* results —
FPS, submits/s, throughput A/Bs — where a contended sample is indistinguishable from a real
regression. A **qualitative** result is contention-invariant by construction: a contended run cannot
fabricate a correctly-rendered logo, and it cannot change which field a deterministic CPU-side
validator rejects. When a gating standard tightens mid-session, re-check *timing* claims made under
the older, looser gate; do not reflexively withdraw qualitative ones, and do not spend a GPU slot a
queued lane needs re-confirming a mechanism that already guarantees the answer. Both directions are
errors: applying it indiscriminately withdraws sound results, ignoring it leaves an
`UNCONTENDED: admissible` line standing on a window too short to have earned it.

So the honest classification is:

| run | Vulkan? |
|---|---|
| `PROSPER_RENDER=1` | yes — renderer **and** compute |
| no `PROSPER_RENDER` | **yes — compute still runs** |
| no `PROSPER_RENDER` + `PROSPER_NO_COMPUTE=1` | no |
| `gpu_replay`, `screenshot`, `prosper-app` | yes |
| `gpu_timeline`, `self_dump`, `shader_histo`, `tools/re/*`, `tools/il2cpp/*` | no |

Ask the orchestrator for an exclusive lease when:

- profiling or reporting FPS/frame time;
- comparing small performance changes where contention would swamp the signal;
- a capture/replay has unusually high VRAM or host-memory demand;
- concurrent jobs have already caused timeouts, device loss, or unstable images;
- the user specifically asks to watch one live run without interference.

An exclusive lease has an owner, purpose, start time, and bounded expected duration. The owner announces when it is
released. Correctness replays do not become exclusive merely because they use Vulkan.

Before starting a GPU run, inspect existing processes. A Prosper process older than roughly 30 minutes is presumed
stale because known tests should finish in minutes. Resolve the exact command and owner, then terminate that exact
process safely. Never use a broad pattern that could kill another agent's unrelated work.

**Count with `pgrep -x`. Both `ps | grep` idioms are wrong, in opposite directions.** This is a safety rule, not a
style preference: a wrong count either blocks you from a free GPU or makes you kill a peer's live run.

| Idiom | Failure | Observed |
|---|---|---|
| `ps aux \| grep -cE "prosper-app\|boot_trace"` | **Over-counts** — matches the Bash wrapper shell and any `sleep` watcher whose command line contains the literal | reported **23** when the true count was **0** |
| `ps -eo comm \| grep -c '^prosper-app$'` | **Under-counts** — `ps` pads the `comm` column, so the `$` anchor never matches | reported **0** against a **live** process |

The second is the dangerous one. On 2026-07-31 an agent reported "GPU released" from that idiom repeatedly; a
blind `pkill -f prosper-app` on the strength of one such reading would have killed a **successor agent's run 35
seconds after it started**. It was caught only because the release check was re-run with `pgrep -x`.

Use:

```bash
pgrep -c -x prosper-app        # count, exact name
pgrep -ax prosper-app          # list with full command lines
readlink /proc/<pid>/cwd       # whose worktree is this?
tr '\0' '\n' < /proc/<pid>/environ | grep PROSPER_   # and whose run?
```

Before terminating anything, confirm the exact PID **and** that its `cwd`/`environ` identify it as yours. Never
kill from a pattern count. Note `comm` is truncated to 15 characters by the kernel, so a longer binary name needs
`pgrep -f` with a **bracketed** pattern (`pgrep -f "prosper[-]app"`) so it cannot match your own shell.

**`pgrep -x screenshot` does not see a snapshot run.** `capture_content` copies the frontend to
`$TMPDIR/snap_<rand>/screenshot_snap` and executes *that*, so every `snapshot.py` profile, `verify`, and
`check` runs under the name **`screenshot_snap`**. Checking only the four obvious consumers reports a free
GPU while a peer is two full boots into a `verify` — the longest-running GPU job in this repo. Include it:

```bash
for n in prosper-app boot_trace gpu_replay screenshot screenshot_snap; do
  printf '%s: %s\n' "$n" "$(pgrep -x "$n" | wc -l)"
done
```

Observed 2026-08-01: a lane's own live profile showed `screenshot: 0` and `screenshot_snap: 1`. The same
applies to any harness that runs a copied or renamed binary — resolve the *executed* name, not the built one.

**A clean count is a snapshot, not a lease.** On the same day, a sustained-idle watch confirmed 120 s with
zero consumers, and a peer lane started a `boot_trace` 60 s later, concurrent with the run that idle check
had authorized. Sustained idle lowers the odds of landing inside a peer's inter-run gap; it cannot reserve
anything. If overlap would invalidate your measurement, say so and re-run — do not kill the peer.

### Fast evidence loop

For rendering correctness, use this order:

1. Reuse an existing `.prgcap`, `.prgbundle`, timeline, raw shader, or issue attachment.
2. Run `gpu_replay --inspect-only`, graph/resource dumps, shader coverage, and source inspection.
3. Replay a bounded operation prefix or one draw/compute stage.
4. Use a narrowly scoped diagnostic substitution with a selector-miss control.
5. Add a reusable diagnostic seam only if current tools cannot separate the remaining stages.
6. Run the title live only to capture missing state or validate the integrated fix.
7. Measure performance only after correctness and with an exclusive GPU window when precision matters.

Every A/B must hold incidental behavior constant. For example, `PROSPER_TESTTEX` enables a generic CPU diagnostic
copy path, so a binding-miss run with the same environment is the control for a binding-hit run.

### Testing, PRs, releases, and screenshots

- Focused tests are part of iteration. The full snapshot matrix is optional for ordinary development and PRs; the PR
  author decides whether it is useful.
- The snapshot matrix is mandatory before every release. A release delay is acceptable; day-to-day development should
  not be delayed solely to keep master regression-free.
- Master may regress during heavy development. A correct fix may intentionally expose a different title regression.
  Record the tradeoff rather than treating a permanently green master as an end in itself.
- Artifacts are uploaded only for a release. A release is created only for a Git release tag. Ordinary PR CI must show
  the release-publication job skipped.
- Keep PRs short-lived so other agents inherit generic improvements quickly.
- **Two mechanical holes in the merge gate, both of which let a PR read as satisfied when it is not.**
  - **A registered review is not the same as a comment.** `gh pr comment` creates an *issue* comment;
    only `gh pr review --comment` appears in `gh pr view --json reviews` / `pulls/N/reviews`. A review
    posted the first way is fully visible on the PR page and invisible to the gate, so the findings
    are right there while the PR reads as unreviewed.
  - **Never gate on `reviewDecision`; it is empty here for TWO independent reasons.** (1) A
    `COMMENTED` review does not populate it **regardless of who posts** — true even for a third-party
    reviewer, and the reason that survives any change to how reviews are authored. (2) Separately,
    `gh` is normally authenticated **as the PR author**, so `--approve` is not available. Both matter:
    someone "fixing" the gate by provisioning a second `gh` identity would find `reviewDecision`
    **still empty** and wrongly conclude this note was wrong.
    *Provenance, for both halves, because stating it for only one is the asymmetry this page exists to
    prevent:* **neither** was measured here. What WAS observed is that `reviewDecision` is empty with
    `COMMENTED` reviews present, and that `gh api user` matches the PR author — consistent with both
    claims but isolating neither, since every review observed came from the author account. That
    `COMMENTED` never populates `reviewDecision`, and that GitHub rejects self-approval, are both
    **documented platform behaviour**. `--approve` was deliberately **not executed**: running it would
    either fail or register an approval nobody intended.
  - **A rebase detaches every review from head, and the gate cannot see it.** `reviews` stays
    non-empty and still reports `state=COMMENTED`/`APPROVED`, but its `commit_id` points at a SHA no
    longer on the branch. State the consequence precisely: a stale `commit_id` proves the review **no
    longer binds to head** — it does *not* prove the content changed, and for a clean rebase the tree
    is usually byte-identical. The remedy is the same either way: compare each review's `commit_id`
    against the current head, and if it is stale, establish by **content** whether the approval still
    applies (`git diff <reviewed-sha> <head>`) rather than assuming in either direction. This is
    **trap 11 in the other direction** — there a squash made merged content look unmerged by SHA;
    here a rebase makes reviewed content look unreviewed by SHA. Both resolve the same way: verify by
    content, not by SHA. The same applies to CI — re-check bound to the exact current SHA, never to
    the PR number.
  - **Reviewers: bind your post to head at the write site, rather than relying on the merger to catch
    a stale one afterwards.** Read `headRefOid` immediately before posting and refuse if it moved:
    prevention costs one API call, detection costs a round trip and depends on the merger remembering.
    This is not hypothetical — on #1687 the head moved **twice after the branch was called frozen**,
    and a reviewer guarding on `headRefOid` correctly refused both times. Posting unconditionally
    would have left `reviews[]` showing a perfectly healthy row bound to code that had moved on.
- The orchestrator opens the PR with a self-contained behavioral contract, issue links, evidence, exact tests, risks,
  and known limitations. Wait for all Linux, Windows, and macOS checks before merging.
- When a game changes from black to visible content, reaches title, reaches gameplay, or materially improves visuals,
  update `README.md`/compatibility docs as appropriate and attach direct unmodified frontend captures. A black frame
  needs no screenshot. A checker/forced-state/debug draw may be attached only as clearly labeled diagnostic evidence.

### The environment is part of the system under test

A sound, log line, or visual artifact that appears to come from the program may be **the desktop responding
to the program's output**, not the program. Rule the environment out before building a theory on top of it.

Worked example (#1630). A short chime was heard repeatedly during library-view test runs and was taken as
evidence of an audio defect — a wrong sample rate, a truncated track, a fade discontinuity. It was none of
those. It was **KDE's terminal bell** (`ocean/stereo/bell-window-system.oga`): Konsole's default bell mode
is "System Notification", so any `0x07` reaching the terminal plays a real sound. Synthetic arrow keys from
an `xdotool`-driven route that miss the app's window land on the shell instead, where readline rings the
bell at the start of a line.

What made it convincing is exactly what made it misleading: the user described it as sounding like "a real
sound effect, identical every time", which reads as strong evidence for real decoded audio being replayed.
It sounded like a real effect because it *is* one — just not ours. **A property that seems to discriminate
between your hypotheses can be equally consistent with one you have not considered.**

Practical consequences for any routed run on a developer's desktop:

- Send every byte of a run's output to a file, never to a terminal, or strip `\007` from it. Free, and it
  removes the confound permanently.
- Guard synthetic input on the target window still existing (`xdotool getwindowname "$WID"`), so keys
  cannot leak to whatever has focus after the app exits.
- Prefer an instrument over an ear or an eye. The default sink can be recorded
  (`parecord --device=<sink>.monitor`) and checked for non-silence, which turns "did it make a sound?" into
  a measurement. Validate the instrument on a silent baseline first.
- When output and observation disagree, suspect the harness before the subject — a capture synchronized to
  a log line can easily record a *different* moment than the line it is labelled with.

## Recently merged foundation

These changes are already on current master and should not be reimplemented:

| PR | Merge commit | Result |
|---|---|---|
| [#1561](https://github.com/mattias800/prosper/pull/1561) | `0b20522a9d8e212f2813d75c299d33b198908aee` | Materializes exact scalar 16-bit buffer tails. |
| [#1562](https://github.com/mattias800/prosper/pull/1562) | `cee8b1525de226551814fe55a64eb1ffa19e513c` | Records Dragon Quest VII name confirmation. |
| [#1563](https://github.com/mattias800/prosper/pull/1563) | `7f89e60b4d70f4f843be22202d396cf75b0762ed` | Records Dragon Quest VII first-run onboarding. |
| [#1564](https://github.com/mattias800/prosper/pull/1564) | `1634c1e7597731b4d2a5da8d616e2755ef202518` | Structures scalar multi-loops; live Plucky stage `0x3017450000` executes. |
| [#1565](https://github.com/mattias800/prosper/pull/1565) | `2d6ff9fb800d5e8689209958c17963d4f49b6a46` | Captures frames at exact guest-log phases; enabled deterministic Astro world-map capture. |
| [#1566](https://github.com/mattias800/prosper/pull/1566) | `81a9548e4cf79f9b653bbf78acb38e149f0c373e` | Composes multiple structured preludes with counted-loop lowering. |
| [#1572](https://github.com/mattias800/prosper/pull/1572) | `276b8f92` | Admits barrier-spanning VCC branches **proved** workgroup-uniform. Plucky stage `0x3017460000` reaches `unsupported=0`. |
| [#1574](https://github.com/mattias800/prosper/pull/1574) | `24a98629` | `gpu_replay --inspect-only` reports resolved `SPI_PS_INPUT_CNTL` linkage, so a synthesized-constant varying is visible offline. |
| [#1575](https://github.com/mattias800/prosper/pull/1575) | `37dfe752` | `shader_inspect` reports a missing resource table as `undetermined` (exit 3), not as a shader rejection. |

Current master is **162/162 ctest** when configured against `PPSA24651`, with all of the above composed.

PR #1566's focused translator/render set was:

```text
recompile_coverage
rdna2_spirv_struct
game_compute_exec
game_compute_exec_no_adaptive_storage_result
rdna2_to_spirv_exec
recompiled_fragment_render
recompiled_shaders_render
```

All seven passed locally, all six platform CI jobs passed, and release publication skipped. The exact retained Plucky
shader moved from four unsupported control-flow instructions to one deliberately rejected barrier-crossing branch.

## Lane A: Astro Bot world map

### Public state

- Issue: [#1459 — world map renders as a dark grayscale pattern](https://github.com/mattias800/prosper/issues/1459).
- Existing status background: `docs/ASTROBOT_LINUX_HANDOFF_2026_07_19.md`.
- Frozen outgoing branch: `investigate/astro-worldmap-closure`, clean at `2d6ff9f`; it contains no source changes and
  is behind current master. Start a fresh branch from `81a9548e` rather than extending it blindly.

Durable evidence comments:

- [submit 6279 bounded failure classification](https://github.com/mattias800/prosper/issues/1459#issuecomment-5141854392)
- [submit 6284 dependency map](https://github.com/mattias800/prosper/issues/1459#issuecomment-5141880607)
- [compute-writer closure source audit](https://github.com/mattias800/prosper/issues/1459#issuecomment-5141933098)
- [seven-submit relevance result and pivot](https://github.com/mattias800/prosper/issues/1459#issuecomment-5142003672)

### Retained evidence

Evidence root:

```text
~/.local/state/prosper/evidence/astro-worldmap-c5698376-froute.KyRLdP/
```

Important files:

```text
worldmap.prgbundle
closure-analysis-2d6ff9fb/submit-{6278..6284}.prgcap
closure-analysis-2d6ff9fb/submit-{6278..6284}.inspect-all.log
closure-analysis-2d6ff9fb/submit-{6278..6284}.graph.json
closure-analysis-2d6ff9fb/current-master-replay.log
closure-analysis-2d6ff9fb/current-master.bmp
closure-analysis-2d6ff9fb/submit-6284-op19-b5-captured.bin
closure-analysis-2d6ff9fb/submit-6284-op19-b6-captured.bin
closure-analysis-2d6ff9fb/submit-6284-compute8-stored.spv
closure-analysis-2d6ff9fb/submit-6284-compute15-stored.spv
closure-analysis-2d6ff9fb/submit-6284-compute15-current.spv
closure-analysis-2d6ff9fb/submit-6284-compute15-stored-v-current.diff
```

The b6 seed SHA-256 is `571bd231a552e2613817c87ca2d7fe911cbda679dd41fbf77c4f80fe0c35ab8b`.
The current full replay hash recorded by the outgoing agent is `28b5ea9a3fbd2ca9`.

The route used PR #1565's exact marker gate:

```text
LevelDocument Loaded: worldmap [worldmap]
```

It captured seven submits after one present, ending at submit 6284. The selected world-map output is still dark/near
black, so no progression screenshot was published.

### Proven facts

1. Submit 6279's apparent Vulkan failure is benign. Its only unrealized operation is an intentional no-effect draw:
   color write mask zero and no depth/stencil write. Both shaders have zero unsupported instructions.
2. Submit 6284's 171 unrealized operations are also benign: 170 no-effect operations and one zero-vertex draw.
   There are no ShaderRecompile, DescriptorContract, or Vulkan rejects.
3. Only one unresolved temporal leaf reaches final scanout:
   - op19 compute program `0x500758300`, raw hash `b88321fae48353e1`, reads b6 and writes b7 at
     `0x556760000`, a 1x1 Float32x4 tile-27 image backed by a 64 KiB footprint;
   - op221 compute program `0x500656100`, hash `5ad826b552f23191`, later reads the same allocation at b6;
   - final op236/draw206 fragment hash `5cb70f55e3902078` reads it at b49.
4. There is no earlier access or overlapping color target in submits 6278 through 6284. The graph leaf means “no
   producer in this bundle,” not “no captured backing.”
5. The full 65,536-byte b6 allocation is captured, materialized, detiled, and seeded. Op19's b7 writeback mutates the
   shared capture-owned resource instance, so later consumers observe it.
6. Extending runtime producer history to reflected compute writers would improve future closure diagnostics but
   cannot change this retained bundle's pixels. Do not implement that abandoned proposal as an Astro visual fix.
7. Op19's inputs at texel zero are exact Float32 values `(1, -1, 0.5, 0.180000007)`, with b2=1.0 and b3=0.1.
   Stored shader math yields b7 `(0.8, -1, 0.5, 0.180000007)`.
8. Current raw op19 recompilation is byte-identical to stored: 1,053 words, hash `b88321fae48353e1`.
9. Op221 only reads this 1x1 allocation and writes b15/b16 at `0x5460b000`, a 1920x1080 R11G11B10 image. It does
   not overwrite the 1x1 allocation. Both op221 and final op236 use only the control value's x component, `0.8`.
10. The 1x1 value is sane and finite. Black does not begin at op19. Current evidence has not yet distinguished
    op221's 1920x1080 result from later/final composition.

### Falsified: the R11G11B10 storage fork

Op221's stored (14,543 words, `5ad826b552f23191`) and current-raw (12,019 words, `4d55000d1c32f046`) SPIR-V are
not binary-identical, and the difference does concentrate in R11G11B10 storage lowering — stored packs float RGB
into an R32ui image, current writes a float vector with `StorageImageWriteWithoutFormat`.

**This is not a defect.** Forcing each module explicitly with `--override-compute-spv 15 PATH` proved the two arms
take genuinely different backend paths (`native-storage=0` "exact packed R11G11B10 storage via R32_UINT" versus
`native-storage=1` typed float) and still produce **byte-identical** output `28b5ea9a3fbd2ca9`. The divergence is
by design: `gpu_executor.cpp` forces `native_storage_format_support=0` whenever a capture is bound, keeping stored
modules device-independent. Do not spend a PR here.

Also falsified: seed-skip/poison corruption (`PROSPER_NO_SKIP_SEED=1` gives an identical hash); op221 itself
(it writes zeros, but from a genuinely empty upstream); and the b49 1x1 control (draw 206's only real image input
is `0x5420f0000`).

### ANSWERED: the zero colour write mask is `CB_COLOR_CONTROL.MODE=DISABLE`, and prosper is faithful

Measured 2026-08-01 on exact master `f7831d8a` with a fresh **v43** world-map capture (the v42 bundle
reported `color-state unavailable`, which is why this could not be settled before) plus a
`PROSPER_REGWATCH` write-event count. Both halves agree and the chain is now complete.

**The masks are exonerated.** Across all 207 colour-state records in the captured frame, **no** register
is absent (`present=0`: zero records) and only 5 have a zero mask. Typical records are
`target-mask=1:00000737 shader-mask=1:0000000f effective=07` — healthy, non-zero.

**The disable is the MODE field.** 201 of 207 records hold `CB_COLOR_CONTROL = 0x00cc0000`
(**MODE=0, CB_DISABLE**), 6 hold `0x00cc0060` (MODE=6, DCC_DECOMPRESS), and **none** holds MODE=1
(CB_NORMAL). On the G-buffer `0x520440000` specifically: **136 draws at `cwm=0`, every one `mode=0`**
(the earlier "28 draws" figure understated it ~5x). `0x00cc0000` is the standard GFX10 hardware reset
for this register — ROP3 `0xCC` SRCCOPY, MODE=0 — and is not a literal anywhere in `src/`.

**The guest never enables normal colour writes.** `PROSPER_REGWATCH=Cx:0x202` over the routed run
observed **17,533 `CB_COLOR_CONTROL` writes, all on the indirect path**, with exactly three distinct
values differing *only* in the MODE field:

| value | MODE | count |
|---|---|---|
| `0x00cc0000` | 0 — CB_DISABLE | 7,859 |
| `0x00cc0020` | 2 — CB_ELIMINATE_FAST_CLEAR | 6,109 |
| `0x00cc0060` | 6 — CB_DCC_DECOMPRESS | 3,565 |
| — | **1 — CB_NORMAL** | **0** |

`CB_TARGET_MASK` (32,134 writes) and `CB_SHADER_MASK` (67,018) are written heavily, so the register
stream is alive and being decoded; the three coherent MODE values argue the same. This is DISABLE plus
two *decompression/maintenance* modes — a frame doing colour-buffer maintenance and never ordinary
shading.

**prosper's handling is correct and deliberate.** `render_state.cpp` computes
`color_write_mask = CB_TARGET_MASK & CB_SHADER_MASK`, then zeroes it when `MODE == DISABLE`
(`ps.color_write_mask = 0; ... for (auto& target : ps.color_targets) target.write_mask = 0;`), with a
comment stating that DISABLE suppresses colour writes but not the draw. Guest writes MODE=0 -> prosper
zeroes the mask -> `cwm=0` -> black. Every step is faithful.

**So the frontier moves upstream, off the colour state entirely.** The question is no longer "who wrote
the zero" — the guest did, 7,859 times — but **why Astro's world map never issues a CB_NORMAL pass**.
Do not re-investigate the masks, the MODE decode, or the register fold: decode correctness is settled
cross-title (Blue Prince renders through this path at MODE=1; Oregon Trail measured ~12,000 writes with
zero mismatches across 1,050 suppressed draws).

**The bound was checked and closed.** `PROSPER_REGWATCH` observes writes only in streams prosper
actually *folds*, so a never-folded submit would be invisible to it **and** to the renderer — the #305
class one lane over. Both halves come back clean:

* **Astro never uses the variant submit path.** Scanning every module in the dump, only `eboot.bin`
  touches the submit imports at all, and it imports `UglJIZjGssM` (ordinary `sceAgcDriverSubmitDcb`)
  and **zero** `w1KFAHVqpaU` (`SubmitDcbFinal`). The `UglJIZjGssM` hit is the scan's own control. So
  the cross-thread mechanism cannot apply here.
* **Every submit folds completely.** Across two full routed runs: **zero** `SHORT FOLD`, **zero**
  `SUBSTITUTED STREAM ADDRESS`, **zero** `invalid arg9` / `no PM4 stream` (the #1665 instruments are
  unconditional, so the existing logs answered this without a new run).

So `CONFIDENCE: HIGH` on both halves: prosper faithfully reproduces what it observes, and what it
observes is everything the guest submits. **The cause is upstream of GPU state entirely** — the world
map's render pass is never set up to shade. That is a scene/pipeline question, not a register one.

### Superseded: "a frame-wide zero colour write mask" (the framing above answers it)

The world map is **not** a "dark grayscale pattern" — contrast-stretching the output shows a uniform diagonal
sawtooth with **zero scene content**. The prior "min 0 / max 13 / mean 5.97" reading is a content-free ramp.

**No geometry draw writes colour anywhere in the captured frame.** Across 7 submits / 43 realized draws, 28 draws
target the G-buffer `0x520440000` with `cwm=0` *and* `cwm1=0` while carrying real indexed geometry
(2,776–5,736 verts) and a **140,825-word** material fragment shader; ~170 further draws are suppressed as
no-effect for the same reason. Only 3 draws write full RGBA. Meanwhile the *captured* G-buffer holds 12,267,587
non-zero bytes of real content.

`render_state.cpp` resolves an **absent** `CB_TARGET_MASK`/`CB_SHADER_MASK` to **write-all**, so `cwm=0` means one
of those registers is **present with value zero**, or `MODE` is `DISABLE`. The live trace never once observed
`MODE=1 (CB_NORMAL)` — only mode 0 and mode 6 — which is not what an ordinary renderer looks like.

A tempting prior was the Gen5 stale-register-fold class. It was **investigated and not confirmed**, and the prior
itself turned out to rest on a superseded diagnosis:

- `SetRegsIndirect` reads a flat `{offset,value}` array and drops only `offset >= kRegOffsetLimit`, so the
  present-and-zero *capability* is real and generic.
- But #1364's "second record format / bit31-tag misparse" family diagnosis was **overturned**. The real mechanism
  was an HLE **success stub**: the SDK interpolant helper `dbOlWdppb4o` advertised 32 output records it never
  wrote, leaving **host stack residue** whose dword halves fold as present-and-zero registers. Fixed by #1368
  and #1411; current master emits zero `[dbbase-clobber]` events on 380 s+ routes.
- PRs #1344 and #1363 are downstream *symptom recoveries*, not fold-level invariants.
- Residue historically lands in the **low** offset cluster (DB `0x00-0x1F`, scissors `0x0C-0x0D`), whereas
  `CB_TARGET_MASK` is `0x8E`, `CB_SHADER_MASK` `0x8F`, and `CB_COLOR_CONTROL` `0x202`. That is counter-evidence.
- `PROSPER_DBBASETRACE` **cannot decide this**: it is hardcoded to DB offsets and only fires on a
  nonzero-then-zero-within-one-array signature, so it never fires for a register the guest never writes.

**Next Astro step.** Capture **v43** (PR #1576) retains the raw colour-state triple per draw with presence flags
and surfaces it in `--inspect`, which is what makes `cwm=0` attributable offline. The retained bundle is v42, so
it reports `color-state unavailable` — a **fresh v43 capture** is required. Take one, then determine whether the
zero masks are genuine guest intent or a prosper defect. If they are genuine intent, "real content exists but
nothing reaches the screen" via small mipped 4 KiB surfaces becomes the natural next suspect.

Do not recapture the world map for any other reason, revisit submit 6279, or implement compute-writer closure:
b6 is already captured, materialized, detiled and seeded, and no earlier producer exists in the bundle.

## Lane B: Dragon Quest VII overexposed composite

### Public state

- Issue: [#1486 — 4K composite is overexposed](https://github.com/mattias800/prosper/issues/1486).
- Status document: `docs/DRAGON_QUEST_STATUS.md`.
- The outgoing worktree was clean with no new repository changes. Its built replay tool and retained capsule were
  based on `2d6ff9f`; compare relevant tile/sampler source with current master before treating byte parity as final.

Durable comments:

- [corrected op89/op90 diagnosis](https://github.com/mattias800/prosper/issues/1486#issuecomment-5141895497)
- [exact op116 to op117 corruption boundary](https://github.com/mattias800/prosper/issues/1486#issuecomment-5141942105)
- [raw recompile result and b36 metadata](https://github.com/mattias800/prosper/issues/1486#issuecomment-5141986842)
- [controlled LUT substitution A/B](https://github.com/mattias800/prosper/issues/1486#issuecomment-5142022769)
- [full-volume offline decode and sampling pivot](https://github.com/mattias800/prosper/issues/1486#issuecomment-5142103444)

### Retained evidence

```text
~/agent-tmp/dq-cap3/black.prgcap
  SHA-256 c7a863d69823964a2f18ced544385caa9986c04c419da2f449524b330cf237cf
  capture v38, submit 1060, 124 operations

~/dq-submit1060-bisect.goFHHE/
~/dq-op117-lut-ab.xEjqqT/
~/dq-op117-lut-decode.WjwK2H/
```

The A/B manifest SHA-256 is `f2f44602c178226539ba456d9a9879c1ab5b04634f3cb590010434577771290d`.

### The "exact corruption boundary" is a BISECT ARTIFACT — retained as a warning

> **FALSIFIED 2026-07-31.** The op116 -> op117 "corruption boundary" below, on which this entire lane was
> founded, **does not exist**. `--through-operation N` renders **the last executed draw target**, so
> consecutive cutoffs display *different surfaces*. The retained logs show it plainly — every cutoff is a
> different resolution: op104 30x34, op111 480x270, op114 960x1080, **op116 1920x1080, op117 3840x2160**.
>
> **op116 and op117 were never the same buffer, so the transition between them cannot be a corruption event.**
>
> op116's identity is pinned independently: it is a *dispatch*, so its output selects op115's target
> `0x3083db0000` = draw90's `b35`, and its content (R .0124 / G .0443 / B .1865) matches `tap-142` —
> draw90's own sample of b35 — to three decimals. **op116 is a post-process input that draw90 consumes.
> It is dark because such buffers are dark**, not because it is the last good frame before corruption.
>
> Do not re-derive a boundary from prefix replay without first confirming both prefixes render the **same
> target at the same resolution**. This is the single most expensive mistake recorded in this document: it
> shaped every hypothesis in the lane for two sessions.

The original (now-falsified) table, retained only so the claim is recognisable if it resurfaces:

| Prefix | Output | Hash / metrics |
|---|---|---|
| op116 | coherent dark ocean, 1920x1080 | replay `bee945066f3fc113`; BMP SHA `1273a16acea96c465db75021903cd0dadf2f16b1b3fa37e84296a0668866806e` |
| op117 | first overexposed/cyan frame, 3840x2160 | replay `31d794e48c2860dc`; BMP SHA `4dcf4f64ea9ed8e6901a92735b76cfa92bf41845c1a62d0f6818175a77728198` |
| op118 | orange overexposure | luma about 0.819; about 80.3% bright pixels |
| op122 | final opaque black overlay | not the first corruption |

Op117 is source draw90/order1436911. It is a realized fullscreen indexed triangle targeting `0x30867e0000`,
3840x2160 format44, write mask `f`, with blend/depth/stencil disabled.

Inputs:

- b34/b37/b38 from op88 at `0x308cfc0000`, 4K;
- b35 from op115 at `0x3083db0000`, 1920x1080;
- b36 grading LUT at `0x3021fe0000`;
- VS b7 from op103.

Stored VS/FS hashes are `6cde109bb55150ba` and `2810ab556bbfbc4d`.

`--recompile-raw --through-operation 117` is pixel-identical to stored replay. A no-render inspection changed the
installed VS hash to `ed25159f6e9d0b87`, and the mass path swaps VS and FS atomically, proving the draw used regenerated
stages. Stale stored shader translation is not the cause.

### Binding 36 identity

| Property | Value |
|---|---|
| Address / extent | `0x3021fe0000`, 32x32x32 |
| Format | enum21 `Unorm2_10_10_10`, four components |
| Declared / captured | 131,072 bytes / 2,097,152-byte footprint |
| Captured content | 107,001 nonzero bytes; hash `b53c7e7c2e9910e3` |
| Tile mode | 27, `SW_64KB_R_X` |
| Address modes / swizzle | 2/2/2; 4/5/6/7 |
| Filters | 1/1/1 |
| DCC metadata | enabled; address `0x306e9b0000`; 131,072 bytes of `0xffffffff`; hash `3735b73346670383` |

All-`0xff` metadata selects the renderer's ordinary uncompressed-base path. Current evidence does not indicate that
compressed blocks need decoding.

### Controlled LUT A/B

All variants reached op117 and used the same global diagnostic-copy path. Binding999 was the selector-miss control.

| Case | Output | Meaning |
|---|---|---|
| Baseline | hash `31d794e48c2860dc`; RGB mean 0.746193; luma 0.742344; 51.96% pixels at luma >=0.75 | Overexposed reference. |
| Selector miss, draw90/b999 | byte-identical to baseline | Diagnostic-copy side effects are excluded. |
| Checker, draw90/b36 | hash `7e9bec1982cda52e`; RGB mean 0.436836; luma 0.498721; no pixels >=0.75 | Every pixel changes; scene structure remains. |
| Zero, draw90/b36 | hash `ccc433ff6d980383`; one-color all-black frame | b36 materially drives the complete composite. |

The binding is active, its 3D view reaches the shader, and the selector is correct. Do not repeat unused-binding,
wrong-draw, or generic diagnostic-copy hypotheses.

### Full-volume decode result

The existing pre-detile resource dump exported the 2 MiB tiled allocation without a GPU replay. A temporary offline
analyzer called the project's own `detile_volume(..., 32, 32, 32, tile27, 4)` and the renderer's packed conversion.

Hashes:

```text
tiled allocation  60c100ff926e6ae226802e12b1510c54047b1472bbca120f6a47618fe9a0cf1b
linear packed     e9de4555ad4b65ea21de498c6d00443741773efa19269bfd98406ef64e61bfe1
linear RGBA8      eb27c3589d67553f86bf6f833894cee5f4ab3970c3b0bf2e0a51873ed0c89dca
metrics           b606c230b3ce953c9158015eacd5fe8eaf15eb9e7dbd223358bceb6d99abc257
z-slice sheet     e928a699a160757b6605c11a289633851440cb84279c989feeb2fd91371e31cb
```

Decoded channel metrics:

| Channel | Min | Max | Mean | Unique values |
|---|---:|---:|---:|---:|
| R | 0 | 255 | 176.9178 | 255 |
| G | 0 | 255 | 178.4504 | 254 |
| B | 0 | 255 | 153.9832 | 250 |
| A | 0 | 0 | 0 | 1 |

Mean adjacent differences are small and smooth:

| Axis | R | G | B |
|---|---:|---:|---:|
| X | 5.0448 | 2.8005 | 3.1537 |
| Y | 3.9890 | 5.1508 | 3.9264 |
| Z | 3.4532 | 3.5657 | 6.3019 |

The 32-slice contact sheet is visually coherent, with smooth RGB transfer surfaces and no checkerboard, tile-sized
scrambling, alternating bad slices, or discontinuity at 8/16-voxel boundaries. Tile27 detiling and packed
10:10:10:2 unpacking are now unlikely primary causes. Before citing exact runtime parity, confirm that current
master's tile conversion source matches the tested `2d6ff9f` version.

### Binding 36 is exonerated — do not re-open it

A point-versus-linear A/B produced `linear` byte-identical to baseline **because the T# already reports
`filt=1/1/1`**, with `point` (`b005d9f5045a2f6d`) differing on 8,243,869 of 8,294,400 pixels. Both arms provably
took distinct paths, so this is a real negative, not a broken experiment.

`PROSPER_FS_TAP` measured draw90's actual sample coordinates as **(0.385, 0.536, 0.688)** — none saturated — and
the LUT correctly returns (0.424, 0.768, 0.975) there. draw90's FS is a faithful UE4 tonemapper with
`unsupported=0`. Ruled out and not to be re-run: LUT contents, tile27 volume detile, DCC, 10:10:10:2 unpack,
binding/view/3D-coordinate wiring, dimensional opcode, dmask, swizzle, address modes, sampler filtering, and
draw90's shader translation.

The LUT's corners look absurd as an RGB cube (pure R gives 255,224,190) but are exactly right for UE4's
**log-encoded `CombineLUTs`** volume.

**Caution:** the older "`--recompile-raw` at op117 is byte-identical" result **predates a `rdna2_to_spirv.cpp`
change** and must be re-measured before being cited again. Verify source parity by hashing the specific
translation units rather than assuming.

### Also cleared: the auto-exposure producer and delivery

- **Producer format** — `b15` is `fmt=4 nc=4` Float16x4, `declared=16320 = 60x34x8`, seed tagged `rgba16f`. Exact
  agreement, so this is **not** a #773-class native-format loss.
- **Producer content** — draw77 writes R .0973 / G .3472 / B .9731, luma `.33929`, with `rgb_nonblack=2040`
  (all texels) and `HIT` on every sample.
- **Stale seed** — falsified; the near-black buffer is draw79 (operation 105, *after* op103), which is capture
  ordering rather than a bug.
- **Delivery** — `1=param1`, a real `PARAM1` export consumed at `pc=0177`. No draw90 input is a
  `ConstantDefault` (`valid=ffffffff`, only `param0..3`), and every PS attribute read falls inside its VS
  export's EN mask. Confirmed with the #1574 seam.

The eye-adaptation buffer reporting average luminance **629** is **not** the cause, and the arithmetic shows why:
too-high average luminance implies too-*small* exposure implies a too-**dark** frame, but the frame is too
**bright**. Against the oracle the error is ~1–2 stops (a factor of **2–4**), nowhere near the ~8192x a broken
exposure implies.

Two retractions worth preserving so they are not re-derived: a "multiplier is exactly 1.0" claim derived by
inverting *means* through a `log` is invalid under Jensen's inequality — it was re-established by direct
measurement instead (`tap-177` uniform at `0.984314`, one distinct value across all 8.29M pixels). And a
"~14x amplification" in the post chain was an artifact of comparing **different registers** via `PROSPER_FS_TAP`'s
`(dst..dst+3)` export window.

### Not bloom: film grain and chromatic aberration

pc=214–256 decodes to literals summing to `0.99999999` (Rec.601 luma), then `1/(2π)` → `v_sin_f32` (which prosper
lowers as `sin(x·2π)`, cancelling it) → `493013.0` → `v_fract_f32`: `fract(sin(x)·493013)` hash noise applied as
`c·(1 + noise·s)`. Zero-centred and multiplicative, so it averages to ~1.0 and **cannot** produce systematic gain.
pc=154/157/164 use the *same* T# and sampler with different coordinate VGPRs and dmasks 1/2/4 — one texture
sampled three times at offset UVs, one channel each: **chromatic aberration**. b34/b37/b38 aliasing one surface is
three descriptor slots on one texture, not three reads.

### Next Dragon Quest assignment

The lane reduces to one number: the frame is bright because b34 is bright and the effective exposure is ≈1.0,
since `s97 x ExposureScale` is a measured uniform `0.984314`. `tap-171` renders exactly 0 across all 8.29M pixels,
bounding delivered `ExposureScale < 1/510`, so **`s97 > 502` is already established** and `s97 = 1.0` is excluded.

Measure `s97` at `b33+0x878` and `s106` at `b32+0x50` directly with two bounded `--dump-resource 90:ps:32` and
`90:ps:33` runs rather than inferring them. Two outcomes:

- **`s97 ≈ 8192`** — a UE4 **PreExposure** reciprocal, meaning the ≈1.0 cancellation is by design and the defect
  is upstream: prosper's base pass never applied the matching PreExposure factor. That would be a shared UE4-path
  change touching Plucky, ArcRunner and The Pathless, so it needs a generic contract and regression first.
- **`s97` is some other value > 502** — op103 wrote a different value in replay than the capture recorded,
  relocating the defect to op103's execution with no cross-title blast radius.

Note the standing counter-evidence against the PreExposure reading: if the base pass had applied
PreExposure = 8192, b34 would hold values in the thousands, but b34 is a smooth gradient at (0.098, 0.348, 0.971).
And the ~2–4x magnitude needed to match the oracle is far too small for a PreExposure-scale defect.

## Lane C: The Plucky Squire skipped compute

### Public state

- Issue: [#1554 — three compute programs skipped after chapter-one setup](https://github.com/mattias800/prosper/issues/1554).
- Current clean branch/worktree may be transferred: `investigate/plucky-pc915-proof` at exact current master
  `81a9548e`. It has no edits or commits.
- Live proof: [issue comment](https://github.com/mattias800/prosper/issues/1554#issuecomment-5141779159).

Retained evidence:

```text
~/plucky-work/post1564-live-retry.fwpVFI/
  guest.log
    SHA-256 b6c9ba2bf32762796cdf342d900c109105f3f80f31934207728e3e7fc7f71c0d
  shader-dumps/exec_cs_3017460000.bin
    SHA-256 993af8855a62ac24203b12bcf2a968efb1ba163b64937f325353217d411d7342
```

### Proven progression

PR #1564 made the preceding target `0x3017450000` execute successfully 48 times with zero skips. The immediate next
unsupported stage is `0x3017460000`.

PR #1566 then composed its two disjoint pre-loop choices with the canonical counted-loop lowering. The exact retained
stage is:

- 8,448 bytes / 2,112 dwords;
- 1,964 decoded dwords / 1,327 instructions / valid END_PGM;
- coverage `total=1326 alu=1260 exp=0 table=65 unsupported=1`;
- 65 table-dependent MUBUF operations: 33 loads and 32 stores;
- one remaining structural rejection: `s_cbranch_vccz` pc915 to pc1963/END_PGM, whose fallthrough arm contains
  `S_BARRIER` pc1118.

Raw stage recompilation without the live resource table first stops at table-dependent MUBUF pc31. That is separate
from the control-flow coverage result.

### Pc915 predicate proof so far

The scalar chain is known exactly:

```text
pc900: load four scalar dwords into s8:s11 from s12:s13 + 0x20
pc903: load s16:s17 from descriptor s8:s11 + 0x0c
pc906: vcc_lo = s17 - 1                         # VCC_LO used as scalar data here
pc907: SCC = unsigned(s16 > 0)
pc908: VOPC writes its wave mask to s8:s9       # does not feed pc914 VCC
pc910: s10:s11 = SCC ? EXEC : 0
pc911: SCC = unsigned(s14 == vcc_lo)
pc913: VCC = SCC ? EXEC : 0
pc914: VCC &= s10:s11
pc915: branch if VCC is zero
```

Therefore, immediately before pc915:

```text
VCC = EXEC iff (s16 > 0) && (s14 == s17 - 1); otherwise VCC = 0
```

This proves uniformity within one guest wave. It does **not** prove that every wave in the workgroup makes the same
decision. `s14` has no decoded writer before pc915 and is an entry/system/user SGPR. `s12:s13` are also entry values
and source the descriptor/data chain. The retained raw file and log do not include enough launch/register metadata to
map those values.

Do not infer safety merely from the fact that the guest shader exists. If these scalar entry values differ by wave,
placing a Vulkan workgroup barrier inside an ordinary structured arm can violate uniform barrier participation.

### Resolved and merged (#1572)

The pc915 branch is **proven workgroup-uniform** and admitted by a generic rule. Two obligations are proved
independently and mechanically:

1. **EXEC is full at the branch on every path**, by forward must-dataflow (greatest fixpoint, sound across
   back-edges). This leg was missing from the original framing and is load-bearing: `VCCZ` is *also* true for a
   wave whose EXEC is empty, a genuinely per-wave property no amount of scalar uniformity recovers.
2. **VCC is `SCC ? EXEC : 0`**, optionally combined by `s_and_b64`/`s_or_b64`, with every contributing SCC from a
   scalar compare whose every scalar input traces to launch data.

Unproved provenance stays rejected: EXEC/SCC as data, `V_READFIRSTLANE`/`V_READLANE`, VOPC masks, SGPRs above the
launch range, and definitions entering from outside the straight-line region.

The uniformity guarantee is a property of **prosper's own compute entry seeding**, not an assumption about the
guest: user SGPRs come from `load_push_constant` (dispatch-uniform), system SGPRs from `b.groupid[]`
(`gl_WorkGroupID`, workgroup-uniform), TG_SIZE is a constant, and local invocation IDs are seeded into **VGPRs**,
which the provenance walk rejects outright. Nothing at compute entry places a wave-varying value into an SGPR.

Supporting CFG facts for the retained stage: in `[916,1963)` there are 12 barriers and 35 branches, all forward
`s_cbranch_execz` with no back-edges, none spanning a barrier; across the whole program only three edges cross any
barrier (`pc83→899`, `pc898→82`, `pc915→1963`), making pc915 the sole barrier-crossing edge. Coverage moved from
`unsupported=1` to `total=1326 alu=1261 table=65 unsupported=0`.

### Live confirmation, and a retraction that matters

**`0x3017460000` executes live**: 196 dispatches, every one `result=ok`, zero skips, deterministic at a single
SPIR-V hash, zero Vulkan errors across 51,808 log lines.

The proof was **genuinely exercised, not trivially satisfied**: `local=128x1x1` with `subgroup=0` (the portable
path) means **more than one guest wave per workgroup**. In a single-wave workgroup, wave-uniform would imply
workgroup-uniform for free and the barrier promotion would never have been under load. Twelve barriers inside a
VCC-entered region, multiple waves that must agree, 196 times.

All three programs named in #1554 now execute: `0x3017d90000` (#1561), `0x3017450000` (#1564),
`0x3017460000` (#1572).

**RETRACTED — the "three remaining blocked programs" finding was wrong.** A follow-up investigation reported that
`0x30133e0000`, `0x3013430000` and `0x30194c0000` were blocked by descriptor-resolution failures. Unfiltered
logging showed they execute **3,501 times** with a **single** transient failure (1 in 763, self-recovering, 108
successes before and 655 after). See instruments #9 and #10 in the list above: deduped skip lines plus a targeted
`PROSPER_COMPUTELOG_CODE` filter hid thousands of successes. Tracked as **#1581**, retitled to match.

What survives unchanged: `image_gather4_lz` (0x47), `image_store` (0x08) and MUBUF `<=0x07` are **all already
implemented**, and offline coverage is `unsupported=0` for all three. This was never an instruction-coverage gap.

Also: **#282's own comment retracts its premise** and its design doc is gone from master, so neither #282 nor
#485 covers this area — do not inherit them as background.

### Next Plucky assignment

**Rung 4 needs a human.** The one thing no agent here can produce is a visual milestone: a screenshot attempt
returned the KDE lock screen, and was correctly quarantined as non-evidence rather than published. An unlocked,
human-present session is required.

The remaining lead is narrow and **not diagnosed**: for the single failing dispatch the realized resource table
is **byte-identical** to succeeding ones, yet `[compute-cfg]` reports `branches=2` on failure and `branches=1` on
success with the same `hash4k`. Recorded on #1581 as a lead. Severity is low — one dropped dispatch in 763,
self-recovering, visible impact unmeasured — so **do not spend a lane on it** until something visible is
attributed to it.

## Lane D: Alex Kidd in Miracle World DX (PPSA02664)

Added 2026-07-31 because the inherited allocation was skewed to the hardest titles. **It reached the gameplay
rung in one session**, and the root cause was a **generic tiling defect**, not a title quirk.

`tiled_mip_level_layout` converted `mip_x`/`mip_y` — which count whole **256-byte** blocks — to elements using a
multiplier derived from the *macroblock* (`block_width >> 4`). That is correct only when the macroblock is 16x the
256-byte block (the 64 KiB modes) and **four times too small for every 4 KiB mode**. The struct contradicted
itself, reporting `byte_offset=0x800` alongside `tail_x=4` (byte `0x80`) for the same level, and the consumer used
`tail_x`. **53 of 130 packed-tail levels violated the origin-agreement invariant; zero after the fix.**

The correct multiplier is the 256-byte block's element extent, which depends only on element size (AddrLib
Block256_2d): 1B=16x16, 2B=16x8, 4B=8x8, 8B=8x4, 16B=4x4 — every entry multiplying to exactly 256 bytes.

The visible consequence shows how far a texture-origin bug can travel: mipped 4x4 SpriteMask sprites decoded as
foreign, fully transparent texels, so every mask fragment failed its alpha test and a **legitimate** "visible
outside mask" fill covered the entire frame. A black screen, several layers from its cause.

Falsified along the way: both prior size hypotheses (the guest's T# really is 4x4 with `base_level=0`), descriptor
selection, Gen5 size decode, the 4x4 fallback, blend/stencil semantics, guest-state divergence, and stale texture
content. The much older "`fs2949` premultiplies by vertex-colour alpha 0" diagnosis remains falsified — that
capsule was the intro narration crawl, where alpha-0 is correct.

**Two things to carry forward.** Existing `.prgcap` files **bake the resolved tail coordinates**, so pre-fix
capsules replay unchanged even on a fixed binary; re-verifying anything tiling-related needs a **fresh** capture.
And scene identity must be established from rendered *semantic content*, never from submit ordinals, guest VAs,
draw counts, or shader-hash signatures — that invariant was violated once here already.

**The title now reaches rung 6.** The `alexkidd-gameplay` guard is reviewed and landed, so PPSA02664 is
"done" by the ladder's own definition and the lane can close.

The adopted contract keeps the proposed `min_colors=20000` but moves the window to **95–145 s**. Measuring the
route first (`tools/snapshot/profile_route.py`, added with the guard) showed the level loads at 75 s at scale 4
and 72 s at native, so the proposed 70–120 s window would have straddled the load transition: five dark frames
sat inside it, and a run loading only ten seconds later would have pushed the guard below its content-match
ratio and failed on healthy output. The window now opens twenty seconds after the observed load **and after the
route's last pad press at 89 s**, so it samples a settled scene whose only motion is cloud drift and enemy
animation. That is what buys the margin: the worst of 100 reviewed frames scores SSIM **0.93** against a
0.85 floor, where the phase-sensitive `blue-prince-hall`/`terminator-boot` guards fail on healthy frames.
Non-black is `min_nonblack_ratio=0.5` (the tool's conservative half of the reviewed 1.0), not the proposed
0.95 — the approval flow derives that floor and it still sits 3.6x above the pre-fix 0.137. Colour count is one
of five conjoined conditions and must never become the contract alone.

Three independent runs: verify CONTENT-STABLE (richest 35,308/34,984, 50/50 qualifying in both), then
`check` OK (50/50 qualifying, 50/50 structural, 50/50 non-black). Pre-fix was 3,010 colours at 13.7%.

**#710 (title/menu renders incorrectly) is substantially resolved but not closed.** Against the hardware oracle
the panel base is now (253,183,0) vs (250,184,1) — it was (91,66,14) — and the logo, previously "faded /
semi-transparent", matches the oracle's regional mean within 3/255. What remains is narrow and measured: the
panel's watermark overlay lifts the base by only about **half** the hardware amount (G +14 vs +26, B +53 vs
+104). That ratio is a consistent 0.52 in display space but 0.51/0.26 in linear, which argues **against** a
linear-vs-sRGB blend-space explanation and leaves the earlier "semi-transparent tiles composite at too-low
alpha" factor as the live hypothesis. Re-verify any tiling-related claim with a **fresh** capture: pre-fix
`.prgcap` files bake the resolved tail coordinates.

## Lane E: The Pathless (PPSA01826)

First triage 2026-07-31. **Not greenfield** — #1213 exists with prior work, was closed apparently by accident,
and has been reopened. Current state is tracked in **#1570**.

UE4 4.27-class with Wwise. Boots remarkably far — Wwise up, full UE4 thread topology including a Navmesh Tile
Builder (so a real level is loaded), 6,150 submits / 28,859 draws / 27,350 dispatches in 75 s — but **every
presented frame is a flat colour** (blue then white) and progression freezes at ~18 s. That is **rung 0**.

The #1213 worker SIGSEGV at `eboot+0x1b70cd0` **did not reproduce** across four runs. An offline capture (485
draws, 1 dispatch, **0 failures**, replaying hash-exact at `a5e7b61cbf984383`) shows 482 draws plus 1 compute
writing a 768x768 Slate/UMG surface, and only **3** draws touching the 3840x2160 front buffer — all three binding
a 1x1 white texture. prosper faithfully realizes what the guest submits; the defect is composition/progression.

Later stepping showed the front buffer goes black → white, and the three front-buffer draws are the ones painting
white. Their pixel shader's entire constant buffer is a single RGBA of `(1.0, 0.4545, 0.0, 1.0)` — **orange** — and
the export is a COMPR packed-FP16 MRT0 export fed by `V_PACK_B32_F16`, consistent with `export=4` (FP16_ABGR). So
the shape is "a tint constant that should produce orange yields white", pointing at the **packed-FP16 export
path** rather than texel decode. The 1x1 white texture looks like a deliberate Slate white-brush dummy.

Open question the capture cannot yet answer: whether the composite lives in a different submit or the game is
genuinely blank — the capture is 1 submit of ~6,150. A `.prgtl` timeline survey of all submits is the next step
and is CPU-only.

Generic gaps found, all `area:infra` and all benefiting other UE titles: DCC `tile=27` sampled images and metadata
extents unsupported; a 400x400x212 image exceeding the 512 MiB backend bound; and 40+
`[agc] WaitRegMem q=F … dependency violated` — the same per-queue barrier gap as ArcRunner #1226. Estimated cost:
**rung 1 ≈ one medium focused investigation** (answerable offline from the existing capture), **rung 3 ≈ several
sessions**.

## Lane F: Nikoderiko (PPSA23760) — the #305 user-data condition

Triaged 2026-08-01. Tracked in **#305**, which was **retitled that day**: its founding premise —
*"first draws run with the PREVIOUS pipeline's user data (missing bind in decoded stream)"* — is
**measured and falsified**. Anyone starting from "stale bind" is starting from a dead tree.

The title renders its logos, title screen and EULA correctly at 3840x2160 and reaches **rung 2**. The
whole 3D world is missing: 25 distinct `(es, ps)` pipelines dropped, the world behind the title black.

**The condition that actually holds.** A stage resolves garbage descriptors exactly when the
user-data block the guest most recently programmed is **larger** than the bound pipeline's user-SGPR
window (`SPI_SHADER_PGM_RSRC2_{GS,PS}.USER_SGPR`, which equals the shader's own
`user_data_range_end`). The shader then dereferences a V# `num_records`/`dword3` tail as a pointer —
the `0x0004dfac…` constant family — the const-fold correctly refuses to invent a descriptor, and the
draw is skipped fail-visibly. 12→12 resolves; 8→12, 12→28, 12→30, 20→30, 24→28, 24→30 all fail. The
original DOLL (`PPSA17942`) evidence in the issue records the **same** numeric relationship (a
12-dword block against an 8-user-SGPR pipeline), so this is one condition across two UE4 titles.

**Falsified, with the measurement — do not re-derive:** the block is not stale (write provenance puts
it a handful of packets before the draw, from the immediately preceding bind); the shader-registry
lookup is not stale (`registrations=1` across a 2,725-entry registry); no bind is missing or
mis-ordered (193,397 bind packets, **0 of 141** distinct register arrays ever applying more than one
`(es_lo, rsrc2)`, and **300,404 of 300,404** draws folding with the immediately preceding bind — see
hazard 16 for the earlier, contaminated version of that number);
the stage's data is **not** the tail of the programmed block (see hazard 12 above); and **#140
(TYPE-0 AGC data packets) is the wrong tree** — every register write that matters arrives as an
ordinary decoded packet whose provenance is now observable, and none is missing.

The same measurement **confirms** two assumptions previously taken on faith: the seeding base
`USER_DATA_<stage>_0 + user_data_range_start`, and the merged-stage convention that user data begins
at shader SGPR `s8`.

**Remaining candidate:** cross-queue / multi-buffer submit ordering — a larger, later `SET_SH_REG`
whose ordered position prosper places differently from the guest's intent (another command buffer, a
`sceAgcDcbJump` segment, or a second submit entry point).

**Acceptance test for any proposed mechanism: it must explain the GS-versus-PS asymmetry.** Every
traced pixel stage in this title has its declared direct pointer readable; only the vertex/GS block
loses. A submit-ordering story that does not say why the PS block survives is incomplete.

**Instruments** (`PROSPER_UDPROV`, `PROSPER_BINDTRACE`, `[udcand]`, `PROSPER_SHADER_HEADER_NEWEST`,
and the deliberately-off `PROSPER_UD_TAIL_ALIGN`) are on PR #1639, with queue/fold/jump-depth write
provenance following. Use them rather than rebuilding the measurement.

**Instrument warning:** the diagnostics serialize draw realization, so the title screen arrives at
~112 s rather than ~90 s. A 200 s window times out in the pre-title load with **zero rejects** and
looks deceptively like the issue is fixed. Budget ~440 s per run.

Estimated cost: **rung 3 ≈ one to two focused sessions** if submit ordering is confirmed; the
condition is now cheap to detect, so the remaining work is attribution and a generic contract.

## Suggested allocation for a new orchestrator

| Agent | First bounded task | GPU |
|---|---|---|
| Astro Bot | Take a **fresh v43** capture, then attribute the frame-wide `cwm=0` from the raw colour-state triple | One bounded capture |
| Dragon Quest VII | Measure `s97`/`s106` with two `--dump-resource` runs; decide PreExposure versus op103 execution | Two ~2 s runs |
| The Plucky Squire | Live route: confirm `0x3017460000` executes; identify the next skipped stage | ~10 min routed run |
| Alex Kidd | Land the snapshot guard; re-check #710 against the tiling fix | Snapshot verify |
| The Pathless | `.prgtl` survey of all ~6,150 submits for the real composite; then the packed-FP16 export path | CPU first |
| Orchestrator | Review diffs, sequence GPU, publish/merge, keep this document current | Coordinates leases |

**Balance breadth against depth.** Astro Bot is a AAA showcase title and has produced no shippable title fix
across two sessions; Dragon Quest and Plucky are deep but tractable. Alex Kidd went from stuck to gameplay in one
session, and The Pathless was never attempted. When a hard lane stalls, reassign rather than persist — overall
library progress matters more than any single title, and the user has said so explicitly.

**Expect the generic wins to come from tooling and shared subsystems.** This session's merges were a recompiler
uniformity rule, two diagnostic seams, a tooling-honesty fix, and one shared tiling fix. Every one of them helps
titles beyond the lane that found it.

## Ready-to-send subagent prompts

Give every agent: its worktree path, its branch, the exact `origin/master` SHA, the dump root, the **distrobox**
build command, and the instruction to keep evidence under `~/` and out of `/tmp`, the repo, and public text.

### Astro Bot

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md (Lane A), and issue #1459.
Do NOT re-run the R11G11B10 storage fork (falsified: byte-identical output across different
backend paths), compute-writer closure, submit 6279, or the b49 1x1 control.
The frontier is that no geometry draw writes colour anywhere in the frame: 28 realized draws
resolve cwm=0 and cwm1=0. An ABSENT target/shader mask resolves to write-all, so cwm=0 means
present-and-zero, or MODE=DISABLE. Capture v43 retains the raw triple; the retained bundle is
v42 and reports "color-state unavailable", so take a FRESH v43 capture first. Then determine
whether the zero masks are guest intent or a prosper defect. The Gen5 stale-fold prior was
investigated and NOT confirmed; read the lane notes before reviving it.
```

### Dragon Quest VII

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md (Lane B), docs/DRAGON_QUEST_STATUS.md, and #1486.
Binding 36 is EXONERATED and the op103 auto-exposure producer and interpolant delivery are CLEARED.
Film grain and chromatic aberration are both excluded. Do not re-open any of them.
Measure s97 at b33+0x878 and s106 at b32+0x50 with two bounded --dump-resource 90:ps:32 and
90:ps:33 runs. s97 > 502 is already established. If s97 is approximately 8192 it is a UE4
PreExposure reciprocal and the defect is upstream in the base pass (shared UE4 blast radius —
name a generic contract first); otherwise op103 wrote a different value in replay than captured.
Note the counter-evidence: b34 is a smooth gradient, not thousands-valued, and the oracle gap is
only 2-4x. Re-state tile.cpp parity against current master rather than citing an old blob hash.
```

### The Plucky Squire

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md (Lane C), #1554, and PRs #1564/#1566/#1572.
The pc915 uniformity work is MERGED; do not redo it. It is not yet a live-execution claim.
Run the ~10-minute native-cadence route and confirm 0x3017460000 executes, then identify the
next skipped stage. Absence of "[compute] skip unsupported program" proves zero skips. Use the
targeted PROSPER_COMPUTELOG_CODE=<addr>, not blanket PROSPER_COMPUTELOG. Prefer the live path's
own rejection reasons over any offline verdict. Keep PROSPER_RENDER_SCALE=1 and RENDER_EVERY=1.
```

### Alex Kidd

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md (Lane D), and #320.
The generic packed-mip-tail tiling defect is FIXED; the title reaches gameplay. Remaining work is
the snapshot guard: route scripts/ppsa02664/reach-first-gameplay.pad, sample 70-120 s, require the
richest frame to have non-black >= 95% AND >= 20,000 distinct colours, paired with the existing
SSIM comparison. Never use colour count alone — a seed-miss gradient outscores real content.
Run snapshot.py verify and inspect every retained image before adopting a baseline. Then re-check
#710, plausibly the same root cause. Pre-fix .prgcap files bake resolved tail coordinates, so any
tiling re-verification needs a fresh capture.
```

### The Pathless

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md (Lane E), #1570, and #1213 (reopened).
The title is at rung 0: it boots deep into the UE4 frame loop with real GPU work but presents flat
colour and freezes at ~18 s. The #1213 SIGSEGV does not reproduce. Survey the .prgtl timeline
across all ~6,150 submits to find whether the real composite lives in another submit — this is
CPU-only. The current lead is the packed-FP16 MRT0 export path: a tint constant of
(1.0, 0.4545, 0.0, 1.0) (orange) is presenting as white through a COMPR V_PACK_B32_F16 export.
Generic gaps worth fixing for all UE titles: DCC tile=27 sampled images, the per-queue WaitRegMem
barrier model (shared with ArcRunner #1226), and the 512 MiB backend image bound.
```

## Orchestrator cadence and reporting

A useful cadence is:

1. Ask each agent for a compact status after one bounded experiment or one source-design pass.
2. Immediately redirect agents that falsify their premise; do not let them implement a now-diagnostic-only idea.
3. Keep one integration step active at a time: inspect diff, verify exact head, publish PR, watch CI, merge.
4. Let the other agents continue evidence gathering while CI runs, provided they are not modifying the same files or
   depending on the unmerged result.
5. After every merge, fetch master and restart dependent branches from the new exact head.
6. Report to the user in plain language: what became visible/correct, what was ruled out, what is running, and whether
   the GPU is shared or exclusive. Performance numbers are ballpark unless the GPU was isolated.

When an agent appears frozen, distinguish a real long computation from a helper blocked on stdin. Avoid `gh ...
--body-file -` in unattended commands; use a real bounded argument/file so the process cannot wait forever for input.

## Updating this document

After a material event:

- replace the exact master SHA;
- move merged work into the foundation table;
- update the lane's “proven,” “falsified,” and “next” sections;
- link the new issue comment/PR;
- record retained artifact hashes, but sanitize paths;
- remove obsolete experiments so a new agent does not repeat them;
- keep unresolved risks explicit;
- preserve enough history to explain why an attractive old hypothesis was rejected.

This document is successful when a new orchestrator can allocate agents, reuse the retained evidence, and run the next
three discriminating experiments without asking the previous session what it meant.
