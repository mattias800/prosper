# AGENTS.md — prosper/tools/determinism

**"Does this renderer produce the same picture twice?" — the campaign, and the rule for reading
it.** Everything here is about #2945: one frozen `.prgbundle`/`.prgcap`, replayed offline many
times with one binary, and whether the output hash changes.

Two files, and the split is the point:

- `replay_determinism.sh` runs the campaign. It replays a capture N times in N separate processes,
  runs `tools/vkprobe` beside it in the same round as a control, alternates their order, alternates
  loaded and unloaded blocks, records the peer-GPU-consumer count on every row, and waits a peer
  out before sampling so another lane's capture is not contaminated. Output is a tidy CSV.
- `replay_determinism_report.py` reads that CSV and issues one of **three** verdicts. Its test,
  `test_replay_determinism_report.py`, is registered as the platform-independent ctest case
  `replay_determinism_report`.

**The third verdict is why this folder exists.** #2945's failure rate drifts machine-wide over
minutes — the identical command measured 0 of 20 failures in one ten-minute window and 12 of 12 in
the next with nothing changed. So "every replay returned the same hash" is produced equally by a
repaired renderer and by a quiet half-hour, and a campaign that reports only PASS/FAIL cannot tell
you which one you have. The report therefore answers:

| verdict | means |
| --- | --- |
| `NONDETERMINISTIC` | a subject arm returned more than one distinct hash. Decisive. |
| `DETERMINISTIC` | one hash per arm **and** the control failed at least once, so the campaign demonstrably met a window in which the class is expressible. |
| `UNDECIDED` | one hash per arm and the control never failed either. **Not a negative.** Quote it as "no instance observed in N replays over T", never as "fixed". |

The control is `tools/vkprobe`, a bare-Vulkan program with no prosper code in the process that
reproduces the same class. **Read its README before quoting it**: a clean vkprobe run proves
nothing on its own, which is precisely why it is used here only to detect the *window* and never to
clear anything. Its arm uses non-identity indices (`3,4,5`) deliberately — with the identity
sequence a correct index fetch and "the shader was handed the ordinals" produce the same readback,
which is what made an earlier campaign on this issue score 91.6% where the diagnostic indices score
1.2%.

## What belongs here, and what does not

This folder owns *campaign design and its reading* — the apparatus that turns many runs into a
statement somebody may act on. The thing being replayed belongs elsewhere: `tools/gpu_replay` owns
capture/replay and every per-draw diagnostic, `tools/vkprobe` owns the bare-Vulkan control and the
hand-written SPIR-V. A new determinism *subject* (a different tool whose output should not vary run
to run) is a new `--arm`, not a new folder; a new way to *decide whether a null means anything* is a
new verdict here.

## Using it

```bash
prosper/tools/determinism/replay_determinism.sh \
    --replay build-linux/gpu_replay --capture ~/work/frame.prgcap --out ~/work/campaign.csv \
    --control build-linux/vkprobe --shaders prosper/tools/vkprobe/shaders \
    --arm 'full=' --arm 'draw42=--draw 42' \
    --rounds 300 --block 12 --load 3 --label host
python3 prosper/tools/determinism/replay_determinism_report.py ~/work/campaign.csv
```

The work directory (`--work`, default a fresh `mktemp -d`) keeps one log per replay and one per
control round, named by round. That is deliberate: when a round finally does deviate, its log is the
only record of what that particular process saw, and a campaign that overwrites them leaves you with
a hash and nothing to explain it. Rendered `.bmp` outputs are deleted as each replay is scored —
they are 24 MB apiece at 4K and the hash is what the campaign compares.

Two things the numbers do not say on their own, so say them yourself:

- **Quote the wall-clock span beside the n.** 400 runs inside 173 s are one drift period, not 400
  samples of what varies. The report prints the span for this reason.
- **Quote the driver and the frontend.** Runs made through a container and on the host can be
  different Mesa builds on the same box; `--label` is there to keep two such arms apart in one CSV,
  and interleaving them round by round is the only shape of that comparison that survives the drift.
