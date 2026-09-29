# Issue #3157: paired AVX2 detile measurements (2026-09-29)

This is the numeric evidence for PR #3952. The input shape is Astro Bot's recurring
3840×2160, eight-byte, mode-27 storage seed. The CSV contains only timing numbers
and path selection; it contains no game memory, shader code, addresses, or images.

## Same-binary game comparison

The retained private harness is
`~/prosper-evidence/astro-3157-image-handoff-20260929/compare_detile.sh`.
These are the two commands recorded by the shared GPU gate, in this order:

```sh
distrobox enter ps5ys -- bash ~/prosper-evidence/astro-3157-image-handoff-20260929/compare_detile.sh forward
distrobox enter ps5ys -- bash ~/prosper-evidence/astro-3157-image-handoff-20260929/compare_detile.sh reverse
```

The harness launched `prosper-app --dump <DUMP_ROOT>/PPSA21564-app0
--fps --present-mode immediate --volume 0` for 70 seconds per arm with
`PROSPER_COMPUTE_IMAGE_TIMING=1` and `PROSPER_COMPUTE_TIMING_CODE=0x500571000`.
The gather arm set `PROSPER_NO_PAIRED_AVX2_DETILE=1`; the paired arm left it unset.
Each arm used an isolated save, cache, Mesa cache, pipeline cache, and temporary
directory. Both orders used the same binary (SHA-256
`8fd365c90827064cf1e95937f8c0d911463ad9407703d687624d2cdc94bbe317`),
verified before and after each pair. The binary predates the test-only amendment
in `81ed94a31`; the detile implementation is unchanged.

[`astro-binding60-4k.csv`](astro-binding60-4k.csv) contains every matching
`[compute-image]` record in run order, including records that did not detile.
The selection is `binding=60`, `class=storage`, `extent=3840x2160x1`.
The reported median is over all selected `detile_ms` values, consistent with
the original capture summarizer. The direct-detile counts make the selection
auditable:

| Run | Records | Direct detiles | Median detile (ms) |
| --- | ---: | ---: | ---: |
| gather | 644 | 632 | 2.335 |
| paired | 592 | 580 | 1.780 |
| paired-reverse | 669 | 657 | 1.774 |
| gather-reverse | 627 | 607 | 2.307 |

This command recomputes the count and median from the attached CSV:

```sh
python3 - <<'PY'
import csv
import statistics
from collections import defaultdict
from pathlib import Path
rows = defaultdict(list)
with Path('prosper/docs/evidence/3157-detile/astro-binding60-4k.csv').open() as file:
    for row in csv.DictReader(file):
        rows[row['run']].append(row)
for run, values in rows.items():
    print(run, len(values), sum(v['direct_detile'] == '1' for v in values),
          round(statistics.median(float(v['detile_ms']) for v in values), 3))
PY
```

All four runs presented zero frames because the main revision used for those
captures had an independent presentation regression, bisected to `cbf3db437`
(PR #3944). PR #3960 subsequently restored presentation; the capture results
below still describe the older binary. These timings
support a local detile-cost comparison only. They do not support an FPS,
equal-work, or visual-correctness claim.

## Dedicated benchmark on the revised head

At `81ed94a31`, build the explicit target and run the same binary in alternating
order. Each invocation first checks exact detiled bytes against its original
linear input. The collected stdout, with both default and single-threaded
worker settings, is attached as [`bench-detile.txt`](bench-detile.txt).

```sh
cmake --build prosper/build-astro-handoff --target bench_detile -j8
python3 ~/prosper-evidence/astro-3157-image-handoff-20260929/bench_detile.py
```

The benchmark binary's SHA-256 was
`d2ca75c0d49874836d67995f662264cb1e15c82367c4189802f8e791140a4771`.
The harness invokes it eight times per worker setting in the order paired,
gather, gather, paired, then repeats that order. It sets
`PROSPER_NO_PAIRED_AVX2_DETILE=1` for gather and
`PROSPER_DETILE_SINGLE_THREADED=1` for single-threaded runs.
