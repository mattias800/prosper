# `prosper/tools/screenshots/` — keeping committed imagery from swallowing the repository

One tool, `shrink.py`, and one rule: **a screenshot committed to `assets/screenshots/` is a 1920-wide
WebP, not a 4K PNG.**

## Why this folder exists

Measured 2026-09-02, on a 621 MiB pack:

| | packed |
| --- | --- |
| `assets/screenshots` | 398 MiB |
| other committed images | 73 MiB |
| **all source, docs and tests, entire history** | **18 MiB** |

Images were ~96% of the blob content, and the cause was resolution rather than count: 121 files
averaging 2.5 MiB, the largest 20.6 MiB, all full 3840x2160 PNG. Re-encoding the same 140 frames to
1920-wide WebP took them from **392 MiB to 14.2 MiB** with no visible loss — small on-screen UI text
stays legible, which is what a progress capture is usually evidence *of*.

The scaling argument is the real one. At 2.5 MiB per capture and the mid-2026 rate of roughly 17 a
week, GitHub's 10 GB repository ceiling arrives in about **four years**. At the size this tool
produces, it does not arrive.

## What it cannot do, so nobody expects it to

Git keeps every blob it has ever seen. Re-encoding **adds** the small versions; the large originals
stay in history, so a *full* clone gets marginally bigger rather than smaller. What improves
immediately is every **shallow or sparse** clone, which fetches only the current tree — which is what
CI does on every job, and what `--filter=blob:none --sparse` gives a developer.

Reclaiming the historical 398 MiB would need a history rewrite. That changes every commit SHA and
breaks the commit references in every existing pull request, which in this project is the durable
record. That trade has deliberately **not** been made.

## Using it

```sh
python3 prosper/tools/screenshots/shrink.py assets/screenshots        # re-encode in place
python3 prosper/tools/screenshots/shrink.py --check assets/screenshots # report only; exit 1 if any
```

Run it on any capture before committing it. `--check` is the form to put in a gate if this ever
starts drifting again; it exits non-zero when a file would change.

WebP rather than PNG because these are photographic game frames: PNG is lossless, so lighting
gradients cannot compress, which is exactly why the originals were enormous. Palette-quantising to
keep the `.png` extension would band those same gradients — so the extension changes and callers are
rewritten instead, which is mechanical and verifiable (every `assets/screenshots/...` reference must
resolve; the conversion checked all 511 of them).
