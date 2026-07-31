#!/usr/bin/env python3
"""Profile a route's content metrics so guard thresholds come from measurement.

`README.md` requires a new baseline's thresholds to sit conservatively below the
observed valid range, but a candidate entry cannot report that range: `verify`
only accepts or rejects the thresholds it was given, and a rejected candidate
saves no evidence. This runs ONE capture with a permissive candidate and prints
every sample in the window, so the evidence window and the `min_colors` floor
are chosen from the title's actual behaviour instead of a guess.

It measures; it never writes to `snapshots.json`. The normal
verify -> inspect -> `update --reviewed` -> `check` workflow is unchanged, and
this does not replace the mandatory image review.

Usage (from `prosper/`, with the entry as a JSON object):

    python3 tools/snapshot/profile_route.py '{"name":"probe",
        "dump":"PPSA02664-app0","scale":4,"timeout":150,"sample_seconds":1,
        "capture_after_seconds":0,"capture_before_seconds":150,
        "pad_script":"scripts/ppsa02664/reach-first-gameplay.pad",
        "savedata_policy":"fresh","env":{},"min_colors":1}' ~/route-profile

Set `capture_after_seconds` to 0 and `capture_before_seconds` past the end of
the route to profile the whole boot, and keep `min_colors` at 1 so no phase is
filtered out before it can be measured. Every captured frame is copied to the
output directory next to a `profile.json` record, because a colour count alone
must never decide a guard: a seed-miss diagnostic gradient can score MORE
distinct colours than real content, so the frames have to be looked at.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snapshot  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    entry = json.loads(sys.argv[1])
    out = os.path.expanduser(sys.argv[2])
    os.makedirs(out, exist_ok=True)

    _, summary, tmp = snapshot.capture_content(
        entry, run_log=os.path.join(out, "profile.log"))
    try:
        rows = [{
            "elapsed": record["elapsed"],
            "colors": record["colors"],
            "nonblack_ratio": round(record["nonblack_ratio"], 6),
            "dims": list(record["dims"]),
            "pixel_crc32": record["hash"],
            "present_count": record["present_count"],
            "filename": record["filename"],
            "luma16x9": "".join(f"{v:02x}" for v in record["luma16x9"]),
        } for record in summary["records"]]
        for row in rows:
            print(f"{row['elapsed']:8.1f}s colors={row['colors']:8d} "
                  f"nonblack={row['nonblack_ratio']:.4f} "
                  f"{row['dims'][0]}x{row['dims'][1]} {row['filename']}")
        changes = sum(a["pixel_crc32"] != b["pixel_crc32"]
                      for a, b in zip(rows, rows[1:]))
        print(f"samples={len(rows)} consecutive-frame changes={changes} "
              f"richest={summary['richest']['colors']} colors at "
              f"{summary['richest']['elapsed']:.1f}s")
        with open(os.path.join(out, "profile.json"), "w") as f:
            json.dump(rows, f, indent=1)
            f.write("\n")
        for record in summary["records"]:
            shutil.copyfile(record["path"], os.path.join(out, record["filename"]))
        print(f"wrote {len(rows)} frames and profile.json to {out} -- inspect the "
              f"frames before trusting any metric")
    finally:
        snapshot._cleanup(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
