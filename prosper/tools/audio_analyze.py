#!/usr/bin/env python3
"""audio_analyze.py — objective health check for PROSPER_AUDIO_DUMP guest PCM captures.

Detects the block-repetition failure mode (guest mixer/stream starvation: the same DSP-block-
sized window is re-emitted, audible as "the buffer replays") without a human listener, by
comparing the autocorrelation at the suspected block lag against neighboring non-multiple lags.
A healthy music/SFX stream has no reason to correlate at one specific ~21 ms lag and not at
its neighbors; a starving mixer produces a sharp spike there.

Usage:
  audio_analyze.py DUMP.raw [--fmt f32|s16] [--channels N] [--rate HZ] [--block FRAMES]
                             [--tail-seconds S] [--json]

Exit code: 0 = no repetition signature, 1 = repetition detected, 2 = not enough signal.
Verdict thresholds are conservative; see REPEAT_SPIKE below.
"""
import argparse, json, math, struct, sys

# Spike = corr(block lag) - max(corr(neighbor lags)). Broken Messenger measures ~0.7 spike;
# clean content measures near 0 (music self-similarity shows up at MUSICAL periods, which are
# not pinned to the DSP block size and also raise the neighbor baseline).
REPEAT_SPIKE = 0.35
MIN_RMS = 1e-4          # below this the capture is effectively silence: no verdict


def load(path, fmt, channels):
    raw = open(path, "rb").read()
    if fmt == "f32":
        n = len(raw) // 4
        v = struct.unpack("<%df" % n, raw[: n * 4])
    else:
        n = len(raw) // 2
        v = [x / 32768.0 for x in struct.unpack("<%dh" % n, raw[: n * 2])]
    return v[0::channels]   # first channel is enough for periodicity


def corr(seg, lag, stride=4):
    m = len(seg) - lag
    if m <= stride:
        return 0.0
    s = e1 = e2 = 0.0
    for i in range(0, m, stride):
        a, b = seg[i], seg[i + lag]
        s += a * b
        e1 += a * a
        e2 += b * b
    return s / math.sqrt(e1 * e2) if e1 > 0 and e2 > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--fmt", choices=["f32", "s16"], default="f32")
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--block", type=int, default=1024, help="suspected repeat period in frames")
    ap.add_argument("--tail-seconds", type=float, default=10.0,
                    help="analyze only the last S seconds (skips boot silence/fade-in)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ch = load(a.dump, a.fmt, a.channels)
    tail = int(a.tail_seconds * a.rate)
    seg = ch[-tail:] if len(ch) > tail else ch
    if len(seg) < 4 * a.block:
        print("audio_analyze: capture too short", file=sys.stderr)
        return 2
    mean = sum(seg) / len(seg)
    seg = [x - mean for x in seg]
    rms = math.sqrt(sum(x * x for x in seg) / len(seg))
    peak = max(abs(x) for x in seg)
    if rms < MIN_RMS:
        out = {"verdict": "silent", "rms": rms, "peak": peak}
        print(json.dumps(out) if a.json else f"SILENT capture (rms={rms:.6f}) — no verdict")
        return 2

    block_corr = corr(seg, a.block)
    # Neighbors: same magnitude, deliberately NOT multiples/divisors of the block.
    neighbors = [int(a.block * f) for f in (0.7, 0.8, 1.3, 1.6)]
    ncorrs = {lag: corr(seg, lag) for lag in neighbors}
    baseline = max(ncorrs.values())
    spike = block_corr - baseline
    broken = spike > REPEAT_SPIKE

    out = {
        "verdict": "block-repetition" if broken else "clean",
        "block_frames": a.block,
        "block_corr": round(block_corr, 4),
        "neighbor_corrs": {str(k): round(v, 4) for k, v in ncorrs.items()},
        "spike": round(spike, 4),
        "threshold": REPEAT_SPIKE,
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "analyzed_seconds": round(len(seg) / a.rate, 2),
    }
    if a.json:
        print(json.dumps(out))
    else:
        print(f"{out['verdict'].upper()}: corr(block {a.block}f)={block_corr:+.3f} "
              f"neighbor-max={baseline:+.3f} spike={spike:+.3f} (threshold {REPEAT_SPIKE}) "
              f"rms={rms:.4f} peak={peak:.4f}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
