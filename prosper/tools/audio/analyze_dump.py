#!/usr/bin/env python3
"""analyze_dump.py — measure a PROSPER_AUDIO_DUMP WAV (prosper audio-sink PCM dump).

Reports duration, per-segment peak/RMS, non-silent segments, and a loudness summary.
Answers "is audio playing, how loud, and when" as a measurement rather than an
impression (#2981). Works for any title; the WAV comes from the guest's mixed
output before the host sink, so host volume/mute cannot skew it.

Usage: analyze_dump.py <dump.wav> [--min-rms 0.001] [--segment-secs 5]
"""
import sys
import struct
import wave


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    min_rms = 0.001
    segment_secs = 5
    argv = sys.argv[2:]
    if "--min-rms" in argv:
        min_rms = float(argv[argv.index("--min-rms") + 1])
    if "--segment-secs" in argv:
        segment_secs = int(argv[argv.index("--segment-secs") + 1])

    w = wave.open(path, "rb")
    rate, channels, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    raw = w.readframes(frames)
    w.close()
    dur = frames / rate if rate else 0
    print(f"{path}: {rate} Hz, {channels} ch, {width * 8}-bit, {frames} frames, {dur:.1f}s")

    if width == 4:
        samples = struct.unpack(f"<{len(raw) // 4}f", raw)
    elif width == 2:
        samples = [s / 32768.0 for s in struct.unpack(f"<{len(raw) // 2}h", raw)]
    else:
        print(f"unsupported sample width {width}")
        return 2

    seg = rate * channels * segment_secs
    print(f"\nper-{segment_secs}s segment: peak / rms  (* = audible, rms >= {min_rms})")
    audible_secs = 0
    segments = []
    for i in range(0, len(samples), seg):
        chunk = samples[i:i + seg]
        if not chunk:
            break
        peak = max(abs(s) for s in chunk)
        rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5
        t0 = i // (rate * channels)
        t1 = (i + len(chunk)) // (rate * channels)
        flag = "*" if rms >= min_rms else " "
        if rms >= min_rms:
            audible_secs += len(chunk) / (rate * channels)
        segments.append((t0, t1, peak, rms, flag))
        print(f"  {t0:4d}-{t1:4d}s  peak={peak:7.4f} rms={rms:7.4f} {flag}")

    print(f"\naudible time (rms >= {min_rms}): {audible_secs:.1f}s of {dur:.1f}s")
    loud = [s for s in segments if s[4] == "*"]
    if loud:
        loudest = max(loud, key=lambda s: s[2])
        print(f"audible segments span {loud[0][0]}s .. {loud[-1][1]}s "
              f"(loudest: peak={loudest[2]:.3f} at {loudest[0]}s)")
    else:
        print("NO AUDIBLE AUDIO in the entire dump")
    return 0


if __name__ == "__main__":
    sys.exit(main())
