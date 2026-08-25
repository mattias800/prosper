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

    rate, channels, width, raw = read_wav(path)
    frames = len(raw) // (channels * width) if rate else 0
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




def read_wav(path):
    """Minimal RIFF parse. Python's wave module rejects WAVE_FORMAT_IEEE_FLOAT (3), which is
    exactly what the sink dumps for f32 ports, so parse the chunks by hand."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 44 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SystemExit(f"{path}: not a RIFF/WAVE file")
    rate = channels = width = None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt " and size >= 16:
            _fmt_tag, channels, rate = struct.unpack("<HHI", body[0:8])
            width = struct.unpack("<H", body[14:16])[0] // 8
        elif cid == b"data":
            if rate is None:
                raise SystemExit(f"{path}: data chunk before fmt chunk")
            body = data[pos + 8:]
            if size == 0 or size > len(body):
                # Unfinalized dump (crashed run, or the pre-#2981-review offset bug): the size
                # field was never written. Everything after the header is still valid PCM.
                print(f"{path}: note: data size field is 0/oversized; using file remainder")
                body = data[pos + 8:]
            return rate, channels, width, body
        pos += 8 + size + (size & 1)
    raise SystemExit(f"{path}: no data chunk")


if __name__ == "__main__":
    sys.exit(main())
