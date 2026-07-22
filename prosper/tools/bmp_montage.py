#!/usr/bin/env python3
"""Fast contact-sheet of many (possibly 4K) BMP dumps, stdlib-only.

Stride-samples each BMP straight into a small thumbnail (never materializes the
full image), so a directory of 4K frame dumps montages quickly. Useful for
scanning a gameplay capture for a specific visual artifact.
Usage: bmp_montage.py <dir-or-glob> <out.png> [thumb=160] [cols=8]
"""
import struct, zlib, glob, os, sys


def bmp_thumb(path, tw, th):
    with open(path, "rb") as f:
        d = f.read()
    if d[:2] != b"BM":
        return None
    off = struct.unpack_from("<I", d, 10)[0]
    w = struct.unpack_from("<i", d, 18)[0]
    h = struct.unpack_from("<i", d, 22)[0]
    bpp = struct.unpack_from("<H", d, 28)[0]
    td = h < 0
    h = abs(h)
    if bpp not in (24, 32) or w <= 0 or h <= 0:
        return None
    bp = bpp // 8
    stride = ((w * bp + 3) // 4) * 4
    out = bytearray(tw * th * 3)
    for ty in range(th):
        sy = ty * h // th
        src_y = sy if td else (h - 1 - sy)
        base = off + src_y * stride
        for tx in range(tw):
            sx = tx * w // tw
            p = base + sx * bp
            o = (ty * tw + tx) * 3
            out[o] = d[p + 2]; out[o + 1] = d[p + 1]; out[o + 2] = d[p]
    return out


def write_png(path, w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb[y * w * 3:(y + 1) * w * 3]) for y in range(h))
    def ck(t, p): c = t + p; return struct.pack(">I", len(p)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
           ck(b"IDAT", zlib.compress(raw, 6)) + ck(b"IEND", b""))
    open(path, "wb").write(png)


if __name__ == "__main__":
    src = sys.argv[1]; out = sys.argv[2]
    TH = int(sys.argv[3]) if len(sys.argv) > 3 else 160
    COLS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    files = sorted(glob.glob(os.path.join(src, "*.bmp")) if os.path.isdir(src) else glob.glob(src))
    thumbs = [(os.path.basename(f), bmp_thumb(f, TH, TH)) for f in files]
    thumbs = [(n, t) for n, t in thumbs if t]
    n = len(thumbs); rows = (n + COLS - 1) // COLS
    CW = COLS * TH; CH = max(rows, 1) * TH
    canvas = bytearray(CW * CH * 3)
    for i, (name, t) in enumerate(thumbs):
        cx = (i % COLS) * TH; cy = (i // COLS) * TH
        for y in range(TH):
            row = t[y * TH * 3:(y + 1) * TH * 3]
            o = ((cy + y) * CW + cx) * 3
            canvas[o:o + TH * 3] = row
    write_png(out, CW, CH, canvas)
    print(f"{n} frames -> {out} ({CW}x{CH})")
    for i, (name, _) in enumerate(thumbs):
        print(i, name)
