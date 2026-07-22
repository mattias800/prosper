#!/usr/bin/env python3
"""Minimal dependency-free BMP -> PNG converter (stdlib zlib only).

Handles the uncompressed 24/32-bit BGR(A) bottom-up BMPs that prosper's frame /
texture dumps produce, so captured frames and PROSPER_DUMP_TEX outputs can be
viewed without Pillow. Usage: bmp2png.py in.bmp [out.png]  (or a dir of *.bmp).
"""
import struct, zlib, sys, os, glob


def bmp_to_png(bmp_path, png_path):
    with open(bmp_path, "rb") as f:
        data = f.read()
    if data[:2] != b"BM":
        raise ValueError("not a BMP")
    pixoff = struct.unpack_from("<I", data, 10)[0]
    hdr = struct.unpack_from("<I", data, 14)[0]
    w = struct.unpack_from("<i", data, 18)[0]
    h = struct.unpack_from("<i", data, 22)[0]
    bpp = struct.unpack_from("<H", data, 28)[0]
    top_down = h < 0
    h = abs(h)
    if bpp not in (24, 32):
        raise ValueError("only 24/32-bit BMP supported, got %d" % bpp)
    bytes_pp = bpp // 8
    row_stride = ((w * bytes_pp + 3) // 4) * 4
    rows = []
    for y in range(h):
        src_y = y if top_down else (h - 1 - y)
        base = pixoff + src_y * row_stride
        out = bytearray(1 + w * 3)  # PNG filter byte 0 + RGB
        out[0] = 0
        for x in range(w):
            p = base + x * bytes_pp
            b, g, r = data[p], data[p + 1], data[p + 2]
            o = 1 + x * 3
            out[o], out[o + 1], out[o + 2] = r, g, b
        rows.append(bytes(out))
    raw = b"".join(rows)

    def chunk(tag, payload):
        c = tag + payload
        return struct.pack(">I", len(payload)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, colour type 2 (RGB)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
           chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(png_path, "wb") as f:
        f.write(png)
    return w, h


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("usage: bmp2png.py <file.bmp|dir> [out.png]"); sys.exit(1)
    src = args[0]
    if os.path.isdir(src):
        for b in sorted(glob.glob(os.path.join(src, "*.bmp"))):
            try:
                wh = bmp_to_png(b, b[:-4] + ".png")
                print("ok", os.path.basename(b), wh)
            except Exception as e:
                print("skip", os.path.basename(b), e)
    else:
        out = args[1] if len(args) > 1 else (src[:-4] + ".png")
        print("ok", bmp_to_png(src, out), "->", out)
