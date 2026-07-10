#!/usr/bin/env python3
# findcalls.py <target_va_hex> [...] — scan DOLL eboot text for e8-rel32 calls to the target VA(s).
# Text LOAD: VA 0 at file offset 0x34050, size 0x669b81c.
import sys, struct

EBOOT = "/root/PPSA17942-app0/eboot.bin"
BASE_OFF = 0x34050
TEXT_SIZE = 0x669b81c

targets = {int(a, 16) for a in sys.argv[1:]}
data = open(EBOOT, "rb").read(BASE_OFF + TEXT_SIZE)[BASE_OFF:]
hits = []
idx = 0
while True:
    idx = data.find(b"\xe8", idx)
    if idx < 0 or idx + 5 > len(data):
        break
    rel = struct.unpack_from("<i", data, idx + 1)[0]
    tgt = (idx + 5 + rel) & 0xFFFFFFFFFFFFFFFF
    if tgt in targets:
        hits.append((idx, tgt))
    idx += 1
for va, tgt in hits:
    print(f"call @0x{va:x} -> 0x{tgt:x}")
print(f"{len(hits)} hits")
