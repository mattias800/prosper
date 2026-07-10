#!/usr/bin/env python3
# findrefs.py <va_hex> — find e9 jumps, e8 calls, lea-rip refs and 8-byte pointer literals to a VA
# anywhere in the DOLL eboot file (text VA0 @0x34050; rodata VA 0x669c000 @0x66e04d0; data VA
# 0x8828000 @0x8871930 per the self_dump segment/phdr tables).
import sys, struct

EBOOT = "/root/PPSA17942-app0/eboot.bin"
target = int(sys.argv[1], 16)
data = open(EBOOT, "rb").read()

TEXT_OFF, TEXT_VA, TEXT_SZ = 0x34050, 0, 0x669b81c
SEGS = [
    (0x34050, 0x0, 0x669b81c, "text"),
    (0x66e04d0, 0x669c000, 0x218b0f8, "rodata"),
    (0x8871930, 0x8828000, 0xc6ba68, "data"),
    (0x9511650, 0x9494000, 0x27050, "data2"),
]

# pointer literals
pat = struct.pack("<Q", target)
for off, va, sz, nm in SEGS:
    seg = data[off:off + sz]
    i = 0
    while True:
        i = seg.find(pat, i)
        if i < 0:
            break
        print(f"ptr-literal in {nm} @va 0x{va + i:x}")
        i += 1

# calls/jmps in text
seg = data[TEXT_OFF:TEXT_OFF + TEXT_SZ]
for opcode, name in ((0xE8, "call"), (0xE9, "jmp")):
    i = 0
    while True:
        i = seg.find(bytes([opcode]), i)
        if i < 0 or i + 5 > len(seg):
            break
        rel = struct.unpack_from("<i", seg, i + 1)[0]
        if (i + 5 + rel) == target:
            print(f"{name} @va 0x{i:x}")
        i += 1
# lea rip-relative (48 8d ?? rel32) — check 3-byte opcodes 48 8d 05..3d
i = 0
while True:
    i = seg.find(b"\x48\x8d", i)
    if i < 0 or i + 7 > len(seg):
        break
    modrm = seg[i + 2]
    if (modrm & 0xC7) == 0x05:
        rel = struct.unpack_from("<i", seg, i + 3)[0]
        if (i + 7 + rel) == target:
            print(f"lea @va 0x{i:x}")
    i += 1
print("done")
