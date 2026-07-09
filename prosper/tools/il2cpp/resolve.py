#!/usr/bin/env python3
# resolve.py — map runtime addresses / btrace frames to C# method names using the
# Il2CppDumper output (script.json). Companion to prx_to_elf.py; see README.md.
#
# The IL2CPP PRX loads at a fixed guest base (0x440000000 for both PPSA24651 and
# PPSA02664), so a method's runtime address = base + RVA, and script.json's
# "Address" field == RVA (the flattened ELF has p_offset == p_vaddr). A prosper
# [btrace] chain prints frames as "il+0x<offset>", where <offset> is already the
# RVA — feed those straight in.
#
# Usage:
#   python3 resolve.py <script.json> 0x16b981 0x11e63c ...      # bare RVAs
#   python3 resolve.py <script.json> il+0x16b981,eb+0xada254    # a btrace chain
#   echo '[btrace] ... chain=il+0x1e23b8,il+0xde92e9' | python3 resolve.py <script.json> -
#   python3 resolve.py <script.json> --base 0x440000000 0x4402140d0   # absolute runtime addr
import json, bisect, re, sys

def load(path):
    sj = json.load(open(path))
    addrs = sorted((m['Address'], m['Name']) for m in sj.get('ScriptMethod', []) if 'Address' in m)
    return addrs, [a for a, _ in addrs]

def resolve_one(addrs, keys, off):
    i = bisect.bisect_right(keys, off) - 1
    if i >= 0 and off - addrs[i][0] < 0x8000:
        a, name = addrs[i]
        return f"{name}  (+0x{off - a:x})"
    return "<il2cpp runtime / no managed method at this offset>"

def main(argv):
    if len(argv) < 2:
        print(__doc__); return 1
    script = argv[1]
    rest = argv[2:]
    base = 0
    if '--base' in rest:
        k = rest.index('--base'); base = int(rest[k+1], 0); del rest[k:k+2]
    addrs, keys = load(script)
    tokens = []
    for a in rest:
        if a == '-':
            tokens += re.findall(r'(?:il|eb)\+0x[0-9a-f]+|0x[0-9a-f]+', sys.stdin.read())
        else:
            tokens += re.split(r'[,\s]+', a.strip())
    for t in tokens:
        if not t: continue
        m = re.match(r'(il|eb)\+0x([0-9a-f]+)$', t)
        if m:
            if m.group(1) == 'eb':
                print(f"{t:22}  (eboot / Unity native — not managed)"); continue
            off = int(m.group(2), 16)
        else:
            v = int(t, 0)
            off = v - base if v >= base and base else v
        print(f"{t:22}  {resolve_one(addrs, keys, off)}")
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
