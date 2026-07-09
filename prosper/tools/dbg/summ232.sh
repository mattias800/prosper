#!/bin/bash
# Summarize gw232 sample files: one line per thread = final-wait-fn | first guest RAs.
for f in "$@"; do
  echo "===== $f ====="
  python3 - "$f" << 'PEOF'
import sys, re, collections
lines = open(sys.argv[1]).read().splitlines()
cur = None; host = ""; guest = ""
rows = []
for l in lines:
    if l.startswith("== thread"):
        if cur: rows.append((cur, host, guest))
        cur = l; host = ""; guest = ""
    elif l.startswith("  host:"):
        # last prosper:: frame or first frame
        fns = [x.strip() for x in l[8:].split("<-")]
        pk = next((x for x in fns if "prosper" in x or "k_" in x), fns[0] if fns else "")
        pk = re.sub(r"\(unsigned long.*\)", "", pk)
        host = pk
    elif l.startswith("  guest:"):
        guest = " ".join(l[9:].split()[:6])
if cur: rows.append((cur, host, guest))
sig = collections.Counter((h, g) for _, h, g in rows)
for (h, g), n in sig.most_common():
    print("%3d  %-50s %s" % (n, h, g))
print("total threads:", len(rows))
PEOF
done
