#!/usr/bin/env python3
"""Report on a PROSPER_HLE_PROF run (fresh3 lane).
usage: hleprof_report.py <dir> [--from S] [--to S] [--top N] [--min-pct P]
Window = delta between the snapshot nearest --from and nearest --to (seconds since profiler start).
Per thread: CPU (schedstat run ns), runqueue wait, context switches, and time inside each HLE import.
"""
import sys, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument('dir'); ap.add_argument('--from', dest='t0', type=float, default=None)
ap.add_argument('--to', dest='t1', type=float, default=None)
ap.add_argument('--top', type=int, default=8); ap.add_argument('--min-pct', type=float, default=1.0)
ap.add_argument('--hist', action='store_true')
a = ap.parse_args()

names = {}
try:
    for line in open(a.dir + '/slots.txt'):
        p = line.split(None, 3)
        names[int(p[0])] = (p[1], p[2], p[3].strip() if len(p) > 3 else '-')
except FileNotFoundError:
    pass

snaps = []  # (t_ms, K{tid:(comm,state,run,wait,slices,vcs,nvcs)}, T{tid:(guest,total)}, S{(tid,slot):[n,tot,max,samp,h..]})
cur = None
for line in open(a.dir + '/hleprof.txt'):
    p = line.split()
    if not p: continue
    if p[0] == '@':
        cur = (int(p[1]), {}, {}, {}, {}); snaps.append(cur)
    elif p[0] == 'K' and cur and len(p) >= 9:
        cur[1][int(p[1])] = (p[2], p[3], int(p[4]), int(p[5]), int(p[6]), int(p[7]), int(p[8]))
    elif p[0] == 'P' and cur:
        cur[4][int(p[1])] = (int(p[2]), int(p[3]), int(p[4]))
    elif p[0] == 'T' and cur:
        cur[2][int(p[1])] = (int(p[2]), int(p[3]))
    elif p[0] == 'S' and cur:
        cur[3][(int(p[1]), int(p[2]))] = [int(x) for x in p[3:]]
# drop a possibly truncated final snapshot
if len(snaps) > 2: snaps = snaps[:-1]

def pick(t):
    if t is None: return None
    return min(snaps, key=lambda s: abs(s[0] - t * 1000))
A = pick(a.t0) or snaps[0]
B = pick(a.t1) or snaps[-1]
W = (B[0] - A[0]) * 1e6  # ns
print(f"window {A[0]/1000:.1f}s .. {B[0]/1000:.1f}s  ({W/1e9:.1f} s), snapshots={len(snaps)}")

def slotname(s):
    lib, nid, nm = names.get(s, ('?', '?', '?'))
    return nm if nm != '-' else f"{lib}:{nid}"

# per-thread
rows = []
for tid, k in B[1].items():
    k0 = A[1].get(tid)
    run = k[2] - (k0[2] if k0 else 0); wait = k[3] - (k0[3] if k0 else 0)
    vcs = k[5] - (k0[5] if k0 else 0); nvcs = k[6] - (k0[6] if k0 else 0)
    rows.append((run, tid, k[0], wait, vcs, nvcs))
rows.sort(reverse=True)
print("\n== threads by CPU in window (run%=CPU/wall, rq%=runnable-but-waiting) ==")
print(f"{'tid':>8} {'comm':<16} {'run%':>6} {'rq%':>6} {'vcs/s':>8} {'nvcs/s':>7}  hle-wall%  top imports (wall% avg max n/s)")
agg = collections.defaultdict(lambda: [0, 0, 0])
for run, tid, comm, wait, vcs, nvcs in rows:
    slots = [(s, v) for (t, s), v in B[3].items() if t == tid]
    items = []
    hle_total = 0
    for s, v in slots:
        v0 = A[3].get((tid, s), [0] * len(v))
        dn = v[0] - v0[0]; dt = v[1] - v0[1]
        if dn <= 0 and dt <= 0: continue
        items.append((dt, dn, v[2], s, [v[4 + i] - v0[4 + i] for i in range(len(v) - 4)]))
        hle_total += dt
        agg[s][0] += dt; agg[s][1] += dn; agg[s][2] = max(agg[s][2], v[2])
    if run / W * 100 < a.min_pct and hle_total / W * 100 < a.min_pct: continue
    items.sort(reverse=True)
    desc = '; '.join(f"{slotname(s)} {dt/W*100:.1f}% {dt/max(dn,1)/1e3:.0f}us {mx/1e6:.1f}ms {dn/(W/1e9):.0f}/s"
                     for dt, dn, mx, s, h in items[:a.top])
    print(f"{tid:>8} {comm:<16} {run/W*100:6.1f} {wait/W*100:6.1f} {vcs/(W/1e9):8.0f} {nvcs/(W/1e9):7.0f}  {hle_total/W*100:6.1f}%   {desc}")
    if a.hist:
        for dt, dn, mx, s, h in items[:a.top]:
            if dt / W * 100 < 1: continue
            print(f"{'':>34}hist {slotname(s)}: " + ' '.join(f"<{2**(i+1)}us:{c}" for i, c in enumerate(h) if c))

print("\n== imports by total wall time across all threads (window) ==")
for s, (dt, dn, mx) in sorted(agg.items(), key=lambda x: -x[1][0])[:40]:
    print(f"{dt/W*100:8.1f}%  n/s={dn/(W/1e9):9.0f}  avg={dt/max(dn,1)/1e3:9.1f}us  max={mx/1e6:8.1f}ms  {slotname(s)}")
print("\n== imports by call rate (window) ==")
for s, (dt, dn, mx) in sorted(agg.items(), key=lambda x: -x[1][1])[:25]:
    print(f"n/s={dn/(W/1e9):9.0f}  avg={dt/max(dn,1)/1e3:9.2f}us  total={dt/W*100:6.1f}%  {slotname(s)}")

PH = {0:'SubmitDcb: g_agc_state_mu wait',1:'SubmitDcb: fold (run_command_buffer)',2:'SubmitDcb: execute_submit_work',3:'SubmitDcb: tail after execute',
 4:'backend vkWaitForFences (graphics batch)',5:'render_locked_queue_submit (vkQueueSubmit)',6:'pend admission wait (submit scope begin)',7:'flip_from_gpu total',
 8:'flip pace sleep (flip_from_gpu)',9:'live compute vkWaitForFences',10:'present blit vkWaitForFences',11:'SubmitDcb total (post scope-begin)',
 12:'shared_present_submit_mutex wait',13:'pend apply_deferred_effect',14:'present_flip'}
if len(A) > 4:
    print("\n== phase timers (window): wall%  n/s  avg  max(cumulative) ==")
    for i in sorted(B[4]):
        n1, t1, m = B[4][i]; n0, t0, _ = A[4].get(i, (0, 0, 0))
        dn, dt = n1 - n0, t1 - t0
        print(f"{i:2d} {PH.get(i,'?'):45s} {dt/W*100:7.1f}%  {dn/(W/1e9):8.1f}/s  avg={dt/max(dn,1)/1e3:9.1f}us  max={m/1e6:8.1f}ms")
