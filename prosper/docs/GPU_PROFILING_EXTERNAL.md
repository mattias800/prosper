# External GPU profiling tools — what works on prosper today

**Read this before building another timer.** prosper is an ordinary Vulkan application, so the free
vendor and open-source GPU tooling works on it directly. Everything below was verified on this
Linux/AMD box on 2026-08-08, on `prosper-app`, with **no change to prosper's own code**.

The project has seventeen `PROSPER_*` timing switches, a per-pass GPU timer, a capture/replay stack
and two report scripts. They are individually good. The tools here answer questions none of them can
— per-draw hardware timing, wavefront occupancy, and barrier/queue events — and they answer them on
**any title, with no per-title work**.

## AMD Radeon GPU Profiler (RGP) — verified working

RADV ships the capture side; nothing needs installing to *produce* a trace.

```bash
# Capture frame 900. Set the output directory FIRST -- see the /tmp warning below.
mkdir -p ~/work/rgp && cd ~/work/rgp
MESA_VK_TRACE=rgp MESA_VK_TRACE_FRAME=900 \
PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
    <WORKTREE>/build/prosper-app <DUMP_ROOT>/<TITLE>-app0
```

Verified output on Blue Prince (`PPSA25009`), first attempt, 1.8 MB:

```
radv: Thread trace support is enabled (initial buffer size: 32 MiB,
      instruction timing: enabled, cache counters: enabled, queue events: enabled)
RGP capture saved to 'prosper-app_2026.08.08_19.58.22_frame901.rgp'
```

**What it gives that we cannot:** per-draw and per-dispatch *hardware* timing, wavefront occupancy,
instruction timing, cache counters, and **queue events / barriers** — i.e. synchronisation stalls,
which no `PROSPER_*` switch measures at all.

Selectors: `MESA_VK_TRACE_FRAME=<n>`, `MESA_VK_TRACE_TRIGGER=<file>` (create the file to trigger),
`MESA_VK_TRACE_PER_SUBMIT=1`, `RADV_THREAD_TRACE_BUFFER_SIZE=<MiB>`. Other backends: `rra` (ray
tracing), `rmv` (memory), `ctxroll`.

### Use the TRIGGER, not a frame ordinal — and capture the regime you mean

**A frame number is not portable across runs, and a title with performance regimes makes it actively
misleading.** Both halves cost a capture here:

* `MESA_VK_TRACE_FRAME=900` on Blue Prince lands at **t ~ 5 s**, because the load phase runs at
  108-223 flips/s while the collapse runs at 3.2. That capture is a **menu frame** — the wrong regime
  for any performance question, and nothing in the file says so.
* `MESA_VK_TRACE_FRAME=7500`, chosen from a previous run's counters, produced **no capture at all**:
  that run reached only 6,763 presents before it ended. Two runs of the same route pace differently.

Use the trigger file and fire it when the condition you care about is true:

```bash
out=~/work/rgp; mkdir -p $out; cd $out; rm -f $out/trigger
MESA_VK_TRACE=rgp MESA_VK_TRACE_TRIGGER=$out/trigger ... ./build/prosper-app <DUMP> &
sleep 110          # or wait for whatever marks the regime -- a log line, a flip rate, a screenshot
touch $out/trigger
```

**Sanity-check the regime from the capture's own size.** On Blue Prince the menu frame is **1.8 MB**
and the collapsed frame is **35.9 MB** — a 20x difference, matching ~12 draws/submit against ~4,060.
A capture that is far smaller than expected is very likely the wrong phase, and that is the cheapest
check available before anyone spends time reading it.

**Reading it needs the RGP GUI** (free download from AMD, Linux build available). There is no headless
reader, so an agent can *produce* a capture but a human opens it. Say so when handing one over rather
than implying you read it.

## RenderDoc — installs here, and `convert` is the headless half

```bash
sudo dnf install -y renderdoc      # 1.45 on this box, Vulkan supported
renderdoccmd convert -f cap.rdc -c chrome.json -o cap.json
```

`convert` targets `chrome.json` (Chrome/Perfetto timeline — parseable by script and openable in the
Perfetto UI) and `xml`. That is the route for headless per-event analysis.

**The Fedora package does NOT ship the Python bindings** — `rpm -ql renderdoc` gives `qrenderdoc` (the
GUI) and `librenderdoc.so`, no `renderdoc.so` python module. So the scripted-analysis route is
`convert`, not the Python API, and any recipe assuming `import renderdoc` will fail here.

`renderdoccmd capture` triggers on a keypress rather than a frame ordinal, which makes it awkward
headless; RGP's `MESA_VK_TRACE_FRAME` is the better automated capture.

## `radeontop` — the 60-second "is this even GPU-bound?" triage

Verified, and it is the cheapest useful answer in this document: **no capture, no GUI, one run, any
title.** Install with `dnf install radeontop`.

```bash
# 1. Start the title. 2. Sample INSIDE the regime you care about, never across a phase boundary.
radeontop -d - -l 60
```

Measured on *Blue Prince* in its collapsed regime (t > 70 s, 60 samples) against a `vkcube` control:

| | `gpu` mean | max | `spi` | `cb` | `sclk` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vkcube` (control) | **56.31%** | 66.67% | 29.64% | 11.17% | 99.95% |
| Blue Prince, collapsed | **4.17%** | 7.50% | 3.43% | 1.32% | 38.1% |

**Blue Prince at ~3.2 fps leaves the GPU ~96% idle.** Combined with #2215's measured 30-45 ms/submit
`gpu_device`, that means those submits are long because they **wait**, not because they work — a
synchronisation question rather than a shading-cost one, and the two have completely different fixes.

### Run the control. It is one command and it is not optional here.

`radeontop` prints **"Unknown Radeon card"** on this STRIX_HALO APU, so a low reading is ambiguous between
*the GPU is idle* and *radeontop cannot read this chip*. `vkcube` renders continuously; if the counters
move for it, they are live.

**The control also found dead counters.** `ta` (texture addresser) and `ee` read **0.00% under vkcube**,
which certainly samples a texture — so they are not readable on this hardware. **Trust `gpu`, `spi`, `cb`,
`sx`, `sclk`. Do not quote `ta` or `ee`.** A triage step that silently reads zero on an unsupported block
is worse than no triage step, and that distinction exists only because a control ran.

## Also available, free, not yet explored

| tool | install | for |
| --- | --- | --- |
| `umr` | `dnf install umr` | AMD GPU register/ring debugger — hangs and resets |
| `perf` | installed | CPU side; already used across this project |

## The `/tmp` warning, and it is the charter's own

**RGP writes to `/tmp` by default.** On this box `/tmp` is a RAM-backed tmpfs with a per-user quota
shared by every concurrent agent, and filling it does not merely fail your write — it kills the Bash
tool for every agent on the machine (`CLAUDE.md`, *Write run artifacts to the real disk*). A single
frame capture was 1.8 MB, but `MESA_VK_TRACE_PER_SUBMIT=1` over a long run is not bounded by anything
you have chosen. **`cd` to a directory under `$HOME` before capturing**, and move any capture that
lands in `/tmp` immediately.

## Why this document exists

The instinct on hitting a performance question here has been to add a `PROSPER_*` timer. That produced
a large, individually-correct, collectively-unattributable set of instruments — and on 2026-08-08 six
published shares from two lanes were wrong for arithmetic reasons rather than measurement ones
(instrument traps 141, 143). Vendor tooling that reads the hardware's own thread trace is not subject
to that class of error, costs nothing to run, and needs no maintenance from us.

Reach for these first. Build a `PROSPER_*` switch only for something the guest-facing layer knows and
the GPU vendor cannot see.
