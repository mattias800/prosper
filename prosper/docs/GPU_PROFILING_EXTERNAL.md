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

## Also available, free, not yet explored

| tool | install | for |
| --- | --- | --- |
| `umr` | `dnf install umr` | AMD GPU register/ring debugger — hangs and resets |
| `radeontop` | `dnf install radeontop` | coarse live GPU utilisation |
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
