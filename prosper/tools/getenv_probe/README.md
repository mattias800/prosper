# getenv_probe — count `getenv()` calls in a live prosper run

An `LD_PRELOAD` shim that intercepts `getenv()`, counts every call, and reports the running total
and rate every 10 seconds. Roughly 40 lines; no build-system integration, because it must be loaded
into the process rather than linked into it.

## Why this exists

prosper gates diagnostics on environment variables, and several of those guards sit on per-draw and
per-resource paths. That is invisible in an ordinary profile on Linux — glibc's `getenv()` is a
lock-free scan of `environ`, so a high call rate costs some CPU and blocks nothing.

On Windows it is not invisible. The UCRT's `getenv()` takes `__acrt_environment_lock` on **every
call**, so the same code serialises every thread that touches the environment. A call rate that is
merely wasteful on one platform is a global lock bottleneck on the other, and the two look nothing
alike in a profile.

The rate itself is platform-independent, which is the point of this tool: **measure the call count on
Linux, reason about its cost on Windows.** First use (2026-08-07, #2215) measured **24.3 million
calls in 90 seconds — a sustained ~295,000/s** on *Blue Prince* `PPSA25009`, which is the load that
had to be explained before any Windows fix could be justified.

## Build and use

```bash
gcc -shared -fPIC -O2 -o probe.so getenv_probe.c -ldl -lpthread

LD_PRELOAD=$PWD/probe.so PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
    ./boot_trace <DUMP_ROOT>/<TITLE_ID>-app0 2> run.log

grep getenv-probe run.log
```

```
[getenv-probe] t=10s total=2002712  delta=2002712 rate=200271/s
[getenv-probe] t=90s total=24324828 delta=2935805 rate=293580/s
```

## Reading the output

- **Reports every 10 s from a background thread, not at exit.** A run ended by `timeout` dies on
  SIGTERM and never runs a destructor, so an exit-time report yields nothing — the first revision of
  this tool produced exactly that, an empty log after a 90-second run.
- **`delta`/`rate` matter more than `total`.** The interesting question is usually whether the rate
  *changes* — at a scene transition, when a movie starts, when a new thread appears. A steady rate
  and a rate that doubles at an FMV are different findings.
- **Expect one report stream per process.** `boot_trace` and the app may run helpers that inherit the
  preload; a stream that stays at `total=0` is a process that makes no `getenv` calls, not an error.

## What it does not tell you

The **cost** of those calls, which is the platform-specific half. This counts; it does not time. It
also cannot see calls made before the constructor runs, or calls inside statically-linked copies of
libc. And it is Linux/macOS only — `LD_PRELOAD` has no Windows equivalent, which is precisely why
the measurement is taken here and applied there.
