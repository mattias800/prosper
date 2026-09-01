// env_cache.hpp — one-shot reads of PROSPER_* diagnostic switches, for HOT paths.
//
// These two macros used to live in hle/dispatch/dispatch.hpp, which is where they were first
// needed. They are not an HLE concept: the renderer backend, the draw executor and the live
// frontend all evaluate diagnostic gates on per-draw and per-resource paths, and none of them
// should have to pull in the HLE dispatch registry (self/module.hpp, the NID tables, the import
// slot types) to ask "is this switch on?". dispatch.hpp includes this header, so every existing
// user keeps compiling unchanged.
//
// WHY THEY EXIST
//
// getenv() is not free: it walks the environment block comparing each entry, on EVERY call. A
// diagnostic that is switched OFF therefore still costs that scan every time its guard is
// evaluated -- and some of these guards sit on per-draw and per-RESOURCE paths, so the cost scales
// with scene complexity rather than with logging.
//
// Measured on Blue Prince (native Windows), sampling the guest primary thread with every
// diagnostic disabled: 97 of 331 samples -- ~30% of that thread's wall time -- were inside
// getenv's strlen loop. On Linux/AMD a 15 s perf record of the gameplay frame put getenv at 1.24%
// of the whole frame (#3094, decomposition in #1284).
//
// THE UNIT IS THE CALL SITE, NOT THE NAME. Each textual expansion declares its own
// function-local static inside its own lambda, so one site reads once however many times it is
// evaluated -- which is the case that matters, since a per-draw guard is a single site evaluated
// thousands of times -- while two sites reading the SAME name each read once, independently. Do
// not reason as though caching a name anywhere caches it everywhere; tests/diagnostics/
// test_env_cache.cpp pins both halves.
//
// A function-local static is initialized thread-safely under C++11 and later, which is why the
// storage is inside the lambda rather than at namespace scope. Namespace-scope storage would also
// be exposed to static-initialization order across translation units; this is not.
//
// THIS CHANGES SEMANTICS, and that is why it is not applied everywhere: the variable is sampled at
// first use, so a caller that sets it LATER never sees it. Several tests arm diagnostics at runtime
// with setenv/_putenv_s and then assert on the behaviour -- and such a test does not FAIL when the
// read is cached, it goes VACUOUS and keeps printing [ok], because the assertion still holds for
// the stale value (#2214). tools/env/check_cached_env.py is the gate that refuses any name
// something in the tree arms at runtime; run it before caching a new name.
//
// The historical record, because it is easy to misread as a blanket prohibition: an early sweep
// that applied these macros to hle_kernel.cpp, gpu_executor and the command processor broke
// win_exception_delivery, eop_write and dynfetch_fold. That was never a property of those FILES --
// it was the specific names they read (PROSPER_WIN_*_EXC, PROSPER_GDSLOG / PROSPER_WRITE_TRAP,
// PROSPER_DYNTRACE_FAIL*), every one of which a test arms between phases, and every one of which
// check_cached_env.py now refuses by name. #3094 cached sixteen names in gpu_execute.hpp -- on the
// same per-draw path, reached from gpu_executor.cpp -- with the suite green, because the gate
// separates the two questions. Screen by NAME, not by file.
//
// A SECOND ARMING MECHANISM THE GATE CANNOT SEE, recorded here because it is invisible at the call
// site: tools/gpu_replay re-applies `GpuCaptureMetadata::renderer_env` -- the allowlist at
// gpu_capture.cpp's `render_env[]` -- through set_environment() with a NON-literal name, once per
// bundle submit. check_cached_env.py matches on literal first arguments, so it cannot see those.
// Their values come from the capturing process's own (constant) environment, so no defect is
// demonstrated today; the names are nonetheless left as live reads, because a bundle replay is the
// project's primary offline debugging tool and a diagnostic silently frozen across submits would
// be a trap. Do not cache a name in render_env[] without re-deriving that argument.
//
// Use these only where the flag is a pure boot-time diagnostic switch AND the path is hot, and run
// the suite after.
#pragma once
#include <cstdlib>

// Cached presence check: "is this diagnostic switched on?"
#define PROSPER_ENV_ON(name) ([]() -> bool { static const bool prosper_env_on_v = std::getenv(name) != nullptr; return prosper_env_on_v; }())

// The same, for a diagnostic that reads a VALUE rather than just presence. Same one-shot read and
// the same consequence: sampled at first use, so a caller that sets it later never sees it.
//
// Worth its own macro because the value form is the one that hides: `const char* m = getenv(...)`
// does not look like a flag check, so a boolean-guard sweep skips it -- and in live_renderer.cpp
// the single hottest getenv site was exactly that shape, on a per-RESOURCE path, calling getenv up
// to three times per resource.
#define PROSPER_ENV_VALUE(name) ([]() -> const char* { static const char* prosper_env_value_v = std::getenv(name); return prosper_env_value_v; }())
