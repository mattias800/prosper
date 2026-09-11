#!/usr/bin/env python3
# test_cli_contract.py (#3372) -- the gpu_replay CLI must SAY what its two shader dumps produce.
#
# This is not a style guard. --dump-shader writes prosper's recompiled SPIR-V and the raw flag
# writes the guest's RDNA2 words that fed it; an investigation read the RDNA2 out of the flag whose
# name suggested SPIR-V, concluded no SPIR-V was obtainable, and filed #3372 -- a session lost to a
# usage string that listed the flags without saying which way round they go. Nothing else in the
# tree can fail when that wording is dropped, so this does.
#
# It drives the real binary rather than grepping the source: the point is what an operator SEES.
import os
import subprocess
import sys

REPLAY = os.environ.get("PROSPER_GPU_REPLAY_BIN")
if not REPLAY or not os.path.exists(REPLAY):
    # 77 -> ctest reports "(Skipped)" BY NAME. Exiting 0 here would make "the tool was never
    # built" and "the contract holds" the same result, which is the failure this file exists
    # to prevent. gpu_replay needs Vulkan, so it is genuinely absent in some environments.
    print("gpu_replay binary not built (PROSPER_GPU_REPLAY_BIN unset or missing) -- skipping")
    sys.exit(77)

failures = []


def check(condition, message):
    if condition:
        print("  [ok]   %s" % message)
    else:
        print("  [FAIL] %s" % message)
        failures.append(message)


def run(args, env_extra=None):
    env = dict(os.environ)
    env.pop("PROSPER_SHADER_DUMP_SUCCESS", None)
    if env_extra:
        env.update(env_extra)
    done = subprocess.run([REPLAY] + args, capture_output=True, text=True, env=env, timeout=120)
    return done.stdout + done.stderr


print("== test_gpu_replay_cli_contract ==")

# No arguments -> usage. The legend must travel with it.
usage = run([])
check("--dump-shader-raw" in usage, "usage names --dump-shader-raw")
check("--dump-realized-shader" in usage,
      "usage still names the --dump-realized-shader alias, so existing scripts stay discoverable")

# The two directions must each be spelled out, and on the correct flag. Find the legend line for
# each flag and assert what it claims -- asserting only that both words appear SOMEWHERE would
# pass with the meanings swapped, which is the exact defect #3372 reports.
lines = usage.splitlines()


def legend_line(flag):
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(flag + " ") or stripped.startswith(flag + "	"):
            return stripped
    return ""


spirv_line = legend_line("--dump-shader")
check("SPIR-V" in spirv_line and "RDNA2" not in spirv_line,
      "--dump-shader is described as producing SPIR-V, not RDNA2")

# The raw flag wraps onto continuation lines, so take the legend block after it.
raw_index = next((i for i, l in enumerate(lines) if "--dump-shader-raw" in l and "[" not in l), None)
check(raw_index is not None, "usage carries a legend entry for --dump-shader-raw")
raw_block = " ".join(lines[raw_index:raw_index + 4]) if raw_index is not None else ""
check("RDNA2" in raw_block, "--dump-shader-raw is described as producing the guest RDNA2")
check("alias" in raw_block and "--dump-realized-shader" in raw_block,
      "the legend names --dump-realized-shader as the alias of --dump-shader-raw")

# The compute pair is the naming precedent the graphics pair now follows; if it ever stops saying
# so, the -raw suffix loses the meaning this fix leaned on.
compute_raw = legend_line("--dump-compute-raw")
check("RDNA2" in compute_raw, "--dump-compute-raw is described as producing the guest RDNA2")

# The legend must not restate the absolute claim the notice used to make.
check("only in the modes that RECOMPILE" in usage,
      "the usage legend scopes PROSPER_SHADER_DUMP_SUCCESS to the recompiling modes")

# An armed-but-inert diagnostic must announce itself. An empty output directory otherwise reads as
# "that program never compiled" -- the trap src/gpu/diagnostics/AGENTS.md warns about.
#
# But it is inert only in SOME modes, and the first version of this notice claimed it was inert
# in all of them. --recompile-raw and --retry-failed-* recompile through compute_recompile.hpp ->
# gpu::recompile_compute_shader_cached, which calls maybe_dump_successful_shader on every exit
# path, so the variable genuinely works there. A notice that is wrong in one mode is the same
# wrong-text-at-the-point-of-use defect this whole file exists to guard, one level up -- so both
# directions are pinned below, not just the warning.

def notice_block(text):
    """Just the PROSPER_SHADER_DUMP_SUCCESS notice, never the usage synopsis.

    Measured necessity, not caution: searching the whole output for '--recompile-raw' PASSES
    against the pre-fix binary, because the synopsis lists the flag. Two of these arms were
    green in both directions until this helper existed -- the same defect they are guarding.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "PROSPER_SHADER_DUMP_SUCCESS" in line and line.lstrip().startswith("[gpureplay]"):
            block = [lines[i]]
            for cont in lines[i + 1:]:
                if not cont.startswith("            "):
                    break
                block.append(cont)
            return " ".join(block)
    return ""


armed_all = run([], {"PROSPER_SHADER_DUMP_SUCCESS": "ignored-on-this-path"})
armed = notice_block(armed_all)
check(armed != "",
      "a plain replay says PROSPER_SHADER_DUMP_SUCCESS is set when it is")
check("no file" in armed.lower() or "NOTHING" in armed,
      "...and says no file will be written for it on that path")
check("--recompile-raw" in armed,
      "...and names a mode that DOES honour it, rather than implying it never works")

# The other direction. A capture path that does not exist is fine: the notice is emitted after
# argument parsing and before the capture is opened, so if it were unconditional it would still
# appear here. It must not.
recompiling = run(["--recompile-raw", "no-such-capture.prgcap"],
                  {"PROSPER_SHADER_DUMP_SUCCESS": "honoured-on-this-path"})
check(notice_block(recompiling) == "",
      "--recompile-raw does NOT claim the variable is inert -- it honours it")

quiet = run([])
check(notice_block(quiet) == "",
      "the notice stays silent when the variable is unset")

print("%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
