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

# An armed-but-inert diagnostic must announce itself. An empty output directory otherwise reads as
# "that program never compiled" -- the trap src/gpu/diagnostics/AGENTS.md warns about.
armed = run([], {"PROSPER_SHADER_DUMP_SUCCESS": "ignored-by-this-tool"})
check("PROSPER_SHADER_DUMP_SUCCESS" in armed,
      "gpu_replay says PROSPER_SHADER_DUMP_SUCCESS is set when it is")
check("NOTHING" in armed or "does not" in armed or "no file" in armed.lower(),
      "gpu_replay says the variable will produce no file here")

quiet = run([])
check("is set but does NOTHING here" not in quiet,
      "the notice stays silent when the variable is unset")

print("%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
