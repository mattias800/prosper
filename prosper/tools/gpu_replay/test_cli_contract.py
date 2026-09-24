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
from pathlib import Path
import re
import subprocess
import sys
import tempfile

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


def run_result(args, env_extra=None):
    env = dict(os.environ)
    env.pop("PROSPER_SHADER_DUMP_SUCCESS", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([REPLAY] + args, capture_output=True, text=True, env=env, timeout=120)


def run(args, env_extra=None):
    done = run_result(args, env_extra)
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

# Direct graphics emitters bypass the executor cache's successful-shader dump hook.
check("honoured by cached COMPUTE recompiles" in usage and
      "Graphics recompiles bypass" in usage,
      "the usage legend scopes PROSPER_SHADER_DUMP_SUCCESS to cached compute recompiles")

# An armed-but-inert diagnostic must announce itself. An empty output directory otherwise reads as
# "that program never compiled" -- the trap src/gpu/diagnostics/AGENTS.md warns about.
#
# Compute --recompile-raw and --retry-failed-stage use recompile_compute_shader_cached, which
# calls the hook. Graphics retries invoke direct emitters instead, so even successful translation
# writes nothing through this variable. Pin both directions rather than equating all recompiles.

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
      "--recompile-raw does not claim the variable is inert before knowing its compute stages")

chain = run(["--retry-failed-chain", "0", "no-such-capture.prgcap"],
            {"PROSPER_SHADER_DUMP_SUCCESS": "unused-for-graphics"})
check("no file" in notice_block(chain) and "--retry-failed-stage-spv" in notice_block(chain),
      "graphics chain retry identifies the inactive hook and names explicit SPIR-V export")
missing_chain = run_result(["--probe-ngg-workgroup-s3", "0x40004040",
                            "no-such-capture.prgcap"])
check(missing_chain.returncode == 2 and "requires --retry-failed-chain" in missing_chain.stderr,
      "workgroup compile probe refuses to become an ignored standalone selector")
missing_s3 = run_result(["--probe-ngg-packed-offsets", "no-such-capture.prgcap"])
check(missing_s3.returncode == 2 and "requires --probe-ngg-workgroup-s3" in missing_s3.stderr,
      "explicit GS offset input cannot silently run outside the workgroup probe")
missing_full_chain = run_result(["--probe-ngg-full-four-wave", "no-such-capture.prgcap"])
check(missing_full_chain.returncode == 2 and "requires a chain retry" in missing_full_chain.stderr,
      "full launch probe cannot silently run outside a failed-chain retry")
missing_full_native = run_result(["--probe-ngg-native-wave64", "no-such-capture.prgcap"])
check(missing_full_native.returncode == 2 and
      "requires --probe-ngg-full-four-wave" in missing_full_native.stderr,
      "native Wave64 selector cannot silently run outside the full launch probe")
missing_trace_native = run_result(["--probe-ngg-trace=7:6", "no-such-capture.prgcap"])
check(missing_trace_native.returncode == 2 and
      "requires --probe-ngg-native-wave64" in missing_trace_native.stderr,
      "NGG trace cannot silently run outside the native compile-only probe")
for invalid in ("bad:6", "7:256", "7:"):
    result = run_result([f"--probe-ngg-trace={invalid}", "no-such-capture.prgcap"])
    check(result.returncode == 2 and "expected --probe-ngg-trace=PC:VGPR" in result.stderr,
          f"NGG trace rejects malformed selector {invalid}")
duplicate_trace = run_result(["--probe-ngg-trace=7:6", "--probe-ngg-trace=8:6",
                              "no-such-capture.prgcap"])
check(duplicate_trace.returncode == 2 and "duplicate NGG trace" in duplicate_trace.stderr,
      "NGG trace refuses duplicate selectors")
armed_trace = run_result(["--retry-failed-chain", "0", "--probe-ngg-full-four-wave",
                          "--probe-ngg-native-wave64", "--probe-ngg-trace=7:6",
                          "no-such-capture.prgcap"])
check(armed_trace.returncode == 2 and
      ("no file" in armed_trace.stderr or "invalid capture file size" in armed_trace.stderr) and
      "expected --probe-ngg-trace" not in armed_trace.stderr,
      "valid NGG trace selector reaches capture loading")

quiet = run([])
check(notice_block(quiet) == "",
      "the notice stays silent when the variable is unset")

# A requested retry export must not be ignored or truncate an existing file on rejection.
check("SPIR-V" in legend_line("--retry-failed-stage-spv"),
      "retry export is described as producing SPIR-V")
check("SPIR-V" in legend_line("--retry-failed-chain-spv"),
      "chain retry export is described as producing SPIR-V")
with tempfile.TemporaryDirectory(prefix="gpu-replay-retry-") as scratch:
    directory = Path(scratch)
    output = directory / "retry.spv"
    sentinel = b"existing output must survive rejected retries"
    output.write_bytes(sentinel)
    chain_output = directory / "chain.spv"
    chain_output.write_bytes(sentinel)
    chain_export = ["--retry-failed-chain-spv", str(chain_output)]
    for args, diagnostic, message in [
        (chain_export, "requires a standalone chain retry", "chain export requires its selector"),
        (["--retry-failed-chain-spv"], "needs a PATH", "chain export rejects a missing path"),
        (chain_export + ["--retry-failed-chain", "0", "--bundle", "absent.prgbundle"],
         "requires a standalone chain retry", "chain export rejects bundle replay"),
    ]:
        result = run_result(args)
        check(result.returncode == 2 and diagnostic in result.stderr and
              chain_output.read_bytes() == sentinel, message)
    export = ["--retry-failed-stage-spv", str(output)]
    for args, diagnostic, message in [
        (export, "requires --retry-failed-stage", "retry export requires its stage selector"),
        (["--retry-failed-stage-spv"], "needs a PATH", "retry export rejects a missing path"),
        (["--retry-failed-stage-spv", ""], "needs a PATH", "retry export rejects an empty path"),
        (export + ["--retry-failed-stage", "0:0", "--bundle", "absent.prgbundle"],
         "requires a standalone capture, not --bundle", "retry export rejects bundle replay"),
        (export + ["--retry-failed-stage", "0:0", "--retry-failed-chain", "0"],
         "cannot combine with another terminal diagnostic", "retry export cannot be bypassed by chain retry"),
    ]:
        result = run_result(args)
        check(result.returncode == 2 and diagnostic in result.stderr and
              output.read_bytes() == sentinel, message)

    fixture_bin = os.environ.get("PROSPER_GPU_REPLAY_FIXTURE_BIN")
    check(bool(fixture_bin and os.path.exists(fixture_bin)),
          "CPU retry fixture generator is available")
    if fixture_bin and os.path.exists(fixture_bin):
        fixture = subprocess.run([fixture_bin, "--write-retry-fixture", scratch],
                                 capture_output=True, text=True, timeout=120)
        check(fixture.returncode == 0, "synthetic retry captures are written without Vulkan")
        if fixture.returncode != 0:
            print(fixture.stdout + fixture.stderr)
        else:
            unknown_instances = run_result(["--inspect-only", str(directory / "unknown-instances.prgcap")])
            known_instances = run_result(["--inspect-only", str(directory / "known-instances.prgcap")])
            check(unknown_instances.returncode == 0 and "instances=?" in unknown_instances.stdout and
                  "instances=0" not in unknown_instances.stdout,
                  "a v58 pre-realization failure prints unknown rather than a fabricated zero")
            check(known_instances.returncode == 0 and "instances=32" in known_instances.stdout,
                  "a v58 failed draw prints its positive instance count")
            check(known_instances.returncode == 0 and
                  "target-volume slot=0 mip-depth=32 raw-start=0 raw-max=32 "
                  "bounded-start=0 bounded-count=32 base=000000309cbf0000 mask=f"
                  in known_instances.stdout and
                  "target-volume slot=1 mip-depth=64 raw-start=0 raw-max=64 "
                  "bounded-start=0 bounded-count=64 base=000000304edd0000 mask=0"
                  in known_instances.stdout and
                  "target-volume" not in unknown_instances.stdout,
                  "inspect distinguishes active volume output from sticky inactive view state")
            chain = run_result(["--retry-failed-chain", "0",
                                str(directory / "rejected-chain.prgcap")],
                               {"PROSPER_DBG_PROGRAM": "0x2000"})
            check(chain.returncode == 1 and
                  "config=defaults lds=0 pixel-inputs=0 capture-position=0 reason=" in chain.stderr and
                  "reason=mbcnt-cross-lane" in chain.stderr and
                  "program=0x2000" in chain.stderr,
                  "split vertex retry names its terminal cross-lane refusal with default config")
            exact_chain = run_result(["--retry-failed-chain", "0",
                                      str(directory / "rejected-chain-exact.prgcap")],
                                     {"PROSPER_DBG_PROGRAM": "0x2000"})
            check(exact_chain.returncode == 1 and
                  "config=captured lds=2176 pixel-inputs=1 capture-position=1" in exact_chain.stderr and
                  "reason=mbcnt-cross-lane" in exact_chain.stderr and
                  "program=0x2000" in exact_chain.stderr,
                  "split vertex retry uses the captured graphics ABI and names its refusal")
            probe_capture = str(directory / "ngg-probe-chain-exact.prgcap")
            workgroup_probe = run_result(["--retry-failed-chain", "0",
                                          "--probe-ngg-workgroup-s3", "0x40004040",
                                          probe_capture])
            check(workgroup_probe.returncode == 0 and
                  "[ngg-workgroup-probe] COMPILE ONLY" in workgroup_probe.stderr and
                  "provisional-s3=0x40004040" in workgroup_probe.stderr and
                  "result=module-emitted" in workgroup_probe.stderr,
                  "captured workgroup probe actually emits the supported guest export stream")
            chain_module = run_result(["--retry-failed-chain", "0",
                                       "--probe-ngg-workgroup-s3", "0x40004040",
                                       "--retry-failed-chain-spv", str(chain_output),
                                       probe_capture])
            word_count = re.search(r"spirv-dwords=(\d+)", chain_module.stderr)
            check(chain_module.returncode == 0 and word_count and
                  len(chain_output.read_bytes()) == int(word_count.group(1)) * 4 and
                  chain_output.read_bytes()[:4] == b"\x03\x02\x23\x07",
                  "accepted chain export writes the exact reported SPIR-V module")
            packed_output = directory / "packed-offsets.spv"
            packed_module = run_result(["--retry-failed-chain", "0",
                                        "--probe-ngg-workgroup-s3", "0x40004040",
                                        "--probe-ngg-packed-offsets",
                                        "--retry-failed-chain-spv", str(packed_output),
                                        probe_capture])
            check(packed_module.returncode == 0 and
                  "packed-offsets=1" in packed_module.stderr and
                  packed_output.exists() and
                  packed_output.read_bytes()[:4] == b"\x03\x02\x23\x07" and
                  packed_output.read_bytes() != chain_output.read_bytes(),
                  "GS offset input changes the actual compiled probe module")
            full_output = directory / "four-wave.spv"
            full_module = run_result(["--retry-failed-chain", "0",
                                      "--probe-ngg-full-four-wave",
                                      "--retry-failed-chain-spv", str(full_output),
                                      probe_capture])
            check(full_module.returncode == 0 and
                  "full-four-wave=1" in full_module.stderr and
                  full_output.exists() and
                  full_output.read_bytes()[:4] == b"\x03\x02\x23\x07" and
                  full_output.read_bytes() != packed_output.read_bytes(),
                  "full four-wave selector compiles a distinct workgroup module")
            native_output = directory / "four-wave-native64.spv"
            native_module = run_result(["--retry-failed-chain", "0",
                                        "--probe-ngg-full-four-wave",
                                        "--probe-ngg-native-wave64",
                                        "--retry-failed-chain-spv", str(native_output),
                                        probe_capture])
            check(native_module.returncode == 0 and
                  "native-wave64=1" in native_module.stderr and
                  native_output.exists() and
                  native_output.read_bytes()[:4] == b"\x03\x02\x23\x07" and
                  b"Prosper.NggProbeExactSubgroup=64" in native_output.read_bytes() and
                  native_output.read_bytes() != full_output.read_bytes(),
                  "native Wave64 selector compiles a distinct full-workgroup module")
            capture_probe = os.environ.get("PROSPER_NGG_CAPTURE_PROBE_BIN")
            if capture_probe:
                check(os.path.exists(capture_probe),
                      "captured-input probe binary is available")
                if os.path.exists(capture_probe) and native_output.exists():
                    absent_capture = str(directory / "absent.prgcap")
                    probe_output = directory / "unexpected-probe-output.bin"
                    valid_inputs = directory / "full-inputs.bin"
                    valid_inputs.write_bytes(bytes(256 * 10 * 4))

                    def run_full_probe(module, inputs):
                        return subprocess.run(
                            [capture_probe, absent_capture, "0", str(module),
                             str(probe_output), "--full-inputs=" + str(inputs)],
                            capture_output=True, text=True, timeout=30)

                    short_inputs = directory / "short-inputs.bin"
                    short_inputs.write_bytes(valid_inputs.read_bytes()[:-4])
                    long_inputs = directory / "long-inputs.bin"
                    long_inputs.write_bytes(valid_inputs.read_bytes() + bytes(4))
                    for malformed in (short_inputs, long_inputs):
                        refused = run_full_probe(native_output, malformed)
                        check(refused.returncode == 2 and
                              "requires exactly 10240 bytes" in refused.stderr and
                              "capture:" not in refused.stderr and
                              not probe_output.exists(),
                              "full launch rejects " + malformed.name + " before loading capture")
                    nonuniform_inputs = directory / "nonuniform-inputs.bin"
                    nonuniform = bytearray(valid_inputs.read_bytes())
                    nonuniform[(10 + 9) * 4] = 1  # lane 1, s3, wave 0
                    nonuniform_inputs.write_bytes(nonuniform)
                    refused = run_full_probe(native_output, nonuniform_inputs)
                    check(refused.returncode == 2 and
                          "s3 must be uniform within guest wave 0" in refused.stderr and
                          "capture:" not in refused.stderr and
                          not probe_output.exists(),
                          "full launch rejects one-lane s3 mutation before loading capture")
                    wrong_module = run_full_probe(full_output, valid_inputs)
                    check(wrong_module.returncode == 2 and
                          "requires a 256x1x1 native-Wave64 probe module" in wrong_module.stderr and
                          "capture:" not in wrong_module.stderr and
                          not probe_output.exists(),
                          "full launch rejects an unmarked 256-lane module")
                    valid_shape = run_full_probe(native_output, valid_inputs)
                    check(valid_shape.returncode == 2 and
                          ("requires supported exact 64-lane full subgroups" in valid_shape.stderr or
                           "capture:" in valid_shape.stderr) and
                          "requires exactly" not in valid_shape.stderr and
                          "s3 must be uniform" not in valid_shape.stderr and
                          not probe_output.exists(),
                          "full launch accepts the input shape before device or capture refusal")
            rejected_export = run_result(["--retry-failed-chain", "0",
                                          "--retry-failed-chain-spv", str(chain_output),
                                          str(directory / "rejected-chain.prgcap")])
            check(rejected_export.returncode == 1 and
                  "refusing chain SPIR-V export" in rejected_export.stderr and
                  chain_output.read_bytes()[:4] == b"\x03\x02\x23\x07",
                  "rejected chain leaves the prior module intact")
            bad_s3 = run_result(["--retry-failed-chain", "0",
                                 "--probe-ngg-workgroup-s3", "not-hex", probe_capture])
            check(bad_s3.returncode == 2 and "explicit 0xS3" in bad_s3.stderr and
                  "[ngg-workgroup-probe]" not in bad_s3.stderr,
                  "workgroup probe refuses an unreviewable implicit merged-wave seed")
            lds_exact = run_result(["--retry-failed-chain", "0",
                                    str(directory / "lds-chain-exact.prgcap")])
            lds_default = run_result(["--retry-failed-chain", "0",
                                      str(directory / "lds-chain-default.prgcap")])
            check(lds_exact.returncode == 0 and "recompiled" in lds_exact.stderr and
                  "config=captured lds=7" in lds_exact.stderr and
                  lds_default.returncode == 1 and "rejected" in lds_default.stderr and
                  "config=defaults lds=0" in lds_default.stderr,
                  "captured LDS changes real chain admission while identical default-config bytes reject")
            input_known = run_result(["--retry-failed-chain", "0",
                                      str(directory / "lds-inputs-known.prgcap")])
            input_unknown = run_result(["--retry-failed-chain", "0",
                                        str(directory / "lds-inputs-unknown.prgcap")])
            known_words = re.search(r"spirv-dwords=(\d+)", input_known.stderr)
            unknown_words = re.search(r"spirv-dwords=(\d+)", input_unknown.stderr)
            check(input_known.returncode == 0 and input_unknown.returncode == 0 and
                  "config=captured lds=7 pixel-inputs=1" in input_known.stderr and
                  "config=captured lds=7 pixel-inputs=1" in input_unknown.stderr and
                  known_words is not None and unknown_words is not None and
                  int(known_words.group(1)) != int(unknown_words.group(1)),
                  "captured fragment consumption reaches successful vertex-chain recompilation")
            args = ["--retry-failed-stage", "0:0"] + export
            accepted = run_result(args + [str(directory / "accepted.prgcap")])
            expected = (directory / "expected.spv").read_bytes()
            check(accepted.returncode == 0 and "contract=accepted" in accepted.stderr and
                  len(expected) > 20 and expected[:4] == b"\x03\x02\x23\x07" and
                  output.read_bytes() == expected,
                  "CPU retry exports the exact successful module before returning")
            if accepted.returncode != 0:
                print(accepted.stdout + accepted.stderr)
            output.write_bytes(sentinel)
            rejected = run_result(args + [str(directory / "rejected.prgcap")])
            check(rejected.returncode == 1 and "refusing retry SPIR-V export" in rejected.stderr and
                  output.read_bytes() == sentinel,
                  "rejected retry refuses export and preserves existing output")
            descriptor_rejected = run_result(args + [str(directory / "descriptor-rejected.prgcap")])
            check(descriptor_rejected.returncode == 1 and
                  "recompiled resources=1" in descriptor_rejected.stderr and
                  "contract=rejected" in descriptor_rejected.stderr and
                  "contract error=undersized buffer" in descriptor_rejected.stderr and
                  "refusing retry SPIR-V export" in descriptor_rejected.stderr and
                  output.read_bytes() == sentinel,
                  "nonempty descriptor-rejected retry refuses export and preserves existing output")
            unwritable = run_result(["--retry-failed-stage", "0:0", "--retry-failed-stage-spv",
                                     str(directory / "absent" / "retry.spv"),
                                     str(directory / "accepted.prgcap")])
            check(unwritable.returncode == 2 and "cannot write retry SPIR-V" in unwritable.stderr,
                  "retry export reports output file errors")

# A failed-prefix compute snapshot must never land at an ordinary-looking path. These parser
# controls run before capture loading and preserve the output sentinel; the retained Kena capsule
# separately exercises strict refusal versus the opt-in unverified write after five failed prefix
# dispatches.
with tempfile.TemporaryDirectory(prefix="gpu-replay-post-unverified-") as scratch:
    directory = Path(scratch)
    plain = directory / "post.linear"
    plain.write_bytes(b"do not overwrite verified-looking output")
    bad_suffix = run_result(["--dump-post-compute-resource-unverified", "0:1",
                             str(plain), "absent.prgcap"])
    check(bad_suffix.returncode == 2 and "must end in .unverified" in bad_suffix.stderr and
          plain.read_bytes() == b"do not overwrite verified-looking output",
          "unverified compute dump refuses a verified-looking path before reading the capture")

    unverified = directory / "post.linear.unverified"
    duplicate = run_result(["--dump-post-compute-resource", "", str(plain),
                            "--dump-post-compute-resource-unverified", "0:1",
                            str(unverified), "absent.prgcap"])
    check(duplicate.returncode == 2 and "duplicate post-compute dump selector" in duplicate.stderr and
          not unverified.exists() and plain.read_bytes() == b"do not overwrite verified-looking output",
          "duplicate selector cannot hide behind an empty first selector or write either path")

    accepted = run_result(["--dump-post-compute-resource-unverified", "0:1",
                           str(unverified), "absent.prgcap"])
    check(accepted.returncode == 2 and "absent.prgcap" in accepted.stderr and
          "must end in .unverified" not in accepted.stderr and not unverified.exists(),
          "a suffixed opt-in reaches capture loading without fabricating an output")

print("%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
