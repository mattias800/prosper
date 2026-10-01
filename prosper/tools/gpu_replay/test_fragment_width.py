#!/usr/bin/env python3
"""#4017: execute raw/tap fragment regeneration and evaluate the actual emitted MRT0 SSA.

No GPU or game runs here. Synthetic lane inputs evaluate the real high-half numeric MBCNT
expression; the metadata, typed live mask and output are separate checks. Restoring either old
integer-to-bool call must fail that route's default/64 arms while its 32 control stays green.
"""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


REPLAY = os.environ.get("PROSPER_GPU_REPLAY_BIN")
FIXTURE = os.environ.get("PROSPER_REPLAY_FRAGMENT_WIDTH_FIXTURE_BIN")
if not REPLAY or not Path(REPLAY).is_file() or not FIXTURE or not Path(FIXTURE).is_file():
    print("gpu_replay or replay-fragment-width fixture binary not built -- skipping")
    sys.exit(77)

failures = []


def check(condition, message):
    print("  [%s] %s" % ("ok" if condition else "FAIL", message))
    if not condition:
        failures.append(message)


def validator():
    explicit = os.environ.get("PROSPER_SPIRV_VAL_BIN")
    if explicit:
        return explicit if Path(explicit).is_file() else None
    found = shutil.which("spirv-val")
    if found:
        return found
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for relative in ("Bin/spirv-val.exe", "bin/spirv-val", "bin/spirv-val.exe"):
            candidate = Path(sdk) / relative
            if candidate.is_file():
                return str(candidate)
    return None


class Module:
    """Fail-closed evaluator for only the typed straight-line SSA used by this fixture.

    Unsupported reachable operations fail, rather than inventing zero or searching for an opcode
    elsewhere. Values are 32-bit raw bits, including float constants/bitcasts and vector members.
    """
    TYPED_RESULTS = {41, 42, 43, 59, 61, 80, 112, 124, 128, 130, 169, 170, 174, 196, 199, 205}

    def __init__(self, data):
        if len(data) < 20 or len(data) % 4:
            raise ValueError("not a complete SPIR-V module")
        words = struct.unpack("<%dI" % (len(data) // 4), data)
        if words[0] != 0x07230203:
            raise ValueError("bad SPIR-V magic")
        self.defs, self.builtins, self.locations = {}, {}, {}
        self.stores, self.markers = [], []
        pc = 5
        while pc < len(words):
            count, op = words[pc] >> 16, words[pc] & 0xffff
            if not count or count > len(words) - pc:
                raise ValueError("malformed instruction boundary")
            a = words[pc + 1:pc + count]
            if op in (21, 22, 23, 32):
                self.defs[a[0]] = (op, a)
            elif op in self.TYPED_RESULTS:
                self.defs[a[1]] = (op, a)
            elif op == 71 and len(a) == 3:
                if a[1] == 11:
                    self.builtins[a[0]] = a[2]
                elif a[1] == 30:
                    self.locations[a[0]] = a[2]
            elif op == 62:
                self.stores.append(a)
            elif op == 330:
                raw = struct.pack("<%dI" % len(a), *a)
                self.markers.append(raw.split(b"\0", 1)[0].decode("ascii"))
            pc += count
        sinks = []
        for pointer, value in self.stores:
            op, a = self.defs.get(pointer, (None, ()))
            if self.locations.get(pointer) == 0 and op == 59 and a[2] == 3:
                sinks.append(value)
        if len(sinks) != 1:
            raise ValueError("fixture must have exactly one MRT0 Output store")
        self.sink = sinks[0]
        op, a = self.defs[self.sink]
        if op != 80 or len(a) != 6:
            raise ValueError("MRT0 sink is not an explicit four-component vector")
        self.red = a[2]

    def unsigned32(self, type_id):
        return self.defs.get(type_id) == (21, (type_id, 32, 0))

    def constant(self, ident, value):
        op, a = self.defs.get(ident, (None, ()))
        return op == 43 and len(a) == 3 and self.unsigned32(a[0]) and a[2] == value

    def lane_load(self, ident):
        op, a = self.defs.get(ident, (None, ()))
        return (op == 61 and len(a) == 3 and self.unsigned32(a[0]) and
                self.builtins.get(a[2]) == 41)

    def ancestry(self, ident):
        found, pending = set(), [ident]
        while pending:
            ident = pending.pop()
            if ident in found:
                continue
            found.add(ident)
            op, a = self.defs.get(ident, (None, ()))
            if op in (80, 112, 124, 128, 130, 169, 170, 174, 196, 199, 205):
                pending.extend(a[2:])
        return found

    def live_width_mask(self, width):
        live = self.ancestry(self.red)
        masks = []
        for ident in live:
            op, a = self.defs.get(ident, (None, ()))
            if op == 199 and len(a) == 4 and self.unsigned32(a[0]):
                if ((self.lane_load(a[2]) and self.constant(a[3], width - 1)) or
                        (self.lane_load(a[3]) and self.constant(a[2], width - 1))):
                    masks.append(ident)
        return (len(masks) == 1 and any(
            op == 174 and len(a) == 4 and a[2] == masks[0] and self.constant(a[3], 32)
            for op, a in (self.defs.get(ident, (None, ())) for ident in live)) and any(
            op == 205 and self.unsigned32(a[0])
            for op, a in (self.defs.get(ident, (None, ())) for ident in live)))

    def output(self, lane):
        memo, active = {}, set()

        def evaluate(ident):
            if ident in memo:
                return memo[ident]
            if ident in active:
                raise ValueError("cycle in fixture SSA")
            active.add(ident)
            op, a = self.defs[ident]
            if op == 43:
                value = a[2]
            elif op in (41, 42):
                value = int(op == 41)
            elif op == 61 and self.lane_load(ident):
                value = lane
            elif op == 80:
                value = tuple(evaluate(source) for source in a[2:])
            elif op == 124:
                value = evaluate(a[2])
            elif op == 112:
                value = struct.unpack("<I", struct.pack("<f", float(evaluate(a[2]))))[0]
            elif op == 169:
                value = evaluate(a[3] if evaluate(a[2]) else a[4])
            elif op == 205:
                value = evaluate(a[2]).bit_count()
            elif op in (128, 130, 170, 174, 196, 199):
                lhs, rhs = evaluate(a[2]), evaluate(a[3])
                if op == 128:
                    value = (lhs + rhs) & 0xffffffff
                elif op == 130:
                    value = (lhs - rhs) & 0xffffffff
                elif op == 170:
                    value = int(lhs == rhs)
                elif op == 174:
                    value = int(lhs >= rhs)
                elif op == 196:
                    if rhs >= 32:
                        raise ValueError("undefined 32-bit shift in reachable mask expression")
                    value = (lhs << rhs) & 0xffffffff
                else:
                    value = lhs & rhs
            else:
                raise ValueError("unsupported reachable opcode %s at SSA %s" % (op, ident))
            active.remove(ident)
            memo[ident] = value
            return value

        return evaluate(self.sink)


def run(args, width=None, tap=False):
    env = dict(os.environ)
    # Inherited diagnostics may change stored/regenerated shaders or their environment replay.
    for name in tuple(env):
        if name.startswith("PROSPER_"):
            env.pop(name)
    if width is not None:
        env["PROSPER_GPU_REPLAY_FRAGMENT_WAVE_SIZE"] = width
    if tap:
        env["PROSPER_FS_TAP"] = "7:2"
    return subprocess.run([REPLAY] + args, capture_output=True, text=True, env=env, timeout=120)


print("== test_gpu_replay_fragment_width_cli ==")
VAL = validator()
check(VAL is not None, "strict spirv-val is available (absence is a failure, not a validation claim)")
scratch_root = Path(os.environ.get("PROSPER_TEST_SCRATCH_DIR") or
                    (Path.cwd() / "prosper-test-scratch"))
scratch_root.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="prosper-fragment-width-", dir=scratch_root) as temp:
    directory = Path(temp)
    fixture_env = dict(os.environ)
    fixture_env.pop("PROSPER_FS_TAP", None)
    fixture = subprocess.run([FIXTURE, "--write-fixture", temp], capture_output=True, text=True,
                             env=fixture_env, timeout=120)
    check(fixture.returncode == 0, "realized synthetic capture fixture builds without a GPU")
    if fixture.returncode:
        print(fixture.stdout + fixture.stderr)
    capture = directory / "fragment-width.prgcap"
    if capture.is_file():
        for route in ("recompile-raw", "fs-tap"):
            for requested in (None, "32", "64"):
                width = 32 if requested == "32" else 64
                label = "%s %s" % (route, requested or "default")
                path = directory / (label.replace(" ", "-") + ".spv")
                args = ["--inspect-only"]
                if route == "recompile-raw":
                    args.append("--recompile-raw")
                args += ["--dump-shader", "7:fs", str(path), str(capture)]
                done = run(args, requested, route == "fs-tap")
                check(done.returncode == 0 and path.is_file(), label + " regenerates and dumps actual FS")
                output = done.stdout + done.stderr
                source = "override" if requested else "legacy-default"
                check("[%s] fragment-wave=%d source=%s captured-width=unavailable" %
                      (route, width, source) in output,
                      label + " reports requested width and unavailable captured ABI honestly")
                if route == "recompile-raw":
                    check("substituted vs=1 fs=1 kept-stored vs=0 fs=0 of 1 draws" in output,
                          label + " substitutes instead of keeping the unrelated stored module")
                else:
                    check("draw=7 item=0 operation=0 FS re-recompiled at pc=2 with colour tap" in output,
                          label + " applies tap to the semantic draw")
                if not path.is_file():
                    print(output)
                    continue
                if VAL:
                    valid = subprocess.run([VAL, "--target-env", "vulkan1.1", str(path)],
                                           capture_output=True, text=True, timeout=30)
                    check(valid.returncode == 0, label + " passes strict Vulkan1.1 SPIR-V validation")
                    if valid.returncode:
                        print(valid.stdout + valid.stderr)
                try:
                    module = Module(path.read_bytes())
                    check(module.markers.count("Prosper.FragmentSubgroupSize=%d" % width) == 1,
                          label + " declares actual effective width")
                    check(module.live_width_mask(width),
                          label + " live MRT0 red traces typed width mask into high-half BitCount")
                    for lane in (0, 31, 32, 33, 63):
                        # Evaluate the emitted physical prefix, not a parallel reference compiler.
                        # IDs >=32 are symbolic inputs to prove the Wave32 wrap mask, not a claim
                        # that a native Wave32 GPU invocation can have those IDs.
                        red = 0x3f800000 if width == 64 and lane >= 33 else 0
                        expected = (red, 0 if route == "fs-tap" else 0x3f800000, 0,
                                    0 if route == "fs-tap" else 0x3f800000)
                        check(module.output(lane) == expected,
                              "%s actual MRT0 at symbolic lane %d is %s" % (label, lane, expected))
                except (ValueError, KeyError, IndexError, struct.error) as exc:
                    check(False, label + " actual SPIR-V semantic oracle: " + str(exc))
            # Every malformed-width arm must reject before overwriting an existing dump.
            for malformed in ("32junk", "64junk", "+32", "-64", " 32", "64 ", "", "128",
                              "18446744073709551616"):
                path = directory / (route + "-invalid.spv")
                sentinel = b"unchanged: malformed replay fragment width"
                path.write_bytes(sentinel)
                args = ["--inspect-only"] + (["--recompile-raw"] if route == "recompile-raw" else [])
                args += ["--dump-shader", "7:fs", str(path), str(capture)]
                done = run(args, malformed, route == "fs-tap")
                diagnostic = "[%s] invalid PROSPER_GPU_REPLAY_FRAGMENT_WAVE_SIZE:" % route
                check(done.returncode == 2 and diagnostic in done.stderr and
                      path.read_bytes() == sentinel,
                      "%s invalid %r fails closed and preserves dump" % (route, malformed))
        # An unrelated armed override must not rewrite stored shaders or get mistaken for an
        # attempted fragment regeneration. It is intentionally inactive without either route.
        stored = directory / "stored.spv"
        done = run(["--inspect-only", "--dump-shader", "7:fs", str(stored), str(capture)], "32junk")
        check(done.returncode == 0 and stored.is_file() and "fragment-wave=" not in done.stderr and
              "invalid PROSPER_GPU_REPLAY_FRAGMENT_WAVE_SIZE:" not in done.stderr,
              "plain stored replay leaves the unrelated width override dormant")
        if stored.is_file():
            try:
                check(Module(stored.read_bytes()).output(33) == (0, 0x3f800000, 0, 0x3f800000),
                      "dormant override preserves the distinct stored MRT0 value")
            except (ValueError, KeyError, IndexError, struct.error) as exc:
                check(False, "stored output oracle: " + str(exc))
    else:
        check(False, "fixture writer produced the realized capture")

print("== %s (%d failures) ==" % ("FAIL" if failures else "PASS", len(failures)))
sys.exit(1 if failures else 0)
