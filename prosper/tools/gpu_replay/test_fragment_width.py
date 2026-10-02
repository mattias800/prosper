#!/usr/bin/env python3
"""#4017/#4041: evaluate real raw/tap regeneration with captured or requested launch width.

No GPU or game runs here. Synthetic lane inputs evaluate the real high-half numeric MBCNT
expression; the metadata, typed live mask and output are separate checks. Restoring either old
integer-to-bool call must fail that route's 64 arms while its 32 control stays green. Ignoring
capture ABI or override priority must change actual high-half outputs, not just a notice.
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


def run(args, width=None, tap=False, draw=7):
    env = dict(os.environ)
    # Inherited diagnostics may change stored/regenerated shaders or their environment replay.
    for name in tuple(env):
        if name.startswith("PROSPER_"):
            env.pop(name)
    if width is not None:
        env["PROSPER_GPU_REPLAY_FRAGMENT_WAVE_SIZE"] = width
    if tap:
        env["PROSPER_FS_TAP"] = "%d:2" % draw
    return subprocess.run([REPLAY] + args, capture_output=True, text=True, env=env, timeout=120)


def dump_args(capture, path, route=None, draw=7):
    args = ["--inspect-only"]
    if route == "recompile-raw":
        args.append("--recompile-raw")
    return args + ["--dump-shader", "%d:fs" % draw, str(path), str(capture)]


def expected_output(width, lane, tap=False):
    # IDs >=32 are symbolic inputs to prove the Wave32 wrap mask, not a claim that a native
    # Wave32 invocation can have those IDs. MBCNT_HI(mask=1, accumulator=0) becomes one only
    # after lane32, and Wave32 never consumes the high-half mask.
    red = 0x3f800000 if width == 64 and lane >= 33 else 0
    return (red, 0 if tap else 0x3f800000, 0, 0 if tap else 0x3f800000)


mutation_controls = set()


def inspect_module(path, label, width, tap=False, route="retry-failed-stage"):
    if VAL:
        valid = subprocess.run([VAL, "--target-env", "vulkan1.1", str(path)],
                               capture_output=True, text=True, timeout=30)
        check(valid.returncode == 0, label + " passes strict Vulkan1.1 SPIR-V validation")
        if valid.returncode:
            print(valid.stdout + valid.stderr)
    try:
        data = path.read_bytes()
        module = Module(data)
        check(module.markers.count("Prosper.FragmentSubgroupSize=%d" % width) == 1,
              label + " declares compiler launch width (not effective backend width)")
        check(module.live_width_mask(width),
              label + " live MRT0 red traces typed width mask into high-half BitCount")
        for lane in (0, 31, 32, 33, 63):
            expected = expected_output(width, lane, tap)
            check(module.output(lane) == expected,
                  "%s actual MRT0 at symbolic lane %d is %s" % (label, lane, expected))
        opposing = 64 if width == 32 else 32
        check(not module.live_width_mask(opposing) and
              module.output(33) != expected_output(opposing, 33, tap),
              label + " distinguishes opposing width through actual output, not metadata alone")

        # Mutate the actual live width-mask constant once per route/width, leaving every marker
        # untouched. A marker-only oracle would stay green; this actual MRT0 expression changes.
        control = (route, width)
        if control not in mutation_controls:
            mutation_controls.add(control)
            live = module.ancestry(module.red)
            mask_constant = None
            for ident in live:
                op, a = module.defs.get(ident, (None, ()))
                if op == 199 and len(a) == 4:
                    for lane_id, constant_id in ((a[2], a[3]), (a[3], a[2])):
                        if module.lane_load(lane_id) and module.constant(constant_id, width - 1):
                            mask_constant = constant_id
            if mask_constant is None:
                raise ValueError("no live mask constant for mutation control")
            words = list(struct.unpack("<%dI" % (len(data) // 4), data))
            pc, mutated = 5, 0
            while pc < len(words):
                count, op = words[pc] >> 16, words[pc] & 0xffff
                if op == 43 and count == 4 and words[pc + 2] == mask_constant:
                    words[pc + 3] = opposing - 1
                    mutated += 1
                pc += count
            broken = Module(struct.pack("<%dI" % len(words), *words))
            check(mutated == 1 and broken.markers == module.markers and
                  broken.output(33) != expected_output(width, 33, tap),
                  label + " live-width mutation changes MRT0 despite unchanged width notice/marker")
    except (ValueError, KeyError, IndexError, struct.error) as exc:
        check(False, label + " actual SPIR-V semantic oracle: " + str(exc))


def regeneration(capture, state, captured_width, route, requested, directory,
                 draw=7, item=0, operation=0, draw_count=1):
    width = int(requested) if requested is not None else (captured_width or 64)
    label = "%s %s draw%d %s" % (route, state, draw, requested or "default")
    path = directory / (label.replace(" ", "-") + ".spv")
    done = run(dump_args(capture, path, route, draw), requested, route == "fs-tap", draw)
    check(done.returncode == 0 and path.is_file(), label + " regenerates and dumps actual FS")
    output = done.stdout + done.stderr
    source = "override" if requested is not None else ("captured" if captured_width else "legacy-default")
    origin = str(captured_width) if captured_width else "unavailable"
    notice = "[%s] fragment-wave=%d source=%s captured-width=%s draw=%d; " % (
        route, width, source, origin, draw)
    check(notice in output and "guest compile request, not effective backend subgroup width" in output,
          label + " separates requested selection from captured origin and backend width")
    if route == "recompile-raw":
        check("substituted vs=%d fs=%d kept-stored vs=0 fs=0 of %d draws" %
              (draw_count, draw_count, draw_count) in output,
              label + " substitutes instead of keeping the unrelated stored module")
    else:
        check("draw=%d item=%d operation=%d FS re-recompiled at pc=2 with colour tap" %
              (draw, item, operation) in output,
              label + " applies tap to the selected semantic draw")
    if path.is_file():
        inspect_module(path, label, width, route == "fs-tap", route)
    else:
        print(output)


print("== test_gpu_replay_fragment_width_cli ==")
VAL = validator()
check(VAL is not None, "strict spirv-val is available (absence is a failure, not a validation claim)")
scratch_root = Path(os.environ.get("PROSPER_TEST_SCRATCH_DIR") or
                    (Path.cwd() / "prosper-test-scratch"))
scratch_root.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="prosper-fragment-width-", dir=scratch_root) as temp:
    directory = Path(temp)
    fixture_env = dict(os.environ)
    for name in tuple(fixture_env):
        if name.startswith("PROSPER_"):
            fixture_env.pop(name)
    fixture = subprocess.run([FIXTURE, "--write-fixture", temp], capture_output=True, text=True,
                             env=fixture_env, timeout=120)
    check(fixture.returncode == 0, "realized synthetic capture fixture builds without a GPU")
    if fixture.returncode:
        print(fixture.stdout + fixture.stderr)
    states = [("unknown-current", "fragment-width.prgcap", None),
              ("captured32", "fragment-width-captured32.prgcap", 32),
              ("captured64", "fragment-width-captured64.prgcap", 64)]
    known32 = directory / "fragment-width-captured32.prgcap"
    if known32.is_file():
        original = known32.read_bytes()
        flags_tail = struct.pack("<I", 1) + b"\x00\x00\x00" + struct.pack("<I", 0)
        genuine66 = (len(original) > 59 and struct.unpack_from("<I", original, 8)[0] == 66 and
                     original.endswith(flags_tail))
        check(genuine66, "current width fixture retains canonical unknown-flags v66 tail")
        if genuine66:
            original = bytearray(original[:-len(flags_tail)])
            struct.pack_into("<I", original, 8, 65)
            original = bytes(original)
        transport_tail = struct.pack("<I", 1) + b"\x00" + struct.pack("<II", 0, 0)
        genuine65 = (len(original) > 48 and struct.unpack_from("<I", original, 8)[0] == 65 and
                     original.endswith(transport_tail))
        check(genuine65, "current width fixture retains canonical unknown-transport v65 tail")
        if genuine65:
            original = bytearray(original[:-len(transport_tail)])
            struct.pack_into("<I", original, 8, 64)
            original = bytes(original)
        genuine64 = (len(original) > 35 and struct.unpack_from("<I", original, 8)[0] == 64 and
                     original[-10:] == struct.pack("<I", 1) + b"\x00\x00" + struct.pack("<I", 0))
        check(genuine64, "current width fixture retains canonical unknown-mode v64 tail")
        if genuine64:
            original = bytearray(original[:-10])
            struct.pack_into("<I", original, 8, 63)
            original = bytes(original)
        genuine63 = (len(original) > 25 and struct.unpack_from("<I", original, 8)[0] == 63 and
                     original[-5:] == struct.pack("<I", 1) + b"\x02")
        check(genuine63, "known32 prefix is genuine v63 with exact one-draw Wave32 tail")
        if genuine63:
            # Relabeling alone leaves the new tail and is NOT a legacy-format fixture. Strip the
            # exact appended draw count/tag; the original known32 is now genuinely unavailable.
            legacy = bytearray(original[:-5])
            struct.pack_into("<I", legacy, 8, 62)
            legacy_path = directory / "fragment-width-v62.prgcap"
            legacy_path.write_bytes(legacy)
            states.append(("legacy62-from32", legacy_path.name, None))
            bad_captures = {
                "count": (original[:-5] + struct.pack("<I", 2) + b"\x02",
                          "invalid realized-draw fragment wave count"),
                "tag": (original[:-1] + b"\x03", "invalid realized-draw fragment wave config"),
                "truncated": (original[:-1], "invalid realized-draw fragment wave config"),
                "trailing": (original + b"\x00", "capture has trailing data"),
                "relabel-only62": (original[:8] + struct.pack("<I", 62) + original[12:],
                                   "capture has trailing data"),
            }
            for name, (data, error) in bad_captures.items():
                malformed_capture = directory / ("bad-width-" + name + ".prgcap")
                malformed_capture.write_bytes(data)
                for route in ("recompile-raw", "fs-tap"):
                    path = directory / ("bad-" + route + ".spv")
                    sentinel = b"unchanged: malformed capture width tail"
                    path.write_bytes(sentinel)
                    done = run(dump_args(malformed_capture, path, route), None, route == "fs-tap")
                    check(done.returncode == 2 and
                          "gpu_replay: %s: %s" % (malformed_capture, error) in done.stderr and
                          "fragment-wave=" not in done.stderr and path.read_bytes() == sentinel,
                          "%s %s malformed capture rejects before regeneration/dump" % (route, name))
    check(len(states) == 4, "genuine legacy v62 remains an explicit matrix arm")

    for state, filename, captured_width in states:
        capture = directory / filename
        check(capture.is_file(), state + " fixture exists")
        if not capture.is_file():
            continue
        for route in ("recompile-raw", "fs-tap"):
            for requested in (None, "32", "64"):
                regeneration(capture, state, captured_width, route, requested, directory)
            # Parsing is strict regardless of whether a valid captured width was available.
            for malformed in ("32junk", "64junk", "+32", "-64", " 32", "64 ", "", "128",
                              "18446744073709551616"):
                path = directory / (route + "-invalid.spv")
                sentinel = b"unchanged: malformed replay fragment width"
                path.write_bytes(sentinel)
                done = run(dump_args(capture, path, route), malformed, route == "fs-tap")
                diagnostic = "[%s] invalid PROSPER_GPU_REPLAY_FRAGMENT_WAVE_SIZE:" % route
                check(done.returncode == 2 and diagnostic in done.stderr and
                      path.read_bytes() == sentinel,
                      "%s %s invalid %r fails closed and preserves dump" % (route, state, malformed))

        stored_bytes = None
        for requested in (None, "32", "64", "32junk"):
            stored = directory / (state + "-stored.spv")
            done = run(dump_args(capture, stored), requested)
            check(done.returncode == 0 and stored.is_file() and "fragment-wave=" not in done.stderr and
                  "invalid PROSPER_GPU_REPLAY_FRAGMENT_WAVE_SIZE:" not in done.stderr,
                  "%s plain stored replay leaves override %r dormant" % (state, requested))
            if not stored.is_file():
                continue
            data = stored.read_bytes()
            if stored_bytes is None:
                stored_bytes = data
            check(data == stored_bytes, state + " dormant override preserves full stored shader bytes")
            try:
                module = Module(data)
                stored_width = 64 if captured_width == 32 or state == "legacy62-from32" else 32
                check(module.output(33) == (0, 0x3f800000, 0, 0x3f800000) and
                      not module.live_width_mask(32) and not module.live_width_mask(64) and
                      module.markers.count("Prosper.FragmentSubgroupSize=%d" % stored_width) == 1,
                      state + " stored width marker is not authority for the distinct MRT0 output")
            except (ValueError, KeyError, IndexError, struct.error) as exc:
                check(False, state + " stored output oracle: " + str(exc))

    mixed = directory / "fragment-width-mixed.prgcap"
    check(mixed.is_file(), "mixed captured32/captured64 submit fixture exists")
    if mixed.is_file():
        for route in ("recompile-raw", "fs-tap"):
            for requested in (None, "32", "64"):
                for draw, width, item in ((7, 32, 0), (11, 64, 1)):
                    regeneration(mixed, "mixed", width, route, requested, directory,
                                 draw=draw, item=item, operation=item, draw_count=2)

    no_raw = directory / "fragment-width-no-raw.prgcap"
    check(no_raw.is_file(), "missing raw-source fallback fixture exists")
    if no_raw.is_file():
        baseline = directory / "no-raw-stored.spv"
        done = run(dump_args(no_raw, baseline))
        check(done.returncode == 0 and baseline.is_file(), "missing-raw stored baseline is readable")
        if baseline.is_file():
            for route in ("recompile-raw", "fs-tap"):
                for requested in (None, "32", "64"):
                    path = directory / (route + "-no-raw.spv")
                    done = run(dump_args(no_raw, path, route), requested, route == "fs-tap")
                    reason = ("substituted vs=0 fs=0 kept-stored vs=1 fs=1 of 1 draws" if
                              route == "recompile-raw" else "draw 7: no captured raw FS stream")
                    check(done.returncode == 0 and reason in done.stderr and path.is_file() and
                          path.read_bytes() == baseline.read_bytes(),
                          "%s missing raw with request %r keeps full stored shader unchanged" %
                          (route, requested))

    for width in (32, 64):
        failed = directory / ("fragment-width-failed%d.prgcap" % width)
        check(failed.is_file(), "failed%d fixture exists" % width)
        if not failed.is_file():
            continue
        baseline = None
        for requested in (None, "64" if width == 32 else "32", "32junk"):
            path = directory / ("retry-failed%d.spv" % width)
            done = run(["--retry-failed-stage", "0:0", "--retry-failed-stage-spv", str(path),
                        str(failed)], requested)
            check(done.returncode == 0 and path.is_file() and
                  "fragment-abi=captured pixel-inputs=0 system-inputs=0 wave=%d" % width in done.stderr and
                  "fragment-wave=" not in done.stderr and
                  "invalid PROSPER_GPU_REPLAY_FRAGMENT_WAVE_SIZE:" not in done.stderr,
                  "failed%d retry leaves unrelated override %r dormant" % (width, requested))
            if path.is_file():
                data = path.read_bytes()
                if baseline is None:
                    baseline = data
                check(data == baseline, "failed%d override preserves exact retry module bytes" % width)
                inspect_module(path, "failed%d request %r" % (width, requested), width)

print("== %s (%d failures) ==" % ("FAIL" if failures else "PASS", len(failures)))
sys.exit(1 if failures else 0)
