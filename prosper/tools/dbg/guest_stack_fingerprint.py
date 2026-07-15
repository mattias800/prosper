"""Print cross-host guest stack fingerprints from a stopped prosper process.

Attach with:
  gdb -p PID -batch -x prosper/tools/dbg/guest_stack_fingerprint.py

The fixed module slots mirror host/boot_program.hpp. Stack words are reported as
return addresses only when the bytes immediately before them decode as an x86-64
near call. This removes most data/code-pointer false positives from raw scans.
"""

import os
import struct
import sys

try:
    import gdb
except ImportError:
    gdb = None


MODULES = (
    ("eboot", 0x400000000, 0x440000000),
    ("il2cpp", 0x440000000, 0x480000000),
    ("psncore", 0x480000000, 0x4A0000000),
    ("psncommon", 0x4A0000000, 0x4C0000000),
    ("ps5util", 0x4C0000000, 0x4E0000000),
    ("psn", 0x4E0000000, 0x4F0000000),
    ("savedata", 0x4F0000000, 0x500000000),
    ("libc", 0x500000000, 0x520000000),
    ("fmodstudio", 0x520000000, 0x540000000),
    ("fmod", 0x540000000, 0x600000000),
    ("stub", 0x600000000, 0x700000000),
)

PREFIXES = {
    0xF0, 0xF2, 0xF3, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65, 0x66, 0x67,
}


def _module_for(value):
    for name, begin, end in MODULES:
        if begin <= value < end:
            return name, begin
    return None


def _symbol(value):
    if value is None:
        return "?"
    module = _module_for(value)
    if module:
        return "%s+0x%x" % (module[0], value - module[1])
    return "0x%x" % value


def _indirect_call_length(code):
    """Return the instruction length when code begins with an FF /2 call."""
    i = 0
    while i < len(code):
        byte = code[i]
        if byte in PREFIXES or 0x40 <= byte <= 0x4F:
            i += 1
            continue
        break

    if i + 2 > len(code) or code[i] != 0xFF:
        return None
    i += 1
    modrm = code[i]
    i += 1
    if ((modrm >> 3) & 7) != 2:
        return None

    mod = modrm >> 6
    rm = modrm & 7
    if mod == 3:
        return i

    # With either 32- or 64-bit addressing, r/m=4 introduces a SIB byte.
    sib_base = None
    if rm == 4:
        if i >= len(code):
            return None
        sib_base = code[i] & 7
        i += 1

    if mod == 0:
        if rm == 5 or (rm == 4 and sib_base == 5):
            i += 4
    elif mod == 1:
        i += 1
    elif mod == 2:
        i += 4

    return i if i <= len(code) else None


def _call_kind(code_before_return):
    """Classify a call instruction ending at the byte after this buffer."""
    if len(code_before_return) >= 5 and code_before_return[-5] == 0xE8:
        return "direct"

    for start in range(max(0, len(code_before_return) - 15), len(code_before_return) - 1):
        candidate = code_before_return[start:]
        length = _indirect_call_length(candidate)
        if length == len(candidate):
            return "indirect"
    return None


def _self_test():
    cases = (
        (b"\x90\xe8\x12\x34\x56\x78", "direct"),
        (b"\xff\xd0", "indirect"),
        (b"\xff\x50\x08", "indirect"),
        (b"\x41\xff\xd4", "indirect"),
        (b"\xff\x94\x24\x34\x12\x00\x00", "indirect"),
        (b"\x67\xff\x14\x85\x00\x10\x00\x00", "indirect"),
        (b"\xff\xe0", None),
        (b"\x90\xe8\x00\x00\x00\x00\x90", None),
    )
    for code, expected in cases:
        actual = _call_kind(code)
        if actual != expected:
            raise AssertionError("%r: expected %r, got %r" % (code, expected, actual))
    print("guest_stack_fingerprint self-test passed")


def _env_int(name, default, minimum, maximum):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw, 0)
    except ValueError:
        print("warning: ignoring invalid %s=%r" % (name, raw))
        return default
    return max(minimum, min(maximum, value))


def _run_gdb():
    scan_bytes = _env_int("PROSPER_GDB_STACK_SCAN_BYTES", 0x2000, 0x100, 0x100000)
    max_candidates = _env_int("PROSPER_GDB_MAX_CANDIDATES", 24, 1, 256)
    max_frames = _env_int("PROSPER_GDB_MAX_FRAMES", 16, 1, 128)
    host_frames = _env_int("PROSPER_GDB_HOST_FRAMES", 5, 0, 32)

    gdb.execute("set pagination off")
    gdb.execute("set confirm off")
    inferior = gdb.selected_inferior()

    def read(address, size):
        try:
            return bytes(inferior.read_memory(address, size))
        except gdb.error:
            return None

    def u64(address):
        data = read(address, 8)
        return struct.unpack("<Q", data)[0] if data else None

    def call_kind(return_address):
        code = read(return_address - 15, 15)
        return _call_kind(code) if code else None

    def register(name):
        try:
            return int(gdb.parse_and_eval("$" + name)) & 0xFFFFFFFFFFFFFFFF
        except gdb.error:
            return None

    print(
        "GUEST-FINGERPRINT version=1 scan=0x%x max_candidates=%d max_frames=%d"
        % (scan_bytes, max_candidates, max_frames)
    )
    threads = sorted(inferior.threads(), key=lambda thread: (thread.ptid[1], thread.num))
    for thread in threads:
        try:
            thread.switch()
            pc = register("pc")
            sp = register("sp")
            bp = register("rbp")
            lwp = thread.ptid[1]
            print(
                "THREAD gdb=%d lwp=%d name=%s pc=%s sp=%s"
                % (thread.num, lwp, thread.name or "?", _symbol(pc), _symbol(sp))
            )

            host = []
            frame = gdb.newest_frame()
            for _ in range(host_frames):
                if frame is None:
                    break
                try:
                    host.append(frame.name() or _symbol(int(frame.pc())))
                    frame = frame.older()
                except gdb.error:
                    break
            if host:
                print("  host=" + " <- ".join(host))

            frame_chain = []
            frame_pointer = bp
            for _ in range(max_frames):
                if frame_pointer is None:
                    break
                next_frame = u64(frame_pointer)
                return_address = u64(frame_pointer + 8)
                if next_frame is None or return_address is None:
                    break
                kind = call_kind(return_address) if _module_for(return_address) else None
                if kind:
                    frame_chain.append("%s:%s" % (_symbol(return_address), kind))
                if next_frame <= frame_pointer or next_frame - frame_pointer > 0x100000:
                    break
                frame_pointer = next_frame
            print("  frame_chain=" + (" > ".join(frame_chain) if frame_chain else "-"))

            candidates = []
            memory = read(sp, scan_bytes) if sp is not None else None
            if memory:
                for offset in range(0, len(memory) - 7, 8):
                    value = struct.unpack_from("<Q", memory, offset)[0]
                    if not _module_for(value):
                        continue
                    kind = call_kind(value)
                    if not kind:
                        continue
                    candidates.append("+0x%x:%s:%s" % (offset, _symbol(value), kind))
                    if len(candidates) >= max_candidates:
                        break
            print("  stack=" + (" ".join(candidates) if candidates else "-"))
            fingerprint = [_symbol(pc)] if pc is not None else ["?"]
            fingerprint.extend(item.split(":", 1)[0] for item in frame_chain)
            if len(fingerprint) == 1:
                fingerprint.extend(item.split(":", 2)[1] for item in candidates[:8])
            print("  fingerprint=" + ">".join(fingerprint))
        except (gdb.error, RuntimeError) as error:
            print("THREAD gdb=%d error=%s" % (thread.num, error))
    print("GUEST-FINGERPRINT-DONE")


if gdb is None:
    if "--self-test" not in sys.argv:
        print("run this script through GDB, or pass --self-test", file=sys.stderr)
        sys.exit(2)
    _self_test()
else:
    _run_gdb()
