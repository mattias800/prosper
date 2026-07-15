# Guest Debugger Helpers

## Guest stack fingerprints

`guest_stack_fingerprint.py` gives a compact cross-host view of every thread in a stopped prosper process. It
prints the current PC, a short native GDB backtrace, the guest frame-pointer chain, and guest return addresses
found by scanning the native stack. Unlike the older title-specific scans, a stack word is accepted only when
the preceding guest bytes encode an x86-64 `call` ending at that address.

Attach to an existing native Windows process from PowerShell:

```powershell
gdb -p $pid -batch -x prosper/tools/dbg/guest_stack_fingerprint.py |
  Tee-Object guest-fingerprint.txt
```

Attach under WSL/Linux:

```bash
gdb -p "$pid" -batch -x prosper/tools/dbg/guest_stack_fingerprint.py \
  | tee guest-fingerprint.txt
```

Linux `ptrace_scope` may reject attachment from an unrelated shell. A launch-under-GDB sample avoids that. The
timeout's `SIGINT` stops the inferior and lets GDB execute the fingerprint script before exiting:

```bash
timeout --signal=INT 25s gdb --batch \
  -ex 'handle SIGSEGV SIGBUS SIGILL nostop noprint pass' \
  -ex 'set environment PROSPER_NO_COMPUTE 1' \
  -ex run \
  -x prosper/tools/dbg/guest_stack_fingerprint.py \
  --args prosper/build-linux/boot_trace /path/to/app0
```

The signal rule is required for launch-under-GDB: Prosper handles those guest faults itself. Without it, GDB
can stop on an emulated guest instruction before the timeout and produce a fingerprint for the wrong moment.

The output is diagnostic evidence, not a symbolic backtrace. Compare repeated samples and focus on stable
chains or host-specific branches. Absolute stack addresses, thread IDs, and a PC moving within one function are
expected to differ.

Set `PROSPER_STUBDUMP=1` on the target process to print the import-stub index, offset, NID, and resolved name.
Use that table to resolve `stub+0x...` return addresses in a fingerprint. The stub stride is printed implicitly by
the offsets and is currently 96 bytes; do not hard-code it into an investigation script.

Configuration is optional:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `PROSPER_GDB_STACK_SCAN_BYTES` | `0x2000` | Bytes scanned above each thread's current stack pointer |
| `PROSPER_GDB_MAX_CANDIDATES` | `24` | Maximum validated stack candidates per thread |
| `PROSPER_GDB_MAX_FRAMES` | `16` | Maximum frame-pointer links followed per thread |
| `PROSPER_GDB_HOST_FRAMES` | `5` | Native GDB frames printed per thread; use `0` to omit |

Run the instruction-decoder smoke test outside GDB with:

```bash
python3 prosper/tools/dbg/guest_stack_fingerprint.py --self-test
```
