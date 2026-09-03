# `src/host/abi` — the guest/host calling-convention boundary

The PS5 guest is always **System V AMD64**. The host is not always the same thing: on Linux and
macOS it is SysV too, so a guest call lands on an HLE handler with nothing in between, but on
Windows it is **Microsoft x64**, and every argument has to be moved before the handler can read it.
This folder owns that translation, and only that translation.

Three files. Two are the fixed-signature half of the problem:

- `call_signature.hpp` — *what a handler's arguments are*. A `CallSignature` records which argument
  positions are floating-point and whether the return is, deduced from the handler's own C++
  declaration by `signature_of`. It lives here rather than in `hle/dispatch` because it describes
  the ABI boundary rather than the registry, but the registry stores one per NID on **every**
  platform: a Linux build registers the same signatures a Windows build does, which is what lets the
  mapping be tested without a Windows host.
- `sysv_ms_bridge.hpp/.cpp` — *where those arguments go*. The two placement tables, and the machine
  code that moves a guest SysV call frame into a Microsoft x64 one. Compiled everywhere, installed
  only by `host/image/exec_image_win.cpp`.

The third is the shape a signature cannot reach:

- `guest_varargs.hpp/.cpp` — *a variadic call*. A real C variadic's argument list is whatever the
  format string says at run time, so no compile-time `CallSignature` describes it (#3246). This
  reads the guest's System V `va_list` by the System V rules, learns each argument's class from the
  format string, and writes the flat array of 8-byte slots that IS a Microsoft `va_list`. The
  handler is then tagged `PROSPER_GUEST_ABI` (`hle/dispatch/dispatch.hpp`) and reached by a bare
  tail-jump, so the compiler's own variadic prologue captures the frame — including the overflow
  area, which nothing else here can see. `emit_guest_abi_tailjump` is those stub bytes.

  Two constraints govern any work in this file, and both are cheap to violate:
  - **A `PROSPER_GUEST_ABI` frame may own no object with a destructor**, and must call nothing that
    could be inlined into it carrying one. MinGW cannot emit SEH unwind data for a `sysv_abi`
    function, so a cleanup landing pad makes the file fail to assemble — and whether one appears is
    an *inlining* decision, so the same source can build at `-O0` and `-O2` and fail at `-O1`.
    Capture, delegate to a `noinline`/`noexcept` host function, return.
  - **A System V `va_list` dies with the frame it came from.** It points into the caller's outgoing
    argument area and into the callee's own register save area, so everything must be read while the
    variadic function is still on the stack. Deferring the read to "just after" is the failure this
    file's test found in its own first draft.

**The boundary against its siblings.** `host/x86` is instruction encoding and decoding for the
fault handlers and the SSE4a patcher — bytes, with no opinion about arguments. `host/image` maps
guest modules and installs the stub table; it asks this folder what bytes a stub contains rather
than choosing them. `hle/dispatch` owns the NID registry and carries a signature per entry, but
knows nothing about registers.

**The thing that is easy to get wrong here**, and the reason the folder exists at all: the two
conventions do not merely name different registers, they *count* differently. SysV runs separate
counters for integer and SSE arguments; Microsoft numbers every argument once and picks the register
file from its type. So a single float in the middle of a signature displaces every argument after it
as well, and a translation that handles "float arguments" by adding xmm moves to a positional
integer shuffle is still wrong (#2955). Any change here should be checked against
`tests/host/abi/test_sysv_ms_bridge.cpp`, which asserts the placement tables directly and then
*executes* the emitted trampoline across a matrix of signatures on any x86-64 host, and against
`tests/host/abi/test_guest_varargs.cpp`, which does the same for the variadic path — both
conventions' `va_arg` are reachable from any x86-64 host through `__builtin_sysv_va_list` and
`__builtin_ms_va_list`, so the round trip is checked against the *compiler's* readers rather than
against this folder's belief about the layouts.

**What no test here can reach is a live guest on a Windows host.** The next best thing is committed
as `tools/probe_win_varargs.cpp`, which is not part of the build: it cross-compiles these production
sources with MinGW and runs them under wine, so the CRT, the ABI and the stub bytes are real. Its
header carries the command. Run it at several `-O` levels — the SEH constraint above is
optimization-dependent, so one level passing says little about the others.
