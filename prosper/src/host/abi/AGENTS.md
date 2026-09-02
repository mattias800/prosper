# `src/host/abi` — the guest/host calling-convention boundary

The PS5 guest is always **System V AMD64**. The host is not always the same thing: on Linux and
macOS it is SysV too, so a guest call lands on an HLE handler with nothing in between, but on
Windows it is **Microsoft x64**, and every argument has to be moved before the handler can read it.
This folder owns that translation, and only that translation.

Two files, one for each half of the problem:

- `call_signature.hpp` — *what a handler's arguments are*. A `CallSignature` records which argument
  positions are floating-point and whether the return is, deduced from the handler's own C++
  declaration by `signature_of`. It lives here rather than in `hle/dispatch` because it describes
  the ABI boundary rather than the registry, but the registry stores one per NID on **every**
  platform: a Linux build registers the same signatures a Windows build does, which is what lets the
  mapping be tested without a Windows host.
- `sysv_ms_bridge.hpp/.cpp` — *where those arguments go*. The two placement tables, and the machine
  code that moves a guest SysV call frame into a Microsoft x64 one. Compiled everywhere, installed
  only by `host/image/exec_image_win.cpp`.

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
*executes* the emitted trampoline across a matrix of signatures on any x86-64 host.
