# libc guest APIs

Registered libc services and their host implementations. The variadic producer is only a trivial
guest-ABI capture/delegate/return boundary; keep formatting, diagnostic capture, exception checkpoints
and all nontrivial lifetime state in the ordinary host compiler. The opt-in native-Clang profile uses
the same-sysroot GCC producer because Clang cannot capture Windows SysV variadics.
