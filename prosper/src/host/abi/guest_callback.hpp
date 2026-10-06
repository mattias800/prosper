// guest_callback.hpp — an HLE handler calling a guest callback, on every host.
//
// A library that delivers an event (a network-state change, a sign-in change) calls back into
// guest code the title registered earlier. Three host-specific facts decide whether that call
// works, and none of them belongs in a library's own source:
//
//   * Linux/macOS: the guest import stub swapped this thread to the HOST %fs on the way in, so the
//     callback must run with the caller's GUEST %fs restored or its initial-exec TLS reads host
//     memory. The guest %fs is recovered from the stub frame (hle/dispatch/callback_fs.hpp), which
//     needs the handler entered through an entry-rsp shim.
//   * Windows: nothing swaps %fs at the import boundary, but the host is Microsoft x64 while the
//     guest is System V, so a raw call hands the guest its arguments in the wrong registers.
//   * A host test double standing in for the guest callback is already in the host ABI and must
//     not be entered through the guest call path.
//
// `PROSPER_HLE_ENTRY_WITH_GUEST_FS` declares the handler entry that carries the stub frame, and
// `call_guest_callback` makes the call. Neither accepts a library-specific argument.
#pragma once

#include <cstdint>

#ifndef _WIN32
#include "host/platform/posix_shim.hpp"
// `target` has the shape (a0..a5, entry_rsp) and is an `extern "C"` function; `entry` is the
// symbol registered as the HLE handler.
#define PROSPER_HLE_ENTRY_WITH_GUEST_FS(entry, target)                                             \
    PROSPER_ASM_TRAMPOLINE(entry, target)                                                          \
    extern "C" void entry();
#else
// No import-boundary %fs swap exists on this host, so there is no stub frame to read: the target
// receives entry_rsp == 0, which callback_guest_fs_from_entry_stack answers with "no guest %fs".
#define PROSPER_HLE_ENTRY_WITH_GUEST_FS(entry, target)                                             \
    extern "C" uint64_t entry(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,     \
                              uint64_t a5) {                                                       \
        return target(a0, a1, a2, a3, a4, a5, 0);                                                  \
    }
#endif

namespace prosper {

// Call `fn(a0, a1, a2)` as a guest callback returning nothing the caller needs. `guest_fs` is the
// caller's guest %fs (0 when unknown, or on a host that never swaps it). Returns once the callback
// has returned and the host's own %fs is back in place, so the caller may log and take locks again.
void call_guest_callback(uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t guest_fs);

}  // namespace prosper
