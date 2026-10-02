// hle_kernel_system.cpp — small libkernel/libSceCoredump entry points that report or record system
// state: sceKernelGetOperationMode and the coredump-handler registration pair.
//
// These used to fall through to the "unimplemented, returning 0" logger. Returning 0 is the right
// RESULT for all three, but a bare 0 is wrong in one way that matters: sceKernelGetOperationMode
// reports through two out-parameters, and an unimplemented stub leaves them untouched. A guest that
// does not pre-zero them (Assassin's Creed Black Flag Resynced does; others need not) would branch on
// uninitialised stack memory. Writing them explicitly is the whole point of this file.
//
// Nothing here is title-specific.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <atomic>
#include <cstdint>

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)

namespace prosper {
namespace {

// sceKernelGetOperationMode(int* mode, int* submode) -> 0
//
// Both out-parameters are 32-bit: a guest caller keeps them as adjacent 4-byte locals, zeroes both
// with `mov DWORD PTR [..],eax`, and reads each back with a 32-bit compare (mode == 3, submode == 1).
// A null pointer is skipped rather than refused; the real contract for null is not known.
//
// WHAT VALUES. We report mode 0 and submode 0. That is what that guest saw during its live boot
// (the unimplemented stub left its pre-zeroed locals alone) and the title proceeded normally, so
// writing 0 preserves demonstrated behaviour while removing the dependence on the caller's
// initialisation. One secondary implementation reports mode 2 instead; a single secondary source is
// a hypothesis only, and the guest above tests for 3 and 1, so it cannot tell the two apart.
// CONFIDENCE: HIGH that both are 32-bit and that the call returns 0; LOW on the numeric values -
// confirm against a PS5 oracle before any title is allowed to branch on them.
HLE(k_get_operation_mode) {
    (void)a2; (void)a3; (void)a4; (void)a5;
    if (a0) *reinterpret_cast<int32_t*>(static_cast<uintptr_t>(a0)) = 0;
    if (a1) *reinterpret_cast<int32_t*>(static_cast<uintptr_t>(a1)) = 0;
    return 0;
}

// sceCoredumpRegisterCoredumpHandler(handler, stack_size, context) / ...Unregister...()
//
// The handler is RECORDED and never invoked: prosper does not turn host crashes into guest core
// dumps, and a guest that registers one only needs the call to succeed so its crash reporting
// initialises. Two independently written secondary implementations both accept a valid
// registration and return 0; they differ on validation (one rejects a null handler or a duplicate
// with libSceCoredump-specific error codes, the other accepts everything), and one source for those
// error values is not enough to adopt them, so every call succeeds. CONFIDENCE: HIGH on success for
// a well-formed call; LOW on the error cases.
std::atomic<uint64_t> g_coredump_handler{0};
std::atomic<uint64_t> g_coredump_context{0};

HLE(k_coredump_register) {
    (void)a1; (void)a3; (void)a4; (void)a5;   // a1: the handler thread's stack size; there is no thread
    g_coredump_handler.store(a0, std::memory_order_relaxed);
    g_coredump_context.store(a2, std::memory_order_relaxed);
    return 0;
}

HLE(k_coredump_unregister) {
    (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    g_coredump_handler.store(0, std::memory_order_relaxed);
    g_coredump_context.store(0, std::memory_order_relaxed);
    return 0;
}

}   // namespace

void register_kernel_system_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceKernelGetOperationMode", k_get_operation_mode);
    R("sceCoredumpRegisterCoredumpHandler", k_coredump_register);
    R("sceCoredumpUnregisterCoredumpHandler", k_coredump_unregister);
    #undef R
}

}   // namespace prosper
