#pragma once
// Process exit status and fault wording for a guest that died, kept free of SDL/Vulkan/windows.h so
// the contract is unit-testable on every host (test_guest_end_status.cpp).
//
// Exit status of prosper-app:
//   0                   clean: the guest returned (or never ended) and any --frames target was met
//   1                   --frames N was requested and fewer frames were presented
//   kExitGuestFault (3) the guest faulted: its entry thread ended with a non-zero BootResult::kind,
//                       a guest worker thread took an unhandled fault, or a host exception escaped
// A script that launches a title can therefore tell "the guest crashed" from "the window was closed".
// The fault status wins over the --frames shortfall: a crash is the more specific fact.
#include <cstdint>
#include <cstdio>
#include <string>

namespace prosper::app {

inline constexpr int kExitGuestFault = 3;

// guest_end_kind: -1 still running, 0 entry returned, >0 fault/abort (BootResult::kind).
inline int exit_status(int guest_end_kind, bool frames_shortfall) {
    if (guest_end_kind > 0) return kExitGuestFault;
    return frames_shortfall ? 1 : 0;
}

// One banner for a fault on any thread other than the guest entry thread, or a host exception.
// `where` is describe_code_address(rip) -- module+offset when the address is in a known module.
inline std::string format_fault_banner(const char* thread_role, unsigned long thread_id,
                                       uint32_t code, uint64_t rip, const std::string& where) {
    char head[256];
    std::snprintf(head, sizeof head,
                  "[app] UNHANDLED FAULT on %s thread tid=%lu code=0x%08x rip=0x%llx ",
                  thread_role, thread_id, (unsigned)code, (unsigned long long)rip);
    return std::string(head) + "(" + where + "); exiting with status " +
           std::to_string(kExitGuestFault) + ".";
}

}  // namespace prosper::app
