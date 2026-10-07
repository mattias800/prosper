// guest_callback.cpp — the per-host half of guest_callback.hpp.
#include "host/abi/guest_callback.hpp"

#include "host/image/boot_program.hpp"  // guest_module_name

#include <atomic>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
extern "C" uint64_t prosper_call_guest_sysv4(uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2,
                                             uint64_t a3);
#endif

namespace prosper {

#ifndef _WIN32
namespace {
inline uint64_t read_fsbase() {
    uint64_t v;
    __asm__ volatile("rdfsbase %0" : "=r"(v));
    return v;
}
inline void write_fsbase(uint64_t v) {
    __asm__ volatile("wrfsbase %0" : : "r"(v));
}
// Run the enclosed callback on the guest %fs (a no-op when guest_fs == 0).
struct GuestFsScope {
    uint64_t saved = 0, active = 0;
    explicit GuestFsScope(uint64_t guest_fs) {
        if (guest_fs) {
            saved = read_fsbase();
            write_fsbase(guest_fs);
            active = guest_fs;
        }
    }
    ~GuestFsScope() {
        if (active) write_fsbase(saved);
    }
};
}  // namespace

void call_guest_callback(uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t guest_fs) {
    GuestFsScope scope(guest_fs);
    reinterpret_cast<void (*)(uint64_t, uint64_t, uint64_t)>(static_cast<uintptr_t>(fn))(a0, a1,
                                                                                         a2);
}
#else
void call_guest_callback(uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2,
                         uint64_t /*guest_fs*/) {
    // Only GUEST code needs the SysV trampoline; a host test double is already host-ABI. Classify
    // by module aperture the way the rest of the tree names guest addresses (see input/ime.cpp,
    // #2136). CONFIDENCE: HIGH for an eboot or labelled-PRX callback; an unclassified guest
    // address would be entered with MS-x64 arguments, so say so instead of failing quietly.
    if (std::strcmp(guest_module_name(fn), "mapped/host") != 0) {
        prosper_call_guest_sysv4(fn, a0, a1, a2, 0);
        return;
    }
    static std::atomic<int> unclassified{0};
    if (unclassified.fetch_add(1) < 4)
        std::fprintf(stderr,
                     "[guest-callback] target 0x%llx is outside every guest module aperture; "
                     "calling it directly (a guest callback there would misread its arguments)\n",
                     static_cast<unsigned long long>(fn));
    reinterpret_cast<void (*)(uint64_t, uint64_t, uint64_t)>(static_cast<uintptr_t>(fn))(a0, a1,
                                                                                         a2);
}
#endif

}  // namespace prosper
