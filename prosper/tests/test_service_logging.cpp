#include "hle/dispatch.hpp"
#include "hle/nid.hpp"

#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

int main() {
#ifdef _WIN32
    _putenv_s("PROSPER_SVCLOG", "1");
#else
    setenv("PROSPER_SVCLOG", "1", 1);
#endif

    prosper::register_builtin_hle();
    auto open = prosper::Hle::lookup("eaFXjfJv3xs"); // sceImeKeyboardOpen
    if (!open) {
        std::fprintf(stderr, "sceImeKeyboardOpen was not registered\n");
        return 1;
    }

#ifdef _WIN32
    void* reserved = VirtualAlloc(nullptr, 0x1000, MEM_RESERVE, PAGE_NOACCESS);
    if (!reserved) {
        std::fprintf(stderr, "VirtualAlloc reserve failed: %lu\n", GetLastError());
        return 1;
    }
#else
    void* reserved = reinterpret_cast<void*>(0xae6d000);
#endif

    // The remaining literals came from a Blasphemous 2 call. The reserved
    // pointer is definitely not readable, so diagnostic logging must skip it.
    uint64_t rc = open(1, 0xdcdfb7a0, reinterpret_cast<uintptr_t>(reserved), 1, 0x83, 0);
#ifdef _WIN32
    VirtualFree(reserved, 0, MEM_RELEASE);
#endif
    if (rc != 0) {
        std::fprintf(stderr, "sceImeKeyboardOpen returned %#llx\n",
                     static_cast<unsigned long long>(rc));
        return 1;
    }

    return 0;
}
