// guest_caller.hpp — capture the frame pointer an HLE handler was entered with.
//
// An HLE handler is called by the guest through a synthesized import stub, so "who called me?" is a
// question about the GUEST's stack, not the host's. The one register that survives that boundary
// intact is `rbp`: it is callee-saved in SysV, and neither the plain tail-jump stub nor the guest-%fs
// swap stub writes it (host/image/exec_image_linux.cpp). Capturing it as the handler's first act is
// therefore enough to reach the guest's frame chain, whether or not this handler kept a frame
// pointer of its own -- see guest_frames_from_rbp in host/image/exec_image.hpp, which explains why
// both cases resolve to the same walk.
//
// Why a macro rather than a function: a function would capture ITS OWN rbp, one frame too deep, and
// on a build that inlined it the answer would silently change. The macro expands in the handler's
// frame and cannot drift.
//
// Use it for diagnosis, not control flow. The walk it feeds is a frame-pointer chain over optimised
// guest code, which can skip a level without saying so.
#pragma once
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#define PROSPER_CAPTURE_RBP(var)                                                                   \
    uint64_t var = 0;                                                                              \
    __asm__ volatile("movq %%rbp, %0" : "=r"(var))
// The stack pointer, for the scan-based sibling (guest_frames_on_stack). Captured the same way and
// for the same reason: taken any later, the handler's own frame has already moved it.
#define PROSPER_CAPTURE_RSP(var)                                                                   \
    uint64_t var = 0;                                                                              \
    __asm__ volatile("movq %%rsp, %0" : "=r"(var))
#else
// Non-x86-64 hosts do not run guest code at all today; keep the call sites compiling and let the
// zero seed produce an empty chain, which callers must already handle.
#define PROSPER_CAPTURE_RBP(var) uint64_t var = 0
#define PROSPER_CAPTURE_RSP(var) uint64_t var = 0
#endif
