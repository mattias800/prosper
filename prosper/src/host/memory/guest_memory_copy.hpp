#pragma once
// guest_memory_copy.hpp — copy bytes between a guest address and a host buffer without faulting
// (#3734). ONE bounds policy, stated once, for every subsystem that marshals guest memory.
//
// Before this, nine helpers in six files did this job with their own names, their own `#ifdef
// _WIN32` arms and — silently — three different policies: one refused a wrapping range and returned
// a bool, one did no overflow check and returned a partial count, one returned a partial count by
// design and its sibling did not. Moving code between subsystems changed the contract without a
// diagnostic. The subsystem helpers now forward here, so the policy below is the only one.
//
// ## The policy
//
//   * NEVER FAULTS. Every copy goes through the kernel (process_vm_readv/writev on Linux, the mach
//     shim in posix_shim.hpp on Darwin, Read/WriteProcessMemory on Windows), which reports an
//     unmapped or protected page as an error instead of delivering SIGSEGV. A guest pointer can be
//     garbage, and a diagnostic must never be able to take the boot down by reading one.
//   * A NULL guest address is never a guest buffer. `addr == 0` copies nothing.
//   * A range that would WRAP past the top of the address space is not a range. The exact forms
//     refuse it; the prefix form stops at the top.
//   * Zero bytes is a trivially successful copy (exact → true, prefix → 0) and touches nothing.
//   * errno is preserved: several callers read errno for the guest after a copy.
//
// ## Two forms, and which one to use
//
//   * guest_read_exact / guest_write_exact — all or nothing. Use these to marshal a structure: a
//     caller must never consume a partially copied guest struct. False means the destination may
//     hold a partial copy and must be ignored -- for a WRITE that destination is guest memory, so a
//     failed write can have landed a prefix. "All or nothing" is about what is REPORTED, not an
//     atomicity guarantee; no host offers one for a copy that crosses into an unmapped page.
//   * guest_read_prefix — the longest readable PREFIX of the range, as a byte count. Use it for
//     best-effort consumers (a mixer draining whatever PCM a voice still has, a capture of a buffer
//     that ends in an unmapped page, a trace that must not read past what is mapped). The count is
//     exact to the byte: readability changes only at page boundaries, and after a short or failed
//     whole-range attempt the rest is walked one page at a time.
//
// ## The page-end rule for diagnostics
//
// A diagnostic that dumps "a few words at this pointer" must not read across the pointer's page end
// with an EXACT read and treat failure as "nothing there": an out-parameter can be a tiny heap
// block whose next page is unmapped. Use guest_read_prefix and print what it returns.
//
// Measured, not assumed (2026-09-22, Linux 7.2): a single-iovec process_vm_readv over
// [mapped, mapped, PROT_NONE) returns the mapped prefix (8092 of 12088 bytes from a +100 start),
// not -1, and a fully unmapped range returns -1/EFAULT. The Darwin mach shim and Windows fail a
// range wholesale, which is why the prefix form falls back to a page walk instead of trusting one
// call.
//
// Not covered here: writes that must first disarm the renderer's write-watch (APR's DMA-style
// destination writes, hle_file.cpp), which have their own contract in guest_write_watch.hpp.

#include <cstddef>
#include <cstdint>

namespace prosper::host {

bool guest_read_exact(uint64_t src, void* dst, size_t bytes);
size_t guest_read_prefix(uint64_t src, void* dst, size_t bytes);
bool guest_write_exact(uint64_t dst, const void* src, size_t bytes);

}  // namespace prosper::host
