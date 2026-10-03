// Windows data write-watchpoint on one absolute guest address, armed on every guest thread.
//
// The Linux build answers "who wrote this word" with PROSPER_HWWATCH (perf_event); Windows had no
// equivalent, which made a corrupted-heap crash unanswerable there. x86 debug registers are
// per-thread state, so a watch armed on one thread is blind to a write from another -- a ZERO from a
// one-thread watch is not evidence (instrument trap 168). This one is therefore armed on each guest
// thread as it enters guest execution (guest_execution_thread_enter), exactly as the Linux
// ALLTHREADS variant does.
//
//   PROSPER_HWWATCH_ABS=0xADDR[:LEN]   absolute address; LEN is 1, 2, 4 or 8 (default 8), and the
//                                      address must be LEN-aligned (the debug register can only
//                                      watch an aligned naturally-sized field)
//   PROSPER_HWWATCH_MAX=N              stop logging after N hits (default 64)
//
// Every hit prints `[hwwatch] write tid=.. addr=.. value=.. rip=..` plus registers and a short
// frame-pointer backtrace. Debug registers are set from a helper thread that has suspended the
// target: changing them on the running thread's own context is not a documented use of
// SetThreadContext.
#pragma once
#include <cstdint>

#ifdef _WIN32
namespace prosper {

// DR7 value enabling slot 0 as a WRITE watch of `len` bytes (1/2/4/8); 0 for any other length.
uint64_t win_data_watch_dr7(unsigned len);

// True when `addr` can be watched at `len` bytes (a supported length and naturally aligned).
bool win_data_watch_addr_ok(uint64_t addr, unsigned len);

// Parse a PROSPER_HWWATCH_ABS value ("0xADDR" or "0xADDR:LEN"). False for anything malformed or
// unwatchable, so a typo disables the watch instead of arming a different one.
bool win_data_watch_parse(const char* spec, uint64_t* addr, unsigned* len);

// Arm the calling thread. A no-op when no watch is configured, so it is cheap to call on every
// guest thread entry.
void win_data_watch_arm_current_thread();

// Test seams: configure explicitly instead of through the environment.
bool win_data_watch_configure(uint64_t addr, unsigned len);
void win_data_watch_reset_for_test();

struct WinDataWatchHit {
    uint64_t rip;
    uint64_t addr;
    uint64_t value;
    unsigned long tid;
};
uint64_t win_data_watch_hit_count();
bool win_data_watch_last_hit(WinDataWatchHit* out);

}   // namespace prosper
#endif
