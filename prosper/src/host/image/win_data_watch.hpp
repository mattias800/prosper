// Windows data write-watchpoint on up to four absolute guest addresses, armed on every guest thread.
//
// The Linux build answers "who wrote this word" with PROSPER_HWWATCH (perf_event); Windows had no
// equivalent, which made a corrupted-heap crash unanswerable there. x86 debug registers are
// per-thread state, so a watch armed on one thread is blind to a write from another -- a ZERO from a
// one-thread watch is not evidence (instrument trap 168). This one is therefore armed on each guest
// thread as it enters guest execution (guest_execution_thread_enter), exactly as the Linux
// ALLTHREADS variant does.
//
//   PROSPER_HWWATCH_ABS=0xADDR[:LEN][,0xADDR[:LEN]...]
//                                      up to four addresses (one per debug register DR0-DR3); LEN is
//                                      1, 2, 4 or 8 (default 8) and each address must be LEN-aligned
//                                      (a debug register can only watch an aligned, naturally sized
//                                      field). One bad entry disables the whole list.
//   PROSPER_HWWATCH_MAX=N              stop logging after N hits (default 64)
//
// Every hit prints `[hwwatch] write ... slot=.. addr=.. value=.. rip=..` plus registers and a short
// frame-pointer backtrace. Debug registers are set from a helper thread that has suspended the
// target: changing them on the running thread's own context is not a documented use of
// SetThreadContext.
#pragma once
#include <cstdint>

#ifdef _WIN32
struct _EXCEPTION_POINTERS;
namespace prosper {

constexpr unsigned kWinDataWatchSlots = 4;

struct WinDataWatchSpec {
    uint64_t addr = 0;
    unsigned len = 0;
};

// DR7 bits enabling `slot` (0-3) as a WRITE watch of `len` bytes (1/2/4/8); 0 for any other length
// or slot.
uint64_t win_data_watch_dr7_slot(unsigned slot, unsigned len);

// Slot 0 only: the single-address form, kept for callers that watch one word.
uint64_t win_data_watch_dr7(unsigned len);

// Filter status to enabled local slots whose address, type and length match the owned snapshot.
// A DR6 bit alone is not ownership: disabled-slot bits may also be set by the processor.
unsigned win_data_watch_matching_slots(uint64_t status, uint64_t control,
                                       const uint64_t addrs[kWinDataWatchSlots],
                                       const WinDataWatchSpec* owned, unsigned count);

// After consuming our slot bits, retain another enabled breakpoint or a requested single step.
bool win_data_watch_needs_other_handler(uint64_t status, uint64_t control, uint64_t flags);

// True when `addr` can be watched at `len` bytes (a supported length and naturally aligned).
bool win_data_watch_addr_ok(uint64_t addr, unsigned len);

// Parse one PROSPER_HWWATCH_ABS entry ("0xADDR" or "0xADDR:LEN"). False for anything malformed or
// unwatchable, so a typo disables the watch instead of arming a different one.
bool win_data_watch_parse(const char* spec, uint64_t* addr, unsigned* len);

// Parse a comma-separated list of entries into `out` (at most kWinDataWatchSlots). Returns the
// number of entries, or 0 when the list is empty, too long, or contains any bad entry.
unsigned win_data_watch_parse_list(const char* spec, WinDataWatchSpec out[kWinDataWatchSlots]);

// Arm the calling thread. A no-op when no watch is configured, so it is cheap to call on every
// guest thread entry.
void win_data_watch_arm_current_thread();

// Test seams: configure explicitly instead of through the environment.
bool win_data_watch_configure(uint64_t addr, unsigned len);
bool win_data_watch_configure_list(const WinDataWatchSpec* specs, unsigned count);
void win_data_watch_reset_for_test();
using WinDataWatchRegistrationHook = void* (*)();
void win_data_watch_set_registration_hook_for_test(WinDataWatchRegistrationHook hook);
long win_data_watch_handle_for_test(::_EXCEPTION_POINTERS* ep);

struct WinDataWatchHit {
    uint64_t rip;
    uint64_t addr;
    uint64_t value;
    unsigned long tid;
    unsigned slot;
    bool value_available;
};
uint64_t win_data_watch_hit_count();
bool win_data_watch_last_hit(WinDataWatchHit* out);

}   // namespace prosper
#endif
