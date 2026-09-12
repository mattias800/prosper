// exec_image.hpp — map a built guest image into real host memory + HLE stubs.
//
// Linux (M2/M3). The relocated image is mapped executable at its guest base. The
// import stub region holds one small executable stub per import: implemented imports
// tail-jump to their C handler (args intact); unimplemented ones log and return 0 so
// the boot advances and reveals the next call. A SIGSEGV handler still catches genuine
// guest faults (null derefs from stubbed-out returns, etc.) and reports RIP.
#pragma once
#include "self/module.hpp"
#include "hle/dispatch/dispatch.hpp"   // ImportSlot
#include <string>
#include <vector>

namespace prosper {

// The address aperture the import-stub table lives in. install_stubs claims exactly this much from
// stub_base and append_stubs (#639) grows inside it, so no stub address ever moves and nothing else
// may be mapped in the window. For the production base this is [BOOT_STUB, BOOT_STUB_END), the same
// range guest_module_name labels STUB and callback_fs.hpp treats as an import-stub return address.
inline constexpr uint64_t kStubApertureBytes = 0x10000000ull;   // 256 MiB

// Map the (already relocated) image at img.base as executable. `false`+*err on failure.
// Call once per module in a linked program.
bool map_image(const LoadedImage& img, std::string* err);

// Create the executable stub region at stub_base and populate one stub per slot:
// implemented imports tail-jump to their C handler, unimplemented ones log + return 0.
bool install_stubs(const std::vector<ImportSlot>& slots, uint64_t stub_base,
                   uint64_t stub_size, std::string* err);

// Extend the stub region in place for slots appended AFTER install_stubs ran (#639: a runtime
// sceKernelLoadStartModule adds a slot for each import no already-loaded module satisfies).
// `slots` must be the SAME vector install_stubs was given, grown at the back; `first_new` is the
// old size. Only the pages the new slots need are mapped — the pages already handed out to
// relocated guest code are never remapped, so a thread executing a stub cannot have it torn away.
// Publishes the grown table to the dispatcher after the last new stub is written.
bool append_stubs(const std::vector<ImportSlot>& slots, size_t first_new, std::string* err);

// The aperture the import-DATA table lives in, sized like the stub aperture. install_import_data
// claims exactly this much from `data_base`; for the production base it is
// [BOOT_IMPORT_DATA, BOOT_IMPORT_DATA_END).
inline constexpr uint64_t kImportDataApertureBytes = 0x10000000ull;   // 256 MiB

// Create the import-DATA region at `data_base`: one `stride`-byte, zero-filled, READ|WRITE (never
// EXEC) slot per unresolved STT_OBJECT import (#3529).
//
// Reads of an unresolved variable then see an honest zero, and a write lands on a page that holds
// nothing but that variable — instead of on the executable stub table, where it silently rewrote a
// trampoline. Non-executable is the second half of that: a control transfer to a data import now
// faults where it is raised rather than executing whatever bytes are there.
//
// No initial values are supplied. prosper does not know what an unimplemented Sony variable
// contains, and zero is the one answer that is at least self-consistent — a guest that stores its
// own value reads that value back, and one that never stores reads the same zero every time.
bool install_import_data(const std::vector<ImportSlot>& slots, uint64_t data_base,
                         uint64_t stride, std::string* err);

// Extend the import-DATA region for slots appended after install_import_data ran, mirroring
// append_stubs (a runtime sceKernelLoadStartModule can import a variable too). Already-mapped pages
// are never remapped: relocated guest code holds addresses inside them and a MAP_FIXED would
// discard whatever the guest has already written there.
bool append_import_data(const std::vector<ImportSlot>& slots, size_t first_new, std::string* err);

// Address of the idx-th import-data slot (for tests/bring-up). 0 when none is installed.
uint64_t import_data_addr(uint64_t idx);

// Install the fault handler for genuine guest faults during a run.
void install_trap_handler();

#ifdef _WIN32
// Test diagnostic: RSP&15 immediately before the recovery thunk calls its compiled MS-x64 helper.
// A valid Microsoft-x64 call site is 0; -1 means the thunk has not run yet.
int recovery_thunk_call_rsp_mod16();

// Number of valid guest entry points preserved by the Windows SSE4a code cache. Successor-consuming
// detours contribute both their EXTRQ entry and the relocated successor entry.
uint64_t sse4a_fastpath_patch_count();
#endif

// Install a per-thread alternate signal stack (so the fault handler survives a guest stack overflow).
// The main thread gets one via install_trap_handler(); worker threads call this in their trampoline.
void install_sigaltstack();

// Address of the idx-th import's stub (for tests/bring-up).
uint64_t stub_addr(uint64_t idx);

// Recover the original caller from an HLE handler's entry stack. Plain POSIX import stubs
// tail-jump, so [entry_rsp] is already the guest return address. Linux guest-FS and Windows ABI
// bridge stubs call the handler and leave a generated-stub return address at [entry_rsp]; this
// unwraps the bridge-specific frame to the guest return address saved behind it.
// `entry_rsp` must be captured before the handler's compiler prologue.
uint64_t hle_guest_return_address(uint64_t entry_rsp);

// Test/bring-up: call the idx-th import's stub as the guest would; returns its result.
uint64_t invoke_stub(uint64_t idx);

// Result of running the guest.
struct BootResult {
    int          kind = 0;      // 0 = returned, 2 = faulted (SIGSEGV/BUS), 3 = SIGILL
    std::string  detail;        // fault description
    uint64_t     fault_addr = 0, fault_rip = 0;
    uint64_t     rbp = 0, rsp = 0, rax = 0, rdi = 0, rsi = 0, rdx = 0, rbx = 0; // regs at fault
    std::vector<uint64_t> backtrace;   // return addresses (rbp chain) at the fault
};
// Describe a code address for a fault report or backtrace.
//
// A guest address becomes "<module>+0x<offset>" via boot_program.hpp's fixed map. A HOST address
// becomes "prosper+0x<rva>" (or "<dll>+0x<rva>"), which is the part that matters: prosper is ASLR'd,
// so a raw host frame is a different number every run and cannot be symbolised or even compared
// across runs. With an RVA it can be fed straight to addr2line/nm.
//
// #2194 is why this exists: a guest thread faulted, and the frame that told us WHO called it was a
// bare host address. The report named the guest side precisely and the caller not at all -- which is
// the half that decides whether a fault is the title's bug or ours.
std::string describe_code_address(uint64_t address);

// Name the GUEST frames on the calling thread's stack, for a handler that must say WHO called it.
//
// `rbp_seed` is the frame-pointer value captured by PROSPER_CAPTURE_RBP at the top of the calling
// HLE handler (host/fault/guest_caller.hpp). Both possible values work and neither needs to be
// distinguished: with a frame pointer, rbp is the handler's own frame whose saved-rbp slot links to
// the guest's; without one, rbp still HOLDS the guest's, because rbp is callee-saved and no import
// stub writes it. Host frames encountered on the way are readable and increasing, so the walk
// crosses them; they are then dropped by the guest-module filter rather than being reported as
// guest code.
//
// Writes up to `max` GUEST return addresses into `out`, newest first, and returns how many. A
// return of 0 means no guest frame was recoverable -- report that as such, never as "no caller":
// this is a frame-pointer walk, and a caller compiled without a prologue is skipped silently
// (instrument traps 114 and 217). It is a lead to confirm by disassembling the named call site,
// not an authority.
int guest_frames_from_rbp(uint64_t rbp_seed, uint64_t* out, int max);

// The same question asked without frame pointers: SCAN the calling thread's stack for guest return
// addresses instead of following a chain.
//
// This exists because the chain walk above is not merely incomplete on optimised titles, it is
// often EMPTY -- FINAL FANTASY TACTICS' fatal raise recovers exactly one frame, the module entry
// point, because every guest frame between omitted its prologue. A scan has no such dependency: a
// return address pushed by `call` is on the stack whether or not the callee kept a frame pointer.
//
// The cost of a scan is false positives, so each candidate is CONFIRMED at the call site: a real
// return address has a `call` instruction ending immediately before it, and the encodings are
// checked here (`call rel32` and the register/memory-indirect forms). A stale value left in a dead
// stack slot almost never satisfies that, which is what makes the output readable rather than a wall
// of plausible-looking numbers. Order is stack order -- innermost first -- not a proven call order:
// a scan cannot tell a live frame from a dead one that still validates.
//
// `rsp_seed` is the stack pointer captured by PROSPER_CAPTURE_RSP at the top of the calling handler.
// Scanning is bounded by the guest thread's registered stack when one is known, and by a fixed
// window otherwise. Returns how many addresses were written.
int guest_frames_on_stack(uint64_t rsp_seed, uint64_t* out, int max);

// Register the stack a guest thread runs on (main thread + workers we spawn), keyed by
// its pthread id, so GC/thread code gets accurate bounds without pthread_getattr_np.
void register_thread_stack(uint64_t tid, void* base, uint64_t size);
// Remove a dead thread's entry. pthread ids are RECYCLED — a stale entry would serve the next
// thread on the same id the old thread's bounds (#138). Called on every HLE thread-exit path.
// The registry is process-lifetime (host/immortal.hpp): a guest thread reaches this from the
// trampoline tail while the process is already running its exit handlers (#2613).
void unregister_thread_stack(uint64_t tid);
// #2613 test hook: true once this translation unit's static-storage objects have been destroyed.
// A canary declared before every registry here flips it, so an exit-time probe can prove it really
// ran after static destruction instead of passing vacuously. Always false while main() runs.
bool exec_image_statics_destroyed();
// Mark the calling host thread as about to execute guest code. Primary entry paths
// (module init + run_entry) always arm PROSPER_HWBP; worker entry paths do so only when
// PROSPER_HWBP_ALLTHREADS is set. Call while host TLS is active, before guest TLS/entry.
void guest_execution_thread_enter(bool primary);
// CPU-only observer for the real entry boundary above. Tests use this instead of requiring
// perf_event permissions; production leaves it null.
using GuestExecutionThreadEnterTestHook = void (*)(bool primary, void* opaque);
void set_guest_execution_thread_enter_test_hook(GuestExecutionThreadEnterTestHook hook,
                                                void* opaque = nullptr);
// Report the calling guest thread's registered stack bounds (false if not registered).
bool guest_stack_for_current_thread(void** base, size_t* size);
bool guest_stack_for_thread(uint64_t tid, void** base, size_t* size);

// Run dependent-module init functions (C++ global ctors etc.) before entry. Each is
// called under a per-thread recovery point; a faulting init is skipped (best-effort).
// Returns the number that ran without faulting.
size_t run_guest_inits(const std::vector<uint64_t>& fns);

// Register guest-address ranges whose module_start expects a real SCE module-param
// descriptor instead of (argc=0, argp=NULL). An init fn whose address falls in one of
// these ranges is called as module_start(argc=0x10, argp=&{u32 0x10, u32 0x200, u64 0}) —
// the descriptor Sony's native PSN.prx / SaveData.prx plugins validate before registering
// their version (their user module_start dereferences argp and null-faults with (0,NULL)).
// Default: empty (every init fn is called (0,0) — the normal boot is unchanged).
void set_module_start_param_ranges(const std::vector<std::pair<uint64_t, uint64_t>>& ranges);

// Set up a SysV-style stack + argc/argv, jump to img.entry, run until it returns or
// faults. Unimplemented imports are logged along the way (see dispatch.hpp).
BootResult run_entry(const LoadedImage& img);

} // namespace prosper
