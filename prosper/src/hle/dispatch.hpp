// dispatch.hpp — HLE call dispatch: registry of implemented Sony functions +
// a logger for unimplemented ones. Because the Linux host ABI == the guest's
// System V AMD64 ABI, an HLE handler is just a C function the guest calls directly
// (args already in rdi/rsi/rdx/rcx/r8/r9); it returns a value in rax and rets to guest.
#pragma once
#include "../self/module.hpp"
#include "nid.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace prosper {

// Guest↔host calling-convention boundary. The guest is always System V AMD64 (PS5/FreeBSD ABI).
// On Linux/macOS the host is *also* SysV, so an HLE handler is a plain C function the guest calls
// directly. On Windows the host ABI is Microsoft x64, so every function the guest calls directly —
// the HLE handlers and any host callback the guest invokes — MUST be tagged SysV, or the six
// integer args (guest: rdi/rsi/rdx/rcx/r8/r9) are read from the wrong registers (MS: rcx/rdx/r8/r9
// + stack). PROSPER_SYSV_ABI is that tag: `__attribute__((sysv_abi))` on Windows, empty elsewhere.
// See docs/PORTING.md "Windows". NOTE: we do NOT tag handlers `__attribute__((sysv_abi))` on
// Windows — that conflicts with SEH-based C++ exception unwinding in MinGW ("`.seh_handlerdata`
// used outside of `.seh_proc` block"), and 537 STL-using handlers can't all drop exceptions.
// Instead the guest↔host ABI conversion is done in the emitted import-stub trampoline
// (exec_image_win.cpp emit_impl/emit_unimpl), so every handler stays a plain host function.
// PROSPER_SYSV_ABI is therefore empty on all platforms today; it is kept as the single documented
// marker of the boundary in case a future toolchain makes the attribute viable.
#define PROSPER_SYSV_ABI

// Cached diagnostic-flag read, for HOT paths.
//
// getenv() is not free: the UCRT walks a locked environment block, strlen-ing and comparing each
// entry, on EVERY call. A diagnostic that is switched OFF therefore still costs that scan every
// time its guard is evaluated -- and some of these guards sit on per-draw and per-RESOURCE paths,
// so the cost scales with scene complexity rather than with logging.
//
// Measured on Blue Prince (native Windows), sampling the guest primary thread with every
// diagnostic disabled: 97 of 331 samples -- ~30% of that thread's wall time -- were inside
// getenv's strlen loop.
//
// Each use site gets its own function-local static, so the read happens exactly once.
//
// THIS CHANGES SEMANTICS, and that is why it is not applied everywhere: the variable is sampled at
// first use, so a caller that sets it LATER never sees it. Several tests arm diagnostics at runtime
// with _putenv_s and then assert on the behaviour -- applying this to hle_kernel.cpp, gpu_executor
// or the command processor broke win_exception_delivery, eop_write and dynfetch_fold for exactly
// that reason. Use it only where the flag is a pure boot-time diagnostic switch AND the path is
// hot, and run the suite after.
#define PROSPER_ENV_ON(name) ([]() -> bool { static const bool prosper_env_on_v = std::getenv(name) != nullptr; return prosper_env_on_v; }())

// The same, for a diagnostic that reads a VALUE rather than just presence. Same one-shot read and
// the same consequence: sampled at first use, so a caller that sets it later never sees it.
//
// Worth its own macro because the value form is the one that hides: `const char* m = getenv(...)`
// does not look like a flag check, so a boolean-guard sweep skips it -- and in live_renderer.cpp
// the single hottest getenv site was exactly that shape, on a per-RESOURCE path, calling getenv up
// to three times per resource.
#define PROSPER_ENV_VALUE(name) ([]() -> const char* { static const char* prosper_env_value_v = std::getenv(name); return prosper_env_value_v; }())

// Generic HLE handler signature (up to 6 integer/pointer args, SysV).
using HleFn = PROSPER_SYSV_ABI uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
using HleReturnHook = void (*)();

// One unresolved import across the whole linked program (deduped by NID). Its index
// is the stub slot number; the trap logger names calls via this table.
struct ImportSlot { std::string lib, nid; };

// One case where register_fn OVERWROTE an already-registered NID with a DIFFERENT handler.
// Registration is last-write-wins with no warning, so whichever register_*() runs last silently
// takes the NID — and if the loser was a real implementation and the winner a naive stub, the guest
// gets the stub (e.g. #330: a Get* that no longer wrote its out-param). Recorded at registration time
// so the winner/loser reflect the true runtime order, not a static guess.
struct ShadowedReg {
    std::string nid;          // the collided key
    std::string prev_name;    // display name of the registration that was overwritten
    std::string new_name;     // display name of the registration that won
};

// Registry of implemented functions, keyed by NID.
class Hle {
public:
    static void  register_fn(const std::string& nid, HleFn fn, const char* name,
                             HleReturnHook return_hook = nullptr);
    // Like register_fn, but marks the entry as a deliberately-overridable placeholder (a diagnostic/
    // tracing thunk that a real handler is expected to replace later). A subsequent register_fn that
    // overwrites a placeholder is NOT flagged as a shadow — that override is the intent.
    static void  register_placeholder(const std::string& nid, HleFn fn, const char* name);
    static HleFn lookup(const std::string& nid);          // nullptr if unimplemented
    static HleReturnHook return_hook_of(const std::string& nid); // called by the import return trampoline
    static const char* name_of(const std::string& nid);   // registered display name or ""
    // Every NID that was registered 2+ times with DIFFERING handlers, in registration order. Empty
    // is the healthy state; a non-empty list is the #330 double-registration-shadow class and must be
    // reviewed (the winner is the LAST registration). Populated as register_fn runs.
    static const std::vector<ShadowedReg>& shadowed_registrations();
};

// Wire the unimplemented-call logger to the global stub-slot table + name DB.
void dispatch_init(const std::vector<ImportSlot>* slots, NidDb* db);

// Re-point the logger at a stub-slot table that GREW after dispatch_init (#639: a runtime-loaded
// PRX appends slots for imports no already-linked module satisfies). Unlike dispatch_init this
// only extends the per-slot call-count vector — it must not reset the counts of slots the guest
// has already called, or the unimplemented-call census silently restarts mid-run.
void dispatch_grow_slots(const std::vector<ImportSlot>* slots);

// Append import slots to the live table under the SAME lock prosper_on_unimpl takes (#639).
// The runtime loader appends from a guest thread while other guest threads may be calling an
// unimplemented import, and an unsynchronised push_back would move the vector's buffer out from
// under `(*g_slots)[idx]`. Returns the index the first appended slot landed on.
size_t dispatch_append_slots(std::vector<ImportSlot>* slots, const std::vector<ImportSlot>& add);

// Register the built-in HLE implementations (libc thunks, CRT no-ops, libkernel
// primitives). Call before install_stubs.
void register_builtin_hle();
// libkernel primitives (pthread/sync/etc.); called by register_builtin_hle().
void register_kernel_hle();
// File I/O (stdio + POSIX fd, with /app0 path translation); called by register_builtin_hle().
void register_file_hle();
// Set the host directory backing the guest's "/app0" (the game data root).
void set_app0_root(const std::string& root);
// Resolve a guest filesystem path through the same mount table used by libkernel file I/O.
// Host frontends such as native media decoders must never receive raw /app0 paths.
std::string resolve_guest_path(const char* guest_path);
// Mount/unmount the guest "/savedata0" area onto a host dir named by the save's dirName.
// The policy/result distinction preserves SaveData's exclusive CREATE versus CREATE2
// (open-or-create) semantics, including whether MountResult should report CREATED or OPENED.
enum class SaveDataMountPolicy { Open, Create, OpenOrCreate };
enum class SaveDataMountOutcome { NotFound, Exists, Opened, Created };
// A guest save dirName must be a SINGLE directory component under the host save root — reject empty,
// "." / "..", and any embedded '/' or '\\' so a crafted/garbled name can't traverse out of the sandbox.
// Shared by savedata0_mount and savedata0_dir_mtime so the guard can't exist in only one of them.
bool savedata_dirname_ok(const std::string& dirname);
SaveDataMountOutcome savedata0_mount(const char* dirname, SaveDataMountPolicy policy);
bool savedata0_umount();
std::vector<std::string> savedata0_list_dirs();   // existing save dirs under the host save root (#299)
// Read a virtual save slot's param.sfo modification time (or directory time if absent). Rejects names
// that are not a single guest directory component; used only for SaveDataDialog date-focus ordering.
bool savedata0_dir_mtime(const std::string& dirname, int64_t& modified);
// libSceSaveData transaction resources (#1905). sceSaveDataCreateTransactionResource returns the
// new resource's ID, NOT 0-on-success: the id is what a later sceSaveDataMount3 descriptor carries
// at +0x28. Sonic Origins' save handler proves the polarity — eboot+0x93fdb0 is
// `call sceSaveDataCreateTransactionResource(0); test eax,eax; jle -> error`, so a zero return is
// read as failure. Ids are handed out from a monotonic counter and recorded, so Delete can reject
// an id that was never created. CONFIDENCE: HIGH on "positive id, negative error" (guest
// disassembly); the id's numeric range is ours to choose, as it is opaque to the guest.
int32_t savedata_tx_resource_create();
bool savedata_tx_resource_destroy(int32_t id);
size_t savedata_tx_resource_live_count();
// PS5 system services (user/NP/mouse/appcontent/dialog); called by register_builtin_hle().
void register_service_hle();
// libSceHttp local URI helpers; called by register_builtin_hle().
void register_http_hle();
// libSceFont/libSceFontFt opaque lifecycle and deterministic metrics; called by register_builtin_hle().
void register_font_hle();
// libSceFiber cooperative guest-stack switching; called by register_builtin_hle().
void register_fiber_hle();
// Optional host-side hooks around a spawned guest thread's guest-code region. Both run on the HOST
// %fs — on_enter as the last host action before the guest entry, on_exit as the first after it
// returns — so either may call host libc. Null hooks are the default and change nothing.
struct GuestThreadHooks {
    void (*on_enter)(void* opaque) = nullptr;
    void (*on_exit)(void* opaque, uint64_t retval) = nullptr;
    void* opaque = nullptr;
};
// Spawn a guest-code thread through the same trampoline scePthreadCreate uses, entering the guest on
// an explicit caller-supplied stack. Returns 0 on success, otherwise an errno-style code. Used by
// libSceUlt to run an ulthread on its guest-supplied context buffer (#1603).
int guest_thread_spawn(uint64_t entry, uint64_t arg, const char* name,
                       void* guest_stack, size_t guest_stack_size,
                       const GuestThreadHooks* hooks, uint64_t* out_thread);

// libSceUlt (user-level threading): the ulthread runtime, waiting-queue resource pool, mutexes and
// condition variables, with the ABI derived from the importing title's own call sites (#1603).
// Entry points not yet implemented stay registered, counted and fail-visible; see hle_ult.cpp.
// Called by register_builtin_hle().
void register_ult_hle();
// Calls made to the unimplemented libSceUlt surface, by NID (0 for an unregistered/uncalled NID).
// Counted PER CALL, unlike the first-seen dedup in prosper_on_unimpl. Diagnostics + tests.
uint64_t ult_call_count(const char* nid);
// Append the libSceUlt call table to `f` (nothing if the title never touched Ult). Called from
// dump_call_log so the existing PROSPER_PROGRESS_UNIMPL periodic dump reports real Ult volume.
void ult_dump_call_log(FILE* f);
// Test seams for the return policies (PROSPER_ULT_RETURN_SUCCESS / PROSPER_ULT_LEGACY_ENOSYS) and
// the counters. Each setter returns the previous policy. Production code must not call these.
bool ult_set_return_success_for_test(bool return_success);
bool ult_set_legacy_enosys_for_test(bool legacy_enosys);
// Highest number of ulthreads of this runtime observed RUNNING at once (the #1603 amendment-2b
// deviation counter), and the last join's measured guest-stack high-water (amendment 6).
uint64_t ult_runtime_high_water_for_test(uint64_t guest_runtime_addr);
uint64_t ult_last_join_stack_used_for_test(uint64_t* context_size);
void ult_reset_counts_for_test();
// libScePad game-controller input (real host controller via input/pad.cpp); called by register_builtin_hle().
void register_pad_hle();
// Headless graphics bring-up (libSceAgc/libSceVideoOut placeholders); see hle_graphics.cpp.
void register_graphics_hle();
// libSceAudioOut HLE backed by a pluggable, headless AudioSink; see hle_audio.cpp / audio.hpp.
void register_audio_hle();
// libSceAgc "Gen5" Draw Command Buffer HLE (real PM4-building Dcb functions); see hle_agc.cpp.
// Call AFTER register_graphics_hle so these override the observe-only glog stubs.
void register_agc_hle();
// libkernel virtual/direct memory (Linux backing); called by register_kernel_hle().
void register_kernel_mem_hle();
// libkernel time/clock + C11 threads + assorted stubs; called by register_kernel_hle().
void register_kernel_time_hle();

#ifdef _WIN32
// Windows pthread-key lifecycle probes used by the focused concurrency regression.
void win_set_key_delete_after_host_hook_for_test(void (*hook)(uint64_t));
size_t win_key_destructor_thunk_count_for_test();
// Observe the ERRORCHECK owner map without attempting a potentially blocking same-thread relock.
bool win_guest_mutex_owned_by_current_thread_for_test(void* mutex);
#endif

// Preserve the unsigned native-thread identifier used by sync/exception diagnostics. Exposed so
// the high-bit formatting contract can be covered without relying on the OS to allocate such a TID.
uint64_t sync_trace_tid_value(uint64_t native_tid);

// The default target for unimplemented imports: logs (first-seen) and returns 0.
// Called by generated stubs with the import index in the first arg. Windows stubs also preserve the
// guest return address and pre-call stack pointer for low-volume caller attribution.
extern "C" uint64_t prosper_on_unimpl(uint64_t import_index, uint64_t guest_return = 0,
                                       uint64_t guest_rsp = 0);

// First-seen order of unimplemented import indices called by the guest.
const std::vector<uint32_t>& call_order();
// Optional external progress counter (e.g. shared memory), incremented once per
// first-seen unimplemented call. Survives across fork() so a crash-prone deep boot can
// still be measured by a parent process.
void dispatch_set_progress(volatile int* counter);
// Optional external counter incremented on each sceKernelRaiseException delivery (the GC's
// stop-the-world). Also fork-safe. Used by tests to prove the boot reached — and got through —
// the IL2CPP GC thread-suspension handshake (a regression guard for the deadlock/GC fixes).
void set_exc_raise_counter(volatile int* counter);
// Optional fork-safe counter bumped when the guest calls into the graphics libs (libSceAgc /
// libSceVideoOut). Tests use it to prove the boot advanced through the whole runtime into GPU/
// display init. Defined in hle_graphics.cpp.
void set_gfx_call_counter(volatile int* counter);
// Per-module TLS template for the general-dynamic model (see linker.cpp / __tls_get_addr).
// init_va = mapped tdata (guest==host addr); a per-thread block of memsz is allocated lazily,
// filesz bytes copied from init_va, the rest zeroed. `align` is the module's PT_TLS p_align — the
// static-TLS layout must round each module's offset to it so an x86-64 Variant II initial-exec
// %fs:-N access lands where the guest static linker compiled it (#143). Indexed by module TLS id.
struct TlsModuleDesc { uint64_t init_va = 0, filesz = 0, memsz = 0, align = 0; };
// Install the TLS templates (call AFTER images are mapped, so init_va is readable). Enables the
// __tls_get_addr HLE to serve real per-thread TLS blocks for loaded modules (e.g. real libc.prx).
void set_tls_modules(const TlsModuleDesc* descs, size_t count);
// Free + forget the CURRENT thread's __tls_get_addr DTV blocks. Called on every HLE-controlled
// thread-exit path (thread_trampoline's normal return, scePthreadExit/pthread_exit): glibc recycles
// pthread ids, so a stale DTV entry would hand a new thread the dead thread's dirty TLS blocks
// instead of the fresh zero/tdata-initialized state the ABI guarantees — and leak them (#68).
// Safe (no-op) on threads that never called __tls_get_addr, incl. the main thread.
void tls_dtv_purge_current_thread();
// Number of threads currently holding live DTV entries (test/diagnostic introspection: lets the
// regression test assert that thread exit really purged — i.e. no per-thread-churn leak).
size_t tls_dtv_thread_count();

// Guest initial-exec TLS. Enabled by default on Linux/Windows (PROSPER_NO_GUEST_FS opts out); macOS
// trap emulation is opt-in with PROSPER_GUEST_FS. Backs guest %fs-relative static thread-locals with a
// guest TCB + Variant-II static TLS; Linux import stubs swap %fs back for HLE calls. See guest_tls.cpp.
void guest_tls_set_templates(const TlsModuleDesc* descs, size_t count);
bool guest_tls_enabled();
uint64_t guest_tls_activate_thread();   // per guest thread at entry; returns guest TP (0 if disabled)
// macOS trap mode only: the calling thread's guest thread-pointer, or 0. The SIGSEGV handler uses it
// to relocate a faulting guest `%fs:`-relative access to guest_TP + offset (Rosetta can't set fs base).
uint64_t guest_tls_tp();
void guest_fs_enter_host_for_signal();  // crash-signal-handler entry: swap guest %fs -> host %fs (no-op if not guest TCB)
uint64_t guest_fs_to_host_scoped();     // diagnostic handler (returns to guest): swap to host %fs, return prev fs
void guest_fs_restore_scoped(uint64_t prev_fs);  // restore the fs returned by guest_fs_to_host_scoped
// This thread's guest TP (0 if guest-fs not active on it). On Windows the fault handler queries this
// to detect a drifted FS base (Windows zeroes the user FS base on every kernel transition).
uint64_t guest_fs_current_tp();
// If guest-fs is active on this thread and the live FS base has drifted from the guest TP, re-apply
// wrfsbase(guest TP) and return true; else false. The Windows VEH calls this to transparently retry a
// faulting guest %fs access after the OS reset the base. Returns false when the base is already
// correct (so a genuine fault at an fs-relative insn is NOT retried forever). No-op → false on Linux.
bool guest_fs_reapply();
// Diagnostic (test): the Variant II static-TLS distance below the thread pointer for module id
// `modid`, and the total below TP — verifies the per-module PT_TLS p_align layout (#143). Linux-only.
uint64_t guest_tls_module_below(uint32_t modid);
uint64_t guest_tls_total_below();

// Per-module info for C++ exception unwinding (sceKernelGetModuleInfoForUnwind). The guest's libunwind
// asks, for a code address, where that module's .eh_frame_hdr / text segment live. `lo/hi` is the module's
// absolute VA span (for the addr→module lookup); ehframe_hdr = absolute VA of its PT_GNU_EH_FRAME segment.
struct UnwindModuleDesc {
    uint64_t lo = 0, hi = 0;             // absolute VA range covering the module
    uint64_t ehframe_hdr = 0, ehframe_hdr_sz = 0;   // .eh_frame_hdr (PT_GNU_EH_FRAME), absolute VA
    uint64_t seg0 = 0, seg0_sz = 0;      // first PT_LOAD, absolute VA
    const char* name = nullptr;          // stable module name (basename)
};
// Install module unwind descriptors (call after images are mapped). Enables sceKernelGetModuleInfoForUnwind.
void set_unwind_modules(const UnwindModuleDesc* descs, size_t count);
// Append one module's unwind descriptor after boot (#639), returning the index it landed on.
// `d.name` must outlive the run. The caller must append the same module's export table
// (add_module_export_table) so the two orders stay parallel — sceKernelGetModuleInfoFromAddr
// derives a handle from the unwind order and sceKernelDlsym consumes it against the export order.
size_t add_unwind_module(const UnwindModuleDesc& d);

// Windows-only fatal-fault diagnostics. Other hosts provide empty implementations.
void trace_guest_stack_query(uint64_t target, bool found, void* base, size_t size);
void trace_guest_thread_lifecycle(bool starting, uint64_t pthread_id, uint64_t native_id,
                                  void* stack_base, size_t stack_size);
struct GuestThreadSnapshot {
    uint32_t native_id = 0;
    uint64_t pthread_id = 0;
    uintptr_t stack_base = 0;
    size_t stack_size = 0;
};
// Lock-free view of one lifecycle registration. The payload is returned only when both publication
// reads observe the same generation token. Other hosts return false.
bool snapshot_guest_thread_registration(uint32_t native_id, GuestThreadSnapshot& snapshot);
// True only for a committed executable Windows page. Other hosts return false.
bool guest_trace_page_executable(uintptr_t address);
// Deterministic test seam at the sampler's handle-pinned, pre-suspend boundary. Production leaves it null.
using GuestThreadTraceTestHook = void (*)(uint32_t native_id, void* opaque);
void set_guest_thread_trace_test_hook(GuestThreadTraceTestHook hook, void* opaque = nullptr);
// Windows cooperative exception checkpoint used when a target was woken from a registered HLE wait.
// Other hosts provide an empty implementation.
void dispatch_pending_guest_exception();
// Windows: publish the CALLING thread's handle for sceKernelRaiseException. Guest worker threads do
// this in win_thread_trampoline; the primary guest thread never runs through it, so it must call
// this itself or the IL2CPP GC cannot stop it (#2139). No-op elsewhere.
void register_guest_execution_thread_handle();
// Number of cooperatively queued Windows exceptions awaiting a target checkpoint (zero elsewhere).
// Exposed for diagnostics and lifetime/withdrawal regression tests.
uint32_t pending_guest_exception_count();
void dump_guest_exception_trace();
// Windows-only snapshot of live guest-thread instruction/stack pointers. Other hosts are a no-op.
void dump_guest_thread_trace(const char* path = nullptr, uint64_t pthread_filter = 0);

// Register the linked program's global export table (NID -> absolute guest address) so
// sceKernelDlsym can resolve exported symbols by name. Unity's native-plugin loader dlsym's
// UnityPluginLoad / PSN_PrxInitialize / etc. by name; k_dlsym hashes the name (nid_hash) and
// returns the matching export address (e.g. into the loaded PSN.prx). Pointer must outlive the run.
void set_module_exports(const std::unordered_map<std::string, uint64_t>* exports);

// Per-module export registration (#147): one entry per linked module — its load path and its OWN
// NID -> address table (Program::mod_exports). sceKernelLoadStartModule hands out a REAL handle
// when the requested path names one of these (basename match), and sceKernelDlsym consults the
// handle's module before the global first-definition-wins table. Table pointers must outlive the run.
struct ModuleExportTable { std::string path; const std::unordered_map<std::string, uint64_t>* nids; };
// Base of the handles sceKernelLoadStartModule and sceKernelGetModuleInfoFromAddr hand out.
// SceKernelModule is a 32-bit int on PS5 (R-Type Delta stores it in an int array and tests it
// against zero), so the handle space must stay small and positive.
inline constexpr uint64_t kSceModuleHandleBase = 0x10000;
void set_module_export_tables(std::vector<ModuleExportTable> tables);
// Append ONE module's export table after boot (#639: real runtime PRX loading). Additive
// counterpart of set_module_export_tables — the pre-linked set keeps its handles and its
// first-definition-wins precedence, and the new module is appended after it. Safe to call while
// the guest runs; `nids` must outlive the run and must not be mutated afterwards. Returns the
// handle sceKernelLoadStartModule reports for the module (the same value module_handle_for_path
// answers for its path from then on).
uint64_t add_module_export_table(std::string path,
                                 const std::unordered_map<std::string, uint64_t>* nids);
// Handle for a guest load path naming a registered module (basename match), or 0 if unknown —
// the caller (k_load_start_mod) then falls back to its synthetic success handle.
uint64_t module_handle_for_path(const char* path);

// Guest address of the main module's SCE_PROCPARAM segment. sceKernelGetProcParam returns this;
// real libc reads its heap/malloc config (sceLibcParam) from it, so a correct value is required for
// real libc.prx's heap to initialize. Set from the eboot's PT_SCE_PROCPARAM segment after mapping.
void set_proc_param(uint64_t guest_va);

// Print the accumulated unimplemented-call trace (index, lib::nid [name], count).
void dump_call_log(FILE* f);
void reset_call_log();

// Optional hook to arm a hardware data write-watchpoint on an address (diagnostic). Set by the
// Linux exec harness (exec_image_linux.cpp, perf_event-based); null elsewhere. HLE code may call
// through it (when non-null) to catch the guest writer of a specific slot — e.g. the AGC
// register-context sub-object fields. One watch at a time; extra calls are ignored.
extern void (*g_hwwatch_hook)(uint64_t addr);

// #312 diagnostic hook: arm a per-thread hardware write-watch on a MallocBinned3 per-thread
// pool-descriptor cache base (base+0x20 = size-class idx=1 head — the corruptor's target) the
// instant the guest is handed that base via pthread_get/setspecific. Set by the Linux exec harness
// under PROSPER_MB3WATCH; null elsewhere. Arms on the CALLING (owning) guest thread so the watch is
// on the right per-thread debug registers, before any corrupting store. HLE calls through it (when
// non-null) from k_getspecific/k_setspecific.
extern void (*g_mb3_arm_hook)(uint64_t base);

} // namespace prosper
