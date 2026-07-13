// dispatch.hpp — HLE call dispatch: registry of implemented Sony functions +
// a logger for unimplemented ones. Because the Linux host ABI == the guest's
// System V AMD64 ABI, an HLE handler is just a C function the guest calls directly
// (args already in rdi/rsi/rdx/rcx/r8/r9); it returns a value in rax and rets to guest.
#pragma once
#include "../self/module.hpp"
#include "nid.hpp"
#include <cstdint>
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

// Generic HLE handler signature (up to 6 integer/pointer args, SysV).
using HleFn = PROSPER_SYSV_ABI uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

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
    static void  register_fn(const std::string& nid, HleFn fn, const char* name);
    // Like register_fn, but marks the entry as a deliberately-overridable placeholder (a diagnostic/
    // tracing thunk that a real handler is expected to replace later). A subsequent register_fn that
    // overwrites a placeholder is NOT flagged as a shadow — that override is the intent.
    static void  register_placeholder(const std::string& nid, HleFn fn, const char* name);
    static HleFn lookup(const std::string& nid);          // nullptr if unimplemented
    static const char* name_of(const std::string& nid);   // registered display name or ""
    // Every NID that was registered 2+ times with DIFFERING handlers, in registration order. Empty
    // is the healthy state; a non-empty list is the #330 double-registration-shadow class and must be
    // reviewed (the winner is the LAST registration). Populated as register_fn runs.
    static const std::vector<ShadowedReg>& shadowed_registrations();
};

// Wire the unimplemented-call logger to the global stub-slot table + name DB.
void dispatch_init(const std::vector<ImportSlot>* slots, NidDb* db);

// Register the built-in HLE implementations (libc thunks, CRT no-ops, libkernel
// primitives). Call before install_stubs.
void register_builtin_hle();
// libkernel primitives (pthread/sync/etc.); called by register_builtin_hle().
void register_kernel_hle();
// File I/O (stdio + POSIX fd, with /app0 path translation); called by register_builtin_hle().
void register_file_hle();
// Set the host directory backing the guest's "/app0" (the game data root).
void set_app0_root(const std::string& root);
// Mount/unmount the guest "/savedata0" area onto a host dir named by the save's dirName
// (sceSaveDataMount3 HLE). create=true makes the dir; create=false fails if it doesn't exist.
bool savedata0_mount(const char* dirname, bool create);
void savedata0_umount();
std::vector<std::string> savedata0_list_dirs();   // existing save dirs under the host save root (#299)
// PS5 system services (user/NP/mouse/appcontent/dialog); called by register_builtin_hle().
void register_service_hle();
// libSceHttp local URI helpers; called by register_builtin_hle().
void register_http_hle();
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

// The default target for unimplemented imports: logs (first-seen) and returns 0.
// Called by generated stubs with the import index in the first arg.
extern "C" uint64_t prosper_on_unimpl(uint64_t import_index);

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

// Guest initial-exec TLS (Linux; GATED behind PROSPER_GUEST_FS, default off). Backs guest %fs-relative
// static thread-locals by giving each guest thread its own guest TCB + Variant-II static TLS and running
// guest code with %fs = guest TP; import stubs swap %fs back to the host TCB for HLE calls. See
// guest_tls.cpp. Call set_templates with the SAME descs as set_tls_modules (after images are mapped).
void guest_tls_set_templates(const TlsModuleDesc* descs, size_t count);
bool guest_tls_enabled();
uint64_t guest_tls_activate_thread();   // per guest thread at entry; returns guest TP (0 if disabled)
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
void set_module_export_tables(std::vector<ModuleExportTable> tables);
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
