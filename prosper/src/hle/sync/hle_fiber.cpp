// hle_fiber.cpp - cooperative libSceFiber execution for native PS5 engines.
//
// Guest code runs natively, so a fiber can retain its real guest call stack. A small private
// register context transfers between suspended HLE frames; first entry is made on the
// guest-provided stack. The CRT setjmp/longjmp cannot be used here: fortified glibc deliberately
// rejects jumps to a lower stack and the Windows CRT tries to unwind SEH frames.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/callback_fs.hpp"
#include "host/platform/posix_shim.hpp"

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <chrono>
#include <thread>
#include <new>
#include <pthread.h>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#endif

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Global-assembly symbol spelling differs between ELF and Mach-O.  Keep the fiber context
// helpers usable on macOS as well as Linux: Mach-O prefixes C symbols with '_' and rejects
// ELF's .type directive.
#ifdef __APPLE__
#define PROSPER_FIBER_ASM_HEADER(name) ".globl _" #name "\n_" #name ":\n"
#else
#define PROSPER_FIBER_ASM_HEADER(name) \
    ".globl " #name "\n.type " #name ",@function\n" #name ":\n"
#endif

namespace prosper {
namespace {

struct alignas(16) FiberContext { uint64_t words[32]{}; };

extern "C" uint64_t prosper_fiber_context_save(FiberContext*)
    __attribute__((returns_twice));
extern "C" [[noreturn]] void prosper_fiber_context_restore(FiberContext*, uint64_t);

#ifndef _WIN32
asm(
    ".text\n"
    ".p2align 4\n"
    PROSPER_FIBER_ASM_HEADER(prosper_fiber_context_save)
    "  movq %rbx,0(%rdi)\n"
    "  movq %rbp,8(%rdi)\n"
    "  movq %r12,16(%rdi)\n"
    "  movq %r13,24(%rdi)\n"
    "  movq %r14,32(%rdi)\n"
    "  movq %r15,40(%rdi)\n"
    "  leaq 8(%rsp),%rax\n"
    "  movq %rax,48(%rdi)\n"
    "  movq (%rsp),%rax\n"
    "  movq %rax,56(%rdi)\n"
    "  xorl %eax,%eax\n"
    "  ret\n"
    ".p2align 4\n"
    PROSPER_FIBER_ASM_HEADER(prosper_fiber_context_restore)
    "  movq 0(%rdi),%rbx\n"
    "  movq 8(%rdi),%rbp\n"
    "  movq 16(%rdi),%r12\n"
    "  movq 24(%rdi),%r13\n"
    "  movq 32(%rdi),%r14\n"
    "  movq 40(%rdi),%r15\n"
    "  movq 48(%rdi),%rsp\n"
    "  movq %rsi,%rax\n"
    "  testq %rax,%rax\n"
    "  jnz 1f\n"
    "  movl $1,%eax\n"
    "1: jmp *56(%rdi)\n"
);
#else
// Microsoft x64 additionally makes rdi/rsi and xmm6-xmm15 nonvolatile. Preserve them so the
// context can resume an optimized C++ HLE frame without depending on CRT unwinding internals.
asm(
    ".text\n"
    ".p2align 4\n"
    ".globl prosper_fiber_context_save\n"
    "prosper_fiber_context_save:\n"
    "  movq %rbx,0(%rcx)\n"
    "  movq %rbp,8(%rcx)\n"
    "  movq %rdi,16(%rcx)\n"
    "  movq %rsi,24(%rcx)\n"
    "  movq %r12,32(%rcx)\n"
    "  movq %r13,40(%rcx)\n"
    "  movq %r14,48(%rcx)\n"
    "  movq %r15,56(%rcx)\n"
    "  leaq 8(%rsp),%rax\n"
    "  movq %rax,64(%rcx)\n"
    "  movq (%rsp),%rax\n"
    "  movq %rax,72(%rcx)\n"
    "  movdqu %xmm6,80(%rcx)\n"
    "  movdqu %xmm7,96(%rcx)\n"
    "  movdqu %xmm8,112(%rcx)\n"
    "  movdqu %xmm9,128(%rcx)\n"
    "  movdqu %xmm10,144(%rcx)\n"
    "  movdqu %xmm11,160(%rcx)\n"
    "  movdqu %xmm12,176(%rcx)\n"
    "  movdqu %xmm13,192(%rcx)\n"
    "  movdqu %xmm14,208(%rcx)\n"
    "  movdqu %xmm15,224(%rcx)\n"
    "  xorl %eax,%eax\n"
    "  ret\n"
    ".p2align 4\n"
    ".globl prosper_fiber_context_restore\n"
    "prosper_fiber_context_restore:\n"
    "  movdqu 80(%rcx),%xmm6\n"
    "  movdqu 96(%rcx),%xmm7\n"
    "  movdqu 112(%rcx),%xmm8\n"
    "  movdqu 128(%rcx),%xmm9\n"
    "  movdqu 144(%rcx),%xmm10\n"
    "  movdqu 160(%rcx),%xmm11\n"
    "  movdqu 176(%rcx),%xmm12\n"
    "  movdqu 192(%rcx),%xmm13\n"
    "  movdqu 208(%rcx),%xmm14\n"
    "  movdqu 224(%rcx),%xmm15\n"
    "  movq 0(%rcx),%rbx\n"
    "  movq 8(%rcx),%rbp\n"
    "  movq 16(%rcx),%rdi\n"
    "  movq 24(%rcx),%rsi\n"
    "  movq 32(%rcx),%r12\n"
    "  movq 40(%rcx),%r13\n"
    "  movq 48(%rcx),%r14\n"
    "  movq 56(%rcx),%r15\n"
    "  movq 64(%rcx),%rsp\n"
    "  movq %rdx,%rax\n"
    "  testq %rax,%rax\n"
    "  jnz 1f\n"
    "  movl $1,%eax\n"
    "1: jmp *72(%rcx)\n"
);
#endif

constexpr uint32_t kMagicStart = 0xdef1649c;
constexpr uint32_t kMagicEnd = 0xb37592a0;
constexpr uint64_t kStackSignature = 0x7149f2ca7149f2caull;
constexpr uint32_t kStateRun = 1, kStateIdle = 2, kStateTerminated = 3;
constexpr uint64_t kErrNull = 0x80590001u, kErrAlignment = 0x80590002u;
constexpr uint64_t kErrRange = 0x80590003u, kErrInvalid = 0x80590004u;
constexpr uint64_t kErrPermission = 0x80590005u, kErrState = 0x80590006u;

using FiberEntry = void (*)(uint64_t, uint64_t);

struct GuestFiber {
    uint32_t magic_start;
    uint32_t state;
    FiberEntry entry;
    uint64_t initialize_arg;
    void* stack;
    uint64_t stack_size;
    char name[32];
    void* saved_context;
    uint32_t flags;
    uint32_t pad;
    void* context_start;
    void* context_end;
    uint32_t magic_end;
};
static_assert(sizeof(GuestFiber) <= 256);

struct FiberRecord {
    GuestFiber* guest = nullptr;
    FiberEntry entry = nullptr;
    uint64_t initialize_arg = 0;
    void* stack = nullptr;
    uint64_t stack_size = 0;
    std::unique_ptr<uint8_t[]> fallback_stack;
    FiberContext resume{};
    bool started = false;
    bool suspended = false;
    uint64_t incoming_arg = 0;
    // #3615: address of the import-stub frame slot holding the guest %fs of the host thread this
    // fiber suspended on. Repaired on resume, because the resuming thread returns through that frame
    // and its epilogue restores %fs from here. 0 when the suspending call had no stub frame.
    uint64_t suspend_fs_slot = 0;
};

struct ThreadFibers {
    FiberContext root{};
    FiberRecord* current = nullptr;
    FiberRecord* previous = nullptr;
    uint64_t return_arg = 0;
    uint64_t guest_fs = 0;
    bool active = false;
#ifdef _WIN32
    void* root_stack_base = nullptr;
    void* root_stack_limit = nullptr;
#endif
};

// SceFiberOptParam is 0x80 bytes; see fiber_opt_param_initialize for how that is derived.
constexpr size_t kFiberOptParamBytes = 0x80;

std::mutex g_fiber_mutex;
std::unordered_map<GuestFiber*, std::unique_ptr<FiberRecord>> g_fibers;
std::unordered_map<uint64_t, std::unique_ptr<ThreadFibers>> g_threads;

uint64_t thread_key() {
#ifdef _WIN32
    return (uint64_t)GetCurrentThreadId();
#elif defined(__linux__)
    return (uint64_t)syscall(SYS_gettid);
#else
    return (uint64_t)(uintptr_t)pthread_self();
#endif
}

ThreadFibers* thread_fibers() {
    const uint64_t key = thread_key();
    std::lock_guard<std::mutex> lock(g_fiber_mutex);
    auto& slot = g_threads[key];
    if (!slot) slot = std::make_unique<ThreadFibers>();
    return slot.get();
}

FiberRecord* find_fiber(GuestFiber* guest) {
    std::lock_guard<std::mutex> lock(g_fiber_mutex);
    auto it = g_fibers.find(guest);
    return it == g_fibers.end() ? nullptr : it->second.get();
}

bool fiber_log() { static const bool enabled = std::getenv("PROSPER_FIBERLOG") != nullptr; return enabled; }

// PROSPER_FIBER_DUMP_MS=<ms> (default off): print every fiber's state on a timer.
//
// On a fiber-based title a PARKED job is invisible to every other instrument prosper and the host
// debugger have. It is not on a host thread's stack (the worker switched away and went back to
// scanning for work), so `thread apply all bt` shows only idle workers; and its own stack lives in a
// guest region nothing points at, so a scan of that memory cannot tell a live frame from the stale
// bytes of a deeper call that already returned. The one place the boundary between the two is known
// exactly is here: `resume.words[8]` is the stack pointer the fiber will be restored with.
//
// Pair it with `tools/re/guest_stacks.py`, which reads the same guest stacks from outside: this
// names where each fiber's live frames STOP, that one shows what is in them.
//
// Word indices below are the save layout in the assembly at the top of this file, and the two ABIs
// do NOT agree: SysV saves rbx, rbp, r12..r15 and puts rsp at +48 / the return address at +56
// (words 6 and 7), while the Microsoft x64 block additionally preserves rdi and rsi first and so
// puts the same two at +64 / +72 (words 8 and 9). Reading the wrong pair prints a plausible 0
// rather than failing, which is exactly the kind of quiet wrong answer this dump exists to avoid.
void fiber_dump_state(std::FILE* out) {
    std::lock_guard<std::mutex> lock(g_fiber_mutex);
    std::fprintf(out, "[fiber-dump] %zu fiber(s), %zu thread record(s)\n", g_fibers.size(),
                 g_threads.size());
    for (const auto& kv : g_fibers) {
        const FiberRecord* r = kv.second.get();
        std::fprintf(out,
                     "[fiber-dump] fiber=%p entry=%p stack=%p+0x%llx started=%d suspended=%d "
                     "resume_rsp=0x%llx resume_rip=0x%llx\n",
                     (void*)kv.first, (void*)(uintptr_t)r->entry, r->stack,
                     (unsigned long long)r->stack_size, (int)r->started, (int)r->suspended,
#ifdef _WIN32
                     (unsigned long long)r->resume.words[8],
                     (unsigned long long)r->resume.words[9]);
#else
                     (unsigned long long)r->resume.words[6],
                     (unsigned long long)r->resume.words[7]);
#endif
    }
}

void start_fiber_dump_thread_once() {
    static std::atomic<bool> started{false};
    if (started.exchange(true)) return;
    const char* e = std::getenv("PROSPER_FIBER_DUMP_MS");
    if (!e) return;
    const long ms = std::atol(e);
    if (ms <= 0) return;   // a malformed value disables its own trigger rather than picking a rate
    std::thread([ms] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            fiber_dump_state(stderr);
        }
    }).detach();
}

// Re-point the stub frame this fiber suspended inside at the RESUMING thread's own guest TP. See
// callback_repair_guest_fs_slot for why the stash goes stale across a fiber migration. Called on the
// resume side of every context save, so it also covers the ordinary same-thread resume (a no-op).
// BOUND: this repairs the SUSPENDING CALL'S OWN frame and no other. An *outer* swap-stub frame still
// on the fiber's stack -- a guest callback invoked from inside another HLE handler, which then yields
// the fiber -- unwinds through its own `pop r11; wrfsbase r11` with the suspending thread's TP and is
// not covered. Not on any measured path (#3615); do not read this as covering it.
void repair_suspend_fs_slot(FiberRecord* fiber) {
    if (!fiber) return;
    const uint64_t slot = fiber->suspend_fs_slot;
    fiber->suspend_fs_slot = 0;
    // A resuming thread that never activated guest TLS reports 0, and the frame is then left alone:
    // writing 0 would make the stub epilogue `wrfsbase 0` and strand the guest with no TLS at all.
    // That thread still runs on the foreign TCB -- declining is the lesser harm, not a repair. Not
    // reachable from guest code today, since sceFiberRun arrives on threads that have activated.
    //
    // HOW THIS FAILS IF IT EVER DOES FAIL, which is the part not to rediscover the hard way.
    // guest_tls_own_tp() is a thread_local, so it follows %fs -- measured, in every TLS model. It is
    // therefore correct only while this thread is on its OWN host TCB, and that holds by INDUCTION:
    // a worker's first resume necessarily happens before any foreign TCB can be installed on it, and
    // each repair keeps it that way. That is the repair's own coverage claim restated, not an
    // independent guarantee, so if a path outside the BOUND above ever corrupts a thread, the
    // induction is already broken -- and the failure is SILENT in the worst way: that thread reads
    // the FOREIGN t_guest_tp, `*p == own_tp` matches, and the repair declines on exactly the thread
    // that needed it. A corrupted worker shows ZERO repairs, indistinguishable from a healthy one.
    // Worse, its own_tp is then non-zero-and-foreign, so it starts writing that foreign TP into
    // THIRD threads' frames. If this is ever suspected, the cheap detector is to record the
    // suspending thread_key() beside suspend_fs_slot and shout when a resume arrives on a different
    // tid whose slot ALREADY equals own_tp -- unreachable in a healthy run, exact signature of this.
    const uint64_t own = guest_tls_own_tp();
    if (!callback_repair_guest_fs_slot(slot, own)) return;
    if (fiber_log())
        std::fprintf(stderr, "[fiber] repaired stub guest-TP on tid=%llu -> 0x%llx\n",
                     (unsigned long long)thread_key(), (unsigned long long)own);
}

#ifdef _WIN32
void set_windows_stack_bounds(void* limit, void* base) {
    NT_TIB* tib = (NT_TIB*)NtCurrentTeb();
    tib->StackLimit = limit;
    tib->StackBase = base;
}

void set_windows_fiber_stack(const FiberRecord* fiber) {
    set_windows_stack_bounds(fiber->stack,
                             (void*)((uintptr_t)fiber->stack + fiber->stack_size));
}

void set_windows_root_stack(const ThreadFibers* thread) {
    set_windows_stack_bounds(thread->root_stack_limit, thread->root_stack_base);
}
#endif

#ifndef _WIN32
#ifndef __APPLE__
inline uint64_t read_fsbase() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
inline void write_fsbase(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }
#endif

extern "C" uint64_t prosper_call_guest_on_stack(uintptr_t stack_top, uint64_t fn,
                                                  uint64_t a0, uint64_t a1);
asm(
    ".text\n"
    ".p2align 4\n"
    PROSPER_FIBER_ASM_HEADER(prosper_call_guest_on_stack)
    "  pushq %r12\n"
    "  movq %rsp,%r12\n"
    "  movq %rsi,%r11\n"
    "  movq %rdi,%rsp\n"
    "  andq $-16,%rsp\n"
    "  movq %rdx,%rdi\n"
    "  movq %rcx,%rsi\n"
    "  call *%r11\n"
    "  movq %r12,%rsp\n"
    "  popq %r12\n"
    "  ret\n"
);
#else
extern "C" uint64_t prosper_call_guest_on_stack(uintptr_t stack_top, uint64_t fn,
                                                  uint64_t a0, uint64_t a1);
#endif

uint64_t call_fiber_entry(ThreadFibers* thread, FiberRecord* fiber, uint64_t run_arg) {
    uintptr_t stack_top = (uintptr_t)fiber->stack + fiber->stack_size;
#if defined(__APPLE__)
    // Rosetta does not expose FSGSBASE and macOS guest TLS is maintained by the import-trap
    // dispatcher.  Fiber entry therefore leaves the host FS base untouched.
    (void)thread;
    return prosper_call_guest_on_stack(stack_top, (uint64_t)(uintptr_t)fiber->entry,
                                       fiber->initialize_arg, run_arg);
#elif !defined(_WIN32)
    const uint64_t host_fs = read_fsbase();
    if (thread->guest_fs) write_fsbase(thread->guest_fs);
    uint64_t result = prosper_call_guest_on_stack(stack_top, (uint64_t)(uintptr_t)fiber->entry,
                                                   fiber->initialize_arg, run_arg);
    // A normal return did not pass through an import gate, so %fs is still the guest value.
    if (thread->guest_fs) write_fsbase(host_fs);
    return result;
#else
    // Windows exception delivery and __chkstk consult NT_TIB.StackBase/StackLimit.  The assembly
    // helper switches RSP to the Sony-provided stack, so describe that stack in the TEB for the
    // duration of guest execution and restore the caller's bounds on a normal return.  Yield/switch
    // paths restore their destination bounds immediately before the non-local context jump below.
    NT_TIB* tib = (NT_TIB*)NtCurrentTeb();
    void* saved_base = tib->StackBase;
    void* saved_limit = tib->StackLimit;
    set_windows_fiber_stack(fiber);
    uint64_t result = prosper_call_guest_on_stack(stack_top, (uint64_t)(uintptr_t)fiber->entry,
                                                   fiber->initialize_arg, run_arg);
    set_windows_stack_bounds(saved_limit, saved_base);
    return result;
#endif
}

uint64_t fiber_initialize(GuestFiber* guest, const char* name, FiberEntry entry,
                          uint64_t initialize_arg, void* stack, uint64_t stack_size,
                          const void*, uint32_t) {
    if (!guest || !name || !entry) return kErrNull;
    if (((uintptr_t)guest & 7) || ((uintptr_t)stack & 15)) return kErrAlignment;
    if (stack_size && stack_size < 512) return kErrRange;
    if ((!stack) != (!stack_size) || (stack_size & 15)) return kErrInvalid;

    auto record = std::make_unique<FiberRecord>();
    record->guest = guest; record->entry = entry; record->initialize_arg = initialize_arg;
    record->stack = stack; record->stack_size = stack_size;
    if (!record->stack) {
        record->stack_size = 1024 * 1024;
        record->fallback_stack = std::make_unique<uint8_t[]>(record->stack_size);
        record->stack = record->fallback_stack.get();
    }

    std::memset(guest, 0, sizeof(*guest));
    guest->magic_start = kMagicStart; guest->state = kStateIdle; guest->entry = entry;
    guest->initialize_arg = initialize_arg; guest->stack = stack; guest->stack_size = stack_size;
    std::strncpy(guest->name, name, sizeof(guest->name) - 1);
    guest->context_start = stack; guest->context_end = stack ? (uint8_t*)stack + stack_size : nullptr;
    guest->magic_end = kMagicEnd;
    if (stack && stack_size >= sizeof(uint64_t)) *(uint64_t*)stack = kStackSignature;

    {
        std::lock_guard<std::mutex> lock(g_fiber_mutex);
        g_fibers[guest] = std::move(record);
    }
    if (fiber_log())
        std::fprintf(stderr, "[fiber] init %p name=%s entry=%p stack=%p+0x%llx\n",
                     (void*)guest, name, (void*)entry, stack, (unsigned long long)stack_size);
    return 0;
}

// sceFiberRename(fiber, name) -- a real implementation rather than the dispatcher's return-0
// default, which answers SCE_OK while leaving the name untouched. prosper already carries the only
// state involved: `GuestFiber::name` is the 32-byte field this file fills at initialize and prints
// in its own `[fiber] init` diagnostic, so no struct layout has to be guessed.
//
// WHY IT IS HERE, stated precisely because an earlier version of this comment got it wrong.
// PPSA05684 IMPORTS this NID -- but importing is not calling. `tools/re/nid_gate_scan.py` finds
// ZERO call sites for `JzyT91ucGDc` in both `eboot.bin` and `tllr-boot.bin`, and that zero is
// controlled rather than silent: the same tool on the same binaries reports call sites for the
// sibling fiber NIDs (`hVYD7Ou2pCQ`, `a0LLrZWac0M`, `asjUJJ+aa8s`). The title links this function
// and never calls it.
//
// So this is NOT justified as "the guest exercises it". It is justified as: the implementation is
// complete, needs no reverse engineering, costs one bounded copy, and removes a silent-wrong answer
// from a linked entry point. That reasoning would not support guessing at a struct layout, and it
// deliberately does not.
//
// CONFIDENCE: HIGH on the copy itself (the field is prosper's own) and on the NID mapping
// (`JzyT91ucGDc` is sceFiberRename, unique across the 3.20 firmware set).
// CONFIDENCE: LOW on three semantics that are inferred rather than established, none of which any
// corpus title can currently exercise:
//   * whether a RUNNING fiber may be renamed. `fiber_finalize` refuses `kStateRun`; this does not,
//     on the reading that a name is metadata rather than context. Unverified.
//   * whether an over-long name truncates-and-succeeds (chosen here) or returns a range error.
//   * the kErrInvalid/kErrNull split for a null fiber versus a null name, which mirrors
//     `fiber_finalize` and `fiber_initialize` respectively rather than any measured contract.
// A title that calls this and inspects the result is what would settle all three.
uint64_t fiber_rename(GuestFiber* guest, const char* name) {
    FiberRecord* fiber = find_fiber(guest);
    if (!guest || !fiber) return kErrInvalid;
    if (!name) return kErrNull;
    std::snprintf(guest->name, sizeof(guest->name), "%s", name);
    if (fiber_log())
        std::fprintf(stderr, "[fiber] rename %p -> \"%s\"\n", (void*)guest, guest->name);
    return 0;
}

// sceFiberOptParamInitialize (NID asjUJJ+aa8s) -- #3636.
//
// Unregistered, this fell to the dispatcher's `return 0`, which reports SUCCESS and leaves the
// caller's buffer untouched: the guest then hands 128 bytes of its own uninitialised stack to
// sceFiberInitialize as an optional-parameter block. Uncharted (PPSA05684) calls it once per fiber,
// 100 times in a 25 s boot, and checks the result (`test eax,eax ; jne <error>` at eboot+0x1f0d).
//
// SIZE: 0x80 bytes, derived from the caller's own frame rather than assumed. At eboot+0x1efe the
// guest takes `lea r15,[rbp-0xe0]` for this call and `lea r13,[rbp-0x60]` for the next local, so the
// block it reserves spans exactly 0x80 -- which also matches the published SceFiberOptParam size.
// CONFIDENCE: HIGH on the size (two independent routes agree), MED on the contents: the documented
// contract is "initialise to defaults" and every default in the published structure is zero, but
// prosper has no live capture of a non-zero field to check that against.
//
// prosper's own fiber implementation does not consume the block today, so this changes no behaviour
// for a title that only passes it through. It closes the hazard the #2951 family is named for: a
// success answer that writes no out-parameter, which the guest cannot distinguish from a real one.
uint64_t fiber_opt_param_initialize(void* opt_param) {
    if (!opt_param) return kErrNull;
    std::memset(opt_param, 0, kFiberOptParamBytes);
    return 0;
}

uint64_t fiber_finalize(GuestFiber* guest) {
    FiberRecord* fiber = find_fiber(guest);
    if (!guest || !fiber) return kErrInvalid;
    if (guest->state == kStateRun) return kErrState;
    guest->state = kStateTerminated;
    std::lock_guard<std::mutex> lock(g_fiber_mutex);
    g_fibers.erase(guest);
    return 0;
}

uint64_t fiber_run_impl(GuestFiber* guest, uint64_t run_arg, uint64_t* return_arg,
                        uint64_t entry_rsp) {
    FiberRecord* fiber = find_fiber(guest);
    if (!guest || !fiber) return kErrInvalid;
    ThreadFibers* thread = thread_fibers();
    if (thread->active) return kErrPermission;
    if (guest->state == kStateTerminated) return kErrState;

    thread->active = true; thread->current = fiber; thread->previous = nullptr;
#if defined(__APPLE__)
    thread->guest_fs = guest_tls_tp();
#elif !defined(_WIN32)
    thread->guest_fs = callback_guest_fs_from_entry_stack(entry_rsp);
#else
    NT_TIB* tib = (NT_TIB*)NtCurrentTeb();
    thread->root_stack_base = tib->StackBase;
    thread->root_stack_limit = tib->StackLimit;
#endif
    fiber->incoming_arg = run_arg; guest->state = kStateRun;
    if (fiber_log())
        std::fprintf(stderr, "[fiber] run %p started=%d arg=0x%llx tid=%llu guest_fs=0x%llx\n",
                     (void*)guest, fiber->started ? 1 : 0, (unsigned long long)run_arg,
                     (unsigned long long)thread_key(), (unsigned long long)thread->guest_fs);

    if (prosper_fiber_context_save(&thread->root) == 0) {
        if (fiber->suspended) {
#ifdef _WIN32
            set_windows_fiber_stack(fiber);
#endif
            prosper_fiber_context_restore(&fiber->resume, 1);
        }
        fiber->started = true;
        (void)call_fiber_entry(thread, fiber, run_arg);
        // Sony entries conventionally yield/return through sceFiberReturnToThread. Treat a plain
        // return as a zero-valued return-to-thread so a malformed title cannot strand the caller.
        thread->return_arg = 0;
    }

    if (return_arg) *return_arg = thread->return_arg;
    if (thread->current && thread->current->guest->state != kStateTerminated)
        thread->current->guest->state = kStateIdle;
    thread->current = nullptr; thread->previous = nullptr; thread->active = false;
    return 0;
}

uint64_t fiber_return_to_thread_impl(uint64_t return_arg, uint64_t* next_run_arg,
                                    uint64_t entry_rsp) {
    ThreadFibers* thread = thread_fibers();
    FiberRecord* current = thread->current;
    if (!thread->active || !current) return kErrPermission;
    if (prosper_fiber_context_save(&current->resume) == 0) {
        current->suspended = true;
        current->guest->saved_context = &current->resume;
        current->guest->state = kStateIdle;
        current->suspend_fs_slot = callback_guest_fs_slot_from_entry_stack(entry_rsp);
        thread->return_arg = return_arg;
        if (fiber_log())
            std::fprintf(stderr, "[fiber] yield %p arg=0x%llx tid=%llu\n", (void*)current->guest,
                         (unsigned long long)return_arg, (unsigned long long)thread_key());
#ifdef _WIN32
        set_windows_root_stack(thread);
#endif
        prosper_fiber_context_restore(&thread->root, 1);
    }
    // Resumed -- possibly on a different host thread than the one that suspended us, in which case the
    // frame we are about to return through carries that thread's guest TP.
    repair_suspend_fs_slot(current);
    // ...and `thread` above is that OTHER thread's record. Everything before the suspend deliberately
    // uses it (thread->root and thread->return_arg belong to the sceFiberRun that is about to be
    // unwound to), but the bookkeeping below describes "which fiber is this host thread running", so
    // it must land on the RESUMING thread. Observed on Uncharted as two sceFiberSwitch refusals per
    // run ("target is the running fiber" / "target already running") and the guest's own
    // "ASSERTION: retval == SCE_OK" in SwitchToFiber; removing just these two lines reproduces it
    // 3 runs out of 3.
    //
    // NO UNIT ARM COVERS THIS, and the reason is worth writing down. Both resume paths already set
    // the RESUMING thread's ThreadFibers::current before they jump, so the write here is
    // redundant-but-correct on that side; the whole benefit is not CLOBBERING the suspending
    // thread's record. Observing that needs the suspending thread to be running a fiber of its own
    // at the moment a second thread resumes this one -- T1 running fiber C, T2 resuming fiber A,
    // then asserting T1's sceFiberSwitch is not refused -- which is constructible with two threads
    // and a handshake inside the fiber bodies, but is a synchronised multi-thread fiber test and was
    // judged more likely to become a flaky guard than to catch a regression. If this line is ever
    // touched again, build that arm.
    thread = thread_fibers();
    current->guest->state = kStateRun;
    thread->current = current;
    if (next_run_arg) *next_run_arg = current->incoming_arg;
    return 0;
}

#ifdef _WIN32
// Windows registers these handlers directly rather than through PROSPER_ASM_TRAMPOLINE, so there is
// no entry-%rsp to forward -- and the MS/SysV integer bridge fills the 4th parameter unconditionally
// (`mov r9,rcx ; MS 4th = a3`, sysv_ms_bridge.cpp:123), which at a three-argument sceFiber* call is
// caller-saved guest SCRATCH. Passing it on as `entry_rsp` would hand a garbage value to a function
// that dereferences it. These wrappers pass an explicit 0 instead; Windows stubs stash no guest %fs
// on the guest stack, so there is nothing to locate or repair there anyway.
uint64_t fiber_return_to_thread_win(uint64_t return_arg, uint64_t* next_run_arg) {
    return fiber_return_to_thread_impl(return_arg, next_run_arg, 0);
}
#endif

uint64_t fiber_switch_impl(GuestFiber* target_guest, uint64_t run_arg, uint64_t* resumed_arg,
                           uint64_t entry_rsp) {
    ThreadFibers* thread = thread_fibers();
    FiberRecord* current = thread->current;
    FiberRecord* target = find_fiber(target_guest);
    // A refusal here surfaces in the guest as a failed sceFiberSwitch, which Naughty Dog titles assert
    // on ("ASSERTION: retval == SCE_OK" in SwitchToFiber). Name the branch rather than making the next
    // reader bisect four of them.
    auto refuse = [&](uint64_t err, const char* why) {
        if (fiber_log())
            std::fprintf(stderr, "[fiber] switch REFUSED (%s) tid=%llu cur=%p target=%p tstate=%d\n",
                         why, (unsigned long long)thread_key(), (void*)current, (void*)target_guest,
                         target_guest ? (int)target_guest->state : -1);
        return err;
    };
    if (!thread->active || !current) return refuse(kErrPermission, "this thread is not running a fiber");
    if (!target_guest || !target) return refuse(kErrInvalid, "target is not an initialized fiber");
    if (target == current) return refuse(kErrInvalid, "target is the running fiber");
    if (target_guest->state == kStateRun) return refuse(kErrState, "target already running");
    if (target_guest->state == kStateTerminated) return refuse(kErrState, "target terminated");

    if (prosper_fiber_context_save(&current->resume) == 0) {
        current->suspended = true; current->guest->saved_context = &current->resume;
        current->guest->state = kStateIdle;
        current->suspend_fs_slot = callback_guest_fs_slot_from_entry_stack(entry_rsp);
        target->incoming_arg = run_arg; target_guest->state = kStateRun;
        thread->previous = current; thread->current = target;
        // A SUCCESSFUL switch was the one fiber transition PROSPER_FIBERLOG did not record — only
        // refusals were logged here, so a run that switched thousands of times and a run that never
        // switched at all produced the same (empty) evidence. On Uncharted that absence was read as
        // "sceFiberSwitch is not involved", which the counts could not support: NdJob parks a waiting
        // job by switching to a pool fiber, and every one of those was invisible.
        if (fiber_log())
            std::fprintf(stderr, "[fiber] switch %p -> %p arg=0x%llx tid=%llu\n",
                         (void*)current->guest, (void*)target_guest,
                         (unsigned long long)run_arg, (unsigned long long)thread_key());
        if (target->suspended) {
#ifdef _WIN32
            set_windows_fiber_stack(target);
#endif
            prosper_fiber_context_restore(&target->resume, 1);
        }
        target->started = true;
        (void)call_fiber_entry(thread, target, run_arg);
        thread->return_arg = 0;
#ifdef _WIN32
        set_windows_root_stack(thread);
#endif
        prosper_fiber_context_restore(&thread->root, 1);
    }
    repair_suspend_fs_slot(current);
    thread = thread_fibers();          // resumed: bookkeeping belongs to THIS host thread (see above)
    thread->current = current; current->guest->state = kStateRun;
    if (resumed_arg) *resumed_arg = current->incoming_arg;
    return 0;
}

#ifdef _WIN32
uint64_t fiber_run_win(uint64_t guest, uint64_t run_arg, uint64_t* return_arg) {
    return fiber_run_impl((GuestFiber*)(uintptr_t)guest, run_arg, return_arg, 0);
}
uint64_t fiber_switch_win(uint64_t target_guest, uint64_t run_arg, uint64_t* resumed_arg) {
    return fiber_switch_impl((GuestFiber*)(uintptr_t)target_guest, run_arg, resumed_arg, 0);
}
#endif

uint64_t fiber_get_self(GuestFiber** out) {
    if (!out) return kErrNull;
    ThreadFibers* thread = thread_fibers();
    if (!thread->active || !thread->current) return kErrPermission;
    *out = thread->current->guest;
    return 0;
}

#ifndef _WIN32
extern "C" uint64_t fiber_run_c(uint64_t a0, uint64_t a1, uint64_t a2,
                                 uint64_t, uint64_t, uint64_t, uint64_t entry_rsp) {
    return fiber_run_impl((GuestFiber*)(uintptr_t)a0, a1, (uint64_t*)(uintptr_t)a2, entry_rsp);
}
PROSPER_ASM_TRAMPOLINE(fiber_run_entry, fiber_run_c)
extern "C" void fiber_run_entry();

extern "C" uint64_t fiber_switch_c(uint64_t a0, uint64_t a1, uint64_t a2,
                                    uint64_t, uint64_t, uint64_t, uint64_t entry_rsp) {
    return fiber_switch_impl((GuestFiber*)(uintptr_t)a0, a1, (uint64_t*)(uintptr_t)a2, entry_rsp);
}
PROSPER_ASM_TRAMPOLINE(fiber_switch_entry, fiber_switch_c)
extern "C" void fiber_switch_entry();

// sceFiberReturnToThread needs the HLE entry %rsp too (#3615): the fiber suspends INSIDE this call,
// so this frame is the one a resuming host thread returns through, and its stashed guest %fs is the
// slot that has to be repaired.
extern "C" uint64_t fiber_return_to_thread_c(uint64_t a0, uint64_t a1, uint64_t, uint64_t,
                                             uint64_t, uint64_t, uint64_t entry_rsp) {
    return fiber_return_to_thread_impl(a0, (uint64_t*)(uintptr_t)a1, entry_rsp);
}
PROSPER_ASM_TRAMPOLINE(fiber_return_to_thread_entry, fiber_return_to_thread_c)
extern "C" void fiber_return_to_thread_entry();
#endif

} // namespace

void register_fiber_hle() {
    start_fiber_dump_thread_once();   // PROSPER_FIBER_DUMP_MS (no-op unless armed)
    Hle::register_fn("hVYD7Ou2pCQ", (HleFn)fiber_initialize, "_sceFiberInitializeImpl");
    Hle::register_fn("JeNX5F-NzQU", (HleFn)fiber_finalize, "sceFiberFinalize");
    Hle::register_fn("JzyT91ucGDc", (HleFn)fiber_rename, "sceFiberRename");
    Hle::register_fn("asjUJJ+aa8s", (HleFn)fiber_opt_param_initialize, "sceFiberOptParamInitialize");
#ifndef _WIN32
    Hle::register_fn("a0LLrZWac0M", (HleFn)fiber_run_entry, "sceFiberRun");
    Hle::register_fn("PFT2S-tJ7Uk", (HleFn)fiber_switch_entry, "sceFiberSwitch");
#else
    Hle::register_fn("a0LLrZWac0M", (HleFn)fiber_run_win, "sceFiberRun");
    Hle::register_fn("PFT2S-tJ7Uk", (HleFn)fiber_switch_win, "sceFiberSwitch");
#endif
#ifndef _WIN32
    Hle::register_fn("B0ZX2hx9DMw", (HleFn)fiber_return_to_thread_entry, "sceFiberReturnToThread");
#else
    Hle::register_fn("B0ZX2hx9DMw", (HleFn)fiber_return_to_thread_win, "sceFiberReturnToThread");
#endif
    Hle::register_fn("p+zLIOg27zU", (HleFn)fiber_get_self, "sceFiberGetSelf");
}

} // namespace prosper
