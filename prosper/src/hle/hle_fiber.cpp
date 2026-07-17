// hle_fiber.cpp - cooperative libSceFiber execution for native PS5 engines.
//
// Guest code runs natively, so a fiber can retain its real guest call stack. A small private
// register context transfers between suspended HLE frames; first entry is made on the
// guest-provided stack. The CRT setjmp/longjmp cannot be used here: fortified glibc deliberately
// rejects jumps to a lower stack and the Windows CRT tries to unwind SEH frames.
#include "dispatch.hpp"
#include "callback_fs.hpp"
#include "../host/posix_shim.hpp"

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
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
    ".globl prosper_fiber_context_save\n"
    ".type prosper_fiber_context_save,@function\n"
    "prosper_fiber_context_save:\n"
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
    ".globl prosper_fiber_context_restore\n"
    ".type prosper_fiber_context_restore,@function\n"
    "prosper_fiber_context_restore:\n"
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
inline uint64_t read_fsbase() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
inline void write_fsbase(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }

extern "C" uint64_t prosper_call_guest_on_stack(uintptr_t stack_top, uint64_t fn,
                                                  uint64_t a0, uint64_t a1);
asm(
    ".text\n"
    ".p2align 4\n"
    ".globl prosper_call_guest_on_stack\n"
    ".type prosper_call_guest_on_stack,@function\n"
    "prosper_call_guest_on_stack:\n"
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
#ifndef _WIN32
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
#ifndef _WIN32
    thread->guest_fs = callback_guest_fs_from_entry_stack(entry_rsp);
#else
    NT_TIB* tib = (NT_TIB*)NtCurrentTeb();
    thread->root_stack_base = tib->StackBase;
    thread->root_stack_limit = tib->StackLimit;
#endif
    fiber->incoming_arg = run_arg; guest->state = kStateRun;
    if (fiber_log())
        std::fprintf(stderr, "[fiber] run %p started=%d arg=0x%llx\n", (void*)guest,
                     fiber->started ? 1 : 0, (unsigned long long)run_arg);

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

uint64_t fiber_return_to_thread(uint64_t return_arg, uint64_t* next_run_arg) {
    ThreadFibers* thread = thread_fibers();
    FiberRecord* current = thread->current;
    if (!thread->active || !current) return kErrPermission;
    if (prosper_fiber_context_save(&current->resume) == 0) {
        current->suspended = true;
        current->guest->saved_context = &current->resume;
        current->guest->state = kStateIdle;
        thread->return_arg = return_arg;
        if (fiber_log())
            std::fprintf(stderr, "[fiber] yield %p arg=0x%llx\n", (void*)current->guest,
                         (unsigned long long)return_arg);
#ifdef _WIN32
        set_windows_root_stack(thread);
#endif
        prosper_fiber_context_restore(&thread->root, 1);
    }
    current->guest->state = kStateRun;
    thread->current = current;
    if (next_run_arg) *next_run_arg = current->incoming_arg;
    return 0;
}

uint64_t fiber_switch_impl(GuestFiber* target_guest, uint64_t run_arg, uint64_t* resumed_arg,
                           uint64_t) {
    ThreadFibers* thread = thread_fibers();
    FiberRecord* current = thread->current;
    FiberRecord* target = find_fiber(target_guest);
    if (!thread->active || !current) return kErrPermission;
    if (!target_guest || !target || target == current) return kErrInvalid;
    if (target_guest->state == kStateRun || target_guest->state == kStateTerminated) return kErrState;

    if (prosper_fiber_context_save(&current->resume) == 0) {
        current->suspended = true; current->guest->saved_context = &current->resume;
        current->guest->state = kStateIdle;
        target->incoming_arg = run_arg; target_guest->state = kStateRun;
        thread->previous = current; thread->current = target;
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
    thread->current = current; current->guest->state = kStateRun;
    if (resumed_arg) *resumed_arg = current->incoming_arg;
    return 0;
}

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
#endif

} // namespace

void register_fiber_hle() {
    Hle::register_fn("hVYD7Ou2pCQ", (HleFn)fiber_initialize, "_sceFiberInitializeImpl");
    Hle::register_fn("JeNX5F-NzQU", (HleFn)fiber_finalize, "sceFiberFinalize");
#ifndef _WIN32
    Hle::register_fn("a0LLrZWac0M", (HleFn)fiber_run_entry, "sceFiberRun");
    Hle::register_fn("PFT2S-tJ7Uk", (HleFn)fiber_switch_entry, "sceFiberSwitch");
#else
    Hle::register_fn("a0LLrZWac0M", (HleFn)fiber_run_impl, "sceFiberRun");
    Hle::register_fn("PFT2S-tJ7Uk", (HleFn)fiber_switch_impl, "sceFiberSwitch");
#endif
    Hle::register_fn("B0ZX2hx9DMw", (HleFn)fiber_return_to_thread, "sceFiberReturnToThread");
    Hle::register_fn("p+zLIOg27zU", (HleFn)fiber_get_self, "sceFiberGetSelf");
}

} // namespace prosper
