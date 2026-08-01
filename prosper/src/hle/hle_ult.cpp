// hle_ult.cpp — libSceUlt: Sony's user-level (cooperative) threading library.
//
// libSceUlt provides "ulthreads" scheduled by a runtime on top of OS worker threads, plus the
// mutexes, condition variables and waiting-queue resource pools built on that scheduler. Phase 1
// (#1603/#1614) registered the entry points and refused them loudly so the missing semantics stopped
// being silent. This file now implements them.
//
// ============================================================================================
// THE ABI, DERIVED FROM THE GUEST'S OWN CALL SITES — NOT FROM A HEADER WE DO NOT HAVE
// ============================================================================================
// Exactly one title in the library imports libSceUlt: Earthion (PPSA28061), 16 functions. Every one
// of its Ult call sites lives in a single engine wrapper layer at eboot+0x9c20..0xa17a, which sets up
// all arguments explicitly, so the layouts below are READ, not inferred. Method: flatten eboot.bin
// with tools/il2cpp/prx_to_elf.py, resolve the 16 JMPREL slots -> PLT stubs -> call sites with
// tools/re/xref.py, disassemble with tools/re/edis.py.
//
// The object names the guest passes settle the provenance: the pool is "waiting queue", the runtime
// is "sample runtime", the mutex "ultmtx", the condvar "ultcv". This is Sony's own libSceUlt SDK
// sample usage pattern, which is why the layouts are the canonical ones.
//
//   sceUltInitialize(void) -> int32                       [0 args: nothing is set up at eboot+0x9c3c]
//   sceUltFinalize(void)   -> int32                       [0 args: eboot+0x9db4]
//
//   size_t sceUltUlthreadRuntimeGetWorkAreaSize(u32 numMaxUlthread, u32 numWorkerThread)
//   size_t sceUltWaitingQueueResourcePoolGetWorkAreaSize(u32 numThreads, u32 numSyncObjects)
//       The size is the RETURN VALUE (rax); there is no out-parameter. See kUltSizeContract (#1618).
//
//   int _sceUltUlthreadRuntimeCreate(runtime, const char* name, u32 numMaxUlthread,
//                                    u32 numWorkerThread, void* workArea, const OptParam*, u32 v)
//   int _sceUltWaitingQueueResourcePoolCreate(pool, const char* name, u32 numThreads,
//                                    u32 numSyncObjects, void* workArea, const OptParam*, u32 v)
//       7 args; `v` arrives on the stack (the import bridge forwards it as a7).
//
//   int _sceUltUlthreadCreate(ult, const char* name, int32_t(*entry)(uint64_t), uint64_t arg,
//                             void* context, size_t sizeContext, runtime, const OptParam*, u32 v)
//   int sceUltUlthreadJoin(ult, int32_t* status)
//
//   int _sceUltMutexCreate(mutex, const char* name, pool, const OptParam*, u32 v)   [5 args, all regs]
//   int sceUltMutexLock(mutex) / sceUltMutexUnlock(mutex) / sceUltMutexDestroy(mutex)
//
//   int _sceUltConditionVariableCreate(cv, const char* name, mutex, const OptParam*, u32 v)
//   int sceUltConditionVariableWait(cv)      <- ONE argument; the mutex is bound at create.
//   int sceUltConditionVariableSignal(cv) / sceUltConditionVariableDestroy(cv)
//
// `v` is 0x12000000 at every create site — the trailing api/SDK-version argument the `_sce*` forms
// take (0x12 = SDK 18.00). CONFIDENCE: HIGH; it is a literal immediate at all five create sites.
// prosper records it and does not gate on it: refusing an unknown version would be inventing policy.
//
// Earthion's own parameters: runtime (numMaxUlthread=16, numWorkerThread=3), pool (numThreads=16,
// numSyncObjects=16), ulthread sizeContext=0x2000, every optParam NULL.
//
// ============================================================================================
// THE GUEST OBJECT IS A 256-BYTE CALLER-OWNED BLOB; prosper WRITES ONLY A 16-BYTE HEADER
// ============================================================================================
// Proven, not assumed. At eboot+0x241730 a mutex is followed by an unrelated 8-byte queue head at
// 0x241830 (read at eboot+0x137c7) — exactly 0x100 later. Corroborated by the C++ class constructed
// at eboot+0xd8a0, whose mutex/condvar/mutex members sit at +0x70/+0x170/+0x270 with the next
// non-Ult member at +0x370, and by pool/runtime at 0x236e30/0x236f30. CONFIDENCE: HIGH.
//
// prosper stores its state host-side and puts only {magic, id} in the guest blob. That is safe
// because the guest never reads inside one: an exhaustive scan of every RIP-relative reference in the
// whole executable segment finds 16 static Ult objects and ZERO references to any of them at a
// non-zero offset — every reference is a `lea` of the base feeding an Ult call. The heap-embedded
// members at 0xd8a0+0x70/+0x170/+0x270 are likewise only ever passed as pointers; the constructor
// initialises nothing in [0x70,0x370). CONFIDENCE: HIGH.
//
// Zeroed BSS therefore reads as "not created" and is refused loudly rather than trusted, and Destroy
// clears the magic so a stale object is detected instead of dereferenced.
//
// ============================================================================================
// SCHEDULING: ONE HOST THREAD PER ULTHREAD — A DELIBERATE, SELF-DETECTING DEVIATION
// ============================================================================================
// Hardware multiplexes up to numMaxUlthread ulthreads cooperatively onto numWorkerThread OS threads.
// prosper runs one real guest thread per ulthread. Two facts make that the right mapping here:
//
//  1. Ult mutexes are locked by ORDINARY (non-ulthread) threads. In the Phase 1 measurement every
//     _sceUltUlthreadCreate was refused, so no ulthread existed — yet sceUltMutexLock was still
//     called 1,005,742 times in 118 s. Whatever prosper does must give correct mutual exclusion to
//     plain guest pthreads, not only to scheduler-managed ulthreads.
//  2. The guest must already be preemption-safe across ulthreads, because 3 workers already preempt
//     each other on hardware, and the engine guards its state with Ult mutexes at 23 lock / 24 unlock
//     / 11 mutex-create / 4 condvar-create sites. CONFIDENCE: HIGH.
//
// The deviation that remains is CONCURRENCY WIDTH: hardware runs at most numWorkerThread ulthreads at
// once, prosper can run all of them. That is not trusted, it is MEASURED — the runtime keeps a
// high-water mark of simultaneously-running ulthreads and logs loudly the first time it exceeds
// numWorkerThread, so an invisible semantic difference becomes a line in the log. It is deliberately
// not capped: capping would trade a visible deviation for a hidden scheduling stall.
//
// ============================================================================================
// BLOCKING IS NOW REAL, SO IT IS WATCHDOGGED
// ============================================================================================
// Before this file, a million lock calls per boot were no-ops. They now block. A cross-thread
// ordering hazard would therefore present as a HANG, which is the worst failure mode to debug — it
// looks exactly like slow asset loading. Every blocking wait is bounded by a watchdog that logs the
// object, the holder and the waiter once, then keeps waiting. The hang becomes a diagnosable event.
//
// Escape hatches, both off by default:
//   PROSPER_ULT_LEGACY_ENOSYS=1   restore Phase 1 wholesale (every entry point refuses).
//   PROSPER_ULT_RETURN_SUCCESS=1  restore the pre-#1603 fake success (return 0, implement nothing).
//   PROSPER_ULT_BLOCK_WARN_MS=N   watchdog threshold for a blocked mutex lock (default 5000).
#include "dispatch.hpp"
#include "sce_errno.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <sched.h>
#endif
#include <time.h>

// guest_readable is a core (non-Vulkan) symbol always linked into prosper_core; declare it rather
// than pulling the Vulkan-gated gpu headers into the HLE. It is a probe that cannot fault, used here
// only to reject a wild object pointer before prosper writes a header into it.
namespace prosper::gpu { bool guest_readable(uint64_t addr, uint32_t bytes); }

namespace prosper {
namespace {

// ---------------------------------------------------------------------------------------------
// Return values.
//
// prosper has NO primary evidence for libSceUlt's own SCE_ULT_ERROR_* numbering: no live capture of
// a real console's return, no published constant, and the firmware symbol database gives names and
// NIDs only, never values. Rather than fabricate a libSceUlt-shaped constant, these use the
// documented libkernel `0x80020000 | freebsd_errno` encoding through the shared helper in
// sce_errno.hpp (#1612), which is the single place that owns the FreeBSD-vs-host numbering — the
// distinction that makes EDEADLK 11 here and 35 on Linux, exactly transposed with EAGAIN.
//
// CONFIDENCE: HIGH that non-zero is correct (0 is SCE_OK for this family);
// CONFIDENCE: LOW that any specific value matches the real library.
//
// That LOW is also cheap, and provably so: Earthion never branches on a libSceUlt return. Scanning
// the whole executable segment finds no comparison against any Ult-family immediate, and the five
// cmp/test instructions that happen to follow an Ult call all compare unrelated memory operands
// ([rbp-0x30], [rbp-0xc], and two rip-relative byte flags), never eax. The exact constant is
// therefore unobservable to the only title that imports the library.
// ---------------------------------------------------------------------------------------------
constexpr uint64_t kUltOk             = 0ull;
constexpr uint64_t kUltErrPerm        = hle::kSceKernelErrorEPERM;    // unlock/wait by a non-owner
constexpr uint64_t kUltErrSrch        = hle::kSceKernelErrorESRCH;    // stale / destroyed / uncreated
constexpr uint64_t kUltErrDeadlk      = hle::kSceKernelErrorEDEADLK;  // self-relock of a non-recursive mutex
constexpr uint64_t kUltErrNoMem       = hle::kSceKernelErrorENOMEM;
constexpr uint64_t kUltErrInval       = hle::kSceKernelErrorEINVAL;
constexpr uint64_t kUltErrAgain       = hle::kSceKernelErrorEAGAIN;   // runtime is at numMaxUlthread
constexpr uint64_t kUltNotImplemented = hle::kSceKernelErrorENOSYS;   // the legacy/Phase 1 policy

// A size-returning contract has no error channel (#1618): whatever these return is read as a byte
// count and handed to malloc. They must therefore never see an error sentinel — the value is always
// prosper's own real requirement, and the matching Create recomputes it from the same arguments and
// genuinely consumes the buffer.
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// Registered surface. Driven by what titles ACTUALLY import: the PS5 3.20 firmware database exports
// 64 sceUlt* symbols, and a byte-scan of every dump plus self_dump --symbols agree that exactly these
// 16 are imported, by exactly one title. Any other Ult NID a future title imports still falls to the
// generic unimplemented-import path, which logs it — add it here when that happens.
// ---------------------------------------------------------------------------------------------
enum class UltRet : uint8_t { Status, Size };   // #1618: what the contract's return value MEANS
struct UltEntry { const char* nid; const char* name; UltRet ret; };
constexpr UltEntry kUlt[] = {
    { "jw9FkZBXo-g", "_sceUltUlthreadRuntimeCreate",                  UltRet::Status },
    { "grs2pbc2awM", "sceUltUlthreadRuntimeGetWorkAreaSize",          UltRet::Size   },
    { "znI3q8S7KQ4", "_sceUltUlthreadCreate",                         UltRet::Status },
    { "gCeAI57LGgI", "sceUltUlthreadJoin",                            UltRet::Status },
    { "mmt8Sa6tL6c", "_sceUltMutexCreate",                            UltRet::Status },
    { "8hEGkR1pfr8", "sceUltMutexLock",                               UltRet::Status },
    { "h0XebKiMBtk", "sceUltMutexUnlock",                             UltRet::Status },
    { "jW+HnafeS3Y", "sceUltMutexDestroy",                            UltRet::Status },
    { "jnKaHGkrxZ4", "_sceUltConditionVariableCreate",                UltRet::Status },
    { "5xGAHCxA8M0", "sceUltConditionVariableWait",                   UltRet::Status },
    { "JTw1cAVkuc0", "sceUltConditionVariableSignal",                 UltRet::Status },
    { "xrmmI832R4U", "sceUltConditionVariableDestroy",                UltRet::Status },
    { "YiHujOG9vXY", "_sceUltWaitingQueueResourcePoolCreate",         UltRet::Status },
    { "WIWV1Qd7PFU", "sceUltWaitingQueueResourcePoolGetWorkAreaSize", UltRet::Size   },
    { "hZIg1EWGsHM", "sceUltInitialize",                              UltRet::Status },
    { "d-kSG2fLrvI", "sceUltFinalize",                                UltRet::Status },
};
constexpr size_t kUltCount = sizeof(kUlt) / sizeof(kUlt[0]);
constexpr size_t kIdxRuntimeCreate = 0,  kIdxRuntimeSize = 1,  kIdxUlthreadCreate = 2,
                 kIdxUlthreadJoin  = 3,  kIdxMutexCreate = 4,  kIdxMutexLock      = 5,
                 kIdxMutexUnlock   = 6,  kIdxMutexDestroy = 7, kIdxCondCreate     = 8,
                 kIdxCondWait      = 9,  kIdxCondSignal  = 10, kIdxCondDestroy    = 11,
                 kIdxPoolCreate    = 12, kIdxPoolSize    = 13, kIdxInitialize     = 14,
                 kIdxFinalize      = 15;

std::atomic<uint64_t> g_calls[kUltCount];
std::atomic<uint64_t> g_next_report[kUltCount];   // 0 = "report the next call"
std::atomic<bool>     g_banner_printed{false};

// PROSPER_ULT_RETURN_SUCCESS=1 restores the pre-#1603 `return 0`; PROSPER_ULT_LEGACY_ENOSYS=1
// restores Phase 1's refusal. Both are OFF by default and neither silences the counting.
std::atomic<bool> g_return_success{false};
std::atomic<bool> g_legacy_enosys{false};
bool env_flag(const char* name) {
    const char* e = getenv(name);
    return e && *e && std::strcmp(e, "0") != 0;
}
uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* e = getenv(name);
    if (!e || !*e) return fallback;
    char* end = nullptr;
    const unsigned long long v = strtoull(e, &end, 0);
    return (end && *end == 0) ? (uint64_t)v : fallback;
}

void log_line(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void log_line(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[prosper] [ult] %s\n", buf);
}

// ---------------------------------------------------------------------------------------------
// Host-side object registry.
//
// The guest blob holds {magic, id}; everything else lives here. Slots are never freed, only marked
// dead and reused, and the id carries a generation, so a stale or destroyed handle is DETECTED
// rather than dereferenced. Objects are heap-allocated individually, so a raw pointer taken under
// the shared lock stays valid after the lock is dropped even if the slot vector reallocates.
// ---------------------------------------------------------------------------------------------
enum class UltType : uint32_t { Runtime = 1, Pool = 2, Mutex = 3, Cond = 4, Ulthread = 5 };

const char* type_name(UltType t) {
    switch (t) {
        case UltType::Runtime:  return "UlthreadRuntime";
        case UltType::Pool:     return "WaitingQueueResourcePool";
        case UltType::Mutex:    return "Mutex";
        case UltType::Cond:     return "ConditionVariable";
        case UltType::Ulthread: return "Ulthread";
    }
    return "?";
}

// Distinct per type so a mutex handed to a condvar entry point is caught rather than reinterpreted.
uint64_t magic_for(UltType t) {
    return 0x554C54ull << 40 | (uint64_t)t << 32 | 0x50525350ull;   // "ULT" | type | "PRSP"
}

struct UltObject {
    UltType  type;
    uint32_t generation = 0;
    uint32_t slot       = 0;
    std::atomic<bool> alive{false};
    std::string name;
    uint64_t guest_addr = 0;
    uint32_t api_version = 0;

    // Mutex / ConditionVariable.
    pthread_mutex_t  mtx = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t   cond = PTHREAD_COND_INITIALIZER;
    bool             mtx_valid = false;
    bool             cond_valid = false;
    std::atomic<uint64_t> owner{0};          // host thread holding this mutex (0 = unheld)
    uint64_t         pool_id = 0;            // Mutex -> its waiting-queue resource pool
    uint64_t         mutex_id = 0;           // Cond  -> the mutex bound at create
    std::atomic<bool> warned_block{false};   // watchdog fires at most once per object
    std::atomic<bool> warned_deadlock{false};
    std::atomic<uint64_t> lock_calls{0}, contended{0};

    // Runtime / Pool: the guest-supplied work area prosper actually consumes.
    uint64_t work_area = 0, work_bytes = 0;
    uint32_t num_max_ulthread = 0, num_worker_thread = 0;
    uint32_t num_threads = 0, num_sync_objects = 0;
    std::atomic<uint32_t> live_ulthreads{0}, running_ulthreads{0}, high_water{0};
    std::atomic<bool> warned_concurrency{false};
    std::atomic<uint32_t> bound_sync_objects{0};
    std::atomic<bool> warned_pool_capacity{false};

    // ConditionVariable wake accounting (see the condvar block for why these are lock-free).
    std::atomic<uint64_t> wait_seq{0}, wake_seq{0};
    std::atomic<bool> destroying{false};

    // Ulthread.
    uint64_t runtime_id = 0;
    uint64_t context = 0, context_size = 0;
    uint64_t host_thread = 0;
    std::atomic<bool> finished{false}, joined{false};
    std::atomic<uint32_t> exit_status{0};
    std::atomic<uint64_t> stack_used{0};
    std::atomic<bool> stack_exact{false};   // false => stack_used is a lower bound (probe < context)
};

std::shared_mutex                          g_reg_lock;
std::vector<std::unique_ptr<UltObject>>    g_objects;

constexpr uint64_t kSlotMask = 0x00000000FFFFFFFFull;
uint64_t make_id(uint32_t slot, uint32_t generation) {
    return ((uint64_t)generation << 32) | slot;
}

UltObject* object_from_id(uint64_t id, UltType expect) {
    const uint32_t slot = (uint32_t)(id & kSlotMask);
    const uint32_t gen  = (uint32_t)(id >> 32);
    UltObject* o = nullptr;
    {
        std::shared_lock<std::shared_mutex> lk(g_reg_lock);
        if (slot >= g_objects.size()) return nullptr;
        o = g_objects[slot].get();
    }
    if (!o || o->generation != gen || o->type != expect) return nullptr;
    if (!o->alive.load(std::memory_order_acquire)) return nullptr;
    return o;
}

UltObject* allocate_object(UltType type) {
    std::unique_lock<std::shared_mutex> lk(g_reg_lock);
    for (auto& up : g_objects) {
        UltObject* o = up.get();
        if (o->type == type && !o->alive.load(std::memory_order_relaxed)) {
            ++o->generation;                 // stale ids referring to the previous incarnation now miss
            o->name.clear();
            o->owner.store(0, std::memory_order_relaxed);
            o->warned_block.store(false, std::memory_order_relaxed);
            o->warned_deadlock.store(false, std::memory_order_relaxed);
            o->lock_calls.store(0, std::memory_order_relaxed);
            o->contended.store(0, std::memory_order_relaxed);
            o->bound_sync_objects.store(0, std::memory_order_relaxed);
            return o;
        }
    }
    auto up = std::make_unique<UltObject>();
    UltObject* o = up.get();
    o->type = type;
    o->slot = (uint32_t)g_objects.size();
    o->generation = 1;
    g_objects.push_back(std::move(up));
    return o;
}

// ---------------------------------------------------------------------------------------------
// The guest object header. 16 bytes at offset 0 of the caller's 256-byte blob.
// ---------------------------------------------------------------------------------------------
struct GuestUltHeader { uint64_t magic; uint64_t id; };

bool guest_object_writable(uint64_t addr) {
    // Reject null and obviously-unaligned pointers outright; guest_readable is a probe that cannot
    // fault and rejects an unmapped address. Ult objects always live in the guest's RW data/BSS.
    if (!addr || (addr & 7u)) return false;
    return gpu::guest_readable(addr, (uint32_t)sizeof(GuestUltHeader));
}

// Publish {magic, id} into the guest blob. Returns false if the pointer is unusable.
bool publish_object(uint64_t guest_addr, UltType type, uint64_t id) {
    if (!guest_object_writable(guest_addr)) return false;
    auto* h = (GuestUltHeader*)(uintptr_t)guest_addr;
    h->id = id;
    // Publish the id BEFORE the magic: another thread that sees the magic must already see the id.
    __atomic_store_n(&h->magic, magic_for(type), __ATOMIC_RELEASE);
    return true;
}

void unpublish_object(uint64_t guest_addr) {
    if (!guest_object_writable(guest_addr)) return;
    auto* h = (GuestUltHeader*)(uintptr_t)guest_addr;
    __atomic_store_n(&h->magic, 0ull, __ATOMIC_RELEASE);
    h->id = 0;
}

// Resolve a guest object pointer to prosper's object, or nullptr with a loud, deduplicated report.
UltObject* resolve(uint64_t guest_addr, UltType type, const char* fn) {
    if (!guest_object_writable(guest_addr)) {
        log_line("%s: unusable object pointer 0x%llx (null, misaligned, or unmapped)", fn,
                 (unsigned long long)guest_addr);
        return nullptr;
    }
    auto* h = (GuestUltHeader*)(uintptr_t)guest_addr;
    const uint64_t magic = __atomic_load_n(&h->magic, __ATOMIC_ACQUIRE);
    if (magic != magic_for(type)) {
        static std::atomic<uint64_t> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 16)
            log_line("%s: object 0x%llx is not a live %s (magic 0x%llx) — never created, already "
                     "destroyed, or the wrong object type",
                     fn, (unsigned long long)guest_addr, type_name(type), (unsigned long long)magic);
        return nullptr;
    }
    UltObject* o = object_from_id(h->id, type);
    if (!o) {
        log_line("%s: object 0x%llx carries a stale %s id 0x%llx", fn,
                 (unsigned long long)guest_addr, type_name(type), (unsigned long long)h->id);
        return nullptr;
    }
    return o;
}

// ---------------------------------------------------------------------------------------------
// Work areas. `*GetWorkAreaSize` returns prosper's OWN requirement, and the matching Create
// recomputes the identical size from the identical arguments the guest passes it again, so the
// buffer prosper asked for is exactly the buffer prosper consumes.
//
// Keeping the registry here rather than host-side is what makes the returned size honest instead of
// decorative. The trade is that it is guest memory: a title that frees the work area while the object
// lives would leave prosper reading freed storage. That is DETECTED, not trusted — every access
// re-validates the header magic, so a freed or reused buffer is reported rather than believed.
// (Earthion never frees either work area; it does not even retain the pointer.)
// ---------------------------------------------------------------------------------------------
constexpr uint32_t kWorkAreaVersion = 1;
constexpr uint64_t kWorkMagicRuntime = 0x50525350'52544141ull;   // "PRSP" "RTAA"
constexpr uint64_t kWorkMagicPool    = 0x50525350'504F4F4Cull;   // "PRSP" "POOL"

struct RuntimeSlot {          // one per numMaxUlthread
    uint64_t ulthread_id;     // 0 = free
    uint64_t host_thread;
    uint64_t context;
    uint64_t context_size;
};
struct RuntimeWorkArea {
    uint64_t magic;
    uint32_t version;
    uint32_t num_max_ulthread;
    uint32_t num_worker_thread;
    uint32_t reserved;
    uint64_t bytes;
    // RuntimeSlot[num_max_ulthread] follows
};
struct PoolSlot { uint64_t sync_object_id; };   // 0 = free
struct PoolWorkArea {
    uint64_t magic;
    uint32_t version;
    uint32_t num_threads;
    uint32_t num_sync_objects;
    uint32_t reserved;
    uint64_t bytes;
    // PoolSlot[num_sync_objects] follows
};

// Bound the guest's own numbers before multiplying: a garbled argument must not become a huge size.
// Earthion asks for 16 ulthreads / 3 workers and 16 threads / 16 sync objects.
constexpr uint32_t kMaxReasonableObjects = 4096;

uint64_t runtime_work_area_size(uint32_t num_max_ulthread) {
    if (num_max_ulthread > kMaxReasonableObjects) num_max_ulthread = kMaxReasonableObjects;
    return sizeof(RuntimeWorkArea) + (uint64_t)num_max_ulthread * sizeof(RuntimeSlot);
}
uint64_t pool_work_area_size(uint32_t num_sync_objects) {
    if (num_sync_objects > kMaxReasonableObjects) num_sync_objects = kMaxReasonableObjects;
    return sizeof(PoolWorkArea) + (uint64_t)num_sync_objects * sizeof(PoolSlot);
}

// ---------------------------------------------------------------------------------------------
// Blocking watchdog. Real blocking replaced a million per-boot no-ops, so an ordering hazard that
// used to be invisible would now be a hang. Every blocking wait is bounded so it can be reported.
// ---------------------------------------------------------------------------------------------
uint64_t block_warn_ms() {
    static const uint64_t v = env_u64("PROSPER_ULT_BLOCK_WARN_MS", 5000);
    return v;
}
// A condition variable legitimately blocks for a long time (an idle worker waiting for jobs), so its
// threshold is far higher and its wording is informational — a loud "hang" line for an idle worker
// would be a false alarm that trains readers to ignore the real one.
uint64_t cond_note_ms() {
    static const uint64_t v = env_u64("PROSPER_ULT_COND_NOTE_MS", 30000);
    return v;
}

// Thread identity for mutex ownership. Deliberately NOT `pthread_self()` cast to an integer:
// pthread_t is opaque by specification — which is exactly why pthread_equal() exists — and on an
// implementation where it is a struct pointer rather than an id, an integer comparison would be
// UNRELIABLE rather than loudly broken. Ownership is what makes the condvar's release/reacquire and
// the self-deadlock diagnosis correct, so it must not rest on a representation assumption.
//
// Each thread instead takes a unique non-zero token from a monotonic counter on first use. That is
// portable by construction, exact, and cheaper on a path the guest hits a million times per boot
// than a pthread_self() call. The token is an identity, never a handle: pthread_join still needs a
// real pthread_t, which UltObject::host_thread carries under the project-wide convention that
// k_pthread_create/k_pthread_join already use.
std::atomic<uint64_t> g_next_thread_token{1};
thread_local uint64_t t_thread_token = 0;
uint64_t self_thread() {
    if (!t_thread_token) t_thread_token = g_next_thread_token.fetch_add(1, std::memory_order_relaxed);
    return t_thread_token;
}

void abstime_in_ms(struct timespec& ts, uint64_t ms) {
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec  += (time_t)(ms / 1000);
    ts.tv_nsec += (long)((ms % 1000) * 1000000ull);
    if (ts.tv_nsec >= 1000000000L) { ts.tv_nsec -= 1000000000L; ts.tv_sec += 1; }
}

// ---------------------------------------------------------------------------------------------
// Library state.
// ---------------------------------------------------------------------------------------------
std::atomic<bool>     g_initialized{false};
std::atomic<uint64_t> g_init_calls{0};
// Last joined ulthread's measured guest-stack high-water (amendment 6). Diagnostics + tests.
std::atomic<uint64_t> g_last_join_stack_used{0}, g_last_join_context_size{0};

void print_banner() {
    if (g_banner_printed.exchange(true)) return;
    fprintf(stderr,
        "[prosper] ==== libSceUlt (user-level threading) ========================================\n"
        "[prosper]   prosper implements the ulthread runtime, waiting-queue resource pool, mutexes\n"
        "[prosper]   and condition variables this title uses. Ulthreads run one-to-one on real guest\n"
        "[prosper]   threads rather than multiplexed onto the requested worker count; the runtime\n"
        "[prosper]   reports if simultaneous ulthreads ever exceed numWorkerThread.\n"
        "[prosper]   Blocking waits are watchdogged (PROSPER_ULT_BLOCK_WARN_MS, default 5000).\n"
        "[prosper] ==============================================================================\n");
}

// Count the call and, at 1 / 10 / 100 / ..., emit a volume line. Bounded (a 1e6-call hot path costs
// 7 lines) but never silent: the generic path's first-seen dedup reported ONE line for 1,005,742
// sceUltMutexLock calls, which is the visibility failure this whole surface exists to fix.
void count_call(size_t index) {
    const uint64_t n = g_calls[index].fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t due = g_next_report[index].load(std::memory_order_relaxed);
    if (n < due) return;
    uint64_t next = due ? due * 10 : 10;
    while (next <= n && next <= (uint64_t)1e18) next *= 10;
    g_next_report[index].store(next, std::memory_order_relaxed);
    print_banner();
    log_line("%s (%s) call #%llu", kUlt[index].name, kUlt[index].nid, (unsigned long long)n);
}

// The policy gate every entry point runs first. Returns true when the caller should execute the real
// implementation; otherwise *out is the policy's return value.
bool implement(size_t index, uint64_t* out) {
    count_call(index);
    if (kUlt[index].ret == UltRet::Size) {
        // #1618: a size contract has no error channel, so neither escape hatch may put a sentinel
        // there. Under a legacy policy prosper consumes no work area, so 0 is both honest and safe.
        if (g_legacy_enosys.load(std::memory_order_relaxed) ||
            g_return_success.load(std::memory_order_relaxed)) { *out = 0; return false; }
        return true;
    }
    if (g_legacy_enosys.load(std::memory_order_relaxed)) { *out = kUltNotImplemented; return false; }
    if (g_return_success.load(std::memory_order_relaxed)) { *out = kUltOk; return false; }
    return true;
}

std::string guest_string(uint64_t addr) {
    if (!addr || !gpu::guest_readable(addr, 1)) return std::string();
    const char* s = (const char*)(uintptr_t)addr;
    size_t n = 0;
    while (n < 64 && gpu::guest_readable(addr + n, 1) && s[n]) ++n;
    return std::string(s, n);
}

// =============================================================================================
// Library lifecycle
// =============================================================================================
PROSPER_SYSV_ABI uint64_t ult_initialize(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxInitialize, &out)) return out;
    g_init_calls.fetch_add(1, std::memory_order_relaxed);
    g_initialized.store(true, std::memory_order_release);
    print_banner();
    return kUltOk;
}

PROSPER_SYSV_ABI uint64_t ult_finalize(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxFinalize, &out)) return out;
    // Finalize does NOT tear prosper's objects down. The guest may still hold live mutex/condvar
    // pointers (its own Destroy calls own that), and destroying a mutex another thread is blocked on
    // would turn an orderly shutdown into undefined behaviour. Recording the transition is enough.
    g_initialized.store(false, std::memory_order_release);
    return kUltOk;
}

// prosper's implementation has no global initialisation requirement, so an object created before
// sceUltInitialize cannot fail for a reason prosper actually has. Report the ordering and continue;
// refusing would invent a failure the library's real behaviour has not been shown to have.
// CONFIDENCE: MED (Earthion always initialises first, so the ordering rule itself is untested here).
void note_uninitialised(const char* fn) {
    if (g_initialized.load(std::memory_order_acquire)) return;
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true))
        log_line("%s called before sceUltInitialize — prosper needs no global init and continues, "
                 "but the guest's ordering is unusual and worth knowing", fn);
}

// =============================================================================================
// Waiting-queue resource pool
// =============================================================================================
PROSPER_SYSV_ABI uint64_t ult_pool_work_area_size(uint64_t a0, uint64_t a1, uint64_t, uint64_t,
                                                  uint64_t, uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxPoolSize, &out)) return out;
    const uint32_t num_threads = (uint32_t)a0, num_sync_objects = (uint32_t)a1;
    const uint64_t bytes = pool_work_area_size(num_sync_objects);
    log_line("sceUltWaitingQueueResourcePoolGetWorkAreaSize(numThreads=%u, numSyncObjects=%u) -> %llu "
             "bytes (prosper's own requirement; the matching Create consumes exactly this buffer)",
             num_threads, num_sync_objects, (unsigned long long)bytes);
    return bytes;
}

// _sceUltWaitingQueueResourcePoolCreate(pool, name, numThreads, numSyncObjects, workArea, opt, ver).
// 7 args; `ver` arrives on the stack and the import bridge forwards it as a6.
PROSPER_SYSV_ABI uint64_t ult_pool_create(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                          uint64_t a4, uint64_t a5, uint64_t a6) {
    uint64_t out = 0;
    if (!implement(kIdxPoolCreate, &out)) return out;
    note_uninitialised("_sceUltWaitingQueueResourcePoolCreate");
    const uint32_t num_threads = (uint32_t)a2, num_sync_objects = (uint32_t)a3;
    const uint64_t need = pool_work_area_size(num_sync_objects);

    if (!guest_object_writable(a0)) {
        log_line("_sceUltWaitingQueueResourcePoolCreate: unusable pool pointer 0x%llx",
                 (unsigned long long)a0);
        return kUltErrInval;
    }
    if (!a4 || !gpu::guest_readable(a4, (uint32_t)std::min<uint64_t>(need, 4096))) {
        log_line("_sceUltWaitingQueueResourcePoolCreate: workArea 0x%llx is unusable for the %llu "
                 "bytes sceUltWaitingQueueResourcePoolGetWorkAreaSize(%u, %u) asked for",
                 (unsigned long long)a4, (unsigned long long)need, num_threads, num_sync_objects);
        return kUltErrInval;
    }

    auto* wa = (PoolWorkArea*)(uintptr_t)a4;
    wa->magic = kWorkMagicPool;
    wa->version = kWorkAreaVersion;
    wa->num_threads = num_threads;
    wa->num_sync_objects = num_sync_objects;
    wa->reserved = 0;
    wa->bytes = need;
    auto* slots = (PoolSlot*)(uintptr_t)(a4 + sizeof(PoolWorkArea));
    const uint32_t capped = num_sync_objects > kMaxReasonableObjects ? kMaxReasonableObjects
                                                                     : num_sync_objects;
    for (uint32_t i = 0; i < capped; ++i) slots[i].sync_object_id = 0;

    UltObject* o = allocate_object(UltType::Pool);
    o->name = guest_string(a1);
    o->guest_addr = a0;
    o->api_version = (uint32_t)a6;
    o->num_threads = num_threads;
    o->num_sync_objects = num_sync_objects;
    o->work_area = a4;
    o->work_bytes = need;
    o->alive.store(true, std::memory_order_release);
    const uint64_t id = make_id(o->slot, o->generation);
    if (!publish_object(a0, UltType::Pool, id)) {
        o->alive.store(false, std::memory_order_release);
        return kUltErrInval;
    }
    log_line("_sceUltWaitingQueueResourcePoolCreate(\"%s\") pool=0x%llx numThreads=%u "
             "numSyncObjects=%u workArea=0x%llx(%llu B) apiVersion=0x%x -> ok",
             o->name.c_str(), (unsigned long long)a0, num_threads, num_sync_objects,
             (unsigned long long)a4, (unsigned long long)need, o->api_version);
    return kUltOk;
}

// =============================================================================================
// Ulthread runtime
// =============================================================================================
PROSPER_SYSV_ABI uint64_t ult_runtime_work_area_size(uint64_t a0, uint64_t a1, uint64_t, uint64_t,
                                                     uint64_t, uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxRuntimeSize, &out)) return out;
    const uint32_t num_max = (uint32_t)a0, num_worker = (uint32_t)a1;
    const uint64_t bytes = runtime_work_area_size(num_max);
    log_line("sceUltUlthreadRuntimeGetWorkAreaSize(numMaxUlthread=%u, numWorkerThread=%u) -> %llu "
             "bytes (prosper's own requirement; the matching Create consumes exactly this buffer)",
             num_max, num_worker, (unsigned long long)bytes);
    return bytes;
}

// _sceUltUlthreadRuntimeCreate(runtime, name, numMaxUlthread, numWorkerThread, workArea, opt, ver).
PROSPER_SYSV_ABI uint64_t ult_runtime_create(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                             uint64_t a4, uint64_t a5, uint64_t a6) {
    uint64_t out = 0;
    if (!implement(kIdxRuntimeCreate, &out)) return out;
    note_uninitialised("_sceUltUlthreadRuntimeCreate");
    const uint32_t num_max = (uint32_t)a2, num_worker = (uint32_t)a3;
    const uint64_t need = runtime_work_area_size(num_max);

    if (!guest_object_writable(a0)) {
        log_line("_sceUltUlthreadRuntimeCreate: unusable runtime pointer 0x%llx",
                 (unsigned long long)a0);
        return kUltErrInval;
    }
    if (!num_max) {
        log_line("_sceUltUlthreadRuntimeCreate: numMaxUlthread is 0 — no ulthread could ever be "
                 "created in this runtime");
        return kUltErrInval;
    }
    if (!a4 || !gpu::guest_readable(a4, (uint32_t)std::min<uint64_t>(need, 4096))) {
        log_line("_sceUltUlthreadRuntimeCreate: workArea 0x%llx is unusable for the %llu bytes "
                 "sceUltUlthreadRuntimeGetWorkAreaSize(%u, %u) asked for",
                 (unsigned long long)a4, (unsigned long long)need, num_max, num_worker);
        return kUltErrInval;
    }

    auto* wa = (RuntimeWorkArea*)(uintptr_t)a4;
    wa->magic = kWorkMagicRuntime;
    wa->version = kWorkAreaVersion;
    wa->num_max_ulthread = num_max;
    wa->num_worker_thread = num_worker;
    wa->reserved = 0;
    wa->bytes = need;
    auto* slots = (RuntimeSlot*)(uintptr_t)(a4 + sizeof(RuntimeWorkArea));
    const uint32_t capped = num_max > kMaxReasonableObjects ? kMaxReasonableObjects : num_max;
    for (uint32_t i = 0; i < capped; ++i) slots[i] = RuntimeSlot{0, 0, 0, 0};

    UltObject* o = allocate_object(UltType::Runtime);
    o->name = guest_string(a1);
    o->guest_addr = a0;
    o->api_version = (uint32_t)a6;
    o->num_max_ulthread = num_max;
    o->num_worker_thread = num_worker;
    o->work_area = a4;
    o->work_bytes = need;
    o->live_ulthreads.store(0, std::memory_order_relaxed);
    o->running_ulthreads.store(0, std::memory_order_relaxed);
    o->high_water.store(0, std::memory_order_relaxed);
    o->warned_concurrency.store(false, std::memory_order_relaxed);
    o->alive.store(true, std::memory_order_release);
    const uint64_t id = make_id(o->slot, o->generation);
    if (!publish_object(a0, UltType::Runtime, id)) {
        o->alive.store(false, std::memory_order_release);
        return kUltErrInval;
    }
    log_line("_sceUltUlthreadRuntimeCreate(\"%s\") runtime=0x%llx numMaxUlthread=%u "
             "numWorkerThread=%u workArea=0x%llx(%llu B) apiVersion=0x%x -> ok",
             o->name.c_str(), (unsigned long long)a0, num_max, num_worker,
             (unsigned long long)a4, (unsigned long long)need, o->api_version);
    return kUltOk;
}

// =============================================================================================
// Mutex
//
// The dominant call by five orders of magnitude: 1,005,742 lock/unlock pairs in a 118 s boot. The
// implementation is an errorcheck pthread mutex plus explicit owner tracking, because the two failure
// modes that matter here are both undefined behaviour on a plain mutex and must instead be reported:
//   * self-relock of a non-recursive mutex (Earthion passes optParam=NULL, i.e. non-recursive), and
//   * unlock by a thread that does not hold it.
// A returned error is diagnosable; a hang is not.
// =============================================================================================
PROSPER_SYSV_ABI uint64_t ult_mutex_create(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                           uint64_t a4, uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxMutexCreate, &out)) return out;
    note_uninitialised("_sceUltMutexCreate");
    if (!guest_object_writable(a0)) {
        log_line("_sceUltMutexCreate: unusable mutex pointer 0x%llx", (unsigned long long)a0);
        return kUltErrInval;
    }
    // A mutex is allocated FROM a waiting-queue resource pool: the pool is the third argument at
    // every call site. Validating it is only possible because the pool family lands in the same
    // change — a mutex that cannot check its own pool argument would accept anything.
    UltObject* pool = resolve(a2, UltType::Pool, "_sceUltMutexCreate(pool)");
    if (!pool) return kUltErrInval;

    UltObject* o = allocate_object(UltType::Mutex);
    if (!o->mtx_valid) {
        pthread_mutexattr_t at;
        pthread_mutexattr_init(&at);
        pthread_mutexattr_settype(&at, PTHREAD_MUTEX_ERRORCHECK);
        const int rc = pthread_mutex_init(&o->mtx, &at);
        pthread_mutexattr_destroy(&at);
        if (rc) {
            log_line("_sceUltMutexCreate: pthread_mutex_init failed (%d)", rc);
            return kUltErrNoMem;
        }
        o->mtx_valid = true;
    }
    o->name = guest_string(a1);
    o->guest_addr = a0;
    o->api_version = (uint32_t)a4;
    o->pool_id = make_id(pool->slot, pool->generation);
    o->owner.store(0, std::memory_order_relaxed);
    o->alive.store(true, std::memory_order_release);
    const uint64_t id = make_id(o->slot, o->generation);
    if (!publish_object(a0, UltType::Mutex, id)) {
        o->alive.store(false, std::memory_order_release);
        return kUltErrInval;
    }

    // numSyncObjects is the pool's stated capacity. prosper reports crossing it but does NOT refuse:
    // the parameter's exact meaning is not proven from title evidence, and refusing a legitimate
    // create on an unproven reading would be worse than a title running with a pool it undersized.
    // CONFIDENCE: MED (Earthion asks for 16 and binds 11-15, which fits the reading but does not
    // prove it).
    const uint32_t bound = pool->bound_sync_objects.fetch_add(1, std::memory_order_relaxed) + 1;
    if (bound > pool->num_sync_objects && !pool->warned_pool_capacity.exchange(true))
        log_line("pool \"%s\" now has %u sync objects bound but was created for numSyncObjects=%u — "
                 "prosper does not enforce this cap; reporting it so an undersized pool is visible",
                 pool->name.c_str(), bound, pool->num_sync_objects);

    log_line("_sceUltMutexCreate(\"%s\") mutex=0x%llx pool=0x%llx apiVersion=0x%x -> ok",
             o->name.c_str(), (unsigned long long)a0, (unsigned long long)a2, o->api_version);
    return kUltOk;
}

uint64_t mutex_lock_impl(UltObject* o, const char* fn) {
    const uint64_t me = self_thread();
    o->lock_calls.fetch_add(1, std::memory_order_relaxed);

    // Self-relock of a non-recursive mutex. On hardware this deadlocks or fails; prosper reports it
    // and fails, because a returned error is diagnosable and a hang is not. Detected by an explicit
    // owner compare rather than by relying on the errorcheck lock's EDEADLK, so the diagnosis is
    // exact and costs one relaxed atomic load on the fast path.
    if (o->owner.load(std::memory_order_relaxed) == me) {
        if (!o->warned_deadlock.exchange(true))
            log_line("SELF-DEADLOCK: sceUltMutexLock on \"%s\" (0x%llx) by the thread that already "
                     "holds it. This mutex was created with optParam=NULL, i.e. NON-RECURSIVE. "
                     "prosper returns 0x%llx rather than blocking forever",
                     o->name.c_str(), (unsigned long long)o->guest_addr,
                     (unsigned long long)kUltErrDeadlk);
        return kUltErrDeadlk;
    }

    int rc = pthread_mutex_trylock(&o->mtx);
    if (rc == EBUSY) {
        o->contended.fetch_add(1, std::memory_order_relaxed);
        // Bounded wait so the watchdog can speak, then an unbounded one so semantics are unchanged.
        const uint64_t warn_ms = block_warn_ms();
#if defined(__APPLE__)
        // macOS has no pthread_mutex_timedlock; poll to the threshold, then block for real.
        const uint64_t deadline_polls = warn_ms;   // 1 ms per poll
        uint64_t polls = 0;
        while ((rc = pthread_mutex_trylock(&o->mtx)) == EBUSY && polls < deadline_polls) {
            struct timespec ms{0, 1000000};
            nanosleep(&ms, nullptr);
            ++polls;
        }
#else
        struct timespec deadline;
        abstime_in_ms(deadline, warn_ms);
        rc = pthread_mutex_timedlock(&o->mtx, &deadline);
#endif
        if (rc == ETIMEDOUT || rc == EBUSY) {
            if (!o->warned_block.exchange(true))
                log_line("BLOCKED >%llums: sceUltMutexLock on \"%s\" (0x%llx) — holder thread 0x%llx, "
                         "waiter thread token 0x%llx. Still waiting. If the title appears hung, this is "
                         "where. (PROSPER_ULT_BLOCK_WARN_MS)",
                         (unsigned long long)warn_ms, o->name.c_str(),
                         (unsigned long long)o->guest_addr,
                         (unsigned long long)o->owner.load(std::memory_order_relaxed),
                         (unsigned long long)me);
            rc = pthread_mutex_lock(&o->mtx);
        }
    }
    if (rc == EDEADLK) {   // errorcheck backstop; the owner compare above should have caught it
        if (!o->warned_deadlock.exchange(true))
            log_line("SELF-DEADLOCK (errorcheck): %s on \"%s\" (0x%llx)", fn, o->name.c_str(),
                     (unsigned long long)o->guest_addr);
        return kUltErrDeadlk;
    }
    if (rc != 0) {
        log_line("%s on \"%s\" failed: pthread rc=%d", fn, o->name.c_str(), rc);
        return kUltErrInval;
    }
    o->owner.store(me, std::memory_order_relaxed);
    return kUltOk;
}

PROSPER_SYSV_ABI uint64_t ult_mutex_lock(uint64_t a0, uint64_t, uint64_t, uint64_t, uint64_t,
                                         uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxMutexLock, &out)) return out;
    UltObject* o = resolve(a0, UltType::Mutex, "sceUltMutexLock");
    if (!o) return kUltErrSrch;
    return mutex_lock_impl(o, "sceUltMutexLock");
}

PROSPER_SYSV_ABI uint64_t ult_mutex_unlock(uint64_t a0, uint64_t, uint64_t, uint64_t, uint64_t,
                                           uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxMutexUnlock, &out)) return out;
    UltObject* o = resolve(a0, UltType::Mutex, "sceUltMutexUnlock");
    if (!o) return kUltErrSrch;
    const uint64_t me = self_thread();
    if (o->owner.load(std::memory_order_relaxed) != me) {
        static std::atomic<uint64_t> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 8)
            log_line("sceUltMutexUnlock on \"%s\" (0x%llx) by thread 0x%llx, which does not hold it "
                     "(holder 0x%llx) — refusing so the real owner's exclusion is not broken",
                     o->name.c_str(), (unsigned long long)o->guest_addr, (unsigned long long)me,
                     (unsigned long long)o->owner.load(std::memory_order_relaxed));
        return kUltErrPerm;
    }
    o->owner.store(0, std::memory_order_relaxed);
    const int rc = pthread_mutex_unlock(&o->mtx);
    if (rc) {
        log_line("sceUltMutexUnlock on \"%s\" failed: pthread rc=%d", o->name.c_str(), rc);
        return kUltErrPerm;
    }
    return kUltOk;
}

PROSPER_SYSV_ABI uint64_t ult_mutex_destroy(uint64_t a0, uint64_t, uint64_t, uint64_t, uint64_t,
                                            uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxMutexDestroy, &out)) return out;
    UltObject* o = resolve(a0, UltType::Mutex, "sceUltMutexDestroy");
    if (!o) return kUltErrSrch;
    if (o->owner.load(std::memory_order_relaxed) != 0)
        log_line("sceUltMutexDestroy on \"%s\" (0x%llx) while thread 0x%llx still holds it",
                 o->name.c_str(), (unsigned long long)o->guest_addr,
                 (unsigned long long)o->owner.load(std::memory_order_relaxed));
    // Unpublish first: a concurrent lock must miss rather than race the teardown. The pthread mutex
    // itself is retained for slot reuse (destroying one a thread is blocked on is undefined).
    unpublish_object(a0);
    o->alive.store(false, std::memory_order_release);
    if (UltObject* pool = object_from_id(o->pool_id, UltType::Pool))
        pool->bound_sync_objects.fetch_sub(1, std::memory_order_relaxed);
    return kUltOk;
}

// =============================================================================================
// Ulthreads
//
// One real guest thread per ulthread, spawned through the same trampoline scePthreadCreate uses so
// the guest %fs TCB, stack registration, sigaltstack and TLS-purge behaviour are the proven ones
// rather than a second copy. The guest's own `context`/`sizeContext` pair IS the ulthread's stack on
// hardware, and prosper enters the guest on exactly that range.
//
// Two deviations from hardware are instrumented rather than trusted:
//   * CONCURRENCY WIDTH. Hardware runs at most numWorkerThread ulthreads simultaneously. The runtime
//     counts RUNNING ulthreads at the real guest-code boundary (the on_enter/on_exit hooks fire on
//     the host %fs immediately around the guest entry) and reports the first crossing. Safe here
//     because numWorkerThread is a function-local constant at eboot+0x9ca3 with exactly two reads —
//     this size query and this Create — so the guest sizes and indexes nothing by worker count.
//   * GUEST STACK. Earthion gives each ulthread 0x2000 bytes, and guest code has never run on a
//     guest-supplied stack in this path before. The context is filled with a pattern at create and
//     the untouched prefix measured at exit, so an ulthread approaching its stack limit is reported
//     instead of silently overflowing into whatever the guest packed next to the buffer.
// =============================================================================================
constexpr uint8_t kStackFillByte = 0xCD;
// Only the BOTTOM of the context is patterned, not all of it. The guest stack grows down from
// base+size, so the bottom is the only region an overflow has to cross, and probing just that region
// answers the question that matters — "did this ulthread come close to the end of its stack?" —
// without the cost of the alternative. Filling the whole buffer would force every page of it
// resident: Earthion's "Leaderboards Thread" asks for an 8 MiB context, so a full fill would commit
// 8 MiB of guest memory the title might never have touched, and the measurement would have changed
// the thing it was measuring. A context at or below the probe size (Earthion's other three are 8 KiB,
// 8 KiB and 32 KiB) is covered end to end, giving an exact high-water figure.
constexpr uint64_t kStackProbeBytes = 64 * 1024;
// Warn when the deepest frame came within a quarter of the probe of the bottom — close enough to
// predict the overflow, far enough not to fire on healthy usage.
constexpr uint64_t kStackWarnFraction = 4;

uint64_t stack_probe_bytes(uint64_t size) {
    return size < kStackProbeBytes ? size : kStackProbeBytes;
}

// Bytes of untouched pattern remaining at the BOTTOM of the context. probe == the full probe region
// means the stack never reached into it.
uint64_t stack_untouched_at_bottom(uint64_t base, uint64_t size) {
    const uint64_t probe = stack_probe_bytes(size);
    if (!base || !probe || !gpu::guest_readable(base, 1)) return probe;
    const auto* p = (const uint8_t*)(uintptr_t)base;
    uint64_t untouched = 0;
    while (untouched < probe && p[untouched] == kStackFillByte) ++untouched;
    return untouched;
}

void ulthread_on_enter(void* opaque) {
    auto* o = (UltObject*)opaque;
    UltObject* rt = object_from_id(o->runtime_id, UltType::Runtime);
    if (!rt) return;
    const uint32_t running = rt->running_ulthreads.fetch_add(1, std::memory_order_acq_rel) + 1;
    uint32_t hw = rt->high_water.load(std::memory_order_relaxed);
    while (running > hw && !rt->high_water.compare_exchange_weak(hw, running,
                                                                 std::memory_order_relaxed)) {}
    if (running > rt->num_worker_thread && !rt->warned_concurrency.exchange(true))
        log_line("CONCURRENCY: %u ulthreads of runtime \"%s\" are running at once, but it was created "
                 "with numWorkerThread=%u. prosper runs ulthreads one-to-one on real guest threads "
                 "instead of multiplexing them onto that many workers, so more can run in parallel "
                 "than on hardware. Reported, deliberately NOT capped — capping would trade a visible "
                 "deviation for a hidden scheduling stall",
                 running, rt->name.c_str(), rt->num_worker_thread);
}

void ulthread_on_exit(void* opaque, uint64_t retval) {
    auto* o = (UltObject*)opaque;
    o->exit_status.store((uint32_t)retval, std::memory_order_relaxed);
    o->finished.store(true, std::memory_order_release);
    const uint64_t probe = stack_probe_bytes(o->context_size);
    const uint64_t untouched = stack_untouched_at_bottom(o->context, o->context_size);
    // Exact when the probe covered the whole context; otherwise a lower bound (the stack never
    // reached the probed bottom, so it used less than size - probe).
    const uint64_t used = untouched >= probe ? o->context_size - probe : o->context_size - untouched;
    o->stack_used.store(used, std::memory_order_relaxed);
    o->stack_exact.store(probe >= o->context_size, std::memory_order_relaxed);
    if (probe && untouched < probe / kStackWarnFraction)
        log_line("GUEST STACK: ulthread \"%s\" came within %llu bytes of the bottom of its "
                 "%llu-byte context. The context buffer IS the ulthread's stack; overflowing it "
                 "corrupts whatever the guest allocated next to it",
                 o->name.c_str(), (unsigned long long)untouched,
                 (unsigned long long)o->context_size);
    if (UltObject* rt = object_from_id(o->runtime_id, UltType::Runtime))
        rt->running_ulthreads.fetch_sub(1, std::memory_order_acq_rel);
}

// Claim / release a slot in the runtime's guest work area. The work area is what makes
// sceUltUlthreadRuntimeGetWorkAreaSize's answer load-bearing rather than decorative.
bool runtime_claim_slot(UltObject* rt, uint64_t ulthread_id, uint64_t context, uint64_t context_size) {
    if (!rt->work_area || !gpu::guest_readable(rt->work_area, sizeof(RuntimeWorkArea))) return false;
    auto* wa = (RuntimeWorkArea*)(uintptr_t)rt->work_area;
    if (wa->magic != kWorkMagicRuntime) {
        log_line("runtime \"%s\" work area at 0x%llx no longer carries prosper's header (0x%llx) — "
                 "freed or reused by the guest while the runtime is still live",
                 rt->name.c_str(), (unsigned long long)rt->work_area, (unsigned long long)wa->magic);
        return false;
    }
    auto* slots = (RuntimeSlot*)(uintptr_t)(rt->work_area + sizeof(RuntimeWorkArea));
    for (uint32_t i = 0; i < wa->num_max_ulthread; ++i) {
        if (slots[i].ulthread_id) continue;
        slots[i].ulthread_id = ulthread_id;
        slots[i].context = context;
        slots[i].context_size = context_size;
        slots[i].host_thread = 0;
        return true;
    }
    return false;
}

void runtime_release_slot(UltObject* rt, uint64_t ulthread_id) {
    if (!rt || !rt->work_area || !gpu::guest_readable(rt->work_area, sizeof(RuntimeWorkArea))) return;
    auto* wa = (RuntimeWorkArea*)(uintptr_t)rt->work_area;
    if (wa->magic != kWorkMagicRuntime) return;
    auto* slots = (RuntimeSlot*)(uintptr_t)(rt->work_area + sizeof(RuntimeWorkArea));
    for (uint32_t i = 0; i < wa->num_max_ulthread; ++i)
        if (slots[i].ulthread_id == ulthread_id) { slots[i] = RuntimeSlot{0, 0, 0, 0}; return; }
}

// _sceUltUlthreadCreate(ult, name, entry, arg, context, sizeContext, runtime, optParam, apiVersion).
// 9 args; runtime/optParam/apiVersion arrive on the guest stack and the import bridge forwards them.
PROSPER_SYSV_ABI uint64_t ult_ulthread_create(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                              uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7,
                                              uint64_t a8) {
    uint64_t out = 0;
    if (!implement(kIdxUlthreadCreate, &out)) return out;
    note_uninitialised("_sceUltUlthreadCreate");
    (void)a7;   // optParam: NULL at every observed call site

    if (!guest_object_writable(a0)) {
        log_line("_sceUltUlthreadCreate: unusable ulthread pointer 0x%llx", (unsigned long long)a0);
        return kUltErrInval;
    }
    if (!a2 || !gpu::guest_readable(a2, 1)) {
        log_line("_sceUltUlthreadCreate: entry 0x%llx is not executable guest code",
                 (unsigned long long)a2);
        return kUltErrInval;
    }
    UltObject* rt = resolve(a6, UltType::Runtime, "_sceUltUlthreadCreate(runtime)");
    if (!rt) return kUltErrInval;
    if (!a4 || !a5 || !gpu::guest_readable(a4, (uint32_t)std::min<uint64_t>(a5, 4096))) {
        log_line("_sceUltUlthreadCreate: context 0x%llx size %llu is unusable as this ulthread's stack",
                 (unsigned long long)a4, (unsigned long long)a5);
        return kUltErrInval;
    }

    // numMaxUlthread is unambiguous and the guest chose it, so it is enforced rather than merely
    // reported: exceeding it means the runtime the guest sized is genuinely full.
    const uint32_t live = rt->live_ulthreads.load(std::memory_order_relaxed);
    if (live >= rt->num_max_ulthread) {
        log_line("_sceUltUlthreadCreate: runtime \"%s\" already holds %u of numMaxUlthread=%u "
                 "ulthreads — refusing", rt->name.c_str(), live, rt->num_max_ulthread);
        return kUltErrAgain;
    }

    UltObject* o = allocate_object(UltType::Ulthread);
    o->name = guest_string(a1);
    o->guest_addr = a0;
    o->api_version = (uint32_t)a8;
    o->runtime_id = make_id(rt->slot, rt->generation);
    o->context = a4;
    o->context_size = a5;
    o->host_thread = 0;
    o->finished.store(false, std::memory_order_relaxed);
    o->joined.store(false, std::memory_order_relaxed);
    o->exit_status.store(0, std::memory_order_relaxed);
    o->stack_used.store(0, std::memory_order_relaxed);
    o->alive.store(true, std::memory_order_release);
    const uint64_t id = make_id(o->slot, o->generation);

    if (!runtime_claim_slot(rt, id, a4, a5)) {
        log_line("_sceUltUlthreadCreate: runtime \"%s\" has no free work-area slot", rt->name.c_str());
        o->alive.store(false, std::memory_order_release);
        return kUltErrAgain;
    }
    if (!publish_object(a0, UltType::Ulthread, id)) {
        runtime_release_slot(rt, id);
        o->alive.store(false, std::memory_order_release);
        return kUltErrInval;
    }

    // Amendment 6: the context IS the stack. Fill it so the untouched prefix at exit measures the
    // high-water mark. Safe to write — the guest malloc'd this buffer immediately before the call
    // (eboot+0x9de0) purely to hand it over, so nothing of its own is in it yet.
    std::memset((void*)(uintptr_t)a4, kStackFillByte, (size_t)stack_probe_bytes(a5));

    rt->live_ulthreads.fetch_add(1, std::memory_order_acq_rel);
    GuestThreadHooks hooks;
    hooks.on_enter = ulthread_on_enter;
    hooks.on_exit  = ulthread_on_exit;
    hooks.opaque   = o;
    uint64_t thread = 0;
    const int rc = guest_thread_spawn(a2, a3, o->name.empty() ? "ulthread" : o->name.c_str(),
                                      (void*)(uintptr_t)a4, (size_t)a5, &hooks, &thread);
    if (rc != 0) {
        log_line("_sceUltUlthreadCreate(\"%s\"): guest thread spawn failed (%d)", o->name.c_str(), rc);
        rt->live_ulthreads.fetch_sub(1, std::memory_order_acq_rel);
        runtime_release_slot(rt, id);
        unpublish_object(a0);
        o->alive.store(false, std::memory_order_release);
        return kUltErrNoMem;
    }
    o->host_thread = thread;
    log_line("_sceUltUlthreadCreate(\"%s\") ulthread=0x%llx entry=0x%llx arg=0x%llx "
             "context=0x%llx+%llu runtime=\"%s\" apiVersion=0x%x -> running on guest thread 0x%llx",
             o->name.c_str(), (unsigned long long)a0, (unsigned long long)a2,
             (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5,
             rt->name.c_str(), o->api_version, (unsigned long long)thread);
    return kUltOk;
}

// sceUltUlthreadJoin(ulthread, int32_t* status). The status out-param is 4 bytes: the call site at
// eboot+0x9f1e passes `lea rsi,[rbp-0xc]`, a 4-byte local below the stack canary at [rbp-0x8].
PROSPER_SYSV_ABI uint64_t ult_ulthread_join(uint64_t a0, uint64_t a1, uint64_t, uint64_t, uint64_t,
                                            uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxUlthreadJoin, &out)) return out;
    UltObject* o = resolve(a0, UltType::Ulthread, "sceUltUlthreadJoin");
    if (!o) return kUltErrSrch;
    if (o->joined.exchange(true, std::memory_order_acq_rel)) {
        log_line("sceUltUlthreadJoin: ulthread \"%s\" (0x%llx) was already joined", o->name.c_str(),
                 (unsigned long long)a0);
        return kUltErrInval;
    }

    // Watchdog: the guest frees the context buffer right after join returns, so a join that never
    // returns is both a hang AND the reason a later use-after-free would look inexplicable.
    if (!o->finished.load(std::memory_order_acquire)) {
        const uint64_t warn_ms = block_warn_ms();
        struct timespec deadline;
        abstime_in_ms(deadline, warn_ms);
        int rc = 0;
#if defined(__linux__)
        rc = pthread_timedjoin_np((pthread_t)(uintptr_t)o->host_thread, nullptr, &deadline);
        if (rc == ETIMEDOUT) {
            log_line("BLOCKED >%llums: sceUltUlthreadJoin on \"%s\" (0x%llx) — the ulthread has not "
                     "returned. Still waiting. (PROSPER_ULT_BLOCK_WARN_MS)",
                     (unsigned long long)warn_ms, o->name.c_str(), (unsigned long long)a0);
            rc = pthread_join((pthread_t)(uintptr_t)o->host_thread, nullptr);
        }
#else
        rc = pthread_join((pthread_t)(uintptr_t)o->host_thread, nullptr);
#endif
        if (rc != 0 && rc != ESRCH) {
            log_line("sceUltUlthreadJoin on \"%s\" failed: pthread rc=%d", o->name.c_str(), rc);
            return kUltErrInval;
        }
    } else {
        pthread_join((pthread_t)(uintptr_t)o->host_thread, nullptr);
    }

    // The ulthread entry is int32_t(*)(uint64_t); its return travels in eax.
    const uint32_t status = o->exit_status.load(std::memory_order_relaxed);
    if (a1) {
        if (!gpu::guest_readable(a1, sizeof(int32_t))) {
            log_line("sceUltUlthreadJoin: status out-param 0x%llx is unusable — not writing it",
                     (unsigned long long)a1);
        } else {
            *(int32_t*)(uintptr_t)a1 = (int32_t)status;
        }
    }
    g_last_join_stack_used.store(o->stack_used.load(std::memory_order_relaxed),
                                 std::memory_order_relaxed);
    g_last_join_context_size.store(o->context_size, std::memory_order_relaxed);
    log_line("sceUltUlthreadJoin(\"%s\") -> status %d, guest stack high-water %s%llu of %llu bytes",
             o->name.c_str(), (int)status,
             o->stack_exact.load(std::memory_order_relaxed) ? "" : "<=",
             (unsigned long long)o->stack_used.load(), (unsigned long long)o->context_size);

    UltObject* rt = object_from_id(o->runtime_id, UltType::Runtime);
    if (rt) {
        runtime_release_slot(rt, make_id(o->slot, o->generation));
        rt->live_ulthreads.fetch_sub(1, std::memory_order_acq_rel);
    }
    unpublish_object(a0);
    o->alive.store(false, std::memory_order_release);
    return kUltOk;
}

// =============================================================================================
// Condition variables
//
// sceUltConditionVariableWait takes ONE argument: the mutex is bound at create. Two facts from the
// guest's own call sites shape this implementation, and both were established before writing it.
//
//  1. THE CALLER HOLDS THE BOUND MUTEX AT EVERY Wait. All five sites are immediately preceded by
//     sceUltMutexLock on exactly the mutex the condvar was created with (eboot 0xbda0->0xbdba,
//     0x13808->0x1381b, 0x1386d->0x13880, 0x1427e->0x14291, 0xff433->0xff446). That is what makes
//     pthread_cond_wait's atomic release/reacquire correct rather than undefined.
//
//  2. THE CALLER DOES *NOT* HOLD IT AT Signal. At every one of the eleven Signal sites the bound
//     mutex is unheld — the guest either holds a DIFFERENT mutex (0xb582 unlocks +0x70 before
//     signalling the condvar at +0x170, whose bound mutex is +0x270) or holds nothing at all
//     (0xff502, 0xff5db). Signalling outside the mutex is legal, but it means the wake accounting
//     cannot be protected by the bound mutex, because Signal never takes it.
//
// So wakes are accounted with two monotonic counters instead of a shared lock. A waiter takes ticket
// N from `wait_seq` and proceeds only once `wake_seq` reaches N; Signal advances `wake_seq` by one
// and broadcasts. This gives exactly-one-waiter-released-per-Signal, suppresses spurious wakeups
// (the guest must not resume on an unmet predicate if any of its five sites turns out not to loop on
// one), and preserves the POSIX rule that a Signal with no waiter is LOST rather than remembered —
// `wake_seq` never overtakes `wait_seq`.
//
// The one window this leaves is a broadcast landing between a waiter's ticket test and its
// pthread_cond_timedwait, which no lock-free scheme can close while Signal does not take the mutex.
// It is bounded rather than papered over: waiters poll on a short interval, so a missed broadcast
// costs at most kCondPollMs, never a hang. With at most numMaxUlthread waiters that is free.
// CONFIDENCE: HIGH on the two call-site facts; MED that real libSceUlt suppresses spurious wakeups
// (prosper has no capture of a real console's Wait returning), and suppressing is the safe direction:
// a guest that loops on its predicate cannot tell the difference, and one that does not is protected.
constexpr uint64_t kCondPollMs = 20;

PROSPER_SYSV_ABI uint64_t ult_cond_create(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                          uint64_t a4, uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxCondCreate, &out)) return out;
    note_uninitialised("_sceUltConditionVariableCreate");
    if (!guest_object_writable(a0)) {
        log_line("_sceUltConditionVariableCreate: unusable condvar pointer 0x%llx",
                 (unsigned long long)a0);
        return kUltErrInval;
    }
    UltObject* mutex = resolve(a2, UltType::Mutex, "_sceUltConditionVariableCreate(mutex)");
    if (!mutex) return kUltErrInval;

    UltObject* o = allocate_object(UltType::Cond);
    if (!o->cond_valid) {
        const int rc = pthread_cond_init(&o->cond, nullptr);
        if (rc) {
            log_line("_sceUltConditionVariableCreate: pthread_cond_init failed (%d)", rc);
            return kUltErrNoMem;
        }
        o->cond_valid = true;
    }
    o->name = guest_string(a1);
    o->guest_addr = a0;
    o->api_version = (uint32_t)a4;
    o->mutex_id = make_id(mutex->slot, mutex->generation);
    o->wait_seq.store(0, std::memory_order_relaxed);
    o->wake_seq.store(0, std::memory_order_relaxed);
    o->destroying.store(false, std::memory_order_relaxed);
    o->alive.store(true, std::memory_order_release);
    const uint64_t id = make_id(o->slot, o->generation);
    if (!publish_object(a0, UltType::Cond, id)) {
        o->alive.store(false, std::memory_order_release);
        return kUltErrInval;
    }
    // Bound to the same pool the mutex came from, so pool accounting sees condvars too.
    if (UltObject* pool = object_from_id(mutex->pool_id, UltType::Pool)) {
        o->pool_id = mutex->pool_id;
        pool->bound_sync_objects.fetch_add(1, std::memory_order_relaxed);
    }
    log_line("_sceUltConditionVariableCreate(\"%s\") cv=0x%llx mutex=0x%llx(\"%s\") apiVersion=0x%x -> ok",
             o->name.c_str(), (unsigned long long)a0, (unsigned long long)a2, mutex->name.c_str(),
             o->api_version);
    return kUltOk;
}

PROSPER_SYSV_ABI uint64_t ult_cond_wait(uint64_t a0, uint64_t, uint64_t, uint64_t, uint64_t,
                                        uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxCondWait, &out)) return out;
    UltObject* o = resolve(a0, UltType::Cond, "sceUltConditionVariableWait");
    if (!o) return kUltErrSrch;
    UltObject* m = object_from_id(o->mutex_id, UltType::Mutex);
    if (!m) {
        log_line("sceUltConditionVariableWait: condvar \"%s\" (0x%llx) has no live bound mutex — it "
                 "was destroyed while the condvar still refers to it",
                 o->name.c_str(), (unsigned long long)a0);
        return kUltErrInval;
    }
    const uint64_t me = self_thread();
    // The classic contract, and one this title does honour at all five sites. Waiting without the
    // mutex is undefined for pthread_cond_wait, so it is refused rather than executed.
    if (m->owner.load(std::memory_order_relaxed) != me) {
        static std::atomic<uint64_t> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 8)
            log_line("sceUltConditionVariableWait on \"%s\" by thread 0x%llx, which does NOT hold the "
                     "bound mutex \"%s\" (holder 0x%llx) — refusing rather than entering undefined "
                     "behaviour", o->name.c_str(), (unsigned long long)me, m->name.c_str(),
                     (unsigned long long)m->owner.load(std::memory_order_relaxed));
        return kUltErrPerm;
    }

    const uint64_t ticket = o->wait_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
    uint64_t waited_ms = 0;
    bool noted = false;
    uint64_t rc = kUltOk;
    m->owner.store(0, std::memory_order_relaxed);       // released for the duration of the wait
    while (o->wake_seq.load(std::memory_order_acquire) < ticket) {
        if (o->destroying.load(std::memory_order_acquire)) {
            log_line("sceUltConditionVariableWait on \"%s\" returning because the condvar is being "
                     "destroyed", o->name.c_str());
            rc = kUltErrSrch;
            break;
        }
        struct timespec deadline;
        abstime_in_ms(deadline, kCondPollMs);
        pthread_cond_timedwait(&o->cond, &m->mtx, &deadline);
        waited_ms += kCondPollMs;
        if (!noted && waited_ms >= cond_note_ms()) {
            noted = true;
            log_line("WAITING >%llums: sceUltConditionVariableWait on \"%s\" (0x%llx), thread 0x%llx. "
                     "This is informational — a worker waiting for work is normal — but if the title "
                     "appears hung, nothing has signalled this condvar. (PROSPER_ULT_COND_NOTE_MS)",
                     (unsigned long long)cond_note_ms(), o->name.c_str(), (unsigned long long)a0,
                     (unsigned long long)me);
        }
    }
    m->owner.store(me, std::memory_order_relaxed);      // reacquired by pthread_cond_timedwait
    return rc;
}

PROSPER_SYSV_ABI uint64_t ult_cond_signal(uint64_t a0, uint64_t, uint64_t, uint64_t, uint64_t,
                                          uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxCondSignal, &out)) return out;
    UltObject* o = resolve(a0, UltType::Cond, "sceUltConditionVariableSignal");
    if (!o) return kUltErrSrch;
    // Release at most one un-woken waiter, and only if one exists: a Signal with nobody waiting is
    // LOST, as it is for a real condition variable. wake_seq therefore never overtakes wait_seq.
    uint64_t woken = o->wake_seq.load(std::memory_order_acquire);
    for (;;) {
        const uint64_t issued = o->wait_seq.load(std::memory_order_acquire);
        if (woken >= issued) return kUltOk;             // no waiter to release
        if (o->wake_seq.compare_exchange_weak(woken, woken + 1, std::memory_order_acq_rel,
                                              std::memory_order_acquire))
            break;
    }
    // Broadcast rather than signal: waiters select themselves by ticket, so every one must
    // re-evaluate. Bounded by numMaxUlthread, which the guest sizes at 16.
    pthread_cond_broadcast(&o->cond);
    return kUltOk;
}

PROSPER_SYSV_ABI uint64_t ult_cond_destroy(uint64_t a0, uint64_t, uint64_t, uint64_t, uint64_t,
                                           uint64_t) {
    uint64_t out = 0;
    if (!implement(kIdxCondDestroy, &out)) return out;
    UltObject* o = resolve(a0, UltType::Cond, "sceUltConditionVariableDestroy");
    if (!o) return kUltErrSrch;
    const uint64_t issued = o->wait_seq.load(std::memory_order_acquire);
    const uint64_t woken = o->wake_seq.load(std::memory_order_acquire);
    if (issued > woken)
        log_line("sceUltConditionVariableDestroy on \"%s\" (0x%llx) with %llu waiter(s) still "
                 "blocked — releasing them with an error rather than leaving them stuck",
                 o->name.c_str(), (unsigned long long)a0, (unsigned long long)(issued - woken));
    o->destroying.store(true, std::memory_order_release);
    pthread_cond_broadcast(&o->cond);
    unpublish_object(a0);
    o->alive.store(false, std::memory_order_release);
    if (UltObject* pool = object_from_id(o->pool_id, UltType::Pool))
        pool->bound_sync_objects.fetch_sub(1, std::memory_order_relaxed);
    return kUltOk;
}
}  // namespace

void register_ult_hle() {
    g_return_success.store(env_flag("PROSPER_ULT_RETURN_SUCCESS"), std::memory_order_relaxed);
    g_legacy_enosys.store(env_flag("PROSPER_ULT_LEGACY_ENOSYS"), std::memory_order_relaxed);

    Hle::register_fn(kUlt[kIdxInitialize].nid,   (HleFn)ult_initialize,   kUlt[kIdxInitialize].name);
    Hle::register_fn(kUlt[kIdxFinalize].nid,     (HleFn)ult_finalize,     kUlt[kIdxFinalize].name);
    Hle::register_fn(kUlt[kIdxPoolSize].nid,     (HleFn)ult_pool_work_area_size, kUlt[kIdxPoolSize].name);
    Hle::register_fn(kUlt[kIdxPoolCreate].nid,   (HleFn)ult_pool_create,  kUlt[kIdxPoolCreate].name);
    Hle::register_fn(kUlt[kIdxRuntimeSize].nid,  (HleFn)ult_runtime_work_area_size,
                     kUlt[kIdxRuntimeSize].name);
    Hle::register_fn(kUlt[kIdxRuntimeCreate].nid, (HleFn)ult_runtime_create,
                     kUlt[kIdxRuntimeCreate].name);
    Hle::register_fn(kUlt[kIdxMutexCreate].nid,  (HleFn)ult_mutex_create,  kUlt[kIdxMutexCreate].name);
    Hle::register_fn(kUlt[kIdxMutexLock].nid,    (HleFn)ult_mutex_lock,    kUlt[kIdxMutexLock].name);
    Hle::register_fn(kUlt[kIdxMutexUnlock].nid,  (HleFn)ult_mutex_unlock,  kUlt[kIdxMutexUnlock].name);
    Hle::register_fn(kUlt[kIdxMutexDestroy].nid, (HleFn)ult_mutex_destroy, kUlt[kIdxMutexDestroy].name);

    Hle::register_fn(kUlt[kIdxUlthreadCreate].nid, (HleFn)ult_ulthread_create,
                     kUlt[kIdxUlthreadCreate].name);
    Hle::register_fn(kUlt[kIdxUlthreadJoin].nid,   (HleFn)ult_ulthread_join,
                     kUlt[kIdxUlthreadJoin].name);
    Hle::register_fn(kUlt[kIdxCondCreate].nid,  (HleFn)ult_cond_create,  kUlt[kIdxCondCreate].name);
    Hle::register_fn(kUlt[kIdxCondWait].nid,    (HleFn)ult_cond_wait,    kUlt[kIdxCondWait].name);
    Hle::register_fn(kUlt[kIdxCondSignal].nid,  (HleFn)ult_cond_signal,  kUlt[kIdxCondSignal].name);
    Hle::register_fn(kUlt[kIdxCondDestroy].nid, (HleFn)ult_cond_destroy, kUlt[kIdxCondDestroy].name);
}

uint64_t ult_call_count(const char* nid) {
    if (!nid) return 0;
    for (size_t i = 0; i < kUltCount; ++i)
        if (std::strcmp(nid, kUlt[i].nid) == 0) return g_calls[i].load(std::memory_order_relaxed);
    return 0;
}

bool ult_set_return_success_for_test(bool return_success) {
    return g_return_success.exchange(return_success, std::memory_order_relaxed);
}

uint64_t ult_runtime_high_water_for_test(uint64_t guest_runtime_addr) {
    UltObject* rt = resolve(guest_runtime_addr, UltType::Runtime, "ult_runtime_high_water_for_test");
    return rt ? rt->high_water.load(std::memory_order_relaxed) : 0;
}

uint64_t ult_last_join_stack_used_for_test(uint64_t* context_size) {
    if (context_size) *context_size = g_last_join_context_size.load(std::memory_order_relaxed);
    return g_last_join_stack_used.load(std::memory_order_relaxed);
}

bool ult_set_legacy_enosys_for_test(bool legacy_enosys) {
    return g_legacy_enosys.exchange(legacy_enosys, std::memory_order_relaxed);
}

void ult_reset_counts_for_test() {
    for (size_t i = 0; i < kUltCount; ++i) {
        g_calls[i].store(0, std::memory_order_relaxed);
        g_next_report[i].store(0, std::memory_order_relaxed);
    }
    g_banner_printed.store(false, std::memory_order_relaxed);
}

void ult_dump_call_log(FILE* f) {
    uint64_t total = 0;
    for (size_t i = 0; i < kUltCount; ++i) total += g_calls[i].load(std::memory_order_relaxed);
    if (!total) return;
    fprintf(f, "\n=== libSceUlt calls (user-level threading) ===\n");
    for (size_t i = 0; i < kUltCount; ++i) {
        const uint64_t n = g_calls[i].load(std::memory_order_relaxed);
        if (!n) continue;
        fprintf(f, "  %10llu x  %-24s %s\n", (unsigned long long)n, kUlt[i].nid, kUlt[i].name);
    }
    std::shared_lock<std::shared_mutex> lk(g_reg_lock);
    for (const auto& up : g_objects) {
        const UltObject* o = up.get();
        if (o->type != UltType::Mutex) continue;
        const uint64_t locks = o->lock_calls.load(std::memory_order_relaxed);
        if (!locks) continue;
        fprintf(f, "     mutex \"%s\" @0x%llx: %llu locks, %llu contended\n", o->name.c_str(),
                (unsigned long long)o->guest_addr, (unsigned long long)locks,
                (unsigned long long)o->contended.load(std::memory_order_relaxed));
    }
}

}  // namespace prosper
