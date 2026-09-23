// Linux/glibc allocation attribution. This is an opt-in LD_PRELOAD instrument, not a timer.
// Keep the hot accounting path allocation-free: it runs from operator new itself.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>

namespace {
#ifndef PROSPER_NEW_PROBE_TEST_SLOTS
constexpr unsigned kSlots = 32768;
#else
constexpr unsigned kSlots = PROSPER_NEW_PROBE_TEST_SLOTS;
#endif
static_assert(kSlots >= 2 && (kSlots & (kSlots - 1)) == 0,
              "the fixed table must have a power-of-two capacity");
constexpr unsigned kMaxProbes = kSlots < 128 ? kSlots : 128;
struct Site {
    std::atomic<unsigned> state; // 0 empty, 1 being published, 2 ready
    unsigned tid;
    uintptr_t caller;
    unsigned kind; // 0 scalar, 1 array
    std::atomic<uint64_t> calls;
    std::atomic<uint64_t> bytes;
};
Site sites[kSlots];
std::atomic<uint64_t> total_calls, total_bytes, overflow_calls, overflow_bytes;
std::atomic<uint64_t> failed_calls, bootstrap_calls, resolution_failures;
std::atomic<uint64_t> report_failures;
std::atomic<unsigned> report_warning_emitted;
std::atomic<unsigned> forked_child;
thread_local bool inside_probe;
thread_local unsigned cached_tid;
using NewFunction = void* (*)(size_t);
static_assert(std::atomic<unsigned>::is_always_lock_free &&
              std::atomic<uint64_t>::is_always_lock_free &&
              std::atomic<NewFunction>::is_always_lock_free,
              "the allocation interposer cannot fall back to allocating atomic locks");
std::atomic<NewFunction> real_scalar, real_array;
pthread_t report_thread;
bool report_thread_started;
std::atomic<unsigned> stop_reporter;
char report_dir[768] = ".";

#define READ(x) (x).load(std::memory_order_relaxed)
#define ADD(x, value) (x).fetch_add((value), std::memory_order_relaxed)

struct ProbeScope {
    bool previous;
    ProbeScope() : previous(inside_probe) { inside_probe = true; }
    ~ProbeScope() { inside_probe = previous; }
    ProbeScope(const ProbeScope&) = delete;
    ProbeScope& operator=(const ProbeScope&) = delete;
};

unsigned current_tid() {
    if (!cached_tid) cached_tid = static_cast<unsigned>(syscall(SYS_gettid));
    return cached_tid;
}

// Only used while dlsym resolves the real C++ operator. This path obeys ordinary
// throwing-new allocation and new_handler behavior; its calls are counted separately.
extern "C" void* __libc_malloc(size_t);
std::byte* bootstrap_new(size_t bytes) {
    ADD(bootstrap_calls, 1);
    for (;;) {
        if (auto* p = __libc_malloc(bytes ? bytes : 1)) return static_cast<std::byte*>(p);
        const std::new_handler handler = std::get_new_handler();
        if (!handler) throw std::bad_alloc();
        handler();
    }
}

NewFunction resolve_real(unsigned kind) {
    auto& real = kind ? real_array : real_scalar;
    NewFunction fn = real.load(std::memory_order_acquire);
    if (fn) return fn;
    if (inside_probe) return nullptr;
    ProbeScope scope;
    fn = reinterpret_cast<NewFunction>(dlsym(RTLD_NEXT, kind ? "_Znam" : "_Znwm"));
    if (!fn) {
        ADD(resolution_failures, 1);
        return nullptr;
    }
    real.store(fn, std::memory_order_release);
    return fn;
}

void record(uintptr_t caller, size_t bytes, unsigned kind) {
    if (READ(forked_child)) return;
    const unsigned tid = current_tid();
    ADD(total_calls, 1);
    ADD(total_bytes, bytes);
    // A key contains an OS TID. Only that thread can publish the same key, and
    // inside_probe prevents signal-time recursion from publishing it twice.
    uint64_t hash = caller ^ (uint64_t(tid) << 32) ^ (uint64_t(kind) << 56);
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    for (unsigned n = 0; n < kMaxProbes; ++n) {
        Site& site = sites[(hash + n) & (kSlots - 1)];
        unsigned state = site.state.load(std::memory_order_acquire);
        if (state == 0) {
            unsigned expected = 0;
            if (site.state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
                site.tid = tid;
                site.caller = caller;
                site.kind = kind;
                site.state.store(2, std::memory_order_release);
                state = 2;
            } else {
                state = expected;
            }
        }
        if (state != 2) continue;
        if (site.tid != tid || site.caller != caller || site.kind != kind) continue;
        ADD(site.calls, 1);
        ADD(site.bytes, bytes);
        return;
    }
    ADD(overflow_calls, 1);
    ADD(overflow_bytes, bytes);
}

void report_error(const char* operation, int error) {
    ADD(report_failures, 1);
    if (report_warning_emitted.exchange(1u, std::memory_order_relaxed) == 0) {
        fprintf(stderr, "[new-probe] REPORT FAILED at %s (errno=%d); "
                        "attribution file may be missing or stale\n", operation, error);
        fflush(stderr);
    }
}

// Escape delimiters, line endings and the escape character itself. This leaves
// spaces intact while keeping every path and symbol within one TSV cell.
void tsv_field(FILE* out, const char* text) {
    static constexpr char hex[] = "0123456789abcdef";
    for (const char* p = text; *p; ++p) {
        const auto ch = static_cast<unsigned char>(*p);
        switch (ch) {
        case '\\': fputs("\\\\", out); break;
        case '\t': fputs("\\t", out); break;
        case '\n': fputs("\\n", out); break;
        case '\r': fputs("\\r", out); break;
        default:
            if (ch < 0x20 || ch == 0x7f) {
                fputc('\\', out); fputc('x', out);
                fputc(hex[ch >> 4], out); fputc(hex[ch & 15], out);
            } else {
                fputc(ch, out);
            }
        }
    }
}

void dump() {
    if (!report_dir[0]) return; // Constructor already reported the invalid directory.
    char path[1024], temporary[1024];
    const int pid = static_cast<int>(getpid());
    const int a = snprintf(path, sizeof(path), "%s/new-sites-%d.tsv", report_dir, pid);
    const int b = snprintf(temporary, sizeof(temporary), "%s/.new-sites-%d.tmp", report_dir, pid);
    if (a < 0 || b < 0 || a >= static_cast<int>(sizeof(path)) ||
        b >= static_cast<int>(sizeof(temporary))) {
        report_error("path construction", ENAMETOOLONG);
        return;
    }
    FILE* out = fopen(temporary, "w");
    if (!out) {
        report_error("open", errno);
        return;
    }
    fprintf(out, "# pid=%d successful_calls=%llu requested_bytes=%llu overflow_calls=%llu "
                     "overflow_bytes=%llu failed_calls=%llu bootstrap_calls=%llu "
                     "resolution_failures=%llu report_failures=%llu\n", pid,
                static_cast<unsigned long long>(READ(total_calls)),
                static_cast<unsigned long long>(READ(total_bytes)),
                static_cast<unsigned long long>(READ(overflow_calls)),
                static_cast<unsigned long long>(READ(overflow_bytes)),
                static_cast<unsigned long long>(READ(failed_calls)),
                static_cast<unsigned long long>(READ(bootstrap_calls)),
                static_cast<unsigned long long>(READ(resolution_failures)),
                static_cast<unsigned long long>(READ(report_failures)));
        fputs("# tid\tkind\tcaller\tcalls\trequested_bytes\tdso\tdso_offset\tsymbol\n", out);
        for (const Site& site : sites) {
            if (site.state.load(std::memory_order_acquire) != 2) continue;
            Dl_info info{};
            const uintptr_t address = site.caller;
            const bool known = address && dladdr(reinterpret_cast<void*>(address - 1), &info);
            const uintptr_t offset = known && info.dli_fbase
                ? address - reinterpret_cast<uintptr_t>(info.dli_fbase) : 0;
            fprintf(out, "%u\t%s\t0x%llx\t%llu\t%llu\t",
                    site.tid, site.kind ? "array" : "scalar",
                    static_cast<unsigned long long>(address),
                    static_cast<unsigned long long>(READ(site.calls)),
                    static_cast<unsigned long long>(READ(site.bytes)));
            tsv_field(out, known && info.dli_fname ? info.dli_fname : "?");
            fprintf(out, "\t0x%llx\t", static_cast<unsigned long long>(offset));
            tsv_field(out, known && info.dli_sname ? info.dli_sname : "?");
            fputc('\n', out);
        }
    const int write_error = ferror(out) ? errno : 0;
    const int close_result = fclose(out);
    if (write_error || close_result != 0) {
        report_error("write/close", write_error ? write_error : errno);
        unlink(temporary);
        return;
    }
    if (rename(temporary, path) != 0) {
        report_error("rename", errno);
        unlink(temporary);
    }
}

void child_after_fork() {
    forked_child.store(1u, std::memory_order_relaxed);
    report_warning_emitted.store(0u, std::memory_order_relaxed);
    report_thread_started = false; // The parent's reporter thread no longer exists.
    static constexpr char message[] =
        "[new-probe] REFUSED forked child without exec: inherited attribution is invalid\n";
    write(STDERR_FILENO, message, sizeof(message) - 1);
    // An exit-time stdio/dladdr dump can deadlock after fork from a multithreaded
    // parent. Publish a refusal marker here using only async-signal-safe calls.
    if (!report_dir[0]) return;
    char path[1024];
    size_t used = 0;
    for (const char* p = report_dir; *p && used + 1 < sizeof(path); ++p)
        path[used++] = *p;
    static constexpr char prefix[] = "/new-sites-";
    for (size_t i = 0; i < sizeof(prefix) - 1 && used + 1 < sizeof(path); ++i)
        path[used++] = prefix[i];
    char digits[16];
    unsigned pid = static_cast<unsigned>(getpid());
    unsigned digit_count = 0;
    do { digits[digit_count++] = static_cast<char>('0' + pid % 10); pid /= 10; }
    while (pid && digit_count < sizeof(digits));
    while (digit_count && used + 1 < sizeof(path)) path[used++] = digits[--digit_count];
    static constexpr char suffix[] = ".tsv";
    for (size_t i = 0; i < sizeof(suffix) - 1 && used + 1 < sizeof(path); ++i)
        path[used++] = suffix[i];
    path[used] = '\0';
    static constexpr char refusal[] =
        "# refused=forked-child inherited counters; launch a fresh process with exec\n";
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd >= 0) {
        write(fd, refusal, sizeof(refusal) - 1);
        close(fd);
    }
}

void* reporter(void*) {
    ProbeScope scope;
    unsigned seconds = 0;
    while (!READ(stop_reporter)) {
        sleep(1);
        if (++seconds == 1 || seconds % 10 == 0) dump();
    }
    return nullptr;
}

__attribute__((constructor)) void init() {
    ProbeScope scope;
    const char* configured_dir = getenv("PROSPER_NEW_PROBE_DIR");
    if (configured_dir && *configured_dir) {
        const int length = snprintf(report_dir, sizeof(report_dir), "%s", configured_dir);
        if (length < 0 || length >= static_cast<int>(sizeof(report_dir))) {
            report_dir[0] = '\0';
            report_error("configured directory", ENAMETOOLONG);
        }
    }
    // Resolve before spawning the reporter. A recursive new during dlsym takes the
    // bootstrap path; normal calls delegate to the original libstdc++ operator.
    real_scalar.store(reinterpret_cast<NewFunction>(dlsym(RTLD_NEXT, "_Znwm")),
                      std::memory_order_release);
    real_array.store(reinterpret_cast<NewFunction>(dlsym(RTLD_NEXT, "_Znam")),
                     std::memory_order_release);
    if (!real_scalar.load(std::memory_order_acquire) ||
        !real_array.load(std::memory_order_acquire)) {
        ADD(resolution_failures, 1);
        fputs("[new-probe] operator new resolution failed; attribution incomplete\n", stderr);
    }
    const int fork_hook_error = pthread_atfork(nullptr, nullptr, child_after_fork);
    if (fork_hook_error) report_error("pthread_atfork", fork_hook_error);
    if (pthread_create(&report_thread, nullptr, reporter, nullptr) == 0) {
        report_thread_started = true;
    } else {
        fputs("[new-probe] periodic reporter could not start; only normal exit can report\n", stderr);
    }
}

__attribute__((destructor)) void finish() {
    ProbeScope scope;
    if (READ(forked_child)) return; // Refusal was published by the atfork child hook.
    stop_reporter.store(1u, std::memory_order_relaxed);
    if (report_thread_started) pthread_join(report_thread, nullptr);
    dump();
}
} // namespace

// The resolved operator new owns the allocation. Keep its original matching
// delete visible through RTLD_NEXT; a probe-defined delete could use the wrong
// allocator if a host replaces the C++ allocation pair. Sonar's matching-delete
// rule does not model LD_PRELOAD delegation.
__attribute__((noinline)) void* operator new(size_t bytes) { // NOSONAR
    const uintptr_t caller = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
    NewFunction real = resolve_real(0);
    if (!real) return bootstrap_new(bytes);
    if (inside_probe) return real(bytes);
    ProbeScope scope;
    try {
        void* p = real(bytes);
        record(caller, bytes, 0);
        return p;
    } catch (...) {
        ADD(failed_calls, 1);
        throw;
    }
}

__attribute__((noinline)) void* operator new[](size_t bytes) { // NOSONAR
    const uintptr_t caller = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
    NewFunction real = resolve_real(1);
    if (!real) return bootstrap_new(bytes);
    if (inside_probe) return real(bytes);
    ProbeScope scope;
    try {
        void* p = real(bytes);
        record(caller, bytes, 1);
        return p;
    } catch (...) {
        ADD(failed_calls, 1);
        throw;
    }
}
