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
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>

extern "C" void* __libc_malloc(size_t);

namespace {
#ifndef PROSPER_NEW_PROBE_TEST_SLOTS
constexpr unsigned kSlots = 32768;
#else
constexpr unsigned kSlots = PROSPER_NEW_PROBE_TEST_SLOTS;
#endif
// C++17 has no std::has_single_bit; this expression is its exact power-of-two test.
static_assert(kSlots >= 2 && (kSlots & (kSlots - 1)) == 0, // NOSONAR
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
// The interposer cannot allocate its own accounting state. These objects need
// static storage and remain mutable: shared counters/flags are atomic, the
// thread-local values guard recursive operator new, and the report directory
// is set before the reporter thread starts. Sonar's global-const rule does not
// model these mutations through atomics and process initialization.
std::array<Site, kSlots> sites; // NOSONAR
std::atomic<uint64_t> total_calls; // NOSONAR
std::atomic<uint64_t> total_bytes; // NOSONAR
std::atomic<uint64_t> overflow_calls; // NOSONAR
std::atomic<uint64_t> overflow_bytes; // NOSONAR
std::atomic<uint64_t> failed_calls; // NOSONAR
std::atomic<uint64_t> bootstrap_calls; // NOSONAR
std::atomic<uint64_t> resolution_failures; // NOSONAR
std::atomic<uint64_t> report_failures; // NOSONAR
std::atomic<unsigned> report_warning_emitted; // NOSONAR
std::atomic<unsigned> forked_child; // NOSONAR
thread_local bool inside_probe; // NOSONAR
thread_local unsigned cached_tid; // NOSONAR
using NewFunction = void* (*)(size_t);
static_assert(std::atomic<unsigned>::is_always_lock_free &&
              std::atomic<uint64_t>::is_always_lock_free &&
              std::atomic<NewFunction>::is_always_lock_free,
              "the allocation interposer cannot fall back to allocating atomic locks");
std::atomic<NewFunction> real_scalar; // NOSONAR
std::atomic<NewFunction> real_array; // NOSONAR
pthread_t report_thread;
bool report_thread_started;
std::atomic<unsigned> stop_reporter; // NOSONAR
std::array<char, 768> report_dir{'.'}; // NOSONAR

// Totals and stop/refusal flags carry no payload; relaxed access is enough.
// Table keys and resolved function pointers use release/acquire publication.
// A global seq_cst order adds probe overhead without strengthening either proof.
#define READ(x) (x).load(std::memory_order_relaxed)
#define ADD(x, value) (x).fetch_add((value), std::memory_order_relaxed)

struct ProbeScope {
    bool previous = inside_probe;
    ProbeScope() { inside_probe = true; }
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
    NewFunction fn = real.load(std::memory_order_acquire); // NOSONAR
    if (fn) return fn;
    if (inside_probe) return nullptr;
    ProbeScope scope;
    fn = reinterpret_cast<NewFunction>(dlsym(RTLD_NEXT, kind ? "_Znam" : "_Znwm"));
    if (!fn) {
        ADD(resolution_failures, 1);
        return nullptr;
    }
    real.store(fn, std::memory_order_release); // NOSONAR
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
        unsigned state = site.state.load(std::memory_order_acquire); // NOSONAR
        if (state == 0) {
            unsigned expected = 0;
            if (site.state.compare_exchange_strong(expected, 1, // NOSONAR
                                                   std::memory_order_acq_rel, // NOSONAR
                                                   std::memory_order_acquire)) { // NOSONAR
                site.tid = tid;
                site.caller = caller;
                site.kind = kind;
                site.state.store(2, std::memory_order_release); // NOSONAR
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
    if (report_warning_emitted.exchange(1u, std::memory_order_relaxed) == 0) { // NOSONAR
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
        case '\\': fputs(R"(\\)", out); break;
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
    std::array<char, 1024> path;
    std::array<char, 1024> temporary;
    const auto pid = static_cast<int>(getpid());
    const int a = snprintf(path.data(), path.size(), "%s/new-sites-%d.tsv", report_dir.data(), pid);
    const int b = snprintf(temporary.data(), temporary.size(), "%s/.new-sites-%d.tmp", report_dir.data(), pid);
    if (a < 0 || b < 0 || a >= static_cast<int>(path.size()) ||
        b >= static_cast<int>(temporary.size())) {
        report_error("path construction", ENAMETOOLONG);
        return;
    }
    FILE* out = fopen(temporary.data(), "w");
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
            if (site.state.load(std::memory_order_acquire) != 2) continue; // NOSONAR
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
    if (const int close_result = fclose(out); write_error || close_result != 0) {
        report_error("write/close", write_error ? write_error : errno);
        unlink(temporary.data());
        return;
    }
    if (rename(temporary.data(), path.data()) != 0) {
        report_error("rename", errno);
        unlink(temporary.data());
    }
}

void child_after_fork() {
    forked_child.store(1u, std::memory_order_relaxed); // NOSONAR
    report_warning_emitted.store(0u, std::memory_order_relaxed); // NOSONAR
    report_thread_started = false; // The parent's reporter thread no longer exists.
    static constexpr char message[] =
        "[new-probe] REFUSED forked child without exec: inherited attribution is invalid\n";
    write(STDERR_FILENO, message, sizeof(message) - 1);
    // An exit-time stdio/dladdr dump can deadlock after fork from a multithreaded
    // parent. Publish a refusal marker here using only async-signal-safe calls.
    if (!report_dir[0]) return;
    std::array<char, 1024> path;
    size_t used = 0;
    for (const char* p = report_dir.data(); *p && used + 1 < path.size(); ++p)
        path[used++] = *p;
    static constexpr char prefix[] = "/new-sites-";
    for (size_t i = 0; i < sizeof(prefix) - 1 && used + 1 < path.size(); ++i)
        path[used++] = prefix[i];
    std::array<char, 16> digits;
    auto pid = static_cast<unsigned>(getpid());
    unsigned digit_count = 0;
    do { digits[digit_count++] = static_cast<char>('0' + pid % 10); pid /= 10; }
    while (pid && digit_count < digits.size());
    while (digit_count && used + 1 < path.size()) path[used++] = digits[--digit_count];
    static constexpr char suffix[] = ".tsv";
    for (size_t i = 0; i < sizeof(suffix) - 1 && used + 1 < path.size(); ++i)
        path[used++] = suffix[i];
    path[used] = '\0';
    static constexpr char refusal[] =
        "# refused=forked-child inherited counters; launch a fresh process with exec\n";
    const int fd = open(path.data(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
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
        const int length = snprintf(report_dir.data(), report_dir.size(), "%s", configured_dir);
        if (length < 0 || length >= static_cast<int>(report_dir.size())) {
            report_dir[0] = '\0';
            report_error("configured directory", ENAMETOOLONG);
        }
    }
    // Resolve before spawning the reporter. A recursive new during dlsym takes the
    // bootstrap path; normal calls delegate to the original libstdc++ operator.
    real_scalar.store(reinterpret_cast<NewFunction>(dlsym(RTLD_NEXT, "_Znwm")), // NOSONAR
                      std::memory_order_release); // NOSONAR
    real_array.store(reinterpret_cast<NewFunction>(dlsym(RTLD_NEXT, "_Znam")), // NOSONAR
                     std::memory_order_release); // NOSONAR
    if (!real_scalar.load(std::memory_order_acquire) || // NOSONAR
        !real_array.load(std::memory_order_acquire)) { // NOSONAR
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
    stop_reporter.store(1u, std::memory_order_relaxed); // NOSONAR
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
        auto* p = real(bytes);
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
        auto* p = real(bytes);
        record(caller, bytes, 1);
        return p;
    } catch (...) {
        ADD(failed_calls, 1);
        throw;
    }
}
