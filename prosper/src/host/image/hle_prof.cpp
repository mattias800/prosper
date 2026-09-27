// hle_prof.cpp — DIAGNOSTIC (fresh3 perf lane, not for merge): per-thread, per-import wall-time
// profile of every guest->HLE call. Enabled by PROSPER_HLE_PROF=<output dir>. When on, every
// implemented import stub jumps through prosper_hle_prof_trampoline, which swaps to the host %fs,
// calls hle_prof_enter(slot), forwards the call (including stack args 7..10 and a return hook, if the
// slot has one), then calls hle_prof_exit(slot) with the return registers preserved.
// A dumper thread samples each thread's in-flight import every 2 ms and writes cumulative snapshots.
#if defined(__linux__)
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/syscall.h>
#include <vector>

extern "C" {
uint64_t g_hle_prof_hooks[65536];          // per-slot return hook (0 = none)
uint64_t g_hle_prof_hook_tramp = 0;        // prosper_hle_hook_host_trampoline
void hle_prof_enter(uint32_t slot);
void hle_prof_exit(uint32_t slot);
void hle_prof_phase(int idx, unsigned long long ns);
}
namespace { std::atomic<uint64_t> g_phase_n[32], g_phase_ns[32], g_phase_max[32]; }
extern "C" void hle_prof_phase(int idx, unsigned long long ns) {
    if (idx < 0 || idx >= 32) return;
    g_phase_n[idx].fetch_add(1, std::memory_order_relaxed);
    g_phase_ns[idx].fetch_add(ns, std::memory_order_relaxed);
    uint64_t m = g_phase_max[idx].load(std::memory_order_relaxed);
    while (ns > m && !g_phase_max[idx].compare_exchange_weak(m, ns, std::memory_order_relaxed)) {}
}

namespace {
constexpr int kBuckets = 16;   // bucket b: [2^b, 2^(b+1)) microseconds; b0 = <2us
struct SlotStat { uint64_t n, tot_ns, max_ns; uint32_t h[kBuckets]; uint64_t samples; };
constexpr int kPage = 256, kPages = 256;
struct ThreadProf {
    pid_t tid = 0;
    std::atomic<int32_t> cur_slot{-1};
    std::atomic<uint64_t> cur_t0{0};
    int depth = 0;
    uint32_t stk_slot[64];
    uint64_t stk_t0[64];
    SlotStat* pages[kPages] = {};
    uint64_t guest_samples = 0;   // dumper-owned
    uint64_t total_samples = 0;
};
std::mutex g_mu;
std::vector<ThreadProf*> g_threads;
thread_local ThreadProf* t_prof = nullptr;
std::string g_dir;
struct SlotName { std::string lib, nid, name; };
std::vector<SlotName> g_names(65536);
std::atomic<bool> g_started{false};

inline uint64_t now_ns() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec; }
inline SlotStat* stat_of(ThreadProf* p, uint32_t slot) {
    SlotStat*& pg = p->pages[(slot / kPage) % kPages];
    if (!pg) pg = (SlotStat*)calloc(kPage, sizeof(SlotStat));
    return &pg[slot % kPage];
}
ThreadProf* self_prof() {
    if (t_prof) return t_prof;
    auto* p = new ThreadProf();
    p->tid = (pid_t)syscall(SYS_gettid);
    { std::lock_guard<std::mutex> lk(g_mu); g_threads.push_back(p); }
    t_prof = p;
    return p;
}

bool read_small(const char* path, char* buf, size_t cap) {
    FILE* f = fopen(path, "r"); if (!f) return false;
    size_t n = fread(buf, 1, cap - 1, f); fclose(f); buf[n] = 0;
    while (n && (buf[n-1] == '\n' || buf[n-1] == ' ')) buf[--n] = 0;
    return true;
}

void dump(FILE* out, uint64_t t_ms) {
    fprintf(out, "@ %llu\n", (unsigned long long)t_ms);
    for (int i = 0; i < 32; ++i) if (g_phase_n[i].load()) fprintf(out, "P %d %llu %llu %llu\n", i,
        (unsigned long long)g_phase_n[i].load(), (unsigned long long)g_phase_ns[i].load(), (unsigned long long)g_phase_max[i].load());
    // Every task in the process: comm + schedstat (run ns, runqueue-wait ns, slices) + state.
    DIR* d = opendir("/proc/self/task");
    if (d) {
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            char p[128], comm[64] = "?", ss[128] = "0 0 0", st[512] = "";
            snprintf(p, sizeof p, "/proc/self/task/%s/comm", e->d_name); read_small(p, comm, sizeof comm);
            for (char* c = comm; *c; ++c) if (*c == ' ') *c = '_';
            snprintf(p, sizeof p, "/proc/self/task/%s/schedstat", e->d_name); read_small(p, ss, sizeof ss);
            snprintf(p, sizeof p, "/proc/self/task/%s/stat", e->d_name); read_small(p, st, sizeof st);
            char state = '?'; if (const char* rp = strrchr(st, ')')) state = rp[2];
            unsigned long long vcs = 0, nvcs = 0;
            snprintf(p, sizeof p, "/proc/self/task/%s/status", e->d_name);
            if (FILE* f = fopen(p, "r")) { char line[256]; while (fgets(line, sizeof line, f)) {
                sscanf(line, "voluntary_ctxt_switches: %llu", &vcs); sscanf(line, "nonvoluntary_ctxt_switches: %llu", &nvcs); } fclose(f); }
            fprintf(out, "K %s %s %c %s %llu %llu\n", e->d_name, comm, state, ss, vcs, nvcs);
        }
        closedir(d);
    }
    std::vector<ThreadProf*> ts; { std::lock_guard<std::mutex> lk(g_mu); ts = g_threads; }
    for (ThreadProf* p : ts) {
        fprintf(out, "T %d %llu %llu\n", p->tid, (unsigned long long)p->guest_samples, (unsigned long long)p->total_samples);
        for (int pg = 0; pg < kPages; ++pg) {
            SlotStat* s = p->pages[pg]; if (!s) continue;
            for (int i = 0; i < kPage; ++i) {
                const SlotStat& x = s[i]; if (!x.n && !x.samples) continue;
                fprintf(out, "S %d %d %llu %llu %llu %llu", p->tid, pg * kPage + i, (unsigned long long)x.n,
                        (unsigned long long)x.tot_ns, (unsigned long long)x.max_ns, (unsigned long long)x.samples);
                for (int b = 0; b < kBuckets; ++b) fprintf(out, " %u", x.h[b]);
                fputc('\n', out);
            }
        }
    }
    fflush(out);
}

void dumper() {
    const char* iv = getenv("PROSPER_HLE_PROF_MS");
    const uint64_t interval_ms = iv ? strtoull(iv, nullptr, 10) : 2000;
    std::string path = g_dir + "/hleprof.txt";
    FILE* out = fopen(path.c_str(), "w");
    if (!out) { fprintf(stderr, "[hleprof] cannot open %s\n", path.c_str()); return; }
    const uint64_t t0 = now_ns();
    uint64_t next_dump = t0 + interval_ms * 1000000ull;
    size_t names_written = 0;
    for (;;) {
        timespec sl{0, 2000000}; nanosleep(&sl, nullptr);   // 2 ms state sampling
        std::vector<ThreadProf*> ts; { std::lock_guard<std::mutex> lk(g_mu); ts = g_threads; }
        for (ThreadProf* p : ts) {
            int32_t s = p->cur_slot.load(std::memory_order_relaxed);
            p->total_samples++;
            if (s < 0) p->guest_samples++;
            else { SlotStat*& pg = p->pages[(s / kPage) % kPages]; if (pg) pg[s % kPage].samples++; }
        }
        const uint64_t n = now_ns();
        if (n >= next_dump) {
            next_dump += interval_ms * 1000000ull;
            dump(out, (n - t0) / 1000000ull);
            // names file (rewrite when grown)
            size_t cnt = 0; for (auto& nm : g_names) if (!nm.nid.empty()) ++cnt;
            if (cnt != names_written) {
                names_written = cnt;
                std::string np = g_dir + "/slots.txt";
                if (FILE* f = fopen(np.c_str(), "w")) {
                    for (size_t i = 0; i < g_names.size(); ++i) if (!g_names[i].nid.empty())
                        fprintf(f, "%zu %s %s %s\n", i, g_names[i].lib.c_str(), g_names[i].nid.c_str(),
                                g_names[i].name.empty() ? "-" : g_names[i].name.c_str());
                    fclose(f);
                }
            }
        }
    }
}
}  // namespace

extern "C" void hle_prof_enter(uint32_t slot) {
    ThreadProf* p = self_prof();
    const uint64_t t = now_ns();
    if (p->depth < 64) { p->stk_slot[p->depth] = slot; p->stk_t0[p->depth] = t; }
    p->depth++;
    p->cur_t0.store(t, std::memory_order_relaxed);
    p->cur_slot.store((int32_t)slot, std::memory_order_relaxed);
}
extern "C" void hle_prof_exit(uint32_t slot) {
    ThreadProf* p = self_prof();
    const uint64_t t = now_ns();
    if (p->depth <= 0) return;
    p->depth--;
    if (p->depth >= 64) return;
    if (p->stk_slot[p->depth] != slot) {   // unbalanced (stack switch / non-returning call): resync
        p->depth = 0; p->cur_slot.store(-1, std::memory_order_relaxed); return;
    }
    const uint64_t dt = t - p->stk_t0[p->depth];
    SlotStat* s = stat_of(p, slot);
    s->n++; s->tot_ns += dt; if (dt > s->max_ns) s->max_ns = dt;
    uint64_t us = dt / 1000; int b = 0; while (us >= 2 && b < kBuckets - 1) { us >>= 1; ++b; }
    s->h[b]++;
    p->cur_slot.store(p->depth > 0 ? (int32_t)p->stk_slot[p->depth - 1] : -1, std::memory_order_relaxed);
}

// r10 = handler, r11d = slot. Entry: [rsp] = guest return, args 7..10 at rsp+8..+32.
extern "C" __attribute__((naked)) void prosper_hle_prof_trampoline() {
    __asm__ volatile(
        "pushq %r11\n"                 // slot
        "pushq %r10\n"                 // handler
        "pushq %rax\n"                 // al (variadic vector count)
        "rdfsbase %rax\n"
        "cmpl $0x50524f53, 0x108(%rax)\n"
        "jne 9f\n"
        "pushq %rax\n"                 // guest fs
        "movq 0x100(%rax), %rax\n"
        "wrfsbase %rax\n"
        "pushq %rdi\n pushq %rsi\n pushq %rdx\n pushq %rcx\n pushq %r8\n pushq %r9\n"
        "subq $136, %rsp\n"
        "movdqu %xmm0, 0(%rsp)\n movdqu %xmm1, 16(%rsp)\n movdqu %xmm2, 32(%rsp)\n movdqu %xmm3, 48(%rsp)\n"
        "movdqu %xmm4, 64(%rsp)\n movdqu %xmm5, 80(%rsp)\n movdqu %xmm6, 96(%rsp)\n movdqu %xmm7, 112(%rsp)\n"
        "movl 208(%rsp), %edi\n"
        "callq hle_prof_enter@PLT\n"
        "movdqu 0(%rsp), %xmm0\n movdqu 16(%rsp), %xmm1\n movdqu 32(%rsp), %xmm2\n movdqu 48(%rsp), %xmm3\n"
        "movdqu 64(%rsp), %xmm4\n movdqu 80(%rsp), %xmm5\n movdqu 96(%rsp), %xmm6\n movdqu 112(%rsp), %xmm7\n"
        "addq $136, %rsp\n"
        "popq %r9\n popq %r8\n popq %rcx\n popq %rdx\n popq %rsi\n popq %rdi\n"
        // frame: [rsp]=guestfs +8=rax +16=handler +24=slot +32=ret +40..+64=args7..10
        "movl 24(%rsp), %r11d\n"
        "leaq g_hle_prof_hooks(%rip), %rax\n"
        "movq (%rax,%r11,8), %r11\n"   // hook or 0
        "movq 8(%rsp), %rax\n"
        "movq 16(%rsp), %r10\n"
        "subq $8, %rsp\n"
        "pushq 72(%rsp)\n pushq 72(%rsp)\n pushq 72(%rsp)\n pushq 72(%rsp)\n"
        "testq %r11, %r11\n"
        "jz 1f\n"
        "callq *g_hle_prof_hook_tramp(%rip)\n"   // r10=handler r11=hook, host fs already active
        "jmp 2f\n"
        "1:\n"
        "callq *%r10\n"
        "2:\n"
        "addq $40, %rsp\n"
        "movq %rax, 8(%rsp)\n"
        "movq %rdx, 16(%rsp)\n"
        "subq $40, %rsp\n"
        "movdqu %xmm0, 0(%rsp)\n movdqu %xmm1, 16(%rsp)\n"
        "movl 64(%rsp), %edi\n"
        "callq hle_prof_exit@PLT\n"
        "movdqu 0(%rsp), %xmm0\n movdqu 16(%rsp), %xmm1\n"
        "addq $40, %rsp\n"
        "movq 8(%rsp), %rax\n"
        "movq 16(%rsp), %rdx\n"
        "movq 0(%rsp), %r11\n"
        "addq $32, %rsp\n"
        "wrfsbase %r11\n"
        "retq\n"
        "9:\n"                          // host-fs caller: no profiling, plain tail call
        "popq %rax\n"                  // stack: [rsp]=handler, +8=slot
        "movl 8(%rsp), %r11d\n"
        "shlq $3, %r11\n"
        "leaq g_hle_prof_hooks(%rip), %r10\n"
        "movq (%r10,%r11), %r11\n"
        "movq (%rsp), %r10\n"
        "addq $16, %rsp\n"
        "testq %r11, %r11\n"
        "jz 3f\n"
        "jmp *g_hle_prof_hook_tramp(%rip)\n"
        "3:\n"
        "jmp *%r10\n");
}

namespace prosper {
bool hle_prof_enabled() {
    static const bool on = [] { const char* d = getenv("PROSPER_HLE_PROF"); if (d && *d) { g_dir = d; return true; } return false; }();
    return on;
}
void hle_prof_note_slot(uint32_t slot, const std::string& lib, const std::string& nid, const std::string& name, uint64_t hook,
                        uint64_t hook_tramp) {
    if (slot >= 65536) return;
    g_names[slot] = SlotName{lib, nid, name};
    g_hle_prof_hooks[slot] = hook;
    g_hle_prof_hook_tramp = hook_tramp;
    if (!g_started.exchange(true)) std::thread(dumper).detach();
}
}  // namespace prosper
#endif
