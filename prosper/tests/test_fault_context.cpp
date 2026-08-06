// test_fault_context — a fatal worker-thread fault report must describe ONE fault (#2018).
//
// exec_image's signal handler keeps the interrupted registers in process globals, because guest
// threads may run on a guest %fs and `__thread` therefore cannot be used inside the handler. Those
// globals are shared, so a second thread faulting while the first was printing rewrote them
// underneath it: the first report kept going with the second fault's rip, rbp, rsp and probe
// registers, and still read as one complete fault. Measured on Crisis Core (PPSA07809): within one
// report `rbp` printed as two different values, and `insn bytes @rip` decoded to `mov eax,[rbx]`
// with `rbx = 0x20` under a header claiming `addr=0x30016000` — bytes fetched from another thread's
// rip.
//
// The fix is a stack-local snapshot taken from the handler's OWN siginfo/ucontext, plus a
// single-owner gate on the fatal report.
//
// **What this test does and does not prove.** The interleave is a live, multi-threaded property of
// the *report*, and the retained evidence for it is in #2018 — two Crisis Core reports in which
// `rbp`, `rsp` and `rax` each print two different values inside one dump. A single-threaded unit
// test cannot reproduce that, and this one does not claim to: it pins the two mechanisms the fix
// introduces. Section 1-4 assert that a capture reads the ucontext it is handed (a capture reading
// any shared or stale source lands values that are not this ucontext's and fails) and that the
// result is a value another capture cannot alter. Sections 5 and 6 are the ones with real mutation
// arms: remove the gate — the pre-fix state, where every faulting thread printed — and section 5's
// second assertion fails; make the gate a non-atomic `if (owner == 0) owner = tid;`, which passes
// every sequential check, and section 6 fails on a round with two winners.
//
// Exit code is truth: 0 = pass.
#include <cstdio>
#include <cstdint>

#if defined(__linux__) || defined(__APPLE__)
#include <atomic>
#include <thread>
#include <type_traits>
#include <vector>

#include "../src/host/fault_context.hpp"

using prosper::host::FaultContext;
using prosper::host::capture_fault_context;

static int failures = 0;

static void check_eq(const char* what, uint64_t got, uint64_t want) {
    if (got != want) {
        printf("  FAIL %-24s got=0x%llx want=0x%llx\n", what,
               (unsigned long long)got, (unsigned long long)want);
        ++failures;
    }
}

// Stand-ins for the process globals the pre-fix report read, set to the *second* fault's values
// before that fault is captured. A capture that consults shared state instead of its own ucontext
// lands these in the first context, and section 3 catches it.
static uint64_t g_poison_rip = 0xBAD0BAD0BAD0BAD0ull;
static uint64_t g_poison_reg = 0xDEADDEADDEADDEADull;

int main() {
    printf("== test_fault_context ==\n");

    // Two distinct fault contexts, standing in for two threads faulting milliseconds apart.
    ucontext_t uc_a{}, uc_b{};
#ifdef __APPLE__
    // Darwin's ucontext_t holds a POINTER to the machine context (`uc_mcontext->__ss`), unlike
    // Linux's inline `uc_mcontext.gregs[]`. A value-initialized ucontext_t therefore has a null
    // uc_mcontext here, and PROSPER_GREGS would dereference it. The kernel always supplies one at a
    // real fault; the test has to.
    std::remove_pointer<mcontext_t>::type mc_a{}, mc_b{};
    uc_a.uc_mcontext = &mc_a;
    uc_b.uc_mcontext = &mc_b;
#endif
    auto set = [](ucontext_t& uc, uint64_t base) {
        auto g = PROSPER_GREGS(&uc);
        g[REG_RIP] = (greg_t)(base + 0x01); g[REG_RBP] = (greg_t)(base + 0x02);
        g[REG_RSP] = (greg_t)(base + 0x03); g[REG_RAX] = (greg_t)(base + 0x04);
        g[REG_RBX] = (greg_t)(base + 0x05); g[REG_RCX] = (greg_t)(base + 0x06);
        g[REG_RDX] = (greg_t)(base + 0x07); g[REG_RSI] = (greg_t)(base + 0x08);
        g[REG_RDI] = (greg_t)(base + 0x09); g[REG_R8]  = (greg_t)(base + 0x0a);
        g[REG_R9]  = (greg_t)(base + 0x0b); g[REG_R10] = (greg_t)(base + 0x0c);
        g[REG_R11] = (greg_t)(base + 0x0d); g[REG_R12] = (greg_t)(base + 0x0e);
        g[REG_R13] = (greg_t)(base + 0x0f); g[REG_R14] = (greg_t)(base + 0x10);
        g[REG_R15] = (greg_t)(base + 0x11);
    };
    const uint64_t base_a = 0x0000414100000000ull;   // "thread A"
    const uint64_t base_b = 0x0000424200000000ull;   // "thread B"
    set(uc_a, base_a);
    set(uc_b, base_b);

    auto verify = [](const char* who, const FaultContext& f, uint64_t base,
                     uint64_t addr, int sig, long tid) {
        printf(" %s\n", who);
        check_eq("addr", f.addr, addr);
        check_eq("sig",  (uint64_t)f.sig, (uint64_t)sig);
        check_eq("tid",  (uint64_t)f.tid, (uint64_t)tid);
        check_eq("rip", f.rip, base + 0x01); check_eq("rbp", f.rbp, base + 0x02);
        check_eq("rsp", f.rsp, base + 0x03); check_eq("rax", f.rax, base + 0x04);
        check_eq("rbx", f.rbx, base + 0x05); check_eq("rcx", f.rcx, base + 0x06);
        check_eq("rdx", f.rdx, base + 0x07); check_eq("rsi", f.rsi, base + 0x08);
        check_eq("rdi", f.rdi, base + 0x09); check_eq("r8",  f.r8,  base + 0x0a);
        check_eq("r9",  f.r9,  base + 0x0b); check_eq("r10", f.r10, base + 0x0c);
        check_eq("r11", f.r11, base + 0x0d); check_eq("r12", f.r12, base + 0x0e);
        check_eq("r13", f.r13, base + 0x0f); check_eq("r14", f.r14, base + 0x10);
        check_eq("r15", f.r15, base + 0x11);
    };

    // 1. A capture carries its own ucontext's registers.
    const FaultContext fa = capture_fault_context(11, 4242, (void*)0x30016000ull, &uc_a);
    verify("A: captured from its own ucontext", fa, base_a, 0x30016000ull, 11, 4242);

    // 2. The interleave. "Thread B" faults and, as the handler does, rewrites every shared global
    //    before A's report has finished printing. A's snapshot must be untouched — this is the
    //    property whose absence produced two rbp values inside one report.
    g_poison_rip = base_b + 0x01;
    g_poison_reg = base_b + 0x04;
    const FaultContext fb = capture_fault_context(11, 4243, (void*)0x2000e2495044ull, &uc_b);
    verify("B: captured from its own ucontext", fb, base_b, 0x2000e2495044ull, 11, 4243);
    printf(" A after B's fault (must be unchanged)\n");
    verify("A: still its own", fa, base_a, 0x30016000ull, 11, 4242);

    // 3. No field of A may carry a sentinel or any of B's values. Stated separately from the
    //    exact-value checks above so a partially-correct capture — right for most registers, reading
    //    shared state for one — is named as such rather than reported as a single field mismatch.
    const uint64_t fields[] = { fa.addr, fa.rip, fa.rbp, fa.rsp, fa.rax, fa.rbx, fa.rcx, fa.rdx,
                                fa.rsi, fa.rdi, fa.r8, fa.r9, fa.r10, fa.r11, fa.r12, fa.r13,
                                fa.r14, fa.r15 };
    for (uint64_t v : fields) {
        if (v == 0xBAD0BAD0BAD0BAD0ull || v == 0xDEADDEADDEADDEADull ||
            v == g_poison_rip || v == g_poison_reg) {
            printf("  FAIL A carries a value from B's fault / a poisoned global: 0x%llx\n",
                   (unsigned long long)v);
            ++failures;
        }
    }

    // 4. A null ucontext must not be dereferenced: the identity fields still describe the fault.
    const FaultContext fn = capture_fault_context(4, 7, (void*)0x1ull, nullptr);
    check_eq("null-uc addr", fn.addr, 0x1ull);
    check_eq("null-uc sig", (uint64_t)fn.sig, 4u);
    check_eq("null-uc rip", fn.rip, 0u);

    // 5. The report gate. A coherent snapshot per fault is only half of it: two coherent reports
    //    written concurrently still arrive spliced. Exactly one thread may own the full dump, and a
    //    fault raised *inside* the dump (same tid) must not be locked out of its own report — that
    //    would truncate the record at the point it got interesting. A gate that always grants
    //    (the pre-fix behavior: no gate at all) fails the second assertion.
    printf(" report gate\n");
    {
        std::atomic<long> owner{0};
        long who = -1;
        if (!prosper::host::claim_fault_report(owner, 4242, &who)) {
            printf("  FAIL first claimant was refused\n"); ++failures;
        }
        check_eq("gate: owner after first claim", (uint64_t)who, 4242u);
        who = -1;
        if (prosper::host::claim_fault_report(owner, 4243, &who)) {
            printf("  FAIL a second thread was granted the dump\n"); ++failures;
        }
        check_eq("gate: loser is told the owner", (uint64_t)who, 4242u);
        if (!prosper::host::claim_fault_report(owner, 4242, &who)) {
            printf("  FAIL the owner was refused re-entry into its own dump\n"); ++failures;
        }
        // A non-owner must not be able to free the gate out from under the owner.
        prosper::host::release_fault_report(owner, 4243);
        check_eq("gate: a non-owner cannot release", (uint64_t)owner.load(), 4242u);
        // And once the owner releases, a LATER fault gets a full report of its own. Under
        // PROSPER_WORKER_PARK a fatal fault does not end the process, so those arrive seconds apart
        // with no concurrency; holding ownership for the life of the process would answer them with
        // "someone else is reporting" about a report that finished long ago.
        prosper::host::release_fault_report(owner, 4242);
        check_eq("gate: released", (uint64_t)owner.load(), 0u);
        if (!prosper::host::claim_fault_report(owner, 4244, &who)) {
            printf("  FAIL a later fault was refused its own report after the gate was released\n");
            ++failures;
        }
        check_eq("gate: later owner", (uint64_t)who, 4244u);
    }

    // 6. The gate must be ATOMIC, not merely first-come. Section 5 is sequential and would pass for
    //    a plain `if (owner == 0) owner = tid;`, which is exactly the shape that loses a race. Every
    //    thread claims the same gate at once; precisely one may win, and the winner's tid must be
    //    what every loser was told.
    printf(" report gate under contention\n");
    {
        constexpr int kThreads = 16;
        // 200 rounds, not thousands. The discriminator is deterministic — every thread claims with
        // a distinct tid, so exactly one can satisfy the predicate under any interleaving — and the
        // non-atomic mutant `if (owner == 0) owner = tid;` fails in round 0. The extra rounds bought
        // no signal and cost 16 thread spawn/joins each in a suite other lanes wait on. Still well
        // above 1, so this remains a race test rather than a second sequential check.
        constexpr int kRounds = 200;
        std::atomic<int> total_winners{0};
        std::atomic<int> bad_reports{0};
        for (int round = 0; round < kRounds; round++) {
            std::atomic<long> owner{0};
            std::atomic<int> winners{0};
            std::atomic<int> ready{0};
            std::atomic<bool> go{false};
            std::vector<std::thread> ts;
            for (int i = 0; i < kThreads; i++) {
                ts.emplace_back([&, i] {
                    ready.fetch_add(1);
                    // Spin rather than sleep so every thread reaches the claim at the same
                    // instant; yield so a 16-thread round cannot monopolise fewer cores.
                    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                    long who = -1;
                    const long tid = 1000 + i;
                    if (prosper::host::claim_fault_report(owner, tid, &who)) {
                        winners.fetch_add(1);
                        if (who != tid) bad_reports.fetch_add(1);   // winner told the wrong owner
                    } else if (who == 0 || who == tid) {
                        bad_reports.fetch_add(1);   // loser told "nobody" / "itself" is reporting
                    }
                });
            }
            while (ready.load() < kThreads) std::this_thread::yield();
            go.store(true, std::memory_order_release);
            for (auto& t : ts) t.join();
            const int w = winners.load();
            if (w != 1) {
                printf("  FAIL round %d had %d winners (must be exactly 1)\n", round, w);
                ++failures;
                break;
            }
            total_winners.fetch_add(w);
        }
        check_eq("contended: one winner per round", (uint64_t)total_winners.load(),
                 (uint64_t)kRounds);
        check_eq("contended: every claimant told the true owner", (uint64_t)bad_reports.load(), 0u);
    }

    printf(failures ? "FAILED (%d)\n" : "ok\n", failures);
    return failures ? 1 : 0;
}
#else
int main() { printf("== test_fault_context == skipped (posix signal context only)\n"); return 0; }
#endif
