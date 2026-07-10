// test_wait_barrier — the #312 per-queue WAIT_REG_MEM barrier model (command_processor.cpp).
//
// Semantics under test (default configuration — the model is ON by default):
//   1. An UNSATISFIED WaitRegMem gates the stream's DOWNSTREAM memory effects (they must not become
//      guest-visible until the condition holds) — the exact ordering whose violation stomped
//      MallocBinned3 free-block headers in DOLL's menu content-load burst.
//   2. Effects UPSTREAM of the barrier still flush promptly (the guest CPU polls those labels).
//   3. Releasing is per-queue INDEPENDENT: a blocked stream never head-of-line blocks another
//      deferred stream whose own condition is satisfied.
//   4. A Jump executed inside a paused stream must NOT un-gate the parent's remaining effects
//      (the recursive fold once reset the pause flag — the WAIT_DEFER wedge/ordering bug), and the
//      jump target's own memory effects are gated too.
//   5. A barrier that never satisfies releases via the bounded timeout (liveness backstop), so a
//      guest polling a gated label can wait at most one timeout.
//   6. A SATISFIED wait is a pass-through no-op (the fast path every healthy frame takes).
#include "../src/gpu/command_processor.hpp"
#include "../src/gpu/pm4_decode.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <chrono>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// PM4 type-3 NOP-wrapped header, identical to hle_agc.cpp PM4(): len = total dwords incl. header.
static uint32_t PM4(uint32_t len, uint32_t op, uint32_t r) {
    return 0xC0000000u | (((len - 2u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) | ((r & (R_NUM - 1u)) << 2u);
}

// Emit a ReleaseMem(label <- value, data_sel=1) at buf: 7 dwords.
static void emit_release(uint32_t* buf, uint64_t addr, uint32_t value) {
    buf[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
    buf[1] = (uint32_t)addr; buf[2] = (uint32_t)(addr >> 32);
    buf[3] = 1;                      // data_sel = Data32Low
    buf[4] = value; buf[5] = 0;
    buf[6] = 0x2d;                   // event action (as DOLL's consumed-marker fences carry)
}
// Emit a WaitRegMem([addr]&mask == ref) at buf: 8 dwords.
static void emit_wait_eq(uint32_t* buf, uint64_t addr, uint64_t ref) {
    buf[0] = PM4(8, IT_NOP, R_WAIT_MEM_64);
    buf[1] = (uint32_t)addr; buf[2] = (uint32_t)(addr >> 32);
    buf[3] = 0xffffffffu; buf[4] = 0;               // mask (low 32)
    buf[5] = (uint32_t)ref; buf[6] = (uint32_t)(ref >> 32);
    buf[7] = 3;                                     // compare function: ==
}

// Fold + drain the pipe-drain queue (upstream effects become guest-visible at a drain point).
static size_t run_cb(const uint32_t* buf, size_t dwords, GpuState& st) {
    size_t n = run_command_buffer(buf, dwords, st);
    prosper_gpu_drain_completion_writes();
    return n;
}

int main() {
    // A generous release timeout so the gating assertions below are not raced by the liveness
    // backstop (test 5 sleeps past it deliberately). Must be set before the first fold caches it.
#ifdef _WIN32
    _putenv_s("PROSPER_WAIT_TIMEOUT_MS", "400");
#else
    setenv("PROSPER_WAIT_TIMEOUT_MS", "400", 1);
#endif
    printf("== test_wait_barrier ==\n");

    // 1+2: [ReleaseMem(pre <- 1)] [WaitRegMem(cond==1)] [ReleaseMem(post <- 1)]
    {
        volatile uint64_t cond = 0;                 // the awaited label (unsatisfied)
        uint64_t pre = 0, post = 0;
        uint32_t buf[7 + 8 + 7];
        emit_release(buf, (uint64_t)(uintptr_t)&pre, 1);
        emit_wait_eq(buf + 7, (uint64_t)(uintptr_t)&cond, 1);
        emit_release(buf + 15, (uint64_t)(uintptr_t)&post, 1);
        GpuState st;
        size_t n = run_cb(buf, 22, st);
        CHECK(n == 3, "stream decoded as 3 packets");
        CHECK(last_fold_deferred(), "unsatisfied WaitRegMem paused the stream");
        CHECK(deferred_pending(), "a deferred stream is pending");
        CHECK(pre == 1, "effect UPSTREAM of the barrier flushed promptly");
        flush_deferred_streams();
        CHECK(post == 0, "effect DOWNSTREAM of the barrier stays gated while the condition is unmet");
        cond = 1;                                    // producer arrives (CPU-visible label write)
        flush_deferred_streams();
        CHECK(post == 1, "downstream effect released once the condition was satisfied");
        CHECK(!deferred_pending(), "stream completed and left the deferred queue");
    }

    // 3: per-queue independence — two paused streams; the SECOND stream's condition satisfies
    // first, and its write must release even though an older stream is still blocked.
    {
        volatile uint64_t cond1 = 0, cond2 = 0;
        uint64_t l1 = 0, l2 = 0;
        uint32_t s1[8 + 7], s2[8 + 7];
        emit_wait_eq(s1, (uint64_t)(uintptr_t)&cond1, 1);
        emit_release(s1 + 8, (uint64_t)(uintptr_t)&l1, 1);
        emit_wait_eq(s2, (uint64_t)(uintptr_t)&cond2, 1);
        emit_release(s2 + 8, (uint64_t)(uintptr_t)&l2, 1);
        GpuState st;
        run_cb(s1, 15, st);
        run_cb(s2, 15, st);
        cond2 = 1;                                   // ONLY the younger stream's producer arrives
        flush_deferred_streams();
        CHECK(l2 == 1, "younger stream released independently (no head-of-line blocking)");
        CHECK(l1 == 0, "older blocked stream stays gated");
        CHECK(deferred_pending(), "older stream still pending");
        cond1 = 1;
        flush_deferred_streams();
        CHECK(l1 == 1 && !deferred_pending(), "older stream released once its own condition held");
    }

    // 4: a Jump inside a paused stream — the jump target's ReleaseMem must be gated with the
    // parent stream, and the parent's post-jump effects must STAY gated (the recursion once reset
    // the pause flag, un-gating them).
    {
        volatile uint64_t cond = 0;
        uint64_t in_jump = 0, post = 0;
        uint32_t seg[7];                             // the jump target: one ReleaseMem
        emit_release(seg, (uint64_t)(uintptr_t)&in_jump, 1);
        uint32_t buf[8 + 5 + 7];
        emit_wait_eq(buf, (uint64_t)(uintptr_t)&cond, 1);
        uint64_t seg_addr = (uint64_t)(uintptr_t)seg;
        buf[8]  = PM4(5, IT_NOP, R_JUMP);
        buf[9]  = (uint32_t)seg_addr; buf[10] = (uint32_t)(seg_addr >> 32);
        buf[11] = 7;                                 // dword count of the target segment
        buf[12] = 0;                                 // not predicated
        emit_release(buf + 13, (uint64_t)(uintptr_t)&post, 1);
        GpuState st;
        run_cb(buf, 20, st);
        flush_deferred_streams();
        CHECK(in_jump == 0, "jump-target effect gated with the paused parent stream");
        CHECK(post == 0, "parent's post-jump effect STAYS gated (recursion must not reset the pause)");
        CHECK(last_fold_deferred(), "fold still reports deferred after an inner Jump");
        cond = 1;
        flush_deferred_streams();
        CHECK(in_jump == 1 && post == 1, "both released in order once the condition held");
    }

    // 5: liveness backstop — a condition NOBODY ever satisfies releases via the bounded timeout.
    {
        volatile uint64_t cond = 0;                  // never written
        uint64_t label = 0;
        uint32_t buf[8 + 7];
        emit_wait_eq(buf, (uint64_t)(uintptr_t)&cond, 1);
        emit_release(buf + 8, (uint64_t)(uintptr_t)&label, 1);
        GpuState st;
        run_cb(buf, 15, st);
        flush_deferred_streams();
        CHECK(label == 0, "gated before the timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(450));   // > PROSPER_WAIT_TIMEOUT_MS
        flush_deferred_streams();
        CHECK(label == 1, "timeout released the gated write (liveness backstop)");
        CHECK(!deferred_pending(), "timed-out stream completed");
    }

    // 6: a SATISFIED wait is a pass-through — nothing defers (the healthy-frame fast path).
    {
        volatile uint64_t cond = 1;
        uint64_t label = 0;
        uint32_t buf[8 + 7];
        emit_wait_eq(buf, (uint64_t)(uintptr_t)&cond, 1);
        emit_release(buf + 8, (uint64_t)(uintptr_t)&label, 1);
        GpuState st;
        run_cb(buf, 15, st);
        CHECK(!last_fold_deferred(), "satisfied wait does not pause the stream");
        CHECK(!deferred_pending(), "no deferred stream created");
        CHECK(label == 1, "downstream effect flushed promptly");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
