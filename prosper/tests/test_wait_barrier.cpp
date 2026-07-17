// test_wait_barrier — the #312 WAIT_REG_MEM barrier model (command_processor.cpp).
//
// Semantics under test (with the model enabled — PROSPER_WAIT_DEFER=1, set below; the model is
// opt-in, see the verdict block in command_processor.cpp):
//   1. An UNSATISFIED WaitRegMem gates the stream's DOWNSTREAM memory effects (they must not become
//      guest-visible until the condition holds) — the exact ordering whose violation stomped
//      MallocBinned3 free-block headers in DOLL's menu content-load burst.
//   2. Effects UPSTREAM of the barrier still flush promptly (the guest CPU polls those labels).
//   3. PER-ADDRESS ORDERING DOMAINS: while gated writes are pending, a LATER submit's effect to
//      the SAME address joins the gated tail in ring order (overtaking swaps fence generations at
//      recycled labels — the #312 stomp), and a later WAIT on a gated address evaluates in ring
//      order (after the gated writes it must observe) — but effects to untouched addresses FLOW
//      (gating them created a CPU<->GPU circular stall, an early-boot wedge measured live).
//   4. A Jump executed inside a paused stream must NOT un-gate the parent's remaining effects
//      (the recursive fold once reset the pause flag — the WAIT_DEFER wedge/ordering bug), and the
//      jump target's own memory effects are gated too.
//   5. A barrier that never satisfies releases via the bounded timeout (liveness backstop), so a
//      guest polling a gated label can wait at most one timeout.
//   6. A SATISFIED wait is a pass-through no-op (the fast path every healthy frame takes).
#include "../src/gpu/command_processor.hpp"
#include "../src/gpu/gpu_execute.hpp"
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
static void emit_dma_copy(uint32_t* buf, uint64_t dst, uint64_t src, uint32_t bytes) {
    buf[0] = PM4(7, IT_NOP, R_DMA_DATA);
    buf[1] = (uint32_t)dst; buf[2] = (uint32_t)(dst >> 32);
    buf[3] = (uint32_t)src; buf[4] = (uint32_t)(src >> 32);
    buf[5] = bytes; buf[6] = 0;
}

// Fold + drain the pipe-drain queue (upstream effects become guest-visible at a drain point).
static size_t run_cb(const uint32_t* buf, size_t dwords, GpuState& st) {
    size_t n = run_command_buffer(buf, dwords, st);
    prosper_gpu_drain_completion_writes();
    return n;
}

int main() {
    // Enable the (opt-in) model, and set a generous release timeout so the gating assertions
    // below are not raced by the liveness backstop (test 5 sleeps past it deliberately). Both
    // must be set before the first fold caches them.
#ifdef _WIN32
    _putenv_s("PROSPER_WAIT_DEFER", "1");
    _putenv_s("PROSPER_WAIT_TIMEOUT_MS", "400");
#else
    setenv("PROSPER_WAIT_DEFER", "1", 1);
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

    // 3: per-address ordering domains — while L has a gated pending write:
    //    - a later submit's write to L joins the tail (ring order; final value = LAST write),
    //    - a later submit's WAIT on L evaluates in ring order (sees the gated writes),
    //    - a later submit's write to an UNRELATED address flows immediately (no circular stall).
    {
        volatile uint64_t cond1 = 0;
        uint64_t L = 0, M = 0, N = 0;
        uint32_t s1[8 + 7], s2[7], s3[7], s4[8 + 7];
        emit_wait_eq(s1, (uint64_t)(uintptr_t)&cond1, 1);
        emit_release(s1 + 8, (uint64_t)(uintptr_t)&L, 1);   // gated: L <- 1
        emit_release(s2, (uint64_t)(uintptr_t)&L, 2);       // later write to L: must NOT overtake
        emit_release(s3, (uint64_t)(uintptr_t)&M, 1);       // unrelated address: must FLOW
        emit_wait_eq(s4, (uint64_t)(uintptr_t)&L, 2);       // wait on gated L: ring order (sees L=2)
        emit_release(s4 + 8, (uint64_t)(uintptr_t)&N, 1);
        GpuState st;
        run_cb(s1, 15, st);
        run_cb(s2, 7, st);
        run_cb(s3, 7, st);
        run_cb(s4, 15, st);
        flush_deferred_streams();
        CHECK(L == 0, "gated-address writes stay gated (no overtake)");
        CHECK(M == 1, "unrelated-address write flowed immediately");
        CHECK(N == 0, "stream behind a gated-address wait stays gated");
        CHECK(deferred_pending(), "tail pending");
        cond1 = 1;                                   // front producer arrives -> tail drains in order
        flush_deferred_streams();
        CHECK(L == 2, "same-address writes landed in ring order (final = later write)");
        CHECK(N == 1, "gated-address wait evaluated in ring order and released its stream");
        CHECK(!deferred_pending(), "queue fully drained");
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

    // Address DMA cannot be released through the legacy deferred-effect path: that would drop it
    // from ordered execution/capture metadata and bypass authoritative renderer source reads. Its
    // same-stream completion suffix must also be discarded, never signaling work that was rejected.
    {
        volatile uint64_t cond = 0;
        uint64_t source = 0x123456789ABCDEF0ull, target = 0, completion = 0;
        uint32_t stream[8 + 7 + 7];
        emit_wait_eq(stream, (uint64_t)(uintptr_t)&cond, 1);
        emit_dma_copy(stream + 8, (uint64_t)(uintptr_t)&target,
                      (uint64_t)(uintptr_t)&source, sizeof(source));
        emit_release(stream + 15, (uint64_t)(uintptr_t)&completion, 1);
        GpuState st;
        run_cb(stream, 22, st);
        CHECK(st.dma_copies.size() == 1 && st.dma_execution_rejected,
              "WAIT_DEFER-gated address DMA remains visible and rejects execution");
        cond = 1;
        flush_deferred_streams();
        execute_nonrender_submit_work(st);
        CHECK(target == 0 && completion == 0,
              "rejected gated DMA discards its deferred completion suffix without signaling");
    }

    // A copy also depends on its source. A prior gated producer to S must prevent a later
    // address DMA(S->D) from reading stale S even when D itself is in an unrelated domain.
    {
        volatile uint64_t cond = 0;
        uint64_t source = 0, target = 0;
        uint32_t producer[8 + 7];
        emit_wait_eq(producer, (uint64_t)(uintptr_t)&cond, 1);
        emit_release(producer + 8, (uint64_t)(uintptr_t)&source, 0x55AAu);
        GpuState producer_state;
        run_cb(producer, 15, producer_state);

        uint32_t copy[7];
        emit_dma_copy(copy, (uint64_t)(uintptr_t)&target,
                      (uint64_t)(uintptr_t)&source, sizeof(source));
        GpuState copy_state;
        run_cb(copy, 7, copy_state);
        CHECK(copy_state.dma_copies.size() == 1 && copy_state.dma_execution_rejected,
              "address DMA rejects a dependency on a previously gated source range");
        CHECK(target == 0, "source-dependent DMA cannot overtake the gated producer");
        cond = 1;
        flush_deferred_streams();
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
