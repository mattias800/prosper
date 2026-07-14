// test_eop_write — Stage B of docs/GPU_EXECUTOR_DESIGN.md: the CommandProcessor honors the Dcb's
// memory-side effects. Because our fold is synchronous (submit == pipe drain), a RELEASE_MEM /
// EVENT_WRITE_EOP label write and a WRITE_DATA are performed at apply time — correct end-of-pipe
// semantics, not a shim. This hand-builds the exact packet layout hle_agc.cpp's builders emit (the
// builders themselves read SysV stack args via __builtin_frame_address, only valid under the loaded
// game) and asserts run_command_buffer writes the right bytes to the target address.
#include "../src/gpu/command_processor.hpp"
#include "../src/gpu/mb3_freelist.hpp"
#include "../src/gpu/pm4_decode.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper::gpu;

// The guest TSC clock the GPU EOP timestamp shares (hle_kernel_time.cpp) — same source as
// sceKernelReadTsc, so a GPU fence timestamp and a CPU TSC read lie on one timeline (#156).
extern "C" uint64_t prosper_guest_tsc_ns();
extern "C" void prosper_label_hist_dma_built(uint64_t addr, uint64_t cb, uint32_t src, uint8_t builder);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// PM4 type-3 NOP-wrapped header, identical to hle_agc.cpp PM4(): len = total dwords incl. header.
static uint32_t PM4(uint32_t len, uint32_t op, uint32_t r) {
    return 0xC0000000u | (((len - 2u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) | ((r & (R_NUM - 1u)) << 2u);
}

// Fold + drain: completion writes ride the modeled pipe-drain queue (#312) and become
// guest-visible at a drain point — tests assert the guest-visible (post-drain) state.
static size_t run_cb(const uint32_t* buf, size_t dwords, GpuState& st) {
    size_t n = run_command_buffer(buf, dwords, st);
    prosper_gpu_drain_completion_writes();
    return n;
}

int main() {
    printf("== test_eop_write ==\n");

    // A RELEASE_MEM writing a 64-bit fence value (data_sel==2) to a label, exactly as agc_cb_release_mem
    // lays it out: [0]=hdr [1..2]=addr [3]=data_sel [4..5]=value lo/hi [6]=action.
    {
        uint64_t label = 0;
        uint32_t buf[7];
        uint64_t addr = (uint64_t)(uintptr_t)&label;
        buf[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
        buf[1] = (uint32_t)(addr & 0xffffffffu); buf[2] = (uint32_t)(addr >> 32);
        buf[3] = 2;                                   // data_sel = Data64
        buf[4] = 0xF00DBEEFu; buf[5] = 0x12345678u;   // value = 0x12345678F00DBEEF
        buf[6] = 0x04;                                // event action
        GpuState st;
        size_t n = run_cb(buf, 7, st);
        CHECK(n == 1, "RELEASE_MEM decoded as one packet");
        CHECK(label == 0x12345678F00DBEEFull, "RELEASE_MEM (data_sel=2) wrote the 64-bit fence value to the label");
    }

    // data_sel==1 -> 32-bit write; the upper 32 bits of the label must be untouched.
    {
        uint64_t label = 0xAAAAAAAAAAAAAAAAull;
        uint32_t buf[7];
        uint64_t addr = (uint64_t)(uintptr_t)&label;
        buf[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
        buf[1] = (uint32_t)(addr & 0xffffffffu); buf[2] = (uint32_t)(addr >> 32);
        buf[3] = 1;                                   // data_sel = Data32Low
        buf[4] = 0x0000CAFEu; buf[5] = 0;
        buf[6] = 0x04;
        GpuState st; run_cb(buf, 7, st);
        CHECK((label & 0xffffffffu) == 0x0000CAFEu, "RELEASE_MEM (data_sel=1) wrote the low 32 bits");
        CHECK((label >> 32) == 0xAAAAAAAAu, "RELEASE_MEM (data_sel=1) left the upper 32 bits untouched");
    }

    // data_sel==3 -> monotonic GPU clock: two successive writes must not decrease.
    {
        uint64_t l1 = 0, l2 = 0;
        auto build = [](uint32_t* buf, uint64_t addr) {
            buf[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
            buf[1] = (uint32_t)(addr & 0xffffffffu); buf[2] = (uint32_t)(addr >> 32);
            buf[3] = 3; buf[4] = 0; buf[5] = 0; buf[6] = 0x04;
        };
        uint32_t b1[7], b2[7];
        build(b1, (uint64_t)(uintptr_t)&l1); build(b2, (uint64_t)(uintptr_t)&l2);
        GpuState st; run_cb(b1, 7, st); run_cb(b2, 7, st);
        CHECK(l1 != 0 && l2 >= l1, "RELEASE_MEM (data_sel=3) wrote a monotonic non-zero GPU clock");
    }

    // #156: the GPU EOP timestamp must share the guest TSC timeline (sceKernelReadTsc) — on real
    // hardware they are the SAME counter. A data_sel=3 fence value must fall between a TSC read
    // taken just before and just after the submit (the old steady_clock had a disjoint epoch, so
    // it would land far outside this window).
    {
        uint64_t label = 0;
        uint32_t buf[7];
        uint64_t addr = (uint64_t)(uintptr_t)&label;
        buf[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
        buf[1] = (uint32_t)(addr & 0xffffffffu); buf[2] = (uint32_t)(addr >> 32);
        buf[3] = 3; buf[4] = 0; buf[5] = 0; buf[6] = 0x04;
        uint64_t before = prosper_guest_tsc_ns();
        GpuState st; run_cb(buf, 7, st);
        uint64_t after = prosper_guest_tsc_ns();
        CHECK(label >= before && label <= after,
              "data_sel=3 fence lies on the sceKernelReadTsc timeline (shared guest TSC, not steady_clock)");
    }

    // WRITE_DATA: [0]=hdr [1]=dst [2..3]=addr [4]=num_dwords [5..]=data.
    {
        uint32_t target[3] = {0, 0, 0};
        uint32_t buf[8];
        uint64_t addr = (uint64_t)(uintptr_t)target;
        buf[0] = PM4(8, IT_NOP, R_WRITE_DATA);
        buf[1] = 0;                                   // dst
        buf[2] = (uint32_t)(addr & 0xffffffffu); buf[3] = (uint32_t)(addr >> 32);
        buf[4] = 3;                                   // num_dwords
        buf[5] = 0x11111111u; buf[6] = 0x22222222u; buf[7] = 0x33333333u;
        GpuState st; run_cb(buf, 8, st);
        CHECK(target[0] == 0x11111111u && target[1] == 0x22222222u && target[2] == 0x33333333u,
              "WRITE_DATA copied all 3 inline dwords to the destination");
    }

    // A WRITE_DATA claiming more dwords than the packet holds must clamp (never read past the packet).
    {
        uint32_t target[4] = {0, 0, 0, 0};
        uint32_t buf[7];
        uint64_t addr = (uint64_t)(uintptr_t)target;
        buf[0] = PM4(7, IT_NOP, R_WRITE_DATA);
        buf[1] = 0; buf[2] = (uint32_t)(addr & 0xffffffffu); buf[3] = (uint32_t)(addr >> 32);
        buf[4] = 99;                                  // lies: claims 99 dwords, only 2 present (buf[5],buf[6])
        buf[5] = 0xDEADu; buf[6] = 0xBEEFu;
        GpuState st; run_cb(buf, 7, st);
        CHECK(target[0] == 0xDEADu && target[1] == 0xBEEFu && target[2] == 0 && target[3] == 0,
              "WRITE_DATA clamped num_dwords to what the packet actually holds");
    }

    // Address-carrying EVENT_WRITE (#132): the widened packet ([0]=hdr [1]=event_type [2..3]=addr)
    // writes a monotonic completion value to its address so a guest waiting on that label unblocks
    // (the old 2-dword packet discarded the address -> the CommandProcessor no-op'd -> wait forever).
    {
        uint64_t label = 0;
        uint32_t buf[4];
        uint64_t addr = (uint64_t)(uintptr_t)&label;
        buf[0] = PM4(4, IT_EVENT_WRITE, 0);
        buf[1] = 0x14;                                // some event_type
        buf[2] = (uint32_t)(addr & 0xffffffffu); buf[3] = (uint32_t)(addr >> 32);
        GpuState st; run_cb(buf, 4, st);
        CHECK(label != 0, "address-carrying EVENT_WRITE wrote a completion value to the label (was dropped)");
    }
    // An address-LESS EVENT_WRITE (event_addr == 0, a pipeline-sync event) must remain a no-op.
    {
        uint32_t buf[4];
        buf[0] = PM4(4, IT_EVENT_WRITE, 0);
        buf[1] = 0x16; buf[2] = 0; buf[3] = 0;        // no address
        GpuState st;
        size_t n = run_cb(buf, 4, st);    // must not fault / write anywhere
        CHECK(n == 1, "address-less EVENT_WRITE decodes and is a harmless no-op");
    }

    // #312: learn a per-thread MB3 pool array from pthread TLS and detect a 0x20-byte block in both
    // of size-class idx=1's freelists. The descriptor address is dynamic; no DOLL address is baked in.
    // Real MB3 pool arrays are 64 KiB allocations, so mb3_note_tls_pool_candidate only accepts a
    // 64-KiB-aligned base ((base & 0xffff) == 0). alignas(0x10000) on a STATIC can't guarantee that:
    // linkers cap static alignment below 64 KiB (macOS reduces __bss to 4 KiB, MinGW rejects > 8 KiB),
    // so the buffer may land off a 64-KiB boundary and the candidate is silently dropped — the four MB3
    // membership subtests then fail on macOS/Windows while passing on Linux. Over-allocate and carve a
    // 64-KiB-aligned 64-KiB window at runtime, which is honored everywhere. Production MB3 logic is
    // unchanged; only the test's backing memory is.
    static uint8_t mb3_pool_backing[0x20000] = {};
    uint8_t* mb3_pool = (uint8_t*)(((uintptr_t)mb3_pool_backing + 0xffffull) & ~0xffffull);
    struct alignas(0x20) FreeNode { uint64_t next; uint64_t pad[3]; };
    static FreeNode free_label{}, free_tail{}, secondary_label{}, live_label{};
    uint64_t* bin = (uint64_t*)(mb3_pool + 0x20);
    mb3_reset_pool_candidates_for_test();
    mb3_note_tls_pool_candidate((uint64_t)(uintptr_t)mb3_pool);
    free_label.next = (uint64_t)(uintptr_t)&free_tail;
    free_tail.next = 0;
    bin[0] = (uint64_t)(uintptr_t)&free_label; bin[1] = 2;
    bin[2] = (uint64_t)(uintptr_t)&secondary_label; bin[3] = 1;
    Mb3FreelistMatch match{};
    CHECK(mb3_freelist_contains_stable((uint64_t)(uintptr_t)&free_tail, &match) &&
          match.list == 1 && match.hops == 1,
          "MB3 membership walks the dynamic primary size-class freelist");
    CHECK(mb3_freelist_contains_stable((uint64_t)(uintptr_t)&secondary_label, &match) &&
          match.list == 2,
          "MB3 membership checks the secondary size-class freelist");
    CHECK(!mb3_freelist_contains_stable((uint64_t)(uintptr_t)&live_label, nullptr),
          "an allocated block absent from both freelists is not classified free");

    // The free label's DmaData(:=0) must be skipped before it erases NextFreeBlock. Then simulate an
    // allocator pop/reuse before ReleaseMem(<-1): the one-generation debt must still suppress that
    // old fence even though direct membership has become false.
    {
        uint64_t addr = (uint64_t)(uintptr_t)&free_label;
        prosper_label_hist_dma_built(addr, 0x1000, 0, 1);
        uint32_t dma[7] = {};
        dma[0] = PM4(7, IT_NOP, R_DMA_DATA);
        dma[1] = (uint32_t)addr; dma[2] = (uint32_t)(addr >> 32);
        dma[3] = 0; dma[4] = 0; dma[5] = 4; dma[6] = 0x303;
        uint64_t next_before = free_label.next;
        GpuState st; run_cb(dma, 7, st);
        CHECK(free_label.next == next_before,
              "DmaData zero-init does not overwrite a block currently on the MB3 freelist");

        bin[0] = 0; bin[1] = 0;                    // allocator popped/reused it after the skipped init
        free_label.next = 0x1122334455667788ull;   // observable data belonging to the new owner
        uint32_t rel[7] = {};
        rel[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
        rel[1] = (uint32_t)addr; rel[2] = (uint32_t)(addr >> 32);
        rel[3] = 1; rel[4] = 1; rel[5] = 0; rel[6] = 4;
        run_cb(rel, 7, st);
        CHECK(free_label.next == 0x1122334455667788ull,
              "paired ReleaseMem remains suppressed after the free block is popped/reused");
    }

    // A normal consumed-marker label that is not on a freelist must retain the real protocol: init
    // low dword to 0, then signal it to 1. This is the false-positive/liveness side of the guard.
    {
        bin[2] = 0; bin[3] = 0;
        live_label.next = 0xAABBCCDDFFFFFFFFull;
        uint64_t addr = (uint64_t)(uintptr_t)&live_label;
        prosper_label_hist_dma_built(addr, 0x2000, 0, 1);
        uint32_t stream[14] = {};
        stream[0] = PM4(7, IT_NOP, R_DMA_DATA);
        stream[1] = (uint32_t)addr; stream[2] = (uint32_t)(addr >> 32);
        stream[3] = 0; stream[4] = 0; stream[5] = 4; stream[6] = 0x303;
        stream[7] = PM4(7, IT_NOP, R_RELEASE_MEM);
        stream[8] = (uint32_t)addr; stream[9] = (uint32_t)(addr >> 32);
        stream[10] = 1; stream[11] = 1; stream[12] = 0; stream[13] = 4;
        GpuState st; run_cb(stream, 14, st);
        CHECK((uint32_t)live_label.next == 1,
              "live consumed-marker labels still execute DmaData(0) then ReleaseMem(1)");
    }
    mb3_reset_pool_candidates_for_test();

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
