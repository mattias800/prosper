// test_eop_write — Stage B of docs/GPU_EXECUTOR_DESIGN.md: the CommandProcessor honors the Dcb's
// memory-side effects. RELEASE_MEM / EVENT_WRITE_EOP label writes ride a completion queue and a
// WRITE_DATA is applied at the renderer/drain boundary — correct end-of-pipe semantics, not a shim.
// This hand-builds the exact packet layout hle_agc.cpp's builders emit (the
// builders themselves read SysV stack args via __builtin_frame_address, only valid under the loaded
// game) and asserts run_command_buffer writes the right bytes to the target address.
#include "../src/gpu/command_processor.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/mb3_freelist.hpp"
#include "../src/gpu/pm4_decode.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>   // setenv/_putenv_s: arm the #1226-retired suppression guards for this test
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper::gpu;

// The guest TSC clock the GPU EOP timestamp shares (hle_kernel_time.cpp) — same source as
// sceKernelReadTsc, so a GPU fence timestamp and a CPU TSC read lie on one timeline (#156).
extern "C" uint64_t prosper_guest_tsc_ns();
extern "C" void prosper_label_hist_dma_built(uint64_t addr, uint64_t cb, uint32_t src, uint8_t builder);
// #1226: uncapped total of PROSPER_WRITE_TRAP payload matches (see command_processor.cpp).
extern "C" uint64_t prosper_gpu_write_trap_matches();
extern "C" void prosper_rel1_forge_suppress_all_override_for_test(int value);
extern "C" void prosper_rel1_forge_decision_reset_for_test();
extern "C" void prosper_rel1_forge_decision_totals(uint64_t* candidates, uint64_t* suppressed,
                                                     uint64_t* landed);
extern "C" bool prosper_rel1_forge_report_due_for_test(uint64_t candidates);

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
    execute_nonrender_submit_work(st);
    prosper_gpu_drain_completion_writes();
    return n;
}

int main() {
    // Exercise the SDK-13 post-submit queue policy; older callers use the eager compatibility path.
    prosper_gpu_enable_post_submit_visibility();
    // Keep an inherited developer environment from turning this unit test into the destructive
    // #1226 all-forge-suppressed arm. Individual checks opt into that arm below.
    prosper_rel1_forge_suppress_all_override_for_test(0);
    prosper_rel1_forge_decision_reset_for_test();
    printf("== test_eop_write ==\n");
    // #1226: the generation and MB3-freelist suppression families are OFF by default (their
    // content/membership premises misfire on live protocols — see generation_guard() /
    // mb3_freelist_guard() in command_processor.cpp). This test verifies the suppression
    // MECHANISMS, so arm them explicitly; the env is read once before any fold below.
#ifdef _WIN32
    _putenv_s("PROSPER_GENERATION_GUARD", "1");
    _putenv_s("PROSPER_MB3_FREELIST_GUARD", "1");
    _putenv_s("PROSPER_WRITE_TRAP", "0x5a5a0001");
#else
    setenv("PROSPER_GENERATION_GUARD", "1", 1);
    setenv("PROSPER_MB3_FREELIST_GUARD", "1", 1);
    setenv("PROSPER_WRITE_TRAP", "0x5a5a0001", 1);
#endif

    // SDK-13 completion writes must remain invisible for the entire guest submit import. The
    // synchronous renderer may consume an unrelated resource upload, but it must not expose either
    // half of an ordered label initialization -> release pair. A drain after scope end publishes the
    // pair; an early publication could let another guest thread recycle the label while the submitter
    // is still updating its allocation lists.
    {
        uint64_t label = 0xaaaaaaaa55555555ull;
        uint32_t resource = 0;
        uint32_t buf[19]{};
        const uint64_t label_addr = (uint64_t)(uintptr_t)&label;
        const uint64_t resource_addr = (uint64_t)(uintptr_t)&resource;
        buf[0] = PM4(6, IT_NOP, R_WRITE_DATA);
        buf[1] = 0;
        buf[2] = (uint32_t)label_addr; buf[3] = (uint32_t)(label_addr >> 32);
        buf[4] = 1; buf[5] = 0;                       // ordered label initialization
        buf[6] = PM4(7, IT_NOP, R_RELEASE_MEM);
        buf[7] = (uint32_t)label_addr; buf[8] = (uint32_t)(label_addr >> 32);
        buf[9] = 2;
        buf[10] = 0x76543210u; buf[11] = 0xfedcba98u;
        buf[12] = 0x04;
        buf[13] = PM4(6, IT_NOP, R_WRITE_DATA);
        buf[14] = 0;
        buf[15] = (uint32_t)resource_addr; buf[16] = (uint32_t)(resource_addr >> 32);
        buf[17] = 1; buf[18] = 0x13579bdfu;           // renderer resource upload
        GpuState st;
        prosper_gpu_submit_scope_begin();
        const size_t n = run_command_buffer(buf, 19, st);
        prosper_gpu_drain_renderer_writes();
        CHECK(n == 3 && prosper_gpu_submit_scope_active(),
              "SDK-13 submit scope retains queued label writes");
        CHECK(resource == 0x13579bdfu,
              "renderer drain applies an unrelated resource upload inside the submit");
        CHECK(label == 0xaaaaaaaa55555555ull,
              "renderer drain keeps an overlapping label init and fence private");
        // Every submit NID has a return hook, including invalid calls that return before scope_begin.
        // Model one of those rejected calls on a second guest thread while this valid submit remains
        // stalled: its unmatched hook must not consume this thread's token or expose this label.
        std::thread rejected_submit_return([] { prosper_gpu_submit_scope_end(); });
        rejected_submit_return.join();
        CHECK(prosper_gpu_submit_scope_active() && label == 0xaaaaaaaa55555555ull,
              "unmatched return hook cannot retire another thread's active submit");
        // A same-thread re-entrant tagged import opens its own scope before validation. Its return
        // checkpoint consumes only that nested token, leaving the outer invocation active.
        prosper_gpu_submit_scope_begin();
        prosper_gpu_submit_scope_end();
        CHECK(prosper_gpu_submit_scope_active() && label == 0xaaaaaaaa55555555ull,
              "matched nested return hook preserves the outer same-thread submit");
        // Model an arbitrarily descheduled submit handler after all HLE work is complete but before
        // its generated import trampoline reaches the return checkpoint. This exceeds the old 1 ms
        // grace timer and proves elapsed time cannot publish a label while the scope is still active.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(prosper_gpu_submit_scope_active() && label == 0xaaaaaaaa55555555ull,
              "completion stays private across a stalled pre-return checkpoint");
        prosper_gpu_submit_scope_end();
        prosper_gpu_drain_completion_writes();
        CHECK(label == 0xfedcba9876543210ull,
              "ordered label initialization and fence become visible after submit scope end");
    }

    // Renderer drains routinely encounter hundreds of resource uploads behind thousands of private
    // completion labels. Exercise the stable batched extraction: every unrelated write must land in
    // packet order while the overlapping label initialization/release pair remains hidden.
    {
        constexpr uint32_t resource_count = 256;
        uint64_t label = 0x123456789abcdef0ull;
        std::vector<uint32_t> resources(resource_count, 0);
        std::vector<uint32_t> stream;
        stream.reserve(13u + resource_count * 6u);
        auto append_write = [&](uint64_t addr, uint32_t value) {
            stream.push_back(PM4(6, IT_NOP, R_WRITE_DATA));
            stream.push_back(0);
            stream.push_back(static_cast<uint32_t>(addr));
            stream.push_back(static_cast<uint32_t>(addr >> 32));
            stream.push_back(1);
            stream.push_back(value);
        };
        append_write(reinterpret_cast<uint64_t>(&label), 0);
        stream.push_back(PM4(7, IT_NOP, R_RELEASE_MEM));
        stream.push_back(static_cast<uint32_t>(reinterpret_cast<uint64_t>(&label)));
        stream.push_back(static_cast<uint32_t>(reinterpret_cast<uint64_t>(&label) >> 32));
        stream.push_back(2);
        stream.push_back(0x76543210u);
        stream.push_back(0xfedcba98u);
        stream.push_back(0x04);
        for (uint32_t i = 0; i < resource_count; ++i)
            append_write(reinterpret_cast<uint64_t>(&resources[i]), 0x61000000u + i);

        GpuState st;
        prosper_gpu_submit_scope_begin();
        const size_t n = run_command_buffer(stream.data(), stream.size(), st);
        prosper_gpu_drain_renderer_writes();
        bool resources_ready = true;
        for (uint32_t i = 0; i < resource_count; ++i)
            resources_ready &= resources[i] == 0x61000000u + i;
        CHECK(n == resource_count + 2u && resources_ready,
              "batched renderer drain applies every unrelated resource write");
        CHECK(label == 0x123456789abcdef0ull,
              "batched renderer drain keeps overlapping completion writes private");
        prosper_gpu_submit_scope_end();
        prosper_gpu_drain_completion_writes();
        CHECK(label == 0xfedcba9876543210ull,
              "batched drain preserves the private label pair's completion order");
    }

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

    // #189: an address-backed DMA_DATA packet copies the exact requested byte span at submit.
    // Use deliberately unaligned byte ranges so this covers the API's byte-count contract rather
    // than accidentally depending on dword-sized data.
    {
        alignas(8) uint8_t source[24] = {};
        alignas(8) uint8_t target[24];
        for (uint32_t i = 0; i < sizeof(source); ++i) source[i] = (uint8_t)(0x30u + i);
        memset(target, 0xCC, sizeof(target));
        const uint64_t src = (uint64_t)(uintptr_t)(source + 1);
        const uint64_t dst = (uint64_t)(uintptr_t)(target + 3);
        uint32_t dma[7] = {};
        dma[0] = PM4(7, IT_NOP, R_DMA_DATA);
        dma[1] = (uint32_t)dst; dma[2] = (uint32_t)(dst >> 32);
        dma[3] = (uint32_t)src; dma[4] = (uint32_t)(src >> 32);
        dma[5] = 17;
        dma[6] = 0; // selectors are retained for diagnostics; both endpoints are mapped memory

        uint64_t observed_addr = 0, observed_size = 0;
        set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
            observed_addr = addr; observed_size = size;
        });
        GpuState st; run_cb(dma, 7, st);
        set_guest_gpu_write_observer({});

        CHECK(st.dma_copies.size() == 1 && st.dma_copies[0].command_order > 0,
              "address-backed DMA_DATA is retained as an ordered submit operation");
        CHECK(memcmp(target + 3, source + 1, 17) == 0,
              "DMA_DATA copied the exact address-backed byte span");
        CHECK(target[2] == 0xCC && target[20] == 0xCC,
              "DMA_DATA did not write before or after the requested byte span");
        CHECK(observed_addr == dst && observed_size == 17,
              "DMA_DATA notified renderer caches about the guest-memory write");
    }

    // #1742: a DMA_DATA whose destination selector is GDS names an OFFSET into the 64 KiB Global
    // Data Share, not a guest address. Such a destination is legitimately below the 0x10000 floor
    // that rejects malformed guest pointers, so every one of these writes used to be discarded as
    // "invalid/unmapped form" — silently, since the report is capped at 24 lines per run. Astro Bot
    // resets its GDS append counters this way, and the drop is a real loss of guest state.
    {
        uint8_t* gds = compute_gds_backing();
        const uint32_t offset = 0xc68;   // an offset this title actually uses
        memset(gds + offset, 0xAB, 8);
        uint32_t dma[7] = {};
        dma[0] = PM4(7, IT_NOP, R_DMA_DATA);
        dma[1] = offset; dma[2] = 0;     // destination is a GDS offset, well below 0x10000
        dma[3] = 0;      dma[4] = 0;     // 32-bit immediate source: zero the counter
        dma[5] = 4;
        dma[6] = 1;                      // dst_sel (low byte) == 1 == GDS
        GpuState st; run_cb(dma, 7, st);
        uint32_t written = 0xFFFFFFFFu;
        memcpy(&written, gds + offset, sizeof written);
        CHECK(written == 0, "DMA_DATA with a GDS destination selector zeroes the GDS counter");
        CHECK(gds[offset + 4] == 0xAB, "GDS fill wrote exactly the requested byte span");
    }

    // ...and the guard that made it fail-closed still rejects a genuinely malformed guest pointer:
    // a sub-0x10000 destination WITHOUT the GDS selector must remain unwritten, or the fix above
    // would have turned every malformed packet into a wild write at a low address. Assert on GDS
    // CONTENT, not on survival: if the selector test were dropped and the range check alone routed
    // the packet, it would land in GDS and a non-crash assertion would still pass.
    {
        uint8_t* gds = compute_gds_backing();
        gds[0xc68] = 0x77;
        uint32_t dma[7] = {};
        dma[0] = PM4(7, IT_NOP, R_DMA_DATA);
        dma[1] = 0xc68; dma[2] = 0;
        dma[3] = 0;     dma[4] = 0;
        dma[5] = 4;
        dma[6] = 3;                      // dst_sel == 3 == memory: 0xc68 is not a valid address
        GpuState st; run_cb(dma, 7, st);
        CHECK(gds[0xc68] == 0x77,
              "a low destination without the GDS selector is not routed into GDS");
    }

    // A GDS SOURCE (high selector byte) makes dd_src a small offset, which the immediate path cannot
    // distinguish from a 32-bit fill value — it would write the offset itself into guest memory.
    // No title emits this form, so it must be rejected rather than mis-executed.
    {
        alignas(4) uint32_t target = 0xDEADBEEFu;
        const uint64_t dst = (uint64_t)(uintptr_t)&target;
        uint32_t dma[7] = {};
        dma[0] = PM4(7, IT_NOP, R_DMA_DATA);
        dma[1] = (uint32_t)dst; dma[2] = (uint32_t)(dst >> 32);
        dma[3] = 0xc68;         dma[4] = 0;     // a GDS offset masquerading as a 32-bit immediate
        dma[5] = 4;
        dma[6] = 3 | (1u << 8);                 // dst_sel = memory, src_sel = GDS
        GpuState st; run_cb(dma, 7, st);
        CHECK(target == 0xDEADBEEFu,
              "a GDS source is rejected, not written as immediate data");
    }

    // The alignment guards were both deletable with every case green. The byte-count one matters:
    // without it a 6-byte fill writes 4 and silently drops a 2-byte tail — the same truncation this
    // change argues against elsewhere. A partial span must be REJECTED, not written short.
    {
        uint8_t* gds = compute_gds_backing();
        const uint32_t offset = 0x80;
        memset(gds + offset, 0x22, 8);
        uint32_t dma[7] = {};
        dma[0] = PM4(7, IT_NOP, R_DMA_DATA);
        dma[1] = offset; dma[2] = 0;
        dma[3] = 0;      dma[4] = 0;
        dma[5] = 6;                      // not a whole number of dwords
        dma[6] = 1;
        GpuState st; run_cb(dma, 7, st);
        CHECK(gds[offset] == 0x22 && gds[offset + 4] == 0x22,
              "a GDS fill whose byte count is not dword-aligned is rejected, not truncated");
    }

    // A misaligned GDS offset is equally a decode error: the share is addressed in dwords.
    {
        uint8_t* gds = compute_gds_backing();
        const uint32_t offset = 0x92;    // not dword-aligned
        memset(gds + offset, 0x33, 4);
        uint32_t dma[7] = {};
        dma[0] = PM4(7, IT_NOP, R_DMA_DATA);
        dma[1] = offset; dma[2] = 0;
        dma[3] = 0;      dma[4] = 0;
        dma[5] = 4;
        dma[6] = 1;
        GpuState st; run_cb(dma, 7, st);
        CHECK(gds[offset] == 0x33, "a misaligned GDS offset is rejected");
    }

    // The replication loop is only ever exercised at 4 bytes by the title, so cover a multi-dword
    // non-zero fill here: every dword must carry the value and the span must stop exactly.
    {
        uint8_t* gds = compute_gds_backing();
        const uint32_t offset = 0x40;
        memset(gds + offset, 0x11, 20);
        uint32_t dma[7] = {};
        dma[0] = PM4(7, IT_NOP, R_DMA_DATA);
        dma[1] = offset; dma[2] = 0;
        dma[3] = 0xA5A5A5A5u; dma[4] = 0;
        dma[5] = 12;
        dma[6] = 1;
        GpuState st; run_cb(dma, 7, st);
        uint32_t words[3] = {};
        memcpy(words, gds + offset, sizeof words);
        CHECK(words[0] == 0xA5A5A5A5u && words[1] == 0xA5A5A5A5u && words[2] == 0xA5A5A5A5u,
              "a multi-dword non-zero GDS fill replicates the value across the span");
        CHECK(gds[offset + 12] == 0x11, "a multi-dword GDS fill stops exactly at its span");
    }

    // A GDS offset that would run past the 64 KiB share must be rejected rather than clamped —
    // an out-of-range offset is a decode error, and writing a truncated span would corrupt state
    // the shaders read.
    {
        uint8_t* gds = compute_gds_backing();
        const uint32_t offset = (uint32_t)compute_gds_size() - 4;
        memset(gds + offset, 0x5A, 4);
        uint32_t dma[7] = {};
        dma[0] = PM4(7, IT_NOP, R_DMA_DATA);
        dma[1] = offset; dma[2] = 0;
        dma[3] = 0;      dma[4] = 0;
        dma[5] = 16;                     // 16 bytes from 4 bytes before the end: out of range
        dma[6] = 1;
        GpuState st; run_cb(dma, 7, st);
        CHECK(gds[offset] == 0x5A, "an out-of-range GDS span is rejected, not clamped");
    }

    // A queued upload before an address copy is the copy's ordered prefix. The first retained copy
    // must drain it before execution; otherwise a compute-only/non-render submit can copy stale bytes.
    {
        uint32_t source = 0;
        uint32_t target = 0;
        const uint64_t src = (uint64_t)(uintptr_t)&source;
        const uint64_t dst = (uint64_t)(uintptr_t)&target;
        uint32_t stream[13] = {};
        stream[0] = PM4(6, IT_NOP, R_WRITE_DATA);
        stream[1] = 0;
        stream[2] = (uint32_t)src; stream[3] = (uint32_t)(src >> 32);
        stream[4] = 1; stream[5] = 0x13579BDFu;
        stream[6] = PM4(7, IT_NOP, R_DMA_DATA);
        stream[7] = (uint32_t)dst; stream[8] = (uint32_t)(dst >> 32);
        stream[9] = (uint32_t)src; stream[10] = (uint32_t)(src >> 32);
        stream[11] = sizeof(source); stream[12] = 0;
        bool source_invalidated = false;
        set_guest_gpu_write_observer([&](uint64_t addr, uint64_t) {
            if (addr == src) source_invalidated = true;
        });
        set_live_target_byte_range_reader(
            [&](uint64_t addr, uint32_t bytes, std::vector<uint8_t>& output) {
                if (addr != src || bytes != sizeof(source))
                    return LiveTargetByteReadResult::NotFound;
                if (source_invalidated) return LiveTargetByteReadResult::NotFound;
                const uint32_t stale = 0xAAAAAAAAu;
                const auto* begin = reinterpret_cast<const uint8_t*>(&stale);
                output.assign(begin, begin + sizeof(stale));
                return LiveTargetByteReadResult::Success;
            });
        GpuState st; run_cb(stream, 13, st);
        set_live_target_byte_range_reader({});
        set_guest_gpu_write_observer({});
        CHECK(source_invalidated && source == 0x13579BDFu && target == source,
              "WRITE_DATA prefix invalidates a renderer-owned source before later address DMA");
    }

    // Guest-memory consumers that are folded eagerly cannot observe a preceding retained DMA.
    // Preserve the DMA in diagnostics, but reject the entire submit instead of snapshotting stale
    // indirect registers and then executing a misleading partial timeline.
    {
        ShaderReg source_reg{0x44, 0xA1B2C3D4u};
        ShaderReg target_reg{0x44, 0x11111111u};
        const uint64_t src = (uint64_t)(uintptr_t)&source_reg;
        const uint64_t dst = (uint64_t)(uintptr_t)&target_reg;
        uint32_t stream[11] = {};
        stream[0] = PM4(7, IT_NOP, R_DMA_DATA);
        stream[1] = (uint32_t)dst; stream[2] = (uint32_t)(dst >> 32);
        stream[3] = (uint32_t)src; stream[4] = (uint32_t)(src >> 32);
        stream[5] = sizeof(source_reg); stream[6] = 0;
        stream[7] = PM4(4, IT_NOP, R_SH_REGS_INDIRECT);
        stream[8] = 1; stream[9] = (uint32_t)dst; stream[10] = (uint32_t)(dst >> 32);
        GpuState st; run_cb(stream, 11, st);
        CHECK(st.dma_copies.size() == 1 && st.dma_execution_rejected,
              "DMA before indirect registers is retained and rejected fail-closed");
        CHECK(target_reg.value == 0x11111111u && st.sh.find(0x44) == st.sh.end(),
              "rejected DMA/indirect submit executes neither copy nor stale register fold");
    }

    // Conversely, a later immediate DMA fill must not enter the completion FIFO and overtake an
    // earlier retained copy. Keeping the suffix in command_order leaves the destination filled.
    {
        uint32_t source = 0xA5C31E79u;
        uint32_t target = 0xFFFFFFFFu;
        const uint64_t src = (uint64_t)(uintptr_t)&source;
        const uint64_t dst = (uint64_t)(uintptr_t)&target;
        uint32_t stream[14] = {};
        stream[0] = PM4(7, IT_NOP, R_DMA_DATA);
        stream[1] = (uint32_t)dst; stream[2] = (uint32_t)(dst >> 32);
        stream[3] = (uint32_t)src; stream[4] = (uint32_t)(src >> 32);
        stream[5] = sizeof(source); stream[6] = 0;
        stream[7] = PM4(7, IT_NOP, R_DMA_DATA);
        stream[8] = (uint32_t)dst; stream[9] = (uint32_t)(dst >> 32);
        stream[10] = 0; stream[11] = 0;
        stream[12] = sizeof(target); stream[13] = 0;
        GpuState st; run_cb(stream, 14, st);
        CHECK(st.ordered_memory_effects.size() == 1 && target == 0,
              "immediate DMA suffix executes after an earlier address copy without FIFO reversal");
    }

    // A malformed/unmapped address source is not an immediate value merely because the destination
    // is valid. It must fail closed without touching the target or notifying renderer caches.
    {
        uint64_t target = 0x8877665544332211ull;
        const uint64_t dst = (uint64_t)(uintptr_t)&target;
        const uint64_t bad_src = 0x00000DEADBEEF000ull;
        uint32_t dma[7] = {};
        dma[0] = PM4(7, IT_NOP, R_DMA_DATA);
        dma[1] = (uint32_t)dst; dma[2] = (uint32_t)(dst >> 32);
        dma[3] = (uint32_t)bad_src; dma[4] = (uint32_t)(bad_src >> 32);
        dma[5] = 8;
        bool notified = false;
        set_guest_gpu_write_observer([&](uint64_t, uint64_t) { notified = true; });
        GpuState st; run_cb(dma, 7, st);
        set_guest_gpu_write_observer({});
        CHECK(target == 0x8877665544332211ull,
              "DMA_DATA skips an unmapped address source without modifying the destination");
        CHECK(!notified, "a skipped DMA_DATA copy does not report a guest-memory write");
    }

    // A mapped but read-only destination is readable, so source/destination mappedness alone is
    // insufficient. The executor must reject it before memmove instead of taking a host fault.
    {
#ifdef _WIN32
        SYSTEM_INFO sys{}; GetSystemInfo(&sys);
        const size_t page_size = sys.dwPageSize;
        uint8_t* page = (uint8_t*)VirtualAlloc(nullptr, page_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        const size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
        uint8_t* page = (uint8_t*)mmap(nullptr, page_size, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (page == MAP_FAILED) page = nullptr;
#endif
        CHECK(page != nullptr, "allocated a page for the read-only DMA destination guard");
        if (page) {
            memset(page, 0x5A, page_size);
#ifdef _WIN32
            DWORD old_protect = 0;
            const bool protected_read_only =
                VirtualProtect(page, page_size, PAGE_READONLY, &old_protect) != 0;
#else
            const bool protected_read_only = mprotect(page, page_size, PROT_READ) == 0;
#endif
            CHECK(protected_read_only, "made the DMA destination page read-only");
            if (protected_read_only) {
                uint64_t source = 0x0123456789ABCDEFull;
                const uint64_t src = (uint64_t)(uintptr_t)&source;
                const uint64_t dst = (uint64_t)(uintptr_t)page;
                uint32_t dma[7] = {};
                dma[0] = PM4(7, IT_NOP, R_DMA_DATA);
                dma[1] = (uint32_t)dst; dma[2] = (uint32_t)(dst >> 32);
                dma[3] = (uint32_t)src; dma[4] = (uint32_t)(src >> 32);
                dma[5] = sizeof(source);
                bool notified = false;
                set_guest_gpu_write_observer([&](uint64_t, uint64_t) { notified = true; });
                GpuState st; run_cb(dma, 7, st);
                set_guest_gpu_write_observer({});
                CHECK(page[0] == 0x5A && page[sizeof(source) - 1] == 0x5A,
                      "DMA_DATA skips a read-only destination without modifying it");
                CHECK(!notified, "a read-only DMA destination does not report a guest-memory write");
            }
#ifdef _WIN32
            DWORD ignored = 0; VirtualProtect(page, page_size, PAGE_READWRITE, &ignored);
            VirtualFree(page, 0, MEM_RELEASE);
#else
            mprotect(page, page_size, PROT_READ | PROT_WRITE);
            munmap(page, page_size);
#endif
        }
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

    // #1226: the self-test that licenses reading that `false` as evidence. Both heads above are, by
    // construction, the first node of their own chain, so the walk must find both — and when the
    // registry is empty it must report probes=0 rather than a confident-looking positives=0. This
    // matters because ArcRunner's whole init/fence conclusion rests on `member=0` readings, and an
    // unarmed walk produces exactly the same zero.
    {
        char self[256] = {};
        const int deep_positives = mb3_freelist_selftest(self, sizeof self);
        CHECK(strstr(self, "probes=2") && strstr(self, "positives=2"),
              "MB3 self-test finds every learned bin head in its own chain");
        // The primary chain is head -> free_tail, so exactly one interior node is reachable; the
        // secondary head has no successor. A head matches at hop 0, so without this the control
        // would pass on a walk that cannot traverse at all.
        CHECK(deep_positives == 1 && strstr(self, "deep=1") && strstr(self, "deep_positives=1"),
              "MB3 self-test reaches an interior node, not just the head");
        CHECK(!strstr(self, "BLIND") && !strstr(self, "NO HEADS") && !strstr(self, "SHALLOW"),
              "MB3 self-test reports no blindness while the walk is working");
        mb3_reset_pool_candidates_for_test();
        char empty[192] = {};
        CHECK(mb3_freelist_selftest(empty, sizeof empty) == 0 && strstr(empty, "probes=0") &&
              strstr(empty, "NO HEADS"),
              "MB3 self-test flags an empty registry instead of reporting a clean zero");
        mb3_note_tls_pool_candidate((uint64_t)(uintptr_t)mb3_pool);   // restore for the tests below
    }

    // Full bundles leave the per-thread secondary root and move into one of eight lock-free global
    // recycler slots. Membership must continue to see those nodes until the slot is popped.
    static uint64_t global_slots[8] = {};
    static FreeNode global_label{}, global_tail{};
    global_label.next = (uint64_t)(uintptr_t)&global_tail;
    global_tail.next = 0;
    global_slots[5] = (uint64_t)(uintptr_t)&global_label;
    mb3_note_global_recycler_bin((uint64_t)(uintptr_t)global_slots);
    CHECK(mb3_freelist_contains_stable((uint64_t)(uintptr_t)&global_tail, &match) &&
          match.list == 8 && match.hops == 1,
          "MB3 membership walks a bundle held in a global recycler slot");
    global_slots[5] = 0;
    CHECK(!mb3_freelist_contains_stable((uint64_t)(uintptr_t)&global_tail, nullptr),
          "a bundle removed from the global recycler is no longer classified free");

    // Once all eight global recycler slots are occupied, MB3 returns a bundle to the central page
    // structure. FFreeBlock is an in-place header at the low end of its contiguous free run. Detect
    // an interior 0x20-byte block without knowing the title-specific FPoolInfoSmall table address.
    static uint8_t central_backing[0x20000] = {};
    uint8_t* central_page =
        (uint8_t*)(((uintptr_t)central_backing + 0xffffull) & ~0xffffull);
    uint32_t* central_header = (uint32_t*)(central_page + 0x2000);
    central_header[0] = 0xe3010002; // DOLL: BlockSizeShifted=2, PoolIndex=1, Canary=0xe3
    central_header[1] = 4;          // four free blocks in [header, header + 4*0x20)
    central_header[2] = ~0u;        // end of the central free-run index chain
    uint64_t central_free = (uint64_t)(uintptr_t)(central_page + 0x2040);
    CHECK(mb3_freelist_contains_stable(central_free, &match) &&
          match.list == 11 && match.pool_base == (uint64_t)(uintptr_t)central_page &&
          match.head == (uint64_t)(uintptr_t)(central_page + 0x2000) && match.hops == 2,
          "MB3 membership detects an interior block in a central in-page free run");
    CHECK(!mb3_freelist_contains_stable((uint64_t)(uintptr_t)(central_page + 0x2080), nullptr),
          "a block immediately outside a central free run is not classified free");
    central_header[0] = 0xe2010002;
    CHECK(!mb3_freelist_contains_stable(central_free, nullptr),
          "a central-run lookalike with the wrong MB3 canary is rejected");
    central_header[0] = 0xe7010002;
    CHECK(mb3_freelist_contains_stable(central_free, nullptr),
          "MB3 membership accepts the newer UE central-run canary");

    // A command packet can outlive the 0x20-byte label generation it captured. If that block was
    // reused, executing the stale DmaData(0) would overwrite its new owner's first dword (observed
    // live as a C++ vtable becoming 0x400000000). Key the build snapshot by the exact packet.
    {
        struct alignas(0x20) ReusedLabel { uint64_t value; uint64_t pad[3]; } reused{};
        reused.value = 0x1020304050607080ull;
        uint64_t addr = (uint64_t)(uintptr_t)&reused;
        prosper_label_hist_dma_built(addr, 0x3000, 0, 1);
        // The HISTORICAL 9-dword packet, deliberately: the generation check reads a build snapshot
        // that only that shape carries. Every capture recorded before #1756 contains it, so this is
        // the compatibility leg — the live shape is exercised immediately below.
        uint32_t dma[9] = {};
        dma[0] = PM4(9, IT_NOP, R_DMA_DATA);
        dma[1] = (uint32_t)addr; dma[2] = (uint32_t)(addr >> 32);
        dma[5] = 4; dma[6] = 0x303;
        dma[7] = (uint32_t)reused.value; dma[8] = (uint32_t)(reused.value >> 32);
        reused.value = 0x409123458ull; // a different owner replaced the build-time label residue
        GpuState st; run_cb(dma, 9, st);
        CHECK(reused.value == 0x409123458ull,
              "stale DmaData does not overwrite a label block reused after packet build");

        uint32_t rel[7] = {};
        rel[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
        rel[1] = (uint32_t)addr; rel[2] = (uint32_t)(addr >> 32);
        rel[3] = 1; rel[4] = 1; rel[6] = 4;
        run_cb(rel, 7, st);
        CHECK(reused.value == 0x409123458ull,
              "the paired stale ReleaseMem does not overwrite the reused block either");
    }

    // ReleaseMem also carries its own build snapshot. Cover that guard without a preceding
    // suppressed DmaData (and therefore without relying on the one-generation suppression debt).
    {
        struct alignas(0x20) ReusedLabel { uint64_t value; uint64_t pad[3]; } reused{};
        reused.value = 0x1020304050607080ull;
        uint64_t addr = (uint64_t)(uintptr_t)&reused;
        prosper_label_hist_dma_built(addr, 0x4000, 0, 1);
        uint32_t rel[9] = {};
        rel[0] = PM4(9, IT_NOP, R_RELEASE_MEM);
        rel[1] = (uint32_t)addr; rel[2] = (uint32_t)(addr >> 32);
        rel[3] = 1; rel[4] = 1; rel[6] = 4;
        rel[7] = (uint32_t)reused.value; rel[8] = (uint32_t)(reused.value >> 32);
        reused.value = 0x409abcdefull;
        GpuState st; run_cb(rel, 9, st);
        CHECK(reused.value == 0x409abcdefull,
              "packet-local ReleaseMem identity protects a reused label without DmaData debt");
    }

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

    // Plain EOP labels have no DmaData-init lifecycle. Their untouched high dword may look like a
    // heap pointer while the low dword is the legitimate fence payload; the consumed-marker forge
    // guard must not permanently suppress that ordinary 32-bit completion write.
    {
        static struct alignas(0x20) PlainLabel { uint64_t value; uint64_t pad[3]; } label{};
        label.value = 0x1000000000ull;
        const uint64_t addr = (uint64_t)(uintptr_t)&label;
        uint32_t rel[7] = {};
        rel[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
        rel[1] = (uint32_t)addr; rel[2] = (uint32_t)(addr >> 32);
        rel[3] = 1; rel[4] = 1; rel[6] = 4;
        GpuState st; run_cb(rel, 7, st);
        CHECK(label.value == 0x1000000001ull,
              "plain pointer-shaped fence label is not treated as a consumed-marker forge");
    }

    // A suppressed LIVE fence must not count as a completed generation. If an earlier generation's
    // init never executes, its pointer-valued REL1/REL2 is suppressed; the next generation's real
    // DmaData(0)+ReleaseMem(1) must still see an outstanding init and signal the label. Counting the
    // suppressed fence makes dma_exec_n == rel_exec_n and the forge guard suppresses this real fence.
    for (uint32_t stale_sel : {1u, 2u}) {
        struct alignas(0x20) Label { uint64_t value; uint64_t pad[3]; } label{};
        const uint64_t addr = (uint64_t)(uintptr_t)&label;
        GpuState st;

        prosper_label_hist_dma_built(addr, 0x5000 + stale_sel, 0, 1);
        label.value = 0x1000000008ull; // freed/relinked pointer: missing init for generation 1
        uint32_t stale_rel[7] = {};
        stale_rel[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
        stale_rel[1] = (uint32_t)addr; stale_rel[2] = (uint32_t)(addr >> 32);
        stale_rel[3] = stale_sel; stale_rel[4] = 1; stale_rel[6] = 4;
        run_cb(stale_rel, 7, st);
        CHECK(label.value == 0x1000000008ull,
              stale_sel == 1 ? "REL1-LIVE stale fence is suppressed"
                             : "REL2-LIVE stale fence is suppressed");

        prosper_label_hist_dma_built(addr, 0x6000 + stale_sel, 0, 1);
        uint32_t live_pair[14] = {};
        live_pair[0] = PM4(7, IT_NOP, R_DMA_DATA);
        live_pair[1] = (uint32_t)addr; live_pair[2] = (uint32_t)(addr >> 32);
        live_pair[3] = 0; live_pair[4] = 0; live_pair[5] = 4; live_pair[6] = 0x303;
        live_pair[7] = PM4(7, IT_NOP, R_RELEASE_MEM);
        live_pair[8] = (uint32_t)addr; live_pair[9] = (uint32_t)(addr >> 32);
        live_pair[10] = 1; live_pair[11] = 1; live_pair[13] = 4;
        run_cb(live_pair, 14, st);
        CHECK((uint32_t)label.value == 1,
              stale_sel == 1 ? "REL1-LIVE suppression does not consume the next generation"
                             : "REL2-LIVE suppression does not consume the next generation");
    }
    mb3_reset_pool_candidates_for_test();

    // #1226 — ArcRunner's exact observed shape, both arms. Its AGC fence labels are 0x20-byte blocks
    // from a pool whose free list is threaded through each block's first qword, so a label handed to
    // the GPU still holds a `0x20xxxxxxxx` link to another label. The guest's own 4-byte DmaData
    // init zeroes only the LOW dword, leaving exactly 0x2000000000 — byte-identical to a freed
    // FFreeBlock — and the paired 4-byte ReleaseMem(<-1) then produces 0x2000000001. That composite
    // is what `forges_freelist_ptr` matches, and performing the write is CORRECT: real hardware
    // writes 32 bits and the consumer polls only the low dword. Suppressing it is the #1245
    // regression (every consumer wait then reads an unsatisfiable 0). So: with the paired init
    // outstanding the fence MUST land, and with no init outstanding the same packet MUST be
    // suppressed. Both arms are asserted here so that widening any guard to make the ArcRunner
    // FORGE-STOMP tripwire fall silent breaks this test rather than the title.
    // The two packets run as SEPARATE folds with an assertion in between. Submitting them together
    // and checking only the end state would accept the wrong path: 0x2020f3d5a0 with its low dword
    // replaced by 1 is ALSO 0x2000000001, so "the init never ran and the fence landed anyway"
    // produces the same final value. Asserting the intermediate 0x2000000000 pins the composition.
    {
        struct alignas(0x20) PoolLabel { uint64_t link; uint64_t pad[3]; } label{};
        const uint64_t addr = (uint64_t)(uintptr_t)&label;
        label.link = 0x2020f3d5a0ull;   // a pool link to a neighbouring label, as the rings record it
        prosper_label_hist_dma_built(addr, 0x7000, 0, 1);
        GpuState st;
        uint32_t init[7] = {PM4(7, IT_NOP, R_DMA_DATA), (uint32_t)addr, (uint32_t)(addr >> 32),
                            0, 0, 4, 0x303};
        run_cb(init, 7, st);
        CHECK(label.link == 0x2000000000ull,
              "ArcRunner shape: the 4-byte init zeroes only the low dword of the pool link");
        uint32_t rel[7] = {PM4(7, IT_NOP, R_RELEASE_MEM), (uint32_t)addr, (uint32_t)(addr >> 32),
                           1, 1, 0, 4};
        run_cb(rel, 7, st);
        CHECK(label.link == 0x2000000001ull,
              "ArcRunner shape: a live paired fence still writes over a pool-link residue");
    }
    {
        // The counter arm. A fresh label reaches the same post-init CONTENT with no executed init
        // behind it, so dma_exec_n == rel_exec_n == 0 and `live_pair` is false. (The built-but-never-
        // executed init is one of the two shapes that gives !live_pair; the other is "the previous
        // generation completed". Both must suppress, and the assertion holds for either.)
        struct alignas(0x20) PoolLabel { uint64_t link; uint64_t pad[3]; } label{};
        const uint64_t addr = (uint64_t)(uintptr_t)&label;
        label.link = 0x2000000000ull;
        prosper_label_hist_dma_built(addr, 0x7008, 0, 1);
        GpuState st;
        uint32_t rel[7] = {PM4(7, IT_NOP, R_RELEASE_MEM), (uint32_t)addr, (uint32_t)(addr >> 32),
                           1, 1, 0, 4};
        run_cb(rel, 7, st);
        CHECK(label.link == 0x2000000000ull,
              "ArcRunner shape: the same fence with no outstanding init is suppressed");
    }
    {
        uint64_t candidates = 99, suppressed = 99, landed = 99;
        prosper_rel1_forge_decision_totals(&candidates, &suppressed, &landed);
        CHECK(candidates == 0 && suppressed == 0 && landed == 0,
              "ArcRunner diagnostic: default-off mode records no suppression decisions");
        CHECK(prosper_rel1_forge_report_due_for_test(1) &&
              prosper_rel1_forge_report_due_for_test(64) &&
              prosper_rel1_forge_report_due_for_test(127) &&
              prosper_rel1_forge_report_due_for_test(256),
              "ArcRunner diagnostic: dense census reports every possible sub-256 terminal total");
        CHECK(!prosper_rel1_forge_report_due_for_test(257) &&
              prosper_rel1_forge_report_due_for_test(512),
              "ArcRunner diagnostic: census tail remains bounded to each 256th candidate");

        // Positive-control every reported number through the real store site. Observe-only mode is
        // test-private: it counts the candidate but deliberately lets it land, proving `landed` can
        // become nonzero before the experiment relies on zero.
        static struct alignas(0x20) PoolLabel { uint64_t link; uint64_t pad[3]; } label{};
        const uint64_t addr = (uint64_t)(uintptr_t)&label;
        label.link = 0x2020f3d5a0ull;
        prosper_label_hist_dma_built(addr, 0x7010, 0, 1);
        GpuState st;
        uint32_t init[7] = {PM4(7, IT_NOP, R_DMA_DATA), (uint32_t)addr, (uint32_t)(addr >> 32),
                            0, 0, 4, 0x303};
        run_cb(init, 7, st);
        CHECK(label.link == 0x2000000000ull,
              "ArcRunner diagnostic: observed candidate begins from the real post-init shape");

        prosper_rel1_forge_decision_reset_for_test();
        prosper_rel1_forge_suppress_all_override_for_test(2);
        uint32_t rel[7] = {PM4(7, IT_NOP, R_RELEASE_MEM), (uint32_t)addr, (uint32_t)(addr >> 32),
                           1, 1, 0, 4};
        run_cb(rel, 7, st);
        prosper_rel1_forge_decision_totals(&candidates, &suppressed, &landed);
        CHECK(label.link == 0x2000000001ull,
              "ArcRunner diagnostic: observe-only positive control reaches the real store");
        CHECK(candidates == 1 && suppressed == 0 && landed == 1,
              "ArcRunner diagnostic: candidate and landed totals have a positive control");

        // The decisive experiment's first half: repeat the SAME live paired candidate and require
        // all mode to stop it. Re-init the next generation while the destructive mode is off.
        prosper_rel1_forge_suppress_all_override_for_test(0);
        label.link = 0x2020f3d5a0ull;
        prosper_label_hist_dma_built(addr, 0x7018, 0, 1);
        run_cb(init, 7, st);
        CHECK(label.link == 0x2000000000ull,
              "ArcRunner diagnostic: all-suppress candidate repeats the real post-init shape");

        prosper_rel1_forge_decision_reset_for_test();
        prosper_rel1_forge_suppress_all_override_for_test(1);
        run_cb(rel, 7, st);
        prosper_rel1_forge_decision_totals(&candidates, &suppressed, &landed);
        CHECK(label.link == 0x2000000000ull,
              "ArcRunner diagnostic: all mode suppresses the live paired forge candidate");
        CHECK(candidates == 1 && suppressed == 1 && landed == 0,
              "ArcRunner diagnostic: all-mode totals prove one candidate suppressed and none landed");

        // Scope the lever to the exact forge predicate: an ordinary non-forging REL1 completion
        // must still land, and it must not change the experiment's candidate totals.
        static struct alignas(0x20) PlainLabel { uint64_t value; uint64_t pad[3]; } plain{};
        const uint64_t plain_addr = (uint64_t)(uintptr_t)&plain;
        plain.value = 0;
        uint32_t plain_rel[7] = {PM4(7, IT_NOP, R_RELEASE_MEM), (uint32_t)plain_addr,
                                 (uint32_t)(plain_addr >> 32), 1, 7, 0, 4};
        run_cb(plain_rel, 7, st);
        prosper_rel1_forge_decision_totals(&candidates, &suppressed, &landed);
        CHECK((uint32_t)plain.value == 7,
              "ArcRunner diagnostic: all mode leaves a non-forging REL1 completion intact");
        CHECK(candidates == 1 && suppressed == 1 && landed == 0,
              "ArcRunner diagnostic: non-forging writes do not enter all-mode totals");
        prosper_rel1_forge_suppress_all_override_for_test(0);
    }

    // #1226 PROSPER_WRITE_TRAP, both arms on every payload-carrying scan site. A hard negative from
    // this instrument is only meaningful if a match would have counted, so assert that the armed
    // value counts on each path and that an adjacent value does not — a refactor that drops one call
    // site otherwise leaves the documented coverage silently false. REL3 and EVENT_WRITE are the two
    // sites deliberately not covered: both store gpu_clock64(), so a test cannot choose their value.
    {
        struct alignas(0x20) TrapLabel { uint64_t value; uint64_t pad[3]; } label{};
        const uint64_t addr = (uint64_t)(uintptr_t)&label;
        uint64_t copy_source = 0;
        const uint64_t src = (uint64_t)(uintptr_t)&copy_source;
        // The scanned payload is the same low dword each path stores, so a hit also pins the store.
        auto rel1 = [&](uint32_t v) {
            GpuState st;
            uint32_t p[7] = {PM4(7, IT_NOP, R_RELEASE_MEM), (uint32_t)addr, (uint32_t)(addr >> 32),
                             1, v, 0, 4};
            run_cb(p, 7, st);
        };
        auto rel2 = [&](uint32_t v) {   // data_sel 2: the 64-bit fence leg
            GpuState st;
            uint32_t p[7] = {PM4(7, IT_NOP, R_RELEASE_MEM), (uint32_t)addr, (uint32_t)(addr >> 32),
                             2, v, 0, 4};
            run_cb(p, 7, st);
        };
        auto dma_imm = [&](uint32_t v) {
            GpuState st;
            uint32_t p[7] = {PM4(7, IT_NOP, R_DMA_DATA), (uint32_t)addr, (uint32_t)(addr >> 32),
                             v, 0, 4, 0x303};
            run_cb(p, 7, st);
        };
        auto dma_copy = [&](uint32_t v) {   // a 64-bit source address selects the copy form
            copy_source = v;
            GpuState st;
            uint32_t p[7] = {PM4(7, IT_NOP, R_DMA_DATA), (uint32_t)addr, (uint32_t)(addr >> 32),
                             (uint32_t)src, (uint32_t)(src >> 32), 4, 0};
            run_cb(p, 7, st);
        };
        auto wdata = [&](uint32_t v) {
            GpuState st;
            uint32_t p[6] = {PM4(6, IT_NOP, R_WRITE_DATA), 0, (uint32_t)addr,
                             (uint32_t)(addr >> 32), 1, v};
            run_cb(p, 6, st);
        };
        auto both_arms = [&](const char* ignores, const char* counts, auto&& emit) {
            label.value = 0;
            const uint64_t before = prosper_gpu_write_trap_matches();
            emit(0x5a5a0000u);                                     // adjacent value: must not match
            CHECK(prosper_gpu_write_trap_matches() == before, ignores);
            emit(0x5a5a0001u);                                     // armed value: must match once
            CHECK(prosper_gpu_write_trap_matches() == before + 1 &&
                  (uint32_t)label.value == 0x5a5a0001u, counts);
        };
        both_arms("PROSPER_WRITE_TRAP/REL1 ignores a value adjacent to the armed one",
                  "PROSPER_WRITE_TRAP/REL1 counts the armed value, and the store lands", rel1);
        both_arms("PROSPER_WRITE_TRAP/REL2 ignores a value adjacent to the armed one",
                  "PROSPER_WRITE_TRAP/REL2 counts the armed value, and the store lands", rel2);
        both_arms("PROSPER_WRITE_TRAP/DMA-imm ignores a value adjacent to the armed one",
                  "PROSPER_WRITE_TRAP/DMA-imm counts the armed value, and the store lands", dma_imm);
        both_arms("PROSPER_WRITE_TRAP/DMA-copy ignores a value adjacent to the armed one",
                  "PROSPER_WRITE_TRAP/DMA-copy counts the armed value, and the store lands", dma_copy);
        both_arms("PROSPER_WRITE_TRAP/WDATA ignores a value adjacent to the armed one",
                  "PROSPER_WRITE_TRAP/WDATA counts the armed value, and the store lands", wdata);
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
