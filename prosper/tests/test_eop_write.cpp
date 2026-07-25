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
    printf("== test_eop_write ==\n");
    // #1226: the generation and MB3-freelist suppression families are OFF by default (their
    // content/membership premises misfire on live protocols — see generation_guard() /
    // mb3_freelist_guard() in command_processor.cpp). This test verifies the suppression
    // MECHANISMS, so arm them explicitly; the env is read once before any fold below.
#ifdef _WIN32
    _putenv_s("PROSPER_GENERATION_GUARD", "1");
    _putenv_s("PROSPER_MB3_FREELIST_GUARD", "1");
#else
    setenv("PROSPER_GENERATION_GUARD", "1", 1);
    setenv("PROSPER_MB3_FREELIST_GUARD", "1", 1);
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

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
