// test_agc_submit — end-to-end guard for the AGC submit path: build a Draw Command Buffer through
// the real Dcb HLE functions (as the game does), submit it via sceAgcDriverSubmitDcb (UglJIZjGssM),
// and assert the CommandProcessor folded the stream into a GpuState — registers written, draw
// recorded, submit/draw stats updated. This locks in the "first real command buffer executes"
// milestone programmatically (previously only observed in a boot log line, which VERIFICATION.md
// says not to rely on). It exercises the exact front-half -> back-half handoff: Dcb build -> PM4 ->
// decode -> GpuState.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/command_processor.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper;

// Front-half stats exposed by hle_agc.cpp.
extern "C" void prosper_agc_submit_stats(uint64_t* submits, uint64_t* draws);
extern "C" bool prosper_agc_submit_sh_reg(uint64_t queue, uint32_t offset, uint32_t* value);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Game's Dcb struct (must match hle_agc.cpp's AgcDcb byte-for-byte).
struct Dcb {
    uint32_t* bottom; uint32_t* top; uint32_t* cursor_up; uint32_t* cursor_down;
    void* callback; void* user_data; uint32_t reserved_dw; uint32_t pad;
};
// The AgcDriver submit descriptor (Kyty Gen5Driver::Packet).
struct Packet { uint32_t* addr; uint32_t dw_num; uint8_t pad[4]; };
using HostHle9 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                              uint64_t, uint64_t, uint64_t, uint64_t);

int main() {
    printf("== test_agc_submit ==\n");
    register_builtin_hle();

    auto reset  = Hle::lookup("TRO721eVt4g");   // GraphicsDcbResetQueue
    auto setcx  = Hle::lookup("ZvwO9euwYzc");   // GraphicsDcbSetCxRegistersIndirect
    auto setsh  = Hle::lookup("pFLArOT53+w");   // GraphicsDcbSetShRegisterDirect
    auto setidx = Hle::lookup("GIIW2J37e70");   // GraphicsDcbSetIndexSize
    auto draw   = Hle::lookup("Yw0jKSqop+E");   // GraphicsDcbDrawIndexAuto
    auto submit = Hle::lookup("UglJIZjGssM");   // GraphicsDriverSubmitDcb
    auto submit_acb = Hle::lookup("gSRnr79F8tQ"); // GraphicsDriverSubmitAcb
    auto regmem = Hle::lookup("AOLcoIkQDgM");   // QueryResourceRegistrationUserMemoryRequirements
    auto maxname = Hle::lookup("uJziRsODk1c");   // sceAgcDriverGetResourceRegistrationMaxNameLength
    auto release = reinterpret_cast<HostHle9>(Hle::lookup("wr23dPKyWc0")); // GraphicsCbReleaseMem
    auto waitmem = reinterpret_cast<HostHle9>(Hle::lookup("VmW0Tdpy420")); // GraphicsDcbWaitRegMem
    CHECK(reset && setcx && setsh && setidx && draw && submit && submit_acb && regmem && maxname && release && waitmem,
          "AGC Dcb/Acb submit and resource-registration functions registered");
    if (!(reset && setcx && setsh && setidx && draw && submit && submit_acb && regmem && maxname && release && waitmem)) {
        printf("== FAIL ==\n"); return 1;
    }

    uint64_t registration_bytes = 0x7a2a67fe6900ull;
    CHECK(regmem((uint64_t)(uintptr_t)&registration_bytes, 0x2000, 0x38, 0, 0, 0) == 0,
          "resource-registration memory query returned OK");
    CHECK(registration_bytes == 0x80800,
          "resource-registration memory query replaced the poisoned size deterministically");
    CHECK((registration_bytes & 0x3f) == 0 && registration_bytes < 0x100000,
          "resource-registration memory requirement is aligned and bounded");
    CHECK((int64_t)regmem(0, 0x2000, 0x38, 0, 0, 0) < 0,
          "resource-registration memory query rejects a null output");

    uint32_t max_name_length = 0xdeadbeefu;
    CHECK(maxname((uint64_t)(uintptr_t)&max_name_length, 0, 0, 0, 0, 0) == 0,
          "resource-registration max-name query returned OK");
    CHECK(max_name_length == 0xfc,
          "resource-registration max-name query initialized the poisoned output");
    CHECK((int64_t)maxname(0, 0, 0, 0, 0, 0) < 0,
          "resource-registration max-name query rejects a null output");

    // Baseline stats (other tests in-process may have submitted; measure deltas).
    uint64_t s0 = 0, d0 = 0; prosper_agc_submit_stats(&s0, &d0);

    // Build a command buffer exactly as the game would: the register-indirect packet carries a
    // pointer to a real {offset,value} array, which the CommandProcessor reads back at submit time.
    static uint32_t buffer[256];
    memset(buffer, 0, sizeof buffer);
    Dcb dcb{};
    dcb.bottom = buffer; dcb.top = buffer + 256; dcb.cursor_up = buffer; dcb.cursor_down = buffer + 256;
    auto D = (uint64_t)(uintptr_t)&dcb;

    struct Reg { uint32_t offset, value; };
    static Reg cx_regs[] = { {0x10, 0xAAAA}, {0x11, 0xBBBB}, {0x12, 0xCCCC} };

    reset(D, 0x3ff, 0, 0, 0, 0);
    setcx(D, (uint64_t)(uintptr_t)cx_regs, 3, 0, 0, 0);
    setidx(D, 1 /*index_size*/, 0, 0, 0, 0);
    draw(D, 42 /*index_count*/, 0, 0, 0, 0);

    uint32_t dw = (uint32_t)(dcb.cursor_up - buffer);
    CHECK(dw > 0, "Dcb build advanced the write cursor");

    // Submit it through the real driver entrypoint.
    Packet pkt{ buffer, dw, {0,0,0,0} };
    uint64_t rc = submit((uint64_t)(uintptr_t)&pkt, 0, 0, 0, 0, 0);
    CHECK(rc == 0, "SubmitDcb returned OK (0)");

    uint64_t s1 = 0, d1 = 0; prosper_agc_submit_stats(&s1, &d1);
    CHECK(s1 == s0 + 1, "submit count incremented by 1");
    CHECK(d1 == d0 + 1, "draw count incremented by 1 (one DrawIndexAuto)");

    // Replay the same buffer through the CommandProcessor directly to inspect the folded GpuState
    // (SubmitDcb uses a private state internally; this asserts the same decode produces the right
    // register file + draw — the exact transformation the submit performed).
    {
        gpu::GpuState st;
        size_t packets = gpu::run_command_buffer(buffer, dw, st);
        CHECK(packets == 4, "CommandProcessor decoded 4 packets (reset/setcx/setidx/draw)");
        CHECK(st.cx[0x10] == 0xAAAA && st.cx[0x11] == 0xBBBB && st.cx[0x12] == 0xCCCC,
              "folded the indirect Cx registers (0x10/0x11/0x12) from the pointed-to array");
        CHECK(st.index_type == 1, "folded the index type");
        CHECK(st.draws.size() == 1 && st.draws[0].index_count == 42,
              "recorded one DrawIndexAuto with index_count=42");
    }

    // Bounded register-file ingest (#1264): a Blue Prince-style indirect array interleaves real
    // registers with out-of-range offsets (prosper's bit31-tagged virtual-register defaults like
    // TEXTURE_GRADIENT_CONTROL=0x80003FFD, plus raw stale arena data). Hardware drops writes to
    // nonexistent register offsets; the fold must too, or the per-draw snapshot copies grow
    // unboundedly (the #1195-family loading crawl). Both cases fail without kRegOffsetLimit.
    {
        static uint32_t buffer2[256];
        memset(buffer2, 0, sizeof buffer2);
        Dcb dcb2{};
        dcb2.bottom = buffer2; dcb2.top = buffer2 + 256;
        dcb2.cursor_up = buffer2; dcb2.cursor_down = buffer2 + 256;
        auto D2 = (uint64_t)(uintptr_t)&dcb2;
        static Reg mixed[] = { {0x20, 0x1}, {0x80003FFD, 0x0}, {0x14225680, 0x7fa3}, {0x21, 0x2} };
        auto setcx_direct = Hle::lookup("LHFXRrlTPD8");   // GraphicsDcbSetCxRegisterDirect
        CHECK(setcx_direct != nullptr, "direct Cx register setter registered");
        reset(D2, 0x3ff, 0, 0, 0, 0);
        setcx(D2, (uint64_t)(uintptr_t)mixed, 4, 0, 0, 0);
        // Direct form: {offset,value} packed into a1 (SysV by-value ShaderRegister). One write at
        // the last in-range offset, one just past the bound.
        setcx_direct(D2, ((uint64_t)0x1234 << 32) | 0xFFFFu, 0, 0, 0, 0);
        setcx_direct(D2, ((uint64_t)0x5678 << 32) | 0x10000u, 0, 0, 0, 0);
        gpu::GpuState st2;
        gpu::run_command_buffer(buffer2, (uint32_t)(dcb2.cursor_up - buffer2), st2);
        CHECK(st2.cx.count(0x20) == 1 && st2.cx[0x20] == 0x1 &&
              st2.cx.count(0x21) == 1 && st2.cx[0x21] == 0x2,
              "in-range indirect registers still fold around dropped neighbors");
        CHECK(st2.cx.count(0x80003FFD) == 0 && st2.cx.count(0x14225680) == 0,
              "out-of-range indirect offsets are dropped at ingest (#1264)");
        CHECK(st2.cx.count(0xFFFF) == 1 && st2.cx[0xFFFF] == 0x1234,
              "direct write at the last in-range offset (0xFFFF) folds");
        CHECK(st2.cx.count(0x10000) == 0,
              "direct write at the bound (0x10000) is dropped at ingest (#1264)");
    }


    // ACB uses the same packet descriptor and must enter the command processor instead of silently
    // disappearing. An empty reset packet is sufficient to prove that the async submit is folded.
    {
        static uint32_t acb_buffer[16]{};
        Dcb acb{};
        acb.bottom = acb_buffer; acb.top = acb_buffer + 16;
        acb.cursor_up = acb_buffer; acb.cursor_down = acb_buffer + 16;
        auto acb_reset = Hle::lookup("JrtiDtKeS38");
        CHECK(acb_reset && acb_reset((uint64_t)(uintptr_t)&acb, 0x3ff, 0, 0, 0, 0) != 0,
              "async-compute queue builder appends a packet");
        Packet packet{acb_buffer, (uint32_t)(acb.cursor_up - acb_buffer), {0,0,0,0}};
        uint64_t before = 0, ignored = 0; prosper_agc_submit_stats(&before, &ignored);
        CHECK(submit_acb(0x40, (uint64_t)(uintptr_t)&packet, 4, packet.dw_num, 0, 0) == 0,
              "SubmitAcb returns OK");
        uint64_t after = 0; prosper_agc_submit_stats(&after, &ignored);
        CHECK(after == before + 1, "SubmitAcb folds one async command stream");

        // ArcRunner (UE4 4.27) async-compute ABI (#1226): the redundant dword-count copy is in a2,
        // while a3 mirrors the QUEUE id (a0), NOT the count. The old `a3 == count` validation wrongly
        // returned kAgcErrInvalidArg for this valid submission, so UE4 aborted with
        // `Agc::submitAsyncCompute(...) failed 0x8a6c000a`. word[1] of the record is authoritative;
        // the register mirror may be a2 or a3. This case fails against the old check (a3=0x20 != count).
        uint64_t before2 = 0; prosper_agc_submit_stats(&before2, &ignored);
        CHECK(submit_acb(0x20 /*queue*/, (uint64_t)(uintptr_t)&packet,
                         packet.dw_num /*count mirrored in a2*/, 0x20 /*a3 = queue id, not count*/,
                         0, 0) == 0,
              "SubmitAcb accepts the ArcRunner ABI (count in a2, queue id mirrored in a3)");
        uint64_t after2 = 0; prosper_agc_submit_stats(&after2, &ignored);
        CHECK(after2 == before2 + 1, "ArcRunner-ABI SubmitAcb folds one async command stream");

        // Plucky sets the record's 32-bit FLAGS field to 1. Reading {count,flags} as one qword made
        // count 0x100000004 here and rejected the otherwise valid packet before range validation.
        Packet flagged_packet = packet;
        flagged_packet.pad[0] = 1;
        uint64_t before3 = 0; prosper_agc_submit_stats(&before3, &ignored);
        CHECK(submit_acb(0x20 /*queue*/, (uint64_t)(uintptr_t)&flagged_packet,
                         packet.dw_num /*count mirrored in a2*/, 0x20 /*queue mirror*/,
                         0, 0) == 0,
              "SubmitAcb reads Plucky's count and non-zero flags as separate 32-bit fields");
        uint64_t after3 = 0; prosper_agc_submit_stats(&after3, &ignored);
        CHECK(after3 == before3 + 1,
              "flagged Plucky-ABI SubmitAcb folds one async command stream");

        // Graphics and async compute have independent SH register files. Plucky submits real ACBs;
        // folding their user-data writes into the persistent graphics state replaced vertex shader
        // pointers with compute descriptors and made otherwise-supported gameplay draws disappear.
        static uint32_t gfx_buffer[16]{};
        Dcb gfx{};
        gfx.bottom = gfx_buffer; gfx.top = gfx_buffer + 16;
        gfx.cursor_up = gfx_buffer; gfx.cursor_down = gfx_buffer + 16;
        reset((uint64_t)(uintptr_t)&gfx, 0x3ff, 0, 0, 0, 0);
        setsh((uint64_t)(uintptr_t)&gfx, ((uint64_t)0x11112222u << 32) | 0x123u, 0, 0, 0, 0);
        Packet gfx_packet{gfx_buffer, (uint32_t)(gfx.cursor_up - gfx_buffer), {0,0,0,0}};
        CHECK(submit((uint64_t)(uintptr_t)&gfx_packet, 0, 0, 0, 0, 0) == 0,
              "graphics SH-register setup submits");

        static uint32_t isolated_acb_buffer[16]{};
        Dcb isolated_acb{};
        isolated_acb.bottom = isolated_acb_buffer;
        isolated_acb.top = isolated_acb_buffer + 16;
        isolated_acb.cursor_up = isolated_acb_buffer;
        isolated_acb.cursor_down = isolated_acb_buffer + 16;
        acb_reset((uint64_t)(uintptr_t)&isolated_acb, 0x3ff, 0, 0, 0, 0);
        setsh((uint64_t)(uintptr_t)&isolated_acb,
              ((uint64_t)0xaaaabbbbu << 32) | 0x123u, 0, 0, 0, 0);
        Packet isolated_packet{isolated_acb_buffer,
                               (uint32_t)(isolated_acb.cursor_up - isolated_acb_buffer),
                               {1,0,0,0}};
        CHECK(submit_acb(0x20, (uint64_t)(uintptr_t)&isolated_packet,
                         isolated_packet.dw_num, 0x20, 0, 0) == 0,
              "async-compute SH-register setup submits");
        static uint32_t second_acb_buffer[16]{};
        Dcb second_acb{};
        second_acb.bottom = second_acb_buffer; second_acb.top = second_acb_buffer + 16;
        second_acb.cursor_up = second_acb_buffer; second_acb.cursor_down = second_acb_buffer + 16;
        acb_reset((uint64_t)(uintptr_t)&second_acb, 0x3ff, 0, 0, 0, 0);
        setsh((uint64_t)(uintptr_t)&second_acb,
              ((uint64_t)0xccccddddu << 32) | 0x123u, 0, 0, 0, 0);
        Packet second_packet{second_acb_buffer,
                             (uint32_t)(second_acb.cursor_up - second_acb_buffer),
                             {1,0,0,0}};
        CHECK(submit_acb(0x40, (uint64_t)(uintptr_t)&second_packet,
                         second_packet.dw_num, 0x40, 0, 0) == 0,
              "second async-compute queue SH-register setup submits");
        uint32_t graphics_value = 0, compute_value = 0, second_compute_value = 0;
        CHECK(prosper_agc_submit_sh_reg(0, 0x123, &graphics_value) &&
              prosper_agc_submit_sh_reg(0x20, 0x123, &compute_value) &&
              prosper_agc_submit_sh_reg(0x40, 0x123, &second_compute_value) &&
              graphics_value == 0x11112222u && compute_value == 0xaaaabbbbu,
              "ACB SH writes remain isolated from persistent graphics bindings");
        CHECK(second_compute_value == 0xccccddddu,
              "independent async-compute queues retain independent SH bindings");
    }

    // A valid header at the last dword of a readable page must not make the decoder walk into the
    // inaccessible next page when the packet advertises two dwords.
    {
#ifdef _WIN32
        const size_t page_size = 0x1000;
        auto* pages = static_cast<uint8_t*>(VirtualAlloc(
            nullptr, page_size * 2, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        DWORD old_protect = 0;
        const bool protected_second = pages && VirtualProtect(
            pages + page_size, page_size, PAGE_NOACCESS, &old_protect) != 0;
#else
        const size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
        auto* pages = static_cast<uint8_t*>(mmap(
            nullptr, page_size * 2, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        const bool protected_second = pages != MAP_FAILED &&
            mprotect(pages + page_size, page_size, PROT_NONE) == 0;
#endif
        CHECK(protected_second, "created a readable-page/guard-page ACB boundary");
        if (protected_second) {
            auto* boundary_stream = reinterpret_cast<uint32_t*>(pages + page_size - 4);
            *boundary_stream = 0x80000000u;
            Packet packet{boundary_stream, 2, {0,0,0,0}};
            CHECK(submit_acb(0x40, (uint64_t)(uintptr_t)&packet, 4, 2, 0, 0) != 0,
                  "SubmitAcb rejects a stream whose full dword range crosses a guard page");
        }
#ifdef _WIN32
        if (pages) VirtualFree(pages, 0, MEM_RELEASE);
#else
        if (pages != MAP_FAILED) munmap(pages, page_size * 2);
#endif
    }

    // Fixed args 7-9 must be real function parameters, not compiler-frame offsets. This is the exact
    // first-fence field pattern whose missing Windows stack forwarding blocked the render thread (#672).
    {
        uint32_t sync_buffer[64] = {};
        Dcb sync{};
        sync.bottom = sync_buffer; sync.top = sync_buffer + 64;
        sync.cursor_up = sync_buffer; sync.cursor_down = sync_buffer + 64;
        uint64_t label = 0;
        const uint64_t S = (uint64_t)(uintptr_t)&sync;
        const uint64_t A = (uint64_t)(uintptr_t)&label;
        release(S, 0x28, 0, 1, 0, A, 2, 1, 0);
        waitmem(S, 8, 3, 0, 0, A, 1, 0xffffffffu, 0x20);

        // ReleaseMem is 8 dwords — the size of the RDNA2 RELEASE_MEM it stands for, which the guest
        // reserves space for (#1748); WaitRegMem is 9.
        CHECK(sync.cursor_up - sync_buffer == 17, "ReleaseMem + WaitRegMem emitted 8 + 9 dwords");
        CHECK(sync_buffer[3] == 2 && sync_buffer[4] == 1 && sync_buffer[5] == 0,
              "ReleaseMem encoded stack args data_sel=2 and data=1");
        const uint32_t* wait = sync_buffer + 8;
        CHECK(wait[3] == 0xffffffffu && wait[4] == 0 && wait[5] == 1 && wait[6] == 0,
              "WaitRegMem encoded stack args mask=0xffffffff and reference=1");
        CHECK(wait[7] == 3 && wait[8] == 0x20,
              "WaitRegMem encoded compare function and arg9 poll interval");
    }

    // Tagged submit imports must open their scope before validation because the generated return
    // hook runs for rejected calls too. Model Windows' same-thread re-entrancy window: the outer
    // handler has returned but its trampoline is dispatching guest exception code, which makes two
    // rejected tagged submits before the outer hook runs. Each nested hook must consume only the
    // scope opened by its own invocation.
    {
        auto submit_final = reinterpret_cast<HostHle9>(Hle::lookup("w1KFAHVqpaU"));
        auto submit_hook = Hle::return_hook_of("UglJIZjGssM");
        CHECK(submit_final && submit_hook &&
              submit_hook == Hle::return_hook_of("gSRnr79F8tQ") &&
              submit_hook == Hle::return_hook_of("w1KFAHVqpaU"),
              "all tagged submit imports expose their shared return hook");
        prosper_gpu_enable_post_submit_visibility();
        CHECK(submit(0, 0, 0, 0, 0, 0) != 0 && prosper_gpu_submit_scope_active(),
              "rejected outer DCB invocation opens a scope before validation");
        CHECK(submit_acb(0, 0, 0, 0, 0, 0) != 0 && prosper_gpu_submit_scope_active(),
              "rejected nested ACB invocation opens its own same-thread scope");
        submit_final(0, 0, 0, 0, 0, 0, 0, 0, 0);
        CHECK(prosper_gpu_submit_scope_active(),
              "rejected nested final-DCB invocation also opens its own scope");
        submit_hook();
        CHECK(prosper_gpu_submit_scope_active(),
              "first nested return hook preserves both enclosing scopes");
        submit_hook();
        CHECK(prosper_gpu_submit_scope_active(),
              "second nested return hook preserves the outer scope");
        submit_hook();
        CHECK(!prosper_gpu_submit_scope_active(),
              "outer return hook retires the final matching scope");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
