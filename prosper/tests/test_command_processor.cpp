// test_command_processor — validates the CommandProcessor's apply stage (src/gpu/command_processor).
// Builds a Draw Command Buffer via the real AGC Dcb functions, with a Set*RegistersIndirect packet
// pointing at a real {offset,value} array, then replays the stream into a GpuState and asserts the
// register files, index type, and draw list. Exercises decode+apply end-to-end against the real
// emitters; independent of the (SDK-gated) boot.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/gpu/command_processor.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper;
using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

struct Dcb {   // mirror of hle_agc.cpp's AgcDcb
    uint32_t* bottom; uint32_t* top; uint32_t* cursor_up; uint32_t* cursor_down;
    void* callback; void* user_data; uint32_t reserved_dw; uint32_t pad;
};

// Fold + drain: completion writes ride the modeled pipe-drain queue (#312) and become
// guest-visible at a drain point — tests assert the guest-visible (post-drain) state.
static size_t run_cb(const uint32_t* buf, size_t dwords, GpuState& st) {
    size_t n = run_command_buffer(buf, dwords, st);
    prosper_gpu_drain_completion_writes();
    return n;
}

int main() {
    printf("== test_command_processor ==\n");
    register_builtin_hle();

    auto reset = Hle::lookup("TRO721eVt4g");   // ResetQueue
    auto idx   = Hle::lookup("GIIW2J37e70");   // SetIndexSize
    auto setcx = Hle::lookup("ZvwO9euwYzc");   // SetCxRegistersIndirect
    auto setsh = Hle::lookup("-HOOCn0JY48");   // SetShRegistersIndirect
    auto draw  = Hle::lookup("Yw0jKSqop+E");   // DrawIndexAuto
    auto dispatch = Hle::lookup("k3GhuSNmBLU"); // DispatchDirect
    auto setcx_direct = Hle::lookup("LHFXRrlTPD8"); // SetCxRegisterDirect (#395 F5)
    auto setsh_direct = Hle::lookup("pFLArOT53+w"); // SetShRegisterDirect (#395 F5)
    auto setuc_direct = Hle::lookup("w4-d0n60hdo"); // SetUcRegisterDirect (#395 F5)
    CHECK(reset && idx && setcx && setsh && draw && dispatch &&
          setcx_direct && setsh_direct && setuc_direct, "AGC Dcb builders registered");
    if (!(reset && idx && setcx && setsh && draw && dispatch &&
          setcx_direct && setsh_direct && setuc_direct)) { printf("== FAIL ==\n"); return 1; }

    uint32_t buffer[256];
    memset(buffer, 0, sizeof buffer);
    Dcb dcb{};
    dcb.bottom = buffer; dcb.top = buffer + 256; dcb.cursor_up = buffer; dcb.cursor_down = buffer + 256;
    auto D = (uint64_t)(uintptr_t)&dcb;

    // Real register arrays the guest would build (offset, value pairs). Their addresses go into the
    // indirect packets; the CommandProcessor reads them 1:1 (guest VA == host addr in prosper).
    ShaderReg cx_regs[3] = { {0xA318u, 0x11111111u}, {0x00C0u, 0x22222222u}, {0x02DEu, 0x33333333u} };
    ShaderReg sh_regs[2] = { {0x2C0Cu, 0xAAAAAAAAu}, {0x2C0Du, 0xBBBBBBBBu} };

    reset(D, 0x3ff, 0, 0, 0, 0);
    idx(D, 2, 0, 0, 0, 0);
    setcx(D, (uint64_t)(uintptr_t)cx_regs, 3, 0, 0, 0);
    setsh(D, (uint64_t)(uintptr_t)sh_regs, 2, 0, 0, 0);
    auto pack_reg = [](uint32_t offset, uint32_t value) {
        return (uint64_t)offset | ((uint64_t)value << 32u);
    };
    setcx_direct(D, pack_reg(0xA318u, 0xCCCCCCCCu), 0, 0, 0, 0);
    setsh_direct(D, pack_reg(0x2C0Du, 0xDDDDDDDDu), 0, 0, 0, 0);
    setuc_direct(D, pack_reg(0x0123u, 0xEEEEEEEEu), 0, 0, 0, 0);
    draw(D, 0x0300, 0, 0, 0, 0);
    draw(D, 0x0006, 0, 0, 0, 0);

    GpuState st;
    size_t n = run_cb(buffer, 256, st);
    printf("  applied %zu packets\n", n);

    // Cx register file: the three offsets set to their values.
    CHECK(st.cx.size() == 3, "cx file has 3 registers");
    CHECK(st.cx.count(0xA318u) && st.cx[0xA318u] == 0xCCCCCCCCu,
          "direct Cx write overrides cx[0xA318]");
    CHECK(st.cx.count(0x00C0u) && st.cx[0x00C0u] == 0x22222222u, "cx[0x00C0] = 0x22222222");
    CHECK(st.cx.count(0x02DEu) && st.cx[0x02DEu] == 0x33333333u, "cx[0x02DE] = 0x33333333");

    // Sh register file: two registers; must NOT leak into cx/uc.
    CHECK(st.sh.size() == 2, "sh file has 2 registers");
    CHECK(st.sh.count(0x2C0Cu) && st.sh[0x2C0Cu] == 0xAAAAAAAAu, "sh[0x2C0C] = 0xAAAAAAAA");
    CHECK(st.sh.count(0x2C0Du) && st.sh[0x2C0Du] == 0xDDDDDDDDu,
          "direct Sh write overrides sh[0x2C0D]");
    CHECK(st.uc.size() == 1 && st.uc.count(0x0123u) && st.uc[0x0123u] == 0xEEEEEEEEu,
          "direct Uc write reaches only the user-config register file");

    CHECK(st.index_type == 2, "index_type = 2 (from SetIndexSize)");

    CHECK(st.draws.size() == 2, "2 draws recorded");
    CHECK(st.draws.size() == 2 && st.draws[0].index_count == 0x0300, "draw0 index_count = 0x300");
    CHECK(st.draws.size() == 2 && st.draws[1].index_count == 0x0006, "draw1 index_count = 0x6");

    // SET_SH_REG (IT_SET_SH_REG=0x76) sets a RANGE: payload[0]=start offset, payload[1..]=consecutive
    // values. The driver uploads the whole user-data descriptor block this way, so the decode+apply MUST
    // write every value — regressing to "first register only" silently drops the shaders' V#/T# SGPRs.
    {
        // Build a type-3 SET_SH_REG packet: header + [offset, v0,v1,v2,v3,v4] (5 registers @ 0x100).
        const uint32_t M = 6;                                   // payload dwords (offset + 5 values)
        uint32_t pkt[1 + M];
        pkt[0] = 0xC0000000u | ((M - 1u) << 16) | (0x76u << 8); // type3, len, op=IT_SET_SH_REG
        pkt[1] = 0x100u;                                        // start register offset
        for (uint32_t k = 0; k < 5; k++) pkt[2 + k] = 0xD0000000u + k;
        GpuState s2;
        run_cb(pkt, 1 + M, s2);
        CHECK(s2.sh.size() == 5, "SET_SH_REG range set all 5 registers (not just the first)");
        bool all = true;
        for (uint32_t k = 0; k < 5; k++) all &= (s2.sh.count(0x100u + k) && s2.sh[0x100u + k] == 0xD0000000u + k);
        CHECK(all, "SET_SH_REG range values sh[0x100..0x104] are the consecutive payload values");
    }

    // In-stream flip (sceAgcDcbSetFlip -> R_FLIP): the game flips via the Dcb, not the sceVideoOut
    // API, so the CommandProcessor MUST perform the videoout flip when it reaches the packet —
    // GetFlipStatus's count/flipArg/currentBuffer are what Unity's frame pacer polls before building
    // the next frame; a dropped in-stream flip stalls the game at one rendered frame.
    {
        auto setflip    = Hle::lookup("YUeqkyT7mEQ");                            // sceAgcDcbSetFlip
        auto flipstatus = Hle::lookup(nid_hash("sceVideoOutGetFlipStatus"));
        CHECK(setflip && flipstatus, "SetFlip builder + GetFlipStatus registered");
        if (setflip && flipstatus) {
            uint8_t st_before[0x40], st_after[0x40];
            flipstatus(0x1001, (uint64_t)(uintptr_t)st_before, 0, 0, 0, 0);
            uint32_t fbuf[64]; memset(fbuf, 0, sizeof fbuf);
            Dcb fd{}; fd.bottom = fbuf; fd.top = fbuf + 64; fd.cursor_up = fbuf; fd.cursor_down = fbuf + 64;
            setflip((uint64_t)(uintptr_t)&fd, 0x1001, 1, 2, 0x1234567890abcdefull, 0);
            GpuState s3;
            run_cb(fbuf, 64, s3);
            flipstatus(0x1001, (uint64_t)(uintptr_t)st_after, 0, 0, 0, 0);
            uint64_t cnt_b = *(uint64_t*)(st_before + 0x00), cnt_a = *(uint64_t*)(st_after + 0x00);
            int64_t  arg_a = *(int64_t*)(st_after + 0x18);
            int32_t  buf_a = *(int32_t*)(st_after + 0x38);
            CHECK(cnt_a == cnt_b + 1, "in-stream Flip advances GetFlipStatus.count");
            CHECK(arg_a == (int64_t)0x1234567890abcdefll, "in-stream Flip publishes its 64-bit flipArg");
            CHECK(buf_a == 1, "in-stream Flip publishes its buffer index");
        }
    }

    // Indexed draw (sceAgcDcbDrawIndex -> R_DRAW_INDEX, issue #63): the packet must land in
    // GpuState::draws carrying its index data — previously it decoded as an unknown NOP and every
    // indexed draw the guest submitted was silently dropped while the HLE returned success.
    {
        auto drawi = Hle::lookup("q88lQ+GP5Yk");                                 // sceAgcDcbDrawIndex
        CHECK(drawi != nullptr, "DrawIndex builder registered");
        if (drawi) {
            static uint16_t indices[6] = { 0, 1, 2, 0, 2, 3 };
            uint32_t ibuf[64]; memset(ibuf, 0, sizeof ibuf);
            Dcb id{}; id.bottom = ibuf; id.top = ibuf + 64; id.cursor_up = ibuf; id.cursor_down = ibuf + 64;
            auto ID = (uint64_t)(uintptr_t)&id;
            idx(ID, /*index_size*/ 2, 0, 0, 0, 0);              // 16-bit indices
            drawi(ID, /*index_count*/ 6, (uint64_t)(uintptr_t)indices, /*modifier*/ 0x40000000ull, 0, 0);
            GpuState s5;
            run_cb(ibuf, 64, s5);
            CHECK(s5.draws.size() == 1, "DrawIndex recorded one draw");
            if (s5.draws.size() == 1) {
                const auto& d = s5.draws[0];
                CHECK(d.indexed, "draw is marked indexed (fully-decoded packet)");
                CHECK(d.index_count == 6, "indexed draw index_count = 6");
                CHECK(d.index_addr == (uint64_t)(uintptr_t)indices, "index-buffer address carried through");
                CHECK(d.modifier == 0x40000000ull, "draw modifier carried through");
                CHECK(d.state && d.state->index_type == 2, "snapshot carries the index element size (SetIndexType)");
            }
        }
    }

    // Per-draw register snapshots: each DrawIndexAuto captures the register state AT the draw, so a
    // register changed between two draws is visible in the first draw's snapshot at its old value.
    // (Consecutive draws with no writes in between share one snapshot.)
    {
        uint32_t pkt[3];
        auto set_sh = [&](uint32_t off, uint32_t val, GpuState& s) {
            pkt[0] = 0xC0000000u | (1u << 16) | (0x76u << 8);   // SET_SH_REG, 2 payload dwords
            pkt[1] = off; pkt[2] = val;
            run_cb(pkt, 3, s);
        };
        uint32_t draw_pkt[3] = { 0xC0000000u | (1u << 16) | (0x10u << 8) | (0x04u << 2), 6, 0 };
        GpuState s4;
        set_sh(0x42, 0xAAAA, s4);
        run_cb(draw_pkt, 3, s4);                    // draw 0 under 0xAAAA
        run_cb(draw_pkt, 3, s4);                    // draw 1: no writes since draw 0
        set_sh(0x42, 0xBBBB, s4);
        run_cb(draw_pkt, 3, s4);                    // draw 2 under 0xBBBB
        CHECK(s4.draws.size() == 3 && s4.draws[0].state && s4.draws[2].state, "3 draws with snapshots");
        if (s4.draws.size() == 3 && s4.draws[0].state && s4.draws[2].state) {
            CHECK(s4.draws[0].state->sh.at(0x42) == 0xAAAA, "draw 0 snapshot holds the value AT the draw");
            CHECK(s4.draws[0].state == s4.draws[1].state, "no-write consecutive draws share one snapshot");
            CHECK(s4.draws[2].state->sh.at(0x42) == 0xBBBB, "draw 2 snapshot holds the updated value");
            CHECK(s4.sh.at(0x42) == 0xBBBB, "folded end state is the last write");
            CHECK(&s4.state_at_draw(0) == s4.draws[0].state.get(), "state_at_draw returns the snapshot");
        }
    }

    // Compute is not executed yet, but each DispatchDirect must retain its exact register state so
    // producer-provenance diagnostics can resolve the bound shader and resources (#524).
    {
        namespace P = prosper::agc::Pm4;
        uint32_t cbuf[128]; memset(cbuf, 0, sizeof cbuf);
        Dcb cd{}; cd.bottom = cbuf; cd.top = cbuf + 128; cd.cursor_up = cbuf; cd.cursor_down = cbuf + 128;
        auto CD = (uint64_t)(uintptr_t)&cd;
        ShaderReg cregs0[2] = {
            {P::COMPUTE_PGM_LO, 0x00123456u},
            {P::COMPUTE_USER_DATA_0 + 3, 0xAAAA1111u},
        };
        ShaderReg cregs1[1] = {{P::COMPUTE_USER_DATA_0 + 3, 0xBBBB2222u}};
        reset(CD, 0x3ff, 0, 0, 0, 0);
        setsh(CD, (uint64_t)(uintptr_t)cregs0, 2, 0, 0, 0);
        ShaderReg thread_regs[3] = {
            {P::COMPUTE_NUM_THREAD_X, 64}, {P::COMPUTE_NUM_THREAD_Y, 2}, {P::COMPUTE_NUM_THREAD_Z, 1},
        };
        setsh(CD, (uint64_t)(uintptr_t)thread_regs, 3, 0, 0, 0);
        dispatch(CD, 130, 5, 3, 0x21, 0); // USE_THREAD_DIMENSIONS
        setsh(CD, (uint64_t)(uintptr_t)cregs1, 1, 0, 0, 0);
        dispatch(CD, 8, 2, 3, 0x1, 0); // native workgroup dimensions
        GpuState cs;
        run_cb(cbuf, 128, cs);
        CHECK(cs.dispatch_count == 2 && cs.dispatches.size() == 2,
              "2 compute dispatches counted and retained");
        if (cs.dispatches.size() == 2) {
            CHECK(cs.dispatches[0].threads_x == 130 && cs.dispatches[0].threads_y == 5 &&
                  cs.dispatches[0].threads_z == 3, "dispatch 0 retains raw API dimensions");
            CHECK(cs.dispatches[0].modifier == 0x21,
                  "dispatch 0 modifier retained");
            auto launch = resolve_compute_launch(cs.dispatches[0]);
            CHECK(launch.local_x == 64 && launch.local_y == 2 && launch.local_z == 1,
                  "dispatch launch resolves local size from retained registers");
            CHECK(launch.groups_x == 3 && launch.groups_y == 3 && launch.groups_z == 3,
                  "dispatch launch ceil-divides non-divisible thread counts");
            CHECK(launch.threads_x == 130 && launch.threads_y == 5 && launch.threads_z == 3,
                  "thread-dimension dispatch retains its requested invocation extent");
            CHECK(cs.dispatches[0].state &&
                  cs.dispatches[0].state->sh.at(P::COMPUTE_USER_DATA_0 + 3) == 0xAAAA1111u,
                  "dispatch 0 snapshot retains pre-update compute user data");
            CHECK(cs.dispatches[1].state &&
                  cs.dispatches[1].state->sh.at(P::COMPUTE_USER_DATA_0 + 3) == 0xBBBB2222u,
                  "dispatch 1 snapshot retains updated compute user data");
            CHECK(cs.dispatches[1].threads_x == 8 && cs.dispatches[1].threads_y == 2 &&
                  cs.dispatches[1].threads_z == 3, "dispatch 1 asymmetric raw dimensions retained");
            launch = resolve_compute_launch(cs.dispatches[1]);
            CHECK(launch.groups_x == 8 && launch.groups_y == 2 && launch.groups_z == 3,
                  "group-dimension dispatch does not divide its dimensions a second time");
            CHECK(launch.threads_x == 512 && launch.threads_y == 4 && launch.threads_z == 3,
                  "group-dimension dispatch reports the full local-size-expanded thread extent");
            CHECK(cs.dispatches[0].command_order > 0 &&
                  cs.dispatches[0].command_order < cs.dispatches[1].command_order,
                  "dispatches retain monotonic PM4 stream order");
        }
    }

    // The decoder also accepts the older three-dimension custom packet used by focused tools. With
    // no trailing modifier it must deterministically use modifier=0 (workgroup dimensions), rather
    // than accidentally treating its dimensions as total threads.
    {
        constexpr uint32_t short_dispatch[] = {
            0xC0000000u | (2u << 16) | (0x10u << 8) | (0x1au << 2),
            11, 7, 3,
        };
        GpuState short_state;
        run_cb(short_dispatch, 4, short_state);
        CHECK(short_state.dispatches.size() == 1 && short_state.dispatches[0].modifier == 0,
              "short dispatch packet defaults its missing modifier to workgroup mode");
        if (short_state.dispatches.size() == 1) {
            const auto launch = resolve_compute_launch(short_state.dispatches[0]);
            CHECK(launch.groups_x == 11 && launch.groups_y == 7 && launch.groups_z == 3,
                  "short dispatch dimensions remain workgroup counts");
        }
    }

    // #911: the workgroup local size is COMPUTE_NUM_THREAD_*.NUM_THREAD_FULL [15:0]; a nonzero
    // NUM_THREAD_PARTIAL in bits [31:16] must NOT be folded into the dimension. Program X with a set
    // PARTIAL field (0x3 << 16) over a real count of 32; the resolved local_x must be 32, not 0x30020.
    {
        namespace P = prosper::agc::Pm4;
        alignas(4) uint32_t cbuf[128];
        Dcb cd{}; cd.bottom = cbuf; cd.top = cbuf + 128; cd.cursor_up = cbuf; cd.cursor_down = cbuf + 128;
        auto CD = (uint64_t)(uintptr_t)&cd;
        reset(CD, 0x3ff, 0, 0, 0, 0);
        ShaderReg thread_regs[3] = {
            {P::COMPUTE_NUM_THREAD_X, (0x3u << 16) | 32u},   // PARTIAL=3, FULL=32
            {P::COMPUTE_NUM_THREAD_Y, (0x7u << 16) | 4u},    // PARTIAL=7, FULL=4
            {P::COMPUTE_NUM_THREAD_Z, 1u},
        };
        setsh(CD, (uint64_t)(uintptr_t)thread_regs, 3, 0, 0, 0);
        dispatch(CD, 32, 4, 1, 0x1, 0);   // native workgroup dimensions
        GpuState cs;
        run_cb(cbuf, 128, cs);
        CHECK(cs.dispatches.size() == 1, "#911 masked-thread dispatch counted");
        if (cs.dispatches.size() == 1) {
            const auto launch = resolve_compute_launch(cs.dispatches[0]);
            CHECK(launch.local_x == 32 && launch.local_y == 4 && launch.local_z == 1,
                  "#911: NUM_THREAD local size masks to FULL [15:0], ignoring PARTIAL [31:16]");
        }
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
