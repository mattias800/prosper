// test_command_provenance — pin the #305 SH-register provenance contract without a title boot.
//
// The diagnostic must distinguish the submit entry point (Dcb / Acb / DcbFinal), top-level fold,
// Jump depth, direct-vs-indirect write path, and write-vs-draw order.  A plausible value alone is
// not evidence of where it came from, so this test drives the real AGC packet builders and PM4
// decoder, then checks the maps retained by the draw snapshots.
#include "../src/gpu/command_processor.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include "../src/hle/dispatch.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>

using namespace prosper;
using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

struct Dcb {
    uint32_t* bottom;
    uint32_t* top;
    uint32_t* cursor_up;
    uint32_t* cursor_down;
    void* callback;
    void* user_data;
    uint32_t reserved_dw;
    uint32_t pad;
};

static Dcb make_dcb(uint32_t* words, size_t count) {
    Dcb dcb{};
    dcb.bottom = words;
    dcb.top = words + count;
    dcb.cursor_up = words;
    dcb.cursor_down = words + count;
    return dcb;
}

static size_t used_dwords(const Dcb& dcb) {
    return static_cast<size_t>(dcb.cursor_up - dcb.bottom);
}

struct Source {
    uint8_t origin = 0;
    uint8_t jump_depth = 0;
    uint32_t fold = 0;
};

static Source unpack_source(uint64_t packed) {
    return {
        static_cast<uint8_t>(packed & 0xffu),
        static_cast<uint8_t>((packed >> 8u) & 0xffu),
        static_cast<uint32_t>(packed >> 16u),
    };
}

static uint64_t pack_reg(uint32_t offset, uint32_t value) {
    return static_cast<uint64_t>(offset) | (static_cast<uint64_t>(value) << 32u);
}

int main() {
    std::printf("== test_command_provenance ==\n");
#ifdef _WIN32
    _putenv_s("PROSPER_UDPROV", "1");
#else
    setenv("PROSPER_UDPROV", "1", 1);
#endif
    CHECK(udprov_enabled(), "PROSPER_UDPROV positive control is armed");

    register_builtin_hle();
    auto setsh = Hle::lookup("-HOOCn0JY48");       // sceAgcDcbSetShRegistersIndirect
    auto setsh_direct = Hle::lookup("pFLArOT53+w"); // sceAgcDcbSetShRegisterDirect
    auto draw = Hle::lookup("Yw0jKSqop+E");         // sceAgcDcbDrawIndexAuto
    auto jump = Hle::lookup("xSAR0LTcRKM");         // sceAgcDcbJump
    CHECK(setsh && setsh_direct && draw && jump,
          "AGC indirect/direct/draw/Jump builders are registered");
    if (!(setsh && setsh_direct && draw && jump)) return 1;

    namespace P = prosper::agc::Pm4;
    constexpr uint32_t kUser = P::SPI_SHADER_USER_DATA_GS_0;

    // q1 / Dcb: bind two program registers indirectly, then draw.  The snapshot must retain the
    // indirect path bit and the root source identity from this first top-level fold.
    uint32_t q1_words[64]{};
    Dcb q1 = make_dcb(q1_words, 64);
    ShaderReg q1_bind[] = {
        {P::SPI_SHADER_PGM_LO_ES, 0x11110000u},
        {P::SPI_SHADER_PGM_RSRC2_GS, 0x0000000cu},
    };
    setsh(reinterpret_cast<uint64_t>(&q1), reinterpret_cast<uint64_t>(q1_bind),
          std::size(q1_bind), 0, 0, 0);
    draw(reinterpret_cast<uint64_t>(&q1), 3, 0, 0, 0, 0);

    GpuState graphics;
    prosper_gpu_set_fold_origin(1);
    run_command_buffer(q1_words, used_dwords(q1), graphics);
    CHECK(graphics.draws.size() == 1 && graphics.draws[0].state,
          "q1 bind+draw produces a retained snapshot");
    if (graphics.draws.size() != 1 || !graphics.draws[0].state) return 1;
    const auto q1_snapshot = graphics.draws[0].state;
    const uint64_t q1_prov = q1_snapshot->sh_prov.at(P::SPI_SHADER_PGM_LO_ES);
    const Source q1_src = unpack_source(
        q1_snapshot->sh_prov_src.at(P::SPI_SHADER_PGM_LO_ES));
    CHECK((q1_prov & GpuState::kProvIndirect) != 0,
          "q1 program bind records the indirect write path");
    CHECK((q1_prov & ~GpuState::kProvIndirect) < graphics.draws[0].command_order,
          "q1 program bind order precedes its draw order");
    CHECK(q1_src.origin == 1, "q1 program bind records Dcb origin");
    CHECK(q1_src.jump_depth == 0, "q1 program bind records root Jump depth");
    CHECK(q1_src.fold != 0, "q1 program bind records a top-level fold identity");

    // q3 / DcbFinal: write the head of the user-data block directly in the root stream, then call
    // an inner segment which writes the tail and draws.  The draw sees one fold, but two depths.
    uint32_t inner_words[64]{};
    Dcb inner = make_dcb(inner_words, 64);
    setsh_direct(reinterpret_cast<uint64_t>(&inner), pack_reg(kUser + 2, 0x33333333u),
                 0, 0, 0, 0);
    setsh_direct(reinterpret_cast<uint64_t>(&inner), pack_reg(kUser + 3, 0x44444444u),
                 0, 0, 0, 0);
    draw(reinterpret_cast<uint64_t>(&inner), 6, 0, 0, 0, 0);

    uint32_t q3_words[64]{};
    Dcb q3 = make_dcb(q3_words, 64);
    setsh_direct(reinterpret_cast<uint64_t>(&q3), pack_reg(kUser + 0, 0x11111111u),
                 0, 0, 0, 0);
    setsh_direct(reinterpret_cast<uint64_t>(&q3), pack_reg(kUser + 1, 0x22222222u),
                 0, 0, 0, 0);
    jump(reinterpret_cast<uint64_t>(&q3), 0, 0,
         reinterpret_cast<uint64_t>(inner_words), used_dwords(inner), 0);
    draw(reinterpret_cast<uint64_t>(&q3), 9, 0, 0, 0, 0);

    graphics.draws.clear();
    prosper_gpu_set_fold_origin(3);
    run_command_buffer(q3_words, used_dwords(q3), graphics);
    CHECK(graphics.draws.size() == 2 && graphics.draws[0].state,
          "q3 root and inner Jump draws are both retained");
    if (graphics.draws.size() != 2 || !graphics.draws[0].state) return 1;
    const auto q3_snapshot = graphics.draws[0].state;
    const uint64_t q3_root_prov = q3_snapshot->sh_prov.at(kUser + 0);
    const uint64_t q3_jump_prov = q3_snapshot->sh_prov.at(kUser + 2);
    const Source q3_root_src = unpack_source(q3_snapshot->sh_prov_src.at(kUser + 0));
    const Source q3_jump_src = unpack_source(q3_snapshot->sh_prov_src.at(kUser + 2));
    CHECK((q3_root_prov & GpuState::kProvIndirect) == 0 &&
              (q3_jump_prov & GpuState::kProvIndirect) == 0,
          "q3 user-data writes record the direct write path");
    CHECK(q3_root_src.origin == 3 && q3_jump_src.origin == 3,
          "q3 root and Jump writes record DcbFinal origin");
    CHECK(q3_root_src.fold > q1_src.fold,
          "q3 top-level fold identity advances beyond q1");
    CHECK(q3_jump_src.fold == q3_root_src.fold,
          "inner Jump retains its parent top-level fold identity");
    CHECK(q3_root_src.jump_depth == 0,
          "q3 root user data records root Jump depth");
    CHECK(q3_jump_src.jump_depth == 1,
          "q3 inner user data records Jump depth one");
    CHECK((q3_root_prov & ~GpuState::kProvIndirect) <
              (q3_jump_prov & ~GpuState::kProvIndirect) &&
              (q3_jump_prov & ~GpuState::kProvIndirect) < graphics.draws[0].command_order,
          "q3 provenance orders root write, Jump write, then inner draw");
    CHECK(q3_snapshot->sh.at(kUser + 0) == 0x11111111u &&
              q3_snapshot->sh.at(kUser + 2) == 0x33333333u,
          "q3 draw snapshot retains root and Jump user-data values");

    // q2 / Acb uses a separate register state in production.  A synthetic q2 fold proves the
    // origin discriminator independently and verifies that no q1/q3 graphics state leaked in.
    uint32_t q2_words[64]{};
    Dcb q2 = make_dcb(q2_words, 64);
    setsh_direct(reinterpret_cast<uint64_t>(&q2), pack_reg(kUser + 4, 0x55555555u),
                 0, 0, 0, 0);
    draw(reinterpret_cast<uint64_t>(&q2), 12, 0, 0, 0, 0);
    GpuState compute;
    prosper_gpu_set_fold_origin(2);
    run_command_buffer(q2_words, used_dwords(q2), compute);
    CHECK(compute.draws.size() == 1 && compute.draws[0].state,
          "q2 write+work produces its own retained snapshot");
    if (compute.draws.size() == 1 && compute.draws[0].state) {
        const auto q2_snapshot = compute.draws[0].state;
        const Source q2_src = unpack_source(q2_snapshot->sh_prov_src.at(kUser + 4));
        CHECK(q2_src.origin == 2, "q2 user data records Acb origin");
        CHECK(q2_src.jump_depth == 0, "q2 user data records root Jump depth");
        CHECK(q2_src.fold > q3_root_src.fold,
              "q2 top-level fold identity is distinct from q3");
        CHECK(q2_snapshot->sh.count(P::SPI_SHADER_PGM_LO_ES) == 0 &&
                  q2_snapshot->sh.count(kUser + 0) == 0,
              "q2 state does not inherit q1/q3 graphics registers");
    }

    prosper_gpu_set_fold_origin(0);
    std::printf("== %s ==\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
