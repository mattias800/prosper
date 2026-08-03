// test_command_provenance — pin the #305 SH-register provenance contract without a title boot.
//
// The diagnostic must distinguish the submit entry point (Dcb / Acb / DcbFinal), top-level fold,
// Jump depth, direct-vs-indirect write path, and write-vs-draw order.  A plausible value alone is
// not evidence of where it came from, so this test drives the real AGC packet builders and PM4
// decoder, then checks the maps retained by the draw snapshots.
#include "../src/gpu/command_processor.hpp"
#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include "../src/hle/dispatch.hpp"
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <mutex>
#include <thread>

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

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

int main() {
    std::printf("== test_command_provenance ==\n");
#ifdef _WIN32
    _putenv_s("PROSPER_UDPROV", "");
    _putenv_s("PROSPER_GPU_CAPTURE_RESOURCE_PROVENANCE", "0:ps:32");
#else
    unsetenv("PROSPER_UDPROV");
    setenv("PROSPER_GPU_CAPTURE_RESOURCE_PROVENANCE", "0:ps:32", 1);
#endif
    const bool selector_only_provenance =
        std::getenv("PROSPER_UDPROV") == nullptr && udprov_enabled();
    CHECK(selector_only_provenance,
          "resource-provenance selector alone arms SH write provenance");
    if (!selector_only_provenance) {
        std::printf("== FAIL ==\n");
        return 1;
    }

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

    // #1853 cross-thread positive control. A Dcb submitter writes the raw PS V#; only after an
    // explicit hand-off does a distinct DcbFinal submitter emit the draw. The selected capture must
    // attach the raw input to the draw's immutable semantic snapshot and name q1 as the last writer,
    // not the host thread/q3 path that happened to issue the draw.
    std::array<uint8_t, 32> cross_thread_bytes{};
    const uint64_t cross_thread_addr =
        reinterpret_cast<uint64_t>(cross_thread_bytes.data());
    const std::array<uint32_t, 4> cross_thread_vsharp = {
        static_cast<uint32_t>(cross_thread_addr),
        static_cast<uint32_t>((cross_thread_addr >> 32u) & 0xffffu) | (16u << 16u),
        1u,
        (22u << 12u) | 0xFACu,
    };
    uint32_t writer_words[64]{};
    Dcb writer_dcb = make_dcb(writer_words, std::size(writer_words));
    for (uint32_t i = 0; i < cross_thread_vsharp.size(); ++i)
        setsh_direct(reinterpret_cast<uint64_t>(&writer_dcb),
                     pack_reg(P::SPI_SHADER_USER_DATA_PS_0 + i,
                              cross_thread_vsharp[i]),
                     0, 0, 0, 0);
    uint32_t consumer_words[64]{};
    Dcb consumer_dcb = make_dcb(consumer_words, std::size(consumer_words));
    draw(reinterpret_cast<uint64_t>(&consumer_dcb), 3, 0, 0, 0, 0);

    // Defect-shaped mutation: leave the selector and capture path armed, but disable the collector
    // only for this fixture.  Earlier .at()-based provenance checks therefore remain safe, and the
    // exact selected-witness assertion below—not the activation predicate—must be the named red.
    const bool mutate_collection =
        std::getenv("PROSPER_TEST_MUTATE_RESOURCE_INPUT_COLLECTION") != nullptr;
    if (mutate_collection)
        set_test_env("PROSPER_TEST_DISABLE_UDPROV_COLLECTION", "1");

    GpuState cross_thread_state;
    std::mutex handoff_mutex;
    std::condition_variable handoff_cv;
    bool writer_finished = false;
    std::thread writer_thread([&] {
        prosper_gpu_set_fold_origin(1);
        run_command_buffer(writer_words, used_dwords(writer_dcb), cross_thread_state);
        {
            std::lock_guard<std::mutex> lock(handoff_mutex);
            writer_finished = true;
        }
        handoff_cv.notify_one();
    });
    std::thread draw_thread([&] {
        std::unique_lock<std::mutex> lock(handoff_mutex);
        handoff_cv.wait(lock, [&] { return writer_finished; });
        lock.unlock();
        prosper_gpu_set_fold_origin(3);
        run_command_buffer(consumer_words, used_dwords(consumer_dcb), cross_thread_state);
    });
    writer_thread.join();
    draw_thread.join();

    uint32_t overwrite_words[64]{};
    Dcb overwrite_dcb = make_dcb(overwrite_words, std::size(overwrite_words));
    for (uint32_t i = 0; i < cross_thread_vsharp.size(); ++i)
        setsh_direct(reinterpret_cast<uint64_t>(&overwrite_dcb),
                     pack_reg(P::SPI_SHADER_USER_DATA_PS_0 + i,
                              cross_thread_vsharp[i] ^ 0x01010101u),
                     0, 0, 0, 0);
    prosper_gpu_set_fold_origin(2);
    run_command_buffer(overwrite_words, used_dwords(overwrite_dcb), cross_thread_state);
    const bool folded_end_was_overwritten =
        cross_thread_state.sh.at(P::SPI_SHADER_USER_DATA_PS_0) !=
            cross_thread_vsharp[0];

    DrawItem cross_thread_draw;
    cross_thread_draw.draw_index = 0;
    if (!cross_thread_state.draws.empty())
        cross_thread_draw.command_order = cross_thread_state.draws[0].command_order;
    auto cross_thread_table = std::make_shared<ShaderResourceTable>();
    ShaderResource cross_thread_resource;
    cross_thread_resource.cls = ResourceClass::ConstantBuffer;
    cross_thread_resource.format = DataFormat::Float32;
    cross_thread_resource.num_components = 1;
    cross_thread_resource.binding = 32;
    cross_thread_resource.gpu_addr = cross_thread_addr;
    cross_thread_resource.size = 16;
    cross_thread_resource.stride = 16;
    cross_thread_resource.sgpr_base = 0;
    cross_thread_resource.direct_vsharp_sh_register_base =
        P::SPI_SHADER_USER_DATA_PS_0;
    cross_thread_draw.prt = cross_thread_table;
    cross_thread_table->resources.push_back(cross_thread_resource);
    PendingGpuCapture cross_thread_pending;
    cross_thread_pending.resource_provenance_armed = true;
    cross_thread_pending.resource_provenance_selector = {
        0, ShaderProgramStage::Fragment, 32};
    snapshot_pending_gpu_capture_draw_resource(
        &cross_thread_pending, cross_thread_draw, {}, &cross_thread_state);

    const auto& cross_thread_witness = cross_thread_pending.resource_provenance;
    bool all_q1_direct_before_draw = cross_thread_witness.input_write_provenance_mask == 0x0f;
    uint64_t writer_fold = 0;
    for (uint32_t i = 0; all_q1_direct_before_draw && i < 4; ++i) {
        const uint64_t write = cross_thread_witness.input_last_writes[i];
        const Source source = unpack_source(cross_thread_witness.input_write_sources[i]);
        if (!i) writer_fold = source.fold;
        all_q1_direct_before_draw =
            (write & GpuState::kProvIndirect) == 0 &&
            (write & ~GpuState::kProvIndirect) < cross_thread_draw.command_order &&
            source.origin == 1 && source.jump_depth == 0 && source.fold == writer_fold;
    }
    CHECK(cross_thread_state.draws.size() == 1 &&
              cross_thread_witness.input_mode ==
                  GpuCaptureResourceInputMode::DirectVSharp &&
              cross_thread_witness.input_dwords == cross_thread_vsharp &&
              gpu_capture_resource_input_verdict(cross_thread_witness,
                                                 cross_thread_resource) ==
                  GpuCaptureResourceInputVerdict::FullMatch &&
              writer_fold != 0 && all_q1_direct_before_draw &&
              folded_end_was_overwritten,
          "selected draw uses immutable pre-overwrite q1 V# state and full-match identity");

    if (mutate_collection) {
        set_test_env("PROSPER_TEST_DISABLE_UDPROV_COLLECTION", nullptr);
        prosper_gpu_set_fold_origin(0);
        std::printf("== %s ==\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    }

    DrawItem mismatched_draw = cross_thread_draw;
    mismatched_draw.prt = std::make_shared<ShaderResourceTable>(*cross_thread_table);
    auto& mismatched_resource = mismatched_draw.prt->resources[0];
    mismatched_resource.gpu_addr += 16;
    PendingGpuCapture mismatched_pending;
    mismatched_pending.resource_provenance_armed = true;
    mismatched_pending.resource_provenance_selector = {
        0, ShaderProgramStage::Fragment, 32};
    snapshot_pending_gpu_capture_draw_resource(
        &mismatched_pending, mismatched_draw, {}, &cross_thread_state);
    CHECK(mismatched_pending.resource_provenance.input_dwords ==
              cross_thread_vsharp &&
              gpu_capture_resource_input_verdict(
                  mismatched_pending.resource_provenance,
                  mismatched_resource) ==
                  GpuCaptureResourceInputVerdict::Mismatch,
          "raw SH origin is selected independently when normalized output mismatches");

    GpuState missing_provenance_state = cross_thread_state;
    if (!missing_provenance_state.draws.empty() &&
        missing_provenance_state.draws[0].state) {
        auto incomplete = std::make_shared<GpuState>(
            *missing_provenance_state.draws[0].state);
        incomplete->sh_prov.erase(P::SPI_SHADER_USER_DATA_PS_0 + 3);
        missing_provenance_state.draws[0].state = std::move(incomplete);
    }
    PendingGpuCapture missing_provenance_pending;
    missing_provenance_pending.resource_provenance_armed = true;
    missing_provenance_pending.resource_provenance_selector = {
        0, ShaderProgramStage::Fragment, 32};
    snapshot_pending_gpu_capture_draw_resource(
        &missing_provenance_pending, cross_thread_draw, {},
        &missing_provenance_state);
    const auto& missing_provenance =
        missing_provenance_pending.resource_provenance;
    CHECK(missing_provenance.input_mode ==
              GpuCaptureResourceInputMode::Unavailable &&
              missing_provenance.input_unavailable_reason ==
                  GpuCaptureResourceInputUnavailableReason::MissingWriteProvenance &&
              missing_provenance.input_write_provenance_mask == 0 &&
              missing_provenance.input_sh_register_base == 0xFFFFFFFFu &&
              std::all_of(missing_provenance.input_dwords.begin(),
                          missing_provenance.input_dwords.end(),
                          [](uint32_t value) { return value == 0; }),
          "partial SH provenance stays explicitly unavailable and inert");

    prosper_gpu_set_fold_origin(0);
    std::printf("== %s ==\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
