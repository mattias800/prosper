// #5052: the shipping producer must pass observed FRONT_FACE encoding to the
// compiler and its cache. A warm draw cannot reuse the other raw-word convention.
#include "fixtures/ps_pull_model_fixture.hpp"
#include <gtest/gtest.h>

namespace prosper::gpu {
namespace {
namespace fixture = prosper::test::ps_pull;
constexpr uint32_t face = 1u << 12, all_bits = 1u << 24;
fixture::Owners* programs() {
    prosper::register_builtin_hle();
    const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    uint64_t address = 0;
    if (!map ||
        map(reinterpret_cast<uint64_t>(&address), 0x10000, 2, 0,
            reinterpret_cast<uint64_t>("front-face-programs"), 0) != 0 ||
        address < 0x100000000ull || address % alignof(fixture::Owners))
        return nullptr;
    // The AGC registry retains the owner. Keep this bounded mapping unchanged for
    // process lifetime, including repeated cache arms using the same raw program.
    auto& owner = *std::construct_at(reinterpret_cast<fixture::Owners*>(address));
    const std::vector<uint32_t> ps{
        0x7d040080u, // genuine CMP_EQ_I32 VCC,0,FRONT_FACE at v0
        0x7e120280u, 0x021012f2u, 0x7e140280u, 0x7e1602f2u, 0xf800180fu, 0x0b0a0908u, 0xbf810000u,
    };
    if (!fixture::register_program(owner.vs, true, fixture::vertex_words(fixture::ClipW::One)) ||
        !fixture::register_program(owner.ps, false, ps))
        return nullptr;
    return &owner;
}
bool realize(fixture::Owners& owner, bool available, uint32_t baryc, DrawItem& out,
             OperationRealizationFailure* failure = nullptr) {
    GpuState state;
    for (const auto* program : {&owner.vs, &owner.ps})
        for (const auto& reg : program->registers) state.sh[reg.offset] = reg.value;
    state.uc[prosper::agc::Pm4::VGT_PRIMITIVE_TYPE] = 4;
    state.cx[prosper::agc::Pm4::CB_TARGET_MASK] = state.cx[prosper::agc::Pm4::CB_SHADER_MASK] = 15;
    state.cx[prosper::agc::Pm4::SPI_PS_IN_CONTROL] = 0;
    state.cx[prosper::agc::Pm4::SPI_PS_INPUT_ENA] = face;
    state.cx[prosper::agc::Pm4::SPI_PS_INPUT_ADDR] = face;
    if (available) state.cx[prosper::agc::Pm4::SPI_BARYC_CNTL] = baryc;
    state.sh[prosper::agc::Pm4::SPI_SHADER_PGM_RSRC2_PS] = 0;
    GpuState::Draw packet;
    packet.index_count = 3;
    packet.instance_count = 1;
    packet.command_order = 1;
    state.draws.push_back(packet);
    return realize_draw_item(state, &state.draws[0], 3, 64, false, out, failure, true);
}
} // namespace

TEST(FrontFaceProducer, ObservedZeroAndBit24ProduceDistinctCompiledModules) {
    auto* owner = programs();
    ASSERT_NE(owner, nullptr);
    DrawItem signed_float, boolean, warm;
    ASSERT_TRUE(realize(*owner, true, 0, signed_float));
    ASSERT_TRUE(realize(*owner, true, all_bits, boolean));
    ASSERT_TRUE(realize(*owner, true, all_bits, warm));
    EXPECT_TRUE(signed_float.has_system_inputs && boolean.has_system_inputs);
    EXPECT_EQ(signed_float.system_inputs.front_face_encoding,
              FragmentFrontFaceEncoding::SignedFloat);
    EXPECT_EQ(boolean.system_inputs.front_face_encoding, FragmentFrontFaceEncoding::AllBits);
    EXPECT_TRUE(signed_float.ps_raster_launch.baryc_cntl_available);
    EXPECT_EQ(signed_float.ps_raster_launch.baryc_cntl, 0u);
    ASSERT_FALSE(signed_float.fs_words().empty());
    ASSERT_FALSE(boolean.fs_words().empty());
    EXPECT_NE(signed_float.fs_words(), boolean.fs_words());
    EXPECT_EQ(boolean.fs_words(), warm.fs_words());
    EXPECT_NE(signed_float.fs_identity, boolean.fs_identity);
    EXPECT_EQ(boolean.fs_identity, warm.fs_identity);
}

TEST(FrontFaceProducer, UnobservedBarycentricRegisterDoesNotBorrowCachedEncoding) {
    auto* owner = programs();
    ASSERT_NE(owner, nullptr);
    DrawItem known, unknown;
    ASSERT_TRUE(realize(*owner, true, all_bits, known));
    OperationRealizationFailure failure;
    EXPECT_FALSE(realize(*owner, false, 0, unknown, &failure));
    EXPECT_TRUE(failure.fragment_retry_config_available && failure.has_system_inputs);
    EXPECT_EQ(failure.system_inputs.front_face_encoding, FragmentFrontFaceEncoding::Unavailable);
    EXPECT_TRUE(unknown.fs_words().empty());
    EXPECT_NE(last_terminal_reject_reason(reinterpret_cast<uint64_t>(owner->ps.code.data()))
                  .find("fabricated-fresh-mask-source"),
              std::string::npos);
}
} // namespace prosper::gpu
