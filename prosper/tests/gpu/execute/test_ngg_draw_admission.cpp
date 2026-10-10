// Live admission of merged ES+GS NGG draws (#3135 P5): the register and draw-shape table
// (admit_ngg_draw), the device half (ngg_device_refusal) and the live producer
// (realize_ngg_live_draw) with its caches. Pure CPU: the Kena arms compile the captured LUT
// producer chain (tests/data) but run nothing.
//
// Every refusal arm changes ONE input of Kena's recorded draw (KENA_STATUS.md and #3135: stages
// 0x2030, onchip 0x10020040, GE_CNTL 0x8040, 192 / 3 / ITEMSIZE 4, RSRC2_GS 0x008b0000,
// PA_SU_SC_MODE_CNTL 0x240, PA_CL_VS_OUT_CNTL 0x01240000, VGT_GS_OUT_PRIM_TYPE 2, a 4-vertex
// strip into a 32-slice volume), so each one is beside an admitted twin.
#include "fixtures/ngg_merged_lut_fixture.hpp"
#include "fixtures/test_data.h"
#include "gpu/execute/ngg_draw_admission.hpp"
#include "gpu/execute/ngg_live_draw.hpp"
#include "gpu/execute/ngg_subgroup_draw.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test;

namespace {

NggDrawRegisters kena_registers() {
    NggDrawRegisters r;
    r.vgt_shader_stages_en = 0x00002030u;
    r.vgt_gs_onchip_cntl = 0x10020040u;
    r.ge_cntl = 0x8040u;
    r.ge_max_output_per_subgroup = 0xc0u;
    r.vgt_gs_max_vert_out = 3u;
    r.vgt_esgs_ring_itemsize = 4u;
    r.spi_shader_pgm_rsrc2_gs = ngg::kKenaRsrc2Gs;
    r.vgt_gs_out_prim_type = 2u;
    r.pa_su_sc_mode_cntl = 0x240u;
    r.pa_cl_vs_out_cntl = 0x01240000u;
    r.pa_cl_clip_cntl = 0x01080000u;
    r.spi_ps_input_ena = 0x2020u;
    r.primitive_type = 6u;   // triangle strip
    return r;
}

NggDrawFacts kena_facts(uint32_t instances = 32) {
    NggDrawFacts f;
    f.vertex_count = 4;
    f.instance_count = instances;
    f.target_slices = 32;
    f.target_first_slice = 0;
    f.user_data_range_known = true;
    f.user_data_range_end = ngg::kKenaUserSgprs;
    return f;
}

// A RADV-like device (the P4 backend tests ran on one).
NggHostCapabilities radv() {
    NggHostCapabilities h;
    h.published = true;
    h.compute = true;
    h.vertex_pipeline_stores = true;
    h.shader_output_layer = true;
    h.geometry_shader = true;
    h.native_wave64 = true;
    h.max_compute_workgroup_subgroups = 16;
    h.max_compute_shared_memory = 65536;
    h.max_compute_workgroup_size_x = 1024;
    h.max_compute_workgroup_invocations = 1024;
    h.max_compute_workgroup_count_x = 65535;
    h.max_storage_buffer_range = 1u << 30;
    h.max_push_constants_size = 256;
    return h;
}

const char* refusal(const NggDrawRegisters& r, const NggDrawFacts& f,
                    const NggHostCapabilities& h = radv()) {
    const NggDrawAdmission a = admit_ngg_draw(r, f, h);
    EXPECT_TRUE(a.applies);
    return a.refusal ? a.refusal : "admitted";
}

// ---- The register table ----------------------------------------------------------------------------

TEST(NggDrawAdmission, KenaDecodes) {
    const NggDrawAdmission a = admit_ngg_draw(kena_registers(), kena_facts(), radv());
    ASSERT_TRUE(a.ok()) << a.refusal;
    const NggSubgroupLimits expected = ngg::kena_limits();
    EXPECT_EQ(a.limits.es_verts_per_subgroup, expected.es_verts_per_subgroup);
    EXPECT_EQ(a.limits.gs_prims_per_subgroup, expected.gs_prims_per_subgroup);
    EXPECT_EQ(a.limits.max_out_verts_per_subgroup, 192u);
    EXPECT_EQ(a.limits.gs_max_vert_out, 3u);
    EXPECT_EQ(a.limits.esgs_item_size, 4u);
    EXPECT_EQ(a.shape.topology, NggInputTopology::TriangleStrip);
    EXPECT_EQ(a.shape.vertex_count, 4u);
    EXPECT_EQ(a.shape.instance_count, 32u);
    EXPECT_EQ(a.lds_granules, 17u) << "RSRC2_GS 0x008b0000: LDS_SIZE [26:19] = 0x11";
    EXPECT_EQ(a.user_sgprs, 8u) << "user_data_range_end, with RSRC2_GS.USER_SGPR = 0";
    EXPECT_EQ(a.topology, NggOutputTopology::TriangleList);
    EXPECT_FALSE(a.provoking_vertex_last);
    EXPECT_TRUE(a.layer_from_pos1) << "PA_CL_VS_OUT_CNTL bit 18";
    EXPECT_EQ(a.layer_slices, 32u);
    EXPECT_EQ(a.route, NggLayerRoute::ShaderOutputLayer);
    EXPECT_TRUE(a.count_violations);
    EXPECT_TRUE(a.native_wave64);

    auto r = kena_registers();
    r.pa_su_sc_mode_cntl |= 1u << 19;
    EXPECT_TRUE(admit_ngg_draw(r, kena_facts(), radv()).provoking_vertex_last);
    r = kena_registers();
    r.pa_cl_vs_out_cntl &= ~kVsOutUseVtxRenderTargetIndx;
    const NggDrawAdmission unlayered = admit_ngg_draw(r, kena_facts(), radv());
    EXPECT_FALSE(unlayered.layer_from_pos1);
    EXPECT_EQ(unlayered.route, NggLayerRoute::None);
    r = kena_registers();
    r.primitive_type = 4u;
    EXPECT_EQ(admit_ngg_draw(r, kena_facts(), radv()).shape.topology,
              NggInputTopology::TriangleList);
    r.spi_shader_pgm_rsrc2_gs = ngg::kKenaRsrc2Gs | (8u << 1);
    EXPECT_STREQ(refusal(r, kena_facts()), "admitted") << "RSRC2 USER_SGPR equal to the range";
    EXPECT_EQ(admit_ngg_draw(r, kena_facts(), radv()).user_sgprs, 8u);
    // #3135: RSRC2's count is what the SPI loads. Kena's 4324d9f3 has USER_SGPR 12 and a 0..24
    // range; the other 12 words are reached through s0:s1.
    auto wide = kena_facts();
    wide.user_data_range_end = 24;
    r.spi_shader_pgm_rsrc2_gs = ngg::kKenaRsrc2Gs | (12u << 1);
    EXPECT_STREQ(refusal(r, wide), "admitted");
    EXPECT_EQ(admit_ngg_draw(r, wide, radv()).user_sgprs, 12u) << "the hardware's count";
    r.spi_shader_pgm_rsrc2_gs = ngg::kKenaRsrc2Gs;
    EXPECT_EQ(admit_ngg_draw(r, wide, radv()).user_sgprs, 24u) << "zero: the AGC range";
    // The whole 32-word push budget is usable: s0:s1's two words are reserved only for a program
    // that reads them, and the live producer checks that (#4735 review).
    r.spi_shader_pgm_rsrc2_gs = ngg::kKenaRsrc2Gs | (1u << 27);   // USER_SGPR_MSB: 32
    EXPECT_STREQ(refusal(r, kena_facts()), "admitted");
    r.spi_shader_pgm_rsrc2_gs = ngg::kKenaRsrc2Gs | (1u << 27) | (1u << 1);   // 33
    EXPECT_STREQ(refusal(r, kena_facts()), "ngg-user-sgpr-count") << "past the push budget";
}

// Without PRIMGEN_EN there is no NGG draw. (PRIMGEN_EN without GS_EN is the VS-only draw, P7.)
TEST(NggDrawAdmission, NotMergedDoesNotApply) {
    for (uint32_t stages : {0x00000030u /* no PRIMGEN_EN */, 0x00000010u, 0u}) {
        auto r = kena_registers();
        r.vgt_shader_stages_en = stages;
        const NggDrawAdmission a = admit_ngg_draw(r, kena_facts(), radv());
        EXPECT_FALSE(a.applies) << std::hex << stages;
        EXPECT_EQ(a.refusal, nullptr);
    }
}

struct Arm {
    const char* reason;
    std::function<void(NggDrawRegisters&, NggDrawFacts&, NggHostCapabilities&)> mutate;
};

TEST(NggDrawAdmission, EveryRefusalIsNamed) {
    const std::vector<Arm> arms = {
        {"ngg-host-unpublished", [](auto&, auto&, auto& h) { h.published = false; }},
        {"ngg-register-missing", [](auto& r, auto&, auto&) { r.missing = "GE_CNTL"; }},
        {"ngg-passthrough-gs", [](auto& r, auto&, auto&) { r.vgt_shader_stages_en |= 1u << 25; }},
        {"ngg-gs-instancing",
         [](auto& r, auto&, auto&) { r.vgt_gs_instance_cnt = kGsInstanceEnable | (2u << 2); }},
        {"ngg-input-topology", [](auto& r, auto&, auto&) { r.primitive_type = 5u; }},   // fan
        {"ngg-input-topology",
         [](auto& r, auto&, auto&) { r.primitive_type = 0xcu; }},   // list adj
        {"ngg-input-topology", [](auto& r, auto&, auto&) { r.primitive_type = 0x11u; }},   // rect
        {"ngg-input-topology", [](auto& r, auto&, auto&) { r.primitive_type = 0x13u; }},   // quad
        {"ngg-input-topology", [](auto& r, auto&, auto&) { r.primitive_type = 2u; }},   // lines
        {"ngg-output-topology", [](auto& r, auto&, auto&) { r.vgt_gs_out_prim_type = 0u; }},
        {"ngg-output-topology", [](auto& r, auto&, auto&) { r.vgt_gs_out_prim_type = 3u; }},
        {"ngg-index-unavailable", [](auto&, auto& f, auto&) { f.indexed = true; }},
        {"ngg-index-restart",
         [](auto&, auto& f, auto&) {
             f.indexed = true;
             f.index_refusal = "ngg-index-restart";
         }},
        {"ngg-indirect", [](auto&, auto& f, auto&) { f.indirect = true; }},
        {"ngg-vertex-offset", [](auto&, auto& f, auto&) { f.vertex_offset = true; }},
        {"ngg-viewport-index",
         [](auto& r, auto&, auto&) { r.pa_cl_vs_out_cntl |= kVsOutUseVtxViewportIndx; }},
        {"ngg-point-size",
         [](auto& r, auto&, auto&) { r.pa_cl_vs_out_cntl |= kVsOutUseVtxPointSize; }},
        {"ngg-clip-cull-distance", [](auto& r, auto&, auto&) { r.pa_cl_vs_out_cntl |= 1u; }},
        {"ngg-clip-cull-distance", [](auto& r, auto&, auto&) { r.pa_cl_vs_out_cntl |= 1u << 8; }},
        {"ngg-clip-cull-distance",
         [](auto& r, auto&, auto&) { r.pa_cl_vs_out_cntl |= kVsOutCcDist0VecEna; }},
        {"ngg-user-clip-plane", [](auto& r, auto&, auto&) { r.pa_cl_clip_cntl |= 1u; }},
        {"ngg-vertex-kill-flag",
         [](auto& r, auto&, auto&) { r.pa_cl_vs_out_cntl |= kVsOutUseVtxKillFlag; }},
        {"ngg-vs-out-undecoded", [](auto& r, auto&, auto&) { r.pa_cl_vs_out_cntl |= 1u << 25; }},
        {"ngg-vs-out-undecoded", [](auto& r, auto&, auto&) { r.pa_cl_vs_out_cntl |= 1u << 31; }},
        {"ngg-layer-target-not-layered", [](auto&, auto& f, auto&) { f.target_slices = 0; }},
        {"ngg-layer-slice-start", [](auto&, auto& f, auto&) { f.target_first_slice = 1; }},
        {"ngg-strip-order-visible", [](auto& r, auto&, auto&) { r.pa_su_sc_mode_cntl |= 2u; }},
        {"ngg-strip-order-visible", [](auto& r, auto&, auto&) { r.pa_su_sc_mode_cntl |= 1u; }},
        {"ngg-strip-order-visible",
         [](auto& r, auto&, auto&) { r.spi_ps_input_ena |= kPsInputFrontFace; }},
        {"ngg-strip-order-visible", [](auto&, auto& f, auto&) { f.flat_input_mask = 1; }},
        {"ngg-strip-order-visible", [](auto&, auto& f, auto&) { f.raw_vertex_input_mask = 2; }},
        {"ngg-user-data-range", [](auto&, auto& f, auto&) { f.user_data_range_known = false; }},
        {"ngg-user-data-range", [](auto&, auto& f, auto&) { f.user_data_range_start = 1; }},
        {"ngg-user-data-range", [](auto&, auto& f, auto&) { f.user_data_range_end = 33; }},
        {"ngg-user-sgpr-count",
         [](auto& r, auto&, auto&) { r.spi_shader_pgm_rsrc2_gs |= (1u << 27) | (1u << 1); }},
        {"ngg-lds-limit", [](auto&, auto&, auto& h) { h.max_compute_shared_memory = 4096; }},
        {"ngg-host-compute", [](auto&, auto&, auto& h) { h.compute = false; }},
        {"ngg-layer-route-unavailable",
         [](auto&, auto&, auto& h) { h.shader_output_layer = h.geometry_shader = false; }},
        {"ngg-layer-route-unavailable",
         [](auto& r, auto& f, auto& h) {
             r.pa_cl_vs_out_cntl &= ~kVsOutUseVtxRenderTargetIndx;
             r.primitive_type = 4u;
             f.interpolation_geometry_required = true;
             h.geometry_shader = false;
         }},
        {"ngg-interpolation-geometry-needs-triangles",
         [](auto& r, auto& f, auto&) {
             r.vgt_gs_out_prim_type = 1u;   // lines
             r.primitive_type = 4u;
             f.interpolation_geometry_required = true;
         }},
    };
    for (const Arm& arm : arms) {
        auto r = kena_registers();
        auto f = kena_facts();
        auto h = radv();
        arm.mutate(r, f, h);
        EXPECT_STREQ(refusal(r, f, h), arm.reason);
    }
}

TEST(NggDrawAdmission, AdmittedTwins) {
    // The states beside each refusal that must stay admitted.
    auto r = kena_registers();
    r.vgt_gs_instance_cnt = kGsInstanceEnable | (1u << 2);   // instancing enabled, one instance
    EXPECT_STREQ(refusal(r, kena_facts()), "admitted");
    r = kena_registers();
    r.primitive_type = 4u;   // a LIST may cull, read FRONT_FACE and flat inputs
    r.pa_su_sc_mode_cntl |= 3u;
    r.spi_ps_input_ena |= kPsInputFrontFace;
    auto f = kena_facts();
    f.flat_input_mask = f.raw_vertex_input_mask = 1;
    EXPECT_STREQ(refusal(r, f), "admitted");
    r = kena_registers();
    r.pa_cl_vs_out_cntl &= ~kVsOutUseVtxRenderTargetIndx;   // no layer: a 2D target is fine
    f = kena_facts();
    f.target_slices = 0;
    f.target_first_slice = 3;
    EXPECT_STREQ(refusal(r, f), "admitted");
    r = kena_registers();
    r.pa_cl_clip_cntl = 0x01080000u | (1u << 16);   // CLIP_DISABLE is not a UCP
    EXPECT_STREQ(refusal(r, kena_facts()), "admitted");
    auto h = radv();
    h.shader_output_layer = false;   // the forwarding GS takes over
    const NggDrawAdmission a = admit_ngg_draw(kena_registers(), kena_facts(), h);
    EXPECT_TRUE(a.ok());
    EXPECT_EQ(a.route, NggLayerRoute::ForwardingGeometry);
    h = radv();
    h.vertex_pipeline_stores = false;
    EXPECT_FALSE(admit_ngg_draw(kena_registers(), kena_facts(), h).count_violations)
        << "no vertex stores: admitted without counting";
}

// #3135 P6: an indexed draw is admitted with its indices, and the INDEX count -- not the packet's
// vertex count the caller passed, which a stale or non-indexed reading would leave -- sizes the
// shape. Kena's past-New-Game shape: an 18-index list, 41 instances, into a 64-slice volume.
TEST(NggDrawAdmission, IndexedListIsAdmittedWithItsIndices) {
    auto r = kena_registers();
    r.primitive_type = 4u;   // triangle list
    auto f = kena_facts(41);
    f.target_slices = 64;
    f.indexed = true;
    std::vector<uint32_t> values;
    for (uint32_t quad = 0; quad < 3u; ++quad)
        values.insert(values.end(), {4 * quad, 4 * quad + 1, 4 * quad + 2, 4 * quad + 2,
                                     4 * quad + 1, 4 * quad + 3});
    f.indices = std::make_shared<const std::vector<uint32_t>>(values);
    f.vertex_count = 4;   // what a non-indexed reading of the packet would say
    const NggDrawAdmission a = admit_ngg_draw(r, f, radv());
    ASSERT_TRUE(a.ok()) << a.refusal;
    EXPECT_EQ(a.shape.indices, f.indices);
    EXPECT_EQ(a.shape.vertex_count, 18u);
    EXPECT_EQ(a.shape.instance_count, 41u);
    EXPECT_EQ(a.shape.topology, NggInputTopology::TriangleList);
    EXPECT_EQ(a.layer_slices, 64u);
    f.indexed = false;   // the same facts read as non-indexed carry no indices into the shape
    EXPECT_EQ(admit_ngg_draw(r, f, radv()).shape.indices, nullptr);
}

TEST(NggDrawAdmission, RegistersAndUserDataAreReadFromTheDrawState) {
    namespace P = prosper::agc::Pm4;
    GpuState state;
    state.cx[P::VGT_SHADER_STAGES_EN] = 0x2030u;
    state.cx[P::VGT_GS_ONCHIP_CNTL] = 0x10020040u;
    state.uc[P::GE_CNTL] = 0x8040u;
    state.cx[P::GE_MAX_OUTPUT_PER_SUBGROUP] = 0xc0u;
    state.cx[P::VGT_GS_MAX_VERT_OUT] = 3u;
    state.cx[P::VGT_ESGS_RING_ITEMSIZE] = 4u;
    state.sh[P::SPI_SHADER_PGM_RSRC2_GS] = ngg::kKenaRsrc2Gs;
    state.cx[P::VGT_GS_OUT_PRIM_TYPE] = 2u;
    state.cx[P::PA_SU_SC_MODE_CNTL] = 0x240u;
    state.cx[P::PA_CL_VS_OUT_CNTL] = 0x01240000u;
    state.cx[P::PA_CL_CLIP_CNTL] = 0x01080000u;
    state.cx[P::SPI_PS_INPUT_ENA] = 0x2020u;
    NggDrawRegisters r = read_ngg_draw_registers(state, 6u);
    EXPECT_EQ(r.missing, nullptr);
    const NggDrawRegisters k = kena_registers();
    EXPECT_EQ(std::memcmp(&r, &k, offsetof(NggDrawRegisters, missing)), 0);
    state.uc.erase(P::GE_CNTL);
    r = read_ngg_draw_registers(state, 6u);
    EXPECT_STREQ(r.missing, "GE_CNTL");
    state.uc[P::GE_CNTL] = 0x8040u;
    state.cx.erase(P::VGT_GS_INSTANCE_CNT);
    EXPECT_EQ(read_ngg_draw_registers(state, 6u).missing, nullptr) << "reset value 0";

    // #3135: the GS user-data address s0:s1, known only when both registers are present and the
    // address is not zero.
    uint32_t address[2] = {};
    EXPECT_FALSE(read_ngg_user_data_address(state, address));
    state.sh[P::SPI_SHADER_USER_DATA_ADDR_LO_GS] = 0x12340000u;
    EXPECT_FALSE(read_ngg_user_data_address(state, address)) << "HI absent";
    state.sh[P::SPI_SHADER_USER_DATA_ADDR_HI_GS] = 0x5u;
    ASSERT_TRUE(read_ngg_user_data_address(state, address));
    EXPECT_EQ(address[0], 0x12340000u);
    EXPECT_EQ(address[1], 0x5u);
    state.sh[P::SPI_SHADER_USER_DATA_ADDR_LO_GS] = 0;
    state.sh[P::SPI_SHADER_USER_DATA_ADDR_HI_GS] = 0;
    EXPECT_FALSE(read_ngg_user_data_address(state, address)) << "a zero address is no address";

    std::vector<uint32_t> words;
    for (uint32_t k2 = 0; k2 < 8; ++k2) state.sh[P::SPI_SHADER_USER_DATA_GS_0 + k2] = 0x100 + k2;
    ASSERT_TRUE(read_ngg_user_data(state, 8, &words));
    EXPECT_EQ(words,
              (std::vector<uint32_t>{0x100, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107}));
    state.sh.erase(P::SPI_SHADER_USER_DATA_GS_0 + 5);
    EXPECT_FALSE(read_ngg_user_data(state, 8, &words));
    EXPECT_TRUE(read_ngg_user_data(state, 5, &words));
}

// ---- The live producer ------------------------------------------------------------------------------

struct KenaProgram {
    std::vector<uint32_t> prolog, main;
    size_t prefix = 0;
    ShaderResourceTable table;
};

const KenaProgram& kena_program() {
    static const KenaProgram program = [] {
        KenaProgram p;
        const auto data = tests_root(__FILE__) / "data";
        p.prolog = ngg::load_words(data / "ngg_merged_es_prolog.bin");
        p.main = ngg::load_words(data / "ngg_merged_gs_main.bin");
        p.prefix = rdna2_vertex_prolog_info(p.prolog.data(), p.prolog.size()).prefix_dwords;
        p.table = ngg::kena_resources(4);
        return p;
    }();
    return program;
}

NggLiveDrawInput kena_input(uint32_t instances = 32, uint32_t push_seed = 0) {
    const KenaProgram& p = kena_program();
    NggLiveDrawInput in;
    in.registers = kena_registers();
    in.facts = kena_facts(instances);
    in.user_data.assign(ngg::kKenaUserSgprs, push_seed);
    in.user_data_complete = true;
    in.linked = ngg_linked_chain(p.prolog.data(), p.prefix, p.main.data(), p.main.size());
    in.resources = &p.table;
    in.program_address = 0x5009440000ull;
    return in;
}

class NggLiveDraw : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_FALSE(kena_program().prolog.empty()) << "tests/data fixture missing";
        reset_ngg_live_draw_cache_for_test();
    }
};

TEST_F(NggLiveDraw, KenaMatchesTheDirectBuilder) {
    const NggLiveDrawResult live = realize_ngg_live_draw(kena_input(), radv());
    ASSERT_TRUE(live.applies);
    ASSERT_TRUE(live.draw) << (live.refusal ? live.refusal : "") << " " << live.detail;

    NggSubgroupDrawRequest request;
    const auto linked = ngg::kena_linked(tests_root(__FILE__) / "data");
    request.linked_code = linked.data();
    request.dwords = linked.size();
    request.resources = &kena_program().table;
    request.shell.rsrc2_gs_lds_size = ngg_rsrc2_gs_lds_size(ngg::kKenaRsrc2Gs);
    request.shell.user_sgprs = ngg::kKenaUserSgprs;
    request.shell.native_wave64 = true;
    request.limits = ngg::kena_limits();
    request.shape = {NggInputTopology::TriangleStrip, 4, 32, 0};
    request.raster.topology = NggOutputTopology::TriangleList;
    request.raster.layer_from_pos1 = true;
    request.raster.layer_slices = 32;
    request.raster.route = NggLayerRoute::ShaderOutputLayer;
    request.raster.count_violations = true;
    request.push_constants.assign(ngg::kKenaUserSgprs, 0u);
    std::string why;
    const auto direct = build_ngg_subgroup_draw(request, &why);
    ASSERT_TRUE(direct) << why;

    const NggSubgroupDraw& a = *live.draw;
    const NggSubgroupDraw& b = *direct;
    ASSERT_EQ(a.groups.size(), b.groups.size());
    for (size_t g = 0; g < a.groups.size(); ++g) {
        EXPECT_EQ(a.groups[g].waves, b.groups[g].waves);
        EXPECT_EQ(a.groups[g].blocks, b.groups[g].blocks);
        EXPECT_EQ(a.groups[g].launch_words, b.groups[g].launch_words);
        EXPECT_EQ(a.groups[g].export_words, b.groups[g].export_words);
        EXPECT_EQ(*a.groups[g].stages->shell, *b.groups[g].stages->shell);
        EXPECT_EQ(*a.groups[g].stages->raster_vertex, *b.groups[g].stages->raster_vertex);
        EXPECT_EQ(a.groups[g].stages->shell_hash, ngg_words_hash(*a.groups[g].stages->shell));
    }
    ASSERT_EQ(a.runs.size(), b.runs.size());
    EXPECT_EQ(a.guest_bindings, b.guest_bindings);
    EXPECT_EQ(a.push_constants, b.push_constants);
    EXPECT_EQ(a.route, NggLayerRoute::ShaderOutputLayer);
    EXPECT_TRUE(a.native_wave64);
    EXPECT_EQ(a.lds_bytes, 17u * 512u);
    EXPECT_EQ(ngg_device_refusal(a, radv()), nullptr);
}

TEST_F(NggLiveDraw, LinkedChainIsTheHardwareOrderAndStable) {
    const KenaProgram& p = kena_program();
    const auto linked = ngg_linked_chain(p.prolog.data(), p.prefix, p.main.data(), p.main.size());
    ASSERT_TRUE(linked);
    EXPECT_EQ(*linked, ngg::kena_linked(tests_root(__FILE__) / "data"));
    EXPECT_EQ(ngg_linked_chain(p.prolog.data(), p.prefix, p.main.data(), p.main.size()), linked)
        << "a repeated chain keeps one stable copy (the fold's decode cache keys by address)";
    EXPECT_FALSE(ngg_linked_chain(p.prolog.data(), p.prefix, nullptr, 0));
    auto in = kena_input();
    in.linked.reset();
    EXPECT_STREQ(realize_ngg_live_draw(in, radv()).refusal, "ngg-program-unavailable");
}

TEST_F(NggLiveDraw, NothingCompilesOnceWarm) {
    const auto first = realize_ngg_live_draw(kena_input(), radv());
    ASSERT_TRUE(first.draw);
    const auto warm = ngg_live_draw_cache_stats();
    EXPECT_EQ(warm.stage_compiles, 1u) << "one wave count";
    EXPECT_EQ(warm.draw_assemblies, 1u);

    const auto again = realize_ngg_live_draw(kena_input(), radv());
    EXPECT_EQ(again.draw, first.draw) << "a repeated draw reuses its description";
    EXPECT_EQ(ngg_live_draw_cache_stats().draw_hits, warm.draw_hits + 1);

    // A different shape and different push-constant words: a new description, no new compile.
    const auto other = realize_ngg_live_draw(kena_input(7, 0x55), radv());
    ASSERT_TRUE(other.draw);
    EXPECT_NE(other.draw, first.draw);
    EXPECT_EQ(other.draw->groups[0].blocks, 7u);
    EXPECT_EQ(other.draw->push_constants, std::vector<uint32_t>(8, 0x55u));
    EXPECT_EQ(other.draw->groups[0].stages, first.draw->groups[0].stages)
        << "the compiled stages are shared";
    const auto after = ngg_live_draw_cache_stats();
    EXPECT_EQ(after.stage_compiles, warm.stage_compiles);
    EXPECT_GE(after.stage_hits, warm.stage_hits + 1);

    // A compile input that changes the module misses: here the slice count.
    auto in = kena_input();
    in.facts.target_slices = 16;
    ASSERT_TRUE(realize_ngg_live_draw(in, radv()).draw);
    EXPECT_EQ(ngg_live_draw_cache_stats().stage_compiles, warm.stage_compiles + 1);
    // So does a resource-table shape change (a binding moves).
    KenaProgram moved = kena_program();
    moved.table.resources[0].binding = 12;
    in = kena_input();
    in.resources = &moved.table;
    const auto before_move = ngg_live_draw_cache_stats().stage_compiles;
    (void)realize_ngg_live_draw(in, radv());
    EXPECT_EQ(ngg_live_draw_cache_stats().stage_compiles, before_move + 1);
}

// The stage key carries the emitter's data-dependent admission of each resource, built by the
// ordinary shader cache's own per-resource function. Here the V# raw register snapshot at pc 33
// loses its backing (8 of its 16 bytes): the ordinary key marks it invalid, so a warm module must
// NOT be reused for it (the module would read words the draw does not have), and its refusal must
// not stick to the valid table either.
TEST_F(NggLiveDraw, AWarmEntryIsNotReusedAcrossAResourceAdmissionChange) {
    const auto warm = realize_ngg_live_draw(kena_input(), radv());
    ASSERT_TRUE(warm.draw);
    const uint64_t compiles = ngg_live_draw_cache_stats().stage_compiles;
    EXPECT_EQ(ngg_live_draw_cache_stats().strip_draws, 1u) << "Kena's producer is a strip";

    KenaProgram unbacked = kena_program();
    bool changed = false;
    for (auto& r : unbacked.table.resources)
        if (r.raw_register_snapshot && r.fetch_pc == 33) {
            r.host_data_size = 8;
            changed = true;
        }
    ASSERT_TRUE(changed);
    auto in = kena_input(32, 7);
    in.resources = &unbacked.table;
    const auto partial = realize_ngg_live_draw(in, radv());
    EXPECT_EQ(ngg_live_draw_cache_stats().stage_compiles, compiles + 1)
        << "the changed admission is a different stage key";
    EXPECT_FALSE(partial.draw) << "the emitter refuses an unbacked register-offset load";

    const auto again = realize_ngg_live_draw(kena_input(32, 7), radv());
    ASSERT_TRUE(again.draw) << "the refusal did not stick to the backed table";
    EXPECT_EQ(again.draw->groups[0].stages, warm.draw->groups[0].stages);
    EXPECT_EQ(ngg_live_draw_cache_stats().stage_compiles, compiles + 1);
}

// #3135 P6: the draw cache keys an indexed draw on its index VALUES. Two indexed draws with the
// same count, instances and push words but different indices must get different descriptions (the
// launch records carry different VertexIDs), the same indices must hit, and the compiled stages are
// shared by all of them.
TEST_F(NggLiveDraw, IndexedDrawsAreCachedByTheirIndexValues) {
    const auto with = [](std::vector<uint32_t> values) {
        auto in = kena_input(3);
        in.registers.primitive_type = 4u;   // triangle list
        in.facts.indexed = true;
        in.facts.indices = std::make_shared<const std::vector<uint32_t>>(std::move(values));
        return in;
    };
    const auto first = realize_ngg_live_draw(with({0, 1, 2, 2, 1, 3}), radv());
    ASSERT_TRUE(first.draw) << (first.refusal ? first.refusal : "") << " " << first.detail;
    EXPECT_TRUE(first.indexed);
    const auto launch_v5 = [](const NggSubgroupDraw& draw, uint32_t lane) {
        return draw.groups[0].launch_words[lane * kNggLaunchWordsPerLane + 5u];
    };
    EXPECT_EQ(launch_v5(*first.draw, 3), 3u);

    const auto same = realize_ngg_live_draw(with({0, 1, 2, 2, 1, 3}), radv());
    EXPECT_EQ(same.draw, first.draw) << "the same index values reuse the description";

    const auto other = realize_ngg_live_draw(with({0, 1, 2, 2, 1, 7}), radv());
    ASSERT_TRUE(other.draw);
    EXPECT_NE(other.draw, first.draw) << "different index values are a different plan";
    EXPECT_EQ(launch_v5(*other.draw, 3), 7u) << "lane 3 runs index 7";
    EXPECT_EQ(other.draw->groups[0].stages, first.draw->groups[0].stages)
        << "indices never reach the compiled stages";

    // The same draw read as non-indexed (vertices 0..5) is not the indexed one.
    auto plain = kena_input(3);
    plain.registers.primitive_type = 4u;
    plain.facts.vertex_count = 6;
    const auto unindexed = realize_ngg_live_draw(plain, radv());
    ASSERT_TRUE(unindexed.draw);
    EXPECT_NE(unindexed.draw, first.draw);
    EXPECT_FALSE(unindexed.indexed);
    EXPECT_EQ(ngg_live_draw_cache_stats().indexed_draws, 3u) << "first, same and other";
}

// #4735 review: a program that never reads s0:s1 is not charged its two push words. Kena's LUT
// chain with 31 user SGPRs and a known user-data address is admitted, and the address is not
// pushed: the push constants are exactly the 31 user words.
TEST_F(NggLiveDraw, TheUserDataAddressIsSuppliedOnlyToAProgramThatReadsIt) {
    const KenaProgram& p = kena_program();
    EXPECT_FALSE(ngg_program_reads_user_data_address(
        ngg_linked_chain(p.prolog.data(), p.prefix, p.main.data(), p.main.size()), 31))
        << "the LUT chain never reads s0:s1";
    auto in = kena_input();
    in.facts.user_data_range_end = 31;
    in.user_data.assign(31, 0u);
    in.user_data_address_known = true;
    in.user_data_address[0] = 0x12340000u;
    in.user_data_address[1] = 0x5u;
    const auto result = realize_ngg_live_draw(in, radv());
    ASSERT_TRUE(result.draw) << (result.refusal ? result.refusal : "") << " " << result.detail;
    EXPECT_EQ(result.draw->push_constants.size(), 31u) << "no s0:s1 words for a non-reader";

    // A program that DOES read s0:s1 (it reloads its user SGPRs from the address, as Kena's
    // 11562c72 does) needs those two words: with 31 user SGPRs there is no room, refused by name
    // before anything compiles; with 30 the same program is not refused by that rule.
    auto reader = std::make_shared<const std::vector<uint32_t>>(std::vector<uint32_t>{
        0xbefe04c1u,   // s_mov_b64 exec, -1
        0xf4100200u, 0xfa000000u,   // s_load_dwordx8 s[8:15], s[0:1], 0
        0xbf8cc07fu,   // s_waitcnt lgkmcnt(0)
        0xb07c3005u, 0xbf900009u,   // s_movk_i32 m0, 0x3005; s_sendmsg GS_ALLOC_REQ
        0xf8000941u, 0x00000009u,   // exp prim v9
        0xf80000cfu, 0x03020100u,   // exp pos0 v0..v3
        0xbf810000u});
    EXPECT_TRUE(ngg_program_reads_user_data_address(reader, 31));
    in.linked = reader;
    EXPECT_STREQ(realize_ngg_live_draw(in, radv()).refusal, "ngg-user-sgpr-count")
        << "31 user SGPRs + s0:s1 exceed the 32-word push budget";
    in.facts.user_data_range_end = 30;
    in.user_data.assign(30, 0u);
    const auto roomy = realize_ngg_live_draw(in, radv());
    EXPECT_FALSE(roomy.refusal && std::string(roomy.refusal) == "ngg-user-sgpr-count")
        << "30 + 2 fits";
}

TEST_F(NggLiveDraw, RefusalsAreNamedAndARefusedCompileIsCached) {
    auto in = kena_input();
    in.user_data_complete = false;
    auto result = realize_ngg_live_draw(in, radv());
    EXPECT_FALSE(result.draw);
    EXPECT_STREQ(result.refusal, "ngg-user-data-unavailable");

    in = kena_input();
    in.facts.indexed = true;
    EXPECT_STREQ(realize_ngg_live_draw(in, radv()).refusal, "ngg-index-unavailable");

    // Four user SGPRs where the chain reads eight: the shell refuses, by the ABI rule's name.
    in = kena_input();
    in.facts.user_data_range_end = 4;
    in.user_data.resize(4);
    result = realize_ngg_live_draw(in, radv());
    ASSERT_FALSE(result.draw);
    EXPECT_STREQ(result.refusal, "ngg-abi-read-undefined-sgpr") << result.detail;
    const auto compiles = ngg_live_draw_cache_stats().stage_compiles;
    result = realize_ngg_live_draw(in, radv());
    EXPECT_STREQ(result.refusal, "ngg-abi-read-undefined-sgpr");
    EXPECT_EQ(ngg_live_draw_cache_stats().stage_compiles, compiles)
        << "the refusal is cached: no per-draw recompile";

    // The device half refuses a description the device cannot run.
    auto host = radv();
    host.max_compute_workgroup_count_x = 31;   // 32 subgroups, one per workgroup
    result = realize_ngg_live_draw(kena_input(), host);
    EXPECT_FALSE(result.draw);
    EXPECT_STREQ(result.refusal, "ngg-backend-workgroup-limit");
}

// ---- The device half --------------------------------------------------------------------------------

TEST_F(NggLiveDraw, DeviceRefusals) {
    const auto live = realize_ngg_live_draw(kena_input(), radv());
    ASSERT_TRUE(live.draw);
    const NggSubgroupDraw& draw = *live.draw;
    EXPECT_EQ(ngg_device_refusal(draw, radv()), nullptr);
    const auto expect = [&](const char* reason, auto mutate_host, auto mutate_draw) {
        NggHostCapabilities h = radv();
        NggSubgroupDraw d = draw;
        mutate_host(h);
        mutate_draw(d);
        EXPECT_STREQ(ngg_device_refusal(d, h) ? ngg_device_refusal(d, h) : "admitted", reason);
    };
    const auto none = [](auto&) {};
    expect("ngg-backend-no-compute", [](auto& h) { h.compute = false; }, none);
    expect("ngg-backend-description", [](auto& h) { h.max_push_constants_size = 16; }, none);
    expect("ngg-backend-description", none, [](auto& d) { d.runs.clear(); });
    expect(
        "ngg-backend-vertex-stores-unavailable", [](auto& h) { h.vertex_pipeline_stores = false; },
        none);
    expect(
        "ngg-backend-layer-route-unavailable", [](auto& h) { h.shader_output_layer = false; },
        none);
    expect(
        "ngg-backend-layer-route-unavailable", [](auto& h) { h.geometry_shader = false; },
        [](auto& d) { d.route = NggLayerRoute::ForwardingGeometry; });
    expect("ngg-backend-wave64-unavailable", [](auto& h) { h.native_wave64 = false; }, none);
    expect("ngg-backend-lds-limit", [](auto& h) { h.max_compute_shared_memory = 8192; }, none);
    expect(
        "ngg-backend-workgroup-limit", [](auto& h) { h.max_compute_workgroup_size_x = 32; }, none);
    expect(
        "ngg-backend-workgroup-limit", [](auto& h) { h.max_compute_workgroup_subgroups = 0; },
        none);
    expect("ngg-backend-buffer-range", [](auto& h) { h.max_storage_buffer_range = 4096; }, none);
}

// ---- #3135 P7: NGG without a GS ---------------------------------------------------------------------
//
// Kena's culling VS programs, as their draws arrive past the first level load: VGT_SHADER_STAGES_EN
// 0x2000 (PRIMGEN_EN, no GS), VGT_GS_OUT_PRIM_TYPE 0 and GS_MAX_VERT_OUT 0, GE_MAX_OUTPUT_PER_SUBGROUP
// 64, an indexed triangle list into a 2D target, 25 user SGPRs.
NggDrawRegisters kena_vs_only_registers() {
    NggDrawRegisters r = kena_registers();
    r.vgt_shader_stages_en = 0x00002000u;
    r.vgt_gs_out_prim_type = 0u;
    r.vgt_gs_max_vert_out = 0u;
    r.ge_max_output_per_subgroup = 0x40u;
    r.spi_shader_pgm_rsrc2_gs = 25u << 1;
    r.pa_su_sc_mode_cntl = 0x240u;
    r.primitive_type = 4u;   // triangle list
    return r;
}

NggDrawFacts kena_vs_only_facts() {
    NggDrawFacts f;
    f.vertex_count = 6;
    f.instance_count = 1;
    f.target_single_slice = true;   // a 2D scene target, CB view slices 0..0
    f.other_attachments_single_slice = true;   // and every other attachment one slice too
    f.user_data_range_known = true;
    f.user_data_range_end = 25;
    return f;
}

// The draw is admitted as a VS-only NGG draw whose output is triangles, although the GS output
// primitive type register says points: there is no GS to apply it. The same registers with GS_EN
// set are a merged draw, where the register does apply, and stay refused by its name.
TEST(NggDrawAdmission, AVsOnlyDrawExportsItsInputTriangles) {
    const NggDrawAdmission vs =
        admit_ngg_draw(kena_vs_only_registers(), kena_vs_only_facts(), radv());
    ASSERT_TRUE(vs.applies);
    ASSERT_TRUE(vs.ok()) << vs.refusal;
    EXPECT_TRUE(vs.vs_only);
    EXPECT_TRUE(vs.limits.vs_only) << "the planner must partition it as VS-only";
    EXPECT_EQ(vs.topology, NggOutputTopology::TriangleList);
    EXPECT_EQ(vs.user_sgprs, 25u);

    NggDrawRegisters merged = kena_vs_only_registers();
    merged.vgt_shader_stages_en = 0x00002030u;
    const NggDrawAdmission gs = admit_ngg_draw(merged, kena_vs_only_facts(), radv());
    EXPECT_FALSE(gs.vs_only);
    ASSERT_TRUE(gs.refusal);
    EXPECT_STREQ(gs.refusal, "ngg-output-topology") << "a GS's point output is still refused";

    // The layer (USE_VTX_RENDER_TARGET_INDX) addresses the one-slice target's only slice.
    EXPECT_TRUE(vs.layer_from_pos1);
    EXPECT_EQ(vs.layer_slices, 1u);
    EXPECT_EQ(vs.route, NggLayerRoute::None) << "one slice needs no layer route";
    NggDrawFacts array = kena_vs_only_facts();
    array.target_single_slice = false;   // a 2D array, or an unprogrammed view
    const NggDrawAdmission unknown = admit_ngg_draw(kena_vs_only_registers(), array, radv());
    ASSERT_TRUE(unknown.refusal);
    EXPECT_STREQ(unknown.refusal, "ngg-layer-target-not-layered")
        << "a target not proven to be one slice keeps the refusal";
    NggDrawFacts layered_depth = kena_vs_only_facts();
    layered_depth.other_attachments_single_slice = false;   // e.g. a cascade depth array
    const NggDrawAdmission depth = admit_ngg_draw(kena_vs_only_registers(), layered_depth, radv());
    ASSERT_TRUE(depth.refusal);
    EXPECT_STREQ(depth.refusal, "ngg-layer-target-not-single-slice")
        << "a layered attachment beside a one-slice colour target is refused, not culled";

    NggDrawRegisters legacy = kena_vs_only_registers();
    legacy.vgt_shader_stages_en = 0u;
    EXPECT_FALSE(admit_ngg_draw(legacy, kena_vs_only_facts(), radv()).applies)
        << "without PRIMGEN_EN the path does not apply";
}

// #4808: Yakuza Kiwami's NGG VS draws, VGT_SHADER_STAGES_EN 0x02402000 -- PRIMGEN_EN, GS_W32_EN and
// PRIMGEN_PASSTHRU_EN, no GS. Admitted as a Wave32, passthrough VS-only draw: the planner gets the
// width and the launch form, and the shell's native subgroup follows the host's 32-lane capability,
// not its 64-lane one. Passthrough with a GS stays refused by name.
TEST(NggDrawAdmission, AWave32PassthroughVsOnlyDrawIsAdmittedWithItsWidth) {
    NggDrawRegisters r = kena_vs_only_registers();
    r.vgt_shader_stages_en = 0x02402000u;
    NggHostCapabilities host = radv();
    host.native_wave64 = true;
    host.native_wave32 = false;
    const NggDrawAdmission w32 = admit_ngg_draw(r, kena_vs_only_facts(), host);
    ASSERT_TRUE(w32.ok()) << w32.refusal;
    EXPECT_TRUE(w32.vs_only);
    EXPECT_EQ(w32.limits.wave_lanes, 32u);
    EXPECT_TRUE(w32.limits.passthrough);
    EXPECT_FALSE(w32.native_wave64) << "a 64-lane native subgroup does not run a Wave32 wave";
    host.native_wave32 = true;
    EXPECT_TRUE(admit_ngg_draw(r, kena_vs_only_facts(), host).native_wave64);

    const NggDrawAdmission w64 =
        admit_ngg_draw(kena_vs_only_registers(), kena_vs_only_facts(), host);
    ASSERT_TRUE(w64.ok()) << w64.refusal;
    EXPECT_EQ(w64.limits.wave_lanes, 64u);
    EXPECT_FALSE(w64.limits.passthrough);

    NggDrawRegisters merged = r;
    merged.vgt_shader_stages_en |= 0x30u;   // GS_EN, ES_EN
    merged.vgt_gs_out_prim_type = 2u;
    const NggDrawAdmission gs = admit_ngg_draw(merged, kena_vs_only_facts(), host);
    ASSERT_TRUE(gs.refusal);
    EXPECT_STREQ(gs.refusal, "ngg-passthrough-gs");
}

// #3135 layered NGG depth. A draw that writes no colour has its layer address the depth array:
// admitted as one replay per slice of the view, routed by no layer stage. Every shape the replay
// cannot represent is refused by name: no depth attachment, an unprogrammed view, and the two mixed
// shapes -- a one-slice colour target beside a depth array, and a layered colour volume beside any
// bound depth/stencil.
TEST(NggDrawAdmission, ADepthOnlyLayeredDrawIsReplayedPerSlice) {
    NggDrawFacts f = kena_vs_only_facts();
    f.target_single_slice = false;   // no colour target is written
    f.other_attachments_single_slice = false;
    f.depth_only = true;
    f.depth_bound = true;
    f.depth_view_known = true;
    f.depth_first_slice = 0;
    f.depth_slice_count = 6;
    const NggDrawAdmission cube = admit_ngg_draw(kena_vs_only_registers(), f, radv());
    ASSERT_TRUE(cube.ok()) << cube.refusal;
    EXPECT_EQ(cube.depth_slice_fanout, 6u);
    EXPECT_EQ(cube.depth_first_slice, 0u);
    EXPECT_EQ(cube.layer_slices, 6u) << "a layer at or above 6 is culled";
    EXPECT_EQ(cube.route, NggLayerRoute::None) << "replayed per slice, not routed";

    NggDrawFacts offset = f;
    offset.depth_first_slice = 2;
    offset.depth_slice_count = 2;
    const NggDrawAdmission pair = admit_ngg_draw(kena_vs_only_registers(), offset, radv());
    ASSERT_TRUE(pair.ok()) << pair.refusal;
    EXPECT_EQ(pair.depth_slice_fanout, 2u);
    EXPECT_EQ(pair.depth_first_slice, 2u);

    NggDrawFacts single = f;
    single.depth_slice_count = 1;
    const NggDrawAdmission one = admit_ngg_draw(kena_vs_only_registers(), single, radv());
    ASSERT_TRUE(one.ok()) << one.refusal;
    EXPECT_EQ(one.depth_slice_fanout, 0u) << "one slice: the layer only culls";
    EXPECT_EQ(one.layer_slices, 1u);

    NggDrawFacts unbound = f;
    unbound.depth_bound = false;
    const NggDrawAdmission nothing = admit_ngg_draw(kena_vs_only_registers(), unbound, radv());
    ASSERT_TRUE(nothing.refusal);
    EXPECT_STREQ(nothing.refusal, "ngg-layer-target-not-layered");
    NggDrawFacts unknown = f;
    unknown.depth_view_known = false;
    unknown.depth_slice_count = 0;
    const NggDrawAdmission unseen = admit_ngg_draw(kena_vs_only_registers(), unknown, radv());
    ASSERT_TRUE(unseen.refusal);
    EXPECT_STREQ(unseen.refusal, "ngg-layer-target-not-single-slice");

    // Mixed: a written one-slice colour target beside the depth array.
    NggDrawFacts colour_2d = kena_vs_only_facts();
    colour_2d.depth_bound = true;
    colour_2d.depth_view_known = true;
    colour_2d.depth_slice_count = 6;
    const NggDrawAdmission mixed = admit_ngg_draw(kena_vs_only_registers(), colour_2d, radv());
    ASSERT_TRUE(mixed.refusal);
    EXPECT_STREQ(mixed.refusal, "ngg-layer-attachments-mixed");
    // Mixed: a layered colour volume beside a bound depth/stencil; without it, admitted.
    NggDrawFacts volume = kena_vs_only_facts();
    volume.target_single_slice = false;
    volume.target_slices = 4;
    const NggDrawAdmission alone = admit_ngg_draw(kena_vs_only_registers(), volume, radv());
    ASSERT_TRUE(alone.ok()) << alone.refusal << ": control, a volume with no depth";
    EXPECT_EQ(alone.depth_slice_fanout, 0u);
    volume.depth_bound = true;
    volume.depth_view_known = true;
    volume.depth_slice_count = 1;
    // Depth writes on but no test: the backend attaches nothing, so the volume still renders.
    const NggDrawAdmission write_only = admit_ngg_draw(kena_vs_only_registers(), volume, radv());
    ASSERT_TRUE(write_only.ok()) << write_only.refusal << ": no attached depth, no mixed shape";
    volume.depth_attached = true;
    const NggDrawAdmission with_depth = admit_ngg_draw(kena_vs_only_registers(), volume, radv());
    ASSERT_TRUE(with_depth.refusal);
    EXPECT_STREQ(with_depth.refusal, "ngg-layer-attachments-mixed");
}

// A VS-only draw whose input the shell cannot represent is refused by name, not admitted as
// triangles.
TEST(NggDrawAdmission, AVsOnlyDrawRefusesInputsItCannotRepresent) {
    for (uint32_t prim : {1u /* points */, 2u /* lines */, 5u /* fan */, 17u /* rect list */}) {
        NggDrawRegisters r = kena_vs_only_registers();
        r.primitive_type = prim;
        const NggDrawAdmission a = admit_ngg_draw(r, kena_vs_only_facts(), radv());
        ASSERT_TRUE(a.refusal) << "prim type " << prim;
        EXPECT_STREQ(a.refusal, "ngg-input-topology") << "prim type " << prim;
    }
    NggDrawRegisters strip = kena_vs_only_registers();
    strip.primitive_type = 6u;
    strip.pa_su_sc_mode_cntl = 0u;   // no culling, so strip order is not visible
    const NggDrawAdmission a = admit_ngg_draw(strip, kena_vs_only_facts(), radv());
    EXPECT_TRUE(a.ok()) << a.refusal << ": control, a strip without culling";
    EXPECT_EQ(a.topology, NggOutputTopology::TriangleList);
}

}   // namespace
