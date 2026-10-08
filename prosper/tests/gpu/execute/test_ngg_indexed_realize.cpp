// An indexed merged-NGG draw through realize_draw_item (#3135 P6, #4733 review). The other P6 tests
// drive the planner, the decode, admission and the backend directly; this one drives the block in
// gpu_execute.hpp that joins them, which no live run has reached yet (both of Kena's indexed producers
// are still refused further down).
//
// The chain is synthetic and small: an ES fetch prolog that only transfers through s[6:7], and a
// merged main that requests GS_ALLOC_REQ, reads a constant buffer through the user-data V# at
// s[8:11] (so the linked fold publishes a table), uses the general SGPR-mask v_mbcnt (fail-closed in
// a vertex stage, so the per-vertex compile refuses and the NGG path is the one tried), and exports
// PRIM, POS0 and PARAM0 = (VertexID, InstanceID, cbuf, ...). Nothing renders: the assertions are on
// the DrawItem realization hands the backend.
//
// Also here, because the merged-NGG fetch and the ordinary fetch share it: the index-source rule's
// DrawIndexOffset branch, the only one where the 2- and 4-byte addresses differ.
#include "gpu/execute/gpu_execute.hpp"

#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/execute/ngg_draw_admission.hpp"
#include "gpu/execute/ngg_draw_indices.hpp"
#include "gpu/execute/ngg_live_draw.hpp"
#include "gpu/execute/ngg_subgroup_draw.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

namespace {

// s_mov_b32 s20, 0; s_setpc_b64 s[6:7]; s_code_end padding.
alignas(256) const uint32_t kProlog[] = {0xbe940380u, 0xbe802006u, 0xbf9f0000u, 0xbf9f0000u};

alignas(256) const uint32_t kMain[] = {
    0xbefe04c1u,   // s_mov_b64 exec, -1
    0x9394ff03u, 0x00040018u,   // s_bfe_u32 s20, s3, [27:24]: wave index
    0xbf068014u,   // s_cmp_eq_u32 s20, 0
    0xbf840002u,   // s_cbranch_scc0 +2
    0xb07c2004u,   // s_movk_i32 m0, 0x2004: 4 vertices, 2 primitives
    0xbf900009u,   // s_sendmsg GS_ALLOC_REQ
    0xf4200544u, 0xfa000000u,   // s_buffer_load_dword s21, s[8:11], 0
    0xbf8cc07fu,   // s_waitcnt lgkmcnt(0)
    0x7e140215u,   // v_mov_b32 v10, s21
    0x7e120302u,   // v_mov_b32 v9, v2
    0xbe9603c1u,   // s_mov_b32 s22, -1
    0xd765000bu, 0x00010016u,   // v_mbcnt_lo_u32_b32 v11, s22, 0 (a general SGPR mask)
    0xf8000941u, 0x00000009u,   // exp prim v9
    0xf80000cfu, 0x03020100u,   // exp pos0 v0..v3
    0xf800020fu, 0x030a0805u,   // exp param0 v5, v8, v10, v3
    0xbf810000u,   // s_endpgm
};

// #3135: the same main, but it first reloads its user SGPRs from the GS user-data address, as
// Kena's indexed producer 11562c72's main does: `s_load_dwordx8 s[8:15], s[0:1], 0` at entry.
alignas(256) const uint32_t kMainReload[] = {
    0xbefe04c1u,   // s_mov_b64 exec, -1
    0xf4100200u, 0xfa000000u,   // s_load_dwordx8 s[8:15], s[0:1], 0
    0xbf8cc07fu,   // s_waitcnt lgkmcnt(0)
    0x9394ff03u, 0x00040018u,   // s_bfe_u32 s20, s3, [27:24]: wave index
    0xbf068014u,   // s_cmp_eq_u32 s20, 0
    0xbf840002u,   // s_cbranch_scc0 +2
    0xb07c2004u,   // s_movk_i32 m0, 0x2004
    0xbf900009u,   // s_sendmsg GS_ALLOC_REQ
    0xf4200544u, 0xfa000000u,   // s_buffer_load_dword s21, s[8:11], 0
    0xbf8cc07fu,   // s_waitcnt lgkmcnt(0)
    0x7e140215u,   // v_mov_b32 v10, s21
    0x7e120302u,   // v_mov_b32 v9, v2
    0xbe9603c1u,   // s_mov_b32 s22, -1
    0xd765000bu, 0x00010016u,   // v_mbcnt_lo_u32_b32 v11, s22, 0 (a general SGPR mask)
    0xf8000941u, 0x00000009u,   // exp prim v9
    0xf80000cfu, 0x03020100u,   // exp pos0 v0..v3
    0xf800020fu, 0x030a0805u,   // exp param0 v5, v8, v10, v3
    0xbf810000u,   // s_endpgm
};

// Solid-green pixel stage (llvm-mc gfx1030; the same words test_gpu_execute uses).
alignas(256) const uint32_t kPs[] = {
    0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};

alignas(16) const uint32_t kConstants[4] = {0x1234u, 0, 0, 0};
// The constant buffer the reloaded user data names, and that user-data table (8 words: a V# for
// s[8:11], then s12..s15). Filled at run time: it holds a host address.
alignas(16) const uint32_t kReloadedConstants[4] = {0x5678u, 0, 0, 0};
alignas(16) uint32_t g_user_table[8] = {};

void set_pgm(GpuState& st, uint32_t lo, uint32_t hi, const void* code) {
    const uint64_t a = reinterpret_cast<uint64_t>(code);
    st.sh[lo] = static_cast<uint32_t>((a >> 8) & 0xffffffffu);
    st.sh[hi] = static_cast<uint32_t>((a >> 40) & 0xffu);
}

// An AGC shader blob as the SDK lays it out: the header's pointer fields are SELF-RELATIVE forward
// offsets that sceAgcCreateShader relocates in place (agc_fix_ptr). A test binary's static data can
// sit below 4 GiB, where an absolute pointer would be mistaken for an offset, so the fields are
// written as offsets to members that follow them.
struct ShaderBlob {
    AgcShaderHeader header{};
    ShaderReg registers[2]{};
    AgcShaderSpecials specials{};
};

template <typename T>
T* self_relative(T* const& field, const void* target) {
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(target) -
                                reinterpret_cast<uintptr_t>(&field));
}

bool register_blob(ShaderBlob& blob, const uint32_t* code, size_t bytes, uint32_t pgm_lo,
                   uint32_t pgm_hi, uint16_t user_data_end) {
    blob.header.file_header = 0x34333231u;
    blob.header.version = 0x18;
    blob.header.type = 2;
    blob.header.num_sh_registers = 2;
    blob.header.shader_size = static_cast<uint32_t>(bytes);
    blob.registers[0] = {pgm_lo, 0};
    blob.registers[1] = {pgm_hi, 0};
    blob.specials.user_data_range_start = 0;
    blob.specials.user_data_range_end = user_data_end;
    blob.header.sh_registers = self_relative(blob.header.sh_registers, blob.registers);
    blob.header.specials = self_relative(blob.header.specials, &blob.specials);
    auto create_shader = prosper::Hle::lookup("f3dg2CSgRKY");
    void* out = nullptr;
    return create_shader &&
           create_shader(reinterpret_cast<uint64_t>(&out), reinterpret_cast<uint64_t>(&blob.header),
                         reinterpret_cast<uint64_t>(code), 0, 0, 0) == 0 &&
           out == &blob.header && blob.header.specials == &blob.specials;
}

// The ES prolog (user data s8..s11: one V#) and the chained main, registered so realization can
// bound both programs and link the chain. Once per process.
bool register_chain_headers() {
    static const bool registered = [] {
        prosper::register_agc_hle();
        static ShaderBlob prolog, main, reload;
        return register_blob(prolog, kProlog, sizeof(kProlog), P::SPI_SHADER_PGM_LO_ES,
                             P::SPI_SHADER_PGM_HI_ES, 4) &&
               register_blob(main, kMain, sizeof(kMain), P::SPI_SHADER_PGM_LO_GS,
                             P::SPI_SHADER_PGM_HI_GS, 4) &&
               register_blob(reload, kMainReload, sizeof(kMainReload), P::SPI_SHADER_PGM_LO_GS,
                             P::SPI_SHADER_PGM_HI_GS, 4);
    }();
    return registered;
}

NggHostCapabilities radv_like() {
    NggHostCapabilities h;
    h.compute = true;
    h.vertex_pipeline_stores = true;
    h.shader_output_layer = true;
    h.geometry_shader = true;
    h.native_wave64 = false;
    h.max_compute_workgroup_subgroups = 16;
    h.max_compute_shared_memory = 65536;
    h.max_compute_workgroup_size_x = 1024;
    h.max_compute_workgroup_invocations = 1024;
    h.max_compute_workgroup_count_x = 65535;
    h.max_storage_buffer_range = 1u << 30;
    h.max_push_constants_size = 256;
    return h;
}

// The merged ES+GS NGG draw state of Kena's recorded registers, as a triangle LIST, into a 2D target.
GpuState merged_state() {
    GpuState st;
    set_pgm(st, P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, kProlog);
    set_pgm(st, P::SPI_SHADER_PGM_LO_GS, P::SPI_SHADER_PGM_HI_GS, kMain);
    set_pgm(st, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, kPs);
    st.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    st.cx[P::CB_TARGET_MASK] = 0xf;
    st.cx[P::VGT_SHADER_STAGES_EN] = 0x2030u;
    st.cx[P::VGT_GS_ONCHIP_CNTL] = 0x10020040u;
    st.uc[P::GE_CNTL] = 0x8040u;
    st.cx[P::GE_MAX_OUTPUT_PER_SUBGROUP] = 0xc0u;
    st.cx[P::VGT_GS_MAX_VERT_OUT] = 3u;
    st.cx[P::VGT_ESGS_RING_ITEMSIZE] = 4u;
    st.sh[P::SPI_SHADER_PGM_RSRC2_GS] = 0;
    st.cx[P::VGT_GS_OUT_PRIM_TYPE] = 2u;
    const uint64_t cbuf = reinterpret_cast<uint64_t>(kConstants);
    st.sh[P::SPI_SHADER_USER_DATA_GS_0 + 0] = static_cast<uint32_t>(cbuf);
    st.sh[P::SPI_SHADER_USER_DATA_GS_0 + 1] = static_cast<uint32_t>((cbuf >> 32) & 0xffffu);
    st.sh[P::SPI_SHADER_USER_DATA_GS_0 + 2] = sizeof(kConstants);
    st.sh[P::SPI_SHADER_USER_DATA_GS_0 + 3] = (22u << 12) | 0xfacu;
    st.index_type = 0;
    st.index_type_announced = true;
    return st;
}

class NggIndexedRealize : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(register_chain_headers());
        publish_ngg_host_capabilities(radv_like());
        reset_ngg_live_draw_cache_for_test();
    }
    // realize_draw_item on one indexed draw of `indices` (16-bit), `instances` instances.
    bool realize(GpuState st, const uint16_t* indices, uint32_t count, uint32_t instances,
                 DrawItem& out) {
        GpuState::Draw draw;
        draw.indexed = true;
        draw.index_count = count;
        draw.index_addr = reinterpret_cast<uint64_t>(indices);
        draw.instance_count = instances;
        st.draws.clear();
        st.draws.push_back(draw);
        return realize_draw_item(st, &st.draws[0], count, 0x10000u, /*log*/ false, out);
    }
};

// Scattered indices whose largest is 12: the realized NGG draw carries no index buffer, its vertex
// count is the vertex RANGE 13 (not the 6 indices), the linked table is folded over that range, and
// lane e of each subgroup runs VertexID = the e-th distinct index.
TEST_F(NggIndexedRealize, AnIndexedMergedDrawIsRealizedThroughTheSubgroupPath) {
    alignas(4) static const uint16_t kIndices[6] = {7, 3, 12, 12, 3, 9};
    DrawItem item;
    ASSERT_TRUE(realize(merged_state(), kIndices, 6, 2, item));
    ASSERT_TRUE(item.ngg_subgroup) << "the draw was not realized through the subgroup path";
    EXPECT_TRUE(item.indices.empty()) << "the backend refuses an NGG draw that carries indices";
    EXPECT_EQ(item.vertex_count, 13u) << "max index + 1, not the index count";
    ASSERT_TRUE(item.vrt);
    EXPECT_EQ(item.vrt->vertices_per_instance, 13u) << "the fold is sized by the vertex range";
    const NggSubgroupDraw& ngg = *item.ngg_subgroup;
    ASSERT_EQ(ngg.plan.subgroups.size(), 2u) << "one subgroup per instance";
    ASSERT_EQ(ngg.groups.size(), 1u);
    const std::vector<uint32_t>& launch = ngg.groups[0].launch_words;
    const uint32_t expect[4] = {7, 3, 12, 9};
    for (uint32_t block = 0; block < 2u; ++block)
        for (uint32_t lane = 0; lane < 4u; ++lane) {
            const size_t at = (static_cast<size_t>(block) * 64u + lane) * kNggLaunchWordsPerLane;
            EXPECT_EQ(launch[at + 5], expect[lane]) << "block " << block << " lane " << lane;
            EXPECT_EQ(launch[at + 8], block) << "InstanceID";
        }
    EXPECT_EQ(ngg_live_draw_cache_stats().indexed_draws, 1u);
}

// #3135: a main that reads s0:s1 is admitted when the GS user-data address is in the draw state.
// The linked fold follows s0:s1 into the table, so the constant buffer it resolves is the one the
// RELOADED V# names (not the one in SPI_SHADER_USER_DATA_GS_0..3), and the shell receives the
// address as the two push-constant words after the user SGPRs. Without the address registers the
// same draw is refused by the ABI rule's name.
TEST_F(NggIndexedRealize, AMainReadingTheUserDataAddressIsSuppliedIt) {
    const uint64_t cbuf = reinterpret_cast<uint64_t>(kReloadedConstants);
    g_user_table[0] = static_cast<uint32_t>(cbuf);
    g_user_table[1] = static_cast<uint32_t>((cbuf >> 32) & 0xffffu);
    g_user_table[2] = sizeof(kReloadedConstants);
    g_user_table[3] = (22u << 12) | 0xfacu;
    GpuState st = merged_state();
    set_pgm(st, P::SPI_SHADER_PGM_LO_GS, P::SPI_SHADER_PGM_HI_GS, kMainReload);
    alignas(4) static const uint16_t kIndices[3] = {0, 1, 2};
    DrawItem refused;
    EXPECT_FALSE(realize(st, kIndices, 3, 1, refused));
    EXPECT_FALSE(refused.ngg_subgroup) << "control: no address registers, s0 is undefined";

    const uint64_t table = reinterpret_cast<uint64_t>(g_user_table);
    st.sh[P::SPI_SHADER_USER_DATA_ADDR_LO_GS] = static_cast<uint32_t>(table);
    st.sh[P::SPI_SHADER_USER_DATA_ADDR_HI_GS] = static_cast<uint32_t>(table >> 32);
    DrawItem item;
    ASSERT_TRUE(realize(st, kIndices, 3, 1, item));
    ASSERT_TRUE(item.ngg_subgroup) << "the draw was not realized through the subgroup path";
    EXPECT_EQ(item.ngg_subgroup->push_constants,
              (std::vector<uint32_t>{
                  st.sh[P::SPI_SHADER_USER_DATA_GS_0], st.sh[P::SPI_SHADER_USER_DATA_GS_0 + 1],
                  st.sh[P::SPI_SHADER_USER_DATA_GS_0 + 2], st.sh[P::SPI_SHADER_USER_DATA_GS_0 + 3],
                  static_cast<uint32_t>(table), static_cast<uint32_t>(table >> 32)}))
        << "s8..s11, then s0:s1";
    ASSERT_TRUE(item.vrt);
    bool reloaded = false, original = false;
    for (const ShaderResource& r : item.vrt->resources) {
        reloaded |= r.gpu_addr == cbuf;
        original |= r.gpu_addr == reinterpret_cast<uint64_t>(kConstants);
    }
    EXPECT_TRUE(reloaded) << "the fold followed s0:s1 to the reloaded V#";
    EXPECT_FALSE(original) << "the stale user-data V# is not what the main reads";

    // A reader of s0:s1 needs two push words beyond its user SGPRs: with RSRC2's count at 31
    // there is no room, and the draw is refused by name rather than pushing past the budget.
    for (uint32_t k = 4; k < 31; ++k) st.sh[P::SPI_SHADER_USER_DATA_GS_0 + k] = 0;
    st.sh[P::SPI_SHADER_PGM_RSRC2_GS] = 31u << 1;
    DrawItem full;
    EXPECT_FALSE(realize(st, kIndices, 3, 1, full));
    EXPECT_FALSE(full.ngg_subgroup) << "31 user SGPRs + s0:s1 exceed the 32-word push budget";
}

// The same draw with a garbage index is refused by name rather than sized to 2^28 vertices (#461).
TEST_F(NggIndexedRealize, AGarbageIndexIsRefusedByName) {
    alignas(4) static const uint32_t kGarbage[3] = {0, 1, 0x0F000000u};
    GpuState st = merged_state();
    st.index_type = 1;   // announced 32-bit
    GpuState::Draw draw;
    draw.indexed = true;
    draw.index_count = 3;
    draw.index_addr = reinterpret_cast<uint64_t>(kGarbage);
    st.draws.push_back(draw);
    DrawItem item;
    OperationRealizationFailure failure;
    EXPECT_FALSE(realize_draw_item(st, &st.draws[0], 3, 0x10000u, /*log*/ false, item, &failure));
    EXPECT_FALSE(item.ngg_subgroup);
    const NggDrawIndexFetch fetched =
        fetch_ngg_draw_indices(st, st.draws[0], /*vb_records_unclamped*/ 0);
    EXPECT_STREQ(fetched.refusal, "ngg-index-range");
}

// A DrawIndexOffset whose guest never announced an index size: the CP computed index_addr at the
// 2-byte stride, and the #304 detector re-reads the buffer at 4 bytes from index_base + offset * 4.
// The rule must move the ADDRESS with the size; reading 4-byte elements at the 2-byte address gives
// different indices. Both paths are checked: the shared rule, the ordinary fetch's realized indices,
// and the merged-NGG fetch.
TEST(DrawIndexSource, AnOffsetDrawRecomputesTheAddressAtTheDetectedSize) {
    // Two leading 32-bit words, then the real quad at 32-bit element 2. Read at the CP's 2-byte
    // address (base + 4) the six 16-bit words are 5, 0, 0, 0, 1, 0: every odd word zero, so the
    // unannounced-32-bit fingerprint matches. Read as 4-byte elements from that same address they
    // would be 5, 0, 1, 2, 2, 1 -- the wrong quad.
    alignas(4) static const uint32_t kBuffer[8] = {0x00090008u, 5, 0, 1, 2, 2, 1, 3};
    GpuState st;
    st.index_type = 0;
    st.index_type_announced = false;
    GpuState::Draw draw;
    draw.indexed = true;
    draw.index_count = 6;
    draw.from_offset = true;
    draw.index_base = reinterpret_cast<uint64_t>(kBuffer);
    draw.index_offset = 2;
    draw.index_addr = draw.index_base + uint64_t{2} * draw.index_offset;   // what the CP computed
    const DrawIndexSource source = resolve_draw_index_source(st, draw, 6, 0);
    EXPECT_STREQ(source.detected, "zero-high-half");
    EXPECT_EQ(source.element_bytes, 4u);
    EXPECT_EQ(source.addr32, draw.index_base + 8u);
    EXPECT_EQ(source.addr, source.addr32) << "the address moves with the detected size";

    const NggDrawIndexFetch fetched = fetch_ngg_draw_indices(st, draw, 0);
    ASSERT_TRUE(fetched.indices) << (fetched.refusal ? fetched.refusal : "");
    EXPECT_EQ(*fetched.indices, (std::vector<uint32_t>{0, 1, 2, 2, 1, 3}));
    EXPECT_EQ(fetched.vertex_range, 4u);

    // Announced 16-bit: no detection, the CP's address and size stand.
    st.index_type_announced = true;
    const DrawIndexSource announced = resolve_draw_index_source(st, draw, 6, 0);
    EXPECT_EQ(announced.detected, nullptr);
    EXPECT_EQ(announced.element_bytes, 2u);
    EXPECT_EQ(announced.addr, draw.index_addr);
}

}   // namespace
