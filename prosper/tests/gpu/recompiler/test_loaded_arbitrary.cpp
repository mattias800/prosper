// #5193: reads at arbitrary-lane addresses, and the arbitrary property across a loop header.
//
// Decision on #5193 (2026-10-15): data loaded at an address whose lanes are arbitrary is itself
// arbitrary, for reads only, when (1) the resource is genuine, (2) the read is bounds-checked and
// side-effect free, and (3) per lane: a lane with a hard address input stays hard. Each accepted
// arm below has a refused twin that differs in exactly one of those: a hard descriptor, a store or
// atomic at the same arbitrary address, a cross-lane LDS form, and a read at a hard address.
//
// The companion rule: an arbitrary value met with a loop-carried word keeps the property and the
// loop's obligations, and a loop header keeps a word arbitrary. The back edge re-checks it: one
// that brings in a hard lane refuses every read that relied on the header.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_launch_vgprs.hpp"
#include "gpu/recompiler/rdna2_loaded_arbitrary.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
using Words = std::vector<uint32_t>;

constexpr uint32_t end = 0xbf810000u;
constexpr uint32_t compare6 = 0x7d840d06u;   // v_cmp_eq_u32 vcc, v6, v6 (a fresh predicate)
constexpr uint32_t mov6_3 = 0x7e0c0283u;     // v_mov_b32 v6, 3
constexpr uint32_t mov6_v7 = 0x7e0c0307u;    // v_mov_b32 v6, v7 (never written: arbitrary)
constexpr uint32_t mov6_s20 = 0x7e0c0214u;   // v_mov_b32 v6, s20 (never written: fabricated)
constexpr uint32_t mov7_s20 = 0x7e0e0214u;   // v_mov_b32 v7, s20
constexpr uint32_t add6_v6_v7 = 0x4a0c0f06u;   // v_add_nc_u32 v6, v6, v7 (accumulates)
// DS packets: word0 selects the opcode, word1 = vdst<<24 | data0<<8 | addr.
constexpr uint32_t ds_read_b32 = 0xd8d80000u;
constexpr uint32_t ds_add_rtn_u32 = 0xd8800000u;      // atomic, returns the old LDS word
constexpr uint32_t ds_wrxchg_rtn_b32 = 0xd8b40000u;   // a store that returns the old LDS word
constexpr uint32_t ds_bpermute_b32 = 0xdacc0000u;     // each lane reads another lane's DATA0
Words ds(uint32_t op, uint32_t vdst, uint32_t addr) {
    return {op, vdst << 24 | addr};
}

Words cat(std::initializer_list<Words> pieces) {
    Words code;
    for (const auto& piece : pieces) code.insert(code.end(), piece.begin(), piece.end());
    return code;
}
std::vector<Rdna2Inst> decode(const Words& code) {
    std::vector<Rdna2Inst> instructions;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), instructions), code.size());
    return instructions;
}

struct Result {
    Words module;
    std::string reason;
};
void accepted(const Result& result) {
    EXPECT_FALSE(result.module.empty()) << result.reason;
}
void refused(const Result& result, const char* reason = "fabricated-fresh-mask-source") {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find(reason), std::string::npos) << result.reason;
}

// A Wave64 compute program, TIDIG_COMP_CNT = 0 (hardware loads v0 only), with a genuine output
// store of the last predicate: v9 = vcc ? 1 : 0; buffer_store_dword v9, v0, s[12:15] offen.
Result compile_compute(const Words& body) {
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    config.user_sgprs.resize(8);
    prosper::test::complete_compute_launch(config, 2);
    ShaderResourceTable table;
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 12;
    table.resources.push_back(output);
    const Words code = cat({body, {0x7e140281u, 0x02121480u, 0xe0702000u, 0x80030900u, end}});
    static uint64_t next_address = 0x5193000000ull;
    const uint64_t address = ++next_address;
    auto module = recompile_compute(code.data(), code.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

// A scalar, top-tested loop (s22 from 0 to 3) whose body reads v6 as lane bits, then runs
// `rewrite`: the back edge carries whatever `rewrite` leaves in v6 into the next read.
Words loop_over_v6(const Words& rewrite) {
    const uint32_t body_size = static_cast<uint32_t>(rewrite.size() + 3);
    return cat({{0xbe960380u, 0xbe970383u, 0xbf0a1716u, 0xbf840000u | body_size, compare6},
                rewrite,
                {0x80168116u, 0xbf820000u | ((0x10000u - body_size - 2u) & 0xffffu)}});
}

size_t count_opcode(const Words& module, uint32_t opcode) {
    size_t count = 0;
    for (size_t at = 5; at < module.size();) {
        const uint32_t words = module[at] >> 16;
        if (!words) break;
        if ((module[at] & 0xffffu) == opcode) ++count;
        at += words;
    }
    return count;
}

// The producing emitter, for the unit arms: compute, launch inventory for TIDIG_COMP_CNT = 0.
struct Emitter {
    SpirvCompute b;
    RegState rs;
    Emitter() {
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.vreg[0] = b.localid_comp[0];
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.max_vgpr = 20;
        const auto inventory = decode({mov6_3, mov6_v7, 0x7e260314u});
        const auto launch = compute_launch_vgprs(0);
        seed_vector_word_definedness(rs, inventory, false, 64, &launch);
    }
};
// buffer_load_dword v6, v<addr>, s[12:15], 0 offen
Rdna2Inst buffer_load(uint32_t addr) {
    const Words code{0xe0301000u, 0x80030600u | addr};
    return decode(code).at(0);
}
}   // namespace

TEST(LoadedArbitrary, EncodingsNameTheFormsTheyClaim) {
    const auto in = decode(cat({ds(ds_read_b32, 6, 7),
                                ds(ds_add_rtn_u32, 6, 7),
                                ds(ds_wrxchg_rtn_b32, 6, 7),
                                ds(ds_bpermute_b32, 6, 7),
                                {add6_v6_v7}}));
    ASSERT_EQ(in.size(), 5u);
    EXPECT_EQ(in[0].opcode, 0x36u);
    EXPECT_EQ(in[1].opcode, 0x20u);
    EXPECT_EQ(in[2].opcode, 0x2du);
    EXPECT_EQ(in[3].opcode, kDsOpcodeBpermuteB32);
    for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(in[k].fmt, Rdna2Format::DS);
        EXPECT_EQ(in[k].src[0].value, 7);
        EXPECT_EQ(in[k].dst.value, 6);
    }
    EXPECT_EQ(in[4].fmt, Rdna2Format::VOP2);
    EXPECT_EQ(in[4].opcode, 0x25u);
    const auto load = buffer_load(7);
    EXPECT_EQ(load.fmt, Rdna2Format::MUBUF);
    EXPECT_EQ(load.src[0].value, 7);
    EXPECT_EQ(load.src[1].value, 12);
    EXPECT_TRUE(arbitrary_read_form(load));
}

// ---- Reads at arbitrary addresses --------------------------------------------------------------

TEST(LoadedArbitrary, AnLdsReadAtAnArbitraryAddressIsArbitrary) {
    accepted(compile_compute(cat({ds(ds_read_b32, 6, 7), {compare6}})));
}

TEST(LoadedArbitrary, AnLdsReadAtAHardAddressStaysHard) {
    // v7 = s20, which nothing wrote: a fabricated scalar, so every lane of the address is hard.
    refused(compile_compute(cat({{mov7_s20}, ds(ds_read_b32, 6, 7), {compare6}})));
}

TEST(LoadedArbitrary, AnAtomicOrStoreAtAnArbitraryAddressStaysHard) {
    // Positive control: the same atomic at the defined address v0 returns ordinary data.
    accepted(compile_compute(cat({ds(ds_add_rtn_u32, 6, 0), {compare6}})));
    refused(compile_compute(cat({ds(ds_add_rtn_u32, 6, 7), {compare6}})));
    refused(compile_compute(cat({ds(ds_wrxchg_rtn_b32, 6, 7), {compare6}})));
}

TEST(LoadedArbitrary, ACrossLaneLdsFormAtAnArbitraryAddressStaysHard) {
    // ds_bpermute_b32 v6, v7, v0: each lane reads lane (v7 >> 2)'s v0, so its result depends on
    // another lane's address. Positive control: the same gather at the defined address v0.
    accepted(compile_compute(cat({ds(ds_bpermute_b32, 6, 0), {compare6}})));
    refused(compile_compute(cat({ds(ds_bpermute_b32, 6, 7), {compare6}})));
    // ds_swizzle_b32 reads another lane's v7. This unit arm reads the instruction's input marks,
    // which its transfer applies.
    constexpr uint32_t ds_swizzle_b32 = 0xd8d40000u;
    Emitter e;
    const auto in = decode(cat({ds(ds_read_b32, 6, 7), ds(ds_swizzle_b32, 6, 7)}));
    ASSERT_EQ(in.at(1).opcode, 0x35u);
    const auto read = scalar_source_marks(e.rs, in[0]);
    EXPECT_TRUE(read.hard && read.arbitrary);
    const auto swizzle = scalar_source_marks(e.rs, in[1]);
    EXPECT_TRUE(swizzle.placeholder && swizzle.hard);
    EXPECT_FALSE(swizzle.arbitrary);
}

TEST(LoadedArbitrary, AnLdsReadAtAnArbitraryAddressBoundsItsIndex) {
    // Workgroup storage has no robust bound: the admitted read selects index 0 outside the
    // allocation and returns zero there (one OpULessThan per bounded dword).
    const Result arbitrary = compile_compute(cat({ds(ds_read_b32, 6, 7), {compare6}}));
    const Result defined = compile_compute(cat({ds(ds_read_b32, 6, 0), {compare6}}));
    ASSERT_FALSE(arbitrary.module.empty()) << arbitrary.reason;
    ASSERT_FALSE(defined.module.empty()) << defined.reason;
    constexpr uint32_t kOpULessThan = 176;
    EXPECT_EQ(count_opcode(arbitrary.module, kOpULessThan),
              count_opcode(defined.module, kOpULessThan) + 1);
}

TEST(LoadedArbitrary, ABufferReadIsArbitraryOnlyThroughAGenuineDescriptor) {
    Emitter e;
    const auto load = buffer_load(7);
    const auto marks = vector_source_marks(e.rs, load);
    EXPECT_TRUE(marks.placeholder && marks.hard);
    EXPECT_TRUE(marks.arbitrary) << "a bounded read at an arbitrary address";
    EXPECT_FALSE(vector_source_marks(e.rs, buffer_load(0)).placeholder) << "a defined address";
    // One V# word merged from an edge that never wrote it: a hard descriptor.
    e.rs.sreg_merge_placeholder.insert(13);
    const auto hard = vector_source_marks(e.rs, load);
    EXPECT_TRUE(hard.placeholder && hard.hard);
    EXPECT_FALSE(hard.arbitrary);
}

TEST(LoadedArbitrary, TheDispatcherAnswersTheSameForms) {
    const auto in =
        decode(cat({ds(ds_read_b32, 6, 7), ds(ds_add_rtn_u32, 6, 7), ds(ds_bpermute_b32, 6, 7)}));
    EXPECT_TRUE(dispatcher_arbitrary_read(in[0], {}, {}, true, nullptr));
    EXPECT_FALSE(dispatcher_arbitrary_read(in[1], {}, {}, true, nullptr));
    EXPECT_FALSE(dispatcher_arbitrary_read(in[2], {}, {}, true, nullptr));
    const auto load = buffer_load(7);
    EXPECT_TRUE(dispatcher_arbitrary_read(load, {}, {}, true, nullptr));
    EXPECT_FALSE(dispatcher_arbitrary_read(load, {}, {13}, true, nullptr)) << "a relayed V# word";
}

// ---- The loop header (companion rule) ----------------------------------------------------------

TEST(LoadedArbitrary, AHeaderKeepsAnArbitraryWordWhoseBackEdgeIsArbitraryOrDefined) {
    // v6 is never written before the loop: arbitrary on entry, read in the body.
    accepted(compile_compute(loop_over_v6({mov6_v7})));
    accepted(compile_compute(loop_over_v6({mov6_3})));
    accepted(compile_compute(loop_over_v6(ds(ds_read_b32, 6, 7))));
}

TEST(LoadedArbitrary, ABackEdgeThatBringsInAHardLaneRefusesTheArbitraryHeader) {
    refused(compile_compute(loop_over_v6({mov6_s20})), "fabricated on the back edge");
    refused(compile_compute(loop_over_v6(ds(ds_add_rtn_u32, 6, 7))), "fabricated on the back edge");
}

TEST(LoadedArbitrary, ADefinedHeaderWordHoldsWeaklyThroughAnArbitraryBackEdge) {
    // v6 = 3 before the loop: the header assumes the back edge defined. An arbitrary back edge
    // keeps the weaker promise a fresh predicate needs; an accumulation keeps it too (the GTA
    // 0x413e1ac shape: an arbitrary value met with a loop-carried word).
    accepted(compile_compute(cat({{mov6_3}, loop_over_v6({mov6_v7})})));
    accepted(compile_compute(cat({{mov6_3}, loop_over_v6({add6_v6_v7})})));
    // A hard back edge breaks it.
    refused(compile_compute(cat({{mov6_3}, loop_over_v6({mov6_s20})})),
            "fabricated on the back edge");
}

TEST(LoadedArbitrary, AnUnboundedAddressReadNeedsTheHeaderDefined) {
    // The body also uses v6 as an LDS address. A defined back edge keeps the address defined; an
    // arbitrary one would make that unbounded read reach any index, so the hold is not weak.
    accepted(
        compile_compute(cat({{mov6_3}, loop_over_v6(cat({ds(ds_read_b32, 8, 6), {mov6_3}}))})));
    refused(compile_compute(cat({{mov6_3}, loop_over_v6(cat({ds(ds_read_b32, 8, 6), {mov6_v7}}))})),
            "fabricated on the back edge");
}

TEST(LoadedArbitrary, ALoopThatReadsAVectorIntoAScalarDoesNotHoldWeakly) {
    // v_readfirstlane_b32 s21, v6 in the body: a scalar can carry no arbitrary value.
    constexpr uint32_t readfirstlane_s21_v6 = 0x7e2a0506u;
    const auto in = decode({readfirstlane_s21_v6});
    ASSERT_EQ(in.at(0).fmt, Rdna2Format::VOP1);
    ASSERT_EQ(in.at(0).opcode, 0x02u);
    accepted(compile_compute(cat({{mov6_3}, loop_over_v6({readfirstlane_s21_v6, mov6_3})})));
    refused(compile_compute(cat({{mov6_3}, loop_over_v6({readfirstlane_s21_v6, mov6_v7})})),
            "fabricated on the back edge");
}

TEST(LoadedArbitrary, ALoadedAddressRetainsItsWeakHeaderObligation) {
    // buffer_load_dword v8, v6, s[12:15], 0 offen; then use that value as an LDS address.
    // The first bounded read cannot grant the second read an unconditionally defined address.
    const Words load{0xe0301000u, 0x80030806u};
    const Words strong_read = ds(ds_read_b32, 9, 8);
    for (const auto& loads : {load, cat({load, {0xe0301000u, 0x80030808u}})}) {
        accepted(
            compile_compute(cat({{mov6_3}, loop_over_v6(cat({loads, strong_read, {mov6_3}}))})));
        accepted(compile_compute(cat({{mov6_3}, loop_over_v6(cat({loads, {mov6_v7}}))})));
        refused(
            compile_compute(cat({{mov6_3}, loop_over_v6(cat({loads, strong_read, {mov6_v7}}))})),
            "fabricated on the back edge");
    }
}
