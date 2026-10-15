// #5173: in dispatcher mode a vector word completed under a narrowed EXEC lost its confinement at
// the next block entry, and a fresh predicate reading it under the same EXEC was refused as
// `fabricated-fresh-mask-source`. The CFG already carries the lattice that answers this -- the
// ActiveVectorWords MUST set of words genuine in every lane of the CURRENT EXEC, re-anchored to the
// reloaded EXEC at each case entry (restore_active_vector_words) -- but two transfers lost it on
// Kena: Bridge of Spirits' compute programs (cs 0x5006f20000, 0x5006f90000, 0x500a1f0000,
// 0x500a210000, 0x500a5c0000, 0x500a620000, 0x500b1d0000):
//   A. ordinary memory reads (image loads/samples, LDS reads, typed and untyped FORMAT loads) never
//      entered the set, although each writes every EXEC lane's result from that lane's addresses;
//   B. the relay MAY fact tainted a destination whose old lanes were genuine whenever a source was
//      relayed, even when that source was genuine in every lane the write reads (a word completed
//      under a narrower EXEC), so the dispatcher reloaded it as a placeholder in every lane.
// Every accepted shape below has refused twins in which the missing lanes CAN be observed: an EXEC
// widening between the write and the edge, a join whose other predecessor carries a different EXEC,
// and a loop back edge that changes EXEC.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_active_vector_words.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <ios>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
using Words = std::vector<uint32_t>;

// Encodings from llvm-mc -mcpu=gfx1030 -mattr=+wavefrontsize64.
constexpr uint32_t save_exec = 0xbe84047eu;   // s_mov_b64 s[4:5], exec
constexpr uint32_t cmpx_a = 0x7da6009fu;   // v_cmpx_le_u32 31, v0: EXEC A = EXEC & (v0 >= 31)
constexpr uint32_t cmpx_b = 0x7da600afu;   // v_cmpx_le_u32 47, v0: EXEC B, a subset of A
constexpr uint32_t restore_exec =
    0xbefe0404u;   // s_mov_b64 exec, s[4:5]: widens back to the parent
constexpr uint32_t full_exec = 0xbefe04c1u;   // s_mov_b64 exec, -1
constexpr uint32_t mov10 = 0x7e140280u;   // v_mov_b32 v10, 0
constexpr uint32_t mov1 = 0x7e020300u;   // v_mov_b32 v1, v0
constexpr uint32_t max10 = 0x20141080u;   // v_max_f32 v10, 0, v8 (complete, lane-wise)
constexpr uint32_t compare8 = 0x7d821084u;   // v_cmp_lt_u32 vcc, 4, v8: a fresh predicate
constexpr uint32_t compare10 = 0x7d821484u;   // v_cmp_lt_u32 vcc, 4, v10
constexpr uint32_t scalar_test = 0xbf068000u;   // s_cmp_eq_u32 s0, 0: a genuine scalar condition
constexpr uint32_t counter0 = 0xbe860380u;   // s_mov_b32 s6, 0
constexpr uint32_t counter_add = 0x80068106u;   // s_add_u32 s6, s6, 1
constexpr uint32_t counter_test = 0xbf0a8306u;   // s_cmp_lt_u32 s6, 3
constexpr uint32_t end = 0xbf810000u;
Words lds_v8() {
    return {0xd8d80000u, 0x08000000u};
}   // ds_read_b32 v8, v0
// ds_read_b32 v8, v20 (v20 never written)
Words lds_v8_v20() {
    return {0xd8d80000u, 0x08000014u};
}
// image_load v8, [v0, v1], s[16:23] dmask:0x1 dim:2D
Words image_v8() {
    return {0xf000010au, 0x00040800u, 0x00000001u};
}
// [v0, v20]
Words image_v8_v20() {
    return {0xf000010au, 0x00040800u, 0x00000014u};
}

uint32_t execz(uint32_t offset) {
    return 0xbf880000u | offset;
}
uint32_t scc1(uint32_t offset) {
    return 0xbf850000u | (offset & 0xffffu);
}

Words cat(std::initializer_list<Words> pieces) {
    Words code;
    for (const auto& piece : pieces) code.insert(code.end(), piece.begin(), piece.end());
    return code;
}

Rdna2Inst decode(const Words& words) {
    std::vector<Rdna2Inst> instructions;
    EXPECT_EQ(rdna2_walk(words.data(), words.size(), instructions), words.size());
    EXPECT_EQ(instructions.size(), 1u);
    return instructions.empty() ? Rdna2Inst{} : instructions.front();
}

struct Result {
    Words module;
    std::string reason;
};

// The CFG dispatcher, forced, so every conditional branch ends a case and every case entry reloads
// the register file from storage. s0..s23 are user data, so the image's T# words are genuine.
Result compile(const Words& body) {
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    config.force_exchange_dispatcher = true;
    config.user_sgprs.resize(24);
    prosper::test::complete_compute_launch(config, 2);
    ShaderResourceTable table;
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 8;
    table.resources.push_back(output);
    ShaderResource texture;
    texture.cls = ResourceClass::Texture;
    texture.binding = 4;
    texture.img_dim = 1;
    texture.width = texture.height = 64;
    texture.format = DataFormat::Float32;
    texture.num_components = 1;
    texture.sgpr_base = 16;
    table.resources.push_back(texture);
    // v9 = 13 is stored, so the program has an observable effect after the body.
    const Words code = cat({body, {0x7e12028du, 0xe0702000u, 0x80020900u, end}});
    static uint64_t next_address = 0x5173000000ull;
    const uint64_t address = ++next_address;
    auto module = recompile_compute(code.data(), code.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}
void accepted(const Result& result) {
    EXPECT_FALSE(result.module.empty()) << result.reason;
}
void refused(const Result& result, uint32_t pc) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << result.reason;
}

// ---- the decoder's image address inventory ----------------------------------------------------

TEST(DispatcherEntryActiveWords, ImageAddressCountsMatchTheRangesTheAssemblerRequires) {
    // Each consecutive encoding below was assembled by llvm-mc gfx1030 with exactly this many
    // address VGPRs; it rejects any other range size ("image address size does not match dim").
    const std::vector<std::pair<Words, uint32_t>> cases = {
        {{0xf0000100u, 0x00040800u}, 1u},   // image_load 1D
        {{0xf0000108u, 0x00040800u}, 2u},   // image_load 2D
        {{0xf0000110u, 0x00040800u}, 3u},   // image_load 3D
        {{0xf0000118u, 0x00040800u}, 3u},   // image_load CUBE
        {{0xf0000120u, 0x00040800u}, 2u},   // image_load 1D_ARRAY
        {{0xf0000128u, 0x00040800u}, 3u},   // image_load 2D_ARRAY
        {{0xf0000130u, 0x00040800u}, 3u},   // image_load 2D_MSAA
        {{0xf0000138u, 0x00040800u}, 4u},   // image_load 2D_MSAA_ARRAY
        {{0xf0040108u, 0x00040800u}, 3u},   // image_load_mip 2D
        {{0xf0040110u, 0x00040800u}, 4u},   // image_load_mip 3D
        {{0xf0800f08u, 0x00c40800u}, 2u},   // image_sample 2D
        {{0xf0840f08u, 0x00c40800u}, 3u},   // image_sample_cl 2D
        {{0xf0900f08u, 0x00c40800u}, 3u},   // image_sample_l 2D
        {{0xf0940f08u, 0x00c40800u}, 3u},   // image_sample_b 2D
        {{0xf0980f08u, 0x00c40800u}, 4u},   // image_sample_b_cl 2D
        {{0xf09c0f08u, 0x00c40800u}, 2u},   // image_sample_lz 2D
        {{0xf0a00f08u, 0x00c40800u}, 3u},   // image_sample_c 2D
        {{0xf0fc0f08u, 0x00c40800u}, 4u},   // image_sample_c_lz_o 2D
        {{0xf0dc0f08u, 0x00c40800u}, 3u},   // image_sample_lz_o 2D
        {{0xf0f80f10u, 0x00c40800u}, 7u},   // image_sample_c_b_cl_o 3D
        {{0xf0900f18u, 0x00c40800u}, 4u},   // image_sample_l CUBE
        {{0xf11c0108u, 0x00c40800u}, 2u},   // image_gather4_lz 2D
        {{0xf17c0108u, 0x00c40800u}, 4u},   // image_gather4_c_lz_o 2D
        // NSA: VADDR, then one byte per further address.
        {{0xf000010au, 0x00040800u, 0x00000014u}, 2u},   // [v0, v20]
        {{0xf0fc0f0au, 0x00c40804u, 0x00030201u}, 4u},   // c_lz_o [v4, v1, v2, v3]
        {{0xf0f80f14u, 0x00c40804u, 0x05030201u, 0x00000706u}, 7u},   // c_b_cl_o 3D, two dwords
    };
    for (const auto& [words, count] : cases)
        EXPECT_EQ(rdna2_mimg_read_address_count(decode(words)), count) << std::hex << words[0];
    const auto nsa = decode({0xf0f80f14u, 0x00c40804u, 0x05030201u, 0x00000706u});
    const std::vector<int> expected = {4, 1, 2, 3, 5, 6, 7};
    for (uint32_t index = 0; index < expected.size(); ++index)
        EXPECT_EQ(rdna2_mimg_address_vgpr(nsa, index), expected[index]) << index;
    EXPECT_EQ(rdna2_mimg_address_vgpr(decode({0xf0000110u, 0x00040805u}), 2u), 7);
}

TEST(DispatcherEntryActiveWords, UnmodelledImageAddressShapesAreUnknown) {
    for (const Words& words : {
             Words{0xf0880f08u, 0x00c40800u},   // image_sample_d: derivatives are not modelled
             Words{0xf1800308u, 0x00c40800u},   // image_get_lod
             Words{0xf0000108u, 0x40040800u},   // image_load 2D a16: one packed address VGPR
             Words{0xf000010au, 0x00040800u, 0x00001400u},   // NSA with a non-zero padding byte
             Words{0xf000010cu, 0x00040800u, 0x00000014u, 0u},   // two NSA dwords for one address
         })
        EXPECT_EQ(rdna2_mimg_read_address_count(decode(words)), 0u) << std::hex << words[0];
}

// ---- which memory reads complete the current EXEC's words -------------------------------------

struct Facts {
    ActiveVectorWords state;
    std::set<int> relayed = {1, 2, 8, 9, 20};   // never written on some path
    std::set<int> masks;
    std::set<int> scalar_words;
    Facts() {
        state.enabled = true;
        state.current = {0, 1};   // v1 is relayed but genuine wherever the current EXEC runs
        for (int reg = 0; reg < 28; ++reg) scalar_words.insert(reg);
    }
    bool defined(const Words& words) {
        return active_vector_sources_defined(state, decode(words), relayed, masks, scalar_words, {},
                                             true);
    }
};

TEST(DispatcherEntryActiveWords, LdsReadsWithACurrentAddressAreGenuineInTheCurrentExec) {
    Facts facts;
    EXPECT_TRUE(facts.defined(lds_v8()));
    EXPECT_FALSE(facts.defined(lds_v8_v20())) << "the address is missing in the current EXEC";
    EXPECT_FALSE(facts.defined({0xd8d40000u, 0x08000000u})) << "ds_swizzle reads other lanes";
    EXPECT_FALSE(facts.defined({0xd8da0000u, 0x08000000u})) << "GDS is device-global";
}

TEST(DispatcherEntryActiveWords, ImageReadsNeedEveryAddressAndDescriptorWord) {
    Facts facts;
    EXPECT_TRUE(facts.defined(image_v8()));
    EXPECT_TRUE(facts.defined({0xf0000108u, 0x00040800u}));   // consecutive v[0:1]
    EXPECT_FALSE(facts.defined(image_v8_v20())) << "the second address is missing";
    facts.relayed.erase(20);
    EXPECT_FALSE(facts.defined(image_v8_v20()))
        << "an address must be current, not merely unrelayed";
    facts.relayed.insert(20);
    EXPECT_FALSE(facts.defined({0xf0000110u, 0x00040800u})) << "3D reads v2, never written";
    EXPECT_FALSE(facts.defined({0xf0000108u, 0x40040800u})) << "an A16 layout is not modelled";
    EXPECT_FALSE(facts.defined({0xf0000108u, 0x80040800u})) << "D16 keeps half of each word";
    EXPECT_FALSE(facts.defined({0xf0010108u, 0x00040800u})) << "TFE adds a status word";
    EXPECT_FALSE(facts.defined({0xf0200108u, 0x00040800u})) << "image_store writes no result";
    facts.scalar_words.erase(19);
    EXPECT_FALSE(facts.defined(image_v8())) << "a T# word without a value";
    facts.scalar_words.insert(19);
    EXPECT_TRUE(facts.defined({0xf09c0f08u, 0x00c40800u}));   // image_sample_lz, sampler s[24:27]
    facts.scalar_words.erase(24);
    EXPECT_FALSE(facts.defined({0xf09c0f08u, 0x00c40800u})) << "an S# word without a value";
}

TEST(DispatcherEntryActiveWords, FormatLoadsAreCompleteButTheirStatusFormsAreNot) {
    Facts facts;
    EXPECT_TRUE(facts.defined({0xe0002000u, 0x80020800u}));   // buffer_load_format_x v8, v0
    EXPECT_TRUE(facts.defined({0xe00c2000u, 0x80020800u}));   // buffer_load_format_xyzw
    EXPECT_TRUE(facts.defined({0xe8b02000u, 0x80020800u}));   // tbuffer_load_format_x
    EXPECT_FALSE(facts.defined({0xe0002000u, 0x80820800u})) << "MUBUF TFE";
    EXPECT_FALSE(facts.defined({0xe8b02000u, 0x80820800u})) << "MTBUF TFE";
    EXPECT_FALSE(facts.defined({0xe2002000u, 0x80020800u})) << "buffer_load_format_d16_x";
    EXPECT_FALSE(facts.defined({0xe0002000u, 0x80020814u})) << "v20 address";
}

// ---- A: a memory read completed under A survives the dispatcher edge under A ------------------

// save; A; execz R; <read v8>; [edge: execz R]; compare v8; R: restore.
Words same_exec_edge(const Words& read, bool widen_before_edge) {
    const uint32_t tail = widen_before_edge ? 1u : 0u;
    Words code =
        cat({{save_exec, cmpx_a, execz(static_cast<uint32_t>(read.size()) + 2u + tail)}, read});
    if (widen_before_edge) code.push_back(full_exec);
    return cat({code, {execz(1u), compare8, restore_exec}});
}

TEST(DispatcherEntryActiveWords, AnLdsReadKeepsItsExecAcrossTheCaseEdge) {
    accepted(compile(same_exec_edge(lds_v8(), false)));
    // Widening between the write and the edge: the compare then reads v8's never-written lanes.
    refused(compile(same_exec_edge(lds_v8(), true)), 7);
}

TEST(DispatcherEntryActiveWords, AnImageReadKeepsItsExecAcrossTheCaseEdge) {
    // cs 0x500a1f0000 / 0x500a210000: image_load [vX, v9] under s_and_saveexec.
    accepted(compile(cat({{mov1}, same_exec_edge(image_v8(), false)})));
    refused(compile(cat({{mov1}, same_exec_edge(image_v8(), true)})), 9);
    refused(compile(cat({{mov1}, same_exec_edge(image_v8_v20(), false)})), 8);
}

// ---- the three ways the reloaded EXEC can differ from the one the word was completed under ------

// save; A; read v8; scalar branch to J; <other arm>; J: compare v8; restore.
Words join(uint32_t other_arm) {
    return cat({{save_exec, cmpx_a},
                lds_v8(),
                {scalar_test, scc1(1u), other_arm, compare8, restore_exec}});
}

TEST(DispatcherEntryActiveWords, AJoinKeepsTheWordOnlyWhenEveryEdgeCarriesASubsetExec) {
    accepted(compile(join(cmpx_b)));   // the other edge narrows A further
    refused(compile(join(restore_exec)), 7);   // the other edge carries the wider parent
}

// save; A; read v8; s6 = 0; L: compare v8; <body EXEC write>; s6++; loop while s6 < 3; restore.
Words loop(uint32_t body_exec) {
    return cat({{save_exec, cmpx_a},
                lds_v8(),
                {counter0, compare8, body_exec, counter_add, counter_test, scc1(0x10000u - 5u),
                 restore_exec}});
}

TEST(DispatcherEntryActiveWords, ALoopBackEdgeKeepsTheWordOnlyUnderASubsetExec) {
    accepted(compile(loop(cmpx_b)));
    refused(compile(loop(restore_exec)), 5);   // the back edge widens EXEC
}

// ---- B: a complete write from genuine-in-EXEC sources keeps the old word's provenance ----------

// cs 0x500a5c0000 / 0x500a620000: [v10 = 0]; save; A; execz R; read v8; v10 = max(0, v8);
// R: <widen>; compare v10 -- under a WIDER EXEC, where v10's lanes outside A are its old ones.
// Widening to all lanes (s_mov_b64 exec, -1, as Kena does at pc168) clears every current-EXEC fact,
// so only the relay MAY fact can say v10 is genuine there; the launch EXEC is the full wave here.
Words accumulate(bool old_written, const Words& read, uint32_t widen = full_exec) {
    Words code;
    if (old_written) code.push_back(mov10);
    return cat({code,
                {save_exec, cmpx_a, execz(static_cast<uint32_t>(read.size()) + 1u)},
                read,
                {max10, widen, compare10}});
}

TEST(DispatcherEntryActiveWords, AWriteFromGenuineSourcesKeepsGenuineOldLanes) {
    accepted(compile(accumulate(true, lds_v8())));
    accepted(compile(cat({{mov1}, accumulate(true, image_v8())})));
    // Restoring the saved parent is the current-EXEC lattice's own case (its saved-mask words).
    accepted(compile(accumulate(true, lds_v8(), restore_exec)));
    // v10 was never written: the lanes outside A observe the missing old word.
    refused(compile(accumulate(false, lds_v8())), 7);
    refused(compile(accumulate(false, lds_v8(), restore_exec)), 7);
    // v8's address is missing, so v10 is not genuine even inside A.
    refused(compile(accumulate(true, lds_v8_v20())), 8);
    refused(compile(cat({{mov1}, accumulate(true, image_v8_v20())})), 10);
}

}   // namespace
