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
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
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

// Every VGPR but the v0 launch input starts with a hard value: s40 is not a user SGPR, so hardware
// never loads it and its value is real but unknown. That is what a "never-written" word below
// stands for. A truly never-written VGPR would not do: under #5185 it holds an arbitrary value
// hardware could hold, which a fresh predicate may read on any route (#5039). Every pc a test
// names is relative to the body, after this prefix.
const Words& hard_prefix() {
    static const Words prefix = [] {
        Words words;
        for (uint32_t vgpr = 1; vgpr < 32; ++vgpr)
            words.push_back(0x7e000200u | (vgpr << 17u) | 40u);   // v_mov_b32 vN, s40
        return words;
    }();
    return prefix;
}

// The CFG dispatcher, forced, so every conditional branch ends a case and every case entry reloads
// the register file from storage. s0..s23 are user data, so the image's T# words are genuine.
// With `natural`, the dispatcher is not forced: an irreducible loop after the body requires it
// (test_cfg_spilled_mask_halves' kTail), so the compiler's refused-program retries run as they do
// for a captured program.
Result compile(const Words& body, bool natural = false) {
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    config.force_exchange_dispatcher = !natural;
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
    const Words irreducible{0x7e020280u, 0x7e040280u, 0x7c020300u, 0xbf860001u,
                            0x7e040281u, 0x7d840100u, 0xbf870001u, 0xbf82fffdu};
    const Words code = cat({hard_prefix(), body, natural ? irreducible : Words{},
                            {0x7e12028du, 0xe0702000u, 0x80020900u, end}});
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
    pc += static_cast<uint32_t>(hard_prefix().size());   // fixture pcs name the body
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

// #5182: the register inventory names every image address, not only VADDR. Each destination is
// below the extra address, so only the address can raise the count.
TEST(DispatcherEntryActiveWords, TheRegisterInventoryCountsEveryImageAddress) {
    const auto highest = [](const Words& words) {
        std::vector<Rdna2Inst> instructions;
        rdna2_walk(words.data(), words.size(), instructions);
        return shader_max_vgpr(instructions);
    };
    // image_load v1, v[10:11]: the consecutive second address.
    EXPECT_EQ(highest({0xf0000108u, 0x0004010au, end}), 11);
    // image_load v1, [v10, v30]: the NSA second address.
    EXPECT_EQ(highest({0xf000010au, 0x0004010au, 0x0000001eu, end}), 30);
    // image_sample_lz v1, [v10, v30], s[16:23], s[24:27].
    EXPECT_EQ(highest({0xf09c010au, 0x00c4010au, 0x0000001eu, end}), 30);
    // image_store v1, [v10, v30]: an unmodelled shape still names its NSA register.
    EXPECT_EQ(highest({0xf020010au, 0x0004010au, 0x0000001eu, end}), 30);
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

// ---- #5181: an EXEC-or-zero select, spilled with V_WRITELANE and reloaded, restores EXEC ----

// Kena cs 0x5008e80000 pc299-405: SCC; vcc = SCC ? exec : 0; s[20:21] = s[6:7] & vcc; both halves
// spilled to v18 lanes 7/6 (high first), s[20:21] reused as data, reloaded with V_READLANE, and
// `s_mov_b64 exec, s[20:21]`. That EXEC is a subset of the one v8 was read under, so v8 stays
// genuine across the next case edge. Encodings from llvm-mc -mcpu=gfx1030 -mattr=+wavefrontsize64.
constexpr uint32_t cselect_exec_0 = 0x85ea807eu;   // s_cselect_b64 vcc, exec, 0
constexpr uint32_t cselect_0_exec = 0x85ea7e80u;   // s_cselect_b64 vcc, 0, exec
constexpr uint32_t cselect_s6_0 = 0x85ea8006u;   // s_cselect_b64 vcc, s[6:7], 0
constexpr uint32_t and20 = 0x87946a06u;   // s_and_b64 s[20:21], s[6:7], vcc
constexpr uint32_t and20_exec = 0x87947e06u;   // s_and_b64 s[20:21], s[6:7], exec
constexpr uint32_t and20_data = 0x87940606u;   // s_and_b64 s[20:21], s[6:7], s[6:7]
constexpr uint32_t clear20 = 0xbe940380u;   // s_mov_b32 s20, 0
constexpr uint32_t clear21 = 0xbe950380u;   // s_mov_b32 s21, 0
constexpr uint32_t exec20 = 0xbefe0414u;   // s_mov_b64 exec, s[20:21]
Words spill21() {
    return {0xd7610012u, 0x00010e15u};
}   // v_writelane_b32 v18, s21, 7
Words spill20() {
    return {0xd7610012u, 0x00010c14u};
}   // v_writelane_b32 v18, s20, 6
Words reload21() {
    return {0xd7600015u, 0x00010f12u};
}   // v_readlane_b32 s21, v18, 7
Words reload20() {
    return {0xd7600014u, 0x00010d12u};
}   // v_readlane_b32 s20, v18, 6

// SCC; <select>; s[20:21] = s[6:7] & <select>; [spill; reuse; reload]; exec = s[20:21]
Words narrowed_restore(uint32_t select, bool spill, uint32_t combine = and20) {
    Words code{scalar_test, select, combine};
    if (spill) code = cat({code, spill21(), spill20(), {clear20, clear21}, reload21(), reload20()});
    code.push_back(exec20);
    return code;
}

// save; A; execz R; read v8; <narrow>; [edge: execz R']; compare v8; R: restore.
Words narrowed_edge(const Words& narrow) {
    const Words body = cat({lds_v8(), narrow});
    return cat({{save_exec, cmpx_a, execz(static_cast<uint32_t>(body.size()) + 2u)},
                body,
                {execz(1u), compare8, restore_exec}});
}
uint32_t compare_pc(const Words& narrow) {
    return 3u + 2u + static_cast<uint32_t>(narrow.size()) + 1u;
}
// Kena's spilled pair has no physical words, so its spill needs the #5175 Bool-half retry, which
// runs only when the dispatcher is not forced.
void narrowed_accepted(const Words& narrow) {
    accepted(compile(narrowed_edge(narrow), true));
}
// The terminal reason names the FIRST attempt's refusal. When that is the word-less spill the
// retry repairs, the retry's own refusal (at the compare, checked by hand with its diagnostics
// committed) is not reported, so only the empty module is asserted; each such twin has an
// accepted counterpart differing in one instruction.
void narrowed_refused(const Words& narrow, bool reason_names_compare = true) {
    const Result result = compile(narrowed_edge(narrow), true);
    if (reason_names_compare)
        refused(result, compare_pc(narrow));
    else
        EXPECT_TRUE(result.module.empty());
}

TEST(DispatcherEntryActiveWords, AnExecOrZeroSelectCarriesTheCurrentExecWords) {
    narrowed_accepted(narrowed_restore(cselect_exec_0, false));
    narrowed_accepted(narrowed_restore(cselect_0_exec, false));
    // The non-zero arm is scalar data, not EXEC: its lanes may lie outside A.
    narrowed_refused(narrowed_restore(cselect_s6_0, false));
}

TEST(DispatcherEntryActiveWords, ASavedMaskKeepsItsWordsThroughAWritelaneSpill) {
    // Kena's shape, both mechanisms together.
    narrowed_accepted(narrowed_restore(cselect_exec_0, true));
    // The spill alone, over a mask whose words come from an AND with EXEC (already carried).
    narrowed_accepted(narrowed_restore(0xbf800000u, true, and20_exec));
    narrowed_refused(narrowed_restore(cselect_s6_0, true));
    // v8 is overwritten from a never-written v20 while the pair is spilled: the saved mask's
    // lanes no longer hold a genuine v8.
    const Words destroyed = cat({{scalar_test, cselect_exec_0, and20},
                                 spill21(),
                                 spill20(),
                                 {clear20, clear21, 0x7e100314u},   // v_mov_b32 v8, v20
                                 reload21(),
                                 reload20(),
                                 {exec20}});
    narrowed_refused(destroyed, false);
}

// Two instances of s[20:21] spilled to different slots: B = s[6:7] & s[6:7] (scalar data, any
// lanes) to lanes 5/4 FIRST, then Kena's A to lanes 7/6. Reloading A's slots restores A's words;
// reloading B's must not, although A was the pair's later (and last spilled) instance.
Words two_instances(bool reload_a) {
    const Words reload = reload_a ? cat({reload21(), reload20()})
                                  : Words{0xd7600015u, 0x00010a12u,   // v_readlane_b32 s21, v18, 5
                                          0xd7600014u, 0x00010812u};   // v_readlane_b32 s20, v18, 4
    return cat(
        {{and20_data},
         {0xd7610012u, 0x00010a15u, 0xd7610012u, 0x00010814u},   // v18[5] = s21, v18[4] = s20
         {scalar_test, cselect_exec_0, and20},
         spill21(),
         spill20(),
         {clear20, clear21},
         reload,
         {exec20}});
}

TEST(DispatcherEntryActiveWords, AReloadRestoresOnlyItsOwnInstancesWords) {
    narrowed_accepted(two_instances(true));
    narrowed_refused(two_instances(false), false);
}

TEST(DispatcherEntryActiveWords, AnExecWriteBetweenTheSaveAndTheSpillKeepsTheSavedLanes) {
    // The mask is saved under the PARENT EXEC, before v8 exists; EXEC then narrows to A, v8 is
    // read there, and the pair is spilled and restored: its lanes outside A see a never-written
    // v8. Words snapshotted from the EXEC current at the spill would wrongly include it.
    const Words code = cat({{save_exec, scalar_test, cselect_exec_0, and20, cmpx_a, execz(15u)},
                            lds_v8(),
                            spill21(),
                            spill20(),
                            {clear20, clear21},
                            reload21(),
                            reload20(),
                            {exec20, execz(1u), compare8, restore_exec}});
    refused(compile(code, true), 20);
    // Saved AFTER the read, under A, the same spill is Kena's accepted shape (above).
}

// ---- #5181: a lane-mask select is lane-wise in its mask's Bool --------------------------------

// Kena cs 0x5008e80000 pc520/562: s[28:29] = (s19 < 1.0) (a VOPC with an SGPR destination), then
// `v_cndmask_b32_e64 v1, v1, v0, s[28:29]`. Each lane selects with its own bit of a live mask, so
// the result is genuine wherever both selected words are.
Words mask28() {
    return {0x7c03e4f9u, 0x86869c13u};
}   // v_cmp_lt_f32_sdwa s[28:29], s19, 1.0
Words select_v8_28() {
    return {0xd501000au, 0x00721080u};
}   // v_cndmask_b32_e64 v10, 0, v8, s[28:29]
Words select_v8_30() {
    return {0xd501000au, 0x007a1080u};
}   // v_cndmask_b32_e64 v10, 0, v8, s[30:31]
Words select_v20_28() {
    return {0xd501000au, 0x00722880u};
}   // v_cndmask_b32_e64 v10, 0, v20, s[28:29]

// save; A; execz R; read v8; <mask; select into v10>; [edge: execz R']; compare v10; R: restore.
Words selected_edge(const Words& select) {
    const Words body = cat({lds_v8(), select});
    return cat({{save_exec, cmpx_a, execz(static_cast<uint32_t>(body.size()) + 2u)},
                body,
                {execz(1u), compare10, restore_exec}});
}
uint32_t selected_compare_pc(const Words& select) {
    return 3u + 2u + static_cast<uint32_t>(select.size()) + 1u;
}

TEST(DispatcherEntryActiveWords, ALaneMaskSelectKeepsItsSelectedWordsGenuine) {
    const Words kena = cat({mask28(), select_v8_28()});
    accepted(compile(selected_edge(kena)));
    // A selected word that is missing in the current EXEC.
    const Words missing = cat({mask28(), select_v20_28()});
    refused(compile(selected_edge(missing)), selected_compare_pc(missing));
    // An absent mask, and one defined on one path only, are refused at the select itself.
    refused(compile(selected_edge(cat({mask28(), select_v8_30()}))), 7);
    refused(compile(selected_edge(cat({{scalar_test, 0xbf850002u}, mask28(), select_v8_28()}))), 9);
}

// The rule itself: the MASK must be a live MUST mask with no relayed word, and only SRC2 of
// v_cndmask_b32 is read that way.
TEST(DispatcherEntryActiveWords, OnlyALiveUnrelayedMaskIsALaneWiseSelector) {
    Facts facts;
    facts.state.current = {0, 8};
    const auto defined = [&](const Words& words, const std::set<int>& relayed_words = {}) {
        return active_vector_sources_defined(facts.state, decode(words), facts.relayed, facts.masks,
                                             facts.scalar_words, relayed_words, true);
    };
    for (int reg = 0; reg < 32; ++reg) facts.scalar_words.erase(reg);
    EXPECT_FALSE(defined(select_v8_28())) << "s[28:29] is not a mask";
    facts.masks.insert(28);
    EXPECT_TRUE(defined(select_v8_28()));
    EXPECT_FALSE(defined(select_v8_28(), {29})) << "a relayed high word";
    EXPECT_FALSE(defined(select_v8_28(), {28})) << "a relayed low word";
    EXPECT_FALSE(defined(select_v20_28())) << "v20 is relayed and not current";
    // v_add_f32_e64 v10, v8, s28: an ordinary scalar DATA source is not a mask read.
    EXPECT_FALSE(defined({0xd503000au, 0x00003908u}));
}

}   // namespace
