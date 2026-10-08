// Static admission of linked merged-NGG programs for the subgroup shell (#3135 P2). Every refusal
// reason is reached by a minimal program, and each is paired with a control one edit away that is
// admitted, so a refusal cannot pass by refusing everything. The captured Kena chain (tests/data)
// must be admitted with its real user-SGPR count and refused with one fewer.
#include "gpu/recompiler/ngg_subgroup_abi.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/test_data.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kExecAllOnes = 0xbefe04c1u;   // s_mov_b64 exec, -1
constexpr uint32_t kEnd = 0xbf810000u;           // s_endpgm
// EXP PRIM v9 (en=1) and EXP POS0 v0..v3.
constexpr std::array<uint32_t, 4> kExports = {0xf8000941u, 0x00000009u, 0xf80000cfu, 0x03020100u};

std::vector<uint32_t> program(std::initializer_list<uint32_t> body, bool exec_first = true,
                              const std::vector<uint32_t>& exports = {kExports.begin(),
                                                                      kExports.end()}) {
    std::vector<uint32_t> code;
    if (exec_first) code.push_back(kExecAllOnes);
    code.push_back(0x7e120280u);   // v_mov_b32 v9, 0
    code.insert(code.end(), body);
    code.push_back(0xb07c3005u);   // s_movk_i32 m0, 0x3005
    code.push_back(0xbf900009u);   // s_sendmsg GS_ALLOC_REQ
    code.insert(code.end(), exports.begin(), exports.end());
    code.push_back(kEnd);
    return code;
}

NggSubgroupAbiFacts analyze(const std::vector<uint32_t>& code, uint32_t user_sgprs = 0,
                            bool address = false) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    NggSubgroupAbiLaunch launch;
    launch.user_sgprs = user_sgprs;
    launch.user_data_address_known = address;
    return analyze_ngg_subgroup_abi(ins, launch);
}

std::vector<uint32_t> load_words(const char* name) {
    const auto path = prosper::test::tests_root(__FILE__) / "data" / name;
    std::vector<uint32_t> words;
    if (FILE* file = std::fopen(path.string().c_str(), "rb")) {
        uint32_t word = 0;
        while (std::fread(&word, 4, 1, file) == 1) words.push_back(word);
        std::fclose(file);
    }
    return words;
}

std::vector<uint32_t> kena_linked() {
    const auto prolog = load_words("ngg_merged_es_prolog.bin");
    const auto main = load_words("ngg_merged_gs_main.bin");
    const VertexPrologInfo info = rdna2_vertex_prolog_info(prolog.data(), prolog.size());
    const size_t span = rdna2_recompile_code_span(main.data(), main.size());
    if (!info.valid || !span) return {};
    std::vector<uint32_t> linked(prolog.begin(), prolog.begin() + info.prefix_dwords);
    linked.insert(linked.end(), main.begin(), main.begin() + span);
    return linked;
}

}   // namespace

TEST(NggSubgroupAbi, MinimalProgramIsAdmittedWithFlagsPrimPos0Layout) {
    const auto facts = analyze(program({}));
    ASSERT_TRUE(facts.ok()) << facts.refusal;
    EXPECT_EQ(facts.layout.words_per_lane, 6u) << "flags + PRIM + POS0";
    EXPECT_EQ(facts.layout.pos1_word, kNggRecordAbsent);
    EXPECT_TRUE(facts.layout.param_targets.empty());
    EXPECT_EQ(facts.layout.prim_channels, 1u);
    EXPECT_EQ(facts.layout.pos0_channels, 0xfu);
}

TEST(NggSubgroupAbi, LayoutOrdersParamTargetsAndPlacesPos1) {
    // POS1.z from v9, then PARAM1 before PARAM0 in program order.
    const std::vector<uint32_t> exports = {0xf8000941u, 0x00000009u, 0xf80000cfu, 0x03020100u,
                                           0xf80000d4u, 0x00090000u, 0xf800021fu, 0x03020100u,
                                           0xf8000203u, 0x00000100u};
    const auto facts = analyze(program({}, true, exports));
    ASSERT_TRUE(facts.ok()) << facts.refusal;
    EXPECT_EQ(facts.layout.pos1_word, 6u);
    EXPECT_EQ(facts.layout.pos1_channels, 4u);
    ASSERT_EQ(facts.layout.param_targets, (std::vector<uint32_t>{32u, 33u}));
    EXPECT_EQ(facts.layout.param_channels, (std::vector<uint32_t>{3u, 0xfu}));
    EXPECT_EQ(facts.layout.first_param_word, 10u);
    EXPECT_EQ(facts.layout.param_word(1), 14u);
    EXPECT_EQ(facts.layout.words_per_lane, 18u);
    EXPECT_EQ(facts.layout.block_words(2), kNggSubgroupHeaderWords + 128u * 18u);
}

TEST(NggSubgroupAbi, LaunchSgprReadsAreRefusedByName) {
    EXPECT_EQ(analyze(program({0xbe940300u})).reason, "ngg-abi-read-s0-s1");   // s_mov s20, s0
    EXPECT_TRUE(analyze(program({0xbe940300u}), 0, true).ok())
        << "a known user-data address makes s0 a launch value";
    EXPECT_EQ(analyze(program({0xbe940302u})).reason, "ngg-abi-read-s2");
    EXPECT_EQ(analyze(program({0xbe940304u})).reason, "ngg-abi-read-s4-s5");
    EXPECT_EQ(analyze(program({0xbe940305u})).reason, "ngg-abi-read-s4-s5");
    EXPECT_EQ(analyze(program({0xbe940306u})).reason, "ngg-abi-read-s6-s7");
    EXPECT_EQ(analyze(program({0xbe940307u})).reason, "ngg-abi-read-s6-s7");
    // Written first, s2 is ordinary scratch.
    EXPECT_TRUE(analyze(program({0xbe820380u, 0xbe940302u})).ok());   // s_mov s2, 0; s_mov s20, s2
}

TEST(NggSubgroupAbi, UserSgprsAboveTheSuppliedRangeAreRefused) {
    const auto code = program({0xbe940309u});   // s_mov_b32 s20, s9
    EXPECT_EQ(analyze(code, 1).reason, "ngg-abi-read-undefined-sgpr");
    EXPECT_TRUE(analyze(code, 2).ok());
}

TEST(NggSubgroupAbi, S3IsAdmittedOnlyWithoutItsGsWaveIdBits) {
    // s_lshr_b32 s20, s3, 16 demands [31:16]; shifting by 24 demands only [31:24].
    EXPECT_EQ(analyze(program({0x90149003u})).reason, "ngg-abi-read-s3-gs-wave-id");
    EXPECT_TRUE(analyze(program({0x90149803u})).ok());
    // s_bfe_u32 s20, s3: wave index [27:24] is admitted, [23:16] is not.
    EXPECT_TRUE(analyze(program({0x9394ff03u, 0x00040018u})).ok());
    EXPECT_EQ(analyze(program({0x9394ff03u, 0x00080010u})).reason, "ngg-abi-read-s3-gs-wave-id");
    // A whole-register copy demands every bit.
    EXPECT_EQ(analyze(program({0xbe940303u})).reason, "ngg-abi-read-s3-gs-wave-id");
    // s_and_b32 with a mask clear of [23:16] is admitted.
    EXPECT_TRUE(analyze(program({0x8714ff03u, 0x0f00ffffu})).ok());
    EXPECT_EQ(analyze(program({0x8714ff03u, 0x00010000u})).reason, "ngg-abi-read-s3-gs-wave-id");
}

TEST(NggSubgroupAbi, ExecMustBeWrittenBeforeAnyVectorInstruction) {
    EXPECT_EQ(analyze(program({}, false)).reason, "ngg-abi-exec-read-before-write");
    EXPECT_TRUE(analyze(program({}, true)).ok());
}

TEST(NggSubgroupAbi, UnsuppliedLaunchVgprsMustBeWrittenOnTheLaneFirst) {
    EXPECT_EQ(analyze(program({0x7e120304u})).reason, "ngg-abi-read-v4");   // v_mov v9, v4
    EXPECT_EQ(analyze(program({0x7e120306u})).reason, "ngg-abi-read-v6-v7");   // v_mov v9, v6
    EXPECT_EQ(analyze(program({0x7e120307u})).reason, "ngg-abi-read-v6-v7");
    EXPECT_TRUE(analyze(program({0x7e080280u, 0x7e120304u})).ok())   // v_mov v4, 0 first
        << "a full-EXEC write defines the register on every lane";
    EXPECT_TRUE(analyze(program({0x7e120305u, 0x7e120308u})).ok()) << "v5 and v8 are supplied";
}

TEST(NggSubgroupAbi, AVgprWrittenUnderNarrowEXECIsDefinedOnlyWhileEXECStaysNarrow) {
    // s_mov_b64 exec, 1; v_mov v6, 0; v_mov v9, v6: the read sees the lanes the write covered.
    EXPECT_TRUE(analyze(program({0xbefe0481u, 0x7e0c0280u, 0x7e120306u})).ok());
    // Widening EXEC again exposes lanes whose v6 was never written.
    EXPECT_EQ(analyze(program({0xbefe0481u, 0x7e0c0280u, kExecAllOnes, 0x7e120306u})).reason,
              "ngg-abi-read-v6-v7");
    // Narrowing with s_and_b64 exec, exec, s[20:21] keeps the definition.
    EXPECT_TRUE(
        analyze(program({0xbefe0481u, 0x7e0c0280u, 0xbe940480u, 0x87fe147eu, 0x7e120306u})).ok());
}

// #3135 P6: the compiler's EXEC save/restore idiom. Kena's indexed producer writes v7 under the ES
// EXEC, saves EXEC with s_mov_b64 s[36:37], exec, changes EXEC, restores it with s_mov_b64 exec,
// s[36:37] and then stores v7 to LDS. The restored EXEC is the saved one, so v7 is defined on every
// lane the store reads. Each control is one edit away and must stay refused.
TEST(NggSubgroupAbi, RestoringASavedExecBringsBackItsDefinitions) {
    constexpr uint32_t kExecOne = 0xbefe0481u;   // s_mov_b64 exec, 1 (a narrow launch EXEC)
    constexpr uint32_t kWriteV7 = 0x7e0e0280u;   // v_mov_b32 v7, 0
    constexpr uint32_t kReadV7 = 0x7e120307u;   // v_mov_b32 v9, v7
    constexpr uint32_t kSave = 0xbea4047eu;   // s_mov_b64 s[36:37], exec
    constexpr uint32_t kRestore = 0xbefe0424u;   // s_mov_b64 exec, s[36:37]
    constexpr uint32_t kClobber = 0xbea40480u;   // s_mov_b64 s[36:37], 0
    constexpr uint32_t kZero20 = 0xbe940480u;   // s_mov_b64 s[20:21], 0
    constexpr uint32_t kNarrow = 0x87fe147eu;   // s_and_b64 exec, exec, s[20:21]
    constexpr uint32_t kAndSave = 0xbea42414u;   // s_and_saveexec_b64 s[36:37], s[20:21]
    EXPECT_TRUE(analyze(program({kExecOne, kWriteV7, kSave, kExecAllOnes, kRestore, kReadV7})).ok())
        << "the restored EXEC is the one v7 was written under";
    EXPECT_EQ(analyze(program({kExecOne, kWriteV7, kSave, kExecAllOnes, kReadV7})).reason,
              "ngg-abi-read-v6-v7")
        << "control: without the restore the widened EXEC exposes unwritten lanes";
    EXPECT_EQ(
        analyze(program({kExecOne, kWriteV7, kSave, kClobber, kExecAllOnes, kRestore, kReadV7}))
            .reason,
        "ngg-abi-read-v6-v7")
        << "control: the saved pair was overwritten, so the restore is not that EXEC";
    EXPECT_TRUE(analyze(program({kExecOne, kSave, kWriteV7, kExecAllOnes, kRestore, kReadV7})).ok())
        << "a write after the save, under the unchanged EXEC, covers the saved lanes too";
    EXPECT_EQ(analyze(program({kExecOne, kSave, kZero20, kNarrow, kWriteV7, kExecAllOnes, kRestore,
                               kReadV7}))
                  .reason,
              "ngg-abi-read-v6-v7")
        << "control: written under a narrower EXEC than the one saved";
    EXPECT_TRUE(
        analyze(program({kExecOne, kWriteV7, kZero20, kAndSave, kExecAllOnes, kRestore, kReadV7}))
            .ok())
        << "a SAVEEXEC saves the EXEC from before it narrows";
    // s_cbranch_execz +1 skips the save on one path (s[36:37] is written on both, so the restore
    // reads a defined pair): the join has no common saved EXEC.
    EXPECT_EQ(analyze(program({kExecOne, kWriteV7, kClobber, 0xbf880001u, kSave, kExecAllOnes,
                               kRestore, kReadV7}))
                  .reason,
              "ngg-abi-read-v6-v7")
        << "control: saved on one path only";
}

TEST(NggSubgroupAbi, MemoryEffectsAreRefused) {
    // buffer_store_dword v9, off, s[8:11], 0 versus buffer_load_dword v10 with the same operands.
    EXPECT_EQ(analyze(program({0xe0700000u, 0x80020900u}), 4).reason, "ngg-side-effect");
    EXPECT_TRUE(analyze(program({0xe0300000u, 0x80020a00u}), 4).ok());
    // ds_add_u32 v0, v1: LDS atomics stay inside the subgroup; the GDS form is refused.
    EXPECT_TRUE(analyze(program({0xd8000000u, 0x00000100u})).ok());
    EXPECT_EQ(analyze(program({0xd8020000u, 0x00000100u})).reason, "ngg-side-effect");
}

TEST(NggSubgroupAbi, OnlyGsAllocReqWithAProvenM0IsAdmitted) {
    const uint32_t movk_m0 = 0xb07c3005u;   // s_movk_i32 m0, 0x3005
    EXPECT_TRUE(analyze(program({movk_m0, 0xbf900009u})).ok());
    EXPECT_EQ(analyze(program({movk_m0, 0xbf900003u})).reason, "ngg-sendmsg-unsupported");
    EXPECT_EQ(analyze(program({0xbf900009u})).reason, "ngg-sendmsg-m0-unproven");
    // Written on only one of two paths: s_cbranch_execz +1 skips the write.
    EXPECT_EQ(analyze(program({0xbf880001u, movk_m0, 0xbf900009u})).reason,
              "ngg-sendmsg-m0-unproven");
    // s_load_dword m0, s[8:9], 0: a memory word is not a proven scalar ALU value.
    EXPECT_EQ(analyze(program({0xf4001f04u, 0xfa000000u, 0xbf900009u}), 2).reason,
              "ngg-sendmsg-m0-unproven");
}

TEST(NggSubgroupAbi, ExportShapesOutsideTheRecordAreRefused) {
    const auto with = [](std::initializer_list<uint32_t> extra) {
        std::vector<uint32_t> exports(kExports.begin(), kExports.end());
        exports.insert(exports.end(), extra);
        return exports;
    };
    EXPECT_EQ(analyze(program({}, true, with({0xf800040fu, 0x00000100u}))).reason,
              "ngg-export-compressed");   // COMPR MRT0
    EXPECT_EQ(analyze(program({}, true, with({0xf800000fu, 0x03020100u}))).reason,
              "ngg-export-target-unsupported");   // MRT0
    EXPECT_EQ(analyze(program({}, true, with({0xf80000efu, 0x03020100u}))).reason,
              "ngg-export-target-unsupported");   // POS2
    EXPECT_EQ(analyze(program({}, true, with({0xf80000cfu, 0x03020100u}))).reason,
              "ngg-export-duplicate-target");
    // POS1.z (the layer) is admitted; POS1.x (point size) and POS1.w (viewport) are not.
    EXPECT_TRUE(analyze(program({}, true, with({0xf80000d4u, 0x00090000u}))).ok());
    EXPECT_EQ(analyze(program({}, true, with({0xf80000d1u, 0x00000009u}))).reason,
              "ngg-export-pos1-channels");
    EXPECT_EQ(analyze(program({}, true, with({0xf80000d8u, 0x09000000u}))).reason,
              "ngg-export-pos1-channels");
    EXPECT_EQ(
        analyze(program({}, true, {0xf8000943u, 0x00000009u, 0xf80000cfu, 0x03020100u})).reason,
        "ngg-export-prim-channels");
    EXPECT_EQ(analyze(program({}, true, {0xf80000cfu, 0x03020100u})).reason,
              "ngg-export-missing-prim");
    EXPECT_EQ(analyze(program({}, true, {0xf8000941u, 0x00000009u})).reason,
              "ngg-export-missing-pos0");
}

TEST(NggSubgroupAbi, AnExportInsideALoopIsRefused) {
    // exp param0 v0..v3; s_cbranch_scc0 back to it.
    const std::vector<uint32_t> looped = {kExecAllOnes, 0x7e120280u, 0xf800020fu, 0x03020100u,
                                          0xbf84fffdu,  0xf8000941u, 0x00000009u, 0xf80000cfu,
                                          0x03020100u,  kEnd};
    EXPECT_EQ(analyze(looped).reason, "ngg-export-in-cycle");
    auto forward = looped;
    forward[4] = 0xbf880000u;   // a forward branch (execz +0) in its place
    forward.insert(forward.end() - 1, {0xb07c3005u, 0xbf900009u});   // m0; GS_ALLOC_REQ
    EXPECT_TRUE(analyze(forward).ok());
}

TEST(NggSubgroupAbi, IndirectControlAndUnresolvedTargetsAreRefused) {
    EXPECT_EQ(analyze(program({0xbe802008u}), 2).reason, "ngg-abi-control-transfer");   // s_setpc
    EXPECT_EQ(analyze(program({0xbf820010u})).reason, "ngg-abi-cfg-unresolved");   // s_branch
}

TEST(NggSubgroupAbi, CapturedKenaChainNeedsItsEightUserSgprs) {
    const auto linked = kena_linked();
    ASSERT_EQ(linked.size(), 514u) << "the 101-dword prolog prefix plus the 413-dword main";
    const auto facts = analyze(linked, 8);
    ASSERT_TRUE(facts.ok()) << facts.refusal;
    EXPECT_EQ(facts.layout.words_per_lane, 14u) << "flags, PRIM, POS0, POS1, PARAM0";
    EXPECT_EQ(facts.layout.pos1_word, 6u);
    EXPECT_EQ(facts.layout.pos1_channels, 4u) << "the layer is the only POS1 channel";
    EXPECT_EQ(facts.layout.param_targets, (std::vector<uint32_t>{32u}));
    EXPECT_EQ(facts.alloc_request_pcs.size(), 1u);
    // RSRC2 says USER_SGPR=0, but the ES part reads s8..s15 (open question 1 on #3135).
    EXPECT_EQ(analyze(linked, 7).reason, "ngg-abi-read-undefined-sgpr");
    EXPECT_EQ(analyze(linked, 0).reason, "ngg-abi-read-undefined-sgpr");
}

TEST(NggSubgroupAbi, AProgramWithoutGsAllocReqIsRefused) {
    std::vector<uint32_t> code = program({});
    // Drop the s_movk m0 / s_sendmsg pair program() adds before the exports.
    code.erase(code.begin() + 2, code.begin() + 4);
    EXPECT_EQ(analyze(code).reason, "ngg-sendmsg-missing");
    EXPECT_TRUE(analyze(program({})).ok());
}

// Every lowered 64-bit vector source names two registers; reading v[3:4] reads launch v4.
TEST(NggSubgroupAbi, SixtyFourBitVectorSourcesChargeTheirHighRegister) {
    // v_cmp_eq_u64 vcc, 0, v[3:4] (VOPC e32) and its v[2:3] control.
    EXPECT_EQ(analyze(program({0x7dc40680u})).reason, "ngg-abi-read-v4");
    EXPECT_TRUE(analyze(program({0x7dc40480u})).ok());
    // v_cmp_eq_u64 s[20:21], v[3:4], 0 (VOP3-encoded VOPC).
    EXPECT_EQ(analyze(program({0xd4e20014u, 0x00010103u})).reason, "ngg-abi-read-v4");
    EXPECT_TRUE(analyze(program({0xd4e20014u, 0x00010102u})).ok());
    // v_cvt_f32_f64 v9, v[5:6] (VOP1) and its VOP3 form read v6.
    EXPECT_EQ(analyze(program({0x7e121f05u})).reason, "ngg-abi-read-v6-v7");
    EXPECT_TRUE(analyze(program({0x7e121f02u})).ok());
    // The unused VOP3 source fields are inline 0: the decoder reports an encoded 0 as a read of s0.
    EXPECT_EQ(analyze(program({0xd58f0009u, 0x02010105u})).reason, "ngg-abi-read-v6-v7");
    EXPECT_TRUE(analyze(program({0xd58f0009u, 0x02010102u})).ok());
    // v_qsad_pk_u16_u8 v[10:11], v[3:4], 0, v[0:1].
    EXPECT_EQ(analyze(program({0xd572000au, 0x04010103u})).reason, "ngg-abi-read-v4");
    EXPECT_TRUE(analyze(program({0xd572000au, 0x04010100u})).ok());
}

TEST(NggSubgroupAbi, AnUnclassifiedVectorWidthIsRefused) {
    // VOP3 opcode 0x161 has no gfx10.3 entry; 0x16f (v_div_fmas_f32) is classified.
    EXPECT_EQ(analyze(program({0xd5610009u, 0x040a0300u})).reason,
              "ngg-abi-unclassified-vector-width");
}

TEST(NggSubgroupAbi, AMimgAddressIsChargedThroughV7) {
    // image_load v[10:13], v0, s[8:15] dmask 0xf: a 2D address could run past v3.
    const std::initializer_list<uint32_t> load = {0xf0000f08u, 0x00020a00u};
    EXPECT_EQ(analyze(program(load), 8).reason, "ngg-abi-read-v4");
    // v_mov v4, 0; v_mov v6, 0; v_mov v7, 0 first.
    EXPECT_TRUE(
        analyze(program({0x7e080280u, 0x7e0c0280u, 0x7e0e0280u, 0xf0000f08u, 0x00020a00u}), 8)
            .ok());
}

TEST(NggSubgroupAbi, ImplicitSccAndVccReadsBeforeWriteAreRefused) {
    // s_cbranch_scc0 +0 before any SCC write; control: s_cmp_eq_u32 0, 0 first.
    EXPECT_EQ(analyze(program({0xbf840000u})).reason, "ngg-abi-read-undefined-scc");
    EXPECT_TRUE(analyze(program({0xbf068080u, 0xbf840000u})).ok());
    // v_cndmask_b32_sdwa v9, v0, v1 reads VCC; control: v_cmp_eq_u32 vcc, 0, v0 first.
    const std::initializer_list<uint32_t> sdwa_cndmask = {0x021202f9u, 0x06060600u};
    EXPECT_EQ(analyze(program(sdwa_cndmask)).reason, "ngg-abi-read-undefined-vcc");
    EXPECT_TRUE(analyze(program({0x7d840080u, 0x021202f9u, 0x06060600u})).ok());
    // v_div_fmas_f32 v9, v0, v1, v2 reads VCC.
    EXPECT_EQ(analyze(program({0xd56f0009u, 0x040a0300u})).reason, "ngg-abi-read-undefined-vcc");
    EXPECT_TRUE(analyze(program({0x7d840080u, 0xd56f0009u, 0x040a0300u})).ok());
}

// v_xor3_b32 (VOP3 0x178) reads three 32-bit VGPRs. It sat in the unclassified 0x178-0x17f band,
// so a program using it was refused for an unknown source width.
TEST(NggSubgroupAbi, Xor3ReadsThirtyTwoBitSources) {
    // v_xor3_b32 v10, v0, v1, v2
    const auto facts = analyze(program({0xd578000au, 0x040a0300u}));
    EXPECT_TRUE(facts.ok()) << facts.refusal;
    // The neighbour 0x179 is still unclassified: the band was widened by one opcode, not dropped.
    EXPECT_EQ(analyze(program({0xd579000au, 0x040a0300u})).reason,
              "ngg-abi-unclassified-vector-width");
}

// A VOP3 encoding always carries three source fields, and the ones the opcode does not read decode
// as s0. Without a known user-data address s0 is not a launch value, so counting that phantom read
// refused programs that never touch s0.
TEST(NggSubgroupAbi, UnusedVop3SourceFieldsAreNotReads) {
    EXPECT_EQ(vop3_architectural_source_count(0x181), 1u) << "VOP3-encoded VOP1";
    EXPECT_EQ(vop3_architectural_source_count(0x125), 2u) << "VOP3-encoded VOP2";
    EXPECT_EQ(vop3_architectural_source_count(0x101), 3u) << "cndmask reads its mask from SRC2";
    EXPECT_EQ(vop3_architectural_source_count(0x14b), 3u) << "v_fma_f32";
    // v_mov_b32_e64 v10, v0: SRC1 and SRC2 are zero fields, i.e. s0.
    EXPECT_TRUE(analyze(program({0xd581000au, 0x00000100u})).ok());
    // v_add_nc_u32_e64 v10, v0, v1: SRC2 is a zero field.
    EXPECT_TRUE(analyze(program({0xd525000au, 0x00020300u})).ok());
    // v_cndmask_b32_e64 v10, v0, v1, s0: the mask is a real read of s0.
    EXPECT_EQ(analyze(program({0xd501000au, 0x00020300u})).reason, "ngg-abi-read-s0-s1");
    // A real read of s0 is still counted: v_add_nc_u32_e64 v10, s0, v1 reads it through SRC0.
    EXPECT_EQ(analyze(program({0xd525000au, 0x00020200u})).reason, "ngg-abi-read-s0-s1");
}
