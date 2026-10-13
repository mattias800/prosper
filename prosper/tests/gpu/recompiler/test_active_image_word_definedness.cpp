// #5026: a real image result defines each full dword in active lanes. Comparing that result
// under the same EXEC must not observe missing inactive lanes. Widening EXEC, atomic results
// that preserve the old destination, and unresolved descriptors remain separate obligations.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace prosper::gpu;
namespace {
constexpr uint32_t kEnd = 0xbf810000u;
constexpr uint32_t kCmpx = 0x7da40480u; // CMPX_EQ_U32 0,v2
constexpr uint32_t kCompare = 0x7c200d06u; // CMP_EQ_U32 v6,v6
constexpr uint32_t kSampleLz = 0xf09c0108u;
constexpr uint32_t kImageWord1 = 0x00820600u; // VADDR v0, VDATA v6, SRSRC s8, SSAMP s16

struct Harness {
    SpirvCompute b;
    RegState rs;
    ShaderResourceTable table;
    Harness() {
        ShaderResource texture;
        texture.cls = ResourceClass::Texture;
        texture.binding = 4;
        texture.img_dim = 1;
        texture.width = texture.height = 8;
        texture.format = DataFormat::Float32;
        texture.num_components = 4;
        texture.sgpr_base = 8;
        texture.sampler_sgpr_base = 16;
        table.resources.push_back(texture);
        b.wave_size = 64;
        b.begin_fragment(&table);
        rs.exec = b.btrue();
        rs.vcc = rs.scc = b.bfalse();
        rs.vreg[0] = rs.vreg[1] = rs.vreg[2] = b.uconst(0);
    }
    bool emit(const Rdna2Inst& in) {
        prepare_saved_mask_word_write(b, rs, in);
        const auto before = snapshot_saved_b64_masks(b, rs, in);
        bool ok = true;
        const bool handled = emit_alu(b, rs, in, ok, true, nullptr, true, &table);
        if (!handled || !ok) return false;
        record_scalar_write(rs, in, false, before);
        retain_saved_mask_words(b, rs, in);
        return true;
    }
    void enter_partial_host_region() {
        // A real enclosing conditional denies the physical full-participation initializer.
        // It cannot prevent a subsequently emitted comparison from defining its current Bool.
        const uint32_t body = b.id(), merge = b.id();
        b.put(b.code, Op_SelectionMerge, {merge, 0});
        b.put(b.code, Op_BranchConditional, {b.btrue(), body, merge});
        b.emit_label(body);
    }
};

std::vector<Rdna2Inst> program(uint32_t image = kSampleLz, uint32_t reader = kCompare,
                               uint32_t descriptor = kImageWord1) {
    const std::vector<uint32_t> words{kCmpx, image, descriptor, reader, kEnd};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    return ins;
}

std::vector<uint32_t> guarded_loop(uint32_t writer, uint32_t extension, bool clean) {
    // Save EXEC, narrow it with a genuine comparison, and guard a counted loop whose header
    // reads v6. The body must define v6 again for every subsequent header iteration. The
    // CNDMASK/EXP makes the header predicate an actual observable output, not a dead compare.
    return {0x7e000280u, 0x7e020280u,
            0x7e040280u, 0xbe94047eu,
            kCmpx,       0xbf88000au,
            writer,      extension,
            0xbe8a0380u, 0x7d040c87u,
            0x021200f2u, 0xbf0a820au,
            0xbf840003u, clean ? 0x7e0c0287u : 0x7e0c0307u,
            0x800a810au, 0xbf82fff9u,
            0xbefe0414u, 0x7e1002f2u,
            0xf800180fu, 0x09090909u,
            kEnd};
}
}   // namespace

TEST(ActiveImageWordDefinedness, GuardedLoopAcceptsAnInitializedBackedge) {
    Harness h;
    for (const auto [writer, extension] :
         {std::pair{kSampleLz, kImageWord1}, std::pair{0x7e0c0287u, 0xbf800000u},
          std::pair{0x7e0c02f9u, 0x00861680u}}) {
        SCOPED_TRACE(writer);
        const auto code = guarded_loop(writer, extension, true);
        const auto module = recompile_fragment(code.data(), code.size(), &h.table);
        EXPECT_FALSE(module.empty()) << "each iteration replaces v6 with genuine active-lane data";
    }
}

TEST(ActiveImageWordDefinedness, GuardedLoopRejectsAnUninitializedBackedge) {
    Harness h;
    for (const auto [writer, extension] :
         {std::pair{kSampleLz, kImageWord1}, std::pair{0x7e0c0287u, 0xbf800000u},
          std::pair{0x7e0c02f9u, 0x00861680u}}) {
        SCOPED_TRACE(writer);
        const auto code = guarded_loop(writer, extension, false);
        constexpr uint64_t address = 0x10001000u;
        const auto module =
            recompile_fragment(code.data(), code.size(), &h.table, nullptr, UINT32_MAX, nullptr,
                               false, {RecompileDiagnosticStage::Fragment, address});
        EXPECT_TRUE(module.empty()) << "the preheader's EXEC owner cannot authenticate later v7";
        EXPECT_NE(last_terminal_reject_reason(address).find("loop-carried v6"), std::string::npos);
    }
}

TEST(ActiveImageWordDefinedness, NarrowerCleanBackedgePreservesTheRequiredWiderDomain) {
    Harness h;
    const std::vector<uint32_t> words{kCmpx, kSampleLz, kImageWord1, kCompare, 0x7e0c0287u};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    seed_vector_word_definedness(h.rs, ins, false, 64);
    ASSERT_TRUE(h.emit(ins[0]));
    const uint32_t owner = h.rs.exec;
    ASSERT_TRUE(h.emit(ins[1]));
    LoopBlanketCheck check(h.rs, std::set<int>{}, ins[3].pc);
    mark_vector_loop_carried(h.rs, ins, ins[3].pc, ins[3].pc + 1, check.carried, check.confined);
    auto exits = loop_check_marks(h.rs);
    ASSERT_TRUE(fresh_mask_sources_defined(h.b, h.rs, ins[2]));
    ASSERT_TRUE(h.emit(ins[0]));   // a second genuine CMPX narrows the original owner
    ASSERT_NE(h.rs.exec, owner);
    ASSERT_TRUE(h.emit(ins[3]));
    EXPECT_EQ(h.rs.vreg_source_marks.at(6).outside_exec, owner)
        << "unwritten lanes inside the wider owner retain their genuine carried value";
    EXPECT_TRUE(check.holds(h.b, h.rs, exits));
}

TEST(ActiveImageWordDefinedness, NarrowerPoisonedBackedgeCannotReuseItsPreheaderOwner) {
    Harness h;
    const std::vector<uint32_t> words{kCmpx, kSampleLz, kImageWord1, kCompare, 0x7e0c0307u};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    seed_vector_word_definedness(h.rs, ins, false, 64);
    ASSERT_TRUE(h.emit(ins[0]));
    ASSERT_TRUE(h.emit(ins[1]));
    LoopBlanketCheck check(h.rs, std::set<int>{}, ins[3].pc);
    mark_vector_loop_carried(h.rs, ins, ins[3].pc, ins[3].pc + 1, check.carried, check.confined);
    auto exits = loop_check_marks(h.rs);
    ASSERT_TRUE(fresh_mask_sources_defined(h.b, h.rs, ins[2]));
    ASSERT_TRUE(h.emit(ins[0]));
    ASSERT_TRUE(h.emit(ins[3]));
    EXPECT_EQ(h.rs.vreg_source_marks.at(6).outside_exec, 0u);
    EXPECT_FALSE(check.holds(h.b, h.rs, exits));
    EXPECT_EQ(exits.vectors.at(6).outside_exec, 0u)
        << "an exit cannot retain a failed loop-header obligation";
}

TEST(ActiveImageWordDefinedness, CompletedSampleAndCompareUseTheSameActualExec) {
    Harness h;
    const auto ins = program();
    ASSERT_EQ(ins.size(), 4u);
    ASSERT_EQ(ins[1].fmt, Rdna2Format::MIMG);
    ASSERT_EQ(ins[1].opcode, 0x27u);
    ASSERT_EQ(ins[1].dst.value, 6);
    ASSERT_EQ(rdna2_vgpr_write_count(ins[1]), 1u);
    seed_vector_word_definedness(h.rs, ins, false, 64);
    ASSERT_TRUE(h.emit(ins[0]));
    ASSERT_TRUE(h.rs.exec_narrowed);
    ASSERT_TRUE(h.emit(ins[1]));
    ASSERT_TRUE(h.rs.vreg.contains(6));
    ASSERT_TRUE(h.rs.vreg_source_marks.contains(6));
    EXPECT_TRUE(h.rs.vreg_source_marks.at(6).hard) << "inactive missing words must remain missing";
    EXPECT_TRUE(fresh_mask_sources_defined(h.b, h.rs, ins[2]));
    EXPECT_TRUE(h.emit(ins[2]));
}

TEST(ActiveImageWordDefinedness, RestoringExecCannotExposeAnUnwrittenImageLane) {
    Harness h;
    const auto ins = program();
    seed_vector_word_definedness(h.rs, ins, false, 64);
    const uint32_t entry = h.rs.exec;
    ASSERT_TRUE(h.emit(ins[0]));
    ASSERT_TRUE(h.emit(ins[1]));
    h.rs.exec = entry;
    h.rs.exec_narrowed = false;
    EXPECT_FALSE(fresh_mask_sources_defined(h.b, h.rs, ins[2]));
    EXPECT_FALSE(h.emit(ins[2]));
}

TEST(ActiveImageWordDefinedness, ADescriptorRefusalDoesNotDefineAnImageResult) {
    Harness h;
    h.table.resources.clear();
    const auto ins = program();
    seed_vector_word_definedness(h.rs, ins, false, 64);
    ASSERT_TRUE(h.emit(ins[0]));
    EXPECT_FALSE(h.emit(ins[1]));
    EXPECT_FALSE(fresh_mask_sources_defined(h.b, h.rs, ins[2]));
}

TEST(ActiveImageWordDefinedness, ACompleteDwordSdwaWriterDefinesEveryActiveBit) {
    Harness h;
    const std::vector<uint32_t> words{kCmpx, 0x060c00f9u, 0x268606f3u, kCompare, kEnd};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    ASSERT_EQ(ins.size(), 4u);
    ASSERT_TRUE(ins[1].has_sdwa);
    ASSERT_EQ(ins[1].sdwa_dst_sel, 6u);
    seed_vector_word_definedness(h.rs, ins, false, 64);
    ASSERT_TRUE(h.emit(ins[0]));
    ASSERT_TRUE(h.emit(ins[1]));
    EXPECT_TRUE(fresh_mask_sources_defined(h.b, h.rs, ins[2]));
    EXPECT_TRUE(h.emit(ins[2]));
}

TEST(ActiveImageWordDefinedness, APreservingSdwaSiblingKeepsItsMissingActiveBits) {
    Harness h;
    const std::vector<uint32_t> words{kCmpx, 0x7e0c02f9u, 0x00851480u, kCompare, kEnd};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    ASSERT_EQ(ins.size(), 4u);
    ASSERT_TRUE(ins[1].has_sdwa);
    ASSERT_EQ(ins[1].sdwa_dst_sel, 4u);
    ASSERT_EQ(ins[1].sdwa_dst_unused, 2u);
    seed_vector_word_definedness(h.rs, ins, false, 64);
    ASSERT_TRUE(h.emit(ins[0]));
    ASSERT_TRUE(h.emit(ins[1]));
    EXPECT_FALSE(fresh_mask_sources_defined(h.b, h.rs, ins[2]));
    EXPECT_FALSE(h.emit(ins[2]));
}

TEST(ActiveImageWordDefinedness, TheDwordSdwaMoveSiblingDefinesItsWholeDestination) {
    Harness h;
    const std::vector<uint32_t> words{kCmpx, 0x7e0c02f9u, 0x00861680u, kCompare, kEnd};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    ASSERT_EQ(ins.size(), 4u);
    ASSERT_TRUE(ins[1].has_sdwa);
    ASSERT_EQ(ins[1].sdwa_dst_sel, 6u);
    ASSERT_EQ((words[2] >> 11) & 3u, 2u)
        << "UNUSED_PRESERVE is immaterial when the destination is DWORD";
    seed_vector_word_definedness(h.rs, ins, false, 64);
    ASSERT_TRUE(h.emit(ins[0]));
    ASSERT_TRUE(h.emit(ins[1]));
    EXPECT_TRUE(fresh_mask_sources_defined(h.b, h.rs, ins[2]));
    EXPECT_TRUE(h.emit(ins[2]));
}

TEST(ActiveImageWordDefinedness, CndmaskReadsTheProducedBoolWithoutInventingItsPhysicalWords) {
    Harness h;
    const std::vector<uint32_t> words{kCmpx,       0x7d8400f9u, 0x0686a080u, 0xd5010006u,
                                      0x00810280u, kCompare,    0x7e100220u, kEnd};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    ASSERT_EQ(ins.size(), 6u);
    ASSERT_EQ(ins[1].dst.value, 32);
    ASSERT_EQ(ins[2].src[2].value, 32);
    h.enter_partial_host_region();
    initialize_saved_mask_word_proof(h.b, ins, words.data(), words.size(), 0, &h.rs);
    seed_vector_word_definedness(h.rs, ins, false, 64);
    ASSERT_TRUE(h.emit(ins[0]));
    ASSERT_TRUE(h.emit(ins[1]));
    ASSERT_TRUE(h.rs.sreg_bool.contains(32));
    ASSERT_TRUE(h.rs.unavailable_saved_mask_words.contains(32))
        << "the logical predicate does not supply an unproved physical ballot word";
    ASSERT_TRUE(h.emit(ins[2]));
    EXPECT_TRUE(fresh_mask_sources_defined(h.b, h.rs, ins[3]));
    EXPECT_TRUE(h.emit(ins[3]));
}

TEST(ActiveImageWordDefinedness, CndmaskCannotUseAnUnavailableOrMarkedMaskAsFreshProof) {
    for (int control = 0; control < 3; ++control) {
        Harness h;
        const std::vector<uint32_t> words{kCmpx,       0x7d8400f9u, 0x0686a080u, 0xd5010006u,
                                          0x00810280u, kCompare,    0x7e100220u, kEnd};
        std::vector<Rdna2Inst> ins;
        rdna2_walk(words.data(), words.size(), ins);
        h.enter_partial_host_region();
        initialize_saved_mask_word_proof(h.b, ins, words.data(), words.size(), 0, &h.rs);
        seed_vector_word_definedness(h.rs, ins, false, 64);
        ASSERT_TRUE(h.emit(ins[0]));
        ASSERT_TRUE(h.emit(ins[1]));
        if (control == 0) {
            h.rs.sreg_bool.erase(32);
            EXPECT_FALSE(h.emit(ins[2]));
        } else {
            h.rs.sreg_merge_placeholder.insert(control == 1 ? 32 : 33);
            ASSERT_TRUE(h.emit(ins[2]));
        }
        EXPECT_FALSE(fresh_mask_sources_defined(h.b, h.rs, ins[3])) << "control " << control;
    }
}

TEST(ActiveImageWordDefinedness, NumericOverwriteCannotResurrectTheOldCndmaskBool) {
    Harness h;
    const std::vector<uint32_t> words{kCmpx,       0x7d8400f9u, 0x0686a080u, 0xbea00380u,
                                      0xd5010006u, 0x00810280u, kCompare,    kEnd};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    seed_vector_word_definedness(h.rs, ins, false, 64);
    ASSERT_TRUE(h.emit(ins[0]));
    ASSERT_TRUE(h.emit(ins[1]));
    ASSERT_TRUE(h.rs.sreg_bool.contains(32));
    ASSERT_TRUE(h.emit(ins[2]));
    EXPECT_FALSE(h.rs.sreg_bool.contains(32));
    EXPECT_FALSE(h.emit(ins[3])) << "a numeric low word with no sibling is not a Bool mask";
    EXPECT_FALSE(fresh_mask_sources_defined(h.b, h.rs, ins[4]));
}

TEST(ActiveImageWordDefinedness, ARelayedMissingWordCannotProduceACndmaskPredicate) {
    Harness h;
    const std::vector<uint32_t> words{kCmpx,       0x7e100214u, 0x7e400508u,
                                      0x7d8400f9u, 0x0686a820u, kEnd};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    seed_vector_word_definedness(h.rs, ins, false, 64);
    ASSERT_TRUE(h.emit(ins[0]));
    ASSERT_TRUE(h.emit(ins[1]));
    ASSERT_TRUE(h.emit(ins[2]));
    EXPECT_TRUE(h.rs.sreg_merge_placeholder.contains(32));
    EXPECT_FALSE(h.emit(ins[3]));
    EXPECT_FALSE(h.rs.sreg_bool.contains(40));
}

TEST(ActiveImageWordDefinedness, AndSaveexecKeepsTheWritersActiveLaneProof) {
    Harness h;
    const std::vector<uint32_t> words{kCmpx, 0x7e0c0280u, 0x7d840080u, 0xbea8246au, kCompare, kEnd};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    ASSERT_EQ(ins.size(), 6u);
    seed_vector_word_definedness(h.rs, ins, false, 64);
    for (size_t i = 0; i < 4; ++i) ASSERT_TRUE(h.emit(ins[i])) << i;
    EXPECT_TRUE(fresh_mask_sources_defined(h.b, h.rs, ins[4]));
    EXPECT_TRUE(h.emit(ins[4]));
}

TEST(ActiveImageWordDefinedness, Andn2AndContainedOrRemainWithinTheSavedWritersExec) {
    Harness h;
    const std::vector<uint32_t> words{kCmpx,       0x7e0c0280u, 0x7d840080u, 0xbea8246au,
                                      0x8afe7e28u, 0x88fe7e28u, kCompare,    kEnd};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    ASSERT_EQ(ins.size(), 8u);
    seed_vector_word_definedness(h.rs, ins, false, 64);
    for (size_t i = 0; i < 6; ++i) ASSERT_TRUE(h.emit(ins[i])) << i;
    EXPECT_TRUE(fresh_mask_sources_defined(h.b, h.rs, ins[6]));
    EXPECT_TRUE(h.emit(ins[6]));
}

TEST(ActiveImageWordDefinedness, OrSaveexecWideningDoesNotBorrowTheOldWriterProof) {
    Harness h;
    const std::vector<uint32_t> words{kCmpx, 0x7e0c0280u, 0xbea825c1u, kCompare, kEnd};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    seed_vector_word_definedness(h.rs, ins, false, 64);
    for (size_t i = 0; i < 3; ++i) ASSERT_TRUE(h.emit(ins[i])) << i;
    EXPECT_FALSE(fresh_mask_sources_defined(h.b, h.rs, ins[3]));
    EXPECT_FALSE(h.emit(ins[3]));
}

TEST(ActiveImageWordDefinedness, PartialAndAtomicResultsCannotConfineMissingWords) {
    for (const auto [word0, word1] :
         {std::pair{kSampleLz | 0x00010000u, kImageWord1},
          std::pair{kSampleLz | 0x00020000u, kImageWord1},
          std::pair{kSampleLz, kImageWord1 | 0x80000000u}, std::pair{0xf0440128u, kImageWord1}}) {
        Harness h;
        const auto ins = program(word0, kCompare, word1);
        seed_vector_word_definedness(h.rs, ins, false, 64);
        ASSERT_TRUE(h.emit(ins[0]));
        // Effects-only negative: do not manufacture a successful unsupported image emitter.
        record_vector_word_marks(h.rs, ins[1], scalar_source_marks(h.rs, ins[1]));
        EXPECT_FALSE(fresh_mask_sources_defined(h.b, h.rs, ins[2])) << std::hex << word0;
    }
}
