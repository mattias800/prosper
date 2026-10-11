// #4921: a completed counted loop may retain an original Bool-only VCC copy proof
// for its exact postlude. An arbitrary slice must not inherit that authority or
// recover physical words for lanes that no longer participate.
#include "fragment_predicate_oracle.hpp"
#include "fixtures/fragment_compare_kill.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

namespace prosper::gpu {
namespace {
constexpr uint32_t end = 0xbf810000u;
constexpr uint64_t mask = 0x80000001abcdef01ull;
constexpr uint32_t mov(uint32_t dst, uint32_t source, bool pair = false) {
    return 0xbe800000u | (dst << 16) | ((pair ? 4u : 3u) << 8) | source;
}
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    return ins;
}
void counted(std::vector<uint32_t>& code, uint32_t iterations = 3) {
    // s2 = 0; while (s2 < iterations) ++s2. The SCC exit and unconditional
    // backedge deliberately select the real counted-loop emitter.
    code.insert(code.end(), {mov(2, 128), 0xbf0a0002u | ((128u + iterations) << 8), 0xbf840002u,
                             0x80028102u, 0xbf82fffcu});
}
std::vector<uint32_t> program(const std::vector<uint32_t>& tail = {}, bool callback = false,
                              uint32_t iterations = 3) {
    std::vector<uint32_t> code{mov(30, 106)};
    if (callback) code.insert(code.end(), {0xf800080fu, 0x00000000u});
    counted(code, iterations);
    code.insert(code.end(), {mov(106, 20, true), 0xd5010000u, 129u | (137u << 9) | (106u << 18)});
    code.insert(code.end(), tail.begin(), tail.end());
    code.push_back(end);
    return code;
}
size_t ballots(const SpirvCompute& b) {
    size_t count = 0;
    for (size_t at = 0; at < b.code.size();) {
        const size_t width = b.code[at] >> 16;
        if (!width || width > b.code.size() - at) return SIZE_MAX;
        count += (b.code[at] & 0xffffu) == 339u;
        at += width;
    }
    return count;
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    explicit Emitter(bool compute = false) {
        if (compute) {
            b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
            b.native_subgroup_size = 64;
        } else {
            b.wave_size = 64;
            b.begin_fragment();
        }
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.native_saved_mask_words = true;
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.sreg[106] = b.uconst(3);
        rs.sreg[107] = b.uconst(5);
        rs.sreg_bool[20] =
            constant_mask_bit(b, static_cast<uint32_t>(mask), static_cast<uint32_t>(mask >> 32));
        rs.vreg[0] = b.uconst(0);
    }
    bool body(const std::vector<uint32_t>& code,
              const std::function<bool(RegState&, const Rdna2Inst&)>& callback = {}) {
        const auto ins = decode(code);
        EXPECT_TRUE(detect_counted_loop(ins).found) << "positive control for the actual seam";
        return emit_body(b, rs, ins, {}, nullptr, true, false, callback, code.data(), code.size());
    }
    void expect_refusal(const std::vector<uint32_t>& code) const {
        const auto ins = decode(code);
        const auto copy = std::find_if(ins.begin(), ins.end(), [](const auto& in) {
            return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                   in.dst.value == 106;
        });
        ASSERT_NE(copy, ins.end());
        EXPECT_EQ(b.stage_reject_pc, copy->pc);
        EXPECT_EQ(b.stage_reject_reason, "saved-mask-copy-words-unavailable");
        EXPECT_EQ(ballots(b), 0u);
    }
};
std::vector<uint32_t> production_program(const std::vector<uint32_t>& tail = {}) {
    namespace fixture = prosper::test::fragment_compare_kill;
    std::vector<uint32_t> code{mov(106, 129, true), mov(30, 106)};
    counted(code);
    code.insert(code.end(), {0x7e000281u, 0x7e020280u, fixture::compare(), fixture::compare_sd(20),
                             mov(106, 20, true), 0xd5010000u, 129u | (137u << 9) | (106u << 18)});
    code.insert(code.end(), tail.begin(), tail.end());
    code.insert(code.end(), {0xf800080fu, 0x00000000u, end});
    return code;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address) {
    return recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX, nullptr,
                              false, {RecompileDiagnosticStage::Fragment, address});
}
template <class Builder>
void expect_no_active_body_frame(const Builder& b) {
    // The shipping-production red arm has no continuation field. Both versions
    // must reject the public suffix below; the candidate must also retire its token.
    if constexpr (requires { b.saved_mask_copy_frame; }) EXPECT_FALSE(b.saved_mask_copy_frame);
}
} // namespace

TEST(CountedCopyContinuation, FragmentAndNativeComputeRetainTheOriginalBoolProof) {
    for (bool compute : {false, true}) {
        for (uint32_t iterations : {0u, 3u}) {
            Emitter e(compute);
            ASSERT_TRUE(e.body(program({}, false, iterations)));
            EXPECT_TRUE(e.b.saved_mask_word_roots.contains(106));
            EXPECT_EQ(e.rs.vcc, e.rs.sreg_bool.at(20));
            EXPECT_FALSE(e.rs.sreg.contains(106));
            EXPECT_FALSE(e.rs.sreg.contains(107));
            EXPECT_EQ(ballots(e.b), 0u);
        }
    }
}

TEST(CountedCopyContinuation, EmittedPostludeSelectsTheCurrentPredicateValues) {
    Emitter e;
    ASSERT_TRUE(e.body(program()));
    // Evaluate the actual emitted SSA sink with supplied lane IDs. This does not
    // execute a fragment entry point or simulate the loop's physical participation.
    fragment_predicate_oracle::Values vm(e.b);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto actual = vm.evaluate(e.rs.vreg.at(0), lane);
        ASSERT_TRUE(actual.has_value()) << vm.error;
        EXPECT_EQ(actual.value_or(UINT32_MAX), (mask >> lane) & 1u ? 9u : 1u) << lane;
    }
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(CountedCopyContinuation, LaterCompleteReplacementOwnsNumericReads) {
    Emitter e;
    ASSERT_TRUE(e.body(program({mov(106, 129, true), mov(31, 107)})));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(CountedCopyContinuation, GenuineCurrentPhysicalWordsRemainUsableAfterTheLoop) {
    Emitter e;
    e.rs.sreg[20] = e.b.uconst(static_cast<uint32_t>(mask));
    e.rs.sreg[21] = e.b.uconst(static_cast<uint32_t>(mask >> 32));
    ASSERT_TRUE(e.body(program({mov(31, 107)})));
    EXPECT_EQ(e.rs.sreg.at(31), e.b.uconst(static_cast<uint32_t>(mask >> 32)));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(CountedCopyContinuation, StructuredNestedPostludeKeepsOnlyBoolCopyAuthority) {
    Emitter e;
    // Nested SCC forward-if regions, both ending at the original END. The
    // postlude uses the existing structurizer rather than flattened test emission.
    const auto code = program({0xbf068302u, 0xbf840003u, 0xbf068302u, 0xbf840001u, mov(8, 129)});
    ASSERT_TRUE(e.body(code));
    EXPECT_EQ(e.rs.vcc, e.rs.sreg_bool.at(20));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(CountedCopyContinuation, CurrentPhysicalLowAndHighReadsStillRefuse) {
    for (uint32_t word : {106u, 107u}) {
        Emitter e;
        const auto code = program({mov(31, word)});
        EXPECT_FALSE(e.body(code));
        e.expect_refusal(code);
    }
}

TEST(CountedCopyContinuation, AliasAndPartialReplacementStillRequirePhysicalWords) {
    for (const auto& tail : {std::vector<uint32_t>{mov(24, 106, true), mov(31, 25)},
                             std::vector<uint32_t>{mov(106, 129), mov(31, 107)},
                             std::vector<uint32_t>{mov(107, 129), mov(31, 106)}}) {
        Emitter e;
        const auto code = program(tail);
        EXPECT_FALSE(e.body(code));
        e.expect_refusal(code);
    }
}

TEST(CountedCopyContinuation, OneArmReplacementCannotRetireTheCurrentGeneration) {
    Emitter e;
    const auto code = program({0xbf840001u, mov(106, 129, true), mov(31, 107)});
    EXPECT_FALSE(e.body(code));
    e.expect_refusal(code);
}

TEST(CountedCopyContinuation, B32AndMissingOriginalSourceCannotBorrowTheContinuation) {
    for (bool missing : {false, true}) {
        Emitter e;
        if (missing)
            e.rs.sreg_bool.erase(20);
        else
            e.rs.sreg_bool_b32.insert(20);
        EXPECT_FALSE(e.body(program()));
    }
}

TEST(CountedCopyContinuation, PublicSliceRevocationSurvivesTheLoopExit) {
    Emitter e;
    const auto code = program({}, true);
    const auto original = decode(code);
    auto slice = original;
    slice.erase(slice.begin());
    unsigned calls = 0;
    EXPECT_FALSE(e.body(code, [&](RegState& current, const Rdna2Inst&) {
        ++calls;
        initialize_saved_mask_word_proof(e.b, slice, code.data(), code.size(), 0, &current);
        initialize_saved_mask_word_proof(e.b, original, code.data(), code.size(), 0, &current);
        return true;
    }));
    EXPECT_EQ(calls, 1u);
    EXPECT_TRUE(e.b.fragment_bool_only_vcc_copy_pcs.empty());
    e.expect_refusal(code);
}

TEST(CountedCopyContinuation, ArtificialEndAndDecodedMutationPermanentlyRevoke) {
    for (bool artificial : {false, true}) {
        Emitter e;
        const auto code = program({}, true);
        auto altered = decode(code);
        if (artificial)
            altered.back().synthetic_terminator = true;
        else
            altered[0].dst.value = 31;
        EXPECT_FALSE(e.body(code, [&](RegState& current, const Rdna2Inst&) {
            initialize_saved_mask_word_proof(e.b, altered, code.data(), code.size(), 0, &current);
            return true;
        }));
        e.expect_refusal(code);
    }
}

TEST(CountedCopyContinuation, ChangedAndTruncatedRawBytesPermanentlyRevoke) {
    for (bool truncate : {false, true}) {
        Emitter e;
        const auto code = program({}, true);
        auto altered = code;
        if (truncate)
            altered.pop_back();
        else
            altered[0] ^= 1u;
        const auto altered_ins = decode(altered);
        EXPECT_FALSE(e.body(code, [&](RegState& current, const Rdna2Inst&) {
            initialize_saved_mask_word_proof(e.b, altered_ins, altered.data(), altered.size(), 0,
                                             &current);
            return true;
        }));
        e.expect_refusal(code);
    }
}

TEST(CountedCopyContinuation, ChangedExecutionDomainCannotBeRestoredIntoAuthority) {
    Emitter e;
    const auto code = program({}, true);
    const auto original = decode(code);
    EXPECT_FALSE(e.body(code, [&](RegState& current, const Rdna2Inst&) {
        e.b.wave_size = 32;
        initialize_saved_mask_word_proof(e.b, original, code.data(), code.size(), 0, &current);
        e.b.wave_size = 64;
        return true;
    }));
    e.expect_refusal(code);
}

TEST(CountedCopyContinuation, CachedBoolAndB32EntrySeedsRemainBoundToTheRoot) {
    for (bool b32 : {false, true}) {
        Emitter e;
        const auto code = program({}, true);
        const auto original = decode(code);
        EXPECT_FALSE(e.body(code, [&](RegState& current, const Rdna2Inst&) {
            RegState changed = current;
            if (b32)
                changed.sreg_bool_b32.insert(22);
            else
                changed.sreg_bool[22] = e.b.btrue();
            initialize_native_logical_data_demand(e.b, original, &changed);
            EXPECT_TRUE(e.b.fragment_bool_only_vcc_copy_pcs.contains(8));
            return true;
        }));
        e.expect_refusal(code);
    }
}

TEST(CountedCopyContinuation, CachedEligibleCopySetsCannotBeExtendedDuringTheBody) {
    Emitter e;
    const auto code = program({}, true);
    EXPECT_FALSE(e.body(code, [&](RegState&, const Rdna2Inst&) {
        EXPECT_TRUE(e.b.fragment_bool_only_vcc_copy_pcs.contains(8));
        e.b.fragment_bool_only_vcc_copy_pcs.insert(0);
        return true;
    }));
    e.expect_refusal(code);
}

TEST(CountedCopyContinuation, CurrentSourceMustStillBeACompleteBoolAtTheCopy) {
    for (bool missing : {false, true}) {
        Emitter e;
        auto code = program();
        code.insert(code.begin() + 6, {0xf800080fu, 0x00000000u});
        EXPECT_FALSE(e.body(code, [&](RegState& current, const Rdna2Inst&) {
            if (missing)
                current.sreg_bool.erase(20);
            else
                current.sreg_bool_b32.insert(20);
            return true;
        }));
    }
}

TEST(CountedCopyContinuation, PublicBodySlicesCannotManufactureAContinuation) {
    for (bool first : {false, true}) {
        Emitter e;
        const auto code = program();
        const auto original = decode(code);
        std::vector<Rdna2Inst> suffix;
        for (const auto& in : original)
            if (in.pc >= 6) suffix.push_back(in);
        if (!first)
            initialize_saved_mask_word_proof(e.b, original, code.data(), code.size(), 0, &e.rs);
        EXPECT_FALSE(
            emit_body(e.b, e.rs, suffix, {}, nullptr, true, false, {}, code.data(), code.size()));
        e.expect_refusal(code);
    }
}

TEST(CountedCopyContinuation, CopiedBuilderCannotReuseAnExpiredBodyFrame) {
    Emitter e;
    const auto code = program({}, true);
    std::unique_ptr<SpirvCompute> copied;
    std::unique_ptr<RegState> copied_state;
    ASSERT_TRUE(e.body(code, [&](RegState& current, const Rdna2Inst&) {
        copied = std::make_unique<SpirvCompute>(e.b);
        copied_state = std::make_unique<RegState>(current);
        return true;
    }));
    ASSERT_NE(copied, nullptr);
    ASSERT_NE(copied_state, nullptr);
    const auto original = decode(code);
    std::vector<Rdna2Inst> suffix;
    for (const auto& in : original)
        if (in.pc >= 8) suffix.push_back(in);
    EXPECT_FALSE(emit_body(*copied, *copied_state, suffix, {}, nullptr, true, false, {},
                           code.data(), code.size()));
    EXPECT_EQ(copied->stage_reject_reason, "saved-mask-copy-words-unavailable");
    EXPECT_EQ(copied->stage_reject_pc, 8u);
}

TEST(CountedCopyContinuation, ThrowingCallbackCannotLeaveAContinuationOffer) {
    Emitter e;
    const auto code = program({}, true);
    EXPECT_THROW(e.body(code,
                        [](RegState&, const Rdna2Inst&) -> bool {
                            throw std::runtime_error("owned test callback");
                        }),
                 std::runtime_error);
    const auto original = decode(code);
    std::vector<Rdna2Inst> suffix;
    for (const auto& in : original)
        if (in.pc >= 8) suffix.push_back(in);
    EXPECT_FALSE(
        emit_body(e.b, e.rs, suffix, {}, nullptr, true, false, {}, code.data(), code.size()));
    e.expect_refusal(code);
}

TEST(CountedCopyContinuation, ThrowingPostludeClosesTheAuthenticatedChildAndParent) {
    Emitter e;
    auto code = program();
    code.insert(code.begin() + 6, {0xf800080fu, 0x00000000u});
    EXPECT_THROW(e.body(code,
                        [](RegState&, const Rdna2Inst&) -> bool {
                            throw std::runtime_error("owned postlude callback");
                        }),
                 std::runtime_error);
    expect_no_active_body_frame(e.b);
    const auto original = decode(code);
    std::vector<Rdna2Inst> suffix;
    for (const auto& in : original)
        if (in.pc >= 8) suffix.push_back(in);
    EXPECT_FALSE(
        emit_body(e.b, e.rs, suffix, {}, nullptr, true, false, {}, code.data(), code.size()));
    e.expect_refusal(code);
}

TEST(CountedCopyContinuation, PostludeEdgeIntoRemovedPrefixCannotInheritTheProof) {
    Emitter e;
    // pc9 VCCZ branches back into the counted header at pc2. Counted-loop
    // recognition ignores this VCC edge; a suffix-only proof must reject it.
    const auto code = program({0xbf86fff8u});
    EXPECT_FALSE(e.body(code));
    e.expect_refusal(code); // exact copy pc6, rather than a later invalid-branch refusal
}

TEST(CountedCopyContinuation, PhysicalParticipationDoesNotSurviveThePostlude) {
    for (bool compute : {false, true}) {
        Emitter e(compute);
        unsigned calls = 0;
        ASSERT_TRUE(e.body(program({}, true), [&](RegState&, const Rdna2Inst&) {
            ++calls;
            // The original body owns this separate physical-participation ledger. The real
            // counted-postlude token preserves Bool-copy authority only, even when a cached
            // producer site remains present before the recursive initializer runs.
            e.b.native_participating_mask_pcs.insert(0);
            return true;
        }));
        EXPECT_EQ(calls, 1u);
        EXPECT_TRUE(e.b.saved_mask_entry_prefix_pcs.empty());
        EXPECT_TRUE(e.b.saved_mask_reconverged_entry_pcs.empty());
        EXPECT_TRUE(e.b.native_logical_uniform_entry_pcs.empty());
        EXPECT_TRUE(e.b.native_participating_mask_pcs.empty());
        EXPECT_EQ(e.rs.vcc, e.rs.sreg_bool.at(20));
        EXPECT_EQ(ballots(e.b), 0u);
    }
}

TEST(CountedCopyContinuation, CompleteFragmentProgramCompilesThroughTheActualSelector) {
    const uint64_t address = 0xa4921001ull;
    const auto module = compile(production_program(), address);
    EXPECT_FALSE(module.empty()) << last_terminal_reject_reason(address);
}

TEST(CountedCopyContinuation, CompleteFragmentNumericReadNeedsGenuineWords) {
    const uint64_t address = 0xa4921002ull;
    EXPECT_TRUE(compile(production_program({mov(31, 107)}), address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("saved-mask-copy-words-unavailable"),
              std::string::npos);
}

TEST(CountedCopyContinuation, CompleteNativeComputeProgramCompilesThroughTheActualSelector) {
    auto code = production_program();
    code.erase(code.end() - 3, code.end() - 1); // compute has no fragment EXP
    ComputeShaderConfig config;
    config.wave_size = config.native_subgroup_size = config.local_x = 64;
    config.user_sgprs.resize(8);
    const uint64_t address = 0xa4921003ull;
    const auto module = recompile_compute(code.data(), code.size(), nullptr, config,
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_FALSE(module.empty()) << last_terminal_reject_reason(address);
}
} // namespace prosper::gpu
