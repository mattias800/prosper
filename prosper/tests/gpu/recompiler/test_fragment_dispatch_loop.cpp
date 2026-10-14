// The new saved-mask continuation must prove finite interior control flow before
// exposing failed lanes to it. These CPU-only certificates grant no emitter route.
#include "fixtures/fragment_dispatch_bool_copy.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_fragment_dispatch_loop.hpp"
#include <gtest/gtest.h>

namespace prosper::gpu {
namespace {
namespace fixture = prosper::test::fragment_dispatch_bool_copy;
std::vector<Rdna2Inst> selected(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    if (!rdna2_specialize_pcrel_dispatch(ins, rdna2_pcrel_dispatch_info(code.data(), code.size()),
                                         17))
        return {};
    return ins;
}
bool finite(const std::vector<Rdna2Inst>& ins) {
    const auto branch = std::find_if(ins.begin(), ins.end(), [](const auto& in) {
        return in.fmt == Rdna2Format::SOPP && in.opcode == kSoppOpcodeCbranchScc0 && in.simm16 > 2;
    });
    if (branch == ins.end() || ins.size() < 3) return false;
    return fragment_dispatch_interior_loops_are_finite(ins, branch->pc, ins[ins.size() - 3].pc);
}
} // namespace

TEST(FragmentDispatchLoop, BoundedUnitRecurrenceHasAFiniteCertificate) {
    const auto ins =
        selected(fixture::program(fixture::Tail::Bool, false, 3, true, fixture::EarlyOut::Killed));
    ASSERT_FALSE(ins.empty());
    ASSERT_TRUE(detect_counted_loop(ins).found);
    EXPECT_TRUE(finite(ins));
}

TEST(FragmentDispatchLoop, ExecAndVccBackedgesBeyondCountedLoopNeedAnotherProof) {
    for (uint32_t opcode : {6u, 7u, 8u, 9u}) {
        SCOPED_TRACE(opcode);
        auto code =
            fixture::program(fixture::Tail::Bool, false, 3, true, fixture::EarlyOut::Killed);
        const auto loop = detect_counted_loop(selected(code));
        ASSERT_TRUE(loop.found);
        code[loop.exit_pc] = 0xbf800000u | (opcode << 16) | 0xffffu;
        const auto ins = selected(code);
        ASSERT_FALSE(ins.empty());
        const auto retained = detect_counted_loop(ins);
        ASSERT_TRUE(retained.found) << "EXEC/VCC backedges are excluded from its count";
        ASSERT_EQ(retained.backedge_pc, loop.backedge_pc);
        EXPECT_FALSE(finite(ins));
    }
}

TEST(FragmentDispatchLoop, WaterfallClassifiedBackedgeNeedsItsOwnTerminationProof) {
    for (uint32_t opcode : {4u, 5u}) {
        SCOPED_TRACE(opcode);
        auto code =
            fixture::program(fixture::Tail::Bool, false, 3, true, fixture::EarlyOut::Killed);
        const auto loop = detect_counted_loop(selected(code));
        ASSERT_TRUE(loop.found);
        code[loop.exit_pc] = 0x87a07e7eu; // s_and_b64 s[32:33], EXEC, EXEC
        code[loop.exit_pc + 1] = 0xbf800000u | (opcode << 16) | 0xfffeu;
        const auto ins = selected(code);
        ASSERT_FALSE(ins.empty());
        ASSERT_TRUE(waterfall_branches(ins).contains(loop.exit_pc + 1));
        ASSERT_TRUE(detect_counted_loop(ins).found);
        EXPECT_FALSE(finite(ins));
    }
}

TEST(FragmentDispatchLoop, EveryEntryAndLatchPathMustVisitItsInductionProducer) {
    for (auto early : {fixture::EarlyOut::SkippedInductionInitializer,
                       fixture::EarlyOut::SkippedInductionStep}) {
        SCOPED_TRACE(static_cast<int>(early));
        auto code = fixture::program(fixture::Tail::Bool, false, 3, true, early);
        const auto before = selected(code);
        const auto skip = std::find_if(before.begin(), before.end(), [](const auto& in) {
            return in.fmt == Rdna2Format::SOPP && in.opcode == 2 && in.simm16 == 1;
        });
        ASSERT_NE(skip, before.end());
        code[skip->pc] = 0xbf880001u; // closed forward EXECZ edge skips one authentic writer
        const auto ins = selected(code);
        ASSERT_TRUE(detect_counted_loop(ins).found);
        EXPECT_FALSE(finite(ins)) << "no earlier emitter refusal can hide this obligation";
    }
}

TEST(FragmentDispatchLoop, InvariantBoundAndWholeCounterWritesAreSeparateObligations) {
    for (bool overwrite_bound : {false, true}) {
        SCOPED_TRACE(overwrite_bound);
        auto code = fixture::program(fixture::Tail::Bool, false, 3, true,
                                     fixture::EarlyOut::ReplacedInductionWord);
        const auto loop = detect_counted_loop(selected(code));
        ASSERT_TRUE(loop.found);
        if (overwrite_bound) {
            code[loop.header_pc] = 0xbf0a0302u; // s_cmp_lt_u32 s2, s3
            code[loop.backedge_pc - 1] = fixture::mov(3, 131);
        } else {
            code[loop.backedge_pc - 1] = fixture::mov(1, 193, true); // overlaps counter s2
        }
        const auto ins = selected(code);
        ASSERT_TRUE(detect_counted_loop(ins).found);
        EXPECT_FALSE(finite(ins));
    }
}
} // namespace prosper::gpu
