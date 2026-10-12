// #4969: a discarded optimized attempt must not enter the final refusal/capture history.
// Paired commits prove absence is transaction selection, rather than a broken recorder.
#include "gpu/recompiler/rdna2_diagnostic_attempt.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>
#include <barrier>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
using Records = std::vector<std::pair<std::string, std::string>>;
constexpr uint64_t base = 0x49692000ull;
void diagnostic(uint64_t address, const char* role, const char* payload) {
    log_recompile_diagnostic({RecompileDiagnosticStage::Compute, address}, "attempt-test", role,
                             "%s", payload);
}
} // namespace

TEST(RecompileDiagnosticAttempt, DiscardSuppressesRecordsAndCommitPublishesTheChosenCause) {
    TerminalRejectCapture capture;
    {
        RecompileDiagnosticAttempt discarded;
        diagnostic(base, "terminal", "discarded");
        discarded.discard();
    }
    EXPECT_TRUE(last_terminal_reject_reason(base).empty());
    EXPECT_TRUE(capture.take().empty());
    {
        RecompileDiagnosticAttempt chosen;
        diagnostic(base, "terminal", "chosen");
        chosen.commit();
    }
    EXPECT_EQ(last_terminal_reject_reason(base), "attempt-test chosen");
    EXPECT_EQ(capture.take(), (Records{{"attempt-test", "chosen"}}));
}
TEST(RecompileDiagnosticAttempt, ChildCommitStaysPrivateWhenItsParentDiscards) {
    TerminalRejectCapture capture;
    RecompileDiagnosticAttempt parent;
    diagnostic(base + 1, "terminal", "parent");
    {
        RecompileDiagnosticAttempt child;
        diagnostic(base + 1, "route-decline", "child");
        child.commit();
    }
    EXPECT_TRUE(capture.take().empty());
    EXPECT_TRUE(last_terminal_reject_reason(base + 1).empty());
    parent.discard();
    EXPECT_TRUE(capture.take().empty());
    EXPECT_TRUE(last_terminal_reject_reason(base + 1).empty());
    diagnostic(base + 1, "terminal", "outside");
    EXPECT_EQ(capture.take(), (Records{{"attempt-test", "outside"}}));
}
TEST(RecompileDiagnosticAttempt, ChildAndParentCommitPreserveOrderedChosenRecords) {
    TerminalRejectCapture capture;
    RecompileDiagnosticAttempt parent;
    diagnostic(base + 8, "terminal", "before child");
    {
        RecompileDiagnosticAttempt child;
        record_terminal_reject_reason(base + 8, "direct-test", "child direct");
        diagnostic(base + 8, "route-decline", "child logged");
        child.commit();
    }
    EXPECT_TRUE(capture.take().empty());
    diagnostic(base + 8, "terminal", "after child");
    parent.commit();
    EXPECT_EQ(capture.take(), (Records{{"attempt-test", "before child"},
                                       {"direct-test", "child direct"},
                                       {"attempt-test", "child logged"},
                                       {"attempt-test", "after child"}}));
    EXPECT_EQ(last_terminal_reject_reason(base + 8), "attempt-test after child");
}
TEST(RecompileDiagnosticAttempt, ChildDiscardPreservesParentRecordOrderAndDirectRecords) {
    TerminalRejectCapture capture;
    RecompileDiagnosticAttempt parent;
    diagnostic(base + 2, "terminal", "first");
    {
        RecompileDiagnosticAttempt child;
        diagnostic(base + 2, "terminal", "discarded");
    }
    record_terminal_reject_reason(base + 2, "direct-test", "second");
    diagnostic(base + 2, "route-decline", "third");
    diagnostic(base + 2, "consequent", "empty module");
    parent.commit();
    EXPECT_EQ(
        capture.take(),
        (Records{{"attempt-test", "first"}, {"direct-test", "second"}, {"attempt-test", "third"}}));
    EXPECT_EQ(last_terminal_reject_reason(base + 2), "attempt-test third");
}
TEST(RecompileDiagnosticAttempt, NullAndEmptyRolesRetainTheDirectRecorderContract) {
    for (bool deferred : {false, true}) {
        TerminalRejectCapture capture;
        RecompileDiagnosticAttempt attempt;
        if (!deferred) attempt.commit();
        const auto address = base + 3 + static_cast<uint64_t>(deferred);
        diagnostic(address, nullptr, "no terminal role");
        attempt.commit();
        EXPECT_TRUE(capture.take().empty());
        EXPECT_TRUE(last_terminal_reject_reason(address).empty());
        {
            RecompileDiagnosticAttempt present;
            if (!deferred) present.commit();
            diagnostic(address, "", "empty role is present");
            present.commit();
        }
        EXPECT_EQ(capture.take(), (Records{{"attempt-test", "empty role is present"}}));
    }
}
TEST(RecompileDiagnosticAttempt, ZeroAddressStillReachesAnEnclosingCapture) {
    TerminalRejectCapture capture;
    RecompileDiagnosticAttempt attempt;
    diagnostic(0, "terminal", "addressless");
    record_terminal_reject_reason(0, "direct-test", "addressless direct");
    attempt.commit();
    EXPECT_EQ(capture.take(),
              (Records{{"attempt-test", "addressless"}, {"direct-test", "addressless direct"}}));
    EXPECT_TRUE(last_terminal_reject_reason(0).empty());
}
TEST(RecompileDiagnosticAttempt, SuccessfulDiscardDoesNotEraseHistoricalAddressRecords) {
    TerminalRejectCapture capture;
    diagnostic(base + 5, "terminal", "history");
    capture.take();
    {
        RecompileDiagnosticAttempt attempt;
        diagnostic(base + 5, "terminal", "discarded");
    }
    EXPECT_TRUE(capture.take().empty());
    EXPECT_EQ(last_terminal_reject_reason(base + 5), "attempt-test history");
}
TEST(RecompileDiagnosticAttempt, ConcurrentThreadsDoNotSharePendingAttemptRecords) {
    Records discarded_records, committed_records;
    std::barrier overlap(2);
    std::thread discard_thread([&] {
        TerminalRejectCapture capture;
        {
            RecompileDiagnosticAttempt attempt;
            diagnostic(base + 6, "terminal", "discarded thread");
            overlap
                .arrive_and_wait();   // keep this attempt open throughout the other thread's commit
            overlap.arrive_and_wait();
        }
        discarded_records = capture.take();
    });
    std::thread commit_thread([&] {
        overlap.arrive_and_wait();
        TerminalRejectCapture capture;
        RecompileDiagnosticAttempt attempt;
        diagnostic(base + 7, "terminal", "committed thread");
        attempt.commit();
        committed_records = capture.take();
        overlap.arrive_and_wait();
    });
    discard_thread.join();
    commit_thread.join();
    EXPECT_TRUE(discarded_records.empty());
    EXPECT_TRUE(last_terminal_reject_reason(base + 6).empty());
    EXPECT_EQ(committed_records, (Records{{"attempt-test", "committed thread"}}));
    EXPECT_EQ(last_terminal_reject_reason(base + 7), "attempt-test committed thread");
}
