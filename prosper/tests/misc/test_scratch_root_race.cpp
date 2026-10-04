// The shared scratch root can be removed by a peer between create_directories' existence check and
// its mkdir of our leaf (test_scratch.h, ~ScratchDirectory's non-recursive remove of the root).
// create_scratch_leaf must survive that by recreating the root. Peer threads run exactly the
// production lifecycle -- create a leaf, delete it, try to remove the shared root -- so each peer
// removes the root at most once per lifetime, as a peer process does at exit.
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace fs = std::filesystem;

// Rounds of create-then-delete for one leaf while four peers churn their own leaves under the same
// root; returns how many of our creations failed.
int failures_among_peers(int attempts, int rounds) {
    const fs::path root = prosper_test::test_scratch_dir() / "shared-root";
    std::atomic<bool> stop{false};
    std::vector<std::thread> peers;
    for (int peer = 0; peer < 4; ++peer)
        peers.emplace_back([&, peer] {
            const fs::path leaf = root / ("peer-" + std::to_string(peer));
            std::error_code ignored;
            while (!stop.load(std::memory_order_relaxed)) {
                prosper_test::detail::create_scratch_leaf(leaf, ignored);
                fs::remove_all(leaf, ignored);
                fs::remove(root, ignored);   // non-recursive, as ~ScratchDirectory does
            }
        });
    int failures = 0;
    for (int round = 0; round < rounds; ++round) {
        const fs::path leaf = root / "subject";
        std::error_code error;
        if (!prosper_test::detail::create_scratch_leaf(leaf, error, attempts)) ++failures;
        fs::remove_all(leaf, error);
        fs::remove(root, error);
    }
    stop = true;
    for (auto& peer : peers) peer.join();
    return failures;
}

TEST(ScratchRootRace, CreationSurvivesPeersRemovingTheSharedRoot) {
    EXPECT_EQ(failures_among_peers(64, 20000), 0)
        << "a peer removing the empty shared root must not fail our creation";
}

// Positive control: the race is real and reachable here, so the zero above is not a property of a
// quiet machine.
TEST(ScratchRootRace, SingleAttemptLosesTheRace) {
    EXPECT_GT(failures_among_peers(1, 20000), 0)
        << "without a retry, a removed root must surface as a failed creation";
}
}   // namespace
