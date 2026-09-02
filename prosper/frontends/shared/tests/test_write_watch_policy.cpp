#include "shared/texture/write_watch_census.hpp"
#include "shared/texture/write_watch_policy.hpp"

#include <cstdio>
#include <cstring>
#include <string>

using prosper::frontend::should_promote_write_watch;
using prosper::frontend::WriteWatchPromotionBudget;
using prosper::frontend::update_write_watch_stability;
using prosper::frontend::format_write_watch_census;
using prosper::frontend::WriteWatchCensus;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    constexpr size_t defer_min = 8u << 20;
    constexpr uint32_t promote_after = 3;

    CHECK(should_promote_write_watch(4u << 20, 0, defer_min, promote_after));
    CHECK(!should_promote_write_watch(32u << 20, 0, defer_min, promote_after));
    CHECK(!should_promote_write_watch(32u << 20, 2, defer_min, promote_after));
    CHECK(should_promote_write_watch(32u << 20, 3, defer_min, promote_after));

    uint32_t stable = 0;
    stable = update_write_watch_stability(stable, true, promote_after);
    stable = update_write_watch_stability(stable, true, promote_after);
    stable = update_write_watch_stability(stable, true, promote_after);
    CHECK(stable == promote_after);
    CHECK(update_write_watch_stability(stable, true, promote_after) == promote_after);
    CHECK(update_write_watch_stability(stable, false, promote_after) == 0);

    // Both knobs accept zero: defer nothing, or promote a deferred source immediately.
    CHECK(should_promote_write_watch(32u << 20, 0, 0, promote_after));
    CHECK(should_promote_write_watch(32u << 20, 0, defer_min, 0));

    WriteWatchPromotionBudget budget;
    budget.reset(8u << 20);
    CHECK(budget.try_consume(32u << 20));
    CHECK(!budget.try_consume(1u << 20));
    budget.reset(8u << 20);
    CHECK(budget.try_consume(3u << 20));
    CHECK(budget.try_consume(5u << 20));
    CHECK(!budget.try_consume(1));
    budget.reset(0);
    CHECK(budget.try_consume(64u << 20));
    CHECK(budget.try_consume(64u << 20));

    // The size exemption is the half of this policy the compute call site cannot reach: it passes
    // defer_min_bytes = 1, so `source_bytes < defer_min_bytes` is false for every real source and
    // only the stability ladder can promote. Pin both readings so the difference is a fact in the
    // test rather than an observation in an issue comment (#3155).
    CHECK(should_promote_write_watch(4u << 20, 0, defer_min, promote_after));
    CHECK(!should_promote_write_watch(4u << 20, 0, 1, promote_after));
    CHECK(!should_promote_write_watch(4u << 10, 0, 1, promote_after));
    CHECK(should_promote_write_watch(4u << 10, promote_after, 1, promote_after));

    // ---- census (#3155) -------------------------------------------------------------------
    // A census whose buckets do not partition its own denominator is worse than none: every ratio
    // it prints is then wrong in a way no reader can see.
    {
        WriteWatchCensus census;
        census.record_promotion_decision(0, false, false);
        census.record_promotion_decision(0, false, false);
        census.record_promotion_decision(1, false, false);
        census.record_promotion_decision(2, true, true);
        census.record_promotion_decision(7, true, false);   // threshold met, budget refused
        census.record_journal_skip(1024);
        census.record_watch_skip(2048);
        census.record_exact_compare(4096);
        census.record_exact_compare(4096);

        const auto snapshot = census.snapshot();
        CHECK(snapshot.decisions == 5);
        CHECK(snapshot.stability_0 == 2 && snapshot.stability_1 == 1 &&
              snapshot.stability_2 == 1 && snapshot.stability_3_plus == 1);
        CHECK(snapshot.stability_0 + snapshot.stability_1 + snapshot.stability_2 +
              snapshot.stability_3_plus == snapshot.decisions);
        CHECK(snapshot.threshold_met == 2 && snapshot.granted == 1 &&
              snapshot.budget_refused == 1);
        CHECK(snapshot.journal_skips == 1 && snapshot.journal_skip_bytes == 1024);
        CHECK(snapshot.watch_skips == 1 && snapshot.watch_skip_bytes == 2048);
        CHECK(snapshot.exact_compares == 2 && snapshot.exact_compare_bytes == 8192);

        // Every report carries the two denominators, so two runs whose reports fired at different
        // points remain comparable. Reading a running tally as a run total is what produced this
        // issue's retracted 55% figure.
        char line[512];
        const size_t used = format_write_watch_census(snapshot, line, sizeof line);
        CHECK(used > 0 && used < sizeof line);
        const std::string text(line, used);
        CHECK(text.find("decisions=5") != std::string::npos);
        CHECK(text.find("validated=4") != std::string::npos);
        CHECK(text.find("running totals") != std::string::npos);
        CHECK(text.find("stability0=2 (40.0%)") != std::string::npos);

        // An empty census must divide by nothing rather than by zero.
        char empty_line[512];
        const size_t empty_used =
            format_write_watch_census(WriteWatchCensus{}.snapshot(), empty_line,
                                      sizeof empty_line);
        CHECK(empty_used > 0);
        CHECK(std::string(empty_line, empty_used).find("stability0=0 (0.0%)") !=
              std::string::npos);

        // A capacity too small to hold the report truncates instead of writing past the buffer.
        // The guard bytes beyond the declared capacity must survive untouched, and the returned
        // length must never include the NUL -- a caller that writes `used` bytes to a log would
        // otherwise emit one.
        char tiny[32];
        std::memset(tiny, 0x7f, sizeof tiny);
        constexpr size_t tiny_capacity = 16;
        const size_t tiny_used = format_write_watch_census(snapshot, tiny, tiny_capacity);
        CHECK(tiny_used == tiny_capacity - 1);
        CHECK(tiny[tiny_capacity - 1] == '\0');
        CHECK(tiny[tiny_capacity] == 0x7f && tiny[sizeof tiny - 1] == 0x7f);
        CHECK(format_write_watch_census(snapshot, nullptr, sizeof tiny) == 0);
        CHECK(format_write_watch_census(snapshot, tiny, 0) == 0);
    }

    if (!failures) std::printf("write_watch_policy: OK\n");
    return failures ? 1 : 0;
}
