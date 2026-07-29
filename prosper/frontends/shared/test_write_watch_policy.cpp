#include "write_watch_policy.hpp"

#include <cstdio>

using prosper::frontend::should_promote_write_watch;
using prosper::frontend::WriteWatchPromotionBudget;
using prosper::frontend::update_write_watch_stability;

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

    if (!failures) std::printf("write_watch_policy: OK\n");
    return failures ? 1 : 0;
}
