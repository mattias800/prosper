#include "shared/texture/write_watch_census.hpp"
#include "shared/texture/write_watch_policy.hpp"
#include "diagnostics/env_numeric.hpp"

#include <cstdio>
#include <cstdlib>
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

    // ---- the knobs' own parsing (#3253) ----------------------------------------------------
    //
    // These five variables tune the policy above, and on every one of them ZERO is a meaningful and
    // maximally aggressive setting rather than "off". A bare `strtoull(value, nullptr, 10)` reaches
    // that zero for anything it cannot start parsing -- and, measured on glibc rather than assumed,
    // reaches TWO other silent wrong answers as well:
    //
    //   "eight", "yes", "\"8192\""  -> 0                    the most aggressive arm
    //   "8mb", "8 KB", "8,192"      -> 8                    a value 1024x smaller than intended
    //   "-1", "18446744073709551616"-> UINT64_MAX           saturates to the cap, i.e. unbounded
    //
    // #3253 names only the first. The middle one is arguably the nastiest, because 8 is a plausible
    // number that no reader would question. All three are asserted below, because a fix that keeps
    // the default is only interesting next to what the old spelling actually produced.
    {
        using prosper::diag::env_u64_or_default;
        using prosper::diag::parse_u64_strict;

        uint64_t parsed = 0;
        CHECK(parse_u64_strict("0", &parsed) && parsed == 0);
        CHECK(parse_u64_strict("8192", &parsed) && parsed == 8192);
        CHECK(parse_u64_strict("18446744073709551615", &parsed) && parsed == UINT64_MAX);

        // Everything a mistyped A/B actually looks like.
        CHECK(!parse_u64_strict(nullptr, &parsed));
        CHECK(!parse_u64_strict("", &parsed));
        CHECK(!parse_u64_strict("8mb", &parsed));       // a unit suffix
        CHECK(!parse_u64_strict("8 KB", &parsed));      // a unit, spaced
        CHECK(!parse_u64_strict("8,192", &parsed));     // a thousands separator
        CHECK(!parse_u64_strict(" 8", &parsed));        // leading whitespace
        CHECK(!parse_u64_strict("8 ", &parsed));        // trailing whitespace
        CHECK(!parse_u64_strict("\"8\"", &parsed));     // a quote that survived the shell
        CHECK(!parse_u64_strict("eight", &parsed));
        CHECK(!parse_u64_strict("-1", &parsed));        // strtoull WRAPS this to UINT64_MAX
        CHECK(!parse_u64_strict("+8", &parsed));
        CHECK(!parse_u64_strict("0x2000", &parsed));
        CHECK(!parse_u64_strict("8.5", &parsed));
        CHECK(!parse_u64_strict("18446744073709551616", &parsed));   // one past the top
        CHECK(!parse_u64_strict("99999999999999999999999", &parsed));

        // A refused parse must leave the caller's value ALONE -- a helper that half-wrote its
        // output would reintroduce the defect one level up.
        uint64_t untouched = 1234;
        CHECK(!parse_u64_strict("8mb", &untouched) && untouched == 1234);

        // What the old spelling really answered. Measured, not assumed: the third of these is the
        // one that makes "a typo yields 0" an incomplete description of the hazard.
        CHECK(std::strtoull("eight", nullptr, 10) == 0);
        CHECK(std::strtoull("\"8192\"", nullptr, 10) == 0);
        CHECK(std::strtoull("8mb", nullptr, 10) == 8);
        CHECK(std::strtoull("-1", nullptr, 10) == UINT64_MAX);
        // Two more the reviewer of #3253 added, both worth keeping. A hex-looking value is the
        // sharpest of the lot: someone reaching for `0x2000` on a byte-valued knob is thinking in
        // hex, and base 10 stops at the '0' -- landing on the aggressive sentinel, not on 8192.
        CHECK(std::strtoull("0x10", nullptr, 10) == 0);
        CHECK(!parse_u64_strict("0x10", &parsed));
        // And a trailing newline, which a heredoc or a `$(cat file)` puts there without anyone
        // typing it. strtoull takes it; this does not.
        CHECK(std::strtoull("8192\n", nullptr, 10) == 8192);
        CHECK(!parse_u64_strict("8192\n", &parsed));

        // THE LOAD-BEARING ARM. Each of those keeps the DEFAULT instead.
        CHECK(env_u64_or_default("PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB", "eight", 8192) == 8192);
        CHECK(env_u64_or_default("PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB", "8mb", 8192) == 8192);
        CHECK(env_u64_or_default("PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_HITS", "-1", 3) == 3);
        // Unset and empty are not typos and take the default in silence.
        CHECK(env_u64_or_default("PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_HITS", nullptr, 3) == 3);
        CHECK(env_u64_or_default("PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_HITS", "", 3) == 3);
        // A well-formed 0 is still honoured: this refuses TYPOS, not the aggressive setting itself.
        CHECK(env_u64_or_default("PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB", "0", 8192) == 0);
        CHECK(prosper::diag::env_u64_or_default_capped(
                  "PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB", "99999", 8, 1024) == 1024);
        CHECK(prosper::diag::env_u64_or_default_capped(
                  "PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB", "12", 8, 1024) == 12);

        // ...and WHY it matters, in the policy's own terms. The value the old parse handed back for
        // a non-numeric typo is not a no-op, it is the OPPOSITE arm: a 32 MiB source the default
        // defers arms immediately under it.
        const size_t typo_zero =
            static_cast<size_t>(std::strtoull("eight", nullptr, 10)) * 1024;
        const size_t kept =
            static_cast<size_t>(env_u64_or_default("PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB",
                                                   "eight", 8192)) * 1024;
        CHECK(should_promote_write_watch(32u << 20, 0, typo_zero, promote_after));
        CHECK(!should_promote_write_watch(32u << 20, 0, kept, promote_after));

        // The 1024x arm, which breaks no test and every experiment. `=8mb` used to mean 8 KiB, and
        // a SMALLER defer minimum exempts FEWER sources -- so this lands on the opposite arm from
        // the zero above: a 1 MiB source that the intended 8 MiB minimum arms on first sight has to
        // climb the stability ladder instead. Which direction a typo moves the policy depends on the
        // typo, which is the whole problem: it is unpredictable and it is silent.
        const size_t typo_scaled = static_cast<size_t>(std::strtoull("8mb", nullptr, 10)) * 1024;
        CHECK(!should_promote_write_watch(1u << 20, 0, typo_scaled, promote_after));
        CHECK(should_promote_write_watch(1u << 20, 0, kept, promote_after));

        // The promotion budget's sentinel: 0 is unbounded, the default is not.
        WriteWatchPromotionBudget typo_budget;
        typo_budget.reset(static_cast<size_t>(std::strtoull("eight", nullptr, 10)) * (1u << 20));
        CHECK(typo_budget.try_consume(64u << 20));
        CHECK(typo_budget.try_consume(64u << 20));   // unbounded: a second large watch also arms
        WriteWatchPromotionBudget kept_budget;
        kept_budget.reset(static_cast<size_t>(env_u64_or_default(
            "PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB", "eight", 8)) * (1u << 20));
        CHECK(kept_budget.try_consume(64u << 20));
        CHECK(!kept_budget.try_consume(64u << 20));  // bounded, which is what was asked for
    }

    if (!failures) std::printf("write_watch_policy: OK\n");
    return failures ? 1 : 0;
}
