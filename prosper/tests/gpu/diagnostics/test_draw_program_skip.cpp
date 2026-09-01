// PROSPER_SKIP_DRAW_PROGRAM — decline graphics draws by shader PROGRAM identity.
//
// The gap this closes: PROSPER_COMPUTE_SKIP_PROGRAM can name a compute program, and that is what
// isolated GTA V's hanging dispatch. The graphics side had only PROSPER_SKIP_DRAW, which selects a
// submit-local `draw_index` ordinal — not stable frame to frame, so it cannot name "every draw that
// runs this shader". Astro Bot's world-map GPU reset is a DRAW_INDEX_2, so that is exactly the
// question that had no instrument.
//
// What is pinned here is the part a wrong answer would be invisible in: the selector must never arm
// on a spec it did not fully understand (a partially-armed selector produces a confident negative
// about a program nobody selected), it must report which stage matched, and its rate limit must
// carry the ordinal so the cap can never be read as the rate.
#include "gpu/diagnostics/draw_program_skip.hpp"

#include <cstdio>
#include <cstring>
#include <string>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

using Configure = DrawProgramSkipSelector::ConfigureResult;

// A draw's three program addresses, in the order evaluate() takes them.
constexpr uint64_t kVs    = 0x5006c6a00ull;
constexpr uint64_t kChain = 0x5006c7100ull;
constexpr uint64_t kPs    = 0x5002af200ull;

}  // namespace

int main() {
    printf("== test_draw_program_skip ==\n");

    // ---- default OFF -------------------------------------------------------------------------
    {
        DrawProgramSkipSelector selector;
        CHECK(selector.configure(nullptr) == Configure::Unset && !selector.armed() &&
                  selector.size() == 0,
              "an unset variable leaves the selector disarmed");
        CHECK(selector.configure("") == Configure::Unset && !selector.armed(),
              "an empty variable leaves the selector disarmed");
        const auto decision = selector.evaluate(kVs, kChain, kPs);
        CHECK(!decision.skip && decision.stage == DrawProgramStage::None &&
                  decision.ordinal == 0 && !decision.print && selector.skipped_total() == 0,
              "a disarmed selector skips nothing and counts nothing");
    }

    // ---- impossible to arm by accident -------------------------------------------------------
    // Strict parsing is delegated to parse_hex_watch_list (pinned in test_watch_list), so what is
    // asserted here is the CONSEQUENCE: a rejected spec must leave the selector inert, not
    // partially armed. A selector holding half a list answers "this program is not responsible"
    // about programs it never examined.
    {
        const char* const bad[] = {
            "21474836480",                 // bare decimal — the base-0 trap
            "0x5006c6a00,",                // trailing comma
            "0x5006c6a00,,0x5002af200",    // empty token
            "0x5006c6a00 junk",            // trailing junk
            "0x0",                         // zero is never a program
            "0x5006c6a00,21474836480",     // one bad token in an otherwise valid list
            "5006c6a00",                   // hex digits without the 0x prefix
        };
        for (const char* spec : bad) {
            DrawProgramSkipSelector selector;
            const bool rejected = selector.configure(spec) == Configure::Malformed &&
                                  !selector.armed() && selector.size() == 0 &&
                                  !selector.evaluate(kVs, kChain, kPs).skip;
            std::string message = "malformed spec \"";
            message += spec;
            message += "\" arms NOTHING and skips nothing";
            CHECK(rejected, message.c_str());
        }
    }

    // ---- a re-configure never leaves the previous list behind --------------------------------
    {
        DrawProgramSkipSelector selector;
        CHECK(selector.configure("0x5006c6a00") == Configure::Armed && selector.armed(),
              "a single 0x-prefixed program address arms the selector");
        CHECK(selector.configure("0x5006c6a00,bogus") == Configure::Malformed &&
                  !selector.armed() && !selector.evaluate(kVs, kChain, kPs).skip,
              "re-configuring with a malformed spec DISARMS rather than keeping the old list");
    }

    // ---- which stage matched -----------------------------------------------------------------
    {
        DrawProgramSkipSelector vs_only;
        vs_only.configure("0x5006c6a00");
        const auto d = vs_only.evaluate(kVs, kChain, kPs);
        CHECK(d.skip && d.stage == DrawProgramStage::Vertex && d.address == kVs &&
                  std::strcmp(draw_program_stage_name(d.stage), "vs") == 0,
              "a named VERTEX program declines the draw and reports stage=vs");

        DrawProgramSkipSelector chain_only;
        chain_only.configure("0x5006c7100");
        const auto c = chain_only.evaluate(kVs, kChain, kPs);
        CHECK(c.skip && c.stage == DrawProgramStage::VertexChain && c.address == kChain &&
                  std::strcmp(draw_program_stage_name(c.stage), "vs-chain") == 0,
              "a named NGG CHAIN continuation declines the draw and reports stage=vs-chain");

        DrawProgramSkipSelector ps_only;
        ps_only.configure("0x5002af200");
        const auto p = ps_only.evaluate(kVs, kChain, kPs);
        CHECK(p.skip && p.stage == DrawProgramStage::Fragment && p.address == kPs &&
                  std::strcmp(draw_program_stage_name(p.stage), "ps") == 0,
              "a named PIXEL program declines the draw and reports stage=ps");
    }

    // ---- a draw that uses none of the named programs is untouched ----------------------------
    {
        DrawProgramSkipSelector selector;
        selector.configure("0x5006c6a00,0x5002af200");
        CHECK(!selector.evaluate(0x400000000ull, 0, 0x400001000ull).skip &&
                  selector.skipped_total() == 0,
              "a draw using none of the named programs is not declined");
        CHECK(!selector.evaluate(0, 0, 0).skip,
              "a draw whose program addresses are all zero never matches an armed selector");
        CHECK(selector.evaluate(kVs, 0, 0).skip && selector.evaluate(0, 0, kPs).skip &&
                  selector.skipped_total() == 2,
              "a comma list declines a draw matching EITHER named program, and both are counted");
    }

    // ---- the rate limit carries its ordinal --------------------------------------------------
    // diag_ratelimit's contract: first 8, then powers of two, with the 1-based ordinal on every
    // line. The ordinal is what makes the last printed line a lower bound on the population; the
    // line COUNT never is, and reading a cap as a rate has cost this project multiple sessions.
    {
        DrawProgramSkipSelector selector;
        selector.configure("0x5006c6a00");
        bool ordinals_dense = true, printed_first_eight = true, quiet_between = true;
        for (uint64_t n = 1; n <= 16; ++n) {
            const auto d = selector.evaluate(kVs, 0, 0);
            if (d.ordinal != n) ordinals_dense = false;
            if (n <= 8 && !d.print) printed_first_eight = false;
            if (n > 8 && n < 16 && d.print) quiet_between = false;
        }
        CHECK(ordinals_dense, "every decline carries a dense 1-based ordinal, printed or not");
        CHECK(printed_first_eight, "the first eight declines of a program print");
        CHECK(quiet_between, "declines 9..15 are suppressed");
        CHECK(selector.evaluate(kVs, 0, 0).ordinal == 17 && selector.skipped_total() == 17,
              "the running total counts suppressed declines too");

        // Per (program, stage) budget: a program declined thousands of times must not exhaust the
        // log before a second, rarer program gets a line at all.
        DrawProgramSkipSelector both;
        both.configure("0x5006c6a00,0x5002af200");
        for (int i = 0; i < 100; ++i) both.evaluate(kVs, 0, 0);
        const auto rare = both.evaluate(0, 0, kPs);
        CHECK(rare.skip && rare.ordinal == 1 && rare.print,
              "a second program's first decline still prints after 100 declines of another");
    }

    // ---- the census that supplies the selector's input ---------------------------------------
    {
        DrawProgramCensus census;
        const auto a = census.observe(kVs, kChain, kPs);
        CHECK(a.print && a.first && a.ordinal == 1 && a.distinct == 1,
              "the first sighting of a program triple prints and is marked NEW");
        const auto b = census.observe(kVs, kChain, kPs);
        CHECK(b.print && !b.first && b.ordinal == 2 && b.distinct == 1,
              "a recurrence is not NEW and does not grow the distinct count");
        const auto c = census.observe(kVs, kChain, 0x400002000ull);
        CHECK(c.print && c.first && c.ordinal == 1 && c.distinct == 2,
              "the SAME vertex program with a different pixel program is a distinct triple");
        for (int i = 0; i < 5; ++i) census.observe(kVs, kChain, kPs);   // ordinals 3..7
        const auto eight = census.observe(kVs, kChain, kPs);
        CHECK(eight.ordinal == 8 && eight.print, "a recurrence prints again at a power of two");
        const auto nine = census.observe(kVs, kChain, kPs);
        CHECK(nine.ordinal == 9 && !nine.print, "an ordinary recurrence is suppressed");
        CHECK(census.distinct_programs() == 2,
              "the census reports how many distinct program triples it has seen");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
