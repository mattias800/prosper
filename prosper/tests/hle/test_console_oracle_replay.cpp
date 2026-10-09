// test_console_oracle_replay -- prosper's HLE against what a REAL PS5 returned.
//
// tests/data/console_oracle/<family>.golden.tsv holds measurements taken on a console by
// tools/console_oracle: for each call, the return value and the final contents of every buffer
// argument. This test rebuilds the same arguments in host memory, calls the HLE handler registered
// for the function, and diffs the result. It needs no console and no game dump.
//
// A mismatch fails the test unless the case is listed in known_gaps.tsv with a reason -- that file is
// the measured backlog. The ratchet runs both ways: a listed case that now matches also fails, so a
// fixed gap cannot linger in the list.
#include "console_oracle.hpp"
#include "fixtures/test_scratch.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;
namespace co = prosper_test::console_oracle;
using namespace prosper;

namespace {

std::string data_dir() {
    return std::string(PROSPER_TEST_DATA_DIR) + "/console_oracle";
}

std::vector<std::string> families() {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(data_dir(), ec)) {
        const std::string name = e.path().filename().string();
        const std::string suffix = ".golden.tsv";
        if (name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            out.push_back(name.substr(0, name.size() - suffix.size()));
    }
    std::sort(out.begin(), out.end());
    return out;
}

}   // namespace

TEST(ConsoleOracleData, GoldenFilesArePresent) {
    EXPECT_FALSE(families().empty()) << "no *.golden.tsv under " << data_dir();
}

TEST(ConsoleOracleData, KnownGapsNameRealCases) {
    std::map<std::string, co::Gap> gaps;
    std::string err;
    ASSERT_TRUE(co::load_known_gaps(data_dir() + "/known_gaps.tsv", &gaps, &err)) << err;
    std::set<std::string> ids;
    for (const auto& fam : families()) {
        std::vector<co::GoldenCase> cases;
        ASSERT_TRUE(co::load_golden(data_dir() + "/" + fam + ".golden.tsv", &cases, &err)) << err;
        for (const auto& c : cases) ids.insert(c.id);
    }
    for (const auto& [id, gap] : gaps)
        EXPECT_TRUE(ids.count(id)) << "known_gaps.tsv lists '" << id << "' (" << gap.reason
                                   << ") but no golden file has that case";
}

TEST(ConsoleOracleData, KnownGapsSayWhetherTheyNeedFixing) {
    // The loader already rejects a row without a valid action; this prints the backlog's shape so a reader
    // sees at a glance how much is `fix` (wrong, should change), `keep` (deliberate) and `triage` (undecided).
    std::map<std::string, co::Gap> gaps;
    std::string err;
    ASSERT_TRUE(co::load_known_gaps(data_dir() + "/known_gaps.tsv", &gaps, &err)) << err;
    std::map<std::string, size_t> by_verb;
    for (const auto& [id, gap] : gaps) by_verb[co::gap_verb(gap.action)]++;
    std::printf("[oracle] known gaps: %zu fix, %zu keep, %zu triage\n", by_verb["fix"], by_verb["keep"],
                by_verb["triage"]);
    EXPECT_EQ(by_verb["fix"] + by_verb["keep"] + by_verb["triage"], gaps.size());
}

TEST(ConsoleOracleHarness, GapActionsAreAClosedVocabularyWithAnOptionalIssue) {
    for (const char* ok : {"fix", "keep", "triage", "fix:#1", "keep:#4757", "triage:#12345"})
        EXPECT_TRUE(co::valid_gap_action(ok)) << ok;
    for (const char* bad : {"", "Fix", "wontfix", "fix:", "fix:4757", "fix:#", "fix:#12a", "fix #4757",
                            "fix:#4757:x", ":#1", "fix,keep"})
        EXPECT_FALSE(co::valid_gap_action(bad)) << bad;
    EXPECT_EQ(co::gap_verb("fix:#4757"), "fix");
    EXPECT_EQ(co::gap_verb("keep"), "keep");
}

TEST(ConsoleOracleHarness, KnownGapsLoaderRequiresAnActionAndAReason) {
    const fs::path p = prosper_test::test_scratch_path("known_gaps_rows.tsv");
    const auto load = [&](const std::string& body, std::string* err) {
        {
            std::ofstream(p) << body;
        }
        std::map<std::string, co::Gap> m;
        const bool ok = co::load_known_gaps(p.string(), &m, err);
        return std::make_pair(ok, m);
    };
    std::string err;
    auto good = load("# c\n\ncase_a\tfix:#7\tprosper is wrong\ncase_b\tkeep\tdeliberate\n", &err);
    ASSERT_TRUE(good.first) << err;
    EXPECT_EQ(good.second.at("case_a").action, "fix:#7");
    EXPECT_EQ(good.second.at("case_a").reason, "prosper is wrong");
    EXPECT_EQ(good.second.at("case_b").action, "keep");
    // the old two column form, a missing reason, an empty id and an unknown action are all errors
    EXPECT_FALSE(load("case_a\tjust a reason\n", &err).first);
    EXPECT_FALSE(load("case_a\tfix\n", &err).first);
    EXPECT_FALSE(load("\tfix\treason\n", &err).first);
    EXPECT_FALSE(load("case_a\twontfix\treason\n", &err).first);
    EXPECT_NE(err.find("expected fix, keep or triage"), std::string::npos) << err;
    fs::remove(p);
}

// Positive controls for the harness itself: a differential test that cannot fail proves nothing, so
// each arm hand-builds a golden line that SHOULD disagree with prosper and checks it is reported.
namespace {
co::GoldenCase make_case(const std::string& id, const std::string& func, const std::string& args,
                         const std::string& expect, uint64_t ret, const std::string& retn = "-",
                         std::vector<std::string> buffers = {}) {
    co::GoldenCase c;
    c.id = id, c.lib = "-", c.func = func, c.args = args, c.expect = expect, c.status = "ok";
    c.ret = ret, c.retn = retn, c.buffers = std::move(buffers);
    return c;
}
std::string replay_and_compare(const co::GoldenCase& c) {
    register_builtin_hle();
    co::State state;
    co::Outcome o;
    std::string err;
    EXPECT_TRUE(co::replay_case(c, state, &o, &err)) << err;
    return co::compare(c, o);
}
}   // namespace

TEST(ConsoleOracleHarness, AgreesWhenGoldenIsRight) {
    EXPECT_EQ(replay_and_compare(make_case("t", "strlen", "s:hello", "r64", 5)), "");
}

TEST(ConsoleOracleHarness, ReportsAWrongReturnValue) {
    const std::string d = replay_and_compare(make_case("t", "strlen", "s:hello", "r64", 6));
    EXPECT_NE(d.find("ret console=0x0000000000000006 prosper=0x0000000000000005"),
              std::string::npos)
        << d;
}

TEST(ConsoleOracleHarness, ComparesOnlyTheLow32BitsByDefault) {
    // The console's int returns carry garbage-free zero extension; prosper's may sign-extend. Both
    // must compare equal on the low 32 bits, but r64 must see the difference.
    EXPECT_EQ(replay_and_compare(make_case("t", "strlen", "s:hello", "", 0xFFFFFFFF00000005ull)),
              "");
    EXPECT_NE(replay_and_compare(make_case("t", "strlen", "s:hello", "r64", 0xFFFFFFFF00000005ull)),
              "");
}

TEST(ConsoleOracleHarness, ReportsAWrongBufferByte) {
    // strcmp does not write its buffers, so a golden claiming a changed byte must be reported.
    const std::string d = replay_and_compare(
        make_case("t", "strcmp", "in:6162,in:6162", "", 0, "-", {"a0=6163", "a1=6162"}));
    EXPECT_NE(d.find("console a0=6163 / prosper a0=6162"), std::string::npos) << d;
}

TEST(ConsoleOracleHarness, PointerReturnsCompareAsOffsets) {
    // strchr returns a pointer into arg 0; the raw address is meaningless across address spaces.
    EXPECT_EQ(replay_and_compare(
                  make_case("t", "strchr", "s:hello,i:108", "r64,retoff:0", 0xdeadbeef, "ptr+2")),
              "");
    const std::string d = replay_and_compare(
        make_case("t", "strchr", "s:hello,i:108", "r64,retoff:0", 0xdeadbeef, "ptr+3"));
    EXPECT_NE(d.find("retn console=ptr+3 prosper=ptr+2"), std::string::npos) << d;
}

TEST(ConsoleOracleHarness, FlagIntDistinguishesAbsentFromMalformed) {
    EXPECT_EQ(co::flag_int("r64,retoff:2", "retoff:"), 2);
    EXPECT_EQ(co::flag_int("r64", "retoff:"), co::kFlagAbsent);
    for (const char* bad : {"retoff:x", "retoff:", "retoff:-1", "retoff:1x", "retoff: 1"})
        EXPECT_EQ(co::flag_int(bad, "retoff:"), co::kFlagMalformed) << bad;
}

TEST(ConsoleOracleHarness, MalformedRetoffIsRejectedNotReadAsZero) {
    register_builtin_hle();
    co::State state;
    co::Outcome o;
    std::string err;
    EXPECT_FALSE(co::replay_case(make_case("t", "strchr", "s:hello,i:108", "r64,retoff:x", 0),
                                 state, &o, &err));
    EXPECT_NE(err.find("malformed retoff"), std::string::npos) << err;
}

TEST(ConsoleOracleHarness, OutPointerIsReportedRelativeToItsBuffer) {
    // strtol("  -123abc") stops after 6 characters; the endptr out-parameter must read ptr+6.
    EXPECT_EQ(replay_and_compare(make_case("t", "strtol", "s:  -123abc,outptr:0,i:10", "r64",
                                           0xFFFFFFFFFFFFFF85ull, "-", {"a1=ptr+6"})),
              "");
    EXPECT_NE(replay_and_compare(make_case("t", "strtol", "s:  -123abc,outptr:0,i:10", "r64",
                                           0xFFFFFFFFFFFFFF85ull, "-", {"a1=ptr+5"})),
              "");
}

TEST(ConsoleOracleHarness, UnimplementedFunctionIsReportedUnlessNone) {
    EXPECT_EQ(replay_and_compare(make_case("t", "sceNoSuchFunctionAnywhere", "-", "", 0)),
              "not implemented in prosper");
    EXPECT_EQ(replay_and_compare(make_case("t", "sceNoSuchFunctionAnywhere", "-", "none", 0)), "");
}

TEST(ConsoleOracleHarness, Default0ComparesAnUnregisteredFunctionAgainstTheDispatcherDefault) {
    // The dispatcher answers an unregistered import with 0. A probe whose console answer is also 0 agrees; one
    // whose console answer is an error is the false-success class and must be reported with both values.
    EXPECT_EQ(replay_and_compare(
                  make_case("t", "sceNoSuchFunctionAnywhere", "out:8,i:1", "ret,default0", 0)),
              "");
    const std::string d = replay_and_compare(make_case(
        "t", "sceNoSuchFunctionAnywhere", "out:8,i:1", "ret,default0", 0x80020016ull));
    EXPECT_EQ(d, "unregistered: prosper's default answers 0, the console answers 0x0000000080020016");
    // Without the flag an unregistered function stays a plain "not implemented" mismatch even at 0, so the
    // rule is opt-in per case and no existing golden changes meaning.
    EXPECT_EQ(replay_and_compare(make_case("t", "sceNoSuchFunctionAnywhere", "out:8,i:1", "ret", 0)),
              "not implemented in prosper");
}

TEST(ConsoleOracleHarness, Default0DoesNotRelaxARegisteredFunction) {
    // strlen is registered: default0 must not turn a wrong console value into agreement.
    EXPECT_NE(replay_and_compare(make_case("t", "strlen", "s:hello", "ret,default0,r64", 6)), "");
    EXPECT_EQ(replay_and_compare(make_case("t", "strlen", "s:hello", "ret,default0,r64", 5)), "");
}

TEST(ConsoleOracleHarness, Default0HonoursR64) {
    // 0x1_0000_0000 is zero in the low 32 bits but not as a 64 bit value.
    EXPECT_EQ(replay_and_compare(make_case("t", "sceNoSuchFunctionAnywhere", "-", "ret,default0",
                                           0x100000000ull)),
              "");
    EXPECT_NE(replay_and_compare(make_case("t", "sceNoSuchFunctionAnywhere", "-", "ret,default0,r64",
                                           0x100000000ull)),
              "");
}

TEST(ConsoleOracleHarness, ProbeBaselineParsesAndRejectsMalformedRows) {
    const fs::path p = prosper_test::test_scratch_path("probe_baseline_rows.tsv");
    co::ProbeBaseline m;
    std::string err;
    {
        std::ofstream(p) << "# comment\n\nfam\tcase_a\t0123456789abcdef\nfam\tcase_b\tfedcba9876543210\n";
    }
    ASSERT_TRUE(co::load_probe_baseline(p.string(), &m, &err)) << err;
    EXPECT_EQ(m.at("fam").size(), 2u);
    EXPECT_EQ(m.at("fam").at("case_b"), "fedcba9876543210");
    for (const char* bad : {"fam\tcase\n", "fam\tcase\tshort\n", "\tcase\t0123456789abcdef\n",
                            "fam\t\t0123456789abcdef\n", "fam\tcase\t0123456789abcdef\textra\n",
                            "fam\tc\t0123456789abcdef\nfam\tc\t0123456789abcdef\n"}) {
        {
            std::ofstream(p) << bad;
        }
        co::ProbeBaseline n;
        EXPECT_FALSE(co::load_probe_baseline(p.string(), &n, &err)) << bad;
    }
    fs::remove(p);
}

TEST(ConsoleOracleHarness, ProbeSignatureSeesTheDwordCountButNotThePayload) {
    const std::string a = "console a0=3:aabbcc / prosper a0=7:bbddee; ";
    EXPECT_EQ(co::probe_signature(a).size(), 16u);
    // Payload bytes are prosper's own packets (and may carry host addresses): not part of the signature.
    EXPECT_EQ(co::probe_signature(a), co::probe_signature("console a0=3:112233 / prosper a0=7:99; "));
    // The dword count is the ABI contract: a builder growing by one dword changes it.
    EXPECT_NE(co::probe_signature(a), co::probe_signature("console a0=3:aabbcc / prosper a0=8:bbddee; "));
    // So does anything outside a buffer report, such as a return value.
    EXPECT_NE(co::probe_signature("ret console=0x0 prosper=0x1; "), co::probe_signature("ret console=0x0 prosper=0x2; "));
    EXPECT_EQ(co::probe_signature_text("x a12=345:ab12 y"), "x a12=345:* y");
}

TEST(ConsoleOracleHarness, UseTokenReadsAnEarlierCasesBuffer) {
    register_builtin_hle();
    co::State state;
    co::Outcome o;
    std::string err;
    // Case one writes a tick into an 8 byte out buffer; case two reads it back through use:.
    ASSERT_TRUE(co::replay_case(make_case("first", "sceRtcTickAddDays", "out:8,p64:100,i:0", "", 0),
                                state, &o, &err))
        << err;
    ASSERT_TRUE(co::replay_case(make_case("second", "sceRtcCompareTick", "p64:100,p64:100", "", 0),
                                state, &o, &err))
        << err;
    co::Arg a;
    ASSERT_TRUE(co::parse_arg("use:first.0.0.8", state, &a, &err)) << err;
    EXPECT_EQ(a.value, 100u);
    EXPECT_FALSE(co::parse_arg("use:nosuchcase.0.0.8", state, &a, &err));
    EXPECT_FALSE(co::parse_arg("use:first.0.4.8", state, &a, &err));   // past the buffer's end
}

TEST(ConsoleOracleHarness, MalformedTokensAreRejected) {
    co::State state;
    co::Arg a;
    std::string err;
    EXPECT_FALSE(co::parse_arg("in:abc", state, &a, &err));   // odd hex digits
    EXPECT_FALSE(co::parse_arg("out:0", state, &a, &err));   // empty out buffer
    EXPECT_FALSE(
        co::parse_arg("outptr:9", state, &a, &err));   // reference past the 6 argument slots
    EXPECT_FALSE(co::parse_arg("bogus:1", state, &a, &err));
    EXPECT_FALSE(co::parse_arg("nocolon", state, &a, &err));
}

TEST(ConsoleOracleHarness, DcbTokenBuildsADescriptorOverAFreshBuffer) {
    co::State state;
    co::Arg a;
    std::string err;
    ASSERT_TRUE(co::parse_arg("dcb:16", state, &a, &err)) << err;
    EXPECT_EQ(a.kind, co::Arg::Kind::Dcb);
    EXPECT_EQ(a.len, 64u);
    ASSERT_EQ(a.aux.size(), co::kDcbStructBytes);
    uint64_t bottom = 0, top = 0, up = 0, down = 0;
    std::memcpy(&bottom, a.aux.data() + 0x00, 8);
    std::memcpy(&top, a.aux.data() + 0x08, 8);
    std::memcpy(&up, a.aux.data() + 0x10, 8);
    std::memcpy(&down, a.aux.data() + 0x18, 8);
    EXPECT_EQ(bottom, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(a.buf.data())));
    EXPECT_EQ(top, bottom + 64);
    EXPECT_EQ(up, bottom);
    EXPECT_EQ(down, top);
    EXPECT_EQ(a.value, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(a.aux.data())));
    for (size_t i = 0; i < a.len; i++) EXPECT_EQ(a.buf[i], co::kSentinel) << "dword buffer byte " << i;
    for (const char* bad : {"dcb:1", "dcb:1025", "dcb:0", "dcb:x", "dcb:"})
        EXPECT_FALSE(co::parse_arg(bad, state, &a, &err)) << bad;
}

TEST(ConsoleOracleHarness, DcbBuilderEmissionIsReportedAsTheDwordsWritten) {
    // sceAgcDcbSetIndexSize appends one packet to the descriptor's buffer and returns its start. The report
    // must say how many dwords it wrote and show exactly those, and the rest of the buffer must stay
    // sentinel -- the property that makes an unwritten tail visible.
    register_builtin_hle();
    co::State state;
    co::Outcome o;
    std::string err;
    const co::GoldenCase c = make_case("t", "sceAgcDcbSetIndexSize", "dcb:8,i:1,i:0", "r64,retoff:0", 0);
    ASSERT_TRUE(co::replay_case(c, state, &o, &err)) << err;
    ASSERT_TRUE(o.implemented);
    EXPECT_EQ(o.retn, "ptr+0");
    ASSERT_EQ(o.buffers.size(), 1u);
    const std::string& rep = o.buffers[0];
    ASSERT_EQ(rep.compare(0, 3, "a0="), 0) << rep;
    const size_t colon = rep.find(':');
    ASSERT_NE(colon, std::string::npos) << rep;
    const size_t written = static_cast<size_t>(std::stoul(rep.substr(3, colon - 3)));
    EXPECT_GE(written, 2u);
    EXPECT_LE(written, 8u);
    EXPECT_EQ(rep.size() - colon - 1, written * 8);   // 8 hex digits per dword
    const co::Arg& x = state.at("t")[0];
    for (size_t i = written * 4; i < x.len; i++) EXPECT_EQ(x.buf[i], co::kSentinel) << "byte " << i;
}

TEST(ConsoleOracleHarness, DcbBuilderThatCannotFitWritesNothingAndSaysSo) {
    // A two dword buffer cannot hold the packet: the builder must refuse, the cursor must not move, and
    // the report is "0:" with no dwords -- not a partial packet.
    register_builtin_hle();
    co::State state;
    co::Outcome o;
    std::string err;
    const co::GoldenCase c = make_case("t", "sceAgcDcbDrawIndexAuto", "dcb:2,i:3,i:0", "r64,retoff:0", 0);
    ASSERT_TRUE(co::replay_case(c, state, &o, &err)) << err;
    ASSERT_TRUE(o.implemented);
    EXPECT_EQ(o.ret, 0u);
    ASSERT_EQ(o.buffers.size(), 1u);
    EXPECT_EQ(o.buffers[0], "a0=0:");
}

class ConsoleOracleReplay : public ::testing::TestWithParam<std::string> {};

TEST_P(ConsoleOracleReplay, MatchesConsole) {
    register_builtin_hle();
    std::string err;
    std::vector<co::GoldenCase> cases;
    ASSERT_TRUE(co::load_golden(data_dir() + "/" + GetParam() + ".golden.tsv", &cases, &err))
        << err;
    std::map<std::string, co::Gap> gaps;
    ASSERT_TRUE(co::load_known_gaps(data_dir() + "/known_gaps.tsv", &gaps, &err)) << err;

    // A `probe_*` family, or any family with rows in probe_baseline.tsv, is a measured INVENTORY (a generated
    // set of calls, e.g. every command-buffer builder): its disagreements are leads rather than reviewed gaps.
    // They are gated by case id AND by a signature of the difference, both ways: a difference that is not listed
    // fails, a listed case that now matches fails (lower the baseline), and a listed case whose difference
    // changed fails -- which a bare count cannot see (a builder growing a dword, one difference swapped for
    // another). Adding a row is a visible line in the diff.
    co::ProbeBaseline baselines;
    ASSERT_TRUE(co::load_probe_baseline(data_dir() + "/probe_baseline.tsv", &baselines, &err)) << err;
    const bool probe = GetParam().rfind("probe_", 0) == 0 || baselines.count(GetParam()) != 0;
    const std::map<std::string, std::string> empty_rows;
    const auto rows_it = baselines.find(GetParam());
    const std::map<std::string, std::string>& rows = rows_it == baselines.end() ? empty_rows : rows_it->second;
    size_t differing = 0;
    std::set<std::string> seen_differing;

    co::State state;
    size_t matched = 0, expected_gaps = 0, skipped = 0;
    for (const auto& c : cases) {
        if (c.status != "ok") {
            // The console could not run this case (missing library, a fault, a bad spec): there is
            // no measurement to compare against, so say so rather than treat it as agreement.
            ADD_FAILURE() << c.id << ": console status '" << c.status
                          << "' -- recapture it or remove the case";
            skipped++;
            continue;
        }
        co::Outcome outcome;
        if (!co::replay_case(c, state, &outcome, &err)) {
            ADD_FAILURE() << c.id << ": malformed case: " << err;
            continue;
        }
        const std::string diff = co::compare(c, outcome);
        const auto gap = gaps.find(c.id);
        if (diff.empty() && gap == gaps.end()) {
            matched++;
        } else if (diff.empty()) {
            ADD_FAILURE() << c.id << ": now matches the console -- remove it from known_gaps.tsv";
        } else if (gap != gaps.end()) {
            expected_gaps++;
        } else if (probe) {
            differing++;
            seen_differing.insert(c.id);
            const std::string sig = co::probe_signature(diff);
            // `[probe-row]` is the line to put in probe_baseline.tsv once the difference has been reviewed.
            std::printf("[probe-row] %s\t%s\t%s\n", GetParam().c_str(), c.id.c_str(), sig.c_str());
            const auto row = rows.find(c.id);
            if (row == rows.end())
                ADD_FAILURE() << c.id << " (" << c.func << "): differs from the console and is not in probe_baseline.tsv: "
                              << diff;
            else if (row->second != sig)
                ADD_FAILURE() << c.id << " (" << c.func << "): its difference changed from the baselined one (signature "
                              << row->second << " -> " << sig << "); review it, then update the row: " << diff;
        } else {
            ADD_FAILURE() << c.id << " (" << c.func << "): " << diff;
        }
    }
    RecordProperty("matched", static_cast<int>(matched));
    RecordProperty("known_gaps", static_cast<int>(expected_gaps));
    std::printf("[oracle] %s: %zu match the console, %zu known gaps, %zu differ (probe), %zu unmeasured\n",
                GetParam().c_str(), matched, expected_gaps, differing, skipped);
    if (probe) {
        for (const auto& [id, sig] : rows)
            if (!seen_differing.count(id))
                ADD_FAILURE() << id << ": listed in probe_baseline.tsv but no longer differs from the console (or is gone "
                              << "from the golden) -- progress: remove the row";
    }
}

INSTANTIATE_TEST_SUITE_P(Golden, ConsoleOracleReplay, ::testing::ValuesIn(families()),
                         [](const ::testing::TestParamInfo<std::string>& i) { return i.param; });
