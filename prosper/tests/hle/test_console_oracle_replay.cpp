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
    std::map<std::string, std::string> gaps;
    std::string err;
    ASSERT_TRUE(co::load_known_gaps(data_dir() + "/known_gaps.tsv", &gaps, &err)) << err;
    std::set<std::string> ids;
    for (const auto& fam : families()) {
        std::vector<co::GoldenCase> cases;
        ASSERT_TRUE(co::load_golden(data_dir() + "/" + fam + ".golden.tsv", &cases, &err)) << err;
        for (const auto& c : cases) ids.insert(c.id);
    }
    for (const auto& [id, reason] : gaps)
        EXPECT_TRUE(ids.count(id)) << "known_gaps.tsv lists '" << id << "' (" << reason
                                   << ") but no golden file has that case";
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

class ConsoleOracleReplay : public ::testing::TestWithParam<std::string> {};

TEST_P(ConsoleOracleReplay, MatchesConsole) {
    register_builtin_hle();
    std::string err;
    std::vector<co::GoldenCase> cases;
    ASSERT_TRUE(co::load_golden(data_dir() + "/" + GetParam() + ".golden.tsv", &cases, &err))
        << err;
    std::map<std::string, std::string> gaps;
    ASSERT_TRUE(co::load_known_gaps(data_dir() + "/known_gaps.tsv", &gaps, &err)) << err;

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
        } else {
            ADD_FAILURE() << c.id << " (" << c.func << "): " << diff;
        }
    }
    RecordProperty("matched", static_cast<int>(matched));
    RecordProperty("known_gaps", static_cast<int>(expected_gaps));
    std::printf("[oracle] %s: %zu match the console, %zu known gaps, %zu unmeasured\n",
                GetParam().c_str(), matched, expected_gaps, skipped);
}

INSTANTIATE_TEST_SUITE_P(Golden, ConsoleOracleReplay, ::testing::ValuesIn(families()),
                         [](const ::testing::TestParamInfo<std::string>& i) { return i.param; });
