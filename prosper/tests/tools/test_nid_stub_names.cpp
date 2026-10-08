// test_nid_stub_names — the shared NID<->name readers in tools/common/nid_stub_names.hpp.
//
// Covers the flat `NID name` database path added for aerolib.csv / ps5rs (parse_nid_csv_line,
// load_nid_csv, load_nid_names) alongside the existing sprx_dlsym firmware-dump path, so nid_census
// names a firmware's exports from the aggregated community map, not only the per-library dump. Pure
// parsing over scratch files; no game dump, no prosper_core (the header has no prosper_core dep, and
// nid_hash verification of flat names lives in nid_census, not here).
#include "../../tools/common/nid_stub_names.hpp"
#include "fixtures/test_scratch.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using namespace prosper_tools;

namespace {
// Scratch files on the test harness's own root (tests/AGENTS.md rule 6), not the RAM tmpfs. The name
// embeds the gtest case so parallel cases never collide.
fs::path scratch(const std::string& leaf) {
    const std::string tn = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    return prosper_test::test_scratch_path("nidstub_" + tn + "_" + leaf);
}
void write_file(const fs::path& p, const std::string& body) {
    std::ofstream(p) << body;
}
}   // namespace

TEST(NidStubNames, ParseCsvLineHappyPath) {
    std::string nid, name;
    ASSERT_TRUE(parse_nid_csv_line("PI7jIZj4pcE sceRandomGetRandomNumber", &nid, &name));
    EXPECT_EQ(nid, "PI7jIZj4pcE");
    EXPECT_EQ(name, "sceRandomGetRandomNumber");
}

TEST(NidStubNames, ParseCsvLineToleratesWhitespaceAndTabs) {
    std::string nid, name;
    ASSERT_TRUE(parse_nid_csv_line("  ab-cdef0123\tsceFoo  ", &nid, &name));
    EXPECT_EQ(nid, "ab-cdef0123");
    EXPECT_EQ(name, "sceFoo");
}

TEST(NidStubNames, ParseCsvLineRejectsCommentBlankAndNidOnly) {
    std::string nid, name;
    EXPECT_FALSE(parse_nid_csv_line("# libSceFoo", &nid, &name));
    EXPECT_FALSE(parse_nid_csv_line("", &nid, &name));
    EXPECT_FALSE(parse_nid_csv_line("   ", &nid, &name));
    EXPECT_FALSE(parse_nid_csv_line("PI7jIZj4pcE", &nid, &name));   // no name
}

TEST(NidStubNames, ParseCsvLineRejectsMalformedNidAndExtraTokens) {
    std::string nid, name;
    // NID must be exactly 11 chars of the Sony base64 alphabet.
    EXPECT_FALSE(parse_nid_csv_line("short sceA", &nid, &name));   // too short
    EXPECT_FALSE(parse_nid_csv_line("has_underscore0 sceA", &nid, &name));   // '_' not in alphabet
    EXPECT_FALSE(parse_nid_csv_line("0x1234 sceA", &nid, &name));   // too short / not a NID
    // Swapped columns: the name-shaped token is not a valid NID.
    EXPECT_FALSE(parse_nid_csv_line("sceRandomGetRandomNumber PI7jIZj4pcE", &nid, &name));
    // A third token (e.g. a trailing library column or inline comment) is malformed.
    EXPECT_FALSE(parse_nid_csv_line("PI7jIZj4pcE sceFoo libSceX", &nid, &name));
    EXPECT_FALSE(parse_nid_csv_line("PI7jIZj4pcE sceFoo # c", &nid, &name));
}

TEST(NidStubNames, LoadCsvCountsPairsLibCommentSourceAndRejects) {
    const fs::path p = scratch("db.csv");
    write_file(p,
               "# a prose header that is not a lib is skipped, not a library\n"
               "aaaaaaaaaaa nameOne\n"
               "\n"
               "# libSceBar\n"
               "bbbbbbbbbbb sceBarTwo\n"
               "ccccccccccc sceBarThree\n"
               "not_a_nid sceBad\n");   // malformed -> rejected, not a pair
    auto t = load_nid_csv(p.string());
    EXPECT_TRUE(t.dir_ok);
    EXPECT_EQ(t.source, NameSource::FlatDb);
    EXPECT_EQ(t.pairs, 3u);
    EXPECT_EQ(t.rejected, 1u);
    EXPECT_EQ(t.by_nid.at("aaaaaaaaaaa"), "nameOne");
    EXPECT_EQ(t.by_nid.at("bbbbbbbbbbb"), "sceBarTwo");
    // The two pairs after `# libSceBar` are attributed to it; the first precedes any lib line.
    EXPECT_EQ(t.lib_of.count("aaaaaaaaaaa"), 0u);
    EXPECT_EQ(t.lib_of.at("bbbbbbbbbbb"), "libSceBar");
    EXPECT_EQ(t.lib_of.at("ccccccccccc"), "libSceBar");
    fs::remove(p);
}

TEST(NidStubNames, LoadCsvKeepsFirstOnDuplicateAndCountsConflict) {
    const fs::path p = scratch("dup.csv");
    write_file(p,
               "aaaaaaaaaaa first\n"
               "aaaaaaaaaaa second\n"   // conflicting duplicate -> first kept, conflict counted
               "bbbbbbbbbbb same\n"
               "bbbbbbbbbbb same\n");   // identical duplicate -> no conflict
    auto t = load_nid_csv(p.string());
    EXPECT_EQ(t.by_nid.at("aaaaaaaaaaa"), "first");
    EXPECT_EQ(t.conflicts, 1u);
    EXPECT_EQ(t.by_nid.size(), 2u);
    fs::remove(p);
}

TEST(NidStubNames, LoadCsvHandlesCrlf) {
    const fs::path p = scratch("crlf.csv");
    write_file(p, "aaaaaaaaaaa nameOne\r\n# libSceBar\r\nbbbbbbbbbbb sceBarTwo\r\n");
    auto t = load_nid_csv(p.string());
    EXPECT_EQ(t.rejected, 0u);
    EXPECT_EQ(t.by_nid.at("aaaaaaaaaaa"), "nameOne");   // trailing \r trimmed from the name
    EXPECT_EQ(t.by_nid.at("bbbbbbbbbbb"), "sceBarTwo");
    EXPECT_EQ(t.lib_of.at("bbbbbbbbbbb"), "libSceBar");   // \r trimmed from the lib token
    fs::remove(p);
}

TEST(NidStubNames, LoadCsvMissingFileIsNotOk) {
    auto t = load_nid_csv(scratch("does_not_exist.csv").string());
    EXPECT_FALSE(t.dir_ok);
    EXPECT_EQ(t.pairs, 0u);
    EXPECT_EQ(t.source, NameSource::FlatDb);   // the loader that was asked for
}

TEST(NidStubNames, LoadNidNamesDispatchesFileVsDirectory) {
    // FILE -> csv path.
    const fs::path csv = scratch("names.csv");
    write_file(csv, "ddddddddddd sceDisp\n");
    auto tf = load_nid_names(csv.string());
    EXPECT_TRUE(tf.dir_ok);
    EXPECT_EQ(tf.source, NameSource::FlatDb);
    EXPECT_EQ(tf.by_nid.at("ddddddddddd"), "sceDisp");

    // DIRECTORY -> sprx_dlsym firmware-dump path.
    const fs::path dir = scratch("dumpdir");
    fs::create_directories(dir);
    std::ofstream(dir / "libSceRandom.c")
        << "  if(sprx_dlsym(__handle, \"PI7jIZj4pcE\", &__ptr_sceRandomGetRandomNumber)) return;\n";
    auto td = load_nid_names(dir.string());
    EXPECT_TRUE(td.dir_ok);
    EXPECT_EQ(td.source, NameSource::FirmwareDump);
    EXPECT_EQ(td.by_nid.at("PI7jIZj4pcE"), "sceRandomGetRandomNumber");
    EXPECT_EQ(td.lib_of.at("PI7jIZj4pcE"), "libSceRandom");

    fs::remove(csv);
    fs::remove_all(dir);
}

TEST(NidStubNames, LoadNidNamesMissingPathIsNotOkAndSourceNone) {
    auto t = load_nid_names(scratch("nope").string());
    EXPECT_FALSE(t.dir_ok);
    EXPECT_EQ(t.source, NameSource::None);   // neither dir nor file -> no loader ran
}
