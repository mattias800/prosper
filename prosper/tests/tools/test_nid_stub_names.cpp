// test_nid_stub_names — the shared NID<->name readers in tools/common/nid_stub_names.hpp.
//
// Covers the flat `NID name` database path added for aerolib.csv / ps5rs (parse_nid_csv_line,
// load_nid_csv, load_nid_names) alongside the existing sprx_dlsym firmware-dump path, so nid_census
// and self_dump can name a firmware's exports from the aggregated community map, not only the
// per-library dump. Pure parsing over temp files; no game dump, no prosper_core.
#include "../../tools/common/nid_stub_names.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using namespace prosper_tools;

namespace {
fs::path write_temp(const std::string& name, const std::string& body) {
    fs::path p = fs::temp_directory_path() /
                 ("nidstub_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                  "_" + name);
    std::ofstream(p) << body;
    return p;
}
}  // namespace

TEST(NidStubNames, ParseCsvLineHappyPath) {
    std::string nid, name;
    ASSERT_TRUE(parse_nid_csv_line("PI7jIZj4pcE sceRandomGetRandomNumber", &nid, &name));
    EXPECT_EQ(nid, "PI7jIZj4pcE");
    EXPECT_EQ(name, "sceRandomGetRandomNumber");
}

TEST(NidStubNames, ParseCsvLineToleratesWhitespaceAndTabs) {
    std::string nid, name;
    ASSERT_TRUE(parse_nid_csv_line("  ab-cd_1\tsceFoo  ", &nid, &name));
    EXPECT_EQ(nid, "ab-cd_1");
    EXPECT_EQ(name, "sceFoo");
}

TEST(NidStubNames, ParseCsvLineRejectsCommentBlankAndNidOnly) {
    std::string nid, name;
    EXPECT_FALSE(parse_nid_csv_line("# libSceFoo", &nid, &name));
    EXPECT_FALSE(parse_nid_csv_line("", &nid, &name));
    EXPECT_FALSE(parse_nid_csv_line("   ", &nid, &name));
    EXPECT_FALSE(parse_nid_csv_line("PI7jIZj4pcE", &nid, &name));  // no name
}

TEST(NidStubNames, LoadCsvCountsPairsAndTracksLibComment) {
    fs::path p = write_temp("db.csv",
        "# header text that is not a lib is still skipped\n"
        "aaaaaaaaaaa nameOne\n"
        "\n"
        "# libSceBar\n"
        "bbbbbbbbbbb sceBarTwo\n"
        "ccccccccccc sceBarThree\n");
    auto t = load_nid_csv(p.string());
    EXPECT_TRUE(t.dir_ok);
    EXPECT_EQ(t.pairs, 3u);
    EXPECT_EQ(t.by_nid.at("aaaaaaaaaaa"), "nameOne");
    EXPECT_EQ(t.by_nid.at("bbbbbbbbbbb"), "sceBarTwo");
    // The two pairs after `# libSceBar` are attributed to it; the first pair precedes any lib line.
    EXPECT_EQ(t.lib_of.count("aaaaaaaaaaa"), 0u);
    EXPECT_EQ(t.lib_of.at("bbbbbbbbbbb"), "libSceBar");
    EXPECT_EQ(t.lib_of.at("ccccccccccc"), "libSceBar");
    fs::remove(p);
}

TEST(NidStubNames, LoadCsvMissingFileIsNotOk) {
    auto t = load_nid_csv((fs::temp_directory_path() / "does_not_exist_nidstub.csv").string());
    EXPECT_FALSE(t.dir_ok);
    EXPECT_EQ(t.pairs, 0u);
}

TEST(NidStubNames, LoadNidNamesDispatchesFileVsDirectory) {
    // FILE -> csv path.
    fs::path csv = write_temp("names.csv", "ddddddddddd sceDisp\n");
    auto tf = load_nid_names(csv.string());
    EXPECT_TRUE(tf.dir_ok);
    EXPECT_EQ(tf.by_nid.at("ddddddddddd"), "sceDisp");

    // DIRECTORY -> sprx_dlsym firmware-dump path.
    fs::path dir = fs::temp_directory_path() / ("nidstub_dir_" +
                   std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    fs::create_directories(dir);
    std::ofstream(dir / "libSceRandom.c")
        << "  if(sprx_dlsym(__handle, \"PI7jIZj4pcE\", &__ptr_sceRandomGetRandomNumber)) return;\n";
    auto td = load_nid_names(dir.string());
    EXPECT_TRUE(td.dir_ok);
    EXPECT_EQ(td.by_nid.at("PI7jIZj4pcE"), "sceRandomGetRandomNumber");
    EXPECT_EQ(td.lib_of.at("PI7jIZj4pcE"), "libSceRandom");

    fs::remove(csv);
    fs::remove_all(dir);
}

TEST(NidStubNames, LoadNidNamesMissingPathIsNotOk) {
    auto t = load_nid_names((fs::temp_directory_path() / "nope_nidstub_xyz").string());
    EXPECT_FALSE(t.dir_ok);
}
