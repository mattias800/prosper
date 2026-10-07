// test_validation_mapping_census -- the census arithmetic behind PROSPER_VALIDATION_MAPPING_CENSUS.
//
// The census exists to answer one decision question: what share of fully-compared guest bytes lives in
// private memory, the only kind Windows `GetWriteWatch` can track. The test pins the figure the
// decision turns on, including the failure modes that would quietly mislead it: a byte counted under
// the wrong class, a changed validation not counted as changed, and a class outside the enum being
// dropped instead of folded into "untracked".
#include "shared/live/submit_renderer/validation_mapping_census.hpp"

#include <gtest/gtest.h>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

using prosper::frontend::submit_renderer::ValidationMappingCensus;

TEST(ValidationMappingCensus, CountsBytesAndChangesPerClass) {
    ValidationMappingCensus c;
    const int priv = static_cast<int>(prosper::host::MappingClass::Private);
    const int dmem = static_cast<int>(prosper::host::MappingClass::MappedView);
    c.record(priv, 1000, true);
    c.record(priv, 3000, false);
    c.record(dmem, 6000, true);

    EXPECT_EQ(c.row(priv).validations, 2u);
    EXPECT_EQ(c.row(priv).bytes, 4000u);
    EXPECT_EQ(c.row(priv).changed, 1u);
    EXPECT_EQ(c.row(priv).changed_bytes, 3000u);
    EXPECT_EQ(c.row(dmem).bytes, 6000u);
    EXPECT_EQ(c.row(dmem).changed, 0u);
}

TEST(ValidationMappingCensus, CoverableShareIsPrivateBytesOverAllBytes) {
    ValidationMappingCensus c;
    c.record(static_cast<int>(prosper::host::MappingClass::Private), 2500, true);
    c.record(static_cast<int>(prosper::host::MappingClass::MappedView), 7500, true);
    const std::string text = c.format();
    EXPECT_NE(text.find("GetWriteWatch-coverable (private memory only) = 25.0% of 10000 compared bytes"),
              std::string::npos) << text;
    EXPECT_NE(text.find("class=mapped-view validations=1 bytes=7500 (75.0%"), std::string::npos) << text;
}

TEST(ValidationMappingCensus, UnknownClassFoldsIntoUntrackedRatherThanVanishing) {
    ValidationMappingCensus c;
    c.record(99, 500, true);
    c.record(-1, 500, false);
    EXPECT_EQ(c.row(static_cast<int>(prosper::host::MappingClass::Untracked)).bytes, 1000u);
    EXPECT_EQ(c.row(static_cast<int>(prosper::host::MappingClass::Untracked)).changed, 1u);
}

TEST(ValidationMappingCensus, EmptyCensusReportsZeroShareWithoutDividingByZero) {
    ValidationMappingCensus c;
    EXPECT_NE(c.format().find("= 0.0% of 0 compared bytes"), std::string::npos);
}

#ifdef _WIN32
// A hand-built positive control for the classifier itself. The census is only as good as the
// answer to "is this address private or a section view?", and a control drawn from the census's own
// input could only show the plumbing runs. So build one of each kind outside it.
TEST(ValidationMappingCensus, ClassifiesPrivateMemoryAndSectionViewsApart) {
    void* priv = VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    ASSERT_NE(priv, nullptr);
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT, 0, 65536, nullptr);
    ASSERT_NE(section, nullptr);
    void* view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, 65536);
    ASSERT_NE(view, nullptr);

    EXPECT_EQ(static_cast<int>(prosper::host::classify_host_mapping(reinterpret_cast<uintptr_t>(priv))),
              static_cast<int>(prosper::host::MappingClass::Private));
    EXPECT_EQ(static_cast<int>(prosper::host::classify_host_mapping(reinterpret_cast<uintptr_t>(view))),
              static_cast<int>(prosper::host::MappingClass::MappedView));

    UnmapViewOfFile(view);
    CloseHandle(section);
    VirtualFree(priv, 0, MEM_RELEASE);
    // Freed memory is not committed host memory any more.
    EXPECT_EQ(static_cast<int>(prosper::host::classify_host_mapping(reinterpret_cast<uintptr_t>(priv))),
              static_cast<int>(prosper::host::MappingClass::Untracked));
}
#endif
