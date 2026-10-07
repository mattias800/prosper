// test_validation_mapping_census -- the census arithmetic behind PROSPER_VALIDATION_MAPPING_CENSUS.
//
// The census exists to answer one decision question: what share of fully-compared guest bytes lives in
// private memory, the only kind Windows `GetWriteWatch` can track. The test pins the figure the
// decision turns on, including the failure modes that would quietly mislead it: a byte counted under
// the wrong class, a changed validation not counted as changed, a class outside the enum being dropped
// instead of folded into "untracked", and a compute-side compare (whose outcome is unknown) being
// reported with a `changed` figure it cannot have. The platform classifier is tested separately, under
// tests/host/platform/.
#include "shared/texture/validation_mapping_census.hpp"

#include <gtest/gtest.h>
#include <string>

using prosper::frontend::ValidationMappingCensus;
using prosper::host::MappingClass;

namespace {
int cls(MappingClass c) { return static_cast<int>(c); }
}  // namespace

TEST(ValidationMappingCensus, CountsBytesAndChangesPerClass) {
    ValidationMappingCensus c;
    c.record(cls(MappingClass::Private), 1000, /*changed=*/false);
    c.record(cls(MappingClass::Private), 3000, /*changed=*/true);
    c.record(cls(MappingClass::MappedView), 6000, /*changed=*/false);

    EXPECT_EQ(c.row(cls(MappingClass::Private)).validations, 2u);
    EXPECT_EQ(c.row(cls(MappingClass::Private)).bytes, 4000u);
    EXPECT_EQ(c.row(cls(MappingClass::Private)).changed, 1u);
    EXPECT_EQ(c.row(cls(MappingClass::Private)).changed_bytes, 3000u);
    EXPECT_EQ(c.row(cls(MappingClass::MappedView)).bytes, 6000u);
    EXPECT_EQ(c.row(cls(MappingClass::MappedView)).changed, 0u);
    EXPECT_EQ(c.total_validations(), 3u);
}

TEST(ValidationMappingCensus, CoverableShareIsPrivateBytesOverAllBytes) {
    ValidationMappingCensus c;
    c.record(cls(MappingClass::Private), 2500, false);
    c.record(cls(MappingClass::MappedView), 7500, false);
    const std::string text = c.format("renderer", true);
    EXPECT_NE(text.find("source=renderer GetWriteWatch-coverable (private memory only) = 25.0% of 10000 "
                        "compared bytes"), std::string::npos) << text;
    EXPECT_NE(text.find("class=mapped-view validations=1 bytes=7500 (75.0%"), std::string::npos) << text;
}

TEST(ValidationMappingCensus, UnknownClassFoldsIntoUntrackedRatherThanVanishing) {
    ValidationMappingCensus c;
    c.record(99, 500, false);
    c.record(-1, 500, true);
    EXPECT_EQ(c.row(cls(MappingClass::Untracked)).bytes, 1000u);
    EXPECT_EQ(c.row(cls(MappingClass::Untracked)).changed, 1u);
}

TEST(ValidationMappingCensus, EmptyCensusReportsZeroShareWithoutDividingByZero) {
    ValidationMappingCensus c;
    EXPECT_NE(c.format("renderer", true).find("= 0.0% of 0 compared bytes"), std::string::npos);
}

TEST(ValidationMappingCensus, ComputeSourceHasNoChangedFigure) {
    // The compute buffer cache records a compare where it runs, before its verdict exists. Printing a
    // `changed` count for it would claim a measurement nobody made.
    ValidationMappingCensus c;
    c.record(cls(MappingClass::MappedView), 4096, false);
    const std::string compute = c.format("compute", false);
    EXPECT_NE(compute.find("source=compute class=mapped-view validations=1 bytes=4096"), std::string::npos)
        << compute;
    EXPECT_NE(compute.find("changed=n/a"), std::string::npos) << compute;
    EXPECT_EQ(compute.find("changed_bytes="), std::string::npos) << compute;
    // ...and the renderer source, whose outcome is known, does print one.
    EXPECT_NE(c.format("renderer", true).find("changed_bytes="), std::string::npos);
}
