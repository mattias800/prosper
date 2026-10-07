// test_validation_mapping_census -- the census arithmetic behind the [validation-census] exit summary.
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

    EXPECT_EQ(c.row(cls(MappingClass::Private)).validations, 2U);
    EXPECT_EQ(c.row(cls(MappingClass::Private)).bytes, 4000U);
    EXPECT_EQ(c.row(cls(MappingClass::Private)).changed, 1U);
    EXPECT_EQ(c.row(cls(MappingClass::Private)).changed_bytes, 3000U);
    EXPECT_EQ(c.row(cls(MappingClass::MappedView)).bytes, 6000U);
    EXPECT_EQ(c.row(cls(MappingClass::MappedView)).changed, 0U);
    EXPECT_EQ(c.total_validations(), 3U);
}

TEST(ValidationMappingCensus, CoverableShareIsPrivateBytesOverAllBytes) {
    ValidationMappingCensus c;
    c.record(cls(MappingClass::Private), 2500, false);
    c.record(cls(MappingClass::MappedView), 7500, false);
    const std::string text = c.format("renderer", true);
    if (prosper::host::host_mapping_classification_informative()) {
        EXPECT_NE(text.find("source=renderer GetWriteWatch-coverable (private memory only) = 25.0% of 10000 "
                            "compared bytes"), std::string::npos) << text;
        EXPECT_NE(text.find("class=mapped-view validations=1 bytes=7500 (75.0%"), std::string::npos) << text;
    } else {
        // Where the classifier is a constant the class rows and the coverable share are not printed.
        EXPECT_EQ(text.find("class="), std::string::npos) << text;
        EXPECT_EQ(text.find("GetWriteWatch-coverable"), std::string::npos) << text;
        EXPECT_NE(text.find(" sizes:"), std::string::npos) << text;
    }
}

TEST(ValidationMappingCensus, UnknownClassFoldsIntoUntrackedRatherThanVanishing) {
    ValidationMappingCensus c;
    c.record(99, 500, false);
    c.record(-1, 500, true);
    EXPECT_EQ(c.row(cls(MappingClass::Untracked)).bytes, 1000U);
    EXPECT_EQ(c.row(cls(MappingClass::Untracked)).changed, 1U);
}

TEST(ValidationMappingCensus, EmptyCensusReportsZeroShareWithoutDividingByZero) {
    ValidationMappingCensus c;
    const std::string text = c.format("renderer", true);
    if (prosper::host::host_mapping_classification_informative())
        EXPECT_NE(text.find("= 0.0% of 0 compared bytes"), std::string::npos);
    EXPECT_NE(text.find(" sizes: <64KiB n=0"), std::string::npos);
}

TEST(ValidationMappingCensus, ComputeSourceHasNoChangedFigure) {
    using enum prosper::host::MappingClass;
    // The compute buffer cache records a compare where it runs, before its verdict exists. Printing a
    // `changed` count for it would claim a measurement nobody made.
    ValidationMappingCensus c;
    c.record(cls(MappedView), 4096, false);
    const std::string compute = c.format("compute", false);
    if (prosper::host::host_mapping_classification_informative()) {
        EXPECT_NE(compute.find("source=compute class=mapped-view validations=1 bytes=4096"), std::string::npos)
            << compute;
        EXPECT_NE(compute.find("changed=n/a"), std::string::npos) << compute;
        EXPECT_EQ(compute.find("changed_bytes="), std::string::npos) << compute;
        // ...and the renderer source, whose outcome is known, does print one.
        EXPECT_NE(c.format("renderer", true).find("changed_bytes="), std::string::npos);
    }
}

TEST(ValidationMappingCensus, SizeHistogramSeparatesFewBigRangesFromManySmallOnes) {
    // 66 GB can be a handful of 43 MiB windows or thousands of small compares; the histogram says which.
    ValidationMappingCensus c;
    c.record(cls(MappingClass::MappedView), 4096, false);                 // <64 KiB
    c.record(cls(MappingClass::MappedView), 200U << 10, false);           // <1 MiB
    c.record(cls(MappingClass::MappedView), 8U << 20, false);             // <16 MiB
    c.record(cls(MappingClass::MappedView), 45088768, false);             // the 43 MiB window: <64 MiB
    c.record(cls(MappingClass::MappedView), 45088768, false);
    c.record(cls(MappingClass::MappedView), 100ULL << 20, false);         // >=64 MiB
    EXPECT_EQ(c.size_validations(0), 1U);
    EXPECT_EQ(c.size_validations(1), 1U);
    EXPECT_EQ(c.size_validations(2), 1U);
    EXPECT_EQ(c.size_validations(3), 2U);
    EXPECT_EQ(c.size_validations(4), 1U);
    EXPECT_EQ(c.size_bytes(3), 2U * 45088768U);
    // Boundaries belong to the larger bucket.
    EXPECT_EQ(ValidationMappingCensus::size_bucket(64U << 10), 1);
    EXPECT_EQ(ValidationMappingCensus::size_bucket((64U << 10) - 1), 0);
    EXPECT_NE(c.format("compute", false).find(" sizes: <64KiB n=1"), std::string::npos);
    EXPECT_NE(c.format("compute", false).find("<64MiB n=2 0.09GB"), std::string::npos);
    // Every edge belongs to the larger bucket; test each edge from both sides so a flipped comparison
    // (`<` for `<=`) at any of them goes red.
    EXPECT_EQ(ValidationMappingCensus::size_bucket((1ULL << 20) - 1), 1);
    EXPECT_EQ(ValidationMappingCensus::size_bucket(1ULL << 20), 2);
    EXPECT_EQ(ValidationMappingCensus::size_bucket((16ULL << 20) - 1), 2);
    EXPECT_EQ(ValidationMappingCensus::size_bucket(16ULL << 20), 3);
    EXPECT_EQ(ValidationMappingCensus::size_bucket((64ULL << 20) - 1), 3);
    EXPECT_EQ(ValidationMappingCensus::size_bucket(64ULL << 20), 4);
}

namespace {
using prosper::frontend::compute_mapping_census;
using prosper::frontend::kMinClassifiedBytes;
using prosper::frontend::renderer_mapping_census;

// The census singletons are process-global, so every assertion is on a delta.
struct Snapshot {
    uint64_t small = 0;
    uint64_t classified = 0;
    uint64_t total = 0;
    static Snapshot of(const ValidationMappingCensus& c) {
        Snapshot snap;
        snap.small = c.row(ValidationMappingCensus::kSmall).validations;
        snap.total = c.total_validations();
        snap.classified = snap.total - snap.small;
        return snap;
    }
};
}  // namespace

TEST(ValidationMappingCensus, SmallThresholdIsExactlySixtyFourKiB) {
    // The threshold is the only thing that bounds the always-on cost (one VirtualQuery per classified
    // compare), so pin its value, not just a lower bound: a larger value silently hides ranges from the
    // classification, a smaller one pays a syscall beside a compare it costs as much as.
    EXPECT_EQ(kMinClassifiedBytes, 64ULL << 10);
    EXPECT_EQ(kMinClassifiedBytes, static_cast<uint64_t>(1) << 16);
}

TEST(ValidationMappingCensus, ComputeEntryPointClassifiesAtTheThresholdNotBelow) {
    char byte = 0;
    const Snapshot before = Snapshot::of(compute_mapping_census());
    prosper::frontend::note_compute_compare_mapping(&byte, kMinClassifiedBytes - 1);
    const Snapshot below = Snapshot::of(compute_mapping_census());
    prosper::frontend::note_compute_compare_mapping(&byte, kMinClassifiedBytes);
    const Snapshot at = Snapshot::of(compute_mapping_census());

    EXPECT_EQ(below.small - before.small, 1U) << "one byte under the threshold is counted, not classified";
    EXPECT_EQ(below.classified - before.classified, 0U);
    EXPECT_EQ(at.small - below.small, 0U) << "a range of exactly the threshold is classified";
    EXPECT_EQ(at.classified - below.classified, 1U);
}

TEST(ValidationMappingCensus, RendererEntryPointClassifiesAtTheThresholdNotBelow) {
    char byte = 0;
    const uint64_t address = reinterpret_cast<uintptr_t>(&byte);
    const Snapshot before = Snapshot::of(renderer_mapping_census());
    prosper::frontend::note_validation_mapping(address, kMinClassifiedBytes - 1, false);
    const Snapshot below = Snapshot::of(renderer_mapping_census());
    prosper::frontend::note_validation_mapping(address, kMinClassifiedBytes, false);
    const Snapshot at = Snapshot::of(renderer_mapping_census());

    EXPECT_EQ(below.small - before.small, 1U);
    EXPECT_EQ(below.classified - before.classified, 0U);
    EXPECT_EQ(at.small - below.small, 0U);
    EXPECT_EQ(at.classified - below.classified, 1U);
}

TEST(ValidationMappingCensus, SmallClassHasItsOwnRowAndLine) {
    // Plumbing for the small row itself: a fifth class needs a name, or its line would be missing.
    ValidationMappingCensus c;
    c.record(ValidationMappingCensus::kSmall, 16, false);
    c.record(cls(MappingClass::MappedView), 128U << 10, false);
    EXPECT_EQ(c.row(ValidationMappingCensus::kSmall).validations, 1U);
    EXPECT_EQ(c.size_validations(0), 1U);
    EXPECT_EQ(c.total_validations(), 2U);
    if (prosper::host::host_mapping_classification_informative())
        EXPECT_NE(c.format("renderer", true).find("class=small-unclassified validations=1 bytes=16"),
                  std::string::npos);
}
