// #2666: a shared method start must report how many raw records could name it, while preserving
// the last (RVA, UTF-8 name) record. Otherwise a plausible backtrace hides ambiguous attribution.
#include "fixtures/test_scratch.h"
#include "host/image/boot_program.hpp"
#include "host/symbols/il2cpp_symbols.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ios>
#include <stdlib.h> // POSIX setenv/unsetenv and Windows _putenv_s declarations
#include <string>

using namespace prosper;
using namespace prosper::il2cpp;

namespace {
constexpr size_t kTiedRecords = 2523;
constexpr char kLastName[] = "\xCE\xA9.Type$$Method";

class Il2cppAmbiguity : public ::testing::Test {
protected:
    void SetUp() override {
        // NOLINTNEXTLINE(concurrency-mt-unsafe) -- this fixture creates no workers.
        if (const char* value = std::getenv("PROSPER_IL2CPP_SYMBOLS")) {
            had_env_ = true;
            previous_env_ = value;
        }
        set_env("");
        clear_symbol_table();
        path = prosper_test::test_scratch_file("ambiguity.symtab");
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out) << "create the owned synthetic ambiguity table";
        out << "prosper-il2cpp-symtab v1 window=0x10 count=2527\n";
        for (size_t i = 0; i < kTiedRecords - 2; ++i)
            out << "100 Tie.Method" << std::setw(4) << std::setfill('0') << i << '\n';
        out << "100 \xC3\xA9.Type$$Method\n100 " << kLastName << '\n'
            << "120 Unique.Next$$Run\n"
            << "200 Duplicate$$Run\n200 Duplicate$$Run\n"
            << "1000 Unique.Last$$Run\n";
        out.close();
        ASSERT_TRUE(out) << "write all 2527 synthetic raw records";
        std::string error;
        ASSERT_TRUE(load_symbol_table(path, &error)) << "load the valid ordered table: " << error;
        ASSERT_EQ(symbol_table_status().count, 2527U) << "preserve every raw table row";
    }
    void TearDown() override {
        clear_symbol_table();
        set_env(had_env_ ? previous_env_ : "");
    }
    static void set_env(const std::string& value) {
        // This pure resolver fixture creates no workers; environment changes stay on its test thread.
#ifdef _WIN32
        _putenv_s("PROSPER_IL2CPP_SYMBOLS", value.c_str());
#else
        if (value.empty()) {
            // NOLINTNEXTLINE(concurrency-mt-unsafe) -- only the test thread accesses the environment.
            unsetenv("PROSPER_IL2CPP_SYMBOLS");
        } else {
            // NOLINTNEXTLINE(concurrency-mt-unsafe) -- only the test thread accesses the environment.
            setenv("PROSPER_IL2CPP_SYMBOLS", value.c_str(), 1);
        }
#endif
    }
    std::string path;

private:
    bool had_env_ = false;
    std::string previous_env_;
};
}   // namespace

TEST_F(Il2cppAmbiguity, LargeUtf8TieKeepsLastChoiceAndReportsAll2523Records) {
    for (uint64_t offset : {0U, 7U, 15U}) {
        SCOPED_TRACE(offset);
        const Resolution result = resolve_rva(0x100 + offset);
        ASSERT_EQ(result.state, ResolveState::Resolved) << "valid large tie resolves";
        EXPECT_EQ(result.name, kLastName) << "last UTF-8 record remains the selected name";
        EXPECT_EQ(result.offset, offset) << "large tie preserves method-relative offset";
        EXPECT_EQ(result.candidate_count, kTiedRecords) << "large tie reports 2523 raw candidates";
        const std::string suffix = offset == 0 ? "" : offset == 7 ? "+0x7" : "+0xf";
        EXPECT_EQ(annotation_for_guest_va(BOOT_IL2CPP + 0x100 + offset),
                  std::string(" ") + kLastName + suffix + " (+2522 more at this address)")
            << "runtime annotation exposes the large tie after the existing offset";
    }
}

TEST_F(Il2cppAmbiguity, IdenticalRowsRemainSeparateCandidatesAndUniqueNamesStayUnmarked) {
    const Resolution duplicate = resolve_rva(0x201);
    ASSERT_EQ(duplicate.state, ResolveState::Resolved) << "identical duplicate rows resolve";
    EXPECT_EQ(duplicate.name, "Duplicate$$Run") << "identical duplicate rows preserve the name";
    EXPECT_EQ(duplicate.offset, 1U) << "identical duplicate rows preserve the offset";
    EXPECT_EQ(duplicate.candidate_count, 2U) << "identical rows count as two raw candidates";
    EXPECT_EQ(annotation_for_guest_va(BOOT_IL2CPP + 0x201),
              " Duplicate$$Run+0x1 (+1 more at this address)")
        << "identical duplicate rows remain visibly ambiguous";

    const Resolution unique = resolve_rva(0x123);
    ASSERT_EQ(unique.state, ResolveState::Resolved) << "next unique group resolves";
    EXPECT_EQ(unique.name, "Unique.Next$$Run") << "next start replaces the tied group";
    EXPECT_EQ(unique.offset, 3U) << "next unique group preserves the offset";
    EXPECT_EQ(unique.candidate_count, 1U) << "next unique group has one candidate";
    EXPECT_EQ(annotation_for_guest_va(BOOT_IL2CPP + 0x120), " Unique.Next$$Run")
        << "unique exact-start annotation stays byte-identical";
    EXPECT_EQ(annotation_for_guest_va(BOOT_IL2CPP + 0x123), " Unique.Next$$Run+0x3")
        << "unique offset annotation stays byte-identical";
}

TEST_F(Il2cppAmbiguity, WindowAndUnresolvedStatesHaveNoCandidates) {
    for (uint64_t rva : {0xFFU, 0x110U, 0x1010U}) {
        SCOPED_TRACE(rva);
        const Resolution result = resolve_rva(rva);
        EXPECT_EQ(result.state, ResolveState::NoMatch)
            << "lookup respects starts and exclusive window";
        EXPECT_EQ(result.candidate_count, 0U) << "no-match carries no candidate count";
    }
    EXPECT_EQ(annotation_for_guest_va(BOOT_IL2CPP + 0x110), " <no-managed-method>")
        << "expired tied group contributes no ambiguity marker";
    const Resolution outside = resolve_guest_va(BOOT_PSNCORE);
    EXPECT_EQ(outside.state, ResolveState::OutsideModule) << "another module stays outside";
    EXPECT_EQ(outside.candidate_count, 0U) << "outside-module carries no candidate count";

    publish_loaded_module_bounds(LoadedModuleBounds{0x110, 0x1100});
    const Resolution excluded = resolve_rva(0x107);
    EXPECT_EQ(excluded.state, ResolveState::OutsideModule)
        << "known loaded minimum excludes the tied start";
    EXPECT_EQ(excluded.candidate_count, 0U) << "bounds exclusion carries no stale tied count";
    const Resolution inside = resolve_rva(0x123);
    EXPECT_EQ(inside.state, ResolveState::Resolved)
        << "known loaded bounds retain the next valid group";
    EXPECT_EQ(inside.candidate_count, 1U) << "in-bounds next group retains its own candidate count";

    clear_symbol_table();
    EXPECT_EQ(resolve_rva(0x100).state, ResolveState::NotConfigured)
        << "cleared offline session is unconfigured";
    EXPECT_EQ(resolve_rva(0x100).candidate_count, 0U)
        << "unconfigured session carries no candidate count";
    EXPECT_EQ(annotation_for_guest_va(BOOT_IL2CPP + 0x100), "")
        << "unconfigured annotation stays empty";
    std::ofstream invalid(path, std::ios::binary | std::ios::trunc);
    invalid << "not a symbol table\n";
    invalid.close();
    ASSERT_TRUE(invalid) << "write a bounded invalid-table control";
    std::string error;
    ASSERT_FALSE(load_symbol_table(path, &error)) << "invalid table reaches the refusal path";
    EXPECT_EQ(resolve_rva(0x100).state, ResolveState::Unavailable) << "failed table is unavailable";
    EXPECT_EQ(resolve_rva(0x100).candidate_count, 0U)
        << "unavailable table carries no candidate count";
    EXPECT_EQ(annotation_for_guest_va(BOOT_IL2CPP + 0x100), " <il2cpp-symbols-unavailable>")
        << "unavailable annotation retains its existing diagnostic";
}

TEST_F(Il2cppAmbiguity, SingletonSelectionAndWindowRemainCompatible) {
    const Resolution result = resolve_rva(0x1005);
    ASSERT_EQ(result.state, ResolveState::Resolved) << "ordinary singleton resolves";
    EXPECT_EQ(result.name, "Unique.Last$$Run") << "ordinary singleton keeps its selected name";
    EXPECT_EQ(result.offset, 5U) << "ordinary singleton keeps its offset";
    EXPECT_EQ(annotation_for_guest_va(BOOT_IL2CPP + 0x1000), " Unique.Last$$Run")
        << "ordinary exact-start annotation remains compatible";
    EXPECT_EQ(annotation_for_guest_va(BOOT_IL2CPP + 0x1005), " Unique.Last$$Run+0x5")
        << "ordinary offset annotation remains compatible";
    EXPECT_EQ(resolve_rva(0x1010).state, ResolveState::NoMatch)
        << "ordinary singleton still expires at the exclusive window";
}
