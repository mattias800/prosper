// #4209: a source-associated PS count grants only the present user prefix, never a zero-filled
// physical window or the following system SGPR. Drive the actual shipping preparation consumer.
#include "fragment_entry_prefix_fixture.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace e = prosper::test::entry_prefix;
using prosper::gpu::prepare_fragment_packet_inputs;

TEST(FragmentEntryPrefix, ActualTypedSourceConsumesPreparedWords) {
    // Diagnostic CPU-test-only retention of these ACTUAL emitted modules. Ordinary registered
    // tests create no files; an author supplies a separate owned evidence directory explicitly.
    std::filesystem::path directory;
    if (const char* root = std::getenv("PROSPER_ENTRY_PREFIX_SPV_DIRECTORY"); root && *root) {
        directory = root;
        std::filesystem::create_directories(directory);
    }
    prosper::gpu::reset_float_controls_support_for_test();
    uint32_t evaluations = 0;
    const auto evaluate = [&](const char* name, const prosper::gpu::FragmentPacketPreparation& prepared,
                              uint32_t value) {
        const auto program = prosper::gpu::recompile_fragment_packet(e::packet(prepared));
        EXPECT_FALSE(program.spirv.empty()) << name << ": " << program.rejection;
        if (program.spirv.empty()) return std::vector<uint32_t>{};
        bpermute_oracle::Interpreter vm(program.spirv); // reuse the project's typed SOURCE VM
        ++evaluations;
        const auto actual = vm.run_packet(program.input_words, program.output_words);
        EXPECT_TRUE(vm.error.empty()) << name << ": " << vm.error;
        EXPECT_EQ(actual, e::expected(value)) << name << ": all64 complete raw EXP records";
        if (!directory.empty()) {
            std::ofstream file(directory / (std::string(name) + ".spv"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(program.spirv.data()),
                       static_cast<std::streamsize>(program.spirv.size() * sizeof(uint32_t)));
            file.close();
            EXPECT_TRUE(bool(file)) << name << ": actual SOURCE retention must fail visibly";
        }
        return actual;
    };
    for (uint32_t count : {0u, 16u, 17u, 31u, 32u}) {
        SCOPED_TRACE(count);
        const uint32_t source = count ? count - 1 : 128; // inline zero is not an entry seed
        const auto in = e::inputs(count, source);
        const auto prepared = prepare_fragment_packet_inputs(in, true);
        ASSERT_TRUE(prepared->user_sgpr_count_available);
        const auto name = "count" + std::to_string(count);
        (void)evaluate(name.c_str(), *prepared, count ? in->entry.user_data[source] : 0);
    }
    const auto zero = prepare_fragment_packet_inputs(e::inputs(1), true);
    (void)evaluate("present_zero", *zero, 0);
    auto in = e::inputs(17, 16);
    const auto original = prepare_fragment_packet_inputs(in, true);
    const uint32_t first_value = in->entry.user_data[16];
    in->entry.user_data[16] ^= 0x80000001u;
    const uint32_t next_value = in->entry.user_data[16];
    const auto changed = prepare_fragment_packet_inputs(in, true);
    const auto first = evaluate("owned_original", *original, first_value);
    const auto next = evaluate("owned_changed", *changed, next_value);
    EXPECT_NE(next, e::expected(first_value)) << "old oracle detects a real producing-word change";
    EXPECT_NE(first, next) << "prior preparation keeps its original owned raw word";
    EXPECT_EQ(evaluations, 8u) << "all eight actual SOURCE rails must execute";
    std::fprintf(stderr, "[entry-prefix-source] evaluation_attempts=%u expected=8; no GPU/raster admission\n",
                 evaluations);
}

TEST(FragmentEntryPrefix, HardwareCountAndSystemBoundary) {
    for (uint32_t count : {0u, 16u, 17u, 31u, 32u}) {
        SCOPED_TRACE(count);
        const auto in = e::inputs(count);
        const auto p = prepare_fragment_packet_inputs(in, true);
        ASSERT_TRUE(p->user_sgpr_count_available) << "known zero count differs from unknown count";
        EXPECT_EQ(p->user_sgpr_count, count);
        EXPECT_EQ(p->missing_user_sgprs, 0u);
        ASSERT_EQ(p->initial_user_sgprs.size(), count);
        for (uint32_t reg = 0; reg < count; ++reg) {
            EXPECT_EQ(p->initial_user_sgprs[reg].first, reg);
            EXPECT_EQ(p->initial_user_sgprs[reg].second, in->entry.user_data[reg]);
        }
        EXPECT_FALSE(p->ready) << "a true user s16 is not M0/parameter or guest-wave authority";
        EXPECT_TRUE(e::gap(*p, "packet-entry-system-sgprs-and-m0-unproved"));
        EXPECT_TRUE(e::gap(*p, "packet-entry-vgpr-and-mask-abi-unproved"));
        EXPECT_TRUE(e::gap(*p, "packet-logical64-composition-unproved"));
        EXPECT_TRUE(e::gap(*p, "packet-ordered-export-commit-unimplemented"));
    }
}

TEST(FragmentEntryPrefix, MissingWordIsNotPresentZero) {
    auto in = e::inputs(17, 16);
    const auto present = prepare_fragment_packet_inputs(in, true);
    ASSERT_EQ(present->initial_user_sgprs.size(), 17u);
    ASSERT_EQ(present->initial_user_sgprs.front(), std::make_pair(0u, 0u));
    in->entry.user_data_available &= ~(1u << 16);
    in->entry.user_data[16] = 0;
    const auto missing = prepare_fragment_packet_inputs(in, true);
    EXPECT_EQ(missing->user_sgpr_count, 17u);
    EXPECT_EQ(missing->missing_user_sgprs, 1u << 16);
    EXPECT_EQ(missing->initial_user_sgprs.size(), 16u);
    EXPECT_TRUE(e::gap(*missing, "packet-entry-user-sgpr-words-unavailable"));
    EXPECT_EQ(present->initial_user_sgprs.back().second, 0x42090110u)
        << "older prepared words are owned, not a later mutable observation";
    const auto positive = prosper::gpu::recompile_fragment_packet(e::packet(*present));
    EXPECT_FALSE(positive.spirv.empty()) << positive.rejection;
    const auto rejected = prosper::gpu::recompile_fragment_packet(e::packet(*missing));
    EXPECT_TRUE(rejected.spirv.empty()) << "the real compiler must not synthesize missing s16";
    EXPECT_EQ(rejected.rejection, "packet-sgpr-read-before-definition");
}

TEST(FragmentEntryPrefix, SystemWordNeverBorrowsPhysicalUserData) {
    const auto p = prepare_fragment_packet_inputs(e::inputs(16, 16), true);
    ASSERT_EQ(p->initial_user_sgprs.size(), 16u);
    const auto rejected = prosper::gpu::recompile_fragment_packet(e::packet(*p));
    EXPECT_TRUE(rejected.spirv.empty()) << "physical UD16 is not system s16 when N=16";
    EXPECT_EQ(rejected.rejection, "packet-sgpr-read-before-definition");
    const auto next = prepare_fragment_packet_inputs(e::inputs(17, 16), true);
    const auto positive = prosper::gpu::recompile_fragment_packet(e::packet(*next));
    EXPECT_FALSE(positive.spirv.empty()) << positive.rejection;
}

TEST(FragmentEntryPrefix, MissingHighWordRetainsFullUnsignedPresenceMask) {
    auto in = e::inputs(32, 31);
    in->entry.user_data_available &= ~(1u << 31);
    in->entry.user_data[31] = 0;
    const auto p = prepare_fragment_packet_inputs(in, true);
    ASSERT_TRUE(p->user_sgpr_count_available);
    EXPECT_EQ(p->user_sgpr_count, 32u);
    EXPECT_EQ(p->missing_user_sgprs, 0x80000000u);
    ASSERT_EQ(p->initial_user_sgprs.size(), 31u);
    EXPECT_EQ(p->initial_user_sgprs.back().first, 30u);
    const auto rejected = prosper::gpu::recompile_fragment_packet(e::packet(*p));
    EXPECT_EQ(rejected.rejection, "packet-sgpr-read-before-definition");
    EXPECT_TRUE(rejected.spirv.empty());
}

TEST(FragmentEntryPrefix, InvalidSixBitCountsNeverClamp) {
    for (uint32_t count = 33; count <= 63; ++count) {
        SCOPED_TRACE(count);
        const auto p = prepare_fragment_packet_inputs(e::inputs(count), true);
        EXPECT_FALSE(p->user_sgpr_count_available);
        EXPECT_TRUE(p->initial_user_sgprs.empty());
        EXPECT_TRUE(e::gap(*p, "packet-entry-user-sgpr-count-out-of-range"));
    }
}

TEST(FragmentEntryPrefix, UnlaunchedWordsAndOtherRsrc2FieldsAreNotInputs) {
    auto in = e::inputs(16);
    in->entry.user_data_available &= ~(1u << 16);
    in->entry.user_data[16] = 0;
    // Scratch/wave/exception/shared-VGPR/reserved bits cannot enlarge the user prefix.
    in->entry.rsrc2 |= ~(0x3eu | (1u << 27));
    const auto p = prepare_fragment_packet_inputs(in, true);
    EXPECT_TRUE(p->user_sgpr_count_available);
    EXPECT_EQ(p->user_sgpr_count, 16u);
    EXPECT_EQ(p->missing_user_sgprs, 0u);
    EXPECT_EQ(p->initial_user_sgprs.size(), 16u);
    EXPECT_FALSE(e::gap(*p, "packet-entry-user-sgpr-words-unavailable"));
}

TEST(FragmentEntryPrefix, UnassociatedOrNoncanonicalFactsGrantNoWords) {
    for (uint32_t mutation = 0; mutation < 6; ++mutation) {
        SCOPED_TRACE(mutation);
        auto in = e::inputs(17);
        bool match = true;
        switch (mutation) {
            case 0: match = false; break;
            case 1: in->raw_matches_producing_source = false; break;
            case 2: in->raw_code = std::make_shared<const std::vector<uint32_t>>(); break;
            case 3: in->source_fs.reset(); break;
            case 4: in->entry.observed = false; break;
            case 5: in->entry.user_data_available &= ~1u; in->entry.user_data[0] = 9; break;
        }
        const auto p = prepare_fragment_packet_inputs(in, match);
        EXPECT_EQ(p->inputs, in) << "raw context is retained without executable entry authority";
        EXPECT_FALSE(p->user_sgpr_count_available);
        EXPECT_TRUE(p->initial_user_sgprs.empty());
    }
    auto in = e::inputs(17);
    in->entry.rsrc2_available = false;
    in->entry.rsrc2 = 0;
    const auto p = prepare_fragment_packet_inputs(in, true);
    EXPECT_FALSE(p->user_sgpr_count_available);
    EXPECT_TRUE(p->initial_user_sgprs.empty());
    EXPECT_TRUE(e::gap(*p, "packet-entry-ps-rsrc2-unavailable"));
}
