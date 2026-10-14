// Missing hardware source words must refuse before any load; target bytes share the parent owner.
#include "gpu/execute/loaded_scalar_global_admission.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>

using namespace prosper::gpu;
namespace {
const LoadedScalarGlobalRead kRead{28, 167, 10, 16, 9, 0, 16, 32};
struct Reader final : FoldReader {
    bool owned = true, parent_current = true, target_current = true;
    uint64_t parent_pointer = 0x200000;
    uint32_t target_bytes = 32;
    std::vector<std::tuple<uint32_t, uint64_t, uint32_t>> requests;
    std::map<uint32_t, std::pair<uint64_t, std::array<uint8_t, 8>>> parents;
    bool owns_raw_wide(uint32_t pc) const override { return owned && (pc == 28 || pc == 167); }
    bool probe(FoldProbe kind, uint32_t pc, uint64_t address, uint32_t bytes) override {
        EXPECT_EQ(kind, FoldProbe::Raw);
        requests.emplace_back(pc, address, bytes);
        if (pc == 28) {
            if (!parent_current || bytes != 8) return false;
            if (!parents.contains(pc)) {
                std::array<uint8_t, 8> owner{};
                std::memcpy(owner.data(), &parent_pointer, owner.size());
                parents.emplace(pc, std::pair{address, owner});
            }
            return parents.at(pc).first == address;
        }
        return target_current && pc == 167 && bytes == target_bytes;
    }
    uint32_t word(uint32_t pc, uint64_t address) override {
        const auto& [base, bytes] = parents.at(pc);
        if (address < base || address - base > 4) throw std::runtime_error("wrong read");
        uint32_t result = 0;
        std::memcpy(&result, bytes.data() + address - base, sizeof(result));
        return result;
    }
    void prefix(uint32_t pc, uint64_t address, void* destination, uint32_t bytes) override {
        const auto& owner = parents.at(pc);
        if (address != owner.first || bytes != owner.second.size())
            throw std::runtime_error("wrong prefix");
        std::memcpy(destination, owner.second.data(), bytes);
    }
};
struct Input {
    std::array<uint32_t, 10> words{0, 0, 0x100000, 0, 0, 0, 0, 0, 0, 0};
    LoadedGlobalUserPrefix prefix() { return {10u, 8u, words, (1u << 2) | (1u << 3)}; }
};
} // namespace

TEST(LoadedScalarGlobalAdmission, MixedWidthsRetainOriginalPcsAndOffsets) {
    EXPECT_EQ(loaded_scalar_global_chains({kRead}),
              (std::vector<RawNestedWideChain>{{28, 167, 8, 32, 0, 16}}));
    auto invalid = kRead;
    invalid.window_bytes = 24;
    EXPECT_TRUE(loaded_scalar_global_chains({kRead, invalid}).empty());
}

TEST(LoadedScalarGlobalAdmission, GenuinePresentZeroHighWordIsKnown) {
    Input input;
    Reader reader;
    ASSERT_TRUE(prepare_loaded_scalar_globals(reader, {kRead}, input.prefix()));
    ASSERT_EQ(reader.requests.size(), 2u);
    EXPECT_EQ(reader.requests[0], (std::tuple<uint32_t, uint64_t, uint32_t>{28, 0x100000, 8}));
    EXPECT_EQ(reader.requests[1], (std::tuple<uint32_t, uint64_t, uint32_t>{167, 0x200010, 32}));
}

TEST(LoadedScalarGlobalAdmission, NonzeroHighPointerWordAndParentOffsetAreRetained) {
    Input input;
    Reader reader;
    input.words[3] = 1;
    auto shifted = kRead;
    shifted.parent_offset = 8;
    shifted.window_bytes = 16;
    reader.target_bytes = 16;
    ASSERT_TRUE(prepare_loaded_scalar_globals(reader, {shifted}, input.prefix()));
    EXPECT_EQ(reader.requests[0], (std::tuple<uint32_t, uint64_t, uint32_t>{28, 0x100100008, 8}));
    EXPECT_EQ(reader.requests[1], (std::tuple<uint32_t, uint64_t, uint32_t>{167, 0x200010, 16}));
}

TEST(LoadedScalarGlobalAdmission, PaddedZeroWithoutPhysicalPresenceRefusesBeforeRead) {
    Input input;
    Reader reader;
    auto prefix = input.prefix();
    prefix.physical_word_mask &= ~(1u << 3);
    EXPECT_FALSE(prepare_loaded_scalar_globals(reader, {kRead}, prefix));
    EXPECT_TRUE(reader.requests.empty());
}

TEST(LoadedScalarGlobalAdmission, MissingOrTooShortDeclaredPrefixRefusesBeforeRead) {
    Input input;
    for (std::optional<uint32_t> count : {std::optional<uint32_t>{}, {3u}, {33u}}) {
        Reader reader;
        auto prefix = input.prefix();
        prefix.declared_count = count;
        EXPECT_FALSE(prepare_loaded_scalar_globals(reader, {kRead}, prefix));
        EXPECT_TRUE(reader.requests.empty());
    }
    Reader reader;
    auto prefix = input.prefix();
    prefix.words = prefix.words.first(3);
    EXPECT_FALSE(prepare_loaded_scalar_globals(reader, {kRead}, prefix));
    EXPECT_TRUE(reader.requests.empty());
}

TEST(LoadedScalarGlobalAdmission, SystemPointerAndUnownedReaderCannotBorrowUserWords) {
    Input input;
    Reader reader;
    auto system = kRead;
    system.entry_pointer_sgpr = 6;
    EXPECT_FALSE(prepare_loaded_scalar_globals(reader, {system}, input.prefix()));
    reader.owned = false;
    EXPECT_FALSE(prepare_loaded_scalar_globals(reader, {kRead}, input.prefix()));
    EXPECT_TRUE(reader.requests.empty());
}

TEST(LoadedScalarGlobalAdmission, EveryEntrySourceIsValidatedBeforeAnyProbe) {
    Input input;
    Reader reader;
    auto missing = kRead;
    missing.entry_pointer_sgpr = 16;
    EXPECT_FALSE(prepare_loaded_scalar_globals(reader, {kRead, missing}, input.prefix()));
    EXPECT_TRUE(reader.requests.empty());
}

TEST(LoadedScalarGlobalAdmission, ParentOrTargetAuthorityRefusalStopsPreparation) {
    Input input;
    Reader reader;
    reader.parent_current = false;
    EXPECT_FALSE(prepare_loaded_scalar_globals(reader, {kRead}, input.prefix()));
    EXPECT_EQ(reader.requests.size(), 1u);
    reader = Reader{};
    reader.target_current = false;
    EXPECT_FALSE(prepare_loaded_scalar_globals(reader, {kRead}, input.prefix()));
    EXPECT_EQ(reader.requests.size(), 2u);
}

TEST(LoadedScalarGlobalAdmission, LaterParentMutationDoesNotChangeObservedFoldWords) {
    Input input;
    Reader reader;
    ASSERT_TRUE(prepare_loaded_scalar_globals(reader, {kRead}, input.prefix()));
    reader.parent_pointer = 0x300000;
    EXPECT_EQ(reader.word(28, 0x100000), 0x200000u);
    ASSERT_TRUE(prepare_loaded_scalar_globals(reader, {kRead}, input.prefix()));
    EXPECT_EQ(std::get<1>(reader.requests.back()), 0x200010u);
}

TEST(LoadedScalarGlobalAdmission, CheckedParentAndTargetAdditionRejectWraparound) {
    Input input;
    Reader reader;
    auto overflow = kRead;
    overflow.parent_offset = 16;
    input.words[2] = 0xfffffff8;
    input.words[3] = UINT32_MAX;
    EXPECT_FALSE(prepare_loaded_scalar_globals(reader, {overflow}, input.prefix()));
    EXPECT_TRUE(reader.requests.empty());
    input = Input{};
    reader.parent_pointer = UINT64_MAX - 7u;
    EXPECT_FALSE(prepare_loaded_scalar_globals(reader, {kRead}, input.prefix()));
    EXPECT_EQ(reader.requests.size(), 1u);
    reader = Reader{};
    reader.parent_pointer = UINT64_MAX - 47u; // Offset fits, but the complete target does not.
    EXPECT_FALSE(prepare_loaded_scalar_globals(reader, {kRead}, input.prefix()));
    EXPECT_EQ(reader.requests.size(), 1u);
}
