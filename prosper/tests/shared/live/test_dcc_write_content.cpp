// Only complete, uniform producer bytes can stand in for an unreadable metadata plane.
#include "shared/live/output_write_content.hpp"
#include "shared/live/submit_renderer/dcc_write_content.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <thread>

namespace {
using namespace prosper::frontend;
using namespace prosper::frontend::submit_renderer;
constexpr std::array<DccWritePlane, 1> kPlane{{{0x1000, 0x2080, 16}}};
}
TEST(DccWriteContent, CompletePlaneInsideLargerWriteOwnsItsUniformByte) {
    std::array<uint8_t, 64> bytes{};
    bytes.fill(0x40);
    const auto facts = capture_dcc_write_facts(kPlane, 0x2070, bytes.size(), bytes);
    ASSERT_EQ(facts.size(), 1u);
    ASSERT_TRUE(facts[0].uniform_byte);
    EXPECT_EQ(*facts[0].uniform_byte, 0x40);
    bytes.fill(0xc0); // the borrowed producer storage is no longer retained
    EXPECT_EQ(*facts[0].uniform_byte, 0x40);
}
TEST(DccWriteContent, PartialUnavailableAndNonuniformWritesHaveNoUniformFact) {
    std::array<uint8_t, 16> bytes{};
    bytes.fill(0x40);
    for (uint32_t arm = 0; arm != 5; ++arm) {
        uint64_t address = 0x2080, count = bytes.size();
        std::span<const uint8_t> view = bytes;
        if (arm == 0) { ++address; --count; view = view.first(count); }
        if (arm == 1) view = {};
        if (arm == 2) view = view.first(15);
        if (arm == 3) count = 1; // a longer borrowed view does not enlarge the announced write
        if (arm == 4) bytes.back() = 0xff;
        const auto facts = capture_dcc_write_facts(kPlane, address, count, view);
        ASSERT_EQ(facts.size(), 1u) << arm;
        EXPECT_FALSE(facts[0].uniform_byte) << arm;
    }
}
TEST(DccWriteContent, UnrelatedAndEmptyWritesDoNotInspectContent) {
    std::array<uint8_t, 16> bytes{};
    EXPECT_TRUE(capture_dcc_write_facts(kPlane, 0, 16, bytes).empty());
    EXPECT_TRUE(capture_dcc_write_facts(kPlane, 0x2080, 0, bytes).empty());
    EXPECT_TRUE(capture_dcc_write_facts(kPlane, 0x2090, 16, bytes).empty());
    EXPECT_TRUE(capture_dcc_write_facts(kPlane, 0x2070, 16, bytes).empty());
}
TEST(DccWriteContent, SeveralPlanesRetainTheirIndependentBytesAndUnknownTails) {
    const DccWritePlane planes[]{{1, 0x2000, 8}, {2, 0x2008, 8}, {3, 0x2010, 8}};
    const uint8_t bytes[]{0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
                          0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0};
    const auto facts = capture_dcc_write_facts(planes, 0x2000, 24, bytes);
    ASSERT_EQ(facts.size(), 3u);
    EXPECT_EQ(facts[0].uniform_byte, 0x40);
    EXPECT_EQ(facts[1].uniform_byte, 0xc0);
    EXPECT_FALSE(facts[2].uniform_byte);
}
TEST(DccWriteContent, BorrowedViewIsBoundedScopedAndThreadLocal) {
    const std::array<uint8_t, 16> bytes{};
    EXPECT_TRUE(OutputWriteContent::for_write(0x2000).empty());
    {
        const OutputWriteContent outer(0x2000, bytes.data(), 32, bytes.size());
        EXPECT_EQ(OutputWriteContent::for_write(0x2000).size(), bytes.size());
        EXPECT_EQ(OutputWriteContent::for_write(reinterpret_cast<uintptr_t>(bytes.data())).data(),
                  bytes.data());
        EXPECT_TRUE(OutputWriteContent::for_write(0x2001).empty());
        std::thread peer([] { EXPECT_TRUE(OutputWriteContent::for_write(0x2000).empty()); });
        peer.join();
        {
            const OutputWriteContent inner(0x3000, bytes.data(), 4, 0);
            EXPECT_TRUE(OutputWriteContent::for_write(0x2000).empty());
            EXPECT_TRUE(OutputWriteContent::for_write(0x3000).empty());
        }
        EXPECT_EQ(OutputWriteContent::for_write(0x2000).size(), bytes.size());
    }
    EXPECT_TRUE(OutputWriteContent::for_write(0x2000).empty());
}
