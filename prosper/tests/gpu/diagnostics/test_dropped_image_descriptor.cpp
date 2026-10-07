// test_dropped_image_descriptor -- the [t8-dropped] witness (#4700). It exists so that a run which
// loses an image descriptor keeps the words that decide the fix, so the line must carry the raw
// words and the fields the fold's gates read, and must stay bounded however often the site recurs.
#include "gpu/diagnostics/dropped_image_descriptor.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace prosper::gpu;

namespace {
// The live words #4592 recorded at the same Kena program's s[8:15]: base zero, TYPE 0, DST_SEL 0.
constexpr std::array<uint32_t, 8> kKenaNullShaped = {0x00000000u, 0x00fff000u, 0x06f00000u,
                                                     0x00000000u, 0x20e1e000u, 0x00100044u,
                                                     0x000000fcu, 0x0004dfacu};
}  // namespace

TEST(DroppedImageDescriptor, LineCarriesRawWordsAndGateFields) {
    const std::string line =
        format_dropped_image_descriptor(0x505b7f0000ull, 30, 8, kKenaNullShaped, "base-zero");
    EXPECT_NE(line.find("[t8-dropped] program=0x505b7f0000 pc=30 srsrc=s8 reason=base-zero"),
              std::string::npos)
        << line;
    EXPECT_NE(line.find("base=0x0 type=0 dst_sel=0x000"), std::string::npos) << line;
    EXPECT_NE(
        line.find("raw 00000000 00fff000 06f00000 00000000 20e1e000 00100044 000000fc 0004dfac"),
        std::string::npos)
        << line;

    // Base40 is stored >>8 across word0 and word1[7:0]; TYPE and DST_SEL come from word3.
    const std::array<uint32_t, 8> real = {0x00505b7fu, 0xc3800012u, 0, 0x90000fa4u, 0, 0, 0, 0};
    const std::string t = format_dropped_image_descriptor(1, 2, -1, real, "unsupported-image-view");
    EXPECT_NE(t.find("srsrc=unknown"), std::string::npos) << t;
    EXPECT_NE(t.find("base=0x1200505b7f00 type=9 dst_sel=0xfa4"), std::string::npos) << t;
}

TEST(DroppedImageDescriptor, OncePerSiteAndBounded) {
    EXPECT_TRUE(note_dropped_image_descriptor(0x1000, 4, 0, kKenaNullShaped, "base-zero"));
    EXPECT_FALSE(note_dropped_image_descriptor(0x1000, 4, 0, kKenaNullShaped, "base-zero"))
        << "a site recurring every draw prints once";
    EXPECT_TRUE(note_dropped_image_descriptor(0x1000, 4, 0, kKenaNullShaped, "unmapped-img-fmt"))
        << "a different reason at the same site is a different fact";
    EXPECT_TRUE(note_dropped_image_descriptor(0x2000, 4, 0, kKenaNullShaped, "base-zero"))
        << "the same pc in another program is another site";
    size_t printed = 3;
    for (uint32_t pc = 100; pc < 100 + 2 * kDroppedImageDescriptorMaxReports; ++pc)
        printed +=
            note_dropped_image_descriptor(0x3000, pc, 0, kKenaNullShaped, "base-zero") ? 1 : 0;
    EXPECT_EQ(printed, kDroppedImageDescriptorMaxReports);
}
