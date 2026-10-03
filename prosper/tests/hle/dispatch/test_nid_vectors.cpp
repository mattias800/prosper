// Golden NID vectors pin the name-to-import-ID contract independently of game dumps.
// A wrong byte order, salt, or Sony base64 alphabet makes HLE exports unresolvable.
#include "hle/dispatch/nid.hpp"

#include <array>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

namespace prosper {
namespace {

struct NidVector {
    std::string_view name;
    std::string_view nid;
};

// Curated, independently cross-checked names from the local PS5 stub catalog.
// These pairs exercise multiple libraries, short/long names, and both '+'/'-' alphabet chars.
constexpr std::array kVectors{
    NidVector{"memcpy", "Q3VBxCXhUHs"},
    NidVector{"memmove", "+P6FRGH4LfA"},
    NidVector{"sceKernelCreateSema", "188x57JYp0g"},
    NidVector{"sceKernelOpen", "1G3lF1Gg1k8"},
    NidVector{"sceSystemServiceReportAbnormalTermination", "3s8cHiCBKBE"},
    NidVector{"sceKernelSignalSema", "4czppHBiriw"},
    NidVector{"sceKernelWrite", "4wSze92BhLI"},
    NidVector{"sceKernelGetProcParam", "959qrazPIrg"},
    NidVector{"scePthreadMutexLock", "9UK1vLZQft4"},
    NidVector{"sceKernelRead", "Cg4srZ6TKbU"},
    NidVector{"sceKernelCreateEqueue", "D0OdFMjp46I"},
    NidVector{"sceKernelMmap", "PGhQHd-dzv8"},
    NidVector{"sceNetSocket", "Q4qBuN-c0ZM"},
    NidVector{"sceVideoOutOpen", "Up36PTk687E"},
    NidVector{"scePthreadCondWait", "WKAXJ4XBPQ4"},
    NidVector{"sceKernelWaitSema", "Zxa0VhQVTsk"},
    NidVector{"sceAudioOutOpen", "ekNvsT22rsY"},
    NidVector{"scePadOpen", "xk0AcarP3V4"},
};

TEST(NidHash, MatchesGoldenVectorsAcrossLibraries) {
    for (const auto& vector : kVectors) {
        EXPECT_EQ(nid_hash(std::string(vector.name)), vector.nid)
            << "NID for " << vector.name;
    }
}

TEST(NidHash, ReturnsElevenCharacters) {
    for (const auto& vector : kVectors) {
        EXPECT_EQ(nid_hash(std::string(vector.name)).size(), 11u)
            << "NID length for " << vector.name;
    }
}

}  // namespace
}  // namespace prosper
