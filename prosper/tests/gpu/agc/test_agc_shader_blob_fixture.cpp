// #5295: the real CreateShader HLE must relocate fixture bytes at low and high host addresses.
// Retain registered blobs through all test cases because the shader registry keeps their pointers.
#include "fixtures/agc_shader_blob.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <sys/mman.h>
#include <vector>

namespace {
constexpr std::size_t kMappingBytes = 4096;
constexpr uintptr_t kFourGiB = 0x100000000ull;
namespace P = prosper::agc::Pm4;

struct Fixture {
    prosper::test::AgcShaderBlob<2> blob;
    alignas(256) std::array<uint32_t, 1> code{0xbf810000u}; // s_endpgm
};
static_assert(sizeof(Fixture) <= kMappingBytes);

Fixture* mapped_fixture(bool high) {
    struct Pages {
        std::vector<void*> values;
        ~Pages() {
            for (void* page : values) munmap(page, kMappingBytes);
        }
    };
    static Pages retained;
    const uintptr_t base = high ? 0x500000000000ull : 0x20000000u;
    for (uintptr_t i = 0; i < 16; ++i) {
        auto* address = reinterpret_cast<void*>(base + i * 0x10000u);
        void* page = mmap(address, kMappingBytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (page == MAP_FAILED) continue;
        if (page != address) {
            munmap(page, kMappingBytes);
            continue;
        }
        retained.values.push_back(page);
        return new (page) Fixture{};
    }
    return nullptr;
}

void reset(Fixture& fixture) {
    fixture.blob.reset({{{P::COMPUTE_PGM_LO, 0}, {P::COMPUTE_PGM_HI, 0}}}, sizeof(fixture.code));
}

void register_and_check(Fixture& fixture) {
    prosper::register_agc_hle();
    const auto create = prosper::Hle::lookup("f3dg2CSgRKY");
    ASSERT_NE(create, nullptr);
    void* registered = nullptr;
    ASSERT_EQ(create(reinterpret_cast<uint64_t>(&registered),
                     reinterpret_cast<uint64_t>(&fixture.blob.header),
                     reinterpret_cast<uint64_t>(fixture.code.data()), 0, 0, 0),
              0u);
    EXPECT_EQ(registered, &fixture.blob.header);
    EXPECT_EQ(fixture.blob.header.sh_registers, fixture.blob.registers.data());
    EXPECT_EQ(fixture.blob.header.code, fixture.code.data());
    const auto code = reinterpret_cast<uintptr_t>(fixture.code.data());
    EXPECT_EQ(fixture.blob.registers[0].value, uint32_t((code >> 8) & 0xffffffffu));
    EXPECT_EQ(fixture.blob.registers[1].value, uint32_t((code >> 40) & 0xffu));
}
} // namespace

TEST(AgcShaderBlobFixture, RelocatesAtALowAddress) {
    Fixture* fixture = mapped_fixture(false);
    ASSERT_NE(fixture, nullptr);
    ASSERT_LT(reinterpret_cast<uintptr_t>(fixture), kFourGiB);
    reset(*fixture);
    register_and_check(*fixture);
}

TEST(AgcShaderBlobFixture, RelocatesAtAHighAddress) {
    Fixture* fixture = mapped_fixture(true);
    ASSERT_NE(fixture, nullptr);
    ASSERT_GE(reinterpret_cast<uintptr_t>(fixture), kFourGiB);
    reset(*fixture);
    register_and_check(*fixture);
}

TEST(AgcShaderBlobFixture, UnrelocatedBytesCanBeCopiedToALowAddress) {
    Fixture seed;
    reset(seed);
    Fixture* fixture = mapped_fixture(false);
    ASSERT_NE(fixture, nullptr);
    fixture->blob = seed.blob;
    register_and_check(*fixture);
}

TEST(AgcShaderBlobFixture, ResetReencodesALowBlobBeforeAnotherRegistration) {
    Fixture* fixture = mapped_fixture(false);
    ASSERT_NE(fixture, nullptr);
    for (int pass = 0; pass < 2; ++pass) {
        reset(*fixture);
        register_and_check(*fixture);
    }
}
