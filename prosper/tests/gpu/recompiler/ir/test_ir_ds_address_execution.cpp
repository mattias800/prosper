// Reference DS accesses use host-computed addresses, so matching read/write offset mistakes cannot
// cancel. Inactive stores are checked by all-active reads of independently initialized LDS words.
#include "gpu/recompiler/ir/ds_address_cases.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
using namespace prosper::gpu;
namespace fixture = prosper::test::ir::ds_address;
namespace {
TEST(IrDsAddressExecution, OrdinaryAndPairedAccessesMatchIndependentPhysicalWordsInBothWaveSizes) {
    size_t index = 0;
    for (const auto& c : fixture::cases()) {
        SCOPED_TRACE(::testing::Message() << "case=" << index << " form=" << c.form
                                          << " literal=" << c.literal << " wave=" << c.wave);
        const auto p = fixture::program(c);
        const uint64_t site = 0x5387000000ull + index * 0x1000;
        const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                              {RecompileDiagnosticStage::Compute, site});
        ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(site);
        const auto d = fixture::data(c);
        std::vector<uint32_t> got;
        const auto dummy = prosper::test::run_compute(module, std::vector<float>(c.wave), c.wave,
                                                      c.wave, d.input, d.initial, &got, c.wave);
        ASSERT_EQ(dummy.size(), c.wave);
        ASSERT_EQ(got, d.expected);
        ++index;
    }
    EXPECT_EQ(index, 68u);
}
TEST(IrDsAddressExecution, OversizedSt64OffsetsAndGdsPairsStayRefused) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t form : {10u, 11u}) {
            auto p = fixture::program({form, wave, 0x0100});
            p.config.lds_bytes = 256;   // OFFSET1=1 means dword64, exactly beyond this allocation.
            EXPECT_TRUE(
                recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
        }
    for (uint32_t wave : {32u, 64u}) {
        auto p = fixture::program({9, wave, 0x0100});
        for (size_t pc = 0; pc < p.code.size();) {
            const auto in = rdna2_decode_one(p.code.data() + pc, p.code.size() - pc);
            ASSERT_GT(in.len_dwords, 0u);
            if (in.fmt == Rdna2Format::DS && in.opcode == 0x37) p.code[pc] |= 1u << 17;
            pc += in.len_dwords;
        }
        EXPECT_TRUE(recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
    }
}
TEST(IrDsAddressExecution, EqualOffsetsHaveDistinctSecondDataAndShareTheFirstPhysicalPair) {
    const fixture::Case c{12, 64, 0x0202};
    EXPECT_EQ(fixture::address(c, 3, 0), 208u);
    EXPECT_EQ(fixture::address(c, 3, 1), 212u);
    EXPECT_EQ(fixture::address(c, 3, 2), 208u);
    EXPECT_EQ(fixture::address(c, 3, 3), 212u);
    EXPECT_NE(fixture::word(3, 0, false), fixture::word(3, 2, false));
    EXPECT_NE(fixture::word(3, 1, false), fixture::word(3, 3, false));
    const fixture::Case scaled{10, 64, 0x0103};
    EXPECT_EQ(fixture::address(scaled, 4, 0), 4864u);
    EXPECT_EQ(fixture::address(scaled, 4, 1), 4352u);
}
}   // namespace
