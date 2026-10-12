// Current symbolic words are genuine Bool halves, while missing/overwritten halves remain
// unusable at SCC consumers and barrier reloads. Existing portable fixtures stay unchanged.
#include "fixtures/portable_mask_common.hpp"
#include "fixtures/portable_mask_slots.hpp"
#include "fixtures/portable_mask_half.hpp"
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_portable_mask_dispatch.hpp"
#include "gpu/recompiler/rdna2_portable_mask_slots.hpp"
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {
namespace c = prosper::test::portable_mask_common;
namespace f = prosper::test::portable_mask_slots;
namespace base = prosper::test::portable_mask_half;
std::vector<uint32_t> relayed(bool reverse = false) {
    auto code = c::halves(false, true, reverse);
    code.insert(code.end(), {base::mov(24, 20), base::mov(25, 21)});
    return code;
}
void values(const std::vector<uint32_t>& code, uint32_t expected) {
    const auto attempt = f::compile(code, true);
    ASSERT_FALSE(attempt.module.empty()) << attempt.refusal;
    bpermute_oracle::Interpreter oracle(attempt.module);
    const auto actual = oracle.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
    ASSERT_TRUE(oracle.error.empty()) << oracle.error;
    EXPECT_EQ(actual, std::vector<uint32_t>(64, expected));
}
void pair_refuses(const std::vector<uint32_t>& prefix, uint32_t root = 24) {
    const auto attempt = f::compile(c::branch(prefix, root), true);
    EXPECT_TRUE(attempt.module.empty());
    EXPECT_NE(attempt.refusal.find("portable-mask-half-pair-unavailable"), std::string::npos)
        << attempt.refusal;
    EXPECT_NE(attempt.refusal.find("pc=" + std::to_string(prefix.size() + 1)), std::string::npos)
        << "the original comparison must refuse: " << attempt.refusal;
}
std::vector<uint32_t> crossed_slots(bool numeric_high) {
    auto code = c::halves(false, true);
    f::spill(code, 20, 13);
    if (numeric_high) code.push_back(base::mov(22, 128));
    f::spill(code, numeric_high ? 22 : 21, 17);
    code.push_back(0xbf8a0000u);
    f::reload(code, 21, 17);
    f::reload(code, 20, 13);
    return code;
}
void scc_mark(const std::vector<uint32_t>& prefix, bool keep, bool memory = false) {
    using namespace prosper::gpu;
    const auto code = c::branch(prefix, 24);
    std::vector<Rdna2Inst> ins;
    ASSERT_TRUE(rdna2_walk(code.data(), code.size(), ins));
    SpirvCompute b;
    b.begin(1, nullptr, 64, 1, 1, 64);
    RegState rs;
    rs.exec = b.btrue();
    PortableMaskHalfBodyScope scope(b, rs, ins);
    const auto found = std::find_if(ins.begin(), ins.end(),
                                    [&](const auto& in) { return in.pc == prefix.size() + 1; });
    ASSERT_NE(found, ins.end());
    const auto& compare = *found;
    ASSERT_EQ(compare.fmt, Rdna2Format::SOPC);
    ASSERT_EQ(compare.opcode, 0x12u);
    SpilledMaskHalves marks;
    marks.relayed_words.insert(kRelayedScc);
    marks.relayed_words.insert(57);
    marks.memory_words.insert(57);
    SpillSlotDomains domains;
    // Adversarial caller flag: the generic whole-mask classification is insufficient to clear
    // a missing or overwritten half. The exact current role proof must independently agree.
    SpilledVectorSourceMarks source;
    source.memory = memory;
    refine_portable_mask_dispatch(b, compare, true, source, marks, domains);
    EXPECT_EQ(marks.relayed_words.contains(kRelayedScc), keep);
    EXPECT_TRUE(marks.relayed_words.contains(57));
    EXPECT_TRUE(marks.memory_words.contains(57));
}
} // namespace

TEST(PortableMaskDispatch, LowFirstRelayedPairKeepsTheActualSccVote) {
    values(c::branch(relayed(), 24), 9);
}
TEST(PortableMaskDispatch, HighFirstRelayedPairKeepsTheActualSccVote) {
    values(c::branch(relayed(true), 24), 9);
}
TEST(PortableMaskDispatch, OverwrittenHighRelayCannotProduceTheOldVote) {
    auto code = relayed();
    code.push_back(base::mov(25, 128));
    pair_refuses(code);
}
TEST(PortableMaskDispatch, OverwrittenLowRelayCannotProduceTheOldVote) {
    auto code = relayed();
    code.push_back(base::mov(24, 128));
    pair_refuses(code);
}
TEST(PortableMaskDispatch, OppositeRelayRolesCannotAuthorizeThePair) {
    auto code = c::halves(false, true);
    code.insert(code.end(), {base::mov(24, 21), base::mov(25, 20)});
    pair_refuses(code);
}
TEST(PortableMaskDispatch, OneArmHighOverwriteCannotBorrowTheOtherArm) {
    auto code = relayed();
    code.insert(code.end(), {0xbf068000u, 0xbf840001u, base::mov(25, 128)});
    pair_refuses(code);
}
TEST(PortableMaskDispatch, OneArmGenuineHighRelayKeepsItsVote) {
    auto code = relayed();
    code.insert(code.end(), {0xbf068000u, 0xbf840001u, base::mov(25, 21)});
    values(c::branch(code, 24), 9);
}
TEST(PortableMaskDispatch, BarrierCarriesGenuineRespilledSlotRoles) {
    values(c::reduction(crossed_slots(false), 0x10), 32);
}
TEST(PortableMaskDispatch, NumericRespillAfterBarrierCannotBorrowTheOldHighRole) {
    const auto prefix = crossed_slots(true);
    const auto attempt = f::compile(c::reduction(prefix, 0x10), true);
    EXPECT_TRUE(attempt.module.empty());
    EXPECT_NE(attempt.refusal.find("portable-mask-half-pair-unavailable"), std::string::npos)
        << attempt.refusal;
    EXPECT_NE(attempt.refusal.find("pc=" + std::to_string(prefix.size() + 1)), std::string::npos)
        << attempt.refusal;
}
TEST(PortableMaskDispatch, ScalarStorageAloneCannotSeedABoolSlot) {
    using namespace prosper::gpu;
    auto code = c::halves(false, true);
    std::vector<Rdna2Inst> ins;
    ASSERT_TRUE(rdna2_walk(code.data(), code.size(), ins));
    SpirvCompute b;
    b.begin(1, nullptr, 64, 1, 1, 64);
    RegState rs;
    rs.portable_mask_half_types.slots[{18, 13}] = 0;
    rs.portable_mask_half_types.may_slots.insert({18, 13});
    rs.vgpr_lane_slots[18][13] = b.uconst(7);
    PortableMaskHalfBodyScope scope(b, rs, ins);
    SpilledMaskHalves marks;
    SpillSlotDomains domains;
    seed_portable_mask_dispatch(b, rs, 0, marks, domains);
    EXPECT_FALSE(marks.defined_slots.contains({18, 13}));
    EXPECT_FALSE(domains.contains({18, 13}));
}
TEST(PortableMaskDispatch, MissingHighStillPoisonsAnOverconfidentSccClassification) {
    auto code = c::missing(true);
    code.push_back(base::mov(24, 20));
    scc_mark(code, true);
}
TEST(PortableMaskDispatch, OverwrittenHighStillPoisonsAnOverconfidentSccClassification) {
    auto code = relayed();
    code.push_back(base::mov(25, 128));
    scc_mark(code, true);
}
TEST(PortableMaskDispatch, CompleteRelayCanReplaceOnlyItsSccMark) {
    scc_mark(relayed(), false);
}
TEST(PortableMaskDispatch, MemorySourceNeverRepairsTheSccMark) {
    scc_mark(relayed(), true, true);
}
TEST(PortableMaskDispatch, RetiredDataStorageCannotShadowCurrentBoolSlotsAcrossABarrier) {
    std::vector<uint32_t> code{0x7e060280u, base::mov(126, 128, true)};
    f::spill(code, 135, 0);
    f::spill(code, 126, 0);
    code.push_back(base::mov(126, 193, true));
    f::spill(code, 136, 1);
    f::spill(code, 127, 1);
    code.push_back(0xbf8a0000u);
    f::reload(code, 21, 1);
    f::reload(code, 20, 0);
    values(c::reduction(code, 0x10), 32);
}
TEST(PortableMaskDispatch, OneEdgeDataSlotCannotImportTheOldHighMaskRole) {
    for (const bool numeric : {false, true}) {
        auto code = c::halves(false, true);
        code.resize(7);
        code.insert(code.end(), {0xbf068000u, 0xbf840002u});
        f::spill(code, numeric ? 128 : 127, 1);
        code.push_back(0xbf8a0000u);
        f::reload(code, 21, 1);
        f::reload(code, 20, 0);
        const auto completed = c::reduction(code, 0x10);
        if (numeric) {
            const auto attempt = f::compile(completed, true);
            EXPECT_TRUE(attempt.module.empty()) << "one edge replaced HIGH with DATA";
        } else
            values(completed, 32);
    }
}
