// A numeric word retained across an NGG barrier cannot authenticate missing spill bits as VCC.
// Both physical halves need independent genuine terminal facts; numeric DATA remains available.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/ngg_lane_slots.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace f = prosper::test::ngg_slots;
struct Emitter {
    SpirvCompute b;
    RegState rs;
    Emitter(unsigned path) {
        b.ngg_workgroup_shell = true;
        b.native_subgroup_size = 64;
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.scc = b.bfalse();
        rs.sreg[4] = b.uconst(path);
        rs.sreg[20] = b.uconst(1);
        rs.sreg[21] = b.uconst(UINT32_MAX);
        rs.vreg[0] = b.uconst(0);
        rs.vreg[3] = b.uconst(11);
    }
    bool phase(const std::vector<uint32_t>& words) {
        std::vector<Rdna2Inst> instructions;
        rdna2_walk(words.data(), words.size(), instructions);
        return emit_cfg_state_machine(b, rs, instructions, {}, nullptr, true, false, {},
                                      words.data(), words.size());
    }
    bool prefix(bool supplied, uint32_t word) {
        // A real SCC diamond. A missing else-arm slot must survive READLANE and ADD as a mark.
        std::vector<uint32_t> code{0xbf068004u};
        const auto other = f::branch(code, 4);
        f::write(code, 20, 0, 20);
        const auto join = f::branch(code, 2);
        f::target(code, other, code.size());
        if (supplied)
            f::write(code, 20, 0, 21);
        else
            code.push_back(0xbf800000u);
        f::target(code, join, code.size());
        f::read(code, 20, 0, 15);
        const uint32_t sibling = word == 106 ? 107 : 106;
        f::append(code, {0x8000810fu | (word << 16), 0xbe800380u | (sibling << 16), 0xbf810000u});
        return phase(code);
    }
    bool after_barrier() {
        b.barrier();
        // VCCZ skips the write; a genuine zero on the second arm keeps the initial 11.
        return phase({0xbf860001u, 0x7e060287u, 0xbf810000u});
    }
    std::vector<uint32_t> finish() {
        b.store_output_word(rs.vreg.at(3), 1, 0);
        return b.finish();
    }
};
std::vector<uint32_t> module(uint32_t word, unsigned path) {
    Emitter e(path);
    if (!e.prefix(true, word) || !e.after_barrier()) return {};
    return e.finish();
}
}   // namespace

TEST(NggScalarVccBarrier, MissingSpillArmRetainsDataWithoutCreatingMaskAuthority) {
    for (uint32_t word : {106u, 107u}) {
        for (unsigned path : {0u, 1u}) {
            SCOPED_TRACE(word);
            SCOPED_TRACE(path);
            Emitter e(path);
            ASSERT_TRUE(e.prefix(false, word));
            ASSERT_TRUE(e.rs.sreg.contains(word));
            EXPECT_TRUE(e.rs.sreg_merge_placeholder.contains(word));
            EXPECT_FALSE(e.rs.terminal_wave64_scalar_words.contains(word));
            EXPECT_FALSE(e.after_barrier());
            EXPECT_EQ(e.rs.vcc, 0u);
        }
    }
}

TEST(NggScalarVccBarrier, BothSuppliedArmsKeepTheirActualBranchResult) {
    for (uint32_t word : {106u, 107u}) {
        for (unsigned path : {0u, 1u}) {
            const auto code = module(word, path);
            ASSERT_FALSE(code.empty());
            bpermute_oracle::Interpreter vm(code);
            vm.native_subgroup_width = 64;
            const auto output = vm.run_buffers(
                64, {{0, std::vector<uint32_t>(256)}, {1, std::vector<uint32_t>(64, 0xdeadbeefu)}},
                1);
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            EXPECT_EQ(output, std::vector<uint32_t>(64, path ? 11u : 7u));
        }
    }
}

TEST(NggScalarVccBarrier, SuppliedBranchesExecuteOnAnActualWave64Subgroup) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 compute unavailable";
    for (uint32_t word : {106u, 107u}) {
        for (unsigned path : {0u, 1u}) {
            const auto code = module(word, path);
            ASSERT_FALSE(code.empty());
            const auto output =
                prosper::test::run_compute(code, std::vector<float>(256), 64, 64, {}, {}, nullptr,
                                           64, nullptr, nullptr, nullptr, 64);
            ASSERT_EQ(output.size(), 64u);
            std::vector<uint32_t> bits(output.size());
            std::memcpy(bits.data(), output.data(), bits.size() * sizeof(uint32_t));
            EXPECT_EQ(bits, std::vector<uint32_t>(64, path ? 11u : 7u));
        }
    }
}
