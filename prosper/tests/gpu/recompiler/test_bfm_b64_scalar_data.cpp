// test_bfm_b64_scalar_data — s_bfm_b64 must record its architectural DATA value,
// not only its lane-mask view (#4694's program).
//
// Black Flag's indirect-args builder computes `s_bfm_b64 s[4:5], 1, 63`,
// selects it with `s_cselect_b64 vcc, s[2:3], s[4:5]`, and stores both VCC halves
// to the buffer indirect draws read. The BFM lowering modeled only the mask domain
// (a per-lane bool in sreg_bool), so the destination pair held no scalar DATA and
// the select refused with an incomplete pair — even though every word involved is
// exactly defined (s2/s3 by s_mov/s_brev, s[4:5] = 1<<63), and even though the
// selected halves are observably consumed by stores (not dead copies).
//
// The fix records the u64 DATA (exact for all width/offset inputs, including
// runtime ones) alongside the mask bool for SGPR destinations. EXEC/VCC
// destinations keep their mask-only form: those registers have no DATA slots in
// the model. s_bfm_b32 is untouched by this change.
//
// The kernels below are hand-written and assembled with llvm-mc -mcpu=gfx1030
// (encodings and branch displacements are the assembler's, not hand-computed).
// Each negative is one change to the positive, so a refusal can only come from
// the property that changed:
//   * the high half genuinely absent (s4 by s_mov, s5 never written): #4749's
//     fabricated-word guard must still fire — this arm pins that the fix does
//     not admit truly-undefined words;
//   * the BFM width from a never-written register: the DATA computation must
//     refuse fail-visible rather than fabricate.
// The positive that can run is executed: the stored words must equal the
// architecturally defined pair (1, 0x80000000), bit for bit.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/compute_runner.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kLanes = 64;

// P. s_mov/bfm/brev-defined pair selected into VCC, both halves stored.
// s_cselect_b64 is at pc 5.
const uint32_t kBfmDefinedPair[] = {
    0xBE800387u, 0xBF068700u, 0xBE820381u, 0x9284BF81u, 0xBE830B81u, 0x85EA0402u, 0x7E04026Au,
    0x7E06026Bu, 0xE0702000u, 0x80020200u, 0xE0702004u, 0x80020300u, 0xBF810000u,
};
// P0. As P with SCC=0 (s_cmp_eq s0, 8): the select takes s[4:5], so executing
// it observes the BFM DATA words rather than the s_mov side.
const uint32_t kBfmSelectSrc1[] = {
    0xBE800387u, 0xBF068800u, 0xBE820381u, 0x9284BF81u, 0xBE830B81u, 0x85EA0402u, 0x7E04026Au,
    0x7E06026Bu, 0xE0702000u, 0x80020200u, 0xE0702004u, 0x80020300u, 0xBF810000u,
};
// P-control. As P with every pair word from s_mov_b32: no BFM involved.
const uint32_t kMovDefinedPair[] = {
    0xBE800387u, 0xBF068700u, 0xBE820381u, 0xBE830382u, 0xBE840383u, 0xBE850384u, 0x85EA0402u,
    0x7E04026Au, 0x7E06026Bu, 0xE0702000u, 0x80020200u, 0xE0702004u, 0x80020300u, 0xBF810000u,
};
// N_absent. As P with s_mov_b32 s4, 0 instead of the BFM: s5 is never written.
const uint32_t kAbsentHighHalf[] = {
    0xBE800387u, 0xBF068700u, 0xBE820381u, 0xBE840380u, 0xBE830B81u, 0x85EA0402u, 0x7E04026Au,
    0x7E06026Bu, 0xE0702000u, 0x80020200u, 0xE0702004u, 0x80020300u, 0xBF810000u,
};
// N_width. As P with the BFM width from never-written s9: unresolvable operand.
const uint32_t kUnresolvableWidth[] = {
    0xBE800387u, 0xBF068700u, 0xBE820381u, 0x9284BF09u, 0xBE830B81u, 0x85EA0402u, 0x7E04026Au,
    0x7E06026Bu, 0xE0702000u, 0x80020200u, 0xE0702004u, 0x80020300u, 0xBF810000u,
};

// The buffer at s[8:11] is the runner's `cbuf` (binding 2), one dword per lane.
ShaderResourceTable buffer_table() {
    ShaderResourceTable table;
    ShaderResource buffer{};
    buffer.cls = ResourceClass::ConstantBuffer;
    buffer.format = DataFormat::Uint32;
    buffer.num_components = 1;
    buffer.binding = 2;
    buffer.stride = 4;
    buffer.size = kLanes * 4;
    buffer.sgpr_base = 8;
    table.resources.push_back(buffer);
    return table;
}

template <size_t N>
std::vector<uint32_t> compile(const uint32_t (&code)[N]) {
    const ShaderResourceTable table = buffer_table();
    return recompile_valu(code, N, /*num_inputs*/ 1, /*out_vgpr*/ 2, &table);
}

std::vector<float> lane_indices() {
    std::vector<float> input(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) input[lane] = static_cast<float>(lane);
    return input;
}

}  // namespace

TEST(BfmB64ScalarData, SelectAdmitsABfmDefinedPair) {
    const ShaderResourceTable table = buffer_table();
    EXPECT_FALSE(recompile_valu(kMovDefinedPair, std::size(kMovDefinedPair),
                                /*num_inputs*/ 1, /*out_vgpr*/ 2, &table)
                     .empty())
        << "control: a select of two s_mov-defined pairs must lower with or without the fix";
    EXPECT_FALSE(compile(kBfmDefinedPair).empty())
        << "s_bfm_b64 s[4:5], 1, 63 defines both DATA words; the select must lower";
}

TEST(BfmB64ScalarData, AbsentHighHalfStillRefuses) {
    EXPECT_TRUE(compile(kAbsentHighHalf).empty())
        << "s5 genuinely never written: #4749's fabricated-word guard must still fire";
}

TEST(BfmB64ScalarData, UnresolvableWidthRefuses) {
    EXPECT_TRUE(compile(kUnresolvableWidth).empty())
        << "BFM width from an undefined register must refuse fail-visible, not fabricate";
}

TEST(BfmB64ScalarData, StoredHalvesAreExactOnDevice) {
    // SCC=0 selects s[4:5] = bfm(1, 63) = 1<<63: lane 0 stores (0, 0x80000000).
    // The SCC=1 shape is covered at compile level above; only the taken source's
    // words are observable on device, so each polarity needs its own kernel.
    const std::vector<uint32_t> spv = compile(kBfmSelectSrc1);
    ASSERT_FALSE(spv.empty());
    const std::vector<uint32_t> zeros(kLanes, 0u);
    std::vector<uint32_t> readback;
    const std::vector<float> got = prosper::test::run_compute(spv, lane_indices(), kLanes, kLanes,
                                                              zeros, {}, nullptr, 64, &readback);
    if (got.empty()) GTEST_SKIP() << "no Vulkan compute device";
    ASSERT_GE(readback.size(), 2u);
    EXPECT_EQ(readback[0], 0u) << "vcc_lo word (bfm low half)";
    EXPECT_EQ(readback[1], 0x80000000u) << "vcc_hi word (bfm high half)";
    // out_vgpr v2 carries the lo word's BITS (integer 0 here).
    uint32_t out_bits = 0;
    ASSERT_FALSE(got.empty());
    std::memcpy(&out_bits, &got[0], sizeof(out_bits));
    EXPECT_EQ(out_bits, 0u) << "output VGPR carries the stored lo word";
}

// The DATA words across the width/offset split points, on device. Each kernel is kBfmSelectSrc1
// with only the s_bfm_b64 word changed (ssrc0 = width, ssrc1 = offset, both inline integers).
// bfm(33, 0) is the arm that needs the high ones to survive a zero offset.
TEST(BfmB64ScalarData, WideAndShiftedDataWordsAreExactOnDevice) {
    struct Case {
        uint32_t bfm_word, lo, hi;
        const char* what;
    };
    const Case cases[] = {
        {0x928480A1u, 0xFFFFFFFFu, 0x00000001u, "bfm(33, 0) = 0x1_ffffffff"},
        {0x928494A8u, 0xFFF00000u, 0x0FFFFFFFu, "bfm(40, 20) = bits [20, 60)"},
        {0x9284A488u, 0x00000000u, 0x00000FF0u, "bfm(8, 36) = bits [36, 44)"},
    };
    for (const Case& c : cases) {
        uint32_t code[std::size(kBfmSelectSrc1)];
        std::memcpy(code, kBfmSelectSrc1, sizeof code);
        code[3] = c.bfm_word;
        const std::vector<uint32_t> spv = compile(code);
        ASSERT_FALSE(spv.empty()) << c.what;
        const std::vector<uint32_t> zeros(kLanes, 0u);
        std::vector<uint32_t> readback;
        const std::vector<float> got = prosper::test::run_compute(
            spv, lane_indices(), kLanes, kLanes, zeros, {}, nullptr, 64, &readback);
        if (got.empty()) GTEST_SKIP() << "no Vulkan compute device";
        ASSERT_GE(readback.size(), 2u) << c.what;
        EXPECT_EQ(readback[0], c.lo) << c.what << ": low word";
        EXPECT_EQ(readback[1], c.hi) << c.what << ": high word";
    }
}
