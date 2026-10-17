// The MUBUF address (ir/ops/buffer_address.cpp) per lane, through the real recompiler and the
// integer SPIR-V interpreter, in Wave64 and Wave32. The program is
//   <harness loads: v2 = index, v3 = byte offset, v1 = old VDST, v4 = active; v_cmpx_ne_u32 0, v4>
//   [s_mov_b32 s12, soffset]
//   buffer_load_dword v1, v[2:3] (or v2 / v3), s[8:11], <s12 | inline | null> [idxen] [offen]
//       offset:OFFSET
// reading the harness's own buffer (binding 3, STRIDE 4), whose contents are the four input
// arrays. Every address lies past the first array: the interpreter runs the invocations one after
// another, and each one's final store overwrites its own word of that array. Every lane names its own address, so an active lane must come back with the word at
// eval_buffer_address's dword index and an inactive lane with its old v1. The out-of-range
// contract (an address past the bound range reads 0) is test_ir_buffer_address_robustness.
#include "gpu/recompiler/ir/ops/buffer_address.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {

constexpr uint32_t kSoffsetSgpr = 12;
enum class Soffset : uint8_t { Sgpr, Inline, Null };

struct Form {
    bool idxen, offen;
    uint32_t offset;
    Soffset soffset;
    uint32_t soffset_value;   // Sgpr: any word; Inline: 0..64; Null: 0
};

const Form kForms[] = {
    {false, false, 300, Soffset::Sgpr, 32},   {true, false, 8, Soffset::Inline, 16},
    {false, true, 0, Soffset::Null, 0},      {true, true, 4, Soffset::Sgpr, 12},
    {true, true, 0, Soffset::Inline, 0},     {false, true, 36, Soffset::Inline, 64},
};

std::vector<uint32_t> load_words(const Form& f) {
    std::vector<uint32_t> words;
    uint32_t soffset = 0x7du;   // SGPR_NULL
    if (f.soffset == Soffset::Sgpr) {
        words.insert(words.end(), {0xbe8003ffu | (kSoffsetSgpr << 16), f.soffset_value});
        soffset = kSoffsetSgpr;
    } else if (f.soffset == Soffset::Inline) {
        soffset = 0x80u + f.soffset_value;
    }
    const uint32_t vaddr = (f.offen && !f.idxen) ? kVsrc1 : kVsrc0;
    words.push_back(0xe0300000u | (uint32_t(f.idxen) << 13) | (uint32_t(f.offen) << 12) | f.offset);
    words.push_back((soffset << 24) | (2u << 16) | (kVdst << 8) | vaddr);
    return words;
}

// Indices 64..103 and dword-aligned byte offsets 256..508 keep every address of every form
// inside the second to fourth arrays of the harness's 4 * 64-dword buffer: bytes 256 to at most
// 4 + 12 + 103 * 4 + 508 = 936.
LaneInputs address_inputs(uint32_t seed) {
    LaneInputs in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        seed = seed * 1664525u + 1013904223u;
        in.s0[lane] = 64u + (seed >> 8) % 40u;
        in.s1[lane] = 256u + ((seed >> 20) % 64u) * 4u;
    }
    return in;
}

uint32_t buffer_word(const LaneInputs& in, uint32_t dword) {
    const uint32_t lanes = static_cast<uint32_t>(in.lanes());
    const uint32_t array = dword / lanes, lane = dword % lanes;
    switch (array) {
        case 1: return in.s1[lane];
        case 2: return in.old_d[lane];
        case 3: return in.active[lane] ? 1u : 0u;
    }
    ADD_FAILURE() << "dword " << dword << " is outside the harness buffer";
    return 0;
}

TEST(IrBufferAddressExecution, EveryLaneLoadsTheWordAtItsOwnAddress) {
    for (const uint32_t wave : {64u, 32u})
        for (const Form& f : kForms) {
            const LaneInputs in = address_inputs(0xb0f0u + f.offset);
            const LaneRun run = run_vector_op_per_lane(load_words(f), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            // STRIDE 4 is the harness V#'s; the form reads index, byte offset and SOFFSET.
            const BufferAddressShape shape{BufferAddressForm::Faithful, f.offset, 4u, f.idxen,
                                           f.offen};
            size_t active = 0;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                const auto address =
                    eval_buffer_address(shape, {in.s0[lane], in.s1[lane], f.soffset_value, 0u});
                ASSERT_TRUE(address[1].has_value());
                ASSERT_GE(*address[1], in.lanes()) << "the address must not reach the first array";
                const uint32_t want = in.active[lane] ? buffer_word(in, *address[1]) : in.old_d[lane];
                EXPECT_EQ(run.d[lane], want) << "idxen " << f.idxen << " offen " << f.offen
                                             << " offset " << f.offset << " wave " << wave
                                             << " lane " << lane;
                active += in.active[lane];
            }
            EXPECT_GT(active, in.lanes() / 2);
        }
}

// The positive control for the arm above: the lanes really do name different addresses, so a
// lowering that read one lane's address for all would fail it.
TEST(IrBufferAddressExecution, LanesNameDifferentAddresses) {
    const LaneInputs in = address_inputs(0xb0f0u);
    const BufferAddressShape shape{BufferAddressForm::Faithful, 4u, 4u, true, true};
    std::vector<uint32_t> seen;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        seen.push_back(*eval_buffer_address(shape, {in.s0[lane], in.s1[lane], 12u, 0u})[1]);
    std::sort(seen.begin(), seen.end());
    EXPECT_GT(std::unique(seen.begin(), seen.end()) - seen.begin(), 32);
}

// The synthetic witness corpus (HOW_TO_ADD_A_FAMILY.md, "Memory instructions"): every form in
// both wave sizes. Every program must replay.
TEST(IrBufferAddressExecution, ExportsReplayableWitnessPrograms) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root("ir_buffer_address_witness");
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const fs::path dump = compare_witness_dump("ir_buffer_address_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    size_t count = 0;
    for (const uint32_t wave : {64u, 32u})
        for (const Form& f : kForms) {
            write_compare_witness_program(dump, 0x5b0a000000ull + count * 0x100, count,
                                          lane_program(load_words(f), 64, wave));
            ++count;
        }
    EXPECT_EQ(count, 12u);
}

}   // namespace
