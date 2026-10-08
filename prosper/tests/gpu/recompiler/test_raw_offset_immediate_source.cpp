// A memory-fed register offset whose x1 source load has a NON-ZERO immediate (#3135, Kena's indexed
// NGG prolog b77161c6: `s_load_dword s38, s[18:19], 0x4`, then `s_lshl_b32 vcc_hi, s38, 4`,
// `s_and_b32 vcc_lo, vcc_hi, 0x1f0` and `s_load_dwordx4 s[8:11], s[16:17], vcc_lo`).
//
// The raw-wide proof, the fold and the emitter used to admit only an immediate-ZERO source. The fold
// already observes the words at the EFFECTIVE address (immediate included) and owns exactly those
// bytes, so the immediate is no obstacle -- provided the emitted source load reads its snapshot from
// index zero rather than from `immediate / 4`. The execution arm pins that: the program stores the
// latched selector itself, so an emitter that indexed the 4-byte snapshot by the immediate would
// store zero.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/compute_runner.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

namespace {

std::vector<uint32_t> register_wide_proof(const std::vector<uint32_t>& code,
                                          std::vector<uint32_t>* sources = nullptr) {
    std::vector<Rdna2Inst> decoded;
    rdna2_walk(code.data(), code.size(), decoded);
    return rdna2_proven_raw_register_wide_data_loads(decoded, sources);
}

// x1 source at `immediate`, the selected x4 at a VCC_HI register offset, then two data observers
// stored to the output V# at s[12:15]: the selected record's last word at +0 and the latched
// selector itself at +4.
std::vector<uint32_t> program(uint32_t immediate) {
    return {
        0xf4000101u, 0xfa000000u | immediate,   // s_load_dword s4, s[2:3], immediate
        0x8f6b8404u,   // s_lshl_b32 vcc_hi, s4, 4
        0x876bff6bu, 0x000001f0u,   // s_and_b32 vcc_hi, vcc_hi, 0x1f0
        0xf4080200u, 0xd6000000u,   // s_load_dwordx4 s[8:11], s[0:1], vcc_hi
        0xbf8cc07fu,   // s_waitcnt lgkmcnt(0)
        0x7e00020bu,   // v_mov_b32 v0, s11
        0x7e020204u,   // v_mov_b32 v1, s4
        0xe0700000u, 0x80030000u,   // buffer_store_dword v0, off, s[12:15], 0
        0xe0700004u, 0x80030100u,   // buffer_store_dword v1, off, s[12:15], 0 offset:4
        0xbf810000u,   // s_endpgm
    };
}

struct Inputs {
    alignas(16) std::array<uint32_t, 128> records{};
    alignas(16) std::array<uint32_t, 4> selector_words{0xdeadu, 2u, 0xbeefu, 0u};   // [1] = 2
    alignas(16) std::array<uint32_t, 4> output{};
    std::array<uint32_t, 16> user{};
    Inputs() {
        for (size_t i = 0; i < records.size(); ++i) records[i] = 100u + static_cast<uint32_t>(i);
        const auto table = reinterpret_cast<uint64_t>(records.data());
        const auto selector = reinterpret_cast<uint64_t>(selector_words.data());
        const auto out = reinterpret_cast<uint64_t>(output.data());
        user[0] = static_cast<uint32_t>(table);
        user[1] = static_cast<uint32_t>(table >> 32u);
        user[2] = static_cast<uint32_t>(selector);
        user[3] = static_cast<uint32_t>(selector >> 32u);
        user[12] = static_cast<uint32_t>(out);
        user[13] = static_cast<uint32_t>((out >> 32u) & 0xffffu);
        user[14] = sizeof(output);
        user[15] = (22u << 12) | 0xfacu;
    }
};

TEST(RawOffsetImmediateSource, AnImmediateOffsetSourceIsProvenAndOwnsItsEffectiveBytes) {
    Inputs in;
    const auto code = program(4u);
    std::vector<uint32_t> sources;
    EXPECT_EQ(register_wide_proof(code, &sources), std::vector<uint32_t>{5u})
        << "the selected x4 at pc 5 is proven through the x1 read at immediate 4";
    EXPECT_EQ(sources, std::vector<uint32_t>{0u}) << "pc 0 is its latched scalar source";

    ShaderResourceTable table;
    add_compute_buffer_resources(table, code.data(), code.size(), in.user.data(),
                                 static_cast<uint32_t>(in.user.size()));
    const ShaderResource* source = table.by_fetch_pc(0u);
    ASSERT_TRUE(source) << "the x1 source owns a snapshot";
    EXPECT_EQ(source->gpu_addr, reinterpret_cast<uint64_t>(&in.selector_words[1]))
        << "the EFFECTIVE address, immediate included";
    uint32_t latched = 0;
    ASSERT_TRUE(source->host_data && source->host_data_size == sizeof(latched));
    std::memcpy(&latched, source->host_data, sizeof(latched));
    EXPECT_EQ(latched, 2u);
    const ShaderResource* selected = table.by_fetch_pc(5u);
    ASSERT_TRUE(selected);
    EXPECT_TRUE(valid_raw_register_snapshot_resource(*selected));
    EXPECT_EQ(selected->gpu_addr, reinterpret_cast<uint64_t>(in.records.data()) + 32u)
        << "selector 2 x 16 bytes";

    // A negative immediate stays out of the proof (the emitter refuses a wrapping offset).
    EXPECT_TRUE(register_wide_proof(program(0x1ffffcu)).empty());
    // The immediate-zero form keeps its proof (the #4578 shape).
    EXPECT_EQ(register_wide_proof(program(0u)), std::vector<uint32_t>{5u});
}

// Executed: the emitted x1 load reads its owned snapshot from index zero. Output word 0 is the
// selected record's last word (100 + 8 + 3), word 1 the latched selector (2).
TEST(RawOffsetImmediateSource, TheEmittedSourceReadsItsSnapshotFromIndexZero) {
    Inputs in;
    const auto code = program(4u);
    ShaderResourceTable table;
    add_compute_buffer_resources(table, code.data(), code.size(), in.user.data(),
                                 static_cast<uint32_t>(in.user.size()));
    // run_compute binds storage buffers 2 and 3 to `cbuf` / `cbuf1` and 4.. to the extras; only
    // 2 and 3 read back. Put the selected record on 2, the output on 3 and the source on 4.
    std::vector<uint32_t> selected_words, source_words;
    bool have_output = false;
    for (ShaderResource& r : table.resources) {
        if (r.fetch_pc == 5u && valid_raw_register_snapshot_resource(r)) {
            r.binding = 2;   // a guest-backed snapshot: its bytes are read at gpu_addr
            selected_words.assign(r.size / 4u, 0u);
            std::memcpy(selected_words.data(), reinterpret_cast<const void*>(r.gpu_addr), r.size);
        } else if (r.fetch_pc == 0u && r.host_data && r.host_data_size == 4u) {
            r.binding = 4;
            source_words.assign(1u, 0u);
            std::memcpy(source_words.data(), r.host_data, 4u);
        } else {
            r.binding = 3;
            have_output = true;
        }
    }
    ASSERT_EQ(selected_words.size(), 4u);
    ASSERT_EQ(source_words, std::vector<uint32_t>{2u});
    ASSERT_TRUE(have_output) << "the output V# at s[12:15] is a resource";
    ComputeShaderConfig config;
    config.user_sgprs.assign(in.user.begin(), in.user.end());
    config.local_x = 64;
    const auto spirv = recompile_compute(code.data(), code.size(), &table, config);
    ASSERT_FALSE(spirv.empty());
    std::vector<uint32_t> out;
    const std::array<std::vector<uint32_t>, 3> extras = {source_words, {}, {}};
    const auto ran = prosper::test::run_compute(spirv, std::vector<float>(64, 0.0f), 64, 64,
                                                selected_words, std::vector<uint32_t>(4, 0u), &out,
                                                64, nullptr, nullptr, &extras);
    if (ran.empty()) GTEST_SKIP() << "no Vulkan compute device";
    ASSERT_GE(out.size(), 2u);
    EXPECT_EQ(out[0], 111u) << "the selected record's last word";
    EXPECT_EQ(out[1], 2u) << "the latched selector, read from snapshot index 0";
}

}   // namespace
