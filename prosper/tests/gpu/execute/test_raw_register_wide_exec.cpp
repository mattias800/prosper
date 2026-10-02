#include "gpu/execute/gpu_execute.hpp"
#include <gtest/gtest.h>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/compute_runner.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper::gpu;
#define CHECK(c, text) EXPECT_TRUE(c) << (text)

TEST(RawRegisterWideExec, Contract) {
    alignas(16) std::array<uint32_t, 64> source{};
    for (size_t i = 0; i < source.size(); ++i) source[i] = 101u + 17u * static_cast<uint32_t>(i);
    const auto address = reinterpret_cast<uint64_t>(source.data());
    std::array<uint32_t, 3> user = {
        static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u), 2u};
    for (uint32_t count : {4u, 8u}) {
        const uint32_t code[] = {
            0x8f148402u, // s_lshl_b32 s20,s2,4
            count == 4u ? 0xf4080200u : 0xf40c0200u, 0x28000010u,
            count == 4u ? 0x7e000c09u : 0x7e000c0fu, // s9 or s15 -> float
            0xbf810000u,
        };
        auto execute = [&]() {
            ShaderResourceTable table;
            add_compute_buffer_resources(table, code, std::size(code), user.data(), user.size());
            assign_convention_bindings(table, 2);
            const auto* resource = table.by_fetch_pc(1u);
            if (!resource || !valid_raw_register_snapshot_resource(*resource))
                return std::vector<float>{};
            const auto spv = recompile_valu(code, std::size(code), 1u, 0u, &table);
            if (spv.empty()) return std::vector<float>{};
            std::vector<uint32_t> current(count);
            std::memcpy(current.data(), reinterpret_cast<const void*>(resource->gpu_addr), count * 4u);
            return prosper::test::run_compute(spv, {0.0f}, 1u, 1u, current);
        };
        user[2] = 2u;
        const uint32_t selected = 12u + (count == 4u ? 1u : 7u);
        const uint32_t before = source[selected];
        const auto first = execute();
        source[selected] = 12345u;
        const auto changed = execute();
        CHECK(first.size() == 1u && changed.size() == 1u &&
              first[0] == static_cast<float>(before) && changed[0] == 12345.0f,
              count == 4u ? "x4 observes changed bytes at the resolved effective address" :
                             "x8 observes changed bytes in its last loaded word");
        source[selected] = before;
        user[2] = 3u;
        const auto shifted = execute();
        CHECK(shifted.size() == 1u && shifted[0] == static_cast<float>(source[selected + 4u]),
              "changed per-draw offset selects a different byte range on the GPU");
    }
}
