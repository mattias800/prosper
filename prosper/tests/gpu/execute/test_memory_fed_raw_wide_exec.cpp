#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/compute_runner.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper::gpu;
static int failures = 0;
#define CHECK(c, text) do { if (!(c)) { std::printf("[FAIL] %s\n", text); ++failures; } \
    else std::printf("[ok] %s\n", text); } while (0)

int main() {
    alignas(16) std::array<uint32_t, 128> source{};
    for (size_t i = 0; i < source.size(); ++i) source[i] = 101u + 17u * static_cast<uint32_t>(i);
    uint32_t selector = 2u;
    const auto address = reinterpret_cast<uint64_t>(source.data());
    const auto selector_address = reinterpret_cast<uint64_t>(&selector);
    const std::array<uint32_t, 4> user{static_cast<uint32_t>(address),
        static_cast<uint32_t>(address >> 32u), static_cast<uint32_t>(selector_address),
        static_cast<uint32_t>(selector_address >> 32u)};
    for (uint32_t count : {4u, 8u}) {
        const uint32_t code[] = {
            0xf4000101u, 0xfa000000u,       // x1 s4,entry s[2:3],0
            0xbe820380u, 0xbe830380u,       // overwrite source pointer after read
            0x8f6b8404u,                   // s_lshl_b32 vcc_hi,s4,4
            0x876bff6bu, 0x000001f0u,      // mask to bounded fixture range
            count == 4u ? 0xf4080200u : 0xf40c0200u, 0xd6000000u,
            count == 4u ? 0x7e000c0bu : 0x7e000c0fu,
            0xbf810000u,
        };
        auto execute = [&](bool mutate_after_realization = false) {
            ShaderResourceTable table;
            add_compute_buffer_resources(table, code, std::size(code), user.data(), user.size());
            assign_convention_bindings(table, 2u);
            const auto* scalar = table.by_fetch_pc(0u);
            const auto* wide = table.by_fetch_pc(7u);
            if (!scalar || scalar->binding != 2u || !wide || wide->binding != 3u ||
                !scalar->host_data || scalar->host_data_size != sizeof(uint32_t) ||
                !valid_raw_register_snapshot_resource(*wide)) return std::vector<float>{};
            if (mutate_after_realization) selector = 3u;
            const auto spv = recompile_valu(code, std::size(code), 1u, 0u, &table);
            if (spv.empty()) return std::vector<float>{};
            std::vector<uint32_t> current(count);
            std::memcpy(current.data(), reinterpret_cast<const void*>(wide->gpu_addr), count * 4u);
            uint32_t latched = UINT32_MAX;
            std::memcpy(&latched, scalar->host_data, sizeof(latched));
            return prosper::test::run_compute(spv, {0.0f}, 1u, 1u, {latched}, current);
        };
        selector = 2u;
        const uint32_t index = 8u + count - 1u;
        const uint32_t before = source[index];
        const auto initial = execute();
        CHECK(initial.size() == 1u && initial[0] == static_cast<float>(before),
              "Wave64 scalar VCC offset returns the last word of the actual selected range");
        const auto latched = execute(true);
        CHECK(selector == 3u && latched.size() == 1u && latched[0] == static_cast<float>(before),
              "post-realization selector mutation preserves the numeric result of the latched draw");
        selector = 2u;
        source[index] = 12345u;
        const auto changed = execute();
        CHECK(changed.size() == 1u && changed[0] == 12345.0f,
              "changed source bytes reach the numeric wide-load consumer");
        source[index] = before;
        selector = 3u;
        const auto shifted = execute();
        CHECK(shifted.size() == 1u && shifted[0] == static_cast<float>(source[index + 4u]),
              "changed memory-fed selector changes the real GPU result");
    }
    std::printf("memory-fed raw wide GPU failures: %d\n", failures);
    return failures != 0;
}
