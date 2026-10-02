// #4060: execute translated scalar FF1 against a bit-scanning CPU oracle.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/compute_runner.h"
#include <bit>
#include <cstdio>
#include <utility>
#include <vector>

using namespace prosper::gpu;
static int failures = 0, checks = 0;
#define CHECK(c, m) do { ++checks; if (!(c)) { std::printf("FAIL: %s\n", m); ++failures; } } while (0)

static uint32_t oracle(uint32_t word) {
    for (uint32_t bit = 0; bit < 32; ++bit) if ((word >> bit) & 1u) return bit;
    return 0xffffffffu;
}

static void execute(const std::vector<uint32_t>& code, const std::vector<uint32_t>& inputs,
                    bool selects_scc = false, bool scc = false) {
    // The scalar load route is already used by the existing compute fixture: binding2, offset8.
    // Compile ONCE before supplying/changing backing, so a constant answer cannot pass these arms.
    const auto module = recompile_valu(code.data(), code.size(), 0, 0, nullptr, 0,
                                      kDefaultComputePgmRsrc1, false, 1);
    CHECK(!module.empty(), "runtime scalar-buffer program recompiles");
    if (module.empty()) return;
    for (uint32_t input : inputs) {
        const auto result = prosper::test::run_compute(module, {0.0f}, 1, 1,
                                                       {0u, 0u, input}, {}, nullptr, 1);
        const uint32_t expected = selects_scc ? (scc ? 1u : 2u) : oracle(input);
        const bool correct = result.size() == 1 && std::bit_cast<uint32_t>(result[0]) == expected;
        CHECK(correct, "changed buffer yields exact 32-bit result and preserved SCC");
        if (!correct) std::printf("input=%08x expected=%08x output=%08x count=%zu\n", input,
            expected, result.empty() ? 0u : std::bit_cast<uint32_t>(result[0]), result.size());
    }
}

int main() {
    // s_buffer_load_dword s1,s[4:7],8; s_mov_b32 s36,s1; FF1 s37,s36; v_mov v0,s37.
    const std::vector<uint32_t> code{0xf4200042u, 0xfa000008u, 0xbea40301u,
                                    0xbea51324u, 0x7e000225u, 0xbf810000u};
    std::vector<uint32_t> values{0u, 0xffffffffu, 0xaaaaaaaau, 0x55555555u, 0x80010000u};
    for (uint32_t bit = 0; bit < 32; ++bit) values.push_back(1u << bit);
    execute(code, values);

    auto self = code;
    self[3] = 0xbea41324u;  // s_ff1_i32_b32 s36,s36
    self[4] = 0x7e000224u;
    execute(self, {0u, 0x80000000u, 0xaaaaaaaau});

    auto poisoned = code;
    poisoned.insert(poisoned.begin() + 3, 0xbea50381u); // s37=1: adjacent high word and old dst
    execute(poisoned, {0u, 0x80000000u, 0xaaaaaaaau});

    for (bool scc : {false, true}) {
        auto select = code;
        select.insert(select.begin() + 3, scc ? 0xbf068080u : 0xbf068180u);
        select.insert(select.begin() + 5, 0x85268281u); // s_cselect_b32 s38,1,2
        select[6] = 0x7e000226u;                       // v_mov v0,s38
        execute(select, {0u, 1u, 0x80000000u}, true, scc);
    }
    // Immediate inputs include zero, negative inline bits, a high-bit literal and an inline float.
    for (const auto& arm : std::vector<std::pair<uint32_t, uint32_t>>{
             {128u, 0u}, {193u, 0xffffffffu}, {240u, 0x3f000000u}, {255u, 0x80000000u}}) {
        std::vector<uint32_t> immediate{0xbea51300u | arm.first};
        if (arm.first == 255u) immediate.push_back(arm.second);
        immediate.insert(immediate.end(), {0x7e000225u, 0xbf810000u});
        const auto module = recompile_valu(immediate.data(), immediate.size(), 0, 0, nullptr, 0,
                                          kDefaultComputePgmRsrc1, false, 1);
        CHECK(!module.empty(), "immediate FF1 program recompiles");
        if (module.empty()) continue;
        const auto result = prosper::test::run_compute(module, {0.0f}, 1, 1, {}, {}, nullptr, 1);
        CHECK(result.size() == 1 && std::bit_cast<uint32_t>(result[0]) == oracle(arm.second),
              "immediate bit pattern matches the independent oracle");
    }
    std::printf("scalar_ff1_b32_exec: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
