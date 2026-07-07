// test_spirv_builder — proves our own SPIR-V emitter (src/gpu/spirv_builder.cpp) produces valid,
// correct, runnable SPIR-V by executing it: build a compute shader (b[i] = a[i]*scale + bias) with
// the builder, run it on real Vulkan compute over known inputs, and assert the numeric outputs.
// This is the code-generation foundation for the RDNA2->SPIR-V recompiler, verified end-to-end
// (emit -> Vulkan accepts it -> numbers are right), and it validated structurally with spirv-val.
#include "../src/gpu/spirv_builder.hpp"
#include "compute_runner.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_spirv_builder ==\n");
    const float scale = 2.0f, bias = 1.0f;
    std::vector<uint32_t> spv = build_compute_scale_bias(scale, bias);
    CHECK(spv.size() > 5 && spv[0] == 0x07230203u, "builder emitted a SPIR-V module (magic ok)");

    const uint32_t N = 256;
    std::vector<float> in(N), expect(N);
    for (uint32_t i = 0; i < N; i++) { in[i] = (float)i * 0.25f - 7.0f; expect[i] = in[i] * scale + bias; }

    std::vector<float> got = prosper::test::run_compute(spv, in);
    CHECK(got.size() == N, "our SPIR-V compiled + ran on Vulkan (shader module accepted)");
    if (got.size() != N) { printf("== FAIL: shader did not run ==\n"); return 1; }

    uint32_t bad = 0; float worst = 0;
    for (uint32_t i = 0; i < N; i++) { float d = std::fabs(got[i] - expect[i]); if (d > 1e-4f) { bad++; worst = d > worst ? d : worst; } }
    printf("  N=%u mismatches=%u worst=%g (out[100]=%g expect=%g)\n", N, bad, worst, got[100], expect[100]);
    CHECK(bad == 0, "our emitted shader computes b[i] = a[i]*2 + 1 correctly (execution-differential)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
