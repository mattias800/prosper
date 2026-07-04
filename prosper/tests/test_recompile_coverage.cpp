// test_recompile_coverage — recompile_coverage() reports per-instruction recompiler support without
// requiring a complete vertex/fragment. Pure (no Vulkan), so it runs in CI. It also drives the
// data-driven coverage report over the real game shaders (shader_histo).
#include "../src/gpu/rdna2_to_spirv.hpp"
#include <cstdio>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_recompile_coverage ==\n");

    // Fully-supported ALU: v_add_f32 v0,v0,v1 ; v_mul_f32 v0,v0,v2 ; s_endpgm.
    const uint32_t ok_code[] = { 0x06000300u, 0x10000500u, 0xBF810000u };
    RecompileCoverage a = recompile_coverage(ok_code, sizeof(ok_code)/sizeof(ok_code[0]));
    printf("  supported-kernel: total=%u alu=%u exports=%u unsupported=%u\n", a.total, a.alu, a.exports, a.unsupported);
    CHECK(a.total == 2 && a.alu == 2 && a.unsupported == 0 && a.first_bad_fmt < 0,
          "a fully-handled ALU kernel reports 100% coverage");

    // Contains an unsupported op: v_add_f32 ; s_branch +5 (unconditional -> rejected) ; s_endpgm.
    const uint32_t bad_code[] = { 0x06000300u, 0xbf820005u, 0xBF810000u };
    RecompileCoverage b = recompile_coverage(bad_code, sizeof(bad_code)/sizeof(bad_code[0]));
    printf("  mixed-kernel: total=%u alu=%u unsupported=%u first_bad_op=0x%x\n", b.total, b.alu, b.unsupported, b.first_bad_op);
    CHECK(b.total == 2 && b.alu == 1 && b.unsupported == 1 && b.first_bad_fmt >= 0 && b.first_bad_op == 0x02,
          "an unconditional s_branch is reported as the first unsupported instruction");

    // A forward VCC branch is not covered by EXEC predication. Treating it as a no-op would execute
    // the skipped block even when VCC says to branch, so the recompiler must reject it for now.
    const uint32_t vcc_branch[] = { 0x7da80300u, 0xbf860001u, 0x4a060300u, 0xBF810000u };
    RecompileCoverage c = recompile_coverage(vcc_branch, sizeof(vcc_branch)/sizeof(vcc_branch[0]));
    CHECK(c.unsupported == 1 && c.first_bad_fmt >= 0 && c.first_bad_op == 0x06,
          "a forward s_cbranch_vccz is rejected instead of linearized as a no-op");
    CHECK(recompile_valu(vcc_branch, sizeof(vcc_branch)/sizeof(vcc_branch[0]), 2, 3).empty(),
          "the production recompiler rejects the unsafe forward VCC branch");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
