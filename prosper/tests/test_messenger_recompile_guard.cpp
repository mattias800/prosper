// test_messenger_recompile_guard (#228) — a CI guard against silently re-breaking The Messenger's
// real shaders. Two recompiler rejects added for OTHER titles (#199 SMEM SOFFSET, #201 packed-
// vertex-fetch alignment) each regressed The Messenger to a 100% blue screen — its vertex shader
// stopped recompiling (vs=0) so all geometry was skipped — and both landed green because nothing
// guarded the Messenger's real shaders. This test commits a representative real scene VS+PS
// (captured RDNA2 binaries under tests/data/, trimmed to their s_endpgm) and asserts the recompiler
// still UNDERSTANDS every instruction they use.
//
// Why a coverage baseline and not "recompile returns non-empty": both real stages resolve their
// memory ops through a per-draw ShaderResourceTable (the VS's bindless vertex fetch, the PS's 13
// texture/constant bindings), built at boot from guest user-data + memory. Recompiling table-less
// therefore returns {} for both — verified below — and committing the tables would tie a binary
// fixture to the ShaderResource struct layout (ABI-fragile). recompile_coverage() instead reports,
// per instruction, whether the recompiler has a lowering for it (alu / export / table_dependent) or
// NOT (unsupported), independent of any table. The #199-class regression flips an instruction the
// Messenger uses into `unsupported`, which this test catches. The with-table recompile is covered by
// the local golden-image snapshot guard (tools/snapshot, from #227); this is its dump-free CI half.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static std::vector<uint32_t> load_shader(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> v(n / 4);
    size_t got = fread(v.data(), 4, v.size(), f);
    fclose(f);
    if (got != v.size()) return {};
    return v;
}

int main() {
    printf("== test_messenger_recompile_guard ==\n");
    const std::string dir = PROSPER_TEST_DATA_DIR;
    auto vs = load_shader((dir + "/messenger_scene_vs.bin").c_str());
    auto ps = load_shader((dir + "/messenger_scene_ps.bin").c_str());

    // The captured fixtures must be present and intact (a truncated/empty fixture would make the
    // coverage numbers meaningless — fail loudly instead of silently passing on nothing).
    CHECK(!vs.empty(), "committed Messenger scene VS binary present + readable");
    CHECK(!ps.empty(), "committed Messenger scene PS binary present + readable");
    CHECK(vs.size() == 555, "VS fixture is the expected 555 dwords (captured length to last s_endpgm)");
    CHECK(ps.size() == 43,  "PS fixture is the expected 43 dwords");
    if (vs.empty() || ps.empty()) { printf("== FAIL ==\n"); return 1; }

    RecompileCoverage cv = recompile_coverage(vs.data(), vs.size());
    RecompileCoverage cp = recompile_coverage(ps.data(), ps.size());
    printf("  VS coverage: total=%u alu=%u exports=%u table_dependent=%u unsupported=%u first_bad(fmt=%d op=0x%x)\n",
           cv.total, cv.alu, cv.exports, cv.table_dependent, cv.unsupported, cv.first_bad_fmt, cv.first_bad_op);
    printf("  PS coverage: total=%u alu=%u exports=%u table_dependent=%u unsupported=%u first_bad(fmt=%d op=0x%x)\n",
           cp.total, cp.alu, cp.exports, cp.table_dependent, cp.unsupported, cp.first_bad_fmt, cp.first_bad_op);

    // Baseline captured on master at the time this fixture landed (2026-07-10). `total`/`exports` are
    // pure properties of the bytecode — a change here means the DECODER mis-parsed the stream (a real
    // regression). `unsupported` may only legitimately DROP (a recompiler improvement adds a lowering);
    // an INCREASE is the #199/#201 blue-screen class — an instruction the game needs became unsupported.
    // So: total/exports are pinned exactly, unsupported is bounded above by its baseline.
    const uint32_t VS_TOTAL = 157, VS_EXPORTS = 6, VS_UNSUP_MAX = 3;
    const uint32_t PS_TOTAL = 37,  PS_EXPORTS = 1, PS_UNSUP_MAX = 0;

    CHECK(cv.total == VS_TOTAL, "VS: decoded instruction count stable (decoder intact)");
    CHECK(cv.exports == VS_EXPORTS, "VS: export count stable (position/param exports still recognized)");
    CHECK(cv.unsupported <= VS_UNSUP_MAX,
          "VS: no NEW unsupported instruction (the #199/#201 re-blue regression class)");

    CHECK(cp.total == PS_TOTAL, "PS: decoded instruction count stable (decoder intact)");
    CHECK(cp.exports >= PS_EXPORTS, "PS: MRT export still recognized");
    CHECK(cp.unsupported <= PS_UNSUP_MAX,
          "PS: no NEW unsupported instruction (every PS op still has a lowering)");

    // Document (and lock) that both real stages need their resource tables: table-less recompile is
    // empty. If a future change makes a stage recompile table-less, that's fine — this assertion just
    // records today's reality and isn't a regression signal, so keep it a soft print, not a hard fail.
    auto rv = recompile_vertex(vs.data(), vs.size(), nullptr);
    auto rf = recompile_fragment(ps.data(), ps.size(), nullptr);
    printf("  (table-less recompile: vs=%zu fs=%zu dwords — both stages resolve memory ops via their "
           "per-draw ShaderResourceTable, so table-less is expected empty)\n", rv.size(), rf.size());

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
