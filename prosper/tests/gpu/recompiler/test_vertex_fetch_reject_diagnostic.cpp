// test_vertex_fetch_reject_diagnostic — the FORMAT-path `[mubuf-unresolved]` reject line must say
// WHICH provenance step failed, not merely that one did.
//
// Why this exists (#3137). A vertex attribute fetch whose V# cannot be resolved rejects the whole
// stage as `mode=unresolved-operand`, and that census line names an INSTRUCTION. Reading it, the
// natural conclusion is that the emitter lacks a lowering for `buffer_load_format_* ... idxen`, or
// that the stage arrived with no resource table at all. Both were proposed for #3137 and both are
// wrong: the opcodes are lowered, the table was present with five resources, and the descriptor was
// the only thing missing.
//
// The RAW MUBUF path (`[mubuf-raw-unresolved]`) has always printed the two fields that separate the
// remaining cases; the FORMAT path did not, so a live investigation had to re-run the title under
// PROSPER_DYNTRACE_FAIL to learn what one log line could have said:
//
//   * `rewritten=1` — the shader ASSEMBLES its own SRSRC (an s_load / patch sequence), so the two
//     direct-SGPR fallbacks are deliberately suppressed and only per-fetch (`by_fetch_pc`) or SRT-tag
//     provenance could have resolved it. A miss here is a front-half / user-data verdict.
//   * `rewritten=0` — the SRSRC still holds entry user data, both direct fallbacks WERE tried, and
//     the resource table genuinely has no entry at that SGPR.
//   * `pc_res` — whether the const-fold published a descriptor for this exact fetch instruction.
//
// Both arms below must still REJECT: an unresolvable V# is fail-visible by design and this test does
// not change that. What it pins is that the reject explains itself.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

// Recompile with stderr redirected to a scratch file and return everything it printed. stderr is not
// restored afterwards; every assertion in this test reports on stdout.
static std::string recompile_capturing_stderr(const uint32_t* code, size_t dwords,
                                              const ShaderResourceTable* rt,
                                              const char* scratch,
                                              bool* rejected) {
    std::fflush(stderr);
    if (!std::freopen(scratch, "w+", stderr)) { printf("  [FAIL] cannot redirect stderr\n"); fails++; return {}; }
    const std::vector<uint32_t> spirv = recompile_vertex(code, dwords, rt);
    std::fflush(stderr);
    *rejected = spirv.empty();
    std::string text;
    if (FILE* f = std::fopen(scratch, "rb")) {
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
        std::fclose(f);
    }
    std::remove(scratch);
    return text;
}

static bool has(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

int main() {
    printf("== test_vertex_fetch_reject_diagnostic ==\n");
    set_test_env("PROSPER_DBG", "1");

    // A resource table that is PRESENT and non-empty but describes an unrelated buffer, so
    // allow_smem is on (rt != nullptr) and every SRSRC lookup below misses. sgpr_base 20 collides
    // with neither arm's SRSRC.
    ShaderResourceTable rt;
    ShaderResource other{};
    other.cls = ResourceClass::ConstantBuffer;
    other.format = DataFormat::Float32;
    other.num_components = 4;
    other.binding = 2;
    other.gpu_addr = 0x1000;
    other.size = 256;
    other.sgpr_base = 20;
    rt.resources.push_back(other);

    // Arm 1 — the shape #3137 actually hit: the shader BUILDS its own SRSRC (here the minimal form,
    // four scalar writes over s[4:7]) and then fetches through it. Encodings assembled with
    // `llvm-mc -arch=amdgcn -mcpu=gfx1030`.
    //   s_mov_b32 s4..s7, 0
    //   buffer_load_format_xyzw v[0:3], v0, s[4:7], 0 idxen     <- pc 4
    //   exp pos0 v0, v1, v2, v3 done
    //   s_endpgm
    const uint32_t rewritten_srsrc[] = {
        0xBE840380u, 0xBE850380u, 0xBE860380u, 0xBE870380u,
        0xE00C2000u, 0x80010000u,
        0xF80008CFu, 0x03020100u,
        0xBF810000u,
    };
    // Arm 2 — the same fetch through an UNTOUCHED entry-user-data SRSRC, s[8:11]. Both direct-SGPR
    // fallbacks are attempted here and miss because the table has no entry at s8.
    //   buffer_load_format_xyzw v[0:3], v0, s[8:11], 0 idxen    <- pc 0
    //   exp pos0 v0, v1, v2, v3 done
    //   s_endpgm
    const uint32_t direct_srsrc[] = {
        0xE00C2000u, 0x80020000u,
        0xF80008CFu, 0x03020100u,
        0xBF810000u,
    };

    bool rejected_rewritten = false, rejected_direct = false;
    const std::string log_rewritten = recompile_capturing_stderr(
        rewritten_srsrc, std::size(rewritten_srsrc), &rt,
        "test_vertex_fetch_reject_diagnostic_rewritten.log", &rejected_rewritten);
    const std::string log_direct = recompile_capturing_stderr(
        direct_srsrc, std::size(direct_srsrc), &rt,
        "test_vertex_fetch_reject_diagnostic_direct.log", &rejected_direct);

    CHECK(rejected_rewritten, "an unresolvable in-shader SRSRC still rejects the stage (fail-visible)");
    CHECK(rejected_direct, "an unresolvable entry-user-data SRSRC still rejects the stage");

    CHECK(has(log_rewritten, "[mubuf-unresolved] pc=4 srsrc=s4"),
          "the format-path reject names the fetch pc and its SRSRC SGPR");
    CHECK(has(log_rewritten, "rewritten=1"),
          "#3137: an SRSRC the shader assembled itself reports rewritten=1 "
          "(direct-SGPR provenance was suppressed, not missed)");
    CHECK(has(log_rewritten, "pc_res=null"),
          "#3137: a fetch with no const-fold descriptor reports pc_res=null");
    CHECK(has(log_rewritten, "srt_tag=NONE"),
          "an SRSRC assembled without an s_load carries no SRT tag");

    CHECK(has(log_direct, "[mubuf-unresolved] pc=0 srsrc=s8"),
          "the entry-user-data arm names its own fetch pc and SRSRC SGPR");
    CHECK(has(log_direct, "rewritten=0"),
          "#3137: an untouched entry-user-data SRSRC reports rewritten=0 "
          "(both direct-SGPR fallbacks were tried and the table has no entry)");
    CHECK(has(log_direct, "pc_res=null"),
          "the entry-user-data arm also reports the missing per-fetch descriptor");

    // The discriminator is only worth anything if the two arms actually differ. Without this the
    // test would pass against a line that hardcoded either value.
    CHECK(has(log_rewritten, "rewritten=1") && !has(log_rewritten, "rewritten=0") &&
              has(log_direct, "rewritten=0") && !has(log_direct, "rewritten=1"),
          "the two arms report OPPOSITE rewritten= verdicts (the field is derived, not constant)");

    set_test_env("PROSPER_DBG", nullptr);
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
