// test_smem_pointer_provenance — an SRT table pointer is provenance ONLY while nothing reads it as
// data, and "nothing reads it" has to include the reads that decode no source operand.
//
// A shader whose resources all live in a shader resource table reaches them through a pointer:
// `s_load_dwordx2 s[2:3], s[0:1], imm` fetches a table address, and a later `s_load_dwordx4/x8`
// through it fetches the real V#/T#/S#. The front half already follows that chain out of guest
// memory and publishes each descriptor at its exact consumer PC, so the pointer itself is
// provenance in SPIR-V and `proven_smem_pointer_loads` represents it with zero placeholders.
//
// That substitution is only sound while the pointer is never used as a VALUE. The proof therefore
// scans every instruction for a read of the loaded pair — and its first version scanned only
// decoded SOURCE operands, which is not the same set. SOPK decodes none at all (`n_src` stays 0),
// so `s_cmpk_eq_i32 s2, 0` read the pointer's low word with the proof none the wiser: the load was
// admitted, replaced by zero, and the shader computed on that zero and rendered silently wrong
// instead of being refused. SOP1's conditional moves and bitset forms read their destination the
// same way.
//
// Pure (no Vulkan), so it runs in CI.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <cstdio>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

// SMEM word0: [31:26]=0b111101, opcode[25:18], SDATA[12:6], SBASE[5:0] (the SGPR PAIR index, so the
// register number halved). word1: SOFFSET[31:25] (125 = NULL = immediate-only), OFFSET[20:0].
constexpr uint32_t smem_w0(uint32_t opcode, uint32_t sdata, uint32_t sbase_reg) {
    return 0xF4000000u | (opcode << 18) | (sdata << 6) | (sbase_reg >> 1);
}
constexpr uint32_t smem_w1_imm(uint32_t byte_offset) { return (125u << 25) | byte_offset; }

// SOPK: [31:28]=0b1011, opcode[27:23], SDST[22:16], SIMM16[15:0].
constexpr uint32_t sopk(uint32_t opcode, uint32_t sdst, uint32_t simm16) {
    return 0xB0000000u | (opcode << 23) | (sdst << 16) | (simm16 & 0xFFFFu);
}

constexpr uint32_t kSmemLoadDwordX2 = 0x01, kSmemLoadDwordX4 = 0x02;
constexpr uint32_t kSopkCmpkEqI32   = 0x03;   // in the s_cmpk_*/addk/mulk band the proof must catch
constexpr uint32_t kEndpgm          = 0xBF810000u;

// No binding 2. The fallback that admits an unresolved scalar load only exists when the table
// actually binds a constant/vertex buffer there, so leaving it out is what makes a refused pointer
// observable as a refused SHADER rather than as a silent fall-through to binding 2.
ShaderResourceTable srt_table() {
    ShaderResourceTable rt;
    ShaderResource texture{};
    texture.cls = ResourceClass::Texture;
    texture.binding = 5;
    texture.img_dim = 1;
    texture.width = texture.height = 4;
    texture.srt_offset = 0x10;
    rt.resources.push_back(texture);
    return rt;
}

ComputeShaderConfig srt_config() {
    ComputeShaderConfig config;
    config.user_sgprs.resize(20);       // s0..s19 are entry-time user data
    return config;
}

bool recompiles(const std::vector<uint32_t>& code) {
    const ShaderResourceTable rt = srt_table();
    return !recompile_compute(code.data(), code.size(), &rt, srt_config()).empty();
}

// The pointer chase both arms share: fetch a table address into s[2:3], then fetch a descriptor
// through it. The second load is what makes s[2:3] a proven POINTER rather than an unread value —
// the proof requires at least one SBASE use, so it asserts a shape instead of merely failing to
// find a counterexample.
std::vector<uint32_t> pointer_chase(std::vector<uint32_t> extra) {
    std::vector<uint32_t> code{
        smem_w0(kSmemLoadDwordX2, /*sdata=*/2, /*sbase=*/0), smem_w1_imm(0x00),
        smem_w0(kSmemLoadDwordX4, /*sdata=*/8, /*sbase=*/2), smem_w1_imm(0x10),
    };
    code.insert(code.end(), extra.begin(), extra.end());
    code.push_back(kEndpgm);
    return code;
}

}  // namespace

int main() {
    printf("test_smem_pointer_provenance\n");

    // Arm 1 — control. Nothing reads the pair as data, so the pointer is provenance and the shader
    // recompiles. Without this the two arms below could both be "refused" for an unrelated reason
    // and the test would assert nothing.
    CHECK(recompiles(pointer_chase({})),
          "control: an SRT pointer read only as an s_load SBASE is provenance and recompiles");

    // Arm 2 — the discriminator's own validity. The SAME SOPK instruction, reading a register
    // OUTSIDE the pointer pair, must still recompile. If it did not, arm 3's refusal would be
    // evidence that prosper cannot emit s_cmpk, not evidence about the pointer.
    CHECK(recompiles(pointer_chase({sopk(kSopkCmpkEqI32, /*sdst=*/20, 0)})),
          "control: the same s_cmpk_eq_i32 on an unrelated SGPR does not disturb the pointer");

    // Arm 3 — the defect. `s_cmpk_eq_i32 s2, 0` reads the pointer's low word as ordinary scalar
    // data, and SOPK decodes NO source operands, so a scan over `n_src` sees nothing. The pointer
    // must not be admitted, and with no binding-2 fallback the shader must be refused outright.
    // Before the implicit-destination-read check this arm RECOMPILED — the load was replaced with
    // zero and the comparison silently read that zero.
    CHECK(!recompiles(pointer_chase({sopk(kSopkCmpkEqI32, /*sdst=*/2, 0)})),
          "s_cmpk_eq_i32 reading the pointer's low word refuses the shader (implicit dest read)");

    // Arm 4 — the same hole one register up, so the check is a RANGE test and not an equality on
    // the load's base register.
    CHECK(!recompiles(pointer_chase({sopk(kSopkCmpkEqI32, /*sdst=*/3, 0)})),
          "the high word of the pair is covered too, not just the base register");

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
