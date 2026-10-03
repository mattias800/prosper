// test_gap_opcode_refusals — fail-visible refusal pins for six opcodes the compute recompiler
// decodes but does not lower: v_perm_b32, v_mad_i64_i32, v_div_fixup_f32, v_sad_u8, s_movrels_b32
// and image_gather4 (implicit LOD).
//
// Per the recompiler charter an unsupported op is a FATAL gap, and the loud refusal is only the
// backstop. These arms pin that backstop: the guest word decodes to the right instruction, the
// compile returns no module, and the terminal reject record names THIS instruction. They are pins,
// not goals. WHEN A LOWERING LANDS, ITS ARM GOES RED (the module is no longer empty); replace that
// arm with an execution test of the new lowering rather than deleting the assertion.
//
// Each arm is paired with a control built from the SAME program with ONLY the gap instruction
// swapped for a lowered sibling with identical operand fields (v_sad_u32 for the three VOP3A arms,
// v_mad_u64_u32 for the VOP3B arm, s_mov_b32 for s_movrels_b32, image_gather4_lz for
// image_gather4). The control compiling is what makes the refusal about the opcode rather than the
// operands, the resource table or the program shape.
//
// Why the reject `mode` is `unresolved-operand` and not `unknown-encoding`: `emit_alu` returns
// handled=true for every instruction in a VALU/SALU/MIMG format and reports a missing lowering by
// clearing `ok` in the format's fall-through arm, which the reject formatter cannot tell apart from
// an operand it failed to resolve (rdna2_emit_cfg.cpp, the `mode=` comment). So the mode is pinned
// as measured, and the opcode/pc/words fields plus the one-instruction-changed control carry the
// discrimination. If the taxonomy is ever split so a missing lowering reports its own mode, update
// the expected mode here.
//
// Encodings: every word below was assembled and disassembled with llvm-mc (-mcpu=gfx1010 and
// gfx1030), used as an encoding oracle only. No device is created; compile-only, runs everywhere.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {

bool is_vgpr(const Operand& o, uint32_t n) {
    return o.kind == OperandKind::VGPR && o.value == static_cast<int>(n);
}
bool is_sgpr(const Operand& o, uint32_t n) {
    return o.kind == OperandKind::SGPR && o.value == static_cast<int>(n);
}

constexpr uint32_t kEndpgm = 0xbf810000u;

// v_mov_b32 v1, 1 / v2, 2 / v3, 3 / v4, 0 -- inline-constant sources (0x81..0x83, 0x80). Every
// source of the VOP3 arms is defined in-shader, including the high half of v[3:4].
const std::vector<uint32_t> kVop3Prologue = {
    0x7e020281u,   // v_mov_b32 v1, 1
    0x7e040282u,   // v_mov_b32 v2, 2
    0x7e060283u,   // v_mov_b32 v3, 3
    0x7e080280u,   // v_mov_b32 v4, 0
};

// One parsed terminal reject record: "<tag> key=value key=value ...".
struct RejectRecord {
    std::string tag;
    std::map<std::string, std::string> fields;
};

RejectRecord parse_reject(const std::string& text) {
    RejectRecord record;
    std::istringstream in(text);
    in >> record.tag;
    std::string token;
    while (in >> token) {
        const size_t eq = token.find('=');
        if (eq == std::string::npos) continue;
        record.fields.emplace(token.substr(0, eq), token.substr(eq + 1));
    }
    return record;
}

std::string hex(uint32_t value, bool prefix) {
    char buf[16];
    std::snprintf(buf, sizeof buf, prefix ? "0x%x" : "%08x", value);
    return buf;
}

std::vector<uint32_t> program(const std::vector<uint32_t>& prologue,
                              const std::vector<uint32_t>& inst) {
    std::vector<uint32_t> code = prologue;
    code.insert(code.end(), inst.begin(), inst.end());
    code.push_back(kEndpgm);
    return code;
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t addr,
                              const ShaderResourceTable* rt = nullptr,
                              const ComputeShaderConfig& config = {}) {
    return recompile_compute(code.data(), code.size(), rt, config,
                             RecompileDiagnosticContext{RecompileDiagnosticStage::Compute, addr});
}

// The refusal contract: no module, and the TERMINAL record for this program is the
// `recompile-reject` line for the gap instruction itself -- its pc, its complete words, its
// format and opcode -- not merely some line mentioning one of its words.
void expect_gap_refusal(const std::vector<uint32_t>& code, uint64_t addr, uint32_t gap_pc,
                        const std::vector<uint32_t>& gap_words, Rdna2Format fmt, uint32_t opcode,
                        const ShaderResourceTable* rt = nullptr,
                        const ComputeShaderConfig& config = {}) {
    const std::vector<uint32_t> spv = compile(code, addr, rt, config);
    const std::string reason = last_terminal_reject_reason(addr);
    std::printf("  [0x%llx] spv_words=%zu reason=%s\n", static_cast<unsigned long long>(addr),
                spv.size(), reason.c_str());
    EXPECT_TRUE(spv.empty()) << "the gap instruction must refuse the whole compile";

    RejectRecord r = parse_reject(reason);
    EXPECT_EQ(r.tag, "recompile-reject") << reason;
    EXPECT_EQ(r.fields["mode"], "unresolved-operand") << reason;
    EXPECT_EQ(r.fields["pc"], std::to_string(gap_pc)) << "the reject must name the gap pc";
    std::string words;
    for (size_t i = 0; i < gap_words.size(); ++i)
        words += (i ? "," : "") + hex(gap_words[i], false);
    EXPECT_EQ(r.fields["words"], words) << reason;
    EXPECT_EQ(r.fields["fmt"], std::to_string(static_cast<int>(fmt))) << reason;
    EXPECT_EQ(r.fields["op"], hex(opcode, true)) << reason;
    EXPECT_EQ(r.fields["len"], std::to_string(gap_words.size())) << reason;
    // Shader identity: the first code dword and the span, so the record belongs to THIS program.
    EXPECT_EQ(r.fields["sh"], hex(code[0], false) + "/" + std::to_string(code.size())) << reason;
}

void expect_compiles(const std::vector<uint32_t>& code, uint64_t addr, const char* what,
                     const ShaderResourceTable* rt = nullptr,
                     const ComputeShaderConfig& config = {}) {
    const std::vector<uint32_t> spv = compile(code, addr, rt, config);
    EXPECT_FALSE(spv.empty()) << what << " -- reason=" << last_terminal_reject_reason(addr);
    if (!spv.empty()) { EXPECT_EQ(spv[0], 0x07230203u) << what << ": not a SPIR-V module"; }
}

// Decode check shared by the three VOP3A arms: vdst v5, sources v1, v2, v3, no modifiers.
void expect_vop3a_v5_v1_v2_v3(const uint32_t (&w)[2], uint32_t opcode) {
    const Rdna2Inst dec = rdna2_decode_one(w, 2);
    EXPECT_EQ(dec.fmt, Rdna2Format::VOP3);
    EXPECT_EQ(dec.opcode, opcode);
    EXPECT_EQ(dec.len_dwords, 2u);
    EXPECT_TRUE(is_vgpr(dec.dst, 5));
    EXPECT_EQ(dec.n_src, 3u);
    EXPECT_TRUE(is_vgpr(dec.src[0], 1));
    EXPECT_TRUE(is_vgpr(dec.src[1], 2));
    EXPECT_TRUE(is_vgpr(dec.src[2], 3));
    EXPECT_EQ(dec.sdst.kind, OperandKind::None);
    for (int k = 0; k < 3; ++k) {
        EXPECT_FALSE(dec.src_abs[k]);
        EXPECT_FALSE(dec.src_neg[k]);
    }
    EXPECT_FALSE(dec.clamp);
    EXPECT_FALSE(dec.has_modifier);
}

// The three VOP3A arms use the same operand fields as this lowered sibling:
// v_sad_u32 v5, v1, v2, v3 (VOP3 0x15d).
constexpr uint32_t kSadU32[2] = {0xd55d0005u, 0x040e0501u};

}   // namespace

// CONTROL for the three VOP3A arms: the identical program with the lowered v_sad_u32 in the gap
// slot. Same prologue, same vdst/src fields; only the opcode differs.
TEST(GapOpcodeRefusals, ControlVop3aSiblingCompiles) {
    expect_vop3a_v5_v1_v2_v3(kSadU32, 0x15du);
    expect_compiles(program(kVop3Prologue, {kSadU32[0], kSadU32[1]}), 0xA000ull,
                    "v_sad_u32 v5, v1, v2, v3 with the shared VOP3 prologue");
}

// V_PERM_B32 v5, v1, v2, v3 (VOP3 0x344).
TEST(GapOpcodeRefusals, PermB32) {
    static const uint32_t w[2] = {0xd7440005u, 0x040e0501u};
    expect_vop3a_v5_v1_v2_v3(w, 0x344u);
    expect_gap_refusal(program(kVop3Prologue, {w[0], w[1]}), 0xA001ull, 4, {w[0], w[1]},
                       Rdna2Format::VOP3, 0x344u);
}

// V_DIV_FIXUP_F32 v5, v1, v2, v3 (VOP3 0x15f).
TEST(GapOpcodeRefusals, DivFixupF32) {
    static const uint32_t w[2] = {0xd55f0005u, 0x040e0501u};
    expect_vop3a_v5_v1_v2_v3(w, 0x15fu);
    expect_gap_refusal(program(kVop3Prologue, {w[0], w[1]}), 0xA021ull, 4, {w[0], w[1]},
                       Rdna2Format::VOP3, 0x15fu);
}

// V_SAD_U8 v5, v1, v2, v3 (VOP3 0x15a). Its 32-bit sibling v_sad_u32 is the control above; this
// pins the per-byte form to a refusal rather than a silent v_sad_u32-style mistranslation.
TEST(GapOpcodeRefusals, SadU8) {
    static const uint32_t w[2] = {0xd55a0005u, 0x040e0501u};
    expect_vop3a_v5_v1_v2_v3(w, 0x15au);
    expect_gap_refusal(program(kVop3Prologue, {w[0], w[1]}), 0xA031ull, 4, {w[0], w[1]},
                       Rdna2Format::VOP3, 0x15au);
}

// V_MAD_I64_I32 v[5:6], s12, v1, v2, v[3:4] (VOP3B 0x177). dword0[14:8] is the SDST carry
// destination (s12 in Wave32, s[12:13] in Wave64 -- the raw words are the same), not a source; the
// decoder's VOP3B list includes 0x177, so the operands are asserted in full. Control: the lowered
// unsigned sibling v_mad_u64_u32 (0x176) with the identical operand fields.
TEST(GapOpcodeRefusals, MadI64I32) {
    static const uint32_t w[2] = {0xd5770c05u, 0x040e0501u};
    static const uint32_t control[2] = {0xd5760c05u, 0x040e0501u};
    for (const uint32_t* words : {w, control}) {
        const Rdna2Inst dec = rdna2_decode_one(words, 2);
        EXPECT_EQ(dec.fmt, Rdna2Format::VOP3);
        EXPECT_EQ(dec.opcode, words == w ? 0x177u : 0x176u);
        EXPECT_EQ(dec.len_dwords, 2u);
        EXPECT_TRUE(is_vgpr(dec.dst, 5));
        EXPECT_TRUE(is_sgpr(dec.sdst, 12)) << "VOP3B SDST is s12";
        EXPECT_EQ(dec.n_src, 3u);
        EXPECT_TRUE(is_vgpr(dec.src[0], 1));
        EXPECT_TRUE(is_vgpr(dec.src[1], 2));
        EXPECT_TRUE(is_vgpr(dec.src[2], 3)) << "addend v[3:4] is named by its base v3";
        for (int k = 0; k < 3; ++k) EXPECT_FALSE(dec.src_abs[k]) << "VOP3B clears the abs field";
    }
    expect_compiles(program(kVop3Prologue, {control[0], control[1]}), 0xA010ull,
                    "control: v_mad_u64_u32 v[5:6], s12, v1, v2, v[3:4]");
    expect_gap_refusal(program(kVop3Prologue, {w[0], w[1]}), 0xA011ull, 4, {w[0], w[1]},
                       Rdna2Format::VOP3, 0x177u);
}

// S_MOVRELS_B32 s0, s1 (SOP1 0x2e): reads s[1 + M0]. M0 and s1 are defined explicitly, so the
// relative source is s1 = 7. Control: s_mov_b32 s0, s1 (SOP1 0x03) in the same slot.
TEST(GapOpcodeRefusals, MovrelsB32) {
    static const uint32_t w = 0xbe802e01u;
    static const uint32_t control = 0xbe800301u;
    const std::vector<uint32_t> prologue = {
        0xbefc0380u,   // s_mov_b32 m0, 0
        0xbe810387u,   // s_mov_b32 s1, 7
    };
    for (uint32_t word : {w, control}) {
        const Rdna2Inst dec = rdna2_decode_one(&word, 1);
        EXPECT_EQ(dec.fmt, Rdna2Format::SOP1);
        EXPECT_EQ(dec.opcode, word == w ? 0x2eu : 0x03u);
        EXPECT_EQ(dec.len_dwords, 1u);
        EXPECT_TRUE(is_sgpr(dec.dst, 0));
        EXPECT_TRUE(is_sgpr(dec.src[0], 1));
    }
    expect_compiles(program(prologue, {control}), 0xA040ull, "control: s_mov_b32 s0, s1");
    expect_gap_refusal(program(prologue, {w}), 0xA041ull, 2, {w}, Rdna2Format::SOP1, 0x2eu);
}

// IMAGE_GATHER4 v[0:3], v[0:1], s[12:19], s[20:23] dmask:0x1 dim:SQ_RSRC_IMG_2D (MIMG 0x40). A
// gather selects ONE channel, so dmask must be 1, 2, 4 or 8 (llvm-mc rejects 0xf). A direct T# at
// s12 and S# at s20 are supplied, with defined coordinates, so the MIMG emitter's resource gate is
// passed and the refusal is the opcode gate. Control: image_gather4_lz (0x47) -- the same word with
// only the opcode field changed -- compiles with that table, and refuses without it, which shows
// the table is what the shared setup needs.
TEST(GapOpcodeRefusals, ImageGather4) {
    static const uint32_t w[2] = {0xf1000108u, 0x00a30000u};
    static const uint32_t control[2] = {0xf11c0108u, 0x00a30000u};
    for (const uint32_t* words : {w, control}) {
        const Rdna2Inst dec = rdna2_decode_one(words, 2);
        EXPECT_EQ(dec.fmt, Rdna2Format::MIMG);
        EXPECT_EQ(dec.opcode, words == w ? 0x40u : 0x47u);
        EXPECT_EQ(dec.len_dwords, 2u);
        EXPECT_TRUE(is_vgpr(dec.dst, 0));
        EXPECT_TRUE(is_vgpr(dec.src[0], 0));
        EXPECT_TRUE(is_sgpr(dec.src[1], 12));
        EXPECT_TRUE(is_sgpr(dec.src[2], 20));
        EXPECT_EQ(dec.mimg_dmask, 0x1u);
        EXPECT_EQ(dec.mimg_dim, 1u) << "SQ_RSRC_IMG_2D";
    }

    ShaderResourceTable rt;
    {
        ShaderResource texture{};
        texture.cls = ResourceClass::Texture;
        texture.binding = 4;
        texture.img_dim = 1;
        texture.width = texture.height = 4;
        texture.sgpr_base = 12;
        texture.sampler_sgpr_base = 20;
        rt.resources.push_back(texture);
    }
    ComputeShaderConfig config;
    config.user_sgprs.resize(24);   // s12..s19 T#, s20..s23 S# are entry-time user data
    const std::vector<uint32_t> prologue = {
        0x7e0002f0u,   // v_mov_b32 v0, 0.5
        0x7e0202f0u,   // v_mov_b32 v1, 0.5
    };

    expect_compiles(program(prologue, {control[0], control[1]}), 0xA050ull,
                    "control: image_gather4_lz with the texture table", &rt, config);
    EXPECT_TRUE(
        compile(program(prologue, {control[0], control[1]}), 0xA052ull, nullptr, config).empty())
        << "control without a resource table must refuse, or the table is not load-bearing";
    expect_gap_refusal(program(prologue, {w[0], w[1]}), 0xA051ull, 2, {w[0], w[1]},
                       Rdna2Format::MIMG, 0x40u, &rt, config);
}
