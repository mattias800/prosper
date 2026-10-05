// test_gap_opcode_refusals — fail-visible refusal guards of lowerings that stay fail-visible
// for inputs they cannot represent (s_movrels_b32). Every opcode it once pinned as an unlowered
// gap is now lowered. The image_gather4 arm below checks where 0x40 is admitted.
//
// Per the recompiler charter an unsupported op is a FATAL gap, and the loud refusal is only the
// backstop. These arms pin that backstop: the guest word decodes to the right instruction, the
// compile returns no module, and the terminal reject record names THIS instruction. They are pins,
// not goals. WHEN A LOWERING LANDS, ITS ARM GOES RED (the module is no longer empty); replace that
// arm with an execution test of the new lowering rather than deleting the assertion.
//
// Each arm is paired with a control built from the SAME program with ONLY the gap instruction
// swapped for a lowered sibling with identical operand fields (v_sad_u32 for the VOP3A arms,
// v_mad_u64_u32 for the VOP3B arm, image_gather4_lz for image_gather4; each s_movrels_b32 guard
// arm instead changes the program in one place so the lowering admits it). The control compiling is what makes the refusal about the opcode rather than the
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

// Decode check shared by the VOP3A arms: vdst v5, sources v1, v2, v3, no modifiers.
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

// CONTROL for the VOP3A arms: the identical program with the lowered v_sad_u32 in the gap
// slot. Same prologue, same vdst/src fields; only the opcode differs.
TEST(GapOpcodeRefusals, ControlVop3aSiblingCompiles) {
    expect_vop3a_v5_v1_v2_v3(kSadU32, 0x15du);
    expect_compiles(program(kVop3Prologue, {kSadU32[0], kSadU32[1]}), 0xA000ull,
                    "v_sad_u32 v5, v1, v2, v3 with the shared VOP3 prologue");
}

// S_MOVRELS_B32 s0, s1 (SOP1 0x2e) reads s[1 + M0] and is lowered (rdna2_movrels.cpp); these arms
// pin its fail-visible guards. Each refusal is paired with a control that differs in one place and
// compiles, so the refusal is about that guard rather than the program shape. Execution coverage
// of the admitted paths is tests/gpu/execute/test_compute_s_movrels.cpp.
//
// Untracked M0: with M0 never written the index is unknown, so it must refuse rather than read
// s1. Control: s_mov_b32 s0, s1 (SOP1 0x03) in the same slot compiles without M0.
TEST(GapOpcodeRefusals, MovrelsB32) {
    static const uint32_t w = 0xbe802e01u;
    static const uint32_t control = 0xbe800301u;
    const std::vector<uint32_t> prologue = {
        0xbe810387u,   // s_mov_b32 s1, 7 (M0 untracked!)
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
    expect_gap_refusal(program(prologue, {w}), 0xA041ull, 1, {w}, Rdna2Format::SOP1, 0x2eu);
}

// Non-SGPR base: s_movrels_b32 s0, vcc_lo (0xbe802e6a) names a special register, not an indexable
// SGPR. Control: the same program with base s1 (0xbe802e01) and the same constant M0 compiles.
TEST(GapOpcodeRefusals, MovrelsB32NonSgprBase) {
    static const uint32_t w = 0xbe802e6au;
    static const uint32_t control = 0xbe802e01u;
    const std::vector<uint32_t> prologue = {
        0xbefc0382u,   // s_mov_b32 m0, 2
    };
    const Rdna2Inst dec = rdna2_decode_one(&w, 1);
    EXPECT_EQ(dec.fmt, Rdna2Format::SOP1);
    EXPECT_EQ(dec.opcode, 0x2eu);
    EXPECT_NE(dec.src[0].kind, OperandKind::SGPR) << "vcc_lo must not decode as an ordinary SGPR";
    expect_compiles(program(prologue, {control}), 0xA042ull, "control: s_movrels_b32 s0, s1");
    expect_gap_refusal(program(prologue, {w}), 0xA043ull, 1, {w}, Rdna2Format::SOP1, 0x2eu);
}

// Constant M0 past s105: m0 = 0xc8 makes s1 + 200 = s201, outside the SGPR file the fold may
// read. Control: the identical s_movrels_b32 with m0 = 2 (s3) compiles.
TEST(GapOpcodeRefusals, MovrelsB32ConstantIndexPastS105) {
    static const uint32_t w = 0xbe802e01u;                            // s_movrels_b32 s0, s1
    const std::vector<uint32_t> far = {0xbefc03ffu, 0x000000c8u};   // s_mov_b32 m0, 0xc8
    const std::vector<uint32_t> near = {0xbefc0382u};                // s_mov_b32 m0, 2
    expect_compiles(program(near, {w}), 0xA044ull, "control: m0 = 2 folds to s3");
    expect_gap_refusal(program(far, {w}), 0xA045ull, 2, {w}, Rdna2Format::SOP1, 0x2eu);
    // m0 = 0x7fffffff: base + M0 must be bounded before it is added, not after (signed overflow).
    const std::vector<uint32_t> huge = {0xbefc03ffu, 0x7fffffffu};   // s_mov_b32 m0, 0x7fffffff
    expect_gap_refusal(program(huge, {w}), 0xA048ull, 2, {w}, Rdna2Format::SOP1, 0x2eu);
    // The bound is inclusive: m0 = 104 from s1 reads s105, the last SGPR the fold may read.
    const std::vector<uint32_t> edge = {0xbefc03ffu, 0x00000068u};   // s_mov_b32 m0, 104
    expect_compiles(program(edge, {w}), 0xA049ull, "control: m0 = 104 folds to s105");
}

// Dynamic M0 reaching an unrepresentable register: s5 saves the ENTRY value of M0 (an opaque
// token, #3133) before M0 is given a runtime value, so the select over s4..s105 would have to
// read the token. Control: the same program reading from base s6, past the token, compiles.
// v_readfirstlane routes this program through the CFG recompiler, whose terminal record is tagged
// `cfg-recompile-reject` and carries no `sh` field, so the record is checked field by field here
// instead of through expect_gap_refusal's straight-line contract.
TEST(GapOpcodeRefusals, MovrelsB32DynamicReachesSavedEntryM0) {
    static const uint32_t w = 0xbe802e04u;         // s_movrels_b32 s0, s4
    static const uint32_t control = 0xbe802e06u;   // s_movrels_b32 s0, s6
    const std::vector<uint32_t> prologue = {
        0xbe85037cu,   // s_mov_b32 s5, m0 (entry M0 saved as a token)
        0x7e000500u,   // v_readfirstlane_b32 s0, v0
        0xbefc0300u,   // s_mov_b32 m0, s0 (M0 now a runtime value)
    };
    expect_compiles(program(prologue, {control}), 0xA046ull, "control: base s6 skips the token");
    EXPECT_TRUE(compile(program(prologue, {w}), 0xA047ull).empty())
        << "a candidate holding the entry-M0 token must refuse the whole compile";
    const std::string reason = last_terminal_reject_reason(0xA047ull);
    RejectRecord r = parse_reject(reason);
    EXPECT_EQ(r.tag, "cfg-recompile-reject") << reason;
    EXPECT_EQ(r.fields["pc"], "3") << "the reject must name the s_movrels_b32 pc";
    EXPECT_EQ(r.fields["words"], hex(w, false)) << reason;
    EXPECT_EQ(r.fields["op"], "0x2e") << reason;
    EXPECT_EQ(r.fields["fmt"], std::to_string(static_cast<int>(Rdna2Format::SOP1))) << reason;
}

// IMAGE_GATHER4 v[0:3], v[0:1], s[12:19], s[20:23] dmask:0x1 dim:SQ_RSRC_IMG_2D (MIMG 0x40). A
// gather selects ONE channel, so dmask must be 1, 2, 4 or 8 (llvm-mc rejects 0xf). A direct T# at
// s12 and S# at s20 are supplied, with defined coordinates. image_gather4 is no longer a gap: it
// is admitted where reading LOD 0 is exact (a single-level resource, or a non-fragment stage with
// no derivatives), and then must compile to the SAME module as image_gather4_lz (0x47, the same
// word with only the opcode field changed), so the _lz execution coverage applies to it. In a
// fragment shader a multi-level resource must refuse; the single-level twin of that program
// compiling is what shows the refusal comes from the mip gate.
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

    const auto spv_control =
        compile(program(prologue, {control[0], control[1]}), 0xA050ull, &rt, config);
    EXPECT_FALSE(spv_control.empty()) << "control: image_gather4_lz with the texture table";
    const auto spv_gather = compile(program(prologue, {w[0], w[1]}), 0xA051ull, &rt, config);
    EXPECT_FALSE(spv_gather.empty()) << "image_gather4 with the texture table";
    EXPECT_EQ(spv_gather, spv_control) << "image_gather4 on single-level resource must produce "
                                          "word-for-word identical module to image_gather4_lz";

    EXPECT_TRUE(
        compile(program(prologue, {control[0], control[1]}), 0xA052ull, nullptr, config).empty())
        << "control without a resource table must refuse, or the table is not load-bearing";
    EXPECT_TRUE(compile(program(prologue, {w[0], w[1]}), 0xA053ull, nullptr, config).empty())
        << "image_gather4 without a resource table must refuse, or the table is not load-bearing";

    // Gate checks in fragment stage: multi-level texture (declared_mip_levels == 2) must refuse;
    // single-level texture (declared_mip_levels == 1) must compile.
    // A fragment program must export, or recompile_fragment refuses it for that reason alone.
    const auto fragment_prog =
        program(prologue, {w[0], w[1], 0xf800180fu, 0x03020100u});   // exp mrt0 v0-v3 done vm
    ShaderResourceTable frag_rt_single = rt;
    frag_rt_single.resources[0].declared_mip_levels = 1u;
    EXPECT_FALSE(
        recompile_fragment(fragment_prog.data(), fragment_prog.size(), &frag_rt_single).empty())
        << "image_gather4 in fragment stage must compile for single-level resource";

    ShaderResourceTable frag_rt_multi = rt;
    frag_rt_multi.resources[0].declared_mip_levels = 2u;
    EXPECT_TRUE(
        recompile_fragment(fragment_prog.data(), fragment_prog.size(), &frag_rt_multi).empty())
        << "image_gather4 in fragment stage must refuse for multi-level resource";
}

// Unlowered float MIMG atomics must refuse fail-visibly. image_atomic_fmin
// (0x1e) and image_atomic_fmax (0x1f) need float-typed atomic lowering, which
// does not exist (only the integer/R32_UINT atomic path is lowered); forcing
// them through it would silently reinterpret float bits as integers. Words are
// llvm-mc gfx1030 round-tripped (an earlier revision of this arm mislabeled
// them by one opcode). The add words are the byte-exact Astro Bot packet from
// test_game_compute.cpp; fmin/fmax differ from them only in the opcode field.
// The refusals use StorageImage entries -- the control's own Uint32 table, and
// the realistic R32_FLOAT storage target -- because a real float lowering must
// make these storage ops (#4211); today the op is classified sampled-only, so
// a storage entry does not resolve and the arm refuses. A Texture entry would
// keep refusing at the sampled-op allowlist after that lowering lands, so it
// could never go red. image_atomic_fcmpswap (0x1d) is pinned separately
// (ImageAtomicFCmpswapRefuse). WHEN a float lowering lands, ITS CASE GOES RED;
// replace it with an execution test of the new lowering.
TEST(GapOpcodeRefusals, ImageAtomicFloatRefuse) {
    // image_atomic_{fmin,fmax,add} v9, v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D glc
    static const uint32_t fmin[2] = {0xf0782108u, 0x00000900u};
    static const uint32_t fmax[2] = {0xf07c2108u, 0x00000900u};
    static const uint32_t add[2] = {0xf0442108u, 0x00000900u};
    for (const uint32_t* words : {fmin, fmax, add}) {
        const Rdna2Inst dec = rdna2_decode_one(words, 2);
        EXPECT_EQ(dec.fmt, Rdna2Format::MIMG);
        EXPECT_EQ(dec.len_dwords, 2u);
    }
    EXPECT_EQ(rdna2_decode_one(fmin, 2).opcode, 0x1eu);
    EXPECT_EQ(rdna2_decode_one(fmax, 2).opcode, 0x1fu);
    EXPECT_EQ(rdna2_decode_one(add, 2).opcode, 0x11u);

    std::vector<uint8_t> backing(8u * 8u * 4u, 0);
    auto table = [&](DataFormat format) {
        ShaderResourceTable rt;
        ShaderResource image{};
        image.cls = ResourceClass::StorageImage;
        image.format = format;
        image.num_components = 1;
        image.binding = 4;
        image.img_dim = 1;
        image.width = image.height = 8;
        image.depth = 1;
        image.sample_count = 1;
        image.sgpr_base = 0;
        image.gpu_addr = reinterpret_cast<uint64_t>(backing.data());
        image.size = static_cast<uint32_t>(backing.size());
        rt.resources.push_back(image);
        return rt;
    };
    ComputeShaderConfig config;
    config.user_sgprs.resize(8);   // s0..s7 T# are entry-time user data
    const std::vector<uint32_t> prologue = {
        0x7e000280u,   // v_mov_b32 v0, 0 (coord)
        0x7e020280u,   // v_mov_b32 v1, 0 (coord)
        0x7e120281u,   // v_mov_b32 v9, 1 (data/dst)
    };
    const ShaderResourceTable uint_rt = table(DataFormat::Uint32);
    const ShaderResourceTable float_rt = table(DataFormat::Float32);
    expect_compiles(program(prologue, {add[0], add[1]}), 0xA010ull,
                    "control: image_atomic_add over the Uint32 storage entry", &uint_rt,
                    config);
    // Same table as the control: only the opcode differs.
    expect_gap_refusal(program(prologue, {fmin[0], fmin[1]}), 0xA011ull, 3,
                       {fmin[0], fmin[1]}, Rdna2Format::MIMG, 0x1eu, &uint_rt, config);
    expect_gap_refusal(program(prologue, {fmax[0], fmax[1]}), 0xA012ull, 3,
                       {fmax[0], fmax[1]}, Rdna2Format::MIMG, 0x1fu, &uint_rt, config);
    // The realistic R32_FLOAT storage target; a float lowering must turn these red too.
    expect_gap_refusal(program(prologue, {fmin[0], fmin[1]}), 0xA013ull, 3,
                       {fmin[0], fmin[1]}, Rdna2Format::MIMG, 0x1eu, &float_rt, config);
    expect_gap_refusal(program(prologue, {fmax[0], fmax[1]}), 0xA014ull, 3,
                       {fmax[0], fmax[1]}, Rdna2Format::MIMG, 0x1fu, &float_rt, config);
}
