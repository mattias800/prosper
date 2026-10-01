// #4048: project-owned guest packets, not captured game code. No Vulkan device is created.
// AMD RDNA2 ISA 70648, 6.4/12.9/12.9.1: ordered F32 compares, input-denorm modes,
// and CLAMP-clear non-signaling comparisons. ABS precedes NEG. A positive NORMAL bound
// makes ABS(U)<=K invariant under input flushing; EQ(U,+/-0) is not invariant alone.
// Reference: https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/spirv_fragment_vote_lowering.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace prosper::gpu;
namespace {
int failures = 0;
size_t assertions = 0, modules = 0, values = 0;
void check(bool condition, const std::string& name) {
    ++assertions;
    if (!condition) { ++failures; std::printf("[FAIL] %s\n", name.c_str()); }
}
constexpr uint32_t type_bool = 20, type_int = 21, type_float = 22, type_vector = 23;
constexpr uint32_t type_runtime_array = 29, type_struct = 30, type_pointer = 32;
constexpr uint32_t constant_true = 41, constant_false = 42, constant = 43;
constexpr uint32_t variable = 59, load = 61, store = 62, access_chain = 65;
constexpr uint32_t decorate = 71, member_decorate = 72, construct = 80, copy = 83;
constexpr uint32_t bitcast = 124, iadd = 128, logical_or = 166, logical_and = 167;
constexpr uint32_t select = 169, unsigned_le = 178, bit_and = 199, any = 335;

struct Instruction { uint32_t op; std::vector<uint32_t> a; };
struct Module {
    std::vector<Instruction> ins;
    bool valid = false;
    explicit Module(const std::vector<uint32_t>& words) {
        if (words.size() < 5 || words[0] != 0x07230203u) return;
        for (size_t at = 5; at < words.size();) {
            const size_t n = words[at] >> 16;
            if (!n || n > words.size() - at) return;
            ins.push_back({words[at] & 0xffffu, {words.begin()+at+1, words.begin()+at+n}});
            at += n;
        }
        valid = true;
    }
    const Instruction* type(uint32_t id) const {
        for (const auto& i : ins)
            if (i.op >= type_bool && i.op <= type_pointer && !i.a.empty() && i.a[0] == id)
                return &i;
        return nullptr;
    }
    bool uint32_type(uint32_t id) const {
        const auto* t = type(id);
        return t && t->op == type_int && t->a.size() == 3 && t->a[1] == 32 && t->a[2] == 0;
    }
    bool bool_type(uint32_t id) const {
        const auto* t = type(id); return t && t->op == type_bool && t->a.size() == 1;
    }
    bool float32_type(uint32_t id) const {
        const auto* t = type(id);
        return t && t->op == type_float && t->a.size() == 2 && t->a[1] == 32;
    }
    const Instruction* def(uint32_t id) const {
        // Explicit result-bearing opcodes only. Unknown dependencies never acquire a guessed value.
        for (const auto& i : ins) {
            switch (i.op) {
                case constant_true: case constant_false: case constant: case variable:
                case load: case access_chain: case construct: case copy: case bitcast:
                case iadd: case logical_or: case logical_and: case select: case unsigned_le:
                case bit_and: case any: case 12: case 180: case 188: case 190:
                    if (i.a.size() >= 2 && i.a[1] == id) return &i;
                    break;
                default: break;
            }
        }
        return nullptr;
    }
    bool decoration(uint32_t id, uint32_t kind, uint32_t value) const {
        for (const auto& i : ins)
            if (i.op == decorate && i.a == std::vector<uint32_t>{id, kind, value}) return true;
        return false;
    }
    bool literal(uint32_t id, uint32_t value) const {
        const auto* d = def(id);
        return d && d->op == constant && d->a.size() == 3 &&
            uint32_type(d->a[0]) && d->a[2] == value;
    }
    size_t count(uint32_t op) const {
        size_t n = 0; for (const auto& i : ins) n += i.op == op; return n;
    }
    bool controls_branch(uint32_t vote) const {
        for (const auto& i : ins) {
            if (i.op != 250 || i.a.size() != 3) continue;
            if (i.a[0] == vote) return true;
            const auto* c = def(i.a[0]);
            if (!c || c->op != select || c->a.size() != 5 || !bool_type(c->a[0]) ||
                c->a[2] != vote) continue;
            const auto* t = def(c->a[3]);
            const auto* f = def(c->a[4]);
            if (t && f && t->a.size() == 2 && f->a.size() == 2 &&
                t->a[0] == c->a[0] && f->a[0] == c->a[0] &&
                ((t->op == constant_true && f->op == constant_false) ||
                 (t->op == constant_false && f->op == constant_true))) return true;
        }
        return false;
    }
    // Follow the actual MRT0 store, vec4 component, F32 bitcast and tagged Uint32 Select.
    uint32_t exported_predicate() const {
        uint32_t found = 0;
        for (const auto& s : ins) {
            if (s.op != store || s.a.size() != 2 || !decoration(s.a[0], 30, 0)) continue;
            const auto* out = def(s.a[0]);
            const auto* ptr = out ? type(out->a[0]) : nullptr;
            const auto* vec = def(s.a[1]);
            const auto* vt = vec ? type(vec->a[0]) : nullptr;
            if (!out || out->op != variable || out->a.size() != 3 || out->a[2] != 3 ||
                !ptr || ptr->op != type_pointer || ptr->a.size() != 3 || ptr->a[1] != 3 ||
                !vec || vec->op != construct || vec->a.size() != 6 || ptr->a[2] != vec->a[0] ||
                !vt || vt->op != type_vector || vt->a.size() != 3 || vt->a[2] != 4 ||
                !float32_type(vt->a[1])) return 0;
            const auto* fp = def(vec->a[2]);
            if (!fp || fp->op != bitcast || fp->a.size() != 3 || fp->a[0] != vt->a[1]) return 0;
            const auto* tagged = def(fp->a[2]);
            if (!tagged || tagged->op != select || tagged->a.size() != 5 ||
                !uint32_type(tagged->a[0]) || !literal(tagged->a[3], 0x3f800000u) ||
                !literal(tagged->a[4], 0)) return 0;
            const auto* p = def(tagged->a[2]);
            if (!p || !bool_type(p->a[0]) || found) return 0;
            found = tagged->a[2];
        }
        return found;
    }
    // Exact guest runtime-word storage layout, not an Input/UBO or arbitrary pointer override.
    bool word_load(const Instruction& d, uint32_t& index) const {
        if (d.op != load || d.a.size() != 3 || !uint32_type(d.a[0])) return false;
        const auto* p = def(d.a[2]);
        if (!p || p->op != access_chain || p->a.size() != 5 || !literal(p->a[3], 0)) return false;
        const auto* ix = def(p->a[4]);
        if (!ix || ix->op != constant || ix->a.size() != 3 || !uint32_type(ix->a[0]) ||
            ix->a[2] > 1) return false;
        const auto* root = def(p->a[2]);
        const auto* pt = type(p->a[0]);
        const auto* rt = root ? type(root->a[0]) : nullptr;
        const auto* st = rt && rt->a.size() == 3 ? type(rt->a[2]) : nullptr;
        const auto* arr = st && st->a.size() == 2 ? type(st->a[1]) : nullptr;
        if (!root || root->op != variable || root->a.size() != 3 || root->a[2] != 12 ||
            !decoration(root->a[1], 34, 1) || !decoration(root->a[1], 33, 32) ||
            !pt || pt->op != type_pointer || pt->a.size() != 3 || pt->a[1] != 12 ||
            pt->a[2] != d.a[0] || !rt || rt->op != type_pointer || rt->a.size() != 3 || rt->a[1] != 12 ||
            !st || st->op != type_struct || !arr || arr->op != type_runtime_array ||
            arr->a.size() != 2 || arr->a[1] != d.a[0] || !decoration(arr->a[0], 6, 4)) return false;
        bool offset = false;
        for (const auto& i : ins)
            offset |= i.op == member_decorate && i.a == std::vector<uint32_t>{st->a[0],0,35,0};
        if (!offset) return false;
        index = ix->a[2]; return true;
    }
};

// A deliberately small typed oracle for the live integer predicate. No default numeric value,
// Undef, arbitrary Input, float->uint cast, Phi, subgroup result or unknown instruction is allowed.
struct Oracle {
    const Module& m;
    std::array<uint32_t,2> words;
    bool good = true;
    size_t leaves = 0;
    uint32_t uint_value(uint32_t id, unsigned depth = 0) {
        const auto* d = m.def(id);
        if (depth > 48 || !d || !m.uint32_type(d->a[0])) { good = false; return 0; }
        const auto& a = d->a;
        if (d->op == constant && a.size() == 3) return a[2];
        if (d->op == load) {
            uint32_t index = 0;
            if (m.word_load(*d, index)) { ++leaves; return words[index]; }
        }
        if (d->op == copy && a.size() == 3) return uint_value(a[2], depth+1);
        if ((d->op == bit_and || d->op == iadd) && a.size() == 4) {
            const auto l = uint_value(a[2], depth+1), r = uint_value(a[3], depth+1);
            return d->op == bit_and ? l & r : l + r;
        }
        good = false; return 0;
    }
    bool bool_value(uint32_t id, unsigned depth = 0) {
        const auto* d = m.def(id);
        if (depth > 48 || !d || !m.bool_type(d->a[0])) { good = false; return false; }
        const auto& a = d->a;
        if ((d->op == constant_true || d->op == constant_false) && a.size() == 2)
            return d->op == constant_true;
        if (d->op == copy && a.size() == 3) return bool_value(a[2], depth+1);
        if (d->op == unsigned_le && a.size() == 4) {
            const auto l = uint_value(a[2],depth+1), r = uint_value(a[3],depth+1); return l <= r;
        }
        if ((d->op == logical_or || d->op == logical_and) && a.size() == 4) {
            // Strict: evaluate both operands, even when the first determines the ordinary truth value.
            const bool l = bool_value(a[2],depth+1), r = bool_value(a[3],depth+1);
            return d->op == logical_or ? l || r : l && r;
        }
        good = false; return false;
    }
};

// Independent guest reference: compare exponent/significand, never use host floating arithmetic
// or the production sign-mask/unsigned-word expression. The bound is already positive NORMAL.
bool guest_range(uint32_t word, uint32_t bound, bool flush_input) {
    uint32_t exponent = (word >> 23) & 255u, fraction = word & 0x007fffffu;
    if (exponent == 255) return false;
    if (!exponent && flush_input) fraction = 0;
    const uint32_t k_exponent = (bound >> 23) & 255u, k_fraction = bound & 0x007fffffu;
    return exponent < k_exponent || (exponent == k_exponent && fraction <= k_fraction);
}
bool guest_zero(uint32_t word, bool flush_input) {
    return ((word >> 23) & 255u) == 0 && (flush_input || (word & 0x007fffffu) == 0);
}

enum class Encoding { Sdwa, E64 };
struct Options {
    Encoding encoding = Encoding::Sdwa;
    bool mirror = false, reverse_or = false, zero_rhs = false, negative_zero = false;
    bool narrow_both = false, narrow_eq = false, narrow_range = false, cmpx = false;
    bool neg_word = false, abs_bound = false, clamp = false, eq_abs = false;
    bool mismatch = false, overwrite = false, unknown_bound = false, varying = false;
    bool bad_sdwa = false, unordered = false, use_and = false;
    uint32_t bound = 0x3a83126fu;
};
void mov_scalar(std::vector<uint32_t>& p, uint32_t reg, uint32_t bits) {
    p.insert(p.end(), {0xbe8003ffu | (reg << 16), bits});
}
void mov_vector(std::vector<uint32_t>& p, uint32_t reg, uint32_t bits) {
    p.insert(p.end(), {0x7e0002ffu | (reg << 17), bits});
}
void buffer_load(std::vector<uint32_t>& p, uint32_t reg, uint32_t index) {
    p.insert(p.end(), {0xf4000000u | (0x08u << 18) | (reg << 6), 0xfa000000u | (index*4)});
}
void compare(std::vector<uint32_t>& p, Encoding encoding, uint32_t op, uint32_t dst,
             uint32_t lhs, uint32_t rhs, bool abs_l = false, bool abs_r = false,
             bool neg_l = false, bool neg_r = false, bool clamp = false, bool bad_sdwa = false) {
    std::array<uint32_t,2> packet;
    if (encoding == Encoding::E64) {
        packet = {0xd4000000u | (op << 16) | dst | (uint32_t(abs_l)<<8) |
                      (uint32_t(abs_r)<<9) | (uint32_t(clamp)<<15),
                  lhs | (rhs<<9) | (uint32_t(neg_l)<<29) | (uint32_t(neg_r)<<30)};
    } else {
        packet = {0x7c000000u | (op<<17) | ((rhs&255u)<<9) | 0xf9u,
                  (lhs&255u) | 0x8000u | (dst<<8) | ((bad_sdwa?4u:6u)<<16) | (6u<<24) |
                  (uint32_t(lhs<256u)<<23) | (uint32_t(rhs<256u)<<31) |
                  (uint32_t(abs_l)<<21) | (uint32_t(abs_r)<<29) |
                  (uint32_t(neg_l)<<20) | (uint32_t(neg_r)<<28)};
    }
    const auto d = rdna2_decode_one(packet.data(), packet.size());
    const auto operand_matches = [](const Operand& decoded, uint32_t encoded) {
        if (encoded >= 256u) return decoded.kind == OperandKind::VGPR &&
            decoded.value == static_cast<int>(encoded-256u);
        if (encoded == 128u) return decoded.kind == OperandKind::InlineInt && decoded.value == 0;
        return decoded.kind == OperandKind::SGPR && decoded.value == static_cast<int>(encoded);
    };
    check(d.fmt == Rdna2Format::VOPC && d.opcode == op && d.len_dwords == 2 &&
          operand_matches(d.src[0],lhs) && operand_matches(d.src[1],rhs) &&
          d.dst.value == static_cast<int>(dst) && d.src_abs[0] == abs_l &&
          d.src_abs[1] == abs_r && d.src_neg[0] == neg_l && d.src_neg[1] == neg_r &&
          d.has_sdwa == (encoding == Encoding::Sdwa) &&
          (encoding != Encoding::E64 || d.clamp == clamp) &&
          (bad_sdwa || !d.has_modifier), "generated compare packet decode contract");
    p.insert(p.end(), packet.begin(), packet.end());
}
std::vector<uint32_t> program(const Options& o, bool branch) {
    std::vector<uint32_t> p;
    buffer_load(p,20,0);
    mov_scalar(p,21,o.bound);
    if (o.mismatch) buffer_load(p,22,1);
    if (o.unknown_bound) buffer_load(p,21,1);
    if (o.negative_zero) mov_scalar(p,23,0x80000000u);
    mov_vector(p,9,0x40000000u); // A live, non-neutral branch export: 2.0 -> 3.0.
    const uint32_t raw = o.varying ? 256u : 20u;
    const uint32_t z = o.negative_zero ? 23u : 128u;
    const auto narrow = [&] {
        compare(p,Encoding::E64,0xd2,126,128,raw); // CMPX_EQ_U32 0,U
    };
    if (o.narrow_both || o.narrow_eq) narrow();
    compare(p,o.encoding,2,8,o.zero_rhs?raw:z,o.zero_rhs?z:raw,
            o.eq_abs && o.zero_rhs,o.eq_abs && !o.zero_rhs);
    if (o.narrow_eq) p.push_back(0xbefe04c1u); // restore full EXEC before range
    if (o.narrow_range) narrow();
    if (o.overwrite) buffer_load(p,20,1); // same physical register, NEW loaded SSA
    const uint32_t range_raw = o.mismatch ? 22u : raw;
    uint32_t op = o.mirror ? 6u : 3u;
    if (o.unordered) op = o.mirror ? 9u : 12u;
    if (o.cmpx) op += 0x10u;
    compare(p,o.encoding,op,o.cmpx?126u:106u,o.mirror?21u:range_raw,o.mirror?range_raw:21u,
            o.mirror?o.abs_bound:true,o.mirror?true:o.abs_bound,
            !o.mirror && o.neg_word,o.mirror && o.neg_word,o.clamp,o.bad_sdwa);
    if (o.cmpx) {
        p.push_back(0xbe8a047eu); // save actual CMPX EXEC destination in s[10:11]
        p.push_back(0xbefe04c1u);
        p.push_back(0xbeea040au); // actual CMPX result -> VCC for the ordinary scalar OR
    }
    p.push_back(0x80000000u | ((o.use_and?0x0fu:0x11u)<<23) | (106u<<16) |
                ((o.reverse_or?106u:8u)<<8) | (o.reverse_or?8u:106u));
    if (o.narrow_both || o.narrow_range) p.push_back(0xbefe04c1u);
    // Typed tagged consumer of the exact VCC result, then real MRT0 red export.
    p.insert(p.end(), {0xd5010008u, 128u | (242u<<9) | (106u<<18)});
    if (branch) {
        p.push_back(0xbf870002u); // VCCNZ skips exactly the two-dword live v9 write
        mov_vector(p,9,0x40400000u);
    }
    p.insert(p.end(), {0xf800180fu,0x09080908u,0xbf810000u});
    return p;
}
std::vector<uint32_t> translate(const Options& o, bool branch, uint32_t wave) {
    const auto p = program(o,branch);
    ShaderResourceTable rt;
    ShaderResource r;
    r.cls = ResourceClass::ConstantBuffer; r.format = DataFormat::Uint32;
    r.sgpr_base = 0; r.binding = 32; r.size = 8; r.stride = 0;
    rt.resources.push_back(r);
    PixelSystemInputMapping input;
    input.ena = input.addr = 1u<<8;
    ++modules;
    return recompile_fragment(p.data(),p.size(),&rt,&input,UINT32_MAX,nullptr,wave==32);
}
void dump(const char* directory, const std::string& name, const std::vector<uint32_t>& words) {
    if (!directory || words.empty()) return;
    std::ofstream out(std::filesystem::path(directory)/(name+".spv"),std::ios::binary);
    const auto bytes = std::as_bytes(std::span(words));
    out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    out.close(); check(bool(out),name+" dump written");
}
std::vector<uint32_t> samples(uint32_t bound) {
    std::vector<uint32_t> v{0,0x80000000u,1,0x80000001u,0x007fffffu,0x807fffffu,
        0x00800000u,0x80800000u,bound-1,bound,bound+1,(bound-1)|0x80000000u,
        bound|0x80000000u,(bound+1)|0x80000000u,0x7f7fffffu,0xff7fffffu,
        0x7f800000u,0xff800000u,0x7f800001u,0xff800001u,0x7fc00000u,0xffc01234u};
    return v;
}
void positive(const Options& o, uint32_t wave, const std::string& name, const char* directory) {
    const auto source = translate(o,true,wave);
    const Module m(source);
    check(m.valid && !source.empty(),name+" real fragment translator emitted SOURCE");
    if (!m.valid) return;
    dump(directory,name+"_source",source);
    const uint32_t predicate = m.exported_predicate();
    const auto* p = m.def(predicate);
    check(predicate && p && p->op == unsigned_le,name+" actual MRT0.x consumes integer magnitude predicate");
    check(m.count(any)==1,name+" actual VCC branch retains one complete-wave Any in SOURCE");
    bool connected = false;
    uint32_t vote_result = 0, vote_type = 0;
    for (const auto& i : m.ins)
        if (i.op==any && i.a.size()==4 && i.a[3]==predicate && m.literal(i.a[2],3)) {
            connected = m.controls_branch(i.a[1]);
            vote_result = i.a[1]; vote_type = i.a[0];
        }
    check(connected,name+" live BranchConditional and MRT export consume the SAME predicate via subgroup Any");
    for (const auto word : samples(o.bound)) {
        Oracle oracle{m,{word,0x7f800001u}};
        const bool actual = oracle.bool_value(predicate);
        check(oracle.good && oracle.leaves==1,name+" live oracle has one defined typed buffer-word dependency");
        for (const bool flush : {false,true}) {
            const bool expected = guest_range(word,o.bound,flush) || guest_zero(word,flush);
            check(actual==expected,name+" typed live predicate matches independent guest class reference");
            ++values;
        }
    }
    const auto unchanged = source;
    for (const bool immutable : {false,true}) for (const bool deterministic : {false,true}) {
        const auto result = lower_fragment_votes(source,immutable,deterministic);
        // ProvenVotes is deliberately a Wave64 fallback, not a generic Wave32 transformer.
        const bool admitted = wave==64 && immutable && deterministic;
        const auto expected_refusal = wave==32 ? FragmentVoteRefusal::InconsistentContract :
            admitted ? FragmentVoteRefusal::None : FragmentVoteRefusal::UnprovedVote;
        check(source==unchanged,name+" SOURCE immutable across admission variants");
        check(result.refusal==expected_refusal &&
              (admitted ? !result.words.empty() && result.uniform_votes==1 &&
                  result.dead_votes==0 && result.neutral_votes==0 :
                  result.words.empty() && result.uniform_votes==0 && result.dead_votes==0 &&
                  result.neutral_votes==0),name+(wave==32 ? " Wave64-only lowerer refuses Wave32 contract" :
                      " exact independent immutable/robust2 lowerer authority"));
        if (admitted) {
            const Module effective(result.words);
            const auto* exact = effective.def(vote_result);
            check(exact && exact->op==copy && exact->a==std::vector<uint32_t>{vote_type,vote_result,predicate} &&
                  effective.controls_branch(vote_result),
                  name+" actual branch controller becomes typed Copy(P), not TRUE or another predicate");
            check(effective.valid && effective.count(any)==0 &&
                  fragment_spirv_required_subgroup_size(result.words)==0 &&
                  fragment_spirv_required_subgroup_reasons(result.words)==0,
                  name+" effective admission erases only proved complete-wave requirement");
            dump(directory,name+"_effective",result.words);
            Oracle oracle{effective,{o.bound+1,0}};
            check(effective.exported_predicate()==predicate &&
                  !oracle.bool_value(predicate) && oracle.good && oracle.leaves==1,
                  name+" effective live predicate preserves original false value/SSA");
        }
    }
}
void negative(const Options& o, const std::string& name, const char* directory, bool reject=false) {
    const auto source = translate(o,false,64);
    if (reject) { check(source.empty(),name+" unsupported packet still rejects beside DWORD positive control"); return; }
    const Module m(source);
    check(m.valid && !source.empty(),name+" negative control still translates");
    if (!m.valid) return;
    dump(directory,name+"_source",source);
    const auto* p = m.def(m.exported_predicate());
    check(p && p->op==(o.use_and?logical_and:logical_or),
          name+" actual exported mask is not absorbed / strict AND unchanged");
    Oracle oracle{m,{0x7f800001u,0}};
    oracle.bool_value(m.exported_predicate());
    check(!oracle.good,name+" oracle refuses unproved FP/unsupported live dependencies");
}
void undefined_control(const char* directory) {
    // This is an explicit SPIR-V obligation control, NOT a guest-regeneration claim. Replacing
    // the one live word load by Undef must not borrow the buffer's definedness certificate.
    Options o;
    auto source = translate(o,true,64);
    const Module baseline(source);
    bool replaced = false;
    for (size_t at=5; at<source.size();) {
        const size_t n=source[at]>>16;
        if (!n || n>source.size()-at) break;
        if ((source[at]&0xffffu)==load && n==4 && baseline.uint32_type(source[at+1])) {
            source[at]=(3u<<16)|1u; // OpUndef Uint32, same result SSA ID, no pointer operand
            source.erase(source.begin()+at+3);
            replaced=true; break;
        }
        at+=n;
    }
    check(replaced,"undefined control replaces real typed storage-word producer");
    const Module m(source);
    Oracle oracle{m,{0,0}};
    oracle.bool_value(m.exported_predicate());
    check(m.valid && !oracle.good,"typed oracle cannot mint a value for Undef raw word");
    const auto unchanged=source;
    const auto lowered=lower_fragment_votes(source,true,true);
    check(source==unchanged && lowered.refusal==FragmentVoteRefusal::UnprovedVote &&
          lowered.words.empty() && lowered.uniform_votes==0 && lowered.dead_votes==0 &&
          lowered.neutral_votes==0,"same-SSA Undef cannot borrow immutable/robust2 uniformity");
    dump(directory,"undef_obligation_source",source);
}
} // namespace

int main(int argc, char** argv) {
    if (argc!=1 && !(argc==3 && std::string_view(argv[1])=="--dump")) return 2;
    const char* directory = argc==3?argv[2]:nullptr;
    undefined_control(directory);
    for (const auto encoding : {Encoding::Sdwa,Encoding::E64}) {
        const std::string enc = encoding==Encoding::Sdwa?"sdwa":"e64";
        for (const bool mirror : {false,true}) for (const bool reverse : {false,true})
            for (const bool zero_rhs : {false,true}) for (const uint32_t wave : {32u,64u})
                for (const uint32_t bound : {0x00800000u,0x3a83126fu,0x7f7fffffu}) {
                    Options o; o.encoding=encoding; o.mirror=mirror; o.reverse_or=reverse;
                    o.zero_rhs=zero_rhs; o.negative_zero=zero_rhs; o.bound=bound;
                    positive(o,wave,enc+"_"+(mirror?"ge":"le")+"_or"+std::to_string(reverse)+
                        "_zero"+std::to_string(zero_rhs)+"_wave"+std::to_string(wave)+
                        "_bound"+std::to_string(bound),directory);
                }
        const auto arm = [&](const char* name, auto modify, bool reject=false) {
            Options o; o.encoding=encoding; modify(o); negative(o,enc+"_"+name,directory,reject);
        };
        arm("narrow_both",[](auto& o){o.narrow_both=true;});
        arm("narrow_eq_only",[](auto& o){o.narrow_eq=true;});
        arm("narrow_range_only",[](auto& o){o.narrow_range=true;});
        arm("cmpx_destination",[](auto& o){o.cmpx=true;});
        arm("neg_after_abs",[](auto& o){o.neg_word=true;});
        arm("abs_threshold",[](auto& o){o.abs_bound=true;});
        arm("eq_abs_modifier",[](auto& o){o.eq_abs=true;});
        arm("different_word",[](auto& o){o.mismatch=true;});
        arm("same_register_new_ssa",[](auto& o){o.overwrite=true;});
        arm("unknown_bound",[](auto& o){o.unknown_bound=true;});
        arm("unordered",[](auto& o){o.unordered=true;});
        arm("strict_scalar_and",[](auto& o){o.use_and=true;});
        for (const auto bound : {0u,0x80000000u,0x007fffffu,0x80000001u,0xbf800000u,
                                0x7f800000u,0x7fc00000u})
            arm(("non_normal_bound"+std::to_string(bound)).c_str(),[&](auto& o){o.bound=bound;});
        if (encoding==Encoding::Sdwa) arm("word_select",[](auto& o){o.bad_sdwa=true;},true);
        else arm("signaling_clamp",[](auto& o){o.clamp=true;});
        Options varying; varying.encoding=encoding; varying.varying=true;
        const auto source=translate(varying,true,64);
        const Module m(source);
        const auto lowered=lower_fragment_votes(source,true,true);
        check(m.valid && m.count(any)==1 && m.def(m.exported_predicate()) &&
              m.def(m.exported_predicate())->op==unsigned_le &&
              lowered.refusal==FragmentVoteRefusal::UnprovedVote && lowered.words.empty(),
              enc+" exact local integer semantics do NOT certify varying Input as wave-uniform");
        Oracle oracle{m,{0,0}};
        oracle.bool_value(m.exported_predicate());
        check(!oracle.good,enc+" unsupported Input dependency cannot get an oracle placeholder");
        dump(directory,enc+"_varying_source",source);
    }
    std::printf("== normal_magnitude_predicate: %zu translator modules, %zu values, %zu assertions, %d failures ==\n",
                modules,values,assertions,failures);
    return failures?1:0;
}
