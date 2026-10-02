// #4056: project-owned guest packets, not captured game code; no Vulkan device.
// AMD70648 section6.4: F32 input denorm modes0/2 flush, modes1/3 preserve.
// EQ and unordered NEQ have Boolean results; rounding/output-denorm bits are inert here.
// Expected values use exponent/significand categories, never host FP or the emitter helper.
// Reference: https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/spirv_fragment_vote_lowering.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "../../fixtures/test_scratch.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace prosper::gpu;
namespace {
int failures = 0;
size_t assertions = 0, modules = 0, values = 0;
uint64_t address = 0x40560000u;
void check(bool condition, const std::string& name) {
    ++assertions;
    if (!condition) { ++failures; std::printf("[FAIL] %s\n", name.c_str()); }
}
constexpr uint32_t tb=20, ti=21, tf=22, tv=23, ta=29, ts=30, tp=32;
constexpr uint32_t ct=41, cf=42, con=43, var=59, load=61, store=62, chain=65;
constexpr uint32_t deco=71, member=72, construct=80, extract=81, copy=83, cast=124;
constexpr uint32_t lor=166, land=167, lnot=168, select=169, ieq=170, ine=171;
constexpr uint32_t add=128, mul=132, uge=174, ule=178, band=199, phi=245, branch=250, any=335;
struct Instruction { uint32_t op; std::vector<uint32_t> a; };
// Adapted from prosper's normal_magnitude_predicate typed sink oracle. Unknown definitions,
// arbitrary Inputs, Undef and unsupported FP dependencies never acquire a default value.
struct Module {
    std::vector<Instruction> ins;
    bool valid=false;
    explicit Module(const std::vector<uint32_t>& w) {
        if (w.size()<5 || w[0]!=0x07230203u) return;
        for (size_t pc=5; pc<w.size();) {
            const size_t n=w[pc]>>16;
            if (!n || n>w.size()-pc) return;
            ins.push_back({w[pc]&0xffffu,{w.begin()+pc+1,w.begin()+pc+n}}); pc+=n;
        }
        valid=true;
    }
    const Instruction* type(uint32_t id) const {
        for (const auto& i:ins) if (i.op>=tb && i.op<=tp && !i.a.empty() && i.a[0]==id) return &i;
        return nullptr;
    }
    const Instruction* def(uint32_t id) const {
        for (const auto& i:ins) switch(i.op) {
            case ct: case cf: case con: case var: case load: case chain: case construct:
            case extract: case copy: case cast: case lor: case land: case lnot: case select:
            case ieq: case ine: case uge: case ule: case band: case add: case mul: case phi: case any:
            case 12: case 127: case 180: case 182: case 183: case 188: case 190:
                if (i.a.size()>=2 && i.a[1]==id) return &i; break;
            default: break;
        }
        return nullptr;
    }
    bool u32(uint32_t id) const {
        const auto* t=type(id); return t && t->op==ti && t->a.size()==3 && t->a[1]==32 && t->a[2]==0;
    }
    bool boolean(uint32_t id) const {
        const auto* t=type(id); return t && t->op==tb && t->a.size()==1;
    }
    bool f32(uint32_t id) const {
        const auto* t=type(id); return t && t->op==tf && t->a.size()==2 && t->a[1]==32;
    }
    bool decoration(uint32_t id,uint32_t kind,uint32_t value) const {
        for (const auto& i:ins) if (i.op==deco && i.a==std::vector<uint32_t>{id,kind,value}) return true;
        return false;
    }
    bool flat(uint32_t id) const {
        for (const auto& i:ins) if(i.op==deco && i.a==std::vector<uint32_t>{id,14}) return true;
        return false;
    }
    bool literal(uint32_t id,uint32_t value) const {
        const auto* d=def(id); return d && d->op==con && d->a.size()==3 && u32(d->a[0]) && d->a[2]==value;
    }
    size_t count(uint32_t op) const { size_t n=0; for (const auto& i:ins) n+=i.op==op; return n; }
    const Instruction* output_vector() const {
        const Instruction* found=nullptr;
        for (const auto& s:ins) {
            if (s.op!=store || s.a.size()!=2 || !decoration(s.a[0],30,0)) continue;
            const auto* root=def(s.a[0]); const auto* p=root?type(root->a[0]):nullptr;
            const auto* v=def(s.a[1]); const auto* t=v?type(v->a[0]):nullptr;
            if (!root || root->op!=var || root->a.size()!=3 || root->a[2]!=3 ||
                !p || p->op!=tp || p->a.size()!=3 || p->a[1]!=3 ||
                !v || v->op!=construct || v->a.size()!=6 || p->a[2]!=v->a[0] ||
                !t || t->op!=tv || t->a.size()!=3 || t->a[2]!=4 || !f32(t->a[1]) || found) return nullptr;
            found=v;
        }
        return found;
    }
    uint32_t exported_predicate(unsigned component=0) const {
        const auto* v=output_vector(); if (!v || component>=4) return 0;
        const auto* fp=def(v->a[2+component]);
        if (!fp || fp->op!=cast || fp->a.size()!=3 || !f32(fp->a[0])) return 0;
        const auto* tag=def(fp->a[2]);
        if (!tag || tag->op!=select || tag->a.size()!=5 || !u32(tag->a[0]) ||
            !literal(tag->a[3],0x3f800000u) || !literal(tag->a[4],0)) return 0;
        const auto* p=def(tag->a[2]); return p && boolean(p->a[0])?tag->a[2]:0;
    }
    bool controls_branch(uint32_t vote) const {
        for (const auto& i:ins) {
            if (i.op!=branch || i.a.size()!=3) continue;
            if (i.a[0]==vote) return true;
            const auto* c=def(i.a[0]);
            if (!c || c->op!=select || c->a.size()!=5 || !boolean(c->a[0]) || c->a[2]!=vote) continue;
            const auto* t=def(c->a[3]); const auto* f=def(c->a[4]);
            if (t && f && t->a.size()==2 && f->a.size()==2 && t->a[0]==c->a[0] && f->a[0]==c->a[0] &&
                ((t->op==ct && f->op==cf)||(t->op==cf && f->op==ct))) return true;
        }
        return false;
    }
    bool live_green_phi() const {
        const auto* v=output_vector(); const auto* fp=v?def(v->a[3]):nullptr;
        const auto* p=fp && fp->op==cast && fp->a.size()==3 && f32(fp->a[0])?def(fp->a[2]):nullptr;
        return p && p->op==phi && p->a.size()==6 && u32(p->a[0]) &&
            ((literal(p->a[2],0x40000000u)&&literal(p->a[4],0x40400000u)) ||
             (literal(p->a[4],0x40000000u)&&literal(p->a[2],0x40400000u)));
    }
    bool word_load(const Instruction& d,uint32_t& index) const {
        if (d.op!=load || d.a.size()!=3 || !u32(d.a[0])) return false;
        const auto* p=def(d.a[2]);
        if (!p || p->op!=chain || p->a.size()!=5 || !literal(p->a[3],0)) return false;
        const auto* ix=def(p->a[4]);
        if (!ix || ix->op!=con || ix->a.size()!=3 || !u32(ix->a[0]) || ix->a[2]>1) return false;
        const auto* root=def(p->a[2]); const auto* pt=type(p->a[0]);
        const auto* rt=root?type(root->a[0]):nullptr;
        const auto* st=rt && rt->a.size()==3?type(rt->a[2]):nullptr;
        const auto* arr=st && st->a.size()==2?type(st->a[1]):nullptr;
        if (!root || root->op!=var || root->a.size()!=3 || root->a[2]!=12 ||
            !decoration(root->a[1],34,1) || !decoration(root->a[1],33,32) ||
            !pt || pt->op!=tp || pt->a.size()!=3 || pt->a[1]!=12 || pt->a[2]!=d.a[0] ||
            !rt || rt->op!=tp || rt->a.size()!=3 || rt->a[1]!=12 ||
            !st || st->op!=ts || !arr || arr->op!=ta || arr->a.size()!=2 ||
            arr->a[1]!=d.a[0] || !decoration(arr->a[0],6,4)) return false;
        bool offset=false;
        for (const auto& i:ins) offset|=i.op==member && i.a==std::vector<uint32_t>{st->a[0],0,35,0};
        if (!offset) return false; index=ix->a[2]; return true;
    }
    bool varying_bits(const Instruction& d) const {
        if (d.op!=cast || d.a.size()!=3 || !u32(d.a[0])) return false;
        const auto* x=def(d.a[2]);
        if (!x || x->op!=extract || x->a.size()!=4 || !f32(x->a[0]) || x->a[3]!=0) return false;
        const auto* l=def(x->a[2]); const auto* vt=l?type(l->a[0]):nullptr;
        const auto* r=l && l->a.size()==3?def(l->a[2]):nullptr;
        const auto* pt=r?type(r->a[0]):nullptr;
        return l && l->op==load && vt && vt->op==tv && vt->a.size()==3 && vt->a[2]==4 &&
            vt->a[1]==x->a[0] && r && r->op==var && r->a.size()==3 && r->a[2]==1 &&
            decoration(r->a[1],30,0) && flat(r->a[1]) && pt && pt->op==tp && pt->a.size()==3 &&
            pt->a[1]==1 && pt->a[2]==l->a[0];
    }
};
struct Oracle {
    const Module& m; std::array<uint32_t,2> words; uint32_t input_word=0;
    bool good=true; std::set<uint32_t> leaves;
    uint32_t uint_value(uint32_t id,unsigned depth=0) {
        const auto* d=m.def(id);
        if (depth>64 || !d || !m.u32(d->a[0])) { good=false; return 0; }
        const auto& a=d->a;
        if (d->op==con && a.size()==3) return a[2];
        if (d->op==load) { uint32_t ix=0; if(m.word_load(*d,ix)) { leaves.insert(id); return words[ix]; } }
        if (m.varying_bits(*d)) { leaves.insert(id); return input_word; }
        if (d->op==copy && a.size()==3) return uint_value(a[2],depth+1);
        if ((d->op==band || d->op==add || d->op==mul) && a.size()==4) {
            const uint32_t l=uint_value(a[2],depth+1),r=uint_value(a[3],depth+1);
            if(d->op==band) return l&r;
            if(d->op==add) return l+r;
            if(d->op==mul) return l*r;
        }
        good=false; return 0;
    }
    bool bool_value(uint32_t id,unsigned depth=0) {
        const auto* d=m.def(id);
        if (depth>64 || !d || !m.boolean(d->a[0])) { good=false; return false; }
        const auto& a=d->a;
        if ((d->op==ct || d->op==cf) && a.size()==2) return d->op==ct;
        if (d->op==copy && a.size()==3) return bool_value(a[2],depth+1);
        if (d->op==lnot && a.size()==3) return !bool_value(a[2],depth+1);
        if ((d->op==ieq || d->op==ine || d->op==uge || d->op==ule) && a.size()==4) {
            const uint32_t l=uint_value(a[2],depth+1),r=uint_value(a[3],depth+1);
            return d->op==ieq?l==r:d->op==ine?l!=r:d->op==uge?l>=r:l<=r;
        }
        if ((d->op==lor || d->op==land) && a.size()==4) {
            // Strict operators consume BOTH operands, including a possibly undefined dependency.
            const bool l=bool_value(a[2],depth+1),r=bool_value(a[3],depth+1); return d->op==lor?l||r:l&&r;
        }
        if (d->op==select && a.size()==5) {
            const bool condition=bool_value(a[2],depth+1);
            // SPIR-V Select does not propagate poison from its unselected value.
            return bool_value(a[condition?3:4],depth+1);
        }
        good=false; return false;
    }
};
bool guest_zero(uint32_t word,uint8_t mode) {
    const uint32_t exponent=(word>>23)&255u,significand=word&0x007fffffu;
    const uint32_t denorm_mode=(mode>>4)&3u;
    return exponent==0 && (significand==0 || denorm_mode==0 || denorm_mode==2);
}
bool guest_compare(uint32_t word,uint8_t mode,bool neq) {
    // NaNs, including signalling NaNs, are NOT EQ-zero and ARE unordered NEQ-zero.
    return neq?!guest_zero(word,mode):guest_zero(word,mode);
}
const std::array<uint32_t,29> samples{0,0x80000000u,1,0x80000001u,0x003fffffu,0x803fffffu,
    0x007fffffu,0x807fffffu,0x00800000u,0x80800000u,0x00800001u,0x80800001u,
    0x3f800000u,0xbf800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,
    0x7f800001u,0xffbfffffu,0x7fc00000u,0xffffffffu,
    0x3fffffffu,0xbfffffffu,0x40000000u,0xc0000000u,0x7ffffffeu,0xfffffffeu,0x7fffffffu};
enum class Encoding { E32,Sdwa,E64 };
struct Options {
    Encoding encoding=Encoding::E64;
    bool neq=true,mirror=false,negative_zero=false,narrow=false,cmpx=false,named=false,varying=false;
    bool abs_word=false,abs_zero=false,neg_word=false,neg_zero=false,clamp=false;
    bool partial=false,sext=false,dpp=false,vector_zero=false; uint32_t omod=0;
};
void mov_scalar(std::vector<uint32_t>& p,uint32_t r,uint32_t v) { p.insert(p.end(),{0xbe8003ffu|(r<<16),v}); }
void mov_vector(std::vector<uint32_t>& p,uint32_t r,uint32_t v) { p.insert(p.end(),{0x7e0002ffu|(r<<17),v}); }
void buffer_load(std::vector<uint32_t>& p,uint32_t r,uint32_t ix) {
    p.insert(p.end(),{0xf4000000u|(0x08u<<18)|(r<<6),0xfa000000u|(ix*4)});
}
void compare(std::vector<uint32_t>& p,const Options& o,uint32_t op,uint32_t dst,uint32_t lhs,uint32_t rhs) {
    std::vector<uint32_t> packet;
    const bool al=o.mirror?o.abs_word:o.abs_zero,ar=o.mirror?o.abs_zero:o.abs_word;
    const bool nl=o.mirror?o.neg_word:o.neg_zero,nr=o.mirror?o.neg_zero:o.neg_word;
    if(o.encoding==Encoding::E32) {
        check(rhs>=256,"e32 RHS is genuinely a VGPR");
        packet={0x7c000000u|(op<<17)|((rhs-256u)<<9)|lhs};
        if(o.dpp) { packet[0]=(packet[0]&~511u)|0xfau; packet.push_back(0xff010000u|(lhs&255u)); }
    } else if(o.encoding==Encoding::E64) {
        packet={0xd4000000u|(op<<16)|dst|(uint32_t(al)<<8)|(uint32_t(ar)<<9)|(uint32_t(o.clamp)<<15),
                lhs|(rhs<<9)|(o.omod<<27)|(uint32_t(nl)<<29)|(uint32_t(nr)<<30)};
    } else {
        packet={0x7c000000u|(op<<17)|((rhs&255u)<<9)|0xf9u,
            (lhs&255u)|0x8000u|(dst<<8)|((o.partial?4u:6u)<<16)|(6u<<24)|
            (uint32_t(lhs<256u)<<23)|(uint32_t(rhs<256u)<<31)|
            (uint32_t(al)<<21)|(uint32_t(ar)<<29)|(uint32_t(nl)<<20)|(uint32_t(nr)<<28)|
            (uint32_t(o.sext)<<19)};
    }
    const auto d=rdna2_decode_one(packet.data(),packet.size());
    const auto operand=[](const Operand& d,uint32_t x) {
        if(x>=256) return d.kind==OperandKind::VGPR && d.value==int(x-256);
        if(x==128) return d.kind==OperandKind::InlineInt && d.value==0;
        return d.kind==OperandKind::SGPR && d.value==int(x);
    };
    check(d.fmt==Rdna2Format::VOPC && d.opcode==op && d.len_dwords==packet.size() &&
          (o.dpp || (operand(d.src[0],lhs)&&operand(d.src[1],rhs))) &&
          (o.encoding==Encoding::E32 || d.dst.value==int(dst)),"actual synthetic compare decoder contract");
    p.insert(p.end(),packet.begin(),packet.end());
}
void tag(std::vector<uint32_t>& p,uint32_t vgpr,uint32_t condition) {
    p.insert(p.end(),{0xd5010000u|vgpr,128u|(242u<<9)|(condition<<18)});
}
std::vector<uint32_t> program(const Options& o,bool with_branch=true) {
    std::vector<uint32_t> p;
    buffer_load(p,20,0); buffer_load(p,21,1);
    mov_scalar(p,23,o.negative_zero?0x80000000u:0);
    mov_vector(p,9,0x40000000u);
    if(o.varying) {
        p.push_back(0xc8020002u); // v_interp_mov_f32 v0,p0,attr0.x: actual arbitrary flat input
        const auto d=rdna2_decode_one(&p.back(),1);
        check(d.fmt==Rdna2Format::VINTRP && d.opcode==2 && d.vintrp_attr==0 && d.vintrp_chan==0,
              "varying relation fixture decodes real P0 attr0.x, not an impossible FragCoord value");
    }
    // Keep the runtime word's SSA identity but use a VGPR compare: scalar-only VOPC branches
    // are already scalar and deliberately do not exercise the fragment Any lowering path.
    if(!o.varying) p.push_back(0x7e000214u); // v0=s20, same runtime word SSA
    uint32_t raw=256u,zero=o.negative_zero?23u:128u;
    if(o.encoding==Encoding::E32) {
        mov_vector(p,1,o.negative_zero?0x80000000u:0);
        zero=o.mirror || o.vector_zero?257u:zero;
    }
    if(o.cmpx) {
        Options plain;
        compare(p,plain,0xc2,106,128,21); // old VCC=(word1==0), deliberately unlike CMPX
    }
    if(o.narrow || o.cmpx) {
        Options plain;
        compare(p,plain,0xd5,126,128,21); // EXEC &= (word1!=0)
    }
    compare(p,o,(o.neq?0x0du:0x02u)+(o.cmpx?0x10u:0u),o.named?8u:o.cmpx?126u:106u,
            o.mirror?raw:zero,o.mirror?zero:raw);
    if(o.cmpx) p.push_back(0xbe8a047eu); // save ACTUAL EXEC in s[10:11]
    if(o.narrow || o.cmpx) p.push_back(0xbefe04c1u); // restore full EXEC before observing masks
    if(o.named) p.push_back(0xbeea0408u); // actual saved scalar destination -> VCC
    tag(p,8,o.named?8u:106u);
    if(o.cmpx) tag(p,10,10); // independent sink for CMPX EXEC, beside preserved VCC
    if(with_branch) { p.push_back(0xbf870002u); mov_vector(p,9,0x40400000u); }
    p.insert(p.end(),{0xf800180fu,o.cmpx?0x090a0908u:0x09080908u,0xbf810000u});
    return p;
}
std::vector<uint32_t> translate(const std::vector<uint32_t>& p,FragmentFloatMode mode,uint32_t wave,
                                uint32_t target=UINT32_MAX,uint64_t diagnostic=0) {
    ShaderResourceTable rt; ShaderResource r;
    r.cls=ResourceClass::ConstantBuffer; r.format=DataFormat::Uint32; r.sgpr_base=0;
    r.binding=32; r.size=8; rt.resources.push_back(r);
    PixelSystemInputMapping input; input.ena=input.addr=1u<<8;
    ++modules;
    return recompile_fragment(p.data(),p.size(),&rt,&input,target,nullptr,wave==32,
        {RecompileDiagnosticStage::Fragment,diagnostic?diagnostic:++address},mode);
}
void dump(const char* dir,const std::string& name,const std::vector<uint32_t>& w) {
    if(!dir || w.empty()) return;
    std::ofstream out(std::filesystem::path(dir)/(name+".spv"),std::ios::binary);
    const auto bytes=std::as_bytes(std::span(w));
    out.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size())); out.close();
    check(bool(out),name+" dump closed successfully");
}
void positive(const Options& o,uint8_t mode,uint32_t wave,const std::string& name,const char* dir,bool authority) {
    const auto source=translate(program(o),{true,mode},wave); const Module m(source);
    check(m.valid && !source.empty(),name+" actual guest translator emitted SOURCE"); if(!m.valid) return;
    dump(dir,name+"_source",source);
    const uint32_t predicate=m.exported_predicate();
    check(predicate!=0,name+" typed MRT0.x is a tagged live Boolean");
    check(m.count(any)==1 && m.live_green_phi(),name+" wave branch changes actual MRT0.y between2 and3");
    uint32_t vote=0,type=0;
    for(const auto& i:m.ins) if(i.op==any && i.a.size()==4 && i.a[3]==predicate && m.literal(i.a[2],3)) {
        vote=i.a[1]; type=i.a[0];
    }
    check(vote && m.controls_branch(vote),name+" real branch and MRT0.x consume SAME predicate");
    for(const uint32_t word:samples) for(const uint32_t gate:{0u,1u}) {
        Oracle oracle{m,{word,gate},word}; const bool actual=oracle.bool_value(predicate);
        const bool expected=o.cmpx?gate==0:guest_compare(word,mode,o.neq)&&(!o.narrow || gate!=0);
        check(oracle.good && !oracle.leaves.empty() && actual==expected,name+" defined live result matches independent guest relation");
        ++values;
        if(o.cmpx) {
            Oracle exec{m,{word,gate},word}; const uint32_t mask=m.exported_predicate(2);
            check(mask && exec.bool_value(mask)==(gate!=0 && guest_compare(word,mode,o.neq)) && exec.good,
                  name+" actual EXEC sink is gated compare; old VCC remains independent");
        }
    }
    if(o.varying) {
        // CPU relation checks over the actual varying Input expression and the full-wave contract.
        // Only a high-half invocation is nonzero; no GPU/native-subgroup admission is claimed.
        check(fragment_spirv_required_subgroup_size(source)==wave,name+" native SOURCE requests actual guest wave width");
        for(const uint32_t high:{31u,32u,63u}) if(high<wave) {
            for(const uint32_t word:{1u,0x00800000u,0x7f800001u}) {
                bool actual_any=false,expected_any=false;
                for(uint32_t lane=0;lane<wave;++lane) {
                    const uint32_t input=lane==high?word:0;
                    Oracle lane_oracle{m,{0,1},input}; const bool bit=lane_oracle.bool_value(predicate);
                    check(lane_oracle.good && bit==guest_compare(input,mode,o.neq),name+" actual varying predicate includes high-half lane");
                    actual_any|=bit; expected_any|=guest_compare(input,mode,o.neq);
                }
                check(actual_any==expected_any,name+" complete-wave Any relation includes lane31/32/63");
            }
        }
        const auto lowered=lower_fragment_votes(source,true,true);
        check(lowered.refusal==(wave==32?FragmentVoteRefusal::InconsistentContract:FragmentVoteRefusal::UnprovedVote) &&
              lowered.words.empty(),
              name+" mode does NOT grant varying Input uniformity"); return;
    }
    if(!authority) return;
    const auto original=source;
    for(const bool immutable:{false,true}) for(const bool deterministic:{false,true}) {
        const auto lowered=lower_fragment_votes(source,immutable,deterministic);
        const bool admitted=wave==64 && immutable && deterministic;
        const auto refused=wave==32?FragmentVoteRefusal::InconsistentContract:
            admitted?FragmentVoteRefusal::None:FragmentVoteRefusal::UnprovedVote;
        check(source==original && lowered.refusal==refused &&
              (admitted?!lowered.words.empty() && lowered.uniform_votes==1 && lowered.dead_votes==0 && lowered.neutral_votes==0:
                         lowered.words.empty() && lowered.uniform_votes==0 && lowered.dead_votes==0 && lowered.neutral_votes==0),
              name+" immutable/read-determinism authorities remain independently necessary");
        if(admitted) {
            const Module effective(lowered.words); const auto* c=effective.def(vote);
            check(c && c->op==copy && c->a==std::vector<uint32_t>{type,vote,predicate} && effective.controls_branch(vote) &&
                  effective.exported_predicate()==predicate && effective.count(any)==0 &&
                  fragment_spirv_required_subgroup_size(lowered.words)==0,name+" effective branch copies exact live predicate, not TRUE");
            for(const auto word:samples) {
                Oracle oracle{effective,{word,1},word}; const bool result=oracle.bool_value(predicate);
                check(oracle.good && result==guest_compare(word,mode,o.neq),name+" effective predicate still obeys guest modes/NaNs");
            }
            dump(dir,name+"_effective",lowered.words);
        }
    }
}
void exclusion(Options o,const std::string& name,const char* dir,bool reject) {
    for(const uint8_t mode:{uint8_t(0),uint8_t(16)}) {
        Options clean=o; clean.abs_word=clean.abs_zero=clean.neg_word=clean.neg_zero=clean.clamp=false;
        clean.partial=clean.sext=clean.dpp=false; clean.omod=0;
        const auto control=translate(program(clean,false),{true,mode},64); const Module positive(control);
        Oracle defined{positive,{1,1}}; const auto sink=positive.exported_predicate(); const bool result=defined.bool_value(sink);
        check(positive.valid && sink && defined.good && result==guest_compare(1,mode,o.neq),
              name+" same-region clean modifier/control reaches actual typed zero predicate");
        dump(dir,name+"_clean_mode"+std::to_string(mode),control);
        const auto source=translate(program(o,false),{true,mode},64);
        if(reject) { check(source.empty(),name+" excluded packet rejects beside clean same-route control"); continue; }
        const Module m(source); check(m.valid && !source.empty(),name+" legacy excluded form still translates");
        if(!m.valid) continue;
        Oracle oracle{m,{1,1},1}; const auto p=m.exported_predicate(); oracle.bool_value(p);
        check(p && !oracle.good,name+" excluded form does not receive certified integer zero predicate");
        dump(dir,name+"_mode"+std::to_string(mode),source);
    }
}
// Capture only this small test diagnostic using the existing disk-backed, per-process scratch.
template<class F> std::pair<std::vector<uint32_t>,std::string> stderr_capture(F compile) {
    const auto path=prosper_test::test_scratch_path("fragment-float-mode.stderr");
    FILE* file=std::fopen(path.string().c_str(),"w+b");
#if defined(_WIN32)
    const auto fileno_fn=::_fileno; const auto dup_fn=::_dup; const auto dup2_fn=::_dup2; const auto close_fn=::_close;
#else
    const auto fileno_fn=::fileno; const auto dup_fn=::dup; const auto dup2_fn=::dup2; const auto close_fn=::close;
#endif
    const int saved=file?dup_fn(fileno_fn(stderr)):-1;
    check(file && saved>=0,"stderr diagnostic capture opened private disk-backed file");
    if(!file || saved<0) { if(file) std::fclose(file); return {compile(),{}}; }
    std::fflush(stderr); const int redirected=dup2_fn(fileno_fn(file),fileno_fn(stderr));
    check(redirected>=0,"stderr diagnostic capture connected");
    auto source=compile(); std::fflush(stderr);
    check(dup2_fn(saved,fileno_fn(stderr))>=0,"stderr diagnostic capture restored original stream"); close_fn(saved);
    std::rewind(file); std::string text; std::array<char,512> buffer{};
    while(const auto n=std::fread(buffer.data(),1,buffer.size(),file)) text.append(buffer.data(),n);
    check(!std::ferror(file),"stderr diagnostic capture read complete"); std::fclose(file);
    return {std::move(source),std::move(text)};
}
void unavailable(const char* dir) {
    Options o; const auto code=program(o);
    const auto first=stderr_capture([&]{return translate(code,{},64,UINT32_MAX,0x4056ffffu);});
    const Module m(first.first); check(m.valid && m.exported_predicate(),"unknown launch retains real legacy shader");
    Oracle oracle{m,{1,1}}; oracle.bool_value(m.exported_predicate());
    check(!oracle.good && m.count(183)==1,"unknown mode retains actual unordered FP NEQ, never fake preserve/flush");
    check(first.second.find("FLOAT_MODE=unavailable")!=std::string::npos &&
          first.second.find("host-dependent legacy FP path")!=std::string::npos &&
          first.second.find("exact guest semantics unverified")!=std::string::npos,
          "unknown eligible compare announces exact compatibility limit");
    const auto second=stderr_capture([&]{return translate(code,{},64,UINT32_MAX,0x4056ffffu);});
    check(second.first==first.first && second.second.find("FLOAT_MODE=unavailable")==std::string::npos,
          "same producer warning is bounded without changing stored shader output");
    const auto lower=lower_fragment_votes(first.first,true,true);
    check(lower.refusal==FragmentVoteRefusal::UnprovedVote && lower.words.empty(),"unknown mode gains no integer uniformity authority");
    check(translate(code,{false,16},64).empty(),"noncanonical unavailable mode rejected, not silently guessed");
    dump(dir,"unknown_legacy_source",first.first);
}
void poison_control(const char* dir) {
    Options o; o.narrow=true; auto source=translate(program(o),{true,0},64); const Module before(source);
    bool replaced=false;
    for(size_t pc=5;pc<source.size();) {
        const size_t n=source[pc]>>16; if(!n || n>source.size()-pc) break;
        if((source[pc]&0xffffu)==load && n==4) {
            const auto* d=before.def(source[pc+2]); uint32_t ix=UINT32_MAX;
            if(d && before.word_load(*d,ix) && ix==0) {
                source[pc]=(3u<<16)|1u; source.erase(source.begin()+pc+3); replaced=true; break;
            }
        }
        pc+=n;
    }
    check(replaced,"explicit SPIR-V obligation replaces exact live typed raw producer with OpUndef1");
    const Module m(source); Oracle inactive{m,{0,0}},active{m,{0,1}};
    const auto p=m.exported_predicate(); const bool inactive_value=inactive.bool_value(p); active.bool_value(p);
    check(p && inactive.good && !inactive_value && !active.good,
          "actual EXEC Select suppresses inactive Undef but active undefined input has no oracle value");
    const auto original=source; const auto lowered=lower_fragment_votes(source,true,true);
    check(source==original && lowered.refusal==FragmentVoteRefusal::UnprovedVote && lowered.words.empty(),
          "mode/immutable/robust declarations cannot mint stability for active Undef");
    dump(dir,"undef_masked_obligation_source",source);
}
std::vector<uint32_t> mode_write(uint32_t op,bool preserve) {
    // Real HW_REG_MODE=1, FP32_DENORM field offset4,width2; no fabricated decoder opcode.
    const uint32_t hwreg=1u|(4u<<6)|(1u<<11);
    if(op==0x13) return {0xb0000000u|(op<<23)|(23u<<16)|hwreg};
    if(op==0x15) return {0xb0000000u|(op<<23)|hwreg,preserve?1u:0u};
    return {0xbf800000u|(op<<16)|(preserve?1u:0u)};
}
std::pair<std::vector<uint32_t>,uint32_t> dispatch(const std::vector<uint32_t>& alternate) {
    // Project-owned PC-relative qword dispatch, same generic pattern as test_dynfetch_fold.
    std::vector<uint32_t> p{0xf4201a8cu,0xfa000010u,0x816ac16au,0x83ea826au,0x8f6a836au,
        0xbea01f00u,0x802020ffu,0,0x82212180u,0xf4040890u,0xd4000000u,
        0xbea81f00u,0x80282228u,0x82292329u,0xbe802028u};
    p.insert(p.end(),alternate.begin(),alternate.end()); const size_t jump=p.size(); p.push_back(0);
    const uint32_t target=static_cast<uint32_t>(p.size()); Options o;
    auto body=program(o,false); body.resize(body.size()-3); p.insert(p.end(),body.begin(),body.end());
    const uint32_t merge=static_cast<uint32_t>(p.size());
    p[jump]=0xbf820000u|static_cast<uint32_t>(merge-jump-1);
    p.insert(p.end(),{0xf800180fu,0x09080908u,0xbf810000u});
    if(p.size()&1u) p.push_back(0);
    p[7]=static_cast<uint32_t>(p.size()*4u-24u);
    for(const uint32_t pc:{15u,target,merge}) p.insert(p.end(),{pc*4u-48u,0u});
    return {std::move(p),target};
}
void mode_controls(const char* dir) {
    for(const uint32_t op:{0x13u,0x15u,0x24u,0x25u}) for(const uint8_t mode:{uint8_t(0),uint8_t(16)}) {
        const auto write=mode_write(op,mode==0); const auto decoded=rdna2_decode_one(write.data(),write.size());
        check(decoded.opcode==op && decoded.len_dwords==write.size() &&
              decoded.fmt==(op<0x20?Rdna2Format::SOPK:Rdna2Format::SOPP),"true MODE write opcode/length decode");
        const std::vector<uint32_t> nops(write.size(),0xbf800000u); Options o; const auto body=program(o,false);
        for(unsigned route=0;route<4;++route) {
            std::vector<uint32_t> bad,good; uint32_t target=UINT32_MAX;
            if(route==3) {
                auto b=dispatch(write),g=dispatch(nops); bad=std::move(b.first); good=std::move(g.first); target=b.second;
                const auto info=rdna2_pcrel_dispatch_info(good.data(),good.size());
                check(info.valid && info.target_pcs.size()==3 && target==g.second,
                      "MODE specialization control has genuine valid PC-relative table/target");
            } else {
                mov_scalar(bad,23,mode==0?1u:0u); good=bad;
                if(route==1) { bad.push_back(0xbefe0480u); good.push_back(0xbefe0480u); }
                if(route==2) {
                    // An unconditional S_BRANCH is itself outside the fragment structurizer's
                    // accepted route. Use an independently always-true scalar SCC branch so the
                    // NOP control reaches the live compare after the skipped write region.
                    const uint32_t equal=0xbf060000u|(23u<<8)|23u; // s_cmp_eq_u32 s23,s23
                    const auto d=rdna2_decode_one(&equal,1);
                    check(d.fmt==Rdna2Format::SOPC && d.opcode==6 && d.len_dwords==1,
                          "dead MODE region uses a real scalar equality before its SCC1 branch");
                    bad.push_back(equal); good.push_back(equal);
                    bad.push_back(0xbf850000u|uint32_t(write.size())); good.push_back(bad.back());
                }
                bad.insert(bad.end(),write.begin(),write.end()); good.insert(good.end(),nops.begin(),nops.end());
                if(route==1) { bad.push_back(0xbefe04c1u); good.push_back(0xbefe04c1u); }
                bad.insert(bad.end(),body.begin(),body.end()); good.insert(good.end(),body.begin(),body.end());
            }
            const auto name="modewrite_"+std::to_string(op)+"_launch"+std::to_string(mode)+"_route"+std::to_string(route);
            const auto control=translate(good,{true,mode},64,target); const Module m(control);
            Oracle oracle{m,{1,1}}; const auto p=m.exported_predicate(); const bool bit=oracle.bool_value(p);
            check(m.valid && p && oracle.good && bit==guest_compare(1,mode,true),name+" same-region S_NOP positive emits exact live zero predicate");
            const uint64_t diagnostic=++address; const auto refused=translate(bad,{true,mode},64,target,diagnostic);
            check(refused.empty() && last_terminal_reject_reason(diagnostic).find("fragment mode write unsupported")!=std::string::npos,
                  name+" MODE authority refused before dead EXEC/branch/specialization can hide write");
            dump(dir,name+"_control",control);
        }
    }
}
void absorption_controls(const char* dir) {
    for(const uint8_t mode:{uint8_t(0),uint8_t(16)}) for(unsigned changed=0;changed<3;++changed) {
        std::vector<uint32_t> p; buffer_load(p,20,0); mov_scalar(p,22,0x00800000u); Options eq; eq.neq=false;
        compare(p,eq,2,8,128,20);
        if(changed) buffer_load(p,changed==1?21u:20u,1);
        Options range; range.mirror=true; range.abs_word=true;
        compare(p,range,3,106,changed==1?21u:20u,22);
        p.push_back(0x80000000u|(0x11u<<23)|(106u<<16)|(8u<<8)|106u); tag(p,8,106);
        p.insert(p.end(),{0xf800180fu,0x08080808u,0xbf810000u});
        const auto source=translate(p,{true,mode},64); const Module m(source); const auto sink=m.exported_predicate(); const auto* d=m.def(sink);
        check(m.valid && d && d->op==(changed?lor:ule),"mode-aware EQ facts absorb only same raw SSA magnitude relation");
        for(const uint32_t first:{0u,1u,0x00800001u,0x7f800001u}) for(const uint32_t second:{0u,1u,0x7f800000u}) {
            const uint32_t word=changed?second:first; const uint32_t exp=(word>>23)&255u,frac=word&0x007fffffu;
            const bool in_range=exp<1 || (exp==1 && frac==0);
            Oracle oracle{m,{first,second}}; const bool actual=oracle.bool_value(sink);
            check(oracle.good && actual==(guest_zero(first,mode)||in_range),"actual absorbed/retained OR respects mode and raw SSA provenance");
            check(oracle.leaves.size()==(changed?2u:1u),
                  "leaf identity counts distinct loads, not repeated use of one SSA word");
        }
        dump(dir,"absorption_mode"+std::to_string(mode)+"_ssa"+std::to_string(changed),source);
    }
}
} // namespace

int main(int argc,char** argv) {
    if(argc!=1 && !(argc==3 && (std::string_view(argv[1])=="--dump-directory" || std::string_view(argv[1])=="--dump"))) return 2;
    const char* dir=argc==3?argv[2]:nullptr; if(dir) std::filesystem::create_directories(dir);
    reset_float_controls_support_for_test();
    // Every other mode bit is exercised, not collapsed at the public caller to a denorm Boolean.
    std::array<std::vector<uint32_t>,2> canonical_eq,canonical_neq;
    for(unsigned mode=0;mode<256;++mode) for(const bool neq:{false,true}) {
        Options o; o.neq=neq;
        const auto source=translate(program(o),{true,static_cast<uint8_t>(mode)},64); const Module m(source);
        const auto p=m.exported_predicate(); auto& canonical=(neq?canonical_neq:canonical_eq)[(mode>>4)&1u];
        if(canonical.empty()) canonical=source;
        check(!source.empty() && source==canonical,"all256 mode bytes: only actual F32 input-denorm bit changes zero module");
        for(const uint32_t word:samples) { Oracle oracle{m,{word,1}}; const bool result=oracle.bool_value(p);
            // Reusing one SSA word is not a second load; a distinct load still adds a leaf.
            check(p && oracle.good && oracle.leaves.size()==1 && result==guest_compare(word,static_cast<uint8_t>(mode),neq),
                  "all256 mode bytes: actual typed exported value matches independent guest categories"); ++values; }
        if(mode==0 || mode==16 || mode==32 || mode==48) dump(dir,"modebyte"+std::to_string(mode)+(neq?"_neq":"_eq"),source);
    }
    for(const auto encoding:{Encoding::E32,Encoding::Sdwa,Encoding::E64})
        for(const bool neq:{false,true}) for(const bool mirror:{false,true}) for(const bool negzero:{false,true})
            for(const uint32_t wave:{32u,64u}) for(const uint8_t mode:{uint8_t(0),uint8_t(16),uint8_t(32),uint8_t(48)}) {
                Options o; o.encoding=encoding; o.neq=neq; o.mirror=mirror; o.negative_zero=negzero;
                positive(o,mode,wave,"enc"+std::to_string(unsigned(encoding))+"_neq"+std::to_string(neq)+
                    "_mirror"+std::to_string(mirror)+"_negzero"+std::to_string(negzero)+"_wave"+std::to_string(wave)+
                    "_mode"+std::to_string(mode),dir,true);
            }
    for(const bool neq:{false,true}) for(const uint8_t mode:{uint8_t(0),uint8_t(16)})
        for(const uint32_t wave:{32u,64u}) for(unsigned route=0;route<4;++route) {
            Options o; o.neq=neq; o.narrow=route==0; o.cmpx=route==1; o.named=route==2; o.varying=route==3;
            positive(o,mode,wave,"maskroute"+std::to_string(route)+"_neq"+std::to_string(neq)+"_mode"+
                std::to_string(mode)+"_wave"+std::to_string(wave),dir,false);
        }
    const auto exclude=[&](const char* name,auto change,bool reject=false) { Options o; change(o); exclusion(o,name,dir,reject); };
    exclude("abs_word",[](auto& o){o.abs_word=true;}); exclude("abs_zero",[](auto& o){o.abs_zero=true;});
    exclude("neg_word",[](auto& o){o.neg_word=true;}); exclude("neg_zero",[](auto& o){o.neg_zero=true;});
    exclude("signaling_clamp",[](auto& o){o.clamp=true;}); exclude("reserved_omod",[](auto& o){o.omod=1;});
    exclude("partial_sdwa",[](auto& o){o.encoding=Encoding::Sdwa;o.partial=true;},true);
    exclude("sext_sdwa",[](auto& o){o.encoding=Encoding::Sdwa;o.sext=true;},true);
    exclude("dpp_compare",[](auto& o){o.encoding=Encoding::E32;o.dpp=true;o.vector_zero=true;},true);
    unavailable(dir); poison_control(dir); mode_controls(dir); absorption_controls(dir);
    std::printf("== fragment_float_mode: %zu translator modules, %zu values, %zu assertions, %d failures ==\n",
                modules,values,assertions,failures); return failures?1:0;
}
