// #4066: project-owned packets and SSA; no device, game bytes or native-wave claim.
// Khronos SPV_KHR_float_controls2: per-instruction None overrides implicit fast math;
// it does not promise exact signaling-NaN payloads, arithmetic or guest denorm modes.
// https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_float_controls2.html
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/spirv_fragment_vote_lowering.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "../../fixtures/spirv_fragment_neutral_fixtures.hpp"
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace prosper::gpu;
namespace {
unsigned failures = 0, checks = 0, modules = 0;
void check(bool ok, const std::string& name) {
    ++checks; if (!ok) { ++failures; std::printf("[FAIL] %s\n", name.c_str()); }
}
struct Inst { uint32_t op; std::vector<uint32_t> a; };
struct Module {
    std::vector<Inst> ins;
    std::map<uint32_t, size_t> defs, types;
    bool valid = false;
    explicit Module(const std::vector<uint32_t>& w) {
        if (w.size() < 5 || w[0] != 0x07230203u) return;
        for (size_t at = 5; at < w.size();) {
            const size_t n = w[at] >> 16;
            if (!n || n > w.size() - at) return;
            Inst i{w[at] & 65535u, {w.begin()+at+1, w.begin()+at+n}};
            if (i.op >= 19 && i.op <= 33 && !i.a.empty()) types[i.a[0]] = ins.size();
            // Never guess that metadata/type/label IDs have values.
            switch (i.op) {
                case 1: case 41: case 42: case 43: case 59: case 61: case 65:
                case 80: case 81: case 83: case 124: case 128: case 129: case 132: case 133:
                case 166: case 167: case 168: case 169: case 170: case 171:
                case 172: case 174: case 176: case 178: case 197: case 198:
                case 199: case 245: case 335:
                    if (i.a.size() >= 2 && !defs.emplace(i.a[1], ins.size()).second) return;
                    break;
                default: break;
            }
            ins.push_back(std::move(i)); at += n;
        }
        valid = true;
    }
    const Inst* def(uint32_t id) const {
        const auto f = defs.find(id); return f == defs.end() ? nullptr : &ins[f->second];
    }
    const Inst* type(uint32_t id) const {
        const auto f = types.find(id); return f == types.end() ? nullptr : &ins[f->second];
    }
    bool f32(uint32_t t) const { const auto* i=type(t); return i && i->op==22 && i->a==std::vector<uint32_t>{t,32}; }
    bool u32(uint32_t t) const { const auto* i=type(t); return i && i->op==21 && i->a==std::vector<uint32_t>{t,32,0}; }
    bool int32(uint32_t t) const { const auto* i=type(t); return i && i->op==21 && i->a.size()==3 && i->a[1]==32; }
    bool boolean(uint32_t t) const { const auto* i=type(t); return i && i->op==20 && i->a.size()==1; }
    bool float_vector(uint32_t t) const {
        const auto* i=type(t); return i && i->op==23 && i->a.size()==3 && f32(i->a[1]) && i->a[2]>=2 && i->a[2]<=4;
    }
    bool decorated(uint32_t id, uint32_t kind, uint32_t value) const {
        for (const auto& i:ins) if(i.op==71 && i.a==std::vector<uint32_t>{id,kind,value}) return true;
        return false;
    }
    bool literal(uint32_t id, uint32_t bits) const {
        const auto* i=def(id); return i && i->op==43 && i->a.size()==3 && u32(i->a[0]) && i->a[2]==bits;
    }
    bool input_pointer(uint32_t id) const {
        const auto* d=def(id); const auto* p=d?type(d->a[0]):nullptr;
        return d && d->op==59 && d->a.size()==3 && d->a[2]==1 && p && p->op==32 && p->a.size()==3 && p->a[1]==1;
    }
    bool transport(const Inst& i) const {
        if (i.a.size()!=3) return false;
        if (i.op==124) {
            const auto* d=def(i.a[2]);
            return d && ((f32(i.a[0]) && int32(d->a[0])) || (int32(i.a[0]) && f32(d->a[0])));
        }
        return i.op==61 && (f32(i.a[0]) || float_vector(i.a[0])) && input_pointer(i.a[2]);
    }
    size_t count(uint32_t op) const { size_t n=0; for(const auto& i:ins) n+=i.op==op; return n; }
    bool extension(std::string_view text) const {
        for(const auto& i:ins) if(i.op==10) {
            std::string s; bool ended=false;
            for(uint32_t word:i.a) for(unsigned b=0;b<4 && !ended;++b) {
                const char c=char((word>>(b*8))&255u); if(!c) ended=true; else s+=c;
            }
            if(ended && s==text) return true;
        }
        return false;
    }
    bool capability(uint32_t cap) const {
        for(const auto& i:ins) if(i.op==17 && i.a==std::vector<uint32_t>{cap}) return true;
        return false;
    }
    bool szi_entry() const {
        uint32_t entry=0; for(const auto& i:ins) if(i.op==15 && i.a.size()>=3) entry=i.a[1];
        for(const auto& i:ins) if(i.op==16 && i.a==std::vector<uint32_t>{entry,4461,32}) return entry!=0;
        return false;
    }
    // The observed value must reach MRT0.r through a typed vec4 export and literal-tag Select.
    uint32_t exported_predicate() const {
        for(const auto& i:ins) if(i.op==62 && i.a.size()==2 && decorated(i.a[0],30,0)) {
            const auto* v=def(i.a[1]); if(!v || v->op!=80 || v->a.size()!=6 || !float_vector(v->a[0])) return 0;
            const auto* f=def(v->a[2]); if(!f || f->op!=124 || f->a.size()!=3 || !f32(f->a[0])) return 0;
            const auto* tag=def(f->a[2]);
            if(!tag || tag->op!=169 || tag->a.size()!=5 || !u32(tag->a[0]) ||
               !literal(tag->a[3],0x3f800000u) || !literal(tag->a[4],0)) return 0;
            const auto* p=def(tag->a[2]); return p && boolean(p->a[0])?tag->a[2]:0;
        }
        return 0;
    }
    bool integer_nonzero_product(uint32_t predicate) const {
        const auto* cmp=def(predicate);
        if(!cmp || cmp->op!=171 || cmp->a.size()!=4 || !boolean(cmp->a[0]) || !literal(cmp->a[3],0)) return false;
        const auto* product=def(cmp->a[2]);
        if(!product || product->op!=132 || product->a.size()!=4 || !u32(product->a[0])) return false;
        const auto* magnitude=def(product->a[2]); const auto* next=def(product->a[3]);
        return magnitude && magnitude->op==199 && magnitude->a.size()==4 && u32(magnitude->a[0]) &&
            literal(magnitude->a[3],0x7fffffffu) && next && next->op==128 && next->a.size()==4 &&
            next->a[0]==magnitude->a[0] && next->a[2]==magnitude->a[1] && literal(next->a[3],1);
    }
    bool live_branch(uint32_t predicate) const {
        uint32_t vote=0;
        for(const auto& i:ins) if(i.op==335 && i.a.size()==4 && i.a[3]==predicate && literal(i.a[2],3)) vote=i.a[1];
        bool branch=false,green_phi=false;
        for(const auto& i:ins) {
            if(i.op==250 && i.a.size()==3) {
                if(i.a[0]==vote && vote) branch=true;
                const auto* a=def(i.a[0]);
                if(a && a->op==169 && a->a.size()==5 && a->a[2]==vote && vote && boolean(a->a[0])) branch=true;
            }
            if(i.op==245 && i.a.size()==6 && u32(i.a[0]))
                green_phi|=(literal(i.a[2],0x40000000u) && literal(i.a[4],0x40400000u)) ||
                    (literal(i.a[4],0x40000000u) && literal(i.a[2],0x40400000u));
        }
        return branch && green_phi;
    }
    bool depends(uint32_t root,uint32_t leaf,unsigned depth=0) const {
        if(root==leaf) return true;
        const auto* d=def(root); if(!d || depth>80) return false;
        size_t end=d->a.size();
        switch(d->op) {
            case 81: end=3; break; // component selector is a literal, not an SSA value
            case 83: case 124: case 128: case 132: case 166: case 167: case 168: case 169:
            case 170: case 171: case 174: case 178: case 197: case 198: case 199: break;
            default: return false;
        }
        for(size_t k=2;k<end;++k) if(depends(d->a[k],leaf,depth+1)) return true;
        return false;
    }
    bool runtime_word(const Inst& i) const {
        if(i.op!=61 || i.a.size()!=3 || !u32(i.a[0])) return false;
        const auto* ac=def(i.a[2]);
        if(!ac || ac->op!=65 || ac->a.size()!=5 || !literal(ac->a[3],0) || !literal(ac->a[4],0)) return false;
        const auto* root=def(ac->a[2]); const auto* p=root?type(root->a[0]):nullptr;
        const auto* st=p && p->a.size()==3?type(p->a[2]):nullptr;
        const auto* arr=st && st->a.size()==2?type(st->a[1]):nullptr;
        bool offset=false;
        if(st) for(const auto& d:ins) offset|=d.op==72 && d.a==std::vector<uint32_t>{st->a[0],0,35,0};
        const auto* element=type(ac->a[0]);
        return offset && element && element->op==32 && element->a==std::vector<uint32_t>{ac->a[0],12,i.a[0]} &&
            root && root->op==59 && root->a.size()==3 && root->a[2]==12 &&
            decorated(root->a[1],34,1) && decorated(root->a[1],33,32) && p && p->op==32 && p->a[1]==12 &&
            st && st->op==30 && arr && arr->op==29 && arr->a.size()==2 && arr->a[1]==i.a[0] && decorated(arr->a[0],6,4);
    }
};

// Fail-closed integer/Boolean evaluator over ACTUAL emitted SSA. Float transport is only a
// symbolic Input carrier; there is no host floating arithmetic or NaN-payload identity oracle.
struct Oracle {
    const Module& m; uint32_t word; bool input; bool good=true; unsigned leaves=0;
    uint32_t value(uint32_t id, unsigned depth=0) {
        const auto* d=m.def(id);
        if(!d || depth>80 || d->a.size()<2) { good=false; return 0; }
        const auto& a=d->a;
        const auto arg=[&](size_t k) { if(k>=a.size()) {good=false;return 0u;} return value(a[k],depth+1); };
        if(d->op==43 && a.size()==3 && m.u32(a[0])) return a[2];
        if((d->op==41 || d->op==42) && a.size()==2 && m.boolean(a[0])) return d->op==41;
        if(d->op==61 && m.runtime_word(*d) && !input) {++leaves;return word;}
        if(d->op==124 && a.size()==3 && m.u32(a[0]) && input) {
            const auto* e=m.def(a[2]); const auto* l=e && e->a.size()==4?m.def(e->a[2]):nullptr;
            if(e && e->op==81 && m.f32(e->a[0]) && e->a[3]==0 && l && l->op==61 &&
               m.float_vector(l->a[0]) && l->a.size()==3 && m.input_pointer(l->a[2])) {++leaves;return word;}
        }
        if(d->op==83 && a.size()==3 && (m.u32(a[0]) || m.boolean(a[0]))) return arg(2);
        if(d->op==169 && a.size()==5) {
            const auto* c=m.def(a[2]); const auto* yes=m.def(a[3]); const auto* no=m.def(a[4]);
            if(!c || !yes || !no || !m.boolean(c->a[0]) ||
               (!m.u32(a[0]) && !m.boolean(a[0])) || yes->a[0]!=a[0] || no->a[0]!=a[0]) {
                good=false;return 0;
            }
            const uint32_t condition=arg(2); return arg(condition?3:4); // unselected poison not consumed
        }
        if(m.u32(a[0]) && a.size()==4) {
            const auto* x=m.def(a[2]); const auto* y=m.def(a[3]);
            if(!x || !y || !m.u32(x->a[0]) || !m.u32(y->a[0])) {good=false;return 0;}
            const uint32_t xvalue=arg(2),yvalue=arg(3);
            if(d->op==128) return xvalue+yvalue;
            if(d->op==132) return xvalue*yvalue;
            if(d->op==199) return xvalue&yvalue;
            if(d->op==197) return xvalue|yvalue;
            if(d->op==198) return xvalue^yvalue;
        }
        if(m.boolean(a[0]) && d->op==168 && a.size()==3) {
            const auto* operand=m.def(a[2]);
            if(!operand || !m.boolean(operand->a[0])) {good=false;return 0;}
            return !arg(2);
        }
        if(m.boolean(a[0]) && a.size()==4) {
            const auto* x=m.def(a[2]); const auto* y=m.def(a[3]);
            if(!x || !y) {good=false;return 0;}
            const uint32_t xv=arg(2),yv=arg(3);
            if((d->op==166 || d->op==167) && m.boolean(x->a[0]) && m.boolean(y->a[0]))
                return d->op==166?(xv||yv):(xv&&yv);
            if(m.u32(x->a[0]) && m.u32(y->a[0])) {
                if(d->op==170) return xv==yv;
                if(d->op==171) return xv!=yv;
                if(d->op==174) return xv>=yv;
                if(d->op==178) return xv<=yv;
            }
        }
        good=false; return 0; // includes Undef, FP, unknown load, Phi and Any
    }
};
constexpr std::array<uint32_t,27> samples{0,0x80000000u,1,0x80000001u,0x007fffffu,0x807fffffu,
    0x00800000u,0x80800000u,0x00800001u,0x3f800000u,0xbf800000u,0x7f7fffffu,0xff7fffffu,
    0x7f800000u,0xff800000u,0x7f800001u,0xff800001u,0x7fc00000u,0xffc00000u,0xffffffffu,
    0x3fffffffu,0xbfffffffu,0x40000000u,0xc0000000u,0x7ffffffeu,0xfffffffeu,0x7fffffffu};
bool expected_nonzero(uint32_t bits, uint8_t mode) {
    const bool zero=(bits&0x7fffffffu)==0;
    const bool subnormal=(bits&0x7f800000u)==0 && (bits&0x007fffffu)!=0;
    return !zero && ((mode&16u)!=0 || !subnormal);
}
std::vector<uint32_t> raw_program(bool input) {
    std::vector<uint32_t> p;
    if(input) p.push_back(0xc8020002u); // v_interp_mov_f32 v0,p0,attr0.x
    else p={0xf4200500u,0xfa000000u,0x7e000214u}; // s_buffer_load s20, then v0=s20
    p.insert(p.end(),{0x7e1202ffu,0x40000000u,0xd40d006au,128u|(256u<<9),
        0xd5010008u,128u|(242u<<9)|(106u<<18),0xbf870002u,0x7e1202ffu,0x40400000u,
        0xf800180fu,0x09080908u,0xbf810000u});
    return p;
}
std::vector<uint32_t> emit(bool input,uint32_t wave,uint8_t mode,FloatTransportProfile profile) {
    const auto raw=raw_program(input);
    const auto cmp=rdna2_decode_one(raw.data()+(input?3:5),2);
    check(cmp.fmt==Rdna2Format::VOPC && cmp.opcode==13,"actual raw packet has unordered NEQ F32 compare");
    ShaderResourceTable table; ShaderResource r;
    r.cls=ResourceClass::ConstantBuffer; r.format=DataFormat::Uint32; r.sgpr_base=0; r.binding=32; r.size=4;
    table.resources.push_back(r); ++modules;
    return recompile_fragment(raw.data(),raw.size(),input?nullptr:&table,nullptr,UINT32_MAX,nullptr,wave==32,
        {RecompileDiagnosticStage::Fragment,0x40660000u+modules},{true,mode},nullptr,{profile});
}
void dump(const char* dir,const std::string& name,const std::vector<uint32_t>& words) {
    if(!dir || words.empty()) return;
    std::ofstream out(std::filesystem::path(dir)/(name+".spv"),std::ios::binary);
    const auto bytes=std::as_bytes(std::span(words));
    out.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size())); out.close();
    check(bool(out),name+" dump close");
}
void inventory(const Module& m,bool explicit_profile,bool input,const std::string& name) {
    check(m.valid,name+" parsed SOURCE"); if(!m.valid) return;
    check(m.capability(6029)==explicit_profile && m.extension("SPV_KHR_float_controls2")==explicit_profile,
          name+" exact cap6029 and extension follow producing profile");
    check(!explicit_profile || (m.capability(4466) && m.szi_entry()),name+" explicit transport forces real SZI32 entry envelope");
    std::set<uint32_t> eligible,decorated; size_t inputs=0;
    for(const auto& i:m.ins) {
        if(m.transport(i)) {eligible.insert(i.a[1]);inputs+=i.op==61;}
        if(i.op==71 && i.a.size()>=2 && i.a[1]==40) {
            check(i.a.size()==3 && i.a[2]==0,name+" only None mask, never implicit fast flags");
            check(!i.a.empty() && decorated.insert(i.a[0]).second,name+" each result decorated exactly once");
        }
        check(!((i.op==16 || i.op==331) && i.a.size()>=2 && i.a[1]==6028),name+" no FPFastMathDefault");
    }
    check(!eligible.empty() && (input?inputs>0:inputs==0),name+" actual transport operations exercised");
    check(decorated==(explicit_profile?eligible:std::set<uint32_t>{}),name+" exact typed per-op None inventory; arithmetic excluded");
    for(uint32_t id:decorated) for(const auto& i:m.ins)
        check(!(i.op==71 && i.a==std::vector<uint32_t>{id,42}),name+" no NoContraction overlap");
}
void emitter_tests(const char* dir) {
    reset_float_controls_support_for_test(); reset_float_transport_config_for_test();
    for(bool input:{false,true}) for(uint32_t wave:{32u,64u}) for(uint8_t mode:{uint8_t(0),uint8_t(16)}) {
        const std::string stem=std::string(input?"input":"raw")+"_w"+std::to_string(wave)+"_m"+std::to_string(mode);
        const auto unknown=emit(input,wave,mode,FloatTransportProfile::Unknown);
        const auto implicit=emit(input,wave,mode,FloatTransportProfile::Implicit);
        check(!unknown.empty() && unknown==implicit,stem+" Unknown/Implicit retain exact legacy SOURCE words");
        inventory(Module(unknown),false,input,stem+"_unknown");
        const auto source=emit(input,wave,mode,FloatTransportProfile::ExplicitNonFinite32);
        const Module m(source); inventory(m,true,input,stem+"_explicit"); if(!m.valid) continue;
        dump(dir,stem+"_source",source);
        const uint32_t p=m.exported_predicate(); check(p!=0,stem+" predicate reaches actual typed MRT0.r tag");
        check((mode&16u)==0 || m.integer_nonzero_product(p),
            stem+" preserve-mode live zero relation uses bounded adjacent integer factors");
        for(uint32_t bits:samples) {
            Oracle o{m,bits,input}; const bool actual=o.value(p)!=0;
            check(o.good && o.leaves>0 && actual==expected_nonzero(bits,mode),stem+" actual integer predicate matches independent class oracle");
        }
        check(m.count(335)==1 && m.live_branch(p) && fragment_spirv_required_subgroup_size(source)==wave,
            stem+" SAME live predicate controls actual complete-wave branch and green Phi");
        const auto before=source; const auto lowered=lower_fragment_votes(source,true,true);
        check(source==before,stem+" lowering never mutates SOURCE");
        if(wave==32) check(lowered.words.empty() && lowered.refusal==FragmentVoteRefusal::InconsistentContract,stem+" width32 not silently admitted as64");
        else if(input) {
            check(lowered.words.empty() && lowered.refusal==FragmentVoteRefusal::UnprovedVote,stem+" explicit Input transport grants no uniform certificate");
            // Symbolic SSA lane witness, not physical Input-denorm or NaN payload preservation.
            for(uint32_t high:{31u,32u,63u}) {
                bool full=false,low=false;
                for(uint32_t lane=0;lane<64;++lane) { Oracle o{m,lane==high?0x7f800001u:0u,true};
                    const bool bit=o.value(p)!=0; check(o.good,stem+" high-half typed symbolic oracle defined"); full|=bit; if(lane<32) low|=bit; }
                check(full && low==(high<32),stem+" complete-wave and low32 differ for live high-half Input");
            }
        } else {
            check(lowered.refusal==FragmentVoteRefusal::None && !lowered.words.empty() && lowered.uniform_votes==1,
                stem+" retained immutable/robust raw-word certificate admits actual SOURCE");
            inventory(Module(lowered.words),true,false,stem+"_effective"); dump(dir,stem+"_effective",lowered.words);
            check(lower_fragment_votes(source,false,true).words.empty() && lower_fragment_votes(source,true,false).words.empty(),
                stem+" profile never replaces either buffer permission gate");
        }
        // Oracle self-controls: actual unknown/Undef dependencies never become fabricated zero.
        Oracle absent{m,0,input}; absent.value(0xffffffffu); check(!absent.good,stem+" missing SSA refused by oracle");
        auto undefined=source; bool inserted=false;
        for(size_t at=5;at<undefined.size();at+=undefined[at]>>16) {
            const uint32_t n=undefined[at]>>16,op=undefined[at]&65535u;
            const auto* d=n>=3?m.def(undefined[at+2]):nullptr;
            const bool leaf=d && m.depends(p,d->a[1]) &&
                (input?(op==124 && m.u32(d->a[0]) && m.transport(*d)):m.runtime_word(*d));
            if(leaf) {undefined[at]=(3u<<16)|1u;undefined.erase(undefined.begin()+at+3,undefined.begin()+at+n);inserted=true;break;}
        }
        const Module um(undefined); Oracle u{um,0,input}; u.value(p);
        check(inserted && um.valid && !u.good,stem+" actual Undef dependency cannot be assigned guessed zero by oracle");
        auto badmask=source; bool changed=false;
        for(size_t at=5;at<badmask.size();at+=badmask[at]>>16)
            if((badmask[at]&65535u)==43 && badmask[at]>>16==4 && badmask[at+3]==0x7fffffffu) {
                badmask[at+3]=0xffffffffu;changed=true;break;
            }
        const Module bm(badmask); Oracle b{bm,0x80000000u,input}; const bool wrong=b.value(p)!=0;
        check(changed && b.good && wrong!=expected_nonzero(0x80000000u,mode),stem+" same typed live sink detects sign-mask mutation");
    }
    // Exercise arithmetic as an ACTUAL emitting route, not only an illegal decoration fixture.
    // Its numerical behavior is outside this change; the transport profile must not decorate it.
    for(uint32_t opcode:{3u,8u}) {
        const std::vector<uint32_t> code{0x7e0002ffu,0x3f800000u,0x7e0202ffu,0x40000000u,
            (opcode<<25)|(1u<<9)|256u,0x7e020280u,0x7e040280u,0x7e0602f2u,
            0xf800180fu,0x03020100u,0xbf810000u};
        const auto decoded=rdna2_decode_one(code.data()+4,1);
        check(decoded.fmt==Rdna2Format::VOP2 && decoded.opcode==opcode,"actual arithmetic packet decoder control");
        ++modules;
        const auto source=recompile_fragment(code.data(),code.size(),nullptr,nullptr,UINT32_MAX,nullptr,false,
            {RecompileDiagnosticStage::Fragment,0x40660000u+modules},{true,16},nullptr,
            {FloatTransportProfile::ExplicitNonFinite32});
        const Module m(source); const std::string name=opcode==3?"actual_add":"actual_mul";
        inventory(m,true,false,name); check(m.count(opcode==3?129:133)==1,name+" genuine FP instruction is present and undecorated");
        dump(dir,name+"_source",source);
    }
    // Generated geometry can force SZI32 without emitting any eligible cross-float transport.
    // Its declaration words must still name the executing Explicit device prerequisite.
    const uint32_t mixed_ps[]{0xc80e0002u,0xc8110002u,0xf800000fu,0x03030303u,0xbf810000u};
    const auto layout=fragment_interpolation_layout(mixed_ps,std::size(mixed_ps));
    check(layout.valid && layout.requires_geometry,"actual mixed parameter/smooth layout needs geometry");
    const uint32_t vertex_code[]{0x7e000280u,0x7e020281u,0xf80008cfu,0x01010000u,0xbf810000u};
    std::array<std::vector<uint32_t>,3> geometry,vertex;
    for(unsigned p=0;p<3;++p) {
        const FloatTransportConfig config{static_cast<FloatTransportProfile>(p)};
        geometry[p]=recompile_interpolation_geometry(layout,false,false,config); ++modules;
        vertex[p]=recompile_vertex(vertex_code,std::size(vertex_code),nullptr,nullptr,false,0,
            {RecompileDiagnosticStage::Vertex,0x40660000u+modules},config); ++modules;
        for(bool generated:{true,false}) {
            const auto& words=generated?geometry[p]:vertex[p]; const Module m(words);
            const std::string name=std::string(generated?"geometry":"vertex")+"_profile"+std::to_string(p);
            check(m.valid && !words.empty(),name+" actual stage emitted");
            check(m.capability(6029)==config.explicit_nonfinite32() &&
                  m.extension("SPV_KHR_float_controls2")==config.explicit_nonfinite32(),
                  name+" actual WORDS name the complete Explicit prerequisite");
            check(m.capability(4466)==config.explicit_nonfinite32() && m.szi_entry()==config.explicit_nonfinite32(),
                  name+" SZI32 declaration follows pinned profile with ambient publisher absent");
            check(float_transport_module_supported(words.data(),words.size(),{FloatTransportProfile::ExplicitNonFinite32}),
                  name+" actual enabled executing profile accepts module");
            for(auto executing:{FloatTransportProfile::Unknown,FloatTransportProfile::Implicit})
                check(float_transport_module_supported(words.data(),words.size(),{executing})==!config.explicit_nonfinite32(),
                      name+" executing profile cannot acquire permission from producing metadata");
            if(generated) {
                size_t none=0; for(const auto& i:m.ins) none+=i.op==71 && i.a.size()>=2 && i.a[1]==40;
                check(m.count(124)==0 && none==0,name+" genuinely has no Bitcast or None decoration");
            }
            dump(dir,name+"_source",words);
        }
    }
    check(geometry[0]==geometry[1] && vertex[0]==vertex[1],"all legacy stage WORDS remain byte-identical Unknown/Implicit");
}

namespace fixture=prosper::test::fragment_neutral;
namespace packet=prosper::test::wave_width;
namespace fv=prosper::test::fragment_votes;
std::vector<uint32_t> explicit_module(std::vector<uint32_t> words,std::initializer_list<uint32_t> targets) {
    std::vector<uint32_t> caps; packet::instruction(caps,17,{6029}); fv::insert(words,5,caps);
    std::vector<uint32_t> ext; packet::extension(ext,"SPV_KHR_float_controls2"); fv::insert(words,fv::find(words,11),ext);
    std::vector<uint32_t> deco; for(uint32_t id:targets) packet::instruction(deco,71,{id,40,0});
    fv::insert(words,fv::find(words,71),deco); return words;
}
void expect_lower(const std::vector<uint32_t>& source,bool admitted,const std::string& name,const char* dir=nullptr) {
    const auto before=source; const auto r=lower_fragment_votes(source);
    check(source==before,name+" immutable input");
    check(admitted?(r.refusal==FragmentVoteRefusal::None && !r.words.empty() && r.neutral_votes==1):
          (r.refusal==FragmentVoteRefusal::UnprovedVote && r.words.empty()),name+" exact neutral proof result");
    if(admitted) {dump(dir,name+"_source",source);dump(dir,name+"_effective",r.words);}
}
void parser_refusal(const std::vector<uint32_t>& words,const std::string& name) {
    const auto r=lower_fragment_votes(words);
    check(r.words.empty() && r.refusal==FragmentVoteRefusal::UnsupportedWaveOperation,name+" exact inventory refusal");
}
void proof_tests(const char* dir) {
    const auto baseline=fixture::make_module(fixture::Shape::BuiltinReconsume);
    expect_lower(baseline,true,"implicit_finite_back",dir);
    expect_lower(explicit_module(baseline,{}),true,"envelope_only_does_not_invent_None_facts",dir);
    // Same stable scalar, SAME SSA, SAME live strict conjunction; only consuming environment
    // differs. Implicit cross-cast of the raw carrier in the newly executed body needs finite bits.
    const auto mixed=explicit_module(baseline,{117,118});
    expect_lower(mixed,false,"explicit_predicate_does_not_certify_finite_body");
    const auto explicit_all=explicit_module(baseline,{50,117,118,120});
    expect_lower(explicit_all,true,"explicit_defined_transport_reconsume",dir);
    auto location=explicit_module(fixture::make_module(fixture::Shape::LocationReconsume),{124,117,118});
    expect_lower(location,false,"explicit_location_not_stable");
    auto undef=explicit_module(fixture::make_module(fixture::Shape::UndefinedReconsume),{118});
    expect_lower(undef,false,"explicit_undef_not_finite_authority");
    // None supplies NP for transport but no equality of float payloads or unselected atoms.
    auto identity=explicit_all;
    const auto phi=fv::find(identity,245); identity[phi+1]=4; identity[phi+3]=51; identity[phi+5]=120;
    // Keep the shader well-typed: the formerly UInt32 state observation is now an F32 copy.
    for(size_t at=5;at<identity.size();at+=identity[at]>>16)
        if((identity[at]&65535u)==112 && identity[at+2]==114) identity[at]=(4u<<16)|83u;
    expect_lower(identity,false,"explicit_float_phi_not_bit_identity");
    for(auto shape:{fixture::Shape::BitcastPoisonConjunction,fixture::Shape::BitcastPoisonSelection}) {
        auto words=fixture::make_module(shape);
        expect_lower(words,shape==fixture::Shape::BitcastPoisonSelection,
            shape==fixture::Shape::BitcastPoisonSelection?"implicit_inactive_poison":"implicit_active_poison",dir);
        words=explicit_module(std::move(words),{85});
        expect_lower(words,true,shape==fixture::Shape::BitcastPoisonSelection?"explicit_inactive_nonfinite":"explicit_active_nonfinite",dir);
    }
    // Arithmetic and GLSL are deliberately still implicit/possibly-poisoned.
    expect_lower(explicit_module(fixture::make_module(fixture::Shape::FloatPoisonConjunction),{50}),false,"input_none_does_not_define_glsl");
    expect_lower(explicit_module(fixture::make_module(fixture::Shape::FloatPoisonSelection),{50}),true,"input_none_inactive_glsl_mask",dir);
    expect_lower(explicit_module(fixture::make_module(fixture::Shape::UnmaskedFloat),{50,85}),false,
        "transport_none_does_not_define_arithmetic");
    expect_lower(explicit_module(fixture::make_module(fixture::Shape::MaskedFloat),{50,85}),true,
        "transport_none_inactive_arithmetic_mask",dir);
    // Each invalid inventory changes one field of the known positive explicit module.
    auto wrong=explicit_all;
    for(size_t at=5;at<wrong.size();at+=wrong[at]>>16) if((wrong[at]&65535u)==71 && wrong[at+2]==40) {wrong[at+3]=1;break;}
    parser_refusal(wrong,"nonzero_fast_mask");
    wrong=explicit_all; fv::insert(wrong,fv::find(wrong,71),{(4u<<16)|71u,117,40,0}); parser_refusal(wrong,"duplicate_none");
    wrong=explicit_all; fv::insert(wrong,fv::find(wrong,71),{(3u<<16)|71u,117,42}); parser_refusal(wrong,"none_no_contraction_overlap");
    for(uint32_t target:{51u,53u,60u,80u,116u,999u}) {
        wrong=explicit_module(baseline,{target}); parser_refusal(wrong,"unsupported_target_"+std::to_string(target));
    }
    for(unsigned envelope=0;envelope<3;++envelope) {
        wrong=explicit_all;
        for(size_t at=5;at<wrong.size();) {
            const uint32_t n=wrong[at]>>16,op=wrong[at]&65535u;
            bool erase=(envelope==0 && op==17 && wrong[at+1]==6029) ||
                (envelope==2 && op==16 && wrong[at+2]==4461);
            if(envelope==1 && op==10) {
                std::vector<uint32_t> snippet{0x07230203u,0x00010300u,0,180,0}; snippet.insert(snippet.end(),wrong.begin()+at,wrong.begin()+at+n);
                erase=Module(snippet).extension("SPV_KHR_float_controls2"); }
            if(erase) {wrong.erase(wrong.begin()+at,wrong.begin()+at+n);break;} at+=n;
        }
        parser_refusal(wrong,"missing_envelope_"+std::to_string(envelope)); // parser-only malformed declaration controls
    }
    for(uint32_t opcode:{16u,331u}) {
        wrong=explicit_all; fv::insert(wrong,fv::find(wrong,71),{(5u<<16)|opcode,40,6028,4,8});
        parser_refusal(wrong,"FPFastMathDefault_"+std::to_string(opcode));
    }
}
} // namespace
int main(int argc,char** argv) {
    const char* dir=nullptr;
    if(argc==3 && std::string_view(argv[1])=="--dump-directory") {dir=argv[2];std::filesystem::create_directories(dir);}
    else if(argc!=1) {std::fprintf(stderr,"usage: test_fragment_float_transport [--dump-directory DIR]\n");return 2;}
    emitter_tests(dir); proof_tests(dir);
    std::printf("fragment float transport: %u checks, %u translator invocations, %u failures\n",checks,modules,failures);
    return failures?1:0;
}
