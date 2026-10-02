// #4086: project-owned guest packets -> actual SOURCE MRT0 values; no Vulkan device.
// AMD RDNA2 ISA 70648 sections 3.5, 6.2.2, 6.4 and 6.5 are the guest authority:
// OMOD is ignored by IEEE=1 or OUTPUT-denorm preservation of the RESULT width;
// CLAMP is independent and follows OMOD; DX10_CLAMP selects zero versus NaN.
// https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf
// AMD's machine-readable RDNA2 CLAMP/OMOD descriptions separately pin their order:
// https://gpuopen.com/machine-readable-isa/
//
// CTest: fragment_float_modifiers. Optional --dump <directory> retains each actual SOURCE.
// Deliberately bounded: exact normal results, qNaN CATEGORY, raw zero signs and
// the documented active-OMOD -0 -> +0 rule with independent disabling controls.
// No CLAMP(-0), signaling-NaN/payload, subnormal arithmetic, half-rounding boundary,
// FP16 overflow-mode, native device execution or general guest arithmetic claim.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace prosper::gpu;
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
unsigned failures = 0, checks = 0, calls = 0, numeric_values = 0, decoded_packets = 0;
void check(bool ok, const std::string& name) {
    ++checks;
    if (!ok) { ++failures; std::printf("[FAIL] %s\n", name.c_str()); }
}
constexpr uint32_t one = 0x3f800000u, two = 0x40000000u, four = 0x40800000u;
constexpr uint32_t one_half = 0x3fc00000u, three_quarters = 0x3f400000u;
constexpr uint32_t qnan = 0x7fc00000u;
bool nan_bits(uint32_t x) { return (x & 0x7fffffffu) > 0x7f800000u; }

struct Inst { uint32_t op; std::vector<uint32_t> a; };
// A typed, straight-line SSA evaluator, not a reimplementation of fresult/clamp01.
// Unknown live definitions, Undef, loads, calls, branches and discard never get a value.
// The expected guest results below are literals, independent of this interpreter and MODE.
struct Module {
    std::vector<Inst> ins;
    std::map<uint32_t, size_t> definitions, types;
    bool valid = false;
    explicit Module(const std::vector<uint32_t>& w) {
        if (w.size() < 5 || w[0] != 0x07230203u) return;
        unsigned functions = 0, labels = 0, returns = 0;
        bool inside = false, terminated = false;
        for (size_t pc = 5; pc < w.size();) {
            const size_t n = w[pc] >> 16;
            if (!n || n > w.size() - pc) return;
            Inst i{w[pc] & 65535u, {w.begin()+pc+1, w.begin()+pc+n}};
            if (inside && terminated && i.op!=56) return;
            if (i.op >= 19 && i.op <= 33 && !i.a.empty()) {
                if (!types.emplace(i.a[0], ins.size()).second) return;
            }
            switch (i.op) {
                case 1: case 12: case 41: case 42: case 43: case 44: case 59: case 61:
                case 80: case 81: case 83: case 124: case 127: case 128: case 129:
                case 130: case 131: case 132: case 133: case 136: case 156:
                case 166: case 167: case 168: case 169: case 170: case 171:
                case 174: case 176: case 178: case 180: case 181: case 182: case 183:
                case 184: case 185: case 186: case 187: case 188: case 189: case 190: case 191:
                case 194: case 196: case 197: case 198: case 199: case 200:
                    if (i.a.size() < 2 || !definitions.emplace(i.a[1], ins.size()).second) return;
                    break;
                default: break; // an unlisted live result is unresolved, not an invented zero
            }
            if (i.op == 54) { if (inside) return; inside = true; terminated = false; ++functions; }
            if (i.op == 248) { if (!inside) return; ++labels; }
            if (i.op == 253) { if (!inside || !i.a.empty()) return; terminated = true; ++returns; }
            if (i.op == 56) { if (!inside) return; inside = false; }
            if (inside) switch (i.op) {
                // Positive instruction inventory, including result forms whose LIVE evaluation
                // refuses below. Unknown effects/control cannot masquerade as dead pure SSA.
                case 0: case 8: case 54: case 248: case 253: case 317:
                case 1: case 12: case 59: case 61: case 62: case 80: case 81: case 83:
                case 124: case 127: case 128: case 129: case 130: case 131: case 132:
                case 133: case 136: case 156: case 166: case 167: case 168: case 169:
                case 170: case 171: case 174: case 176: case 178: case 180: case 181:
                case 182: case 183: case 184: case 185: case 186: case 187: case 188:
                case 189: case 190: case 191: case 194: case 196: case 197: case 198:
                case 199: case 200: break;
                default: return;
            }
            ins.push_back(std::move(i)); pc += n;
        }
        valid = !inside && functions == 1 && labels == 1 && returns == 1;
    }
    const Inst* def(uint32_t id) const {
        const auto i = definitions.find(id); return i == definitions.end() ? nullptr : &ins[i->second];
    }
    const Inst* type(uint32_t id) const {
        const auto i = types.find(id); return i == types.end() ? nullptr : &ins[i->second];
    }
    bool boolean(uint32_t t) const { const auto* i=type(t); return i && i->op==20 && i->a.size()==1; }
    bool u32(uint32_t t) const { const auto* i=type(t); return i && i->op==21 && i->a==std::vector<uint32_t>{t,32,0}; }
    bool f32(uint32_t t) const { const auto* i=type(t); return i && i->op==22 && i->a==std::vector<uint32_t>{t,32}; }
    unsigned float_vector(uint32_t t) const {
        const auto* i=type(t);
        return i && i->op==23 && i->a.size()==3 && f32(i->a[1]) && i->a[2]>=2 && i->a[2]<=4 ? i->a[2] : 0;
    }
    bool decorated(uint32_t id, uint32_t kind, uint32_t value) const {
        unsigned found=0;
        for (const auto& i:ins) if (i.op==71 && i.a.size()>=2 && i.a[0]==id && i.a[1]==kind) {
            if (i.a!=std::vector<uint32_t>{id,kind,value}) return false;
            ++found;
        }
        return found==1;
    }
    bool glsl(uint32_t id) const {
        for (const auto& i:ins) if (i.op==11 && i.a.size()>=2 && i.a[0]==id) {
            std::string name;
            for (size_t k=1;k<i.a.size();++k) for (unsigned b=0;b<4;++b) {
                const char c=char((i.a[k]>>(8*b))&255u);
                if (!c) return name=="GLSL.std.450";
                name+=c;
            }
        }
        return false;
    }
    uint32_t output() const {
        uint32_t result=0;
        for (const auto& i:ins) if (i.op==62) {
            if (i.a.size()!=2 || result || !decorated(i.a[0],30,0)) return 0;
            const auto* v=def(i.a[0]); const auto* p=v ? type(v->a[0]) : nullptr;
            const auto* value=def(i.a[1]);
            if (!v || v->op!=59 || v->a.size()!=3 || v->a[2]!=3 || !p || p->op!=32 ||
                p->a.size()!=3 || p->a[1]!=3 || !value || p->a[2]!=value->a[0] ||
                float_vector(value->a[0])!=4) return 0;
            result=i.a[1];
        }
        return result;
    }
    std::vector<uint32_t> words(const std::vector<uint32_t>& header) const {
        std::vector<uint32_t> w(header.begin(),header.begin()+5);
        for (const auto& i:ins) {
            w.push_back((uint32_t(i.a.size()+1)<<16)|i.op); w.insert(w.end(),i.a.begin(),i.a.end());
        }
        return w;
    }
};
struct Value { uint32_t type=0; std::array<uint32_t,4> bits{}; unsigned size=0; };
struct Oracle {
    const Module& m;
    bool good=true;
    uint32_t unpack(uint16_t h) {
        const uint32_t sign=uint32_t(h&0x8000u)<<16, exponent=(h>>10)&31u, mantissa=h&1023u;
        if (!exponent) {
            if (!mantissa) return sign;
            // These fixtures deliberately never ask for denorm arithmetic or conversion authority.
            good=false; return 0;
        }
        if (exponent==31) return sign|0x7f800000u|(mantissa?0x00400000u:0u);
        return sign|((exponent+112u)<<23)|(mantissa<<13);
    }
    uint16_t pack_exact(uint32_t x) {
        const uint16_t sign=uint16_t((x>>16)&0x8000u);
        const uint32_t exponent=(x>>23)&255u, mantissa=x&0x007fffffu;
        if (!exponent && !mantissa) return sign;
        if (exponent==255) return uint16_t(sign|0x7c00u|(mantissa?0x0200u:0u));
        if (exponent<113 || exponent>142 || (mantissa&0x1fffu)) { good=false; return 0; }
        return uint16_t(sign|((exponent-112u)<<10)|(mantissa>>13));
    }
    Value value(uint32_t id, unsigned depth=0) {
        const auto* d=m.def(id);
        if (!m.valid || !d || depth>96 || d->a.size()<2) {good=false;return {};}
        const auto& a=d->a;
        Value r{a[0],{},m.float_vector(a[0])};
        if (m.f32(a[0]) || m.u32(a[0]) || m.boolean(a[0])) r.size=1;
        if (!r.size) {good=false;return {};}
        const auto arg=[&](size_t n) { if(n>=a.size()) {good=false;return Value{};} return value(a[n],depth+1); };
        const auto same=[&](const Value& v) { if(v.type!=r.type || v.size!=r.size) good=false; };
        if (d->op==43 && a.size()==3 && r.size==1 && (m.u32(r.type) || m.f32(r.type))) {r.bits[0]=a[2];return r;}
        if ((d->op==41 || d->op==42) && a.size()==2 && m.boolean(r.type)) {r.bits[0]=d->op==41;return r;}
        if (d->op==83 && a.size()==3) {const Value v=arg(2);same(v);return v;}
        if (d->op==124 && a.size()==3 && r.size==1) {
            const Value v=arg(2);
            if (v.size!=1 || !((m.u32(r.type)&&m.f32(v.type)) || (m.f32(r.type)&&m.u32(v.type)))) good=false;
            r.bits=v.bits; return r;
        }
        if ((d->op==80 || d->op==44) && a.size()==r.size+2 && m.float_vector(r.type)) {
            for (unsigned k=0;k<r.size;++k) {const Value v=arg(k+2);if(v.size!=1 || !m.f32(v.type)) good=false;r.bits[k]=v.bits[0];}
            return r;
        }
        if (d->op==81 && a.size()==4 && m.f32(r.type)) {
            const Value v=arg(2); if(!m.float_vector(v.type) || a[3]>=v.size) {good=false;return {};}
            r.bits[0]=v.bits[a[3]];return r;
        }
        if (d->op==169 && a.size()==5) {
            const Value c=arg(2);
            const auto* yes=m.def(a[3]); const auto* no=m.def(a[4]);
            if (!m.boolean(c.type) || c.size!=1 || !yes || !no || yes->a[0]!=r.type || no->a[0]!=r.type) {good=false;return {};}
            // Select consumes only its chosen value; a STRICT operation below consumes both.
            const Value v=arg(c.bits[0]?3:4);same(v);return v;
        }
        if (d->op==12 && a.size()>=5 && m.glsl(a[2])) {
            const Value x=arg(4);
            if (a[3]==62 && a.size()==5 && r.size==2 && m.float_vector(r.type) && m.u32(x.type) && x.size==1) {
                r.bits[0]=unpack(uint16_t(x.bits[0]));r.bits[1]=unpack(uint16_t(x.bits[0]>>16));return r;
            }
            if (a[3]==58 && a.size()==5 && m.u32(r.type) && x.size==2 && m.float_vector(x.type)) {
                const uint16_t lo=pack_exact(x.bits[0]),hi=pack_exact(x.bits[1]);r.bits[0]=lo|(uint32_t(hi)<<16);return r;
            }
            if ((a[3]==79 || a[3]==80) && a.size()==6 && m.f32(r.type) && x.type==r.type) {
                const Value y=arg(5);same(y);
                // Vulkan's default FP environment permits NotNaN for GLSL NMin/NMax
                // even under SignedZeroInfNanPreserve. A live NaN cannot acquire the
                // host library's preferred value here; the actual integer Select
                // must keep this operation out of the chosen MRT0 dependency.
                if(nan_bits(x.bits[0]) || nan_bits(y.bits[0])) {good=false;return {};}
                {
                    const float xf=std::bit_cast<float>(x.bits[0]),yf=std::bit_cast<float>(y.bits[0]);
                    // We do NOT choose the unspecified equal-zero sign of GLSL min/max.
                    if(xf==0 && yf==0 && x.bits[0]!=y.bits[0]) {good=false;return {};}
                    r.bits[0]=(a[3]==79 ? xf<yf : xf>yf) ? x.bits[0] : y.bits[0];
                }
                return r;
            }
            good=false;return {};
        }
        if (d->op==156 && a.size()==3 && m.boolean(r.type)) {
            const Value x=arg(2);if(!m.f32(x.type)) good=false;r.bits[0]=nan_bits(x.bits[0]);return r;
        }
        if (d->op==168 && a.size()==3 && m.boolean(r.type)) {
            const Value x=arg(2);same(x);r.bits[0]=!x.bits[0];return r;
        }
        if (a.size()==4 && r.size==1) {
            const Value x=arg(2),y=arg(3); // no short-circuit, including logical AND/OR
            const uint32_t xb=x.bits[0],yb=y.bits[0];
            if (m.f32(r.type) && x.type==r.type && y.type==r.type &&
                (d->op==129 || d->op==131 || d->op==133)) {
                if(nan_bits(xb) || nan_bits(yb)) {r.bits[0]=qnan;return r;}
                const auto outside_normal_or_zero=[](uint32_t b) {
                    const uint32_t magnitude=b&0x7fffffffu;
                    return magnitude && (magnitude<0x00800000u || magnitude>=0x7f800000u);
                };
                if(outside_normal_or_zero(xb) || outside_normal_or_zero(yb)) {good=false;return {};}
                const float xf=std::bit_cast<float>(xb),yf=std::bit_cast<float>(yb);
                const float f=d->op==129 ? xf+yf : d->op==131 ? xf-yf : xf*yf;
                r.bits[0]=std::bit_cast<uint32_t>(f);
                if(outside_normal_or_zero(r.bits[0])) good=false;
                return r;
            }
            if (m.u32(r.type) && x.type==r.type && y.type==r.type) {
                switch(d->op) {
                    case 128:r.bits[0]=xb+yb;return r; case 130:r.bits[0]=xb-yb;return r;
                    case 132:r.bits[0]=xb*yb;return r;
                    case 194:case 196:if(yb>=32) {good=false;return {};}r.bits[0]=d->op==194?xb>>yb:xb<<yb;return r;
                    case 197:r.bits[0]=xb|yb;return r;case 198:r.bits[0]=xb^yb;return r;case 199:r.bits[0]=xb&yb;return r;
                    default:break;
                }
            }
            if (m.boolean(r.type) && x.size==1 && y.size==1 && x.type==y.type) {
                if (m.boolean(x.type) && (d->op==166 || d->op==167)) {r.bits[0]=d->op==166?(xb||yb):(xb&&yb);return r;}
                if (m.u32(x.type)) switch(d->op) {
                    case 170:r.bits[0]=xb==yb;return r;case 171:r.bits[0]=xb!=yb;return r;
                    case 174:r.bits[0]=xb>=yb;return r;case 176:r.bits[0]=xb<yb;return r;case 178:r.bits[0]=xb<=yb;return r;
                    default:break;
                }
                if (m.f32(x.type) && d->op>=180 && d->op<=191) {
                    const bool unordered=nan_bits(xb)||nan_bits(yb);
                    const float xf=std::bit_cast<float>(xb),yf=std::bit_cast<float>(yb);
                    bool relation=false;
                    switch ((d->op-180)/2) {
                        case 0:relation=xf==yf;break;case 1:relation=xf!=yf;break;
                        case 2:relation=xf<yf;break;case 3:relation=xf>yf;break;
                        case 4:relation=xf<=yf;break;case 5:relation=xf>=yf;break;
                    }
                    r.bits[0]=(d->op&1u) ? unordered||relation : !unordered&&relation;return r;
                }
            }
        }
        good=false;return {}; // OpUndef, OpLoad, FDiv and every unhandled live opcode refuse
    }
};

enum class Path { Vop1Sdwa, Vop2Sdwa, Vop3, ScalarF16, Move };
const char* path_name(Path p) {
    switch(p) {case Path::Vop1Sdwa:return "vop1_sdwa";case Path::Vop2Sdwa:return "vop2_sdwa";
        case Path::Vop3:return "vop3";case Path::ScalarF16:return "scalar_f16";case Path::Move:return "move";}
    return "invalid";
}
struct Case {
    std::string name;
    Path path=Path::Vop3;
    uint32_t input=one, omod=0;
    bool clamp=false, ieee=false, dx10=false;
    uint8_t mode=0;
    uint32_t expected=one, opcode=0;
    bool expect_nan=false;
};
uint16_t fixture_half(uint32_t x) {
    // Explicit fixture encodings, NOT either the production packer or a guest numeric algorithm.
    switch(x) {
        case 0:return 0;case 0x80000000u:return 0x8000;
        case one:return 0x3c00;case one_half:return 0x3e00;case two:return 0x4000;
        case qnan:return 0x7e00;default:check(false,"unsupported fixture literal");return 0;
    }
}
void mov(std::vector<uint32_t>& w,uint32_t dst,uint32_t bits) { w.insert(w.end(),{0x7e0002ffu|(dst<<17),bits}); }
std::vector<uint32_t> program(const Case& c) {
    std::vector<uint32_t> w;
    mov(w,1,c.path==Path::Vop1Sdwa || c.path==Path::ScalarF16 ? fixture_half(c.input) : c.input);
    mov(w,2,c.path==Path::ScalarF16 ? fixture_half(c.opcode==0x34b ? one : c.input) : one);
    mov(w,3,c.path==Path::ScalarF16 ? fixture_half(c.opcode==0x34b ? 0u : c.input) : 0u);
    mov(w,4,0x35550000u); // scalar F16 preserves the other half; selected low half starts defined
    std::vector<uint32_t> packet;
    Rdna2Format format=Rdna2Format::VOP3;uint32_t opcode=0x108u;
    if(c.path==Path::Vop1Sdwa) {
        format=Rdna2Format::VOP1;opcode=0x0bu;
        packet={0x7e000000u|(4u<<17)|(opcode<<9)|0xf9u,
            1u|(6u<<8)|(6u<<16)|(uint32_t(c.clamp)<<13)|(c.omod<<14)};
    } else if(c.path==Path::Vop2Sdwa) {
        format=Rdna2Format::VOP2;opcode=8;
        packet={(opcode<<25)|(4u<<17)|(2u<<9)|0xf9u,
            1u|(6u<<8)|(6u<<16)|(6u<<24)|(uint32_t(c.clamp)<<13)|(c.omod<<14)};
    } else if(c.path==Path::Move) {
        format=Rdna2Format::VOP1;opcode=1;
        packet={0x7e000000u|(4u<<17)|(opcode<<9)|257u};
    } else {
        if(c.path==Path::ScalarF16) opcode=c.opcode;
        packet={0xd4000000u|(opcode<<16)|4u|(uint32_t(c.clamp)<<15),
            257u|(258u<<9)|(259u<<18)|(c.omod<<27)};
    }
    const auto d=rdna2_decode_one(packet.data(),packet.size());++decoded_packets;
    check(d.fmt==format && d.opcode==opcode && d.len_dwords==packet.size() &&
        d.dst.kind==OperandKind::VGPR && d.dst.value==4 && d.src[0].kind==OperandKind::VGPR &&
        d.src[0].value==1 && d.omod==c.omod && d.clamp==c.clamp && !d.has_modifier &&
        !d.src_abs[0] && !d.src_abs[1] && !d.src_abs[2] && !d.src_neg[0] && !d.src_neg[1] && !d.src_neg[2],
        c.name+" actual project decoder modifier path");
    if(c.path==Path::Vop1Sdwa || c.path==Path::Vop2Sdwa)
        check(d.has_sdwa && d.sdwa_dst_sel==6 && d.sdwa_src0_sel==6 &&
            (c.path==Path::Vop1Sdwa || d.sdwa_src1_sel==6),c.name+" actual DWORD SDWA selectors");
    if(c.path==Path::ScalarF16)
        check(d.vop3p_opsel==0,c.name+" actual unpacked scalar F16 low-half source/destination selectors");
    if(c.path==Path::Vop2Sdwa || c.path==Path::Vop3 || c.path==Path::ScalarF16)
        check(d.src[1].kind==OperandKind::VGPR && d.src[1].value==2 &&
            (c.path!=Path::ScalarF16 || (d.src[2].kind==OperandKind::VGPR && d.src[2].value==3)) &&
            d.n_src==(c.path==Path::ScalarF16 ? 3 : 2),
            c.name+" actual remaining arithmetic operands");
    w.insert(w.end(),packet.begin(),packet.end());
    const uint32_t sink_move=0x7e000000u|(5u<<17)|((c.path==Path::ScalarF16?0x0bu:1u)<<9)|260u;
    const auto sink_decode=rdna2_decode_one(&sink_move,1);
    check(sink_decode.fmt==Rdna2Format::VOP1 && sink_decode.opcode==(c.path==Path::ScalarF16?0x0bu:1u) &&
        sink_decode.src[0].kind==OperandKind::VGPR && sink_decode.src[0].value==4 && sink_decode.dst.value==5,
        c.name+" real result-to-export move/half conversion");
    w.push_back(sink_move);
    mov(w,6,0x3e800000u);mov(w,7,0xbf000000u);mov(w,8,one);
    w.insert(w.end(),{0xf800180fu,0x08070605u,0xbf810000u}); // real MRT0 RGBA, DONE/VM, ENDPGM
    return w;
}
bool matches(const Value& actual,const Case& c,const Module& m) {
    return actual.size==4 && m.float_vector(actual.type)==4 &&
        (c.expect_nan ? nan_bits(actual.bits[0]) : actual.bits[0]==c.expected) &&
        actual.bits[1]==0x3e800000u && actual.bits[2]==0xbf000000u && actual.bits[3]==one;
}
void dump(const std::filesystem::path& dir,const std::string& name,const std::vector<uint32_t>& w) {
    if(dir.empty() || w.empty()) return;
    std::ofstream file(dir/(name+".spv"),std::ios::binary);
    file.write(reinterpret_cast<const char*>(w.data()),std::streamsize(w.size()*4));
    check(bool(file),name+" dump writes entire actual SOURCE");
}
std::vector<uint32_t> run(const Case& c,unsigned wave,const std::filesystem::path& dir) {
    const auto raw=program(c);++calls;
    FragmentArithmeticObservation observation;
    const auto source=recompile_fragment(raw.data(),raw.size(),nullptr,nullptr,UINT32_MAX,nullptr,wave==32,
        {RecompileDiagnosticStage::Fragment,0x40860000u+calls*0x100u},{true,c.mode},&observation,
        {FloatTransportProfile::ExplicitNonFinite32},{true,c.ieee,c.dx10});
    const std::string name=c.name+"_w"+std::to_string(wave);
    check(!source.empty(),name+" compiler produces SOURCE");
    Module m(source);check(m.valid,name+" SOURCE has one straight-line function, not ignored control flow");
    const uint32_t sink=m.output();check(sink!=0,name+" actual typed MRT0 vec4 store");
    Oracle oracle{m};const Value actual=oracle.value(sink);
    check(oracle.good && actual.size==4,name+" all live typed dependencies evaluated, no invented values");
    for(unsigned k=0;k<4;++k) {
        ++numeric_values;
        const uint32_t expected=k==0?c.expected:k==1?0x3e800000u:k==2?0xbf000000u:one;
        const bool equal=k==0 && c.expect_nan ? nan_bits(actual.bits[k]) : actual.bits[k]==expected;
        check(oracle.good && actual.size==4 && equal,name+" actual MRT0["+std::to_string(k)+"] matches independent guest value");
    }
    dump(dir,name,source);return source;
}
void oracle_controls(const std::vector<uint32_t>& source,const std::filesystem::path& dir) {
    const Case expected{"oracle_seed",Path::Vop3,one,1,false,false,false,0,two};
    Module original(source);Oracle positive{original};
    const Value baseline=positive.value(original.output());
    check(positive.good && matches(baseline,expected,original),"oracle control has real numeric positive witness");
    bool found=false;
    for(const auto& i:original.ins) if(i.op==133 && i.a.size()==4) {found=true;break;}
    check(found,"oracle controls target an actual live FMul, not a marker");
    if(!found) return;
    for(unsigned variant=0;variant<4;++variant) {
        Module changed(source);
        for(auto& i:changed.ins) if(i.op==133 && i.a.size()==4) {
            if(variant==0) i.op=129; // legal typed FAdd: changed value must be noticed
            if(variant==1) i.op=136; // legal typed FDiv: unsupported live instruction must REFUSE
            if(variant==2) {i.op=1;i.a.resize(2);} // genuine, valid typed OpUndef, not opcode45
            if(variant==3) i.a[0]=0; // malformed type must not acquire a value
            break;
        }
        const auto words=changed.words(source);Module reparsed(words);Oracle oracle{reparsed};
        const Value result=oracle.value(reparsed.output());
        check(variant==0 ? oracle.good && !matches(result,expected,reparsed) : !oracle.good,
            "oracle live-value sensitivity/refusal control "+std::to_string(variant));
        // Only valid typed mutations are submitted to strict validation, never the malformed type arm.
        if(variant<3) dump(dir,"oracle_control_"+std::to_string(variant),words);
    }
    // Hand-construct live NaN NMin/NMax outside the guest fixture generator.
    // These valid typed modules must be unresolved under their default NotNaN
    // environment, not assigned the host library's preferred finite answer.
    uint32_t glsl_import=0;
    for(const auto& i:original.ins) if(i.op==11 && !i.a.empty() && original.glsl(i.a[0]))
        glsl_import=i.a[0];
    check(glsl_import!=0,"NaN live-operation controls use the actual GLSL import");
    for(uint32_t opcode:{79u,80u}) {
        Module changed(source);
        const uint32_t constant_id=source[3];
        uint32_t float_type=0, result_id=0;
        for(auto& i:changed.ins) if(i.op==133 && i.a.size()==4) {
            float_type=i.a[0];result_id=i.a[1];const uint32_t rhs=i.a[3];
            i.op=12;i.a={float_type,result_id,glsl_import,opcode,constant_id,rhs};break;
        }
        check(float_type!=0 && !original.decorated(result_id,40,0),
            "hand NaN control has no per-operation FPFastMathMode None");
        for(auto i=changed.ins.begin();i!=changed.ins.end();++i) if(i->op==54) {
            changed.ins.insert(i,Inst{43,{float_type,constant_id,qnan}});break;
        }
        auto header=source;header[3]=constant_id+1;
        const auto words=changed.words(header);Module reparsed(words);Oracle oracle{reparsed};
        (void)oracle.value(reparsed.output());
        check(reparsed.valid && reparsed.output()!=0 && !oracle.good,
            "oracle refuses a genuine live NaN NMin/NMax dependency "+std::to_string(opcode));
        dump(dir,"oracle_nan_control_"+std::to_string(opcode),words);
    }
    Module killed(source);
    for(auto& i:killed.ins) if(i.op==253) {i.op=252;break;}
    Module kill_module(killed.words(source));Oracle kill{kill_module};(void)kill.value(kill_module.output());
    check(!kill.good,"oracle refuses a real fragment Kill terminator even after a store");
    dump(dir,"oracle_kill_control",killed.words(source));
}
} // namespace

int main(int argc,char** argv) {
    std::filesystem::path dir;
    if(argc==3 && std::string_view(argv[1])=="--dump") {
        dir=argv[2];std::error_code error;std::filesystem::create_directories(dir,error);
        if(error || !std::filesystem::is_directory(dir)) {std::fprintf(stderr,"cannot create explicit dump directory\n");return 2;}
    } else if(argc!=1) {std::fprintf(stderr,"usage: test_fragment_float_modifiers [--dump directory]\n");return 2;}
    check(FragmentFloatFlags{}.canonical() && FragmentFloatFlags{}!=FragmentFloatFlags{true,false,false},
        "unavailable flags remain distinct from observed clear bits; no unknown modifier result prescribed");
    std::vector<uint32_t> seed;
    for(Path path:{Path::Vop1Sdwa,Path::Vop2Sdwa,Path::Vop3}) {
        const std::string prefix=path_name(path);
        std::vector<Case> cases;
        const auto add=[&](std::string name,uint32_t input,uint32_t omod,bool clamp,bool ieee,bool dx,uint8_t mode,uint32_t expected,bool nan=false) {
            cases.push_back({prefix+"_"+name,path,input,omod,clamp,ieee,dx,mode,expected,0,nan});
        };
        add("scale2_ieee0",one,1,false,false,false,0,two);
        add("scale2_ieee1",one,1,false,true,false,0,one);
        add("scale4_ieee0",one,2,false,false,false,0,four);
        add("scale4_ieee1",one,2,false,true,false,0,one);
        add("half_ieee0",one_half,3,false,false,false,0,three_quarters);
        add("half_ieee1",one_half,3,false,true,false,0,one_half);
        add("half_then_clamp_ieee0",one_half,3,true,false,false,0,three_quarters);
        add("half_then_clamp_ieee1",one_half,3,true,true,false,0,one);
        add("clamp_independent_ieee1",two,0,true,true,false,0,one);
        add("clamp_control_ieee0",two,0,true,false,false,0,one);
        add("f32_output_preserve",one,1,false,false,false,0x20,one);
        add("f32_input_preserve_only",one,1,false,false,false,0x10,two);
        add("f16_output_preserve_wrong_width",one,1,false,false,false,0x80,two);
        add("f16_input_preserve_wrong_width",one,1,false,false,false,0x40,two);
        add("clamp_independent_output_preserve",two,0,true,true,false,0x20,one);
        add("negative_zero_active_omod",0x80000000u,1,false,false,false,0,0);
        add("negative_zero_ieee_disables_omod",0x80000000u,1,false,true,false,0,0x80000000u);
        add("negative_zero_output_preserve_disables_omod",0x80000000u,1,false,false,false,0x20,0x80000000u);
        add("positive_zero_active_omod",0,1,false,false,false,0,0);
        add("qnan_passthrough_no_modifier",qnan,0,false,false,false,0,0,true);
        for(bool ieee:{false,true}) {
            add("qnan_clamp_dx0_ieee"+std::to_string(ieee),qnan,0,true,ieee,false,0,0,true);
            add("qnan_clamp_dx1_ieee"+std::to_string(ieee),qnan,0,true,ieee,true,0,0);
        }
        for(const auto& c:cases) for(unsigned wave:{32u,64u}) {
            auto source=run(c,wave,dir);
            if(path==Path::Vop3 && c.name=="vop3_scale2_ieee0" && wave==64) seed=std::move(source);
        }
    }
    for(uint32_t op:{0x34bu,0x351u,0x354u,0x357u}) {
        const std::string prefix="scalar_f16_"+std::to_string(op);
        // Same normal result for FMA(1,1,0) and min/max/median(1,1,1), all actually unpacked F16.
        const std::array<std::pair<uint8_t,uint32_t>,5> modes{{{0,two},{0x20,two},{0x80,one},{0xc0,one},{0x40,two}}};
        for(const auto [mode,expected]:modes) for(unsigned wave:{32u,64u})
            run({prefix+"_output_width_"+std::to_string(mode),Path::ScalarF16,one,1,false,false,false,mode,expected,op},wave,dir);
        for(uint8_t mode:{uint8_t(0),uint8_t(0x20),uint8_t(0x80)}) for(unsigned wave:{32u,64u})
            run({prefix+"_ieee1_"+std::to_string(mode),Path::ScalarF16,one,1,false,true,false,mode,one,op},wave,dir);
        for(unsigned wave:{32u,64u}) {
            run({prefix+"_clamp_ieee1",Path::ScalarF16,two,0,true,true,false,0,one,op},wave,dir);
            run({prefix+"_half_before_clamp",Path::ScalarF16,one_half,3,true,false,false,0,three_quarters,op},wave,dir);
        }
    }
    // Identical negative zeros avoid min/max's unspecified mixed-zero tie.
    // No FMA cancellation sign or half-denorm/rounding boundary is invented here.
    for(uint32_t op:{0x351u,0x354u,0x357u}) for(unsigned wave:{32u,64u}) {
        const std::string prefix="scalar_f16_negative_zero_"+std::to_string(op);
        run({prefix+"_active",Path::ScalarF16,0x80000000u,1,false,false,false,0,0,op},wave,dir);
        run({prefix+"_ieee_disables",Path::ScalarF16,0x80000000u,1,false,true,false,0,0x80000000u,op},wave,dir);
        run({prefix+"_output_preserve_disables",Path::ScalarF16,0x80000000u,1,false,false,false,0x80,0x80000000u,op},wave,dir);
    }
    // Only raw moves: no guessed CLAMP(-0) or OMOD-zero rounding rule is asserted.
    for(bool ieee:{false,true}) for(bool dx:{false,true}) for(uint32_t zero:{0u,0x80000000u}) for(unsigned wave:{32u,64u})
        run({"move_zero_"+std::to_string(ieee)+"_"+std::to_string(dx)+"_"+std::to_string(zero),Path::Move,zero,0,false,ieee,dx,0,zero},wave,dir);
    oracle_controls(seed,dir);
    check(calls==258 && decoded_packets==258 && numeric_values==1032,"all 258 SOURCE cases and 1032 live MRT0 components checked");
    std::printf("fragment_float_modifiers: %u translator calls, %u decoder witnesses, %u MRT0 values, %u checks, %u failures\n",
        calls,decoded_packets,numeric_values,checks,failures);
    return failures ? 1 : 0;
}
