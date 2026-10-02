// #4066: actual enabled-device transport, not arbitrary NaN payload or Wave64 authority.
// Flat inputs take the provoking vertex value without interpolation:
// https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-interpolation-decorations
// Integer predicates expose classes through finite colors; no shader FP arithmetic acts on
// the transported data. Signaling NaNs may quiet, so their payload bits are never the oracle.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/render_runner.h"
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>

using namespace prosper::gpu;
namespace {
unsigned checks=0,failures=0,draws=0;
void check(bool ok,const char* message) { ++checks; if(!ok) {++failures;std::printf("[FAIL] %s\n",message);} }
void move(std::vector<uint32_t>& words,uint32_t reg,uint32_t bits) {
    words.insert(words.end(),{0x7e0002ffu|(reg<<17),bits});
}
void compare(std::vector<uint32_t>& words,uint32_t opcode,uint32_t left,uint32_t right) {
    const std::array<uint32_t,2> packet{0xd400006au|(opcode<<16),left|(right<<9)};
    const auto decoded=rdna2_decode_one(packet.data(),packet.size());
    check(decoded.fmt==Rdna2Format::VOPC && decoded.opcode==opcode,"actual compare packet decoder control");
    words.insert(words.end(),packet.begin(),packet.end());
}
void tag(std::vector<uint32_t>& words,uint32_t reg) {
    words.insert(words.end(),{0xd5010000u|reg,128u|(242u<<9)|(106u<<18)});
}
void bit_and(std::vector<uint32_t>& words,uint32_t reg,uint32_t mask) {
    const uint32_t packet=(0x1bu<<25)|(reg<<17)|255u; // v_and_b32 vN,literal,v0
    const std::array<uint32_t,2> encoded{packet,mask};
    const auto decoded=rdna2_decode_one(encoded.data(),encoded.size());
    check(decoded.fmt==Rdna2Format::VOP2 && decoded.opcode==0x1b && decoded.dst.value==int(reg) &&
          decoded.src[1].kind==OperandKind::VGPR && decoded.src[1].value==0,"actual raw bit-mask decoder control");
    words.insert(words.end(),encoded.begin(),encoded.end());
}
std::vector<uint32_t> vertex(uint32_t bits) {
    // The finite fullscreen position path is independent of the raw PARAM0 value.
    std::vector<uint32_t> words{0x36020081u,0x2c040081u,0x7e020d01u,0x7e040d02u,
        0x100202f6u,0x100404f6u,0x060202f3u,0x060404f3u,0x7e060280u,0x7e0802f2u,
        0xf80008cfu,0x04030201u};
    move(words,10,bits);
    words.insert(words.end(),{0xf800020fu,0x0403030au,0xbf810000u});
    return words;
}
std::vector<uint32_t> fragment(bool input) {
    std::vector<uint32_t> words;
    if(input) words.push_back(0xc8020002u); // v_interp_mov_f32 v0,p0,attr0.x
    else words={0xf4200500u,0xfa000000u,0x7e000214u}; // runtime Uint32 s20 -> v0
    // Known guest input-preserve mode makes NEQ-zero an exact integer relation, including NaN.
    // The Input arm necessarily crosses F32->U32; the raw arm independently checks source bits.
    compare(words,13,128,256); tag(words,8);
    bit_and(words,1,0x7f800000u); move(words,2,0x7f800000u);
    compare(words,0xc2,257,258); tag(words,9); // exponent==255
    bit_and(words,1,0x007fffffu);
    compare(words,0xc5,257,128); tag(words,10); // fraction!=0
    bit_and(words,1,0x80000000u);
    compare(words,0xc5,257,128); tag(words,11); // sign, not checked for NaNs
    words.insert(words.end(),{0xf800180fu,0x0b0a0908u,0xbf810000u});
    return words;
}
struct Inventory { unsigned bitcasts=0,inputs=0;bool none=false,flat=false,cap=false; };
Inventory inventory(const std::vector<uint32_t>& words) {
    Inventory result;
    std::set<uint32_t> f32,i32,vectors,decorated;
    std::map<uint32_t,uint32_t> values,pointers;
    for(size_t at=5;at<words.size();) {
        const uint32_t n=words[at]>>16,op=words[at]&65535u;
        if(!n || n>words.size()-at) return {};
        if(op==17 && n==2 && words[at+1]==6029) result.cap=true;
        if(op==22 && n==3 && words[at+2]==32) f32.insert(words[at+1]);
        if(op==21 && n==4 && words[at+2]==32) i32.insert(words[at+1]);
        if(op==23 && n==4 && f32.contains(words[at+2])) vectors.insert(words[at+1]);
        if(op==32 && n==4) pointers[words[at+1]]=words[at+2];
        if(op==71 && n==3 && words[at+2]==14) result.flat=true;
        if(op==71 && n==4 && words[at+2]==40 && words[at+3]==0) decorated.insert(words[at+1]);
        if(n>=3 && (op==59 || op==61 || op==81 || op==124 || op==43 || op==169 || op==199))
            values[words[at+2]]=words[at+1];
        at+=n;
    }
    result.none=true;
    for(size_t at=5;at<words.size();at+=words[at]>>16) {
        const uint32_t n=words[at]>>16,op=words[at]&65535u;
        if(op==124 && n==4 && ((f32.contains(words[at+1]) && i32.contains(values[words[at+3]])) ||
           (i32.contains(words[at+1]) && f32.contains(values[words[at+3]])))) {
            ++result.bitcasts;result.none &= decorated.contains(words[at+2]);
        }
        if(op==61 && n==4 && (f32.contains(words[at+1]) || vectors.contains(words[at+1])) &&
           pointers[values[words[at+3]]]==1) {
            ++result.inputs;result.none &= decorated.contains(words[at+2]);
        }
    }
    return result;
}
void dump(const char* directory,const std::string& name,const std::vector<uint32_t>& words) {
    if(!directory) return;
    std::ofstream file(std::filesystem::path(directory)/(name+".spv"),std::ios::binary);
    const auto bytes=std::as_bytes(std::span(words));
    file.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));file.close();
    check(bool(file),"SOURCE dump closes successfully");
}
}
int main(int argc,char** argv) {
    const char* directory=nullptr;
    const bool query_only=argc==2 && std::string_view(argv[1])=="--device-query-only";
    if(argc==3 && std::string_view(argv[1])=="--dump-directory") {directory=argv[2];std::filesystem::create_directories(directory);}
    else if(argc!=1 && !query_only) return 2;
    const auto& device=prosper::test::render_vk_ctx(); // query AND request AND successful creation first
    if(!device.ok) {std::fprintf(stderr,"float transport render: Vulkan device unavailable\n");return 1;}
    if(!device.float_transport.explicit_nonfinite32()) {
        std::fprintf(stderr,"float transport render: enabled FloatControls2+SZI32 profile unavailable; no nonfinite pixel claim\n");return 77;
    }
    const auto shared=shared_vulkan_context();
    check(shared.valid() && shared.float_transport==device.float_transport,"shared/adopted device retains actual enabled profile");
    // These independent real physical-device queries inspect REQUEST bytes only. They do not
    // publish a witness, create a second device, or submit an invalid-feature shader.
    for(bool request:{false,true}) {
        VkPhysicalDeviceFeatures2 prior{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};create.pNext=&prior;
        VkPhysicalDeviceShaderFloatControls2Features requested{};
        const auto choice=prosper::frontend::acquire_float_transport_device_features(
            "transport-request-control",device.phys,prosper::frontend::kVulkanRuntimeVersion,
            requested,create,request);
        check(request?(choice.explicit_nonfinite32() && requested.shaderFloatControls2==VK_TRUE &&
                      create.pNext==&requested && requested.pNext==&prior):
                     (choice.profile==FloatTransportProfile::Implicit && requested.shaderFloatControls2==VK_FALSE &&
                      create.pNext==&prior),"actual query/request control retains prior pNext and distinguishes supported from enabled request");
    }
    if(query_only) {
        std::printf("float transport device-query: %u checks, %u failures; no shader/pipeline/submission\n",checks,failures);
        return failures?1:0;
    }
    if(failures) return 1; // never submit a shader if the enabled-request contract already failed
    constexpr std::array<uint32_t,11> samples{0,0x80000000u,1,0x80000001u,0x007fffffu,
        0x00800000u,0x3f800000u,0x7f800000u,0xff800000u,0x7fc01234u,0xffc01234u};
    PixelInputMapping flat;flat.valid_mask=1;flat.controls[0]=0x400; // actual Flat Location0, no custom GS
    const auto raw=fragment(false),input=fragment(true);
    ShaderResourceTable table;ShaderResource resource;
    resource.cls=ResourceClass::ConstantBuffer;resource.format=DataFormat::Uint32;
    resource.binding=32;resource.sgpr_base=0;resource.size=4;table.resources.push_back(resource);
    const auto flat_layout=fragment_interpolation_layout(input.data(),input.size(),nullptr,&flat);
    check(flat_layout.valid && !flat_layout.requires_geometry,"flat P0 route has no interpolation arithmetic or geometry stage");
    for(bool use_input:{false,true}) {
        const auto& program=use_input?input:raw;
        const auto fs=recompile_fragment(program.data(),program.size(),use_input?nullptr:&table,nullptr,
            UINT32_MAX,use_input?&flat_layout:nullptr,true,{}, {true,16},nullptr,device.float_transport);
        const auto facts=inventory(fs);
        check(!fs.empty() && facts.cap && facts.none && facts.bitcasts &&
              (use_input?(facts.inputs && facts.flat):!facts.inputs),"actual SOURCE typed transport has None and correct flat/raw route");
        check(float_transport_module_supported(fs.data(),fs.size(),device.float_transport) &&
              !float_transport_module_supported(fs.data(),fs.size(),{FloatTransportProfile::Implicit}),
              "actual module capability requires enabled execution profile, not advertised support");
        dump(directory,use_input?"input_fragment":"raw_fragment",fs);
        for(uint32_t bits:samples) {
            const auto vs_raw=vertex(bits);
            const auto vs=recompile_vertex(vs_raw.data(),vs_raw.size(),nullptr,use_input?&flat:nullptr,
                                           false,0,{},device.float_transport);
            if(directory) dump(directory,(use_input?"input_vertex_":"raw_vertex_")+std::to_string(bits),vs);
            std::vector<prosper::test::FrameResource> resources;
            if(!use_input) {prosper::test::FrameResource r;r.set=1;r.binding=32;r.dwords={bits};resources.push_back(std::move(r));}
            const auto pixels=prosper::test::render_triangle_rgba(vs,fs,16,16,nullptr,nullptr,nullptr,nullptr,&resources);
            check(pixels.size()==16u*16u*4u,"live finite color readback exists");
            if(pixels.size()!=16u*16u*4u) continue;
            const uint32_t exponent=(bits>>23)&255u,fraction=bits&0x007fffffu;
            const std::array<uint8_t,4> expected{uint8_t((bits&0x7fffffffu)?255:0),
                uint8_t(exponent==255?255:0),uint8_t(fraction?255:0),uint8_t(bits>>31?255:0)};
            const bool nan=exponent==255 && fraction;
            bool correct=true;
            size_t first_bad=SIZE_MAX;
            unsigned first_channel=0,bad_pixels=0;
            for(size_t i=0;i<pixels.size();i+=4) {
                bool pixel_correct=true;
                for(unsigned c=0;c<(nan?3u:4u);++c) if(pixels[i+c]!=expected[c]) {
                    if(first_bad==SIZE_MAX) {first_bad=i;first_channel=c;}
                    pixel_correct=false;
                }
                if(!pixel_correct) {correct=false;++bad_pixels;}
            }
            if(!correct) {
                std::printf("  transport mismatch: route=%s bits=0x%08x pixel=%zu,%zu channel=%u "
                            "actual=%u,%u,%u,%u expected=%u,%u,%u,%u bad-pixels=%u/256\n",
                    use_input?"flat-input":"raw-u32",bits,(first_bad/4)%16,(first_bad/4)/16,
                    first_channel,unsigned(pixels[first_bad]),unsigned(pixels[first_bad+1]),
                    unsigned(pixels[first_bad+2]),unsigned(pixels[first_bad+3]),
                    unsigned(expected[0]),unsigned(expected[1]),unsigned(expected[2]),
                    unsigned(expected[3]),bad_pixels);
            }
            check(correct,"actual raw/Input class and signed-zero colors match independent integer oracle");++draws;
        }
    }
    std::printf("float transport render: %u checks, %u actual draws, %u failures (categories only; no NaN payload identity)\n",checks,draws,failures);
    return failures?1:0;
}
