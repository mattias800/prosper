#include "rdna2_gta5_compute_contracts.hpp"

#include "agc_shader_layout.hpp"
#include "rdna2_decode.hpp"
#include "rdna2_to_spirv.hpp"
#include "shader_resources.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace prosper::gpu {

// Implemented by the guest-memory registry. Live markers retain a guest table address; replay
// markers instead carry owned host_data and never call this path.
bool guest_readable(uint64_t address, uint32_t bytes);

namespace {

enum class NullableProgram : uint8_t { None, WorkgroupStore, WorkgroupProcess };
enum class SelectedSbufferDomain : uint8_t { Reject, AllOob, Record4 };

constexpr uint32_t kSelectedSbufferSourcePc = 70u;
constexpr uint32_t kSelectedSbufferOuterPc = 153u;
constexpr uint32_t kSelectedSbufferLoadX4Pc = 156u;
constexpr uint32_t kSelectedSbufferLoadX2Pc = 158u;
constexpr uint32_t kSelectedSbufferSourceRecords = kGtaSelectedSbufferThreads;
constexpr uint32_t kSelectedSbufferSourceStride = 8u;
constexpr uint32_t kSelectedSbufferOuterStride = 120u;
constexpr uint32_t kSelectedSbufferOuterRecords = 5u;
constexpr uint32_t kSelectedSbufferLoadBytes = 4u * sizeof(uint32_t);

// Exact consumed words from routed GTA V PPSA04263 kernel 0x413ce6000. Full comparison is the
// static authority boundary: the specialization depends on the complete scalar/EXEC path that
// carries v5>>3 through readfirstlane and s_mul_i32 into pc153/156/158.
constexpr std::array<uint32_t, 276> kSelectedSbufferProgram = {
0xbfa00003u, 0xd746000bu, 0x04010c0bu, 0x8f6a820au,
0xf4080400u, 0xfa000070u, 0xf40c0800u, 0xfa000040u,
0xbf8cc07fu, 0x81161211u, 0x8815ff27u, 0x00040000u,
0xbe940326u, 0xbe9703ffu, 0x00016204u, 0x8819ff23u,
0x000c0000u, 0xbe980322u, 0xbe9a0311u, 0xbe9b03ffu,
0x00016204u, 0xbea60311u, 0xbea703ffu, 0x10016204u,
0xbea20312u, 0xbea303ffu, 0x00016204u, 0x84118212u,
0xbf076a80u, 0xbe9c0302u, 0xbe9d0303u, 0xbe9e0304u,
0xbe9f0305u, 0xbea51d96u, 0xbea11d94u, 0xbf84001au,
0xe0302008u, 0x8008060bu, 0x92009788u, 0x7e0602ffu,
0xff800000u, 0x7e0802ffu, 0xff800000u, 0x7e0a02ffu,
0xff800000u, 0xbf8c3f70u, 0xe03c2000u, 0x80070006u,
0xbf8c3f70u, 0x7c100100u, 0x021800f9u, 0x86060600u,
0x021a00f9u, 0x86060601u, 0x021c00f9u, 0x86060602u,
0xbeea376au, 0xbf880002u, 0xe03c200cu, 0x80070306u,
0xbefe046au, 0xbf820076u, 0xf4040100u, 0xfa000098u,
0xbe860312u, 0xbe8703ffu, 0x00016204u, 0xbf8cc07fu,
0xbe851d93u, 0xbe82047eu, 0xe0342000u, 0x8001050bu,
0xbf8c3f70u, 0x36000a87u, 0x2c0e0a83u, 0x7daa0087u,
0xbf88003eu, 0x26000082u, 0xd54f0006u, 0x0415fe80u,
0x3024240cu, 0xd54f0008u, 0x0415fe80u, 0x00301818u,
0x81ea1380u, 0xf4080100u, 0xfa0000b8u, 0x34000082u,
0xd76d0003u, 0x041d886au, 0xbeea03ffu, 0x7f7fffffu,
0x7e1802ffu, 0x7f800000u, 0x7e1a02ffu, 0x7f800000u,
0xd7460004u, 0x04010300u, 0x7e1c02ffu, 0x7f800000u,
0xbf8cc07fu, 0xe03c3000u, 0x80010003u, 0x36080cffu,
0x000000ffu, 0xe03c3000u, 0x80010503u, 0x360810ffu,
0x000000ffu, 0xe03c3000u, 0x80010803u, 0xbf8c3f70u,
0xd554000fu, 0x04160108u, 0x7c061ef9u, 0x0686846au,
0xbeea3704u, 0xbf880006u, 0xd551000cu, 0x04160108u,
0xd551000du, 0x041a0309u, 0xd551000eu, 0x041e050au,
0xbefe046au, 0x7e0602ffu, 0xff800000u, 0x7e0802ffu,
0xff800000u, 0x7e0a02ffu, 0xff800000u, 0xbeea3704u,
0xbf880005u, 0xd5540004u, 0x041a0309u, 0xd5540005u,
0x041e050au, 0x7e06030fu, 0xbefe046au, 0x8afe7e02u,
0xbf880026u, 0x7e100281u, 0x7daa1080u, 0xbf880023u,
0x7ed40507u, 0xbe92047eu, 0x7da40e6au, 0xbf88001du,
0x7e100280u, 0xb86a0078u, 0xf4080100u, 0xfa0000a8u,
0xbf8cc07fu, 0xf4280202u, 0xd4000008u, 0xbf8cc07fu,
0xe0382000u, 0x80020006u, 0xe0342010u, 0x80020406u,
0x920c9788u, 0x920d9789u, 0xbf8c3f71u, 0x7c100100u,
0x021818f9u, 0x86060600u, 0x021a18f9u, 0x86060601u,
0x021c18f9u, 0x86060602u, 0x02061af9u, 0x86060603u,
0xbf8c3f70u, 0x02081af9u, 0x86060604u, 0x020a1af9u,
0x86060605u, 0xbefe0412u, 0xbf82ffdbu, 0xbefe0402u,
0xd76d0000u, 0x042d8211u, 0xbf8c3f70u, 0x7c0218f9u,
0x06068003u, 0x7c021b04u, 0x7c021cf9u, 0x06068205u,
0x92049788u, 0xe0302000u, 0x80050600u, 0x92059789u,
0x88ea6a00u, 0x88ea026au, 0x020008f9u, 0x8606060cu,
0x020208f9u, 0x8606060du, 0x020408f9u, 0x8606060eu,
0x02060af9u, 0x86060603u, 0x02100af9u, 0x86060604u,
0x02120af9u, 0x86060605u, 0xbf8c3f70u, 0xd548000au,
0x026d0706u, 0xbf880041u, 0x2c180c9eu, 0x16161898u,
0xe0787010u, 0x8009000au, 0xe0747020u, 0x8009080au,
0xbbfd0000u, 0x7e080281u, 0xe0c86008u, 0x8006040au,
0xbf8c3f70u, 0x3608088fu, 0x7daa0880u, 0xbf880033u,
0xe0342000u, 0x8006040au, 0x7e0c02c1u, 0x7e0e02c1u,
0x4c161880u, 0x34161683u, 0xd746000bu, 0x042d030bu,
0xbf8c3f70u, 0x301c0881u, 0x30180a81u, 0xd5490004u,
0x02050104u, 0xd5490005u, 0x02050105u, 0xd746000fu,
0x0281070eu, 0xd746000du, 0x0281070cu, 0xd747000eu,
0x020e1c10u, 0xd747000cu, 0x020e1810u, 0x381e1e85u,
0x381a1a85u, 0xd54a0004u, 0x043e1d04u, 0xd54a0005u,
0x04361905u, 0xe0786000u, 0x8009040au, 0xe038f028u,
0x8009040au, 0xe034f038u, 0x80090b0au, 0xbf8c3f71u,
0x1e000104u, 0x1e020b01u, 0x1e040d02u, 0x20060707u,
0xbf8c3f70u, 0x20101708u, 0x20121909u, 0x7daa1480u,
0xbf880006u, 0xe0302000u, 0x8005060au, 0xbf8c3f70u,
0xd548000au, 0x026d0706u, 0xbf82ffbeu, 0xbf810000u,
};

// Exact consumed words from routed GTA V PPSA04263 kernels 0x413e19200 and 0x413e1ac00. Full byte
// comparison is intentional: a write-elision proof must not inherit the scalar fold's linear-walk
// assumptions across an edited CFG. The arrays are data, not another instruction implementation.
constexpr std::array<uint32_t, 103> kWorkgroupStoreProgram = {
    0xbfa00003u, 0x938aff08u, 0x00040018u, 0xbeea0380u,
    0xbf0a0a6au, 0xbf840008u, 0x34040082u, 0x7e020280u,
    0xd7460002u, 0x0409146au, 0x816a816au, 0xd8340000u,
    0x00000102u, 0xbf82fff6u, 0xd7460001u, 0x04011009u,
    0xbf8cc07fu, 0xbf8a0000u, 0x7da8020au, 0xbf880006u,
    0x7e040280u, 0xf4080300u, 0xfa000010u, 0xbf8cc07fu,
    0xe0702000u, 0x80030201u, 0xbefe04c1u, 0xf4040300u,
    0xfa000020u, 0x7e060280u, 0x34040082u, 0x7e020209u,
    0xbe8e0307u, 0xbe8f03ffu, 0x00016204u, 0xbf8cc07fu,
    0xbe8d1d9au, 0x996a8008u, 0xe0703000u, 0x80030301u,
    0x8805ff03u, 0x00100000u, 0xbe840302u, 0xbe8703ffu,
    0x00016204u, 0x9002886au, 0x93036a09u, 0x9388ff08u,
    0x00080010u, 0xbe890380u, 0xbf0a0209u, 0xbf84001eu,
    0xbf8c3f70u, 0x7e0402c1u, 0x8f6a8809u, 0x7e0202c1u,
    0xd76d0003u, 0x0400d403u, 0x7da80606u, 0xbf880002u,
    0xe0342000u, 0x80010103u, 0xbe8b0380u, 0xbefe04c1u,
    0xbf0a0a0bu, 0xbeeb030bu, 0xbf84000du, 0x7e0a0281u,
    0x986a080bu, 0x810b810bu, 0xbf8c3f70u, 0xd7000003u,
    0x0002026au, 0x340606f9u, 0x00860682u, 0xd7460003u,
    0x040d146bu, 0xd8000000u, 0x00000503u, 0xbf82fff0u,
    0x81098109u, 0xbf82ffe0u, 0xf4080100u, 0xfa000000u,
    0xbf8cc07fu, 0xbf8a0000u, 0xbeea0380u, 0xbf0a0a6au,
    0xbf84000du, 0xbf8c3f70u, 0x34040082u, 0xd7460001u,
    0x0401106au, 0xd7460002u, 0x0409146au, 0x816a816au,
    0xd8d80000u, 0x02000002u, 0xbf8cc07fu, 0xe0c82000u,
    0x80010201u, 0xbf82fff1u, 0xbf810000u,
};

constexpr std::array<uint32_t, 282> kWorkgroupProcessProgram = {
    0xbfa00003u, 0x7d8a0080u, 0x90109c08u, 0xbeea086au,
    0xbefe046au, 0xbf88000bu, 0x7e020210u, 0x7e040281u,
    0xf4080300u, 0xfa000010u, 0x7e060280u, 0xbf8cc07fu,
    0xe0c86000u, 0x80030201u, 0xbf8c3f70u, 0xd8340000u,
    0x00000203u, 0xbefe04c1u, 0x93eaff08u, 0x00080010u,
    0x34060083u, 0x880dff03u, 0x00100000u, 0xbe8c0302u,
    0xbe8e0306u, 0xbe8f03ffu, 0x00016204u, 0x98026a10u,
    0xbf8cc07fu, 0xbf8a0000u, 0x7e080280u, 0x996a8008u,
    0x7e020280u, 0x7e040280u, 0x2c180086u, 0xd8d80000u,
    0x09000004u, 0x9011886au, 0xd9340400u, 0x00000103u,
    0xbf8cc07fu, 0xbf8a0000u, 0xd5690003u, 0x0002126au,
    0xbf0a1180u, 0x4a020700u, 0x7d880206u, 0xbf840018u,
    0x7e0e02c1u, 0x7e1002c1u, 0x7e0c02c1u, 0x7e0a02c1u,
    0xbefe046au, 0xbf880002u, 0xe0382000u, 0x80030501u,
    0xbefe04c1u, 0x34081883u, 0xbf8c3f70u, 0xd7000001u,
    0x00020a02u, 0x341502f9u, 0x86060604u, 0x340402f9u,
    0x00860683u, 0x361e02ffu, 0x000000ffu, 0xd8800400u,
    0x01000a02u, 0xbf8cc07fu, 0xd5480010u, 0x02220901u,
    0xd76d000au, 0x040e00ffu, 0x00000100u, 0xbf0a1181u,
    0x7d881406u, 0xbf840018u, 0x7e0602c1u, 0x7e0802c1u,
    0x7e0402c1u, 0x7e0202c1u, 0xbefe046au, 0xbf880002u,
    0xe0382000u, 0x8003010au, 0xbefe04c1u, 0x341c1883u,
    0xbf8c3f70u, 0xd700000au, 0x00020202u, 0x342302f9u,
    0x8606060eu, 0x341614f9u, 0x00860683u, 0x361a14ffu,
    0x000000ffu, 0xd8800404u, 0x0a00110bu, 0xbf8cc07fu,
    0xd548000eu, 0x02221d0au, 0xf4040200u, 0xfa000020u,
    0xbf8cc07fu, 0x880dff09u, 0x04000000u, 0xbe8a0307u,
    0xbe89030du, 0xbe8b03ffu, 0x00016204u, 0xbf8a0000u,
    0x34140083u, 0x7d8212f9u, 0x06868280u, 0xbe8c0308u,
    0xbe8e0307u, 0x87078110u, 0xd9d80400u, 0x0a00000au,
    0x856a8084u, 0xbf8cc07fu, 0x362214ffu, 0x000000ffu,
    0x2c241498u, 0x2c261698u, 0xbe8f03ffu, 0x00016204u,
    0xd76d0011u, 0x04462712u, 0xd5480012u, 0x0221110au,
    0xd548000au, 0x0221210au, 0xd76d000au, 0x042a2511u,
    0x362216ffu, 0x000000ffu, 0xd5480012u, 0x0221110bu,
    0xd76d000au, 0x044a230au, 0x7e220280u, 0x4a2416f9u,
    0x0206060au, 0xd501000bu, 0x00090282u, 0x34140082u,
    0x3816166au, 0x8aea027eu, 0xd76f000bu, 0x04493b0bu,
    0xe0707000u, 0x80030b09u, 0xbf870028u, 0x4a1412c1u,
    0x7e220280u, 0xbf078007u, 0xbe82047eu, 0x850c8480u,
    0x7e160280u, 0xbeea047eu, 0x2c26169du, 0x7dac2684u,
    0xbf88000cu, 0x34160082u, 0xe030f000u, 0x8002130au,
    0xbf8c3f70u, 0x361626ffu, 0x1fffffffu, 0x2c26269du,
    0x3a26260cu, 0xd76f000bu, 0x042d3b13u, 0xbbfd0000u,
    0xbf82fff1u, 0xbefe046au, 0x34261682u, 0x361616ffu,
    0x1fffffffu, 0x4a1414c1u, 0x4a221711u, 0x7d282680u,
    0xbf89ffe7u, 0xbefe0402u, 0xbf078007u, 0x4a162312u,
    0x856af4f5u, 0x34140082u, 0xd771000bu, 0x01aa16ffu,
    0x1fffffffu, 0xe0707000u, 0x80020b09u, 0xd7460009u,
    0x04011010u, 0xf4080200u, 0xfa000000u, 0x34000082u,
    0xbe8703ffu, 0x00016204u, 0xbe851d94u, 0xbf8cc07fu,
    0xe0302000u, 0x80020909u, 0xbf8c3f70u, 0x4a122309u,
    0xd8340000u, 0x00000900u, 0xbf8cc07fu, 0xbf8a0000u,
    0xbf0a1180u, 0xbf84001bu, 0x7e000280u, 0xbeeb0380u,
    0xbe80047eu, 0x7da2186bu, 0xbf88000au, 0x34121e83u,
    0x8f6a836bu, 0x816b816bu, 0xd8d80400u, 0x09000009u,
    0xbf8cc07fu, 0xd5480009u, 0x0220d509u, 0x4a001300u,
    0xbf82fff4u, 0xbefe0400u, 0x34121e82u, 0xd8d80000u,
    0x09000009u, 0xbf8cc07fu, 0xd76d0000u, 0x04022109u,
    0x7da80006u, 0xbf880002u, 0xe0782000u, 0x80010500u,
    0xbefe04c1u, 0xbf0a1181u, 0xbf840026u, 0x34001a83u,
    0xbeeb0380u, 0xbe80047eu, 0xd8d80400u, 0x00000000u,
    0xbf8cc07fu, 0x360a00ffu, 0x000000ffu, 0x2c0c0098u,
    0xd5480007u, 0x02211100u, 0xd76d0005u, 0x041e0b06u,
    0x4a0000f9u, 0x02060605u, 0x7da2186bu, 0xbf88000au,
    0x340a1a83u, 0x8f6a836bu, 0x816b816bu, 0xd8d80404u,
    0x05000005u, 0xbf8cc07fu, 0xd5480005u, 0x0220d505u,
    0x4a000b00u, 0xbf82fff4u, 0xbefe0400u, 0x340a1a82u,
    0xd8d80000u, 0x05000005u, 0xbf8cc07fu, 0xd76d0000u,
    0x04021d05u, 0x7da80006u, 0xbf880002u, 0xe0782000u,
    0x80010100u, 0xbf810000u,
};

template <size_t N>
bool exact_program(const uint32_t* code, size_t dwords,
                   const std::array<uint32_t, N>& expected) {
    if (!code || dwords < N || !std::equal(expected.begin(), expected.end(), code))
        return false;
    std::vector<Rdna2Inst> instructions;
    const size_t consumed = rdna2_walk(code, dwords, instructions);
    return consumed == N && !instructions.empty() && instructions.back().is_end;
}

NullableProgram nullable_program(const uint32_t* code, size_t dwords) {
    if (exact_program(code, dwords, kWorkgroupStoreProgram))
        return NullableProgram::WorkgroupStore;
    if (exact_program(code, dwords, kWorkgroupProcessProgram))
        return NullableProgram::WorkgroupProcess;
    return NullableProgram::None;
}

bool raw_dword_packet(const Rdna2Inst& in, uint32_t opcode, uint32_t vdata,
                      uint32_t vaddr, uint32_t srsrc,
                      uint32_t word0, uint32_t word1) {
    return in.fmt == Rdna2Format::MUBUF && in.opcode == opcode && in.len_dwords == 2u &&
           in.dst.kind == OperandKind::VGPR && in.dst.value == static_cast<int32_t>(vdata) &&
           in.src[0].kind == OperandKind::VGPR &&
           in.src[0].value == static_cast<int32_t>(vaddr) &&
           in.src[1].kind == OperandKind::SGPR &&
           in.src[1].value == static_cast<int32_t>(srsrc) &&
           in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0 &&
           // The raw words intentionally remain authoritative for GLC/DLC/LDS and every other
           // packet bit: the long kernel sets GLC on all three sites and DLC on its zero-load.
           in.words[0] == word0 && in.words[1] == word1;
}

bool witness_is_zero(const ShaderResource& marker) {
    const uint8_t* bytes = marker.host_data;
    if (bytes) {
        if (marker.host_data_size < kGtaNullableOutputWitnessBytes) return false;
    } else {
        if (!guest_readable(marker.gpu_addr, kGtaNullableOutputWitnessBytes)) return false;
        bytes = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(marker.gpu_addr));
    }
    uint64_t pointer = UINT64_MAX;
    std::memcpy(&pointer, bytes + kGtaNullableOutputPointerOffset, sizeof(pointer));
    return pointer == 0u;
}

const uint8_t* complete_resource_bytes(const ShaderResource& resource, uint32_t bytes) {
    if (resource.host_data)
        return resource.host_data_size >= bytes ? resource.host_data : nullptr;
    if (!resource.gpu_addr || !guest_readable(resource.gpu_addr, bytes)) return nullptr;
    return reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(resource.gpu_addr));
}

bool selected_sbuffer_source_shape(const ShaderResource& resource) {
    return resource.cls == ResourceClass::ConstantBuffer &&
           resource.fetch_pc == kSelectedSbufferSourcePc &&
           resource.gpu_addr > 0x10000u &&
           resource.stride == kSelectedSbufferSourceStride &&
           resource.size == kSelectedSbufferSourceRecords * kSelectedSbufferSourceStride;
}

bool selected_sbuffer_outer_shape(const ShaderResource& resource) {
    return resource.cls == ResourceClass::ConstantBuffer &&
           resource.fetch_pc == kSelectedSbufferOuterPc &&
           resource.gpu_addr > 0x10000u &&
           resource.stride == kSelectedSbufferOuterStride &&
           resource.size == kSelectedSbufferOuterRecords * kSelectedSbufferOuterStride;
}

bool selected_sbuffer_target_descriptor(const std::array<uint32_t, 4>& words,
                                        DecodedBufferDescriptor& decoded) {
    decoded = decode_buffer_descriptor(words.data());
    return decoded.base > 0x10000u && decoded.stride == 20u &&
           decoded.num_records == 668u && decoded.size_bytes == 13360u &&
           decoded.format == DataFormat::Float32 && decoded.num_components == 3u &&
           !decoded.forbid_unknown_fallback;
}

bool selected_sbuffer_target_resource(const ShaderResource& resource,
                                      uint32_t pc,
                                      const DecodedBufferDescriptor& decoded) {
    return resource.cls == ResourceClass::ConstantBuffer && resource.fetch_pc == pc &&
           resource.gpu_addr == decoded.base && resource.size == decoded.size_bytes &&
           resource.stride == decoded.stride && resource.format == decoded.format &&
           resource.num_components == decoded.num_components &&
           resource.table_index_count == 0u;
}

SelectedSbufferDomain selected_sbuffer_domain(const ShaderResource& source) {
    const uint8_t* bytes = complete_resource_bytes(source, source.size);
    if (!bytes) return SelectedSbufferDomain::Reject;
    bool uses_record4 = false;
    for (uint32_t record = 0; record < kSelectedSbufferSourceRecords; ++record) {
        uint32_t first = 0;
        std::memcpy(&first, bytes + record * kSelectedSbufferSourceStride, sizeof(first));
        const uint32_t selector = first >> 3u;
        const uint32_t soffset = selector * kSelectedSbufferOuterStride;
        if (soffset == kGtaSelectedSbufferRecord4Soffset) {
            uses_record4 = true;
            continue;
        }
        // pc153 adds eight and reads four consecutive dwords. Partial in-bounds reads would need
        // per-component table semantics, so admit only a wholly OOB access. Mirror the hardware's
        // uint32 address addition before testing the bound: a large SOFFSET can wrap back into the
        // five-record table and must not be mistaken for OOB merely because its pre-wrap value is big.
        const uint32_t first_byte = soffset + 8u;
        if (first_byte < kSelectedSbufferOuterRecords * kSelectedSbufferOuterStride ||
            first_byte > UINT32_MAX - (kSelectedSbufferLoadBytes - 1u))
            return SelectedSbufferDomain::Reject;
    }
    return uses_record4 ? SelectedSbufferDomain::Record4
                        : SelectedSbufferDomain::AllOob;
}

template <typename Table>
auto exact_fetch_resource(Table& resources, uint32_t pc) {
    using ResourcePtr = decltype(&resources.resources.front());
    ResourcePtr found = nullptr;
    for (auto& resource : resources.resources) {
        if (resource.fetch_pc != pc) continue;
        if (found) return ResourcePtr{};
        found = &resource;
    }
    return found;
}

} // namespace

bool rdna2_gta5_null_guarded_raw_store_site(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::MUBUF || in.len_dwords != 2u ||
        in.src[0].kind != OperandKind::VGPR || in.src[0].value != 0 ||
        in.src[1].kind != OperandKind::SGPR || in.src[1].value != 0 ||
        in.src[2].kind != OperandKind::InlineInt || in.src[2].value != 0)
        return false;
    switch (in.pc) {
        case 74u:
            return in.opcode == kMubufOpcodeStoreDwordX2 &&
                   in.words[0] == 0xe0740030u && in.words[1] == 0x80000700u;
        case 76u:
            return in.opcode == kMubufOpcodeStoreDwordX4 &&
                   in.words[0] == 0xe0780020u && in.words[1] == 0x80000000u;
        case 78u:
            return in.opcode == kMubufOpcodeStoreDwordX3 &&
                   in.words[0] == 0xe07c0000u && in.words[1] == 0x80000400u;
        default:
            return false;
    }
}

bool rdna2_gta5_null_guarded_raw_store_shader(const uint32_t* code, size_t dwords) {
    if (!code || dwords <= 80u || code[42] != 0xbf128002u ||
        code[45] != 0x85ea8012u || code[46] != 0x8dea006au ||
        code[47] != 0x87fe126au || code[48] != 0xbf88001fu ||
        code[80] != 0xbf810000u)
        return false;

    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    for (const Rdna2Inst& in : instructions) {
        if (in.fmt == Rdna2Format::SOP1 &&
            (in.opcode == kSop1OpcodeSetpcB64 ||
             in.opcode == kSop1OpcodeSwappcB64 ||
             in.opcode == kSop1OpcodeRfeB64))
            return false;
        if (in.fmt == Rdna2Format::SOPK &&
            (in.opcode == kSopkOpcodeCallB64 ||
             in.opcode == kSopkOpcodeSubvectorLoopBegin ||
             in.opcode == kSopkOpcodeSubvectorLoopEnd))
            return false;
        if (in.fmt == Rdna2Format::SOPP && in.opcode == kSoppOpcodeTrap)
            return false;
    }
    auto at = [&](uint32_t pc) -> const Rdna2Inst* {
        const auto found = std::find_if(instructions.begin(), instructions.end(),
                                        [&](const Rdna2Inst& in) { return in.pc == pc; });
        return found == instructions.end() ? nullptr : &*found;
    };
    const Rdna2Inst* compare = at(42u);
    const Rdna2Inst* mask = at(43u);
    const Rdna2Inst* select = at(45u);
    const Rdna2Inst* nor_mask = at(46u);
    const Rdna2Inst* empty_exec = at(47u);
    const Rdna2Inst* exit = at(48u);
    const Rdna2Inst* merge = at(80u);
    if (!compare || !mask || !select || !nor_mask || !empty_exec || !exit || !merge)
        return false;
    auto branch_target = [](const Rdna2Inst& in) {
        return static_cast<int64_t>(in.pc) + static_cast<int64_t>(in.len_dwords) +
               static_cast<int64_t>(in.simm16);
    };
    if (compare->fmt != Rdna2Format::SOPC || compare->opcode != kSopcOpcodeCmpEqU64 ||
        compare->src[0].kind != OperandKind::SGPR || compare->src[0].value != 2 ||
        compare->src[1].kind != OperandKind::InlineInt || compare->src[1].value != 0 ||
        compare->pc + compare->len_dwords != mask->pc ||
        mask->fmt != Rdna2Format::VOPC || mask->dst.kind != OperandKind::SGPR ||
        mask->dst.value != 0 || mask->pc + mask->len_dwords != select->pc ||
        select->fmt != Rdna2Format::SOP2 ||
        select->opcode != kSop2OpcodeCselectB64 ||
        select->dst.kind != OperandKind::SGPR || select->dst.value != 106 ||
        select->src[0].kind != OperandKind::SGPR || select->src[0].value != 18 ||
        select->src[1].kind != OperandKind::InlineInt || select->src[1].value != 0 ||
        select->pc + select->len_dwords != nor_mask->pc ||
        nor_mask->fmt != Rdna2Format::SOP2 || nor_mask->opcode != kSop2OpcodeNorB64 ||
        nor_mask->dst.kind != OperandKind::SGPR || nor_mask->dst.value != 106 ||
        nor_mask->src[0].kind != OperandKind::Special || nor_mask->src[0].value != 106 ||
        nor_mask->src[1].kind != OperandKind::SGPR || nor_mask->src[1].value != 0 ||
        nor_mask->pc + nor_mask->len_dwords != empty_exec->pc ||
        empty_exec->fmt != Rdna2Format::SOP2 ||
        empty_exec->opcode != kSop2OpcodeAndB64 ||
        empty_exec->dst.kind != OperandKind::SGPR || empty_exec->dst.value != 126 ||
        empty_exec->src[0].kind != OperandKind::Special || empty_exec->src[0].value != 106 ||
        empty_exec->src[1].kind != OperandKind::SGPR || empty_exec->src[1].value != 18 ||
        empty_exec->pc + empty_exec->len_dwords != exit->pc ||
        exit->fmt != Rdna2Format::SOPP || exit->opcode != kSoppOpcodeCbranchExecz ||
        branch_target(*exit) != 80 ||
        merge->fmt != Rdna2Format::SOPP || !merge->is_end)
        return false;

    auto is_proof_direct_branch = [](uint32_t opcode) {
        return sopp_opcode_is_direct_branch(opcode) ||
               (opcode >= kSoppOpcodeCbranchCdbgsys &&
                opcode <= kSoppOpcodeCbranchCdbgsysAndUser);
    };
    for (const Rdna2Inst& in : instructions) {
        if (in.fmt != Rdna2Format::SOPP ||
            !is_proof_direct_branch(in.opcode) || &in == exit)
            continue;
        const int64_t target = branch_target(in);
        const bool source_inside = in.pc > compare->pc && in.pc < merge->pc;
        const bool target_inside = target >= static_cast<int64_t>(compare->pc) &&
                                   target < static_cast<int64_t>(merge->pc);
        if (source_inside || (target_inside && !source_inside)) return false;
    }
    return true;
}

bool rdna2_gta5_null_guarded_raw_store_dispatch(
        const uint32_t* code, size_t dwords,
        const uint32_t* user_sgprs, size_t user_sgpr_count) {
    if (!rdna2_gta5_null_guarded_raw_store_shader(code, dwords) ||
        !user_sgprs || user_sgpr_count < 4u ||
        user_sgprs[2] != 0u || user_sgprs[3] != 0u)
        return false;

    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    for (const Rdna2Inst& in : instructions) {
        if (in.pc >= 42u) break;
        const bool dynamic_scalar_destination =
            in.fmt == Rdna2Format::SOP1 &&
            (in.opcode == kSop1OpcodeMovreldB32 ||
             in.opcode == kSop1OpcodeMovreldB64 ||
             in.opcode == kSop1OpcodeMovrelsd2B32);
        if (dynamic_scalar_destination) return false;
        const bool scalar_dst_overlaps =
            ((in.dst.kind == OperandKind::SGPR ||
              in.dst.kind == OperandKind::Special) &&
             in.dst.value >= 1 && in.dst.value <= 3) ||
            ((in.sdst.kind == OperandKind::SGPR ||
              in.sdst.kind == OperandKind::Special) &&
             in.sdst.value >= 1 && in.sdst.value <= 3) ||
            (in.fmt == Rdna2Format::SMEM &&
             (in.dst.kind == OperandKind::SGPR ||
              in.dst.kind == OperandKind::Special) &&
             in.dst.value >= 0 && in.dst.value <= 3);
        if (scalar_dst_overlaps) return false;
    }
    return true;
}

Gta5NullableOutputAccess rdna2_gta5_nullable_output_site(const Rdna2Inst& in) {
    switch (in.pc) {
        case 38u:
            return raw_dword_packet(in, kMubufOpcodeStoreDword, 3u, 1u, 12u,
                                    0xe0703000u, 0x80030301u)
                ? Gta5NullableOutputAccess::StoreDword
                : Gta5NullableOutputAccess::None;
        case 152u:
            return raw_dword_packet(in, kMubufOpcodeStoreDword, 11u, 9u, 12u,
                                    0xe0707000u, 0x80030b09u)
                ? Gta5NullableOutputAccess::StoreDword
                : Gta5NullableOutputAccess::None;
        case 166u:
            return raw_dword_packet(in, kMubufOpcodeLoadDword, 19u, 10u, 8u,
                                    0xe030f000u, 0x8002130au)
                ? Gta5NullableOutputAccess::LoadDword
                : Gta5NullableOutputAccess::None;
        case 193u:
            return raw_dword_packet(in, kMubufOpcodeStoreDword, 11u, 9u, 8u,
                                    0xe0707000u, 0x80020b09u)
                ? Gta5NullableOutputAccess::StoreDword
                : Gta5NullableOutputAccess::None;
        default:
            return Gta5NullableOutputAccess::None;
    }
}

bool rdna2_gta5_nullable_output_shader(const uint32_t* code, size_t dwords) {
    return nullable_program(code, dwords) != NullableProgram::None;
}

bool rdna2_gta5_nullable_output_launch(
        const uint32_t* code, size_t dwords,
        const uint32_t* user_sgprs, size_t user_sgpr_count,
        uint32_t local_x, uint32_t local_y, uint32_t local_z,
        uint32_t threads_x, uint32_t threads_y, uint32_t threads_z,
        bool /*exact_thread_extent*/, uint32_t wave_size,
        bool tgid_x_en, bool tgid_y_en, bool tgid_z_en,
        uint32_t tidig_comp_cnt) {
    const NullableProgram program = nullable_program(code, dwords);
    const bool user_sgpr8_matches = user_sgprs && user_sgpr_count == 9u &&
        (user_sgprs[8] == kGtaNullableOutputUserSgpr8 ||
         (program == NullableProgram::WorkgroupProcess &&
          (user_sgprs[8] & ~kGtaNullableOutputProcessSelectorMask) ==
              kGtaNullableOutputUserSgpr8));
    if (program == NullableProgram::None || !user_sgprs || user_sgpr_count != 9u ||
        user_sgprs[7] == 0u ||
        user_sgprs[7] > kGtaNullableOutputMaxRecordCount ||
        !user_sgpr8_matches ||
        local_x != kGtaNullableOutputLocalSize || local_y != 1u || local_z != 1u ||
        static_cast<uint64_t>(threads_x) !=
            static_cast<uint64_t>(user_sgprs[7]) * kGtaNullableOutputLocalSize ||
        threads_y != 1u || threads_z != 1u ||
        wave_size != 64u || tgid_y_en || tgid_z_en ||
        tidig_comp_cnt != 0u)
        return false;
    return tgid_x_en == (program == NullableProgram::WorkgroupStore);
}

bool rdna2_gta5_nullable_output_dispatch(
        const uint32_t* code, size_t dwords,
        const ComputeShaderConfig& config,
        const ShaderResourceTable& resources) {
    if (!rdna2_gta5_nullable_output_launch(
            code, dwords, config.user_sgprs.data(), config.user_sgprs.size(),
            config.local_x, config.local_y, config.local_z,
            config.threads_x, config.threads_y, config.threads_z,
            config.exact_thread_extent, config.wave_size,
            config.tgid_x_en, config.tgid_y_en, config.tgid_z_en,
            config.tidig_comp_cnt))
        return false;

    const NullableProgram program = nullable_program(code, dwords);
    const std::array<uint32_t, 3> expected = program == NullableProgram::WorkgroupStore
        ? std::array<uint32_t, 3>{38u, UINT32_MAX, UINT32_MAX}
        : std::array<uint32_t, 3>{152u, 166u, 193u};
    const size_t expected_count = program == NullableProgram::WorkgroupStore ? 1u : 3u;
    const uint64_t table_root = static_cast<uint64_t>(config.user_sgprs[0]) |
                                (static_cast<uint64_t>(config.user_sgprs[1]) << 32u);
    if (table_root <= 0x10000u) return false;

    std::array<bool, 3> found{};
    size_t marker_count = 0;
    for (const ShaderResource& resource : resources.resources) {
        const bool sentinel = is_nullable_raw_buffer_marker_candidate(resource);
        if (!is_proven_null_nullable_raw_buffer(resource)) {
            if (sentinel) return false;
            continue;
        }
        if (resource.gpu_addr != table_root || !witness_is_zero(resource)) return false;
        const auto pc = std::find(expected.begin(), expected.begin() + expected_count,
                                  resource.fetch_pc);
        if (pc == expected.begin() + expected_count) return false;
        const size_t index = static_cast<size_t>(pc - expected.begin());
        if (found[index]) return false;
        found[index] = true;
        ++marker_count;
    }
    return marker_count == expected_count &&
           std::all_of(found.begin(), found.begin() + expected_count,
                       [](bool value) { return value; });
}

bool rdna2_gta5_selected_sbuffer_shader(const uint32_t* code, size_t dwords) {
    return exact_program(code, dwords, kSelectedSbufferProgram);
}

bool rdna2_gta5_selected_sbuffer_launch(
        const uint32_t* code, size_t dwords,
        const ComputeShaderConfig& config) {
    return rdna2_gta5_selected_sbuffer_shader(code, dwords) &&
           config.user_sgprs.size() == 11u &&
           config.local_x == 64u && config.local_y == 1u && config.local_z == 1u &&
           config.exact_thread_extent &&
           config.threads_x == kSelectedSbufferSourceRecords &&
           config.threads_y == 1u && config.threads_z == 1u &&
           config.wave_size == 64u && config.tgid_x_en &&
           !config.tgid_y_en && !config.tgid_z_en && config.tidig_comp_cnt == 0u;
}

bool discover_rdna2_gta5_selected_sbuffer(
        const uint32_t* code, size_t dwords,
        const ComputeShaderConfig& config,
        ShaderResourceTable& resources) {
    // Derived authority never survives a failed refresh. Consumer resources are ordinary bindings
    // and remain inert unless a new marker passes the final dispatch validator.
    for (ShaderResource& resource : resources.resources) {
        resource.selected_sbuffer_soffset = UINT32_MAX;
        resource.selected_sbuffer_words = {};
    }
    if (!rdna2_gta5_selected_sbuffer_launch(code, dwords, config)) {
        if (rdna2_gta5_selected_sbuffer_shader(code, dwords) &&
            config.threads_x == kGtaSelectedSbufferThreads && std::getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[gta-selected-sbuffer] reject=launch user=%zu local=%ux%ux%u "
                         "exact=%d threads=%ux%ux%u wave=%u tgid=%d/%d/%d tidig=%u\n",
                         config.user_sgprs.size(), config.local_x, config.local_y, config.local_z,
                         static_cast<int>(config.exact_thread_extent),
                         config.threads_x, config.threads_y, config.threads_z, config.wave_size,
                         static_cast<int>(config.tgid_x_en), static_cast<int>(config.tgid_y_en),
                         static_cast<int>(config.tgid_z_en), config.tidig_comp_cnt);
        return false;
    }

    ShaderResource* source = exact_fetch_resource(resources, kSelectedSbufferSourcePc);
    ShaderResource* outer = exact_fetch_resource(resources, kSelectedSbufferOuterPc);
    const bool zero_source = source && is_zero_record_raw_buffer(*source);
    const bool zero_outer = outer && is_zero_record_raw_buffer(*outer);
    if (zero_source || zero_outer) {
        if (!zero_source || !zero_outer) return false;
        ShaderResource* target_x4 = exact_fetch_resource(resources, kSelectedSbufferLoadX4Pc);
        ShaderResource* target_x2 = exact_fetch_resource(resources, kSelectedSbufferLoadX2Pc);
        if ((target_x4 && !is_zero_record_raw_buffer(*target_x4)) ||
            (target_x2 && !is_zero_record_raw_buffer(*target_x2)))
            return false;
        auto add_zero_target = [&](uint32_t pc) {
            ShaderResource target;
            target.cls = ResourceClass::ConstantBuffer;
            target.format = DataFormat::Unknown;
            target.num_components = 0u;
            target.fetch_pc = pc;
            resources.resources.push_back(target);
        };
        if (!target_x4) add_zero_target(kSelectedSbufferLoadX4Pc);
        if (!target_x2) add_zero_target(kSelectedSbufferLoadX2Pc);
        outer = exact_fetch_resource(resources, kSelectedSbufferOuterPc);
        if (!outer) return false;
        outer->selected_sbuffer_soffset = kGtaSelectedSbufferZeroChainSoffset;
        return rdna2_gta5_selected_sbuffer_dispatch(code, dwords, config, resources);
    }
    if (!source || !outer || !selected_sbuffer_source_shape(*source) ||
        !selected_sbuffer_outer_shape(*outer)) {
        if (std::getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[gta-selected-sbuffer] reject=resource-shape source=%p outer=%p "
                         "source-shape=%d outer-shape=%d resources=%zu "
                         "source{class=%u addr=0x%llx size=%u stride=%u pc=%u} "
                         "outer{class=%u addr=0x%llx size=%u stride=%u pc=%u}\n",
                         static_cast<void*>(source), static_cast<void*>(outer),
                         source && selected_sbuffer_source_shape(*source),
                         outer && selected_sbuffer_outer_shape(*outer), resources.resources.size(),
                         source ? static_cast<uint32_t>(source->cls) : UINT32_MAX,
                         source ? static_cast<unsigned long long>(source->gpu_addr) : 0ull,
                         source ? source->size : 0u, source ? source->stride : 0u,
                         source ? source->fetch_pc : UINT32_MAX,
                         outer ? static_cast<uint32_t>(outer->cls) : UINT32_MAX,
                         outer ? static_cast<unsigned long long>(outer->gpu_addr) : 0ull,
                         outer ? outer->size : 0u, outer ? outer->stride : 0u,
                         outer ? outer->fetch_pc : UINT32_MAX);
        return false;
    }
    const SelectedSbufferDomain domain = selected_sbuffer_domain(*source);
    if (domain == SelectedSbufferDomain::Reject) {
        if (std::getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[gta-selected-sbuffer] reject=selector-domain\n");
        return false;
    }

    if (domain == SelectedSbufferDomain::AllOob) {
        ShaderResource* target_x4 = exact_fetch_resource(resources, kSelectedSbufferLoadX4Pc);
        ShaderResource* target_x2 = exact_fetch_resource(resources, kSelectedSbufferLoadX2Pc);
        if ((target_x4 && !is_zero_record_raw_buffer(*target_x4)) ||
            (target_x2 && !is_zero_record_raw_buffer(*target_x2)))
            return false;
        auto add_zero_target = [&](uint32_t pc) {
            ShaderResource target;
            target.cls = ResourceClass::ConstantBuffer;
            target.format = DataFormat::Unknown;
            target.num_components = 0u;
            target.fetch_pc = pc;
            resources.resources.push_back(target);
        };
        if (!target_x4) add_zero_target(kSelectedSbufferLoadX4Pc);
        if (!target_x2) add_zero_target(kSelectedSbufferLoadX2Pc);
        outer = exact_fetch_resource(resources, kSelectedSbufferOuterPc);
        if (!outer) return false;
        outer->selected_sbuffer_soffset = kGtaSelectedSbufferAllOobSoffset;
        if (std::getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[gta-selected-sbuffer] domain=all-oob\n");
        return rdna2_gta5_selected_sbuffer_dispatch(code, dwords, config, resources);
    }

    const uint8_t* outer_bytes = complete_resource_bytes(*outer, outer->size);
    if (!outer_bytes) {
        if (std::getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[gta-selected-sbuffer] reject=outer-unreadable\n");
        return false;
    }
    std::array<uint32_t, 4> selected_words{};
    std::memcpy(selected_words.data(),
                outer_bytes + kGtaSelectedSbufferRecord4Soffset + 8u,
                sizeof(selected_words));
    DecodedBufferDescriptor selected{};
    if (!selected_sbuffer_target_descriptor(selected_words, selected)) {
        if (std::getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[gta-selected-sbuffer] reject=selected-vsharp words=%08x:%08x:%08x:%08x "
                         "base=0x%llx stride=%u records=%u size=%u format=%u components=%u\n",
                         selected_words[0], selected_words[1], selected_words[2], selected_words[3],
                         static_cast<unsigned long long>(selected.base), selected.stride,
                         selected.num_records, selected.size_bytes,
                         static_cast<uint32_t>(selected.format), selected.num_components);
        return false;
    }

    ShaderResource* target_x4 = exact_fetch_resource(resources, kSelectedSbufferLoadX4Pc);
    ShaderResource* target_x2 = exact_fetch_resource(resources, kSelectedSbufferLoadX2Pc);
    if ((target_x4 && !selected_sbuffer_target_resource(
                          *target_x4, kSelectedSbufferLoadX4Pc, selected)) ||
        (target_x2 && !selected_sbuffer_target_resource(
                          *target_x2, kSelectedSbufferLoadX2Pc, selected)))
    {
        if (std::getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[gta-selected-sbuffer] reject=consumer-resource\n");
        return false;
    }

    auto add_target = [&](uint32_t pc) {
        ShaderResource target;
        target.cls = ResourceClass::ConstantBuffer;
        target.format = selected.format;
        target.num_components = selected.num_components;
        target.gpu_addr = selected.base;
        target.size = selected.size_bytes;
        target.stride = selected.stride;
        target.fetch_pc = pc;
        resources.resources.push_back(target);
    };
    if (!target_x4) add_target(kSelectedSbufferLoadX4Pc);
    if (!target_x2) add_target(kSelectedSbufferLoadX2Pc);

    // push_back may have moved the vector, so reacquire the owning outer record.
    outer = exact_fetch_resource(resources, kSelectedSbufferOuterPc);
    if (!outer) return false;
    outer->selected_sbuffer_soffset = kGtaSelectedSbufferRecord4Soffset;
    outer->selected_sbuffer_words = selected_words;
    return rdna2_gta5_selected_sbuffer_dispatch(code, dwords, config, resources);
}

bool rdna2_gta5_selected_sbuffer_dispatch(
        const uint32_t* code, size_t dwords,
        const ComputeShaderConfig& config,
        const ShaderResourceTable& resources) {
    if (!rdna2_gta5_selected_sbuffer_launch(code, dwords, config)) return false;
    const ShaderResource* source = exact_fetch_resource(resources, kSelectedSbufferSourcePc);
    const ShaderResource* outer = exact_fetch_resource(resources, kSelectedSbufferOuterPc);
    const ShaderResource* target_x4 = exact_fetch_resource(resources, kSelectedSbufferLoadX4Pc);
    const ShaderResource* target_x2 = exact_fetch_resource(resources, kSelectedSbufferLoadX2Pc);
    if (!source || !outer || !target_x4 || !target_x2)
        return false;

    size_t markers = 0;
    for (const ShaderResource& resource : resources.resources) {
        if (!is_gta5_selected_sbuffer_marker_candidate(resource)) continue;
        if (&resource != outer || !is_gta5_selected_sbuffer_descriptor(resource)) return false;
        ++markers;
    }
    if (markers != 1u) return false;

    if (outer->selected_sbuffer_soffset == kGtaSelectedSbufferZeroChainSoffset)
        return is_zero_record_raw_buffer(*source) && is_zero_record_raw_buffer(*outer) &&
               is_zero_record_raw_buffer(*target_x4) && is_zero_record_raw_buffer(*target_x2);

    if (
        !selected_sbuffer_source_shape(*source) ||
        !selected_sbuffer_outer_shape(*outer) ||
        !is_gta5_selected_sbuffer_descriptor(*outer))
        return false;

    const SelectedSbufferDomain domain = selected_sbuffer_domain(*source);
    if (outer->selected_sbuffer_soffset == kGtaSelectedSbufferAllOobSoffset)
        return domain == SelectedSbufferDomain::AllOob &&
               is_zero_record_raw_buffer(*target_x4) &&
               is_zero_record_raw_buffer(*target_x2);
    if (domain != SelectedSbufferDomain::Record4) return false;

    const uint8_t* outer_bytes = complete_resource_bytes(*outer, outer->size);
    if (!outer_bytes) return false;
    std::array<uint32_t, 4> live_words{};
    std::memcpy(live_words.data(),
                outer_bytes + kGtaSelectedSbufferRecord4Soffset + 8u,
                sizeof(live_words));
    if (live_words != outer->selected_sbuffer_words) return false;
    DecodedBufferDescriptor selected{};
    if (!selected_sbuffer_target_descriptor(live_words, selected)) return false;
    return selected_sbuffer_target_resource(
               *target_x4, kSelectedSbufferLoadX4Pc, selected) &&
           selected_sbuffer_target_resource(
               *target_x2, kSelectedSbufferLoadX2Pc, selected);
}

} // namespace prosper::gpu
