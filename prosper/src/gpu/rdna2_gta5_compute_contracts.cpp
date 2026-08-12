#include "rdna2_gta5_compute_contracts.hpp"

#include "rdna2_decode.hpp"
#include "rdna2_to_spirv.hpp"
#include "shader_resources.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace prosper::gpu {

// Implemented by the guest-memory registry. Live markers retain a guest table address; replay
// markers instead carry owned host_data and never call this path.
bool guest_readable(uint64_t address, uint32_t bytes);

namespace {

enum class NullableProgram : uint8_t { None, WorkgroupStore, WorkgroupProcess };

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

} // namespace prosper::gpu
