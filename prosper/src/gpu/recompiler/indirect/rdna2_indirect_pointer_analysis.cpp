#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace prosper::gpu {

bool guest_readable(uint64_t address, uint32_t bytes);

namespace {

constexpr uint32_t kSourceFetchPc = 6u;
constexpr uint32_t kSelectorFetchPc = 16u;
constexpr uint32_t kSourceStride = 96u;
constexpr uint32_t kPointerByteOffset = 0u;
constexpr uint32_t kSelectorByteOffset = 88u;
constexpr uint32_t kPointerVgpr = 24u;

struct PacketSite {
    uint32_t pc;
    uint32_t word0;
    uint32_t word1;
    uint32_t dwords;
};

// These packets are the complete authority surface, not a complete-program identity. Unrelated
// ALU/wait packets may change without affecting the proof, while each producer, EXEC transition,
// branch target, restore, and indirect consumer remains exact.
constexpr std::array<PacketSite, 30> kProofPackets{{
    {1u,   0xd746002bu, 0x04010c05u, 2u}, // global record index
    {3u,   0xf4080200u, 0xfa0000a0u, 2u}, // source V# at SRT +0xa0
    {5u,   0xbf8cc07fu, 0u,          1u}, // descriptor-load wait, no scalar clobber
    {6u,   0xe0382000u, 0x8002182bu, 2u}, // pointer qword in dwordx4
    {8u,   0xe0382010u, 0x8002132bu, 2u}, // intervening record load, preserves v43
    {10u,  0xe0382020u, 0x80020d2bu, 2u}, // intervening record load, preserves v43
    {12u,  0xe0382030u, 0x8002052bu, 2u}, // intervening record load, preserves v43
    {14u,  0xe0382040u, 0x8002092bu, 2u}, // intervening record load, preserves v43
    {16u,  0xe0382050u, 0x8002002bu, 2u}, // selector at record byte 88
    {29u,  0x7d8a04f9u, 0x06868682u, 2u}, // selector == 2 mask in s[6:7]
    {38u,  0xbe80047eu, 0u,          1u}, // save pre-filter EXEC
    {39u,  0x7daa0481u, 0u,          1u}, // selector != 1 CMPX
    {43u,  0xbf88014au, 0u,          1u}, // empty filtered set -> final restore
    {44u,  0xbe822406u, 0u,          1u}, // save filtered mask/select mode 2
    {45u,  0xbf8800ddu, 0u,          1u}, // no mode 2 -> complementary arm
    {47u,  0xbe86047eu, 0u,          1u}, // save mode-2 EXEC before null guard
    {48u,  0x7dea3080u, 0u,          1u}, // complete U64 pointer != 0 CMPX
    {49u,  0xbf88001cu, 0u,          1u}, // null pointer skips mode-2 load
    {50u,  0xdc308018u, 0x027d0018u, 2u}, // mode 2: dword at +24
    {78u,  0xbefe0406u, 0u,          1u}, // restore mode-2 EXEC
    {267u, 0x8afe7e02u, 0u,          1u}, // complementary filtered mask
    {268u, 0xbf880052u, 0u,          1u}, // empty complement -> final restore
    {271u, 0xbe86047eu, 0u,          1u}, // save mode-0 EXEC before null guard
    {272u, 0x7dea3080u, 0u,          1u}, // complete U64 pointer != 0 CMPX
    {273u, 0xbf880018u, 0u,          1u}, // null pointer skips mode-0 loads
    {274u, 0xdc308018u, 0x027d0018u, 2u}, // mode 0: dword at +24
    {276u, 0xdc308030u, 0x227d0018u, 2u}, // mode 0: dword at +48
    {298u, 0xbefe0406u, 0u,          1u}, // restore mode-0 EXEC
    {351u, 0xbefe0402u, 0u,          1u}, // restore pre-arm filtered EXEC
    {374u, 0x8afe7e00u, 0u,          1u}, // select original lanes omitted by the filter
}};

const Rdna2Inst* instruction_at(const std::vector<Rdna2Inst>& instructions, uint32_t pc) {
    const auto found = std::find_if(
        instructions.begin(), instructions.end(),
        [&](const Rdna2Inst& instruction) { return instruction.pc == pc; });
    return found == instructions.end() ? nullptr : &*found;
}

bool operand_is(const Operand& operand, OperandKind kind, int32_t value) {
    return operand.kind == kind && operand.value == value;
}

bool packet_is(const Rdna2Inst& instruction, const PacketSite& site) {
    return instruction.pc == site.pc && instruction.len_dwords == site.dwords &&
           instruction.words[0] == site.word0 &&
           (site.dwords == 1u || instruction.words[1] == site.word1);
}

bool branch_target_is(const Rdna2Inst& instruction, uint32_t target) {
    if (instruction.fmt != Rdna2Format::SOPP ||
        !sopp_opcode_is_direct_branch(instruction.opcode))
        return false;
    const int64_t computed = static_cast<int64_t>(instruction.pc) +
        instruction.len_dwords + instruction.simm16;
    return computed >= 0 && static_cast<uint64_t>(computed) == target;
}

bool raw_dwordx4_indexed_load(const Rdna2Inst& instruction, uint32_t pc,
                              uint32_t offset, uint32_t destination) {
    return instruction.pc == pc && instruction.fmt == Rdna2Format::MUBUF &&
           instruction.opcode == kMubufOpcodeLoadDwordX4 &&
           instruction.len_dwords == 2u && instruction.dst.kind == OperandKind::VGPR &&
           instruction.dst.value == static_cast<int32_t>(destination) &&
           operand_is(instruction.src[0], OperandKind::VGPR, 43) &&
           operand_is(instruction.src[1], OperandKind::SGPR, 8) &&
           operand_is(instruction.src[2], OperandKind::InlineInt, 0) &&
           instruction.literal == (offset | (1u << 13u)) &&
           !instruction.mubuf_glc && !instruction.mubuf_dlc &&
           !instruction.mubuf_lds && !instruction.mubuf_tfe;
}

bool global_dword_load(const Rdna2Inst& instruction, uint32_t pc,
                       uint32_t offset, uint32_t destination) {
    return instruction.pc == pc && instruction.fmt == Rdna2Format::FLAT &&
           instruction.flat_segment == 2u &&
           instruction.opcode == kMubufOpcodeLoadDword &&
           instruction.len_dwords == 2u &&
           operand_is(instruction.dst, OperandKind::VGPR, destination) &&
           operand_is(instruction.src[0], OperandKind::VGPR, kPointerVgpr) &&
           operand_is(instruction.src[1], OperandKind::Special, 125) &&
           instruction.literal == offset && !instruction.flat_glc &&
           !instruction.flat_slc && !instruction.flat_dlc && !instruction.flat_lds;
}

bool writes_pointer_pair(const Rdna2Inst& instruction) {
    if (instruction.dst.kind != OperandKind::VGPR) return false;
    const uint32_t count = rdna2_vgpr_destination_span(instruction);
    if (!count) return false;
    const uint32_t first = static_cast<uint32_t>(instruction.dst.value);
    const uint64_t end = static_cast<uint64_t>(first) + count;
    return first <= kPointerVgpr + 1u && end > kPointerVgpr;
}

bool pointer_pair_unchanged(const std::vector<Rdna2Inst>& instructions,
                            uint32_t begin_pc, uint32_t end_pc) {
    return std::none_of(instructions.begin(), instructions.end(), [&](const Rdna2Inst& in) {
        return in.pc > begin_pc && in.pc < end_pc && writes_pointer_pair(in);
    });
}

bool vgpr_unchanged(const std::vector<Rdna2Inst>& instructions,
                    uint32_t begin_pc, uint32_t end_pc, uint32_t vgpr) {
    return std::none_of(instructions.begin(), instructions.end(), [&](const Rdna2Inst& in) {
        if (in.pc <= begin_pc || in.pc >= end_pc || in.dst.kind != OperandKind::VGPR)
            return false;
        const uint32_t count = rdna2_vgpr_destination_span(in);
        if (!count || in.dst.value < 0) return false;
        const uint32_t first = static_cast<uint32_t>(in.dst.value);
        return first <= vgpr && static_cast<uint64_t>(first) + count > vgpr;
    });
}

uint32_t conservative_scalar_write_width(const Rdna2Inst& instruction) {
    if (instruction.dst.kind != OperandKind::SGPR &&
        instruction.dst.kind != OperandKind::Special)
        return 0u;
    if (instruction.fmt == Rdna2Format::SMEM) {
        switch (instruction.opcode) {
            case kSmemOpcodeLoadDword: case kSmemOpcodeBufferLoadDword: return 1u;
            case kSmemOpcodeLoadDwordX2: case kSmemOpcodeBufferLoadDwordX2: return 2u;
            case kSmemOpcodeLoadDwordX4: case kSmemOpcodeBufferLoadDwordX4: return 4u;
            case kSmemOpcodeLoadDwordX8: case kSmemOpcodeBufferLoadDwordX8: return 8u;
            case kSmemOpcodeLoadDwordX16: case kSmemOpcodeBufferLoadDwordX16: return 16u;
            default: return 1u;
        }
    }
    // The analyzer deliberately admits unrelated same-length scalar packets. Count every SOP1/SOP2
    // destination as a pair rather than maintaining a second, inevitably partial B64 opcode table:
    // under-counting an unbound B64 mutation could hide a write that starts one SGPR before a saved
    // mask. The false-positive edge is harmless here; rejecting an otherwise unrelated mutation is
    // preferable to granting relocation authority after a partially hidden mask clobber.
    if (instruction.fmt == Rdna2Format::SOP1 || instruction.fmt == Rdna2Format::SOP2)
        return 2u;
    if (instruction.fmt == Rdna2Format::VOPC) return 2u;
    return 1u;
}

bool scalar_range_unchanged(const std::vector<Rdna2Inst>& instructions,
                            uint32_t begin_pc, uint32_t end_pc,
                            uint32_t first_sgpr, uint32_t count) {
    const uint64_t wanted_end = static_cast<uint64_t>(first_sgpr) + count;
    return std::none_of(instructions.begin(), instructions.end(), [&](const Rdna2Inst& in) {
        if (in.pc <= begin_pc || in.pc >= end_pc) return false;
        const uint32_t width = conservative_scalar_write_width(in);
        if (width && in.dst.value >= 0) {
            const uint32_t first = static_cast<uint32_t>(in.dst.value);
            if (first < wanted_end && static_cast<uint64_t>(first) + width > first_sgpr)
                return true;
        }
        if ((in.sdst.kind == OperandKind::SGPR || in.sdst.kind == OperandKind::Special) &&
            in.sdst.value >= 0) {
            const uint32_t first = static_cast<uint32_t>(in.sdst.value);
            return first < wanted_end && static_cast<uint64_t>(first) + 1u > first_sgpr;
        }
        return false;
    });
}

bool indirect_control_transfer(const Rdna2Inst& instruction) {
    return (instruction.fmt == Rdna2Format::SOP1 &&
            (instruction.opcode == kSop1OpcodeSetpcB64 ||
             instruction.opcode == kSop1OpcodeSwappcB64 ||
             instruction.opcode == kSop1OpcodeRfeB64)) ||
           (instruction.fmt == Rdna2Format::SOPK &&
            (instruction.opcode == kSopkOpcodeCallB64 ||
             instruction.opcode == kSopkOpcodeSubvectorLoopBegin ||
             instruction.opcode == kSopkOpcodeSubvectorLoopEnd)) ||
           (instruction.fmt == Rdna2Format::SOPP &&
            (instruction.opcode == kSoppOpcodeTrap ||
             instruction.opcode == kSoppOpcodeCbranchCdbgsys ||
             instruction.opcode == kSoppOpcodeCbranchCdbguser ||
             instruction.opcode == kSoppOpcodeCbranchCdbgsysOrUser ||
             instruction.opcode == kSoppOpcodeCbranchCdbgsysAndUser));
}

const ShaderResource* unique_resource_at(const ShaderResourceTable& resources,
                                         uint32_t fetch_pc) {
    const ShaderResource* result = nullptr;
    for (const ShaderResource& resource : resources.resources) {
        if (resource.fetch_pc != fetch_pc) continue;
        if (result) return nullptr;
        result = &resource;
    }
    return result;
}

const uint8_t* complete_source_bytes(const ShaderResource& source) {
    if (source.host_data && source.host_data_size >= source.size) return source.host_data;
    return source.size <= UINT32_MAX && guest_readable(source.gpu_addr, source.size)
        ? reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(source.gpu_addr))
        : nullptr;
}

uint32_t load_u32(const uint8_t* bytes, size_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

uint64_t load_u64(const uint8_t* bytes, size_t offset) {
    uint64_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

uint64_t fingerprint_mix(uint64_t hash, uint64_t value) {
    for (uint32_t byte = 0; byte < 8u; ++byte) {
        hash ^= static_cast<uint8_t>(value >> (byte * 8u));
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t proof_fingerprint(const IndirectPointerRelocationProof& proof) {
    uint64_t hash = 1469598103934665603ull;
    hash = fingerprint_mix(hash, proof.schema_version);
    hash = fingerprint_mix(hash, static_cast<uint32_t>(proof.bound_kind));
    hash = fingerprint_mix(hash, static_cast<uint32_t>(proof.guard_kind));
    hash = fingerprint_mix(hash, proof.source_fetch_pc);
    hash = fingerprint_mix(hash, proof.source_stride);
    hash = fingerprint_mix(hash, proof.pointer_byte_offset);
    hash = fingerprint_mix(hash, proof.footprint_selector_byte_offset);
    hash = fingerprint_mix(hash, proof.record_count);
    hash = fingerprint_mix(hash, proof.max_footprint_bytes);
    hash = fingerprint_mix(hash, proof.selector_footprint_bytes.size());
    for (uint32_t bytes : proof.selector_footprint_bytes)
        hash = fingerprint_mix(hash, bytes);
    hash = fingerprint_mix(hash, proof.accesses.size());
    for (const IndirectPointerAccessProof& access : proof.accesses) {
        hash = fingerprint_mix(hash, access.pc);
        hash = fingerprint_mix(hash, access.words[0]);
        hash = fingerprint_mix(hash, access.words[1]);
        hash = fingerprint_mix(hash, access.address_vgpr);
        hash = fingerprint_mix(hash, access.immediate_byte_offset);
        hash = fingerprint_mix(hash, access.component_bytes);
        hash = fingerprint_mix(hash, access.components);
    }
    return hash;
}

} // namespace

bool analyze_rdna2_static_pointer_footprint(
        const uint32_t* code, size_t dwords,
        const ComputeShaderConfig& config,
        const ShaderResourceTable& resources,
        IndirectPointerRelocationProof& proof) {
    proof = {};
    if (!code || dwords < 386u || config.user_sgprs.size() != 5u ||
        config.local_x != 64u || config.local_y != 1u || config.local_z != 1u ||
        !config.exact_thread_extent || !config.threads_x ||
        config.threads_x > kIndirectPointerStaticFootprintLayout.max_records ||
        config.threads_y != 1u || config.threads_z != 1u || config.wave_size != 64u ||
        !config.tgid_x_en || config.tgid_y_en || config.tgid_z_en ||
        config.tg_size_en || config.tidig_comp_cnt != 0u)
        return false;

    std::vector<Rdna2Inst> instructions;
    const size_t consumed = rdna2_walk(code, dwords, instructions);
    if (consumed != 386u || instructions.empty() || !instructions.back().is_end ||
        instructions.back().pc != 385u)
        return false;
    for (const PacketSite& site : kProofPackets) {
        const Rdna2Inst* instruction = instruction_at(instructions, site.pc);
        if (!instruction || !packet_is(*instruction, site)) return false;
    }

    const Rdna2Inst* index = instruction_at(instructions, 1u);
    const Rdna2Inst* descriptor = instruction_at(instructions, 3u);
    const Rdna2Inst* pointer_load = instruction_at(instructions, kSourceFetchPc);
    const Rdna2Inst* selector_load = instruction_at(instructions, kSelectorFetchPc);
    if (!index || index->fmt != Rdna2Format::VOP3 || index->opcode != 0x346u ||
        !operand_is(index->dst, OperandKind::VGPR, 43) ||
        !operand_is(index->src[0], OperandKind::SGPR, 5) ||
        !operand_is(index->src[1], OperandKind::InlineInt, 6) ||
        !operand_is(index->src[2], OperandKind::VGPR, 0) || index->has_modifier ||
        !descriptor || descriptor->fmt != Rdna2Format::SMEM ||
        descriptor->opcode != kSmemOpcodeLoadDwordX4 ||
        !operand_is(descriptor->dst, OperandKind::SGPR, 8) ||
        !operand_is(descriptor->src[0], OperandKind::SGPR, 0) ||
        !operand_is(descriptor->src[1], OperandKind::Special, 125) ||
        descriptor->literal != 0xa0u ||
        !pointer_load || !raw_dwordx4_indexed_load(
            *pointer_load, kSourceFetchPc, 0u, kPointerVgpr) ||
        !selector_load || !raw_dwordx4_indexed_load(
            *selector_load, kSelectorFetchPc, 80u, 0u))
        return false;

    const Rdna2Inst* selector_filter = instruction_at(instructions, 39u);
    const Rdna2Inst* first_guard = instruction_at(instructions, 48u);
    const Rdna2Inst* second_guard = instruction_at(instructions, 272u);
    if (!selector_filter || selector_filter->fmt != Rdna2Format::VOPC ||
        selector_filter->opcode != 0xd5u || !vopc_is_cmpx(selector_filter->opcode) ||
        !operand_is(selector_filter->src[0], OperandKind::InlineInt, 1) ||
        !operand_is(selector_filter->src[1], OperandKind::VGPR, 2) ||
        !first_guard || !second_guard || first_guard->fmt != Rdna2Format::VOPC ||
        second_guard->fmt != Rdna2Format::VOPC || first_guard->opcode != 0xf5u ||
        second_guard->opcode != 0xf5u || !vopc_is_cmpx(first_guard->opcode) ||
        !vopc_is_cmpx(second_guard->opcode) ||
        !operand_is(first_guard->src[0], OperandKind::InlineInt, 0) ||
        !operand_is(first_guard->src[1], OperandKind::VGPR, kPointerVgpr) ||
        !operand_is(second_guard->src[0], OperandKind::InlineInt, 0) ||
        !operand_is(second_guard->src[1], OperandKind::VGPR, kPointerVgpr))
        return false;

    struct ExpectedBranch { uint32_t pc, target; };
    constexpr std::array<ExpectedBranch, 5> branches{{
        {43u, 374u}, {45u, 267u}, {49u, 78u}, {268u, 351u}, {273u, 298u},
    }};
    for (const auto& expected : branches) {
        const Rdna2Inst* branch = instruction_at(instructions, expected.pc);
        if (!branch || branch->opcode != kSoppOpcodeCbranchExecz ||
            !branch_target_is(*branch, expected.target))
            return false;
    }
    for (const Rdna2Inst& instruction : instructions) {
        if (instruction.fmt != Rdna2Format::SOPP ||
            !sopp_opcode_is_direct_branch(instruction.opcode))
            continue;
        // pc1/3/6/16 are the only admitted producer chain. A prefix branch could skip one or more
        // of them and still fall through to the guarded consumers with undefined pointer state.
        if (instruction.pc < 39u) return false;
        const bool inside_proof_region = instruction.pc >= 39u && instruction.pc < 375u;
        const bool expected = std::any_of(branches.begin(), branches.end(),
            [&](const ExpectedBranch& branch) { return branch.pc == instruction.pc; });
        if (inside_proof_region && !expected) return false;
        const int64_t target = static_cast<int64_t>(instruction.pc) +
            instruction.len_dwords + instruction.simm16;
        if (!inside_proof_region && target >= 39 && target < 375) return false;
    }

    // The complementary pc267 mask is only selector 0 if the mode-2 arm restored every temporary
    // EXEC narrowing. Make the authority inventory executable: a same-length mutation anywhere in
    // the arm cannot introduce an unaccounted CMPX/saveexec/explicit EXEC write or indirect transfer.
    constexpr std::array<uint32_t, 9> expected_exec_writers{
        39u, 44u, 48u, 78u, 267u, 272u, 298u, 351u, 374u,
    };
    for (const Rdna2Inst& instruction : instructions) {
        if (indirect_control_transfer(instruction)) return false;
        if (instruction.pc < 39u || instruction.pc > 374u) continue;
        if (instruction.is_end) return false;
        if (!rdna2_instruction_may_change_exec(instruction)) continue;
        if (std::find(expected_exec_writers.begin(), expected_exec_writers.end(),
                      instruction.pc) == expected_exec_writers.end())
            return false;
    }
    for (uint32_t pc : expected_exec_writers) {
        const Rdna2Inst* instruction = instruction_at(instructions, pc);
        if (!instruction || !rdna2_instruction_may_change_exec(*instruction)) return false;
    }

    // Both pc6 and pc16 consume the pc1 global-record index and pc3 descriptor. Binding their own
    // packets is insufficient if an intervening same-length instruction overwrites either producer.
    if (!vgpr_unchanged(instructions, 1u, 17u, 43u) ||
        !vgpr_unchanged(instructions, kSelectorFetchPc, 40u, 2u) ||
        !scalar_range_unchanged(instructions, 3u, 17u, 8u, 4u))
        return false;

    // Retain each saved mask until every consumer. In particular, s[2:3] is the filtered pre-arm
    // mask used both by pc267's complement and pc351's restore; a scalar clobber in the long mode-2
    // body would otherwise let a selector-2 lane enter the 52-byte arm with only 28 bytes captured.
    if (!scalar_range_unchanged(instructions, 38u, 374u, 0u, 2u) ||
        !scalar_range_unchanged(instructions, 29u, 44u, 6u, 2u) ||
        !scalar_range_unchanged(instructions, 44u, 351u, 2u, 2u) ||
        !scalar_range_unchanged(instructions, 47u, 78u, 6u, 2u) ||
        !scalar_range_unchanged(instructions, 271u, 298u, 6u, 2u))
        return false;

    if (!pointer_pair_unchanged(instructions, kSourceFetchPc, 50u) ||
        !pointer_pair_unchanged(instructions, 267u, 277u))
        return false;

    const Rdna2Inst* access50 = instruction_at(instructions, 50u);
    const Rdna2Inst* access274 = instruction_at(instructions, 274u);
    const Rdna2Inst* access276 = instruction_at(instructions, 276u);
    if (!access50 || !global_dword_load(*access50, 50u, 24u, 2u) ||
        !access274 || !global_dword_load(*access274, 274u, 24u, 2u) ||
        !access276 || !global_dword_load(*access276, 276u, 48u, 34u))
        return false;
    const size_t flat_count = static_cast<size_t>(std::count_if(
        instructions.begin(), instructions.end(),
        [](const Rdna2Inst& instruction) { return instruction.fmt == Rdna2Format::FLAT; }));
    if (flat_count != 3u) return false;

    const ShaderResource* source = unique_resource_at(resources, kSourceFetchPc);
    const ShaderResource* selector_source = unique_resource_at(resources, kSelectorFetchPc);
    const uint64_t expected_source_bytes =
        static_cast<uint64_t>(config.threads_x) * kSourceStride;
    if (!source || !selector_source || expected_source_bytes > UINT32_MAX ||
        source->cls != ResourceClass::ConstantBuffer || source->fetch_pc != kSourceFetchPc ||
        source->size != expected_source_bytes || source->stride != kSourceStride ||
        !source->gpu_addr || source->table_index_count != 0u || source->srt_offset != 0xa0u ||
        source->sgpr_base != UINT32_MAX ||
        selector_source->cls != ResourceClass::ConstantBuffer ||
        selector_source->size != source->size || selector_source->stride != source->stride ||
        selector_source->gpu_addr != source->gpu_addr ||
        selector_source->table_index_count != 0u)
        return false;
    const uint8_t* source_bytes = complete_source_bytes(*source);
    const uint8_t* selector_source_bytes = complete_source_bytes(*selector_source);
    if (!source_bytes || !selector_source_bytes ||
        std::memcmp(source_bytes, selector_source_bytes, source->size) != 0)
        return false;

    IndirectPointerRelocationProof candidate;
    candidate.schema_version = kIndirectPointerProofSchema;
    candidate.bound_kind = IndirectPointerBoundKind::StaticFootprint;
    candidate.guard_kind = IndirectPointerGuardKind::Full64NonZero;
    candidate.source_fetch_pc = kSourceFetchPc;
    candidate.source_stride = kSourceStride;
    candidate.pointer_byte_offset = kPointerByteOffset;
    candidate.footprint_selector_byte_offset = kSelectorByteOffset;
    candidate.selector_footprint_bytes = {52u, 0u, 28u};
    candidate.record_count = config.threads_x;
    candidate.max_footprint_bytes = 52u;
    candidate.accesses = {
        {50u,  {0xdc308018u, 0x027d0018u}, kPointerVgpr, 24u, 4u, 1u},
        {274u, {0xdc308018u, 0x027d0018u}, kPointerVgpr, 24u, 4u, 1u},
        {276u, {0xdc308030u, 0x227d0018u}, kPointerVgpr, 48u, 4u, 1u},
    };
    candidate.records.reserve(config.threads_x);
    for (uint32_t record_index = 0; record_index < config.threads_x; ++record_index) {
        const uint32_t source_offset = record_index * kSourceStride;
        const uint64_t pointer = load_u64(source_bytes, source_offset + kPointerByteOffset);
        const uint32_t selector = load_u32(
            source_bytes, source_offset + kSelectorByteOffset);
        if (selector >= candidate.selector_footprint_bytes.size()) return false;
        candidate.records.push_back({
            source_offset,
            pointer,
            pointer ? candidate.selector_footprint_bytes[selector] : 0u,
        });
    }
    candidate.fingerprint = proof_fingerprint(candidate);
    const uint32_t fingerprint_low = static_cast<uint32_t>(candidate.fingerprint);
    const uint32_t fingerprint_high = static_cast<uint32_t>(candidate.fingerprint >> 32u);
    candidate.witness_words = {
        candidate.schema_version,
        fingerprint_low,
        fingerprint_high,
        kIndirectPointerStaticFootprintTag ^ candidate.schema_version ^
            fingerprint_low ^ fingerprint_high,
    };
    proof = std::move(candidate);
    return true;
}

const IndirectPointerAccessProof* rdna2_indirect_pointer_access(
        const IndirectPointerRelocationProof& proof,
        const Rdna2Inst& instruction) {
    const auto found = std::find_if(
        proof.accesses.begin(), proof.accesses.end(),
        [&](const IndirectPointerAccessProof& access) {
            return instruction.pc == access.pc && instruction.len_dwords == 2u &&
                   instruction.words[0] == access.words[0] &&
                   instruction.words[1] == access.words[1];
        });
    return found == proof.accesses.end() ? nullptr : &*found;
}

} // namespace prosper::gpu
