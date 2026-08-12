#include "rdna2_gta5_packed_pointer.hpp"

#include "rdna2_decode.hpp"
#include "rdna2_indirect_buffer_shadow.hpp"
#include "rdna2_to_spirv.hpp"
#include "shader_resources.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace prosper::gpu {

bool guest_readable(uint64_t address, uint32_t bytes);
bool guest_writable(uint64_t address, uint32_t bytes);

namespace {

constexpr uint32_t kProgramDwords = 362u;
constexpr uint32_t kDescriptorStrideOrPc = 347u;
constexpr uint32_t kDescriptorStrideOrPacket = 0x8801ff05u;
constexpr uint32_t kDescriptorStrideOrLiteral = 0x01000000u;
static_assert(kDescriptorStrideOrLiteral == kGta5PackedPointerDescriptorStride << 16u);
constexpr std::array<uint32_t, kGta5PackedPointerMaxSlots> kRecords{0u, 64u, 128u};
constexpr IndirectBufferShadowLayout kShadowLayout{
    kGta5PackedPointerTag,
    1u,
    kGta5PackedPointerSourceStride,
    kGta5PackedPointerSlotBytes,
    static_cast<uint32_t>(kRecords.size()),
    2u,
};
static_assert(kGta5PackedPointerHeaderBytes ==
              16u + kRecords.size() * sizeof(uint64_t) + 2u * sizeof(uint32_t));

// Exact consumed prefix of routed GTA V 0x413cf9d00. Byte comparison avoids a 362-element magic-
// number initializer while retaining the complete-program authority boundary.
constexpr char kProgramHex[] =
"0300a0bf8124eabeff020a7e000080ffff020c7e000080ffff02027e0000807fff02047e0000807fff02067e0000807fff02087e000080ff80020e7e100034d90705000000007cdb070100006a04febe250046d70e0c010406038cbe07038dbe08038ebe09038fbe002038e025000380102038e025040380582030e02527038024203ce0251f038034203ce0251c038044203ce025220380733f8cbff94e8a7d82888606804eaa7d6a0486be4e0088bf082488be070088bf05031c7e04031a7e0203147e0103127e0003107e80024c7e8002507e087efe8a420088bfff021c7e000080ffff021a7e000080ffff02067e000080ffff02147e0000807fff02127e0000807fc102107e81024c7e8002507e8000ea7d330088bf188030dc00007d037e04eabe703f8cbf8d0606348006287d010088bf010092bf6a04febefc8030dc00007d03703f8cbf8906aa7d010088bf010092bf6a04febe108138dc00007d0d208138dc00007d11308138dc00007d15408138dc00007d04508138dc00007d098004847d608138dc00007d00f902510280068686733f8cbf12311020112f2220102d20200f2b1e1e0e29241e0d27261e703f8cbf0e0054d503112a040d0054d502232604030054d501211e040a0051d5001f1a04090051d50c251604080051d50b2712040804febe067efe8a850088bfff021c7e000080ffff021a7e000080ffff02067e000080ffff02147e0000807fff02127e0000807fc102107e80024c7e8002507e8000ea7d760088bf188030dc00007d037e04eabe703f8cbf8d0606348006287d010088bf010092bf6a04febefc8030dc00007d03703f8cbf8906aa7d010088bf010092bf6a04febe188030dc00007d03108138dc00007d18308138dc00007d10408138dc00007d14508138dc00007d08608138dc00007d0ca00408369004043680088a7df9048a7d80888606753f8cbfff06083600100000208138dc00007d00f908847d808a86060a6aea8a086aea88f902510280068686743f8cbf1b230820703f8cbf1805041e1907061e012702201a21201e040054d50d095e040a0051d50a055204020051d50b07560400251620090054d50f0326040c0051d50c215a04f90806060ac60606f914080804c606060a0054d50e172204f91202060cc60606030f483e07090e10030d463e030b443e06090610f9140c0602c60606f90404080ac6060605091410f918000809c606060643483e040741d521051e04090741d520050e04050741d51f052a04070741d521051ea40a0741d51f052aa40641463e063f443e013d483e04031fd5003d0200060341d51c012a24013b463e09031fd5003b02000139443e010741d520050ea405031fd50039020007031fd5003d024004491c0609471a0601031fd5003b0240054506060749140606451006014712060604febef91c00060ac60606f91a020609c60606f906040608c60606000304f4900000faff038fbe04620100020388be030389be04038abe05038bbe7fc08cbf921d8dbe8002087e000048d804020000040048d804010000080048d8040000000c004cd80402000010004cd80401000014004cd8040000009d4e0034ff4a0236ffffff1f03026ab00303167ef9d4107d08000686000071d7ff00060400000060002078e025080280f9000302810686860f006fd7283f010410207ce0250d0280002070e025010380804c8a7d8106fe878002087e01ff058800000400000384be860386beff0387be046201000000fcdb0400000081020c7e82020e7e7fc08cbf0000fce0000001808302007e8402107e8502127e1000d8d904000004806a13bf0020fce0060101800020fce007020180002000e1000301807fc08cbf002000e108040180002000e109050180140084bfff02007e000010008002027e000104f4880000fa7fc08cbf05ff018800000001040380be810382beff0383be04620100821a00f4180000fa180068e1000000807fc08cbf6a840dbf010084bf010092bf000081bf";

uint8_t nibble(char value) {
    return value >= '0' && value <= '9' ? static_cast<uint8_t>(value - '0')
                                        : static_cast<uint8_t>(value - 'a' + 10);
}

const std::array<uint8_t, kProgramDwords * sizeof(uint32_t)>& program_bytes() {
    static const auto bytes = [] {
        static_assert(sizeof(kProgramHex) - 1u == kProgramDwords * sizeof(uint32_t) * 2u);
        std::array<uint8_t, kProgramDwords * sizeof(uint32_t)> result{};
        for (size_t index = 0; index < result.size(); ++index)
            result[index] = static_cast<uint8_t>(
                nibble(kProgramHex[index * 2u]) << 4u | nibble(kProgramHex[index * 2u + 1u]));
        return result;
    }();
    return bytes;
}

uint32_t pointer_record_count(const ShaderResource& resource) {
    const uint64_t records = resource.size / kGta5PackedPointerSourceStride;
    return static_cast<uint32_t>((records + 63u) / 64u);
}

std::span<const uint32_t> pointer_records(const ShaderResource& resource) {
    return std::span<const uint32_t>(kRecords).first(pointer_record_count(resource));
}

const ShaderResource* unique_source(const ShaderResourceTable& table) {
    const ShaderResource* result = nullptr;
    for (const ShaderResource& resource : table.resources) {
        if (resource.fetch_pc != kGta5PackedPointerSourcePc) continue;
        if (result) return nullptr;
        result = &resource;
    }
    return result;
}

const ShaderResource* unique_resource_at(const ShaderResourceTable& table, uint32_t fetch_pc) {
    const ShaderResource* result = nullptr;
    for (const ShaderResource& resource : table.resources) {
        if (resource.fetch_pc != fetch_pc) continue;
        if (result) return nullptr;
        result = &resource;
    }
    return result;
}

ShaderResource* unique_resource_at(ShaderResourceTable& table, uint32_t fetch_pc) {
    return const_cast<ShaderResource*>(unique_resource_at(
        static_cast<const ShaderResourceTable&>(table), fetch_pc));
}

ShaderResource* unique_source(ShaderResourceTable& table) {
    return const_cast<ShaderResource*>(unique_source(
        static_cast<const ShaderResourceTable&>(table)));
}

bool source_shape(const ShaderResource& resource) {
    const bool complete_records = resource.size != 0u &&
        resource.size % kGta5PackedPointerSourceStride == 0u &&
        resource.size / kGta5PackedPointerSourceStride <= kGta5PackedPointerMaxThreads;
    return resource.cls == ResourceClass::ConstantBuffer &&
           resource.fetch_pc == kGta5PackedPointerSourcePc &&
           complete_records &&
           resource.stride == kGta5PackedPointerSourceStride &&
           resource.table_index_count == 0u && resource.srt_offset == UINT32_MAX &&
           resource.sgpr_base == UINT32_MAX && resource.gpu_addr > 0x10000u;
}

bool atomic_source_shape(const ShaderResource& resource, bool expanded) {
    const uint32_t expected_size = expanded ? kGta5PackedPointerAtomicBindingBytes
                                            : kGta5PackedPointerAtomicLoadBytes;
    const bool complete_host = resource.host_data &&
        resource.host_data_size >= kGta5PackedPointerAtomicBindingBytes;
    return resource.cls == ResourceClass::ConstantBuffer &&
           resource.fetch_pc == kGta5PackedPointerAtomicSourcePc &&
           resource.size == expected_size &&
           resource.stride == 0u && resource.table_index_count == 0u &&
           resource.srt_offset == UINT32_MAX && resource.sgpr_base == UINT32_MAX &&
           resource.gpu_addr > 0x10000u && (resource.gpu_addr & 7u) == 0u &&
           resource.scalar_raw_pointer_word_hi != UINT32_MAX &&
           (resource.scalar_raw_pointer_word_hi & 0xffffu) ==
               static_cast<uint32_t>(resource.gpu_addr >> 32u) &&
           (resource.scalar_raw_pointer_word_hi & kGta5PackedPointerRawWordHiMetadataMask) == 0u &&
           (complete_host ||
            (guest_readable(resource.gpu_addr, kGta5PackedPointerAtomicBindingBytes) &&
             guest_writable(resource.gpu_addr, kGta5PackedPointerAtomicBindingBytes)));
}

const uint8_t* complete_source_bytes(const ShaderResource& source) {
    if (source.host_data && source.host_data_size >= source.size) return source.host_data;
    return guest_readable(source.gpu_addr, source.size)
        ? reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(source.gpu_addr)) : nullptr;
}

bool parse_shadow(const ShaderResource& resource, const uint8_t* bytes, size_t byte_count,
                  uint32_t& slot_count, uint32_t* atomic_word_hi = nullptr) {
    if (!source_shape(resource)) return false;
    IndirectBufferShadowInfo info;
    if (!parse_indirect_buffer_shadow(
            resource, bytes, byte_count, kShadowLayout, pointer_records(resource), info) ||
        info.witness_words.size() != 2u ||
        info.witness_words[1] != (info.witness_words[0] ^ kGta5PackedPointerTag))
        return false;
    slot_count = info.slot_count;
    if (atomic_word_hi) *atomic_word_hi = info.witness_words[0];
    return true;
}

bool current_live_bytes_match(const ShaderResourceTable& table, const ShaderResource& source) {
    return current_indirect_buffer_shadow_matches(
        table, source, kShadowLayout, pointer_records(source));
}

} // namespace

bool rdna2_gta5_packed_pointer_shader(const uint32_t* code, size_t dwords) {
    return code && dwords >= kProgramDwords &&
           std::memcmp(code, program_bytes().data(), program_bytes().size()) == 0 &&
           code[kDescriptorStrideOrPc] == kDescriptorStrideOrPacket &&
           code[kDescriptorStrideOrPc + 1u] == kDescriptorStrideOrLiteral;
}

bool rdna2_gta5_packed_pointer_launch(
        const uint32_t* code, size_t dwords, const ComputeShaderConfig& config) {
    return rdna2_gta5_packed_pointer_shader(code, dwords) &&
           config.user_sgprs.size() == 14u &&
           config.local_x == 64u && config.local_y == 1u && config.local_z == 1u &&
           config.exact_thread_extent && config.threads_x >= 1u &&
           config.threads_x <= kGta5PackedPointerMaxThreads &&
           config.threads_y == 1u && config.threads_z == 1u &&
           config.wave_size == 64u && config.tgid_x_en &&
           !config.tgid_y_en && !config.tgid_z_en && !config.tg_size_en &&
           config.tidig_comp_cnt == 0u;
}

bool is_gta5_packed_pointer_marker_candidate(const ShaderResource& resource) {
    return resource.indirect_buffer_contract_tag != 0u ||
           resource.indirect_buffer_binding_bytes != 0u ||
           resource.indirect_buffer_slot_count != 0u ||
           resource.indirect_buffer_header_bytes != 0u ||
           resource.indirect_buffer_slot_bytes != 0u;
}

bool is_gta5_packed_pointer_serialized_shadow(
        const ShaderResource& resource, const uint8_t* bytes, size_t byte_count) {
    uint32_t slot_count = 0;
    return parse_shadow(resource, bytes, byte_count, slot_count);
}

bool is_gta5_packed_pointer_resource(const ShaderResource& resource) {
    uint32_t slot_count = 0;
    return resource.indirect_buffer_contract_tag == kGta5PackedPointerTag &&
           resource.indirect_buffer_binding_bytes == resource.host_data_size &&
           resource.indirect_buffer_slot_count != 0u &&
           resource.indirect_buffer_header_bytes == kGta5PackedPointerHeaderBytes &&
           resource.indirect_buffer_slot_bytes == kGta5PackedPointerSlotBytes &&
           parse_shadow(resource, resource.host_data, resource.host_data_size, slot_count) &&
           slot_count == resource.indirect_buffer_slot_count;
}

bool discover_rdna2_gta5_packed_pointer(
        const uint32_t* code, size_t dwords, const ComputeShaderConfig& config,
    ShaderResourceTable& resources) {
    for (ShaderResource& resource : resources.resources) {
        resource.indirect_buffer_contract_tag = 0u;
        resource.indirect_buffer_binding_bytes = 0u;
        resource.indirect_buffer_slot_count = 0u;
        resource.indirect_buffer_header_bytes = 0u;
        resource.indirect_buffer_slot_bytes = 0u;
    }
    if (!rdna2_gta5_packed_pointer_launch(code, dwords, config)) return false;
    ShaderResource* source = unique_source(resources);
    if (!source || !source_shape(*source) ||
        source->size != static_cast<uint64_t>(config.threads_x) *
            kGta5PackedPointerSourceStride)
        return false;

    uint32_t replay_slots = 0;
    uint32_t replay_atomic_word_hi = UINT32_MAX;
    if (parse_shadow(*source, source->host_data, source->host_data_size,
                     replay_slots, &replay_atomic_word_hi)) {
        ShaderResource* atomic_source = unique_resource_at(
            resources, kGta5PackedPointerAtomicSourcePc);
        if (!atomic_source) return false;
        const uint32_t prior_atomic_word_hi = atomic_source->scalar_raw_pointer_word_hi;
        const uint64_t prior_atomic_size = atomic_source->size;
        atomic_source->scalar_raw_pointer_word_hi = replay_atomic_word_hi;
        if (!atomic_source_shape(*atomic_source, false) &&
            !atomic_source_shape(*atomic_source, true)) {
            atomic_source->scalar_raw_pointer_word_hi = prior_atomic_word_hi;
            return false;
        }
        atomic_source->size = kGta5PackedPointerAtomicBindingBytes;
        source->indirect_buffer_contract_tag = kGta5PackedPointerTag;
        source->indirect_buffer_binding_bytes = static_cast<uint32_t>(source->host_data_size);
        source->indirect_buffer_slot_count = replay_slots;
        source->indirect_buffer_header_bytes = kGta5PackedPointerHeaderBytes;
        source->indirect_buffer_slot_bytes = kGta5PackedPointerSlotBytes;
        if (rdna2_gta5_packed_pointer_dispatch(code, dwords, config, resources)) return true;
        source->indirect_buffer_contract_tag = 0u;
        source->indirect_buffer_binding_bytes = 0u;
        source->indirect_buffer_slot_count = 0u;
        source->indirect_buffer_header_bytes = 0u;
        source->indirect_buffer_slot_bytes = 0u;
        atomic_source->scalar_raw_pointer_word_hi = prior_atomic_word_hi;
        atomic_source->size = prior_atomic_size;
        return false;
    }

    const uint8_t* original = complete_source_bytes(*source);
    if (!original) return false;
    ShaderResource* atomic_source = unique_resource_at(
        resources, kGta5PackedPointerAtomicSourcePc);
    if (!atomic_source || !atomic_source_shape(*atomic_source, false)) return false;
    const uint32_t atomic_word_hi = atomic_source->scalar_raw_pointer_word_hi;
    const std::array<uint32_t, 2> witnesses{
        atomic_word_hi, atomic_word_hi ^ kGta5PackedPointerTag};
    std::shared_ptr<std::vector<uint8_t>> owner;
    uint32_t slot_count = 0;
    if (!build_indirect_buffer_shadow(
            *source, original, kShadowLayout, pointer_records(*source), witnesses,
            owner, slot_count))
        return false;

    const uint8_t* prior_host_data = source->host_data;
    const uint64_t prior_host_data_size = source->host_data_size;
    source->host_data = owner->data();
    source->host_data_size = owner->size();
    source->indirect_buffer_contract_tag = kGta5PackedPointerTag;
    source->indirect_buffer_binding_bytes = static_cast<uint32_t>(owner->size());
    source->indirect_buffer_slot_count = slot_count;
    source->indirect_buffer_header_bytes = kGta5PackedPointerHeaderBytes;
    source->indirect_buffer_slot_bytes = kGta5PackedPointerSlotBytes;
    const uint64_t prior_atomic_size = atomic_source->size;
    atomic_source->size = kGta5PackedPointerAtomicBindingBytes;
    resources.owned_host_data.push_back(std::move(owner));
    if (rdna2_gta5_packed_pointer_dispatch(code, dwords, config, resources)) return true;
    resources.owned_host_data.pop_back();
    source->host_data = const_cast<uint8_t*>(prior_host_data);
    source->host_data_size = prior_host_data_size;
    source->indirect_buffer_contract_tag = 0u;
    source->indirect_buffer_binding_bytes = 0u;
    source->indirect_buffer_slot_count = 0u;
    source->indirect_buffer_header_bytes = 0u;
    source->indirect_buffer_slot_bytes = 0u;
    atomic_source->size = prior_atomic_size;
    return false;
}

bool rdna2_gta5_packed_pointer_dispatch(
        const uint32_t* code, size_t dwords, const ComputeShaderConfig& config,
        const ShaderResourceTable& resources) {
    if (!rdna2_gta5_packed_pointer_launch(code, dwords, config)) return false;
    const ShaderResource* source = unique_source(resources);
    const ShaderResource* atomic_source = unique_resource_at(
        resources, kGta5PackedPointerAtomicSourcePc);
    if (!source || source->size != static_cast<uint64_t>(config.threads_x) *
            kGta5PackedPointerSourceStride ||
        !is_gta5_packed_pointer_resource(*source) ||
        !atomic_source || !atomic_source_shape(*atomic_source, true) ||
        !config.storage_buffer_int64_atomics)
        return false;
    size_t markers = 0;
    for (const ShaderResource& resource : resources.resources) {
        if (!is_gta5_packed_pointer_marker_candidate(resource)) continue;
        if (&resource != source || !is_gta5_packed_pointer_resource(resource)) return false;
        ++markers;
    }
    return markers == 1u && current_live_bytes_match(resources, *source);
}

bool rdna2_gta5_packed_pointer_access(
        const Rdna2Inst& instruction, IndirectBufferShadowAccess& access) {
    struct Site { uint32_t pc, word0, word1, offset, components; };
    static constexpr std::array<Site, 17> sites{{
        {70u, 0xdc308018u, 0x037d0000u, 24u, 1u},
        {79u, 0xdc3080fcu, 0x037d0000u, 252u, 1u},
        {86u, 0xdc388110u, 0x0d7d0000u, 272u, 4u},
        {88u, 0xdc388120u, 0x117d0000u, 288u, 4u},
        {90u, 0xdc388130u, 0x157d0000u, 304u, 4u},
        {92u, 0xdc388140u, 0x047d0000u, 320u, 4u},
        {94u, 0xdc388150u, 0x097d0000u, 336u, 4u},
        {97u, 0xdc388160u, 0x007d0000u, 352u, 4u},
        {139u, 0xdc308018u, 0x037d0000u, 24u, 1u},
        {148u, 0xdc3080fcu, 0x037d0000u, 252u, 1u},
        {155u, 0xdc308018u, 0x037d0000u, 24u, 1u},
        {157u, 0xdc388110u, 0x187d0000u, 272u, 4u},
        {159u, 0xdc388130u, 0x107d0000u, 304u, 4u},
        {161u, 0xdc388140u, 0x147d0000u, 320u, 4u},
        {163u, 0xdc388150u, 0x087d0000u, 336u, 4u},
        {165u, 0xdc388160u, 0x0c7d0000u, 352u, 4u},
        {175u, 0xdc388120u, 0x007d0000u, 288u, 4u},
    }};
    const auto found = std::find_if(sites.begin(), sites.end(), [&](const Site& site) {
        return instruction.pc == site.pc && instruction.len_dwords == 2u &&
               instruction.words[0] == site.word0 && instruction.words[1] == site.word1;
    });
    if (found == sites.end()) return false;
    access = {found->offset, found->components};
    return true;
}

bool rdna2_gta5_packed_pointer_atomic_site(const Rdna2Inst& instruction) {
    // pc347 is S_OR_B32 s1,s5,0x01000000: it forces descriptor stride bit 8, hence a 256-byte
    // stride. pc349..351 finish {base, stride=256, NUM_RECORDS=1, 0x00016204}, so this qword at
    // byte 24 is in bounds and must execute as one real 64-bit RMW. Full-program identity makes
    // the producer algebra part of the authority boundary; this helper recognizes the consumer.
    return instruction.pc == 355u && instruction.fmt == Rdna2Format::MUBUF &&
           instruction.opcode == kMubufOpcodeAtomicOrX2 && instruction.len_dwords == 2u &&
           instruction.words[0] == 0xe1680018u && instruction.words[1] == 0x80000000u &&
           !instruction.mubuf_glc && !instruction.mubuf_dlc &&
           !instruction.mubuf_lds && !instruction.mubuf_tfe;
}

} // namespace prosper::gpu
