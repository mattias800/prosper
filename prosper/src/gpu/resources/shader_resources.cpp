// shader_resources.cpp — see shader_resources.hpp. Pure lookups + format sizing; no Vulkan, no state.
#include "gpu/resources/shader_resources.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string_view>
#include <unordered_map>

namespace prosper::gpu {

uint32_t data_format_bytes(DataFormat f) {
    switch (f) {
        case DataFormat::Float32: case DataFormat::Uint32: case DataFormat::Sint32: return 4;
        case DataFormat::Float16: case DataFormat::Unorm16: case DataFormat::Snorm16:
        case DataFormat::Uint16:  case DataFormat::Sint16:
        case DataFormat::Uscaled16: case DataFormat::Sscaled16: return 2;
        case DataFormat::Unorm8:  case DataFormat::Snorm8:
        case DataFormat::Uint8:   case DataFormat::Sint8:
        case DataFormat::Uscaled8: case DataFormat::Sscaled8: return 1;
        default: return 0;
    }
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t man  = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign << 31;                          // +/- zero
        else {                                                    // subnormal: normalize into f32
            uint32_t e = 0, m = man;                              // value = man * 2^-24; after k shifts
            while (!(m & 0x400u)) { m <<= 1; e++; }               // it's 1.frac * 2^(-14-k), so the f32
            m &= 0x3FFu;                                          // exponent field is 127-14-k = 113-k
            bits = (sign << 31) | ((113u - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | 0x7F800000u | (man << 13);          // inf / NaN (payload preserved)
    } else {
        bits = (sign << 31) | ((exp + 112u) << 23) | (man << 13); // normal: rebias 15 -> 127
    }
    float f;
    static_assert(sizeof(f) == sizeof(bits), "float is 32-bit");
    __builtin_memcpy(&f, &bits, sizeof f);
    return f;
}

uint16_t float_to_half(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        if (!mantissa) return static_cast<uint16_t>(sign | 0x7c00u);
        uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
        if (!payload) payload = 1;
        return static_cast<uint16_t>(sign | 0x7c00u | payload);
    }

    const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        mantissa |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) rounded++;
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t rounded = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) rounded++;
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exponent) << 10) | rounded);
}

uint8_t unorm16_to_unorm8(uint16_t value) {
    // Scale the complete normalized code rather than selecting either endian byte. Adding half the
    // source denominator implements nearest rounding while preserving both endpoints exactly.
    return static_cast<uint8_t>((static_cast<uint32_t>(value) * 255u + 32767u) / 65535u);
}

float f11_to_float(uint16_t v) {
    // 5-bit exp (bias 15) + 6-bit mantissa, unsigned. Widen the mantissa into a half's 10-bit
    // field: subnormal m/64*2^-14 == (m<<4)/1024*2^-14, normal/inf/NaN carry the exponent as-is.
    return half_to_float((uint16_t)(((v & 0x7FFu) >> 6 << 10) | ((v & 0x3Fu) << 4)));
}

float f10_to_float(uint16_t v) {
    // 5-bit exp (bias 15) + 5-bit mantissa, unsigned.
    return half_to_float((uint16_t)(((v & 0x3FFu) >> 5 << 10) | ((v & 0x1Fu) << 5)));
}

namespace {

uint32_t round_shift_even(uint32_t value, uint32_t shift) {
    if (!shift) return value;
    // value is at most 24 bits here. For shifts above that it is strictly below halfway.
    if (shift >= 32) return 0;
    uint32_t rounded = value >> shift;
    const uint32_t remainder = value & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
    return rounded;
}

uint16_t float_to_unsigned_small(float f, uint32_t mantissa_bits) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = bits >> 31;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    const uint32_t mantissa_mask = (1u << mantissa_bits) - 1u;

    // Both formats use the binary16 exponent (5 bits, bias 15), but have no sign bit.
    if (exponent == 0xffu) {
        if (!mantissa) return static_cast<uint16_t>(0x1fu << mantissa_bits); // +inf
        uint32_t payload = mantissa >> (23u - mantissa_bits);
        if (!payload) payload = 1;
        return static_cast<uint16_t>((0x1fu << mantissa_bits) | payload);    // NaN
    }
    if (sign || exponent == 0) return 0; // negative values clamp to +0; f32 subnormals underflow

    int32_t target_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (target_exponent >= 31)
        return static_cast<uint16_t>(0x1fu << mantissa_bits);                // overflow -> +inf

    const uint32_t significand = 0x800000u | mantissa;
    if (target_exponent <= 0) {
        // Target subnormal unit is 2^(-14-mantissa_bits). Express the exact f32 significand in
        // those units, then perform one round-to-nearest-even step (no double rounding via f16).
        const int32_t unbiased = static_cast<int32_t>(exponent) - 127;
        const uint32_t shift = static_cast<uint32_t>(9 - static_cast<int32_t>(mantissa_bits) - unbiased);
        const uint32_t rounded = round_shift_even(significand, shift);
        // Rounding the largest subnormal upward produces the smallest normal (exp=1, mantissa=0).
        if (rounded >= (1u << mantissa_bits))
            return static_cast<uint16_t>(1u << mantissa_bits);
        return static_cast<uint16_t>(rounded);
    }

    uint32_t rounded = round_shift_even(significand, 23u - mantissa_bits);
    if (rounded == (1u << (mantissa_bits + 1u))) {
        rounded >>= 1;
        if (++target_exponent >= 31)
            return static_cast<uint16_t>(0x1fu << mantissa_bits);
    }
    return static_cast<uint16_t>((static_cast<uint32_t>(target_exponent) << mantissa_bits) |
                                 (rounded & mantissa_mask));
}

} // namespace

uint16_t float_to_f11(float f) { return float_to_unsigned_small(f, 6); }
uint16_t float_to_f10(float f) { return float_to_unsigned_small(f, 5); }

void unorm2_10_10_10_to_rgba8(uint32_t packed, uint8_t rgba[4]) {
    // Mesa's GFX10 format table maps a logical R10G10B10A2 description to hardware IMG_FMT 50,
    // whose name lists the packed fields high-to-low. Scale with integer rounding so both endpoints
    // and intermediate UNORM values match a normalized hardware sample as closely as RGBA8 permits.
    rgba[0] = (uint8_t)((((packed >>  0) & 0x3FFu) * 255u + 511u) / 1023u);
    rgba[1] = (uint8_t)((((packed >> 10) & 0x3FFu) * 255u + 511u) / 1023u);
    rgba[2] = (uint8_t)((((packed >> 20) & 0x3FFu) * 255u + 511u) / 1023u);
    rgba[3] = (uint8_t)(((packed >> 30) & 0x3u) * 85u);
}

bool valid_shader_buffer_table_contract(const ShaderResource& resource) {
    constexpr uint32_t kMaxDescriptorArrayEntries = 4096u;
    const bool array = resource.table_index_count != 0u;
    if (!array) {
        return resource.table_entry_stride == 0u &&
               resource.table_index_sgpr == UINT32_MAX &&
               resource.table_entries.empty() &&
               resource.table_selector_mode == BufferTableSelectorMode::None &&
               resource.table_load_pc == UINT32_MAX;
    }
    if ((resource.cls != ResourceClass::ConstantBuffer &&
         resource.cls != ResourceClass::VertexBuffer) ||
        resource.table_index_count > kMaxDescriptorArrayEntries ||
        resource.table_entries.size() != resource.table_index_count ||
        resource.table_entry_stride < 16u || (resource.table_entry_stride & 3u))
        return false;

    switch (resource.table_selector_mode) {
        case BufferTableSelectorMode::UserSgprIndex:
            if (resource.table_index_sgpr == UINT32_MAX ||
                resource.table_load_pc != UINT32_MAX)
                return false;
            break;
        case BufferTableSelectorMode::DynamicSbufferByteOffset:
            if (resource.table_index_sgpr != UINT32_MAX ||
                resource.table_load_pc == UINT32_MAX)
                return false;
            break;
        case BufferTableSelectorMode::None:
        default:
            return false;
    }

    // DST_SEL is a FORMAT-fetch control: it names which component each returned channel takes. The
    // dynamic-scalar-offset selection is admitted for untyped loads only — the emitter rejects a
    // typed fetch, store or atomic through a selected descriptor — and an untyped load moves raw
    // dwords regardless of the descriptor's declared swizzle. So for that mode the swizzle cannot
    // decide whether the table is bindable, while the user-SGPR mode, whose consumers may be typed,
    // keeps requiring the identity mapping. GTA V's tables are three-component (0x3ac, X/Y/Z/1);
    // requiring 0xfac rejected every one of them for a field none of their consumers reads (#2481).
    const bool raw_only_selection =
        resource.table_selector_mode == BufferTableSelectorMode::DynamicSbufferByteOffset;
    for (const ShaderBufferTableEntry& entry : resource.table_entries) {
        const uint64_t decoded_addr =
            (static_cast<uint64_t>(entry.vsharp[0]) |
             (static_cast<uint64_t>(entry.vsharp[1]) << 32u)) &
            0x0000FFFFFFFFFFFFull;
        const uint32_t decoded_stride = (entry.vsharp[1] >> 16u) & 0x3fffu;
        const uint64_t decoded_size64 = decoded_stride
            ? static_cast<uint64_t>(entry.vsharp[2]) * decoded_stride
            : static_cast<uint64_t>(entry.vsharp[2]);
        const uint32_t decoded_size = decoded_size64 > UINT32_MAX
            ? UINT32_MAX : static_cast<uint32_t>(decoded_size64);
        const bool null_descriptor = entry.gpu_addr == 0u && entry.size == 0u &&
                                     entry.host_data == nullptr;
        const uint32_t dst_sel = entry.vsharp[3] & 0xfffu;
        DataFormat decoded_format = DataFormat::Unknown;
        uint32_t decoded_components = 0;
        rdna2_buffer_format((entry.vsharp[3] >> 12u) & 0x7fu,
                            &decoded_format, &decoded_components);
        // The normalized entry retains none of swizzled addressing, INDEX_STRIDE, ADD_TID,
        // RESOURCE_LEVEL, OOB_SELECT, TYPE, or the reserved controls. Admit only the conventional
        // linear zero-valued form; otherwise the Vulkan backing would not reproduce the raw V#.
        if ((entry.vsharp[1] & 0xc0000000u) != 0u ||
            (entry.vsharp[3] & 0xfff80000u) != 0u ||
            entry.gpu_addr != decoded_addr ||
            entry.stride != decoded_stride || entry.size != decoded_size ||
            (!null_descriptor &&
             (entry.stride != resource.stride ||
              decoded_format != resource.format ||
              decoded_components != resource.num_components ||
              (dst_sel != 0xfacu && !raw_only_selection))) ||
            (null_descriptor && dst_sel != 0u && dst_sel != 0xfacu) ||
            (!entry.host_data && entry.host_data_size != 0u) ||
            (entry.host_data && entry.host_data_size < entry.size))
            return false;
    }
    return true;
}

const ShaderResource* ShaderResourceTable::by_srt_offset(uint32_t srt_offset) const {
    if (srt_offset == 0xFFFFFFFFu) return nullptr;
    for (const auto& r : resources) if (r.srt_offset == srt_offset) return &r;
    return nullptr;
}

const ShaderResource* ShaderResourceTable::by_sgpr_base(uint32_t sgpr) const {
    if (sgpr == 0xFFFFFFFFu) return nullptr;
    for (const auto& r : resources) if (r.sgpr_base == sgpr) return &r;
    return nullptr;
}

const ShaderResource* ShaderResourceTable::by_sgpr_base_cls(uint32_t sgpr, ResourceClass cls) const {
    if (sgpr == 0xFFFFFFFFu) return nullptr;
    for (const auto& r : resources) if (r.sgpr_base == sgpr && r.cls == cls) return &r;
    return nullptr;
}

const ShaderResource* ShaderResourceTable::by_fetch_pc(uint32_t pc) const {
    if (pc == 0xFFFFFFFFu) return nullptr;
    for (const auto& r : resources) if (r.fetch_pc == pc) return &r;
    return nullptr;
}

const ShaderResource* ShaderResourceTable::image_by_srt_offset(
        uint32_t srt_offset, ImageResourceRequirement requirement) const {
    if (srt_offset == 0xFFFFFFFFu) return nullptr;
    for (const auto& r : resources)
        if (r.srt_offset == srt_offset && image_resource_class_satisfies(r.cls, requirement))
            return &r;
    return nullptr;
}

const ShaderResource* ShaderResourceTable::image_by_sgpr_base(
        uint32_t sgpr, ImageResourceRequirement requirement) const {
    if (sgpr == 0xFFFFFFFFu) return nullptr;
    for (const auto& r : resources)
        if (r.sgpr_base == sgpr && image_resource_class_satisfies(r.cls, requirement))
            return &r;
    return nullptr;
}

const ShaderResource* ShaderResourceTable::image_by_fetch_pc(
        uint32_t pc, ImageResourceRequirement requirement) const {
    if (pc == 0xFFFFFFFFu) return nullptr;
    for (const auto& r : resources)
        if (r.fetch_pc == pc && image_resource_class_satisfies(r.cls, requirement))
            return &r;
    return nullptr;
}

const ShaderResource* ShaderResourceTable::by_binding(uint32_t binding) const {
    for (const auto& r : resources) if (r.binding == binding) return &r;
    return nullptr;
}

namespace {

constexpr uint32_t kSpirvMagic = 0x07230203u;
enum : uint32_t {
    OpEntryPoint = 15,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeImage = 25,
    OpTypeSampledImage = 27,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpConstant = 43,
    OpVariable = 59,
    OpImageTexelPointer = 60,
    OpLoad = 61,
    OpStore = 62,
    OpAccessChain = 65,
    OpInBoundsAccessChain = 66,
    OpAtomicLoad = 227,
    OpAtomicStore = 228,
    OpAtomicExchange = 229,
    OpAtomicCompareExchange = 230,
    OpAtomicCompareExchangeWeak = 231,
    OpAtomicIIncrement = 232,
    OpAtomicIDecrement = 233,
    OpAtomicIAdd = 234,
    OpAtomicXor = 242,
    OpAtomicFlagTestAndSet = 318,
    OpAtomicFlagClear = 319,
    OpDecorate = 71,
    OpMemberDecorate = 72,
    OpModuleProcessed = 330,
};
enum : uint32_t {
    StorageUniformConstant = 0,
    StorageImage = 11,
    StorageBuffer = 12,
    DecorationArrayStride = 6,
    DecorationBinding = 33,
    DecorationDescriptorSet = 34,
    DecorationOffset = 35,
};

struct Instruction {
    uint32_t offset = 0;
    uint16_t words = 0;
    uint16_t opcode = 0;
};

struct StorageBufferZeroPadMarker {
    uint32_t set = 0;
    uint32_t binding = 0;
    uint64_t logical_bytes = 0;
    uint64_t binding_bytes = 0;
    StorageBufferTailSemantic semantic = StorageBufferTailSemantic::None;
};

std::string instruction_string(const std::vector<uint32_t>& spirv,
                               const Instruction& instruction) {
    std::string value;
    bool terminated = false;
    for (uint32_t operand = 0; operand + 1u < instruction.words; ++operand) {
        const uint32_t packed = spirv[instruction.offset + 1u + operand];
        for (uint32_t byte = 0; byte < 4; ++byte) {
            const char c = static_cast<char>(packed >> (byte * 8u));
            if (!c) {
                terminated = true;
                break;
            }
            value.push_back(c);
        }
        if (terminated) break;
    }
    return terminated ? value : std::string{};
}

bool parse_storage_buffer_zero_pad_marker(const std::string& text,
                                          StorageBufferZeroPadMarker& marker) {
    constexpr const char kPrefix[] = "Prosper.StorageBufferZeroPad=";
    if (!text.starts_with(kPrefix)) return false;
    const char* cursor = text.data() + sizeof(kPrefix) - 1u;
    const char* end = text.data() + text.size();
    auto parse = [&](auto& value, char delimiter) {
        const auto result = std::from_chars(cursor, end, value);
        if (result.ec != std::errc{} || result.ptr == cursor) return false;
        cursor = result.ptr;
        if (delimiter) {
            if (cursor == end || *cursor != delimiter) return false;
            ++cursor;
        }
        return true;
    };
    if (!parse(marker.set, ',') || !parse(marker.binding, ',') ||
        !parse(marker.logical_bytes, ',') || !parse(marker.binding_bytes, ','))
        return false;
    const std::string_view token(cursor, static_cast<size_t>(end - cursor));
    if (token == "u16") marker.semantic = StorageBufferTailSemantic::Uint16;
    else if (token == "f16") marker.semantic = StorageBufferTailSemantic::Float16;
    else return false;
    return true;
}

enum class TypeKind {
    Unknown,
    Scalar,
    Vector,
    Image,
    SampledImage,
    Array,
    RuntimeArray,
    Struct,
    Pointer,
};

struct TypeInfo {
    TypeKind kind = TypeKind::Unknown;
    uint32_t element = 0;
    uint32_t count_id = 0;
    uint32_t count = 0;
    uint32_t width = 0;
    uint32_t storage = 0;
    uint32_t image_sampled = 0;
    uint32_t image_format = 0;
    uint32_t image_dim = 0xFFFFFFFFu;
    bool image_arrayed = false;
    bool image_depth = false;
    bool image_multisampled = false;
    bool scalar_float = false;
    bool scalar_integer = false;
    bool scalar_signed = false;
    std::vector<uint32_t> members;
};

struct VariableInfo {
    uint32_t pointer_type = 0;
    uint32_t storage = 0;
};

struct PointerAccess {
    uint32_t variable = 0;
    uint32_t pointee_type = 0;
    uint64_t offset = 0;
    bool dynamic = false;
};

uint64_t member_key(uint32_t type, uint32_t member) {
    return (static_cast<uint64_t>(type) << 32) | member;
}

uint64_t type_size(uint32_t id,
                   const std::unordered_map<uint32_t, TypeInfo>& types,
                   const std::unordered_map<uint32_t, uint64_t>& array_strides,
                   const std::unordered_map<uint64_t, uint64_t>& member_offsets,
                   uint32_t depth = 0) {
    if (!id || depth > 16) return 0;
    auto it = types.find(id);
    if (it == types.end()) return 0;
    const TypeInfo& t = it->second;
    switch (t.kind) {
        case TypeKind::Scalar:
            return t.width ? (t.width + 7u) / 8u : 0;
        case TypeKind::Vector:
            return type_size(t.element, types, array_strides, member_offsets, depth + 1) * t.count;
        case TypeKind::Array: {
            uint64_t stride = type_size(t.element, types, array_strides, member_offsets, depth + 1);
            auto si = array_strides.find(id); if (si != array_strides.end()) stride = si->second;
            return stride * t.count;
        }
        case TypeKind::Struct: {
            uint64_t end = 0, sequential = 0;
            for (uint32_t i = 0; i < t.members.size(); ++i) {
                uint64_t off = sequential;
                auto oi = member_offsets.find(member_key(id, i));
                if (oi != member_offsets.end()) off = oi->second;
                uint64_t size = type_size(t.members[i], types, array_strides, member_offsets, depth + 1);
                end = std::max(end, off + size); sequential = off + size;
            }
            return end;
        }
        case TypeKind::Pointer:
            return type_size(t.element, types, array_strides, member_offsets, depth + 1);
        default:
            return 0;
    }
}

SpirvDescriptorKind descriptor_kind(const VariableInfo& var,
                                    const std::unordered_map<uint32_t, TypeInfo>& types) {
    auto pi = types.find(var.pointer_type);
    if (pi == types.end() || pi->second.kind != TypeKind::Pointer) return SpirvDescriptorKind::Unknown;
    uint32_t type = pi->second.element;
    for (uint32_t depth = 0; depth < 8; ++depth) {
        auto ti = types.find(type); if (ti == types.end()) return SpirvDescriptorKind::Unknown;
        if (ti->second.kind == TypeKind::Array) { type = ti->second.element; continue; }
        if (var.storage == StorageBuffer) return SpirvDescriptorKind::StorageBuffer;
        if (var.storage != StorageUniformConstant) return SpirvDescriptorKind::Unknown;
        if (ti->second.kind == TypeKind::SampledImage) return SpirvDescriptorKind::CombinedImageSampler;
        if (ti->second.kind == TypeKind::Image)
            return ti->second.image_sampled == 2 ? SpirvDescriptorKind::StorageImage
                                                 : SpirvDescriptorKind::CombinedImageSampler;
        return SpirvDescriptorKind::Unknown;
    }
    return SpirvDescriptorKind::Unknown;
}

// How many descriptors this ONE binding declares (#2412 stage 4). 1 for an ordinary binding, N for
// `OpTypeArray` of descriptors with a resolved length, 0 for `OpTypeRuntimeArray` (length supplied at
// bind time).
//
// Deliberately conservative, because every existing shader must keep reporting 1 and a wrong answer
// here now REJECTS a draw (stage 3 turned an arity mismatch into an error). Two guards:
//
//   - Only the OUTERMOST wrapper counts. `descriptor_kind` walks *through* arrays to find the
//     underlying image or block, so it cannot distinguish "an array of eight buffers" from "one buffer
//     whose block contains an array" -- and those are entirely different bindings. A storage buffer's
//     block almost always contains a runtime array (that is the buffer's data), so anything deeper than
//     one level is the buffer's own contents and not a descriptor count.
//   - The wrapper's element must itself be descriptor-shaped (Struct, Image, SampledImage). A pointer
//     to a bare array of scalars is not a descriptor array, and reading it as one would report an
//     arity that rejects a binding that has always worked.
//
// An `OpTypeArray` whose length did not resolve reports `kDescriptorArityUnknown`, NOT 0 and not 1.
// All three were considered and the first two are wrong in opposite directions: 1 would silently bind
// element 0 of an array whose size we could not read, and 0 means `OpTypeRuntimeArray` -- which
// validation treats as compatible with any table size, so a decode failure would become the most
// permissive answer available. Unknown must be the least permissive, because it is the case we know
// least about; validation rejects it regardless of what the table supplies.
uint32_t descriptor_array_arity(const VariableInfo& var,
                                const std::unordered_map<uint32_t, TypeInfo>& types) {
    auto pi = types.find(var.pointer_type);
    if (pi == types.end() || pi->second.kind != TypeKind::Pointer) return 1;
    auto outer = types.find(pi->second.element);
    if (outer == types.end()) return 1;
    if (outer->second.kind != TypeKind::Array && outer->second.kind != TypeKind::RuntimeArray)
        return 1;
    auto elem = types.find(outer->second.element);
    if (elem == types.end()) return 1;
    if (elem->second.kind != TypeKind::Struct && elem->second.kind != TypeKind::Image &&
        elem->second.kind != TypeKind::SampledImage)
        return 1;
    if (outer->second.kind == TypeKind::RuntimeArray) return 0;
    // No `count ? count : 0` no-op here: an unresolved length is its own answer. `count` stays 0 when
    // the length id resolved to no `OpConstant` in this pass -- an `OpSpecConstant`, or a constant this
    // pass does not decode.
    if (outer->second.count == 0) return kDescriptorArityUnknown;
    return outer->second.count;
}

const TypeInfo* descriptor_image_type(const VariableInfo& var,
                                      const std::unordered_map<uint32_t, TypeInfo>& types) {
    auto pi = types.find(var.pointer_type);
    if (pi == types.end() || pi->second.kind != TypeKind::Pointer) return nullptr;
    uint32_t type = pi->second.element;
    for (uint32_t depth = 0; depth < 8; ++depth) {
        auto ti = types.find(type);
        if (ti == types.end()) return nullptr;
        if (ti->second.kind == TypeKind::Array || ti->second.kind == TypeKind::SampledImage) {
            type = ti->second.element;
            continue;
        }
        return ti->second.kind == TypeKind::Image ? &ti->second : nullptr;
    }
    return nullptr;
}

SpirvImageNumericClass descriptor_image_numeric_class(
    const VariableInfo& var, const std::unordered_map<uint32_t, TypeInfo>& types) {
    const TypeInfo* image = descriptor_image_type(var, types);
    if (!image) return SpirvImageNumericClass::Unknown;
    const auto sampled_type = types.find(image->element);
    if (sampled_type == types.end() || sampled_type->second.kind != TypeKind::Scalar)
        return SpirvImageNumericClass::Unknown;
    if (sampled_type->second.scalar_float) return SpirvImageNumericClass::Float;
    if (sampled_type->second.scalar_integer)
        return sampled_type->second.scalar_signed
            ? SpirvImageNumericClass::Sint : SpirvImageNumericClass::Uint;
    return SpirvImageNumericClass::Unknown;
}

SpirvDescriptorKind resource_kind(const ShaderResource& r) {
    switch (r.cls) {
        case ResourceClass::ConstantBuffer:
        case ResourceClass::VertexBuffer: return SpirvDescriptorKind::StorageBuffer;
        case ResourceClass::Texture:      return SpirvDescriptorKind::CombinedImageSampler;
        case ResourceClass::StorageImage: return SpirvDescriptorKind::StorageImage;
        default:                          return SpirvDescriptorKind::Unknown;
    }
}

void add_malformed(DescriptorValidationReport& report) {
    report.issues.push_back({DescriptorIssueCode::MalformedSpirv, true});
}

struct DescriptorReflectionCacheEntry {
    std::vector<uint32_t> spirv;
    DescriptorValidationReport reflection;
    SpirvShaderStage stage = SpirvShaderStage::Unknown;
    uint64_t last_use = 0;
};

struct DescriptorReflectionCache {
    std::mutex mutex;
    std::unordered_map<uint64_t, DescriptorReflectionCacheEntry> entries;
    uint64_t use_clock = 0;
    uint64_t bytes = 0;
};

DescriptorReflectionCache& descriptor_reflection_cache() {
    static DescriptorReflectionCache cache;
    return cache;
}

uint64_t spirv_reflection_hash(const std::vector<uint32_t>& spirv) {
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t word : spirv) {
        hash ^= word;
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t reflection_entry_bytes(const DescriptorReflectionCacheEntry& entry) {
    return static_cast<uint64_t>(entry.spirv.size()) * sizeof(uint32_t) +
           static_cast<uint64_t>(entry.reflection.descriptors.size()) *
               sizeof(SpirvDescriptorBinding) +
           static_cast<uint64_t>(entry.reflection.issues.size()) *
               sizeof(DescriptorValidationIssue);
}

} // namespace

uint64_t shader_resource_buffer_binding_bytes(const ShaderResource& resource) {
    if (!resource.scalar_buffer_dword_count) return resource.size;

    // This metadata authenticates the emitted per-component scalar bound. Accept only the normalized
    // shape produced for a real bounded S_BUFFER consumer; a manually-built table or a hostile
    // capture must not reinterpret an unrelated generic constant-buffer load by setting one integer.
    if (resource.cls != ResourceClass::ConstantBuffer || resource.gpu_addr == 0u ||
        resource.table_index_count != 0u ||
        (resource.srt_offset == UINT32_MAX && resource.sgpr_base == UINT32_MAX &&
         resource.fetch_pc == UINT32_MAX))
        return 0u;

    constexpr uint64_t kMaximumScalarBufferBytes = 0x10000000ull;
    const uint64_t scalar_bytes =
        static_cast<uint64_t>(resource.scalar_buffer_dword_count) * sizeof(uint32_t);
    if (!scalar_bytes || scalar_bytes > kMaximumScalarBufferBytes ||
        resource.size > kMaximumScalarBufferBytes)
        return 0u;

    // #2528: the scalar span IS the descriptor's byte footprint, so it can never exceed `size` and
    // this function never widens a binding. `size` authenticates the carried dword count exactly,
    // for strided and unstrided V#s alike — the two no longer need separate cross-checks, because
    // there is no longer a second bound for them to disagree about.
    if (resource.size / sizeof(uint32_t) != resource.scalar_buffer_dword_count)
        return 0u;
    return resource.size;
}

StorageBufferMaterializationPlan plan_storage_buffer_materialization(
    const SpirvDescriptorBinding& descriptor,
    const ShaderResource& resource) {
    StorageBufferMaterializationPlan plan;
    plan.logical_bytes = resource.size;
    plan.binding_bytes = resource.size;

    if (resource.scalar_buffer_dword_count) {
        const uint64_t scalar_bytes = shader_resource_buffer_binding_bytes(resource);
        if (!scalar_bytes || descriptor.kind != SpirvDescriptorKind::StorageBuffer ||
            !descriptor.readable || descriptor.writable || descriptor.atomic_access)
            return plan;
        plan.logical_bytes = scalar_bytes;
        plan.binding_bytes = scalar_bytes;
        plan.valid = true;
        return plan;
    }

    if (is_gta5_packed_pointer_marker_candidate(resource)) {
        if (!is_gta5_packed_pointer_resource(resource) ||
            descriptor.kind != SpirvDescriptorKind::StorageBuffer ||
            !descriptor.readable || descriptor.writable || descriptor.atomic_access ||
            descriptor.required_bytes > resource.indirect_buffer_binding_bytes)
            return plan;
        plan.binding_bytes = resource.indirect_buffer_binding_bytes;
        plan.valid = true;
        return plan;
    }

    if (is_indirect_pointer_relocation_marker_candidate(resource)) {
        if (!is_indirect_pointer_relocation_resource(resource) ||
            descriptor.kind != SpirvDescriptorKind::StorageBuffer ||
            !descriptor.readable || descriptor.writable || descriptor.atomic_access ||
            descriptor.required_bytes > resource.indirect_pointer_relocation.binding_bytes)
            return plan;
        plan.logical_bytes = resource.indirect_pointer_relocation.binding_bytes;
        plan.binding_bytes = resource.indirect_pointer_relocation.binding_bytes;
        plan.valid = true;
        return plan;
    }

    const bool has_logical = descriptor.zero_pad_logical_bytes != 0;
    const bool has_binding = descriptor.zero_pad_binding_bytes != 0;
    const bool has_semantic =
        descriptor.zero_pad_semantic != StorageBufferTailSemantic::None;
    if (!has_logical && !has_binding && !has_semantic) {
        plan.valid = true;
        return plan;
    }
    if (!has_logical || !has_binding || !has_semantic) return plan;

    const bool buffer_resource = resource.cls == ResourceClass::ConstantBuffer ||
                                 resource.cls == ResourceClass::VertexBuffer;
    const bool semantic_matches =
        (descriptor.zero_pad_semantic == StorageBufferTailSemantic::Uint16 &&
         resource.format == DataFormat::Uint16) ||
        (descriptor.zero_pad_semantic == StorageBufferTailSemantic::Float16 &&
         resource.format == DataFormat::Float16);
    // `resource.size` is the RESOURCE's byte count, not the shader's. It used to be pinned to
    // exactly 2 -- one Float16/Uint16 record -- which silently excluded every guest buffer that
    // holds a record ARRAY and is read at element zero. The shader's own access is already bounded
    // by the three terms above it: `required_bytes == 4` (one zero-padded record) and
    // `dynamic_access == false` (that access is static), so a larger resource cannot be indexed past
    // the record this plan materialises. What is required of the resource is therefore that it hold
    // AT LEAST one whole record, which is what the two terms below now say.
    //
    // Measured on Dragon Quest VII (#1486): its field phase emits 549,623 materialisation rejects in
    // one routed run, ALL for one address at set 0, and every one of them matched every term here
    // except this one -- `size=16` at `stride=2`, i.e. eight records read at element zero. Each
    // reject SKIPS the binding (`continue` at the call site), so the shader ran with the descriptor
    // absent.
    if (descriptor.kind != SpirvDescriptorKind::StorageBuffer || !buffer_resource ||
        !descriptor.readable || descriptor.writable || descriptor.atomic_access ||
        descriptor.dynamic_access || descriptor.required_bytes != 4u ||
        descriptor.zero_pad_logical_bytes != 2u ||
        descriptor.zero_pad_binding_bytes != 4u ||
        !semantic_matches || resource.num_components != 1u ||
        resource.stride != 2u || resource.size < 2u || (resource.size % 2u) != 0u)
        return plan;

    plan.logical_bytes = 2;
    plan.binding_bytes = 4;
    plan.semantic = descriptor.zero_pad_semantic;
    plan.zero_padded_tail = true;
    plan.valid = true;
    return plan;
}

bool materialize_storage_buffer_bytes(
    const StorageBufferMaterializationPlan& plan,
    const uint8_t* source,
    uint64_t source_bytes,
    uint8_t* destination,
    uint64_t destination_bytes) {
    if (!plan.valid || plan.logical_bytes > plan.binding_bytes ||
        source_bytes < plan.logical_bytes || destination_bytes < plan.binding_bytes ||
        (plan.logical_bytes && !source) || (plan.binding_bytes && !destination))
        return false;
    if (plan.binding_bytes)
        std::memset(destination, 0, static_cast<size_t>(plan.binding_bytes));
    if (plan.logical_bytes)
        std::memcpy(destination, source, static_cast<size_t>(plan.logical_bytes));
    return true;
}

bool DescriptorValidationReport::ok() const {
    for (const auto& issue : issues) if (issue.error) return false;
    return true;
}

bool spirv_descriptor_reflection_complete(const DescriptorValidationReport& report) {
    return std::none_of(
        report.issues.begin(), report.issues.end(),
        [](const DescriptorValidationIssue& issue) {
            return issue.code == DescriptorIssueCode::MalformedSpirv;
        });
}

const SpirvDescriptorBinding* find_spirv_descriptor_binding(
    const DescriptorValidationReport& report, uint32_t set, uint32_t binding) {
    const auto found = std::find_if(
        report.descriptors.begin(), report.descriptors.end(),
        [&](const SpirvDescriptorBinding& descriptor) {
            return descriptor.set == set && descriptor.binding == binding;
        });
    return found == report.descriptors.end() ? nullptr : &*found;
}

const char* spirv_descriptor_kind_name(SpirvDescriptorKind kind) {
    switch (kind) {
        case SpirvDescriptorKind::StorageBuffer:        return "storage-buffer";
        case SpirvDescriptorKind::CombinedImageSampler: return "combined-image-sampler";
        case SpirvDescriptorKind::StorageImage:         return "storage-image";
        default:                                        return "unknown";
    }
}

const char* descriptor_issue_name(DescriptorIssueCode code) {
    switch (code) {
        case DescriptorIssueCode::MalformedSpirv:       return "malformed SPIR-V";
        case DescriptorIssueCode::StageMismatch:       return "shader stage mismatch";
        case DescriptorIssueCode::SetMismatch:         return "descriptor set mismatch";
        case DescriptorIssueCode::MissingBinding:      return "missing runtime binding";
        case DescriptorIssueCode::DuplicateBinding:    return "duplicate runtime binding";
        case DescriptorIssueCode::WrongType:           return "wrong descriptor type";
        case DescriptorIssueCode::InvalidAddress:      return "invalid resource address";
        case DescriptorIssueCode::InvalidBufferMetadata:return "suspicious buffer metadata";
        case DescriptorIssueCode::InvalidImageMetadata:return "invalid image metadata";
        case DescriptorIssueCode::UndersizedBuffer:    return "undersized buffer";
        case DescriptorIssueCode::UnusedRuntimeBinding:return "unused runtime binding";
        case DescriptorIssueCode::ArrayBindingArityMismatch:
                                                        return "descriptor array arity mismatch";
        default:                                        return "unknown descriptor issue";
    }
}

DescriptorValidationReport validate_spirv_descriptor_interface(
    const std::vector<uint32_t>& spirv,
    const ShaderResourceTable* runtime,
    uint32_t expected_set,
    SpirvShaderStage expected_stage,
    bool report_unused) {
    DescriptorValidationReport report;
    SpirvShaderStage stage = SpirvShaderStage::Unknown;
    const uint64_t reflection_hash = spirv_reflection_hash(spirv);
    bool reflection_cached = false;
    {
        DescriptorReflectionCache& cache = descriptor_reflection_cache();
        std::lock_guard lock(cache.mutex);
        auto found = cache.entries.find(reflection_hash);
        // Equality is authoritative; the hash only selects a candidate, so collisions cannot reuse
        // another module's descriptor contract.
        if (found != cache.entries.end() && found->second.spirv == spirv) {
            found->second.last_use = ++cache.use_clock;
            report = found->second.reflection;
            stage = found->second.stage;
            reflection_cached = true;
        }
    }

    if (!reflection_cached) {
    if (spirv.size() < 5 || spirv[0] != kSpirvMagic) { add_malformed(report); return report; }

    std::vector<Instruction> insts;
    for (uint32_t off = 5; off < spirv.size();) {
        uint32_t first = spirv[off];
        uint16_t words = static_cast<uint16_t>(first >> 16);
        uint16_t opcode = static_cast<uint16_t>(first & 0xFFFFu);
        if (!words || static_cast<uint64_t>(off) + words > spirv.size()) {
            add_malformed(report); return report;
        }
        insts.push_back({off, words, opcode}); off += words;
    }

    std::unordered_map<uint32_t, TypeInfo> types;
    std::unordered_map<uint32_t, uint64_t> constants;
    std::unordered_map<uint32_t, VariableInfo> variables;
    std::unordered_map<uint32_t, uint32_t> sets, bindings;
    std::unordered_map<uint32_t, uint64_t> array_strides;
    std::unordered_map<uint64_t, uint64_t> member_offsets;
    std::map<uint64_t, StorageBufferZeroPadMarker> zero_pad_markers;
    bool malformed_zero_pad_marker = false;
    auto word = [&](const Instruction& in, uint32_t operand) { return spirv[in.offset + 1u + operand]; };
    for (const Instruction& in : insts) {
        const uint32_t n = in.words - 1u;
        switch (in.opcode) {
            case OpEntryPoint:
                if (n >= 2 && stage == SpirvShaderStage::Unknown)
                    stage = static_cast<SpirvShaderStage>(word(in, 0));
                break;
            case OpTypeInt:
                if (n >= 3) {
                    TypeInfo t;
                    t.kind = TypeKind::Scalar;
                    t.width = word(in, 1);
                    t.scalar_integer = true;
                    t.scalar_signed = word(in, 2) != 0;
                    types[word(in, 0)] = t;
                }
                break;
            case OpTypeFloat:
                if (n >= 2) { TypeInfo t; t.kind = TypeKind::Scalar; t.width = word(in, 1); t.scalar_float = true; types[word(in, 0)] = t; }
                break;
            case OpTypeVector:
                if (n >= 3) { TypeInfo t; t.kind = TypeKind::Vector; t.element = word(in, 1); t.count = word(in, 2); types[word(in, 0)] = t; }
                break;
            case OpTypeImage:
                if (n >= 8) {
                    TypeInfo t;
                    t.kind = TypeKind::Image;
                    t.element = word(in, 1);
                    t.image_dim = word(in, 2);
                    t.image_depth = word(in, 3) == 1u;
                    t.image_arrayed = word(in, 4) != 0;
                    t.image_multisampled = word(in, 5) != 0;
                    t.image_sampled = word(in, 6);
                    t.image_format = word(in, 7);
                    types[word(in, 0)] = t;
                }
                break;
            case OpTypeSampledImage:
                if (n >= 2) { TypeInfo t; t.kind = TypeKind::SampledImage; t.element = word(in, 1); types[word(in, 0)] = t; }
                break;
            case OpTypeArray:
                if (n >= 3) { TypeInfo t; t.kind = TypeKind::Array; t.element = word(in, 1); t.count_id = word(in, 2); types[word(in, 0)] = t; }
                break;
            case OpTypeRuntimeArray:
                if (n >= 2) { TypeInfo t; t.kind = TypeKind::RuntimeArray; t.element = word(in, 1); types[word(in, 0)] = t; }
                break;
            case OpTypeStruct:
                if (n >= 1) { TypeInfo t; t.kind = TypeKind::Struct; for (uint32_t i = 1; i < n; ++i) t.members.push_back(word(in, i)); types[word(in, 0)] = std::move(t); }
                break;
            case OpTypePointer:
                if (n >= 3) { TypeInfo t; t.kind = TypeKind::Pointer; t.storage = word(in, 1); t.element = word(in, 2); types[word(in, 0)] = t; }
                break;
            case OpConstant:
                if (n >= 3) constants[word(in, 1)] = word(in, 2);
                break;
            case OpVariable:
                if (n >= 3) variables[word(in, 1)] = {word(in, 0), word(in, 2)};
                break;
            case OpDecorate:
                if (n >= 2) {
                    uint32_t target = word(in, 0), decoration = word(in, 1);
                    if (decoration == DecorationBinding && n >= 3) bindings[target] = word(in, 2);
                    else if (decoration == DecorationDescriptorSet && n >= 3) sets[target] = word(in, 2);
                    else if (decoration == DecorationArrayStride && n >= 3) array_strides[target] = word(in, 2);
                }
                break;
            case OpMemberDecorate:
                if (n >= 4 && word(in, 2) == DecorationOffset)
                    member_offsets[member_key(word(in, 0), word(in, 1))] = word(in, 3);
                break;
            case OpModuleProcessed: {
                const std::string marker_text = instruction_string(spirv, in);
                constexpr const char kPrefix[] = "Prosper.StorageBufferZeroPad=";
                if (!marker_text.starts_with(kPrefix)) break;
                StorageBufferZeroPadMarker marker;
                if (!parse_storage_buffer_zero_pad_marker(marker_text, marker) ||
                    marker.logical_bytes != 2u || marker.binding_bytes != 4u ||
                    marker.semantic == StorageBufferTailSemantic::None) {
                    malformed_zero_pad_marker = true;
                    break;
                }
                const uint64_t key = (static_cast<uint64_t>(marker.set) << 32u) |
                                     marker.binding;
                if (!zero_pad_markers.emplace(key, marker).second)
                    malformed_zero_pad_marker = true;
                break;
            }
            default:
                break;
        }
    }
    if (malformed_zero_pad_marker) add_malformed(report);
    for (auto& [id, type] : types) if (type.kind == TypeKind::Array) {
        auto ci = constants.find(type.count_id);
        if (ci != constants.end() && ci->second <= std::numeric_limits<uint32_t>::max())
            type.count = static_cast<uint32_t>(ci->second);
    }

    std::unordered_map<uint32_t, SpirvDescriptorKind> descriptor_vars;
    std::unordered_map<uint32_t, uint32_t> descriptor_arities;
    std::set<uint32_t> sampled_float_vars;
    std::set<uint32_t> storage_float_vars;
    std::unordered_map<uint32_t, SpirvImageNumericClass> image_numeric_classes;
    for (const auto& [id, var] : variables) {
        SpirvDescriptorKind kind = descriptor_kind(var, types);
        if (kind != SpirvDescriptorKind::Unknown) {
            descriptor_vars[id] = kind;
            descriptor_arities[id] = descriptor_array_arity(var, types);
            const SpirvImageNumericClass numeric_class =
                descriptor_image_numeric_class(var, types);
            image_numeric_classes[id] = numeric_class;
            if (numeric_class == SpirvImageNumericClass::Float) {
                if (kind == SpirvDescriptorKind::CombinedImageSampler)
                    sampled_float_vars.insert(id);
                else if (kind == SpirvDescriptorKind::StorageImage)
                    storage_float_vars.insert(id);
            }
        }
    }

    std::set<uint32_t> used_vars;
    std::set<uint32_t> read_vars;
    std::set<uint32_t> written_vars;
    std::set<uint32_t> atomic_vars;
    std::set<uint32_t> normalized_sample_vars;
    std::set<uint32_t> texel_access_vars;
    std::unordered_map<uint32_t, PointerAccess> accesses;
    // Result id of OpLoad/OpSampledImage -> descriptor variable. An image descriptor load accesses
    // the descriptor object, not its texels; only the later image opcode establishes read/write data
    // access. Keeping that distinction makes write-only storage outputs visible to the backend even
    // when the same shader reads a different storage image.
    std::unordered_map<uint32_t, uint32_t> image_objects;
    std::unordered_map<uint32_t, uint64_t> required;
    std::unordered_map<uint32_t, bool> dynamic;

    auto mark = [&](uint32_t var, bool read, bool write) {
        if (!descriptor_vars.count(var)) return;
        used_vars.insert(var);
        if (read) read_vars.insert(var);
        if (write) written_vars.insert(var);
    };
    auto pointer_root = [&](uint32_t pointer) -> uint32_t {
        if (descriptor_vars.count(pointer)) return pointer;
        auto access = accesses.find(pointer);
        return access != accesses.end() ? access->second.variable : 0u;
    };
    auto mark_pointer = [&](uint32_t pointer, bool read, bool write) {
        const uint32_t root = pointer_root(pointer);
        if (root) mark(root, read, write);
    };
    auto mark_image = [&](uint32_t object, bool read, bool write) {
        auto origin = image_objects.find(object);
        if (origin != image_objects.end()) mark(origin->second, read, write);
    };
    auto mark_image_coordinate_contract = [&](uint32_t object, bool normalized) {
        auto origin = image_objects.find(object);
        if (origin == image_objects.end()) return;
        if (normalized) normalized_sample_vars.insert(origin->second);
        else texel_access_vars.insert(origin->second);
    };
    for (const Instruction& in : insts) {
        const uint32_t n = in.words - 1u;
        if (in.opcode == OpImageTexelPointer && n >= 5) {
            // Unlike OpImageRead/Write, this instruction consumes the storage-image descriptor
            // variable directly and returns a pointer in the Image storage class. Preserve that
            // provenance so the following OpAtomic* marks the image readable+writable.
            const uint32_t result_type = word(in, 0);
            const uint32_t result = word(in, 1);
            const uint32_t image_var = word(in, 2);
            const auto descriptor = descriptor_vars.find(image_var);
            const auto pointer_type = types.find(result_type);
            if (descriptor != descriptor_vars.end() &&
                descriptor->second == SpirvDescriptorKind::StorageImage &&
                pointer_type != types.end() &&
                pointer_type->second.kind == TypeKind::Pointer &&
                pointer_type->second.storage == StorageImage) {
                PointerAccess access;
                access.variable = image_var;
                access.pointee_type = pointer_type->second.element;
                accesses[result] = access;
                texel_access_vars.insert(image_var);
            }
        } else if (in.opcode == OpLoad && n >= 3) {
            const uint32_t root = pointer_root(word(in, 2));
            const auto descriptor = descriptor_vars.find(root);
            const bool image_descriptor = descriptor != descriptor_vars.end() &&
                descriptor->second != SpirvDescriptorKind::StorageBuffer;
            mark_pointer(word(in, 2), !image_descriptor, false);
            if (image_descriptor) image_objects[word(in, 1)] = root;
        } else if (in.opcode == OpStore && n >= 1) {
            mark_pointer(word(in, 0), false, true);
        } else if (in.opcode == OpAtomicLoad && n >= 3) {
            mark_pointer(word(in, 2), true, false);
        } else if (((in.opcode >= OpAtomicExchange && in.opcode <= OpAtomicIDecrement) ||
                    (in.opcode >= OpAtomicIAdd && in.opcode <= OpAtomicXor) ||
                    in.opcode == OpAtomicFlagTestAndSet) && n >= 3) {
            // Result-producing atomic instructions place their pointer after result type/result id.
            const uint32_t root = pointer_root(word(in, 2));
            if (root) atomic_vars.insert(root);
            mark_pointer(word(in, 2), true, true);
        } else if ((in.opcode == OpAtomicStore || in.opcode == OpAtomicFlagClear) && n >= 1) {
            // The two result-less atomic instructions place the pointer first, like OpStore.
            mark_pointer(word(in, 0), false, true);
        } else if (in.opcode == 86u /* OpSampledImage */ && n >= 3) {
            auto origin = image_objects.find(word(in, 2));
            if (origin != image_objects.end()) image_objects[word(in, 1)] = origin->second;
        } else if (in.opcode >= 87u && in.opcode <= 98u && n >= 3) {
            // All sampled/fetch/gather forms plus OpImageRead place their image object after result
            // type/result id. The descriptor itself was loaded earlier; this is the texel read.
            mark_image(word(in, 2), true, false);
            // OpImageSample* (87..94) and Gather/DrefGather (96..97) consume normalized sampler
            // coordinates. OpImageFetch (95) and OpImageRead (98) consume integer texel coordinates
            // and therefore cannot observe a reduced render target as if it had the native extent.
            const bool normalized = (in.opcode >= 87u && in.opcode <= 94u) ||
                                    in.opcode == 96u || in.opcode == 97u;
            mark_image_coordinate_contract(word(in, 2), normalized);
        } else if (in.opcode == 99u /* OpImageWrite */ && n >= 1) {
            mark_image(word(in, 0), false, true);
        } else if (in.opcode == 100u /* OpImage */ && n >= 3) {
            auto origin = image_objects.find(word(in, 2));
            if (origin != image_objects.end()) image_objects[word(in, 1)] = origin->second;
        } else if (in.opcode >= 101u && in.opcode <= 107u && n >= 3) {
            // OpImageQueryFormat/Order/SizeLod/Size/Lod/Levels/Samples all place the image object
            // after result type/result id. Query-only descriptors must remain in the reflected
            // layout, and their reported dimensions/LOD contract requires the guest's exact extent.
            mark_image(word(in, 2), true, false);
            mark_image_coordinate_contract(word(in, 2), false);
        } else if ((in.opcode == OpAccessChain || in.opcode == OpInBoundsAccessChain) && n >= 3) {
            const uint32_t result = word(in, 1), base = word(in, 2);
            PointerAccess a;
            auto vi = variables.find(base);
            if (vi != variables.end()) {
                auto pi = types.find(vi->second.pointer_type);
                if (pi == types.end() || pi->second.kind != TypeKind::Pointer) continue;
                a.variable = base; a.pointee_type = pi->second.element;
            } else {
                auto ai = accesses.find(base); if (ai == accesses.end()) continue;
                a = ai->second;
            }
            uint32_t current = a.pointee_type;
            for (uint32_t oi = 3; oi < n; ++oi) {
                auto ti = types.find(current); if (ti == types.end()) { a.dynamic = true; break; }
                auto ci = constants.find(word(in, oi));
                const bool fixed = ci != constants.end();
                uint64_t index = fixed ? ci->second : 0;
                if (!fixed) a.dynamic = true;
                const TypeInfo& t = ti->second;
                if (t.kind == TypeKind::Struct) {
                    if (!fixed || index >= t.members.size()) { a.dynamic = true; current = 0; break; }
                    uint64_t off = 0;
                    auto mo = member_offsets.find(member_key(current, static_cast<uint32_t>(index)));
                    if (mo != member_offsets.end()) off = mo->second;
                    else for (uint32_t m = 0; m < index; ++m)
                        off += type_size(t.members[m], types, array_strides, member_offsets);
                    a.offset += off; current = t.members[static_cast<size_t>(index)];
                } else if (t.kind == TypeKind::Array || t.kind == TypeKind::RuntimeArray || t.kind == TypeKind::Vector) {
                    uint64_t stride = type_size(t.element, types, array_strides, member_offsets);
                    auto si = array_strides.find(current); if (si != array_strides.end()) stride = si->second;
                    if (fixed) a.offset += index * stride;
                    current = t.element;
                } else {
                    a.dynamic = true; current = 0; break;
                }
            }
            a.pointee_type = current; accesses[result] = a;
            if (descriptor_vars.count(a.variable)) {
                uint64_t bytes = type_size(current, types, array_strides, member_offsets);
                bytes = std::max<uint64_t>(bytes, 4);
                required[a.variable] = std::max(required[a.variable], a.offset + bytes);
                dynamic[a.variable] = dynamic[a.variable] || a.dynamic;
            }
        }
    }

    std::set<uint64_t> consumed_zero_pad_markers;
    for (uint32_t var : used_vars) {
        auto si = sets.find(var), bi = bindings.find(var);
        if (si == sets.end() || bi == bindings.end()) { add_malformed(report); continue; }
        uint32_t image_dim = UINT32_MAX;
        uint32_t image_format = 0;
        bool image_arrayed = false, image_multisampled = false, image_depth = false;
        auto vi = variables.find(var);
        const TypeInfo* image = vi == variables.end()
            ? nullptr : descriptor_image_type(vi->second, types);
        if (image) {
            image_dim = image->image_dim;
            image_format = image->image_format;
            image_arrayed = image->image_arrayed;
            image_multisampled = image->image_multisampled;
            image_depth = image->image_depth;
        }
        SpirvDescriptorBinding descriptor{
            var, si->second, bi->second, descriptor_vars[var], stage,
            required[var], dynamic[var], read_vars.count(var) != 0,
            written_vars.count(var) != 0,
            normalized_sample_vars.count(var) != 0,
            texel_access_vars.count(var) != 0,
            sampled_float_vars.count(var) != 0,
            storage_float_vars.count(var) != 0, image_format, image_dim,
            image_arrayed, image_multisampled, image_depth,
            atomic_vars.count(var) != 0};
        descriptor.image_numeric_class = image_numeric_classes[var];
        // Assigned here rather than added to the initializer above: that initializer is positional and
        // names no field, so extending it is how #2462 happens. See the note on the member.
        if (auto ai = descriptor_arities.find(var); ai != descriptor_arities.end())
            descriptor.descriptor_count = ai->second;
        const uint64_t marker_key = (static_cast<uint64_t>(descriptor.set) << 32u) |
                                    descriptor.binding;
        if (auto marker = zero_pad_markers.find(marker_key); marker != zero_pad_markers.end()) {
            if (descriptor.kind != SpirvDescriptorKind::StorageBuffer ||
                descriptor.required_bytes != 4u || descriptor.dynamic_access ||
                !descriptor.readable || descriptor.writable || descriptor.atomic_access) {
                add_malformed(report);
            } else {
                descriptor.zero_pad_logical_bytes = marker->second.logical_bytes;
                descriptor.zero_pad_binding_bytes = marker->second.binding_bytes;
                descriptor.zero_pad_semantic = marker->second.semantic;
                consumed_zero_pad_markers.insert(marker_key);
            }
        }
        report.descriptors.push_back(descriptor);
    }
    if (consumed_zero_pad_markers.size() != zero_pad_markers.size()) add_malformed(report);
    std::sort(report.descriptors.begin(), report.descriptors.end(), [](const auto& a, const auto& b) {
        if (a.set != b.set) return a.set < b.set;
        if (a.binding != b.binding) return a.binding < b.binding;
        return a.variable_id < b.variable_id;
    });
    // Logical-addressing SPIR-V needs a separate runtime-u64 Block variable for a 64-bit atomic over
    // the ordinary runtime-u32 storage-buffer ABI. Both variables deliberately name one Vulkan
    // descriptor binding. Reflection is a binding contract, not a variable list: coalesce compatible
    // storage-buffer aliases so the backend creates one VkDescriptorSetLayoutBinding, while retaining
    // the strongest byte/access requirements from either view. Any incompatible duplicate remains a
    // malformed module instead of being papered over here.
    std::vector<SpirvDescriptorBinding> coalesced;
    coalesced.reserve(report.descriptors.size());
    for (const SpirvDescriptorBinding& descriptor : report.descriptors) {
        if (coalesced.empty() || coalesced.back().set != descriptor.set ||
            coalesced.back().binding != descriptor.binding) {
            coalesced.push_back(descriptor);
            continue;
        }
        SpirvDescriptorBinding& prior = coalesced.back();
        const bool compatible =
            prior.kind == SpirvDescriptorKind::StorageBuffer &&
            descriptor.kind == SpirvDescriptorKind::StorageBuffer &&
            prior.stage == descriptor.stage &&
            prior.descriptor_count == descriptor.descriptor_count &&
            prior.zero_pad_logical_bytes == descriptor.zero_pad_logical_bytes &&
            prior.zero_pad_binding_bytes == descriptor.zero_pad_binding_bytes &&
            prior.zero_pad_semantic == descriptor.zero_pad_semantic;
        if (!compatible) {
            add_malformed(report);
            continue;
        }
        prior.required_bytes = std::max(prior.required_bytes, descriptor.required_bytes);
        prior.dynamic_access |= descriptor.dynamic_access;
        prior.readable |= descriptor.readable;
        prior.writable |= descriptor.writable;
        prior.atomic_access |= descriptor.atomic_access;
    }
    report.descriptors = std::move(coalesced);

    // Reflection depends only on immutable SPIR-V. Runtime addresses, sizes, metadata, and the
    // expected stage/set are validated below on every call. Keeping those out of this cache lets a
    // shader dispatched against different guest allocations reuse the expensive instruction/type
    // walk without weakening any per-dispatch checks.
    {
        DescriptorReflectionCache& cache = descriptor_reflection_cache();
        std::lock_guard lock(cache.mutex);
        DescriptorReflectionCacheEntry entry;
        entry.spirv = spirv;
        entry.reflection = report;
        entry.stage = stage;
        entry.last_use = ++cache.use_clock;
        constexpr uint64_t kCacheLimit = 64ull * 1024 * 1024;
        constexpr size_t kMaxEntries = 4096;
        const uint64_t entry_bytes = reflection_entry_bytes(entry);
        auto collision = cache.entries.find(reflection_hash);
        if (collision != cache.entries.end()) {
            cache.bytes -= reflection_entry_bytes(collision->second);
            cache.entries.erase(collision);
        }
        while (!cache.entries.empty() &&
               (cache.entries.size() >= kMaxEntries || cache.bytes + entry_bytes > kCacheLimit)) {
            auto oldest = cache.entries.begin();
            for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
                if (it->second.last_use < oldest->second.last_use) oldest = it;
            cache.bytes -= reflection_entry_bytes(oldest->second);
            cache.entries.erase(oldest);
        }
        if (entry_bytes <= kCacheLimit) {
            cache.bytes += entry_bytes;
            cache.entries.emplace(reflection_hash, std::move(entry));
        }
    }
    }

    if (stage != expected_stage)
        report.issues.push_back({DescriptorIssueCode::StageMismatch, true, expected_set, 0});

    std::set<uint32_t> used_bindings;
    for (const auto& d : report.descriptors) {
        used_bindings.insert(d.binding);
        if (d.set != expected_set)
            report.issues.push_back({DescriptorIssueCode::SetMismatch, true, d.set, d.binding, d.kind});

        // Fragment set 1 binding 0 is reserved for the renderer-owned 64 KiB GDS backing.
        // It has no guest descriptor-table entry; the live and replay backends inject it from
        // the SPIR-V contract so capture artifacts remain self-contained.
        const bool internal_gds = expected_stage == SpirvShaderStage::Fragment &&
            d.set == 1 && d.binding == 0 && d.kind == SpirvDescriptorKind::StorageBuffer;
        if (internal_gds) continue;

        std::vector<const ShaderResource*> matches;
        if (runtime) for (const auto& r : runtime->resources)
            if (r.binding == d.binding && r.cls != ResourceClass::Sampler) matches.push_back(&r);
        if (matches.empty()) {
            report.issues.push_back({DescriptorIssueCode::MissingBinding, true, d.set, d.binding, d.kind});
            continue;
        }
        if (matches.size() != 1) {
            report.issues.push_back({DescriptorIssueCode::DuplicateBinding, true, d.set, d.binding, d.kind});
            continue;
        }
        const ShaderResource& r = *matches.front();
        SpirvDescriptorKind actual = resource_kind(r);
        // #2265: the shape test is shared with the lowering gate and the backend materialization so
        // the three cannot drift again -- see shader_resource_supports_atomic_image_buffer. The size
        // bound is LOGICAL (width*height*layers*4): a descriptor's `size` is the linear extent, not
        // the tiled physical footprint, which is larger. Measured on CrossWorlds' 3840x2160x2 R32
        // image: size = 66,355,200 = w*h*depth*4 exactly, while the tiled footprint is 66,846,720.
        const bool atomic_image_buffer =
            expected_stage == SpirvShaderStage::Compute &&
            d.kind == SpirvDescriptorKind::StorageBuffer && d.atomic_access &&
            actual == SpirvDescriptorKind::StorageImage &&
            shader_resource_supports_atomic_image_buffer(r) &&
            static_cast<uint64_t>(r.width) * r.height *
                    shader_resource_atomic_image_layers(r) * sizeof(uint32_t) <= r.size;
        if (actual != d.kind && !atomic_image_buffer) {
            report.issues.push_back({DescriptorIssueCode::WrongType, true, d.set, d.binding,
                                     d.kind, actual, d.required_bytes, r.size});
            continue;
        }
        // ARITY (#2412, stage 3). Checked before any byte-range reasoning, because for an array binding
        // `required_bytes` is not a byte range at all: reflection folds the array index into the same
        // offset arithmetic it uses inside a buffer, so an eight-element array reports one binding with
        // a nonsense requirement. Comparing that against `r.size` produces a confident verdict about a
        // quantity neither side is talking about.
        //
        // Both directions are errors and for different reasons. Array SPIR-V against a scalar resource
        // means the shader can index past the one descriptor the backend will bind — reading whatever
        // the next binding holds. A scalar shader against a table-indexed resource means the executor
        // built an array the module cannot address, so entries beyond the first are unreachable and the
        // dispatch silently uses one slot of a table it was given.
        //
        // This is deliberately introduced BEFORE arrays work. Until stage 4 emits them and stage 5
        // materialises them, the correct behaviour for a mismatch is to REJECT, and rejecting loudly is
        // the whole point: the pre-stage-3 code compared an array against a scalar and said nothing.
        const uint32_t shader_count  = d.descriptor_count;
        const uint32_t runtime_count = r.table_index_count;
        const bool shader_is_array  = shader_count != 1;
        const bool runtime_is_array = runtime_count != 0;
        // An unread array length is a mismatch whatever the table supplies -- tested FIRST and named,
        // rather than left to fall out of the comparison below. The arithmetic would reject it today
        // (kDescriptorArityUnknown equals neither 1 nor any plausible table size), but only by accident:
        // the sentinel would stop rejecting the moment someone changed the comparison, and nothing in
        // the expression would show that a decode failure was ever meant to be caught here.
        const bool shader_arity_unknown = shader_count == kDescriptorArityUnknown;
        if (shader_arity_unknown || shader_is_array != runtime_is_array ||
            (shader_is_array && shader_count != 0 && shader_count != runtime_count)) {
            // The counts go in their own fields and the byte slots stay zero. An earlier revision rode
            // them in `required_bytes`/`available_bytes` on the belief that nothing rendered those --
            // false: `gpu_executor.cpp` prints both for every issue in the `[descriptor]` and
            // `[compute-descriptor]` dumps, and mixes them into the dedupe hash for error issues, so a
            // fired check printed `required=8 available=0` about descriptors rather than bytes.
            DescriptorValidationIssue arity{DescriptorIssueCode::ArrayBindingArityMismatch, true, d.set,
                                            d.binding, d.kind, actual};
            arity.shader_count = shader_count;
            arity.runtime_count = runtime_count;
            report.issues.push_back(arity);
            continue;
        }
        if (!valid_shader_buffer_table_contract(r)) {
            report.issues.push_back({DescriptorIssueCode::InvalidBufferMetadata, true, d.set,
                                     d.binding, d.kind, actual});
            continue;
        }
        if (runtime_is_array) {
            // Every array slot is a separate V# and therefore a separate byte-range promise. The
            // parent ShaderResource describes the binding, not an arbitrary representative entry;
            // validating only its legacy scalar address would leave all other selected slots unchecked.
            bool invalid_address = false;
            bool invalid_metadata = d.kind != SpirvDescriptorKind::StorageBuffer;
            bool undersized = false;
            uint64_t smallest_available = UINT64_MAX;
            for (const ShaderBufferTableEntry& table_entry : r.table_entries) {
                const bool null_entry = table_entry.gpu_addr == 0u && table_entry.size == 0u &&
                                        table_entry.host_data == nullptr;
                if (!null_entry && table_entry.gpu_addr == 0u && !table_entry.host_data) {
                    invalid_address = true;
                    break;
                }
                ShaderResource entry_resource = r;
                entry_resource.gpu_addr = table_entry.gpu_addr;
                entry_resource.size = table_entry.size;
                entry_resource.stride = table_entry.stride;
                entry_resource.host_data = table_entry.host_data;
                entry_resource.host_data_size = table_entry.host_data_size;
                // A table entry's backing is its own allocation; the parent's prefix does not apply.
                entry_resource.host_data_prefix_bytes = 0;
                entry_resource.table_index_count = 0u;
                entry_resource.table_entry_stride = 0u;
                entry_resource.table_index_sgpr = UINT32_MAX;
                entry_resource.table_selector_mode = BufferTableSelectorMode::None;
                entry_resource.table_load_pc = UINT32_MAX;
                entry_resource.table_entries.clear();
                const StorageBufferMaterializationPlan materialization =
                    plan_storage_buffer_materialization(d, entry_resource);
                if (!materialization.valid) {
                    invalid_metadata = true;
                    break;
                }
                const uint64_t minimum = materialization.zero_padded_tail
                    ? materialization.logical_bytes
                    : std::max<uint64_t>(d.required_bytes, 4u);
                uint64_t available = materialization.binding_bytes;
                if (table_entry.host_data)
                    available = std::min<uint64_t>(available, table_entry.host_data_size);
                smallest_available = std::min(smallest_available, available);
                if (!null_entry && available < minimum) undersized = true;
            }
            if (invalid_address)
                report.issues.push_back({DescriptorIssueCode::InvalidAddress, true, d.set,
                                         d.binding, d.kind, actual});
            else if (invalid_metadata)
                report.issues.push_back({DescriptorIssueCode::InvalidBufferMetadata, true, d.set,
                                         d.binding, d.kind, actual});
            else if (undersized)
                report.issues.push_back({DescriptorIssueCode::UndersizedBuffer, true, d.set,
                                         d.binding, d.kind, actual,
                                         std::max<uint64_t>(d.required_bytes, 4u),
                                         smallest_available == UINT64_MAX ? 0u
                                                                         : smallest_available});
            continue;
        }
        // Explicit null descriptors are valid guest state: resource reads return zero and the backend
        // binds a zero-filled dummy. An absent table entry is still an error above.
        const bool null_descriptor = r.gpu_addr == 0 && r.size == 0 && !r.host_data;
        if (!null_descriptor && r.gpu_addr == 0 && !r.host_data) {
            report.issues.push_back({DescriptorIssueCode::InvalidAddress, true, d.set, d.binding,
                                     d.kind, actual});
            continue;
        }
        if (d.kind == SpirvDescriptorKind::StorageBuffer) {
            const StorageBufferMaterializationPlan materialization =
                plan_storage_buffer_materialization(d, r);
            if (!materialization.valid) {
                report.issues.push_back({DescriptorIssueCode::InvalidBufferMetadata, true, d.set,
                                         d.binding, d.kind, actual, d.required_bytes, r.size});
                continue;
            }
            const uint64_t minimum = materialization.zero_padded_tail
                ? materialization.logical_bytes
                : std::max<uint64_t>(d.required_bytes, 4);
            uint64_t available = materialization.binding_bytes;
            if (r.host_data) available = std::min<uint64_t>(available, r.host_data_size);
            if (!null_descriptor && available < minimum)
                report.issues.push_back({DescriptorIssueCode::UndersizedBuffer, true, d.set, d.binding,
                                         d.kind, actual, minimum, available});
            // SPIR-V storage buffers do not encode the source V#'s stride/format. Preserve definite
            // metadata anomalies as warnings so investigations see them without rejecting legal raw
            // or zero-stride loads whose instruction semantics do not consume those fields.
            if (!null_descriptor && r.cls == ResourceClass::VertexBuffer &&
                (!r.stride || r.format == DataFormat::Unknown || r.num_components < 1 || r.num_components > 4))
                report.issues.push_back({DescriptorIssueCode::InvalidBufferMetadata, false, d.set,
                                         d.binding, d.kind, actual});
        } else if (!null_descriptor) {
            // The renderer needs a concrete image extent and channel count to materialize a sampled
            // or storage image. Unknown storage-image format remains legal SPIR-V (WithoutFormat).
            const bool bad_components = r.num_components < 1 || r.num_components > 4;
            if (!r.width || !r.height || bad_components || (!r.size && !r.host_data))
                report.issues.push_back({DescriptorIssueCode::InvalidImageMetadata, true, d.set,
                                         d.binding, d.kind, actual});
            else if (d.kind == SpirvDescriptorKind::CombinedImageSampler && r.format == DataFormat::Unknown)
                report.issues.push_back({DescriptorIssueCode::InvalidImageMetadata, false, d.set,
                                         d.binding, d.kind, actual});
        }
    }

    if (report_unused && runtime) {
        std::set<uint32_t> warned;
        for (const auto& r : runtime->resources) {
            if (r.cls == ResourceClass::Sampler || used_bindings.count(r.binding) || !warned.insert(r.binding).second) continue;
            report.issues.push_back({DescriptorIssueCode::UnusedRuntimeBinding, false, expected_set,
                                     r.binding, SpirvDescriptorKind::Unknown, resource_kind(r), 0, r.size});
        }
    }
    return report;
}

} // namespace prosper::gpu
