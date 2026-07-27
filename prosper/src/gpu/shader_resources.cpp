// shader_resources.cpp — see shader_resources.hpp. Pure lookups + format sizing; no Vulkan, no state.
#include "shader_resources.hpp"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

namespace prosper::gpu {

uint32_t data_format_bytes(DataFormat f) {
    switch (f) {
        case DataFormat::Float32: case DataFormat::Uint32: case DataFormat::Sint32: return 4;
        case DataFormat::Float16: case DataFormat::Unorm16: case DataFormat::Snorm16:
        case DataFormat::Uint16:  case DataFormat::Sint16:  return 2;
        case DataFormat::Unorm8:  case DataFormat::Snorm8:
        case DataFormat::Uint8:   case DataFormat::Sint8:   return 1;
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
    bool scalar_float = false;
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

bool descriptor_storage_float(const VariableInfo& var,
                              const std::unordered_map<uint32_t, TypeInfo>& types) {
    auto pi = types.find(var.pointer_type);
    if (pi == types.end() || pi->second.kind != TypeKind::Pointer) return false;
    uint32_t type = pi->second.element;
    for (uint32_t depth = 0; depth < 8; ++depth) {
        auto ti = types.find(type);
        if (ti == types.end()) return false;
        if (ti->second.kind == TypeKind::Array) {
            type = ti->second.element;
            continue;
        }
        if (ti->second.kind != TypeKind::Image || ti->second.image_sampled != 2) return false;
        auto sampled_type = types.find(ti->second.element);
        return sampled_type != types.end() && sampled_type->second.kind == TypeKind::Scalar &&
               sampled_type->second.scalar_float;
    }
    return false;
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

bool DescriptorValidationReport::ok() const {
    for (const auto& issue : issues) if (issue.error) return false;
    return true;
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
    auto word = [&](const Instruction& in, uint32_t operand) { return spirv[in.offset + 1u + operand]; };
    for (const Instruction& in : insts) {
        const uint32_t n = in.words - 1u;
        switch (in.opcode) {
            case OpEntryPoint:
                if (n >= 2 && stage == SpirvShaderStage::Unknown)
                    stage = static_cast<SpirvShaderStage>(word(in, 0));
                break;
            case OpTypeInt:
                if (n >= 2) { TypeInfo t; t.kind = TypeKind::Scalar; t.width = word(in, 1); types[word(in, 0)] = t; }
                break;
            case OpTypeFloat:
                if (n >= 2) { TypeInfo t; t.kind = TypeKind::Scalar; t.width = word(in, 1); t.scalar_float = true; types[word(in, 0)] = t; }
                break;
            case OpTypeVector:
                if (n >= 3) { TypeInfo t; t.kind = TypeKind::Vector; t.element = word(in, 1); t.count = word(in, 2); types[word(in, 0)] = t; }
                break;
            case OpTypeImage:
                if (n >= 8) { TypeInfo t; t.kind = TypeKind::Image; t.element = word(in, 1); t.image_sampled = word(in, 6); types[word(in, 0)] = t; }
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
            default:
                break;
        }
    }
    for (auto& [id, type] : types) if (type.kind == TypeKind::Array) {
        auto ci = constants.find(type.count_id);
        if (ci != constants.end() && ci->second <= std::numeric_limits<uint32_t>::max())
            type.count = static_cast<uint32_t>(ci->second);
    }

    std::unordered_map<uint32_t, SpirvDescriptorKind> descriptor_vars;
    std::set<uint32_t> storage_float_vars;
    for (const auto& [id, var] : variables) {
        SpirvDescriptorKind kind = descriptor_kind(var, types);
        if (kind != SpirvDescriptorKind::Unknown) {
            descriptor_vars[id] = kind;
            if (kind == SpirvDescriptorKind::StorageImage &&
                descriptor_storage_float(var, types))
                storage_float_vars.insert(id);
        }
    }

    std::set<uint32_t> used_vars;
    std::set<uint32_t> read_vars;
    std::set<uint32_t> written_vars;
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
            mark_image_coordinate_contract(
                word(in, 2), (in.opcode >= 87u && in.opcode <= 94u) ||
                                 in.opcode == 96u || in.opcode == 97u);
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

    for (uint32_t var : used_vars) {
        auto si = sets.find(var), bi = bindings.find(var);
        if (si == sets.end() || bi == bindings.end()) { add_malformed(report); continue; }
        report.descriptors.push_back({var, si->second, bi->second, descriptor_vars[var], stage,
                                      required[var], dynamic[var], read_vars.count(var) != 0,
                                      written_vars.count(var) != 0,
                                      normalized_sample_vars.count(var) != 0,
                                      texel_access_vars.count(var) != 0,
                                      storage_float_vars.count(var) != 0});
    }
    std::sort(report.descriptors.begin(), report.descriptors.end(), [](const auto& a, const auto& b) {
        return a.set != b.set ? a.set < b.set : a.binding < b.binding;
    });

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
        if (actual != d.kind) {
            report.issues.push_back({DescriptorIssueCode::WrongType, true, d.set, d.binding,
                                     d.kind, actual, d.required_bytes, r.size});
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
            uint64_t minimum = std::max<uint64_t>(d.required_bytes, 4);
            uint64_t available = r.size;
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
