// Exact GTA V packed-pointer compute contract. This is deliberately separate from the generic FLAT
// lowering: only one fully identified program may replace its lane-zero guest pointers with bounded
// offsets.
#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper::gpu {

struct ComputeShaderConfig;
struct IndirectBufferShadowAccess;
struct Rdna2Inst;
struct ShaderResource;
struct ShaderResourceTable;

inline constexpr uint32_t kGta5PackedPointerSourcePc = 26u;
inline constexpr uint32_t kGta5PackedPointerAtomicSourcePc = 353u;
inline constexpr uint32_t kGta5PackedPointerAtomicLoadBytes = 28u;
inline constexpr uint32_t kGta5PackedPointerAtomicBindingBytes = 32u;
inline constexpr uint32_t kGta5PackedPointerAtomicByteOffset = 24u;
inline constexpr uint32_t kGta5PackedPointerDescriptorStride = 256u;
inline constexpr uint32_t kGta5PackedPointerRawWordHiMetadataMask = 0xffff0000u;
inline constexpr uint32_t kGta5PackedPointerRawWordHiFirstMetadataBit = 1u << 16u;
inline constexpr uint32_t kGta5PackedPointerSourceStride = 96u;
inline constexpr uint32_t kGta5PackedPointerMaxThreads = 192u;
inline constexpr uint32_t kGta5PackedPointerMaxSlots = 3u;
inline constexpr uint32_t kGta5PackedPointerHeaderBytes = 48u;
inline constexpr uint32_t kGta5PackedPointerSlotBytes = 368u;
inline constexpr uint32_t kGta5PackedPointerTag = 0xf5012481u;

bool rdna2_gta5_packed_pointer_shader(const uint32_t* code, size_t dwords);
bool rdna2_gta5_packed_pointer_launch(
    const uint32_t* code, size_t dwords, const ComputeShaderConfig& config);

bool discover_rdna2_gta5_packed_pointer(
    const uint32_t* code, size_t dwords, const ComputeShaderConfig& config,
    ShaderResourceTable& resources);

bool rdna2_gta5_packed_pointer_dispatch(
    const uint32_t* code, size_t dwords, const ComputeShaderConfig& config,
    const ShaderResourceTable& resources);

bool is_gta5_packed_pointer_marker_candidate(const ShaderResource& resource);
bool is_gta5_packed_pointer_resource(const ShaderResource& resource);

// Capture-side recognition after derived marker fields have intentionally not been serialized.
bool is_gta5_packed_pointer_serialized_shadow(
    const ShaderResource& resource, const uint8_t* bytes, size_t byte_count);

bool rdna2_gta5_packed_pointer_access(
    const Rdna2Inst& instruction, IndirectBufferShadowAccess& access);

// The same exact program ends with one BUFFER_ATOMIC_OR_X2 through the separately proven writable
// source. This recognizes only that consumer; dispatch validation owns its descriptor algebra.
bool rdna2_gta5_packed_pointer_atomic_site(const Rdna2Inst& instruction);

} // namespace prosper::gpu
