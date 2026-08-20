// Title-neutral proof for bounded guest pointers loaded from dispatch-indexed source records.
//
// This layer proves producer/guard/consumer structure and derives one bounded record per real
// invocation. It does not grant arbitrary FLAT access: every relocated consumer retains its exact
// PC, packet, address pair, width, and immediate in the proof. The generic v2 carrier owns snapshot
// layout and lifetime separately (rdna2_indirect_buffer_shadow.hpp).
#pragma once

#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::gpu {

struct ComputeShaderConfig;
struct Rdna2Inst;
struct ShaderResource;
struct ShaderResourceTable;

enum class IndirectPointerBoundKind : uint32_t {
    None = 0,
    StaticFootprint = 1,
    DescriptorRange = 2,
};

enum class IndirectPointerGuardKind : uint32_t {
    None = 0,
    Full64NonZero = 1,
};

struct IndirectPointerAccessProof {
    uint32_t pc = UINT32_MAX;
    std::array<uint32_t, 2> words{};
    uint32_t address_vgpr = UINT32_MAX;
    uint32_t immediate_byte_offset = 0;
    uint32_t component_bytes = 0;
    uint32_t components = 0;

    bool operator==(const IndirectPointerAccessProof&) const = default;
};

struct IndirectPointerRelocationProof {
    uint32_t schema_version = 1;
    IndirectPointerBoundKind bound_kind = IndirectPointerBoundKind::None;
    IndirectPointerGuardKind guard_kind = IndirectPointerGuardKind::None;

    uint32_t source_fetch_pc = UINT32_MAX;
    std::array<uint32_t, 2> source_words{};
    uint32_t source_result_vgpr = UINT32_MAX;
    uint32_t source_stride = 0;
    uint32_t pointer_byte_offset = 0;
    // DescriptorRange retains the exact dynamic source-record identity at the producer so a
    // consumer rooted in record A cannot borrow an adjacent packed record B merely because its
    // final address happens to fall there. StaticFootprint does not need a runtime record index.
    uint32_t source_record_index_vgpr = UINT32_MAX;
    IndirectBufferRelocationRecord::SourceAddressKind source_address_kind =
        IndirectBufferRelocationRecord::SourceAddressKind::RawU64;
    uint32_t footprint_selector_byte_offset = UINT32_MAX;
    std::vector<uint32_t> selector_footprint_bytes;
    uint32_t record_count = 0;
    uint32_t max_footprint_bytes = 0;

    uint64_t fingerprint = 0;
    std::vector<uint32_t> witness_words;
    std::vector<IndirectPointerAccessProof> accesses;
    std::vector<IndirectBufferRelocationRecord> records;
};

inline constexpr uint32_t kIndirectPointerStaticFootprintTag = 0x52504653u;
inline constexpr uint32_t kIndirectPointerDescriptorRangeTag = 0x52445044u;
inline constexpr uint32_t kIndirectPointerProofSchema = 1u;
inline constexpr uint32_t kIndirectPointerMaxBindingBytes = 128u << 20u;

inline constexpr IndirectBufferRelocationLayout kIndirectPointerStaticFootprintLayout{
    kIndirectPointerStaticFootprintTag,
    2u,
    65536u,
    65536u,
    kIndirectPointerMaxBindingBytes,
    4u,
};

// DescriptorRange uses carrier v3 because its source record is a complete V# whose Base48 address
// is not the literal 64-bit value of words 0/1. Version 2's final record dword remains reserved and
// zero, preserving already-merged v2 carrier compatibility.
inline constexpr IndirectBufferRelocationLayout kIndirectPointerDescriptorRangeLayout{
    kIndirectPointerDescriptorRangeTag,
    3u,
    65536u,
    65536u,
    kIndirectPointerMaxBindingBytes,
    4u,
};

// Prove the complete direct-record StaticFootprint shape currently emitted by the retained
// Wave64 kernel. The matcher is structural: unrelated packet changes remain admissible, while every
// producer, mask/guard, branch target, and GLOBAL consumer that grants the bound is exact.
bool analyze_rdna2_static_pointer_footprint(
    const uint32_t* code, size_t dwords,
    const ComputeShaderConfig& config,
    const ShaderResourceTable& resources,
    IndirectPointerRelocationProof& proof);

// Prove the bounded start/count -> main-record -> stride-16 V# producer shared by the retained
// Wave32 siblings. The complete V# supplies the range; every GLOBAL consumer remains authorized by
// exact PC+packet and by the producer's dynamic source-record identity.
bool analyze_rdna2_descriptor_pointer_range(
    const uint32_t* code, size_t dwords,
    const ComputeShaderConfig& config,
    const ShaderResourceTable& resources,
    IndirectPointerRelocationProof& proof);

// Materialize or remint the generic v2 relocation binding, then independently re-establish its
// shader/data proof. Optional outputs let the final emitter consume only freshly validated state.
bool discover_rdna2_indirect_pointer_relocations(
    const uint32_t* code, size_t dwords,
    const ComputeShaderConfig& config,
    ShaderResourceTable& resources);

bool validate_rdna2_indirect_pointer_relocations(
    const uint32_t* code, size_t dwords,
    const ComputeShaderConfig& config,
    const ShaderResourceTable& resources,
    IndirectPointerRelocationProof* validated_proof = nullptr,
    IndirectBufferRelocationInfo* validated_info = nullptr);

bool is_indirect_pointer_relocation_marker_candidate(const ShaderResource& resource);
bool is_indirect_pointer_relocation_resource(const ShaderResource& resource);

// Syntax-only capture classifier. Compilation authority always requires the proof-bearing
// validation function above.
bool is_indirect_pointer_relocation_serialized(
    const ShaderResource& resource, const uint8_t* bytes, size_t byte_count);

// Return the exact packet-authorized relocated access, or nullptr. A PC match alone is never
// authority; callers at the final emission boundary must use this helper with the decoded packet.
const IndirectPointerAccessProof* rdna2_indirect_pointer_access(
    const IndirectPointerRelocationProof& proof,
    const Rdna2Inst& instruction);

// DescriptorRange captures the dynamic source-record identity at one exact producer packet. This
// helper is the final-boundary authority check; a matching PC without matching packet words is not
// a source producer.
bool rdna2_indirect_pointer_source(
    const IndirectPointerRelocationProof& proof,
    const Rdna2Inst& instruction);

} // namespace prosper::gpu
