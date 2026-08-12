// Exact GTA V compute contract for one root-record consumer. The generic recompiler must never
// infer no-backing semantics from an invalid RDNA2 descriptor alone; this module combines complete
// program/launch identity with the live root-table bytes that select the title's inactive variant.
#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper::gpu {

struct ComputeShaderConfig;
struct Rdna2Inst;
struct ShaderResource;
struct ShaderResourceTable;

inline constexpr uint32_t kGtaCf9200RootPc = 64u;
inline constexpr uint32_t kGtaCf9200RootBytes = 224u;

// Outside the 14-bit RDNA2 V# STRIDE domain. Captures already serialize this field, so the marker
// retains a distinct fail-closed shape without changing their on-disk resource schema.
inline constexpr uint32_t kGtaCf9200NoBackingStride = UINT32_MAX - 2u;

enum class GtaCf9200NoBackingAccess : uint8_t {
    None,
    LoadZero,
    DropStore,
};

bool rdna2_gta5_cf9200_shader(const uint32_t* code, size_t dwords);
bool rdna2_gta5_cf9200_launch(
    const uint32_t* code, size_t dwords,
    const ComputeShaderConfig& config);

// Exact packet/PC identity for the one zero-producing load and fourteen droppable stores. Complete
// dispatch validation remains mandatory before a marker can authorize either behavior.
GtaCf9200NoBackingAccess rdna2_gta5_cf9200_no_backing_site(const Rdna2Inst& instruction);

bool is_gta5_cf9200_no_backing_marker_candidate(const ShaderResource& resource);
bool is_proven_gta5_cf9200_no_backing(const ShaderResource& resource);

// Read the current root at command-ordered realization. Only the exact program, 1x1x1 launch,
// ordinary 224-byte self descriptor, and established application-record witnesses may manufacture
// exact-PC no-backing markers. A normal valid source/output pair is intentionally left untouched.
bool discover_rdna2_gta5_cf9200_no_backing(
    const uint32_t* code, size_t dwords,
    const ComputeShaderConfig& config,
    ShaderResourceTable& resources);

// Final compiler/cache/capture trust boundary. Repeats byte, launch, root, witness, and complete
// marker-set validation without mutating the table.
bool rdna2_gta5_cf9200_no_backing_dispatch(
    const uint32_t* code, size_t dwords,
    const ComputeShaderConfig& config,
    const ShaderResourceTable& resources);

} // namespace prosper::gpu
