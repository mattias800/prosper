// Title-observed GTA V compute contracts whose safety depends on complete shader and dispatch
// identity. Keeping them outside the generic decoder makes the trust boundary explicit and gives
// resource discovery, cached compilation, capture replay, and final emission one shared proof.
#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper::gpu {

struct ComputeShaderConfig;
struct Rdna2Inst;
struct ShaderResourceTable;

// Exact raw-store consumers in GTA V 0x413cf9a00's null-output region. This is packet identity only;
// callers must independently prove the pc42..48 EXEC guard and the dispatch's null pointer.
bool rdna2_gta5_null_guarded_raw_store_site(const Rdna2Inst& in);

// Complete byte and direct-CFG identity for GTA V 0x413cf9a00's pc42..80 nullable-output guard.
// This proves only the static program shape; callers must independently prove that entry user SGPRs
// s2:s3 are zero for the dispatch before treating the exact pc74/76/78 stores as no-ops.
bool rdna2_gta5_null_guarded_raw_store_shader(const uint32_t* code, size_t dwords);

// Complete dispatch proof for the same conditional stores. In addition to the static shader shape,
// entry s2:s3 must be null and no decoded scalar destination may overlap either word before pc42.
// This is the transferable trust-boundary check used by cached compilation and capture replay.
bool rdna2_gta5_null_guarded_raw_store_dispatch(
    const uint32_t* code, size_t dwords,
    const uint32_t* user_sgprs, size_t user_sgpr_count);

enum class Gta5NullableOutputAccess : uint8_t {
    None,
    LoadDword,
    StoreDword,
};

// Exact packet/PC identity for the four nullable raw-buffer consumers in GTA V's two production
// kernels. Complete program and dispatch validation remains a separate requirement.
Gta5NullableOutputAccess rdna2_gta5_nullable_output_site(const Rdna2Inst& in);

// Complete byte identity for the two production kernels. The comparison stops at their exact
// S_ENDPGM, so live callers may continue to pass the conservatively-sized mapped shader window.
bool rdna2_gta5_nullable_output_shader(const uint32_t* code, size_t dwords);

// Static program plus exact one-dimensional launch ABI. This does not itself prove the mapped
// +0x20 qword was null; discovery supplies that witness before manufacturing a marker.
bool rdna2_gta5_nullable_output_launch(
    const uint32_t* code, size_t dwords,
    const uint32_t* user_sgprs, size_t user_sgpr_count,
    uint32_t local_x, uint32_t local_y, uint32_t local_z,
    uint32_t threads_x, uint32_t threads_y, uint32_t threads_z,
    bool exact_thread_extent, uint32_t wave_size,
    bool tgid_x_en, bool tgid_y_en, bool tgid_z_en,
    uint32_t tidig_comp_cnt);

// Final trust boundary: repeat byte/launch validation, require the complete exact marker set, and
// read its retained 40-byte table witness to prove s0:s1+0x20 is still the zero qword.
bool rdna2_gta5_nullable_output_dispatch(
    const uint32_t* code, size_t dwords,
    const ComputeShaderConfig& config,
    const ShaderResourceTable& resources);

} // namespace prosper::gpu
