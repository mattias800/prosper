// ngg_subgroup_shell.hpp -- compile a LINKED merged ES+GS NGG program into a compute module that runs
// one guest subgroup per Vulkan workgroup and writes every lane's exports to a record buffer
// (#3135 phase P2). Compile and offline execution only: nothing live dispatches it yet.
//
// Execution model. LocalSize is L * waves, one guest wave per L invocations (L = 64, or 32 for a
// Wave32 program: VGT_SHADER_STAGES_EN.GS_W32_EN), so the guest's waves share LDS and s_barrier
// exactly as in one hardware subgroup. The host runs one dispatch per
// wave count W, one workgroup per subgroup of that W; a planner (gpu/execute/ngg_subgroup_plan)
// supplies the launch values.
//
// Shell I/O lives in descriptor set 2, so the guest's own resources keep their set-0 bindings:
//
//   set 2, binding 0  LAUNCH records, kNggLaunchWordsPerLane raw uint words per invocation:
//                     v0..v8, then s3. Invocation (g, i) -- workgroup g, guest thread
//                     i = wave * L + lane -- reads record g * L * W + i. That is the identity
//                     mapping from workgroup ordinal to subgroup: there is no subgroup table yet,
//                     so the host lays the records of the subgroups it dispatches out in dispatch
//                     order and binds the buffer at that dispatch's first record. s3 must be equal
//                     for the L lanes of a wave (the planner's ngg_merged_wave_info). s2 is not
//                     stored: every invocation derives it from the s3 of its subgroup's waves
//                     (ES and GS thread counts summed over the W records at g * L * W + w * L),
//                     which is ngg_group_info by construction and needs no capsule change.
//   set 2, binding 1  EXPORT records: one block per workgroup, see ngg_export_record.hpp.
//
// Push constants carry the user SGPRs: word k is s(8 + k). When the user-data address is known, the
// two words after them are s0 and s1.
//
// GS_ALLOC_REQ (`s_sendmsg 9`) stores M0[11:0] / M0[23:12] into the block header from lane 0 of
// wave 0, and counts every request so the consumer can reject a subgroup whose program requested
// zero times, twice, or from another wave. Lane 0 of every wave also checks its s3 against the
// compiled shell (s3[31:28] == waves, s3[27:24] == its wave index) and counts a mismatch. Barriers keep the barrier-phase proof
// (`analyze_barrier_phased_compute`), which refuses a per-wave terminal guard before a barrier.
//
// Everything ngg_subgroup_abi.hpp refuses is refused here first, with its named reason in
// `refusal`. A refusal by the compiler proper after admission is reported as
// "reason=ngg-compile-rejected" with the recompiler's own terminal reason appended.
#pragma once

#include "gpu/recompiler/ngg_export_record.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {

inline constexpr uint32_t kNggLaunchWordsPerLane = 10;   // v0..v8, s3
inline constexpr uint32_t kNggLaunchS3Word = 9;
inline constexpr uint32_t kNggShellDescriptorSet = 2;
inline constexpr uint32_t kNggShellLaunchBinding = 0;
inline constexpr uint32_t kNggShellExportBinding = 1;

inline constexpr uint32_t kNggLdsGranuleDwords = 128;   // RSRC2_GS.LDS_SIZE unit (512 bytes)
inline constexpr uint32_t kNggMaxLdsGranules = 128;   // 64 KiB

// The LDS_SIZE field [26:19] of a raw SPI_SHADER_PGM_RSRC2_GS value.
inline uint32_t ngg_rsrc2_gs_lds_size(uint32_t spi_shader_pgm_rsrc2_gs) {
    return (spi_shader_pgm_rsrc2_gs >> 19) & 0xffu;
}

struct NggSubgroupShellConfig {
    uint32_t waves = 1;   // 1..4 Wave64 waves, or 1..8 Wave32 waves; LocalSize is lanes * waves
    uint32_t wave_lanes = 64;   // 64, or 32 for a Wave32 program (GS_W32_EN)
    // The RAW SPI_SHADER_PGM_RSRC2_GS.LDS_SIZE field: 512-byte (128-dword) granules, at most 128.
    // Decode it with ngg_rsrc2_gs_lds_size(); never pass bytes or dwords here. A program that uses
    // LDS with this left at 0 is refused (ngg-shell-config cause=lds-unsized).
    uint32_t rsrc2_gs_lds_size = 0;
    uint32_t user_sgprs = 0;   // s8.. count (AGC user_data_range_end); one push-constant word each
    bool user_data_address_known = false;   // two more push-constant words supply s0:s1
    // Require one exact Vulkan subgroup of `wave_lanes` invocations per guest wave. The pipeline must
    // then be created with requiredSubgroupSize = wave_lanes and full subgroups; without it the
    // portable shell emulates cross-lane operations through workgroup memory. (The name predates
    // Wave32: for a Wave32 program it means a native 32-lane subgroup.)
    bool native_wave64 = false;
};

// Returns the SPIR-V module, or {} with `refusal` set ("reason=<name> ..."). `layout_out`, when
// given, receives the export record layout the module writes.
std::vector<uint32_t> recompile_ngg_subgroup(
    const uint32_t* linked_code, size_t dwords, const ShaderResourceTable* resources,
    const NggSubgroupShellConfig& config, NggExportRecordLayout* layout_out = nullptr,
    RecompileDiagnosticContext diagnostic = {RecompileDiagnosticStage::Vertex, 0},
    std::string* refusal = nullptr);

}   // namespace prosper::gpu
