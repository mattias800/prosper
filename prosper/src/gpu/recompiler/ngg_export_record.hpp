// ngg_export_record.hpp -- the export record buffer a merged ES+GS NGG subgroup shell writes, and
// that a later pass-through draw reads (#3135: P2 writes it, P3/P4 consume it).
//
// One Vulkan workgroup runs one guest subgroup of W Wave64 waves (64*W invocations). The shell
// writes one BLOCK per workgroup into the export buffer (descriptor set 2, binding 1):
//
//   block g starts at word  g * block_words(W)
//   [0] verts_alloc        GS_ALLOC_REQ M0[11:0], written by lane 0 of wave 0
//   [1] prims_alloc        GS_ALLOC_REQ M0[23:12] (prims start at bit 12)
//   [2] alloc_requests     how many times wave 0 sent GS_ALLOC_REQ (atomic count)
//   [3] stray_requests     how many times any OTHER wave sent it (atomic count)
//   [4] launch_mismatches  how many waves launched with an s3 that disagrees with the compiled
//                          shell: s3[31:28] != W or s3[27:24] != the wave's index (atomic count)
//   [5 + t * words_per_lane ...]  the record of subgroup thread t = wave * 64 + lane, t < 64*W
//
// A block is valid only when alloc_requests == 1, stray_requests == 0 and launch_mismatches == 0;
// anything else is a guest-protocol or launch violation the consumer must treat as "no
// primitives". The launch check matters because the guest sizes its own allocation from s3's wave
// count: a block launched with the wrong W computes different triangles, not missing ones. The buffer is zeroed
// before every dispatch (vkCmdFillBuffer), so an unwritten word is 0 and an unwritten flag bit
// means "not exported".
//
// The record of one thread, in words:
//   0                flags: bit0 PRIM written, bit1 POS0 written, bit2 POS1 written,
//                    bit(3+k) param_targets[k] written. A bit is set only for lanes whose EXEC
//                    bit was on when the EXP executed.
//   1                PRIM: the raw export word (vertex indices at [8:0], [18:10], [28:20], null at
//                    bit 31). Indices name subgroup THREADS (records), not input vertices.
//   2..5             POS0.xyzw raw bits
//   6..9             POS1.xyzw raw bits, present only when the program exports POS1
//                    (pos1_word != kNggRecordAbsent). The layer is POS1.z.
//   then             one vec4 per PARAM target the program exports, ascending target order.
//
// Channels a target's EXP did not enable stay 0; `*_channels` says which ones are meaningful.
// The record carries no POS2..4, MRT, or compressed exports: the shell refuses programs that
// export them.
#pragma once

#include <cstdint>
#include <vector>

namespace prosper::gpu {

inline constexpr uint32_t kNggRecordAbsent = UINT32_MAX;
inline constexpr uint32_t kNggSubgroupHeaderWords = 5;
inline constexpr uint32_t kNggHeaderVertsAlloc = 0;
inline constexpr uint32_t kNggHeaderPrimsAlloc = 1;
inline constexpr uint32_t kNggHeaderAllocRequests = 2;
inline constexpr uint32_t kNggHeaderStrayRequests = 3;
inline constexpr uint32_t kNggHeaderLaunchMismatches = 4;

inline constexpr uint32_t kNggRecordFlagsWord = 0;
inline constexpr uint32_t kNggRecordPrimWord = 1;
inline constexpr uint32_t kNggRecordPos0Word = 2;
inline constexpr uint32_t kNggFlagPrim = 1u << 0;
inline constexpr uint32_t kNggFlagPos0 = 1u << 1;
inline constexpr uint32_t kNggFlagPos1 = 1u << 2;
inline constexpr uint32_t kNggFlagParamShift = 3;
// 29 PARAM targets fit the 32-bit flags word; a program exporting more is refused.
inline constexpr uint32_t kNggMaxParamTargets = 32 - kNggFlagParamShift;

// EXP target numbers (RDNA2 ISA, EXP TARGET field).
inline constexpr uint32_t kExpTargetPos0 = 12;
inline constexpr uint32_t kExpTargetPos1 = 13;
inline constexpr uint32_t kExpTargetPrim = 20;
inline constexpr uint32_t kExpTargetParam0 = 32;

struct NggExportRecordLayout {
    uint32_t words_per_lane = 0;   // record stride
    uint32_t pos1_word = kNggRecordAbsent;
    uint32_t first_param_word = kNggRecordAbsent;
    std::vector<uint32_t> param_targets;   // EXP targets 32+k in record order (ascending)
    // The EXP enable mask (bit c = channel c) each recorded target was exported with.
    uint32_t prim_channels = 0;
    uint32_t pos0_channels = 0;
    uint32_t pos1_channels = 0;
    std::vector<uint32_t> param_channels;   // parallel to param_targets

    uint32_t param_word(uint32_t index) const { return first_param_word + 4u * index; }
    // Words in one workgroup's block for a subgroup of `waves` Wave64 waves.
    uint32_t block_words(uint32_t waves) const {
        return kNggSubgroupHeaderWords + 64u * waves * words_per_lane;
    }
    uint32_t record_offset(uint32_t waves, uint32_t block, uint32_t thread) const {
        return block * block_words(waves) + kNggSubgroupHeaderWords + thread * words_per_lane;
    }
};

}   // namespace prosper::gpu
