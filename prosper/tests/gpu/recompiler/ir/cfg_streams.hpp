// cfg_streams.hpp -- synthetic RDNA2 streams for the ir/cfg tests: one encoder per instruction
// the CFG tests need, and a decode helper. Branch offsets are SIMM16 dwords relative to the
// instruction after the branch (target = pc + 1 + simm).
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <vector>

namespace prosper::test::cfg_streams {

inline constexpr uint32_t kEndpgm = 0xBF810000u;
inline constexpr uint32_t kNop = 0xBF800000u;
inline constexpr uint32_t kBarrier = 0xBF8A0000u;
inline constexpr uint32_t kTrap = 0xBF920000u;
inline constexpr uint32_t kSMovS0Zero = 0xBE800380u;   // s_mov_b32 s0, 0
inline constexpr uint32_t kSMovVccLoZero = 0xBEEA0380u;   // s_mov_b32 vcc_lo, 0
inline constexpr uint32_t kVNop = 0x7E000000u;   // v_nop
inline constexpr uint32_t kSetpc = 0xBE802000u;   // s_setpc_b64 s[0:1]
inline constexpr uint32_t kSwappc = 0xBE822100u;   // s_swappc_b64 s[2:3], s[0:1]
inline constexpr uint32_t kCodeEnd = 0xBF9F0000u;
inline constexpr uint32_t kEndpgmSaved = 0xBF9B0000u;

// SOPP opcodes.
inline constexpr uint32_t kBranch = 0x02, kScc0 = 0x04, kScc1 = 0x05, kVccz = 0x06, kVccnz = 0x07,
                          kExecz = 0x08, kExecnz = 0x09;

inline uint32_t sopp(uint32_t opcode, int simm) {
    return 0xBF800000u | (opcode << 16) | (static_cast<uint32_t>(simm) & 0xFFFFu);
}
// s_call_b64 s[0:1], simm (SOPK opcode 0x16).
inline uint32_t call(int simm) {
    return 0xB0000000u | (0x16u << 23) | (static_cast<uint32_t>(simm) & 0xFFFFu);
}

inline std::vector<prosper::gpu::Rdna2Inst> walk(const std::vector<uint32_t>& words) {
    std::vector<prosper::gpu::Rdna2Inst> ins;
    prosper::gpu::rdna2_walk(words.data(), words.size(), ins);
    return ins;
}

}   // namespace prosper::test::cfg_streams
