#pragma once

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;

// Complete raw guest groups AND complete native guest waves. This grants only original entry
// participation; current producer/source and immutable-program proofs are still required.
bool native_compute_has_complete_entry(const SpirvCompute&);
// Authenticate the launch identity captured by the saved-mask original-body initializer.
// Fragments retain their separate exact-subgroup contract; compute tokens cannot become fragments.
bool saved_mask_physical_launch_matches_original(const SpirvCompute&);
// Conservative original launch-DATA precondition for the uniform EXEC-to-MOV producer prefix.
bool native_uniform_mov_entry_sources(const RegState*, const Rdna2Inst&);
bool native_uniform_exec_mov_has_full_participation(const SpirvCompute&, const Rdna2Inst&);
} // namespace prosper::gpu
