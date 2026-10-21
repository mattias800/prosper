// Interior and final-shape admission for divergent loops.
#pragma once
#include "gpu/recompiler/rdna2_cfg_divergent_shapes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
DivLoopRefusal admit_div_loop(const std::vector<Rdna2Inst>& ins,
                              const std::unordered_set<uint32_t>& safe,
                              bool exact_fragment_wave_breaks, std::vector<DivLoop>& out, size_t li,
                              uint32_t backedge_op_li);
DivLoopRefusal admit_div_loop_exit(uint32_t& tgt, DivLoop& L, const prosper::gpu::Rdna2Inst& in,
                                   const bool& execnz, bool& exact_fragment_wave_breaks,
                                   const std::vector<Rdna2Inst>& ins);
DivLoopRefusal admit_div_loop_final_shape(DivLoop& L, const bool& scc_do_while,
                                          const std::vector<Rdna2Inst>& ins,
                                          uint32_t& backedge_op_li, const bool& execnz,
                                          const std::unordered_set<uint32_t>& safe);
}  // namespace prosper::gpu
