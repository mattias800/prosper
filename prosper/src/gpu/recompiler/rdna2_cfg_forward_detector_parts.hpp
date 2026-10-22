// Private forward-if detector passes; original closures remain with the detector.
#pragma once
#include "gpu/recompiler/rdna2_cfg_divergent_shapes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_cfg_forward_admission.hpp"
#include <functional>
#include <optional>

namespace prosper::gpu {
void forward_if_break_shape(const prosper::gpu::Rdna2Inst& in, const DivLoop*& L,
                            std::vector<ForwardIf>& out);
std::optional<std::vector<ForwardIf>>
forward_if_else_shape(bool& early, uint32_t& tgt, uint32_t& end_pc,
                      const std::vector<Rdna2Inst>& ins, const prosper::gpu::Rdna2Inst& in,
                      const std::function<bool(uint32_t)>& loop_backedge,
                      RecompileDiagnosticContext& diagnostic,
                      const std::function<std::vector<ForwardIf>()>& reject, ForwardIf& F);
void forward_if_workgroup_mark(const std::vector<Rdna2Inst>& ins, const prosper::gpu::Rdna2Inst& in,
                               bool& compute_uniform_vcc, bool& compute_wave_branches,
                               ForwardIf& F);
std::optional<std::vector<ForwardIf>>
forward_if_collect(const std::vector<Rdna2Inst>& ins, bool& allow_vcc, bool& compute_wave_branches,
                   const std::unordered_set<uint32_t>*& skip, const std::vector<DivLoop>*& loops,
                   const std::function<std::vector<ForwardIf>(const ForwardIfRefusal&)>& decline,
                   RecompileDiagnosticContext& diagnostic,
                   const std::function<std::vector<ForwardIf>()>& reject,
                   std::vector<ForwardIf>& out, uint32_t& end_pc, const uint32_t*& code,
                   size_t& dwords, const std::function<bool(uint32_t)>& loop_backedge);
std::optional<std::vector<ForwardIf>>
forward_if_claimed_branches(const std::vector<Rdna2Inst>& ins,
                            const std::function<bool(uint32_t)>& loop_backedge,
                            std::vector<ForwardIf>& out, RecompileDiagnosticContext& diagnostic,
                            const std::function<std::vector<ForwardIf>()>& reject);
std::optional<std::vector<ForwardIf>>
forward_if_nesting(const std::vector<DivLoop>*& loops, std::vector<ForwardIf>& out,
                   std::vector<uint32_t>& open, RecompileDiagnosticContext& diagnostic,
                   const std::function<std::vector<ForwardIf>()>& reject);
}  // namespace prosper::gpu
