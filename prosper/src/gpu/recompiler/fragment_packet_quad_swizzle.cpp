#include "gpu/recompiler/fragment_packet_quad_swizzle.hpp"
#include "gpu/recompiler/fragment_packet_exports.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_waitcnt.hpp"
#include <algorithm>
#include <bitset>
#include <map>

namespace prosper::gpu {
const char* packet_quad_swizzle_gap(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::DS || in.opcode != 0x35 || in.ds_gds ||
        in.src[0].kind != OperandKind::VGPR || in.dst.kind != OperandKind::VGPR ||
        in.src[0].value < 0 || in.src[0].value > 255 || in.dst.value < 0 || in.dst.value > 255 ||
        in.n_src != 3 || in.len_dwords != 2 || in.has_modifier || in.has_sdwa || in.has_dpp)
        return "packet-quad-swizzle-form-unimplemented";
    if (in.literal >= 0xc000u || !(in.literal & 0x8000u))
        return "packet-quad-swizzle-mode-unimplemented";
    return nullptr;
}
uint32_t packet_quad_swizzle_source_lane(SpirvCompute& b, const Rdna2Inst& in) {
    if (!b.is_fragment_packet() || b.wave_size != 64 || b.local_count != 64 ||
        b.native_subgroup_size || packet_quad_swizzle_gap(in))
        return 0;
    const auto lane = b.guest_lane_id();
    const auto quad_lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(3));
    auto selected = b.uconst(in.literal & 3u);
    for (uint32_t output_lane = 1; output_lane < 4; ++output_lane)
        selected = b.sel(b.ucmp(Op_IEqual, quad_lane, b.uconst(output_lane)),
                         b.uconst((in.literal >> (2u * output_lane)) & 3u), selected);
    return b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, lane, b.uconst(~3u)), selected);
}
const char* packet_quad_swizzle_completion_gap(const std::vector<Rdna2Inst>& ins, uint32_t& pc) {
    // This is not a second CFG admission inventory for existing DS-free packet programs.
    // In particular, their already-proved backedges must remain outside this forward-only proof.
    if (std::none_of(ins.begin(), ins.end(),
                     [](const auto& in) { return in.fmt == Rdna2Format::DS && in.opcode == 0x35; }))
        return nullptr;
    std::map<uint32_t, size_t> indices;
    for (size_t i = 0; i < ins.size(); ++i) indices.emplace(ins[i].pc, i);
    std::vector<std::bitset<256>> entry(ins.size());
    std::vector<bool> reached(ins.size());
    if (ins.empty()) return nullptr;
    reached.front() = true;
    const auto join = [&](size_t next, const std::bitset<256>& pending) {
        reached[next] = true;
        entry[next] |= pending;
    };
    for (size_t i = 0; i < ins.size(); ++i) {
        if (!reached[i]) continue;
        const auto& in = ins[i];
        pc = in.pc;
        auto pending = entry[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0c) {
            if (decode_rdna2_waitcnt(uint16_t(in.simm16)).drains_scalar_reads()) pending.reset();
        } else {
            for (uint32_t source = 0; source < in.n_src; ++source) {
                // DATA0/DATA1 are unused encoding fields of DS_SWIZZLE, not guest reads.
                if (in.fmt == Rdna2Format::DS && source) continue;
                if (in.fmt == Rdna2Format::EXP &&
                    !(fragment_packet_export_source_mask(in.exp_en, in.exp_compr) & (1u << source)))
                    continue;
                const auto& operand = in.src[source];
                if (operand.kind != OperandKind::VGPR) continue;
                const auto words = in.fmt == Rdna2Format::DS ? 1u
                                   : in.fmt == Rdna2Format::MIMG && source == 0
                                       ? (in.opcode == 0x24 ? 3u : 2u)
                                       : rdna2_vgpr_source_span(in, source);
                for (uint32_t word = 0; word < words; ++word)
                    if (operand.value + word < 256 && pending.test(operand.value + word))
                        return "packet-quad-swizzle-result-read-before-lgkm-wait";
            }
            if (in.fmt == Rdna2Format::VINTRP && in.opcode == 1 && pending.test(in.dst.value))
                return "packet-quad-swizzle-result-read-before-lgkm-wait";
            for (uint32_t word = 0; word < rdna2_vgpr_write_count(in); ++word)
                if (pending.test(in.dst.value + word))
                    return "packet-quad-swizzle-result-overwrite-before-lgkm-wait";
            if (in.fmt == Rdna2Format::DS && in.opcode == 0x35) pending.set(in.dst.value);
        }
        if (in.is_end) {
            if (pending.any()) return "packet-quad-swizzle-result-pending-at-end";
            continue;
        }
        const bool branch = in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
        if (branch) {
            const auto target = branch_target(in);
            if (target <= in.pc || !indices.contains(target))
                return "packet-quad-swizzle-control-flow-unimplemented";
            join(indices.at(target), pending);
        }
        if ((!branch || in.opcode != 2) && i + 1 < ins.size()) join(i + 1, pending);
    }
    pc = UINT32_MAX;
    return nullptr;
}
}   // namespace prosper::gpu
