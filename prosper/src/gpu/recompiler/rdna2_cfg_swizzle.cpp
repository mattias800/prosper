#include "gpu/recompiler/rdna2_cfg_swizzle.hpp"
#include "gpu/recompiler/fragment_packet_definedness.hpp"

namespace prosper::gpu {
bool emit_cfg_swizzle_phase(SpirvCompute& b, const std::unordered_set<uint32_t>& swizzle_pcs,
                            uint32_t swizzle_pending_var, uint32_t swizzle_active_var,
                            uint32_t swizzle_source_var, uint32_t swizzle_source_lane_var,
                            uint32_t swizzle_dst_var, uint32_t zero, uint32_t no,
                            const std::map<int, uint32_t>& vv,
                            const std::map<std::pair<int, int>, uint32_t>& lv,
                            const std::map<std::pair<int, int>, uint32_t>& lmv,
                            PacketVgprDefinedness* definedness) {
    // DS_SWIZZLE common phase. The dispatcher cases only publish source data and a lane selector;
    // every invocation executes the actual subgroup gathers here in uniform control flow. This is
    // required even though the instruction does not touch LDS: subgroup operations in a divergent
    // switch arm would have undefined participation on Vulkan.
    if (!swizzle_pcs.empty()) {
        const uint32_t swizzle_pending = b.load_function(b.t_bool, swizzle_pending_var);
        const uint32_t swizzle_active = b.load_function(b.t_bool, swizzle_active_var);
        const uint32_t swizzle_lane = b.load_function(b.t_u32, swizzle_source_lane_var);
        uint32_t swizzle_result = 0;
        if (b.is_fragment_packet()) {
            if (!definedness || b.wave_size != 64 || b.local_count != 64 || b.native_subgroup_size)
                return false;
            // Original source values/validity are captured BEFORE any destination update, even
            // when VDST==ADDR. All64 workers rendezvous in this uniform dispatcher merge;
            // neither native subgroup width nor guest EXEC controls physical participation.
            b.cfg_scratch_store(b.linear_localid, b.load_function(b.t_u32, swizzle_source_var));
            definedness->publish_peer(b, swizzle_pending);
            b.barrier();
            definedness->consume_peer(b, b.land(swizzle_pending, swizzle_active), swizzle_lane,
                                      FragmentPacketVgprRead::QuadPeer);
            swizzle_result = b.cfg_scratch_load(swizzle_lane);
            // Complete every selected OLD source read before any later phase reuses scratch.
            b.barrier();
            // RDNA2 12.13.1's thread_valid wording alone is not an EXEC/inactive-zero proof.
            // The initial owned domain refuses an active consumer of an inactive/absent peer;
            // WQM recipes prove all four genuine live/backed values before this instruction.
        } else {
            const uint32_t swizzle_value =
                b.subgroup_shuffle(b.load_function(b.t_u32, swizzle_source_var), swizzle_lane);
            const uint32_t source_active_word =
                b.sel(b.land(swizzle_pending, swizzle_active), b.uconst(1), zero);
            const uint32_t source_active =
                b.ucmp(Op_INotEqual, b.subgroup_shuffle(source_active_word, swizzle_lane), zero);
            swizzle_result = b.sel(source_active, swizzle_value, zero);
        }
        const uint32_t swizzle_dst = b.load_function(b.t_u32, swizzle_dst_var);
        const uint32_t swizzle_write = b.land(swizzle_pending, swizzle_active);
        for (const auto& kv : vv) {
            const uint32_t selected =
                b.land(swizzle_write,
                       b.ucmp(Op_IEqual, swizzle_dst, b.uconst(static_cast<uint32_t>(kv.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, swizzle_result, old));
        }
        // An ordinary VGPR definition ends a scalar lane-spill lifetime at that physical register.
        for (const auto& kv : lv) {
            const uint32_t selected =
                b.land(swizzle_pending, b.ucmp(Op_IEqual, swizzle_dst,
                                               b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, zero, old));
        }
        for (const auto& kv : lmv) {
            const uint32_t selected =
                b.land(swizzle_pending, b.ucmp(Op_IEqual, swizzle_dst,
                                               b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_bool, kv.second);
            b.store_function(kv.second, b.bsel(selected, no, old));
        }
    }

    return true;
}
}   // namespace prosper::gpu
