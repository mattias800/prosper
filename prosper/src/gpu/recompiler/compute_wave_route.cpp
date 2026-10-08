// Compute Wave64 route selection -- see compute_wave_route.hpp and ADR 0028.
#include "gpu/recompiler/compute_wave_route.hpp"

#include "diagnostics/env_cache.hpp"
#include "diagnostics/perf/wave64_refusal.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"

namespace prosper::gpu {

const char* compute_cross_lane_kind_name(ComputeCrossLaneKind kind) {
    switch (kind) {
        case ComputeCrossLaneKind::ReadLane: return "readlane";
        case ComputeCrossLaneKind::ReadFirstLane: return "readfirstlane";
        case ComputeCrossLaneKind::Mbcnt: return "mbcnt";
        case ComputeCrossLaneKind::WaveVote: return "wave-vote";
        case ComputeCrossLaneKind::Ballot: return "ballot";
        case ComputeCrossLaneKind::MaskConsumer: return "mask-consumer";
        case ComputeCrossLaneKind::Dpp: return "dpp";
        case ComputeCrossLaneKind::PermLane: return "permlane";
        case ComputeCrossLaneKind::DsSwizzle: return "ds-swizzle";
        case ComputeCrossLaneKind::DsBpermute: return "ds-bpermute";
        case ComputeCrossLaneKind::DsAppend: return "ds-append";
        case ComputeCrossLaneKind::MaskScc: return "mask-scc";
        case ComputeCrossLaneKind::MaskOther: return "mask-other";
        case ComputeCrossLaneKind::WriteLane: return "writelane";
        case ComputeCrossLaneKind::LdsWaveSync: return "lds-wave-sync";
        case ComputeCrossLaneKind::DsPermute: return "ds-permute";
        case ComputeCrossLaneKind::Count: break;
    }
    return "none";
}

const char* compute_wave_route_name(ComputeWaveRoute route) {
    // The single route vocabulary is owned by #4754 (wave64_refusal.hpp); use its table directly.
    using prosper::diagnostics::perf::wave64_route_name;
    using prosper::diagnostics::perf::Wave64Route;
    switch (route) {
        case ComputeWaveRoute::Native: return wave64_route_name(Wave64Route::Native);
        case ComputeWaveRoute::WidthIndependent:
            return wave64_route_name(Wave64Route::ProvenWidthIndependent);
        case ComputeWaveRoute::WorkgroupExchange:
            return wave64_route_name(Wave64Route::WorkgroupExchange);
        case ComputeWaveRoute::NeedsNLanes: return wave64_route_name(Wave64Route::NLanes);
        case ComputeWaveRoute::Refused: return wave64_route_name(Wave64Route::Refused);
    }
    return wave64_route_name(Wave64Route::Refused);
}

uint32_t compute_exchange_scratch_bytes(uint32_t local_invocations, uint32_t guest_wave) {
    if (guest_wave == 0) return 0;
    const uint32_t waves = (local_invocations + guest_wave - 1u) / guest_wave;
    const uint32_t padded = waves * guest_wave;
    // An upper bound over both lowering families. The scratch-vote helpers declare
    // `local + waves` u32 slots (SpirvCompute::declare_wave_lds); the exact dispatcher declares up
    // to two per-lane planes over the padded wave count, one result slot per wave, the group-active
    // slot and two mask-result slots (emit_cfg_state_machine's declare_cfg_scratch).
    return std::max(local_invocations + waves, 2u * padded + waves + 3u) * 4u;
}

bool compute_exchange_carries(ComputeCrossLaneKind kind) {
    switch (kind) {
        // Lowered through workgroup memory (SpirvCompute::guest_wave_* and the portable bpermute).
        case ComputeCrossLaneKind::ReadLane:
        case ComputeCrossLaneKind::ReadFirstLane:
        case ComputeCrossLaneKind::Mbcnt:
        case ComputeCrossLaneKind::WaveVote:
        case ComputeCrossLaneKind::Ballot:
        case ComputeCrossLaneKind::MaskConsumer:
        case ComputeCrossLaneKind::MaskScc:
        case ComputeCrossLaneKind::MaskOther:
        case ComputeCrossLaneKind::DsBpermute: return true;
        // Native shuffles only, and the wave-collective counter: no exchange form exists, so a
        // program using them is refused unless the host subgroup holds the whole shuffle domain.
        case ComputeCrossLaneKind::Dpp:
        case ComputeCrossLaneKind::PermLane:
        case ComputeCrossLaneKind::DsSwizzle:
        case ComputeCrossLaneKind::DsAppend:
        case ComputeCrossLaneKind::WriteLane:
        case ComputeCrossLaneKind::LdsWaveSync:
        case ComputeCrossLaneKind::DsPermute:
        case ComputeCrossLaneKind::Count: return false;
    }
    return false;
}

namespace {

// DPP control -> lanes a native subgroup shuffle needs (AMD RDNA2 70648, DPP_CTRL table).
uint32_t dpp_native_lanes(uint32_t ctrl) {
    if (ctrl < 0x100u) return 4;   // quad_perm
    if (ctrl >= 0x101u && ctrl <= 0x12fu) return 16;   // row_shl/shr/ror
    if (ctrl == 0x140u || ctrl == 0x141u) return 16;   // row_mirror / row_half_mirror
    if (ctrl == 0x142u) return 32;   // row_bcast:15 crosses a row pair
    return 64;   // wave_shl/rol/shr/ror, row_bcast:31
}

}   // namespace

ComputeWaveOpFacts analyze_compute_wave_ops(const std::vector<Rdna2Inst>& ins, const uint32_t* code,
                                            size_t dwords) {
    ComputeWaveOpFacts facts;
    facts.analyzed = true;

    const std::unordered_set<uint32_t> waterfalls = waterfall_branches(ins);
    std::unordered_set<uint32_t> safe = safe_execz_branches(ins);
    for (uint32_t pc : waterfalls) safe.insert(pc);

    uint32_t first_exec_write = UINT32_MAX;
    bool any_backward_branch = false;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (first_exec_write == UINT32_MAX && rdna2_instruction_may_change_exec(in))
            first_exec_write = in.pc;
        if (in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
            in.opcode != 0x03 && in.simm16 < 0 && !waterfalls.contains(in.pc))
            any_backward_branch = true;
    }

    const auto add = [&](const Rdna2Inst& in, ComputeCrossLaneKind kind, uint32_t native_lanes) {
        ComputeCrossLaneOp op;
        op.pc = in.pc;
        op.kind = kind;
        op.exec_full = in.pc < first_exec_write;
        op.native_lanes = native_lanes;
        facts.ops.push_back(op);
    };
    for (const auto& in : ins) {
        if (in.is_end) break;
        switch (in.fmt) {
            case Rdna2Format::VOP3:
                if (in.opcode == 0x360)
                    add(in, ComputeCrossLaneKind::ReadLane, 64);
                else if (in.opcode == 0x365 || in.opcode == 0x366) {
                    // `v_mbcnt_lo_u32_b32 vN, -1, 0` is the lane index, not a lane-mask reduction.
                    if (!(in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1))
                        add(in, ComputeCrossLaneKind::Mbcnt, 0);
                } else if (in.opcode == 0x377)
                    add(in, ComputeCrossLaneKind::PermLane, 16);
                else if (in.opcode == 0x378)
                    add(in, ComputeCrossLaneKind::PermLane, 32);
                break;
            case Rdna2Format::VOP1:
                if (in.opcode == 0x02) add(in, ComputeCrossLaneKind::ReadFirstLane, 0);
                break;
            case Rdna2Format::SOPP:
                // A scalar branch on the whole-wave EXEC/VCC mask is an any/all vote. A branch the
                // recompiler already linearizes as safe (an execz skip to the end) consumes no vote.
                if (in.opcode >= 0x06 && in.opcode <= 0x09 && !safe.contains(in.pc))
                    add(in, ComputeCrossLaneKind::WaveVote, 0);
                break;
            case Rdna2Format::SOP1:
                if (in.opcode == kSop1OpcodeBcnt1I32B64 || in.opcode == kSop1OpcodeFf1I32B64) {
                    const Operand& s = in.src[0];
                    if (s.kind == OperandKind::Special &&
                        (s.value == 126 || s.value == 127 || s.value == 106 || s.value == 107))
                        add(in, ComputeCrossLaneKind::Ballot, 0);
                    else if (s.kind == OperandKind::SGPR)
                        add(in, ComputeCrossLaneKind::MaskConsumer, 0);
                }
                break;
            case Rdna2Format::DS:
                if (in.opcode == 0x35)
                    add(in, ComputeCrossLaneKind::DsSwizzle, (in.literal & 0x8000u) ? 4u : 32u);
                else if (in.opcode == kDsOpcodeBpermuteB32)
                    add(in, ComputeCrossLaneKind::DsBpermute, 0);
                else if (in.opcode == 0x3d || in.opcode == 0x3e)
                    add(in, ComputeCrossLaneKind::DsAppend, 0);
                break;
            default: break;
        }
        if (in.has_dpp) add(in, ComputeCrossLaneKind::Dpp, dpp_native_lanes(in.dpp_ctrl));
    }
    // Scalar lane-mask sites. A "mask" operand is EXEC/VCC (specials 126/127/106/107).
    const auto is_mask_special = [](const Operand& o) {
        return o.kind == OperandKind::Special &&
               (o.value == 126 || o.value == 127 || o.value == 106 || o.value == 107);
    };
    bool lds_written_since_barrier = false;
    for (const auto& in : ins) {
        if (in.is_end) break;
        bool mask_operand = is_mask_special(in.dst);
        for (uint32_t k = 0; k < in.n_src && k < 3; ++k) mask_operand |= is_mask_special(in.src[k]);
        if (in.fmt == Rdna2Format::SOPC) {
            // s_cmp_eq/lg_u64 (0x12/0x13) read a whole 64-bit mask; the b32 forms on a mask half too.
            if (in.opcode == 0x12 || in.opcode == 0x13 || mask_operand)
                add(in, ComputeCrossLaneKind::MaskScc, 0);
        } else if (in.fmt == Rdna2Format::SOP2) {
            const bool b64_logic = in.opcode >= 0x0f && in.opcode <= 0x1d && (in.opcode & 1);
            if (b64_logic && (mask_operand || in.dst.kind == OperandKind::SGPR))
                add(in, ComputeCrossLaneKind::MaskScc, 0);
            else if (mask_operand)
                add(in, ComputeCrossLaneKind::MaskOther, 0);
        } else if (in.fmt == Rdna2Format::SOP1) {
            const bool already =
                in.opcode == kSop1OpcodeBcnt1I32B64 || in.opcode == kSop1OpcodeFf1I32B64;
            const bool scc_mask =
                in.opcode == 0x08 || in.opcode == 0x0a || (in.opcode >= 0x24 && in.opcode <= 0x2b);
            const bool plain_move = in.opcode == 0x03 || in.opcode == 0x04;
            // The saveexec family writes EXEC implicitly, so it is a mask site whatever its source.
            const bool saveexec = in.opcode >= 0x24 && in.opcode <= 0x2b;
            if (!already && scc_mask && (mask_operand || saveexec))
                add(in, ComputeCrossLaneKind::MaskScc, 0);
            else if (!already && !plain_move && !scc_mask && mask_operand)
                add(in, ComputeCrossLaneKind::MaskOther, 0);
        } else if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361) {
            // v_writelane_b32: an inline selector is the lane-slot spill form and crosses no lane.
            if (in.src[1].kind != OperandKind::InlineInt)
                add(in, ComputeCrossLaneKind::WriteLane, 64);
        } else if (in.fmt == Rdna2Format::DS && !in.ds_gds) {
            if (in.opcode == 0xb2) add(in, ComputeCrossLaneKind::DsPermute, 0);
            // A returning atomic is both: it reads what an earlier lane stored, then writes.
            if (ds_opcode_is_lds_read(in.opcode) && lds_written_since_barrier)
                add(in, ComputeCrossLaneKind::LdsWaveSync, 64);
            if (ds_opcode_is_lds_write(in.opcode)) lds_written_since_barrier = true;
        }
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a) lds_written_since_barrier = false;
    }
    std::sort(facts.ops.begin(), facts.ops.end(),
              [](const ComputeCrossLaneOp& a, const ComputeCrossLaneOp& b) { return a.pc < b.pc; });
    for (const auto& op : facts.ops) ++facts.count[static_cast<size_t>(op.kind)];
    if (facts.ops.empty()) return facts;

    // Control-flow context, taken from the recompiler's own region detectors so "uniform" means
    // what the compute emitter's `top_level_pc` means.
    const std::vector<DivLoop> loops = detect_divergent_loops(ins, safe);
    bool rejected = false;
    const std::vector<ForwardIf> ifs = detect_forward_ifs(
        ins, /*allow_vcc*/ false, code, dwords, &safe, loops.empty() ? nullptr : &loops, &rejected,
        /*compute_wave_branches*/ true);
    facts.control_flow_unmodelled = rejected || (any_backward_branch && loops.empty());
    for (auto& op : facts.ops) {
        if (facts.control_flow_unmodelled) {
            op.context = ComputeWaveContext::UnprovenRegion;
            continue;
        }
        bool in_loop = false;
        for (const auto& loop : loops)
            if (op.pc >= loop.header_pc && op.pc <= loop.backedge_pc) in_loop = true;
        // A waterfall is a loop with a per-wave trip count (one iteration per distinct lane value).
        for (const auto& br : ins)
            if (waterfalls.contains(br.pc) && branch_target(br) <= op.pc && op.pc <= br.pc)
                in_loop = true;
        if (in_loop) {
            op.context = ComputeWaveContext::Loop;
            continue;
        }
        ComputeWaveContext context = ComputeWaveContext::TopLevel;
        for (const auto& region : ifs) {
            const uint32_t end = region.has_else ? region.merge_pc : region.target_pc;
            if (!(region.branch_pc < op.pc && op.pc < end)) continue;
            if (!region.uniform_workgroup) {
                context = ComputeWaveContext::UnprovenRegion;
                break;
            }
            context = ComputeWaveContext::UniformRegion;
        }
        op.context = context;
    }
    return facts;
}

const char* compute_exchange_launch_refusal(const ComputeWaveHost& host) {
    const uint32_t width = host.host_subgroup_min ? host.host_subgroup_min : host.host_subgroup_max;
    const uint32_t local = host.local_x * host.local_y * host.local_z;
    if (!width) return "host-subgroup-unknown";
    if (host.guest_wave % width != 0) return "subgroup-does-not-divide-guest-wave";
    if (local == 0 || local % host.guest_wave != 0) return "workgroup-not-guest-wave-multiple";
    if (host.max_workgroup_invocations && local > host.max_workgroup_invocations)
        return "workgroup-exceeds-device-limit";
    if (host.partial_workgroup) return "partial-workgroup-barrier";
    if (!host.max_shared_bytes) return "shared-memory-limit-unknown";
    if (static_cast<uint64_t>(host.guest_lds_bytes) +
            compute_exchange_scratch_bytes(local, host.guest_wave) >
        host.max_shared_bytes)
        return "shared-memory-budget";
    return nullptr;
}

ComputeWaveRouteDecision select_compute_wave_route(const ComputeWaveOpFacts& facts,
                                                   const ComputeWaveHost& host) {
    ComputeWaveRouteDecision d;
    d.cross_lane_ops = facts.total();
    const auto done = [&](ComputeWaveRoute route, const char* reason) {
        d.route = route;
        d.reason = reason;
        return d;
    };
    if (host.native_contract ||
        (host.host_subgroup_min && host.host_subgroup_min >= host.guest_wave))
        return done(ComputeWaveRoute::Native, "host-subgroup-covers-guest-wave");
    if (!facts.analyzed) return done(ComputeWaveRoute::Refused, "unanalyzed");
    if (facts.ops.empty()) return done(ComputeWaveRoute::WidthIndependent, "no-cross-lane-op");
    // Without a required size Vulkan may choose any width in the device range, so the narrowest
    // width is the one a lowering must be correct for.
    const uint32_t width = host.host_subgroup_min ? host.host_subgroup_min : host.host_subgroup_max;
    if (!width) return done(ComputeWaveRoute::Refused, "host-subgroup-unknown");

    bool needs_exchange = false;
    bool in_loop = false, in_region = false;
    uint32_t loop_pc = UINT32_MAX, region_pc = UINT32_MAX;
    ComputeCrossLaneKind loop_kind = ComputeCrossLaneKind::Count, region_kind = loop_kind;
    const uint32_t local = host.local_x * host.local_y * host.local_z;
    const bool single_wave = local != 0 && local <= host.guest_wave;
    for (const auto& op : facts.ops) {
        const bool shuffle_form_only = !compute_exchange_carries(op.kind);
        if (shuffle_form_only) {
            // No exchange form: only a subgroup that holds the whole shuffle domain is exact.
            if (op.native_lanes && op.native_lanes <= width && host.guest_wave % width == 0)
                continue;
            d.blocker_pc = op.pc;
            d.blocker_kind = op.kind;
            return done(ComputeWaveRoute::Refused, "exchange-lowering-unavailable");
        }
        needs_exchange = true;
        if (op.context == ComputeWaveContext::Loop && !in_loop && !host.exchange_dispatcher) {
            in_loop = true;
            loop_pc = op.pc;
            loop_kind = op.kind;
        } else if (op.context == ComputeWaveContext::UnprovenRegion && !single_wave && !in_region &&
                   !host.exchange_dispatcher) {
            in_region = true;
            region_pc = op.pc;
            region_kind = op.kind;
        }
    }
    // Nothing can be proved about a program whose control flow the region detectors could not
    // model, whatever the workgroup size: the operation may sit in a loop we cannot see.
    if (needs_exchange && facts.control_flow_unmodelled)
        return done(ComputeWaveRoute::Refused, "control-flow-unmodelled");
    if (in_loop) {
        d.blocker_pc = loop_pc;
        d.blocker_kind = loop_kind;
        return done(ComputeWaveRoute::NeedsNLanes, "cross-lane-in-loop");
    }
    if (in_region) {
        d.blocker_pc = region_pc;
        d.blocker_kind = region_kind;
        return done(ComputeWaveRoute::NeedsNLanes, "cross-lane-in-nonuniform-region");
    }
    if (!needs_exchange) return done(ComputeWaveRoute::Native, "cross-lane-within-host-subgroup");

    // Launch and host constraints of the exchange itself.
    d.exchange_scratch_bytes = compute_exchange_scratch_bytes(local, host.guest_wave);
    if (const char* why = compute_exchange_launch_refusal(host))
        return done(ComputeWaveRoute::Refused, why);
    return done(ComputeWaveRoute::WorkgroupExchange, "cross-lane-in-uniform-flow");
}

}   // namespace prosper::gpu
