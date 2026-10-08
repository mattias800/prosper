#pragma once
// Compute Wave64 route selection (ADR 0028, migration steps 3-4): for a guest compute program on a
// host whose compute subgroup is narrower than the guest wave, classify every cross-lane operation
// and decide which route the program may take. Pure analysis over decoded RDNA2 instructions plus
// the launch/host facts -- no Vulkan, no guest memory, no emission.
//
// The route names the ADR's routes by what they cost, not by how they are implemented:
//   Native            the host subgroup covers the guest wave; nothing below applies.
//   WidthIndependent  no cross-lane operation at all (ADR route 2): one lane per invocation is exact.
//   WorkgroupExchange every cross-lane operation sits in control flow that every invocation of the
//                     workgroup reaches together, so an exchange across the 64/W host subgroups of
//                     one workgroup through workgroup memory and barriers is exact (ADR route 3).
//   NeedsNLanes       a cross-lane operation sits where a workgroup barrier would be divergent
//                     (a region whose branch is not proven workgroup-uniform, or a loop). Exact only
//                     with N = 64/W guest lanes per invocation (ADR route 4, not implemented).
//   Refused           something the exchange route cannot carry, or a launch/host constraint fails.
//                     Always carries a named reason; the caller must refuse visibly.
//
// The decision is a PROOF OVER THE GUEST PROGRAM (the 64-lane wave is the semantic target), never
// a statement about what the emitter did. It reuses the recompiler's own region detectors
// (detect_forward_ifs / detect_divergent_loops and their workgroup-uniformity proofs), so "uniform"
// here means exactly what `top_level_pc` means in the compute emitter.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu/recompiler/rdna2_decode.hpp"

namespace prosper::gpu {

enum class ComputeCrossLaneKind : uint8_t {
    ReadLane,   // v_readlane_b32: any lane of the guest wave
    ReadFirstLane,   // v_readfirstlane_b32: lowest ACTIVE lane of the guest wave
    Mbcnt,   // v_mbcnt_{lo,hi}_u32_b32 over a lane mask (not the all-ones lane-index form)
    WaveVote,   // scalar branch on the whole-wave EXEC/VCC mask (any / all vote)
    Ballot,   // s_bcnt1/s_ff1 b64 over EXEC or VCC: a popcount/first-set of a ballot
    MaskConsumer,   // s_bcnt1/s_ff1 b64 over a saved mask whose origin is not proven
    Dpp,   // DPP quad/row/bank permutation
    PermLane,   // v_permlane16 / v_permlanex16
    DsSwizzle,   // ds_swizzle_b32
    DsBpermute,   // ds_bpermute_b32
    DsAppend,   // ds_append / ds_consume: a wave-collective counter update
    MaskScc,         // SCC produced from a whole-wave mask (s_cmp_*_u64 / s_and,or,xor,andn2_b64 / saveexec / wqm / not)
    MaskOther,       // any other scalar op that reads or writes EXEC/VCC as a lane mask
    WriteLane,       // v_writelane_b32 with a non-inline lane selector
    LdsWaveSync,     // an LDS read after an LDS write with no s_barrier between: wave-synchronous LDS
    DsPermute,       // ds_permute_b32
    Count
};
inline constexpr size_t kComputeCrossLaneKindCount =
    static_cast<size_t>(ComputeCrossLaneKind::Count);
const char* compute_cross_lane_kind_name(ComputeCrossLaneKind kind);

// Names follow the single route vocabulary of #4754 (kWave64RouteNames): native,
// proven-width-independent, workgroup-exchange, n-lanes, refused.
enum class ComputeWaveRoute : uint8_t {
    Native,
    WidthIndependent,
    WorkgroupExchange,
    NeedsNLanes,
    Refused
};
const char* compute_wave_route_name(ComputeWaveRoute route);

// Where in the program's control flow an operation sits.
enum class ComputeWaveContext : uint8_t {
    TopLevel,   // no enclosing region
    UniformRegion,   // only regions proven to decide identically for every wave of the workgroup
    UnprovenRegion,   // inside a forward region whose branch is NOT proven workgroup-uniform
    Loop,   // inside a loop (or the program has a backward branch no loop model covers)
};

struct ComputeCrossLaneOp {
    uint32_t pc = 0;
    ComputeCrossLaneKind kind = ComputeCrossLaneKind::ReadLane;
    ComputeWaveContext context = ComputeWaveContext::TopLevel;
    // No instruction earlier in the stream can change EXEC. False only means "not proven".
    bool exec_full = true;
    // Lanes one host subgroup must hold for a native shuffle form of this operation (quad 4, row
    // 16, permlanex16 32, wave 64), or 0 when it has no native form. An operation whose
    // native_lanes fits the host subgroup needs no exchange for the shuffle itself.
    uint32_t native_lanes = 0;
};

// Facts about the program that do not depend on the host. A pure function of the program bytes, so
// it can be memoized with the program (see compute_program_facts).
struct ComputeWaveOpFacts {
    bool analyzed = false;
    // The structured region/loop detectors could not model the program's control flow. Every
    // cross-lane operation is then reported as being in an unproven region.
    bool control_flow_unmodelled = false;
    std::vector<ComputeCrossLaneOp> ops;
    // Convenience tallies, derived from `ops`.
    uint16_t count[kComputeCrossLaneKindCount] = {};
    uint16_t total() const {
        uint32_t n = 0;
        for (uint16_t c : count) n += c;
        return static_cast<uint16_t>(n > 0xffffu ? 0xffffu : n);
    }
};

// Decode-level analysis. `code`/`dwords` are the exact bytes `ins` was walked from (the region
// detectors re-read branch targets from them).
ComputeWaveOpFacts analyze_compute_wave_ops(const std::vector<Rdna2Inst>& ins, const uint32_t* code,
                                            size_t dwords);

// What the host offers and what the launch demands.
struct ComputeWaveHost {
    uint32_t guest_wave = 64;   // 32 or 64
    uint32_t host_subgroup_min = 0;   // device compute subgroup range; 0 = unknown
    uint32_t host_subgroup_max = 0;
    uint32_t local_x = 64, local_y = 1, local_z = 1;
    uint32_t guest_lds_bytes = 0;
    uint32_t max_shared_bytes =
        0;   // VkPhysicalDeviceLimits::maxComputeSharedMemorySize; 0 = unknown
    // True when the module is dispatched with an exact required subgroup size equal to the guest
    // wave (the native contract), which makes every lane operation exact on its own.
    bool native_contract = false;
    // The dispatch ends in a partial workgroup (exact thread extent not a multiple of the local
    // size). Vulkan requires every invocation of a workgroup to reach an OpControlBarrier, so an
    // entry guard that retires the padded invocations makes an exchange barrier divergent.
    bool partial_workgroup = false;
    uint32_t max_workgroup_invocations = 0;   // 0 = unknown, not checked
};

struct ComputeWaveRouteDecision {
    ComputeWaveRoute route = ComputeWaveRoute::Refused;
    // A stable, hyphenated name: the reason a route was chosen or refused. Never null.
    const char* reason = "unanalyzed";
    uint32_t blocker_pc = UINT32_MAX;   // first operation that decided a non-exchange route
    ComputeCrossLaneKind blocker_kind = ComputeCrossLaneKind::Count;
    uint16_t cross_lane_ops = 0;
    uint32_t exchange_scratch_bytes = 0;   // workgroup memory the exchange needs, when admitted
};

// Workgroup memory the exchange lowerings may declare for one workgroup, as an upper bound over both
// families (see the definition). Kept next to the route so the budget check and the declarations
// cannot silently drift; a test pins it against the Workgroup arrays of an emitted module.
uint32_t compute_exchange_scratch_bytes(uint32_t local_invocations, uint32_t guest_wave);

// Whether the exchange lowering (compute_wave_exchange in the emitter) carries `kind`. The route
// refuses programs with an operation it cannot carry rather than hand them to a lowering that would
// approximate. See the ADR's "unsupported op is a fatal gap" rule.
bool compute_exchange_carries(ComputeCrossLaneKind kind);

// The launch/host constraints of the exchange itself, independent of the program: nullptr when the
// exchange may run, else the refusal reason. select_compute_wave_route applies it after the program
// facts; the backend applies it alone to a module that is already compiled through the exchange.
const char* compute_exchange_launch_refusal(const ComputeWaveHost& host);

ComputeWaveRouteDecision select_compute_wave_route(const ComputeWaveOpFacts& facts,
                                                   const ComputeWaveHost& host);

}   // namespace prosper::gpu
