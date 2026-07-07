// command_processor.hpp — apply a decoded PM4 stream to a GPU state snapshot.
//
// This is the second CommandProcessor stage (after pm4_decode): walk the decoded ops and fold them
// into a `GpuState` — the register files (context/shader/user-config) plus index type and the list of
// draws. A later stage translates a GpuState + its draws into Vulkan pipelines/draw calls (M4).
//
// Register application matches Kyty's CommandProcessor (GraphicsRun.cpp `cp_op_indirect_cx_regs`,
// MIT): a Set{Cx,Sh,Uc}RegistersIndirect packet points at an array of `num_regs` {offset, value}
// pairs (a ShaderRegister[]); each writes `value` into the class's register file at `offset`. Because
// prosper maps the guest image 1:1, the packet's `regs_vaddr` is directly readable as a host pointer.
#pragma once
#include "pm4_decode.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace prosper::gpu {

// A {register offset, value} pair — the element type of a Set*RegistersIndirect array.
struct ShaderReg { uint32_t offset; uint32_t value; };

// Folded GPU state after replaying a command stream.
struct GpuState {
    using RegFile = std::unordered_map<uint32_t, uint32_t>;
    RegFile cx, sh, uc;                                  // register files by offset
    uint32_t index_type = 0;                             // last SetIndexType
    uint32_t flips = 0;                                  // incremented per in-stream Flip (frame boundary)

    // A draw records its vertex count AND a snapshot of the register state at the moment it was issued.
    // Real frames are built from MANY draws, each with its own bound shaders/pipeline/resources; folding
    // every write into one final state (and rendering only draws[0]) collapses that to a single primitive.
    // Snapshotting per draw lets the executor resolve+render each draw with its own state, in order, into
    // one accumulating framebuffer. `snapshot` is true for draws captured with a snapshot (game path);
    // legacy/synthetic draws leave it false and fall back to the folded GpuState.
    struct Draw {
        uint32_t index_count = 0;
        bool     snapshot = false;
        RegFile  cx, sh, uc;
        uint32_t index_type = 0;
    };
    std::vector<Draw> draws;                             // one per DrawIndexAuto, in submit order

    // Safety cap on an indirect register count (a malformed/huge count won't run away).
    static constexpr uint32_t kMaxRegsPerPacket = 4096;

    // Fold one decoded command into the state. Reads register arrays via their (1:1-mapped) vaddr.
    void apply(const Pm4Command& c);

    // Reconstruct a single-draw GpuState from draw `i`'s snapshot: its register files + that one draw.
    // extract_render_state()/build_stage_table() then resolve exactly the state that draw was issued
    // under. Falls back to the current folded state for a draw captured without a snapshot.
    GpuState state_at_draw(size_t i) const {
        GpuState s;
        const Draw& d = draws[i];
        if (d.snapshot) { s.cx = d.cx; s.sh = d.sh; s.uc = d.uc; s.index_type = d.index_type; }
        else            { s.cx = cx;   s.sh = sh;   s.uc = uc;   s.index_type = index_type; }
        s.draws.push_back(Draw{ d.index_count, false, {}, {}, {}, d.index_type });
        return s;
    }
};

// Decode `dwords` dwords at `buf` and apply every op to `st`. Returns the number of packets applied.
size_t run_command_buffer(const uint32_t* buf, size_t dwords, GpuState& st);

} // namespace prosper::gpu
