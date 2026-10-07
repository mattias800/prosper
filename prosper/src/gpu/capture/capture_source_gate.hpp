// capture_source_gate.hpp -- recording a draw buffer the live renderer bound as its all-zero fallback.
//
// The live draw path (frontends/shared/live/submit_renderer/draw_resources.cpp) asks one question of
// every non-image draw resource that has no host-owned bytes: is anything mapped at this guest
// address? When the answer is no (buffer_source_gate.hpp), it does NOT read the declared range. It
// binds a zero-filled buffer sized from the shader's reflected requirement, and robust buffer access
// makes every access beyond that zero as well. The guest's declared length plays no part.
//
// UE4 titles bind exactly that shape: a whole-range V# (NUM_RECORDS 0xffffffff) over an address with
// nothing behind it. Kena (PPSA01802) binds one at 0xf00000000000, past the top of the 47-bit user
// address space, so no guest mapping can contain it (#3807). The capture used to plan that range as
// guest bytes to own, hit the 1 GiB per-resource ceiling, and abort the whole frame -- for a binding
// whose live content was zeros, with no 4 GiB of anything behind it.
//
// So the capture asks the live renderer's own gate, through a probe the renderer registers
// (set_gpu_capture_buffer_source_probe), and records a binding the gate refuses as what it is: a
// placeholder with no blob (`GpuCapturedResource::source_unavailable`, capture v74). Replay publishes
// the mark as `ShaderResource::replay_source_unavailable`, and the same gate then takes the same
// all-zero branch BECAUSE THE RECORD SAYS SO, not because the replay process happens to have nothing
// mapped at that address. Nothing is dropped and no content is fabricated: the record states that
// the live renderer read no guest bytes for this binding.
//
// What this deliberately does NOT cover:
//   * A MAPPED source over the per-resource ceiling. Its bytes are real; it still fails closed with
//     the ceiling error, which is the contract #2440 set and tests pin.
//   * Images. The image path has its own source handling and never consults this gate.
//   * Compute tables. The live compute backend does not use this gate; a compute resource keeps the
//     ordinary capture contract.
//   * Owned fragment-wave inputs (a producing table distinct from `prt`): their reads are not the
//     ordinary draw-resource path.
//   * Anything with an obligation to own bytes (owned/raw/nested snapshots, descriptor arrays,
//     capture-authority resources, internal GDS, packed-pointer shadows).
#pragma once

#include "gpu/capture/gpu_capture.hpp"

#include <cstdint>
#include <functional>
#include <set>

namespace prosper::gpu {

// Shape test only: could this draw-table resource be a gate-refused placeholder at all? The probe
// answers the mapping question; this answers "is it the kind of resource the live gate is asked
// about, with nothing else obliging the capture to own its bytes".
inline bool capture_source_gate_candidate(const ShaderResource& r) {
    return r.cls != ResourceClass::Texture && r.cls != ResourceClass::StorageImage &&
        !r.host_data && !r.host_data_size && !r.host_data_prefix_bytes &&
        !r.table_index_count && r.table_entries.empty() &&
        !r.raw_register_snapshot && !r.owned_raw_snapshot_bytes &&
        !r.owned_nested_snapshot_bytes && !r.nested_raw_snapshot_admitted &&
        !r.indirect_buffer_contract_tag && !r.indirect_pointer_relocation.binding_bytes &&
        !r.compression_enabled && !r.metadata_addr;
}

// The record a placeholder must be to be read or written: no backing of any kind, and a resource
// shape the live gate is asked about. Serialization and replay both refuse anything else, so a
// corrupt or hand-edited file cannot turn a real resource into zeros by setting one byte.
inline bool valid_source_unavailable_record(const GpuCapturedResource& c) {
    return c.source_unavailable && c.blob_index == 0xFFFFFFFFu && c.blob_offset == 0 &&
        c.metadata_blob_index == 0xFFFFFFFFu && c.metadata_size == 0 &&
        c.internal_bytes.empty() && c.table_entry_blobs.empty() &&
        capture_source_gate_candidate(c.resource);
}

}  // namespace prosper::gpu
