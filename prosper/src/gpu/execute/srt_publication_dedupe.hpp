// srt_publication_dedupe.hpp -- which descriptor-table uses build_stage_table publishes (#294, #273).
//
// A KEYED constant-buffer use is published once per key: its consumers resolve through the
// resource's srt_offset. Textures, key-less buffers and exact MTBUF uses are published once per
// CONSUMING INSTRUCTION, because they resolve by fetch_pc. A keyed buffer use whose key CLASHES with
// a resource already holding it is published key-less, so it too resolves only by fetch_pc -- and
// from then on that key must be deduplicated per instruction, or every consumer after the first is
// left with no entry and refused as unresolved-cbuf (Kena's title pixel shader: one V#, two scalar
// buffer loads, the key already held by another resource -- a metadata sharp or a descriptor loaded
// at the same immediate from another table; test_agc_shader pins the texture case). Compute
// publishes every buffer use per instruction for the same reason.
#pragma once

#include "gpu/execute/gpu_execute.hpp"

#include <cstdint>
#include <set>

namespace prosper::gpu {

class SrtPublicationDedupe {
public:
    // True the first time this use's identity is seen; false for a duplicate to skip.
    bool admit(const SrtUse& u) {
        const bool exact_mtbuf = u.kind == 1 && u.instruction_format != UINT32_MAX;
        const bool per_pc = u.kind == 0 || u.key == 0xFFFFFFFFu || exact_mtbuf ||
                            (u.kind == 1 && clashed_cbuf_keys_.count(u.key) != 0);
        // Distinct namespaces: pc identities must never collide with byte-offset keys.
        const uint64_t identity =
            per_pc ? (0x8000000000000000ull | (uint64_t(uint32_t(u.kind)) << 32) | u.use_pc)
                   : (uint64_t(uint32_t(u.kind)) << 32) | u.key;
        return seen_.insert(identity).second;
    }
    // Record that this admitted keyed use found its key already held by another resource, so it is
    // published key-less and every later use of that key must resolve by its own pc.
    void note_clash(const SrtUse& u) {
        if (u.kind == 1 && u.key != 0xFFFFFFFFu) clashed_cbuf_keys_.insert(u.key);
    }

private:
    std::set<uint64_t> seen_;
    std::set<uint32_t> clashed_cbuf_keys_;
};

}   // namespace prosper::gpu
