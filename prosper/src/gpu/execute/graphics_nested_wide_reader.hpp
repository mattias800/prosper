#pragma once

#include "gpu/resources/fold_reader.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include <map>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

namespace prosper::gpu {

// Supplied only at an ordered draw boundary, after successful synchronous producers and
// retirement of deferred work. This excludes producer races, not unsynchronized guest CPU
// stores; as with other draw inputs the guest must keep the submitted input alive and stable.
struct GraphicsRawSnapshotContext {
    bool producers_complete = false;
    // Current draw attachment (base, complete physical extent) pairs. Zero extent is unproved.
    // Even a shader with no explicit stores writes its framebuffer through exports; a same-
    // allocation scalar input cannot be a wave-invariant snapshot during that write.
    std::vector<std::pair<uint64_t, uint64_t>> output_allocations;
};

// The live renderer must prove that raw guest bytes are current, including physical aliases of
// retained images. Missing provider, retained/unknown backing, or pending work is a refusal.
using GraphicsRawSourceAuthorityFn = std::function<bool(const prosper::GuestMappingLease&,
                                                       uint64_t, uint32_t)>;
void set_graphics_raw_source_authority(GraphicsRawSourceAuthorityFn fn);
bool graphics_raw_source_is_guest_current(const prosper::GuestMappingLease& lease,
                                          uint64_t address, uint32_t bytes);
struct GraphicsProducerStatus {
    bool known = false;
    bool pending = true;
    uint64_t failures = 0;
};
using GraphicsProducerStatusFn = std::function<GraphicsProducerStatus()>;
void set_graphics_producer_status_query(GraphicsProducerStatusFn fn);
GraphicsProducerStatus graphics_producer_status();

// One reader/lease per stage realization. The first exact-PC probe copies the effective load;
// the scalar fold and emitted resource consume that same owner, never a second guest read.
class GraphicsNestedWideReader final : public FoldReader {
    struct Observation {
        uint64_t address = 0;
        std::shared_ptr<std::vector<uint8_t>> bytes;
    };
    std::vector<RawNestedWideChain> chains_;
    std::map<uint32_t, uint32_t> widths_;
    std::map<uint32_t, Observation> observations_;
    std::vector<std::pair<uint64_t, uint64_t>> output_allocations_;
    std::unique_ptr<prosper::GuestMappingLease> lease_;
    bool allowed_ = false;
public:
    GraphicsNestedWideReader(std::vector<RawNestedWideChain> chains,
                             const GraphicsRawSnapshotContext* context);
    bool owns_raw_wide(uint32_t pc) const override { return widths_.contains(pc); }
    bool probe(FoldProbe kind, uint32_t pc, uint64_t address, uint32_t bytes) override;
    uint32_t word(uint32_t pc, uint64_t address) override;
    void prefix(uint32_t pc, uint64_t address, void* destination, uint32_t bytes) override;
    // No publication until every parent/child equation is validated. Adds code-derived capture
    // obligations even on refusal; these vectors are not serialized as admission authority.
    bool publish(ShaderResourceTable& table) const;
};

} // namespace prosper::gpu
