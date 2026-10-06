#pragma once

#include "gpu/resources/fold_reader.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include <map>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct SrtUse;
struct FragmentPacketVgprRequirements;
bool raw_snapshot_storage_extent(const ShaderResource&, uint64_t& bytes);
// Immutable facts of the full original program, cached alongside its proven x2 chains.
// Dispatch admission below validates real descriptors, allocations and source authority.
struct RawSnapshotWritePlan {
    std::vector<uint32_t> descriptor_pcs;
    std::vector<uint32_t> storage_write_pcs;
    uint32_t required_user_words = 0;
    bool complete = true;
};
RawSnapshotWritePlan raw_snapshot_write_plan(const std::vector<Rdna2Inst>&,
                                             const std::vector<RawNestedWideChain>& = {});

// Executor-issued stale-permission authority, not resource backing or shader admission. Only the
// ordered executor can construct one; consumers supply their expected operation/source association.
// Its version checks do not serialize byte writers against copying. Existing submitted-input
// stability remains required, including normal HLE submit serialization.
class OrderedGraphicsReadPoint {
    friend struct OrderedGraphicsReadPointIssuer;
    friend class OrderedScalarBankReadPoint;
    struct Epoch {
        std::atomic<uint64_t> sequence{0};
        std::atomic<bool> active{true};
    };
    struct Source {
        uint64_t address;
        std::shared_ptr<const std::vector<uint32_t>> words;
        std::shared_ptr<const std::vector<RawNestedWideChain>> chains;
        std::shared_ptr<const FragmentPacketVgprRequirements> packet_requirements;
        bool fragment = false;
    };
    const uint64_t identity_, submit_, order_, failure_generation_;
    const std::vector<Source> sources_;
    const std::shared_ptr<Epoch> epoch_;
    const uint64_t sequence_, execution_version_;
    OrderedGraphicsReadPoint(uint64_t identity, uint64_t submit, uint64_t order, uint64_t failures,
                             std::vector<Source> sources, std::shared_ptr<Epoch> epoch,
                             uint64_t sequence, uint64_t execution_version)
        : identity_(identity), submit_(submit), order_(order), failure_generation_(failures),
          sources_(std::move(sources)), epoch_(std::move(epoch)), sequence_(sequence),
          execution_version_(execution_version) {}

public:
    uint64_t identity() const { return identity_; }
    std::shared_ptr<const std::vector<uint32_t>> source(uint64_t address) const;
    std::shared_ptr<const std::vector<uint32_t>> packet_source(uint64_t address) const;
    std::shared_ptr<const FragmentPacketVgprRequirements>
    packet_requirements(uint64_t address) const;
    bool owns_chains(uint64_t address, const std::vector<RawNestedWideChain>& chains) const;
    bool valid_for(uint64_t submit, uint64_t order, uint64_t source_address,
                   const std::shared_ptr<const std::vector<uint32_t>>& source) const;
    bool valid_for_packet(uint64_t submit, uint64_t order, uint64_t source_address,
                          const std::shared_ptr<const std::vector<uint32_t>>& source,
                          const std::shared_ptr<const FragmentPacketVgprRequirements>&) const;
};

// Supplied only at an ordered draw or compute boundary, after successful synchronous producers and
// retirement of deferred work. Neither this context nor a read point excludes concurrent byte
// writers; as with other draw inputs the guest must keep submitted backing alive and stable.
struct GraphicsRawSnapshotContext {
    bool producers_complete = false;
    // Current draw attachment (base, complete physical extent) pairs. Zero extent is unproved.
    // Even a shader with no explicit stores writes its framebuffer through exports; a same-
    // allocation scalar input cannot be a wave-invariant snapshot during that write.
    std::vector<std::pair<uint64_t, uint64_t>> output_allocations;
    std::shared_ptr<const OrderedGraphicsReadPoint> ordered_read_point;
    uint64_t source_submit = 0;
    bool requires_ordered_read_point = false;
};

// The live renderer must prove that raw guest bytes are current, including physical aliases of
// retained images. Missing provider, retained/unknown backing, or pending work is a refusal.
using GraphicsRawSourceAuthorityFn = std::function<bool(const prosper::GuestMappingLease&,
                                                       uint64_t, uint32_t)>;
void set_graphics_raw_source_authority(GraphicsRawSourceAuthorityFn fn);
bool graphics_raw_source_is_guest_current(const prosper::GuestMappingLease& lease,
                                          uint64_t address, uint32_t bytes);
// The bank copies demanded intervals but excludes retained writers over the source's complete
// original physical allocation, including unmapped/unused tails. This is not a byte-read extent.
using GraphicsRawAllocationAuthorityFn =
    std::function<bool(const prosper::GuestMappingLease&, const prosper::GuestDirectAllocation&)>;
void set_graphics_raw_allocation_authority(GraphicsRawAllocationAuthorityFn fn);
bool graphics_raw_allocation_is_guest_current(const prosper::GuestMappingLease&,
                                              const prosper::GuestDirectAllocation&);
struct GraphicsProducerStatus {
    bool known = false;
    bool pending = true;
    uint64_t failures = 0;
};
using GraphicsProducerStatusFn = std::function<GraphicsProducerStatus()>;
void set_graphics_producer_status_query(GraphicsProducerStatusFn fn);
GraphicsProducerStatus graphics_producer_status();

// Observe the complete certified runtime domain BEFORE any compiler/stage consumes source bytes.
// This produces packet inputs, not a live stage membership or output-commit certificate. A caller
// without a complete ordered producer context or authenticated direct allocation gets no owners.
bool observe_graphics_raw_wave_windows(
    const std::vector<uint32_t>& original,
    const std::vector<std::pair<uint32_t, uint32_t>>& entry_scalars,
    const GraphicsRawSnapshotContext* context, std::vector<PacketRawWaveWindow>& windows,
    std::string& refusal);
// Captured/owned bytes are inputs, not a proof token. This independently rederives every exact
// PC, original no-writer certificate, entry base and complete selected byte domain.
bool validate_graphics_raw_wave_windows(
    const std::vector<uint32_t>& original,
    const std::vector<std::pair<uint32_t, uint32_t>>& entry_scalars,
    const std::vector<PacketRawWaveWindow>& windows, std::string& refusal);

// Explicit MED logical assembly policy. It owns architectural invocation identities and the
// selected inputs; it makes no claim about PS5 physical scheduler occupancy/quad interleave.
// Full64 currently means every slot is an actual initialized invocation. Partial final groups
// are refused here rather than filled with invented inputs or called a complete Wave64.
enum class GraphicsWaveAssembly : uint8_t { VertexDrawOrder, FragmentPrimitiveQuadOrder };
struct GraphicsWaveInvocation {
    uint32_t vertex_index = 0, instance_index = 0, primitive_index = 0;
    std::array<uint32_t, 4> fragcoord{};
    bool helper = false;
};
struct GraphicsWaveStagePlan {
    GraphicsWaveAssembly assembly = GraphicsWaveAssembly::VertexDrawOrder;
    std::vector<FragmentInvocationPacket> packets;
    std::vector<std::array<GraphicsWaveInvocation, kFragmentPacketLanes>> invocations;
};
// Completed private scratch records, validated against the exact chosen ISA/input plan before
// any guest attachment is touched. Failure never publishes a partial earlier wave transaction.
struct GraphicsWaveOutputTransaction {
    std::vector<std::vector<uint32_t>> records;
};
// Immutable deferred live-stage input. A missing native module is permitted only with this
// typed plan; every byte was admitted at the ordered producer boundary, not at backend lookup.
struct GraphicsOwnedWaveDraw {
    bool vertex_pending = false, fragment_pending = false;
    GraphicsWaveStagePlan vertex;
    std::shared_ptr<const std::vector<uint32_t>> fragment_code;
    std::vector<std::pair<uint32_t, uint32_t>> fragment_scalars;
    std::vector<PacketRawWaveWindow> fragment_windows;
    PixelInputMapping pixel_inputs{};
    bool has_pixel_inputs = false;
    FragmentFloatMode fragment_float_mode{};
    FragmentFloatFlags fragment_float_flags{};
    std::shared_ptr<const struct RasterQuadInputs> fragment_raster_inputs;
};

// Whether the owner a successful preparation returned prepares exactly the stages the caller
// routed to the owned-wave path. No owner prepares neither. Realization refuses a draw for which
// this is false (#4555 review): the two decide separately, and a stage on which they disagree
// would otherwise have neither a native module nor an owner, or both.
inline bool owned_wave_owner_matches(const GraphicsOwnedWaveDraw* owner, bool vertex,
                                     bool fragment) {
    return (owner && owner->vertex_pending) == vertex &&
           (owner && owner->fragment_pending) == fragment;
}
bool validate_graphics_wave_outputs(const GraphicsWaveStagePlan& plan,
                                    const std::vector<std::vector<uint32_t>>& completed_records,
                                    GraphicsWaveOutputTransaction& transaction,
                                    std::string& refusal);
struct GraphicsVertexExportCommit {
    std::vector<uint32_t> shader, words;
    uint32_t vertices_per_instance = 0, instances = 0, record_words = 0;
};
bool prepare_owned_vertex_export_commit(const GraphicsWaveStagePlan& plan,
                                        const GraphicsWaveOutputTransaction& transaction,
                                        const PixelInputMapping* pixel_inputs,
                                        GraphicsVertexExportCommit& commit, std::string& refusal);
struct GraphicsFragmentExportCommit {
    std::vector<uint32_t> shader, words;
    uint32_t records = 0;
};
bool prepare_owned_fragment_export_commit(const GraphicsWaveStagePlan& plan,
                                          const GraphicsWaveOutputTransaction& transaction,
                                          GraphicsFragmentExportCommit& commit,
                                          std::string& refusal);
struct RasterQuadInputs;
struct RasterQuadResult;
bool prepare_owned_vertex_waves(const std::vector<uint32_t>& original,
                                const std::vector<std::pair<uint32_t, uint32_t>>& entry_scalars,
                                const std::vector<uint32_t>& owned_indices,
                                uint32_t nonindexed_count, int32_t vertex_offset,
                                uint32_t instance_count, FloatTransportConfig profile,
                                const GraphicsRawSnapshotContext* context,
                                GraphicsWaveStagePlan& plan, std::string& refusal);
bool prepare_owned_fragment_waves(const RasterQuadInputs& inputs, const RasterQuadResult& raster,
                                  const std::vector<std::pair<uint32_t, uint32_t>>& entry_scalars,
                                  const std::vector<PacketRawWaveWindow>& observed_windows,
                                  FragmentFloatMode float_mode, FragmentFloatFlags float_flags,
                                  GraphicsWaveStagePlan& plan, std::string& refusal);

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
    std::unique_ptr<prosper::GuestMappingLease> owned_lease_;
    const prosper::GuestMappingLease* lease_ = nullptr;
    bool allowed_ = false;
    std::shared_ptr<const OrderedGraphicsReadPoint> read_point_;
    std::shared_ptr<const std::vector<uint32_t>> source_;
    uint64_t source_address_ = 0, source_submit_ = 0, command_order_ = 0;
    bool checked_order_ = false;
    bool ordered_source_current() const;

public:
    GraphicsNestedWideReader(std::vector<RawNestedWideChain> chains,
                             const GraphicsRawSnapshotContext* context,
                             const prosper::GuestMappingLease* borrowed_lease = nullptr);
    // Live graphics uses this checked overload. The legacy overload remains for compute and
    // existing explicit offline contexts; it cannot manufacture an executor-issued ticket.
    GraphicsNestedWideReader(std::vector<RawNestedWideChain> chains,
                             const GraphicsRawSnapshotContext* context, uint64_t source_address,
                             std::shared_ptr<const std::vector<uint32_t>> source,
                             uint64_t source_submit, uint64_t command_order,
                             const prosper::GuestMappingLease* borrowed_lease = nullptr);
    bool owns_raw_wide(uint32_t pc) const override { return widths_.contains(pc); }
    bool probe(FoldProbe kind, uint32_t pc, uint64_t address, uint32_t bytes) override;
    uint32_t word(uint32_t pc, uint64_t address) override;
    void prefix(uint32_t pc, uint64_t address, void* destination, uint32_t bytes) override;
    // No publication until every parent/child equation is validated. Adds code-derived capture
    // obligations even on refusal; these vectors are not serialized as admission authority.
    bool publish(ShaderResourceTable& table) const;
    // Compute validates every original guest writer and descriptor source before publication.
    bool publish_compute_x2(ShaderResourceTable&, const RawSnapshotWritePlan&,
                            const std::vector<SrtUse>&) const;
};

} // namespace prosper::gpu
