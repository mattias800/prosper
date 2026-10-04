#pragma once
#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "gpu/state/fragment_entry_facts.hpp"
#include "host/memory/guest_direct_allocation.hpp"

namespace prosper::gpu {
struct GpuState;
struct OriginalGraphicsStageEffects;
struct DecodedShader;
struct AgcShaderHeader;
class RegisteredNativeGraphicsAnalysis;

// DISTINCT from the nested RAW permission: a queued graphics span need not be submitted or
// waited here. The issuer proves complete original no-writer effects and captures every current
// and prior output allocation before permitting the checked scalar copy. This does not exclude
// arbitrary guest CPU writes; the submitted-input stability contract still applies.
class OrderedScalarBankReadPoint {
    friend struct OrderedGraphicsReadPointIssuer;
    struct Source {
        uint64_t address = 0;
        std::shared_ptr<const std::vector<uint32_t>> words;
        std::shared_ptr<const OriginalGraphicsStageEffects> effects;
        std::shared_ptr<const FragmentPacketVgprRequirements> packet_requirements;
        std::shared_ptr<const DecodedShader> decoded;
        std::shared_ptr<const RegisteredNativeGraphicsAnalysis> native_analysis;
        const AgcShaderHeader* registered_header = nullptr;
        bool fragment = false;
    };
    struct PendingDraw {
        std::shared_ptr<const PendingDraw> previous;
        std::vector<Source> sources;
        uint64_t command_order = 0, draw_index = 0;
    };
    const uint64_t identity_, submit_, order_, failure_generation_;
    const GpuState* const draw_state_;
    const FragmentEntryFacts entry_;
    const std::vector<Source> sources_;
    const std::shared_ptr<const std::vector<prosper::GuestDirectAllocation>> outputs_;
    const std::shared_ptr<const PendingDraw> prior_effects_;
    const std::shared_ptr<OrderedGraphicsReadPoint::Epoch> epoch_;
    const uint64_t sequence_, execution_version_;
    OrderedScalarBankReadPoint(
        uint64_t identity, uint64_t submit, uint64_t order, uint64_t failures,
        const GpuState& state, FragmentEntryFacts entry, std::vector<Source> sources,
        std::shared_ptr<const std::vector<prosper::GuestDirectAllocation>> outputs,
        std::shared_ptr<const PendingDraw> prior_effects,
        std::shared_ptr<OrderedGraphicsReadPoint::Epoch> epoch, uint64_t sequence,
        uint64_t execution_version)
        : identity_(identity), submit_(submit), order_(order), failure_generation_(failures),
          draw_state_(&state), entry_(std::move(entry)), sources_(std::move(sources)),
          outputs_(std::move(outputs)), prior_effects_(std::move(prior_effects)),
          epoch_(std::move(epoch)), sequence_(sequence), execution_version_(execution_version) {}

public:
    uint64_t identity() const { return identity_; }
    uint64_t source_submit() const { return submit_; }
    const FragmentEntryFacts& entry() const { return entry_; }
    const auto& output_allocations() const { return *outputs_; }
    uint64_t fragment_address() const;
    std::shared_ptr<const std::vector<uint32_t>> original_source(uint64_t address) const;
    const AgcShaderHeader* registered_header(uint64_t address) const;
    std::shared_ptr<const std::vector<uint32_t>> packet_source(uint64_t address) const;
    std::shared_ptr<const FragmentPacketVgprRequirements>
    packet_requirements(uint64_t address) const;
    std::shared_ptr<const DecodedShader> decoded_source(uint64_t address) const;
    std::shared_ptr<const RegisteredNativeGraphicsAnalysis>
    native_analysis(uint64_t address) const;
    bool belongs_to_draw(const GpuState&, uint64_t submit, uint64_t order,
                         uint64_t fragment_address) const;
    bool valid_for_packet(uint64_t submit, uint64_t order, uint64_t fragment_address,
                          const std::shared_ptr<const std::vector<uint32_t>>&,
                          const std::shared_ptr<const FragmentPacketVgprRequirements>&) const;
};
}   // namespace prosper::gpu
