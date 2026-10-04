#include "gpu/execute/scalar_bank_read_point.hpp"
#include "gpu/execute/graphics_execution_activity.hpp"
#include "gpu/recompiler/fragment_packet_vgpr_requirements.hpp"

namespace prosper::gpu {
uint64_t OrderedScalarBankReadPoint::fragment_address() const {
    for (const auto& source : sources_)
        if (source.fragment) return source.address;
    return 0;
}
std::shared_ptr<const std::vector<uint32_t>>
OrderedScalarBankReadPoint::original_source(uint64_t address) const {
    for (const auto& source : sources_)
        if (source.address == address) return source.words;
    return {};
}
const AgcShaderHeader* OrderedScalarBankReadPoint::registered_header(uint64_t address) const {
    for (const auto& source : sources_)
        if (source.address == address) return source.registered_header;
    return nullptr;
}
std::shared_ptr<const AgcShaderHeader>
OrderedScalarBankReadPoint::header_snapshot(uint64_t address) const {
    for (const auto& source : sources_)
        if (source.address == address) return source.header_snapshot;
    return {};
}
std::shared_ptr<const std::vector<uint32_t>>
OrderedScalarBankReadPoint::packet_source(uint64_t address) const {
    for (const auto& source : sources_)
        if (source.fragment && source.address == address) return source.words;
    return {};
}
std::shared_ptr<const FragmentPacketVgprRequirements>
OrderedScalarBankReadPoint::packet_requirements(uint64_t address) const {
    for (const auto& source : sources_)
        if (source.fragment && source.address == address) return source.packet_requirements;
    return {};
}
std::shared_ptr<const DecodedShader>
OrderedScalarBankReadPoint::decoded_source(uint64_t address) const {
    for (const auto& source : sources_)
        if (source.address == address) return source.decoded;
    return {};
}
std::shared_ptr<const RegisteredNativeGraphicsAnalysis>
OrderedScalarBankReadPoint::native_analysis(uint64_t address) const {
    for (const auto& source : sources_)
        if (source.address == address) return source.native_analysis;
    return {};
}
bool OrderedScalarBankReadPoint::belongs_to_draw(const GpuState& state, uint64_t submit,
                                                 uint64_t order, uint64_t address) const {
    return draw_state_ == &state && valid_for_packet(submit, order, address, packet_source(address),
                                                     packet_requirements(address));
}
bool OrderedScalarBankReadPoint::valid_for_packet(
    uint64_t submit, uint64_t order, uint64_t address,
    const std::shared_ptr<const std::vector<uint32_t>>& words,
    const std::shared_ptr<const FragmentPacketVgprRequirements>& requirements) const {
    if (!identity_ || !submit || submit != submit_ || order != order_ || !words || !requirements)
        return false;
    const auto expected_words = packet_source(address);
    const auto expected_requirements = packet_requirements(address);
    if (!expected_words || !expected_requirements || expected_words.get() != words.get() ||
        expected_words.owner_before(words) || words.owner_before(expected_words) ||
        expected_requirements.get() != requirements.get() ||
        expected_requirements.owner_before(requirements) ||
        requirements.owner_before(expected_requirements) || words.owner_before(requirements) ||
        requirements.owner_before(words) || requirements->source_words != words.get() ||
        requirements->masks.source_words != words.get() ||
        requirements->scalar_reads.source_words != words.get())
        return false;
    const auto ordered = [&] {
        return epoch_ && epoch_->active.load(std::memory_order_acquire) &&
               epoch_->sequence.load(std::memory_order_acquire) == sequence_ &&
               GraphicsExecutionActivity::current_version() == execution_version_;
    };
    if (!ordered()) return false;
    const auto current = graphics_producer_status();
    return current.known && !current.pending && current.failures == failure_generation_ &&
           ordered();
}
}   // namespace prosper::gpu
