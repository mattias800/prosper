#include "gpu/pm4/pending_memory_view.hpp"
#include "gpu/pm4/command_processor.hpp"
#include <algorithm>
#include <cstring>

namespace prosper::gpu {
PendingMemorySpan pending_memory_span(const Pm4Command& command) {
    using K = Pm4Command::Kind;
    switch (command.kind) {
        case K::ReleaseMem:
            if (command.rel_data_sel == 3 ||
                (command.rel_value_valid &&
                 (command.rel_data_sel == 1 || command.rel_data_sel == 2)))
                return {command.rel_addr, command.rel_data_sel == 1 ? 4u : 8u};
            break;
        case K::EventWrite: return {command.event_addr, 8};
        case K::WriteData:
            if (command.wd_valid && command.wd_data)
                return {command.wd_addr, uint64_t(command.wd_num) * 4};
            break;
        case K::DmaData:
            if (command.dd_valid && (command.dd_sels & 0xffu) != 1u)
                return {command.dd_dst, command.dd_bytes};
            break;
        default: break;
    }
    return {};
}

bool pending_memory_overlaps(PendingMemorySpan first, PendingMemorySpan second) {
    return first.address && first.bytes && second.address && second.bytes &&
           first.address <= UINT64_MAX - first.bytes &&
           second.address <= UINT64_MAX - second.bytes &&
           first.address < second.address + second.bytes &&
           second.address < first.address + first.bytes;
}

bool pending_memory_completion(const Pm4Command& command) {
    return command.kind == Pm4Command::Kind::ReleaseMem ||
           command.kind == Pm4Command::Kind::EventWrite;
}

PendingMemoryOverlay overlay_pending_memory(const Pm4Command& command,
                                            std::span<const uint8_t> captured_dma, uint64_t address,
                                            std::span<uint8_t> destination) {
    const auto written = pending_memory_span(command);
    if (!pending_memory_overlaps(written, {address, destination.size()}))
        return PendingMemoryOverlay::Disjoint;
    const uint64_t start = std::max(address, written.address);
    const size_t count =
        std::min(address + destination.size(), written.address + written.bytes) - start;
    const size_t source_offset = start - written.address, target_offset = start - address;
    const uint8_t* source = nullptr;
    using K = Pm4Command::Kind;
    if (command.kind == K::ReleaseMem && command.rel_data_sel != 3) {
        source = reinterpret_cast<const uint8_t*>(&command.rel_value);
    } else if (command.kind == K::WriteData) {
        source = reinterpret_cast<const uint8_t*>(command.wd_data);
    } else if (command.kind == K::DmaData) {
        if (captured_dma.size() >= written.bytes) {
            source = captured_dma.data();
        } else if (!(command.dd_sels & kDmaDataAddressSource) && command.dd_src <= UINT32_MAX &&
                   ((command.dd_sels >> 8u) & 0xffu) != 1u) {
            const uint32_t word = static_cast<uint32_t>(command.dd_src);
            const auto* pattern = reinterpret_cast<const uint8_t*>(&word);
            for (size_t n = 0; n < count; ++n)
                destination[target_offset + n] = pattern[(source_offset + n) & 3u];
            return PendingMemoryOverlay::Applied;
        }
    }
    if (!source) return PendingMemoryOverlay::Unresolved;
    std::memcpy(destination.data() + target_offset, source + source_offset, count);
    return PendingMemoryOverlay::Applied;
}
} // namespace prosper::gpu
