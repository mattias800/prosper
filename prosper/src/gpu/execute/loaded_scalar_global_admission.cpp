// Code facts do not make padded user registers or borrowed guest pointers authoritative.
#include "gpu/execute/loaded_scalar_global_admission.hpp"
#include <array>
#include <cstring>

namespace prosper::gpu {
namespace {
bool canonical(const LoadedScalarGlobalRead& read) {
    return read.parent_pc < read.global_pc && read.entry_pointer_sgpr <= 104u &&
        read.loaded_pointer_sgpr <= 104u && read.address_vgpr < 256u &&
        !(read.parent_offset & 3u) && int32_t(read.parent_offset) >= 0 &&
        !(read.window_offset & 3u) &&
        (read.window_bytes == 16u || read.window_bytes == 32u);
}

std::optional<uint64_t> parent_address(const LoadedScalarGlobalRead& read,
                                      const LoadedGlobalUserPrefix& prefix) {
    if (!canonical(read) || !prefix.declared_count || *prefix.declared_count > 32u ||
        prefix.words.size() > 32u || read.entry_pointer_sgpr < prefix.sgpr_base)
        return {};
    const uint32_t index = read.entry_pointer_sgpr - prefix.sgpr_base;
    if (index >= 31u || index + 2u > *prefix.declared_count ||
        index + 2u > prefix.words.size() ||
        !(prefix.physical_word_mask & (uint32_t(1) << index)) ||
        !(prefix.physical_word_mask & (uint32_t(1) << (index + 1u)))) return {};
    const uint64_t pointer = uint64_t(prefix.words[index]) |
                             (uint64_t(prefix.words[index + 1u]) << 32u);
    if (pointer > UINT64_MAX - read.parent_offset) return {};
    const uint64_t address = pointer + read.parent_offset;
    if (address <= 0x10000u || (address & 3u) || address > UINT64_MAX - 8u) return {};
    return address;
}
} // namespace

std::vector<RawNestedWideChain> loaded_scalar_global_chains(
        const std::vector<LoadedScalarGlobalRead>& reads) {
    std::vector<RawNestedWideChain> chains;
    chains.reserve(reads.size());
    for (const auto& read : reads) {
        if (!canonical(read)) return {};
        chains.push_back({read.parent_pc, read.global_pc, 8u, read.window_bytes,
                          read.parent_offset, read.window_offset});
    }
    return chains;
}

bool prepare_loaded_scalar_globals(FoldReader& reader,
        const std::vector<LoadedScalarGlobalRead>& reads, const LoadedGlobalUserPrefix& prefix) {
    // Validate every launch source before any request. No ordinary fold may replace a refusal.
    for (const auto& read : reads)
        if (!reader.owns_raw_wide(read.parent_pc) || !reader.owns_raw_wide(read.global_pc) ||
            !parent_address(read, prefix)) return false;
    for (const auto& read : reads) {
        const uint64_t address = *parent_address(read, prefix);
        if (!reader.probe(FoldProbe::Raw, read.parent_pc, address, 8u)) return false;
        std::array<uint8_t, 8> bytes{};
        reader.prefix(read.parent_pc, address, bytes.data(), bytes.size());
        uint64_t pointer = 0;
        std::memcpy(&pointer, bytes.data(), sizeof(pointer));
        if (pointer > UINT64_MAX - read.window_offset) return false;
        const uint64_t target = pointer + read.window_offset;
        if (target <= 0x10000u || (target & 3u) || target > UINT64_MAX - read.window_bytes ||
            !reader.probe(FoldProbe::Raw, read.global_pc, target, read.window_bytes)) return false;
    }
    return true;
}
} // namespace prosper::gpu
