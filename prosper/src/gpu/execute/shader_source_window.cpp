#include "gpu/execute/shader_source_window.hpp"

#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "host/memory/guest_memory_copy.hpp"

#include <algorithm>
#include <array>

extern "C" const void* prosper_agc_shader_header_for_code(uint64_t code_addr);

namespace prosper::gpu {
bool guest_readable(uint64_t address, uint32_t bytes);

size_t shader_source_dwords(uint64_t address, size_t maximum_dwords,
                            const AgcShaderHeader* registered_header) {
    if (address < 0x1000u || (address & 3u) || !maximum_dwords ||
        maximum_dwords > UINT32_MAX / sizeof(uint32_t) ||
        address > UINT64_MAX - maximum_dwords * sizeof(uint32_t))
        return 0;
    if (registered_header) {
        AgcShaderHeader header{};
        if (!host::guest_read_exact(uint64_t(uintptr_t(registered_header)), &header,
                                    sizeof(header)) ||
            uint64_t(uintptr_t(header.code)) != address || header.shader_size < sizeof(uint32_t) ||
            address > UINT64_MAX - header.shader_size ||
            !guest_readable(address, header.shader_size))
            return 0;
        return std::min(maximum_dwords, size_t(header.shader_size / sizeof(uint32_t)));
    }
    const auto readable = [&](size_t words) {
        return guest_readable(address, static_cast<uint32_t>(words * sizeof(uint32_t)));
    };
    if (readable(maximum_dwords)) return maximum_dwords;
    // Readability of [address,address+n) is monotonic in n. Search complete dwords, not END
    // instructions or a first-word heuristic; every later decoder receives this exact bound.
    size_t lower = 0, upper = maximum_dwords;
    while (upper - lower > 1) {
        const size_t middle = lower + (upper - lower) / 2;
        if (readable(middle))
            lower = middle;
        else
            upper = middle;
    }
    return lower;
}

size_t native_shader_source_dwords(uint64_t address, size_t maximum_dwords) {
    if (!address || !maximum_dwords) return 0;
    const auto* header =
        static_cast<const AgcShaderHeader*>(prosper_agc_shader_header_for_code(address));
    return shader_source_dwords(address, maximum_dwords, header);
}

namespace {
bool complete_instruction_prefix(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return false;
    for (size_t pc = 0; pc < dwords;) {
        // The existing decoder knows mandatory literals, DPP/SDWA and NSA widths. Give it
        // local storage large enough to report the encoded width without reading past guest VA.
        std::array<uint32_t, 5> scratch{};
        std::copy_n(code + pc, std::min(scratch.size(), dwords - pc), scratch.begin());
        const auto instruction = rdna2_decode_one(scratch.data(), scratch.size());
        if (!instruction.len_dwords || instruction.len_dwords > dwords - pc) return false;
        if (instruction.is_end || instruction.fmt == Rdna2Format::Unknown) return true;
        pc += instruction.len_dwords;
    }
    return true;
}
}   // namespace

size_t shader_source_code_span(const uint32_t* code, size_t dwords) {
    return complete_instruction_prefix(code, dwords) ? rdna2_recompile_code_span(code, dwords) : 0;
}

std::vector<uint32_t> shader_source_snapshot(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return {};
    std::vector<uint32_t> snapshot(code, code + dwords);
    if (!complete_instruction_prefix(snapshot.data(), snapshot.size())) snapshot.clear();
    return snapshot;
}
}   // namespace prosper::gpu
