#pragma once
// Project-owned complete programs observe LOW/HIGH mask-slot composition through an EXEC
// restore and the ordinary buffer sink. Slot numbers do not determine a physical half's role.
#include "portable_mask_half.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace prosper::test::portable_mask_slots {
inline void spill(std::vector<uint32_t>& code, uint32_t source, uint32_t lane) {
    code.insert(code.end(), {0xd7610012u, source | ((128u + lane) << 9)});
}
inline void reload(std::vector<uint32_t>& code, uint32_t destination, uint32_t lane) {
    code.insert(code.end(), {0xd7600000u | destination, 0x112u | ((128u + lane) << 9)});
}
inline std::vector<uint32_t> program(bool low_full, bool high_full, bool reverse = false,
                                     uint32_t low_lane = 0, uint32_t high_lane = 1,
                                     bool copied = false, bool loop = false) {
    using portable_mask_half::mov;
    std::vector<uint32_t> code{0x7e060280u, mov(126, low_full ? 193 : 128, true)};
    spill(code, 126, low_lane);
    code.push_back(mov(126, high_full ? 193 : 128, true));
    spill(code, 127, high_lane);
    if (reverse) {
        reload(code, 21, high_lane);
        reload(code, 20, low_lane);
    } else {
        reload(code, 20, low_lane);
        reload(code, 21, high_lane);
    }
    if (copied) code.insert(code.end(), {mov(24, 20), mov(25, 21)});
    if (loop) portable_mask_half::counted(code);
    portable_mask_half::sink(code, copied ? 24 : 20);
    return code;
}
inline std::vector<uint32_t> missing_half(bool high) {
    using portable_mask_half::mov;
    std::vector<uint32_t> code{0x7e060280u, mov(126, 193, true)};
    spill(code, high ? 126 : 127, high ? 0 : 1);
    reload(code, high ? 20 : 21, high ? 0 : 1);
    portable_mask_half::sink(code);
    return code;
}
inline std::vector<uint32_t> joined(bool seed_high, bool conflicting_role = false) {
    using portable_mask_half::mov;
    std::vector<uint32_t> code{0x7e060280u, mov(126, 128, true)};
    spill(code, 126, 0);
    if (seed_high) {
        code.push_back(mov(126, 193, true));
        spill(code, 127, 1);
    }
    // Genuine runtime group ID controls the arm. The alternate edge must also supply HIGH.
    code.insert(code.end(), {0xbf068000u, 0xbf840003u, mov(126, 128, true)});
    spill(code, conflicting_role ? 126 : 127, 1);
    reload(code, 20, 0);
    reload(code, 21, 1);
    portable_mask_half::sink(code);
    return code;
}
inline std::vector<uint32_t> loop_slots(bool conflicting_role) {
    using portable_mask_half::mov;
    auto code = program(false, true);
    code.resize(7); // keep both original spill producers, before either reload
    code.insert(code.end(), {mov(2, 128), 0xbf0a8302u, 0xbf840004u});
    spill(code, conflicting_role ? 126 : 127, 1);
    code.insert(code.end(), {0x80028102u, 0xbf82fffau});
    reload(code, 20, 0);
    reload(code, 21, 1);
    portable_mask_half::sink(code);
    return code;
}
inline std::vector<uint32_t> special_destination(uint32_t destination, bool readlane) {
    using portable_mask_half::mov;
    auto code = program(false, true);
    code.resize(11);
    const bool high = destination == 127 || destination == 107;
    code.push_back(mov(126, high ? 128 : 193, true));
    if (destination < 126) code.push_back(mov(106, high ? 128 : 193, true));
    if (readlane)
        reload(code, destination, high ? 1 : 0);
    else
        code.push_back(mov(destination, high ? 21 : 20));
    code.insert(code.end(), {0x7e060289u, mov(126, 193, true), 0xe0702000u, 0x80020300u,
                             portable_mask_half::end});
    return code;
}
struct Attempt {
    std::vector<uint32_t> module;
    std::string refusal;
};
inline Attempt compile(const std::vector<uint32_t>& code, bool dispatcher = false,
                       uint32_t old_input_words = 0) {
    using namespace gpu;
    ComputeShaderConfig config;
    config.local_x = config.wave_size = 64;
    config.tgid_x_en = true;
    config.force_exchange_dispatcher = dispatcher;
    config.user_sgprs.resize(old_input_words, UINT32_MAX);
    ShaderResourceTable table;
    ShaderResource output{};
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 8;
    table.resources.push_back(output);
    static std::atomic<uint64_t> next{0x493500000000ull};
    const uint64_t address = next.fetch_add(1);
    auto module = recompile_compute(code.data(), code.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}
} // namespace prosper::test::portable_mask_slots
