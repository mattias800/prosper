#pragma once
// Project-owned Wave64 programs observe physical half replacements through the original EXEC
// write and buffer sink. No native wave vote or supplied guest capture establishes their words.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <vector>

namespace prosper::test::portable_mask_half {
using namespace gpu;
inline constexpr uint32_t end = 0xbf810000u;
inline constexpr uint32_t mov(uint32_t dst, uint32_t source, bool pair = false) {
    return 0xbe800000u | (dst << 16) | ((pair ? 4u : 3u) << 8) | source;
}
inline void word(std::vector<uint32_t>& code, uint32_t dst, uint32_t value) {
    code.insert(code.end(), {mov(dst, 255), value});
}
inline void counted(std::vector<uint32_t>& code) {
    // s2=0; while(s2<3) ++s2. The real counted-loop emitter owns this region.
    code.insert(code.end(), {mov(2, 128), 0xbf0a8302u, 0xbf840002u, 0x80028102u, 0xbf82fffcu});
}
inline void sink(std::vector<uint32_t>& code, uint32_t source = 20) {
    code.insert(code.end(), {mov(126, source, true), 0x7e060289u, mov(126, 193, true), 0xe0702000u,
                             0x80020300u, end});
}
inline std::vector<uint32_t> program(bool numeric, int replaced = -1, uint32_t value = 0,
                                     bool loop = false, bool copied = false) {
    std::vector<uint32_t> code{0x7e060280u, mov(20, numeric ? 193 : 126, true)};
    if (copied) code.push_back(mov(24, 20, true));
    if (loop) counted(code);
    if (replaced >= 0) word(code, (copied ? 24 : 20) + replaced, value);
    sink(code, copied ? 24 : 20);
    return code;
}
inline std::vector<uint32_t> one_arm(bool complete_entry) {
    std::vector<uint32_t> code{0x7e060280u, mov(20, 193, complete_entry)};
    // The scalar comparison reads genuine runtime WorkGroupId.x, never a baked current value.
    code.insert(code.end(), {0xbf068000u, 0xbf840002u});
    word(code, 21, 0);
    sink(code);
    return code;
}
inline std::vector<uint32_t> early_lane_exits(bool numeric, bool replaced = true) {
    // Increment v4 while localX>v4, removing lanes from EXEC at different iterations.
    // The saved pair predates that dynamic region; its words cannot be recovered from late EXEC.
    std::vector<uint32_t> code{0x7e060280u, mov(20, numeric ? 193 : 126, true),
                               0x7e080280u, 0x4a080881u,
                               0x7da80900u, 0xbf89fffdu};
    if (replaced) word(code, 21, 0);
    sink(code);
    return code;
}
inline std::vector<uint32_t> compile(const std::vector<uint32_t>& code, bool dispatcher = false) {
    ComputeShaderConfig config;
    config.local_x = config.wave_size = 64;
    config.native_subgroup_size = 0;
    config.tgid_x_en = true;
    config.force_exchange_dispatcher = dispatcher;
    ShaderResourceTable table;
    ShaderResource output{};
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 8;
    table.resources.push_back(output);
    return recompile_compute(code.data(), code.size(), &table, config);
}
}   // namespace prosper::test::portable_mask_half
