#pragma once
// Project-owned common-phase observations. LOW/HIGH payloads retain their actual producers;
// scalar votes, bit indices and prefix counts must not use a lone LOW carrier as a whole mask.
#include "portable_mask_slots.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>
namespace prosper::test::portable_mask_common {
namespace f = portable_mask_slots;
namespace base = portable_mask_half;
inline std::vector<uint32_t> halves(bool low, bool high, bool reverse = true) {
    auto code = f::program(low, high, reverse);
    code.resize(11);
    return code;
}
inline std::vector<uint32_t> missing(bool high) {
    auto code = f::missing_half(high);
    code.resize(6);
    return code;
}
inline std::vector<uint32_t> branch(std::vector<uint32_t> code, uint32_t root = 20,
                                    bool vopc = false, uint32_t other = 128) {
    code.push_back(base::mov(126, 193, true));
    if (vopc)
        code.insert(code.end(), {0xd0e4006au, root | (128u << 9), 0xbf860001u});
    else
        code.insert(code.end(), {0xbf120000u | (other << 8) | root, 0xbf850001u});
    code.insert(code.end(),
                {0x7e060289u, base::mov(126, 193, true), 0xe0702000u, 0x80020300u, base::end});
    return code;
}
inline void store(std::vector<uint32_t>& code) {
    code.insert(code.end(), {0xe0702000u, 0x80020300u, base::end});
}
inline std::vector<uint32_t> reduction(std::vector<uint32_t> code, uint32_t opcode) {
    code.insert(code.end(),
                {base::mov(126, 193, true), 0xbe980000u | (opcode << 8) | 20u, 0x7e060218u});
    store(code);
    return code;
}
inline std::vector<uint32_t> selected_half(bool high, uint32_t bits) {
    std::vector<uint32_t> code{0x7e060280u};
    base::word(code, 30, high ? 0u : bits);
    base::word(code, 31, high ? bits : 0u);
    code.push_back(base::mov(126, 30, true));
    f::spill(code, high ? 127 : 126, 13);
    f::reload(code, high ? 21 : 20, 13);
    code.push_back(base::mov(126, 193, true));
    return code;
}
inline std::vector<uint32_t> ffbh(std::vector<uint32_t> code, uint32_t source) {
    code.push_back(0x7e060000u | (gpu::kVop1OpcodeFfbhU32 << 9) | source);
    store(code);
    return code;
}
inline std::vector<uint32_t> mbcnt(std::vector<uint32_t> code, uint32_t source, bool high,
                                   uint32_t accumulator = 7) {
    code.insert(code.end(),
                {high ? 0xd7660003u : 0xd7650003u, source | ((128u + accumulator) << 9)});
    store(code);
    return code;
}
// A separate complete mask stays current between unrelated symbolic half words.
inline std::vector<uint32_t> whole_neighbors() {
    auto code = halves(false, true);
    code.insert(code.end(),
                {base::mov(126, 193, true), base::mov(22, 126, true), base::mov(24, 20)});
    return code;
}
inline std::vector<uint32_t> whole_mbcnt(std::vector<uint32_t> code) {
    code.insert(code.end(), {0xd7650003u, 0x00010016u, 0xd7660003u, 0x00020617u});
    store(code);
    return code;
}
// Keep partial-wave fixtures separate from the original fixed-64 slot programs.
inline f::Attempt compile(const std::vector<uint32_t>& code,
                          const gpu::ComputeShaderConfig& config) {
    using namespace gpu;
    ShaderResourceTable table;
    ShaderResource output{};
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 8;
    table.resources.push_back(output);
    static std::atomic<uint64_t> next{0x493502000000ull};
    const uint64_t address = next.fetch_add(1);
    auto module = recompile_compute(code.data(), code.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}
inline f::Attempt compile(const std::vector<uint32_t>& code, uint32_t local,
                          bool dispatcher = true) {
    gpu::ComputeShaderConfig config;
    config.local_x = local;
    config.wave_size = 64;
    config.tgid_x_en = true;
    config.force_exchange_dispatcher = dispatcher;
    return compile(code, config);
}
inline gpu::ComputeShaderConfig exact_config(uint32_t x, uint32_t y, uint32_t z, uint32_t threads_x,
                                             uint32_t threads_y, uint32_t threads_z) {
    gpu::ComputeShaderConfig config;
    config.local_x = x;
    config.local_y = y;
    config.local_z = z;
    config.threads_x = threads_x;
    config.threads_y = threads_y;
    config.threads_z = threads_z;
    config.exact_thread_extent = true;
    config.force_exchange_dispatcher = true;
    config.tgid_x_en = true;
    config.wave_size = 64;
    return config;
}
inline std::vector<uint32_t> readfirstlane(std::vector<uint32_t> code) {
    // All programs initialize v3=0 first. The genuine unused source triggers the exact ACTIVE route.
    code.insert(code.begin() + 1, 0x7e3c0503u);   // s30 = readfirstlane(v3)
    return code;
}
// Owned oracle calibration: dead poison may be discarded; live storage and branch uses refuse.
// These harness boundaries are separate from the RDNA2 compiler admissions above.
inline std::vector<uint32_t> poison_select(bool live, bool branch = false,
                                           bool unsupported = false) {
    std::vector<uint32_t> m{0x07230203u, 0x00010300u, 0u,
                            unsupported ? 25u
                            : branch    ? 23u
                                        : 20u,
                            0u};
    const auto emit = [&](uint32_t op, std::initializer_list<uint32_t> a) {
        m.push_back((static_cast<uint32_t>(a.size() + 1) << 16) | op);
        m.insert(m.end(), a);
    };
    emit(17, {1});
    emit(14, {0, 1});
    emit(15, {5, 1, 0x6e69616du, 0});
    emit(16, {1, 17, 64, 1, 1});
    emit(71, {7, 6, 4});
    emit(72, {8, 0, 35, 0});
    emit(71, {8, 2});
    emit(71, {14, 34, 0});
    emit(71, {14, 33, 3});
    emit(19, {2});
    emit(33, {3, 2});
    emit(21, {4, 32, 0});
    emit(20, {5});
    emit(32, {6, 7, 5});
    emit(29, {7, 4});
    emit(30, {8, 7});
    emit(32, {9, 12, 8});
    emit(32, {10, 12, 4});
    emit(43, {4, 11, 0});
    emit(43, {4, 12, 9});
    if (unsupported) {
        emit(22, {23, 32});
        emit(43, {23, 24, 0});
    }
    emit(59, {9, 14, 12});
    emit(54, {2, 1, 0, 3});
    emit(248, {15});
    emit(59, {6, 16, 7});
    if (unsupported)
        emit(156, {5, 17, 24});   // valid but unmodeled OpIsNan
    else
        emit(61, {5, 17, 16});   // explicit uninitialized Function Bool storage
    emit(169, {4, 18, 17, 12, 11});
    if (branch) {
        emit(247, {22, 0});
        emit(250, {17, 20, 21});
        emit(248, {20});
        emit(249, {22});
        emit(248, {21});
        emit(249, {22});
        emit(248, {22});
    }
    emit(65, {10, 19, 14, 11, 11});
    emit(62, {19, live ? 18u : 12u});
    emit(253, {});
    emit(56, {});
    return m;
}
}   // namespace prosper::test::portable_mask_common
