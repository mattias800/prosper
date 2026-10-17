// Synthetic sidecars bind the exact saved module to the buffers used by the Vulkan test.
#pragma once
#include "gpu/recompiler/ir/buffer_store_cases.hpp"
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <cstddef>
#include <ios>
#include <ostream>
#include <vector>

namespace prosper::test::ir::buffer_store {
struct WitnessCase {
    Case c;
    uint32_t opcode = 7;
    bool routed = false, admitted = true;
};
inline std::vector<WitnessCase> witness_cases() {
    std::vector<WitnessCase> out;
    for (const Case& c : cases()) out.push_back({c});
    for (uint32_t wave : {32u, 64u}) {
        for (uint32_t opcode : {0x1cu, 0x1du, 0x1eu, 0x1fu})
            out.push_back({{0, wave, 0, false, opcode}});
        for (uint32_t f : {9u, 10u, 11u, 12u, 13u})
            for (bool typed : {false, true}) out.push_back({{f, wave, 0, typed}, 5, false, false});
        out.push_back({{0, wave, 0, false}, 7, true, false});
        for (uint32_t f : {2u, 3u, 4u, 5u, 6u, 7u, 8u})
            for (bool typed : {false, true}) out.push_back({{f, wave, 1, typed}, 7, false, false});
    }
    return out;
}
inline bool save_words(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.close();
    return out.good();
}
inline std::vector<uint32_t> reopen_words(const std::filesystem::path& path, size_t count) {
    std::vector<uint32_t> out(count);
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(count * sizeof(uint32_t)));
    if (!in.good()) return {};
    return out;
}
inline void attributes(std::ostream& out, size_t index, const WitnessCase& x,
                       const std::filesystem::path& stem) {
    out << "{\"index\":" << index << ",\"format\":" << x.c.format << ",\"wave\":" << x.c.wave
        << ",\"typed\":" << x.c.typed << ",\"offset\":" << x.c.offset
        << ",\"raw_opcode\":" << x.c.raw_opcode << ",\"opcode\":" << x.opcode
        << ",\"routed\":" << x.routed << ",\"accepted\":" << x.admitted << ",\"stem\":\""
        << stem.filename().string() << "\"}\n";
}
} // namespace prosper::test::ir::buffer_store
