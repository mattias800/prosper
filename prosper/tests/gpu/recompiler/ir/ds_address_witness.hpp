// Bind each saved DS module to its independently derived buffers and matched refusal contract.
#pragma once
#include "gpu/recompiler/ir/ds_address_cases.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <cstddef>
#include <ios>
#include <ostream>
#include <vector>

namespace prosper::test::ir::ds_address {
struct WitnessCase {
    Case c;
    bool small_lds = false, gds_pair = false;
    bool admitted() const { return !small_lds && !gds_pair; }
};
inline std::vector<WitnessCase> witness_cases() {
    std::vector<WitnessCase> out;
    out.reserve(74);
    for (const auto& c : cases()) out.push_back({c});
    for (uint32_t wave : {32u, 64u}) {
        for (uint32_t form : {10u, 11u}) out.push_back({{form, wave, 0x0100}, true});
        out.push_back({{9, wave, 0x0100}, false, true});
    }
    return out;
}
inline LaneProgram witness_program(const WitnessCase& x) {
    auto p = program(x.c);
    if (x.small_lds) p.config.lds_bytes = 256;
    if (x.gds_pair)
        for (size_t pc = 0; pc < p.code.size();) {
            const auto in = prosper::gpu::rdna2_decode_one(p.code.data() + pc, p.code.size() - pc);
            if (!in.len_dwords) break;
            if (in.fmt == prosper::gpu::Rdna2Format::DS && in.opcode == 0x37)
                p.code[pc] |= 1u << 17;
            pc += in.len_dwords;
        }
    return p;
}
inline bool save_words(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.close();
    return out.good();
}
inline std::vector<uint32_t> reopen_words(const std::filesystem::path& path, size_t count) {
    std::vector<uint32_t> words(count);
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(words.data()),
            static_cast<std::streamsize>(count * sizeof(uint32_t)));
    if (!in.good()) return {};
    return words;
}
inline void attributes(std::ostream& out, size_t index, const WitnessCase& x,
                       const std::filesystem::path& stem) {
    out << "{\"index\":" << index << ",\"form\":" << x.c.form << ",\"wave\":" << x.c.wave
        << ",\"literal\":" << x.c.literal << ",\"small_lds\":" << x.small_lds
        << ",\"gds_pair\":" << x.gds_pair << ",\"accepted\":" << x.admitted() << ",\"stem\":\""
        << stem.filename().string() << "\"}\n";
}
} // namespace prosper::test::ir::ds_address
