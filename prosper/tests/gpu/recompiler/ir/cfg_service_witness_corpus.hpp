// Ordinary profiles use the existing live sidecar codec. Only the owned packet needs a test-only
// profile: V1 freezes packet()'s inputs and exact code. Change its marker if those inputs change.
#pragma once

#include "cfg_service_witnesses.hpp"
#include "gpu/diagnostics/refused_graphics_inputs.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/compiler_choices.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace prosper::test::cfg_services {
inline constexpr std::string_view kProfileMagic = "PROSPER_CFG_SERVICE_PACKET_WITNESS_V1\n";

inline bool write_words(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream file(path, std::ios::binary);
    for (uint32_t word : words)
        for (uint32_t byte = 0; byte < 4; ++byte)
            file.put(static_cast<char>((word >> (byte * 8)) & 0xffu));
    file.close();
    return bool(file);
}

inline bool write_witness(const std::filesystem::path& stem, const Witness& w) {
    if (!write_words(stem.string() + ".bin", w.code)) return false;
    const auto path = stem.string() + ".inputs";
    constexpr const char* reason = "synthetic-witness (no live refusal)";
    if (w.profile == Profile::Compute)
        return gpu::write_refused_compute_inputs(path, 0, nullptr, compute_config(), reason);
    if (w.profile == Profile::Fragment) {
        gpu::RefusedGraphicsInputs inputs;
        inputs.selector = gpu::RefusedGraphicsCompiler::Fragment;
        inputs.raw = inputs.invocation.code = w.code;
        inputs.invocation.compiler = "recompile_fragment";
        inputs.live_reject = reason;
        {
            gpu::CompilerChoiceScope recording(inputs.invocation.choices);
            inputs.invocation.source = compile(w);
        }
        inputs.invocation.expected_produced = !inputs.invocation.source.empty();
        gpu::own_refused_graphics_resources(inputs, nullptr);
        return gpu::write_refused_graphics_inputs(path, "ps", inputs);
    }
    std::ofstream profile(stem.string() + ".inputs", std::ios::binary);
    profile << kProfileMagic << w.name << '\n';
    profile.close();
    return bool(profile);
}

inline std::vector<uint32_t> read_words(const std::filesystem::path& path) {
    const auto bytes = std::filesystem::file_size(path);
    if (bytes == 0 || bytes % 4 || bytes > 4096)
        throw std::runtime_error("invalid synthetic witness size");
    std::ifstream file(path, std::ios::binary);
    std::vector<uint32_t> words(bytes / 4);
    for (auto& word : words)
        for (uint32_t byte = 0; byte < 4; ++byte) {
            const auto value = file.get();
            if (value == std::char_traits<char>::eof())
                throw std::runtime_error("truncated synthetic witness");
            word |= static_cast<uint32_t>(static_cast<unsigned char>(value)) << (byte * 8);
        }
    return words;
}

inline const Witness& read_packet_witness(const std::filesystem::path& code,
                                          const std::filesystem::path& profile) {
    if (std::filesystem::file_size(profile) > 256)
        throw std::runtime_error("oversized synthetic witness profile");
    std::ifstream file(profile, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(file)), {});
    for (const auto& w : witnesses()) {
        if (w.profile != Profile::Packet || text != std::string(kProfileMagic) + w.name + '\n')
            continue;
        if (read_words(code) != w.code)
            throw std::runtime_error("synthetic witness code/profile mismatch");
        return w;
    }
    throw std::runtime_error("unknown synthetic witness profile");
}

inline gpu::RefusedShaderReplay replay_witness(const std::filesystem::path& code,
                                               const std::filesystem::path& profile) {
    std::ifstream file(profile, std::ios::binary);
    std::string magic(kProfileMagic.size(), '\0');
    file.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (magic != kProfileMagic)
        return gpu::replay_refused_shader_inputs(read_words(code),
                                                 gpu::read_refused_shader_inputs(profile));
    const auto& w = read_packet_witness(code, profile);
    auto program = gpu::recompile_fragment_packet(packet(w));
    gpu::RefusedShaderReplay replay;
    replay.produced = !program.spirv.empty();
    replay.spirv = std::move(program.spirv);
    replay.reject = program.rejection;
    return replay;
}

inline void export_corpus(const std::filesystem::path& root) {
    if (std::filesystem::exists(root) || !std::filesystem::create_directories(root))
        throw std::runtime_error("synthetic export requires a new directory");
    // The ordinary corpus harvester's established synthetic sentinel, never a live capture date.
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    std::filesystem::create_directory(dump);
    std::ofstream index(root / "witnesses.tsv");
    size_t ordinal = 0;
    for (const auto& w : witnesses()) {
        const char* prefix = w.profile == Profile::Compute ? "cs_" : "ps_";
        // Decimal digits are also hexadecimal, as required by the ordinary dump harvester.
        // Its corpus key hashes the actual code and inputs; this ID only labels a synthetic source.
        const auto stem = dump / (std::string(prefix) + "0_" + std::to_string(ordinal++));
        if (!write_witness(stem, w)) throw std::runtime_error("failed to write synthetic witness");
        index << stem.filename().string() << '\t' << w.name << '\n';
    }
    index.close();
    if (!index) throw std::runtime_error("failed to write synthetic witness index");
}
}   // namespace prosper::test::cfg_services
