// cfg_fixtures.hpp -- the committed guest-program fixtures in tests/data, for the ir/cfg shadow
// tests: every `.hex` program (little-endian bytes as hex text) and raw shader `.bin` file.
#pragma once

#include "fixtures/test_data.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace prosper::test::cfg_fixtures {

inline uint8_t nibble(char c) {
    return static_cast<uint8_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10);
}

// A tests/data program: `.hex` holds the little-endian bytes as hex text, `.bin` the raw bytes.
inline std::vector<uint32_t> load_words(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::vector<uint8_t> bytes;
    if (path.extension() == ".hex") {
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        for (size_t i = 0; i + 1 < text.size(); i += 2)
            bytes.push_back(static_cast<uint8_t>(nibble(text[i]) << 4 | nibble(text[i + 1])));
    } else {
        bytes.assign(text.begin(), text.end());
    }
    std::vector<uint32_t> words(bytes.size() / 4);
    for (size_t i = 0; i < words.size(); ++i)
        words[i] = bytes[4 * i] | bytes[4 * i + 1] << 8 | bytes[4 * i + 2] << 16 |
                   static_cast<uint32_t>(bytes[4 * i + 3]) << 24;
    return words;
}

// `source_file` is the caller's __FILE__, which locates tests/.
inline std::vector<std::filesystem::path> guest_program_fixtures(const char* source_file) {
    std::vector<std::filesystem::path> out;
    const auto data = prosper::test::tests_root(source_file) / "data";
    for (const auto& entry : std::filesystem::directory_iterator(data)) {
        const auto ext = entry.path().extension();
        const auto name = entry.path().filename().string();
        // Every guest-code fixture: the .hex programs and the raw shader .bin files. The
        // il2cpp symbol table and the console oracle folder are not guest code.
        if (ext == ".hex" || (ext == ".bin" && name.find("symtab") == std::string::npos))
            out.push_back(entry.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

}   // namespace prosper::test::cfg_fixtures
