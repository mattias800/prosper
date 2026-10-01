// #4007: report every reject/positive arm, so a fault control cannot stop at its first failure.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "../../fixtures/spirv_wave_width_fixtures.hpp"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    const auto corpus = prosper::test::wave_width::fixtures();
    const bool dumping = argc == 3 && std::string(argv[1]) == "--dump";
    if (argc != 1 && !dumping) {
        std::fprintf(stderr, "usage: test_wave_width_independent [--dump DIRECTORY]\n");
        return 2;
    }
    int failures = 0;
    if (dumping) std::filesystem::create_directories(argv[2]);
    for (const auto& fixture : corpus) {
        if (dumping) {
            const auto path = std::filesystem::path(argv[2]) / (fixture.name + ".spv");
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(fixture.words.data()),
                       static_cast<std::streamsize>(fixture.words.size() * sizeof(uint32_t)));
            if (!file) { std::printf("  [FAIL] write %s\n", fixture.name.c_str()); ++failures; }
            continue;
        }
        const bool got = prosper::gpu::fragment_spirv_wave_width_independent(fixture.words);
        const bool ok = got == fixture.independent;
        std::printf("  [%s] %s: %s, expected %s%s\n", ok ? "ok" : "FAIL", fixture.name.c_str(),
                    got ? "independent" : "NOT PROVEN",
                    fixture.independent ? "independent" : "NOT PROVEN",
                    fixture.strict_vulkan ? "" : " (parser-only, unsupported Vulkan environment)");
        failures += !ok;
    }
    if (!dumping) {
        // Every ordinary pair changes just the observable index/branch operand, not the vote or
        // initialized data. The opaque-call control removes one instruction rather than one word.
        for (size_t i = 0; i + 1 < corpus.size() - 3; i += 2) {
            if (corpus[i].name.starts_with("opaque_call") ||
                corpus[i].name.starts_with("callee_return")) continue;
            const auto& a = corpus[i].words;
            const auto& b = corpus[i + 1].words;
            size_t changed = a.size() == b.size() ? 0 : 2;
            if (a.size() == b.size())
                for (size_t j = 0; j < a.size(); ++j) changed += a[j] != b[j];
            if (changed != 1) {
                std::printf("  [FAIL] %s pair changes %zu words, expected 1\n",
                            corpus[i].name.c_str(), changed);
                ++failures;
            }
        }
    }
    std::printf("== %s: %zu fixtures, %d failures ==\n", failures ? "FAIL" : "PASS",
                corpus.size(), failures);
    return failures ? 1 : 0;
}
