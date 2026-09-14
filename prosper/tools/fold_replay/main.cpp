#include "gpu/capture/fold_capture.hpp"
#include "gpu/resources/fold_control_plan.hpp"
#include "build_revision.hpp"
#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace prosper::gpu;
namespace {
uint32_t number(std::string_view text, uint32_t cap) {
    uint32_t n = 0;
    int base = 10;
    if (text.starts_with("0x")) { text.remove_prefix(2); base = 16; }
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), n, base);
    if (error != std::errc{} || end != text.data() + text.size() || n > cap)
        throw std::runtime_error("invalid bounded integer");
    return n;
}
template<class T> size_t changed(const std::vector<T>& a, const std::vector<T>& b) {
    size_t count = std::max(a.size(), b.size()) - std::min(a.size(), b.size());
    for (size_t n = 0; n < std::min(a.size(), b.size()); ++n) count += a[n] != b[n];
    return count;
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string_view(argv[1]) == "--help") {
            std::cout << "Usage: fold_replay FILE [--iterations N] [--cold-each] "
                         "[--mutate-word EVENT=VALUE]\n"
                         "Offline evaluator timings include transcript validation and output comparison; "
                         "they exclude live mapping probes and GPU work.\n";
            return argc < 2 ? 2 : 0;
        }
        auto capture = read_fold_capture(argv[1]);
        uint32_t iterations = 1000;
        bool mutation = false, cold_each = false;
        for (int n = 2; n < argc; ++n) {
            const std::string_view option = argv[n];
            if (option == "--cold-each") { cold_each = true; continue; }
            if (++n == argc) throw std::runtime_error("missing option value");
            const std::string_view value = argv[n];
            if (option == "--iterations") iterations = number(value, 1000000);
            else if (option == "--mutate-word") {
                const auto separator = value.find('=');
                if (separator == value.npos) throw std::runtime_error("mutation requires EVENT=VALUE");
                const auto index = number(value.substr(0, separator), kFoldMaxEvents);
                const auto word = number(value.substr(separator + 1), UINT32_MAX);
                if (index >= capture.events.size() || capture.events[index].kind != FoldEventKind::Word)
                    throw std::runtime_error("mutation must select an existing word event");
                for (unsigned k = 0; k < 4; ++k) capture.events[index].data[k] = uint8_t(word >> (8 * k));
                mutation = true;
            } else throw std::runtime_error("unknown option");
        }
        size_t payload = 0;
        for (const auto& e : capture.events) payload += e.data.size();
        std::cout << "recorded_revision=" << std::quoted(capture.input.revision)
                  << " evaluator_revision=" << std::quoted(prosper::embedded_build_revision()) << '\n'
                  << "mode=" << (mutation ? "explicit-mutation" : "compare-all-outputs")
                  << " code_dwords=" << capture.input.code.size()
                  << " decoded_dwords=" << capture.decoded_dwords
                  << " events=" << capture.events.size() << " payload_bytes=" << payload
                  << " file_bytes=" << std::filesystem::file_size(argv[1]) << '\n';
        clear_shader_decode_cache();
        auto builds = fold_control_plan_builds.load();
        auto start = std::chrono::steady_clock::now();
        auto result = replay_fold(capture, !mutation);
        const auto cold_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "cold_replay_ns=" << cold_ns
                  << " control_plan_builds=" << fold_control_plan_builds.load() - builds
                  << " evaluated_instructions=" << result.evaluated_instructions << '\n';
        uint64_t elapsed = 0, instructions = 0;
        builds = fold_control_plan_builds.load();
        for (uint32_t n = 0; n < iterations; ++n) {
            if (cold_each) clear_shader_decode_cache();
            start = std::chrono::steady_clock::now();
            auto next = replay_fold(capture, !mutation);
            elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
            instructions += next.evaluated_instructions;
        }
        std::cout << "repeat_mode=" << (cold_each ? "cold" : "warm") << " iterations=" << iterations
                  << " total_replay_ns=" << elapsed << " mean_replay_ns="
                  << (iterations ? double(elapsed) / iterations : 0)
                  << " control_plan_builds=" << fold_control_plan_builds.load() - builds
                  << " evaluated_instructions=" << instructions << '\n'
                  << "fetches=" << result.fetches.size() << " uses=" << result.uses.size()
                  << " changed_fetches=" << changed(result.fetches, capture.fetches)
                  << " changed_uses=" << changed(result.uses, capture.uses) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fold_replay: REFUSED: " << error.what() << '\n';
        return 1;
    }
}
