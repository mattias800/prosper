#include "gpu/capture/fragment_compile_case.hpp"
#include "build_revision.hpp"
#include <iostream>
#include <string_view>

using namespace prosper::gpu;
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--build-revision") {
            std::cout << prosper::embedded_build_revision() << '\n'; return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--compiler-identity") {
            std::cout << prosper::embedded_build_revision() << ':' << prosper::embedded_build_source_identity() << '\n'; return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "Usage: fragment_compile_replay --baseline|--candidate|--inspect-only CASE [--output SOURCE.spv]\n"
                         "CPU-only RDNA2-to-SOURCE replay; no Vulkan, subgroup admission or pixel oracle.\n"
                         "Exit 0: matched baseline / produced candidate / inspected. Exit 2: invalid or incomplete.\n"
                         "Exit 3: complete candidate still refused (no output written).\n"; return 0;
        }
        if (argc != 3 && argc != 5) throw std::runtime_error("invalid arguments; use --help");
        const std::string_view mode = argv[1];
        if (mode != "--baseline" && mode != "--candidate" && mode != "--inspect-only")
            throw std::runtime_error("exactly one replay mode is required");
        if (argc == 5 && (std::string_view(argv[3]) != "--output" || mode == "--inspect-only"))
            throw std::runtime_error("invalid output option");
        const auto c = read_fragment_compile_case(argv[2]);
        std::cout << "input=" << (c.complete ? "COMPLETE" : "INCOMPLETE")
                  << " reason=" << (c.reason.empty() ? "none" : c.reason)
                  << " expected=" << (c.expected_produced ? "PRODUCED" : "REFUSED")
                  << " raw_words=" << c.code.size() << " source_words=" << c.source.size()
                  << " semantic_reads=" << c.choices.reads.size()
                  << " wave_width=" << (c.wave32 ? 32 : 64) << " guest_float_mode="
                  << (c.float_mode.available ? std::to_string(c.float_mode.value) : "unknown")
                  << " guest_ieee_mode="
                  << (c.float_flags.available ? (c.float_flags.ieee_mode ? "1" : "0") : "unknown")
                  << " guest_dx10_clamp="
                  << (c.float_flags.available ? (c.float_flags.dx10_clamp ? "1" : "0") : "unknown")
                  << " rsrc1_ps_evidence="
                  << (c.launch_rsrc1.available ? std::to_string(c.launch_rsrc1.value) : "unknown")
                  << " host_float_transport=" << float_transport_profile_name(c.float_transport)
                  << " export_formats=" << c.export_formats.compressed_formats << '/'
                  << unsigned(c.export_formats.uint_outputs) << '/'
                  << unsigned(c.export_formats.sint_outputs) << '\n'
                  << "producing_compiler=" << c.compiler
                  << " replay_compiler=" << prosper::embedded_build_revision() << ':'
                  << prosper::embedded_build_source_identity() << '\n';
        if (!c.expected_reject.empty()) std::cout << "producer_refusal=" << c.expected_reject << '\n';
        if (mode == "--inspect-only") return 0;
        const bool baseline = mode == "--baseline";
        std::string actual_refusal;
        const auto result = replay_fragment_compile_case(c, baseline, &actual_refusal);
        std::cout << (baseline ? "BASELINE MATCH " : "CANDIDATE ")
                  << (result.empty() ? "REFUSED" : "PRODUCED") << " source_words=" << result.size() << '\n';
        if (!actual_refusal.empty()) std::cout << "actual_refusal=" << actual_refusal << '\n';
        if (!result.empty() && argc == 5) write_fragment_compile_source(argv[4], result);
        return !baseline && result.empty() ? 3 : 0;
    } catch (const std::exception& e) {
        std::cerr << "REPLAY REFUSED: " << e.what() << '\n'; return 2;
    }
}
