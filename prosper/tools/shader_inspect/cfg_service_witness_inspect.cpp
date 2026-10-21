// Test-only adapter: native replay for ordinary sidecars, plus the owned packet absent from that
// codec. V1's fixed packet profile supplies all logical slots, masks and quad topology explicitly.
// Reproduce: build test_cfg_service_inspect, --export-corpus <NEW_SOURCE>, then run
// recompile_corpus.py harvest <NEW_CORPUS> --root <NEW_SOURCE>. Compare <NEW_CORPUS> using the
// same adapter source linked against each build's core; only PacketWqm uses the V1 test profile.
// Add --cfg-services to inspect real per-block nodes and independent dispatcher-tail events.
#include "gpu/recompiler/ir/cfg_service_witness_corpus.hpp"
#include "gpu/recompiler/ir/cfg_service_witnesses.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "shader_inspect/cfg_services_report.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
namespace fixture = prosper::test::cfg_services;
namespace fs = std::filesystem;

struct Args {
    fs::path code, profile, output, export_root;
    std::string stage;
    bool census = false, synthetic = false;
};

Args parse(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--cfg-services")
            args.census = true;
        else if (key == "--synthetic-table")
            args.synthetic = true;
        else if (key == "--inputs" || key == "--spirv-out" || key == "--stage" ||
                 key == "--export-corpus") {
            if (++i == argc) throw std::runtime_error("missing option value");
            if (key == "--inputs") args.profile = argv[i];
            if (key == "--spirv-out") args.output = argv[i];
            if (key == "--stage") args.stage = argv[i];
            if (key == "--export-corpus") args.export_root = argv[i];
        } else if (key.starts_with('-') || !args.code.empty()) {
            throw std::runtime_error("unknown argument");
        } else {
            args.code = key;
        }
    }
    if (!args.export_root.empty()) {
        if (!args.code.empty() || !args.profile.empty() || !args.stage.empty() ||
            !args.output.empty() || args.census || args.synthetic)
            throw std::runtime_error("export takes only a new directory");
    } else if (args.code.empty() || (args.profile.empty() == args.stage.empty()) ||
               (args.census && args.profile.empty()) || (args.synthetic && !args.profile.empty())) {
        throw std::runtime_error("supply a program and one explicit input mode");
    }
    return args;
}

void usage() {
    std::fprintf(stderr, "usage: test_cfg_service_inspect <program.bin> --inputs <PROFILE> "
                         "[--spirv-out FILE] [--cfg-services]\n"
                         "       test_cfg_service_inspect --export-corpus <NEW_DIRECTORY>\n"
                         "       --stage <STAGE> [--synthetic-table] reports profile-required; "
                         "never infers owned packet inputs\n");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            usage();
            return 2;
        }
        const Args args = parse(argc, argv);
        if (!args.export_root.empty()) {
            fixture::export_corpus(args.export_root);
            std::printf("cfg-services-export witnesses=%zu\n", fixture::witnesses().size());
            return 0;
        }
        if (!args.stage.empty()) {
            (void)fixture::read_words(args.code); // IO failure is not a shader refusal.
            std::printf("stage-recompile stage=%s status=undetermined\n", args.stage.c_str());
            std::fprintf(stderr,
                         "[recompile-reject] synthetic-profile-required stage=%s "
                         "program=none role=terminal\n",
                         args.stage.c_str());
            return args.synthetic ? 5 : 1;
        }
        fixture::publish_support();
        std::unique_ptr<prosper::tools::CfgServicesReport> census;
        if (args.census) census = std::make_unique<prosper::tools::CfgServicesReport>();
        const auto replay = fixture::replay_witness(args.code, args.profile);
        if (!replay.produced) {
            std::string reason = replay.incomplete.empty() ? replay.reject : replay.incomplete;
            for (char& c : reason)
                if (c == '"' || c == '\n' || c == '\r') c = ' ';
            std::printf("inputs-replay status=%s reject=\"%s\"\n",
                        replay.incomplete.empty() ? "rejected" : "incomplete", reason.c_str());
            return 0;
        }
        const auto& module = replay.spirv;
        if (!args.output.empty() && !fixture::write_words(args.output, module))
            throw std::runtime_error("failed to write SPIR-V output");
        std::printf("inputs-replay status=ok reject=\"\" spirv_dwords=%zu\n", module.size());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "test_cfg_service_inspect: %s\n", error.what());
        return 2;
    }
}
