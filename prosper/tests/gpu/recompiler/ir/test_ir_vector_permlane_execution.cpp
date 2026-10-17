// Actual RDNA2 scalar selector tables must gather different lanes and preserve destination EXEC.
// FI=0/BC=0 expectations explicitly retain the old implementation's compatibility contract.
#include "gpu/recompiler/ir/permlane_cases.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::permlane;
namespace {
std::vector<uint32_t> compile(const LaneProgram& p, std::string* reason = nullptr) {
    const auto address = lane_harness_address();
    auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                    {RecompileDiagnosticStage::Compute, address});
    if (reason) *reason = last_terminal_reject_reason(address);
    return module;
}
void check(const std::vector<uint32_t>& module, const LaneInputs& in, const cases::Table& table,
           bool across, bool fetch, bool bound, bool alias = false) {
    ASSERT_FALSE(module.empty());
    ASSERT_TRUE(cases::has_shuffle(module));
    const auto run = cases::execute(module, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (uint32_t lane = 0; lane < in.lanes(); ++lane) {
        const auto answer = cases::value(in, lane, table, across, fetch, bound, alias);
        ASSERT_EQ(run.d[lane], in.active[lane] ? answer : in.old_d[lane]) << lane;
    }
}
void save(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.close();
    ASSERT_TRUE(out.good()) << path;
}
std::vector<uint32_t> load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        ADD_FAILURE() << path;
        return {};
    }
    const auto bytes = in.tellg();
    if (bytes <= 0 || bytes % 4) {
        ADD_FAILURE() << path;
        return {};
    }
    std::vector<uint32_t> out(static_cast<size_t>(bytes) / sizeof(uint32_t));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    EXPECT_TRUE(in.good());
    return out;
}
void export_one(const std::filesystem::path& root, std::ofstream& manifest, size_t index,
                bool across, const cases::Table& table, bool fetch, bool bound, uint32_t wave,
                bool alias) {
    const auto p = cases::program(across, table, fetch, bound, wave, alias);
    const uint64_t address = 0x5372000000ull + index * 0x100;
    const auto stem = root / "refused_shaders_20000101-000000Z_synthetic" /
                      ("cs_" + std::to_string(address) + "_" + std::to_string(index));
    const auto bin = std::filesystem::path(stem.string() + ".bin");
    const auto sidecar = std::filesystem::path(stem.string() + ".inputs");
    save(bin, p.code);
    ASSERT_TRUE(write_refused_compute_inputs(sidecar, address, &p.table, p.config,
                                             "synthetic permlane migration witness"));
    const auto raw = load(bin);
    ASSERT_EQ(raw, p.code);
    const auto inputs = read_refused_shader_inputs(sidecar);
    ASSERT_TRUE(inputs.incomplete.empty()) << inputs.incomplete;
    const auto replay = replay_refused_shader_inputs(raw, inputs);
    ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
    ASSERT_TRUE(replay.produced) << replay.reject;
    const auto spv = root / "modules" / (std::to_string(index) + ".spv");
    save(spv, replay.spirv);
    const auto saved = load(spv);
    ASSERT_EQ(saved, replay.spirv);
    check(saved, cases::inputs(wave), table, across, fetch, bound, alias);
    manifest << stem.filename().string() << '\t' << across << '\t' << fetch << '\t' << bound << '\t'
             << wave << '\t' << alias << '\t' << table.lo << '\t' << table.hi << '\n';
}
} // namespace
TEST(IrVectorPermlaneExecution, PackedScalarTablesMatchHandRowsInBothWaveSizes) {
    for (uint32_t wave : {32u, 64u})
        for (bool across : {false, true})
            for (const auto& table : cases::kTables)
                for (bool fetch : {false, true})
                    for (bool bound : {false, true}) {
                        const auto in = cases::inputs(wave);
                        check(compile(cases::program(across, table, fetch, bound, wave)), in, table,
                              across, fetch, bound);
                    }
}
TEST(IrVectorPermlaneExecution, SourceDestinationAliasAndEveryDestinationExecShapeStayExact) {
    for (uint32_t wave : {32u, 64u})
        for (bool across : {false, true})
            for (bool alias : {false, true})
                for (uint32_t mask : {0u, 1u, 2u})
                    for (bool fetch : {false, true})
                        for (bool bound : {false, true}) {
                            const auto in = cases::inputs(wave, mask);
                            check(compile(cases::program(across, cases::kTables[1], fetch, bound,
                                                         wave, alias)),
                                  in, cases::kTables[1], across, fetch, bound, alias);
                        }
}
TEST(IrVectorPermlaneExecution, InactivePeerAndUnmaskedDestinationAreIndependentControls) {
    for (uint32_t wave : {32u, 64u}) {
        const auto in = cases::inputs(wave);
        ASSERT_FALSE(in.active[0]);
        const auto& table = cases::kTables[3]; // Every row selects its first lane.
        for (bool across : {false, true}) {
            check(compile(cases::program(across, table, true, false, wave)), in, table, across,
                  true, false);
            const auto full =
                compile(cases::program(across, table, true, false, wave, false, true));
            ASSERT_FALSE(full.empty());
            const auto run = cases::execute(full, in);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), wave);
            size_t inactive = 0, overwritten = 0;
            for (uint32_t lane = 0; lane < wave; ++lane) {
                ASSERT_EQ(run.d[lane], in.s0[cases::source_lane(lane, table, across)]);
                if (!in.active[lane]) {
                    ++inactive;
                    overwritten += run.d[lane] != in.old_d[lane];
                }
            }
            ASSERT_GT(inactive, 0u);
            EXPECT_EQ(overwritten, inactive);
        }
    }
}
TEST(IrVectorPermlaneExecution, VectorSelectorRefusesAtThePermlaneBesideItsScalarControl) {
    for (bool across : {false, true})
        for (bool vector_selector : {false, true}) {
            const auto table = cases::kTables[0];
            auto p = cases::program(across, table, true, true, 64);
            std::vector<Rdna2Inst> decoded;
            rdna2_walk(p.code.data(), p.code.size(), decoded);
            const auto it = std::find_if(decoded.begin(), decoded.end(), [across](const auto& i) {
                return i.fmt == Rdna2Format::VOP3 && i.opcode == (across ? 0x378u : 0x377u);
            });
            ASSERT_NE(it, decoded.end());
            if (vector_selector)
                p.code[it->pc + 1] = (p.code[it->pc + 1] & ~(511u << 9)) | (258u << 9);
            std::string reason;
            const auto module = compile(p, &reason);
            EXPECT_EQ(module.empty(), vector_selector) << reason;
            if (vector_selector)
                EXPECT_NE(reason.find("pc=" + std::to_string(it->pc)), std::string::npos) << reason;
        }
}
TEST(IrVectorPermlaneExecution, SavedReplaysAreReloadedExecutedAndReadyForHashBinding) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "permlane16";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error);
    }
    ASSERT_FALSE(fs::exists(root)) << "export requires a new directory";
    ASSERT_TRUE(fs::create_directories(root / "modules"));
    ASSERT_TRUE(fs::create_directory(root / "refused_shaders_20000101-000000Z_synthetic"));
    std::ofstream manifest(root / "manifest.tsv");
    size_t index = 0;
    for (bool across : {false, true})
        for (uint32_t wave : {32u, 64u})
            for (bool fetch : {false, true})
                for (bool bound : {false, true})
                    for (bool alias : {false, true})
                        for (const auto& table : cases::kTables)
                            export_one(root, manifest, index++, across, table, fetch, bound, wave,
                                       alias);
    manifest.close();
    EXPECT_TRUE(manifest.good());
    EXPECT_EQ(index, 320u);
}
