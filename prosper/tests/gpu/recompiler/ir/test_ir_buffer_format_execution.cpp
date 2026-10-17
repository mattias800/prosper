// Each admitted buffer format executes actual emitted SPIR-V with independent binding4 bytes.
// The witness retains exact raw code, compile inputs and the saved/reopened/executed module.
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/buffer_format_cases.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/ir/buffer_format_execution_cases.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <system_error>
#include <set>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace fixture = prosper::test::ir::buffer_format;

namespace {
uint64_t address(size_t index) {
    return 0x5384000000ull + index * 0x1000;
}

bool save(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * 4));
    out.close();
    return out.good();
}

TEST(IrBufferFormatExecution, ExportsAndExecutesEveryAdmittedWitnessWithMatchedRefusals) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): launch environment is immutable in this process.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root("ir_buffer_format_witness");
    if (!configured) {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const fs::path dump = compare_witness_dump("ir_buffer_format_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream attributes(root / "attributes.jsonl");
    size_t index = 0, accepted = 0, refused = 0;
    for (const auto& c : fixture::cases()) {
        SCOPED_TRACE(::testing::Message()
                     << "case=" << index << " format=" << c.format << " component=" << c.component
                     << " typed=" << c.typed << " dynamic=" << c.dynamic << " raw=" << c.raw_opcode
                     << " wave=" << c.wave);
        const auto p = fixture::program(c);
        const Rdna2Inst decoded = rdna2_decode_one(p.code.data() + 10, p.code.size() - 10);
        ASSERT_EQ(decoded.fmt, c.typed ? Rdna2Format::MTBUF : Rdna2Format::MUBUF);
        ASSERT_EQ(decoded.opcode, c.raw_opcode ? c.raw_opcode : 3u);
        const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                              {RecompileDiagnosticStage::Compute, address(index)});
        ASSERT_EQ(!module.empty(), c.admitted()) << last_terminal_reject_reason(address(index));
        const std::string rejection = module.empty() ? last_terminal_reject_reason(address(index))
                                                     : "synthetic-witness (no live refusal)";
        const fs::path stem =
            dump / ("cs_" + std::to_string(address(index)) + "_" + std::to_string(index));
        ASSERT_TRUE(save(stem.string() + ".bin", p.code));
        ASSERT_TRUE(write_refused_compute_inputs(stem.string() + ".inputs", address(index),
                                                 &p.table, p.config, rejection));
        const auto replay = replay_refused_shader_inputs(
            p.code, read_refused_shader_inputs(stem.string() + ".inputs"));
        ASSERT_EQ(replay.produced, c.admitted()) << replay.reject << " " << replay.incomplete;
        if (c.admitted()) {
            ASSERT_EQ(replay.spirv, module);
            ASSERT_TRUE(save(stem.string() + ".spv", module));
            std::ifstream in(stem.string() + ".spv", std::ios::binary);
            std::vector<uint32_t> reopened(module.size());
            in.read(reinterpret_cast<char*>(reopened.data()),
                    static_cast<std::streamsize>(module.size() * 4));
            ASSERT_TRUE(in.good());
            ASSERT_EQ(reopened, module);
            const auto data = fixture::data(c);
            bpermute_oracle::Interpreter vm(reopened);
            vm.half_rte_preserve_denorm = true;   // selected CPU profile; no device-mode inference
            auto out = vm.run_buffers(c.wave, {{3, data.input}, {4, data.payload}}, 3);
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            ASSERT_GE(out.size(), c.wave);
            out.resize(c.wave);
            ASSERT_TRUE(save(stem.string() + ".buffer3", data.input));
            ASSERT_TRUE(save(stem.string() + ".buffer4", data.payload));
            ASSERT_TRUE(save(stem.string() + ".actual", out));
            ASSERT_TRUE(save(stem.string() + ".expected", data.expected));
            ASSERT_EQ(out, data.expected);
            ++accepted;
        } else {
            ASSERT_EQ(replay.reject, rejection);
            ASSERT_FALSE(rejection.empty());
            ++refused;
        }
        attributes << "{\"index\":" << index << ",\"format\":" << c.format
                   << ",\"component\":" << c.component << ",\"wave\":" << c.wave
                   << ",\"typed\":" << c.typed << ",\"dynamic\":" << c.dynamic
                   << ",\"raw_opcode\":" << c.raw_opcode << ",\"offset\":" << c.offset
                   << ",\"accepted\":" << c.admitted() << ",\"stem\":\"" << stem.filename().string()
                   << "\"}\n";
        ++index;
    }
    attributes.close();
    ASSERT_TRUE(attributes.good());
    EXPECT_EQ(index, 464u);
    EXPECT_EQ(accepted, 384u);
    EXPECT_EQ(refused, 80u);
}

TEST(IrBufferFormatExecution, AppendedTfeStatusRemainsARefusalForEveryFormat) {
    size_t index = 0, checked = 0;
    for (const auto& c : fixture::cases()) {
        if (!c.admitted() || c.component || c.wave != 32) continue;
        auto p = fixture::program(c);
        p.code[11] |= 1u << 23;
        const Rdna2Inst decoded = rdna2_decode_one(p.code.data() + 10, p.code.size() - 10);
        ASSERT_TRUE(c.typed ? decoded.mtbuf_tfe : decoded.mubuf_tfe);
        const uint64_t site = address(index++) + 0x10000000;
        const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                              {RecompileDiagnosticStage::Compute, site});
        EXPECT_TRUE(module.empty());
        EXPECT_FALSE(last_terminal_reject_reason(site).empty());
        ++checked;
    }
    EXPECT_EQ(checked, 54u);   // 14*2 static + 9*2 dynamic + four raw opcodes at two offsets
}

TEST(IrBufferFormatExecution, TheFixtureHasDistinctDataAndInactiveOldDestinations) {
    for (const auto& c : fixture::cases()) {
        if (!c.admitted() || c.component || c.wave != 64) continue;
        const auto data = fixture::data(c);
        std::set<uint32_t> active;
        std::set<uint32_t> active_byte_positions;
        size_t inactive = 0;
        for (uint32_t lane = 0; lane < c.wave; ++lane) {
            if (data.lanes.active[lane]) {
                active.insert(data.expected[lane]);
                active_byte_positions.insert((data.lanes.s0[lane] + c.offset) & 3u);
            } else {
                ++inactive;
                EXPECT_EQ(data.expected[lane], data.lanes.old_d[lane]);
            }
        }
        EXPECT_GT(active.size(), !c.raw_opcode && fixture::kFormats[c.format].kind ==
                                                      fixture::BufferFieldKind::Half
                                     ? 2u
                                     : 8u);
        EXPECT_GE(inactive, 2u);
        // The modulo-five EXEC pattern must leave every modulo-four address residue active.
        if (c.dynamic) EXPECT_EQ(active_byte_positions, (std::set<uint32_t>{0, 1, 2, 3}));
    }
}
}   // namespace
