// CPU-only exact ISA proofs and hosted-source admission. Renderer/upload/replay execution is
// covered separately: these controls do not create a device or execute a captured SPIR-V module.
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <array>
#include <cstdio>
#include <vector>

using namespace prosper::gpu;
static int failures = 0;
static void check(bool ok, const char* stage, const char* label) {
    std::printf("[%s] %s: %s\n", ok ? "ok" : "FAIL", stage, label);
    failures += !ok;
}
static std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> decoded;
    if (rdna2_walk(code.data(), code.size(), decoded) != code.size()) return {};
    return decoded;
}
static std::vector<uint32_t> program(bool wide8, uint32_t immediate, bool overwrite,
                                     uint32_t export_word) {
    const uint32_t final_parent = wide8 ? 31u : 27u;
    std::vector<uint32_t> code{wide8 ? 0xf40c0600u : 0xf4080600u,
                              0xfa000000u | immediate};
    if (overwrite) code.insert(code.end(), {0xbe800380u, 0xbe810380u});
    code.insert(code.end(), {0x8f6b8400u | final_parent, // VCC_HI = final parent word << 4
        0x876bff6bu, 0x1f0u,
        wide8 ? 0xf40c0201u : 0xf4080201u, 0xd6000000u,
        wide8 ? 0x7e00020fu : 0x7e00020bu,
        export_word, 0u, 0xbf810000u});
    return code;
}

TEST(RawWideReadPoint, Contract) {
    for (bool pixel : {false, true}) {
        const char* stage = pixel ? "PS" : "VS";
        const uint32_t export_word = pixel ? 0xf800180fu : 0xf80008cfu;
        for (bool wide8 : {false, true}) {
            const uint32_t bytes = wide8 ? 32u : 16u;
            for (uint32_t immediate : {0u, 4u, 20u}) {
                const auto code = program(wide8, immediate, true, export_word);
                const auto decoded = decode(code);
                std::vector<uint32_t> sources;
                check(decoded.size() == 9u && decoded.front().literal == immediate &&
                      decoded.front().dst.value == 24 &&
                      decoded.front().opcode == (wide8 ? 3u : 2u), stage,
                      "complete parent/child ISA decode preserves width and nonzero immediate");
                check(rdna2_owned_raw_wide_data_loads(decoded) == std::vector<uint32_t>{0u} &&
                      rdna2_proven_raw_register_wide_data_loads(decoded, &sources) ==
                          std::vector<uint32_t>{7u} && sources == std::vector<uint32_t>{0u},
                      stage, "last parent word selects a proven child after source-pointer overwrite");
                auto before = code;
                before.insert(before.begin(), 0xbe800380u);
                check(rdna2_owned_raw_wide_data_loads(decode(before)).empty() &&
                      rdna2_proven_raw_register_wide_data_loads(decode(before)).empty(),
                      stage, "pointer overwrite before parent read refuses owned and child admission");
                auto write_after = decoded;
                // An actual store-shaped instruction after the reads still excludes this first
                // owned subset. The unchanged legacy immediate proof is deliberately separate.
                auto store = write_after.back();
                store.fmt = Rdna2Format::FLAT;
                store.is_end = false;
                write_after.insert(write_after.end() - 1, store);
                write_after[write_after.size() - 2].pc = write_after.back().pc;
                write_after.back().pc += store.len_dwords;
                check(rdna2_owned_raw_wide_data_loads(write_after).empty() &&
                      rdna2_proven_raw_register_wide_data_loads(write_after).empty(), stage,
                      "guest write after the read still refuses the new owned-parent subset");
            }
            const auto strict_parent = decode(program(wide8, 4u, false, export_word));
            std::vector<uint32_t> sources;
            check(rdna2_proven_raw_immediate_wide_data_loads(strict_parent) ==
                      std::vector<uint32_t>{0u} &&
                  rdna2_owned_raw_wide_data_loads(strict_parent) == std::vector<uint32_t>{0u} &&
                  rdna2_proven_raw_register_wide_data_loads(strict_parent, &sources) ==
                      std::vector<uint32_t>{5u}, stage,
                  "a strict immediate parent owns its selector observation for Route A");

            std::array<uint8_t, 32> storage{};
            ShaderResource source;
            source.cls = ResourceClass::ConstantBuffer;
            source.format = DataFormat::Uint32;
            source.num_components = 1u;
            source.gpu_addr = 0x2000000000ull;
            source.size = bytes;
            source.fetch_pc = 0u;
            source.host_data = storage.data();
            source.host_data_size = bytes;
            source.owned_raw_snapshot_bytes = bytes;
            ShaderResourceTable table;
            table.resources.push_back(source);
            check(owned_raw_snapshot_at(table, 0u, bytes) != nullptr, stage,
                  "complete exact-width owned view is admitted at its read PC");
            for (uint32_t wrong : {wide8 ? 16u : 32u, bytes - 4u}) {
                auto malformed = table;
                malformed.resources[0].size = wrong;
                malformed.resources[0].host_data_size = wrong;
                malformed.resources[0].owned_raw_snapshot_bytes = wrong;
                check(!owned_raw_snapshot_at(malformed, 0u, bytes), stage,
                      "wrong exact decoded width cannot substitute a complete alternative view");
            }
            auto short_host = table;
            short_host.resources[0].host_data_size = bytes - 4u;
            check(!owned_raw_snapshot_at(short_host, 0u, bytes), stage,
                  "truncating the highest word refuses hosted admission");
            auto poisoned = table;
            poisoned.resources[0].format = DataFormat::Float32;
            poisoned.resources.push_back(source);
            check(!owned_raw_snapshot_at(poisoned, 0u, bytes), stage,
                  "valid later duplicate cannot conceal poisoned first exact-PC binding");
            auto duplicate = table;
            duplicate.resources.push_back(source);
            check(!owned_raw_snapshot_at(duplicate, 0u, bytes), stage,
                  "two valid exact-PC entries remain ambiguous and refuse admission");
            auto prefix = table;
            prefix.resources[0].host_data_prefix_bytes = 4u;
            check(!owned_raw_snapshot_at(prefix, 0u, bytes), stage,
                  "effective hosted source cannot invent an immediate-index prefix");
            auto overflow = table;
            overflow.resources[0].gpu_addr = UINT64_MAX - bytes + 4u;
            check(!owned_raw_snapshot_at(overflow, 0u, bytes), stage,
                  "effective hosted source range overflow refuses admission");
        }
    }
    EXPECT_EQ(failures, 0);
}
