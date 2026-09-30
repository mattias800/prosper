// Numeric raw register-offset loads need exact backing in both graphics stages. Descriptor-only
// assembly remains the independently resolved consumer route. No Vulkan device is created here.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper::gpu;
static int failures = 0;
static void check(bool ok, const char* stage, const char* label) {
    std::printf("[%s] %s: %s\n", ok ? "ok" : "FAIL", stage, label);
    failures += !ok;
}

static std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> result;
    rdna2_walk(code.data(), code.size(), result);
    return result;
}

static std::vector<uint32_t> compile(ShaderProgramStage stage,
                                    const std::vector<uint32_t>& code,
                                    const ShaderResourceTable& table) {
    return stage == ShaderProgramStage::Vertex
        ? recompile_vertex(code.data(), code.size(), &table)
        : recompile_fragment(code.data(), code.size(), &table);
}

int main() {
    alignas(16) std::array<uint32_t, 128> words{};
    for (size_t i = 0; i < words.size(); ++i) words[i] = 100u + static_cast<uint32_t>(i);
    const uint64_t address = reinterpret_cast<uint64_t>(words.data());
    std::array<uint32_t, 3> user{static_cast<uint32_t>(address),
        static_cast<uint32_t>(address >> 32u), 2u};
    for (auto stage : {ShaderProgramStage::Vertex, ShaderProgramStage::Fragment}) {
        const char* name = stage == ShaderProgramStage::Vertex ? "VS" : "PS";
        const uint32_t export_word = stage == ShaderProgramStage::Vertex
            ? 0xf80008cfu : 0xf800180fu;
        for (bool wide8 : {false, true}) {
            const std::vector<uint32_t> code{
                0x8f148402u, // s_lshl_b32 s20,entry s2,4
                wide8 ? 0xf40c0200u : 0xf4080200u, 0x28000010u,
                wide8 ? 0x7e00020fu : 0x7e00020bu, // v0 = final loaded word
                export_word, 0x00000000u, 0xbf810000u};
            ShaderResourceTable table;
            add_compute_buffer_resources(table, code.data(), code.size(),
                                         user.data(), user.size());
            assign_convention_bindings(table, 2u);
            const ShaderResource* source = table.by_fetch_pc(1u);
            check(rdna2_raw_wide_data_loads(decode(code)) == std::vector<uint32_t>{1u} &&
                  rdna2_proven_raw_register_wide_data_loads(decode(code)) ==
                      std::vector<uint32_t>{1u}, name,
                  wide8 ? "x8 numeric observer and code-side backing proof agree"
                        : "x4 numeric observer and code-side backing proof agree");
            check(source && valid_raw_register_snapshot_resource(*source) &&
                  source->gpu_addr == address + 48u && source->size == (wide8 ? 32u : 16u),
                  name, "fold produces the complete effective exact-PC range");
            if (!source) continue;
            check(!compile(stage, code, table).empty(), name,
                  wide8 ? "backed x8 graphics module compiles" : "backed x4 graphics module compiles");
            clear_shader_recompile_cache();
            uint64_t valid_identity = 0;
            const auto valid = recompile_graphics_shader_cached(stage, code.data(), code.size(),
                &table, nullptr, nullptr, &valid_identity);
            check(!valid.empty(), name, "valid exact-PC module is admitted to the graphics cache");
            const auto refuse = [&](ShaderResourceTable malformed, const char* label) {
                const bool cold = compile(stage, code, malformed).empty();
                uint64_t invalid_identity = 0, restored_identity = 0;
                const bool warm = recompile_graphics_shader_cached(stage, code.data(), code.size(),
                    &malformed, nullptr, nullptr, &invalid_identity).empty();
                const bool restored = !recompile_graphics_shader_cached(stage, code.data(),
                    code.size(), &table, nullptr, nullptr, &restored_identity).empty() &&
                    restored_identity == valid_identity;
                std::printf("[admission] %s x%u %s: cold=%d warm=%d restored=%d\n",
                    name, wide8 ? 8u : 4u, label, cold, warm, restored);
                check(cold && warm && restored, name, label);
            };
            ShaderResourceTable absent;
            refuse(absent, "absent exact-PC backing refuses cold/warm then valid recovers");
            auto unrelated = table;
            unrelated.resources.front().fetch_pc = UINT32_MAX;
            unrelated.resources.front().sgpr_base = 0u;
            unrelated.resources.front().raw_register_snapshot = false;
            refuse(unrelated, "unrelated conventional binding2 cannot authorize numeric bytes");
            auto unmarked = table;
            unmarked.resources.front().raw_register_snapshot = false;
            refuse(unmarked, "unmarked source cannot borrow cached admission");
            auto short_range = table;
            short_range.resources.front().size -= sizeof(uint32_t);
            refuse(short_range, "short source range cannot borrow cached admission");
            auto truncated = table;
            truncated.resources.front().host_data = reinterpret_cast<uint8_t*>(words.data());
            truncated.resources.front().host_data_size = source->size - 1u;
            refuse(truncated, "truncated hosted bytes cannot borrow cached admission");
            auto wrong_pc = table;
            wrong_pc.resources.front().fetch_pc = 0u;
            refuse(wrong_pc, "a different fetch PC cannot authorize numeric bytes");
            auto unaligned = table;
            ++unaligned.resources.front().gpu_addr;
            refuse(unaligned, "unaligned source cannot borrow cached admission");
            auto low = table;
            low.resources.front().gpu_addr = 0u;
            refuse(low, "null logical source cannot borrow cached admission");
            auto overflow = table;
            overflow.resources.front().gpu_addr = UINT64_MAX - 3u;
            refuse(overflow, "overflowing logical source cannot borrow cached admission");
            auto hosted = table;
            hosted.resources.front().host_data = reinterpret_cast<uint8_t*>(words.data()) + 48u;
            hosted.resources.front().host_data_size = source->size;
            hosted.resources.front().host_data_prefix_bytes = 48u;
            uint64_t hosted_identity = 0;
            check(!compile(stage, code, hosted).empty() &&
                  !recompile_graphics_shader_cached(stage, code.data(), code.size(), &hosted,
                      nullptr, nullptr, &hosted_identity).empty() &&
                  hosted_identity == valid_identity, name,
                  "an exact hosted subrange inside a merged capture blob remains valid");
            user[2] = 3u;
            ShaderResourceTable shifted;
            add_compute_buffer_resources(shifted, code.data(), code.size(),
                                         user.data(), user.size());
            assign_convention_bindings(shifted, 2u);
            uint64_t shifted_identity = 0;
            check(shifted.by_fetch_pc(1u) && shifted.by_fetch_pc(1u)->gpu_addr == address + 64u &&
                  !recompile_graphics_shader_cached(stage, code.data(), code.size(), &shifted,
                      nullptr, nullptr, &shifted_identity).empty() &&
                  shifted_identity == valid_identity, name,
                  "new legitimate source address reuses code without reusing source selection");
            user[2] = 2u;

            auto firstlane = code;
            firstlane.insert(firstlane.begin(), 0x7e040500u); // v_readfirstlane_b32 s2,v0
            auto marked_unproven = table;
            marked_unproven.resources.front().fetch_pc = 2u;
            check(rdna2_raw_wide_data_loads(decode(firstlane)) == std::vector<uint32_t>{2u} &&
                  rdna2_proven_raw_register_wide_data_loads(decode(firstlane)).empty() &&
                  compile(stage, firstlane, marked_unproven).empty(), name,
                  "a marked range cannot bypass a genuine runtime-selector code-proof refusal");
            auto vcc = code;
            vcc[0] = 0x8f6b8402u; // scalar-defined vcc_hi
            vcc[2] = 0xd6000010u;
            auto vcc_table = table;
            check(!compile(stage, vcc, vcc_table).empty(), name,
                  "explicit scalar VCC address has backed graphics admission");
            auto prepared_vcc = vcc;
            prepared_vcc.insert(prepared_vcc.begin(), 0x7e000280u); // v0 = 0 before scalar VCC
            auto prepared_vcc_table = vcc_table;
            prepared_vcc_table.resources.front().fetch_pc = 2u;
            const auto setup_decoded = decode(prepared_vcc);
            const auto& setup = setup_decoded.front();
            std::printf("[decode-control] %s x%u word=0x7e000280 pc=%u opcode=0x%x "
                        "dst=v%d src0=%d\n", name, wide8 ? 8u : 4u,
                        setup.pc, setup.opcode, setup.dst.value, setup.src[0].value);
            check(setup.opcode == 1u && setup.dst.kind == OperandKind::VGPR &&
                  setup.dst.value == 0 && setup.src[0].kind == OperandKind::InlineInt &&
                  setup.src[0].value == 0 &&
                  rdna2_proven_raw_register_wide_data_loads(setup_decoded) ==
                      std::vector<uint32_t>{2u} &&
                  !compile(stage, prepared_vcc, prepared_vcc_table).empty(), name,
                  "known VGPR setup preserves backed scalar VCC graphics admission");
            auto clobber = prepared_vcc;
            clobber.insert(clobber.begin() + 2, 0x7d880084u); // v_cmp_gt_u32 vcc,4,v0
            auto clobber_table = prepared_vcc_table;
            clobber_table.resources.front().fetch_pc = 3u;
            const auto clobber_decoded = decode(clobber);
            const auto& compare = clobber_decoded.at(2u);
            std::printf("[decode-control] %s x%u word=0x7d880084 pc=%u opcode=0x%x "
                        "src0=%d src1=v%d cmpx=%d\n", name, wide8 ? 8u : 4u,
                        compare.pc, compare.opcode, compare.src[0].value, compare.src[1].value,
                        vopc_is_cmpx(compare.opcode));
            check(compare.opcode == 0xc4u && compare.src[0].kind == OperandKind::InlineInt &&
                  compare.src[0].value == 4 && compare.src[1].kind == OperandKind::VGPR &&
                  compare.src[1].value == 0 && !vopc_is_cmpx(compare.opcode) &&
                  rdna2_proven_raw_register_wide_data_loads(clobber_decoded).empty() &&
                  compile(stage, clobber, clobber_table).empty(), name,
                  "implicit non-EXEC VCC compare cannot borrow a marked graphics range");
        }

        // A raw descriptor bundle may be patched before its exact-PC consumer. Its words never
        // become a numeric scalar result, so refusing all register-offset bundles would be wrong.
        const std::vector<uint32_t> descriptor{
            0xbe940380u, 0xf4080600u, 0x28000000u, 0x8f188118u,
            0xe0302000u, 0x80060404u,
            export_word, 0x00000000u, 0xbf810000u};
        ShaderResourceTable descriptor_table;
        ShaderResource buffer;
        buffer.cls = ResourceClass::VertexBuffer;
        buffer.format = DataFormat::Uint32;
        buffer.num_components = 1u;
        buffer.binding = 4u;
        buffer.fetch_pc = 4u;
        buffer.gpu_addr = address;
        buffer.size = sizeof(words);
        descriptor_table.resources.push_back(buffer);
        check(rdna2_raw_wide_data_loads(decode(descriptor)).empty() &&
              !compile(stage, descriptor, descriptor_table).empty(), name,
              "descriptor-only raw bundle still resolves its exact MUBUF consumer");
        auto numeric_descriptor = descriptor;
        numeric_descriptor.insert(numeric_descriptor.begin() + 4, 0x7e000218u);
        auto numeric_table = descriptor_table;
        numeric_table.resources.front().fetch_pc = 5u;
        check(rdna2_raw_wide_data_loads(decode(numeric_descriptor)) == std::vector<uint32_t>{1u} &&
              compile(stage, numeric_descriptor, numeric_table).empty(), name,
              "adding one numeric observer requires raw backing beside the descriptor resource");
        auto missing_consumer = descriptor_table;
        missing_consumer.resources.front().fetch_pc = UINT32_MAX;
        check(compile(stage, descriptor, missing_consumer).empty(), name,
              "missing exact descriptor consumer refuses independently of raw numeric policy");
    }
    std::printf("graphics raw register-wide failures: %d\n", failures);
    return failures != 0;
}
