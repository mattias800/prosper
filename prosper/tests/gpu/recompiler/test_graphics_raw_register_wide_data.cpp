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
        // A standalone relaxed read point has no Route A dependency or old strict-PC tag.
        // Its exact owned binding cannot become the conventional fallback for another load.
        const std::vector<uint32_t> owned_only{
            0xf4080600u, 0xfa000004u, 0xbe800380u, 0xbe810380u,
            0x7e00021bu, export_word, 0u, 0xbf810000u};
        ShaderResourceTable owned_table;
        ShaderResource owned_source;
        owned_source.cls = ResourceClass::ConstantBuffer;
        owned_source.format = DataFormat::Uint32;
        owned_source.num_components = 1u;
        owned_source.binding = 2u;
        owned_source.fetch_pc = 0u;
        owned_source.gpu_addr = address + 4u;
        owned_source.size = 16u;
        owned_source.host_data = reinterpret_cast<uint8_t*>(words.data() + 1u);
        owned_source.host_data_size = 16u;
        owned_source.owned_raw_snapshot_bytes = 16u;
        owned_table.resources.push_back(owned_source);
        check(rdna2_owned_raw_wide_data_loads(decode(owned_only)) ==
                  std::vector<uint32_t>{0u} && !compile(stage, owned_only, owned_table).empty(),
              name, "standalone nonzero-immediate read executes from its complete owned binding");
        auto unrelated_owned = owned_only;
        unrelated_owned.insert(unrelated_owned.begin() + 5,
            {0xf4000a01u, 0xfa000000u, 0x7e020228u}); // unrelated x1 s40,entry2:3; v1=s40
        check(compile(stage, unrelated_owned, owned_table).empty(), name,
              "owned-only binding2 cannot authorize an unrelated unresolved raw scalar load");
        if (stage == ShaderProgramStage::Vertex) {
            auto compute_owned = owned_only;
            compute_owned.erase(compute_owned.end() - 3, compute_owned.end() - 1);
            auto compute_unrelated = unrelated_owned;
            compute_unrelated.erase(compute_unrelated.end() - 3, compute_unrelated.end() - 1);
            check(!recompile_valu(compute_owned.data(), compute_owned.size(), 1u, 0u,
                                  &owned_table).empty() &&
                  recompile_valu(compute_unrelated.data(), compute_unrelated.size(), 2u, 0u,
                                 &owned_table).empty(), "compute",
                  "complete owned numeric source does not supply an unrelated binding2 fallback");
            const std::vector<uint32_t> prolog{0xbfa00003u, 0xbe802006u};
            const std::vector<uint32_t> legacy_main{
                0x7e000280u, export_word, 0u, 0xbf810000u};
            ShaderResourceTable legacy_table;
            uint64_t legacy_identity = 0u;
            const auto legacy_chain = recompile_vertex_chain_cached_shared(
                prolog.data(), prolog.size(), legacy_main.data(), legacy_main.size(),
                &legacy_table, nullptr, &legacy_identity);
            check(!rdna2_vertex_chain_has_owned_raw_wide_inputs(
                      prolog.data(), prolog.size(), legacy_main.data(), legacy_main.size(),
                      &legacy_table) && legacy_chain && !legacy_chain->empty() &&
                  !recompile_vertex_chain(prolog.data(), prolog.size(), legacy_main.data(),
                                          legacy_main.size(), &legacy_table).empty(), name,
                  "a real unowned non-passthrough vertex chain remains valid");
            auto requirement_only = legacy_table;
            requirement_only.owned_raw_snapshot_requirements.emplace_back(0u, 16u);
            uint64_t requirement_identity = 99u;
            const auto requirement_warm = recompile_vertex_chain_cached_shared(
                prolog.data(), prolog.size(), legacy_main.data(), legacy_main.size(),
                &requirement_only, nullptr, &requirement_identity);
            check(!requirement_warm && requirement_identity == 0u &&
                  recompile_vertex_chain(prolog.data(), prolog.size(), legacy_main.data(),
                      legacy_main.size(), &requirement_only).empty(), name,
                  "same-code warm chain refuses a requirement omitted from its ordinary cache key");
            auto ordinary_table = legacy_table;
            auto ordinary_source = owned_source;
            ordinary_source.fetch_pc = UINT32_MAX;
            ordinary_source.owned_raw_snapshot_bytes = 0u;
            ordinary_table.resources.push_back(ordinary_source);
            uint64_t ordinary_identity = 0u, marked_identity = 99u;
            const auto ordinary_chain = recompile_vertex_chain_cached_shared(
                prolog.data(), prolog.size(), legacy_main.data(), legacy_main.size(),
                &ordinary_table, nullptr, &ordinary_identity);
            auto marker_only = ordinary_table;
            marker_only.resources.front().owned_raw_snapshot_bytes = 16u;
            check(ordinary_chain && !ordinary_chain->empty() && !recompile_vertex_chain_cached_shared(
                    prolog.data(), prolog.size(), legacy_main.data(), legacy_main.size(),
                    &marker_only, nullptr, &marked_identity) && marked_identity == 0u, name,
                  "same-code warm chain refuses a marker omitted from its ordinary cache key");
            // Both allocations stay fixed while their active bytes change. A proof cached by
            // address instead of byte-validated analysis versions would borrow the legacy verdict.
            std::array<uint32_t, 32> mutable_main{}, mutable_prolog{};
            std::copy(legacy_main.begin(), legacy_main.end(), mutable_main.begin());
            std::copy(prolog.begin(), prolog.end(), mutable_prolog.begin());
            uint64_t version_identity = 0u;
            const auto version_legacy = recompile_vertex_chain_cached_shared(
                mutable_prolog.data(), prolog.size(), mutable_main.data(), legacy_main.size(),
                &legacy_table, nullptr, &version_identity);
            check(version_legacy && !version_legacy->empty(), name,
                  "fixed-address chain establishes a valid unowned code version");
            std::copy(owned_only.begin(), owned_only.end(), mutable_main.begin());
            uint64_t changed_identity = 99u;
            check(!recompile_vertex_chain_cached_shared(mutable_prolog.data(), prolog.size(),
                      mutable_main.data(), owned_only.size(), &legacy_table, nullptr,
                      &changed_identity) && changed_identity == 0u, name,
                  "same-address changed main cannot borrow an unowned cached code proof");
            std::copy(legacy_main.begin(), legacy_main.end(), mutable_main.begin());
            uint64_t recovered_version = 0u;
            check(recompile_vertex_chain_cached_shared(mutable_prolog.data(), prolog.size(),
                      mutable_main.data(), legacy_main.size(), &legacy_table, nullptr,
                      &recovered_version) == version_legacy && recovered_version == version_identity,
                  name, "restored main code recovers the original shared chain module");
            const std::array<uint32_t, 5> owned_prolog{
                owned_only[0], owned_only[1], owned_only[2], owned_only[3], 0xbe802006u};
            const std::vector<uint32_t> linked_observer_main{
                owned_only[4], export_word, 0u, 0xbf810000u};
            std::copy(linked_observer_main.begin(), linked_observer_main.end(), mutable_main.begin());
            uint64_t linked_legacy_identity = 0u;
            const auto linked_legacy = recompile_vertex_chain_cached_shared(mutable_prolog.data(),
                prolog.size(), mutable_main.data(), linked_observer_main.size(),
                &legacy_table, nullptr, &linked_legacy_identity);
            std::copy(owned_prolog.begin(), owned_prolog.end(), mutable_prolog.begin());
            const auto prolog_decode = decode(std::vector<uint32_t>(owned_prolog.begin(), owned_prolog.end()));
            auto linked_code = std::vector<uint32_t>(owned_prolog.begin(), owned_prolog.end() - 1);
            linked_code.insert(linked_code.end(), linked_observer_main.begin(), linked_observer_main.end());
            const bool linked_only_owned = rdna2_owned_raw_wide_data_loads(prolog_decode).empty() &&
                rdna2_owned_raw_wide_data_loads(decode(linked_observer_main)).empty() &&
                rdna2_owned_raw_wide_data_loads(decode(linked_code)) == std::vector<uint32_t>{0u};
            changed_identity = 99u;
            check(linked_legacy && !linked_legacy->empty() && linked_only_owned &&
                  !recompile_vertex_chain_cached_shared(mutable_prolog.data(), owned_prolog.size(),
                      mutable_main.data(), linked_observer_main.size(), &legacy_table,
                      nullptr, &changed_identity) && changed_identity == 0u, name,
                  "same-address changed prolog checks ownership arising only in the linked stream");
            const auto chain_refuses = [&](const ShaderResourceTable& supplied, const char* label) {
                uint64_t refused_identity = 77u, recovered_identity = 0u;
                const bool cold = recompile_vertex_chain(prolog.data(), prolog.size(),
                    owned_only.data(), owned_only.size(), &supplied).empty();
                const bool warm = !recompile_vertex_chain_cached_shared(
                    prolog.data(), prolog.size(), owned_only.data(), owned_only.size(),
                    &supplied, nullptr, &refused_identity) && refused_identity == 0u;
                const auto recovered = recompile_vertex_chain_cached_shared(
                    prolog.data(), prolog.size(), legacy_main.data(), legacy_main.size(),
                    &legacy_table, nullptr, &recovered_identity);
                check(cold && warm && recovered == legacy_chain &&
                      recovered_identity == legacy_identity, name, label);
            };
            chain_refuses(owned_table, "complete owned main refuses direct and warm chained admission");
            auto marker_free = owned_table;
            marker_free.resources.front().owned_raw_snapshot_bytes = 0u;
            chain_refuses(marker_free, "marker-free owned main is still a code-derived chain refusal");
            chain_refuses(legacy_table, "missing whole owned chain table cannot borrow warm legacy code");
            auto short_chain = owned_table;
            short_chain.resources.front().host_data_size = 15u;
            chain_refuses(short_chain, "truncated owned main cannot borrow warm legacy chain code");
            auto owned_prolog = owned_only;
            owned_prolog.erase(owned_prolog.end() - 3, owned_prolog.end());
            owned_prolog.push_back(0xbe802006u);
            check(rdna2_vertex_chain_has_owned_raw_wide_inputs(
                      owned_prolog.data(), owned_prolog.size(), legacy_main.data(), legacy_main.size(),
                      &legacy_table) &&
                  recompile_vertex_chain(owned_prolog.data(), owned_prolog.size(),
                      legacy_main.data(), legacy_main.size(), &legacy_table).empty(), name,
                  "linked prolog prefix requires owned authority even without any resource marker");
        } else {
            // The selector dispatch is the production-recognized bounded idiom. Selecting A
            // exposes an owned read that original-stream proof cannot admit across s_setpc.
            // Selecting B remains an ordinary legacy specialization with a real nonempty module.
            std::vector<uint32_t> dispatch{
                0xf4201a8cu, 0xfa000010u, 0x816ac16au, 0x83ea826au, 0x8f6a836au,
                0xbea01f00u, 0x802020ffu, 80u, 0x82212180u, 0xf4040890u, 0xd4000000u,
                0xbea81f00u, 0x80282228u, 0x82292329u, 0xbe802028u,
                0xf4080600u, 0xfa000004u, 0xbe800380u, 0xbe810380u, 0x7e00021bu,
                0xbf820001u, 0x7e000280u, export_word, 0u, 0xbf810000u, 0u,
                12u, 0u, 36u, 0u, 40u, 0u};
            const auto info = rdna2_pcrel_dispatch_info(dispatch.data(), dispatch.size());
            auto selected = decode(dispatch);
            const bool selected_ok = rdna2_specialize_pcrel_dispatch(selected, info, 15u);
            auto dispatch_table = owned_table;
            dispatch_table.resources.front().fetch_pc = 15u;
            std::array<uint32_t, 5> selector{};
            ShaderResource selector_resource;
            selector_resource.cls = ResourceClass::ConstantBuffer;
            selector_resource.binding = 0u;
            selector_resource.sgpr_base = 24u;
            selector_resource.size = sizeof(selector);
            selector_resource.host_data = reinterpret_cast<uint8_t*>(selector.data());
            selector_resource.host_data_size = sizeof(selector);
            dispatch_table.resources.push_back(selector_resource);
            check(info.valid && selected_ok && rdna2_owned_raw_wide_data_loads(decode(dispatch)).empty() &&
                  rdna2_owned_raw_wide_data_loads(selected) == std::vector<uint32_t>{15u}, name,
                  "selected body would introduce an owned PC absent from original code authority");
            check(!recompile_fragment(dispatch.data(), dispatch.size(), &dispatch_table,
                                      nullptr, 21u).empty(), name,
                  "legacy dispatch specialization retains a nonempty fragment positive");
            const auto selected_refuses = [&](ShaderResourceTable supplied, const char* label) {
                const bool cold = recompile_fragment(dispatch.data(), dispatch.size(), &supplied,
                                                      nullptr, 15u).empty();
                selector[4] = 2u; // B / pc21: warm a valid legacy specialized module first.
                const auto valid = recompile_graphics_shader_cached(
                    stage, dispatch.data(), dispatch.size(), &supplied);
                selector[4] = 1u; // A / pc15: absent original authority must remain refusal.
                const auto refused = recompile_graphics_shader_cached(
                    stage, dispatch.data(), dispatch.size(), &supplied);
                selector[4] = 2u;
                const auto recovered = recompile_graphics_shader_cached(
                    stage, dispatch.data(), dispatch.size(), &supplied);
                check(cold && !valid.empty() && refused.empty() && recovered == valid, name, label);
            };
            selected_refuses(dispatch_table, "specialized owned source cannot borrow valid legacy warm admission");
            auto short_selected = dispatch_table;
            short_selected.resources.front().host_data_size = 15u;
            selected_refuses(short_selected, "truncated selected source cannot borrow valid legacy warm admission");
        }
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
