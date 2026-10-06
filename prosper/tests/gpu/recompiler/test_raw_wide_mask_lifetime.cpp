// A fresh emitted Bool mask and its possible physical high-word data are distinct lifetimes.
// No device is created. Synthetic ISA packets exercise the actual decoder/classifier/emitter.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
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

// Match the existing image_sample_dref_manual_2d nearest path, which emits
// OpImageSampleExplicitLod (88) followed by its manual depth comparison.
static bool has_explicit_sample(const std::vector<uint32_t>& module) {
    constexpr uint32_t OpImageSampleExplicitLod = 88u;
    if (module.size() < 5 || module[0] != 0x07230203u) return false;
    bool sampled = false;
    for (size_t at = 5; at < module.size();) {
        const uint32_t words = module[at] >> 16u;
        if (!words || words > module.size() - at) return false;
        sampled |= (module[at] & 0xffffu) == OpImageSampleExplicitLod && words >= 7u;
        at += words;
    }
    return sampled;
}

static std::array<uint32_t, 2> compare(uint32_t root, uint32_t source = 128u,
                                       uint32_t opcode = 2u) {
    // v_cmp_eq_f32_sdwa s[root:root+1], 0, v0; DWORD source selects, no modifiers.
    return {0x7c0000f9u | (opcode << 17u), 0x06868000u | (root << 8u) | source};
}

static void append(std::vector<uint32_t>& code, std::array<uint32_t, 2> packet) {
    code.insert(code.end(), packet.begin(), packet.end());
}

static std::vector<uint32_t> shader(uint32_t export_word,
                                   const std::vector<uint32_t>& after_fetch,
                                   bool descriptor_patch = true) {
    std::vector<uint32_t> code{0x7e000280u, // v0 = 0, independent compare input
        0xbe940380u,                       // s20 = 0, register SOFFSET
        0xf4080600u, 0x28000000u};         // raw x4 s24:27 from entry s12:13 / s20
    if (descriptor_patch) code.push_back(0x8f188118u); // descriptor assembly only
    code.insert(code.end(), {0xe0302000u, 0x80060404u}); // exact MUBUF uses s24:27
    code.insert(code.end(), after_fetch.begin(), after_fetch.end());
    code.insert(code.end(), {export_word, 0u, 0xbf810000u});
    return code;
}

static std::array<uint32_t, 2> select(uint32_t root = 24u) {
    // v_cndmask_b32_e64 v0, 0, 1.0, s[root:root+1]. Only src2 is a Bool consumer.
    return {0xd5010000u, 128u | (242u << 9u) | (root << 18u)};
}

int main(int argc, char** argv) {
    std::filesystem::path module_dir;
    if (argc != 1) {
        if (argc != 3 || std::string(argv[1]) != "--dump-mimg-readonly-modules") {
            std::fprintf(stderr, "usage: %s [--dump-mimg-readonly-modules NEW_DIR]\n", argv[0]);
            return 2;
        }
        module_dir = argv[2];
        std::error_code error;
        if (!std::filesystem::create_directory(module_dir, error)) {
            std::fprintf(stderr, "module output directory must be new: %s (%s)\n",
                argv[2], error.message().c_str());
            return 2;
        }
    }
    alignas(16) std::array<uint32_t, 128> data{};
    uint64_t last_program = 0x1ee0000000ull;
    for (auto stage : {RecompileDiagnosticStage::Vertex, RecompileDiagnosticStage::Fragment}) {
        const char* name = stage == RecompileDiagnosticStage::Vertex ? "VS" : "PS";
        const uint32_t export_word = stage == RecompileDiagnosticStage::Vertex
            ? 0xf80008cfu : 0xf800180fu;
        const auto table_for = [&](const std::vector<uint32_t>& code) {
            ShaderResourceTable table;
            ShaderResource buffer;
            buffer.cls = ResourceClass::VertexBuffer;
            buffer.format = DataFormat::Uint32;
            buffer.num_components = 1u;
            buffer.binding = 4u;
            for (const auto& instruction : decode(code))
                if (instruction.fmt == Rdna2Format::MUBUF) buffer.fetch_pc = instruction.pc;
            buffer.gpu_addr = reinterpret_cast<uint64_t>(data.data());
            buffer.size = sizeof(data);
            table.resources.push_back(buffer);
            return table;
        };
        const auto compile = [&](const std::vector<uint32_t>& code,
                                 const ShaderResourceTable& table) {
            const uint64_t program = ++last_program;
            return stage == RecompileDiagnosticStage::Vertex
                ? recompile_vertex(code.data(), code.size(), &table, nullptr, false, 0,
                    {RecompileDiagnosticStage::Vertex, program})
                : recompile_fragment(code.data(), code.size(), &table, nullptr, UINT32_MAX,
                    nullptr, false, {RecompileDiagnosticStage::Fragment, program});
        };
        const auto numeric = [&](const std::vector<uint32_t>& code, const char* label) {
            const auto table = table_for(code);
            const bool classified = rdna2_raw_wide_data_loads(decode(code)) ==
                std::vector<uint32_t>{2u};
            const bool refused = compile(code, table).empty();
            const auto reason = last_terminal_reject_reason(last_program);
            std::printf("[refusal] %s %s classified=%d rejected=%d reason=%s\n",
                name, label, classified, refused, reason.c_str());
            check(classified && refused && reason.find("pc=2 ") != std::string::npos &&
                  reason.find("op=0x2") != std::string::npos, name, label);
        };
        // A read-only comparison sample elsewhere cannot turn descriptor assembly into a
        // numeric read or aliasing guest write. The independent image uses an entry T#/S#;
        // the raw quad is consumed only through its exact MUBUF resource below.
        const auto image_packet = [](uint32_t opcode) {
            return std::array<uint32_t, 2>{0xf0000108u | (opcode << 18u),
                (16u << 16u) | (18u << 21u) | (4u << 8u)};
        };
        const auto sample_shader = [&](uint32_t opcode) {
            std::vector<uint32_t> sample{0x7e020280u, 0x7e040280u}; // independent v1/v2 coords
            append(sample, image_packet(opcode));
            return shader(export_word, sample);
        };
        const auto sample_table_for = [&](const std::vector<uint32_t>& code) {
            auto table = table_for(code);
            ShaderResource image;
            image.cls = ResourceClass::Texture;
            image.format = DataFormat::Float32;
            image.num_components = 1u;
            image.binding = 6u;
            image.sgpr_base = 64u;
            image.sampler_sgpr_base = 72u;
            image.width = 1u;
            image.height = 1u;
            image.depth = 1u;
            image.depth_compare = true;
            image.depth_compare_func = 3u;
            image.mag_filter = 0u; // explicit nearest path emits OpImageSampleExplicitLod
            image.min_filter = 0u;
            image.gpu_addr = reinterpret_cast<uint64_t>(data.data());
            image.size = sizeof(uint32_t);
            for (const auto& instruction : decode(code))
                if (instruction.fmt == Rdna2Format::MIMG) image.fetch_pc = instruction.pc;
            table.resources.push_back(image);
            return table;
        };
        const auto sample_code = sample_shader(0x2fu);
        const auto sample_ins = decode(sample_code);
        const auto sample_it = std::find_if(sample_ins.begin(), sample_ins.end(),
            [](const auto& in) { return in.fmt == Rdna2Format::MIMG; });
        check(sample_it != sample_ins.end() && sample_it->opcode == 0x2fu &&
              sample_it->len_dwords == 2u && sample_it->mimg_dim == 1u &&
              sample_it->mimg_dmask == 1u && sample_it->src[1].kind == OperandKind::SGPR &&
              sample_it->src[1].value == 64 && sample_it->src[2].kind == OperandKind::SGPR &&
              sample_it->src[2].value == 72,
              name, "comparison sample decodes independent entry image and sampler descriptors");
        const auto sample_module = compile(sample_code, sample_table_for(sample_code));
        check(rdna2_raw_wide_data_loads(sample_ins).empty() &&
              has_explicit_sample(sample_module), name,
              "descriptor assembly beside a read-only comparison sample remains a nonempty module");
        if (!module_dir.empty()) {
            const auto path = module_dir / (std::string(name) + "_image_sample_c_lz.spv");
            std::ofstream output(path, std::ios::binary);
            output.write(reinterpret_cast<const char*>(sample_module.data()),
                static_cast<std::streamsize>(sample_module.size() * sizeof(uint32_t)));
            output.close();
            if (sample_module.empty() || !output) {
                std::fprintf(stderr, "failed to write nonempty module: %s\n", path.string().c_str());
                return 2;
            }
            std::printf("[module] %s words=%zu path=%s\n", name,
                sample_module.size(), path.string().c_str());
        }
        const auto sample_refused = [&](const std::vector<uint32_t>& code,
                                        const char* label, uint32_t raw_opcode = 2u) {
            const auto classified = rdna2_raw_wide_data_loads(decode(code));
            const bool refused = compile(code, sample_table_for(code)).empty();
            const auto reason = last_terminal_reject_reason(last_program);
            check(classified == std::vector<uint32_t>{2u} && refused &&
                  reason.find("pc=2 ") != std::string::npos &&
                  reason.find(raw_opcode == 2u ? "op=0x2" : "op=0x3") != std::string::npos,
                  name, label);
        };
        auto sampled_numeric = sample_code;
        sampled_numeric.insert(sampled_numeric.end() - 3, 0x7e00021bu); // v0 = physical s27
        sample_refused(sampled_numeric,
              "comparison sample preserves genuine x4 highest-word backing refusal");
        for (uint32_t opcode : {0x08u, 0x0fu, 0x11u, 0x7fu}) {
            sample_refused(sample_shader(opcode), opcode == 0x08u
                ? "image store preserves descriptor alias backing refusal"
                : opcode == 0x0fu
                ? "image atomic opcode0f preserves descriptor alias backing refusal"
                : opcode == 0x11u
                ? "image atomic opcode11 preserves descriptor alias backing refusal"
                : "unknown image opcode preserves conservative descriptor backing refusal");
        }
        auto sampled_x8_numeric = sample_code;
        sampled_x8_numeric[2] = 0xf40c0600u; // same raw source/offset, x8 s24:31
        sampled_x8_numeric.insert(sampled_x8_numeric.end() - 3, 0x7e00021fu); // v0 = physical s31
        sample_refused(sampled_x8_numeric,
              "comparison sample preserves genuine x8 highest-word backing refusal", 3u);

        std::vector<uint32_t> tail;
        append(tail, compare(24));
        append(tail, select());
        const auto fresh = shader(export_word, tail);
        const auto decoded = decode(fresh);
        const auto& cmp = decoded.at(5);
        const auto& condition = decoded.at(6);
        check(cmp.fmt == Rdna2Format::VOPC && cmp.opcode == 2u &&
              cmp.dst.kind == OperandKind::SGPR && cmp.dst.value == 24 &&
              cmp.src[0].kind == OperandKind::InlineInt && cmp.src[0].value == 0 &&
              cmp.src[1].kind == OperandKind::VGPR && cmp.src[1].value == 0 &&
              !vopc_is_cmpx(cmp.opcode) && !cmp.has_modifier &&
              condition.fmt == Rdna2Format::VOP3 && condition.opcode == 0x101u &&
              condition.src[2].kind == OperandKind::SGPR && condition.src[2].value == 24,
              name, "fresh compare and exact Bool condition decode without width assumptions");
        check(rdna2_raw_wide_data_loads(decoded).empty() &&
              !compile(fresh, table_for(fresh)).empty(), name,
              "descriptor then independent fresh mask remains a nonempty graphics module");

        auto raw_input = tail;
        raw_input[1] = compare(24, 24u)[1];
        numeric(shader(export_word, raw_input), "raw compare input retains numeric backing refusal");
        for (bool patch : {false, true}) {
            auto high = tail;
            high.insert(high.begin() + 2, 0x7e000219u); // v0 = still-possible old s25
            numeric(shader(export_word, high, patch), patch
                ? "outer lifetime gate retains high-word observer after descriptor patch"
                : "outer lifetime gate retains high-word observer without descriptor patch");
        }
        auto cmpx = tail;
        cmpx[0] = compare(24, 128u, 0x12u)[0];
        numeric(shader(export_word, cmpx), "CMPX cannot publish the decoded SDST mask lifetime");

        auto cmpx_then_mask = cmpx;
        cmpx_then_mask.erase(cmpx_then_mask.begin() + 2, cmpx_then_mask.end());
        cmpx_then_mask.insert(cmpx_then_mask.end(), tail.begin(), tail.end());
        cmpx_then_mask.push_back(0xbefe04c1u); // full EXEC before VS position export
        const auto cmpx_fresh = shader(export_word, cmpx_then_mask);
        const auto cmpx_decoded = decode(cmpx_fresh);
        check(cmpx_decoded.at(5).fmt == Rdna2Format::VOPC &&
              cmpx_decoded.at(5).opcode == 0x12u &&
              vopc_is_cmpx(cmpx_decoded.at(5).opcode) &&
              cmpx_decoded.at(5).dst.kind == OperandKind::SGPR &&
              cmpx_decoded.at(5).dst.value == 24 &&
              cmpx_decoded.at(5).src[0].kind == OperandKind::InlineInt &&
              cmpx_decoded.at(5).src[1].kind == OperandKind::VGPR,
              name, "CMPX decodes explicit independent inputs and an unwritten SDST field");
        check(rdna2_raw_wide_data_loads(cmpx_decoded).empty() &&
              !compile(cmpx_fresh, table_for(cmpx_fresh)).empty(), name,
              "independent CMPX retains EXEC independence for a later fresh Bool mask");
        auto cmpx_raw = cmpx_then_mask;
        cmpx_raw[1] = compare(24, 24u, 0x12u)[1];
        numeric(shader(export_word, cmpx_raw),
                "CMPX explicit raw input cannot publish independent EXEC");
        auto cmpx_uncertain_exec = cmpx_then_mask;
        cmpx_uncertain_exec.insert(cmpx_uncertain_exec.begin(), 0xbefe0481u);
        // The narrow proof admits only inline zero/all-ones mask sources. A non-certified
        // incoming EXEC cannot acquire MUST authority merely through an independent CMPX.
        numeric(shader(export_word, cmpx_uncertain_exec),
                "CMPX cannot restore an uncertified incoming EXEC lifetime");
        auto cmpx_high = cmpx_then_mask;
        cmpx_high.insert(cmpx_high.begin() + 4, 0x7e000219u);
        numeric(shader(export_word, cmpx_high),
                "independent CMPX preserves a possible old numeric high-word reader");
        auto cmpx_unknown = cmpx_then_mask;
        cmpx_unknown[0] = compare(24, 128u, 0x97u)[0];
        numeric(shader(export_word, cmpx_unknown),
                "unknown effective CMPX compare cannot grant EXEC independence");
        auto cmpx_join = cmpx_then_mask;
        // One branch bypasses the independent compare replacing the old raw mask. CMPX
        // does not turn its SDST field into that missing definition at the join.
        cmpx_join.insert(cmpx_join.begin() + 2, {0xbf068014u, 0xbf840002u});
        numeric(shader(export_word, cmpx_join),
                "CMPX independence cannot manufacture a joined saved-mask definition");
        auto cmpx_exec_join = cmpx_then_mask;
        cmpx_exec_join.insert(cmpx_exec_join.begin(),
            {0xbf068014u, 0xbf840001u, 0xbefe0481u});
        numeric(shader(export_word, cmpx_exec_join),
                "CMPX retains the MUST meet of unequal incoming EXEC authority");
        for (uint32_t overwrite : {0xbe98031au, 0xbe99031au}) {
            auto expired = tail;
            expired.insert(expired.begin() + 2, overwrite); // root/sibling = old raw s26
            numeric(shader(export_word, expired), overwrite == 0xbe98031au
                ? "root overwrite expires fresh mask before derived scalar condition"
                : "sibling overwrite expires fresh mask before derived scalar condition");
        }
        auto unknown = tail;
        unknown.insert(unknown.begin() + 2, 0x8f188118u); // ordinary shift, not a mask transfer
        numeric(shader(export_word, unknown), "ordinary scalar transfer cannot carry fresh Bool authority");

        // The branch skips only the compare. The join MUST retain no mask proof for that path.
        auto one_arm = tail;
        one_arm.insert(one_arm.begin(), {0xbf068014u, // overwrite SCC from independent s20/0
                                        0xbf840002u}); // s_cbranch_scc0 +2 dwords
        numeric(shader(export_word, one_arm), "one-arm fresh mask does not dominate the joined reader");
        auto loop = tail;
        loop.insert(loop.begin() + 2, 0xbf82fffdu); // back to the independent compare
        check(rdna2_raw_wide_data_loads(decode(shader(export_word, loop))).empty(), name,
              "independent backward mask facts converge without inventing a numeric observer");
        auto loop_reader = tail;
        loop_reader.insert(loop_reader.begin() + 2,
            {0x7e000219u, 0xbf82fffcu}); // real old-high-word reader, then back to compare
        numeric(shader(export_word, loop_reader),
                "converged loop retains a real numeric high-word reader");
        auto descriptor_loop = shader(export_word,
            {0xbea80382u,                  // s40 = two iterations
             0x81a88128u,                  // s40 -= 1
             0xbf068028u, 0xbf84fffdu,     // loop on independent counter SCC
             0xbe980480u, 0xbe9a0480u,     // overwrite raw words after the loop
             0xbefe04c1u});
        check(rdna2_raw_wide_data_loads(decode(descriptor_loop)).empty() &&
              !compile(descriptor_loop, table_for(descriptor_loop)).empty(), name,
              "descriptor-only finite loop after a once-executed load remains a nonempty module");
        auto loop_exit_reader = descriptor_loop;
        loop_exit_reader.insert(loop_exit_reader.begin() + 11, 0x7e000219u);
        numeric(loop_exit_reader, "loop-carried high-word data remains required at the exit reader");
        std::vector<uint32_t> first_iteration_mask{0xbea80382u};
        append(first_iteration_mask, compare(24));
        append(first_iteration_mask, select());
        first_iteration_mask.insert(first_iteration_mask.end(),
            {0xbe99031au, 0x81a88128u, 0xbf068028u, 0xbf84fffau});
        numeric(shader(export_word, first_iteration_mask),
                "a mask valid on the first iteration expires before the next loop reader");
        auto replayed_load = loop;
        replayed_load[2] = 0xbf82fff8u; // branch pc9 back to the raw load pc2
        numeric(shader(export_word, replayed_load),
                "same raw load reexecution cannot reuse an earlier descriptor observation");

        // Save a real Bool EXEC before the raw load; restoring it later must retain that
        // independent lifetime. The conditional body overwrites all raw words before data use.
        auto saved_exec = shader(export_word,
            {0xbefe0420u,                  // EXEC = saved s32:33
             compare(24, 128u, 0x12u)[0], compare(24, 128u, 0x12u)[1],
             0xbf880002u,                 // EXECZ skips the two complete pair overwrites
             0xbe980480u, 0xbe9a0480u,
             0xbefe04c1u});               // full EXEC before VS position export
        saved_exec.insert(saved_exec.begin() + 2, 0xbea0047eu); // s32:33 = EXEC
        const auto saved_decoded = decode(saved_exec);
        check(saved_decoded.at(2).fmt == Rdna2Format::SOP1 &&
              saved_decoded.at(2).opcode == 4u && saved_decoded.at(2).dst.value == 32 &&
              saved_decoded.at(2).src[0].kind == OperandKind::Special &&
              saved_decoded.at(2).src[0].value == 126 &&
              saved_decoded.at(3).fmt == Rdna2Format::SMEM && saved_decoded.at(3).pc == 3u,
              name, "saved EXEC and later raw load decode as distinct exact definitions");
        check(rdna2_raw_wide_data_loads(saved_decoded).empty() &&
              !compile(saved_exec, table_for(saved_exec)).empty(), name,
              "dominating unchanged pre-load saved EXEC remains a nonempty graphics module");
        const auto saved_numeric = [&](const std::vector<uint32_t>& code, const char* label) {
            const auto instructions = decode(code);
            const auto load = std::find_if(instructions.begin(), instructions.end(),
                [](const auto& in) { return in.fmt == Rdna2Format::SMEM && in.opcode == 2u; });
            const auto classified = rdna2_raw_wide_data_loads(instructions);
            const bool refused = compile(code, table_for(code)).empty();
            const auto reason = last_terminal_reject_reason(last_program);
            const auto pc = load == instructions.end() ? UINT32_MAX : load->pc;
            check(classified == std::vector<uint32_t>{pc} && refused &&
                  reason.find("pc=" + std::to_string(pc) + " ") != std::string::npos &&
                  reason.find("op=0x2") != std::string::npos, name, label);
        };
        for (uint32_t overwrite : {0xbea00319u, 0xbea10319u}) {
            auto expired_saved = saved_exec;
            expired_saved.insert(expired_saved.begin() + 8, overwrite); // root/sibling = raw s25
            saved_numeric(expired_saved, overwrite == 0xbea00319u
                ? "post-load root overwrite expires a pre-load saved EXEC proof"
                : "post-load sibling overwrite expires a pre-load saved EXEC proof");
        }
        auto conditional_save = saved_exec;
        conditional_save.insert(conditional_save.begin() + 2,
            {0xbf068080u, 0xbf840001u}); // one path bypasses the only saved definition
        saved_numeric(conditional_save,
                "pre-load saved EXEC must dominate every reaching path");
        auto unknown_prefix = saved_exec;
        // s_movreld_b32 s40, s40 (SOP1 0x30): writes SGPR[40 + M0], which may be the saved pair.
        // This arm used 0xbea82828 under the same comment until #4529; that word is
        // s_orn2_saveexec_b64 s[40:41], s[40:41], an ordinary mask write that names its
        // destination, and it is the control below.
        unknown_prefix.insert(unknown_prefix.begin() + 3, 0xbea83028u);
        check(decode(unknown_prefix).at(3).opcode == kSop1OpcodeMovreldB32 &&
                  rdna2_raw_wide_data_loads(decode(unknown_prefix)) == std::vector<uint32_t>{4u},
              name, "unknown relative prefix write cannot seed a textual saved EXEC alias");
        auto saveexec_prefix = saved_exec;
        saveexec_prefix.insert(saveexec_prefix.begin() + 3, 0xbea82828u);
        check(decode(saveexec_prefix).at(3).opcode == 0x28u &&
                  rdna2_raw_wide_data_loads(decode(saveexec_prefix)).empty(),
              name, "a saveexec prefix names its destination: the saved EXEC alias survives it");
        auto before_load_expired = saved_exec;
        before_load_expired.insert(before_load_expired.begin() + 3, 0xbea10314u);
        saved_numeric(before_load_expired,
                "pre-load sibling overwrite invalidates the saved EXEC pair");
        auto overlap_save = saved_exec;
        overlap_save[2] = 0xbe98047eu; // save in s24:25, overwritten by the load itself
        overlap_save[8] = 0xbefe0418u;
        saved_numeric(overlap_save,
                "raw load overlap cannot retain the old saved EXEC pair");
        auto saved_high = saved_exec;
        saved_high.insert(saved_high.begin() + 9, 0x7e000219u);
        saved_numeric(saved_high,
                "pre-load saved EXEC does not erase an ordinary loaded high-word reader");

        std::vector<uint32_t> combined;
        append(combined, compare(24));
        append(combined, compare(26));
        combined.push_back(0x8898181au); // s_or_b64 s24:25, s26:27, s24:25
        combined.push_back(0xbea02418u); // s_and_saveexec_b64 s32:33, s24:25
        append(combined, select());
        combined.push_back(0xbefe04c1u); // restore full EXEC before a vertex position export
        const auto saved = shader(export_word, combined);
        check(rdna2_raw_wide_data_loads(decode(saved)).empty() &&
              !compile(saved, table_for(saved)).empty(), name,
              "fresh logical mask and SAVEEXEC preserve independent emitted mask consumers");
        auto scalar_branch = combined;
        scalar_branch.insert(scalar_branch.begin() + 5, 0xbf840001u);
        numeric(shader(export_word, scalar_branch), "mask transfer cannot erase possible raw SCC provenance");
        auto combined_high = combined;
        combined_high.insert(combined_high.begin() + 6, 0x7e000219u);
        numeric(shader(export_word, combined_high), "mask transfer retains its distinct possible numeric high-word view");
        auto vcc_high = tail;
        vcc_high.insert(vcc_high.begin() + 2,
            {0xbeea0418u, 0x7e00026bu}); // vcc=s24:25; numeric v0=VCC_HI
        numeric(shader(export_word, vcc_high),
                "saved mask copied to VCC retains its possible numeric high-word view");

        const auto spill_read = shader(export_word,
            {0xd7600018u, 0x0001010au, // s24 = independent pre-load v10/lane0 slot
             0x7e000218u});            // consume the new scalar s24
        auto reloaded = spill_read;
        reloaded.insert(reloaded.begin() + 2,
            {0xd761000au, 0x00010081u}); // spill independent inline1 before the raw load
        check(rdna2_raw_wide_data_loads(decode(reloaded)).empty() &&
              !compile(reloaded, table_for(reloaded)).empty(), name,
              "unpredicated READLANE replaces old raw scalar with an independent pre-load spill");
        numeric(shader(export_word, {0xd761000au, 0x00010018u}),
                "raw WRITELANE input remains a numeric observer before any later reload");
    }
    std::printf("raw wide mask lifetime failures: %d\n", failures);
    return failures != 0;
}
