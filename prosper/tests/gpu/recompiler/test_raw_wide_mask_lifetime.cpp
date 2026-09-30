// A fresh emitted Bool mask and its possible physical high-word data are distinct lifetimes.
// No device is created. Synthetic ISA packets exercise the actual decoder/classifier/emitter.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <array>
#include <cstdio>
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

int main() {
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
        loop.insert(loop.begin() + 2, 0xbf82fffdu); // back to the compare
        numeric(shader(export_word, loop), "backwards mask proof requires an unsupported convergence proof");

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
