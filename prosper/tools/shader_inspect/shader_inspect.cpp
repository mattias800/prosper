// shader_inspect - print the decoded RDNA2 stream and resolved control-flow targets from a raw dump.
#include "../../src/gpu/rdna2_decode.hpp"
#include "../../src/gpu/rdna2_to_spirv.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {

const char* format_name(Rdna2Format format) {
    static const char* names[] = {
        "SOP2", "SOP1", "SOPK", "SOPC", "SOPP", "SMEM", "VOP2", "VOP1", "VOPC",
        "VOP3", "VINTRP", "DS", "MUBUF", "MTBUF", "MIMG", "FLAT", "EXP", "VOP3P",
        "Unknown",
    };
    const size_t index = static_cast<size_t>(format);
    return index < std::size(names) ? names[index] : "Invalid";
}

const char* operand_kind_name(OperandKind kind) {
    static const char* names[] = {
        "none", "sgpr", "vgpr", "inline-int", "inline-float", "literal", "special",
    };
    const size_t index = static_cast<size_t>(kind);
    return index < std::size(names) ? names[index] : "invalid";
}

bool is_branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
           in.opcode != 0x03;
}

void print_operand(const char* label, const Operand& operand) {
    if (operand.kind == OperandKind::None) return;
    std::printf(" %s=%s:%d", label, operand_kind_name(operand.kind), operand.value);
}

// Does lowering this instruction require a ShaderResourceTable in this stage?
//
// shader_inspect reads a RAW shader dump, which carries instructions but no descriptors, so it can
// never supply a resource table. The stage recompilers then legitimately refuse to lower any
// instruction that resolves a V#/T#/S# through that table — see `emit_alu` in rdna2_to_spirv.cpp,
// which rejects MIMG on `!allow_smem || !rt` and SMEM on `!allow_smem`. That refusal is a property
// of THIS TOOL's inputs, not of the shader, and must never be reported as an unsupported
// instruction (#1571).
//
// The table-dependent formats are the same ones RecompileCoverage documents (MIMG / MUBUF / MTBUF).
// SMEM is additionally table-dependent in the GRAPHICS stages only, because `recompile_fragment_impl`
// and `recompile_vertex_impl` pass `allow_smem = (rt != nullptr)` while `recompile_compute` passes
// `allow_smem = true` unconditionally.
//
// Line numbers drift, so the functions above are named rather than cited by line. At the time of
// writing (master 276b8f92) they were: emit_alu SMEM reject rdna2_to_spirv.cpp:8280, emit_alu MIMG
// gate :9109, recompile_compute :14295, recompile_fragment_impl :14828, recompile_vertex_impl :15676.
bool needs_resource_table(Rdna2Format fmt, const std::string& stage) {
    switch (fmt) {
        case Rdna2Format::MIMG:
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF:
            return true;
        case Rdna2Format::SMEM:
            return stage == "vertex" || stage == "fragment";
        default:
            return false;
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string stage;
    if (argc == 4 && std::strcmp(argv[2], "--stage") == 0) stage = argv[3];
    if ((argc != 2 && argc != 4) ||
        (!stage.empty() && stage != "vertex" && stage != "fragment" && stage != "compute")) {
        std::fprintf(stderr, "usage: %s <raw-rdna2.bin> [--stage vertex|fragment|compute]\n", argv[0]);
        std::fprintf(stderr,
            "\n"
            "Decodes a raw RDNA2 shader dump. With --stage it also attempts a stage recompile.\n"
            "\n"
            "IMPORTANT: shader_inspect has NO resource table (a raw dump carries no descriptors), so\n"
            "the recompiler cannot lower MIMG/MUBUF/MTBUF — nor SMEM in the vertex/fragment stages.\n"
            "When such an instruction is present, a failed stage recompile is reported as\n"
            "status=undetermined-no-resource-table and is NOT evidence of an unsupported shader.\n"
            "For a table-accurate verdict use gpu_replay, which has the real descriptors:\n"
            "  gpu_replay <capture>.prgcap --inspect-only\n"
            "\n"
            "Exit: 0 ok, 1 genuine defect, 2 usage/IO error, 3 undetermined (no resource table).\n");
        return 2;
    }

    std::ifstream input(argv[1], std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "shader_inspect: cannot open %s\n", argv[1]);
        return 2;
    }
    constexpr size_t kMaxBytes = 16u << 20;
    input.seekg(0, std::ios::end);
    const std::streamoff input_size = input.tellg();
    if (input_size <= 0 || input_size % sizeof(uint32_t)) {
        std::fprintf(stderr, "shader_inspect: input must contain a non-empty whole number of dwords\n");
        return 2;
    }
    if (input_size > static_cast<std::streamoff>(kMaxBytes)) {
        std::fprintf(stderr, "shader_inspect: input exceeds the 16 MiB diagnostic bound\n");
        return 2;
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(input_size));
    if (!input.read(reinterpret_cast<char*>(bytes.data()), input_size)) {
        std::fprintf(stderr, "shader_inspect: failed to read %s\n", argv[1]);
        return 2;
    }

    std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    std::vector<Rdna2Inst> instructions;
    const size_t consumed = rdna2_walk(words.data(), words.size(), instructions);
    const RecompileCoverage coverage = recompile_coverage(words.data(), words.size());
    const bool ended = !instructions.empty() && instructions.back().is_end;
    std::printf("file=%s bytes=%zu dwords=%zu consumed=%zu instructions=%zu endpgm=%d\n",
                argv[1], bytes.size(), words.size(), consumed, instructions.size(), ended);
    std::printf("generic-coverage total=%u alu=%u exp=%u table=%u unsupported=%u first=%s/0x%x\n",
                coverage.total, coverage.alu, coverage.exports, coverage.table_dependent,
                coverage.unsupported,
                coverage.first_bad_fmt < 0 ? "none" : format_name(static_cast<Rdna2Format>(coverage.first_bad_fmt)),
                coverage.first_bad_op);
    bool stage_ok = true;
    bool stage_undetermined = false;
    if (!stage.empty()) {
        std::vector<uint32_t> spirv;
        if (stage == "vertex") {
            spirv = recompile_vertex(words.data(), words.size());
        } else if (stage == "fragment") {
            spirv = recompile_fragment(words.data(), words.size());
        } else {
            ComputeShaderConfig config;
            if (const char* wave = std::getenv("PROSPER_SHADER_INSPECT_WAVE_SIZE"))
                config.wave_size = static_cast<uint32_t>(std::strtoul(wave, nullptr, 0));
            if (const char* native = std::getenv("PROSPER_SHADER_INSPECT_NATIVE_SUBGROUP"))
                config.native_subgroup_size = static_cast<uint32_t>(std::strtoul(native, nullptr, 0));
            spirv = recompile_compute(words.data(), words.size(), nullptr, config);
        }
        stage_ok = !spirv.empty();

        // A table-less rejection is only attributable to the SHADER when the shader contains no
        // instruction that would have needed a resource table anyway. Otherwise the rejection is
        // unattributable and must be reported as such (#1571).
        uint32_t table_dependent = 0;
        uint32_t per_format[static_cast<size_t>(Rdna2Format::Unknown) + 1] = {};
        for (const auto& in : instructions) {
            if (!needs_resource_table(in.fmt, stage)) continue;
            ++table_dependent;
            const size_t index = static_cast<size_t>(in.fmt);
            if (index < std::size(per_format)) ++per_format[index];
        }
        stage_undetermined = !stage_ok && table_dependent > 0;

        std::printf("stage-recompile stage=%s status=%s spirv_dwords=%zu table_dependent=%u\n",
                    stage.c_str(),
                    stage_ok ? "ok" : (stage_undetermined ? "undetermined-no-resource-table" : "rejected"),
                    spirv.size(), table_dependent);
        if (stage_undetermined) {
            std::printf("stage-recompile NOTE: TOOL LIMITATION - THIS IS NOT EVIDENCE OF A SHADER DEFECT.\n");
            std::printf("stage-recompile NOTE: shader_inspect reads a raw dump, which carries no descriptors, so it\n");
            std::printf("stage-recompile NOTE: cannot supply a resource table. This shader has %u instruction(s)\n",
                        table_dependent);
            std::printf("stage-recompile NOTE: that need one (");
            bool first = true;
            for (size_t i = 0; i < std::size(per_format); ++i) {
                if (!per_format[i]) continue;
                std::printf("%s%s=%u", first ? "" : " ", format_name(static_cast<Rdna2Format>(i)), per_format[i]);
                first = false;
            }
            std::printf("), so the recompiler refused them BY DESIGN and this\n");
            std::printf("stage-recompile NOTE: rejection is UNATTRIBUTABLE. It is not evidence that the shader is\n");
            std::printf("stage-recompile NOTE: unsupported - and equally not evidence that it is fine: a genuine\n");
            std::printf("stage-recompile NOTE: defect may also be present and would look identical here.\n");
            std::printf("stage-recompile NOTE: For a verdict use gpu_replay, which has the real descriptors:\n");
            std::printf("stage-recompile NOTE:   gpu_replay <capture>.prgcap --inspect-only\n");
        }
    }
    const PcrelDispatchInfo dispatch = rdna2_pcrel_dispatch_info(words.data(), words.size());
    if (dispatch.valid) {
        std::printf("pcrel-dispatch selector=s%u+0x%x add=%d max=%u setpc=%u merge=%u "
                    "targets=",
                    dispatch.selector_sgpr_base, dispatch.selector_byte_offset,
                    dispatch.selector_addend, dispatch.selector_max, dispatch.setpc_pc,
                    dispatch.merge_pc);
        for (size_t i = 0; i < dispatch.target_pcs.size(); ++i)
            std::printf("%s%u", i ? "," : "", dispatch.target_pcs[i]);
        std::printf(" span=%zu\n", dispatch.required_dwords);
    }

    for (const auto& in : instructions) {
        std::printf("pc=%04u len=%u fmt=%-7s op=0x%03x words=", in.pc, in.len_dwords,
                    format_name(in.fmt), in.opcode);
        const uint32_t available = static_cast<uint32_t>(words.size() - in.pc);
        const uint32_t count = std::min(in.len_dwords, available);
        for (uint32_t i = 0; i < count; ++i)
            std::printf("%s%08x", i ? "," : "", words[in.pc + i]);
        print_operand("dst", in.dst);
        for (uint32_t i = 0; i < in.n_src; ++i) {
            char label[8];
            std::snprintf(label, sizeof(label), "src%u", i);
            print_operand(label, in.src[i]);
        }
        if (in.sdst.kind != OperandKind::None) print_operand("sdst", in.sdst);
        if (is_branch(in)) {
            const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
            std::printf(" simm=%d target=%lld", in.simm16, static_cast<long long>(target));
        } else if (in.fmt == Rdna2Format::SOPK) {
            std::printf(" simm=%d", in.simm16);
        }
        if (in.is_end) std::printf(" endpgm");
        std::printf("\n");
    }

    // Exit codes: 0 = stage recompiled (or no --stage given and the stream ends cleanly);
    // 1 = genuine defect (truncated stream, or a rejection attributable to the shader);
    // 2 = usage/IO error; 3 = verdict UNDETERMINED because no resource table could be supplied (#1571).
    // 3 is deliberately non-zero: shader_inspect must never report a table-less run as a pass.
    if (!ended) return 1;
    if (stage_undetermined) return 3;
    return stage_ok ? 0 : 1;
}
