// shader_inspect - print the decoded RDNA2 stream and resolved control-flow targets from a raw dump.
#include "../../src/gpu/rdna2_decode.hpp"
#include "../../src/gpu/rdna2_to_spirv.hpp"

#include <algorithm>
#include <cstdint>
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

} // namespace

int main(int argc, char** argv) {
    std::string stage;
    if (argc == 4 && std::strcmp(argv[2], "--stage") == 0) stage = argv[3];
    if ((argc != 2 && argc != 4) ||
        (!stage.empty() && stage != "vertex" && stage != "fragment")) {
        std::fprintf(stderr, "usage: %s <raw-rdna2.bin> [--stage vertex|fragment]\n", argv[0]);
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
    if (!stage.empty()) {
        std::vector<uint32_t> spirv = stage == "vertex"
            ? recompile_vertex(words.data(), words.size())
            : recompile_fragment(words.data(), words.size());
        stage_ok = !spirv.empty();
        std::printf("stage-recompile stage=%s status=%s spirv_dwords=%zu\n",
                    stage.c_str(), stage_ok ? "ok" : "rejected", spirv.size());
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

    return ended && stage_ok ? 0 : 1;
}
