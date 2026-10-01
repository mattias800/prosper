// Project-owned packets and actual emitted live output; no Vulkan device or game source.
// Normal exact values guard unchanged translation, NOT denormal/rounding implementation.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/diagnostics/fragment_arithmetic.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "diagnostics/perf/perf_ledger.hpp"
#include "fixtures/test_scratch.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
using namespace prosper::gpu;
namespace perf = prosper::diagnostics::perf;
namespace {
int failures = 0;
size_t assertions = 0;
void check(bool condition, const char* label) {
    ++assertions;
    if (!condition) { ++failures; std::printf("[FAIL] %s\n", label); }
}
uint64_t count(perf::Counter c) { return perf::ledger().counters[size_t(c)].load(); }
void env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
template<class F> std::string capture(F&& body) {
    static unsigned sequence = 0;
    const auto path = prosper_test::test_scratch_path("arithmetic-" + std::to_string(sequence++) + ".log");
    FILE* out = std::fopen(path.string().c_str(), "w+b");
    if (!out) { check(false, "capture opens"); return {}; }
    std::fflush(stderr);
#ifdef _WIN32
    const int fd = _fileno(stderr), saved = _dup(fd);
    const bool redirected = saved >= 0 && _dup2(_fileno(out), fd) == 0;
#else
    const int fd = fileno(stderr), saved = dup(fd);
    const bool redirected = saved >= 0 && dup2(fileno(out), fd) >= 0;
#endif
    check(redirected, "capture redirects");
    if (redirected) body();
    std::fflush(stderr);
#ifdef _WIN32
    if (saved >= 0) { check(_dup2(saved, fd) == 0, "capture restores"); _close(saved); }
#else
    if (saved >= 0) { check(dup2(saved, fd) >= 0, "capture restores"); close(saved); }
#endif
    std::rewind(out);
    std::string result;
    char bytes[4096];
    for (size_t n; (n = std::fread(bytes, 1, sizeof bytes, out)) != 0;) result.append(bytes, n);
    std::fclose(out);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return result;
}
size_t occurrences(const std::string& s, const std::string& needle) {
    size_t n = 0;
    for (size_t p = 0; (p = s.find(needle, p)) != std::string::npos; p += needle.size()) ++n;
    return n;
}
std::string address(uint64_t value) { char text[32]; std::snprintf(text, sizeof text, "0x%llx", (unsigned long long)value); return text; }
uint64_t addr(const std::vector<uint32_t>& code) { return reinterpret_cast<uintptr_t>(code.data()); }
std::vector<uint32_t> program(uint32_t op, bool e64, unsigned repetitions = 1) {
    std::vector<uint32_t> code{0x7e0002ffu, 0x3f800000u, 0x7e0202ffu, 0x40000000u};
    for (unsigned n = 0; n < repetitions; ++n) {
        if (e64) { code.push_back(0xd4000000u | ((0x100u + op) << 16)); code.push_back(256u | (257u << 9)); }
        else code.push_back((op << 25) | (1u << 9) | 256u);
    }
    code.insert(code.end(), {0xf8001801u, 0u, 0xbf810000u});
    return code;
}
struct Instruction { uint32_t op; std::vector<uint32_t> a; };
struct Module {
    std::vector<Instruction> ins;
    bool valid = false;
    explicit Module(const std::vector<uint32_t>& words) {
        if (words.size() < 5 || words[0] != 0x07230203u) return;
        for (size_t pc = 5; pc < words.size();) {
            const size_t n = words[pc] >> 16;
            if (!n || n > words.size() - pc) return;
            ins.push_back({words[pc] & 65535u, {words.begin()+pc+1, words.begin()+pc+n}}); pc += n;
        }
        valid = true;
    }
    const Instruction* type(uint32_t id) const {
        for (const auto& i : ins) if (i.op >= 20 && i.op <= 32 && !i.a.empty() && i.a[0] == id) return &i;
        return nullptr;
    }
    const Instruction* def(uint32_t id) const {
        for (const auto& i : ins)
            if ((i.op == 43 || i.op == 59 || i.op == 80 || i.op == 124 || i.op == 129 || i.op == 133) &&
                i.a.size() >= 2 && i.a[1] == id) return &i;
        return nullptr;
    }
    bool scalar(uint32_t id, bool fp) const {
        const auto* t = type(id);
        return t && (fp ? t->op == 22 && t->a == std::vector<uint32_t>{id,32}
                        : t->op == 21 && t->a == std::vector<uint32_t>{id,32,0});
    }
    bool literal_float(uint32_t id, uint32_t bits) const {
        const auto* d = def(id);
        if (!d || d->a.size() != 3 || d->op != 124 || !scalar(d->a[0],true)) return false;
        const auto* c = def(d->a[2]);
        return c && c->op == 43 && c->a.size() == 3 && scalar(c->a[0],false) && c->a[2] == bits;
    }
    // Fail closed: follow the actual Location0 Output store's red SSA through the exact cast pair.
    // An unused ADD/MUL, Undef, wrong type/operand, stale output or opcode-count-only match fails.
    bool live_binary(uint32_t opcode) const {
        const Instruction* output = nullptr;
        uint32_t output_type = 0;
        for (const auto& s : ins) if (s.op == 62 && s.a.size() == 2) {
            const auto* v = def(s.a[0]); const auto* p = v ? type(v->a[0]) : nullptr;
            bool location0 = false;
            for (const auto& d : ins) location0 |= d.op == 71 && d.a == std::vector<uint32_t>{s.a[0],30,0};
            if (v && v->op == 59 && v->a.size() == 3 && v->a[2] == 3 && p && p->op == 32 &&
                p->a.size() == 3 && p->a[1] == 3 && location0) {
                if (output) return false;
                output = def(s.a[1]);
                output_type = p->a[2];
            }
        }
        if (!valid || !output || output->op != 80 || output->a.size() != 6 || output->a[0] != output_type) return false;
        const auto* vec = type(output->a[0]);
        if (!vec || vec->op != 23 || vec->a.size() != 3 || vec->a[2] != 4 || !scalar(vec->a[1],true)) return false;
        const auto* red = def(output->a[2]);
        const auto* bits = red && red->op == 124 && red->a.size() == 3 && scalar(red->a[0],true) ? def(red->a[2]) : nullptr;
        const auto* bin = bits && bits->op == 124 && bits->a.size() == 3 && scalar(bits->a[0],false) ? def(bits->a[2]) : nullptr;
        return bin && bin->op == opcode && bin->a.size() == 4 && scalar(bin->a[0],true) &&
            literal_float(bin->a[2],0x3f800000u) && literal_float(bin->a[3],0x40000000u);
    }
};
SharedShaderWords cached(const std::vector<uint32_t>& code, FragmentFloatMode mode, uint64_t* identity = nullptr) {
    return recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment, code.data(), code.size(),
        nullptr,nullptr,nullptr,identity,false,0,false,{},mode);
}
std::vector<uint32_t> direct(const std::vector<uint32_t>& code, FragmentFloatMode mode,
                            uint64_t program_address, FragmentArithmeticObservation* out = nullptr) {
    return recompile_fragment(code.data(),code.size(),nullptr,nullptr,UINT32_MAX,nullptr,false,
        {RecompileDiagnosticStage::Fragment,program_address},mode,out);
}
void dump(const std::filesystem::path& directory, const char* name, const std::vector<uint32_t>& words) {
    if (directory.empty()) return;
    std::ofstream stream(directory / name, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size()*4));
    check(bool(stream), "SOURCE dump writes");
}
}
int main(int argc, char** argv) {
    std::filesystem::path dumps;
    if (argc == 3 && std::string(argv[1]) == "--dump-directory") {
        dumps = argv[2]; std::filesystem::create_directories(dumps);
    } else if (argc != 1) return 2;
    env("PROSPER_NO_SHADER_CACHE", nullptr);
    env("PROSPER_SHADER_DUMP_SUCCESS", nullptr);
    env("PROSPER_FS_TAP", nullptr);
    for (bool e64 : {false,true}) for (uint32_t op : {3u,8u}) {
        auto code = program(op,e64);
        std::vector<Rdna2Inst> decoded; rdna2_walk(code.data(),code.size(),decoded);
        check(decoded.size() == 5 && decoded[2].pc == 4 && decoded[2].opcode == (e64?0x100u+op:op),
              "project decoder confirms actual arithmetic boundary");
        FragmentArithmeticObservation unknown;
        auto baseline = direct(code,{},0x40620000u,&unknown);
        check(!baseline.empty() && Module(baseline).live_binary(op==3?129:133), "actual arithmetic reaches typed live red output");
        auto wrong_op=baseline, poison=baseline, stale=baseline, wrong_input=baseline, wrong_output_type=baseline;
        uint32_t uint_type = 0;
        for (size_t pc=5;pc<baseline.size();) {
            const size_t size=baseline[pc]>>16; const auto opcode=baseline[pc]&65535u;
            if (!size || size>baseline.size()-pc) break;
            if (opcode==21 && size==4 && baseline[pc+2]==32 && baseline[pc+3]==0) uint_type=baseline[pc+1];
            if (opcode==32 && size==4 && baseline[pc+2]==3) wrong_output_type[pc+3]=uint_type;
            if (opcode==(op==3?129u:133u) && size==5) {
                wrong_op[pc]=(uint32_t(size)<<16)|(op==3?133u:129u);
                // Keep result type/id and replace the actual arithmetic definition with Undef1;
                // remove the two now-invalid operands rather than constructing a malformed opcode.
                poison[pc]=(3u<<16)|1u; poison.erase(poison.begin()+pc+3,poison.begin()+pc+5);
                wrong_input[pc+3]=baseline[pc+4];
            }
            if (opcode==80 && size==7) stale[pc+3]=baseline[pc+4];
            pc+=size;
        }
        check(!Module(wrong_op).live_binary(op==3?129:133) && !Module(poison).live_binary(op==3?129:133) &&
            !Module(stale).live_binary(op==3?129:133) && !Module(wrong_input).live_binary(op==3?129:133) &&
            !Module(wrong_output_type).live_binary(op==3?129:133),
            "independent sink oracle refuses wrong op, Undef, stale output, duplicated input and mismatched output type");
        check(unknown.site_count == 1 && unknown.sites[0].pc == 4 && !unknown.float_mode.available &&
            unknown.families == (op==3?1:2) && unknown.producing_program == 0x40620000u && unknown.source_fingerprint != 0,
            "unknown producing provenance and actual guest PC/family retained");
        for (unsigned mode = 0; mode < 256; ++mode) {
            FragmentArithmeticObservation observation;
            const auto words = direct(code,{true,uint8_t(mode)},0x40620100u+mode,&observation);
            check(words == baseline && observation.float_mode == FragmentFloatMode{true,uint8_t(mode)} &&
                observation.sites[0] == unknown.sites[0] && observation.families == unknown.families,
                "all mode bytes retained without changing arithmetic SOURCE");
        }
        const auto file = std::string(e64?"e64_":"e32_")+(op==3?"add.spv":"mul.spv");
        dump(dumps,file.c_str(),baseline);
        const auto raw_file = std::string(e64?"e64_":"e32_")+(op==3?"add.bin":"mul.bin");
        dump(dumps,raw_file.c_str(),code);
        const auto before = count(perf::Counter::FragmentArithmeticRequests);
        const auto log = capture([&] { direct(code,{true,0x31},0x40620200u+(e64?16:0)+op); });
        check(count(perf::Counter::FragmentArithmeticRequests)==before+1 &&
            log.find("FLOAT_MODE=0x31")!=std::string::npos && log.find("pc=4")!=std::string::npos &&
            log.find("pc-unit=dwords")!=std::string::npos &&
            log.find(op==3?"family=F32-ADD":"family=F32-MUL")!=std::string::npos &&
            log.find("observation=compiler-request")!=std::string::npos && log.find("not-a-measured-GPU-failure")!=std::string::npos,
            "normal offline direct request announces actual known-mode site once");
    }
    auto mixed=program(3,false);
    mixed.insert(mixed.end()-3,(8u<<25)|(1u<<9)|256u);
    const auto mixed_requests=count(perf::Counter::FragmentArithmeticRequests);
    const auto mixed_add=count(perf::Counter::FragmentArithmeticAddRequests);
    const auto mixed_mul=count(perf::Counter::FragmentArithmeticMulRequests);
    const auto mixed_log=capture([&] { direct(mixed,{true,0x11},0x40620600u); });
    check(count(perf::Counter::FragmentArithmeticRequests)==mixed_requests+1 &&
        count(perf::Counter::FragmentArithmeticAddRequests)==mixed_add+1 &&
        count(perf::Counter::FragmentArithmeticMulRequests)==mixed_mul+1 &&
        occurrences(mixed_log,"family=F32-ADD")==1 && occurrences(mixed_log,"family=F32-MUL")==1,
        "actual mixed ADD/MUL request counts once with both site families");
    auto code = program(3,false), alias = code;
    clear_shader_recompile_cache();
    const auto before = count(perf::Counter::FragmentArithmeticRequests);
    const auto known = count(perf::Counter::FragmentArithmeticKnownMode);
    const auto unknown = count(perf::Counter::FragmentArithmeticUnknownMode);
    uint64_t cold_id=0,warm_id=0,unknown_id=0;
    SharedShaderWords cold,warm,missing;
    auto cold_log = capture([&] { cold=cached(code,{true,0x31},&cold_id); });
    auto warm_log = capture([&] { warm=cached(alias,{true,0x31},&warm_id); });
    auto unknown_log = capture([&] { missing=cached(alias,{},&unknown_id); });
    check(cold && warm && missing && !cold->empty() && cold==warm && *cold==*missing && cold_id==warm_id && cold_id && unknown_id!=cold_id,
        "warm different-address alias shares exact immutable module; mode remains a key input");
    check(count(perf::Counter::FragmentArithmeticRequests)==before+3 && count(perf::Counter::FragmentArithmeticKnownMode)==known+2 &&
        count(perf::Counter::FragmentArithmeticUnknownMode)==unknown+1, "cold nested compilation does not double-count; warm request continues accounting");
    check(cold_log.find("FLOAT_MODE=0x31")!=std::string::npos && warm_log.find("program="+address(addr(alias))+" ")!=std::string::npos &&
        warm_log.find("producing-program="+address(addr(code))+" ")!=std::string::npos && unknown_log.find("FLOAT_MODE=unavailable")!=std::string::npos,
        "warm warning separates current lookup address from immutable producer mode/address");
    const auto repeated = capture([&] { for (int i=0;i<5;++i) cached(alias,{true,0x31}); });
    check(repeated.empty() && count(perf::Counter::FragmentArithmeticRequests)==before+8,
        "bounded repeated lines do not suppress request accounting");
    std::vector<uint32_t> copied;
    capture([&] { copied=recompile_graphics_shader_cached(ShaderProgramStage::Fragment,alias.data(),alias.size(),nullptr,nullptr,nullptr,nullptr,false,0,false,{}, {true,0x31}); });
    check(copied==*cold && count(perf::Counter::FragmentArithmeticRequests)==before+9, "copied public warm route counts once");
    env("PROSPER_NO_SHADER_CACHE","1");
    SharedShaderWords bypass;
    const auto bypass_log=capture([&] { bypass=cached(alias,{true,0x20}); });
    env("PROSPER_NO_SHADER_CACHE",nullptr);
    check(bypass && *bypass==*cold && bypass_log.find("FLOAT_MODE=0x20")!=std::string::npos && count(perf::Counter::FragmentArithmeticRequests)==before+10,
        "bypass keeps actual producing mode and counts once");
    {
        perf::SuppressDrawDropCounting suppress;
        const auto quiet=capture([&] { cached(alias,{true,0x31}); direct(code,{true,0x20},0x4062ffffu); });
        check(quiet.empty() && count(perf::Counter::FragmentArithmeticRequests)==before+10, "F9 re-realization does not create observations");
    }
    auto captured_cold=code;
    const auto capture_before=count(perf::Counter::FragmentArithmeticRequests);
    const auto capture_log=capture([&] {
        perf::SuppressDrawDropCounting suppress;
        cached(captured_cold,{true,0x77});
    });
    const auto after_capture=capture([&] { cached(captured_cold,{true,0x77}); });
    check(capture_log.empty() && count(perf::Counter::FragmentArithmeticRequests)==capture_before+1 &&
        after_capture.find("FLOAT_MODE=0x77")!=std::string::npos,
        "suppressed cold capture retains facts; later normal warm request still announces");
    auto no_arithmetic = program(3,false,0);
    const auto no_log=capture([&] { direct(no_arithmetic,{},0x40620300u); });
    const auto no_before=count(perf::Counter::FragmentArithmeticKnownMode)+count(perf::Counter::FragmentArithmeticUnknownMode);
    capture([&] { direct(no_arithmetic,{true,0},0x40620301u); });
    check(no_log.empty() && count(perf::Counter::FragmentArithmeticKnownMode)+count(perf::Counter::FragmentArithmeticUnknownMode)==no_before,
        "observed no-arithmetic requests do not announce arithmetic");
    auto vertex=code;
    vertex[vertex.size()-3]=0xf8001a01u; // EXP PARAM0 -> Location0, red carries the same ADD SSA
    vertex.insert(vertex.end()-3,{0xf80018cfu,0u}); // complete POS0 export also required by vertex stage
    const auto vertex_before=count(perf::Counter::FragmentArithmeticRequests);
    std::vector<uint32_t> vertex_words;
    const auto vertex_log=capture([&] { vertex_words=recompile_vertex(vertex.data(),vertex.size()); });
    check(!vertex_words.empty() && Module(vertex_words).live_binary(129) && vertex_log.empty() &&
        count(perf::Counter::FragmentArithmeticRequests)==vertex_before,
        "actual live vertex ADD output does not create fragment observations");
    dump(dumps,"vertex_add.spv",vertex_words);
    dump(dumps,"vertex_add.bin",vertex);
    auto many=program(3,false,12);
    FragmentArithmeticObservation truncated;
    const auto many_words=direct(many,{true,0x10},0x40620400u,&truncated);
    const auto truncated_log=capture([&] { observe_fragment_arithmetic(truncated,0x40620400u,!many_words.empty()); });
    check(!many_words.empty() && truncated.site_count==8 && truncated.truncated_emissions==4 &&
        occurrences(truncated_log,"family=F32-ADD")==8 && truncated_log.find("sites-truncated=yes")!=std::string::npos,
        "actual over-cap emission retains bounded prefix with explicit truncation");
    // Observer presentation-state copies, not a claim of a second guest compilation/admission.
    // Equal source/prefix provenance must not erase a newly reported truncation state.
    auto complete_prefix=truncated; complete_prefix.truncated_emissions=0;
    const auto complete_prefix_log=capture([&] { observe_fragment_arithmetic(complete_prefix,0x40620401u,true); });
    const auto newly_truncated_log=capture([&] { observe_fragment_arithmetic(truncated,0x40620401u,true); });
    check(complete_prefix_log.find("sites-truncated=no")!=std::string::npos &&
        newly_truncated_log.find("sites-truncated=yes")!=std::string::npos,
        "first-use dedup retains explicit changed truncation state even with an equal site prefix");
    // Refuse AFTER the valid arithmetic boundary. An independent no-export refuse happens before
    // emission and must not report merely because raw bytes contain ADD.
    auto refuse=code; refuse.insert(refuse.end()-3,0x7e00026au); // numeric VCC source in later v_mov
    const auto refused_before=count(perf::Counter::FragmentArithmeticRefusedRequests);
    std::vector<uint32_t> refused;
    const auto refusal_log=capture([&] { refused=direct(refuse,{true,0},0x40620500u); });
    check(refused.empty() && count(perf::Counter::FragmentArithmeticRefusedRequests)==refused_before+1 && refusal_log.find("module-produced=no")!=std::string::npos,
        "later refusal preserves actual attempted-lowering observation without execution claim");
    auto refuse_alias=refuse;
    SharedShaderWords refused_cold,refused_warm;
    const auto refused_requests=count(perf::Counter::FragmentArithmeticRefusedRequests);
    capture([&] { refused_cold=cached(refuse,{true,0}); });
    const auto refused_warm_log=capture([&] { refused_warm=cached(refuse_alias,{true,0}); });
    check(refused_cold && refused_warm && refused_cold==refused_warm && refused_cold->empty() &&
        count(perf::Counter::FragmentArithmeticRefusedRequests)==refused_requests+2 &&
        refused_warm_log.find("module-produced=no")!=std::string::npos &&
        refused_warm_log.find("producing-program="+address(addr(refuse))+" ")!=std::string::npos,
        "cached refusal replays producer facts and counts once on both cold and warm requests");
    auto no_export=code; no_export.erase(no_export.end()-3,no_export.end()-1);
    const auto early_log=capture([&] { direct(no_export,{true,0},0x40620501u); });
    check(early_log.find("[fragment-arithmetic-unverified]")==std::string::npos, "pre-emission refusal does not inventory raw ADD bytes");
    const auto invalid_before=count(perf::Counter::FragmentArithmeticRequests);
    SharedShaderWords invalid_cached;
    std::vector<uint32_t> invalid_direct;
    const auto invalid_log=capture([&] {
        invalid_direct=direct(code,{false,0x31},0x40620502u);
        invalid_cached=cached(code,{false,0x31});
    });
    check(invalid_direct.empty() && !invalid_cached && invalid_log.empty() &&
        count(perf::Counter::FragmentArithmeticRequests)==invalid_before+2,
        "both public noncanonical-mode refusals count requests once without inventing arithmetic sites");
    // Exercise the real observer's bounded installed inventory, then an already-known key. Counts
    // remain request counts at saturation; this is not a claim of 1024 distinct shader contents.
    const auto overflow_before=count(perf::Counter::FragmentArithmeticInventoryOverflowRequests);
    const auto full_log=capture([&] { for (uint64_t n=0;n<1030;++n) observe_fragment_arithmetic(truncated,0x40621000u+n,true); });
    check(occurrences(full_log,"announcement inventory full")==1 && count(perf::Counter::FragmentArithmeticInventoryOverflowRequests)>overflow_before,
        "inventory saturation announces once while further requests remain counted");
    const auto final_known=count(perf::Counter::FragmentArithmeticKnownMode);
    const auto after_full=capture([&] { cached(alias,{true,0x31}); });
    check(after_full.empty() && count(perf::Counter::FragmentArithmeticKnownMode)==final_known+1,
        "known warm request still counted after inventory fills");
    std::printf("fragment arithmetic diagnostics: %zu assertions, %d failures\n", assertions,failures);
    return failures?1:0;
}
