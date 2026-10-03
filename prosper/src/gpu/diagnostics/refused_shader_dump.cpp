#include "gpu/diagnostics/refused_shader_dump.hpp"

#include "diagnostics/env_cache.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"   // recompile_coverage: the first unsupported instruction

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <set>
#include <utility>

namespace prosper::gpu {
namespace {

struct DumpState {
    std::mutex mutex;
    std::string root;            // empty: derive from PROSPER_CAPTURE_DIR on first use
    std::string directory;       // created lazily on the first refusal
    std::set<std::pair<std::string, uint64_t>> seen;   // (stage, code hash)
    bool cap_announced = false;
};

DumpState& state() {
    static DumpState s;
    return s;
}

uint64_t hash_code(const uint32_t* code, size_t dwords) {
    uint64_t h = 0xcbf29ce484222325ull;   // FNV-1a over the words
    for (size_t i = 0; i < dwords; ++i) {
        h ^= code[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

std::string make_directory(DumpState& s) {
    if (!s.directory.empty()) return s.directory;
    std::string root = s.root;
    if (root.empty()) {
        const char* capture = PROSPER_ENV_VALUE("PROSPER_CAPTURE_DIR");
        root = capture && *capture ? capture : ".";
    }
    char name[96];
    const std::time_t now = std::time(nullptr);
    // std::localtime's static buffer is safe here: every caller holds DumpState::mutex. The
    // nanosecond suffix keeps two runs started in the same second apart.
    const std::tm tm = *std::localtime(&now);
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::snprintf(name, sizeof name, "refused_shaders_%04d%02d%02d-%02d%02d%02d_%06llu",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,
                  (unsigned long long)(ns % 1000000));
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::path(root) / name;
    std::filesystem::create_directories(dir, ec);
    if (ec) return {};
    s.directory = dir.string();
    return s.directory;
}

}  // namespace

bool note_refused_shader(const char* stage, uint64_t address, const uint32_t* code, size_t dwords,
                         const std::string& detail) {
    if (PROSPER_ENV_ON("PROSPER_NO_REFUSED_SHADER_DUMP")) return false;
    if (!stage || !code || !dwords) return false;
    const uint64_t hash = hash_code(code, dwords);
    DumpState& s = state();
    std::lock_guard lock(s.mutex);
    if (s.seen.count({stage, hash})) return false;
    if (s.seen.size() >= kRefusedShaderDumpMaxPrograms) {
        if (!s.cap_announced) {
            s.cap_announced = true;
            std::fprintf(stderr, "[refused-shader] %zu distinct programs recorded; further refusals "
                                 "are not dumped (set PROSPER_SHADER_DUMP for an unbounded dump)\n",
                         kRefusedShaderDumpMaxPrograms);
        }
        return false;
    }
    s.seen.insert({stage, hash});
    const std::string dir = make_directory(s);
    if (dir.empty()) return false;
    char file[64];
    std::snprintf(file, sizeof file, "%s_%llx_%016llx.bin", stage, (unsigned long long)address,
                  (unsigned long long)hash);
    const std::filesystem::path path = std::filesystem::path(dir) / file;
    bool written = false;
    if (FILE* f = std::fopen(path.string().c_str(), "wb")) {
        written = std::fwrite(code, sizeof(uint32_t), dwords, f) == dwords;
        written = (std::fclose(f) == 0) && written;
    }
    const RecompileCoverage coverage = recompile_coverage(code, dwords);
    if (FILE* index = std::fopen((std::filesystem::path(dir) / "index.txt").string().c_str(), "a")) {
        std::fprintf(index, "%s addr=0x%llx dwords=%zu hash=%016llx first_bad_fmt=%d "
                            "first_bad_op=0x%x unsupported=%u file=%s %s\n",
                     stage, (unsigned long long)address, dwords, (unsigned long long)hash,
                     coverage.first_bad_fmt, coverage.first_bad_op, coverage.unsupported, file,
                     detail.c_str());
        std::fclose(index);
    }
    std::fprintf(stderr, "[refused-shader] %s 0x%llx (%zu dwords, first unsupported fmt=%d "
                         "op=0x%x) -> %s%s\n",
                 stage, (unsigned long long)address, dwords, coverage.first_bad_fmt,
                 coverage.first_bad_op, path.string().c_str(), written ? "" : " (WRITE FAILED)");
    return written;
}

std::string refused_shader_dump_directory() {
    DumpState& s = state();
    std::lock_guard lock(s.mutex);
    return s.directory;
}

void reset_refused_shader_dump_for_test(const std::string& root) {
    DumpState& s = state();
    std::lock_guard lock(s.mutex);
    s.root = root;
    s.directory.clear();
    s.seen.clear();
    s.cap_announced = false;
}

}  // namespace prosper::gpu
