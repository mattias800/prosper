#include "gpu/diagnostics/refused_shader_dump.hpp"

#include "diagnostics/env_cache.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"   // recompile_coverage: the first unsupported instruction

#include <chrono>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <atomic>
#include <set>
#include <utility>

namespace prosper::gpu {
namespace {

// One bit per stage tag (vs, ps, cs, vsmain) under an epoch that advances in steps of 16.
constexpr uint64_t kStageBits = 15u;

struct DumpState {
    std::mutex mutex;
    std::string root;            // empty: derive from PROSPER_CAPTURE_DIR on first use
    std::string directory;       // created lazily on the first refusal
    std::set<std::pair<std::string, uint64_t>> seen;   // (stage, code hash)
    bool dir_failure_announced = false;
    std::atomic<bool> full{false};
    std::atomic<uint64_t> epoch{kStageBits + 1u};
    std::atomic<uint64_t> hash_evaluations{0};
    std::atomic<uint64_t> hashed_dwords{0};
};

DumpState& state() {
    static DumpState s;
    return s;
}

uint64_t hash_code(const uint32_t* code, size_t dwords) {
    state().hash_evaluations.fetch_add(1, std::memory_order_relaxed);
    state().hashed_dwords.fetch_add(dwords, std::memory_order_relaxed);
    uint64_t h = 0xcbf29ce484222325ull;   // FNV-1a over the words
    for (size_t i = 0; i < dwords; ++i) {
        h ^= code[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

uint64_t stage_bit(const char* stage) {
    return !stage                          ? 0u
           : !std::strcmp(stage, "vs")     ? 1u
           : !std::strcmp(stage, "ps")     ? 2u
           : !std::strcmp(stage, "cs")     ? 4u
           : !std::strcmp(stage, "vsmain") ? 8u   // a chained vertex program's main (#3135 P0)
                                           : 0u;
}

std::string make_directory(DumpState& s) {
    if (!s.directory.empty()) return s.directory;
    std::string root = s.root;
    if (root.empty()) {
        const char* capture = PROSPER_ENV_VALUE("PROSPER_CAPTURE_DIR");
        root = capture && *capture ? capture : ".";
    }
    // UTC from std::chrono's calendar: no localtime (not thread-safe) and no platform #if.
    const auto now = std::chrono::system_clock::now();
    const auto day = std::chrono::floor<std::chrono::days>(now);
    const std::chrono::year_month_day ymd{day};
    const std::chrono::hh_mm_ss hms{std::chrono::floor<std::chrono::seconds>(now - day)};
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    char name[96];
    std::snprintf(name, sizeof name, "refused_shaders_%04d%02u%02u-%02d%02d%02dZ_%06llu",
                  (int)ymd.year(), (unsigned)ymd.month(), (unsigned)ymd.day(),
                  (int)hms.hours().count(), (int)hms.minutes().count(),
                  (int)hms.seconds().count(), (unsigned long long)(ns % 1000000));
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::path(root) / name;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        if (!s.dir_failure_announced) {
            s.dir_failure_announced = true;
            std::fprintf(stderr, "[refused-shader] cannot create %s (%s); refused shaders are not "
                                 "dumped this run\n", dir.string().c_str(), ec.message().c_str());
        }
        return {};
    }
    s.directory = dir.string();
    return s.directory;
}

}  // namespace

bool note_refused_shader(const char* stage, uint64_t address, const uint32_t* code, size_t dwords,
                         const std::string& detail) {
    if (refused_shader_dump_full()) return false;
    if (!stage || !code || !dwords) return false;
    const uint64_t hash = hash_code(code, dwords);
    DumpState& s = state();
    std::lock_guard lock(s.mutex);
    if (s.seen.count({stage, hash})) return false;
    if (s.seen.size() >= kRefusedShaderDumpMaxPrograms) return false;
    s.seen.insert({stage, hash});
    if (s.seen.size() >= kRefusedShaderDumpMaxPrograms) {
        s.full.store(true, std::memory_order_relaxed);
        std::fprintf(stderr,
                     "[refused-shader] %zu distinct programs recorded; further refusals "
                     "are not dumped (set PROSPER_SHADER_DUMP for an unbounded dump)\n",
                     kRefusedShaderDumpMaxPrograms);
    }
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
    // The recompiler's own last terminal reject for this program, when it recorded one. The
    // coverage census above is compute-safe and stage-agnostic, so on a graphics program it often
    // names an instruction the stage translator accepts; this is the translator's actual answer,
    // and without it the only way to learn it was a PROSPER_DBG run that desyncs the route.
    std::string reject = last_terminal_reject_reason(address);
    for (char& c : reject)
        if (c == '\n' || c == '\r' || c == '"') c = ' ';
    while (!reject.empty() && reject.back() == ' ') reject.pop_back();
    if (reject.size() > 240) reject.resize(240);
    if (FILE* index = std::fopen((std::filesystem::path(dir) / "index.txt").string().c_str(), "a")) {
        std::fprintf(index,
                     "%s addr=0x%llx dwords=%zu hash=%016llx first_bad_fmt=%d "
                     "first_bad_op=0x%x unsupported=%u file=%s %s%s%s%s\n",
                     stage, (unsigned long long)address, dwords, (unsigned long long)hash,
                     coverage.first_bad_fmt, coverage.first_bad_op, coverage.unsupported, file,
                     detail.c_str(), reject.empty() ? "" : " reject=\"", reject.c_str(),
                     reject.empty() ? "" : "\"");
        std::fclose(index);
    }
    // A draw a gate refused was never recompiled, so its "first unsupported" instruction is only
    // the generic coverage census and not why the draw was lost. The reason is in the detail the
    // caller passed; repeat it on this line, which is the one a person reads. It used to be in
    // index.txt alone, and #4580 was filed saying nothing logged it.
    std::string refusal;
    if (const size_t at = detail.find("refusal="); at != std::string::npos) {
        const size_t end = detail.find(' ', at);
        refusal = " " + detail.substr(at, end == std::string::npos ? end : end - at);
    }
    std::fprintf(stderr,
                 "[refused-shader] %s 0x%llx (%zu dwords, first unsupported fmt=%d "
                 "op=0x%x)%s -> %s%s%s%s%s\n",
                 stage, (unsigned long long)address, dwords, coverage.first_bad_fmt,
                 coverage.first_bad_op, refusal.c_str(), path.string().c_str(),
                 written ? "" : " (WRITE FAILED)", reject.empty() ? "" : " reject=\"",
                 reject.c_str(), reject.empty() ? "" : "\"");
    return written;
}

bool note_refused_shader(const char* stage, uint64_t address, const RefusedShaderSource& source,
                         const std::string& detail) {
    if (refused_shader_dump_full() || !stage || !source.words || source.words->empty())
        return false;
    const uint64_t bit = stage_bit(stage);
    if (source.memo && bit) {
        const uint64_t epoch = state().epoch.load(std::memory_order_relaxed);
        uint64_t previous = source.memo->epoch_stages.load(std::memory_order_relaxed);
        for (;;) {
            if ((previous & ~kStageBits) == epoch && (previous & bit)) return false;
            const uint64_t desired =
                epoch | bit | (((previous & ~kStageBits) == epoch) ? (previous & kStageBits) : 0u);
            if (source.memo->epoch_stages.compare_exchange_weak(previous, desired,
                                                                std::memory_order_relaxed))
                break;
        }
    }
    return note_refused_shader(stage, address, source.words->data(), source.words->size(), detail);
}

bool refused_shader_already_noted(const char* stage, const RefusedShaderSource& source) {
    if (refused_shader_dump_full() || !stage || !source.words || source.words->empty()) return true;
    if (!source.memo) return false;
    const uint64_t observed = source.memo->epoch_stages.load(std::memory_order_relaxed);
    return (observed & ~kStageBits) == state().epoch.load(std::memory_order_relaxed) &&
           (observed & stage_bit(stage));
}

bool note_refused_compute_shader(uint64_t address, const RefusedShaderSource& source,
                                 uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    if (refused_shader_already_noted("cs", source)) return false;
    char detail[96];
    std::snprintf(detail, sizeof detail, "dispatch groups=%ux%ux%u", groups_x, groups_y, groups_z);
    return note_refused_shader("cs", address, source, detail);
}

bool refused_shader_dump_full() {
    if (PROSPER_ENV_ON("PROSPER_NO_REFUSED_SHADER_DUMP")) return true;
    return state().full.load(std::memory_order_relaxed);
}

RefusedShaderDumpStats refused_shader_dump_stats() {
    DumpState& s = state();
    std::lock_guard lock(s.mutex);
    return {s.seen.size(), s.hash_evaluations.load(std::memory_order_relaxed),
            s.hashed_dwords.load(std::memory_order_relaxed)};
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
    s.full.store(false);
    s.epoch.fetch_add(kStageBits + 1u, std::memory_order_relaxed);
    s.hash_evaluations.store(0, std::memory_order_relaxed);
    s.hashed_dwords.store(0, std::memory_order_relaxed);
    s.dir_failure_announced = false;
}

}  // namespace prosper::gpu
