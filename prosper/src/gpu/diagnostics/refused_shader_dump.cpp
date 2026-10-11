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
#include <map>
#include <set>
#include <utility>

namespace prosper::gpu {
struct RefusedShaderInputCount {
    std::mutex mutex;
    std::string stem;
    std::set<uint64_t> identities;
    std::atomic<bool> full{false};
};
namespace {

// One bit per stage tag (vs, ps, cs, vsmain) under an epoch that advances in steps of 16.
constexpr uint64_t kStageBits = 15u;

struct DumpState {
    std::mutex mutex;
    std::string root;            // empty: derive from PROSPER_CAPTURE_DIR on first use
    std::string directory;       // created lazily on the first refusal
    std::set<std::pair<std::string, uint64_t>> seen;   // (stage, code hash)
    // Same identity as program deduplication, so VA reuse and relocation cannot mix counters.
    std::map<std::pair<std::string, uint64_t>, std::shared_ptr<RefusedShaderInputCount>> input_sets;
    std::atomic<uint64_t> input_set_global_locks{0};
    std::atomic<uint64_t> input_identity_evaluations{0};
    bool dir_failure_announced = false;
    size_t refusal_count = 0;   // distinct programs recorded under each budget
    size_t declined_count = 0;
    std::atomic<bool> full{false};
    std::atomic<bool> declined_full{false};
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

void write_input_set_count(const RefusedShaderInputCount& sets) {
    const std::string path = sets.stem + ".input_sets";
    if (FILE* f = std::fopen(path.c_str(), "w")) {
        std::fprintf(f, "distinct_input_sets=%zu%s\n", sets.identities.size(),
                     sets.identities.size() >= kRefusedShaderMaxInputSetsCounted ? "+" : "");
        std::fclose(f);
    }
}

}  // namespace

namespace {
bool budget_full(RefusedShaderBudget budget) {
    return budget == RefusedShaderBudget::BackendDeclined ? backend_declined_dump_full()
                                                          : refused_shader_dump_full();
}
}   // namespace

static bool note_refused_shader_impl(const char* stage, uint64_t address, const uint32_t* code,
                                     size_t dwords, const std::string& detail,
                                     RefusedShaderBudget budget,
                                     const RefusedShaderInputSet* inputs,
                                     const RefusedShaderMemo* memo = nullptr) {
    if (budget_full(budget)) return false;
    if (!stage || !code || !dwords) return false;
    const uint64_t hash = hash_code(code, dwords);
    if (memo) memo->code_hash.store(hash, std::memory_order_relaxed);
    DumpState& s = state();
    std::lock_guard lock(s.mutex);
    if (s.seen.count({stage, hash})) return false;
    const bool declined = budget == RefusedShaderBudget::BackendDeclined;
    size_t& recorded = declined ? s.declined_count : s.refusal_count;
    const size_t cap = declined ? kBackendDeclinedDumpMaxPrograms : kRefusedShaderDumpMaxPrograms;
    if (recorded >= cap) return false;
    s.seen.insert({stage, hash});
    ++recorded;
    if (recorded >= cap) {
        (declined ? s.declined_full : s.full).store(true, std::memory_order_relaxed);
        std::fprintf(stderr,
                     "[refused-shader] %zu distinct %sprograms recorded; further %s "
                     "are not dumped (set PROSPER_SHADER_DUMP for an unbounded dump)\n",
                     cap, declined ? "backend-declined " : "",
                     declined ? "backend declines" : "refusals");
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
    // The live compile inputs (#4808), beside the program and only for a newly written one.
    std::string inputs_field;
    if (written && inputs && inputs->write) {
        std::string stem = path.string();
        stem.resize(stem.size() - 4);   // ".bin"
        if (inputs->write(stem)) {
            auto sets = std::make_shared<RefusedShaderInputCount>();
            sets->stem = stem;
            sets->identities = {inputs->identity};
            write_input_set_count(*sets);
            s.input_sets[{stage, hash}] = std::move(sets);
            inputs_field = std::string(" inputs=") +
                           std::filesystem::path(stem).filename().string() + ".inputs";
        }
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
                     "first_bad_op=0x%x unsupported=%u file=%s %s%s%s%s%s\n",
                     stage, (unsigned long long)address, dwords, (unsigned long long)hash,
                     coverage.first_bad_fmt, coverage.first_bad_op, coverage.unsupported, file,
                     detail.c_str(), inputs_field.c_str(), reject.empty() ? "" : " reject=\"",
                     reject.c_str(), reject.empty() ? "" : "\"");
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

bool note_refused_shader(const char* stage, uint64_t address, const uint32_t* code, size_t dwords,
                         const std::string& detail, RefusedShaderBudget budget,
                         const RefusedShaderInputSet* inputs) {
    return note_refused_shader_impl(stage, address, code, dwords, detail, budget, inputs);
}

bool note_refused_shader(const char* stage, uint64_t address, const RefusedShaderSource& source,
                         const std::string& detail, RefusedShaderBudget budget,
                         const RefusedShaderInputSet* inputs) {
    if (budget_full(budget) || !stage || !source.words || source.words->empty()) return false;
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
    return note_refused_shader_impl(stage, address, source.words->data(), source.words->size(),
                                    detail, budget, inputs, source.memo);
}

bool refused_shader_already_noted(const char* stage, const RefusedShaderSource& source) {
    if (refused_shader_dump_full() || !stage || !source.words || source.words->empty()) return true;
    if (!source.memo) return false;
    const uint64_t observed = source.memo->epoch_stages.load(std::memory_order_relaxed);
    return (observed & ~kStageBits) == state().epoch.load(std::memory_order_relaxed) &&
           (observed & stage_bit(stage));
}

bool note_refused_compute_shader(uint64_t address, const RefusedShaderSource& source,
                                 uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
                                 const RefusedShaderInputSet* inputs) {
    if (refused_shader_dump_full() || !source.words || source.words->empty()) return false;
    char detail[96];
    std::snprintf(detail, sizeof detail, "dispatch groups=%ux%ux%u", groups_x, groups_y, groups_z);
    if (!inputs || !source.memo) {
        if (refused_shader_already_noted("cs", source)) return false;
        return note_refused_shader("cs", address, source, detail, RefusedShaderBudget::Refusal,
                                   inputs);
    }
    return note_refused_shader_inputs("cs", address, source, detail, *inputs);
}

bool refused_shader_inputs_needed(const char* stage, const RefusedShaderSource& source) {
    if (refused_shader_dump_full() || !stage || !source.memo || !source.words ||
        source.words->empty())
        return false;
    const auto bit = stage_bit(stage);
    const size_t slot = bit == 1 ? 0 : bit == 2 ? 1 : bit == 4 ? 2 : 3;
    const auto& memo = *source.memo;
    std::lock_guard owner_lock(memo.input_mutex);
    if (!refused_shader_already_noted(stage, source)) return true;
    return memo.input_epoch == refused_shader_dump_epoch() && memo.input_counts[slot] &&
           !memo.input_counts[slot]->full.load(std::memory_order_relaxed);
}

bool note_refused_shader_inputs(const char* stage, uint64_t address,
                                const RefusedShaderSource& source, const std::string& detail,
                                const RefusedShaderInputSet& inputs) {
    if (!source.memo || !stage || refused_shader_dump_full()) return false;
    const auto bit = stage_bit(stage);
    const size_t slot = bit == 1 ? 0 : bit == 2 ? 1 : bit == 4 ? 2 : 3;
    const auto& memo = *source.memo;
    std::lock_guard owner_lock(memo.input_mutex);
    const uint64_t epoch = refused_shader_dump_epoch();
    if (memo.input_epoch != epoch) {
        memo.input_epoch = epoch;
        for (auto& count : memo.input_counts) count.reset();
    }
    auto& input_count = memo.input_counts[slot];
    const bool noted = refused_shader_already_noted(stage, source);
    // A previous dump with no sidecar has nothing to count. Saturation precedes hashing.
    if (noted && (!input_count || input_count->full.load(std::memory_order_relaxed))) return false;
    state().input_identity_evaluations.fetch_add(1, std::memory_order_relaxed);
    RefusedShaderInputSet resolved = inputs;
    if (inputs.identify) resolved.identity = inputs.identify();
    bool written = false;
    if (!noted) {
        written = note_refused_shader(stage, address, source, detail, RefusedShaderBudget::Refusal,
                                      &resolved);
        DumpState& s = state();
        s.input_set_global_locks.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard cold_lock(s.mutex);
        const auto found =
            s.input_sets.find({stage, memo.code_hash.load(std::memory_order_relaxed)});
        if (found != s.input_sets.end()) input_count = found->second;
    }
    if (input_count) {
        auto& sets = *input_count;
        std::lock_guard count_lock(sets.mutex);
        if (sets.identities.size() < kRefusedShaderMaxInputSetsCounted &&
            sets.identities.insert(resolved.identity).second) {
            sets.full.store(sets.identities.size() >= kRefusedShaderMaxInputSetsCounted,
                            std::memory_order_relaxed);
            write_input_set_count(sets);
        }
    }
    return written;
}

bool refused_shader_dump_full() {
    if (PROSPER_ENV_ON("PROSPER_NO_REFUSED_SHADER_DUMP")) return true;
    return state().full.load(std::memory_order_relaxed);
}

bool backend_declined_dump_full() {
    if (PROSPER_ENV_ON("PROSPER_NO_REFUSED_SHADER_DUMP")) return true;
    return state().declined_full.load(std::memory_order_relaxed);
}

uint64_t refused_shader_dump_epoch() {
    return state().epoch.load(std::memory_order_relaxed);
}

RefusedShaderDumpStats refused_shader_dump_stats() {
    DumpState& s = state();
    std::lock_guard lock(s.mutex);
    return {s.seen.size(), s.hash_evaluations.load(std::memory_order_relaxed),
            s.input_set_global_locks.load(std::memory_order_relaxed),
            s.input_identity_evaluations.load(std::memory_order_relaxed),
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
    s.input_sets.clear();
    s.full.store(false);
    s.declined_full.store(false);
    s.refusal_count = 0;
    s.declined_count = 0;
    s.epoch.fetch_add(kStageBits + 1u, std::memory_order_relaxed);
    s.hash_evaluations.store(0, std::memory_order_relaxed);
    s.hashed_dwords.store(0, std::memory_order_relaxed);
    s.input_set_global_locks.store(0, std::memory_order_relaxed);
    s.input_identity_evaluations.store(0, std::memory_order_relaxed);
    s.dir_failure_announced = false;
}

}  // namespace prosper::gpu
