#include "gpu/execute/compute_program_facts.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"   // recompile_diagnostic_verbose

namespace prosper::gpu {
namespace {

// A program's facts are a few hundred bytes per instruction; 64 MiB holds thousands of programs.
// Exceeding it drops everything (a later dispatch simply re-analyzes), which keeps the bound
// trivially correct without an LRU on a path whose whole point is being cheap.
constexpr uint64_t kFactsCacheLimitBytes = 64ull << 20;

struct FactsKey {
    uint64_t address;
    uint64_t dwords;
    bool operator==(const FactsKey&) const = default;
};
struct FactsKeyHash {
    size_t operator()(const FactsKey& key) const {
        return std::hash<uint64_t>{}(key.address * 0x9e3779b97f4a7c15ull ^ key.dwords);
    }
};

// Counters at powers of two from 1024 on, so an A/B run shows its lever moved (hits in the
// default arm, bypasses in the opt-out arm) in a handful of log lines.
void report_counts(const char* event, uint64_t count, const ComputeProgramFactsStats& stats,
                   size_t entries) {
    if (count < 1024 || (count & (count - 1)) != 0) return;
    std::fprintf(stderr, "[compute-facts] %s=%llu hits=%llu misses=%llu bypasses=%llu entries=%zu\n",
                 event, (unsigned long long)count, (unsigned long long)stats.hits,
                 (unsigned long long)stats.misses, (unsigned long long)stats.bypasses, entries);
}

struct FactsCache {
    std::mutex mutex;
    std::unordered_map<FactsKey, std::shared_ptr<const ComputeProgramFacts>, FactsKeyHash> entries;
    uint64_t bytes = 0;
    ComputeProgramFactsStats stats;
};

FactsCache& facts_cache() {
    static FactsCache cache;
    return cache;
}

uint64_t facts_bytes(const ComputeProgramFacts& facts) {
    uint64_t bytes = sizeof(ComputeProgramFacts) + facts.code.size() * sizeof(uint32_t) +
                     facts.decoded.size() * sizeof(Rdna2Inst);
    for (const auto& [tag, payload] : facts.probe_reject_reasons)
        bytes += tag.size() + payload.size() + 2 * sizeof(std::string);
    return bytes;
}

std::shared_ptr<ComputeProgramFacts> analyze(const uint32_t* code, size_t dwords,
                                             const RecompileDiagnosticContext& diagnostic) {
    auto facts = std::make_shared<ComputeProgramFacts>();
    facts->address = diagnostic.program_address;
    facts->code.assign(code, code + dwords);
    rdna2_walk(code, dwords, facts->decoded);
    {
        TerminalRejectCapture capture;
        facts->prefers_native_multiwave =
            compute_shader_prefers_native_multiwave(facts->decoded, code, dwords, diagnostic);
        facts->probe_reject_reasons = capture.take();
    }
    facts->uses_gds =
        std::any_of(facts->decoded.begin(), facts->decoded.end(), [](const Rdna2Inst& in) {
            return in.fmt == Rdna2Format::DS && in.ds_gds &&
                   (in.opcode == 0x0d || in.opcode == 0x36 || in.opcode == 0x3d ||
                    in.opcode == 0x3e);
        });
    return facts;
}

} // namespace

bool compute_program_facts_cache_enabled() {
    static const bool enabled = std::getenv("PROSPER_NO_COMPUTE_PROGRAM_FACTS_CACHE") == nullptr;
    return enabled;
}

std::shared_ptr<const ComputeProgramFacts> compute_program_facts(
        const uint32_t* code, size_t dwords, const RecompileDiagnosticContext& diagnostic) {
    FactsCache& cache = facts_cache();
    // The verbose stream prints per evaluation; keep printing it per dispatch.
    if (!compute_program_facts_cache_enabled() || recompile_diagnostic_verbose(diagnostic.program_address)) {
        auto facts = analyze(code, dwords, diagnostic);   // records its reasons as it runs
        std::lock_guard lock(cache.mutex);
        ++cache.stats.bypasses;
        ++cache.stats.probe_evaluations;
        report_counts("bypasses", cache.stats.bypasses, cache.stats, cache.entries.size());
        return facts;
    }
    const FactsKey key{diagnostic.program_address, dwords};
    std::shared_ptr<const ComputeProgramFacts> hit;
    {
        std::lock_guard lock(cache.mutex);
        const auto found = cache.entries.find(key);
        if (found != cache.entries.end() && found->second->code.size() == dwords &&
            (dwords == 0 ||
             std::memcmp(found->second->code.data(), code, dwords * sizeof(uint32_t)) == 0)) {
            ++cache.stats.hits;
            hit = found->second;
            report_counts("hits", cache.stats.hits, cache.stats, cache.entries.size());
        }
    }
    if (hit) {
        replay_terminal_reject_reasons(diagnostic.program_address, hit->probe_reject_reasons);
        return hit;
    }
    // Analyze outside the lock. The capture inside analyze() collects the probe's records while
    // record_terminal_reject_reason() applies them, so this evaluation's side effects are already
    // complete; only later hits replay.
    std::shared_ptr<ComputeProgramFacts> facts = analyze(code, dwords, diagnostic);
    const uint64_t bytes = facts_bytes(*facts);
    std::lock_guard lock(cache.mutex);
    ++cache.stats.misses;
    ++cache.stats.probe_evaluations;
    if (bytes <= kFactsCacheLimitBytes) {
        auto existing = cache.entries.find(key);
        if (existing != cache.entries.end()) {
            cache.bytes -= facts_bytes(*existing->second);
            cache.entries.erase(existing);
        }
        if (cache.bytes + bytes > kFactsCacheLimitBytes) {
            cache.entries.clear();
            cache.bytes = 0;
        }
        cache.entries.emplace(key, facts);
        cache.bytes += bytes;
    }
    return facts;
}

ComputeProgramFactsStats compute_program_facts_stats() {
    FactsCache& cache = facts_cache();
    std::lock_guard lock(cache.mutex);
    ComputeProgramFactsStats stats = cache.stats;
    stats.entries = cache.entries.size();
    stats.bytes = cache.bytes;
    return stats;
}

void reset_compute_program_facts_for_test() {
    FactsCache& cache = facts_cache();
    std::lock_guard lock(cache.mutex);
    cache.entries.clear();
    cache.bytes = 0;
    cache.stats = {};
}

bool compute_resource_paths_may_specialize(const ShaderResourceTable& table, uint32_t wave_size) {
    return std::any_of(table.resources.begin(), table.resources.end(),
                       [wave_size](const ShaderResource& resource) {
        return is_proven_null_bvh(resource) ||
               (wave_size == 64 && is_zero_record_raw_buffer(resource));
    });
}

} // namespace prosper::gpu
