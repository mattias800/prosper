#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/execute/refused_shader_source.hpp"
#include "gpu/execute/compute_program_facts.hpp"
#include "gpu/execute/shader_source_window.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <tuple>
#include <unordered_set>

namespace prosper::gpu {

namespace {
RefusedShaderSource shader_analysis_refused_source(const SharedShaderAnalysis& analysis) {
    // Both aliases share ownership with the exact immutable analysis. This observation cannot
    // grant shader/resource admission; early refusals use the same bounded byte-validating cache.
    return analysis
               ? RefusedShaderSource{{analysis, &analysis->code}, &analysis->refused_shader_memo}
               : RefusedShaderSource{};
}
}   // namespace

void note_refused_draw_shaders(const RefusedDrawShaders& shaders) {
    if (refused_shader_dump_full()) return;
    for (auto [tag, address, failed, original] :
         {std::tuple{"vs", shaders.vs_address, shaders.vs_failed, shaders.vs},
          std::tuple{"vsmain", shaders.chain_address, shaders.vs_failed, RefusedShaderSource{}},
          std::tuple{"ps", shaders.ps_address, shaders.ps_failed, shaders.ps}}) {
        if (!failed || !address) continue;
        // Pre-key guards have no compile-key owner. Observe their original readable program
        // through the same existing bounded analysis cache, not a guessed address generation.
        if (!original.words) {
            const size_t source_dwords = native_shader_source_dwords(address, shaders.max_dwords);
            if (!source_dwords) continue;
            original = shader_analysis_refused_source(
                tag[0] == 'p' && shaders.ps_analysis
                    ? shaders.ps_analysis
                    : acquire_shader_analysis(
                          reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(address)),
                          source_dwords));
        }
        if (refused_shader_already_noted(tag, original)) continue;
        char detail[512];
        std::snprintf(detail, sizeof detail,
                      "draw-order=%llu vs=%zu gs=%zu fs=%zu es=0x%llx ps=0x%llx ngg=%s link=%s "
                      "main=0x%llx%s%s",
                      static_cast<unsigned long long>(shaders.command_order), shaders.vs_words,
                      shaders.gs_words, shaders.fs_words,
                      static_cast<unsigned long long>(shaders.es_address),
                      static_cast<unsigned long long>(shaders.ps_address),
                      shaders.ngg_class ? shaders.ngg_class : "unknown",
                      shaders.link ? shaders.link : "unknown",
                      static_cast<unsigned long long>(shaders.chain_address),
                      shaders.refusal ? " refusal=" : "", shaders.refusal ? shaders.refusal : "");
        note_refused_shader(tag, address, original, detail);
    }
}

void note_refused_compute_program(const std::shared_ptr<const ComputeProgramFacts>& facts,
                                  uint64_t address, uint32_t groups_x, uint32_t groups_y,
                                  uint32_t groups_z) {
    if (!facts) return;
    const RefusedShaderSource source{{facts, &facts->code}, &facts->refused_shader_memo};
    note_refused_compute_shader(address, source, groups_x, groups_y, groups_z);
}

namespace {
thread_local const char* g_compute_decline_reason = nullptr;
}   // namespace

void set_compute_decline_reason(const char* reason) {
    g_compute_decline_reason = reason;
}

const char* take_compute_decline_reason() {
    const char* reason = g_compute_decline_reason;
    g_compute_decline_reason = nullptr;
    return reason;
}

namespace {

// What this thread has already offered the dump for a backend decline. Declines repeat every frame
// for the same program, so the question "is this one already dumped?" must cost no lock and no read
// of the program: the key is the address, the size and three sampled words (first, middle, last),
// which tells a different program reusing a shader-pool address from the one already dumped.
// Per thread, so a second thread may offer the same program once more; the dump's content hash
// then drops it. Bounded: once full, a thread offers no further programs.
constexpr size_t kDeclinedKeysPerThread = 256;

struct DeclinedKey {
    uint64_t address;
    uint32_t dwords;
    uint32_t sample[3];
    bool operator==(const DeclinedKey& other) const {
        return address == other.address && dwords == other.dwords &&
               std::equal(sample, sample + 3, other.sample);
    }
};
struct DeclinedKeyHash {
    size_t operator()(const DeclinedKey& key) const {
        uint64_t h = key.address * 0x9e3779b97f4a7c15ull ^ key.dwords;
        for (uint32_t word : key.sample) h = (h ^ word) * 0x100000001b3ull;
        return static_cast<size_t>(h);
    }
};
struct DeclinedKeys {
    uint64_t epoch = 0;
    std::unordered_set<DeclinedKey, DeclinedKeyHash> seen;
};

}   // namespace

void note_backend_declined_compute(uint64_t address, uint32_t dwords, uint32_t groups_x,
                                   uint32_t groups_y, uint32_t groups_z, const char* reason) {
    if (!address || !dwords || backend_declined_dump_full()) return;
    // The key is the cheap question; everything below runs once per distinct program per thread.
    thread_local DeclinedKeys keys;
    if (const uint64_t epoch = refused_shader_dump_epoch(); keys.epoch != epoch) {
        keys.seen.clear();   // the dump was reset (a test); what it holds is gone
        keys.epoch = epoch;
    }
    const uint32_t* code = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(address));
    const DeclinedKey key{address, dwords, {code[0], code[dwords / 2], code[dwords - 1]}};
    if (keys.seen.size() >= kDeclinedKeysPerThread || !keys.seen.insert(key).second) return;
    // The dispatch was realized, so its facts are normally cached; peek neither stores nor
    // replays terminal reject reasons, so it cannot overwrite a recompile reject's record.
    const auto facts = compute_program_facts_peek(code, dwords, address);
    if (!facts) return;
    const RefusedShaderSource source{{facts, &facts->code}, &facts->refused_shader_memo};
    // The [refused-shader] line repeats the refusal= field up to its first space, so the reason
    // (often a sentence such as "layered image deferred to #657") is written without spaces.
    std::string why = reason ? reason : "unrecorded";
    for (char& c : why)
        if (c == ' ' || c == '\t' || c == '"') c = '-';
    char groups[64];
    std::snprintf(groups, sizeof groups, "dispatch groups=%ux%ux%u", groups_x, groups_y, groups_z);
    note_refused_shader("cs", address, source,
                        std::string(groups) + " refusal=backend-declined:" + why,
                        RefusedShaderBudget::BackendDeclined);
}

}   // namespace prosper::gpu
