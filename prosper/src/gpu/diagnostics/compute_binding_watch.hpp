#pragma once
#include <cstdint>
#include <set>
#include <tuple>
#include <vector>

namespace prosper::gpu {
struct ShaderResource;
struct ShaderResourceTable;

// clang-format off

// One match from an address watch: which resource, and whether it came from the scalar parent or a
// realized array entry.
struct ComputeAddressWatchHit {
    uint32_t binding = 0;
    uint32_t fetch_pc = 0;
    uint32_t entry_index = 0xFFFFFFFFu;   // 0xFFFFFFFF for a scalar parent
    uint64_t base = 0;
    uint32_t size = 0;
    bool from_array_entry = false;
};

// What became of the dispatch a [compute-binds] row describes. `RecompileEmpty` rows come from a
// resource table that is incomplete by construction, so they are a lower bound on that program's
// bindings -- see the contract above report_compute_binding_watch.
enum class ComputeBindOutcome { Executed, SkippedDescriptors, RecompileEmpty };

// The identity a [compute-binds] row asserts, and therefore the exact key its dedup must use.
//
// A key NARROWER than the line silently drops rows that differ in what the reader is reading, and
// this reporter had that defect twice. First it was keyed on (program, gpu_addr) alone, so distinct
// bindings at one guest base collapsed: compute tables deliberately carry per-use resources -- the
// same allocation appears once as a sampled Texture and again as a StorageImage, or under two fetch
// pcs -- and whichever was encountered first suppressed the rest, so in a producer hunt the
// surviving row could be the READ view while the WRITE view went unprinted, which is the exact
// "nothing writes this surface" conclusion the instrument exists to prevent. Then, with that fixed,
// the key still omitted the four SHAPE fields the line prints (size, width, height, format) while
// the comment claimed it spanned everything -- and a dynamic descriptor can hold program, base,
// binding, fetch pc, class and outcome steady while its view shape changes, so the first row's
// metadata would be presented as if it described every later observation. On this title that is the
// worst possible field to collapse: extent and f16-versus-f11f11f10 ARE the conversion under
// investigation.
//
// So every field the row prints is a key field. Do not add a printed column without adding it here.
struct ComputeBindWatchKey {
    uint64_t program = 0;
    uint64_t resource_addr = 0;
    uint64_t watched = 0;
    uint32_t binding = 0;
    uint32_t fetch_pc = 0;
    uint32_t cls = 0;
    uint32_t outcome = 0;
    uint32_t size = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t format = 0;
    bool operator<(const ComputeBindWatchKey& other) const {
        return std::tie(program, resource_addr, watched, binding, fetch_pc, cls, outcome,
                        size, width, height, format) <
               std::tie(other.program, other.resource_addr, other.watched, other.binding,
                        other.fetch_pc, other.cls, other.outcome, other.size, other.width,
                        other.height, other.format);
    }
};

bool compute_address_range_contains(uint64_t base, uint64_t size, uint64_t wanted);
bool compute_address_ranges_overlap(uint64_t a_base, uint64_t a_size, uint64_t b_base, uint64_t b_size);
std::vector<ComputeAddressWatchHit> compute_address_window_hits(const ShaderResourceTable& table, uint64_t base, uint64_t bytes);
std::vector<ComputeAddressWatchHit> compute_address_watch_hits(const ShaderResourceTable& table, uint64_t wanted);
ComputeBindWatchKey compute_bind_watch_key(uint64_t program, const ShaderResource& resource, uint64_t watched, ComputeBindOutcome outcome);
std::vector<ComputeBindWatchKey> compute_bind_watch_rows(std::set<ComputeBindWatchKey>& reported, const std::vector<uint64_t>& watched, uint64_t code_addr, const ShaderResourceTable* resources, ComputeBindOutcome outcome);
const char* compute_bind_outcome_name(ComputeBindOutcome outcome);
// clang-format on
} // namespace prosper::gpu
