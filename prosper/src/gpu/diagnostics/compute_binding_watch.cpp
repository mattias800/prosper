#include "gpu/diagnostics/compute_binding_watch.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace prosper::gpu {
// clang-format off

// Does the half-open range [base, base+size) contain `wanted`? The address-watch diagnostic's
// matcher, exposed so its edge cases can be tested rather than reasoned about.
//
// Base equality is NOT sufficient for an address census: an interior address, an overlapping
// subview, and a runtime-selected array element all live inside a range whose base differs. The
// subtraction is ordered after the `wanted >= base` test so it cannot wrap for any base, including
// one near UINT64_MAX, and a zero-size range contains nothing.
bool compute_address_range_contains(uint64_t base, uint64_t size, uint64_t wanted) {
    // No explicit `size != 0`: for unsigned operands `(wanted - base) < 0` is already false, so a
    // zero-size interval is rejected by the comparison itself. An earlier revision carried the
    // redundant test AND claimed mutation evidence for it; deleting it alone changes nothing, and
    // the mutation that appeared to cover it had also flipped `<` to `<=`, so it isolated neither.
    return wanted >= base && (wanted - base) < size;
}

// Collect every resource in `table` whose BACKING RANGE contains `wanted`.
//
// The scalar/array distinction is load-bearing and is the same one gpu_dependency_graph.cpp makes.
// For a runtime-selected array the parent's `gpu_addr` is the DESCRIPTOR TABLE address and its
// `size` is the widest element size — that pair is not a backing range at all, so testing it
// unconditionally both fabricates matches against the descriptor table and double-reports an entry
// whenever the parent and entry 0 happen to share an address. Branch, exactly as the graph does:
// `table_index_count != 0` means the realized entries are the backing ranges and the parent is not.
// Overlap of two ranges, ordered so it cannot wrap near UINT64_MAX.
bool compute_address_ranges_overlap(uint64_t a_base, uint64_t a_size,
                                           uint64_t b_base, uint64_t b_size) {
    if (!a_size || !b_size) return false;
    if (a_base >= b_base) return (a_base - b_base) < b_size;
    return (b_base - a_base) < a_size;
}

// Windowed variant. The single-address form answers "does this program bind the exact byte", which
// is the wrong question whenever the caller is watching a RANGE: a program binding
// [window_base + 128, +160) touches the window and does not contain its first byte, so the
// address form reports it as a non-toucher.
//
// That is not hypothetical. It produced a false "this program writes bytes it does not bind" on GTA
// V -- reported as an out-of-bounds-write lead before the resource map showed the binding sitting
// 128 bytes inside the window. An instrument whose two halves ask different questions manufactures
// exactly the finding it was built to detect.
std::vector<ComputeAddressWatchHit> compute_address_window_hits(
        const ShaderResourceTable& table, uint64_t base, uint64_t bytes) {
    std::vector<ComputeAddressWatchHit> hits;
    for (const ShaderResource& r : table.resources) {
        if (r.table_index_count) {
            for (size_t e = 0; e < r.table_entries.size(); ++e) {
                const ShaderBufferTableEntry& entry = r.table_entries[e];
                if (!compute_address_ranges_overlap(entry.gpu_addr, entry.size, base, bytes))
                    continue;
                hits.push_back({r.binding, r.fetch_pc, static_cast<uint32_t>(e), entry.gpu_addr,
                                entry.size, true});
            }
            continue;
        }
        if (compute_address_ranges_overlap(r.gpu_addr, r.size, base, bytes))
            hits.push_back({r.binding, r.fetch_pc, 0xFFFFFFFFu, r.gpu_addr, r.size, false});
    }
    return hits;
}

std::vector<ComputeAddressWatchHit> compute_address_watch_hits(
        const ShaderResourceTable& table, uint64_t wanted) {
    std::vector<ComputeAddressWatchHit> hits;
    for (const ShaderResource& r : table.resources) {
        if (r.table_index_count) {
            for (size_t e = 0; e < r.table_entries.size(); ++e) {
                const ShaderBufferTableEntry& entry = r.table_entries[e];
                if (!compute_address_range_contains(entry.gpu_addr, entry.size, wanted)) continue;
                hits.push_back({r.binding, r.fetch_pc, static_cast<uint32_t>(e), entry.gpu_addr,
                                entry.size, true});
            }
            continue;
        }
        if (compute_address_range_contains(r.gpu_addr, r.size, wanted))
            hits.push_back({r.binding, r.fetch_pc, 0xFFFFFFFFu, r.gpu_addr, r.size, false});
    }
    return hits;
}

ComputeBindWatchKey compute_bind_watch_key(uint64_t program, const ShaderResource& resource,
                                                  uint64_t watched, ComputeBindOutcome outcome) {
    ComputeBindWatchKey key;
    key.program = program;
    key.resource_addr = resource.gpu_addr;
    key.watched = watched;
    key.binding = resource.binding;
    key.fetch_pc = resource.fetch_pc;
    key.cls = static_cast<uint32_t>(resource.cls);
    key.outcome = static_cast<uint32_t>(outcome);
    key.size = resource.size;
    key.width = resource.width;
    key.height = resource.height;
    key.format = static_cast<uint32_t>(resource.format);
    return key;
}

// THE production selection + dedup seam: which rows [compute-binds] emits for one dispatch. Range
// filtering, key construction and the dedup insert all live here, so a test that calls this function
// exercises the same code the reporter runs and a narrowing mutation anywhere in it is visible to
// that test. report_compute_binding_watch below is then only environment parsing and fprintf.
//
// Split out for exactly that reason: an earlier regression tested compute_bind_watch_key alone and
// therefore proved the helper builds the intended tuple while proving nothing about whether the
// reporter deduped on it -- narrowing the key at the reporter's own insert left the test green.
std::vector<ComputeBindWatchKey> compute_bind_watch_rows(
    std::set<ComputeBindWatchKey>& reported, const std::vector<uint64_t>& watched,
    uint64_t code_addr, const ShaderResourceTable* resources, ComputeBindOutcome outcome) {
    std::vector<ComputeBindWatchKey> rows;
    if (watched.empty() || !resources) return rows;
    for (const auto& resource : resources->resources) {
        // Match the whole span, not the base: a consumer names a surface by its base while a
        // producer may bind a subrange or a mip, and requiring equality would report "nobody binds
        // it" for a producer that plainly does. A size of 0 is one byte, not an open span.
        const uint64_t begin = resource.gpu_addr;
        const uint64_t end = begin + (resource.size ? resource.size : 1);
        for (const uint64_t want : watched) {
            if (!begin || want < begin || want >= end) continue;
            const ComputeBindWatchKey key =
                compute_bind_watch_key(code_addr, resource, want, outcome);
            if (!reported.insert(key).second) continue;
            rows.push_back(key);
        }
    }
    return rows;
}

const char* compute_bind_outcome_name(ComputeBindOutcome outcome) {
    return outcome == ComputeBindOutcome::Executed              ? "executed"
         : outcome == ComputeBindOutcome::SkippedDescriptors    ? "skipped-descriptors"
                                                                : "partial-recompile-empty";
}
// clang-format on
} // namespace prosper::gpu
