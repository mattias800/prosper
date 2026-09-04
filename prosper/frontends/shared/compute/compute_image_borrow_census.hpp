#pragma once

// Why this file exists.
//
// A compute dispatch produces a storage image on the device; a graphics draw in the same submit
// wants to sample exactly those pixels. `import_live_compute_storage_image` exists so the second
// can borrow the first's `VkImage` instead of routing 63.8 MiB through guest memory -- compute
// re-tiling its result out, graphics detiling it back in. On Stray's title screen that borrow
// misses on every reference (#3307), and the round trip it fails to remove is the single largest
// host cost in the frame.
//
// The importer and the borrow it calls have at least SIX independent ways to decline, and before
// this header not one of them logged a reason:
//
//   1. the consumer's own precondition (shape, class, host-backed data, mip levels, srgb, format);
//   2. no cache entry under the key;
//   3. an entry exists but its export/content/image authority is not published;
//   4. the entry is published but no proof says guest memory still matches it;
//   5. the producer never authorized an export in the first place;
//   6. the borrow succeeded and the RENDERER rejected the imported image afterwards.
//
// `STRAY_STATUS.md` asserted (2) from the outside. That is a guess about a five-way branch, and it
// is the kind of guess this project has repeatedly paid for: a plausible reason quoted as a measured
// one. So every branch is counted, unconditionally, and one of them will turn out to carry the mass.
//
// The counters partition. Every import lands in exactly one decline reason or exactly one borrow
// outcome, and both arrays print the denominator they are a fraction of, so two reports taken at
// different points in a run remain comparable -- the failure that produced #3155's retracted
// numbers. The `## Ruled out` value of a partition is that it cannot be read selectively.
//
// Counting is unconditional and per-IMPORT (a handful of relaxed atomic adds against a path whose
// alternative is a multi-megabyte detile). Only the REPORT is gated, on
// PROSPER_COMPUTE_BORROW_CENSUS. The one genuinely non-free part -- scanning the image cache for a
// near-miss key when the lookup found nothing -- is additionally gated on that same variable,
// because it is O(cache) rather than O(1).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace prosper::frontend {

// ---------------------------------------------------------------------------
// (1) The consumer's precondition, as a pure classifier.
//
// The production code used to spell this as one `||` chain that returned false, so a declined
// import was indistinguishable from any other declined import. The classifier answers WHICH term
// declined. `classify_compute_image_import(...) == None` must remain exactly equivalent to that
// chain's negation; `test_compute_image_borrow_census` asserts the equivalence over the complete
// boolean product rather than over hand-picked cases, because the failure mode being guarded is a
// term silently dropped during the rewrite.
// ---------------------------------------------------------------------------

enum class ComputeImageImportDecline : uint8_t {
    None = 0,          // accepted: the borrow was attempted
    NoContext,         // no live compute backend / no device
    NoGuestAddress,
    NotTextureClass,
    HostBackedData,    // a capture/replay resource whose bytes are host-owned
    GuestByteRange,    // zero, or wider than the u32 the cache key carries
    Shape,             // neither an ordinary 2D/2D-array/3D nor the cube-array alias
    MipLevels,         // declared_mip_levels != 1
    MipTail,
    Srgb,
    DepthCompare,
    NativeFormat,      // no exact Vulkan storage format for this guest format/component count
    Count,
};

struct ComputeImageImportInputs {
    bool have_context = false;
    bool have_gpu_addr = false;
    bool texture_class = false;
    bool host_data = false;
    bool guest_bytes_in_range = false;
    bool shape_supported = false;
    bool single_mip_level = false;
    bool in_mip_tail = false;
    bool srgb = false;
    bool depth_compare = false;
    bool native_format_defined = false;
};

constexpr ComputeImageImportDecline classify_compute_image_import(
    const ComputeImageImportInputs& in) {
    if (!in.have_context) return ComputeImageImportDecline::NoContext;
    if (!in.have_gpu_addr) return ComputeImageImportDecline::NoGuestAddress;
    if (!in.texture_class) return ComputeImageImportDecline::NotTextureClass;
    if (in.host_data) return ComputeImageImportDecline::HostBackedData;
    if (!in.guest_bytes_in_range) return ComputeImageImportDecline::GuestByteRange;
    if (!in.shape_supported) return ComputeImageImportDecline::Shape;
    if (!in.single_mip_level) return ComputeImageImportDecline::MipLevels;
    if (in.in_mip_tail) return ComputeImageImportDecline::MipTail;
    if (in.srgb) return ComputeImageImportDecline::Srgb;
    if (in.depth_compare) return ComputeImageImportDecline::DepthCompare;
    if (!in.native_format_defined) return ComputeImageImportDecline::NativeFormat;
    return ComputeImageImportDecline::None;
}

constexpr const char* compute_image_import_decline_name(ComputeImageImportDecline reason) {
    switch (reason) {
    case ComputeImageImportDecline::None:            return "accepted";
    case ComputeImageImportDecline::NoContext:       return "no_context";
    case ComputeImageImportDecline::NoGuestAddress:  return "no_gpu_addr";
    case ComputeImageImportDecline::NotTextureClass: return "not_texture";
    case ComputeImageImportDecline::HostBackedData:  return "host_data";
    case ComputeImageImportDecline::GuestByteRange:  return "guest_bytes";
    case ComputeImageImportDecline::Shape:           return "shape";
    case ComputeImageImportDecline::MipLevels:       return "mip_levels";
    case ComputeImageImportDecline::MipTail:         return "mip_tail";
    case ComputeImageImportDecline::Srgb:            return "srgb";
    case ComputeImageImportDecline::DepthCompare:    return "depth_compare";
    case ComputeImageImportDecline::NativeFormat:    return "native_format";
    case ComputeImageImportDecline::Count:           break;
    }
    return "?";
}

// ---------------------------------------------------------------------------
// (2)(3)(4) The borrow itself.
// ---------------------------------------------------------------------------

enum class ComputeImageBorrowOutcome : uint8_t {
    NotAttempted = 0,    // the precondition declined first
    NoCacheEntry,        // image_cache.find(key) == end()
    ExportNotAuthorized, // the entry exists; no producer published it to graphics
    ContentInvalid,      // published once, then invalidated by a later source change
    NoImage,             // entry without a VkImage (allocation released)
    AuthorityChanged,    // published, but no proof says guest memory still matches it
    Hit,
    Count,
};

constexpr const char* compute_image_borrow_outcome_name(ComputeImageBorrowOutcome outcome) {
    switch (outcome) {
    case ComputeImageBorrowOutcome::NotAttempted:        return "not_attempted";
    case ComputeImageBorrowOutcome::NoCacheEntry:        return "no_cache_entry";
    case ComputeImageBorrowOutcome::ExportNotAuthorized: return "export_unpublished";
    case ComputeImageBorrowOutcome::ContentInvalid:      return "content_invalid";
    case ComputeImageBorrowOutcome::NoImage:             return "no_image";
    case ComputeImageBorrowOutcome::AuthorityChanged:    return "authority_changed";
    case ComputeImageBorrowOutcome::Hit:                 return "hit";
    case ComputeImageBorrowOutcome::Count:               break;
    }
    return "?";
}

// What the three `*_unchanged` predicates saw, recorded ONLY for AuthorityChanged. Every field is
// the value the production code already computed -- nothing here evaluates a predicate the borrow
// would not have evaluated, because a query that the borrow short-circuits past is a query whose
// cost this census would then be inventing.
//
// `journal_armed` is the field that separates two very different worlds and is the reason this
// struct exists at all. `guest_gpu_writes_since` answers Unknown both when the per-thread submit
// journal is not armed on the consuming thread AND when it is armed but the export snapshot belongs
// to an earlier submit, and the caller cannot tell those apart (`gpu_execute.hpp` says so
// explicitly). The first would mean the borrow is structurally unreachable from wherever graphics
// materializes textures; the second would mean producer and consumer simply do not meet inside one
// submit. Those have different fixes, so they get different counters.
struct ComputeImageBorrowObservation {
    ComputeImageBorrowOutcome outcome = ComputeImageBorrowOutcome::NotAttempted;
    bool journal_armed = false;    // guest_gpu_write_tracking_active() on THIS thread
    uint8_t submit_query = 0;      // prosper::gpu::GuestGpuWriteQuery ordinal
    bool watch_present = false;    // the entry carries an armed page watch
    uint8_t watch_query = 0;       // prosper::host::GuestWriteWatchQuery ordinal; watch_present only
    bool exact_mirror_supported = false; // the byte-mirror fallback exists (Windows only)

    // Filled only when the lookup found nothing AND the report is enabled: the cache was scanned
    // for an entry at the same guest address, and `no_entry_field_diff_mask` names the fields the
    // nearest such entry disagreed on. The importer tries two keys, so the caller records this once
    // from the FINAL observation rather than per call.
    bool no_entry_scanned = false;
    bool no_entry_same_addr = false;
    uint32_t no_entry_field_diff_mask = 0;
};

// The cache key's fields, so a NoCacheEntry can say WHICH field a same-address entry disagreed on
// rather than only that the lookup failed. Order matches `ComputeImageCacheKey`'s declaration;
// `live_compute.cpp` owns the comparison because the key type is private to it.
enum class ComputeImageKeyField : uint8_t {
    HostData = 0,
    GuestBytes,
    ResourceBytes,
    Width,
    Height,
    Depth,
    Format,
    Components,
    TileMode,
    ImgDim,
    LinearRowPitch,
    LayerStride,
    LayerMipOffset,
    MipTailOffset,
    MipTailBytes,
    MipTailX,
    MipTailY,
    VkFormat,
    Storage,
    InMipTail,
    Srgb,
    DepthCompare,
    MipLevels,
    Count,
};

constexpr const char* compute_image_key_field_name(ComputeImageKeyField field) {
    switch (field) {
    case ComputeImageKeyField::HostData:       return "host_data";
    case ComputeImageKeyField::GuestBytes:     return "guest_bytes";
    case ComputeImageKeyField::ResourceBytes:  return "resource_bytes";
    case ComputeImageKeyField::Width:          return "width";
    case ComputeImageKeyField::Height:         return "height";
    case ComputeImageKeyField::Depth:          return "depth";
    case ComputeImageKeyField::Format:         return "format";
    case ComputeImageKeyField::Components:     return "components";
    case ComputeImageKeyField::TileMode:       return "tile_mode";
    case ComputeImageKeyField::ImgDim:         return "img_dim";
    case ComputeImageKeyField::LinearRowPitch: return "linear_row_pitch";
    case ComputeImageKeyField::LayerStride:    return "layer_stride";
    case ComputeImageKeyField::LayerMipOffset: return "layer_mip_offset";
    case ComputeImageKeyField::MipTailOffset:  return "mip_tail_offset";
    case ComputeImageKeyField::MipTailBytes:   return "mip_tail_bytes";
    case ComputeImageKeyField::MipTailX:       return "mip_tail_x";
    case ComputeImageKeyField::MipTailY:       return "mip_tail_y";
    case ComputeImageKeyField::VkFormat:       return "vk_format";
    case ComputeImageKeyField::Storage:        return "storage";
    case ComputeImageKeyField::InMipTail:      return "in_mip_tail";
    case ComputeImageKeyField::Srgb:           return "srgb";
    case ComputeImageKeyField::DepthCompare:   return "depth_compare";
    case ComputeImageKeyField::MipLevels:      return "mip_levels";
    case ComputeImageKeyField::Count:          break;
    }
    return "?";
}

// ---------------------------------------------------------------------------
// (5) The producer's publish gate, decomposed the same way.
//
// `authorize_cached_image_export` is reached only through
// `native_exact_storage && unique && cache_candidate && persistent && graphics_sampled_usage`, and
// a consumer that never sees an entry cannot distinguish "the producer declined to publish" from
// "the key differs". Counting the producer's terms is what makes (2) and (5) separable at all.
// ---------------------------------------------------------------------------

enum class ComputeImagePublishDecline : uint8_t {
    Authorized = 0,
    NotNativeExactStorage,
    NotUnique,          // an alias of another binding in the same dispatch
    NotCacheCandidate,
    NotPersistent,
    NoSampledUsage,     // the device would not create the image with SAMPLED usage
    ExportRefused,      // publish_eligible, but the cache had no valid entry to authorize
    Count,
};

struct ComputeImagePublishInputs {
    bool native_exact_storage = false;
    bool unique = false;
    bool cache_candidate = false;
    bool persistent = false;
    bool graphics_sampled_usage = false;
    bool export_authorized = false;   // authorize_cached_image_export found a valid entry
};

constexpr ComputeImagePublishDecline classify_compute_image_publish(
    const ComputeImagePublishInputs& in) {
    if (!in.native_exact_storage) return ComputeImagePublishDecline::NotNativeExactStorage;
    if (!in.unique) return ComputeImagePublishDecline::NotUnique;
    if (!in.cache_candidate) return ComputeImagePublishDecline::NotCacheCandidate;
    if (!in.persistent) return ComputeImagePublishDecline::NotPersistent;
    if (!in.graphics_sampled_usage) return ComputeImagePublishDecline::NoSampledUsage;
    if (!in.export_authorized) return ComputeImagePublishDecline::ExportRefused;
    return ComputeImagePublishDecline::Authorized;
}

constexpr const char* compute_image_publish_decline_name(ComputeImagePublishDecline reason) {
    switch (reason) {
    case ComputeImagePublishDecline::Authorized:            return "authorized";
    case ComputeImagePublishDecline::NotNativeExactStorage: return "not_native_exact";
    case ComputeImagePublishDecline::NotUnique:             return "not_unique";
    case ComputeImagePublishDecline::NotCacheCandidate:     return "not_cache_candidate";
    case ComputeImagePublishDecline::NotPersistent:         return "not_persistent";
    case ComputeImagePublishDecline::NoSampledUsage:        return "no_sampled_usage";
    case ComputeImagePublishDecline::ExportRefused:         return "export_refused";
    case ComputeImagePublishDecline::Count:                 break;
    }
    return "?";
}

// ---------------------------------------------------------------------------
// The running totals.
// ---------------------------------------------------------------------------

struct ComputeImageBorrowCensusSnapshot {
    uint64_t declines[static_cast<size_t>(ComputeImageImportDecline::Count)] = {};
    uint64_t outcomes[static_cast<size_t>(ComputeImageBorrowOutcome::Count)] = {};
    uint64_t publishes[static_cast<size_t>(ComputeImagePublishDecline::Count)] = {};
    uint64_t key_field_diffs[static_cast<size_t>(ComputeImageKeyField::Count)] = {};

    uint64_t imports = 0;          // == sum(declines)
    uint64_t alias_retries = 0;    // the format-alias key was tried after the exact key missed
    uint64_t lease_failures = 0;   // borrowed, then the lease allocation threw
    uint64_t hit_bytes = 0;        // guest bytes NOT decoded, because the borrow hit
    uint64_t miss_bytes = 0;       // guest bytes the graphics decode had to read instead

    // (2), refined: of the misses that found no entry, how many had an entry at the same guest
    // address under a different key. `no_entry_same_addr == 0` means the producer is absent, not
    // that the key disagrees -- which is the whole distinction `STRAY_STATUS.md` guessed at.
    uint64_t no_entry_scans = 0;
    uint64_t no_entry_same_addr = 0;

    // (4), refined.
    uint64_t authority_journal_overlap = 0;      // a real guest write covered the range
    uint64_t authority_journal_unarmed = 0;      // the journal is not armed on the consuming thread
    uint64_t authority_journal_cross_submit = 0; // armed, but the export is from an earlier submit
    uint64_t authority_watch_absent = 0;
    uint64_t authority_watch_dirty = 0;
    uint64_t authority_watch_unknown = 0;

    // (6). Recorded by the renderer after `import_live_compute_storage_image` returned true.
    uint64_t renderer_accepted = 0;
    uint64_t renderer_rejected = 0;

    uint64_t publish_evaluated = 0;  // == sum(publishes)
};

class ComputeImageBorrowCensus {
public:
    void record_import(ComputeImageImportDecline reason) {
        bump(imports_);
        bump(declines_[static_cast<size_t>(reason)]);
    }
    void record_alias_retry() { bump(alias_retries_); }
    void record_lease_failure() { bump(lease_failures_); }

    void record_outcome(const ComputeImageBorrowObservation& observation, uint64_t guest_bytes) {
        bump(outcomes_[static_cast<size_t>(observation.outcome)]);
        if (observation.outcome == ComputeImageBorrowOutcome::Hit) {
            add(hit_bytes_, guest_bytes);
            return;
        }
        add(miss_bytes_, guest_bytes);
        if (observation.outcome != ComputeImageBorrowOutcome::AuthorityChanged) return;
        // GuestGpuWriteQuery: 0 Unchanged, 1 Overlap, 2 Unknown. Unchanged cannot reach here (it is
        // the hit), so the partition below is over Overlap and Unknown only.
        if (observation.submit_query == 1) bump(authority_journal_overlap_);
        else if (!observation.journal_armed) bump(authority_journal_unarmed_);
        else bump(authority_journal_cross_submit_);
        if (!observation.watch_present) bump(authority_watch_absent_);
        else if (observation.watch_query == 1) bump(authority_watch_dirty_);
        else bump(authority_watch_unknown_);
    }

    void record_no_entry_scan(bool same_address_entry, uint32_t field_diff_mask) {
        bump(no_entry_scans_);
        if (!same_address_entry) return;
        bump(no_entry_same_addr_);
        for (size_t i = 0; i < static_cast<size_t>(ComputeImageKeyField::Count); ++i)
            if (field_diff_mask & (1u << i)) bump(key_field_diffs_[i]);
    }

    void record_renderer_verdict(bool accepted) {
        bump(accepted ? renderer_accepted_ : renderer_rejected_);
    }

    void record_publish(ComputeImagePublishDecline reason) {
        bump(publish_evaluated_);
        bump(publishes_[static_cast<size_t>(reason)]);
    }

    ComputeImageBorrowCensusSnapshot snapshot() const {
        ComputeImageBorrowCensusSnapshot out;
        for (size_t i = 0; i < static_cast<size_t>(ComputeImageImportDecline::Count); ++i)
            out.declines[i] = get(declines_[i]);
        for (size_t i = 0; i < static_cast<size_t>(ComputeImageBorrowOutcome::Count); ++i)
            out.outcomes[i] = get(outcomes_[i]);
        for (size_t i = 0; i < static_cast<size_t>(ComputeImagePublishDecline::Count); ++i)
            out.publishes[i] = get(publishes_[i]);
        for (size_t i = 0; i < static_cast<size_t>(ComputeImageKeyField::Count); ++i)
            out.key_field_diffs[i] = get(key_field_diffs_[i]);
        out.imports = get(imports_);
        out.alias_retries = get(alias_retries_);
        out.lease_failures = get(lease_failures_);
        out.hit_bytes = get(hit_bytes_);
        out.miss_bytes = get(miss_bytes_);
        out.no_entry_scans = get(no_entry_scans_);
        out.no_entry_same_addr = get(no_entry_same_addr_);
        out.authority_journal_overlap = get(authority_journal_overlap_);
        out.authority_journal_unarmed = get(authority_journal_unarmed_);
        out.authority_journal_cross_submit = get(authority_journal_cross_submit_);
        out.authority_watch_absent = get(authority_watch_absent_);
        out.authority_watch_dirty = get(authority_watch_dirty_);
        out.authority_watch_unknown = get(authority_watch_unknown_);
        out.renderer_accepted = get(renderer_accepted_);
        out.renderer_rejected = get(renderer_rejected_);
        out.publish_evaluated = get(publish_evaluated_);
        return out;
    }

private:
    static void bump(std::atomic<uint64_t>& counter) {
        counter.fetch_add(1, std::memory_order_relaxed);
    }
    static void add(std::atomic<uint64_t>& counter, uint64_t value) {
        counter.fetch_add(value, std::memory_order_relaxed);
    }
    static uint64_t get(const std::atomic<uint64_t>& counter) {
        return counter.load(std::memory_order_relaxed);
    }

    std::atomic<uint64_t> declines_[static_cast<size_t>(ComputeImageImportDecline::Count)]{};
    std::atomic<uint64_t> outcomes_[static_cast<size_t>(ComputeImageBorrowOutcome::Count)]{};
    std::atomic<uint64_t> publishes_[static_cast<size_t>(ComputeImagePublishDecline::Count)]{};
    std::atomic<uint64_t> key_field_diffs_[static_cast<size_t>(ComputeImageKeyField::Count)]{};
    std::atomic<uint64_t> imports_{0};
    std::atomic<uint64_t> alias_retries_{0};
    std::atomic<uint64_t> lease_failures_{0};
    std::atomic<uint64_t> hit_bytes_{0};
    std::atomic<uint64_t> miss_bytes_{0};
    std::atomic<uint64_t> no_entry_scans_{0};
    std::atomic<uint64_t> no_entry_same_addr_{0};
    std::atomic<uint64_t> authority_journal_overlap_{0};
    std::atomic<uint64_t> authority_journal_unarmed_{0};
    std::atomic<uint64_t> authority_journal_cross_submit_{0};
    std::atomic<uint64_t> authority_watch_absent_{0};
    std::atomic<uint64_t> authority_watch_dirty_{0};
    std::atomic<uint64_t> authority_watch_unknown_{0};
    std::atomic<uint64_t> renderer_accepted_{0};
    std::atomic<uint64_t> renderer_rejected_{0};
    std::atomic<uint64_t> publish_evaluated_{0};
};

// Four lines, each a RUNNING TOTAL carrying its own denominator. Zero-valued buckets are omitted so
// the line stays readable, EXCEPT the denominators and the outcome partition, which always print --
// a bucket that is absent because it is zero and a bucket that is absent because the instrument
// never reached it look identical otherwise, and this project has already lost a night to exactly
// that ambiguity (instrument trap 256).
inline size_t format_compute_image_borrow_census(
    const ComputeImageBorrowCensusSnapshot& census, char* output, size_t capacity) {
    if (!output || !capacity) return 0;
    // Terminate up front: with capacity 1 there is room for the NUL and nothing else, and every
    // emit below then declines. A caller that trusts the returned length would otherwise read an
    // unterminated buffer on exactly the path where the report is least useful.
    output[0] = '\0';
    size_t used = 0;
    // snprintf writes at most `room` characters plus a NUL and returns the length it WOULD have
    // written, so the cursor must advance by the clamped value or a truncated line silently moves
    // `used` past the end of the buffer.
    const auto emit = [&](const char* format, auto... args) {
        if (used + 1 >= capacity) return;
        const size_t room = capacity - used - 1;
        const int written = std::snprintf(output + used, room + 1, format, args...);
        if (written < 0) return;
        used += static_cast<size_t>(written) <= room ? static_cast<size_t>(written) : room;
    };
    const auto percent = [](uint64_t part, uint64_t whole) {
        return whole ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
    };
    const double mib = 1024.0 * 1024.0;

    emit("[compute-borrow-census] running totals -- imports=%llu",
         (unsigned long long)census.imports);
    for (size_t i = 1; i < static_cast<size_t>(ComputeImageImportDecline::Count); ++i)
        if (census.declines[i])
            emit(" %s=%llu",
                 compute_image_import_decline_name(static_cast<ComputeImageImportDecline>(i)),
                 (unsigned long long)census.declines[i]);
    emit(" accepted=%llu (%.1f%%)\n",
         (unsigned long long)census.declines[0],
         percent(census.declines[0], census.imports));

    const uint64_t attempted =
        census.declines[static_cast<size_t>(ComputeImageImportDecline::None)];
    emit("[compute-borrow-census] running totals -- attempted=%llu", (unsigned long long)attempted);
    for (size_t i = 1; i < static_cast<size_t>(ComputeImageBorrowOutcome::Count); ++i)
        emit(" %s=%llu (%.1f%%)",
             compute_image_borrow_outcome_name(static_cast<ComputeImageBorrowOutcome>(i)),
             (unsigned long long)census.outcomes[i], percent(census.outcomes[i], attempted));
    emit(" alias_retry=%llu lease_fail=%llu spared=%.1f MiB decoded=%.1f MiB\n",
         (unsigned long long)census.alias_retries, (unsigned long long)census.lease_failures,
         static_cast<double>(census.hit_bytes) / mib,
         static_cast<double>(census.miss_bytes) / mib);

    const uint64_t authority =
        census.outcomes[static_cast<size_t>(ComputeImageBorrowOutcome::AuthorityChanged)];
    emit("[compute-borrow-census] running totals -- authority_changed=%llu journal_overlap=%llu "
         "journal_unarmed=%llu journal_cross_submit=%llu watch_absent=%llu watch_dirty=%llu "
         "watch_unknown=%llu | no_entry_scans=%llu same_addr=%llu",
         (unsigned long long)authority,
         (unsigned long long)census.authority_journal_overlap,
         (unsigned long long)census.authority_journal_unarmed,
         (unsigned long long)census.authority_journal_cross_submit,
         (unsigned long long)census.authority_watch_absent,
         (unsigned long long)census.authority_watch_dirty,
         (unsigned long long)census.authority_watch_unknown,
         (unsigned long long)census.no_entry_scans,
         (unsigned long long)census.no_entry_same_addr);
    for (size_t i = 0; i < static_cast<size_t>(ComputeImageKeyField::Count); ++i)
        if (census.key_field_diffs[i])
            emit(" %s=%llu", compute_image_key_field_name(static_cast<ComputeImageKeyField>(i)),
                 (unsigned long long)census.key_field_diffs[i]);
    emit("%s", "\n");

    emit("[compute-borrow-census] running totals -- publish_evaluated=%llu",
         (unsigned long long)census.publish_evaluated);
    for (size_t i = 0; i < static_cast<size_t>(ComputeImagePublishDecline::Count); ++i)
        emit(" %s=%llu (%.1f%%)",
             compute_image_publish_decline_name(static_cast<ComputeImagePublishDecline>(i)),
             (unsigned long long)census.publishes[i],
             percent(census.publishes[i], census.publish_evaluated));
    emit(" | renderer_accepted=%llu renderer_rejected=%llu\n",
         (unsigned long long)census.renderer_accepted,
         (unsigned long long)census.renderer_rejected);
    return used;
}

} // namespace prosper::frontend
