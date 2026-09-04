// #3307. The compute->graphics image borrow census.
//
// The load-bearing test here is the FIRST one, and it is not a census test at all. Replacing
// `import_live_compute_storage_image`'s single `||` chain with a classifier that names the failing
// term is only safe if the two accept exactly the same resources; a term silently dropped in that
// rewrite would turn a diagnostic change into a correctness change, and it would show up as a
// borrowed image handed to a descriptor whose shape the chain used to refuse. So the equivalence is
// asserted over the COMPLETE boolean product of the eleven terms (2^11 = 2048 cases) against the
// chain written out independently below, rather than over cases chosen by whoever did the rewrite.
//
// The rest pin the partitions: every import lands in exactly one decline bucket, every attempted
// borrow in exactly one outcome bucket, every publish in exactly one publish bucket, and the
// formatted report carries its own denominators.

#include "shared/compute/compute_image_borrow_census.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

using prosper::frontend::ComputeImageBorrowCensus;
using prosper::frontend::ComputeImageBorrowObservation;
using prosper::frontend::ComputeImageBorrowOutcome;
using prosper::frontend::ComputeImageImportDecline;
using prosper::frontend::ComputeImageImportInputs;
using prosper::frontend::ComputeImageKeyField;
using prosper::frontend::ComputeImagePublishDecline;
using prosper::frontend::ComputeImagePublishInputs;
using prosper::frontend::classify_compute_image_import;
using prosper::frontend::classify_compute_image_publish;
using prosper::frontend::format_compute_image_borrow_census;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

// The precondition exactly as `import_live_compute_storage_image` used to spell it, transcribed
// from the shape of that expression rather than from the classifier. `true` means "declined".
static bool legacy_import_chain_declines(const ComputeImageImportInputs& in) {
    if (!in.have_context || !in.have_gpu_addr || !in.texture_class ||
        in.host_data || !in.guest_bytes_in_range ||
        !in.shape_supported ||
        !in.single_mip_level || in.in_mip_tail ||
        in.srgb ||
        in.depth_compare)
        return true;
    return !in.native_format_defined;
}

static ComputeImageImportInputs import_inputs_from_bits(unsigned bits) {
    ComputeImageImportInputs in;
    in.have_context          = (bits >> 0) & 1u;
    in.have_gpu_addr         = (bits >> 1) & 1u;
    in.texture_class         = (bits >> 2) & 1u;
    in.host_data             = (bits >> 3) & 1u;
    in.guest_bytes_in_range  = (bits >> 4) & 1u;
    in.shape_supported       = (bits >> 5) & 1u;
    in.single_mip_level      = (bits >> 6) & 1u;
    in.in_mip_tail           = (bits >> 7) & 1u;
    in.srgb                  = (bits >> 8) & 1u;
    in.depth_compare         = (bits >> 9) & 1u;
    in.native_format_defined = (bits >> 10) & 1u;
    return in;
}

static ComputeImageImportInputs accepting_import_inputs() {
    ComputeImageImportInputs in;
    in.have_context = true;
    in.have_gpu_addr = true;
    in.texture_class = true;
    in.host_data = false;
    in.guest_bytes_in_range = true;
    in.shape_supported = true;
    in.single_mip_level = true;
    in.in_mip_tail = false;
    in.srgb = false;
    in.depth_compare = false;
    in.native_format_defined = true;
    return in;
}

int main() {
    // ---- 1. Equivalence over the complete boolean product ---------------------------------
    constexpr unsigned kImportTerms = 11;
    unsigned accepted = 0, declined = 0;
    for (unsigned bits = 0; bits < (1u << kImportTerms); ++bits) {
        const ComputeImageImportInputs in = import_inputs_from_bits(bits);
        const bool classifier_declines =
            classify_compute_image_import(in) != ComputeImageImportDecline::None;
        CHECK(classifier_declines == legacy_import_chain_declines(in));
        if (classifier_declines) ++declined; else ++accepted;
    }
    // The sweep is only evidence if BOTH verdicts occur in it: an accept-only or decline-only sweep
    // agrees with any classifier that answers one constant, which is the vacuity shape this
    // project keeps rediscovering. Exactly one of 2048 assignments accepts, and knowing which
    // number that is makes the arm falsifiable rather than merely non-empty.
    CHECK(accepted == 1);
    CHECK(declined == (1u << kImportTerms) - 1);

    // ---- 2. Each term declines under its own name, one at a time ---------------------------
    struct TermCase { const char* what; ComputeImageImportInputs in; ComputeImageImportDecline want; };
    auto without = [](void (*mutate)(ComputeImageImportInputs&)) {
        ComputeImageImportInputs in = accepting_import_inputs();
        mutate(in);
        return in;
    };
    const TermCase cases[] = {
        {"context",   without([](ComputeImageImportInputs& i) { i.have_context = false; }),
         ComputeImageImportDecline::NoContext},
        {"addr",      without([](ComputeImageImportInputs& i) { i.have_gpu_addr = false; }),
         ComputeImageImportDecline::NoGuestAddress},
        {"class",     without([](ComputeImageImportInputs& i) { i.texture_class = false; }),
         ComputeImageImportDecline::NotTextureClass},
        {"host",      without([](ComputeImageImportInputs& i) { i.host_data = true; }),
         ComputeImageImportDecline::HostBackedData},
        {"bytes",     without([](ComputeImageImportInputs& i) { i.guest_bytes_in_range = false; }),
         ComputeImageImportDecline::GuestByteRange},
        {"shape",     without([](ComputeImageImportInputs& i) { i.shape_supported = false; }),
         ComputeImageImportDecline::Shape},
        {"mips",      without([](ComputeImageImportInputs& i) { i.single_mip_level = false; }),
         ComputeImageImportDecline::MipLevels},
        {"mip_tail",  without([](ComputeImageImportInputs& i) { i.in_mip_tail = true; }),
         ComputeImageImportDecline::MipTail},
        {"srgb",      without([](ComputeImageImportInputs& i) { i.srgb = true; }),
         ComputeImageImportDecline::Srgb},
        {"depth_cmp", without([](ComputeImageImportInputs& i) { i.depth_compare = true; }),
         ComputeImageImportDecline::DepthCompare},
        {"format",    without([](ComputeImageImportInputs& i) { i.native_format_defined = false; }),
         ComputeImageImportDecline::NativeFormat},
    };
    for (const TermCase& c : cases) {
        const ComputeImageImportDecline got = classify_compute_image_import(c.in);
        if (got != c.want)
            std::fprintf(stderr, "FAIL %s: term %s reported %s, wanted %s\n", __func__, c.what,
                         prosper::frontend::compute_image_import_decline_name(got),
                         prosper::frontend::compute_image_import_decline_name(c.want));
        CHECK(got == c.want);
    }
    CHECK(classify_compute_image_import(accepting_import_inputs()) ==
          ComputeImageImportDecline::None);
    // Every enumerator is reachable, so no bucket can be a permanent zero that a reader mistakes
    // for "this never happens".
    CHECK(sizeof(cases) / sizeof(cases[0]) ==
          static_cast<size_t>(ComputeImageImportDecline::Count) - 1);

    // ---- 3. The publish gate, same shape --------------------------------------------------
    ComputeImagePublishInputs publish;
    CHECK(classify_compute_image_publish(publish) ==
          ComputeImagePublishDecline::NotNativeExactStorage);
    publish.native_exact_storage = true;
    CHECK(classify_compute_image_publish(publish) == ComputeImagePublishDecline::NotUnique);
    publish.unique = true;
    CHECK(classify_compute_image_publish(publish) == ComputeImagePublishDecline::NotCacheCandidate);
    publish.cache_candidate = true;
    CHECK(classify_compute_image_publish(publish) == ComputeImagePublishDecline::NotPersistent);
    publish.persistent = true;
    CHECK(classify_compute_image_publish(publish) == ComputeImagePublishDecline::NoSampledUsage);
    publish.graphics_sampled_usage = true;
    // Eligible in every respect and the cache still had nothing valid to publish: the entry was
    // evicted or invalidated between writeback and the publish loop. This is the arm that would
    // otherwise be indistinguishable from an ineligible binding.
    CHECK(classify_compute_image_publish(publish) == ComputeImagePublishDecline::ExportRefused);
    publish.export_authorized = true;
    CHECK(classify_compute_image_publish(publish) == ComputeImagePublishDecline::Authorized);

    // ---- 4. The authority partition -------------------------------------------------------
    // GuestGpuWriteQuery: 0 Unchanged, 1 Overlap, 2 Unknown. GuestWriteWatchQuery: 0 Unchanged,
    // 1 Dirty, 2 Unknown. Unchanged cannot reach AuthorityChanged on either axis -- it is the hit.
    {
        ComputeImageBorrowCensus census;
        ComputeImageBorrowObservation overlap;
        overlap.outcome = ComputeImageBorrowOutcome::AuthorityChanged;
        overlap.submit_query = 1;          // Overlap: a real guest write covered the range
        overlap.journal_armed = true;
        census.record_outcome(overlap, 1024);

        ComputeImageBorrowObservation unarmed;
        unarmed.outcome = ComputeImageBorrowOutcome::AuthorityChanged;
        unarmed.submit_query = 2;          // Unknown, and the journal is not armed here
        unarmed.journal_armed = false;
        census.record_outcome(unarmed, 2048);

        ComputeImageBorrowObservation cross;
        cross.outcome = ComputeImageBorrowOutcome::AuthorityChanged;
        cross.submit_query = 2;            // Unknown, but armed: the export is from another submit
        cross.journal_armed = true;
        cross.watch_present = true;
        cross.watch_query = 1;             // Dirty
        census.record_outcome(cross, 4096);

        const auto snapshot = census.snapshot();
        CHECK(snapshot.authority_journal_overlap == 1);
        CHECK(snapshot.authority_journal_unarmed == 1);
        CHECK(snapshot.authority_journal_cross_submit == 1);
        CHECK(snapshot.authority_watch_absent == 2);
        CHECK(snapshot.authority_watch_dirty == 1);
        CHECK(snapshot.authority_watch_unknown == 0);
        CHECK(snapshot.miss_bytes == 1024 + 2048 + 4096);
        CHECK(snapshot.hit_bytes == 0);
        // The journal partition and the watch partition each sum to the denominator, independently.
        const uint64_t authority =
            snapshot.outcomes[static_cast<size_t>(ComputeImageBorrowOutcome::AuthorityChanged)];
        CHECK(authority == 3);
        CHECK(snapshot.authority_journal_overlap + snapshot.authority_journal_unarmed +
              snapshot.authority_journal_cross_submit == authority);
        CHECK(snapshot.authority_watch_absent + snapshot.authority_watch_dirty +
              snapshot.authority_watch_unknown == authority);
    }

    // ---- 5. A hit is counted as spared bytes, and only a hit -------------------------------
    {
        ComputeImageBorrowCensus census;
        ComputeImageBorrowObservation hit;
        hit.outcome = ComputeImageBorrowOutcome::Hit;
        census.record_outcome(hit, 66846720);
        ComputeImageBorrowObservation miss;
        miss.outcome = ComputeImageBorrowOutcome::NoCacheEntry;
        census.record_outcome(miss, 66846720);
        const auto snapshot = census.snapshot();
        CHECK(snapshot.hit_bytes == 66846720);
        CHECK(snapshot.miss_bytes == 66846720);
        // A NoCacheEntry must NOT fall into the authority breakdown: those counters exist to
        // explain the fourth gate, and folding a second gate's misses into them is how a
        // partition stops being one.
        CHECK(snapshot.authority_journal_overlap == 0);
        CHECK(snapshot.authority_journal_unarmed == 0);
        CHECK(snapshot.authority_journal_cross_submit == 0);
        CHECK(snapshot.authority_watch_absent == 0);
    }

    // ---- 6. The near-miss key scan ---------------------------------------------------------
    {
        ComputeImageBorrowCensus census;
        census.record_no_entry_scan(false, 0);   // nothing at this address at all
        const uint32_t tile_and_pitch =
            (1u << static_cast<uint32_t>(ComputeImageKeyField::TileMode)) |
            (1u << static_cast<uint32_t>(ComputeImageKeyField::LinearRowPitch));
        census.record_no_entry_scan(true, tile_and_pitch);
        const auto snapshot = census.snapshot();
        CHECK(snapshot.no_entry_scans == 2);
        CHECK(snapshot.no_entry_same_addr == 1);
        CHECK(snapshot.key_field_diffs[static_cast<size_t>(ComputeImageKeyField::TileMode)] == 1);
        CHECK(snapshot.key_field_diffs[static_cast<size_t>(ComputeImageKeyField::LinearRowPitch)] == 1);
        CHECK(snapshot.key_field_diffs[static_cast<size_t>(ComputeImageKeyField::VkFormat)] == 0);
    }

    // ---- 7. The report carries its denominators and every outcome bucket -------------------
    {
        ComputeImageBorrowCensus census;
        census.record_import(ComputeImageImportDecline::None);
        census.record_import(ComputeImageImportDecline::Shape);
        ComputeImageBorrowObservation miss;
        miss.outcome = ComputeImageBorrowOutcome::AuthorityChanged;
        miss.submit_query = 2;
        census.record_outcome(miss, 4096);
        census.record_alias_retry();
        census.record_renderer_verdict(false);
        census.record_publish(ComputeImagePublishDecline::NoSampledUsage);

        char line[2048];
        const size_t used = format_compute_image_borrow_census(census.snapshot(), line, sizeof line);
        CHECK(used != 0);
        CHECK(used < sizeof line);
        CHECK(line[used] == '\0');
        const std::string text(line, used);
        CHECK(text.find("imports=2") != std::string::npos);
        CHECK(text.find("shape=1") != std::string::npos);
        CHECK(text.find("accepted=1 (50.0%)") != std::string::npos);
        CHECK(text.find("attempted=1") != std::string::npos);
        // Present even at zero: an outcome bucket that vanishes when empty cannot be told from one
        // the instrument never reached. That ambiguity is instrument trap 256 in this repository.
        CHECK(text.find("no_cache_entry=0") != std::string::npos);
        CHECK(text.find("hit=0 (0.0%)") != std::string::npos);
        CHECK(text.find("authority_changed=1 (100.0%)") != std::string::npos);
        CHECK(text.find("journal_unarmed=1") != std::string::npos);
        CHECK(text.find("alias_retry=1") != std::string::npos);
        CHECK(text.find("publish_evaluated=1") != std::string::npos);
        CHECK(text.find("no_sampled_usage=1 (100.0%)") != std::string::npos);
        CHECK(text.find("renderer_rejected=1") != std::string::npos);
        // Four lines, one report.
        CHECK(std::count(text.begin(), text.end(), '\n') == 4);
    }

    // ---- 8. Truncation is bounded, not corrupting ------------------------------------------
    {
        ComputeImageBorrowCensus census;
        census.record_import(ComputeImageImportDecline::None);
        for (size_t capacity = 1; capacity <= 96; ++capacity) {
            char small[128];
            std::memset(small, 0x7f, sizeof small);
            const size_t used = format_compute_image_borrow_census(census.snapshot(), small, capacity);
            CHECK(used < capacity);
            CHECK(small[used] == '\0');
            // Nothing written past the buffer it was given.
            for (size_t i = capacity; i < sizeof small; ++i) CHECK(small[i] == 0x7f);
        }
        CHECK(format_compute_image_borrow_census(census.snapshot(), nullptr, 64) == 0);
        char one[1];
        CHECK(format_compute_image_borrow_census(census.snapshot(), one, 0) == 0);
    }

    if (failures) {
        std::fprintf(stderr, "compute_image_borrow_census: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("compute_image_borrow_census: all checks passed\n");
    return 0;
}
