// present_frame_rate.cpp — see present_frame_rate.hpp.
#include "gpu/present/present_frame_rate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace prosper::gpu {
namespace {

// FNV-1a's constants with an extra shift-xor per step. The mixing quality that matters here is only
// "two different frames rarely collide in 64 bits"; nothing cryptographic is claimed or needed.
inline void mix(uint64_t& h, uint64_t value) {
    h ^= value;
    h *= 0x100000001b3ull;
    h ^= h >> 29;
}

using RateClock = std::chrono::steady_clock;

// One epoch for every snapshot in the process, so two snapshots are always subtractable. Function-
// local so it is initialised on first use rather than in a static-init order nobody controls.
double now_seconds() {
    static const RateClock::time_point epoch = RateClock::now();
    return std::chrono::duration<double>(RateClock::now() - epoch).count();
}

std::mutex g_rate_mx;
FrameRateCounter g_rate;   // guarded by g_rate_mx

} // namespace

uint64_t frame_content_signature(const uint8_t* pixels, size_t bytes, uint32_t width, uint32_t height) {
    uint64_t h = 0xcbf29ce484222325ull;
    mix(h, static_cast<uint64_t>(bytes));
    mix(h, (static_cast<uint64_t>(width) << 32) | height);
    if (!pixels || bytes == 0) return h;

    for (size_t block = 0; block < bytes; block += kFrameSignatureBlockBytes) {
        const size_t take = std::min(kFrameSignatureBytesPerBlock, bytes - block);
        uint64_t words[2] = {0, 0};
        std::memcpy(words, pixels + block, take);
        // The offset is folded in so the signature is position-sensitive: two frames that hold the
        // same multiset of sampled windows in a different order are different frames.
        mix(h, static_cast<uint64_t>(block));
        mix(h, words[0]);
        mix(h, words[1]);
    }
    // The final partial block is sampled from its START above, which can leave the last few KiB
    // unread when `bytes` is not a multiple of the block size. Cover the tail explicitly so the
    // bottom-right of an image is never outside the signature.
    if (bytes >= kFrameSignatureBytesPerBlock) {
        uint64_t tail[2] = {0, 0};
        std::memcpy(tail, pixels + bytes - kFrameSignatureBytesPerBlock, kFrameSignatureBytesPerBlock);
        mix(h, tail[0]);
        mix(h, tail[1]);
    }
    return h;
}

void FrameRateCounter::observe(uint64_t signature, double at_seconds) {
    // The first publication is distinct by definition: there is nothing before it that it could be
    // a repeat of. That is why the rates below divide by `published`/`distinct` rather than by
    // (n - 1) — see frame_rate_since_first_publication for the window this pairs with.
    const bool changed = published_ == 0 || signature != last_signature_;
    if (published_ == 0) first_ = at_seconds;
    published_++;
    if (changed) {
        distinct_++;
        // Intervals are between consecutive DISTINCT frames, which is the whole point: the gap
        // between two publications of the same picture is not a frame time.
        if (have_distinct_) record_interval(at_seconds - last_distinct_);
        last_distinct_ = at_seconds;
        have_distinct_ = true;
    }
    last_signature_ = signature;
    last_ = at_seconds;
}

size_t FrameRateCounter::bucket_for(double seconds) const {
    if (seconds <= kIntervalMinSeconds) return 0;
    const double steps = std::log(seconds / kIntervalMinSeconds) / std::log(kIntervalGrowth);
    if (steps >= static_cast<double>(kIntervalBuckets - 1)) return kIntervalBuckets - 1;
    return static_cast<size_t>(steps) + 1;
}

void FrameRateCounter::record_interval(double seconds) {
    if (!(seconds > 0)) return;   // also rejects NaN
    const size_t bucket = bucket_for(seconds);
    interval_counts_[bucket]++;
    interval_seconds_[bucket] += seconds;
    interval_samples_++;
}

double FrameRateCounter::typical_interval_seconds() const {
    if (interval_samples_ == 0) return 0;   // no interval exists; NOT a fast frame time
    // The median BY COUNT. Weighting by frame rather than by time is exactly what makes this immune
    // to idling: a two-minute pause contributes one sample, not two minutes of pull.
    const uint64_t half = interval_samples_ / 2;
    uint64_t seen = 0;
    for (size_t i = 0; i < kIntervalBuckets; i++) {
        seen += interval_counts_[i];
        if (seen > half) {
            const double high =
                kIntervalMinSeconds * std::pow(kIntervalGrowth, static_cast<double>(i));
            if (i == 0) return high;
            const double low =
                kIntervalMinSeconds * std::pow(kIntervalGrowth, static_cast<double>(i - 1));
            // Geometric midpoint: the right centre for log-spaced edges.
            return std::sqrt(low * high);
        }
    }
    return 0;
}

double FrameRateCounter::active_seconds() const {
    const double typical = typical_interval_seconds();
    if (typical <= 0) return 0;
    const double cutoff = typical * kActiveIntervalMultiple;
    double active = 0;
    for (size_t i = 0; i < kIntervalBuckets; i++) {
        if (interval_counts_[i] == 0) continue;
        // Attribute a whole bucket by its own MEAN interval, so a bucket straddling the cutoff is
        // decided by where its intervals actually are rather than by its nominal edges.
        const double mean = interval_seconds_[i] / static_cast<double>(interval_counts_[i]);
        if (mean <= cutoff) active += interval_seconds_[i];
    }
    return active;
}

void FrameRateCounter::reset() { *this = FrameRateCounter{}; }

namespace {

FrameRate make_rate(uint64_t published, uint64_t distinct, double window_seconds) {
    FrameRate r;
    r.published = published;
    r.distinct = distinct;
    r.window_seconds = window_seconds > 0 ? window_seconds : 0;
    if (published != 0)
        r.distinct_fraction = static_cast<double>(distinct) / static_cast<double>(published);
    if (published == 0 || window_seconds <= 0) return r;   // measured stays false
    r.measured = true;
    r.presented_fps = static_cast<double>(published) / window_seconds;
    r.distinct_fps = static_cast<double>(distinct) / window_seconds;
    return r;
}

} // namespace

FrameRate frame_rate_since_first_publication(const PresentRateSnapshot& s) {
    if (s.published == 0) return make_rate(0, 0, 0);
    FrameRate r = make_rate(s.published, s.distinct, s.now_seconds - s.first_publication_seconds);
    r.interval_samples = s.interval_samples;
    r.typical_interval_seconds = s.typical_interval_seconds;
    // Fewer than two distinct frames means there is no interval, so there is no typical rate. That
    // is reported as ABSENT rather than as zero: "0.0 fps" is a measurement and this is not one.
    if (s.typical_interval_seconds > 0) {
        r.typical_measured = true;
        r.typical_fps = 1.0 / s.typical_interval_seconds;
    }
    if (r.window_seconds > 0) {
        r.active_fraction = std::min(1.0, s.active_seconds / r.window_seconds);
        // Measured, not defaulted -- this is the one constructor that can say so. A 0 here is the
        // verdict "the run produced nothing"; a 0 from frame_rate_between is an unfilled field.
        r.active_fraction_measured = true;
    }
    return r;
}

FrameRate frame_rate_between(const PresentRateSnapshot& earlier, const PresentRateSnapshot& later) {
    // A backwards counter means reset_present_rate() ran between the two readings. There is no
    // honest window to report, so report none rather than a negative or wrapped one.
    if (later.published < earlier.published || later.distinct < earlier.distinct)
        return make_rate(0, 0, 0);
    return make_rate(later.published - earlier.published, later.distinct - earlier.distinct,
                     later.now_seconds - earlier.now_seconds);
}

std::string format_frame_rate(const FrameRate& rate) {
    char text[512];
    if (!rate.measured) {
        std::snprintf(text, sizeof text,
                      "not measured (%llu published, %llu distinct, %.3f s window)",
                      static_cast<unsigned long long>(rate.published),
                      static_cast<unsigned long long>(rate.distinct), rate.window_seconds);
        return text;
    }
    // The headline and its qualifier on the first line and nothing else; everything the headline was
    // derived from on the second. A reader who quotes the first line quotes a rate that idling
    // cannot have dragged down, and a percentage that stops it being quoted bare.
    if (!rate.typical_measured) {
        std::snprintf(text, sizeof text,
                      "-- fps while producing frames, 0%% of the %.1f s run active\n"
                      "     the title produced %llu distinct frame(s) in %llu publications "
                      "(presented %.1f fps) -- there is no framerate to report",
                      rate.window_seconds,
                      static_cast<unsigned long long>(rate.distinct),
                      static_cast<unsigned long long>(rate.published), rate.presented_fps);
        return text;
    }
    std::snprintf(text, sizeof text,
                  "%.1f fps while producing frames, %.0f%% of the %.1f s run active\n"
                  "     %llu distinct of %llu published; run average %.1f fps; presented %.1f fps",
                  rate.typical_fps, rate.active_fraction * 100.0, rate.window_seconds,
                  static_cast<unsigned long long>(rate.distinct),
                  static_cast<unsigned long long>(rate.published),
                  rate.distinct_fps, rate.presented_fps);
    return text;
}

std::string format_frame_rate_short(const FrameRate& rate, uint32_t width, uint32_t height) {
    char text[160];
    if (!rate.measured || (!rate.typical_measured && rate.distinct <= 1)) {
        // "0% active" asserts that the run produced nothing, so it may only be printed when the
        // fraction was actually measured. A differenced window never measures it (#3027), and one
        // still second is not evidence about the run.
        if (rate.active_fraction_measured)
            std::snprintf(text, sizeof text, "-- fps  0%% active  %ux%u", width, height);
        else
            std::snprintf(text, sizeof text, "-- fps  %ux%u", width, height);
        return text;
    }
    if (!rate.typical_measured) {
        // A measured window with no interval histogram behind it (frame_rate_between). The average
        // is the only rate available; it is labelled so it cannot be read as the typical one.
        std::snprintf(text, sizeof text, "%.1f fps avg  %ux%u", rate.distinct_fps, width, height);
        return text;
    }
    std::snprintf(text, sizeof text, "%.1f fps  %.0f%% active  %ux%u",
                  rate.typical_fps, rate.active_fraction * 100.0, width, height);
    return text;
}

bool frame_rate_is_mostly_unchanged(const FrameRate& rate, uint64_t min_published,
                                   double max_distinct_fraction) {
    return rate.published >= min_published && rate.distinct_fraction < max_distinct_fraction;
}

UnchangedPicture unchanged_picture(const FrameRate& window, const PresentRateSnapshot& run,
                                   uint64_t min_published, double max_distinct_fraction) {
    UnchangedPicture out;
    // The run figures are read whatever the verdict, so a caller that keeps the struct around has
    // them for a later line too, and so the `changing` case is not a different shape of value.
    const FrameRate run_rate = frame_rate_since_first_publication(run);
    out.run_typical_measured = run_rate.typical_measured;
    out.run_typical_fps = run_rate.typical_fps;
    out.run_active_fraction = run_rate.active_fraction;
    out.run_distinct = run_rate.distinct;
    if (!frame_rate_is_mostly_unchanged(window, min_published, max_distinct_fraction)) return out;
    // The window says the picture stopped; only the run can say whether it ever started. An absent
    // typical rate over the WHOLE run is the R-Type Delta (#2783) shape; a measured one means the
    // title produced frames and is now sitting on a picture, which is correct behaviour.
    out.change = run_rate.typical_measured ? PictureChange::static_picture
                                           : PictureChange::nothing_produced;
    return out;
}

std::string format_unchanged_picture(const UnchangedPicture& picture) {
    if (picture.change == PictureChange::changing) return {};
    char text[192];
    if (picture.change == PictureChange::nothing_produced)
        // "--", never "0.0": there is no rate, and the module's whole argument is that reporting one
        // would be a measurement where none exists.
        std::snprintf(text, sizeof text,
                      "picture not changing; run so far -- fps, 0%% active, %llu distinct "
                      "(nothing produced yet)",
                      static_cast<unsigned long long>(picture.run_distinct));
    else
        std::snprintf(text, sizeof text,
                      "picture not changing; run so far %.1f fps, %.0f%% active, %llu distinct",
                      picture.run_typical_fps, picture.run_active_fraction * 100.0,
                      static_cast<unsigned long long>(picture.run_distinct));
    return text;
}

void note_present_publication(const uint8_t* pixels, size_t bytes, uint32_t width, uint32_t height) {
    // Hashed OUTSIDE the lock. The signature depends only on the caller's bytes, so holding
    // g_rate_mx across it would serialise publications on work that needs no serialising.
    const uint64_t signature = frame_content_signature(pixels, bytes, width, height);
    const double at = now_seconds();
    std::lock_guard<std::mutex> lk(g_rate_mx);
    g_rate.observe(signature, at);
}

uint64_t dense_content_signature(const uint8_t* pixels, size_t bytes) {
    uint64_t h = 0xcbf29ce484222325ull;
    mix(h, static_cast<uint64_t>(bytes));
    if (!pixels || bytes == 0) return h;
    // Every 8-byte word, then the unaligned tail. The buffer is small by construction (the caller
    // owns the sample grid), so there is nothing to gain by skipping any of it.
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= bytes; i += sizeof(uint64_t)) {
        uint64_t word = 0;
        std::memcpy(&word, pixels + i, sizeof word);
        mix(h, word);
    }
    if (i < bytes) {
        uint64_t tail = 0;
        std::memcpy(&tail, pixels + i, bytes - i);
        mix(h, tail);
    }
    return h;
}

void note_present_publication_signature(uint64_t signature) {
    const double at = now_seconds();
    std::lock_guard<std::mutex> lk(g_rate_mx);
    g_rate.observe(signature, at);
}

void reset_present_rate() {
    std::lock_guard<std::mutex> lk(g_rate_mx);
    g_rate.reset();
}

PresentRateSnapshot present_rate_snapshot() {
    PresentRateSnapshot out;
    out.now_seconds = now_seconds();
    std::lock_guard<std::mutex> lk(g_rate_mx);
    out.published = g_rate.published();
    out.distinct = g_rate.distinct();
    out.first_publication_seconds = g_rate.first_publication_seconds();
    out.last_publication_seconds = g_rate.last_publication_seconds();
    out.typical_interval_seconds = g_rate.typical_interval_seconds();
    out.active_seconds = g_rate.active_seconds();
    out.interval_samples = g_rate.interval_samples();
    return out;
}

} // namespace prosper::gpu
