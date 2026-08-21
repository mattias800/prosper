// present_frame_rate.cpp — see present_frame_rate.hpp.
#include "gpu/present/present_frame_rate.hpp"

#include <algorithm>
#include <chrono>
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
    if (changed) distinct_++;
    last_signature_ = signature;
    last_ = at_seconds;
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
    return make_rate(s.published, s.distinct, s.now_seconds - s.first_publication_seconds);
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
    char text[256];
    if (!rate.measured) {
        std::snprintf(text, sizeof text,
                      "not measured (%llu published, %llu distinct, %.3f s window)",
                      static_cast<unsigned long long>(rate.published),
                      static_cast<unsigned long long>(rate.distinct), rate.window_seconds);
        return text;
    }
    std::snprintf(text, sizeof text,
                  "distinct %.1f fps / presented %.1f fps over %.1f s "
                  "(%llu of %llu published frames carried new content, %.1f%%)",
                  rate.distinct_fps, rate.presented_fps, rate.window_seconds,
                  static_cast<unsigned long long>(rate.distinct),
                  static_cast<unsigned long long>(rate.published),
                  rate.distinct_fraction * 100.0);
    return text;
}

std::string format_frame_rate_short(const FrameRate& rate, uint32_t width, uint32_t height) {
    char text[128];
    if (!rate.measured) {
        std::snprintf(text, sizeof text, "-- fps  %ux%u", width, height);
        return text;
    }
    std::snprintf(text, sizeof text, "%.1f fps  (%.1f presented)  %ux%u",
                  rate.distinct_fps, rate.presented_fps, width, height);
    return text;
}

bool frame_rate_is_mostly_retained(const FrameRate& rate, uint64_t min_published,
                                   double max_distinct_fraction) {
    return rate.published >= min_published && rate.distinct_fraction < max_distinct_fraction;
}

void note_present_publication(const uint8_t* pixels, size_t bytes, uint32_t width, uint32_t height) {
    // Hashed OUTSIDE the lock. The signature depends only on the caller's bytes, so holding
    // g_rate_mx across it would serialise publications on work that needs no serialising.
    const uint64_t signature = frame_content_signature(pixels, bytes, width, height);
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
    return out;
}

} // namespace prosper::gpu
