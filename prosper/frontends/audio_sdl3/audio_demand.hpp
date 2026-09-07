#pragma once
#include "audio_sdl3.hpp"
#include <SDL3/SDL.h>
#include <atomic>

namespace prosper {

// The callback owns no samples and supplies none. SDL calls it while holding the
// stream lock, before consuming queued input. Its additional byte estimate can
// include resampler padding, so this is demand accounting, not inserted silence.
class AudioDemandMeter {
public:
    static_assert(std::atomic<uint64_t>::is_always_lock_free);
    void reset(AudioDemandPhase phase) {
        // Only before attaching the callback, after any previous stream is destroyed.
        for (auto& c : counts_) {
            c.calls = 0; c.requested = 0; c.short_calls = 0; c.additional = 0;
            c.first = 0; c.last = 0;
        }
        set_phase(phase);
    }
    void set_phase(AudioDemandPhase phase) { phase_.store(phase, std::memory_order_relaxed); }

    static void SDLCALL consume(void* userdata, SDL_AudioStream*, int additional, int total) {
        auto& self = *static_cast<AudioDemandMeter*>(userdata);
        auto& c = self.counts_[static_cast<unsigned>(self.phase_.load(std::memory_order_relaxed))];
        const auto now = SDL_GetTicksNS();
        if (!c.calls.load(std::memory_order_relaxed)) c.first.store(now, std::memory_order_relaxed);
        c.calls.fetch_add(1, std::memory_order_relaxed);
        c.requested.fetch_add(total > 0 ? uint64_t(total) : 0, std::memory_order_relaxed);
        if (additional > 0) {
            c.short_calls.fetch_add(1, std::memory_order_relaxed);
            c.additional.fetch_add(uint64_t(additional), std::memory_order_relaxed);
        }
        c.last.store(now, std::memory_order_relaxed);
    }

    bool snapshot(SDL_AudioStream* stream, Sdl3AudioDemandSnapshot& out) const {
        // The caller protects stream lifetime. This lock makes the set of atomic
        // counters consistent; the callback never acquires the outer sink lock.
        if (!SDL_LockAudioStream(stream)) return false;
        for (unsigned i = 0; i < counts_.size(); ++i) {
            const auto& c = counts_[i];
            out.phases[i] = {c.calls.load(std::memory_order_relaxed),
                c.requested.load(std::memory_order_relaxed), c.short_calls.load(std::memory_order_relaxed),
                c.additional.load(std::memory_order_relaxed), c.first.load(std::memory_order_relaxed),
                c.last.load(std::memory_order_relaxed)};
        }
        SDL_UnlockAudioStream(stream);
        return true;
    }
private:
    struct Counts {
        std::atomic<uint64_t> calls{0}, requested{0}, short_calls{0}, additional{0}, first{0}, last{0};
    };
    std::array<Counts, 3> counts_{};
    std::atomic<AudioDemandPhase> phase_{AudioDemandPhase::Startup};
};
} // namespace prosper
