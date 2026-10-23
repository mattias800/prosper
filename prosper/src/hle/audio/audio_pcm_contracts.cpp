#include "hle/audio/audio.hpp"
#include <cstddef>
#include <cstdint>

namespace prosper {
// clang-format off: preserve the byte-identical moved definitions.

void audio_decode_format(uint32_t param, int& channels, AudioFmt& fmt) {
    switch (param & 0xff) {                                   // SceAudioOutParamFormat (low byte)
        case 0: channels = 1; fmt = AudioFmt::S16; break;     // S16_MONO
        case 1: channels = 2; fmt = AudioFmt::S16; break;     // S16_STEREO
        case 2: channels = 8; fmt = AudioFmt::S16; break;     // S16_8CH
        case 3: channels = 1; fmt = AudioFmt::F32; break;     // FLOAT_MONO
        case 4: channels = 2; fmt = AudioFmt::F32; break;     // FLOAT_STEREO
        case 5: channels = 8; fmt = AudioFmt::F32; break;     // FLOAT_8CH
        case 6: channels = 8; fmt = AudioFmt::S16; break;     // S16_8CH_STD
        case 7: channels = 8; fmt = AudioFmt::F32; break;     // FLOAT_8CH_STD
        default: channels = 2; fmt = AudioFmt::S16; break;    // unknown -> S16 stereo
    }
}

void audio_apply_channel_volumes(int dst[8], uint32_t mask, const int* vols) {
    if (!dst || !vols) return;
    for (int c = 0; c < 8; c++)
        if (mask & (1u << c)) dst[c] = vols[c];
}

int audio_peak_channel_volume(uint32_t mask, const int* vols) {
    if (!vols) return 0;
    int peak = 0;
    for (int c = 0; c < 8; c++)
        if ((mask & (1u << c)) && vols[c] > peak) peak = vols[c];
    return peak;
}

bool audio2_reserve_queue_slot(uint32_t& queued, uint32_t queue_depth) {
    if (queued >= queue_depth) return false;
    ++queued;
    return true;
}

uint64_t audio_count_nonzero_samples(const void* pcm, std::size_t bytes, bool s16) {
    if (!pcm) return 0;
    uint64_t n = 0;
    if (s16) {
        const auto* s = static_cast<const int16_t*>(pcm);
        for (size_t i = 0, m = bytes / sizeof(int16_t); i < m; ++i)
            if (s[i] != 0) ++n;
    } else {
        const auto* s = static_cast<const float*>(pcm);
        // `!= 0.0f`, never a byte test: -0.0f compares equal to zero and has a non-zero bit
        // pattern, and NaN compares unequal to zero, which is the same treatment
        // AudioSignalStats gives both.
        for (size_t i = 0, m = bytes / sizeof(float); i < m; ++i)
            if (s[i] != 0.0f) ++n;
    }
    return n;
}

AudioGrainVerdict audio_classify_stamped_grain(bool same_address, bool matches_stamp,
                                               uint64_t nonzero_samples) {
    // Two properties are load-bearing, and a third that looks like it is, is not. Recorded because
    // the difference was established by mutation rather than by reading, and reading had it wrong:
    //   1. The address check must precede both content tests. A stamp read back from a DIFFERENT
    //      buffer says nothing about either one, and the double-buffered case is otherwise reported
    //      as a clean verdict about a buffer nobody looked at.
    //   2. The identity test must EXIST. The stamp is deliberately non-zero in every sample, so a
    //      surviving stamp carries a large non-zero count; drop this line and "prosper is reading
    //      memory the guest never writes" is reported as "the port is healthy" — the exact inversion
    //      the probe exists to prevent.
    //   3. Whether identity is tested BEFORE or AFTER emptiness is immaterial, and the first version
    //      of this comment claimed otherwise. Swapping them is behaviour-preserving for as long as
    //      property 2's invariant holds, because the two conditions are then mutually exclusive: a
    //      surviving stamp is never empty. The swap arm PASSED, which is what corrected the comment.
    if (!same_address) return AudioGrainVerdict::PointerMoved;
    if (matches_stamp) return AudioGrainVerdict::Intact;
    if (nonzero_samples == 0) return AudioGrainVerdict::Cleared;
    return AudioGrainVerdict::Overwritten;
}

// Stereo fold-down for a MAIN speaker bed. See audio.hpp for the contract.
//
// The bed order for 1..8 channels is FL, FR, FC, LFE, then surround pairs, with the small layouts
// kept useful (3ch adds FC, 4ch is quad, 5ch is FL/FR/FC/SL/SR, 6ch is 5.1).
//
// A stereo fold only needs each source channel's SIDE (left, right, or both) — the front/back and
// height identity within a side does not change the result — and that is what prosper can measure
// with PROSPER_AUDIO_LAYOUT (see the probe below and docs/subsystems/AUDIO.md). Measured on Dragon Quest VII
// Reimagined's 12-channel MAIN port over 19,962,368 frames (#1700):
//   ch0/ch1  carry all of the content (rms 3.1e-2 / 4.3e-2, peak 0.407) and correlate +0.46 — a
//            real decorrelated stereo pair, not a duplicated mono feed.
//   ch2      correlates near-equally with both side groups below (0.49/0.46/0.41/0.40): centre-like.
//   ch4,ch6  correlate +0.96 with each other and ch5,ch7 correlate +0.95, while ch4-ch5 is -0.04.
//            The surround tier therefore pairs as even=left, odd=right, which is what the 6..8
//            mapping below already assumed. CONFIDENCE: MED — the grouping is unambiguous but the
//            channels sit at a ~1e-9 residue level, so it corroborates rather than proves.
//   ch3 and ch8..ch11 are EXACTLY zero for the whole run — the title's panner writes neither the
//            LFE nor the height position under the order assumed here. Stated as indices on
//            purpose: which POSITIONS those are is the mapping this evidence supports, not a
//            premise it may borrow.
// Channels 8..15 consequently have no measured side and are deliberately left UNPLACED (see the
// return value): a fold-down that sounds plausible but images content to the wrong side is worse
// than one that reports the gap, and no title in evidence puts signal there. The leading hypothesis
// is that the same even=left/odd=right pairing continues, but that is inference from the tiers
// below it, not a measurement, so it is tracked as an issue instead of shipped.
// What the measurement CANNOT settle, stated so nobody reads more into it than it carries. The
// probe GROUPS channels by side; it does not ORIENT the groups, and those are different claims:
//   - Correlation is symmetric, so it proves ch0/ch1 are a genuine front pair but cannot tell a
//     left/right SWAP from the correct assignment. The same is true of the {ch4,ch6} / {ch5,ch7}
//     grouping: which group is the left one is not measured anywhere here.
//     **CONFIDENCE: MED for the left/right ORIENTATION of every pair in this table, and the basis is
//     convention, not evidence** — index 0 = FrontLeft is universal across published multichannel
//     bed layouts, and prosper's own v1 sceAudioOut path assumes it. TWO conventions, not three: a
//     third was checked and REJECTED. CRI Atom's `EAtomSpeakerID` names are present in this title's
//     eboot and order FrontLeft first, which looked like a property of the code producing these
//     samples rather than an inherited assumption — but `tools/re/xref.py` finds ZERO code
//     references to that string and one data-pointer relocation into a UE reflection table. The
//     names are Blueprint metadata that survived into the image; nothing calls them, so their
//     presence says nothing about the bed's order. Conventions can share one ancestor, and a symbol
//     in a binary is not a code path. prosper's own test suite does pin index -> side for widths
//     6..8, so a self-inconsistent fold fails in CI; what is unconfirmed is the match to hardware.
//   - A listening test cannot settle it either, and specifically cannot settle this title's layout
//     at all: ten of its twelve channels are measured empty, so every mapping that routes ch0 and
//     ch1 to the two sides produces a host bed differing only by the ~1e-9 residue on ch2 and
//     ch4..ch7, some 150 dB below the content — not "bit-identical", because that wording would
//     re-merge "exactly zero" with "1e-9 residue", the very distinction this table draws.
//     Broadly stereo music also folds down
//     plausibly under a swap. A human confirming "it sounds right" is real rung-4 evidence that
//     audio reaches the device through the guest's own path, and is NOT evidence for this mapping.
//   - THE DISCRIMINATING EXPERIMENT, for whoever gets a title that supports it: content with
//     distinct per-channel placement — a hard-panned effect whose true side is known from the game,
//     or centre-channel dialogue — measured with PROSPER_AUDIO_LAYOUT. It settles orientation in
//     one run, where any amount of music cannot. Tracked with the height tier on #1720, since one
//     capture answers both.
// CONFIDENCE: HIGH that ch0/ch1 are the front pair and that 1..2-channel beds are right (a mono or
// stereo bed has no orientation to get wrong); MED for which channel of a pair is the left one, and
// for the 3..8 placements; 9..16 places only its first eight and reports the rest.
unsigned audio_stereo_downmix(unsigned channels, AudioStereoGain* out, unsigned out_capacity) {
    if (!out || !channels || channels > kAudioMaxBedChannels || out_capacity < channels)
        return channels;
    for (unsigned c = 0; c < channels; ++c) out[c] = AudioStereoGain{};

    constexpr float kUnity    = 1.0f;
    constexpr float kCenter   = 0.70710678f;   // -3 dB, centre split equally across the pair
    constexpr float kLfe      = 0.5f;          // -6 dB
    constexpr float kSurround = 0.70710678f;   // -3 dB

    if (channels == 1) { out[0] = {kUnity, kUnity}; return 0; }   // mono feeds both sides
    out[0] = {kUnity, 0.0f};                                      // FL
    out[1] = {0.0f, kUnity};                                      // FR
    if (channels == 2) return 0;
    if (channels == 3) { out[2] = {kCenter, kCenter}; return 0; }
    if (channels == 4) {                                          // quad: FL FR SL SR
        out[2] = {kSurround, 0.0f};
        out[3] = {0.0f, kSurround};
        return 0;
    }
    if (channels == 5) {                                          // FL FR FC SL SR
        out[2] = {kCenter, kCenter};
        out[3] = {kSurround, 0.0f};
        out[4] = {0.0f, kSurround};
        return 0;
    }
    // 6..16: FL FR FC LFE then surround pairs.
    out[2] = {kCenter, kCenter};
    out[3] = {kLfe, kLfe};
    out[4] = {kSurround, 0.0f};
    out[5] = {0.0f, kSurround};
    if (channels == 6) return 0;
    if (channels == 7) {                                          // 6.1: one rear-centre channel
        out[6] = {kSurround * 0.5f, kSurround * 0.5f};
        return 0;
    }
    out[6] = {kSurround, 0.0f};                                   // 7.1 second surround pair
    out[7] = {0.0f, kSurround};
    unsigned unplaced = 0;
    for (unsigned c = 8; c < channels; ++c) ++unplaced;            // height tier: not yet placed
    return unplaced;
}
// clang-format on
} // namespace prosper
