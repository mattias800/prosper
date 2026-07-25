// test_ajm2_gapless (#1097) — the AJM batch-2.0 decode path implements REAL gapless semantics.
//
// FMOD on PS5 programs every ATRAC9 stream with sceAjmBatchJobSetGaplessDecode{total, skip}: the
// decoder must drop `skip` priming sample-frames at the start of the program, deliver EXACTLY
// `total` trimmed frames, and then report end-of-stream by pinning uiTotalDecodedSamples at
// `total` with no further PCM. FMOD's channel-end / codec-recycling logic keys off that exact
// landing: with prosper's old behavior (whole superframes, no trim, cumulative raw counter) the
// reported total OVERSHOT the programmed total, FMOD never saw a one-shot SFX finish, never
// recycled its AJM codec slot, and after ~32 sounds every later SFX was silent (#1097).
//
// Drives the real HLE handlers (host pointers stand in for guest addresses, as in test_ajm) over
// the shared 16-superframe test vector, with reference PCM built by the same Atrac9Decoder glue
// that test_atrac9 proves bit-exact against LibAtrac9.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/atrac9_decode.hpp"
#include "at9_testvec.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

using Hle10Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                             uint64_t, uint64_t, uint64_t, uint64_t);

struct Sideband {
    int32_t iResult; int32_t iCodecResult;
    uint32_t iSizeConsumed; uint32_t iSizeProduced;
    uint64_t uiTotalDecodedSamples;
    uint32_t numFrames; uint32_t reserved;
};
static_assert(sizeof(Sideband) == 32, "AJM decode sideband includes the MFrame result");

static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }

int main() {
    printf("== test_ajm2_gapless ==\n");
    register_builtin_hle();

    auto job_init = reinterpret_cast<Hle10Fn>(Hle::lookup(nid_hash("sceAjmBatchJobInitialize")));
    auto job_gapl = reinterpret_cast<Hle10Fn>(Hle::lookup(nid_hash("sceAjmBatchJobSetGaplessDecode")));
    auto job_dec  = reinterpret_cast<Hle10Fn>(Hle::lookup(nid_hash("sceAjmBatchJobDecode")));
    auto batch_go = reinterpret_cast<Hle10Fn>(Hle::lookup(nid_hash("sceAjmBatchStart")));
    CHECK(job_init && job_gapl && job_dec && batch_go, "batch-2.0 handlers registered");
    if (!(job_init && job_gapl && job_dec && batch_go)) return 1;

    // Reference PCM: the full 16-superframe decode, bit-exact per test_atrac9.
    Atrac9Decoder ref;
    CHECK(ref.init(kAt9Config), "reference decoder init");
    if (!ref.valid()) return 1;
    const int ch = ref.channels();
    const uint32_t frame_bytes = (uint32_t)ch * sizeof(int16_t);
    const int sfb = ref.superframe_bytes();
    const int sfs = ref.superframe_samples();
    const uint32_t nsf = kAt9DataLen / (uint32_t)sfb;
    std::vector<int16_t> reference;
    { std::vector<int16_t> sf((size_t)sfs * ch);
      for (uint32_t i = 0; i < nsf; i++) {
          CHECK(ref.decode_superframe(kAt9Data + (size_t)i * sfb, sf.data()) == sfs,
                "reference superframe decodes");
          reference.insert(reference.end(), sf.begin(), sf.end());
      } }
    const uint64_t raw_frames = (uint64_t)nsf * sfs;

    // A one-shot's gapless program: skip 256 priming frames, deliver a total that ends INSIDE the
    // final superframe (mirrors real AT9 files: raw = skip + total + end padding).
    const uint32_t kSkip = 256;
    const uint64_t kTotal = raw_frames - kSkip - 100;

    uint64_t batch = 0x1000;                      // opaque key, only identity matters
    const uint32_t inst = 7;
    Sideband sb{};
    uint32_t gapless[2] = {(uint32_t)kTotal, kSkip};
    std::vector<uint8_t> out(reference.size() * sizeof(int16_t) + 4096, 0xAA);

    job_init(batch, inst, addr(kAt9Config), 8, addr(&sb), 0, 0, 0, 0, 0);
    job_gapl(batch, inst, addr(gapless), 1, addr(&sb), 0, 0, 0, 0, 0);
    std::memset(&sb, 0, sizeof sb);
    job_dec(batch, inst, addr(kAt9Data), kAt9DataLen, addr(out.data()), out.size(),
            addr(&sb), 0, 0, 0);
    batch_go(0, batch, 0, 0, 0, 0, 0, 0, 0, 0);

    CHECK(sb.iResult == 0, "one-shot decode succeeds");
    CHECK(sb.uiTotalDecodedSamples == kTotal,
          "uiTotalDecodedSamples lands EXACTLY on the gapless total (EOS)");
    CHECK(sb.iSizeProduced == kTotal * frame_bytes,
          "produced PCM is exactly total trimmed frames");
    // Content: delivery starts AFTER the skip -> output[0..] == reference[skip..skip+total].
    CHECK(std::memcmp(out.data(), reference.data() + (size_t)kSkip * ch,
                      (size_t)kTotal * frame_bytes) == 0,
          "delivered PCM equals reference with the priming skip dropped");

    // After EOS: more input must produce NOTHING and the total must stay pinned at `total`.
    Sideband sb2{};
    job_dec(batch, inst, addr(kAt9Data), (uint32_t)sfb, addr(out.data()), out.size(),
            addr(&sb2), 0, 0, 0);
    batch_go(0, batch, 0, 0, 0, 0, 0, 0, 0, 0);
    CHECK(sb2.iResult == 0, "post-EOS decode is not an error");
    CHECK(sb2.iSizeProduced == 0, "post-EOS decode produces no PCM");
    CHECK(sb2.uiTotalDecodedSamples == kTotal, "post-EOS total stays pinned at the gapless total");

    // Re-arming gapless starts a NEW program (FMOD's loop path): counter restarts, skip applies
    // again, and delivery is exactly the new total.
    const uint64_t kTotal2 = (uint64_t)sfs;       // one superframe's worth, ends inside sf 2
    uint32_t gapless2[2] = {(uint32_t)kTotal2, kSkip};
    Sideband sb3{};
    job_gapl(batch, inst, addr(gapless2), 1, addr(&sb3), 0, 0, 0, 0, 0);
    job_dec(batch, inst, addr(kAt9Data), (uint32_t)(2 * sfb), addr(out.data()), out.size(),
            addr(&sb3), 0, 0, 0);
    batch_go(0, batch, 0, 0, 0, 0, 0, 0, 0, 0);
    CHECK(sb3.iResult == 0, "re-armed program decodes");
    CHECK(sb3.uiTotalDecodedSamples == kTotal2,
          "re-armed program's total restarts and lands exactly on the new total");
    CHECK(sb3.iSizeProduced == kTotal2 * frame_bytes, "re-armed program delivers the new total");

    // --- Streaming a gapless program through a TINY output buffer across many jobs (review #1097).
    // Exercises the subtlest paths the big-buffer cases skip: room-starvation spill UNDER an active
    // gapless program, the carry erase-tail branch, and multi-batch program continuation. Feeds input
    // the way a real guest does -- advancing by each job's reported iSizeConsumed -- and accumulates
    // the delivered PCM, which must equal reference[skip .. skip+total] and land exactly on total.
    const uint32_t inst3 = 9;
    const uint32_t kSkip3 = (uint32_t)sfs + 137;             // skip spans MORE than one superframe
    const uint64_t kTotal3 = raw_frames - kSkip3 - 500;
    uint32_t gapless3[2] = {(uint32_t)kTotal3, kSkip3};
    job_init(batch, inst3, addr(kAt9Config), 8, addr(&sb), 0, 0, 0, 0, 0);
    job_gapl(batch, inst3, addr(gapless3), 1, addr(&sb), 0, 0, 0, 0, 0);

    // CONTROL: single job, whole input, output sized to EXACTLY total. Isolates the code's
    // skip>superframe + total-cap from the multi-job streaming model.
    {
        Sideband sc{};
        std::vector<uint8_t> exact((size_t)kTotal3 * frame_bytes, 0);
        job_dec(batch, inst3, addr(kAt9Data), kAt9DataLen, addr(exact.data()), exact.size(),
                addr(&sc), 0, 0, 0);
        batch_go(0, batch, 0, 0, 0, 0, 0, 0, 0, 0);
        CHECK(sc.uiTotalDecodedSamples == kTotal3 &&
              std::memcmp(exact.data(), reference.data()+(size_t)kSkip3*ch,
                          (size_t)kTotal3*frame_bytes)==0,
              "control: single-job skip>superframe + total-cap lands exactly");
    }
    // Fresh instance for the streaming test: the control above decoded inst3 to stream-end, and
    // re-arming gapless resets the trim counters but NOT the Atrac9Decoder's MDCT position, so the
    // streaming pass needs its own instance (new id -> fresh decoder on first JobInitialize).
    const uint32_t inst4 = 10;
    job_init(batch, inst4, addr(kAt9Config), 8, addr(&sb), 0, 0, 0, 0, 0);
    job_gapl(batch, inst4, addr(gapless3), 1, addr(&sb), 0, 0, 0, 0, 0);

    std::vector<int16_t> collected;
    const uint32_t tiny = (uint32_t)(300 * ch) * sizeof(int16_t);   // ~300 frames per job
    std::vector<uint8_t> tinyout(tiny, 0);
    uint32_t in_off = 0; int guard = 0; bool tiny_ok = true;
    while (guard++ < 100000) {
        Sideband s{};
        const uint32_t in_left = (in_off < kAt9DataLen) ? (kAt9DataLen - in_off) : 0;
        // Real feed while input remains; once exhausted, a valid sub-superframe input that decodes
        // nothing new but lets the handler accept the job and drain the carry (mirrors the guest's
        // next feed, whose carry-drain runs first).
        const uint64_t din = in_left ? addr(kAt9Data + in_off) : addr(kAt9Data);
        const uint32_t dsz = in_left ? in_left : 1u;
        job_dec(batch, inst4, din, dsz, addr(tinyout.data()), tinyout.size(), addr(&s), 0, 0, 0);
        batch_go(0, batch, 0, 0, 0, 0, 0, 0, 0, 0);
        if (s.iResult != 0) { tiny_ok = false; break; }
        const uint32_t got_frames = s.iSizeProduced / frame_bytes;
        collected.insert(collected.end(),
                         reinterpret_cast<const int16_t*>(tinyout.data()),
                         reinterpret_cast<const int16_t*>(tinyout.data()) + (size_t)got_frames * ch);
        in_off += s.iSizeConsumed;
        if (s.uiTotalDecodedSamples >= kTotal3) break;      // program complete
        if (in_left == 0 && s.iSizeProduced == 0) break;    // input gone and carry drained
    }
    CHECK(tiny_ok, "tiny-buffer streaming has no decode error");
    CHECK(collected.size() == (size_t)kTotal3 * ch,
          "tiny-buffer streaming delivers exactly total frames across many jobs");
    CHECK(collected.size() == (size_t)kTotal3 * ch &&
          std::memcmp(collected.data(), reference.data() + (size_t)kSkip3 * ch,
                      (size_t)kTotal3 * frame_bytes) == 0,
          "tiny-buffer streamed PCM equals reference[skip..skip+total] (multi-superframe skip)");

    // A fresh instance with NO gapless program keeps the historical cumulative behavior
    // (streaming titles that never program gapless must not change).
    const uint32_t inst2 = 8;
    Sideband sb4{};
    job_init(batch, inst2, addr(kAt9Config), 8, addr(&sb4), 0, 0, 0, 0, 0);
    job_dec(batch, inst2, addr(kAt9Data), kAt9DataLen, addr(out.data()), out.size(),
            addr(&sb4), 0, 0, 0);
    batch_go(0, batch, 0, 0, 0, 0, 0, 0, 0, 0);
    CHECK(sb4.iResult == 0, "no-gapless instance decodes");
    CHECK(sb4.uiTotalDecodedSamples == raw_frames,
          "no-gapless instance keeps the cumulative raw-frame counter");

    // Result publication is the commit point for a persistent ATRAC9 decode. If the result pointer
    // is inaccessible after PCM and codec state advance, the instance must become terminal before
    // either a later job in the same batch or a later batch can decode against hidden state.
    const uint32_t inst5 = 11;
    Sideband terminal_same_batch{};
    std::vector<int16_t> terminal_first((size_t)sfs * ch, (int16_t)0x5555);
    std::vector<int16_t> terminal_second((size_t)sfs * ch, (int16_t)0x6666);
    job_init(batch, inst5, addr(kAt9Config), 8, 0, 0, 0, 0, 0, 0);
    constexpr uint64_t inaccessible_result = 0x0000deadbeef0000ull;
    CHECK(job_dec(batch, inst5, addr(kAt9Data), (uint32_t)sfb,
                  addr(terminal_first.data()), (uint32_t)(terminal_first.size() * sizeof(int16_t)),
                  inaccessible_result, 0, 0, 0) == 0 &&
          job_dec(batch, inst5, addr(kAt9Data + sfb), (uint32_t)sfb,
                  addr(terminal_second.data()), (uint32_t)(terminal_second.size() * sizeof(int16_t)),
                  addr(&terminal_same_batch), 0, 0, 0) == 0,
          "ATRAC9 publication fixture queues two jobs on one persistent instance");
    batch_go(0, batch, 0, 0, 0, 0, 0, 0, 0, 0);
    CHECK(std::memcmp(terminal_first.data(), reference.data(),
                      (size_t)sfs * frame_bytes) == 0,
          "failed result publication occurs after the first ATRAC9 decode reaches PCM output");
    CHECK(terminal_same_batch.iResult != 0 &&
          terminal_same_batch.iSizeConsumed == 0 &&
          terminal_same_batch.iSizeProduced == 0 &&
          terminal_same_batch.uiTotalDecodedSamples == (uint64_t)sfs &&
          terminal_same_batch.numFrames == 0 &&
          terminal_second.front() == (int16_t)0x6666,
          "failed result publication terminalizes before the same-batch job");

    Sideband terminal_later_batch{};
    std::vector<int16_t> terminal_third((size_t)sfs * ch, (int16_t)0x7777);
    CHECK(job_dec(batch, inst5, addr(kAt9Data + 2 * sfb), (uint32_t)sfb,
                  addr(terminal_third.data()), (uint32_t)(terminal_third.size() * sizeof(int16_t)),
                  addr(&terminal_later_batch), 0, 0, 0) == 0,
          "terminal ATRAC9 instance accepts a later job for an error sideband");
    batch_go(0, batch, 0, 0, 0, 0, 0, 0, 0, 0);
    CHECK(terminal_later_batch.iResult != 0 &&
          terminal_later_batch.iSizeConsumed == 0 &&
          terminal_later_batch.iSizeProduced == 0 &&
          terminal_later_batch.uiTotalDecodedSamples == (uint64_t)sfs &&
          terminal_later_batch.numFrames == 0 &&
          terminal_third.front() == (int16_t)0x7777,
          "terminal ATRAC9 state persists across batches with a stable cumulative total");

    printf(fails ? "test_ajm2_gapless: %d FAILURE(S)\n" : "test_ajm2_gapless: all ok\n", fails);
    return fails ? 1 : 0;
}
