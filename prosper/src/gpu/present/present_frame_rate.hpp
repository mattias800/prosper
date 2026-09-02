// present_frame_rate.hpp — how fast is this title actually running?
//
// WHY THIS IS NOT A PRESENT COUNTER, AND WHY THAT DISTINCTION IS THE WHOLE POINT
// ------------------------------------------------------------------------------
// The obvious framerate is "publications per second": count present_write_frame calls, divide by
// wall time. That number is WRONG in the one case anybody needs a framerate for, and wrong in the
// direction that hides the defect — it reads FULL SPEED for a completely frozen title.
//
// The reason is `RetainedFrameAction::ServeRetained` (frontends/shared/live/live_renderer.cpp):
// when a submit produces no publishable present source, the renderer re-serves the frame it
// retained, and that re-serve goes through the ordinary publish path. The guest keeps flipping,
// the renderer keeps publishing, `present_frame_seq()` keeps climbing — and the screen never
// changes. That is instrument trap 90 stated as a counter, and it is exactly the R-Type Delta
// (PPSA26414) regression #2783: for nine days the guest reached stage 1 while every presented frame
// was the same retained one, with `no pass produced a 1920x1080 present source … offered 0 bytes`
// on every flip. A present-rate fps would have read ~60 through the entire failure. The A/B that
// finally diagnosed it (#2799) counted DISTINCT late `pixel_crc32` values — 46 of 46 with the fix,
// 2 without — precisely because a rate alone could not see it.
//
// So this module counts two things and every consumer reports BOTH:
//
//   presented fps — publications accepted by the present layer, per second of wall clock.
//   distinct  fps — publications whose CONTENT differed from the immediately preceding
//                   publication, per second of wall clock.
//
// A healthy title has the two roughly equal. A title where they diverge is a title with a problem,
// and the divergence is itself the diagnostic: `distinct 0.0 / presented 59.8` is a frozen picture
// being re-served at vblank, not a game running at 60 fps.
//
// WHY A RUN AVERAGE IS NOT THE HEADLINE, AND WHAT IS
// ---------------------------------------------------
// Both rates above are averages over wall clock, and that makes them a bad summary of any route
// that pauses. Measured on *The Messenger* over 380 s at native 1080p: `distinct 3.0 fps`, while the
// per-sample counters show the rate was either ~15-23 fps or EXACTLY zero, with nothing in between
// — 120 consecutive seconds of it a title screen across which 25,015 publications carried exactly
// ONE change.
// 3.0 is a true average of a bimodal signal and describes neither mode. It also contradicts the
// July performance pass, which measured this title's first level at 12-24 fps; a reader would
// reasonably conclude a regression that never happened. Both of those figures are readback-path
// measurements, so they remain comparable WITH EACH OTHER -- what neither describes is the shipped
// renderer. The July figure predates #1270, which removed the CPU readback from the app's present
// path on 2026-07-24; uncapped windowed, the same level now reports a median 156 presented fps.
// See CLAUDE.md's harness bullet and #3083 before comparing any two rates from this project.
//
// THE REAL FIX IS THE WINDOW, NOT THE STATISTIC. No single scalar can describe a bimodal sample;
// every candidate misleads somewhere, so arguing about which one to use is choosing which way to be
// wrong. A framerate is only meaningful over a window in which the title was doing ONE thing — and
// the record it ends up in commits to that anyway, because it names a scene. Measure gameplay over
// gameplay. If a route never leaves its menus, the honest output is no framerate at all rather than
// a figure describing a title screen.
//
// What follows exists to make that discipline CHECKABLE rather than to substitute for it: the pair
// below says whether the window you chose actually was homogeneous, and a run average is only worth
// quoting when it says yes.
//
//   typical fps — the reciprocal of the MEDIAN interval between consecutive distinct frames.
//
// A median over intervals weights by FRAME, not by time: a 120-second pause is one long interval,
// not 120 seconds of pull, so idling cannot drag the figure down. It needs no threshold and no
// window, and it is the ordinary way frame times are reported.
//
// It is never quoted alone, because on its own it could describe two frames half a second apart in
// an otherwise dead run. It travels with:
//
//   active fraction — the share of the window spent producing frames at roughly the typical rate.
//
// **Read `active_fraction` as a verdict on your WINDOW, not as a property of the title.** Near 100%
// means the window was homogeneous, so the average and the typical rate agree and either is
// quotable. Well below it means the window mixed two regimes and no single number from it means
// anything — re-measure over a narrower window rather than reaching for a different statistic.
//
// **IT IS NOT A CONTENT ORACLE, AND THE FAILURE IS SEDUCTIVE.** "95% active" says frames kept
// CHANGING; it says nothing about whether anything was on them. Measured on *Blue Prince* over
// 899.6 s: `4.7 fps while producing frames, 95% active` — a reading that looks like a clean
// homogeneous measurement, and one of whose samples is a uniform near-black frame. The run was in
// fact fine (57 of 60 samples fully non-black, peaking at 128,506 distinct colours), but nothing in
// THIS module could have told you that, and a black-screened title rendering a changing black
// screen would report exactly the same 95%.
//
// It is equally blind to whether frames are NOVEL or merely alternating, and that shape is likelier
// to fool a reader because it produces a textbook-clean measurement. "Distinct" here means "differs
// from the IMMEDIATELY PRECEDING publication", so a title flipping A,B,A,B at 60 Hz reports ~60 fps
// typical at ~100% active with perfect homogeneity — which is also what a two-frame flicker, a stale
// double-buffered pair, or an alternating composite looks like. **The metric answers "did the bytes
// change", never "was progress made".**
//
// Content is `tools/screenshot`'s job and it already measures it: `distinct_rgb_colors` and
// `nonblack_rgb_pixels` per sample, and the pixel-distinct assertions. Pair a rate with those before
// believing a scene was rendered — a framerate is a statement about time, never about pixels.
//
// It is a percentage rather than a rate so it cannot be misread as a rival framerate. Together the
// two separate the three cases that matter, and no pair of averages can:
//
//   19.8 fps, 97% active   a homogeneous gameplay window — THIS is what a record is made from
//    1.0 fps, 98% active   homogeneous and genuinely slow — the "we have work to do" bucket
//   18.5 fps, 62% active   a mixed window: real, but do not file it; narrow the window and re-run
//     -- fps,  0% active   the title produced nothing: the R-Type Delta shape
//
// The `--` case is load-bearing. A frozen title must never read as a high framerate OR as an absent
// one, so fewer than two distinct frames yields no rate at all rather than a number, and the 0%
// beside it says why.
//
// WHAT "DIFFERENT CONTENT" MEANS HERE, EXACTLY
// --------------------------------------------
// Two publications are the same frame when `frame_content_signature` agrees on them. The signature
// is deliberately SUB-LINEAR (see the constants below) because it runs on every publication of a
// frame that may be 33 MB, and a full hash of that at 60 Hz would cost more than the render.
//
// The sampling error is therefore real, and it is one-sided in the safe direction:
//   * it can NEVER report a re-served frame as new — identical bytes give an identical signature,
//     so the failure mode this module exists to catch cannot be laundered by the sampling;
//   * it CAN miss a change too small and too scattered to hit a sampled window, which UNDER-reports
//     the distinct rate.
//
// THE SCOPE OF THAT GUARANTEE: it holds for a SINGLE PUBLISHER, which is what prosper has. The
// signature is computed before this module's lock and `present_write_frame` takes its own lock
// separately, so two threads publishing concurrently could be observed in an order neither the
// present layer nor the guest saw — a true X,X,Y,Y recorded as X,Y,X,Y, which counts 4 distinct
// frames instead of 2. That is the OVER-reporting direction, the one the paragraph above says is
// impossible, so the condition is stated rather than left implicit. Today publication is serialized
// on the renderer thread (gpu_executor.cpp's two present_write_frame sites), and the app's
// test-pattern path only runs when no guest is publishing. **If a second concurrent publisher is
// ever added, this guarantee has to be re-established** — by folding the signature into the same
// critical section as the publication, or by ordering both against one sequence number.
// Concretely: any change to a contiguous run of at least
// kFrameSignatureBlockBytes + kFrameSignatureBytesPerBlock bytes is guaranteed to be seen. The
// distinct rate is thus a lower bound on the true rate of new frames, and it is documented as one
// wherever it is printed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace prosper::gpu {

// One 16-byte window is hashed out of every 4 KiB -- 0.391% of the frame at any resolution. Any
// change to a contiguous run of >= 4096 + 16 bytes must contain a whole window and is therefore
// detected.
//
// MEASURED, not estimated (Linux, -O2, 200 iterations per case, 2026-08-21):
//
//     3840x2160   92.1 us/frame   (31 MiB)
//     1920x1080   14.4 us/frame   (7 MiB)
//     1280x720     6.5 us/frame   (3 MiB)
//
// The 4K figure is the one that matters, and it is 0.55% of a 60 Hz frame budget -- against titles
// in this project that presently render at between 1 and 25 fps, where it is not measurable. It is
// also hashed OUTSIDE the present mutex (see note_present_publication), so it does not lengthen the
// window in which a reader is blocked. Re-measure with the bench in the PR before quoting these on
// other hardware; they are this machine's numbers, not a property of the algorithm.
constexpr size_t kFrameSignatureBlockBytes = 4096;
constexpr size_t kFrameSignatureBytesPerBlock = 16;

// Inter-distinct-frame intervals are bucketed rather than stored. A histogram costs the same for a
// ten-second run and an eight-hour one, which matters because this accumulates in the present layer
// for every title whether or not anyone asked for a framerate. 0.5 ms to ~90 s at 1.1x per bucket,
// which costs about 2 KiB.
//
// THE GROWTH FACTOR IS THE ACCURACY OF EVERY FRAMERATE THIS MODULE REPORTS, and the bound is
// analytic rather than tuned. A true interval anywhere in a bucket is reported as that bucket's
// geometric midpoint sqrt(lo*hi), and the bucket spans [lo, lo*growth], so the error is largest at
// an edge -- and the two edges are NOT symmetric. A true interval at the lower edge is over-reported
// by sqrt(growth) - 1; one at the upper edge is under-reported by 1/sqrt(growth) - 1, which is
// smaller. The bound below is the larger of the two, attained at the lower edge only:
//
//     growth 1.10  ->  +4.88% at the lower edge, -4.65% at the upper     <- chosen
//     growth 1.25  ->  +11.80%              ,     -10.56%
//
// 1.25 was the first choice and it was too coarse: it recovered a known 1.000 fps signal as 1.106
// fps, caught by the accuracy arm below. These figures get filed in game trackers and compared
// across releases, so if a future reader is tempted to widen the bucket to save the ~2 KiB, that is
// the arithmetic to face -- and `interval_estimator_is_accurate` will fail rather than let the
// numbers quietly get worse.
//
// ABOVE THE RANGE, the top bucket saturates: it merges the last normal bucket with the overflow, so
// any interval longer than ~82.1 s is reported as ~86.1 s rather than clamped to 90. That cannot
// reach `active_seconds` (which sums true durations, never bucket midpoints) and cannot move the
// median unless a title produces a frame less often than every 82 seconds -- at which point the
// distinction between "very slow" and "slower still" is not one anybody is reading this for.
constexpr size_t kIntervalBuckets = 128;
constexpr double kIntervalMinSeconds = 0.0005;
constexpr double kIntervalGrowth = 1.1;
// Worst-case relative error of a recovered interval, from the bucket width above. Exposed so the
// tests assert the DOCUMENTED tolerance rather than a number somebody tuned until it passed.
constexpr double kIntervalRelativeError = 0.05;

// An interval counts as ACTIVE when it is no longer than this multiple of the run's typical
// interval. Deliberately a multiple rather than a fixed duration: titles here range from ~1 fps to
// over 200 fps, and any absolute cutoff would classify one end of that range wrongly. At 60 fps
// (16.7 ms typical) a gap over 67 ms is a pause; at 1 fps (1 s typical) a gap over 4 s is. The
// constant shapes only the QUALIFIER -- the headline rate is a percentile and has no threshold in
// it at all, which is the property that makes it quotable.
constexpr double kActiveIntervalMultiple = 4.0;

// A bounded-cost content signature of one frame. Dimensions and byte count are folded in, so a
// resolution change is a content change even in the (impossible in practice) event that the sampled
// bytes agree. `pixels` may be null only when `bytes` is 0.
uint64_t frame_content_signature(const uint8_t* pixels, size_t bytes, uint32_t width, uint32_t height);

// The pure accumulator: publications in, counters out. Holds no clock and no lock of its own, so it
// is unit-testable without a present layer, a renderer or a Vulkan device — which is what lets the
// frozen-title arm be a fast, deterministic ctest case rather than a live run somebody has to
// remember to do.
class FrameRateCounter {
public:
    // `at_seconds` must come from a monotonic clock; the counter only ever subtracts two of them.
    void observe(uint64_t signature, double at_seconds);
    void reset();

    uint64_t published() const { return published_; }
    uint64_t distinct() const { return distinct_; }
    double first_publication_seconds() const { return first_; }
    double last_publication_seconds() const { return last_; }

    // Median interval between consecutive distinct frames, in seconds. Zero means "not measured":
    // fewer than two distinct frames, so there is no interval to take a median of. Zero is NOT a
    // fast frame time and callers must not treat it as one -- see FrameRate::typical_measured.
    double typical_interval_seconds() const;
    // Seconds spent in intervals no longer than kActiveIntervalMultiple x the typical interval.
    double active_seconds() const;
    uint64_t interval_samples() const { return interval_samples_; }

private:
    void record_interval(double seconds);
    size_t bucket_for(double seconds) const;

    uint64_t published_ = 0;
    uint64_t distinct_ = 0;
    uint64_t last_signature_ = 0;
    double first_ = 0;
    double last_ = 0;
    // Time of the previous DISTINCT publication, which is what the intervals are between.
    double last_distinct_ = 0;
    bool have_distinct_ = false;
    uint64_t interval_samples_ = 0;
    // Per bucket: how many intervals landed here, and how much wall time they account for. The
    // counts give the median; the seconds give the active share, and the two cannot disagree
    // because they are accumulated from the same event.
    uint64_t interval_counts_[kIntervalBuckets] = {};
    double interval_seconds_[kIntervalBuckets] = {};
};

// A reading of the process-wide counters, taken at `now_seconds` on the module's own monotonic
// epoch. Two snapshots subtract cleanly, so any window can be measured after the fact.
struct PresentRateSnapshot {
    uint64_t published = 0;
    uint64_t distinct = 0;
    double now_seconds = 0;                  // when this snapshot was taken
    double first_publication_seconds = 0;    // 0 when nothing has been published
    double last_publication_seconds = 0;
    // Derived from the interval histogram, which is cumulative for the whole process and therefore
    // cannot be differenced between two snapshots. Meaningful only via
    // frame_rate_since_first_publication; frame_rate_between leaves the corresponding FrameRate
    // fields unmeasured rather than computing something that would look valid and not be.
    double typical_interval_seconds = 0;     // 0 => fewer than two distinct frames
    double active_seconds = 0;
    uint64_t interval_samples = 0;
};

// Both rates over one window, plus the raw counts they were derived from. The counts travel with
// the rates on purpose: a rate with no population behind it is the thing this project's instrument
// traps are mostly made of, and "0.4 fps" from three frames is not the same claim as "0.4 fps" from
// four hundred.
struct FrameRate {
    bool measured = false;        // false => the window was empty or degenerate; the rates are 0
    double window_seconds = 0;
    uint64_t published = 0;
    uint64_t distinct = 0;
    double presented_fps = 0;
    double distinct_fps = 0;      // the run AVERAGE -- see the header note on why this is not the headline
    double distinct_fraction = 0; // distinct / published, 0 when nothing was published

    // THE HEADLINE. `typical_measured` false means fewer than two distinct frames arrived, which is
    // a different claim from 0 fps and must be rendered differently ("--", never "0.0").
    bool typical_measured = false;
    double typical_fps = 0;
    double typical_interval_seconds = 0;
    uint64_t interval_samples = 0;
    // Share of the window spent producing frames at roughly the typical rate, in [0, 1]. Always
    // shown beside typical_fps: on its own the headline could describe two frames half a second
    // apart in an otherwise dead run, and this is the field that says so.
    //
    // `active_fraction_measured` says whether the number below is a MEASUREMENT or an unset default,
    // and it exists because 0 otherwise means two incompatible things. On a run rate 0% is the
    // load-bearing verdict "this title produced nothing"; on a differenced window it means only that
    // the field was never filled, because frame_rate_between cannot fill it (see there). Anything
    // that prints the fraction -- or that reports frame_rate_is_mostly_unchanged, whose contract is
    // to print it -- has to test this first. #3027.
    bool active_fraction_measured = false;
    double active_fraction = 0;
};

// Window = [first publication, the moment the snapshot was taken]. Wall clock, not "time between
// the first and last publication": a title that STOPS publishing must decay toward zero rather than
// freeze at whatever it last managed.
FrameRate frame_rate_since_first_publication(const PresentRateSnapshot& snapshot);

// Window = [earlier, later]. Returns an unmeasured result if the counters moved backwards, which
// can only mean a reset happened between the two readings.
//
// `typical_fps` and `active_fraction` are NOT filled in here; `typical_measured` and
// `active_fraction_measured` both stay false. Callers that want a live rate over a short window
// (prosper-app's HUD) should use `distinct_fps`, which over a one-second window is not meaningfully
// an average of anything.
//
// TWO INDEPENDENT REASONS, AND THE SECOND IS THE ONE THAT SETTLES IT (#3027).
//
// 1. `active_seconds` is a whole-run RECOMPUTATION, not an accumulator, so it cannot be differenced.
//    Each snapshot re-derives it against the CURRENT median interval, and that cutoff moves as the
//    run goes on -- so the quantity subtracted at the near end is not the quantity added at the far
//    one. Measured on the counter itself (the arm in tests/gpu/present pins all three):
//      * 201 publications at 1 fps then 1000 at 60 fps takes active_seconds from 200.000 s to
//        16.650 s, so the difference is NEGATIVE: -183.4 s across a 16.7 s window, "-1100% active";
//      * a window holding 6000 consecutive frames at a perfect 60 fps differences to 1.0% active;
//      * a window holding 100 frames at a healthy-and-slow 1 fps differences to 0.02%.
//    The last two are the dangerous ones: they are positive, plausible, and read as exactly the
//    "produced nothing" shape this metric exists to flag.
// 2. Even a perfectly differenceable active share would answer the wrong question. The distinction
//    active_fraction protects -- "a static menu that HAS produced frames" against "a title that has
//    produced nothing" -- is a property of the RUN, not of the window. Inside one second the two are
//    indistinguishable by construction, because in both cases nothing changed during that second. No
//    statistic computed from the window alone can separate them; the answer lives in the cumulative
//    snapshot. That is what `unchanged_picture` below takes, and why it takes it.
FrameRate frame_rate_between(const PresentRateSnapshot& earlier, const PresentRateSnapshot& later);

// Two lines. The first is the headline and its qualifier and nothing else; the second is every
// number it was derived from, so the headline can be checked without being competed with:
//
//   18.5 fps while producing frames, 62% of the 380.0 s run active
//   (1142 distinct of 78743 published; run average 3.0 fps; presented 207.2 fps)
//
// A run that produced fewer than two distinct frames gets "-- fps ... 0% active" and a sentence
// naming the count, because that is the case a number of any kind would misrepresent.
std::string format_frame_rate(const FrameRate& rate);

// A compact form for burning into an image or drawing in a HUD:
// "18.5 fps  62% active  3840x2160". Falls back to the run average when the typical rate is not
// available (a short window, or frame_rate_between), and to "--" when nothing was produced.
//
// The "0% active" that accompanies "--" is a MEASUREMENT -- it is the sentence "this run produced
// nothing" -- so it is emitted only when `active_fraction_measured` says the fraction was computed.
// A differenced window gets "-- fps  WxH" with no active claim, because one second of stillness is
// not evidence about the run (#3027).
std::string format_frame_rate_short(const FrameRate& rate, uint32_t width, uint32_t height);

// True when publications kept arriving but almost none of them carried new content.
//
// READ THE NAME LITERALLY. It says UNCHANGED, not "retained", and the difference is the whole
// caveat: this predicate CANNOT distinguish
//
//   (a) the renderer re-serving its retained frame because a submit produced no present source
//       — a defect, the R-Type Delta (#2783) shape; from
//   (b) a title sitting on a static picture — a menu, a title screen, a pause — where publishing
//       the same image repeatedly is completely correct.
//
// Both look identical from here BY CONSTRUCTION: in each case the bytes do not change. An earlier
// version of this was called `..._is_mostly_retained` and its caller printed a warning naming the
// renderer, which fired on *The Messenger*'s own title screen — a rung-6 title, on its guarded
// route — and told the reader to go and investigate a renderer defect that was not there. A
// measurement that manufactures phantom defects in a tool whose output lands in game trackers is
// worse than no measurement, so the name states the observation and nothing about its cause.
//
// `active_fraction` is what separates (a) from (b): a static menu still produced frames before it
// arrived, so its active share is non-zero; a title that produced nothing has an absent typical rate
// and 0% active. Any caller that reports this predicate must report that alongside it -- WHICH IS
// POSSIBLE EXACTLY WHEN `rate.active_fraction_measured` IS TRUE, i.e. when the rate came from
// `frame_rate_since_first_publication`. Those are tools/screenshot's summary line and its manifest;
// they print the fraction on the spot, and this predicate is theirs.
//
// A WINDOWED CALLER MUST NOT REPORT THIS PREDICATE BARE -- use `unchanged_picture` below. Until
// #3027 the requirement above was stated unconditionally, which made it unsatisfiable for every
// caller of `frame_rate_between`: it demanded a field that constructor cannot fill, for the two
// reasons spelled out there. So the requirement stands, and the windowed form of it takes the
// cumulative snapshot as an ARGUMENT rather than leaving it as an instruction nobody could follow.
//
// The thresholds are named parameters so nobody has to guess what "almost none" meant, and the
// predicate keys on the FRACTION rather than an absolute rate because titles here legitimately run
// at ~1 fps and an absolute threshold would call those frozen.
bool frame_rate_is_mostly_unchanged(const FrameRate& rate,
                                    uint64_t min_published = 30,
                                    double max_distinct_fraction = 0.5);

// The windowed form of that predicate, and the reason this file has two.
//
// A rolling window can see that the picture stopped changing. It cannot see whether the title ever
// produced anything, because within one second a static menu and a re-served retained frame are the
// same bytes. So the classification takes BOTH: the window that fired the predicate, and the
// cumulative snapshot that says what the run has managed so far. The second argument is not a
// convenience -- it IS the disambiguating figure the contract above demands, made impossible to
// omit.
enum class PictureChange : uint8_t {
    changing,          // the window carried new content -- there is nothing to report
    static_picture,    // unchanged across this window, and the run HAS produced frames: case (b)
    nothing_produced,  // unchanged, and the run has produced nothing at all: the case (a) shape
};

// The verdict and the run-wide evidence for it in one value, so a caller cannot hold the first
// without the second. The `run_*` fields come from frame_rate_since_first_publication(run) and are
// filled whatever the verdict is.
struct UnchangedPicture {
    PictureChange change = PictureChange::changing;
    bool run_typical_measured = false;   // false => fewer than two distinct frames in the WHOLE run
    double run_typical_fps = 0;
    double run_active_fraction = 0;
    uint64_t run_distinct = 0;
};

// `window` is what frame_rate_between returned; `run` is the cumulative snapshot the window's later
// end was read from. The two thresholds are frame_rate_is_mostly_unchanged's, unchanged.
//
// `nothing_produced` keys on the RUN having no typical rate -- which is literally the "absent
// typical rate and 0% active" the paragraph above names -- rather than on a threshold somebody
// tuned. The borderline case (two distinct frames a long way apart) is left LEGIBLE rather than
// arbitrated: the formatter prints the rate, the share and the distinct COUNT, so a reader can see
// the population a verdict rests on instead of inheriting a hidden cutoff.
UnchangedPicture unchanged_picture(const FrameRate& window, const PresentRateSnapshot& run,
                                   uint64_t min_published = 30,
                                   double max_distinct_fraction = 0.5);

// The line a caller prints for that verdict; empty for `changing`. It always carries the run
// figures, which is the entire point -- the words on their own are what was ambiguous:
//
//   picture not changing; run so far 59.4 fps, 96% active, 1204 distinct
//   picture not changing; run so far -- fps, 0% active, 1 distinct (nothing produced yet)
std::string format_unchanged_picture(const UnchangedPicture& picture);

// ---- process-wide counters, fed by the present layer ---------------------------------------
//
// Called by present_write_frame for every accepted publication, before it takes the present mutex:
// the signature is computed from the caller's own bytes and needs none of the present layer's
// state, so it must not lengthen that critical section.
void note_present_publication(const uint8_t* pixels, size_t bytes, uint32_t width, uint32_t height);

// The GPU present path (#1270) has no CPU pixels: prosper-app blits the renderer's front-buffer image
// straight to the swapchain and the renderer skips the CPU readback entirely (live_renderer.cpp), so
// note_present_publication above is never reached and this counter stayed unfed for the whole life of
// every interactive boot -- the HUD read "no frames published yet" while thousands of frames were on
// screen (#3010). That path feeds these two instead: it samples the presented image into a small
// host-visible buffer on the GPU, signs the sample DENSELY, and publishes the signature.
//
// Dense, not sampled, and the distinction is the point. frame_content_signature reads 16 bytes per
// 4 KiB block, which is the right ratio for a multi-megabyte frame but would reduce a 256x144 sample
// to ~144 sampled pixels -- weak enough to call a changing picture frozen. Signing every byte of a
// deliberately small point-sampled image keeps the sensitivity in ONE place (the sample grid) instead
// of multiplying two lossy stages together.
//
// The resulting distinct rate remains a LOWER bound on the true rate of new frames, for the same
// reason the CPU path's is: a change confined to pixels the grid misses is invisible to it. The grid
// is 36,864 points against the CPU path's ~8,100 sampled pixels at 1080p, so it is a tighter bound
// than the path it stands in for -- but it is still a bound, and must not be read as an exact count.
//
// One asymmetry to know about: these two entry points sign in DIFFERENT spaces (a dense 256x144
// sample here, a sparse whole-frame hash there), so a run that alternates between them -- GPU
// present with occasional publish misses falling back to the CPU path -- records a distinct frame at
// every crossing even when the picture did not change. Misses are rare enough that this does not
// move a rate in practice, but it means the distinct count is not a strict lower bound on a MIXED
// run the way it is on either path alone.

uint64_t dense_content_signature(const uint8_t* pixels, size_t bytes);

// Publish a signature computed by the caller. Same accounting as note_present_publication -- one
// publication, distinct if the signature differs from the immediately preceding one.
void note_present_publication_signature(uint64_t signature);


// Cleared by present_reset(), so a test that resets the present layer resets the rate with it.
void reset_present_rate();

PresentRateSnapshot present_rate_snapshot();

} // namespace prosper::gpu
