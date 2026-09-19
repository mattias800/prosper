// hle_service.cpp — HLE of PS5 system services (user, NP/online, mouse, app content,
// dialogs). Bring-up policy: openers return a valid positive handle; queries zero their
// output struct and report a sane "not signed in / no device" state and success, so the
// game gets consistent values instead of uninitialized memory.
// (Game-controller input — libScePad — moved to hle_pad.cpp with a real host backend.)
#include "hle/dispatch/dispatch.hpp"
#include "hle/service/hle_addcontent.hpp"
#include "hle/fs/save_paths.hpp"   // per-title save roots (#2734)
#include "hle/fs/save_param.hpp"   // the save's parameter block (#2786)
#include "hle/util/hle_json2.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"   // libkernel error encoding (libSceRandom reject arms)
#include "diagnostics/env_numeric.hpp"   // #3267: a typo must not unregister a default-ON NID family
#include "hle/dispatch/callback_fs.hpp"
#include "hle/input/ime_input.hpp"
#include "hle/service/platform_ui.hpp"
#include "hle/video/video_backend.hpp"   // sceAvPlayer -> host hardware-decode backend (#705)
#include "hle/video/h264_sps.hpp"        // SPS/VUI extraction for GetPictureInfo (#2898)
#include "gpu/texture/guest_texture_layout.hpp" // exact HLE-produced sampled-linear layouts
#include "host/platform/posix_shim.hpp"   // Darwin process_vm_readv shim + asm portability
#include "host/image/boot_program.hpp"  // guest_module_name: is a callback target guest code?
#include "host/platform/lifecycle.hpp"  // cooperative stop when the guest reports its own crash (#3119)
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>
#include <deque>
#include <filesystem>
#include <mutex>
#include <new>          // std::bad_alloc — the guest file-replacement buffer is guest-sized (#1955)
#include <set>
#include <unordered_map>
#include <vector>
#include <string>
#ifdef _WIN32
#include <direct.h>     // _mkdir (SaveDataMemory persistence dir)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>   // mkdir
#include <sys/uio.h>    // process_vm_readv: fault-contained diagnostic snapshots
#include <sys/random.h> // getentropy: the host CSPRNG behind sceRandomGetRandomNumber
#include <unistd.h>
#endif
#include "hle/service/service_trace.hpp"
#include "hle/service/hle_handles.hpp"
#ifdef _WIN32
#include <bcrypt.h>     // BCryptGenRandom (prosper_core already links bcrypt on Windows)
#endif

namespace prosper {

#ifdef _WIN32
extern "C" uint64_t prosper_call_guest_sysv4(uint64_t fn, uint64_t a0, uint64_t a1,
                                               uint64_t a2, uint64_t a3);
#endif

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define PW(x) ((void*)(uintptr_t)(x))


// Diagnostic logging for the Sony service families in this file, gated on PROSPER_SVCLOG=1 (same
// pattern as PROSPER_FILELOG/[file]). Dumps call args and a bounded hexdump of pointer-shaped args
// so PS5-only ABIs with no Kyty/shadPS4 reference can be pinned from live captures instead of
// guessed.

// PS5 Game Intent is how the shell starts a title at a selected activity instead of its default
// entry point. A host frontend can model that user action by setting an activity id before boot;
// without it the console truthfully has no pending intent. Keep the payload opaque to the guest:
// titles consume it through sceNpGameIntentGetPropertyValueString, as on the console.
// CONFIDENCE: HIGH — ABI/error contracts match Kyty; event id, type, key, and offsets are also
// independently exercised by Sonic's eboot and its launchActivity declaration in param.json.
namespace {
}
// --- libSceVideodec2 ---------------------------------------------------------------------------
// These ABI layouts/NIDs are the PS5 3.20 interfaces.  Decoding is intentionally a no-picture
// implementation for now: it consumes each access unit, reports a valid lifecycle, and lets the
// movie reach EOF instead of faking video pixels.  Critically, every successful query/open writes
// all output fields.  The former generic success stubs left a null compute queue and crashed libc
// at +0xfe66 on its first queue use.
//
// WHO ACTUALLY EXERCISES THIS, as recorded rather than as assumed.  A header line here used to read
// "Sonic's CRI Mana movie backend uses the native VideoDec2 compute queue and decoder lifecycle."
// It came from #1368 (8b37be95) — the same commit that added the unexplained compute_queue check
// below and an equally unsupported "Sonic Origins' CRI Mana backend passes it", which was already
// withdrawn.  No Videodec2 evidence for Sonic exists anywhere in docs/ or COMPATIBILITY.md, so it is
// withdrawn on the same basis rather than left to be inherited a third time.  What IS recorded:
//   - QueryComputeMemoryInfo only — Dragon Quest VII Reimagined and the UE4 bring-up.
//   - The full decoder lifecycle — Tales of Graces f Remastered (PPSA19991) via criMvPly, and only
//     after #1687; before that no title in this repo's history had ever reached CreateDecoder.
namespace {
// The current no-picture backend has no hardware decoder workspace: it only needs a stable,
// caller-owned address for the opaque compute-queue identity.  Keep one guest page so clients that
// require a non-null allocation can follow the real query/allocate lifecycle without reserving the
// 16 MiB placeholder used by software-decoding implementations.  That placeholder exhausted
// Sonic Origins' 80 MiB CRI CPU/GPU pool before its movie workers could allocate their own buffers.
constexpr uint64_t VDEC_MIN_MEMORY = 64ull << 10;
constexpr uint64_t VDEC_ERR_STRUCT = 0x811d0101ull;
constexpr uint64_t VDEC_ERR_ARG = 0x811d0102ull;
constexpr uint64_t VDEC_ERR_DECODER = 0x811d0103ull;
constexpr uint64_t VDEC_ERR_MEMORY_SIZE = 0x811d0104ull;
constexpr uint64_t VDEC_ERR_MEMORY_PTR = 0x811d0105ull;
constexpr uint64_t VDEC_ERR_FRAME_SIZE = 0x811d0106ull;
constexpr uint64_t VDEC_ERR_FRAME_PTR = 0x811d0107ull;
constexpr uint64_t VDEC_ERR_CONFIG = 0x811d0200ull;
constexpr uint64_t VDEC_ERR_PIPE = 0x811d0201ull;
constexpr uint64_t VDEC_ERR_QUEUE = 0x811d0202ull;
constexpr uint64_t VDEC_ERR_RESOURCE = 0x811d0203ull;
constexpr uint64_t VDEC_ERR_INPUT_DEPTH = 0x811d0206ull;
constexpr uint64_t VDEC_ERR_DPB = 0x811d0209ull;
constexpr uint64_t VDEC_ERR_DIMS = 0x811d020aull;

struct VdecComputeMemory { uint64_t size, shared_size, shared; };
struct VdecComputeConfig {
    uint64_t size; uint16_t pipe, queue; uint8_t check_memory, reserved0; uint16_t reserved1;
};
struct VdecConfig {
    uint64_t size; uint32_t resource, codec, profile, max_level;
    int32_t max_width, max_height, max_dpb; uint32_t input_depth;
    uint64_t compute_queue, affinity; int32_t priority;
    uint8_t optimize, check_memory, reserved0, reserved1; uint64_t extra;
};
struct VdecMemory {
    uint64_t size, cpu_size, cpu, gpu_size, gpu, shared_size, shared, max_frame_size;
    uint32_t frame_alignment, reserved;
};
struct VdecInput { uint64_t size, data, data_size, pts, dts, attached; };
struct VdecFrame { uint64_t size, data, data_size; uint8_t accepted, pad[7]; };
struct VdecOutput {
    uint64_t size; uint8_t valid, error, pictures, discarded;
    uint32_t codec, width, pitch, height; uint64_t frame, frame_size;
    uint32_t format, pitch_bytes;
};
static_assert(sizeof(VdecComputeMemory) == 24 && sizeof(VdecComputeConfig) == 16);
static_assert(sizeof(VdecConfig) == 72 && sizeof(VdecMemory) == 72);
static_assert(sizeof(VdecInput) == 48 && sizeof(VdecFrame) == 32 && sizeof(VdecOutput) == 56);
// LIVE CONFIRMATION of the two structs above, from the first title to reach the decoder lifecycle
// (Tales of Graces f Remastered PPSA19991, criMvPly, PROSPER_SVCLOG; see #1658). These were written
// in #1368 without title evidence; a real guest now writes values that are meaningful at nearly
// every offset, which is much stronger than a matching size:
//
// The two calls are labelled, because they are NOT the same dump: everything down to +0x28 appears
// in both, while affinity/priority differ between them and are therefore CreateDecoder's values.
//
//   VdecConfig  +0x00 size=0x48  +0x08 resource=1     +0x0c codec=1 (AVC)          [both calls]
//               +0x10 profile=100 (High)              +0x14 max_level=41 (4.1)     [both calls]
//               +0x18 max_width=1920                  +0x1c max_height=1088        [both calls]
//               +0x20 max_dpb=-1 (auto)               +0x24 input_depth=4          [both calls]
//               +0x28 compute_queue = 0                                            [sizing query]
//               +0x28 compute_queue = the allocated handle                         [CreateDecoder]
//               +0x30 affinity=0x1fff                 +0x38 priority=700           [CreateDecoder]
//               (the sizing query's affinity/priority are 0 / -1)
//   VdecMemory  +0x00 size=0x48, then alternating size/pointer slots at +0x08..+0x38
//
// NOTE the observation boundary: svc_log clamps to 8 qwords, so despite both call sites asking for
// 9, NOTHING at +0x40 was ever dumped — VdecConfig::extra and VdecMemory::frame_alignment/reserved
// are unobserved, not confirmed. The table stops at +0x38 for that reason. (The clamp is silent and
// is itself instrument-trap-13 shaped; filed rather than fixed here to keep this PR narrow.)
//
// Most of those pairings are FORCED, not merely consistent, which is what makes this evidence and
// not a coincidence: swap profile/max_level and you get profile=41 (not a valid H.264 profile_idc)
// with level=100 (not a valid level_idc); swap max_dpb/input_depth and input_depth is negative;
// swapping width/height would make the movie portrait 1088x1920.
//
// One pair is NOT discriminated and is called out rather than glossed: `resource` and `codec` are
// BOTH 1 in this title, so this evidence cannot tell their order apart. It rests on #1368 there, and
// a title using a non-AVC codec or a different resource selector would settle it in one line of log.
// So: CONFIDENCE: HIGH on VdecConfig's layout except the resource/codec pair, which stays MED.
//
// VdecMemory needs the SAME discipline applied to it, and it does not survive it as well. "The guest
// echoed back the sizes we wrote" is near-tautological: this file wrote one identical constant into
// cpu_size, gpu_size, shared_size AND max_frame_size, so the echo is byte-identical under any
// permutation of those four, and the three pointer slots are three distinct guest addresses carrying
// no role marker. What the echo genuinely proves is the SHAPE — size=0x48 at +0x00 and alternating
// size/pointer slots thereafter — which is real and non-trivial. So: CONFIDENCE: HIGH on VdecMemory's
// shape, MED on which of the four size roles and three pointer roles sits in which slot; those still
// rest on #1368. (Now that the sizing query reports a max_frame_size distinct from the other three,
// the next boot's Decode call discriminates that one field for free.)
//
// VdecInput's first five fields are confirmed (size=0x30, data, data_size) with the same caveat on
// the last two: pts and dts are BOTH -1 here and are mutually undiscriminated. VdecFrame's first
// three are confirmed (size=0x20, data, data_size). Only the SIZE of VdecOutput is confirmed (0x38)
// — its interior is written by us, not the guest, and the AVC picture-info layout remains open
// (#1658).
//
// One naming caveat, deliberately not "fixed": VdecInput's last field, named `attached` here,
// carried a clean 0,1,2,...  sequence across consecutive access units rather than a flag or a
// pointer. That is evidence it may be an access-unit index, not an attachment. Nothing depends on
// the name today and no title evidence settles it, so it is recorded rather than renamed.

std::mutex g_vdec_mx;
std::unordered_map<uint64_t, uint32_t> g_vdec_codecs;
// #2270 access-unit decoders, one per guest decoder handle. `id` is the backend's decoder id, or
// -1 when the backend refused the codec (or there is no backend at all) -- `opened` distinguishes
// "not tried yet" from "tried and refused", so the refusal is announced ONCE rather than on every
// access unit or, worse, never. `no_picture_run` counts consecutive access units that produced no
// picture from a decoder that DID open: a real decoder needs a handful of units before its first
// frame, so a long run of them is a decoder that is never going to produce one, and that is the
// exact shape #2270 was filed about -- indistinguishable, without this, from "no frame ready yet".
// `announced` survives Reset while `opened` does not, and that difference is the finding it fixes:
// Reset re-arms the open so a supported codec gets a fresh decoder, but a REFUSED codec would then
// re-print its two-line NO DECODER banner on every reset cycle. A title that resets in a retry loop
// turns the fail-visible path into a flood, which is how fail-visible paths get muted (#2571 N3).
struct VdecAu { int id = -1; bool opened = false; bool announced = false; unsigned no_picture_run = 0; };
std::unordered_map<uint64_t, VdecAu> g_vdec_au;

// Per-decoded-picture metadata for sceVideodec2GetPictureInfo (#2898).
//
// GetPictureInfo takes NO decoder handle -- its arguments are (outputInfo*, pictureInfo*,
// 0, 0) at every observed call site (PPSA06367 x2, PPSA29343 x3) -- so the only way to
// know which picture to describe is the one thing that identifies it: outputInfo->frame,
// the pointer Decode wrote into the caller's output struct. Keying on it means a title
// that recycles frame buffers collapses those pictures onto one metadata entry; SPS/VUI
// values rarely change within a stream, so the practical cost is negligible and it is
// recorded rather than hidden (CONFIDENCE: MED on cross-picture identity, HIGH on the
// offsets themselves).
//
// `record` is a small prosper-owned block handed to the guest through pictureInfo+0x20.
// Guest evidence about its SHAPE (#2898, refined by live boot once the pointer was no
// longer null -- the fault moved from "deref of the null pointer field" to "walk of the
// block behind it", which is what named the extra indirection):
//
//   P = *(void**)record;                    // +0x00 is a POINTER to the payload block
//   string_assign(s, (char*)P + 0x08);      // C-string at payload+0x08
//   if (payload[+0x30]) use qword payload[+0x28] else built-in default text;
//
//   (PPSA29343 never dereferences; it compares `record` values for identity across its
//   per-decoder arrays, which any stable per-picture address satisfies.)
//
// Our block therefore points its first qword at its own +0x08, where the payload lives:
// an EMPTY string and flag = 0, which takes the guest's default-text path -- nothing is
// fabricated. Records are allocated once per distinct frame buffer and deliberately
// NEVER freed while the process lives -- freeing them would turn Beast's stored keys
// (and any reference the guest kept) dangling, which is worse than a bounded few hundred
// bytes.
struct VdecPicMeta {
    bool has_meta = false;
    prosper::h264::SpsPictureMeta meta;
    void* record = nullptr;
};
constexpr size_t kVdecInfoRecordSize = 0x48;  // 8-byte self-pointer + 0x40 payload
std::unordered_map<uint64_t, VdecPicMeta> g_vdec_picmeta;

// A rejected decoder config is a hard stop for a title's movie playback, and the guest only prints the
// bare SCE code — 0x811d0200 covers TWO different conditions here, so "err=0x811d0200" alone cannot tell
// anyone which. Name the failing field and its observed value, unconditionally but rate-limited: this
// fires at most a handful of times per boot and is the difference between a one-run diagnosis and a
// guess. (Tales of Graces f / criMvPly stopped exactly here, on `compute_queue`, which is the
// evidence the split below rests on; see #1658.)
uint64_t vdec_reject(const char* field, uint64_t observed, uint64_t err) {
    // Rate-limit PER FIELD, not globally. One shared counter meant a caller looping on an early
    // condition could burn the whole budget and silence a later, different rejection — including the
    // one a run was started to capture. `field` is always a string literal, so the pointer is a stable
    // key and the map stays bounded by the number of call sites.
    static std::mutex mx;
    static std::unordered_map<const char*, int> shown;
    {
        std::lock_guard<std::mutex> lk(mx);
        if (shown[field]++ >= 4) return err;
    }
    fprintf(stderr, "[vdec] decoder config REJECTED: %s = 0x%llx -> err=0x%llx\n",
            field, (unsigned long long)observed, (unsigned long long)err);
    return err;
}
// `require_queue` splits the compute-queue requirement between the two callers of this validator.
// See the long note at the check itself for why the two differ.
uint64_t vdec_validate_config(const VdecConfig* c, bool require_queue) {
    if (!c) return VDEC_ERR_ARG;
    if (c->size != sizeof(*c)) return vdec_reject("size", c->size, VDEC_ERR_STRUCT);
    if (c->resource != 1) return vdec_reject("resource", c->resource, VDEC_ERR_RESOURCE);
    if (c->reserved0 || c->reserved1)
        return vdec_reject("reserved0/1", ((uint64_t)c->reserved0 << 8) | c->reserved1,
                           VDEC_ERR_CONFIG);
    if (!c->input_depth) return vdec_reject("input_depth", c->input_depth, VDEC_ERR_INPUT_DEPTH);
    if (c->max_dpb < -1 || c->max_dpb == 0)
        return vdec_reject("max_dpb", (uint64_t)(int64_t)c->max_dpb, VDEC_ERR_DPB);
    if (c->max_width < -1 || c->max_height < -1 || !c->max_width || !c->max_height)
        return vdec_reject("max_width/height",
                           ((uint64_t)(uint32_t)c->max_width << 32) | (uint32_t)c->max_height,
                           VDEC_ERR_DIMS);
    // THE SPLIT (#1658). A compute queue is required to CREATE a decoder and is NOT required to ask
    // how large one would be.
    //
    // The check arrived in #1368 (8b37be95) with no comment, no rationale and no test — defensive,
    // not derived — and applied to both callers. The predecessor comment here predicted exactly how
    // that would fail ("most likely to reject a caller that queries memory sizes BEFORE it allocates
    // a compute queue") and pre-registered this split as the evidenced move if a live log ever named
    // the field. A Tales of Graces f (PPSA19991) boot then named it: 7 calls to
    // sceVideodec2QueryDecoderMemoryInfo, all 7 rejected on `compute_queue = 0`, zero calls to
    // sceVideodec2CreateDecoder — so the guest is sizing a decoder it has not built yet, which is
    // precisely when it cannot hold a queue handle. Everything else in that config validates
    // (size=0x48, resource=1, AVC High@4.1, 1920x1088, max_dpb=-1 auto, input_depth=4).
    //
    // Rejecting the sizing query also protected no size contract: s_videodec2_query_decoder_memory
    // ignores compute_queue entirely, so the rejection only failed EARLIER than succeeding would.
    //
    // The requirement stays on CreateDecoder, which genuinely needs a queue to build a decoder on,
    // and which is where a caller that truly has none must still fail visibly rather than receive an
    // invented handle.
    //
    // CONFIDENCE: HIGH that the sizing query must not require it — live title evidence, and
    // *prosper's* query cannot use the field. (That second clause argues from our own implementation,
    // not from Sony's: a real library could legitimately consult a queue handle for queue-specific
    // alignment. The HIGH rests on the live evidence alone.)
    //
    // CONFIDENCE: MED that CreateDecoder must require it. The same title now DOES reach CreateDecoder
    // and supplies a real queue there — but that proves the guest HAS one, not that the library
    // demands one, and those are different claims. So this half remains the #1368 default,
    // deliberately kept fail-visible rather than relaxed on the strength of a caller being
    // well-behaved.
    if (require_queue && !c->compute_queue)
        return vdec_reject("compute_queue", c->compute_queue, VDEC_ERR_CONFIG);
    return 0;
}
// How many consecutive no-picture access units from an OPEN decoder are still plausible before the
// run is announced as a stall. A conforming decoder emits its first picture within a small multiple
// of its reordering depth; the guest's own config asks for input_depth=4 and max_dpb=-1 (auto), and
// the observed streams are IDR-first. 64 is two orders of magnitude above that and still fires
// within about a second of movie time, so it cannot mistake warm-up for a stall in either direction.
constexpr unsigned kVdecNoPictureAlarm = 64;

// The fail-visible announcement #2270 exists for. Everything below this returns SCE_OK with no
// picture, which is the CORRECT answer for "the decoder needs more input" and the catastrophic one
// for "there is no decoder" -- the guest cannot tell them apart, so it waits forever with nothing in
// the log. Print the codec AND the first bytes of the access unit: the codec field alone does not
// name a bitstream, while the head does (00 00 00 01 09 / 67 is Annex-B H.264 with an access-unit
// delimiter and an SPS; a VP9 key frame carries the sync code 49 83 42 at bytes 1..3).
void vdec_no_decoder_warning(uint32_t codec, const VdecInput* input, const char* why) {
    char head[64] = {0};
    if (input && input->data && input->data_size) {
        const auto* au = (const uint8_t*)(uintptr_t)input->data;
        const uint64_t n = input->data_size < 16 ? input->data_size : 16;
        for (uint64_t i = 0; i < n; ++i) snprintf(head + i * 3, 4, "%02x ", au[i]);
    }
    fprintf(stderr,
            "[vdec2] sceVideodec2Decode: NO DECODER (%s) for codec=%u -- au_bytes=%llu head=%s\n"
            "[vdec2]   Every call will report SCE_OK with NO PICTURE, which a title cannot "
            "distinguish from \"no frame ready yet\": if it waits for a decoded frame it will wait "
            "forever. This is #2270; the head bytes above name the bitstream a decoder must accept.\n",
            why, codec, input ? (unsigned long long)input->data_size : 0ull, head);
}

// PROSPER_VDEC2_DUMP_DIR -- the instrument that lets a decoded picture be CHECKED rather than
// believed (#2270). It writes two files into the named directory:
//
//   au.bin     every access unit the guest submitted, concatenated in submission order. The
//              observed streams are Annex-B (start-code delimited), so the concatenation is itself
//              a playable elementary stream: `ffmpeg -i au.bin -f rawvideo -pix_fmt nv12 ref.nv12`.
//   pic.nv12   the exact bytes prosper wrote into the guest's frame buffer, same order.
//
// `cmp ref.nv12 pic.nv12` is then an INDEPENDENT decode of the guest's own bitstream against ours.
// That distinction is the reason this exists: a decoded picture that merely "looks like a movie" is
// not evidence — this project has a recorded trap where a plausible palette convinced the eye and a
// gradient beat real content on a metric. H.264 reconstruction is normatively exact, so byte
// equality against another decoder's output is a claim that cannot be faked by a wrong chroma plane,
// a swapped U/V, or a stale reference frame, all of which produce something that still looks filmic.
//
// BOUNDED, because it is not: a 1920x1088 NV12 picture is 3.1 MB and these titles feed thousands.
// PROSPER_VDEC2_DUMP_FRAMES (default 16) caps both files together, so a full-rate movie costs 50 MB
// rather than 5 GB. Nothing is written unless the directory variable is set.
struct VdecDump {
    FILE* au = nullptr;
    FILE* pic = nullptr;
    unsigned limit = 16;
    unsigned aus = 0, pics = 0;
    std::mutex mx;
};
VdecDump& vdec_dump() {
    // A std::mutex member makes VdecDump immovable, so the usual `static X x = []{...}()` form does
    // not compile here. Initialise in place under a once-flag instead.
    static VdecDump d;
    static std::once_flag once;
    std::call_once(once, [] {
        const char* dir = getenv("PROSPER_VDEC2_DUMP_DIR");
        if (!dir || !*dir) return;
        if (const char* n = getenv("PROSPER_VDEC2_DUMP_FRAMES")) {
            // `if (parsed)` refuses only a value that parses to 0. `=-1` saturated to ULONG_MAX and
            // lifted the 16-frame cap this block's own comment describes as the difference between
            // 50 MB and 5 GB on disk. Base 0 was the pre-existing grammar, so keep `0x` (#3267 B1).
            const uint64_t parsed = prosper::diag::env_u64_or_default_auto_capped(
                "PROSPER_VDEC2_DUMP_FRAMES", n, 16ull, UINT32_MAX, "frames");
            if (parsed) d.limit = (unsigned)parsed;
        }
        const std::string base(dir);
        d.au = fopen((base + "/au.bin").c_str(), "wb");
        d.pic = fopen((base + "/pic.nv12").c_str(), "wb");
        fprintf(stderr, "[vdec2] dump: au=%s pic=%s limit=%u frames (#2270)\n",
                d.au ? "open" : "FAILED", d.pic ? "open" : "FAILED", d.limit);
    });
    return d;
}
void vdec_dump_au(const uint8_t* au, uint64_t bytes) {
    VdecDump& d = vdec_dump();
    if (!d.au || !au || !bytes) return;
    std::lock_guard<std::mutex> lk(d.mx);
    if (d.aus >= d.limit) return;
    fwrite(au, 1, (size_t)bytes, d.au); fflush(d.au);
    ++d.aus;
}
void vdec_dump_picture(const uint8_t* nv12, size_t bytes, uint32_t w, uint32_t h) {
    VdecDump& d = vdec_dump();
    if (!d.pic || !nv12 || !bytes) return;
    std::lock_guard<std::mutex> lk(d.mx);
    if (d.pics >= d.limit) return;
    if (d.pics == 0)
        fprintf(stderr, "[vdec2] dump: pictures are %ux%u NV12, %zu bytes each -- compare with\n"
                        "[vdec2]   ffmpeg -i <dir>/au.bin -f rawvideo -pix_fmt nv12 <dir>/ref.nv12\n"
                        "[vdec2]   cmp <dir>/ref.nv12 <dir>/pic.nv12\n", w, h, bytes);
    fwrite(nv12, 1, bytes, d.pic); fflush(d.pic);
    ++d.pics;
}

void vdec_no_picture(VdecFrame* frame, VdecOutput* out, uint32_t codec) {
    out->valid = out->error = out->pictures = out->discarded = 0;
    out->codec = codec; out->width = out->pitch = out->height = 0;
    out->frame = frame ? frame->data : 0;
    out->frame_size = frame ? frame->data_size : 0;
    out->format = out->pitch_bytes = 0;
}
}

HLE(s_videodec2_query_compute_memory) {
    svc_log("sceVideodec2QueryComputeMemoryInfo", a0, a1, a2, a3, a4, a5, 3);
    auto* info = (VdecComputeMemory*)PW(a0);
    if (!info) return VDEC_ERR_ARG;
    if (info->size != sizeof(*info)) return VDEC_ERR_STRUCT;
    info->shared_size = VDEC_MIN_MEMORY; info->shared = 0;
    return 0;
}
HLE(s_videodec2_allocate_compute_queue) {
    svc_log("sceVideodec2AllocateComputeQueue", a0, a1, a2, a3, a4, a5, 3);
    auto* config = (const VdecComputeConfig*)PW(a0);
    auto* memory = (const VdecComputeMemory*)PW(a1);
    auto* out = (uint64_t*)PW(a2);
    if (!config || !memory || !out) return VDEC_ERR_ARG;
    if (config->size != sizeof(*config) || memory->size != sizeof(*memory)) return VDEC_ERR_STRUCT;
    // These share 0x811d0200/0x811d02xx with the decoder-config validator, so a bare guest error code
    // cannot say which call produced it. Name them too, or a run that fails HERE prints nothing.
    if (config->reserved0 || config->reserved1)
        return vdec_reject("computeQueue.reserved0/1",
                           ((uint64_t)config->reserved0 << 8) | config->reserved1, VDEC_ERR_CONFIG);
    if (config->pipe > 4) return vdec_reject("computeQueue.pipe", config->pipe, VDEC_ERR_PIPE);
    if (config->queue > 7) return vdec_reject("computeQueue.queue", config->queue, VDEC_ERR_QUEUE);
    if (memory->shared_size < VDEC_MIN_MEMORY)
        return vdec_reject("computeQueue.shared_size", memory->shared_size, VDEC_ERR_MEMORY_SIZE);
    if (!memory->shared)
        return vdec_reject("computeQueue.shared", memory->shared, VDEC_ERR_MEMORY_PTR);
    *out = memory->shared;
    if (svclog()) fprintf(stderr, "[svc]   Videodec2 compute queue -> 0x%llx (%llu bytes)\n",
                          (unsigned long long)*out, (unsigned long long)memory->shared_size);
    return 0;
}
HLE(s_videodec2_release_compute_queue) { return a0 ? 0 : VDEC_ERR_QUEUE; }
HLE(s_videodec2_query_decoder_memory) {
    svc_log("sceVideodec2QueryDecoderMemoryInfo", a0, a1, a2, a3, a4, a5, 9);
    auto* config = (const VdecConfig*)PW(a0); auto* info = (VdecMemory*)PW(a1);
    if (!config || !info) return VDEC_ERR_ARG;
    if (info->size != sizeof(*info)) return VDEC_ERR_STRUCT;
    // Pure sizing query: no compute queue required (#1658).
    uint64_t valid = vdec_validate_config(config, /*require_queue=*/false); if (valid) return valid;

    // max_frame_size is the buffer the GUEST allocates and hands straight back as VdecFrame.data for
    // every decoded picture. That coupling is observed, not assumed: this file reported
    // VDEC_MIN_MEMORY (0x10000) and the title's next VdecFrame.data_size came back as exactly
    // 0x10000.
    //
    // VDEC_MIN_MEMORY was derived as a compute-queue identity page (see its definition), never as a
    // decoder frame estimate, and it does not vary with resolution — a 4K request got the same
    // 64 KiB. That is safe ONLY while vdec_no_picture never writes into frame->data. The moment a
    // real decoder lands (#1688), writing a 1920x1088 NV12 picture into a 64 KiB guest allocation is
    // a ~3 MiB guest-heap overflow, and it would present as a decoder bug rather than as the sizing
    // answer that caused it. Report a size the guest can actually decode into, now, while doing so
    // costs nothing and cannot be mistaken for a decoder defect later.
    //
    // NV12 is the format PS5 video decode delivers (see video_backend.hpp): a full-resolution luma
    // plane plus a half-resolution interleaved chroma plane, each chroma dimension rounded UP —
    // `w*h + 2*ceil(w/2)*ceil(h/2)` — then rounded up to frame_alignment. The exact expression and
    // why the `*3/2` shorthand is NOT used are below, at the computation. An unreasonable request
    // yields an honestly unreasonable size that the guest's own allocator rejects visibly, which is
    // the behaviour we want.
    //
    // Dimensions may legitimately be -1 ("auto"). prosper has no evidence for what a real library
    // reports then, and a level-implied maximum would be invention, so that path keeps the
    // documented floor unchanged and is called out rather than papered over.
    //
    // CONFIDENCE: MED on the NV12 derivation — the format is established and the guest's observed
    // 1920x1088 (macroblock-rounded) request fits it exactly, but whether Sony's library adds
    // per-plane padding or a decoder-private tail is unproven. LOW on the auto-dimension path.
    // The cpu/gpu/shared workspace sizes stay at the floor: nothing here is evidence for those, and
    // inflating them has broken a title before (see VDEC_MIN_MEMORY's own note on the 16 MiB
    // placeholder exhausting a CRI pool).
    // The EXACT NV12 size comes from prosper::video::nv12_bytes(), which is the SAME function the
    // decoder backend uses to check the caller's buffer and to fill it. That shared call is the point:
    // this query tells the guest how large a frame buffer to allocate, and the backend writes into
    // the buffer the guest allocated from that answer — two rules would be a guest buffer sized by
    // one and written by the other, i.e. a heap overflow presenting as a decoder bug. The rule itself,
    // why `w*h*3/2` is wrong for an odd dimension, and the widen-before-arithmetic trap are all
    // written out at the function. (Instrument-trap 34; consolidated on #2571 review D1, which
    // pointed out the property had no test that could see it — `test_videodec2_decode` now has one.)
    constexpr uint32_t VDEC_FRAME_ALIGN = 0x100;
    uint64_t frame_bytes = 0;
    if (config->max_width > 0 && config->max_height > 0) {
        // Use the dimensions the guest ASKED FOR, which are macroblock-rounded (this title requests
        // 1088 for a 1080-line movie). That is not a safety margin to be trimmed later: H.264 codes
        // in 16x16 macroblocks, so a 1080-row picture is physically written as 1088 rows and cropped
        // for display via the SPS frame_cropping fields. Sizing from 1080 would be short by 8 rows
        // times the pitch — a real overflow, not conservatism.
        frame_bytes = prosper::video::nv12_bytes((uint32_t)config->max_width,
                                                 (uint32_t)config->max_height);
    } else {
        // Auto dimensions: the value below is a FLOOR, not a derivation, and the original
        // resolution-independent hazard survives on this path. Nothing in the code distinguishes the
        // two cases at the call site, so say it once rather than let the next reader — or the real
        // decoder that #1688 adds — treat 64 KiB as a computed answer.
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            fprintf(stderr,
                    "[vdec] QueryDecoderMemoryInfo: max_width/max_height are auto (%d x %d), so "
                    "max_frame_size falls back to the 0x%llx FLOOR -- this is not a derived size, and "
                    "a real decoder must not write a picture into it unchecked (#1688)\n",
                    config->max_width, config->max_height, (unsigned long long)VDEC_MIN_MEMORY);
    }
    frame_bytes = (frame_bytes + VDEC_FRAME_ALIGN - 1) & ~(uint64_t)(VDEC_FRAME_ALIGN - 1);

    info->cpu_size = info->gpu_size = info->shared_size = VDEC_MIN_MEMORY;
    info->max_frame_size = frame_bytes > VDEC_MIN_MEMORY ? frame_bytes : VDEC_MIN_MEMORY;
    info->cpu = info->gpu = info->shared = 0;
    info->frame_alignment = VDEC_FRAME_ALIGN; info->reserved = 0;
    return 0;
}
HLE(s_videodec2_create_decoder) {
    svc_log("sceVideodec2CreateDecoder", a0, a1, a2, a3, a4, a5, 9);
    auto* config = (const VdecConfig*)PW(a0); auto* memory = (const VdecMemory*)PW(a1);
    auto* out = (uint64_t*)PW(a2);
    if (!config || !memory || !out) return VDEC_ERR_ARG;
    if (memory->size != sizeof(*memory)) return VDEC_ERR_STRUCT;
    // Building a decoder DOES need a queue to build it on, and a caller that reaches here without one
    // must fail visibly rather than be handed a decoder handle backed by no queue (#1658).
    uint64_t valid = vdec_validate_config(config, /*require_queue=*/true); if (valid) return valid;
    // The other half of the #2270 contract capture: what the title asked the decoder to BE.
    // codec/profile name the bitstream a real decoder must accept; max_width/height and
    // VdecMemory::max_frame_size together pin the output layout arithmetically -- max_frame_size
    // divided by max_width*max_height is bytes-per-pixel, and 1.5 vs 4 is YUV420 vs packed 32-bit.
    if (getenv("PROSPER_VDEC2_CONTRACT"))
        fprintf(stderr,
                "[vdec-contract] create codec=%u profile=%u max=%dx%d dpb=%d input_depth=%u | "
                "mem cpu=%llu gpu=%llu shared=%llu max_frame=%llu align=%u\n",
                config->codec, config->profile, config->max_width, config->max_height,
                config->max_dpb, config->input_depth,
                (unsigned long long)memory->cpu_size, (unsigned long long)memory->gpu_size,
                (unsigned long long)memory->shared_size,
                (unsigned long long)memory->max_frame_size, memory->frame_alignment);
    if (memory->cpu_size < VDEC_MIN_MEMORY || memory->gpu_size < VDEC_MIN_MEMORY ||
        memory->shared_size < VDEC_MIN_MEMORY || memory->max_frame_size < VDEC_MIN_MEMORY)
        return VDEC_ERR_MEMORY_SIZE;
    if (!memory->cpu || !memory->gpu || !memory->shared) return VDEC_ERR_MEMORY_PTR;
    const uint64_t handle = g_handle.fetch_add(1) + 0x10000;
    { std::lock_guard<std::mutex> lk(g_vdec_mx); g_vdec_codecs[handle] = config->codec; }
    *out = handle;
    if (svclog()) fprintf(stderr, "[svc]   Videodec2 decoder -> 0x%llx codec=%u\n",
                          (unsigned long long)handle, config->codec);
    return 0;
}
// --- libSceVdecsw decoder-config adapter (#3669) -----------------------------------------------
//
// libSceVdecsw's decoder config is libSceVideodec2's plus 8 bytes -- 0x50 against 0x48 -- and the two
// agree field-for-field over the part they share. That is read off a live struct rather than assumed
// (Uncharted PPSA05684, `eboot+0x3289c70` at the moment it calls QueryDecoderMemoryInfo):
//
//   +0x00 size=0x50   +0x08 resource=1   +0x0c codec=1        +0x10 profile=100 (High)
//   +0x14 level=52    +0x18 3840x2160    +0x20 max_dpb=4      +0x24 input_depth=3
//   +0x28 compute_queue=0x2000000000  <- exactly the handle our AllocateComputeQueue just returned
//   +0x30 affinity=0x3f   +0x38 priority=-1   +0x3c optimize=1   +0x48 (the extra qword) = 0
//
// Every field lands where VdecConfig puts it, and the compute_queue matching our own return value is
// the strongest single confirmation available: the guest read it back from us and wrote it here.
//
// TWO differences are handled rather than papered over:
//   * the trailing qword at +0x48 has no Videodec2 counterpart and nothing here reads it;
//   * the byte at +0x3f, which Videodec2 requires to be zero as `reserved1`, carries 1. It is a
//     Vdecsw field, not a violated reservation, so the shared validator must not reject it -- the
//     adapted copy zeroes the two reserved bytes and this comment is the record that it did.
//     CONFIDENCE: LOW on what that byte means; nothing in prosper consumes it.
struct VdecswConfig { VdecConfig base; uint64_t extra2; };
static_assert(sizeof(VdecswConfig) == 0x50, "libSceVdecsw decoder config is 0x50 bytes");

// Copy a guest Vdecsw config into a Videodec2-shaped one. Returns false when the guest's declared
// size is not Vdecsw's, so a wrong ABI guess fails visibly here instead of mis-parsing downstream.
static bool vdecsw_adapt_config(uint64_t guest_config, VdecConfig* out) {
    const auto* c = (const VdecswConfig*)PW(guest_config);
    if (!c || !out) return false;
    if (c->base.size != sizeof(VdecswConfig)) {
        vdec_reject("vdecsw.config.size", c->base.size, VDEC_ERR_STRUCT);
        return false;
    }
    *out = c->base;
    out->size = sizeof(VdecConfig);   // the shared validator checks its OWN struct's size
    out->reserved0 = 0; out->reserved1 = 0;
    return true;
}
HLE(s_vdecsw_query_decoder_memory) {
    VdecConfig cfg{};
    if (!vdecsw_adapt_config(a0, &cfg)) return VDEC_ERR_STRUCT;
    return s_videodec2_query_decoder_memory((uint64_t)(uintptr_t)&cfg, a1, a2, a3, a4, a5);
}
HLE(s_vdecsw_create_decoder) {
    VdecConfig cfg{};
    if (!vdecsw_adapt_config(a0, &cfg)) return VDEC_ERR_STRUCT;
    return s_videodec2_create_decoder((uint64_t)(uintptr_t)&cfg, a1, a2, a3, a4, a5);
}
HLE(s_videodec2_delete_decoder) {
    // Tear the access-unit decoder down with the guest's decoder (#2270). Without this every
    // deleted decoder strands an AVCodecContext, two AVFrames, an AVPacket and the NV12 staging
    // vector for the life of the process.
    //
    // A LEAK, not a correctness bug, and the distinction is worth keeping straight: a stale
    // g_vdec_au_ids entry could only decode against another movie's reference frames if a handle
    // were ever REUSED, and it is not -- g_handle is fetch_add-only and never reset (:66, :529),
    // so the recycled-handle path is unreachable.
    //
    // close_decoder runs OUTSIDE g_vdec_mx: it calls into the video backend, and holding an HLE
    // lock across a backend call is how lock-order problems get built.
    int au_id = -1;
    {
        std::lock_guard<std::mutex> lk(g_vdec_mx);
        if (!g_vdec_codecs.erase(a0)) return VDEC_ERR_DECODER;
        auto it = g_vdec_au.find(a0);
        if (it != g_vdec_au.end()) { au_id = it->second.id; g_vdec_au.erase(it); }
    }
    if (au_id >= 0)
        if (auto* vb = prosper::video::backend()) vb->close_decoder(au_id);
    return 0;
}
HLE(s_videodec2_decode) {
    // This log is CAPPED at 8 access units, and it is the only Videodec2 entry point that caps.
    // Say so in the log itself: once a title actually feeds this path, "8 decodes in the log" reads
    // exactly like "the guest decoded 8 frames and stopped", which is a different and much more
    // alarming finding than "the cap was reached". (Instrument-trap 13 in
    // docs/GAME_COMPAT_ORCHESTRATION.md — a print cap misread as a frequency. Nearly banked as a
    // result on the very first boot that reached here, #1658.)
    if (svclog()) {
        static std::atomic<unsigned> n{0};
        unsigned seq = n.fetch_add(1);
        if (seq < 8) svc_log("sceVideodec2Decode", a0, a1, a2, a3, a4, a5, 7);
        else if (seq == 8)
            fprintf(stderr, "[vdec] sceVideodec2Decode: log capped at 8 access units; further "
                            "calls still execute but are NOT logged (this cap is not a count)\n");
    }
    uint32_t codec;
    { std::lock_guard<std::mutex> lk(g_vdec_mx); auto it = g_vdec_codecs.find(a0);
      if (it == g_vdec_codecs.end()) return VDEC_ERR_DECODER; codec = it->second; }
    auto* input = (const VdecInput*)PW(a1); auto* frame = (VdecFrame*)PW(a2);
    auto* out = (VdecOutput*)PW(a3);
    if (!input || !frame || !out) return VDEC_ERR_ARG;
    if (input->size != sizeof(*input) || frame->size != sizeof(*frame) ||
        out->size != sizeof(*out)) return VDEC_ERR_STRUCT;
    if (input->data_size && !input->data) return VDEC_ERR_ARG;
    if (!frame->data_size) return VDEC_ERR_FRAME_SIZE;
    if (!frame->data) return VDEC_ERR_FRAME_PTR;
    // PROSPER_VDEC2_CONTRACT: what a title actually asks this decoder for (#2270). prosper has no
    // Videodec2 decoder, and the two facts a fix needs -- the output pixel FORMAT enum and the
    // LAYOUT to write into frame->data -- are the two that cannot be read off a struct definition.
    //
    // There is a catch-22 in the middle of it: what the guest does with a DECODED picture cannot be
    // observed while every decode returns no-picture. What CAN be observed is what the guest asked
    // for and what it allocated, and the allocation constrains the layout arithmetically -- a
    // frame buffer of w*h*3/2 is a YUV420/NV12 request, w*h*4 is a packed 32-bit one. That is a
    // derivation from bytes the guest wrote, not an inference from what seems likely.
    if (getenv("PROSPER_VDEC2_CONTRACT")) {
        static std::atomic<unsigned> seq{0};
        const unsigned k = seq.fetch_add(1);
        if (k < 12) {
            const double px = 0.0;
            (void)px;
            // The first bytes of the access unit NAME the bitstream, which the codec field does not
            // when its value is not one we have seen before. H.264 carries a 1-byte NAL header after
            // the start code (SPS = 0x67); HEVC carries a 2-byte one (VPS = 0x40 0x01). Printing the
            // head is what separates "implement HEVC" from "we are reading the wrong field".
            char head[64] = {0};
            if (input->data && input->data_size) {
                const auto* au = (const uint8_t*)(uintptr_t)input->data;
                const uint64_t n = input->data_size < 16 ? input->data_size : 16;
                for (uint64_t i = 0; i < n; ++i)
                    snprintf(head + i * 3, 4, "%02x ", au[i]);
            }
            // The first 16 bytes name only the FIRST NAL, and "does this access unit contain an SPS"
            // is a different question that the head cannot answer -- an SPS/PPS pair sitting behind a
            // leading slice looks identical to a stream that has none. Walk the Annex-B start codes
            // and list every nal_unit_type, so "no parameter sets anywhere" is a measurement.
            char nals[128] = {0}; int np = 0;
            if (input->data && input->data_size > 4) {
                const auto* au = (const uint8_t*)(uintptr_t)input->data;
                for (uint64_t i = 0; i + 4 < input->data_size && np < (int)sizeof(nals) - 8; ++i) {
                    if (au[i] || au[i + 1] || au[i + 2] != 1) continue;   // 00 00 01 start code
                    np += snprintf(nals + np, sizeof(nals) - np, "%u ", au[i + 3] & 0x1fu);
                    i += 3;
                }
            }
            if (np) fprintf(stderr, "[vdec-contract]   au nal_unit_types: %s\n", nals);
            fprintf(stderr,
                    "[vdec-contract] decode#%u handle=0x%llx au_bytes=%llu frame_buf=%llu "
                    "frame_ptr=0x%llx head=%s\n",
                    k, (unsigned long long)a0, (unsigned long long)input->data_size,
                    (unsigned long long)frame->data_size,
                    (unsigned long long)frame->data, head);
        }
    }
    // Real decode through the backend's access-unit path (#2270).
    //
    // ON BY DEFAULT. This path landed opt-in behind PROSPER_VDEC2_DECODE while the output FORMAT
    // enum was unestablished, reasoning that a wrong constant hands the guest a correctly-decoded
    // picture it reads wrongly. That weighed one unknown field against the entire image and got the
    // trade backwards: with the path off, out->format is 0 AND out->pictures is 0, forever. The
    // guest receives the SAME unestablished format value it would receive with decoding on, plus no
    // image at all, and a title that waits for a frame waits for one that cannot arrive. Enabling
    // this cannot make `format` more wrong than leaving it off already does.
    //
    // PROSPER_VDEC2_NO_DECODE=1 restores the no-picture behaviour, so the A/B that justified this
    // stays reproducible.
    static const bool vdec2_no_decode = getenv("PROSPER_VDEC2_NO_DECODE") != nullptr;
    auto* vb = vdec2_no_decode ? nullptr : prosper::video::backend();
    if (vb) {
        int au_id;
        bool announce = false;
        {
            // open_decoder runs UNDER g_vdec_mx, unlike close_decoder below, and the asymmetry is
            // deliberate rather than an oversight. "Open at most once per open attempt, and never
            // once per access unit" is what the fail-visible path rests on -- one refusal, and not a
            // fresh AVCodecContext for every unit -- and holding the lock across the call is what
            // makes it one. Doing it outside would need a published "opening" state that a
            // concurrent decode on the same handle would read as a refusal. It is safe because the
            // backend's access-unit registry is a LEAF lock: nothing inside the backend re-enters
            // the HLE, so no path acquires the two in the opposite order.
            //
            // Note the invariant is per OPEN ATTEMPT, not per handle for all time: Reset deliberately
            // re-arms it. The BANNER is per handle -- see VdecAu::announced.
            std::lock_guard<std::mutex> lk(g_vdec_mx);
            VdecAu& st = g_vdec_au[a0];
            if (!st.opened) { st.opened = true; st.id = vb->open_decoder(codec); }
            au_id = st.id;
            if (au_id < 0 && !st.announced) { st.announced = true; announce = true; }
        }
        if (au_id < 0) {
            // FAIL-VISIBLE, and this is the whole point of #2270. Announce the refusal once per
            // guest decoder, with the codec AND the head of the access unit: the codec field alone
            // does not name a bitstream (2382845 is not an ordinal anyone can look up), while the
            // first bytes do -- 00 00 00 01 is Annex-B H.264/HEVC, and a VP9 key frame carries the
            // sync code 49 83 42. Whoever reads this log can tell "implement this codec" apart from
            // "we read the wrong field" without another run.
            if (announce) vdec_no_decoder_warning(codec, input, "the backend refused the codec");
        } else {
            vdec_dump_au((const uint8_t*)PW(input->data), input->data_size);
            // The decoder writes STRAIGHT INTO the guest's frame buffer, under its own lock. It used
            // to hand back pointers into its staging buffer for this function to copy from with no
            // lock held, which raced a concurrent Reset/DeleteDecoder freeing exactly that buffer
            // (#2571 review N5). Passing the destination down removes the lifetime question instead
            // of documenting it, and drops a 3.1 MB per-frame copy on the way.
            prosper::video::VideoBackend::AuPicture pic;
            auto* dst = (uint8_t*)PW(frame->data);
            const auto decoded = vb->decode_au(au_id, (const uint8_t*)PW(input->data),
                                               (size_t)input->data_size, dst, frame->data_size, pic);
            using AuResult = prosper::video::VideoBackend::AuResult;
            // A decoded picture that does not fit is the shape a WRONG LAYOUT takes, so it must not
            // be silent. `pic.nv12_bytes` is the exact NV12 derivation the whole output rests on; if
            // that derivation is wrong, what you observe is exactly this -- a buffer the guest sized
            // correctly and we think is too small. Reporting it as "no picture" without a word makes
            // it indistinguishable from a decoder that merely needs more input, which is the benign
            // case it looks like. Rate-limited, and it names both numbers so the ratio is checkable
            // on sight (#2270).
            if (decoded == AuResult::FrameTooSmall) {
                static std::atomic<unsigned> warned{0};
                if (warned.fetch_add(1) < 8)
                    fprintf(stderr,
                            "[vdec2] frame buffer too small: need=%llu (%ux%u NV12) but "
                            "frame->data_size=%llu -- reporting NO PICTURE. If this fires, "
                            "suspect the bytes-per-pixel derivation before the decoder (#2270)\n",
                            (unsigned long long)pic.nv12_bytes, pic.width, pic.height,
                            (unsigned long long)frame->data_size);
            }
            // THE GUARD. A zero-size picture must never become a reported one: `w*h*3/2` is 0 for a
            // 0x0 frame, so the old size check passed it, copied nothing, and answered valid=1 /
            // pictures=1 -- a confident, well-formed "here is your picture" over nothing, which is
            // precisely the false-success shape this whole change exists to remove, re-created one
            // layer down in its own new code. Only reachable through a backend bug; the backend
            // guards it too, and this one covers EVERY backend rather than the one that has it
            // today. (#2571 review N8.)
            if (decoded == AuResult::Decoded && (!pic.width || !pic.height || !pic.nv12_bytes)) {
                static std::atomic<unsigned> warned{0};
                if (warned.fetch_add(1) < 8)
                    fprintf(stderr,
                            "[vdec2] backend reported a DECODED %ux%u picture (%llu bytes) -- that "
                            "is not a picture, and reporting it as one would be #2270 again. "
                            "Answering NO PICTURE.\n",
                            pic.width, pic.height, (unsigned long long)pic.nv12_bytes);
            } else if (decoded == AuResult::Decoded) {
                // ONE line, once per run, naming the exact NV12 layout prosper wrote and where.
                // A guest reads that buffer through T# descriptors whose pitch, height and second-
                // plane offset it chose itself, and the only way to tell a layout disagreement from
                // a sampling defect is to have both numbers on one screen: this line supplies the
                // producer's half, the renderer's [avpchroma] line the consumer's.
                {
                    static std::atomic<bool> said{false};
                    if (!said.exchange(true))
                        fprintf(stderr,
                                "[vdec2] first picture: %ux%u NV12 y_stride=%u uv_stride=%u "
                                "nv12_bytes=%llu -> guest buffer 0x%llx (%llu bytes); chroma plane "
                                "starts at +%llu\n",
                                pic.width, pic.height, pic.y_stride, pic.uv_stride,
                                (unsigned long long)pic.nv12_bytes,
                                (unsigned long long)frame->data,
                                (unsigned long long)frame->data_size,
                                (unsigned long long)((uint64_t)pic.y_stride * pic.height));
                }
                // Dump what the GUEST receives, not what the backend produced: the copy is part of
                // what a reference comparison has to cover.
                vdec_dump_picture(dst, (size_t)pic.nv12_bytes, pic.width, pic.height);
                { std::lock_guard<std::mutex> lk(g_vdec_mx); g_vdec_au[a0].no_picture_run = 0; }
                frame->accepted = 1;
                out->valid = 1; out->error = 0; out->pictures = 1; out->discarded = 0;
                out->codec = codec;
                out->width = pic.width; out->height = pic.height;
                out->pitch = pic.y_stride; out->pitch_bytes = pic.y_stride;
                out->frame = frame->data; out->frame_size = frame->data_size;
                // #2898: remember this picture's SPS/VUI metadata so GetPictureInfo can
                // answer from the real stream. The access-unit bytes are exactly what the
                // guest handed us this call -- no extra I/O, no guessing.
                {
                    prosper::h264::SpsPictureMeta m;
                    bool have = prosper::h264::parse_first_sps(
                        (const uint8_t*)PW(input->data), (size_t)input->data_size, &m);
                    std::lock_guard<std::mutex> lk(g_vdec_mx);
                    VdecPicMeta& e = g_vdec_picmeta[frame->data];
                    if (!e.record) {
                        e.record = calloc(1, kVdecInfoRecordSize);
                        if (e.record) {
                            // *(void**)record = &record[8]: the payload block. calloc
                            // zeroes it, so the C-string at payload+0x08 is empty and
                            // the flag byte at payload+0x30 is 0 (guest default path).
                            *(void**)e.record = (uint8_t*)e.record + 8;
                        }
                    }
                    e.has_meta = have;
                    if (have) e.meta = m;
                    static std::atomic<bool> said{false};
                    if (!said.exchange(true))
                        fprintf(stderr,
                                "[vdec2] picture meta captured for GetPictureInfo (#2898): "
                                "record=%p sps=%s crop=%u[%u,%u,%u,%u] ar=%u(idc=%u sar=%ux%u) "
                                "timing=%u(%u/%u)\n",
                                e.record,
                                have ? "parsed" : "absent-in-AU", m.crop_flag ? 1u : 0u,
                                m.crop[0], m.crop[1], m.crop[2], m.crop[3],
                                m.ar_flag ? 1u : 0u, m.ar_idc, m.sar_w, m.sar_h,
                                m.timing_flag ? 1u : 0u, m.num_units_in_tick, m.time_scale);
                }
                // The one field still unestablished (#2270). Left 0 rather than filled with a
                // plausible constant, because a wrong value here is a correctly-decoded picture
                // the guest misreads, which is silent. PROSPER_VDEC2_FORMAT exists so the first
                // title observed misreading a picture can be swept over candidate values in one
                // sitting instead of one rebuild per candidate; it is a diagnostic, and a value
                // discovered through it belongs in code with the title evidence that found it.
                static const uint32_t format_override = [] {
                    const char* v = getenv("PROSPER_VDEC2_FORMAT");
                    return v ? (uint32_t)strtoul(v, nullptr, 0) : 0u;
                }();
                out->format = format_override;
                return 0;
            }
            // The decoder opened and is consuming units without ever producing a picture. A few of
            // those are correct (a decoder builds reference state before its first frame); an
            // unbounded run of them is #2270 wearing the benign costume, so say so once, loudly.
            unsigned run;
            { std::lock_guard<std::mutex> lk(g_vdec_mx); run = ++g_vdec_au[a0].no_picture_run; }
            if (run == kVdecNoPictureAlarm)
                fprintf(stderr,
                        "[vdec2] sceVideodec2Decode: %u consecutive access units with NO PICTURE "
                        "from an open codec=%u decoder. Each one reported SCE_OK, which a title "
                        "cannot tell apart from \"no frame ready yet\" -- if it is waiting for a "
                        "frame it will wait forever (#2270)\n",
                        run, codec);
        }
    } else if (!vdec2_no_decode) {
        // No backend at all: headless tools and any build without a host video decoder. Same
        // announcement, same reason -- silence here is the defect.
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            vdec_no_decoder_warning(codec, input, "no host video backend is registered");
    }
    frame->accepted = 0; vdec_no_picture(frame, out, codec);
    return 0;
}
// --- libSceVdecsw decode loop (#3669) ----------------------------------------------------------
//
// Vdecsw splits what sceVideodec2Decode does in ONE synchronous call into four asynchronous ones:
// the title stages an access unit and a frame buffer, then polls for each to complete. The structs
// are Videodec2's, measured live rather than assumed (PPSA05684, decoder handle 0x10001):
//
//   SetDecodeOutput's frame struct    size = 0x18  {size, data = 0x1f00200000, data_size = 12 MiB}
//   TrySyncDecodeOutput's out struct  size = 0x38  == sizeof(VdecOutput) exactly
//
// 12 MiB against the config's 3840x2160 is 4K NV12 (3840*2160*3/2 = 11.87 MiB rounded up), which
// agrees with the format the decoder backend produces -- so the staged pieces can be handed to the
// SAME decode path Videodec2 uses instead of a second implementation of it.
//
// What is NOT yet measured is logged rather than guessed: the declared size of SetDecodeInput's
// struct, and whatever TrySyncDecodeInput writes through its second argument. Both are printed under
// PROSPER_SVCLOG so the next run reads them off a real call.
struct VdecswFrame { uint64_t size, data, data_size; };
static_assert(sizeof(VdecswFrame) == 0x18, "libSceVdecsw frame struct is 0x18 bytes");

// The input side is a QUEUE, not a slot, and that is a correction rather than a refinement.
//
// Vdecsw is asynchronous: nothing stops the title staging several access units before it polls for a
// picture, and a single slot silently keeps only the last of them. Measured with a slot: the first
// access unit prosper ever saw already carried nal_unit_type 1 exclusively -- no SPS, no PPS, no IDR
// in any unit, for the whole movie -- which reads as "this stream has no parameter sets" and is a
// conclusion about our own dropped writes. A queue keeps every staged unit in order, so what the
// decoder sees is what the title sent.
// The staged access unit is COPIED, not pointed at, and that is the difference between an
// asynchronous API and a synchronous one.
//
// sceVideodec2Decode consumes its bitstream inside the call, so holding the guest's pointer is safe
// there. Vdecsw's SetDecodeInput returns immediately and the picture arrives at a later poll, which
// means the guest is free to refill its staging buffer in between -- and it does. Measured: the
// access unit prosper logged at SetDecodeInput time carried `7 8 6 5 5 ...` (SPS, PPS, SEI, IDR
// slices), and the SAME pointer read microseconds later inside the decoder backend carried nothing
// but type 1. Every parameter set in the movie was overwritten before the decoder saw it, which
// libavcodec then reported, correctly, as "non-existing PPS 0 referenced".
struct VdecswInput { std::vector<uint8_t> au; uint64_t pts = 0; };
struct VdecswPending {
    std::deque<VdecswInput> inputs;               // staged access units, in order, owned
    uint64_t out_data = 0, out_bytes = 0;
    bool out_set = false;
};
std::mutex g_vdecsw_mx;
std::unordered_map<uint64_t, VdecswPending> g_vdecsw;

// Report an unknown struct ONCE per call site with its declared size, so an ABI mismatch is a named
// finding in the log rather than a silent mis-parse. Rate-limited like vdec_reject for the same
// reason: these sit in a per-frame loop.
static void vdecsw_note_struct(const char* what, uint64_t size, uint64_t ptr) {
    static std::mutex mx;
    static std::unordered_map<const char*, int> shown;
    { std::lock_guard<std::mutex> lk(mx); if (shown[what]++ >= 2) return; }
    fprintf(stderr, "[vdecsw] %s: declared size = 0x%llx at 0x%llx\n", what,
            (unsigned long long)size, (unsigned long long)ptr);
}

HLE(s_vdecsw_set_decode_input) {
    const auto* in = (const uint64_t*)PW(a1);
    if (!in) return VDEC_ERR_ARG;
    vdecsw_note_struct("SetDecodeInput", in[0], a1);
    // The whole struct, twice, under the contract lever: the access units this title stages carry
    // nal_unit_type 1 ONLY -- no SPS, no PPS, no IDR anywhere -- so H.264 parameter sets must reach
    // the decoder by some other field, and these six qwords are where to look first.
    if (getenv("PROSPER_VDEC2_CONTRACT")) {
        static std::atomic<unsigned> seq{0};
        if (seq.fetch_add(1) < 2)
            fprintf(stderr, "[vdecsw] SetDecodeInput struct: %016llx %016llx %016llx %016llx "
                            "%016llx %016llx\n",
                    (unsigned long long)in[0], (unsigned long long)in[1], (unsigned long long)in[2],
                    (unsigned long long)in[3], (unsigned long long)in[4], (unsigned long long)in[5]);
    }
    // MEASURED: the title declares 0x30, which is sizeof(VdecInput) exactly -- Vdecsw's input struct
    // IS Videodec2's. Accept only that, so a different title with a different shape is a named
    // rejection rather than a silent mis-read of somebody else's fields.
    if (in[0] != sizeof(VdecInput)) return vdec_reject("vdecsw.input.size", in[0], VDEC_ERR_STRUCT);
    if (in[2] && !in[1]) return VDEC_ERR_ARG;
    std::lock_guard<std::mutex> lk(g_vdecsw_mx);
    auto& p = g_vdecsw[a0];
    // Bounded: a title that stages without ever polling must not grow this without limit. The cap is
    // generous against any plausible decode depth (the config asks for input_depth=3) and dropping
    // the OLDEST keeps the stream's head, which is where the parameter sets live.
    constexpr size_t kMaxStaged = 64;
    if (p.inputs.size() >= kMaxStaged) p.inputs.pop_front();
    VdecswInput staged;
    if (in[1] && in[2]) {
        const auto* src = (const uint8_t*)PW(in[1]);
        staged.au.assign(src, src + in[2]);   // copy NOW: see the note on VdecswPending
    }
    staged.pts = in[3];
    p.inputs.push_back(std::move(staged));
    return 0;
}
HLE(s_vdecsw_set_decode_output) {
    const auto* f = (const VdecswFrame*)PW(a1);
    if (!f) return VDEC_ERR_ARG;
    if (f->size != sizeof(*f)) return vdec_reject("vdecsw.frame.size", f->size, VDEC_ERR_STRUCT);
    if (!f->data) return VDEC_ERR_FRAME_PTR;
    if (!f->data_size) return VDEC_ERR_FRAME_SIZE;
    std::lock_guard<std::mutex> lk(g_vdecsw_mx);
    auto& p = g_vdecsw[a0];
    p.out_data = f->data; p.out_bytes = f->data_size; p.out_set = true;
    return 0;
}
HLE(s_vdecsw_try_sync_decode_input) {
    // The title polls here for the staged access unit to have been consumed. prosper's decode is
    // synchronous inside TrySyncDecodeOutput, so by the time the guest can observe anything the
    // input either was staged or was not.
    //
    // The caller struct is 0x18: a size field it sets itself, then two qwords it does NOT initialise
    // (measured: +0x08 reads 0 and +0x10 reads stack residue -- 0x0000000400000000 on one call,
    // 0x00007fa700000000 on the next). Answering success while leaving those untouched hands the
    // title its own stack back as an answer, which is the #2951 failure exactly, and it is how a
    // movie-player queue ends up holding an entry whose pictureCount is garbage.
    //
    // They are ZEROED rather than filled with a guess. Zero is not an invented value here: it is
    // "no information", the same resolution s_gamepresets settles on for the same reason, and it
    // makes the answer a property of prosper rather than of whatever the guest last pushed. If a
    // title turns out to need a real count there, it will be visible as a title that stops making
    // progress rather than as one reading noise. CONFIDENCE: MED on zero being adequate,
    // HIGH on untouched being wrong.
    if (a1 && getenv("PROSPER_VDEC2_CONTRACT")) {
        const auto* q = (const uint64_t*)PW(a1);
        static std::atomic<unsigned> seq{0};
        if (seq.fetch_add(1) < 3)
            fprintf(stderr, "[vdecsw] TrySyncDecodeInput struct: %016llx %016llx %016llx\n",
                    (unsigned long long)q[0], (unsigned long long)q[1], (unsigned long long)q[2]);
    }
    {
        std::lock_guard<std::mutex> lk(g_vdecsw_mx);
        if (g_vdecsw.find(a0) == g_vdecsw.end()) return VDEC_ERR_DECODER;
    }
    if (a1) {
        auto* q = (uint64_t*)PW(a1);
        if (q[0] != 0x18) return vdec_reject("vdecsw.syncinput.size", q[0], VDEC_ERR_STRUCT);
        q[1] = 0; q[2] = 0;
    }
    return 0;
}
// libSceVdecsw's output struct is the same SIZE as Videodec2's (0x38, measured) but does not put
// the picture count in the same byte. The title reads it at **+0x0B**, which is where Videodec2's
// layout keeps `discarded`:
//
//     eboot+0x1a405b9: cmp BYTE PTR [r12+0x30],0    ; struct+0x08 -- `valid`, same as Videodec2
//     eboot+0x1a407a0: cmp BYTE PTR [r13+0x0b],1    ; struct+0x0B -- pictureCount
//     eboot+0x1a407a5: jne <ASSERTION: pNextDecodedFrame->m_outputInfo.pictureCount == 1>
//
// So `valid` agrees and the count does not, and a run that gets both wrong is indistinguishable
// from one that decodes nothing: the pictures were real, and the title threw every one of them away
// 63 times a run on an assertion about the byte next door.
//
// Only that one byte is re-placed. +0x09 and +0x0A have no measured meaning under Vdecsw and are
// left exactly as the shared decode wrote them rather than being zeroed on a guess; `discarded` is
// the field being overwritten and prosper's decode always sets it to 0, so nothing is lost.
// CONFIDENCE: HIGH on the offset (the guest's own compare), LOW on the rest of the layout.
static void vdecsw_place_picture_count(VdecOutput* out) {
    if (!out) return;
    ((uint8_t*)out)[0x0B] = out->pictures;
}
HLE(s_vdecsw_try_sync_decode_output) {
    VdecswPending p;
    {
        std::lock_guard<std::mutex> lk(g_vdecsw_mx);
        auto it = g_vdecsw.find(a0);
        if (it == g_vdecsw.end()) return VDEC_ERR_DECODER;
        p = it->second;
    }
    // A POLL IS NOT AN ACCESS UNIT. This is the one place the async shape differs from
    // sceVideodec2Decode in a way that matters: the title calls this repeatedly until a picture
    // appears, and feeding the staged unit to the decoder on every call re-sends the same bytes.
    // Measured before this guard existed: eight logged decodes were seven copies of one 4253-byte
    // unit, and the decoder's own send_failures counter climbed once per poll rather than once per
    // frame -- which reads exactly like a stream the decoder cannot parse, and is not.
    //
    // So the staged input is CONSUMED here. A poll with nothing staged is answered honestly as "no
    // picture", which is a state the Videodec2 path already knows how to express.
    if (!p.out_set) return VDEC_ERR_FRAME_PTR;
    auto* out = (VdecOutput*)PW(a1);
    if (!out) return VDEC_ERR_ARG;
    if (out->size != sizeof(*out)) return vdec_reject("vdecsw.output.size", out->size, VDEC_ERR_STRUCT);
    if (p.inputs.empty()) {
        uint32_t codec = 0;
        { std::lock_guard<std::mutex> lk(g_vdec_mx); auto it = g_vdec_codecs.find(a0);
          if (it == g_vdec_codecs.end()) return VDEC_ERR_DECODER; codec = it->second; }
        VdecFrame fr{}; fr.size = sizeof(fr); fr.data = p.out_data; fr.data_size = p.out_bytes;
        vdec_no_picture(&fr, out, codec);
        vdecsw_place_picture_count(out);
        return 0;
    }
    VdecswInput staged;
    {
        std::lock_guard<std::mutex> lk(g_vdecsw_mx);
        auto& q = g_vdecsw[a0].inputs;
        if (q.empty()) return VDEC_ERR_ARG;   // raced another poller; nothing to do
        staged = std::move(q.front()); q.pop_front();
    }
    // Hand the staged pieces to the SAME decode the Videodec2 path uses -- struct checks, backend
    // call, NV12 write and VdecOutput fill all included -- rather than reimplementing any of it.
    // The guest's own output struct (0x38 == sizeof(VdecOutput)) is passed straight through.
    VdecInput in{}; in.size = sizeof(in);
    in.data = (uint64_t)(uintptr_t)staged.au.data(); in.data_size = staged.au.size();
    in.pts = staged.pts;
    VdecFrame fr{}; fr.size = sizeof(fr); fr.data = p.out_data; fr.data_size = p.out_bytes;
    const uint64_t rc = s_videodec2_decode(a0, (uint64_t)(uintptr_t)&in,
                                           (uint64_t)(uintptr_t)&fr, a1, 0, 0);
    vdecsw_place_picture_count(out);
    return rc;
}
HLE(s_videodec2_flush) {
    uint32_t codec;
    { std::lock_guard<std::mutex> lk(g_vdec_mx); auto it = g_vdec_codecs.find(a0);
      if (it == g_vdec_codecs.end()) return VDEC_ERR_DECODER; codec = it->second; }
    auto* frame = (VdecFrame*)PW(a1); auto* out = (VdecOutput*)PW(a2);
    if (!frame || !out) return VDEC_ERR_ARG;
    if (frame->size != sizeof(*frame) || out->size != sizeof(*out)) return VDEC_ERR_STRUCT;
    // "Nothing buffered" is a legitimate answer to a flush and terminates a guest's drain loop, so
    // this is NOT the #2270 false-success shape and must not be shouted about like one. It is still
    // incomplete: a decoder holding reordered pictures loses them here, which costs a movie its tail
    // frames rather than hanging a title. Said once, only when a real decoder is actually attached,
    // so the log reflects a consequence someone can observe (#2562).
    {
        bool has_decoder = false;
        { std::lock_guard<std::mutex> lk(g_vdec_mx);
          auto it = g_vdec_au.find(a0); has_decoder = it != g_vdec_au.end() && it->second.id >= 0; }
        static std::atomic<bool> said{false};
        if (has_decoder && !said.exchange(true))
            fprintf(stderr, "[vdec2] sceVideodec2Flush reports \"no pictures buffered\" without "
                            "draining the decoder: any reordered pictures still held are lost, so a "
                            "movie may end a few frames early (#2562)\n");
    }
    frame->accepted = 0; vdec_no_picture(frame, out, codec);
    return 0;
}
HLE(s_videodec2_reset) {
    // sceVideodec2Reset DISCARDS the decoder's buffered state and leaves the decoder usable. It is
    // neither of its two neighbours, and #2585 was implementing it as the third:
    //
    //   Flush  -- DRAIN.   Carries a VdecFrame/VdecOutput, so it hands buffered pictures back.
    //   Reset  -- DISCARD. Carries neither, so it returns nothing; it throws the state away.
    //   Delete -- DESTROY. The decoder is gone and the handle with it.
    //
    // THE WARRANT IS THE DOMINANCE ARGUMENT BELOW, not the confidence label. A weaker supporting
    // claim -- that Reset must leave the decoder usable, derivable from the export list because
    // libSceVideodec2 has NO re-initialise entry point between Create and Delete -- is CONFIDENCE:
    // HIGH but is NOT load-bearing: the pre-fix code already satisfied it in the guest-visible sense,
    // since the handle stayed valid and the next Decode reopened. Cite the dominance argument.
    //
    // THIS USED TO CLOSE THE BACKEND DECODER and let the next Decode open a fresh one. That forgets
    // the parsed SPS/PPS along with the DPB, and the two are not equally re-suppliable: Videodec2's
    // caller demuxes itself, so whether parameter sets are repeated in-band is title-dependent, and
    // a title that sends them once and resets mid-stream got a decoder that could not decode until
    // the next in-band SPS -- #2270's own hang shape in a new place. Measured on this build's
    // libavcodec with the committed Annex-B asset: a FLUSHED context fed access units with every
    // SPS/PPS stripped decodes 6 pictures, a FRESH one decodes 0. See VaapiBackend::reset_decoder.
    //
    // Which disposition the Sony contract actually specifies for the sequence headers is NOT
    // established (CONFIDENCE: MED) -- and it does not have to be, because flushing DOMINATES. If
    // the contract keeps them, close-and-reopen is broken and flushing is right. If the contract
    // drops them, flushing is NO WORSE: a stream that repeats its parameter sets replaces the
    // retained ones in-band by id, and a stream that does not repeat them is one where retaining is
    // the only thing that decodes at all. (Stated as "no worse" rather than "correct" on purpose --
    // under that reading, retention can make prosper decode where a faithful implementation would
    // not. It does not change the decision.) There is no reading under which closing wins.
    //
    // LOGGED, and that is not incidental. #2585's own text argued the defect was bounded because
    // "no title in this repository's history is recorded calling sceVideodec2Reset at all" -- but
    // this handler had no svc_log call, so no boot could ever have recorded one. That was a fact
    // about the instrument being used as a fact about the titles. `dump_words = 0`: the only
    // argument is a decoder handle, and a handle is >= 0x10000, which svc_ptrish would otherwise
    // mistake for a pointer and try to dump.
    svc_log("sceVideodec2Reset", a0, a1, a2, a3, a4, a5, 0);
    int au_id = -1;
    {
        std::lock_guard<std::mutex> lk(g_vdec_mx);
        if (!g_vdec_codecs.count(a0)) return VDEC_ERR_DECODER;
        auto it = g_vdec_au.find(a0);
        if (it != g_vdec_au.end()) {
            au_id = it->second.id;
            it->second.no_picture_run = 0;
            // A handle with NO live backend decoder re-arms the open so the next Decode retries.
            // That is the refused-codec path, and `announced` is deliberately kept so the refusal
            // does not re-print its banner on every reset cycle (#2571 N3). A LIVE decoder must NOT
            // re-arm: it is about to be flushed in place and never closes, so re-arming would open a
            // second decoder and strand the first.
            if (au_id < 0) it->second.opened = false;
        }
    }
    if (au_id < 0) return 0;
    // Outside the lock: these call into the backend, and holding an HLE lock across a backend call
    // is how lock-order problems get built (same rule as DeleteDecoder above).
    auto* vb = prosper::video::backend();
    if (vb && vb->reset_decoder(au_id)) return 0;   // flushed in place; the decoder stays open
    // FALLBACK -- a backend with no in-place reset (or none registered any more). Close and re-arm,
    // which is what this function used to do unconditionally. It forgets too much rather than too
    // little, which is the right direction to fail: leaving the DPB live would decode the guest's
    // next access units against references it just asked us to forget, and that is a corrupt picture
    // rather than an error. Said once, because a fail-visible backstop nobody can see is not one.
    if (vb) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            fprintf(stderr,
                    "[vdec2] sceVideodec2Reset: this backend has no in-place reset, so the decoder "
                    "is being CLOSED and reopened. That discards the parsed sequence headers as well "
                    "as the DPB, so a title whose stream carries its parameter sets only once cannot "
                    "decode again until its next in-band SPS (#2585)\n");
    }
    {
        std::lock_guard<std::mutex> lk(g_vdec_mx);
        auto it = g_vdec_au.find(a0);
        if (it != g_vdec_au.end() && it->second.id == au_id) {
            it->second.id = -1;
            it->second.opened = false;
        }
    }
    if (vb) vb->close_decoder(au_id);
    return 0;
}
// sceVideodec2GetPictureInfo / ...GetAvcPictureInfo. `which` names the entry point so a size mismatch
// identifies itself (#1658).
//
// #2898, now implemented for the generic form. Guest evidence (PPSA06367 x2, PPSA29343 x3 --
// all five sites disassembled):
//   * the caller pre-zeroes its picture-info block and sets its FIRST field to the block
//     size -- 0x78 (Gollum both sites), 0xb8 and 0x58 (Beast) -- the same self-sizing
//     convention every other struct in this library uses. The size is the variant
//     discriminator, so it is honoured rather than matched against one constant.
//   * on return the guest reads +0x20 (a pointer), +0x35/+0x38..+0x44 (crop flag +
//     quad), +0x48/+0x49/+0x4a/+0x4c (aspect flag/idc/SAR pair), +0x55/+0x58/+0x5c
//     (timing flag/num_units_in_tick/time_scale). That set IS H.264 SPS/VUI, and every
//     value comes from the stream we actually decoded (see h264_sps.cpp).
//   * +0x20 points at a decoder-side record; Gollum reads an inline C-string at +0x08 of
//     it and a flag byte at +0x30, Beast only compares it for identity. Our record is a
//     zeroed block with an empty string: flag 0 takes the guest's own default path.
//
// The AVC form (kjrLbcyhEiw) keeps the old behaviour: no title has ever been observed
// calling it, so its layout remains unestablished, and inventing one would be fabricating
// an ABI. The mismatch path still reports the size the caller declared.
static uint64_t vdec_picture_info(const char* which, uint64_t a0, uint64_t a1, uint64_t a2,
                                  uint64_t a3) {
    // Report the FIRST call to each entry point unconditionally, with the later arguments.
    //
    // Relying on a size mismatch alone would have been a weak plan: both forms take the same
    // OutputInfo in a0, so if what distinguishes them is what they WRITE to a later out-param, a0
    // matches, the mismatch branch never fires, and a run set up to capture the AVC layout captures
    // nothing while appearing to work. Printing a1..a3 once costs two lines a boot and cannot miss.
    static std::mutex mx;
    static std::unordered_map<const char*, int> seen;   // `which` is a literal: stable key
    bool first = false;
    { std::lock_guard<std::mutex> lk(mx); first = (seen[which]++ == 0); }
    if (first)
        fprintf(stderr, "[vdec] %s FIRST CALL: a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx\n",
                which, (unsigned long long)a0, (unsigned long long)a1,
                (unsigned long long)a2, (unsigned long long)a3);
    auto* out = (const VdecOutput*)PW(a0);
    if (!out) return VDEC_ERR_ARG;
    if (out->size != sizeof(*out)) {
        static std::mutex bad_mx;
        static std::unordered_map<const char*, int> bad;
        bool report = false;
        { std::lock_guard<std::mutex> lk(bad_mx); report = (bad[which]++ < 4); }
        if (report)
            fprintf(stderr, "[vdec] %s: outputInfo size 0x%llx != VdecOutput 0x%zx -- rejected. If this "
                            "is the AVC form, that size IS its layout evidence (#1658).\n",
                    which, (unsigned long long)out->size, sizeof(*out));
        return VDEC_ERR_STRUCT;
    }

    // The AVC form stops here: no title has ever been observed calling it, its layout is
    // unestablished, and inventing one would be fabricating an ABI. It accepts the same
    // OutputInfo and stays permissive about everything else.
    if (strcmp(which, "sceVideodec2GetPictureInfo") != 0) return 0;

    // Generic form (#2898): fill the caller's picture-info from real SPS/VUI metadata.
    auto* pic = PW(a1);
    if (!pic) return VDEC_ERR_ARG;
    // Self-sizing convention: the caller's first field declares the block's length.
    // Observed variants: 0xb8 and 0x78 and 0x58 (PPSA29343's three sites). The fill is
    // tiered (see h264_sps.hpp), so anything from 0x28 up gets every group that fits;
    // below that the guest could not read even the +0x20 pointer back.
    const uint64_t declared = *(const uint64_t*)pic;
    if (declared < 0x28 || declared > 0x1000) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 4)
            fprintf(stderr, "[vdec] %s: pictureInfo size 0x%llx cannot hold the observed "
                            "field set (smallest tier 0x28; observed variants "
                            "0x58/0x78/0xb8, #2898) -- rejected\n",
                    which, (unsigned long long)declared);
        return VDEC_ERR_STRUCT;
    }

    prosper::h264::SpsPictureMeta meta;
    void* record = nullptr;
    bool have = false;
    // #2898 diagnosis: the guest FREES the block behind pictureInfo+0x20 through its own
    // allocator (measured: prosper's calloc'd record came back as "FMallocBinned3
    // Attempt to free an unrecognized block", same address printed). So whatever lives
    // there must be memory the GUEST allocated. The strongest candidate is the decoded
    // frame buffer itself -- out->frame -- which the guest owns and Beast's identity
    // arrays could key on. PROSPER_VDEC2_PICINFO_ECHO_FRAME=1 tests that reading.
    static const bool echo_frame = [] {
        const char* v = getenv("PROSPER_VDEC2_PICINFO_ECHO_FRAME");
        if (v && *v == '1') {
            fprintf(stderr, "[vdec] %s: PROSPER_VDEC2_PICINFO_ECHO_FRAME armed -- +0x20 "
                            "will echo outputInfo->frame. DIAGNOSTIC ONLY: measured on "
                            "PPSA06367 this faults at image+0x105ed54 (the guest derefs "
                            "*record as a pointer to the payload), which is the evidence "
                            "that falsified the frame-echo reading (#2967)\n",
                    "sceVideodec2GetPictureInfo");
            return true;
        }
        return false;
    }();
    {
        std::lock_guard<std::mutex> lk(g_vdec_mx);
        auto it = g_vdec_picmeta.find(out->frame);
        if (it != g_vdec_picmeta.end()) {
            have = it->second.has_meta;
            meta = it->second.meta;
            record = echo_frame ? (void*)(uintptr_t)out->frame : it->second.record;
        }
    }
    if (!record) {
        // Never decoded through this outputInfo: there is no picture to describe. An
        // error here is honest, and both observed callers cope -- Gollum ignores the
        // return but only calls after a successful Decode, Beast tests the return and
        // takes its error path (its three sites all do `test eax,eax`).
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 4)
            fprintf(stderr, "[vdec] %s: no decoded picture behind frame=0x%llx yet -- "
                            "returning VDEC_ERR_PIPE (#2898)\n",
                    which, (unsigned long long)out->frame);
        return VDEC_ERR_PIPE;
    }
    if (!have) {
        // A picture exists but its AU carried no parseable SPS (parameter sets can arrive
        // out-of-band or in a later unit). Still publish the record pointer -- identity
        // is what Beast needs and Gollum dereferences -- with all optional flags zero,
        // which is the truthful answer "this stream declares none of those".
        static std::atomic<int> said{0};
        if (said.fetch_add(1) < 2)
            fprintf(stderr, "[vdec] %s: no SPS parsed for frame=0x%llx; filling flags as absent "
                            "(#2898)\n", which, (unsigned long long)out->frame);
    }
    if (!prosper::h264::fill_picture_info(meta, record, pic, (size_t)declared))
        return VDEC_ERR_STRUCT;  // unreachable: the guard above admits only tiers the fill handles
    return 0;
}
HLE(s_videodec2_picture_info) {
    return vdec_picture_info("sceVideodec2GetPictureInfo", a0, a1, a2, a3);
}
HLE(s_videodec2_avc_picture_info) {
    return vdec_picture_info("sceVideodec2GetAvcPictureInfo", a0, a1, a2, a3);
}

// libScePsml: NOT IMPLEMENTED, and it must SAY so rather than answering SCE_OK.
//
// The dispatcher's default for an unregistered NID is `return 0`, which the guest reads as SCE_OK.
// For a query with OUT-PARAMETERS that is not a benign no-op, it is a success claim about data
// prosper never wrote -- so the guest goes on to read whatever the out-parameter's stack slot
// happened to hold. `CLAUDE.md` records the same shape on Metaphor: ReFantazio, where an
// unregistered `sceFontRenderCharGlyphImage` answered SCE_OK, wrote neither out-parameter, and the
// title divided by the resulting zero (#2951).
//
// The First Berserker: Khazan (PPSA20447) is the second title to pay for it, and it multiplies
// instead of dividing. Its `eboot+0xe74b20` does:
//
//     e74b51  mov $0x137,%edi ; call <sceSysmoduleLoadModule>
//     e74b74  call <libScePsml::3WVD91e12ZQ>          ; -> r15d
//     e74b8e  lea -0x48(%rbp),%rdi                    ; an OUT struct, NOT initialised
//     e74b9e  call <libScePsml::+2KpvixvL6E>          ; (&out, buf) -> eax
//     e74ba3  or %r15d,%eax ; jne 0xe74dc2            ; either non-zero -> clean early return
//     e74bac  mov -0x38(%rbp),%rdi                    ; N := out[+0x10]  (uninitialised stack)
//     e74bb5  shl $0x4,%rdi                           ; size = N * 16
//     e74bb9  call <FMemory::Malloc>                  ; ~244 TiB, and the title asserts
//
// Two things make returning an error the right answer here rather than a guess, and both were
// checked rather than assumed -- this project has a recorded trap in the OPPOSITE direction, where
// an error sentinel returned from a VALUE-returning contract was read as data and produced a 2 GiB
// allocation:
//   * The return value is a STATUS, not data: the guest ORs the two results and branches on
//     non-zero. `CONFIDENCE: HIGH` -- disassembled.
//   * The non-zero branch at `eboot+0xe74dc2` is a stack-check plus epilogue plus `ret`. The guest's
//     own code treats this as "the feature is unavailable" and skips it. Nothing is faked and no
//     output is manufactured; prosper is answering truthfully that it does not implement this
//     library, through the only channel the guest reads.
//
// `CONFIDENCE: LOW` on the specific value. libScePsml is absent from the PS5 3.20 reference set
// (this dump requires system software 12.70), so its own error space is not derivable and neither
// are these NIDs' names or signatures. The guest tests only zero versus non-zero, so ENOSYS is a
// truthful placeholder for "not implemented" and not a claim about Sony's encoding. If the real
// error space is ever recovered, replace the value -- not the policy.
HLE(s_psml_unimplemented) { return prosper::hle::kSceKernelErrorENOSYS; }

// Registration for the decode libraries in this file, called by register_builtin_hle().
// One entry point per library is what keeps every handler `static` to the translation unit
// that implements it (#3735).
void register_videodec_hle() {
    Hle::register_fn("RnDibcGCPKw", (HleFn)s_videodec2_query_compute_memory,
                     "sceVideodec2QueryComputeMemoryInfo");
    Hle::register_fn("eD+X2SmxUt4", (HleFn)s_videodec2_allocate_compute_queue,
                     "sceVideodec2AllocateComputeQueue");
    Hle::register_fn("UvtA3FAiF4Y", (HleFn)s_videodec2_release_compute_queue,
                     "sceVideodec2ReleaseComputeQueue");
    Hle::register_fn("qqMCwlULR+E", (HleFn)s_videodec2_query_decoder_memory,
                     "sceVideodec2QueryDecoderMemoryInfo");
    Hle::register_fn("CNNRoRYd8XI", (HleFn)s_videodec2_create_decoder, "sceVideodec2CreateDecoder");
    Hle::register_fn("jwImxXRGSKA", (HleFn)s_videodec2_delete_decoder, "sceVideodec2DeleteDecoder");
    Hle::register_fn("852F5+q6+iM", (HleFn)s_videodec2_decode, "sceVideodec2Decode");
    Hle::register_fn("l1hXwscLuCY", (HleFn)s_videodec2_flush, "sceVideodec2Flush");
    Hle::register_fn("wJXikG6QFN8", (HleFn)s_videodec2_reset, "sceVideodec2Reset");
    // #1658: kjrLbcyhEiw is sceVideodec2GetAvcPictureInfo, not a second NID for the generic form —
    // the 3.20 firmware database names them separately. They still share a handler because no title
    // evidence yet shows the AVC output layout differs; the handler says so and captures the first
    // mismatch rather than assuming they are identical.
    Hle::register_fn("NtXRa3dRzU0", (HleFn)s_videodec2_picture_info, "sceVideodec2GetPictureInfo");
    Hle::register_fn("kjrLbcyhEiw", (HleFn)s_videodec2_avc_picture_info,
                     "sceVideodec2GetAvcPictureInfo");
    // --- libSceVdecsw: the SOFTWARE-decode sibling of libSceVideodec2 (#3669) -------------------
    //
    // Uncharted (PPSA05684) decodes its movies through libSceVdecsw rather than libSceVideodec2, and
    // the two libraries share a set-up ABI. That is measured, not assumed: the title builds its
    // compute-memory struct with `size = 0x18` and its compute-queue config with `size = 0x10`
    // (eboot+0x1a3f75d and +0x1a3f74a), which are exactly sizeof(VdecComputeMemory) and
    // sizeof(VdecComputeConfig) -- and prosper's handlers REJECT a mismatched size, so a wrong guess
    // here fails visibly instead of silently mis-parsing. The argument positions were captured live
    // (PROSPER_HWBP_ARGS on each stub) and match one-for-one:
    //
    //   QueryComputeMemoryInfo(info*)                 <- rdi = &{size=0x18,...}
    //   AllocateComputeQueue(config*, memory*, out*)  <- rdi = &{size=0x10,...}, rsi = memInfo,
    //                                                    rdx = &[rbx+0x168] (the queue handle)
    //   QueryDecoderMemoryInfo(config*, info*)        <- rdi, rsi: two eboot .data structs
    //   CreateDecoder(config*, memory*, out*)         <- the same two, plus the handle output
    //
    // Only the four SET-UP calls are registered here. The decode loop (SetDecodeInput /
    // SetDecodeOutput / TrySyncDecodeInput / TrySyncDecodeOutput) is libSceVdecsw's own ASYNCHRONOUS
    // shape and has no Videodec2 equivalent -- sceVideodec2Decode does all four in one synchronous
    // call -- so mapping it onto these handlers would be an invention, and its live argument layout
    // cannot be read until CreateDecoder hands back a decoder (every one of those four is currently
    // entered with a null handle, which is the bug, not the ABI). Those stay unregistered, and the
    // title keeps failing VISIBLY at TryFetchDecodedFrame rather than being told a decode succeeded.
    // CONFIDENCE: HIGH on the four registered here (size-checked structs plus live argument capture).
    Hle::register_fn("0moTubWCsTM", (HleFn)s_videodec2_query_compute_memory,
                     "sceVdecswQueryComputeMemoryInfo");
    Hle::register_fn("hIgrg5h4V6s", (HleFn)s_videodec2_allocate_compute_queue,
                     "sceVdecswAllocateComputeQueue");
    Hle::register_fn("A+2M7EivuOU", (HleFn)s_vdecsw_query_decoder_memory,
                     "sceVdecswQueryDecoderMemoryInfo");
    Hle::register_fn("+L5ArV1tPGA", (HleFn)s_vdecsw_create_decoder, "sceVdecswCreateDecoder");
    Hle::register_fn("aqMiF0AgUYI", (HleFn)s_vdecsw_set_decode_input, "sceVdecswSetDecodeInput");
    Hle::register_fn("rgtMCOpyBSc", (HleFn)s_vdecsw_set_decode_output, "sceVdecswSetDecodeOutput");
    Hle::register_fn("l4sQYy5wPkc", (HleFn)s_vdecsw_try_sync_decode_input,
                     "sceVdecswTrySyncDecodeInput");
    Hle::register_fn("kMBw37oH8nI", (HleFn)s_vdecsw_try_sync_decode_output,
                     "sceVdecswTrySyncDecodeOutput");
    // libScePsml (PPSA20447): unregistered, these answered SCE_OK and left the caller's out-struct
    // untouched -- see s_psml_unimplemented. Names unknown: libScePsml is not in the 3.20 reference
    // set, so the raw NIDs are the only identity available and the label says exactly that.
    Hle::register_fn("3WVD91e12ZQ", (HleFn)s_psml_unimplemented, "libScePsml::3WVD91e12ZQ");
    Hle::register_fn("+2KpvixvL6E", (HleFn)s_psml_unimplemented, "libScePsml::+2KpvixvL6E");
}
// The local network libraries allocate opaque contexts even on a disconnected console; connection
// state is reported separately through NetCtl/NP. Returning generic success (0) from these ID-returning
// constructors instead creates an invalid context and makes their owner's initialization fail. The
// signatures and positive-return contract agree with the PS5 3.20 symbol table and the independently
// implemented SDK surface; no online identity or connectivity is fabricated here.
namespace {
}

// --- libSceIme keyboard API (#186) ---
// PPSA02664 polls this every frame in its input loop. We have no physical PS5 keyboard, so we report a
// consistent "no keyboard connected" state: Open/Update succeed, GetResourceId returns an empty
// resource array (no keyboards), GetInfo reports a disconnected device — the game then uses controller
// / on-screen input instead of waiting on a keyboard. Struct layouts + field offsets verified against
// shadPS4 src/core/libraries/ime/ime_common.h; error codes from ime_error.h. Note sceImeKeyboardOpen
// returns an Error (0 = SCE_OK), NOT a handle — so success is 0. CONFIDENCE: HIGH on the layouts; MED
// on the device/status enum "disconnected" == 0.
//   OrbisImeKeyboardResourceIdArray: user_id@0(s32) resource_id[5]@4(u32)                     size 24
//   OrbisImeKeyboardInfo: user_id@0 device@4 type@8 repeat_delay@12 repeat_rate@16 status@20 rsv[12]@24  size 36
// --- IME keyboard event injection (issue #1093) -------------------------------------------------
// PPSA02664 (Alex Kidd) reads its "press any button" / menu input ONLY through the IME keyboard
// path: it opens a keyboard and calls sceImeUpdate(handler) every frame, and never touches
// libScePad or sceUserService. sceImeUpdate must invoke handler(arg, &event) for each queued
// keyboard event; the old stub returned without calling, delivered no input, and the title never
// advanced (injected pad input was inert — the game wasn't reading the pad).
//
// Event ABI derived directly from the guest handler at eboot+0xf2c540 (PPSA02664 disassembly):
//   handler(rdi=arg, rsi=SceImeEvent*)
//   SceImeEvent:  +0x00 u32 id   (0x101 = KEY_DOWN, 0x102 = KEY_UP; 0x103..0x106 = other kbd events)
//                 +0x08 u16 keycode  (USB HID usage id; indexes the guest's keycode->bit table)
//                 +0x0a u16 (modifier/char)   +0x0c u32 (status)
//   KEY_DOWN sets the guest's current + newly-pressed keyboard bitmasks; KEY_UP clears them.
// HID keycodes verified against the guest's table: Enter=0x28, Space=0x2c, Z=0x1d, A=0x04.
// CONFIDENCE: HIGH on the id/keycode/offset layout (read from the guest handler + its HID table).
namespace {
} // namespace

// --- app content ---
namespace {
} // namespace

// sceSystemServiceGetStatus(SceSystemServiceStatus* status) — the out-struct is ARG 0 (single-arg
// call). This was aliased to s_appcontent_int, which returned success while writing 4 bytes through
// a1 — whatever stale value the caller left in RSI — and left the real status struct uninitialized.
// Layout cross-checked against Kyty LibSystemService.cpp:83: int32 eventNum @0; bool
// isSystemUiOverlaid @4, isInBackgroundExecution @5, isCpuMode7CpuNormal @6 (defaults TRUE),
// isGameLiveStreamingOnAir @7, isOutOfVrPlayArea @8; 12 bytes with tail padding.
namespace {
// ErrorDialog is system-owned UI. Platform glue uses SystemService status' background -> foreground
// transition to resume requests after it closes, independently of the CommonDialog FINISHED value.
// A real backend may stay active across many status samples; the headless backend dismisses during
// Open and must still expose one observable enter/return pair instead of collapsing both edges into
// a permanent foreground value. Oregon Trail #1606 positive-controlled this exact contract twice:
// the account-completion writer ran only after both SystemService samples. No title address or
// title identity participates in this state machine.
}

// Message-dialog LIFECYCLE (#144). Status enum: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3.
// The old handler returned FINISHED(3) UNCONDITIONALLY — including before any Open, where the real
// API reports NONE/INITIALIZED — so a guest polling GetStatus as a guard saw "dialog already done"
// at the wrong stage and skipped/duplicated its dialog logic. We track the real transitions:
// Initialize -> INITIALIZED, Open -> auto-dismiss to FINISHED (headless: no interactive UI, so the
// game's "wait until dismissed" loop still exits immediately), Close/Terminate -> back to NONE.
// GetResult -> zeroed struct = OK/no button pressed.
// A registered PlatformUi gets first refusal at Open; if it takes the dialog, status/result/close
// route there (real message box). Otherwise the core auto-dismisses headlessly. `g_msgdialog_backed`
// records which path Open chose. See platform_ui.hpp (#347).

// --- libSceSaveDataDialog ---------------------------------------------------------------
// Uses the common-dialog status enum: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3. A generic
// success stub is not safe for UpdateStatus: Dead Cells polled it 1,135 times in 16 seconds because
// zero means NONE, leaving its level transition in a permanent wait (#768). The headless policy
// matches MsgDialog above: Open auto-dismisses to FINISHED so the guest can consume a neutral result.
//
// PS4/PS5 SDK layout used below (also used by shadPS4): param.mode is u32 at +0x34 and param.userData
// is a pointer at +0x70. Result begins with mode/result/buttonId/pad followed by three pointers. The
// dirName/param pointer slots are caller-owned output destinations, not fields for the service to
// replace. Write only mode/result/buttonId/userData and preserve every padding, pointer, and reserved
// byte. A PlatformUi that accepts Open remains the intended owner, but every callback obtains a
// registry lease first: unregistering the backend safely abandons the dialog instead of dereferencing
// a stale non-owning pointer or rerouting the dialog to a replacement backend.
namespace {
}

// --- libSceImeDialog (on-screen text-entry dialog) (#191). We have no keyboard UI, so the dialog
// auto-completes: Init -> FINISHED immediately, so the game's "poll GetStatus until Finished" loop
// exits at once instead of hanging on a dialog that never appears; GetResult reports endStatus =
// OK/ENTER with the (unchanged/empty) input buffer; Term/Abort return to NONE.
// IMPORTANT: sceImeDialogGetStatus returns the IME-dialog's OWN enum (OrbisImeDialogStatus: NONE=0,
// RUNNING=1, FINISHED=2) — NOT the 4-value SceCommonDialogStatus that MsgDialog/ErrorDialog use
// (…RUNNING=2, FINISHED=3). Verified in shadPS4 ime_dialog.cpp. Returning 3 here made a game's poll
// loop never see Finished(2). We write only the 4-byte endStatus at GetResult offset 0 (the field the
// game branches on), never more, so a wrong tail-field guess can't corrupt the caller's struct.
// A registered PlatformUi (the app frontend) gets first refusal on the dialog: if it takes it
// (imeDialogOpen -> true), status/result/close route there so a real text field is shown; otherwise
// the core auto-completes headlessly (below). `g_imedialog_backed` records which path Init chose so a
// later poll/result/term goes to the same place. See platform_ui.hpp (#347).

// --- Transaction resources (#1905). sceSaveDataCreateTransactionResource returns the NEW RESOURCE'S
// ID, not 0-on-success. Returning 0 is the shape of the call that succeeds, so this looked correct
// and is not: Sonic Origins' (PPSA05325) save handler at eboot+0x93fdb0 is exactly
//     xor edi,edi; call sceSaveDataCreateTransactionResource; test eax,eax; jle <fail>
// — a zero return is read as failure, the handler records error 3, every later save operation
// returns that sticky error without running, and the title's boot coroutine polls the failed job
// forever. That single wrong return value is why PPSA05325 never leaves its first boot step.
// It really is a HANDLE and not merely "a positive number" — Sonic's full lifecycle proves it:
//   0x93fdc7  mov [r14+0xc0],eax        Create's result is retained in a member
//   0x9402cd  mov eax,[r14+0xc0]        ...loaded again to build the Mount3 descriptor,
//   0x9402d4  mov [rsp+0x68],eax        ...at descriptor base (rsp+0x40) + 0x28,
//   0x93f24e  mov edi,[rdi+0xc0]        ...and handed back as Delete's sole argument,
//   0x93f254  cmp edi,0xffffffff        guarded by the guest's own -1 "no resource" sentinel.
// That also independently reproduces the Mount3 +0x28 field this file documents from DQ7. The id is
// opaque to the guest, so any positive value satisfies it; we hand out a monotonic counter and keep
// the live set so Delete can reject one that was never created.
// CONFIDENCE: HIGH on the polarity, on "returns the resource id", and on Delete's argument being a
// scalar int32 in edi (the mov above precedes all four of Sonic's Delete sites and Oregon Trail's).
// Delete's error CODE for an unknown id is the residual unknown: PARAMETER is the facility's generic
// bad-argument code, and no title is observed passing an id Create did not return.
namespace {
}   // namespace

// --- libSceSaveData "save-data memory" API (#191). A per-(user,slot) fixed-size memory block the
// managed SaveData layer reads/writes by offset and syncs to storage; on PS5 this is how a title
// keeps a small always-resident save (settings/progress). We back each slot with a host-memory
// block: Setup allocates it, Set copies guest->block, Get copies block->guest, Sync commits. Struct
// layouts + field offsets verified against shadPS4 save_data/savedata.cpp; error codes from
// savedata_error.h. CONFIDENCE: HIGH on the happy-path round-trip (the contract a title depends on).
//   OrbisSaveDataMemoryData:  buf@0(ptr) bufSize@8(u64) offset@16(s64)               size 64
//   OrbisSaveDataMemorySetup2: option@0 userId@4 memorySize@8 iconMemorySize@16
//                              initParam@24 initIcon@32 slotId@40                    size 64
//   OrbisSaveDataMemorySet2:  userId@0 [pad@4] data@8(ptr) param@16 icon@24 dataNum@32 slotId@36
//   OrbisSaveDataMemoryGet2:  userId@0 [pad@4] data@8(ptr) param@16 icon@24 slotId@32
namespace {
}

// ===== Issue #306: honest OFFLINE console for the online/update/entitlement boot chain. =========
// DOLL's UE4 front-end runs a patch/entitlement check before the title screen; every subsystem in
// that chain answered success-with-garbage, so the check could neither succeed nor FAIL — the flow
// waited forever (docs/DOLL_LOADING_PROGRESSION.md). The blocks below give the chain the answers a
// real, network-disconnected, signed-out console gives.

// --- Guest-callback delivery discipline (shared by NetCtl + Np state callbacks). ----------------
// A registered callback is guest code: under PROSPER_GUEST_FS the HLE runs on the HOST %fs (the
// import swap-stub switched), so the guest callback must run with the GUEST %fs restored or its
// TLS accesses (UE MallocBinned caches!) read host TLS garbage. The swap-stub saves the guest fs
// base in its frame (push r11), so an asm entry shim (the f_apr_read_submit_entry pattern) hands
// the handler its entry %rsp. The guest swap path re-pushes args7/8/9, an alignment pad, then the
// saved r11: [rsp]=ret-to-stub, args at +8/+0x10/+0x18, pad at +0x20, guest fs at +0x28, and guest
// RA at +0x30. A [rsp] outside the stub region [0x6_0000_0000,0x7_0000_0000) means the
// host-context tail-jmp path (no swap happened) — call the callback on the current fs.
// Mechanism proven live by the PROSPER_NETCTL_CB experiment (run 7/9: delivered + consumed
// cleanly, no crash). CONFIDENCE: HIGH.
namespace {
#ifndef _WIN32
#endif
}

// --- libSceNetCtl: a network-DISCONNECTED console (default ON since #306). ----------------------
// DOLL registers a NetCtl state callback once at boot (sceNetCtlRegisterCallback) and then pumps
// sceNetCtlCheckCallback EXACTLY once per frame forever (14,191 calls in a 240 s run). On real
// hardware CheckCallback invokes the registered callback on the calling thread with the current
// state — an offline console still delivers an immediate DISCONNECTED. Register records {func,arg}
// and writes the callback id (Kyty Network.cpp NetCtlRegisterCallback); CheckCallback invokes the
// callback ONCE with SCE_NET_CTL_EVENT_TYPE_DISCONNECTED (PS4-inherited constant = 1; identical
// export names+NIDs on PS5 3.20 — CONFIDENCE MED on the PS5 value). Was the gated experiment
// PROSPER_NETCTL_CB=1; proven correct+consumed live (DOLL run 7/9), now default ON.
// PROSPER_NETCTL_CB=0 restores the old unimplemented behavior.
namespace {
#ifndef _WIN32
#endif
}

// --- libSceNpManager state callback: deliver SIGNED_OUT once (#306). ----------------------------
// DOLL registers its Np sign-in state callback via sceNpRegisterStateCallbackA and pumps
// sceNpCheckCallback. shadPS4 (offline mode) queues exactly one SIGNED_OUT event for the initial
// user and delivers it inside sceNpCheckCallback on the pumping thread; we mirror that. Callback-A
// prototype (shadPS4 np_manager.h): void cb(s32 userId, s32 state, void* userdata); state
// SIGNED_OUT = 1 (Unknown=0, SignedOut=1, SignedIn=2 — Kyty + shadPS4 agree). Register returns the
// positive callback id (shadPS4 RegisterStateCallbackA returns slot+1). CONFIDENCE: HIGH on the
// contract (two agreeing PS4 references, PS4-inherited surface; PS5 3.20 exports the same names).
namespace {
#ifndef _WIN32
#endif
}

// --- libSceErrorDialog: the real Initialize/Open/Close lifecycle (auto-dismiss, headless). ------
// Status enum shared with CommonDialog: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3 (shadPS4
// commondialog.h + error_dialog.cpp; PS4-inherited, identical export NIDs on PS5 3.20).
// GetStatus/UpdateStatus RETURN the status value directly (not an out-param). DOLL pumps
// sceErrorDialogUpdateStatus once per frame from boot (housekeeping); if the honest offline chain
// makes it Open the "could not download patch data" dialog, auto-dismiss to FINISHED so the flow's
// "wait until dismissed" loop exits — the MsgDialog precedent (#144). Open's param is
// { s32 size; s32 errorCode; s32 userId; s32 reserved } (shadPS4 error_dialog.cpp Param); the
// errorCode is printed unconditionally — a one-shot, high-value diagnostic of WHAT the game
// thinks failed. A PlatformUi that accepts Open is pinned as the non-owning expected backend; every
// later callback obtains a registry lease for that exact pointer. Unregister/replacement abandons
// safely to headless FINISHED and can never reroute to a backend that did not accept the request.
// CONFIDENCE: HIGH (shadPS4 implements this exact lifecycle).
namespace {
}

// ===== libSceRandom ============================================================================
//
// `sceRandomGetRandomNumber(void* buf, size_t size)` is the library's ONLY export (PS5 3.20
// `libSceRandom.c` lists exactly one `sprx_dlsym` line), and 27 of the 44 local dumps import it.
//
// Unregistered, it reached the dispatcher's `return 0` — and for this contract 0 is SCE_OK. The
// guest was told its buffer had been filled while nothing wrote to it, so it read back whatever
// its own stack or heap already held and used that as entropy. The worst property of that failure
// is that it is STABLE: a fresh stack allocation at the same call site tends to hold the same
// residue on every call, so a "random" session id or nonce can be identical all run. Nothing
// crashes and no diagnostic fires; only code that inspected the buffer could notice. (#2065, swept
// under #2081.)
//
// The fix is a real implementation rather than a fail-visible stub, because the honest answer here
// is cheap: the host has a CSPRNG. Returning an error instead would be strictly worse — this is a
// *value-producing* contract, and guests do gate on it.
//
// How the guests actually consume it, measured with `tools/re/nid_gate_scan.py --nid PI7jIZj4pcE`
// over the local dumps (call sites classified by what happens to eax):
//
//     PPSA08804  nonzero=2 ignored=1        PPSA04263  nonzero=2
//     PPSA24651  nonzero=1                  PPSA13579  nonzero=1
//     PPSA17942  ignored=1                  PPSA19244  forward=1 ignored=1
//
// So most call sites branch on "did this fail", and — the load-bearing part — NOT ONE compares the
// result against a specific constant (no `const` / `other-cmp` site anywhere). That is what makes
// the reject arms safe to add: the guest reads only the SIGN of the answer, never its value.
// CONFIDENCE: HIGH that success must fill the buffer and that failure must be non-zero;
// LOW on the exact error constant — the SCE_RANDOM error space is not in the 3.20 dump (which
// carries names and NIDs only) and no call site discriminates it, so the libkernel encoding is
// used per the project's default rather than an invented SCE_RANDOM_* value.
namespace {

// The published single-request cap. A guest asking for more than this gets an error on hardware, so
// prosper must not quietly serve it: a partial fill reported as success would recreate the exact
// bug this handler exists to remove, and an over-long fill would paper over a guest bug that real
// hardware rejects.
//
// CONFIDENCE: MED — the cap is from the API documentation, not from a live capture. This is also
// the ONLY arm of this handler that can newly FAIL a call the previous stub "succeeded", so it is
// the one place a wrong constant costs something rather than merely being imprecise.
//
// What is NOT established: the request sizes local titles actually pass. `nid_gate_scan` classifies
// what the guest does with `eax` and cannot recover an argument, so no instrument here has measured
// the `size` operand at any call site — deliberately stated rather than left as an implied "we
// checked". Settling it needs the argument registers read at the call, e.g. PROSPER_SVCLOG-style
// logging on this NID or a hardware breakpoint at the import. If a title is ever found requesting
// more, this constant is where to look first.

} // namespace

} // namespace prosper
