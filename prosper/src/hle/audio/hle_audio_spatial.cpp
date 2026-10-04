// hle_audio_spatial.cpp — libSceAudio3d / libSceAcm HLE.
//
// The spatial/batch corner of guest audio: the object-based 3D port (Audio3d) and the Acm
// batch-command queue. Both were unregistered, so the dispatcher answered `0` — SCE_OK for
// status-returning contracts — while writing nothing (see audio/AGENTS.md).
//
// Evidence: the PS5 system modules for both libraries are plain ELF, so every contract below is
// read from the module itself (offsets are text-segment offsets into that module). No Sony code is
// reproduced; only the guest-visible ABI is re-derived. Local importing call sites (Alex Kidd DX,
// BALAN Wonderworld, R-Type Delta) were read to check how the guest consumes each answer.
// shadPS4's audio3d headers were the first hypothesis for the struct layout and error values;
// every one of those was then confirmed or corrected against the module.
//
// Model, per library:
// - Audio3d has ONE port: the module's lookup (+0x1a00) maps only id 0, to a single global
//   handle, so PortOpen hands out id 0 and a second open while it is held fails OUT_OF_RESOURCES
//   (+0x1a60). PortClose frees it (+0x2110); Terminate refuses while it is open (+0xf90). The
//   port is a headless silent sink: nothing is queued, the queue always reads empty.
// - Acm: contexts get local ids standing in for the module's /dev/acm fds; prosper has no batch
//   engine, so the batch calls fail with the module's catch-all code and never write the batch id
//   (the module writes it only on success).
// - libSceAudioPropagation is deliberately NOT registered. Its one local importer (Astro Bot)
//   asserts on a failed QueryMemory/SystemCreate and then carries on as if it had succeeded, so a
//   failing stub adds a debug-trap path without changing what the title ends up doing.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/audio/audio.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace prosper {

#define HLE(name)                                                                                  \
    static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,      \
                                          uint64_t a4, uint64_t a5)

namespace {

constexpr uint64_t sext32(uint32_t v) {
    return (uint64_t)(int64_t)(int32_t)v;
}

// libSceAudio3d facility (0x80EA): the module's own error-to-string table (+0x1040) names
// 0x80EA0002 "port does not exist", 0x80EA0004 "invalid parameter", 0x80EA0006 "out of
// resources", 0x80EA0007 "not ready"; 0x80EA0008 is what the per-port calls return for a buffer
// mode they do not support. CONFIDENCE: HIGH.
constexpr uint64_t kA3ErrInvalidPort = sext32(0x80EA0002u);
constexpr uint64_t kA3ErrInvalidParam = sext32(0x80EA0004u);
constexpr uint64_t kA3ErrNoResources = sext32(0x80EA0006u);
constexpr uint64_t kA3ErrNotReady = sext32(0x80EA0007u);
constexpr uint64_t kA3ErrUnsupportedMode = sext32(0x80EA0008u);

// libSceAcm facility (0x8194), from the module: 0x81940006 for a null context pointer
// (ContextCreate +0xd0), 0x81940005 when the fd table is full (ENFILE/EMFILE), 0x81940002 for a
// bad context (EBADF and friends, every call that takes one), and 0x81940001 as the catch-all for
// any other failure. CONFIDENCE: HIGH on the values; MED that the catch-all is the right answer
// for "no batch engine", which no real console ever reports.
constexpr uint64_t kAcmErrGeneric = sext32(0x81940001u);
constexpr uint64_t kAcmErrBadContext = sext32(0x81940002u);
constexpr uint64_t kAcmErrNoContexts = sext32(0x81940005u);
constexpr uint64_t kAcmErrNullPointer = sext32(0x81940006u);

constexpr uint32_t kA3Port = 0u;   // the only port id the module maps
constexpr uint32_t kA3UserId = 0xFFu;   // PortOpen requires user id 0xFF (+0x1b4c)

// Open-parameter block, normalised by the module's parser (+0x1d70) to its 0x28 form:
// size_this(u64) granularity(u32) rate(u32) max_objects(u32) queue_depth(u32) buffer_mode(u32)
// pad(u32) num_beds(u32). Shorter versions keep defaults for the fields they lack.
constexpr uint64_t kA3DefaultSizeThis = 0x20u;
constexpr uint32_t kA3DefaultGranularity = 0x100u;
constexpr uint32_t kA3DefaultRate = 0u;   // 48 kHz
constexpr uint32_t kA3DefaultMaxObjects = 0x200u;
constexpr uint32_t kA3DefaultQueueDepth = 2u;
constexpr uint32_t kA3DefaultBufferMode = 2u;   // advance-and-push
constexpr uint32_t kA3DefaultBeds = 2u;

bool load_u32(uint64_t addr, uint32_t* out) {
    uint32_t v = 0;
    if (!addr || !audio_read_bytes(addr, &v, sizeof v)) return false;
    *out = v;
    return true;
}
bool load_u64(uint64_t addr, uint64_t* out) {
    uint64_t v = 0;
    if (!addr || !audio_read_bytes(addr, &v, sizeof v)) return false;
    *out = v;
    return true;
}
bool store_u32(uint64_t addr, uint32_t v) {
    return addr && audio_store_bytes(addr, &v, sizeof v);
}

std::mutex g_a3_mx;
bool g_a3_ready = false;
bool g_a3_port_open = false;
uint32_t g_a3_depth = kA3DefaultQueueDepth;
uint32_t g_a3_buffer_mode = kA3DefaultBufferMode;

// Caller holds g_a3_mx. The module maps only id 0, and only while the port is open.
bool a3_port_live(uint64_t id) {
    return (uint32_t)id == kA3Port && g_a3_port_open;
}

std::mutex g_acm_mx;
std::array<bool, 8> g_acm_used{};

}  // namespace

// sceAudio3dInitialize(int64 reserved) -> 0. A nonzero reserved argument is INVALID_PARAM
// (+0xce0). A repeated Initialize returns NOT_READY in the module only behind an SDK-version
// gate; the local FMOD call site accepts 0 or NOT_READY and BALAN treats any non-zero as fatal,
// so answering 0 is the choice that is safe for every local caller, not an absence of evidence.
HLE(audio3d_initialize) {
    if ((int64_t)a0 != 0) return kA3ErrInvalidParam;
    std::lock_guard<std::mutex> lk(g_a3_mx);
    g_a3_ready = true;
    return 0;
}

// sceAudio3dTerminate() -> 0. NOT_READY while the port is open or before Initialize (+0xf90);
// otherwise it clears the ready flag.
HLE(audio3d_terminate) {
    std::lock_guard<std::mutex> lk(g_a3_mx);
    if (g_a3_port_open || !g_a3_ready) return kA3ErrNotReady;
    g_a3_ready = false;
    return 0;
}

// void sceAudio3dGetDefaultOpenParameters(params*). The module (+0x19a0) writes bytes
// 0x00..0x1b — size_this 0x20 and the six fields after it — never the pad at 0x1c, and does
// nothing for a null pointer. It returns no status; 0 is returned only because every HLE
// function returns something.
HLE(audio3d_get_default_open_parameters) {
    if (!a0) return 0;
    uint8_t raw[0x1c]{};
    const uint64_t size_this = kA3DefaultSizeThis;
    const uint32_t fields[5] = {kA3DefaultGranularity, kA3DefaultRate, kA3DefaultMaxObjects,
                                kA3DefaultQueueDepth, kA3DefaultBufferMode};
    std::memcpy(raw + 0x00, &size_this, sizeof size_this);
    std::memcpy(raw + 0x08, fields, sizeof fields);
    // granularity@0x08 rate@0x0c max_objects@0x10 queue_depth@0x14 buffer_mode@0x18
    audio_store_bytes(a0, raw, sizeof raw);
    return 0;
}

// sceAudio3dPortOpen(user_id, params*, port*) -> 0, writing *port = 0. Order follows the module
// (+0x1a60): NOT_READY before Initialize; INVALID_PARAM for a bad user id, a null port pointer or
// a parameter block that fails validation; OUT_OF_RESOURCES while the one port is held. *port is
// left untouched on all of those, so a guest's own preset survives a refusal.
HLE(audio3d_port_open) {
    {
        std::lock_guard<std::mutex> lk(g_a3_mx);
        if (!g_a3_ready) return kA3ErrNotReady;
    }
    if ((uint32_t)a0 != kA3UserId || !a1 || !a2) return kA3ErrInvalidParam;
    uint64_t size_this = 0;
    uint32_t granularity = 0, rate = 0, max_objects = kA3DefaultMaxObjects,
             queue_depth = kA3DefaultQueueDepth, buffer_mode = 0, beds = kA3DefaultBeds;
    if (!load_u64(a1 + 0x00, &size_this) || !load_u32(a1 + 0x08, &granularity) ||
        !load_u32(a1 + 0x0c, &rate))
        return kA3ErrInvalidParam;
    // The parser accepts these four sizes exactly (no masking) and fills the missing fields:
    // 0x10 -> buffer_mode 0; 0x18 -> buffer_mode 1; beds default to 2 below 0x28.
    switch (size_this) {
        case 0x10: buffer_mode = 0; break;
        case 0x18: buffer_mode = 1; break;
        case 0x20:
        case 0x28:
            if (!load_u32(a1 + 0x18, &buffer_mode)) return kA3ErrInvalidParam;
            break;
        default: return kA3ErrInvalidParam;
    }
    if (size_this >= 0x18 &&
        (!load_u32(a1 + 0x10, &max_objects) || !load_u32(a1 + 0x14, &queue_depth)))
        return kA3ErrInvalidParam;
    if (size_this == 0x28 && !load_u32(a1 + 0x20, &beds)) return kA3ErrInvalidParam;
    // Field checks from the module (+0x1b4c..+0x1bd4). There is no upper bound on queue_depth.
    if (rate != kA3DefaultRate) return kA3ErrInvalidParam;
    if (granularity < 0x100 || (granularity & 0xFF) != 0) return kA3ErrInvalidParam;
    if (max_objects == 0 || queue_depth == 0) return kA3ErrInvalidParam;
    // The module accepts buffer modes 0..2 and 2..3 beds when an SDK-version query (an import
    // prosper does not model) is non-zero, and only buffer mode 2 otherwise. The permissive branch
    // is taken here; every local caller passes mode 2 and 2 beds. CONFIDENCE: MED.
    if (buffer_mode > 2) return kA3ErrInvalidParam;
    if (beds != 2 && beds != 3) return kA3ErrInvalidParam;

    std::lock_guard<std::mutex> lk(g_a3_mx);
    if (!g_a3_ready) return kA3ErrNotReady;
    if (g_a3_port_open) return kA3ErrNoResources;
    if (!store_u32(a2, kA3Port)) return kA3ErrInvalidParam;
    g_a3_port_open = true;
    g_a3_depth = queue_depth;
    g_a3_buffer_mode = buffer_mode;
    return 0;
}

// sceAudio3dPortClose(port) -> 0. PORT_DOES_NOT_EXIST for anything but the open port 0 (+0x2110);
// it does not check Initialize.
HLE(audio3d_port_close) {
    std::lock_guard<std::mutex> lk(g_a3_mx);
    if (!a3_port_live(a0)) return kA3ErrInvalidPort;
    g_a3_port_open = false;
    return 0;
}

// sceAudio3dPortSetAttribute(port, attribute_id, attribute*, size) -> 0.
HLE(audio3d_port_set_attribute) {
    std::lock_guard<std::mutex> lk(g_a3_mx);
    if (!a3_port_live(a0)) return kA3ErrInvalidPort;
    if (!a2) return kA3ErrInvalidParam;
    return 0;
}

// sceAudio3dPortGetQueueLevel(port, level*, available*) -> 0. Buffer modes 0 and 1 are
// UNSUPPORTED_MODE (+0x2550). The headless queue is always empty, so available is the port's
// depth, the divisor a guest paces pushes against. A null level pointer is valid (BALAN passes
// one); both null is refused.
HLE(audio3d_port_get_queue_level) {
    std::lock_guard<std::mutex> lk(g_a3_mx);
    if (!a3_port_live(a0)) return kA3ErrInvalidPort;
    if (g_a3_buffer_mode == 0 || g_a3_buffer_mode == 1) return kA3ErrUnsupportedMode;
    if (!a1 && !a2) return kA3ErrInvalidParam;
    if (a1 && !store_u32(a1, 0)) return kA3ErrInvalidParam;
    if (a2 && !store_u32(a2, g_a3_depth)) return kA3ErrInvalidParam;
    return 0;
}

// sceAudio3dPortAdvance(port) -> 0. Buffer mode 0 is UNSUPPORTED_MODE (+0x23b0). Nothing is
// queued headless, so there is nothing to advance.
HLE(audio3d_port_advance) {
    std::lock_guard<std::mutex> lk(g_a3_mx);
    if (!a3_port_live(a0)) return kA3ErrInvalidPort;
    if (g_a3_buffer_mode == 0) return kA3ErrUnsupportedMode;
    return 0;
}

// sceAudio3dPortPush(port, blocking) -> 0. Buffer modes 0 and 1 are UNSUPPORTED_MODE (+0x2470).
// A headless no-op on the live port (silent-sink philosophy). It does not pace a blocking push;
// BALAN's audio thread therefore spins, exactly as it did while this NID was unregistered.
HLE(audio3d_port_push) {
    std::lock_guard<std::mutex> lk(g_a3_mx);
    if (!a3_port_live(a0)) return kA3ErrInvalidPort;
    if (g_a3_buffer_mode == 0 || g_a3_buffer_mode == 1) return kA3ErrUnsupportedMode;
    return 0;
}

// sceAcmContextCreate(int* context) -> 0. The module (+0xd0) refuses a null pointer, writes -1,
// then opens /dev/acm and stores the fd. Here a local id stands in for the fd.
HLE(acm_context_create) {
    if (!a0) return kAcmErrNullPointer;
    std::lock_guard<std::mutex> lk(g_acm_mx);
    if (!store_u32(a0, 0xFFFFFFFFu)) return kAcmErrNullPointer;
    for (size_t i = 0; i < g_acm_used.size(); i++) {
        if (g_acm_used[i]) continue;
        if (!store_u32(a0, (uint32_t)(i + 1))) return kAcmErrNullPointer;
        g_acm_used[i] = true;
        return 0;
    }
    return kAcmErrNoContexts;
}

// Caller holds g_acm_mx.
static bool acm_context_live(uint64_t ctx) {
    const uint32_t slot = (uint32_t)ctx;
    return slot >= 1 && slot <= g_acm_used.size() && g_acm_used[slot - 1];
}

// sceAcmContextDestroy(int context) -> 0. A context it did not hand out is BAD_CONTEXT (+0x140).
HLE(acm_context_destroy) {
    std::lock_guard<std::mutex> lk(g_acm_mx);
    if (!acm_context_live(a0)) return kAcmErrBadContext;
    g_acm_used[(uint32_t)a0 - 1] = false;
    return 0;
}

// sceAcmBatchStartBuffers(ctx, count, list*, ..., batch_id*) / sceAcmBatchStartBuffer (the
// module's single-buffer wrapper, +0x290) / sceAcmBatchWait(ctx, batch_id, timeout). prosper has
// no batch engine, so a live context gets the catch-all failure; an unknown one is BAD_CONTEXT.
// The batch id is never written: the module writes it only on success (+0x22b).
HLE(acm_batch_start_buffer) {
    std::lock_guard<std::mutex> lk(g_acm_mx);
    return acm_context_live(a0) ? kAcmErrGeneric : kAcmErrBadContext;
}
HLE(acm_batch_start_buffers) {
    std::lock_guard<std::mutex> lk(g_acm_mx);
    return acm_context_live(a0) ? kAcmErrGeneric : kAcmErrBadContext;
}
HLE(acm_batch_wait) {
    std::lock_guard<std::mutex> lk(g_acm_mx);
    return acm_context_live(a0) ? kAcmErrGeneric : kAcmErrBadContext;
}

void register_audio_spatial_hle() {
#define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    // libSceAudio3d: one local port, module-evidenced 0x80EA00xx errors.
    R("sceAudio3dInitialize", audio3d_initialize);
    R("sceAudio3dTerminate", audio3d_terminate);
    R("sceAudio3dGetDefaultOpenParameters", audio3d_get_default_open_parameters);
    R("sceAudio3dPortOpen", audio3d_port_open);
    R("sceAudio3dPortClose", audio3d_port_close);
    R("sceAudio3dPortSetAttribute", audio3d_port_set_attribute);
    R("sceAudio3dPortGetQueueLevel", audio3d_port_get_queue_level);
    R("sceAudio3dPortAdvance", audio3d_port_advance);
    R("sceAudio3dPortPush", audio3d_port_push);
    // libSceAcm: local context ids; the batch engine reports the module's catch-all failure.
    R("sceAcmContextCreate", acm_context_create);
    R("sceAcmContextDestroy", acm_context_destroy);
    R("sceAcmBatchStartBuffer", acm_batch_start_buffer);
    R("sceAcmBatchStartBuffers", acm_batch_start_buffers);
    R("sceAcmBatchWait", acm_batch_wait);
// libSceAudioPropagation is deliberately left unregistered (see the file header).
#undef R
}

}   // namespace prosper
