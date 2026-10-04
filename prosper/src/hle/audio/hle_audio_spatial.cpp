// hle_audio_spatial.cpp — libSceAudio3d / libSceAudioPropagation / libSceAcm HLE.
//
// The spatial/batch corner of guest audio: object-based 3D ports (Audio3d), the Tempest
// ray-propagation system (AudioPropagation), and the Acm batch-command queue. All three were
// unregistered, so the dispatcher answered `0` — SCE_OK for status-returning contracts — while
// writing nothing. Audio out-parameters are overwhelmingly divisors (queue depths, ray counts,
// batch ids), so the crash lands in the guest far from the cause (see audio/AGENTS.md).
//
// Evidence, per source (no Sony code is reproduced; only the guest-visible ABI is re-derived):
// - Names and arities: the firmware stub export lists (7/6/5 functions respectively), cross-checked
//   for Audio3d against the PS5 3.20 NID set (all 7 NIDs) and the ps4libdoc name list.
// - Audio3d struct layout (OrbisAudio3dOpenParameters), signatures, validation shape, port-id
//   allocation (first free of 0..3) and the 0x80EA00xx error facility: shadPS4's audio3d headers,
//   a single secondary implementation. CONFIDENCE: MED on the facility values; HIGH on the struct
//   layout, which two independent reimplementations agree on field-for-field.
// - sceAcmContextCreate: called once each by two live titles (BEAST_OF_REINCARNATION_STATUS,
//   SYBERIA_STATUS `libSceAcm::ZIXln2K3XMk`); nid_hash("sceAcmContextCreate") == ZIXln2K3XMk
//   resolves that orphan NID and confirms the name. Neither title blocks on it.
// - AudioPropagation and the Acm batch path have no backend in prosper (no ray tracer, no batch
//   engine) and no evidenced error facility, so they report the subsystem unavailable with the
//   same hard, non-retryable generic the voice library uses for missing host hardware
//   (hle_audio.cpp `voice_init_unavailable`, 0x80410002 — the guest only sign-checks via `js`).
//   CONFIDENCE: LOW on the exact code; HIGH that an error (never silent success) is the contract.
//
// Honest model, per library:
// - Audio3d: real local lifecycle. Initialize is idempotent success (no title evidence for a
//   double-init failure, and refusing one would invent a contract — cf. hle_audiodec.cpp).
//   GetDefaultOpenParameters fills the versioned 0x20 prefix; PortOpen validates the parameter
//   block and hands out ports 0..3; queue level reports an empty queue against the stored depth;
//   Push/Advance/SetAttribute on a live port are headless no-ops (the silent-sink philosophy:
//   nothing is queued, nothing plays, the call succeeds). Unknown ports fail INVALID_PORT and
//   failed opens publish PORT_INVALID, so the guest never proceeds on a garbage handle.
// - Acm: contexts get local ids (create/destroy validated); the batch engine itself is
//   unavailable, so StartBuffer(s)/Wait fail without touching their out-parameters.
// - AudioPropagation: the whole subsystem is unavailable; every entry point fails without
//   touching its out-parameters, so a title routes around propagation instead of dividing by
//   never-written ray counts or allocating a zero-byte system block.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/audio/audio.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                                        uint64_t a3, uint64_t a4, uint64_t a5)

namespace {

constexpr uint64_t sext32(uint32_t v) { return (uint64_t)(int64_t)(int32_t)v; }

// libSceAudio3d facility (0x80EA). Source: shadPS4 audio3d_error.h. CONFIDENCE: MED.
constexpr uint64_t kA3ErrInvalidPort = sext32(0x80EA0002u);
constexpr uint64_t kA3ErrInvalidParam = sext32(0x80EA0004u);
constexpr uint64_t kA3ErrNoResources = sext32(0x80EA0006u);
constexpr uint64_t kA3ErrNotReady = sext32(0x80EA0007u);

// No-backend refusal for AudioPropagation and the Acm batch path. Same hard, non-retryable
// generic the voice library uses for missing host hardware (NOT an already-initialized-style
// code a guest could mistake for success). CONFIDENCE: LOW on the exact value.
constexpr uint64_t kAudioNoBackend = sext32(0x80410002u);

constexpr uint32_t kA3PortInvalid = 0xFFFFFFFFu;
constexpr uint32_t kA3MaxPorts = 4;

// OrbisAudio3dOpenParameters, versioned prefix (size_this == 0x20). Both reimplementations
// agree: size_this, granularity, rate, max_objects, queue_depth, buffer_mode, pad.
constexpr uint64_t kA3DefaultSizeThis = 0x20u;
constexpr uint32_t kA3DefaultGranularity = 0x100u;
constexpr uint32_t kA3DefaultRate = 0u;  // ORBIS_AUDIO3D_RATE_48000
constexpr uint32_t kA3DefaultMaxObjects = 512u;
constexpr uint32_t kA3DefaultQueueDepth = 2u;
constexpr uint32_t kA3DefaultBufferMode = 2u;  // ORBIS_AUDIO3D_BUFFER_ADVANCE_AND_PUSH

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
struct A3Port {
    bool used = false;
    uint32_t depth = kA3DefaultQueueDepth;
};
std::array<A3Port, kA3MaxPorts> g_a3_ports;

// Caller holds g_a3_mx.
int a3_port_index(uint32_t id) {
    if (id >= kA3MaxPorts || !g_a3_ports[id].used) return -1;
    return (int)id;
}

std::mutex g_acm_mx;
std::array<bool, 8> g_acm_used{};

}  // namespace

// sceAudio3dInitialize(int64 reserved) -> 0. Idempotent: no title evidence supports a
// double-init failure, and refusing one would invent a contract.
HLE(audio3d_initialize) {
    if ((int64_t)a0 != 0) return kA3ErrInvalidParam;
    std::lock_guard<std::mutex> lk(g_a3_mx);
    g_a3_ready = true;
    return 0;
}

// sceAudio3dGetDefaultOpenParameters(params*) -> 0. Fills the versioned 0x20 prefix only:
// size_this declares the struct's known size, so writing past it could corrupt a smaller
// guest buffer, while the fields inside it are exactly what a guest divides by.
HLE(audio3d_get_default_open_parameters) {
    if (!a0) return kA3ErrInvalidParam;
    uint8_t raw[0x20]{};
    uint64_t size_this = kA3DefaultSizeThis;
    uint32_t granularity = kA3DefaultGranularity, rate = kA3DefaultRate,
             max_objects = kA3DefaultMaxObjects, queue_depth = kA3DefaultQueueDepth,
             buffer_mode = kA3DefaultBufferMode, pad = 0;
    std::memcpy(raw + 0x00, &size_this, sizeof size_this);
    std::memcpy(raw + 0x08, &granularity, sizeof granularity);
    std::memcpy(raw + 0x0c, &rate, sizeof rate);
    std::memcpy(raw + 0x10, &max_objects, sizeof max_objects);
    std::memcpy(raw + 0x14, &queue_depth, sizeof queue_depth);
    std::memcpy(raw + 0x18, &buffer_mode, sizeof buffer_mode);
    std::memcpy(raw + 0x1c, &pad, sizeof pad);
    if (!audio_store_bytes(a0, raw, sizeof raw)) return kA3ErrInvalidParam;
    return 0;
}

// sceAudio3dPortOpen(user_id, params*, id*) -> 0. Validates the versioned parameter block,
// then hands out the first free port of 0..3. Any failure publishes PORT_INVALID when the
// id pointer is writable, so the guest never proceeds on a garbage handle.
HLE(audio3d_port_open) {
    if (!a1 || !a2) {
        if (a2) store_u32(a2, kA3PortInvalid);
        return kA3ErrInvalidParam;
    }
    store_u32(a2, kA3PortInvalid);
    {
        std::lock_guard<std::mutex> lk(g_a3_mx);
        if (!g_a3_ready) return kA3ErrNotReady;
    }
    uint64_t size_this = 0;
    uint32_t granularity = 0, rate = 0, max_objects = 0, queue_depth = 0, buffer_mode = 0;
    if (!load_u64(a1 + 0x00, &size_this) || !load_u32(a1 + 0x08, &granularity))
        return kA3ErrInvalidParam;
    switch (size_this & ~0x7ull) {
        case 0x10:
        case 0x18:
        case 0x20:
        case 0x28:
            break;
        default:
            return kA3ErrInvalidParam;
    }
    // rate/max_objects/queue_depth/buffer_mode live past 0x10; older blocks keep defaults.
    if ((size_this & ~0x7ull) >= 0x10 && !load_u32(a1 + 0x0c, &rate)) return kA3ErrInvalidParam;
    if ((size_this & ~0x7ull) >= 0x18 &&
        (!load_u32(a1 + 0x10, &max_objects) || !load_u32(a1 + 0x14, &queue_depth)))
        return kA3ErrInvalidParam;
    if ((size_this & ~0x7ull) >= 0x20 && !load_u32(a1 + 0x18, &buffer_mode))
        return kA3ErrInvalidParam;
    if (rate != kA3DefaultRate) return kA3ErrInvalidParam;
    if (granularity < 0x100 || (granularity & 0xFF) != 0) return kA3ErrInvalidParam;
    if ((size_this & ~0x7ull) >= 0x18) {
        if (max_objects == 0 || queue_depth == 0 || queue_depth > 0x40) return kA3ErrInvalidParam;
    }
    if ((size_this & ~0x7ull) >= 0x20 && buffer_mode > 2) return kA3ErrInvalidParam;

    std::lock_guard<std::mutex> lk(g_a3_mx);
    if (!g_a3_ready) return kA3ErrNotReady;
    for (uint32_t i = 0; i < kA3MaxPorts; i++) {
        if (g_a3_ports[i].used) continue;
        g_a3_ports[i].used = true;
        g_a3_ports[i].depth =
            ((size_this & ~0x7ull) >= 0x18 && queue_depth != 0) ? queue_depth
                                                               : kA3DefaultQueueDepth;
        if (!store_u32(a2, i)) {
            g_a3_ports[i] = {};
            return kA3ErrInvalidParam;
        }
        return 0;
    }
    return kA3ErrNoResources;
}

// sceAudio3dPortSetAttribute(port, attribute_id, attribute*, size) -> 0.
HLE(audio3d_port_set_attribute) {
    std::lock_guard<std::mutex> lk(g_a3_mx);
    if (a3_port_index((uint32_t)a0) < 0) return kA3ErrInvalidPort;
    if (!a2) return kA3ErrInvalidParam;
    return 0;
}

// sceAudio3dPortGetQueueLevel(port, level*, available*) -> 0. The headless queue is always
// empty; available is the port's depth, which is the divisor a guest paces pushes against.
HLE(audio3d_port_get_queue_level) {
    std::lock_guard<std::mutex> lk(g_a3_mx);
    const int idx = a3_port_index((uint32_t)a0);
    if (idx < 0) return kA3ErrInvalidPort;
    if (!a1 && !a2) return kA3ErrInvalidParam;
    if (a1 && !store_u32(a1, 0)) return kA3ErrInvalidParam;
    if (a2 && !store_u32(a2, g_a3_ports[(size_t)idx].depth)) return kA3ErrInvalidParam;
    return 0;
}

// sceAudio3dPortAdvance(port) -> 0. Nothing is queued headless, so there is nothing to advance.
HLE(audio3d_port_advance) {
    std::lock_guard<std::mutex> lk(g_a3_mx);
    if (a3_port_index((uint32_t)a0) < 0) return kA3ErrInvalidPort;
    return 0;
}

// sceAudio3dPortPush(port, blocking) -> 0. Headless no-op on a live port (silent-sink
// philosophy); an unknown port is INVALID_PORT, never silent success on a garbage handle.
HLE(audio3d_port_push) {
    std::lock_guard<std::mutex> lk(g_a3_mx);
    if (a3_port_index((uint32_t)a0) < 0) return kA3ErrInvalidPort;
    return 0;
}

// sceAcmContextCreate(context*) -> 0. Hands out a local context id. Called once each by two
// live titles without blocking either, so a local id plus success keeps that behavior while
// giving the guest a handle its destroy path can validate.
HLE(acm_context_create) {
    if (!a0) return kAudioNoBackend;
    std::lock_guard<std::mutex> lk(g_acm_mx);
    for (size_t i = 0; i < g_acm_used.size(); i++) {
        if (g_acm_used[i]) continue;
        g_acm_used[i] = true;
        if (!store_u32(a0, (uint32_t)(i + 1))) {
            g_acm_used[i] = false;
            return kAudioNoBackend;
        }
        return 0;
    }
    return kAudioNoBackend;
}

// sceAcmContextDestroy(context) -> 0. Frees exactly the ids ContextCreate handed out.
HLE(acm_context_destroy) {
    std::lock_guard<std::mutex> lk(g_acm_mx);
    const uint32_t slot = (uint32_t)a0;
    if (slot < 1 || slot > g_acm_used.size() || !g_acm_used[slot - 1]) return kAudioNoBackend;
    g_acm_used[slot - 1] = false;
    return 0;
}

// sceAcmBatchStartBuffer / sceAcmBatchStartBuffers / sceAcmBatchWait: prosper has no batch
// engine, so the batch path is unavailable. Fail without touching the sideband out-parameters
// (batch_error, batch id), which the guest would otherwise read as a completed batch.
HLE(acm_batch_start_buffer) {
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    return kAudioNoBackend;
}
HLE(acm_batch_start_buffers) {
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    return kAudioNoBackend;
}
HLE(acm_batch_wait) {
    (void)a0;
    (void)a1;
    (void)a2;
    return kAudioNoBackend;
}

// libSceAudioPropagation: the whole subsystem is unavailable (no ray tracer in prosper), so
// every entry point fails without touching its out-parameters. Failing at SystemCreate routes
// a title around propagation up front (the voice-init precedent); writing a zeroed memory
// block or zero ray counts plus success would strand it dividing by never-filled fields.
HLE(audio_propagation_system_create) {
    (void)a0;
    (void)a1;
    (void)a2;
    return kAudioNoBackend;
}
HLE(audio_propagation_system_query_memory) {
    (void)a0;
    (void)a1;
    return kAudioNoBackend;
}
HLE(audio_propagation_room_create) {
    (void)a0;
    (void)a1;
    return kAudioNoBackend;
}
HLE(audio_propagation_system_register_material) {
    (void)a0;
    (void)a1;
    (void)a2;
    return kAudioNoBackend;
}
HLE(audio_propagation_system_set_attributes) {
    (void)a0;
    (void)a1;
    (void)a2;
    return kAudioNoBackend;
}
HLE(audio_propagation_system_get_rays) {
    (void)a0;
    (void)a1;
    (void)a2;
    return kAudioNoBackend;
}

void register_audio_spatial_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    // libSceAudio3d: local port lifecycle + evidenced 0x80EA00xx errors.
    R("sceAudio3dInitialize", audio3d_initialize);
    R("sceAudio3dGetDefaultOpenParameters", audio3d_get_default_open_parameters);
    R("sceAudio3dPortOpen", audio3d_port_open);
    R("sceAudio3dPortSetAttribute", audio3d_port_set_attribute);
    R("sceAudio3dPortGetQueueLevel", audio3d_port_get_queue_level);
    R("sceAudio3dPortAdvance", audio3d_port_advance);
    R("sceAudio3dPortPush", audio3d_port_push);
    // libSceAcm: local context ids; the batch engine reports unavailable.
    R("sceAcmContextCreate", acm_context_create);
    R("sceAcmContextDestroy", acm_context_destroy);
    R("sceAcmBatchStartBuffer", acm_batch_start_buffer);
    R("sceAcmBatchStartBuffers", acm_batch_start_buffers);
    R("sceAcmBatchWait", acm_batch_wait);
    // libSceAudioPropagation: subsystem unavailable; every entry point fails loud.
    R("sceAudioPropagationSystemCreate", audio_propagation_system_create);
    R("sceAudioPropagationSystemQueryMemory", audio_propagation_system_query_memory);
    R("sceAudioPropagationRoomCreate", audio_propagation_room_create);
    R("sceAudioPropagationSystemRegisterMaterial", audio_propagation_system_register_material);
    R("sceAudioPropagationSystemSetAttributes", audio_propagation_system_set_attributes);
    R("sceAudioPropagationSystemGetRays", audio_propagation_system_get_rays);
    #undef R
}

}  // namespace prosper
