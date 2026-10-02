// hle_prt_aperture.cpp — sceKernelSetPrtAperture / sceKernelGetPrtAperture.
//
// A PRT (partially resident texture) aperture is a window of the guest's virtual address space that
// the GPU treats as sparsely backed. prosper's guest memory is host memory and its GPU path resolves
// textures from guest addresses directly, so an aperture needs no host-side mapping: the useful and
// observable behaviour is the contract of the two calls themselves, which is bookkeeping with a fixed
// number of slots. These used to fall through to the "unimplemented, returning 0" logger, which gets
// a valid call right and gets two things wrong: an out-of-range slot index is accepted, and a
// GetPrtAperture reads back nothing.
//
// Slots: three. Two independently written secondary implementations agree on a count of 3 and on
// EINVAL for an index outside [0, 3). CONFIDENCE: MED on both (secondary agreement only; no title or
// firmware evidence yet). One of them additionally rejects an unaligned address or one outside a
// fixed PRT area, but that area is a PS4 constant and nothing here shows the PS5 value, so no
// address validation is added.
//
// Nothing here is title-specific.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"

#include <cstdint>
#include <mutex>

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)

namespace prosper {
namespace {

constexpr int kPrtApertureCount = 3;

struct PrtAperture {
    uint64_t address = 0;
    uint64_t length = 0;
};

std::mutex g_prt_mutex;
PrtAperture g_prt_apertures[kPrtApertureCount];

bool valid_index(uint64_t raw, int* out) {
    const int32_t index = static_cast<int32_t>(raw);   // `int` parameter: only the low 32 bits count
    if (index < 0 || index >= kPrtApertureCount) return false;
    *out = index;
    return true;
}

// sceKernelSetPrtAperture(int index, void* address, size_t length) -> 0 or EINVAL
HLE(k_set_prt_aperture) {
    (void)a3; (void)a4; (void)a5;
    int index = 0;
    if (!valid_index(a0, &index)) return prosper::hle::kSceKernelErrorEINVAL;
    std::lock_guard<std::mutex> lock(g_prt_mutex);
    g_prt_apertures[index] = {a1, a2};
    return 0;
}

// sceKernelGetPrtAperture(int index, void** address, size_t* length) -> 0 or EINVAL
// A slot never set reads back {0, 0}. A null out-pointer is EINVAL: the real contract is not known
// and writing through it would fault the host (CONFIDENCE: LOW on that choice).
HLE(k_get_prt_aperture) {
    (void)a3; (void)a4; (void)a5;
    int index = 0;
    if (!valid_index(a0, &index) || !a1 || !a2) return prosper::hle::kSceKernelErrorEINVAL;
    PrtAperture snapshot;
    {
        std::lock_guard<std::mutex> lock(g_prt_mutex);
        snapshot = g_prt_apertures[index];
    }
    *reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(a1)) = snapshot.address;
    *reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(a2)) = snapshot.length;
    return 0;
}

}   // namespace

void register_prt_aperture_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceKernelSetPrtAperture", k_set_prt_aperture);
    R("sceKernelGetPrtAperture", k_get_prt_aperture);
    #undef R
}

}   // namespace prosper
