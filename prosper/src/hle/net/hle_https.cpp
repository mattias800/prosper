// libSceHttps — TLS setup for Http objects (sceHttpsSetSslCallback, sceHttpsDisableOption).
//
// Both were unregistered, so the dispatcher answered `0`. Uncharted's single SetSslCallback
// call site drops the result entirely (#3630 bucket B: genuinely harmless either way), but the
// unimplemented line hides the call in census noise and leaves the next Https entry point to
// the same default. These are pure setters — no out-parameters — so a documented no-op cannot
// recreate the unwritten-buffer shape; prosper performs no TLS verification headless, and a
// callback it would never invoke is acknowledged rather than stored. Deliberately NOT validated
// against Http's id table: with no outputs and no state, a garbage id and a live id answer
// identically, so validation would add cross-file coupling for zero behavioral gain.
// Signatures (id, callback, user_arg) and (id, ssl_flags) from the stub interface; NIDs from
// the firmware set (verified by nid_hash round-trip). CONFIDENCE: MED on the arities (single
// secondary source), HIGH that no-op success is safe here (no outputs; the one observed caller
// ignores the answer).
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <cstdint>

namespace prosper {

#define HLE(name)                                                                                  \
    static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,      \
                                          uint64_t a4, uint64_t a5)

HLE(h_https_set_ssl_callback) {  // (id, callback, user_arg) -> SCE_OK, acknowledged
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    return 0;
}

HLE(h_https_disable_option) {  // (id, ssl_flags) -> SCE_OK, acknowledged
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    return 0;
}

void register_https_hle() {
    Hle::register_fn("htyBOoWeS58", (HleFn)h_https_set_ssl_callback, "sceHttpsSetSslCallback");
    Hle::register_fn("mSQCxzWTwVI", (HleFn)h_https_disable_option, "sceHttpsDisableOption");
}

}  // namespace prosper
