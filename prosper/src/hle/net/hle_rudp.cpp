// libSceRudp — reliable-UDP setup calls (sceRudpEnableInternalIOThread,
// sceRudpSetEventHandler).
//
// Both were unregistered, so the dispatcher answered `0`. They are setup calls with no
// out-parameters: enabling an IO thread nobody schedules and registering a handler nothing ever
// calls. prosper runs no network IO thread and delivers no RUDP events, so both are acknowledged
// and nothing is stored. Both NIDs are in the 3.20 libSceRudp export set, and the contracts below
// follow the shipped libSceRudp module.
//
// sceRudpInit (amuBfI-AQc4, also in the 3.20 set; ASTRO BOT imports it) stays unregistered for
// now: its dispatcher answer of 0 already reads as "initialized", and a real Init needs the
// memory-pool contract modelled. Known gap: on hardware both setup calls return 0x80770001 when
// sceRudpInit has not run (+0x60f7, +0x620d); that is not modelled while Init is unregistered.
// CONFIDENCE: HIGH on the NULL-handler refusal (+0x61f2 -> +0x6214), HIGH that acknowledgement
// is safe here (no outputs exist to lie about).
#include "hle/dispatch/dispatch.hpp"

#include <cstdint>

namespace prosper {

#define HLE(name)                                                                                  \
    static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,      \
                                          uint64_t a4, uint64_t a5)

HLE(rudp_enable_io_thread) {  // (stack_size, priority) -> SCE_OK, no thread started
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    return 0;
}

HLE(rudp_set_event_handler) {  // (handler, arg) -> SCE_OK, stored nowhere, never called
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    if (!a0) return 0x80770022u;   // a NULL handler is refused (+0x61f2 -> +0x6214)
    return 0;
}

void register_rudp_hle() {
    Hle::register_fn("6PBNpsgyaxw", (HleFn)rudp_enable_io_thread, "sceRudpEnableInternalIOThread");
    Hle::register_fn("SUEVes8gvmw", (HleFn)rudp_set_event_handler, "sceRudpSetEventHandler");
}

}  // namespace prosper
