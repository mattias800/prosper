// libSceRudp — reliable-UDP setup calls (sceRudpEnableInternalIOThread,
// sceRudpSetEventHandler).
//
// Both were unregistered, so the dispatcher answered `0`. They are fire-and-forget setup with
// no out-parameters: enabling an IO thread nobody schedules and registering a handler nothing
// ever calls. prosper runs no network IO thread and delivers no RUDP events, so both are
// acknowledged and nothing is stored. Signatures from the stub interface; NIDs via nid_hash
// (no firmware entries for this library). The Init entry point keeps its portps5-internal
// placeholder name and stays out: its Sony name is not evidenced anywhere.
// CONFIDENCE: MED on the arities (single secondary source), HIGH that acknowledgement is safe
// here (no outputs exist to lie about).
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

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
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    return 0;
}

void register_rudp_hle() {
    Hle::register_fn("6PBNpsgyaxw", (HleFn)rudp_enable_io_thread, "sceRudpEnableInternalIOThread");
    Hle::register_fn("SUEVes8gvmw", (HleFn)rudp_set_event_handler, "sceRudpSetEventHandler");
}

}  // namespace prosper
