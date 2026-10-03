// APR completion-event dialects: how a binding's tag (the `data` argument of the
// sceAmprCommandBufferWriteKernelEventQueue bind, NID H896Pt-yB4I / o67gODLFpls) must be delivered.
//
// The bind call is (cb, equeue, id, data, ...). Three guest consumers share it and need different
// completion semantics, and NOTHING in the call says which one is calling: no argument other than
// `id` and `data` is read by this project or by either secondary implementation consulted
// (verification only), and the observed a4/a5 values (0, 7|0xf) were not shown to track the
// consumer. So the dialect is derived from (id, data) alone, ONCE, at bind time, by
// classify_apr_dialect below -- the single place the heuristic lives -- and stored on the binding.
// Delivery then switches on the stored enum, never on a tag magnitude.
#pragma once
#include <cstdint>

namespace prosper {

enum class AprDialect : uint8_t {
    // UE4 FAPREventQueueListener (#208): id = 0x74fe+ring, tag = (ring<<58)|counter, dense from
    // 1000. Posted as the ring's high-water mark and coalesced ("completed up to").
    Counter,
    // IoDispatcher (id 0, #210) and Black Flag Resynced (id 1): tag is the guest's request pointer,
    // read back through event.data. Delivered exactly, one queued event per submit, in order.
    RequestPointer,
    // CRI ADX2 (id != 0, literal zero tag, waiter tests only the event ident): every completion is
    // a discrete count. Posted individually, never coalesced.
    ConstantZero,
};

// Evidence for each rule: hle_kernel_time.cpp (prosper_eq_post_apr_token) and
// tests/hle/test_apr_equeue_completion.cpp. CONFIDENCE: MED -- the id/tag partition holds for every
// capture so far (UE4/IoStore, Tales of Graces f, Sonic Origins, Black Flag) but it is inferred,
// not declared by the guest; a Counter consumer that bound a tag >= 2^32 would be misread.
constexpr AprDialect classify_apr_dialect(int64_t id, uint64_t tag) {
    if (id == 0) return AprDialect::RequestPointer;
    const uint64_t cnt = tag & ((1ull << 58) - 1);
    if (cnt == 0) return AprDialect::ConstantZero;   // the listener's counters start at 1000
    if (cnt >= (1ull << 32)) return AprDialect::RequestPointer;   // heap pointer, not a counter
    return AprDialect::Counter;
}

}   // namespace prosper
