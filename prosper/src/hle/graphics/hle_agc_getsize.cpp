// hle_agc_getsize.cpp -- the sceAgc*GetSize queries whose answer is a fixed size and that prosper had
// no answer for.
//
// A guest reserves command-buffer space with sceAgc*GetSize before it calls the matching builder, so
// an unanswered query (the dispatcher's generic stub returns 0) makes it reserve nothing -- the failure
// docs/gpu/AGC_PACKET_SIZES.md records for #1137 and #1748. Each size below is what the console's own
// libSceAgc returned for that query (tests/data/console_oracle/agc.golden.tsv, measured with the
// console oracle), so it is the reservation a real guest makes, not a count read off a PM4 layout.
//
// Only queries with NO builder in prosper are answered here. A query whose builder exists keeps the
// size of prosper's own emission in hle_agc.cpp: answering the console's smaller number there would
// under-reserve for a builder that writes more. Those differences are listed in known_gaps.tsv.
// Each registration is a literal call so the inventory tools (tools/re/hle_handler_map.py,
// tools/progress) see every NID.
#include "hle/dispatch/dispatch.hpp"

#include <cstdint>

namespace prosper {

#define HLE(name)                                                                                  \
    static PROSPER_SYSV_ABI uint64_t name(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,        \
                                          uint64_t)

// One handler per distinct size, in bytes.
HLE(size_8) {
    return 8;
}
HLE(size_16) {
    return 16;
}
HLE(size_20) {
    return 20;
}
HLE(size_24) {
    return 24;
}
HLE(size_32) {
    return 32;
}
HLE(size_36) {
    return 36;
}
HLE(size_44) {
    return 44;
}
HLE(size_56) {
    return 56;
}
HLE(size_60) {
    return 60;
}
HLE(size_64) {
    return 64;
}

void register_agc_getsize_hle() {
    Hle::register_fn("hcIxS8pmXF4", (HleFn)size_44, "sceAgcAcbAtomicGdsGetSize");
    Hle::register_fn("da1Sm8-QDoU", (HleFn)size_36, "sceAgcAcbAtomicMemGetSize");
    Hle::register_fn("ozKzBP4aki4", (HleFn)size_20, "sceAgcAcbCondExecGetSize");
    Hle::register_fn("CbQh3DKMSno", (HleFn)size_24, "sceAgcAcbCopyDataGetSize");
    Hle::register_fn("b-oySn+G2tE", (HleFn)size_16, "sceAgcAcbJumpGetSize");
    Hle::register_fn("eCjKaqeeQ5s", (HleFn)size_20, "sceAgcAcbPrimeUtcl2GetSize");
    Hle::register_fn("F8NLhWvFemI", (HleFn)size_32, "sceAgcAcbQueueEndOfShaderActionGetSize");
    Hle::register_fn("0ZOG0jc9nRg", (HleFn)size_8, "sceAgcAcbRewindGetSize");
    Hle::register_fn("idlaArvdXEs", (HleFn)size_56, "sceAgcAcbWaitOnAddressGetSize");
    Hle::register_fn("FuVbkyKlf+s", (HleFn)size_36, "sceAgcCbCondWriteGetSize");
    Hle::register_fn("1tB0xkLNjcw", (HleFn)size_44, "sceAgcDcbAtomicGdsGetSize");
    Hle::register_fn("oz6zQq1JwCE", (HleFn)size_36, "sceAgcDcbAtomicMemGetSize");
    Hle::register_fn("ms1xVoZ-Vwc", (HleFn)size_16, "sceAgcDcbBeginOcclusionQueryGetSize");
    Hle::register_fn("ou16V5hh5sg", (HleFn)size_20, "sceAgcDcbCondExecGetSize");
    Hle::register_fn("H6vHS5cidSA", (HleFn)size_20, "sceAgcDcbContextStateOpGetSize");
    Hle::register_fn("b5u0Jzm8TF8", (HleFn)size_24, "sceAgcDcbCopyDataGetSize");
    Hle::register_fn("r98I08t+LOg", (HleFn)size_64, "sceAgcDcbDrawIndexIndirectMultiGetSize");
    Hle::register_fn("mR9j7+SfM34", (HleFn)size_60, "sceAgcDcbDrawIndexMultiInstancedGetSize");
    Hle::register_fn("pYoKs3lPy88", (HleFn)size_64, "sceAgcDcbDrawIndirectMultiGetSize");
    Hle::register_fn("P1CugZ99Uzc", (HleFn)size_16, "sceAgcDcbEndOcclusionQueryGetSize");
    Hle::register_fn("rUuVjyR+Rd4", (HleFn)size_20, "sceAgcDcbGetLodStatsGetSize");
    Hle::register_fn("KjPeVduz6jU", (HleFn)size_20, "sceAgcDcbPrimeUtcl2GetSize");
    Hle::register_fn("zg6u-N6Otxs", (HleFn)size_32, "sceAgcDcbQueueEndOfShaderActionGetSize");
    Hle::register_fn("9S4noWrUI0s", (HleFn)size_16, "sceAgcDcbSetBaseDispatchIndirectArgsGetSize");
    Hle::register_fn("MMlmJAL7N5w", (HleFn)size_16, "sceAgcDcbSetBaseDrawIndirectArgsGetSize");
    Hle::register_fn("yheJGN-ay+A", (HleFn)size_16, "sceAgcDcbSetBoolPredicationEnableGetSize");
    Hle::register_fn("AFIh8SQkYlQ", (HleFn)size_16, "sceAgcDcbSetIndexIndirectArgsGetSize");
    Hle::register_fn("vLrBL8DQiz8", (HleFn)size_16, "sceAgcDcbSetPredicationDisableGetSize");
    Hle::register_fn("XN+Iuu7XsM8", (HleFn)size_16, "sceAgcDcbSetZPassPredicationEnableGetSize");
    Hle::register_fn("43WJ08sSugE", (HleFn)size_56, "sceAgcDcbWaitOnAddressGetSize");
}

}   // namespace prosper
