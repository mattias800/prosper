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
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <cstdint>

namespace prosper {

namespace {

template <uint32_t Bytes>
PROSPER_SYSV_ABI uint64_t fixed_size(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) {
    return Bytes;
}

struct FixedSize {
    const char* nid;
    HleFn fn;
    const char* name;
};

const FixedSize kFixedSizes[] = {
    {"hcIxS8pmXF4", &fixed_size<44>, "sceAgcAcbAtomicGdsGetSize"},
    {"da1Sm8-QDoU", &fixed_size<36>, "sceAgcAcbAtomicMemGetSize"},
    {"ozKzBP4aki4", &fixed_size<20>, "sceAgcAcbCondExecGetSize"},
    {"CbQh3DKMSno", &fixed_size<24>, "sceAgcAcbCopyDataGetSize"},
    {"b-oySn+G2tE", &fixed_size<16>, "sceAgcAcbJumpGetSize"},
    {"eCjKaqeeQ5s", &fixed_size<20>, "sceAgcAcbPrimeUtcl2GetSize"},
    {"F8NLhWvFemI", &fixed_size<32>, "sceAgcAcbQueueEndOfShaderActionGetSize"},
    {"0ZOG0jc9nRg", &fixed_size<8>, "sceAgcAcbRewindGetSize"},
    {"idlaArvdXEs", &fixed_size<56>, "sceAgcAcbWaitOnAddressGetSize"},
    {"FuVbkyKlf+s", &fixed_size<36>, "sceAgcCbCondWriteGetSize"},
    {"1tB0xkLNjcw", &fixed_size<44>, "sceAgcDcbAtomicGdsGetSize"},
    {"oz6zQq1JwCE", &fixed_size<36>, "sceAgcDcbAtomicMemGetSize"},
    {"ms1xVoZ-Vwc", &fixed_size<16>, "sceAgcDcbBeginOcclusionQueryGetSize"},
    {"ou16V5hh5sg", &fixed_size<20>, "sceAgcDcbCondExecGetSize"},
    {"H6vHS5cidSA", &fixed_size<20>, "sceAgcDcbContextStateOpGetSize"},
    {"b5u0Jzm8TF8", &fixed_size<24>, "sceAgcDcbCopyDataGetSize"},
    {"r98I08t+LOg", &fixed_size<64>, "sceAgcDcbDrawIndexIndirectMultiGetSize"},
    {"mR9j7+SfM34", &fixed_size<60>, "sceAgcDcbDrawIndexMultiInstancedGetSize"},
    {"pYoKs3lPy88", &fixed_size<64>, "sceAgcDcbDrawIndirectMultiGetSize"},
    {"P1CugZ99Uzc", &fixed_size<16>, "sceAgcDcbEndOcclusionQueryGetSize"},
    {"rUuVjyR+Rd4", &fixed_size<20>, "sceAgcDcbGetLodStatsGetSize"},
    {"KjPeVduz6jU", &fixed_size<20>, "sceAgcDcbPrimeUtcl2GetSize"},
    {"zg6u-N6Otxs", &fixed_size<32>, "sceAgcDcbQueueEndOfShaderActionGetSize"},
    {"9S4noWrUI0s", &fixed_size<16>, "sceAgcDcbSetBaseDispatchIndirectArgsGetSize"},
    {"MMlmJAL7N5w", &fixed_size<16>, "sceAgcDcbSetBaseDrawIndirectArgsGetSize"},
    {"yheJGN-ay+A", &fixed_size<16>, "sceAgcDcbSetBoolPredicationEnableGetSize"},
    {"AFIh8SQkYlQ", &fixed_size<16>, "sceAgcDcbSetIndexIndirectArgsGetSize"},
    {"vLrBL8DQiz8", &fixed_size<16>, "sceAgcDcbSetPredicationDisableGetSize"},
    {"XN+Iuu7XsM8", &fixed_size<16>, "sceAgcDcbSetZPassPredicationEnableGetSize"},
    {"43WJ08sSugE", &fixed_size<56>, "sceAgcDcbWaitOnAddressGetSize"},
};

}  // namespace

void register_agc_getsize_hle() {
    for (const FixedSize& s : kFixedSizes) Hle::register_fn(s.nid, s.fn, s.name);
}

}  // namespace prosper
