#pragma once
#include <cstdint>

namespace prosper::gpu {
// SBR2 is a distinct shared original-PC demand-window plane. It does not reinterpret WAT1's
// complete-buffer words, and its payload never lives inside an individual wave input span.
// Placement/site metadata is the immutable checked host producer's wire output. GPU descriptor,
// site and bounds checks are not a second authenticity authority for arbitrary bounds-valid
// metadata rerouting. Private malformed-wire calibrations must retain that narrower scope.
inline constexpr uint32_t kFragmentScalarBankBinding = 5;
inline constexpr uint32_t kFragmentScalarBankMagic = 0x32524253u;
inline constexpr uint32_t kFragmentScalarBankVersion = 1;
inline constexpr uint32_t kFragmentScalarBankHeaderWords = 8;
inline constexpr uint32_t kFragmentScalarBankSiteWords = 14;
enum FragmentScalarBankHeaderWord : uint32_t {
    ScalarBankMagic,
    ScalarBankVersion,
    ScalarBankTotalWords,
    ScalarBankSiteCount,
    ScalarBankPayloadBegin,
    ScalarBankPayloadWords,
    ScalarBankReserved0,
    ScalarBankReserved1,
};
enum FragmentScalarBankSiteWord : uint32_t {
    ScalarBankPc,
    ScalarBankOpcode,
    ScalarBankLoadWords,
    ScalarBankByteOffset,
    ScalarBankDeclaredLo,
    ScalarBankDeclaredHi,
    ScalarBankDescriptor0,
    ScalarBankDescriptor1,
    ScalarBankDescriptor2,
    ScalarBankDescriptor3,
    ScalarBankIntervalBase,
    ScalarBankIntervalWords,
    ScalarBankIntervalByteOffset,
    ScalarBankSiteReserved,
};
} // namespace prosper::gpu
