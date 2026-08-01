// SCE_KERNEL_ERROR_* — the PS5 libkernel error encoding, which is `0x80020000 | errno`.
//
//   ############################################################################################
//   #  The low byte is a **FreeBSD** errno. It is NOT this host's <errno.h> value.             #
//   #  Never write `0x80020000 | errno` with a host errno, and never pick the low byte by      #
//   #  looking a name up in the Linux (or Windows) headers on your desk.                       #
//   ############################################################################################
//
// The PS5 kernel is FreeBSD-derived, so the guest interprets that byte with FreeBSD's numbering.
// Linux and FreeBSD agree on 1..34 and diverge from 35 upward — which is exactly why a wrong
// constant survives review. Every value a developer double-checks by reflex (EINVAL 22, ENOENT 2,
// EFAULT 14, ENOMEM 12) is identical on both, so the mistake only bites on the values nobody
// second-guesses:
//
//      value  |  FreeBSD (what the guest reads)  |  Linux (where the wrong value came from)
//      -------+----------------------------------+------------------------------------------
//         11  |  EDEADLK                         |  EAGAIN
//         35  |  EAGAIN / EWOULDBLOCK            |  EDEADLK
//         38  |  ENOTSOCK                        |  ENOSYS
//         60  |  ETIMEDOUT                       |  ENOTCONN
//         78  |  ENOSYS                          |  EBADFD
//        110  |  (unallocated)                   |  ETIMEDOUT
//
// The failure is silent and easy to misdiagnose: a guest that branches on the errno sees a
// plausible-but-different condition. `EAGAIN` is the worst of them, because it is a *retry* hint —
// a guest handed EAGAIN where the emulator meant "deadlock" or "unsupported" can loop forever on a
// condition that will never change, and the resulting hang looks like a scheduling bug rather than
// a wrong constant. This is #1612.
//
// Use these helpers instead of a literal:
//   * a fixed condition the emulator itself decides  -> sce_kernel_error(FreeBsdErrno::EDeadlk)
//   * a host errno from a failed libc/pthread call   -> sce_error_from_host_errno(e)
//   * a Win32 GetLastError() code                    -> sce_error_from_win32(e)
//
// CONFIDENCE: HIGH on the numbering — it is FreeBSD's <sys/errno.h>, unchanged since 4.2BSD for the
// low range, and corroborated inside prosper by the values that were already right: ETIMEDOUT is
// encoded as 0x8002003c (60) in a dozen places, which is the BSD value (Linux ETIMEDOUT is 110),
// and hle_ult.cpp encodes ENOSYS as 0x8002004E (78), also the BSD value.

#pragma once

#include <cerrno>
#include <cstdint>

namespace prosper::hle {

// FreeBSD <sys/errno.h> numbering. Deliberately spelled as an enum rather than macros so it can
// never be confused with the host's EINVAL/EAGAIN/... macros in the same translation unit.
enum class FreeBsdErrno : uint32_t {
    EPerm           = 1,
    ENoEnt          = 2,
    ESrch           = 3,
    EIntr           = 4,
    EIo             = 5,
    ENxIo           = 6,
    E2Big           = 7,
    ENoExec         = 8,
    EBadF           = 9,
    EChild          = 10,
    EDeadlk         = 11,   // Linux calls 11 EAGAIN
    ENoMem          = 12,
    EAcces          = 13,
    EFault          = 14,
    EBusy           = 16,
    EExist          = 17,
    EXDev           = 18,
    ENoDev          = 19,
    ENotDir         = 20,
    EIsDir          = 21,
    EInval          = 22,
    ENFile          = 23,
    EMFile          = 24,
    ENotTty         = 25,
    ETxtBsy         = 26,
    EFBig           = 27,
    ENoSpc          = 28,
    ESPipe          = 29,
    ERoFs           = 30,
    EMLink          = 31,
    EPipe           = 32,
    EDom            = 33,
    ERange          = 34,
    EAgain          = 35,   // Linux calls 35 EDEADLK
    EInProgress     = 36,
    EAlready        = 37,
    ENotSock        = 38,   // Linux calls 38 ENOSYS
    EDestAddrReq    = 39,
    EMsgSize        = 40,
    EOpNotSupp      = 45,
    EAddrInUse      = 48,
    ENetDown        = 50,
    ENetUnreach     = 51,
    EConnAborted    = 53,
    EConnReset      = 54,
    ENoBufs         = 55,
    EIsConn         = 56,
    ENotConn        = 57,
    ETimedOut       = 60,   // Linux calls 60 ENOTCONN; Linux ETIMEDOUT is 110
    EConnRefused    = 61,
    ELoop           = 62,
    ENameTooLong    = 63,
    EHostUnreach    = 65,
    ENotEmpty       = 66,
    EDQuot          = 69,
    EStale          = 70,
    ENoLck          = 77,
    ENoSys          = 78,   // Linux calls 78 EBADFD
    EIdRm           = 82,
    ENoMsg          = 83,
    EOverflow       = 84,
    ECanceled       = 85,
    EIlSeq          = 86,
    ENoAttr         = 87,
    EBadMsg         = 89,
    EMultiHop       = 90,
    ENoLink         = 91,
    EProto          = 92,
    ENotRecoverable = 95,
    EOwnerDead      = 96,
};

// The libkernel encoding. Guests test the result as a signed 32-bit value, where 0x8002xxxx is
// negative, so a nonzero return reads as failure.
constexpr uint64_t sce_kernel_error(FreeBsdErrno e) {
    return 0x80020000ull | static_cast<uint64_t>(e);
}

// A handful of names used often enough to be worth spelling out at the call site.
inline constexpr uint64_t kSceKernelErrorEPERM     = sce_kernel_error(FreeBsdErrno::EPerm);
inline constexpr uint64_t kSceKernelErrorENOENT    = sce_kernel_error(FreeBsdErrno::ENoEnt);
inline constexpr uint64_t kSceKernelErrorESRCH     = sce_kernel_error(FreeBsdErrno::ESrch);
inline constexpr uint64_t kSceKernelErrorEBADF     = sce_kernel_error(FreeBsdErrno::EBadF);
inline constexpr uint64_t kSceKernelErrorEDEADLK   = sce_kernel_error(FreeBsdErrno::EDeadlk);
inline constexpr uint64_t kSceKernelErrorENOMEM    = sce_kernel_error(FreeBsdErrno::ENoMem);
inline constexpr uint64_t kSceKernelErrorEACCES    = sce_kernel_error(FreeBsdErrno::EAcces);
inline constexpr uint64_t kSceKernelErrorEFAULT    = sce_kernel_error(FreeBsdErrno::EFault);
inline constexpr uint64_t kSceKernelErrorEBUSY     = sce_kernel_error(FreeBsdErrno::EBusy);
inline constexpr uint64_t kSceKernelErrorEINVAL    = sce_kernel_error(FreeBsdErrno::EInval);
inline constexpr uint64_t kSceKernelErrorEAGAIN    = sce_kernel_error(FreeBsdErrno::EAgain);
inline constexpr uint64_t kSceKernelErrorETIMEDOUT = sce_kernel_error(FreeBsdErrno::ETimedOut);
inline constexpr uint64_t kSceKernelErrorENOSYS    = sce_kernel_error(FreeBsdErrno::ENoSys);

// Translate one of THIS host's errno values to the FreeBSD number the guest expects.
//
// The table is keyed on the host's own macro NAMES, never on numbers, so it is correct wherever it
// is compiled: on a BSD-numbered host (macOS) most entries are identities, on Linux the divergent
// ones are remapped, and a host that lacks a macro entirely just drops that row. An unrecognized
// errno falls back to EInval rather than being passed through raw, because passing it through is
// precisely the defect this function exists to prevent — a raw Linux 38 would reach the guest as
// ENOTSOCK.
//
// `fallback` is what an unrecognized errno becomes; callers that already had a considered answer
// keep it (the file layer uses EIo, "an unclassified host failure", rather than claiming EInval).
//
// CONFIDENCE: HIGH on the mapping; the only judgement is the fallback, and a wrong-but-in-range
// EINVAL is strictly safer than a number that means something unrelated.
inline FreeBsdErrno freebsd_errno_from_host(int host_errno,
                                            FreeBsdErrno fallback = FreeBsdErrno::EInval) {
    switch (host_errno) {
#define PROSPER_ERRNO_ROW(host_macro, freebsd_name) \
    case host_macro: return FreeBsdErrno::freebsd_name;
        PROSPER_ERRNO_ROW(EPERM,        EPerm)
        PROSPER_ERRNO_ROW(ENOENT,       ENoEnt)
        PROSPER_ERRNO_ROW(ESRCH,        ESrch)
        PROSPER_ERRNO_ROW(EINTR,        EIntr)
        PROSPER_ERRNO_ROW(EIO,          EIo)
        PROSPER_ERRNO_ROW(ENXIO,        ENxIo)
        PROSPER_ERRNO_ROW(E2BIG,        E2Big)
        PROSPER_ERRNO_ROW(ENOEXEC,      ENoExec)
        PROSPER_ERRNO_ROW(EBADF,        EBadF)
        PROSPER_ERRNO_ROW(ECHILD,       EChild)
        PROSPER_ERRNO_ROW(EDEADLK,      EDeadlk)      // Linux 35 -> FreeBSD 11
        PROSPER_ERRNO_ROW(ENOMEM,       ENoMem)
        PROSPER_ERRNO_ROW(EACCES,       EAcces)
        PROSPER_ERRNO_ROW(EFAULT,       EFault)
        PROSPER_ERRNO_ROW(EBUSY,        EBusy)
        PROSPER_ERRNO_ROW(EEXIST,       EExist)
        PROSPER_ERRNO_ROW(EXDEV,        EXDev)
        PROSPER_ERRNO_ROW(ENODEV,       ENoDev)
        PROSPER_ERRNO_ROW(ENOTDIR,      ENotDir)
        PROSPER_ERRNO_ROW(EISDIR,       EIsDir)
        PROSPER_ERRNO_ROW(EINVAL,       EInval)
        PROSPER_ERRNO_ROW(ENFILE,       ENFile)
        PROSPER_ERRNO_ROW(EMFILE,       EMFile)
        PROSPER_ERRNO_ROW(ENOTTY,       ENotTty)
#ifdef ETXTBSY
        PROSPER_ERRNO_ROW(ETXTBSY,      ETxtBsy)
#endif
        PROSPER_ERRNO_ROW(EFBIG,        EFBig)
        PROSPER_ERRNO_ROW(ENOSPC,       ENoSpc)
        PROSPER_ERRNO_ROW(ESPIPE,       ESPipe)
        PROSPER_ERRNO_ROW(EROFS,        ERoFs)
        PROSPER_ERRNO_ROW(EMLINK,       EMLink)
        PROSPER_ERRNO_ROW(EPIPE,        EPipe)
        PROSPER_ERRNO_ROW(EDOM,         EDom)
        PROSPER_ERRNO_ROW(ERANGE,       ERange)
        PROSPER_ERRNO_ROW(EAGAIN,       EAgain)       // Linux 11 -> FreeBSD 35
        PROSPER_ERRNO_ROW(EINPROGRESS,  EInProgress)
        PROSPER_ERRNO_ROW(EALREADY,     EAlready)
        PROSPER_ERRNO_ROW(ENOTSOCK,     ENotSock)
        PROSPER_ERRNO_ROW(EDESTADDRREQ, EDestAddrReq)
        PROSPER_ERRNO_ROW(EMSGSIZE,     EMsgSize)
        PROSPER_ERRNO_ROW(EOPNOTSUPP,   EOpNotSupp)
        PROSPER_ERRNO_ROW(EADDRINUSE,   EAddrInUse)
        PROSPER_ERRNO_ROW(ENETDOWN,     ENetDown)
        PROSPER_ERRNO_ROW(ENETUNREACH,  ENetUnreach)
        PROSPER_ERRNO_ROW(ECONNABORTED, EConnAborted)
        PROSPER_ERRNO_ROW(ECONNRESET,   EConnReset)
        PROSPER_ERRNO_ROW(ENOBUFS,      ENoBufs)
        PROSPER_ERRNO_ROW(EISCONN,      EIsConn)
        PROSPER_ERRNO_ROW(ENOTCONN,     ENotConn)
        PROSPER_ERRNO_ROW(ETIMEDOUT,    ETimedOut)    // Linux 110 -> FreeBSD 60
        PROSPER_ERRNO_ROW(ECONNREFUSED, EConnRefused)
        PROSPER_ERRNO_ROW(ELOOP,        ELoop)        // Linux 40 -> FreeBSD 62
        PROSPER_ERRNO_ROW(ENAMETOOLONG, ENameTooLong) // Linux 36 -> FreeBSD 63
        PROSPER_ERRNO_ROW(EHOSTUNREACH, EHostUnreach)
        PROSPER_ERRNO_ROW(ENOTEMPTY,    ENotEmpty)    // Linux 39 -> FreeBSD 66
        PROSPER_ERRNO_ROW(EDQUOT,       EDQuot)
        PROSPER_ERRNO_ROW(ESTALE,       EStale)
        PROSPER_ERRNO_ROW(ENOLCK,       ENoLck)       // Linux 37 -> FreeBSD 77
        PROSPER_ERRNO_ROW(ENOSYS,       ENoSys)       // Linux 38 -> FreeBSD 78
        PROSPER_ERRNO_ROW(EIDRM,        EIdRm)
        PROSPER_ERRNO_ROW(ENOMSG,       ENoMsg)
        PROSPER_ERRNO_ROW(EOVERFLOW,    EOverflow)
        PROSPER_ERRNO_ROW(ECANCELED,    ECanceled)
        PROSPER_ERRNO_ROW(EILSEQ,       EIlSeq)
        PROSPER_ERRNO_ROW(EBADMSG,      EBadMsg)
        PROSPER_ERRNO_ROW(EPROTO,       EProto)
        PROSPER_ERRNO_ROW(EMULTIHOP,    EMultiHop)
        PROSPER_ERRNO_ROW(ENOLINK,      ENoLink)
#ifdef ENOTRECOVERABLE
        PROSPER_ERRNO_ROW(ENOTRECOVERABLE, ENotRecoverable)
#endif
#ifdef EOWNERDEAD
        PROSPER_ERRNO_ROW(EOWNERDEAD,   EOwnerDead)
#endif
#undef PROSPER_ERRNO_ROW
    default: return fallback;
    }
}

// The inverse, for the few places that must hand a guest-facing SCE error back to host libc (the
// POSIX shims that answer with -1 and set the host's errno). Without this the round trip would
// re-introduce the bug in the opposite direction.
inline int host_errno_from_freebsd(FreeBsdErrno e) {
    switch (e) {
#define PROSPER_ERRNO_ROW(host_macro, freebsd_name) \
    case FreeBsdErrno::freebsd_name: return host_macro;
        PROSPER_ERRNO_ROW(EPERM,        EPerm)
        PROSPER_ERRNO_ROW(ENOENT,       ENoEnt)
        PROSPER_ERRNO_ROW(ESRCH,        ESrch)
        PROSPER_ERRNO_ROW(EINTR,        EIntr)
        PROSPER_ERRNO_ROW(EIO,          EIo)
        PROSPER_ERRNO_ROW(ENXIO,        ENxIo)
        PROSPER_ERRNO_ROW(E2BIG,        E2Big)
        PROSPER_ERRNO_ROW(ENOEXEC,      ENoExec)
        PROSPER_ERRNO_ROW(EBADF,        EBadF)
        PROSPER_ERRNO_ROW(ECHILD,       EChild)
        PROSPER_ERRNO_ROW(EDEADLK,      EDeadlk)
        PROSPER_ERRNO_ROW(ENOMEM,       ENoMem)
        PROSPER_ERRNO_ROW(EACCES,       EAcces)
        PROSPER_ERRNO_ROW(EFAULT,       EFault)
        PROSPER_ERRNO_ROW(EBUSY,        EBusy)
        PROSPER_ERRNO_ROW(EEXIST,       EExist)
        PROSPER_ERRNO_ROW(EXDEV,        EXDev)
        PROSPER_ERRNO_ROW(ENODEV,       ENoDev)
        PROSPER_ERRNO_ROW(ENOTDIR,      ENotDir)
        PROSPER_ERRNO_ROW(EISDIR,       EIsDir)
        PROSPER_ERRNO_ROW(EINVAL,       EInval)
        PROSPER_ERRNO_ROW(ENFILE,       ENFile)
        PROSPER_ERRNO_ROW(EMFILE,       EMFile)
        PROSPER_ERRNO_ROW(ENOTTY,       ENotTty)
#ifdef ETXTBSY
        PROSPER_ERRNO_ROW(ETXTBSY,      ETxtBsy)
#endif
        PROSPER_ERRNO_ROW(EFBIG,        EFBig)
        PROSPER_ERRNO_ROW(ENOSPC,       ENoSpc)
        PROSPER_ERRNO_ROW(ESPIPE,       ESPipe)
        PROSPER_ERRNO_ROW(EROFS,        ERoFs)
        PROSPER_ERRNO_ROW(EMLINK,       EMLink)
        PROSPER_ERRNO_ROW(EPIPE,        EPipe)
        PROSPER_ERRNO_ROW(EDOM,         EDom)
        PROSPER_ERRNO_ROW(ERANGE,       ERange)
        PROSPER_ERRNO_ROW(EAGAIN,       EAgain)
        PROSPER_ERRNO_ROW(EINPROGRESS,  EInProgress)
        PROSPER_ERRNO_ROW(EALREADY,     EAlready)
        PROSPER_ERRNO_ROW(ENOTSOCK,     ENotSock)
        PROSPER_ERRNO_ROW(EDESTADDRREQ, EDestAddrReq)
        PROSPER_ERRNO_ROW(EMSGSIZE,     EMsgSize)
        PROSPER_ERRNO_ROW(EOPNOTSUPP,   EOpNotSupp)
        PROSPER_ERRNO_ROW(EADDRINUSE,   EAddrInUse)
        PROSPER_ERRNO_ROW(ENETDOWN,     ENetDown)
        PROSPER_ERRNO_ROW(ENETUNREACH,  ENetUnreach)
        PROSPER_ERRNO_ROW(ECONNABORTED, EConnAborted)
        PROSPER_ERRNO_ROW(ECONNRESET,   EConnReset)
        PROSPER_ERRNO_ROW(ENOBUFS,      ENoBufs)
        PROSPER_ERRNO_ROW(EISCONN,      EIsConn)
        PROSPER_ERRNO_ROW(ENOTCONN,     ENotConn)
        PROSPER_ERRNO_ROW(ETIMEDOUT,    ETimedOut)
        PROSPER_ERRNO_ROW(ECONNREFUSED, EConnRefused)
        PROSPER_ERRNO_ROW(ELOOP,        ELoop)
        PROSPER_ERRNO_ROW(ENAMETOOLONG, ENameTooLong)
        PROSPER_ERRNO_ROW(EHOSTUNREACH, EHostUnreach)
        PROSPER_ERRNO_ROW(ENOTEMPTY,    ENotEmpty)
        PROSPER_ERRNO_ROW(EDQUOT,       EDQuot)
        PROSPER_ERRNO_ROW(ESTALE,       EStale)
        PROSPER_ERRNO_ROW(ENOLCK,       ENoLck)
        PROSPER_ERRNO_ROW(ENOSYS,       ENoSys)
        PROSPER_ERRNO_ROW(EIDRM,        EIdRm)
        PROSPER_ERRNO_ROW(ENOMSG,       ENoMsg)
        PROSPER_ERRNO_ROW(EOVERFLOW,    EOverflow)
        PROSPER_ERRNO_ROW(ECANCELED,    ECanceled)
        PROSPER_ERRNO_ROW(EILSEQ,       EIlSeq)
        PROSPER_ERRNO_ROW(EBADMSG,      EBadMsg)
        PROSPER_ERRNO_ROW(EPROTO,       EProto)
        PROSPER_ERRNO_ROW(EMULTIHOP,    EMultiHop)
        PROSPER_ERRNO_ROW(ENOLINK,      ENoLink)
#ifdef ENOTRECOVERABLE
        PROSPER_ERRNO_ROW(ENOTRECOVERABLE, ENotRecoverable)
#endif
#ifdef EOWNERDEAD
        PROSPER_ERRNO_ROW(EOWNERDEAD,   EOwnerDead)
#endif
#undef PROSPER_ERRNO_ROW
    default: return EINVAL;
    }
}

// Encode a failed host libc/pthread call for the guest.
inline uint64_t sce_error_from_host_errno(int host_errno,
                                          FreeBsdErrno fallback = FreeBsdErrno::EInval) {
    return sce_kernel_error(freebsd_errno_from_host(host_errno, fallback));
}

// Decode the errno out of an encoded SCE error. Returns false when `value` is not in the family.
inline bool sce_kernel_error_errno(uint64_t value, FreeBsdErrno& out) {
    if ((value & ~0xffull) != 0x80020000ull) return false;
    out = static_cast<FreeBsdErrno>(value & 0xffull);
    return true;
}

#if defined(_WIN32)
// Encode a Win32 GetLastError() code for the guest.
//
// A Win32 status is NOT an errno in any numbering, so it cannot be poured into the low byte: the
// codes overlap the errno range while meaning something else (ERROR_ACCESS_DENIED is 5, which the
// guest would read as EIO), and many exceed 255 outright, so masking pushes the value out of the
// errno byte and into an unrelated part of the SCE error space (ERROR_INVALID_ADDRESS, 487, lands
// on 0x800201E7). Map deliberately, and keep the raw code in the emulator log rather than in the
// guest-visible value.
//
// CONFIDENCE: MED — the mapping is the conventional Win32->POSIX correspondence, but which errno a
// real PS5 libkernel would report for these conditions is not observable from a Windows host. The
// fallback is EIO ("an unclassified host failure"), chosen because it is unambiguous: no caller
// treats it as retryable, which is the property that made the original bug harmful.
inline uint64_t sce_error_from_win32(unsigned long win32_error) {
    switch (win32_error) {
    case 5UL:    return sce_kernel_error(FreeBsdErrno::EPerm);    // ERROR_ACCESS_DENIED
    case 6UL:    return sce_kernel_error(FreeBsdErrno::ESrch);    // ERROR_INVALID_HANDLE
    case 8UL:                                                     // ERROR_NOT_ENOUGH_MEMORY
    case 14UL:   return sce_kernel_error(FreeBsdErrno::ENoMem);   // ERROR_OUTOFMEMORY
    case 87UL:   return sce_kernel_error(FreeBsdErrno::EInval);   // ERROR_INVALID_PARAMETER
    case 170UL:  return sce_kernel_error(FreeBsdErrno::EBusy);    // ERROR_BUSY
    case 1450UL: return sce_kernel_error(FreeBsdErrno::ENoMem);   // ERROR_NO_SYSTEM_RESOURCES
    case 1816UL: return sce_kernel_error(FreeBsdErrno::ENoMem);   // ERROR_NOT_ENOUGH_QUOTA
    default:     return sce_kernel_error(FreeBsdErrno::EIo);
    }
}
#endif

}   // namespace prosper::hle
