// SCE_KERNEL_ERROR_* — the PS5 libkernel error encoding, which is `0x80020000 | errno`.
//
//   ############################################################################################
//   #  The low byte is a **FreeBSD** errno. It is NOT this host's <errno.h> value.             #
//   #  Never write `0x80020000 | errno` with a host errno, and never pick the low byte by      #
//   #  looking a name up in the Linux (or Windows) headers on your desk.                       #
//   ############################################################################################
//
// The PS5 kernel is FreeBSD-derived, so the guest interprets that byte with FreeBSD's numbering.
// Linux and FreeBSD agree across most of 1..34 and diverge from 35 upward — which is exactly why a
// wrong constant survives review. Every value a developer double-checks by reflex (EINVAL 22,
// ENOENT 2, EFAULT 14, ENOMEM 12) is identical on both, so the mistake only bites on the values
// nobody second-guesses. Note 11 is INSIDE the "low" range and still differs — it is the headline
// case here, so "anything under 35 is safe" is not a valid shortcut:
//
//      value  |  FreeBSD (what the guest reads)  |  Linux (where the wrong value came from)
//      -------+----------------------------------+------------------------------------------
//         11  |  EDEADLK                         |  EAGAIN
//         35  |  EAGAIN / EWOULDBLOCK            |  EDEADLK
//         38  |  ENOTSOCK                        |  ENOSYS
//         60  |  ETIMEDOUT                       |  ENOSTR
//         78  |  ENOSYS                          |  EREMCHG
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
// ones are remapped, and a host that lacks a macro entirely drops that row — every row carries its
// own #ifdef, because MinGW/UCRT genuinely lacks EDQUOT, ESTALE and EMULTIHOP.
//
// EWOULDBLOCK and ENOTSUP need an inequality guard as well as an #ifdef: they alias EAGAIN and
// EOPNOTSUPP on Linux/macOS (where an unguarded row is a duplicate case label and will not compile)
// but are DISTINCT values on MinGW/UCRT, where omitting the row silently drops them to the
// caller's fallback. An unrecognized
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
#ifdef EPERM
    case EPERM: return FreeBsdErrno::EPerm;
#endif
#ifdef ENOENT
    case ENOENT: return FreeBsdErrno::ENoEnt;
#endif
#ifdef ESRCH
    case ESRCH: return FreeBsdErrno::ESrch;
#endif
#ifdef EINTR
    case EINTR: return FreeBsdErrno::EIntr;
#endif
#ifdef EIO
    case EIO: return FreeBsdErrno::EIo;
#endif
#ifdef ENXIO
    case ENXIO: return FreeBsdErrno::ENxIo;
#endif
#ifdef E2BIG
    case E2BIG: return FreeBsdErrno::E2Big;
#endif
#ifdef ENOEXEC
    case ENOEXEC: return FreeBsdErrno::ENoExec;
#endif
#ifdef EBADF
    case EBADF: return FreeBsdErrno::EBadF;
#endif
#ifdef ECHILD
    case ECHILD: return FreeBsdErrno::EChild;
#endif
#ifdef EDEADLK
    case EDEADLK: return FreeBsdErrno::EDeadlk;
#endif
#ifdef ENOMEM
    case ENOMEM: return FreeBsdErrno::ENoMem;
#endif
#ifdef EACCES
    case EACCES: return FreeBsdErrno::EAcces;
#endif
#ifdef EFAULT
    case EFAULT: return FreeBsdErrno::EFault;
#endif
#ifdef EBUSY
    case EBUSY: return FreeBsdErrno::EBusy;
#endif
#ifdef EEXIST
    case EEXIST: return FreeBsdErrno::EExist;
#endif
#ifdef EXDEV
    case EXDEV: return FreeBsdErrno::EXDev;
#endif
#ifdef ENODEV
    case ENODEV: return FreeBsdErrno::ENoDev;
#endif
#ifdef ENOTDIR
    case ENOTDIR: return FreeBsdErrno::ENotDir;
#endif
#ifdef EISDIR
    case EISDIR: return FreeBsdErrno::EIsDir;
#endif
#ifdef EINVAL
    case EINVAL: return FreeBsdErrno::EInval;
#endif
#ifdef ENFILE
    case ENFILE: return FreeBsdErrno::ENFile;
#endif
#ifdef EMFILE
    case EMFILE: return FreeBsdErrno::EMFile;
#endif
#ifdef ENOTTY
    case ENOTTY: return FreeBsdErrno::ENotTty;
#endif
#ifdef ETXTBSY
    case ETXTBSY: return FreeBsdErrno::ETxtBsy;
#endif
#ifdef EFBIG
    case EFBIG: return FreeBsdErrno::EFBig;
#endif
#ifdef ENOSPC
    case ENOSPC: return FreeBsdErrno::ENoSpc;
#endif
#ifdef ESPIPE
    case ESPIPE: return FreeBsdErrno::ESPipe;
#endif
#ifdef EROFS
    case EROFS: return FreeBsdErrno::ERoFs;
#endif
#ifdef EMLINK
    case EMLINK: return FreeBsdErrno::EMLink;
#endif
#ifdef EPIPE
    case EPIPE: return FreeBsdErrno::EPipe;
#endif
#ifdef EDOM
    case EDOM: return FreeBsdErrno::EDom;
#endif
#ifdef ERANGE
    case ERANGE: return FreeBsdErrno::ERange;
#endif
#ifdef EAGAIN
    case EAGAIN: return FreeBsdErrno::EAgain;
#endif
#ifdef EINPROGRESS
    case EINPROGRESS: return FreeBsdErrno::EInProgress;
#endif
#ifdef EALREADY
    case EALREADY: return FreeBsdErrno::EAlready;
#endif
#ifdef ENOTSOCK
    case ENOTSOCK: return FreeBsdErrno::ENotSock;
#endif
#ifdef EDESTADDRREQ
    case EDESTADDRREQ: return FreeBsdErrno::EDestAddrReq;
#endif
#ifdef EMSGSIZE
    case EMSGSIZE: return FreeBsdErrno::EMsgSize;
#endif
#ifdef EOPNOTSUPP
    case EOPNOTSUPP: return FreeBsdErrno::EOpNotSupp;
#endif
#ifdef EADDRINUSE
    case EADDRINUSE: return FreeBsdErrno::EAddrInUse;
#endif
#ifdef ENETDOWN
    case ENETDOWN: return FreeBsdErrno::ENetDown;
#endif
#ifdef ENETUNREACH
    case ENETUNREACH: return FreeBsdErrno::ENetUnreach;
#endif
#ifdef ECONNABORTED
    case ECONNABORTED: return FreeBsdErrno::EConnAborted;
#endif
#ifdef ECONNRESET
    case ECONNRESET: return FreeBsdErrno::EConnReset;
#endif
#ifdef ENOBUFS
    case ENOBUFS: return FreeBsdErrno::ENoBufs;
#endif
#ifdef EISCONN
    case EISCONN: return FreeBsdErrno::EIsConn;
#endif
#ifdef ENOTCONN
    case ENOTCONN: return FreeBsdErrno::ENotConn;
#endif
#ifdef ETIMEDOUT
    case ETIMEDOUT: return FreeBsdErrno::ETimedOut;
#endif
#ifdef ECONNREFUSED
    case ECONNREFUSED: return FreeBsdErrno::EConnRefused;
#endif
#ifdef ELOOP
    case ELOOP: return FreeBsdErrno::ELoop;
#endif
#ifdef ENAMETOOLONG
    case ENAMETOOLONG: return FreeBsdErrno::ENameTooLong;
#endif
#ifdef EHOSTUNREACH
    case EHOSTUNREACH: return FreeBsdErrno::EHostUnreach;
#endif
#ifdef ENOTEMPTY
    case ENOTEMPTY: return FreeBsdErrno::ENotEmpty;
#endif
#ifdef EDQUOT
    case EDQUOT: return FreeBsdErrno::EDQuot;
#endif
#ifdef ESTALE
    case ESTALE: return FreeBsdErrno::EStale;
#endif
#ifdef ENOLCK
    case ENOLCK: return FreeBsdErrno::ENoLck;
#endif
#ifdef ENOSYS
    case ENOSYS: return FreeBsdErrno::ENoSys;
#endif
#ifdef EIDRM
    case EIDRM: return FreeBsdErrno::EIdRm;
#endif
#ifdef ENOMSG
    case ENOMSG: return FreeBsdErrno::ENoMsg;
#endif
#ifdef EOVERFLOW
    case EOVERFLOW: return FreeBsdErrno::EOverflow;
#endif
#ifdef ECANCELED
    case ECANCELED: return FreeBsdErrno::ECanceled;
#endif
#ifdef EILSEQ
    case EILSEQ: return FreeBsdErrno::EIlSeq;
#endif
#ifdef EBADMSG
    case EBADMSG: return FreeBsdErrno::EBadMsg;
#endif
#ifdef EPROTO
    case EPROTO: return FreeBsdErrno::EProto;
#endif
#ifdef EMULTIHOP
    case EMULTIHOP: return FreeBsdErrno::EMultiHop;
#endif
#ifdef ENOLINK
    case ENOLINK: return FreeBsdErrno::ENoLink;
#endif
#ifdef ENOTRECOVERABLE
    case ENOTRECOVERABLE: return FreeBsdErrno::ENotRecoverable;
#endif
#ifdef EOWNERDEAD
    case EOWNERDEAD: return FreeBsdErrno::EOwnerDead;
#endif
#if defined(EWOULDBLOCK) && defined(EAGAIN) && EWOULDBLOCK != EAGAIN
    case EWOULDBLOCK: return FreeBsdErrno::EAgain;
#endif
#if defined(ENOTSUP) && defined(EOPNOTSUPP) && ENOTSUP != EOPNOTSUPP
    case ENOTSUP: return FreeBsdErrno::EOpNotSupp;
#endif
    default: return fallback;
    }
}

// The inverse, for the few places that must hand a guest-facing SCE error back to host libc (the
// POSIX shims that answer with -1 and set the host's errno). Without this the round trip would
// re-introduce the bug in the opposite direction.
inline int host_errno_from_freebsd(FreeBsdErrno e) {
    switch (e) {
#ifdef EPERM
    case FreeBsdErrno::EPerm: return EPERM;
#endif
#ifdef ENOENT
    case FreeBsdErrno::ENoEnt: return ENOENT;
#endif
#ifdef ESRCH
    case FreeBsdErrno::ESrch: return ESRCH;
#endif
#ifdef EINTR
    case FreeBsdErrno::EIntr: return EINTR;
#endif
#ifdef EIO
    case FreeBsdErrno::EIo: return EIO;
#endif
#ifdef ENXIO
    case FreeBsdErrno::ENxIo: return ENXIO;
#endif
#ifdef E2BIG
    case FreeBsdErrno::E2Big: return E2BIG;
#endif
#ifdef ENOEXEC
    case FreeBsdErrno::ENoExec: return ENOEXEC;
#endif
#ifdef EBADF
    case FreeBsdErrno::EBadF: return EBADF;
#endif
#ifdef ECHILD
    case FreeBsdErrno::EChild: return ECHILD;
#endif
#ifdef EDEADLK
    case FreeBsdErrno::EDeadlk: return EDEADLK;
#endif
#ifdef ENOMEM
    case FreeBsdErrno::ENoMem: return ENOMEM;
#endif
#ifdef EACCES
    case FreeBsdErrno::EAcces: return EACCES;
#endif
#ifdef EFAULT
    case FreeBsdErrno::EFault: return EFAULT;
#endif
#ifdef EBUSY
    case FreeBsdErrno::EBusy: return EBUSY;
#endif
#ifdef EEXIST
    case FreeBsdErrno::EExist: return EEXIST;
#endif
#ifdef EXDEV
    case FreeBsdErrno::EXDev: return EXDEV;
#endif
#ifdef ENODEV
    case FreeBsdErrno::ENoDev: return ENODEV;
#endif
#ifdef ENOTDIR
    case FreeBsdErrno::ENotDir: return ENOTDIR;
#endif
#ifdef EISDIR
    case FreeBsdErrno::EIsDir: return EISDIR;
#endif
#ifdef EINVAL
    case FreeBsdErrno::EInval: return EINVAL;
#endif
#ifdef ENFILE
    case FreeBsdErrno::ENFile: return ENFILE;
#endif
#ifdef EMFILE
    case FreeBsdErrno::EMFile: return EMFILE;
#endif
#ifdef ENOTTY
    case FreeBsdErrno::ENotTty: return ENOTTY;
#endif
#ifdef ETXTBSY
    case FreeBsdErrno::ETxtBsy: return ETXTBSY;
#endif
#ifdef EFBIG
    case FreeBsdErrno::EFBig: return EFBIG;
#endif
#ifdef ENOSPC
    case FreeBsdErrno::ENoSpc: return ENOSPC;
#endif
#ifdef ESPIPE
    case FreeBsdErrno::ESPipe: return ESPIPE;
#endif
#ifdef EROFS
    case FreeBsdErrno::ERoFs: return EROFS;
#endif
#ifdef EMLINK
    case FreeBsdErrno::EMLink: return EMLINK;
#endif
#ifdef EPIPE
    case FreeBsdErrno::EPipe: return EPIPE;
#endif
#ifdef EDOM
    case FreeBsdErrno::EDom: return EDOM;
#endif
#ifdef ERANGE
    case FreeBsdErrno::ERange: return ERANGE;
#endif
#ifdef EAGAIN
    case FreeBsdErrno::EAgain: return EAGAIN;
#endif
#ifdef EINPROGRESS
    case FreeBsdErrno::EInProgress: return EINPROGRESS;
#endif
#ifdef EALREADY
    case FreeBsdErrno::EAlready: return EALREADY;
#endif
#ifdef ENOTSOCK
    case FreeBsdErrno::ENotSock: return ENOTSOCK;
#endif
#ifdef EDESTADDRREQ
    case FreeBsdErrno::EDestAddrReq: return EDESTADDRREQ;
#endif
#ifdef EMSGSIZE
    case FreeBsdErrno::EMsgSize: return EMSGSIZE;
#endif
#ifdef EOPNOTSUPP
    case FreeBsdErrno::EOpNotSupp: return EOPNOTSUPP;
#endif
#ifdef EADDRINUSE
    case FreeBsdErrno::EAddrInUse: return EADDRINUSE;
#endif
#ifdef ENETDOWN
    case FreeBsdErrno::ENetDown: return ENETDOWN;
#endif
#ifdef ENETUNREACH
    case FreeBsdErrno::ENetUnreach: return ENETUNREACH;
#endif
#ifdef ECONNABORTED
    case FreeBsdErrno::EConnAborted: return ECONNABORTED;
#endif
#ifdef ECONNRESET
    case FreeBsdErrno::EConnReset: return ECONNRESET;
#endif
#ifdef ENOBUFS
    case FreeBsdErrno::ENoBufs: return ENOBUFS;
#endif
#ifdef EISCONN
    case FreeBsdErrno::EIsConn: return EISCONN;
#endif
#ifdef ENOTCONN
    case FreeBsdErrno::ENotConn: return ENOTCONN;
#endif
#ifdef ETIMEDOUT
    case FreeBsdErrno::ETimedOut: return ETIMEDOUT;
#endif
#ifdef ECONNREFUSED
    case FreeBsdErrno::EConnRefused: return ECONNREFUSED;
#endif
#ifdef ELOOP
    case FreeBsdErrno::ELoop: return ELOOP;
#endif
#ifdef ENAMETOOLONG
    case FreeBsdErrno::ENameTooLong: return ENAMETOOLONG;
#endif
#ifdef EHOSTUNREACH
    case FreeBsdErrno::EHostUnreach: return EHOSTUNREACH;
#endif
#ifdef ENOTEMPTY
    case FreeBsdErrno::ENotEmpty: return ENOTEMPTY;
#endif
#ifdef EDQUOT
    case FreeBsdErrno::EDQuot: return EDQUOT;
#endif
#ifdef ESTALE
    case FreeBsdErrno::EStale: return ESTALE;
#endif
#ifdef ENOLCK
    case FreeBsdErrno::ENoLck: return ENOLCK;
#endif
#ifdef ENOSYS
    case FreeBsdErrno::ENoSys: return ENOSYS;
#endif
#ifdef EIDRM
    case FreeBsdErrno::EIdRm: return EIDRM;
#endif
#ifdef ENOMSG
    case FreeBsdErrno::ENoMsg: return ENOMSG;
#endif
#ifdef EOVERFLOW
    case FreeBsdErrno::EOverflow: return EOVERFLOW;
#endif
#ifdef ECANCELED
    case FreeBsdErrno::ECanceled: return ECANCELED;
#endif
#ifdef EILSEQ
    case FreeBsdErrno::EIlSeq: return EILSEQ;
#endif
#ifdef EBADMSG
    case FreeBsdErrno::EBadMsg: return EBADMSG;
#endif
#ifdef EPROTO
    case FreeBsdErrno::EProto: return EPROTO;
#endif
#ifdef EMULTIHOP
    case FreeBsdErrno::EMultiHop: return EMULTIHOP;
#endif
#ifdef ENOLINK
    case FreeBsdErrno::ENoLink: return ENOLINK;
#endif
#ifdef ENOTRECOVERABLE
    case FreeBsdErrno::ENotRecoverable: return ENOTRECOVERABLE;
#endif
#ifdef EOWNERDEAD
    case FreeBsdErrno::EOwnerDead: return EOWNERDEAD;
#endif
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
// on 0x800201E7). Map deliberately; this helper does not log, so a caller that wants the raw status
// preserved must log it itself (hle_kernel.cpp's win_exc_sce_error does exactly that).
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
