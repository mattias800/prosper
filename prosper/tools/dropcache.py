#!/usr/bin/env python3
"""dropcache.py — evict a path's pages from the host page cache, and prove it happened.

Why this exists
---------------
A guest's `read`/`pread` is served from the **host page cache** on every run after the first, so a
title's asset load can be several times faster on prosper than the same bytes off real storage. For
most work that is invisible. For a title whose startup is a *race* it is decisive: PPSA26414
(R-Type Delta) reaches its `DLLInit` in 289-362 ms warm and 797-1158 ms with the dump evicted, and
it faults in the first case and boots in the second (`docs/R_TYPE_DELTA_STATUS.md`). Any startup
timing taken without recording cache state is therefore unreproducible.

This is an unprivileged, *scoped* eviction — `posix_fadvise(POSIX_FADV_DONTNEED)` on the named files
only. It does not need root, does not touch `/proc/sys/vm/drop_caches`, and cannot disturb another
agent's working set, so it is safe to run on a shared box.

It reports resident pages **before and after** via `mincore(2)`, because "I evicted it" and "it was
already evicted" and "eviction silently did nothing" otherwise look identical — and a run whose
premise is the cache state must be able to show its lever moved. Dirty pages cannot be dropped;
if `after` stays high, that is the reason, and the number says so instead of hiding it.

Usage
-----
    python3 tools/dropcache.py <DUMP_ROOT>/PPSA26414-app0          # evict a whole dump
    python3 tools/dropcache.py <path> [<path> ...] [--quiet]
    python3 tools/dropcache.py --report-only <path>                # measure, evict nothing
"""
import ctypes
import ctypes.util
import mmap
import os
import sys

_PAGE = os.sysconf("SC_PAGE_SIZE")
_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
_libc.mincore.restype = ctypes.c_int


def resident_pages(fd, size):
    """Pages of this file currently in the page cache, or None if it cannot be measured."""
    if size == 0:
        return 0
    try:
        # MAP_PRIVATE with PROT_WRITE only so ctypes will hand out the mapping's address
        # (from_buffer refuses a read-only buffer). Nothing is ever written, so no page is
        # copied and mincore still reports the file's own residency.
        mm = mmap.mmap(fd, size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    except (ValueError, OSError):
        return None
    try:
        npages = (size + _PAGE - 1) // _PAGE
        vec = (ctypes.c_ubyte * npages)()
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        if _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
            return None
        return sum(1 for b in vec if b & 1)
    finally:
        mm.close()


def walk(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
        else:
            for dirpath, _dirs, files in os.walk(p):
                for f in files:
                    yield os.path.join(dirpath, f)


def main(argv):
    report_only = "--report-only" in argv
    quiet = "--quiet" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print(__doc__.strip().split("Usage\n-----\n", 1)[1])
        return 2

    files = before = after = total = 0
    unmeasured = 0
    for path in walk(paths):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            size = os.fstat(fd).st_size
            r0 = resident_pages(fd, size)
            if not report_only:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            r1 = resident_pages(fd, size) if not report_only else r0
            files += 1
            total += size
            if r0 is None or r1 is None:
                unmeasured += 1
            else:
                before += r0
                after += r1
        finally:
            os.close(fd)

    if not quiet:
        mb = lambda pages: pages * _PAGE / 1e6
        verb = "measured" if report_only else "evicted"
        print("%s %d files, %.1f MB on disk" % (verb, files, total / 1e6))
        print("  page cache: %.1f MB resident before -> %.1f MB after" % (mb(before), mb(after)))
        if unmeasured:
            print("  (%d files not measurable by mincore; eviction was still requested)" % unmeasured)
        if not report_only and after > before * 0.1 and before > 0:
            print("  WARNING: most pages are still resident — dirty or otherwise unevictable. "
                  "Do NOT report this run as a cold-cache arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
