#!/usr/bin/env python3
"""dropcache.py — evict a path's pages from the host page cache, and prove it happened.

Why this exists
---------------
A guest's `read`/`pread` is served from the **host page cache** on every run after the first, so a
title's asset load can be several times faster on prosper than the same bytes off real storage. For
most work that is invisible. For a title whose startup is a *race* it is decisive: PPSA26414
(R-Type Delta) reaches its `DLLInit` in 289-362 ms warm and 779-1158 ms with the dump evicted, and
it faults in the first case and boots in the second (`docs/R_TYPE_DELTA_STATUS.md`). Any startup
timing taken without recording cache state is therefore unreproducible.

Scope, and the hazard it does NOT remove
----------------------------------------
`posix_fadvise(POSIX_FADV_DONTNEED)` needs no root and touches nothing outside the files named on
the command line — unlike `/proc/sys/vm/drop_caches`, which drops the whole machine's cache. That is
the only safety property it has. **The page cache is global**, so evicting a shared dump evicts it
for *every* process on the host: pointing this at a tree another agent is currently timing silently
invalidates their run. On a shared box, say what you are about to evict first.

It reports resident pages **before and after** via `mincore(2)`, because "I evicted it" and "it was
already evicted" and "eviction silently did nothing" otherwise look identical — and a run whose
premise is the cache state must be able to show its lever moved. Dirty pages cannot be dropped; if
`after` stays high the tool says so and exits non-zero, so a scripted cold arm cannot quietly
inherit a warm cache. A path that does not exist, or that contains no files, is an error for the
same reason: a typo must not be indistinguishable from a successful eviction.

Usage
-----
    python3 tools/dropcache.py <DUMP_ROOT>/PPSA26414-app0          # evict a whole dump
    python3 tools/dropcache.py <path> [<path> ...] [--quiet]
    python3 tools/dropcache.py --report-only <path>                # measure, evict nothing
    python3 tools/dropcache.py --self-test                         # no path needed

Exit status: 0 evicted (or measured) cleanly, 1 pages remained resident / nothing was measurable,
2 usage error (unknown flag, missing or empty path).
"""
import ctypes
import ctypes.util
import mmap
import os
import sys

USAGE = """usage: dropcache.py [--report-only] [--quiet] <path> [<path> ...]
       dropcache.py --self-test
exit: 0 clean, 1 pages still resident / nothing measurable, 2 usage error"""

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
        buf = ctypes.c_char.from_buffer(mm)
        addr = ctypes.addressof(buf)
        rc = _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec)
        resident = None if rc != 0 else sum(1 for b in vec if b & 1)
        del buf                      # release the buffer export before mm.close()
        return resident
    finally:
        mm.close()


def walk(paths):
    """Yield every regular file under `paths`. Raises FileNotFoundError on a path that is absent."""
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        if os.path.isfile(p):
            yield p
        else:
            for dirpath, _dirs, files in os.walk(p):
                for f in files:
                    yield os.path.join(dirpath, f)


def run(paths, report_only=False, quiet=False, out=None):
    """Evict (or measure) `paths`. Returns a process exit status."""
    out = out or sys.stdout
    files = before = after = total = unmeasured = 0
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
            r1 = r0 if report_only else resident_pages(fd, size)
            files += 1
            total += size
            if r0 is None or r1 is None:
                unmeasured += 1
            else:
                before += r0
                after += r1
        finally:
            os.close(fd)

    mb = lambda pages: pages * _PAGE / 1e6
    if files == 0:
        print("error: no readable files under %s — nothing was evicted. A typo must not look like a "
              "successful cold arm." % ", ".join(paths), file=sys.stderr)
        return 2
    if not quiet:
        print("%s %d files, %.1f MB on disk" % ("measured" if report_only else "evicted",
                                                files, total / 1e6), file=out)
        print("  page cache: %.1f MB resident before -> %.1f MB after" % (mb(before), mb(after)),
              file=out)
        if unmeasured:
            print("  (%d files not measurable by mincore; eviction was still requested)" % unmeasured,
                  file=out)
    # Warnings go to stderr and set the exit status, so `--quiet` and a scripted arm still see them.
    if not report_only and before > 0 and after > before * 0.1:
        print("WARNING: %.1f MB of %.1f MB is still resident — dirty or otherwise unevictable. "
              "Do NOT report this run as a cold-cache arm." % (mb(after), mb(before)), file=sys.stderr)
        return 1
    if unmeasured == files:
        print("WARNING: residency was not measurable for any file; this run cannot prove it was cold.",
              file=sys.stderr)
        return 1
    return 0


def self_test():
    """Exercise the answer shapes without needing a dump: warm, evicted, and a bad path.

    It uses this script's own file as the subject — already on disk, already clean, so no write
    and no `fsync` is needed. (An earlier version wrote and fsync'd a temp file and could block
    for minutes in `D` state on a busy host; the subject only has to be *clean*, not fresh.)
    """
    ok = True
    me = os.path.abspath(__file__)
    with open(os.devnull, "w") as null:
        with open(me, "rb") as f:                       # warm it
            f.read()
        fd = os.open(me, os.O_RDONLY)
        try:
            warm = resident_pages(fd, os.fstat(fd).st_size)
        finally:
            os.close(fd)
        if not warm:
            print("self-test: a just-read file reported 0 resident pages — mincore is not working")
            ok = False
        if run([me], report_only=True, out=null) != 0:
            print("self-test: report-only on a warm file should be status 0")
            ok = False
        if run([me], out=null) != 0:
            print("self-test: eviction of a clean file should be status 0")
            ok = False
        fd = os.open(me, os.O_RDONLY)
        try:
            cold = resident_pages(fd, os.fstat(fd).st_size)
        finally:
            os.close(fd)
        if cold:
            print("self-test: %d pages still resident after eviction — either this filesystem "
                  "cannot be evicted (tmpfs?) or another process is holding the file" % cold)
            ok = False
        # The next two arms print their own error line to stderr; that IS the behaviour under test.
        if main([me + "-no-such-path"]) != 2:
            print("self-test: an absent path must be status 2")
            ok = False
        if main(["--report_only", me]) != 2:
            print("self-test: a misspelled flag must be status 2, never a silent eviction")
            ok = False
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()
    report_only = quiet = False
    paths = []
    for a in argv:
        if a == "--report-only":
            report_only = True
        elif a == "--quiet":
            quiet = True
        elif a.startswith("-"):
            print("error: unknown option %r\n%s" % (a, USAGE), file=sys.stderr)
            return 2
        else:
            paths.append(a)
    if not paths:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        return run(paths, report_only=report_only, quiet=quiet)
    except FileNotFoundError as e:
        print("error: no such path: %s\n%s" % (e.args[0], USAGE), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
