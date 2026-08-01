#!/usr/bin/env python3
"""pak_index — resolve UE4 `.pak` byte offsets to asset names, and decode a prosper APR read log.

Why this exists
---------------
A UE4 title on PS5 streams its content through the Ampr/APR async-read path. prosper logs every
one of those reads under `PROSPER_FILELOG=1`:

    [apr] read-submit id=1 <host-path-to.pak> -> dst=0x…(guest) off=0x89a1be4 size=3080 got=3080 OK

That tells you *how much* the guest read and *whether it succeeded*, but not **what it read**.
The offsets are raw byte positions inside a multi-gigabyte `.pak`, and a stripped shipping eboot
gives no other handle on the engine's loading state.

This tool closes that gap: it parses the pak's own index, so an offset becomes an asset path.
The read stream then reads like a load trace — which maps opened, which blueprints/widgets/
textures/sound banks were pulled in, in what order, and exactly where loading stopped. On
PPSA19244 (The Oregon Trail) that turned "the base pass draws nothing" into "the only map the
title ever loads is `Content/Maps/L_GameloftSplash.umap`" in one pass, with no game run at all.

It is offline and read-only: it needs the `.pak` and (optionally) a log file. No GPU, no boot.

Supported containers
--------------------
Pak index versions 8..11 (UE 4.22-4.27), unencrypted index, with either the v10+
path-hash/full-directory index pair or the older single directory index. IoStore
(`.utoc`/`.ucas`) containers are a different format and are **not** handled here.

Usage
-----
    # what asset lives at these pak offsets?
    pak_index.py GAME.pak --resolve 0x89a1be4 0xf161d288

    # decode a prosper run's whole APR read stream into an asset load trace
    pak_index.py GAME.pak --log run.log                 # every read, in order
    pak_index.py GAME.pak --log run.log --distinct      # first read of each asset, in order
    pak_index.py GAME.pak --log run.log --summary       # counts by extension / top directories

    # list what is in the container at all (does the title even ship map X?)
    pak_index.py GAME.pak --list '*.umap'

    pak_index.py --self-test        # no pak needed
"""

import argparse
import bisect
import fnmatch
import os
import re
import struct
import sys
from collections import Counter

PAK_MAGIC = 0x5A6F12E1

# The footer grew over time; probe the known sizes from the end of the file. Only the fields this
# tool needs are read, so a footer whose tail carries extra data still parses.
#   u32 magic, u32 version, u64 indexOffset, u64 indexSize, u8 indexHash[20], then version extras.
_FOOTER_FIXED = 4 + 4 + 8 + 8 + 20


class PakError(Exception):
    pass


def _find_footer(data: bytes, file_size: int):
    """Return (version, index_offset, index_size) from a trailing chunk of the pak."""
    magic_le = struct.pack('<I', PAK_MAGIC)
    # Search from the end: the real footer is the last magic in the file.
    pos = data.rfind(magic_le)
    while pos >= 0:
        if len(data) - pos >= _FOOTER_FIXED:
            version, = struct.unpack_from('<I', data, pos + 4)
            index_off, index_size = struct.unpack_from('<QQ', data, pos + 8)
            if 8 <= version <= 11 and 0 < index_size and index_off + index_size <= file_size:
                return version, index_off, index_size
        pos = data.rfind(magic_le, 0, pos)
    raise PakError('no usable pak footer found (encrypted index, IoStore container, or truncated file)')


class _Cursor:
    """Little-endian reader for UE4's FArchive primitives."""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.off = 0

    def u32(self):
        v, = struct.unpack_from('<I', self.buf, self.off)
        self.off += 4
        return v

    def i32(self):
        v, = struct.unpack_from('<i', self.buf, self.off)
        self.off += 4
        return v

    def i64(self):
        v, = struct.unpack_from('<q', self.buf, self.off)
        self.off += 8
        return v

    def u64(self):
        v, = struct.unpack_from('<Q', self.buf, self.off)
        self.off += 8
        return v

    def skip(self, n):
        self.off += n

    def fstring(self):
        n = self.i32()
        if n < 0:                       # negative length => UTF-16, count is in characters
            raw = self.buf[self.off:self.off + (-n) * 2]
            self.off += (-n) * 2
            return raw.decode('utf-16-le', 'replace').rstrip('\0')
        raw = self.buf[self.off:self.off + n]
        self.off += n
        return raw.decode('utf-8', 'replace').rstrip('\0')


def decode_entry(blob: bytes, at: int):
    """Decode one FPakEntry from UE4's EncodedPakEntries bitfield form.

    Layout of the leading u32 (FPakFile::DecodePakEntry):
      bit 31    offset is 32-bit
      bit 30    uncompressed size is 32-bit
      bit 29    compressed size is 32-bit (only present when compressed)
      bits 28-23 compression method index
      bit 22    encrypted
      bits 21-6 compression block count
      bits 5-0  compression block size / 2048
    Returns (offset, size, uncompressed_size, compression_method_index).
    """
    c = _Cursor(blob)
    c.off = at
    value = c.u32()
    method = (value >> 23) & 0x3F
    offset = c.u32() if value & (1 << 31) else c.i64()
    uncompressed = c.u32() if value & (1 << 30) else c.i64()
    if method != 0:
        size = c.u32() if value & (1 << 29) else c.i64()
    else:
        size = uncompressed
    return offset, size, uncompressed, method


def load_entries(pak_path: str):
    """Return (mount_point, sorted list of (offset, size, uncompressed, name, method))."""
    size = os.path.getsize(pak_path)
    with open(pak_path, 'rb') as f:
        tail_len = min(size, 4096)
        f.seek(size - tail_len)
        version, index_off, index_size = _find_footer(f.read(tail_len), size)

        f.seek(index_off)
        index = _Cursor(f.read(index_size))
        mount = index.fstring()
        index.i32()                                  # num entries (recomputed below)

        if version >= 10:
            index.u64()                              # path hash seed
            if index.i32():                          # has path-hash index
                index.i64(); index.i64(); index.skip(20)
            has_full_dir = index.i32()
            if not has_full_dir:
                raise PakError('pak has no full directory index; filenames are unavailable '
                               '(only the path-hash index is present)')
            dir_off, dir_size = index.i64(), index.i64()
            index.skip(20)
            encoded_size = index.i32()
            encoded = index.buf[index.off:index.off + encoded_size]
            f.seek(dir_off)
            directory = _Cursor(f.read(dir_size))
        else:
            # v8/v9 keep the directory inline and store full FPakEntry records, not the encoded form.
            raise PakError('pak index version %d is not supported yet (needs the v10+ '
                           'encoded-entry + full-directory layout)' % version)

        out = []
        for _ in range(directory.i32()):
            dir_name = directory.fstring()
            for _ in range(directory.i32()):
                file_name = directory.fstring()
                enc_at = directory.u32()
                try:
                    off, sz, unc, method = decode_entry(encoded, enc_at)
                except struct.error:
                    continue
                out.append((off, sz, unc, dir_name + file_name, method))
    out.sort()
    return mount, out


class PakMap:
    """Offset -> asset-name lookup over a decoded pak index."""

    # A read may start at the entry header rather than the payload; UE4's per-entry header is well
    # under 4 KiB, so allow that much slack before calling a hit "past the end of this entry".
    HEADER_SLACK = 4096

    def __init__(self, entries):
        self.entries = entries
        self.starts = [e[0] for e in entries]

    def resolve(self, offset: int):
        """Return (name, exact) — exact=False means the offset fell past the entry's payload."""
        i = bisect.bisect_right(self.starts, offset) - 1
        if i < 0:
            return None, False
        off, size, _unc, name, _m = self.entries[i]
        return name, offset < off + size + self.HEADER_SLACK


_READ_RE = re.compile(
    r'\[apr\] read-submit id=(?P<id>\d+) (?P<path>\S+) -> dst=\S+ '
    r'off=(?P<off>0x[0-9a-fA-F]+) size=(?P<size>\d+) got=(?P<got>\d+) (?P<status>\S+)')


def parse_read_log(path: str):
    """Yield (offset, size, got, status) for each `[apr] read-submit` result line."""
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = _READ_RE.search(line)
            if m:
                yield (int(m.group('off'), 16), int(m.group('size')),
                       int(m.group('got')), m.group('status'))


def _self_test():
    # decode_entry: 32-bit offset + 32-bit uncompressed, uncompressed (method 0).
    value = (1 << 31) | (1 << 30)
    blob = struct.pack('<III', value, 0x1000, 0x40)
    assert decode_entry(blob, 0) == (0x1000, 0x40, 0x40, 0), decode_entry(blob, 0)

    # 64-bit offset + 64-bit sizes, compressed (method 1) so the compressed size is a separate field.
    value = (1 << 23)
    blob = struct.pack('<Iqqq', value, 0x1_0000_0000, 0x200, 0x100)
    assert decode_entry(blob, 0) == (0x1_0000_0000, 0x100, 0x200, 1), decode_entry(blob, 0)

    # PakMap: exact hit, header slack, and a miss past the entry.
    m = PakMap([(0, 100, 100, 'a.uasset', 0), (1000, 50, 50, 'b.uexp', 0)])
    assert m.resolve(0) == ('a.uasset', True)
    assert m.resolve(99) == ('a.uasset', True)
    assert m.resolve(1049) == ('b.uexp', True)
    assert m.resolve(1000 + 50 + PakMap.HEADER_SLACK) == ('b.uexp', False)
    assert m.resolve(-1) == (None, False)

    # A read landing in the gap after an entry still attributes to that entry while it is inside
    # the header slack; past the slack the caller is told the match is not exact. 500 is only
    # 400 bytes past a.uasset's payload, so it is still reported as an exact hit.
    assert m.resolve(500) == ('a.uasset', True)
    # Past the LAST entry's payload+slack there is nothing left to attribute to, so the nearest
    # preceding entry is returned with exact=False rather than silently claimed as a hit.
    assert m.resolve(1000 + 50 + PakMap.HEADER_SLACK + 1) == ('b.uexp', False)

    # Log parsing, including a failed read (status is preserved, not filtered).
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as fh:
        fh.write("noise\n")
        fh.write("[apr] read-submit id=1 /x/y.pak -> dst=0x30(guest) off=0x1000 size=64 got=64 OK\n")
        fh.write("[apr] read-submit id=1 /x/y.pak -> dst=0x30(guest) off=0x2000 size=8 got=0 FAIL\n")
        name = fh.name
    try:
        rows = list(parse_read_log(name))
        assert rows == [(0x1000, 64, 64, 'OK'), (0x2000, 8, 0, 'FAIL')], rows
    finally:
        os.unlink(name)

    # _find_footer picks the LAST magic, and rejects a bogus one.
    body = b'\x00' * 64 + struct.pack('<I', PAK_MAGIC) + b'\xff' * 60
    footer = struct.pack('<IIQQ', PAK_MAGIC, 11, 16, 32) + b'\x00' * 20
    data = body + footer
    assert _find_footer(data, 4096) == (11, 16, 32)

    print('pak_index self-test OK')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pak', nargs='?', help='path to the title .pak')
    ap.add_argument('--resolve', nargs='+', metavar='OFF',
                    help='resolve these pak byte offsets (decimal or 0x…) to asset names')
    ap.add_argument('--log', metavar='FILE',
                    help='a prosper run log captured with PROSPER_FILELOG=1; decode its '
                         '[apr] read-submit stream into an asset load trace')
    ap.add_argument('--distinct', action='store_true',
                    help='with --log: print each asset once, in first-read order')
    ap.add_argument('--summary', action='store_true',
                    help='with --log: print extension and directory histograms instead of a trace')
    ap.add_argument('--list', metavar='GLOB',
                    help='list container entries whose path matches this glob (e.g. "*.umap")')
    ap.add_argument('--self-test', action='store_true', help='run unit checks; no pak needed')
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.pak:
        ap.error('a .pak path is required (or use --self-test)')

    mount, entries = load_entries(args.pak)
    pak = PakMap(entries)
    print(f'# {args.pak}: {len(entries)} entries, mount point {mount!r}', file=sys.stderr)

    if args.list:
        for _off, _sz, _unc, name, _m in entries:
            if fnmatch.fnmatch(name, args.list):
                print(name)

    if args.resolve:
        for token in args.resolve:
            off = int(token, 0)
            name, exact = pak.resolve(off)
            mark = '' if exact else '   (past this entry — offset is in a gap or unmapped)'
            print(f'{off:#x}  {name}{mark}')

    if args.log:
        reads = list(parse_read_log(args.log))
        if not reads:
            print('no [apr] read-submit result lines found — was PROSPER_FILELOG=1 set?',
                  file=sys.stderr)
            return 1
        resolved = [(off, size, status, pak.resolve(off)[0]) for off, size, _got, status in reads]
        failed = sum(1 for _o, _s, st, _n in resolved if st != 'OK')
        total = sum(s for _o, s, _st, _n in resolved)
        print(f'# {len(resolved)} reads, {total/1e6:.1f} MB, {failed} not OK', file=sys.stderr)

        if args.summary:
            names = [n for _o, _s, _st, n in resolved if n]
            uniq = list(dict.fromkeys(names))
            print(f'distinct assets: {len(uniq)}')
            ext = Counter(n.rsplit('.', 1)[-1] if '.' in n else '?' for n in uniq)
            print('by extension:', ext.most_common(20))
            top = Counter('/'.join(n.split('/')[:3]) for n in uniq)
            print('by directory:', top.most_common(20))
        elif args.distinct:
            for n in dict.fromkeys(n for _o, _s, _st, n in resolved):
                print(n)
        else:
            for off, size, status, name in resolved:
                flag = '' if status == 'OK' else f'  [{status}]'
                print(f'{off:#012x} {size:>9} {name}{flag}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
