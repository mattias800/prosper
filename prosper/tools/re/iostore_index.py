#!/usr/bin/env python3
"""iostore_index — resolve UE4/UE5 IoStore (`.utoc`/`.ucas`) byte offsets to package names.

Why this exists
---------------
`pak_index.py` turns a `.pak` byte offset into an asset path, and on a pre-IoStore title that is
enough to read a run's `PROSPER_FILELOG=1` stream as a load trace. Modern Unreal titles ship their
packages in **IoStore** containers instead: the `.pak` keeps only configs, fonts and localization,
while every `.uasset`/`.umap`/`.ubulk` lives in a `.ucas` addressed through a `.utoc` directory.
`pak_index.py` says so itself and stops there — so on an IoStore title the read stream is a list of
raw offsets into a multi-gigabyte blob and names nothing.

This tool closes that gap for IoStore, giving Unreal titles the equivalent of Unity's
`Media/levelNN` scene oracle: **which package the guest actually read, in order**. That is the one
measurement that separates "the run reached gameplay" from "the run sat on an animating menu",
which no aggregate frame metric can do (see `docs/GAME_COMPAT_ORCHESTRATION.md`).

It is self-checking in the same way the Unity oracle is: IoStore package paths are semantic
(`.../Map/Product/Title/Title_PL.umap` vs `.../Map/Product/World/Field/F110/...`), so a
mis-derived mapping surfaces as nonsense rather than as a plausible wrong answer.

Offline and read-only: it needs the `.utoc` (never the multi-GB `.ucas`) and optionally a log.

Format notes (why the two-step offset mapping)
---------------------------------------------
A `.utoc` chunk's `FIoOffsetAndLength` is an offset in the container's **logical** (uncompressed)
space, and chunks are aligned up to `CompressionBlockSize`. The `.ucas` is written as packed
compression blocks, so a **physical** file offset — which is what a read syscall carries — is not
the same number. Logical block `k` spans `[k*CompressionBlockSize, …)` and its bytes live at
`CompressionBlocks[k].Offset`. So a read offset is resolved physical -> block -> logical -> chunk.
On this project's first IoStore title the two spaces differ by 5.3 GB over a 17.8 GB container, so
skipping the block step silently misnames most of the stream.

Usage
-----
    # what package lives at these .ucas byte offsets?
    iostore_index.py GAME.utoc --resolve 0x4a1be40 0x1f161d288

    # decode a prosper run's read stream into a package load trace
    iostore_index.py GAME.utoc --log run.log                 # every read, in order
    iostore_index.py GAME.utoc --log run.log --distinct      # first read of each package
    iostore_index.py GAME.utoc --log run.log --summary       # counts by extension / directory
    iostore_index.py GAME.utoc --log run.log --maps          # just the .umap sequence

    # what does the container hold at all?
    iostore_index.py GAME.utoc --list '*/Map/Product/World/*.umap'

    iostore_index.py --self-test        # no container needed

Supported containers
--------------------
`.utoc` versions 1..5 (UE 4.25-5.x) with an **unencrypted** directory index. An encrypted
container (`ContainerFlags & Encrypted`) needs the title's AES key and is refused, loudly. The
directory index is optional in the format; a container written without one (`--no-index`) can
still resolve offsets to a chunk index, just not to a name, and the tool says so.
"""

import argparse
import bisect
import fnmatch
import os
import re
import struct
import sys
from collections import Counter

TOC_MAGIC = b'-==--==--==--==-'
INVALID = 0xFFFFFFFF

# EIoContainerFlags
FLAG_COMPRESSED = 1 << 0
FLAG_ENCRYPTED = 1 << 1
FLAG_SIGNED = 1 << 2
FLAG_INDEXED = 1 << 3


class TocError(Exception):
    pass


def _read_fstring(buf: bytes, off: int):
    """UE FString: int32 length, then chars. Negative length means UTF-16. Includes the NUL."""
    (length,) = struct.unpack_from('<i', buf, off)
    off += 4
    if length == 0:
        return '', off
    if length < 0:
        count = -length
        text = buf[off:off + 2 * count - 2].decode('utf-16-le', 'replace')
        off += 2 * count
    else:
        text = buf[off:off + length - 1].decode('utf-8', 'replace')
        off += length
    return text, off


class IoStoreToc:
    """Parsed `.utoc`: chunk extents, the physical<->logical block map, and the directory index."""

    def __init__(self, path: str):
        with open(path, 'rb') as f:
            data = f.read()
        self.path = path
        self.data = data
        self._parse_header()
        self._parse_layout()
        self._parse_offsets()
        self._parse_blocks()
        self._parse_directory()

    # ---- header -------------------------------------------------------------------------
    def _parse_header(self):
        d = self.data
        if len(d) < 144 or d[0:16] != TOC_MAGIC:
            raise TocError(f'{self.path}: not an IoStore .utoc (bad magic)')
        self.version = d[16]
        if not 1 <= self.version <= 5:
            raise TocError(f'{self.path}: unsupported .utoc version {self.version}')
        (self.header_size, self.entry_count, self.block_count, self.block_entry_size,
         self.method_count, self.method_length, self.compression_block_size,
         self.dir_index_size, self.partition_count) = struct.unpack_from('<9I', d, 20)
        off = 20 + 36
        (self.container_id,) = struct.unpack_from('<Q', d, off)
        off += 8
        self.encryption_key_guid = d[off:off + 16]
        off += 16
        self.flags = d[off]
        off += 4                                    # flags u8 + 3 reserved
        self.perfect_hash_seed_count = 0
        if self.version >= 4:
            (self.perfect_hash_seed_count,) = struct.unpack_from('<I', d, off)
        if self.flags & FLAG_ENCRYPTED:
            raise TocError(
                f'{self.path}: container directory index is ENCRYPTED '
                f'(key guid {self.encryption_key_guid.hex()}) — this tool cannot name its chunks')

    # ---- section layout -----------------------------------------------------------------
    def _parse_layout(self):
        n, d = self.entry_count, self.data
        off = self.header_size
        self.chunk_id_off = off
        off += 12 * n
        self.offlen_off = off
        off += 10 * n
        if self.version >= 4:
            off += 4 * self.perfect_hash_seed_count
            # v5 also stores the chunk indices that missed the perfect hash.
            if self.version >= 5:
                (without,) = struct.unpack_from('<I', d, 20 + 36 + 8 + 16 + 4 + 4 + 8)
                off += 4 * without
        self.block_off = off
        off += self.block_entry_size * self.block_count
        off += self.method_length * self.method_count
        if self.flags & FLAG_SIGNED:
            (sig_size,) = struct.unpack_from('<I', d, off)
            off += 4 + 2 * sig_size + 20 * self.block_count
        self.dir_off = off
        off += self.dir_index_size
        self.meta_off = off
        off += 33 * n
        # A layout that does not consume the file exactly means a field was mis-sized; that would
        # silently misname every offset, so it is a hard error rather than a warning.
        if off != len(d):
            raise TocError(f'{self.path}: .utoc layout mismatch — computed {off} bytes, file is {len(d)}')

    # ---- chunk extents ------------------------------------------------------------------
    def _parse_offsets(self):
        d, n, base = self.data, self.entry_count, self.offlen_off
        offs, lens = [], []
        for i in range(n):
            rec = d[base + 10 * i: base + 10 * i + 10]
            offs.append(int.from_bytes(rec[0:5], 'big'))    # FIoOffsetAndLength is big-endian
            lens.append(int.from_bytes(rec[5:10], 'big'))
        order = sorted(range(n), key=lambda i: offs[i])
        self.chunk_offsets = offs
        self.chunk_lengths = lens
        self._sorted_chunks = order
        self._sorted_starts = [offs[i] for i in order]

    def _parse_blocks(self):
        d, base, sz = self.data, self.block_off, self.block_entry_size
        phys = []
        for i in range(self.block_count):
            rec = d[base + sz * i: base + sz * i + sz]
            phys.append(int.from_bytes(rec[0:5], 'little'))  # compressed offset, little-endian
        self.block_phys = phys
        self.block_csize = [int.from_bytes(d[base + sz * i + 5: base + sz * i + 8], 'little')
                            for i in range(self.block_count)]
        order = sorted(range(self.block_count), key=lambda i: phys[i])
        self._sorted_blocks = order
        self._sorted_block_starts = [phys[i] for i in order]

    # ---- directory index ----------------------------------------------------------------
    def _parse_directory(self):
        self.mount = ''
        self.paths = {}
        if self.dir_index_size == 0:
            return
        d = self.data
        off = self.dir_off
        self.mount, off = _read_fstring(d, off)
        (dir_count,) = struct.unpack_from('<i', d, off)
        off += 4
        dirs = [struct.unpack_from('<4I', d, off + 16 * i) for i in range(dir_count)]
        off += 16 * dir_count
        (file_count,) = struct.unpack_from('<i', d, off)
        off += 4
        files = [struct.unpack_from('<3I', d, off + 12 * i) for i in range(file_count)]
        off += 12 * file_count
        (str_count,) = struct.unpack_from('<i', d, off)
        off += 4
        strings = []
        for _ in range(str_count):
            text, off = _read_fstring(d, off)
            strings.append(text)
        if off != self.dir_off + self.dir_index_size:
            raise TocError(f'{self.path}: directory index tail mismatch '
                           f'({off} != {self.dir_off + self.dir_index_size})')

        prefix = self.mount if self.mount.endswith('/') else self.mount + '/'
        paths = {}
        # Iterative sibling/child walk: the tree is ~100k nodes deep-ish and recursion would need a
        # raised limit to survive a deeply nested content tree.
        stack = [(0, prefix)]
        while stack:
            index, parent = stack.pop()
            while index != INVALID:
                name, first_child, next_sibling, first_file = dirs[index]
                here = parent if name == INVALID else parent + strings[name] + '/'
                file_index = first_file
                while file_index != INVALID:
                    fname, next_file, user_data = files[file_index]
                    paths[user_data] = here + strings[fname]
                    file_index = next_file
                if first_child != INVALID:
                    stack.append((first_child, here))
                index = next_sibling
        self.paths = paths

    # ---- resolution ---------------------------------------------------------------------
    def physical_to_logical(self, offset: int):
        """Map a `.ucas` file offset to the container's logical (uncompressed) offset."""
        i = bisect.bisect_right(self._sorted_block_starts, offset) - 1
        if i < 0:
            return None
        block = self._sorted_blocks[i]
        within = offset - self.block_phys[block]
        if within >= self.block_csize[block]:
            # Past this block's payload — the read starts in inter-block padding. Attribute it to
            # the block it lands after rather than inventing a chunk.
            within = self.block_csize[block]
        return block * self.compression_block_size + within

    def chunk_at_logical(self, logical: int):
        i = bisect.bisect_right(self._sorted_starts, logical) - 1
        if i < 0:
            return None, False
        chunk = self._sorted_chunks[i]
        start = self.chunk_offsets[chunk]
        exact = logical < start + self.chunk_lengths[chunk]
        return chunk, exact

    def resolve(self, offset: int):
        """Return (name, exact). `name` is None when the container has no directory index."""
        logical = self.physical_to_logical(offset)
        if logical is None:
            return None, False
        chunk, exact = self.chunk_at_logical(logical)
        if chunk is None:
            return None, False
        return self.paths.get(chunk, f'<chunk {chunk}>'), exact


_READ_RE = re.compile(
    r'\[apr\] read-submit id=(?P<id>\d+) (?P<path>\S+) -> dst=\S+ '
    r'off=(?P<off>0x[0-9a-fA-F]+) size=(?P<size>\d+) got=(?P<got>\d+) (?P<status>\S+)')

# Not every Unreal title streams IoStore through APR: an ordinary pread/read carries the same
# information under PROSPER_FILELOG=1, with the resolved host path attached to the fd.
_FD_READ_RE = re.compile(
    r"\[file\] (?:pread|read) fd=\d+ path='(?P<path>[^']*)' "
    r"off=(?P<off>0x[0-9a-fA-F]+) count=(?P<size>0x[0-9a-fA-F]+) -> (?P<got>-?\d+)")


def parse_read_log(path: str, container_stem: str = None):
    """Yield (offset, size, got, status) for read lines that name this container.

    Both the `[apr] read-submit` and the ordinary `[file] pread/read` forms are accepted, because
    which one a title uses is a property of the title, not of the container.

    `container_stem` filters by host path, so a title that streams from several `.ucas` files does
    not attribute one container's offsets to another's index.
    """
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = _READ_RE.search(line)
            if m:
                if container_stem and container_stem not in m.group('path'):
                    continue
                yield (int(m.group('off'), 16), int(m.group('size')),
                       int(m.group('got')), m.group('status'))
                continue
            m = _FD_READ_RE.search(line)
            if m:
                if container_stem and container_stem not in m.group('path'):
                    continue
                got = int(m.group('got'))
                yield (int(m.group('off'), 16), int(m.group('size'), 16),
                       max(got, 0), 'OK' if got >= 0 else 'FAIL')


def _self_test():
    # FString round-trips, both encodings.
    assert _read_fstring(struct.pack('<i', 0), 0) == ('', 4)
    blob = struct.pack('<i', 4) + b'abc\0'
    assert _read_fstring(blob, 0) == ('abc', 8)
    blob = struct.pack('<i', -3) + 'ab\0'.encode('utf-16-le')
    assert _read_fstring(blob, 0) == ('ab', 10)

    # Offset mapping: the physical->logical step is the whole point, so test it against a layout
    # where the two spaces genuinely differ. Two 64 KiB logical blocks whose payloads are 16 and
    # 32 bytes: logical block 1 starts at 65536, physical block 1 starts at 16.
    class Fake(IoStoreToc):
        def __init__(self):
            self.compression_block_size = 65536
            self.block_phys = [0, 16]
            self.block_csize = [16, 32]
            self._sorted_blocks = [0, 1]
            self._sorted_block_starts = [0, 16]
            self.chunk_offsets = [0, 65536]
            self.chunk_lengths = [16, 32]
            self._sorted_chunks = [0, 1]
            self._sorted_starts = [0, 65536]
            self.paths = {0: 'A.uasset', 1: 'B.umap'}

    f = Fake()
    assert f.physical_to_logical(0) == 0
    assert f.physical_to_logical(8) == 8
    assert f.physical_to_logical(16) == 65536
    assert f.physical_to_logical(40) == 65560
    assert f.resolve(0) == ('A.uasset', True)
    assert f.resolve(20) == ('B.umap', True)
    # A naive "physical == logical" reading would call offset 20 part of chunk 0; it must not.
    assert f.chunk_at_logical(20)[0] == 0 and f.resolve(20)[0] == 'B.umap'
    # Past the last chunk's payload: named, but not exact.
    assert f.resolve(47) == ('B.umap', True)
    assert f.physical_to_logical(48) == 65568
    assert f.resolve(48)[1] is False

    # Both read-log dialects decode, and the container filter uses the host path from each.
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as fh:
        fh.write("[apr] read-submit id=1 /d/pakchunk0-ps5.ucas -> dst=0x30(guest) "
                 "off=0x1000 size=64 got=64 OK method=id requested=64\n")
        fh.write("[apr] read-submit id=1 /d/other.ucas -> dst=0x30(guest) "
                 "off=0x2000 size=8 got=8 OK method=id requested=8\n")
        fh.write("[file] pread fd=7 path='/d/pakchunk0-ps5.ucas' off=0x3000 count=0x40 "
                 "-> 64 error=0\n")
        fh.write("[file] pread fd=7 path='/d/other.ucas' off=0x4000 count=0x40 -> 64 error=0\n")
        # The sibling .pak shares the container's stem and has a completely different offset
        # space; matching it would name pak offsets against the IoStore index.
        fh.write("[apr] read-submit id=2 /d/pakchunk0-ps5.pak -> dst=0x30(guest) "
                 "off=0x5000 size=8 got=8 OK method=id requested=8\n")
        log = fh.name
    got = list(parse_read_log(log, 'pakchunk0-ps5.ucas'))
    assert got == [(0x1000, 64, 64, 'OK'), (0x3000, 64, 64, 'OK')], got
    assert len(list(parse_read_log(log))) == 5
    os.unlink(log)

    print('self-test OK')


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='iostore_index — resolve UE4/UE5 IoStore .ucas offsets to package names, and '
                    'decode a prosper APR read log into a package load trace.')
    ap.add_argument('utoc', nargs='?', help='path to the title .utoc')
    ap.add_argument('--resolve', nargs='+', metavar='OFF',
                    help='resolve these .ucas byte offsets (decimal or 0x…) to package names')
    ap.add_argument('--log', metavar='FILE',
                    help='a prosper run log captured with PROSPER_FILELOG=1; decode its '
                         '[apr] read-submit stream into a package load trace')
    ap.add_argument('--distinct', action='store_true',
                    help='with --log: print each package once, in first-read order')
    ap.add_argument('--summary', action='store_true',
                    help='with --log: print extension and directory histograms instead of a trace')
    ap.add_argument('--maps', action='store_true',
                    help='with --log: print only the .umap sequence, once each, in order')
    ap.add_argument('--list', metavar='GLOB',
                    help='list container entries whose path matches this glob (e.g. "*.umap")')
    ap.add_argument('--all-containers', action='store_true',
                    help='with --log: do not filter read lines by this container\'s file stem')
    ap.add_argument('--self-test', action='store_true', help='run unit checks; no container needed')
    args = ap.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.utoc:
        ap.error('a .utoc path is required (or --self-test)')

    try:
        toc = IoStoreToc(args.utoc)
    except TocError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(f'# {args.utoc}: v{toc.version}, {toc.entry_count} chunks, '
          f'{len(toc.paths)} named, mount {toc.mount!r}, block {toc.compression_block_size}')
    if not toc.paths:
        print('# no directory index in this container — offsets resolve to chunk indices only',
              file=sys.stderr)

    if args.list:
        pattern = args.list
        for chunk in sorted(toc.paths, key=lambda c: toc.chunk_offsets[c]):
            name = toc.paths[chunk]
            if fnmatch.fnmatch(name, pattern):
                print(name)
        return 0

    if args.resolve:
        for text in args.resolve:
            offset = int(text, 0)
            name, exact = toc.resolve(offset)
            mark = '' if exact else '  (past payload)'
            print(f'0x{offset:x}  {name}{mark}')
        return 0

    if args.log:
        # Match the container's own `.ucas`, not merely its stem: a title normally ships
        # `pakchunk0-ps5.pak` beside `pakchunk0-ps5.ucas`, and a stem filter would feed the pak's
        # own offsets to this index and name them confidently and wrongly.
        stem = None if args.all_containers else os.path.basename(args.utoc)[:-len('.utoc')] + '.ucas'
        reads = list(parse_read_log(args.log, stem))
        if not reads:
            print('no [apr] read-submit result lines matched — was PROSPER_FILELOG=1 set, and does '
                  'this title stream through APR?', file=sys.stderr)
            return 1
        names = []
        for offset, size, got, status in reads:
            name, exact = toc.resolve(offset)
            names.append((name or '<unmapped>', offset, size, got, status, exact))

        if args.summary:
            print(f'# {len(names)} reads')
            ext = Counter(n.rsplit('.', 1)[-1].lower() for n, *_ in names)
            print('## by extension')
            for k, v in ext.most_common():
                print(f'{v:8d}  .{k}')
            print('## top directories')
            top = Counter(n.rsplit('/', 1)[0] for n, *_ in names)
            for k, v in top.most_common(25):
                print(f'{v:8d}  {k}')
            return 0

        if args.maps or args.distinct:
            seen = set()
            for name, offset, size, got, status, exact in names:
                if args.maps and not name.lower().endswith('.umap'):
                    continue
                if name in seen:
                    continue
                seen.add(name)
                print(f'0x{offset:012x}  {name}')
            return 0

        for name, offset, size, got, status, exact in names:
            mark = '' if exact else ' (past payload)'
            print(f'0x{offset:012x} size={size:<9d} {status:<5s} {name}{mark}')
        return 0

    ap.error('nothing to do — pass --list, --resolve, --log or --self-test')


if __name__ == '__main__':
    sys.exit(main())
