#!/usr/bin/env python3
"""Step 1: hoist render_draw_pass_rgba's local type vocabulary to namespace scope.

Splits the two `struct X { ... } var{...};` declarations into a declaration and a definition
first (so each struct is whole lines), then drives prosper/tools/refactor/hoist_spans.py, whose
inverse round trip proves nothing but position changed.
"""
from __future__ import annotations
import sys, pathlib, importlib.util

FILE = pathlib.Path('prosper/tests/fixtures/render_runner.h')
TOOL = pathlib.Path('prosper/tools/refactor/hoist_spans.py')
ANCHOR_TEXT = ('inline std::vector<uint8_t> render_draw_pass_rgba('
               'std::span<const BackendDraw> draws,')

spec = importlib.util.spec_from_file_location('hoist_spans', TOOL)
hs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hs)

SPLITS = [
    ("    } volume_attachment_view{dev};",
     ["    };", "    VolumeAttachmentViewGuard volume_attachment_view{dev};"]),
    ("    } volume_attempt{volume_color ? color_target->persistent_id : 0u};",
     ["    };",
      "    VolumeAttemptGuard volume_attempt{volume_color ? color_target->persistent_id : 0u};"]),
]

# head line, leading comment lines to take with it, trailing lines to take with it
TYPES = [
    ('    struct VolumeAttemptGuard {', 3, 0),
    ('    struct VolumeAttachmentViewGuard {', 2, 0),
    ('    struct DV {', 0, 2),          # the static_assert that pins DV trivially destructible
    ('    struct TextureUploadKey {', 0, 0),
    ('    struct TextureUploadKeyHash {', 0, 0),
    ('    struct TextureBindingKey {', 0, 0),
    ('    struct TextureBindingKeyHash {', 0, 0),
    ('    struct SharedTextureUpload {', 0, 0),
    ('    struct PersistentTextureKey {', 0, 0),
    ('    struct PersistentTextureKeyHash {', 0, 0),
    ('    struct PersistentTextureBinding {', 0, 0),
    ('    struct PersistentTextureImage {', 0, 0),
    ('    struct SharedBufferKey {', 0, 0),
    ('    struct SharedBufferKeyHash {', 0, 0),
    ('    struct SharedBufferUpload {', 0, 0),
    ('    struct SharedBufferArena {', 0, 0),
    ('    struct SharedTextureBinding {', 0, 0),
    ('    struct SharedDescriptorSetLayout {', 0, 0),
    ('    struct SharedPipelineLayout {', 0, 0),
    ('    struct BufferVerifyRecord {', 0, 0),
    ('    struct BufferRefMemoKey {', 0, 0),
    ('    struct BufferRefMemoKeyHash {', 0, 0),
    ('    struct BufferUploadSlotMemo {', 0, 0),
    ('    struct TextureUploadSlotMemo {', 0, 0),
    ('    struct TextureBindingSlotMemo {', 0, 0),
    ('    struct LastDescriptorSetMemo {', 0, 0),
    ('    struct LastPipelineLayoutMemo {', 0, 0),
    ('    struct EffectiveResource {', 3, 0),
    ('    struct ResourcePhaseTimer {', 3, 0),
    ('    struct BufferRangeUpload {', 0, 0),
    ('    struct Wave64Census {', 0, 0),
    ('    struct LastPipelineMemo {', 0, 0),
]

# Single lines that must move too, because a hoisted type names them.
SINGLES = [
    '    using TimingClock = std::chrono::steady_clock;',      # ResourcePhaseTimer
    '    constexpr size_t kMaxDescriptorSets = 8;',            # DV, and two memo arrays
]


def strip_code(s: str, state: dict) -> str:
    out, j = [], 0
    while j < len(s):
        if state['bc']:
            k = s.find('*/', j)
            if k < 0:
                break
            state['bc'] = False; j = k + 2; continue
        if s.startswith('//', j):
            break
        if s.startswith('/*', j):
            state['bc'] = True; j += 2; continue
        if s[j] == '"':
            j += 1
            while j < len(s):
                if s[j] == '\\': j += 2; continue
                if s[j] == '"': j += 1; break
                j += 1
            continue
        if s[j] == "'":
            j += 1
            while j < len(s):
                if s[j] == '\\': j += 2; continue
                if s[j] == "'": j += 1; break
                j += 1
            continue
        out.append(s[j]); j += 1
    return ''.join(out)


def find_unique(lines, text):
    hits = [i for i, l in enumerate(lines) if l == text]
    if len(hits) != 1:
        sys.exit(f"expected exactly one {text!r}, found {len(hits)}")
    return hits[0]


def struct_end(lines, start_idx):
    state = {'bc': False}
    for k in range(start_idx):
        strip_code(lines[k], state)
    depth, j = 0, start_idx
    while j < len(lines):
        for c in strip_code(lines[j], state):
            if c == '{': depth += 1
            elif c == '}': depth -= 1
        if j > start_idx and depth == 0:
            return j
        j += 1
    sys.exit(f"unterminated struct at line {start_idx + 1}")


HEADER = pathlib.Path('.lane/hdr_types.txt').read_text().split('\n')
if HEADER and HEADER[-1] == '':
    HEADER.pop()


def main() -> int:
    text = FILE.read_text()
    trailing = text.endswith('\n')
    lines = text.split('\n')
    if trailing:
        lines.pop()

    for old, new in SPLITS:
        i = find_unique(lines, old)
        lines[i:i + 1] = new
        print(f"split line {i + 1}: {old.strip()}")
    FILE.write_text('\n'.join(lines) + ('\n' if trailing else ''))

    spans = []
    for single in SINGLES:
        i = find_unique(lines, single)
        spans.append((i + 1, i + 1))
    for head, lead, trail in TYPES:
        h = find_unique(lines, head)
        end = struct_end(lines, h)
        if lines[end] != '    };':
            sys.exit("%r does not close with '    };' (got %r)" % (head, lines[end]))
        for c in lines[h - lead:h] if lead else []:
            if not c.strip().startswith('//'):
                sys.exit(f"lead line for {head!r} is not a comment: {c!r}")
        spans.append((h - lead + 1, end + 1 + trail))
    spans.sort()

    anchor = find_unique(lines, ANCHOR_TEXT) + 1
    produced = hs.plan(lines, anchor, spans, HEADER, [], 0)
    hs.verify(lines, produced, anchor, spans, HEADER, [], 0)
    FILE.write_text('\n'.join(produced) + ('\n' if trailing else ''))
    back = FILE.read_text().split('\n')
    if trailing:
        back.pop()
    hs.verify(lines, back, anchor, spans, HEADER, [], 0)
    moved = sum(hi - lo + 1 for lo, hi in spans)
    print(f"hoisted {len(spans)} span(s), {moved} lines, to line {anchor}; "
          f"inverse round trip reproduces the pre-hoist file")
    return 0


if __name__ == '__main__':
    sys.exit(main())
