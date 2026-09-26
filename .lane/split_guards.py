#!/usr/bin/env python3
"""Split the two `struct X { ... } var{...};` declarations in render_draw_pass_rgba into a
declaration line and a definition line, so the struct span can be moved as whole lines.
Idempotent: refuses if the expected text is not found."""
import sys, pathlib

p = pathlib.Path('prosper/tests/fixtures/render_runner.h')
lines = p.read_text().split('\n')

edits = [
    ("    } volume_attachment_view{dev};",
     ["    };", "    VolumeAttachmentViewGuard volume_attachment_view{dev};"]),
    ("    } volume_attempt{volume_color ? color_target->persistent_id : 0u};",
     ["    };",
      "    VolumeAttemptGuard volume_attempt{volume_color ? color_target->persistent_id : 0u};"]),
]

for old, new in edits:
    hits = [i for i, l in enumerate(lines) if l == old]
    if len(hits) != 1:
        sys.exit(f"expected exactly one {old!r}, found {len(hits)}")
    i = hits[0]
    lines[i:i+1] = new
    print(f"split line {i+1}: {old.strip()}")

p.write_text('\n'.join(lines))
