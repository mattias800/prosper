#!/usr/bin/env python3
import pathlib
p = pathlib.Path('.lane/msg2.txt')
s = p.read_text()
s = s.replace("""Verification (Linux, RelWithDebInfo, RADV):
  cmake --build build-linux -j3                exit 0
  ctest --no-tests=error -j3                   651/651 passed, exit 0
""",
"""Verification (Linux, RelWithDebInfo, RADV):
  cmake --build build-linux -j3                exit 0
  ctest --no-tests=error -j3                   651/651 passed, exit 0

The first full run of that suite reported 650/651 with `profiling_access` (Timeout). That case has
a 30 s ctest TIMEOUT and reads perf_event_paranoid; it touches nothing in this diff. Another lane
was building and running its own suite at the time (box load average 65), and the case passes in
1.64 s on a targeted re-run and in the clean full run quoted above. Recorded rather than dropped,
because "re-run until green" is not a method -- the discriminator here is that the case is
unrelated to the diff AND fast in isolation, not merely that a second run was greener.
""")
p.write_text(s)
print('ok')
