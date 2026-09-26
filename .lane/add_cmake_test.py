#!/usr/bin/env python3
"""Register hoist_spans.py's selftest beside its siblings in prosper/CMakeLists.txt."""
import pathlib, sys

p = pathlib.Path('prosper/CMakeLists.txt')
s = p.read_text()
anchor = '''  add_test(NAME refactor_survey_classifier
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/refactor/survey_sizes.py" --selftest)
'''
addition = '''  # hoist_spans moves exact line spans to an earlier anchor in the same file and proves it by
  # INVERSE -- it cuts each moved span back out of its own output and requires the original file
  # back byte for byte. Three of its selftest's cases are CORRUPTED outputs that must be rejected,
  # because the thing worth gating is not that the mover works but that the checker can fail.
  # Textual and dependency-free, so it runs on a runner with no libclang rather than skipping.
  add_test(NAME refactor_hoist_spans
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/refactor/hoist_spans.py" --selftest)
'''
if 'refactor_hoist_spans' in s:
    sys.exit('already registered')
if anchor not in s:
    sys.exit('anchor not found')
s = s.replace(anchor, anchor + addition, 1)
p.write_text(s)
print('registered refactor_hoist_spans')
