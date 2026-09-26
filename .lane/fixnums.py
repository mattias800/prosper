#!/usr/bin/env python3
import pathlib
p = pathlib.Path('.lane/msg3.txt')
s = p.read_text()
s = s.replace("  render_draw_pass_rgba: 6,613 -> 6,004 lines, 44.6% -> 40.3% of the file",
              "  render_draw_pass_rgba: 6,613 -> 6,006 lines, 44.6% -> 40.3% of the file")
p.write_text(s)
q = pathlib.Path('.lane/msg4.txt')
t = q.read_text()
t = t.replace("  render_draw_pass_rgba: 6,064 -> 5,783 lines, 40.7% -> 38.7% of the file",
              "  render_draw_pass_rgba: 6,006 -> 5,723 lines, 40.3% -> 38.3% of the file")
q.write_text(t)
print('ok')
