# Pixel history — why is this pixel this colour?

This folder answers one question about a rendered frame, and it is the question most of
this project's black-render investigations actually turn on: **a pixel is the wrong
colour — did anything draw there at all, was it drawn and thrown away, did the shader run
and compute black, or did the shader compute a colour that the store then lost?**

Those answers send you to different files. Four more exist for cases that would otherwise be
misreported: `CLEARED_AFTER_DRAW` (a later clear wiped surviving draws),
`OUTPUT_UNTRUSTED` (the explaining event has an unbound pixel shader, so its output is
undefined and the tool refuses to guess), `VALUE_UNKNOWN` (RenderDoc recorded no value
for the event that would explain the pixel, so no colour can be read off it), and
`TRANSFER_WROTE_PIXEL` (a copy, blit, resolve or mip generation wrote it — no shader was
involved, and the value came from that operation's source). Telling them apart by inference —
draw censuses, disabling passes, bisecting state — is what has historically cost this
project hours per defect, and the census route has produced retracted issues. RenderDoc
answers it directly, per event, and `pixel_history.py` turns that into one call with one
verdict.

What lives here:

- `pixel_history.py` — the tool. Takes an `.rdc`, picks a pixel (the **brightest** one by
  default, because on a black frame the centre is exactly where nothing happened), and
  reports `NOTHING_DREW` / `ALL_REJECTED` / `SHADER_WROTE_BLACK` / `STORE_LOST_IT` with
  the per-event detail behind it. It refuses to report at all if the replay driver says it
  has no pixel-history support, because an empty history and an unsupported driver look
  identical and mean opposite things.
- `control.c` + `shaders/` — a construction with a **known answer**, in seven regions producing
  **five different verdicts**: the draw sequence (`PIXEL_WAS_WRITTEN`), a passing draw that
  computes black (`SHADER_WROTE_BLACK`), a passing draw through a `colorWriteMask=0` pipeline
  (`STORE_LOST_IT`), a region where every draw is discarded (`ALL_REJECTED`), and a blit landing on
  the finished target (`TRANSFER_WROTE_PIXEL`). A control that
  only ever produced one verdict would test the machinery and not the distinctions. Run
  `pixel_history.py --expect-control` against its capture on any new driver before trusting the
  tool there; it checks every region's verdict *and* every rejection reason, which is the part a
  final picture cannot see. Region **F** is a blit into the finished target, and it is built from a
  two-colour source for a reason worth copying: it lands black over a black draw, which is exactly
  the misattribution being guarded — and black over black is *invisible* to a self-check that only
  reads pixels. F' reads the half that can only be orange if the blit really landed. A control that
  cannot see its own arm guards nothing, the same way A' exists to prove A's first arm ran.
  F's blit also sits in a band **no other probe is inside**, which makes the control answer the one
  question about this channel nobody could check without RenderDoc: whether a copy appears in the
  history of pixels its destination rectangle does not cover. If it does, A/A'/B/C/E acquire a
  transfer event and `check_control` says so by name — because the tool blames the last transfer
  without asking whether it covered anything, and if non-covering copies are listed that would make
  `SHADER_WROTE_BLACK` unreachable on exactly the composited targets this tool exists for.
- `test_pixel_history.py` — the verdict logic, which is the part CI can reach. It also owns the
  cases no capture can contain on demand: `FakeModification` stands in for RenderDoc's
  `PixelModification` so a "RenderDoc has no value for this" event can be constructed by hand and
  run through `modification_event()`, the same builder the replay path uses.

## The control is not ceremony

It has earned itself **four times**, every time against an assumption that looked safe — and
each one was written down as a confident comment before it was measured. The fifth entry below
is the one the control could **not** reach, which is why the list is not the whole guard.

1. **RenderDoc populates `shaderOut` for events whose fragment shader never ran** — a scissored
   draw reports the *previous* draw's colour. On a game capture that is invisible: the number
   looks like a shader output and there is nothing to check it against. The control caught it
   because the value belonged to a draw whose colour was known to be different. Instrument
   trap 268.
2. **A clear IS a pixel-history event, and it passes.** The control's own comment had asserted
   the opposite. A region whose every draw was discarded read `SHADER_WROTE_BLACK` instead of
   `ALL_REJECTED`, because the clear sat in its history as a passing black event.
3. **And the two kinds of clear are not the same set.** Having keyed the fix on
   `ActionFlags::Clear`, a renderpass `loadOp = CLEAR` — the *common* form — still read
   `SHADER_WROTE_BLACK`, because it records `ResourceUsage::Clear` but carries no
   `ActionFlags::Clear`. The control could not see it, because it built only the flagged form:
   **it was validating the path the code handled.** It now builds both, and the check fails if
   either detection path is lost. Instrument trap 269.

4. **A pixel-history event has three kinds, not two, and the third one has no shader in it.**
   RenderDoc pushes copy/blit/resolve/genmips into the history through the *same*
   `clear || directWrite` branch as a clear — passed, no test evaluated — so a blit landing black
   was the last "passing draw" and read `SHADER_WROTE_BLACK`, sending the reader to resource
   binding and shaders for a pixel no shader wrote. The kind now comes from `GetUsage()`, and
   region F builds the case — though **that region has not yet been read back through RenderDoc**,
   which is not installed on the machine this was written on, so the transfer channel is verified by
   unit arms and by the control's rendered pixels and not by a live `--expect-control` run. Run one
   before trusting a `TRANSFER_WROTE_PIXEL` verdict on a new build; `check_control` is arranged to
   answer the open question in one line (see the control bullet above). Note what it is **not**: RenderDoc's `IsDirectWrite` also covers the
   RW-resource usages, and a compute shader writing a storage image genuinely *is* shader work —
   excluding "direct writes" wholesale would silently reclassify every compute write as a copy.
   The line is fixed-function transfer versus programmable write. Instrument trap 279, #3403.
5. **An absence of data arrives as a number, and that number is black.** RenderDoc marks a value
   it has nothing to report for by stamping a `0xdeadbeef` sentinel into it, which is read back
   through the same float union as a real colour: word 0 becomes about `-6.26e18` and the rest stay
   zero, so `max(rgb)` is **exactly 0.0** and the verdict came back `SHADER_WROTE_BLACK` — a
   confident answer derived from a value whose whole meaning is "I do not know", on exactly the
   black-pixel investigations people bring here. The control cannot construct this one: nothing a
   Vulkan program does makes RenderDoc have no data. So the sentinel is built **by hand** in
   `test_pixel_history.py`, in both of the shapes `SetInvalid()` might leave, and the verdict is now
   `VALUE_UNKNOWN`. Instrument trap 278, #3404. **It and (4) are adjacent and were measured not to
   share a cause, in the form that does not rest on an unverifiable premise:** neither fix ever
   produces the other's verdict. With only (5) fixed a blit reads `SHADER_WROTE_BLACK` if RenderDoc
   recorded its values and `VALUE_UNKNOWN` if it did not — never `TRANSFER_WROTE_PIXEL`; with only
   (4) fixed a sentinel still reads `SHADER_WROTE_BLACK`. One is the event's kind, the other is its
   value.

Generalise: **a tool that reads a value cannot tell you the value is meaningful.** Only a
construction whose answer you already know can — and only if that construction contains the hard
case. The first attempt at (2) fixed the control by moving the clear out of the captured frame,
which made it pass by removing the phenomenon. That is the failure mode to watch for: a control
that has been adjusted until it agrees with the code is no longer evidence about the code.

Event ids order events by submission and are what `PixelModification` sorts by, so the
"a clear landed after the last surviving draw" test is sound within one queue. It is not
meaningful across queues — a limitation inherited from RenderDoc's pixel history, not introduced
here.

## Boundary

This folder does not own capture (`renderdoccmd`, or prosper's own F9 bundles), replay of
prosper's own command stream (`tools/gpu_replay/`), or instrument capability checks
(`tools/doctor/`). It owns the question above and the control that makes its answers
trustworthy. Prosper's seeded F9 bundles retain guest-side state an `.rdc` cannot replace;
prefer them when the question is about prosper's translation rather than about what the GPU
was asked to do.
