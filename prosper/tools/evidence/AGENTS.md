# tools/evidence

Tools that check whether a capture actually shows what a claim says it shows.

Everything else in `tools/` helps you *produce* evidence — run a route, grab a frame, replay a
bundle. This folder is the other half: given a frame you are about to publish, does it support the
sentence you are about to write next to it? That question has its own failure modes, and they are
not the renderer's.

The founding case is `prerender_check.py` (instrument trap 230). A title that blits a full-screen
loading picture yields a frame that looks *better* than the emulator could plausibly render — which
reads as success rather than as the warning it is. One such capture passed a PR review, a `BLOG.md`
entry and `COMPATIBILITY.md` before a human recognised the artwork; no automated check existed that
could have caught it, and the frame was paired with a genuine "before" shot, so the A/B looked
rigorous while comparing a 3D render against a 2D blit.

**What belongs here:** checks whose subject is the *evidence*, not the emulator — "is this frame the
game's own artwork", "is this frame a duplicate of an earlier one", "does this capture actually
contain rendered geometry". They run against a candidate image plus a dump, need no build, and
should be cheap enough that nobody skips them before publishing.

**What does not:** anything that measures prosper's behaviour. A diagnostic that answers "what did
the renderer do" is a `PROSPER_*` switch or a `tools/gpu_*` tool, even when its output is an image.

**The rule these tools follow, and the reason they exit the way they do:** *could not check* must
never be reportable as *checked and clean*. `prerender_check.py` exits 0 only when it actually
compared something and nothing matched; when it finds no comparable asset it exits 1, distinct from
both the pass and the match. A tool in this folder that fails silently is worse than no tool, since
its whole purpose is to be trusted at the moment somebody is about to publish a claim.
