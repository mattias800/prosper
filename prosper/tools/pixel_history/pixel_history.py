#!/usr/bin/env python3
"""Answer "why is this pixel this colour" from an RDC, in one call.

Usage: pixel_history.py CAPTURE --output NEW_DIRECTORY [--pixel X,Y] [--target N]
                        [--expect-control]

Verdicts, each steering a different investigation:

  NOTHING_DREW        no DRAW touched the pixel (a clear may have) -- look upstream at
                      geometry, viewport, culling or whether the draw was submitted
  ALL_REJECTED        fragments arrived and every one was thrown out; the report names
                      which test did it (depth, stencil, scissor, discard, cull, ...)
  SHADER_WROTE_BLACK  the shader ran, passed every test, and computed black -- look at
                      resource binding, textures, uniforms, the shader itself
  STORE_LOST_IT       the shader computed a non-black colour and the target ended black
                      anyway -- look at blend state, write masks, later overdraw
  PIXEL_WAS_WRITTEN   a draw survived and the pixel is not black; not a defect here
  CLEARED_AFTER_DRAW  draws survived and a later clear wiped them -- what you see is the
                      clear, not the shading
  OUTPUT_UNTRUSTED    the explaining event's shader output is undefined (an unbound pixel
                      shader), so "computed black" and "store lost it" cannot be separated
  VALUE_UNKNOWN       RenderDoc records NO VALUE for the event that would explain the pixel
                      (its 0xdeadbeef "no information" sentinel, or nothing at all), so no
                      colour can be read off it -- the tool names no cause it cannot establish
  TRANSFER_WROTE_PIXEL  a copy, blit, resolve or mip generation wrote this pixel, not a
                      shader -- the value came from that operation's SOURCE, so the
                      investigation belongs wherever the source was produced

An event is one of three kinds, not two. Clears pass and evaluate no test, so they are
never the subject of a verdict -- only the ground one is stated against (trap 269). Nor are
fixed-function TRANSFERS a draw: RenderDoc pushes copy/blit/resolve/genmips into the
history through the same "no test evaluated" branch as a clear, so a blit landing black was
read as "the shader computed black" (trap 279). A compute shader writing a storage image is
NOT in that set and must not be: that genuinely is shader work, and the reader should be
sent to the shader for it. The distinction is fixed-function transfer versus programmable
write, never "direct" versus "not".

Run against `pixel_history_control` first on any new driver: `--expect-control` checks
this tool's own reading against a construction with a known answer.

Exit 0 a verdict was produced, 1 replay/analysis failed, 2 usage.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import traceback

# Rejections that happen BEFORE the fragment shader runs. RenderDoc still populates
# shaderOut for these, and the value is the previous event's residue -- measured on the
# known-answer control, where a scissored white draw reported the earlier draw's blue.
# Reporting that as "the shader computed blue" would be a confident lie, so these
# events carry no shader output at all.
# Rejections RenderDoc's Vulkan pixel history evaluates and returns on BEFORE running the
# fragment shader (vk_pixelhistory.cpp: scissor at :4470, sample mask at :4478, each
# `return`ing immediately). shaderOut is still populated for these and carries the previous
# event's residue -- see instrument trap 268.
PRE_FRAGMENT = ("scissorClipped", "backfaceCulled", "depthClipped", "viewClipped",
                "predicationSkipped", "sampleMasked")
# Rejections that normally happen AFTER the shader, so their shaderOut is real and is often
# the whole answer. The exception is a shader declaring EarlyFragmentTests, where RenderDoc
# evaluates these first (vk_pixelhistory.cpp:4525) and the output is residue again. The
# Python API exposes no per-event early-fragment flag, so these are reported with a caveat
# rather than suppressed -- and no verdict is ever derived from a rejected event's output.
POST_FRAGMENT = ("depthTestFailed", "stencilTestFailed", "depthBoundsFailed")
# Not a rejection at all: Passed() excludes unboundPS (data_types.h:2759), so such an event
# arrives passed=True while its own documentation says the output "may have undefined
# values" (:2664). Trusting it is trap 268 one list entry over, so it is called out by name
# and its output is never used.
UNDEFINED_OUTPUT = ("unboundPS",)
REJECTIONS = PRE_FRAGMENT + POST_FRAGMENT + ("shaderDiscarded",) + UNDEFINED_OUTPUT
# Vulkan-only note: predicationSkipped is set only by the D3D11 backend, and on D3D11/12 the
# sample mask is the output-merger mask applied AFTER the shader. This tool reads prosper's
# Vulkan captures; a D3D11 capture would need sampleMasked moved out of PRE_FRAGMENT.
BLACK = 1.0 / 255.0
# RenderDoc marks a ModificationValue it has no data for by stamping a 0xdeadbeef sentinel
# into it (ModificationValue::SetInvalid()), not with a null, a flag or an exception -- and
# that value is read back through the same float union as a real colour. So "I have no
# information" arrives as four well-formed floats: the first is the sentinel's bit pattern
# (about -6.26e18) and the rest are zero, which makes max(rgb) exactly 0.0. That is
# byte-identical to a shader that computed black, and black is the value people bring to
# this tool -- so an absence of data was being reported as SHADER_WROTE_BLACK, a confident
# answer derived from a value whose whole meaning is "I do not know". #3404, and trap 268
# one field over: a plausible, well-formed number that means something else.
# The ResourceUsage values that mean "a fixed-function transfer WROTE this target". This is
# RenderDoc's IsDirectWrite set minus the RW-resource (shader storage) usages, and minus the
# read halves: CopySrc and ResolveSrc are this target being READ and explain nothing about
# its own contents. Dropping the RW usages is deliberate and is the point of the split -- a
# compute shader writing a storage image is shader work, and a reader told "a copy did this"
# about a dispatch would be steered away from the file that holds the answer. Names rather
# than values, resolved with getattr, so a RenderDoc build missing one is harmless. #3403.
TRANSFER_USAGES = ("Copy", "CopyDst", "Resolve", "ResolveDst", "GenMips")
NO_INFORMATION = 0xdeadbeef
NO_INFORMATION_AS_FLOAT = struct.unpack("<f", struct.pack("<I", NO_INFORMATION))[0]


def suppress_shader_output(rejected_by):
    """Is this event's shaderOut another draw's residue, or undefined?

    Extracted from the replay path deliberately: that path needs RenderDoc and a GPU, so
    nothing in CI can reach it. Leaving the decision inline meant deleting PRE_FRAGMENT
    left every test green while the tool began reporting a neighbouring draw's colour as
    this draw's shader output -- measured, not hypothetical.
    """
    return any(f in PRE_FRAGMENT or f in UNDEFINED_OUTPUT for f in rejected_by)


def carries_no_information(mv):
    """Is this ModificationValue the sentinel rather than a colour?

    Three routes, ANY of which is enough, because what the Python binding exposes is not
    guaranteed across RenderDoc builds: RenderDoc's own IsValid() if there is one, the
    integer view of the union, and the float view. A colour whose first component is exactly
    the sentinel's bit pattern is not one any render target can hold, so the float route is
    a sound test rather than a heuristic.

    Deliberately a UNION and not a precedence. An earlier draft let IsValid() returning True
    short-circuit past the bit-pattern routes, which made the one route nobody here can run
    against a real binding -- RenderDoc is not installed on this machine -- the only one
    able to re-open #3404. Every route can now only ever ADD a detection, so being wrong
    about any of them costs a detection rather than restoring the defect.
    """
    if mv is None:
        return True
    is_valid = getattr(mv, "IsValid", None)
    if callable(is_valid):
        try:
            if not bool(is_valid()):
                return True
        except Exception:
            pass
    col = getattr(mv, "col", None)
    if col is None:
        return True
    ints = list(getattr(col, "uintValue", None) or [])
    if ints and int(ints[0]) == NO_INFORMATION:
        return True
    floats = list(getattr(col, "floatValue", None) or [])
    if floats and float(floats[0]) == NO_INFORMATION_AS_FLOAT:
        return True
    return False


def colour_values(mv):
    """This event's RGBA, or None when the API carries no value for it.

    None is the whole point, and it is why every consumer below tests for it instead of
    indexing: "the value was black" and "there is no value" send a reader to different
    files, so the second must be representable rather than encoded as a number something
    downstream will read as data. An empty value is the same finding as the sentinel.
    """
    if carries_no_information(mv):
        return None
    col = getattr(mv, "col", None)
    return [float(x) for x in (list(getattr(col, "floatValue", None) or []))][:4] or None


def rejection_reasons(m):
    """Which of the rejection flags this PixelModification carries."""
    return [f for f in REJECTIONS if getattr(m, f, False)]


def modification_event(m, kind, usage=None):
    """One event record from a RenderDoc PixelModification.

    `kind` is "draw", "clear" or "transfer" and comes from GetUsage(), never from the
    event itself: a clear and a blit both arrive passed=True with no test evaluated, and
    are distinguishable from a draw only by what the API says they did to this resource.
    `usage` is the RenderDoc usage name behind a clear or a transfer, so the verdict can
    say "CopyDst" rather than "something".

    Module level, and used by BOTH the analysed pixel and the control regions, for the
    reason suppress_shader_output() gives: the replay path needs RenderDoc and a GPU, so
    nothing in CI can reach it, and two hand-copied copies of this dict drift apart. It
    also lets a test build a stand-in `m` -- which is the only way to exercise the
    no-information sentinel, since no control program can make RenderDoc have no data.
    """
    rejected = rejection_reasons(m)
    suppressed = suppress_shader_output(rejected)
    pre, post = colour_values(m.preMod), colour_values(m.postMod)
    # Suppressed rather than reported as zero: "the shader ran and produced nothing" and
    # "the shader never ran" are different findings.
    shader = None if suppressed else colour_values(m.shaderOut)
    no_value = [n for n, v in (("preMod", pre), ("postMod", post)) if v is None]
    if not suppressed and shader is None:
        no_value.append("shaderOut")
    return {"eventId": int(m.eventId),
            "kind": kind,
            "usage": usage,
            "passed": bool(m.Passed()),
            "rejected_by": rejected,
            "shaderOut": shader,
            "shader_output_suppressed": suppressed,
            "preMod": pre,
            "postMod": post,
            # Named, not merely absent: a reader of the report has to be able to tell a
            # suppressed shader output ("no fragment ran") from a value RenderDoc simply
            # never recorded, and a verdict may rest on which one it was.
            "no_value": no_value}


def explaining_touch(events, after_eid):
    """The newest passing clear/transfer after `after_eid` that can explain the pixel.

    Newest first, because the last thing to write a pixel is what the reader is looking at.
    A clear that provably changed nothing is stepped over -- blaming it would move the
    reader off the event that did write -- while a transfer is never stepped over: it is by
    construction the last writer of its destination region, and a copy that lands the same
    value it found is still where that value came from.

    `after_eid` None means "over the whole history". Returns None when nothing later than
    `after_eid` explains the pixel.

    **PRESENCE INSIDE A WINDOW IS NOT THE SAME AS BEING THE LAST WRITER, and every site that
    reasons about a transfer must ask this function rather than filter a list.** That rule
    is written here because four separate places got it wrong in four review rounds, each in
    a way that looked like a different bug: "a transfer is in the history" where the question
    was "is it last" (the ALL_REJECTED note announced a copy that PRECEDED the draws); "a
    transfer exists" where it was "and nothing cleared over it" (the no-draw path named a
    copy a later clear had wiped); "a transfer is under the clear" where it was "and nothing
    between them explains it first" (the unrecorded-clear text named a copy that an
    intervening clear had already wiped). Filtering by position answers a question about
    ORDER; this function answers the question about ATTRIBUTION, and they differ exactly
    whenever something in between did the writing.

    Read that as a rule about ATTRIBUTION, not as "never ask whether a transfer is
    present". Presence is genuinely the question in three places and they must stay:
    the `if moved` guard below, and check_control()'s checks that the CONSTRUCTION
    contains a transfer -- which exist precisely to catch the detection being lost.
    """
    for e in sorted((x for x in events
                     if x["passed"] and x["kind"] in ("clear", "transfer")
                     and (after_eid is None or x["eventId"] > after_eid)),
                    key=lambda x: x["eventId"], reverse=True):
        if e["kind"] == "transfer":
            return e
        if e["preMod"] is None or e["postMod"] is None:
            return e            # a clear whose effect cannot be established
        if e["preMod"][:3] != e["postMod"][:3]:
            return e            # a clear that demonstrably changed the pixel
    return None


def classify(events):
    """One verdict from the event list. Order matters: the earliest true statement wins."""
    if not events:
        return "NOTHING_DREW", "No event in this frame touched the pixel."
    # A clear is a pixel-history event and it PASSES (trap 269), with no test evaluated.
    # Reasoning about the pixel from it would answer a question nobody asked: on a target
    # cleared to black, "the last passing event computed black" is the CLEAR, and reporting
    # SHADER_WROTE_BLACK sends the reader to a shader that never ran. Clears are therefore
    # never the subject of a verdict -- only ever the ground a verdict is stated against.
    drawn = [e for e in events if e["kind"] == "draw"]
    moved = [e for e in events if e["kind"] == "transfer" and e["passed"]]
    if not drawn:
        # A target whose whole content arrived by copy is not a target nothing touched, and
        # saying "only clear events touched this pixel" about one is a plain falsehood --
        # prosper's own renderer composites and blits, so this is reachable. But the copy is
        # only the answer if nothing cleared over it afterwards, which is why this walks the
        # history instead of taking the newest transfer: the latter reported "a copy wrote
        # this" about a pixel a later clear had wiped, reintroducing -- with no draw in
        # sight -- exactly the confident misattribution this file exists to prevent.
        last = explaining_touch(events, None) if moved else None
        if last is not None and last["kind"] == "transfer":
            return "TRANSFER_WROTE_PIXEL", (
                f"no draw touched this pixel; event {last['eventId']} "
                f"({last['usage'] or 'a transfer'}) wrote it. What you see is that "
                f"operation's SOURCE -- look at how the source was produced, not at a "
                f"shader that never ran here.")
        if last is not None and (last["preMod"] is None or last["postMod"] is None):
            return "VALUE_UNKNOWN", (
                f"no draw touched this pixel and a copy wrote it, then event "
                f"{last['eventId']} cleared this target with no value recorded -- whether "
                f"the clear wiped the copy cannot be established.")
        if last is not None:
            return "NOTHING_DREW", (
                f"no draw touched this pixel: a copy wrote it and event "
                f"{last['eventId']} cleared over the copy afterwards, so what you see is "
                f"the clear.")
        cleared = "; the pixel holds its clear value" if events else ""
        return "NOTHING_DREW", f"only clear events touched this pixel{cleared}."
    # By eventId, not by list position: explaining_touch() below orders the same question
    # that way, and one function must not hold two notions of "last".
    passed = sorted((e for e in drawn if e["passed"]), key=lambda e: e["eventId"])
    if not passed:
        why = {}
        for e in drawn:
            for r in e["rejected_by"]:
                why[r] = why.get(r, 0) + 1
        named = ", ".join(f"{k}x{v}" for k, v in sorted(why.items(), key=lambda kv: -kv[1]))
        # The rejections are still the finding -- but if a transfer then wrote the pixel,
        # not saying so leaves the reader to conclude the colour came from the clear. It has
        # to be the LAST touch, not merely present: a copy that PRECEDED the draws, or one a
        # later clear wiped, is not what the reader is looking at, and calling either "a
        # later transfer [that] then wrote this pixel" is a plain falsehood.
        after = explaining_touch(events, max(e["eventId"] for e in drawn))
        copied = (f" A later transfer ({after['usage'] or 'copy'}, event "
                  f"{after['eventId']}) then wrote this pixel."
                  if after is not None and after["kind"] == "transfer" else "")
        return "ALL_REJECTED", (f"{len(drawn)} draw(s) reached the pixel, none survived: "
                                f"{named or 'no reason flagged'}.{copied}")

    # Everything that touched this pixel AFTER the last surviving draw, newest first: the
    # first one that can explain the pixel wins. Blaming the shader for a pixel a later
    # clear wiped is the same error as blaming it for the ground clear, one ordering along;
    # a clear that provably changed nothing explains nothing either, so it is stepped over
    # rather than taking the blame from the draw that did the work.
    #
    # eventId ordering is RenderDoc's own submission order and is what PixelModification
    # sorts by; it is not meaningful ACROSS queues, a limitation this inherits from pixel
    # history rather than introduces.
    e = explaining_touch(events, passed[-1]["eventId"])
    # A transfer is blamed without asking whether it changed the pixel, and a clear is not.
    # The asymmetry is deliberate and lives in explaining_touch(): every frame clears its
    # targets, so a ubiquitous no-op clear must not take the blame from the draw, while a
    # copy is occasional and IS the thing that last wrote its destination region.
    if e is not None and e["kind"] == "transfer":
        return "TRANSFER_WROTE_PIXEL", (
            f"{len(passed)} draw(s) passed, then event {e['eventId']} "
            f"({e['usage'] or 'a transfer'}) copied over them -- the pixel you see was "
            f"written by that operation, not computed by a shader here. Look at how "
            f"its SOURCE was produced.")
    # A late clear whose values RenderDoc did not record cannot be shown to have changed
    # this pixel -- and cannot be shown not to have. Falling through would name the draw
    # below it, which is the misattribution this section exists to prevent, arrived at by
    # an absence of evidence instead of by evidence.
    if e is not None and (e["preMod"] is None or e["postMod"] is None):
        # Name the OTHER live possibility, which is not the draw. If the clear turns out to
        # have changed nothing, the explaining event is whatever wrote last underneath it --
        # and a transfer sitting between the last passing draw and this clear is that event,
        # not the draw. Saying only "inspect the clear" leaves the reader to assume the draw
        # is the alternative, which is the one thing it cannot be here.
        # Asked, not filtered -- see explaining_touch()'s docstring. A copy UNDER this
        # clear is only the alternative if nothing between them already explains the pixel:
        # a clear at 25 that demonstrably wiped a copy at 20 means 25, not 20, is what you
        # are looking at if 30 turns out to have changed nothing.
        under = explaining_touch([x for x in events if x["eventId"] < e["eventId"]],
                                 passed[-1]["eventId"])
        alt = (f" If it did not, event {under['eventId']} "
               f"({under['usage'] or 'a transfer'}) is what you are looking at, not the "
               f"draw." if under is not None and under["kind"] == "transfer" else "")
        return "VALUE_UNKNOWN", (
            f"event {e['eventId']} cleared this target after the last surviving draw "
            f"and RenderDoc records no value for it, so whether it wiped this pixel "
            f"cannot be established -- inspect that event directly.{alt}")
    if e is not None:
        return "CLEARED_AFTER_DRAW", (
            f"{len(passed)} draw(s) passed, then event {e['eventId']} cleared the "
            f"target over them -- the pixel you see is the clear, not the shading.")

    # Only the LAST passing draw can explain the pixel's final state. An earlier bright
    # writer that was legitimately overdrawn -- a white sky behind a black object -- is not
    # evidence of a lost store, and reading it as one made SHADER_WROTE_BLACK unreachable
    # on any pixel with history, which is most of a real frame.
    final = passed[-1]
    # The headline of #3404: this is where the sentinel used to become a colour. max() over
    # a value that means "no information" is arithmetic on a non-number, and it produced
    # exactly 0.0 -- the answer the reader was already afraid of.
    if final["postMod"] is None:
        return "VALUE_UNKNOWN", (
            f"the last passing event ({final['eventId']}) carries no recorded value for "
            f"this pixel, so the colour it left cannot be read and no cause is being "
            f"named -- inspect that event directly, or pick another pixel.")
    lit = max(final["postMod"][:3])
    if lit > BLACK:
        return "PIXEL_WAS_WRITTEN", f"{len(passed)} of {len(events)} events passed; final colour is not black."
    if final["shader_output_suppressed"]:
        named = ", ".join(final["rejected_by"]) or "unknown"
        return "OUTPUT_UNTRUSTED", (
            f"the last passing event ({named}) has no trustworthy shader output, so "
            f"'computed black' and 'store lost it' cannot be separated here -- pick "
            f"another pixel, or inspect that event directly.")
    # Suppression returned above, so a missing output here means RenderDoc recorded none.
    if final["shaderOut"] is None:
        return "VALUE_UNKNOWN", (
            f"the pixel is black and the last passing event ({final['eventId']}) has no "
            f"recorded shader output, so 'the shader computed black' and 'the store lost "
            f"it' cannot be separated -- which is the whole distinction this tool is for.")
    if max(final["shaderOut"][:3]) > BLACK:
        return "STORE_LOST_IT", (
            "the last passing event computed a non-black colour and the target is black "
            "anyway -- blend state, write mask, or a later event that is not in this list.")
    return "SHADER_WROTE_BLACK", (
        f"{len(passed)} event(s) passed every test and the last one computed black -- "
        f"resource binding, textures, uniforms or the shader itself.")


def embedded():
    req = json.loads(Path(os.environ["PROSPER_PIXHIST_REQUEST"]).read_text())
    out = Path(req["output"])
    result = {"status": "FAILED"}
    try:
        import renderdoc as rd

        cap = rd.OpenCaptureFile()
        if cap.OpenFile(req["capture"], "", None) != rd.ResultCode.Succeeded:
            raise RuntimeError("OpenFile failed; not an RDC this build can read")
        st, ctl = cap.OpenCapture(rd.ReplayOptions(), None)
        if st != rd.ResultCode.Succeeded:
            raise RuntimeError(f"OpenCapture failed: {st}")

        # Ask the API whether it supports this at all. Without it an unsupported driver
        # returns an empty history, which is byte-identical to "nothing drew here" -- the
        # exact confusion this tool exists to remove.
        props = ctl.GetAPIProperties()
        if not props.pixelHistory:
            raise RuntimeError(
                "this replay driver reports no pixel-history support; an empty result here "
                "would be indistinguishable from 'nothing drew', so refusing to report one")

        actions = []

        def walk(nodes):
            for a in nodes:
                if a.flags & rd.ActionFlags.Drawcall:
                    actions.append(a)
                walk(a.children)

        walk(ctl.GetRootActions())
        if not actions:
            raise RuntimeError("no draw actions in capture")
        ctl.SetFrameEvent(actions[-1].eventId, True)

        # Prefer the targets actually BOUND at the selected event. TextureCategory.ColorTarget
        # is set for VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT as well as COLOR_ATTACHMENT
        # (vk_info.cpp:2601-2603), so scanning every texture can hand back a transient
        # attachment nothing presents -- and its empty history reads as NOTHING_DREW.
        by_id = {t.resourceId: t for t in ctl.GetTextures()}
        bound = [by_id[o.resource] for o in ctl.GetPipelineState().GetOutputTargets()
                 if o.resource != rd.ResourceId.Null() and o.resource in by_id]
        source, targets = "bound output targets at the last draw", bound
        if not targets:
            targets = [t for t in ctl.GetTextures()
                       if t.creationFlags & rd.TextureCategory.ColorTarget]
            source = "texture scan (no bound output target; may include transient attachments)"
        if not targets:
            raise RuntimeError("no colour targets in capture")
        if not 0 <= req["target"] < len(targets):
            raise RuntimeError(
                f"--target {req['target']} out of range: {len(targets)} target(s) available "
                f"from {source}. Silently clamping would analyse a different image than asked.")
        tex = targets[req["target"]]

        # Save the target first: selection needs to see the image, and the image is also
        # the evidence a reader wants beside the verdict.
        image = str(out / "target.png")
        save = rd.TextureSave()
        save.resourceId = tex.resourceId
        save.destType = rd.FileType.PNG
        if ctl.SaveTexture(save, image) != rd.ResultCode.Succeeded:
            image = None

        pixel, how = req["pixel"], "requested"
        if pixel is None:
            pixel, how = (tex.width // 2, tex.height // 2), "centre (no content guidance)"
            if image:
                try:
                    from PIL import Image
                    im = Image.open(image).convert("RGB")
                    # The brightest pixel, not the centre: on a mostly-black frame the
                    # centre is exactly where nothing happened, and its empty history
                    # says nothing about why the frame is black.
                    small = im.resize((min(im.width, 256), min(im.height, 256)))
                    sx, sy = im.width / small.width, im.height / small.height
                    best, bxy = -1, None
                    for y in range(small.height):
                        for x in range(small.width):
                            r_, g_, b_ = small.getpixel((x, y))
                            lum = r_ * 2 + g_ * 5 + b_
                            if lum > best:
                                best, bxy = lum, (x, y)
                    pixel = (int(bxy[0] * sx), int(bxy[1] * sy))
                    how = f"brightest pixel (luma {best})"
                except Exception as exc:
                    how = f"centre (content guidance unavailable: {exc})"

        # Which events CLEARED this target. Keyed on ResourceUsage, not ActionFlags, because
        # they are not the same set and the difference is the common case: a renderpass
        # `loadOp = CLEAR` records ResourceUsage::Clear (vk_cmd_funcs.cpp:2336) while its
        # action carries only PassBoundary|BeginPass (:2351) -- ActionFlags::Clear is set
        # solely by vk_draw_funcs.cpp, i.e. vkCmdClear*. Keying on the flag therefore missed
        # every loadOp clear, and a loadOp-cleared black target that nothing drew to was
        # reported as SHADER_WROTE_BLACK: the exact string this distinction exists to stop.
        # This is character-for-character RenderDoc's own predicate (vk_pixelhistory.cpp:4636).
        # The same GetUsage() pass also finds the THIRD kind of event. RenderDoc pushes a
        # copy, blit, resolve or mip generation into the pixel history through the same
        # "passed, no test evaluated" branch as a clear (IsDirectWrite), so without this a
        # blit landing black was the last passing event and read as SHADER_WROTE_BLACK --
        # sending a reader to resource binding and shaders for a pixel no shader wrote.
        # Trap 279. The RW-resource half of IsDirectWrite is deliberately NOT here: see
        # TRANSFER_USAGES.
        transfer_usage = {n: getattr(rd.ResourceUsage, n, None) for n in TRANSFER_USAGES}
        kinds = {}
        for u in ctl.GetUsage(tex.resourceId):
            eid = int(u.eventId)
            if u.usage == rd.ResourceUsage.Clear:
                kinds[eid] = ("clear", "Clear")
            elif eid not in kinds:
                for name, value in transfer_usage.items():
                    if value is not None and u.usage == value:
                        kinds[eid] = ("transfer", name)
                        break

        def kind_of(eid):
            return kinds.get(int(eid), ("draw", None))

        if not (0 <= pixel[0] < tex.width and 0 <= pixel[1] < tex.height):
            raise RuntimeError(
                f"pixel {tuple(pixel)} is outside the {tex.width}x{tex.height} target; an "
                f"out-of-bounds history is empty and would read as NOTHING_DREW")
        hist = ctl.PixelHistory(tex.resourceId, pixel[0], pixel[1],
                                rd.Subresource(0, 0, 0), rd.CompType.Typeless)
        events = [modification_event(m, *kind_of(m.eventId)) for m in hist]
        verdict, reason = classify(events)

        control = {}
        if req.get("expect_control"):
            for name, (cx, cy), _ in CONTROL_REGIONS:
                ch = ctl.PixelHistory(tex.resourceId, cx, cy, rd.Subresource(0, 0, 0),
                                      rd.CompType.Typeless)
                ce = [modification_event(m, *kind_of(m.eventId)) for m in ch]
                control[name] = {"verdict": classify(ce)[0], "events": ce}

        result = {"status": "REPLAYED", "verdict": verdict, "reason": reason,
                  "control_regions": control,
                  "pixel": list(pixel), "pixel_choice": how,
                  "target": {"id": str(tex.resourceId), "width": tex.width,
                             "height": tex.height, "chosen_from": source,
                             "candidates": len(targets)},
                  "target_image": image, "draw_count": len(actions),
                  "events": events,
                  "api": {"pixelHistory": bool(props.pixelHistory),
                          "shaderDebugging": bool(props.shaderDebugging),
                          # Which transfer usages this RenderDoc build actually exposes. A
                          # name it does not have resolves to None and is silently skipped,
                          # which would narrow the copy/blit detection without saying so --
                          # so the coverage is reported rather than assumed.
                          "transfer_usages": sorted(n for n, v in transfer_usage.items()
                                                    if v is not None)}}
    except Exception:
        result = {"status": "FAILED", "error": traceback.format_exc()}
    (out / "pixel_history.json").write_text(json.dumps(result, indent=2) + "\n")
    # qrenderdoc would otherwise enter its UI event loop after the startup script.
    os._exit(0 if result["status"] == "REPLAYED" else 1)


# The control's five regions, restated here independently of the C source that builds them.
# Each exists to construct ONE verdict: a control that only ever produces one tests the
# machinery and not the distinctions the tool is for.
CONTROL_REGIONS = [
    ("A  sequence", (16, 16), "PIXEL_WAS_WRITTEN"),
    ("A' arm-1 only", (16, 36), "PIXEL_WAS_WRITTEN"),
    ("B  black draw", (48, 16), "SHADER_WROTE_BLACK"),
    ("C  no write", (16, 48), "STORE_LOST_IT"),
    ("E  all killed", (48, 48), "ALL_REJECTED"),
    ("F  blit black", (54, 4), "TRANSFER_WROTE_PIXEL"),
    ("F' blit orange", (61, 4), "TRANSFER_WROTE_PIXEL"),
]


def check_control(regions):
    """Check a full control reading: every verdict, and the suppression trap 268 needs.

    `regions` maps region name -> {"verdict": str, "events": [...]}.
    """
    problems = []
    for name, _, want in CONTROL_REGIONS:
        got = regions.get(name)
        if got is None:
            problems.append(f"{name}: region missing from the reading")
        elif got["verdict"] != want:
            problems.append(f"{name}: expected {want}, read {got['verdict']}")

    seq = regions.get("A  sequence")
    if seq:
        reasons = [r for e in seq["events"] for r in e["rejected_by"]]
        for needed in ("depthTestFailed", "shaderDiscarded"):
            if needed not in reasons:
                problems.append(f"A: {needed} was constructed but not reported; a rejection "
                                f"is being misattributed")
        # A's own scissored arm runs BEFORE its last surviving draw; the other regions'
        # draws are all later. Without the ordering check this requirement is vacuous --
        # B, C and E supply a scissorClipped at A whether or not arm 2 exists.
        drawn = [e for e in seq["events"] if e["kind"] == "draw"]
        survived = [e for e in drawn if e["passed"]]
        cutoff = survived[-1]["eventId"] if survived else 0
        if not any("scissorClipped" in e["rejected_by"] and e["eventId"] < cutoff
                   for e in drawn):
            problems.append("A: no scissorClipped event before A's last surviving draw, so "
                            "arm 2 is missing or misattributed (later regions' draws also "
                            "scissor-clip here, which is why the ordering matters)")
        # Trap 268: the scissored arm's shaderOut is the PREVIOUS draw's colour. If any
        # positionally-rejected event still carries an output, the suppression is off and the
        # tool is one step from reporting a neighbouring draw's colour as this one's.
        leaked = [e["eventId"] for e in seq["events"]
                  if suppress_shader_output(e["rejected_by"]) and e["shaderOut"] is not None]
        if leaked:
            problems.append(f"A: events {leaked} were rejected before the fragment shader ran "
                            f"and still reported a shader output; the stale-value suppression "
                            f"is not working")
    killed = regions.get("E  all killed")
    if killed:
        # E's verdict alone is reachable with its own draw deleted: the seven scissored
        # neighbours produce ALL_REJECTED by themselves, and the C self-check sees black
        # either way. Name the arm the region exists for.
        if not any("shaderDiscarded" in e["rejected_by"] for e in killed["events"]):
            problems.append("E: no shaderDiscarded event; the region's own draw is missing "
                            "and its verdict is coming from scissored neighbours alone")
        # E is the control's guard for trap 269. If the clear stopped being recognised, its
        # verdict would silently become SHADER_WROTE_BLACK with the clear as the subject.
        clears = [e for e in killed["events"] if e["kind"] == "clear"]
        # TWO clears, because the control builds two FORMS of clear and they are recorded
        # differently: vkCmdClearColorImage carries ActionFlags::Clear, a loadOp clear does
        # not. Requiring only one let the flag-keyed detection pass while every loadOp clear
        # in a real capture went unrecognised.
        if len(clears) < 2:
            problems.append(f"E: {len(clears)} clear event(s) recognised, expected 2 (the "
                            f"transfer clear and the renderpass loadOp clear). One detection "
                            f"path is broken; loadOp clears are the common form and carry "
                            f"ResourceUsage::Clear but NOT ActionFlags::Clear")
    # The blit regions, derived from CONTROL_REGIONS rather than restated: a renamed region
    # must not silently drop out of the checks below, which are what stands in for a live
    # --expect-control run this machine cannot perform.
    blitted = [name for name, _, want in CONTROL_REGIONS if want == "TRANSFER_WROTE_PIXEL"]
    # F is the trap-279 guard. Its verdict alone is not enough: a tool that lost the
    # transfer detection would call F's blit a draw and report SHADER_WROTE_BLACK over the
    # black half and PIXEL_WAS_WRITTEN over the orange half, so both halves are required to
    # name the transfer AND to contain one, as the last passing event in the history.
    for name in blitted:
        got = regions.get(name)
        if got is None:
            continue          # the missing-region problem is already reported above
        moved = [e for e in got["events"] if e["kind"] == "transfer"]
        if not moved:
            problems.append(f"{name}: no transfer event recognised in the history, so the "
                            f"blit is being read as a draw -- the copy/blit/resolve "
                            f"detection is lost")
            continue
        passing = [e for e in got["events"] if e["passed"]]
        if passing and passing[-1]["kind"] != "transfer":
            problems.append(f"{name}: the blit is not the last passing event "
                            f"(last is {passing[-1]['kind']} {passing[-1]['eventId']}), so "
                            f"the construction did not land where it was aimed")
    # The assumption the whole transfer channel rests on, made self-detecting: that a copy
    # appears only in the history of pixels its DESTINATION RECTANGLE covers. Nobody can
    # check that without RenderDoc, so the control is arranged to answer it -- F's blit
    # covers a band no other probe is inside, and if non-covering direct writes are listed
    # anyway, every region below acquires a transfer event and flips to
    # TRANSFER_WROTE_PIXEL. Reported as one named hypothesis rather than as three verdicts
    # the reader has to notice changed.
    for name in [n for n, _, want in CONTROL_REGIONS
                 if want != "TRANSFER_WROTE_PIXEL"]:
        got = regions.get(name)
        if got and any(e["kind"] == "transfer" for e in got["events"]):
            problems.append(f"{name}: a transfer event appears in a region OUTSIDE the blit "
                            f"rectangle, so RenderDoc lists direct writes that do not cover "
                            f"the pixel -- this tool blames the last transfer without asking "
                            f"whether it covered anything, which would make SHADER_WROTE_BLACK "
                            f"unreachable on any composited target. Do not trust a "
                            f"TRANSFER_WROTE_PIXEL verdict on this build until it is fixed")

    black, orange = ((regions.get(blitted[0]), regions.get(blitted[1]))
                     if len(blitted) == 2 else (None, None))
    if black and orange:
        # The two halves come from ONE blit of a two-colour source. If they disagree about
        # which event wrote them, the control is reading two different operations and the
        # colour half of the construction proves nothing about the verdict half.
        def last_transfer(r):
            t = [e["eventId"] for e in r["events"] if e["kind"] == "transfer"]
            return t[-1] if t else None
        if last_transfer(black) != last_transfer(orange):
            problems.append("F/F': the two halves name different transfer events; they are "
                            "one blit of a two-colour source and must name the same one")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixel", help="X,Y; default is the brightest pixel in the target")
    parser.add_argument("--target", type=int, default=0, help="colour target index")
    parser.add_argument("--expect-control", action="store_true",
                        help="check this tool's reading against pixel_history_control")
    args = parser.parse_args()
    if not args.capture.is_file():
        parser.error("capture does not exist")
    if not shutil.which("qrenderdoc"):
        parser.error("qrenderdoc is not installed in this environment")
    pixel = None
    if args.pixel:
        try:
            x, y = (int(v) for v in args.pixel.split(","))
            pixel = [x, y]
        except ValueError:
            parser.error("--pixel must be X,Y")
    try:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(f"output must be a new directory: {exc}")

    request = {"capture": str(args.capture.resolve()), "output": str(output),
               "pixel": pixel, "target": args.target,
               "expect_control": args.expect_control}
    request_path = output / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    env = {**os.environ, "PROSPER_PIXHIST_REQUEST": str(request_path),
           "QT_QPA_PLATFORM": "offscreen"}
    try:
        proc = subprocess.run(["qrenderdoc", "--python", str(Path(__file__).resolve())],
                              env=env, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        (output / "process.log").write_text(str(exc))
        print(f"Replay unavailable: {exc}", file=sys.stderr)
        return 1
    (output / "process.log").write_text(proc.stdout + proc.stderr)
    report = output / "pixel_history.json"
    if not report.is_file():
        print(f"Replay produced no report (exit={proc.returncode}); inspect {output}",
              file=sys.stderr)
        return 1
    data = json.loads(report.read_text())
    if data.get("status") != "REPLAYED":
        print(f"Analysis failed; inspect {output}\n{data.get('error', '')}", file=sys.stderr)
        return 1

    if args.expect_control:
        regions = data.get("control_regions") or {}
        problems = check_control(regions)
        for name, _, _ in CONTROL_REGIONS:
            got = regions.get(name)
            print(f"  {name:<16} {got['verdict'] if got else 'MISSING'}")
        for problem in problems:
            print(f"CONTROL FAILED: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"CONTROL VERIFIED: {len(CONTROL_REGIONS)} regions, each constructed verdict "
              f"read back correctly, plus A's depth/discard/scissor arms in order and E's "
              f"own discard and clear. Rejection reasons are checked in A and E, not in "
              f"every region.")

    print(f"{data['verdict']}: {data['reason']}")
    print(f"  pixel {tuple(data['pixel'])} of {data['target']['width']}x"
          f"{data['target']['height']}, chosen by {data['pixel_choice']}")
    print(f"  target {data['target']['id']} of {data['target']['candidates']}, "
          f"from {data['target']['chosen_from']}")
    print(f"  {data['draw_count']} draws in frame, {len(data['events'])} touched this pixel")
    resolved = set(data.get("api", {}).get("transfer_usages") or [])
    if not resolved:
        # Silence here would be the defect coming back with nothing to show for it: an
        # empty set means every name failed to resolve and every copy is read as a draw.
        print(f"  WARNING: this RenderDoc build exposed NONE of {list(TRANSFER_USAGES)}, "
              f"so every copy, blit and resolve here is being read as a draw -- #3403 is "
              f"live again on this build")
    elif resolved != set(TRANSFER_USAGES):
        print(f"  NOTE: this RenderDoc build exposes only {sorted(resolved)} of "
              f"{list(TRANSFER_USAGES)}; a copy through a missing one would be read as a "
              f"draw")
    for e in data["events"]:
        # Three different absences, printed differently on purpose. "-" used to cover all
        # of them, which is how a value meaning "no information" reached a reader looking
        # like a measurement.
        out = ("suppressed (no fragment ran)" if e["shader_output_suppressed"]
               else ("[" + ", ".join(f"{v:.3f}" for v in e["shaderOut"]) + "]"
                     if e["shaderOut"] else "no value recorded"))
        gaps = [n for n in e.get("no_value", []) if n != "shaderOut"]
        # Passed() ignores unboundPS, so an event with no bound pixel shader arrives as a
        # pass. Printing a bare "PASS" would hide the one fact that matters about it.
        undefined = [f for f in e["rejected_by"] if f in UNDEFINED_OUTPUT]
        state = ("CLEAR" if e["kind"] == "clear"
                 else ("COPY:" + (e["usage"] or "?")) if e["kind"] == "transfer"
                 else ("PASS!" + ",".join(undefined)) if e["passed"] and undefined
                 else "PASS" if e["passed"]
                 else ",".join(e["rejected_by"]) or "rejected")
        print(f"    eid {e['eventId']:>6}  {state:<20} shaderOut {out}"
              + (f"  (no value recorded: {', '.join(gaps)})" if gaps else ""))
    print(f"  evidence: {args.output}")
    return 0


if os.environ.get("PROSPER_PIXHIST_REQUEST") and "pyrenderdoc" in globals():
    embedded()
elif __name__ == "__main__":
    sys.exit(main())
