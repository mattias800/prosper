#!/usr/bin/env python3
"""The verdict must not soften: a black frame has four causes and they are not interchangeable."""
import struct
import unittest
import pixel_history as ph


def clear(post=(0.0, 0.0, 0.0, 1.0), eid=0, pre=(0.5, 0.5, 0.5, 1.0)):
    """A clear: passes, evaluates no test, and is never the subject of a verdict.

    `pre` differs from `post` by default, i.e. a clear that actually changed the pixel.
    """
    return ev(True, post=post, eid=eid, kind="clear", usage="Clear", pre=pre)


def transfer(post=(0.0, 0.0, 0.0, 1.0), eid=20, usage="CopyDst", pre=(0.0, 0.0, 0.0, 1.0)):
    """A copy, blit, resolve or mip generation landing on the pixel.

    It arrives from RenderDoc exactly as a clear does -- passed, no test evaluated, and no
    fragment shader anywhere near it -- which is why it must be told apart by what the API
    says it did to the resource rather than by anything on the event.
    """
    return ev(True, post=post, eid=eid, kind="transfer", usage=usage, pre=pre)


def ev(passed, rejected=(), shader=None, post=(0.0, 0.0, 0.0, 1.0), eid=1, suppressed=None,
       kind="draw", usage=None, pre=None):
    if suppressed is None:
        suppressed = ph.suppress_shader_output(rejected)
    out = None if suppressed else (None if shader is None else list(shader))
    before, after = list(pre if pre is not None else post), list(post)
    no_value = [n for n, v in (("preMod", before), ("postMod", after)) if v is None]
    if not suppressed and out is None:
        no_value.append("shaderOut")
    return {"eventId": eid, "passed": passed, "rejected_by": list(rejected),
            "kind": kind, "usage": usage, "preMod": before, "shaderOut": out,
            "shader_output_suppressed": suppressed, "postMod": after,
            "no_value": no_value}


class FakePixelValue:
    """RenderDoc's PixelValue is a union: one set of four words, several views of it."""

    def __init__(self, floats=None, uints=None):
        if floats is not None:
            self.floatValue = list(floats)
        if uints is not None:
            self.uintValue = list(uints)


class FakeModification:
    """A stand-in for RenderDoc's PixelModification, shaped like the real one.

    It exists because the case it carries cannot be captured on demand: the sentinel below
    appears exactly where RenderDoc has nothing to report, and no control program can make
    that happen. Waiting for a capture that happened to contain one would mean trusting a
    null produced by the same machinery the null is about.
    """

    def __init__(self, eid=1, passed=True, pre=None, post=None, shader=None, rejected=()):
        self.eventId = eid
        self._passed = passed
        self.preMod = pre if pre is not None else post
        self.postMod = post
        self.shaderOut = shader if shader is not None else post
        for flag in rejected:
            setattr(self, flag, True)

    def Passed(self):
        return self._passed


def real(rgba):
    """A ModificationValue holding a colour RenderDoc actually measured."""
    return type("MV", (), {"col": FakePixelValue(floats=list(rgba), uints=[0, 0, 0, 0])})()


def no_information(shape="stamped_first"):
    """What RenderDoc leaves behind for "I have no data" (ModificationValue::SetInvalid).

    Two shapes, because which words the sentinel is stamped into is a property of a
    RenderDoc version rather than of this tool. The measured one (#3404) stamps word 0 and
    leaves the rest zero, which is what made `max(rgb)` come out as exactly 0.0 and read as
    a black pixel; the all-four form must be caught as well.
    """
    words = ([ph.NO_INFORMATION, 0, 0, 0] if shape == "stamped_first"
             else [ph.NO_INFORMATION] * 4)
    floats = [struct.unpack("<f", struct.pack("<I", w))[0] for w in words]
    return type("MV", (), {"col": FakePixelValue(floats=floats, uints=words)})()


class ClassifyTests(unittest.TestCase):
    def test_a_cleared_target_that_nothing_drew_to_is_not_a_shader_defect(self):
        # THE headline case: a black region of a cleared target. The clear passes and its
        # postMod is black, so reading it as the explaining event returned SHADER_WROTE_BLACK
        # and sent the reader to a shader that never ran.
        v, why = ph.classify([clear()])
        self.assertEqual(v, "NOTHING_DREW")
        self.assertIn("clear", why)

    def test_a_coloured_clear_that_nothing_drew_to_is_also_nothing_drew(self):
        # The same defect in its other direction: a sky-blue clear read as PIXEL_WAS_WRITTEN,
        # retiring the question with "something drew here" when nothing did.
        self.assertEqual(ph.classify([clear(post=(0.3, 0.6, 0.9, 1))])[0], "NOTHING_DREW")

    def test_a_clear_does_not_rescue_a_pixel_whose_draws_all_failed(self):
        v, why = ph.classify([clear(), ev(False, ["depthTestFailed"], eid=5)])
        self.assertEqual(v, "ALL_REJECTED")
        self.assertIn("1 draw", why)

    def test_a_clear_after_the_last_draw_is_the_thing_you_are_looking_at(self):
        v, why = ph.classify([ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1), eid=5),
                              clear(eid=9)])
        self.assertEqual(v, "CLEARED_AFTER_DRAW")
        self.assertIn("9", why)

    def test_a_clear_that_changed_nothing_does_not_take_the_blame(self):
        # A late clear whose preMod equals its postMod explains nothing. Blaming it would
        # move the reader off the draw that actually produced the pixel.
        v, _ = ph.classify([ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=5),
                            clear(post=(0, 0, 0, 1), pre=(0, 0, 0, 1), eid=9)])
        self.assertEqual(v, "SHADER_WROTE_BLACK")

    def test_a_clear_before_the_draws_does_not_change_a_real_verdict(self):
        drawn = ev(True, shader=(0.7, 0.2, 0.1, 1), post=(0, 0, 0, 1), eid=5)
        self.assertEqual(ph.classify([clear(), drawn])[0], "STORE_LOST_IT")

    def test_no_events_is_not_a_rendering_bug(self):
        v, why = ph.classify([])
        self.assertEqual(v, "NOTHING_DREW")
        self.assertIn("No event", why)

    def test_all_rejected_names_the_test_that_did_it(self):
        v, why = ph.classify([ev(False, ["depthTestFailed"]), ev(False, ["depthTestFailed"]),
                              ev(False, ["scissorClipped"])])
        self.assertEqual(v, "ALL_REJECTED")
        self.assertIn("depthTestFailedx2", why)
        self.assertIn("scissorClippedx1", why)

    def test_written_pixel_is_not_a_defect(self):
        v, _ = ph.classify([ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1))])
        self.assertEqual(v, "PIXEL_WAS_WRITTEN")

    def test_shader_black_and_store_lost_are_distinguished(self):
        # Same final pixel, same pass/fail pattern, opposite investigations.
        black = ph.classify([ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1))])[0]
        lost = ph.classify([ev(True, shader=(0.7, 0.2, 0.1, 1), post=(0, 0, 0, 1))])[0]
        self.assertEqual(black, "SHADER_WROTE_BLACK")
        self.assertEqual(lost, "STORE_LOST_IT")

    def test_stale_shader_output_cannot_manufacture_store_lost_it(self):
        # Trap 268 end to end: the scissored event carries the PREVIOUS draw's bright
        # colour. `suppressed=False` forces the pre-fix behaviour into the fixture, so this
        # fails unless classify() refuses to derive a verdict from a rejected event.
        stale = ev(False, ["scissorClipped"], shader=(0.9, 0.9, 0.9, 1), suppressed=False)
        passing_black = ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1))
        v, _ = ph.classify([stale, passing_black])
        self.assertEqual(v, "SHADER_WROTE_BLACK")

    def test_legitimately_overdrawn_bright_writer_is_not_a_lost_store(self):
        # A white sky, then a black object correctly drawn in front of it. Reading ANY
        # passing event made SHADER_WROTE_BLACK unreachable on every pixel with history --
        # which is most of a real frame, and precisely the frames people investigate.
        sky = ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1), eid=10)
        obj = ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=20)
        self.assertEqual(ph.classify([sky, obj])[0], "SHADER_WROTE_BLACK")

    def test_store_lost_it_still_reachable_after_an_overdraw(self):
        sky = ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1), eid=10)
        lost = ev(True, shader=(0.6, 0.3, 0.1, 1), post=(0, 0, 0, 1), eid=20)
        self.assertEqual(ph.classify([sky, lost])[0], "STORE_LOST_IT")

    def test_unbound_pixel_shader_is_not_a_clean_pass(self):
        # Passed() excludes unboundPS (data_types.h:2759), so the event arrives passed=True
        # while its output is documented as undefined. Deriving STORE_LOST_IT from that is
        # trap 268 one list entry over.
        self.assertTrue(ph.suppress_shader_output(["unboundPS"]))
        v, why = ph.classify([ev(True, ["unboundPS"], shader=(0.8, 0.8, 0.8, 1),
                                  post=(0, 0, 0, 1))])
        self.assertEqual(v, "OUTPUT_UNTRUSTED")
        self.assertIn("unboundPS", why)

    def test_a_rejected_event_never_counts_as_the_final_colour(self):
        v, _ = ph.classify([ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1)),
                            ev(False, ["depthTestFailed"], shader=(1, 1, 1, 1),
                               post=(1, 1, 1, 1))])
        self.assertEqual(v, "SHADER_WROTE_BLACK")

    def test_near_black_is_black(self):
        # A pixel at 1/255 is black to a viewer; a threshold of exactly 0 would call
        # every dithered or rounded frame "written" and retire the question wrongly.
        v, _ = ph.classify([ev(True, shader=(0.002, 0.001, 0.0, 1), post=(0.002, 0, 0, 1))])
        self.assertEqual(v, "SHADER_WROTE_BLACK")


class ControlCheckTests(unittest.TestCase):
    """check_control must fail on a wrong READING, not merely on a wrong final picture."""

    def regions(self):
        seq = [clear(eid=0),
               ev(True, shader=(0, 1, 0, 1), post=(0, 1, 0, 1), eid=1),
               ev(False, ["scissorClipped"], eid=2),
               ev(False, ["shaderDiscarded"], shader=(0, 0, 0, 0), eid=3),
               ev(True, shader=(1, 1, 0, 1), post=(1, 1, 0, 1), eid=4),
               ev(False, ["depthTestFailed"], shader=(0, 0, 1, 1), eid=5),
               ev(False, ["scissorClipped"], eid=9)]   # a later region's draw
        killed = [clear(eid=0), clear(eid=3), ev(False, ["scissorClipped"], eid=4),
                  ev(False, ["shaderDiscarded"], shader=(0, 0, 0, 0), eid=8)]
        blit = [clear(eid=0), clear(eid=3),
                ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=9),
                transfer(eid=40)]
        bright = [clear(eid=0), clear(eid=3),
                  ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=9),
                  transfer(eid=40, post=(1, 0.5, 0, 1))]
        return {"A  sequence": {"verdict": "PIXEL_WAS_WRITTEN", "events": seq},
                "A' arm-1 only": {"verdict": "PIXEL_WAS_WRITTEN", "events": []},
                "B  black draw": {"verdict": "SHADER_WROTE_BLACK", "events": []},
                "C  no write": {"verdict": "STORE_LOST_IT", "events": []},
                "E  all killed": {"verdict": "ALL_REJECTED", "events": killed},
                "F  blit black": {"verdict": "TRANSFER_WROTE_PIXEL", "events": blit},
                "F' blit orange": {"verdict": "TRANSFER_WROTE_PIXEL", "events": bright}}

    def test_As_own_scissor_arm_must_precede_its_last_surviving_draw(self):
        # Deleting arm 2 leaves a later region's scissorClipped in A's history, so a check
        # that only asks "is scissorClipped present" stays green with the arm gone.
        bad = self.regions()
        bad["A  sequence"]["events"] = [e for e in bad["A  sequence"]["events"]
                                        if e["eventId"] != 2]
        self.assertIn("arm 2", " ".join(ph.check_control(bad)))

    def test_E_must_contain_its_own_discarded_draw(self):
        bad = self.regions()
        bad["E  all killed"]["events"] = [e for e in bad["E  all killed"]["events"]
                                          if "shaderDiscarded" not in e["rejected_by"]]
        self.assertIn("region's own draw is missing", " ".join(ph.check_control(bad)))

    def test_E_must_contain_both_clear_forms(self):
        # One clear is not enough: the transfer clear carries ActionFlags::Clear and the
        # loadOp clear does not, so a reading with only the flagged one stays green while
        # every loadOp clear in a real capture goes unrecognised.
        for keep in (0, 1):
            bad = self.regions()
            cs = [e for e in bad["E  all killed"]["events"] if e["kind"] == "clear"]
            bad["E  all killed"]["events"] = [
                e for e in bad["E  all killed"]["events"]
                if e["kind"] != "clear" or e["eventId"] == cs[keep]["eventId"]]
            self.assertIn("expected 2", " ".join(ph.check_control(bad)))

    def test_correct_reading_passes(self):
        self.assertEqual(ph.check_control(self.regions()), [])

    def test_every_region_verdict_is_checked(self):
        # Each region exists to construct one verdict; a tool that collapsed them all to the
        # same answer would still satisfy a check that only looked at one.
        for name in ("A  sequence", "B  black draw", "C  no write", "E  all killed",
                     "F  blit black", "F' blit orange"):
            bad = self.regions()
            bad[name]["verdict"] = "NOTHING_DREW"
            self.assertTrue(ph.check_control(bad), name)

    def test_missing_region_is_caught(self):
        bad = self.regions()
        del bad["C  no write"]
        self.assertIn("missing", " ".join(ph.check_control(bad)))

    def test_misattributed_rejection_is_caught(self):
        # The reading stays well-formed and the verdict stays right; only the REASON is
        # wrong. That is the failure a final-picture check cannot see.
        bad = self.regions()
        # Target by event id, not position: the fixture's indices move whenever the
        # construction gains an event, and a positional fixture silently retargets.
        bad["A  sequence"]["events"] = [
            ev(False, ["stencilTestFailed"], shader=(0, 0, 1, 1), eid=5)
            if e["eventId"] == 5 else e
            for e in bad["A  sequence"]["events"]]
        self.assertIn("depthTestFailed", " ".join(ph.check_control(bad)))

    def test_unsuppressed_stale_output_is_caught(self):
        bad = self.regions()
        bad["A  sequence"]["events"][1] = ev(False, ["scissorClipped"], shader=(0, 0, 1, 1),
                                             eid=2, suppressed=False)
        self.assertIn("stale-value suppression", " ".join(ph.check_control(bad)))


class SuppressionTests(unittest.TestCase):
    """Guards the decision the replay path makes; CI has no GPU, so this is its only cover."""

    def test_pre_fragment_rejections_suppress(self):
        for reason in ("scissorClipped", "backfaceCulled", "depthClipped", "viewClipped",
                       "predicationSkipped", "sampleMasked"):
            self.assertTrue(ph.suppress_shader_output([reason]), reason)

    def test_post_fragment_rejections_keep_their_output(self):
        # The shader DID run for these, so its output is real and often the whole answer:
        # a depth-failed draw that computed the missing colour names the defect.
        for reason in ("depthTestFailed", "stencilTestFailed", "shaderDiscarded",
                       "depthBoundsFailed"):
            self.assertFalse(ph.suppress_shader_output([reason]), reason)

    def test_passing_event_keeps_its_output(self):
        self.assertFalse(ph.suppress_shader_output([]))

    def test_mixed_rejection_suppresses(self):
        self.assertTrue(ph.suppress_shader_output(["depthTestFailed", "scissorClipped"]))

class NoInformationTests(unittest.TestCase):
    """#3404: a sentinel meaning "I do not know" was being consumed as the colour black.

    Every arm builds the sentinel BY HAND rather than looking for one in a capture, for the
    reason FakeModification gives.
    """

    def test_the_sentinel_constant_really_is_the_bit_pattern(self):
        # Not a fixture assertion: if the float form stopped being the reinterpretation of
        # 0xdeadbeef, the fallback detection route would silently match nothing.
        self.assertEqual(struct.pack("<f", ph.NO_INFORMATION_AS_FLOAT),
                         struct.pack("<I", 0xdeadbeef))
        self.assertLess(ph.NO_INFORMATION_AS_FLOAT, 0.0)

    def test_no_information_is_not_a_colour(self):
        for shape in ("stamped_first", "stamped_all"):
            self.assertTrue(ph.carries_no_information(no_information(shape)), shape)
            self.assertIsNone(ph.colour_values(no_information(shape)), shape)

    def test_a_real_colour_is_still_a_colour(self):
        # The discriminator has to reject the sentinel and nothing else: a detector that
        # answered "no information" to everything would satisfy every arm above.
        self.assertFalse(ph.carries_no_information(real((0.0, 0.0, 0.0, 1.0))))
        self.assertEqual(ph.colour_values(real((0.0, 0.0, 0.0, 1.0))), [0, 0, 0, 1])
        self.assertEqual(ph.colour_values(real((0.7, 0.2, 0.1, 1.0))), [0.7, 0.2, 0.1, 1.0])

    def test_the_float_view_alone_is_enough_to_detect_it(self):
        # The Python binding is not guaranteed to expose the integer view of the union.
        floats = [ph.NO_INFORMATION_AS_FLOAT, 0.0, 0.0, 0.0]
        self.assertIsNone(ph.colour_values(
            type("MV", (), {"col": FakePixelValue(floats=floats)})()))

    def test_the_integer_view_alone_is_enough_to_detect_it(self):
        # And the other way round: 0xdeadbeef is an INTEGER sentinel, so the integer view
        # is the direct reading of it and the float view is the reinterpretation. Either
        # route alone must catch it, since which the binding exposes is not our choice.
        only_ints = type("MV", (), {"col": FakePixelValue(uints=[ph.NO_INFORMATION] * 4)})()
        self.assertTrue(ph.carries_no_information(only_ints))
        self.assertIsNone(ph.colour_values(only_ints))

    def test_renderdocs_own_IsValid_is_enough_on_its_own(self):
        # A value whose bits say nothing and whose IsValid() says "no data": detected on
        # that route alone, since the bit patterns here are a perfectly ordinary black.
        invalid = type("MV", (), {"col": FakePixelValue(floats=[0.0] * 4),
                                  "IsValid": lambda self: False})()
        valid = type("MV", (), {"col": FakePixelValue(floats=[0.0, 0.0, 0.0, 1.0]),
                                "IsValid": lambda self: True})()
        self.assertIsNone(ph.colour_values(invalid))
        self.assertEqual(ph.colour_values(valid), [0, 0, 0, 1])

    def test_IsValid_cannot_overrule_the_sentinel_bits(self):
        # The three routes are a UNION, not a precedence. IsValid() is the one route nobody
        # here can run against a real binding, so letting it short-circuit past the bit
        # patterns would make it the only thing able to re-open #3404 -- a value whose
        # word 0 IS the sentinel is no-information whatever IsValid() claims.
        w = [ph.NO_INFORMATION, 0, 0, 0]
        f = [struct.unpack("<f", struct.pack("<I", x))[0] for x in w]
        lying = type("MV", (), {"col": FakePixelValue(floats=f, uints=w),
                                "IsValid": lambda self: True})()
        self.assertTrue(ph.carries_no_information(lying))
        self.assertIsNone(ph.colour_values(lying))

    def test_a_binding_whose_IsValid_raises_falls_through_to_the_bits(self):
        def boom(self):
            raise RuntimeError("no such method on this build")
        w = [ph.NO_INFORMATION, 0, 0, 0]
        f = [struct.unpack("<f", struct.pack("<I", x))[0] for x in w]
        odd = type("MV", (), {"col": FakePixelValue(floats=f, uints=w), "IsValid": boom})()
        self.assertTrue(ph.carries_no_information(odd))

    def test_a_pixel_with_no_recorded_value_is_not_reported_as_shader_wrote_black(self):
        # THE headline case, end to end through the builder the replay path itself uses.
        # Before the fix this returned SHADER_WROTE_BLACK -- max() over the float view of
        # the sentinel is exactly 0.0, so an absence of data was reported as a measured
        # black and the reader was sent to resource binding and shaders.
        for shape in ("stamped_first", "stamped_all"):
            e = ph.modification_event(FakeModification(eid=7, post=no_information(shape)),
                                      "draw")
            # The VERDICT first, deliberately: this arm has to distinguish "the tool said
            # something" from "the tool said the right thing about a pixel it cannot read",
            # so the assertion that goes red without the fix must be the verdict itself.
            verdict, why = ph.classify([e])
            self.assertNotEqual(verdict, "SHADER_WROTE_BLACK", shape)
            self.assertEqual(verdict, "VALUE_UNKNOWN", shape)
            self.assertIn("7", why)
            self.assertIsNone(e["postMod"], shape)
            self.assertIn("postMod", e["no_value"], shape)

    def test_a_pixel_with_no_recorded_value_is_not_a_lost_store_either(self):
        # The other direction of the same absence, and the arm that makes the postMod
        # branch load-bearing on its own: RenderDoc records a real bright shader output
        # and NO final value. Reading the missing value as 0.0 makes that STORE_LOST_IT --
        # blend state, write masks -- which is a second confident answer from no data.
        e = ph.modification_event(
            FakeModification(eid=13, post=no_information(), shader=real((0.7, 0.2, 0.1, 1))),
            "draw")
        v, _ = ph.classify([e])
        self.assertNotIn(v, ("STORE_LOST_IT", "SHADER_WROTE_BLACK"))
        self.assertEqual(v, "VALUE_UNKNOWN")

    def test_a_measured_black_pixel_still_reads_shader_wrote_black(self):
        # The domain control for the arm above: the refusal must not have swallowed the
        # verdict standing next to it. Same construction path, a real value instead.
        e = ph.modification_event(FakeModification(eid=7, post=real((0, 0, 0, 1))), "draw")
        self.assertEqual(e["postMod"], [0, 0, 0, 1])
        self.assertEqual(e["no_value"], [])
        self.assertEqual(ph.classify([e])[0], "SHADER_WROTE_BLACK")

    def test_a_shader_output_with_no_recorded_value_cannot_settle_the_question(self):
        # postMod is a measured black, so the pixel really is black -- but with no shader
        # output there is nothing to separate "computed black" from "the store lost it",
        # which is the entire distinction this tool exists to make.
        e = ph.modification_event(
            FakeModification(eid=11, post=real((0, 0, 0, 1)), shader=no_information()),
            "draw")
        self.assertIsNone(e["shaderOut"])
        self.assertIn("shaderOut", e["no_value"])
        v, why = ph.classify([e])
        self.assertEqual(v, "VALUE_UNKNOWN")
        self.assertIn("store lost", why)

    def test_a_suppressed_shader_output_is_still_reported_as_suppressed(self):
        # "No fragment ran" and "RenderDoc recorded nothing" are different findings, and
        # must not collapse into each other now that both arrive as None.
        e = ph.modification_event(
            FakeModification(eid=3, passed=False, post=real((0, 0, 0, 1)),
                             rejected=["scissorClipped"]), "draw")
        self.assertTrue(e["shader_output_suppressed"])
        self.assertNotIn("shaderOut", e["no_value"])

    def test_a_late_clear_with_no_recorded_value_does_not_blame_the_draw(self):
        # SetInvalid stamps preMod and postMod alike, so "did this clear change the pixel"
        # is unanswerable -- and answering it by falling through names the draw underneath,
        # which is the misattribution CLEARED_AFTER_DRAW exists to prevent.
        drawn = ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=5)
        blind = ph.modification_event(
            FakeModification(eid=9, pre=no_information(), post=no_information()),
            "clear", "Clear")
        v, why = ph.classify([drawn, blind])
        self.assertEqual(v, "VALUE_UNKNOWN")
        self.assertIn("9", why)

    def test_a_late_clear_with_real_values_still_fires(self):
        # The domain control for the arm above.
        drawn = ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=5)
        wiped = ph.modification_event(
            FakeModification(eid=9, pre=real((1, 1, 1, 1)), post=real((0, 0, 0, 1))),
            "clear", "Clear")
        self.assertEqual(ph.classify([drawn, wiped])[0], "CLEARED_AFTER_DRAW")

    def test_the_replay_record_and_the_fixtures_have_the_same_shape(self):
        # These two descriptions of an event used to be written out twice by hand, and
        # everything CI can reach reads the fixture rather than the record.
        built = ph.modification_event(FakeModification(post=real((0, 0, 0, 1))), "draw")
        self.assertEqual(sorted(built), sorted(ev(True, shader=(0, 0, 0, 1))))


class TransferTests(unittest.TestCase):
    """#3403: a pixel that arrived by copy was attributed to a shader that never wrote it.

    RenderDoc pushes copy/blit/resolve/genmips into the history through the same
    `clear || directWrite` branch as a clear -- passed, no test evaluated -- so the last
    such event was the one a verdict rested on, and a blit landing black read as
    SHADER_WROTE_BLACK.
    """

    def test_a_blit_that_lands_black_is_not_a_shader_computing_black(self):
        # THE headline case: a composited target whose last touch is a copy. Before the fix
        # the blit was the last "passing draw" and its black postMod produced
        # SHADER_WROTE_BLACK -- resource binding, textures, uniforms -- for a pixel no
        # shader wrote.
        drawn = ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=10)
        v, why = ph.classify([drawn, transfer(eid=20)])
        self.assertNotEqual(v, "SHADER_WROTE_BLACK")
        self.assertEqual(v, "TRANSFER_WROTE_PIXEL")
        self.assertIn("CopyDst", why)
        self.assertIn("20", why)

    def test_the_verdict_names_the_operation_that_actually_wrote_it(self):
        # "A copy did this" and "a resolve did this" are different places to look, and the
        # reader has no other way to find out which it was.
        for usage in ("CopyDst", "Resolve", "ResolveDst", "GenMips", "Copy"):
            drawn = ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=10)
            v, why = ph.classify([drawn, transfer(eid=20, usage=usage)])
            self.assertEqual(v, "TRANSFER_WROTE_PIXEL", usage)
            self.assertIn(usage, why)

    def test_a_bright_blit_is_not_retired_as_pixel_was_written(self):
        # The other direction of the same misattribution, and the quieter one: "something
        # drew here, not a defect" closes the question with the wrong stage named.
        drawn = ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=10)
        v, _ = ph.classify([drawn, transfer(eid=20, post=(1, 0.5, 0, 1))])
        self.assertEqual(v, "TRANSFER_WROTE_PIXEL")

    def test_a_target_written_only_by_a_copy_is_not_nothing_drew(self):
        # And specifically not "only clear events touched this pixel", which is false.
        v, why = ph.classify([clear(eid=0), transfer(eid=20)])
        self.assertEqual(v, "TRANSFER_WROTE_PIXEL")
        self.assertNotIn("only clear", why)

    def test_a_clear_after_the_copy_is_not_reported_as_the_copy(self):
        # Found in review. The no-draw path took the newest TRANSFER instead of walking the
        # history, so a copy a later clear had wiped was reported as "event 10 wrote it;
        # what you see is that operation's SOURCE" -- a confident false statement with no
        # draw anywhere in the history, and a regression against the pre-fix code, which
        # said CLEARED_AFTER_DRAW here.
        v, why = ph.classify([transfer(eid=10, post=(1, 0.5, 0, 1)),
                              clear(eid=20, pre=(1, 0.5, 0, 1), post=(0, 0, 0, 1))])
        self.assertNotEqual(v, "TRANSFER_WROTE_PIXEL")
        self.assertIn("20", why)
        self.assertIn("clear", why)

    def test_a_no_op_clear_after_the_copy_does_not_hide_it(self):
        # The domain control for the arm above: a clear that changed nothing explains
        # nothing, so the copy is still the answer. Without this the repair could be "any
        # later clear wins", which would lose the defect the PR is about.
        v, why = ph.classify([transfer(eid=10, post=(1, 0.5, 0, 1)),
                              clear(eid=20, pre=(1, 0.5, 0, 1), post=(1, 0.5, 0, 1))])
        self.assertEqual(v, "TRANSFER_WROTE_PIXEL")
        self.assertIn("10", why)

    def test_a_clear_with_no_recorded_value_after_a_copy_is_unknown(self):
        blind = ph.modification_event(
            FakeModification(eid=20, pre=no_information(), post=no_information()),
            "clear", "Clear")
        v, why = ph.classify([transfer(eid=10, post=(1, 0.5, 0, 1)), blind])
        self.assertEqual(v, "VALUE_UNKNOWN")
        self.assertIn("20", why)

    def test_the_rejected_note_does_not_name_a_transfer_that_came_first(self):
        # Found in review. The note was appended whenever ANY transfer was present, so a
        # copy that PRECEDED every draw was announced as "a later transfer then wrote this
        # pixel" -- the note exists to stop a wrong inference and was making one.
        v, why = ph.classify([transfer(eid=5),
                              ev(False, ["depthTestFailed"], eid=10),
                              ev(False, ["depthTestFailed"], eid=11)])
        self.assertEqual(v, "ALL_REJECTED")
        self.assertNotIn("later transfer", why)

    def test_the_rejected_note_does_not_name_a_transfer_a_clear_wiped(self):
        v, why = ph.classify([ev(False, ["depthTestFailed"], eid=10),
                              transfer(eid=20, post=(1, 0.5, 0, 1)),
                              clear(eid=30, pre=(1, 0.5, 0, 1), post=(0, 0, 0, 1))])
        self.assertEqual(v, "ALL_REJECTED")
        self.assertNotIn("later transfer", why)

    def test_the_rejected_note_names_a_transfer_that_really_is_last(self):
        # The domain control for the two arms above.
        v, why = ph.classify([ev(False, ["depthTestFailed"], eid=10), transfer(eid=20)])
        self.assertEqual(v, "ALL_REJECTED")
        self.assertIn("later transfer", why)
        self.assertIn("20", why)

    def test_the_last_draw_is_the_last_by_event_id_not_by_list_position(self):
        # Raised in review: explaining_touch() orders this question by eventId while the
        # draw it is asked about was taken by list position, so one function held two
        # notions of "last". RenderDoc sorts PixelModification by eventId, so they agree
        # today -- which is exactly why a disagreement would be silent if they ever stopped.
        out_of_order = [ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1), eid=20),
                        ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=10)]
        self.assertEqual(ph.classify(out_of_order)[0], "PIXEL_WAS_WRITTEN")

    def test_a_transfer_before_the_last_draw_does_not_take_the_blame(self):
        # A copy the draws then painted over explains nothing; the draw is the last writer
        # and the shader verdict is the right one.
        v, _ = ph.classify([transfer(eid=5),
                            ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=10)])
        self.assertEqual(v, "SHADER_WROTE_BLACK")

    def test_a_transfer_is_not_counted_among_the_draws(self):
        # It used to be, so "3 draw(s) reached the pixel" counted operations with no
        # fragment shader in them.
        v, why = ph.classify([ev(False, ["depthTestFailed"], eid=10), transfer(eid=20)])
        self.assertEqual(v, "ALL_REJECTED")
        self.assertIn("1 draw(s)", why)
        # ...and the reader is still told the copy wrote the colour they are looking at.
        self.assertIn("CopyDst", why)

    def test_an_unknown_clear_names_the_copy_under_it_not_the_draw(self):
        # Raised in review, and the reasoning is the point. When an unrecorded clear sits
        # over a copy that sits over a passing draw, the two live possibilities are "the
        # clear wiped it" and "the copy is what you see" -- the DRAW is not in the running
        # under either, so a refusal that names only the clear leaves the reader to assume
        # the wrong alternative.
        blind = ph.modification_event(
            FakeModification(eid=30, pre=no_information(), post=no_information()),
            "clear", "Clear")
        v, why = ph.classify([ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=10),
                              transfer(eid=20), blind])
        self.assertEqual(v, "VALUE_UNKNOWN")
        self.assertIn("30", why)
        self.assertIn("20", why)
        self.assertIn("CopyDst", why)

    def test_an_unknown_clear_does_not_name_a_copy_another_clear_wiped(self):
        # The fourth instance of one pattern, found in review: presence inside a window is
        # not the same as being the last writer. The copy at 20 sits between the draw and
        # the unrecorded clear, so a positional filter names it -- but the clear at 25
        # demonstrably wiped it, so if 30 changed nothing then 25, not 20, is what you are
        # looking at. Answered by asking explaining_touch() instead of filtering.
        blind = ph.modification_event(
            FakeModification(eid=30, pre=no_information(), post=no_information()),
            "clear", "Clear")
        v, why = ph.classify([ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=10),
                              transfer(eid=20, post=(1, 0.5, 0, 1)),
                              clear(eid=25, pre=(1, 0.5, 0, 1), post=(0, 0, 0, 1)),
                              blind])
        self.assertEqual(v, "VALUE_UNKNOWN")
        self.assertNotIn("what you are looking at", why)

    def test_an_unknown_clear_still_names_a_copy_a_no_op_clear_did_not_wipe(self):
        # The domain control for the arm above: a clear that changed nothing does not
        # displace the copy, or the repair would have deleted the sentence rather than
        # narrowed it.
        blind = ph.modification_event(
            FakeModification(eid=30, pre=no_information(), post=no_information()),
            "clear", "Clear")
        v, why = ph.classify([ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=10),
                              transfer(eid=20, post=(1, 0.5, 0, 1)),
                              clear(eid=25, pre=(1, 0.5, 0, 1), post=(1, 0.5, 0, 1)),
                              blind])
        self.assertEqual(v, "VALUE_UNKNOWN")
        self.assertIn("20", why)
        self.assertIn("CopyDst", why)

    def test_an_unknown_clear_does_not_name_a_copy_the_draw_painted_over(self):
        # The window matters, not just the presence of a copy: one that PRECEDED the last
        # passing draw was overwritten by it, so it is not the alternative to the clear --
        # the draw is. Naming it would be the same class of false lead as B2.
        blind = ph.modification_event(
            FakeModification(eid=30, pre=no_information(), post=no_information()),
            "clear", "Clear")
        v, why = ph.classify([transfer(eid=5),
                              ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=10),
                              blind])
        self.assertEqual(v, "VALUE_UNKNOWN")
        self.assertNotIn("what you are looking at", why)

    def test_an_unknown_clear_with_no_copy_under_it_names_no_copy(self):
        # The domain control: the sentence must not appear when there is nothing to name.
        blind = ph.modification_event(
            FakeModification(eid=30, pre=no_information(), post=no_information()),
            "clear", "Clear")
        v, why = ph.classify([ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=10),
                              blind])
        self.assertEqual(v, "VALUE_UNKNOWN")
        self.assertNotIn("what you are looking at", why)

    def test_a_later_clear_beats_an_earlier_transfer(self):
        drawn = ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1), eid=10)
        v, _ = ph.classify([drawn, transfer(eid=20), clear(eid=30)])
        self.assertEqual(v, "CLEARED_AFTER_DRAW")

    def test_a_later_transfer_beats_an_earlier_clear(self):
        drawn = ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1), eid=10)
        v, _ = ph.classify([drawn, clear(eid=20), transfer(eid=30)])
        self.assertEqual(v, "TRANSFER_WROTE_PIXEL")

    def test_a_no_op_clear_is_stepped_over_to_reach_the_transfer(self):
        # A clear that provably changed nothing explains nothing, so the search continues
        # past it rather than stopping at the newest late event.
        drawn = ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1), eid=10)
        idle = clear(eid=30, post=(0, 0, 0, 1), pre=(0, 0, 0, 1))
        v, _ = ph.classify([drawn, transfer(eid=20), idle])
        self.assertEqual(v, "TRANSFER_WROTE_PIXEL")

    def test_a_shader_storage_write_is_not_in_the_transfer_set(self):
        # The distinction is fixed-function transfer versus PROGRAMMABLE write, not
        # RenderDoc's "direct" versus "not". A compute shader writing a storage image is
        # shader work and the reader must still be sent to the shader; collapsing the set
        # to IsDirectWrite would silently reclassify every compute write as a copy.
        for rw in ("VS_RWResource", "PS_RWResource", "CS_RWResource", "All_RWResource"):
            self.assertNotIn(rw, ph.TRANSFER_USAGES, rw)
        # The READ halves are not writes to this target either, and including them would
        # blame a copy that merely sampled it.
        for read in ("CopySrc", "ResolveSrc"):
            self.assertNotIn(read, ph.TRANSFER_USAGES, read)
        for write in ("Copy", "CopyDst", "Resolve", "ResolveDst", "GenMips"):
            self.assertIn(write, ph.TRANSFER_USAGES, write)


class TransferControlCheckTests(unittest.TestCase):
    """The control's F region is the trap-279 guard; check_control must see it fail."""

    def regions(self):
        return ControlCheckTests.regions(ControlCheckTests())

    def test_a_blit_read_as_a_draw_is_caught(self):
        # The mutation that matters: the transfer detection is lost, so F's blit arrives as
        # a draw. Its verdict would then be SHADER_WROTE_BLACK -- but even a reading that
        # somehow kept the verdict must fail, because the history no longer contains the
        # thing the region was built to contain.
        bad = self.regions()
        for name in ("F  blit black", "F' blit orange"):
            bad[name]["events"] = [dict(e, kind="draw", usage=None)
                                   for e in bad[name]["events"]]
        self.assertIn("detection is lost", " ".join(ph.check_control(bad)))

    def test_a_blit_that_is_not_the_last_passing_event_is_caught(self):
        bad = self.regions()
        bad["F  blit black"]["events"].append(
            ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=99))
        self.assertIn("not the last passing event", " ".join(ph.check_control(bad)))

    def test_a_transfer_appearing_in_a_region_the_blit_misses_is_caught(self):
        # The assumption this whole channel rests on and that nobody here can test, since
        # RenderDoc is not installed: that a copy appears ONLY in the history of pixels its
        # destination rectangle covers. F's blit covers x in [52,64), y in [0,12), which no
        # other probe is inside -- so if RenderDoc lists non-covering direct writes, B, C
        # and E acquire a transfer event and every one of them flips to
        # TRANSFER_WROTE_PIXEL. The control now says so in one line instead of the reader
        # having to notice three verdicts changed.
        bad = self.regions()
        bad["B  black draw"]["events"] = [transfer(eid=40)]
        self.assertIn("OUTSIDE the blit", " ".join(ph.check_control(bad)))

    def test_the_two_halves_must_name_the_same_blit(self):
        bad = self.regions()
        bad["F' blit orange"]["events"][-1] = transfer(eid=77, post=(1, 0.5, 0, 1))
        self.assertIn("must name the same one", " ".join(ph.check_control(bad)))

    def test_a_correct_reading_of_both_halves_passes(self):
        self.assertEqual(ph.check_control(self.regions()), [])


if __name__ == "__main__":
    unittest.main()
