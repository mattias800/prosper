#!/usr/bin/env python3
"""test_dump_hygiene — the classifier must not delete authentic dump content.

This tool has a destructive mode, and during development it was wrong twice in exactly the
direction that matters:

  * `.bin` was in the module-suffix list, so a sweep reported 1,012 `Media/StreamingAssets/**/*.bin`
    and 914 `COMMON/ui/uv/*.bin` GAME DATA files as refused modules.
  * `sce_sys/about/right.sprx` — a Sony file present in every dump, including ones with no
    third-party additions at all — was classified REFUSED, so `--strip` would have deleted it from
    all 50.

Both were caught by running against a dump known to be clean, not by reading the code. So the
control arms below matter more than the positive ones: it is easy to write a detector that finds
`fakelib`, and hard to write one that finds only `fakelib`.

A note on WHY those two arms now hold, because it is not the reason you would guess and the
distinction decides what they are worth. The classifier was rewritten to key on "impersonates a Sony
library" rather than "is a module somewhere unexpected". Under that rule `settings.bin` and
`sce_sys/about/right.sprx` are ignored because neither is named `libSce*` -- NOT because of
MODULE_SUFFIXES or SONY_DATA_DIRS. Mutation-checking proved it: re-adding `.bin` to MODULE_SUFFIXES
and deleting the SONY_DATA_DIRS check both leave the suite green, because both are now
defence-in-depth rather than load-bearing.

Those two arms are therefore kept as REGRESSION arms against the old design returning, and each is
paired below with an arm that does exercise the guard it is named after.
"""
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import dump_hygiene as dh  # noqa: E402

HEADER = Path(__file__).resolve().parents[2] / "src/host/image/module_path_policy.hpp"


class ClassifierAgreesWithTheLoader(unittest.TestCase):
    def test_permitted_dirs_match_the_cpp_header(self):
        """The tool's allowlist must equal the loader's, or it reports on a policy nobody enforces.

        Parsed out of the header rather than duplicated in a comment, so the two cannot drift.
        """
        text = HEADER.read_text()
        start = text.index("kPermittedModuleDirs[] = {")
        body = text[start:text.index("};", start)]
        from_cpp = tuple(s.strip().strip('",') for s in body.splitlines()[1:] if '"' in s)
        self.assertEqual(from_cpp, dh.PERMITTED_MODULE_DIRS,
                         "dump_hygiene.PERMITTED_MODULE_DIRS has drifted from "
                         "module_path_policy.hpp's kPermittedModuleDirs")


class FindsTheRealThing(unittest.TestCase):
    def test_fakelib_modules_are_refused(self):
        for name in ("libSceAppContent.sprx", "libSceNpEntitlementAccess.sprx",
                     "libSceGameUpdate.sprx", "libSceAmpr.sprx"):
            got = dh.classify(f"fakelib/{name}", is_dir=False)
            self.assertIsNotNone(got, name)
            self.assertEqual(got[0], dh.REFUSED, name)

    def test_a_sony_library_outside_sce_module_is_refused_wherever_it_hides(self):
        # Moving the file must not launder it.
        for rel in ("Media/Plugins/libSceNpEntitlementAccess.prx",
                    "Media/Modules/libSceAppContent.prx",
                    "libSceAmpr.prx",
                    "vendor/libSceGameUpdate.sprx"):
            got = dh.classify(rel, is_dir=False)
            self.assertIsNotNone(got, rel)
            self.assertEqual(got[0], dh.REFUSED, rel)

    def test_release_group_markers(self):
        self.assertEqual(dh.classify("_DUPLEX_", is_dir=True)[0], dh.MARKER)
        self.assertEqual(dh.classify("_UNLiMiTED_", is_dir=True)[0], dh.MARKER)
        self.assertEqual(dh.classify("some/where/release.nfo", is_dir=False)[0], dh.MARKER)


class DoesNotTouchAuthenticContent(unittest.TestCase):
    """The arms that would have caught both real bugs."""

    def test_sce_sys_is_never_reported(self):
        # Present in EVERY dump, clean ones included. Reported once => --strip deletes it from all.
        # Holds today because right.sprx is not libSce*-named, not because of SONY_DATA_DIRS; the
        # arm below is the one that actually exercises that constant.
        self.assertIsNone(dh.classify("sce_sys/about/right.sprx", is_dir=False))
        self.assertIsNone(dh.classify("sce_sys/param.json", is_dir=False))
        self.assertIsNone(dh.classify("sce_sys/about", is_dir=True))

    def test_sce_sys_exclusion_is_load_bearing_for_marker_suffixes(self):
        """The arm that reddens when SONY_DATA_DIRS is removed.

        A `.nfo` under sce_sys would otherwise classify as MARKER -- i.e. strippable -- because the
        marker-suffix test runs before any location check. Authentic Sony metadata must not become
        deletable just because of its extension.
        """
        self.assertIsNone(dh.classify("sce_sys/about/readme.nfo", is_dir=False))
        self.assertIsNone(dh.classify("sce_sys/_DUPLEX_", is_dir=True))

    def test_module_suffix_gate_is_load_bearing_for_sony_named_data(self):
        """The arm that reddens when `.bin` is added back to MODULE_SUFFIXES.

        A title is free to ship a data file whose name begins with the Sony prefix. Only real module
        extensions may reach the impersonation test, or such a file becomes strippable.
        """
        self.assertIsNone(dh.classify("data/libSceFontCache.bin", is_dir=False))
        self.assertIsNone(dh.classify("Media/libSceSettings.dat", is_dir=False))

    def test_game_data_is_never_reported(self):
        for rel in ("Media/StreamingAssets/PostProcess/settings.bin",
                    "COMMON/ui/uv/atlas.bin",
                    "data/tex/character.bin",
                    "data/2dpar/sprite_c.par",
                    "engine/content/renderer/shaders.bin"):
            self.assertIsNone(dh.classify(rel, is_dir=False), rel)

    def test_a_titles_own_native_plugins_are_not_our_business(self):
        # Wwise, shipped by the developer. prosper not linking from prx/ is a loader question,
        # not a hygiene one -- and deleting these would remove real game content.
        for rel in ("prx/akaudioinput.prx", "prx/akcompressor.prx", "Media/Plugins/PSN.prx"):
            self.assertIsNone(dh.classify(rel, is_dir=False), rel)

    def test_sce_module_sony_libraries_are_permitted(self):
        for rel in ("sce_module/libc.prx", "sce_module/libSceNpCppWebApi.prx",
                    "sce_module/libSceJobManager.prx"):
            self.assertIsNone(dh.classify(rel, is_dir=False), rel)

    def test_consumed_and_retained_are_never_strippable(self):
        for rel, cat in (("dlc_emu.ini", dh.CONSUMED_K),
                         ("ampr_emu.index", dh.RETAINED_K)):
            got = dh.classify(rel, is_dir=False)
            self.assertEqual(got[0], cat, rel)
            self.assertNotIn(got[0], (dh.REFUSED, dh.MARKER),
                             f"{rel} must never fall into a strippable category")

    def test_original_files_is_retained_not_stripped(self):
        # Holds the dump's encrypted original eboot -- the only local evidence of the untouched
        # file. Stripping it would destroy exactly the thing that proves provenance.
        got = dh.classify("_original_files", is_dir=True)
        self.assertEqual(got[0], dh.RETAINED_K)


class StripSafety(unittest.TestCase):
    def test_strip_refuses_a_directory_that_is_not_a_dump(self, ):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "fakelib").mkdir()
            (root / "fakelib" / "libSceAppContent.sprx").write_bytes(b"x")
            self.assertFalse(dh.looks_like_a_dump(root))
            r = subprocess.run(
                [sys.executable, str(Path(dh.__file__)), "--strip", "--yes", str(root)],
                capture_output=True, text=True)
            self.assertTrue((root / "fakelib" / "libSceAppContent.sprx").exists(),
                            "refused to strip, so the file must still be there")
            self.assertIn("does not look", r.stderr)

    def test_strip_removes_refused_but_keeps_consumed_and_retained(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "sce_sys").mkdir()
            (root / "sce_sys" / "param.json").write_text("{}")
            (root / "sce_sys" / "about").mkdir()
            (root / "sce_sys" / "about" / "right.sprx").write_bytes(b"authentic")
            (root / "sce_module").mkdir()
            (root / "sce_module" / "libc.prx").write_bytes(b"authentic")
            (root / "fakelib").mkdir()
            (root / "fakelib" / "libSceAppContent.sprx").write_bytes(b"foreign")
            (root / "_DUPLEX_").mkdir()
            (root / "_DUPLEX_" / "duplex.nfo").write_text("nfo")
            (root / "dlc_emu.ini").write_text("[PSAL]")
            (root / "ampr_emu.index").write_bytes(b"AMPRIDX3")

            r = subprocess.run(
                [sys.executable, str(Path(dh.__file__)), "--strip", "--yes", str(root)],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)

            self.assertFalse((root / "fakelib" / "libSceAppContent.sprx").exists(), "REFUSED gone")
            self.assertFalse((root / "_DUPLEX_").exists(), "MARKER gone")
            self.assertTrue((root / "dlc_emu.ini").exists(), "CONSUMED kept")
            self.assertTrue((root / "ampr_emu.index").exists(), "RETAINED kept")
            self.assertTrue((root / "sce_module" / "libc.prx").exists(), "Sony library kept")
            self.assertTrue((root / "sce_sys" / "about" / "right.sprx").exists(),
                            "authentic sce_sys content kept -- this is the arm that would have "
                            "caught the right.sprx bug")

    def test_check_exit_code(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "sce_sys").mkdir()
            (root / "sce_sys" / "param.json").write_text("{}")
            clean = subprocess.run(
                [sys.executable, str(Path(dh.__file__)), "--check", str(root)],
                capture_output=True, text=True)
            self.assertEqual(clean.returncode, 0)
            (root / "fakelib").mkdir()
            (root / "fakelib" / "libSceAmpr.sprx").write_bytes(b"x")
            dirty = subprocess.run(
                [sys.executable, str(Path(dh.__file__)), "--check", str(root)],
                capture_output=True, text=True)
            self.assertEqual(dirty.returncode, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
