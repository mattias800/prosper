"""Tests for check_cpp_lint: the diff/finding parsers always, plus the end-to-end selftest (a real
throwaway repo through the whole gate) when the pinned clang tools are installed.

The end-to-end arm skips, loudly, without the tools (`uv sync --group cpp-lint`), because ctest and
the python-quality job may not have them. It is not left unexercised: the Linux CI job installs the
group and runs `check_cpp_lint.py --selftest` directly, where a missing tool is exit 2.
"""

import shutil
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_cpp_lint as lint  # noqa: E402


def have_tools():
    """True when clang-format and clang-tidy are reachable the way the gate looks them up."""
    scripts = Path(sys.executable).parent
    return all(
        (scripts / n).exists() or (scripts / f"{n}.exe").exists() or shutil.which(n)
        for n in ("clang-format", "clang-tidy")
    )


class Parsers(unittest.TestCase):
    """Pure functions; no tools, no filesystem."""

    def test_hunks_map_to_new_side_line_ranges(self):
        diff = (
            "diff --git a/prosper/src/a.cpp b/prosper/src/a.cpp\n"
            "--- a/prosper/src/a.cpp\n"
            "+++ b/prosper/src/a.cpp\n"
            "@@ -10,2 +10,3 @@ ctx\n"
            "@@ -40 +41 @@ ctx\n"
            "@@ -50,4 +51,0 @@ ctx\n"
        )
        self.assertEqual(lint.parse_diff(diff), {"prosper/src/a.cpp": [(10, 12), (41, 41)]})

    def test_deleted_file_non_cpp_and_third_party_are_dropped(self):
        diff = (
            "+++ /dev/null\n@@ -1,3 +0,0 @@\n"
            "+++ b/prosper/tools/x.py\n@@ -1 +1 @@\n"
            "+++ b/prosper/third_party/v.cpp\n@@ -1 +1 @@\n"
        )
        self.assertEqual(lint.parse_diff(diff), {})

    def test_findings_keep_level_and_check_name(self):
        out = (
            "C:\\r\\a.cpp:3:4: error: function is not thread safe "
            "[concurrency-mt-unsafe,-warnings-as-errors]\n"
            '/r/b.hpp:7:1: warning: no header providing "x" [misc-include-cleaner]\n'
            "   7 | context line that is not a finding\n"
        )
        found = [(f[1], f[2], f[3]) for f in lint.parse_findings(out)]
        self.assertEqual(
            found,
            [
                (3, "error", "concurrency-mt-unsafe,-warnings-as-errors"),
                (7, "warning", "misc-include-cleaner"),
            ],
        )

    def test_ref_that_looks_like_an_option_is_refused(self):
        with self.assertRaises(ValueError):
            lint.validate_ref("--output=/tmp/x")


@unittest.skipUnless(have_tools(), "clang tools not installed (uv sync --group cpp-lint)")
class EndToEnd(unittest.TestCase):
    """The gate's own selftest: each enabled family must fail, clean and legacy lines must pass."""

    def test_selftest_discriminates(self):
        # A MinGW compile database needs the target spelled out; clang defaults to MSVC on Windows.
        extra = ["--target=x86_64-w64-mingw32"] if sys.platform == "win32" else []
        self.assertEqual(lint.selftest(extra_args=extra), 0)


if __name__ == "__main__":
    unittest.main()
