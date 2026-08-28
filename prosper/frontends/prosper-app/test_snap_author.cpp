// Unit tests for the pure half of human-authored render snapshots (snap_author.hpp).
#include "snap_author.hpp"

#include <cstdio>
#include <string>

using prosper::frontend::SnapVerdict;
using prosper::frontend::kSnapFlipUnanchored;
using prosper::frontend::snap_author_line;
using prosper::frontend::snap_file_name;
using prosper::frontend::snap_json_escape;
using prosper::frontend::snap_record_line;
using prosper::frontend::parse_snap_flip_list;
using prosper::frontend::snap_actual_file_name;
using prosper::frontend::snap_actual_record_line;
using prosper::frontend::snap_verdict_token;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::fprintf(stderr, "%s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) failures++;
}
static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

int main() {
    // ---- 1. The two verdicts are distinguishable, and spelled the way the manifest reads --------
    {
        check(std::string(snap_verdict_token(SnapVerdict::correct)) == "correct",
              "the F6 verdict is spelled 'correct'");
        check(std::string(snap_verdict_token(SnapVerdict::incorrect)) == "incorrect",
              "the F7 verdict is spelled 'incorrect'");
    }

    // ---- 2. File names sort in capture order, and describe themselves --------------------------
    // The zero padding is the point: a listing of an authoring session must read in the order the
    // person pressed the keys, and "snap_10" must not sort before "snap_2".
    {
        const std::string second = snap_file_name(2, SnapVerdict::correct, 900);
        const std::string tenth = snap_file_name(10, SnapVerdict::correct, 4000);
        check(second == "snap_0002_correct_f900.bmp", "a snap name carries index, verdict and anchor");
        check(second < tenth, "zero padding keeps lexicographic order equal to capture order");

        const std::string wrong = snap_file_name(3, SnapVerdict::incorrect, 1234);
        check(wrong == "snap_0003_incorrect_f1234.bmp", "an incorrect snap is named as such");
        check(wrong != snap_file_name(3, SnapVerdict::correct, 1234),
              "verdict is part of the name, so the two kinds cannot collide at one index");
    }

    // ---- 3. An unanchored snap is named as unanchored, never as flip zero ----------------------
    // Flip 0 is a REAL anchor -- the first flip after the guest's first pad poll. Formatting the
    // unset sentinel as "f0" would produce a snap that replays to the boot logo and compares it
    // against whatever the person actually judged.
    {
        const std::string unanchored = snap_file_name(1, SnapVerdict::correct, kSnapFlipUnanchored);
        const std::string at_zero = snap_file_name(1, SnapVerdict::correct, 0);
        check(unanchored == "snap_0001_correct_unanchored.bmp", "an unanchored snap says so");
        check(at_zero == "snap_0001_correct_f0.bmp", "flip 0 is a real anchor and formats as one");
        check(unanchored != at_zero, "the unset sentinel never collides with the first flip");
    }

    // ---- 4. The manifest record carries everything the import step needs -----------------------
    {
        const std::string line = snap_record_line(
            /*index=*/7, SnapVerdict::correct, /*flip=*/12345, /*guest_present=*/9876,
            /*width=*/1920, /*height=*/1080, "snap_0007_correct_f12345.bmp", "PPSA25009");
        check(contains(line, "\"index\":7"), "record carries the index");
        check(contains(line, "\"verdict\":\"correct\""), "record carries the verdict");
        check(contains(line, "\"pad_flip\":12345"), "record carries the replay anchor");
        check(contains(line, "\"guest_present\":9876"), "record carries the guest present count");
        check(contains(line, "\"width\":1920") && contains(line, "\"height\":1080"),
              "record carries the dimensions the signature will be computed at");
        check(contains(line, "\"file\":\"snap_0007_correct_f12345.bmp\""), "record names its image");
        check(contains(line, "\"title_id\":\"PPSA25009\""), "record carries the title id");
        check(line.front() == '{' && line.back() == '}' && !contains(line, "\n"),
              "a record is exactly one JSON object on one line");
    }

    // ---- 5. The unanchored sentinel survives into the manifest as -1 ---------------------------
    // The import step rejects on this value, so it has to arrive intact rather than as 0 or as a
    // huge unsigned number from a careless cast.
    {
        const std::string line = snap_record_line(
            0, SnapVerdict::incorrect, kSnapFlipUnanchored, 0, 640, 360,
            "snap_0000_incorrect_unanchored.bmp", "PPSA25009");
        check(contains(line, "\"pad_flip\":-1"),
              "an unanchored snap records -1, not 0 and not a wrapped unsigned");
    }

    // ---- 6. Escaping keeps the manifest parseable ----------------------------------------------
    {
        check(snap_json_escape("plain") == "plain", "ordinary text is untouched");
        check(snap_json_escape("a\"b") == "a\\\"b", "a quote is escaped");
        check(snap_json_escape("a\\b") == "a\\\\b", "a backslash is escaped");
        check(snap_json_escape("a\nb") == "a\\nb", "a newline is escaped, so a record stays one line");
        check(snap_json_escape(std::string("a\x01" "b")) == "a\\u0001b", "a control byte is escaped");

        // The whole point: a hostile-ish value must not be able to terminate the record early.
        const std::string line = snap_record_line(
            1, SnapVerdict::correct, 5, 5, 1, 1, "we\"ird\nname.bmp", "PPSA\\25009");
        check(!contains(line, "\n"), "an embedded newline cannot split one record into two");
        check(contains(line, "we\\\"ird"), "an embedded quote is escaped rather than closing the string");
    }

    // ---- 7. The console line tells the person what just happened -------------------------------
    // Including, loudly, the case where their keypress produced an unusable snap -- while they are
    // still sitting at the keyboard and can take another one.
    {
        const std::string good = snap_author_line(4, SnapVerdict::correct, 2000,
                                                  "snap_0004_correct_f2000.bmp");
        check(contains(good, "correct") && contains(good, "2000") &&
                  contains(good, "snap_0004_correct_f2000.bmp"),
              "an anchored snap reports verdict, anchor and file");
        check(!contains(good, "NOT ANCHORED"), "an anchored snap does not warn");

        const std::string bad = snap_author_line(0, SnapVerdict::correct, kSnapFlipUnanchored,
                                                 "snap_0000_correct_unanchored.bmp");
        check(contains(bad, "NOT ANCHORED"), "an unanchored snap warns at the moment it is taken");
        check(contains(bad, "rejected at import"),
              "...and says what will happen to it, not merely that something is odd");
    }

    // ---- 8. The replay flip list parses, sorts and de-duplicates -------------------------------
    {
        const auto three = parse_snap_flip_list("1200,300,4000");
        check(three.size() == 3 && three[0] == 300 && three[1] == 1200 && three[2] == 4000,
              "a flip list is parsed and sorted ascending");
        const auto deduped = parse_snap_flip_list("500,500,500");
        check(deduped.size() == 1 && deduped[0] == 500, "duplicate anchors collapse to one capture");
        check(parse_snap_flip_list("900").size() == 1, "a single anchor is a valid list");
        check(parse_snap_flip_list("100, 200").size() == 2, "spaces after commas are tolerated");
        check(parse_snap_flip_list("0").size() == 1 && parse_snap_flip_list("0")[0] == 0,
              "flip 0 is a legal anchor, not an empty list");
    }

    // ---- 9. A malformed list disables the trigger ENTIRELY, never partially --------------------
    // A partial parse is the dangerous failure: it would capture some of the authored moments and
    // silently skip the rest, and the run would report a clean pass with holes in it. Losing the
    // whole capture is loud; losing half of it is not.
    {
        check(parse_snap_flip_list("1200,oops,4000").empty(),
              "one bad token voids the whole list rather than capturing the good ones");
        check(parse_snap_flip_list("12.5").empty(), "a non-integer voids the list");
        check(parse_snap_flip_list("-300").empty(), "a negative anchor voids the list");
        check(parse_snap_flip_list("99999999999999999999999").empty(),
              "an out-of-range value voids the list instead of wrapping");
        check(parse_snap_flip_list("").empty() && parse_snap_flip_list(nullptr).empty(),
              "unset and empty mean no trigger");
    }

    // ---- 10. An actual-frame name records BOTH the target and where it landed ------------------
    // They differ whenever a slow frame overshoots the anchor. Recording only the target would hide
    // the drift, which is the first thing a reader needs when a diff looks inexplicable.
    {
        const std::string exact = snap_actual_file_name(1200, 1200);
        const std::string overshot = snap_actual_file_name(1200, 1204);
        check(exact == "actual_f1200_at1200.bmp", "an on-target capture names both as equal");
        check(overshot == "actual_f1200_at1204.bmp", "an overshoot is visible in the file name");
        check(exact != overshot, "the two cases cannot be confused");

        const std::string line = snap_actual_record_line(1200, 1204, 1920, 1080, overshot);
        check(contains(line, "\"target_flip\":1200") && contains(line, "\"actual_flip\":1204"),
              "the actual record carries the requested and reached anchors separately");
    }

    if (failures) { std::fprintf(stderr, "== FAIL: %d ==\n", failures); return 1; }
    std::fprintf(stderr, "== PASS ==\n");
    return 0;
}
