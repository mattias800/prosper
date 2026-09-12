// Fixture for survey_sizes.py --selftest. NOT built: nothing globs tools/refactor/testdata.
//
// The discriminator for "can the survey see an unparsed span that is not at the end of the file".
// The _WIN32 arm below sits in the MIDDLE, with real code after it, so it is folded into the
// following cursor's region and carries that cursor's kind. A survey that infers unparsed lines
// from region kinds cannot see it; the preprocessor's own skipped-range record can.
namespace prosper {

int before() { return 1; }

#ifdef _WIN32
static int win_only_1() { return 1; }
static int win_only_2() { return 2; }
static int win_only_3() { return 3; }
static int win_only_4() { return 4; }
static int win_only_5() { return 5; }
static int win_only_6() { return 6; }
static int win_only_7() { return 7; }
static int win_only_8() { return 8; }
static int win_only_9() { return 9; }
static int win_only_10() { return 10; }
static int win_only_11() { return 11; }
static int win_only_12() { return 12; }
static int win_only_13() { return 13; }
static int win_only_14() { return 14; }
static int win_only_15() { return 15; }
static int win_only_16() { return 16; }
static int win_only_17() { return 17; }
static int win_only_18() { return 18; }
static int win_only_19() { return 19; }
static int win_only_20() { return 20; }
static int win_only_21() { return 21; }
static int win_only_22() { return 22; }
static int win_only_23() { return 23; }
static int win_only_24() { return 24; }
static int win_only_25() { return 25; }
static int win_only_26() { return 26; }
static int win_only_27() { return 27; }
static int win_only_28() { return 28; }
static int win_only_29() { return 29; }
static int win_only_30() { return 30; }
static int win_only_31() { return 31; }
static int win_only_32() { return 32; }
static int win_only_33() { return 33; }
static int win_only_34() { return 34; }
static int win_only_35() { return 35; }
static int win_only_36() { return 36; }
static int win_only_37() { return 37; }
static int win_only_38() { return 38; }
static int win_only_39() { return 39; }
static int win_only_40() { return 40; }
static int win_only_41() { return 41; }
static int win_only_42() { return 42; }
static int win_only_43() { return 43; }
static int win_only_44() { return 44; }
static int win_only_45() { return 45; }
static int win_only_46() { return 46; }
static int win_only_47() { return 47; }
static int win_only_48() { return 48; }
static int win_only_49() { return 49; }
static int win_only_50() { return 50; }
static int win_only_51() { return 51; }
static int win_only_52() { return 52; }
static int win_only_53() { return 53; }
static int win_only_54() { return 54; }
static int win_only_55() { return 55; }
static int win_only_56() { return 56; }
static int win_only_57() { return 57; }
static int win_only_58() { return 58; }
static int win_only_59() { return 59; }
static int win_only_60() { return 60; }
#else
static int posix_only_1() { return 1; }
static int posix_only_2() { return 2; }
static int posix_only_3() { return 3; }
static int posix_only_4() { return 4; }
static int posix_only_5() { return 5; }
static int posix_only_6() { return 6; }
static int posix_only_7() { return 7; }
static int posix_only_8() { return 8; }
static int posix_only_9() { return 9; }
static int posix_only_10() { return 10; }
static int posix_only_11() { return 11; }
static int posix_only_12() { return 12; }
#endif

int after_a() { return 2; }
int after_b() { return 3; }
int after_c() { return 4; }

}  // namespace prosper
