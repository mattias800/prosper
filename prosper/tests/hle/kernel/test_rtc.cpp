// test_rtc — libSceRtc's calendar arithmetic must compute, not answer SCE_OK over garbage.
//
// The 22 exports below were unregistered, so the dispatcher answered `0` while writing nothing.
// Calendar outputs are manual-arithmetic inputs downstream (ticks feed duration math, datetimes
// feed UE4's FDateTime validation which fatals on month=0/day=0): every TEST therefore asserts
// WRITTEN values against independently known answers (fixed dates, not the host clock), plus
// refusal arms for null and invalid inputs.
//
// NID provenance: all 22 names hash cleanly with nid_hash (the method re-verified against the
// firmware set in every prior batch); three are pinned to literals below with a wrong-literal
// positive control.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

using namespace prosper;

using HleFn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
static uint64_t addr(const void* p) {
    return (uint64_t)(uintptr_t)p;
}
static uint64_t sx(uint32_t v) {
    return (uint64_t)(int64_t)(int32_t)v;
}

// SceRtcDateTime { u16 year..second; u32 microsecond } — 16 bytes, as fill_rtc_datetime writes.
struct DateTime {
    uint16_t year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;
    uint32_t microsecond = 0;
};
static DateTime ymd(int y, int mo, int d, int h = 0, int mi = 0, int s = 0, uint32_t us = 0) {
    DateTime t{};
    t.year = (uint16_t)y;
    t.month = (uint16_t)mo;
    t.day = (uint16_t)d;
    t.hour = (uint16_t)h;
    t.minute = (uint16_t)mi;
    t.second = (uint16_t)s;
    t.microsecond = us;
    return t;
}

// 2024-01-01T00:00:00Z: unix 1704067200 -> tick (1704067200 + 62135596800) * 1e6.
static constexpr uint64_t kTick20240101 = 63839664000000000ull;
static constexpr int64_t kUnix20240101 = 1704067200ll;

TEST(Rtc, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "sceRtcCheckValid",
        "sceRtcConvertLocalTimeToUtc",
        "sceRtcConvertUtcToLocalTime",
        "sceRtcFormatRFC3339",
        "sceRtcGetDayOfWeek",
        "sceRtcGetDaysInMonth",
        "sceRtcGetTickResolution",
        "sceRtcGetTime_t",
        "sceRtcGetWin32FileTime",
        "sceRtcIsLeapYear",
        "sceRtcParseRFC3339",
        "sceRtcSetTime_t",
        "sceRtcSetWin32FileTime",
        "sceRtcTickAddDays",
        "sceRtcTickAddHours",
        "sceRtcTickAddMicroseconds",
        "sceRtcTickAddMinutes",
        "sceRtcTickAddMonths",
        "sceRtcTickAddSeconds",
        "sceRtcTickAddTicks",
        "sceRtcTickAddWeeks",
        "sceRtcTickAddYears",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 22, "all 22 calendar exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(Rtc, NidsResolveToKnownValues) {
    EXPECT_EQ(nid_hash("sceRtcCheckValid"), "lPEBYdVX0XQ");
    EXPECT_EQ(nid_hash("sceRtcTickAddDays"), "NR1J0N7L2xY");
    EXPECT_EQ(nid_hash("sceRtcParseRFC3339"), "99bMGglFW3I");
    EXPECT_NE(nid_hash("sceRtcCheckValid"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

TEST(Rtc, CheckValidRefusesEachBadField) {
    register_builtin_hle();
    HleFn check = Hle::lookup(nid_hash("sceRtcCheckValid"));
    ASSERT_NE(check, nullptr);
    EXPECT_EQ(check(0, 0, 0, 0, 0, 0), sx(0x80B50002u)) << "null is INVALID_POINTER";
    DateTime good = ymd(2024, 2, 29);
    EXPECT_EQ(check(addr(&good), 0, 0, 0, 0, 0), 0u) << "leap-day 2024 is valid";
    DateTime t = ymd(0, 1, 1);
    EXPECT_EQ(check(addr(&t), 0, 0, 0, 0, 0), sx(0x80B50008u)) << "year 0";
    t = ymd(2024, 13, 1);
    EXPECT_EQ(check(addr(&t), 0, 0, 0, 0, 0), sx(0x80B50009u)) << "month 13";
    t = ymd(2023, 2, 29);
    EXPECT_EQ(check(addr(&t), 0, 0, 0, 0, 0), sx(0x80B5000Au)) << "Feb 29 2023 never existed";
    t = ymd(2024, 1, 1, 24);
    EXPECT_EQ(check(addr(&t), 0, 0, 0, 0, 0), sx(0x80B5000Bu)) << "hour 24";
    t = ymd(2024, 1, 1, 0, 60);
    EXPECT_EQ(check(addr(&t), 0, 0, 0, 0, 0), sx(0x80B5000Cu)) << "minute 60";
    t = ymd(2024, 1, 1, 0, 0, 60);
    EXPECT_EQ(check(addr(&t), 0, 0, 0, 0, 0), sx(0x80B5000Du)) << "second 60 (no leap second)";
    t = ymd(2024, 1, 1, 0, 0, 0, 1000000u);
    EXPECT_EQ(check(addr(&t), 0, 0, 0, 0, 0), sx(0x80B5000Eu)) << "microsecond 1e6";
}

TEST(Rtc, WeekdayMonthLeapKnownAnswers) {
    register_builtin_hle();
    HleFn dow = Hle::lookup(nid_hash("sceRtcGetDayOfWeek"));
    HleFn dim = Hle::lookup(nid_hash("sceRtcGetDaysInMonth"));
    HleFn leap = Hle::lookup(nid_hash("sceRtcIsLeapYear"));
    ASSERT_NE(dow, nullptr);
    ASSERT_NE(dim, nullptr);
    ASSERT_NE(leap, nullptr);
    EXPECT_EQ(dow(2024, 1, 1, 0, 0, 0), 1u) << "2024-01-01 was a Monday";
    EXPECT_EQ(dow(2000, 1, 1, 0, 0, 0), 6u) << "2000-01-01 was a Saturday";
    EXPECT_EQ(dow(1970, 1, 1, 0, 0, 0), 4u) << "1970-01-01 was a Thursday";
    EXPECT_TRUE((int64_t)dow(2024, 13, 1, 0, 0, 0) < 0) << "month 13 is an error, not a weekday";
    EXPECT_EQ(dim(2000, 2, 0, 0, 0, 0), 29u);
    EXPECT_EQ(dim(1900, 2, 0, 0, 0, 0), 28u) << "1900 is not a leap year";
    EXPECT_EQ(dim(2024, 2, 0, 0, 0, 0), 29u);
    EXPECT_EQ(dim(2023, 2, 0, 0, 0, 0), 28u);
    EXPECT_TRUE((int64_t)dim(2024, 0, 0, 0, 0, 0) < 0) << "month 0 is an error";
    EXPECT_EQ(leap(2000, 0, 0, 0, 0, 0), 1u);
    EXPECT_EQ(leap(1900, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(leap(2024, 0, 0, 0, 0, 0), 1u);
    EXPECT_EQ(leap(2023, 0, 0, 0, 0, 0), 0u);
    EXPECT_TRUE((int64_t)leap(0, 0, 0, 0, 0, 0) < 0) << "year 0 is an error, not 'common'";
    EXPECT_EQ(Hle::lookup(nid_hash("sceRtcGetTickResolution"))(0, 0, 0, 0, 0, 0), 1000000ull);
}

TEST(Rtc, TimeTAndWin32FileTimeRoundTrip) {
    register_builtin_hle();
    HleFn get_tt = Hle::lookup(nid_hash("sceRtcGetTime_t"));
    HleFn set_tt = Hle::lookup(nid_hash("sceRtcSetTime_t"));
    HleFn get_wft = Hle::lookup(nid_hash("sceRtcGetWin32FileTime"));
    HleFn set_wft = Hle::lookup(nid_hash("sceRtcSetWin32FileTime"));
    for (HleFn f : {get_tt, set_tt, get_wft, set_wft}) ASSERT_NE(f, nullptr);
    DateTime dt = ymd(2024, 1, 1);
    int64_t tt = -1;
    ASSERT_EQ(get_tt(addr(&dt), addr(&tt), 0, 0, 0, 0), 0u);
    EXPECT_EQ(tt, kUnix20240101);
    uint64_t wft = 0;
    ASSERT_EQ(get_wft(addr(&dt), addr(&wft), 0, 0, 0, 0), 0u);
    EXPECT_EQ(wft, 133485408000000000ull) << "(unix + 11644473600) * 10";
    DateTime back{};
    ASSERT_EQ(set_tt(addr(&back), (uint64_t)kUnix20240101, 0, 0, 0, 0), 0u);
    EXPECT_EQ(back.year, 2024u);
    EXPECT_EQ(back.month, 1u);
    EXPECT_EQ(back.day, 1u);
    DateTime back2{};
    ASSERT_EQ(set_wft(addr(&back2), wft, 0, 0, 0, 0), 0u);
    EXPECT_EQ(back2.year, 2024u);
    EXPECT_EQ(back2.day, 1u);
    EXPECT_EQ(get_tt(0, addr(&tt), 0, 0, 0, 0), sx(0x80B50002u)) << "null datetime";
    EXPECT_EQ(set_tt(addr(&back), (uint64_t)(int64_t)-1, 0, 0, 0, 0), sx(0x80B50003u))
        << "negative time_t refused (new-SDK behavior)";
}

TEST(Rtc, TickAddFixedUnits) {
    register_builtin_hle();
    HleFn days = Hle::lookup(nid_hash("sceRtcTickAddDays"));
    HleFn hours = Hle::lookup(nid_hash("sceRtcTickAddHours"));
    HleFn us = Hle::lookup(nid_hash("sceRtcTickAddMicroseconds"));
    HleFn minutes = Hle::lookup(nid_hash("sceRtcTickAddMinutes"));
    HleFn seconds = Hle::lookup(nid_hash("sceRtcTickAddSeconds"));
    HleFn ticks = Hle::lookup(nid_hash("sceRtcTickAddTicks"));
    HleFn weeks = Hle::lookup(nid_hash("sceRtcTickAddWeeks"));
    for (HleFn f : {days, hours, us, minutes, seconds, ticks, weeks}) ASSERT_NE(f, nullptr);
    uint64_t out = 0;
    const uint64_t base = kTick20240101;
    EXPECT_EQ(days(addr(&out), addr(&base), 1, 0, 0, 0), 0u);
    EXPECT_EQ(out, base + 86400000000ull);
    EXPECT_EQ(hours(addr(&out), addr(&base), 25, 0, 0, 0), 0u);
    EXPECT_EQ(out, base + 25ull * 3600000000ull);
    EXPECT_EQ(us(addr(&out), addr(&base), (uint64_t)(int64_t)-1, 0, 0, 0), 0u);
    EXPECT_EQ(out, base - 1u);
    EXPECT_EQ(minutes(addr(&out), addr(&base), 90, 0, 0, 0), 0u);
    EXPECT_EQ(out, base + 90ull * 60000000ull);
    EXPECT_EQ(seconds(addr(&out), addr(&base), 3600, 0, 0, 0), 0u);
    EXPECT_EQ(out, base + 3600000000ull);
    EXPECT_EQ(ticks(addr(&out), addr(&base), 7, 0, 0, 0), 0u);
    EXPECT_EQ(out, base + 7u);
    EXPECT_EQ(weeks(addr(&out), addr(&base), 2, 0, 0, 0), 0u);
    EXPECT_EQ(out, base + 2ull * 604800000000ull);
    EXPECT_EQ(days(0, addr(&base), 1, 0, 0, 0), sx(0x80B50002u)) << "null out";
    EXPECT_EQ(days(addr(&out), 0, 1, 0, 0, 0), sx(0x80B50002u)) << "null in";
}

TEST(Rtc, TickAddCalendarClamps) {
    register_builtin_hle();
    HleFn months = Hle::lookup(nid_hash("sceRtcTickAddMonths"));
    HleFn years = Hle::lookup(nid_hash("sceRtcTickAddYears"));
    ASSERT_NE(months, nullptr);
    ASSERT_NE(years, nullptr);
    DateTime jan31 = ymd(2024, 1, 31);
    uint64_t in = 0, out = 0;
    // Build the input tick through the registered GetTick (same registry the guest uses).
    HleFn get_tick = Hle::lookup(nid_hash("sceRtcGetTick"));
    ASSERT_NE(get_tick, nullptr);
    ASSERT_EQ(get_tick(addr(&jan31), addr(&in), 0, 0, 0, 0), 0u);
    ASSERT_EQ(months(addr(&out), addr(&in), 1, 0, 0, 0), 0u);
    // Jan 31 + 1 month = Feb 29 (2024 is leap): the clamp that matters.
    EXPECT_EQ(out, 63844761600000000ull) << "2024-02-29T00:00:00Z";
    ASSERT_EQ(months(addr(&out), addr(&in), (uint64_t)(int32_t)-1, 0, 0, 0), 0u);
    DateTime back{};
    HleFn set_tick = Hle::lookup(nid_hash("sceRtcSetTick"));
    ASSERT_NE(set_tick, nullptr);
    ASSERT_EQ(set_tick(addr(&back), addr(&out), 0, 0, 0, 0), 0u);
    EXPECT_EQ(back.year, 2023u);
    EXPECT_EQ(back.month, 12u);
    EXPECT_EQ(back.day, 31u) << "Jan 31 - 1 month = Dec 31";
    ASSERT_EQ(years(addr(&out), addr(&in), 1, 0, 0, 0), 0u);
    ASSERT_EQ(set_tick(addr(&back), addr(&out), 0, 0, 0, 0), 0u);
    EXPECT_EQ(back.year, 2025u);
    EXPECT_EQ(back.month, 1u) << "+ 1 year keeps month/day";
    EXPECT_EQ(months(addr(&out), addr(&in), 0, 0, 0, 0), 0u);
    EXPECT_EQ(out, in) << "add 0 copies";
    EXPECT_EQ(months(0, addr(&in), 1, 0, 0, 0), sx(0x80B50002u));
}

TEST(Rtc, FormatParseRoundTrip) {
    register_builtin_hle();
    HleFn format = Hle::lookup(nid_hash("sceRtcFormatRFC3339"));
    HleFn parse = Hle::lookup(nid_hash("sceRtcParseRFC3339"));
    ASSERT_NE(format, nullptr);
    ASSERT_NE(parse, nullptr);
    const uint64_t base = kTick20240101;
    EXPECT_EQ(format(0, addr(&base), 0, 0, 0, 0), sx(0x80B50002u)) << "null out";
    char buf[64];
    std::memset(buf, 0xAA, sizeof buf);
    ASSERT_EQ(format(addr(buf), addr(&base), 0, 0, 0, 0), 0u);
    EXPECT_STREQ(buf, "2024-01-01T00:00:00.00Z");
    std::memset(buf, 0, sizeof buf);
    ASSERT_EQ(format(addr(buf), addr(&base), 60, 0, 0, 0), 0u);
    EXPECT_STREQ(buf, "2024-01-01T01:00:00.00+01:00");
    uint64_t tick = 0;
    ASSERT_EQ(parse(addr(&tick), addr("2024-01-01T00:00:00.00Z"), 0, 0, 0, 0), 0u);
    EXPECT_EQ(tick, kTick20240101);
    // The corrected tz direction: +05:00 local is 5h ahead, so UTC is 5h back.
    ASSERT_EQ(parse(addr(&tick), addr("2024-01-01T05:00:00+05:00"), 0, 0, 0, 0), 0u);
    EXPECT_EQ(tick, kTick20240101);
    EXPECT_EQ(parse(addr(&tick), addr("not a date"), 0, 0, 0, 0), sx(0x80B50007u))
        << "garbage is BAD_PARSE, not an exception";
    EXPECT_EQ(parse(0, addr("2024-01-01T00:00:00.00Z"), 0, 0, 0, 0), sx(0x80B50002u));
    EXPECT_EQ(parse(addr(&tick), 0, 0, 0, 0, 0), sx(0x80B50002u));
}

TEST(Rtc, ConvertRoundTripIsIdentity) {
    register_builtin_hle();
    HleFn to_local = Hle::lookup(nid_hash("sceRtcConvertUtcToLocalTime"));
    HleFn to_utc = Hle::lookup(nid_hash("sceRtcConvertLocalTimeToUtc"));
    ASSERT_NE(to_local, nullptr);
    ASSERT_NE(to_utc, nullptr);
    // TZ-independent: whatever the host zone is, the round trip is identity.
    const uint64_t base = kTick20240101;
    uint64_t local = 0, back = 0;
    ASSERT_EQ(to_local(addr(&base), addr(&local), 0, 0, 0, 0), 0u);
    ASSERT_EQ(to_utc(addr(&local), addr(&back), 0, 0, 0, 0), 0u);
    EXPECT_EQ(back, kTick20240101);
    EXPECT_EQ(to_local(0, addr(&local), 0, 0, 0, 0), sx(0x80B50002u));
    EXPECT_EQ(to_utc(addr(&local), 0, 0, 0, 0, 0), sx(0x80B50002u));
}
