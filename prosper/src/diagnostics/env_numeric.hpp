// env_numeric.hpp — reading a PROSPER_* variable as a NUMBER, and refusing a typo out loud.
//
// `env_cache.hpp` answers "was this set, and to what text?". This answers "what number is that
// text?", and exists because the obvious spelling is quietly wrong:
//
//     const uint64_t kib = value ? std::strtoull(value, nullptr, 10) : 8192ull;
//
// `strtoull` returns **0** for anything it cannot parse, and reports that only through an end
// pointer nobody passed. So `FOO=8mb`, `FOO=8 KB`, `FOO="8"` and `FOO=eight` all select **0**.
//
// On a knob where 0 means "off" that is merely a lost experiment. On several of prosper's it is
// worse than that, because **0 is a meaningful and MAXIMALLY AGGRESSIVE setting** — the write-watch
// family's `defer_min_bytes == 0` means "defer nothing, arm every source on first sight", and a
// promotion budget of 0 means "unbounded". A typo there does not disable the experiment; it silently
// selects a different, more aggressive one, and nothing in the run's output says so. An agent then
// attributes what it measured to the value it believed it set. That is `GAME_COMPAT_ORCHESTRATION.md`'s
// instrument-trap shape exactly, and this family has already produced one retracted measurement
// (#3155, #3253).
//
// The rule these implement is the one `PROSPER_LAZY_COMMIT_STRICT` already follows: **a malformed
// value refuses LOUDLY and keeps the default**, rather than firing at an unintended setting.
//
// Deliberately strict. The accepted grammar is exactly `[0-9]+` — no sign, no leading or trailing
// whitespace, no `0x`, no suffix. `strtoull` would take a leading space and a leading `-` (wrapping
// it to a huge unsigned), and both are far likelier to be a typo than an intention. An UNSET or
// EMPTY variable is not a typo and takes the default in silence.
//
// The `_auto` family below widens that to `0x`-hex for the six sites whose PRE-EXISTING spelling was
// `strtol(e, nullptr, 0)`. That is not a relaxation of the rule but an application of it: on those
// sites `0x2000` was already a valid input, so refusing it would be this header changing a
// well-formed setting — the very thing it exists to prevent (#3267 N1).
#pragma once
#include <cstdint>
#include <cstdio>

namespace prosper::diag {

// Parse `text` as a plain non-negative decimal integer. Returns false — leaving `*out` untouched —
// for null, empty, any non-digit character anywhere, or a value that would exceed uint64_t.
inline bool parse_u64_strict(const char* text, uint64_t* out) {
    if (!text || !*text || !out) return false;
    uint64_t value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        const uint64_t digit = static_cast<uint64_t>(*p - '0');
        if (value > (UINT64_MAX - digit) / 10u) return false;   // would overflow
        value = value * 10u + digit;
    }
    *out = value;
    return true;
}

// `text` as a number, or `fallback` — reporting the refusal on stderr, once per call, naming the
// variable, the text it was given and the default it is keeping. Pass the text rather than the name
// alone so the caller keeps its own choice of cached (`PROSPER_ENV_VALUE`) or live (`std::getenv`)
// read; `name` is for the message.
//
// `unit` is an optional trailing note for the message ("KiB", "MiB", "validations", ...), because a
// knob's units live at its call site and a reader who mistyped one wants to be told which was
// expected. It is rendered as "... is not a plain non-negative count of KiB", so pass a bare noun.
//
// `default_note` glosses the default in the message, and exists because on two knobs the default is
// itself the PERMISSIVE sentinel: `PROSPER_WRITE_WATCH_MAX_KB`'s 0 means "no limit" and
// `PROSPER_MAX_DISPATCH_GROUPS`'s means "no cap". Telling an operator who typed a value in order to
// IMPOSE a bound that we are "keeping the default (0) and changing NOTHING" is true and useless --
// they need to know they still have no bound. Pass "0 = unbounded" and the line says so.
inline uint64_t env_u64_or_default(const char* name, const char* text, uint64_t fallback,
                                   const char* unit = nullptr,
                                   const char* default_note = nullptr) {
    uint64_t value = 0;
    if (parse_u64_strict(text, &value)) return value;
    if (!text || !*text) return fallback;   // unset or empty: not a typo, and not worth a line
    std::fprintf(stderr,
                 "[env] %s='%s' is not a plain non-negative %s%s -- keeping the default (%llu%s%s) "
                 "and changing NOTHING\n",
                 name, text, unit ? "count of " : "integer", unit ? unit : "",
                 static_cast<unsigned long long>(fallback),
                 default_note ? ", " : "", default_note ? default_note : "");
    return fallback;
}

// The same, saturating at `cap` instead of overflowing a later multiply. NOTE that the saturation
// itself is SILENT -- only a malformed value is reported. A well-formed but absurd number is a
// deliberate act ("as large as possible"), where clamping is the expected answer rather than a
// surprise; a typo cannot reach here, because it was already refused above. Every byte-valued knob in
// the tree scales its number by 1024 or 1024*1024, and a value near UINT64_MAX would wrap that
// product to something small — the same class of silent wrong setting this header exists to remove.
inline uint64_t env_u64_or_default_capped(const char* name, const char* text, uint64_t fallback,
                                          uint64_t cap, const char* unit = nullptr,
                                          const char* default_note = nullptr) {
    const uint64_t value = env_u64_or_default(name, text, fallback, unit, default_note);
    return value < cap ? value : cap;
}

// --- the same, for a knob whose pre-existing grammar was strtol/strtoul BASE 0 -------------------
//
// Six sites read their value with an explicit base of 0, which makes `0x2000` a WELL-FORMED input
// there. Converting those to the decimal-only grammar above would refuse a spelling that used to
// work -- and on `PROSPER_MAX_DISPATCH_GROUPS`, whose fallback is "no cap", refusing `0x2000` would
// newly introduce the exact "a typo removes the bound" outcome this header exists to remove. So the
// grammar is widened to match what those sites already accepted, and no further: `[0-9]+` or
// `0x[0-9a-fA-F]+` (either case of the `x`), still no sign, no whitespace, no suffix, still
// overflow-checked. Base-0's OCTAL leg is deliberately not carried over -- a leading zero in a
// hand-typed count is far likelier to be padding than an intent to write base 8, and `010` meaning
// 8 is its own silent-wrong-setting hazard.
inline bool parse_u64_auto_base(const char* text, uint64_t* out) {
    if (!text || !*text || !out) return false;
    if (text[0] != '0' || (text[1] != 'x' && text[1] != 'X')) return parse_u64_strict(text, out);
    const char* p = text + 2;
    if (!*p) return false;                       // a bare "0x" is not a number
    uint64_t value = 0;
    for (; *p; ++p) {
        uint64_t digit;
        if (*p >= '0' && *p <= '9')      digit = static_cast<uint64_t>(*p - '0');
        else if (*p >= 'a' && *p <= 'f') digit = static_cast<uint64_t>(*p - 'a') + 10u;
        else if (*p >= 'A' && *p <= 'F') digit = static_cast<uint64_t>(*p - 'A') + 10u;
        else return false;
        if (value > (UINT64_MAX - digit) / 16u) return false;   // would overflow
        value = value * 16u + digit;
    }
    *out = value;
    return true;
}

inline uint64_t env_u64_or_default_auto(const char* name, const char* text, uint64_t fallback,
                                        const char* unit = nullptr,
                                        const char* default_note = nullptr) {
    uint64_t value = 0;
    if (parse_u64_auto_base(text, &value)) return value;
    if (!text || !*text) return fallback;
    std::fprintf(stderr,
                 "[env] %s='%s' is not a decimal or 0x-hex non-negative %s%s -- keeping the default "
                 "(%llu%s%s) and changing NOTHING\n",
                 name, text, unit ? "count of " : "integer", unit ? unit : "",
                 static_cast<unsigned long long>(fallback),
                 default_note ? ", " : "", default_note ? default_note : "");
    return fallback;
}

inline uint64_t env_u64_or_default_auto_capped(const char* name, const char* text, uint64_t fallback,
                                               uint64_t cap, const char* unit = nullptr,
                                               const char* default_note = nullptr) {
    const uint64_t value = env_u64_or_default_auto(name, text, fallback, unit, default_note);
    return value < cap ? value : cap;
}

// --- for a knob whose three answers are ON, OFF and "not set at all" ------------------------------
//
// A tri-state A/B lever (`PROSPER_POST_SUBMIT_VISIBILITY`, #1226/#2217) has a deliberate third
// answer: `1` forces the contract on, `0` forces it off, and UNSET means "follow the SDK version the
// guest asked for". Read with `strtol(e, nullptr, 0)` that collapses, because strtol answers **0**
// for text it cannot parse -- so `=on`, `=true`, `=yes` and `=enabled`, every spelling an operator
// reaches for, select the FORCED-OFF arm, and the site then announces "FORCED OFF" as though it had
// been asked for. On an ordinary switch that is a lost experiment. On a lever whose verdict is still
// open it is a MISLABELLED measurement, which is worse than no measurement at all (#3304).
//
// So the grammar is: `1` / `on` / `true` / `yes` / `enabled` -> 1, `0` / `off` / `false` / `no` /
// `disabled` -> 0, either case; `0x1` and `0x0` too, because the sites converted here read base 0
// and refusing a spelling that used to work is this header changing a well-formed setting (the same
// N1 rule the `_auto` family exists for). Anything else -- INCLUDING a number that is neither 0 nor
// 1, which strtol used to turn into a silent forced-ON -- is **unset** (-1) with a line on stderr
// naming the value. Never a silent third answer. Unset or empty is -1 in silence.
inline bool env_word_matches_nocase(const char* text, const char* lower_word) {
    for (;; ++text, ++lower_word) {
        if (!*lower_word) return !*text;
        char c = *text;
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        if (c != *lower_word) return false;
    }
}

inline int env_tristate_or_unset(const char* name, const char* text) {
    if (!text || !*text) return -1;   // unset or empty: not a typo, and not a third arm
    static const char* const kOn[] = {"on", "true", "yes", "enabled"};
    static const char* const kOff[] = {"off", "false", "no", "disabled"};
    for (const char* word : kOn)
        if (env_word_matches_nocase(text, word)) return 1;
    for (const char* word : kOff)
        if (env_word_matches_nocase(text, word)) return 0;
    uint64_t value = 0;
    if (parse_u64_auto_base(text, &value) && value <= 1) return static_cast<int>(value);
    std::fprintf(stderr,
                 "[env] %s='%s' is neither on nor off -- treating it as UNSET and changing NOTHING. "
                 "Accepted: 1/on/true/yes/enabled, 0/off/false/no/disabled (either case)\n",
                 name, text);
    return -1;
}

// --- for a knob whose "unset" answer is NOT a number ---------------------------------------------
//
// `PROSPER_COMPUTE_IMAGE_CACHE_MB` unset means "derive the budget from device memory", so there is
// no fallback to hand the functions above -- the refusal has to fall back to a different CODE PATH.
// Returns true only on a well-formed value; reports and returns false on a malformed one; returns
// false in silence when unset or empty. Exists as a named helper rather than a hand-written
// fprintf at the call site so that tools/env/check_env_numeric_arms.py can SEE the site: a bespoke
// parse is invisible to that gate, which is how the one site with bespoke arithmetic became the one
// site the anti-drift gate did not cover.
inline bool env_u64_or_report(const char* name, const char* text, uint64_t* out,
                              const char* unit = nullptr, const char* unset_note = nullptr) {
    if (!text || !*text) return false;
    if (parse_u64_strict(text, out)) return true;
    std::fprintf(stderr,
                 "[env] %s='%s' is not a plain non-negative %s%s -- keeping %s and changing "
                 "NOTHING\n",
                 name, text, unit ? "count of " : "integer", unit ? unit : "",
                 unset_note ? unset_note : "the default");
    return false;
}

} // namespace prosper::diag
