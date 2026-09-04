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

// Tri-state for an A/B experiment lever whose contract is:
//   "1"            -> forced on (1)
//   "0"            -> forced off (0)
//   unset / empty  -> fallback (-1, follow title/SDK contract)
//   anything else  -> loud refusal on stderr, keeping fallback (-1) and changing nothing
//
// Exists because strtol(e, nullptr, 0) returned 0 for non-numeric text, so every spelling an
// operator plausibly tries (=on, =true, =yes, =enabled) selected FORCED OFF rather than unset (#3304).
inline int env_tristate_or_default(const char* name, const char* text, int fallback = -1) {
    if (!text || !*text) return fallback;
    if (text[0] == '1' && text[1] == '\0') return 1;
    if (text[0] == '0' && text[1] == '\0') return 0;
    std::fprintf(stderr,
                 "[env] %s='%s' is not '0' or '1' -- keeping the default (%d) and changing NOTHING\n",
                 name, text, fallback);
    return fallback;
}

} // namespace prosper::diag
