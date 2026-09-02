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
// `unit` is an optional trailing note for the message ("KiB", "MiB", ...), because a knob's units
// live at its call site and a reader who mistyped one wants to be told which was expected.
inline uint64_t env_u64_or_default(const char* name, const char* text, uint64_t fallback,
                                   const char* unit = nullptr) {
    uint64_t value = 0;
    if (parse_u64_strict(text, &value)) return value;
    if (!text || !*text) return fallback;   // unset or empty: not a typo, and not worth a line
    std::fprintf(stderr,
                 "[env] %s='%s' is not a plain non-negative integer%s%s -- keeping the default "
                 "(%llu) and changing NOTHING\n",
                 name, text, unit ? " of " : "", unit ? unit : "",
                 static_cast<unsigned long long>(fallback));
    return fallback;
}

// The same, saturating at `cap` instead of overflowing a later multiply. Every byte-valued knob in
// the tree scales its number by 1024 or 1024*1024, and a value near UINT64_MAX would wrap that
// product to something small — the same class of silent wrong setting this header exists to remove.
inline uint64_t env_u64_or_default_capped(const char* name, const char* text, uint64_t fallback,
                                          uint64_t cap, const char* unit = nullptr) {
    const uint64_t value = env_u64_or_default(name, text, fallback, unit);
    return value < cap ? value : cap;
}

} // namespace prosper::diag
