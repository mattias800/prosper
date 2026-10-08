// console_oracle.hpp -- read the console-oracle golden files and replay a case through prosper's HLE.
//
// A golden file (tests/data/console_oracle/<family>.golden.tsv) records what a REAL PS5 returned for a
// table of library calls, measured by tools/console_oracle (see its AGENTS.md and the grammar at the
// top of payload/lib_oracle.c). This header rebuilds the same arguments in host memory, calls the HLE
// handler registered for the function, and renders the result in the payload's own format so the two
// can be compared string for string. Header-only and free of any gtest dependency.
#pragma once

#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace prosper_test::console_oracle {

constexpr int kMaxArgs = 6;
constexpr uint8_t kSentinel = 0xAB;

struct Arg {
    enum class Kind { Int, Null, In, Out, Str, OutPtr } kind = Kind::Null;
    uint64_t value = 0;   // the register value passed
    std::vector<uint8_t> buf;   // backing store; never resized after its address is taken
    size_t len = 0;   // reported length of buf (a string's NUL is excluded)
    int ref = 0;   // OutPtr: argument whose buffer the pointer is reported relative to
};

struct GoldenCase {
    std::string id, lib, func, args, expect, status, retn;
    uint64_t ret = 0;
    std::vector<std::string> buffers;   // "a<k>=<value>" fields as the payload printed them
};

struct Outcome {
    bool implemented = false;
    uint64_t ret = 0;
    std::string retn = "-";
    std::vector<std::string> buffers;
};

using State = std::map<std::string, std::vector<Arg>>;   // case id -> its arguments after the call

inline std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        const size_t p = s.find(sep, start);
        if (p == std::string::npos) {
            out.push_back(s.substr(start));
            return out;
        }
        out.push_back(s.substr(start, p - start));
        start = p + 1;
    }
}

inline int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

inline bool decode_hex(const std::string& s, std::vector<uint8_t>* out) {
    if (s.size() % 2) return false;
    for (size_t i = 0; i < s.size(); i += 2) {
        const int hi = hexval(s[i]), lo = hexval(s[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out->push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return true;
}

// \xHH and \\ escapes; everything else is literal. Mirrors lib_oracle.c's decode_text.
inline std::vector<uint8_t> decode_text(const std::string& s) {
    std::vector<uint8_t> out;
    for (size_t i = 0; i < s.size();) {
        if (s[i] == '\\' && i + 3 < s.size() && s[i + 1] == 'x' && hexval(s[i + 2]) >= 0 &&
            hexval(s[i + 3]) >= 0) {
            out.push_back(static_cast<uint8_t>((hexval(s[i + 2]) << 4) | hexval(s[i + 3])));
            i += 4;
        } else if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == '\\') {
            out.push_back('\\');
            i += 2;
        } else {
            out.push_back(static_cast<uint8_t>(s[i++]));
        }
    }
    return out;
}

inline bool parse_int(const std::string& s, uint64_t* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    *out = (s[0] == '-') ? static_cast<uint64_t>(std::strtoll(s.c_str(), &end, 0))
                         : static_cast<uint64_t>(std::strtoull(s.c_str(), &end, 0));
    return end && *end == '\0';
}

inline void stabilise(Arg* a, size_t len, size_t slack) {
    a->buf.assign(len + slack, 0);
    a->len = len;
    a->value = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(a->buf.data()));
}

inline bool parse_arg(const std::string& tok, const State& prior, Arg* a, std::string* err) {
    if (tok == "null") {
        a->kind = Arg::Kind::Null;
        return true;
    }
    const size_t colon = tok.find(':');
    if (colon == std::string::npos) {
        *err = "no ':' in token '" + tok + "'";
        return false;
    }
    const std::string key = tok.substr(0, colon), val = tok.substr(colon + 1);
    uint64_t v = 0;
    if (key == "i") {
        if (!parse_int(val, &v)) return *err = "bad integer '" + val + "'", false;
        a->kind = Arg::Kind::Int;
        a->value = v;
    } else if (key == "in") {
        std::vector<uint8_t> bytes;
        if (!decode_hex(val, &bytes)) return *err = "bad hex '" + val + "'", false;
        a->kind = Arg::Kind::In;
        stabilise(a, bytes.size(), 16);
        std::memcpy(a->buf.data(), bytes.data(), bytes.size());
    } else if (key == "p32" || key == "p64") {
        const size_t w = key == "p32" ? 4 : 8;
        if (!parse_int(val, &v)) return *err = "bad integer '" + val + "'", false;
        a->kind = Arg::Kind::In;
        stabilise(a, w, 16);
        std::memcpy(a->buf.data(), &v, w);   // little endian host
    } else if (key == "out") {
        if (!parse_int(val, &v) || v == 0 || v > 4096)
            return *err = "bad out size '" + val + "'", false;
        a->kind = Arg::Kind::Out;
        stabilise(a, static_cast<size_t>(v), 16);
        std::memset(a->buf.data(), kSentinel, a->len);
    } else if (key == "s") {
        const std::vector<uint8_t> bytes = decode_text(val);
        a->kind = Arg::Kind::Str;
        stabilise(a, bytes.size(), 16);
        std::memcpy(a->buf.data(), bytes.data(), bytes.size());
    } else if (key == "outptr") {
        if (!parse_int(val, &v) || v >= kMaxArgs)
            return *err = "bad outptr ref '" + val + "'", false;
        a->kind = Arg::Kind::OutPtr;
        stabilise(a, 8, 16);
        std::memset(a->buf.data(), kSentinel, 8);
        a->ref = static_cast<int>(v);
    } else if (key == "use") {
        const std::vector<std::string> p = split(val, '.');
        uint64_t k = 0, off = 0, size = 0;
        if (p.size() != 4 || !parse_int(p[1], &k) || !parse_int(p[2], &off) ||
            !parse_int(p[3], &size))
            return *err = "bad use spec '" + val + "'", false;
        const auto it = prior.find(p[0]);
        if (it == prior.end() || k >= it->second.size() || (size != 4 && size != 8) ||
            off + size > it->second[k].len)
            return *err = "use:" + val + " does not name an earlier buffer", false;
        a->kind = Arg::Kind::Int;
        a->value = 0;
        std::memcpy(&a->value, it->second[k].buf.data() + off, size);
    } else {
        return *err = "unknown token kind '" + key + "'", false;
    }
    return true;
}

// ptr+<off> | null | ptr:foreign, relative to `base`. Same rule as lib_oracle.c.
inline std::string rel(uint64_t v, const Arg* base) {
    if (v == 0) return "null";
    if (base && !base->buf.empty()) {
        const uint64_t b = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base->buf.data()));
        if (v >= b && v <= b + base->len) return "ptr+" + std::to_string(v - b);
    }
    return "ptr:foreign";
}

inline std::string hex(const std::vector<uint8_t>& b, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += d[b[i] >> 4];
        s += d[b[i] & 15];
    }
    return s;
}

inline bool has_flag(const std::string& expect, const std::string& flag) {
    for (const auto& f : split(expect, ','))
        if (f == flag) return true;
    return false;
}

inline int flag_int(const std::string& expect, const std::string& prefix) {
    for (const auto& f : split(expect, ','))
        if (f.compare(0, prefix.size(), prefix) == 0) return std::atoi(f.c_str() + prefix.size());
    return -1;
}

// Run one case through the HLE handler. Returns false (with `err`) when the case itself is malformed;
// an unregistered function is NOT an error here -- it is reported as !outcome->implemented.
inline bool replay_case(const GoldenCase& c, State& state, Outcome* outcome, std::string* err) {
    std::vector<Arg> args;
    if (c.args != "-" && !c.args.empty()) {
        for (const auto& tok : split(c.args, ',')) {
            if (static_cast<int>(args.size()) >= kMaxArgs)
                return *err = "more than 6 arguments", false;
            Arg a;
            if (!parse_arg(tok, state, &a, err)) return false;
            args.push_back(std::move(a));
        }
    }
    for (const auto& a : args)
        if (a.kind == Arg::Kind::OutPtr &&
            (a.ref >= static_cast<int>(args.size()) || args[a.ref].buf.empty()))
            return *err = "outptr reference has no buffer", false;

    const prosper::HleFn fn = prosper::Hle::lookup(prosper::nid_hash(c.func));
    outcome->implemented = fn != nullptr;
    if (fn) {
        uint64_t a[kMaxArgs] = {0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < args.size(); i++) a[i] = args[i].value;
        outcome->ret = fn(a[0], a[1], a[2], a[3], a[4], a[5]);

        const int k = flag_int(c.expect, "retoff:");
        if (k >= 0 && k < static_cast<int>(args.size()) && !args[k].buf.empty())
            outcome->retn = rel(outcome->ret, &args[k]);
        for (size_t i = 0; i < args.size(); i++) {
            const Arg& x = args[i];
            if (x.kind == Arg::Kind::In || x.kind == Arg::Kind::Out) {
                outcome->buffers.push_back("a" + std::to_string(i) + "=" + hex(x.buf, x.len));
            } else if (x.kind == Arg::Kind::OutPtr) {
                bool untouched = true;
                for (size_t j = 0; j < 8; j++)
                    if (x.buf[j] != kSentinel) untouched = false;
                uint64_t v = 0;
                std::memcpy(&v, x.buf.data(), 8);
                outcome->buffers.push_back("a" + std::to_string(i) + "=" +
                                           (untouched ? "untouched" : rel(v, &args[x.ref])));
            }
        }
    }
    state[c.id] = std::move(args);
    return true;
}

inline std::string hex64(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%016llx", static_cast<unsigned long long>(v));
    return b;
}

// "" when prosper agrees with the console under the case's expect flags, else a one-line diff.
inline std::string compare(const GoldenCase& c, const Outcome& o) {
    if (has_flag(c.expect, "none")) return "";
    if (!o.implemented) return "not implemented in prosper";
    std::ostringstream d;
    const uint64_t mask = has_flag(c.expect, "r64") ? ~0ull : 0xFFFFFFFFull;
    // A pointer return differs between address spaces by construction, so a retoff:<k> case compares
    // the normalised offset (retn) instead of the raw register.
    const bool pointer_return = flag_int(c.expect, "retoff:") >= 0;
    if (!pointer_return && (o.ret & mask) != (c.ret & mask))
        d << "ret console=" << hex64(c.ret & mask) << " prosper=" << hex64(o.ret & mask) << "; ";
    if (pointer_return && o.retn != c.retn)
        d << "retn console=" << c.retn << " prosper=" << o.retn << "; ";
    if (!has_flag(c.expect, "ret") && o.buffers != c.buffers) {
        for (size_t i = 0; i < std::max(o.buffers.size(), c.buffers.size()); i++) {
            const std::string want = i < c.buffers.size() ? c.buffers[i] : "<none>";
            const std::string got = i < o.buffers.size() ? o.buffers[i] : "<none>";
            if (want != got) d << "console " << want << " / prosper " << got << "; ";
        }
    }
    return d.str();
}

inline bool load_golden(const std::string& path, std::vector<GoldenCase>* out, std::string* err) {
    std::ifstream in(path);
    if (!in) return *err = "cannot open " + path, false;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> f = split(line, '\t');
        if (f.size() < 8) return *err = "short golden line: " + line, false;
        GoldenCase c;
        c.id = f[0], c.lib = f[1], c.func = f[2], c.args = f[3], c.expect = f[4], c.status = f[5];
        c.ret = std::strtoull(f[6].c_str(), nullptr, 16);
        c.retn = f[7];
        c.buffers.assign(f.begin() + 8, f.end());
        out->push_back(std::move(c));
    }
    return true;
}

// known_gaps.tsv: `id<TAB>reason`. Every entry needs a reason (an issue number or an explanation).
inline bool load_known_gaps(const std::string& path, std::map<std::string, std::string>* out,
                            std::string* err) {
    std::ifstream in(path);
    if (!in) return true;   // no file == no known gaps
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> f = split(line, '\t');
        if (f.size() < 2 || f[0].empty() || f[1].empty())
            return *err = "known gap needs 'id<TAB>reason': " + line, false;
        (*out)[f[0]] = f[1];
    }
    return true;
}

}   // namespace prosper_test::console_oracle
