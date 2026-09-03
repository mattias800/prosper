#include "host/abi/guest_varargs.hpp"
#include <cstring>

namespace prosper::abi {

namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }
// ' is the SUSv2 thousands-grouping flag; the rest are C's.
bool is_printf_flag(char c) {
    return c == '-' || c == '+' || c == ' ' || c == '#' || c == '0' || c == '\'';
}
bool is_float_conversion(char c) {
    return c == 'e' || c == 'E' || c == 'f' || c == 'F' ||
           c == 'g' || c == 'G' || c == 'a' || c == 'A';
}
// Conversions that consume exactly one integer-file slot: an int of some width, a pointer, or a
// character promoted to int. `S` and `C` are the BSD wide forms, which are still one pointer/int.
bool is_integer_conversion(char c) {
    return c == 'd' || c == 'i' || c == 'o' || c == 'u' || c == 'x' || c == 'X' ||
           c == 'c' || c == 'C' || c == 's' || c == 'S' || c == 'p' || c == 'n';
}

// THE RULE THIS WHOLE FILE OBEYS: what is accepted here must be a SUBSET of what the host CRT
// consumes an argument for. A guest that lies about its own format string gets the same garbage it
// would get on hardware, and that is its business — but if THIS walker and the CRT disagree about
// the same string, the CRT reads a slot the walker never wrote or skips one it did, and every
// argument behind that point is wrong. That is precisely the failure this file exists to remove,
// reintroduced one level up. So an extension is accepted only once it has been MEASURED, and
// anything unmeasured falls through to the refusal path, which truncates rather than desynchronises.
//
// Measured 2026-09-03 against the MinGW CRT under wine (tools/probe_win_varargs.cpp's toolchain), by
// asking whether a trailing sentinel argument still lands where the caller put it:
//
//   %zu %jd %td %hhd %a %S %C   consume one argument  -> safe to accept
//   %'d                         consumes one argument -> accepted, but it is the one entry resting
//                                                        on MinGW's ANSI stdio rather than on
//                                                        standard C; re-measure if the CRT changes
//   %qd                         prints "%qd" LITERALLY and consumes NOTHING -> must be refused
//
// `q` is therefore absent from the switch below on purpose: a BSD long-long modifier the guest may
// well emit (332 occurrences across 50 dumps) that the host CRT does not implement. Refusing it
// costs the tail of one line; accepting it would shift every argument behind it.

// Skip a length modifier, reporting whether it was the x87 `L`. `L` matters because a System V
// `long double` is an 80-bit x87 value passed in MEMORY with 16-byte alignment — neither an integer
// slot nor an SSE one — so a format carrying it is refused rather than mis-read as a double.
const char* skip_length(const char* p, bool* long_double) {
    *long_double = false;
    switch (*p) {
    case 'h': ++p; if (*p == 'h') ++p; break;
    case 'l': ++p; if (*p == 'l') ++p; break;
    case 'L': *long_double = true; ++p; break;
    case 'j': case 'z': case 't': ++p; break;   // NOT 'q' — see the subset rule above
    default: break;
    }
    return p;
}

// A conversion beginning at `p` (just past the '%'): is it a POSIX positional `%2$d`? Those name
// their argument rather than consuming the next one, so a sequential model would read the wrong
// slots for every conversion in the string.
bool is_positional(const char* p) {
    const char* q = p;
    while (is_digit(*q)) ++q;
    return q != p && *q == '$';
}

} // namespace

uint64_t sysv_va_arg(SysvVaList& ap, VarargClass cls) {
    uint64_t value = 0;
    if (cls == VarargClass::Sse) {
        // An SSE argument occupies a 16-byte slot in the save area but only ever eight meaningful
        // bytes: a double is the whole slot, and a float was promoted to double by the caller.
        if (ap.fp_offset + 8 <= kSysvRegSaveEnd) {
            std::memcpy(&value, (const void*)(uintptr_t)(ap.reg_save_area + ap.fp_offset), 8);
            ap.fp_offset += 16;
            return value;
        }
    } else if (ap.gp_offset + 8 <= kSysvGpSaveEnd) {
        std::memcpy(&value, (const void*)(uintptr_t)(ap.reg_save_area + ap.gp_offset), 8);
        ap.gp_offset += 8;
        return value;
    }
    // Both files spill to the same overflow area, in declaration order, eight bytes each.
    std::memcpy(&value, (const void*)(uintptr_t)ap.overflow_arg_area, 8);
    ap.overflow_arg_area += 8;
    return value;
}

FormatPlan plan_format(const char* fmt, FormatGrammar grammar) {
    FormatPlan plan{};
    if (!fmt) {
        plan.complete = false;
        plan.reject = "null format string";
        return plan;
    }

    const char* p = fmt;
    // Refuse everything from `spec` onward, rewinding to the argument count the accepted prefix
    // consumed: a `*` width pushes its argument before the conversion character is known, so a
    // half-parsed conversion must not leave that push behind.
    auto refuse = [&](const char* spec, unsigned accepted_args, const char* why) {
        plan.count = accepted_args;
        plan.complete = false;
        plan.reject = why;
        plan.modelled_bytes = (size_t)(spec - fmt);
        return plan;
    };
    auto push = [&](VarargClass cls) {
        if (plan.count >= kMaxFormatArgs) return false;
        plan.cls[plan.count++] = cls;
        return true;
    };

    while (*p) {
        if (*p != '%') { ++p; continue; }
        const char* const spec = p;
        const unsigned before = plan.count;
        ++p;
        if (*p == '%') { ++p; continue; }        // a literal percent consumes nothing
        if (!*p) return refuse(spec, before, "format string ends inside a conversion");
        if (is_positional(p)) return refuse(spec, before, "POSIX positional conversion (%n$)");

        if (grammar == FormatGrammar::Scanf) {
            // %[*][width][length]conv, and a scanset. Every conversion that assigns takes a POINTER,
            // so the class is never in doubt — only whether an argument is consumed at all.
            const bool suppressed = (*p == '*');
            if (suppressed) ++p;
            while (is_digit(*p)) ++p;
            bool long_double = false;
            p = skip_length(p, &long_double);
            if (!*p) return refuse(spec, before, "format string ends inside a conversion");
            if (*p == '[') {
                // A scanset: ']' loses its meaning immediately after '[' or "[^".
                ++p;
                if (*p == '^') ++p;
                if (*p == ']') ++p;
                while (*p && *p != ']') ++p;
                if (!*p) return refuse(spec, before, "unterminated scanset");
                ++p;
            } else if (is_integer_conversion(*p) || is_float_conversion(*p)) {
                ++p;
            } else {
                return refuse(spec, before, "unrecognised scanf conversion");
            }
            if (!suppressed && !push(VarargClass::Integer))
                return refuse(spec, before, "more arguments than the model holds");
            plan.modelled_bytes = (size_t)(p - fmt);
            continue;
        }

        while (is_printf_flag(*p)) ++p;
        if (*p == '*') {
            if (!push(VarargClass::Integer)) return refuse(spec, before, "more arguments than the model holds");
            ++p;
        } else {
            while (is_digit(*p)) ++p;
        }
        if (*p == '.') {
            ++p;
            if (*p == '*') {
                if (!push(VarargClass::Integer))
                    return refuse(spec, before, "more arguments than the model holds");
                ++p;
            } else {
                while (is_digit(*p)) ++p;
            }
        }
        bool long_double = false;
        p = skip_length(p, &long_double);
        const char conv = *p;
        if (!conv) return refuse(spec, before, "format string ends inside a conversion");
        if (is_float_conversion(conv)) {
            if (long_double) return refuse(spec, before, "x87 long double conversion (%L)");
            if (!push(VarargClass::Sse)) return refuse(spec, before, "more arguments than the model holds");
        } else if (is_integer_conversion(conv)) {
            if (!push(VarargClass::Integer))
                return refuse(spec, before, "more arguments than the model holds");
        } else {
            // Deliberately conservative. An unknown conversion may or may not consume an argument,
            // and guessing wrong shifts every argument behind it.
            return refuse(spec, before, "unrecognised printf conversion");
        }
        ++p;
        plan.modelled_bytes = (size_t)(p - fmt);
    }

    plan.modelled_bytes = (size_t)(p - fmt);
    return plan;
}

void pack_ms_va_slots(const FormatPlan& plan, SysvVaList ap, uint64_t* slots) {
    for (unsigned i = 0; i < plan.count; ++i) slots[i] = sysv_va_arg(ap, plan.cls[i]);
}

MsVarargCall::MsVarargCall(const char* fmt, const SysvVaList& ap, FormatGrammar grammar)
    : plan_(plan_format(fmt, grammar)) {
    complete_ = plan_.complete;
    reject_   = plan_.reject;
    if (plan_.complete) {
        pack_ms_va_slots(plan_, ap, slots_);
        format_ = fmt;
        return;
    }

    size_t keep = plan_.modelled_bytes;
    if (keep >= kMaxPrefixBytes) keep = kMaxPrefixBytes - 1;
    if (keep && fmt) std::memcpy(prefix_, fmt, keep);
    prefix_[keep] = '\0';

    // Re-plan the PREFIX rather than trusting the cap. `modelled_bytes` is always a conversion
    // boundary, but kMaxPrefixBytes is not, so a long format can be cut in the middle of a `%...`
    // spec — and the slot count planned for the whole format would then no longer describe the
    // string actually handed to the CRT. Planning what will really be formatted keeps the two
    // definitionally in step, and shortens the prefix again when the cut landed inside a conversion.
    // The prefix's own plan may come back `complete`, since the cut can have removed the very thing
    // that was refused; the CALLER's verdict stays the original one, which is why it is captured
    // above rather than read back from here.
    plan_ = plan_format(prefix_, grammar);
    prefix_[plan_.modelled_bytes] = '\0';
    pack_ms_va_slots(plan_, ap, slots_);
    format_ = prefix_;
}

} // namespace prosper::abi
