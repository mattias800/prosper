#pragma once

// Project-owned PC-relative dispatch followed by a counted loop and a fresh
// Bool-only VCC generation. No captured shader bytes or guest resources are used.
#include "fragment_compare_kill.hpp"
#include <cstdint>
#include <vector>

namespace prosper::test::fragment_dispatch_bool_copy {
constexpr uint32_t end = 0xbf810000u;
constexpr uint32_t mov(uint32_t destination, uint32_t source, bool pair = false) {
    return 0xbe800000u | (destination << 16) | ((pair ? 4u : 3u) << 8) | source;
}
enum class Tail {
    Bool,
    LowData,
    HighData,
    RelayData,
    OneArmReplacement,
    CompleteReplacement,
    NumericPair
};
enum class EarlyOut {
    Sound,
    Killed,
    NonTerminatingInteriorLoop,
    OverflowInteriorLoop,
    WrongSignInteriorLoop,
    ReplacedInductionWord,
    SkippedInductionStep,
    SkippedInductionInitializer,
    MissingRestore,
    StaleSurvivor,
    HighSurvivorOverwrite,
    OverlappingSurvivorOverwrite,
    ConditionalSurvivorOverwrite,
    BypassFinalGate,
    NumericScc,
    Scc1,
    TailReentry,
    TailData,
    LateStore,
    TailStore,
    ExportBeforeGate,
    ReenableAfterGate
};
inline std::vector<uint32_t> program(Tail tail = Tail::Bool, bool equal = false,
                                     uint32_t iterations = 3, bool mask_early_out = false,
                                     EarlyOut early = EarlyOut::Sound) {
    std::vector<uint32_t> code{
        0xf4201a8cu, 0xfa000010u, // selector load
        0x816ac16au, 0x83ea826au, 0x8f6a836au, 0xbea01f00u, 0x802020ffu, 0u,
        0x82212180u, 0xf4040890u, 0xd4000000u, 0xbea81f00u, 0x80282228u, 0x82292329u,
        0xbe802028u, 0x7e000280u, 0xbf820000u, // alternate prefix joins the selected routine
    };
    // A numeric read of the EARLIER VCC generation keeps the global word demand
    // present. It must not require physical words from the later Bool-only copy.
    code.insert(code.end(), {mov(106, 129, true), mov(30, 106)});
    size_t early_branch = 0;
    if (mask_early_out) {
        code.insert(
            code.end(),
            {mov(44, early == EarlyOut::NumericScc ? 129 : 126, true), 0x7e000281u,
             (early == EarlyOut::Killed || early == EarlyOut::NonTerminatingInteriorLoop ||
              early == EarlyOut::OverflowInteriorLoop || early == EarlyOut::WrongSignInteriorLoop ||
              early == EarlyOut::ReplacedInductionWord || early == EarlyOut::SkippedInductionStep ||
              early == EarlyOut::SkippedInductionInitializer)
                 ? 0x7e020281u
                 : 0x7e020280u,
             0x7d840300u, 0x8aac6a2cu});
        early_branch = code.size();
        code.push_back(0);
    }
    if (early == EarlyOut::SkippedInductionInitializer) code.push_back(0xbf820001u);
    code.push_back(mov(2, 128));
    const uint32_t loop_header = static_cast<uint32_t>(code.size());
    code.push_back(early == EarlyOut::OverflowInteriorLoop ||
                           early == EarlyOut::WrongSignInteriorLoop
                       ? 0xbf0ac102u // unsigned LT UINT32_MAX
                       : 0xbf0a0002u | ((128u + iterations) << 8));
    const size_t loop_exit = code.size();
    code.push_back(0);
    if (early == EarlyOut::SkippedInductionStep) code.push_back(0xbf820001u);
    const uint32_t step = early == EarlyOut::NonTerminatingInteriorLoop ? 128u
                          : early == EarlyOut::OverflowInteriorLoop     ? 130u
                          : early == EarlyOut::WrongSignInteriorLoop    ? 194u // -2
                                                                        : 129u;
    code.push_back(0x80020002u | (step << 8));
    if (early == EarlyOut::ReplacedInductionWord) code.push_back(mov(2, 128));
    const uint32_t backedge = static_cast<uint32_t>(code.size());
    code.push_back(0xbf820000u | ((loop_header - backedge - 1u) & 0xffffu));
    code[loop_exit] =
        0xbf840000u | (static_cast<uint32_t>(code.size()) - static_cast<uint32_t>(loop_exit) - 1u);
    code.insert(code.end(), {0x7e000281u, equal ? 0x7e020281u : 0x7e020280u});
    if (tail == Tail::NumericPair)
        code.push_back(mov(20, 129, true));
    else
        code.insert(code.end(),
                    {fragment_compare_kill::compare(), fragment_compare_kill::compare_sd(20)});
    const uint32_t saved_copy = static_cast<uint32_t>(code.size());
    code.push_back(mov(106, 20, true));
    switch (tail) {
        case Tail::LowData: code.push_back(mov(31, 106)); break;
        case Tail::HighData: code.push_back(mov(31, 107)); break;
        case Tail::RelayData: code.insert(code.end(), {mov(24, 106, true), mov(31, 25)}); break;
        case Tail::OneArmReplacement:
            code.insert(code.end(), {0xbf840001u, mov(106, 129, true), mov(31, 107)});
            break;
        case Tail::CompleteReplacement:
            code.insert(code.end(), {mov(106, 129, true), mov(31, 107)});
            break;
        default: break;
    }
    code.insert(code.end(), {0xd5010000u, 129u | (137u << 9) | (106u << 18)});
    if (mask_early_out) {
        if (early == EarlyOut::StaleSurvivor) code.push_back(mov(44, 193, true));
        if (early == EarlyOut::HighSurvivorOverwrite) code.push_back(mov(45, 129));
        if (early == EarlyOut::OverlappingSurvivorOverwrite) code.push_back(mov(43, 193, true));
        if (early == EarlyOut::ConditionalSurvivorOverwrite)
            code.insert(code.end(), {0xbf840001u, mov(44, 193, true)});
        if (early == EarlyOut::BypassFinalGate) code.push_back(0xbf840001u);
        if (early == EarlyOut::LateStore)
            code.insert(code.end(),
                        {0xe0700000u, 0x80020000u}); // buffer_store_dword v0, off, s[8:11], 0
        if (early == EarlyOut::ExportBeforeGate) code.insert(code.end(), {0xf800080fu, 0u});
        if (early != EarlyOut::MissingRestore) code.push_back(mov(126, 44, true));
        if (early == EarlyOut::ReenableAfterGate) code.push_back(mov(126, 193, true));
    }
    const uint32_t merge = static_cast<uint32_t>(code.size());
    code.insert(code.end(), {0xf800080fu, 0u, end});
    if (mask_early_out) {
        const uint32_t terminal = static_cast<uint32_t>(code.size());
        code[early_branch] =
            (early == EarlyOut::Scc1 ? 0xbf850000u : 0xbf840000u) | (terminal - early_branch - 1u);
        if (early == EarlyOut::TailData) code.push_back(mov(31, 107));
        if (early == EarlyOut::TailStore) code.insert(code.end(), {0xe0700000u, 0x80020000u});
        if (early == EarlyOut::TailReentry)
            code.push_back(0xbf820000u |
                           ((saved_copy - static_cast<uint32_t>(code.size()) - 1u) & 0xffffu));
        else
            code.insert(code.end(), {mov(126, 128, true), 0xf8001c00u, 0u, end});
        // The external arm kills all lanes, enables no export channels and cannot re-enter.
    }
    if (code.size() % 2u) code.push_back(0u);
    const uint32_t table = static_cast<uint32_t>(code.size());
    code[7] = table * 4u - 24u;   // table GETPC returns byte24
    code.insert(code.end(), {(17u - 12u) * 4u, 0u, (15u - 12u) * 4u, 0u, (merge - 12u) * 4u, 0u});
    return code;
}

inline std::vector<uint32_t> retained_copy_before_dispatch(bool genuine_pair = false) {
    auto code = program();
    // The original SETPC observes every current word. Specialization removes it and
    // the selected routine replaces VCC before its numeric read, so selected deadness
    // alone incorrectly certifies the retained PC6 copy. The original proof must win.
    // Eight prefix dwords preserve all PC-relative table offsets and qword alignment.
    const std::vector<uint32_t> prefix{
        0xbf800000u,
        0xbf800000u,
        0x7e000281u,
        0x7e020280u,
        genuine_pair ? mov(20, 129, true) : fragment_compare_kill::compare(),
        genuine_pair ? 0xbf800000u : fragment_compare_kill::compare_sd(20),
        mov(106, 20, true),
        0xbf800000u};
    code.insert(code.begin(), prefix.begin(), prefix.end());
    return code;
}
}   // namespace prosper::test::fragment_dispatch_bool_copy
