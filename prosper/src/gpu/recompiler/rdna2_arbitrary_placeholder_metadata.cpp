// Compile-time read sites in the chosen module; this parser grants no admission.
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include <set>
#include <string>
#include <string_view>

namespace prosper::gpu {
ArbitraryPlaceholderReads arbitrary_placeholder_reads(std::span<const uint32_t> module) {
    ArbitraryPlaceholderReads result;
    if (module.size() < 5 || module[0] != 0x07230203u) return result;
    std::set<std::string> sites;
    for (size_t at = 5; at < module.size();) {
        const uint32_t length = module[at] >> 16u;
        if (!length || length > module.size() - at) return {};
        if ((module[at] & 0xffffu) == 330 && length > 1) {
            std::string text;
            bool ended = false;
            for (size_t word = at + 1; word < at + length && !ended; ++word)
                for (unsigned byte = 0; byte < 4; ++byte) {
                    const char c = static_cast<char>((module[word] >> (8 * byte)) & 255u);
                    if (!c) {
                        ended = true;
                        break;
                    }
                    text += c;
                }
            if (ended) sites.insert(std::move(text));
        }
        at += length;
    }
    for (const auto& site : sites) {
        const std::string_view text = site;
        if (text.starts_with("Prosper.ArbitraryPlaceholderRead=fresh-mask:")) ++result.fresh_mask;
        if (text.starts_with("Prosper.ArbitraryPlaceholderRead=loop-carry:")) ++result.loop_carry;
    }
    return result;
}
} // namespace prosper::gpu
