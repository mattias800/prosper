// Pure PROSPER_REGWATCH selector parsing; reporting and live configuration stay in the processor.
#include "gpu/pm4/command_processor.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <string>

namespace prosper::gpu {

std::vector<RegWatchEntry> parse_reg_watch(const char* setting) {
    std::vector<RegWatchEntry> entries;
    if (!setting || !*setting) return entries;
    const std::string text(setting);
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        std::string item =
            text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        start = comma == std::string::npos ? text.size() + 1 : comma + 1;
        // Trim surrounding spaces so a human-written list stays forgiving.
        while (!item.empty() && std::isspace(static_cast<unsigned char>(item.front())))
            item.erase(item.begin());
        while (!item.empty() && std::isspace(static_cast<unsigned char>(item.back())))
            item.pop_back();
        if (item.empty()) continue;
        RegWatchEntry entry;
        const size_t colon = item.find(':');
        if (colon != std::string::npos) {
            const std::string cls = item.substr(0, colon);
            if (cls == "Cx" || cls == "cx")
                entry.reg_class = RegClass::Cx;
            else if (cls == "Sh" || cls == "sh")
                entry.reg_class = RegClass::Sh;
            else if (cls == "Uc" || cls == "uc")
                entry.reg_class = RegClass::Uc;
            else
                continue;   // unknown class: skip this entry, keep the rest of the list
            item = item.substr(colon + 1);
            if (item.empty()) continue;
        }
        if (item == "*") {
            // Whole-class watch; only meaningful with an explicit class prefix (a bare "*" would
            // not say which file to watch, so it is skipped like any other unparsable entry).
            if (colon == std::string::npos) continue;
            entry.all_offsets = true;
            if (std::find(entries.begin(), entries.end(), entry) == entries.end())
                entries.push_back(entry);
            continue;
        }
        char* end = nullptr;
        errno = 0;
        const unsigned long parsed = std::strtoul(item.c_str(), &end, 0);
        if (errno || !end || *end || parsed > 0xFFFFFFFFul) continue;
        entry.offset = static_cast<uint32_t>(parsed);
        if (std::find(entries.begin(), entries.end(), entry) == entries.end())
            entries.push_back(entry);
    }
    return entries;
}

}   // namespace prosper::gpu
