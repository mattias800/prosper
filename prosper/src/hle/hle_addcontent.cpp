// hle_addcontent.cpp — parse and validate the dump tool's local add-content inventory.
#include "hle_addcontent.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>

namespace prosper {
namespace {

constexpr size_t kMaxManifestBytes = 32 * 1024; // dlc_emu producer limit
constexpr size_t kMaxParamBytes = 2 * 1024 * 1024;
constexpr size_t kMaxEntries = 1024;             // dlc_emu producer limit

std::mutex g_inventory_mutex;
AddcontentInventorySnapshot g_inventory;

enum class BoundedFileState {
    Missing,
    Ready,
    Invalid,
};

struct BoundedFile {
    BoundedFileState state = BoundedFileState::Invalid;
    std::string data;
};

struct PendingRecord {
    std::string section;
    std::string content_id;
    std::string download_status;
    std::string mount_point;
    std::string entitlement_key;
    std::string np_service_label;
    std::set<std::string> seen_keys;
};

bool ascii_digit(char ch) { return ch >= '0' && ch <= '9'; }
bool ascii_upper(char ch) { return ch >= 'A' && ch <= 'Z'; }
bool ascii_alpha(char ch) { return ascii_upper(ch) || (ch >= 'a' && ch <= 'z'); }

bool valid_title_id(std::string_view value) {
    return value.size() == 9 && value.substr(0, 4) == "PPSA" &&
           std::all_of(value.begin() + 4, value.end(), ascii_digit);
}

bool valid_entitlement_label(std::string_view value) {
    if (value.empty() || value.size() > 16) return false;
    return std::all_of(value.begin(), value.end(), [](char ch) {
        return ascii_alpha(ch) || ascii_digit(ch);
    });
}

bool is_missing_error(const std::error_code& ec) {
    return ec == std::make_error_code(std::errc::no_such_file_or_directory);
}

bool path_is_strict_descendant(const std::filesystem::path& root,
                               const std::filesystem::path& child) {
    auto root_it = root.begin();
    auto child_it = child.begin();
    for (; root_it != root.end(); ++root_it, ++child_it) {
        if (child_it == child.end() || *root_it != *child_it) return false;
    }
    return child_it != child.end();
}

// The identity-bearing files must be real regular files below /app0. Walk every relative component
// without following symlinks, then canonicalize as a second containment check. Only a genuine ENOENT
// is Missing; dangling symlinks, wrong types, lookup errors, escapes, and read errors are Invalid.
BoundedFile read_bounded_file(const std::filesystem::path& root,
                              const std::filesystem::path& relative, size_t limit) {
    BoundedFile result;
    std::filesystem::path current = root;
    std::error_code ec;
    for (auto it = relative.begin(); it != relative.end(); ++it) {
        current /= *it;
        ec.clear();
        const std::filesystem::file_status status = std::filesystem::symlink_status(current, ec);
        if (status.type() == std::filesystem::file_type::not_found || is_missing_error(ec)) {
            result.state = BoundedFileState::Missing;
            return result;
        }
        if (ec || std::filesystem::is_symlink(status)) return result;
        const bool last = std::next(it) == relative.end();
        if ((last && !std::filesystem::is_regular_file(status)) ||
            (!last && !std::filesystem::is_directory(status))) return result;
    }

    ec.clear();
    const std::filesystem::path canonical_root = std::filesystem::canonical(root, ec);
    if (ec) return result;
    const std::filesystem::path canonical_file = std::filesystem::canonical(current, ec);
    if (ec || !path_is_strict_descendant(canonical_root, canonical_file)) return result;

    const uintmax_t size = std::filesystem::file_size(current, ec);
    if (ec || size > limit || size > std::numeric_limits<size_t>::max()) return result;
    std::ifstream input(current, std::ios::binary);
    if (!input) return result;
    result.data.assign(static_cast<size_t>(size), '\0');
    if (!result.data.empty() &&
        !input.read(result.data.data(), static_cast<std::streamsize>(result.data.size()))) {
        result.data.clear();
        return result;
    }
    result.state = BoundedFileState::Ready;
    return result;
}

// Small bounded JSON reader for one authorization-bearing metadata field. It validates the document
// and accepts exactly one top-level string named titleId; a textual occurrence in a nested
// object or string must never authorize a foreign dlc_emu.ini record. Escapes in the title value are
// rejected so its identity has one byte representation. Nesting is capped to avoid hostile recursion.
class ParamJsonReader {
public:
    explicit ParamJsonReader(std::string_view input) : input_(input) {}

    std::optional<std::string> top_level_title_id() {
        skip_space();
        if (!take('{')) return std::nullopt;
        skip_space();
        if (take('}')) return finish() ? title_id_ : std::nullopt;
        for (;;) {
            std::string key;
            if (!parse_string(&key)) return std::nullopt;
            skip_space();
            if (!take(':')) return std::nullopt;
            skip_space();
            if (key == "titleId") {
                if (saw_title_id_) return std::nullopt;
                saw_title_id_ = true;
                bool value_escaped = false;
                std::string value;
                if (!parse_string(&value, &value_escaped) || value_escaped) return std::nullopt;
                title_id_ = std::move(value);
            } else if (!skip_value(1)) {
                return std::nullopt;
            }
            skip_space();
            if (take('}')) break;
            if (!take(',')) return std::nullopt;
            skip_space();
        }
        return finish() && saw_title_id_ ? title_id_ : std::nullopt;
    }

private:
    static bool hex_digit(char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
               (ch >= 'A' && ch <= 'F');
    }
    static uint32_t hex_value(char ch) {
        if (ch >= '0' && ch <= '9') return static_cast<uint32_t>(ch - '0');
        if (ch >= 'a' && ch <= 'f') return static_cast<uint32_t>(ch - 'a' + 10);
        return static_cast<uint32_t>(ch - 'A' + 10);
    }
    void skip_space() {
        while (pos_ < input_.size() &&
               (input_[pos_] == ' ' || input_[pos_] == '\t' || input_[pos_] == '\r' ||
                input_[pos_] == '\n')) ++pos_;
    }
    bool take(char expected) {
        if (pos_ == input_.size() || input_[pos_] != expected) return false;
        ++pos_;
        return true;
    }
    bool finish() {
        skip_space();
        return pos_ == input_.size();
    }
    bool parse_string(std::string* output = nullptr, bool* escaped = nullptr) {
        if (!take('"')) return false;
        if (output) output->clear();
        if (escaped) *escaped = false;
        while (pos_ < input_.size()) {
            const unsigned char byte = static_cast<unsigned char>(input_[pos_++]);
            if (byte == '"') return true;
            if (byte < 0x20) return false;
            if (byte != '\\') {
                if (output) output->push_back(static_cast<char>(byte));
                continue;
            }
            if (escaped) *escaped = true;
            if (pos_ == input_.size()) return false;
            const char escape = input_[pos_++];
            char decoded = 0;
            switch (escape) {
            case '"': decoded = '"'; break;
            case '\\': decoded = '\\'; break;
            case '/': decoded = '/'; break;
            case 'b': decoded = '\b'; break;
            case 'f': decoded = '\f'; break;
            case 'n': decoded = '\n'; break;
            case 'r': decoded = '\r'; break;
            case 't': decoded = '\t'; break;
            case 'u': {
                if (input_.size() - pos_ < 4) return false;
                uint32_t codepoint = 0;
                for (size_t i = 0; i < 4; ++i) {
                    const char digit = input_[pos_++];
                    if (!hex_digit(digit)) return false;
                    codepoint = (codepoint << 4) | hex_value(digit);
                }
                decoded = codepoint <= 0x7f ? static_cast<char>(codepoint) : '\x80';
                break;
            }
            default: return false;
            }
            if (output) output->push_back(decoded);
        }
        return false;
    }
    bool skip_number() {
        const size_t begin = pos_;
        take('-');
        if (take('0')) {
            // A following decimal digit is rejected by the caller's delimiter check.
        } else {
            const size_t digits = pos_;
            while (pos_ < input_.size() && ascii_digit(input_[pos_])) ++pos_;
            if (digits == pos_) return false;
        }
        if (take('.')) {
            const size_t digits = pos_;
            while (pos_ < input_.size() && ascii_digit(input_[pos_])) ++pos_;
            if (digits == pos_) return false;
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) ++pos_;
            const size_t digits = pos_;
            while (pos_ < input_.size() && ascii_digit(input_[pos_])) ++pos_;
            if (digits == pos_) return false;
        }
        return pos_ != begin;
    }
    bool skip_literal(std::string_view literal) {
        if (input_.substr(pos_, literal.size()) != literal) return false;
        pos_ += literal.size();
        return true;
    }
    bool skip_value(unsigned depth) {
        if (depth > 64) return false;
        skip_space();
        if (pos_ == input_.size()) return false;
        if (input_[pos_] == '"') return parse_string();
        if (input_[pos_] == '{') {
            ++pos_;
            skip_space();
            if (take('}')) return true;
            for (;;) {
                if (!parse_string()) return false;
                skip_space();
                if (!take(':') || !skip_value(depth + 1)) return false;
                skip_space();
                if (take('}')) return true;
                if (!take(',')) return false;
                skip_space();
            }
        }
        if (input_[pos_] == '[') {
            ++pos_;
            skip_space();
            if (take(']')) return true;
            for (;;) {
                if (!skip_value(depth + 1)) return false;
                skip_space();
                if (take(']')) return true;
                if (!take(',')) return false;
                skip_space();
            }
        }
        if (input_[pos_] == 't') return skip_literal("true");
        if (input_[pos_] == 'f') return skip_literal("false");
        if (input_[pos_] == 'n') return skip_literal("null");
        return skip_number();
    }

    std::string_view input_;
    size_t pos_ = 0;
    bool saw_title_id_ = false;
    std::optional<std::string> title_id_;
};

bool parse_hex_key(std::string_view text, std::array<uint8_t, 16>& out) {
    if (text.size() != out.size() * 2) return false;
    auto nibble = [](char ch) -> int {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
        if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < out.size(); ++i) {
        const int high = nibble(text[i * 2]);
        const int low = nibble(text[i * 2 + 1]);
        if (high < 0 || low < 0) return false;
        out[i] = static_cast<uint8_t>((high << 4) | low);
    }
    return true;
}

void make_default_key(size_t accepted_index, std::array<uint8_t, 16>& out) {
    const uint64_t value = 1024u + static_cast<uint64_t>(accepted_index);
    out.fill(0);
    for (size_t i = 0; i < sizeof(value); ++i)
        out[i] = static_cast<uint8_t>(value >> (i * 8u));
}

bool parse_np_service_label(std::string_view text, int64_t& out) {
    if (text.empty()) return true; // dlc_emu's documented default: wildcard
    if (text == "-1") {
        out = -1;
        return true;
    }
    uint64_t value = 0;
    if (text.size() > 10) return false;
    for (char ch : text) {
        if (!ascii_digit(ch)) return false;
        value = value * 10 + static_cast<uint64_t>(ch - '0');
        if (value > UINT32_MAX) return false;
    }
    out = static_cast<int64_t>(value);
    return true;
}

bool canonical_child_directory(const std::filesystem::path& root,
                               std::string_view guest_mount_point) {
    if (guest_mount_point.size() >= 16 || guest_mount_point.substr(0, 6) != "/app0/")
        return false;
    const std::string_view relative = guest_mount_point.substr(6);
    if (relative.empty() || relative.front() == '/' || relative.back() == '/' ||
        relative.find('\\') != std::string_view::npos) return false;
    for (size_t pos = 0; pos <= relative.size();) {
        const size_t slash = relative.find('/', pos);
        const size_t end = slash == std::string_view::npos ? relative.size() : slash;
        const std::string_view component = relative.substr(pos, end - pos);
        if (component.empty() || component == "." || component == "..") return false;
        if (slash == std::string_view::npos) break;
        pos = slash + 1;
    }

    std::error_code ec;
    const std::filesystem::path canonical_root = std::filesystem::canonical(root, ec);
    if (ec) return false;
    const std::filesystem::path candidate =
        std::filesystem::canonical(root / std::filesystem::path(relative), ec);
    if (ec || !std::filesystem::is_directory(candidate, ec) || ec) return false;

    auto root_it = canonical_root.begin();
    auto child_it = candidate.begin();
    for (; root_it != canonical_root.end(); ++root_it, ++child_it) {
        if (child_it == candidate.end() || *root_it != *child_it) return false;
    }
    return child_it != candidate.end(); // the add-content root must be below, not equal to, /app0
}

struct ParseResult {
    AddcontentInventorySnapshot inventory;
    std::string error;
};

ParseResult parse_inventory(const std::filesystem::path& root, std::string_view manifest,
                            std::string_view title_id) {
    ParseResult result;
    result.inventory.state = AddcontentInventoryState::Invalid;
    std::vector<PendingRecord> pending;
    PendingRecord* current = nullptr;

    size_t line_number = 0;
    for (size_t pos = 0; pos <= manifest.size();) {
        size_t end = manifest.find('\n', pos);
        if (end == std::string_view::npos) end = manifest.size();
        std::string_view line = manifest.substr(pos, end - pos);
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (!line.empty() && line.back() == ' ') {
            result.error = "trailing whitespace at line " + std::to_string(line_number);
            return result;
        }
        if (!line.empty() && line.front() == '[') {
            if (line != "[PSAC]" && line != "[PSAL]") {
                result.error = "unknown section at line " + std::to_string(line_number);
                return result;
            }
            if (pending.size() == kMaxEntries) {
                result.error = "manifest exceeds the installed-entry limit";
                return result;
            }
            pending.push_back({});
            current = &pending.back();
            current->section = std::string(line.substr(1, 4));
        } else if (!line.empty() && line.front() != '#' && line.front() != ';') {
            if (!current) {
                result.error = "property before section at line " + std::to_string(line_number);
                return result;
            }
            const size_t equal = line.find('=');
            if (equal == std::string_view::npos || equal == 0 || equal + 1 == line.size()) {
                result.error = "malformed property at line " + std::to_string(line_number);
                return result;
            }
            const std::string key(line.substr(0, equal));
            const std::string value(line.substr(equal + 1));
            if (!current->seen_keys.insert(key).second) {
                result.error = "duplicate property at line " + std::to_string(line_number);
                return result;
            }
            if (key == "content_id") current->content_id = value;
            else if (key == "download_status") current->download_status = value;
            else if (key == "mount_point") current->mount_point = value;
            else if (key == "entitlement_key") current->entitlement_key = value;
            else if (key == "np_service_label") current->np_service_label = value;
            else {
                result.error = "unknown property at line " + std::to_string(line_number);
                return result;
            }
        }
        if (end == manifest.size()) break;
        pos = end + 1;
    }
    if (pending.empty()) {
        result.error = "manifest contains no records";
        return result;
    }

    std::set<std::string> labels;
    std::set<std::string> content_ids;
    std::set<std::string> mount_points;
    for (size_t index = 0; index < pending.size(); ++index) {
        const PendingRecord& input = pending[index];
        const std::string record = "record " + std::to_string(index + 1);
        // XX0000-PPSA00000_00-ENTITLEMENTLABEL is the declared add-content identity. The final
        // 16 characters are the Sony entitlement label. API service scoping is a separate optional
        // np_service_label property; dlc_emu defines its absent/-1 value as wildcard.
        if (input.content_id.size() != 36 || !ascii_upper(input.content_id[0]) ||
            !ascii_upper(input.content_id[1]) ||
            !std::all_of(input.content_id.begin() + 2, input.content_id.begin() + 6, ascii_digit) ||
            input.content_id[6] != '-' ||
            input.content_id.substr(16, 4) != "_00-" ||
            input.content_id.substr(7, 9) != title_id) {
            result.error = record + " has an invalid or foreign content_id";
            return result;
        }
        const std::string label = input.content_id.substr(20);
        if (!valid_entitlement_label(label)) {
            result.error = record + " has an invalid entitlement label";
            return result;
        }
        if (input.download_status != "INSTALLED") {
            result.error = record + " has an unsupported download_status";
            return result;
        }
        int64_t service_label = -1;
        if (!parse_np_service_label(input.np_service_label, service_label)) {
            result.error = record + " has an invalid np_service_label";
            return result;
        }
        if (!labels.insert(label).second) {
            result.error = record + " duplicates an entitlement label";
            return result;
        }
        if (!content_ids.insert(input.content_id).second) {
            result.error = record + " duplicates a content_id";
            return result;
        }

        InstalledAddcontent output;
        output.service_label = service_label;
        output.entitlement_label = label;
        output.package_type = input.section == "PSAC" ? 2u : 3u;
        if (!input.mount_point.empty()) {
            if (!mount_points.insert(input.mount_point).second) {
                result.error = record + " duplicates a mount point";
                return result;
            }
            if (!canonical_child_directory(root, input.mount_point)) {
                result.error = record + " has an absent or out-of-root mount point";
                return result;
            }
            output.guest_mount_point = input.mount_point;
        }
        output.mountable = output.package_type == 2u && !output.guest_mount_point.empty();
        if (!input.entitlement_key.empty()) {
            if (!parse_hex_key(input.entitlement_key, output.entitlement_key)) {
                result.error = record + " has a malformed entitlement key";
                return result;
            }
        } else make_default_key(result.inventory.entries.size(), output.entitlement_key);
        result.inventory.entries.push_back(std::move(output));
    }
    result.inventory.state = AddcontentInventoryState::Ready;
    return result;
}

} // namespace

void addcontent_configure_for_app0(const std::string& app0_root) {
    AddcontentInventorySnapshot next;
    if (app0_root.empty()) {
        std::lock_guard<std::mutex> lock(g_inventory_mutex);
        g_inventory = std::move(next);
        return;
    }
    const std::filesystem::path root(app0_root);
    const BoundedFile manifest = read_bounded_file(root, "dlc_emu.ini", kMaxManifestBytes);
    if (manifest.state == BoundedFileState::Missing) {
        next.state = AddcontentInventoryState::None;
    } else if (manifest.state != BoundedFileState::Ready) {
        next.state = AddcontentInventoryState::Invalid;
        std::fprintf(stderr, "[addcontent] invalid install manifest: unreadable or oversized\n");
    } else {
        const BoundedFile param = read_bounded_file(root, std::filesystem::path("sce_sys") /
                                                   "param.json", kMaxParamBytes);
        const std::optional<std::string> title_id = param.state == BoundedFileState::Ready
            ? ParamJsonReader(param.data).top_level_title_id() : std::nullopt;
        if (!title_id || !valid_title_id(*title_id)) {
            next.state = AddcontentInventoryState::Invalid;
            std::fprintf(stderr,
                         "[addcontent] invalid install manifest: app metadata has no valid titleId\n");
        } else {
            ParseResult parsed = parse_inventory(root, manifest.data, *title_id);
            next = std::move(parsed.inventory);
            if (next.state == AddcontentInventoryState::Invalid)
                std::fprintf(stderr, "[addcontent] invalid install manifest: %s\n",
                             parsed.error.c_str());
        }
    }
    std::lock_guard<std::mutex> lock(g_inventory_mutex);
    g_inventory = std::move(next);
}

AddcontentInventorySnapshot addcontent_inventory_snapshot() {
    std::lock_guard<std::mutex> lock(g_inventory_mutex);
    return g_inventory;
}

AddcontentMountResult addcontent_mount(uint32_t service_label, std::string_view entitlement_label,
                                       uint64_t output_address, AddcontentMountWriter writer) {
    std::lock_guard<std::mutex> lock(g_inventory_mutex);
    if (g_inventory.state != AddcontentInventoryState::Ready)
        return AddcontentMountResult::NotFound;
    size_t index = g_inventory.entries.size();
    for (size_t i = 0; i < g_inventory.entries.size(); ++i) {
        const InstalledAddcontent& entry = g_inventory.entries[i];
        if ((entry.service_label == -1 ||
             static_cast<uint32_t>(entry.service_label) == service_label) &&
            entry.entitlement_label == entitlement_label) {
            index = i;
            break;
        }
    }
    if (index == g_inventory.entries.size()) return AddcontentMountResult::NotFound;
    InstalledAddcontent& entry = g_inventory.entries[index];
    if (!entry.mountable || entry.guest_mount_point.empty() || entry.download_status != 4)
        return AddcontentMountResult::NotFound;
    if (entry.mounted) return AddcontentMountResult::Busy;
    for (const InstalledAddcontent& other : g_inventory.entries) {
        if (other.mounted && other.guest_mount_point == entry.guest_mount_point)
            return AddcontentMountResult::Busy;
    }
    std::array<char, 16> mount_point{};
    std::memcpy(mount_point.data(), entry.guest_mount_point.data(),
                entry.guest_mount_point.size());
    if (!writer || !writer(output_address, mount_point.data(), mount_point.size()))
        return AddcontentMountResult::OutputError;
    entry.mounted = true;
    return AddcontentMountResult::Mounted;
}

} // namespace prosper
