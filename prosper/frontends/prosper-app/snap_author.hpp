// Human-authored render snapshots ("snaps") — the pure half.
//
// A snap is a judgement, not a measurement: the person playing the game presses F6 to say "this
// frame is CORRECT" or F7 to say "this frame is WRONG", and the app records the presented pixels
// plus enough anchoring for a later automated run to find the same moment again.
//
// Why both verdicts are worth storing. A correct snap is an ordinary regression guard. An
// INCORRECT snap is the half that does not exist today, and it is why titles improve silently: a
// broken title simply has no guard, so nothing notices when it starts rendering. Recording the
// known-bad frame turns "broken" into tracked state — if it stops matching, something changed, and
// a human should look at whether it got better. That is information, never a failure.
//
// The anchor is the PAD FLIP ORDINAL — flips since the guest's first pad poll — because that is the
// axis `PROSPER_PAD_SCRIPT` routes already replay against, and it is boot-speed-invariant: unlike
// wall-clock it lands the same input on the same game state across builds (#302). An authoring run
// records the route with `PROSPER_PAD_RECORD` and the snaps against the same counter, so the two
// files index each other.
//
// This header is deliberately free of SDL, Vulkan and filesystem access so it can be unit-tested;
// main.cpp owns the hotkeys, the pixels and the writing.
#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace prosper::frontend {

enum class SnapVerdict {
    correct,    // F6 — "the game looks right here"
    incorrect,  // F7 — "the game does NOT look right here"
};

// How the check should LOOK for this frame again.
//
// `anchor` searches a tight window around the recorded flip and is right for a frame whose position
// is stable. `scan` sweeps a wide span forward and is for a frame that sits after something of
// variable length -- a loading screen, an FMV, anything whose duration depends on the machine. It is
// the person authoring who knows which is which, at the moment they press the key, so the choice is
// theirs: F6/F7 anchor, SHIFT+F6/F7 scan.
enum class SnapMode {
    anchor,
    scan,
};

inline const char* snap_mode_token(SnapMode mode) {
    return mode == SnapMode::scan ? "scan" : "anchor";
}

inline const char* snap_verdict_token(SnapVerdict verdict) {
    return verdict == SnapVerdict::correct ? "correct" : "incorrect";
}

// A snap taken before the guest's first pad poll cannot be anchored: the route's f<N> origin does
// not exist yet, so there is no ordinal to replay to. The app still writes such a snap (throwing
// away a human's judgement silently would be worse) and marks it with this sentinel so the import
// step can reject it loudly rather than quietly recording an anchor of zero — which would replay to
// the first frame of the boot and compare a menu against a logo.
inline constexpr int64_t kSnapFlipUnanchored = -1;

// Zero-padded so lexicographic order matches capture order in any file listing, and carrying both
// the verdict and the anchor so a snap separated from its manifest still describes itself.
inline std::string snap_file_name(uint32_t index, SnapVerdict verdict, int64_t flip) {
    char buffer[96];
    if (flip < 0)
        std::snprintf(buffer, sizeof buffer, "snap_%04u_%s_unanchored.bmp",
                      index, snap_verdict_token(verdict));
    else
        std::snprintf(buffer, sizeof buffer, "snap_%04u_%s_f%lld.bmp",
                      index, snap_verdict_token(verdict),
                      static_cast<long long>(flip));
    return buffer;
}

// Minimal JSON string escaping. The only values that reach this are names this header generated and
// a title id from the dump, so the input is tame — but a manifest that silently produces invalid
// JSON on an unexpected character would fail at import time, far from the cause.
inline std::string snap_json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char c : value) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char esc[8];
                    std::snprintf(esc, sizeof esc, "\\u%04x", static_cast<unsigned char>(c));
                    out += esc;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// One JSON Lines record. Appended per snap rather than rewritten as an array, so a run that is
// killed — which is the normal way an authoring session ends — still leaves every snap taken before
// that point readable, instead of a truncated array that parses as nothing.
inline std::string snap_record_line(uint32_t index,
                                    SnapVerdict verdict,
                                    SnapMode mode,
                                    int64_t flip,
                                    uint64_t guest_present,
                                    uint32_t width,
                                    uint32_t height,
                                    const std::string& file,
                                    const std::string& title_id) {
    std::string out = "{\"index\":";
    out += std::to_string(index);
    out += ",\"verdict\":\"";
    out += snap_verdict_token(verdict);
    out += "\",\"mode\":\"";
    out += snap_mode_token(mode);
    out += "\",\"pad_flip\":";
    out += std::to_string(flip);
    out += ",\"guest_present\":";
    out += std::to_string(guest_present);
    out += ",\"width\":";
    out += std::to_string(width);
    out += ",\"height\":";
    out += std::to_string(height);
    out += ",\"file\":\"";
    out += snap_json_escape(file);
    out += "\",\"title_id\":\"";
    out += snap_json_escape(title_id);
    out += "\"}";
    return out;
}

// The console line for one authored snap. Says what was judged, where it landed, and — when the
// snap could not be anchored — why that snap is unusable, at the moment the person can still do
// something about it.
inline std::string snap_author_line(uint32_t index,
                                    SnapVerdict verdict,
                                    SnapMode mode,
                                    int64_t flip,
                                    const std::string& file) {
    std::string out = "[snap] ";
    out += snap_verdict_token(verdict);
    if (mode == SnapMode::scan) out += " (scan)";
    out += " #";
    out += std::to_string(index);
    if (flip < 0) {
        out += " NOT ANCHORED (the guest has not polled the pad yet, so this snap has no route "
               "position and will be rejected at import) -> ";
    } else {
        out += " at pad flip ";
        out += std::to_string(flip);
        out += " -> ";
    }
    out += file;
    return out;
}

// ---- Replaying an authored run --------------------------------------------------------------
//
// The CHECK side captures the same moments automatically: PROSPER_SNAP_AT_FLIPS lists the pad flip
// ordinals an authored run recorded, and the app grabs the presented frame as each one arrives.
//
// This deliberately runs through prosper-app, the same frontend the snaps were authored in, rather
// than through a separate screenshot tool. The whole comparison rests on "the same pixels a person
// looked at", and two frontends can composite differently -- which would show up as a diff nobody
// can act on.
//
// Malformed input disables the trigger rather than firing at an unintended flip, matching the
// established behaviour of PROSPER_GRAB_BUNDLE_AT_FRAME and friends: a typo should cost you a
// capture, never a wrong measurement.
//
// BURSTS (#5152). A single frame cannot see a picture that alternates good / broken / good: it lands
// on one parity and calls the build good or broken by chance. So an entry may also be a range of
// consecutive flips, and ranges mix freely with single entries:
//   `1200-1207`   flips 1200 through 1207 inclusive (a reversed range is malformed);
//   `1200+8`      eight flips starting at 1200 (`+0` is malformed);
//   `400-407,1000,2000+4`.
// A range is capped at kSnapFlipRangeMaxFrames frames -- every anchor is a full-resolution BMP,
// ~24 MiB at 4K -- and the whole list at kSnapFlipListMaxEntries DISTINCT anchors, counted after
// duplicates collapse, so overlapping ranges and the generated lists of tools/snapshot/snaps.py
// (7 to 33 anchors per authored snap) are not refused for size they do not have. A list over either
// cap is void like any other malformed list, rather than truncated.
inline constexpr size_t kSnapFlipRangeMaxFrames = 256;
inline constexpr size_t kSnapFlipListMaxEntries = 4096;

namespace detail {
// A non-empty run of decimal digits that fits int64_t, or nullopt-like false.
inline bool parse_snap_flip_number(const std::string& digits, int64_t& out) {
    if (digits.empty()) return false;
    for (const char d : digits)
        if (d < '0' || d > '9') return false;
    try {
        out = std::stoll(digits);
    } catch (...) {
        return false;
    }
    return true;
}
}  // namespace detail

inline std::vector<int64_t> parse_snap_flip_list(const char* value) {
    std::vector<int64_t> out;
    if (!value || !*value) return out;
    const std::string text(value);
    std::string token;
    for (size_t i = 0; i <= text.size(); ++i) {
        const char c = i < text.size() ? text[i] : ',';
        if (c == ',' || c == ' ') {
            if (!token.empty()) {
                // Any non-digit anywhere, or a value that does not fit, voids the WHOLE list. A
                // partially-parsed list is the dangerous outcome: it would capture some moments and
                // silently skip others, and the run would look like a clean pass with holes in it.
                const size_t dash = token.find('-');
                const size_t plus = token.find('+');
                int64_t first = 0, last = 0;
                if (dash != std::string::npos && plus == std::string::npos) {
                    if (!detail::parse_snap_flip_number(token.substr(0, dash), first) ||
                        !detail::parse_snap_flip_number(token.substr(dash + 1), last) ||
                        last < first)
                        return {};
                } else if (plus != std::string::npos && dash == std::string::npos) {
                    int64_t count = 0;
                    if (!detail::parse_snap_flip_number(token.substr(0, plus), first) ||
                        !detail::parse_snap_flip_number(token.substr(plus + 1), count) ||
                        count < 1 || count > static_cast<int64_t>(kSnapFlipRangeMaxFrames) ||
                        first > INT64_MAX - (count - 1))
                        return {};
                    last = first + (count - 1);
                } else if (dash == std::string::npos && plus == std::string::npos) {
                    if (!detail::parse_snap_flip_number(token, first)) return {};
                    last = first;
                } else {
                    return {};
                }
                // Checked BEFORE expanding, so `0-99999999999` costs a comparison, not memory; the
                // second bound keeps a list of many overlapping ranges from growing without limit
                // before the duplicates collapse.
                if (static_cast<uint64_t>(last - first) >= kSnapFlipRangeMaxFrames ||
                    out.size() + static_cast<size_t>(last - first) + 1 > 16 * kSnapFlipListMaxEntries)
                    return {};
                // Counted, not `f <= last`: a range ending at INT64_MAX must not wrap.
                for (int64_t k = 0; k <= last - first; ++k) out.push_back(first + k);
                token.clear();
            }
        } else {
            token += c;
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (out.size() > kSnapFlipListMaxEntries) return {};
    return out;
}

// Name for a frame captured by the check run. Carries BOTH the flip that was asked for and the flip
// actually reached, because they can differ: the app grabs the first presented frame at or after the
// target, and a slow frame can overshoot. A comparison that silently used the target alone would
// hide exactly the drift a reader needs to see when a diff looks inexplicable.
inline std::string snap_actual_file_name(int64_t target_flip, int64_t actual_flip) {
    char buffer[96];
    std::snprintf(buffer, sizeof buffer, "actual_f%lld_at%lld.bmp",
                  static_cast<long long>(target_flip), static_cast<long long>(actual_flip));
    return buffer;
}

// The GPU handoff's publication ID and the CPU frame sequence have independent namespaces.
// Record the path beside the numbers so a later analysis never joins them by number alone.
enum class SnapActualPresentPath { gpu_scanout, gpu_cpu_fallback, cpu };
inline const char* snap_actual_present_path_token(SnapActualPresentPath path) {
    switch (path) {
    case SnapActualPresentPath::gpu_scanout: return "gpu-scanout";
    case SnapActualPresentPath::gpu_cpu_fallback: return "gpu-cpu-fallback";
    case SnapActualPresentPath::cpu: return "cpu";
    }
    return "unknown";
}
struct SnapActualIdentity {
    uint64_t source_seq = 0;       // guest flip handed to presentation
    uint64_t publication_id = 0;   // GPU handoff ID or CPU frame sequence, per path
    SnapActualPresentPath path = SnapActualPresentPath::cpu;
};

inline std::string snap_actual_record_line(int64_t target_flip,
                                           int64_t actual_flip,
                                           uint32_t width,
                                           uint32_t height,
                                           const std::string& file,
                                           SnapActualIdentity identity) {
    std::string out = "{\"target_flip\":";
    out += std::to_string(target_flip);
    out += ",\"actual_flip\":";
    out += std::to_string(actual_flip);
    out += ",\"width\":";
    out += std::to_string(width);
    out += ",\"height\":";
    out += std::to_string(height);
    out += ",\"source_seq\":";
    out += std::to_string(identity.source_seq);
    out += ",\"publication_id\":";
    out += std::to_string(identity.publication_id);
    out += ",\"present_path\":\"";
    out += snap_actual_present_path_token(identity.path);
    out += "\"";
    out += ",\"file\":\"";
    out += snap_json_escape(file);
    out += "\"}";
    return out;
}

}  // namespace prosper::frontend
