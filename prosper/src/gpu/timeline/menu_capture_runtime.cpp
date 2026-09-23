#include "gpu/timeline/menu_capture_runtime.hpp"

#include "build_revision.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_dependency_graph.hpp"
#include "gpu/state/render_state.hpp"
#include "gpu/timeline/menu_capture_policy.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
namespace {
using Clock = std::chrono::steady_clock;
constexpr uint64_t kPerCandidateResourceLimit = 512ull << 20;

uint64_t now_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now().time_since_epoch()).count());
}

bool parse_positive_address(const char* value, uint64_t& out) {
    out = 0;
    if (!value || !*value || *value == '-' || *value == '+') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    if (errno || end == value || *end || !parsed) return false;
    out = parsed;
    return true;
}

bool parse_extent(const char* value, uint32_t& width, uint32_t& height) {
    width = height = 0;
    if (!value) return false;
    const char* separator = std::strchr(value, 'x');
    if (!separator) return false;
    const char* end = value + std::strlen(value);
    const auto w = std::from_chars(value, separator, width);
    const auto h = std::from_chars(separator + 1, end, height);
    return w.ec == std::errc{} && w.ptr == separator &&
           h.ec == std::errc{} && h.ptr == end && width && height;
}

const char* refusal_name(MenuCaptureRefusal refusal) {
    switch (refusal) {
        case MenuCaptureRefusal::None: return "none";
        case MenuCaptureRefusal::WaitExpired: return "menu-wait-expired";
        case MenuCaptureRefusal::CandidateExpired: return "candidate-wait-expired";
        case MenuCaptureRefusal::CandidateLimit: return "candidate-limit";
        case MenuCaptureRefusal::ByteBudget: return "byte-budget";
        case MenuCaptureRefusal::MissingPresentation: return "missing-publication";
        case MenuCaptureRefusal::AmbiguousDraw: return "ambiguous-draw";
        case MenuCaptureRefusal::SameSubmitDependency: return "same-submit-dependency";
        case MenuCaptureRefusal::UnsupportedAttachment: return "unsupported-attachment";
        case MenuCaptureRefusal::CaptureFailed: return "capture-failed";
        case MenuCaptureRefusal::WriteFailed: return "write-failed";
    }
    return "unknown";
}

bool prior_writer_to_selected(const GpuState& state, std::vector<DrawItem> draws,
                              const DrawItem& selected, std::string& error) {
    GpuReplayFrame replay;
    replay.items = std::move(draws);
    const auto planned = plan_submit_operations(state);
    std::unordered_set<uint64_t> realized_draws;
    for (const DrawItem& draw : replay.items) realized_draws.insert(draw.draw_index);
    uint32_t selected_operation = UINT32_MAX;
    for (const SubmitOperation& operation : planned) {
        if (operation.kind == SubmitOperationKind::Draw &&
            operation.index == selected.draw_index &&
            operation.command_order == selected.command_order) {
            selected_operation = static_cast<uint32_t>(replay.operations.size());
            replay.operations.push_back({operation.kind, operation.index,
                                         operation.command_order, true});
            break;
        }
        // An earlier dispatch or DMA can change a later draw's input at an address that no
        // draw-only capsule can reconstruct. Its absence from the materialized graphics list is
        // not a proof of non-interference.
        if (operation.kind != SubmitOperationKind::Draw) {
            error = "dispatch or DMA precedes the selected draw in the same submit";
            return true;
        }
        replay.operations.push_back({operation.kind, operation.index, operation.command_order,
                                     realized_draws.contains(operation.index)});
    }
    if (selected_operation == UINT32_MAX) {
        error = "selected draw is absent from the ordered submit";
        return true;
    }
    GpuDependencyGraph graph;
    if (!build_gpu_dependency_graph(replay, graph, error)) return true;
    if (std::any_of(graph.edges.begin(), graph.edges.end(), [selected_operation](const auto& edge) {
            return edge.consumer_operation == selected_operation;
        })) {
        error = "selected draw consumes an earlier same-submit producer";
        return true;
    }
    // Attachment loads are not descriptor edges in the dependency graph. Reject a prior write to
    // the selected target even if the PS does not sample it explicitly.
    for (const GpuCapturedOperation& operation : replay.operations) {
        if (operation.command_order == selected.command_order &&
            operation.source_index == selected.draw_index) break;
        if (!operation.realized) continue;
        const auto it = std::find_if(replay.items.begin(), replay.items.end(),
            [&](const DrawItem& draw) { return draw.draw_index == operation.source_index; });
        if (it == replay.items.end()) {
            error = "prior realized draw is missing";
            return true;
        }
        for (size_t slot = 0; slot < it->color_targets.size(); ++slot) {
            uint64_t base = it->color_targets[slot].base;
            if (!base && slot == 0) base = it->color0_base;
            const uint32_t mask = it->ps.color_targets[slot].write_mask ||
                    (slot == 0 ? it->ps.color_write_mask : 0);
            if (base == selected.color0_base && mask) {
                error = "earlier draw writes the selected color attachment";
                return true;
            }
        }
    }
    return false;
}

class MenuDrawCapture {
public:
    MenuDrawCapture() : policy_({}, now_ms()) {
        const char* destination = std::getenv("PROSPER_MENU_DRAW_CAPTURE");
        if (!destination || !*destination) return;
        path_ = destination;
        uint32_t target_width = 0, target_height = 0;
        std::error_code fs_error;
        if (!parse_menu_frame_gate_spec(std::getenv("PROSPER_MENU_FRAME_GATE"), gate_) ||
            !parse_positive_address(std::getenv("PROSPER_MENU_DRAW_FRAGMENT_PROGRAM"), ps_) ||
            !parse_positive_address(std::getenv("PROSPER_MENU_DRAW_TARGET_ADDRESS"), target_) ||
            !parse_extent(std::getenv("PROSPER_MENU_DRAW_TARGET_DIM"),
                          target_width, target_height) ||
            std::filesystem::exists(path_, fs_error) || fs_error) {
            std::fprintf(stderr, "[menu-capture] refused: invalid gate, draw identity, or occupied path\n");
            return;
        }
        target_width_ = target_width;
        target_height_ = target_height;
        active_ = true;
        std::fprintf(stderr, "[menu-capture] armed path=%s ps=0x%llx target=0x%llx %ux%u\n",
                     path_.c_str(), static_cast<unsigned long long>(ps_),
                     static_cast<unsigned long long>(target_), target_width_, target_height_);
        watchdog_ = std::jthread([this](std::stop_token stop) {
            while (!stop.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::lock_guard lock(mx_);
                const MenuSourceCensusPhase census_before = census_.phase();
                census_.tick(now_ms());
                report_census_limit(census_before);
                const MenuCapturePhase before = policy_.phase();
                policy_.tick(now_ms());
                report_refusal(before);
            }
        });
    }

    ~MenuDrawCapture() {
        if (!active_) return;
        watchdog_.request_stop();
        std::lock_guard lock(mx_);
        if (census_.phase() == MenuSourceCensusPhase::WaitingForNewSource)
            std::fprintf(stderr,
                         "[menu-capture] source census incomplete: process ended after %u "
                         "menu-positive publications without an exact source callback\n",
                         census_.attempts());
        std::fprintf(stderr,
                     "[menu-capture] post-menu census ps=%llu target=%llu extent=%llu exact=%llu "
                     "attempts=%u phase=%u source-census=%u/%u\n",
                     static_cast<unsigned long long>(ps_hits_),
                     static_cast<unsigned long long>(target_hits_),
                     static_cast<unsigned long long>(extent_hits_),
                     static_cast<unsigned long long>(exact_hits_),
                     policy_.attempted_candidates(), static_cast<unsigned>(policy_.phase()),
                     census_.attempts(), static_cast<unsigned>(census_.phase()));
    }

    void on_submit(const GpuState& state, uint64_t submit_no) {
        if (!active_) return;
        std::lock_guard lock(mx_);
        const MenuCapturePhase before = policy_.phase();
        policy_.tick(now_ms());
        report_refusal(before);
        if (policy_.phase() != MenuCapturePhase::Armed) return;
        std::vector<size_t> matches;
        for (size_t i = 0; i < state.draws.size(); ++i) {
            const RenderState rs = extract_render_state(state.state_at_draw(i));
            if (rs.ps_addr != ps_) continue;
            ++ps_hits_;
            const bool target_match = rs.color0_base == target_;
            const bool extent_match = rs.color0_width == target_width_ &&
                                      rs.color0_height == target_height_;
            target_hits_ += target_match;
            extent_hits_ += extent_match;
            exact_hits_ += target_match && extent_match;
            if (ps_hits_ <= 8)
                std::fprintf(stderr,
                             "[menu-capture] post-menu PS submit=%llu draw=%zu target=0x%llx "
                             "%ux%u mask=%x target-match=%d extent-match=%d\n",
                             static_cast<unsigned long long>(submit_no), i,
                             static_cast<unsigned long long>(rs.color0_base),
                             rs.color0_width, rs.color0_height,
                             rs.cb_target_mask & rs.cb_shader_mask & 0xfu,
                             target_match, extent_match);
            if (target_match && extent_match) matches.push_back(i);
        }
        if (matches.empty()) return;
        if (matches.size() != 1) {
            decline(MenuCaptureRefusal::AmbiguousDraw, "multiple matching semantic draws");
            return;
        }
        std::vector<DrawItem> draws = realize_gpustate_draws(state);
        const auto it = std::find_if(draws.begin(), draws.end(), [&](const DrawItem& draw) {
            return draw.draw_index == matches.front();
        });
        if (it == draws.end() || it->fs_guest_addr != ps_ ||
            it->color0_base != target_ || it->color0_width != target_width_ ||
            it->color0_height != target_height_) {
            decline(MenuCaptureRefusal::AmbiguousDraw, "matching semantic draw did not realize exactly");
            return;
        }
        const DrawItem& selected = *it;
        if (selected.ps.cb_resolve || selected.ps.blend_enable ||
            selected.ps.depth_test_enable || selected.ps.depth_write_enable ||
            selected.ps.stencil_enable ||
            !(selected.ps.color_write_mask || selected.ps.color_targets[0].write_mask)) {
            decline(MenuCaptureRefusal::UnsupportedAttachment,
                    "selected draw has attachment state requiring additional closure");
            return;
        }
        std::string error;
        if (prior_writer_to_selected(state, draws, selected, error)) {
            decline(MenuCaptureRefusal::SameSubmitDependency, error);
            return;
        }
        const uint64_t remaining = policy_.remaining_bytes();
        const uint64_t resource_limit = std::min(kPerCandidateResourceLimit, remaining);
        uint64_t planned_bytes = 0;
        if (!resource_limit ||
            !preflight_gpu_capture_draw_resources(selected, resource_limit, planned_bytes, error)) {
            decline(MenuCaptureRefusal::ByteBudget, error);
            return;
        }
        GpuCaptureMetadata metadata;
        metadata.width = gate_.width;
        metadata.height = gate_.height;
        metadata.submit_index = submit_no;
        metadata.revision = embedded_build_revision();
        if (const char* title = std::getenv("PROSPER_CAPTURE_TITLE")) metadata.title_id = title;
        annotate_gpu_capture_save_roots(metadata);
        annotate_gpu_capture_scanout(metadata);
        GpuCaptureFile capture;
        if (!capture_gpustate_selected_draw(state, selected, metadata,
                                            resource_limit, capture, error)) {
            decline(MenuCaptureRefusal::CaptureFailed, error);
            return;
        }
        uint64_t captured_bytes = 0;
        for (const auto& blob : capture.blobs) captured_bytes += blob.bytes.size();
        for (const auto& seed : capture.rtt_seeds) captured_bytes += seed.rgba.size();
        if (!policy_.begin_candidate(submit_no, captured_bytes, now_ms())) {
            report_refusal(MenuCapturePhase::Armed);
            return;
        }
        candidate_ = std::move(capture);
        std::fprintf(stderr,
                     "[menu-capture] candidate submit=%llu draw=%zu resources=%llu seeds+blobs=%llu "
                     "attempt=%u\n",
                     static_cast<unsigned long long>(submit_no), matches.front(),
                     static_cast<unsigned long long>(planned_bytes),
                     static_cast<unsigned long long>(captured_bytes),
                     policy_.attempted_candidates());
    }

    void on_publication(const GpuState& state,
                        std::span<const MenuRealizedDrawIdentity> realized_draws,
                        uint64_t execution_submit, uint64_t source_submit,
                        bool published, PresentFrameOrigin origin,
                        std::span<const uint8_t> rgba, uint32_t width, uint32_t height) {
        if (!active_) return;
        std::lock_guard lock(mx_);
        const MenuCapturePhase before = policy_.phase();
        const uint64_t publication_ms = now_ms();
        const MenuSourceCensusPhase census_before = census_.phase();
        census_.tick(publication_ms);
        report_census_limit(census_before);
        policy_.tick(publication_ms);
        if (!published) {
            policy_.missing_presentation(execution_submit, publication_ms);
            if (before == MenuCapturePhase::Pending && policy_.phase() != before)
                candidate_.reset();
            report_refusal(before);
            return;
        }
        const MenuFrameGateResult frame = classify_menu_frame(rgba, width, height, gate_);
        const bool composited = origin == PresentFrameOrigin::Composited;
        const bool pending = policy_.phase() == MenuCapturePhase::Pending;
        const bool exact_candidate = menu_capture_exact_candidate(
            execution_submit, source_submit, policy_.pending_submit(), pending);
        const bool menu = composited &&
            (frame.verdict == MenuFrameGateVerdict::MenuWithLogo ||
             frame.verdict == MenuFrameGateVerdict::MenuWithoutLogo);
        const MenuSourceCensusPhase observed_before = census_.phase();
        if (census_.observe(execution_submit, source_submit, menu, publication_ms))
            report_menu_source_writers(state, realized_draws, execution_submit, source_submit);
        else if (menu && observed_before == MenuSourceCensusPhase::WaitingForMenu &&
                 census_.phase() == MenuSourceCensusPhase::WaitingForNewSource)
            std::fprintf(stderr,
                         "[menu-capture] menu source-submit=%llu retained/unknown in execution=%llu; "
                         "retrying exact source census (max 8 positives/30 s)\n",
                         static_cast<unsigned long long>(source_submit),
                         static_cast<unsigned long long>(execution_submit));
        report_census_limit(observed_before);
        if (policy_.phase() == MenuCapturePhase::Refused) {
            candidate_.reset();
            report_refusal(before);
            return;
        }
        if (policy_.phase() == MenuCapturePhase::Pending &&
            policy_.pending_submit() == execution_submit &&
            source_submit != execution_submit) {
            policy_.missing_presentation(execution_submit, publication_ms);
            if (before == MenuCapturePhase::Pending && policy_.phase() != before)
                candidate_.reset();
            report_refusal(before);
            return;
        }
        if (exact_candidate && menu) {
            std::string error;
            if (!candidate_ || !write_gpu_capture(path_, *candidate_, error)) {
                decline(MenuCaptureRefusal::WriteFailed, error);
                candidate_.reset();
                return;
            }
        }
        // The policy may arm on a known earlier menu frame, but while a candidate is pending its
        // acceptance must use exactly the same source predicate that guarded the capsule write.
        const bool source_known_for_policy = pending ? exact_candidate : source_submit != 0;
        policy_.published(source_submit, source_known_for_policy,
                          frame.verdict, composited, publication_ms);
        if (policy_.phase() == MenuCapturePhase::Armed && before == MenuCapturePhase::WaitingForMenu)
            std::fprintf(stderr, "[menu-capture] menu-positive source-submit=%llu; future candidates armed\n",
                         static_cast<unsigned long long>(source_submit));
        if (exact_candidate) {
            candidate_.reset();
            if (policy_.phase() == MenuCapturePhase::Accepted)
                std::fprintf(stderr,
                             "[menu-capture] captured exact source-submit=%llu menu-bright=%llu "
                             "logo-bright=%llu -> %s (replay parity unverified)\n",
                             static_cast<unsigned long long>(source_submit),
                             static_cast<unsigned long long>(frame.menu_bright),
                             static_cast<unsigned long long>(frame.logo_bright),
                             path_.c_str());
        }
        report_refusal(before);
    }

private:
    void report_menu_source_writers(
        const GpuState& state, std::span<const MenuRealizedDrawIdentity> realized_draws,
        uint64_t execution_submit, uint64_t source_submit) {
        if (execution_submit != source_submit) {
            std::fprintf(stderr,
                         "[menu-capture] menu source-submit=%llu is retained in execution=%llu; "
                         "exact source draw state unavailable\n",
                         static_cast<unsigned long long>(source_submit),
                         static_cast<unsigned long long>(execution_submit));
            return;
        }
        struct Writer {
            size_t index = 0;
            uint64_t ps = 0, base = 0;
            uint32_t width = 0, height = 0, format = 0, mask = 0;
        };
        std::array<Writer, 8> scene{}, screen{};
        size_t scene_count = 0, screen_count = 0;
        for (size_t i = 0; i < state.draws.size(); ++i) {
            const RenderState rs = extract_render_state(state.state_at_draw(i));
            const Writer writer{i, rs.ps_addr, rs.color0_base, rs.color0_width,
                                rs.color0_height, rs.color0_format,
                                rs.cb_target_mask & rs.cb_shader_mask & 0xfu};
            if (writer.width == target_width_ && writer.height == target_height_)
                scene[scene_count++ % scene.size()] = writer;
            else if (writer.width == gate_.width && writer.height == gate_.height)
                screen[screen_count++ % screen.size()] = writer;
        }
        std::fprintf(stderr,
                     "[menu-capture] menu source-submit=%llu semantic=%zu realized=%zu "
                     "scene-size=%zu screen-size=%zu (last eight semantic target candidates "
                     "of each; realization is not pixel-write proof)\n",
                     static_cast<unsigned long long>(source_submit), state.draws.size(),
                     realized_draws.size(), scene_count, screen_count);
        auto print = [&](const char* kind, const auto& writers, size_t count) {
            const size_t first = count > writers.size() ? count - writers.size() : 0;
            for (size_t number = first; number < count; ++number) {
                const Writer& writer = writers[number % writers.size()];
                const auto found = std::find_if(realized_draws.begin(), realized_draws.end(),
                    [&](const MenuRealizedDrawIdentity& draw) {
                        return draw.draw_index == writer.index;
                    });
                const char* agreement = found == realized_draws.end() ? "missing" :
                    found->fs_guest_addr == writer.ps &&
                    found->color0_base == writer.base &&
                    found->color0_width == writer.width &&
                    found->color0_height == writer.height ? "exact" : "diverged";
                std::fprintf(stderr,
                             "[menu-capture] menu-candidate=%s draw=%zu ps=0x%llx target=0x%llx "
                             "%ux%u fmt=%u raw-mask=%x raw-active=%d realized=%s "
                             "realized-mask=%x\n",
                             kind, writer.index,
                             static_cast<unsigned long long>(writer.ps),
                             static_cast<unsigned long long>(writer.base),
                             writer.width, writer.height, writer.format, writer.mask,
                             writer.base && writer.format && writer.mask, agreement,
                             found == realized_draws.end() ? 0 : found->write_mask);
            }
        };
        print("scene", scene, scene_count);
        print("screen", screen, screen_count);
    }

    void report_census_limit(MenuSourceCensusPhase before) {
        if (before == census_.phase()) return;
        if (census_.phase() == MenuSourceCensusPhase::AttemptLimit)
            std::fprintf(stderr,
                         "[menu-capture] source census refused: eight menu-positive publications "
                         "without exact same-submit pixels\n");
        else if (census_.phase() == MenuSourceCensusPhase::WaitExpired)
            std::fprintf(stderr,
                         "[menu-capture] source census refused: no exact same-submit menu "
                         "publication within 30 s\n");
    }

    void report_refusal(MenuCapturePhase before) {
        if (before != MenuCapturePhase::Refused && policy_.phase() == MenuCapturePhase::Refused)
            std::fprintf(stderr, "[menu-capture] refused: %s\n",
                         refusal_name(policy_.refusal()));
    }
    void decline(MenuCaptureRefusal reason, const std::string& detail) {
        policy_.abort(reason);
        std::fprintf(stderr, "[menu-capture] refused: %s: %s\n",
                     refusal_name(reason), detail.c_str());
    }

    std::mutex mx_;
    bool active_ = false;
    std::string path_;
    MenuFrameGateSpec gate_;
    uint64_t ps_ = 0, target_ = 0;
    uint32_t target_width_ = 0, target_height_ = 0;
    MenuCapturePolicy policy_;
    MenuSourceCensusPolicy census_;
    std::optional<GpuCaptureFile> candidate_;
    uint64_t ps_hits_ = 0, target_hits_ = 0, extent_hits_ = 0, exact_hits_ = 0;
    std::jthread watchdog_;
};

MenuDrawCapture& menu_draw_capture() {
    static MenuDrawCapture runtime;
    return runtime;
}
} // namespace

void menu_draw_capture_on_submit(const GpuState& state, uint64_t submit_no) {
    menu_draw_capture().on_submit(state, submit_no);
}

void menu_draw_capture_on_publication(const GpuState& state,
                                      std::span<const MenuRealizedDrawIdentity> realized_draws,
                                      uint64_t execution_submit, uint64_t source_submit,
                                      bool published,
                                      PresentFrameOrigin origin,
                                      std::span<const uint8_t> rgba,
                                      uint32_t width, uint32_t height) {
    menu_draw_capture().on_publication(state, realized_draws, execution_submit, source_submit,
                                       published, origin, rgba, width, height);
}
} // namespace prosper::gpu
