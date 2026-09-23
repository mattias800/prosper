#include "gpu/timeline/menu_capture_runtime.hpp"

#include "build_revision.hpp"
#include "diagnostics/exit_reports.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_dependency_graph.hpp"
#include "gpu/resources/shader_resources.hpp"
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
        case MenuCaptureRefusal::MenuTargetUnresolved: return "menu-target-unresolved";
        case MenuCaptureRefusal::CaptureFailed: return "capture-failed";
        case MenuCaptureRefusal::WriteFailed: return "write-failed";
    }
    return "unknown";
}

const char* target_reject_name(MenuTargetReject rejection) {
    switch (rejection) {
        case MenuTargetReject::None: return "none";
        case MenuTargetReject::MissingScene: return "missing-scene";
        case MenuTargetReject::UnrealizedFinalScene: return "unrealized-final-scene";
        case MenuTargetReject::WrongSceneProgram: return "wrong-scene-program";
        case MenuTargetReject::MissingScreen: return "missing-screen";
        case MenuTargetReject::UnrealizedLaterWriter: return "unrealized-later-writer";
        case MenuTargetReject::OtherWriterAfterScene: return "other-writer-after-scene";
        case MenuTargetReject::TooManyDraws: return "too-many-draws";
    }
    return "unknown";
}

// One refused candidate only. This records a bounded metadata slice of the actual ordered submit;
// it does not copy any shader, descriptor, or guest bytes. Declared VA overlap is useful for
// choosing the next capture, but cannot establish execution, physical-alias, or indirect-pointer
// dependencies and never changes the draw-only capsule's fail-closed admission decision.
void report_prior_effect_census(const GpuState& state,
                                std::span<const SubmitOperation> planned,
                                size_t selected_position,
                                const DrawItem& selected, uint64_t submit_no) {
    struct InputRange {
        const char* stage = nullptr;
        uint32_t binding = 0;
        uint64_t addr = 0, bytes = 0;
    };
    constexpr size_t kMaxInputRanges = 64, kMaxInputVisits = 512, kMaxEffects = 64;
    std::array<InputRange, kMaxInputRanges> inputs{};
    size_t input_count = 0, input_visits = 0, unknown_inputs = 0, input_truncated = 0;
    uint64_t declared_bytes_sum = 0;
    bool declared_bytes_overflow = false;
    auto add_input = [&](const char* stage, uint32_t binding, uint64_t addr, uint64_t bytes) {
        if (!addr || !bytes || bytes > UINT64_MAX - addr) { ++unknown_inputs; return; }
        if (bytes > UINT64_MAX - declared_bytes_sum) declared_bytes_overflow = true;
        else declared_bytes_sum += bytes;
        if (input_count < inputs.size()) inputs[input_count++] = {stage, binding, addr, bytes};
        else ++input_truncated;
    };
    auto add_table = [&](const char* stage, const ShaderResourceTable* table) {
        if (!table) { ++unknown_inputs; return; }
        for (const ShaderResource& resource : table->resources) {
            if (++input_visits > kMaxInputVisits) { ++input_truncated; break; }
            if (resource.cls == ResourceClass::Sampler) continue;
            if (resource.table_index_count) {
                if (resource.table_entries.size() != resource.table_index_count)
                    ++unknown_inputs;
                for (const ShaderBufferTableEntry& entry : resource.table_entries) {
                    if (++input_visits > kMaxInputVisits) { ++input_truncated; break; }
                    add_input(stage, resource.binding, entry.gpu_addr,
                              std::max<uint64_t>(entry.size, entry.host_data_size));
                }
                continue;
            }
            uint64_t bytes = std::max<uint64_t>(resource.size, resource.host_data_size);
            if (resource.scalar_buffer_dword_count)
                bytes = std::max(bytes, shader_resource_buffer_binding_bytes(resource));
            add_input(stage, resource.binding, resource.gpu_addr, bytes);
        }
    };
    add_table("vs", selected.vrt.get());
    add_table("ps", selected.prt.get());
    bool indirect = false;
    uint64_t indirect_addr = 0;
    if (selected.draw_index < state.draws.size()) {
        const auto& draw = state.draws[selected.draw_index];
        indirect = draw.indirect;
        indirect_addr = draw.indirect_args_addr;
        if (indirect) add_input("indirect-args", UINT32_MAX, indirect_addr, 20);
    } else {
        ++unknown_inputs;
    }
    std::fprintf(stderr,
                 "[menu-dependency] submit=%llu draw=%llu order=%llu vs=0x%llx ps=0x%llx "
                 "target=0x%llx %ux%u indirect=%u args=0x%llx "
                 "inputs-stored=%zu inputs-unknown=%zu inputs-truncated=%zu "
                 "declared-bytes-sum=%llu sum-overflow=%u "
                 "(pre-execution resource realization; not dependency closure)\n",
                 static_cast<unsigned long long>(submit_no),
                 static_cast<unsigned long long>(selected.draw_index),
                 static_cast<unsigned long long>(selected.command_order),
                 static_cast<unsigned long long>(selected.vs_guest_addr),
                 static_cast<unsigned long long>(selected.fs_guest_addr),
                 static_cast<unsigned long long>(selected.color0_base),
                 selected.color0_width, selected.color0_height, indirect ? 1u : 0u,
                 static_cast<unsigned long long>(indirect_addr), input_count,
                 unknown_inputs, input_truncated,
                 static_cast<unsigned long long>(declared_bytes_sum),
                 declared_bytes_overflow ? 1u : 0u);
    for (size_t i = 0; i < std::min(input_count, size_t{16}); ++i) {
        const auto& input = inputs[i];
        std::fprintf(stderr,
                     "[menu-dependency] input stage=%s binding=%u addr=0x%llx bytes=%llu\n",
                     input.stage, input.binding,
                     static_cast<unsigned long long>(input.addr),
                     static_cast<unsigned long long>(input.bytes));
    }
    if (input_count > 16)
        std::fprintf(stderr, "[menu-dependency] inputs omitted=%zu\n", input_count - 16);
    std::array<SubmitOperation, kMaxEffects> tail{};
    size_t prior_draws = 0, prior_dispatches = 0, prior_dmas = 0, prior_effects = 0;
    for (const SubmitOperation& operation : planned.first(selected_position)) {
        if (operation.kind == SubmitOperationKind::Draw) { ++prior_draws; continue; }
        if (operation.kind == SubmitOperationKind::Dispatch) ++prior_dispatches;
        else ++prior_dmas;
        tail[prior_effects++ % tail.size()] = operation;
    }
    std::fprintf(stderr,
                 "[menu-dependency] before-selected draws=%zu dispatches=%zu dmas=%zu "
                 "effects-shown=%zu effects-omitted=%zu\n",
                 prior_draws, prior_dispatches, prior_dmas,
                 std::min(prior_effects, tail.size()),
                 prior_effects > tail.size() ? prior_effects - tail.size() : 0);
    const size_t first = prior_effects > tail.size() ? prior_effects - tail.size() : 0;
    for (size_t n = first; n < prior_effects; ++n) {
        const SubmitOperation& operation = tail[n % tail.size()];
        if (operation.kind == SubmitOperationKind::Dispatch) {
            if (operation.index >= state.dispatches.size()) {
                std::fprintf(stderr, "[menu-dependency] dispatch index=%zu missing\n",
                             operation.index);
                continue;
            }
            const auto& dispatch = state.dispatches[operation.index];
            const uint64_t code = compute_dispatch_code_addr(state, dispatch);
            std::fprintf(stderr,
                         "[menu-dependency] dispatch index=%zu order=%llu program=0x%llx "
                         "program-known=%u "
                         "raw-dim=%u,%u,%u modifier=0x%llx indirect=%u args=0x%llx "
                         "outputs=unknown\n",
                         operation.index,
                         static_cast<unsigned long long>(operation.command_order),
                         static_cast<unsigned long long>(code), code ? 1u : 0u,
                         dispatch.threads_x, dispatch.threads_y, dispatch.threads_z,
                         static_cast<unsigned long long>(dispatch.modifier),
                         dispatch.indirect ? 1u : 0u,
                         static_cast<unsigned long long>(dispatch.indirect_args_addr));
            continue;
        }
        if (operation.index >= state.dma_copies.size()) {
            std::fprintf(stderr, "[menu-dependency] dma index=%zu missing\n", operation.index);
            continue;
        }
        const auto& copy = state.dma_copies[operation.index];
        bool declared_overlap = false, declared_unknown = unknown_inputs || input_truncated;
        const bool guest_destination = (copy.sels & 0xffu) != 1u;
        if (!guest_destination) declared_unknown = true;
        if (guest_destination)
            for (const auto& input : inputs) {
                if (!input.addr) continue;
                const auto relation = menu_declared_range_relation(
                    copy.dst, copy.bytes, input.addr, input.bytes);
                declared_overlap |= relation == MenuDeclaredRangeRelation::Overlap;
                declared_unknown |= relation == MenuDeclaredRangeRelation::Unknown;
            }
        std::fprintf(stderr,
                     "[menu-dependency] dma index=%zu order=%llu src=0x%llx dst=0x%llx "
                     "bytes=%u sels=0x%x guest-dst=%u declared-overlap=%u "
                     "declared-unknown=%u scope=declared-VA-only\n",
                     operation.index,
                     static_cast<unsigned long long>(operation.command_order),
                     static_cast<unsigned long long>(copy.src),
                     static_cast<unsigned long long>(copy.dst),
                     copy.bytes, copy.sels, guest_destination ? 1u : 0u,
                     declared_overlap ? 1u : 0u, declared_unknown ? 1u : 0u);
    }
}

// Diagnostic only: use the existing conservative dependency graph to rank the producers that
// may feed the selected draw. This neither admits the draw-only capture nor proves a dynamically
// sampled value; failed or unmaterialized operations leave the graph incomplete.
void report_prior_effect_graph(const GpuState& state, GpuReplayFrame replay,
                               std::span<const SubmitOperation> planned,
                               size_t selected_position, const DrawItem& selected_draw,
                               uint64_t submit_no) {
    constexpr size_t kMaxOperations = 512, kMaxReported = 24;
    if (selected_position >= kMaxOperations || state.dispatches.size() > 128) {
        std::fprintf(stderr, "[menu-graph] refused: operation bound exceeded\n");
        return;
    }
    std::vector<OperationRealizationFailure> failures;
    replay.computes = realize_compute_dispatches(state, submit_no, &failures);
    std::unordered_set<uint64_t> draw_indices, compute_indices;
    for (const DrawItem& draw : replay.items) draw_indices.insert(draw.draw_index);
    for (const ComputeItem& compute : replay.computes)
        compute_indices.insert(compute.dispatch_index);
    replay.dma_copies.resize(state.dma_copies.size());
    for (size_t i = 0; i < state.dma_copies.size(); ++i) {
        const auto& source = state.dma_copies[i];
        replay.dma_copies[i] = {source.dst, source.src, source.bytes,
                                source.sels, source.command_order, source.packet_addr};
    }
    // The caller has already accumulated a prefix while checking for prior non-draw work.
    // Rebuild from operation zero so selected_position still names the selected draw.
    replay.operations.clear();
    for (const SubmitOperation& operation : planned.first(selected_position + 1)) {
        const bool realized = operation.kind == SubmitOperationKind::Draw
            ? draw_indices.contains(operation.index)
            : operation.kind == SubmitOperationKind::Dispatch
                ? compute_indices.contains(operation.index)
                : operation.index < replay.dma_copies.size();
        replay.operations.push_back({operation.kind, operation.index,
                                     operation.command_order, realized});
    }
    const auto& selected_operation = replay.operations[selected_position];
    if (selected_operation.kind != SubmitOperationKind::Draw ||
        selected_operation.source_index != selected_draw.draw_index ||
        selected_operation.command_order != selected_draw.command_order ||
        !selected_operation.realized) {
        std::fprintf(stderr, "[menu-graph] refused: selected operation identity mismatch\n");
        return;
    }
    GpuDependencyGraph graph;
    std::string graph_error;
    if (!build_gpu_dependency_graph(replay, graph, graph_error)) {
        std::fprintf(stderr, "[menu-graph] failed: %s\n", graph_error.c_str());
        return;
    }
    const uint32_t selected = static_cast<uint32_t>(selected_position);
    size_t incoming = 0, external = 0;
    for (const auto& edge : graph.edges) incoming += edge.consumer_operation == selected;
    for (const auto& leaf : graph.external_leaves)
        external += std::find(leaf.consumer_operations.begin(),
                              leaf.consumer_operations.end(), selected) !=
                    leaf.consumer_operations.end();
    std::fprintf(stderr,
                 "[menu-graph] submit=%llu selected-op=%u draw=%llu order=%llu "
                 "realized-compute=%zu "
                 "compute-failures=%zu incoming=%zu external=%zu "
                 "scope=declared-resource-graph-only\n",
                 static_cast<unsigned long long>(submit_no), selected,
                 static_cast<unsigned long long>(selected_draw.draw_index),
                 static_cast<unsigned long long>(selected_draw.command_order),
                 replay.computes.size(), failures.size(), incoming, external);
    size_t printed = 0;
    for (const auto& edge : graph.edges) {
        if (edge.consumer_operation != selected || printed++ >= kMaxReported) continue;
        const auto& producer = graph.nodes[edge.producer_operation];
        std::fprintf(stderr,
                     "[menu-graph] input stage=%s binding=%u addr=0x%llx "
                     "producer=%u kind=%u source-index=%llu order=%llu\n",
                     edge.access.stage.c_str(), edge.access.binding,
                     static_cast<unsigned long long>(edge.access.addr),
                     edge.producer_operation, static_cast<unsigned>(producer.kind),
                     static_cast<unsigned long long>(producer.source_index),
                     static_cast<unsigned long long>(producer.command_order));
    }
    if (incoming > kMaxReported)
        std::fprintf(stderr, "[menu-graph] inputs omitted=%zu\n", incoming - kMaxReported);
    printed = 0;
    for (const auto& leaf : graph.external_leaves) {
        if (std::find(leaf.consumer_operations.begin(), leaf.consumer_operations.end(),
                      selected) == leaf.consumer_operations.end() || printed++ >= kMaxReported)
            continue;
        std::fprintf(stderr,
                     "[menu-graph] external stage=%s binding=%u addr=0x%llx bytes=%llu\n",
                     leaf.access.stage.c_str(), leaf.access.binding,
                     static_cast<unsigned long long>(leaf.access.addr),
                     static_cast<unsigned long long>(leaf.access.size));
    }
    if (external > kMaxReported)
        std::fprintf(stderr, "[menu-graph] external omitted=%zu\n", external - kMaxReported);
}

bool prior_writer_to_selected(const GpuState& state, std::vector<DrawItem> draws,
                              const DrawItem& selected, uint64_t submit_no,
                              std::string& error) {
    GpuReplayFrame replay;
    replay.items = std::move(draws);
    const auto planned = plan_submit_operations(state);
    const size_t selected_position = menu_capture_selected_draw_position(
        std::span<const SubmitOperation>(planned), SubmitOperationKind::Draw,
        selected.draw_index, selected.command_order);
    if (selected_position == std::numeric_limits<size_t>::max()) {
        error = "selected draw is absent from the ordered submit";
        return true;
    }
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
            report_prior_effect_census(state, planned, selected_position, selected, submit_no);
            report_prior_effect_graph(state, std::move(replay), planned,
                                      selected_position, selected, submit_no);
            error = "dispatch or DMA precedes the selected draw in the same submit";
            return true;
        }
        replay.operations.push_back({operation.kind, operation.index, operation.command_order,
                                     realized_draws.contains(operation.index)});
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
        const char* target_text = std::getenv("PROSPER_MENU_DRAW_TARGET_ADDRESS");
        const bool auto_target = target_text && std::strcmp(target_text, "auto") == 0;
        std::error_code fs_error;
        if (!parse_menu_frame_gate_spec(std::getenv("PROSPER_MENU_FRAME_GATE"), gate_) ||
            !parse_positive_address(std::getenv("PROSPER_MENU_DRAW_FRAGMENT_PROGRAM"), ps_) ||
            (!auto_target && !parse_positive_address(target_text, target_)) ||
            !parse_extent(std::getenv("PROSPER_MENU_DRAW_TARGET_DIM"),
                          target_width, target_height) ||
            std::filesystem::exists(path_, fs_error) || fs_error) {
            std::fprintf(stderr, "[menu-capture] refused: invalid gate, draw identity, or occupied path\n");
            return;
        }
        target_width_ = target_width;
        target_height_ = target_height;
        configured_target_ = target_;
        active_ = true;
        std::fprintf(stderr, "[menu-capture] armed path=%s ps=0x%llx target=%s0x%llx %ux%u\n",
                     path_.c_str(), static_cast<unsigned long long>(ps_),
                     auto_target ? "same-run-auto/" : "expected/",
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
        // screenshot and prosper-app deliberately bypass C++ static destructors on exit. Register
        // the terminal census through their shared explicit exit path, and keep this diagnostic
        // alive until process termination so an ordinary atexit flush cannot read a dead object.
        prosper::diagnostics::register_exit_report([this] { report_exit(); });
    }

    void report_exit() {
        if (!active_) return;
        watchdog_.request_stop();
        // The GPU submit thread may be copying a bounded but large capsule while a frontend
        // exits. A diagnostic must not make exit wait for that copy to finish.
        std::unique_lock lock(mx_, std::try_to_lock);
        if (!lock.owns_lock()) {
            std::fprintf(stderr,
                         "[menu-capture] exit census unavailable: capture callback still active\n");
            return;
        }
        if (census_.phase() == MenuSourceCensusPhase::WaitingForNewSource)
            std::fprintf(stderr,
                         "[menu-capture] source census incomplete: process ended after %u "
                         "distinct known retained sources without an exact source callback\n",
                         census_.distinct_retained_sources());
        std::fprintf(stderr,
                     "[menu-capture] post-menu census ps=%llu target=%llu extent=%llu exact=%llu "
                     "attempts=%u phase=%u source-census-retained=%u source-census-phase=%u\n",
                     static_cast<unsigned long long>(ps_hits_),
                     static_cast<unsigned long long>(target_hits_),
                     static_cast<unsigned long long>(extent_hits_),
                     static_cast<unsigned long long>(exact_hits_),
                     policy_.attempted_candidates(), static_cast<unsigned>(policy_.phase()),
                     census_.distinct_retained_sources(), static_cast<unsigned>(census_.phase()));
    }

    void on_submit(const GpuState& state, uint64_t submit_no) {
        if (!active_) return;
        std::lock_guard lock(mx_);
        const MenuCapturePhase before = policy_.phase();
        policy_.tick(now_ms());
        report_refusal(before);
        if (policy_.phase() != MenuCapturePhase::Armed) return;
        // A guest address is run-local. The target is bound only by the exact menu-positive source
        // submit's admitted scene-to-screen semantic chain below, even when an explicit address was
        // supplied as an additional assertion.
        if (!target_bound_) return;
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
        if (prior_writer_to_selected(state, draws, selected, submit_no, error)) {
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
        const bool exact_source = source_submit != 0 && source_submit == execution_submit;
        bool has_selected_scene = false;
        if (menu && exact_source &&
            (census_.phase() == MenuSourceCensusPhase::WaitingForMenu ||
             census_.phase() == MenuSourceCensusPhase::WaitingForNewSource)) {
            for (size_t i = 0; i < state.draws.size(); ++i) {
                const RenderState rs = extract_render_state(state.state_at_draw(i));
                if (rs.ps_addr == ps_ && rs.color0_width == target_width_ &&
                    rs.color0_height == target_height_) {
                    has_selected_scene = true;
                    break;
                }
            }
        }
        if (census_.observe(execution_submit, source_submit, menu, publication_ms,
                            !exact_source || has_selected_scene)) {
            const MenuTargetChoice choice = report_menu_source_writers(
                state, realized_draws, execution_submit, source_submit);
            if (!menu_capture_target_matches_request(choice, configured_target_)) {
                std::fprintf(stderr,
                             "[menu-capture] target choice refused source-submit=%llu "
                             "reason=%s draw=%llu derived=0x%llx expected=0x%llx\n",
                             static_cast<unsigned long long>(source_submit),
                             target_reject_name(choice.rejection),
                             static_cast<unsigned long long>(choice.draw_index),
                             static_cast<unsigned long long>(choice.target_addr),
                             static_cast<unsigned long long>(configured_target_));
                decline(MenuCaptureRefusal::MenuTargetUnresolved,
                        "exact menu source has no unique admitted final scene-to-screen candidate "
                        "matching the configured target");
            } else {
                target_ = choice.target_addr;
                target_bound_ = true;
                std::fprintf(stderr,
                             "[menu-capture] same-run target=0x%llx draw=%llu "
                             "source-submit=%llu (candidate, not pixel-flow proof)\n",
                             static_cast<unsigned long long>(target_),
                             static_cast<unsigned long long>(choice.draw_index),
                             static_cast<unsigned long long>(source_submit));
            }
        }
        else if (menu && observed_before == MenuSourceCensusPhase::WaitingForMenu &&
                 census_.phase() == MenuSourceCensusPhase::WaitingForNewSource)
            std::fprintf(stderr,
                         "[menu-capture] menu source-submit=%llu retained/unknown in execution=%llu; "
                         "retrying exact source census (max 8 distinct known sources/30 s)\n",
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
    MenuTargetChoice report_menu_source_writers(
        const GpuState& state, std::span<const MenuRealizedDrawIdentity> realized_draws,
        uint64_t execution_submit, uint64_t source_submit) {
        if (execution_submit != source_submit) {
            std::fprintf(stderr,
                         "[menu-capture] menu source-submit=%llu is retained in execution=%llu; "
                         "exact source draw state unavailable\n",
                         static_cast<unsigned long long>(source_submit),
                         static_cast<unsigned long long>(execution_submit));
            return {};
        }
        if (state.draws.size() > 4096) {
            std::fprintf(stderr,
                         "[menu-capture] menu source-submit=%llu has %zu draws; "
                         "source census exceeds 4096-draw bound\n",
                         static_cast<unsigned long long>(source_submit), state.draws.size());
            return {0, 0, MenuTargetReject::TooManyDraws};
        }
        struct Writer {
            size_t index = 0;
            uint64_t ps = 0, base = 0;
            uint32_t width = 0, height = 0, format = 0, mask = 0;
        };
        std::array<Writer, 8> scene{}, screen{};
        std::vector<MenuWriterEvidence> evidence;
        evidence.reserve(state.draws.size());
        size_t scene_count = 0, screen_count = 0;
        for (size_t i = 0; i < state.draws.size(); ++i) {
            const RenderState rs = extract_render_state(state.state_at_draw(i));
            const Writer writer{i, rs.ps_addr, rs.color0_base, rs.color0_width,
                                rs.color0_height, rs.color0_format,
                                rs.cb_target_mask & rs.cb_shader_mask & 0xfu};
            const auto found = std::find_if(realized_draws.begin(), realized_draws.end(),
                [&](const MenuRealizedDrawIdentity& draw) { return draw.draw_index == i; });
            const bool realized_exact = found != realized_draws.end() &&
                found->fs_guest_addr == writer.ps && found->color0_base == writer.base &&
                found->color0_width == writer.width &&
                found->color0_height == writer.height;
            evidence.push_back({i, writer.ps, writer.base, writer.width, writer.height,
                                writer.format, writer.mask, realized_exact,
                                found == realized_draws.end() ? 0 : found->write_mask});
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
        return menu_capture_select_target(evidence, ps_, target_width_, target_height_,
                                          gate_.width, gate_.height);
    }

    void report_census_limit(MenuSourceCensusPhase before) {
        if (before == census_.phase()) return;
        if (census_.phase() == MenuSourceCensusPhase::AttemptLimit)
            std::fprintf(stderr,
                         "[menu-capture] source census refused: eight distinct known retained "
                         "sources without exact same-submit pixels\n");
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
    uint64_t ps_ = 0, target_ = 0, configured_target_ = 0;
    bool target_bound_ = false;
    uint32_t target_width_ = 0, target_height_ = 0;
    MenuCapturePolicy policy_;
    MenuSourceCensusPolicy census_;
    std::optional<GpuCaptureFile> candidate_;
    uint64_t ps_hits_ = 0, target_hits_ = 0, extent_hits_ = 0, exact_hits_ = 0;
    std::jthread watchdog_;
};

MenuDrawCapture& menu_draw_capture() {
    static MenuDrawCapture* runtime = new MenuDrawCapture;
    return *runtime;
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
