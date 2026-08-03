#include "gpu/gpu_timeline.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

using namespace prosper::gpu;

const char* writer_kind_name(GpuTimelineWriterKind kind) {
    switch (kind) {
        case GpuTimelineWriterKind::Graphics: return "graphics";
        case GpuTimelineWriterKind::Compute: return "compute";
        case GpuTimelineWriterKind::DmaData: return "dma-data";
        case GpuTimelineWriterKind::WriteData: return "write-data";
        default: return "unknown";
    }
}

const char* producer_provenance_name(GpuTimelineProducerProvenance provenance) {
    switch (provenance) {
        case GpuTimelineProducerProvenance::ProducerHistory: return "producer-history";
        case GpuTimelineProducerProvenance::ExactRttSeed: return "exact-rtt-seed";
        case GpuTimelineProducerProvenance::PhaseHistoryBounded:
            return "phase-history-bounded/unknown";
        default: return "unknown";
    }
}

struct DepthKey {
    GpuTimelineDepthSurface ds;
    bool operator<(const DepthKey& other) const {
        const auto& a = ds; const auto& b = other.ds;
        return std::tie(a.depth_read_base, a.depth_write_base,
                        a.stencil_read_base, a.stencil_write_base, a.htile_data_base,
                        a.db_depth_view, a.db_render_override, a.db_render_override2,
                        a.db_depth_size_xy, a.db_dfsm_control, a.db_depth_info,
                        a.db_z_info, a.db_stencil_info, a.db_depth_size, a.db_depth_slice,
                        a.db_htile_surface, a.db_rmi_l2_cache_control,
                        a.target_width, a.target_height) <
               std::tie(b.depth_read_base, b.depth_write_base,
                        b.stencil_read_base, b.stencil_write_base, b.htile_data_base,
                        b.db_depth_view, b.db_render_override, b.db_render_override2,
                        b.db_depth_size_xy, b.db_dfsm_control, b.db_depth_info,
                        b.db_z_info, b.db_stencil_info, b.db_depth_size, b.db_depth_slice,
                        b.db_htile_surface, b.db_rmi_l2_cache_control,
                        b.target_width, b.target_height);
    }
};

struct DepthBaseKey {
    uint64_t dr = 0, dw = 0, sr = 0, sw = 0;
    uint32_t width = 0, height = 0;
    bool operator<(const DepthBaseKey& other) const {
        return std::tie(dr, dw, sr, sw, width, height) <
               std::tie(other.dr, other.dw, other.sr, other.sw,
                        other.width, other.height);
    }
};

int print_depth_summary(const GpuTimelineFile& timeline, uint32_t filter_width,
                        uint32_t filter_height) {
    struct Stats {
        uint64_t first_submit = UINT64_MAX, last_submit = 0;
        uint64_t submits = 0, draws = 0, tests = 0, writes = 0, clears = 0;
        uint32_t compare_mask = 0;
        std::set<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>> backing_versions;
        GpuTimelineDepthSurface latest_writer;
    };
    std::map<DepthKey, Stats> totals;
    std::map<DepthBaseKey, DepthKey> previous;
    std::map<DepthKey, std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>> previous_backing;
    uint64_t transitions = 0, backing_transitions = 0;
    for (const auto& submit : timeline.submits) {
        for (const auto& ds : submit.depth_surfaces) {
            if (filter_width && (ds.target_width != filter_width || ds.target_height != filter_height))
                continue;
            DepthKey key{ds};
            Stats& stats = totals[key];
            stats.first_submit = std::min(stats.first_submit, submit.submit_no);
            stats.last_submit = std::max(stats.last_submit, submit.submit_no);
            ++stats.submits;
            stats.draws += ds.draw_count;
            stats.tests += ds.depth_test_count;
            stats.writes += ds.depth_write_count;
            stats.clears += ds.clear_count;
            stats.compare_mask |= ds.compare_mask;
            const auto backing = std::tuple{ds.backing_hash_mask, ds.depth_backing_hash,
                                            ds.stencil_backing_hash, ds.htile_backing_hash};
            if (ds.backing_hash_mask) {
                stats.backing_versions.insert(backing);
                auto prior_backing = previous_backing.find(key);
                if (prior_backing != previous_backing.end() && prior_backing->second != backing)
                    ++backing_transitions;
                previous_backing[key] = backing;
            }
            if (ds.backing_writer_sequence > stats.latest_writer.backing_writer_sequence)
                stats.latest_writer = ds;
            DepthBaseKey base{ds.depth_read_base, ds.depth_write_base,
                              ds.stencil_read_base, ds.stencil_write_base,
                              ds.target_width, ds.target_height};
            auto prior = previous.find(base);
            if (prior != previous.end() && (prior->second < key || key < prior->second)) ++transitions;
            previous[base] = key;
        }
    }
    for (const auto& [key, stats] : totals) {
        const auto& ds = key.ds;
        std::printf("depth z=%016llx/%016llx s=%016llx/%016llx htile=%016llx "
                    "target=%ux%u submits=%llu first=%llu last=%llu draws=%llu "
                    "tests=%llu writes=%llu clears=%llu compare-mask=%08x "
                    "view=%08x override=%08x/%08x size=%08x/%08x/%08x "
                    "info=%08x/%08x/%08x hsurface=%08x dfsm=%08x rmi=%08x\n",
                    static_cast<unsigned long long>(ds.depth_read_base),
                    static_cast<unsigned long long>(ds.depth_write_base),
                    static_cast<unsigned long long>(ds.stencil_read_base),
                    static_cast<unsigned long long>(ds.stencil_write_base),
                    static_cast<unsigned long long>(ds.htile_data_base),
                    ds.target_width, ds.target_height,
                    static_cast<unsigned long long>(stats.submits),
                    static_cast<unsigned long long>(stats.first_submit),
                    static_cast<unsigned long long>(stats.last_submit),
                    static_cast<unsigned long long>(stats.draws),
                    static_cast<unsigned long long>(stats.tests),
                    static_cast<unsigned long long>(stats.writes),
                    static_cast<unsigned long long>(stats.clears), stats.compare_mask,
                    ds.db_depth_view, ds.db_render_override, ds.db_render_override2,
                    ds.db_depth_size_xy, ds.db_depth_size, ds.db_depth_slice,
                    ds.db_depth_info, ds.db_z_info, ds.db_stencil_info,
                    ds.db_htile_surface, ds.db_dfsm_control, ds.db_rmi_l2_cache_control);
        if (!stats.backing_versions.empty()) {
            const auto& [mask, depth, stencil, htile] = *stats.backing_versions.begin();
            std::printf("  backing versions=%zu mask=%x first=%016llx/%016llx/%016llx\n",
                        stats.backing_versions.size(), mask,
                        static_cast<unsigned long long>(depth),
                        static_cast<unsigned long long>(stencil),
                        static_cast<unsigned long long>(htile));
        }
        if (stats.latest_writer.backing_writer_sequence) {
            const auto& writer = stats.latest_writer;
            std::printf("  writer kind=%s sequence=%llu range=%016llx/+%llx order=%llu identity=%016llx\n",
                        writer_kind_name(static_cast<GpuTimelineWriterKind>(writer.backing_writer_kind)),
                        static_cast<unsigned long long>(writer.backing_writer_sequence),
                        static_cast<unsigned long long>(writer.backing_writer_addr),
                        static_cast<unsigned long long>(writer.backing_writer_size),
                        static_cast<unsigned long long>(writer.backing_writer_order),
                        static_cast<unsigned long long>(writer.backing_writer_identity));
        }
    }
    std::fprintf(stderr, "gpu_timeline: depth-summary surfaces=%zu base-identities=%zu "
                         "transitions=%llu backing-transitions=%llu%s\n",
                 totals.size(), previous.size(), static_cast<unsigned long long>(transitions),
                 static_cast<unsigned long long>(backing_transitions),
                 timeline.version < 5 ? " (timeline predates v5 depth metadata)" : "");
    return totals.empty() ? 1 : 0;
}

bool parse_u32_range(const char* text, uint32_t& first, uint32_t& last) {
    const std::string value(text ? text : "");
    const size_t colon = value.find(':');
    auto parse_one = [](const std::string& part, uint32_t& out) {
        if (part.empty()) return false;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(part.c_str(), &end, 0);
        if (!end || *end || parsed > UINT32_MAX) return false;
        out = static_cast<uint32_t>(parsed);
        return true;
    };
    if (colon == std::string::npos) {
        if (!parse_one(value, first)) return false;
        last = first;
        return true;
    }
    if (value.find(':', colon + 1) != std::string::npos ||
        !parse_one(value.substr(0, colon), first) ||
        !parse_one(value.substr(colon + 1), last))
        return false;
    return first <= last;
}

bool parse_dimensions(const char* text, uint32_t& width, uint32_t& height) {
    unsigned parsed_width = 0, parsed_height = 0;
    char tail = 0;
    if (!text || std::sscanf(text, "%ux%u%c", &parsed_width, &parsed_height, &tail) != 2 ||
        !parsed_width || !parsed_height)
        return false;
    width = parsed_width;
    height = parsed_height;
    return true;
}

std::string target_signature(const GpuTimelineSubmit& submit) {
    std::ostringstream out;
    if (submit.target_spans.empty()) out << "unavailable";
    for (size_t i = 0; i < submit.target_spans.size(); ++i) {
        const auto& span = submit.target_spans[i];
        if (i) out << ',';
        out << span.first_draw;
        if (span.draw_count > 1) out << '-' << (uint64_t{span.first_draw} + span.draw_count - 1);
        out << ':' << span.width << 'x' << span.height;
    }
    if (submit.target_spans_truncated) out << ",truncated";
    return out.str();
}

int main(int argc, char** argv) {
    const bool records = argc == 3 && std::string(argv[2]) == "--records";
    const bool depth_summary = (argc == 3 || argc == 4) &&
                               std::string(argv[2]) == "--depth-summary";
    const bool signatures = argc == 5 && std::string(argv[2]) == "--signatures";
    const bool select = argc == 7 && std::string(argv[2]) == "--select";
    uint32_t depth_width = 0, depth_height = 0;
    if (depth_summary && argc == 4) {
        unsigned width = 0, height = 0; char tail = 0;
        if (std::sscanf(argv[3], "%ux%u%c", &width, &height, &tail) == 2 && width && height) {
            depth_width = width; depth_height = height;
        }
    }
    uint32_t filter_min_draws = 0, filter_max_draws = 0;
    uint32_t filter_min_dispatches = 0, filter_max_dispatches = 0;
    uint32_t select_width = 0, select_height = 0;
    uint32_t select_min_draw = 0, select_max_draw = 0;
    const bool filter_ok = !signatures ||
        (parse_u32_range(argv[3], filter_min_draws, filter_max_draws) &&
         parse_u32_range(argv[4], filter_min_dispatches, filter_max_dispatches));
    const bool select_ok = !select ||
        (parse_dimensions(argv[3], select_width, select_height) &&
         parse_u32_range(argv[4], select_min_draw, select_max_draw) &&
         parse_u32_range(argv[5], filter_min_draws, filter_max_draws) &&
         parse_u32_range(argv[6], filter_min_dispatches, filter_max_dispatches));
    const bool basic = argc == 2;
    if (argc < 2 || (!basic && !records && !depth_summary && !signatures && !select) ||
        (depth_summary && argc == 4 && !depth_width) || !filter_ok || !select_ok) {
        std::fprintf(stderr,
            "usage: gpu_timeline <capture.prgtl> [--records|--depth-summary [WxH]]\n"
            "       gpu_timeline <capture.prgtl> --signatures DRAWS DISPATCHES\n"
            "       gpu_timeline <capture.prgtl> --select WxH DRAW_INDEX DRAWS DISPATCHES\n"
            "ranges accept N or MIN:MAX\n");
        return 2;
    }
    GpuTimelineFile timeline;
    std::string error;
    if (!read_gpu_timeline(argv[1], timeline, error)) {
        std::fprintf(stderr, "gpu_timeline: %s: %s\n", argv[1], error.c_str());
        return 1;
    }
    if (depth_summary) return print_depth_summary(timeline, depth_width, depth_height);
    if (signatures || select) {
        if (timeline.version < 6) {
            std::fprintf(stderr,
                         "gpu_timeline: target signatures require timeline version 6 (file is v%u)\n",
                         timeline.version);
            return 1;
        }
        if (signatures) {
            struct Stats { uint64_t count = 0, first = UINT64_MAX, last = 0; };
            std::map<std::string, Stats> grouped;
            for (const auto& submit : timeline.submits) {
                if (submit.draw_count < filter_min_draws || submit.draw_count > filter_max_draws ||
                    submit.dispatch_count < filter_min_dispatches ||
                    submit.dispatch_count > filter_max_dispatches)
                    continue;
                const std::string signature = "draws=" + std::to_string(submit.draw_count) +
                    " dispatches=" + std::to_string(submit.dispatch_count) +
                    " dmas=" + std::to_string(timeline.version >= 10
                                                   ? submit.dma_data_count
                                                   : submit.dma_copy_count) +
                    " capture=" + (submit.capture_incomplete ? "incomplete" : "complete") +
                    " targets=" + target_signature(submit);
                Stats& stats = grouped[signature];
                ++stats.count;
                stats.first = std::min(stats.first, submit.submit_no);
                stats.last = std::max(stats.last, submit.submit_no);
            }
            for (const auto& [signature, stats] : grouped)
                std::printf("signature submits=%llu first=%llu last=%llu %s\n",
                            static_cast<unsigned long long>(stats.count),
                            static_cast<unsigned long long>(stats.first),
                            static_cast<unsigned long long>(stats.last), signature.c_str());
            std::fprintf(stderr, "gpu_timeline: signatures=%zu\n", grouped.size());
            return grouped.empty() ? 1 : 0;
        }
        GpuTimelineSelector selector;
        selector.target_width = select_width;
        selector.target_height = select_height;
        selector.target_min_draw = select_min_draw;
        selector.target_max_draw = select_max_draw;
        selector.min_draws = filter_min_draws;
        selector.max_draws = filter_max_draws;
        selector.min_dispatches = filter_min_dispatches;
        selector.max_dispatches = filter_max_dispatches;
        uint64_t matches = 0;
        uint64_t incomplete_candidates = 0;
        const GpuTimelineSubmit* first_match = nullptr;
        const GpuTimelineSubmit* last_match = nullptr;
        for (const auto& submit : timeline.submits) {
            if (submit.target_spans_truncated &&
                submit.draw_count >= selector.min_draws && submit.draw_count <= selector.max_draws &&
                submit.dispatch_count >= selector.min_dispatches &&
                submit.dispatch_count <= selector.max_dispatches)
                ++incomplete_candidates;
            if (!gpu_timeline_submit_matches(submit, selector)) continue;
            if (!first_match) first_match = &submit;
            last_match = &submit;
            ++matches;
        }
        auto print_match = [](const char* label, const GpuTimelineSubmit& submit) {
            std::printf("%s submit=%llu t=%.6f draws=%u dispatches=%u targets=%s\n", label,
                        static_cast<unsigned long long>(submit.submit_no), submit.elapsed_ns / 1e9,
                        submit.draw_count, submit.dispatch_count, target_signature(submit).c_str());
        };
        if (first_match) {
            print_match("first-match", *first_match);
            if (last_match != first_match) print_match("last-match", *last_match);
        }
        std::fprintf(stderr, "gpu_timeline: matches=%llu%s\n",
                     static_cast<unsigned long long>(matches),
                     matches ? " (capture selects first match)" : "");
        if (incomplete_candidates) {
            std::fprintf(stderr,
                         "gpu_timeline: selector is inconclusive: %llu count-matching submits have "
                         "truncated target spans\n",
                         static_cast<unsigned long long>(incomplete_candidates));
            return 2;
        }
        return matches ? 0 : 1;
    }

    uint64_t last_ns = 0, draws = 0, dispatches = 0, dmas = 0, dma_copies = 0;
    uint64_t incomplete = 0, dma_journals_truncated = 0;
    std::map<std::pair<uint32_t, uint32_t>, uint64_t> target_extents;
    for (const auto& submit : timeline.submits) {
        last_ns = std::max(last_ns, submit.elapsed_ns);
        draws += submit.draw_count;
        dispatches += submit.dispatch_count;
        dmas += timeline.version >= 10 ? submit.dma_data_count : submit.dma_copy_count;
        dma_copies += submit.dma_copy_count;
        dma_journals_truncated += submit.dma_data_records_truncated;
        incomplete += submit.capture_incomplete;
        if (submit.color0_width && submit.color0_height)
            target_extents[{submit.color0_width, submit.color0_height}]++;
    }
    for (const auto& present : timeline.presents) last_ns = std::max(last_ns, present.elapsed_ns);
    const double seconds = static_cast<double>(last_ns) / 1e9;
    std::printf("timeline version=%u revision=%s title=%s route=%s\n",
                timeline.version, timeline.metadata.revision.c_str(), timeline.metadata.title_id.c_str(),
                timeline.metadata.input_route.c_str());
    std::printf("duration=%.3fs submits=%zu presents=%zu details=%zu producers=%zu "
                "draws=%llu dispatches=%llu dmas=%llu dma_copies=%llu "
                "dma_journals_truncated=%llu capture_incomplete=%llu truncated_tail=%s\n",
                seconds, timeline.submits.size(), timeline.presents.size(), timeline.details.size(),
                timeline.producers.size(),
                static_cast<unsigned long long>(draws), static_cast<unsigned long long>(dispatches),
                static_cast<unsigned long long>(dmas),
                static_cast<unsigned long long>(dma_copies),
                static_cast<unsigned long long>(dma_journals_truncated),
                static_cast<unsigned long long>(incomplete),
                timeline.truncated_tail ? "yes" : "no");
    if (seconds > 0)
        std::printf("rates submits=%.1f/s presents=%.1f/s\n",
                    timeline.submits.size() / seconds, timeline.presents.size() / seconds);
    for (const auto& [extent, count] : target_extents)
        std::printf("target %ux%u submits=%llu\n", extent.first, extent.second,
                    static_cast<unsigned long long>(count));
    for (const auto& detail : timeline.details)
        std::printf("detail submit=%llu realized=%u/%u draws %u/%u dispatches operations=%u missing=%u "
                    "versions=%u shaders/%u resources bytes=%llu capture=%s\n",
                    static_cast<unsigned long long>(detail.submit_no), detail.realized_draw_count,
                    detail.semantic_draw_count, detail.realized_dispatch_count,
                    detail.semantic_dispatch_count, detail.operation_count,
                    detail.missing_operation_count, detail.shader_version_count,
                    detail.resource_version_count, static_cast<unsigned long long>(detail.resource_bytes),
                    detail.capture_path.c_str());
    for (const auto& producer : timeline.producers)
        std::printf("producer consumer=%llu/op%u resource=%016llx/%ux%u -> %s"
                    " submit=%llu draw=%llu order=%llu target=%016llx/%ux%u "
                    "lifetime=%llu..%llu submits=%llu writes=%llu window-first=%llu "
                    "lifetime-truncated=%s window-truncated=%s lower-bound=%llu provenance=%s "
                    "first-kind=%s first-draw=%llu first-order=%llu clear=%s/%08x:%08x "
                    "mode=%u mask=%x format=%u\n",
                    static_cast<unsigned long long>(producer.consumer_submit_no),
                    producer.consumer_operation,
                    static_cast<unsigned long long>(producer.resource_addr),
                    producer.resource_width, producer.resource_height,
                    producer.resolved ? "resolved" : "unresolved",
                    static_cast<unsigned long long>(producer.producer_submit_no),
                    static_cast<unsigned long long>(producer.producer_draw_index),
                    static_cast<unsigned long long>(producer.producer_command_order),
                    static_cast<unsigned long long>(producer.producer_target_addr),
                    producer.producer_width, producer.producer_height,
                    static_cast<unsigned long long>(producer.history_first_submit_no),
                    static_cast<unsigned long long>(producer.producer_submit_no),
                    static_cast<unsigned long long>(producer.history_submit_count),
                    static_cast<unsigned long long>(producer.history_write_count),
                    static_cast<unsigned long long>(producer.history_window_first_submit_no),
                    producer.lifetime_truncated ? "yes" : "no",
                    producer.history_window_truncated ? "yes" : "no",
                    static_cast<unsigned long long>(producer.history_lower_bound_submit_no),
                    producer_provenance_name(producer.provenance),
                    writer_kind_name(producer.first_writer_kind),
                    static_cast<unsigned long long>(producer.history_first_draw_index),
                    static_cast<unsigned long long>(producer.history_first_command_order),
                    producer.first_color_has_clear ? "programmed" : "absent",
                    producer.first_color_clear_word0, producer.first_color_clear_word1,
                    producer.first_color_control_mode, producer.first_target_mask,
                    producer.first_color_format);

    if (records) {
        size_t si = 0, pi = 0, di = 0, xi = 0;
        while (si < timeline.submits.size() || pi < timeline.presents.size() ||
               di < timeline.details.size() || xi < timeline.producers.size()) {
            const uint64_t ss = si < timeline.submits.size() ? timeline.submits[si].sequence : UINT64_MAX;
            const uint64_t ps = pi < timeline.presents.size() ? timeline.presents[pi].sequence : UINT64_MAX;
            const uint64_t ds = di < timeline.details.size() ? timeline.details[di].sequence : UINT64_MAX;
            const uint64_t xs = xi < timeline.producers.size() ? timeline.producers[xi].sequence : UINT64_MAX;
            if (ss <= ps && ss <= ds && ss <= xs) {
                const auto& s = timeline.submits[si++];
                const uint64_t dma_count = timeline.version >= 10
                    ? s.dma_data_count : s.dma_copy_count;
                std::printf("%llu %.6f submit=%llu draws=%u dispatches=%u dmas=%llu "
                            "dma-copies=%u dma-journal=%s capture=%s "
                            "order=%llu..%llu target=%016llx/%ux%u\n",
                            static_cast<unsigned long long>(s.sequence), s.elapsed_ns / 1e9,
                            static_cast<unsigned long long>(s.submit_no), s.draw_count, s.dispatch_count,
                            static_cast<unsigned long long>(dma_count), s.dma_copy_count,
                            timeline.version < 10 ? "unavailable" :
                                (s.dma_data_records_truncated ? "truncated" : "complete"),
                            s.capture_incomplete ? "incomplete" : "complete",
                            static_cast<unsigned long long>(s.first_command_order),
                            static_cast<unsigned long long>(s.last_command_order),
                            static_cast<unsigned long long>(s.color0_base),
                            s.color0_width, s.color0_height);
                for (const auto& copy : s.dma_copies)
                    std::printf("  dma-copy order=%llu src=%016llx dst=%016llx bytes=%u sels=%08x packet=%016llx\n",
                                static_cast<unsigned long long>(copy.command_order),
                                static_cast<unsigned long long>(copy.src),
                                static_cast<unsigned long long>(copy.dst), copy.bytes, copy.sels,
                                static_cast<unsigned long long>(copy.packet_addr));
                for (const auto& record : s.dma_data_records)
                    std::printf("  dma-data order=%llu src=%016llx dst=%016llx bytes=%u "
                                "dst-sel=%02x src-sel=%02x sels=%08x packet=%016llx\n",
                                static_cast<unsigned long long>(record.command_order),
                                static_cast<unsigned long long>(record.src),
                                static_cast<unsigned long long>(record.dst), record.bytes,
                                record.sels & 0xffu, (record.sels >> 8) & 0xffu, record.sels,
                                static_cast<unsigned long long>(record.packet_addr));
                for (const auto& depth : s.depth_surfaces)
                    std::printf("  depth z=%016llx/%016llx s=%016llx/%016llx htile=%016llx "
                                "target=%ux%u draws=%u tests=%u writes=%u clears=%u compare-mask=%08x\n",
                                static_cast<unsigned long long>(depth.depth_read_base),
                                static_cast<unsigned long long>(depth.depth_write_base),
                                static_cast<unsigned long long>(depth.stencil_read_base),
                                static_cast<unsigned long long>(depth.stencil_write_base),
                                static_cast<unsigned long long>(depth.htile_data_base),
                                depth.target_width, depth.target_height, depth.draw_count,
                                depth.depth_test_count, depth.depth_write_count,
                                depth.clear_count, depth.compare_mask);
                for (const auto& depth : s.depth_surfaces)
                    if (depth.backing_hash_mask)
                        std::printf("    backing mask=%x hash=%016llx/%016llx/%016llx\n",
                                    depth.backing_hash_mask,
                                    static_cast<unsigned long long>(depth.depth_backing_hash),
                                    static_cast<unsigned long long>(depth.stencil_backing_hash),
                                    static_cast<unsigned long long>(depth.htile_backing_hash));
                for (const auto& depth : s.depth_surfaces)
                    if (depth.backing_writer_sequence)
                        std::printf("    writer kind=%s sequence=%llu range=%016llx/+%llx "
                                    "order=%llu identity=%016llx\n",
                                    writer_kind_name(static_cast<GpuTimelineWriterKind>(depth.backing_writer_kind)),
                                    static_cast<unsigned long long>(depth.backing_writer_sequence),
                                    static_cast<unsigned long long>(depth.backing_writer_addr),
                                    static_cast<unsigned long long>(depth.backing_writer_size),
                                    static_cast<unsigned long long>(depth.backing_writer_order),
                                    static_cast<unsigned long long>(depth.backing_writer_identity));
                for (const auto& span : s.target_spans)
                    std::printf("  target-span draws=%u..%llu target=%ux%u\n",
                                span.first_draw,
                                static_cast<unsigned long long>(
                                    uint64_t{span.first_draw} + span.draw_count - 1),
                                span.width, span.height);
                if (s.target_spans_truncated)
                    std::printf("  target-spans truncated=yes\n");
            } else if (ps <= ds && ps <= xs) {
                const auto& p = timeline.presents[pi++];
                std::printf("%llu %.6f present=%llu latest-submit=%llu buffer=%d flip-arg=%lld extent=%ux%u\n",
                            static_cast<unsigned long long>(p.sequence), p.elapsed_ns / 1e9,
                            static_cast<unsigned long long>(p.present_count),
                            static_cast<unsigned long long>(p.latest_submit_no), p.buffer_index,
                            static_cast<long long>(p.flip_arg), p.width, p.height);
            } else if (ds <= xs) {
                const auto& d = timeline.details[di++];
                std::printf("%llu %.6f detail-submit=%llu realized=%u/%u+%u/%u operations=%u missing=%u "
                            "versions=%u/%u bytes=%llu capture=%s\n",
                            static_cast<unsigned long long>(d.sequence), d.elapsed_ns / 1e9,
                            static_cast<unsigned long long>(d.submit_no), d.realized_draw_count,
                            d.semantic_draw_count, d.realized_dispatch_count, d.semantic_dispatch_count,
                            d.operation_count, d.missing_operation_count, d.shader_version_count,
                            d.resource_version_count, static_cast<unsigned long long>(d.resource_bytes),
                            d.capture_path.c_str());
            } else {
                const auto& x = timeline.producers[xi++];
                std::printf("%llu %.6f producer consumer=%llu/op%u resource=%016llx/%ux%u "
                            "future=%u resolved=%s submit=%llu draw=%llu order=%llu "
                            "lifetime=%llu..%llu/%llu writes=%llu window=%llu lower-bound=%llu "
                            "lifetime-truncated=%s window-truncated=%s provenance=%s kind=%s\n",
                            static_cast<unsigned long long>(x.sequence), x.elapsed_ns / 1e9,
                            static_cast<unsigned long long>(x.consumer_submit_no), x.consumer_operation,
                            static_cast<unsigned long long>(x.resource_addr), x.resource_width,
                            x.resource_height, x.future_writer_operation, x.resolved ? "yes" : "no",
                            static_cast<unsigned long long>(x.producer_submit_no),
                            static_cast<unsigned long long>(x.producer_draw_index),
                            static_cast<unsigned long long>(x.producer_command_order),
                            static_cast<unsigned long long>(x.history_first_submit_no),
                            static_cast<unsigned long long>(x.producer_submit_no),
                            static_cast<unsigned long long>(x.history_submit_count),
                            static_cast<unsigned long long>(x.history_write_count),
                            static_cast<unsigned long long>(x.history_window_first_submit_no),
                            static_cast<unsigned long long>(x.history_lower_bound_submit_no),
                            x.lifetime_truncated ? "yes" : "no",
                            x.history_window_truncated ? "yes" : "no",
                            producer_provenance_name(x.provenance),
                            writer_kind_name(x.first_writer_kind));
            }
        }
    }
    if (timeline.truncated_tail)
        std::fprintf(stderr, "gpu_timeline: warning: ignored a truncated final record\n");
    return 0;
}
