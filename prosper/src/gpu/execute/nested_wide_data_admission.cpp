#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/compute_program_facts.hpp"
#include "gpu/execute/graphics_nested_wide_reader.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/memory/guest_memory_topology.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <vector>
#include <utility>

namespace prosper::gpu {

namespace {
struct Range { uint64_t address, bytes; };

bool exact_storage_write_range(const ShaderResource& r, Range& out) {
    uint64_t bytes = 0;
    if (!raw_snapshot_storage_extent(r, bytes)) return false;
    out = {r.gpu_addr, bytes};
    return true;
}

const ShaderResource* exact_buffer(const ShaderResourceTable& table, uint32_t pc) {
    const ShaderResource* r = table.by_fetch_pc(pc);
    return r && r->cls == ResourceClass::ConstantBuffer && r->fetch_pc == pc &&
                   r->gpu_addr && r->size && r->size <= (1u << 20u) && !r->host_data
        ? r : nullptr;
}
} // namespace

bool admit_compute_nested_wide_data(const GuestMappingLease* lease,
                                    const std::vector<Rdna2Inst>& decoded,
                                    const std::vector<SrtUse>& uses,
                                    ShaderResourceTable& table) {
    NestedWideDataFacts inventory;
    inventory.nested = rdna2_proven_raw_nested_wide_data_loads(decoded);
    if (inventory.nested.empty()) return true;
    if (!lease) return false;
    inventory.parents = rdna2_proven_raw_immediate_wide_data_loads(decoded);
    return admit_compute_nested_wide_data(lease, decoded, inventory, uses, table);
}

bool admit_compute_nested_wide_data(const GuestMappingLease* lease,
                                    const std::vector<Rdna2Inst>& decoded,
                                    const NestedWideDataFacts& inventory,
                                    const std::vector<SrtUse>& uses,
                                    ShaderResourceTable& table) {
    const std::vector<uint32_t>& nested = inventory.nested;
    if (nested.empty()) return true;
    if (!lease) return false;
    const std::vector<uint32_t>& parents = inventory.parents;
    std::vector<std::pair<size_t, size_t>> chains;
    std::set<uint32_t> source_pcs;
    for (uint32_t child_pc : nested) {
        const auto child = std::find_if(decoded.begin(), decoded.end(),
                                        [child_pc](const Rdna2Inst& in) {
                                            return in.pc == child_pc;
                                        });
        if (child == decoded.end() || !exact_buffer(table, child_pc)) return false;
        size_t parent_index = decoded.size();
        const size_t child_index = static_cast<size_t>(child - decoded.begin());
        for (size_t i = 0; i < child_index; ++i)
            if (decoded[i].fmt == Rdna2Format::SMEM &&
                (decoded[i].opcode == 0x2u || decoded[i].opcode == 0x3u) &&
                decoded[i].dst.kind == OperandKind::SGPR &&
                decoded[i].dst.value == child->src[0].value &&
                std::binary_search(parents.begin(), parents.end(), decoded[i].pc))
                parent_index = i;
        if (parent_index == decoded.size() ||
            !exact_buffer(table, decoded[parent_index].pc)) return false;
        chains.emplace_back(parent_index, child_index);
        source_pcs.insert(decoded[parent_index].pc);
        source_pcs.insert(child_pc);
    }

    std::vector<Range> writes;
    for (const Rdna2Inst& in : decoded) {
        if (in.fmt == Rdna2Format::MIMG && in.opcode == 0x08u) {
            const ShaderResource* r = table.by_fetch_pc(in.pc);
            Range range{};
            if (!r || r->fetch_pc != in.pc || !exact_storage_write_range(*r, range))
                return false;
            writes.push_back(range);
            continue;
        }
        // Known read-only FLAT and comparison sample opcodes are conservatively classified as
        // possible writes by the older raw-load inventory. They cannot change a source here.
        if ((in.fmt == Rdna2Format::FLAT && in.opcode >= 0x08u &&
             in.opcode <= 0x0fu) ||
            (in.fmt == Rdna2Format::MIMG && in.opcode == 0x2fu)) continue;
        if (rdna2_may_write_guest_memory(in)) return false;
    }
    if (writes.empty()) return false;

    std::vector<Range> sources;
    for (uint32_t pc : source_pcs) {
        const ShaderResource* r = exact_buffer(table, pc);
        if (!r) return false;
        sources.push_back({r->gpu_addr, r->size});
    }
    // Every image descriptor that this shader consumes must have an exact byte source. A folded
    // but untraced descriptor could itself alias the store and change between waves.
    for (const Rdna2Inst& in : decoded) {
        if (in.fmt != Rdna2Format::MIMG) continue;
        const auto use = std::find_if(uses.begin(), uses.end(), [&](const SrtUse& candidate) {
            return candidate.kind == 0 && candidate.use_pc == in.pc;
        });
        if (use == uses.end() || !use->descriptor_source_addr ||
            use->descriptor_source_addr > UINT64_MAX - 32u ||
            !guest_memory_direct_range_fault_safe(*lease, use->descriptor_source_addr, 32u) ||
            !guest_readable(use->descriptor_source_addr, 32u) ||
            std::memcmp(reinterpret_cast<const void*>(
                            static_cast<uintptr_t>(use->descriptor_source_addr)),
                        use->t8.data(), 32u) != 0) return false;
        sources.push_back({use->descriptor_source_addr, 32u});
    }
    for (const Range& write : writes)
        if (!guest_memory_direct_range_fault_safe(*lease, write.address, write.bytes))
            return false;
    for (const Range& source : sources) {
        if (!guest_memory_direct_range_fault_safe(*lease, source.address, source.bytes))
            return false;
        for (const Range& write : writes)
            if (guest_memory_topology_relation(source.address, source.bytes,
                                               write.address, write.bytes) !=
                GuestMemoryTopologyRelation::Disjoint)
                return false;
    }

    // Stage every member of the pointer chain before publishing any host_data pointer. The
    // original fold computed child addresses from current parent bytes. Verify those exact bytes
    // again: if the parent changed between fold and staging, refuse the mixed snapshot.
    std::vector<std::pair<uint32_t, std::shared_ptr<std::vector<uint8_t>>>> snapshots;
    for (uint32_t pc : source_pcs) {
        const ShaderResource* r = exact_buffer(table, pc);
        if (!r || !guest_readable(r->gpu_addr, r->size)) return false;
        const auto* src = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(r->gpu_addr));
        snapshots.emplace_back(pc, std::make_shared<std::vector<uint8_t>>(
            src, src + r->size));
    }
    const auto snapshot_for = [&](uint32_t pc) -> const std::vector<uint8_t>* {
        const auto it = std::find_if(snapshots.begin(), snapshots.end(),
                                     [pc](const auto& entry) { return entry.first == pc; });
        return it == snapshots.end() ? nullptr : it->second.get();
    };
    for (const auto [parent_index, child_index] : chains) {
        const Rdna2Inst& parent = decoded[parent_index];
        const Rdna2Inst& child = decoded[child_index];
        const auto* bytes = snapshot_for(parent.pc);
        const ShaderResource* child_resource = exact_buffer(table, child.pc);
        if (!bytes || !child_resource ||
            static_cast<uint64_t>(parent.literal) + sizeof(uint64_t) > bytes->size())
            return false;
        uint64_t pointer = 0;
        std::memcpy(&pointer, bytes->data() + parent.literal, sizeof(pointer));
        if (pointer != child_resource->gpu_addr) return false;
    }
    for (auto& [pc, bytes] : snapshots) {
        auto* r = const_cast<ShaderResource*>(exact_buffer(table, pc));
        if (!r) return false; // no table mutation before this point
        r->host_data = bytes->data();
        r->host_data_size = bytes->size();
        if (std::binary_search(nested.begin(), nested.end(), pc))
            r->nested_raw_snapshot_admitted = true;
        table.owned_host_data.push_back(std::move(bytes));
    }
    return true;
}
static GraphicsRawSourceAuthorityFn g_graphics_raw_source_authority;
void set_graphics_raw_source_authority(GraphicsRawSourceAuthorityFn fn) {
    g_graphics_raw_source_authority = std::move(fn);
}
bool graphics_raw_source_is_guest_current(const GuestMappingLease& lease,
                                          uint64_t address, uint32_t bytes) {
    return g_graphics_raw_source_authority && g_graphics_raw_source_authority(lease, address, bytes);
}
static GraphicsProducerStatusFn g_graphics_producer_status;
void set_graphics_producer_status_query(GraphicsProducerStatusFn fn) {
    g_graphics_producer_status = std::move(fn);
}
GraphicsProducerStatus graphics_producer_status() {
    return g_graphics_producer_status ? g_graphics_producer_status() : GraphicsProducerStatus{};
}

bool observe_graphics_raw_wave_windows(
    const std::vector<uint32_t>& original,
    const std::vector<std::pair<uint32_t, uint32_t>>& entry_scalars,
    const GraphicsRawSnapshotContext* context, std::vector<PacketRawWaveWindow>& windows,
    std::string& refusal) {
    windows.clear();
    refusal.clear();
    const auto reject = [&](const char* why) {
        windows.clear();
        refusal = why;
        return false;
    };
    if (!context || !context->producers_complete) return reject("wave-window-producer-incomplete");
    std::vector<Rdna2Inst> decoded;
    if (original.empty() ||
        rdna2_walk(original.data(), original.size(), decoded) != original.size())
        return reject("wave-window-original-code-incomplete");
    const auto certificates = rdna2_raw_wave_wide_certificates(decoded);
    if (certificates.empty()) return reject("wave-window-selector-unproved");
    std::map<uint32_t, uint32_t> scalars;
    for (const auto& [reg, value] : entry_scalars)
        if (reg > 105u || !scalars.emplace(reg, value).second)
            return reject("wave-window-entry-scalars-invalid");
    GuestMappingLease lease;
    struct Pending {
        uint32_t pc;
        uint64_t base, address;
        uint32_t bytes;
    };
    std::vector<Pending> pending;
    uint64_t total = 0;
    for (const auto& proof : certificates) {
        if (!scalars.contains(proof.base_sgpr) || !scalars.contains(proof.base_sgpr + 1u))
            return reject("wave-window-entry-base-unavailable");
        const uint64_t base = uint64_t(scalars.at(proof.base_sgpr)) |
                              (uint64_t(scalars.at(proof.base_sgpr + 1u)) << 32u);
        const uint64_t begin = uint64_t(int64_t(proof.offset_min) + proof.immediate);
        const uint64_t bytes = uint64_t(proof.offset_max) - proof.offset_min + proof.bytes;
        if (base <= 0x10000u || (base & 3u) || base > UINT64_MAX - begin ||
            base + begin > UINT64_MAX - bytes || !bytes || bytes > UINT32_MAX ||
            certificates.size() > 8u || total + bytes > 64u * 1024u * 1024u)
            return reject("wave-window-domain-unrepresentable");
        const uint64_t address = base + begin;
        const auto allocation = guest_memory_direct_readable_window(lease, address);
        if (!allocation || address < allocation.virtual_begin ||
            bytes > allocation.virtual_end - address)
            return reject("wave-window-readable-allocation-incomplete");
        if (!graphics_raw_source_is_guest_current(lease, address, uint32_t(bytes)))
            return reject("wave-window-current-source-unproved");
        for (const auto& [output, physical_bytes] : context->output_allocations)
            if (output && guest_memory_direct_allocation_relation(lease, address, bytes, output,
                                                                  physical_bytes) !=
                              GuestMemoryTopologyRelation::Disjoint)
                return reject("wave-window-output-allocation-not-disjoint");
        pending.push_back({proof.load_pc, base, address, uint32_t(bytes)});
        total += bytes;
    }
    // All domains and retained/current attachment exclusions are checked before the FIRST read.
    // The lease authenticates topology through the copies; submitted input stability is the same
    // explicit guest contract as other draw inputs, not an invented guest-CPU write exclusion.
    for (const auto& range : pending) {
        PacketRawWaveWindow window;
        window.load_pc = range.pc;
        window.guest_base = range.base;
        window.guest_begin = range.address;
        window.words.resize(range.bytes / sizeof(uint32_t));
        std::memcpy(window.words.data(), reinterpret_cast<const void*>(uintptr_t(range.address)),
                    range.bytes);
        windows.push_back(std::move(window));
    }
    return true;
}

bool validate_graphics_raw_wave_windows(
    const std::vector<uint32_t>& original,
    const std::vector<std::pair<uint32_t, uint32_t>>& entry_scalars,
    const std::vector<PacketRawWaveWindow>& windows, std::string& refusal) {
    refusal.clear();
    const auto reject = [&](const char* reason) {
        refusal = reason;
        return false;
    };
    std::vector<Rdna2Inst> decoded;
    if (original.empty() || original.size() > 4096u ||
        rdna2_walk(original.data(), original.size(), decoded) != original.size() ||
        decoded.empty() || !decoded.back().is_end)
        return reject("wave-window-original-code-unavailable");
    const auto proofs = rdna2_raw_wave_wide_certificates(decoded);
    if (proofs.empty() || proofs.size() != windows.size() || windows.size() > 8u)
        return reject("wave-window-exact-obligations-incomplete");
    std::map<uint32_t, uint32_t> scalars;
    for (const auto& [reg, value] : entry_scalars)
        if (reg >= 106u || !scalars.emplace(reg, value).second)
            return reject("wave-window-entry-scalar-unavailable");
    std::set<uint32_t> pcs;
    uint64_t total_words = 0;
    for (const auto& window : windows) {
        const auto proof = std::find_if(proofs.begin(), proofs.end(),
                                        [&](const auto& p) { return p.load_pc == window.load_pc; });
        if (proof == proofs.end() || !pcs.insert(window.load_pc).second ||
            !scalars.contains(proof->base_sgpr) || !scalars.contains(proof->base_sgpr + 1u))
            return reject("wave-window-exact-code-or-base-unavailable");
        const uint64_t base =
            scalars.at(proof->base_sgpr) | (uint64_t(scalars.at(proof->base_sgpr + 1u)) << 32u);
        const int64_t signed_offset = int64_t(proof->offset_min) + proof->immediate;
        const uint64_t bytes = uint64_t(proof->offset_max) - proof->offset_min + proof->bytes;
        if (!base || (base & 3u) || signed_offset < 0 ||
            base > UINT64_MAX - uint64_t(signed_offset) ||
            base + uint64_t(signed_offset) > UINT64_MAX - bytes || window.guest_base != base ||
            window.guest_begin != base + uint64_t(signed_offset) || (bytes & 3u) ||
            window.words.size() != bytes / 4u)
            return reject("wave-window-complete-owner-shape-invalid");
        total_words += window.words.size();
        if (total_words > 16u * 1024u * 1024u) return reject("wave-window-owner-budget");
    }
    return true;
}

bool prepare_owned_vertex_waves(const std::vector<uint32_t>& original,
                                const std::vector<std::pair<uint32_t, uint32_t>>& entry_scalars,
                                const std::vector<uint32_t>& owned_indices,
                                uint32_t nonindexed_count, int32_t vertex_offset,
                                uint32_t instance_count, FloatTransportConfig profile,
                                const GraphicsRawSnapshotContext* context,
                                GraphicsWaveStagePlan& plan, std::string& refusal) {
    plan = {};
    refusal.clear();
    const auto reject = [&](const char* reason) {
        plan = {};
        refusal = reason;
        return false;
    };
    const uint64_t vertices = owned_indices.empty() ? nonindexed_count : owned_indices.size();
    if (!profile.canonical() || !vertices || !instance_count || vertices > 4096u ||
        instance_count > 64u || vertices * instance_count > 4096u ||
        (vertices * instance_count) % kFragmentPacketLanes)
        return reject("vertex-wave-complete-domain-unavailable");
    // Numeric sources are observed once for the whole draw, after completion/isolation admission.
    // Different waves select independently from this same immutable current-byte observation.
    std::vector<PacketRawWaveWindow> windows;
    if (!observe_graphics_raw_wave_windows(original, entry_scalars, context, windows, refusal))
        return false;
    uint64_t window_bytes = 0;
    for (const auto& window : windows) window_bytes += window.words.size() * uint64_t(4);
    if (window_bytes * (vertices * instance_count / kFragmentPacketLanes) > 64u * 1024u * 1024u)
        return reject("vertex-wave-owned-input-budget");
    GraphicsWaveStagePlan pending;
    pending.assembly = GraphicsWaveAssembly::VertexDrawOrder;
    for (uint64_t ordinal = 0; ordinal < vertices * instance_count;
         ordinal += kFragmentPacketLanes) {
        FragmentInvocationPacket packet;
        packet.stage = GraphicsPacketStage::Vertex;
        packet.guest_code = original;
        packet.sgprs = entry_scalars;
        packet.raw_windows = windows;
        packet.float_transport = profile;
        packet.slots_available.fill(true);
        packet.mask_state_available = true;
        packet.exec_mask = UINT64_MAX;
        packet.export_enabled.fill(1);
        FragmentPacketVgpr vertex, instance;
        vertex.reg = 0;
        instance.reg = 3;
        std::array<GraphicsWaveInvocation, kFragmentPacketLanes> identities;
        for (uint32_t lane = 0; lane < kFragmentPacketLanes; ++lane) {
            const uint64_t occurrence = (ordinal + lane) % vertices;
            const int64_t index =
                int64_t(owned_indices.empty() ? occurrence : owned_indices[occurrence]) +
                vertex_offset;
            if (index < 0 || index > UINT32_MAX) return reject("vertex-wave-index-unrepresentable");
            identities[lane].vertex_index = vertex.words[lane] = uint32_t(index);
            identities[lane].instance_index = instance.words[lane] =
                uint32_t((ordinal + lane) / vertices);
        }
        packet.vgprs = {std::move(vertex), std::move(instance)};
        if (!complete_graphics_packet_locals(packet, refusal, false)) return false;
        const auto program =
            recompile_fragment_packet(packet, {RecompileDiagnosticStage::Vertex, 0});
        if (program.spirv.empty()) {
            refusal = program.rejection;
            return false;
        }
        pending.packets.push_back(std::move(packet));
        pending.invocations.push_back(identities);
    }
    plan = std::move(pending);
    return true;
}

bool prepare_owned_fragment_waves(const RasterQuadInputs& inputs, const RasterQuadResult& raster,
                                  const std::vector<std::pair<uint32_t, uint32_t>>& entry_scalars,
                                  const std::vector<PacketRawWaveWindow>& observed_windows,
                                  FragmentFloatMode float_mode, FragmentFloatFlags float_flags,
                                  GraphicsWaveStagePlan& plan, std::string& refusal) {
    plan = {};
    refusal.clear();
    const auto reject = [&](const char* reason) {
        plan = {};
        refusal = reason;
        return false;
    };
    if (!raster.complete || !raster.rejection.empty() || !raster.host_raster_domain_available ||
        raster.host_sample_count != 1u || raster.host_sample_index || raster.host_layer ||
        raster.host_view_index || raster.host_instance_count != 1u || raster.host_first_instance ||
        !inputs.raw_code || inputs.raw_code->empty() || !inputs.raw_matches_producing_source ||
        !inputs.source_vs || !raster.vertex_source || *inputs.source_vs != *raster.vertex_source ||
        !inputs.float_transport.canonical() || !float_mode.canonical() || !float_flags.canonical())
        return reject("fragment-wave-completed-input-domain-unavailable");
    if (!inputs.entry.observed || !inputs.entry.canonical() || !inputs.entry.rsrc2_available)
        return reject("fragment-wave-user-prefix-unavailable");
    namespace P = prosper::agc::Pm4;
    const uint32_t loaded =
        ((inputs.entry.rsrc2 >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT) &
         P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MASK) |
        (((inputs.entry.rsrc2 >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_SHIFT) &
          P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_MASK)
         << 5u);
    if (loaded > inputs.entry.user_data.size())
        return reject("fragment-wave-user-prefix-count-out-of-range");
    // Same producing PS prefix as fragment_packet_preparation. Missing physical words stay
    // absent; the per-slot MUST proof decides whether this original actually reads them. Physical
    // USER_DATA[loaded] cannot supply the first system SGPR or parameter/M0 authority.
    std::vector<std::pair<uint32_t, uint32_t>> observed_scalars;
    for (uint32_t reg = 0; reg < loaded; ++reg)
        if (inputs.entry.user_data_available & (uint32_t(1) << reg))
            observed_scalars.emplace_back(reg, inputs.entry.user_data[reg]);
    if (observed_scalars != entry_scalars)
        return reject("fragment-wave-user-prefix-observation-disagrees");
    if (!inputs.has_system_inputs || !inputs.launch.input_ena_available ||
        !inputs.launch.input_addr_available || !raster.launch.input_ena_available ||
        !raster.launch.input_addr_available ||
        inputs.system_inputs.ena != inputs.launch.input_ena ||
        inputs.system_inputs.addr != inputs.launch.input_addr ||
        raster.launch.input_ena != inputs.launch.input_ena ||
        raster.launch.input_addr != inputs.launch.input_addr)
        return reject("fragment-wave-system-input-routing-unavailable");
    // This bridge consumes fixed-function raw position/layer input words. It does not upgrade an
    // arbitrary collected interpolant's necessary writer evidence into an all-path definition.
    // VINTRP is still refused by the packet's actual instruction inventory.
    if (raster.quads.empty() || raster.quads.size() > 4096u || raster.quads.size() % 16u ||
        raster.lane_words < kRasterQuadLaneFixedWords)
        return reject("fragment-wave-complete-quad-domain-unavailable");
    uint64_t window_bytes = 0;
    for (const auto& window : observed_windows) {
        if (window.words.size() > 16u * 1024u * 1024u)
            return reject("fragment-wave-owned-input-budget");
        window_bytes += window.words.size() * uint64_t(4);
    }
    if (observed_windows.size() > 8u ||
        window_bytes * (raster.quads.size() / 16u) > 64u * 1024u * 1024u)
        return reject("fragment-wave-owned-input-budget");
    std::vector<std::vector<uint32_t>> quads = raster.quads;
    for (const auto& quad : quads)
        if (quad.size() != uint64_t(raster.lane_words) * 4u)
            return reject("fragment-wave-quad-record-shape-invalid");
    // Decoder-authenticated quads are sorted by explicit primitive and spatial origin. Raw words
    // break otherwise identical disjoint scopes deterministically; atomic append order is unused.
    // This is our logical assembly policy, not a claim about physical console scheduling.
    std::sort(quads.begin(), quads.end(), [](const auto& a, const auto& b) {
        const auto key = [](const auto& q) { return std::array<uint32_t, 3>{q[4], q[6], q[5]}; };
        return key(a) != key(b) ? key(a) < key(b) : a < b;
    });
    GraphicsWaveStagePlan pending;
    pending.assembly = GraphicsWaveAssembly::FragmentPrimitiveQuadOrder;
    static constexpr uint8_t widths[16] = {2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    for (size_t first = 0; first < quads.size(); first += 16u) {
        FragmentInvocationPacket packet;
        packet.guest_code = *inputs.raw_code;
        packet.sgprs = entry_scalars;
        packet.raw_windows = observed_windows;
        packet.float_mode = float_mode;
        packet.float_flags = float_flags;
        packet.float_transport = inputs.float_transport;
        packet.quad_topology = FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
        packet.slots_available.fill(true);
        packet.mask_state_available = true;
        std::array<GraphicsWaveInvocation, kFragmentPacketLanes> identities;
        std::map<uint32_t, FragmentPacketVgpr> columns;
        uint32_t vgpr = 0;
        for (uint32_t field = 0; field < 16u; ++field) {
            if (!(inputs.system_inputs.addr & (1u << field))) continue;
            if (inputs.system_inputs.ena & (1u << field)) {
                if ((field >= 8u && field <= 11u) || field == 13u) {
                    auto& column = columns[vgpr];
                    column.reg = vgpr;
                    for (uint32_t lane = 0; lane < kFragmentPacketLanes; ++lane) {
                        const auto& quad = quads[first + lane / 4u];
                        const uint32_t base = (lane & 3u) * raster.lane_words;
                        // Layer is proved zero by the actual single-layer collection pipeline.
                        column.words[lane] = field == 13u ? 0u : quad[base + 5u + field - 8u];
                    }
                }
                // Unsupported enabled fields still reserve their documented slots. The MUST input
                // proof below refuses an actual read rather than inventing their guest value.
            }
            vgpr += widths[field];
        }
        for (uint32_t lane = 0; lane < kFragmentPacketLanes; ++lane) {
            const auto& quad = quads[first + lane / 4u];
            const uint32_t base = (lane & 3u) * raster.lane_words;
            if (quad[base] > 1u || quad[base + 1u] != (quad[base] ? 0u : 1u) ||
                (!quad[base] && !(quad[base + 2u] & 1u)))
                return reject("fragment-wave-coverage-domain-unavailable");
            auto& identity = identities[lane];
            identity.helper = quad[base] != 0;
            identity.primitive_index = quad[base + 4u];
            std::copy_n(quad.begin() + base + 5u, 4u, identity.fragcoord.begin());
            packet.export_enabled[lane] = !identity.helper;
            if (!identity.helper) packet.exec_mask |= uint64_t(1) << lane;
        }
        for (auto& [reg, column] : columns) {
            (void)reg;
            packet.vgprs.push_back(std::move(column));
        }
        if (!complete_graphics_packet_locals(packet, refusal, false)) return false;
        const auto program = recompile_fragment_packet(packet);
        if (program.spirv.empty()) {
            refusal = program.rejection;
            return false;
        }
        pending.packets.push_back(std::move(packet));
        pending.invocations.push_back(identities);
    }
    plan = std::move(pending);
    return true;
}

bool validate_graphics_wave_outputs(const GraphicsWaveStagePlan& plan,
                                    const std::vector<std::vector<uint32_t>>& completed_records,
                                    GraphicsWaveOutputTransaction& transaction,
                                    std::string& refusal) {
    transaction = {};
    refusal.clear();
    const auto reject = [&](const char* reason) {
        refusal = reason;
        return false;
    };
    if ((plan.assembly != GraphicsWaveAssembly::VertexDrawOrder &&
         plan.assembly != GraphicsWaveAssembly::FragmentPrimitiveQuadOrder) ||
        plan.packets.empty() || plan.packets.size() > 256u ||
        plan.packets.size() != plan.invocations.size() ||
        completed_records.size() != plan.packets.size())
        return reject("graphics-wave-output-domain-incomplete");
    for (size_t wave = 0; wave < plan.packets.size(); ++wave) {
        const auto& packet = plan.packets[wave];
        if ((plan.assembly == GraphicsWaveAssembly::VertexDrawOrder &&
             packet.stage != GraphicsPacketStage::Vertex) ||
            (plan.assembly == GraphicsWaveAssembly::FragmentPrimitiveQuadOrder &&
             packet.stage != GraphicsPacketStage::Fragment))
            return reject("graphics-wave-output-stage-mismatch");
        const auto program = recompile_fragment_packet(packet);
        if (program.spirv.empty() || completed_records[wave].size() != program.output_words.size())
            return reject("graphics-wave-output-record-extent-invalid");
        if (program.vgpr_status_offset != UINT32_MAX) {
            const auto decoded = decode_fragment_packet(program, completed_records[wave], true);
            if (!decoded.rejection.empty()) {
                refusal = decoded.rejection;
                return false;
            }
        }
        std::vector<Rdna2Inst> ins;
        if (rdna2_walk(packet.guest_code.data(), packet.guest_code.size(), ins) !=
            packet.guest_code.size())
            return reject("graphics-wave-output-original-unavailable");
        std::vector<const Rdna2Inst*> exports;
        for (const auto& in : ins)
            if (in.fmt == Rdna2Format::EXP) exports.push_back(&in);
        if (exports.size() != program.exports_per_lane)
            return reject("graphics-wave-output-export-count-invalid");
        if (packet.stage == GraphicsPacketStage::Vertex) {
            const auto entry = [&](uint32_t reg) {
                return std::find_if(packet.vgprs.begin(), packet.vgprs.end(),
                                    [reg](const auto& column) { return column.reg == reg; });
            };
            const auto vertices = entry(0u), instances = entry(3u);
            if (vertices == packet.vgprs.end() || instances == packet.vgprs.end())
                return reject("vertex-wave-output-invocation-input-unavailable");
            for (uint32_t lane = 0; lane < kFragmentPacketLanes; ++lane)
                if (vertices->words[lane] != plan.invocations[wave][lane].vertex_index ||
                    instances->words[lane] != plan.invocations[wave][lane].instance_index)
                    return reject("vertex-wave-output-invocation-input-mismatch");
        }
        for (uint32_t lane = 0; lane < kFragmentPacketLanes; ++lane) {
            uint32_t vertex_positions = 0;
            for (uint32_t ordinal = 0; ordinal < exports.size(); ++ordinal) {
                const uint32_t* record =
                    completed_records[wave].data() +
                    (uint64_t(lane) * exports.size() + ordinal) * kFragmentPacketExportWords;
                if (!record[0]) {
                    if (std::any_of(record, record + kFragmentPacketExportWords,
                                    [](uint32_t word) { return word != 0; }))
                        return reject("graphics-wave-output-unreached-record-invalid");
                    continue;
                }
                const auto& in = *exports[ordinal];
                if (record[0] != 1u || record[1] > 1u || record[2] != packet.export_enabled[lane] ||
                    record[3] != in.exp_target || record[4] != in.exp_en ||
                    record[5] != in.exp_compr || record[6] != ((in.words[0] >> 11u) & 1u) ||
                    record[7] != ((in.words[0] >> 12u) & 1u))
                    return reject("graphics-wave-output-export-metadata-invalid");
                if (packet.stage == GraphicsPacketStage::Vertex && in.exp_target == 12u) {
                    if (!record[1] || !record[2] || record[4] != 15u)
                        return reject("vertex-wave-position-inactive-or-incomplete");
                    ++vertex_positions;
                }
            }
            if (packet.stage == GraphicsPacketStage::Vertex && vertex_positions != 1u)
                return reject("vertex-wave-position-transaction-incomplete");
        }
    }
    // No transfer of the first wave's data until the last wave, metadata, and mandatory position
    // have passed. These are immutable scratch outputs, not a completed guest render publication.
    transaction.records = completed_records;
    return true;
}

bool prepare_owned_vertex_export_commit(const GraphicsWaveStagePlan& plan,
                                        const GraphicsWaveOutputTransaction& transaction,
                                        const PixelInputMapping* pixel_inputs,
                                        GraphicsVertexExportCommit& commit, std::string& refusal) {
    commit = {};
    refusal.clear();
    const auto reject = [&](const char* reason) {
        refusal = reason;
        return false;
    };
    if (plan.assembly != GraphicsWaveAssembly::VertexDrawOrder)
        return reject("vertex-wave-commit-stage-invalid");
    GraphicsWaveOutputTransaction checked;
    if (!validate_graphics_wave_outputs(plan, transaction.records, checked, refusal)) return false;
    std::vector<uint32_t> parameters;
    std::vector<Rdna2Inst> original;
    rdna2_walk(plan.packets.front().guest_code.data(), plan.packets.front().guest_code.size(),
               original);
    std::vector<uint32_t> targets;
    for (const auto& in : original)
        if (in.fmt == Rdna2Format::EXP) {
            targets.push_back(in.exp_target);
            if (in.exp_target >= 32u) parameters.push_back(in.exp_target - 32u);
        }
    std::sort(parameters.begin(), parameters.end());
    parameters.erase(std::unique(parameters.begin(), parameters.end()), parameters.end());
    if (parameters.size() > 31u) return reject("vertex-wave-commit-output-budget");
    GraphicsVertexExportCommit pending;
    pending.record_words = uint32_t(parameters.size() + 1u) * 4u;
    uint32_t expected_instance = 0, occurrence = 0;
    for (size_t wave = 0; wave < plan.packets.size(); ++wave) {
        if (plan.packets[wave].guest_code != plan.packets.front().guest_code ||
            plan.packets[wave].float_transport != plan.packets.front().float_transport)
            return reject("vertex-wave-commit-producing-version-mismatch");
        for (uint32_t lane = 0; lane < kFragmentPacketLanes; ++lane) {
            const uint32_t instance = plan.invocations[wave][lane].instance_index;
            if (instance != expected_instance) {
                if (instance != expected_instance + 1u || !occurrence ||
                    (pending.vertices_per_instance && pending.vertices_per_instance != occurrence))
                    return reject("vertex-wave-commit-invocation-order-invalid");
                pending.vertices_per_instance = occurrence;
                occurrence = 0;
                expected_instance = instance;
            }
            std::vector<uint32_t> output(pending.record_words);
            std::vector<bool> available(parameters.size() + 1u, false);
            for (uint32_t ordinal = 0; ordinal < targets.size(); ++ordinal) {
                const uint32_t* record =
                    checked.records[wave].data() +
                    (uint64_t(lane) * targets.size() + ordinal) * kFragmentPacketExportWords;
                if (!record[0] || !record[1] || !record[2]) continue;
                const uint32_t field =
                    targets[ordinal] == 12u
                        ? 0u
                        : uint32_t(std::lower_bound(parameters.begin(), parameters.end(),
                                                    targets[ordinal] - 32u) -
                                   parameters.begin()) +
                              1u;
                if (record[4] != 15u || field >= available.size())
                    return reject("vertex-wave-commit-component-unavailable");
                available[field] = true;
                std::copy_n(record + 8u, 4u, output.begin() + field * 4u);
            }
            if (!std::all_of(available.begin(), available.end(), [](bool ready) { return ready; }))
                return reject("vertex-wave-commit-output-uninitialized");
            pending.words.insert(pending.words.end(), output.begin(), output.end());
            ++occurrence;
        }
    }
    if (!pending.vertices_per_instance) pending.vertices_per_instance = occurrence;
    if (occurrence != pending.vertices_per_instance || expected_instance >= 64u)
        return reject("vertex-wave-commit-instance-domain-invalid");
    pending.instances = expected_instance + 1u;
    pending.shader =
        build_owned_vertex_export_commit(parameters, pending.vertices_per_instance, pixel_inputs,
                                         plan.packets.front().float_transport);
    if (pending.shader.empty()) return reject("vertex-wave-commit-module-refused");
    commit = std::move(pending);
    return true;
}

bool prepare_owned_fragment_export_commit(const GraphicsWaveStagePlan& plan,
                                          const GraphicsWaveOutputTransaction& transaction,
                                          GraphicsFragmentExportCommit& commit,
                                          std::string& refusal) {
    commit = {};
    refusal.clear();
    const auto reject = [&](const char* reason) {
        refusal = reason;
        return false;
    };
    if (plan.assembly != GraphicsWaveAssembly::FragmentPrimitiveQuadOrder)
        return reject("fragment-wave-commit-stage-invalid");
    GraphicsWaveOutputTransaction checked;
    if (!validate_graphics_wave_outputs(plan, transaction.records, checked, refusal)) return false;
    GraphicsFragmentExportCommit pending;
    std::set<std::array<uint32_t, 3>> keys;
    for (size_t wave = 0; wave < plan.packets.size(); ++wave) {
        const auto& packet = plan.packets[wave];
        if (packet.guest_code != plan.packets.front().guest_code ||
            packet.float_transport != plan.packets.front().float_transport)
            return reject("fragment-wave-commit-producing-version-mismatch");
        std::vector<Rdna2Inst> original;
        rdna2_walk(packet.guest_code.data(), packet.guest_code.size(), original);
        std::vector<const Rdna2Inst*> exports;
        for (const auto& in : original)
            if (in.fmt == Rdna2Format::EXP) exports.push_back(&in);
        // One complete colour output in this initial commit slice. Partial/channel-dependent,
        // repeated, depth/sample-mask and MRT exports need their own ordered commit contract.
        if (exports.size() != 1u || exports[0]->exp_target != 0u || exports[0]->exp_en != 15u ||
            exports[0]->exp_compr)
            return reject("fragment-wave-commit-export-shape-unimplemented");
        for (uint32_t lane = 0; lane < kFragmentPacketLanes; ++lane) {
            const auto& identity = plan.invocations[wave][lane];
            if (identity.helper) continue;
            const std::array<uint32_t, 3> key{identity.primitive_index, identity.fragcoord[0],
                                              identity.fragcoord[1]};
            if (!keys.insert(key).second)
                return reject("fragment-wave-commit-invocation-duplicate");
            const auto* record = checked.records[wave].data() + lane * kFragmentPacketExportWords;
            const uint32_t enabled = record[0] && record[1] && record[2];
            pending.words.insert(pending.words.end(),
                                 {key[0], key[1], key[2], enabled, enabled ? record[8] : 0u,
                                  enabled ? record[9] : 0u, enabled ? record[10] : 0u,
                                  enabled ? record[11] : 0u});
        }
    }
    pending.records = uint32_t(pending.words.size() / 8u);
    if (!pending.records) return reject("fragment-wave-commit-invocation-domain-empty");
    pending.shader =
        build_owned_fragment_export_commit(pending.records, plan.packets.front().float_transport);
    if (pending.shader.empty()) return reject("fragment-wave-commit-module-refused");
    commit = std::move(pending);
    return true;
}

GraphicsNestedWideReader::GraphicsNestedWideReader(std::vector<RawNestedWideChain> chains,
                                                   const GraphicsRawSnapshotContext* context,
                                                   const GuestMappingLease* borrowed_lease)
    : chains_(std::move(chains)) {
    allowed_ = context && context->producers_complete && !chains_.empty();
    if (context) output_allocations_ = context->output_allocations;
    for (const auto& chain : chains_) {
        for (const auto [pc, bytes] : {std::pair{chain.parent_pc, chain.parent_bytes},
                                      std::pair{chain.child_pc, chain.child_bytes}}) {
            const auto [it, inserted] = widths_.emplace(pc, bytes);
            if (!inserted && it->second != bytes) allowed_ = false;
        }
    }
    if (allowed_) {
        if (!borrowed_lease) owned_lease_ = std::make_unique<GuestMappingLease>();
        lease_ = borrowed_lease ? borrowed_lease : owned_lease_.get();
    }
}

GraphicsNestedWideReader::GraphicsNestedWideReader(
    std::vector<RawNestedWideChain> chains, const GraphicsRawSnapshotContext* context,
    uint64_t source_address, std::shared_ptr<const std::vector<uint32_t>> source,
    uint64_t source_submit, uint64_t command_order, const GuestMappingLease* borrowed_lease)
    : GraphicsNestedWideReader(std::move(chains), context, borrowed_lease) {
    checked_order_ = true;
    if (context) read_point_ = context->ordered_read_point;
    source_ = std::move(source);
    source_address_ = source_address;
    source_submit_ = source_submit;
    command_order_ = command_order;
    allowed_ =
        allowed_ && ordered_source_current() && read_point_->owns_chains(source_address_, chains_);
}

bool GraphicsNestedWideReader::ordered_source_current() const {
    return !checked_order_ || (read_point_ && read_point_->valid_for(source_submit_, command_order_,
                                                                     source_address_, source_));
}

bool GraphicsNestedWideReader::probe(FoldProbe kind, uint32_t pc, uint64_t address,
                                     uint32_t bytes) {
    const auto width = widths_.find(pc);
    if (width == widths_.end()) return guest_readable(address, bytes);
    // Raw pointers may not be repaired by Base48/Base40 fallback or a readable host VMA.
    if (!allowed_ || !ordered_source_current() || !lease_ || kind != FoldProbe::Raw ||
        bytes != width->second || address <= 0x10000u || (address & 3u) ||
        address > UINT64_MAX - bytes)
        return false;
    const auto prior = observations_.find(pc);
    if (prior != observations_.end())
        return prior->second.address == address && prior->second.bytes->size() == bytes;
    for (const auto& chain : chains_) {
        if (chain.child_pc != pc) continue;
        const auto parent = observations_.find(chain.parent_pc);
        if (parent == observations_.end() || address < chain.child_offset) return false;
        uint64_t pointer = 0;
        std::memcpy(&pointer, parent->second.bytes->data(), sizeof(pointer));
        if (pointer != address - chain.child_offset) return false;
    }
    if (!guest_memory_direct_range_fault_safe(*lease_, address, bytes) ||
        guest_memory_direct_allocation_relation(*lease_, address, bytes, address, bytes) !=
            GuestMemoryTopologyRelation::Overlap ||
        !guest_readable(address, bytes) ||
        !graphics_raw_source_is_guest_current(*lease_, address, bytes)) return false;
    for (const auto [output, physical_bytes] : output_allocations_)
        if (output && guest_memory_direct_allocation_relation(*lease_, address, bytes,
                output, physical_bytes) != GuestMemoryTopologyRelation::Disjoint) return false;
    auto owner = std::make_shared<std::vector<uint8_t>>(bytes);
    // The pre/post permission checks detect stale publication, not overlapping byte writes.
    // The existing submitted-input stability contract still applies to this actual copy.
    std::memcpy(owner->data(), reinterpret_cast<const void*>(uintptr_t(address)), bytes);
    if (!ordered_source_current()) return false;
    observations_.emplace(pc, Observation{address, std::move(owner)});
    return true;
}

uint32_t GraphicsNestedWideReader::word(uint32_t pc, uint64_t address) {
    if (!owns_raw_wide(pc)) return *reinterpret_cast<const uint32_t*>(uintptr_t(address));
    const auto it = observations_.find(pc);
    if (!ordered_source_current() || it == observations_.end() || address < it->second.address ||
        address - it->second.address > it->second.bytes->size() - sizeof(uint32_t))
        throw std::runtime_error("nested raw fold word lacks exact observation");
    uint32_t value = 0;
    std::memcpy(&value, it->second.bytes->data() + address - it->second.address, sizeof(value));
    return value;
}

void GraphicsNestedWideReader::prefix(uint32_t pc, uint64_t address, void* destination,
                                      uint32_t bytes) {
    if (!owns_raw_wide(pc)) {
        std::memcpy(destination, reinterpret_cast<const void*>(uintptr_t(address)), bytes);
        return;
    }
    const auto it = observations_.find(pc);
    if (!ordered_source_current() || it == observations_.end() || address != it->second.address ||
        bytes != it->second.bytes->size())
        throw std::runtime_error("nested raw fold prefix lacks exact observation");
    std::memcpy(destination, it->second.bytes->data(), bytes);
}

bool GraphicsNestedWideReader::publish(ShaderResourceTable& table) const {
    table.owned_nested_snapshot_requirements.assign(widths_.begin(), widths_.end());
    if (!allowed_ || !ordered_source_current() || observations_.size() != widths_.size())
        return false;
    for (const auto& chain : chains_) {
        const auto& parent = observations_.at(chain.parent_pc);
        const auto& child = observations_.at(chain.child_pc);
        uint64_t pointer = 0;
        std::memcpy(&pointer, parent.bytes->data(), sizeof(pointer));
        if (child.address < chain.child_offset || pointer != child.address - chain.child_offset)
            return false;
    }
    // Existing exact-PC resources may only be the identical owned parent from the same fold.
    // A descriptor, duplicate, or conflicting binding cannot be replaced by a later valid owner.
    for (const auto& [pc, observation] : observations_) {
        size_t matches = 0;
        for (const auto& resource : table.resources) {
            if (resource.fetch_pc != pc) continue;
            if (++matches > 1 || !valid_owned_raw_snapshot_resource(resource, widths_.at(pc)) ||
                resource.gpu_addr != observation.address ||
                std::memcmp(resource.host_data, observation.bytes->data(), widths_.at(pc)) != 0)
                return false;
        }
    }
    for (const auto& [pc, observation] : observations_) {
        ShaderResource resource;
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = DataFormat::Uint32;
        resource.num_components = 1;
        resource.gpu_addr = observation.address;
        resource.size = widths_.at(pc);
        resource.fetch_pc = pc;
        resource.owned_nested_snapshot_bytes = resource.size;
        resource.host_data = observation.bytes->data();
        resource.host_data_size = observation.bytes->size();
        const auto existing = std::find_if(table.resources.begin(), table.resources.end(),
            [pc](const ShaderResource& candidate) { return candidate.fetch_pc == pc; });
        if (existing == table.resources.end()) table.resources.push_back(resource);
        else {
            resource.owned_raw_snapshot_bytes = existing->owned_raw_snapshot_bytes;
            *existing = resource;
        }
        table.owned_host_data.push_back(observation.bytes);
    }
    return true;
}

} // namespace prosper::gpu
