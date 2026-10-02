// Project-owned SOURCE shape and transactional host-wire controls, no device or guest data.
// Shape checks do not replace strict spirv-val or the actual registered DrawItem execution case.
#include "gpu/recompiler/raster_quad_collector.hpp"
#include "gpu/state/render_state.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

using namespace prosper::gpu;
static unsigned checks = 0, failures = 0;
static void check(bool ok, const char* name) {
    ++checks; failures += !ok;
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", name);
}
static void dump(const std::filesystem::path& directory, const char* name,
                 const std::vector<uint32_t>& words) {
    if (directory.empty()) return;
    std::ofstream file(directory / name, std::ios::binary);
    file.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(uint32_t));
    check(bool(file), "positive SOURCE dump written");
}
static RasterQuadInputs inputs() {
    RasterQuadInputs in;
    // V_READLANE_B32 s20,v0,40; V_MOV_B32 v4,s20; EXP MRT0(v4); END.
    const std::vector<uint32_t> raw{0xd7600014u, 256u | (168u << 9),
        0x7e080214u, 0xf800180fu, 0x04040404u, 0xbf810000u};
    in.raw_code = std::make_shared<const std::vector<uint32_t>>(raw);
    in.raw_matches_producing_source = true;
    in.system_inputs = {0x300u, 0x300u}; in.has_system_inputs = true;
    in.launch.input_ena_available = in.launch.input_addr_available = true;
    in.launch.input_ena = in.launch.input_addr = 0x300u;
    in.float_transport = {FloatTransportProfile::ExplicitNonFinite32};
    in.interpolation = fragment_interpolation_layout(raw.data(), raw.size(), &in.system_inputs);
    in.source_fs = std::make_shared<const std::vector<uint32_t>>(recompile_fragment(raw.data(), raw.size(),
        nullptr, &in.system_inputs, UINT32_MAX, &in.interpolation, false, {}, {}, nullptr, in.float_transport));
    return in;
}
static bool quad_operand_types_ok(const std::vector<uint32_t>& words) {
    std::set<uint32_t> unsigned32;
    std::map<uint32_t, uint32_t> types;
    std::map<uint32_t, uint32_t> constants;
    uint32_t count = 0;
    for (size_t p = 5; p < words.size();) {
        const uint32_t n = words[p] >> 16, op = words[p] & 65535;
        if (!n || p + n > words.size()) return false;
        if (op == 21 && n == 4 && words[p + 2] == 32 && words[p + 3] == 0)
            unsigned32.insert(words[p + 1]);
        if ((op == 43 || op == 61 || op == 81 || op == 124 || op == 169 || op == 365) && n >= 3)
            types[words[p + 2]] = words[p + 1];
        if (op == 43 && n == 4) constants[words[p + 2]] = words[p + 3];
        if (op == 365) {
            ++count;
            if (n != 6 || !unsigned32.count(words[p + 1]) ||
                !types.count(words[p + 4]) || !unsigned32.count(types.at(words[p + 4])) ||
                !types.count(words[p + 3]) || !unsigned32.count(types.at(words[p + 3])) ||
                !constants.count(words[p + 3]) || constants.at(words[p + 3]) != 3 ||
                !types.count(words[p + 5]) || !unsigned32.count(types.at(words[p + 5])) ||
                !constants.count(words[p + 5]) || constants.at(words[p + 5]) >= 4) return false;
        }
        p += n;
    }
    return count != 0;
}
static bool retry_loop_scopes_ok(const std::vector<uint32_t>& words) {
    std::set<uint32_t> continues, selections;
    for (size_t p = 5; p < words.size();) {
        const uint32_t n = words[p] >> 16, op = words[p] & 65535;
        if (!n || p + n > words.size()) return false;
        if (op == 246) {
            if (n != 4 || words[p + 1] == words[p + 2]) return false;
            continues.insert(words[p + 2]);
        }
        if (op == 247) {
            if (n != 3) return false;
            selections.insert(words[p + 1]);
        }
        p += n;
    }
    if (continues.size() != 1) return false; // this fixture has exactly one bounded CAS loop
    for (uint32_t target : continues) if (selections.count(target)) return false;
    return true;
}
static std::vector<uint32_t> former_cas_selection(const std::vector<uint32_t>& words) {
    // Restore the actual former invalid CFG in emitted SOURCE, not a call-shaped assertion:
    // CAS body selects loop-merge/continue while naming the continue as its own selection merge.
    uint32_t block = 0, header = 0, cont = 0, done = 0, attempt = 0, won = 0;
    for (size_t p = 5; p < words.size();) {
        const uint32_t n = words[p] >> 16, op = words[p] & 65535;
        if (!n || p + n > words.size()) return {};
        if (op == 248 && n == 2) block = words[p + 1];
        if (op == 246 && n == 4) { header = block; done = words[p + 1]; cont = words[p + 2]; }
        if (op == 230) attempt = block;
        if (op == 170 && n == 5 && block == attempt && attempt) won = words[p + 2];
        p += n;
    }
    if (!header || !cont || !done || !attempt || !won) return {};
    std::vector<uint32_t> out(words.begin(),words.begin() + 5);
    block = 0;
    for (size_t p = 5; p < words.size();) {
        const uint32_t n = words[p] >> 16, op = words[p] & 65535;
        if (op == 248) block = words[p + 1];
        if (block == attempt && op == 249 && n == 2) {
            out.insert(out.end(),{(3u << 16) | 247u,cont,0,(4u << 16) | 250u,won,done,cont});
        } else if (block == cont && op == 250 && n == 4) {
            out.insert(out.end(),{(2u << 16) | 249u,header});
        } else {
            const auto start = out.size();
            out.insert(out.end(),words.begin() + p,words.begin() + p + n);
            if (block == done && op == 245 && n == 7 && out[start + 6] == cont)
                out[start + 6] = attempt;
        }
        p += n;
    }
    return out;
}
static void source_shape(const std::vector<uint32_t>& words, const RasterQuadCollector& contract) {
    bool before_control = true, malformed = words.size() < 5;
    uint32_t broadcasts = 0, cas = 0, load = 0, overflow = 0, adds = 0, outputs = 0;
    std::array<uint32_t, 4> index_counts{};
    std::map<uint32_t, uint32_t> constants;
    for (size_t p = 5; p < words.size();) {
        const uint32_t n = words[p] >> 16, op = words[p] & 0xffff;
        if (!n || p + n > words.size()) { malformed = true; break; }
        if (op == 43 && n == 4) constants[words[p + 2]] = words[p + 3];
        if (op == 365) {
            ++broadcasts;
            check(before_control && n == 6, "each actual QuadBroadcast precedes all election/control");
            const auto index = n == 6 ? constants.find(words[p + 5]) : constants.end();
            check(index != constants.end() && index->second < 4, "quad source index is literal0..3");
            if (index != constants.end() && index->second < 4) ++index_counts[index->second];
        }
        if (op == 246 || op == 247 || op == 249 || op == 250 || op == 251) before_control = false;
        if (op == 230) ++cas;
        if (op == 227) ++load;
        if (op == 241) ++overflow;
        if (op == 234) ++adds;
        if (op == 59 && n >= 4 && words[p + 3] == 3) ++outputs;
        p += n;
    }
    check(!malformed && broadcasts == contract.lane_words * 4, "all four actual quad sources are retained");
    for (uint32_t count : index_counts) check(count == contract.lane_words, "every word is gathered from each source");
    check(cas == 1 && load == 1 && overflow == 1 && adds == 0, "bounded append uses saturating CAS, never wrapping Add");
    check(outputs == 0, "collector has no attachment, depth or coverage Output variables");
    check(fragment_spirv_required_subgroup_size(words) == 0, "collector grants no exact native64 contract");
    check(quad_operand_types_ok(words), "actual scope/index/value/result domains are all exact U32");
    check(retry_loop_scopes_ok(words), "bounded CAS loop never aliases a selection merge with its continue construct");
}
static std::vector<uint32_t> wire(const RasterQuadCollector& c) {
    std::vector<uint32_t> out(kRasterQuadBufferHeaderWords + c.record_words * c.max_quads, 0);
    out[0] = 2; out[2] = c.record_words; out[3] = kRasterQuadMagic;
    for (uint32_t q = 0; q < 2; ++q) for (uint32_t lane = 0; lane < 4; ++lane) {
        auto* w = out.data() + kRasterQuadBufferHeaderWords + q * c.record_words + lane * c.lane_words;
        w[0] = lane != q; w[1] = !w[0]; w[2] = w[0] ? 0xdeadbeefu : 1u;
        w[3] = 1; w[4] = q; // overlapping primitives must remain distinct
        w[5] = std::bit_cast<uint32_t>(0.5f + float(lane & 1));
        w[6] = std::bit_cast<uint32_t>(0.5f + float(lane >> 1));
        w[7] = 0x80000000u; w[8] = 0x3f800000u;
    }
    return out;
}
int main(int argc, char** argv) {
    std::filesystem::path directory;
    if (argc == 3 && std::strcmp(argv[1], "--dump-directory") == 0) {
        directory = argv[2]; std::filesystem::create_directories(directory);
    } else if (argc != 1) return 2;
    auto in = inputs(); RasterQuadCollector c;
    const auto source = build_raster_quad_collector(in, 2, c);
    check(in.source_fs && !in.source_fs->empty() && !source.empty() && c.rejection.empty(), "actual owned guest SOURCE produces collector");
    if (source.empty()) return 1;
    source_shape(source, c); dump(directory, "raster_quad_source.spv", source);
    const auto invalid_loop = former_cas_selection(source);
    check(!invalid_loop.empty() && !retry_loop_scopes_ok(invalid_loop),
        "actual former invalid CAS/selection CFG fails the loop-scope observer");
    auto wrong_type = source;
    uint32_t signed_type = 0;
    for (size_t p = 5; p < wrong_type.size();) {
        const uint32_t n = wrong_type[p] >> 16, op = wrong_type[p] & 65535;
        if (!n || p + n > wrong_type.size()) break;
        if (op == 21 && n == 4 && wrong_type[p + 2] == 32 && wrong_type[p + 3] == 1)
            signed_type = wrong_type[p + 1];
        if (op == 365 && signed_type) { wrong_type[p + 1] = signed_type; break; }
        p += n;
    }
    check(signed_type && !quad_operand_types_ok(wrong_type), "actual changed result type cannot fool SOURCE shape oracle");
    dump(directory, "raster_quad_guest_source.spv", *in.source_fs);
    // Actual project-owned explicit-parameter interpolation program, also used by strict corpus.
    const uint32_t parameter_raw[]{0xc80e0000u,0xc8120001u,0xc8160002u,
        0xd54b0003u,0x04160103u,0xd54b0003u,0x040e0304u,0x7e080280u,0x7e0a0280u,
        0x7e0c02f2u,0xf800000fu,0x06050403u,0xbf810000u};
    PixelSystemInputMapping center{2,2}; PixelInputMapping pixel;
    pixel.valid_mask = 1;
    const auto layout = fragment_interpolation_layout(parameter_raw,std::size(parameter_raw),&center,&pixel);
    const auto native_geometry = recompile_interpolation_geometry(layout,false,false,in.float_transport);
    check(layout.valid && layout.requires_geometry && !native_geometry.empty() && native_geometry ==
        recompile_interpolation_geometry(layout,false,false,in.float_transport,false),
        "default GS API selects unchanged no-PrimitiveId-publisher form");
    const auto id_geometry = recompile_interpolation_geometry(layout,false,false,in.float_transport,true);
    std::map<uint32_t,uint32_t> builtin, storage;
    uint32_t id_input = 0, id_output = 0, id_value = 0, stores = 0, emits = 0;
    bool id_ready = false, all_emits_owned = true;
    for (size_t p = 5; p < id_geometry.size();) {
        const auto n = id_geometry[p] >> 16, op = id_geometry[p] & 65535;
        if (!n || p + n > id_geometry.size()) { all_emits_owned = false; break; }
        if (op == 71 && n == 4 && id_geometry[p + 2] == 11) builtin[id_geometry[p + 1]] = id_geometry[p + 3];
        if (op == 59 && n >= 4) storage[id_geometry[p + 2]] = id_geometry[p + 3];
        for (const auto& [variable,kind] : storage) if (builtin.count(variable) && builtin.at(variable) == 7) {
            if (kind == 1) id_input = variable;
            if (kind == 3) id_output = variable;
        }
        if (op == 61 && n == 4 && id_geometry[p + 3] == id_input) id_value = id_geometry[p + 2];
        if (op == 62 && n == 3 && id_geometry[p + 1] == id_output) {
            id_ready = id_geometry[p + 2] == id_value && id_value != 0; ++stores;
        }
        if (op == 218) { ++emits; all_emits_owned &= id_ready; id_ready = false; }
        p += n;
    }
    check(id_input && id_output && stores == 3 && emits == 3 && all_emits_owned,
        "actual generated GS publishes the same primitive to every emitted vertex");
    check(recompile_interpolation_geometry(layout,false,true,in.float_transport,true).empty(),
        "PrimitiveId publisher never guesses RectList primitive identity");
    dump(directory,"raster_quad_geometry_native.spv",native_geometry);
    dump(directory,"raster_quad_geometry_ids.spv",id_geometry);
    auto parameter = in;
    parameter.raw_code = std::make_shared<const std::vector<uint32_t>>(parameter_raw,parameter_raw+std::size(parameter_raw));
    parameter.system_inputs = center; parameter.pixel_inputs = pixel; parameter.has_pixel_inputs = true;
    parameter.interpolation = layout; parameter.generated_interpolation_geometry = true;
    parameter.launch.input_ena = parameter.launch.input_addr = 2;
    parameter.source_fs = std::make_shared<const std::vector<uint32_t>>(recompile_fragment(
        parameter_raw,std::size(parameter_raw),nullptr,&center,UINT32_MAX,&layout,false,{}, {},nullptr,in.float_transport));
    RasterQuadCollector parameter_contract;
    const auto parameter_collector = build_raster_quad_collector(parameter,2,parameter_contract);
    check(!parameter_collector.empty() && parameter_contract.rejection.empty() && !parameter_contract.fields.empty(),
        "actual parameter SOURCE uses explicit routed input fields");
    if (!parameter_collector.empty()) source_shape(parameter_collector,parameter_contract);
    for (const auto& field : parameter_contract.fields)
        check(!field.guest_initialization_proved, "interface presence cannot prove guest register initialization");
    dump(directory,"raster_quad_parameter_source.spv",parameter_collector);
    dump(directory,"raster_quad_parameter_guest.spv",*parameter.source_fs);
    auto reject = [&](RasterQuadInputs bad, uint32_t maximum, const char* reason) {
        RasterQuadCollector got;
        check(build_raster_quad_collector(bad, maximum, got).empty() && got.rejection == reason, reason);
    };
    auto bad = in; bad.raw_code.reset(); reject(bad, 2, "quad-collector-producing-source-unavailable");
    bad = in; bad.raw_matches_producing_source = false;
    reject(bad, 2, "quad-collector-producing-source-replaced");
    bad = in; bad.interpolation.valid = false; reject(bad, 2, "quad-collector-interpolation-layout-unavailable");
    bad = in; bad.float_transport = {}; reject(bad, 2, "quad-collector-explicit-transport-unavailable");
    reject(in, 0, "quad-collector-budget-invalid"); reject(in, 4097, "quad-collector-budget-invalid");
    bad = in; bad.interpolation.requires_geometry = true;
    reject(bad, 2, "quad-collector-parameter-producer-unavailable");
    bad.generated_interpolation_geometry = true; bad.interpolation.system_locations[0] = 1;
    reject(bad, 2, "quad-collector-sample-interpolation-unimplemented");
    bad = in; bad.interpolation.attribute_mask = 1;
    reject(bad, 2, "quad-collector-interpolant-routing-unavailable");

    const auto original = wire(c);
    std::vector<std::vector<uint32_t>> quads;
    check(decode_raster_quad_records(c, original.data(), original.size(), 2, quads).empty() && quads.size() == 2,
        "exact-capacity complete overlapping primitive records remain distinct");
    check(quads.size() == 2 && quads[0][c.lane_words + 2] == 0xdeadbeefu &&
        quads[0][c.lane_words + 1] == 0, "helper mask stays observed but unavailable, never synthesized as coverage");
    auto reject_wire = [&](std::vector<uint32_t> raw, size_t length, uint32_t primitives, const char* reason) {
        quads = {{0xa5a5a5a5u}};
        check(decode_raster_quad_records(c, raw.data(), length, primitives, quads) == reason && quads.empty(), reason);
    };
    auto damaged = original; damaged[1] = 1;
    reject_wire(damaged, damaged.size(), 2, "quad-collector-output-overflow");
    damaged = original; damaged[0] = 3;
    reject_wire(damaged, damaged.size(), 2, "quad-collector-output-header-malformed");
    reject_wire(original, original.size() - 1, 2, "quad-collector-output-header-malformed");
    reject_wire(original, original.size(), 1, "quad-collector-output-provenance-malformed");
    damaged = original; damaged[4 + c.record_words + c.lane_words + 5] ^= 0x00800000u;
    reject_wire(damaged, damaged.size(), 2, "quad-collector-output-topology-malformed");
    damaged = original; damaged[4 + c.lane_words + 1] = 1;
    reject_wire(damaged, damaged.size(), 2, "quad-collector-output-provenance-malformed");
    damaged = original; damaged[4 + 2] = 0;
    reject_wire(damaged, damaged.size(), 2, "quad-collector-output-provenance-malformed");
    damaged = original;
    for (uint32_t lane = 0; lane < 4; ++lane) damaged[4 + c.record_words + lane * c.lane_words + 4] = 0;
    reject_wire(damaged, damaged.size(), 2, "quad-collector-duplicate-quad-unproved");
    damaged = original;
    for (uint32_t lane = 0; lane < 4; ++lane) {
        damaged[4 + c.record_words + lane * c.lane_words] = 1;
        damaged[4 + c.record_words + lane * c.lane_words + 1] = 0;
    }
    reject_wire(damaged, damaged.size(), 2, "quad-collector-helper-only-publication");

    namespace P = prosper::agc::Pm4;
    GpuState state;
    const auto unknown = extract_render_state(state).ps_raster_launch;
    check(!unknown.ps_in_control_available && !unknown.baryc_cntl_available &&
        !unknown.input_ena_available && !unknown.input_addr_available, "absent launch is unknown, not mode0");
    state.cx[P::SPI_PS_IN_CONTROL] = 0; state.cx[P::SPI_BARYC_CNTL] = 0xa5a5f00du;
    state.cx[P::SPI_PS_INPUT_ENA] = 0; state.cx[P::SPI_PS_INPUT_ADDR] = 0xfedcba98u;
    const auto known = extract_render_state(state).ps_raster_launch;
    check(known.ps_in_control_available && known.ps_in_control == 0 && known.baryc_cntl_available &&
        known.baryc_cntl == 0xa5a5f00du && known.input_ena_available && known.input_ena == 0 &&
        known.input_addr_available && known.input_addr == 0xfedcba98u, "all full raw launch bits and known zero survive");
    std::printf("[raster-quad-contract] checks=%u failures=%u; no GPU or guest raster admission\n", checks, failures);
    return failures ? 1 : 0;
}
