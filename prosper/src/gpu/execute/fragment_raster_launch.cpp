#include "gpu/execute/fragment_raster_launch.hpp"
#include "gpu/execute/fragment_raster_contract.hpp"
#include "gpu/execute/fragment_raster_program.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include <map>
#include <tuple>

namespace prosper::gpu {
namespace {
template <class T>
bool same_owner(const std::shared_ptr<const T>& a, const std::shared_ptr<const T>& b) {
    return a && b && a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
}
bool same_parameters(const FragmentInterpolationLayout& a, const FragmentInterpolationLayout& b) {
    return a.valid == b.valid && a.requires_geometry == b.requires_geometry &&
           a.attribute_mask == b.attribute_mask && a.smooth_mask == b.smooth_mask &&
           a.passthrough_mask == b.passthrough_mask && a.flat_mask == b.flat_mask &&
           a.parameter_locations == b.parameter_locations &&
           a.system_locations == b.system_locations;
}
bool actual_launch(const GpuState& state, const RasterLaunchFacts& launch) {
    namespace P = prosper::agc::Pm4;
    struct Word {
        uint32_t reg;
        bool present;
        uint32_t value;
    };
    const Word words[]{
        {P::SPI_PS_IN_CONTROL, launch.ps_in_control_available, launch.ps_in_control},
        {P::SPI_BARYC_CNTL, launch.baryc_cntl_available, launch.baryc_cntl},
        {P::SPI_PS_INPUT_ENA, launch.input_ena_available, launch.input_ena},
        {P::SPI_PS_INPUT_ADDR, launch.input_addr_available, launch.input_addr},
        {P::PA_SC_SHADER_CONTROL, launch.sc_shader_control_available, launch.sc_shader_control},
        {P::PA_SC_MODE_CNTL_0, launch.sc_mode_cntl_0_available, launch.sc_mode_cntl_0},
        {P::PA_SC_MODE_CNTL_1, launch.sc_mode_cntl_1_available, launch.sc_mode_cntl_1},
        {P::PA_SC_AA_CONFIG, launch.sc_aa_config_available, launch.sc_aa_config},
        {P::DB_SHADER_CONTROL, launch.db_shader_control_available, launch.db_shader_control}};
    const auto matches = [&](uint32_t reg, bool present, uint32_t value) {
        const auto observed = state.cx.find(reg);
        return present == (observed != state.cx.end()) && (!present || observed->second == value);
    };
    for (const auto& word : words)
        if (!matches(word.reg, word.present, word.value)) return false;
    // Literal register window order mirrors the retained raw schema, including all four pixel
    // sample tables. A caller cannot substitute another draw's programmed words or knownness.
    constexpr uint32_t coverage_registers[]{0x200, 0x003, 0x004, 0x201, 0x2f9, 0x313, 0x30e,
                                            0x30f, 0x2fe, 0x2ff, 0x300, 0x301, 0x302, 0x303,
                                            0x304, 0x305, 0x306, 0x307, 0x308, 0x309, 0x30a,
                                            0x30b, 0x30c, 0x30d, 0x2f5, 0x2f6, 0x000};
    static_assert(std::size(coverage_registers) == RasterCoverageFacts::count);
    for (uint32_t index = 0; index < std::size(coverage_registers); ++index)
        if (!matches(coverage_registers[index],
                     bool(launch.coverage.available & (uint32_t{1} << index)),
                     launch.coverage.words[index]))
            return false;
    return true;
}
} // namespace

FragmentRasterLaunchSource::FragmentRasterLaunchSource(const RasterQuadInputs& in,
                                                       bool pending_original_proved)
    : raw_(in.raw_code), vertex_(in.source_vs), geometry_(in.source_gs), fragment_(in.source_fs),
      requirements_(in.vgpr_requirements), launch_(in.launch), entry_(in.entry),
      parameters_(in.interpolation), pixel_(in.pixel_inputs), system_(in.system_inputs),
      transport_(in.float_transport), float_mode_(in.float_mode), float_flags_(in.float_flags),
      rsrc1_(in.launch_rsrc1), has_pixel_(in.has_pixel_inputs), has_system_(in.has_system_inputs),
      generated_geometry_(in.generated_interpolation_geometry),
      owned_wave_pending_(in.owned_wave_pending),
      pending_original_proved_(pending_original_proved) {}

std::shared_ptr<const std::vector<uint32_t>> FragmentRasterLaunchSource::selected_empty_words() {
    static const auto words = std::make_shared<const std::vector<uint32_t>>();
    return words;
}

bool fragment_raster_pending_original(const GpuState& state, const RasterLaunchFacts& launch,
                                      const std::shared_ptr<const ShaderCodeAnalysis>& analysis,
                                      PixelSystemInputMapping system) {
    namespace P = prosper::agc::Pm4;
    if (!analysis || fragment_raster_workitem_gap(launch) || !launch.ps_in_control_available ||
        (launch.ps_in_control & (1u << 15)) || !launch.input_ena_available ||
        !launch.input_addr_available || system.ena != launch.input_ena ||
        system.addr != launch.input_addr)
        return false;
    const auto rsrc2 = state.sh.find(P::SPI_SHADER_PGM_RSRC2_PS);
    if (rsrc2 == state.sh.end()) return false;
    const uint32_t count = ((rsrc2->second >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT) &
                            P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MASK) |
                           (((rsrc2->second >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_SHIFT) &
                             P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_MASK)
                            << 5);
    if (count > 32) return false;
    uint32_t presence = 0;
    for (uint32_t reg = 0; reg < count; ++reg)
        if (state.sh.contains(P::SPI_SHADER_USER_DATA_PS_0 + reg)) presence |= 1u << reg;
    const auto raw = shader_analysis_owned_words(analysis);
    if (!raw || raw->empty() || raw->size() > 4096) return false;
    using Key = std::tuple<const std::vector<uint32_t>*, uint32_t, uint32_t, uint32_t>;
    struct Entry {
        std::weak_ptr<const std::vector<uint32_t>> generation;
        bool proved;
    };
    static thread_local std::map<Key, Entry> cache;
    const Key key{raw.get(), system.ena, system.addr, presence};
    if (const auto found = cache.find(key); found != cache.end()) {
        if (!found->second.generation.expired() && !found->second.generation.owner_before(raw) &&
            !raw.owner_before(found->second.generation))
            return found->second.proved;
        cache.erase(found);
    }
    std::erase_if(cache, [](const auto& entry) { return entry.second.generation.expired(); });
    std::vector<Rdna2Inst> original;
    rdna2_walk(raw->data(), raw->size(), original);
    const bool proved = fragment_raster_program(original, uint32_t(raw->size()), system, presence)
                            .rejection.empty();
    cache.emplace(key,
                  Entry{raw, proved}); // weak source tracking never self-retains a generation
    return proved;
}

bool FragmentRasterLaunchSource::matches(const RasterQuadInputs& in) const {
    return in.raw_matches_producing_source && same_owner(raw_, in.raw_code) &&
           same_owner(vertex_, in.source_vs) && same_owner(geometry_, in.source_gs) &&
           same_owner(fragment_, in.source_fs) && same_owner(requirements_, in.vgpr_requirements) &&
           in.launch == launch_ && in.entry == entry_ &&
           same_parameters(in.interpolation, parameters_) && in.pixel_inputs == pixel_ &&
           in.system_inputs == system_ && in.has_pixel_inputs == has_pixel_ &&
           in.has_system_inputs == has_system_ &&
           in.generated_interpolation_geometry == generated_geometry_ &&
           in.owned_wave_pending == owned_wave_pending_ && in.float_transport == transport_ &&
           in.float_mode == float_mode_ && in.float_flags == float_flags_ &&
           in.launch_rsrc1 == rsrc1_;
}

void FragmentRasterLaunchSource::bind(const GpuState& state, const RasterLaunchFacts& launch,
                                      std::shared_ptr<const ShaderCodeAnalysis> analysis,
                                      RasterQuadInputs& in) {
    in.launch_source.reset();
    in.launch = launch;
    in.raw_code = shader_analysis_owned_words(analysis);
    in.vgpr_requirements = shader_analysis_packet_vgpr_requirements(analysis);
    in.entry = {};
    const auto lo = state.sh.find(prosper::agc::Pm4::SPI_SHADER_PGM_LO_PS);
    const auto hi = state.sh.find(prosper::agc::Pm4::SPI_SHADER_PGM_HI_PS);
    if ((lo == state.sh.end() || !lo->second) && (hi == state.sh.end() || !(hi->second & 0xffu)))
        return; // preserve the realizer's no-PS observation, including stale USER_DATA storage
    in.entry.observed = true;
    for (uint32_t index = 0; index < in.entry.user_data.size(); ++index) {
        const auto word = state.sh.find(prosper::agc::Pm4::SPI_SHADER_USER_DATA_PS_0 + index);
        if (word != state.sh.end()) {
            in.entry.user_data_available |= uint32_t{1} << index;
            in.entry.user_data[index] = word->second;
        }
    }
    const auto rsrc2 = state.sh.find(prosper::agc::Pm4::SPI_SHADER_PGM_RSRC2_PS);
    if (rsrc2 != state.sh.end()) {
        in.entry.rsrc2_available = true;
        in.entry.rsrc2 = rsrc2->second;
    }
    if (!analysis || !in.raw_matches_producing_source || !in.raw_code || in.raw_code->empty() ||
        !in.vgpr_requirements || in.vgpr_requirements->source_words != in.raw_code.get() ||
        !in.source_vs || in.source_vs->empty() || !in.source_gs || !in.source_fs ||
        !in.entry.canonical() || !in.launch.canonical() || !actual_launch(state, in.launch))
        return;
    const bool pending = in.owned_wave_pending && in.source_fs->empty();
    const bool pending_proved =
        pending && fragment_raster_pending_original(state, launch, analysis, in.system_inputs);
    if (pending && !pending_proved) return;
    in.launch_source = std::shared_ptr<const FragmentRasterLaunchSource>(
        new FragmentRasterLaunchSource(in, pending_proved));
}
} // namespace prosper::gpu
