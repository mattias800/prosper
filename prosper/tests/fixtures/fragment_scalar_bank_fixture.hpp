// Genuine AGC originals and direct guest backing for the shared scalar-bank producer/renderer.
// The descriptor is real PS USER_DATA; neither a folded table nor fabricated FS reflection
// grants bank permission. CPU users observe sealed ownership; GPU users use the same producer.
#pragma once
#include "fixtures/fragment_draw_fixture.hpp"
#include "gpu/execute/fragment_scalar_bank.hpp"
#include <cstring>

namespace prosper::test::scalar_bank {
namespace g = prosper::gpu;
namespace p = prosper::agc::Pm4;
inline constexpr uint64_t page = 0x10000;

inline std::vector<uint32_t> fragment_words(uint32_t offset = 0) {
    // Scalar memory executes even EXEC=0. A real LGKM drain precedes the SGPR consumers;
    // the later dominating EXEC=-1 separately proves full vector writers/EXP for this recipe.
    return {0xbefe0480u, 0xf4280500u, 0xfa000000u | offset, 0xbf8c0000u, 0xbefe04c1u, 0x7e000214u,
            0x7e020215u, 0x7e040216u, 0x7e060217u,          0xf800180fu, 0x03020100u, 0xbf810000u};
}

struct DirectRegion {
    uint64_t address = 0, physical = 0;
    bool allocated = false;
    DirectRegion() = default;
    DirectRegion(const DirectRegion&) = delete;
    DirectRegion& operator=(const DirectRegion&) = delete;
    ~DirectRegion() { close(); }
    bool create() {
        const auto allocate =
            prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
        const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
        if (!allocate || !map ||
            allocate(0, 0x200000000ull, page, page, 0, reinterpret_cast<uint64_t>(&physical)) != 0)
            return false;
        allocated = true;
        return map(reinterpret_cast<uint64_t>(&address), page, 3, 0, physical, page) == 0 &&
               address;
    }
    bool close() {
        bool ok = true;
        if (address) {
            const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
            ok = unmap && unmap(address, page, 0, 0, 0, 0) == 0;
            if (ok) address = 0;
        }
        if (allocated && !address) {
            const auto release =
                prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"));
            const bool released = release && release(physical, page, 0, 0, 0, 0) == 0;
            ok &= released;
            if (released) allocated = false;
        }
        return ok;
    }
};

inline ps_pull::Program* register_original(bool vertex, const std::vector<uint32_t>& words) {
    // AGC retains registered pointers until process exit. Heap owners survive the fixture and
    // sit above the SDK relative-pointer domain; source rewrite tests mutate them explicitly.
    static std::vector<std::unique_ptr<ps_pull::Program>> owners;
    auto owner = std::make_unique<ps_pull::Program>();
    if (reinterpret_cast<uint64_t>(owner.get()) <= UINT32_MAX ||
        !ps_pull::register_program(*owner, vertex, words))
        return nullptr;
    const auto result = owner.get();
    owners.push_back(std::move(owner));
    return result;
}

struct Scene {
    DirectRegion data, color;
    ps_pull::Program* vertex = nullptr;
    ps_pull::Program* fragment = nullptr;
    std::array<uint32_t, 4> descriptor{};
    bool create(const std::array<float, 4>& rgba = fragment_draw::color_a,
                const std::vector<uint32_t>& ps_words = fragment_words()) {
        prosper::register_builtin_hle();
        if (!data.create() || !color.create()) return false;
        vertex = register_original(true, fragment_draw::vertex_words());
        fragment = register_original(false, ps_words);
        if (!vertex || !fragment) return false;
        descriptor = {uint32_t(data.address), uint32_t(data.address >> 32u), uint32_t(page),
                      0x21000000u};   // genuine ignored format/control word, NOT synthetic raw zero
        write(rgba);
        return true;
    }
    void write(const std::array<float, 4>& rgba, uint32_t offset = 0) {
        std::memcpy(reinterpret_cast<void*>(data.address + offset), rgba.data(), sizeof(rgba));
    }
    g::GpuState state(bool wave32 = false) const {
        g::GpuState result;
        for (const auto* program : {vertex, fragment})
            for (const auto& reg : program->registers) result.sh[reg.offset] = reg.value;
        result.uc[p::VGT_PRIMITIVE_TYPE] = 4;
        result.cx[p::SPI_PS_IN_CONTROL] =
            wave32 ? (1u << p::SPI_PS_IN_CONTROL_PS_W32_EN_SHIFT) : 0u;
        result.cx[p::SPI_BARYC_CNTL] = result.cx[p::SPI_PS_INPUT_ENA] =
            result.cx[p::SPI_PS_INPUT_ADDR] = 0;
        result.sh[p::SPI_SHADER_PGM_RSRC1_PS] = fragment_draw::ieee_rsrc1;
        result.sh[p::SPI_SHADER_PGM_RSRC2_PS] = 4u << p::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT;
        for (uint32_t index = 0; index < descriptor.size(); ++index)
            result.sh[p::SPI_SHADER_USER_DATA_PS_0 + index] = descriptor[index];
        for (uint32_t reg :
             {p::VGT_SHADER_STAGES_EN, p::VGT_STRMOUT_CONFIG, p::VGT_STRMOUT_BUFFER_CONFIG,
              p::DB_DEPTH_CONTROL, p::DB_RENDER_CONTROL, p::DB_RENDER_OVERRIDE,
              p::DB_RENDER_OVERRIDE2, p::CB_COLOR0_CMASK, p::CB_COLOR0_CMASK_BASE_EXT,
              p::CB_COLOR0_FMASK, p::CB_COLOR0_FMASK_BASE_EXT, p::CB_COLOR0_DCC_BASE,
              p::CB_COLOR0_DCC_BASE_EXT})
            result.cx[reg] = 0;   // actual physical observations, never absent => hardware default
        result.cx[p::CB_COLOR_CONTROL] = p::CB_COLOR_CONTROL_MODE_NORMAL
                                         << p::CB_COLOR_CONTROL_MODE_SHIFT;
        result.cx[p::CB_TARGET_MASK] = result.cx[p::CB_SHADER_MASK] = 15;
        result.cx[p::CB_COLOR0_BASE] = uint32_t(color.address >> 8u);
        result.cx[p::CB_COLOR0_BASE_EXT] = uint32_t(color.address >> 40u);
        result.cx[p::CB_COLOR0_INFO] =
            (0xeu << p::CB_COLOR0_INFO_FORMAT_SHIFT) |
            (7u << p::CB_COLOR0_INFO_NUMBER_TYPE_SHIFT);   // real RGBA32_FLOAT linear target
        result.cx[p::CB_COLOR0_ATTRIB2] =
            ((fragment_draw::width - 1u) << p::CB_COLOR0_ATTRIB2_MIP0_WIDTH_SHIFT) |
            (fragment_draw::height - 1u);
        result.cx[p::CB_COLOR0_ATTRIB3] = 1u << p::CB_COLOR0_ATTRIB3_RESOURCE_TYPE_SHIFT;
        result.cx[p::CB_COLOR0_VIEW] = result.cx[p::CB_COLOR0_ATTRIB] = 0;
        g::GpuState::Draw draw;
        draw.index_count = 3;
        draw.instance_count = 1;
        draw.command_order = 100;
        result.draws.push_back(draw);
        return result;
    }
};
}   // namespace prosper::test::scalar_bank
