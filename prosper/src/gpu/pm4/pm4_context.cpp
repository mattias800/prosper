#include "gpu/pm4/pm4_context.hpp"
#include "gpu/pm4/command_processor.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>

namespace prosper::gpu {
bool guest_readable(uint64_t address, uint32_t bytes);
bool guest_writable(uint64_t address, uint32_t bytes);
void notify_guest_gpu_write(uint64_t address, uint64_t bytes);
void set_guest_gpu_write_origin(const char* origin);
void record_loaded_shader_register(GpuState& state, const Pm4Command& command, uint32_t offset);

namespace {
constexpr uint32_t kUpdate = 1u << 31;
constexpr std::array<uint32_t, 4> kSelectors{1u << 1, 1u << 15, 1u << 16, 1u << 24};

size_t bank(const Pm4Command& command) {
    if (command.reg_class == RegClass::Cx) return 0;
    if (command.reg_class == RegClass::Uc) return 1;
    return (command.header & 2u) ? 3 : 2;
}

bool refuse(const Pm4Command& command, GpuState& state, const char* reason) {
    static std::atomic<uint64_t> count{0};
    const uint64_t ordinal = count.fetch_add(1) + 1;
    if (ordinal <= 8 || (ordinal & (ordinal - 1)) == 0)
        std::fprintf(stderr, "[agc] PM4 context command REFUSED #%llu: op=0x%x reason=%s\n",
                     static_cast<unsigned long long>(ordinal), command.op, reason);
    state.dma_execution_rejected = true;
    return false;
}

bool address_range(uint64_t base, uint32_t offset, uint32_t count, uint64_t* address) {
    const uint64_t displacement = uint64_t(offset) * 4;
    const uint64_t bytes = uint64_t(count) * 4;
    if (!base || base > UINT64_MAX - displacement || base + displacement > UINT64_MAX - bytes)
        return false;
    *address = base + displacement;
    return true;
}
}  // namespace

void apply_context_control(const Pm4Command& command, GpuState& state) {
    // Published GFX10 fields and AMD's SI programming guide agree on the independent update
    // bits; each unselected dword leaves its existing enables intact. CONFIDENCE: HIGH.
    const uint32_t load = command.payload[0], shadow = command.payload[1];
    if (load & kUpdate) {
        state.context_control.load = load & ~kUpdate;
        state.context_control.load_known = true;
    }
    if (shadow & kUpdate) state.context_control.shadow = shadow & ~kUpdate;
}

bool apply_register_ranges(const Pm4Command& command, GpuState& state) {
    auto& controls = state.context_control;
    const size_t selected = bank(command);
    // No PS5 reset-enable contract is established. An explicit CONTEXT_CONTROL establishes the
    // ordinary load contract; never guess disabled/enabled from a default-constructed HLE state.
    if (!controls.load_known) return refuse(command, state, "load-enables-not-established");
    if (!(controls.load & kSelectors[selected])) return false;

    // An enabled count-zero LOAD still installs the shadow base. Offset/count pairs describe
    // consecutive register and memory ranges, both relative to that base. CONFIDENCE: HIGH on
    // published AMD behavior; MED on PS5 use outside the retained pair-load console fixture.
    uint64_t total = 0;
    for (uint32_t i = 0; i < command.reg_count; ++i) total += command.reg_data[i * 2 + 1];
    if (total > GpuState::kMaxRegsPerPacket)
        return refuse(command, state, "range-count-exceeds-supported-bound");
    if (total && (!state.dispatches.empty() || !state.draws.empty() || !state.dma_copies.empty() ||
                  state.dma_execution_rejected || last_fold_deferred()))
        return refuse(command, state, "range-producer-not-retired");

    std::vector<ShaderReg> registers;
    registers.reserve(total);
    for (uint32_t i = 0; i < command.reg_count; ++i) {
        const uint32_t offset = command.reg_data[i * 2], count = command.reg_data[i * 2 + 1];
        if (!count) continue;
        uint64_t address = 0;
        if (offset >= GpuState::kRegOffsetLimit || count > GpuState::kRegOffsetLimit - offset ||
            !address_range(command.regs_vaddr, offset, count, &address) ||
            !guest_readable(address, count * 4))
            return refuse(command, state, "range-address-unreadable");
        for (uint32_t n = 0; n < count; ++n) {
            const uint64_t word_address = address + uint64_t(n) * 4;
            auto value = pending_conditional_value(word_address);
            for (const auto& effect : state.ordered_memory_effects) {
                if (!value) break;
                value = overlay_conditional_memory_effect(effect, word_address, *value);
            }
            if (!value) return refuse(command, state, "range-pending-effect-unresolved");
            registers.push_back({offset + n, *value});
        }
    }
    controls.bases[selected] = command.regs_vaddr;
    auto& file = command.reg_class == RegClass::Cx   ? state.cx
                 : command.reg_class == RegClass::Uc ? state.uc
                                                     : state.sh;
    for (const auto& reg : registers) {
        file.set(reg.offset, reg.value);
        if (command.reg_class == RegClass::Sh)
            record_loaded_shader_register(state, command, reg.offset);
    }
    return !registers.empty();
}

void shadow_direct_registers(const Pm4Command& command, GpuState& state) {
    const size_t selected = bank(command);
    if (!(state.context_control.shadow & kSelectors[selected]) || !command.reg_data ||
        !command.reg_count || command.reg_count > GpuState::kMaxRegsPerPacket ||
        command.reg_offset >= GpuState::kRegOffsetLimit)
        return;
    const uint32_t count =
        std::min(command.reg_count, GpuState::kRegOffsetLimit - command.reg_offset);
    uint64_t address = 0;
    if (!address_range(state.context_control.bases[selected], command.reg_offset, count,
                       &address) ||
        !guest_writable(address, count * 4)) {
        refuse(command, state, "shadow-base-unwritable");
        return;
    }
    // Keep the shadow write beside earlier producers and completion aliases. The ordinary
    // LOAD path can consume owned fixed-write overlays without publishing labels prematurely.
    Pm4Command shadow = command;
    shadow.kind = Pm4Command::Kind::WriteData;
    shadow.wd_addr = address;
    shadow.wd_data = command.reg_data;
    shadow.wd_num = shadow.wd_declared_num = count;
    shadow.wd_valid = true;
    shadow.wd_shadow = true;
    state.ordered_memory_effects.emplace_back(shadow, state.command_order);
}

bool register_shadow_operand_valid(const Pm4Command& command) {
    return command.wd_valid && command.wd_data && command.wd_addr && !(command.wd_addr & 3u) &&
           command.wd_num && command.wd_num <= GpuState::kMaxRegsPerPacket &&
           command.wd_num == command.wd_declared_num &&
           guest_writable(command.wd_addr, command.wd_num * 4);
}

bool execute_register_shadow_write(const Pm4Command& command) {
    if (!register_shadow_operand_valid(command)) return false;
    // Shadowing is a GPU write of exactly the established register range. It is neither an
    // allocator label initialization nor an eight-byte stomp-catcher probe at a mapping edge.
    const uint32_t bytes = command.wd_num * 4;
    std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(command.wd_addr)), command.wd_data,
                bytes);
    set_guest_gpu_write_origin("SET_REG_SHADOW");
    notify_guest_gpu_write(command.wd_addr, bytes);
    set_guest_gpu_write_origin(nullptr);
    return true;
}
}   // namespace prosper::gpu
