#include "gpu/pm4/pm4_memory.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "host/memory/guest_memory_topology.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>

namespace prosper::gpu {

bool guest_readable(uint64_t address, uint32_t bytes);
bool guest_writable(uint64_t address, uint32_t bytes);
void notify_guest_gpu_write(uint64_t address, uint64_t bytes);
void set_guest_gpu_write_origin(const char* origin);
bool dma_data_guest_destination(const Pm4Command& command);
bool dma_data_immediate_operand(const Pm4Command& command);

// The guest-memory span a command writes (0 bytes = writes nothing we track).
void pm4_memory_effect_span(const Pm4Command& c, uint64_t* addr, uint64_t* bytes) {
    using K = Pm4Command::Kind;
    switch (c.kind) {
        case K::ReleaseMem:
            *addr = c.rel_addr;
            *bytes = 8;
            break;
        case K::EventWrite:
            *addr = c.event_addr;
            *bytes = 8;
            break;
        case K::WriteData:
            *addr = c.wd_addr;
            *bytes = (uint64_t)c.wd_declared_num * 4;
            break;
        case K::DmaData:
            *addr = c.dd_dst;
            *bytes = c.dd_bytes;
            break;
        case K::AtomicMem:
            *addr = c.atomic_addr;
            *bytes = atomic_mem_bytes(c.atomic_op);
            break;
        default:
            *addr = 0;
            *bytes = 0;
            break;
    }
}

bool integer_atomic_mem_op(uint32_t op) {
    if (op > 0x7f) return false;
    const uint32_t operation = op & 0x1f;
    return operation == 7 || operation == 8 || (operation >= 15 && operation <= 25);
}

uint32_t atomic_mem_bytes(uint32_t op) {
    return (op & 0x20) ? 8u : 4u;
}

std::optional<uint64_t> atomic_mem_value(uint32_t op, uint64_t old_value, uint64_t source,
                                         uint64_t compare) {
    if (!integer_atomic_mem_op(op)) return std::nullopt;
    // TC_OP bit5 selects 64 bits; bit6 selects the no-return variant. Neither changes the RMW.
    // AMD's published GFX10 TC_OP enum defines the integer operations below. CONFIDENCE: HIGH.
    const uint64_t mask = atomic_mem_bytes(op) == 4 ? UINT32_MAX : UINT64_MAX;
    const uint64_t sign = atomic_mem_bytes(op) == 4 ? uint64_t{1} << 31 : uint64_t{1} << 63;
    old_value &= mask;
    source &= mask;
    compare &= mask;
    uint64_t value = old_value;
    switch (op & 0x1f) {
        case 7: value = source; break;
        case 8: value = old_value == compare ? source : old_value; break;
        case 15: value = old_value + source; break;
        case 16: value = old_value - source; break;
        case 17: value = (old_value ^ sign) < (source ^ sign) ? old_value : source; break;
        case 18: value = std::min(old_value, source); break;
        case 19: value = (old_value ^ sign) > (source ^ sign) ? old_value : source; break;
        case 20: value = std::max(old_value, source); break;
        case 21: value = old_value & source; break;
        case 22: value = old_value | source; break;
        case 23: value = old_value ^ source; break;
        case 24: value = old_value >= source ? 0 : old_value + 1; break;
        case 25: value = old_value == 0 || old_value > source ? source : old_value - 1; break;
    }
    return value & mask;
}

namespace {

template <typename T>
void atomic_update(const Pm4Command& command) {
    auto& word = *reinterpret_cast<T*>(static_cast<uintptr_t>(command.atomic_addr));
    std::atomic_ref<T> memory(word);
    T old_value = memory.load(std::memory_order_seq_cst);
    for (;;) {
        const T value = static_cast<T>(*atomic_mem_value(
            command.atomic_op, old_value, command.atomic_source, command.atomic_compare));
        if (memory.compare_exchange_weak(old_value, value, std::memory_order_seq_cst)) return;
    }
}

void refused(const char* reason, const Pm4Command& command) {
    static std::atomic<uint64_t> count{0};
    const uint64_t ordinal = count.fetch_add(1) + 1;
    if (ordinal <= 8 || (ordinal & (ordinal - 1)) == 0)
        std::fprintf(stderr, "[agc] PM4 memory command REFUSED #%llu: op=0x%x reason=%s\n",
                     static_cast<unsigned long long>(ordinal), command.op, reason);
}

std::optional<bool> conditional_value(const Pm4Command& command, const GpuState& state) {
    const uint64_t address = command.cond_exec_addr;
    // A dispatch is retained until the ordered executor runs. No write-range proof exists here,
    // so its CPU view cannot establish a predicate. Keep that unsupported producer shape visible.
    if (last_fold_deferred() || !state.dispatches.empty() || !state.dma_copies.empty()) {
        refused("conditional-producer-not-retired", command);
        return std::nullopt;
    }
    if (!address || (address & 3u) || !guest_readable(address, 4)) {
        refused("conditional-address-unreadable", command);
        return std::nullopt;
    }
    const auto pending = pending_conditional_value(address);
    if (!pending) {
        refused("conditional-pending-effect-unresolved", command);
        return std::nullopt;
    }
    uint32_t word = *pending;
    // Earlier fixed memory effects can establish the low dword without publishing a completion
    // early. Reuse the processor's checked overlay; ambiguous/aliased/dynamic effects refuse.
    for (const auto& effect : state.ordered_memory_effects) {
        const auto value = overlay_conditional_memory_effect(effect, address, word);
        if (!value) {
            refused("conditional-ordered-effect-unresolved", command);
            return std::nullopt;
        }
        word = *value;
    }
    return word != 0;
}

}   // namespace

std::optional<uint32_t> overlay_conditional_memory_effect(const GpuState::MemoryEffect& effect,
                                                          uint64_t address, uint32_t before) {
    const auto& command = effect.cmd;
    using Kind = Pm4Command::Kind;
    if (command.kind == Kind::DmaData && !dma_data_guest_destination(command)) return before;
    uint64_t destination = 0, bytes = 0;
    pm4_memory_effect_span(command, &destination, &bytes);
    if (command.kind == Kind::ReleaseMem)
        bytes = command.rel_data_sel == 0 ? 0 : command.rel_data_sel == 1 ? 4 : 8;
    if (!destination || !bytes ||
        guest_memory_topology_relation(address, 4, destination, bytes) ==
            GuestMemoryTopologyRelation::Disjoint)
        return before;
    // Only an exact virtual subrange can be projected from fixed owned data. Physical aliases,
    // partial writes and unknown producer topology need execution-time authority.
    if (address < destination || address - destination > bytes ||
        bytes - (address - destination) < 4)
        return std::nullopt;
    const uint64_t offset = address - destination;
    if (command.kind == Kind::WriteData) {
        if (!command.wd_valid || !command.wd_data || command.wd_num != command.wd_declared_num ||
            effect.write_data.size() != command.wd_num ||
            command.wd_data != effect.write_data.data() || (offset & 3u))
            return std::nullopt;
        return effect.write_data[offset / 4];
    }
    if (command.kind == Kind::ReleaseMem && command.rel_value_valid &&
        (command.rel_data_sel == 1 || command.rel_data_sel == 2))
        return uint32_t(command.rel_value >> (offset * 8));
    if (command.kind == Kind::DmaData && command.dd_valid && dma_data_immediate_operand(command) &&
        !(offset & 3u))
        return uint32_t(command.dd_src);
    // A 64-bit RMW needs its upper dword too. A four-byte parser read cannot invent that operand.
    if (command.kind == Kind::AtomicMem && offset == 0 &&
        atomic_mem_bytes(command.atomic_op) == 4) {
        const auto value = atomic_mem_value(command.atomic_op, before, command.atomic_source,
                                            command.atomic_compare);
        if (value) return uint32_t(*value);
    }
    return std::nullopt;
}

bool atomic_mem_operand_valid(const Pm4Command& command) {
    const uint32_t bytes = atomic_mem_bytes(command.atomic_op);
    return integer_atomic_mem_op(command.atomic_op) && command.atomic_addr &&
           !(command.atomic_addr & (bytes - 1)) && guest_readable(command.atomic_addr, bytes) &&
           guest_writable(command.atomic_addr, bytes);
}

bool refuse_atomic_mem_producer(const Pm4Command& command) {
    refused("atomic-producer-not-retired", command);
    return false;
}

bool execute_atomic_mem(const Pm4Command& command) {
    const uint32_t bytes = atomic_mem_bytes(command.atomic_op);
    if (!atomic_mem_operand_valid(command)) {
        refused("atomic-unsupported-or-unwritable", command);
        return false;
    }
    if (bytes == 4)
        atomic_update<uint32_t>(command);
    else
        atomic_update<uint64_t>(command);
    set_guest_gpu_write_origin("ATOMIC_MEM");
    notify_guest_gpu_write(command.atomic_addr, bytes);
    return true;
}

size_t fold_pm4_segment(const uint32_t* buffer, size_t dwords, GpuState& state,
                        std::vector<Pm4Command>& executed, uint8_t queue_origin) {
    size_t consumed = 0;
    while (consumed < dwords) {
        const size_t previous = executed.size();
        const size_t decoded = decode_pm4(buffer + consumed, dwords - consumed, executed, 1);
        consumed += decoded;
        if (executed.size() == previous) break;
        auto& command = executed.back();
        command.stream_order = state.command_order + 1;
        command.queue_origin = queue_origin;
        state.apply(command);
        if (command.kind != Pm4Command::Kind::CondExec) continue;
        const auto execute = conditional_value(command, state);
        if (!execute) {
            state.dma_execution_rejected = true;
            break;
        }
        if (*execute) continue;
        if (command.cond_exec_dwords > dwords - consumed) {
            refused("conditional-span-exceeds-buffer", command);
            state.dma_execution_rejected = true;
            break;
        }
        consumed += command.cond_exec_dwords;
    }
    return consumed;
}

}   // namespace prosper::gpu
