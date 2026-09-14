#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

namespace prosper::gpu {

// Incremented only for explicit workbench execution, at the real evaluator entry.
inline std::atomic<uint64_t> fold_workbench_evaluations{0};

// These are observable resource requests, not host decode-cache mapping checks.
enum class FoldProbe : uint32_t { Raw, Base48, Base40, ScalarBuffer, OptionalTable, NullableOutput };

// One reader belongs to one evaluation. Exceptions mean invalid replay, never an unreadable
// guest address. Live capture readers must keep executing after their recording budget expires.
struct FoldReader {
    bool branch_exclusive_disabled = false;
    uint64_t logical_code_address = 0;
    uint64_t evaluations = 0, evaluated_instructions = 0;
    uint32_t decoded_dwords = 0;
    bool shader_constant_specialized = false;
    virtual ~FoldReader() = default;
    virtual bool probe(FoldProbe kind, uint32_t pc, uint64_t address, uint32_t bytes) = 0;
    virtual uint32_t word(uint32_t pc, uint64_t address) = 0;
    virtual void prefix(uint32_t pc, uint64_t address, void* destination, uint32_t bytes) = 0;
};

// Preserve the evaluator's repeated full-load reads and short-circuit expressions. Only a
// partially OOB scalar-buffer load uses a local snapshot; its suffix is architectural zero.
class FoldWords {
    FoldReader* reader_;
    uint32_t pc_;
    uint64_t address_;
    const uint32_t* memory_;
public:
    FoldWords(FoldReader* reader, uint32_t pc, uint64_t address)
        : reader_(reader), pc_(pc), address_(address),
          memory_(reinterpret_cast<const uint32_t*>(uintptr_t(address))) {}
    uint32_t operator[](uint32_t index) const {
        return reader_ ? reader_->word(pc_, address_ + uint64_t(index) * sizeof(uint32_t))
                       : memory_[index];
    }
    void snapshot_prefix(uint32_t* storage, uint32_t bytes) {
        if (reader_) reader_->prefix(pc_, address_, storage, bytes);
        else std::memcpy(storage, reinterpret_cast<const void*>(uintptr_t(address_)), bytes);
        memory_ = storage;
        reader_ = nullptr;
    }
};

} // namespace prosper::gpu
