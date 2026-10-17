// buffer_address.hpp — the byte address and dword index of a MUBUF/MTBUF access, the first memory
// definition (HOW_TO_ADD_A_FAMILY.md, "Memory instructions").
//
// A definition here is VALUE only: the address arithmetic. Which descriptor the access resolves to
// (and so STRIDE), whether the V# is a folded per-vertex fetch, the access itself (load, store,
// atomic, robustness), and every provenance, mark and proof step stay in the lowering and in
// emit_alu, where they were.
#pragma once

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
}

namespace prosper::gpu::ir {

// Which address the lowering selected.
enum class BufferAddressForm : uint8_t {
    // The architectural address (AMD 70648, "Buffer Instructions", address calculation):
    // OFFSET + (IDXEN ? VADDR[0] * STRIDE : 0) + (OFFEN ? VADDR[IDXEN ? 1 : 0] : 0) + SOFFSET.
    // emit_alu drops the index term when STRIDE is 0, which is the same value.
    Faithful,
    // A folded per-vertex fetch whose V# base already holds OFFSET and SOFFSET (#206):
    // ELEMENT * STRIDE.
    FoldedElement,
    // The same with STRIDE 0 and no OFFEN: no dynamic term remains.
    FoldedZero,
};

struct BufferAddressShape {
    BufferAddressForm form = BufferAddressForm::Faithful;
    uint32_t offset = 0;     // Faithful: the instruction's OFFSET
    uint32_t stride = 0;     // Faithful (with `index`) and FoldedElement: the V#'s STRIDE
    bool index = false;      // Faithful: IDXEN with a nonzero STRIDE
    bool voffset = false;    // Faithful: OFFEN
};

// The guest words a shape reads, in the order the Block reads them.
enum class BufferAddressInput : uint8_t { Index, Voffset, Soffset, Element };

struct BufferAddressBlock {
    Block block;
    std::vector<BufferAddressInput> inputs;   // input slot -> what it reads
    Value addr;                               // the byte address
    Value idx;                                // addr >> 2
};

// Builds the definition. Inputs and constants are appended where the pre-IR emitter read the
// operand or allocated the constant, so the emitted SPIR-V is the old SPIR-V.
BufferAddressBlock build_buffer_address(const BufferAddressShape& shape);

// {addr, idx} for known words (Index, Voffset, Soffset, Element; an unread one is ignored).
std::array<Known, 2> eval_buffer_address(const BufferAddressShape& shape,
                                         const std::array<Known, 4>& words);

// Emits the definition; `read` is called for each input at its position in Block order.
using BufferAddressReader = std::function<uint32_t(BufferAddressInput)>;
std::array<uint32_t, 2> emit_buffer_address(SpirvCompute& b, const BufferAddressShape& shape,
                                            const BufferAddressReader& read);

}   // namespace prosper::gpu::ir
