// The MUBUF/MTBUF address lowering (ir/lower/buffer_address.cpp): which form emit_alu's facts
// select, observed through which operands it reads. SOFFSET names an SGPR marked as an unavailable
// saved-mask word, so reading it refuses the instruction (`ok == false`); a form that must not read
// it leaves `ok` set.
#include "gpu/recompiler/ir/lower/buffer_address_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"

#include <gtest/gtest.h>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {

// buffer_load_dword v1, v2, s[8:11], s12 idxen offset:8
const std::vector<uint32_t> kLoad = {0xe0302008u, 0x0c020102u};

bool lower(const BufferAddressFacts& facts, BufferAddressIds* ids = nullptr) {
    LoweringFixture f;
    f.rs.unavailable_saved_mask_words.insert(12);
    const Rdna2Inst in = LoweringFixture::decode(kLoad);
    EXPECT_EQ(in.fmt, Rdna2Format::MUBUF);
    const BufferAddressIds out = lower_buffer_address(f.b, f.rs, in, f.ok, facts);
    if (ids) {
        *ids = out;
        EXPECT_EQ(out.addr == f.b.uconst(0), facts.folded_vfetch && facts.stride == 0)
            << "only the folded zero-stride form is the constant 0";
    }
    return f.ok;
}

TEST(IrLowerBufferAddress, TheFaithfulAddressReadsSoffset) {
    EXPECT_FALSE(lower({8, 4, false, true, false, false, false}));
    // A folded fetch with OFFEN and STRIDE 0 keeps the faithful address (#206's limit).
    EXPECT_FALSE(lower({8, 0, true, true, true, true, false}));
    // IDXEN without a folded V#: faithful.
    EXPECT_FALSE(lower({8, 4, false, true, false, true, false}));
}

TEST(IrLowerBufferAddress, AFoldedFetchNeverAddsSoffsetAgain) {
    BufferAddressIds ids;
    EXPECT_TRUE(lower({8, 12, false, true, true, true, false}, &ids));   // gl_VertexIndex * 12
    EXPECT_TRUE(lower({8, 12, false, true, true, false, true}, &ids));   // gl_InstanceIndex * 12
    EXPECT_TRUE(lower({8, 0, false, true, true, false, false}, &ids));   // no dynamic term: 0
}

}   // namespace
