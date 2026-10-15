// #5052: the enabled FRONT_FACE VGPR must contain the hardware-selected raw word.
// A missing initial value rejects genuine integer predicates; a fabricated constant
// would instead admit the shader with the wrong side of every face-dependent branch.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <set>

namespace prosper::gpu {
namespace {
// Evaluate only typed uint constants, a real Input Bool BuiltIn FrontFacing load, and
// selects. No unknown SSA id, wrong storage class or arbitrary input becomes a zero.
std::optional<uint32_t> facing_word(const SpirvCompute& b, uint32_t value, bool front) {
    struct Node {
        uint32_t type, op;
        std::vector<uint32_t> args;
    };
    std::map<uint32_t, Node> nodes;
    std::map<uint32_t, uint32_t> pointer_pointee;
    std::set<uint32_t> input_variables, facing_variables;
    uint32_t uint_type = 0, bool_type = 0;
    bool valid = true;
    const auto collect = [&](const std::vector<uint32_t>& words) {
        for (size_t pc = 0; pc < words.size();) {
            const size_t n = words[pc] >> 16;
            if (!n || n > words.size() - pc) {
                valid = false;
                return;
            }
            const uint32_t op = words[pc] & 0xffffu;
            if (op == Op_TypeBool && n == 2) bool_type = words[pc + 1];
            if (op == Op_TypeInt && n == 4 && words[pc + 2] == 32 && words[pc + 3] == 0)
                uint_type = words[pc + 1];
            if (op == Op_TypePointer && n == 4 && words[pc + 2] == SC_Input)
                pointer_pointee[words[pc + 1]] = words[pc + 3];
            if (op == Op_Variable && n == 4 && words[pc + 3] == SC_Input &&
                pointer_pointee.contains(words[pc + 1]) &&
                pointer_pointee.at(words[pc + 1]) == bool_type)
                input_variables.insert(words[pc + 2]);
            if (op == Op_Decorate && n == 4 && words[pc + 2] == Dec_BuiltIn && words[pc + 3] == 17)
                facing_variables.insert(words[pc + 1]);
            if (op == Op_Constant || op == Op_Load || op == Op_Select) {
                if (n < 3 || !nodes
                                  .emplace(words[pc + 2],
                                           Node{words[pc + 1],
                                                op,
                                                {words.begin() + pc + 3, words.begin() + pc + n}})
                                  .second)
                    valid = false;
            }
            pc += n;
        }
    };
    collect(b.types);
    collect(b.deco);
    collect(b.code);
    if (!valid || !uint_type || !bool_type) return {};
    std::set<uint32_t> visiting;
    const auto read = [&](const auto& self, uint32_t id) -> std::optional<uint32_t> {
        if (!nodes.contains(id) || !visiting.insert(id).second || visiting.size() > 32) return {};
        const auto& n = nodes.at(id);
        std::optional<uint32_t> result;
        if (n.op == Op_Constant && n.type == uint_type && n.args.size() == 1)
            result = n.args[0];
        else if (n.op == Op_Load && n.type == bool_type && n.args.size() == 1 &&
                 input_variables.contains(n.args[0]) && facing_variables.contains(n.args[0]) &&
                 std::find(b.iface.begin(), b.iface.end(), n.args[0]) != b.iface.end())
            result = front;
        else if (n.op == Op_Select && n.type == uint_type && n.args.size() == 3 &&
                 nodes.contains(n.args[0]) && nodes.at(n.args[0]).type == bool_type &&
                 nodes.contains(n.args[1]) && nodes.at(n.args[1]).type == uint_type &&
                 nodes.contains(n.args[2]) && nodes.at(n.args[2]).type == uint_type) {
            const auto condition = self(self, n.args[0]);
            const auto yes = self(self, n.args[1]), no = self(self, n.args[2]);
            if (condition && yes && no) result = *condition ? *yes : *no;
        }
        visiting.erase(id);
        return result;
    };
    return read(read, value);
}

std::vector<uint32_t> program(uint32_t face_vgpr) {
    return {
        0x7d040080u | (face_vgpr << 9), // CMP_EQ_I32 VCC,0,face (back in all-bits mode)
        0x7e120280u,                   // v9=0
        0x021012f2u,                   // CNDMASK v8,1.0,v9,VCC
        0x7e140280u,
        0x7e1602f2u,      // v10=0,v11=1
        0xf800180fu,
        0x0b0a0908u,
        0xbf810000u,
    };
}
std::vector<uint32_t> compile(uint32_t vgpr, const PixelSystemInputMapping* inputs,
                              uint64_t address = 0x10004000u) {
    const auto raw = program(vgpr);
    return recompile_fragment(raw.data(), raw.size(), nullptr, inputs, UINT32_MAX, nullptr, true,
                              {RecompileDiagnosticStage::Fragment, address});
}
bool has_facing_input(const std::vector<uint32_t>& words) {
    for (size_t pc = 5; pc < words.size();) {
        const size_t n = words[pc] >> 16;
        if (!n || n > words.size() - pc) return false;
        if ((words[pc] & 0xffffu) == Op_Decorate && n == 4 && words[pc + 2] == Dec_BuiltIn &&
            words[pc + 3] == 17)
            return true;
        pc += n;
    }
    return false;
}
} // namespace

TEST(FragmentFrontFace, PublishedRawWordsFollowActualBooleanInput) {
    for (const auto encoding :
         {FragmentFrontFaceEncoding::SignedFloat, FragmentFrontFaceEncoding::AllBits}) {
        SpirvCompute b;
        b.begin_fragment();
        const uint32_t value = b.fragment_front_face_word(encoding);
        ASSERT_NE(value, 0u);
        const auto back = facing_word(b, value, false), front = facing_word(b, value, true);
        if (!back.has_value() || !front.has_value()) {
            FAIL() << "both faces must produce a typed raw input word";
        }
        // Independent raw ABI expectations, rather than constants read back from a round trip.
        EXPECT_EQ(*front,
                  encoding == FragmentFrontFaceEncoding::AllBits ? 0xffffffffu : 0x3f800000u);
        EXPECT_EQ(*back, encoding == FragmentFrontFaceEncoding::AllBits ? 0u : 0xbf800000u);
        EXPECT_NE(*front, *back) << "the host-facing condition must remain runtime input";
        EXPECT_FALSE(facing_word(b, UINT32_MAX, true).has_value());
    }
}

TEST(FragmentFrontFace, TypedOracleRejectsWrongBuiltinAndStorage) {
    SpirvCompute b;
    b.begin_fragment();
    const auto value = b.fragment_front_face_word(FragmentFrontFaceEncoding::AllBits);
    ASSERT_EQ(facing_word(b, value, true), std::optional<uint32_t>(0xffffffffu));
    const auto decorations = b.deco;
    for (size_t pc = 0; pc < b.deco.size(); pc += b.deco[pc] >> 16)
        if ((b.deco[pc] & 0xffffu) == Op_Decorate && (b.deco[pc] >> 16) == 4 &&
            b.deco[pc + 2] == Dec_BuiltIn && b.deco[pc + 3] == 17)
            b.deco[pc + 3] = 23;
    EXPECT_FALSE(facing_word(b, value, true).has_value()) << "HelperInvocation is not FrontFacing";
    b.deco = decorations;
    for (size_t pc = 0; pc < b.types.size(); pc += b.types[pc] >> 16)
        if ((b.types[pc] & 0xffffu) == Op_Variable && b.types[pc + 2] == b.v_front_facing)
            b.types[pc + 3] = 6u;   // SPIR-V Private storage class
    EXPECT_FALSE(facing_word(b, value, true).has_value()) << "private Bool is not a raster input";
}

TEST(FragmentFrontFace, UnknownOrNonFragmentModeNeverCreatesAWord) {
    SpirvCompute fragment;
    fragment.begin_fragment();
    for (const auto encoding :
         {FragmentFrontFaceEncoding::Unavailable, static_cast<FragmentFrontFaceEncoding>(0xff)})
        EXPECT_EQ(fragment.fragment_front_face_word(encoding), 0u);
    EXPECT_EQ(fragment.v_front_facing, 0u);
    SpirvCompute vertex;
    vertex.begin_vertex();
    EXPECT_EQ(vertex.fragment_front_face_word(FragmentFrontFaceEncoding::AllBits), 0u);
    EXPECT_EQ(vertex.v_front_facing, 0u);
}

TEST(FragmentFrontFace, EnabledInputCompilesInItsAddressedSlot) {
    for (const auto& [addr, slot] :
         std::array{std::pair{0x1000u, 0u}, std::pair{0x1002u, 2u}, std::pair{0x1f02u, 6u}}) {
        for (const uint32_t baryc : {0u, 1u << 24}) {
            PixelSystemInputMapping inputs{0x1000u, addr};
            inputs.retain_front_face_encoding(true, baryc);
            const auto module = compile(slot, &inputs);
            ASSERT_FALSE(module.empty()) << "ADDR must reserve disabled preceding fields";
            EXPECT_TRUE(has_facing_input(module));
            // The adjacent word is reserved by ADDR only: hardware loads nothing there, so
            // #5174 admits a never-written read of it as an arbitrary value. The refusal this arm
            // used to assert is an over-refusal under that decision; the addressed slot above and
            // the unsupplied-encoding arms below still pin where FRONT_FACE is bound.
            if (slot)
                EXPECT_FALSE(compile(slot - 1, &inputs).empty());
        }
    }
}

TEST(FragmentFrontFace, AbsentDisabledAndUnobservedInputsKeepTheOriginalRefusal) {
    constexpr uint64_t address = 0x10004100u;
    EXPECT_TRUE(compile(0, nullptr, address).empty());
    for (const auto& inputs :
         std::array{PixelSystemInputMapping{0, 0x1000u, FragmentFrontFaceEncoding::AllBits},
                    PixelSystemInputMapping{0x1000u, 0, FragmentFrontFaceEncoding::AllBits},
                    PixelSystemInputMapping{0x1000u, 0x1000u},
                    PixelSystemInputMapping{0x1000u, 0x1000u,
                                            static_cast<FragmentFrontFaceEncoding>(0xff)}}) {
        EXPECT_TRUE(compile(0, &inputs, address).empty());
        EXPECT_NE(last_terminal_reject_reason(address).find("fabricated-fresh-mask-source"),
                  std::string::npos);
    }
}
}   // namespace prosper::gpu
