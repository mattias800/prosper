// Hardware indexed loads must select their own register file (#4866). Routing UC pairs to CX
// changes render state while leaving the intended user-config state stale. Synthetic packets pin
// the published GFX10 opcode identities without depending on a console capture or firmware data.
#include "gpu/pm4/command_processor.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace prosper::gpu {
namespace {

std::array<uint32_t, 5> indexed_load(uint32_t opcode, const ShaderReg* pairs, uint32_t count) {
    const uint64_t address = reinterpret_cast<uintptr_t>(pairs);
    return {0xc0030000u | (opcode << 8), static_cast<uint32_t>(address),
            static_cast<uint32_t>(address >> 32), 0x80000000u, count};
}

TEST(Pm4IndexedLoad, UserConfigPairsDoNotClobberContextRegisters) {
    constexpr uint32_t offset = 0x243;
    const std::array<ShaderReg, 2> pairs = {{{offset, 0x12345678u}, {offset + 1, 0u}}};
    const auto packet = indexed_load(0x64, pairs.data(), pairs.size());
    GpuState state;
    state.cx[offset] = 0xfeedfaceu;
    state.sh[offset] = 0xcafef00du;
    state.uc[offset + 1] = 0xffffffffu;

    EXPECT_EQ(run_command_buffer(packet.data(), packet.size(), state), 1u);
    const auto uc = state.uc.find(offset);
    ASSERT_NE(uc, state.uc.end()) << "0x64 is LOAD_UCONFIG_REG_INDEX";
    EXPECT_EQ(uc->second, pairs[0].value);
    EXPECT_EQ(state.uc.at(offset + 1), 0u) << "zero-valued pairs are real register writes";
    EXPECT_EQ(state.cx.at(offset), 0xfeedfaceu);
    EXPECT_EQ(state.sh.at(offset), 0xcafef00du);
}

TEST(Pm4IndexedLoad, IndexedOpcodesKeepAllThreeRegisterFilesDistinct) {
    constexpr uint32_t offset = 0x80;
    const ShaderReg shader = {offset, 11};
    const ShaderReg user_config = {offset, 22};
    const ShaderReg context = {offset, 33};
    const auto sh = indexed_load(0x63, &shader, 1);
    const auto uc = indexed_load(0x64, &user_config, 1);
    const auto cx = indexed_load(0x9f, &context, 1);
    std::vector<uint32_t> stream;
    for (const auto& packet : {sh, uc, cx})
        stream.insert(stream.end(), packet.begin(), packet.end());

    GpuState state;
    EXPECT_EQ(run_command_buffer(stream.data(), stream.size(), state), 3u);
    ASSERT_NE(state.uc.find(offset), state.uc.end());
    EXPECT_EQ(state.sh.at(offset), shader.value);
    EXPECT_EQ(state.uc.at(offset), user_config.value);
    EXPECT_EQ(state.cx.at(offset), context.value);
}

TEST(Pm4IndexedLoad, OrdinaryLoadOpcodesDoNotAcceptIndexedPairFields) {
    const ShaderReg pair = {0x80, 0x12345678u};
    for (const uint32_t opcode : {0x5eu, 0x5fu, 0x61u}) {
        const auto packet = indexed_load(opcode, &pair, 1);
        std::vector<Pm4Command> commands;
        EXPECT_EQ(decode_pm4(packet.data(), packet.size(), commands), packet.size());
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, Pm4Command::Kind::Unknown)
            << "ordinary LOAD packets have range fields, not bit31 offset/data pairs";
    }
}

TEST(Pm4IndexedLoad, UnsupportedIndexedAddressAndDataModesStayUnknown) {
    const ShaderReg pair = {0x80, 0x12345678u};
    for (const uint32_t opcode : {0x63u, 0x64u, 0x9fu}) {
        auto packet = indexed_load(opcode, &pair, 1);
        packet[1] |= 1u;
        std::vector<Pm4Command> commands;
        decode_pm4(packet.data(), packet.size(), commands);
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, Pm4Command::Kind::Unknown);

        packet = indexed_load(opcode, &pair, 1);
        packet[3] = 0u;
        commands.clear();
        decode_pm4(packet.data(), packet.size(), commands);
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, Pm4Command::Kind::Unknown);
    }
}

} // namespace
} // namespace prosper::gpu
