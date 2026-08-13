#include "rdna2_indirect_pointer_analysis.hpp"

#include "rdna2_decode.hpp"
#include "rdna2_to_spirv.hpp"
#include "shader_resources.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace prosper::gpu {

bool guest_readable(uint64_t address, uint32_t bytes);

namespace {

struct PacketSite {
    uint32_t pc;
    uint32_t word0;
    uint32_t word1 = 0;
    uint32_t dwords = 1;
};

struct VgprLiveRange {
    uint32_t begin_pc;
    uint32_t end_pc;
    uint32_t first_vgpr;
    uint32_t count = 1u;
};

struct ScalarLiveRange {
    uint32_t begin_pc;
    uint32_t end_pc;
    uint32_t first_sgpr;
    uint32_t count = 1u;
};

struct AddressPairLiveRange {
    uint32_t low_producer_pc;
    uint32_t high_producer_pc;
    uint32_t access_pc;
    uint32_t low_vgpr;
};

struct DescriptorRangeShape {
    uint32_t dwords;
    uint32_t user_sgprs;
    uint32_t tid_group_shift;
    uint32_t outer_descriptor_pc;
    uint32_t outer_fetch_pc;
    uint32_t main_descriptor_pc;
    uint32_t main_range_pc;
    uint32_t main_index_pc;
    uint32_t pointer_descriptor_pc;
    uint32_t pointer_fetch_pc;
    uint32_t pointer_vgpr;
    uint32_t full_descriptor_index_pc;
    uint32_t full_descriptor_fetch_pc;
    std::span<const PacketSite> selection_loop;
    std::span<const PacketSite> authority;
    std::span<const PacketSite> branches;
    std::span<const PacketSite> accesses;
    std::span<const PacketSite> residual_slice;
    std::span<const PacketSite> exec_writers;
    std::span<const VgprLiveRange> value_liveness;
    std::span<const AddressPairLiveRange> address_liveness;
    std::span<const ScalarLiveRange> scalar_liveness;
};

// The host-side main-record enumeration mirrors the shader's pc5..49 ordinal/mask/record
// selection loop. Every non-wait packet in that loop is part of the authority: changing even an
// apparently incidental initialization or cndmask can change which record owns a lane.
constexpr std::array<PacketSite, 36> kShape642SelectionLoop{{
    {5u,  0xbeeb0380u}, {6u,  0xbe85037eu}, {7u,  0xbe90037eu},
    {9u,  0xf4240184u, 0xfa000014u, 2u},
    {11u, 0xf4080301u, 0xfa000020u, 2u}, {13u, 0xbf88001du},
    {14u, 0x7e3e02c1u}, {16u, 0xbf0a076bu}, {17u, 0xbf800000u},
    {18u, 0x856a807eu}, {19u, 0x8b087e6au}, {20u, 0xbeea0308u},
    {21u, 0x87100810u}, {22u, 0x877e087eu}, {23u, 0xbf860013u},
    {24u, 0x7e04026bu}, {25u, 0x816a6b06u}, {26u, 0x810a816bu},
    {27u, 0xb86a0050u}, {28u, 0xf4240206u, 0xd4000024u, 2u},
    {31u, 0x816a0809u}, {32u, 0x7d8600f9u, 0x0686eb09u, 2u},
    {34u, 0x7d88006au}, {35u, 0x876a6a6bu},
    {36u, 0x020306f9u, 0x86860682u, 2u}, {38u, 0x7da40282u},
    {39u, 0xbf880003u}, {40u, 0x7e020280u}, {41u, 0xbeeb030au},
    {42u, 0xbf82ffe2u}, {43u, 0xbefe0310u}, {44u, 0x7da40283u},
    {46u, 0x4a3e0406u}, {47u, 0xbefe0305u}, {48u, 0x7daa3ec1u},
    {49u, 0xbf88024fu},
}};

constexpr std::array<PacketSite, 36> kShape662SelectionLoop{{
    {5u,  0xbeeb0380u}, {6u,  0xbe88037eu}, {7u,  0xbe89037eu},
    {9u,  0xf4240086u, 0xfa000014u, 2u},
    {11u, 0xf4080402u, 0xfa000020u, 2u}, {13u, 0xbf88001du},
    {14u, 0x7e3c02c1u}, {16u, 0xbf0a036bu}, {17u, 0xbf800000u},
    {18u, 0x856a807eu}, {19u, 0x8b067e6au}, {20u, 0xbeea0306u},
    {21u, 0x87090609u}, {22u, 0x877e067eu}, {23u, 0xbf860013u},
    {24u, 0x7e04026bu}, {25u, 0x816a6b02u}, {26u, 0x810a816bu},
    {27u, 0xb86a0050u}, {28u, 0xf4240188u, 0xd4000024u, 2u},
    {31u, 0x816a0607u}, {32u, 0x7d8600f9u, 0x0686eb07u, 2u},
    {34u, 0x7d88006au}, {35u, 0x876a6a6bu},
    {36u, 0x020306f9u, 0x86860682u, 2u}, {38u, 0x7da40282u},
    {39u, 0xbf880003u}, {40u, 0x7e020280u}, {41u, 0xbeeb030au},
    {42u, 0xbf82ffe2u}, {43u, 0xbefe0309u}, {44u, 0x7da40283u},
    {46u, 0x4a3c0402u}, {47u, 0xbefe0308u}, {48u, 0x7daa3cc1u},
    {49u, 0xbf880263u},
}};

constexpr std::array<PacketSite, 29> kShape642Authority{{
    {1u,   0xd7460000u, 0x04010a05u, 2u},
    {3u,   0xf4080201u, 0xfa000080u, 2u},
    {9u,   0xf4240184u, 0xfa000014u, 2u},
    {11u,  0xf4080301u, 0xfa000020u, 2u},
    {24u,  0x7e04026bu},
    {25u,  0x816a6b06u},
    {27u,  0xb86a0050u},
    {28u,  0xf4240206u, 0xd4000024u, 2u},
    {31u,  0x816a0809u},
    {32u,  0x7d8600f9u, 0x0686eb09u, 2u},
    {34u,  0x7d88006au},
    {35u,  0x876a6a6bu},
    {38u,  0x7da40282u},
    {40u,  0x7e020280u},
    {41u,  0xbeeb030au},
    {42u,  0xbf82ffe2u},
    {46u,  0x4a3e0406u},
    {48u,  0x7daa3ec1u},
    {49u,  0xbf88024fu},
    {51u,  0xe0302030u, 0x8003021fu, 2u},
    {57u,  0xf4080200u, 0xfa000020u, 2u},
    {64u,  0xe0342000u, 0x80021b02u, 2u},
    {79u,  0xd70f6a01u, 0x0002031bu, 2u},
    {81u,  0x7e4002f9u, 0x000c061cu, 2u},
    {83u,  0x500438f9u, 0x0c860680u, 2u},
    {85u,  0xd70f6a00u, 0x0002011bu, 2u},
    {89u,  0x500238f9u, 0x0c860680u, 2u},
    {587u, 0x8f6a846au},
    {588u, 0xf4280004u, 0xd4000000u, 2u},
}};

constexpr std::array<PacketSite, 14> kShape642Branches{{
    {13u,  0xbf88001du}, {23u,  0xbf860013u}, {39u,  0xbf880003u},
    {42u,  0xbf82ffe2u}, {49u,  0xbf88024fu}, {117u, 0xbf8800b8u},
    {246u, 0xbf880036u}, {303u, 0xbf8800f4u}, {305u, 0xbf84000bu},
    {316u, 0xbf820067u},
    {503u, 0xbf88002cu}, {600u, 0xbf89ffebu}, {603u, 0xbf880025u},
    {640u, 0xbf89fff3u},
}};

constexpr std::array<PacketSite, 19> kShape642Accesses{{
    {87u,  0xdc308000u, 0x027d0001u, 2u},
    {91u,  0xdc308000u, 0x007d0000u, 2u},
    {204u, 0xdc308000u, 0x197d0018u, 2u},
    {216u, 0xdc308000u, 0x277d0016u, 2u},
    {222u, 0xdc308000u, 0x2b7d002bu, 2u},
    {261u, 0xdc308000u, 0x0b7d0002u, 2u},
    {265u, 0xdc308000u, 0x0d7d0000u, 2u},
    {268u, 0xdc308000u, 0x0c7d0003u, 2u},
    {280u, 0xdc308000u, 0x1c7d0009u, 2u},
    {444u, 0xdc308000u, 0x197d0011u, 2u},
    {446u, 0xdc308000u, 0x2b7d000fu, 2u},
    {451u, 0xdc308000u, 0x277d0010u, 2u},
    {457u, 0xdc308000u, 0x177d0016u, 2u},
    {465u, 0xdc308000u, 0x107d0022u, 2u},
    {475u, 0xdc308000u, 0x0e7d000eu, 2u},
    {516u, 0xdc308000u, 0x037d000du, 2u},
    {518u, 0xdc308000u, 0x0d7d000bu, 2u},
    {523u, 0xdc308000u, 0x077d000bu, 2u},
    {528u, 0xdc308000u, 0x1c7d000bu, 2u},
}};

constexpr std::array<PacketSite, 29> kShape662Authority{{
    {1u,   0xd7460000u, 0x04010a06u, 2u},
    {3u,   0xf4080302u, 0xfa000080u, 2u},
    {9u,   0xf4240086u, 0xfa000014u, 2u},
    {11u,  0xf4080402u, 0xfa000020u, 2u},
    {24u,  0x7e04026bu},
    {25u,  0x816a6b02u},
    {27u,  0xb86a0050u},
    {28u,  0xf4240188u, 0xd4000024u, 2u},
    {31u,  0x816a0607u},
    {32u,  0x7d8600f9u, 0x0686eb07u, 2u},
    {34u,  0x7d88006au},
    {35u,  0x876a6a6bu},
    {38u,  0x7da40282u},
    {40u,  0x7e020280u},
    {41u,  0xbeeb030au},
    {42u,  0xbf82ffe2u},
    {46u,  0x4a3c0402u},
    {48u,  0x7daa3cc1u},
    {49u,  0xbf880263u},
    {51u,  0xe0302030u, 0x8004021eu, 2u},
    {59u,  0xf4080200u, 0xfa000020u, 2u},
    {66u,  0xe0342000u, 0x80021202u, 2u},
    {81u,  0xd70f6a01u, 0x00020312u, 2u},
    {83u,  0x7e3a02f9u, 0x000c0613u, 2u},
    {85u,  0x500426f9u, 0x0c860680u, 2u},
    {87u,  0xd70f6a00u, 0x00020112u, 2u},
    {91u,  0x500226f9u, 0x0c860680u, 2u},
    {607u, 0x8f6a846au},
    {608u, 0xf4280004u, 0xd4000000u, 2u},
}};

constexpr std::array<PacketSite, 12> kShape662Branches{{
    {13u,  0xbf88001du}, {23u,  0xbf860013u}, {39u,  0xbf880003u},
    {42u,  0xbf82ffe2u}, {49u,  0xbf880263u}, {119u, 0xbf8800bbu},
    {251u, 0xbf880036u}, {308u, 0xbf8800d9u}, {481u, 0xbf88002cu},
    {620u, 0xbf89ffecu}, {623u, 0xbf880025u}, {660u, 0xbf89fff3u},
}};

constexpr std::array<PacketSite, 20> kShape662Accesses{{
    {89u,  0xdc308000u, 0x027d0001u, 2u},
    {93u,  0xdc308000u, 0x017d0000u, 2u},
    {170u, 0xdc308000u, 0x197d0018u, 2u},
    {195u, 0xdc308000u, 0x187d0017u, 2u},
    {197u, 0xdc308000u, 0x177d0015u, 2u},
    {266u, 0xdc308000u, 0x117d0002u, 2u},
    {270u, 0xdc308000u, 0x147d0000u, 2u},
    {273u, 0xdc308000u, 0x137d0003u, 2u},
    {285u, 0xdc308000u, 0x1f7d0009u, 2u},
    {344u, 0xdc308000u, 0x177d0015u, 2u},
    {351u, 0xdc308000u, 0x197d0014u, 2u},
    {365u, 0xdc308000u, 0x187d0032u, 2u},
    {372u, 0xdc308000u, 0x2c7d0013u, 2u},
    {384u, 0xdc308000u, 0x157d0015u, 2u},
    {387u, 0xdc308000u, 0x137d0013u, 2u},
    {494u, 0xdc308000u, 0x027d0002u, 2u},
    {496u, 0xdc308000u, 0x0a7d0000u, 2u},
    {501u, 0xdc308000u, 0x037d0000u, 2u},
    {506u, 0xdc308000u, 0x1f7d0000u, 2u},
    {560u, 0xdc308000u, 0x087d0008u, 2u},
}};

// These are the complete address-producing slices between the selected main record and every
// raw GLOBAL consumer.  Keeping them separate from the packet/branch inventories lets unrelated
// shader work vary without granting authority to a merely-tainted address pair.
constexpr std::array<PacketSite, 77> kShape642ResidualSlice{{
    {6u, 0xbe85037eu}, {7u, 0xbe90037eu}, {21u, 0x87100810u},
    {53u, 0xe0302028u, 0x8003011fu, 2u}, {55u, 0xe0302020u, 0x80031e1fu, 2u},
    {67u, 0x4c3a0300u}, {69u, 0x2c003c98u},
    {70u, 0xd5690021u, 0x00023b00u, 2u}, {72u, 0x4a0242f9u, 0x0600061eu, 2u},
    {74u, 0x4a0042f9u, 0x0601061eu, 2u}, {76u, 0x360202c4u},
    {77u, 0x360000c4u}, {95u, 0x7e062502u},
    {107u, 0x7e0402f9u, 0x00020600u, 2u}, {116u, 0x7da2046bu},
    {59u, 0xb06b00feu}, {118u, 0xe0382038u, 0x8003001fu, 2u},
    {137u, 0x7d8a06f9u, 0x06868080u, 2u}, {245u, 0xbeeb3c00u},
    {60u, 0xbe85037eu},
    {152u, 0x4a004302u}, {155u, 0x360400c4u}, {156u, 0x4a020484u},
    {157u, 0x4a2c0488u}, {158u, 0xd70f6b2bu, 0x0002051bu, 2u},
    {160u, 0xd70f6a18u, 0x0002031bu, 2u}, {162u, 0x50324080u},
    {201u, 0xd70f6a16u, 0x00022d1bu, 2u}, {209u, 0x502e4080u},
    {214u, 0xd5286a2cu, 0x01ae4080u, 2u}, {247u, 0x4a004303u},
    {250u, 0x360400c4u}, {251u, 0x4a020484u},
    {252u, 0xd70f0000u, 0x0002051bu, 2u}, {254u, 0x4a080488u},
    {255u, 0x4a0e048cu}, {256u, 0xd70f6a02u, 0x0002031bu, 2u},
    {258u, 0x50064080u}, {259u, 0xd5286a01u, 0x00024080u, 2u},
    {263u, 0xd70f6a03u, 0x0002091bu, 2u}, {267u, 0x50084080u},
    {275u, 0xd70f6a09u, 0x00020f1bu, 2u}, {279u, 0x50144080u},
    {420u, 0xe0342040u, 0x80030c1fu, 2u}, {422u, 0x361e42c4u},
    {430u, 0x7d8a1af9u, 0x06868180u, 2u},
    {424u, 0xd70f0016u, 0x00021f1bu, 2u}, {426u, 0x4a2a1e84u},
    {427u, 0x4a281e88u}, {429u, 0x4a18430cu}, {432u, 0x362018c4u},
    {435u, 0x4a182084u}, {436u, 0xd70f6b0fu, 0x0002211bu, 2u},
    {438u, 0x4a262088u}, {439u, 0xd70f6a11u, 0x0002191bu, 2u},
    {441u, 0x50244080u}, {442u, 0xd5286a10u, 0x01ae4080u, 2u},
    {448u, 0xd70f6a10u, 0x0002271bu, 2u}, {450u, 0x50224080u},
    {453u, 0xd5286a17u, 0x00024080u, 2u},
    {455u, 0xd70f6a22u, 0x00022b1bu, 2u}, {459u, 0x50464080u},
    {463u, 0xd70f6a0eu, 0x0002291bu, 2u}, {470u, 0x501e4080u},
    {504u, 0x4a06430du}, {505u, 0x361806c4u}, {506u, 0x4a061884u},
    {507u, 0xd70f6b0bu, 0x0002191bu, 2u}, {509u, 0x4a0e1888u},
    {510u, 0x4a24188cu}, {511u, 0xd70f6a0du, 0x0002071bu, 2u},
    {513u, 0x501c4080u}, {514u, 0xd5286a0cu, 0x01ae4080u, 2u},
    {520u, 0xd70f6a0bu, 0x00020f1bu, 2u}, {522u, 0x50184080u},
    {525u, 0xd70f6a0bu, 0x0002251bu, 2u}, {527u, 0x50184080u},
}};

constexpr std::array<PacketSite, 78> kShape662ResidualSlice{{
    {6u, 0xbe88037eu}, {7u, 0xbe89037eu}, {21u, 0x87090609u},
    {53u, 0xe0302028u, 0x8004011eu, 2u}, {55u, 0xe0302020u, 0x80041c1eu, 2u},
    {57u, 0xe030203cu, 0x8004201eu, 2u}, {69u, 0x4c340300u},
    {71u, 0x2c003898u}, {72u, 0xd5690021u, 0x00023500u, 2u},
    {74u, 0x4a0242f9u, 0x0600061cu, 2u}, {76u, 0x4a0042f9u, 0x0601061cu, 2u},
    {78u, 0x360202c4u}, {79u, 0x360000c4u},
    {61u, 0xb06b00feu},
    {120u, 0xe0342040u, 0x8004131eu, 2u}, {142u, 0x4a1a4313u},
    {62u, 0xbe86037eu}, {150u, 0x7d8a28f9u, 0x06868480u, 2u},
    {152u, 0x36261ac4u}, {161u, 0x4a2c2684u},
    {162u, 0xd70f6b15u, 0x00022712u, 2u}, {164u, 0x4a262688u},
    {165u, 0xd70f6a18u, 0x00022d12u, 2u}, {167u, 0x50323a80u},
    {168u, 0xd70f6a17u, 0x00022712u, 2u}, {172u, 0x50303a80u},
    {177u, 0xd5286a16u, 0x01ae3a80u, 2u},
    {252u, 0x4a004314u}, {255u, 0x360400c4u}, {256u, 0x4a020484u},
    {257u, 0xd70f0400u, 0x00020512u, 2u}, {259u, 0x4a080488u},
    {260u, 0x4a0e048cu}, {261u, 0xd70f6a02u, 0x00020312u, 2u},
    {263u, 0x50063a80u}, {264u, 0xd5286a01u, 0x00123a80u, 2u},
    {268u, 0xd70f6a03u, 0x00020912u, 2u}, {272u, 0x50083a80u},
    {280u, 0xd70f6a09u, 0x00020f12u, 2u}, {284u, 0x50143a80u},
    {311u, 0xe0342040u, 0x8004101eu, 2u}, {313u, 0x360242c4u},
    {332u, 0x7d8a22f9u, 0x06868480u, 2u},
    {315u, 0xd70f6b13u, 0x00020312u, 2u}, {317u, 0x4a5a0284u},
    {318u, 0x4a500288u}, {324u, 0x4a004310u}, {334u, 0x360000c4u},
    {337u, 0xd70f6a15u, 0x00020112u, 2u}, {339u, 0x4a280084u},
    {340u, 0x4a000088u}, {341u, 0x502c3a80u},
    {342u, 0xd70f6a14u, 0x00022912u, 2u}, {346u, 0x502a3a80u},
    {349u, 0xd70f6a32u, 0x00020112u, 2u}, {357u, 0x50663a80u},
    {363u, 0xd5286a14u, 0x01ae3a80u, 2u},
    {370u, 0xd70f6a15u, 0x00025b12u, 2u}, {381u, 0x502c3a80u},
    {382u, 0xd70f6a13u, 0x00025112u, 2u}, {386u, 0x50283a80u},
    {482u, 0x4a004311u}, {483u, 0x360400c4u}, {484u, 0x4a020484u},
    {485u, 0xd70f6b00u, 0x00020512u, 2u}, {487u, 0x4a220488u},
    {488u, 0x4a0e048cu}, {489u, 0xd70f6a02u, 0x00020312u, 2u},
    {491u, 0x50063a80u}, {492u, 0xd5286a01u, 0x01ae3a80u, 2u},
    {498u, 0xd70f6a00u, 0x00022312u, 2u}, {500u, 0x50023a80u},
    {503u, 0xd70f6a00u, 0x00020f12u, 2u}, {505u, 0x50023a80u},
    {529u, 0x4a1040f9u, 0x00060621u, 2u}, {545u, 0x361010c4u},
    {552u, 0xd70f6a08u, 0x00021112u, 2u}, {558u, 0x50123a80u},
}};

constexpr std::array<PacketSite, 18> kShape642ExecWriters{{
    {22u, 0x877e087eu}, {38u, 0x7da40282u}, {43u, 0xbefe0310u},
    {44u, 0x7da40283u}, {47u, 0xbefe0305u}, {48u, 0x7daa3ec1u},
    {116u, 0x7da2046bu}, {245u, 0xbeeb3c00u}, {301u, 0xbefe036bu},
    {302u, 0x8a7e7e05u}, {502u, 0xbefe0301u}, {548u, 0xbefe0305u},
    {584u, 0xbeeb3c04u}, {599u, 0x8a7e046bu}, {601u, 0xbefe0305u},
    {602u, 0x7daa3480u}, {631u, 0xbeeb3c04u}, {639u, 0x8a7e046bu},
}};

constexpr std::array<PacketSite, 18> kShape662ExecWriters{{
    {22u, 0x877e067eu}, {38u, 0x7da40282u}, {43u, 0xbefe0309u},
    {44u, 0x7da40283u}, {47u, 0xbefe0308u}, {48u, 0x7daa3cc1u},
    {118u, 0x7da2046bu}, {250u, 0xbeeb3c04u}, {306u, 0xbefe036bu},
    {307u, 0x8a7e7e06u}, {480u, 0xbefe0304u}, {526u, 0xbefe0306u},
    {604u, 0xbeeb3c04u}, {619u, 0x8a7e046bu}, {621u, 0xbefe0306u},
    {622u, 0x7daa3680u}, {651u, 0xbeeb3c04u}, {659u, 0x8a7e046bu},
}};

// Producer-to-last-consumer intervals for every dispatch value read by the numerical residual
// model. The producing packet is separately exact; these intervals prove that an otherwise
// unrelated same-length instruction cannot silently replace the value before its consumer.
constexpr std::array<VgprLiveRange, 21> kShape642ValueLiveness{{
    {1u, 67u, 0u}, {46u, 420u, 31u}, {51u, 64u, 2u}, {53u, 67u, 1u},
    {55u, 75u, 30u},
    {70u, 505u, 33u}, {64u, 526u, 27u}, {81u, 528u, 32u},
    {107u, 116u, 2u}, {118u, 247u, 3u},
    {152u, 155u, 0u}, {155u, 158u, 2u}, {156u, 161u, 1u},
    {157u, 201u, 22u}, {247u, 250u, 0u}, {250u, 253u, 2u},
    {251u, 257u, 1u}, {254u, 264u, 4u}, {255u, 276u, 7u},
    {420u, 429u, 12u}, {420u, 505u, 13u},
}};

constexpr std::array<VgprLiveRange, 17> kShape662ValueLiveness{{
    {1u, 69u, 0u}, {46u, 311u, 30u}, {51u, 66u, 2u}, {53u, 69u, 1u},
    {55u, 77u, 28u},
    {57u, 530u, 32u}, {72u, 530u, 33u}, {66u, 553u, 18u},
    {83u, 559u, 29u}, {120u, 143u, 19u}, {120u, 253u, 20u},
    {311u, 325u, 16u}, {311u, 483u, 17u},
    {142u, 153u, 13u}, {252u, 256u, 0u}, {255u, 258u, 2u},
    {482u, 484u, 0u},
}};

constexpr std::array<AddressPairLiveRange, 19> kShape642AddressLiveness{{
    {79u, 83u, 87u, 1u}, {85u, 89u, 91u, 0u},
    {160u, 162u, 204u, 24u}, {201u, 209u, 216u, 22u},
    {158u, 214u, 222u, 43u}, {256u, 258u, 261u, 2u},
    {252u, 259u, 265u, 0u}, {263u, 267u, 268u, 3u},
    {275u, 279u, 280u, 9u}, {439u, 441u, 444u, 17u},
    {436u, 442u, 446u, 15u}, {448u, 450u, 451u, 16u},
    {424u, 453u, 457u, 22u}, {455u, 459u, 465u, 34u},
    {463u, 470u, 475u, 14u}, {511u, 513u, 516u, 13u},
    {507u, 514u, 518u, 11u}, {520u, 522u, 523u, 11u},
    {525u, 527u, 528u, 11u},
}};

constexpr std::array<AddressPairLiveRange, 20> kShape662AddressLiveness{{
    {81u, 85u, 89u, 1u}, {87u, 91u, 93u, 0u},
    {165u, 167u, 170u, 24u}, {168u, 172u, 195u, 23u},
    {162u, 177u, 197u, 21u}, {261u, 263u, 266u, 2u},
    {257u, 264u, 270u, 0u}, {268u, 272u, 273u, 3u},
    {280u, 284u, 285u, 9u}, {337u, 341u, 344u, 21u},
    {342u, 346u, 351u, 20u}, {349u, 357u, 365u, 50u},
    {315u, 363u, 372u, 19u}, {370u, 381u, 384u, 21u},
    {382u, 386u, 387u, 19u}, {489u, 491u, 494u, 2u},
    {485u, 492u, 496u, 0u}, {498u, 500u, 501u, 0u},
    {503u, 505u, 506u, 0u}, {552u, 558u, 560u, 8u},
}};

constexpr std::array<ScalarLiveRange, 19> kShape642ScalarLiveness{{
    {3u, 9u, 8u, 4u}, {6u, 47u, 5u}, {7u, 21u, 16u}, {21u, 43u, 16u},
    {9u, 46u, 6u, 2u}, {11u, 420u, 12u, 4u},
    {25u, 27u, 106u}, {27u, 28u, 106u}, {28u, 31u, 8u, 2u},
    {31u, 34u, 106u}, {32u, 35u, 107u}, {34u, 35u, 106u},
    {41u, 42u, 107u}, {57u, 64u, 8u, 4u}, {59u, 116u, 107u},
    {60u, 302u, 5u},
    {137u, 245u, 0u}, {245u, 301u, 107u}, {430u, 502u, 1u},
}};

constexpr std::array<ScalarLiveRange, 19> kShape662ScalarLiveness{{
    {3u, 9u, 12u, 4u}, {6u, 47u, 8u}, {7u, 21u, 9u}, {21u, 43u, 9u},
    {9u, 46u, 2u, 2u}, {11u, 311u, 16u, 4u},
    {25u, 27u, 106u}, {27u, 28u, 106u}, {28u, 31u, 6u, 2u},
    {31u, 34u, 106u}, {32u, 35u, 107u}, {34u, 35u, 106u},
    {41u, 42u, 107u}, {59u, 66u, 8u, 4u}, {61u, 118u, 107u},
    {62u, 526u, 6u},
    {150u, 250u, 4u}, {250u, 306u, 107u}, {332u, 480u, 4u},
}};

constexpr DescriptorRangeShape kShapes[]{
    {642u, 5u, 5u, 3u, 9u, 11u, 28u, 51u, 57u, 64u, 27u, 587u, 588u,
     kShape642SelectionLoop, kShape642Authority, kShape642Branches, kShape642Accesses,
     kShape642ResidualSlice,
     kShape642ExecWriters, kShape642ValueLiveness, kShape642AddressLiveness,
     kShape642ScalarLiveness},
    {662u, 6u, 6u, 3u, 9u, 11u, 28u, 51u, 59u, 66u, 18u, 607u, 608u,
     kShape662SelectionLoop, kShape662Authority, kShape662Branches, kShape662Accesses,
     kShape662ResidualSlice,
     kShape662ExecWriters, kShape662ValueLiveness, kShape662AddressLiveness,
     kShape662ScalarLiveness},
};

const Rdna2Inst* instruction_at(const std::vector<Rdna2Inst>& instructions, uint32_t pc) {
    const auto found = std::find_if(
        instructions.begin(), instructions.end(),
        [&](const Rdna2Inst& instruction) { return instruction.pc == pc; });
    return found == instructions.end() ? nullptr : &*found;
}

bool packet_is(const Rdna2Inst& instruction, const PacketSite& site) {
    return instruction.pc == site.pc && instruction.len_dwords == site.dwords &&
           instruction.words[0] == site.word0 &&
           (site.dwords == 1u || instruction.words[1] == site.word1);
}

bool packets_match(const std::vector<Rdna2Inst>& instructions,
                   std::span<const PacketSite> sites) {
    return std::all_of(sites.begin(), sites.end(), [&](const PacketSite& site) {
        const Rdna2Inst* instruction = instruction_at(instructions, site.pc);
        return instruction && packet_is(*instruction, site);
    });
}

bool indirect_control_transfer(const Rdna2Inst& instruction) {
    return (instruction.fmt == Rdna2Format::SOP1 &&
            (instruction.opcode == kSop1OpcodeSetpcB64 ||
             instruction.opcode == kSop1OpcodeSwappcB64 ||
             instruction.opcode == kSop1OpcodeRfeB64)) ||
           (instruction.fmt == Rdna2Format::SOPK &&
            (instruction.opcode == kSopkOpcodeCallB64 ||
             instruction.opcode == kSopkOpcodeSubvectorLoopBegin ||
             instruction.opcode == kSopkOpcodeSubvectorLoopEnd)) ||
           (instruction.fmt == Rdna2Format::SOPP &&
            (instruction.opcode == kSoppOpcodeTrap ||
             instruction.opcode == kSoppOpcodeCbranchCdbgsys ||
             instruction.opcode == kSoppOpcodeCbranchCdbguser ||
             instruction.opcode == kSoppOpcodeCbranchCdbgsysOrUser ||
             instruction.opcode == kSoppOpcodeCbranchCdbgsysAndUser));
}

bool control_inventory_matches(const std::vector<Rdna2Inst>& instructions,
                               const DescriptorRangeShape& shape) {
    if (!packets_match(instructions, shape.branches)) return false;
    size_t direct_branches = 0;
    for (const Rdna2Inst& instruction : instructions) {
        if (indirect_control_transfer(instruction)) return false;
        if (instruction.fmt != Rdna2Format::SOPP ||
            !sopp_opcode_is_direct_branch(instruction.opcode))
            continue;
        ++direct_branches;
        const auto expected = std::find_if(
            shape.branches.begin(), shape.branches.end(),
            [&](const PacketSite& site) { return site.pc == instruction.pc; });
        if (expected == shape.branches.end() || !packet_is(instruction, *expected))
            return false;
    }
    return direct_branches == shape.branches.size();
}

const ShaderResource* unique_resource_at(const ShaderResourceTable& resources,
                                         uint32_t fetch_pc) {
    const ShaderResource* result = nullptr;
    for (const ShaderResource& resource : resources.resources) {
        if (resource.fetch_pc != fetch_pc) continue;
        if (result) return nullptr;
        result = &resource;
    }
    return result;
}

const uint8_t* complete_bytes(const ShaderResource& resource) {
    if (resource.host_data && resource.host_data_size >= resource.size)
        return resource.host_data;
    return resource.size <= UINT32_MAX && guest_readable(resource.gpu_addr, resource.size)
        ? reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(resource.gpu_addr))
        : nullptr;
}

uint32_t load_u32(const uint8_t* bytes, size_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

bool resource_is(const ShaderResource* resource, uint32_t fetch_pc,
                 uint32_t size, uint32_t stride) {
    return resource && resource->cls == ResourceClass::ConstantBuffer &&
           resource->fetch_pc == fetch_pc && resource->size == size &&
           resource->stride == stride && resource->gpu_addr &&
           resource->table_index_count == 0u;
}

bool vgpr_range_unchanged(const std::vector<Rdna2Inst>& instructions,
                          const VgprLiveRange& live) {
    const uint64_t wanted_end = static_cast<uint64_t>(live.first_vgpr) + live.count;
    return std::none_of(instructions.begin(), instructions.end(), [&](const Rdna2Inst& in) {
        if (in.pc <= live.begin_pc || in.pc >= live.end_pc ||
            in.dst.kind != OperandKind::VGPR || in.dst.value < 0)
            return false;
        const uint32_t width = rdna2_vgpr_destination_span(in);
        if (!width) return false;
        const uint32_t first = static_cast<uint32_t>(in.dst.value);
        return first < wanted_end && static_cast<uint64_t>(first) + width > live.first_vgpr;
    });
}

uint32_t conservative_scalar_write_width(const Rdna2Inst& instruction) {
    if (instruction.dst.kind != OperandKind::SGPR &&
        instruction.dst.kind != OperandKind::Special)
        return 0u;
    if (instruction.fmt == Rdna2Format::SMEM) {
        switch (instruction.opcode) {
        case kSmemOpcodeLoadDword: case kSmemOpcodeBufferLoadDword: return 1u;
        case kSmemOpcodeLoadDwordX2: case kSmemOpcodeBufferLoadDwordX2: return 2u;
        case kSmemOpcodeLoadDwordX4: case kSmemOpcodeBufferLoadDwordX4: return 4u;
        case kSmemOpcodeLoadDwordX8: case kSmemOpcodeBufferLoadDwordX8: return 8u;
        case kSmemOpcodeLoadDwordX16: case kSmemOpcodeBufferLoadDwordX16: return 16u;
        default: return 1u;
        }
    }
    if (instruction.fmt == Rdna2Format::SOP1 || instruction.fmt == Rdna2Format::SOP2)
        return 2u;
    if (instruction.fmt == Rdna2Format::VOPC) return 1u;
    return 1u;
}

bool scalar_range_unchanged(const std::vector<Rdna2Inst>& instructions,
                            uint32_t begin_pc, uint32_t end_pc,
                            uint32_t first_sgpr, uint32_t count) {
    const uint64_t wanted_end = static_cast<uint64_t>(first_sgpr) + count;
    return std::none_of(instructions.begin(), instructions.end(), [&](const Rdna2Inst& in) {
        if (in.pc <= begin_pc || in.pc >= end_pc) return false;
        const uint32_t width = conservative_scalar_write_width(in);
        if (width && in.dst.value >= 0) {
            const uint32_t first = static_cast<uint32_t>(in.dst.value);
            if (first < wanted_end && static_cast<uint64_t>(first) + width > first_sgpr)
                return true;
        }
        if ((in.sdst.kind == OperandKind::SGPR || in.sdst.kind == OperandKind::Special) &&
            in.sdst.value >= 0) {
            const uint32_t first = static_cast<uint32_t>(in.sdst.value);
            return first < wanted_end && static_cast<uint64_t>(first) + 1u > first_sgpr;
        }
        return false;
    });
}

bool state_liveness_matches(const std::vector<Rdna2Inst>& instructions,
                            const DescriptorRangeShape& shape) {
    if (!packets_match(instructions, shape.exec_writers)) return false;
    size_t seen_exec_writers = 0u;
    for (const Rdna2Inst& instruction : instructions) {
        if (!rdna2_instruction_may_change_exec(instruction)) continue;
        const auto expected = std::find_if(
            shape.exec_writers.begin(), shape.exec_writers.end(),
            [&](const PacketSite& site) { return site.pc == instruction.pc; });
        if (expected == shape.exec_writers.end() || !packet_is(instruction, *expected))
            return false;
        ++seen_exec_writers;
    }
    if (seen_exec_writers != shape.exec_writers.size()) return false;

    for (const VgprLiveRange& live : shape.value_liveness)
        if (!vgpr_range_unchanged(instructions, live)) return false;
    for (const AddressPairLiveRange& address : shape.address_liveness) {
        if (!vgpr_range_unchanged(instructions, {
                address.low_producer_pc, address.access_pc, address.low_vgpr, 1u}) ||
            !vgpr_range_unchanged(instructions, {
                address.high_producer_pc, address.access_pc, address.low_vgpr + 1u, 1u}))
            return false;
    }
    for (const ScalarLiveRange& live : shape.scalar_liveness)
        if (!scalar_range_unchanged(
                instructions, live.begin_pc, live.end_pc,
                live.first_sgpr, live.count))
            return false;
    return true;
}

enum class ResidualInput : uint8_t {
    MainByte0,
    MainByte1,
    Zero,
    Main60,
    Main64,
    Main68,
    LoadedByte1,
    LoadedByte2,
};

enum class LanePredicate : uint8_t {
    Always,
    LoadedByte2Is255,
    LoadedByte2Is255AndMain68Nonzero,
};

struct ResidualObligation {
    uint32_t pc;
    ResidualInput input;
    uint32_t extra;
    LanePredicate predicate = LanePredicate::Always;
};

constexpr std::array<ResidualObligation, 19> kShape642Obligations{{
    {87u, ResidualInput::MainByte0, 0u},
    {91u, ResidualInput::MainByte1, 0u},
    {204u, ResidualInput::LoadedByte2, 4u, LanePredicate::LoadedByte2Is255},
    {216u, ResidualInput::LoadedByte2, 8u, LanePredicate::LoadedByte2Is255},
    {222u, ResidualInput::LoadedByte2, 0u, LanePredicate::LoadedByte2Is255},
    {261u, ResidualInput::Main68, 4u,
     LanePredicate::LoadedByte2Is255AndMain68Nonzero},
    {265u, ResidualInput::Main68, 0u,
     LanePredicate::LoadedByte2Is255AndMain68Nonzero},
    {268u, ResidualInput::Main68, 8u,
     LanePredicate::LoadedByte2Is255AndMain68Nonzero},
    {280u, ResidualInput::Main68, 12u,
     LanePredicate::LoadedByte2Is255AndMain68Nonzero},
    {444u, ResidualInput::Main64, 4u}, {446u, ResidualInput::Main64, 0u},
    {451u, ResidualInput::Main64, 8u}, {457u, ResidualInput::Zero, 0u},
    {465u, ResidualInput::Zero, 4u}, {475u, ResidualInput::Zero, 8u},
    {516u, ResidualInput::Main68, 4u}, {518u, ResidualInput::Main68, 0u},
    {523u, ResidualInput::Main68, 8u}, {528u, ResidualInput::Main68, 12u},
}};

constexpr std::array<ResidualObligation, 20> kShape662Obligations{{
    {89u, ResidualInput::MainByte0, 0u}, {93u, ResidualInput::MainByte1, 0u},
    {170u, ResidualInput::Main64, 4u}, {195u, ResidualInput::Main64, 8u},
    {197u, ResidualInput::Main64, 0u}, {266u, ResidualInput::Main68, 4u},
    {270u, ResidualInput::Main68, 0u}, {273u, ResidualInput::Main68, 8u},
    {285u, ResidualInput::Main68, 12u}, {344u, ResidualInput::Main64, 0u},
    {351u, ResidualInput::Main64, 4u}, {365u, ResidualInput::Main64, 8u},
    {372u, ResidualInput::Zero, 0u}, {384u, ResidualInput::Zero, 4u},
    {387u, ResidualInput::Zero, 8u}, {494u, ResidualInput::Main68, 4u},
    {496u, ResidualInput::Main68, 0u}, {501u, ResidualInput::Main68, 8u},
    {506u, ResidualInput::Main68, 12u}, {560u, ResidualInput::Main60, 0u},
}};

struct SelectedLane {
    uint32_t tid;
    uint32_t main_record;
    uint32_t pointer_index;
    uint32_t first_thread;
};

struct MainSelectionRecord {
    uint32_t record;
    uint32_t first_thread;
    uint32_t thread_end;
    uint32_t pointer_index;
};

struct DecodedPointerRecord {
    uint32_t pointer_index;
    uint64_t address;
    uint32_t byte_count;
};

const DecodedPointerRecord* pointer_record_for(
        std::span<const DecodedPointerRecord> records, uint32_t pointer_index) {
    const auto found = std::find_if(records.begin(), records.end(), [&](const auto& record) {
        return record.pointer_index == pointer_index;
    });
    return found == records.end() ? nullptr : &*found;
}

bool predicate_holds(LanePredicate predicate, uint8_t q2, uint32_t main68) {
    switch (predicate) {
    case LanePredicate::Always: return true;
    case LanePredicate::LoadedByte2Is255: return q2 == 255u;
    case LanePredicate::LoadedByte2Is255AndMain68Nonzero:
        return q2 == 255u && main68 != 0u;
    }
    return false;
}

bool residuals_are_contained(const DescriptorRangeShape& shape,
                             const uint8_t* main_bytes,
                             std::span<const SelectedLane> lanes,
                             std::span<const DecodedPointerRecord> records,
                             const uint8_t* relocation_carrier,
                             size_t relocation_carrier_bytes,
                             const IndirectBufferRelocationInfo* relocation_info) {
    const std::span<const ResidualObligation> obligations = shape.dwords == 642u
        ? std::span<const ResidualObligation>(kShape642Obligations)
        : std::span<const ResidualObligation>(kShape662Obligations);
    if (obligations.size() != shape.accesses.size()) return false;
    for (size_t index = 0; index < obligations.size(); ++index)
        if (obligations[index].pc != shape.accesses[index].pc) return false;

    constexpr uint32_t kMainStride = 80u;
    for (const SelectedLane& lane : lanes) {
        const DecodedPointerRecord* record = pointer_record_for(records, lane.pointer_index);
        if (!record) return false;
        const size_t main_offset = static_cast<size_t>(lane.main_record) * kMainStride;
        const uint32_t delta = lane.tid - lane.first_thread;
        const uint32_t packed_bytes = load_u32(main_bytes, main_offset + 32u);
        const uint8_t b0 = static_cast<uint8_t>(packed_bytes);
        const uint8_t b1 = static_cast<uint8_t>(packed_bytes >> 8u);
        const uint8_t multiplier = static_cast<uint8_t>(packed_bytes >> 24u);
        const uint32_t scaled = multiplier * delta;
        const uint32_t main60 = load_u32(main_bytes, main_offset + 60u);
        const uint32_t main64 = load_u32(main_bytes, main_offset + 64u);
        const uint32_t main68 = load_u32(main_bytes, main_offset + 68u);
        const auto aligned = [&](uint32_t value) {
            return static_cast<uint32_t>(scaled + value) & ~3u;
        };
        const uint32_t first_residual = aligned(b0);
        const uint32_t second_residual = aligned(b1);
        if (static_cast<uint64_t>(first_residual) + sizeof(uint32_t) > record->byte_count ||
            static_cast<uint64_t>(second_residual) + sizeof(uint32_t) > record->byte_count)
            return false;

        uint8_t q2 = 0u;
        if (shape.dwords == 642u) {
            const uint64_t q_address = record->address + second_residual;
            const uint8_t* bytes = relocation_info
                ? indirect_buffer_relocation_payload_bytes(
                      relocation_carrier, relocation_carrier_bytes,
                      *relocation_info, q_address, sizeof(uint32_t))
                : reinterpret_cast<const uint8_t*>(
                      static_cast<uintptr_t>(q_address));
            if (!bytes) return false;
            q2 = static_cast<uint8_t>(load_u32(bytes, 0u) >> 16u);
        }

        for (const ResidualObligation& obligation : obligations) {
            // In the 642 shape pc116 is V_CMPX_LT_U32, so the pc117 arm retains exactly q2=255
            // lanes in EXEC. Pc118 then replaces v0:v3 from main+56, so pc137/245 further restricts
            // its subgroup with main+68 != 0 (not with the earlier pc95 byte extraction).
            if (!predicate_holds(obligation.predicate, q2, main68)) continue;
            uint32_t input = 0u;
            switch (obligation.input) {
            case ResidualInput::MainByte0: input = b0; break;
            case ResidualInput::MainByte1: input = b1; break;
            case ResidualInput::Zero: input = 0u; break;
            case ResidualInput::Main60: input = main60; break;
            case ResidualInput::Main64: input = main64; break;
            case ResidualInput::Main68: input = main68; break;
            case ResidualInput::LoadedByte1: return false;
            case ResidualInput::LoadedByte2: input = q2; break;
            }
            const uint32_t residual = aligned(input) + obligation.extra;
            if (static_cast<uint64_t>(residual) + sizeof(uint32_t) > record->byte_count)
                return false;
        }
    }
    return true;
}

uint64_t fingerprint_mix(uint64_t hash, uint64_t value) {
    for (uint32_t byte = 0; byte < 8u; ++byte) {
        hash ^= static_cast<uint8_t>(value >> (byte * 8u));
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t descriptor_proof_fingerprint(const IndirectPointerRelocationProof& proof,
                                      const DescriptorRangeShape& shape) {
    uint64_t hash = 1469598103934665603ull;
    hash = fingerprint_mix(hash, proof.schema_version);
    hash = fingerprint_mix(hash, static_cast<uint32_t>(proof.bound_kind));
    hash = fingerprint_mix(hash, static_cast<uint32_t>(proof.guard_kind));
    hash = fingerprint_mix(hash, proof.source_fetch_pc);
    hash = fingerprint_mix(hash, proof.source_words[0]);
    hash = fingerprint_mix(hash, proof.source_words[1]);
    hash = fingerprint_mix(hash, proof.source_result_vgpr);
    hash = fingerprint_mix(hash, proof.source_stride);
    hash = fingerprint_mix(hash, proof.pointer_byte_offset);
    hash = fingerprint_mix(hash, proof.source_record_index_vgpr);
    hash = fingerprint_mix(hash, static_cast<uint32_t>(proof.source_address_kind));
    hash = fingerprint_mix(hash, proof.record_count);
    hash = fingerprint_mix(hash, proof.max_footprint_bytes);
    hash = fingerprint_mix(hash, proof.accesses.size());
    for (const IndirectPointerAccessProof& access : proof.accesses) {
        hash = fingerprint_mix(hash, access.pc);
        hash = fingerprint_mix(hash, access.words[0]);
        hash = fingerprint_mix(hash, access.words[1]);
        hash = fingerprint_mix(hash, access.address_vgpr);
        hash = fingerprint_mix(hash, access.immediate_byte_offset);
        hash = fingerprint_mix(hash, access.component_bytes);
        hash = fingerprint_mix(hash, access.components);
    }
    for (const PacketSite& site : shape.residual_slice) {
        hash = fingerprint_mix(hash, site.pc);
        hash = fingerprint_mix(hash, site.word0);
        hash = fingerprint_mix(hash, site.word1);
        hash = fingerprint_mix(hash, site.dwords);
    }
    const std::span<const ResidualObligation> obligations = shape.dwords == 642u
        ? std::span<const ResidualObligation>(kShape642Obligations)
        : std::span<const ResidualObligation>(kShape662Obligations);
    for (const ResidualObligation& obligation : obligations) {
        hash = fingerprint_mix(hash, obligation.pc);
        hash = fingerprint_mix(hash, static_cast<uint32_t>(obligation.input));
        hash = fingerprint_mix(hash, obligation.extra);
        hash = fingerprint_mix(hash, static_cast<uint32_t>(obligation.predicate));
    }
    return hash;
}

const DescriptorRangeShape* match_shape(const uint32_t* code, size_t dwords,
                                        const ComputeShaderConfig& config,
                                        std::vector<Rdna2Inst>& instructions) {
    for (const DescriptorRangeShape& shape : kShapes) {
        if (dwords < shape.dwords || config.user_sgprs.size() != shape.user_sgprs)
            continue;
        instructions.clear();
        // Registered AGC spans can include allocation padding after the executable S_ENDPGM.
        // Decode and authorize only the exact executable prefix; bytes before END remain fully
        // structural, while unreachable trailing storage neither grants nor fragments authority.
        if (rdna2_walk(code, shape.dwords, instructions) != shape.dwords ||
            instructions.empty() || !instructions.back().is_end ||
            instructions.back().pc != shape.dwords - 1u ||
            !packets_match(instructions, shape.selection_loop) ||
            !packets_match(instructions, shape.authority) ||
            !packets_match(instructions, shape.accesses) ||
            !packets_match(instructions, shape.residual_slice) ||
            !control_inventory_matches(instructions, shape) ||
            !state_liveness_matches(instructions, shape) ||
            shape.residual_slice.empty())
            continue;
        const size_t flat_count = static_cast<size_t>(std::count_if(
            instructions.begin(), instructions.end(),
            [](const Rdna2Inst& instruction) {
                return instruction.fmt == Rdna2Format::FLAT;
            }));
        if (flat_count == shape.accesses.size()) return &shape;
    }
    return nullptr;
}

} // namespace

bool analyze_rdna2_descriptor_pointer_range(
        const uint32_t* code, size_t dwords,
        const ComputeShaderConfig& config,
        const ShaderResourceTable& resources,
        IndirectPointerRelocationProof& proof) {
    proof = {};
    constexpr uint32_t kMaxProofInvocations = 65536u;
    if (!code || config.local_x != 32u || config.local_y != 1u ||
        config.local_z != 1u || config.wave_size != 32u ||
        !config.exact_thread_extent || !config.threads_x ||
        config.threads_x > kMaxProofInvocations ||
        config.threads_y != 1u || config.threads_z != 1u ||
        !config.tgid_x_en || config.tgid_y_en || config.tgid_z_en ||
        config.tg_size_en || config.tidig_comp_cnt != 0u)
        return false;

    std::vector<Rdna2Inst> instructions;
    const DescriptorRangeShape* shape = match_shape(code, dwords, config, instructions);
    if (!shape) return false;

    constexpr uint32_t kOuterBytes = 2992u;
    constexpr uint32_t kOuterStride = 16u;
    constexpr uint32_t kMainBytes = 1310720u;
    constexpr uint32_t kMainStride = 80u;
    constexpr uint32_t kPointerBytes = 2097152u;
    constexpr uint32_t kPointerStride = 16u;
    constexpr uint32_t kOuterRangeOffset = 20u;
    constexpr uint32_t kMainRangeOffset = 36u;
    constexpr uint32_t kMainPointerIndexOffset = 48u;

    const ShaderResource* outer = unique_resource_at(resources, shape->outer_fetch_pc);
    const ShaderResource* main_range = unique_resource_at(resources, shape->main_range_pc);
    const ShaderResource* main_index = unique_resource_at(resources, shape->main_index_pc);
    const ShaderResource* pointer = unique_resource_at(resources, shape->pointer_fetch_pc);
    if (!resource_is(outer, shape->outer_fetch_pc, kOuterBytes, kOuterStride) ||
        !resource_is(main_range, shape->main_range_pc, kMainBytes, kMainStride) ||
        !resource_is(main_index, shape->main_index_pc, kMainBytes, kMainStride) ||
        !resource_is(pointer, shape->pointer_fetch_pc, kPointerBytes, kPointerStride) ||
        main_range->gpu_addr != main_index->gpu_addr)
        return false;
    const uint8_t* outer_bytes = complete_bytes(*outer);
    const uint8_t* main_range_bytes = complete_bytes(*main_range);
    const uint8_t* main_index_bytes = complete_bytes(*main_index);
    const uint8_t* pointer_bytes = complete_bytes(*pointer);
    if (!outer_bytes || !main_range_bytes || !main_index_bytes || !pointer_bytes ||
        std::memcmp(main_range_bytes, main_index_bytes, kMainBytes) != 0)
        return false;
    const size_t carrier_header = static_cast<size_t>(pointer->size);
    const bool relocation_carrier_candidate = pointer->host_data &&
        carrier_header <= pointer->host_data_size &&
        kIndirectBufferRelocationHeaderBytes <=
            pointer->host_data_size - carrier_header &&
        load_u32(pointer->host_data, carrier_header) ==
            kIndirectPointerDescriptorRangeLayout.tag &&
        load_u32(pointer->host_data, carrier_header + sizeof(uint32_t)) ==
            kIndirectPointerDescriptorRangeLayout.version;
    IndirectBufferRelocationInfo inspected_relocation;
    const bool has_relocation_carrier = relocation_carrier_candidate &&
        inspect_indirect_buffer_relocation(
            *pointer, pointer->host_data, pointer->host_data_size,
            kIndirectPointerDescriptorRangeLayout, inspected_relocation);
    // A buffer blob may expose a larger host span than the logical V#. Trailing ordinary bytes do
    // not turn it into a carrier, while a header that explicitly advertises v3 must parse fully.
    if (relocation_carrier_candidate && !has_relocation_carrier)
        return false;

    const uint32_t start = load_u32(outer_bytes, kOuterRangeOffset);
    const uint32_t count = load_u32(outer_bytes, kOuterRangeOffset + sizeof(uint32_t));
    const uint32_t main_records = kMainBytes / kMainStride;
    if (!count || start > main_records || count > main_records - start ||
        count > kIndirectPointerDescriptorRangeLayout.max_records)
        return false;

    std::vector<MainSelectionRecord> main_selections;
    main_selections.reserve(count);
    for (uint32_t ordinal = 0; ordinal < count; ++ordinal) {
        const uint32_t record = start + ordinal;
        const size_t record_offset = static_cast<size_t>(record) * kMainStride;
        const uint32_t thread_count = load_u32(
            main_range_bytes, record_offset + kMainRangeOffset);
        const uint32_t first_thread = load_u32(
            main_range_bytes, record_offset + kMainRangeOffset + sizeof(uint32_t));
        if (thread_count > UINT32_MAX - first_thread) return false;
        const uint32_t thread_end = first_thread + thread_count;
        const uint32_t pointer_index = load_u32(
            main_index_bytes, record_offset + kMainPointerIndexOffset);
        if (thread_count)
            main_selections.push_back({record, first_thread, thread_end, pointer_index});
    }
    std::vector<SelectedLane> selected_lanes;
    selected_lanes.reserve(config.threads_x);
    for (uint32_t invocation = 0; invocation < config.threads_x; ++invocation) {
        const uint32_t group = invocation / config.local_x;
        const uint32_t local_tid = invocation % config.local_x;
        const uint64_t logical_tid64 =
            (static_cast<uint64_t>(group) << shape->tid_group_shift) + local_tid;
        if (logical_tid64 > UINT32_MAX) return false;
        const uint32_t logical_tid = static_cast<uint32_t>(logical_tid64);
        const MainSelectionRecord* selected = nullptr;
        for (const MainSelectionRecord& candidate : main_selections) {
            if (logical_tid < candidate.first_thread || logical_tid >= candidate.thread_end)
                continue;
            // The matched pc13..43 loop retains one main-record ordinal per lane. Reject overlaps
            // rather than choosing an ordering convention that the host has not proved.
            if (selected) return false;
            selected = &candidate;
        }
        if (!selected) continue;
        if (selected->pointer_index >= kPointerBytes / kPointerStride) return false;
        selected_lanes.push_back({
            logical_tid, selected->record, selected->pointer_index,
            selected->first_thread,
        });
    }
    // Unmatched padding lanes are removed from EXEC by the exact pc44/48 CMPX sequence and never
    // reach the pointer fetch or a GLOBAL consumer.  At least one selected lane is still required.
    if (selected_lanes.empty()) return false;

    std::vector<uint32_t> pointer_indices;
    pointer_indices.reserve(count);
    for (const SelectedLane& lane : selected_lanes)
        pointer_indices.push_back(lane.pointer_index);
    std::sort(pointer_indices.begin(), pointer_indices.end());
    pointer_indices.erase(
        std::unique(pointer_indices.begin(), pointer_indices.end()), pointer_indices.end());
    if (pointer_indices.empty() ||
        pointer_indices.size() > kIndirectPointerDescriptorRangeLayout.max_records)
        return false;

    IndirectPointerRelocationProof candidate;
    candidate.schema_version = kIndirectPointerProofSchema;
    candidate.bound_kind = IndirectPointerBoundKind::DescriptorRange;
    candidate.guard_kind = IndirectPointerGuardKind::None;
    candidate.source_fetch_pc = shape->pointer_fetch_pc;
    const Rdna2Inst* source_instruction = instruction_at(
        instructions, shape->pointer_fetch_pc);
    if (!source_instruction || source_instruction->len_dwords != 2u) return false;
    candidate.source_words = {
        source_instruction->words[0], source_instruction->words[1],
    };
    candidate.source_result_vgpr = shape->pointer_vgpr;
    candidate.source_stride = kPointerStride;
    candidate.pointer_byte_offset = 0u;
    candidate.source_record_index_vgpr = 2u;
    candidate.source_address_kind =
        IndirectBufferRelocationRecord::SourceAddressKind::BufferDescriptorBase48;
    candidate.record_count = static_cast<uint32_t>(pointer_indices.size());
    candidate.records.reserve(pointer_indices.size());
    std::vector<DecodedPointerRecord> decoded_records;
    decoded_records.reserve(pointer_indices.size());
    for (uint32_t pointer_index : pointer_indices) {
        const uint32_t source_offset = pointer_index * kPointerStride;
        const uint32_t word0 = load_u32(pointer_bytes, source_offset);
        const uint32_t word1 = load_u32(pointer_bytes, source_offset + 4u);
        const uint32_t num_records = load_u32(pointer_bytes, source_offset + 8u);
        const uint64_t address = static_cast<uint64_t>(word0) |
            (static_cast<uint64_t>(word1 & 0xffffu) << 32u);
        // The exact shader sign-extends WORD_0 of V# word1. Prosper's guest mappings are in the
        // positive canonical half; do not silently reinterpret a bit-47-set descriptor.
        if (!address || (address & (1ull << 47u))) return false;
        const uint32_t stride = (word1 >> 16u) & 0x3fffu;
        const uint64_t byte_count64 = stride
            ? static_cast<uint64_t>(num_records) * stride
            : static_cast<uint64_t>(num_records);
        if (!byte_count64 || byte_count64 > UINT32_MAX ||
            byte_count64 > kIndirectPointerMaxBindingBytes ||
            address > UINT64_MAX - byte_count64 ||
            (!has_relocation_carrier &&
             !guest_readable(address, static_cast<uint32_t>(byte_count64))))
            return false;
        const uint32_t byte_count = static_cast<uint32_t>(byte_count64);
        candidate.records.push_back({
            source_offset, address, byte_count,
            IndirectBufferRelocationRecord::SourceAddressKind::BufferDescriptorBase48,
        });
        decoded_records.push_back({pointer_index, address, byte_count});
        candidate.max_footprint_bytes = std::max(
            candidate.max_footprint_bytes, byte_count);
    }

    IndirectBufferRelocationInfo parsed_relocation;
    if (has_relocation_carrier &&
        !parse_indirect_buffer_relocation(
            *pointer, pointer->host_data, pointer->host_data_size,
            kIndirectPointerDescriptorRangeLayout, candidate.records,
            parsed_relocation))
        return false;
    if (!residuals_are_contained(
            *shape, main_range_bytes, selected_lanes, decoded_records,
            has_relocation_carrier ? pointer->host_data : nullptr,
            has_relocation_carrier ? pointer->host_data_size : 0u,
            has_relocation_carrier ? &parsed_relocation : nullptr))
        return false;

    candidate.accesses.reserve(shape->accesses.size());
    for (const PacketSite& site : shape->accesses) {
        const Rdna2Inst* instruction = instruction_at(instructions, site.pc);
        if (!instruction || instruction->fmt != Rdna2Format::FLAT ||
            instruction->flat_segment != 2u ||
            instruction->opcode != kMubufOpcodeLoadDword ||
            instruction->src[0].kind != OperandKind::VGPR ||
            instruction->src[0].value < 0 ||
            instruction->src[1].kind != OperandKind::Special ||
            instruction->src[1].value != 125 || instruction->literal != 0u ||
            instruction->flat_glc || instruction->flat_slc ||
            instruction->flat_dlc || instruction->flat_lds)
            return false;
        candidate.accesses.push_back({
            site.pc, {site.word0, site.word1},
            static_cast<uint32_t>(instruction->src[0].value),
            0u, sizeof(uint32_t), 1u,
        });
    }

    candidate.fingerprint = descriptor_proof_fingerprint(candidate, *shape);
    const uint32_t fingerprint_low = static_cast<uint32_t>(candidate.fingerprint);
    const uint32_t fingerprint_high = static_cast<uint32_t>(candidate.fingerprint >> 32u);
    candidate.witness_words = {
        candidate.schema_version,
        fingerprint_low,
        fingerprint_high,
        kIndirectPointerDescriptorRangeTag ^ candidate.schema_version ^
            fingerprint_low ^ fingerprint_high,
    };
    proof = std::move(candidate);
    return true;
}

bool rdna2_indirect_pointer_source(
        const IndirectPointerRelocationProof& proof,
        const Rdna2Inst& instruction) {
    return proof.bound_kind == IndirectPointerBoundKind::DescriptorRange &&
           proof.source_fetch_pc != UINT32_MAX &&
           proof.source_result_vgpr != UINT32_MAX &&
           proof.source_record_index_vgpr != UINT32_MAX &&
           instruction.pc == proof.source_fetch_pc &&
           instruction.len_dwords == 2u &&
           instruction.words[0] == proof.source_words[0] &&
           instruction.words[1] == proof.source_words[1];
}

} // namespace prosper::gpu
