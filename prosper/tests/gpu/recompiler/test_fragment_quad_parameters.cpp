#include "gpu/recompiler/fragment_quad_parameters.hpp"
#include "gpu/recompiler/fragment_raster_interface.hpp"
#include <gtest/gtest.h>
#include <map>
#include <optional>
#include <tuple>

namespace {
using namespace prosper::gpu;
struct Origin {
    uint32_t attribute = UINT32_MAX, vertex = UINT32_MAX;
    bool operator==(const Origin&) const = default;
};
struct Value {
    Origin first, second;
    bool difference = false;
    bool operator==(const Value&) const = default;
};
void original_planes(const std::vector<uint32_t>& module,
                     const FragmentInterpolationLayout& layout) {
    // An independent symbolic SOURCE oracle: each flat coefficient store must name actual
    // per-vertex Input loads, or their exact A1-A0/A2-A0 subtraction. Final smooth interpolation
    // and host raster payload never participate in this graph.
    std::map<uint32_t, uint32_t> location, storage, constants;
    std::map<uint32_t, Origin> pointers;
    std::map<uint32_t, Value> values;
    std::map<uint32_t, Value> outputs;
    uint32_t emits = 0;
    for (size_t pc = 5; pc < module.size();) {
        const uint32_t count = module[pc] >> 16, op = module[pc] & 0xffffu;
        ASSERT_NE(count, 0u);
        ASSERT_LE(count, module.size() - pc);
        const auto* in = module.data() + pc;
        if (op == 71 && count == 4 && in[2] == 30) location[in[1]] = in[3];
        if (op == 59 && count >= 4) storage[in[2]] = in[3];
        if (op == 43 && count == 4) constants[in[2]] = in[3];
        if (op == 65 && count == 5 && storage[in[3]] == 1 && location.contains(in[3]) &&
            constants.contains(in[4]))
            pointers[in[2]] = {location.at(in[3]), constants.at(in[4])};
        if (op == 61 && count == 4 && pointers.contains(in[3]))
            values[in[2]] = {pointers.at(in[3]), {}, false};
        if (op == 131 && count == 5 && values.contains(in[3]) && values.contains(in[4]))
            values[in[2]] = {values.at(in[3]).first, values.at(in[4]).first, true};
        if (op == 62 && count == 3 && location.contains(in[1]) && storage[in[1]] == 3 &&
            values.contains(in[2]))
            outputs[location.at(in[1])] = values.at(in[2]);
        if (op == 218) {
            ++emits;
            for (uint32_t attribute = 0; attribute < 32; ++attribute) {
                if (!(layout.attribute_mask & (1u << attribute))) continue;
                for (uint32_t selector = 0; selector < 3; ++selector) {
                    const auto at = layout.parameter_locations[attribute][selector];
                    ASSERT_TRUE(outputs.contains(at)) << "before original vertex emit";
                    const Value expected =
                        selector == 2 ? Value{{attribute, 0}, {}, false}
                                      : Value{{attribute, selector + 1}, {attribute, 0}, true};
                    EXPECT_EQ(outputs.at(at), expected) << attribute << ':' << selector;
                }
            }
            outputs.clear();
        }
        pc += count;
    }
    EXPECT_EQ(emits, 3u);
}
}   // namespace
TEST(FragmentQuadParameters, OrdinaryP1P2RequestsThreeGenuinePerVertexPlanes) {
    const uint32_t original[]{0xc8080000u, 0xc8090001u, 0xf8001801u, 2u, 0xbf810000u};
    PixelInputMapping pixel;
    pixel.valid_mask = 1;
    const PixelSystemInputMapping system{2, 2};
    const auto native =
        fragment_interpolation_layout(original, std::size(original), &system, &pixel);
    ASSERT_TRUE(native.valid);
    EXPECT_FALSE(native.requires_geometry);
    EXPECT_EQ(native.smooth_mask, 1u);
    const auto complete = fragment_quad_parameter_layout(native, system, 128);
    ASSERT_TRUE(complete.rejection.empty()) << complete.rejection;
    EXPECT_TRUE(complete.layout.requires_geometry);
    EXPECT_EQ(complete.layout.smooth_mask, 1u);
    EXPECT_EQ(complete.layout.parameter_locations[0], (std::array<uint32_t, 3>{1, 2, 3}));
    EXPECT_EQ(complete.layout.system_locations[1], 4u);
    const auto geometry = recompile_interpolation_geometry(complete.layout, false, false, {}, true);
    ASSERT_FALSE(geometry.empty());
    original_planes(geometry, complete.layout);
    const auto interface = fragment_raster_output_interface(geometry, 3);
    ASSERT_TRUE(interface.available);
    EXPECT_EQ(interface.components, 25u)
        << "position4 + primitive1 + original4 + parameter12 + center4";
    EXPECT_EQ(interface.location_end, 20u);
    EXPECT_EQ(interface.float4_locations, 31u);
    EXPECT_EQ(interface.output_vertices, 3u);
    EXPECT_TRUE(recompile_interpolation_geometry(native).empty())
        << "the unchanged native default does not request the new producer";
}

TEST(FragmentQuadParameters, IncompleteOrCyclicSelectedInterfacesCannotProveDeviceBudget) {
    FragmentInterpolationLayout original;
    original.attribute_mask = original.smooth_mask = 1;
    const auto complete = fragment_quad_parameter_layout(original, {}, 128);
    ASSERT_TRUE(complete.rejection.empty());
    const auto source = recompile_interpolation_geometry(complete.layout, false, false, {}, true);
    ASSERT_FALSE(source.empty());
    const auto original_interface = fragment_raster_output_interface(source, 3);
    ASSERT_TRUE(original_interface.available);
    for (uint32_t kind = 0; kind < 3; ++kind) {
        auto broken = source;
        bool changed = false;
        for (size_t pc = 5; pc < broken.size();) {
            const uint32_t count = broken[pc] >> 16, op = broken[pc] & 0xffffu;
            ASSERT_NE(count, 0u);
            ASSERT_LE(count, broken.size() - pc);
            if ((kind == 0 && op == 23) || (kind == 1 && op == 15) ||
                (kind == 2 && op == 32 && count == 4 && broken[pc + 2] == 3)) {
                if (kind == 0) broken[pc + 2] = broken[pc + 1];   // vector type cycle
                if (kind == 1) broken[pc] = 0;   // truncated/zero-length instruction
                if (kind == 2) broken[pc + 3] = broken[3] + 1;   // no declared pointee
                changed = true;
                break;
            }
            pc += count;
        }
        ASSERT_TRUE(changed);
        EXPECT_FALSE(fragment_raster_output_interface(broken, 3).available) << kind;
    }
    EXPECT_FALSE(fragment_raster_output_interface(source, 0).available)
        << "a GS interface cannot certify a selected vertex module";
}
TEST(FragmentQuadParameters, EveryOriginalAttributeAndEnabledAddrHoleKeepsItsLocation) {
    FragmentInterpolationLayout original;
    original.attribute_mask = original.smooth_mask = (1u << 2) | (1u << 7);
    const auto complete = fragment_quad_parameter_layout(original, {0x4a, 0x52}, 128);
    ASSERT_TRUE(complete.rejection.empty());
    EXPECT_EQ(complete.layout.parameter_locations[2], (std::array<uint32_t, 3>{8, 9, 10}));
    EXPECT_EQ(complete.layout.parameter_locations[7], (std::array<uint32_t, 3>{11, 12, 13}));
    EXPECT_EQ(complete.layout.system_locations[1], 14u);
    EXPECT_EQ(complete.layout.system_locations[6], 15u);
    for (uint32_t field : {0u, 2u, 3u, 4u, 5u})
        EXPECT_EQ(complete.layout.system_locations[field], UINT32_MAX);
    for (uint32_t attribute = 0; attribute < 32; ++attribute)
        if (attribute != 2 && attribute != 7)
            EXPECT_EQ(complete.layout.parameter_locations[attribute],
                      (std::array<uint32_t, 3>{UINT32_MAX, UINT32_MAX, UINT32_MAX}));
    const auto geometry = recompile_interpolation_geometry(complete.layout);
    ASSERT_FALSE(geometry.empty());
    original_planes(geometry, complete.layout);
}
TEST(FragmentQuadParameters, ActualDeviceBudgetDoesNotTruncateTheOriginalParameterContract) {
    FragmentInterpolationLayout original;
    original.attribute_mask = original.smooth_mask = UINT32_MAX;
    const auto portable = fragment_quad_parameter_layout(original, {}, 128);
    EXPECT_EQ(portable.rejection, "fragment-quad-enabled-input-component-budget");
    EXPECT_FALSE(portable.layout.valid);
    const auto wider = fragment_quad_parameter_layout(original, {}, 512);
    ASSERT_TRUE(wider.rejection.empty());
    EXPECT_EQ(wider.layout.attribute_mask, UINT32_MAX);
    EXPECT_EQ(wider.layout.parameter_locations[31], (std::array<uint32_t, 3>{125, 126, 127}));
    const auto exact = fragment_quad_parameter_layout(original, {}, 508);
    EXPECT_EQ(exact.rejection, "fragment-quad-enabled-input-component-budget");
    for (uint32_t limit : {0u, 129u})
        EXPECT_EQ(fragment_quad_parameter_layout(original, {}, limit).rejection,
                  "fragment-quad-enabled-input-component-limit-unavailable");
}
TEST(FragmentQuadParameters, InvalidOriginalMasksCannotProduceIndependentParameterPlanes) {
    for (uint32_t field = 0; field < 3; ++field) {
        FragmentInterpolationLayout original;
        if (field == 0) original.smooth_mask = 1;
        if (field == 1) original.flat_mask = 1;
        if (field == 2) original.passthrough_mask = 1;
        const auto result = fragment_quad_parameter_layout(original, {}, 128);
        EXPECT_EQ(result.rejection, "fragment-quad-original-parameter-layout-unavailable");
        EXPECT_FALSE(result.layout.valid);
    }
}
