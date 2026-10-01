#pragma once

// Project-owned counterexamples for #4007. Independent of the recompiler and proof implementation.
// The paired index/branch arms change exactly one operand. With a false lower half and true upper
// half, Any(input) differs between one Wave64 and two host32 groups. Choosing between distinct
// initialized elements, stores, copies, image samples or an early return therefore is observable.
// A fixed index/condition makes the SAME vote dead. spv_validate checks the baseline Vulkan1.1
// corpus; Sized copies/flag atomics are separately marked parser-only (Addresses/Kernel).
// External non-atomic writers use distinct FragCoord-derived locations: the concrete witness is
// a single primitive covering a single-sampled 64x1 framebuffer, with 64x1 storage images and a
// 128-element output buffer. Real pixel x owns buffer slots 2*x/2*x+1 and image coordinate (x,0),
// so neither choice races with another invocation. Host initializes buffers before the draw.
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace prosper::test::wave_width {

enum class Shape {
    BufferRead, ExternalStore, LocalRead, LocalStore, CopySourceExternal,
    CopyDestinationExternal, CopySourceLocal, CopyDestinationLocal, GuardedLocalCopy,
    EarlyReturn, AtomicAddress, AtomicStoreAddress, ImageIdentity, ImageSample, DrefSample, OpaqueCall,
    CalleeReturn, CopySizeParserOnly, FlagClearParserOnly,
};

struct Fixture {
    std::string name;
    std::vector<uint32_t> words;
    bool independent;
    bool strict_vulkan;
};

inline void instruction(std::vector<uint32_t>& words, uint32_t opcode,
                        std::initializer_list<uint32_t> operands) {
    words.push_back((static_cast<uint32_t>(operands.size() + 1) << 16) | opcode);
    words.insert(words.end(), operands);
}

inline void extension(std::vector<uint32_t>& words, const std::string& name) {
    std::vector<uint32_t> literal((name.size() + 4) / 4, 0);
    for (size_t i = 0; i < name.size(); ++i)
        literal[i / 4] |= static_cast<uint32_t>(static_cast<unsigned char>(name[i])) << (8 * (i % 4));
    words.push_back((static_cast<uint32_t>(literal.size() + 1) << 16) | 10u);
    words.insert(words.end(), literal.begin(), literal.end());
}

inline std::vector<uint32_t> module(Shape shape, bool dependent, bool uniform_vote = false,
                                    bool mask_literal_matches_vote_id = false) {
    std::vector<uint32_t> w{0x07230203u, 0x00010300u, 0, 96, 0};
    const auto op = [&](uint32_t code, std::initializer_list<uint32_t> args) {
        instruction(w, code, args);
    };
    const bool image = shape == Shape::ImageIdentity || shape == Shape::ImageSample;
    const bool dref = shape == Shape::DrefSample;
    const bool image_resource = image || dref;
    const bool external_writer = shape == Shape::ExternalStore ||
                                 shape == Shape::CopyDestinationExternal;
    const bool coordinates = image || external_writer;
    const bool atomic = shape == Shape::AtomicAddress || shape == Shape::AtomicStoreAddress ||
                        shape == Shape::FlagClearParserOnly;
    op(17, {1});                         // Shader
    op(17, {61}); op(17, {62});           // GroupNonUniform, GroupNonUniformVote
    if (shape == Shape::ImageIdentity) {
        // A vote need not be DRAW-uniform. NonUniform descriptor indexing makes both image
        // choices defined, rather than passing the validator with an undefined dynamic index.
        op(17, {5301}); op(17, {5309}); // ShaderNonUniform, StorageImageArrayNonUniformIndexing
        extension(w, "SPV_EXT_descriptor_indexing");
    }
    if (shape == Shape::ImageSample) op(17, {27});    // StorageImageMultisample
    op(14, {0, 1});                      // Logical GLSL450
    if (coordinates) op(15, {4, 40, 0x6e69616du, 0, 22, 23, 72});
    else op(15, {4, 40, 0x6e69616du, 0, 22, 23}); // Fragment %40 "main"
    op(16, {40, 7});                     // OriginUpperLeft
    op(71, {22, 30, 0}); op(71, {23, 30, 0}); // Location0
    op(71, {15, 6, 4});                  // ArrayStride4
    op(71, {16, 2}); op(72, {16, 0, 35, 0}); // Block, member Offset0
    op(71, {24, 34, 0}); op(71, {24, 33, 0}); // DescriptorSet0 Binding0
    if (image_resource) {
        op(71, {33, 34, 0}); op(71, {33, 33, 1});
    }
    if (dref) { op(71, {82, 34, 0}); op(71, {82, 33, 4}); }
    if (shape == Shape::ImageIdentity) {
        op(71, {53, 5300}); op(71, {54, 5300}); op(71, {56, 5300}); // NonUniform
    }
    if (atomic) {
        op(71, {34, 6, 4}); op(71, {35, 2}); op(72, {35, 0, 35, 0});
        op(71, {38, 34, 0}); op(71, {38, 33, 2});
    }
    if (coordinates) op(71, {72, 11, 15}); // BuiltIn FragCoord
    if (external_writer) {
        op(71, {66, 6, 4}); op(71, {67, 2}); op(72, {67, 0, 35, 0});
        op(71, {69, 34, 0}); op(71, {69, 33, 3});
    }
    op(19, {1}); op(20, {2}); op(21, {3, 32, 0}); op(22, {4, 32});
    op(33, {5, 1});                     // void()
    op(32, {6, 1, 4}); op(32, {7, 3, 4}); // Input/Output float pointers
    op(43, {3, 8, 0}); op(43, {3, 9, 1}); op(43, {3, 10, 2}); op(43, {3, 11, 3});
    op(43, {4, 12, 0}); op(43, {4, 13, 0x3f800000u}); op(41, {2, 14});
    op(28, {15, 4, 10}); op(30, {16, 15});
    op(32, {17, 12, 16}); op(32, {18, 12, 4});
    op(32, {19, 7, 15}); op(32, {20, 7, 4});
    op(44, {15, 21, 12, 13});            // local array is initialized {0,1}
    op(59, {6, 22, 1}); op(59, {7, 23, 3}); op(59, {17, 24, 12});
    if (coordinates || dref) op(23, {25, 4, 4});
    if (coordinates) {
        op(32, {71, 1, 25}); op(59, {71, 72, 1});
    }
    if (external_writer) {
        op(43, {3, 65, 128}); op(28, {66, 4, 65}); op(30, {67, 66});
        op(32, {68, 12, 67}); op(59, {68, 69, 12});
    }
    if (image_resource) {
        op(23, {26, 3, 2});
        op(44, {26, 27, 8, 8}); op(44, {25, 28, 13, 13, 13, 13});
        op(25, {29, 4, 1, dref ? 1u : 0u, 0, shape == Shape::ImageSample ? 1u : 0u,
                dref ? 1u : 2u, dref ? 0u : 1u});
        op(28, {30, 29, 10}); op(32, {31, 0, 30}); op(32, {32, 0, 29});
        op(59, {31, 33, 0});            // two rgba32f storage images
    }
    if (dref) {
        // A read-only depth image holding 0.5 and a comparison sampler distinguish refs 0/1.
        op(26, {79}); op(27, {80, 29}); op(32, {81, 0, 79}); op(59, {81, 82, 0});
        op(23, {86, 4, 2}); op(44, {86, 84, 12, 12});
    }
    if (atomic) {
        op(28, {34, 3, 10}); op(30, {35, 34});
        op(32, {36, 12, 35}); op(32, {37, 12, 3}); op(59, {36, 38, 12});
    }
    if (shape == Shape::OpaqueCall) op(33, {39, 1, 2}); // void(bool)
    if (shape == Shape::CalleeReturn) op(33, {39, 2}); // bool()
    op(54, {1, 40, 0, 5}); op(248, {45}); // main and first block
    op(59, {19, 41, 7, 21}); op(59, {20, 42, 7, 12}); op(59, {20, 43, 7, 13});
    op(61, {4, 50, 22}); op(186, {2, 51, 50, 12}); // input > 0
    const uint32_t vote = mask_literal_matches_vote_id ? 64u : 52u;
    op(335, {2, vote, 11, uniform_vote ? 14u : 51u}); // Any at Subgroup scope
    op(169, {3, 53, vote, 9, 8});        // vote-selected index 1/0
    op(62, {23, 12});                    // every path has a defined initial colour
    if (coordinates) {
        op(61, {25, 78, 72}); op(81, {4, 73, 78, 0}); op(109, {3, 74, 73});
        if (external_writer) op(132, {3, 75, 74, 10}); // pixel x owns two distinct slots
        if (image) op(80, {26, 77, 74, 8});            // distinct image coordinate (x,0)
    }
    const uint32_t index = dependent ? 53u : 8u;
    const auto buffer = [&](bool per_invocation = false) {
        if (per_invocation) op(128, {3, 76, 75, index});
        op(65, {18, 54, per_invocation ? 69u : 24u, 8, per_invocation ? 76u : index});
    };
    const auto local = [&] { op(65, {20, 55, 41, index}); };
    const auto fixed_local_output = [&] {
        op(65, {20, 57, 41, 8}); op(61, {4, 56, 57}); op(62, {23, 56});
    };
    const auto scalar_output = [&] { op(61, {4, 56, 42}); op(62, {23, 56}); };
    switch (shape) {
        case Shape::BufferRead:
            buffer(); op(61, {4, 56, 54}); op(62, {23, 56}); break;
        case Shape::ExternalStore:
            buffer(true); op(62, {54, 13}); break;
        case Shape::LocalRead:
            local(); op(61, {4, 56, 55}); op(62, {23, 56}); break;
        case Shape::LocalStore:
            local(); op(62, {55, 13}); fixed_local_output(); break;
        case Shape::CopySourceExternal:
            local(); op(63, {23, 55}); break;
        case Shape::CopyDestinationExternal:
            buffer(true); op(63, {54, 43}); break;
        case Shape::CopySourceLocal:
            local(); op(63, {42, 55}); scalar_output(); break;
        case Shape::CopyDestinationLocal:
            local(); op(63, {55, 43}); fixed_local_output(); break;
        case Shape::GuardedLocalCopy:
        case Shape::EarlyReturn:
            op(247, {62, 0}); op(250, {dependent ? vote : 14u, 61, 62});
            op(248, {61});
            if (shape == Shape::EarlyReturn) op(253, {});
            else { op(63, {42, 43}); op(249, {62}); }
            op(248, {62});
            if (shape == Shape::EarlyReturn) op(62, {23, 13});
            else scalar_output();
            break;
        case Shape::AtomicAddress:
        case Shape::AtomicStoreAddress:
        case Shape::FlagClearParserOnly:
            op(65, {37, 54, 38, 8, index});
            if (shape == Shape::AtomicAddress) op(234, {3, 56, 54, 9, 8, 9});
            else if (shape == Shape::AtomicStoreAddress) op(228, {54, 9, 8, 9});
            else op(319, {54, 9, 8}); // Kernel-only parser control, never a Vulkan module
            break;
        case Shape::ImageIdentity:
        case Shape::ImageSample:
            op(65, {32, 54, 33, shape == Shape::ImageIdentity ? index : 8u});
            op(61, {29, 56, 54});
            if (shape == Shape::ImageSample) op(99, {56, 77, 28, 64, index}); // Sample mask/ID
            else op(99, {56, 77, 28});
            break;
        case Shape::OpaqueCall:
            if (dependent) op(57, {1, 56, 80, vote}); // opaque effect, no return SSA links
            break;
        case Shape::DrefSample:
            op(65, {32, 54, 33, 8}); op(61, {29, 56, 54}); op(61, {79, 87, 82});
            op(86, {80, 88, 56, 87}); op(169, {4, 85, vote, 13, 12});
            op(89, {4, 89, 88, 84, dependent ? 85u : 12u}); op(62, {23, 89});
            break;
        case Shape::CalleeReturn:
            if (dependent) {
                op(57, {2, 56, 80}); op(169, {4, 57, 56, 13, 12}); op(62, {23, 57});
            }
            break;
        case Shape::CopySizeParserOnly:
            // Deliberately unsupported Vulkan environment, testing the parser's defensive size
            // edge only. This is NEVER emitted by the strict corpus or executed on a device.
            op(64, {23, 43, dependent ? 53u : 9u}); break;
    }
    op(253, {}); op(56, {});
    if (shape == Shape::OpaqueCall) {
        op(54, {1, 80, 0, 39}); op(55, {2, 81}); op(248, {82});
        op(247, {85, 0}); op(250, {81, 83, 84});
        op(248, {83}); op(62, {23, 13}); op(249, {85});
        op(248, {84}); op(62, {23, 12}); op(249, {85});
        op(248, {85}); op(253, {}); op(56, {});
    }
    if (shape == Shape::CalleeReturn) {
        op(54, {2, 80, 0, 39}); op(248, {82});
        op(61, {4, 81, 22}); op(186, {2, 83, 81, 12}); op(335, {2, 84, 11, 83});
        op(254, {84}); op(56, {});
    }
    return w;
}

inline std::vector<Fixture> fixtures() {
    const struct { const char* name; Shape shape; } shapes[] = {
        {"buffer_read", Shape::BufferRead}, {"external_store", Shape::ExternalStore},
        {"local_read", Shape::LocalRead}, {"local_store", Shape::LocalStore},
        {"copy_source_external", Shape::CopySourceExternal},
        {"copy_destination_external", Shape::CopyDestinationExternal},
        {"copy_source_local", Shape::CopySourceLocal},
        {"copy_destination_local", Shape::CopyDestinationLocal},
        {"guarded_local_copy", Shape::GuardedLocalCopy}, {"early_return", Shape::EarlyReturn},
        {"atomic_address", Shape::AtomicAddress}, {"image_identity", Shape::ImageIdentity},
        {"atomic_store_address", Shape::AtomicStoreAddress},
        {"image_sample", Shape::ImageSample}, {"opaque_call", Shape::OpaqueCall},
        {"dref_sample", Shape::DrefSample},
        {"callee_return", Shape::CalleeReturn},
        {"parser_copy_size", Shape::CopySizeParserOnly},
        {"parser_flag_clear", Shape::FlagClearParserOnly},
    };
    std::vector<Fixture> out;
    for (const auto& shape : shapes) {
        for (bool dependent : {true, false}) {
            out.push_back({std::string(shape.name) + (dependent ? "_dependent" : "_dead"),
                           module(shape.shape, dependent), !dependent,
                           shape.shape != Shape::CopySizeParserOnly &&
                           shape.shape != Shape::FlagClearParserOnly});
        }
    }
    out.push_back({"uniform_vote_index", module(Shape::BufferRead, true, true), true, true});
    out.push_back({"uniform_vote_call", module(Shape::OpaqueCall, true, true), true, true});
    out.push_back({"image_literal_mask_is_not_id", module(Shape::ImageSample, false, false, true),
                   true, true});
    return out;
}

} // namespace prosper::test::wave_width
