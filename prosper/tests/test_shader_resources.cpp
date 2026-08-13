// test_shader_resources — fixes the resource-binding contract (shader_resources.hpp): format sizing
// and the recompiler/pipeline lookups both halves rely on. Pure (no Vulkan), runs in CI.
#include "../src/gpu/shader_resources.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include "../frontends/shared/live_compute.hpp"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void emit(std::vector<uint32_t>& spv, uint16_t op, std::initializer_list<uint32_t> words) {
    spv.push_back((static_cast<uint32_t>(words.size() + 1) << 16) | op);
    spv.insert(spv.end(), words.begin(), words.end());
}

static void emit_string(std::vector<uint32_t>& spv, uint16_t op, const char* string) {
    std::vector<uint32_t> words;
    uint32_t packed = 0;
    uint32_t byte = 0;
    do {
        const uint8_t value = static_cast<uint8_t>(*string);
        packed |= static_cast<uint32_t>(value) << (byte * 8u);
        if (++byte == 4u) {
            words.push_back(packed);
            packed = 0;
            byte = 0;
        }
        if (!value) break;
        ++string;
    } while (true);
    if (byte) words.push_back(packed);
    spv.push_back((static_cast<uint32_t>(words.size() + 1u) << 16u) | op);
    spv.insert(spv.end(), words.begin(), words.end());
}

static std::vector<uint32_t> tail_marker_test_spirv(bool writable = false,
                                                    bool dynamic = false) {
    std::vector<uint32_t> s = {0x07230203u, 0x00010000u, 0, 32, 0};
    emit(s, 15, {5, 20, 0x6e69616d, 0});                    // OpEntryPoint Compute %20 "main"
    emit_string(s, 330, "Prosper.StorageBufferZeroPad=0,9,2,4,u16");
    emit(s, 21, {1, 32, 0});                               // %1 = u32
    emit(s, 29, {2, 1});                                   // %2 = runtime array u32
    emit(s, 30, {3, 2});                                   // %3 = struct {%2}
    emit(s, 32, {4, 12, 3});                               // %4 = StorageBuffer pointer to %3
    emit(s, 32, {5, 12, 1});                               // %5 = StorageBuffer pointer to u32
    emit(s, 43, {1, 6, 0});                                // %6 = 0
    emit(s, 59, {4, 8, 12});                               // %8 = buffer
    emit(s, 71, {2, 6, 4});                                // ArrayStride 4
    emit(s, 72, {3, 0, 35, 0});                            // member 0 Offset 0
    emit(s, 71, {8, 34, 0}); emit(s, 71, {8, 33, 9});     // set 0 binding 9
    emit(s, 65, {5, 9, 8, 6, dynamic ? 20u : 6u});        // %9 = &buffer[0][index]
    if (writable) emit(s, 62, {9, 6});                     // store %9
    else emit(s, 61, {1, 10, 9});                         // %10 = load %9
    return s;
}

static std::vector<uint32_t> descriptor_test_spirv() {
    // Minimal reflection fixture. Binding 9 is a used storage buffer whose u32 element 4 is loaded,
    // proving a 20-byte minimum. Binding 10 has a dead access chain but no load/store and must be ignored.
    std::vector<uint32_t> s = {0x07230203u, 0x00010000u, 0, 32, 0};
    emit(s, 15, {0, 20, 0x6e69616d, 0});                    // OpEntryPoint Vertex %20 "main"
    emit(s, 21, {1, 32, 0});                               // %1 = u32
    emit(s, 29, {2, 1});                                   // %2 = runtime array u32
    emit(s, 30, {3, 2});                                   // %3 = struct {%2}
    emit(s, 32, {4, 12, 3});                               // %4 = StorageBuffer pointer to %3
    emit(s, 32, {5, 12, 1});                               // %5 = StorageBuffer pointer to u32
    emit(s, 43, {1, 6, 0});                                // %6 = 0
    emit(s, 43, {1, 7, 4});                                // %7 = 4
    emit(s, 59, {4, 8, 12});                               // %8 = used buffer
    emit(s, 59, {4, 11, 12});                              // %11 = inactive declaration
    emit(s, 71, {2, 6, 4});                                // ArrayStride 4
    emit(s, 72, {3, 0, 35, 0});                            // member 0 Offset 0
    emit(s, 71, {8, 34, 0}); emit(s, 71, {8, 33, 9});     // set 0 binding 9
    emit(s, 71, {11, 34, 0}); emit(s, 71, {11, 33, 10});  // inactive set 0 binding 10
    emit(s, 65, {5, 9, 8, 6, 7});                         // %9 = &buffer[0][4]
    emit(s, 61, {1, 10, 9});                               // %10 = load %9
    emit(s, 65, {5, 12, 11, 6, 7});                        // %12 = dead &inactive[0][4]
    return s;
}

// An ARRAY of eight storage-buffer descriptors at one binding (#2412 stage 4). Same shape as
// descriptor_test_spirv, with the variable's pointee wrapped in `OpTypeArray %Block %8` so the binding
// declares eight descriptors rather than one, and the access chain carrying the extra leading index that
// selects among them.
static std::vector<uint32_t> descriptor_array_test_spirv() {
    std::vector<uint32_t> s = {0x07230203u, 0x00010000u, 0, 32, 0};
    emit(s, 15, {0, 20, 0x6e69616d, 0});                    // OpEntryPoint Vertex %20 "main"
    emit(s, 21, {1, 32, 0});                                // %1 = u32
    emit(s, 29, {2, 1});                                    // %2 = runtime array u32 (the block's data)
    emit(s, 30, {3, 2});                                    // %3 = struct {%2}  -- the Block
    emit(s, 43, {1, 14, 8});                                // %14 = 8
    emit(s, 28, {15, 3, 14});                               // %15 = array of 8 x %3  <- descriptor array
    emit(s, 32, {4, 12, 15});                               // %4 = StorageBuffer pointer to the ARRAY
    emit(s, 32, {5, 12, 1});                                // %5 = StorageBuffer pointer to u32
    emit(s, 43, {1, 6, 0});                                 // %6 = 0
    emit(s, 43, {1, 7, 4});                                 // %7 = 4
    emit(s, 59, {4, 8, 12});                                // %8 = the array-of-descriptors variable
    emit(s, 71, {2, 6, 4});                                 // ArrayStride 4 on the block's data
    emit(s, 72, {3, 0, 35, 0});                             // member 0 Offset 0
    emit(s, 71, {8, 34, 0}); emit(s, 71, {8, 33, 9});       // set 0 binding 9
    emit(s, 65, {5, 9, 8, 6, 6, 7});                        // %9 = &buffer[0][0][4] -- extra leading index
    emit(s, 61, {1, 10, 9});                                // %10 = load %9
    return s;
}

// The same eight-descriptor array with its length as an `OpSpecConstant` instead of `OpConstant`, so the
// length id resolves to nothing in this pass and the arity is genuinely UNREADABLE (#2412). Valid SPIR-V;
// `OpSpecConstant` (50) is not among the opcodes reflection decodes into `constants`.
static std::vector<uint32_t> descriptor_array_specconst_test_spirv() {
    std::vector<uint32_t> s = {0x07230203u, 0x00010000u, 0, 32, 0};
    emit(s, 15, {0, 20, 0x6e69616d, 0});                    // OpEntryPoint Vertex %20 "main"
    emit(s, 21, {1, 32, 0});                                // %1 = u32
    emit(s, 29, {2, 1});                                    // %2 = runtime array u32 (the block's data)
    emit(s, 30, {3, 2});                                    // %3 = struct {%2}  -- the Block
    emit(s, 50, {1, 14, 8});                                // %14 = OpSpecConstant 8  <- NOT resolved
    emit(s, 28, {15, 3, 14});                               // %15 = array of %14 x %3
    emit(s, 32, {4, 12, 15});                               // %4 = StorageBuffer pointer to the ARRAY
    emit(s, 32, {5, 12, 1});                                // %5 = StorageBuffer pointer to u32
    emit(s, 43, {1, 6, 0});                                 // %6 = 0
    emit(s, 43, {1, 7, 4});                                 // %7 = 4
    emit(s, 59, {4, 8, 12});                                // %8 = the array-of-descriptors variable
    emit(s, 71, {2, 6, 4});                                 // ArrayStride 4 on the block's data
    emit(s, 72, {3, 0, 35, 0});                             // member 0 Offset 0
    emit(s, 71, {8, 34, 0}); emit(s, 71, {8, 33, 9});       // set 0 binding 9
    emit(s, 65, {5, 9, 8, 6, 6, 7});                        // %9 = &buffer[0][0][4]
    emit(s, 61, {1, 10, 9});                                // %10 = load %9
    return s;
}

static std::vector<uint32_t> atomic_descriptor_test_spirv() {
    std::vector<uint32_t> s = descriptor_test_spirv();
    // %12 points at binding 10. AtomicAnd is result-producing, so its pointer is operand 2 after
    // result type/result id. The descriptor is write-only and must still be reflected for binding.
    emit(s, 240, {1, 13, 12, 6, 6, 6});                    // %13 = atomicAnd %12
    return s;
}

static std::vector<uint32_t> image_test_spirv(bool sampled_float = true) {
    std::vector<uint32_t> s = {0x07230203u, 0x00010000u, 0, 16, 0};
    emit(s, 15, {4, 10, 0x6e69616d, 0});                    // OpEntryPoint Fragment %10 "main"
    if (sampled_float) emit(s, 22, {1, 32});                // %1 = f32
    else emit(s, 21, {1, 32, 0});                           // %1 = u32
    emit(s, 25, {2, 1, 1, 0, 0, 0, 1, 0});                // %2 = sampled 2D image
    emit(s, 27, {3, 2});                                   // %3 = sampled-image %2
    emit(s, 32, {4, 0, 3});                                // %4 = UniformConstant pointer
    emit(s, 59, {4, 5, 0});                                // %5 = image variable
    emit(s, 71, {5, 34, 1}); emit(s, 71, {5, 33, 4});      // set 1 binding 4
    emit(s, 61, {3, 6, 5});                                // %6 = load %5
    emit(s, 87, {1, 7, 6, 8});                             // %7 = OpImageSampleImplicitLod %6
    return s;
}

static std::vector<uint32_t> image_fetch_test_spirv(bool sampled_float = true) {
    std::vector<uint32_t> s = image_test_spirv(sampled_float);
    // Replace the terminal normalized sample with OpImageFetch. Reflection only needs the image
    // object's provenance and opcode class; the fixture is not submitted to a SPIR-V implementation.
    s[s.size() - 5] = (5u << 16) | 95u;
    return s;
}

static std::vector<uint32_t> image_sample_fetch_test_spirv() {
    std::vector<uint32_t> s = image_test_spirv();
    emit(s, 95, {1, 9, 6, 8});                             // %9 = OpImageFetch %6
    return s;
}

static std::vector<uint32_t> image_sample_query_test_spirv() {
    std::vector<uint32_t> s = image_test_spirv();
    emit(s, 100, {2, 9, 6});                               // %9 = OpImage %6
    emit(s, 106, {1, 10, 9});                              // %10 = OpImageQueryLevels %9
    return s;
}

static std::vector<uint32_t> image_query_test_spirv() {
    std::vector<uint32_t> s = {0x07230203u, 0x00010000u, 0, 16, 0};
    emit(s, 15, {5, 12, 0x6e69616d, 0});                    // OpEntryPoint Compute %12 "main"
    emit(s, 21, {1, 32, 0});                               // %1 = u32
    emit(s, 25, {2, 1, 1, 0, 0, 0, 1, 0});                // %2 = sampled 2D image
    emit(s, 27, {3, 2});                                   // %3 = sampled-image %2
    emit(s, 32, {4, 0, 3});                                // %4 = UniformConstant pointer
    emit(s, 59, {4, 5, 0});                                // %5 = image variable
    emit(s, 71, {5, 34, 0}); emit(s, 71, {5, 33, 4});      // set 0 binding 4
    emit(s, 61, {3, 6, 5});                                // %6 = load sampled image
    emit(s, 100, {2, 7, 6});                               // %7 = OpImage %6
    emit(s, 106, {1, 8, 7});                               // %8 = OpImageQueryLevels %7
    emit(s, 103, {1, 9, 7, 10});                           // %9 = OpImageQuerySizeLod %7 %10
    return s;
}

static std::vector<uint32_t> storage_image_access_test_spirv(
    SpirvImageNumericClass numeric_class = SpirvImageNumericClass::Uint) {
    // Two storage-image descriptors: binding 5 is read, binding 6 is independently write-only.
    // This is the shape that permits the compute backend to skip seeding binding 6 even though the
    // shader has an OpImageRead for binding 5.
    std::vector<uint32_t> s = {0x07230203u, 0x00010000u, 0, 24, 0};
    emit(s, 15, {5, 20, 0x6e69616d, 0});                    // OpEntryPoint Compute %20 "main"
    if (numeric_class == SpirvImageNumericClass::Float)
        emit(s, 22, {1, 32});                              // %1 = f32
    else
        emit(s, 21, {1, 32,
                     numeric_class == SpirvImageNumericClass::Sint ? 1u : 0u}); // %1 = i32/u32
    emit(s, 25, {2, 1, 1, 0, 0, 0, 2, 0});                // %2 = storage 2D image
    emit(s, 32, {3, 0, 2});                                // %3 = UniformConstant pointer
    emit(s, 59, {3, 4, 0}); emit(s, 59, {3, 5, 0});        // %4 source, %5 destination
    emit(s, 71, {4, 34, 0}); emit(s, 71, {4, 33, 5});      // set 0 binding 5
    emit(s, 71, {5, 34, 0}); emit(s, 71, {5, 33, 6});      // set 0 binding 6
    emit(s, 61, {2, 6, 4}); emit(s, 61, {2, 7, 5});        // load image objects
    emit(s, 98, {8, 9, 6, 10});                            // %9 = OpImageRead %6
    emit(s, 99, {7, 10, 9});                               // OpImageWrite %7
    return s;
}

static std::vector<uint32_t> storage_image_atomic_test_spirv() {
    // R32_UINT storage image -> OpImageTexelPointer -> OpAtomicExchange. The image variable is an
    // operand of OpImageTexelPointer directly, so reflection must carry that provenance through the
    // returned Image-storage pointer to the atomic read/write.
    std::vector<uint32_t> s = {0x07230203u, 0x00010000u, 0, 24, 0};
    emit(s, 15, {4, 20, 0x6e69616d, 0});                    // OpEntryPoint Fragment %20 "main"
    emit(s, 21, {1, 32, 0});                               // %1 = u32
    emit(s, 23, {2, 1, 2});                                // %2 = uvec2
    emit(s, 25, {3, 1, 1, 0, 0, 0, 2, 33});               // %3 = storage 2D R32ui image
    emit(s, 32, {4, 0, 3});                                // %4 = UniformConstant pointer to image
    emit(s, 32, {5, 11, 1});                               // %5 = Image pointer to u32
    emit(s, 59, {4, 6, 0});                                // %6 = image variable
    emit(s, 71, {6, 34, 1}); emit(s, 71, {6, 33, 7});      // set 1 binding 7
    emit(s, 43, {1, 7, 0});                                // %7 = 0 (sample/value)
    emit(s, 60, {5, 8, 6, 9, 7});                          // %8 = OpImageTexelPointer %6
    emit(s, 229, {1, 10, 8, 7, 7, 7});                     // %10 = atomicExchange %8
    return s;
}

static bool has_issue(const DescriptorValidationReport& r, DescriptorIssueCode code) {
    for (const auto& issue : r.issues) if (issue.code == code) return true;
    return false;
}

static void set_descriptor_mode(const char* value) {
#ifdef _WIN32
    _putenv_s("PROSPER_DESCRIPTOR_VALIDATE", value);
#else
    setenv("PROSPER_DESCRIPTOR_VALIDATE", value, 1);
#endif
}

int main() {
    printf("== test_shader_resources ==\n");

    CHECK(data_format_bytes(DataFormat::Float32) == 4 && data_format_bytes(DataFormat::Uint32) == 4,
          "32-bit formats are 4 bytes");
    CHECK(data_format_bytes(DataFormat::Float16) == 2 && data_format_bytes(DataFormat::Unorm16) == 2,
          "16-bit formats are 2 bytes");
    CHECK(data_format_bytes(DataFormat::Unorm8) == 1 && data_format_bytes(DataFormat::Sint8) == 1,
          "8-bit formats are 1 byte");
    CHECK(data_format_bytes(DataFormat::Unknown) == 0, "Unknown format is 0 bytes");

    SpirvDescriptorBinding tail_descriptor;
    tail_descriptor.kind = SpirvDescriptorKind::StorageBuffer;
    tail_descriptor.required_bytes = 4;
    tail_descriptor.readable = true;
    tail_descriptor.zero_pad_logical_bytes = 2;
    tail_descriptor.zero_pad_binding_bytes = 4;
    tail_descriptor.zero_pad_semantic = StorageBufferTailSemantic::Uint16;
    ShaderResource tail_resource;
    tail_resource.cls = ResourceClass::ConstantBuffer;
    tail_resource.format = DataFormat::Uint16;
    tail_resource.num_components = 1;
    tail_resource.size = 2;
    tail_resource.stride = 2;
    const StorageBufferMaterializationPlan tail_plan =
        plan_storage_buffer_materialization(tail_descriptor, tail_resource);
    const uint8_t tail_source[2] = {0x34, 0x12};
    uint8_t tail_destination[4] = {0xa5, 0xa5, 0xa5, 0xa5};
    CHECK(tail_plan.valid && tail_plan.zero_padded_tail &&
          tail_plan.logical_bytes == 2 && tail_plan.binding_bytes == 4 &&
          materialize_storage_buffer_bytes(
              tail_plan, tail_source, sizeof(tail_source),
              tail_destination, sizeof(tail_destination)) &&
          tail_destination[0] == 0x34 && tail_destination[1] == 0x12 &&
          tail_destination[2] == 0 && tail_destination[3] == 0,
          "one-record Uint16 materialization copies two bytes and deterministically clears the tail");
    ShaderResource ordinary_four_byte_resource = tail_resource;
    ordinary_four_byte_resource.format = DataFormat::Uint32;
    ordinary_four_byte_resource.size = 4;
    ordinary_four_byte_resource.stride = 4;
    SpirvDescriptorBinding ordinary_descriptor;
    ordinary_descriptor.kind = SpirvDescriptorKind::StorageBuffer;
    ordinary_descriptor.required_bytes = 4;
    ordinary_descriptor.readable = true;
    const StorageBufferMaterializationPlan ordinary_four_byte_plan =
        plan_storage_buffer_materialization(
            ordinary_descriptor, ordinary_four_byte_resource);
    auto float_tail_descriptor = tail_descriptor;
    float_tail_descriptor.zero_pad_semantic = StorageBufferTailSemantic::Float16;
    ShaderResource float_tail_resource = tail_resource;
    float_tail_resource.format = DataFormat::Float16;
    const StorageBufferMaterializationPlan float_tail_plan =
        plan_storage_buffer_materialization(float_tail_descriptor, float_tail_resource);
    const auto tail_cache_discriminator =
        prosper::frontend::compute_buffer_materialization_discriminator(tail_plan);
    const auto ordinary_cache_discriminator =
        prosper::frontend::compute_buffer_materialization_discriminator(
            ordinary_four_byte_plan);
    const auto float_tail_cache_discriminator =
        prosper::frontend::compute_buffer_materialization_discriminator(float_tail_plan);
    CHECK(ordinary_four_byte_plan.valid && float_tail_plan.valid &&
          !(tail_cache_discriminator == ordinary_cache_discriminator) &&
          !(tail_cache_discriminator == float_tail_cache_discriminator),
          "persistent compute cache partitions logical2/u16/f16 from ordinary bound4 sources");
    auto writable_tail_descriptor = tail_descriptor;
    writable_tail_descriptor.writable = true;
    CHECK(!plan_storage_buffer_materialization(
               writable_tail_descriptor, tail_resource).valid,
          "zero-padded tail contract rejects writable storage buffers");
    SpirvDescriptorBinding partial_tail_descriptor;
    partial_tail_descriptor.zero_pad_semantic = StorageBufferTailSemantic::Uint16;
    CHECK(!plan_storage_buffer_materialization(
               partial_tail_descriptor, tail_resource).valid,
          "semantic-only partial zero-pad metadata fails closed");

    uint8_t marker_host[2] = {0x34, 0x12};
    tail_resource.binding = 9;
    tail_resource.host_data = marker_host;
    tail_resource.host_data_size = sizeof(marker_host);
    ShaderResourceTable tail_runtime;
    tail_runtime.resources.push_back(tail_resource);
    const DescriptorValidationReport valid_tail_marker =
        validate_spirv_descriptor_interface(
            tail_marker_test_spirv(), &tail_runtime, 0, SpirvShaderStage::Compute, false);
    const SpirvDescriptorBinding* valid_tail_binding =
        find_spirv_descriptor_binding(valid_tail_marker, 0, 9);
    CHECK(valid_tail_marker.ok() && valid_tail_binding &&
          valid_tail_binding->zero_pad_logical_bytes == 2 &&
          valid_tail_binding->zero_pad_binding_bytes == 4,
          "strict zero-pad marker reflects one exact read-only constant-index SSBO contract");
    ShaderResourceTable mismatched_tail_runtime = tail_runtime;
    mismatched_tail_runtime.resources[0].format = DataFormat::Float16;
    const DescriptorValidationReport mismatched_tail_marker =
        validate_spirv_descriptor_interface(
            tail_marker_test_spirv(), &mismatched_tail_runtime,
            0, SpirvShaderStage::Compute, false);
    CHECK(!mismatched_tail_marker.ok() &&
          has_issue(mismatched_tail_marker, DescriptorIssueCode::InvalidBufferMetadata),
          "u16 zero-pad marker rejects a runtime Float16 V# instead of inferring semantics");

    auto malformed_tail_marker = tail_marker_test_spirv();
    emit_string(malformed_tail_marker, 330,
                "Prosper.StorageBufferZeroPad=0,9,two,4,u16");
    auto duplicate_tail_marker = tail_marker_test_spirv();
    emit_string(duplicate_tail_marker, 330,
                "Prosper.StorageBufferZeroPad=0,9,2,4,u16");
    auto wrong_binding_tail_marker = tail_marker_test_spirv();
    emit_string(wrong_binding_tail_marker, 330,
                "Prosper.StorageBufferZeroPad=0,10,2,4,u16");
    auto non_ssbo_tail_marker = image_test_spirv();
    emit_string(non_ssbo_tail_marker, 330,
                "Prosper.StorageBufferZeroPad=1,4,2,4,u16");
    auto unknown_semantic_tail_marker = tail_marker_test_spirv();
    emit_string(unknown_semantic_tail_marker, 330,
                "Prosper.StorageBufferZeroPad=0,9,2,4,unknown");
    const auto strict_marker_rejects = [&](const std::vector<uint32_t>& spirv) {
        const DescriptorValidationReport report = validate_spirv_descriptor_interface(
            spirv, &tail_runtime, 0, SpirvShaderStage::Compute, false);
        return !report.ok() && has_issue(report, DescriptorIssueCode::MalformedSpirv);
    };
    CHECK(strict_marker_rejects(malformed_tail_marker),
          "malformed zero-pad marker rejects as malformed SPIR-V");
    CHECK(strict_marker_rejects(unknown_semantic_tail_marker),
          "unknown zero-pad semantic token rejects as malformed SPIR-V");
    CHECK(strict_marker_rejects(duplicate_tail_marker),
          "duplicate zero-pad marker for one binding rejects as malformed SPIR-V");
    CHECK(strict_marker_rejects(wrong_binding_tail_marker),
          "unconsumed zero-pad marker for the wrong binding rejects as malformed SPIR-V");
    CHECK(strict_marker_rejects(tail_marker_test_spirv(true, false)),
          "zero-pad marker inconsistent with a writable SSBO rejects as malformed SPIR-V");
    CHECK(strict_marker_rejects(tail_marker_test_spirv(false, true)),
          "zero-pad marker inconsistent with a dynamic SSBO access rejects as malformed SPIR-V");
    CHECK(strict_marker_rejects(non_ssbo_tail_marker),
          "zero-pad marker inconsistent with a non-SSBO descriptor rejects as malformed SPIR-V");

    CHECK(float_to_half(0.0f) == 0x0000u && float_to_half(-0.0f) == 0x8000u &&
          float_to_half(1.0f) == 0x3c00u && float_to_half(-2.0f) == 0xc000u &&
          float_to_half(65504.0f) == 0x7bffu,
          "float32 -> float16 conversion preserves zero/sign and exact finite values");
    CHECK(float_to_half(half_to_float(0x0001u)) == 0x0001u &&
          float_to_half(half_to_float(0x3555u)) == 0x3555u &&
          float_to_half(half_to_float(0x7bffu)) == 0x7bffu,
          "float16 -> float32 -> float16 round-trip is bit exact for finite values");

    CHECK(unorm16_to_unorm8(0x0000u) == 0u && unorm16_to_unorm8(0x00ffu) == 1u &&
          unorm16_to_unorm8(0x8000u) == 128u && unorm16_to_unorm8(0xffffu) == 255u,
          "UNORM16 -> UNORM8 uses the complete little-endian value and preserves endpoints");
    uint32_t unorm16_reversals = 0;
    uint8_t previous_unorm8 = unorm16_to_unorm8(0);
    for (uint32_t value = 1; value <= 0xffffu; ++value) {
        const uint8_t current = unorm16_to_unorm8(static_cast<uint16_t>(value));
        if (current < previous_unorm8) ++unorm16_reversals;
        previous_unorm8 = current;
    }
    CHECK(unorm16_reversals == 0,
          "UNORM16 -> UNORM8 remains monotonic instead of wrapping at byte boundaries");

    // A table as the front-half would build it: a float32×4 constant buffer (descriptor at SRT 0x20)
    // and a unorm8×4 vertex buffer (descriptor at SRT 0x40).
    ShaderResourceTable t;
    t.resources.push_back({ResourceClass::ConstantBuffer, DataFormat::Float32, 4, /*binding*/2,
                           /*gpu_addr*/0xC0000000ull, /*size*/256, /*stride*/0, /*srt_offset*/0x20});
    t.resources.push_back({ResourceClass::VertexBuffer, DataFormat::Unorm8, 4, /*binding*/3,
                           /*gpu_addr*/0xD0000000ull, /*size*/4096, /*stride*/4, /*srt_offset*/0x40});

    // Recompiler's provenance lookup: descriptor at SRT 0x40 -> the vertex buffer at binding 3.
    const ShaderResource* v = t.by_srt_offset(0x40);
    CHECK(v && v->cls == ResourceClass::VertexBuffer && v->format == DataFormat::Unorm8 &&
          v->binding == 3 && v->stride == 4, "by_srt_offset resolves the descriptor to its resource");
    const ShaderResource* c = t.by_srt_offset(0x20);
    CHECK(c && c->cls == ResourceClass::ConstantBuffer && c->binding == 2 && c->gpu_addr == 0xC0000000ull,
          "by_srt_offset resolves the constant buffer");
    CHECK(t.by_srt_offset(0x99) == nullptr && t.by_srt_offset(0xFFFFFFFFu) == nullptr,
          "unknown / not-descriptor-keyed offset resolves to null");

    // Pipeline's lookup: bind by binding number.
    const ShaderResource* b3 = t.by_binding(3);
    CHECK(b3 && b3->gpu_addr == 0xD0000000ull && b3->size == 4096,
          "by_binding gives the pipeline the bytes to bind");

    // DIRECT provenance: a vertex-buffer V# placed straight in user-data SGPRs (s[8:11]) — keyed by
    // sgpr_base, not srt_offset (that's how vertex descriptors reach the shader; no in-shader s_load).
    ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32;
    vb.num_components = 3; vb.binding = 4; vb.stride = 12; vb.sgpr_base = 8;   // srt_offset stays 0xFFFFFFFF
    t.resources.push_back(vb);
    const ShaderResource* v2 = t.by_sgpr_base(8);
    CHECK(v2 && v2->cls == ResourceClass::VertexBuffer && v2->binding == 4 && v2->num_components == 3,
          "by_sgpr_base resolves a direct (user-data) vertex descriptor");
    CHECK(t.by_sgpr_base(0x99) == nullptr && t.by_sgpr_base(0xFFFFFFFFu) == nullptr,
          "unknown / not-SGPR-keyed resolves to null");

    // half_to_float (#290 fp16 texture upload): exact IEEE binary16 decode incl. subnormals/inf/NaN.
    CHECK(half_to_float(0x3C00) == 1.0f && half_to_float(0xBC00) == -1.0f, "half 0x3C00/0xBC00 = +/-1.0");
    CHECK(half_to_float(0x3800) == 0.5f, "half 0x3800 = 0.5 (mid-gray magnitude, not saturated)");
    CHECK(half_to_float(0x0000) == 0.0f && half_to_float(0x8000) == 0.0f, "half +/-0 = 0.0");
    CHECK(half_to_float(0x4248) == 3.140625f, "half 0x4248 = 3.140625 (pi to half precision)");
    CHECK(half_to_float(0x7BFF) == 65504.0f, "half 0x7BFF = 65504 (max finite)");
    CHECK(half_to_float(0x0001) == 5.9604644775390625e-8f, "half 0x0001 = smallest subnormal");
    CHECK(half_to_float(0x03FF) == 6.0975551605224609375e-5f, "half 0x03FF = largest subnormal");
    { float inf = half_to_float(0x7C00); CHECK(inf > 3.4e38f && inf == inf * 2, "half 0x7C00 = +inf"); }
    { float nan = half_to_float(0x7E01); CHECK(nan != nan, "half 0x7E01 = NaN"); }

    // f11/f10 unsigned small floats (#294, packed R11G11B10F scene color): binary16 exponent
    // (bias 15), shortened mantissa (6/5 bits), no sign.
    CHECK(f11_to_float(0x3C0) == 1.0f && f10_to_float(0x1E0) == 1.0f, "f11 0x3C0 / f10 0x1E0 = 1.0");
    CHECK(f11_to_float(0x380) == 0.5f && f10_to_float(0x1C0) == 0.5f, "f11 0x380 / f10 0x1C0 = 0.5");
    CHECK(f11_to_float(0) == 0.0f && f10_to_float(0) == 0.0f, "f11/f10 0 = 0.0");
    CHECK(f11_to_float(0x3E0) == 1.5f, "f11 0x3E0 = 1.5 (mantissa MSB)");
    CHECK(f10_to_float(0x1F0) == 1.5f, "f10 0x1F0 = 1.5 (mantissa MSB)");
    CHECK(f11_to_float(0x001) == half_to_float(0x0010), "f11 smallest subnormal == half m<<4 subnormal");
    CHECK(f10_to_float(0x001) == half_to_float(0x0020), "f10 smallest subnormal == half m<<5 subnormal");
    CHECK(f11_to_float(0x7BF) == 65024.0f, "f11 0x7BF = max finite (65024)");
    { float inf = f11_to_float(0x7C0); CHECK(inf > 3.4e38f && inf == inf * 2, "f11 0x7C0 = +inf"); }
    { float nan = f10_to_float(0x3E1); CHECK(nan != nan, "f10 0x3E1 = NaN"); }
    CHECK(float_to_f11(1.0f) == 0x3C0 && float_to_f10(1.0f) == 0x1E0,
          "1.0 packs to the exact f11/f10 exponent fields");
    CHECK(float_to_f11(-1.0f) == 0 && float_to_f10(-0.5f) == 0,
          "negative unsigned-small-float values clamp to zero");
    CHECK(float_to_f11(1.0078125f) == 0x3C0 && float_to_f11(1.0234375f) == 0x3C2,
          "f11 halfway cases round to an even mantissa");
    CHECK(float_to_f11(65024.0f) == 0x7BF && float_to_f10(64512.0f) == 0x3DF,
          "maximum finite f11/f10 values pack without overflowing");
    uint32_t bad_small_float_roundtrip = 0;
    for (uint16_t v = 0; v <= 0x7BF; ++v)
        if (float_to_f11(f11_to_float(v)) != v) ++bad_small_float_roundtrip;
    for (uint16_t v = 0; v <= 0x3DF; ++v)
        if (float_to_f10(f10_to_float(v)) != v) ++bad_small_float_roundtrip;
    CHECK(bad_small_float_roundtrip == 0,
          "every finite f11/f10 code survives unpack then repack exactly");
    CHECK(data_format_bytes(DataFormat::Float10_11_11) == 0,
          "Float10_11_11 is packed: per-component bytes = 0 (texel size lives in bytes_per_block)");
    CHECK(data_format_bytes(DataFormat::Unorm2_10_10_10) == 0,
          "Unorm2_10_10_10 is packed: per-component bytes = 0");
    CHECK(data_format_bytes(DataFormat::Snorm2_10_10_10) == 0 &&
          data_format_bytes(DataFormat::Uint2_10_10_10) == 0 &&
          data_format_bytes(DataFormat::Sint2_10_10_10) == 0,
          "packed 2_10_10_10 vertex variants have no uniform per-component byte size");

    CHECK(native_float_storage_image_supported(DataFormat::Unorm8, 1, false, true) &&
              native_float_storage_image_supported(DataFormat::Unorm8, 2, false, true) &&
              native_float_storage_image_supported(
                  DataFormat::Float10_11_11, 3, false, true),
          "native typed storage accepts supported R8, RG8, and packed R11G11B10 formats");
    CHECK(!native_float_storage_image_supported(DataFormat::Unorm8, 1, false, false) &&
              !native_float_storage_image_supported(DataFormat::Unorm8, 2, false, false) &&
              !native_float_storage_image_supported(
                  DataFormat::Float10_11_11, 3, false, false),
          "missing Vulkan storage-image support forces optional typed formats to the raw fallback");
    CHECK(!native_float_storage_image_supported(DataFormat::Unorm8, 4, true, true) &&
              !native_float_storage_image_supported(DataFormat::Float16, 3, false, true),
          "device support cannot override semantic native-storage exclusions");
    const uint32_t r8_storage =
        native_storage_format_support_bit(DataFormat::Unorm8, 1);
    const uint32_t rg8_storage =
        native_storage_format_support_bit(DataFormat::Unorm8, 2);
    const uint32_t packed_storage =
        native_storage_format_support_bit(DataFormat::Float10_11_11, 3);
    const uint32_t fp16_3d_storage =
        native_storage_3d_format_support_bit(DataFormat::Float16, 4);
    CHECK(r8_storage && rg8_storage && packed_storage && fp16_3d_storage &&
              r8_storage != rg8_storage &&
              rg8_storage != packed_storage &&
              !(fp16_3d_storage & ((1u << 10) - 1u)) &&
              !(fp16_3d_storage & ~kNativeStorageFormatSupportMask) &&
              native_storage_format_support_bit(DataFormat::Float16, 3) == 0,
          "native storage capability bits distinguish exact typed VkFormat and dimension candidates");

    uint8_t rgba10[4] = {};
    unorm2_10_10_10_to_rgba8(0xFFFFFFFFu, rgba10);
    CHECK(rgba10[0] == 255 && rgba10[1] == 255 && rgba10[2] == 255 && rgba10[3] == 255,
          "packed R10G10B10A2 all-ones maps to opaque white");
    unorm2_10_10_10_to_rgba8((512u << 0) | (256u << 10) | (1u << 20) | (2u << 30), rgba10);
    CHECK(rgba10[0] == 128 && rgba10[1] == 64 && rgba10[2] == 0 && rgba10[3] == 170,
          "packed R10G10B10A2 fields normalize and round independently");

    const std::vector<uint32_t> spv = descriptor_test_spirv();
    ShaderResource good{}; good.cls = ResourceClass::VertexBuffer; good.binding = 9;
    good.size = 20; good.gpu_addr = 0x12340000; good.stride = 4;
    ShaderResourceTable valid; valid.resources.push_back(good);
    DescriptorValidationReport vr = validate_spirv_descriptor_interface(
        spv, &valid, 0, SpirvShaderStage::Vertex);
    CHECK(vr.ok() && vr.descriptors.size() == 1,
          "reflection validates one statically-used descriptor and ignores inactive declarations");
    CHECK(vr.descriptors.size() == 1 && vr.descriptors[0].binding == 9 &&
          vr.descriptors[0].kind == SpirvDescriptorKind::StorageBuffer &&
          vr.descriptors[0].required_bytes == 20 && !vr.descriptors[0].dynamic_access &&
          !vr.descriptors[0].writable,
          "constant access chain reflects storage-buffer binding 9 with a 20-byte minimum");

    ShaderResource atomic = good; atomic.binding = 10;
    ShaderResourceTable atomic_table; atomic_table.resources = {good, atomic};
    const DescriptorValidationReport atomic_report = validate_spirv_descriptor_interface(
        atomic_descriptor_test_spirv(), &atomic_table, 0, SpirvShaderStage::Vertex);
    CHECK(atomic_report.ok() && atomic_report.descriptors.size() == 2 &&
              atomic_report.descriptors[1].binding == 10 &&
              atomic_report.descriptors[1].writable,
          "write-only atomic access reflects its writable storage-buffer descriptor");
    ShaderResource graphics_atomic_image = atomic;
    graphics_atomic_image.cls = ResourceClass::StorageImage;
    graphics_atomic_image.format = DataFormat::Uint32;
    graphics_atomic_image.num_components = 1;
    graphics_atomic_image.img_dim = 1;
    graphics_atomic_image.width = graphics_atomic_image.height = 1;
    graphics_atomic_image.depth = 1;
    ShaderResourceTable graphics_atomic_mismatch;
    graphics_atomic_mismatch.resources = {good, graphics_atomic_image};
    const DescriptorValidationReport graphics_atomic_mismatch_report =
        validate_spirv_descriptor_interface(
            atomic_descriptor_test_spirv(), &graphics_atomic_mismatch, 0,
            SpirvShaderStage::Vertex);
    CHECK(!graphics_atomic_mismatch_report.ok() &&
              has_issue(graphics_atomic_mismatch_report, DescriptorIssueCode::WrongType),
          "graphics atomic buffers cannot use the compute-only storage-image exception");

    ShaderResourceTable missing;
    auto mr = validate_spirv_descriptor_interface(spv, &missing, 0, SpirvShaderStage::Vertex);
    CHECK(!mr.ok() && has_issue(mr, DescriptorIssueCode::MissingBinding),
          "missing vertex-color-style binding fails before backend submission");

    ShaderResource wrong = good; wrong.cls = ResourceClass::Texture; wrong.width = wrong.height = 1;
    ShaderResourceTable wrong_table; wrong_table.resources.push_back(wrong);
    auto wr = validate_spirv_descriptor_interface(spv, &wrong_table, 0, SpirvShaderStage::Vertex);
    CHECK(!wr.ok() && has_issue(wr, DescriptorIssueCode::WrongType),
          "combined image at a storage-buffer binding is rejected as wrong type");

    ShaderResourceTable duplicate; duplicate.resources = {good, good};
    auto dr = validate_spirv_descriptor_interface(spv, &duplicate, 0, SpirvShaderStage::Vertex);
    CHECK(!dr.ok() && has_issue(dr, DescriptorIssueCode::DuplicateBinding),
          "duplicate runtime binding is rejected as ambiguous");

    // Descriptor-array arity (#2412). A resource that declares a table of descriptors selected by a
    // runtime index, bound against SPIR-V that reflects ONE descriptor at that binding, must be
    // rejected. Before this code existed the pair validated SILENTLY: the array count was invisible to
    // validation, so a half-finished lift -- a count carried but no reflection or emission behind it --
    // would have bound as though it were an ordinary single descriptor and produced wrong pixels rather
    // than a diagnostic.
    //
    // Only this DIRECTION is constructible today, and that is the point rather than a shortcut: the
    // reverse (SPIR-V declaring an array against a scalar resource) needs reflection to report
    // `descriptor_count != 1`, which is exactly the defect the next stage fixes -- reflection folds an
    // array index into byte-offset arithmetic and reports one binding. So the arm for the reverse
    // direction cannot exist until that lands, and it must land WITH one; asserting it now would be a
    // test whose inputs cannot express the case it claims to cover.
    ShaderResource table_indexed = good;
    table_indexed.table_index_count = 8;
    table_indexed.table_entry_stride = 16;
    table_indexed.table_index_sgpr = 6;
    table_indexed.table_selector_mode = BufferTableSelectorMode::UserSgprIndex;
    for (uint32_t index = 0; index < table_indexed.table_index_count; ++index) {
        ShaderBufferTableEntry entry;
        entry.gpu_addr = good.gpu_addr + index * 0x100u;
        entry.size = good.size;
        entry.stride = good.stride;
        entry.vsharp = {
            static_cast<uint32_t>(entry.gpu_addr),
            static_cast<uint32_t>(entry.gpu_addr >> 32u) | (entry.stride << 16u),
            entry.size / entry.stride,
            (22u << 12u) | 0xfacu,
        };
        table_indexed.table_entries.push_back(entry);
    }
    ShaderResourceTable arity; arity.resources.push_back(table_indexed);
    auto arr = validate_spirv_descriptor_interface(spv, &arity, 0, SpirvShaderStage::Vertex);
    CHECK(!arr.ok() && has_issue(arr, DescriptorIssueCode::ArrayBindingArityMismatch),
          "a table-indexed resource at a binding the shader declares as a single descriptor is rejected");

    // Discriminator: the arity check must not reject the ordinary case. `good` is byte-for-byte the
    // resource above minus the count, so a failure here would mean the new code rejects every scalar
    // binding -- which the arm above could not distinguish from working correctly.
    ShaderResourceTable scalar_ok; scalar_ok.resources.push_back(good);
    auto sok = validate_spirv_descriptor_interface(spv, &scalar_ok, 0, SpirvShaderStage::Vertex);
    CHECK(!has_issue(sok, DescriptorIssueCode::ArrayBindingArityMismatch),
          "an ordinary scalar resource is not reported as an arity mismatch");

    // Stage 4: reflection reports the DECLARED arity. Before this, an array of eight descriptors
    // reflected as one binding and the count was invisible, which is what made the mismatch above
    // undetectable from the shader side.
    const std::vector<uint32_t> array_spv = descriptor_array_test_spirv();
    ShaderResourceTable array_runtime; array_runtime.resources.push_back(table_indexed);
    auto avr = validate_spirv_descriptor_interface(array_spv, &array_runtime, 0,
                                                   SpirvShaderStage::Vertex);
    const SpirvDescriptorBinding* abind = find_spirv_descriptor_binding(avr, 0, 9);
    CHECK(abind && abind->descriptor_count == 8,
          "reflection reports eight descriptors for an OpTypeArray of eight blocks at one binding");

    // The DIRECTION stage 3 could not construct, now constructible and asserted: SPIR-V declaring an
    // array against a runtime table supplying a single descriptor must be rejected. This is the arm
    // stage 3's PR promised would land with stage 4.
    ShaderResourceTable array_vs_scalar; array_vs_scalar.resources.push_back(good);
    auto asr = validate_spirv_descriptor_interface(array_spv, &array_vs_scalar, 0,
                                                   SpirvShaderStage::Vertex);
    CHECK(!asr.ok() && has_issue(asr, DescriptorIssueCode::ArrayBindingArityMismatch),
          "a shader-declared descriptor array bound against a single resource is rejected");

    // Discriminator for the arity REPORT, not merely for the issue: the ordinary fixture must still
    // report exactly one descriptor. Without this, `descriptor_count = 8` everywhere would satisfy the
    // two arms above.
    ShaderResourceTable one; one.resources.push_back(good);
    auto onr = validate_spirv_descriptor_interface(spv, &one, 0, SpirvShaderStage::Vertex);
    const SpirvDescriptorBinding* obind = find_spirv_descriptor_binding(onr, 0, 9);
    CHECK(obind && obind->descriptor_count == 1,
          "an ordinary non-array binding still reflects exactly one descriptor");

    // And the matched pair validates: eight declared against eight supplied raises no arity issue.
    ShaderResource eight = table_indexed;
    ShaderResourceTable matched; matched.resources.push_back(eight);
    auto mar = validate_spirv_descriptor_interface(array_spv, &matched, 0, SpirvShaderStage::Vertex);
    CHECK(mar.ok(),
          "eight declared descriptors against eight complete table entries validate");

    ShaderResource controlled_null = eight;
    controlled_null.table_entries[3].gpu_addr = 0u;
    controlled_null.table_entries[3].size = 0u;
    controlled_null.table_entries[3].stride = 4u;
    controlled_null.table_entries[3].vsharp = {
        0u, 4u << 16u, 0u, 20u << 12u,
    };
    ShaderResourceTable controlled_null_table;
    controlled_null_table.resources.push_back(controlled_null);
    const auto controlled_null_report = validate_spirv_descriptor_interface(
        array_spv, &controlled_null_table, 0, SpirvShaderStage::Vertex);
    CHECK(controlled_null_report.ok(),
          "a non-all-zero zero-record V# remains a valid null table entry");

    ShaderResource missing_payload = eight;
    missing_payload.table_entries.pop_back();
    ShaderResourceTable missing_payload_table;
    missing_payload_table.resources.push_back(missing_payload);
    const auto missing_payload_report = validate_spirv_descriptor_interface(
        array_spv, &missing_payload_table, 0, SpirvShaderStage::Vertex);
    CHECK(!missing_payload_report.ok() &&
              has_issue(missing_payload_report, DescriptorIssueCode::InvalidBufferMetadata),
          "a declared array whose concrete payload is short rejects at the runtime boundary");

    ShaderResource stale_raw = eight;
    stale_raw.table_entries[3].vsharp[2] += 1u;
    ShaderResourceTable stale_raw_table;
    stale_raw_table.resources.push_back(stale_raw);
    const auto stale_raw_report = validate_spirv_descriptor_interface(
        array_spv, &stale_raw_table, 0, SpirvShaderStage::Vertex);
    CHECK(!stale_raw_report.ok() &&
              has_issue(stale_raw_report, DescriptorIssueCode::InvalidBufferMetadata),
          "an entry whose normalized byte span disagrees with its raw V# rejects");

    ShaderResource stale_format = eight;
    stale_format.table_entries[3].vsharp[3] =
        (stale_format.table_entries[3].vsharp[3] & ~(0x7fu << 12u)) |
        (20u << 12u);
    ShaderResourceTable stale_format_table;
    stale_format_table.resources.push_back(stale_format);
    const auto stale_format_report = validate_spirv_descriptor_interface(
        array_spv, &stale_format_table, 0, SpirvShaderStage::Vertex);
    CHECK(!stale_format_report.ok() &&
              has_issue(stale_format_report, DescriptorIssueCode::InvalidBufferMetadata),
          "an entry whose raw V# format disagrees with the parent binding rejects");

    ShaderResource unsupported_control = eight;
    unsupported_control.table_entries[3].vsharp[3] |= 1u << 19u;
    ShaderResourceTable unsupported_control_table;
    unsupported_control_table.resources.push_back(unsupported_control);
    const auto unsupported_control_report = validate_spirv_descriptor_interface(
        array_spv, &unsupported_control_table, 0, SpirvShaderStage::Vertex);
    CHECK(!unsupported_control_report.ok() &&
              has_issue(unsupported_control_report,
                        DescriptorIssueCode::InvalidBufferMetadata),
          "an array V# using an unrepresented control bit rejects at the shared contract");

    ShaderResource unsupported_dst_sel = eight;
    unsupported_dst_sel.table_entries[3].vsharp[3] ^= 1u;
    ShaderResourceTable unsupported_dst_sel_table;
    unsupported_dst_sel_table.resources.push_back(unsupported_dst_sel);
    const auto unsupported_dst_sel_report = validate_spirv_descriptor_interface(
        array_spv, &unsupported_dst_sel_table, 0, SpirvShaderStage::Vertex);
    CHECK(!unsupported_dst_sel_report.ok() &&
              has_issue(unsupported_dst_sel_report,
                        DescriptorIssueCode::InvalidBufferMetadata),
          "same-entry mutation: an array V# using an unrepresented DST_SEL rejects");

    // An UNREADABLE array length is its own value, not folded onto 0 (which means OpTypeRuntimeArray and
    // is treated as compatible with any table size). Reported as a review finding on #2463: collapsing
    // the two made a decode failure the most permissive answer in the space.
    const std::vector<uint32_t> spec_spv = descriptor_array_specconst_test_spirv();
    ShaderResourceTable spec_eight; spec_eight.resources.push_back(eight);
    auto spr = validate_spirv_descriptor_interface(spec_spv, &spec_eight, 0, SpirvShaderStage::Vertex);
    const SpirvDescriptorBinding* sbind = find_spirv_descriptor_binding(spr, 0, 9);
    CHECK(sbind && sbind->descriptor_count == kDescriptorArityUnknown,
          "an OpSpecConstant array length reflects as unknown arity, not as zero or one");

    // The property that matters, and the one the previous revision got wrong: rejected REGARDLESS of what
    // the table supplies. Eight entries is the size that would have been ACCEPTED when unknown collapsed
    // onto 0, so this arm fails on the exact defect rather than merely near it.
    CHECK(!spr.ok() && has_issue(spr, DescriptorIssueCode::ArrayBindingArityMismatch),
          "an unreadable array length is rejected even against a table that supplies eight entries");

    ShaderResourceTable spec_scalar; spec_scalar.resources.push_back(good);
    auto ssr = validate_spirv_descriptor_interface(spec_spv, &spec_scalar, 0, SpirvShaderStage::Vertex);
    CHECK(!ssr.ok() && has_issue(ssr, DescriptorIssueCode::ArrayBindingArityMismatch),
          "an unreadable array length is rejected against a scalar resource too");

    // Discriminator: a genuine OpTypeRuntimeArray must still mean "any length", so the fix above must not
    // have made every unresolved-looking array reject. Zero and unknown are different answers.
    SpirvDescriptorBinding probe{};
    probe.descriptor_count = 0;
    CHECK(probe.descriptor_count != kDescriptorArityUnknown,
          "zero (OpTypeRuntimeArray) and unknown arity are distinct values");

    // --- validate_runtime_descriptor_contract: the mode-carrying overload (#2239 candidate 3) ---
    //
    // The per-draw form takes its mode from the caller so a hot loop can hoist the getenv to submit
    // scope. These four arms pin the property that makes that split safe, in both directions: the
    // overload must honour the mode it is GIVEN and ignore the environment, while the original
    // signature must keep reading the environment LIVE -- this file and test_gpu_capture_render.cpp
    // both arm the variable at runtime, and a cached read would make those arms vacuous rather than
    // failing (#2214, gated by cached_env_arming_logic).
    //
    // `missing` is an empty table against a module that statically uses binding 9, so strict
    // validation must REJECT it. That is what makes each arm falsifiable: without a table that
    // genuinely fails, an arm returning true because validation was switched off would be
    // indistinguishable from one returning true because validation ran and passed.
    set_descriptor_mode("");                         // environment says: validation off
    CHECK(!validate_runtime_descriptor_contract("test", spv, &missing, 0,
                                                SpirvShaderStage::Vertex, "strict"),
          "overload validates on the mode it is passed, with the environment cleared");
    set_descriptor_mode("strict");                   // environment says: validation on
    CHECK(validate_runtime_descriptor_contract("test", spv, &missing, 0,
                                               SpirvShaderStage::Vertex, nullptr),
          "overload ignores the environment: a null mode stays off even when the env says strict");
    // The two arms above would both still pass if the overload read the environment AND the
    // environment happened to agree with the argument. They cannot both pass in that case, which is
    // the point of running them with the env and the argument set to OPPOSITE values.
    CHECK(!validate_runtime_descriptor_contract("test", spv, &missing, 0,
                                                SpirvShaderStage::Vertex),
          "env-reading signature still rejects while the environment says strict");
    set_descriptor_mode("");
    CHECK(validate_runtime_descriptor_contract("test", spv, &missing, 0,
                                               SpirvShaderStage::Vertex),
          "env-reading signature still observes a runtime write that clears the mode");

    ShaderResource short_buffer = good; short_buffer.size = 16;
    ShaderResourceTable short_table; short_table.resources.push_back(short_buffer);
    auto sr = validate_spirv_descriptor_interface(spv, &short_table, 0, SpirvShaderStage::Vertex);
    CHECK(!sr.ok() && has_issue(sr, DescriptorIssueCode::UndersizedBuffer) &&
          sr.issues.back().required_bytes == 20 && sr.issues.back().available_bytes == 16,
          "statically known byte range rejects an undersized buffer with required/available sizes");

    ShaderResource null_buffer = good; null_buffer.gpu_addr = 0; null_buffer.size = 0;
    null_buffer.format = DataFormat::Unknown; null_buffer.num_components = 0;
    ShaderResourceTable null_table; null_table.resources.push_back(null_buffer);
    auto nr = validate_spirv_descriptor_interface(spv, &null_table, 0, SpirvShaderStage::Vertex);
    CHECK(nr.ok(), "an explicitly bound null buffer is valid zero-read semantics, not a missing binding");

    ShaderResource bad_address = good; bad_address.gpu_addr = 0;
    ShaderResourceTable bad_address_table; bad_address_table.resources.push_back(bad_address);
    auto ar = validate_spirv_descriptor_interface(spv, &bad_address_table, 0, SpirvShaderStage::Vertex);
    CHECK(!ar.ok() && has_issue(ar, DescriptorIssueCode::InvalidAddress),
          "a non-null resource range without guest or replay backing is rejected");

    ShaderResource weak_metadata = good; weak_metadata.stride = 0;
    weak_metadata.format = DataFormat::Unknown; weak_metadata.num_components = 0;
    ShaderResourceTable weak_metadata_table; weak_metadata_table.resources.push_back(weak_metadata);
    auto wmr = validate_spirv_descriptor_interface(spv, &weak_metadata_table, 0, SpirvShaderStage::Vertex);
    CHECK(wmr.ok() && has_issue(wmr, DescriptorIssueCode::InvalidBufferMetadata),
          "stride/format anomalies are deterministic warnings when SPIR-V uses raw storage-buffer access");

    const std::vector<uint32_t> image_spv = image_test_spirv();
    ShaderResource image{}; image.cls = ResourceClass::Texture; image.binding = 4;
    image.gpu_addr = 0x56780000; image.size = 4; image.width = image.height = 1;
    image.format = DataFormat::Unorm8; image.num_components = 4;
    ShaderResourceTable image_table; image_table.resources.push_back(image);
    auto ir = validate_spirv_descriptor_interface(
        image_spv, &image_table, 1, SpirvShaderStage::Fragment);
    CHECK(ir.ok() && ir.descriptors.size() == 1 &&
          ir.descriptors[0].kind == SpirvDescriptorKind::CombinedImageSampler &&
          ir.descriptors[0].normalized_sampling && !ir.descriptors[0].texel_access &&
          ir.descriptors[0].sampled_float &&
          ir.descriptors[0].image_dim == 1 && !ir.descriptors[0].image_arrayed &&
          !ir.descriptors[0].image_multisampled,
          "sampled-image reflection validates its concrete 2D non-array texture binding");
    CHECK(ir.descriptors.size() == 1 && !ir.descriptors[0].writable,
          "sampled descriptor is not mistaken for an image output");
    const auto fetch_report = validate_spirv_descriptor_interface(
        image_fetch_test_spirv(), &image_table, 1, SpirvShaderStage::Fragment);
    CHECK(fetch_report.ok() && fetch_report.descriptors.size() == 1 &&
              !fetch_report.descriptors[0].normalized_sampling &&
              fetch_report.descriptors[0].texel_access &&
              fetch_report.descriptors[0].sampled_float,
          "OpImageFetch reflection requires the descriptor's exact texel extent");
    const auto uint_fetch_report = validate_spirv_descriptor_interface(
        image_fetch_test_spirv(false), &image_table, 1, SpirvShaderStage::Fragment);
    CHECK(uint_fetch_report.ok() && uint_fetch_report.descriptors.size() == 1 &&
              !uint_fetch_report.descriptors[0].sampled_float,
          "sampled-image reflection preserves an integer OpTypeImage component contract");
    const auto mixed_access_report = validate_spirv_descriptor_interface(
        image_sample_fetch_test_spirv(), &image_table, 1, SpirvShaderStage::Fragment);
    CHECK(mixed_access_report.ok() && mixed_access_report.descriptors.size() == 1 &&
              mixed_access_report.descriptors[0].normalized_sampling &&
              mixed_access_report.descriptors[0].texel_access &&
              mixed_access_report.descriptors[0].sampled_float,
          "a normalized sample does not hide an unnormalized texel read on the same image");
    const auto query_report = validate_spirv_descriptor_interface(
        image_query_test_spirv(), &image_table, 0, SpirvShaderStage::Compute);
    CHECK(query_report.ok() && query_report.descriptors.size() == 1 &&
              query_report.descriptors[0].readable &&
              !query_report.descriptors[0].normalized_sampling &&
              query_report.descriptors[0].texel_access &&
              !query_report.descriptors[0].sampled_float,
          "query-only sampled images stay reflected and require their exact guest extent");
    const auto sample_query_report = validate_spirv_descriptor_interface(
        image_sample_query_test_spirv(), &image_table, 1, SpirvShaderStage::Fragment);
    CHECK(sample_query_report.ok() && sample_query_report.descriptors.size() == 1 &&
              sample_query_report.descriptors[0].normalized_sampling &&
              sample_query_report.descriptors[0].texel_access &&
              sample_query_report.descriptors[0].sampled_float,
          "image queries preserve exact-extent requirements without blocking normalized values");

    ShaderResource storage_src{};
    storage_src.cls = ResourceClass::StorageImage; storage_src.binding = 5;
    storage_src.gpu_addr = 0x56790000; storage_src.size = 4;
    storage_src.width = storage_src.height = 1;
    storage_src.format = DataFormat::Unorm8; storage_src.num_components = 4;
    ShaderResource storage_dst = storage_src;
    storage_dst.binding = 6; storage_dst.gpu_addr += 0x10000;
    ShaderResourceTable storage_table; storage_table.resources = {storage_src, storage_dst};
    const auto storage_report = validate_spirv_descriptor_interface(
        storage_image_access_test_spirv(), &storage_table, 0, SpirvShaderStage::Compute);
    CHECK(storage_report.ok() && storage_report.descriptors.size() == 2,
          "storage-image read/write fixture reflects both bindings");
    CHECK(storage_report.descriptors.size() == 2 && storage_report.descriptors[0].readable &&
              !storage_report.descriptors[0].writable &&
              !storage_report.descriptors[1].readable && storage_report.descriptors[1].writable &&
              storage_report.descriptors[0].texel_access &&
              !storage_report.descriptors[0].normalized_sampling &&
              !storage_report.descriptors[0].sampled_float &&
              !storage_report.descriptors[0].storage_float &&
              storage_report.descriptors[0].image_numeric_class ==
                  SpirvImageNumericClass::Uint,
          "storage-image texel access is classified per binding");
    const auto float_storage_report = validate_spirv_descriptor_interface(
        storage_image_access_test_spirv(SpirvImageNumericClass::Float),
        &storage_table, 0, SpirvShaderStage::Compute);
    CHECK(float_storage_report.ok() && float_storage_report.descriptors.size() == 2 &&
              float_storage_report.descriptors[0].storage_float &&
              !float_storage_report.descriptors[0].sampled_float &&
              float_storage_report.descriptors[1].storage_float &&
              float_storage_report.descriptors[0].image_numeric_class ==
                  SpirvImageNumericClass::Float,
          "storage-image reflection preserves the SPIR-V float sampled-type contract");
    const auto sint_storage_report = validate_spirv_descriptor_interface(
        storage_image_access_test_spirv(SpirvImageNumericClass::Sint),
        &storage_table, 0, SpirvShaderStage::Compute);
    CHECK(sint_storage_report.ok() && sint_storage_report.descriptors.size() == 2 &&
              !sint_storage_report.descriptors[0].storage_float &&
              sint_storage_report.descriptors[0].image_numeric_class ==
                  SpirvImageNumericClass::Sint,
          "storage-image reflection keeps signed and unsigned integer sampled types distinct");
    ShaderResource atomic_image = storage_src;
    atomic_image.binding = 7;
    atomic_image.format = DataFormat::Uint32;
    atomic_image.num_components = 1;
    ShaderResourceTable atomic_image_table;
    atomic_image_table.resources = {atomic_image};
    const auto atomic_image_report = validate_spirv_descriptor_interface(
        storage_image_atomic_test_spirv(), &atomic_image_table, 1,
        SpirvShaderStage::Fragment);
    CHECK(atomic_image_report.ok() && atomic_image_report.descriptors.size() == 1 &&
              atomic_image_report.descriptors[0].kind == SpirvDescriptorKind::StorageImage &&
              atomic_image_report.descriptors[0].readable &&
              atomic_image_report.descriptors[0].writable &&
              atomic_image_report.descriptors[0].texel_access &&
              !atomic_image_report.descriptors[0].storage_float,
          "image-texel-pointer atomic reflects an exact readable+writable integer storage image");
    CHECK(atomic_image_report.descriptors.size() == 1 &&
              atomic_image_report.descriptors[0].storage_image_format == kSpirvImageFormatR32ui,
          "typed storage-image reflection preserves the exact SPIR-V image format");
    image_table.resources[0].size = 0;
    auto izr = validate_spirv_descriptor_interface(
        image_spv, &image_table, 1, SpirvShaderStage::Fragment);
    CHECK(!izr.ok() && has_issue(izr, DescriptorIssueCode::InvalidImageMetadata),
          "a non-null image with no declared backing range is rejected");
    image_table.resources[0].gpu_addr = 0;
    image_table.resources[0].width = image_table.resources[0].height = 0;
    image_table.resources[0].format = DataFormat::Unknown;
    image_table.resources[0].num_components = 0;
    auto inr = validate_spirv_descriptor_interface(
        image_spv, &image_table, 1, SpirvShaderStage::Fragment);
    CHECK(inr.ok(), "an explicitly bound null image is valid zero-sample semantics");

    ShaderResource unused = good; unused.binding = 12;
    ShaderResourceTable extra; extra.resources = {good, unused};
    auto ur = validate_spirv_descriptor_interface(spv, &extra, 0, SpirvShaderStage::Vertex);
    CHECK(ur.ok() && has_issue(ur, DescriptorIssueCode::UnusedRuntimeBinding),
          "unused runtime binding is reported as a non-fatal warning");
    CHECK(find_spirv_descriptor_binding(ur, 0, good.binding) != nullptr &&
              find_spirv_descriptor_binding(ur, 0, unused.binding) == nullptr,
          "reflected binding lookup distinguishes shader-used resources from runtime extras");

    auto setr = validate_spirv_descriptor_interface(spv, &valid, 1, SpirvShaderStage::Fragment);
    CHECK(!setr.ok() && has_issue(setr, DescriptorIssueCode::SetMismatch) &&
          has_issue(setr, DescriptorIssueCode::StageMismatch),
          "descriptor set and shader stage visibility are validated");

    std::vector<uint32_t> broken = spv; broken.push_back((2u << 16) | 61u);
    auto br = validate_spirv_descriptor_interface(broken, &valid, 0, SpirvShaderStage::Vertex);
    CHECK(!br.ok() && has_issue(br, DescriptorIssueCode::MalformedSpirv),
          "truncated/malformed SPIR-V fails deterministically");

    set_descriptor_mode("strict");
    CHECK(validate_runtime_descriptor_contract("test", spv, &valid, 0, SpirvShaderStage::Vertex),
          "strict runtime gate accepts a valid reflected contract");
    CHECK(!validate_runtime_descriptor_contract("test", spv, &missing, 0, SpirvShaderStage::Vertex),
          "strict runtime gate rejects a missing binding before backend submission");
    set_descriptor_mode("off");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
