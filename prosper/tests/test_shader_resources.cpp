// test_shader_resources — fixes the resource-binding contract (shader_resources.hpp): format sizing
// and the recompiler/pipeline lookups both halves rely on. Pure (no Vulkan), runs in CI.
#include "../src/gpu/shader_resources.hpp"
#include "../src/gpu/gpu_execute.hpp"
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

static std::vector<uint32_t> atomic_descriptor_test_spirv() {
    std::vector<uint32_t> s = descriptor_test_spirv();
    // %12 points at binding 10. AtomicAnd is result-producing, so its pointer is operand 2 after
    // result type/result id. The descriptor is write-only and must still be reflected for binding.
    emit(s, 240, {1, 13, 12, 6, 6, 6});                    // %13 = atomicAnd %12
    return s;
}

static std::vector<uint32_t> image_test_spirv() {
    std::vector<uint32_t> s = {0x07230203u, 0x00010000u, 0, 16, 0};
    emit(s, 15, {4, 10, 0x6e69616d, 0});                    // OpEntryPoint Fragment %10 "main"
    emit(s, 22, {1, 32});                                  // %1 = f32
    emit(s, 25, {2, 1, 1, 0, 0, 0, 1, 0});                // %2 = sampled 2D image
    emit(s, 27, {3, 2});                                   // %3 = sampled-image %2
    emit(s, 32, {4, 0, 3});                                // %4 = UniformConstant pointer
    emit(s, 59, {4, 5, 0});                                // %5 = image variable
    emit(s, 71, {5, 34, 1}); emit(s, 71, {5, 33, 4});      // set 1 binding 4
    emit(s, 61, {3, 6, 5});                                // %6 = load %5
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

    CHECK(float_to_half(0.0f) == 0x0000u && float_to_half(-0.0f) == 0x8000u &&
          float_to_half(1.0f) == 0x3c00u && float_to_half(-2.0f) == 0xc000u &&
          float_to_half(65504.0f) == 0x7bffu,
          "float32 -> float16 conversion preserves zero/sign and exact finite values");
    CHECK(float_to_half(half_to_float(0x0001u)) == 0x0001u &&
          float_to_half(half_to_float(0x3555u)) == 0x3555u &&
          float_to_half(half_to_float(0x7bffu)) == 0x7bffu,
          "float16 -> float32 -> float16 round-trip is bit exact for finite values");

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
          vr.descriptors[0].required_bytes == 20 && !vr.descriptors[0].dynamic_access,
          "constant access chain reflects storage-buffer binding 9 with a 20-byte minimum");

    ShaderResource atomic = good; atomic.binding = 10;
    ShaderResourceTable atomic_table; atomic_table.resources = {good, atomic};
    const DescriptorValidationReport atomic_report = validate_spirv_descriptor_interface(
        atomic_descriptor_test_spirv(), &atomic_table, 0, SpirvShaderStage::Vertex);
    CHECK(atomic_report.ok() && atomic_report.descriptors.size() == 2 &&
              atomic_report.descriptors[1].binding == 10,
          "write-only atomic access reflects its storage-buffer descriptor");

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
          ir.descriptors[0].kind == SpirvDescriptorKind::CombinedImageSampler,
          "sampled-image reflection validates a concrete texture binding");
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
