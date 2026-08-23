// vkprobe — drive a dumped SPIR-V vertex/fragment pair through a bare Vulkan pipeline, with NO
// prosper code anywhere in the process, and report whether the draw rasterizes.
//
// WHY THIS EXISTS
//
// When a draw stops producing pixels, "is this prosper or is this the driver?" is the first question
// and the most expensive one to answer, because every other instrument in this repository runs
// inside prosper. This tool is the control: it links no prosper header, allocates its own device,
// and executes the SAME shader modules prosper produced. A draw that renders here and not through
// the backend points at what prosper does AROUND the draw -- descriptor wiring, synchronisation,
// resource lifetime. It does NOT clear the recompiler: the SPIR-V under test is prosper's own
// output, so this separates prosper's HOST-side Vulkan usage from everything else, and no more.
//
// It also does not clear the driver, and #2945 is why that sentence is here. Run with a VALID
// pipeline this program reproduces the defect it was built to exonerate the driver from: indexed
// draws intermittently rasterizing nothing while the non-indexed arm beside them stays constant, in
// a process containing no prosper code. **A clean run proves nothing on its own** -- see the
// README's "Reading a result".
//
// It was first written for #2937 and reported 1,500 clean indexed draws, which is what turned "RADV
// is broken" into "prosper is broken". That program was deleted during cleanup and rebuilt here from
// the issue comment -- and the rebuild found the reason to distrust the original number: a device
// created without vertexPipelineStoresAndAtomics makes every pipeline built from a prosper vertex
// module INVALID, and with no layers loaded the program reports coverage for it regardless. See the
// device-creation comment below. The tool lives in the tree so the next person inherits the control
// AND the ways it has already been wrong.
//
// WHAT IT MEASURES
//
// Each iteration renders twice into two fresh targets — once with vkCmdDraw over vertices 0..N-1 and
// once with vkCmdBindIndexBuffer + vkCmdDrawIndexed over the supplied indices — then counts pixels
// that differ from the clear colour. The arms are directly comparable only when those indices are
// the identity sequence (the default); with a capture's real indices, read each arm on its own.
// A run reports, per draw kind, how many iterations produced ZERO covered pixels, and the same
// failures split by submission position beside it. An indexed arm that fails while the non-indexed
// arm beside it does not is the #2937 signature; either arm failing on its own is a device or
// shader problem rather than an indexing one; and if the POSITION counts are lopsided while the arm
// counts are not, it is submission order and not indexing at all -- which is why the order
// alternates and both breakdowns are printed.
//
// The descriptor interface is read out of BOTH modules rather than hardcoded: every
// `OpDecorate <id> DescriptorSet 0` + `Binding N` becomes one STORAGE_BUFFER binding, all pointing at
// the same host-visible vertex-record buffer, and anything the tool cannot model is refused. That
// is what prosper's recompiled vertex shaders want (they fetch through V#-derived storage buffers),
// and it keeps the tool usable for any dumped pair without editing it.
//
// USAGE
//
//   vkprobe --vs vs.spv --fs fs.spv [--iterations 300] [--records FILE] [--indices 0,1,2]
//           [--extent 64x64] [--device N] [--verbose] [--device-local-indices]
//
// `--records FILE` supplies the raw bytes of the vertex-record buffer; the default is three records
// of 32 bytes forming a fullscreen triangle in the layout prosper's vertex-fetch shaders use
// (x,y,z,w as the first four dwords of each 32-byte record). Dump a real one with
// `gpu_replay --dump-resource DRAW:vs:BINDING`.
//
// Exit status: 0 when every iteration of both arms covered pixels, 1 otherwise, 2 on a setup error.
// A setup error is never reported as a failing draw — a probe that cannot run must not read as a
// negative result.
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string vs_path, fs_path, records_path;
    // A SECOND module pair, rendered in the SAME process and interleaved render-by-render with the
    // first. #2945's failure rate drifts machine-wide over minutes -- 0/20 in one ten-minute window
    // and 12/12 in the next, with nothing changed -- so an A/B measured as "run arm A, then run arm
    // B" is void here rather than negative. Two pipelines alive at once, alternating within each
    // iteration, is the only shape of this comparison that survives that drift.
    std::string vs_b_path, fs_b_path;
    std::vector<uint32_t> indices{0, 1, 2};
    uint32_t iterations = 300;
    uint32_t width = 64, height = 64;
    uint32_t device_index = 0;
    bool verbose = false;
    // Where the index data lives. The arms differ in more than indexedness -- only the indexed one
    // binds an index buffer -- so if the index FETCH is what reads zeros, the arm attribution is
    // really an attribution to host-coherent index memory. `device` stages the same bytes into
    // DEVICE_LOCAL memory through a transfer, which is what a real application does, and turns that
    // into a single-variable A/B on one binary.
    bool device_local_indices = false;
    // Dwords of the record buffer to reset to a sentinel before every render and read back after
    // it. A vertex shader that STORES what it saw (see shaders/index_readback_vs.spvasm) turns
    // "the draw came back empty" -- which is the same picture whether the shader never ran, ran
    // with all-zero indices, or ran correctly and lost the primitive later -- into three
    // distinguishable answers.
    uint32_t readback_first = 0, readback_count = 0;
    // 16- or 32-bit indices. The defect being characterised is specific to indexed draws, and the
    // index type is one of the few remaining free variables in the draw itself.
    uint32_t index_bits = 32;
};

// A sentinel that is not a plausible vertex index, a plausible marker, or zero, so "never written"
// stays distinguishable from "written with zero".
constexpr uint32_t kReadbackSentinel = 0xdeadbeefu;

[[noreturn]] void fail_setup(const char* what) {
    std::fprintf(stderr, "vkprobe: setup failed: %s\n", what);
    std::exit(2);
}

std::vector<uint32_t> read_spirv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || (bytes % 4) != 0) return {};
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), bytes);
    if (!file || words.empty() || words[0] != 0x07230203u) return {};
    return words;
}

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize bytes = file.tellg();
    std::vector<uint8_t> data(static_cast<size_t>(bytes < 0 ? 0 : bytes));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return data;
}

// Output-interface Locations a module decorates, for the budget refusal below.
std::set<uint32_t> output_locations(const std::vector<uint32_t>& spirv) {
    constexpr uint32_t kOpDecorate = 71, kOpVariable = 59;
    constexpr uint32_t kDecLocation = 30, kStorageClassOutput = 3;
    std::map<uint32_t, uint32_t> location_of;
    std::set<uint32_t> outputs, result;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t count = spirv[word] >> 16;
        const uint32_t opcode = spirv[word] & 0xffffu;
        if (count == 0 || word + count > spirv.size()) break;
        if (opcode == kOpDecorate && count >= 4 && spirv[word + 2] == kDecLocation)
            location_of[spirv[word + 1]] = spirv[word + 3];
        if (opcode == kOpVariable && count >= 4 && spirv[word + 3] == kStorageClassOutput)
            outputs.insert(spirv[word + 2]);
        word += count;
    }
    for (const auto& [id, location] : location_of)
        if (outputs.count(id)) result.insert(location);
    return result;
}

// Storage-buffer bindings the module declares, and a REFUSAL for anything this tool cannot model.
//
// The refusal half matters more than the reflection half. vkprobe builds one descriptor set layout
// with every binding typed STORAGE_BUFFER, so a module that declares a sampled image, or a set other
// than 0, would get a pipeline layout inconsistent with its own shaders
// (VUID-VkGraphicsPipelineCreateInfo-layout-00756). No validation layers are loaded here by default,
// so nothing would say so: the tool would run, print a coverage number, and that number would be
// undefined. A control that answers confidently on inputs it cannot model is worse than no control,
// so those inputs exit 2 instead.
void collect_storage_bindings(const std::vector<uint32_t>& spirv, const char* stage,
                              std::set<uint32_t>& out) {
    constexpr uint32_t kOpDecorate = 71, kOpVariable = 59;
    constexpr uint32_t kDecDescriptorSet = 34, kDecBinding = 33;
    constexpr uint32_t kStorageClassStorageBuffer = 12;
    std::map<uint32_t, uint32_t> set_of, binding_of, storage_class_of;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t count = spirv[word] >> 16;
        const uint32_t opcode = spirv[word] & 0xffffu;
        if (count == 0 || word + count > spirv.size()) break;
        if (opcode == kOpDecorate && count >= 4) {
            if (spirv[word + 2] == kDecDescriptorSet) set_of[spirv[word + 1]] = spirv[word + 3];
            if (spirv[word + 2] == kDecBinding) binding_of[spirv[word + 1]] = spirv[word + 3];
        }
        if (opcode == kOpVariable && count >= 4) storage_class_of[spirv[word + 2]] = spirv[word + 3];
        word += count;
    }
    // A PushConstant block is not a descriptor and so carries no Binding decoration -- it would slip
    // past the loop below and out of the pipeline layout entirely, which is
    // VUID-VkGraphicsPipelineCreateInfo-layout-00756 all over again. prosper's recompiler does emit
    // them, so refuse rather than build a layout that does not describe the shader.
    constexpr uint32_t kStorageClassPushConstant = 9;
    for (const auto& [id, storage_class] : storage_class_of) {
        (void)id;
        if (storage_class != kStorageClassPushConstant) continue;
        std::fprintf(stderr,
                     "vkprobe: the %s module declares a push-constant block; this probe builds a "
                     "pipeline layout with no push-constant range\n", stage);
        fail_setup("unsupported push constants (nothing has been measured)");
    }
    for (const auto& [id, binding] : binding_of) {
        const auto set = set_of.find(id);
        const uint32_t descriptor_set = set == set_of.end() ? 0u : set->second;
        if (descriptor_set != 0) {
            std::fprintf(stderr,
                         "vkprobe: the %s module declares binding %u in descriptor set %u; this "
                         "probe models set 0 only\n", stage, binding, descriptor_set);
            fail_setup("unsupported descriptor set (nothing has been measured)");
        }
        const auto storage_class = storage_class_of.find(id);
        if (storage_class == storage_class_of.end() ||
            storage_class->second != kStorageClassStorageBuffer) {
            std::fprintf(stderr,
                         "vkprobe: the %s module's binding %u is not a StorageBuffer variable "
                         "(storage class %u); this probe types every binding STORAGE_BUFFER\n",
                         stage, binding,
                         storage_class == storage_class_of.end() ? 0xffffffffu
                                                                 : storage_class->second);
            fail_setup("unsupported descriptor type (nothing has been measured)");
        }
        out.insert(binding);
    }
}

uint32_t memory_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags wanted,
                     VkMemoryPropertyFlags forbidden = 0) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(phys, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if (!(bits & (1u << i))) continue;
        const VkMemoryPropertyFlags flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) != wanted) continue;
        // `forbidden` exists because this is a SUPERSET match. Asking for DEVICE_LOCAL on an APU,
        // on lavapipe, or on any device that orders a ReBAR heap first hands back memory that is
        // also HOST_VISIBLE -- so `--device-local-indices` would print "DEVICE_LOCAL (staged)" and
        // change nothing about where the indices live. An experiment that fails silently behind a
        // loud label is worse than no experiment; the caller passes HOST_VISIBLE here and gets
        // UINT32_MAX rather than a false positive.
        if (flags & forbidden) continue;
        return i;
    }
    return UINT32_MAX;
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

Buffer make_host_buffer(VkPhysicalDevice phys, VkDevice device, VkDeviceSize size,
                        VkBufferUsageFlags usage) {
    Buffer buffer;
    buffer.size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    if (vkCreateBuffer(device, &info, nullptr, &buffer.buffer) != VK_SUCCESS)
        fail_setup("vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = memory_type(
        phys, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (allocate.memoryTypeIndex == UINT32_MAX) fail_setup("no HOST_VISIBLE|HOST_COHERENT type");
    if (vkAllocateMemory(device, &allocate, nullptr, &buffer.memory) != VK_SUCCESS)
        fail_setup("vkAllocateMemory");
    if (vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0) != VK_SUCCESS)
        fail_setup("vkBindBufferMemory");
    if (vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped) != VK_SUCCESS)
        fail_setup("vkMapMemory");
    return buffer;
}

// Three 32-byte records forming a triangle that covers the whole viewport, in the layout prosper's
// vertex-fetch shaders address: position in the first four dwords, 32-byte stride.
std::vector<uint8_t> default_records() {
    const float positions[3][4] = {
        {-1.0f, -1.0f, 0.0f, 1.0f},
        { 3.0f, -1.0f, 0.0f, 1.0f},
        {-1.0f,  3.0f, 0.0f, 1.0f},
    };
    std::vector<uint8_t> data(3 * 32, 0);
    for (int record = 0; record < 3; ++record)
        std::memcpy(data.data() + record * 32, positions[record], sizeof(positions[record]));
    return data;
}

bool parse_extent(const char* text, uint32_t& width, uint32_t& height) {
    char* end = nullptr;
    const unsigned long w = std::strtoul(text, &end, 10);
    if (!end || *end != 'x' || !w) return false;
    const unsigned long h = std::strtoul(end + 1, &end, 10);
    if (!end || *end || !h) return false;
    width = static_cast<uint32_t>(w);
    height = static_cast<uint32_t>(h);
    return true;
}

// A malformed value must disable the run, not quietly produce a different measurement -- the same
// convention the PROSPER_* capture triggers use.
uint32_t strict_u32(const char* text, const char* option) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!end || end == text || *end || value > 0xffffffffu) {
        std::fprintf(stderr, "vkprobe: %s expects a number, got '%s'\n", option, text);
        fail_setup("malformed argument (nothing has been measured)");
    }
    return static_cast<uint32_t>(value);
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --vs FILE --fs FILE [--iterations N] [--records FILE]\n"
                 "          [--indices i,j,k] [--extent WxH] [--device N] [--verbose]\n"
                 "          [--device-local-indices] [--vs-b FILE --fs-b FILE]\n"
                 "          [--readback-dwords FIRST:COUNT] [--index-bits 16|32]\n", argv0);
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(2); }
            (void)name;
            return argv[++i];
        };
        if (argument == "--vs") options.vs_path = next("--vs");
        else if (argument == "--fs") options.fs_path = next("--fs");
        else if (argument == "--vs-b") options.vs_b_path = next("--vs-b");
        else if (argument == "--fs-b") options.fs_b_path = next("--fs-b");
        else if (argument == "--index-bits") {
            options.index_bits = strict_u32(next("--index-bits"), "--index-bits");
            if (options.index_bits != 16 && options.index_bits != 32)
                fail_setup("--index-bits accepts 16 or 32 (nothing has been measured)");
        } else if (argument == "--readback-dwords") {
            // FIRST:COUNT, and a malformed value disables the run rather than quietly measuring a
            // different region -- the same convention as strict_u32 above.
            const char* text = next("--readback-dwords");
            const char* colon = std::strchr(text, ':');
            if (!colon) fail_setup("--readback-dwords expects FIRST:COUNT");
            options.readback_first = strict_u32(std::string(text, colon).c_str(),
                                                "--readback-dwords FIRST");
            options.readback_count = strict_u32(colon + 1, "--readback-dwords COUNT");
            if (!options.readback_count) fail_setup("--readback-dwords COUNT 0 reads nothing");
        }
        else if (argument == "--records") options.records_path = next("--records");
        else if (argument == "--iterations")
            options.iterations = strict_u32(next("--iterations"), "--iterations");
        else if (argument == "--device") options.device_index = strict_u32(next("--device"), "--device");
        else if (argument == "--extent") {
            if (!parse_extent(next("--extent"), options.width, options.height)) { usage(argv[0]); return 2; }
        } else if (argument == "--indices") {
            options.indices.clear();
            const char* cursor = next("--indices");
            while (*cursor) {
                char* end = nullptr;
                options.indices.push_back(static_cast<uint32_t>(std::strtoul(cursor, &end, 10)));
                if (end == cursor) break;
                cursor = (*end == ',') ? end + 1 : end;
            }
        } else if (argument == "--verbose") options.verbose = true;
        else if (argument == "--device-local-indices") options.device_local_indices = true;
        else { usage(argv[0]); return 2; }
    }
    if (options.vs_path.empty() || options.fs_path.empty()) { usage(argv[0]); return 2; }
    if (options.vs_b_path.empty() != options.fs_b_path.empty())
        fail_setup("--vs-b and --fs-b go together (nothing has been measured)");
    if (!options.iterations) fail_setup("--iterations 0 measures nothing");
    if (options.indices.size() < 3) fail_setup("--indices needs at least three values");

    const std::vector<uint32_t> vs = read_spirv(options.vs_path);
    const std::vector<uint32_t> fs = read_spirv(options.fs_path);
    if (vs.empty()) fail_setup("cannot read the vertex module (missing, truncated, or not SPIR-V)");
    if (fs.empty()) fail_setup("cannot read the fragment module (missing, truncated, or not SPIR-V)");
    const bool paired = !options.vs_b_path.empty();
    std::vector<uint32_t> vs_b, fs_b;
    if (paired) {
        vs_b = read_spirv(options.vs_b_path);
        fs_b = read_spirv(options.fs_b_path);
        if (vs_b.empty()) fail_setup("cannot read --vs-b");
        if (fs_b.empty()) fail_setup("cannot read --fs-b");
    }
    std::vector<uint8_t> records =
        options.records_path.empty() ? default_records() : read_bytes(options.records_path);
    if (records.empty()) fail_setup("cannot read --records");

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.apiVersion = VK_API_VERSION_1_2;
    application.pApplicationName = "vkprobe";
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS)
        fail_setup("vkCreateInstance");

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
    if (options.device_index >= device_count) fail_setup("--device out of range");
    VkPhysicalDevice phys = devices[options.device_index];
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(phys, &properties);

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, families.data());
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i)
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { family = i; break; }
    if (family == UINT32_MAX) fail_setup("no graphics queue family");

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    // vertexPipelineStoresAndAtomics, and this line is the difference between a control and a lie.
    // prosper's recompiled VERTEX shaders fetch through STORAGE_BUFFER descriptors, and Vulkan
    // requires every storage buffer in the vertex stage to be decorated NonWritable UNLESS this
    // feature is enabled (VUID-RuntimeSpirv-NonWritable-06341). prosper's own renderer enables it
    // (render_runner.h). The first version of this program did not, so every pipeline it ever built
    // from a prosper vertex module was INVALID -- and it reported coverage numbers anyway, because
    // it loads no validation layers. Both its clean runs and its failures were void, and one of each
    // had already been quoted as evidence about the driver.
    VkPhysicalDeviceFeatures supported_features{};
    vkGetPhysicalDeviceFeatures(phys, &supported_features);
    if (!supported_features.vertexPipelineStoresAndAtomics)
        fail_setup("the device does not support vertexPipelineStoresAndAtomics, which prosper's "
                   "vertex modules require (nothing has been measured)");
    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.vertexPipelineStoresAndAtomics = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &enabled_features;
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &device_info, nullptr, &device) != VK_SUCCESS)
        fail_setup("vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &queue);

    std::printf("[vkprobe] device=%s driver=%u api=%u.%u.%u iterations=%u extent=%ux%u\n",
                properties.deviceName, properties.driverVersion,
                VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion),
                VK_VERSION_PATCH(properties.apiVersion),
                options.iterations, options.width, options.height);

    // --- resources -----------------------------------------------------------------------------
    Buffer record_buffer = make_host_buffer(phys, device, records.size(),
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memcpy(record_buffer.mapped, records.data(), records.size());

    if (options.readback_count &&
        (uint64_t(options.readback_first) + options.readback_count) * 4 > records.size())
        fail_setup("--readback-dwords runs past the end of the record buffer (nothing measured)");
    const VkDeviceSize index_stride = options.index_bits / 8;
    const VkDeviceSize index_bytes = options.indices.size() * index_stride;
    uint32_t index_memory_type = UINT32_MAX;
    // N21: TRANSFER_SRC only in the staged arm. An added usage bit can narrow
    // memoryRequirements.memoryTypeBits, so putting it on the default path would mean the
    // host-index arm of an A/B is not the configuration every earlier number was measured on.
    Buffer index_staging = make_host_buffer(
        phys, device, index_bytes,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            (options.device_local_indices ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0u));
    if (options.index_bits == 32) {
        std::memcpy(index_staging.mapped, options.indices.data(), index_bytes);
    } else {
        auto* narrow = static_cast<uint16_t*>(index_staging.mapped);
        for (size_t i = 0; i < options.indices.size(); ++i) {
            if (options.indices[i] > 0xffffu)
                fail_setup("--indices holds a value 16-bit indices cannot carry");
            narrow[i] = static_cast<uint16_t>(options.indices[i]);
        }
    }
    // Device-local index buffer, filled by a transfer, when asked for. Everything after this point
    // binds `index_buffer`, which is either the host-coherent staging allocation itself or a
    // DEVICE_LOCAL copy of it.
    Buffer index_device;
    if (options.device_local_indices) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = index_bytes;
        info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(device, &info, nullptr, &index_device.buffer) != VK_SUCCESS)
            fail_setup("vkCreateBuffer (device-local indices)");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, index_device.buffer, &requirements);
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memory_type(phys, requirements.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (allocate.memoryTypeIndex == UINT32_MAX)
            fail_setup("this device exposes no DEVICE_LOCAL memory type that is not also "
                       "HOST_VISIBLE, so --device-local-indices could not change where the indices "
                       "live (nothing has been measured)");
        if (vkAllocateMemory(device, &allocate, nullptr, &index_device.memory) != VK_SUCCESS)
            fail_setup("vkAllocateMemory (device-local indices)");
        if (vkBindBufferMemory(device, index_device.buffer, index_device.memory, 0) != VK_SUCCESS)
            fail_setup("vkBindBufferMemory (device-local indices)");
        index_device.size = index_bytes;
        index_memory_type = allocate.memoryTypeIndex;
    }
    const VkBuffer index_binding =
        options.device_local_indices ? index_device.buffer : index_staging.buffer;
    Buffer& index_buffer = index_staging;   // named for the teardown loop below

    const VkDeviceSize readback_bytes =
        VkDeviceSize(options.width) * options.height * 4;
    Buffer readback = make_host_buffer(phys, device, readback_bytes,
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Refuse a vertex interface the device cannot carry, for the same reason the descriptor
    // refusals exist: an over-budget interface is an INVALID pipeline
    // (VUID-RuntimeSpirv-Location-06272), the driver's behaviour is then undefined, and with no
    // layers loaded this program would print a coverage number for it anyway. #2945's whole subject
    // is a 132-component vertex interface, so this is not hypothetical -- feeding vkprobe a module
    // dumped before that bound was applied would produce exactly the confident, meaningless answer
    // this tool exists not to give.
    const std::vector<uint32_t>* vertex_modules[2] = {&vs, &vs_b};
    for (const std::vector<uint32_t>* module : vertex_modules) {
        if (module->empty()) continue;
        const std::set<uint32_t> vertex_out = output_locations(*module);
        const uint32_t highest = vertex_out.empty() ? 0u : (*vertex_out.rbegin() + 1u);
        const uint32_t components = highest * 4u + 4u;   // + gl_Position, as the layer counts it
        if (!vertex_out.empty() && components > properties.limits.maxVertexOutputComponents) {
            std::fprintf(stderr,
                         "vkprobe: the vertex module declares %u output locations = %u components "
                         "with gl_Position, over this device's maxVertexOutputComponents of %u\n",
                         highest, components, properties.limits.maxVertexOutputComponents);
            fail_setup("over-budget vertex interface (nothing has been measured)");
        }
    }
    // One descriptor set layout serves BOTH pipelines, so it is the UNION of what all the supplied
    // modules declare. A pipeline may legally carry a binding its shader never reads, and a shared
    // layout is what makes the paired comparison a comparison of the MODULES rather than of two
    // differently-shaped pipelines. It does mean a paired run is not bit-identical in setup to the
    // same module run alone -- run both ways when that could matter.
    std::set<uint32_t> bindings;
    collect_storage_bindings(vs, "vertex", bindings);
    collect_storage_bindings(fs, "fragment", bindings);
    if (paired) {
        collect_storage_bindings(vs_b, "vertex (b)", bindings);
        collect_storage_bindings(fs_b, "fragment (b)", bindings);
    }
    if (bindings.empty()) fail_setup("neither module declares a set-0 storage-buffer binding");
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
    for (uint32_t binding : bindings)
        layout_bindings.push_back({binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   nullptr});
    VkDescriptorSetLayoutCreateInfo set_layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_layout_info.bindingCount = static_cast<uint32_t>(layout_bindings.size());
    set_layout_info.pBindings = layout_bindings.data();
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &set_layout) != VK_SUCCESS)
        fail_setup("vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                   static_cast<uint32_t>(layout_bindings.size())};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &pool) != VK_SUCCESS)
        fail_setup("vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo set_allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_allocate.descriptorPool = pool;
    set_allocate.descriptorSetCount = 1;
    set_allocate.pSetLayouts = &set_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &set_allocate, &descriptor_set) != VK_SUCCESS)
        fail_setup("vkAllocateDescriptorSets");
    std::vector<VkDescriptorBufferInfo> buffer_infos(layout_bindings.size());
    std::vector<VkWriteDescriptorSet> writes(layout_bindings.size());
    for (size_t i = 0; i < layout_bindings.size(); ++i) {
        buffer_infos[i] = {record_buffer.buffer, 0, record_buffer.size};
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = layout_bindings[i].binding;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
        fail_setup("vkCreatePipelineLayout");

    auto make_module = [&](const std::vector<uint32_t>& words) {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = words.size() * 4;
        info.pCode = words.data();
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device, &info, nullptr, &module) != VK_SUCCESS)
            fail_setup("vkCreateShaderModule");
        return module;
    };
    VkShaderModule vs_module = make_module(vs), fs_module = make_module(fs);
    VkShaderModule vs_b_module = VK_NULL_HANDLE, fs_b_module = VK_NULL_HANDLE;
    if (paired) { vs_b_module = make_module(vs_b); fs_b_module = make_module(fs_b); }

    VkAttachmentDescription attachment{};
    attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;
    // The dependency prosper's own render passes were missing (#2945): make the pass's writes and
    // the final layout transition visible to the transfer that reads them.
    VkSubpassDependency dependency{};
    dependency.srcSubpass = 0;
    dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dependency.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkRenderPassCreateInfo render_pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &attachment;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    if (vkCreateRenderPass(device, &render_pass_info, nullptr, &render_pass) != VK_SUCCESS)
        fail_setup("vkCreateRenderPass");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs_module;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs_module;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertex_input{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo input_assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport viewport{0.0f, 0.0f, float(options.width), float(options.height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {options.width, options.height}};
    VkPipelineViewportStateCreateInfo viewport_state{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo rasterization{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    VkGraphicsPipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterization;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pColorBlendState = &blend;
    pipeline_info.layout = pipeline_layout;
    pipeline_info.renderPass = render_pass;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) !=
        VK_SUCCESS)
        fail_setup("vkCreateGraphicsPipelines (an invalid shader interface fails here)");
    VkPipeline pipeline_b = VK_NULL_HANDLE;
    if (paired) {
        VkPipelineShaderStageCreateInfo stages_b[2] = {stages[0], stages[1]};
        stages_b[0].module = vs_b_module;
        stages_b[1].module = fs_b_module;
        VkGraphicsPipelineCreateInfo info_b = pipeline_info;
        info_b.pStages = stages_b;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info_b, nullptr, &pipeline_b) !=
            VK_SUCCESS)
            fail_setup("vkCreateGraphicsPipelines (--vs-b/--fs-b)");
    }

    VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_info.queueFamilyIndex = family;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool) != VK_SUCCESS)
        fail_setup("vkCreateCommandPool");

    // One render of the triangle, indexed or not, into a fresh image. Returns the covered-pixel
    // count (pixels that differ from the blue clear).
    // What one render reports back: how many pixels differ from the clear, and -- when
    // --readback-dwords is on -- the sentinel window the vertex shader may have written.
    struct RenderResult {
        uint64_t covered = 0;
        std::vector<uint32_t> words;
    };
    auto render = [&](VkPipeline use_pipeline, bool indexed) -> RenderResult {
        // Reset the readback window BEFORE recording, so it is a host write ordered ahead of the
        // submit by the implicit host-write domain operation vkQueueSubmit performs.
        if (options.readback_count) {
            auto* dwords = static_cast<uint32_t*>(record_buffer.mapped);
            for (uint32_t i = 0; i < options.readback_count; ++i)
                dwords[options.readback_first + i] = kReadbackSentinel;
        }
        VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        image_info.extent = {options.width, options.height, 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage image = VK_NULL_HANDLE;
        if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS)
            fail_setup("vkCreateImage");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, image, &requirements);
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memory_type(phys, requirements.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (allocate.memoryTypeIndex == UINT32_MAX) fail_setup("no DEVICE_LOCAL memory type");
        VkDeviceMemory image_memory = VK_NULL_HANDLE;
        if (vkAllocateMemory(device, &allocate, nullptr, &image_memory) != VK_SUCCESS)
            fail_setup("vkAllocateMemory (image)");
        if (vkBindImageMemory(device, image, image_memory, 0) != VK_SUCCESS)
            fail_setup("vkBindImageMemory (image)");
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        if (vkCreateImageView(device, &view_info, nullptr, &view) != VK_SUCCESS)
            fail_setup("vkCreateImageView");
        VkFramebufferCreateInfo framebuffer_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer_info.renderPass = render_pass;
        framebuffer_info.attachmentCount = 1;
        framebuffer_info.pAttachments = &view;
        framebuffer_info.width = options.width;
        framebuffer_info.height = options.height;
        framebuffer_info.layers = 1;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        if (vkCreateFramebuffer(device, &framebuffer_info, nullptr, &framebuffer) != VK_SUCCESS)
            fail_setup("vkCreateFramebuffer");

        VkCommandBufferAllocateInfo command_allocate{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_allocate.commandPool = command_pool;
        command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_allocate.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &command_allocate, &command) != VK_SUCCESS)
            fail_setup("vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS)
            fail_setup("vkBeginCommandBuffer");
        VkClearValue clear{};
        clear.color = {{0.0f, 0.0f, 1.0f, 1.0f}};
        VkRenderPassBeginInfo pass_begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        pass_begin.renderPass = render_pass;
        pass_begin.framebuffer = framebuffer;
        pass_begin.renderArea = scissor;
        pass_begin.clearValueCount = 1;
        pass_begin.pClearValues = &clear;
        vkCmdBeginRenderPass(command, &pass_begin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, use_pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
                                &descriptor_set, 0, nullptr);
        if (indexed) {
            vkCmdBindIndexBuffer(command, index_binding, 0,
                                 options.index_bits == 16 ? VK_INDEX_TYPE_UINT16
                                                          : VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(command, static_cast<uint32_t>(options.indices.size()), 1, 0, 0, 0);
        } else {
            vkCmdDraw(command, static_cast<uint32_t>(options.indices.size()), 1, 0, 0);
        }
        vkCmdEndRenderPass(command);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {options.width, options.height, 1};
        vkCmdCopyImageToBuffer(command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               readback.buffer, 1, &copy);
        VkMemoryBarrier host_read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        // SHADER_WRITE as well as TRANSFER_WRITE: with --readback-dwords the vertex shader itself
        // writes the record buffer, and making that available to the host is a separate operation
        // from waiting the fence. Ordering is not availability -- the distinction #2944 is about.
        host_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        host_read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_read, 0, nullptr, 0, nullptr);
        if (vkEndCommandBuffer(command) != VK_SUCCESS) fail_setup("vkEndCommandBuffer");

        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS)
            fail_setup("vkCreateFence");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        if (vkQueueSubmit(queue, 1, &submit, fence) != VK_SUCCESS) fail_setup("vkQueueSubmit");
        // A timeout or a device loss here is the state this control gets pointed at, and reading the
        // staging buffer anyway would report stale bytes as a coverage number -- exit 1, "a failing
        // draw", for something that never ran. The contract at the top of this file says 2.
        if (vkWaitForFences(device, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) != VK_SUCCESS)
            fail_setup("vkWaitForFences did not complete (hang or device loss); nothing measured");

        RenderResult result;
        const uint8_t* pixels = static_cast<const uint8_t*>(readback.mapped);
        for (uint64_t i = 0; i < uint64_t(options.width) * options.height; ++i) {
            const uint8_t* pixel = pixels + i * 4;
            if (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 255) ++result.covered;
        }
        if (options.readback_count) {
            const auto* dwords = static_cast<const uint32_t*>(record_buffer.mapped);
            result.words.assign(dwords + options.readback_first,
                                dwords + options.readback_first + options.readback_count);
        }

        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, command_pool, 1, &command);
        vkDestroyFramebuffer(device, framebuffer, nullptr);
        vkDestroyImageView(device, view, nullptr);
        vkDestroyImage(device, image, nullptr);
        vkFreeMemory(device, image_memory, nullptr);
        return result;
    };

    // ORDER IS ALTERNATED, and that is not tidiness. Running non-indexed then indexed every time
    // confounds the arm with its POSITION: "the indexed draw came back empty" is then exactly as
    // well supported as "the second submit came back empty", and this tool's whole purpose is to
    // attribute a failure to the draw KIND. Swapping the order on odd iterations separates them, and
    // both breakdowns are reported so a reader can see which one moved.
    if (options.device_local_indices) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = command_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer upload = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &ai, &upload) != VK_SUCCESS)
            fail_setup("vkAllocateCommandBuffers (index upload)");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(upload, &bi) != VK_SUCCESS)
            fail_setup("vkBeginCommandBuffer (index upload)");
        VkBufferCopy copy{0, 0, index_bytes};
        vkCmdCopyBuffer(upload, index_staging.buffer, index_device.buffer, 1, &copy);
        VkMemoryBarrier to_index{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        to_index.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_index.dstAccessMask = VK_ACCESS_INDEX_READ_BIT;
        vkCmdPipelineBarrier(upload, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &to_index,
                             0, nullptr, 0, nullptr);
        if (vkEndCommandBuffer(upload) != VK_SUCCESS) fail_setup("vkEndCommandBuffer (index)");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(device, &fi, nullptr, &fence) != VK_SUCCESS)
            fail_setup("vkCreateFence (index upload)");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &upload;
        if (vkQueueSubmit(queue, 1, &submit, fence) != VK_SUCCESS)
            fail_setup("vkQueueSubmit (index upload)");
        if (vkWaitForFences(device, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) != VK_SUCCESS)
            fail_setup("index upload did not complete");
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, command_pool, 1, &upload);
    }
    if (options.device_local_indices) {
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(phys, &memory_properties);
        std::printf("[vkprobe] index memory: DEVICE_LOCAL (staged), memory type %u, property "
                    "flags 0x%x -- not HOST_VISIBLE, checked\n",
                    index_memory_type,
                    memory_properties.memoryTypes[index_memory_type].propertyFlags);
    } else {
        std::printf("[vkprobe] index memory: HOST_VISIBLE|HOST_COHERENT, bound directly\n");
    }

    // One module pair's tally. Two of these exist in a paired run.
    struct Tally {
        const char* label = "";
        uint32_t plain_empty = 0, indexed_empty = 0, mismatched = 0;
        uint64_t plain_min = UINT64_MAX, plain_max = 0;
        uint64_t indexed_min = UINT64_MAX, indexed_max = 0;
        std::map<std::string, uint32_t> plain_words, indexed_words;
    };
    auto describe_words = [](const std::vector<uint32_t>& words) {
        std::string text;
        for (size_t i = 0; i < words.size(); ++i) {
            char buffer[16];
            if (words[i] == kReadbackSentinel) std::snprintf(buffer, sizeof buffer, "--");
            else std::snprintf(buffer, sizeof buffer, "%u", words[i]);
            if (i) text += ",";
            text += buffer;
        }
        return text;
    };
    auto record = [&](Tally& tally, const RenderResult& plain, const RenderResult& indexed) {
        if (!plain.covered) ++tally.plain_empty;
        if (!indexed.covered) ++tally.indexed_empty;
        if (plain.covered != indexed.covered) ++tally.mismatched;
        tally.plain_min = std::min(tally.plain_min, plain.covered);
        tally.plain_max = std::max(tally.plain_max, plain.covered);
        tally.indexed_min = std::min(tally.indexed_min, indexed.covered);
        tally.indexed_max = std::max(tally.indexed_max, indexed.covered);
        if (options.readback_count) {
            ++tally.plain_words[describe_words(plain.words)];
            ++tally.indexed_words[describe_words(indexed.words)];
        }
    };

    Tally tally_a, tally_b;
    tally_a.label = "a";
    tally_b.label = "b";
    uint32_t position_empty[4] = {0, 0, 0, 0};
    const uint32_t positions = paired ? 4u : 2u;
    for (uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        const bool indexed_first = (iteration & 1u) != 0;
        if (!paired) {
            const RenderResult first = render(pipeline, indexed_first);
            const RenderResult second = render(pipeline, !indexed_first);
            record(tally_a, indexed_first ? second : first, indexed_first ? first : second);
            if (!first.covered) ++position_empty[0];
            if (!second.covered) ++position_empty[1];
            if (options.verbose)
                std::printf("[vkprobe] iteration=%u %s plain=%llu indexed=%llu\n", iteration,
                            indexed_first ? "indexed-first" : "plain-first",
                            (unsigned long long)(indexed_first ? second : first).covered,
                            (unsigned long long)(indexed_first ? first : second).covered);
            continue;
        }
        // ABBA within the iteration, with the leading MODULE alternating on top of the leading ARM.
        // The modules are the thing under comparison, so they must not sit at fixed submission
        // positions: over four iterations each module occupies each of the four slots equally, and
        // the by-position line below is what shows whether position moved instead of module.
        const bool b_first = (iteration & 2u) != 0;
        VkPipeline lead = b_first ? pipeline_b : pipeline;
        VkPipeline trail = b_first ? pipeline : pipeline_b;
        const RenderResult r0 = render(lead, indexed_first);
        const RenderResult r1 = render(trail, indexed_first);
        const RenderResult r2 = render(trail, !indexed_first);
        const RenderResult r3 = render(lead, !indexed_first);
        const RenderResult& lead_plain = indexed_first ? r3 : r0;
        const RenderResult& lead_indexed = indexed_first ? r0 : r3;
        const RenderResult& trail_plain = indexed_first ? r2 : r1;
        const RenderResult& trail_indexed = indexed_first ? r1 : r2;
        record(b_first ? tally_b : tally_a, lead_plain, lead_indexed);
        record(b_first ? tally_a : tally_b, trail_plain, trail_indexed);
        const RenderResult* slots[4] = {&r0, &r1, &r2, &r3};
        for (uint32_t slot = 0; slot < 4; ++slot)
            if (!slots[slot]->covered) ++position_empty[slot];
        if (options.verbose)
            std::printf("[vkprobe] iteration=%u %s %s a-plain=%llu a-indexed=%llu b-plain=%llu "
                        "b-indexed=%llu\n", iteration, b_first ? "b-first" : "a-first",
                        indexed_first ? "indexed-first" : "plain-first",
                        (unsigned long long)(b_first ? trail_plain : lead_plain).covered,
                        (unsigned long long)(b_first ? trail_indexed : lead_indexed).covered,
                        (unsigned long long)(b_first ? lead_plain : trail_plain).covered,
                        (unsigned long long)(b_first ? lead_indexed : trail_indexed).covered);
    }

    auto report = [&](const Tally& tally, const char* prefix) {
        std::printf("[vkprobe] %snon-indexed: covered [%llu..%llu], EMPTY on %u of %u\n", prefix,
                    (unsigned long long)tally.plain_min, (unsigned long long)tally.plain_max,
                    tally.plain_empty, options.iterations);
        std::printf("[vkprobe] %sindexed:     covered [%llu..%llu], EMPTY on %u of %u\n", prefix,
                    (unsigned long long)tally.indexed_min, (unsigned long long)tally.indexed_max,
                    tally.indexed_empty, options.iterations);
        std::printf("[vkprobe] %sthe two arms disagreed on %u of %u iterations\n", prefix,
                    tally.mismatched, options.iterations);
        if (!options.readback_count) return;
        // "--" is the sentinel, i.e. a dword no vertex invocation wrote. Read the patterns, not
        // just the coverage: they separate "the shader never ran" from "it ran on index 0 three
        // times" from "it ran correctly and the primitive was lost afterwards".
        for (const auto& [pattern, count] : tally.plain_words)
            std::printf("[vkprobe] %snon-indexed vertex-index readback [%s] x%u\n", prefix,
                        pattern.c_str(), count);
        for (const auto& [pattern, count] : tally.indexed_words)
            std::printf("[vkprobe] %sindexed     vertex-index readback [%s] x%u\n", prefix,
                        pattern.c_str(), count);
    };
    if (paired) {
        std::printf("[vkprobe] a = %s + %s\n", options.vs_path.c_str(), options.fs_path.c_str());
        std::printf("[vkprobe] b = %s + %s\n", options.vs_b_path.c_str(), options.fs_b_path.c_str());
        report(tally_a, "a: ");
        report(tally_b, "b: ");
    } else {
        report(tally_a, "");
    }
    // The same failures split by POSITION rather than by draw kind or module. Three readings, not
    // two: arm counts lopsided and position counts comparable means the defect follows the draw
    // kind; position lopsided and arm comparable means it follows submission order and the arm
    // attribution is an artefact; BOTH lopsided is neither, and is what a period-4 effect looks
    // like through this strict-parity alternation -- the alternation kills period 2, not period 4.
    std::printf("[vkprobe] by submission position (order alternates each iteration):");
    for (uint32_t slot = 0; slot < positions; ++slot)
        std::printf(" #%u EMPTY on %u", slot, position_empty[slot]);
    std::printf("\n");

    vkDestroyPipeline(device, pipeline, nullptr);
    if (pipeline_b) vkDestroyPipeline(device, pipeline_b, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyRenderPass(device, render_pass, nullptr);
    vkDestroyShaderModule(device, vs_module, nullptr);
    vkDestroyShaderModule(device, fs_module, nullptr);
    if (vs_b_module) vkDestroyShaderModule(device, vs_b_module, nullptr);
    if (fs_b_module) vkDestroyShaderModule(device, fs_b_module, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
    for (Buffer* buffer : {&record_buffer, &index_buffer, &readback}) {
        vkUnmapMemory(device, buffer->memory);
        vkDestroyBuffer(device, buffer->buffer, nullptr);
        vkFreeMemory(device, buffer->memory, nullptr);
    }
    if (index_device.buffer) {
        vkDestroyBuffer(device, index_device.buffer, nullptr);
        vkFreeMemory(device, index_device.memory, nullptr);
    }
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return (tally_a.plain_empty || tally_a.indexed_empty || tally_b.plain_empty ||
            tally_b.indexed_empty)
               ? 1
               : 0;
}
