// test_image_robustness — the device-side half of the recompiler's out-of-bounds image contract
// (#3531).
//
// The recompiled storage-image load issues an OpImageRead for ALL invocations, including
// EXEC-inactive lanes whose coordinate may be out of range; the value is discarded by write-back
// predication. That is defined ONLY when the device enabled `robustImageAccess`. The compute
// backend's own device did not chain the feature while the renderer's did, and no test could see it
// because every execution harness in this repository chains it too.
//
// That last sentence is why this file exists and why it is shaped the way it is. A test that builds
// its device the way the harnesses do can only ever exercise the ACCEPTING branch: no device
// available to this project lacks `robustImageAccess`, so the refusing case is not merely rare, it
// is unreachable from a real device. The positive-control rule in CLAUDE.md names that failure — a
// control drawn from the same source tests the discriminator, not the domain — so the refusing
// instance below is constructed BY HAND, outside anything that queries a driver.
//
// Three arms, and it is worth being precise about what each does and does not prove:
//
//  1. the pure decision, including the hand-built device that lacks robustness -> must refuse;
//  2. the SHIPPED helper against a real physical device: the feature struct must actually appear in
//     the VkDeviceCreateInfo pNext chain, and an unrelated struct already on that chain must
//     survive. This is what pins "chained", rather than a copy of the chaining code asserting about
//     itself;
//  3. both device paths' capability is computed by the same function, so a run cannot enable the
//     feature on one device and silently skip it on the other.
//
// What no arm here proves is that the compute backend CALLS the helper — that is asserted on the
// real backend by test_game_compute (`live_compute_storage_image_device()`), and enforced by
// construction, since `image_support` is assigned from the helper's verdict and from nowhere else.

#include "shared/device/image_robustness.hpp"
#include "shared/device/vulkan_runtime.hpp"
#include "gpu/execute/gpu_execute.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

static int fails = 0;
static int checks = 0;
#define CHECK(c, m) do { ++checks; if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

using prosper::frontend::storage_image_device_features;

int main() {
    // Unbuffered: a crash must not swallow the arms that already reported. Without this a fault
    // anywhere below turns a run that had printed twelve [ok] lines into a run that printed none,
    // which reads as "the test never started".
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("== test_image_robustness ==\n");

    // --- 1. The pure decision, hand-built ------------------------------------------------------
    // The #3531 device: it can express the raw uvec4 texel model (both format-free features) and
    // cannot make the recompiler's deliberate out-of-range reads defined. Executing there is UB, so
    // the only correct verdict is refusal. No driver on this project's machines produces this input,
    // which is exactly why it is written out literally.
    const auto no_robustness = storage_image_device_features(false, true, true);
    CHECK(!no_robustness.storage_image_capable(),
          "a device with both format-free features but NO robustImageAccess refuses storage images");
    CHECK(!no_robustness.robust_image_access, "the refusing device reports the missing term");

    const auto complete = storage_image_device_features(true, true, true);
    CHECK(complete.storage_image_capable(),
          "a device with all three terms may execute recompiled storage-image kernels");

    // The other two terms still gate independently: robustness alone is not a licence to run a
    // kernel whose capabilities the device cannot declare.
    CHECK(!storage_image_device_features(true, false, true).storage_image_capable(),
          "robustness does not substitute for shaderStorageImageReadWithoutFormat");
    CHECK(!storage_image_device_features(true, true, false).storage_image_capable(),
          "robustness does not substitute for shaderStorageImageWriteWithoutFormat");
    CHECK(!storage_image_device_features(false, false, false).storage_image_capable(),
          "a device offering none of the three refuses");

    // --- 3. One decision function, both device paths -------------------------------------------
    // The adopt path feeds the renderer's PUBLISHED capabilities through the same function the
    // own-device path feeds its physical-device query through. Divergence between those two is the
    // defect this guards: the renderer chained the feature and the compute device did not.
    prosper::gpu::SharedVulkanContext published;
    published.storage_image_read_without_format = true;
    published.storage_image_write_without_format = true;
    published.image_robustness = false;
    CHECK(!storage_image_device_features(published.image_robustness,
                                         published.storage_image_read_without_format,
                                         published.storage_image_write_without_format)
               .storage_image_capable(),
          "adopting a renderer device that did NOT enable robustness refuses storage images");
    published.image_robustness = true;
    CHECK(storage_image_device_features(published.image_robustness,
                                        published.storage_image_read_without_format,
                                        published.storage_image_write_without_format)
               .storage_image_capable(),
          "adopting a renderer device that DID enable robustness accepts storage images");

    // --- 2. The shipped helper against a real physical device ----------------------------------
    // Queries only; no device is created and no GPU work is submitted, so this runs anywhere a
    // Vulkan ICD is installed. When there is none the arm reports that it did not run rather than
    // passing silently -- an arm that cannot distinguish "verified" from "never executed" is the
    // instrument trap this project keeps hitting.
    if (!prosper::frontend::require_vulkan_runtime_loader("test")) {
        std::printf("  [skip] no Vulkan 1.4 loader: the real-device chain arm did not run\n");
    } else {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = prosper::frontend::kVulkanRuntimeVersion;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS || !instance) {
            std::printf("  [skip] vkCreateInstance failed: the real-device chain arm did not run\n");
        } else {
            uint32_t count = 0;
            vkEnumeratePhysicalDevices(instance, &count, nullptr);
            std::vector<VkPhysicalDevice> devices(count);
            if (count) vkEnumeratePhysicalDevices(instance, &count, devices.data());
            if (!count) {
                std::printf("  [skip] no physical device: the real-device chain arm did not run\n");
            } else {
                const VkPhysicalDevice physical = devices[0];

                // Ask the driver INDEPENDENTLY of the helper, so the expectation is not derived from
                // the thing being tested.
                VkPhysicalDeviceImageRobustnessFeatures advertised{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES};
                VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                query.pNext = &advertised;
                vkGetPhysicalDeviceFeatures2(physical, &query);
                const bool supported = advertised.robustImageAccess == VK_TRUE;

                // A struct already on the chain. The helper must extend the chain, never replace it:
                // clobbering it here would silently drop descriptor indexing or int64 atomics at the
                // one call site that matters.
                VkPhysicalDeviceDescriptorIndexingFeatures incumbent{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
                VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
                dci.pNext = &incumbent;

                VkPhysicalDeviceImageRobustnessFeatures storage{};
                const auto acquired = prosper::frontend::acquire_storage_image_device_features(
                    "test", physical, true, true, storage, dci);

                CHECK(acquired.robust_image_access == supported,
                      "the helper reports exactly what the physical device advertises");

                // Walk the chain by COPYING each node's header out rather than reading the node
                // through a VkBaseInStructure lvalue. Reading one struct type through a pointer to
                // an unrelated one is a strict-aliasing violation even when the two share an initial
                // sequence, and it is not theoretical here: the first draft of this arm did the
                // obvious reinterpret walk and GCC 15 at -O2 turned the without-fix run into a
                // SIGSEGV instead of the clean failure the arm is supposed to produce -- adding an
                // unrelated fprintf made the fault vanish. memcpy is always defined, and a regression
                // arm that can crash instead of failing is not one you can read.
                bool chained = false, incumbent_survived = false;
                for (const void* node = dci.pNext; node;) {
                    VkBaseInStructure header{};
                    std::memcpy(&header, node, sizeof header);
                    if (header.sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES) {
                        chained = true;
                        VkPhysicalDeviceImageRobustnessFeatures requested{};
                        std::memcpy(&requested, node, sizeof requested);
                        CHECK(requested.robustImageAccess == VK_TRUE,
                              "the chained struct REQUESTS robustImageAccess (not merely present)");
                    }
                    if (header.sType ==
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES)
                        incumbent_survived = true;
                    node = header.pNext;
                }
                CHECK(chained == supported,
                      supported ? "VkPhysicalDeviceImageRobustnessFeatures reaches the "
                                  "VkDeviceCreateInfo pNext chain"
                                : "an unsupporting device is not sent a feature it cannot enable");
                CHECK(incumbent_survived,
                      "chaining preserves the structs already on VkDeviceCreateInfo::pNext");
                CHECK(acquired.storage_image_capable() == supported,
                      "the storage-image verdict follows the feature actually requested");
                std::printf("  real device: robustImageAccess advertised=%d chained=%d\n",
                            (int)supported, (int)chained);
                if (!supported)
                    std::printf("  NOTE: this device cannot run recompiled storage-image kernels "
                                "defined; prosper declines them there (#3531)\n");
            }
            vkDestroyInstance(instance, nullptr);
        }
    }

    if (fails) { std::printf("== FAIL: %d == (%d assertions executed)\n", fails, checks); return 1; }
    std::printf("== PASS == (%d assertions executed)\n", checks);
    return 0;
}
