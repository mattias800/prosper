// Private same-TU companion, included inside prosper::test at the original declaration site.
// Vulkan/device support types are already declared there; no linkage or lifetime change.
#pragma once

// `seed_rgba` (optional): native-format pixels to PRELOAD the color attachment with before the draws
// run (loadOp LOAD instead of the blue clear). This is real render-target memory semantics: a game
// pass that draws into a target it (or an earlier submit) already rendered composites OVER that
// content — without it every pass starts from the diagnostic blue clear, so cross-submit
// accumulation (UE4's UI-onto-backbuffer after a separate composite submit) is lost. Null (the
// default) keeps the blue-clear behavior byte-identical for every existing caller.
// `clear_rgba` (optional): 4 floats (RGBA, Vulkan order) to clear the color attachment to when no
// seed is supplied. Null keeps the legacy diagnostic blue — every test harness caller passes null,
// so their behavior is byte-identical. The live renderer passes the game's decoded fast-clear color
// (or opaque black when none), so real frames no longer start from blue (#309). PROSPER_CLEAR_DEBUG
// forces the blue back on regardless, so unrendered areas can still be spotted during development.
// Persistent Vulkan context. Creating a fresh instance+device PER render_draws_rgba call dominated
// wall-clock — every submit paid full device init — which made a many-draw frame (real gameplay is
// hundreds of draws/submit) impossibly slow and blocked headless scene investigation (#320). Create the
// instance/physical-device/device/queue ONCE (lazy, thread-safe static init) and reuse it across every
// call. Per-call Vulkan resources are created independently and are retained until their direct call
// or explicit ordered submission batch completes. The context intentionally leaks at process exit.
// LIFETIME INVARIANT: this context is intentionally never destroyed (no destructor; the device and
// instance leak at process exit). The compute backend BORROWS this device (#1091) and calls
// vkDestroyPipeline/vkFreeMemory on it at exit. Adding a destructor here that destroys the device
// would therefore create an immediate use-after-free in ~VulkanComputeContext. Do not add one
// without first giving compute an explicit release-before-teardown handshake.
//
// The mechanism on the compute side changed in #1704: that teardown now runs from a std::atexit
// handler registered after vkCreateInstance, not from a function-local static's destructor, so it is
// sequenced before any enabled Vulkan layer's own statics. That also makes the ordering against THIS
// context defined rather than unspecified — borrowing this device requires this context to already
// exist, so compute's handler is always registered later and therefore always runs first.
//
// Which means the specific use-after-free warned about above is now ordered away: a destructor added
// here would be registered earlier and would run after compute has released its objects. Do not read
// that as permission. The reasons not to add one are now different, not gone: guest threads can still
// be dispatching when exit() begins (execute_live_compute_items declines once the handler has run,
// but the window is not closed), and BorrowedComputeImageLease holds a raw VulkanComputeContext*.
// Give compute an explicit release-before-teardown handshake before adding a destructor here.
struct RenderVkCtx {
    VkInstance inst = VK_NULL_HANDLE; VkPhysicalDevice phys = VK_NULL_HANDLE;
    // Non-null only under PROSPER_VK_VALIDATION; without it the layer has no output sink.
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE; VkQueue queue = VK_NULL_HANDLE; uint32_t qfi = UINT32_MAX;
    // Driver compilation data, distinct from the map retaining prosper's VkPipeline handles.
    // Retained with this process-lifetime device. Access uses graphics_driver_cache_mutex().
    VkPipelineCache driver_pipeline_cache = VK_NULL_HANDLE;
    prosper::frontend::PipelineCacheFile driver_cache_file;
    size_t driver_cache_loaded_bytes = 0;
    VkDeviceSize storage_buffer_alignment = 1;
    double timestamp_period_ns = 0.0;
    uint32_t timestamp_valid_bits = 0;
    bool aniso_enabled = false; float max_aniso_limit = 1.0f;
    bool depth_bias_clamp_enabled = false;   // VkPhysicalDeviceFeatures::depthBiasClamp (#1349)
    bool logic_op_enabled = false; bool ok = false;
    bool geometry_shader_enabled = false;
    bool fragment_stores_atomics = false;
    // Enabled robust2 plus <=4-byte range rounding: word-buffer OOB reads deterministically zero.
    bool deterministic_storage_reads = false;
    prosper::gpu::FloatTransportConfig float_transport{};
    // Per-draw "fragment funnel" diagnostic (PROSPER_DRAW_STATS): pipeline-statistics + precise
    // occlusion queries. Enabled at device creation only when advertised; inert otherwise.
    bool pipeline_stats_enabled = false;
    bool occlusion_precise = false;
    // Geometry-probe (PROSPER_GEOM_PROBE): transform feedback for final clip-space positions.
    bool transform_feedback_enabled = false;
    bool subgroup_size_control = false;
    // Optional workgroup-based pre-rasterization path for merged NGG/GS programs. A native
    // subgroup of 64 is not required: guest wave64 can span multiple host subgroups.
    bool mesh_shader_enabled = false;
    // A 3D image's 2D-array attachment view is core on ordinary Vulkan 1.1+ devices, but
    // portability-subset implementations may decline this particular image-view operation.
    bool image_view_2d_on_3d = true;
    // VkPhysicalDeviceFeatures::textureCompressionBC, enabled when advertised. Native BCn sampled
    // uploads additionally require per-format optimal-tiling support; see
    // backend_native_bc_sampled_supported().
    bool texture_compression_bc = false;
    // VK_EXT_memory_budget, enabled when advertised (#3873): lets the texture-cache budget follow the
    // driver's live heapBudget/heapUsage instead of a fixed fraction of the heap size. `unified_memory`
    // is true for any device that is not a discrete GPU, whose device-local heap is system RAM.
    bool memory_budget_enabled = false;
    bool unified_memory = false;
    VkPhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    PFN_vkCmdDrawMeshTasksEXT cmd_draw_mesh_tasks = nullptr;
    // Runtime-selected storage buffers (#2412). Successful contracts are bounded fixed arrays, so the
    // only descriptor-indexing feature they require is non-uniform storage-buffer array indexing.
    //
    // Measured available on both lanes' hardware: RADV STRIX_HALO and RTX 4090 report all of them true.
    // The AMD device additionally reports `…NonUniformIndexingNative = false`, which is a performance
    // note and not a correctness one here: our index is computed in scalar registers so it is
    // wave-uniform, and a driver waterfall over the distinct values present converges in one iteration.
    bool descriptor_indexing = false;
    bool storage_buffer_int64_atomics = false;
    // What this device offers the recompiled storage-image path (#3531): acquired by the same
    // shared helper the compute backend's own device uses, and published to SharedVulkanContext so
    // an adopting consumer inherits the verdict instead of assuming it.
    prosper::frontend::StorageImageDeviceFeatures storage_image_features{};
    bool compute_full_subgroups = false;
    uint32_t min_subgroup_size = 0, max_subgroup_size = 0;
    uint32_t max_compute_workgroup_subgroups = 0;
    uint32_t max_compute_workgroup_size_x = 0;
    uint32_t max_compute_workgroup_invocations = 0;
    VkPhysicalDeviceLimits detile_limits{};
    bool queue_supports_compute = false;
    // Same intentional process lifetime as the device; creation is protected by
    // BackendPersistentResourceGuard, including frontend preflight calls.
    mutable std::array<GpuDetilePipeline*, 4> detile_pipelines{};
    VkShaderStageFlags required_subgroup_size_stages = 0;
    VkShaderStageFlags subgroup_stages = 0;
    VkSubgroupFeatureFlags subgroup_operations = 0;
    // Present unification (#1270): so prosper-app can adopt THIS device for its swapchain and blit the
    // renderer's front-buffer image straight to the screen (no 4K CPU round-trip). All additive and
    // only when advertised, so the headless test/screenshot path is byte-for-byte unchanged: on a
    // display-less target the surface instance-extensions and VK_KHR_swapchain are simply absent, these
    // stay false, and prosper-app falls back to its own separate present device + CPU pixels.
    bool present_surface_capable = false;   // instance enabled VK_KHR_surface (+ a platform surface ext)
    bool present_swapchain_capable = false; // device enabled VK_KHR_swapchain
    VkQueue present_queue = VK_NULL_HANDLE; // dedicated 2nd queue when the family has >=2, else == queue
    bool present_queue_shared = false;      // present_queue aliases the render queue -> submits need a mutex
};
