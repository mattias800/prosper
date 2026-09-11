// gpu_memory_budget.hpp — how much device memory has prosper allocated, against how much exists?
//
// prosper answered that question nowhere, and the cost of not answering it is measured rather than
// hypothetical. On 2026-09-11 this project's development machine hard-froze FIVE times under
// ordinary prosper GPU work, every time with the kernel reporting
//
//     amdgpu: [drm] *ERROR* Not enough memory for command submission!
//     amdgpu: [drm] *ERROR* amdgpu_vm_validate() failed.
//
// and THREE separate root causes were published and retracted before anyone asked the simplest
// question in the list. Each retraction came from reading a kernel trace as confirmation of a story
// rather than as a new sample (#3533). A log line saying "prosper holds N MiB of an M MiB heap"
// would have settled it on the first freeze; there was no such line, so five freezes produced three
// wrong answers instead of one right one.
//
// WHAT THIS MEASURES, AND WHAT IT DOES NOT. It counts the bytes prosper itself passes to
// vkAllocateMemory, per heap, and compares them against that heap's `VkMemoryHeap::size`. It does
// NOT see any other process, the compositor, or the driver's own overhead — on an integrated GPU the
// desktop shares these heaps, so prosper holding well under the heap size is not on its own proof
// that the heap has room. Reading this as a whole-system budget is the obvious misuse and it is why
// the printed line names prosper explicitly.
//
// The honest upgrade is `VK_EXT_memory_budget`, whose `heapUsage` is process-wide and whose
// `heapBudget` is what the driver will actually grant. That extension is not enabled on the device
// today; enabling it is a separate change and this header is deliberately independent of it, so the
// instrument exists now rather than after that plumbing lands.
//
// THE GROWTH LINE REPORTS A HIGH-WATER MARK, NOT CURRENT GROWTH. Each heap's step threshold only
// ever ratchets up, so a workload that climbs to 3 GiB, frees it, and climbs to 3 GiB again prints
// once, not twice. That bounds the output of a cycling workload, and it is the right default for
// the question this exists to answer -- but do not read the absence of a line as "prosper stopped
// allocating". `report_device_memory` prints the standing on demand and is not subject to it.
//
// ON BY DEFAULT, which is the point. `PROSPER_GPU_MEM_LOG=0` silences it and
// `PROSPER_GPU_MEM_LOG_MIB=<n>` changes the growth step. A diagnostic that must be switched on is
// one nobody had switched on for the run that mattered — which is exactly what happened here.
#pragma once
#include <cstdint>

namespace prosper::gpu {

// Record the device's heap layout, ONCE, before anything is allocated. `type_heap_index[i]` is
// `memoryTypes[i].heapIndex` and `heap_sizes[j]` is `memoryHeaps[j].size`, copied out of
// VkPhysicalDeviceMemoryProperties by the caller so this header needs no Vulkan types and can be
// included anywhere. Storing it here is what lets every allocation site pass only a memory type
// index -- no call site needs the properties struct in scope, which is what makes wiring the
// renderer's thirty-odd sites a mechanical change rather than a judgement call at each one.
//
// The FIRST layout wins. A second, different layout means two physical devices are in play and the
// per-heap totals would be mixing them, so it warns once and keeps the first rather than silently
// renumbering the heaps under a running count.
void set_device_heaps(const uint32_t* type_heap_index, uint32_t type_count,
                      const uint64_t* heap_sizes, uint32_t heap_count);

// Record a real vkAllocateMemory / vkFreeMemory. Keyed by the VkDeviceMemory handle (as a bare
// uint64_t), because the free sites hold only the handle -- they do not know the allocation's size
// or its heap, and making each one carry them would be dozens of places to get wrong instead of one.
//
// POOLED REUSE MUST NOT BE COUNTED. prosper recycles device allocations through its own pools, so
// only call these where the DRIVER is really allocating or really freeing. Counting a pool hit would
// measure prosper's churn rather than its footprint, which is the opposite of the question.
void note_device_alloc(uint64_t handle, uint32_t memory_type, uint64_t bytes);
void note_device_free(uint64_t handle);

// Bytes prosper currently holds on `heap`, for a caller that wants the number rather than the line.
uint64_t device_bytes_held(uint32_t heap);

// The high-water mark on `heap`. This is the number that explains a submission failure AFTER the
// fact: a run that demanded 3 GiB and gave it back holds nothing by the time anyone looks, and the
// held figure alone would say the heap was never under pressure.
uint64_t device_peak_bytes(uint32_t heap);

// Print the current per-heap standing immediately, whatever the step counter says. `why` names the
// occasion ("startup", "allocation failed", ...) so a line in a log can be attributed.
void report_device_memory(const char* why);

// A device allocation just failed: say what was asked for and print the standing. BOUNDED to the
// first few, because a device under real pressure fails in bursts and an unbounded report would push
// the first -- and most informative -- one out of a scrolling log.
void report_allocation_failure(int result, uint64_t bytes, uint32_t memory_type);

// Print the standing if it has been more than PROSPER_GPU_MEM_REPORT_S seconds (default 30) since
// the last report. Cheap enough to call on every allocation and free.
//
// This is what puts a RECENT and UNQUANTISED number in the log of a run that never exits. The growth
// line alone cannot do either. It ratchets, so a process that reached its high-water mark early and
// then held steady goes quiet -- and a host that locks up an hour later leaves a log whose last
// memory figure is an hour stale. It also only ever fires at a step boundary, so everything below
// one step is invisible: on The Messenger at a 32 MiB step the growth line reported 64 MiB on one
// heap and NOTHING on the other, where the true standing was 81 MiB and 16 MiB. A figure read off
// the growth line is a floor rounded down to a step, and it was published as a footprint once.
//
// It is driven by ACTIVITY rather than by a clock, and the activity has to include submits, not just
// allocations. An earlier version of this comment argued allocations were enough because "a device
// running out of memory is allocating by definition". That is wrong, and wrong in the direction that
// matters: #3533's signature is `amdgpu_vm_validate() failed` alongside "Not enough memory for
// command submission", which is a SUBMIT-time validation over the resident BO set. A process holding
// a perfectly steady footprint can hit it having allocated nothing at all -- so the allocation-time
// argument covers the `vkAllocateMemory` failure that report_allocation_failure already handles, and
// misses the one this cadence exists for.
//
// So it is called from the renderer's submit funnel as well as from the two allocation wrappers.
// A process that neither allocates nor submits still leaves no fresh line; that is a real limit, and
// deliberately not a background thread, whose shutdown ordering would put stderr and a mutex in the
// path of static destruction for a diagnostic.
void report_device_memory_periodically();

}  // namespace prosper::gpu
