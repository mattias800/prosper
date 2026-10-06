// Private same-backend storage view, included inside prosper::test. This is neither a guest
// resource nor a serializable/replay authority. Only backend-internal producers construct it: the
// retained draw transaction, and the merged-NGG prelude (ngg_subgroup_gpu.h), whose views name an
// aligned slice of a scratch-ring chunk and therefore carry an offset.
#pragma once

class FragmentDrawGpuOwner;
class NggSubgroupBackendBatch;
class FragmentDrawGpuBuffer {
public:
    VkDevice device() const { return device_; }
    VkBuffer buffer() const { return buffer_; }
    VkDeviceSize offset() const { return offset_; }
    VkDeviceSize bytes() const { return bytes_; }
    bool valid_for(VkDevice device, VkDeviceSize maximum) const {
        return owner_ && device && device == device_ && buffer_ && bytes_ && bytes_ <= maximum;
    }

private:
    friend class FragmentDrawGpuOwner;
    friend class NggSubgroupBackendBatch;
    FragmentDrawGpuBuffer(VkDevice device, VkBuffer buffer, VkDeviceSize bytes,
                          std::shared_ptr<const void> owner, VkDeviceSize offset = 0)
        : device_(device), buffer_(buffer), offset_(offset), bytes_(bytes),
          owner_(std::move(owner)) {}
    static std::shared_ptr<const FragmentDrawGpuBuffer> view(VkDevice device, VkBuffer buffer,
                                                             VkDeviceSize offset,
                                                             VkDeviceSize bytes,
                                                             std::shared_ptr<const void> owner) {
        return std::shared_ptr<const FragmentDrawGpuBuffer>(
            new FragmentDrawGpuBuffer(device, buffer, bytes, std::move(owner), offset));
    }
    const VkDevice device_;
    const VkBuffer buffer_;
    const VkDeviceSize offset_;
    const VkDeviceSize bytes_;
    // Keeps the backing alive while a draw holds the view; never dereferenced.
    const std::shared_ptr<const void> owner_;
};
