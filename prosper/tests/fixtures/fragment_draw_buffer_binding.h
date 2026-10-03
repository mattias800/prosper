// Private same-backend storage view, included inside prosper::test. This is neither a guest
// resource nor a serializable/replay authority. Only the retained draw transaction constructs it.
#pragma once

class FragmentDrawGpuOwner;
class FragmentDrawGpuBuffer {
public:
    VkDevice device() const { return device_; }
    VkBuffer buffer() const { return buffer_; }
    VkDeviceSize bytes() const { return bytes_; }
    bool valid_for(VkDevice device, VkDeviceSize maximum) const {
        return owner_ && device && device == device_ && buffer_ && bytes_ && bytes_ <= maximum;
    }

private:
    friend class FragmentDrawGpuOwner;
    FragmentDrawGpuBuffer(VkDevice device, VkBuffer buffer, VkDeviceSize bytes,
                          std::shared_ptr<const FragmentDrawGpuOwner> owner)
        : device_(device), buffer_(buffer), bytes_(bytes), owner_(std::move(owner)) {}
    const VkDevice device_;
    const VkBuffer buffer_;
    const VkDeviceSize bytes_;
    const std::shared_ptr<const FragmentDrawGpuOwner> owner_;
};
