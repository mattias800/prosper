#pragma once

#include <utility>

namespace prosper::video {

// Minimal owning COM reference for the Media Foundation backends. Deliberately not WRL's ComPtr:
// the MinGW toolchains this project builds with do not all ship <wrl/client.h>, and the backends
// need nothing beyond put/get/reset.
template <typename T>
class ComPtr {
public:
    ComPtr() = default;
    ~ComPtr() { reset(); }
    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;
    ComPtr(ComPtr&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
    ComPtr& operator=(ComPtr&& other) noexcept {
        if (this != &other) { reset(); ptr_ = std::exchange(other.ptr_, nullptr); }
        return *this;
    }

    T* get() const { return ptr_; }
    T** put() { reset(); return &ptr_; }
    T* operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    void reset(T* replacement = nullptr) {
        if (ptr_) ptr_->Release();
        ptr_ = replacement;
    }

private:
    T* ptr_ = nullptr;
};

} // namespace prosper::video
