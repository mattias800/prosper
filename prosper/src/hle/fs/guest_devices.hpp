// guest_devices.hpp — the device nodes the guest namespace serves outside the mounts.
//
// #4782 stopped guest paths outside the mounts from reaching the host. One family of such paths is
// real on the console and read by titles: the random devices. Every Unity/IL2CPP title's crypto
// provider opens "/dev/urandom" (O_RDONLY), falls back to "/dev/random" (O_NONBLOCK), and reads 16
// bytes (The Messenger's eboot, function +0xab8b50; observed in PROSPER_FILELOG runs of five
// titles). On failure it seeds from the clock instead, so serving nothing degrades silently.
//
// The device is served from prosper's own host CSPRNG (host/platform/host_entropy.hpp) on every
// host. The guest's spelling never reaches the host: a device descriptor is a host descriptor on
// the host's null device, opened read-only, whose reads prosper answers with entropy.
//
// CONFIDENCE: MED that the console serves both names to a title (the title's runtime opens them as
// its first choice, but it also carries a fallback; no console measurement). LOW for everything
// beyond a successful read: write access and namespace changes are refused as on a read-only
// device, and stat reports a character device with mode 0444.
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace prosper {

enum class GuestDevice : uint8_t {
    Random,   // /dev/random and /dev/urandom: one device, two names (as on FreeBSD)
};

// The device a ROOTED, lexically normalized guest path names, if any.
std::optional<GuestDevice> guest_device_named(std::string_view rooted_path);

// The errno an open of `device` with these FreeBSD open(2) flags fails with, or 0 if it may open.
int guest_device_open_errno(GuestDevice device, uint64_t guest_flags);

// Open a descriptor for `device` at fd >= 3 and mark it. Returns -1 with errno on failure.
int guest_device_open_fd(GuestDevice device);

// Descriptor bookkeeping. guest_device_is_fd() is lock-free and cheap enough for every read.
// Every path that frees or hands out a descriptor number keeps the mark in step with it.
bool guest_device_is_fd(int fd);
void guest_device_forget_fd(int fd);
void guest_device_copy_fd(int from, int to);

// The bytes a read of a random device returns: `count` bytes from the host CSPRNG. Returns false
// if the host has no entropy to give, which the caller reports as EIO.
bool guest_device_fill(void* buffer, std::size_t count);

}   // namespace prosper
