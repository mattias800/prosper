#pragma once
#ifdef _WIN32
// Must precede EVERY header — including <thread>/<mutex>/<atomic>, which pull in MinGW's _mingw.h and
// default _WIN32_WINNT to 0x0601 (Windows 7). A later `#ifndef _WIN32_WINNT` is then a silent no-op, and
// GetCurrentThreadStackLimits / WaitOnAddress / WakeByAddress* (all >= 0x0602, Win8) go undeclared.
// Same placement rule as hle_kernel.cpp and exec_image_win.cpp.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#endif
#include <stdint.h>

namespace prosper {
bool ampr_a2_is_capacity(uint64_t a2);
bool ampr_cb_tracks_offset_arg(uint64_t a2);
uint64_t ampr_cb_capacity_arg(uint64_t a1, uint64_t a2, uint64_t a5);
void ampr_cb_construct(uint64_t cb, uint64_t capacity, bool tracks_offset);
void ampr_cb_reset(uint64_t cb);
void ampr_cb_destroy_320(uint64_t cb);
uint64_t ampr_cb_capacity(uint64_t cb);
uint64_t ampr_cb_offset(uint64_t cb);
void ampr_cb_set_buffer(uint64_t cb, uint64_t base, uint64_t size);
uint64_t ampr_cb_buffer_base(uint64_t cb);
void prosper_ampr_advance(uint64_t cb, uint64_t bytes);
void ampr_cb_erase(uint64_t a0);
}   // namespace prosper
