// state_callbacks.hpp — guest state callbacks of libSceNetCtl and libSceNpManager.
//
// Both libraries let a title register a function that the library calls with the console's
// current state, from inside the title's own per-frame pump (`sceNetCtlCheckCallback`,
// `sceNpCheckCallback`), on the calling thread. prosper's console is offline and signed out, so the
// one state each library has to report is a constant — what is modelled here is the delivery
// contract, not a changing state: every registration is owed exactly one delivery of the state the
// library's other entry points report, and later pumps deliver nothing because nothing changed.
#pragma once

#include <array>
#include <cstdint>
#include <mutex>

namespace prosper::np {

// A fixed-capacity table of registered guest callbacks, each owed one delivery. Pure bookkeeping:
// it never calls guest code itself, so it is testable without a guest.
//
// Both shipped modules keep 8 slots of {fn, arg, ...} and hand out the lowest free one:
// libSceNetCtl.sprx's sceNetCtlRegisterCallback (RVA 0x2010) scans 0x11048 + 0x18*i for i < 8, and
// libSceNpManager.sprx's sceNpRegisterStateCallbackA (RVA 0x15580) scans 0x565a0 + 0x18*i for i < 8.
// They differ on a repeated function: NetCtl gives it another slot, NpManager refuses it (see add()).
// CONFIDENCE: HIGH (read from the modules).
class StateCallbackTable {
public:
    static constexpr int kCapacity = 8;

    // add()'s refusals. A slot index is never negative.
    static constexpr int kFull = -1;
    static constexpr int kDuplicate = -2;

    // Registers `fn` with its user argument in the lowest free slot and returns the slot index, or
    // kFull when every slot is taken. With `refuse_duplicate`, a `fn` already present in ANY slot is
    // kDuplicate instead -- and that is decided before fullness, as NpManager's register compares
    // all 8 slots' functions before its free-slot answer is used (a full table holding `fn` answers
    // ALREADY_REGISTERED, not CALLBACK_MAX). Without it the same function may hold several slots,
    // which is NetCtl's behaviour.
    int add(uint64_t fn, uint64_t arg, bool refuse_duplicate = false);

    // Releases a slot. False when the slot is out of range or not registered.
    bool remove(int slot);

    // Number of registered slots.
    int size() const;

    // Calls `deliver(fn, arg)` for every registered slot that has not yet been delivered its
    // state, marking it delivered BEFORE the call. `deliver` runs without the table's lock held, so
    // a callback may itself register another callback. Returns the number of deliveries made.
    template <class Deliver>
    int pump(Deliver&& deliver) {
        int delivered = 0;
        for (int i = 0; i < kCapacity; ++i) {
            uint64_t fn = 0, arg = 0;
            {
                std::lock_guard<std::mutex> lk(mx_);
                if (!slots_[i].used || slots_[i].delivered) continue;
                slots_[i].delivered = true;
                fn = slots_[i].fn;
                arg = slots_[i].arg;
            }
            deliver(fn, arg);
            ++delivered;
        }
        return delivered;
    }

    // Drops every registration (process teardown and tests).
    void clear();

private:
    struct Slot {
        bool used = false;
        bool delivered = false;
        uint64_t fn = 0;
        uint64_t arg = 0;
    };
    mutable std::mutex mx_;
    std::array<Slot, kCapacity> slots_{};
};

// Registers the NetCtl and NpManager state-callback exports; called by register_np_hle().
void register_state_callbacks_hle();

// Test hook: drops every registration in both libraries' tables.
void reset_state_callbacks_for_test();

}   // namespace prosper::np
