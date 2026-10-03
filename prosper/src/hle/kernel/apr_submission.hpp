#pragma once
// APR eager execution and submit handles. Event binding tags are a separate namespace.
#include <cstdint>

namespace prosper {
void apr_execution_reset(uint64_t cb);
uint32_t apr_execution_status(uint64_t cb);
uint64_t apr_publish_completed_submit(uint64_t result, uint64_t id_slot, uint32_t* id);
uint64_t apr_wait_completed_submit(uint32_t id);

// Covers early refusals as well as the actual read/copy. Submit cannot turn an in-flight or
// failed builder into a successful result. The current read builders execute synchronously.
class AprReadExecution {
public:
    explicit AprReadExecution(uint64_t cb);
    ~AprReadExecution();
    AprReadExecution(const AprReadExecution&) = delete;
    AprReadExecution& operator=(const AprReadExecution&) = delete;
    uint64_t finish(uint64_t status);
private:
    uint64_t cb_;
    uint32_t status_;
    bool tracked_ = false;
};
}
