#pragma once

#include <cstdint>
#include <string>

#include "device/executor.h"

namespace cpusim {

// Debug execution of one launch (CPUSIM_DEBUG): everything runs on the calling
// thread, one step at a time (a step is one warp running from its start or a
// barrier to the next barrier or its end). Shared memory and every warp frame
// sit between PROT_NONE guard pages, so an access out of their bounds faults
// on the offending instruction; the fault is reported (step, address, region,
// code offset in the kernel module) and the launch fails instead of the
// process. With `trace`, every step is logged.
//
// Returns false on a fault; *report describes it.
bool run_debug(cpusim_entry_fn entry, const cpusim_kernel_meta& meta, const LaunchConfig& cfg, const uint8_t* params,
               uint64_t* counters, bool trace, std::string* report);

// "symbol+offset (in module+offset)" for a code address.
std::string where_in_code(uintptr_t ip);

}  // namespace cpusim
