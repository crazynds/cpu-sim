#pragma once

// Recognizing a launch of a wmma GEMM kernel (docs/optimization/iteration-3/35):
// the module's __cpusim_trace_<kernel> (built by cpusim-devc, wmma_trace.cpp
// in src/compiler) is run for every warp of the launch, its wmma operations
// recorded, and the records checked to be exactly C [+]= A * B; the result
// is the GEMM's arguments, for amx_gemm.

#include <cstdint>

#include "device/amx_gemm.h"
#include "device/executor.h"

namespace cpusim {

using cpusim_trace_fn = void (*)(const uint8_t* params, const uint32_t* sregs, void* rec);

// The GEMM the launch computes, or false (then the kernel runs as written).
// Runs the trace on the executor's workers.
bool wmma_gemm_plan(Executor& ex, cpusim_trace_fn trace, const LaunchConfig& cfg, const uint8_t* params,
                    GemmArgs* out, const char** why);

}  // namespace cpusim
