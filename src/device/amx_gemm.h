#pragma once

// Matrix multiplication on AMX tiles, run on the executor's workers (one per
// physical core). The SIMT compiler recognizes CUDA kernels that are a GEMM
// written with wmma (src/compiler, wmma_gemm) and the runtime runs them here
// instead: on a Xeon with AMX, 4096^3 in bf16 at ~43 TFLOPS against ~5 with
// the kernel's vdpbf16ps (docs/optimization/iteration-3/35).

#include <cstddef>
#include <cstdint>

#include "device/executor.h"

namespace cpusim {

enum class GemmType : uint32_t {
    BF16_F32 = 1,  // C (float) [+]= A (bf16) * B (bf16)
    S8_S32 = 2,    // C (int32) [+]= A (int8) * B (int8)
};

struct GemmArgs {
    GemmType type = GemmType::BF16_F32;
    const void* A = nullptr;  // M x K, row-major, lda elements per row
    const void* B = nullptr;  // K x N, row-major
    void* C = nullptr;        // M x N, row-major
    uint64_t lda = 0, ldb = 0, ldc = 0;
    uint32_t M = 0, N = 0, K = 0;
    bool accumulate = false;  // C += A * B (else C = A * B)
};

// The CPU has AMX for this type and the OS gave the process its state.
bool amx_gemm_available(GemmType type);

// The shapes this implementation takes (multiples of the tile sizes).
bool amx_gemm_supported(const GemmArgs& args);

// Runs it; returns the fault that stopped it (an invalid pointer), if any.
KernelFault amx_gemm(Executor& executor, const GemmArgs& args);

}  // namespace cpusim
