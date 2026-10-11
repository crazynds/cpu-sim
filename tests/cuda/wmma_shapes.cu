// Tensor cores (nvcuda::wmma) in the shapes m32n8k16 and m8n32k16: half ->
// float and half accumulators, bfloat16 -> float, int8 and uint8 -> int. The
// GEMM is in wmma_gemm.cuh.
#include "wmma_gemm.cuh"

int main() {
    Checker check("wmma_shapes");
    cudaDeviceProp p;
    CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    const int cc = p.major * 10 + p.minor;
    std::printf("compute capability %d.%d\n", p.major, p.minor);
    run<__half, __half, float, 32, 8, 16, true>(check, "half->float m32n8k16", 70, cc, -8, 8);
    run<__half, __half, __half, 32, 8, 16, false>(check, "half->half m32n8k16", 70, cc, -1, 1);
    run<__nv_bfloat16, __nv_bfloat16, float, 32, 8, 16, false>(check, "bf16->float m32n8k16", 80, cc, -8, 8);
    run<signed char, signed char, int, 32, 8, 16, false>(check, "int8->int m32n8k16", 72, cc, -128, 127);
    run<unsigned char, unsigned char, int, 32, 8, 16, false>(check, "uint8->int m32n8k16", 72, cc, 0, 255);
    run<__half, __half, float, 8, 32, 16, true>(check, "half->float m8n32k16", 70, cc, -8, 8);
    run<__half, __half, __half, 8, 32, 16, false>(check, "half->half m8n32k16", 70, cc, -1, 1);
    run<__nv_bfloat16, __nv_bfloat16, float, 8, 32, 16, false>(check, "bf16->float m8n32k16", 80, cc, -8, 8);
    run<signed char, signed char, int, 8, 32, 16, false>(check, "int8->int m8n32k16", 72, cc, -128, 127);
    run<unsigned char, unsigned char, int, 8, 32, 16, false>(check, "uint8->int m8n32k16", 72, cc, 0, 255);
    return check.finish();
}
