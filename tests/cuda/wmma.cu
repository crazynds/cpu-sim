// Tensor cores (nvcuda::wmma) in the 16x16x16 shape: half -> float and half
// accumulators, bfloat16 -> float, int8 and uint8 -> int; tf32 -> float
// (16x16x8) and double -> double (8x8x4); __float_to_tf32's rounding, and that
// a tf32 mma ignores the 13 low mantissa bits of its inputs. The GEMM is in
// wmma_gemm.cuh; wmma_shapes.cu covers the shapes m32n8k16 and m8n32k16.
#include "wmma_gemm.cuh"

/* ------------------------------------------------------------------ tf32 */

__global__ void to_tf32(const float* in, float* out, int n) {
    const int i = threadIdx.x;
    if (i < n) out[i] = wmma::__float_to_tf32(in[i]);
}
// One 16x16x8 mma of A = (1 + 2^-20) and B = 1, not rounded first: the
// tensor core ignores the low bits, so every element of D is exactly 8.
__global__ void tf32_ignores_low_bits(float* d) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
    wmma::fill_fragment(fa, 1.0f + 0x1p-20f);
    wmma::fill_fragment(fb, 1.0f);
    wmma::fill_fragment(acc, 0.0f);
    wmma::mma_sync(acc, fa, fb, acc);
    wmma::store_matrix_sync(d, acc, 16, wmma::mem_row_major);
}

static float host_tf32(float x) {  // round to nearest, ties away from zero, 10-bit mantissa
    uint32_t u;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7f800000u) != 0x7f800000u) u = (u + 0x1000u) & 0xffffe000u;
    std::memcpy(&x, &u, 4);
    return x;
}

static void run_tf32_details(Checker& check, int cc) {
    if (cc < 80) return;  // the GEMM above already checks the refusal
    const float in[] = {1.0f, 1.0f + 0x1p-11f,  1.0f + 0x1p-11f + 0x1p-23f, 1.0f + 3 * 0x1p-11f, -(1.0f + 0x1p-11f),
                        1.0f + 0x1p-12f, 3.4028235e38f, 1e-40f, -0.0f, __builtin_inff(), -__builtin_inff(), 1.5f};
    const int n = int(sizeof in / sizeof in[0]);
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, sizeof in));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof in));
    CHECK_CUDA(cudaMemcpy(d_in, in, sizeof in, cudaMemcpyHostToDevice));
    to_tf32<<<1, 32>>>(d_in, d_out, n);
    float out[sizeof in / sizeof in[0]];
    CHECK_CUDA(cudaMemcpy(out, d_out, sizeof in, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        const float want = host_tf32(in[i]);
        check.expect(std::memcmp(&out[i], &want, 4) == 0, "__float_to_tf32", i, out[i], want);
    }
    float* d_d;
    CHECK_CUDA(cudaMalloc(&d_d, 256 * sizeof(float)));
    tf32_ignores_low_bits<<<1, 32>>>(d_d);
    std::vector<float> d(256);
    CHECK_CUDA(cudaMemcpy(d.data(), d_d, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 256; ++i) check.expect(d[i] == 8.0f, "tf32 mma ignores the low 13 bits", i, d[i], 8.0);
    cudaFree(d_in), cudaFree(d_out), cudaFree(d_d);
}

int main() {
    Checker check("wmma");
    cudaDeviceProp p;
    CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    const int cc = p.major * 10 + p.minor;
    std::printf("compute capability %d.%d\n", p.major, p.minor);
    // Ranges that keep every sum exact: |sum| < 2^24 for float, < 2^11 for half.
    run<__half, __half, float, 16, 16, 16, true>(check, "half->float", 70, cc, -8, 8);
    run<__half, __half, __half, 16, 16, 16, false>(check, "half->half", 70, cc, -1, 1);
    run<__nv_bfloat16, __nv_bfloat16, float, 16, 16, 16, false>(check, "bf16->float", 80, cc, -8, 8);
    run<signed char, signed char, int, 16, 16, 16, false>(check, "int8->int", 72, cc, -128, 127);
    run<unsigned char, unsigned char, int, 16, 16, 16, false>(check, "uint8->int", 72, cc, 0, 255);
    run<wmma::precision::tf32, float, float, 16, 16, 8, true>(check, "tf32->float", 80, cc, -8, 8);
    run<double, double, double, 8, 8, 4, true>(check, "double->double", 80, cc, -8, 8);
    run_tf32_details(check, cc);
    return check.finish();
}
