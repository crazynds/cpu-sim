// Pieces of CUDA's API that real programs use (found by compiling HeCBench):
// FP8 conversions (known encodings), cuComplex, the __nv_atomic_* family,
// mixed-signedness min/max, CUDA's math functions on the host, memory pools
// and the 5-argument cudaGraphInstantiate.
#include <cuComplex.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "check.cuh"

struct Fp8Case {
    float x;
    __nv_fp8_interpretation_t fmt;
    __nv_saturation_t sat;
    unsigned char want;
};
// 1.0, the largest finite, the smallest subnormal, a tie to even, overflow
// with and without saturation, a negative number, NaN.
static const Fp8Case kFp8[] = {
    {1.0f, __NV_E4M3, __NV_NOSAT, 0x38},      {448.0f, __NV_E4M3, __NV_NOSAT, 0x7e},
    {0.001953125f, __NV_E4M3, __NV_NOSAT, 0x01}, {1.0625f, __NV_E4M3, __NV_NOSAT, 0x38},
    {500.0f, __NV_E4M3, __NV_SATFINITE, 0x7e}, {500.0f, __NV_E4M3, __NV_NOSAT, 0x7f},
    {-2.0f, __NV_E4M3, __NV_NOSAT, 0xc0},      {1.0f, __NV_E5M2, __NV_NOSAT, 0x3c},
    {57344.0f, __NV_E5M2, __NV_NOSAT, 0x7b},   {70000.0f, __NV_E5M2, __NV_NOSAT, 0x7c},
    {70000.0f, __NV_E5M2, __NV_SATFINITE, 0x7b}, {-0.5f, __NV_E5M2, __NV_NOSAT, 0xb8},
};
constexpr int kCases = sizeof(kFp8) / sizeof(kFp8[0]);

__global__ void fp8_kernel(const Fp8Case* cases, unsigned char* code, float* back, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    code[i] = __nv_cvt_float_to_fp8(cases[i].x, cases[i].sat, cases[i].fmt);
    back[i] = float(cases[i].fmt == __NV_E4M3 ? float(__nv_fp8_e4m3(cases[i].x)) : float(__nv_fp8_e5m2(cases[i].x)));
}

__global__ void complex_kernel(const cuFloatComplex* a, const cuFloatComplex* b, cuFloatComplex* q, float* m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q[i] = cuCdivf(cuCmulf(a[i], cuConjf(b[i])), b[i]);
    m[i] = cuCabsf(a[i]);
}

__global__ void atomics_kernel(unsigned long long* sum, int* lo, unsigned* hi, const int* v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_atomic_fetch_add(sum, (unsigned long long)v[i], __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    __nv_atomic_fetch_min(lo, v[i], __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    int expected = __nv_atomic_load_n(lo, __NV_ATOMIC_ACQUIRE);
    (void)expected;
    hi[i] = max(v[i], 100u) + min(unsigned(i), -1);  // mixed signedness: compared unsigned
}

__global__ void noop() {}

int main() {
    Checker c("api_compat");

    Fp8Case* dcases;
    unsigned char* dcode;
    float* dback;
    CHECK_CUDA(cudaMalloc(&dcases, sizeof(kFp8)));
    CHECK_CUDA(cudaMalloc(&dcode, kCases));
    CHECK_CUDA(cudaMalloc(&dback, kCases * 4));
    CHECK_CUDA(cudaMemcpy(dcases, kFp8, sizeof(kFp8), cudaMemcpyHostToDevice));
    fp8_kernel<<<1, 32>>>(dcases, dcode, dback, kCases);
    unsigned char code[kCases];
    float back[kCases];
    CHECK_CUDA(cudaMemcpy(code, dcode, kCases, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(back, dback, kCases * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < kCases; ++i) {
        c.expect(code[i] == kFp8[i].want, "fp8 encoding (device)", i, code[i], kFp8[i].want);
        const unsigned char h = __nv_cvt_float_to_fp8(kFp8[i].x, kFp8[i].sat, kFp8[i].fmt);
        c.expect(h == kFp8[i].want, "fp8 encoding (host)", i, h, kFp8[i].want);
        // Back to float through the class (saturating), exact for finite values.
        if (kFp8[i].sat == __NV_SATFINITE || (kFp8[i].want & 0x7f) < 0x7b) {
            const float want = kFp8[i].fmt == __NV_E4M3 ? float(__nv_fp8_e4m3(kFp8[i].x)) : float(__nv_fp8_e5m2(kFp8[i].x));
            c.expect(back[i] == want, "fp8 round trip", i, back[i], want);
        }
    }

    const int n = 200;
    std::vector<cuFloatComplex> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = make_cuFloatComplex(float(i % 17) - 8.0f, float(i % 5) + 0.5f);
        b[i] = make_cuFloatComplex(float(i % 7) + 1.0f, float(i % 3) - 1.0f);
    }
    cuFloatComplex *da, *db, *dq;
    float* dm;
    CHECK_CUDA(cudaMalloc(&da, n * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&db, n * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&dq, n * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&dm, n * 4));
    CHECK_CUDA(cudaMemcpy(da, a.data(), n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db, b.data(), n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    complex_kernel<<<(n + 63) / 64, 64>>>(da, db, dq, dm, n);
    std::vector<cuFloatComplex> q(n);
    std::vector<float> m(n);
    CHECK_CUDA(cudaMemcpy(q.data(), dq, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(m.data(), dm, n * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        // (a * conj(b)) / b, against double precision complex arithmetic.
        const double ar = a[i].x, ai = a[i].y, br = b[i].x, bi = b[i].y;
        const double pr = ar * br + ai * bi, pi = ai * br - ar * bi, d = br * br + bi * bi;
        const double wr = (pr * br + pi * bi) / d, wi = (pi * br - pr * bi) / d;
        c.expect(close_rel(q[i].x, wr, 1e-5, 1e-5) && close_rel(q[i].y, wi, 1e-5, 1e-5), "cuCdivf(cuCmulf)", i, q[i].x, wr);
        c.expect(close_rel(m[i], std::hypot(ar, ai), 1e-6), "cuCabsf", i, m[i], std::hypot(ar, ai));
    }

    std::vector<int> v(n);
    unsigned long long want_sum = 0;
    int want_lo = 1 << 30;
    for (int i = 0; i < n; ++i) v[i] = (i * 37) % 251 - 120, want_sum += (unsigned long long)v[i], want_lo = std::min(want_lo, v[i]);
    int *dv, *dlo;
    unsigned long long* dsum;
    unsigned* dhi;
    CHECK_CUDA(cudaMalloc(&dv, n * 4));
    CHECK_CUDA(cudaMalloc(&dlo, 4));
    CHECK_CUDA(cudaMalloc(&dsum, 8));
    CHECK_CUDA(cudaMalloc(&dhi, n * 4));
    CHECK_CUDA(cudaMemcpy(dv, v.data(), n * 4, cudaMemcpyHostToDevice));
    const int big = 1 << 30;
    CHECK_CUDA(cudaMemcpy(dlo, &big, 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dsum, 0, 8));
    atomics_kernel<<<(n + 63) / 64, 64>>>(dsum, dlo, dhi, dv, n);
    unsigned long long sum;
    int lo;
    std::vector<unsigned> hi(n);
    CHECK_CUDA(cudaMemcpy(&sum, dsum, 8, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&lo, dlo, 4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hi.data(), dhi, n * 4, cudaMemcpyDeviceToHost));
    c.expect(sum == want_sum, "__nv_atomic_fetch_add", -1, double(sum), double(want_sum));
    c.expect(lo == want_lo, "__nv_atomic_fetch_min", -1, lo, want_lo);
    for (int i = 0; i < n; ++i) {
        const unsigned w = std::max(unsigned(v[i]), 100u) + std::min(unsigned(i), unsigned(-1));
        c.expect(hi[i] == w, "min/max of mixed signedness", i, hi[i], w);
    }

    // CUDA's own math functions on the host.
    c.expect(rsqrtf(4.0f) == 0.5f, "rsqrtf on the host", -1, rsqrtf(4.0f), 0.5);
    c.expect(std::fabs(normcdf(0.0) - 0.5) < 1e-15, "normcdf on the host", -1, normcdf(0.0), 0.5);

    // Memory pools and CUDA 11's cudaGraphInstantiate.
    cudaMemPool_t pool;
    CHECK_CUDA(cudaDeviceGetDefaultMemPool(&pool, 0));
    uint64_t threshold = UINT64_MAX, got = 0;
    CHECK_CUDA(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    CHECK_CUDA(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &got));
    c.expect(got == threshold, "memory pool attribute", -1, double(got), double(threshold));
    void* p = nullptr;
    CHECK_CUDA(cudaMallocFromPoolAsync(&p, 1 << 20, pool, 0));
    CHECK_CUDA(cudaFreeAsync(p, 0));
    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));
    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    noop<<<1, 1, 0, s>>>();
    CHECK_CUDA(cudaStreamEndCapture(s, &graph));
    cudaGraphExec_t exec;
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
    CHECK_CUDA(cudaGraphLaunch(exec, s));
    CHECK_CUDA(cudaStreamSynchronize(s));
    int can = -1;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can, 0, 0));
    c.expect(can == 0, "no peer", -1, can, 0);
    return c.finish();
}
