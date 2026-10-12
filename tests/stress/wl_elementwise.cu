// Elementwise workloads: vector add, saxpy (one thread per element and a
// grid-stride loop), a fused chain of 20 elementwise operations including
// transcendental functions, int8 -> float dequantization (per-tensor and
// per-channel scale + zero point, char4/float4 vectorized), and a float4
// vectorized copy. Every case is memory or ALU bound with no inter-thread
// communication.
#include "stress.cuh"

// ---------------------------------------------------------------------------
// Kernels

__global__ void vecadd(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void saxpy(int n, float a, const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = a * x[i] + y[i];
}

__global__ void saxpy_grid_stride(int n, float a, const float* __restrict__ x, const float* __restrict__ y,
                                  float* __restrict__ z) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) z[i] = a * x[i] + y[i];
}

// The same 20-step chain on the host (reference) and on the device.
__host__ __device__ inline float chain20(float x, float y) {
    float a = x * 1.5f;                // 1
    a = a + y;                         // 2
    a = fmaf(a, x, 0.5f);              // 3
    a = sqrtf(fabsf(a) + 1.0f);        // 4, 5, 6
    a = expf(-0.5f * a);               // 7, 8
    a = sinf(3.0f * a + x);            // 9, 10, 11
    a = fminf(a, 0.75f);               // 12
    a = fmaxf(a, -0.75f);              // 13
    a = fmaf(a, a, y);                 // 14
    a = logf(a * a + 1.0f);            // 15, 16, 17
    a = cosf(a - y);                   // 18, 19
    return a * 0.5f + x * y;           // 20 (fma)
}

__global__ void fused_chain(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = chain20(x[i], y[i]);
}

constexpr int kChannels = 256;
__constant__ float c_scale[kChannels];
__constant__ int c_zero_point[kChannels];

// Per-tensor: one element per thread.
__global__ void dequant_per_tensor(const int8_t* __restrict__ q, float* __restrict__ out, int n, float scale, int zp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = float(int(q[i]) - zp) * scale;
}

// Per-channel, layout [C][HW] with HW % 4 == 0: four elements per thread
// (char4 load, float4 store), scale and zero point from constant memory.
__global__ void dequant_per_channel(const char4* __restrict__ q, float4* __restrict__ out, int n4, int hw4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    int c = i / hw4;
    float s = c_scale[c];
    int zp = c_zero_point[c];
    char4 v = q[i];
    out[i] = make_float4(float(v.x - zp) * s, float(v.y - zp) * s, float(v.z - zp) * s, float(v.w - zp) * s);
}

__global__ void copy_float4(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;
    if (i < n4) reinterpret_cast<float4*>(out)[i] = reinterpret_cast<const float4*>(in)[i];
    // Tail (n not a multiple of 4).
    if (i < n - n4 * 4) out[n4 * 4 + i] = in[n4 * 4 + i];
}

// ---------------------------------------------------------------------------

int main() {
    using namespace stress;
    const int n = 1 << 24;
    const int B = 256;
    auto x = random_floats(n, 1), y = random_floats(n, 2);
    Buffer<float> dx(x), dy(y), dz(n);
    std::vector<float> want(n);

    {  // vecadd
        double ref = best_ms([&] {
#pragma omp parallel for simd
            for (int i = 0; i < n; ++i) want[i] = x[i] + y[i];
        });
        double k = time_device([&] { vecadd<<<(n + B - 1) / B, B>>>(dx, dy, dz, n); });
        auto got = dz.download();
        report("vecadd", equal(got.data(), want.data(), n), k, ref);
    }

    const float alpha = 2.5f;
    {  // saxpy
        double ref = best_ms([&] {
#pragma omp parallel for simd
            for (int i = 0; i < n; ++i) want[i] = alpha * x[i] + y[i];
        });
        dz.zero();
        double k = time_device([&] { saxpy<<<(n + B - 1) / B, B>>>(n, alpha, dx, dy, dz); });
        auto got = dz.download();
        report("saxpy", close(got.data(), want.data(), n, 1e-6, 1e-6), k, ref);
    }

    {  // saxpy, grid-stride loop over a grid sized to the device
        int dev = 0, sms = 0;
        STRESS_CHECK(cudaGetDevice(&dev));
        STRESS_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        int grid = sms * 8;
        const float a2 = -0.75f;
        double ref = best_ms([&] {
#pragma omp parallel for simd
            for (int i = 0; i < n; ++i) want[i] = a2 * x[i] + y[i];
        });
        dz.zero();
        double k = time_device([&] { saxpy_grid_stride<<<grid, B>>>(n, a2, dx, dy, dz); });
        auto got = dz.download();
        report("saxpy_grid_stride", close(got.data(), want.data(), n, 1e-6, 1e-6), k, ref);
    }

    {  // fused chain of 20 ops
        const int m = n;
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < m; ++i) want[i] = chain20(x[i], y[i]);
        });
        dz.zero();
        double k = time_device([&] { fused_chain<<<(m + B - 1) / B, B>>>(dx, dy, dz, m); });
        auto got = dz.download();
        report("fused_chain20", close(got.data(), want.data(), m, 1e-4, 1e-5), k, ref);
    }

    // int8 tensor [C][HW]
    std::vector<int8_t> q(n);
    {
        auto r = random_u32(n, 3);
        for (int i = 0; i < n; ++i) q[i] = int8_t(r[i] & 0xff);
    }
    Buffer<int8_t> dq(q);

    {  // per-tensor dequantization
        const float scale = 0.0371f;
        const int zp = -7;
        double ref = best_ms([&] {
#pragma omp parallel for simd
            for (int i = 0; i < n; ++i) want[i] = float(int(q[i]) - zp) * scale;
        });
        dz.zero();
        double k = time_device([&] { dequant_per_tensor<<<(n + B - 1) / B, B>>>(dq, dz, n, scale, zp); });
        auto got = dz.download();
        report("dequant_int8_per_tensor", equal(got.data(), want.data(), n), k, ref);
    }

    {  // per-channel dequantization
        const int hw = n / kChannels;
        float scale[kChannels];
        int zp[kChannels];
        for (int c = 0; c < kChannels; ++c) {
            scale[c] = 0.001f + 0.0005f * float(hash32(c + 11) % 200);
            zp[c] = int(hash32(c + 77) % 31) - 15;
        }
        STRESS_CHECK(cudaMemcpyToSymbol(c_scale, scale, sizeof(scale)));
        STRESS_CHECK(cudaMemcpyToSymbol(c_zero_point, zp, sizeof(zp)));
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int c = 0; c < kChannels; ++c) {
                const float s = scale[c];
                const int z = zp[c];
                const int8_t* src = q.data() + size_t(c) * hw;
                float* dst = want.data() + size_t(c) * hw;
#pragma omp simd
                for (int j = 0; j < hw; ++j) dst[j] = float(int(src[j]) - z) * s;
            }
        });
        dz.zero();
        const int n4 = n / 4;
        double k = time_device([&] {
            dequant_per_channel<<<(n4 + B - 1) / B, B>>>(reinterpret_cast<const char4*>(dq.p),
                                                         reinterpret_cast<float4*>(dz.p), n4, hw / 4);
        });
        auto got = dz.download();
        report("dequant_int8_per_channel", equal(got.data(), want.data(), n), k, ref);
    }

    {  // float4 copy (odd length exercises the tail)
        const int m = n - 3;
        double ref = best_ms([&] {
#pragma omp parallel for simd
            for (int i = 0; i < m; ++i) want[i] = x[i];
        });
        dz.zero();
        double k = time_device([&] { copy_float4<<<(m / 4 + B - 1) / B, B>>>(dx, dz, m); });
        auto got = dz.download();
        report("copy_float4", equal(got.data(), want.data(), m), k, ref);
    }

    return finish();
}
