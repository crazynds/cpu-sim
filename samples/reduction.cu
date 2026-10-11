// Parallel sum of a float array, three ways.
//
// Exercises: grid-stride loops, tree reduction in shared memory with a
// shrinking active set (lanes drop out every step), warp shuffles
// (__shfl_down_sync), lane/warp id arithmetic, and float atomicAdd on a single
// global address: every lane hits the same address, the case for warp-level
// atomic aggregation in our compiler.
#include "common.cuh"

constexpr int RED_THREADS = 256;
constexpr int RED_BLOCKS = 256;

__device__ inline float grid_stride_sum(const float* in, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += in[i];
    return sum;
}

__device__ inline float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void reduce_shared(const float* in, float* out, int n) {
    __shared__ float sdata[RED_THREADS];
    sdata[threadIdx.x] = grid_stride_sum(in, n);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sdata[0]);
}

__global__ void reduce_warp_shuffle(const float* in, float* out, int n) {
    __shared__ float warp_sums[RED_THREADS / 32];
    float sum = warp_reduce_sum(grid_stride_sum(in, n));

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < RED_THREADS / 32) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) atomicAdd(out, sum);
    }
}

__global__ void reduce_atomic(const float* in, float* out, int n) {
    atomicAdd(out, grid_stride_sum(in, n));
}

int main() {
    const int n = (1 << 28) + 1234;

    Lcg rng(123);
    std::vector<float> data(n);
    double ref_sum = 0.0;
    for (auto& v : data) {
        v = rng.next_float(0.0f, 1.0f);
        ref_sum += v;
    }
    float ref = static_cast<float>(ref_sum);

    DeviceBuffer<float> d_in(n), d_out(1);
    d_in.upload(data);
    std::vector<float> got;
    bool ok = true;

    // Summation order differs from the reference, hence the relative tolerance.
    report_time("reduce_shared", time_ms([&] {
        d_out.zero();
        reduce_shared<<<RED_BLOCKS, RED_THREADS>>>(d_in.get(), d_out.get(), n);
    }));
    d_out.download(got);
    ok &= check_close(&ref, got.data(), 1, 1e-4, 0.0, "reduce_shared");

    report_time("reduce_warp_shuffle", time_ms([&] {
        d_out.zero();
        reduce_warp_shuffle<<<RED_BLOCKS, RED_THREADS>>>(d_in.get(), d_out.get(), n);
    }));
    d_out.download(got);
    ok &= check_close(&ref, got.data(), 1, 1e-4, 0.0, "reduce_warp_shuffle");

    report_time("reduce_atomic", time_ms([&] {
        d_out.zero();
        reduce_atomic<<<RED_BLOCKS, RED_THREADS>>>(d_in.get(), d_out.get(), n);
    }));
    d_out.download(got);
    ok &= check_close(&ref, got.data(), 1, 5e-4, 0.0, "reduce_atomic");

    return finish("reduction", ok);
}
