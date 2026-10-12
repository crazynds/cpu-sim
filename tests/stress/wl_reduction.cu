// Reductions over 2^24 elements: the seven shared-memory tree kernels of
// Mark Harris, "Optimizing Parallel Reduction in CUDA" (reduce0..reduce6),
// each reduced to a single value by relaunching over the block partials; a
// warp-shuffle reduction finishing with one atomicAdd per block; a two-pass
// reduction without atomics (block partials, then one block); an argmax with
// lowest-index tie-break; and an exact 64-bit integer sum.
#include "stress.cuh"

#include <climits>

// ---------------------------------------------------------------------------
// Harris reduce0..reduce6. Each kernel writes one partial sum per block.

// reduce0: interleaved addressing, divergent branch on tid % (2*s).
__global__ void reduce0(const float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// reduce1: interleaved addressing, strided index (no divergence, bank conflicts).
__global__ void reduce1(const float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        unsigned int index = 2 * s * tid;
        if (index < blockDim.x) sdata[index] += sdata[index + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// reduce2: sequential addressing.
__global__ void reduce2(const float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// reduce3: first add during the global load (each block covers 2*blockDim).
__global__ void reduce3(const float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = mySum = mySum + sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

// Last warp unrolled through volatile shared memory (the classic form).
template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] = sdata[tid] + sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] = sdata[tid] + sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] = sdata[tid] + sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] = sdata[tid] + sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] = sdata[tid] + sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] = sdata[tid] + sdata[tid + 1];
}

// reduce4: reduce3 plus the last warp unrolled.
__global__ void reduce4(const float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] = mySum = mySum + sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) warpReduce<64>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// reduce5: completely unrolled with the block size as a template parameter.
template <unsigned int blockSize>
__global__ void reduce5(const float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockSize < n) mySum += g_idata[i + blockSize];
    sdata[tid] = mySum;
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) sdata[tid] = mySum = mySum + sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] = mySum = mySum + sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] = mySum = mySum + sdata[tid + 64]; __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// reduce6: multiple elements per thread through a grid-stride loop.
template <unsigned int blockSize>
__global__ void reduce6(const float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    float mySum = 0.0f;
    while (i < n) {
        mySum += g_idata[i];
        if (i + blockSize < n) mySum += g_idata[i + blockSize];
        i += gridSize;
    }
    sdata[tid] = mySum;
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) sdata[tid] = mySum = mySum + sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] = mySum = mySum + sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] = mySum = mySum + sdata[tid + 64]; __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Warp shuffle building blocks.

template <class T>
__device__ __forceinline__ T warp_sum(T v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// Block sum: shuffle within each warp, warp totals through shared memory,
// then the first warp reduces them. The result is valid in thread 0.
template <class T>
__device__ T block_sum(T v) {
    __shared__ T warp_totals[32];
    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    v = warp_sum(v);
    if (lane == 0) warp_totals[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    v = (threadIdx.x < nwarps) ? warp_totals[lane] : T(0);
    if (wid == 0) v = warp_sum(v);
    return v;
}

__global__ void reduce_shfl_atomic(const float* __restrict__ in, float* __restrict__ out, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) sum += in[i];
    sum = block_sum(sum);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

// Pass 1 and pass 2 of the two-pass reduction: the same kernel, the second
// launched with a single block over the partials.
__global__ void reduce_partials(const float* __restrict__ in, float* __restrict__ out, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) sum += in[i];
    sum = block_sum(sum);
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

__global__ void reduce_u32_to_u64(const unsigned int* __restrict__ in, unsigned long long* __restrict__ out, int n) {
    unsigned long long sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) sum += in[i];
    sum = block_sum(sum);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

// ---------------------------------------------------------------------------
// Argmax: largest value, lowest index among equal values.

struct ValIdx {
    float v;
    int i;
};

__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

__device__ __forceinline__ void warp_argmax(float& v, int& i) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, v, offset);
        int oi = __shfl_down_sync(0xffffffffu, i, offset);
        if (better(ov, oi, v, i)) { v = ov; i = oi; }
    }
}

// in_idx == nullptr: the values are the input and their indices are positions;
// otherwise (the second pass) the indices come from the partials.
__global__ void argmax_pass(const float* __restrict__ in_v, const int* __restrict__ in_idx, int n,
                            float* __restrict__ out_v, int* __restrict__ out_idx) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += blockDim.x * gridDim.x) {
        float v = in_v[k];
        int idx = in_idx ? in_idx[k] : k;
        if (better(v, idx, bv, bi)) { bv = v; bi = idx; }
    }
    warp_argmax(bv, bi);
    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    if (lane == 0) { sv[wid] = bv; si[wid] = bi; }
    __syncthreads();
    if (wid == 0) {
        int nwarps = (blockDim.x + warpSize - 1) / warpSize;
        bv = lane < nwarps ? sv[lane] : -INFINITY;
        bi = lane < nwarps ? si[lane] : INT_MAX;
        warp_argmax(bv, bi);
        if (lane == 0) { out_v[blockIdx.x] = bv; out_idx[blockIdx.x] = bi; }
    }
}

// ---------------------------------------------------------------------------

using ReduceKernel = void (*)(const float*, float*, unsigned int);

// Reduces in[0..n) to one value by relaunching `k` over the partials until a
// single value remains (in the returned buffer). elems_per_block: how many
// inputs one block consumes; max_blocks caps the grid (reduce6).
static float* reduce_full(ReduceKernel k, const float* in, unsigned int n, float* tmpA, float* tmpB, int threads,
                          int elems_per_block, unsigned int max_blocks) {
    const float* src = in;
    float* dst = tmpA;
    do {
        unsigned int blocks = (n + elems_per_block - 1) / elems_per_block;
        if (blocks > max_blocks) blocks = max_blocks;
        k<<<blocks, threads, threads * sizeof(float)>>>(src, dst, n);
        n = blocks;
        src = dst;
        dst = (dst == tmpA) ? tmpB : tmpA;
    } while (n > 1);
    return const_cast<float*>(src);
}

int main() {
    using namespace stress;
    const int n = 1 << 24;
    const int T = 256;

    // Values in [0, 1): the sum is ~8.4e6, far from cancellation.
    auto x = random_floats(n, 5, 0.0f, 1.0f);
    Buffer<float> dx(x);
    Buffer<float> tmpA(n / T + 1), tmpB(n / T + 1);

    double want = 0;
    double ref = best_ms([&] {
        double s = 0;
#pragma omp parallel for simd reduction(+ : s)
        for (int i = 0; i < n; ++i) s += x[i];
        want = s;
    });

    struct Variant {
        const char* name;
        ReduceKernel k;
        int elems_per_block;
        unsigned int max_blocks;
    };
    const Variant variants[] = {
        {"reduce0_interleaved_divergent", reduce0, T, 1u << 30},
        {"reduce1_interleaved_strided", reduce1, T, 1u << 30},
        {"reduce2_sequential", reduce2, T, 1u << 30},
        {"reduce3_first_add_on_load", reduce3, 2 * T, 1u << 30},
        {"reduce4_unroll_last_warp", reduce4, 2 * T, 1u << 30},
        {"reduce5_complete_unroll", reduce5<T>, 2 * T, 1u << 30},
        {"reduce6_multiple_per_thread", reduce6<T>, 2 * T, 512},
    };
    for (const Variant& v : variants) {
        float* result = nullptr;
        double k = time_device([&] { result = reduce_full(v.k, dx, n, tmpA, tmpB, T, v.elems_per_block, v.max_blocks); });
        float got = 0;
        STRESS_CHECK(cudaMemcpy(&got, result, sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = std::fabs(got - want) <= 1e-4 * std::fabs(want);
        if (!ok) std::printf("  %s: got %.9g want %.9g\n", v.name, got, want);
        report(v.name, ok, k, ref);
    }

    int dev = 0, sms = 0;
    STRESS_CHECK(cudaGetDevice(&dev));
    STRESS_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int grid = sms * 8;

    {  // warp shuffle + one atomicAdd per block
        Buffer<float> out(1);
        double k = time_device([&] {
            STRESS_CHECK(cudaMemsetAsync(out, 0, sizeof(float)));
            reduce_shfl_atomic<<<grid, T>>>(dx, out, n);
        });
        float got = out.download()[0];
        bool ok = std::fabs(got - want) <= 1e-4 * std::fabs(want);
        if (!ok) std::printf("  got %.9g want %.9g\n", got, want);
        report("shfl_down_atomic", ok, k, ref);
    }

    {  // two passes, no atomics
        Buffer<float> partials(grid), out(1);
        double k = time_device([&] {
            reduce_partials<<<grid, T>>>(dx, partials, n);
            reduce_partials<<<1, 1024>>>(partials, out, grid);
        });
        float got = out.download()[0];
        bool ok = std::fabs(got - want) <= 1e-4 * std::fabs(want);
        if (!ok) std::printf("  got %.9g want %.9g\n", got, want);
        report("two_pass_no_atomics", ok, k, ref);
    }

    {  // argmax with ties: values on a coarse grid so the maximum repeats
        std::vector<float> v(n);
        {
            auto r = random_u32(n, 9);
            for (int i = 0; i < n; ++i) v[i] = float(int(r[i] % 100000u) - 50000) * 0.25f;
        }
        float want_v = 0;
        int want_i = 0, count = 0;
        double ref_am = best_ms([&] {
            float bv = -INFINITY;
            int bi = INT_MAX;
#pragma omp parallel
            {
                float lv = -INFINITY;
                int li = INT_MAX;
#pragma omp for nowait
                for (int i = 0; i < n; ++i)
                    if (v[i] > lv) { lv = v[i]; li = i; }  // ascending i: first occurrence wins
#pragma omp critical
                if (lv > bv || (lv == bv && li < bi)) { bv = lv; bi = li; }
            }
            want_v = bv;
            want_i = bi;
        });
        for (int i = 0; i < n; ++i) count += v[i] == want_v;
        Buffer<float> dv(v), pv(grid), outv(1);
        Buffer<int> pi(grid), outi(1);
        double k = time_device([&] {
            argmax_pass<<<grid, T>>>(dv, nullptr, n, pv, pi);
            argmax_pass<<<1, 1024>>>(pv, pi, grid, outv, outi);
        });
        float got_v = outv.download()[0];
        int got_i = outi.download()[0];
        bool ok = got_v == want_v && got_i == want_i;
        if (!ok) std::printf("  got (%g, %d) want (%g, %d)\n", got_v, got_i, want_v, want_i);
        char note[64];
        std::snprintf(note, sizeof note, "ties=%d", count);
        report("argmax_lowest_index", ok, k, ref_am, note);
    }

    {  // exact integer sum: 32-bit inputs, 64-bit accumulation
        auto u = random_u32(n, 13);
        unsigned long long want_u = 0;
        double ref_u = best_ms([&] {
            unsigned long long s = 0;
#pragma omp parallel for simd reduction(+ : s)
            for (int i = 0; i < n; ++i) s += u[i];
            want_u = s;
        });
        Buffer<unsigned int> du(u);
        Buffer<unsigned long long> out(1);
        double k = time_device([&] {
            STRESS_CHECK(cudaMemsetAsync(out, 0, sizeof(unsigned long long)));
            reduce_u32_to_u64<<<grid, T>>>(du, out, n);
        });
        unsigned long long got = out.download()[0];
        bool ok = got == want_u;
        if (!ok) std::printf("  got %llu want %llu\n", got, want_u);
        report("int_sum_u64_exact", ok, k, ref_u);
    }

    return finish();
}
