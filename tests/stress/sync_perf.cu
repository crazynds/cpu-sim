// Atomics at scale, each against an OpenMP host version: a 256-bin
// histogram of 2^24 bytes with global atomics and with shared-memory
// privatization; a reduction of 2^24 floats by (a) shared tree + one atomic
// per block, (b) warp shuffles + one atomic per warp, (c) one atomic per
// thread (on 2^21 of them); stream compaction of 2^22 floats with an atomic
// counter, per thread and warp-aggregated.
#include "stress.cuh"

constexpr int BS = 256;
constexpr int N = 1 << 24;

__global__ void hist_global(const unsigned char* d, unsigned* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bins[d[i]], 1u);
}

__global__ void hist_private(const unsigned char* d, unsigned* bins, int n) {
    __shared__ unsigned h[256];
    h[threadIdx.x] = 0;  // blockDim.x == 256
    __syncthreads();
    const uchar4* d4 = reinterpret_cast<const uchar4*>(d);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 4; i += gridDim.x * blockDim.x) {
        uchar4 c = d4[i];
        atomicAdd(&h[c.x], 1u);
        atomicAdd(&h[c.y], 1u);
        atomicAdd(&h[c.z], 1u);
        atomicAdd(&h[c.w], 1u);
    }
    __syncthreads();
    atomicAdd(&bins[threadIdx.x], h[threadIdx.x]);
}

__global__ void reduce_tree_atomic(const float* in, float* total, int n) {
    __shared__ float s[BS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (int k = BS / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(total, s[0]);
}

__global__ void reduce_warp_atomic(const float* in, float* total, int n) {
    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) v += in[i];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(total, v);
}

__global__ void reduce_atomic_each(const float* in, float* total, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(total, in[i]);
}

__global__ void compact_each(const float* in, float* out, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && in[i] > 0.0f) out[atomicAdd(count, 1)] = in[i];
}

__global__ void compact_warp(const float* in, float* out, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    float x = i < n ? in[i] : 0.0f;
    bool keep = x > 0.0f;
    unsigned m = __ballot_sync(0xffffffffu, keep);
    int base = 0;
    if (lane == 0 && m) base = atomicAdd(count, __popc(m));
    base = __shfl_sync(0xffffffffu, base, 0);
    if (keep) out[base + __popc(m & ((1u << lane) - 1))] = x;
}

int main() {
    using namespace stress;

    {
        auto r = random_u32(N / 4, 111);
        std::vector<unsigned char> data(N);
        std::memcpy(data.data(), r.data(), N);
        Buffer<unsigned char> dd(data);
        Buffer<unsigned> bins(256);
        std::vector<unsigned> want(256);
        double ref = best_ms([&] {
            std::fill(want.begin(), want.end(), 0u);
#pragma omp parallel
            {
                unsigned local[256] = {};
#pragma omp for nowait
                for (int i = 0; i < N; ++i) local[data[i]]++;
#pragma omp critical
                for (int b = 0; b < 256; ++b) want[b] += local[b];
            }
        });
        double k = time_device([&] {
            bins.zero();
            hist_global<<<N / BS, BS>>>(dd, bins, N);
        });
        auto got = bins.download();
        report("hist256_global_atomics", equal(got.data(), want.data(), 256), k, ref);

        int sms = 0;
        STRESS_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        k = time_device([&] {
            bins.zero();
            hist_private<<<sms * 8, 256>>>(dd, bins, N);
        });
        got = bins.download();
        report("hist256_shared_private", equal(got.data(), want.data(), 256), k, ref);
    }

    {
        // Multiples of 2^-8 in [-1,1]: every partial sum (any grouping) stays
        // a multiple of 2^-8 well below 2^16 in magnitude for this data, so
        // all three variants and the host agree exactly.
        auto r = random_u32(N, 112);
        std::vector<float> v(N);
        for (int i = 0; i < N; ++i) v[i] = float(int(r[i] % 513) - 256) / 256.0f;
        Buffer<float> dv(v), total(1);
        float want = 0;
        double ref = best_ms([&] {
            double s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int i = 0; i < N; ++i) s += v[i];
            want = float(s);
        });

        double k = time_device([&] {
            total.zero();
            reduce_tree_atomic<<<N / BS, BS>>>(dv, total, N);
        });
        report("reduce_tree_atomic_per_block", total.download()[0] == want, k, ref);

        k = time_device([&] {
            total.zero();
            reduce_warp_atomic<<<4096, BS>>>(dv, total, N);
        });
        report("reduce_shfl_atomic_per_warp", total.download()[0] == want, k, ref);

        // One float atomic per element on one address serializes in the L2
        // (~20 ms on the GPU for 2^24), so this variant takes the first 2^21.
        const int m = 1 << 21;
        float want_m = 0;
        double ref_m = best_ms([&] {
            double s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int i = 0; i < m; ++i) s += v[i];
            want_m = float(s);
        });
        k = time_device([&] {
            total.zero();
            reduce_atomic_each<<<m / BS, BS>>>(dv, total, m);
        });
        report("reduce_atomic_per_thread", total.download()[0] == want_m, k, ref_m, "n=2^21");
    }

    {
        const int nc = 1 << 22;  // the host sorts both sides: keep it short
        auto v = random_floats(nc, 113);
        Buffer<float> dv(v), out(nc);
        Buffer<int> count(1);
        std::vector<float> want;
        double ref = best_ms([&] {
            int nt = omp_get_max_threads();
            std::vector<std::vector<float>> parts(nt);
#pragma omp parallel
            {
                auto& p = parts[omp_get_thread_num()];
                p.clear();
#pragma omp for
                for (int i = 0; i < nc; ++i)
                    if (v[i] > 0.0f) p.push_back(v[i]);
            }
            want.clear();
            for (auto& p : parts) want.insert(want.end(), p.begin(), p.end());
        });
        std::sort(want.begin(), want.end());
        for (int variant = 0; variant < 2; ++variant) {
            double k = time_device([&] {
                count.zero();
                if (variant == 0) compact_each<<<nc / BS, BS>>>(dv, out, count, nc);
                else compact_warp<<<nc / BS, BS>>>(dv, out, count, nc);
            });
            int c = count.download()[0];
            auto o = out.download();
            std::vector<float> got(o.begin(), o.begin() + std::min(c, nc));
            std::sort(got.begin(), got.end());
            bool ok = c == int(want.size()) && got == want;
            report(variant == 0 ? "compact_atomic_per_thread" : "compact_atomic_per_warp", ok, k, ref);
        }
    }

    return finish();
}
