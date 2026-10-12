// Block-level barriers and fences: __syncthreads_count / _and / _or (also
// with a block that is not a multiple of 32), the warp-synchronous tail of
// the classic reduction (volatile shared memory, last 6 steps without
// __syncthreads), and a pipeline of warps handing data to each other through
// volatile shared flags with __threadfence_block.
#include "stress.cuh"

__global__ void vote_block(const float* v, int* count, int* all, int* any, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = i < n ? v[i] : 0.0f;
    int c = __syncthreads_count(x > 0.5f);
    int a = __syncthreads_and(x > -0.99f);
    int o = __syncthreads_or(x > 0.999f);
    if (threadIdx.x == blockDim.x - 1) count[blockIdx.x] = c, all[blockIdx.x] = a != 0, any[blockIdx.x] = o != 0;
}

// Vote results used to steer control flow: every thread of the block must
// agree on the branch.
__global__ void vote_branch(const int* v, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = i < n ? v[i] : 0;
    int r;
    if (__syncthreads_or(x == 12345)) r = 1;
    else if (__syncthreads_and(x % 2 == 0)) r = 2;
    else r = 3 + __syncthreads_count(x % 3 == 0);
    if (i < n) out[i] = r;
}

__device__ void warpReduce(volatile float* sdata, unsigned tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

template <unsigned blockSize>
__global__ void reduce_volatile(const float* g_idata, float* g_odata) {
    __shared__ float sdata[blockSize];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockSize];
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Integer variant with the unroll written inline.
__global__ void reduce_volatile_int(const int* in, int* out) {
    __shared__ int s[256];
    unsigned tid = threadIdx.x;
    s[tid] = in[blockIdx.x * 256 + tid];
    __syncthreads();
    for (unsigned k = 128; k > 32; k >>= 1) {
        if (tid < k) s[tid] += s[tid + k];
        __syncthreads();
    }
    if (tid < 32) {
        volatile int* vs = s;
        vs[tid] += vs[tid + 32];
        vs[tid] += vs[tid + 16];
        vs[tid] += vs[tid + 8];
        vs[tid] += vs[tid + 4];
        vs[tid] += vs[tid + 2];
        vs[tid] += vs[tid + 1];
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// Warp w waits for warp w-1's stage, transforms it into stage w and
// publishes it; the last warp writes the result. Stage 0 comes from global.
__global__ void warp_pipeline(const int* in, int* out) {
    constexpr int W = 8;
    __shared__ volatile int stage[W][32];
    __shared__ volatile int ready[W];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (threadIdx.x < W) ready[threadIdx.x] = 0;
    __syncthreads();
    int x;
    if (w == 0) {
        x = in[blockIdx.x * 32 + lane];
    } else {
        while (ready[w - 1] == 0) {
        }
        __threadfence_block();
        x = stage[w - 1][(lane + w) & 31] * 3 + stage[w - 1][lane] - w;
    }
    stage[w][lane] = x;
    __threadfence_block();
    __syncwarp();
    if (lane == 0) ready[w] = 1;
    if (w == W - 1) out[blockIdx.x * 32 + lane] = x;
}

// Single-flag variant: thread 0 writes a block of data, fences, raises the
// flag; every other thread polls it and then reads.
__global__ void flag_broadcast(const float* in, float* out) {
    __shared__ volatile float data[64];
    __shared__ volatile int flag;
    if (threadIdx.x == 0) flag = 0;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int j = 0; j < 64; ++j) data[j] = in[blockIdx.x * 64 + j] * 0.5f;
        __threadfence_block();
        flag = 1;
    } else {
        while (flag == 0) {
        }
        __threadfence_block();
        out[blockIdx.x * blockDim.x + threadIdx.x] = data[threadIdx.x % 64] + data[(threadIdx.x * 7) % 64];
    }
}

int main() {
    using namespace stress;

    for (int bs : {256, 100, 1024}) {
        const int n = 1 << 20, nb = (n + bs - 1) / bs;
        auto v = random_floats(n, 51 + bs);
        Buffer<float> dv(v);
        Buffer<int> count(nb), all(nb), any(nb);
        double k = time_device([&] { vote_block<<<nb, bs>>>(dv, count, all, any, n); });
        std::vector<int> wc(nb), wa(nb), wo(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                int c = 0, a = 1, o = 0;
                for (int t = 0; t < bs; ++t) {
                    int i = b * bs + t;
                    float x = i < n ? v[i] : 0.0f;
                    c += x > 0.5f, a &= x > -0.99f, o |= x > 0.999f;
                }
                wc[b] = c, wa[b] = a, wo[b] = o;
            }
        });
        auto gc = count.download(), ga = all.download(), go = any.download();
        bool ok = equal(gc.data(), wc.data(), nb, "count") && equal(ga.data(), wa.data(), nb, "and") &&
                  equal(go.data(), wo.data(), nb, "or");
        std::string name = "syncthreads_count_and_or_bs" + std::to_string(bs);
        report(name.c_str(), ok, k, ref);
    }

    {
        const int bs = 128, nb = 4096, n = bs * nb;
        auto r = random_u32(n, 55);
        std::vector<int> v(n);
        for (int i = 0; i < n; ++i) {
            int b = i / bs;
            v[i] = b % 3 == 0 ? int(r[i] % 1000) * 2 : int(r[i] % 1000);  // some all-even blocks
            if (b % 7 == 0 && i % bs == 77) v[i] = 12345;
        }
        Buffer<int> dv(v), out(n);
        double k = time_device([&] { vote_branch<<<nb, bs>>>(dv, out, n); });
        std::vector<int> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                bool o = false, a = true;
                int c = 0;
                for (int t = 0; t < bs; ++t) {
                    int x = v[b * bs + t];
                    o |= x == 12345, a &= x % 2 == 0, c += x % 3 == 0;
                }
                int r = o ? 1 : a ? 2 : 3 + c;
                for (int t = 0; t < bs; ++t) want[b * bs + t] = r;
            }
        });
        auto got = out.download();
        report("syncthreads_vote_branches", equal(got.data(), want.data(), n), k, ref);
    }

    {  // The host replays the same tree, so the float result is bit-exact.
        constexpr unsigned BS = 256;
        const int nb = 4096, n = nb * BS * 2;
        auto v = random_floats(n, 56);
        Buffer<float> dv(v), out(nb);
        double k = time_device([&] { reduce_volatile<BS><<<nb, BS>>>(dv, out); });
        std::vector<float> want(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                float s[BS];
                for (unsigned t = 0; t < BS; ++t) s[t] = v[b * BS * 2 + t] + v[b * BS * 2 + t + BS];
                for (unsigned h = BS / 2; h >= 1; h >>= 1) {
                    float tmp[BS];
                    for (unsigned t = 0; t < h; ++t) tmp[t] = s[t] + s[t + h];
                    for (unsigned t = 0; t < h; ++t) s[t] = tmp[t];
                }
                want[b] = s[0];
            }
        });
        auto got = out.download();
        report("volatile_warp_reduce_float", equal(got.data(), want.data(), nb), k, ref);
    }

    {
        const int nb = 8192, n = nb * 256;
        auto r = random_u32(n, 57);
        std::vector<int> v(n);
        for (int i = 0; i < n; ++i) v[i] = int(r[i] % 20001) - 10000;
        Buffer<int> dv(v), out(nb);
        double k = time_device([&] { reduce_volatile_int<<<nb, 256>>>(dv, out); });
        std::vector<int> want(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                int s = 0;
                for (int t = 0; t < 256; ++t) s += v[b * 256 + t];
                want[b] = s;
            }
        });
        auto got = out.download();
        report("volatile_warp_reduce_int", equal(got.data(), want.data(), nb), k, ref);
    }

    {
        const int nb = 4096;
        auto r = random_u32(nb * 32, 58);
        std::vector<int> in(nb * 32);
        for (size_t i = 0; i < in.size(); ++i) in[i] = int(r[i] % 100);
        Buffer<int> din(in), out(nb * 32);
        double k = time_device([&] { warp_pipeline<<<nb, 256>>>(din, out); });
        std::vector<int> want(nb * 32);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                int cur[32], nxt[32];
                for (int l = 0; l < 32; ++l) cur[l] = in[b * 32 + l];
                for (int w = 1; w < 8; ++w) {
                    for (int l = 0; l < 32; ++l) nxt[l] = cur[(l + w) & 31] * 3 + cur[l] - w;
                    std::memcpy(cur, nxt, sizeof cur);
                }
                for (int l = 0; l < 32; ++l) want[b * 32 + l] = cur[l];
            }
        });
        auto got = out.download();
        report("threadfence_block_warp_pipeline", equal(got.data(), want.data(), nb * 32), k, ref);
    }

    {
        const int nb = 2048, bs = 192;
        auto in = random_floats(nb * 64, 59);
        Buffer<float> din(in), out(nb * bs);
        out.zero();
        double k = time_device([&] { flag_broadcast<<<nb, bs>>>(din, out); });
        std::vector<float> want(nb * bs, 0.0f);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b)
                for (int t = 1; t < bs; ++t)
                    want[b * bs + t] = in[b * 64 + t % 64] * 0.5f + in[b * 64 + (t * 7) % 64] * 0.5f;
        });
        auto got = out.download();
        report("volatile_flag_polling", equal(got.data(), want.data(), nb * bs), k, ref);
    }

    return finish();
}
