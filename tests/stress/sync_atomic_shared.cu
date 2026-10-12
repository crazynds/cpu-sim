// Atomics on shared memory: per-block counters, a privatized histogram
// flushed with global atomics, float / double / unsigned long long sums,
// min / max / and / or / xor, atomicInc wrap, atomicExch and a CAS float max,
// and in-block compaction through a shared slot counter.
#include "stress.cuh"

constexpr int BS = 256;

// Same exact-float data as in sync_atomic_global (multiples of 2^-8 in [-1,1]).
std::vector<float> quantized_floats(size_t n, uint32_t seed) {
    auto r = stress::random_u32(n, seed);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = float(int(r[i] % 513) - 256) / 256.0f;
    return v;
}

__global__ void count_per_block(const float* v, int* out, int n) {
    __shared__ int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && v[i] > 0.25f) atomicAdd(&count, 1);
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = count;
}

__global__ void hist_shared(const unsigned char* data, unsigned* bins, int n) {
    __shared__ unsigned local[256];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) atomicAdd(&local[data[i]], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x)
        if (local[b]) atomicAdd(&bins[b], local[b]);
}

__global__ void sums_shared(const float* v, float* fout, double* dout, unsigned long long* uout, int n) {
    __shared__ float fs;
    __shared__ double ds;
    __shared__ unsigned long long us;
    if (threadIdx.x == 0) fs = 0, ds = 0, us = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&fs, v[i]);
        atomicAdd(&ds, double(v[i]) * 3.0);
        atomicAdd(&us, (unsigned long long)(v[i] * 256.0f + 256.0f) << 33);
    }
    __syncthreads();
    if (threadIdx.x == 0) fout[blockIdx.x] = fs, dout[blockIdx.x] = ds, uout[blockIdx.x] = us;
}

struct BitStats {
    int mn, mx;
    unsigned umn, umx, a, o, x;
};

__global__ void bits_shared(const unsigned* v, BitStats* out, int n) {
    __shared__ int mn, mx;
    __shared__ unsigned umn, umx, a, o, x;
    if (threadIdx.x == 0) mn = INT_MAX, mx = INT_MIN, umn = UINT_MAX, umx = 0, a = ~0u, o = 0, x = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned u = v[i];
        atomicMin(&mn, int(u));
        atomicMax(&mx, int(u));
        atomicMin(&umn, u);
        atomicMax(&umx, u);
        atomicAnd(&a, u | 0x11111111u);
        atomicOr(&o, u & 0x01010101u);
        atomicXor(&x, u);
    }
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = {mn, mx, umn, umx, a, o, x};
}

// atomicInc on shared memory as a modular slot allocator: every thread gets
// a slot in [0, 16]; each slot is taken exactly ceil/floor(256/17) times.
__global__ void inc_shared(unsigned* slot_counts, unsigned* finals) {
    __shared__ unsigned c;
    __shared__ unsigned counts[17];
    if (threadIdx.x == 0) c = 0;
    if (threadIdx.x < 17) counts[threadIdx.x] = 0;
    __syncthreads();
    unsigned s = atomicInc(&c, 16);
    atomicAdd(&counts[s], 1u);
    __syncthreads();
    if (threadIdx.x < 17) slot_counts[blockIdx.x * 17 + threadIdx.x] = counts[threadIdx.x];
    if (threadIdx.x == 0) finals[blockIdx.x] = c;
}

// Exch on shared memory: each thread swaps its id into one of 4 slots; the
// block writes out its olds and the 4 finals for the host to check the chain.
__global__ void exch_shared(int* olds, int* finals) {
    __shared__ int slot[4];
    if (threadIdx.x < 4) slot[threadIdx.x] = -1;
    __syncthreads();
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    olds[g] = atomicExch(&slot[threadIdx.x % 4], g);
    __syncthreads();
    if (threadIdx.x < 4) finals[blockIdx.x * 4 + threadIdx.x] = slot[threadIdx.x];
}

__device__ void atomicMaxFloatShared(float* addr, float value) {
    int* p = (int*)addr;
    int old = *p;
    while (value > __int_as_float(old)) {
        int assumed = old;
        old = atomicCAS(p, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

__global__ void max_shared(const float* v, float* out, int n) {
    __shared__ float m;
    if (threadIdx.x == 0) m = -INFINITY;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicMaxFloatShared(&m, v[i]);
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = m;
}

// Keep the positive elements of each block, in arbitrary order, at the
// block's segment of the output; the count goes to counts[block].
__global__ void compact_block(const float* v, float* out, int* counts, int n) {
    __shared__ int pos;
    __shared__ float kept[BS];
    if (threadIdx.x == 0) pos = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && v[i] > 0) kept[atomicAdd(&pos, 1)] = v[i];
    __syncthreads();
    if (threadIdx.x < pos) out[blockIdx.x * BS + threadIdx.x] = kept[threadIdx.x];
    if (threadIdx.x == 0) counts[blockIdx.x] = pos;
}

int main() {
    using namespace stress;

    {
        const int n = 1 << 20, nb = n / BS;
        auto v = random_floats(n, 31);
        Buffer<float> dv(v);
        Buffer<int> out(nb);
        double k = time_device([&] { count_per_block<<<nb, BS>>>(dv, out, n); });
        std::vector<int> want(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                int c = 0;
                for (int i = b * BS; i < (b + 1) * BS; ++i) c += v[i] > 0.25f;
                want[b] = c;
            }
        });
        auto got = out.download();
        report("shared_counter", equal(got.data(), want.data(), nb), k, ref);
    }

    {
        const int n = 1 << 22;
        auto r = random_u32(n / 4, 32);
        std::vector<unsigned char> data(n);
        std::memcpy(data.data(), r.data(), n);
        for (int i = 0; i < n; i += 5) data[i] = 7;  // a hot bin
        Buffer<unsigned char> dd(data);
        Buffer<unsigned> bins(256);
        double k = time_device([&] {
            bins.zero();
            hist_shared<<<240, BS>>>(dd, bins, n);
        });
        std::vector<unsigned> want(256);
        double ref = best_ms([&] {
            std::fill(want.begin(), want.end(), 0u);
#pragma omp parallel
            {
                unsigned local[256] = {};
#pragma omp for nowait
                for (int i = 0; i < n; ++i) local[data[i]]++;
#pragma omp critical
                for (int b = 0; b < 256; ++b) want[b] += local[b];
            }
        });
        auto got = bins.download();
        report("shared_hist_flush", equal(got.data(), want.data(), 256), k, ref);
    }

    {  // Exact data, so float and double block sums match exactly.
        const int n = 1 << 20, nb = n / BS;
        auto v = quantized_floats(n, 33);
        Buffer<float> dv(v), fo(nb);
        Buffer<double> d_o(nb);
        Buffer<unsigned long long> uo(nb);
        double k = time_device([&] { sums_shared<<<nb, BS>>>(dv, fo, d_o, uo, n); });
        std::vector<float> wf(nb);
        std::vector<double> wd(nb);
        std::vector<unsigned long long> wu(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                double f = 0, d = 0;
                unsigned long long u = 0;
                for (int i = b * BS; i < (b + 1) * BS; ++i)
                    f += v[i], d += double(v[i]) * 3.0, u += (unsigned long long)(v[i] * 256.0f + 256.0f) << 33;
                wf[b] = float(f), wd[b] = d, wu[b] = u;
            }
        });
        auto gf = fo.download();
        auto gd = d_o.download();
        auto gu = uo.download();
        bool ok = equal(gf.data(), wf.data(), nb, "float") && equal(gd.data(), wd.data(), nb, "double") &&
                  equal(gu.data(), wu.data(), nb, "ull");
        report("shared_float_double_ull_sums", ok, k, ref);
    }

    {
        const int n = 1 << 20, nb = n / BS;
        auto v = random_u32(n, 34);
        Buffer<unsigned> dv(v);
        Buffer<BitStats> out(nb);
        double k = time_device([&] { bits_shared<<<nb, BS>>>(dv, out, n); });
        std::vector<BitStats> want(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                BitStats s{INT_MAX, INT_MIN, UINT_MAX, 0, ~0u, 0, 0};
                for (int i = b * BS; i < (b + 1) * BS; ++i) {
                    unsigned u = v[i];
                    s.mn = std::min(s.mn, int(u)), s.mx = std::max(s.mx, int(u));
                    s.umn = std::min(s.umn, u), s.umx = std::max(s.umx, u);
                    s.a &= u | 0x11111111u, s.o |= u & 0x01010101u, s.x ^= u;
                }
                want[b] = s;
            }
        });
        auto got = out.download();
        bool ok = true;
        for (int b = 0; b < nb && ok; ++b)
            if (std::memcmp(&got[b], &want[b], sizeof(BitStats)) != 0) {
                std::printf("  block %d differs\n", b);
                ok = false;
            }
        report("shared_min_max_bitwise", ok, k, ref);
    }

    {
        const int nb = 2048;
        Buffer<unsigned> counts(nb * 17), finals(nb);
        double k = time_device([&] { inc_shared<<<nb, BS>>>(counts, finals); });
        auto c = counts.download();
        auto f = finals.download();
        bool ok = true;
        for (int b = 0; b < nb && ok; ++b) {
            if (f[b] != BS % 17) ok = false;
            for (int s = 0; s < 17; ++s)
                if (c[b * 17 + s] != unsigned(BS / 17 + (s < BS % 17))) ok = false;
            if (!ok) std::printf("  block %d wrong (final %u)\n", b, f[b]);
        }
        report("shared_inc_wrap", ok, k, -1);
    }

    {
        const int nb = 1024, n = nb * BS;
        Buffer<int> olds(n), finals(nb * 4);
        double k = time_device([&] { exch_shared<<<nb, BS>>>(olds, finals); });
        auto o = olds.download(), f = finals.download();
        bool ok = true;
        for (int b = 0; b < nb && ok; ++b)
            for (int s = 0; s < 4 && ok; ++s) {
                std::vector<int> got{f[b * 4 + s]}, want{-1};
                for (int t = s; t < BS; t += 4) got.push_back(o[b * BS + t]), want.push_back(b * BS + t);
                std::sort(got.begin(), got.end());
                std::sort(want.begin(), want.end());
                if (got != want) {
                    std::printf("  block %d slot %d history differs\n", b, s);
                    ok = false;
                }
            }
        report("shared_exch_chain", ok, k, -1);
    }

    {
        const int n = 1 << 20, nb = n / BS;
        auto v = random_floats(n, 35, -1000.0f, 1000.0f);
        Buffer<float> dv(v), out(nb);
        double k = time_device([&] { max_shared<<<nb, BS>>>(dv, out, n); });
        std::vector<float> want(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) want[b] = *std::max_element(v.begin() + b * BS, v.begin() + (b + 1) * BS);
        });
        auto got = out.download();
        report("shared_cas_float_max", equal(got.data(), want.data(), nb), k, ref);
    }

    {
        const int n = 1 << 20, nb = n / BS;
        auto v = random_floats(n, 36);
        Buffer<float> dv(v), out(n);
        Buffer<int> counts(nb);
        double k = time_device([&] { compact_block<<<nb, BS>>>(dv, out, counts, n); });
        auto o = out.download();
        auto c = counts.download();
        bool ok = true;
        for (int b = 0; b < nb && ok; ++b) {
            std::vector<float> want;
            for (int i = b * BS; i < (b + 1) * BS; ++i)
                if (v[i] > 0) want.push_back(v[i]);
            std::vector<float> got(o.begin() + b * BS, o.begin() + b * BS + std::min(c[b], BS));
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            if (got != want) {
                std::printf("  block %d: %d kept, want %zu\n", b, c[b], want.size());
                ok = false;
            }
        }
        report("shared_compact_slots", ok, k, -1);
    }

    return finish();
}
