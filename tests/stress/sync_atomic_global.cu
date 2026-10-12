// atomicAdd on global memory: int, unsigned, unsigned long long, float and
// double, with and without the returned value, with low contention (many
// bins), per-warp addresses and every thread on one address.
#include "stress.cuh"

constexpr int BS = 256;
inline int blocks_for(int n) { return (n + BS - 1) / BS; }

// Values k/256 with |k| <= 256: every partial sum of up to ~2^20 of them is
// a multiple of 2^-8 far below 2^16 in magnitude, so float addition is exact
// in any order and the results can be compared exactly.
std::vector<float> quantized_floats(size_t n, uint32_t seed) {
    auto r = stress::random_u32(n, seed);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = float(int(r[i] % 513) - 256) / 256.0f;
    return v;
}

__global__ void count_all(int* counter, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(counter, 1);
}

__global__ void take_ticket(int* counter, int* slot_owner, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int t = atomicAdd(counter, 1);
        slot_owner[t] = i;
    }
}

__global__ void signed_bins(const int* v, int* bins, int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bins[(unsigned)v[i] % nbins], v[i] >> 8);
}

__global__ void unsigned_hist(const uint32_t* v, unsigned* bins, int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bins[v[i] % nbins], 1u);
}

__global__ void ull_sum(const uint32_t* v, unsigned long long* total, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(total, (unsigned long long)v[i] * 977ull);
}

// One address per warp; the returned value says the lane's position in the
// order the warp's atomics were applied.
__global__ void per_warp_rank(const uint32_t* v, unsigned long long* warp_sum, int* warp_count, int* rank, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int w = i / 32;
        atomicAdd(&warp_sum[w], (unsigned long long)v[i] << 20);
        rank[i] = atomicAdd(&warp_count[w], 1);
    }
}

__global__ void float_one_address(const float* v, float* total, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(total, v[i]);
}

__global__ void float_bins(const float* v, float* bins, int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bins[i % nbins], v[i]);
}

// A float counter: adding 1.0f returns 0, 1, 2, ... (exact below 2^24).
__global__ void float_counter(float* counter, int* slot_owner, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float old = atomicAdd(counter, 1.0f);
        slot_owner[int(old)] = i;
    }
}

__global__ void double_bins(const double* v, double* bins, int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bins[(i * 7) % nbins], v[i]);
}

__global__ void double_one_address(const double* v, double* total, double* olds, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) olds[i] = atomicAdd(total, v[i]);
}

bool is_permutation_of_iota(std::vector<int> v) {
    std::sort(v.begin(), v.end());
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] != int(i)) {
            std::printf("  not a permutation at %zu: %d\n", i, v[i]);
            return false;
        }
    return true;
}

int main() {
    using namespace stress;

    {  // Every thread on one int.
        const int n = 1 << 20;
        Buffer<int> c(1);
        double k = time_device([&] {
            c.zero();
            count_all<<<blocks_for(n), BS>>>(c, n);
        });
        int want = 0;
        double ref = best_ms([&] {
            int s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int i = 0; i < n; ++i) s += 1;
            want = s;
        });
        report("int_one_address", c.download()[0] == want, k, ref);
    }

    {  // Returned value used as a unique slot.
        const int n = 1 << 20;
        Buffer<int> c(1), owner(n);
        double k = time_device([&] {
            c.zero();
            take_ticket<<<blocks_for(n), BS>>>(c, owner, n);
        });
        bool ok = c.download()[0] == n && is_permutation_of_iota(owner.download());
        report("int_ticket_returned", ok, k, -1);
    }

    {  // Signed adds into 4096 bins.
        const int n = 1 << 20, nb = 4096;
        auto r = random_u32(n, 11);
        std::vector<int> v(r.begin(), r.end());
        Buffer<int> dv(v), bins(nb);
        double k = time_device([&] {
            bins.zero();
            signed_bins<<<blocks_for(n), BS>>>(dv, bins, n, nb);
        });
        std::vector<int> want(nb);
        double ref = best_ms([&] {
            std::fill(want.begin(), want.end(), 0);
#pragma omp parallel
            {
                std::vector<int> local(nb, 0);
#pragma omp for nowait
                for (int i = 0; i < n; ++i) local[(unsigned)v[i] % nb] += v[i] >> 8;
#pragma omp critical
                for (int b = 0; b < nb; ++b) want[b] += local[b];
            }
        });
        auto got = bins.download();
        report("int_signed_bins", equal(got.data(), want.data(), nb), k, ref);
    }

    {  // Unsigned histogram, 64 bins (moderate contention).
        const int n = 1 << 21, nb = 64;
        auto v = random_u32(n, 12);
        Buffer<uint32_t> dv(v);
        Buffer<unsigned> bins(nb);
        double k = time_device([&] {
            bins.zero();
            unsigned_hist<<<blocks_for(n), BS>>>(dv, bins, n, nb);
        });
        std::vector<unsigned> want(nb);
        double ref = best_ms([&] {
            std::fill(want.begin(), want.end(), 0u);
#pragma omp parallel
            {
                std::vector<unsigned> local(nb, 0);
#pragma omp for nowait
                for (int i = 0; i < n; ++i) local[v[i] % nb]++;
#pragma omp critical
                for (int b = 0; b < nb; ++b) want[b] += local[b];
            }
        });
        auto got = bins.download();
        report("unsigned_hist64", equal(got.data(), want.data(), nb), k, ref);
    }

    {  // unsigned long long, one address, grid-stride loop.
        const int n = 1 << 21;
        auto v = random_u32(n, 13);
        Buffer<uint32_t> dv(v);
        Buffer<unsigned long long> total(1);
        double k = time_device([&] {
            total.zero();
            ull_sum<<<256, BS>>>(dv, total, n);
        });
        unsigned long long want = 0;
        double ref = best_ms([&] {
            unsigned long long s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int i = 0; i < n; ++i) s += (unsigned long long)v[i] * 977ull;
            want = s;
        });
        report("ull_one_address", total.download()[0] == want, k, ref);
    }

    {  // Per-warp addresses (ull sum and int rank counter).
        const int n = 1 << 20, nw = n / 32;
        auto v = random_u32(n, 14);
        Buffer<uint32_t> dv(v);
        Buffer<unsigned long long> ws(nw);
        Buffer<int> wc(nw), rank(n);
        double k = time_device([&] {
            ws.zero();
            wc.zero();
            per_warp_rank<<<blocks_for(n), BS>>>(dv, ws, wc, rank, n);
        });
        auto got = ws.download();
        auto counts = wc.download();
        auto rk = rank.download();
        std::vector<unsigned long long> want(nw, 0);
        for (int i = 0; i < n; ++i) want[i / 32] += (unsigned long long)v[i] << 20;
        bool ok = equal(got.data(), want.data(), nw, "warp_sum");
        for (int w = 0; ok && w < nw; ++w) {
            if (counts[w] != 32) { std::printf("  warp %d count %d\n", w, counts[w]); ok = false; }
            unsigned seen = 0;
            for (int l = 0; l < 32; ++l) seen |= 1u << rk[w * 32 + l];
            if (seen != 0xffffffffu) { std::printf("  warp %d ranks %08x\n", w, seen); ok = false; }
        }
        report("ull_per_warp_ranked", ok, k, -1);
    }

    {  // float, one address, exact data (see quantized_floats).
        const int n = 1 << 20;
        auto v = quantized_floats(n, 15);
        Buffer<float> dv(v), total(1);
        double k = time_device([&] {
            total.zero();
            float_one_address<<<blocks_for(n), BS>>>(dv, total, n);
        });
        float want = 0;
        double ref = best_ms([&] {
            double s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int i = 0; i < n; ++i) s += v[i];
            want = float(s);
        });
        report("float_one_address_exact", total.download()[0] == want, k, ref);
    }

    {  // float, one address, arbitrary data in [0,1): the result depends on the
       // order of the additions. Typical error of 2^18 adds in float near a sum
       // of ~1.3e5 is ~1e-5 relative; 1e-3 bounds it with a wide margin.
        const int n = 1 << 18;
        auto v = random_floats(n, 16, 0.0f, 1.0f);
        Buffer<float> dv(v), total(1);
        double k = time_device([&] {
            total.zero();
            float_one_address<<<blocks_for(n), BS>>>(dv, total, n);
        });
        double s = 0;
        double ref = best_ms([&] {
            double acc = 0;
#pragma omp parallel for reduction(+ : acc)
            for (int i = 0; i < n; ++i) acc += v[i];
            s = acc;
        });
        double got = total.download()[0];
        report("float_one_address_tol", close(&got, &s, 1, 1e-3), k, ref);
    }

    {  // float, 1024 bins, exact data.
        const int n = 1 << 21, nb = 1024;
        auto v = quantized_floats(n, 17);
        Buffer<float> dv(v), bins(nb);
        double k = time_device([&] {
            bins.zero();
            float_bins<<<blocks_for(n), BS>>>(dv, bins, n, nb);
        });
        std::vector<float> want(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                double s = 0;
                for (int i = b; i < n; i += nb) s += v[i];
                want[b] = float(s);
            }
        });
        auto got = bins.download();
        report("float_bins_exact", equal(got.data(), want.data(), nb), k, ref);
    }

    {  // float counter with the returned value.
        const int n = 1 << 20;
        Buffer<float> c(1);
        Buffer<int> owner(n);
        double k = time_device([&] {
            c.zero();
            float_counter<<<blocks_for(n), BS>>>(c, owner, n);
        });
        bool ok = c.download()[0] == float(n) && is_permutation_of_iota(owner.download());
        report("float_counter_returned", ok, k, -1);
    }

    {  // double, 512 bins, arbitrary data: per-bin error of 4096 double adds is
       // far below 1e-12 of the sum of magnitudes.
        const int n = 1 << 21, nb = 512;
        auto f = random_floats(n, 18);
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i) v[i] = double(f[i]) * 1.000000123456789;
        Buffer<double> dv(v), bins(nb);
        double k = time_device([&] {
            bins.zero();
            double_bins<<<blocks_for(n), BS>>>(dv, bins, n, nb);
        });
        std::vector<double> want(nb, 0.0), mag(nb, 0.0);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) want[b] = 0, mag[b] = 0;
#pragma omp parallel
            {
                std::vector<double> local(nb, 0.0);
#pragma omp for nowait
                for (int i = 0; i < n; ++i) local[(i * 7) % nb] += v[i];
#pragma omp critical
                for (int b = 0; b < nb; ++b) want[b] += local[b];
            }
        });
        for (int i = 0; i < n; ++i) mag[(i * 7) % nb] += std::fabs(v[i]);
        auto got = bins.download();
        bool ok = true;
        for (int b = 0; b < nb && ok; ++b) ok = close(&got[b], &want[b], 1, 0, 1e-12 * mag[b], "bin");
        report("double_bins_tol", ok, k, ref);
    }

    {  // double, one address, returned values: with integer data the returned
       // partial sums are exact, so sorted by value they must chain: each
       // old + its own addend is another thread's old (or the final total).
        const int n = 1 << 18;
        auto r = random_u32(n, 19);
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i) v[i] = double(r[i] % 1000 + 1);  // positive: olds strictly increase
        Buffer<double> dv(v), total(1), olds(n);
        double k = time_device([&] {
            total.zero();
            double_one_address<<<blocks_for(n), BS>>>(dv, total, olds, n);
        });
        double want = 0;
        double ref = best_ms([&] {
            double s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int i = 0; i < n; ++i) s += v[i];
            want = s;
        });
        auto o = olds.download();
        double got = total.download()[0];
        bool ok = got == want;
        std::vector<std::pair<double, double>> chain(n);
        for (int i = 0; i < n; ++i) chain[i] = {o[i], v[i]};
        std::sort(chain.begin(), chain.end());
        if (chain[0].first != 0) ok = false;
        for (int i = 0; ok && i < n; ++i) {
            double next = i + 1 < n ? chain[i + 1].first : got;
            if (chain[i].first + chain[i].second != next) {
                std::printf("  chain broken at %d: %g + %g != %g\n", i, chain[i].first, chain[i].second, next);
                ok = false;
            }
        }
        report("double_one_address_chain", ok, k, ref);
    }

    return finish();
}
