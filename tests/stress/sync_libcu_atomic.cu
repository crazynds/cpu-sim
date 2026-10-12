// libcu++ atomics (<cuda/atomic>, <cuda/std/atomic>): cuda::atomic_ref with
// device and block scope, fetch_add / fetch_min / fetch_max / bitwise ops /
// exchange / compare_exchange loops, release-acquire ordering in the "last
// block done" pattern, cuda::atomic_thread_fence, cuda::atomic objects living
// in device memory, and an atomic_flag spin lock.
#include "stress.cuh"

#include <cuda/atomic>
#include <cuda/std/atomic>

#include <new>

constexpr int BS = 256;

using dev_int = cuda::atomic_ref<int, cuda::thread_scope_device>;

__global__ void ref_tickets(int* counter, int* owner, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dev_int c(*counter);
        owner[c.fetch_add(1, cuda::std::memory_order_relaxed)] = i;
    }
}

__global__ void ref_float_add(const float* v, float* bins, int n, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuda::atomic_ref<float, cuda::thread_scope_device> b(bins[i % nb]);
        b.fetch_add(v[i], cuda::std::memory_order_relaxed);
    }
}

struct Stats {
    int mn, mx;
    unsigned umn, umx, o, a, x, last;
};

__global__ void ref_minmax_bits(const unsigned* v, Stats* s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned u = v[i];
        cuda::atomic_ref<int, cuda::thread_scope_device>(s->mn).fetch_min(int(u));
        cuda::atomic_ref<int, cuda::thread_scope_device>(s->mx).fetch_max(int(u));
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> umn(s->umn), umx(s->umx), o(s->o), a(s->a), x(s->x);
        umn.fetch_min(u);
        umx.fetch_max(u);
        o.fetch_or(u & 0x00ff00ffu);
        a.fetch_and(u | 0x7fff7fffu);
        x.fetch_xor(u);
        cuda::atomic_ref<unsigned, cuda::thread_scope_device>(s->last).exchange(u >> 28);  // any of the values
    }
}

// Saturating add via compare_exchange_weak: the bin never exceeds `cap`.
__global__ void ref_cas_saturate(const int* v, int* bins, int* clipped, int n, int nb, int cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dev_int b(bins[i % nb]);
        int old = b.load(cuda::std::memory_order_relaxed);
        int desired;
        do {
            desired = old + v[i] > cap ? cap : old + v[i];
        } while (!b.compare_exchange_weak(old, desired, cuda::std::memory_order_relaxed));
        if (desired == cap && old + v[i] > cap) dev_int(*clipped).fetch_add(1);
    }
}

__global__ void ref_shared_hist(const unsigned char* data, unsigned* bins, int n) {
    __shared__ unsigned local[256];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        cuda::atomic_ref<unsigned, cuda::thread_scope_block>(local[data[i]]).fetch_add(1, cuda::std::memory_order_relaxed);
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x)
        cuda::atomic_ref<unsigned, cuda::thread_scope_device>(bins[b]).fetch_add(local[b], cuda::std::memory_order_relaxed);
}

// Last block done with acq_rel on the counter (no __threadfence), or with
// relaxed RMW framed by cuda::atomic_thread_fence.
template <bool use_fence>
__global__ void ref_last_block(const int* in, long long* partial, unsigned* done, long long* result, int per) {
    __shared__ long long s[BS];
    __shared__ bool last;
    long long acc = 0;
    for (int j = threadIdx.x; j < per; j += BS) acc += in[blockIdx.x * per + j];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int k = BS / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = s[0];
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> d(*done);
        unsigned t;
        if (use_fence) {
            cuda::atomic_thread_fence(cuda::std::memory_order_release, cuda::thread_scope_device);
            t = d.fetch_add(1, cuda::std::memory_order_relaxed);
            cuda::atomic_thread_fence(cuda::std::memory_order_acquire, cuda::thread_scope_device);
        } else {
            t = d.fetch_add(1, cuda::std::memory_order_acq_rel);
        }
        last = t == gridDim.x - 1;
    }
    __syncthreads();
    if (last && threadIdx.x == 0) {
        long long t = 0;
        for (unsigned b = 0; b < gridDim.x; ++b) t += partial[b];
        *result = t;
        cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*done).store(0, cuda::std::memory_order_relaxed);
    }
}

using counter_t = cuda::atomic<unsigned long long, cuda::thread_scope_device>;

__global__ void init_counters(counter_t* c, int nc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nc) new (&c[i]) counter_t(0);
}

__global__ void bump_counters(const unsigned* v, counter_t* c, unsigned long long* snap, int n, int nc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[v[i] % nc] += (unsigned long long)(v[i] >> 16);
    if (i < nc) snap[i] = 0;
}

__global__ void read_counters(counter_t* c, unsigned long long* snap, int nc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nc) snap[i] = c[i].load();
}

// A spin lock made of a cuda::std::atomic_flag; one thread per block appends
// to a plain array under it.
__global__ void flag_lock_append(cuda::std::atomic_flag* lock, int* count, int* log, long long* sum, const int* v) {
    __shared__ long long s[BS];
    s[threadIdx.x] = v[blockIdx.x * BS + threadIdx.x];
    __syncthreads();
    for (int k = BS / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        while (lock->test_and_set(cuda::std::memory_order_acquire)) {
        }
        int c = *count;
        log[c] = blockIdx.x;
        *count = c + 1;
        *sum += s[0];
        lock->clear(cuda::std::memory_order_release);
    }
}

int main() {
    using namespace stress;
    const int n = 1 << 20, nb = n / BS;

    {
        Buffer<int> c(1), owner(n);
        double k = time_device([&] {
            c.zero();
            ref_tickets<<<nb, BS>>>(c, owner, n);
        });
        auto o = owner.download();
        std::sort(o.begin(), o.end());
        bool ok = c.download()[0] == n;
        for (int i = 0; ok && i < n; ++i) ok = o[i] == i;
        report("atomic_ref_tickets", ok, k, -1);
    }

    {  // Multiples of 2^-8 in [-1,1]: exact in any order (see sync_atomic_global).
        const int bins = 512;
        auto r = random_u32(n, 101);
        std::vector<float> v(n);
        for (int i = 0; i < n; ++i) v[i] = float(int(r[i] % 513) - 256) / 256.0f;
        Buffer<float> dv(v), db(bins);
        double k = time_device([&] {
            db.zero();
            ref_float_add<<<nb, BS>>>(dv, db, n, bins);
        });
        std::vector<float> want(bins);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < bins; ++b) {
                double s = 0;
                for (int i = b; i < n; i += bins) s += v[i];
                want[b] = float(s);
            }
        });
        auto got = db.download();
        report("atomic_ref_float_add", equal(got.data(), want.data(), bins), k, ref);
    }

    {
        auto v = random_u32(n, 102);
        Buffer<unsigned> dv(v);
        Buffer<Stats> ds(1);
        Stats init{INT_MAX, INT_MIN, UINT_MAX, 0, 0, ~0u, 0, 99};
        double k = time_device([&] {
            ds.upload({init});
            ref_minmax_bits<<<nb, BS>>>(dv, ds, n);
        });
        Stats w = init;
        double ref = best_ms([&] {
            int mn = INT_MAX, mx = INT_MIN;
            unsigned umn = UINT_MAX, umx = 0, o = 0, a = ~0u, x = 0;
#pragma omp parallel for reduction(min : mn, umn) reduction(max : mx, umx) reduction(| : o) reduction(& : a) reduction(^ : x)
            for (int i = 0; i < n; ++i) {
                unsigned u = v[i];
                mn = std::min(mn, int(u)), mx = std::max(mx, int(u)), umn = std::min(umn, u), umx = std::max(umx, u);
                o |= u & 0x00ff00ffu, a &= u | 0x7fff7fffu, x ^= u;
            }
            w = {mn, mx, umn, umx, o, a, x, 0};
        });
        Stats g = ds.download()[0];
        bool ok = g.mn == w.mn && g.mx == w.mx && g.umn == w.umn && g.umx == w.umx && g.o == w.o && g.a == w.a &&
                  g.x == w.x && g.last < 16;
        report("atomic_ref_minmax_bitwise_exch", ok, k, ref);
    }

    {  // Saturation: the final value is min(cap, sum) since all addends are
       // positive; the clip count is how many adds overflowed, which depends
       // on order, so only the bin values are checked exactly, and the
       // clip count must be at least one per saturated bin.
        const int bins = 1024, cap = 40000;
        auto r = random_u32(n, 103);
        std::vector<int> v(n);
        for (int i = 0; i < n; ++i) v[i] = int(r[i] % 100) + 1;
        Buffer<int> dv(v), db(bins), clipped(1);
        double k = time_device([&] {
            db.zero();
            clipped.zero();
            ref_cas_saturate<<<nb, BS>>>(dv, db, clipped, n, bins, cap);
        });
        std::vector<int> want(bins);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < bins; ++b) {
                int s = 0;
                for (int i = b; i < n; i += bins) s = std::min(cap, s + v[i]);
                want[b] = s;
            }
        });
        auto got = db.download();
        int sat = 0;
        for (int b = 0; b < bins; ++b) sat += want[b] == cap;
        bool ok = equal(got.data(), want.data(), bins) && clipped.download()[0] >= sat && sat > 0;
        report("atomic_ref_cas_weak_saturate", ok, k, ref);
    }

    {
        const int m = 1 << 22;
        auto r = random_u32(m / 4, 104);
        std::vector<unsigned char> data(m);
        std::memcpy(data.data(), r.data(), m);
        Buffer<unsigned char> dd(data);
        Buffer<unsigned> bins(256);
        double k = time_device([&] {
            bins.zero();
            ref_shared_hist<<<240, BS>>>(dd, bins, m);
        });
        std::vector<unsigned> want(256);
        double ref = best_ms([&] {
            std::fill(want.begin(), want.end(), 0u);
#pragma omp parallel
            {
                unsigned local[256] = {};
#pragma omp for nowait
                for (int i = 0; i < m; ++i) local[data[i]]++;
#pragma omp critical
                for (int b = 0; b < 256; ++b) want[b] += local[b];
            }
        });
        auto got = bins.download();
        report("atomic_ref_block_scope_hist", equal(got.data(), want.data(), 256), k, ref);
    }

    for (int fence = 0; fence < 2; ++fence) {
        const int blocks = 1000, per = 2048, m = blocks * per;
        auto r = random_u32(m, 105);
        std::vector<int> in(m);
        for (int i = 0; i < m; ++i) in[i] = int(r[i] % 2000001) - 1000000;
        Buffer<int> din(in);
        Buffer<long long> partial(blocks), result(1);
        Buffer<unsigned> done(1);
        done.zero();
        double k = time_device([&] {
            if (fence) ref_last_block<true><<<blocks, BS>>>(din, partial, done, result, per);
            else ref_last_block<false><<<blocks, BS>>>(din, partial, done, result, per);
        });
        long long want = 0;
        double ref = best_ms([&] {
            long long s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int i = 0; i < m; ++i) s += in[i];
            want = s;
        });
        bool ok = result.download()[0] == want && done.download()[0] == 0;
        report(fence ? "atomic_thread_fence_last_block" : "atomic_ref_acq_rel_last_block", ok, k, ref);
    }

    {
        const int nc = 4096;
        auto v = random_u32(n, 106);
        Buffer<unsigned> dv(v);
        Buffer<counter_t> c(nc);
        Buffer<unsigned long long> snap(nc);
        double k = time_device([&] {
            init_counters<<<(nc + BS - 1) / BS, BS>>>(c, nc);
            bump_counters<<<nb, BS>>>(dv, c, snap, n, nc);
            read_counters<<<(nc + BS - 1) / BS, BS>>>(c, snap, nc);
        });
        std::vector<unsigned long long> want(nc);
        double ref = best_ms([&] {
            std::fill(want.begin(), want.end(), 0ull);
#pragma omp parallel
            {
                std::vector<unsigned long long> local(nc, 0);
#pragma omp for nowait
                for (int i = 0; i < n; ++i) local[v[i] % nc] += v[i] >> 16;
#pragma omp critical
                for (int b = 0; b < nc; ++b) want[b] += local[b];
            }
        });
        auto got = snap.download();
        report("cuda_atomic_objects_in_global", equal(got.data(), want.data(), nc), k, ref);
    }

    {
        const int blocks = 512;
        auto r = random_u32(blocks * BS, 107);
        std::vector<int> v(blocks * BS);
        for (size_t i = 0; i < v.size(); ++i) v[i] = int(r[i] % 1000);
        Buffer<int> dv(v), count(1), log(blocks);
        Buffer<long long> sum(1);
        Buffer<cuda::std::atomic_flag> lock(1);
        double k = time_device([&] {
            lock.zero();  // a clear atomic_flag is all-zero bytes
            count.zero();
            sum.zero();
            flag_lock_append<<<blocks, BS>>>(lock, count, log, sum, dv);
        });
        long long want = 0;
        for (int x : v) want += x;
        auto lg = log.download();
        std::sort(lg.begin(), lg.end());
        bool ok = count.download()[0] == blocks && sum.download()[0] == want;
        for (int b = 0; ok && b < blocks; ++b) ok = lg[b] == b;
        report("atomic_flag_spinlock", ok, k, -1);
    }

    return finish();
}
