// Locks and inter-block coordination built from atomics: a global spin lock
// (atomicCAS + atomicExch) taken by one thread per block, an array of
// per-bin locks taken by one lane per warp, a ticket counter that hands out
// block ids in arrival order, and the "last block done" reduction of the
// threadFenceReduction sample (__threadfence + atomicInc).
#include "stress.cuh"

constexpr int BS = 256;

__device__ void lock(int* mutex) {
    while (atomicCAS(mutex, 0, 1) != 0) {
    }
    __threadfence();
}

__device__ void unlock(int* mutex) {
    __threadfence();
    atomicExch(mutex, 0);
}

struct Shared {
    long long sum;
    int count;
    int log[4096];
};

// The block reduces its slice, then thread 0 updates the plain (non-atomic)
// structure under the lock: sum, count, and an append to the log.
__global__ void locked_append(const int* v, Shared* st, int* mutex, int n) {
    __shared__ long long part[BS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    part[threadIdx.x] = i < n ? v[i] : 0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        lock(mutex);
        volatile Shared* vs = st;
        vs->sum = vs->sum + part[0];
        int c = vs->count;
        vs->log[c] = blockIdx.x;
        vs->count = c + 1;
        unlock(mutex);
    }
}

// Lane 0 of each warp adds the warp's values to one of `nbins` (sum, sum of
// squares) pairs under that bin's lock; both fields must stay consistent.
__global__ void per_bin_locks(const int* v, long long* pairs, int* mutexes, int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = i < n ? v[i] : 0;
    long long s = x, q = (long long)x * x;
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, o);
        q += __shfl_down_sync(0xffffffffu, q, o);
    }
    if ((threadIdx.x & 31) == 0) {
        int b = (i / 32) % nbins;
        lock(&mutexes[b]);
        volatile long long* p = pairs + 2 * b;
        p[0] = p[0] + s;
        p[1] = p[1] + q;
        unlock(&mutexes[b]);
    }
}

// Blocks take a ticket on arrival and process the chunk named by the
// ticket, not by blockIdx (the "persistent / dynamic block id" idiom).
__global__ void ticket_blocks(const float* in, float* out, int* ticket, int* owner, int n) {
    __shared__ int id;
    if (threadIdx.x == 0) id = atomicAdd(ticket, 1);
    __syncthreads();
    int chunk = id;
    int i = chunk * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f + 1.0f;
    if (threadIdx.x == 0) owner[chunk] = blockIdx.x;
}

// threadFenceReduction: each block writes its partial sum, fences, and takes
// a ticket; the block that takes the last ticket sums all partials and
// resets the counter, so the kernel can run again.
__device__ unsigned int retirementCount = 0;

template <class T>
__global__ void fence_reduce(const T* in, T* partial, T* result, int n) {
    __shared__ T sdata[BS];
    __shared__ bool amLast;
    T s = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += in[i];
    sdata[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sdata[threadIdx.x] += sdata[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sdata[0];
        __threadfence();
        unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
        amLast = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (amLast) {
        T t = 0;
        for (int b = threadIdx.x; b < gridDim.x; b += blockDim.x) t += ((volatile T*)partial)[b];
        sdata[threadIdx.x] = t;
        __syncthreads();
        for (int k = blockDim.x / 2; k > 0; k >>= 1) {
            if (threadIdx.x < k) sdata[threadIdx.x] += sdata[threadIdx.x + k];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            *result = sdata[0];
            retirementCount = 0;
        }
    }
}

// Same pattern with a plain atomicAdd counter and a per-launch generation:
// the last block also records how many blocks it saw finish.
__global__ void last_block_count(const unsigned* in, unsigned long long* partial, unsigned* done,
                                 unsigned long long* result, unsigned* seen, int per_block) {
    __shared__ unsigned long long s[BS];
    __shared__ unsigned ticket;
    unsigned long long acc = 0;
    for (int j = threadIdx.x; j < per_block; j += blockDim.x) acc += in[blockIdx.x * per_block + j];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = s[0];
        __threadfence();
        ticket = atomicAdd(done, 1u);
    }
    __syncthreads();
    if (ticket == gridDim.x - 1 && threadIdx.x == 0) {
        __threadfence();
        unsigned long long t = 0;
        for (unsigned b = 0; b < gridDim.x; ++b) t += ((volatile unsigned long long*)partial)[b];
        *result = t;
        *seen = atomicAdd(done, 0u);
        *done = 0;
    }
}

int main() {
    using namespace stress;

    {
        const int nb = 512, n = nb * BS;
        auto r = random_u32(n, 41);
        std::vector<int> v(n);
        for (int i = 0; i < n; ++i) v[i] = int(r[i] % 100000) - 50000;
        Buffer<int> dv(v), mutex(1);
        Buffer<Shared> st(1);
        double k = time_device([&] {
            mutex.zero();
            st.zero();
            locked_append<<<nb, BS>>>(dv, st, mutex, n);
        });
        long long want = 0;
        double ref = best_ms([&] {
            long long s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int i = 0; i < n; ++i) s += v[i];
            want = s;
        });
        Shared got = st.download()[0];
        bool ok = got.sum == want && got.count == nb && mutex.download()[0] == 0;
        std::vector<int> log(got.log, got.log + nb);
        std::sort(log.begin(), log.end());
        for (int b = 0; ok && b < nb; ++b) ok = log[b] == b;
        if (!ok) std::printf("  sum %lld want %lld, count %d\n", got.sum, want, got.count);
        report("spinlock_block_critical", ok, k, ref);
    }

    {
        const int n = 1 << 18, nbins = 64;
        auto r = random_u32(n, 42);
        std::vector<int> v(n);
        for (int i = 0; i < n; ++i) v[i] = int(r[i] % 2001) - 1000;
        Buffer<int> dv(v), mutexes(nbins);
        Buffer<long long> pairs(2 * nbins);
        double k = time_device([&] {
            mutexes.zero();
            pairs.zero();
            per_bin_locks<<<n / BS, BS>>>(dv, pairs, mutexes, n, nbins);
        });
        std::vector<long long> want(2 * nbins);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nbins; ++b) {
                long long s = 0, q = 0;
                for (int w = b; w < n / 32; w += nbins)
                    for (int l = 0; l < 32; ++l) {
                        long long x = v[w * 32 + l];
                        s += x, q += x * x;
                    }
                want[2 * b] = s, want[2 * b + 1] = q;
            }
        });
        auto got = pairs.download();
        report("spinlock_per_bin_warps", equal(got.data(), want.data(), 2 * nbins), k, ref);
    }

    {
        const int n = 1 << 20, nb = n / BS;
        auto in = random_floats(n, 43);
        Buffer<float> din(in), dout(n);
        Buffer<int> ticket(1), owner(nb);
        double k = time_device([&] {
            ticket.zero();
            ticket_blocks<<<nb, BS>>>(din, dout, ticket, owner, n);
        });
        std::vector<float> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = in[i] * 2.0f + 1.0f;
        });
        auto got = dout.download();
        auto own = owner.download();
        bool ok = ticket.download()[0] == nb && equal(got.data(), want.data(), n);
        std::sort(own.begin(), own.end());
        for (int b = 0; ok && b < nb; ++b) ok = own[b] == b;
        report("ticket_block_ids", ok, k, ref);
    }

    {  // Float: the tree order differs from the host's, so compare against a
       // double sum with a tolerance of 1e-5 of the sum of magnitudes (the
       // float error of a 2^22-term sum split in ~2^14-term pieces is ~1e-7 of that).
        const int n = 1 << 22, nb = 128;
        auto in = random_floats(n, 44);
        Buffer<float> din(in), partial(nb), result(1);
        double k = time_device([&] { fence_reduce<float><<<nb, BS>>>(din, partial, result, n); });
        double want = 0, mag = 0;
        double ref = best_ms([&] {
            double s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int i = 0; i < n; ++i) s += in[i];
            want = s;
        });
        for (int i = 0; i < n; ++i) mag += std::fabs(in[i]);
        double got = result.download()[0];
        report("threadfence_reduction_float", close(&got, &want, 1, 0, 1e-5 * mag), k, ref);
    }

    {
        const int n = 1 << 22, nb = 200;
        auto r = random_u32(n, 45);
        std::vector<int> in(n);
        for (int i = 0; i < n; ++i) in[i] = int(r[i] % 1000) - 500;
        Buffer<int> din(in), partial(nb), result(1);
        double k = time_device([&] { fence_reduce<int><<<nb, BS>>>(din, partial, result, n); });
        int want = 0;
        double ref = best_ms([&] {
            int s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int i = 0; i < n; ++i) s += in[i];
            want = s;
        });
        report("threadfence_reduction_int", result.download()[0] == want, k, ref);
    }

    {
        const int nb = 1000, per = 2048, n = nb * per;
        auto in = random_u32(n, 46);
        Buffer<unsigned> din(in), done(1), seen(1);
        Buffer<unsigned long long> partial(nb), result(1);
        done.zero();
        double k = time_device([&] { last_block_count<<<nb, BS>>>(din, partial, done, result, seen, per); });
        unsigned long long want = 0;
        double ref = best_ms([&] {
            unsigned long long s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int i = 0; i < n; ++i) s += in[i];
            want = s;
        });
        bool ok = result.download()[0] == want && seen.download()[0] == unsigned(nb) && done.download()[0] == 0;
        report("last_block_done_ull", ok, k, ref);
    }

    return finish();
}
