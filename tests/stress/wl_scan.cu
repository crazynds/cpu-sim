// Prefix sums (scan) in the styles found in CUDA textbooks and samples:
// Hillis-Steele double-buffered shared-memory scan, Blelloch work-efficient
// scan with bank-conflict avoidance (GPU Gems 3, ch. 39), a large scan built
// from block scans + scan of block sums + uniform add, a warp-shuffle scan
// (__shfl_up_sync, then a block-level combine of warp totals) on integers and
// floats, and scan-based stream compaction. Integer cases are checked exactly
// against OpenMP host scans.
#include "stress.cuh"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

typedef unsigned int u32;

// ---------------------------------------------------------------------------
// Hillis-Steele inclusive scan of each BLOCK-sized segment, double-buffered.
template <typename T, int BLOCK>
__global__ void hillis_steele_scan(const T* __restrict__ in, T* __restrict__ out, int n) {
    __shared__ T buf[2][BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    int pout = 0, pin = 1;
    buf[pout][tid] = i < n ? in[i] : T(0);
    __syncthreads();
    for (int offset = 1; offset < BLOCK; offset <<= 1) {
        pout = 1 - pout;
        pin = 1 - pout;
        if (tid >= offset)
            buf[pout][tid] = buf[pin][tid] + buf[pin][tid - offset];
        else
            buf[pout][tid] = buf[pin][tid];
        __syncthreads();
    }
    if (i < n) out[i] = buf[pout][tid];
}

// ---------------------------------------------------------------------------
// Blelloch exclusive scan of 2*blockDim.x elements per block (GPU Gems 3).
// Optionally writes the block total into sums[blockIdx.x].
template <typename T>
__global__ void prescan(T* __restrict__ g_odata, const T* __restrict__ g_idata, T* __restrict__ sums, int n) {
    extern __shared__ unsigned char smem_raw[];
    T* temp = reinterpret_cast<T*>(smem_raw);
    int thid = threadIdx.x;
    int nb = 2 * blockDim.x;
    int base = blockIdx.x * nb;

    int ai = thid;
    int bi = thid + blockDim.x;
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = base + ai < n ? g_idata[base + ai] : T(0);
    temp[bi + bankOffsetB] = base + bi < n ? g_idata[base + bi] : T(0);

    int offset = 1;
    for (int d = nb >> 1; d > 0; d >>= 1) {  // up-sweep
        __syncthreads();
        if (thid < d) {
            int a = offset * (2 * thid + 1) - 1;
            int b = offset * (2 * thid + 2) - 1;
            a += CONFLICT_FREE_OFFSET(a);
            b += CONFLICT_FREE_OFFSET(b);
            temp[b] += temp[a];
        }
        offset *= 2;
    }
    if (thid == 0) {
        int last = nb - 1 + CONFLICT_FREE_OFFSET(nb - 1);
        if (sums) sums[blockIdx.x] = temp[last];
        temp[last] = T(0);
    }
    for (int d = 1; d < nb; d *= 2) {  // down-sweep
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int a = offset * (2 * thid + 1) - 1;
            int b = offset * (2 * thid + 2) - 1;
            a += CONFLICT_FREE_OFFSET(a);
            b += CONFLICT_FREE_OFFSET(b);
            T t = temp[a];
            temp[a] = temp[b];
            temp[b] += t;
        }
    }
    __syncthreads();
    if (base + ai < n) g_odata[base + ai] = temp[ai + bankOffsetA];
    if (base + bi < n) g_odata[base + bi] = temp[bi + bankOffsetB];
}

// ---------------------------------------------------------------------------
// Warp-shuffle helpers.
template <typename T>
__device__ __forceinline__ T warp_inclusive_scan(T v) {
    int lane = threadIdx.x & 31;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        T y = __shfl_up_sync(0xffffffffu, v, off);
        if (lane >= off) v += y;
    }
    return v;
}

// Block-wide inclusive scan of one value per thread (blockDim.x multiple of
// 32, at most 1024). Returns the inclusive prefix; *total gets the block sum.
template <typename T>
__device__ __forceinline__ T block_inclusive_scan(T v, T* total) {
    __shared__ T warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    v = warp_inclusive_scan(v);
    if (lane == 31) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        T s = lane < nwarps ? warp_sums[lane] : T(0);
        s = warp_inclusive_scan(s);
        warp_sums[lane] = s;
    }
    __syncthreads();
    if (warp > 0) v += warp_sums[warp - 1];
    if (total) *total = warp_sums[nwarps - 1];
    __syncthreads();  // warp_sums may be reused by the caller's next call
    return v;
}

// Inclusive scan of each 4*blockDim.x segment: each thread loads a vector of
// 4 elements, scans them locally, and the thread totals are scanned with
// shuffles. V is uint4 or float4.
template <typename T, typename V>
__global__ void warp_shuffle_scan(const V* __restrict__ in, V* __restrict__ out, int nvec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    V x = i < nvec ? in[i] : V{0, 0, 0, 0};
    x.y += x.x;
    x.z += x.y;
    x.w += x.z;
    T excl = block_inclusive_scan<T>(x.w, nullptr) - x.w;
    x.x += excl;
    x.y += excl;
    x.z += excl;
    x.w += excl;
    if (i < nvec) out[i] = x;
}

// ---------------------------------------------------------------------------
// Large scan: per-block Blelloch scan (prescan with sums), scan of the block
// sums in one block (looping with a carry), uniform add.
template <typename T>
__global__ void scan_block_sums(T* sums, int nsums) {
    __shared__ T carry_s;
    if (threadIdx.x == 0) carry_s = T(0);
    __syncthreads();
    for (int start = 0; start < nsums; start += blockDim.x) {
        int i = start + threadIdx.x;
        T v = i < nsums ? sums[i] : T(0);
        T total;
        T incl = block_inclusive_scan<T>(v, &total);
        T carry = carry_s;
        if (i < nsums) sums[i] = carry + incl - v;  // exclusive
        __syncthreads();
        if (threadIdx.x == 0) carry_s = carry + total;
        __syncthreads();
    }
}

template <typename T>
__global__ void add_uniform(T* data, const T* __restrict__ incr, int n) {
    // blockDim.x threads, 2*blockDim.x elements per block (matches prescan).
    __shared__ T uni;
    if (threadIdx.x == 0) uni = incr[blockIdx.x];
    __syncthreads();
    int base = blockIdx.x * 2 * blockDim.x;
    int a = base + threadIdx.x, b = a + blockDim.x;
    if (a < n) data[a] += uni;
    if (b < n) data[b] += uni;
}

template <typename T>
struct LargeScan {
    static const int THREADS = 512;
    static const int ELEMS = 2 * THREADS;
    int n, nblocks;
    stress::Buffer<T> sums;
    explicit LargeScan(int n_) : n(n_), nblocks((n_ + ELEMS - 1) / ELEMS), sums(size_t((n_ + ELEMS - 1) / ELEMS)) {}
    void run(T* out, const T* in) {
        size_t sh = (ELEMS + CONFLICT_FREE_OFFSET(ELEMS - 1) + 1) * sizeof(T);
        prescan<T><<<nblocks, THREADS, sh>>>(out, in, sums.p, n);
        scan_block_sums<T><<<1, 1024>>>(sums.p, nblocks);
        add_uniform<T><<<nblocks, THREADS>>>(out, sums.p, n);
    }
};

// ---------------------------------------------------------------------------
// Stream compaction: flags, exclusive scan, scatter.
__host__ __device__ __forceinline__ bool keep(u32 v) { return (v & 0xFFu) < 100u; }

__global__ void compute_flags(const u32* __restrict__ in, u32* __restrict__ flags, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) flags[i] = keep(in[i]) ? 1u : 0u;
}

__global__ void scatter_kept(const u32* __restrict__ in, const u32* __restrict__ flags, const u32* __restrict__ pos,
                             u32* __restrict__ out, u32* __restrict__ count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (flags[i]) out[pos[i]] = in[i];
        if (i == n - 1) *count = pos[i] + flags[i];
    }
}

// ---------------------------------------------------------------------------
// Host references.
template <typename T>
static void ref_segmented_inclusive(const T* in, T* out, int n, int seg) {
    int nseg = (n + seg - 1) / seg;
#pragma omp parallel for schedule(static)
    for (int s = 0; s < nseg; ++s) {
        int lo = s * seg, hi = std::min(n, lo + seg);
        T acc = T(0);
        for (int i = lo; i < hi; ++i) {
            acc += in[i];
            out[i] = acc;
        }
    }
}

template <typename T>
static void ref_segmented_exclusive(const T* in, T* out, int n, int seg) {
    int nseg = (n + seg - 1) / seg;
#pragma omp parallel for schedule(static)
    for (int s = 0; s < nseg; ++s) {
        int lo = s * seg, hi = std::min(n, lo + seg);
        T acc = T(0);
        for (int i = lo; i < hi; ++i) {
            out[i] = acc;
            acc += in[i];
        }
    }
}

// Two-pass parallel exclusive scan over the whole array.
template <typename T>
static void ref_exclusive_scan(const T* in, T* out, size_t n) {
    std::vector<T> part(omp_get_max_threads() + 1);
#pragma omp parallel
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        size_t lo = n * t / nt, hi = n * (t + 1) / nt;
        T s = T(0);
#pragma omp simd reduction(+ : s)
        for (size_t i = lo; i < hi; ++i) s += in[i];
        part[t + 1] = s;
#pragma omp barrier
#pragma omp single
        {
            part[0] = T(0);
            for (int k = 0; k < nt; ++k) part[k + 1] += part[k];
        }
        T run = part[t];
        for (size_t i = lo; i < hi; ++i) {
            out[i] = run;
            run += in[i];
        }
    }
}

static size_t ref_compact(const u32* in, u32* out, size_t n) {
    std::vector<size_t> part(omp_get_max_threads() + 1);
    size_t total = 0;
#pragma omp parallel
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        size_t lo = n * t / nt, hi = n * (t + 1) / nt;
        size_t c = 0;
#pragma omp simd reduction(+ : c)
        for (size_t i = lo; i < hi; ++i) c += keep(in[i]) ? 1 : 0;
        part[t + 1] = c;
#pragma omp barrier
#pragma omp single
        {
            part[0] = 0;
            for (int k = 0; k < nt; ++k) part[k + 1] += part[k];
            total = part[nt];
        }
        size_t w = part[t];
        for (size_t i = lo; i < hi; ++i)
            if (keep(in[i])) out[w++] = in[i];
    }
    return total;
}

static std::vector<u32> small_u32(size_t n, uint32_t seed) {
    auto v = stress::random_u32(n, seed);
    for (auto& x : v) x &= 0xFFu;
    return v;
}

int main() {
    using namespace stress;

    // Hillis-Steele, 1024-element segments.
    {
        const int n = 1 << 24;
        auto in = small_u32(n, 11);
        std::vector<u32> want(n);
        double ref = best_ms([&] { ref_segmented_inclusive(in.data(), want.data(), n, 1024); });
        Buffer<u32> din(in), dout(n);
        double k = time_device([&] { hillis_steele_scan<u32, 1024><<<(n + 1023) / 1024, 1024>>>(din, dout, n); });
        auto got = dout.download();
        report("hillis_steele_1024", equal(got.data(), want.data(), n), k, ref);
    }

    // Blelloch per-block exclusive scan, 1024 elements per block of 512 threads,
    // with a partial last block.
    {
        const int n = (1 << 25) + 333;
        auto in = small_u32(n, 12);
        std::vector<u32> want(n);
        double ref = best_ms([&] { ref_segmented_exclusive(in.data(), want.data(), n, 1024); });
        Buffer<u32> din(in), dout(n);
        size_t sh = (1024 + CONFLICT_FREE_OFFSET(1023) + 1) * sizeof(u32);
        double k = time_device([&] { prescan<u32><<<(n + 1023) / 1024, 512, sh>>>(dout, din, nullptr, n); });
        auto got = dout.download();
        report("blelloch_block_cfo", equal(got.data(), want.data(), n), k, ref);
    }

    // Large exclusive scan over the whole array (3 kernels).
    {
        const int n = (1 << 24) + 777;
        auto in = random_u32(n, 13);  // full-range values: sums wrap, identically on both sides
        std::vector<u32> want(n);
        double ref = best_ms([&] { ref_exclusive_scan(in.data(), want.data(), n); });
        Buffer<u32> din(in), dout(n);
        LargeScan<u32> scan(n);
        double k = time_device([&] { scan.run(dout, din); });
        auto got = dout.download();
        report("large_scan_3kernels", equal(got.data(), want.data(), n), k, ref);
    }

    // Warp-shuffle scan, uint4 per thread, 4096-element segments.
    {
        const int n = 1 << 25;
        auto in = small_u32(n, 14);
        std::vector<u32> want(n);
        double ref = best_ms([&] { ref_segmented_inclusive(in.data(), want.data(), n, 4096); });
        Buffer<u32> din(in), dout(n);
        int nvec = n / 4;
        double k = time_device([&] {
            warp_shuffle_scan<u32, uint4><<<(nvec + 1023) / 1024, 1024>>>((const uint4*)din.p, (uint4*)dout.p, nvec);
        });
        auto got = dout.download();
        report("warp_shuffle_u32", equal(got.data(), want.data(), n), k, ref);
    }

    // Warp-shuffle scan, float4 per thread (positive values, tolerance for order).
    {
        const int n = 1 << 25;
        auto in = random_floats(n, 15, 0.0f, 1.0f);
        std::vector<float> want(n);
        double ref = best_ms([&] { ref_segmented_inclusive(in.data(), want.data(), n, 4096); });
        Buffer<float> din(in), dout(n);
        int nvec = n / 4;
        double k = time_device([&] {
            warp_shuffle_scan<float, float4><<<(nvec + 1023) / 1024, 1024>>>((const float4*)din.p, (float4*)dout.p, nvec);
        });
        auto got = dout.download();
        report("warp_shuffle_f32", close(got.data(), want.data(), n, 2e-4, 1e-5), k, ref);
    }

    // Stream compaction: flags + large scan + scatter.
    {
        const int n = (1 << 24) + 99;
        auto in = random_u32(n, 16);
        std::vector<u32> want(n);
        size_t want_count = 0;
        double ref = best_ms([&] { want_count = ref_compact(in.data(), want.data(), n); });
        Buffer<u32> din(in), dflags(n), dpos(n), dout(n), dcount(1);
        LargeScan<u32> scan(n);
        int blocks = (n + 255) / 256;
        double k = time_device([&] {
            compute_flags<<<blocks, 256>>>(din, dflags, n);
            scan.run(dpos, dflags);
            scatter_kept<<<blocks, 256>>>(din, dflags, dpos, dout, dcount, n);
        });
        auto cnt = dcount.download();
        auto got = dout.download();
        bool ok = cnt[0] == want_count;
        if (!ok) std::printf("  count: got %u want %zu\n", cnt[0], want_count);
        ok = ok && equal(got.data(), want.data(), want_count, "compact");
        report("stream_compaction", ok, k, ref);
    }

    return finish();
}
