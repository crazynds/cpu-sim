// Sorting and searching workloads: bitonic sort and Batcher odd-even merge
// sort (NVIDIA sortingNetworks sample style: shared-memory sort of 1024-key
// chunks, then global merge steps for the large strides), LSD radix sort with
// 4-bit digits (histogram, scan, stable scatter; ping-pong buffers), lower_bound
// of many queries in a sorted array, and merge path of two sorted arrays.
// Host references: OpenMP LSD radix sort, OpenMP lower_bound, OpenMP merge path.
#include "stress.cuh"

typedef unsigned int uint;

// ---------------------------------------------------------------------------
// Sorting networks (keys only, ascending when dir == 1).
// ---------------------------------------------------------------------------
#define SHARED_SIZE_LIMIT 1024U

__device__ inline void Comparator(uint& keyA, uint& keyB, uint dir) {
    if ((keyA > keyB) == dir) {
        uint t = keyA;
        keyA = keyB;
        keyB = t;
    }
}

// Sorts every 1024-key chunk; even chunks ascending, odd chunks descending, so
// each pair of neighbours forms a bitonic sequence for the global merges.
__global__ void bitonicSortShared1(uint* d_Key) {
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    d_Key += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x] = d_Key[0];
    s_key[threadIdx.x + SHARED_SIZE_LIMIT / 2] = d_Key[SHARED_SIZE_LIMIT / 2];

    for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) {
        uint ddd = (threadIdx.x & (size / 2)) != 0;
        for (uint stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos], s_key[pos + stride], ddd);
        }
    }
    uint ddd = blockIdx.x & 1;
    for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(s_key[pos], s_key[pos + stride], ddd);
    }
    __syncthreads();
    d_Key[0] = s_key[threadIdx.x];
    d_Key[SHARED_SIZE_LIMIT / 2] = s_key[threadIdx.x + SHARED_SIZE_LIMIT / 2];
}

// One merge step with a stride too large for shared memory.
__global__ void bitonicMergeGlobal(uint* d_Key, uint arrayLength, uint size, uint stride, uint dir) {
    uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
    uint comparatorI = global_comparatorI & (arrayLength / 2 - 1);
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));
    uint keyA = d_Key[pos], keyB = d_Key[pos + stride];
    Comparator(keyA, keyB, ddd);
    d_Key[pos] = keyA;
    d_Key[pos + stride] = keyB;
}

// The remaining merge steps (stride < 1024) of one size, in shared memory.
__global__ void bitonicMergeShared(uint* d_Key, uint arrayLength, uint size, uint dir) {
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    d_Key += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x] = d_Key[0];
    s_key[threadIdx.x + SHARED_SIZE_LIMIT / 2] = d_Key[SHARED_SIZE_LIMIT / 2];

    uint comparatorI = (blockIdx.x * blockDim.x + threadIdx.x) & ((arrayLength / 2) - 1);
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(s_key[pos], s_key[pos + stride], ddd);
    }
    __syncthreads();
    d_Key[0] = s_key[threadIdx.x];
    d_Key[SHARED_SIZE_LIMIT / 2] = s_key[threadIdx.x + SHARED_SIZE_LIMIT / 2];
}

static void bitonicSort(uint* d_Key, uint arrayLength, uint dir) {
    uint blockCount = arrayLength / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;
    bitonicSortShared1<<<blockCount, threadCount>>>(d_Key);
    for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
        for (uint stride = size / 2; stride > 0; stride >>= 1)
            if (stride >= SHARED_SIZE_LIMIT) {
                bitonicMergeGlobal<<<(arrayLength / 2) / 256, 256>>>(d_Key, arrayLength, size, stride, dir);
            } else {
                bitonicMergeShared<<<blockCount, threadCount>>>(d_Key, arrayLength, size, dir);
                break;
            }
}

// Batcher odd-even merge sort of each 1024-key chunk in shared memory.
__global__ void oddEvenMergeSortShared(uint* d_Key, uint dir) {
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    d_Key += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x] = d_Key[0];
    s_key[threadIdx.x + SHARED_SIZE_LIMIT / 2] = d_Key[SHARED_SIZE_LIMIT / 2];

    for (uint size = 2; size <= SHARED_SIZE_LIMIT; size <<= 1) {
        uint stride = size / 2;
        uint offset = threadIdx.x & (stride - 1);
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos], s_key[pos + stride], dir);
            stride >>= 1;
        }
        for (; stride > 0; stride >>= 1) {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            if (offset >= stride) Comparator(s_key[pos - stride], s_key[pos], dir);
        }
    }
    __syncthreads();
    d_Key[0] = s_key[threadIdx.x];
    d_Key[SHARED_SIZE_LIMIT / 2] = s_key[threadIdx.x + SHARED_SIZE_LIMIT / 2];
}

// One odd-even merge step of the global phase.
__global__ void oddEvenMergeGlobal(uint* d_Key, uint arrayLength, uint size, uint stride, uint dir) {
    uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));
    if (stride < size / 2) {
        uint offset = global_comparatorI & ((size / 2) - 1);
        if (offset >= stride) {
            uint keyA = d_Key[pos - stride], keyB = d_Key[pos];
            Comparator(keyA, keyB, dir);
            d_Key[pos - stride] = keyA;
            d_Key[pos] = keyB;
        }
    } else {
        uint keyA = d_Key[pos], keyB = d_Key[pos + stride];
        Comparator(keyA, keyB, dir);
        d_Key[pos] = keyA;
        d_Key[pos + stride] = keyB;
    }
}

static void oddEvenMergeSort(uint* d_Key, uint arrayLength, uint dir) {
    uint blockCount = arrayLength / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;
    oddEvenMergeSortShared<<<blockCount, threadCount>>>(d_Key, dir);
    for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
        for (uint stride = size / 2; stride > 0; stride >>= 1)
            oddEvenMergeGlobal<<<(arrayLength / 2) / 256, 256>>>(d_Key, arrayLength, size, stride, dir);
}

// ---------------------------------------------------------------------------
// LSD radix sort, 4-bit digits.
// ---------------------------------------------------------------------------
constexpr int RADIX_BITS = 4;
constexpr int BUCKETS = 1 << RADIX_BITS;
constexpr int RS_THREADS = 256;
constexpr int RS_ITEMS = 4;
constexpr int RS_TILE = RS_THREADS * RS_ITEMS;

// Inclusive scan of one value per thread across the block (warp shuffles plus
// one shared array of warp totals). Returns the inclusive prefix; *total gets
// the block sum.
template <int THREADS>
__device__ uint blockInclusiveScan(uint v, uint* total) {
    __shared__ uint s_warp[THREADS / 32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        uint n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) s_warp[warp] = v;
    __syncthreads();
    if (warp == 0) {
        uint w = lane < THREADS / 32 ? s_warp[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            uint n = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += n;
        }
        if (lane < THREADS / 32) s_warp[lane] = w;
    }
    __syncthreads();
    if (warp > 0) v += s_warp[warp - 1];
    *total = s_warp[THREADS / 32 - 1];
    __syncthreads();
    return v;
}

// Digit counts of one tile, stored digit-major: hist[d * numTiles + tile].
__global__ void radixHistogram(const uint* keys, uint* hist, int shift, int numTiles) {
    __shared__ uint s_hist[BUCKETS];
    if (threadIdx.x < BUCKETS) s_hist[threadIdx.x] = 0;
    __syncthreads();
    const uint* tile = keys + size_t(blockIdx.x) * RS_TILE;
#pragma unroll
    for (int j = 0; j < RS_ITEMS; ++j) {
        uint d = (tile[j * RS_THREADS + threadIdx.x] >> shift) & (BUCKETS - 1);
        atomicAdd(&s_hist[d], 1u);
    }
    __syncthreads();
    if (threadIdx.x < BUCKETS) hist[threadIdx.x * numTiles + blockIdx.x] = s_hist[threadIdx.x];
}

// Exclusive scan in place of the whole histogram table by one block.
template <int THREADS>
__global__ void scanSingleBlock(uint* data, int n) {
    int per = (n + THREADS - 1) / THREADS;
    int begin = threadIdx.x * per, end = min(begin + per, n);
    uint sum = 0;
    for (int i = begin; i < end; ++i) sum += data[i];
    uint total;
    uint incl = blockInclusiveScan<THREADS>(sum, &total);
    uint run = incl - sum;
    for (int i = begin; i < end; ++i) {
        uint v = data[i];
        data[i] = run;
        run += v;
    }
}

// Stable scatter of one tile: each thread owns RS_ITEMS consecutive keys; the
// per-thread digit counts are scanned digit-major so that keys of a digit keep
// their input order.
__global__ void radixScatter(const uint* in, uint* out, const uint* histScan, int shift, int numTiles) {
    __shared__ uint s_cnt[BUCKETS * RS_THREADS];
    __shared__ uint s_base[BUCKETS];
    const int tid = threadIdx.x;
    uint4 k4 = reinterpret_cast<const uint4*>(in + size_t(blockIdx.x) * RS_TILE)[tid];
    uint k[RS_ITEMS] = {k4.x, k4.y, k4.z, k4.w};
    uint d[RS_ITEMS];
#pragma unroll
    for (int b = 0; b < BUCKETS; ++b) s_cnt[b * RS_THREADS + tid] = 0;
#pragma unroll
    for (int j = 0; j < RS_ITEMS; ++j) d[j] = (k[j] >> shift) & (BUCKETS - 1);
    // Each thread writes only its own column, no atomics needed.
#pragma unroll
    for (int j = 0; j < RS_ITEMS; ++j) s_cnt[d[j] * RS_THREADS + tid]++;
    __syncthreads();

    // Exclusive scan of the BUCKETS * RS_THREADS counters: each thread scans a
    // contiguous run of BUCKETS entries, then the run totals are block-scanned.
    uint* seg = s_cnt + tid * BUCKETS;
    uint vals[BUCKETS];
    uint sum = 0;
#pragma unroll
    for (int i = 0; i < BUCKETS; ++i) {
        vals[i] = seg[i];
        sum += vals[i];
    }
    uint total;
    uint run = blockInclusiveScan<RS_THREADS>(sum, &total) - sum;
#pragma unroll
    for (int i = 0; i < BUCKETS; ++i) {
        seg[i] = run;
        run += vals[i];
    }
    __syncthreads();
    if (tid < BUCKETS) s_base[tid] = histScan[tid * numTiles + blockIdx.x] - s_cnt[tid * RS_THREADS];
    __syncthreads();

#pragma unroll
    for (int j = 0; j < RS_ITEMS; ++j) {
        uint rank = 0;
#pragma unroll
        for (int i = 0; i < j; ++i) rank += d[i] == d[j];
        out[s_base[d[j]] + s_cnt[d[j] * RS_THREADS + tid] + rank] = k[j];
    }
}

// Sorts d_keys (n a multiple of RS_TILE) using d_tmp; the result ends in d_keys
// (an even number of passes).
static void radixSort(uint* d_keys, uint* d_tmp, uint* d_hist, int n) {
    int numTiles = n / RS_TILE;
    uint* src = d_keys;
    uint* dst = d_tmp;
    for (int shift = 0; shift < 32; shift += RADIX_BITS) {
        radixHistogram<<<numTiles, RS_THREADS>>>(src, d_hist, shift, numTiles);
        scanSingleBlock<1024><<<1, 1024>>>(d_hist, numTiles * BUCKETS);
        radixScatter<<<numTiles, RS_THREADS>>>(src, dst, d_hist, shift, numTiles);
        std::swap(src, dst);
    }
}

// ---------------------------------------------------------------------------
// Binary search: lower_bound of every query.
// ---------------------------------------------------------------------------
__global__ void lowerBoundKernel(const uint* __restrict__ sorted, int n, const uint* __restrict__ queries,
                                 int* __restrict__ result, int nq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nq) return;
    uint q = queries[i];
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (sorted[mid] < q)
            lo = mid + 1;
        else
            hi = mid;
    }
    result[i] = lo;
}

// ---------------------------------------------------------------------------
// Merge path. Ties go to A first (std::merge order).
// ---------------------------------------------------------------------------
constexpr int MP_THREADS = 128;
constexpr int MP_ITEMS = 8;
constexpr int MP_TILE = MP_THREADS * MP_ITEMS;

// Number of A elements among the first `diag` outputs.
template <class T>
__host__ __device__ inline int mergePath(const T* a, int na, const T* b, int nb, int diag) {
    int begin = diag > nb ? diag - nb : 0;
    int end = diag < na ? diag : na;
    while (begin < end) {
        int mid = (begin + end) >> 1;
        if (!(b[diag - 1 - mid] < a[mid]))
            begin = mid + 1;
        else
            end = mid;
    }
    return begin;
}

__global__ void mergePartition(const uint* a, int na, const uint* b, int nb, int* parts, int numParts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParts) return;
    int diag = min(i * MP_TILE, na + nb);
    parts[i] = mergePath(a, na, b, nb, diag);
}

__global__ void mergeTiles(const uint* a, int na, const uint* b, int nb, const int* parts, uint* out) {
    __shared__ uint s_keys[MP_TILE + 1];
    const int tid = threadIdx.x;
    int diag0 = blockIdx.x * MP_TILE;
    int diag1 = min(diag0 + MP_TILE, na + nb);
    int a0 = parts[blockIdx.x], a1 = parts[blockIdx.x + 1];
    int b0 = diag0 - a0, b1 = diag1 - a1;
    int nA = a1 - a0, nB = b1 - b0, count = nA + nB;

    // A segment then B segment, contiguous in shared memory.
    for (int i = tid; i < count; i += MP_THREADS) s_keys[i] = i < nA ? a[a0 + i] : b[b0 + i - nA];
    __syncthreads();

    const uint* sA = s_keys;
    const uint* sB = s_keys + nA;
    int diag = min(tid * MP_ITEMS, count);
    int ai = mergePath(sA, nA, sB, nB, diag);
    int bi = diag - ai;
    uint res[MP_ITEMS];
#pragma unroll
    for (int k = 0; k < MP_ITEMS; ++k) {
        bool takeA = ai < nA && (bi >= nB || !(sB[bi] < sA[ai]));
        res[k] = takeA ? sA[ai] : sB[bi < nB ? bi : 0];
        if (takeA)
            ++ai;
        else
            ++bi;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < MP_ITEMS; ++k)
        if (tid * MP_ITEMS + k < count) s_keys[tid * MP_ITEMS + k] = res[k];
    __syncthreads();
    for (int i = tid; i < count; i += MP_THREADS) out[diag0 + i] = s_keys[i];
}

// ---------------------------------------------------------------------------
// Host references.
// ---------------------------------------------------------------------------

// Parallel LSD radix sort, 8-bit digits, per-thread histograms.
static void host_radix_sort(std::vector<uint>& v) {
    const size_t n = v.size();
    std::vector<uint> tmp(n);
    int T = omp_get_max_threads();
    std::vector<size_t> hist(size_t(T) * 256);
    uint* src = v.data();
    uint* dst = tmp.data();
    for (int shift = 0; shift < 32; shift += 8) {
#pragma omp parallel num_threads(T)
        {
            int t = omp_get_thread_num();
            size_t lo = n * t / T, hi = n * (t + 1) / T;
            size_t* h = &hist[size_t(t) * 256];
            std::fill(h, h + 256, 0);
            for (size_t i = lo; i < hi; ++i) ++h[(src[i] >> shift) & 255];
#pragma omp barrier
#pragma omp single
            {
                size_t run = 0;
                for (int d = 0; d < 256; ++d)
                    for (int tt = 0; tt < T; ++tt) {
                        size_t c = hist[size_t(tt) * 256 + d];
                        hist[size_t(tt) * 256 + d] = run;
                        run += c;
                    }
            }
            for (size_t i = lo; i < hi; ++i) dst[h[(src[i] >> shift) & 255]++] = src[i];
        }
        std::swap(src, dst);
    }
}

// Parallel merge: every thread merges its own merge-path slice.
static void host_merge(const std::vector<uint>& a, const std::vector<uint>& b, std::vector<uint>& out) {
    int na = int(a.size()), nb = int(b.size()), total = na + nb;
#pragma omp parallel
    {
        int T = omp_get_num_threads(), t = omp_get_thread_num();
        int d0 = int(int64_t(total) * t / T), d1 = int(int64_t(total) * (t + 1) / T);
        int a0 = mergePath(a.data(), na, b.data(), nb, d0), a1 = mergePath(a.data(), na, b.data(), nb, d1);
        std::merge(a.begin() + a0, a.begin() + a1, b.begin() + (d0 - a0), b.begin() + (d1 - a1), out.begin() + d0);
    }
}

int main() {
    using namespace stress;

    // Bitonic and odd-even merge sort of 2^20 keys.
    {
        const uint n = 1u << 20;
        auto keys = random_u32(n, 11);
        std::vector<uint> want;
        double ref = best_ms([&] {
            want = keys;
            host_radix_sort(want);
        });
        Buffer<uint> d_in(keys), d_work(n);

        double k = time_device([&] {
            STRESS_CHECK(cudaMemcpy(d_work, d_in, n * sizeof(uint), cudaMemcpyDeviceToDevice));
            bitonicSort(d_work, n, 1);
        });
        auto got = d_work.download();
        report("bitonic_sort_1M", equal(got.data(), want.data(), n, "bitonic"), k, ref);

        k = time_device([&] {
            STRESS_CHECK(cudaMemcpy(d_work, d_in, n * sizeof(uint), cudaMemcpyDeviceToDevice));
            oddEvenMergeSort(d_work, n, 1);
        });
        got = d_work.download();
        report("odd_even_merge_sort_1M", equal(got.data(), want.data(), n, "oddeven"), k, ref);
    }

    // Bitonic sort with many duplicate keys (only 256 distinct values).
    {
        const uint n = 1u << 19;
        auto keys = random_u32(n, 12);
        for (auto& x : keys) x &= 0xff00u;
        std::vector<uint> want;
        double ref = best_ms([&] {
            want = keys;
            host_radix_sort(want);
        });
        Buffer<uint> d_in(keys), d_work(n);
        double k = time_device([&] {
            STRESS_CHECK(cudaMemcpy(d_work, d_in, n * sizeof(uint), cudaMemcpyDeviceToDevice));
            bitonicSort(d_work, n, 1);
        });
        auto got = d_work.download();
        report("bitonic_sort_dups_512K", equal(got.data(), want.data(), n, "bitonic"), k, ref);
    }

    // Radix sort of 2^20 keys, 8 passes of 4 bits.
    {
        const int n = 1 << 20;
        auto keys = random_u32(n, 21);
        std::vector<uint> want;
        double ref = best_ms([&] {
            want = keys;
            host_radix_sort(want);
        });
        Buffer<uint> d_in(keys), d_keys(n), d_tmp(n), d_hist(size_t(n / RS_TILE) * BUCKETS);
        double k = time_device([&] {
            STRESS_CHECK(cudaMemcpy(d_keys, d_in, n * sizeof(uint), cudaMemcpyDeviceToDevice));
            radixSort(d_keys, d_tmp, d_hist, n);
        });
        auto got = d_keys.download();
        report("radix_sort_1M", equal(got.data(), want.data(), n, "radix"), k, ref);
    }

    // lower_bound of 2^22 queries in a sorted array of 2^20 keys (some queries
    // are exact hits, some fall past the end).
    {
        const int n = 1 << 20, nq = 1 << 22;
        auto sorted = random_u32(n, 31);
        host_radix_sort(sorted);
        auto queries = random_u32(nq, 32);
        for (int i = 0; i < nq; i += 4) queries[i] = sorted[stress::hash32(i) % n];
        std::vector<int> want(nq);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < nq; ++i)
                want[i] = int(std::lower_bound(sorted.begin(), sorted.end(), queries[i]) - sorted.begin());
        });
        Buffer<uint> d_sorted(sorted), d_q(queries);
        Buffer<int> d_res(nq);
        double k = time_device([&] { lowerBoundKernel<<<(nq + 255) / 256, 256>>>(d_sorted, n, d_q, d_res, nq); });
        auto got = d_res.download();
        report("lower_bound_4M_queries", equal(got.data(), want.data(), nq, "lb"), k, ref);
    }

    // Merge path of two sorted arrays of different lengths with shared keys.
    {
        const int na = (1 << 23) + 12345, nb = (1 << 23) - 777;
        auto a = random_u32(na, 41), b = random_u32(nb, 42);
        for (auto& x : a) x >>= 8;  // a narrower key range than b: many ties and long runs
        for (auto& x : b) x >>= 9;
        host_radix_sort(a);
        host_radix_sort(b);
        const int total = na + nb;
        std::vector<uint> want(total);
        double ref = best_ms([&] { host_merge(a, b, want); });
        Buffer<uint> d_a(a), d_b(b), d_out(total);
        const int numTiles = (total + MP_TILE - 1) / MP_TILE;
        Buffer<int> d_parts(numTiles + 1);
        double k = time_device([&] {
            mergePartition<<<(numTiles + 1 + 255) / 256, 256>>>(d_a, na, d_b, nb, d_parts, numTiles + 1);
            mergeTiles<<<numTiles, MP_THREADS>>>(d_a, na, d_b, nb, d_parts, d_out);
        });
        auto got = d_out.download();
        report("merge_path_16M", equal(got.data(), want.data(), total, "merge"), k, ref);
    }

    return finish();
}
