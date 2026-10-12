/* Cooperative groups: the subset programs use, on the warp primitives of
 * __cpusim_device.h. A warp is 32 consecutive threads of the block in
 * linear order (x fastest), as on a GPU.
 *
 *   thread_block            this_thread_block(): sync, thread_rank, size...
 *   thread_block_tile<N>    tiled_partition<N>(parent), N a power of two up
 *                           to 32: shfl*, any, all, ballot, match_any/all
 *   thread_group            tiled_partition(parent, n): rank, size, sync
 *   coalesced_group         coalesced_threads(): the lanes active here
 *   grid_group              this_grid(): sync is a grid-wide barrier, valid
 *                           in a cudaLaunchCooperativeKernel launch (every
 *                           block resident at once)
 *   reduce, inclusive_scan, exclusive_scan with plus, less, greater,
 *   bit_and, bit_or, bit_xor (<cooperative_groups/reduce.h>, scan.h). */
#ifndef CPUSIM_COOPERATIVE_GROUPS_H
#define CPUSIM_COOPERATIVE_GROUPS_H

#include "cuda_runtime.h"

namespace cooperative_groups {

namespace __cg {
__device__ __forceinline__ unsigned linear_tid() {
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}
__device__ __forceinline__ unsigned block_threads() { return blockDim.x * blockDim.y * blockDim.z; }
__device__ __forceinline__ unsigned lane() { return linear_tid() & 31u; }
/* Lane of the n-th set bit of mask. */
__device__ __forceinline__ int nth_lane(unsigned mask, unsigned n) {
    for (int l = 0; l < 32; ++l)
        if (mask >> l & 1u) {
            if (n == 0) return l;
            --n;
        }
    return 0;
}
/* The bits of `bits` at the positions set in `mask`, packed (pext). */
__device__ __forceinline__ unsigned pack(unsigned bits, unsigned mask) {
    unsigned r = 0, k = 0;
    for (int l = 0; l < 32; ++l)
        if (mask >> l & 1u) r |= (bits >> l & 1u) << k++;
    return r;
}
}  // namespace __cg

class thread_block {
public:
    __device__ void sync() const { __syncthreads(); }
    __device__ unsigned long long thread_rank() const { return __cg::linear_tid(); }
    __device__ unsigned long long size() const { return __cg::block_threads(); }
    __device__ unsigned long long num_threads() const { return __cg::block_threads(); }
    __device__ dim3 group_index() const { return dim3(blockIdx.x, blockIdx.y, blockIdx.z); }
    __device__ dim3 thread_index() const { return dim3(threadIdx.x, threadIdx.y, threadIdx.z); }
    __device__ dim3 group_dim() const { return dim3(blockDim.x, blockDim.y, blockDim.z); }
    __device__ dim3 dim_threads() const { return group_dim(); }
    __device__ bool is_valid() const { return true; }
};
__device__ __forceinline__ thread_block this_thread_block() { return thread_block(); }

/* A runtime-sized partition (tiled_partition(parent, n)), n up to 32. */
class thread_group {
public:
    __device__ explicit thread_group(unsigned n) : n_(n) {}
    __device__ unsigned long long thread_rank() const { return __cg::linear_tid() % n_; }
    __device__ unsigned long long size() const { return n_; }
    __device__ unsigned long long num_threads() const { return n_; }
    __device__ unsigned long long meta_group_rank() const { return __cg::linear_tid() / n_; }
    __device__ unsigned long long meta_group_size() const { return (__cg::block_threads() + n_ - 1) / n_; }
    __device__ void sync() const {
        if (n_ > 32) __syncthreads();
        else __syncwarp(mask());
    }
    __device__ bool is_valid() const { return true; }
    __device__ unsigned mask() const {
        if (n_ >= 32) return 0xffffffffu;
        return ((1u << n_) - 1u) << (__cg::lane() / n_ * n_);
    }

private:
    unsigned n_;
};

template <unsigned N>
class thread_block_tile {
    static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0, "tile sizes are powers of two up to 32");

public:
    static constexpr unsigned num_threads_v = N;
    __device__ unsigned long long thread_rank() const { return __cg::linear_tid() & (N - 1); }
    __device__ unsigned long long size() const { return N; }
    __device__ unsigned long long num_threads() const { return N; }
    __device__ unsigned long long meta_group_rank() const { return __cg::linear_tid() / N; }
    __device__ unsigned long long meta_group_size() const { return (__cg::block_threads() + N - 1) / N; }
    __device__ bool is_valid() const { return true; }
    __device__ unsigned mask() const {
        return N == 32 ? 0xffffffffu : ((1u << N) - 1u) << (__cg::lane() / N * N);
    }
    __device__ void sync() const { __syncwarp(mask()); }
    template <class T>
    __device__ T shfl(T v, int src) const { return __shfl_sync(mask(), v, src, N); }
    template <class T>
    __device__ T shfl_down(T v, unsigned d) const { return __shfl_down_sync(mask(), v, d, N); }
    template <class T>
    __device__ T shfl_up(T v, unsigned d) const { return __shfl_up_sync(mask(), v, d, N); }
    template <class T>
    __device__ T shfl_xor(T v, int m) const { return __shfl_xor_sync(mask(), v, m, N); }
    __device__ unsigned ballot(int p) const {
        return (__ballot_sync(mask(), p) >> (__cg::lane() / N * N)) & (N == 32 ? 0xffffffffu : (1u << N) - 1u);
    }
    __device__ int any(int p) const { return ballot(p) != 0; }
    __device__ int all(int p) const { return ballot(p) == (N == 32 ? 0xffffffffu : (1u << N) - 1u); }
    template <class T>
    __device__ unsigned match_any(T v) const {
        return (__match_any_sync(mask(), v) >> (__cg::lane() / N * N)) & (N == 32 ? 0xffffffffu : (1u << N) - 1u);
    }
    template <class T>
    __device__ unsigned match_all(T v, int& pred) const {
        return (__match_all_sync(mask(), v, &pred) >> (__cg::lane() / N * N)) & (N == 32 ? 0xffffffffu : (1u << N) - 1u);
    }
};

template <unsigned N, class Parent>
__device__ __forceinline__ thread_block_tile<N> tiled_partition(const Parent&) {
    return thread_block_tile<N>();
}
__device__ __forceinline__ thread_group tiled_partition(const thread_block&, unsigned n) { return thread_group(n); }
__device__ __forceinline__ thread_group tiled_partition(const thread_group&, unsigned n) { return thread_group(n); }

/* The lanes of the warp active where it is created. */
class coalesced_group {
public:
    __device__ explicit coalesced_group(unsigned m) : mask_(m) {}
    __device__ unsigned long long thread_rank() const { return __popc(mask_ & ((1u << __cg::lane()) - 1u)); }
    __device__ unsigned long long size() const { return __popc(mask_); }
    __device__ unsigned long long num_threads() const { return __popc(mask_); }
    __device__ unsigned long long meta_group_rank() const { return 0; }
    __device__ unsigned long long meta_group_size() const { return 1; }
    __device__ bool is_valid() const { return true; }
    __device__ void sync() const { __syncwarp(mask_); }
    __device__ unsigned mask() const { return mask_; }
    template <class T>
    __device__ T shfl(T v, unsigned rank) const { return __shfl_sync(mask_, v, __cg::nth_lane(mask_, rank)); }
    template <class T>
    __device__ T shfl_down(T v, unsigned d) const {
        unsigned r = unsigned(thread_rank()) + d;
        return __shfl_sync(mask_, v, __cg::nth_lane(mask_, r < size() ? r : unsigned(thread_rank())));
    }
    template <class T>
    __device__ T shfl_up(T v, unsigned d) const {
        unsigned r = unsigned(thread_rank());
        return __shfl_sync(mask_, v, __cg::nth_lane(mask_, r >= d ? r - d : r));
    }
    __device__ unsigned ballot(int p) const { return __cg::pack(__ballot_sync(mask_, p), mask_); }
    __device__ int any(int p) const { return __ballot_sync(mask_, p) != 0; }
    __device__ int all(int p) const { return __ballot_sync(mask_, p) == mask_; }

private:
    unsigned mask_;
};
__device__ __forceinline__ coalesced_group coalesced_threads() { return coalesced_group(__activemask()); }

/* Grid-wide barrier: a counter and a generation in device memory. Thread 0
 * of each block arrives once its block is at a barrier; the last one to
 * arrive starts the next generation. */
static __device__ unsigned __cpusim_grid_barrier[2];
class grid_group {
public:
    __device__ void sync() const {
        __syncthreads();
        if (__cg::linear_tid() == 0) {
            volatile unsigned* generation = &__cpusim_grid_barrier[1];
            const unsigned g = *generation;
            const unsigned blocks = gridDim.x * gridDim.y * gridDim.z;
            __threadfence();
            if (atomicAdd(&__cpusim_grid_barrier[0], 1u) == blocks - 1) {
                __cpusim_grid_barrier[0] = 0;
                __threadfence();
                atomicAdd(&__cpusim_grid_barrier[1], 1u);
            } else {
                while (*generation == g) {
                }
            }
            __threadfence();
        }
        __syncthreads();
    }
    __device__ unsigned long long thread_rank() const {
        unsigned long long b = blockIdx.x + gridDim.x * (blockIdx.y + (unsigned long long)gridDim.y * blockIdx.z);
        return b * __cg::block_threads() + __cg::linear_tid();
    }
    __device__ unsigned long long size() const {
        return (unsigned long long)gridDim.x * gridDim.y * gridDim.z * __cg::block_threads();
    }
    __device__ unsigned long long num_threads() const { return size(); }
    __device__ bool is_valid() const { return true; }
    __device__ dim3 group_dim() const { return dim3(gridDim.x, gridDim.y, gridDim.z); }
};
__device__ __forceinline__ grid_group this_grid() { return grid_group(); }

template <class Group>
__device__ __forceinline__ void sync(const Group& g) {
    g.sync();
}

/* Reductions and scans. */
template <class T>
struct plus {
    __device__ T operator()(T a, T b) const { return a + b; }
};
template <class T>
struct less {
    __device__ T operator()(T a, T b) const { return b < a ? b : a; }
};
template <class T>
struct greater {
    __device__ T operator()(T a, T b) const { return a < b ? b : a; }
};
template <class T>
struct bit_and {
    __device__ T operator()(T a, T b) const { return a & b; }
};
template <class T>
struct bit_or {
    __device__ T operator()(T a, T b) const { return a | b; }
};
template <class T>
struct bit_xor {
    __device__ T operator()(T a, T b) const { return a ^ b; }
};

template <unsigned N, class T, class Op>
__device__ T reduce(const thread_block_tile<N>& g, T v, Op op) {
    for (unsigned o = N / 2; o > 0; o /= 2) v = op(v, g.shfl_xor(v, o));
    return v;
}
template <class T, class Op>
__device__ T reduce(const coalesced_group& g, T v, Op op) {
    T r = g.shfl(v, 0);
    for (unsigned i = 1; i < g.size(); ++i) r = op(r, g.shfl(v, i));
    return r;
}
template <unsigned N, class T, class Op>
__device__ T inclusive_scan(const thread_block_tile<N>& g, T v, Op op) {
    for (unsigned d = 1; d < N; d *= 2) {
        T w = g.shfl_up(v, d);
        if (g.thread_rank() >= d) v = op(v, w);
    }
    return v;
}
template <unsigned N, class T>
__device__ T inclusive_scan(const thread_block_tile<N>& g, T v) {
    return inclusive_scan(g, v, plus<T>());
}
template <unsigned N, class T, class Op>
__device__ T exclusive_scan(const thread_block_tile<N>& g, T v, Op op) {
    T inc = inclusive_scan(g, v, op);
    T prev = g.shfl_up(inc, 1);
    return g.thread_rank() == 0 ? T() : prev;
}
template <unsigned N, class T>
__device__ T exclusive_scan(const thread_block_tile<N>& g, T v) {
    return exclusive_scan(g, v, plus<T>());
}
template <class T, class Op>
__device__ T inclusive_scan(const coalesced_group& g, T v, Op op) {
    T r = g.shfl(v, 0);
    for (unsigned i = 1; i <= g.thread_rank(); ++i) r = op(r, g.shfl(v, i));
    return r;
}

}  // namespace cooperative_groups

#endif
