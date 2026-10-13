// Warp-level operations written against warpSize, built with each warp size
// (CPUSIM_WARP_SIZE=16, 32 or 64 at compile time; CMakeLists.txt runs all
// three): reductions and scans with shuffles, votes, ballots of tiles, match,
// lane ids, blocks smaller than a warp (whose warp entry has fewer lanes) and
// odd blocks with barriers. The host computes the expected values with the
// same warpSize; on a GPU (nvcc) it is the 32-thread build.
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <vector>

#include "check.cuh"

namespace cg = cooperative_groups;

// warpSize on the host: cucc's is a constant; nvcc's exists only on the device.
#ifdef __NVCC__
constexpr int kWarp = 32;
#else
constexpr int kWarp = warpSize;
#endif
constexpr int kTile = kWarp < 16 ? kWarp : 16;

__device__ unsigned lane_id() {
    unsigned t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    return t % warpSize;
}

// Per thread: the sum of its warp (butterfly), its inclusive scan in the warp,
// the max of its tile of kTile, any/all over the warp, its lane id, the
// number of lanes of its tile with an odd value (tile ballot) and of its
// tile holding the same value as it (match_any), and the warp size seen.
__global__ void warp_ops(const int* in, int* out, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int v = t < n ? in[t] : 0;
    int sum = v;
    for (int o = warpSize / 2; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    int scan = v;
    for (int d = 1; d < warpSize; d *= 2) {
        int w = __shfl_up_sync(0xffffffffu, scan, d);
        if ((int)lane_id() >= d) scan += w;
    }
    auto tile = cg::tiled_partition<kTile>(cg::this_thread_block());
    int mx = cg::reduce(tile, v, cg::greater<int>());
    int any_neg = __any_sync(0xffffffffu, v < 0);
    int all_small = __all_sync(0xffffffffu, v < 1000);
    int odd = __popc(tile.ballot(v & 1));
    int same = __popc(tile.match_any(v % 3));
    if (t < n) {
        int* o = out + 9 * t;
        o[0] = sum, o[1] = scan, o[2] = mx, o[3] = any_neg, o[4] = all_small;
        o[5] = (int)lane_id(), o[6] = odd, o[7] = same, o[8] = warpSize;
    }
}

// A block smaller than a warp: every thread exists only in the first lanes.
__global__ void small_block(const int* in, int* out) {
    int t = threadIdx.x;
    int v = in[blockIdx.x * blockDim.x + t];
    unsigned active = __popc(__activemask());
    int any_big = __any_sync(__activemask(), v > 90);
    int pair = __shfl_xor_sync(__activemask(), v, 1);  // threads come in pairs here
    auto tile = cg::tiled_partition<4>(cg::this_thread_block());
    int tsum = cg::reduce(tile, v, cg::plus<int>());
    int* o = out + 4 * (blockIdx.x * blockDim.x + t);
    o[0] = (int)active, o[1] = any_big, o[2] = pair, o[3] = tsum;
}

// An odd block with barriers: per-warp partial sums in shared memory.
__global__ void block_sum(const int* in, int* out) {
    __shared__ int partial[1024 / 16];
    int t = threadIdx.x, w = t / warpSize, nw = (blockDim.x + warpSize - 1) / warpSize;
    int v = in[blockIdx.x * blockDim.x + t];
    // The last warp may be partial: lanes beyond the block add nothing.
    for (int o = warpSize / 2; o > 0; o >>= 1) {
        int u = __shfl_down_sync(__activemask(), v, o);
        if ((int)lane_id() + o < warpSize && t + o < (int)blockDim.x) v += u;
    }
    if (lane_id() == 0) partial[w] = v;
    __syncthreads();
    if (t == 0) {
        int s = 0;
        for (int i = 0; i < nw; ++i) s += partial[i];
        out[blockIdx.x] = s;
    }
}

int main() {
    Checker check("warp_size");
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    check.expect(prop.warpSize == kWarp, "cudaDeviceProp::warpSize", -1, prop.warpSize, kWarp);
    int attr = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&attr, cudaDevAttrWarpSize, 0));
    check.expect(attr == kWarp, "cudaDevAttrWarpSize", -1, attr, kWarp);

    {  // whole warps
        const int block = 256, blocks = 40, n = block * blocks;
        std::vector<int> in(n), out(9 * n);
        for (int i = 0; i < n; ++i) in[i] = int((i * 2654435761u) >> 23) % 997 - (i % 97 == 0 ? 5000 : 0);
        int *din, *dout;
        CHECK_CUDA(cudaMalloc(&din, n * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&dout, 9 * n * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(din, in.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        warp_ops<<<blocks, block>>>(din, dout, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(out.data(), dout, 9 * n * sizeof(int), cudaMemcpyDeviceToHost));
        for (int t = 0; t < n; ++t) {
            const int w0 = t / kWarp * kWarp, l = t % kWarp, t0 = t / kTile * kTile;
            int sum = 0, scan = 0, any_neg = 0, all_small = 1, mx = -1 << 30, odd = 0, same = 0;
            for (int i = 0; i < kWarp; ++i) {
                sum += in[w0 + i];
                if (i <= l) scan += in[w0 + i];
                any_neg |= in[w0 + i] < 0;
                all_small &= in[w0 + i] < 1000;
            }
            for (int i = 0; i < kTile; ++i) {
                mx = std::max(mx, in[t0 + i]);
                odd += in[t0 + i] & 1;
                same += in[t0 + i] % 3 == in[t] % 3;
            }
            const int want[9] = {sum, scan, mx, any_neg, all_small, l, odd, same, kWarp};
            static const char* what[9] = {"warp sum", "warp scan", "tile max", "any", "all", "lane id",
                                          "tile ballot", "tile match_any", "warpSize"};
            for (int k = 0; k < 9; ++k) check.expect(out[9 * t + k] == want[k], what[k], t, out[9 * t + k], want[k]);
        }
        CHECK_CUDA(cudaFree(din));
        CHECK_CUDA(cudaFree(dout));
    }

    for (int block : {8, 16}) {  // blocks smaller than a warp
        const int blocks = 33, n = block * blocks;
        std::vector<int> in(n), out(4 * n);
        for (int i = 0; i < n; ++i) in[i] = int((i * 40503u) >> 4) % 100;
        int *din, *dout;
        CHECK_CUDA(cudaMalloc(&din, n * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&dout, 4 * n * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(din, in.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        small_block<<<blocks, block>>>(din, dout);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(out.data(), dout, 4 * n * sizeof(int), cudaMemcpyDeviceToHost));
        for (int b = 0; b < blocks; ++b)
            for (int t = 0; t < block; ++t) {
                const int i = b * block + t;
                int any_big = 0, tsum = 0;
                for (int k = 0; k < block; ++k) any_big |= in[b * block + k] > 90;
                for (int k = t / 4 * 4; k < t / 4 * 4 + 4; ++k) tsum += in[b * block + k];
                const int want[4] = {std::min(block, (int)kWarp), any_big, in[i ^ 1], tsum};
                static const char* what[4] = {"small: active lanes", "small: any", "small: shfl_xor", "small: tile sum"};
                for (int k = 0; k < 4; ++k) check.expect(out[4 * i + k] == want[k], what[k], i, out[4 * i + k], want[k]);
            }
        CHECK_CUDA(cudaFree(din));
        CHECK_CUDA(cudaFree(dout));
    }

    for (int block : {7, 100, 3 * kWarp + 5, 1000}) {  // odd blocks with a barrier
        const int blocks = 50, n = block * blocks;
        std::vector<int> in(n), out(blocks);
        for (int i = 0; i < n; ++i) in[i] = int((i * 2246822519u) >> 20) % 1000 - 300;
        int *din, *dout;
        CHECK_CUDA(cudaMalloc(&din, n * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&dout, blocks * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(din, in.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        block_sum<<<blocks, block>>>(din, dout);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(out.data(), dout, blocks * sizeof(int), cudaMemcpyDeviceToHost));
        for (int b = 0; b < blocks; ++b) {
            int want = 0;
            for (int k = 0; k < block; ++k) want += in[b * block + k];
            check.expect(out[b] == want, "block sum", b, out[b], want);
        }
        CHECK_CUDA(cudaFree(din));
        CHECK_CUDA(cudaFree(dout));
    }
    return check.finish();
}
