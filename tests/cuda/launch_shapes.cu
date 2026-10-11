// Launch geometry: every thread of every shape must run exactly once and see
// the right built-ins. Covers 1, 2 and 3D blocks and grids, sizes that are not
// multiples of 32 (partial last warp), single-thread blocks, the 1024-thread
// maximum, grids far larger than the worker count, dynamic shared memory and a
// block-wide barrier in each shape.
#include <cstdint>
#include <vector>

#include "check.cuh"

// Each thread writes a code of its coordinates at its linear global index, and
// thread 0 of each block writes the block sum computed through shared memory.
__global__ void shapes(unsigned* ids, unsigned* sums, unsigned* hits) {
    extern __shared__ unsigned s[];
    const unsigned tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    const unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const size_t g = size_t(block) * threads + tid;
    ids[g] = (threadIdx.x | threadIdx.y << 10 | threadIdx.z << 20) ^ (block * 2654435761u);
    hits[g] += 1;  // g is unique to the thread: 2 means it ran twice
    s[tid] = tid + block;
    __syncthreads();
    if (tid == 0) {
        unsigned sum = 0;
        for (unsigned t = 0; t < threads; ++t) sum += s[t];
        sums[block] = sum ^ (blockDim.x << 1) ^ (gridDim.y << 7);
    }
}

// The same without a barrier, so the block-splitting path can be taken.
__global__ void shapes_nobarrier(unsigned* ids, unsigned* hits) {
    const unsigned tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    const unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const size_t g = size_t(block) * threads + tid;
    ids[g] = (threadIdx.x | threadIdx.y << 10 | threadIdx.z << 20) ^ (block * 2654435761u);
    hits[g] += 1;  // g is unique to the thread: 2 means it ran twice
}

struct Shape {
    dim3 grid, block;
};

int main() {
    Checker check("launch_shapes");
    const Shape shapes_list[] = {
        {dim3(1), dim3(1)},           {dim3(7), dim3(1)},           {dim3(1), dim3(33)},
        {dim3(5), dim3(100)},         {dim3(3), dim3(1024)},        {dim3(20000), dim3(32)},
        {dim3(4, 3), dim3(16, 16)},   {dim3(3, 5), dim3(17, 3)},    {dim3(2, 2, 3), dim3(8, 4, 2)},
        {dim3(1, 1, 9), dim3(5, 5, 5)}, {dim3(2, 70), dim3(96)},    {dim3(1000), dim3(31)},
        {dim3(1, 300), dim3(64, 2)},  {dim3(9), dim3(1, 1, 64)},    {dim3(4), dim3(2, 512)},
    };
    for (const Shape& sh : shapes_list) {
        const unsigned threads = sh.block.x * sh.block.y * sh.block.z;
        const unsigned blocks = sh.grid.x * sh.grid.y * sh.grid.z;
        const size_t n = size_t(threads) * blocks;
        unsigned *d_ids, *d_sums, *d_hits;
        CHECK_CUDA(cudaMalloc(&d_ids, n * sizeof(unsigned)));
        CHECK_CUDA(cudaMalloc(&d_hits, n * sizeof(unsigned)));
        CHECK_CUDA(cudaMalloc(&d_sums, blocks * sizeof(unsigned)));
        std::vector<unsigned> ids(n), hits(n), sums(blocks);
        char what[96];
        for (int variant = 0; variant < 2; ++variant) {
            CHECK_CUDA(cudaMemset(d_hits, 0, n * sizeof(unsigned)));
            CHECK_CUDA(cudaMemset(d_ids, 0xff, n * sizeof(unsigned)));
            if (variant == 0)
                shapes<<<sh.grid, sh.block, threads * sizeof(unsigned)>>>(d_ids, d_sums, d_hits);
            else
                shapes_nobarrier<<<sh.grid, sh.block>>>(d_ids, d_hits);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaMemcpy(ids.data(), d_ids, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hits.data(), d_hits, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
            std::snprintf(what, sizeof what, "%s grid(%u,%u,%u) block(%u,%u,%u)", variant ? "nobarrier" : "barrier",
                          sh.grid.x, sh.grid.y, sh.grid.z, sh.block.x, sh.block.y, sh.block.z);
            for (unsigned b = 0; b < blocks; ++b)
                for (unsigned z = 0; z < sh.block.z; ++z)
                    for (unsigned y = 0; y < sh.block.y; ++y)
                        for (unsigned x = 0; x < sh.block.x; ++x) {
                            const size_t g = size_t(b) * threads + x + sh.block.x * (y + sh.block.y * z);
                            const unsigned want = (x | y << 10 | z << 20) ^ (b * 2654435761u);
                            check.expect(ids[g] == want, what, g, ids[g], want);
                            check.expect(hits[g] == 1, "thread ran exactly once", g, hits[g], 1);
                        }
        }
        CHECK_CUDA(cudaMemcpy(sums.data(), d_sums, blocks * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (unsigned b = 0; b < blocks; ++b) {
            unsigned sum = 0;
            for (unsigned t = 0; t < threads; ++t) sum += t + b;
            const unsigned want = sum ^ (sh.block.x << 1) ^ (sh.grid.y << 7);
            check.expect(sums[b] == want, "shared sum", b, sums[b], want);
        }
        cudaFree(d_ids), cudaFree(d_sums), cudaFree(d_hits);
    }
    return check.finish();
}
