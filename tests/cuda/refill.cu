// Kernels whose threads loop for different counts, as the refill entry
// (lanes that take a new thread when theirs is done) rewrites them: an early
// return, the thread's indices read before, inside and after the loop, a
// per-thread array, 2D and 3D blocks and grids, the lane id, and a grid
// whose last block is partly out of range. Compared exactly with the host.
#include <cstdint>
#include <vector>

#include "check.cuh"

// Collatz steps of n (a different count for every thread).
__host__ __device__ inline uint32_t collatz(uint64_t n) {
    uint32_t steps = 0;
    while (n != 1) {
        n = (n & 1) ? 3 * n + 1 : n / 2;
        ++steps;
    }
    return steps;
}

__global__ void steps_1d(const uint32_t* start, uint32_t* out, uint32_t* lane, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t v = start[i];
    uint32_t s = 0;
    while (v != 1) {
        v = (v & 1) ? 3 * v + 1 : v / 2;
        ++s;
    }
    out[i] = s + threadIdx.x;  // the thread's index again, after the loop
    lane[i] = (threadIdx.x & 31) == (i % 32) ? 1 : 0;
}

// A per-thread array filled before the loop and read in it; 3D blocks and a 2D grid.
__global__ void sums_3d(uint32_t* out, int width) {
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int t = (tz * blockDim.y + ty) * blockDim.x + tx;
    const int b = by * gridDim.x + bx;
    const int id = b * blockDim.x * blockDim.y * blockDim.z + t;
    uint32_t digits[8];
    for (int k = 0; k < 8; ++k) digits[k] = (id >> (2 * k)) & 3;
    uint32_t acc = 0;
    for (uint32_t k = 0; k < (uint32_t(id) % 97) * 3 + digits[id % 8]; ++k) acc += digits[k % 8] * k;
    if (acc % 5 == 4) return;  // some threads leave after the loop without writing
    out[id] = acc + uint32_t(width) * tz + by;
}

int main() {
    Checker c("refill");
    const int n = 5000, threads = 128, blocks = (n + threads - 1) / threads;
    std::vector<uint32_t> start(n);
    for (int i = 0; i < n; ++i) start[i] = uint32_t(i * 7919 % 100000 + 1);
    uint32_t *dstart, *dout, *dlane;
    CHECK_CUDA(cudaMalloc(&dstart, n * 4));
    CHECK_CUDA(cudaMalloc(&dout, n * 4));
    CHECK_CUDA(cudaMalloc(&dlane, n * 4));
    CHECK_CUDA(cudaMemcpy(dstart, start.data(), n * 4, cudaMemcpyHostToDevice));
    for (int rep = 0; rep < 4; ++rep) {  // later launches run the width the measurement picked
        CHECK_CUDA(cudaMemset(dout, 0, n * 4));
        steps_1d<<<blocks, threads>>>(dstart, dout, dlane, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<uint32_t> out(n), lane(n);
        CHECK_CUDA(cudaMemcpy(out.data(), dout, n * 4, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(lane.data(), dlane, n * 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            c.expect(out[i] == collatz(start[i]) + uint32_t(i % threads), "collatz steps", i, out[i],
                     collatz(start[i]) + uint32_t(i % threads));
            c.expect(lane[i] == 1, "lane id", i, lane[i], 1);
        }
    }

    const dim3 grid(5, 3), block(8, 4, 2);
    const int total = 5 * 3 * 8 * 4 * 2, width = 1000;
    uint32_t* d3;
    CHECK_CUDA(cudaMalloc(&d3, total * 4));
    for (int rep = 0; rep < 4; ++rep) {
        CHECK_CUDA(cudaMemset(d3, 0xff, total * 4));
        sums_3d<<<grid, block>>>(d3, width);
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<uint32_t> out(total);
        CHECK_CUDA(cudaMemcpy(out.data(), d3, total * 4, cudaMemcpyDeviceToHost));
        for (int id = 0; id < total; ++id) {
            const int per = 8 * 4 * 2, t = id % per, b = id / per;
            const int tz = t / 32, by = b / 5;
            uint32_t digits[8];
            for (int k = 0; k < 8; ++k) digits[k] = (id >> (2 * k)) & 3;
            uint32_t acc = 0;
            for (uint32_t k = 0; k < (uint32_t(id) % 97) * 3 + digits[id % 8]; ++k) acc += digits[k % 8] * k;
            const uint32_t want = acc % 5 == 4 ? 0xffffffffu : acc + uint32_t(width) * tz + by;
            c.expect(out[id] == want, "3D sums", id, out[id], want);
        }
    }
    CHECK_CUDA(cudaFree(dstart));
    CHECK_CUDA(cudaFree(dout));
    CHECK_CUDA(cudaFree(dlane));
    CHECK_CUDA(cudaFree(d3));
    return c.finish();
}
