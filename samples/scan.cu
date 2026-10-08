// Exclusive prefix sum (Blelloch work-efficient scan), multi-level.
//
// Exercises: up-sweep/down-sweep loops with a __syncthreads per level, an
// active set that halves then doubles (heavy intra-warp divergence), computed
// shared-memory indices (gather/scatter on shared memory), an optional pointer
// argument checked against nullptr, a per-block uniform load, and recursive
// host-side launches.
#include "common.cuh"

constexpr int SCAN_THREADS = 256;
constexpr int SCAN_ELEMS = 2 * SCAN_THREADS;

__global__ void scan_blocks(const int* in, int* out, int* block_sums, int n) {
    __shared__ int temp[SCAN_ELEMS];

    int tid = threadIdx.x;
    int base = blockIdx.x * SCAN_ELEMS;
    int ai = tid;
    int bi = tid + SCAN_THREADS;
    temp[ai] = (base + ai < n) ? in[base + ai] : 0;
    temp[bi] = (base + bi < n) ? in[base + bi] : 0;

    int offset = 1;
    for (int d = SCAN_ELEMS >> 1; d > 0; d >>= 1) {  // up-sweep
        __syncthreads();
        if (tid < d) {
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            temp[b] += temp[a];
        }
        offset <<= 1;
    }

    if (tid == 0) {
        if (block_sums != nullptr) block_sums[blockIdx.x] = temp[SCAN_ELEMS - 1];
        temp[SCAN_ELEMS - 1] = 0;
    }

    for (int d = 1; d < SCAN_ELEMS; d <<= 1) {  // down-sweep
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            int t = temp[a];
            temp[a] = temp[b];
            temp[b] += t;
        }
    }
    __syncthreads();

    if (base + ai < n) out[base + ai] = temp[ai];
    if (base + bi < n) out[base + bi] = temp[bi];
}

__global__ void add_block_offsets(int* data, const int* offsets, int n) {
    int base = blockIdx.x * SCAN_ELEMS;
    int add = offsets[blockIdx.x];  // uniform across the block
    if (base + threadIdx.x < n) data[base + threadIdx.x] += add;
    if (base + threadIdx.x + SCAN_THREADS < n) data[base + threadIdx.x + SCAN_THREADS] += add;
}

static void scan_device(const int* d_in, int* d_out, int n) {
    int blocks = (n + SCAN_ELEMS - 1) / SCAN_ELEMS;
    if (blocks == 1) {
        scan_blocks<<<1, SCAN_THREADS>>>(d_in, d_out, nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    DeviceBuffer<int> sums(blocks), offsets(blocks);
    scan_blocks<<<blocks, SCAN_THREADS>>>(d_in, d_out, sums.get(), n);
    CUDA_CHECK(cudaGetLastError());
    scan_device(sums.get(), offsets.get(), blocks);
    add_block_offsets<<<blocks, SCAN_THREADS>>>(d_out, offsets.get(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());  // before sums/offsets are freed
}

int main() {
    const int n = (1 << 20) + 7777;

    Lcg rng(11);
    std::vector<int> data(n), ref(n);
    for (auto& v : data) v = static_cast<int>(rng.next_u32() % 10);
    int running = 0;
    for (int i = 0; i < n; ++i) {
        ref[i] = running;
        running += data[i];
    }

    DeviceBuffer<int> d_in(n), d_out(n);
    d_in.upload(data);

    report_time("scan (3 levels)", time_ms([&] { scan_device(d_in.get(), d_out.get(), n); }));

    std::vector<int> got;
    d_out.download(got);
    bool ok = check_equal(ref.data(), got.data(), n, "exclusive scan");
    return finish("scan", ok);
}
