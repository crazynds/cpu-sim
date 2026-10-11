// 256-bin histogram of a byte array with a skewed distribution.
//
// Exercises: byte loads (AVX-512 has no byte gather), integer atomicAdd with
// heavy address conflicts inside a warp, shared-memory atomics (both SMT
// threads of a core touch the same block), a cooperative init loop strided by
// blockDim, and a conditional merge into global memory.
#include "common.cuh"

constexpr int BINS = 256;
constexpr int HIST_THREADS = 256;
constexpr int HIST_BLOCKS = 128;

__global__ void hist_global(const uint8_t* data, unsigned int* bins, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&bins[data[i]], 1u);
}

__global__ void hist_shared(const uint8_t* data, unsigned int* bins, int n) {
    __shared__ unsigned int local[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) local[b] = 0;
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&local[data[i]], 1u);
    __syncthreads();

    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        if (local[b] != 0) atomicAdd(&bins[b], local[b]);
}

int main() {
    const int n = (5 << 27) + 77;

    // Half of the samples fall into 4 hot bins to force atomic conflicts.
    Lcg rng(99);
    std::vector<uint8_t> data(n);
    std::vector<unsigned int> ref(BINS, 0);
    for (auto& v : data) {
        uint32_t r = rng.next_u32();
        v = static_cast<uint8_t>((r & 1) ? (r >> 1) % 4 : (r >> 1) % BINS);
        ++ref[v];
    }

    DeviceBuffer<uint8_t> d_data(n);
    DeviceBuffer<unsigned int> d_bins(BINS);
    d_data.upload(data);
    std::vector<unsigned int> got;
    bool ok = true;

    report_time("hist_global", time_ms([&] {
        d_bins.zero();
        hist_global<<<HIST_BLOCKS, HIST_THREADS>>>(d_data.get(), d_bins.get(), n);
    }));
    d_bins.download(got);
    ok &= check_equal(ref.data(), got.data(), BINS, "hist_global");

    report_time("hist_shared", time_ms([&] {
        d_bins.zero();
        hist_shared<<<HIST_BLOCKS, HIST_THREADS>>>(d_data.get(), d_bins.get(), n);
    }));
    d_bins.download(got);
    ok &= check_equal(ref.data(), got.data(), BINS, "hist_shared");

    return finish("histogram", ok);
}
