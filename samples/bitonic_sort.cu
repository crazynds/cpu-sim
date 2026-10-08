// Bitonic sort of independent 1024-element segments, one segment per block.
//
// Exercises: nested loops with a __syncthreads in the inner one (55 barriers
// per block), bit-twiddled pair indices (gather/scatter on shared memory), and
// a data-dependent swap: which lanes write depends on the values themselves.
#include "common.cuh"

#include <algorithm>

constexpr int SORT_N = 1024;
constexpr int SORT_THREADS = SORT_N / 2;

__global__ void bitonic_sort_segments(int* data) {
    __shared__ int s[SORT_N];

    int base = blockIdx.x * SORT_N;
    int t = threadIdx.x;
    s[t] = data[base + t];
    s[t + SORT_THREADS] = data[base + t + SORT_THREADS];

    for (int size = 2; size <= SORT_N; size <<= 1) {
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            int pos = 2 * t - (t & (stride - 1));  // first element of this lane's pair
            bool ascending = (pos & size) == 0;
            int a = s[pos];
            int b = s[pos + stride];
            if ((a > b) == ascending) {
                s[pos] = b;
                s[pos + stride] = a;
            }
        }
    }
    __syncthreads();

    data[base + t] = s[t];
    data[base + t + SORT_THREADS] = s[t + SORT_THREADS];
}

int main() {
    const int segments = 2000;
    const int n = segments * SORT_N;

    Lcg rng(21);
    std::vector<int> data(n);
    for (auto& v : data) v = static_cast<int>(rng.next_u32() % 1000000);

    std::vector<int> ref = data;
    for (int s = 0; s < segments; ++s)
        std::sort(ref.begin() + size_t(s) * SORT_N, ref.begin() + size_t(s + 1) * SORT_N);

    DeviceBuffer<int> d_data(n);
    report_time("bitonic_sort_segments", time_ms([&] {
        d_data.upload(data);
        bitonic_sort_segments<<<segments, SORT_THREADS>>>(d_data.get());
    }));

    std::vector<int> got;
    d_data.download(got);
    bool ok = check_equal(ref.data(), got.data(), n, "sorted segments");
    return finish("bitonic_sort", ok);
}
