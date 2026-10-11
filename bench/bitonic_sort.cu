#define main sample_main
#include "../samples/bitonic_sort.cu"
#undef main
#include "bench.cuh"

int main() {
    const int segments = 240000;  // samples/bitonic_sort.cu
    std::vector<int> data(size_t(segments) * SORT_N);
    Lcg rng(11);
    for (auto& v : data) v = int(rng.next_u32());
    DeviceBuffer<int> d(data.size());
    // Sorting sorted data is a different workload; re-upload outside the timing.
    BenchResult best{1e30, 0};
    for (int i = 0; i < 4; ++i) {
        d.upload(data);
        BenchResult r = bench_ms([&] { bitonic_sort_segments<<<segments, SORT_THREADS>>>(d.get()); }, 1);
        if (i > 0 && r.ms < best.ms) best = r;
    }
    bench_report("bitonic_sort", best);
    bench_checksum("bitonic_sort", d);
}
