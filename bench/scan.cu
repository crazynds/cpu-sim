#define main sample_main
#include "../samples/scan.cu"
#undef main
#include "bench.cuh"

int main() {
    const int n = (1 << 29) + 7777;  // samples/scan.cu
    std::vector<int> in(n);
    Lcg rng(9);
    for (auto& v : in) v = int(rng.next_u32() % 17) - 8;
    DeviceBuffer<int> d_in(n), d_out(n);
    d_in.upload(in);
    int blocks = (n + SCAN_ELEMS - 1) / SCAN_ELEMS;
    DeviceBuffer<int> sums(blocks);
    bench_report("scan_blocks", bench_ms([&] { scan_blocks<<<blocks, SCAN_THREADS>>>(d_in.get(), d_out.get(), sums.get(), n); }));
    bench_checksum("scan_blocks", d_out);
    bench_report("scan (3 levels)", bench_ms([&] { scan_device(d_in.get(), d_out.get(), n); }));
    bench_checksum("scan", d_out);
}
