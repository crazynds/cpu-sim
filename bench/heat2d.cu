#define main sample_main
#include "../samples/heat2d.cu"
#undef main
#include "bench.cuh"

int main() {
    const int w = 8000, h = 6000;  // samples/heat2d.cu (one step; the sample runs 60)
    auto g = random_floats<float>(size_t(w) * h, 5, 0, 1);
    DeviceBuffer<float> a(g.size()), b(g.size());
    a.upload(g);
    dim3 block(BX, BY), blocks((w + BX - 1) / BX, (h + BY - 1) / BY);
    bench_report("heat_step", bench_ms([&] { heat_step<<<blocks, block>>>(a.get(), b.get(), w, h, 0.2f); }));
    bench_checksum("heat_step", b);
}
