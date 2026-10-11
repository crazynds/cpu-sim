#define main sample_main
#include "../samples/transpose.cu"
#undef main
#include "bench.cuh"

int main() {
    const int rows = 24000, cols = 18000;  // samples/transpose.cu
    auto in = random_floats<float>(size_t(rows) * cols, 3, -1, 1);
    DeviceBuffer<float> d_in(in.size()), d_out(in.size());
    d_in.upload(in);
    dim3 block(TT, BROWS), grid((cols + TT - 1) / TT, (rows + TT - 1) / TT);
    bench_report("transpose_naive", bench_ms([&] { transpose_naive<<<grid, block>>>(d_in.get(), d_out.get(), rows, cols); }));
    bench_checksum("transpose_naive", d_out);
    bench_report("transpose_tiled", bench_ms([&] { transpose_tiled<<<grid, block>>>(d_in.get(), d_out.get(), rows, cols); }));
    bench_checksum("transpose_tiled", d_out);
}
