#define main sample_main
#include "../samples/mandelbrot.cu"
#undef main
#include "bench.cuh"

int main() {
    const int w = 4096, h = 3072;  // samples/mandelbrot.cu
    DeviceBuffer<int> out(size_t(w) * h);
    dim3 block(32, 8), grid((w + 31) / 32, (h + 7) / 8);
    bench_report("mandelbrot", bench_ms([&] { mandelbrot<<<grid, block>>>(out.get(), w, h); }, 3));
    bench_checksum("mandelbrot", out);
}
