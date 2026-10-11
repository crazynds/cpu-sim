// Mandelbrot iteration counts, 4096 x 3072 (bench/mandelbrot.cu).
#include "native.hpp"

constexpr int MAX_ITER = 22528;  // samples/mandelbrot.cu

static int mandel_iters(int px, int py, int w, int h) {  // same arithmetic as the kernel
    float cx = -2.0f + 3.0f * static_cast<float>(px) / static_cast<float>(w);
    float cy = -1.2f + 2.4f * static_cast<float>(py) / static_cast<float>(h);
    float x = 0.0f, y = 0.0f;
    int it = 0;
    while (x * x + y * y <= 4.0f && it < MAX_ITER) {
        float xt = x * x - y * y + cx;
        y = 2.0f * x * y + cy;
        x = xt;
        ++it;
    }
    return it;
}

int main() {
    const int w = 4096, h = 3072;
    std::vector<int> out(size_t(w) * h);
    // Rows take very different times (the set is in the middle): dynamic schedule.
    native_report("mandelbrot", native_ms(
                                    [&] {
#pragma omp parallel for schedule(dynamic, 4)
                                        for (int py = 0; py < h; ++py)
                                            for (int px = 0; px < w; ++px)
                                                out[size_t(py) * w + px] = mandel_iters(px, py, w, h);
                                    },
                                    2));
    native_checksum("mandelbrot", out);
}
