// Mandelbrot set escape-time iteration counts.
//
// No shared memory: this is the divergent-loop stress test. Neighbouring
// pixels escape after very different iteration counts, so a warp keeps
// looping while any lane is still active and the active mask shrinks lane by
// lane. Built with -fmad=false so the GPU result is bit-exact with the CPU.
#include "common.cuh"

constexpr int MAX_ITER = 22528;

__host__ __device__ inline int mandel_iters(int px, int py, int w, int h) {
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

__global__ void mandelbrot(int* out, int w, int h) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px < w && py < h) out[py * w + px] = mandel_iters(px, py, w, h);
}

int main() {
    const int w = SAMPLE_SIZE(4096, 1024), h = SAMPLE_SIZE(3072, 770);

    std::vector<int> ref(size_t(w) * h);
#pragma omp parallel for schedule(dynamic, 8)
    for (int py = 0; py < h; ++py)
        for (int px = 0; px < w; ++px) ref[size_t(py) * w + px] = mandel_iters(px, py, w, h);

    DeviceBuffer<int> d_out(ref.size());
    dim3 block(32, 8);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    report_time("mandelbrot", time_ms([&] { mandelbrot<<<grid, block>>>(d_out.get(), w, h); }));

    std::vector<int> got;
    d_out.download(got);
    bool ok = check_equal(ref.data(), got.data(), got.size(), "iteration counts");
    return finish("mandelbrot", ok);
}
