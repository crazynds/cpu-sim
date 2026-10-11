// One step of the 5-point heat stencil on 8000 x 6000 (bench/heat2d.cu): the
// border stays fixed, every other cell takes the kernel's update.
#include "native.hpp"

static inline float heat_update(float c, float l, float r, float u, float d, float alpha) {  // same as the kernel
    return c + alpha * (l + r + u + d - 4.0f * c);
}

int main() {
    const int w = 8000, h = 6000;
    const float alpha = 0.2f;
    auto in = random_floats(size_t(w) * h, 5, 0, 1);
    std::vector<float> out(in.size());
    native_report("heat_step", native_ms([&] {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                size_t i = size_t(y) * w + x;
                if (x == 0 || y == 0 || x == w - 1 || y == h - 1)
                    out[i] = in[i];
                else
                    out[i] = heat_update(in[i], in[i - 1], in[i + 1], in[i - w], in[i + w], alpha);
            }
    }));
    native_checksum("heat_step", out);
}
