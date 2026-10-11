// out = transpose(in), in is 24000 x 18000 (bench/transpose.cu).
#include "native.hpp"

int main() {
    const int rows = 24000, cols = 18000;
    auto in = random_floats(size_t(rows) * cols, 3, -1, 1);
    std::vector<float> out(in.size());

    // Direct: read rows, write columns (strided).
    native_report("transpose_direct", native_ms([&] {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < cols; ++x) out[size_t(x) * rows + y] = in[size_t(y) * cols + x];
    }));
    native_checksum("transpose_naive", out);

    // Blocked: 32 x 32 tiles, so both sides stay within a few cache lines.
    constexpr int T = 32;
    native_report("transpose_blocked", native_ms([&] {
#pragma omp parallel for collapse(2) schedule(static)
        for (int y0 = 0; y0 < rows; y0 += T)
            for (int x0 = 0; x0 < cols; x0 += T)
                for (int x = x0; x < x0 + T && x < cols; ++x)
                    for (int y = y0; y < y0 + T && y < rows; ++y) out[size_t(x) * rows + y] = in[size_t(y) * cols + x];
    }));
    native_checksum("transpose_tiled", out);
}
