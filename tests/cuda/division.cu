// 32-bit division and remainder by a divisor that is the same for every
// thread but only known at run time (a kernel argument): the compiler turns
// it into a multiplication by a precomputed magic number. Checked against the
// host for extreme dividends and divisors, signed and unsigned.
#include <climits>
#include <cstdint>
#include <vector>

#include "check.cuh"

__global__ void divide(const unsigned* un, const int* sn, unsigned* out, int n, unsigned ud, int sd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[4 * i + 0] = un[i] / ud;
    out[4 * i + 1] = un[i] % ud;
    out[4 * i + 2] = unsigned(sn[i] / sd);
    out[4 * i + 3] = unsigned(sn[i] % sd);
}

int main() {
    Checker c("division");
    std::vector<unsigned> un = {0u, 1u, 2u, 3u, 7u, 8u, 100u, 0x7fffffffu, 0x80000000u, 0x80000001u, 0xfffffffeu, 0xffffffffu};
    std::vector<int> sn = {0, 1, -1, 2, -2, 7, -7, 100, -100, INT_MAX, INT_MIN + 1, INT_MIN};
    uint32_t x = 12345;
    while (un.size() < 4096) {
        x = x * 1664525u + 1013904223u;
        un.push_back(x);
        sn.push_back(int(x ^ (x >> 7)));
    }
    const int n = int(un.size());
    unsigned *dun, *dout;
    int* dsn;
    CHECK_CUDA(cudaMalloc(&dun, n * sizeof(unsigned)));
    CHECK_CUDA(cudaMalloc(&dsn, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dout, 4 * n * sizeof(unsigned)));
    CHECK_CUDA(cudaMemcpy(dun, un.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dsn, sn.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    const unsigned uds[] = {1u, 2u, 3u, 5u, 7u, 10u, 16u, 641u, 4096u, 65535u, 0x7fffffffu, 0x80000000u,
                            0x80000001u, 0xfffffffeu, 0xffffffffu};
    const int sds[] = {1, 2, 3, -3, 7, -7, 10, 641, -4096, 65535, INT_MAX, -INT_MAX, INT_MIN, -1, 1};
    std::vector<unsigned> out(4 * n);
    for (size_t k = 0; k < sizeof(uds) / sizeof(uds[0]); ++k) {
        const unsigned ud = uds[k];
        const int sd = sds[k];
        divide<<<(n + 127) / 128, 128>>>(dun, dsn, dout, n, ud, sd);
        CHECK_CUDA(cudaMemcpy(out.data(), dout, out.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            c.expect(out[4 * i] == un[i] / ud, "unsigned /", i, out[4 * i], un[i] / ud);
            c.expect(out[4 * i + 1] == un[i] % ud, "unsigned %", i, out[4 * i + 1], un[i] % ud);
            if (sd == -1 && sn[i] == INT_MIN) continue;  // overflows: undefined in C and CUDA
            c.expect(int(out[4 * i + 2]) == sn[i] / sd, "signed /", i, int(out[4 * i + 2]), sn[i] / sd);
            c.expect(int(out[4 * i + 3]) == sn[i] % sd, "signed %", i, int(out[4 * i + 3]), sn[i] % sd);
        }
    }
    cudaFree(dun);
    cudaFree(dsn);
    cudaFree(dout);
    return c.finish();
}
