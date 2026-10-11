// memtest: allocates device memory, checks round trips and holds it, so the
// allocation shows up in cpu-smi.
//   memtest [MiB] [seconds]
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda_runtime.h"

#define CHECK(x)                                                                                    \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            return 1;                                                                               \
        }                                                                                           \
    } while (0)

int main(int argc, char** argv) {
    size_t mib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 256;
    int seconds = argc > 2 ? std::atoi(argv[2]) : 0;
    size_t n = mib << 20;

    std::vector<unsigned char> host(n), back(n);
    for (size_t i = 0; i < n; ++i) host[i] = static_cast<unsigned char>(i * 2654435761u >> 24);

    unsigned char *a = nullptr, *b = nullptr;
    CHECK(cudaMalloc(&a, n));
    CHECK(cudaMalloc(&b, n));
    CHECK(cudaMemcpy(a, host.data(), n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b, a, n, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(back.data(), b, n, cudaMemcpyDeviceToHost));
    bool ok = std::memcmp(host.data(), back.data(), n) == 0;

    CHECK(cudaMemset(a, 0x5a, n));
    CHECK(cudaMemcpy(back.data(), a, n, cudaMemcpyDefault));
    for (size_t i = 0; i < n && ok; ++i) ok = back[i] == 0x5a;

    size_t free = 0, total = 0;
    CHECK(cudaMemGetInfo(&free, &total));
    std::printf("memtest: 2 x %zu MiB allocated, round trip %s, %zu / %zu MiB free\n", mib, ok ? "OK" : "FAILED",
                free >> 20, total >> 20);
    if (seconds > 0) {
        std::printf("holding the allocation for %d s (run cpu-smi now)\n", seconds);
        std::fflush(stdout);
        sleep(static_cast<unsigned>(seconds));
    }
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    return ok ? 0 : 1;
}
