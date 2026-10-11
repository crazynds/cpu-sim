// Exclusive prefix sums of 2^29 + 7777 ints (bench/scan.cu).
//   scan_blocks: each 512-element segment scanned on its own (the first kernel
//                of the CUDA scan, whose output the GPU benchmark hashes);
//   scan:        the whole array.
#include <algorithm>

#include "native.hpp"

constexpr int SEG = 512;  // SCAN_ELEMS in samples/scan.cu

int main() {
    const int n = (1 << 29) + 7777;
    std::vector<int> in(n), out(n);
    Lcg rng(9);
    for (auto& v : in) v = int(rng.next_u32() % 17) - 8;

    const int segments = (n + SEG - 1) / SEG;
    native_report("scan_blocks", native_ms([&] {
#pragma omp parallel for schedule(static)
        for (int s = 0; s < segments; ++s) {
            int sum = 0;
            for (int i = s * SEG, end = std::min(n, i + SEG); i < end; ++i) {
                out[i] = sum;
                sum += in[i];
            }
        }
    }));
    native_checksum("scan_blocks", out);

    // Whole array: one pass per thread chunk, then the chunk offsets, then a
    // second pass adding them (with 1 thread it is a single plain pass).
    std::vector<int> chunk_sum;
    native_report("scan", native_ms([&] {
        const int threads = omp_get_max_threads();
        chunk_sum.assign(threads + 1, 0);
#pragma omp parallel num_threads(threads)
        {
            const int t = omp_get_thread_num(), nt = omp_get_num_threads();
            const size_t lo = size_t(n) * t / nt, hi = size_t(n) * (t + 1) / nt;
            int sum = 0;
            for (size_t i = lo; i < hi; ++i) {
                out[i] = sum;
                sum += in[i];
            }
            chunk_sum[t + 1] = sum;
#pragma omp barrier
#pragma omp single
            for (int c = 1; c <= nt; ++c) chunk_sum[c] += chunk_sum[c - 1];
            const int add = chunk_sum[t];
            if (add)
                for (size_t i = lo; i < hi; ++i) out[i] += add;
        }
    }));
    native_checksum("scan", out);
}
