// Large data: one 1 GiB allocation written by a kernel; 10000 small
// allocations written through a device array of pointers and freed; a grid of
// (2^31-1)>>6 blocks of 64 threads over more than 2^31 bytes, indexed with
// size_t in a grid-stride loop.
#include "stress.cuh"

__host__ __device__ inline unsigned pattern(size_t i) {
    return unsigned(i) * 2654435761u ^ unsigned(i >> 32) * 0x9e3779b9u;
}
__host__ __device__ inline unsigned char byte_pattern(size_t i) {
    return (unsigned char)((unsigned(i) * 2654435761u) >> 24) ^ (unsigned char)(i >> 31);
}

__global__ void fill_words(unsigned* p, size_t n) {
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) p[i] = pattern(i);
}

__global__ void fill_bytes(unsigned char* p, size_t n) {
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) p[i] = byte_pattern(i);
}

__global__ void write_many(int** ptrs, int count, int words) {
    // One block per allocation.
    for (int a = blockIdx.x; a < count; a += gridDim.x) {
        int* p = ptrs[a];
        for (int w = threadIdx.x; w < words; w += blockDim.x) p[w] = a * words + w;
    }
}

int main() {
    using namespace stress;

    {  // 1 GiB in one allocation
        const size_t bytes = size_t(1) << 30, n = bytes / sizeof(unsigned);
        unsigned* p;
        STRESS_CHECK(cudaMalloc(&p, bytes));
        double k = time_device([&] { fill_words<<<26 * 64, 256>>>(p, n); }, 1);
        bool ok = true;
        std::vector<size_t> idx = {0, 1, n / 2, n - 1, n - 2};
        auto r = random_u32(4000, 1);
        for (unsigned x : r) idx.push_back(size_t(x) % n);
        for (size_t i = 0; i < 64; ++i) idx.push_back(n - 1 - i * 4093);
        for (size_t i : idx) {
            unsigned v = 0;
            STRESS_CHECK(cudaMemcpy(&v, p + i, sizeof(v), cudaMemcpyDeviceToHost));
            if (v != pattern(i)) {
                std::printf("  [%zu] got %08x want %08x\n", i, v, pattern(i));
                ok = false;
                break;
            }
        }
        // And a contiguous 16 MiB window at the end.
        const size_t win = 4 << 20;
        std::vector<unsigned> tail(win);
        STRESS_CHECK(cudaMemcpy(tail.data(), p + n - win, win * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < win && ok; ++i) ok = tail[i] == pattern(n - win + i);
        STRESS_CHECK(cudaFree(p));
        report("alloc_1gib_kernel_fill", ok, k, -1);
    }

    {  // 10000 x 4 KiB allocations
        const int count = 10000, words = 1024;
        std::vector<int*> ptrs(count, nullptr);
        double t0 = now_ms();
        for (int a = 0; a < count; ++a) STRESS_CHECK(cudaMalloc(&ptrs[a], words * sizeof(int)));
        double t_alloc = now_ms() - t0;
        int** dptrs;
        STRESS_CHECK(cudaMalloc(&dptrs, count * sizeof(int*)));
        STRESS_CHECK(cudaMemcpy(dptrs, ptrs.data(), count * sizeof(int*), cudaMemcpyHostToDevice));
        double k = time_device([&] { write_many<<<2048, 256>>>(dptrs, count, words); });
        bool ok = true;
        // Distinct, non-overlapping allocations.
        std::vector<int*> sorted(ptrs);
        std::sort(sorted.begin(), sorted.end());
        for (int a = 1; a < count && ok; ++a) ok = sorted[a] >= sorted[a - 1] + words;
        std::vector<int> h(words);
        for (int a = 0; a < count && ok; a += 37) {
            STRESS_CHECK(cudaMemcpy(h.data(), ptrs[a], words * sizeof(int), cudaMemcpyDeviceToHost));
            for (int w = 0; w < words && ok; ++w) ok = h[w] == a * words + w;
        }
        t0 = now_ms();
        for (int a = 0; a < count; ++a) STRESS_CHECK(cudaFree(ptrs[a]));
        double t_free = now_ms() - t0;
        STRESS_CHECK(cudaFree(dptrs));
        std::printf("  alloc %.2f ms, free %.2f ms\n", t_alloc, t_free);
        report("many_small_allocs", ok, k + t_alloc + t_free, -1);
    }

    {  // (2^31-1)>>6 blocks of 64 threads over 2^31 + 1 MiB bytes
        const size_t n = (size_t(1) << 31) + (1 << 20);
        unsigned char* p;
        STRESS_CHECK(cudaMalloc(&p, n));
        const unsigned blocks = 2147483647u >> 6;
        double k = time_device([&] { fill_bytes<<<blocks, 64>>>(p, n); }, 1);
        bool ok = true;
        std::vector<size_t> idx = {0, 1, 2147483583, 2147483584, 2147483647, 2147483648ull, 2147483649ull, n - 1};
        auto r = random_u32(3000, 2);
        for (unsigned x : r) idx.push_back((size_t(x) * 3) % n);
        for (size_t i : idx) {
            unsigned char v = 0;
            STRESS_CHECK(cudaMemcpy(&v, p + i, 1, cudaMemcpyDeviceToHost));
            if (v != byte_pattern(i)) {
                std::printf("  [%zu] got %02x want %02x\n", i, v, byte_pattern(i));
                ok = false;
                break;
            }
        }
        // The whole region above 2^31 - 64 (the part reached by the second grid-stride step).
        const size_t from = (size_t(blocks) * 64) - 4096;
        std::vector<unsigned char> tail(n - from);
        STRESS_CHECK(cudaMemcpy(tail.data(), p + from, n - from, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < tail.size() && ok; ++i)
            if (tail[i] != byte_pattern(from + i)) {
                std::printf("  [%zu] got %02x want %02x\n", from + i, tail[i], byte_pattern(from + i));
                ok = false;
            }
        STRESS_CHECK(cudaFree(p));
        report("huge_grid_size_t_index", ok, k, -1);
    }

    return finish();
}
