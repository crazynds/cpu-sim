// Large data: one 1 GiB allocation written by a kernel; 10000 small
// allocations written through a device array of pointers and freed; a grid of
// (2^31-1)>>6 blocks of 64 threads over more than 2^31 bytes, indexed with
// size_t in a grid-stride loop, then gathered from at random; gathers and
// scatters at random positions of a 3 GiB allocation (beyond 2 GiB, and back
// from its end pointer) in the same kernel as gathers from a small one.
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

__global__ void gather_two(const unsigned* big, const size_t* bidx, const unsigned* small, const unsigned* sidx,
                           unsigned* out, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) out[i] = big[bidx[i]] ^ small[sidx[i]];
}

__global__ void gather_back(const unsigned* end, const size_t* back, unsigned* out, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) out[i] = end[-1 - (long long)back[i]];
}

__global__ void scatter_big(unsigned* big, const size_t* bidx, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) big[bidx[i]] = ~pattern(bidx[i]);
}

__global__ void gather_bytes(const unsigned char* p, const size_t* idx, unsigned char* out, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) out[i] = p[idx[i]];
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
        report("huge_grid_size_t_index", ok, k, -1);

        // Bytes gathered from random positions, most beyond 2 GiB.
        const int m = 1 << 20;
        std::vector<size_t> hidx(m);
        auto rr = random_u32(m, 7);
        for (int i = 0; i < m; ++i) hidx[i] = (size_t(rr[i]) * 7 + (size_t(i) << 13)) % n;
        size_t* didx;
        unsigned char* dout;
        STRESS_CHECK(cudaMalloc(&didx, m * sizeof(size_t)));
        STRESS_CHECK(cudaMalloc(&dout, m));
        STRESS_CHECK(cudaMemcpy(didx, hidx.data(), m * sizeof(size_t), cudaMemcpyHostToDevice));
        double kg = time_device([&] { gather_bytes<<<m / 256, 256>>>(p, didx, dout, m); });
        std::vector<unsigned char> hout(m);
        STRESS_CHECK(cudaMemcpy(hout.data(), dout, m, cudaMemcpyDeviceToHost));
        bool okg = true;
        for (int i = 0; i < m && okg; ++i)
            if (hout[i] != byte_pattern(hidx[i])) {
                std::printf("  byte [%zu] got %02x want %02x\n", hidx[i], hout[i], byte_pattern(hidx[i]));
                okg = false;
            }
        STRESS_CHECK(cudaFree(didx));
        STRESS_CHECK(cudaFree(dout));
        STRESS_CHECK(cudaFree(p));
        report("gather_bytes_past_2gib", okg, kg, -1);
    }

    {  // 3 GiB of words: gathers, scatters and indexing back from the end
        const size_t n = (size_t(3) << 30) / sizeof(unsigned), ns = 1 << 18;
        const int m = 1 << 20;
        unsigned *big, *small, *out;
        size_t *bidx, *back;
        unsigned* sidx;
        STRESS_CHECK(cudaMalloc(&big, n * sizeof(unsigned)));
        STRESS_CHECK(cudaMalloc(&small, ns * sizeof(unsigned)));
        STRESS_CHECK(cudaMalloc(&out, m * sizeof(unsigned)));
        STRESS_CHECK(cudaMalloc(&bidx, m * sizeof(size_t)));
        STRESS_CHECK(cudaMalloc(&back, m * sizeof(size_t)));
        STRESS_CHECK(cudaMalloc(&sidx, m * sizeof(unsigned)));
        fill_words<<<26 * 64, 256>>>(big, n);
        std::vector<unsigned> hs(ns);
        for (size_t i = 0; i < ns; ++i) hs[i] = pattern(i + 12345);
        STRESS_CHECK(cudaMemcpy(small, hs.data(), ns * sizeof(unsigned), cudaMemcpyHostToDevice));
        std::vector<size_t> hb(m), hk(m);
        std::vector<unsigned> hsi(m);
        auto r1 = random_u32(m, 11), r2 = random_u32(m, 12), r3 = random_u32(m, 13);
        for (int i = 0; i < m; ++i) {
            hb[i] = (size_t(r1[i]) * 3 + (size_t(i) << 9)) % n;  // up to 3 GiB
            hk[i] = (size_t(r2[i]) * 5 + (size_t(i) << 10)) % n;  // back from the end
            hsi[i] = r3[i] % ns;
        }
        STRESS_CHECK(cudaMemcpy(bidx, hb.data(), m * sizeof(size_t), cudaMemcpyHostToDevice));
        STRESS_CHECK(cudaMemcpy(back, hk.data(), m * sizeof(size_t), cudaMemcpyHostToDevice));
        STRESS_CHECK(cudaMemcpy(sidx, hsi.data(), m * sizeof(unsigned), cudaMemcpyHostToDevice));
        std::vector<unsigned> h(m);
        bool ok = true;
        double k = time_device([&] { gather_two<<<m / 256, 256>>>(big, bidx, small, sidx, out, m); });
        STRESS_CHECK(cudaMemcpy(h.data(), out, m * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (int i = 0; i < m && ok; ++i)
            if (h[i] != (pattern(hb[i]) ^ hs[hsi[i]])) {
                std::printf("  gather [%zu] got %08x want %08x\n", hb[i], h[i], pattern(hb[i]) ^ hs[hsi[i]]);
                ok = false;
            }
        k += time_device([&] { gather_back<<<m / 256, 256>>>(big + n, back, out, m); });
        STRESS_CHECK(cudaMemcpy(h.data(), out, m * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (int i = 0; i < m && ok; ++i)
            if (h[i] != pattern(n - 1 - hk[i])) {
                std::printf("  back [%zu] got %08x want %08x\n", n - 1 - hk[i], h[i], pattern(n - 1 - hk[i]));
                ok = false;
            }
        k += time_device([&] { scatter_big<<<m / 256, 256>>>(big, bidx, m); }, 1);
        for (int i = 0; i < m && ok; i += 61) {
            unsigned v = 0;
            STRESS_CHECK(cudaMemcpy(&v, big + hb[i], sizeof(v), cudaMemcpyDeviceToHost));
            if (v != ~pattern(hb[i])) {
                std::printf("  scatter [%zu] got %08x want %08x\n", hb[i], v, ~pattern(hb[i]));
                ok = false;
            }
        }
        for (void* q : {(void*)big, (void*)small, (void*)out, (void*)bidx, (void*)back, (void*)sidx})
            STRESS_CHECK(cudaFree(q));
        report("gather_scatter_3gib", ok, k, -1);
    }

    return finish();
}
