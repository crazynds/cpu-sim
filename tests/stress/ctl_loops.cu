// Divergent loops: per-thread trip counts with break and continue, a while
// loop where half the lanes run 1000x longer, return from inside nested
// loops, do/while with divergent exits, Collatz, for(;;) with break,
// downward unsigned loops, and a __syncthreads() inside a uniform loop with
// divergent work around it.
#include "stress.cuh"

__host__ __device__ inline unsigned step(unsigned h, unsigned k) {
    h ^= k + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h * 0x01000193u;
}

// Trip count from the data; continue on some iterations, break on others.
__host__ __device__ inline unsigned trip_loop(unsigned v) {
    unsigned trips = v % 200, h = v;
    for (unsigned k = 0; k < trips; ++k) {
        if ((h & 7) == 3) {
            h += k;
            continue;
        }
        h = step(h, k);
        if ((h & 0xfff) == 0x123) break;
        if (k > 150 && (h & 1)) break;
    }
    return h;
}
__global__ void k_trip(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = trip_loop(in[i]);
}

// Half the lanes (odd global index) run 1000x more iterations.
__host__ __device__ inline unsigned long_short(unsigned v, int i) {
    int iters = (i & 1) ? 20000 : 20;
    unsigned h = v;
    int k = 0;
    while (k < iters) {
        h = step(h, unsigned(k));
        ++k;
    }
    return h;
}
__global__ void k_long_short(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = long_short(in[i], i);
}

// Search a per-thread 16x16 window: return from inside the double loop.
__host__ __device__ inline int find_first(const unsigned char* grid, int w, int x0, int y0, unsigned char target) {
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            unsigned char c = grid[(y0 + y) * w + x0 + x];
            if (c == target) return y * 16 + x;
            if (c == 0) break;  // skip the rest of the row
        }
    }
    return -1;
}
__global__ void k_find(const unsigned char* grid, int w, const unsigned char* targets, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = find_first(grid, w, (i * 7) % (w - 16), (i * 13) % 200, targets[i]);
}

// do/while: exit conditions differ per lane.
__host__ __device__ inline int do_while(unsigned v) {
    int count = 0;
    unsigned x = v | 1;
    do {
        x = x * 1103515245u + 12345u;
        ++count;
        if ((x >> 28) == 7) break;
    } while ((x & 0x3f) != 0 && count < 500);
    return count * 65536 + int(x >> 16);
}
__global__ void k_dowhile(const unsigned* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = do_while(in[i]);
}

// Collatz step counts (64-bit), the classic data-dependent while loop.
__host__ __device__ inline int collatz(unsigned long long x) {
    int steps = 0;
    while (x != 1) {
        x = (x & 1) ? 3 * x + 1 : x >> 1;
        ++steps;
    }
    return steps;
}
__global__ void k_collatz(int* out, int n, unsigned long long base) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = collatz(base + i);
}

// for(;;) with break, downward unsigned loop, triangular nested loops.
__host__ __device__ inline unsigned misc_loops(unsigned v) {
    unsigned h = 0, k = 0;
    for (;;) {
        h = step(h, v + k);
        if (++k > (v & 31) || (h & 0xff) == 0) break;
    }
    for (unsigned j = v % 37; j-- > 0;) h += j * j;  // runs j = n-1 .. 0
    int tri = 0;
    for (int a = 0; a < int(v % 23); ++a)
        for (int b = 0; b <= a; ++b)
            if ((a ^ b) & 1) tri += a - b;
            else tri -= 1;
    return h ^ unsigned(tri);
}
__global__ void k_misc(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = misc_loops(in[i]);
}

// Uniform loop with barriers; divergent work between them. Each iteration a
// thread mixes its value with its neighbour's (read after the barrier).
constexpr int TB = 256;
__host__ __device__ inline unsigned divergent_work(unsigned x, int t, int it) {
    int extra = (t % 5 == 0) ? 40 : (t & 1) ? 3 : 0;
    for (int k = 0; k < extra; ++k) x = step(x, unsigned(k + it));
    return x;
}
__global__ void k_barrier_loop(const unsigned* in, unsigned* out, int iters) {
    __shared__ unsigned buf[TB];
    int t = threadIdx.x;
    unsigned x = in[blockIdx.x * TB + t];
    for (int it = 0; it < iters; ++it) {
        x = divergent_work(x, t, it);
        if (t & 2) x += 1;  // divergent, before the barrier
        buf[t] = x;
        __syncthreads();
        unsigned nb = buf[(t + 1 + it) % TB];
        __syncthreads();
        x = (t % 3 == 0) ? x ^ nb : x + nb;  // divergent, after
    }
    out[blockIdx.x * TB + t] = x;
}
static void barrier_ref(const unsigned* in, unsigned* out, int iters) {
    unsigned x[TB], buf[TB];
    for (int t = 0; t < TB; ++t) x[t] = in[t];
    for (int it = 0; it < iters; ++it) {
        for (int t = 0; t < TB; ++t) {
            x[t] = divergent_work(x[t], t, it);
            if (t & 2) x[t] += 1;
            buf[t] = x[t];
        }
        for (int t = 0; t < TB; ++t) {
            unsigned nb = buf[(t + 1 + it) % TB];
            x[t] = (t % 3 == 0) ? x[t] ^ nb : x[t] + nb;
        }
    }
    for (int t = 0; t < TB; ++t) out[t] = x[t];
}

int main() {
    using namespace stress;
    const int n = 1 << 18, B = 256, G = (n + B - 1) / B;
    auto r = random_u32(n, 1);
    Buffer<unsigned> din(r);

    auto check_u = [&](const char* name, int m, auto f, auto launch) {
        std::vector<unsigned> want(m);
        double ref = best_ms([&] {
#pragma omp parallel for schedule(dynamic, 256)
            for (int i = 0; i < m; ++i) want[i] = f(i);
        });
        Buffer<unsigned> dout(m);
        double k = time_device([&] { launch(dout.p); });
        report(name, equal(dout.download().data(), want.data(), m), k, ref);
    };
    auto check_i = [&](const char* name, int m, auto f, auto launch) {
        std::vector<int> want(m);
        double ref = best_ms([&] {
#pragma omp parallel for schedule(dynamic, 256)
            for (int i = 0; i < m; ++i) want[i] = f(i);
        });
        Buffer<int> dout(m);
        double k = time_device([&] { launch(dout.p); });
        report(name, equal(dout.download().data(), want.data(), m), k, ref);
    };

    check_u("trip_count_break_continue", n, [&](int i) { return trip_loop(r[i]); }, [&](unsigned* o) { k_trip<<<G, B>>>(din, o, n); });
    {
        const int m = 1 << 14;
        check_u("while_half_lanes_1000x", m, [&](int i) { return long_short(r[i], i); },
                [&](unsigned* o) { k_long_short<<<m / B, B>>>(din, o, m); });
    }
    {
        const int w = 512, h = 216;
        std::vector<unsigned char> grid(size_t(w) * h);
        auto g = random_u32(grid.size(), 2);
        for (size_t k = 0; k < grid.size(); ++k) grid[k] = (unsigned char)(g[k] % 61);  // some zeros, some targets
        std::vector<unsigned char> targets(n);
        for (int i = 0; i < n; ++i) targets[i] = (unsigned char)(1 + r[i] % 70);  // some never found
        Buffer<unsigned char> dg(grid), dtg(targets);
        check_i("early_return_nested_loops", n, [&](int i) { return find_first(grid.data(), w, (i * 7) % (w - 16), (i * 13) % 200, targets[i]); },
                [&](int* o) { k_find<<<G, B>>>(dg, w, dtg, o, n); });
    }
    check_i("do_while_divergent_exit", n, [&](int i) { return do_while(r[i]); }, [&](int* o) { k_dowhile<<<G, B>>>(din, o, n); });
    {
        const int m = 1 << 16;
        const unsigned long long base = 1000000007ull;
        check_i("collatz_while", m, [&](int i) { return collatz(base + i); }, [&](int* o) { k_collatz<<<m / B, B>>>(o, m, base); });
    }
    check_u("for_ever_downward_triangular", n, [&](int i) { return misc_loops(r[i]); }, [&](unsigned* o) { k_misc<<<G, B>>>(din, o, n); });
    {
        const int blocks = 256, iters = 32, m = blocks * TB;
        std::vector<unsigned> want(m);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < blocks; ++b) barrier_ref(r.data() + b * TB, want.data() + b * TB, iters);
        });
        Buffer<unsigned> dout(m);
        double k = time_device([&] { k_barrier_loop<<<blocks, TB>>>(din, dout, iters); });
        report("barrier_in_uniform_loop", equal(dout.download().data(), want.data(), m), k, ref);
    }
    return finish();
}
