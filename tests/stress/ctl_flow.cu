// Control flow: deep divergent nesting, switch with fallthrough, goto, loops
// with per-thread trip counts, early returns inside nested loops, do/while
// with divergent exits, a state machine, divergent calls to large
// __noinline__ functions, barriers inside uniform loops around divergent
// work, and a long unrolled kernel. The same __host__ __device__ code runs on
// the host as the reference.
#include "stress.cuh"

using stress::hash32;

__host__ __device__ unsigned nest6(unsigned v) {
    unsigned r = 0;
    if (v & 1) {
        if (v & 2) {
            if (v & 4) {
                if (v & 8) {
                    if (v & 16) r = (v & 32) ? v * 3 : v ^ 0x55;
                    else r = v + 7;
                } else {
                    r = (v & 32) ? v >> 2 : v << 1;
                }
            } else {
                r = v - 9;
            }
        } else if (v & 64) {
            r = v * v;
        } else {
            r = ~v;
        }
    } else {
        r = (v & 128) ? v / 3 : v % 17;
    }
    return r;
}

__host__ __device__ unsigned switch16(unsigned v) {
    unsigned r = 1;
    switch (v & 15) {
        case 0: r += 3;
        case 1: r *= 5; break;
        case 2: r = v; break;
        case 3: r ^= v;
        case 4: r += v >> 3;
        case 5: r *= 7; break;
        case 6: r = 6; break;
        case 7: r = v & 0xff; break;
        case 8: case 9: r = v * 9; break;
        case 10: r = v + 10;
        case 11: r -= 11; break;
        case 12: r = v << 2; break;
        case 13: r = v >> 1; break;
        case 14: r = v ^ 0xabcdef; break;
        default: r = 0xffff;
    }
    return r;
}

__host__ __device__ unsigned with_goto(unsigned v) {
    unsigned acc = 0, i = 0;
again:
    acc += v >> (i & 7);
    if (++i < (v & 15)) goto again;
    if (acc & 1) goto odd;
    return acc * 2;
odd:
    return acc * 3 + 1;
}

// break, continue and early return inside nested loops with per-thread bounds.
__host__ __device__ unsigned loops(unsigned v) {
    unsigned acc = 0;
    for (unsigned i = 0; i < (v & 31); ++i) {
        if (i % 5 == 4) continue;
        for (unsigned j = 0; j < ((v >> 5) & 15); ++j) {
            acc += i * j + (v >> j);
            if (acc > 0x7fffffffu) return acc ^ 0x1234;
            if ((acc & 63) == 7) break;
        }
        if (acc % 1000 == 999) break;
    }
    unsigned k = v & 7;
    do {
        acc = acc * 31 + k;
    } while (k-- > 0 && (acc & 3) != 1);
    return acc;
}

// Half the lanes run ~1000x longer.
__host__ __device__ unsigned skewed(unsigned v) {
    unsigned n = (v & 1) ? 2000 + (v & 255) : 2 + (v & 3);
    unsigned x = v;
    while (n--) x = x * 1664525u + 1013904223u;
    return x;
}

// A tiny state machine reading its input from memory.
__host__ __device__ unsigned machine(const unsigned char* prog, unsigned v) {
    unsigned pc = v % 16, acc = v, steps = 0;
    while (steps++ < 64) {
        unsigned char op = prog[pc];
        switch (op & 3) {
            case 0: acc += op; pc = (pc + 1) & 15; break;
            case 1: acc ^= acc >> 3; pc = (pc + 3) & 15; break;
            case 2: if (acc & 1) pc = (pc + 5) & 15; else pc = (pc + 1) & 15; break;
            default: if ((acc & 7) == 3) return acc; pc = (pc + 7) & 15; break;
        }
    }
    return acc ^ steps;
}

__host__ __device__ __noinline__ unsigned big_a(unsigned v) {
    unsigned x = v;
#pragma unroll
    for (int i = 0; i < 64; ++i) x = (x ^ (x << 7)) * 0x9e3779b1u + i;
    return x;
}
__host__ __device__ __noinline__ unsigned big_b(unsigned v) {
    unsigned x = v;
#pragma unroll
    for (int i = 0; i < 64; ++i) x = (x ^ (x >> 5)) * 0x85ebca6bu - i;
    return x;
}

__host__ __device__ unsigned all_paths(const unsigned char* prog, unsigned v) {
    unsigned r = nest6(v);
    r = r * 3 + switch16(v >> 4);
    r = r * 5 + with_goto(v >> 8);
    r = r * 7 + loops(v);
    r = r * 11 + machine(prog, v);
    r = r * 13 + ((v & 3) == 0 ? big_a(v) : (v & 3) == 1 ? big_b(v) : v);
    return r;
}

__global__ void k_paths(const unsigned* in, const unsigned char* prog, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = all_paths(prog, in[i]);
}

__global__ void k_skewed(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = skewed(in[i]);
}

// Uniform loop with barriers; divergent work between them.
__global__ void k_barrier_loop(const unsigned* in, unsigned* out, int n, int rounds) {
    __shared__ unsigned s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned v = i < n ? in[i] : 0;
    for (int r = 0; r < rounds; ++r) {
        if (v & 1) v = v * 3 + 1;
        else v >>= 1;
        s[threadIdx.x] = v;
        __syncthreads();
        if (threadIdx.x & 1) v += s[(threadIdx.x + 1) % blockDim.x];
        else v ^= s[(threadIdx.x + blockDim.x - 1) % blockDim.x];
        __syncthreads();
    }
    if (i < n) out[i] = v;
}

static void barrier_loop_host(const std::vector<unsigned>& in, std::vector<unsigned>& out, int n, int block, int rounds) {
#pragma omp parallel for
    for (int b = 0; b < (n + block - 1) / block; ++b) {
        std::vector<unsigned> v(block), s(block);
        for (int t = 0; t < block; ++t) v[t] = b * block + t < n ? in[b * block + t] : 0;
        for (int r = 0; r < rounds; ++r) {
            for (int t = 0; t < block; ++t) v[t] = (v[t] & 1) ? v[t] * 3 + 1 : v[t] >> 1;
            s = v;
            for (int t = 0; t < block; ++t)
                v[t] = (t & 1) ? v[t] + s[(t + 1) % block] : v[t] ^ s[(t + block - 1) % block];
        }
        for (int t = 0; t < block; ++t)
            if (b * block + t < n) out[b * block + t] = v[t];
    }
}

// A long kernel: 2048 dependent operations, fully unrolled.
__host__ __device__ __forceinline__ unsigned unrolled(unsigned x) {
#pragma unroll
    for (int k = 2048; k > 0; --k)
        x = (k % 3 == 0) ? x * 0x2545f491u + k : (k % 3 == 1) ? x ^ (x >> (k % 13 + 1)) : x + (x << (k % 7 + 1));
    return x;
}
__global__ void k_long(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = unrolled(in[i]);
}

int main() {
    using namespace stress;
    const int n = 1 << 18;
    auto in = random_u32(n, 7);
    std::vector<unsigned char> prog(16);
    for (int i = 0; i < 16; ++i) prog[i] = (unsigned char)(hash32(i + 99) & 0xff);
    Buffer<unsigned> din(in), dout(n);
    Buffer<unsigned char> dprog(prog);
    std::vector<unsigned> want(n);

    {
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = all_paths(prog.data(), in[i]);
        });
        bool ok = true;
        double k = 0;
        for (int block : {32, 100, 256, 1024}) {
            k = time_device([&] { k_paths<<<(n + block - 1) / block, block>>>(din, dprog, dout, n); });
            auto got = dout.download();
            ok &= equal(got.data(), want.data(), n, "paths");
        }
        report("divergent_paths", ok, k, ref);
    }
    {
        const int m = 1 << 14;
        double ref = best_ms([&] {
#pragma omp parallel for schedule(dynamic, 64)
            for (int i = 0; i < m; ++i) want[i] = skewed(in[i]);
        });
        double k = time_device([&] { k_skewed<<<(m + 127) / 128, 128>>>(din, dout, m); });
        auto got = dout.download();
        report("skewed_trip_counts", equal(got.data(), want.data(), m), k, ref);
    }
    {
        const int rounds = 16;
        bool ok = true;
        double k = 0, ref = 0;
        for (int block : {64, 96, 256}) {
            ref = best_ms([&] { barrier_loop_host(in, want, n, block, rounds); });
            k = time_device([&] { k_barrier_loop<<<(n + block - 1) / block, block>>>(din, dout, n, rounds); });
            auto got = dout.download();
            ok &= equal(got.data(), want.data(), n, "barrier_loop");
        }
        report("barriers_in_uniform_loop", ok, k, ref);
    }
    {
        const int m = 1 << 16;
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < m; ++i) want[i] = unrolled(in[i]);
        });
        double k = time_device([&] { k_long<<<(m + 255) / 256, 256>>>(din, dout, m); });
        auto got = dout.download();
        report("long_unrolled_kernel", equal(got.data(), want.data(), m), k, ref);
    }
    return finish();
}
