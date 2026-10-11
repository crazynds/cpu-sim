// Divergent control flow: the same __host__ __device__ functions run on the
// device (where the lanes of a warp take different paths and the compiler
// linearizes them under masks) and on the host, and the integer results must
// match exactly. Covers nested branches, loops with per-lane trip counts,
// break/continue, early return, switch, recursion-free calls, and uniform
// loops with barriers around divergent regions.
#include <cstdint>
#include <vector>

#include "check.cuh"

__host__ __device__ unsigned mix(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

// Collatz steps with a cap: a loop whose trip count differs per lane.
__host__ __device__ unsigned collatz(unsigned v) {
    unsigned steps = 0;
    while (v != 1 && steps < 500) {
        v = (v & 1) ? 3 * v + 1 : v / 2;
        ++steps;
    }
    return steps;
}

__host__ __device__ unsigned branchy(unsigned v) {
    unsigned acc = 0;
    for (unsigned i = 0; i < (v & 31); ++i) {  // per-lane trip count
        if (i % 3 == 0) continue;
        if (acc > 0x40000000u) break;
        switch ((v >> i) & 3) {
        case 0:
            acc += i * 7;
            break;
        case 1:
            acc ^= mix(i + v);
            break;
        case 2:
            if (acc & 1)
                acc = acc * 3 + 1;
            else
                acc >>= 1;
            break;
        default:
            for (unsigned j = 0; j < (i & 7); ++j) acc += j * j;  // nested varying loop
        }
    }
    return acc;
}

__host__ __device__ unsigned early_exit(unsigned v) {
    if (v % 7 == 0) return 7;
    if (v % 11 == 0) {
        unsigned r = 0;
        for (unsigned k = 0; k < v % 13; ++k) {
            if (k == 5) return r + 1000;
            r += k;
        }
        return r;
    }
    return v % 5 == 0 ? mix(v) : collatz(v % 1000 + 1);
}

__global__ void divergent(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned v = in[i];
    out[3 * i + 0] = collatz(v % 10000 + 1);
    out[3 * i + 1] = branchy(v);
    out[3 * i + 2] = early_exit(v);
}

// Uniform loop with barriers; divergent work between them, including threads
// that skip the work entirely. Each round: s[t] = f(s[t], s[t^1], s[t^5]).
constexpr int kRounds = 6;
__global__ void rounds(const unsigned* in, unsigned* out) {
    __shared__ unsigned s[128];
    const int t = threadIdx.x;
    s[t] = in[blockIdx.x * 128 + t];
    __syncthreads();
    for (int r = 0; r < kRounds; ++r) {
        unsigned a = s[t], b = s[t ^ 1], c = s[t ^ 5], v = a;
        if ((a + r) & 1) {
            v = branchy(b ^ c);
        } else if (a % 3 != 0) {
            for (unsigned k = 0; k < (c & 15); ++k) v = mix(v + k);
        }
        __syncthreads();
        s[t] = v;
        __syncthreads();
    }
    out[blockIdx.x * 128 + t] = s[t];
}

int main() {
    Checker check("control_flow");
    const int n = 20000;
    std::vector<unsigned> in(n);
    for (int i = 0; i < n; ++i) in[i] = mix(i * 2654435761u + 17);
    unsigned *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(unsigned)));
    CHECK_CUDA(cudaMalloc(&d_out, 3 * n * sizeof(unsigned)));
    CHECK_CUDA(cudaMemcpy(d_in, in.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));

    std::vector<unsigned> out(3 * n);
    divergent<<<(n + 159) / 160, 160>>>(d_in, d_out, n);
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, 3 * n * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        unsigned v = in[i];
        check.expect(out[3 * i + 0] == collatz(v % 10000 + 1), "collatz", i, out[3 * i], collatz(v % 10000 + 1));
        check.expect(out[3 * i + 1] == branchy(v), "branchy", i, out[3 * i + 1], branchy(v));
        check.expect(out[3 * i + 2] == early_exit(v), "early_exit", i, out[3 * i + 2], early_exit(v));
    }

    const int blocks = n / 128;
    rounds<<<blocks, 128>>>(d_in, d_out);
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, blocks * 128 * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (int b = 0; b < blocks; ++b) {
        unsigned s[128], next[128];
        std::memcpy(s, &in[b * 128], sizeof s);
        for (int r = 0; r < kRounds; ++r) {
            for (int t = 0; t < 128; ++t) {
                unsigned a = s[t], bb = s[t ^ 1], c = s[t ^ 5], v = a;
                if ((a + r) & 1) {
                    v = branchy(bb ^ c);
                } else if (a % 3 != 0) {
                    for (unsigned k = 0; k < (c & 15); ++k) v = mix(v + k);
                }
                next[t] = v;
            }
            std::memcpy(s, next, sizeof s);
        }
        for (int t = 0; t < 128; ++t) check.expect(out[b * 128 + t] == s[t], "rounds", b * 128 + t, out[b * 128 + t], s[t]);
    }
    cudaFree(d_in), cudaFree(d_out);
    return check.finish();
}
