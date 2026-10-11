// Warp-level operations against a host model of their CUDA semantics:
// __shfl_sync (and _up/_down/_xor, with widths, 32 and 64-bit values),
// __ballot_sync/__any_sync/__all_sync/__activemask (also in divergent code and
// in a partial last warp), __reduce_*_sync, __match_any_sync and the integer
// bit intrinsics.
#include <cstdint>
#include <vector>

#include "check.cuh"

constexpr int kOut = 16;

__global__ void warp_ops(const int* in, unsigned* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int v = in[t];
    unsigned* o = out + size_t(t) * kOut;
    o[0] = __shfl_sync(0xffffffffu, v, (lane * 7 + 3) % 32);
    o[1] = __shfl_up_sync(0xffffffffu, v, 3);
    o[2] = __shfl_down_sync(0xffffffffu, v, 5);
    o[3] = __shfl_xor_sync(0xffffffffu, v, 6);
    o[4] = __shfl_sync(0xffffffffu, v, 2, 8);
    o[5] = __float_as_uint(__shfl_down_sync(0xffffffffu, v * 0.5f, 1, 16));
    o[6] = unsigned(__double2loint(__shfl_xor_sync(0xffffffffu, v * 0.25, 17)));
    o[7] = __ballot_sync(0xffffffffu, v & 1);
    unsigned b = 0, act = 0, all = 7, any = 7;
    if (v % 3 == 0) {  // divergent: only some lanes of the warp
        act = __activemask();
        b = __ballot_sync(act, v & 2);
        all = __all_sync(act, v > 0);
        any = __any_sync(act, v > 900);
    }
    o[8] = b;
    o[9] = act;
    o[10] = all;
    o[11] = any;
    int s = v;
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    o[12] = unsigned(s);
    o[13] = __reduce_max_sync(0xffffffffu, unsigned(v));
    o[14] = __match_any_sync(0xffffffffu, v % 4);
    o[15] = unsigned(__popc(v) + __clz(v) * 64 + __ffs(v) * 4096) + (__brev(unsigned(v)) >> 20);
}

// A block of 80 threads: the last warp has 16 lanes.
__global__ void partial_warp(const int* in, unsigned* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned act = __activemask();
    out[t * 3 + 0] = act;
    out[t * 3 + 1] = __ballot_sync(act, in[t] & 1);
    out[t * 3 + 2] = __any_sync(act, in[t] > 990);
}

static unsigned brev(unsigned x) {
    unsigned r = 0;
    for (int i = 0; i < 32; ++i) r |= ((x >> i) & 1u) << (31 - i);
    return r;
}

int main() {
    Checker c("warp_ops");
    const int threads = 256, blocks = 12, n = threads * blocks;
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) v[i] = int((uint32_t(i) * 2654435761u >> 7) % 997);
    int* din;
    unsigned* dout;
    CHECK_CUDA(cudaMalloc(&din, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dout, size_t(n) * kOut * sizeof(unsigned)));
    CHECK_CUDA(cudaMemcpy(din, v.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    warp_ops<<<blocks, threads>>>(din, dout);
    CHECK_CUDA(cudaGetLastError());
    std::vector<unsigned> o(size_t(n) * kOut);
    CHECK_CUDA(cudaMemcpy(o.data(), dout, o.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));

    for (int t = 0; t < n; ++t) {
        const int base = t - t % 32, lane = t % 32;
        auto at = [&](int l) { return v[base + l]; };
        unsigned want[kOut];
        want[0] = at((lane * 7 + 3) % 32);
        want[1] = at(lane < 3 ? lane : lane - 3);
        want[2] = at(lane + 5 >= 32 ? lane : lane + 5);
        want[3] = at(lane ^ 6);
        want[4] = at((lane & ~7) + 2);
        float f = at(lane % 16 == 15 ? lane : lane + 1) * 0.5f;
        std::memcpy(&want[5], &f, 4);
        double d = at(lane ^ 17) * 0.25;
        uint64_t dbits;
        std::memcpy(&dbits, &d, 8);
        want[6] = unsigned(dbits);
        unsigned ballot = 0, act = 0, bdiv = 0, sum = 0, mx = 0, match = 0;
        bool all = true, any = false;
        for (int l = 0; l < 32; ++l) {
            ballot |= unsigned(at(l) & 1) << l;
            if (at(l) % 3 == 0) {
                act |= 1u << l;
                bdiv |= unsigned((at(l) & 2) != 0) << l;
                all &= at(l) > 0;
                any |= at(l) > 900;
            }
            sum += unsigned(at(l));
            mx = std::max(mx, unsigned(at(l)));
            match |= unsigned(at(l) % 4 == v[t] % 4) << l;
        }
        want[7] = ballot;
        bool in_div = v[t] % 3 == 0;
        want[8] = in_div ? bdiv : 0;
        want[9] = in_div ? act : 0;
        want[10] = in_div ? unsigned(all) : 7;
        want[11] = in_div ? unsigned(any) : 7;
        want[12] = sum;
        want[13] = mx;
        want[14] = match;
        unsigned x = unsigned(v[t]);
        int clz = x ? __builtin_clz(x) : 32, ffs = __builtin_ffs(int(x));
        want[15] = unsigned(__builtin_popcount(x) + clz * 64 + ffs * 4096) + (brev(x) >> 20);
        static const char* names[kOut] = {"shfl idx", "shfl up", "shfl down", "shfl xor", "shfl width 8",
                                          "shfl float width 16", "shfl double", "ballot", "ballot divergent",
                                          "activemask divergent", "all divergent", "any divergent", "butterfly sum",
                                          "reduce max", "match any", "bit intrinsics"};
        for (int k = 0; k < kOut; ++k) c.expect(o[size_t(t) * kOut + k] == want[k], names[k], t, o[size_t(t) * kOut + k], want[k]);
    }

    // Partial last warp.
    const int pt = 80, pb = 3, pn = pt * pb;
    unsigned* dp;
    CHECK_CUDA(cudaMalloc(&dp, pn * 3 * sizeof(unsigned)));
    partial_warp<<<pb, pt>>>(din, dp);
    CHECK_CUDA(cudaGetLastError());
    std::vector<unsigned> p(pn * 3);
    CHECK_CUDA(cudaMemcpy(p.data(), dp, p.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (int t = 0; t < pn; ++t) {
        int in_block = t % pt, wbase = t - in_block % 32, lanes = std::min(32, pt - in_block / 32 * 32);
        unsigned act = lanes == 32 ? 0xffffffffu : (1u << lanes) - 1, bal = 0, any = 0;
        for (int l = 0; l < lanes; ++l) {
            bal |= unsigned(v[wbase + l] & 1) << l;
            any |= v[wbase + l] > 990;
        }
        c.expect(p[t * 3] == act, "partial warp activemask", t, p[t * 3], act);
        c.expect(p[t * 3 + 1] == bal, "partial warp ballot", t, p[t * 3 + 1], bal);
        c.expect(p[t * 3 + 2] == any, "partial warp any", t, p[t * 3 + 2], any);
    }
    cudaFree(din);
    cudaFree(dout);
    cudaFree(dp);
    return c.finish();
}
