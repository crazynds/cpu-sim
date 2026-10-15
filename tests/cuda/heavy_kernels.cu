// Kernels whose code once took minutes to compile, kept here with a compile
// time budget (COMPILE_BUDGET in CMakeLists.txt) and checked at every width:
//  - rounds: a loop of 32 prefix sums with barriers inside, unroll pragma
//    and all (HeCBench sc). Unrolled, every copy of the barrier is one more
//    resume point: 8 minutes of register allocation, then a hang at run
//    time. Loops with a barrier stay rolled.
//  - butterflies: 256 floats per thread through butterflies and warp
//    shuffles, with a barrier (HeCBench hadamard). At 128 lanes and more,
//    each live value takes 8 to 32 vector registers: 20 minutes to compile
//    a variant that ran 2-5x slower. No variant of 128 lanes or more is
//    built past 64 live values.
// Inputs are small integers, so every sum is exact in float.
#include <vector>

#include "check.cuh"

constexpr int kBlock = 256, kRounds = 32, kItems = 256, kBlocks = 8;

__global__ void rounds(const int* in, int* out) {
    __shared__ int s[kBlock];
    const int t = threadIdx.x;
    const int v = in[blockIdx.x * kBlock + t];
    int acc = 0;
#pragma unroll
    for (int r = 0; r < kRounds; ++r) {
        s[t] = v + r;
        __syncthreads();
#pragma unroll
        for (int d = 1; d < kBlock; d <<= 1) {
            const int x = t >= d ? s[t - d] : 0;
            __syncthreads();
            s[t] += x;
            __syncthreads();
        }
        acc += s[t];
        __syncthreads();
    }
    out[blockIdx.x * kBlock + t] = acc;
}

__global__ void butterflies(const float* in, float* out) {
    __shared__ float sh[kBlock];
    const int t = threadIdx.x;
    const long base = (long(blockIdx.x) * kBlock + t) * kItems;
    float x[kItems];
#pragma unroll
    for (int i = 0; i < kItems; ++i) x[i] = in[base + i];
#pragma unroll
    for (int s = 1; s < kItems; s <<= 1)
#pragma unroll
        for (int i = 0; i < kItems; ++i)
            if ((i & s) == 0) {
                const float a = x[i], b = x[i + s];
                x[i] = a + b;
                x[i + s] = a - b;
            }
    const int lane = t & 31;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, x[i], m);
            x[i] = (lane & m) ? o - x[i] : x[i] + o;
        }
    sh[t] = x[0];
    __syncthreads();
    x[0] += sh[(t + 32) % kBlock];
#pragma unroll
    for (int i = 0; i < kItems; ++i) out[base + i] = x[i];
}

int main() {
    Checker c("heavy_kernels");

    // rounds: thread t's sum over the rounds of the inclusive prefix sum of
    // (in + r) at t, that is 32 * P(t) + (t + 1) * (0 + 1 + ... + 31).
    {
        const int n = kBlocks * kBlock;
        std::vector<int> in(n), out(n);
        for (int i = 0; i < n; ++i) in[i] = (i * 7) % 11 - 5;
        int *din, *dout;
        CHECK_CUDA(cudaMalloc(&din, n * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&dout, n * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(din, in.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        rounds<<<kBlocks, kBlock>>>(din, dout);
        CHECK_CUDA(cudaMemcpy(out.data(), dout, n * sizeof(int), cudaMemcpyDeviceToHost));
        for (int b = 0; b < kBlocks; ++b) {
            long prefix = 0;
            for (int t = 0; t < kBlock; ++t) {
                prefix += in[b * kBlock + t];
                const long want = kRounds * prefix + long(t + 1) * (kRounds * (kRounds - 1) / 2);
                c.expect(out[b * kBlock + t] == want, "rounds", b * kBlock + t, out[b * kBlock + t], double(want));
            }
        }
        CHECK_CUDA(cudaFree(din));
        CHECK_CUDA(cudaFree(dout));
    }

    // butterflies: the same steps on the host, thread by thread.
    {
        const long n = long(kBlocks) * kBlock * kItems;
        std::vector<float> in(n), out(n), x(n);
        for (long i = 0; i < n; ++i) in[i] = float(int((i * 13) % 9) - 4);
        float *din, *dout;
        CHECK_CUDA(cudaMalloc(&din, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dout, n * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(din, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        butterflies<<<kBlocks, kBlock>>>(din, dout);
        CHECK_CUDA(cudaMemcpy(out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost));
        x = in;
        auto at = [&](long thread, int i) -> float& { return x[thread * kItems + i]; };
        const long threads = long(kBlocks) * kBlock;
        for (long th = 0; th < threads; ++th)
            for (int s = 1; s < kItems; s <<= 1)
                for (int i = 0; i < kItems; ++i)
                    if ((i & s) == 0) {
                        const float a = at(th, i), b = at(th, i + s);
                        at(th, i) = a + b;
                        at(th, i + s) = a - b;
                    }
        for (int m = 1; m < 32; m <<= 1) {
            std::vector<float> prev = x;
            for (long th = 0; th < threads; ++th) {
                const long other = th ^ m;
                for (int i = 0; i < kItems; ++i) {
                    const float mine = prev[th * kItems + i], o = prev[other * kItems + i];
                    at(th, i) = (th & m) ? o - mine : mine + o;
                }
            }
        }
        std::vector<float> first(threads);
        for (long th = 0; th < threads; ++th) first[th] = at(th, 0);
        for (long th = 0; th < threads; ++th) {
            const long b = th / kBlock, t = th % kBlock;
            at(th, 0) = first[th] + first[b * kBlock + (t + 32) % kBlock];
        }
        for (long i = 0; i < n; ++i) c.expect(out[i] == x[i], "butterflies", i, out[i], x[i]);
        CHECK_CUDA(cudaFree(din));
        CHECK_CUDA(cudaFree(dout));
    }
    return c.finish();
}
