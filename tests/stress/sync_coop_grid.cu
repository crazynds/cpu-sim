// Grid-wide synchronization: cooperative_groups::this_grid().sync() in
// kernels started with cudaLaunchCooperativeKernel, sized to be fully
// resident (occupancy x multiprocessors).
#include "stress.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int BS = 256;

// Two-phase reduction then a map that needs the total:
// out[i] = in[i] * total - grid size, with one grid.sync between phases.
__global__ void reduce_then_scale(const int* in, long long* partial, long long* out, int* valid, int n) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    __shared__ long long s[BS];
    long long acc = 0;
    for (unsigned long long i = grid.thread_rank(); i < (unsigned long long)n; i += grid.size()) acc += in[i];
    s[block.thread_rank()] = acc;
    block.sync();
    for (unsigned k = BS / 2; k > 0; k >>= 1) {
        if (block.thread_rank() < k) s[block.thread_rank()] += s[block.thread_rank() + k];
        block.sync();
    }
    if (block.thread_rank() == 0) partial[blockIdx.x] = s[0];
    grid.sync();
    long long total = 0;
    for (unsigned b = 0; b < gridDim.x; ++b) total += partial[b];
    for (unsigned long long i = grid.thread_rank(); i < (unsigned long long)n; i += grid.size())
        out[i] = in[i] * total - (long long)grid.size();
    if (grid.thread_rank() == 0) *valid = grid.is_valid();
}

// Integer smoothing iterated `steps` times over ping-pong buffers, with a
// grid.sync between steps (every step reads neighbors written by other blocks).
__global__ void iterate_stencil(int* a, int* b, int n, int steps) {
    cg::grid_group grid = cg::this_grid();
    for (int s = 0; s < steps; ++s) {
        const int* src = (s & 1) ? b : a;
        int* dst = (s & 1) ? a : b;
        for (unsigned long long i = grid.thread_rank(); i < (unsigned long long)n; i += grid.size()) {
            int l = i > 0 ? src[i - 1] : src[i];
            int r = i + 1 < (unsigned long long)n ? src[i + 1] : src[i];
            dst[i] = (l + 2 * src[i] + r + 1) / 4 + int(s & 1);
        }
        grid.sync();
    }
}

int main() {
    using namespace stress;
    int dev = 0, coop = 0, sms = 0;
    STRESS_CHECK(cudaGetDevice(&dev));
    STRESS_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
    STRESS_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    if (!coop) {
        std::printf("ERROR device does not support cooperative launch\n");
        return 2;
    }

    {
        int per_sm = 0;
        STRESS_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, reduce_then_scale, BS, 0));
        int nb = per_sm * sms;
        const int n = 1 << 20;
        auto r = random_u32(n, 91);
        std::vector<int> in(n);
        for (int i = 0; i < n; ++i) in[i] = int(r[i] % 2001) - 1000;
        Buffer<int> din(in), valid(1);
        Buffer<long long> partial(nb), out(n);
        int nn = n;
        int* pin = din.p;
        long long* ppart = partial.p;
        long long* pout = out.p;
        int* pvalid = valid.p;
        void* args[] = {&pin, &ppart, &pout, &pvalid, &nn};
        double k = time_device([&] {
            STRESS_CHECK(cudaLaunchCooperativeKernel((void*)reduce_then_scale, dim3(nb), dim3(BS), args, 0, 0));
        });
        std::vector<long long> want(n);
        double ref = best_ms([&] {
            long long t = 0;
#pragma omp parallel for reduction(+ : t)
            for (int i = 0; i < n; ++i) t += in[i];
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = in[i] * t - (long long)nb * BS;
        });
        auto got = out.download();
        report("grid_sync_reduce_scale", valid.download()[0] == 1 && equal(got.data(), want.data(), n), k, ref);
    }

    {
        int per_sm = 0;
        STRESS_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, iterate_stencil, BS, 0));
        int nb = per_sm * sms;
        const int n = 1 << 18, steps = 64;
        auto r = random_u32(n, 92);
        std::vector<int> init(n);
        for (int i = 0; i < n; ++i) init[i] = int(r[i] % 100000);
        Buffer<int> a(n), b(n);
        int nn = n, st = steps;
        int* pa = a.p;
        int* pb = b.p;
        void* args[] = {&pa, &pb, &nn, &st};
        double k = time_device([&] {
            a.upload(init);
            STRESS_CHECK(cudaLaunchCooperativeKernel((void*)iterate_stencil, dim3(nb), dim3(BS), args, 0, 0));
        });
        std::vector<int> x(n), y(n);
        double ref = best_ms([&] {
            x = init;
#pragma omp parallel
            for (int s = 0; s < steps; ++s) {
                const int* src = (s & 1) ? y.data() : x.data();
                int* dst = (s & 1) ? x.data() : y.data();
#pragma omp for
                for (int i = 0; i < n; ++i) {
                    int l = i > 0 ? src[i - 1] : src[i], rr = i + 1 < n ? src[i + 1] : src[i];
                    dst[i] = (l + 2 * src[i] + rr + 1) / 4 + (s & 1);
                }
            }
        });
        auto got = (steps % 2 == 0 ? a : b).download();
        const auto& want = steps % 2 == 0 ? x : y;
        report("grid_sync_stencil_steps", equal(got.data(), want.data(), n), k, ref);
    }

    return finish();
}
