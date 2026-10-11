#pragma once
// The GEMM the tensor core tests run (tests/cuda/wmma.cu, wmma_shapes.cu): a
// block of 4 warps stages tiles of A and B in shared memory (barriers), every
// warp keeps two accumulator fragments (an array of fragments), A and B are
// read row- or column-major from memory with a padded leading dimension, C is
// loaded and D stored in either layout, and D = alpha * A * B + beta * C is
// done with element-wise operations on the fragments. Only even warps
// multiply in the last step: odd warps' fragments must stay as they were.
//
// Inputs are small integers, so every product and sum is exact in the
// accumulator type and the result must match the host exactly. When the
// device's compute capability is below what a type needs, the launch must
// fail with cudaErrorNoKernelImageForDevice instead.
#include <mma.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "check.cuh"

using namespace nvcuda;

constexpr int kWarps = 4;               // per block, each with a TM x 2TN strip of D
constexpr int M = 128, N = 64, K = 96;  // D is M x N
constexpr int kPad = 8;                 // leading dimensions are padded
constexpr int LDA_ROW = K + kPad, LDA_COL = M + kPad, LDB_ROW = N + kPad, LDB_COL = K + kPad;
constexpr int LDC = (M > N ? M : N) + kPad;  // C and D in either layout

// TF: the fragments' element type (precision::tf32 for tf32), TS: what memory
// and the fragments hold, TC: the accumulator, TM x TN x TK: the shape.
template <typename TF, typename TS, typename TC, int TM, int TN, int TK, typename LA, typename LB>
__global__ void gemm(const TS* a, const TS* b, const TC* c, TC* d, TC alpha, TC beta, int c_col_major, int d_col_major) {
    __shared__ TS sa[kWarps][TM * TK], sb[2][TK * TN];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int tile_m = blockIdx.x * kWarps + warp;  // TM-row band of D
    const int tile_n = blockIdx.y * 2;               // two TN-column tiles
    constexpr bool a_row = __is_same(LA, wmma::row_major), b_row = __is_same(LB, wmma::row_major);

    wmma::fragment<wmma::matrix_a, TM, TN, TK, TF, LA> fa;
    wmma::fragment<wmma::matrix_b, TM, TN, TK, TF, LB> fb[2];
    wmma::fragment<wmma::accumulator, TM, TN, TK, TC> acc[2];
    wmma::fill_fragment(acc[0], 0);
    wmma::fill_fragment(acc[1], 0);
    for (int k0 = 0; k0 < K; k0 += TK) {
        // Stage this warp's A tile and the block's two B tiles, in the
        // fragments' memory layout, with the tiles' own leading dimension.
        for (int e = lane; e < TM * TK; e += 32) {
            int r = e / TK, q = e % TK;  // row (m) and column (k) of the A tile
            int ar = tile_m * TM + r, ak = k0 + q;
            sa[warp][a_row ? r * TK + q : q * TM + r] = a_row ? a[ar * LDA_ROW + ak] : a[ak * LDA_COL + ar];
        }
        if (warp < 2)
            for (int e = lane; e < TK * TN; e += 32) {
                int r = e / TN, q = e % TN;  // row (k) and column (n) of the B tile
                int bk = k0 + r, bn = (tile_n + warp) * TN + q;
                sb[warp][b_row ? r * TN + q : q * TK + r] = b_row ? b[bk * LDB_ROW + bn] : b[bn * LDB_COL + bk];
            }
        __syncthreads();
        wmma::load_matrix_sync(fa, sa[warp], a_row ? TK : TM);
        wmma::load_matrix_sync(fb[0], sb[0], b_row ? TN : TK);
        wmma::load_matrix_sync(fb[1], sb[1], b_row ? TN : TK);
        if constexpr (__is_same(TF, wmma::precision::tf32)) {  // as CUDA programs do
            for (int i = 0; i < fa.num_elements; ++i) fa.x[i] = wmma::__float_to_tf32(fa.x[i]);
            for (int t = 0; t < 2; ++t)
                for (int i = 0; i < fb[t].num_elements; ++i) fb[t].x[i] = wmma::__float_to_tf32(fb[t].x[i]);
        }
        // The last step only in even warps.
        if (k0 + TK < K || warp % 2 == 0) {
            wmma::mma_sync(acc[0], fa, fb[0], acc[0]);
            wmma::mma_sync(acc[1], fa, fb[1], acc[1]);
        }
        __syncthreads();
    }
    for (int t = 0; t < 2; ++t) {
        wmma::fragment<wmma::accumulator, TM, TN, TK, TC> fc;
        const TC* cp = c + (c_col_major ? (tile_n + t) * TN * LDC + tile_m * TM : tile_m * TM * LDC + (tile_n + t) * TN);
        wmma::load_matrix_sync(fc, cp, LDC, c_col_major ? wmma::mem_col_major : wmma::mem_row_major);
        for (int i = 0; i < acc[t].num_elements; ++i) acc[t].x[i] = alpha * acc[t].x[i] + beta * fc.x[i];
        TC* dp = d + (d_col_major ? (tile_n + t) * TN * LDC + tile_m * TM : tile_m * TM * LDC + (tile_n + t) * TN);
        wmma::store_matrix_sync(dp, acc[t], LDC, d_col_major ? wmma::mem_col_major : wmma::mem_row_major);
    }
}

template <typename TF, typename TS, typename TC, int TM, int TN, int TK, bool kAllLayouts>
static void run(Checker& check, const char* name, int need_sm, int cc, int lo, int hi) {
    std::vector<TS> a(size_t(K + kPad) * (M + kPad) + 64), b(size_t(K + kPad) * (N + kPad) + 64);
    std::vector<TC> c(size_t(LDC) * LDC), d(c.size());
    std::vector<double> av(M * K), bv(K * N), cv(M * N);  // logical values: av[m*K+k], bv[k*N+n], cv[m*N+n]
    uint32_t s = 12345u + need_sm + TK;
    auto next = [&](int l, int h) {
        s = s * 1664525u + 1013904223u;
        return l + int((s >> 8) % uint32_t(h - l + 1));
    };
    // Every layout combination for one type (the loads and stores are the
    // same code for all), A row-major with B column-major for the others.
    for (int layout = kAllLayouts ? 0 : 1; layout < (kAllLayouts ? 4 : 2); ++layout) {
        const bool a_row = layout & 1, b_row = layout & 2, c_col = layout == 1 || layout == 2, d_col = layout >= 2;
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) {
                int v = next(lo, hi);
                av[m * K + k] = v;
                a[a_row ? m * LDA_ROW + k : k * LDA_COL + m] = TS(v);
            }
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n) {
                int v = next(lo, hi);
                bv[k * N + n] = v;
                b[b_row ? k * LDB_ROW + n : n * LDB_COL + k] = TS(v);
            }
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                int v = next(-20, 20);
                cv[m * N + n] = v;
                c[c_col ? n * LDC + m : m * LDC + n] = TC(v);
            }
        TS *d_a, *d_b;
        TC *d_c, *d_d;
        CHECK_CUDA(cudaMalloc(&d_a, a.size() * sizeof(TS)));
        CHECK_CUDA(cudaMalloc(&d_b, b.size() * sizeof(TS)));
        CHECK_CUDA(cudaMalloc(&d_c, c.size() * sizeof(TC)));
        CHECK_CUDA(cudaMalloc(&d_d, d.size() * sizeof(TC)));
        CHECK_CUDA(cudaMemcpy(d_a, a.data(), a.size() * sizeof(TS), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, b.data(), b.size() * sizeof(TS), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_c, c.data(), c.size() * sizeof(TC), cudaMemcpyHostToDevice));
        const dim3 grid(M / TM / kWarps, N / (2 * TN)), block(32 * kWarps);
        const TC alpha = TC(2), beta = TC(-1);
        using RM = wmma::row_major;
        using CM = wmma::col_major;
        if constexpr (kAllLayouts) {
            if (a_row && b_row) gemm<TF, TS, TC, TM, TN, TK, RM, RM><<<grid, block>>>(d_a, d_b, d_c, d_d, alpha, beta, c_col, d_col);
            else if (a_row) gemm<TF, TS, TC, TM, TN, TK, RM, CM><<<grid, block>>>(d_a, d_b, d_c, d_d, alpha, beta, c_col, d_col);
            else if (b_row) gemm<TF, TS, TC, TM, TN, TK, CM, RM><<<grid, block>>>(d_a, d_b, d_c, d_d, alpha, beta, c_col, d_col);
            else gemm<TF, TS, TC, TM, TN, TK, CM, CM><<<grid, block>>>(d_a, d_b, d_c, d_d, alpha, beta, c_col, d_col);
        } else {
            gemm<TF, TS, TC, TM, TN, TK, RM, CM><<<grid, block>>>(d_a, d_b, d_c, d_d, alpha, beta, c_col, d_col);
        }
        cudaError_t launch = cudaGetLastError();
        char what[96];
        std::snprintf(what, sizeof what, "%s layout %d: launch on cc %d (needs %d)", name, layout, cc, need_sm);
        if (cc < need_sm) {
            check.expect(launch == cudaErrorNoKernelImageForDevice, what, launch, launch, cudaErrorNoKernelImageForDevice);
        } else {
            check.expect(launch == cudaSuccess, what, launch, launch, cudaSuccess);
            CHECK_CUDA(cudaMemcpy(d.data(), d_d, d.size() * sizeof(TC), cudaMemcpyDeviceToHost));
            std::snprintf(what, sizeof what, "%s layout %d", name, layout);
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n) {
                    // Odd warps (TM-row bands 1, 3, 5, ... within a block) skipped the last k step.
                    const bool odd = (m / TM % kWarps) % 2 == 1;
                    double sum = 0;
                    for (int k = 0; k < (odd ? K - TK : K); ++k) sum += av[m * K + k] * bv[k * N + n];
                    const double want = 2 * sum - cv[m * N + n];
                    const double got = double(d[d_col ? n * LDC + m : m * LDC + n]);
                    check.expect(got == want, what, m * N + n, got, want);
                }
        }
        cudaFree(d_a), cudaFree(d_b), cudaFree(d_c), cudaFree(d_d);
    }
}

