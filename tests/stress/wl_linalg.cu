// Dense and sparse linear algebra: matmul (naive, 16x16 shared tiles, 4x4
// register blocking over 64x64 shared tiles), row-major gemv with one warp
// per row and a shuffle reduction, batched 8x8 matmuls, and CSR spmv with one
// thread per row (scalar) and one warp per row (vector).
#include "stress.cuh"

// ------------------------------------------------------------------ matmul
__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float s = 0.0f;
        for (int k = 0; k < n; ++k) s += A[row * n + k] * B[k * n + col];
        C[row * n + col] = s;
    }
}

#define TILE 16

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float s = 0.0f;
    for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
        int ka = t * TILE + tx, kb = t * TILE + ty;
        As[ty][tx] = row < n && ka < n ? A[row * n + ka] : 0.0f;
        Bs[ty][tx] = kb < n && col < n ? B[kb * n + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) s += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = s;
}

// Each block computes a BM x BN tile of C with (BM/TM) x (BN/TN) threads, each
// thread a TM x TN sub-block kept in registers. A is stored transposed in
// shared memory so the inner loop reads both operands along a row. n must be a
// multiple of BM, BN and BK.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_regblock(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.x;
    int tcol = tid % (BN / TN), trow = tid / (BN / TN);
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < n; k0 += BK) {
        for (int i = tid; i < BM * BK; i += THREADS) {
            int r = i / BK, c = i % BK;
            As[c][r] = A[(row0 + r) * n + k0 + c];
        }
        for (int i = tid; i < BK * BN; i += THREADS) {
            int r = i / BN, c = i % BN;
            Bs[r][c] = B[(k0 + r) * n + col0 + c];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][trow * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tcol * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        float4* out = reinterpret_cast<float4*>(&C[(row0 + trow * TM + i) * n + col0 + tcol * TN]);
#pragma unroll
        for (int j = 0; j < TN / 4; ++j) out[j] = make_float4(acc[i][4 * j], acc[i][4 * j + 1], acc[i][4 * j + 2], acc[i][4 * j + 3]);
    }
}

// -------------------------------------------------------------------- gemv
__inline__ __device__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// y = A x, A row-major m x n; one warp per row.
__global__ void gemv_warp(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int m, int n) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    if (warp >= m) return;
    const float* row = A + size_t(warp) * n;
    float s = 0.0f;
    if ((n & 3) == 0) {
        const float4* r4 = reinterpret_cast<const float4*>(row);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int j = lane; j < n / 4; j += warpSize) {
            float4 a = r4[j], b = x4[j];
            s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int j = lane; j < n; j += warpSize) s += row[j] * x[j];
    }
    s = warp_reduce_sum(s);
    if (lane == 0) y[warp] = s;
}

// ------------------------------------------------------- batched 8x8 matmul
#define BS 8
#define MATS_PER_BLOCK 4

// One thread per output element; each block multiplies MATS_PER_BLOCK pairs,
// staged in shared memory.
__global__ void batched_matmul8(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int batch) {
    __shared__ float As[MATS_PER_BLOCK][BS][BS];
    __shared__ float Bs[MATS_PER_BLOCK][BS][BS];
    int m = threadIdx.z;
    int mat = blockIdx.x * MATS_PER_BLOCK + m;
    int r = threadIdx.y, c = threadIdx.x;
    bool valid = mat < batch;
    size_t base = size_t(mat) * BS * BS;
    if (valid) {
        As[m][r][c] = A[base + r * BS + c];
        Bs[m][r][c] = B[base + r * BS + c];
    }
    __syncthreads();
    if (valid) {
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < BS; ++k) s += As[m][r][k] * Bs[m][k][c];
        C[base + r * BS + c] = s;
    }
}

// -------------------------------------------------------------------- spmv
__global__ void spmv_csr_scalar(const int* __restrict__ rowptr, const int* __restrict__ cols,
                                const float* __restrict__ vals, const float* __restrict__ x, float* __restrict__ y,
                                int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        float s = 0.0f;
        for (int j = rowptr[row]; j < rowptr[row + 1]; ++j) s += vals[j] * x[cols[j]];
        y[row] = s;
    }
}

__global__ void spmv_csr_vector(const int* __restrict__ rowptr, const int* __restrict__ cols,
                                const float* __restrict__ vals, const float* __restrict__ x, float* __restrict__ y,
                                int m) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x & (warpSize - 1);
    if (row >= m) return;
    int begin = rowptr[row], end = rowptr[row + 1];
    float s = 0.0f;
    for (int j = begin + lane; j < end; j += warpSize) s += vals[j] * x[cols[j]];
    s = warp_reduce_sum(s);
    if (lane == 0) y[row] = s;
}

// ===================================================================== cases
using namespace stress;

static void cases_matmul() {
    const int n = 1280;
    const size_t nn = size_t(n) * n;
    auto a = random_floats(nn, 101), b = random_floats(nn, 102);
    std::vector<float> want(nn);
    // Blocked i-k-j: 8 rows of C at a time, 256 x 256 panels of B.
    double ref = best_ms([&] {
        constexpr int RB = 8, JB = 256, KB = 256;  // n is a multiple of all three
#pragma omp parallel for schedule(static)
        for (int ib = 0; ib < n; ib += RB) {
            for (int i = ib; i < ib + RB; ++i) std::fill(&want[size_t(i) * n], &want[size_t(i) * n] + n, 0.0f);
            for (int jb = 0; jb < n; jb += JB)
                for (int kb = 0; kb < n; kb += KB)
                    for (int i = ib; i < ib + RB; ++i) {
                        float* c = &want[size_t(i) * n];
                        for (int k = kb; k < kb + KB; ++k) {
                            float av = a[size_t(i) * n + k];
                            const float* br = &b[size_t(k) * n];
#pragma omp simd
                            for (int j = jb; j < jb + JB; ++j) c[j] += av * br[j];
                        }
                    }
        }
    });
    Buffer<float> da(a), db(b), dc(nn);
    const double rel = 1e-4, abs_tol = 1e-3;

    dc.zero();
    dim3 block(16, 16), grid((n + 15) / 16, (n + 15) / 16);
    double k = time_device([&] { matmul_naive<<<grid, block>>>(da, db, dc, n); });
    report("matmul_naive", close(dc.download().data(), want.data(), nn, rel, abs_tol), k, ref);

    dc.zero();
    k = time_device([&] { matmul_tiled<<<grid, block>>>(da, db, dc, n); });
    report("matmul_tiled16", close(dc.download().data(), want.data(), nn, rel, abs_tol), k, ref);

    dc.zero();
    constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    dim3 rgrid(n / BN, n / BM);
    k = time_device([&] { matmul_regblock<BM, BN, BK, TM, TN><<<rgrid, (BM / TM) * (BN / TN)>>>(da, db, dc, n); });
    report("matmul_regblock4x4", close(dc.download().data(), want.data(), nn, rel, abs_tol), k, ref);
}

static void case_gemv() {
    const int m = 16384, n = 4096;
    auto a = random_floats(size_t(m) * n, 201), x = random_floats(n, 202);
    std::vector<float> want(m);
    double ref = best_ms([&] {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < m; ++i) {
            const float* r = &a[size_t(i) * n];
            float s = 0.0f;
#pragma omp simd reduction(+ : s)
            for (int j = 0; j < n; ++j) s += r[j] * x[j];
            want[i] = s;
        }
    });
    Buffer<float> da(a), dx(x), dy(m);
    const int threads = 256, rows_per_block = threads / 32;
    double k = time_device([&] { gemv_warp<<<(m + rows_per_block - 1) / rows_per_block, threads>>>(da, dx, dy, m, n); });
    report("gemv_warp_shfl", close(dy.download().data(), want.data(), m, 1e-4, 1e-4), k, ref);
}

static void case_batched() {
    const int batch = 1 << 19;
    const size_t total = size_t(batch) * BS * BS;
    auto a = random_floats(total, 301), b = random_floats(total, 302);
    std::vector<float> want(total);
    double ref = best_ms([&] {
#pragma omp parallel for schedule(static)
        for (int mi = 0; mi < batch; ++mi) {
            const float* A = &a[size_t(mi) * BS * BS];
            const float* B = &b[size_t(mi) * BS * BS];
            float* C = &want[size_t(mi) * BS * BS];
            for (int r = 0; r < BS; ++r) {
                float acc[BS] = {};
                for (int k = 0; k < BS; ++k) {
                    float av = A[r * BS + k];
#pragma omp simd
                    for (int c = 0; c < BS; ++c) acc[c] += av * B[k * BS + c];
                }
                for (int c = 0; c < BS; ++c) C[r * BS + c] = acc[c];
            }
        }
    });
    Buffer<float> da(a), db(b), dc(total);
    dim3 block(BS, BS, MATS_PER_BLOCK);
    double k = time_device([&] { batched_matmul8<<<(batch + MATS_PER_BLOCK - 1) / MATS_PER_BLOCK, block>>>(da, db, dc, batch); });
    report("batched_matmul8x8", close(dc.download().data(), want.data(), total, 1e-5, 1e-5), k, ref);
}

static void cases_spmv() {
    // Random sparse matrix: most rows have 0..47 nonzeros, every 64th row is
    // long (256..767), so the row lengths vary widely.
    const int m = 1 << 20, ncols = 1 << 20;
    std::vector<int> rowptr(m + 1);
    rowptr[0] = 0;
    for (int i = 0; i < m; ++i) {
        uint32_t h = hash32(uint32_t(i) + 0x9e3779b9u);
        int len = i % 64 == 0 ? 256 + int(h % 512) : int(h % 48);
        rowptr[i + 1] = rowptr[i] + len;
    }
    const int nnz = rowptr[m];
    std::vector<int> cols(nnz);
    auto vals = random_floats(nnz, 401);
    auto x = random_floats(ncols, 402);
#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        // Columns sorted within a row, clustered near the diagonal with some
        // far entries, as in real matrices.
        int b = rowptr[i], len = rowptr[i + 1] - b;
        int c = std::max(0, i - len);
        for (int j = 0; j < len; ++j) {
            uint32_t h = hash32(uint32_t(b + j) * 747796405u + 7);
            c += 1 + int(h % 4 == 0 ? h % 4096 : h % 4);
            cols[b + j] = c % ncols;
        }
    }
    std::vector<float> want(m);
    double ref = best_ms([&] {
#pragma omp parallel for schedule(dynamic, 1024)
        for (int i = 0; i < m; ++i) {
            float s = 0.0f;
            int e = rowptr[i + 1];
#pragma omp simd reduction(+ : s)
            for (int j = rowptr[i]; j < e; ++j) s += vals[j] * x[cols[j]];
            want[i] = s;
        }
    });
    Buffer<int> drow(rowptr), dcol(cols);
    Buffer<float> dval(vals), dx(x), dy(m);

    dy.zero();
    double k = time_device([&] { spmv_csr_scalar<<<(m + 255) / 256, 256>>>(drow, dcol, dval, dx, dy, m); });
    report("spmv_csr_scalar", close(dy.download().data(), want.data(), m, 1e-4, 1e-4), k, ref);

    dy.zero();
    const int threads = 256, rows_per_block = threads / 32;
    k = time_device([&] {
        spmv_csr_vector<<<(m + rows_per_block - 1) / rows_per_block, threads>>>(drow, dcol, dval, dx, dy, m);
    });
    report("spmv_csr_vector", close(dy.download().data(), want.data(), m, 1e-4, 1e-4), k, ref);
}

int main() {
    cases_matmul();
    case_gemv();
    case_batched();
    cases_spmv();
    return finish();
}
