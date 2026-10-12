// Dynamic programming with wavefront dependencies, driven by many small
// launches from a host loop: Smith-Waterman with one launch per anti-diagonal
// of cells, longest common subsequence with one launch per anti-diagonal of
// tiles (each block solves its tile in shared memory with an inner
// anti-diagonal loop and __syncthreads), and Floyd-Warshall on 1024 nodes,
// both the plain k-loop of 1024 launches and the three-phase shared-memory
// blocked version. All results are integers and checked exactly.
#include "stress.cuh"

// Host reference for the 2D recurrences: tiles of TB x TB cells, the tiles of
// one tile anti-diagonal in parallel, row-major inside a tile.
template <class Cell>
static void host_wavefront(int N, int M, int TB, Cell cell) {
    const int nti = (N + TB - 1) / TB, ntj = (M + TB - 1) / TB;
    for (int d = 0; d <= nti + ntj - 2; ++d) {
        const int lo = std::max(0, d - ntj + 1), hi = std::min(nti - 1, d);
#pragma omp parallel for schedule(dynamic, 1)
        for (int ti = lo; ti <= hi; ++ti) {
            const int tj = d - ti;
            const int i1 = std::min(N, (ti + 1) * TB), j1 = std::min(M, (tj + 1) * TB);
            for (int i = ti * TB + 1; i <= i1; ++i)
                for (int j = tj * TB + 1; j <= j1; ++j) cell(i, j);
        }
    }
}

static std::vector<char> random_dna(int n, uint32_t seed) {
    std::vector<char> s(n);
    for (int i = 0; i < n; ++i) s[i] = "ACGT"[stress::hash32(uint32_t(i) * 2654435761u + seed) & 3];
    return s;
}

// ---------------------------------------------------------------------------
// Smith-Waterman, linear gap, one launch per anti-diagonal
// ---------------------------------------------------------------------------

constexpr int kMatch = 3, kMismatch = -3, kGap = 2;

// H is (N+1) x (M+1), row 0 and column 0 are zero. Diagonal d holds the cells
// with i + j == d; thread t takes i = i_min + t.
__global__ void sw_diagonal(const char* __restrict__ a, const char* __restrict__ b, int* __restrict__ H, int N, int M,
                            int d, int i_min, int count) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int i = i_min + t, j = d - i;
    const int w = M + 1;
    int diag = H[(i - 1) * w + (j - 1)] + (a[i - 1] == b[j - 1] ? kMatch : kMismatch);
    int up = H[(i - 1) * w + j] - kGap;
    int left = H[i * w + (j - 1)] - kGap;
    H[i * w + j] = max(0, max(diag, max(up, left)));
}

static void sw_case() {
    using namespace stress;
    const int N = 1024, M = 1024, w = M + 1;
    auto a = random_dna(N, 11), b = random_dna(M, 23);
    std::vector<int> want(size_t(N + 1) * w, 0);
    double ref = best_ms([&] {
        host_wavefront(N, M, 64, [&](int i, int j) {
            int diag = want[(i - 1) * w + j - 1] + (a[i - 1] == b[j - 1] ? kMatch : kMismatch);
            int up = want[(i - 1) * w + j] - kGap, left = want[i * w + j - 1] - kGap;
            want[i * w + j] = std::max(0, std::max(diag, std::max(up, left)));
        });
    });
    Buffer<char> da(a), db(b);
    Buffer<int> dH(want.size());
    dH.zero();  // the borders; every launch sequence rewrites the whole interior
    double k = time_device([&] {
        for (int d = 2; d <= N + M; ++d) {
            int i_min = std::max(1, d - M), i_max = std::min(N, d - 1);
            int count = i_max - i_min + 1;
            sw_diagonal<<<(count + 127) / 128, 128>>>(da, db, dH, N, M, d, i_min, count);
        }
    });
    auto got = dH.download();
    int best = *std::max_element(want.begin(), want.end());
    char note[64];
    std::snprintf(note, sizeof note, "launches=%d best=%d", N + M - 1, best);
    report("smith_waterman_diag_launches", equal(got.data(), want.data(), want.size(), "H"), k, ref, note);
}

// ---------------------------------------------------------------------------
// LCS, tiled wavefront
// ---------------------------------------------------------------------------

// One block per tile of the tile anti-diagonal d; T threads, thread t owns
// row t of the tile. The tile plus its top row and left column of halo live
// in shared memory.
template <int T>
__global__ void lcs_tile_diagonal(const char* __restrict__ a, const char* __restrict__ b, int* __restrict__ L, int M,
                                  int d, int ti_min) {
    __shared__ int s[T + 1][T + 1];
    __shared__ char sa[T], sb[T];
    const int ti = ti_min + blockIdx.x, tj = d - ti;
    const int i0 = ti * T, j0 = tj * T, w = M + 1, t = threadIdx.x;
    s[0][t + 1] = L[i0 * w + j0 + 1 + t];
    s[t + 1][0] = L[(i0 + 1 + t) * w + j0];
    if (t == 0) s[0][0] = L[i0 * w + j0];
    sa[t] = a[i0 + t];
    sb[t] = b[j0 + t];
    __syncthreads();
    for (int k = 0; k < 2 * T - 1; ++k) {
        int j = k - t;
        if (j >= 0 && j < T) {
            int r = t + 1, c = j + 1;
            s[r][c] = sa[t] == sb[j] ? s[r - 1][c - 1] + 1 : max(s[r - 1][c], s[r][c - 1]);
        }
        __syncthreads();
    }
    for (int r = 1; r <= T; ++r) L[(i0 + r) * w + j0 + 1 + t] = s[r][t + 1];
}

static void lcs_case() {
    using namespace stress;
    constexpr int T = 32;
    const int N = 4096, M = 4096, w = M + 1;
    auto a = random_dna(N, 5), b = random_dna(M, 9);
    std::vector<int> want(size_t(N + 1) * w, 0);
    double ref = best_ms([&] {
        host_wavefront(N, M, 64, [&](int i, int j) {
            want[size_t(i) * w + j] = a[i - 1] == b[j - 1]
                                          ? want[size_t(i - 1) * w + j - 1] + 1
                                          : std::max(want[size_t(i - 1) * w + j], want[size_t(i) * w + j - 1]);
        });
    });
    Buffer<char> da(a), db(b);
    Buffer<int> dL(want.size());
    dL.zero();
    const int nti = N / T, ntj = M / T;
    double k = time_device([&] {
        for (int d = 0; d <= nti + ntj - 2; ++d) {
            int lo = std::max(0, d - ntj + 1), hi = std::min(nti - 1, d);
            lcs_tile_diagonal<T><<<hi - lo + 1, T>>>(da, db, dL, M, d, lo);
        }
    });
    auto got = dL.download();
    char note[64];
    std::snprintf(note, sizeof note, "launches=%d lcs=%d", nti + ntj - 1, want.back());
    report("lcs_tiled_wavefront", equal(got.data(), want.data(), want.size(), "L"), k, ref, note);
}

// ---------------------------------------------------------------------------
// Floyd-Warshall
// ---------------------------------------------------------------------------

constexpr int INF = 1000000000;

__host__ __device__ __forceinline__ int add_inf(int x, int y) { return x >= INF || y >= INF ? INF : x + y; }

__global__ void fw_step(int* __restrict__ D, int n, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    int dik = D[i * n + k], dkj = D[k * n + j];
    if (dik < INF && dkj < INF) {
        int via = dik + dkj;
        if (via < D[i * n + j]) D[i * n + j] = via;
    }
}

constexpr int FB = 32;

// Phase 1: the pivot tile (kb, kb) closes over its own k.
__global__ void fw_phase1(int* D, int n, int kb) {
    __shared__ int t[FB][FB];
    const int tx = threadIdx.x, ty = threadIdx.y, base = kb * FB;
    t[ty][tx] = D[(base + ty) * n + base + tx];
    __syncthreads();
    for (int k = 0; k < FB; ++k) {
        int via = add_inf(t[ty][k], t[k][tx]);
        __syncthreads();
        if (via < t[ty][tx]) t[ty][tx] = via;
        __syncthreads();
    }
    D[(base + ty) * n + base + tx] = t[ty][tx];
}

// Phase 2: the tiles of the pivot row (blockIdx.y == 0) and pivot column
// (blockIdx.y == 1), each against the finished pivot tile.
__global__ void fw_phase2(int* D, int n, int kb) {
    if (blockIdx.x == kb) return;
    __shared__ int piv[FB][FB], own[FB][FB];
    const int tx = threadIdx.x, ty = threadIdx.y, pb = kb * FB;
    const bool row = blockIdx.y == 0;
    const int r0 = row ? pb : blockIdx.x * FB, c0 = row ? blockIdx.x * FB : pb;
    piv[ty][tx] = D[(pb + ty) * n + pb + tx];
    own[ty][tx] = D[(r0 + ty) * n + c0 + tx];
    __syncthreads();
    for (int k = 0; k < FB; ++k) {
        int via = row ? add_inf(piv[ty][k], own[k][tx]) : add_inf(own[ty][k], piv[k][tx]);
        __syncthreads();
        if (via < own[ty][tx]) own[ty][tx] = via;
        __syncthreads();
    }
    D[(r0 + ty) * n + c0 + tx] = own[ty][tx];
}

// Phase 3: every other tile, from its pivot-column and pivot-row tiles.
__global__ void fw_phase3(int* D, int n, int kb) {
    if (blockIdx.x == kb || blockIdx.y == kb) return;
    __shared__ int colt[FB][FB], rowt[FB][FB];
    const int tx = threadIdx.x, ty = threadIdx.y, pb = kb * FB;
    const int r0 = blockIdx.y * FB, c0 = blockIdx.x * FB;
    colt[ty][tx] = D[(r0 + ty) * n + pb + tx];
    rowt[ty][tx] = D[(pb + ty) * n + c0 + tx];
    __syncthreads();
    int d = D[(r0 + ty) * n + c0 + tx];
#pragma unroll 8
    for (int k = 0; k < FB; ++k) d = min(d, add_inf(colt[ty][k], rowt[k][tx]));
    D[(r0 + ty) * n + c0 + tx] = d;
}

static void fw_cases() {
    using namespace stress;
    const int n = 1024;
    std::vector<int> init(size_t(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            uint32_t h = hash32(uint32_t(i * n + j) * 2654435761u + 77);
            int v = (h & 31) == 0 ? 1 + int((h >> 8) % 100) : INF;
            if (j >= n - 16) v = INF;  // no edges into the last 16 nodes: they stay unreachable
            init[size_t(i) * n + j] = i == j ? 0 : v;
        }
    std::vector<int> want;
    double ref = best_ms([&] {
        want = init;
        int* D = want.data();
        for (int k = 0; k < n; ++k) {
            const int* rk = D + size_t(k) * n;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                int* ri = D + size_t(i) * n;
                const int dik = ri[k];
                if (dik >= INF) continue;
#pragma omp simd
                for (int j = 0; j < n; ++j) {
                    int via = rk[j] >= INF ? INF : dik + rk[j];
                    ri[j] = std::min(ri[j], via);
                }
            }
        }
    });
    long long unreachable = 0;
    for (int v : want) unreachable += v >= INF;
    char note[64];
    std::snprintf(note, sizeof note, "unreachable=%lld", unreachable);

    Buffer<int> d_init(init), dD(init.size());
    const size_t bytes = init.size() * sizeof(int);
    double k1 = time_device([&] {
        STRESS_CHECK(cudaMemcpy(dD.p, d_init.p, bytes, cudaMemcpyDeviceToDevice));
        dim3 blk(32, 8), grid(n / 32, n / 8);
        for (int k = 0; k < n; ++k) fw_step<<<grid, blk>>>(dD, n, k);
    });
    auto got = dD.download();
    report("floyd_warshall_k_launches", equal(got.data(), want.data(), want.size(), "D"), k1, ref, note);

    dD.zero();
    double k2 = time_device([&] {
        STRESS_CHECK(cudaMemcpy(dD.p, d_init.p, bytes, cudaMemcpyDeviceToDevice));
        const int nb = n / FB;
        dim3 blk(FB, FB);
        for (int kb = 0; kb < nb; ++kb) {
            fw_phase1<<<1, blk>>>(dD, n, kb);
            fw_phase2<<<dim3(nb, 2), blk>>>(dD, n, kb);
            fw_phase3<<<dim3(nb, nb), blk>>>(dD, n, kb);
        }
    });
    got = dD.download();
    report("floyd_warshall_blocked_shared", equal(got.data(), want.data(), want.size(), "D"), k2, ref, note);
}

int main() {
    sw_case();
    lcs_case();
    fw_cases();
    return stress::finish();
}
