// Graph and irregular workloads on a deterministic random undirected graph in
// CSR form: level-synchronous BFS with a frontier queue (atomicCAS on the
// distance, atomicAdd on the queue tail), connected components by min-label
// propagation, pull-based PageRank, triangle counting by sorted-adjacency
// intersection, a 1024-bin histogram (global atomics and shared-memory
// privatization) and k-means (assignment with centroids in __constant__,
// update with privatized atomic sums). Every case is checked against an
// OpenMP host implementation of the same algorithm.
#include "stress.cuh"

#include <atomic>

// ---------------------------------------------------------------------------
// Graph construction (host).
// ---------------------------------------------------------------------------
struct Csr {
    int n = 0;
    std::vector<int> rowPtr, col;
};

// Undirected graph from an edge list: both directions, no self loops, no
// duplicate edges, sorted adjacency.
static Csr build_csr(int n, const std::vector<std::pair<int, int>>& edges) {
    std::vector<std::pair<int, int>> both;
    both.reserve(edges.size() * 2);
    for (auto [u, v] : edges)
        if (u != v) {
            both.push_back({u, v});
            both.push_back({v, u});
        }
    std::sort(both.begin(), both.end());
    both.erase(std::unique(both.begin(), both.end()), both.end());
    Csr g;
    g.n = n;
    g.rowPtr.assign(n + 1, 0);
    for (auto& e : both) ++g.rowPtr[e.first + 1];
    for (int i = 0; i < n; ++i) g.rowPtr[i + 1] += g.rowPtr[i];
    g.col.resize(both.size());
    for (size_t i = 0; i < both.size(); ++i) g.col[i] = both[i].second;
    return g;
}

static Csr random_graph(int n, int edgesPerNode, uint32_t seed) {
    std::vector<std::pair<int, int>> edges;
    edges.reserve(size_t(n) * edgesPerNode);
    for (int u = 0; u < n; ++u)
        for (int j = 0; j < edgesPerNode; ++j)
            edges.push_back({u, int(stress::hash32(uint32_t(u) * 64 + j + seed) % uint32_t(n))});
    return build_csr(n, edges);
}

// ---------------------------------------------------------------------------
// BFS.
// ---------------------------------------------------------------------------
__global__ void bfsLevel(const int* __restrict__ rowPtr, const int* __restrict__ col, int* dist,
                         const int* __restrict__ frontier, int frontierSize, int* nextFrontier, int* nextSize,
                         int level) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= frontierSize) return;
    int u = frontier[i];
    for (int e = rowPtr[u]; e < rowPtr[u + 1]; ++e) {
        int v = col[e];
        if (dist[v] == -1 && atomicCAS(&dist[v], -1, level + 1) == -1) nextFrontier[atomicAdd(nextSize, 1)] = v;
    }
}

__global__ void bfsInit(int* dist, int n, int src, int* frontier, int* nextSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dist[i] = i == src ? 0 : -1;
    if (i == 0) {
        frontier[0] = src;
        *nextSize = 0;
    }
}

static std::vector<int> host_bfs(const Csr& g, int src) {
    std::vector<int> dist(g.n, -1);
    std::vector<int> frontier{src}, next(g.n);
    dist[src] = 0;
    int level = 0;
    while (!frontier.empty()) {
        int nextSize = 0;
#pragma omp parallel
        {
            std::vector<int> local;
            local.reserve(1024);
#pragma omp for schedule(dynamic, 256) nowait
            for (int i = 0; i < int(frontier.size()); ++i) {
                int u = frontier[i];
                for (int e = g.rowPtr[u]; e < g.rowPtr[u + 1]; ++e) {
                    int v = g.col[e];
                    std::atomic_ref<int> d(dist[v]);
                    int expected = -1;
                    if (d.load(std::memory_order_relaxed) == -1 &&
                        d.compare_exchange_strong(expected, level + 1, std::memory_order_relaxed))
                        local.push_back(v);
                }
            }
            int at;
#pragma omp atomic capture
            {
                at = nextSize;
                nextSize += int(local.size());
            }
            std::copy(local.begin(), local.end(), next.begin() + at);
        }
        frontier.assign(next.begin(), next.begin() + nextSize);
        ++level;
    }
    return dist;
}

// ---------------------------------------------------------------------------
// Connected components: min-label propagation until no label changes.
// ---------------------------------------------------------------------------
__global__ void ccInit(int* label, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) label[i] = i;
}

__global__ void ccPropagate(const int* __restrict__ rowPtr, const int* __restrict__ col, int* label, int n,
                            int* changed) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    int mine = label[u];
    int m = mine;
    for (int e = rowPtr[u]; e < rowPtr[u + 1]; ++e) m = min(m, label[col[e]]);
    if (m < mine) {
        atomicMin(&label[u], m);
        *changed = 1;
    }
}

static std::vector<int> host_cc(const Csr& g) {
    std::vector<int> label(g.n), next(g.n);
    for (int i = 0; i < g.n; ++i) label[i] = i;
    bool changed = true;
    while (changed) {
        changed = false;
#pragma omp parallel for schedule(static, 1024) reduction(|| : changed)
        for (int u = 0; u < g.n; ++u) {
            int m = label[u];
            for (int e = g.rowPtr[u]; e < g.rowPtr[u + 1]; ++e) m = std::min(m, label[g.col[e]]);
            next[u] = m;
            changed = changed || m != label[u];
        }
        label.swap(next);
    }
    return label;
}

// ---------------------------------------------------------------------------
// PageRank, pull over in-edges (the graph is undirected: in = out).
// ---------------------------------------------------------------------------
__global__ void prInit(float* pr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pr[i] = 1.0f / n;
}

__global__ void prContrib(const float* __restrict__ pr, const int* __restrict__ rowPtr, float* contrib, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int deg = rowPtr[i + 1] - rowPtr[i];
    contrib[i] = deg > 0 ? pr[i] / deg : 0.0f;
}

__global__ void prPull(const int* __restrict__ rowPtr, const int* __restrict__ col,
                       const float* __restrict__ contrib, float* prNew, int n, float damping) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    float sum = 0.0f;
    for (int e = rowPtr[v]; e < rowPtr[v + 1]; ++e) sum += contrib[col[e]];
    prNew[v] = (1.0f - damping) / n + damping * sum;
}

// ---------------------------------------------------------------------------
// Triangle counting: for every forward edge (u, v), u < v, intersect the
// forward adjacencies of u and v (sorted).
// ---------------------------------------------------------------------------
__global__ void triangleCount(const int* __restrict__ rowPtr, const int* __restrict__ col,
                              const int* __restrict__ edgeSrc, int numEdges, unsigned long long* total) {
    __shared__ unsigned long long s_warpSum[8];
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long count = 0;
    if (e < numEdges) {
        int u = edgeSrc[e], v = col[e];
        int i = rowPtr[u], iEnd = rowPtr[u + 1];
        int j = rowPtr[v], jEnd = rowPtr[v + 1];
        while (i < iEnd && j < jEnd) {
            int a = col[i], b = col[j];
            count += a == b;
            i += a <= b;
            j += b <= a;
        }
    }
    for (int o = 16; o > 0; o >>= 1) count += __shfl_down_sync(0xffffffffu, count, o);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) s_warpSum[warp] = count;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long s = 0;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += s_warpSum[w];
        if (s) atomicAdd(total, s);
    }
}

// ---------------------------------------------------------------------------
// Histogram of 1024 bins.
// ---------------------------------------------------------------------------
constexpr int HIST_BINS = 1024;

__global__ void histGlobal(const unsigned* __restrict__ data, size_t n, unsigned* hist) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        atomicAdd(&hist[data[i] % HIST_BINS], 1u);
}

__global__ void histShared(const unsigned* __restrict__ data, size_t n, unsigned* hist) {
    __shared__ unsigned s_hist[HIST_BINS];
    for (int b = threadIdx.x; b < HIST_BINS; b += blockDim.x) s_hist[b] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        atomicAdd(&s_hist[data[i] % HIST_BINS], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < HIST_BINS; b += blockDim.x)
        if (s_hist[b]) atomicAdd(&hist[b], s_hist[b]);
}

// ---------------------------------------------------------------------------
// k-means on integer (fixed-point) 4D points, so the result is exact.
// ---------------------------------------------------------------------------
constexpr int KM_K = 32;
__constant__ int4 c_centroids[KM_K];

__device__ __host__ inline int sqdist(int4 p, int4 c) {
    int dx = p.x - c.x, dy = p.y - c.y, dz = p.z - c.z, dw = p.w - c.w;
    return dx * dx + dy * dy + dz * dz + dw * dw;
}

__global__ void kmeansAssign(const int4* __restrict__ points, int n, int* assign) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int4 p = points[i];
    int best = 0, bestD = sqdist(p, c_centroids[0]);
#pragma unroll 8
    for (int k = 1; k < KM_K; ++k) {
        int d = sqdist(p, c_centroids[k]);
        if (d < bestD) {
            bestD = d;
            best = k;
        }
    }
    assign[i] = best;
}

// Per-block sums in shared memory, then one global atomic per (cluster, dim).
__global__ void kmeansAccumulate(const int4* __restrict__ points, const int* __restrict__ assign, int n,
                                 unsigned long long* sums, unsigned* counts) {
    __shared__ unsigned long long s_sum[KM_K * 4];
    __shared__ unsigned s_cnt[KM_K];
    for (int t = threadIdx.x; t < KM_K * 4; t += blockDim.x) s_sum[t] = 0;
    for (int t = threadIdx.x; t < KM_K; t += blockDim.x) s_cnt[t] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int4 p = points[i];
        int k = assign[i];
        atomicAdd(&s_sum[k * 4 + 0], (unsigned long long)p.x);
        atomicAdd(&s_sum[k * 4 + 1], (unsigned long long)p.y);
        atomicAdd(&s_sum[k * 4 + 2], (unsigned long long)p.z);
        atomicAdd(&s_sum[k * 4 + 3], (unsigned long long)p.w);
        atomicAdd(&s_cnt[k], 1u);
    }
    __syncthreads();
    for (int t = threadIdx.x; t < KM_K * 4; t += blockDim.x)
        if (s_sum[t]) atomicAdd(&sums[t], s_sum[t]);
    for (int t = threadIdx.x; t < KM_K; t += blockDim.x)
        if (s_cnt[t]) atomicAdd(&counts[t], s_cnt[t]);
}

// New centroid = rounded mean; an empty cluster keeps its centroid.
__host__ __device__ inline int roundedMean(unsigned long long sum, unsigned count) {
    return int((sum + count / 2) / count);
}

__global__ void kmeansUpdate(const unsigned long long* sums, const unsigned* counts, int4* centroids) {
    int k = threadIdx.x;
    if (k >= KM_K) return;
    unsigned c = counts[k];
    if (c == 0) return;
    centroids[k] = make_int4(roundedMean(sums[k * 4 + 0], c), roundedMean(sums[k * 4 + 1], c),
                             roundedMean(sums[k * 4 + 2], c), roundedMean(sums[k * 4 + 3], c));
}

int main() {
    using namespace stress;

    // Random undirected graph, 2^18 nodes, average degree about 8.
    const int n = 1 << 18;
    Csr g = random_graph(n, 4, 7);
    Buffer<int> d_rowPtr(g.rowPtr), d_col(g.col);

    // BFS from node 0.
    {
        const int src = 0;
        std::vector<int> want;
        double ref = best_ms([&] { want = host_bfs(g, src); });
        Buffer<int> d_dist(n), d_qa(n), d_qb(n), d_size(1);
        double k = time_device([&] {
            bfsInit<<<(n + 255) / 256, 256>>>(d_dist, n, src, d_qa, d_size);
            int* cur = d_qa;
            int* nxt = d_qb;
            int size = 1;
            for (int level = 0; size > 0; ++level) {
                bfsLevel<<<(size + 255) / 256, 256>>>(d_rowPtr, d_col, d_dist, cur, size, nxt, d_size, level);
                STRESS_CHECK(cudaMemcpy(&size, d_size, sizeof(int), cudaMemcpyDeviceToHost));
                STRESS_CHECK(cudaMemset(d_size, 0, sizeof(int)));
                std::swap(cur, nxt);
            }
        });
        auto got = d_dist.download();
        report("bfs_256K_nodes", equal(got.data(), want.data(), n, "dist"), k, ref);
    }

    // Connected components: 8 random subgraphs (node u only links to nodes
    // of its own class u % 8) plus isolated nodes, so there are many components.
    {
        std::vector<std::pair<int, int>> edges;
        for (int u = 0; u < n; ++u) {
            if (stress::hash32(uint32_t(u) + 31337) % 64 == 0) continue;  // leaves some nodes isolated
            for (int j = 0; j < 3; ++j) {
                int v = u % 8 + 8 * int(stress::hash32(uint32_t(u) * 8 + j + 99) % uint32_t(n / 8));
                edges.push_back({u, v});
            }
        }
        Csr gs = build_csr(n, edges);
        std::vector<int> want;
        double ref = best_ms([&] { want = host_cc(gs); });
        Buffer<int> d_rp(gs.rowPtr), d_c(gs.col), d_label(n), d_changed(1);
        int iterations = 0;
        double k = time_device([&] {
            ccInit<<<(n + 255) / 256, 256>>>(d_label, n);
            int changed = 1;
            iterations = 0;
            while (changed) {
                STRESS_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
                ccPropagate<<<(n + 255) / 256, 256>>>(d_rp, d_c, d_label, n, d_changed);
                STRESS_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
                ++iterations;
            }
        });
        auto got = d_label.download();
        char note[64];
        std::snprintf(note, sizeof note, "iterations=%d", iterations);
        report("cc_label_propagation", equal(got.data(), want.data(), n, "label"), k, ref, note);
    }

    // PageRank, 20 iterations.
    {
        const int iters = 20;
        const float damping = 0.85f;
        std::vector<float> want(n), contrib(n);
        double ref = best_ms([&] {
            std::fill(want.begin(), want.end(), 1.0f / n);
            for (int it = 0; it < iters; ++it) {
#pragma omp parallel for simd
                for (int i = 0; i < n; ++i) {
                    int deg = g.rowPtr[i + 1] - g.rowPtr[i];
                    contrib[i] = deg > 0 ? want[i] / deg : 0.0f;
                }
#pragma omp parallel for schedule(static, 1024)
                for (int v = 0; v < n; ++v) {
                    float sum = 0.0f;
                    for (int e = g.rowPtr[v]; e < g.rowPtr[v + 1]; ++e) sum += contrib[g.col[e]];
                    want[v] = (1.0f - damping) / n + damping * sum;
                }
            }
        });
        Buffer<float> d_pr(n), d_contrib(n);
        double k = time_device([&] {
            prInit<<<(n + 255) / 256, 256>>>(d_pr, n);
            for (int it = 0; it < iters; ++it) {
                prContrib<<<(n + 255) / 256, 256>>>(d_pr, d_rowPtr, d_contrib, n);
                prPull<<<(n + 255) / 256, 256>>>(d_rowPtr, d_col, d_contrib, d_pr, n, damping);
            }
        });
        auto got = d_pr.download();
        report("pagerank_20_iters", close(got.data(), want.data(), n, 1e-4, 1e-9, "pr"), k, ref);
    }

    // Triangle counting on a graph with local structure (neighbours within a
    // window of ids, plus random long edges), so there are many triangles.
    {
        const int tn = 1 << 18;
        std::vector<std::pair<int, int>> edges;
        for (int u = 0; u < tn; ++u)
            for (int j = 0; j < 12; ++j) {
                uint32_t h = stress::hash32(uint32_t(u) * 16 + j + 555);
                int v = j < 10 ? (u + 1 + int(h % 64)) % tn : int(h % tn);
                edges.push_back({u, v});
            }
        Csr full = build_csr(tn, edges);
        // Forward graph: keep v > u, adjacency stays sorted.
        std::vector<int> fRow(tn + 1, 0), fCol, edgeSrc;
        for (int u = 0; u < tn; ++u) {
            for (int e = full.rowPtr[u]; e < full.rowPtr[u + 1]; ++e)
                if (full.col[e] > u) {
                    fCol.push_back(full.col[e]);
                    edgeSrc.push_back(u);
                }
            fRow[u + 1] = int(fCol.size());
        }
        const int fe = int(fCol.size());
        unsigned long long want = 0;
        double ref = best_ms([&] {
            unsigned long long t = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : t)
            for (int u = 0; u < tn; ++u)
                for (int e = fRow[u]; e < fRow[u + 1]; ++e) {
                    int v = fCol[e];
                    int i = fRow[u], iEnd = fRow[u + 1], j = fRow[v], jEnd = fRow[v + 1];
                    while (i < iEnd && j < jEnd) {
                        int a = fCol[i], b = fCol[j];
                        t += a == b;
                        i += a <= b;
                        j += b <= a;
                    }
                }
            want = t;
        });
        Buffer<int> d_fr(fRow), d_fc(fCol), d_src(edgeSrc);
        Buffer<unsigned long long> d_total(1);
        double k = time_device([&] {
            STRESS_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
            triangleCount<<<(fe + 255) / 256, 256>>>(d_fr, d_fc, d_src, fe, d_total);
        });
        auto got = d_total.download();
        char note[64];
        std::snprintf(note, sizeof note, "triangles=%llu", want);
        report("triangle_count", got[0] == want && want > 0, k, ref, note);
    }

    // Histograms of 2^24 values into 1024 bins, skewed: half the values fall
    // in 16 hot bins.
    {
        const size_t hn = size_t(1) << 24;
        auto data = random_u32(hn, 77);
        for (size_t i = 0; i < hn; ++i)
            if (data[i] & 0x80000000u) data[i] = (data[i] >> 8) & 15;
        std::vector<unsigned> want(HIST_BINS);
        double ref = best_ms([&] {
            unsigned h[HIST_BINS] = {};
#pragma omp parallel for reduction(+ : h[:HIST_BINS])
            for (size_t i = 0; i < hn; ++i) ++h[data[i] % HIST_BINS];
            std::copy(h, h + HIST_BINS, want.begin());
        });
        Buffer<unsigned> d_data(data), d_hist(HIST_BINS);
        int sms = 0;
        STRESS_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        double k = time_device([&] {
            STRESS_CHECK(cudaMemset(d_hist, 0, HIST_BINS * sizeof(unsigned)));
            histGlobal<<<(unsigned)((hn + 255) / 256), 256>>>(d_data, hn, d_hist);
        });
        auto got = d_hist.download();
        report("histogram_global_atomics", equal(got.data(), want.data(), HIST_BINS, "hist"), k, ref);

        k = time_device([&] {
            STRESS_CHECK(cudaMemset(d_hist, 0, HIST_BINS * sizeof(unsigned)));
            histShared<<<sms * 8, 256>>>(d_data, hn, d_hist);
        });
        got = d_hist.download();
        report("histogram_shared_private", equal(got.data(), want.data(), HIST_BINS, "hist"), k, ref);
    }

    // k-means: 2^20 4D points around 32 hidden centres, 10 Lloyd iterations.
    {
        const int pn = 1 << 20, iters = 10;
        std::vector<int4> points(pn), init(KM_K);
        for (int i = 0; i < pn; ++i) {
            uint32_t c = stress::hash32(i * 3 + 1) % KM_K;
            auto coord = [&](int d) {
                int centre = int(stress::hash32(c * 4 + d + 1000) % 14000) + 1000;
                int jitter = int(stress::hash32(uint32_t(i) * 4 + d + 9) % 2001) - 1000;
                return centre + jitter;
            };
            points[i] = make_int4(coord(0), coord(1), coord(2), coord(3));
        }
        for (int k = 0; k < KM_K; ++k) init[k] = points[stress::hash32(k + 4242) % pn];

        std::vector<int4> want;
        std::vector<int> wantAssign(pn);
        double ref = best_ms([&] {
            std::vector<int4> cent = init;
            for (int it = 0; it < iters; ++it) {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < pn; ++i) {
                    int best = 0, bestD = sqdist(points[i], cent[0]);
                    for (int k = 1; k < KM_K; ++k) {
                        int d = sqdist(points[i], cent[k]);
                        if (d < bestD) {
                            bestD = d;
                            best = k;
                        }
                    }
                    wantAssign[i] = best;
                }
                std::vector<unsigned long long> sums(KM_K * 4, 0);
                std::vector<unsigned> counts(KM_K, 0);
#pragma omp parallel
                {
                    unsigned long long ls[KM_K * 4] = {};
                    unsigned lc[KM_K] = {};
#pragma omp for schedule(static) nowait
                    for (int i = 0; i < pn; ++i) {
                        int k = wantAssign[i];
                        ls[k * 4 + 0] += points[i].x;
                        ls[k * 4 + 1] += points[i].y;
                        ls[k * 4 + 2] += points[i].z;
                        ls[k * 4 + 3] += points[i].w;
                        ++lc[k];
                    }
#pragma omp critical
                    {
                        for (int t = 0; t < KM_K * 4; ++t) sums[t] += ls[t];
                        for (int t = 0; t < KM_K; ++t) counts[t] += lc[t];
                    }
                }
                for (int k = 0; k < KM_K; ++k)
                    if (counts[k])
                        cent[k] = make_int4(roundedMean(sums[k * 4 + 0], counts[k]), roundedMean(sums[k * 4 + 1], counts[k]),
                                            roundedMean(sums[k * 4 + 2], counts[k]), roundedMean(sums[k * 4 + 3], counts[k]));
            }
            want = cent;
        });

        Buffer<int4> d_points(points), d_cent(KM_K);
        Buffer<int> d_assign(pn);
        Buffer<unsigned long long> d_sums(KM_K * 4);
        Buffer<unsigned> d_counts(KM_K);
        int sms = 0;
        STRESS_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        double k = time_device([&] {
            STRESS_CHECK(cudaMemcpy(d_cent, init.data(), KM_K * sizeof(int4), cudaMemcpyHostToDevice));
            for (int it = 0; it < iters; ++it) {
                STRESS_CHECK(cudaMemcpyToSymbol(c_centroids, d_cent, KM_K * sizeof(int4), 0, cudaMemcpyDeviceToDevice));
                kmeansAssign<<<(pn + 255) / 256, 256>>>(d_points, pn, d_assign);
                STRESS_CHECK(cudaMemset(d_sums, 0, KM_K * 4 * sizeof(unsigned long long)));
                STRESS_CHECK(cudaMemset(d_counts, 0, KM_K * sizeof(unsigned)));
                kmeansAccumulate<<<sms * 4, 256>>>(d_points, d_assign, pn, d_sums, d_counts);
                kmeansUpdate<<<1, KM_K>>>(d_sums, d_counts, d_cent);
            }
        });
        auto gotCent = d_cent.download();
        auto gotAssign = d_assign.download();
        bool ok = equal(gotAssign.data(), wantAssign.data(), pn, "assign");
        for (int c = 0; ok && c < KM_K; ++c) {
            int4 a = gotCent[c], b = want[c];
            if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) {
                std::printf("  mismatch centroid %d: got (%d %d %d %d) want (%d %d %d %d)\n", c, a.x, a.y, a.z, a.w,
                            b.x, b.y, b.z, b.w);
                ok = false;
            }
        }
        report("kmeans_1M_points_k32", ok, k, ref);
    }

    return finish();
}
